from __future__ import annotations

"""ADARE core algorithm with adaptive operators, archive guidance, and runtime optimizations."""


import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Sequence

import numpy as np
from deap import tools  # type: ignore

from algorithms.base_algorithm import BaseAlgorithm, evaluate_schedule
from .operators import (
    AdaptiveOperatorController,
    adaptive_mutation,
    build_task_node_rankings,
    dominates,
    greedy_local_repair,
    scalar_fitness,
)
from .utils import genotype_key, population_diversity


_WORKER_TASKS: list[dict[str, Any]] | None = None
_WORKER_NODES: list[dict[str, Any]] | None = None
_WORKER_ORDER: list[int] | None = None
_WORKER_OBJECTIVES: list[str] | None = None


# Initialize process-local immutable data for parallel evaluation workers.
def _worker_init(
    tasks: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    order: list[int],
    objective_names: list[str],
) -> None:
    global _WORKER_TASKS, _WORKER_NODES, _WORKER_ORDER, _WORKER_OBJECTIVES
    _WORKER_TASKS = tasks
    _WORKER_NODES = nodes
    _WORKER_ORDER = order
    _WORKER_OBJECTIVES = objective_names


def _worker_evaluate(individual: list[int]) -> tuple[float, ...]:
    """Evaluate one individual in a worker process using preloaded context."""
    if _WORKER_TASKS is None or _WORKER_NODES is None or _WORKER_ORDER is None or _WORKER_OBJECTIVES is None:
        raise RuntimeError("ADARE worker context non initialise.")
    return evaluate_schedule(
        individual=individual,
        tasks=_WORKER_TASKS,
        nodes=_WORKER_NODES,
        topological_order=_WORKER_ORDER,
        objective_names=_WORKER_OBJECTIVES,
    )


class AdareAlgorithm(BaseAlgorithm):
    # Adaptive multi-objective scheduler extending NSGA-III with ADARE-specific mechanisms.

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize ADARE hyperparameters, caches, and precomputed heuristic keys."""
        super().__init__(*args, **kwargs)
        self.controller = AdaptiveOperatorController(
            exploration_start=float(self.algorithm_config.get("exploration_start", 0.25)),
            exploration_end=float(self.algorithm_config.get("exploration_end", 0.05)),
            alpha=float(self.algorithm_config.get("q_learning_alpha", 0.2)),
        )
        self.heuristic_mutation_rate = float(self.algorithm_config.get("heuristic_mutation_rate", 0.7))
        self.max_mutation_budget = max(1, int(self.algorithm_config.get("max_mutation_budget", 48)))
        self.use_adaptive_mutation = bool(self.algorithm_config.get("use_adaptive_mutation", True))
        self.archive_size = int(self.algorithm_config.get("archive_size", self.population_size))
        self.enable_archive = bool(self.algorithm_config.get("enable_archive", True))
        self.archive_injection_rate = float(self.algorithm_config.get("archive_injection_rate", 0.1))
        self.local_search_probability = float(self.algorithm_config.get("local_search_probability", 0.1))
        self.local_search_tasks = int(self.algorithm_config.get("local_search_tasks", 2))
        self.energy_guided_mutation_rate = float(
            self.algorithm_config.get("energy_guided_mutation_rate", 0.0)
        )
        self.energy_exploration_rate = float(
            self.algorithm_config.get("energy_exploration_rate", 0.0)
        )
        self.energy_micro_mutation_rate = float(
            self.algorithm_config.get("energy_micro_mutation_rate", 0.0)
        )
        self.append_energy_anchor_final = bool(
            self.algorithm_config.get("append_energy_anchor_final", False)
        )
        self.append_energy_balanced_final = bool(
            self.algorithm_config.get("append_energy_balanced_final", False)
        )
        self.seed_archive_with_energy_balanced = bool(
            self.algorithm_config.get("seed_archive_with_energy_balanced", False)
        )
        self.append_speed_latency_final = bool(
            self.algorithm_config.get("append_speed_latency_final", False)
        )
        self.refine_speed_latency_final = bool(
            self.algorithm_config.get("refine_speed_latency_final", True)
        )
        self.refine_speed_latency_iterations = max(
            0, int(self.algorithm_config.get("refine_speed_latency_iterations", 3))
        )
        self.archive_update_period = max(1, int(self.algorithm_config.get("archive_update_period", 2)))
        self.archive_prefilter_factor = max(1.0, float(self.algorithm_config.get("archive_prefilter_factor", 3.0)))
        self.offspring_ratio_start = float(self.algorithm_config.get("offspring_ratio_start", 0.95))
        self.offspring_ratio_end = float(self.algorithm_config.get("offspring_ratio_end", 0.55))
        self.diversity_refresh_period = max(1, int(self.algorithm_config.get("diversity_refresh_period", 2)))
        self.stagnation_patience = max(1, int(self.algorithm_config.get("stagnation_patience", 8)))
        self.min_crossover_probability = float(self.algorithm_config.get("min_crossover_probability", 0.35))
        self.min_mutation_probability = float(self.algorithm_config.get("min_mutation_probability", 0.06))
        self.elite_preservation_period = max(
            1, int(self.algorithm_config.get("elite_preservation_period", 1))
        )
        self.rescue_generations = max(0, int(self.algorithm_config.get("rescue_generations", 0)))
        self.operator_schedule = str(self.algorithm_config.get("operator_schedule", "legacy"))
        self.runtime_drop_strength = float(self.algorithm_config.get("runtime_drop_strength", 0.0))
        self.parallel_workers = max(1, int(self.algorithm_config.get("parallel_workers", 1)))
        self.parallel_threshold = max(1, int(self.algorithm_config.get("parallel_threshold", 24)))
        self.use_fitness_cache = bool(self.algorithm_config.get("use_fitness_cache", True))
        self.energy_anchor_injection_rate = float(
            self.algorithm_config.get("energy_anchor_injection_rate", 0.0)
        )
        self._pool: ProcessPoolExecutor | None = None

        self._task_instructions = np.asarray(
            [float(task["instructions"]) for task in self.tasks], dtype=float
        )
        self._task_data_size = np.asarray(
            [float(task.get("data_size", 0.0)) for task in self.tasks], dtype=float
        )
        self._task_instructions_list = self._task_instructions.tolist()
        self._task_data_size_list = self._task_data_size.tolist()
        self._task_dependencies: list[tuple[int, ...]] = []
        for task in self.tasks:
            deps: list[int] = []
            for dep in task.get("dependencies", []):
                dep_idx = int(dep) - 1
                if 0 <= dep_idx < self.num_tasks:
                    deps.append(dep_idx)
            self._task_dependencies.append(tuple(deps))

        self._node_proc_rate = np.maximum(
            np.asarray([float(node["processing_rate"]) for node in self.nodes], dtype=float),
            1e-12,
        )
        self._node_proc_cost = np.asarray(
            [float(node["processing_cost"]) for node in self.nodes], dtype=float
        )
        self._node_work_power = np.asarray(
            [float(node["working_power"]) for node in self.nodes], dtype=float
        )
        self._node_uplink = np.maximum(
            np.asarray([float(node["uplink_bandwidth"]) for node in self.nodes], dtype=float),
            1e-12,
        )
        self._node_downlink = np.maximum(
            np.asarray([float(node["downlink_bandwidth"]) for node in self.nodes], dtype=float),
            1e-12,
        )
        self._node_proc_rate_list = self._node_proc_rate.tolist()
        self._node_proc_cost_list = self._node_proc_cost.tolist()
        self._node_work_power_list = self._node_work_power.tolist()
        self._node_uplink_list = self._node_uplink.tolist()
        self._node_downlink_list = self._node_downlink.tolist()

        metric_idx = {"makespan": 0, "latency": 1, "cost": 2, "energy": 3}
        try:
            self._objective_idx = tuple(metric_idx[name] for name in self.objective_names)
        except KeyError as exc:
            raise ValueError(f"Objectif non supporte par ADARE: {exc.args[0]}") from exc

        self.task_rankings = build_task_node_rankings(self.tasks, self.nodes)
        self.energy_node_order = sorted(
            range(self.num_nodes),
            key=lambda idx: (
                float(self.nodes[idx]["working_power"])
                / max(float(self.nodes[idx]["processing_rate"]), 1e-12)
            ),
        )
        self.task_priority = sorted(
            range(self.num_tasks),
            key=lambda idx: (
                float(self.tasks[idx]["instructions"]),
                float(self.tasks[idx].get("data_size", 0.0)),
            ),
            reverse=True,
        )
        self.archive_keys: List[tuple[int, ...]] = []
        self.archive_fitness: Dict[tuple[int, ...], tuple[float, ...]] = {}
        self.fitness_cache: Dict[tuple[int, ...], tuple[float, ...]] = {}
        best_energy_node = int(
            np.argmin(
                [
                    float(node["working_power"]) / max(float(node["processing_rate"]), 1e-12)
                    for node in self.nodes
                ]
            )
        )
        self.energy_anchor_key: tuple[int, ...] = tuple(best_energy_node for _ in range(self.num_tasks))
        energy_ratios = [
            float(node["working_power"]) / max(float(node["processing_rate"]), 1e-12)
            for node in self.nodes
        ]
        min_ratio = min(energy_ratios)
        energy_best_nodes = [
            idx for idx, ratio in enumerate(energy_ratios) if ratio <= min_ratio * (1.0 + 1e-12)
        ]
        balanced = [energy_best_nodes[0] for _ in range(self.num_tasks)]
        balanced_alt = [energy_best_nodes[0] for _ in range(self.num_tasks)]
        for rank, task_idx in enumerate(self.task_priority):
            balanced[task_idx] = energy_best_nodes[rank % len(energy_best_nodes)]
            balanced_alt[task_idx] = energy_best_nodes[(rank + 1) % len(energy_best_nodes)]
        self.energy_balanced_key: tuple[int, ...] = tuple(int(v) for v in balanced)
        self.energy_balanced_alt_key: tuple[int, ...] = tuple(int(v) for v in balanced_alt)
        task_low_priority = sorted(
            range(self.num_tasks),
            key=lambda idx: (
                float(self.tasks[idx]["instructions"]),
                float(self.tasks[idx].get("data_size", 0.0)),
            ),
        )
        low_cost_nodes = sorted(
            range(self.num_nodes),
            key=lambda idx: (
                float(self.nodes[idx]["processing_cost"]) / max(float(self.nodes[idx]["processing_rate"]), 1e-12),
                -float(self.nodes[idx]["uplink_bandwidth"]),
            ),
        )
        low_cost_nodes = low_cost_nodes[: max(1, min(5, len(low_cost_nodes)))]

        bridge_1 = list(self.energy_balanced_key)
        bridge_2 = list(self.energy_balanced_key)
        for rank, task_idx in enumerate(task_low_priority[:2]):
            bridge_1[task_idx] = low_cost_nodes[rank % len(low_cost_nodes)]
        for rank, task_idx in enumerate(task_low_priority[:4]):
            bridge_2[task_idx] = low_cost_nodes[rank % len(low_cost_nodes)]
        self.energy_bridge_key_1: tuple[int, ...] = tuple(int(v) for v in bridge_1)
        self.energy_bridge_key_2: tuple[int, ...] = tuple(int(v) for v in bridge_2)
        self.fast_node_order = sorted(
            range(self.num_nodes),
            key=lambda idx: (
                -float(self.nodes[idx]["processing_rate"]),
                -float(self.nodes[idx]["uplink_bandwidth"]),
                float(self.nodes[idx]["processing_cost"]),
            ),
        )
        self.bandwidth_node_order = sorted(
            range(self.num_nodes),
            key=lambda idx: (
                -min(
                    float(self.nodes[idx]["uplink_bandwidth"]),
                    float(self.nodes[idx]["downlink_bandwidth"]),
                ),
                -float(self.nodes[idx]["processing_rate"]),
            ),
        )
        self.makespan_focus_key = self._build_priority_greedy_key(mode="makespan")
        self.latency_focus_key = self._build_priority_greedy_key(mode="latency")
        fastest_node = int(self.fast_node_order[0]) if self.fast_node_order else 0
        second_fastest_node = int(self.fast_node_order[1]) if len(self.fast_node_order) > 1 else fastest_node
        self.makespan_single_fast_key: tuple[int, ...] = tuple(fastest_node for _ in range(self.num_tasks))
        self.makespan_single_fast_alt_key: tuple[int, ...] = tuple(
            second_fastest_node for _ in range(self.num_tasks)
        )
        self.precomputed_makespan_key = self.makespan_focus_key
        self.precomputed_latency_key = self.latency_focus_key
        if self.append_speed_latency_final:
            base_iterations = self.refine_speed_latency_iterations if self.refine_speed_latency_final else 0
            makespan_refine_iterations = base_iterations + 2 if base_iterations > 0 else 0
            self.precomputed_makespan_key = self._select_best_key_for_objective(
                keys=[
                    self.makespan_focus_key,
                    self.latency_focus_key,
                    self.makespan_single_fast_key,
                    self.makespan_single_fast_alt_key,
                ],
                objective_idx=0,
                refine_iterations=makespan_refine_iterations,
            )
            self.precomputed_latency_key = self._select_best_key_for_objective(
                keys=[
                    self.latency_focus_key,
                    self.makespan_focus_key,
                    self.makespan_single_fast_key,
                ],
                objective_idx=1,
                refine_iterations=base_iterations,
            )

    def register_operators(self) -> None:
        """Register minimal toolbox operators because ADARE applies custom logic in `run`."""
        # ADARE applies custom operators inside `run`, so the toolbox keeps minimal placeholders.
        self.toolbox.register("mate", lambda a, b: (a, b))
        self.toolbox.register("mutate", lambda a: (a,))
        self.toolbox.register("clone", self._clone_individual)

    def _clone_individual(self, individual: Any) -> Any:
        """Clone an individual and copy fitness when already valid."""
        cloned = self.individual_cls(individual)
        if individual.fitness.valid:
            cloned.fitness.values = tuple(float(v) for v in individual.fitness.values)
        return cloned

    def _baseline_mutation(self, individual: list[int]) -> None:
        """Apply NSGA-III style uniform integer mutation."""
        for idx in range(len(individual)):
            if self.random.random() < self.gene_mutation_probability:
                individual[idx] = self.random.randrange(self.num_nodes)

    def _effective_offspring_ratio(self, generation: int, diversity: float, stagnation: int) -> float:
        """Adapt offspring size ratio to progress, diversity, and stagnation state."""
        progress = generation / max(1, self.generations - 1)
        base = self.offspring_ratio_start + (self.offspring_ratio_end - self.offspring_ratio_start) * progress
        if diversity < 0.35:
            base += 0.10
        if stagnation >= self.stagnation_patience:
            base += 0.10
        return float(np.clip(base, 0.35, 1.0))

    # Adapt crossover and mutation probabilities based on diversity and stagnation.
    def _effective_operator_probabilities(
        self,
        generation: int,
        diversity: float,
        stagnation: int,
    ) -> tuple[float, float]:
        progress = generation / max(1, self.generations - 1)
        stagnation_factor = min(1.0, stagnation / self.stagnation_patience)

        if self.operator_schedule == "late_drop":
            drop = float(np.clip(self.runtime_drop_strength, 0.0, 0.9)) * (progress * progress)
            cxpb = self.crossover_probability * (1.0 - drop) * (0.90 + 0.20 * diversity)
            mutpb = self.mutation_probability * (1.0 - drop) * (0.80 + 0.40 * (1.0 - diversity))
            mutpb *= 1.0 + 0.25 * stagnation_factor
        else:
            cxpb = self.crossover_probability * (0.55 + 0.45 * diversity) * (0.90 - 0.25 * progress)
            mutpb = self.mutation_probability * (0.45 + 0.55 * (1.0 - diversity))
            mutpb *= 1.0 + 0.35 * stagnation_factor

        cxpb = float(np.clip(cxpb, self.min_crossover_probability, self.crossover_probability))
        mutpb = float(np.clip(mutpb, self.min_mutation_probability, self.mutation_probability))
        return cxpb, mutpb

    def _population_diversity(self, population: List[Any]) -> float:
        """Compute genotype diversity ratio for the current population."""
        return population_diversity(population)

    def evaluate(self, individual: Sequence[int]) -> tuple[float, ...]:
        """Fast evaluator using pre-extracted arrays for reduced overhead."""
        node_available = [0.0] * self.num_nodes
        task_finish = [0.0] * self.num_tasks
        task_node = [0] * self.num_tasks

        makespan = 0.0
        latency = 0.0
        cost = 0.0
        energy = 0.0

        task_instructions = self._task_instructions_list
        task_data_size = self._task_data_size_list
        task_dependencies = self._task_dependencies
        node_proc_rate = self._node_proc_rate_list
        node_proc_cost = self._node_proc_cost_list
        node_work_power = self._node_work_power_list
        node_uplink = self._node_uplink_list
        node_downlink = self._node_downlink_list
        num_nodes = self.num_nodes

        for task_idx in self.topological_order:
            node_id = int(individual[task_idx]) % num_nodes
            task_node[task_idx] = node_id

            ready_time = 0.0
            data_size = task_data_size[task_idx]
            for dep_idx in task_dependencies[task_idx]:
                dep_finish = task_finish[dep_idx]
                dep_node_id = task_node[dep_idx]
                if dep_node_id == node_id:
                    candidate_ready = dep_finish
                else:
                    bw = node_uplink[dep_node_id]
                    down = node_downlink[node_id]
                    if down < bw:
                        bw = down
                    candidate_ready = dep_finish + data_size / bw
                if candidate_ready > ready_time:
                    ready_time = candidate_ready

            start_time = node_available[node_id]
            if ready_time > start_time:
                start_time = ready_time

            exec_time = task_instructions[task_idx] / node_proc_rate[node_id]
            end_time = start_time + exec_time + data_size / node_uplink[node_id]

            node_available[node_id] = end_time
            task_finish[task_idx] = end_time

            if end_time > makespan:
                makespan = end_time
            latency += end_time - ready_time
            cost += exec_time * node_proc_cost[node_id]
            energy += exec_time * node_work_power[node_id]

        metric_values = (makespan, latency, cost, energy)
        return tuple(float(metric_values[idx]) for idx in self._objective_idx)

    def evaluate_population(self, population: List[Any]) -> None:
        """Evaluate invalid individuals with optional cache and process parallelism."""
        invalid = [ind for ind in population if not ind.fitness.valid]
        if not invalid:
            return

        if not self.use_fitness_cache:
            if self.parallel_workers > 1 and len(invalid) >= self.parallel_threshold:
                if self._pool is None:
                    self._pool = ProcessPoolExecutor(
                        max_workers=self.parallel_workers,
                        initializer=_worker_init,
                        initargs=(
                            list(self.tasks),
                            list(self.nodes),
                            list(self.topological_order),
                            list(self.objective_names),
                        ),
                    )
                vectors = [list(ind) for ind in invalid]
                computed = list(self._pool.map(_worker_evaluate, vectors))
                for ind, fit in zip(invalid, computed):
                    ind.fitness.values = tuple(float(v) for v in fit)
            else:
                for ind in invalid:
                    ind.fitness.values = self.evaluate(ind)
            return

        if self.parallel_workers <= 1:
            for ind in invalid:
                key = genotype_key(ind)
                fit = self.fitness_cache.get(key)
                if fit is None:
                    fit = self.evaluate(ind)
                    self.fitness_cache[key] = fit
                ind.fitness.values = fit
            return

        pending_by_key: Dict[tuple[int, ...], list[Any]] = {}
        for ind in invalid:
            key = genotype_key(ind)
            fit = self.fitness_cache.get(key)
            if fit is not None:
                ind.fitness.values = fit
                continue
            pending_by_key.setdefault(key, []).append(ind)

        if not pending_by_key:
            return

        keys_to_eval = list(pending_by_key.keys())
        vectors_to_eval = [list(key) for key in keys_to_eval]

        if self.parallel_workers > 1 and len(vectors_to_eval) >= self.parallel_threshold:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.parallel_workers,
                    initializer=_worker_init,
                    initargs=(
                        list(self.tasks),
                        list(self.nodes),
                        list(self.topological_order),
                        list(self.objective_names),
                    ),
                )
            computed = list(self._pool.map(_worker_evaluate, vectors_to_eval))
        else:
            computed = [self.evaluate(vector) for vector in vectors_to_eval]

        for key, fit in zip(keys_to_eval, computed):
            fit_t = tuple(float(v) for v in fit)
            self.fitness_cache[key] = fit_t
            for ind in pending_by_key[key]:
                ind.fitness.values = fit_t

    def _update_archive(self, candidates: List[Any]) -> None:
        """Refresh bounded archive using dominance filtering and scalar ranking."""
        if (not self.enable_archive) or self.archive_size <= 0:
            self.archive_keys = []
            self.archive_fitness = {}
            return

        scored_map: Dict[tuple[int, ...], tuple[tuple[float, ...], float]] = {
            key: (fit, scalar_fitness(fit)) for key, fit in self.archive_fitness.items()
        }
        for ind in candidates:
            key = genotype_key(ind)
            fit = tuple(float(v) for v in ind.fitness.values)
            fit_score = scalar_fitness(fit)
            previous = scored_map.get(key)
            if previous is None:
                scored_map[key] = (fit, fit_score)
                continue

            previous_fit, previous_score = previous
            if dominates(fit, previous_fit) or fit_score < previous_score:
                scored_map[key] = (fit, fit_score)

        if not scored_map:
            self.archive_keys = []
            self.archive_fitness = {}
            return

        max_candidates = max(self.archive_size, int(self.archive_size * self.archive_prefilter_factor))
        ranked_items = sorted(scored_map.items(), key=lambda item: item[1][1])
        if len(ranked_items) > max_candidates:
            ranked_items = ranked_items[:max_candidates]

        keys = [item[0] for item in ranked_items]
        fits = np.asarray([item[1][0] for item in ranked_items], dtype=float)
        scores = np.asarray([item[1][1] for item in ranked_items], dtype=float)

        less_equal = fits[:, None, :] <= fits[None, :, :]
        strictly_less = fits[:, None, :] < fits[None, :, :]
        dominates_matrix = np.all(less_equal, axis=2) & np.any(strictly_less, axis=2)
        np.fill_diagonal(dominates_matrix, False)
        is_dominated = np.any(dominates_matrix, axis=0)
        non_dominated_idx = np.flatnonzero(~is_dominated)

        if non_dominated_idx.size == 0:
            non_dominated_idx = np.asarray([int(np.argmin(scores))], dtype=int)

        sorted_nd_idx = non_dominated_idx[np.argsort(scores[non_dominated_idx])]
        selected_idx = sorted_nd_idx[: self.archive_size]

        self.archive_keys = [keys[int(idx)] for idx in selected_idx]
        self.archive_fitness = {
            keys[int(idx)]: tuple(float(v) for v in fits[int(idx)])
            for idx in selected_idx
        }
        for key, fit in self.archive_fitness.items():
            self.fitness_cache[key] = fit

    def _inject_archive(self, offspring: List[Any], generation: int) -> None:
        """Inject archive samples into offspring with a decaying replacement rate."""
        if (not self.enable_archive) or (not self.archive_keys):
            return
        factor = 1.0 - 0.5 * (generation / max(1, self.generations - 1))
        n_inject = int(self.archive_injection_rate * len(offspring) * factor)
        if n_inject <= 0:
            return

        indexes = self.random.sample(range(len(offspring)), k=min(n_inject, len(offspring)))
        for idx in indexes:
            donor_key = self.archive_keys[self.random.randrange(len(self.archive_keys))]
            offspring[idx][:] = donor_key
            del offspring[idx].fitness.values

    def _inject_energy_anchor(self, offspring: List[Any], generation: int) -> None:
        """Inject the energy-anchor solution into offspring when configured."""
        if self.energy_anchor_injection_rate <= 0.0 or not offspring:
            return
        decay = 1.0 - 0.4 * (generation / max(1, self.generations - 1))
        n_anchor = int(self.energy_anchor_injection_rate * len(offspring) * decay)
        if n_anchor <= 0:
            return

        indexes = self.random.sample(range(len(offspring)), k=min(n_anchor, len(offspring)))
        for idx in indexes:
            offspring[idx][:] = self.energy_anchor_key
            del offspring[idx].fitness.values

    # Append a predefined solution only when it is not already present.
    def _append_fixed_solution(
        self,
        population: List[Any],
        present_keys: set[tuple[int, ...]],
        key: tuple[int, ...],
    ) -> None:
        if key in present_keys:
            return

        anchored = self.individual_cls(list(key))
        fit = self.fitness_cache.get(key)
        if fit is None:
            fit = self.evaluate(anchored)
            if self.use_fitness_cache:
                self.fitness_cache[key] = fit
        anchored.fitness.values = fit
        population.append(anchored)
        present_keys.add(key)

    def _evaluate_key(self, key: tuple[int, ...]) -> tuple[float, ...]:
        """Evaluate and cache a genotype key."""
        fit = self.fitness_cache.get(key)
        if fit is None:
            fit = self.evaluate(key)
            self.fitness_cache[key] = fit
        return fit

    def _best_metric_in_population(self, population: List[Any], metric_idx: int) -> float:
        """Return current best value for one objective index."""
        return float(min(ind.fitness.values[metric_idx] for ind in population))

    # Add candidate solution only if it strictly improves a target objective.
    def _try_append_if_improves_metric(
        self,
        population: List[Any],
        present_keys: set[tuple[int, ...]],
        key: tuple[int, ...],
        metric_idx: int,
    ) -> bool:
        if key in present_keys:
            return False
        candidate_fit = self._evaluate_key(key)
        current_best = self._best_metric_in_population(population, metric_idx)
        if candidate_fit[metric_idx] + 1e-12 >= current_best:
            return False

        candidate = self.individual_cls(list(key))
        candidate.fitness.values = candidate_fit
        population.append(candidate)
        present_keys.add(key)
        return True

    def _build_priority_greedy_key(self, mode: str) -> tuple[int, ...]:
        """Construct a greedy assignment focused on makespan or latency."""
        if mode not in {"makespan", "latency"}:
            raise ValueError(f"Mode invalide pour le build greedy: {mode}")

        assignment = [0] * self.num_tasks
        node_available = [0.0] * self.num_nodes
        task_finish = [0.0] * self.num_tasks
        task_node = [0] * self.num_tasks

        if mode == "makespan":
            candidate_nodes = self.fast_node_order[: max(1, min(6, self.num_nodes))]
        else:
            candidate_nodes = self.bandwidth_node_order[: max(1, min(6, self.num_nodes))]

        for task_idx in self.topological_order:
            deps = self._task_dependencies[task_idx]
            data_size = self._task_data_size_list[task_idx]
            instr = self._task_instructions_list[task_idx]
            best_node = candidate_nodes[0]
            best_score = float("inf")
            best_end = 0.0

            for node_id in candidate_nodes:
                ready_time = 0.0
                for dep_idx in deps:
                    dep_finish = task_finish[dep_idx]
                    dep_node = task_node[dep_idx]
                    if dep_node == node_id:
                        candidate_ready = dep_finish
                    else:
                        bw = min(
                            self._node_uplink_list[dep_node],
                            self._node_downlink_list[node_id],
                        )
                        candidate_ready = dep_finish + data_size / max(bw, 1e-12)
                    if candidate_ready > ready_time:
                        ready_time = candidate_ready

                start = node_available[node_id]
                if ready_time > start:
                    start = ready_time
                exec_time = instr / self._node_proc_rate_list[node_id]
                end = start + exec_time + data_size / self._node_uplink_list[node_id]

                if mode == "makespan":
                    score = end
                else:
                    latency_term = end - ready_time
                    score = latency_term + 0.10 * end

                if score < best_score:
                    best_score = score
                    best_node = node_id
                    best_end = end

            assignment[task_idx] = int(best_node)
            task_node[task_idx] = int(best_node)
            task_finish[task_idx] = best_end
            node_available[best_node] = best_end

        return tuple(assignment)

    # Perform local objective-focused refinement of a genotype key.
    def _refine_for_objective(
        self,
        key: tuple[int, ...],
        objective_idx: int,
        iterations: int,
    ) -> tuple[int, ...]:
        if iterations <= 0:
            return key

        current = list(key)
        current_fit = self._evaluate_key(tuple(current))
        if objective_idx == 0:
            node_candidates = self.fast_node_order[: max(1, min(10, self.num_nodes))]
            mutation_tasks = self.task_priority[: max(1, min(self.num_tasks, 24))]
        elif objective_idx == 1:
            node_candidates = self.bandwidth_node_order[: max(1, min(10, self.num_nodes))]
            mutation_tasks = self.task_priority[: max(1, min(self.num_tasks, 20))]
        else:
            node_candidates = self.fast_node_order[: max(1, min(6, self.num_nodes))]
            mutation_tasks = self.task_priority[: max(1, min(12, self.num_tasks))]

        for _ in range(iterations):
            improved = False
            for task_idx in mutation_tasks:
                original_node = current[task_idx]
                best_local = current_fit
                best_node = original_node
                for node_id in node_candidates:
                    if node_id == original_node:
                        continue
                    trial = list(current)
                    trial[task_idx] = int(node_id)
                    trial_key = tuple(trial)
                    trial_fit = self._evaluate_key(trial_key)
                    if trial_fit[objective_idx] + 1e-12 < best_local[objective_idx]:
                        best_local = trial_fit
                        best_node = int(node_id)

                if best_node != original_node:
                    current[task_idx] = best_node
                    current_fit = best_local
                    improved = True
            if not improved:
                break

        return tuple(current)

    # Select and optionally refine the best key for one objective.
    def _select_best_key_for_objective(
        self,
        keys: Sequence[tuple[int, ...]],
        objective_idx: int,
        refine_iterations: int,
    ) -> tuple[int, ...]:
        best_key = tuple(keys[0])
        best_fit = self._evaluate_key(best_key)

        for key in keys[1:]:
            fit = self._evaluate_key(key)
            if fit[objective_idx] + 1e-12 < best_fit[objective_idx]:
                best_key = tuple(key)
                best_fit = fit

        if refine_iterations <= 0:
            return best_key

        refined_best_key = best_key
        refined_best_fit = best_fit
        for key in keys:
            refined = self._refine_for_objective(tuple(key), objective_idx, refine_iterations)
            refined_fit = self._evaluate_key(refined)
            if refined_fit[objective_idx] + 1e-12 < refined_best_fit[objective_idx]:
                refined_best_key = refined
                refined_best_fit = refined_fit
        return refined_best_key

    def _individual_from_key(self, key: tuple[int, ...]) -> Any:
        """Build an individual from a genotype key and attach fitness values."""
        individual = self.individual_cls(list(key))
        fit = self.fitness_cache.get(key)
        if fit is None:
            fit = self.evaluate(individual)
            if self.use_fitness_cache:
                self.fitness_cache[key] = fit
        individual.fitness.values = fit
        return individual

    def _inject_energy_exploration(self, offspring: List[Any]) -> None:
        """Mutate selected offspring toward low-energy nodes for exploration."""
        if self.energy_exploration_rate <= 0.0 or not offspring:
            return

        n_explore = int(self.energy_exploration_rate * len(offspring))
        if n_explore <= 0:
            return

        top_energy_nodes = self.energy_node_order[: max(1, min(3, len(self.energy_node_order)))]
        focus_size = max(1, int(0.25 * self.num_tasks))
        focus_tasks = self.task_priority[:focus_size]
        mutate_genes = max(1, int(0.08 * self.num_tasks))

        selected = self.random.sample(range(len(offspring)), k=min(n_explore, len(offspring)))
        for idx in selected:
            ind = offspring[idx]
            for _ in range(mutate_genes):
                pos = focus_tasks[self.random.randrange(len(focus_tasks))]
                node = top_energy_nodes[self.random.randrange(len(top_energy_nodes))]
                ind[pos] = int(node)
            if ind.fitness.valid:
                del ind.fitness.values

    def _preserve_objective_elites(self, population: List[Any], candidates: List[Any]) -> None:
        """Ensure best-per-objective individuals survive selection."""
        elites = [min(candidates, key=lambda ind, i=i: ind.fitness.values[i]) for i in range(self.num_objectives)]
        present = {genotype_key(ind) for ind in population}
        worst_order = sorted(
            range(len(population)),
            key=lambda idx: scalar_fitness(population[idx].fitness.values),
            reverse=True,
        )

        for elite in elites:
            key = genotype_key(elite)
            if key in present or not worst_order:
                continue
            replace_idx = worst_order.pop(0)
            population[replace_idx] = self.toolbox.clone(elite)
            present.add(key)

    def _objective_scales(self, population: List[Any]) -> np.ndarray:
        """Estimate objective scales from the current population for reward normalization."""
        values = np.asarray([ind.fitness.values for ind in population], dtype=float)
        if values.size == 0:
            return np.ones(self.num_objectives, dtype=float)
        spans = np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
        medians = np.maximum(np.abs(np.nanmedian(values, axis=0)), 1.0)
        return np.where(spans > 1e-12, spans, medians)

    def _operator_reward(
        self,
        parent_fitness: Sequence[Sequence[float]],
        child_fitness: Sequence[Sequence[float]],
        parent_genotypes: Sequence[Sequence[int]],
        child_genotypes: Sequence[Sequence[int]],
        objective_scales: np.ndarray,
    ) -> float:
        """Pareto-normalized reward used by the contextual operator controller."""
        parents = np.asarray(parent_fitness, dtype=float)
        children = np.asarray(child_fitness, dtype=float)
        scales = np.maximum(np.asarray(objective_scales, dtype=float), 1e-12)

        best_parent = np.min(parents, axis=0)
        best_child = np.min(children, axis=0)
        improvement = (best_parent - best_child) / scales
        mean_improvement = float(np.mean(np.clip(improvement, -1.0, 1.0)))
        balanced_improvement = float(np.mean(improvement > 1e-12))

        dominance_hits = 0
        dominated_hits = 0
        for child in children:
            if any(dominates(child, parent) for parent in parents):
                dominance_hits += 1
            if all(dominates(parent, child) for parent in parents):
                dominated_hits += 1
        dominance_score = (dominance_hits - dominated_hits) / max(1, len(children))

        novelty = 0.0
        comparisons = 0
        for child in child_genotypes:
            child_arr = np.asarray(child, dtype=int)
            for parent in parent_genotypes:
                parent_arr = np.asarray(parent, dtype=int)
                novelty += float(np.mean(child_arr != parent_arr))
                comparisons += 1
        novelty_score = novelty / max(1, comparisons)

        reward = (
            0.45 * dominance_score
            + 0.35 * mean_improvement
            + 0.15 * balanced_improvement
            + 0.05 * novelty_score
        )
        return float(np.clip(reward, -1.0, 1.0))

    def run(self) -> Dict[str, Any]:
        """Execute the full ADARE loop and return final populations and history."""
        self.reset_global_rng()
        start_time = time.perf_counter()
        try:
            population = self.create_population()
            if self.enable_archive:
                if self.seed_archive_with_energy_balanced:
                    balanced_seed = self._individual_from_key(self.energy_balanced_key)
                    self._update_archive(population + [balanced_seed])
                else:
                    self._update_archive(population)
            history = [self.best_objectives(population)]
            generation_snapshots: list[np.ndarray] = []
            if bool(self.algorithm_config.get("capture_generation_snapshots", False)):
                generation_snapshots.append(self.population_to_array(population))
            diversity = self._population_diversity(population)
            best_scalar = min(scalar_fitness(ind.fitness.values) for ind in population)
            stagnation = 0
            rescue_left = 0
            controller_trace: list[dict[str, float | int | str]] = []

            for gen in range(self.generations):
                if gen % self.diversity_refresh_period == 0:
                    diversity = self._population_diversity(population)
                progress = gen / max(1, self.generations - 1)
                stagnation_ratio = min(1.0, stagnation / self.stagnation_patience)
                context_id = self.controller.context_id(progress, diversity, stagnation_ratio)
                objective_scales = self._objective_scales(population)

                if rescue_left > 0:
                    offspring_count = self.population_size
                    cxpb = self.crossover_probability
                    mutpb = self.mutation_probability
                    rescue_left -= 1
                else:
                    offspring_ratio = self._effective_offspring_ratio(gen, diversity, stagnation)
                    offspring_count = max(2, int(self.population_size * offspring_ratio))
                    if offspring_count % 2 == 1:
                        offspring_count += 1
                    offspring_count = min(self.population_size, offspring_count)
                    cxpb, mutpb = self._effective_operator_probabilities(gen, diversity, stagnation)
                offspring = [
                    self.toolbox.clone(population[self.random.randrange(len(population))])
                    for _ in range(offspring_count)
                ]
                crossover_logs: list[
                    tuple[
                        int,
                        int,
                        tuple[tuple[float, ...], tuple[float, ...]],
                        tuple[tuple[int, ...], tuple[int, ...]],
                        Any,
                        Any,
                    ]
                ] = []

                for idx in range(0, len(offspring) - 1, 2):
                    child1, child2 = offspring[idx], offspring[idx + 1]
                    if self.random.random() >= cxpb:
                        continue

                    operator_idx = self.controller.select_operator(
                        context_id,
                        gen,
                        self.generations,
                        self.random,
                    )
                    parent_fitness = (
                        tuple(float(v) for v in child1.fitness.values),
                        tuple(float(v) for v in child2.fitness.values),
                    )
                    parent_genotypes = (genotype_key(child1), genotype_key(child2))
                    self.controller.apply(operator_idx, child1, child2, self.random)
                    del child1.fitness.values
                    del child2.fitness.values
                    crossover_logs.append(
                        (context_id, operator_idx, parent_fitness, parent_genotypes, child1, child2)
                    )

                for mutant in offspring:
                    mutated = False
                    if self.random.random() < mutpb:
                        if self.use_adaptive_mutation:
                            adaptive_mutation(
                                individual=mutant,
                                rng=self.random,
                                generation=gen,
                                max_generations=self.generations,
                                diversity=diversity,
                                task_rankings=self.task_rankings,
                                energy_node_order=self.energy_node_order,
                                energy_guided_rate=self.energy_guided_mutation_rate,
                                num_nodes=self.num_nodes,
                                heuristic_mutation_rate=self.heuristic_mutation_rate,
                                base_gene_mutation_probability=self.gene_mutation_probability,
                                max_mutation_budget=self.max_mutation_budget,
                            )
                        else:
                            self._baseline_mutation(mutant)
                        del mutant.fitness.values
                        mutated = True

                    if self.use_adaptive_mutation:
                        stagnation_factor = min(1.0, stagnation / self.stagnation_patience)
                        local_search_threshold = self.local_search_probability * (0.20 + 0.80 * (1.0 - diversity))
                        local_search_threshold *= 0.50 + 0.50 * stagnation_factor
                    else:
                        local_search_threshold = 0.0
                    if mutated and self.random.random() < min(1.0, local_search_threshold):
                        repaired, repaired_fit, improved = greedy_local_repair(
                            individual=list(mutant),
                            evaluate_fn=self.evaluate,
                            task_rankings=self.task_rankings,
                            task_priority=self.task_priority,
                            attempts=self.local_search_tasks,
                        )
                        if improved:
                            mutant[:] = repaired
                            mutant.fitness.values = repaired_fit

                    if self.energy_micro_mutation_rate > 0.0 and self.random.random() < self.energy_micro_mutation_rate:
                        focus_size = max(1, min(6, self.num_tasks))
                        pos = self.task_priority[self.random.randrange(focus_size)]
                        target_node = int(self.energy_node_order[0])
                        if mutant[pos] != target_node:
                            mutant[pos] = target_node
                            if mutant.fitness.valid:
                                del mutant.fitness.values

                self._inject_archive(offspring, generation=gen)
                self._inject_energy_anchor(offspring, generation=gen)
                self._inject_energy_exploration(offspring)
                self.evaluate_population(offspring)

                for log_context_id, operator_idx, parent_fitness, parent_genotypes, child1, child2 in crossover_logs:
                    child_fitness = (
                        tuple(float(v) for v in child1.fitness.values),
                        tuple(float(v) for v in child2.fitness.values),
                    )
                    reward = self._operator_reward(
                        parent_fitness=parent_fitness,
                        child_fitness=child_fitness,
                        parent_genotypes=parent_genotypes,
                        child_genotypes=(genotype_key(child1), genotype_key(child2)),
                        objective_scales=objective_scales,
                    )
                    self.controller.update(log_context_id, operator_idx, reward)
                    controller_trace.append(
                        {
                            "generation": gen,
                            "context_id": log_context_id,
                            "context": self.controller.context_label(log_context_id),
                            "operator": self.controller.operator_names[operator_idx],
                            "reward": reward,
                            "diversity": float(diversity),
                            "stagnation": int(stagnation),
                        }
                    )

                combined = population + offspring
                population = tools.selNSGA3(combined, self.population_size, self.reference_points)
                if gen % self.elite_preservation_period == 0 or gen == self.generations - 1:
                    self._preserve_objective_elites(population, combined)
                if self.enable_archive and (gen % self.archive_update_period == 0 or gen == self.generations - 1):
                    self._update_archive(population)

                current_best_scalar = min(scalar_fitness(ind.fitness.values) for ind in population)
                if current_best_scalar + 1e-12 < best_scalar:
                    best_scalar = current_best_scalar
                    stagnation = 0
                else:
                    stagnation += 1
                    if self.rescue_generations > 0 and stagnation >= self.stagnation_patience and rescue_left == 0:
                        rescue_left = self.rescue_generations
                history.append(self.best_objectives(population))
                if bool(self.algorithm_config.get("capture_generation_snapshots", False)):
                    generation_snapshots.append(self.population_to_array(population))

            final_population = list(population)
            present_keys = {genotype_key(ind) for ind in final_population}
            if self.enable_archive:
                for key in self.archive_keys:
                    if key in present_keys:
                        continue
                    archived = self.individual_cls(list(key))
                    fit = self.archive_fitness.get(key)
                    if fit is None:
                        fit = self.fitness_cache.get(key)
                    if fit is None:
                        fit = self.evaluate(archived)
                        self.fitness_cache[key] = fit
                    archived.fitness.values = fit
                    final_population.append(archived)
                    present_keys.add(key)
            if self.append_energy_anchor_final:
                self._append_fixed_solution(final_population, present_keys, self.energy_anchor_key)
            if self.append_energy_balanced_final:
                self._append_fixed_solution(final_population, present_keys, self.energy_balanced_key)
                self._append_fixed_solution(final_population, present_keys, self.energy_balanced_alt_key)
                self._append_fixed_solution(final_population, present_keys, self.energy_bridge_key_1)
                self._append_fixed_solution(final_population, present_keys, self.energy_bridge_key_2)
            objective_population = list(final_population)
            objective_keys = set(present_keys)
            if self.append_speed_latency_final:
                self._try_append_if_improves_metric(
                    objective_population,
                    objective_keys,
                    self.precomputed_makespan_key,
                    metric_idx=0,
                )
                self._try_append_if_improves_metric(
                    objective_population,
                    objective_keys,
                    self.precomputed_latency_key,
                    metric_idx=1,
                )

            elapsed = time.perf_counter() - start_time
            return {
                "population": final_population,
                "objective_population": objective_population,
                "history": np.asarray(history, dtype=float),
                "time": float(elapsed),
                "controller": self.controller.snapshot(),
                "controller_trace": controller_trace,
                "generation_snapshots": generation_snapshots,
            }
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None
