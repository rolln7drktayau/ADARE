from __future__ import annotations

"""QMOEA/D-AWA-style decomposition baseline.

The implementation adapts MOEA/D with two mechanisms used by the QMOEA/D-AWA
family: periodic weight-vector adjustment and Q-learning control of the
neighborhood size. Problem-specific FJSP local-search heuristics from the
manufacturing paper are not copied because ADARE uses heterogeneous workflow
DAGs, not operation-machine sequences.
"""

import time
from typing import Any, Dict, List

import numpy as np
from deap import tools  # type: ignore

from algorithms.moead.core import MOEADAlgorithm


class QMOEADAWAAlgorithm(MOEADAlgorithm):
    """MOEA/D-AWA variant with Q-learning adaptive neighborhood control."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = float(self.algorithm_config.get("q_alpha", 0.25))
        self.gamma = float(self.algorithm_config.get("q_gamma", 0.70))
        self.epsilon_start = float(self.algorithm_config.get("epsilon_start", 0.25))
        self.epsilon_end = float(self.algorithm_config.get("epsilon_end", 0.04))
        self.weight_update_interval = max(1, int(self.algorithm_config.get("weight_update_interval", 5)))
        self.weight_archive_size = max(self.population_size, int(self.algorithm_config.get("weight_archive_size", 200)))
        self.t_actions = [max(2, int(v)) for v in self.algorithm_config.get("neighborhood_actions", [8, 16, 24, 32])]
        self.t_actions = [min(v, self.population_size) for v in self.t_actions]
        self.q_values = np.zeros((6, len(self.t_actions)), dtype=float)
        self.usage_count = np.zeros((6, len(self.t_actions)), dtype=int)
        self.archive: list[Any] = []

    def _state_id(self, generation: int, improvement: float) -> int:
        progress = generation / max(1, self.generations - 1)
        phase = 0 if progress < 0.34 else 1 if progress < 0.67 else 2
        progress_bin = 0 if improvement <= 1e-9 else 1
        return int(phase * 2 + progress_bin)

    def _epsilon(self, generation: int) -> float:
        progress = generation / max(1, self.generations - 1)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def _select_action(self, state_id: int, generation: int) -> int:
        if self.random.random() < self._epsilon(generation):
            action_idx = self.random.randrange(len(self.t_actions))
        else:
            action_idx = int(np.argmax(self.q_values[state_id]))
        self.usage_count[state_id, action_idx] += 1
        self.neighborhood_size = max(2, min(self.t_actions[action_idx], self.population_size))
        self.neighbors = self._build_neighbors()
        return action_idx

    def _update_q(self, state_id: int, action_idx: int, reward: float, next_state_id: int) -> None:
        old = self.q_values[state_id, action_idx]
        target = reward + self.gamma * float(np.max(self.q_values[next_state_id]))
        self.q_values[state_id, action_idx] = (1.0 - self.alpha) * old + self.alpha * target

    @staticmethod
    def _core_score(values: np.ndarray) -> float:
        return float(np.mean(np.min(values, axis=0)))

    def _update_archive(self, population: List[Any]) -> None:
        self.archive.extend(self.toolbox.clone(ind) for ind in population)
        unique: dict[tuple[int, ...], Any] = {}
        for ind in self.archive:
            unique.setdefault(tuple(int(g) for g in ind), ind)
        candidates = list(unique.values())
        fronts = tools.sortNondominated(candidates, len(candidates), first_front_only=True)
        nd = fronts[0] if fronts else candidates
        ranked = sorted(nd, key=lambda ind: float(np.mean(ind.fitness.values)))
        self.archive = [self.toolbox.clone(ind) for ind in ranked[: self.weight_archive_size]]

    def _adapt_weights(self, population: List[Any]) -> None:
        if not self.archive:
            return
        values = np.asarray([ind.fitness.values for ind in self.archive], dtype=float)
        ideal = np.min(values, axis=0)
        translated = np.maximum(values - ideal, 1e-9)
        directions = translated / np.maximum(np.sum(translated, axis=1, keepdims=True), 1e-12)
        if len(directions) < self.population_size:
            extra = self.np_random.dirichlet(np.ones(self.num_objectives), self.population_size - len(directions))
            directions = np.vstack([directions, extra])
        diversity_order = np.argsort(np.linalg.norm(directions - np.mean(directions, axis=0), axis=1))[::-1]
        adapted = directions[diversity_order[: self.population_size]]
        self.weights = np.maximum(adapted / np.maximum(np.sum(adapted, axis=1, keepdims=True), 1e-12), 1e-6)
        self.neighbors = self._build_neighbors()

    def run(self) -> Dict[str, Any]:
        self.reset_global_rng()
        start_time = time.perf_counter()
        try:
            population = self.create_population()
            ideal = np.min(np.asarray([ind.fitness.values for ind in population], dtype=float), axis=0)
            history = [self.best_objectives(population)]
            prev_score = self._core_score(np.asarray([ind.fitness.values for ind in population], dtype=float))

            for gen in range(self.generations):
                state_id = self._state_id(gen, 0.0)
                action_idx = self._select_action(state_id, gen)
                for subproblem_idx in range(self.population_size):
                    neighborhood = self.neighbors[subproblem_idx]
                    p1_idx = int(neighborhood[self.random.randrange(len(neighborhood))])
                    p2_idx = int(neighborhood[self.random.randrange(len(neighborhood))])
                    child = self.toolbox.clone(population[p1_idx])
                    donor = self.toolbox.clone(population[p2_idx])

                    if self.random.random() < self.crossover_probability:
                        self.toolbox.mate(child, donor)
                        del child.fitness.values
                    if self.random.random() < self.mutation_probability:
                        self.toolbox.mutate(child)
                        if child.fitness.valid:
                            del child.fitness.values

                    self.evaluate_population([child])
                    ideal = np.minimum(ideal, np.asarray(child.fitness.values, dtype=float))

                    shuffled_neighbors = list(int(idx) for idx in neighborhood)
                    self.random.shuffle(shuffled_neighbors)
                    replacements = 0
                    for neighbor_idx in shuffled_neighbors:
                        child_score = self._scalar_tchebycheff(child.fitness.values, ideal, self.weights[neighbor_idx])
                        current_score = self._scalar_tchebycheff(
                            population[neighbor_idx].fitness.values,
                            ideal,
                            self.weights[neighbor_idx],
                        )
                        if child_score <= current_score:
                            population[neighbor_idx] = self.toolbox.clone(child)
                            replacements += 1
                            if replacements >= self.replacement_limit:
                                break

                self._update_archive(population)
                if (gen + 1) % self.weight_update_interval == 0:
                    self._adapt_weights(population)

                current_score = self._core_score(np.asarray([ind.fitness.values for ind in population], dtype=float))
                improvement = max(0.0, prev_score - current_score)
                next_state_id = self._state_id(gen, improvement)
                reward = float(np.clip(improvement / max(abs(prev_score), 1e-9), -1.0, 1.0))
                self._update_q(state_id, action_idx, reward, next_state_id)
                prev_score = current_score
                history.append(self.best_objectives(population))

            final_population = tools.selNSGA3(population, min(len(population), self.population_size), self.reference_points)
            elapsed = time.perf_counter() - start_time
            return {
                "population": final_population,
                "objective_population": population,
                "history": np.asarray(history, dtype=float),
                "time": float(elapsed),
                "q_values": self.q_values.tolist(),
                "usage_count": self.usage_count.tolist(),
            }
        finally:
            self.shutdown()
