from __future__ import annotations

"""Base abstractions shared by ADARE and NSGA-III implementations.

This module centralizes:
- DEAP type creation.
- Objective evaluation for a DAG schedule.
- Common population lifecycle utilities.
"""

import copy
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from deap import base, creator, tools  # type: ignore


def _ensure_deap_types(num_objectives: int) -> type:
    """Create (or validate) DEAP Fitness and Individual classes for minimization."""
    fitness_name = "FitnessMultiMin"
    individual_name = "AssignmentIndividual"

    if hasattr(creator, fitness_name):
        fitness_cls = getattr(creator, fitness_name)
        if len(fitness_cls.weights) != num_objectives:
            raise RuntimeError(
                "Existing DEAP fitness uses a different number of objectives. "
                "Restart the Python process to reset DEAP creators."
            )
    else:
        creator.create(fitness_name, base.Fitness, weights=(-1.0,) * num_objectives)

    if not hasattr(creator, individual_name):
        creator.create(individual_name, list, fitness=getattr(creator, fitness_name))

    return getattr(creator, individual_name)


def evaluate_schedule(
    individual: Sequence[int],
    tasks: Sequence[Dict[str, Any]],
    nodes: Sequence[Dict[str, Any]],
    topological_order: Sequence[int],
    objective_names: Sequence[str],
) -> Tuple[float, ...]:
    """Evaluate one task-to-node assignment over all requested objectives.

    The evaluator follows topological order, tracks node availability,
    models transfer delay when dependencies cross nodes, and accumulates:
    makespan, latency, monetary cost, and energy.
    """
    node_available_time = [0.0 for _ in nodes]
    task_finish_time = [0.0 for _ in tasks]
    task_node = [0 for _ in tasks]

    metrics = {
        "makespan": 0.0,
        "latency": 0.0,
        "cost": 0.0,
        "energy": 0.0,
    }

    for task_idx in topological_order:
        task = tasks[task_idx]
        node_id = int(individual[task_idx]) % len(nodes)
        node = nodes[node_id]
        task_node[task_idx] = node_id

        ready_time = 0.0
        for dep_id in task.get("dependencies", []):
            dep_idx = int(dep_id) - 1
            if dep_idx < 0 or dep_idx >= len(tasks):
                continue
            dep_finish = task_finish_time[dep_idx]
            dep_node = nodes[task_node[dep_idx]]
            if task_node[dep_idx] == node_id:
                transfer_delay = 0.0
            else:
                bw = min(
                    float(dep_node["uplink_bandwidth"]),
                    float(node["downlink_bandwidth"]),
                )
                transfer_delay = float(task.get("data_size", 0.0)) / max(bw, 1e-9)
            ready_time = max(ready_time, dep_finish + transfer_delay)

        start_time = max(ready_time, node_available_time[node_id])
        exec_time = float(task["instructions"]) / max(float(node["processing_rate"]), 1e-9)
        transfer_overhead = float(task.get("data_size", 0.0)) / max(
            float(node["uplink_bandwidth"]), 1e-9
        )
        end_time = start_time + exec_time + transfer_overhead

        node_available_time[node_id] = end_time
        task_finish_time[task_idx] = end_time

        metrics["makespan"] = max(metrics["makespan"], end_time)
        metrics["latency"] += end_time - ready_time
        metrics["cost"] += exec_time * float(node["processing_cost"])
        metrics["energy"] += exec_time * float(node["working_power"])

    return tuple(float(metrics[name]) for name in objective_names)


class BaseAlgorithm(ABC):
    """Abstract base class defining the common optimizer contract."""

    def __init__(
        self,
        shared_config: Dict[str, Any],
        algorithm_config: Dict[str, Any],
        nodes: Sequence[Dict[str, Any]],
        tasks: Sequence[Dict[str, Any]],
        topological_order: Sequence[int],
        objective_names: Sequence[str],
        seed: int,
        initial_population: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Initialize common state, random generators, and DEAP toolbox."""
        self.shared_config = shared_config
        self.algorithm_config = algorithm_config
        self.nodes = list(nodes)
        self.tasks = list(tasks)
        self.topological_order = list(topological_order)
        self.objective_names = list(objective_names)
        self.seed = int(seed)
        self.initial_population = initial_population

        self.population_size = int(shared_config["population_size"])
        self.generations = int(shared_config["generations"])
        self.crossover_probability = float(shared_config["crossover_probability"])
        self.mutation_probability = float(shared_config["mutation_probability"])
        self.reference_points_divisions = int(shared_config["reference_points_divisions"])
        self.gene_mutation_probability = float(
            shared_config.get("gene_mutation_probability", 1.0 / max(1, len(self.tasks)))
        )

        self.num_tasks = len(self.tasks)
        self.num_nodes = len(self.nodes)
        self.num_objectives = len(self.objective_names)
        if self.num_tasks == 0:
            raise ValueError("Le probleme ne contient aucune tache.")
        if self.num_nodes == 0:
            raise ValueError("Le probleme ne contient aucun noeud.")
        if self.num_objectives == 0:
            raise ValueError("Aucun objectif n'est defini pour l'optimisation.")

        self.random = random.Random(self.seed)
        self.np_random = np.random.default_rng(self.seed)

        self.individual_cls = _ensure_deap_types(self.num_objectives)
        self.reference_points = tools.uniform_reference_points(
            self.num_objectives, p=self.reference_points_divisions
        )

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_node", self._random_node)
        self.toolbox.register(
            "individual", tools.initRepeat, self.individual_cls, self.toolbox.attr_node, n=self.num_tasks
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("clone", copy.deepcopy)
        self.toolbox.register("evaluate", self.evaluate)

        self.register_operators()

    def _random_node(self) -> int:
        """Sample a random node index for one task gene."""
        return self.random.randrange(self.num_nodes)

    def evaluate(self, individual: Sequence[int]) -> Tuple[float, ...]:
        """Evaluate an individual using the shared schedule evaluator."""
        return evaluate_schedule(
            individual=individual,
            tasks=self.tasks,
            nodes=self.nodes,
            topological_order=self.topological_order,
            objective_names=self.objective_names,
        )

    def create_population(self) -> List[Any]:
        """Build initial population, optionally seeded with a fixed assignment pool."""
        population: List[Any] = []
        if self.initial_population:
            for assignment in self.initial_population[: self.population_size]:
                vector = list(assignment[: self.num_tasks])
                if len(vector) < self.num_tasks:
                    vector.extend(self._random_node() for _ in range(self.num_tasks - len(vector)))
                individual = self.individual_cls(vector)
                population.append(individual)

        while len(population) < self.population_size:
            population.append(self.toolbox.individual())

        self.evaluate_population(population)
        return population

    def reset_global_rng(self) -> None:
        """Reset global RNG states to keep runs reproducible across modules."""
        random.seed(self.seed)
        np.random.seed(self.seed % (2**32 - 1))

    def evaluate_population(self, population: Sequence[Any]) -> None:
        """Evaluate only invalid individuals to avoid redundant computations."""
        invalid = [ind for ind in population if not ind.fitness.valid]
        if not invalid:
            return
        fitnesses = [self.toolbox.evaluate(ind) for ind in invalid]
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

    def best_objectives(self, population: Sequence[Any]) -> np.ndarray:
        """Return the per-objective best value in the given population."""
        values = np.asarray([ind.fitness.values for ind in population], dtype=float)
        return np.min(values, axis=0)

    @staticmethod
    def population_to_array(population: Sequence[Any]) -> np.ndarray:
        """Convert a DEAP population into a 2D objective-value array."""
        return np.asarray([ind.fitness.values for ind in population], dtype=float)

    @abstractmethod
    def register_operators(self) -> None:
        """Register algorithm-specific evolutionary operators."""
        ...

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Execute one full optimization run and return artifacts."""
        ...
