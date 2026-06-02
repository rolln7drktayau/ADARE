from __future__ import annotations

"""Q-learning assisted NSGA-III baseline inspired by adaptive operator selection."""

import time
from typing import Any, Callable, Dict, List

import numpy as np
from deap import tools  # type: ignore

from algorithms.adare.operators import (
    _cx_one_point,
    _cx_two_point,
    _cx_uniform,
    dominates,
    scalar_fitness,
)
from algorithms.base_algorithm import BaseAlgorithm
from algorithms.nsga3.operators import uniform_int_mutation
from algorithms.nsga3.utils import pairwise
from algorithms.adare.utils import population_diversity


class QLNSGA3Algorithm(BaseAlgorithm):
    """NSGA-III with Q-learning adaptive crossover selection."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = float(self.algorithm_config.get("q_alpha", 0.25))
        self.gamma = float(self.algorithm_config.get("q_gamma", 0.70))
        self.epsilon_start = float(self.algorithm_config.get("epsilon_start", 0.30))
        self.epsilon_end = float(self.algorithm_config.get("epsilon_end", 0.05))
        self.operator_names = ["one_point", "two_point", "uniform_0_5", "uniform_0_8"]
        self.operators: list[Callable[[list[int], list[int]], None]] = [
            lambda a, b: _cx_one_point(a, b, self.random),
            lambda a, b: _cx_two_point(a, b, self.random),
            lambda a, b: _cx_uniform(a, b, self.random, indpb=0.5),
            lambda a, b: _cx_uniform(a, b, self.random, indpb=0.8),
        ]
        self.q_values = np.zeros((6, len(self.operators)), dtype=float)
        self.usage_count = np.zeros((6, len(self.operators)), dtype=int)

    def register_operators(self) -> None:
        self.toolbox.register("mutate", self._mutate)

    def _mutate(self, individual: List[int]) -> tuple[List[int]]:
        return uniform_int_mutation(
            individual=individual,
            rng=self.random,
            num_nodes=self.num_nodes,
            indpb=self.gene_mutation_probability,
        )

    def _state_id(self, generation: int, diversity: float) -> int:
        progress = generation / max(1, self.generations - 1)
        phase = 0 if progress < 0.34 else 1 if progress < 0.67 else 2
        diversity_bin = 0 if diversity < 0.45 else 1
        return int(phase * 2 + diversity_bin)

    def _epsilon(self, generation: int) -> float:
        progress = generation / max(1, self.generations - 1)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def _select_operator(self, state_id: int, generation: int) -> int:
        if self.random.random() < self._epsilon(generation):
            op_idx = self.random.randrange(len(self.operators))
        else:
            op_idx = int(np.argmax(self.q_values[state_id]))
        self.usage_count[state_id, op_idx] += 1
        return op_idx

    @staticmethod
    def _reward(parent_values: tuple[float, ...], child_values: tuple[float, ...]) -> float:
        reward = scalar_fitness(parent_values) - scalar_fitness(child_values)
        if dominates(child_values, parent_values):
            reward += 0.50
        elif dominates(parent_values, child_values):
            reward -= 0.50
        return float(np.clip(reward, -1.0, 1.0))

    def _update(self, state_id: int, op_idx: int, reward: float, next_state_id: int) -> None:
        old = self.q_values[state_id, op_idx]
        target = reward + self.gamma * float(np.max(self.q_values[next_state_id]))
        self.q_values[state_id, op_idx] = (1.0 - self.alpha) * old + self.alpha * target

    def run(self) -> Dict[str, Any]:
        self.reset_global_rng()
        start_time = time.perf_counter()
        try:
            population = self.create_population()
            history = [self.best_objectives(population)]

            for gen in range(self.generations):
                diversity = population_diversity(population)
                state_id = self._state_id(gen, diversity)
                offspring = [
                    self.toolbox.clone(population[self.random.randrange(len(population))])
                    for _ in range(len(population))
                ]
                crossover_logs: list[tuple[int, int, tuple[float, ...], Any, Any]] = []

                for child1, child2 in pairwise(offspring):
                    if self.random.random() < self.crossover_probability:
                        op_idx = self._select_operator(state_id, gen)
                        parent_values = min(
                            tuple(float(v) for v in child1.fitness.values),
                            tuple(float(v) for v in child2.fitness.values),
                            key=scalar_fitness,
                        )
                        self.operators[op_idx](child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                        crossover_logs.append((state_id, op_idx, parent_values, child1, child2))

                for mutant in offspring:
                    if self.random.random() < self.mutation_probability:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                self.evaluate_population(offspring)
                next_state_id = self._state_id(gen, population_diversity(offspring))
                for log_state_id, op_idx, parent_values, child1, child2 in crossover_logs:
                    child_values = min(
                        tuple(float(v) for v in child1.fitness.values),
                        tuple(float(v) for v in child2.fitness.values),
                        key=scalar_fitness,
                    )
                    self._update(log_state_id, op_idx, self._reward(parent_values, child_values), next_state_id)

                population = tools.selNSGA3(population + offspring, self.population_size, self.reference_points)
                history.append(self.best_objectives(population))

            elapsed = time.perf_counter() - start_time
            return {
                "population": population,
                "objective_population": population,
                "history": np.asarray(history, dtype=float),
                "time": float(elapsed),
                "q_values": self.q_values.tolist(),
                "usage_count": self.usage_count.tolist(),
            }
        finally:
            self.shutdown()
