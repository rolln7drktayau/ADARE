from __future__ import annotations

"""OVEA-style reference-vector MOEA with Q-learning operator adaptation.

This is an executable workflow-scheduling adaptation of the OVEA idea:
reference-vector environmental selection plus Q-learning crossover choice.
It is intentionally separated from ADARE so the learning policy, reward, and
selection pressure remain a genuine external baseline.
"""

import time
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
from deap import tools  # type: ignore

from algorithms.adare.operators import _cx_one_point, _cx_two_point, _cx_uniform, dominates, scalar_fitness
from algorithms.adare.utils import population_diversity
from algorithms.base_algorithm import BaseAlgorithm
from algorithms.nsga3.operators import uniform_int_mutation


class OVEAAlgorithm(BaseAlgorithm):
    """Reference-vector selection with Q-learning adaptive crossover choice."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = float(self.algorithm_config.get("q_alpha", 0.20))
        self.gamma = float(self.algorithm_config.get("q_gamma", 0.65))
        self.epsilon_start = float(self.algorithm_config.get("epsilon_start", 0.25))
        self.epsilon_end = float(self.algorithm_config.get("epsilon_end", 0.03))
        self.theta = float(self.algorithm_config.get("apd_theta", 5.0))
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
    def _reward(parent_values: Sequence[float], child_values: Sequence[float]) -> float:
        reward = scalar_fitness(tuple(parent_values)) - scalar_fitness(tuple(child_values))
        if dominates(child_values, parent_values):
            reward += 0.50
        elif dominates(parent_values, child_values):
            reward -= 0.50
        return float(np.clip(reward, -1.0, 1.0))

    def _update(self, state_id: int, op_idx: int, reward: float, next_state_id: int) -> None:
        old = self.q_values[state_id, op_idx]
        target = reward + self.gamma * float(np.max(self.q_values[next_state_id]))
        self.q_values[state_id, op_idx] = (1.0 - self.alpha) * old + self.alpha * target

    def _select_by_reference_vectors(self, candidates: Sequence[Any], generation: int) -> list[Any]:
        if len(candidates) <= self.population_size:
            return list(candidates)

        values = np.asarray([ind.fitness.values for ind in candidates], dtype=float)
        ideal = np.min(values, axis=0)
        translated = values - ideal
        norms = np.linalg.norm(translated, axis=1)
        normed = translated / np.maximum(norms[:, None], 1e-12)

        refs = np.asarray(self.reference_points, dtype=float)
        refs = refs / np.maximum(np.linalg.norm(refs, axis=1, keepdims=True), 1e-12)
        cosine = np.clip(normed @ refs.T, -1.0, 1.0)
        angles = np.arccos(cosine)
        associations = np.argmin(angles, axis=1)

        progress = (generation + 1) / max(1, self.generations)
        selected: list[Any] = []
        used: set[int] = set()
        for ref_idx in range(len(refs)):
            members = np.where(associations == ref_idx)[0]
            if len(members) == 0:
                continue
            member_angles = angles[members, ref_idx]
            apd = norms[members] * (1.0 + self.theta * progress * member_angles)
            best_local = int(members[int(np.argmin(apd))])
            if best_local not in used:
                selected.append(self.toolbox.clone(candidates[best_local]))
                used.add(best_local)
            if len(selected) >= self.population_size:
                break

        remaining = [idx for idx in np.argsort([scalar_fitness(tuple(v)) for v in values]) if int(idx) not in used]
        for idx in remaining:
            selected.append(self.toolbox.clone(candidates[int(idx)]))
            if len(selected) >= self.population_size:
                break
        return selected

    def run(self) -> Dict[str, Any]:
        self.reset_global_rng()
        start_time = time.perf_counter()
        try:
            population = self.create_population()
            history = [self.best_objectives(population)]

            for gen in range(self.generations):
                diversity = population_diversity(population)
                state_id = self._state_id(gen, diversity)
                offspring: list[Any] = []
                logs: list[tuple[int, int, tuple[float, ...], Any, Any]] = []

                while len(offspring) < self.population_size:
                    parent1 = self.toolbox.clone(population[self.random.randrange(len(population))])
                    parent2 = self.toolbox.clone(population[self.random.randrange(len(population))])
                    if self.random.random() < self.crossover_probability:
                        op_idx = self._select_operator(state_id, gen)
                        parent_values = min(
                            tuple(float(v) for v in parent1.fitness.values),
                            tuple(float(v) for v in parent2.fitness.values),
                            key=scalar_fitness,
                        )
                        self.operators[op_idx](parent1, parent2)
                        del parent1.fitness.values
                        del parent2.fitness.values
                        logs.append((state_id, op_idx, parent_values, parent1, parent2))
                    for child in (parent1, parent2):
                        if self.random.random() < self.mutation_probability:
                            self.toolbox.mutate(child)
                            if child.fitness.valid:
                                del child.fitness.values
                        offspring.append(child)
                        if len(offspring) >= self.population_size:
                            break

                self.evaluate_population(offspring)
                next_state_id = self._state_id(gen, population_diversity(offspring))
                for log_state, op_idx, parent_values, child1, child2 in logs:
                    child_values = min(
                        tuple(float(v) for v in child1.fitness.values),
                        tuple(float(v) for v in child2.fitness.values),
                        key=scalar_fitness,
                    )
                    self._update(log_state, op_idx, self._reward(parent_values, child_values), next_state_id)

                population = self._select_by_reference_vectors(population + offspring, gen)
                history.append(self.best_objectives(population))

            elapsed = time.perf_counter() - start_time
            final_population = tools.selNSGA3(population, min(len(population), self.population_size), self.reference_points)
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
