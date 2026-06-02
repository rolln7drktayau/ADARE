from __future__ import annotations

"""Lightweight MOEA/D baseline using Tchebycheff decomposition."""

import time
from typing import Any, Dict, List, Sequence

import numpy as np
from deap import tools  # type: ignore

from algorithms.base_algorithm import BaseAlgorithm
from algorithms.nsga3.operators import two_point_crossover, uniform_int_mutation


class MOEADAlgorithm(BaseAlgorithm):
    """MOEA/D with uniform reference weights and neighborhood replacement."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.neighborhood_size = max(2, int(self.algorithm_config.get("neighborhood_size", 20)))
        self.replacement_limit = max(1, int(self.algorithm_config.get("replacement_limit", 2)))
        self.weights = self._build_weights()
        self.neighbors = self._build_neighbors()

    def register_operators(self) -> None:
        self.toolbox.register("mate", self._mate)
        self.toolbox.register("mutate", self._mutate)

    def _mate(self, ind1: List[int], ind2: List[int]) -> tuple[List[int], List[int]]:
        return two_point_crossover(ind1, ind2, self.random)

    def _mutate(self, individual: List[int]) -> tuple[List[int]]:
        return uniform_int_mutation(
            individual=individual,
            rng=self.random,
            num_nodes=self.num_nodes,
            indpb=self.gene_mutation_probability,
        )

    def _build_weights(self) -> np.ndarray:
        weights = np.asarray(self.reference_points, dtype=float)
        if len(weights) < self.population_size:
            extra = self.np_random.dirichlet(np.ones(self.num_objectives), self.population_size - len(weights))
            weights = np.vstack([weights, extra])
        weights = weights[: self.population_size]
        row_sums = np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
        return np.maximum(weights / row_sums, 1e-6)

    def _build_neighbors(self) -> np.ndarray:
        distances = np.linalg.norm(self.weights[:, None, :] - self.weights[None, :, :], axis=2)
        return np.argsort(distances, axis=1)[:, : min(self.neighborhood_size, self.population_size)]

    @staticmethod
    def _scalar_tchebycheff(values: Sequence[float], ideal: np.ndarray, weight: np.ndarray) -> float:
        vec = np.asarray(values, dtype=float)
        return float(np.max(weight * np.abs(vec - ideal)))

    def run(self) -> Dict[str, Any]:
        self.reset_global_rng()
        start_time = time.perf_counter()
        try:
            population = self.create_population()
            ideal = np.min(np.asarray([ind.fitness.values for ind in population], dtype=float), axis=0)
            history = [self.best_objectives(population)]

            for _ in range(self.generations):
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
                        child_score = self._scalar_tchebycheff(
                            child.fitness.values,
                            ideal,
                            self.weights[neighbor_idx],
                        )
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

                history.append(self.best_objectives(population))

            final_population = tools.selNSGA3(population, min(len(population), self.population_size), self.reference_points)
            elapsed = time.perf_counter() - start_time
            return {
                "population": final_population,
                "objective_population": population,
                "history": np.asarray(history, dtype=float),
                "time": float(elapsed),
            }
        finally:
            self.shutdown()
