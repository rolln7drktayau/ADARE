from __future__ import annotations

"""NSGA-II baseline implementation for broader MOEA comparison."""

import time
from typing import Any, Dict, List

import numpy as np
from deap import tools  # type: ignore

from algorithms.base_algorithm import BaseAlgorithm
from algorithms.nsga3.operators import two_point_crossover, uniform_int_mutation
from algorithms.nsga3.utils import pairwise


class NSGA2Algorithm(BaseAlgorithm):
    """Reference NSGA-II with fixed two-point crossover and uniform mutation."""

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

    def run(self) -> Dict[str, Any]:
        self.reset_global_rng()
        start_time = time.perf_counter()
        try:
            population = self.create_population()
            population = tools.selNSGA2(population, len(population))
            history = [self.best_objectives(population)]

            for _ in range(self.generations):
                offspring = [
                    self.toolbox.clone(population[self.random.randrange(len(population))])
                    for _ in range(len(population))
                ]

                for child1, child2 in pairwise(offspring):
                    if self.random.random() < self.crossover_probability:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    if self.random.random() < self.mutation_probability:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                self.evaluate_population(offspring)
                population = tools.selNSGA2(population + offspring, self.population_size)
                history.append(self.best_objectives(population))

            elapsed = time.perf_counter() - start_time
            return {
                "population": population,
                "objective_population": population,
                "history": np.asarray(history, dtype=float),
                "time": float(elapsed),
            }
        finally:
            self.shutdown()
