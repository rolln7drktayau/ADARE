from __future__ import annotations

"""NSGA-III baseline implementation used for fair comparison with ADARE."""

import time
from typing import Any, Dict, List

import numpy as np
from deap import tools  # type: ignore

from algorithms.base_algorithm import BaseAlgorithm
from .operators import two_point_crossover, uniform_int_mutation
from .utils import pairwise


class NSGA3Algorithm(BaseAlgorithm):
    """Reference NSGA-III algorithm with standard crossover and mutation."""

    def register_operators(self) -> None:
        """Register baseline operators into the DEAP toolbox."""
        self.toolbox.register("mate", self._mate)
        self.toolbox.register("mutate", self._mutate)

    def _mate(self, ind1: List[int], ind2: List[int]) -> tuple[List[int], List[int]]:
        """Apply two-point crossover."""
        return two_point_crossover(ind1, ind2, self.random)

    def _mutate(self, individual: List[int]) -> tuple[List[int]]:
        """Apply uniform integer mutation over node assignments."""
        return uniform_int_mutation(
            individual=individual,
            rng=self.random,
            num_nodes=self.num_nodes,
            indpb=self.gene_mutation_probability,
        )

    def run(self) -> Dict[str, Any]:
        """Execute one NSGA-III optimization run and return full artifacts."""
        self.reset_global_rng()
        start_time = time.perf_counter()
        try:
            population = self.create_population()
            history = [self.best_objectives(population)]

            for _ in range(self.generations):
                # Offspring are sampled from the current population with replacement.
                offspring = [
                    self.toolbox.clone(population[self.random.randrange(len(population))])
                    for _ in range(len(population))
                ]

                # Apply crossover pairwise.
                for child1, child2 in pairwise(offspring):
                    if self.random.random() < self.crossover_probability:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Apply mutation independently.
                for mutant in offspring:
                    if self.random.random() < self.mutation_probability:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                self.evaluate_population(offspring)
                combined = population + offspring
                population = tools.selNSGA3(combined, self.population_size, self.reference_points)
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
