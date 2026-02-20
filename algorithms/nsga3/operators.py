from __future__ import annotations

"""Genetic operators used by the NSGA-III baseline."""

import random
from typing import Tuple


def two_point_crossover(ind1: list[int], ind2: list[int], rng: random.Random) -> Tuple[list[int], list[int]]:
    """Swap one random segment between two individuals."""
    size = min(len(ind1), len(ind2))
    if size < 2:
        return ind1, ind2

    cx1 = rng.randrange(1, size)
    cx2 = rng.randrange(1, size)
    if cx2 < cx1:
        cx1, cx2 = cx2, cx1
    if cx1 == cx2:
        cx2 = min(size, cx1 + 1)
    ind1[cx1:cx2], ind2[cx1:cx2] = ind2[cx1:cx2], ind1[cx1:cx2]
    return ind1, ind2


def uniform_int_mutation(
    individual: list[int],
    rng: random.Random,
    num_nodes: int,
    indpb: float,
) -> Tuple[list[int]]:
    """Mutate each gene independently with probability `indpb`."""
    for idx in range(len(individual)):
        if rng.random() < indpb:
            individual[idx] = rng.randrange(num_nodes)
    return (individual,)
