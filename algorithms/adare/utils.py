from __future__ import annotations

"""Utility helpers shared inside the ADARE implementation."""

from typing import Any, Iterable


def genotype_key(individual: Iterable[int]) -> tuple[int, ...]:
    """Return a hashable representation of an individual genotype.

    The key is used for caching fitness values and deduplicating populations.
    """
    if isinstance(individual, tuple):
        return individual
    return tuple(individual)


def population_diversity(population: list[Any]) -> float:
    """Estimate diversity as the ratio of unique genotypes in a population."""
    if not population:
        return 0.0
    unique = {genotype_key(ind) for ind in population}
    return float(len(unique) / len(population))
