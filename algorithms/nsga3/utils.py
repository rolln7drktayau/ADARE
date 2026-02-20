from __future__ import annotations

"""Small NSGA-III helpers used to keep core logic readable."""

from typing import Iterator, Sequence, Tuple, TypeVar

T = TypeVar("T")


def pairwise(sequence: Sequence[T]) -> Iterator[Tuple[T, T]]:
    """Yield elements two by two for crossover operations."""
    for idx in range(0, len(sequence) - 1, 2):
        yield sequence[idx], sequence[idx + 1]
