from __future__ import annotations

"""ADARE-specific operators and adaptive operator-selection logic."""

import math
import random
from typing import Callable, Sequence, Tuple

import numpy as np


def _cx_one_point(ind1: list[int], ind2: list[int], rng: random.Random) -> Tuple[list[int], list[int]]:
    """Apply one-point crossover."""
    size = min(len(ind1), len(ind2))
    if size < 2:
        return ind1, ind2
    point = rng.randrange(1, size)
    ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
    return ind1, ind2


def _cx_two_point(ind1: list[int], ind2: list[int], rng: random.Random) -> Tuple[list[int], list[int]]:
    """Apply two-point crossover."""
    size = min(len(ind1), len(ind2))
    if size < 2:
        return ind1, ind2
    p1 = rng.randrange(1, size)
    p2 = rng.randrange(1, size)
    if p2 < p1:
        p1, p2 = p2, p1
    if p1 == p2:
        p2 = min(size, p1 + 1)
    ind1[p1:p2], ind2[p1:p2] = ind2[p1:p2], ind1[p1:p2]
    return ind1, ind2


def _cx_uniform(ind1: list[int], ind2: list[int], rng: random.Random, indpb: float) -> Tuple[list[int], list[int]]:
    """Apply uniform crossover with per-gene probability `indpb`."""
    for idx in range(min(len(ind1), len(ind2))):
        if rng.random() < indpb:
            ind1[idx], ind2[idx] = ind2[idx], ind1[idx]
    return ind1, ind2


def dominates(values_a: Sequence[float], values_b: Sequence[float]) -> bool:
    """Return True if A dominates B in minimization sense."""
    strictly_better = False
    for value_a, value_b in zip(values_a, values_b):
        if value_a > value_b:
            return False
        if value_a < value_b:
            strictly_better = True
    return strictly_better


def scalar_fitness(values: Sequence[float]) -> float:
    """Compute a smooth scalar score used for ranking and operator reward."""
    if len(values) == 4:
        return (
            0.30 * math.log1p(float(values[0]))
            + 0.30 * math.log1p(float(values[1]))
            + 0.20 * math.log1p(float(values[2]))
            + 0.20 * math.log1p(float(values[3]))
        )

    n = len(values)
    if n == 0:
        return 0.0
    weight = 1.0 / n
    return float(sum(weight * math.log1p(float(v)) for v in values))


class AdaptiveOperatorController:
    """Contextual UCB controller for search-phase-aware crossover selection."""

    def __init__(
        self,
        exploration_start: float,
        exploration_end: float,
        alpha: float,
    ) -> None:
        """Initialize exploration schedule and operator value estimates."""
        self.exploration_start = float(exploration_start)
        self.exploration_end = float(exploration_end)
        self.alpha = float(alpha)

        self.operator_names = ["one_point", "two_point", "uniform_0_5", "uniform_0_8"]
        self.operators: list[Callable[[list[int], list[int], random.Random], Tuple[list[int], list[int]]]] = [
            _cx_one_point,
            _cx_two_point,
            lambda a, b, r: _cx_uniform(a, b, r, indpb=0.5),
            lambda a, b, r: _cx_uniform(a, b, r, indpb=0.8),
        ]
        self.num_contexts = 12
        self.q_values = np.zeros((self.num_contexts, len(self.operators)), dtype=float)
        self.usage_count = np.zeros((self.num_contexts, len(self.operators)), dtype=int)
        self.context_usage = np.zeros(self.num_contexts, dtype=int)

    def _epsilon(self, generation: int, max_generations: int) -> float:
        """Linearly anneal exploration from start to end over generations."""
        progress = generation / max(1, max_generations - 1)
        return float(
            self.exploration_start + (self.exploration_end - self.exploration_start) * progress
        )

    def context_id(self, progress: float, diversity: float, stagnation_ratio: float) -> int:
        """Discretize the search state into a compact context id."""
        if progress < 0.34:
            phase = 0
        elif progress < 0.67:
            phase = 1
        else:
            phase = 2
        diversity_bin = 0 if diversity < 0.45 else 1
        stagnation_bin = 1 if stagnation_ratio >= 0.70 else 0
        return int(phase * 4 + diversity_bin * 2 + stagnation_bin)

    def context_label(self, context_id: int) -> str:
        """Return a compact human-readable label for a context id."""
        phase_names = ("early", "middle", "late")
        phase = int(context_id) // 4
        rem = int(context_id) % 4
        diversity_label = "low_div" if rem < 2 else "high_div"
        stagnation_label = "stagnant" if rem % 2 else "moving"
        return f"{phase_names[phase]}:{diversity_label}:{stagnation_label}"

    def select_operator(
        self,
        context_id: int,
        generation: int,
        max_generations: int,
        rng: random.Random,
    ) -> int:
        """Pick an operator with contextual epsilon-greedy + UCB exploration."""
        epsilon = self._epsilon(generation, max_generations)
        context_id = int(np.clip(context_id, 0, self.num_contexts - 1))
        if rng.random() < epsilon:
            op_idx = rng.randrange(len(self.operators))
        else:
            total = float(self.context_usage[context_id] + 1)
            counts = self.usage_count[context_id]
            bonus = np.sqrt(2.0 * np.log(total + 1.0) / (counts + 1))
            op_idx = int(np.argmax(self.q_values[context_id] + bonus))
        self.usage_count[context_id, op_idx] += 1
        self.context_usage[context_id] += 1
        return op_idx

    def apply(self, operator_index: int, ind1: list[int], ind2: list[int], rng: random.Random) -> None:
        """Apply the selected crossover operator in place."""
        self.operators[operator_index](ind1, ind2, rng)

    def update(self, context_id: int, operator_index: int, reward: float) -> None:
        """Update context-specific operator value with exponential smoothing."""
        context_id = int(np.clip(context_id, 0, self.num_contexts - 1))
        self.q_values[context_id, operator_index] = (
            (1.0 - self.alpha) * self.q_values[context_id, operator_index] + self.alpha * float(reward)
        )

    def snapshot(self) -> dict[str, object]:
        """Return serializable controller state for interpretation and reporting."""
        return {
            "operator_names": list(self.operator_names),
            "context_labels": [self.context_label(idx) for idx in range(self.num_contexts)],
            "q_values": self.q_values.tolist(),
            "usage_count": self.usage_count.tolist(),
        }


def build_task_node_rankings(
    tasks: Sequence[dict[str, float]],
    nodes: Sequence[dict[str, float]],
) -> list[list[int]]:
    """Build per-task node ranking mixing speed, bandwidth, cost, and energy."""
    proc = np.asarray([float(node["processing_rate"]) for node in nodes], dtype=float)
    bw = np.asarray([float(node["uplink_bandwidth"]) for node in nodes], dtype=float)
    cost = np.asarray([float(node["processing_cost"]) for node in nodes], dtype=float)
    energy = np.asarray([float(node["working_power"]) for node in nodes], dtype=float)

    def norm(arr: np.ndarray) -> np.ndarray:
        """Normalize feature values into [0, 1]."""
        span = arr.max() - arr.min()
        if span < 1e-12:
            return np.ones_like(arr)
        return (arr - arr.min()) / span

    proc_n = norm(proc)
    bw_n = norm(bw)
    cost_n = norm(cost)
    energy_n = norm(energy)

    rankings: list[list[int]] = []
    for task in tasks:
        instructions = float(task["instructions"])
        data_size = float(task.get("data_size", 0.0))
        compute_ratio = instructions / max(instructions + data_size * 1e5, 1e-9)
        data_ratio = 1.0 - compute_ratio
        score = compute_ratio * proc_n + data_ratio * bw_n - 0.12 * cost_n - 0.11 * energy_n
        rankings.append(np.argsort(-score).tolist())
    return rankings


def adaptive_mutation(
    individual: list[int],
    rng: random.Random,
    generation: int,
    max_generations: int,
    diversity: float,
    task_rankings: Sequence[Sequence[int]],
    energy_node_order: Sequence[int] | None,
    energy_guided_rate: float,
    num_nodes: int,
    heuristic_mutation_rate: float,
    base_gene_mutation_probability: float,
    max_mutation_budget: int,
) -> tuple[list[int]]:
    """Perform diversity-aware and progress-aware adaptive mutation."""
    progress = generation / max(1, max_generations - 1)
    dynamic_budget = max(
        1,
        int((0.01 + (1.0 - progress) * 0.08 + (1.0 - diversity) * 0.10) * len(individual)),
    )
    if max_mutation_budget > 0:
        dynamic_budget = min(dynamic_budget, max(1, int(max_mutation_budget)))

    for _ in range(dynamic_budget):
        position = rng.randrange(len(individual))
        if rng.random() < heuristic_mutation_rate:
            use_energy_guidance = False
            if energy_node_order and energy_guided_rate > 0.0:
                late_factor = max(0.0, (progress - 0.55) / 0.45)
                if late_factor > 0.0 and rng.random() < energy_guided_rate * late_factor:
                    use_energy_guidance = True

            if use_energy_guidance:
                top_k_energy = max(1, min(2, len(energy_node_order)))
                individual[position] = int(energy_node_order[rng.randrange(top_k_energy)])
            else:
                ranked = task_rankings[position]
                top_k = max(1, min(3, len(ranked)))
                individual[position] = ranked[rng.randrange(top_k)]
        else:
            individual[position] = rng.randrange(num_nodes)

    dynamic_gene_prob = base_gene_mutation_probability * (0.20 + 0.30 * (1.0 - diversity))
    for idx in range(len(individual)):
        if rng.random() < dynamic_gene_prob:
            individual[idx] = rng.randrange(num_nodes)

    return (individual,)


def greedy_local_repair(
    individual: list[int],
    evaluate_fn: Callable[[Sequence[int]], Tuple[float, ...]],
    task_rankings: Sequence[Sequence[int]],
    task_priority: Sequence[int],
    attempts: int,
) -> tuple[list[int], Tuple[float, ...], bool]:
    """Greedy local search that tries high-priority tasks on better ranked nodes."""
    best_solution = list(individual)
    best_fitness = evaluate_fn(best_solution)
    improved = False

    for task_idx in task_priority[:attempts]:
        for candidate_node in task_rankings[task_idx][:2]:
            if candidate_node == best_solution[task_idx]:
                continue
            candidate = list(best_solution)
            candidate[task_idx] = candidate_node
            candidate_fitness = evaluate_fn(candidate)

            if dominates(candidate_fitness, best_fitness) or (
                scalar_fitness(candidate_fitness) < scalar_fitness(best_fitness)
            ):
                best_solution = candidate
                best_fitness = candidate_fitness
                improved = True

    return best_solution, best_fitness, improved
