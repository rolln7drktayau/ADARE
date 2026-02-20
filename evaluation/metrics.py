from __future__ import annotations

"""Quality metrics used to compare ADARE and NSGA-III fronts."""

from typing import Dict, Sequence

import numpy as np
from pymoo.indicators.hv import Hypervolume  # type: ignore
from pymoo.indicators.igd import IGD  # type: ignore
from scipy.stats import wilcoxon  # type: ignore


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Return True if point `a` dominates point `b` in minimization."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return bool(np.all(a_arr <= b_arr) and np.any(a_arr < b_arr))


def filter_non_dominated(front: np.ndarray) -> np.ndarray:
    """Extract unique non-dominated points from a candidate front."""
    points = np.asarray(front, dtype=float)
    if points.size == 0:
        return points.reshape(0, 0)

    keep = np.ones(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        if not keep[i]:
            continue
        for j in range(points.shape[0]):
            if i == j or not keep[j]:
                continue
            if dominates(points[j], points[i]):
                keep[i] = False
                break
    nd = points[keep]
    return np.unique(nd, axis=0)


def spacing_metric(front: np.ndarray) -> float:
    """Compute Schott spacing (lower is better spread regularity)."""
    points = np.asarray(front, dtype=float)
    if len(points) <= 1:
        return 0.0
    distances = np.abs(points[:, None, :] - points[None, :, :]).sum(axis=2)
    distances += np.eye(len(points)) * 1e12
    nearest = distances.min(axis=1)
    return float(np.std(nearest))


def epsilon_indicator(approx_front: np.ndarray, reference_front: np.ndarray) -> float:
    """Compute additive epsilon indicator (lower is better)."""
    approx = np.asarray(approx_front, dtype=float)
    reference = np.asarray(reference_front, dtype=float)
    if len(approx) == 0 or len(reference) == 0:
        return float("nan")

    eps = -np.inf
    for ref_point in reference:
        eps_for_ref = np.inf
        for approx_point in approx:
            eps_for_ref = min(eps_for_ref, np.max(approx_point - ref_point))
        eps = max(eps, eps_for_ref)
    return float(eps)


def coverage_metric(front_a: np.ndarray, front_b: np.ndarray) -> float:
    """Compute C(A,B): fraction of B points dominated by at least one point in A."""
    a = np.asarray(front_a, dtype=float)
    b = np.asarray(front_b, dtype=float)
    if len(b) == 0:
        return 0.0

    dominated = 0
    for point_b in b:
        if any(dominates(point_a, point_b) for point_a in a):
            dominated += 1
    return float(dominated / len(b))


def _normalize_front(front: np.ndarray, mins: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """Normalize a front using reference min/max bounds."""
    normalized = (front - mins) / spans
    return np.clip(normalized, 0.0, 1.2)


def quality_indicators(front: np.ndarray, reference_front: np.ndarray) -> Dict[str, float]:
    """Compute HV, IGD, spacing, and epsilon against a reference front."""
    nd_front = filter_non_dominated(front)
    nd_reference = filter_non_dominated(reference_front)
    if len(nd_front) == 0 or len(nd_reference) == 0:
        return {"hv": float("nan"), "igd": float("nan"), "spacing": float("nan"), "epsilon": float("nan")}

    mins = np.min(nd_reference, axis=0)
    maxs = np.max(nd_reference, axis=0)
    spans = np.where((maxs - mins) < 1e-12, 1.0, maxs - mins)

    norm_front = _normalize_front(nd_front, mins, spans)
    norm_reference = _normalize_front(nd_reference, mins, spans)

    hv = Hypervolume(ref_point=np.full(norm_front.shape[1], 1.1)).do(norm_front)
    igd = IGD(norm_reference).do(norm_front)
    spacing = spacing_metric(norm_front)
    epsilon = epsilon_indicator(norm_front, norm_reference)

    return {
        "hv": float(hv),
        "igd": float(igd),
        "spacing": float(spacing),
        "epsilon": float(epsilon),
    }


def metric_gain_percent(adare_value: float, nsga_value: float, direction: str) -> float:
    """Convert metric comparison into a signed percentage gain for ADARE."""
    if abs(nsga_value) < 1e-12:
        return 0.0
    if direction == "max":
        return float((adare_value - nsga_value) / abs(nsga_value) * 100.0)
    if direction == "min":
        return float((nsga_value - adare_value) / abs(nsga_value) * 100.0)
    raise ValueError(f"Direction inconnue: {direction}")


def aggregate_metric(values: Sequence[float]) -> Dict[str, float]:
    """Return mean, sample std, and median for one metric."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.nanmedian(arr)),
    }


def paired_wilcoxon(adare_values: Sequence[float], nsga_values: Sequence[float], direction: str) -> float:
    """Run paired one-sided Wilcoxon test and return p-value."""
    a = np.asarray(adare_values, dtype=float)
    b = np.asarray(nsga_values, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return float("nan")

    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) != len(b) or len(a) < 2:
        return float("nan")

    if np.allclose(a, b):
        return 1.0

    alternative = "greater" if direction == "max" else "less"
    try:
        result = wilcoxon(a, b, alternative=alternative, zero_method="wilcox", method="auto")
    except ValueError:
        return float("nan")
    return float(result.pvalue)
