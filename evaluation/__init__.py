"""Evaluation layer exports for metrics and visualization helpers."""

from .metrics import (
    aggregate_metric,
    coverage_metric,
    dominates,
    filter_non_dominated,
    metric_gain_percent,
    paired_wilcoxon,
    quality_indicators,
)
from .visualization import plot_convergence, plot_metric_boxplots, plot_pareto_projections

__all__ = [
    "aggregate_metric",
    "coverage_metric",
    "dominates",
    "filter_non_dominated",
    "metric_gain_percent",
    "paired_wilcoxon",
    "quality_indicators",
    "plot_convergence",
    "plot_metric_boxplots",
    "plot_pareto_projections",
]
