from __future__ import annotations

"""Plotting helpers used to export result figures to disk."""

from pathlib import Path
from typing import Dict, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ensure_parent(path: Path) -> None:
    """Create output parent directory if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_pareto_projections(
    fronts: Dict[str, np.ndarray],
    objective_names: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    """Save 2D Pareto projections for selected objective pairs."""
    _ensure_parent(output_path)
    pairs = [(0, 1), (0, 2), (0, 3), (2, 3)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for ax, (i, j) in zip(axes, pairs):
        for label, front in fronts.items():
            ax.scatter(front[:, i], front[:, j], s=18, alpha=0.7, label=label)
        ax.set_xlabel(objective_names[i])
        ax.set_ylabel(objective_names[j])
        ax.grid(alpha=0.3, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(fronts)))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_convergence(
    histories: Dict[str, np.ndarray],
    objective_names: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    """Save convergence curves with mean and standard-deviation bands."""
    _ensure_parent(output_path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for obj_idx, obj_name in enumerate(objective_names):
        ax = axes[obj_idx]
        for label, data in histories.items():
            # data shape: (runs, generations+1, objectives)
            mean_curve = np.mean(data[:, :, obj_idx], axis=0)
            std_curve = np.std(data[:, :, obj_idx], axis=0)
            x = np.arange(mean_curve.shape[0])
            ax.plot(x, mean_curve, label=label)
            ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

        ax.set_title(obj_name)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best value")
        ax.grid(alpha=0.3, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(histories)))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_metric_boxplots(
    metric_name: str,
    adare_values: Sequence[float],
    nsga_values: Sequence[float],
    output_path: Path,
    title: str,
) -> None:
    """Save side-by-side metric distribution boxplots for both algorithms."""
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([adare_values, nsga_values], labels=["ADARE", "NSGA-III"], showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(metric_name)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
