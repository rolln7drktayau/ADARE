from __future__ import annotations

"""Create explicit multi-algorithm figures from extended run metrics."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ALGORITHMS = [
    "ADARE",
    "NSGA-III",
    "NSGA-II",
    "MOEA/D",
    "QL-NSGA-III",
    "OVEA-style",
    "QMOEA/D-AWA-style",
]
CORE_METRICS = ["hv", "igd", "spacing", "epsilon"]
OBJECTIVE_METRICS = ["makespan_best", "latency_best", "cost_best", "energy_best"]
LOWER_IS_BETTER = {"igd", "spacing", "epsilon", *OBJECTIVE_METRICS, "time"}
COLORS = {
    "ADARE": "#D55E00",
    "NSGA-III": "#0072B2",
    "NSGA-II": "#009E73",
    "MOEA/D": "#CC79A7",
    "QL-NSGA-III": "#E69F00",
    "OVEA-style": "#56B4E9",
    "QMOEA/D-AWA-style": "#000000",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite(values: list[float]) -> list[float]:
    return [value for value in values if np.isfinite(value)]


def metric_values(rows: list[dict[str, str]], metric: str) -> list[list[float]]:
    by_algo: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        try:
            by_algo[row["algorithm"]].append(float(row[metric]))
        except (KeyError, ValueError):
            continue
    return [finite(by_algo[algo]) for algo in ALGORITHMS]


def style_boxplot(boxplot: dict[str, list], algorithms: list[str]) -> None:
    for patch, algo in zip(boxplot["boxes"], algorithms):
        patch.set_facecolor(COLORS[algo])
        patch.set_alpha(0.78)
        patch.set_edgecolor("#111111")
    for median in boxplot["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.4)
    for whisker in boxplot["whiskers"]:
        whisker.set_color("#333333")
    for cap in boxplot["caps"]:
        cap.set_color("#333333")


def plot_core_boxplots(rows: list[dict[str, str]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.ravel()
    labels = ["HV", "IGD", "Spacing", "Epsilon"]
    for ax, metric, label in zip(axes, CORE_METRICS, labels):
        values = metric_values(rows, metric)
        boxplot = ax.boxplot(values, patch_artist=True, showmeans=True, tick_labels=ALGORITHMS)
        style_boxplot(boxplot, ALGORITHMS)
        ax.set_title(label, fontsize=13)
        ax.set_ylabel("Value", fontsize=11)
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.suptitle("Small Workflow Multi-Algorithm Quality Distributions", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def normalized_scores(rows: list[dict[str, str]], metrics: list[str]) -> dict[str, float]:
    grouped: dict[tuple[str, str, str], float] = {}
    for row in rows:
        benchmark = row["benchmark"]
        run = row["run"]
        algo = row["algorithm"]
        for metric in metrics:
            try:
                value = float(row[metric])
            except (KeyError, ValueError):
                continue
            if np.isfinite(value):
                grouped[(benchmark, run, algo, metric)] = value

    score_values: dict[str, list[float]] = defaultdict(list)
    keys = sorted({(b, r, m) for b, r, _a, m in grouped})
    for benchmark, run, metric in keys:
        values = {
            algo: grouped[(benchmark, run, algo, metric)]
            for algo in ALGORITHMS
            if (benchmark, run, algo, metric) in grouped
        }
        if not values:
            continue
        if metric in LOWER_IS_BETTER:
            positives = [value for value in values.values() if value > 0]
            if not positives:
                continue
            best = min(positives)
            for algo, value in values.items():
                if value > 0:
                    score_values[algo].append(100.0 * best / value)
        else:
            best = max(values.values())
            if best <= 0:
                continue
            for algo, value in values.items():
                score_values[algo].append(100.0 * value / best)
    return {algo: float(np.mean(score_values[algo])) for algo in ALGORITHMS if score_values[algo]}


def plot_normalized_scores(rows: list[dict[str, str]], output_path: Path) -> None:
    groups = {
        "Front quality": CORE_METRICS,
        "Objective extremes": OBJECTIVE_METRICS,
        "Runtime": ["time"],
    }
    x = np.arange(len(groups))
    width = 0.1
    center = (len(ALGORITHMS) - 1) / 2
    fig, ax = plt.subplots(figsize=(14, 6.8))
    for offset, algo in enumerate(ALGORITHMS):
        values = [normalized_scores(rows, metrics).get(algo, np.nan) for metrics in groups.values()]
        ax.bar(
            x + (offset - center) * width,
            values,
            width=width,
            label=algo,
            color=COLORS[algo],
            edgecolor="#111111",
            linewidth=1.2 if algo == "ADARE" else 0.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(groups), fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Normalized score (%)", fontsize=12)
    ax.set_title("Explicit Small Workflow Comparison: ADARE vs All Baselines", fontsize=14)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_summary_gain_heatmap(summary_rows: list[dict[str, str]], output_path: Path) -> None:
    benchmarks = sorted({row["benchmark"] for row in summary_rows})
    baselines = [algo for algo in ALGORITHMS if algo != "ADARE"]
    gains: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in summary_rows:
        if row["metric"] not in {"hv", "igd", "spacing", "epsilon", "coverage"}:
            continue
        gains[(row["benchmark"], row["baseline"])].append(float(row["gain_percent"]))

    matrix = np.array(
        [[np.mean(gains.get((benchmark, baseline), [np.nan])) for baseline in baselines] for benchmark in benchmarks]
    )
    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-40, vmax=80, aspect="auto")
    ax.set_xticks(np.arange(len(baselines)))
    ax.set_xticklabels(baselines, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(benchmarks)))
    ax.set_yticklabels(benchmarks, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color="#111111")
    ax.set_title("Mean ADARE Core-Metric Gain by Workflow and Baseline (%)", fontsize=14)
    fig.colorbar(im, ax=ax, label="ADARE gain (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create explicit multi-algorithm comparison figures.")
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("Figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = load_rows(args.run_metrics)
    summary_rows = load_rows(args.summary)
    plot_core_boxplots(run_rows, args.output_dir / "multialgo_small_core_metric_boxplots.png")
    plot_normalized_scores(run_rows, args.output_dir / "multialgo_small_normalized_scores.png")
    plot_summary_gain_heatmap(summary_rows, args.output_dir / "multialgo_small_adare_gain_heatmap.png")


if __name__ == "__main__":
    main()
