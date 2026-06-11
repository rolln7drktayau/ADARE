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


def plot_pareto_from_rows(front_rows: list[dict[str, str]], output_path: Path) -> None:
    benchmark_fronts: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for row in front_rows:
        benchmark_fronts[row["benchmark"]][row["algorithm"]].append(
            [float(row[name]) for name in ("makespan", "latency", "cost", "energy")]
        )

    benchmarks = sorted(benchmark_fronts)
    pairs = [(0, 1), (2, 3)]
    objective_labels = ["Makespan", "Latency", "Cost", "Energy"]
    fig, axes = plt.subplots(len(benchmarks), len(pairs), figsize=(15, 4.2 * len(benchmarks)))
    if len(benchmarks) == 1:
        axes = np.asarray([axes])
    for row_idx, benchmark in enumerate(benchmarks):
        for col_idx, (i, j) in enumerate(pairs):
            ax = axes[row_idx, col_idx]
            for algo in ALGORITHMS:
                points = benchmark_fronts[benchmark].get(algo)
                if not points:
                    continue
                front = np.asarray(points, dtype=float)
                ax.scatter(
                    front[:, i],
                    front[:, j],
                    s=28 if algo == "ADARE" else 18,
                    alpha=0.85 if algo == "ADARE" else 0.58,
                    label=algo,
                    color=COLORS[algo],
                    edgecolors="#111111" if algo == "ADARE" else "none",
                    linewidths=0.35 if algo == "ADARE" else 0.0,
                )
            ax.set_title(f"{benchmark}: {objective_labels[i]} vs {objective_labels[j]}", fontsize=11)
            ax.set_xlabel(objective_labels[i], fontsize=10)
            ax.set_ylabel(objective_labels[j], fontsize=10)
            ax.grid(alpha=0.25, linestyle="--")
            ax.tick_params(axis="both", labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Representative Pareto Fronts: ADARE vs All Baselines", fontsize=15)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_convergence_from_rows(history_rows: list[dict[str, str]], output_path: Path) -> None:
    grouped: dict[str, dict[str, dict[int, list[list[float]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in history_rows:
        grouped[row["benchmark"]][row["algorithm"]][int(row["run"])].append(
            [
                float(row["makespan_best"]),
                float(row["latency_best"]),
                float(row["cost_best"]),
                float(row["energy_best"]),
            ]
        )

    benchmarks = sorted(grouped)
    objective_labels = ["Makespan", "Latency", "Cost", "Energy"]
    fig, axes = plt.subplots(len(benchmarks), len(objective_labels), figsize=(18, 3.4 * len(benchmarks)))
    if len(benchmarks) == 1:
        axes = np.asarray([axes])
    for row_idx, benchmark in enumerate(benchmarks):
        for obj_idx, obj_name in enumerate(objective_labels):
            ax = axes[row_idx, obj_idx]
            for algo in ALGORITHMS:
                runs = grouped[benchmark].get(algo)
                if not runs:
                    continue
                histories = np.asarray([runs[run_id] for run_id in sorted(runs)], dtype=float)
                mean_curve = np.nanmean(histories[:, :, obj_idx], axis=0)
                std_curve = np.nanstd(histories[:, :, obj_idx], axis=0)
                x = np.arange(mean_curve.shape[0])
                ax.plot(
                    x,
                    mean_curve,
                    label=algo,
                    color=COLORS[algo],
                    linewidth=2.4 if algo == "ADARE" else 1.5,
                    alpha=1.0 if algo == "ADARE" else 0.86,
                )
                ax.fill_between(
                    x,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color=COLORS[algo],
                    alpha=0.16 if algo == "ADARE" else 0.08,
                )
            ax.set_title(f"{benchmark} - {obj_name}", fontsize=10)
            ax.set_xlabel("Generation", fontsize=9)
            ax.set_ylabel("Best value", fontsize=9)
            ax.grid(alpha=0.25, linestyle="--")
            ax.tick_params(axis="both", labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Convergence Traces: ADARE vs All Baselines", fontsize=15)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create explicit multi-algorithm comparison figures.")
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--fronts", type=Path, default=Path("results/extended/extended_small_r5_representative_fronts.csv"))
    parser.add_argument("--histories", type=Path, default=Path("results/extended/extended_small_r5_histories.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("Figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = load_rows(args.run_metrics)
    summary_rows = load_rows(args.summary)
    plot_core_boxplots(run_rows, args.output_dir / "multialgo_small_core_metric_boxplots.png")
    plot_normalized_scores(run_rows, args.output_dir / "multialgo_small_normalized_scores.png")
    plot_summary_gain_heatmap(summary_rows, args.output_dir / "multialgo_small_adare_gain_heatmap.png")
    if args.fronts.exists():
        plot_pareto_from_rows(load_rows(args.fronts), args.output_dir / "multialgo_pareto_small_workflows.png")
    if args.histories.exists():
        plot_convergence_from_rows(load_rows(args.histories), args.output_dir / "multialgo_convergence_small_workflows.png")


if __name__ == "__main__":
    main()
