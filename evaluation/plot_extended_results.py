from __future__ import annotations

"""Create summary figures for extended ADARE baseline comparisons."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CORE_METRICS = {"hv", "igd", "spacing", "epsilon", "coverage"}
SCORE_METRICS = {"hv", "igd", "spacing", "epsilon"}
HIGHER_IS_BETTER = {"hv"}
ALGORITHM_COLORS = {
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


def row_scale(row: dict[str, str], fallback: str | None = None) -> str:
    if "scale" in row and row["scale"]:
        return row["scale"]
    if fallback:
        return fallback
    benchmark = row["benchmark"]
    if "3000" in benchmark:
        return "3000"
    if "1000" in benchmark:
        return "1000"
    return "small"


def algorithm_metric_values(rows: list[dict[str, str]], fallback_scale: str | None = None) -> dict[tuple[str, str, str], dict[str, float]]:
    values: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        metric = row["metric"]
        if metric not in SCORE_METRICS:
            continue
        key = (row_scale(row, fallback_scale), row["benchmark"], metric)
        values[key]["ADARE"] = float(row["adare_mean"])
        values[key][row["baseline"]] = float(row["baseline_mean"])
    return values


def normalized_algorithm_scores(rows: list[dict[str, str]], fallback_scale: str | None = None) -> dict[tuple[str, str], float]:
    score_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (scale, _benchmark, metric), algo_values in algorithm_metric_values(rows, fallback_scale).items():
        finite = {algo: value for algo, value in algo_values.items() if np.isfinite(value)}
        if not finite:
            continue
        if metric in HIGHER_IS_BETTER:
            best = max(finite.values())
            if best <= 0:
                continue
            for algo, value in finite.items():
                score_values[(scale, algo)].append(100.0 * value / best)
        else:
            positive = [value for value in finite.values() if value > 0]
            if not positive:
                continue
            best = min(positive)
            for algo, value in finite.items():
                if value > 0:
                    score_values[(scale, algo)].append(100.0 * best / value)

    return {key: float(np.mean(values)) for key, values in score_values.items() if values}


def plot_algorithm_score(
    rows: list[dict[str, str]],
    output_path: Path,
    title: str,
    scales: list[str],
    baselines: list[str],
    fallback_scale: str | None = None,
) -> None:
    scores = normalized_algorithm_scores(rows, fallback_scale)
    algorithms = ["ADARE", *baselines]
    x = np.arange(len(scales))
    width = min(0.72 / max(1, len(algorithms)), 0.14)
    center = (len(algorithms) - 1) / 2

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for offset, algorithm in enumerate(algorithms):
        values = [scores.get((scale, algorithm), np.nan) for scale in scales]
        style = {"color": ALGORITHM_COLORS.get(algorithm), "edgecolor": "#111111", "linewidth": 0.8}
        if algorithm == "ADARE":
            style["linewidth"] = 1.4
        ax.bar(x + (offset - center) * width, values, width=width, label=algorithm, **style)

    ax.set_xticks(x)
    ax.set_xticklabels(["Small", "1000 tasks", "3000 tasks"] if scales == ["small", "1000", "3000"] else [f"{scale} tasks" for scale in scales], fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Normalized front-quality score (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot extended comparison summary.")
    parser.add_argument("--input", type=Path, default=Path("results/extended/extended_global_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("Figures"))
    args = parser.parse_args()

    rows = [row for row in load_rows(args.input) if row["metric"] in CORE_METRICS]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    wins: dict[tuple[str, str], int] = defaultdict(int)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        key = (row["scale"], row["baseline"])
        gain = float(row["gain_percent"])
        grouped[key].append(gain)
        counts[key] += 1
        if gain > 0.0:
            wins[key] += 1

    scales = ["small", "1000", "3000"]
    baselines = [
        "NSGA-III",
        "NSGA-II",
        "MOEA/D",
        "QL-NSGA-III",
        "OVEA-style",
        "QMOEA/D-AWA-style",
    ]
    x = np.arange(len(scales))
    width = 0.13

    plot_algorithm_score(
        rows=[row for row in load_rows(args.input) if row["metric"] in SCORE_METRICS],
        output_path=args.output_dir / "extended_algorithm_core_score_by_scale.png",
        title="Explicit Front-Quality Comparison: ADARE vs Baselines",
        scales=scales,
        baselines=baselines,
    )

    r20_paths = [
        (Path("results/extended/extended_1000_r20_summary.csv"), "1000"),
        (Path("results/extended/extended_3000_r20_summary.csv"), "3000"),
    ]
    r20_rows: list[dict[str, str]] = []
    for path, scale in r20_paths:
        if not path.exists():
            continue
        for row in load_rows(path):
            row = dict(row)
            row["scale"] = scale
            r20_rows.append(row)
    if r20_rows:
        plot_algorithm_score(
            rows=[row for row in r20_rows if row["metric"] in SCORE_METRICS],
            output_path=args.output_dir / "large_r20_algorithm_core_score.png",
            title="Explicit 20-Run Large-Scale Comparison: ADARE vs Baselines",
            scales=["1000", "3000"],
            baselines=["NSGA-III", "QL-NSGA-III", "OVEA-style", "QMOEA/D-AWA-style"],
        )


if __name__ == "__main__":
    main()
