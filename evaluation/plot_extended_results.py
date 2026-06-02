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


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    baselines = ["NSGA-III", "NSGA-II", "MOEA/D", "QL-NSGA-III"]
    x = np.arange(len(scales))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, baseline in enumerate(baselines):
        values = [np.mean(grouped.get((scale, baseline), [np.nan])) for scale in scales]
        ax.bar(x + (offset - 1.5) * width, values, width=width, label=baseline)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Small", "1000 tasks", "3000 tasks"], fontsize=11)
    ax.set_ylabel("Mean ADARE core-quality gain (%)", fontsize=12)
    ax.set_title("Extended ADARE Comparison Across Workflow Scales", fontsize=14)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output_dir / "extended_core_gain_by_scale.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, baseline in enumerate(baselines):
        values = [
            100.0 * wins.get((scale, baseline), 0) / max(1, counts.get((scale, baseline), 0))
            for scale in scales
        ]
        ax.bar(x + (offset - 1.5) * width, values, width=width, label=baseline)
    ax.set_xticks(x)
    ax.set_xticklabels(["Small", "1000 tasks", "3000 tasks"], fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("ADARE core-quality win rate (%)", fontsize=12)
    ax.set_title("ADARE Core-Metric Win Rate vs Extended Baselines", fontsize=14)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output_dir / "extended_core_win_rate_by_scale.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
