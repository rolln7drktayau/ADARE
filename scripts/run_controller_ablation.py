from __future__ import annotations

"""Controlled controller ablation requested in the SwEvo major review.

All variants keep mutation, archive, survival, evaluator, population, budget,
initial population, and stochastic seeds identical. Only controller context and
reward design change.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import AdareAlgorithm
from problem import build_nodes, load_environments
from run_extended_comparison import (
    ALGORITHMS,
    ALGORITHM_COLORS,
    load_json,
    plot_multialgo_convergence,
    plot_multialgo_pareto,
    run_benchmark,
    write_csv,
)


VARIANTS: dict[str, dict[str, Any]] = {
    "ADARE": {"controller_mode": "contextual", "reward_mode": "proposed"},
    "Static": {"controller_mode": "static", "reward_mode": "dominance_only", "static_operator_index": 1},
    "Global-UCB": {"controller_mode": "global", "reward_mode": "dominance_only"},
    "Contextual-UCB": {"controller_mode": "contextual", "reward_mode": "dominance_only"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled static/global/contextual/reward ablation")
    parser.add_argument("--benchmarks", nargs="+", default=["Montage_25", "CyberShake_30", "Epigenomics_24"])
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--generations", type=int, default=70)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260219)
    parser.add_argument("--output-dir", default="output/major_revision/controller_ablation_r20")
    parser.add_argument("--figure-dir", default="output/major_revision/figures/controller_ablation_r20")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs <= 0 or args.generations <= 0 or args.population_size <= 1:
        raise ValueError("runs/generations must be > 0 and population-size must be > 1")

    config_path = "config/adare_config.json"
    for label in VARIANTS:
        ALGORITHMS[label] = (AdareAlgorithm, config_path)
    ALGORITHM_COLORS.update(
        {"Static": "#0072B2", "Global-UCB": "#009E73", "Contextual-UCB": "#E69F00"}
    )

    shared = dict(load_json(ROOT / "config" / "main_config.json")["shared_parameters"])
    shared["generations"] = args.generations
    shared["population_size"] = args.population_size
    nodes = build_nodes(load_environments())
    labels = list(VARIANTS)
    all_summary: list[dict[str, Any]] = []
    all_fronts = {}
    all_histories = {}
    for benchmark_idx, benchmark in enumerate(args.benchmarks):
        overall_percent = 100.0 * benchmark_idx / len(args.benchmarks)
        print(
            f"Controller ablation: benchmark {benchmark_idx + 1}/{len(args.benchmarks)} "
            f"[{overall_percent:6.2f}%] | {benchmark}",
            flush=True,
        )
        summary, fronts, histories = run_benchmark(
            benchmark=benchmark,
            selected_algorithms=labels,
            shared_config=shared,
            nodes=nodes,
            runs=args.runs,
            base_seed=args.seed + 1000 * benchmark_idx,
            output_root=Path(args.output_dir),
            config_overrides=VARIANTS,
        )
        all_summary.extend(summary)
        all_fronts[benchmark] = fronts
        all_histories[benchmark] = histories
    print("Controller ablation: [100.00%] complete", flush=True)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "summary" / "controller_ablation_summary.csv", all_summary)
    figure_dir = Path(args.figure_dir)
    plot_multialgo_pareto(all_fronts, labels, figure_dir / "controller_ablation_pareto.png")
    plot_multialgo_convergence(all_histories, labels, figure_dir / "controller_ablation_convergence.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
