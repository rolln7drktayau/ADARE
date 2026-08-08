from __future__ import annotations

"""Paired sensitivity analysis for ADARE reward weights and clipping."""

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import AdareAlgorithm
from problem import build_nodes, load_environments
from run_extended_comparison import ALGORITHMS, ALGORITHM_COLORS, load_json, run_benchmark, write_csv


VARIANTS: dict[str, dict[str, Any]] = {
    "ADARE": {"reward_weights": [0.45, 0.35, 0.15, 0.05], "reward_clip": 1.0},
    "Equal-weights": {"reward_weights": [0.25, 0.25, 0.25, 0.25], "reward_clip": 1.0},
    "Dominance-heavy": {"reward_weights": [0.60, 0.20, 0.15, 0.05], "reward_clip": 1.0},
    "Improvement-heavy": {"reward_weights": [0.30, 0.50, 0.15, 0.05], "reward_clip": 1.0},
    "Tight-clipping": {"reward_weights": [0.45, 0.35, 0.15, 0.05], "reward_clip": 0.5},
    "No-clipping": {"reward_weights": [0.45, 0.35, 0.15, 0.05], "reward_clip": 0.0},
    "Low-alpha": {"q_learning_alpha": 0.10},
    "High-alpha": {"q_learning_alpha": 0.40},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADARE reward/operator sensitivity with paired seeds")
    parser.add_argument("--benchmarks", nargs="+", default=["Montage_25", "CyberShake_30", "Epigenomics_24"])
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--generations", type=int, default=70)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260219)
    parser.add_argument("--output-dir", default="output/major_revision/reward_sensitivity_r20")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs <= 0 or args.generations <= 0 or args.population_size <= 1:
        raise ValueError("runs/generations must be > 0 and population-size must be > 1")
    config_path = "config/adare_config.json"
    palette = ["#D55E00", "#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#000000", "#999999"]
    for (label, _), color in zip(VARIANTS.items(), palette):
        ALGORITHMS[label] = (AdareAlgorithm, config_path)
        ALGORITHM_COLORS[label] = color

    shared = dict(load_json(ROOT / "config" / "main_config.json")["shared_parameters"])
    shared["generations"] = args.generations
    shared["population_size"] = args.population_size
    nodes = build_nodes(load_environments())
    summaries: list[dict[str, Any]] = []
    for benchmark_idx, benchmark in enumerate(args.benchmarks):
        overall_percent = 100.0 * benchmark_idx / len(args.benchmarks)
        print(
            f"Reward sensitivity: benchmark {benchmark_idx + 1}/{len(args.benchmarks)} "
            f"[{overall_percent:6.2f}%] | {benchmark}",
            flush=True,
        )
        summary, _, _ = run_benchmark(
            benchmark=benchmark,
            selected_algorithms=list(VARIANTS),
            shared_config=shared,
            nodes=nodes,
            runs=args.runs,
            base_seed=args.seed + 1000 * benchmark_idx,
            output_root=Path(args.output_dir),
            config_overrides=VARIANTS,
        )
        summaries.extend(summary)
    print("Reward sensitivity: [100.00%] complete", flush=True)
    write_csv(Path(args.output_dir) / "summary" / "reward_sensitivity_summary.csv", summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
