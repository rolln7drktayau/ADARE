from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from main import CORE_QUALITY_METRICS, QUALITY_METRICS, load_json, run_benchmark
from problem import build_nodes, load_environments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run V1-V5 ablation and export compact numeric summary")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--output-dir", default="output/ablation_full")
    return p.parse_args()


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def feature_flags(v: str) -> tuple[str, str, str]:
    if v == "V1":
        return "No", "No", "No"
    if v == "V2":
        return "Yes", "No", "No"
    if v == "V3":
        return "Yes", "Yes", "No"
    if v == "V4":
        return "Yes", "Yes", "Yes (partial)"
    return "Yes", "Yes", "Yes"


def build_variants(base_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    # V2: UCB crossover only
    v2 = copy.deepcopy(base_cfg)
    v2["use_adaptive_mutation"] = False
    v2["heuristic_mutation_rate"] = 0.0
    v2["local_search_probability"] = 0.0
    v2["enable_archive"] = False
    v2["archive_size"] = 0
    v2["archive_injection_rate"] = 0.0
    v2["append_speed_latency_final"] = False
    v2["refine_speed_latency_final"] = False

    # V3: V2 + density-aware mutation
    v3 = copy.deepcopy(base_cfg)
    v3["use_adaptive_mutation"] = True
    v3["enable_archive"] = False
    v3["archive_size"] = 0
    v3["archive_injection_rate"] = 0.0
    v3["append_speed_latency_final"] = False
    v3["refine_speed_latency_final"] = False

    # V4: V3 + partial archive guidance
    v4 = copy.deepcopy(base_cfg)
    v4["use_adaptive_mutation"] = True
    v4["enable_archive"] = True
    v4["archive_size"] = min(40, int(base_cfg.get("archive_size", 80)))
    v4["archive_injection_rate"] = min(0.01, float(base_cfg.get("archive_injection_rate", 0.02)))
    v4["append_speed_latency_final"] = False
    v4["refine_speed_latency_final"] = False

    # V5: full ADARE
    v5 = copy.deepcopy(base_cfg)
    v5["use_adaptive_mutation"] = True
    v5["enable_archive"] = True

    return {"V2": v2, "V3": v3, "V4": v4, "V5": v5}


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent

    main_cfg = load_json(root / "config" / "main_config.json")
    adare_cfg = load_json(root / "config" / "adare_config.json")
    nsga3_cfg = load_json(root / "config" / "nsga3_config.json")

    runs = int(args.runs)
    if runs <= 0:
        raise ValueError("runs must be > 0")

    seed = int(args.seed if args.seed is not None else main_cfg["general_parameters"]["base_seed"])
    objective_names = list(main_cfg["objectives"])
    benchmarks = list(main_cfg["benchmarks"])
    shared_cfg = dict(main_cfg["shared_parameters"])

    output_root = root / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    nodes = build_nodes(load_environments())
    variants = build_variants(adare_cfg)

    per_benchmark_rows: List[Dict[str, Any]] = []

    for vname, vcfg in variants.items():
        print(f"\n=== {vname} ===")
        for bidx, benchmark in enumerate(benchmarks):
            res = run_benchmark(
                benchmark=benchmark,
                objective_names=objective_names,
                shared_config=shared_cfg,
                adare_config=vcfg,
                nsga3_config=nsga3_cfg,
                nodes=nodes,
                runs=runs,
                base_seed=seed + bidx * 1000,
                output_root=output_root / vname,
                save_plots=not args.no_plots,
            )
            srows = res["summary_rows"]
            core = [r for r in srows if r["metric"] in CORE_QUALITY_METRICS]
            objectives = [r for r in srows if str(r["metric"]).endswith("_best")]
            runtime = next(r for r in srows if r["metric"] == "time")
            per_benchmark_rows.append(
                {
                    "variant": vname,
                    "benchmark": benchmark,
                    "core_quality_wins": int(res["core_quality_wins"]),
                    "quality_wins": int(res["quality_wins"]),
                    "mean_core_quality_gain_percent": float(np.mean([r["gain_percent"] for r in core])),
                    "mean_objective_gain_percent": float(np.mean([r["gain_percent"] for r in objectives])),
                    "runtime_gain_percent": float(runtime["gain_percent"]),
                }
            )

    # aggregate across benchmarks
    agg_rows: List[Dict[str, Any]] = []
    agg_rows.append(
        {
            "variant": "V1",
            "ucb": "No",
            "density_mutation": "No",
            "archive": "No",
            "mean_core_quality_gain_percent": 0.0,
            "mean_objective_gain_percent": 0.0,
            "mean_runtime_gain_percent": 0.0,
            "core_wins_total": 0,
        }
    )

    for v in ["V2", "V3", "V4", "V5"]:
        rows = [r for r in per_benchmark_rows if r["variant"] == v]
        ucb, dens, arch = feature_flags(v)
        agg_rows.append(
            {
                "variant": v,
                "ucb": ucb,
                "density_mutation": dens,
                "archive": arch,
                "mean_core_quality_gain_percent": float(np.mean([r["mean_core_quality_gain_percent"] for r in rows])),
                "mean_objective_gain_percent": float(np.mean([r["mean_objective_gain_percent"] for r in rows])),
                "mean_runtime_gain_percent": float(np.mean([r["runtime_gain_percent"] for r in rows])),
                "core_wins_total": int(sum(r["core_quality_wins"] for r in rows)),
            }
        )

    write_csv(output_root / "ablation_v1_v5_per_benchmark.csv", per_benchmark_rows)
    write_csv(output_root / "ablation_v1_v5_global.csv", agg_rows)

    total_core_slots = len(benchmarks) * len(CORE_QUALITY_METRICS)
    lines = [
        r"\begin{table}[t]",
        r"\caption{Numerical Ablation Results (V1--V5, 20 paired runs per benchmark)}",
        r"\label{tab:ablation_numeric}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lccc ccc c@{}}",
        r"\toprule",
        r"Variant & UCB & Density mut. & Archive & Core gain (\%) & Obj. gain (\%) & Runtime gain (\%) & Core wins \\",
        r"\midrule",
    ]
    for r in agg_rows:
        if r["variant"] == "V1":
            lines.append(r"V1 (NSGA-III) & No & No & No & 0.00 & 0.00 & 0.00 & -- \\")
        else:
            lines.append(
                f"{r['variant']} & {r['ucb']} & {r['density_mutation']} & {r['archive']} & "
                f"{r['mean_core_quality_gain_percent']:.2f} & {r['mean_objective_gain_percent']:.2f} & "
                f"{r['mean_runtime_gain_percent']:.2f} & {r['core_wins_total']}/{total_core_slots} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (output_root / "ablation_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nGenerated:")
    print(output_root / "ablation_v1_v5_per_benchmark.csv")
    print(output_root / "ablation_v1_v5_global.csv")
    print(output_root / "ablation_table.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())