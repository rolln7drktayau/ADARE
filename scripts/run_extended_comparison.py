from __future__ import annotations

"""Extended ADARE comparison against classical and learning-assisted MOEA baselines."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from algorithms import (
    AdareAlgorithm,
    MOEADAlgorithm,
    NSGA2Algorithm,
    NSGA3Algorithm,
    OVEAAlgorithm,
    QMOEADAWAAlgorithm,
    QLNSGA3Algorithm,
)
from evaluation import coverage_metric, filter_non_dominated, metric_gain_percent, quality_indicators
from main import build_initial_population
from problem import build_nodes, load_environments, load_tasks, topological_sort


ALGORITHMS = {
    "ADARE": (AdareAlgorithm, "config/adare_config.json"),
    "NSGA-III": (NSGA3Algorithm, "config/nsga3_config.json"),
    "NSGA-II": (NSGA2Algorithm, "config/nsga2_config.json"),
    "MOEA/D": (MOEADAlgorithm, "config/moead_config.json"),
    "QL-NSGA-III": (QLNSGA3Algorithm, "config/qlnsga3_config.json"),
    "OVEA-style": (OVEAAlgorithm, "config/ovea_config.json"),
    "QMOEA/D-AWA-style": (QMOEADAWAAlgorithm, "config/qmoeadawa_config.json"),
}

QUALITY_DIRECTIONS = {
    "hv": "max",
    "igd": "min",
    "spacing": "min",
    "epsilon": "min",
    "coverage": "max",
    "time": "min",
}
CORE_METRICS = ("hv", "igd", "spacing", "epsilon", "coverage")
OBJECTIVES = ("makespan", "latency", "cost", "energy")
ALGORITHM_COLORS = {
    "ADARE": "#D55E00",
    "NSGA-III": "#0072B2",
    "NSGA-II": "#009E73",
    "MOEA/D": "#CC79A7",
    "QL-NSGA-III": "#E69F00",
    "OVEA-style": "#56B4E9",
    "QMOEA/D-AWA-style": "#000000",
}


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0


def objective_label(name: str) -> str:
    return {
        "makespan": "Makespan",
        "latency": "Latency",
        "cost": "Cost",
        "energy": "Energy",
    }.get(name, name)


def plot_multialgo_pareto(
    benchmark_fronts: Dict[str, Dict[str, np.ndarray]],
    selected_algorithms: Sequence[str],
    output_path: Path,
) -> None:
    benchmarks = list(benchmark_fronts.keys())
    pairs = [(0, 1), (2, 3)]
    fig, axes = plt.subplots(len(benchmarks), len(pairs), figsize=(15, 4.2 * len(benchmarks)))
    if len(benchmarks) == 1:
        axes = np.asarray([axes])
    for row_idx, benchmark in enumerate(benchmarks):
        for col_idx, (i, j) in enumerate(pairs):
            ax = axes[row_idx, col_idx]
            for label in selected_algorithms:
                front = benchmark_fronts[benchmark].get(label)
                if front is None or len(front) == 0:
                    continue
                ax.scatter(
                    front[:, i],
                    front[:, j],
                    s=28 if label == "ADARE" else 18,
                    alpha=0.85 if label == "ADARE" else 0.58,
                    label=label,
                    color=ALGORITHM_COLORS.get(label),
                    edgecolors="#111111" if label == "ADARE" else "none",
                    linewidths=0.35 if label == "ADARE" else 0.0,
                )
            ax.set_title(f"{benchmark}: {objective_label(OBJECTIVES[i])} vs {objective_label(OBJECTIVES[j])}", fontsize=11)
            ax.set_xlabel(objective_label(OBJECTIVES[i]), fontsize=10)
            ax.set_ylabel(objective_label(OBJECTIVES[j]), fontsize=10)
            ax.grid(alpha=0.25, linestyle="--")
            ax.tick_params(axis="both", labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Representative Pareto Fronts: ADARE vs All Baselines", fontsize=15)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_multialgo_convergence(
    benchmark_histories: Dict[str, Dict[str, np.ndarray]],
    selected_algorithms: Sequence[str],
    output_path: Path,
) -> None:
    benchmarks = list(benchmark_histories.keys())
    fig, axes = plt.subplots(len(benchmarks), len(OBJECTIVES), figsize=(18, 3.4 * len(benchmarks)), sharex=False)
    if len(benchmarks) == 1:
        axes = np.asarray([axes])
    for row_idx, benchmark in enumerate(benchmarks):
        for obj_idx, obj_name in enumerate(OBJECTIVES):
            ax = axes[row_idx, obj_idx]
            for label in selected_algorithms:
                history = benchmark_histories[benchmark].get(label)
                if history is None or history.size == 0:
                    continue
                mean_curve = np.nanmean(history[:, :, obj_idx], axis=0)
                std_curve = np.nanstd(history[:, :, obj_idx], axis=0)
                x = np.arange(mean_curve.shape[0])
                ax.plot(
                    x,
                    mean_curve,
                    label=label,
                    color=ALGORITHM_COLORS.get(label),
                    linewidth=2.4 if label == "ADARE" else 1.5,
                    alpha=1.0 if label == "ADARE" else 0.86,
                )
                ax.fill_between(
                    x,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color=ALGORITHM_COLORS.get(label),
                    alpha=0.16 if label == "ADARE" else 0.08,
                )
            ax.set_title(f"{benchmark} - {objective_label(obj_name)}", fontsize=10)
            ax.set_xlabel("Generation", fontsize=9)
            ax.set_ylabel("Best value", fontsize=9)
            ax.grid(alpha=0.25, linestyle="--")
            ax.tick_params(axis="both", labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Convergence Traces: ADARE vs All Baselines", fontsize=15)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_benchmark(
    benchmark: str,
    selected_algorithms: Sequence[str],
    shared_config: Dict[str, Any],
    nodes: Sequence[Dict[str, Any]],
    runs: int,
    base_seed: int,
    output_root: Path,
) -> tuple[List[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    print(f"\nExtended benchmark {benchmark} | runs={runs} | algos={', '.join(selected_algorithms)}")
    tasks = load_tasks(benchmark)
    order = topological_sort(tasks)
    benchmark_dir = output_root / benchmark.split("_", 1)[0] / "reports"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    fronts: Dict[str, List[np.ndarray]] = {label: [] for label in selected_algorithms}
    histories: Dict[str, List[np.ndarray]] = {label: [] for label in selected_algorithms}
    times: Dict[str, List[float]] = {label: [] for label in selected_algorithms}
    objective_best: Dict[str, List[np.ndarray]] = {label: [] for label in selected_algorithms}

    configs = {label: load_json(config_path) for label, (_, config_path) in ALGORITHMS.items()}
    for run_idx in range(runs):
        run_seed = base_seed + 97 * run_idx
        print(f"  Run {run_idx + 1}/{runs} | seed={run_seed}")
        init_pop = build_initial_population(
            seed=run_seed,
            population_size=int(shared_config["population_size"]),
            num_tasks=len(tasks),
            num_nodes=len(nodes),
        )
        for algo_offset, label in enumerate(selected_algorithms):
            cls, _ = ALGORITHMS[label]
            algo_shared = dict(shared_config)
            algo_config = dict(configs[label])
            if "gene_mutation_probability" in algo_config:
                algo_shared["gene_mutation_probability"] = algo_config["gene_mutation_probability"]
            algorithm = cls(
                shared_config=algo_shared,
                algorithm_config=algo_config,
                nodes=nodes,
                tasks=tasks,
                topological_order=order,
                objective_names=OBJECTIVES,
                seed=run_seed + 11 + algo_offset * 13,
                initial_population=init_pop,
            )
            result = algorithm.run()
            front = filter_non_dominated(cls.population_to_array(result["population"]))
            pool = cls.population_to_array(result.get("objective_population", result["population"]))
            fronts[label].append(front)
            histories[label].append(np.asarray(result["history"], dtype=float))
            times[label].append(float(result["time"]))
            objective_best[label].append(np.min(pool, axis=0))

    reference_front = filter_non_dominated(
        np.vstack([front for label in selected_algorithms for front in fronts[label]])
    )
    run_rows: List[Dict[str, Any]] = []
    metrics_by_algo: Dict[str, List[Dict[str, float]]] = {label: [] for label in selected_algorithms}

    for run_idx in range(runs):
        for label in selected_algorithms:
            metrics = quality_indicators(fronts[label][run_idx], reference_front)
            metrics["coverage"] = coverage_metric(fronts[label][run_idx], fronts["ADARE"][run_idx])
            if label == "ADARE":
                metrics["coverage"] = np.nan
            metrics["time"] = times[label][run_idx]
            for obj_idx, obj_name in enumerate(OBJECTIVES):
                metrics[f"{obj_name}_best"] = float(objective_best[label][run_idx][obj_idx])
            metrics_by_algo[label].append(metrics)
            row = {"benchmark": benchmark, "run": run_idx + 1, "algorithm": label}
            row.update(metrics)
            run_rows.append(row)

    summary_rows: List[Dict[str, Any]] = []
    for baseline in selected_algorithms:
        if baseline == "ADARE":
            continue
        for metric, direction in QUALITY_DIRECTIONS.items():
            adare_values = [row[metric] for row in metrics_by_algo["ADARE"]]
            base_values = [row[metric] for row in metrics_by_algo[baseline]]
            if metric == "coverage":
                adare_values = [
                    coverage_metric(fronts["ADARE"][idx], fronts[baseline][idx])
                    for idx in range(runs)
                ]
                base_values = [
                    coverage_metric(fronts[baseline][idx], fronts["ADARE"][idx])
                    for idx in range(runs)
                ]
            adare_mean, adare_std = aggregate(adare_values)
            base_mean, base_std = aggregate(base_values)
            summary_rows.append(
                {
                    "benchmark": benchmark,
                    "baseline": baseline,
                    "metric": metric,
                    "adare_mean": adare_mean,
                    "adare_std": adare_std,
                    "baseline_mean": base_mean,
                    "baseline_std": base_std,
                    "gain_percent": metric_gain_percent(adare_mean, base_mean, direction),
                }
            )
        for obj_name in OBJECTIVES:
            metric = f"{obj_name}_best"
            adare_values = [row[metric] for row in metrics_by_algo["ADARE"]]
            base_values = [row[metric] for row in metrics_by_algo[baseline]]
            adare_mean, adare_std = aggregate(adare_values)
            base_mean, base_std = aggregate(base_values)
            summary_rows.append(
                {
                    "benchmark": benchmark,
                    "baseline": baseline,
                    "metric": metric,
                    "adare_mean": adare_mean,
                    "adare_std": adare_std,
                    "baseline_mean": base_mean,
                    "baseline_std": base_std,
                    "gain_percent": metric_gain_percent(adare_mean, base_mean, "min"),
                }
            )

    write_csv(benchmark_dir / f"{benchmark}_extended_run_metrics.csv", run_rows)
    write_csv(benchmark_dir / f"{benchmark}_extended_summary.csv", summary_rows)

    front_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    representative_fronts: Dict[str, np.ndarray] = {}
    history_arrays: Dict[str, np.ndarray] = {}
    for label in selected_algorithms:
        hv_values = np.asarray([row["hv"] for row in metrics_by_algo[label]], dtype=float)
        best_idx = int(np.nanargmax(hv_values)) if np.any(np.isfinite(hv_values)) else 0
        representative_fronts[label] = fronts[label][best_idx]
        history_arrays[label] = np.asarray(histories[label], dtype=float)
        for point_idx, point in enumerate(representative_fronts[label], start=1):
            row = {
                "benchmark": benchmark,
                "algorithm": label,
                "source_run": best_idx + 1,
                "point": point_idx,
            }
            row.update({name: float(point[idx]) for idx, name in enumerate(OBJECTIVES)})
            front_rows.append(row)
        for run_idx, history in enumerate(histories[label], start=1):
            for generation, values in enumerate(history):
                row = {
                    "benchmark": benchmark,
                    "algorithm": label,
                    "run": run_idx,
                    "generation": generation,
                }
                row.update({f"{name}_best": float(values[idx]) for idx, name in enumerate(OBJECTIVES)})
                history_rows.append(row)
    write_csv(benchmark_dir / f"{benchmark}_extended_representative_fronts.csv", front_rows)
    write_csv(benchmark_dir / f"{benchmark}_extended_histories.csv", history_rows)
    return summary_rows, representative_fronts, history_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADARE against extended MOEA baselines.")
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS.keys()))
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260219)
    parser.add_argument("--output-dir", default="output/extended")
    parser.add_argument("--figure-dir", default=None, help="Optional directory for multi-algorithm Pareto/convergence figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [label for label in args.algorithms if label in ALGORITHMS]
    if "ADARE" not in selected:
        selected.insert(0, "ADARE")
    shared_config = load_json("config/main_config.json")["shared_parameters"]
    shared_config = dict(shared_config)
    shared_config["generations"] = args.generations
    shared_config["population_size"] = args.population_size
    nodes = build_nodes(load_environments())

    all_summary: List[Dict[str, Any]] = []
    all_fronts: Dict[str, Dict[str, np.ndarray]] = {}
    all_histories: Dict[str, Dict[str, np.ndarray]] = {}
    for bench_idx, benchmark in enumerate(args.benchmarks):
        summary_rows, representative_fronts, history_arrays = run_benchmark(
            benchmark=benchmark,
            selected_algorithms=selected,
            shared_config=shared_config,
            nodes=nodes,
            runs=args.runs,
            base_seed=args.seed + 1000 * bench_idx,
            output_root=Path(args.output_dir),
        )
        all_summary.extend(summary_rows)
        all_fronts[benchmark] = representative_fronts
        all_histories[benchmark] = history_arrays
    write_csv(Path(args.output_dir) / "summary" / "extended_global_summary.csv", all_summary)
    if args.figure_dir:
        figure_dir = Path(args.figure_dir)
        plot_multialgo_pareto(all_fronts, selected, figure_dir / "multialgo_pareto_small_workflows.png")
        plot_multialgo_convergence(all_histories, selected, figure_dir / "multialgo_convergence_small_workflows.png")

    print("\nExtended comparison summary")
    for baseline in [label for label in selected if label != "ADARE"]:
        rows = [row for row in all_summary if row["baseline"] == baseline and row["metric"] in CORE_METRICS]
        wins = sum(1 for row in rows if float(row["gain_percent"]) > 0.0)
        mean_gain = float(np.mean([float(row["gain_percent"]) for row in rows])) if rows else float("nan")
        print(f"- ADARE vs {baseline}: core wins={wins}/{len(rows)}, mean core gain={mean_gain:.2f}%")


if __name__ == "__main__":
    main()
