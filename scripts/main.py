from __future__ import annotations

"""Main experiment runner for fair ADARE vs NSGA-III benchmark comparisons."""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from algorithms import AdareAlgorithm, NSGA3Algorithm
from evaluation import (
    aggregate_metric,
    coverage_metric,
    filter_non_dominated,
    metric_gain_percent,
    paired_wilcoxon,
    plot_convergence,
    plot_metric_boxplots,
    plot_pareto_projections,
    quality_indicators,
)
from problem import build_nodes, load_environments, load_tasks, topological_sort


QUALITY_METRICS = ("hv", "igd", "spacing", "epsilon", "coverage", "time")
CORE_QUALITY_METRICS = ("hv", "igd", "spacing", "epsilon", "coverage")


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file and enforce dictionary root type.
    
    This helper centralizes robust error messages for configuration files.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Fichier de configuration introuvable: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalide dans {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Le fichier {path} doit contenir un objet JSON.")
    return data


def parse_args() -> argparse.Namespace:
    """Parse command-line options for benchmark execution.
    """
    parser = argparse.ArgumentParser(
        description="Comparaison equitable ADARE vs NSGA-III sur des workflows DAG."
    )
    parser.add_argument("--benchmarks", nargs="+", help="Liste des benchmarks (ex: Montage_25 CyberShake_30)")
    parser.add_argument("--runs", type=int, help="Nombre de runs par benchmark")
    parser.add_argument("--seed", type=int, help="Seed de base")
    parser.add_argument("--population-size", type=int, help="Taille de population")
    parser.add_argument("--generations", type=int, help="Nombre de generations")
    parser.add_argument("--no-plots", action="store_true", help="Desactive la generation des figures")
    return parser.parse_args()


def build_initial_population(seed: int, population_size: int, num_tasks: int, num_nodes: int) -> List[List[int]]:
    """Create the shared initial population used by both algorithms.
    
    Using the same sampled individuals keeps the benchmark protocol fair.
    """
    rng = random.Random(seed)
    return [
        [rng.randrange(num_nodes) for _ in range(num_tasks)]
        for _ in range(population_size)
    ]


def benchmark_dirs(output_root: Path, benchmark: str) -> Dict[str, Path]:
    """Create and return benchmark-specific output directories.
    """
    workflow = benchmark.split("_", 1)[0] if "_" in benchmark else "Default"
    base = output_root / workflow
    plots = base / "plots"
    reports = base / "reports"
    plots.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return {"base": base, "plots": plots, "reports": reports}


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    """Write a sequence of dictionaries to CSV with explicit headers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_benchmark(
    benchmark: str,
    objective_names: Sequence[str],
    adare_runs: Sequence[Dict[str, float]],
    nsga_runs: Sequence[Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Aggregate per-run metrics into benchmark-level summary statistics.
    
    Returned rows are later used for reports and global aggregation.
    """
    metric_directions = {
        "hv": "max",
        "igd": "min",
        "spacing": "min",
        "epsilon": "min",
        "coverage": "max",
        "time": "min",
    }
    for obj in objective_names:
        metric_directions[f"{obj}_best"] = "min"

    summary_rows: List[Dict[str, Any]] = []
    for metric, direction in metric_directions.items():
        ad_values = np.asarray([run[metric] for run in adare_runs], dtype=float)
        ns_values = np.asarray([run[metric] for run in nsga_runs], dtype=float)

        ad_stats = aggregate_metric(ad_values)
        ns_stats = aggregate_metric(ns_values)
        gain = metric_gain_percent(ad_stats["mean"], ns_stats["mean"], direction)

        if direction == "max":
            wins = int(np.sum(ad_values > ns_values))
        else:
            wins = int(np.sum(ad_values < ns_values))
        ties = int(np.sum(np.isclose(ad_values, ns_values)))
        p_value = paired_wilcoxon(ad_values, ns_values, direction)

        summary_rows.append(
            {
                "benchmark": benchmark,
                "metric": metric,
                "direction": direction,
                "adare_mean": ad_stats["mean"],
                "adare_std": ad_stats["std"],
                "nsga3_mean": ns_stats["mean"],
                "nsga3_std": ns_stats["std"],
                "gain_percent": gain,
                "adare_run_wins": wins,
                "ties": ties,
                "wilcoxon_p": p_value,
            }
        )
    return summary_rows


def run_benchmark(
    benchmark: str,
    objective_names: Sequence[str],
    shared_config: Dict[str, Any],
    adare_config: Dict[str, Any],
    nsga3_config: Dict[str, Any],
    nodes: Sequence[Dict[str, Any]],
    runs: int,
    base_seed: int,
    output_root: Path,
    save_plots: bool,
) -> Dict[str, Any]:
    """Run ADARE and NSGA-III on one benchmark across multiple seeds.
    
    This function enforces fairness (same initial populations, same budget)
    and saves both metrics and optional plots to disk.
    """
    if runs <= 0:
        raise ValueError("Le nombre de runs doit etre strictement positif.")
    if not objective_names:
        raise ValueError("Aucun objectif defini pour l'evaluation.")
    if not nodes:
        raise ValueError("Aucun noeud disponible pour l'evaluation.")

    print(f"\nBenchmark {benchmark} | runs={runs}")
    tasks = load_tasks(benchmark)
    order = topological_sort(tasks)
    dirs = benchmark_dirs(output_root, benchmark)

    fronts: Dict[str, List[np.ndarray]] = {"ADARE": [], "NSGA-III": []}
    histories: Dict[str, List[np.ndarray]] = {"ADARE": [], "NSGA-III": []}
    times: Dict[str, List[float]] = {"ADARE": [], "NSGA-III": []}
    objective_best: Dict[str, List[np.ndarray]] = {"ADARE": [], "NSGA-III": []}
    objective_evaluations: Dict[str, List[int]] = {"ADARE": [], "NSGA-III": []}
    controller_trace_rows: List[Dict[str, Any]] = []

    nsga_shared = dict(shared_config)
    adare_shared = dict(shared_config)
    if "gene_mutation_probability" in nsga3_config:
        nsga_shared["gene_mutation_probability"] = nsga3_config["gene_mutation_probability"]
    if "gene_mutation_probability" in adare_config:
        adare_shared["gene_mutation_probability"] = adare_config["gene_mutation_probability"]

    for run_idx in range(runs):
        run_seed = base_seed + 97 * run_idx
        run_percent = 100.0 * (run_idx + 1) / runs
        print(f"  Run {run_idx + 1}/{runs} [{run_percent:6.2f}%] | seed={run_seed}", flush=True)
        init_pop = build_initial_population(
            seed=run_seed,
            population_size=int(shared_config["population_size"]),
            num_tasks=len(tasks),
            num_nodes=len(nodes),
        )

        # Use the same stochastic seed as well as the same initial population.
        # Algorithm-specific RNG offsets would break the matched-pairs design.
        algorithm_seed = run_seed + 11
        construction_started = time.perf_counter()
        nsga = NSGA3Algorithm(
            shared_config=nsga_shared,
            algorithm_config=nsga3_config,
            nodes=nodes,
            tasks=tasks,
            topological_order=order,
            objective_names=objective_names,
            seed=algorithm_seed,
            initial_population=init_pop,
        )
        nsga_construction_seconds = time.perf_counter() - construction_started
        construction_started = time.perf_counter()
        adare = AdareAlgorithm(
            shared_config=adare_shared,
            algorithm_config=adare_config,
            nodes=nodes,
            tasks=tasks,
            topological_order=order,
            objective_names=objective_names,
            seed=algorithm_seed,
            initial_population=init_pop,
        )
        adare_construction_seconds = time.perf_counter() - construction_started

        nsga_result = nsga.run()
        adare_result = adare.run()

        nsga_population = nsga_result.get("survival_population", nsga_result["population"])
        adare_population = adare_result.get("survival_population", adare_result["population"])
        nsga_front = filter_non_dominated(NSGA3Algorithm.population_to_array(nsga_population))
        adare_front = filter_non_dominated(AdareAlgorithm.population_to_array(adare_population))
        nsga_objective_pool = NSGA3Algorithm.population_to_array(nsga_population)
        adare_objective_pool = AdareAlgorithm.population_to_array(adare_population)

        fronts["NSGA-III"].append(nsga_front)
        fronts["ADARE"].append(adare_front)
        histories["NSGA-III"].append(nsga_result["history"])
        histories["ADARE"].append(adare_result["history"])
        times["NSGA-III"].append(nsga_construction_seconds + float(nsga_result["time"]))
        times["ADARE"].append(adare_construction_seconds + float(adare_result["time"]))
        objective_best["NSGA-III"].append(np.min(nsga_objective_pool, axis=0))
        objective_best["ADARE"].append(np.min(adare_objective_pool, axis=0))
        objective_evaluations["NSGA-III"].append(int(nsga.objective_evaluations))
        objective_evaluations["ADARE"].append(int(adare.objective_evaluations))
        for trace_row in adare_result.get("controller_trace", []):
            row = {"benchmark": benchmark, "run": run_idx + 1}
            row.update(trace_row)
            controller_trace_rows.append(row)

    reference_front = filter_non_dominated(np.vstack(fronts["ADARE"] + fronts["NSGA-III"]))

    adare_runs: List[Dict[str, float]] = []
    nsga_runs: List[Dict[str, float]] = []
    run_rows: List[Dict[str, Any]] = []

    for run_idx in range(runs):
        adare_metrics = quality_indicators(fronts["ADARE"][run_idx], reference_front)
        nsga_metrics = quality_indicators(fronts["NSGA-III"][run_idx], reference_front)

        adare_metrics["coverage"] = coverage_metric(fronts["ADARE"][run_idx], fronts["NSGA-III"][run_idx])
        nsga_metrics["coverage"] = coverage_metric(fronts["NSGA-III"][run_idx], fronts["ADARE"][run_idx])
        adare_metrics["time"] = times["ADARE"][run_idx]
        nsga_metrics["time"] = times["NSGA-III"][run_idx]

        for obj_idx, obj in enumerate(objective_names):
            adare_metrics[f"{obj}_best"] = float(objective_best["ADARE"][run_idx][obj_idx])
            nsga_metrics[f"{obj}_best"] = float(objective_best["NSGA-III"][run_idx][obj_idx])

        adare_runs.append(adare_metrics)
        nsga_runs.append(nsga_metrics)

        for algo_label, metrics in (("ADARE", adare_metrics), ("NSGA-III", nsga_metrics)):
            row = {
                "benchmark": benchmark,
                "run": run_idx + 1,
                "seed": base_seed + 97 * run_idx,
                "algorithm_seed": base_seed + 97 * run_idx + 11,
                "algorithm": algo_label,
            }
            row.update(metrics)
            row["objective_evaluations"] = objective_evaluations[algo_label][run_idx]
            run_rows.append(row)

    summary_rows = summarize_benchmark(benchmark, objective_names, adare_runs, nsga_runs)

    quality_wins = sum(
        1
        for row in summary_rows
        if row["metric"] in QUALITY_METRICS and row["gain_percent"] > 0.0
    )
    quality_significant_wins = sum(
        1
        for row in summary_rows
        if row["metric"] in QUALITY_METRICS and row["gain_percent"] > 0.0 and row["wilcoxon_p"] < 0.05
    )
    core_quality_wins = sum(
        1
        for row in summary_rows
        if row["metric"] in CORE_QUALITY_METRICS and row["gain_percent"] > 0.0
    )
    core_quality_significant_wins = sum(
        1
        for row in summary_rows
        if row["metric"] in CORE_QUALITY_METRICS and row["gain_percent"] > 0.0 and row["wilcoxon_p"] < 0.05
    )

    run_csv = dirs["reports"] / f"{benchmark}_run_metrics.csv"
    summary_csv = dirs["reports"] / f"{benchmark}_summary_metrics.csv"
    write_csv(run_csv, fieldnames=list(run_rows[0].keys()), rows=run_rows)
    write_csv(summary_csv, fieldnames=list(summary_rows[0].keys()), rows=summary_rows)
    if controller_trace_rows:
        trace_csv = dirs["reports"] / f"{benchmark}_adare_controller_trace.csv"
        write_csv(trace_csv, fieldnames=list(controller_trace_rows[0].keys()), rows=controller_trace_rows)

        aggregate: Dict[tuple[str, str], Dict[str, float]] = {}
        for row in controller_trace_rows:
            key = (str(row["context"]), str(row["operator"]))
            stats = aggregate.setdefault(key, {"uses": 0.0, "reward_sum": 0.0})
            stats["uses"] += 1.0
            stats["reward_sum"] += float(row["reward"])
        aggregate_rows = [
            {
                "benchmark": benchmark,
                "context": context,
                "operator": operator,
                "uses": int(stats["uses"]),
                "mean_reward": stats["reward_sum"] / max(1.0, stats["uses"]),
            }
            for (context, operator), stats in sorted(aggregate.items())
        ]
        aggregate_csv = dirs["reports"] / f"{benchmark}_adare_controller_summary.csv"
        write_csv(aggregate_csv, fieldnames=list(aggregate_rows[0].keys()), rows=aggregate_rows)

    report_txt = dirs["reports"] / f"{benchmark}_comparison_report.txt"
    with report_txt.open("w", encoding="utf-8") as handle:
        handle.write(f"Benchmark: {benchmark}\n")
        handle.write("Protocol: same initial population, same generation budget, same objective evaluator.\n")
        handle.write(f"Runs: {runs}\n\n")
        handle.write("Metric, ADARE(mean±std), NSGA-III(mean±std), Gain%, Wilcoxon p\n")
        for row in summary_rows:
            handle.write(
                f"{row['metric']}, "
                f"{row['adare_mean']:.6g}±{row['adare_std']:.3g}, "
                f"{row['nsga3_mean']:.6g}±{row['nsga3_std']:.3g}, "
                f"{row['gain_percent']:.2f}, "
                f"{row['wilcoxon_p']:.4f}\n"
            )
        handle.write(
            f"\nADARE quality wins (all): {quality_wins}/{len(QUALITY_METRICS)} "
            f"(significant: {quality_significant_wins}/{len(QUALITY_METRICS)}).\n"
        )
        handle.write(
            f"ADARE quality wins (without runtime): {core_quality_wins}/{len(CORE_QUALITY_METRICS)} "
            f"(significant: {core_quality_significant_wins}/{len(CORE_QUALITY_METRICS)}).\n"
        )

    if save_plots:
        generated_plot_paths: List[Path] = []
        histories_np = {
            "ADARE": np.asarray(histories["ADARE"], dtype=float),
            "NSGA-III": np.asarray(histories["NSGA-III"], dtype=float),
        }
        hv_adare = [row["hv"] for row in adare_runs]
        hv_nsga = [row["hv"] for row in nsga_runs]
        adare_arr = np.asarray(hv_adare, dtype=float)
        nsga_arr = np.asarray(hv_nsga, dtype=float)
        best_adare_idx = 0 if np.all(~np.isfinite(adare_arr)) else int(np.nanargmax(adare_arr))
        best_nsga_idx = 0 if np.all(~np.isfinite(nsga_arr)) else int(np.nanargmax(nsga_arr))

        convergence_path = dirs["plots"] / f"convergence_{benchmark}.png"
        plot_convergence(
            histories=histories_np,
            objective_names=objective_names,
            output_path=convergence_path,
            title=f"Convergence | {benchmark}",
        )
        generated_plot_paths.append(convergence_path)

        pareto_path = dirs["plots"] / f"pareto_{benchmark}.png"
        plot_pareto_projections(
            fronts={
                "ADARE": fronts["ADARE"][best_adare_idx],
                "NSGA-III": fronts["NSGA-III"][best_nsga_idx],
            },
            objective_names=objective_names,
            output_path=pareto_path,
            title=f"Pareto projections | {benchmark}",
        )
        generated_plot_paths.append(pareto_path)

        for metric in QUALITY_METRICS:
            boxplot_path = dirs["plots"] / f"box_{metric}_{benchmark}.png"
            plot_metric_boxplots(
                metric_name=metric,
                adare_values=[row[metric] for row in adare_runs],
                nsga_values=[row[metric] for row in nsga_runs],
                output_path=boxplot_path,
                title=f"{metric} distribution | {benchmark}",
            )
            generated_plot_paths.append(boxplot_path)

        print("  Graphiques générés :")
        for path in generated_plot_paths:
            print(f"    - {path.resolve()}")

    print(
        f"  -> ADARE quality wins: {quality_wins}/{len(QUALITY_METRICS)} "
        f"(sans runtime: {core_quality_wins}/{len(CORE_QUALITY_METRICS)})"
    )

    return {
        "benchmark": benchmark,
        "summary_rows": summary_rows,
        "quality_wins": quality_wins,
        "quality_significant_wins": quality_significant_wins,
        "core_quality_wins": core_quality_wins,
        "core_quality_significant_wins": core_quality_significant_wins,
        "report_path": str(report_txt),
    }


def build_global_report(results: Sequence[Dict[str, Any]], output_root: Path) -> Path:
    """Build consolidated CSV/TXT reports across all executed benchmarks.
    """
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for result in results:
        quality_rows = [r for r in result["summary_rows"] if r["metric"] in QUALITY_METRICS]
        core_quality_rows = [r for r in result["summary_rows"] if r["metric"] in CORE_QUALITY_METRICS]
        objective_rows = [r for r in result["summary_rows"] if r["metric"].endswith("_best")]
        runtime_row = next((r for r in result["summary_rows"] if r["metric"] == "time"), None)
        quality_gain = (
            float(np.mean([r["gain_percent"] for r in core_quality_rows]))
            if core_quality_rows
            else float("nan")
        )
        objective_gain = (
            float(np.mean([r["gain_percent"] for r in objective_rows]))
            if objective_rows
            else float("nan")
        )
        rows.append(
            {
                "benchmark": result["benchmark"],
                "quality_wins": result["quality_wins"],
                "quality_significant_wins": result["quality_significant_wins"],
                "core_quality_wins": result["core_quality_wins"],
                "core_quality_significant_wins": result["core_quality_significant_wins"],
                "mean_core_quality_gain_percent": quality_gain,
                "mean_objective_gain_percent": objective_gain,
                "runtime_gain_percent": float(runtime_row["gain_percent"]) if runtime_row else float("nan"),
                "report_path": result["report_path"],
            }
        )

    csv_path = summary_dir / "global_summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "benchmark",
            "quality_wins",
            "quality_significant_wins",
            "core_quality_wins",
            "core_quality_significant_wins",
            "mean_core_quality_gain_percent",
            "mean_objective_gain_percent",
            "runtime_gain_percent",
            "report_path",
        ]
    write_csv(csv_path, fieldnames=fieldnames, rows=rows)

    txt_path = summary_dir / "global_comparison_report.txt"
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("ADARE vs NSGA-III - Global summary\n")
        handle.write("=" * 60 + "\n")
        if not rows:
            handle.write("Aucun resultat disponible.\n")
            return txt_path

        for row in rows:
            handle.write(
                f"{row['benchmark']}: "
                f"quality(all)={row['quality_wins']}/{len(QUALITY_METRICS)}, "
                f"quality(core)={row['core_quality_wins']}/{len(CORE_QUALITY_METRICS)}, "
                f"objective_mean_gain={row['mean_objective_gain_percent']:.2f}%, "
                f"core_quality_mean_gain={row['mean_core_quality_gain_percent']:.2f}%, "
                f"runtime_gain={row['runtime_gain_percent']:.2f}%\n"
            )

        avg_gain = float(np.mean([row["mean_core_quality_gain_percent"] for row in rows]))
        total_wins = sum(row["quality_wins"] for row in rows)
        total_sig_wins = sum(row["quality_significant_wins"] for row in rows)
        total_slots = len(rows) * len(QUALITY_METRICS)
        total_core_wins = sum(row["core_quality_wins"] for row in rows)
        total_core_sig = sum(row["core_quality_significant_wins"] for row in rows)
        total_core_slots = len(rows) * len(CORE_QUALITY_METRICS)
        avg_objective_gain = float(np.mean([row["mean_objective_gain_percent"] for row in rows]))
        avg_runtime_gain = float(np.mean([row["runtime_gain_percent"] for row in rows]))
        handle.write("\nOverall:\n")
        handle.write(f"- Mean core quality gain (without runtime): {avg_gain:.2f}%\n")
        handle.write(f"- Mean objective gain: {avg_objective_gain:.2f}%\n")
        handle.write(f"- Mean runtime gain: {avg_runtime_gain:.2f}%\n")
        handle.write(f"- ADARE wins (all quality): {total_wins}/{total_slots}\n")
        handle.write(f"- ADARE wins (core quality): {total_core_wins}/{total_core_slots}\n")
        handle.write(f"- ADARE significant wins (all quality): {total_sig_wins}/{total_slots}\n")
        handle.write(f"- ADARE significant wins (core quality): {total_core_sig}/{total_core_slots}\n")

    return txt_path


def main() -> int:
    """Program entrypoint: load config, run benchmarks, and export reports.
    """
    args = parse_args()
    root = ROOT
    config_dir = root / "config"

    main_config = load_json(config_dir / "main_config.json")
    adare_config = load_json(config_dir / "adare_config.json")
    nsga3_config = load_json(config_dir / "nsga3_config.json")

    objective_names = list(main_config["objectives"])
    benchmarks = args.benchmarks if args.benchmarks else list(main_config["benchmarks"])
    runs = int(args.runs if args.runs is not None else main_config["general_parameters"]["num_runs"])
    base_seed = int(args.seed if args.seed is not None else main_config["general_parameters"]["base_seed"])
    save_plots = bool(main_config["general_parameters"].get("save_plots", True)) and not args.no_plots

    if runs <= 0:
        raise ValueError("Le nombre de runs doit etre strictement positif.")
    if not benchmarks:
        raise ValueError("Aucun benchmark fourni. Verifie main_config.json ou --benchmarks.")

    shared = dict(main_config["shared_parameters"])
    if args.population_size is not None:
        shared["population_size"] = int(args.population_size)
    if args.generations is not None:
        shared["generations"] = int(args.generations)
    if int(shared["population_size"]) <= 1:
        raise ValueError("population_size doit etre > 1.")
    if int(shared["generations"]) <= 0:
        raise ValueError("generations doit etre > 0.")

    output_root = root / str(main_config["general_parameters"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    environments = load_environments()
    nodes = build_nodes(environments)

    print("Protocol: same initial populations, same generation budget, same objective evaluator.")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(
        "Shared params: "
        f"pop={shared['population_size']}, gen={shared['generations']}, "
        f"cxpb={shared['crossover_probability']}, mutpb={shared['mutation_probability']}"
    )

    benchmark_results = []
    for idx, benchmark in enumerate(benchmarks):
        result = run_benchmark(
            benchmark=benchmark,
            objective_names=objective_names,
            shared_config=shared,
            adare_config=adare_config,
            nsga3_config=nsga3_config,
            nodes=nodes,
            runs=runs,
            base_seed=base_seed + idx * 1000,
            output_root=output_root,
            save_plots=save_plots,
        )
        benchmark_results.append(result)

    global_report = build_global_report(benchmark_results, output_root)
    print(f"\nRapport global: {global_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
