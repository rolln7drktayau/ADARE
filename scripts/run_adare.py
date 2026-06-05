from __future__ import annotations

"""Standalone ADARE runner for executing the algorithm without baselines."""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from algorithms import AdareAlgorithm
from evaluation import filter_non_dominated
from main import build_initial_population, load_json, write_csv
from problem import build_nodes, load_environments, load_tasks, topological_sort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADARE alone on one or more workflow benchmarks.")
    parser.add_argument("--benchmarks", nargs="+", required=True, help="Benchmarks, e.g. Montage_25 CyberShake_1000")
    parser.add_argument("--runs", type=int, default=1, help="Number of ADARE runs per benchmark")
    parser.add_argument("--seed", type=int, help="Base seed")
    parser.add_argument("--population-size", type=int, help="Population size")
    parser.add_argument("--generations", type=int, help="Number of generations")
    parser.add_argument("--output-dir", default="output/adare_only", help="Output directory")
    return parser.parse_args()


def benchmark_dirs(output_root: Path, benchmark: str) -> Dict[str, Path]:
    workflow = benchmark.split("_", 1)[0] if "_" in benchmark else "Default"
    reports = output_root / workflow / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return {"reports": reports}


def rows_from_objective_array(
    benchmark: str,
    run_idx: int,
    objective_names: Sequence[str],
    values: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for solution_idx, vector in enumerate(values, 1):
        row: Dict[str, Any] = {"benchmark": benchmark, "run": run_idx, "solution": solution_idx}
        row.update({name: float(vector[idx]) for idx, name in enumerate(objective_names)})
        rows.append(row)
    return rows


def run_adare_benchmark(
    benchmark: str,
    objective_names: Sequence[str],
    shared_config: Dict[str, Any],
    adare_config: Dict[str, Any],
    nodes: Sequence[Dict[str, Any]],
    runs: int,
    base_seed: int,
    output_root: Path,
) -> Dict[str, Any]:
    if runs <= 0:
        raise ValueError("runs must be > 0")

    tasks = load_tasks(benchmark)
    order = topological_sort(tasks)
    dirs = benchmark_dirs(output_root, benchmark)

    run_rows: List[Dict[str, Any]] = []
    front_rows: List[Dict[str, Any]] = []
    objective_pool_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    print(f"\nADARE only | benchmark={benchmark} | runs={runs}")
    for run_idx in range(runs):
        run_seed = base_seed + 97 * run_idx
        print(f"  Run {run_idx + 1}/{runs} | seed={run_seed}")
        init_pop = build_initial_population(
            seed=run_seed,
            population_size=int(shared_config["population_size"]),
            num_tasks=len(tasks),
            num_nodes=len(nodes),
        )
        algorithm = AdareAlgorithm(
            shared_config=shared_config,
            algorithm_config=adare_config,
            nodes=nodes,
            tasks=tasks,
            topological_order=order,
            objective_names=objective_names,
            seed=run_seed,
            initial_population=init_pop,
        )

        started = time.perf_counter()
        result = algorithm.run()
        elapsed = float(result.get("time", time.perf_counter() - started))

        population_values = AdareAlgorithm.population_to_array(result["population"])
        objective_pool = AdareAlgorithm.population_to_array(result.get("objective_population", result["population"]))
        front = filter_non_dominated(population_values)
        best_values = np.min(objective_pool, axis=0)

        row: Dict[str, Any] = {
            "benchmark": benchmark,
            "run": run_idx + 1,
            "seed": run_seed,
            "time": elapsed,
            "population_size": int(shared_config["population_size"]),
            "generations": int(shared_config["generations"]),
            "front_size": int(len(front)),
            "objective_pool_size": int(len(objective_pool)),
        }
        row.update({f"{name}_best": float(best_values[idx]) for idx, name in enumerate(objective_names)})
        run_rows.append(row)

        front_rows.extend(rows_from_objective_array(benchmark, run_idx + 1, objective_names, front))
        objective_pool_rows.extend(rows_from_objective_array(benchmark, run_idx + 1, objective_names, objective_pool))

        history = np.asarray(result["history"], dtype=float)
        for generation, vector in enumerate(history):
            hrow: Dict[str, Any] = {"benchmark": benchmark, "run": run_idx + 1, "generation": generation}
            hrow.update({f"{name}_best": float(vector[idx]) for idx, name in enumerate(objective_names)})
            history_rows.append(hrow)

        for trace in result.get("controller_trace", []):
            trow = {"benchmark": benchmark, "run": run_idx + 1}
            trow.update(trace)
            trace_rows.append(trow)

    objective_fields = ["benchmark", "run", "solution", *objective_names]
    metric_fields = [
        "benchmark",
        "run",
        "seed",
        "time",
        "population_size",
        "generations",
        "front_size",
        "objective_pool_size",
        *[f"{name}_best" for name in objective_names],
    ]
    history_fields = ["benchmark", "run", "generation", *[f"{name}_best" for name in objective_names]]
    trace_fields = [
        "benchmark",
        "run",
        "generation",
        "context_id",
        "context",
        "operator",
        "reward",
        "diversity",
        "stagnation",
    ]

    prefix = dirs["reports"] / f"{benchmark}_adare_only"
    write_csv(prefix.with_name(f"{benchmark}_adare_only_run_metrics.csv"), metric_fields, run_rows)
    write_csv(prefix.with_name(f"{benchmark}_adare_only_front.csv"), objective_fields, front_rows)
    write_csv(prefix.with_name(f"{benchmark}_adare_only_objective_pool.csv"), objective_fields, objective_pool_rows)
    write_csv(prefix.with_name(f"{benchmark}_adare_only_history.csv"), history_fields, history_rows)
    if trace_rows:
        write_csv(prefix.with_name(f"{benchmark}_adare_only_controller_trace.csv"), trace_fields, trace_rows)

    report_path = prefix.with_name(f"{benchmark}_adare_only_report.txt")
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Benchmark: {benchmark}\n")
        handle.write("Mode: ADARE only, no baseline comparison.\n")
        handle.write(f"Runs: {runs}\n")
        handle.write(f"Population: {shared_config['population_size']}\n")
        handle.write(f"Generations: {shared_config['generations']}\n\n")
        for row in run_rows:
            best = ", ".join(f"{name}={row[f'{name}_best']:.6g}" for name in objective_names)
            handle.write(
                f"Run {row['run']} | seed={row['seed']} | time={row['time']:.2f}s | "
                f"front={row['front_size']} | {best}\n"
            )

    print(f"  Report: {report_path}")
    return {"benchmark": benchmark, "report": report_path, "runs": run_rows}


def main() -> int:
    args = parse_args()
    root = ROOT
    config_dir = root / "config"
    main_config = load_json(config_dir / "main_config.json")
    adare_config = load_json(config_dir / "adare_config.json")

    objective_names = list(main_config["objectives"])
    shared = dict(main_config["shared_parameters"])
    if args.population_size is not None:
        shared["population_size"] = int(args.population_size)
    if args.generations is not None:
        shared["generations"] = int(args.generations)

    base_seed = int(args.seed if args.seed is not None else main_config["general_parameters"]["base_seed"])
    output_root = root / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    nodes = build_nodes(load_environments())
    for idx, benchmark in enumerate(args.benchmarks):
        run_adare_benchmark(
            benchmark=benchmark,
            objective_names=objective_names,
            shared_config=shared,
            adare_config=adare_config,
            nodes=nodes,
            runs=args.runs,
            base_seed=base_seed + idx * 1000,
            output_root=output_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
