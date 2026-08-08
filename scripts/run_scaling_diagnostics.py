from __future__ import annotations

"""Targeted scaling, anytime-convergence, memory, and resource sensitivity study."""

import argparse
import copy
import csv
import gc
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # type: ignore  # noqa: E402

from algorithms import AdareAlgorithm, NSGA3Algorithm, QLNSGA3Algorithm  # noqa: E402
from evaluation import coverage_metric, filter_non_dominated, quality_indicators  # noqa: E402
from main import build_initial_population  # noqa: E402
from problem import build_nodes, load_environments, load_tasks, topological_sort  # noqa: E402


OBJECTIVES = ("makespan", "latency", "cost", "energy")
ALGORITHMS = {
    "ADARE": (AdareAlgorithm, "config/adare_config.json"),
    "NSGA-III": (NSGA3Algorithm, "config/nsga3_config.json"),
    "QL-NSGA-III": (QLNSGA3Algorithm, "config/qlnsga3_config.json"),
}
COLORS = {"ADARE": "#D55E00", "NSGA-III": "#0072B2", "QL-NSGA-III": "#E69F00"}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def process_tree_rss(process: psutil.Process) -> int:
    """Return current RSS for the runner and its live evaluator workers."""
    total = 0
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.Error, OSError):
        pass
    for current in processes:
        try:
            total += int(current.memory_info().rss)
        except (psutil.Error, OSError):
            continue
    return total


class PeakRssSampler:
    """Sample process-tree RSS during one algorithm run."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.baseline_bytes = process_tree_rss(self.process)
        self.peak_bytes = self.baseline_bytes
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, process_tree_rss(self.process))

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> tuple[int, int]:
        self.peak_bytes = max(self.peak_bytes, process_tree_rss(self.process))
        self._stop.set()
        self._thread.join(timeout=2.0)
        return self.peak_bytes, max(0, self.peak_bytes - self.baseline_bytes)


def resource_scenarios() -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    nominal = load_environments()
    compute_skew = copy.deepcopy(nominal)
    compute_skew["Edge"]["processing_rate"] *= 0.65
    compute_skew["Cloud"]["processing_rate"] *= 1.50

    network_scarce = copy.deepcopy(nominal)
    for layer in network_scarce.values():
        layer["uplink_bandwidth"] *= 0.25
        layer["downlink_bandwidth"] *= 0.25

    manifest = {
        "nominal": "Unmodified data/environments.json",
        "compute_skew": "Edge processing rate x0.65; Cloud processing rate x1.50; other fields fixed",
        "network_scarce": "All uplink and downlink bandwidths x0.25; other fields fixed",
    }
    return {
        "nominal": nominal,
        "compute_skew": compute_skew,
        "network_scarce": network_scarce,
    }, manifest


def experiment_cases() -> list[dict[str, str]]:
    return [
        {
            "case_id": "scale_montage_25_nominal",
            "study": "scaling",
            "benchmark": "Montage_25",
            "resource_scenario": "nominal",
        },
        {
            "case_id": "scale_montage_1000_nominal",
            "study": "scaling_resource",
            "benchmark": "Montage_1000",
            "resource_scenario": "nominal",
        },
        {
            "case_id": "scale_montage_3000_nominal",
            "study": "scaling",
            "benchmark": "Montage_3000_wfcommons",
            "resource_scenario": "nominal",
        },
        {
            "case_id": "resource_montage_1000_compute_skew",
            "study": "resource",
            "benchmark": "Montage_1000",
            "resource_scenario": "compute_skew",
        },
        {
            "case_id": "resource_montage_1000_network_scarce",
            "study": "resource",
            "benchmark": "Montage_1000",
            "resource_scenario": "network_scarce",
        },
    ]


def latest_at_budget(rows: list[dict[str, Any]], coordinate: str, budget: float) -> dict[str, Any]:
    eligible = [row for row in rows if float(row[coordinate]) <= budget + 1e-12]
    return eligible[-1] if eligible else rows[0]


def build_budget_rows(generation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in generation_rows:
        key = (str(row["case_id"]), int(row["run"]), str(row["algorithm"]))
        grouped.setdefault(key, []).append(row)
    by_case: dict[str, list[list[dict[str, Any]]]] = {}
    for (case_id, _run, _algorithm), rows in grouped.items():
        rows.sort(key=lambda item: int(item["generation"]))
        by_case.setdefault(case_id, []).append(rows)

    for case_id, trajectories in by_case.items():
        for axis, coordinate in (("evaluations", "objective_evaluations"), ("time", "elapsed_seconds")):
            starts = [float(rows[0][coordinate]) for rows in trajectories]
            ends = [float(rows[-1][coordinate]) for rows in trajectories]
            low = max(starts)
            high = min(ends)
            if high < low:
                continue
            for fraction in np.linspace(0.0, 1.0, 11):
                budget = low + float(fraction) * (high - low)
                for rows in trajectories:
                    selected = latest_at_budget(rows, coordinate, budget)
                    out.append(
                        {
                            "case_id": case_id,
                            "study": selected["study"],
                            "benchmark": selected["benchmark"],
                            "task_count": selected["task_count"],
                            "resource_scenario": selected["resource_scenario"],
                            "run": selected["run"],
                            "seed": selected["seed"],
                            "algorithm": selected["algorithm"],
                            "axis": axis,
                            "budget_fraction": float(fraction),
                            "budget": budget,
                            "observed_generation": selected["generation"],
                            "observed_evaluations": selected["objective_evaluations"],
                            "observed_seconds": selected["elapsed_seconds"],
                            "hv": selected["hv"],
                            "igd": selected["igd"],
                            "spacing": selected["spacing"],
                            "epsilon": selected["epsilon"],
                            "algorithm_coverage_of_adare": selected["algorithm_coverage_of_adare"],
                            "adare_coverage_of_algorithm": selected["adare_coverage_of_algorithm"],
                        }
                    )
    return out


def plot_anytime(generation_rows: list[dict[str, Any]], output_dir: Path, coordinate: str) -> None:
    scale_cases = [case["case_id"] for case in experiment_cases() if case["study"] in {"scaling", "scaling_resource"}]
    metric_specs = (("hv", "HV (higher is better)"), ("igd", "IGD (lower is better)"))
    fig, axes = plt.subplots(len(scale_cases), len(metric_specs), figsize=(13, 10.5))
    for row_idx, case_id in enumerate(scale_cases):
        case_rows = [row for row in generation_rows if row["case_id"] == case_id]
        benchmark = str(case_rows[0]["benchmark"])
        for col_idx, (metric, ylabel) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            for algorithm in ALGORITHMS:
                algo_rows = [row for row in case_rows if row["algorithm"] == algorithm]
                by_generation: dict[int, list[dict[str, Any]]] = {}
                for row in algo_rows:
                    by_generation.setdefault(int(row["generation"]), []).append(row)
                x = [float(np.mean([float(item[coordinate]) for item in by_generation[g]])) for g in sorted(by_generation)]
                y = [float(np.mean([float(item[metric]) for item in by_generation[g]])) for g in sorted(by_generation)]
                ax.plot(x, y, label=algorithm, color=COLORS[algorithm], linewidth=2.0)
            ax.set_title(benchmark)
            ax.set_xlabel("Objective evaluations" if coordinate == "objective_evaluations" else "Elapsed seconds")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25, linestyle="--")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    suffix = "evaluations" if coordinate == "objective_evaluations" else "time"
    fig.savefig(output_dir / f"scaling_convergence_vs_{suffix}.png", dpi=300)
    plt.close(fig)


def plot_memory(run_rows: list[dict[str, Any]], output_dir: Path) -> None:
    rows = [row for row in run_rows if row["study"] in {"scaling", "scaling_resource"}]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for algorithm in ALGORITHMS:
        algo_rows = [row for row in rows if row["algorithm"] == algorithm]
        task_counts = sorted({int(row["task_count"]) for row in algo_rows})
        means = [
            float(np.mean([float(row["peak_incremental_rss_mib"]) for row in algo_rows if int(row["task_count"]) == count]))
            for count in task_counts
        ]
        ax.plot(task_counts, means, marker="o", label=algorithm, color=COLORS[algorithm], linewidth=2.0)
    ax.set_xscale("log")
    ax.set_xlabel("Workflow tasks (log scale)")
    ax.set_ylabel("Peak incremental process-tree RSS (MiB)")
    ax.set_title("Sampled memory scaling")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "scaling_peak_memory.png", dpi=300)
    plt.close(fig)


def plot_resource_sensitivity(run_rows: list[dict[str, Any]], output_dir: Path) -> None:
    rows = [row for row in run_rows if int(row["task_count"]) == 1000]
    scenarios = ["nominal", "compute_skew", "network_scarce"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, metric, title in ((axes[0], "hv", "Final HV"), (axes[1], "igd", "Final IGD")):
        x = np.arange(len(scenarios), dtype=float)
        width = 0.24
        for idx, algorithm in enumerate(ALGORITHMS):
            means = [
                float(
                    np.mean(
                        [
                            float(row[metric])
                            for row in rows
                            if row["resource_scenario"] == scenario and row["algorithm"] == algorithm
                        ]
                    )
                )
                for scenario in scenarios
            ]
            ax.bar(x + (idx - 1) * width, means, width=width, label=algorithm, color=COLORS[algorithm])
        ax.set_xticks(x, scenarios, rotation=15)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(output_dir / "resource_sensitivity_hv_igd.png", dpi=300)
    plt.close(fig)


def run_case(
    case: dict[str, str],
    environments: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = load_tasks(case["benchmark"])
    order = topological_sort(tasks)
    nodes = build_nodes(environments)
    shared = dict(load_json("config/main_config.json")["shared_parameters"])
    shared["generations"] = args.generations
    shared["population_size"] = args.population_size
    configs = {name: load_json(path) for name, (_cls, path) in ALGORITHMS.items()}
    for config in configs.values():
        config["capture_generation_snapshots"] = True
        config["capture_generation_telemetry"] = True

    trajectories: list[dict[str, Any]] = []
    raw_runs: list[dict[str, Any]] = []
    total_steps = args.runs * len(ALGORITHMS)
    completed = 0
    print(
        f"\nDiagnostic case {case['case_id']} | tasks={len(tasks)} | "
        f"scenario={case['resource_scenario']}",
        flush=True,
    )
    for run_idx in range(args.runs):
        seed = args.seed + 97 * run_idx
        initial_population = build_initial_population(
            seed=seed,
            population_size=args.population_size,
            num_tasks=len(tasks),
            num_nodes=len(nodes),
        )
        algorithm_names = list(ALGORITHMS)
        shift = run_idx % len(algorithm_names)
        algorithm_order = algorithm_names[shift:] + algorithm_names[:shift]
        for algorithm_name in algorithm_order:
            algorithm_cls, _config_path = ALGORITHMS[algorithm_name]
            completed += 1
            print(
                f"  {algorithm_name} run {run_idx + 1}/{args.runs} "
                f"[{100.0 * completed / total_steps:6.2f}%]",
                flush=True,
            )
            sampler = PeakRssSampler(args.memory_sample_interval)
            sampler.start()
            try:
                construction_started = time.perf_counter()
                algorithm = algorithm_cls(
                    shared_config=dict(shared),
                    algorithm_config=dict(configs[algorithm_name]),
                    nodes=nodes,
                    tasks=tasks,
                    topological_order=order,
                    objective_names=OBJECTIVES,
                    seed=seed + 11,
                    initial_population=initial_population,
                )
                construction_seconds = time.perf_counter() - construction_started
                result = algorithm.run()
            finally:
                peak_rss, incremental_rss = sampler.stop()
            snapshots = [np.asarray(snapshot, dtype=float) for snapshot in result["generation_snapshots"]]
            telemetry = list(result["generation_telemetry"])
            if len(snapshots) != args.generations + 1 or len(telemetry) != args.generations + 1:
                raise RuntimeError(
                    f"Incomplete telemetry for {case['case_id']} / {algorithm_name} / run {run_idx + 1}: "
                    f"snapshots={len(snapshots)}, telemetry={len(telemetry)}"
                )
            final_population = result.get("survival_population", result["population"])
            final_front = filter_non_dominated(algorithm_cls.population_to_array(final_population))
            trajectories.append(
                {
                    "case": case,
                    "task_count": len(tasks),
                    "run": run_idx + 1,
                    "seed": seed,
                    "algorithm": algorithm_name,
                    "snapshots": snapshots,
                    "telemetry": telemetry,
                    "final_front": final_front,
                    "construction_seconds": construction_seconds,
                    "run_seconds": float(result["time"]),
                    "peak_rss_mib": peak_rss / (1024.0**2),
                    "peak_incremental_rss_mib": incremental_rss / (1024.0**2),
                }
            )
            del result, algorithm
            gc.collect()

    reference_front = filter_non_dominated(np.vstack([item["final_front"] for item in trajectories]))
    generation_rows: list[dict[str, Any]] = []
    for item in trajectories:
        for generation, (snapshot, telemetry) in enumerate(zip(item["snapshots"], item["telemetry"])):
            front = filter_non_dominated(snapshot)
            metrics = quality_indicators(front, reference_front)
            paired_adare = next(
                candidate
                for candidate in trajectories
                if candidate["run"] == item["run"] and candidate["algorithm"] == "ADARE"
            )
            adare_front = filter_non_dominated(paired_adare["snapshots"][generation])
            algorithm_coverage = float("nan")
            adare_coverage = float("nan")
            if item["algorithm"] != "ADARE":
                algorithm_coverage = coverage_metric(front, adare_front)
                adare_coverage = coverage_metric(adare_front, front)
            row = {
                "case_id": case["case_id"],
                "study": case["study"],
                "benchmark": case["benchmark"],
                "task_count": item["task_count"],
                "resource_scenario": case["resource_scenario"],
                "run": item["run"],
                "seed": item["seed"],
                "algorithm": item["algorithm"],
                "generation": generation,
                "objective_evaluations": telemetry["objective_evaluations"],
                "elapsed_seconds": float(telemetry["elapsed_seconds"]) + item["construction_seconds"],
                **metrics,
                "algorithm_coverage_of_adare": algorithm_coverage,
                "adare_coverage_of_algorithm": adare_coverage,
            }
            generation_rows.append(row)

        final_metrics = quality_indicators(item["final_front"], reference_front)
        paired_adare = next(
            candidate
            for candidate in trajectories
            if candidate["run"] == item["run"] and candidate["algorithm"] == "ADARE"
        )
        algorithm_coverage = float("nan")
        adare_coverage = float("nan")
        if item["algorithm"] != "ADARE":
            algorithm_coverage = coverage_metric(item["final_front"], paired_adare["final_front"])
            adare_coverage = coverage_metric(paired_adare["final_front"], item["final_front"])
        raw_runs.append(
            {
                "case_id": case["case_id"],
                "study": case["study"],
                "benchmark": case["benchmark"],
                "task_count": item["task_count"],
                "resource_scenario": case["resource_scenario"],
                "run": item["run"],
                "seed": item["seed"],
                "algorithm": item["algorithm"],
                **final_metrics,
                "algorithm_coverage_of_adare": algorithm_coverage,
                "adare_coverage_of_algorithm": adare_coverage,
                "objective_evaluations": item["telemetry"][-1]["objective_evaluations"],
                "construction_seconds": item["construction_seconds"],
                "run_seconds": item["run_seconds"],
                "total_seconds": item["construction_seconds"] + item["run_seconds"],
                "peak_process_tree_rss_mib": item["peak_rss_mib"],
                "peak_incremental_rss_mib": item["peak_incremental_rss_mib"],
            }
        )
    return generation_rows, raw_runs


def build_pairwise_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create report-compatible paired comparisons for every diagnostic case."""
    directions = {
        "hv": "max",
        "igd": "min",
        "spacing": "min",
        "epsilon": "min",
        "coverage": "max",
        "time": "min",
        "peak_incremental_rss_mib": "min",
    }
    out: list[dict[str, Any]] = []
    cases = sorted({str(row["case_id"]) for row in run_rows})
    for case_id in cases:
        case_rows = [row for row in run_rows if row["case_id"] == case_id]
        runs = sorted({int(row["run"]) for row in case_rows})
        for run in runs:
            paired = [row for row in case_rows if int(row["run"]) == run]
            adare = next(row for row in paired if row["algorithm"] == "ADARE")
            for baseline in ("NSGA-III", "QL-NSGA-III"):
                other = next(row for row in paired if row["algorithm"] == baseline)
                for metric, direction in directions.items():
                    if metric == "coverage":
                        adare_value = other["adare_coverage_of_algorithm"]
                        baseline_value = other["algorithm_coverage_of_adare"]
                    elif metric == "time":
                        adare_value = adare["total_seconds"]
                        baseline_value = other["total_seconds"]
                    else:
                        adare_value = adare[metric]
                        baseline_value = other[metric]
                    out.append(
                        {
                            "benchmark": case_id,
                            "run": run,
                            "seed": adare["seed"],
                            "baseline": baseline,
                            "metric": metric,
                            "direction": direction,
                            "adare_value": adare_value,
                            "baseline_value": baseline_value,
                        }
                    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted scaling and resource diagnostics.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260219)
    parser.add_argument("--memory-sample-interval", type=float, default=0.05)
    parser.add_argument("--output-dir", default="output/major_revision/scaling_diagnostics_r10_g30")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios, scenario_manifest = resource_scenarios()
    cases = experiment_cases()
    manifest = {
        "runs": args.runs,
        "generations": args.generations,
        "population_size": args.population_size,
        "seed": args.seed,
        "algorithms": list(ALGORITHMS),
        "memory_measure": (
            "Peak process-tree resident set size sampled every "
            f"{args.memory_sample_interval:.3f} seconds; incremental RSS is relative to the pre-construction sample"
        ),
        "resource_scenarios": scenario_manifest,
        "cases": cases,
    }
    (output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    all_generation_rows: list[dict[str, Any]] = []
    all_run_rows: list[dict[str, Any]] = []
    for case in cases:
        generation_rows, run_rows = run_case(case, scenarios[case["resource_scenario"]], args)
        all_generation_rows.extend(generation_rows)
        all_run_rows.extend(run_rows)
        write_csv(output_dir / case["case_id"] / "generation_metrics.csv", generation_rows)
        write_csv(output_dir / case["case_id"] / "run_metrics.csv", run_rows)

    budget_rows = build_budget_rows(all_generation_rows)
    pairwise_rows = build_pairwise_rows(all_run_rows)
    write_csv(output_dir / "scaling_generation_metrics.csv", all_generation_rows)
    write_csv(output_dir / "scaling_run_metrics.csv", all_run_rows)
    write_csv(output_dir / "scaling_budget_checkpoints.csv", budget_rows)
    write_csv(output_dir / "scaling_diagnostics_extended_pairwise_metrics.csv", pairwise_rows)
    plot_anytime(all_generation_rows, output_dir, "objective_evaluations")
    plot_anytime(all_generation_rows, output_dir, "elapsed_seconds")
    plot_memory(all_run_rows, output_dir)
    plot_resource_sensitivity(all_run_rows, output_dir)
    print(f"\nDiagnostics complete: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
