from __future__ import annotations

"""Run ADARE and display an evolution dashboard."""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from algorithms import AdareAlgorithm
from evaluation import coverage_metric, filter_non_dominated, quality_indicators
from main import build_initial_population, load_json, write_csv
from problem import build_nodes, load_environments, load_tasks, topological_sort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch an ADARE evolution dashboard.")
    parser.add_argument("--benchmark", default="Montage_25")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--output-dir", default="output/live_view")
    parser.add_argument("--save-only", action="store_true", help="Save the dashboard without opening a window.")
    return parser.parse_args()


def metric_rows(
    benchmark: str,
    objective_names: Sequence[str],
    history: np.ndarray,
    snapshots: Sequence[np.ndarray],
) -> List[Dict[str, Any]]:
    reference = filter_non_dominated(np.vstack([np.asarray(s, dtype=float) for s in snapshots]))
    rows: List[Dict[str, Any]] = []
    for generation, values in enumerate(history):
        snapshot = np.asarray(snapshots[min(generation, len(snapshots) - 1)], dtype=float)
        front = filter_non_dominated(snapshot)
        q = quality_indicators(front, reference)
        row: Dict[str, Any] = {
            "benchmark": benchmark,
            "generation": generation,
            "front_size": int(len(front)),
            "coverage_to_reference": coverage_metric(front, reference),
            "hv": q["hv"],
            "igd": q["igd"],
            "spacing": q["spacing"],
            "epsilon": q["epsilon"],
        }
        row.update({f"{name}_best": float(values[idx]) for idx, name in enumerate(objective_names)})
        rows.append(row)
    return rows


def plot_dashboard(
    benchmark: str,
    objective_names: Sequence[str],
    rows: Sequence[Dict[str, Any]],
    snapshots: Sequence[np.ndarray],
    elapsed: float,
    output_path: Path,
    save_only: bool,
) -> None:
    generations = np.asarray([row["generation"] for row in rows], dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), dpi=140)
    fig.suptitle(f"ADARE evolution | {benchmark} | {len(generations) - 1} generations | {elapsed:.2f}s")

    ax = axes[0, 0]
    for name in objective_names:
        values = np.asarray([row[f"{name}_best"] for row in rows], dtype=float)
        normalized = values / max(values[0], 1e-12)
        ax.plot(generations, normalized, marker="o", markersize=2.5, linewidth=1.5, label=name)
    ax.set_title("Convergence of objective best values")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized best value")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for metric in ("hv", "igd", "spacing", "epsilon", "coverage_to_reference"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        scale = np.nanmax(np.abs(values[finite]))
        normalized = values / max(scale, 1e-12)
        ax.plot(generations, normalized, marker="o", markersize=2.5, linewidth=1.5, label=metric)
    ax.set_title("Quality metric evolution")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Metric value / max abs")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    if len(objective_names) >= 2:
        stride = max(1, len(snapshots) // 8)
        selected = list(range(0, len(snapshots), stride))
        if selected[-1] != len(snapshots) - 1:
            selected.append(len(snapshots) - 1)
        for idx in selected:
            front = filter_non_dominated(np.asarray(snapshots[idx], dtype=float))
            alpha = 0.22 + 0.65 * (idx / max(1, len(snapshots) - 1))
            ax.scatter(front[:, 0], front[:, 1], s=18, alpha=alpha, label=f"g{idx}")
        ax.set_xlabel(objective_names[0])
        ax.set_ylabel(objective_names[1])
        ax.set_title("Pareto-front projection over generations")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1, 1]
    final = rows[-1]
    ax.axis("off")
    lines = [
        f"Current generation: {final['generation']}",
        f"Final front size: {final['front_size']}",
        f"HV: {final['hv']:.4f}",
        f"IGD: {final['igd']:.4f}",
        f"Spacing: {final['spacing']:.4f}",
        f"Epsilon: {final['epsilon']:.4f}",
        f"Coverage to reference: {final['coverage_to_reference']:.4f}",
        "",
        "Best objective values:",
    ]
    lines.extend(f"{name}: {final[f'{name}_best']:.6g}" for name in objective_names)
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Dashboard saved: {output_path}")
    if not save_only:
        plt.show()
    plt.close(fig)


def main() -> int:
    args = parse_args()
    config_dir = ROOT / "config"
    main_config = load_json(config_dir / "main_config.json")
    adare_config = load_json(config_dir / "adare_config.json")
    adare_config["capture_generation_snapshots"] = True

    objective_names = list(main_config["objectives"])
    shared = dict(main_config["shared_parameters"])
    shared["population_size"] = int(args.population_size)
    shared["generations"] = int(args.generations)
    seed = int(args.seed if args.seed is not None else main_config["general_parameters"]["base_seed"])

    tasks = load_tasks(args.benchmark)
    nodes = build_nodes(load_environments())
    order = topological_sort(tasks)
    init_pop = build_initial_population(seed, int(shared["population_size"]), len(tasks), len(nodes))

    algorithm = AdareAlgorithm(
        shared_config=shared,
        algorithm_config=adare_config,
        nodes=nodes,
        tasks=tasks,
        topological_order=order,
        objective_names=objective_names,
        seed=seed,
        initial_population=init_pop,
    )

    print(
        f"Live dashboard run | benchmark={args.benchmark} | "
        f"pop={shared['population_size']} | gen={shared['generations']} | seed={seed}"
    )
    started = time.perf_counter()
    result = algorithm.run()
    elapsed = time.perf_counter() - started

    history = np.asarray(result["history"], dtype=float)
    snapshots = result.get("generation_snapshots") or [AdareAlgorithm.population_to_array(result["population"])]
    rows = metric_rows(args.benchmark, objective_names, history, snapshots)

    out_dir = ROOT / args.output_dir
    fields = [
        "benchmark",
        "generation",
        "front_size",
        "coverage_to_reference",
        "hv",
        "igd",
        "spacing",
        "epsilon",
        *[f"{name}_best" for name in objective_names],
    ]
    write_csv(out_dir / f"{args.benchmark}_live_metrics.csv", fields, rows)
    plot_dashboard(
        benchmark=args.benchmark,
        objective_names=objective_names,
        rows=rows,
        snapshots=snapshots,
        elapsed=elapsed,
        output_path=out_dir / f"{args.benchmark}_live_dashboard.png",
        save_only=args.save_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
