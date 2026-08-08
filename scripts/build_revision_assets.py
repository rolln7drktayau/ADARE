from __future__ import annotations

"""Build publication tables and figures directly from major-revision CSVs."""

import argparse
import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # type: ignore  # noqa: E402


CORE = {"hv", "igd", "spacing", "epsilon", "coverage"}
COLORS = {"ADARE": "#D55E00", "NSGA-III": "#0072B2", "QL-NSGA-III": "#E69F00"}
OPERATORS = ("one_point", "two_point", "uniform_0_5", "uniform_0_8")
OPERATOR_LABELS = {
    "one_point": "One-point",
    "two_point": "Two-point",
    "uniform_0_5": "Uniform $p=0.5$",
    "uniform_0_8": "Uniform $p=0.8$",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def mean_ci(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, 0.0
    # t critical is 2.262 for n=10 and close to 2 for n=20; using 2.262
    # is conservative for every table/figure generated here.
    half = 2.262 * float(np.std(arr, ddof=1)) / math.sqrt(len(arr))
    return mean, half


def write_simple_table(path: Path, column_spec: str, header: str, rows: Sequence[str]) -> None:
    text = [
        "\\begin{tabular}{" + column_spec + "}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        *[row + " \\\\" for row in rows],
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")


def build_protocol_tables(stats: list[dict[str, str]], generated: Path) -> None:
    protocols = [
        ("small_all_algorithms_r20_g70", "Small, 70 gen."),
        ("large_1000_r20_g50", "1000 tasks, 50 gen."),
        ("large_1000_r20_g100", "1000 tasks, 100 gen."),
        ("large_3000_r10_g20", "3000-task class, 20 gen."),
    ]
    rows = []
    for prefix, label in protocols:
        group = [row for row in stats if row["source_file"].startswith(prefix) and row["metric"] in CORE]
        wins = sum(float(row["paired_improvement_mean"]) > 0 for row in group)
        holm = sum(
            float(row["paired_improvement_mean"]) > 0 and float(row["holm_p"]) < 0.05
            for row in group
        )
        ci = sum(float(row["paired_improvement_ci95_low"]) > 0 for row in group)
        rows.append(f"{label} & {len(group)} & {wins} & {holm} & {ci}")
    write_simple_table(
        generated / "protocol_summary.tex",
        "lrrrr",
        "Protocol & Tests & ADARE wins & Holm-significant wins & Positive 95\\% CI",
        rows,
    )

    long_rows = []
    for prefix, label in protocols[1:]:
        group = [row for row in stats if row["source_file"].startswith(prefix) and row["metric"] in CORE]
        baselines = sorted({row["baseline"] for row in group})
        for baseline in baselines:
            subset = [row for row in group if row["baseline"] == baseline]
            wins = sum(float(row["paired_improvement_mean"]) > 0 for row in subset)
            holm = sum(
                float(row["paired_improvement_mean"]) > 0 and float(row["holm_p"]) < 0.05
                for row in subset
            )
            long_rows.append(
                f"{label} & {latex_escape(baseline)} & {wins}/{len(subset)} & {holm}/{len(subset)}"
            )
    write_simple_table(
        generated / "long_budget_summary.tex",
        "llrr",
        "Protocol & Baseline & ADARE wins & Holm-significant wins",
        long_rows,
    )


def build_ablation_tables(stats: list[dict[str, str]], generated: Path) -> None:
    specs = [
        ("controller_ablation_r20_g70", "controller_ablation_summary.tex"),
        ("reward_sensitivity_r20_g70", "reward_sensitivity_summary.tex"),
    ]
    for prefix, filename in specs:
        group = [row for row in stats if row["source_file"].startswith(prefix) and row["metric"] in CORE]
        rows = []
        for baseline in sorted({row["baseline"] for row in group}):
            subset = [row for row in group if row["baseline"] == baseline]
            wins = sum(float(row["paired_improvement_mean"]) > 0 for row in subset)
            raw = sum(
                float(row["paired_improvement_mean"]) > 0 and float(row["wilcoxon_one_sided_p"]) < 0.05
                for row in subset
            )
            holm = sum(
                float(row["paired_improvement_mean"]) > 0 and float(row["holm_p"]) < 0.05
                for row in subset
            )
            rows.append(f"{latex_escape(baseline)} & {wins}/{len(subset)} & {raw} & {holm}")
        write_simple_table(
            generated / filename,
            "lrrr",
            "Comparator/variant & ADARE wins & Raw $p<0.05$ & Holm $p<0.05$",
            rows,
        )


def build_diagnostic_tables(
    stats: list[dict[str, str]], run_rows: list[dict[str, str]], generated: Path
) -> None:
    diagnostic = [
        row for row in stats if row["source_file"].startswith("scaling_diagnostics_r10_g30") and row["metric"] in CORE
    ]
    rows = []
    for key in sorted({(row["benchmark"], row["baseline"]) for row in diagnostic}):
        subset = [row for row in diagnostic if (row["benchmark"], row["baseline"]) == key]
        wins = sum(float(row["paired_improvement_mean"]) > 0 for row in subset)
        holm = sum(
            float(row["paired_improvement_mean"]) > 0 and float(row["holm_p"]) < 0.05
            for row in subset
        )
        rows.append(f"{latex_escape(key[0])} & {latex_escape(key[1])} & {wins}/5 & {holm}/5")
    write_simple_table(
        generated / "scaling_diagnostic_summary.tex",
        "llrr",
        "Case & Baseline & ADARE wins & Holm-significant wins",
        rows,
    )

    resource_rows = []
    for scenario in ("nominal", "compute_skew", "network_scarce"):
        for algorithm in COLORS:
            subset = [
                row
                for row in run_rows
                if row["resource_scenario"] == scenario
                and row["algorithm"] == algorithm
                and int(row["task_count"]) == 1000
            ]
            hv_mean, hv_ci = mean_ci(float(row["hv"]) for row in subset)
            igd_mean, igd_ci = mean_ci(float(row["igd"]) for row in subset)
            resource_rows.append(
                f"{latex_escape(scenario)} & {latex_escape(algorithm)} & "
                f"{hv_mean:.3f} $\\pm$ {hv_ci:.3f} & {igd_mean:.3f} $\\pm$ {igd_ci:.3f}"
            )
    write_simple_table(
        generated / "resource_sensitivity.tex",
        "llrr",
        "Resource scenario & Algorithm & HV (mean $\\pm$ 95\\% CI) & IGD (mean $\\pm$ 95\\% CI)",
        resource_rows,
    )

    memory_rows = []
    nominal = [row for row in run_rows if row["study"] in {"scaling", "scaling_resource"}]
    for task_count in sorted({int(row["task_count"]) for row in nominal}):
        for algorithm in COLORS:
            subset = [
                row for row in nominal if int(row["task_count"]) == task_count and row["algorithm"] == algorithm
            ]
            mem_mean, mem_ci = mean_ci(float(row["peak_incremental_rss_mib"]) for row in subset)
            time_mean, time_ci = mean_ci(float(row["total_seconds"]) for row in subset)
            memory_rows.append(
                f"{task_count} & {latex_escape(algorithm)} & {time_mean:.2f} $\\pm$ {time_ci:.2f} & "
                f"{mem_mean:.1f} $\\pm$ {mem_ci:.1f}"
            )
    write_simple_table(
        generated / "runtime_memory_scaling.tex",
        "rlrr",
        "Tasks & Algorithm & Time (s) & Incremental peak RSS (MiB)",
        memory_rows,
    )


def build_controller_tables(reports: Path, generated: Path) -> None:
    corr = [
        row
        for row in read_csv(reports / "reward_survival_correlation.csv")
        if row["source_file"].startswith("adare_controller_traces_1000_r20_g100")
    ]
    rows = [
        f"{latex_escape(row['benchmark'])} & {row['run_n']} & "
        f"{float(row['mean_run_spearman_rho']):.3f} & "
        f"[{float(row['mean_run_rho_ci95_low']):.3f}, {float(row['mean_run_rho_ci95_high']):.3f}] & "
        f"{float(row['wilcoxon_run_rho_one_sided_p']):.2e} & {row['positive_run_correlations']}/{row['run_n']}"
        for row in corr
    ]
    write_simple_table(
        generated / "reward_survival.tex",
        "lrrrrr",
        "Benchmark & Runs & Mean $\\rho$ & 95\\% CI & One-sided $p$ & Positive runs",
        rows,
    )

    runtime = [
        row
        for row in read_csv(reports / "runtime_breakdown.csv")
        if row["source_file"].startswith("adare_controller_traces_1000_r20_g100")
        and row["component"]
        in {"controller_selection", "reward_and_controller_update", "archive_operations", "fitness_evaluation"}
    ]
    runtime_rows = [
        f"{latex_escape(row['benchmark'])} & {latex_escape(row['component'])} & "
        f"{float(row['mean_seconds']):.3f} & {float(row['mean_percent_of_total']):.2f}\\%"
        for row in sorted(runtime, key=lambda item: (item["benchmark"], item["component"]))
    ]
    write_simple_table(
        generated / "runtime_breakdown.tex",
        "llrr",
        "Benchmark & Component & Mean seconds & Share of runtime",
        runtime_rows,
    )


def build_objective_correlations(root: Path, generated: Path) -> list[dict[str, Any]]:
    files = sorted((root / "adare_controller_traces_1000_r20_g100").rglob("*_adare_only_front.csv"))
    detail: list[dict[str, Any]] = []
    objectives = ("makespan", "latency", "cost", "energy")
    for path in files:
        rows = read_csv(path)
        by_run: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_run[int(row["run"])].append(row)
        for run, run_rows in sorted(by_run.items()):
            for left, right in combinations(objectives, 2):
                a = np.asarray([float(row[left]) for row in run_rows], dtype=float)
                b = np.asarray([float(row[right]) for row in run_rows], dtype=float)
                rho = float(spearmanr(a, b).statistic) if len(a) >= 3 else float("nan")
                detail.append(
                    {
                        "benchmark": run_rows[0]["benchmark"],
                        "run": run,
                        "objective_a": left,
                        "objective_b": right,
                        "n_solutions": len(run_rows),
                        "spearman_rho": rho,
                    }
                )
    write_csv(generated / "objective_correlations_by_run.csv", detail)
    table_rows = []
    keys = sorted({(row["benchmark"], row["objective_a"], row["objective_b"]) for row in detail})
    for benchmark, left, right in keys:
        values = np.asarray(
            [
                float(row["spearman_rho"])
                for row in detail
                if (row["benchmark"], row["objective_a"], row["objective_b"]) == (benchmark, left, right)
            ],
            dtype=float,
        )
        q1, med, q3 = np.nanpercentile(values, [25, 50, 75])
        table_rows.append(
            f"{latex_escape(benchmark)} & {latex_escape(left)}--{latex_escape(right)} & "
            f"{med:.2f} & [{q1:.2f}, {q3:.2f}]"
        )
    write_simple_table(
        generated / "objective_correlations.tex",
        "llrr",
        "Benchmark & Objective pair & Median $\\rho$ & IQR",
        table_rows,
    )
    return detail


def plot_controller(reports: Path, figures: Path) -> None:
    rows = [
        row
        for row in read_csv(reports / "controller_behavior.csv")
        if row["source_file"].startswith("adare_controller_traces_1000_r20_g100")
    ]
    benchmarks = sorted({row["benchmark"] for row in rows})
    phases = ("early", "middle", "late")
    fig, axes = plt.subplots(len(benchmarks), 2, figsize=(12, 7.2))
    for row_idx, benchmark in enumerate(benchmarks):
        benchmark_rows = [row for row in rows if row["benchmark"] == benchmark]
        for op_idx, operator in enumerate(OPERATORS):
            shares = []
            rewards = []
            for phase in phases:
                phase_rows = [row for row in benchmark_rows if row["context"].startswith(phase + ":")]
                total = sum(int(row["uses"]) for row in phase_rows)
                op_rows = [row for row in phase_rows if row["operator"] == operator]
                uses = sum(int(row["uses"]) for row in op_rows)
                weighted_reward = sum(int(row["uses"]) * float(row["mean_reward"]) for row in op_rows)
                shares.append(100.0 * uses / max(1, total))
                rewards.append(weighted_reward / max(1, uses))
            axes[row_idx, 0].plot(phases, shares, marker="o", label=OPERATOR_LABELS[operator])
            axes[row_idx, 1].plot(phases, rewards, marker="o", label=OPERATOR_LABELS[operator])
        axes[row_idx, 0].set_title(f"{benchmark}: selection share")
        axes[row_idx, 0].set_ylabel("Selections (\\%)")
        axes[row_idx, 1].set_title(f"{benchmark}: mean immediate reward")
        axes[row_idx, 1].set_ylabel("Reward")
        for ax in axes[row_idx]:
            ax.grid(alpha=0.25, linestyle="--")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(figures / "revision_controller_behavior.png", dpi=300)
    plt.close(fig)


def plot_diagnostics(run_rows: list[dict[str, str]], generation_rows: list[dict[str, str]], figures: Path) -> None:
    scale_cases = (
        "scale_montage_25_nominal",
        "scale_montage_1000_nominal",
        "scale_montage_3000_nominal",
    )
    for coordinate, suffix, xlabel in (
        ("objective_evaluations", "evaluations", "Objective evaluations"),
        ("elapsed_seconds", "time", "Elapsed seconds"),
    ):
        fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.2))
        for row_idx, case_id in enumerate(scale_cases):
            case_rows = [row for row in generation_rows if row["case_id"] == case_id]
            for col_idx, (metric, ylabel) in enumerate((("hv", "HV"), ("igd", "IGD"))):
                ax = axes[row_idx, col_idx]
                for algorithm in COLORS:
                    algo_rows = [row for row in case_rows if row["algorithm"] == algorithm]
                    generations = sorted({int(row["generation"]) for row in algo_rows})
                    x, y, ci = [], [], []
                    for generation in generations:
                        subset = [row for row in algo_rows if int(row["generation"]) == generation]
                        x.append(float(np.mean([float(row[coordinate]) for row in subset])))
                        mean, half = mean_ci(float(row[metric]) for row in subset)
                        y.append(mean)
                        ci.append(half)
                    x_arr, y_arr, ci_arr = map(np.asarray, (x, y, ci))
                    ax.plot(x_arr, y_arr, color=COLORS[algorithm], label=algorithm, linewidth=2.0)
                    ax.fill_between(x_arr, y_arr - ci_arr, y_arr + ci_arr, color=COLORS[algorithm], alpha=0.12)
                ax.set_title(case_rows[0]["benchmark"])
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.25, linestyle="--")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(figures / f"revision_convergence_vs_{suffix}.png", dpi=300)
        plt.close(fig)

    nominal = [row for row in run_rows if row["study"] in {"scaling", "scaling_resource"}]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for algorithm in COLORS:
        subset = [row for row in nominal if row["algorithm"] == algorithm]
        counts = sorted({int(row["task_count"]) for row in subset})
        means, cis = [], []
        for count in counts:
            mean, half = mean_ci(
                float(row["peak_incremental_rss_mib"]) for row in subset if int(row["task_count"]) == count
            )
            means.append(mean)
            cis.append(half)
        ax.errorbar(counts, means, yerr=cis, marker="o", capsize=3, label=algorithm, color=COLORS[algorithm])
    ax.set_xscale("log")
    ax.set_xlabel("Workflow tasks (log scale)")
    ax.set_ylabel("Incremental peak process-tree RSS (MiB)")
    ax.set_title("Runtime-memory scaling (mean and 95\\% CI, 10 runs)")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "revision_memory_scaling.png", dpi=300)
    plt.close(fig)

    resource = [row for row in run_rows if int(row["task_count"]) == 1000]
    scenarios = ("nominal", "compute_skew", "network_scarce")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    width = 0.24
    x = np.arange(len(scenarios), dtype=float)
    for ax, metric, title in ((axes[0], "hv", "Final HV"), (axes[1], "igd", "Final IGD")):
        for idx, algorithm in enumerate(COLORS):
            means, cis = [], []
            for scenario in scenarios:
                mean, half = mean_ci(
                    float(row[metric])
                    for row in resource
                    if row["resource_scenario"] == scenario and row["algorithm"] == algorithm
                )
                means.append(mean)
                cis.append(half)
            ax.bar(
                x + (idx - 1) * width,
                means,
                width=width,
                yerr=cis,
                capsize=3,
                label=algorithm,
                color=COLORS[algorithm],
            )
        ax.set_xticks(x, ("Nominal", "Compute-skew", "Network-scarce"), rotation=12)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(figures / "revision_resource_sensitivity.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final major-revision manuscript assets.")
    parser.add_argument("--root", default="output/major_revision")
    parser.add_argument("--paper-dir", default="papers")
    parser.add_argument("--figure-dir", default="Figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    reports = root / "reports"
    paper_dir = Path(args.paper_dir)
    generated = paper_dir / "generated"
    figures = Path(args.figure_dir)
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    stats = read_csv(reports / "major_revision_statistics.csv")
    diagnostics = root / "scaling_diagnostics_r10_g30"
    run_rows = read_csv(diagnostics / "scaling_run_metrics.csv")
    generation_rows = read_csv(diagnostics / "scaling_generation_metrics.csv")
    build_protocol_tables(stats, generated)
    build_ablation_tables(stats, generated)
    build_diagnostic_tables(stats, run_rows, generated)
    build_controller_tables(reports, generated)
    build_objective_correlations(root, generated)
    plot_controller(reports, figures)
    plot_diagnostics(run_rows, generation_rows, figures)
    print(generated.resolve())
    print(figures.resolve())


if __name__ == "__main__":
    main()
