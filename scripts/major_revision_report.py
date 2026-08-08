from __future__ import annotations

"""Collect major-revision experiment outputs into reviewer-facing reports."""

import argparse
import csv
import math
import hashlib
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy.stats import spearmanr, wilcoxon  # type: ignore


METRIC_DIRECTIONS = {
    "hv": "max",
    "igd": "min",
    "spacing": "min",
    "epsilon": "min",
    "coverage": "max",
    "time": "min",
    "makespan_best": "min",
    "latency_best": "min",
    "cost_best": "min",
    "energy_best": "min",
}

CORE_METRICS = ["hv", "igd", "spacing", "epsilon", "coverage"]
RUNTIME_PREFIX = "runtime_"
RUNTIME_SUFFIX = "_seconds"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def floats(rows: list[dict[str, str]], metric: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row.get(metric, "nan"))
        except ValueError:
            value = float("nan")
        if math.isfinite(value):
            values.append(value)
    return values


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"), "iqr": float("nan")}
    arr = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(median(values)),
        "iqr": float(q3 - q1),
    }


def rank_biserial_paired(a: list[float], b: list[float], direction: str) -> float:
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if direction == "min":
        diffs = -diffs
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[np.abs(diffs) > 1e-12]
    n = len(diffs)
    if n == 0:
        return 0.0
    order = np.argsort(np.abs(diffs))
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    w_pos = float(np.sum(ranks[diffs > 0]))
    w_neg = float(np.sum(ranks[diffs < 0]))
    denom = n * (n + 1) / 2.0
    return float((w_pos - w_neg) / denom)


def paired_test(a: list[float], b: list[float], direction: str) -> tuple[float, float]:
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    valid = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if len(valid) < 2:
        return float("nan"), float("nan")
    av = [x for x, _ in valid]
    bv = [y for _, y in valid]
    if np.allclose(av, bv):
        return 1.0, 0.0
    alternative = "greater" if direction == "max" else "less"
    try:
        p = float(wilcoxon(av, bv, alternative=alternative, zero_method="wilcox", method="auto").pvalue)
    except ValueError:
        p = float("nan")
    return p, rank_biserial_paired(av, bv, direction)


def paired_difference_ci(
    a: list[float],
    b: list[float],
    direction: str,
    key: str,
    confidence: float = 0.95,
    resamples: int = 10_000,
) -> tuple[float, float, float]:
    """Return a deterministic paired-bootstrap CI for improvement differences.

    Positive differences always favor ADARE, independently of metric direction.
    """
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    valid = np.isfinite(av) & np.isfinite(bv)
    diffs = av[valid] - bv[valid]
    if direction == "min":
        diffs = -diffs
    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(np.mean(diffs))
    if len(diffs) == 1:
        return estimate, estimate, estimate
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(diffs, size=(resamples, len(diffs)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(sampled, [alpha, 1.0 - alpha])
    return estimate, float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = [(idx, p) for idx, p in enumerate(p_values) if math.isfinite(p)]
    m = len(indexed)
    adjusted = [float("nan")] * len(p_values)
    running_max = 0.0
    for rank, (idx, p) in enumerate(sorted(indexed, key=lambda item: item[1]), start=1):
        adj = min(1.0, (m - rank + 1) * p)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def collect_extended_statistics(root: Path) -> list[dict[str, Any]]:
    files = sorted(root.rglob("*_extended_pairwise_metrics.csv"))
    rows_out: list[dict[str, Any]] = []
    for path in files:
        rows = read_csv(path)
        grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(row["benchmark"], row["baseline"], row["metric"], row["direction"])].append(row)
        for (benchmark, baseline, metric, direction), group_rows in sorted(grouped.items()):
            group_rows.sort(key=lambda row: (int(row["run"]), int(row["seed"])))
            adare_values = [float(row["adare_value"]) for row in group_rows]
            base_values = [float(row["baseline_value"]) for row in group_rows]
            ad = describe(adare_values)
            bd = describe(base_values)
            p, rb = paired_test(adare_values, base_values, direction)
            diff, ci_low, ci_high = paired_difference_ci(
                adare_values,
                base_values,
                direction,
                key=f"{path}:{benchmark}:{baseline}:{metric}",
            )
            gain = float("nan")
            if math.isfinite(ad["mean"]) and math.isfinite(bd["mean"]) and abs(bd["mean"]) > 1e-12:
                if direction == "max":
                    gain = (ad["mean"] - bd["mean"]) / abs(bd["mean"]) * 100.0
                else:
                    gain = (bd["mean"] - ad["mean"]) / abs(bd["mean"]) * 100.0
            rows_out.append(
                {
                    "source_file": str(path.relative_to(root)),
                    "benchmark": benchmark,
                    "baseline": baseline,
                    "metric": metric,
                    "direction": direction,
                    "adare_n": ad["n"],
                    "adare_mean": ad["mean"],
                    "adare_std": ad["std"],
                    "adare_median": ad["median"],
                    "adare_iqr": ad["iqr"],
                    "baseline_mean": bd["mean"],
                    "baseline_std": bd["std"],
                    "baseline_median": bd["median"],
                    "baseline_iqr": bd["iqr"],
                    "gain_percent": gain,
                    "paired_improvement_mean": diff,
                    "paired_improvement_ci95_low": ci_low,
                    "paired_improvement_ci95_high": ci_high,
                    "wilcoxon_one_sided_p": p,
                    "holm_p": float("nan"),
                    "rank_biserial_paired": rb,
                }
            )

    # Treat all baseline/metric hypotheses within one protocol and benchmark as
    # a correction family. This avoids mixing unrelated quick/full protocols.
    families: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows_out):
        families[(str(row["source_file"]), str(row["benchmark"]))].append(idx)
    for indices in families.values():
        adjusted = holm_adjust([float(rows_out[idx]["wilcoxon_one_sided_p"]) for idx in indices])
        for idx, adj in zip(indices, adjusted):
            rows_out[idx]["holm_p"] = adj
    return rows_out


def collect_controller_behavior(root: Path) -> list[dict[str, Any]]:
    files = sorted(root.rglob("*controller_trace.csv"))
    aggregate: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(
        lambda: {"uses": 0.0, "reward": 0.0}
    )
    for path in files:
        rows = read_csv(path)
        # Traces created before the corrected protocol lack survival outcome
        # instrumentation and must not be mixed with current evidence.
        if not rows or "survival_fraction" not in rows[0]:
            continue
        source_file = str(path.relative_to(root))
        for row in rows:
            key = (
                source_file,
                row.get("benchmark", ""),
                row.get("context", ""),
                row.get("operator", ""),
            )
            aggregate[key]["uses"] += 1.0
            try:
                aggregate[key]["reward"] += float(row.get("reward", "0"))
            except ValueError:
                pass
    out = []
    for (source_file, benchmark, context, operator), stats in sorted(aggregate.items()):
        uses = int(stats["uses"])
        out.append(
            {
                "source_file": source_file,
                "benchmark": benchmark,
                "context": context,
                "operator": operator,
                "uses": uses,
                "mean_reward": stats["reward"] / max(1, uses),
            }
        )
    return out


def collect_reward_survival_correlation(root: Path) -> list[dict[str, Any]]:
    """Relate immediate reward to survival using independent runs as replicates.

    The pooled event-level coefficient is retained as a descriptive quantity,
    but inference is performed over one Spearman coefficient per run. Controller
    decisions from the same evolutionary run are not independent observations.
    """
    out: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*controller_trace.csv")):
        rows = read_csv(path)
        pairs: list[tuple[float, float]] = []
        pairs_by_run: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in rows:
            try:
                reward = float(row["reward"])
                survival = float(row["survival_fraction"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(reward) and math.isfinite(survival):
                pairs.append((reward, survival))
                pairs_by_run[row.get("run", "1")].append((reward, survival))
        if len(pairs) < 3:
            continue
        rewards = np.asarray([pair[0] for pair in pairs], dtype=float)
        survivals = np.asarray([pair[1] for pair in pairs], dtype=float)
        pooled = spearmanr(rewards, survivals)
        run_rhos: list[float] = []
        for run_pairs in pairs_by_run.values():
            if len(run_pairs) < 3:
                continue
            run_rewards = np.asarray([pair[0] for pair in run_pairs], dtype=float)
            run_survivals = np.asarray([pair[1] for pair in run_pairs], dtype=float)
            if len(np.unique(run_rewards)) < 2 or len(np.unique(run_survivals)) < 2:
                continue
            rho = float(spearmanr(run_rewards, run_survivals).statistic)
            if math.isfinite(rho):
                run_rhos.append(rho)
        if not run_rhos:
            continue
        zeros = [0.0] * len(run_rhos)
        mean_rho, ci_low, ci_high = paired_difference_ci(
            run_rhos,
            zeros,
            "max",
            key=f"reward-survival:{path}",
        )
        try:
            run_p = float(
                wilcoxon(run_rhos, alternative="greater", zero_method="wilcox", method="auto").pvalue
            )
        except ValueError:
            run_p = float("nan")
        out.append(
            {
                "source_file": str(path.relative_to(root)),
                "benchmark": rows[0].get("benchmark", "") if rows else "",
                "event_n": len(pairs),
                "run_n": len(run_rhos),
                "pooled_spearman_rho_descriptive": float(pooled.statistic),
                "mean_run_spearman_rho": mean_rho,
                "median_run_spearman_rho": float(np.median(run_rhos)),
                "mean_run_rho_ci95_low": ci_low,
                "mean_run_rho_ci95_high": ci_high,
                "wilcoxon_run_rho_one_sided_p": run_p,
                "rank_biserial_run_rho_vs_zero": rank_biserial_paired(run_rhos, zeros, "max"),
                "positive_run_correlations": sum(rho > 0.0 for rho in run_rhos),
                "mean_reward": float(np.mean(rewards)),
                "mean_survival_fraction": float(np.mean(survivals)),
            }
        )
    return out


def collect_runtime_breakdown(root: Path) -> list[dict[str, Any]]:
    """Aggregate ADARE runtime components by protocol and benchmark."""
    out: list[dict[str, Any]] = []
    files = sorted(root.rglob("*_run_metrics.csv"))
    for path in files:
        rows = read_csv(path)
        adare_rows = [row for row in rows if row.get("algorithm", "ADARE") == "ADARE"]
        if not adare_rows:
            continue
        component_columns = [
            name
            for name in adare_rows[0]
            if name.startswith(RUNTIME_PREFIX) and name.endswith(RUNTIME_SUFFIX)
        ]
        for component_column in component_columns:
            values = floats(adare_rows, component_column)
            totals = floats(adare_rows, "time")
            stats = describe(values)
            total_mean = float(np.mean(totals)) if totals else float("nan")
            out.append(
                {
                    "source_file": str(path.relative_to(root)),
                    "benchmark": adare_rows[0].get("benchmark", ""),
                    "component": component_column[len(RUNTIME_PREFIX) : -len(RUNTIME_SUFFIX)],
                    "n": stats["n"],
                    "mean_seconds": stats["mean"],
                    "std_seconds": stats["std"],
                    "median_seconds": stats["median"],
                    "mean_percent_of_total": (
                        stats["mean"] / total_mean * 100.0
                        if math.isfinite(stats["mean"]) and math.isfinite(total_mean) and total_mean > 0.0
                        else float("nan")
                    ),
                }
            )
    return out


def collect_evaluation_budget(root: Path) -> list[dict[str, Any]]:
    """Summarize actual objective-function computations, including local repair."""
    out: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*_run_metrics.csv")):
        rows = read_csv(path)
        if not rows or "objective_evaluations" not in rows[0]:
            continue
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(row.get("benchmark", ""), row.get("algorithm", "ADARE"))].append(row)
        for (benchmark, algorithm), group_rows in sorted(grouped.items()):
            stats = describe(floats(group_rows, "objective_evaluations"))
            out.append(
                {
                    "source_file": str(path.relative_to(root)),
                    "benchmark": benchmark,
                    "algorithm": algorithm,
                    "n": stats["n"],
                    "mean_objective_evaluations": stats["mean"],
                    "std_objective_evaluations": stats["std"],
                    "median_objective_evaluations": stats["median"],
                }
            )
    return out


def write_review_matrix(path: Path) -> None:
    lines = [
        "# SwEvo Major Revision Response Matrix",
        "",
        "Final evidence-to-manuscript checklist for the revised submission.",
        "",
        "| Reviewer issue | Evidence to generate | Expected manuscript change | Status |",
        "|---|---|---|---|",
        "| R1.1/R4.2/R4.3/R5 reward and context underspecified | Method audit plus exact equations from code | Operational definitions, thresholds, reward and update added | addressed |",
        "| R1.2/R2.3/R4.1 ablation too weak | Incremental plus controlled ablations | Tables added; weak isolated-controller evidence discussed | addressed |",
        "| R1.3/R5 runtime overhead unclear | Instrumented runtime breakdown | Large-instance component shares and scaling reported | addressed |",
        "| R1.4/R4.8/R5 reward sensitivity | Seven 20-run sensitivity variants | Weights, alpha and clipping analysis added | addressed |",
        "| R1.5/R4.5 fairness protocol | Comparable survival-set scoring and budget export | Seeds, budgets, cache and final scoring documented | addressed |",
        "| R1.8 controller visualization | 20-run 1000-task traces | Phase/operator visualization and survival audit added | addressed |",
        "| R2.2/R4.12/R5 shallow large-scale budgets | 50/100-gen 1000-task and 20-gen 3000-task runs | Evaluation/time convergence and cautious claims added | addressed |",
        "| R3 abstract/motivation/comments | Text revision | Abstract/motivation rewritten and algorithms annotated | addressed |",
        "| R4.6 objective model | Model audit and run-level correlations | Evaluator equations, omissions, units and dependence added | addressed |",
        "| R4.7 uncertainty wording | Text audit | Deterministic scope stated; systems uncertainty claims removed | addressed |",
        "| R4.9 statistics incomplete | Complete statistics CSV | Direction, Holm, paired rank-biserial and bootstrap CI added | addressed |",
        "| R4.10 aggregate core gain misleading | Per-metric statistics | Main conclusions use per-indicator evidence | addressed |",
        "| R4.11 adapted baselines | Baseline audit | Adaptations declared and family-level claims removed | addressed |",
        "| R4.13 one resource configuration | Compute-skew and network-scarce runs | Two perturbations added; node-count variation remains a limitation | partially addressed |",
        "| R4.14 related work too long | Text revision | Reduced to three concise subsections | addressed |",
        "| R4.15 modern baselines | Literature/baseline audit | References corrected and claims moderated; exact artifacts not executed | partially addressed |",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_summary(
    root: Path,
    stats_rows: list[dict[str, Any]],
    controller_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    reward_survival_rows: list[dict[str, Any]],
    evaluation_budget_rows: list[dict[str, Any]],
) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SwEvo Major Revision Run Summary",
        "",
        f"Output root: `{root}`",
        "",
        "## Generated Files",
        "",
        f"- Statistics CSV: `{reports / 'major_revision_statistics.csv'}`",
        f"- Controller behavior CSV: `{reports / 'controller_behavior.csv'}`",
        f"- Runtime breakdown CSV: `{reports / 'runtime_breakdown.csv'}`",
        f"- Reward/survival correlation CSV: `{reports / 'reward_survival_correlation.csv'}`",
        f"- Evaluation budget CSV: `{reports / 'evaluation_budget.csv'}`",
        f"- Reviewer matrix: `{reports / 'reviewer_response_matrix.md'}`",
        "",
        "## Completed Evidence Snapshot",
        "",
    ]
    if stats_rows:
        core = [row for row in stats_rows if row["metric"] in CORE_METRICS]
        # Paired improvement is already oriented so that positive always
        # favors ADARE, including comparisons whose baseline mean is zero and
        # whose relative percentage is therefore undefined.
        wins = sum(1 for row in core if float(row["paired_improvement_mean"]) > 0)
        significant = sum(
            1
            for row in core
            if float(row["paired_improvement_mean"]) > 0
            and math.isfinite(float(row["holm_p"]))
            and float(row["holm_p"]) < 0.05
        )
        lines.append(f"- Core metric comparisons found: {len(core)}")
        lines.append(f"- ADARE positive core gains: {wins}/{len(core)}")
        lines.append(f"- Positive core gains significant after Holm correction: {significant}/{len(core)}")
    else:
        lines.append("- No extended run metrics found yet.")
    if controller_rows:
        total_uses = sum(int(row["uses"]) for row in controller_rows)
        lines.append(f"- Controller trace rows aggregated: {len(controller_rows)} context/operator groups, {total_uses} selections.")
    else:
        lines.append("- No controller traces found yet.")
    if runtime_rows:
        lines.append(f"- Runtime component summaries found: {len(runtime_rows)}.")
    else:
        lines.append("- No instrumented runtime breakdown found yet.")
    if reward_survival_rows:
        lines.append(f"- Reward/survival correlation summaries found: {len(reward_survival_rows)}.")
    else:
        lines.append("- No reward/survival trace correlation found yet.")
    if evaluation_budget_rows:
        lines.append(f"- Evaluation-budget summaries found: {len(evaluation_budget_rows)}.")
    else:
        lines.append("- No instrumented evaluation budgets found yet.")
    lines.extend(
        [
            "",
            "## Interpretation Completed",
            "",
            "1. Long-budget 1000-task results were checked; QL-NSGA-III is explicitly identified as competitive.",
            "2. Controlled ablation does not support attributing the full gain to contextual control alone.",
            "3. The manuscript prioritizes per-metric paired evidence over averaged core gain.",
            "4. Phase/operator behavior and run-level reward--survival association are reported.",
        ]
    )
    (reports / "major_revision_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SwEvo major-revision outputs.")
    parser.add_argument("--root", default="output/major_revision")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stats = collect_extended_statistics(root)
    controller = collect_controller_behavior(root)
    runtime = collect_runtime_breakdown(root)
    reward_survival = collect_reward_survival_correlation(root)
    evaluation_budget = collect_evaluation_budget(root)
    write_csv(reports / "major_revision_statistics.csv", stats)
    write_csv(reports / "controller_behavior.csv", controller)
    write_csv(reports / "runtime_breakdown.csv", runtime)
    write_csv(reports / "reward_survival_correlation.csv", reward_survival)
    write_csv(reports / "evaluation_budget.csv", evaluation_budget)
    write_review_matrix(reports / "reviewer_response_matrix.md")
    write_markdown_summary(root, stats, controller, runtime, reward_survival, evaluation_budget)
    print(reports / "major_revision_summary.md")
    print(reports / "major_revision_statistics.csv")
    print(reports / "controller_behavior.csv")
    print(reports / "runtime_breakdown.csv")
    print(reports / "reward_survival_correlation.csv")
    print(reports / "evaluation_budget.csv")
    print(reports / "reviewer_response_matrix.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
