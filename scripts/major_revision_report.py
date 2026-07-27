from __future__ import annotations

"""Collect major-revision experiment outputs into reviewer-facing reports."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy.stats import wilcoxon  # type: ignore


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
    files = sorted(root.rglob("*_extended_run_metrics.csv"))
    rows_out: list[dict[str, Any]] = []
    p_values: list[float] = []
    p_indices: list[int] = []
    for path in files:
        rows = read_csv(path)
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(row["benchmark"], row["algorithm"])].append(row)
        benchmarks = sorted({key[0] for key in grouped})
        for benchmark in benchmarks:
            adare_rows = grouped.get((benchmark, "ADARE"), [])
            if not adare_rows:
                continue
            baselines = sorted(algo for bench, algo in grouped if bench == benchmark and algo != "ADARE")
            for baseline in baselines:
                base_rows = grouped[(benchmark, baseline)]
                for metric, direction in METRIC_DIRECTIONS.items():
                    if metric not in adare_rows[0] or metric not in base_rows[0]:
                        continue
                    adare_values = floats(adare_rows, metric)
                    base_values = floats(base_rows, metric)
                    ad = describe(adare_values)
                    bd = describe(base_values)
                    p, rb = paired_test(adare_values, base_values, direction)
                    gain = float("nan")
                    if math.isfinite(ad["mean"]) and math.isfinite(bd["mean"]) and abs(bd["mean"]) > 1e-12:
                        if direction == "max":
                            gain = (ad["mean"] - bd["mean"]) / abs(bd["mean"]) * 100.0
                        else:
                            gain = (bd["mean"] - ad["mean"]) / abs(bd["mean"]) * 100.0
                    out = {
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
                        "wilcoxon_one_sided_p": p,
                        "holm_p": float("nan"),
                        "rank_biserial_paired": rb,
                    }
                    rows_out.append(out)
                    p_values.append(p)
                    p_indices.append(len(rows_out) - 1)
    adjusted = holm_adjust(p_values)
    for idx, adj in zip(p_indices, adjusted):
        rows_out[idx]["holm_p"] = adj
    return rows_out


def collect_controller_behavior(root: Path) -> list[dict[str, Any]]:
    files = sorted(root.rglob("*controller_trace.csv"))
    aggregate: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: {"uses": 0.0, "reward": 0.0})
    for path in files:
        for row in read_csv(path):
            key = (row.get("benchmark", ""), row.get("context", ""), row.get("operator", ""))
            aggregate[key]["uses"] += 1.0
            try:
                aggregate[key]["reward"] += float(row.get("reward", "0"))
            except ValueError:
                pass
    out = []
    for (benchmark, context, operator), stats in sorted(aggregate.items()):
        uses = int(stats["uses"])
        out.append(
            {
                "benchmark": benchmark,
                "context": context,
                "operator": operator,
                "uses": uses,
                "mean_reward": stats["reward"] / max(1, uses),
            }
        )
    return out


def write_review_matrix(path: Path) -> None:
    lines = [
        "# SwEvo Major Revision Response Matrix",
        "",
        "Use this file as the working checklist before editing the response letter.",
        "",
        "| Reviewer issue | Evidence to generate | Expected manuscript change | Status |",
        "|---|---|---|---|",
        "| R1.1/R4.2/R4.3/R5 reward and context underspecified | Method audit plus exact equations from code | Add operational definitions for context, reward, clipping, Q update, operator portfolio | pending |",
        "| R1.2/R2.3/R4.1 ablation too weak | `10_ablation_v1_v5_r20` plus any factorial extension | Replace/extend ablation table and discuss module-specific effects | pending |",
        "| R1.3/R5 runtime overhead unclear | Runtime columns from extended runs plus breakdown if instrumented | Report small vs large overhead and scaling interpretation | pending |",
        "| R1.4/R4.8/R5 reward sensitivity | Sensitivity runs or explicit limitation if not run | Add weights/clipping robustness analysis | pending |",
        "| R1.5/R4.5 fairness protocol | Code audit and comparable nondominated-set reporting | Explain seeds, budgets, initialization, archive and post-processing | pending |",
        "| R1.8 controller visualization | ADARE trace CSV and controller behavior table/plots | Add operator preference by phase/context | pending |",
        "| R2.2/R4.12/R5 shallow large-scale budgets | 1000/3000 budget sweeps | Add convergence-vs-evaluations and moderate scalability claims | pending |",
        "| R3 abstract/motivation/comments | Text revision | Rewrite abstract, add motivation, comment algorithms | pending |",
        "| R4.6 objective model | Model audit | Clarify energy, latency, transfer, serialization, units and correlations | pending |",
        "| R4.7 uncertainty wording | Text audit or robustness runs | Remove overclaim or add uncertainty protocol | pending |",
        "| R4.9 statistics incomplete | `major_revision_statistics.csv` | Add Wilcoxon direction, Holm correction, paired rank-biserial, CI if added | pending |",
        "| R4.10 aggregate core gain misleading | Per-metric statistics | De-emphasize averaged gain and prioritize raw metrics | pending |",
        "| R4.11 adapted baselines | Baseline documentation | Moderate claims and document deviations from original algorithms | pending |",
        "| R4.13 one resource configuration | Optional resource sensitivity | Add heterogeneity/communication sensitivity or limitation | pending |",
        "| R4.14 related work too long | Text revision | Shorten related work and move baseline details to experiments | pending |",
        "| R4.15 modern baselines | Literature/baseline audit | Add feasible baselines or explicitly moderate advancement claims | pending |",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_summary(root: Path, stats_rows: list[dict[str, Any]], controller_rows: list[dict[str, Any]]) -> None:
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
        f"- Reviewer matrix: `{reports / 'reviewer_response_matrix.md'}`",
        "",
        "## Completed Evidence Snapshot",
        "",
    ]
    if stats_rows:
        core = [row for row in stats_rows if row["metric"] in CORE_METRICS]
        wins = sum(1 for row in core if float(row["gain_percent"]) > 0)
        significant = sum(
            1
            for row in core
            if float(row["gain_percent"]) > 0 and math.isfinite(float(row["holm_p"])) and float(row["holm_p"]) < 0.05
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
    lines.extend(
        [
            "",
            "## Next Interpretation Tasks",
            "",
            "1. Check whether long-budget 1000-task results preserve ADARE gains.",
            "2. Check whether the ablation isolates contextual control from archive guidance.",
            "3. Use per-metric rows instead of averaged core gain when writing the revision.",
            "4. Use controller behavior rows to explain operator preference shifts by phase/context.",
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
    write_csv(reports / "major_revision_statistics.csv", stats)
    write_csv(reports / "controller_behavior.csv", controller)
    write_review_matrix(reports / "reviewer_response_matrix.md")
    write_markdown_summary(root, stats, controller)
    print(reports / "major_revision_summary.md")
    print(reports / "major_revision_statistics.csv")
    print(reports / "controller_behavior.csv")
    print(reports / "reviewer_response_matrix.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
