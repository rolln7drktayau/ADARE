# Consolidated major-revision results

These files are compact, publication-facing summaries generated from the complete local experiment tree under `output/major_revision/`.

- `major_revision_statistics.csv`: paired per-protocol, benchmark, baseline and metric inference, including Wilcoxon direction, Holm correction, paired rank-biserial effect and bootstrap confidence interval.
- `controller_behavior.csv`: controller action counts and rewards by context/operator.
- `runtime_breakdown.csv`: instrumented runtime-component summaries.
- `reward_survival_correlation.csv`: run-level reward/offspring-survival association.
- `evaluation_budget.csv`: nominal and realized evaluation-budget summaries.
- `major_revision_summary.md`: compact completeness snapshot.

The full raw output is intentionally not versioned because it is substantially larger and can be regenerated with `run_swevo_major_revision.ps1` or `scripts/major_revision_pipeline.py`. To rebuild these summaries after a rerun:

```bash
python scripts/major_revision_report.py --root output/major_revision
```

The committed CSV files correspond to the revision completed on 2026-08-08.
