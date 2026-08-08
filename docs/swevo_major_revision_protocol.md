# SwEvo Major Revision Protocol

This protocol prepares the long-running evidence requested by the Swarm and Evolutionary Computation reviewers without editing the manuscript.

## Commands

Prepare the full protocol without running experiments:

```bash
make revision-plan
```

Run a quick structural check:

```bash
make revision-quick
```

Run the full long protocol:

```bash
make revision-full
```

Rebuild reports after any completed or partial run:

```bash
make revision-report
```

## Outputs

All generated files are written under:

```text
output/major_revision/
```

Main files to inspect or share back for interpretation:

```text
output/major_revision/reports/major_revision_summary.md
output/major_revision/reports/major_revision_statistics.csv
output/major_revision/reports/controller_behavior.csv
output/major_revision/reports/runtime_breakdown.csv
output/major_revision/reports/reward_survival_correlation.csv
output/major_revision/reports/evaluation_budget.csv
output/major_revision/reports/reviewer_response_matrix.md
output/major_revision/reports/major_revision_full_protocol.md
output/major_revision/logs/
```

## Planned Evidence

The full protocol prepares:

- V1-V5 ablation evidence for module contribution.
- Controlled static/global-UCB/contextual-UCB/proposed-reward ablation with mutation and archive held fixed.
- Paired reward-weight, clipping, and learning-rate sensitivity analysis.
- Small-workflow 20-run comparisons against all implemented baselines.
- 1000-task budget sweeps with deeper generation budgets.
- 3000-task deeper stress tests.
- ADARE controller traces for operator preference and interpretability analysis.
- Per-metric statistics with one-sided paired Wilcoxon tests, Holm correction, and paired rank-biserial effect size.

## Resume

If a long run stops, regenerate the plan and resume from a specific step:

```bash
python scripts/major_revision_pipeline.py --preset full --start-at 31_1000_budget_sweep_r20_g100 --execute
```

To run only selected steps:

```bash
python scripts/major_revision_pipeline.py --preset full --only 30_1000_budget_sweep_r20_g50 90_collect_reports --execute
```

## Interpretation Step

After the long runs finish, ask Codex to inspect:

```text
output/major_revision/reports/major_revision_summary.md
output/major_revision/reports/major_revision_statistics.csv
output/major_revision/reports/controller_behavior.csv
output/major_revision/logs/
```

The manuscript and response letter should be updated only after those outputs are reviewed.
