# SwEvo Major Revision Run Summary

Output root: `output\major_revision`

## Generated Files

- Statistics CSV: `output\major_revision\reports\major_revision_statistics.csv`
- Controller behavior CSV: `output\major_revision\reports\controller_behavior.csv`
- Runtime breakdown CSV: `output\major_revision\reports\runtime_breakdown.csv`
- Reward/survival correlation CSV: `output\major_revision\reports\reward_survival_correlation.csv`
- Evaluation budget CSV: `output\major_revision\reports\evaluation_budget.csv`
- Reviewer matrix: `output\major_revision\reports\reviewer_response_matrix.md`

## Completed Evidence Snapshot

- Core metric comparisons found: 575
- ADARE positive core gains: 446/575
- Positive core gains significant after Holm correction: 199/575
- Controller trace rows aggregated: 624 context/operator groups, 885550 selections.
- Runtime component summaries found: 243.
- Reward/survival correlation summaries found: 27.
- Evaluation-budget summaries found: 189.

## Interpretation Completed

1. Long-budget 1000-task results were checked; QL-NSGA-III is explicitly identified as competitive.
2. Controlled ablation does not support attributing the full gain to contextual control alone.
3. The manuscript prioritizes per-metric paired evidence over averaged core gain.
4. Phase/operator behavior and run-level reward--survival association are reported.
