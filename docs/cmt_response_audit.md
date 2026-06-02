# CMT Response Audit

This note maps the main CMT criticisms to concrete project changes made in the revised ADARE artifact and paper.

## Reviewer Concerns and Responses

| CMT concern | Current response in artifact/paper | Status |
| --- | --- | --- |
| Limited novelty; ADARE looked like UCB on top of NSGA-III. | ADARE is reframed as a workflow-specific Pareto-contextual online controller with context state, multi-signal normalized rewards, density-aware mutation, archive guidance, and exported controller traces. The paper no longer presents it as a tuned NSGA-III extension. | Addressed substantially. |
| Evaluation used only tiny workflows around 24-30 tasks. | Added 20-run 1000-task protocol on CyberShake, Inspiral, Montage, and Sipht; added 20-run WfCommons 3000-task protocol on Montage, Epigenomics, Seismology, SoyKB, and SRA-search. | Addressed. |
| Missing comparisons with learning/adaptive recent methods such as QLMOEA/D-AWA and related RL-MOEA methods. | Added QL-NSGA-III, OVEA-style, and QMOEA/D-AWA-style executable baselines under the same workflow evaluator and seeds; paper cites OVEA, MRL-MOEA, and QMOEA/D-AWA. | Addressed, with caveat that OVEA/QMOEA/D-AWA are workflow adaptations rather than exact imported public artifacts. |
| Weak reproducibility. | Added config files, runnable extended comparison script, result CSVs, controller trace export, and versioned summaries for 1000/3000 20-run protocols. | Addressed. |
| Statistical protocol too small. | Main small-workflow protocol uses 20 paired runs; large 1000/3000 confirmatory protocols now use 20 runs with 10 evaluator workers. | Addressed. |
| Figures/presentation were hard to read. | Larger comparative figures are already present; added dedicated 20-run large-scale gain and win-rate figures. | Addressed. |
| Need stronger scalability evidence. | 1000-task core wins: 72/80, mean core gain +26.13%. 3000-task core wins: 86/100, mean core gain +15.53%. | Addressed for 1000-3000 task scale. |
| Need clarity about limitations. | Limitations now state remaining risks: short generation budgets at large scale, workflow-domain coverage, and exact public baselines when available. | Addressed honestly. |

## Large-Scale 20-Run Summary

ADARE wins 72/80 core-metric comparisons on 1000-task workflows and 86/100 on 3000-task workflows. Against the adaptive baselines that most directly answer the negative reviews, ADARE obtains:

| Scale | Baseline | Core wins | Mean core gain |
| --- | --- | ---: | ---: |
| 1000 tasks | QL-NSGA-III | 17/20 | +7.26% |
| 1000 tasks | OVEA-style | 20/20 | +49.74% |
| 1000 tasks | QMOEA/D-AWA-style | 17/20 | +29.44% |
| 3000 tasks | QL-NSGA-III | 20/25 | +3.54% |
| 3000 tasks | OVEA-style | 22/25 | +21.39% |
| 3000 tasks | QMOEA/D-AWA-style | 21/25 | +17.92% |

## Recommendation Assessment

The revised package is now in the strong-recommendable range on empirical adequacy: the biggest original weaknesses (scale, adaptive baselines, repeated runs, reproducibility, novelty framing) have direct responses. A cautious reviewer may still question the use of workflow-adapted OVEA/QMOEA/D-AWA instead of exact released implementations and the short generation budget on 3000-task instances. The paper should therefore avoid absolute claims such as "most optimal" and instead emphasize controlled superiority under a reproducible heterogeneous workflow protocol.

Estimated reviewer movement after this revision:

- Negative novelty/scale reviewer: from reject/weak reject toward weak accept or accept if the adaptation caveat is accepted.
- Positive reviewer: likely stronger accept because the requested scalability and statistical evidence are now present.
- Meta-review risk: much lower, because the major consensus weaknesses now have concrete experiments and text.

Overall expected score: accept to strong accept range, not guaranteed strong accept.
