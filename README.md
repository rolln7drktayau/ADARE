# ADARE

ADARE is a research-software project for reproducible many-objective workflow-scheduling experiments in heterogeneous Edge/Fog/Cloud systems. The current method is an adaptive extension of NSGA-III: reference-vector environmental survival is retained, while contextual crossover control, density-aware mutation, local repair, and bounded archive guidance are added around it.

The repository is intended for researchers who want to reproduce the reported study, audit the implementation, compare additional optimizers, or reuse the deterministic workflow evaluator for controlled tests.

## Research scope

- Objectives: makespan, aggregate post-readiness latency, compute cost, and a compute-only working-power energy proxy.
- Algorithms: ADARE, NSGA-II, NSGA-III, MOEA/D, QL-NSGA-III, and declared workflow adaptations inspired by OVEA and QMOEA/D-AWA.
- Workflows: Pegasus-style XML and WfCommons JSON instances from roughly 24 to 3000 tasks.
- Reproducibility: fixed seeds, shared initial populations, matched population/generation budgets, paired inference, and exported evaluation counts.
- Execution: CPU-oriented Python implementation with optional multiprocessing. A GPU is not required.

The evaluator is deterministic. It does not model failures, contention, time-varying resources, idle or communication energy, or transfer charges. See the manuscript for the exact scope and limitations.

## Repository layout

```text
algorithms/      Optimizers and ADARE control logic
config/          Shared and algorithm-specific JSON configurations
data/            Workflow and resource inputs
evaluation/      Pareto metrics and plotting utilities
problem/         Workflow/resource loading
scripts/         Experiment, ablation, reporting and diagnostic entrypoints
results/         Lightweight committed result summaries
Figures/         Publication figures generated from the reported study
papers/          Revised manuscript, response letter and generated tables
docs/            Reproduction and execution guides
```

Raw run outputs are written to `output/` and intentionally excluded from Git because a complete revision campaign is large. Consolidated CSV evidence is committed under `results/major_revision/`.

## Installation

Python 3.11 or newer is recommended. The reported revision used Python 3.13.1.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick reproducibility checks

Run one ADARE test:

```bash
python scripts/run_adare.py --benchmarks Montage_25 --runs 1 --generations 5 --population-size 20 --output-dir output/smoke_adare
```

Run a small matched multi-algorithm comparison:

```bash
python scripts/run_extended_comparison.py --benchmarks Montage_25 --algorithms ADARE NSGA-III QL-NSGA-III --runs 2 --generations 5 --population-size 20 --output-dir output/smoke_comparison
```

On Windows, the complete journal-revision protocol is described in [the French execution guide](docs/swevo_execution_tutorial_fr.md). Its launcher supports individual resumable stages:

```powershell
.\run_swevo_major_revision.ps1 -Preset full -Only 20_small_all_algorithms_r20_g70
```

To regenerate consolidated tables and reports without rerunning experiments:

```bash
python scripts/major_revision_report.py --root output/major_revision
python scripts/build_revision_assets.py
```

## Reported research artifacts

- [Revised manuscript](papers/article_swevo.pdf)
- [LaTeX manuscript source](papers/article_swevo.tex)
- [Response to reviewers](papers/response_to_reviewers.pdf)
- [Consolidated revision results](results/major_revision/)
- [Detailed reproduction guide](docs/reproduction_guide.md)

The main paper deliberately avoids a universal-superiority claim. QL-NSGA-III remains competitive under longer budgets, and resource sensitivity is reported explicitly.

## Extending the project

New optimizers should subclass the shared base implementation, consume the same initial population, return a `survival_population`, and use the common evaluator. Comparisons should preserve paired seeds and report realized objective evaluations in addition to nominal generation/population budgets.

Generated files should remain under `output/`; only compact, interpretable summaries should be promoted to `results/`.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). The code is released under the permissive [MIT License](LICENSE). Workflow datasets and cited third-party methods retain their original provenance and terms.
