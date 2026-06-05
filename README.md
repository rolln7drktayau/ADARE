# ADARE

ADARE is a learning-guided many-objective workflow scheduler for heterogeneous Edge/Fog/Cloud systems. The project contains the algorithm, workflow benchmarks, comparative runners, paper artifacts, and reproducible result summaries.

## Rapid Start

Use Make as the main entrypoint:

```bash
make
```

This runs environment setup/preparation, then opens the interactive ADARE menu with estimated durations for each action.

If the environment is already ready:

```bash
make menu
```

## Common Commands

```bash
make adare             # ADARE only on Montage_25
make adare-1000        # ADARE only on CyberShake_1000
make live              # evolution dashboard: generation, convergence, metrics, Pareto view
make smoke             # quick ADARE vs NSGA-III sanity check
make main20            # main 20-run paper protocol
make extended-1000-r20 # long 1000-task 20-run protocol
make extended-3000-r20 # long 3000-task 20-run protocol
make paper             # compile and sync article PDFs
```

## Script Layout

Runnable Python entrypoints live in `scripts/`:

- `scripts/make_menu.py`: interactive Make menu.
- `scripts/run_adare.py`: ADARE-only workflow execution.
- `scripts/main.py`: paired ADARE vs NSGA-III protocol.
- `scripts/run_extended_comparison.py`: extended baseline comparisons.
- `scripts/run_ablation_v1_v5.py`: V1-V5 ablation protocol.
- `scripts/live_view.py`: visual evolution dashboard.
- `scripts/adare_vs_nsga3.py`: compatibility wrapper.

Core algorithm code remains under `algorithms/`, workflow/resource loading under `problem/`, benchmarks under `data/benchmarks/`, and plotting/metrics under `evaluation/`.

## Evolution Dashboard

`make live` launches an ADARE run and shows a dashboard containing:

- current generation summary,
- convergence curves for all objective best values,
- evolution of HV, IGD, spacing, epsilon, and coverage-to-reference,
- Pareto-front projection across generations.

The dashboard stores CSV/PNG outputs in `output/live_view/`.

The dashboard has a small overhead because it captures objective arrays by generation and computes visual metrics after the run. Normal runs do not enable this capture and are not affected.
