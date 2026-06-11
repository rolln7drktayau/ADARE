"""Interactive Make menu for ADARE experiments and maintenance."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


@dataclass(frozen=True)
class MenuAction:
    key: str
    label: str
    estimate: str
    description: str
    runner: Callable[[], int]


def run_command(args: Sequence[str], title: str, cwd: Path = ROOT) -> int:
    print(f"\n== {title} ==")
    print("Commande:", " ".join(args))
    return subprocess.run(list(args), cwd=cwd).returncode


def run_commands(commands: Sequence[tuple[Sequence[str], str] | tuple[Sequence[str], str, Path]]) -> int:
    for command in commands:
        if len(command) == 3:
            args, title, cwd = command
        else:
            args, title = command
            cwd = ROOT
        code = run_command(args, title, cwd)
        if code != 0:
            return code
    return 0


def ensure_dirs() -> None:
    for rel in ("output/plots", "output/reports", "results/extended", "Figures"):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def clean_outputs() -> int:
    print("\n== Nettoyage des sorties generees ==")
    targets = [
        ROOT / "output",
        ROOT / "__pycache__",
        ROOT / ".pytest_cache",
        ROOT / "data" / "history",
        ROOT / "har_and_cookies",
    ]
    for target in targets:
        if target.exists():
            print(f"Suppression: {target.relative_to(ROOT)}")
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
    ensure_dirs()
    return 0


def compile_paper() -> int:
    commands = [
        (["pdflatex", "-interaction=nonstopmode", "article_ecml.tex"], "Compilation LaTeX 1/2", ROOT / "papers"),
        (["pdflatex", "-interaction=nonstopmode", "article_ecml.tex"], "Compilation LaTeX 2/2", ROOT / "papers"),
        (
            [
                PYTHON,
                "-c",
                (
                    "import shutil; "
                    "shutil.copyfile('papers/article_ecml.pdf','ADARE_Adaptive_Data-driven_Algorithm_for_Resource_Evolution.pdf')"
                ),
            ],
            "Synchroniser les PDFs",
        ),
    ]
    return run_commands(commands)


def custom_benchmark() -> int:
    print("\nBenchmarks disponibles:")
    available = sorted(
        path.stem
        for path in (ROOT / "data" / "benchmarks").rglob("*.json")
        if path.name != "tasks.json"
    )
    for idx, name in enumerate(available, 1):
        print(f"  {idx:>2}. {name}")
    raw = input("\nBenchmark ou numero: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(available):
        benchmark = available[int(raw) - 1]
    else:
        benchmark = raw
    runs = input("Runs [2]: ").strip() or "2"
    generations = input("Generations [15]: ").strip() or "15"
    population = input("Population [80]: ").strip() or "80"
    return run_command(
        [
            PYTHON,
            "scripts/main.py",
            "--benchmarks",
            benchmark,
            "--runs",
            runs,
            "--generations",
            generations,
            "--population-size",
            population,
        ],
        f"Run ADARE vs NSGA-III sur {benchmark}",
    )


def adare_only_custom() -> int:
    print("\nBenchmarks disponibles:")
    available = sorted(
        path.stem
        for path in (ROOT / "data" / "benchmarks").rglob("*.json")
        if path.name != "tasks.json"
    )
    for idx, name in enumerate(available, 1):
        print(f"  {idx:>2}. {name}")
    raw = input("\nBenchmark ou numero: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(available):
        benchmark = available[int(raw) - 1]
    else:
        benchmark = raw
    runs = input("Runs [1]: ").strip() or "1"
    generations = input("Generations [70]: ").strip() or "70"
    population = input("Population [100]: ").strip() or "100"
    return run_command(
        [
            PYTHON,
            "scripts/run_adare.py",
            "--benchmarks",
            benchmark,
            "--runs",
            runs,
            "--generations",
            generations,
            "--population-size",
            population,
        ],
        f"ADARE seul sur {benchmark}",
    )


def extended_custom() -> int:
    benchmarks = input("Benchmarks separes par espaces [Montage_25 CyberShake_30 Epigenomics_24]: ").strip()
    if not benchmarks:
        benchmarks = "Montage_25 CyberShake_30 Epigenomics_24"
    runs = input("Runs [2]: ").strip() or "2"
    generations = input("Generations [15]: ").strip() or "15"
    population = input("Population [80]: ").strip() or "80"
    algos = input("Algorithmes [ADARE NSGA-III QL-NSGA-III OVEA-style QMOEA/D-AWA-style]: ").strip()
    if not algos:
        algos = "ADARE NSGA-III QL-NSGA-III OVEA-style QMOEA/D-AWA-style"
    return run_command(
        [
            PYTHON,
            "scripts/run_extended_comparison.py",
            "--benchmarks",
            *benchmarks.split(),
            "--algorithms",
            *algos.split(),
            "--runs",
            runs,
            "--generations",
            generations,
            "--population-size",
            population,
            "--output-dir",
            "output/extended_custom",
        ],
        "Comparaison etendue personnalisee",
    )


def actions() -> list[MenuAction]:
    small = ["Montage_25", "CyberShake_30", "Epigenomics_24"]
    wf1000 = ["CyberShake_1000", "Inspiral_1000", "Montage_1000", "Sipht_1000"]
    wf3000 = [
        "Montage_3000_wfcommons",
        "Epigenomics_3000_wfcommons",
        "Seismology_3000_wfcommons",
        "Soykb_3000_wfcommons",
        "Srasearch_3000_wfcommons",
    ]
    adaptive_algos = ["ADARE", "NSGA-III", "QL-NSGA-III", "OVEA-style", "QMOEA/D-AWA-style"]
    all_algos = ["ADARE", "NSGA-III", "NSGA-II", "MOEA/D", "QL-NSGA-III", "OVEA-style", "QMOEA/D-AWA-style"]

    return [
        MenuAction("1", "Setup environnement", "2-8 min", "Installe/actualise les dependances Python.", lambda: run_command([PYTHON, "-m", "pip", "install", "-r", "requirements.txt"], "Setup")),
        MenuAction("2", "ADARE seul rapide", "1-5 min", "Execute seulement ADARE sur Montage_25.", lambda: run_command([PYTHON, "scripts/run_adare.py", "--benchmarks", "Montage_25", "--runs", "1", "--generations", "70", "--population-size", "100"], "ADARE seul rapide")),
        MenuAction("3", "ADARE seul 1000", "5-15 min", "Execute seulement ADARE sur CyberShake_1000.", lambda: run_command([PYTHON, "scripts/run_adare.py", "--benchmarks", "CyberShake_1000", "--runs", "1", "--generations", "15", "--population-size", "80"], "ADARE seul 1000")),
        MenuAction("4", "ADARE seul personnalise", "variable", "Choisir workflow, runs, generations, population.", adare_only_custom),
        MenuAction("5", "Vue evolution ADARE", "2-8 min + affichage", "Lance ADARE et affiche generation, convergence, metriques et front.", lambda: run_command([PYTHON, "scripts/live_view.py", "--benchmark", "Montage_25", "--generations", "40", "--population-size", "80"], "Vue evolution ADARE")),
        MenuAction("6", "Smoke test comparatif", "1-3 min", "Petit run ADARE vs NSGA-III pour verifier que tout demarre.", lambda: run_command([PYTHON, "scripts/main.py", "--benchmarks", "Montage_25", "--runs", "1", "--generations", "5", "--population-size", "30"], "Smoke test")),
        MenuAction("7", "Protocole papier principal", "45-90 min", "20 runs sur Montage_25, CyberShake_30, Epigenomics_24.", lambda: run_command([PYTHON, "scripts/main.py", "--benchmarks", *small, "--runs", "20", "--generations", "70", "--population-size", "100"], "Protocole papier principal")),
        MenuAction("8", "Comparaison etendue rapide", "20-45 min", "Small suite contre toutes les baselines.", lambda: run_command([PYTHON, "scripts/run_extended_comparison.py", "--benchmarks", *small, "--algorithms", *all_algos, "--runs", "5", "--generations", "15", "--population-size", "80", "--output-dir", "output/extended_small_menu", "--figure-dir", "Figures"], "Comparaison etendue rapide")),
        MenuAction("9", "Long 1000 r20", "2h-2h15", "20 runs sur les workflows 1000 avec baselines adaptatives.", lambda: run_command([PYTHON, "scripts/run_extended_comparison.py", "--benchmarks", *wf1000, "--algorithms", *adaptive_algos, "--runs", "20", "--generations", "15", "--population-size", "80", "--output-dir", "output/extended_1000_r20"], "Long 1000 r20")),
        MenuAction("10", "Long 3000 r20", "3h-4h", "20 runs sur les workflows WfCommons 3000 avec baselines adaptatives.", lambda: run_command([PYTHON, "scripts/run_extended_comparison.py", "--benchmarks", *wf3000, "--algorithms", *adaptive_algos, "--runs", "20", "--generations", "8", "--population-size", "60", "--output-dir", "output/extended_3000_r20"], "Long 3000 r20")),
        MenuAction("11", "Ablation V1-V5", "30-75 min", "Recalcule l'ablation des modules ADARE.", lambda: run_command([PYTHON, "scripts/run_ablation_v1_v5.py", "--runs", "20", "--output-dir", "output/ablation_full"], "Ablation V1-V5")),
        MenuAction("12", "Figures etendues", "<1 min", "Regenere les figures depuis results/extended.", lambda: run_commands([
            ([PYTHON, "evaluation/plot_extended_results.py", "--input", "results/extended/extended_global_summary.csv", "--output-dir", "Figures"], "Figures de synthese etendue"),
            ([PYTHON, "evaluation/plot_multialgo_figures.py", "--run-metrics", "results/extended/extended_small_r5_run_metrics.csv", "--summary", "results/extended/extended_small_r5_summary.csv", "--output-dir", "Figures"], "Figures multi-algorithmes explicites"),
        ])),
        MenuAction("13", "Compiler et synchroniser PDF", "10-30 sec", "Compile article_ecml.pdf et met a jour les deux PDFs demandes.", compile_paper),
        MenuAction("14", "Benchmark comparatif personnalise", "variable", "Choisir benchmark, runs, generations, population.", custom_benchmark),
        MenuAction("15", "Comparaison etendue personnalisee", "variable", "Choisir workflows, algorithmes, runs et budget.", extended_custom),
        MenuAction("16", "Lint", "1-3 min", "Execute flake8.", lambda: run_command([PYTHON, "-m", "flake8", "."], "Lint")),
        MenuAction("17", "Format", "1-3 min", "Execute black.", lambda: run_command([PYTHON, "-m", "black", "."], "Format")),
        MenuAction("18", "Nettoyer sorties", "<1 min", "Nettoie output/cache/history sans toucher aux resultats versionnes.", clean_outputs),
    ]


def print_menu(items: Sequence[MenuAction]) -> None:
    print("\nADARE - menu Make")
    print("=" * 72)
    for item in items:
        print(f"{item.key:>2}. {item.label:<36} [{item.estimate}]")
        print(f"    {item.description}")
    print(" q. Quitter")


def main() -> int:
    os.chdir(ROOT)
    ensure_dirs()
    items = actions()
    by_key = {item.key: item for item in items}

    while True:
        print_menu(items)
        choice = input("\nChoix: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return 0
        action = by_key.get(choice)
        if action is None:
            print("Choix invalide.")
            continue
        code = action.runner()
        print(f"\nResultat: {'OK' if code == 0 else f'echec ({code})'}")
        again = input("Retour au menu ? [Y/n]: ").strip().lower()
        if again in {"n", "no", "non"}:
            return code


if __name__ == "__main__":
    raise SystemExit(main())
