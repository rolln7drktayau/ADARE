from __future__ import annotations

"""Prepare and optionally run the SwEvo major-revision experimental protocol."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "major_revision"
LOGS = OUT / "logs"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"

SMALL = ["Montage_25", "CyberShake_30", "Epigenomics_24"]
WF1000 = ["CyberShake_1000", "Inspiral_1000", "Montage_1000", "Sipht_1000"]
WF3000 = [
    "Montage_3000_wfcommons",
    "Epigenomics_3000_wfcommons",
    "Seismology_3000_wfcommons",
    "Soykb_3000_wfcommons",
    "Srasearch_3000_wfcommons",
]
ALL_ALGOS = ["ADARE", "NSGA-III", "NSGA-II", "MOEA/D", "QL-NSGA-III", "OVEA-style", "QMOEA/D-AWA-style"]
ADAPTIVE_ALGOS = ["ADARE", "NSGA-III", "QL-NSGA-III", "OVEA-style", "QMOEA/D-AWA-style"]


def py() -> str:
    return sys.executable or "python"


def cmd(*parts: str | Path) -> list[str]:
    return [str(part) for part in parts]


def command_specs(preset: str) -> list[dict[str, object]]:
    quick = [
        {
            "id": "00_quick_extended_smoke",
            "reviewer_targets": ["fair baseline protocol", "multi-algorithm figures", "pipeline sanity"],
            "estimated_time": "2-5 min",
            "command": cmd(
                py(),
                "scripts/run_extended_comparison.py",
                "--benchmarks",
                "Montage_25",
                "--algorithms",
                "ADARE",
                "NSGA-III",
                "QL-NSGA-III",
                "--runs",
                "1",
                "--generations",
                "4",
                "--population-size",
                "30",
                "--output-dir",
                OUT / "quick_extended",
                "--figure-dir",
                FIGURES / "quick",
            ),
        },
        {
            "id": "01_quick_ablation_smoke",
            "reviewer_targets": ["ablation sanity"],
            "estimated_time": "2-5 min",
            "command": cmd(
                py(),
                "scripts/run_ablation_v1_v5.py",
                "--runs",
                "1",
                "--no-plots",
                "--output-dir",
                OUT / "quick_ablation",
            ),
        },
        {
            "id": "02_quick_adare_trace",
            "reviewer_targets": ["controller behavior visualization", "interpretability"],
            "estimated_time": "1-3 min",
            "command": cmd(
                py(),
                "scripts/run_adare.py",
                "--benchmarks",
                "Montage_25",
                "--runs",
                "1",
                "--generations",
                "8",
                "--population-size",
                "40",
                "--output-dir",
                OUT / "quick_adare_trace",
            ),
        },
        {
            "id": "03_quick_controller_ablation",
            "reviewer_targets": ["R4.1 controlled controller ablation sanity"],
            "estimated_time": "1-3 min",
            "command": cmd(
                py(),
                "scripts/run_controller_ablation.py",
                "--benchmarks",
                "Montage_25",
                "--runs",
                "1",
                "--generations",
                "4",
                "--population-size",
                "30",
                "--output-dir",
                OUT / "quick_controller_ablation",
                "--figure-dir",
                FIGURES / "quick_controller_ablation",
            ),
        },
    ]
    full = [
        {
            "id": "10_ablation_v1_v5_r20",
            "reviewer_targets": ["R1.2", "R2.3", "R4.1", "R5 reward/controller contribution"],
            "estimated_time": "10-25 min",
            "command": cmd(py(), "scripts/run_ablation_v1_v5.py", "--runs", "20", "--output-dir", OUT / "ablation_v1_v5_r20"),
        },
        {
            "id": "11_controller_ablation_r20_g70",
            "reviewer_targets": ["R1.2", "R2.3", "R4.1 static/global/contextual/reward isolation"],
            "estimated_time": "1-3 h",
            "command": cmd(
                py(),
                "scripts/run_controller_ablation.py",
                "--runs",
                "20",
                "--generations",
                "70",
                "--population-size",
                "100",
                "--output-dir",
                OUT / "controller_ablation_r20_g70",
                "--figure-dir",
                FIGURES / "controller_ablation_r20_g70",
            ),
        },
        {
            "id": "12_reward_sensitivity_r20_g70",
            "reviewer_targets": ["R1.4", "R4.3", "R4.8", "R5 reward weights/clipping/alpha sensitivity"],
            "estimated_time": "45-90 min",
            "command": cmd(
                py(),
                "scripts/run_reward_sensitivity.py",
                "--runs",
                "20",
                "--generations",
                "70",
                "--population-size",
                "100",
                "--output-dir",
                OUT / "reward_sensitivity_r20_g70",
            ),
        },
        {
            "id": "20_small_all_algorithms_r20_g70",
            "reviewer_targets": ["R1 fairness", "R3 experiments", "R4.9 statistics", "R4.15 baselines"],
            "estimated_time": "40-90 min",
            "command": cmd(
                py(),
                "scripts/run_extended_comparison.py",
                "--benchmarks",
                *SMALL,
                "--algorithms",
                *ALL_ALGOS,
                "--runs",
                "20",
                "--generations",
                "70",
                "--population-size",
                "100",
                "--output-dir",
                OUT / "small_all_algorithms_r20_g70",
                "--figure-dir",
                FIGURES / "small_all_algorithms_r20_g70",
            ),
        },
        {
            "id": "30_1000_budget_sweep_r20_g50",
            "reviewer_targets": ["R2.2", "R4.12", "R5 large-scale depth"],
            "estimated_time": "1.5-4 h",
            "command": cmd(
                py(),
                "scripts/run_extended_comparison.py",
                "--benchmarks",
                *WF1000,
                "--algorithms",
                *ADAPTIVE_ALGOS,
                "--runs",
                "20",
                "--generations",
                "50",
                "--population-size",
                "80",
                "--output-dir",
                OUT / "large_1000_r20_g50",
            ),
        },
        {
            "id": "31_1000_budget_sweep_r20_g100",
            "reviewer_targets": ["R2.2", "R4.12", "R5 requested 100-generation 1000-task evidence"],
            "estimated_time": "3-8 h",
            "command": cmd(
                py(),
                "scripts/run_extended_comparison.py",
                "--benchmarks",
                *WF1000,
                "--algorithms",
                *ADAPTIVE_ALGOS,
                "--runs",
                "20",
                "--generations",
                "100",
                "--population-size",
                "80",
                "--output-dir",
                OUT / "large_1000_r20_g100",
            ),
        },
        {
            "id": "40_3000_budget_sweep_r10_g20",
            "reviewer_targets": ["R4.12", "3000-task deeper-than-original stress test"],
            "estimated_time": "2-6 h",
            "command": cmd(
                py(),
                "scripts/run_extended_comparison.py",
                "--benchmarks",
                *WF3000,
                "--algorithms",
                *ADAPTIVE_ALGOS,
                "--runs",
                "10",
                "--generations",
                "20",
                "--population-size",
                "60",
                "--output-dir",
                OUT / "large_3000_r10_g20",
            ),
        },
        {
            "id": "50_adare_controller_traces_1000_r20_g100",
            "reviewer_targets": ["R1.8", "controller phase behavior", "operator preference dynamics"],
            "estimated_time": "45 min-2 h",
            "command": cmd(
                py(),
                "scripts/run_adare.py",
                "--benchmarks",
                "CyberShake_1000",
                "Montage_1000",
                "--runs",
                "20",
                "--generations",
                "100",
                "--population-size",
                "80",
                "--output-dir",
                OUT / "adare_controller_traces_1000_r20_g100",
            ),
        },
        {
            "id": "60_scaling_resource_diagnostics_r10_g30",
            "reviewer_targets": [
                "R4.12 evaluation/time convergence and memory scaling",
                "R4.13 resource-configuration sensitivity",
            ],
            "estimated_time": "30-90 min",
            "command": cmd(
                py(),
                "scripts/run_scaling_diagnostics.py",
                "--runs",
                "10",
                "--generations",
                "30",
                "--population-size",
                "60",
                "--output-dir",
                OUT / "scaling_diagnostics_r10_g30",
            ),
        },
    ]
    report = [
        {
            "id": "90_collect_reports",
            "reviewer_targets": ["summary reports", "statistics", "response matrix"],
            "estimated_time": "<1 min",
            "command": cmd(py(), "scripts/major_revision_report.py", "--root", OUT),
        }
    ]
    if preset == "quick":
        return quick + report
    if preset == "full":
        return full + report
    if preset == "reports":
        return report
    raise ValueError(f"Unknown preset: {preset}")


def quote_command(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def powershell_command(parts: Iterable[str]) -> str:
    quoted = []
    for part in parts:
        s = str(part)
        if any(ch in s for ch in " ()&;/\\"):
            quoted.append("'" + s.replace("'", "''") + "'")
        else:
            quoted.append(s)
    return " ".join(quoted)


def write_manifest(specs: list[dict[str, object]], preset: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preset": preset,
        "output_root": str(OUT),
        "commands": specs,
    }
    (OUT / f"major_revision_{preset}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        f"# SwEvo Major Revision Protocol ({preset})",
        "",
        f"Created: {manifest['created_at']}",
        f"Output root: `{OUT}`",
        "",
        "| Step | Estimated time | Reviewer targets | Command |",
        "|---|---:|---|---|",
    ]
    for spec in specs:
        lines.append(
            "| {id} | {time} | {targets} | `{command}` |".format(
                id=spec["id"],
                time=spec["estimated_time"],
                targets=", ".join(str(x) for x in spec["reviewer_targets"]),
                command=powershell_command(spec["command"]),
            )
        )
    (REPORTS / f"major_revision_{preset}_protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ps1_lines = [
        "$ErrorActionPreference = 'Stop'",
        f"Set-Location '{ROOT}'",
        f"New-Item -ItemType Directory -Force '{LOGS}' | Out-Null",
    ]
    for spec in specs:
        step_id = str(spec["id"])
        log_path = LOGS / f"{step_id}.log"
        ps1_lines.append(f"Write-Host '=== {step_id} ==='")
        ps1_lines.append(f"{powershell_command(spec['command'])} *> '{log_path}'")
    script_path = OUT / f"run_major_revision_{preset}.ps1"
    script_path.write_text("\n".join(ps1_lines) + "\n", encoding="utf-8")


def run_step(spec: dict[str, object]) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    step_id = str(spec["id"])
    log_path = LOGS / f"{step_id}.log"
    started = datetime.now().isoformat(timespec="seconds")
    print(f"\n=== {step_id} | started {started} ===")
    print(quote_command(spec["command"]))
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started: {started}\n")
        log.write(f"Command: {quote_command(spec['command'])}\n\n")
        log.flush()
        child_env = dict(os.environ)
        child_env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            list(spec["command"]),
            cwd=ROOT,
            env=child_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if proc.stdout is None:
            raise RuntimeError(f"Unable to capture output for step {step_id}")
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        proc.wait()
        log.write(f"\nFinished: {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"Exit code: {proc.returncode}\n")
    print(f"Log: {log_path}")
    print(f"Exit code: {proc.returncode}")
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run the SwEvo major-revision protocol.")
    parser.add_argument("--preset", choices=["quick", "full", "reports"], default="full")
    parser.add_argument("--execute", action="store_true", help="Run the selected protocol now.")
    parser.add_argument("--only", nargs="*", help="Run only selected step ids.")
    parser.add_argument("--start-at", help="Skip steps before this step id.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = command_specs(args.preset)
    if args.start_at:
        ids = [str(spec["id"]) for spec in specs]
        if args.start_at not in ids:
            raise ValueError(f"Unknown --start-at step: {args.start_at}")
        specs = specs[ids.index(args.start_at) :]
    if args.only:
        keep = set(args.only)
        specs = [spec for spec in specs if str(spec["id"]) in keep]
    write_manifest(specs, args.preset)
    print(f"Protocol written under: {OUT}")
    print(f"PowerShell launcher: {OUT / f'run_major_revision_{args.preset}.ps1'}")
    print(f"Markdown protocol: {REPORTS / f'major_revision_{args.preset}_protocol.md'}")
    if not args.execute:
        print("Dry run only. Add --execute to run the selected steps.")
        return 0
    failures = 0
    for spec in specs:
        failures += 1 if run_step(spec) != 0 else 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
