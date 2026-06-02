"""Convert WfCommons workflow JSON files into ADARE benchmark task JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _numeric_sort_key(task_id: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", task_id)
    if match is None:
        return task_id, 0
    return task_id[: match.start()], int(match.group(1))


def convert_wfcommons(input_json_path: Path, output_json_path: Path) -> None:
    """Convert one WfCommons JSON instance to the local `tasks` schema."""
    data = json.loads(input_json_path.read_text(encoding="utf-8"))
    workflow = data.get("workflow", {})
    specification = workflow.get("specification", {})
    execution = workflow.get("execution", {})

    spec_tasks: list[dict[str, Any]] = list(specification.get("tasks", []))
    exec_tasks = {
        str(task["id"]): task
        for task in execution.get("tasks", [])
        if isinstance(task, dict) and "id" in task
    }
    file_sizes = {
        str(file_info["id"]): float(file_info.get("sizeInBytes", 0.0))
        for file_info in specification.get("files", [])
        if isinstance(file_info, dict) and "id" in file_info
    }

    if not spec_tasks:
        raise ValueError(f"No WfCommons tasks found in {input_json_path}")

    ordered_ids = sorted((str(task["id"]) for task in spec_tasks), key=_numeric_sort_key)
    id_mapping = {task_id: idx + 1 for idx, task_id in enumerate(ordered_ids)}

    tasks = []
    for spec_task in sorted(spec_tasks, key=lambda task: id_mapping[str(task["id"])]):
        task_id = str(spec_task["id"])
        runtime = float(exec_tasks.get(task_id, {}).get("runtimeInSeconds", 1.0))
        input_files = spec_task.get("inputFiles", [])
        data_size_bytes = sum(file_sizes.get(str(file_id), 0.0) for file_id in input_files)
        dependencies = [
            id_mapping[str(parent)]
            for parent in spec_task.get("parents", [])
            if str(parent) in id_mapping
        ]

        tasks.append(
            {
                "id": id_mapping[task_id],
                "instructions": max(1, int(runtime * 75000)),
                "data_size": round(data_size_bytes / 1e6, 3),
                "dependencies": dependencies,
                "deadline": max(1, int(runtime * 2)),
            }
        )

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps({"tasks": tasks}, indent=4), encoding="utf-8")
    print(f"Converted {input_json_path} -> {output_json_path} ({len(tasks)} tasks)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert WfCommons JSON to ADARE benchmark JSON.")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    convert_wfcommons(args.input_json, args.output)


if __name__ == "__main__":
    main()
