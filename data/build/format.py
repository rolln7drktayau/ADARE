"""Convert intermediate workflow JSON into normalized benchmark task JSON."""

import argparse
import json
import os
import re


def parse_job_id(job_id: str) -> int:
    """Extract numeric part from Pegasus-style job identifier (e.g., ID00042 -> 42)."""
    return int(re.search(r"\d+", job_id).group())


def convert_workflow(input_json_path: str, output_json_path: str) -> None:
    """Convert workflow job/dependency JSON into the benchmark `tasks` schema.

    IDs are remapped to start at 1 and remain contiguous.
    Runtime and input-size fields are converted to the expected scales.
    """
    # Conversion constants chosen to align with the scheduler input domain.
    instruction_factor = 75000
    data_size_factor = 1e6

    with open(input_json_path, encoding="utf-8") as f:
        montage = json.load(f)

    original_ids = [parse_job_id(job["id"]) for job in montage["jobs"]]
    id_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(sorted(original_ids))}

    dependencies_map = {id_mapping[parse_job_id(job["id"])]: [] for job in montage["jobs"]}
    for dep in montage["dependencies"]:
        to_id = parse_job_id(dep["to"])
        from_id = parse_job_id(dep["from"])
        dependencies_map[id_mapping[to_id]].append(id_mapping[from_id])

    tasks = []
    for job in montage["jobs"]:
        original_job_id = parse_job_id(job["id"])
        new_job_id = id_mapping[original_job_id]

        data_size = sum(f["size"] for f in job["inputs"]) / data_size_factor

        tasks.append(
            {
                "id": new_job_id,
                "instructions": int(job["runtime"] * instruction_factor),
                "data_size": round(data_size, 1),
                "dependencies": dependencies_map.get(new_job_id, []),
                "deadline": int(job["runtime"] * 2),
            }
        )

    tasks = sorted(tasks, key=lambda x: x["id"])

    output = {"tasks": tasks}
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"Conversion completed. Saved: {output_json_path}")
    print("Task IDs were normalized to start at 1.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert workflow JSON into normalized benchmark task format."
    )
    parser.add_argument("input_json", help="Path to input JSON file")
    parser.add_argument("-o", "--output", help="Optional output JSON path")

    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.input_json)[0]
        output_path = f"{base_name}_tasks.json"

    convert_workflow(args.input_json, output_path)
