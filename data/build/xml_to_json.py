"""Convert Pegasus DAX XML workflows into an intermediate JSON representation."""

import argparse
import json
import os
import xml.etree.ElementTree as ET


# Namespace required to query DAX tags correctly.
namespaces = {"dax": "http://pegasus.isi.edu/schema/DAX"}


def parse_job(job_elem: ET.Element) -> dict:
    """Parse one `<job>` node and collect job metadata and IO files."""
    job = {
        "id": job_elem.get("id"),
        "namespace": job_elem.get("namespace"),
        "name": job_elem.get("name"),
        "version": job_elem.get("version"),
        "runtime": float(job_elem.get("runtime")),
        "inputs": [],
        "outputs": [],
    }

    for use in job_elem.findall("dax:uses", namespaces):
        file_info = {
            "file": use.get("file"),
            "register": use.get("register", "false").lower() == "true",
            "transfer": use.get("transfer", "false").lower() == "true",
            "optional": use.get("optional", "false").lower() == "true",
            "type": use.get("type"),
            "size": int(use.get("size")),
        }
        link = use.get("link")
        if link == "input":
            job["inputs"].append(file_info)
        elif link == "output":
            job["outputs"].append(file_info)
    return job


def parse_dependencies(root: ET.Element) -> list[dict[str, str]]:
    """Parse `<child>/<parent>` edges into dependency list."""
    dependencies = []
    for child_elem in root.findall("dax:child", namespaces):
        child_ref = child_elem.get("ref")
        for parent_elem in child_elem.findall("dax:parent", namespaces):
            parent_ref = parent_elem.get("ref")
            dependencies.append({"from": parent_ref, "to": child_ref})
    return dependencies


def convert_xml_to_json(input_xml_path: str, output_json_path: str) -> None:
    """Convert one DAX XML file to intermediate JSON format."""
    tree = ET.parse(input_xml_path)
    root = tree.getroot()

    jobs = [parse_job(job_elem) for job_elem in root.findall("dax:job", namespaces)]
    dependencies = parse_dependencies(root)

    data = {"jobs": jobs, "dependencies": dependencies}

    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a DAX XML workflow description to JSON.")
    parser.add_argument("input_xml", help="Path to the input XML file")
    parser.add_argument("-o", "--output", help="Path to the output JSON file (optional)")

    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.input_xml)[0]
        output_path = f"{base_name}.json"

    convert_xml_to_json(args.input_xml, output_path)
    print(f"Successfully converted {args.input_xml} to {output_path}")
