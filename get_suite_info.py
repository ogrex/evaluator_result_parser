#!/usr/bin/env python3
"""Fetch and display ALL available suite information from a Tier4 vehicle catalog.

Optionally generates:
    - t4datasets.csv       — table of unique t4_dataset_ids with rich context
    - download_commands.sh — shell script with webauto pull commands for all datasets
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from t4_visualizer.downloader import list_catalog_suites, get_suite_info, get_scenario_info

def format_value(value, indent=2):
    """Format a value for pretty printing."""
    spaces = " " * indent
    if isinstance(value, list):
        if not value:
            return "[]"
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append("{")
                for k, v in item.items():
                    lines.append(f"{spaces}  {k}: {v!r}")
                lines.append(f"{spaces}}}")
            else:
                lines.append(f"{spaces}- {item!r}")
        return "\n".join(lines)
    elif isinstance(value, dict):
        if not value:
            return "{}"
        lines = []
        for k, v in value.items():
            lines.append(f"{spaces}{k}: {v!r}")
        return "\n".join(lines)
    else:
        return repr(value)


def flatten_suite(data: dict) -> dict:
    """Flatten nested fields into JSON strings for CSV output."""
    flat = {}
    for key, value in data.items():
        if isinstance(value, (list, dict)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value
    return flat


def save_suites_csv(suites: list, output_path: str) -> None:
    """Save a list of suite dicts to a CSV file."""
    if not suites:
        print(f"[get_suite_info] No suites to save")
        return
    flattened = [flatten_suite(s) for s in suites]
    fieldnames = sorted(flattened[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)
    print(f"[get_suite_info] Saved {len(suites)} suites to {output_path}")


def flatten_testcase(suite: dict, spec: dict, scenario_data: dict) -> dict:
    """Flatten suite, spec, and scenario data into a single test case row."""
    tc = {
        # Suite info
        "suite_id": suite.get("id", ""),
        "suite_name": suite.get("display_name", ""),
        "suite_description": suite.get("description", ""),
        "suite_type": suite.get("type", ""),
        "suite_created_at": suite.get("created_at", ""),
        "suite_created_by": suite.get("created_by", ""),
        "suite_version_id": suite.get("version_id", ""),
        # Scenario ref info
        "scenario_id": spec.get("scenario_id", ""),
        "scenario_display_name": spec.get("scenario_display_name", ""),
        "scenario_version_id": spec.get("scenario_version_id", ""),
        "parameter_set_overrides": json.dumps(spec.get("parameter_set_overrides", []), ensure_ascii=False),
        # Full scenario details
        "scenario_name": scenario_data.get("name", ""),
        "scenario_description": scenario_data.get("description", ""),
        "scenario_created_at": scenario_data.get("created_at", ""),
        "scenario_updated_at": scenario_data.get("updated_at", ""),
        "scenario_created_by": scenario_data.get("created_by", ""),
        "scenario_updated_by": scenario_data.get("updated_by", ""),
        "t4_dataset_ids": json.dumps(scenario_data.get("t4_dataset_ids", []), ensure_ascii=False),
    }
    # Add labels as JSON
    suite_labels = suite.get("labels", [])
    if isinstance(suite_labels, list):
        tc["suite_labels"] = json.dumps(suite_labels, ensure_ascii=False)
    else:
        tc["suite_labels"] = ""
    # Add attachments as JSON
    attachments = suite.get("attachments", [])
    if isinstance(attachments, list):
        tc["suite_attachments"] = json.dumps(attachments, ensure_ascii=False)
    else:
        tc["suite_attachments"] = ""
    return tc


def save_testcases_csv(testcases: list, output_path: str) -> None:
    """Save a list of test case dicts to a CSV file."""
    if not testcases:
        print(f"[get_suite_info] No test cases to save")
        return
    fieldnames = sorted(testcases[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(testcases)
    print(f"[get_suite_info] Saved {len(testcases)} test cases to {output_path}")


# ---------------------------------------------------------------------------
# Download command generation
# ---------------------------------------------------------------------------

def build_dataset_table(testcases_csv_path: str, project_id: str, version_id: str) -> list[dict]:
    """Build a table of unique t4_dataset_ids with rich context from testcases.csv."""
    dataset_map: dict = defaultdict(lambda: {"scenarios": [], "suites": set()})

    with open(testcases_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t4_ids_str = row.get("t4_dataset_ids", "")
            if not t4_ids_str:
                continue
            try:
                t4_ids = json.loads(t4_ids_str)
            except json.JSONDecodeError:
                continue
            if not isinstance(t4_ids, list):
                continue

            suite_id = row.get("suite_id", "").strip()
            suite_name = row.get("suite_name", "").strip()
            suite_type = row.get("suite_type", "").strip()
            suite_version_id = row.get("suite_version_id", "").strip()
            scenario_id = row.get("scenario_id", "").strip()
            scenario_name = row.get("scenario_name", "").strip()
            scenario_display_name = row.get("scenario_display_name", "").strip()
            scenario_desc = row.get("scenario_description", "").strip()
            scenario_version_id = row.get("scenario_version_id", "").strip()

            for ds_id in t4_ids:
                if not isinstance(ds_id, str) or not ds_id.strip():
                    continue
                ds_id = ds_id.strip()
                entry = dataset_map[ds_id]
                entry["scenarios"].append({
                    "suite_id": suite_id,
                    "suite_name": suite_name,
                    "suite_type": suite_type,
                    "suite_version_id": suite_version_id,
                    "scenario_id": scenario_id,
                    "scenario_name": scenario_name,
                    "scenario_display_name": scenario_display_name,
                    "scenario_desc": scenario_desc,
                    "scenario_version_id": scenario_version_id,
                })
                entry["suites"].add(suite_id)

    rows = []
    for ds_id, entry in sorted(dataset_map.items()):
        scenarios = entry["scenarios"]
        suite_ids = list(sorted(entry["suites"]))
        suite_names = list(sorted(set(s["suite_name"] for s in scenarios)))
        scenario_names = list(sorted(set(s["scenario_name"] for s in scenarios)))
        scenario_display_names = list(sorted(set(s["scenario_display_name"] for s in scenarios)))
        suite_types = list(sorted(set(s["suite_type"] for s in scenarios)))
        scenarios_per_suite = defaultdict(int)
        for s in scenarios:
            scenarios_per_suite[s["suite_name"]] += 1
        scenario_ids = list(sorted(set(s["scenario_id"] for s in scenarios)))
        download_cmd = (
            f"webauto data annotation-dataset pull "
            f"--project-id {project_id} "
            f"--annotation-dataset-id {ds_id} "
            f"--annotation-dataset-version-id {version_id}"
        )
        rows.append({
            "dataset_id": ds_id,
            "download_command": download_cmd,
            "project_id": project_id,
            "version_id": version_id,
            "total_scenarios": len(scenarios),
            "total_suites": len(suite_ids),
            "suite_names": "; ".join(suite_names),
            "suite_ids": "; ".join(suite_ids),
            "suite_types": "; ".join(suite_types),
            "scenario_count_per_suite": "; ".join(
                f"{name}({cnt})" for name, cnt in sorted(scenarios_per_suite.items())
            ),
            "scenario_names": "; ".join(scenario_names),
            "scenario_display_names": "; ".join(scenario_display_names),
            "scenario_ids": "; ".join(scenario_ids),
            "scenario_descriptions": " || ".join(
                sorted(set(s["scenario_desc"] for s in scenarios if s["scenario_desc"]))
            ),
        })
    return rows


def save_dataset_csv(rows: list[dict], output_path: str):
    """Save dataset table to CSV."""
    if not rows:
        print(f"[get_suite_info] No datasets to save")
        return
    fieldnames = [
        "dataset_id", "download_command", "project_id", "version_id",
        "total_scenarios", "total_suites", "suite_names", "suite_ids",
        "suite_types", "scenario_count_per_suite", "scenario_names",
        "scenario_display_names", "scenario_ids", "scenario_descriptions",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[get_suite_info] Saved {len(rows)} datasets to {output_path}")


def generate_download_script(rows: list[dict], output_path: str):
    """Generate a shell script with webauto pull commands for all datasets."""
    if not rows:
        print(f"[get_suite_info] No datasets to generate commands for")
        return
    lines = []
    lines.append("#!/bin/bash")
    lines.append(f"# Auto-generated by get_suite_info.py")
    lines.append(f"# Total datasets: {len(rows)}")
    lines.append("")
    lines.append("set -e  # Stop on first error (remove if you want to continue on error)")
    lines.append("")
    lines.append("OUTPUT_DIR=${OUTPUT_DIR:-./datasets}")
    lines.append('mkdir -p "$OUTPUT_DIR"')
    lines.append('cd "$OUTPUT_DIR"')
    lines.append("")
    lines.append("# --- Download commands ---")
    for row in rows:
        lines.append(row["download_command"])
    lines.append("")
    lines.append('echo "All downloads complete!"')
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(output_path, 0o755)
    print(f"[get_suite_info] Generated download script ({len(rows)} commands) -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch suite/testcase info from a Tier4 vehicle catalog and generate download artifacts."
    )
    parser.add_argument("project_id", nargs="?", default="x2_dev",
                        help="Project ID (default: x2_dev)")
    parser.add_argument("vehicle_catalog_id", nargs="?", default="e2efe01d-e0c6-4d49-8223-817ff5d73204",
                        help="Vehicle catalog ID (default: Perception_Performance_X2_Gen2)")
    parser.add_argument("limit", nargs="?", type=int, default=999999,
                        help="Max number of suites to fetch details for (default: all)")
    parser.add_argument("output_dir", nargs="?", default=".",
                        help="Output directory (default: .)")
    parser.add_argument("--download-project-id", "-p", default="x2_dev",
                        help="Project ID for download commands (default: same as project_id)")
    parser.add_argument("--download-version-id", "-v", default="0",
                        help="Annotation dataset version ID for download commands (default: 0)")
    parser.add_argument("--no-download-script", action="store_true",
                        help="Skip generating t4datasets.csv and download_commands.sh")
    args = parser.parse_args()

    project_id = args.project_id
    vehicle_catalog_id = args.vehicle_catalog_id
    limit = args.limit
    output_dir = args.output_dir
    download_project_id = args.download_project_id
    download_version_id = args.download_version_id

    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching suites for catalog {vehicle_catalog_id} (project: {project_id})...\n")
    suites = list_catalog_suites(project_id, vehicle_catalog_id)
    print(f"Found {len(suites)} suites\n")

    # Save all catalog suites to CSV
    catalog_csv = os.path.join(output_dir, "catalog_suites.csv")
    save_suites_csv(suites, catalog_csv)

    full_details = []
    testcases = []
    for i, suite in enumerate(suites[:limit], 1):
        suite_id = suite.get("id")
        print(f"{'='*70}")
        print(f"Suite {i} (ID: {suite_id})")
        print(f"{'='*70}")

        # Fetch full details for this suite
        full = get_suite_info(project_id, suite_id)
        full_details.append(full)

        # Collect test cases (scenarios) with full details
        specs = full.get("specs") or []
        print(f"[get_suite_info] Suite has {len(specs)} scenarios, fetching details...")
        for spec in specs:
            scenario_id = spec.get("scenario_id")
            scenario_version_id = spec.get("scenario_version_id")
            if scenario_version_id and isinstance(scenario_version_id, str):
                try:
                    scenario_version_id = int(scenario_version_id)
                except (ValueError, TypeError):
                    scenario_version_id = None
            scenario_data = get_scenario_info(project_id, scenario_id, scenario_version_id) if scenario_id else {}
            tc = flatten_testcase(full, spec, scenario_data)
            testcases.append(tc)

        # Print ALL fields with proper formatting
        for key in sorted(full.keys()):
            value = full[key]
            if key == "specs" and isinstance(value, list):
                print(f"\n  {key}: [{len(value)} scenarios]")
                for spec in value[:5]:
                    print(f"    - scenario_id: {spec.get('scenario_id')}")
                    print(f"      display_name: {spec.get('scenario_display_name')}")
                if len(value) > 5:
                    print(f"    ... and {len(value) - 5} more")
            elif key == "attachments" and isinstance(value, list):
                print(f"\n  {key}: [{len(value)} attachments]")
                for att in value:
                    print(f"    - catalog_display_name: {att.get('catalog_display_name')}")
                    print(f"      catalog_id: {att.get('catalog_id')}")
            elif key == "labels" and isinstance(value, list):
                print(f"\n  {key}:")
                for label in value:
                    print(f"    - {label.get('key')}: {label.get('value')}")
            elif isinstance(value, (list, dict)) and value:
                print(f"\n  {key}:")
                print(f"    {format_value(value, 4)}")
            else:
                print(f"  {key}: {format_value(value)}")
        print()

    # Save full suite details to CSV
    if full_details:
        details_csv = os.path.join(output_dir, "suite_details.csv")
        save_suites_csv(full_details, details_csv)

    # Save test cases (scenarios) to CSV
    if testcases:
        testcases_csv = os.path.join(output_dir, "testcases.csv")
        save_testcases_csv(testcases, testcases_csv)

        # Generate dataset table + download script
        if not args.no_download_script:
            print()
            dataset_rows = build_dataset_table(testcases_csv, download_project_id, download_version_id)
            if dataset_rows:
                datasets_csv = os.path.join(output_dir, "t4datasets.csv")
                save_dataset_csv(dataset_rows, datasets_csv)

                script_path = os.path.join(output_dir, "download_commands.sh")
                generate_download_script(dataset_rows, script_path)

    if len(suites) > limit:
        print(f"\n... (fetching details for first {limit} of {len(suites)} suites)")
