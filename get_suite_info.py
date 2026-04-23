#!/usr/bin/env python3
"""Fetch and display ALL available suite information from a Tier4 vehicle catalog."""

import csv
import json
import os
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

if __name__ == "__main__":
    import sys

    project_id = sys.argv[1] if len(sys.argv) > 1 else "x2_dev"
    vehicle_catalog_id = sys.argv[2] if len(sys.argv) > 2 else "e2efe01d-e0c6-4d49-8223-817ff5d73204"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "."

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

    if len(suites) > limit:
        print(f"\n... (showing first {limit} of {len(suites)} suites)")
