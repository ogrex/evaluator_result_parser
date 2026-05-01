#!/usr/bin/env python3
"""CLI wrapper for fetching suite information from Tier4 vehicle catalog.

This script provides a command-line interface to the suite_info module.
For programmatic use, import from t4_visualizer.suite_info instead.

Usage:
    python get_suite_info.py [project_id] [vehicle_catalog_id] [limit] [output_dir]

    # Exclude already downloaded datasets
    python get_suite_info.py x2_dev <catalog_id> 10 . --exclude-data-dir ./t4datasets
"""

import argparse
import os
import sys
from pathlib import Path

from t4_visualizer.suite_info import (
    format_value,
    flatten_testcase,
    save_suites_csv,
    save_testcases_csv,
    build_dataset_table,
    save_dataset_csv,
    generate_download_script,
    list_webauto_annotation_dataset_ids,
)
from t4_visualizer.downloader import list_catalog_suites, get_suite_info, get_scenario_info


def main():
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
    parser.add_argument("--no-download-script", action="store_true",
                        help="Skip generating t4datasets.csv and download_commands.sh")
    parser.add_argument("--exclude-data-dir", "-e", metavar="PATH",
                        help="Directory containing already-downloaded datasets to exclude from download list (default: ./t4datasets)")
    args = parser.parse_args()

    project_id = args.project_id
    vehicle_catalog_id = args.vehicle_catalog_id
    limit = args.limit
    output_dir = args.output_dir
    download_project_id = args.download_project_id
    exclude_data_dir = args.exclude_data_dir or "./t4datasets"

    os.makedirs(output_dir, exist_ok=True)

    # Get existing dataset IDs to exclude
    exclude_existing_ids = set()
    if exclude_data_dir and os.path.isdir(exclude_data_dir):
        exclude_existing_ids = set(list_webauto_annotation_dataset_ids(Path(exclude_data_dir)))
        print(f"[get_suite_info] Excluding {len(exclude_existing_ids)} existing datasets from {exclude_data_dir}\n")

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
            dataset_rows = build_dataset_table(
                testcases_csv, download_project_id,
                exclude_existing_ids=exclude_existing_ids
            )
            if dataset_rows:
                datasets_csv = os.path.join(output_dir, "t4datasets.csv")
                save_dataset_csv(dataset_rows, datasets_csv)

                script_path = os.path.join(output_dir, "download_commands.sh")
                generate_download_script(dataset_rows, script_path)

    if len(suites) > limit:
        print(f"\n... (fetching details for first {limit} of {len(suites)} suites)")


if __name__ == "__main__":
    main()
