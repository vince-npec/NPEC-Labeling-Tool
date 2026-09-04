#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resources.hades_data import ensure_hades_series_folders


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Group flat Hades image files into one subfolder per Petri dish."
    )
    parser.add_argument("root", type=Path, help="Folder containing the flat image set")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move the files. Without this flag, show what would be organized.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Folder not found: {root}")

    if args.apply:
        result = ensure_hades_series_folders(root)
        action = "Moved"
    else:
        from resources.hades_data import collect_flat_hades_groups

        grouped = collect_flat_hades_groups(root)
        if not grouped:
            print("No matching files found.")
            return 1
        for group_name in sorted(grouped):
            print(f"Would move {len(grouped[group_name]):>4} file(s) -> {group_name}/")
        return 0

    if not result.series_names:
        print("No matching files found.")
        return 1

    for group_name in sorted(result.series_names):
        print(f"{action} {result.series_counts[group_name]:>4} file(s) -> {group_name}/")
    if result.unmatched_root_files:
        print(f"Unmatched files left in place: {len(result.unmatched_root_files)}")
    if result.conflict_files:
        print(f"Files skipped due to conflicts: {len(result.conflict_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
