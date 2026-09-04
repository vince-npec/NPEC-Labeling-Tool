from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resources.yolo_tip_anchors import export_yolo26_anchor_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO26 crown/tip anchor datasets from an NPEC .oclp project.")
    parser.add_argument("project", type=Path, help="Path to the source .oclp project.")
    parser.add_argument("output_dir", type=Path, help="Directory to write the YOLO26-ready dataset into.")
    parser.add_argument("--val-stride", type=int, default=5, help="Every Nth sorted image goes to validation. Default: 5")
    parser.add_argument("--tip-box-size", type=int, default=24, help="Primary tip box size in pixels for detect labels. Default: 24")
    parser.add_argument("--expected-tracks", type=int, default=0, help="Optional expected number of seedlings per frame. Default: 0 (auto)")
    args = parser.parse_args()

    summary = export_yolo26_anchor_dataset(
        args.project,
        args.output_dir,
        val_stride=max(2, int(args.val_stride)),
        tip_box_size_px=max(4, int(args.tip_box_size)),
        expected_track_count=max(0, int(args.expected_tracks)),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
