from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.general_root_starter import SCHEMA_KEYS  # noqa: E402


DIMENSION_RE = re.compile(r"(?P<width>\d+)x(?P<height>\d+)")


def _parse_dimensions(folder_name: str) -> tuple[int, int]:
    match = DIMENSION_RE.search(str(folder_name))
    if not match:
        raise ValueError(f"Could not parse image dimensions from {folder_name!r}")
    return int(match.group("width")), int(match.group("height"))


def build_prmi_public_corpus(raw_root: Path, output_dir: Path) -> dict[str, object]:
    raw_root = Path(raw_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, object]] = []
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    crop_counts: dict[str, int] = {}
    positive_count = 0
    missing_pairs: list[dict[str, str]] = []

    for split in ("train", "val", "test"):
        label_dir = raw_root / split / "labels_image_gt"
        image_root = raw_root / split / "images"
        mask_root = raw_root / split / "masks_pixel_gt"
        if not label_dir.exists():
            continue
        split_index = 0
        for json_path in sorted(label_dir.glob("*.json")):
            folder_name = json_path.stem
            if folder_name.endswith(f"_{split}"):
                folder_name = folder_name[: -(len(split) + 1)]
            try:
                width, height = _parse_dimensions(folder_name)
            except ValueError:
                width = height = 0
            rows = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                image_name = str(row.get("image_name") or "").strip()
                mask_name = str(row.get("binary_mask") or "").strip()
                if not image_name or not mask_name:
                    continue
                image_path = image_root / folder_name / image_name
                mask_path = mask_root / folder_name / mask_name
                if not image_path.exists() or not mask_path.exists():
                    missing_pairs.append(
                        {
                            "split": split,
                            "folder": folder_name,
                            "image_name": image_name,
                            "mask_name": mask_name,
                        }
                    )
                    continue
                has_root = int(row.get("has_root", 0) or 0) == 1
                if has_root:
                    positive_count += 1
                crop = str(row.get("crop") or folder_name.split("_", 1)[0] or "unknown")
                crop_counts[crop] = int(crop_counts.get(crop, 0) + 1)
                sample_id = f"prmi_{split}_{split_index:06d}_{Path(image_name).stem}"
                split_index += 1
                split_counts[split] = int(split_counts.get(split, 0) + 1)
                samples.append(
                    {
                        "sample_id": sample_id,
                        "uid": Path(image_name).stem,
                        "name": image_name,
                        "source_name": "prmi_official",
                        "source_project_path": str(raw_root),
                        "family": "minirhizotron_public",
                        "split": split,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "width": int(width),
                        "height": int(height),
                        "tasks_available": {
                            "root_binary": True,
                            "shoot": False,
                            "primary_root": False,
                            "lateral_root": False,
                            "seed_crown": False,
                        },
                        "metadata": {
                            "crop": crop,
                            "location": str(row.get("location") or ""),
                            "tube_num": str(row.get("tube_num") or ""),
                            "date": str(row.get("date") or ""),
                            "depth": str(row.get("depth") or ""),
                            "dpi": str(row.get("dpi") or ""),
                            "has_root": bool(has_root),
                            "mask_foreground": "white",
                        },
                    }
                )

    manifest = {
        "name": "general_root_starter_corpus_prmi_public",
        "version": 1,
        "schema_keys": list(SCHEMA_KEYS),
        "sources": [
            {
                "project_path": str(raw_root),
                "source_name": "prmi_official",
                "family": "minirhizotron_public",
                "image_count": int(len(samples)),
                "positive_count": int(positive_count),
                "task_coverage": {
                    "root_binary": True,
                    "shoot": False,
                    "primary_root": False,
                    "lateral_root": False,
                    "seed_crown": False,
                },
                "crop_counts": crop_counts,
            }
        ],
        "family_counts": {"minirhizotron_public": int(len(samples))},
        "split_counts": split_counts,
        "sample_count": int(len(samples)),
        "samples": samples,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "raw_root": str(raw_root),
        "manifest_path": str(manifest_path),
        "sample_count": int(len(samples)),
        "positive_count": int(positive_count),
        "split_counts": split_counts,
        "crop_counts": crop_counts,
        "missing_pair_count": int(len(missing_pairs)),
        "missing_pairs_preview": missing_pairs[:50],
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a direct General Root Starter corpus manifest from the PRMI official dataset.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_prmi_public_corpus(args.raw_root, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
