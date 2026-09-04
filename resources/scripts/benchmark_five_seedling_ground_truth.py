from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.app import _decompose_main_lateral_root_masks  # noqa: E402
from resources.bw_arabidopsis_ownership import (  # noqa: E402
    _measure_mask_length_px,
    build_bw_arabidopsis_measurements,
)
from resources.five_seedling_ownership import (  # noqa: E402
    build_ownership_targets,
    discover_five_seedling_corpus,
    load_semantic_masks,
)
from resources.models import DatasetImageItem  # noqa: E402
from resources.pyphenotyper_adapter import PyPhenotyperConfig  # noqa: E402


DEFAULT_CORPUS = REPO_ROOT / "data" / "yang_ground_truth"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "five_seedling_ground_truth_benchmark"


def _score_binary(predicted: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(predicted, dtype=np.uint8) > 0
    truth = np.asarray(expected, dtype=np.uint8) > 0
    intersection = int(np.count_nonzero(pred & truth))
    union = int(np.count_nonzero(pred | truth))
    total = int(np.count_nonzero(pred)) + int(np.count_nonzero(truth))
    iou = 1.0 if union <= 0 else float(intersection / union)
    dice = 1.0 if total <= 0 else float((2 * intersection) / total)
    return iou, dice


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _prediction_from_semantics(masks: dict[str, np.ndarray]) -> np.ndarray:
    shape_hw = masks["root"].shape[:2]
    prediction = np.zeros(shape_hw, dtype=np.uint8)
    prediction[masks["root"] > 0] = 1
    prediction[masks["lateral"] > 0] = 3
    prediction[masks["shoot"] > 0] = 2
    return prediction


def _owner_map_from_boxes(
    shape_hw: tuple[int, int],
    bboxes: dict[str, list[int]],
    root_union: np.ndarray,
) -> np.ndarray:
    owner = np.zeros(shape_hw, dtype=np.uint8)
    for slot in range(1, 6):
        raw = (
            bboxes.get(f"lane_{slot:02d}")
            or bboxes.get(f"plant_{slot:02d}")
            or [0, 0, 0, 0]
        )
        x, y, width, height = [int(value) for value in raw]
        x0 = max(0, min(x, shape_hw[1]))
        y0 = max(0, min(y, shape_hw[0]))
        x1 = max(x0, min(x + width, shape_hw[1]))
        y1 = max(y0, min(y + height, shape_hw[0]))
        owner[y0:y1, x0:x1] = np.uint8(slot)
    owner[np.asarray(root_union, dtype=np.uint8) == 0] = 0
    return owner


def _ownership_scores(
    predicted_owner: np.ndarray,
    expected_owner: np.ndarray,
    valid_pixels: np.ndarray,
) -> dict[str, float]:
    valid = np.asarray(valid_pixels, dtype=np.uint8) > 0
    expected = np.asarray(expected_owner, dtype=np.uint8)
    predicted = np.asarray(predicted_owner, dtype=np.uint8)
    valid_count = int(np.count_nonzero(valid))
    accuracy = 0.0 if valid_count <= 0 else float(np.count_nonzero((expected == predicted) & valid) / valid_count)
    unassigned = 0.0 if valid_count <= 0 else float(np.count_nonzero((predicted == 0) & valid) / valid_count)
    dice_scores: list[float] = []
    length_errors: list[float] = []
    for slot in range(1, 6):
        truth_mask = ((expected == slot) & valid).astype(np.uint8)
        pred_mask = ((predicted == slot) & valid).astype(np.uint8)
        if int(np.count_nonzero(truth_mask)) <= 0:
            continue
        _iou, dice = _score_binary(pred_mask, truth_mask)
        dice_scores.append(float(dice))
        truth_length, _area, _perimeter = _measure_mask_length_px(truth_mask)
        pred_length, _area, _perimeter = _measure_mask_length_px(pred_mask)
        if truth_length > 0.0:
            length_errors.append(float(abs(pred_length - truth_length) / truth_length))
    return {
        "ownership_pixel_accuracy": float(accuracy),
        "ownership_macro_dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "ownership_unassigned_fraction": float(unassigned),
        "root_length_mape": float(np.mean(length_errors)) if length_errors else 0.0,
    }


def run_benchmark(corpus: Path, output_dir: Path, *, limit: int | None = None) -> dict[str, object]:
    samples = discover_five_seedling_corpus(corpus)
    if limit is not None and int(limit) > 0:
        samples = samples[: int(limit)]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for sample in samples:
        image = _read_rgb(sample.image_path)
        semantic = load_semantic_masks(sample)
        target = build_ownership_targets(sample, owner_organs=("root", "lateral"))
        prediction = _prediction_from_semantics(semantic)
        item = DatasetImageItem(sample.sample_id, sample.image_path.name, sample.image_path, image)
        config = PyPhenotyperConfig(
            pipeline_dir=REPO_ROOT / "resources" / "npec_pyphenotyper",
            root_model_path=Path("unused-root.keras"),
            shoot_model_path=Path("unused-shoot.keras"),
            root_class_id=1,
            shoot_class_id=2,
            lateral_class_id=3,
            pixel_size_mm=111.88 / 4200.0,
            pipeline_overrides={"expected_plant_count": 5, "lateral_class_id": 3},
        )
        bbox_payload = build_bw_arabidopsis_measurements(
            [item],
            {item.uid: prediction},
            config,
        )
        frame_payload = dict((bbox_payload.get("per_uid") or {}).get(item.uid) or {})
        root_union = np.logical_or(semantic["root"] > 0, semantic["lateral"] > 0).astype(np.uint8)
        bbox_owner = _owner_map_from_boxes(
            root_union.shape[:2],
            dict(frame_payload.get("bboxes_xywh") or {}),
            root_union,
        )
        ownership = _ownership_scores(bbox_owner, target.owner_mask, target.valid_owner_pixels)

        primary_pred, lateral_pred, _stats = _decompose_main_lateral_root_masks(root_union)
        primary_iou, primary_dice = _score_binary(primary_pred, semantic["root"])
        lateral_iou, lateral_dice = _score_binary(lateral_pred, semantic["lateral"])
        measurement_rows = list((bbox_payload.get("summary") or {}).get("measurements") or [])
        rows.append(
            {
                "sample_id": sample.sample_id,
                "plate_group": sample.plate_group,
                **ownership,
                "primary_iou_from_perfect_root_union": float(primary_iou),
                "primary_dice_from_perfect_root_union": float(primary_dice),
                "lateral_iou_from_perfect_root_union": float(lateral_iou),
                "lateral_dice_from_perfect_root_union": float(lateral_dice),
                "conflict_pixels_ignored": int(np.count_nonzero(target.conflict_pixels)),
                "overlap_risk": bool(any(bool(row.get("neighbor_overlap_risk")) for row in measurement_rows)),
            }
        )

    csv_path = output_dir / "five_seedling_ground_truth_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)

    numeric_keys = [
        "ownership_pixel_accuracy",
        "ownership_macro_dice",
        "ownership_unassigned_fraction",
        "root_length_mape",
        "primary_iou_from_perfect_root_union",
        "primary_dice_from_perfect_root_union",
        "lateral_iou_from_perfect_root_union",
        "lateral_dice_from_perfect_root_union",
    ]
    summary = {
        key: float(np.mean([float(row[key]) for row in rows])) if rows else None
        for key in numeric_keys
    }
    report = {
        "corpus": str(corpus),
        "sample_count": len(rows),
        "benchmark_input": "perfect hand-labelled visible semantic masks",
        "scientific_scope": {
            "bacterial_gaps": "unknown/background; never counted as visible root",
            "conflicting_instance_pixels": "ignored",
            "bbox_output": "stable non-overlapping fallback, not topological ownership",
        },
        "mean": summary,
        "overlap_risk_cases": int(sum(bool(row["overlap_risk"]) for row in rows)),
        "case_csv": str(csv_path),
    }
    report_path = output_dir / "five_seedling_ground_truth_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark NPEC five-seedling ownership against hand labels.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test sample limit.")
    args = parser.parse_args()
    report = run_benchmark(args.corpus.expanduser(), args.output_dir.expanduser(), limit=args.limit)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
