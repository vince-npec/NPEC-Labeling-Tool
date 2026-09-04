from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.app import LUCIFER_PIXEL_SIZE_MM, _decompose_main_lateral_root_masks  # noqa: E402
from resources.bw_arabidopsis_ownership import (  # noqa: E402
    _measure_mask_length_px,
    build_bw_arabidopsis_measurements,
)
from resources.five_seedling_ownership import (  # noqa: E402
    FiveSeedlingSample,
    build_ownership_targets,
    discover_five_seedling_corpus,
    grouped_split,
    load_semantic_masks,
)
from resources.models import DatasetImageItem  # noqa: E402
from resources.pyphenotyper_adapter import PyPhenotyperConfig, run_pyphenotyper_item  # noqa: E402


DEFAULT_CORPUS = REPO_ROOT / "data" / "yang_ground_truth"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "yang_hybrid_5plate_ownership_challenge"
DEFAULT_SEED = 20260723
OWNER_COLORS_RGB = (
    (238, 82, 83),
    (39, 174, 160),
    (245, 183, 49),
    (113, 99, 193),
    (69, 137, 210),
)
SEMANTIC_COLORS_RGB = {
    1: (255, 139, 36),
    2: (220, 47, 120),
    3: (82, 145, 246),
}


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_rgb(path: Path, image: np.ndarray, *, quality: int = 94) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if path.suffix.lower() in {".jpg", ".jpeg"} else []
    if not cv2.imwrite(str(path), bgr, params):
        raise OSError(f"Unable to write image: {path}")


def _write_index_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8)):
        raise OSError(f"Unable to write mask: {path}")


def _score_binary(predicted: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    pred = np.asarray(predicted, dtype=np.uint8) > 0
    truth = np.asarray(expected, dtype=np.uint8) > 0
    intersection = int(np.count_nonzero(pred & truth))
    union = int(np.count_nonzero(pred | truth))
    total = int(np.count_nonzero(pred)) + int(np.count_nonzero(truth))
    return {
        "iou": 1.0 if union <= 0 else float(intersection / union),
        "dice": 1.0 if total <= 0 else float((2 * intersection) / total),
    }


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
        dice_scores.append(_score_binary(pred_mask, truth_mask)["dice"])
        truth_length, _area, _perimeter = _measure_mask_length_px(truth_mask)
        pred_length, _area, _perimeter = _measure_mask_length_px(pred_mask)
        if truth_length > 0.0:
            length_errors.append(float(abs(pred_length - truth_length) / truth_length))
    return {
        "pixel_accuracy": float(accuracy),
        "macro_dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "unassigned_fraction": float(unassigned),
        "root_length_mape": float(np.mean(length_errors)) if length_errors else 0.0,
    }


def _select_heldout_samples(
    corpus: Path,
    *,
    count: int,
    seed: int,
) -> list[FiveSeedlingSample]:
    samples = discover_five_seedling_corpus(corpus)
    _train, holdout = grouped_split(samples)
    by_group: dict[str, list[FiveSeedlingSample]] = {}
    for sample in holdout:
        by_group.setdefault(sample.plate_group, []).append(sample)
    if len(by_group) < int(count):
        raise ValueError(f"Only {len(by_group)} held-out plate groups are available; requested {count}.")
    rng = random.Random(int(seed))
    selected: list[FiveSeedlingSample] = []
    for group in rng.sample(sorted(by_group), int(count)):
        selected.append(rng.choice(sorted(by_group[group], key=lambda sample: sample.sample_id)))
    return selected


def _hybrid_config() -> PyPhenotyperConfig:
    return PyPhenotyperConfig(
        pipeline_dir=REPO_ROOT / "resources" / "npec_pyphenotyper",
        root_model_path=(
            REPO_ROOT
            / "resources"
            / "builtin_models"
            / "hades_lucifer"
            / "best_root_model_patch_256_max_f10.8_max_IoU0.905.h5"
        ),
        shoot_model_path=(
            REPO_ROOT
            / "resources"
            / "builtin_models"
            / "rgb_inoculated"
            / "rgb_inoculated_shoot_v3.keras"
        ),
        patch_size=256,
        refinement_steps=2,
        root_class_id=1,
        shoot_class_id=2,
        lateral_class_id=3,
        seed_class_id=None,
        include_occlusion=False,
        enable_bbox_tracking=True,
        min_component_area=40,
        bbox_padding=12,
        tracking_search_margin=26,
        pixel_size_mm=float(LUCIFER_PIXEL_SIZE_MM),
        profile_overrides=None,
        root_profile_overrides=None,
        shoot_profile_overrides={
            "name": "rgb_inoculated_shoot_v3",
            "mode": "multiclass",
            "input_mode": "rgb",
            "flip_horizontal": False,
            "return_original_coords": True,
            "root_label_ids": (3,),
            "lateral_label_ids": (),
            "shoot_label_ids": (2,),
            "seed_label_ids": (1,),
            "include_seed_in_shoot": False,
            "batch_size": 8,
            "tile_halo": 32,
            "class_probability_scales": (1.0, 1.0, 1.0, 1.0),
        },
        pipeline_overrides={
            "shoot_postprocess_mode": "native",
            "adapter_shoot_guard_mode": "off",
            "ownership_backend": "bw_arabidopsis_seed_centers",
            "expected_plant_count": 5,
            "root_min_area": 1,
            "shoot_min_area": 1,
            "root_edge_margin_fraction": 0.0,
            "shoot_edge_margin_fraction": 0.0,
            "shoot_color_rescue_mode": "off",
            "lateral_class_id": 3,
        },
    )


def _prediction_with_lateral(raw_prediction: np.ndarray) -> np.ndarray:
    prediction = np.asarray(raw_prediction, dtype=np.uint8)
    root_union = (prediction == 1).astype(np.uint8)
    primary, lateral, _stats = _decompose_main_lateral_root_masks(root_union)
    output = np.zeros_like(prediction, dtype=np.uint8)
    output[primary > 0] = np.uint8(1)
    output[lateral > 0] = np.uint8(3)
    output[prediction == 2] = np.uint8(2)
    return output


def _ground_truth_prediction(semantic: dict[str, np.ndarray]) -> np.ndarray:
    prediction = np.zeros(semantic["root"].shape[:2], dtype=np.uint8)
    prediction[semantic["root"] > 0] = np.uint8(1)
    prediction[semantic["lateral"] > 0] = np.uint8(3)
    prediction[semantic["shoot"] > 0] = np.uint8(2)
    return prediction


def _owner_map_from_boxes(
    shape_hw: tuple[int, int],
    bboxes: dict[str, list[int]],
    root_union: np.ndarray,
) -> np.ndarray:
    owner = np.zeros(shape_hw, dtype=np.uint8)
    for slot in range(1, 6):
        raw = bboxes.get(f"lane_{slot:02d}") or bboxes.get(f"plant_{slot:02d}") or [0, 0, 0, 0]
        x, y, width, height = [int(value) for value in raw]
        x0 = max(0, min(x, shape_hw[1]))
        y0 = max(0, min(y, shape_hw[0]))
        x1 = max(x0, min(x + width, shape_hw[1]))
        y1 = max(y0, min(y + height, shape_hw[0]))
        owner[y0:y1, x0:x1] = np.uint8(slot)
    owner[np.asarray(root_union, dtype=np.uint8) == 0] = 0
    return owner


def _lane_payload(
    sample: FiveSeedlingSample,
    image: np.ndarray,
    prediction: np.ndarray,
    config: PyPhenotyperConfig,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    item = DatasetImageItem(sample.sample_id, sample.image_path.name, sample.image_path, image)
    payload = build_bw_arabidopsis_measurements([item], {item.uid: prediction}, config)
    frame = dict((payload.get("per_uid") or {}).get(item.uid) or {})
    rows = list((payload.get("summary") or {}).get("measurements") or [])
    root_union = np.isin(prediction, (1, 3)).astype(np.uint8)
    owner = _owner_map_from_boxes(
        root_union.shape[:2],
        dict(frame.get("bboxes_xywh") or {}),
        root_union,
    )
    return owner, frame, rows


def _semantic_overlay(image: np.ndarray, prediction: np.ndarray, alpha: float = 0.82) -> np.ndarray:
    output = np.asarray(image, dtype=np.float32).copy()
    for class_id, color in SEMANTIC_COLORS_RGB.items():
        mask = np.asarray(prediction, dtype=np.uint8) == np.uint8(class_id)
        if np.any(mask):
            output[mask] = ((1.0 - alpha) * output[mask]) + (alpha * np.asarray(color, dtype=np.float32))
    return np.clip(output, 0, 255).astype(np.uint8)


def _owner_overlay(image: np.ndarray, owner: np.ndarray, alpha: float = 0.88) -> np.ndarray:
    output = np.asarray(image, dtype=np.float32).copy()
    for slot, color in enumerate(OWNER_COLORS_RGB, start=1):
        mask = np.asarray(owner, dtype=np.uint8) == np.uint8(slot)
        if np.any(mask):
            output[mask] = ((1.0 - alpha) * output[mask]) + (alpha * np.asarray(color, dtype=np.float32))
    return np.clip(output, 0, 255).astype(np.uint8)


def _annotated_panel(image: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    panel = np.asarray(image, dtype=np.uint8).copy()
    scale = max(0.75, panel.shape[1] / 2600.0)
    thickness = max(2, int(round(scale * 2)))
    line_height = max(32, int(round(44 * scale)))
    pad = max(16, int(round(22 * scale)))
    box_height = pad * 2 + line_height * (1 + len(lines))
    cv2.rectangle(panel, (0, 0), (panel.shape[1], box_height), (12, 16, 23), thickness=-1)
    cv2.putText(
        panel,
        title,
        (pad, pad + line_height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    for index, line in enumerate(lines, start=1):
        cv2.putText(
            panel,
            line,
            (pad, pad + line_height * (index + 1) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.70,
            (210, 220, 232),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    return panel


def _comparison_image(
    image: np.ndarray,
    prediction: np.ndarray,
    model_owner: np.ndarray,
    oracle_lane_owner: np.ndarray,
    truth_owner: np.ndarray,
    metrics: dict[str, object],
) -> np.ndarray:
    panels = [
        _annotated_panel(
            _semantic_overlay(image, prediction),
            "Hybrid semantic prediction",
            [
                f"root IoU {float(metrics['root_union_iou']):.3f}",
                f"shoot IoU {float(metrics['shoot_iou']):.3f}",
            ],
        ),
        _annotated_panel(
            _owner_overlay(image, model_owner),
            "Model + spatial lanes",
            [
                f"owner Dice {float(metrics['model_lane_macro_dice']):.3f}",
                f"unassigned {100.0 * float(metrics['model_lane_unassigned_fraction']):.1f}%",
            ],
        ),
        _annotated_panel(
            _owner_overlay(image, oracle_lane_owner),
            "Perfect masks + spatial lanes",
            [
                f"owner Dice {float(metrics['oracle_lane_macro_dice']):.3f}",
                "isolates ownership algorithm",
            ],
        ),
        _annotated_panel(
            _owner_overlay(image, truth_owner),
            "Hand-labelled ownership",
            ["conflicts excluded", "bacterial gaps remain unknown"],
        ),
    ]
    target_height = 1000
    resized: list[np.ndarray] = []
    for panel in panels:
        width = max(1, int(round(panel.shape[1] * target_height / panel.shape[0])))
        resized.append(cv2.resize(panel, (width, target_height), interpolation=cv2.INTER_AREA))
    return np.concatenate(resized, axis=1)


def _measure_slot(
    mask: np.ndarray,
    *,
    pixel_size_mm: float,
) -> tuple[float, float, int]:
    length_px, area_px, _perimeter = _measure_mask_length_px(mask.astype(np.uint8))
    return (
        float(length_px),
        float(length_px * pixel_size_mm),
        int(area_px),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["sample_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_challenge(
    corpus: Path,
    output_dir: Path,
    *,
    count: int,
    seed: int,
    reuse_predictions: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _select_heldout_samples(corpus, count=count, seed=seed)
    config = _hybrid_config()
    case_rows: list[dict[str, object]] = []
    measurement_rows: list[dict[str, object]] = []
    comparison_paths: list[Path] = []

    for sample_index, sample in enumerate(samples, start=1):
        print(f"[{sample_index}/{len(samples)}] {sample.sample_id}", flush=True)
        image = _read_rgb(sample.image_path)
        cache_path = output_dir / "masks" / f"{sample.sample_id}__hybrid_indexed.png"
        if reuse_predictions and cache_path.exists():
            cached = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
            if cached is None or tuple(cached.shape[:2]) != tuple(image.shape[:2]):
                raise ValueError(f"Invalid cached prediction: {cache_path}")
            prediction = np.asarray(cached, dtype=np.uint8)
        else:
            item = DatasetImageItem(sample.sample_id, sample.image_path.name, sample.image_path, image)
            raw_prediction, _details = run_pyphenotyper_item(item, config)
            prediction = _prediction_with_lateral(raw_prediction)
            _write_index_mask(cache_path, prediction)

        semantic = load_semantic_masks(sample)
        truth_prediction = _ground_truth_prediction(semantic)
        target = build_ownership_targets(sample, owner_organs=("root", "lateral"))
        truth_root_union = np.logical_or(semantic["root"] > 0, semantic["lateral"] > 0).astype(np.uint8)
        model_root_union = np.isin(prediction, (1, 3)).astype(np.uint8)

        model_owner, model_frame, model_rows = _lane_payload(sample, image, prediction, config)
        oracle_owner, oracle_frame, _oracle_rows = _lane_payload(sample, image, truth_prediction, config)
        model_scores = _ownership_scores(model_owner, target.owner_mask, target.valid_owner_pixels)
        oracle_scores = _ownership_scores(oracle_owner, target.owner_mask, target.valid_owner_pixels)
        root_scores = _score_binary(model_root_union, truth_root_union)
        shoot_scores = _score_binary(prediction == 2, semantic["shoot"])
        primary_scores = _score_binary(prediction == 1, semantic["root"])
        lateral_scores = _score_binary(prediction == 3, semantic["lateral"])
        overlap_risk = bool(any(bool(row.get("neighbor_overlap_risk")) for row in model_rows))

        row = {
            "sample_id": sample.sample_id,
            "plate_group": sample.plate_group,
            "selection_partition": "heldout",
            "root_union_iou": root_scores["iou"],
            "root_union_dice": root_scores["dice"],
            "shoot_iou": shoot_scores["iou"],
            "shoot_dice": shoot_scores["dice"],
            "primary_iou": primary_scores["iou"],
            "primary_dice": primary_scores["dice"],
            "lateral_iou": lateral_scores["iou"],
            "lateral_dice": lateral_scores["dice"],
            "model_lane_pixel_accuracy": model_scores["pixel_accuracy"],
            "model_lane_macro_dice": model_scores["macro_dice"],
            "model_lane_unassigned_fraction": model_scores["unassigned_fraction"],
            "model_lane_root_length_mape": model_scores["root_length_mape"],
            "oracle_lane_pixel_accuracy": oracle_scores["pixel_accuracy"],
            "oracle_lane_macro_dice": oracle_scores["macro_dice"],
            "oracle_lane_unassigned_fraction": oracle_scores["unassigned_fraction"],
            "oracle_lane_root_length_mape": oracle_scores["root_length_mape"],
            "conflict_pixels_ignored": int(np.count_nonzero(target.conflict_pixels)),
            "overlap_risk": overlap_risk,
            "lane_center_source": str(model_frame.get("lane_center_source") or ""),
        }
        case_rows.append(row)

        for slot in range(1, 6):
            gt_mask = ((target.owner_mask == slot) & (target.valid_owner_pixels > 0)).astype(np.uint8)
            pred_slot = model_owner == slot
            pred_primary = (pred_slot & (prediction == 1)).astype(np.uint8)
            pred_lateral = (pred_slot & (prediction == 3)).astype(np.uint8)
            pred_total = np.logical_or(pred_primary > 0, pred_lateral > 0).astype(np.uint8)
            gt_px, gt_mm, gt_area = _measure_slot(gt_mask, pixel_size_mm=float(config.pixel_size_mm))
            pred_px, pred_mm, pred_area = _measure_slot(pred_total, pixel_size_mm=float(config.pixel_size_mm))
            primary_px, primary_mm, _primary_area = _measure_slot(
                pred_primary,
                pixel_size_mm=float(config.pixel_size_mm),
            )
            lateral_px, lateral_mm, _lateral_area = _measure_slot(
                pred_lateral,
                pixel_size_mm=float(config.pixel_size_mm),
            )
            lane_id = f"lane_{slot:02d}"
            lane_row = next((candidate for candidate in model_rows if candidate.get("lane_id") == lane_id), {})
            measurement_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "lane_id": lane_id,
                    "ownership_claim": False,
                    "spatial_lane_qc_tier": lane_row.get("spatial_lane_qc_tier", ""),
                    "neighbor_overlap_risk": bool(lane_row.get("neighbor_overlap_risk", False)),
                    "ground_truth_total_root_length_px": gt_px,
                    "ground_truth_total_root_length_mm": gt_mm,
                    "ground_truth_root_area_px": gt_area,
                    "predicted_total_root_length_px": pred_px,
                    "predicted_total_root_length_mm": pred_mm,
                    "predicted_root_area_px": pred_area,
                    "predicted_primary_root_length_px": primary_px,
                    "predicted_primary_root_length_mm": primary_mm,
                    "predicted_lateral_root_length_px": lateral_px,
                    "predicted_lateral_root_length_mm": lateral_mm,
                }
            )

        _write_index_mask(output_dir / "ownership" / f"{sample.sample_id}__model_lane_owner.png", model_owner)
        _write_index_mask(output_dir / "ownership" / f"{sample.sample_id}__oracle_lane_owner.png", oracle_owner)
        _write_index_mask(output_dir / "ownership" / f"{sample.sample_id}__ground_truth_owner.png", target.owner_mask)
        _write_rgb(
            output_dir / "overlays" / f"{sample.sample_id}__semantic_overlay.jpg",
            _semantic_overlay(image, prediction),
        )
        comparison_path = output_dir / "comparisons" / f"{sample.sample_id}__comparison.jpg"
        _write_rgb(
            comparison_path,
            _comparison_image(image, prediction, model_owner, oracle_owner, target.owner_mask, row),
            quality=92,
        )
        comparison_paths.append(comparison_path)

    _write_csv(output_dir / "five_plate_case_metrics.csv", case_rows)
    _write_csv(output_dir / "per_seedling_measurements.csv", measurement_rows)

    numeric_keys = [
        key
        for key in case_rows[0]
        if key not in {"sample_id", "plate_group", "selection_partition", "overlap_risk", "lane_center_source"}
        and isinstance(case_rows[0][key], (int, float))
    ]
    means = {
        key: float(np.mean([float(row[key]) for row in case_rows]))
        for key in numeric_keys
    }
    report = {
        "title": "Yang five-plate held-out hybrid ownership challenge",
        "corpus": str(corpus),
        "output_dir": str(output_dir),
        "selection_seed": int(seed),
        "selection_partition": "heldout plate groups",
        "selected_samples": [sample.sample_id for sample in samples],
        "models": {
            "root": str(config.root_model_path),
            "shoot": str(config.shoot_model_path),
            "primary_lateral": "whole-root topology decomposition",
            "ownership": "crown-aligned spatial-lane fallback",
        },
        "scientific_policy": {
            "ownership_claim": False,
            "bacterial_gaps": "unknown/background; not counted as visible ground truth",
            "conflicting_instance_pixels": "excluded from ownership scoring",
        },
        "mean": means,
        "overlap_risk_cases": int(sum(bool(row["overlap_risk"]) for row in case_rows)),
        "case_metrics_csv": str(output_dir / "five_plate_case_metrics.csv"),
        "per_seedling_csv": str(output_dir / "per_seedling_measurements.csv"),
        "comparison_images": [str(path) for path in comparison_paths],
    }
    (output_dir / "challenge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "selected_samples.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": sample.sample_id,
                    "plate_group": sample.plate_group,
                    "image_path": str(sample.image_path),
                }
                for sample in samples
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible five-plate Yang hybrid ownership challenge.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()
    run_challenge(
        args.corpus.expanduser(),
        args.output_dir.expanduser(),
        count=max(1, int(args.count)),
        seed=int(args.seed),
        reuse_predictions=bool(args.reuse_predictions),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
