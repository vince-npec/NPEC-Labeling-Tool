from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.bw_arabidopsis_ownership import _measure_mask_length_px  # noqa: E402
from resources.five_seedling_ownership import (  # noqa: E402
    FiveSeedlingSample,
    build_ownership_targets,
    discover_five_seedling_corpus,
    load_semantic_masks,
)
from resources.models import DatasetImageItem  # noqa: E402
from resources.pyphenotyper_adapter import run_pyphenotyper_item  # noqa: E402
from resources.scripts.experiment_learned_owner_assignment import (  # noqa: E402
    SharedOwnerRanker,
    _orientation_channels,
    _pair_features,
    _sample_orientation,
    _sample_training_pixels,
)
from resources.scripts.run_yang_five_plate_hybrid_challenge import (  # noqa: E402
    OWNER_COLORS_RGB,
    _hybrid_config,
    _prediction_with_lateral,
    _read_rgb,
    _semantic_overlay,
    _write_index_mask,
    _write_rgb,
    _lane_payload,
)


DEFAULT_CORPUS = REPO_ROOT / "data" / "yang_ground_truth"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "yang_5plate_seedling_ownership_retrial"
DEFAULT_SHOOT_PROFILE = (
    REPO_ROOT
    / "docs"
    / "validation"
    / "yang_2026-07-23"
    / "semantic_split_profile.json"
)
DEFAULT_SEED = 20260723
FEATURE_SET = "crown_coordinates_orientation"
THRESHOLD_CANDIDATES = (
    0.0,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
    2.50,
    3.00,
    4.00,
    6.00,
)
_VALID_SAMPLE_RE = re.compile(r"^\d+-\d+$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["sample_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _shoot_holdout(
    samples: Iterable[FiveSeedlingSample],
    profile_path: Path,
) -> tuple[dict[str, list[FiveSeedlingSample]], dict[str, object]]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    split = dict(profile.get("split") or {})
    holdout = dict(split.get("holdout") or {})
    holdout_groups = {str(value) for value in holdout.get("plate_groups") or []}
    if len(holdout_groups) < 6:
        raise ValueError(f"Shoot profile does not provide enough held-out plate groups: {profile_path}")
    by_group: dict[str, list[FiveSeedlingSample]] = {}
    for sample in samples:
        if sample.plate_group in holdout_groups and _VALID_SAMPLE_RE.match(sample.sample_id):
            by_group.setdefault(sample.plate_group, []).append(sample)
    missing = sorted(holdout_groups - set(by_group))
    if missing:
        raise ValueError(f"Shoot holdout groups are absent from the annotation corpus: {missing}")
    return by_group, profile


def _split_evaluation_groups(
    by_group: dict[str, list[FiveSeedlingSample]],
    *,
    count: int,
    seed: int,
) -> tuple[list[FiveSeedlingSample], list[FiveSeedlingSample]]:
    if len(by_group) <= int(count):
        raise ValueError("At least one independent calibration plate group is required.")
    rng = random.Random(int(seed))
    challenge_groups = rng.sample(sorted(by_group), int(count))
    challenge: list[FiveSeedlingSample] = []
    for group in challenge_groups:
        challenge.append(rng.choice(sorted(by_group[group], key=lambda sample: sample.sample_id)))
    calibration = [
        sample
        for group, group_samples in sorted(by_group.items())
        if group not in set(challenge_groups)
        for sample in sorted(group_samples, key=lambda candidate: candidate.sample_id)
    ]
    return challenge, calibration


def _train_owner_ranker(
    samples: Iterable[FiveSeedlingSample],
    *,
    excluded_groups: set[str],
    max_pixels_per_owner: int,
) -> tuple[SharedOwnerRanker, dict[str, object]]:
    train_samples = [
        sample
        for sample in samples
        if sample.plate_group not in excluded_groups and _VALID_SAMPLE_RE.match(sample.sample_id)
    ]
    feature_batches: list[np.ndarray] = []
    owner_batches: list[np.ndarray] = []
    for sample in train_samples:
        features, owners = _sample_training_pixels(
            sample,
            max_pixels_per_owner=max_pixels_per_owner,
            feature_set=FEATURE_SET,
        )
        feature_batches.append(features)
        owner_batches.append(owners)
    if not feature_batches:
        raise ValueError("No owner-ranker training samples remain after group exclusions.")
    train_features = np.concatenate(feature_batches, axis=0)
    train_owners = np.concatenate(owner_batches, axis=0)
    ranker = SharedOwnerRanker(l2=1e-3).fit(train_features, train_owners)
    return ranker, {
        "feature_set": FEATURE_SET,
        "sample_count": len(train_samples),
        "plate_group_count": len({sample.plate_group for sample in train_samples}),
        "sample_ids": [sample.sample_id for sample in train_samples],
        "sampled_pixel_rows": int(train_owners.size),
        "optimization": dict(ranker.optimization_),
    }


def _prediction_manifest(
    sample: FiveSeedlingSample,
    *,
    root_model: Path,
    shoot_model: Path,
) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "image_path": str(sample.image_path),
        "image_sha256": _sha256_file(sample.image_path),
        "root_model_path": str(root_model),
        "root_model_sha256": _sha256_file(root_model),
        "shoot_model_path": str(shoot_model),
        "shoot_model_sha256": _sha256_file(shoot_model),
        "postprocess": "whole-root topology decomposition",
    }


def _load_or_run_prediction(
    sample: FiveSeedlingSample,
    output_dir: Path,
    *,
    reuse_predictions: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    config = _hybrid_config()
    image = _read_rgb(sample.image_path)
    mask_path = output_dir / "masks" / f"{sample.sample_id}__hybrid_indexed.png"
    manifest_path = output_dir / "masks" / f"{sample.sample_id}__hybrid_indexed.json"
    expected_manifest = _prediction_manifest(
        sample,
        root_model=Path(config.root_model_path),
        shoot_model=Path(config.shoot_model_path),
    )
    prediction: np.ndarray | None = None
    if reuse_predictions and mask_path.exists() and manifest_path.exists():
        actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual_manifest == expected_manifest:
            cached = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if cached is not None and tuple(cached.shape[:2]) == tuple(image.shape[:2]):
                prediction = np.asarray(cached, dtype=np.uint8)
    if prediction is None:
        item = DatasetImageItem(sample.sample_id, sample.image_path.name, sample.image_path, image)
        raw_prediction, _details = run_pyphenotyper_item(item, config)
        prediction = _prediction_with_lateral(raw_prediction)
        _write_index_mask(mask_path, prediction)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(expected_manifest, indent=2) + "\n", encoding="utf-8")
    return image, prediction, expected_manifest


def _automatic_crowns(
    lane_owner: np.ndarray,
    root_union: np.ndarray,
    frame: dict[str, object],
) -> np.ndarray:
    boxes = dict(frame.get("bboxes_xywh") or {})
    crowns: list[tuple[float, float]] = []
    for slot in range(1, 6):
        ys, xs = np.where((lane_owner == slot) & root_union)
        if xs.size:
            top_y = float(np.quantile(ys.astype(np.float64), 0.015))
            keep = ys <= top_y + 35.0
            crowns.append((float(np.median(xs[keep])), float(np.median(ys[keep]))))
            continue
        raw = boxes.get(f"lane_{slot:02d}") or boxes.get(f"plant_{slot:02d}") or [0, 0, 0, 0]
        x, y, width, _height = [float(value) for value in raw]
        crowns.append((x + (width * 0.5), y))
    result = np.asarray(crowns, dtype=np.float64)
    if result.shape != (5, 2) or np.any(~np.isfinite(result)):
        raise ValueError("Five finite automatic crown points are required.")
    return result


def _learned_owner_candidate(
    prediction: np.ndarray,
    crowns: np.ndarray,
    ranker: SharedOwnerRanker,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    root_union = np.isin(prediction, (1, 3))
    ys, xs = np.where(root_union)
    learned = np.zeros(root_union.shape, dtype=np.uint8)
    margin_image = np.zeros(root_union.shape, dtype=np.float32)
    orientation = _orientation_channels(root_union.astype(np.uint8))
    for start in range(0, xs.size, max(1, int(chunk_size))):
        stop = min(xs.size, start + max(1, int(chunk_size)))
        chunk_y = ys[start:stop]
        chunk_x = xs[start:stop]
        features = _pair_features(
            chunk_x,
            chunk_y,
            crowns,
            root_union.shape,
            feature_set=FEATURE_SET,
            tangent_xy=_sample_orientation(
                orientation,
                chunk_y,
                chunk_x,
                root_union.shape,
            ),
            lateral_flags=prediction[chunk_y, chunk_x] == 3,
        )
        scores = ranker.decision_scores(features)
        top_two = np.partition(scores, -2, axis=1)
        learned[chunk_y, chunk_x] = np.argmax(scores, axis=1).astype(np.uint8) + np.uint8(1)
        margin_image[chunk_y, chunk_x] = (top_two[:, -1] - top_two[:, -2]).astype(np.float32)
    return learned, margin_image


def _gated_owner(
    lane_owner: np.ndarray,
    learned_owner: np.ndarray,
    margin: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    lane = np.asarray(lane_owner, dtype=np.uint8)
    learned = np.asarray(learned_owner, dtype=np.uint8)
    use_learned = (
        (lane > 0)
        & (learned > 0)
        & (learned != lane)
        & (np.asarray(margin, dtype=np.float32) >= float(threshold))
    )
    gated = lane.copy()
    gated[use_learned] = learned[use_learned]
    return gated, use_learned.astype(np.uint8)


def _binary_dice(predicted: np.ndarray, expected: np.ndarray) -> float:
    pred = np.asarray(predicted, dtype=bool)
    truth = np.asarray(expected, dtype=bool)
    intersection = int(np.count_nonzero(pred & truth))
    denominator = int(np.count_nonzero(pred)) + int(np.count_nonzero(truth))
    return 1.0 if denominator <= 0 else float((2 * intersection) / denominator)


def _owner_scores(
    predicted_owner: np.ndarray,
    target: object,
) -> dict[str, float]:
    predicted = np.asarray(predicted_owner, dtype=np.uint8)
    expected = np.asarray(target.owner_mask, dtype=np.uint8)
    valid = np.asarray(target.valid_owner_pixels, dtype=np.uint8) > 0
    conflicts = np.asarray(target.conflict_pixels, dtype=np.uint8) > 0
    scored_prediction = predicted.copy()
    scored_prediction[conflicts] = 0

    valid_count = int(np.count_nonzero(valid))
    conditional_correct = int(np.count_nonzero((scored_prediction == expected) & valid))
    conditional_dice: list[float] = []
    full_dice: list[float] = []
    length_ape: list[float] = []
    absolute_length_error = 0.0
    total_truth_length = 0.0
    for owner in range(1, 6):
        truth_mask = (expected == owner) & valid
        conditional_pred = (scored_prediction == owner) & valid
        full_pred = scored_prediction == owner
        conditional_dice.append(_binary_dice(conditional_pred, truth_mask))
        full_dice.append(_binary_dice(full_pred, truth_mask))
        truth_length, _area, _perimeter = _measure_mask_length_px(truth_mask.astype(np.uint8))
        pred_length, _area, _perimeter = _measure_mask_length_px(full_pred.astype(np.uint8))
        if truth_length > 0.0:
            error = abs(float(pred_length) - float(truth_length))
            length_ape.append(float(error / truth_length))
            absolute_length_error += error
            total_truth_length += float(truth_length)
    return {
        "conditional_pixel_accuracy": float(conditional_correct / max(valid_count, 1)),
        "conditional_macro_dice": float(np.mean(conditional_dice)),
        "unassigned_truth_fraction": float(
            np.count_nonzero((scored_prediction == 0) & valid) / max(valid_count, 1)
        ),
        "visible_label_full_macro_dice": float(np.mean(full_dice)),
        "visible_label_root_length_mape": float(np.mean(length_ape)) if length_ape else 0.0,
        "visible_label_root_length_wape": float(
            absolute_length_error / max(total_truth_length, 1e-9)
        ),
    }


def _mean_metrics(cases: list[dict[str, object]], prefix: str) -> dict[str, float]:
    metric_names = (
        "conditional_pixel_accuracy",
        "conditional_macro_dice",
        "unassigned_truth_fraction",
        "visible_label_full_macro_dice",
        "visible_label_root_length_mape",
        "visible_label_root_length_wape",
    )
    return {
        metric: float(np.mean([float(case[f"{prefix}_{metric}"]) for case in cases]))
        for metric in metric_names
    }


def _threshold_case(
    prepared: dict[str, object],
    threshold: float,
) -> dict[str, object]:
    gated, switched = _gated_owner(
        np.asarray(prepared["lane_owner"], dtype=np.uint8),
        np.asarray(prepared["learned_owner"], dtype=np.uint8),
        np.asarray(prepared["margin"], dtype=np.float32),
        threshold=threshold,
    )
    scores = _owner_scores(gated, prepared["target"])
    root_pixels = int(np.count_nonzero(np.asarray(prepared["lane_owner"], dtype=np.uint8)))
    return {
        "sample_id": str(prepared["sample"].sample_id),
        **scores,
        "switched_root_fraction": float(np.count_nonzero(switched) / max(root_pixels, 1)),
    }


def _select_threshold(
    prepared_cases: list[dict[str, object]],
    thresholds: Iterable[float] = THRESHOLD_CANDIDATES,
) -> tuple[float, list[dict[str, object]]]:
    lane_cases: list[dict[str, object]] = []
    for prepared in prepared_cases:
        lane_scores = _owner_scores(np.asarray(prepared["lane_owner"]), prepared["target"])
        lane_cases.append({"sample_id": prepared["sample"].sample_id, **lane_scores})
    lane_mean_dice = float(
        np.mean([float(case["visible_label_full_macro_dice"]) for case in lane_cases])
    )
    lane_mean_mape = float(
        np.mean([float(case["visible_label_root_length_mape"]) for case in lane_cases])
    )
    lane_mean_wape = float(
        np.mean([float(case["visible_label_root_length_wape"]) for case in lane_cases])
    )

    rows: list[dict[str, object]] = []
    feasible: list[dict[str, object]] = []
    for threshold in thresholds:
        cases = [_threshold_case(prepared, float(threshold)) for prepared in prepared_cases]
        mean_dice = float(
            np.mean([float(case["visible_label_full_macro_dice"]) for case in cases])
        )
        mean_mape = float(
            np.mean([float(case["visible_label_root_length_mape"]) for case in cases])
        )
        mean_wape = float(
            np.mean([float(case["visible_label_root_length_wape"]) for case in cases])
        )
        worst_dice_delta = min(
            float(case["visible_label_full_macro_dice"])
            - float(lane["visible_label_full_macro_dice"])
            for case, lane in zip(cases, lane_cases, strict=True)
        )
        row = {
            "threshold": float(threshold),
            "mean_visible_label_full_macro_dice": mean_dice,
            "mean_visible_label_root_length_mape": mean_mape,
            "mean_visible_label_root_length_wape": mean_wape,
            "worst_plate_full_dice_delta": worst_dice_delta,
            "mean_switched_root_fraction": float(
                np.mean([float(case["switched_root_fraction"]) for case in cases])
            ),
            "passes_constraints": bool(
                mean_mape <= lane_mean_mape + 0.01
                and mean_wape <= lane_mean_wape + 0.01
                and worst_dice_delta >= -0.02
            ),
            "lane_mean_visible_label_full_macro_dice": lane_mean_dice,
            "lane_mean_visible_label_root_length_mape": lane_mean_mape,
            "lane_mean_visible_label_root_length_wape": lane_mean_wape,
        }
        rows.append(row)
        if bool(row["passes_constraints"]):
            feasible.append(row)
    candidates = feasible or rows
    selected = max(
        candidates,
        key=lambda row: (
            float(row["mean_visible_label_full_macro_dice"]),
            -float(row["mean_visible_label_root_length_mape"]),
            -float(row["mean_visible_label_root_length_wape"]),
            float(row["threshold"]),
        ),
    )
    return float(selected["threshold"]), rows


def _prepare_case(
    sample: FiveSeedlingSample,
    output_dir: Path,
    ranker: SharedOwnerRanker,
    *,
    reuse_predictions: bool,
    chunk_size: int,
) -> dict[str, object]:
    image, prediction, manifest = _load_or_run_prediction(
        sample,
        output_dir,
        reuse_predictions=reuse_predictions,
    )
    lane_owner, frame, lane_rows = _lane_payload(sample, image, prediction, _hybrid_config())
    root_union = np.isin(prediction, (1, 3))
    crowns = _automatic_crowns(lane_owner, root_union, frame)
    learned_owner, margin = _learned_owner_candidate(
        prediction,
        crowns,
        ranker,
        chunk_size=chunk_size,
    )
    return {
        "sample": sample,
        "image": image,
        "prediction": prediction,
        "lane_owner": lane_owner,
        "learned_owner": learned_owner,
        "margin": margin,
        "crowns": crowns,
        "target": build_ownership_targets(sample, owner_organs=("root", "lateral")),
        "lane_rows": lane_rows,
        "prediction_manifest": manifest,
    }


def _owner_overlay(image: np.ndarray, owner: np.ndarray, alpha: float = 0.88) -> np.ndarray:
    output = np.asarray(image, dtype=np.float32).copy()
    for slot, color in enumerate(OWNER_COLORS_RGB, start=1):
        mask = np.asarray(owner, dtype=np.uint8) == np.uint8(slot)
        if np.any(mask):
            output[mask] = (
                ((1.0 - alpha) * output[mask])
                + (alpha * np.asarray(color, dtype=np.float32))
            )
    return np.clip(output, 0, 255).astype(np.uint8)


def _label_panel(image: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    panel = np.asarray(image, dtype=np.uint8).copy()
    scale = max(0.70, panel.shape[1] / 2700.0)
    thickness = max(2, int(round(2 * scale)))
    line_height = max(30, int(round(42 * scale)))
    pad = max(14, int(round(20 * scale)))
    box_height = (2 * pad) + ((len(lines) + 1) * line_height)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], box_height), (12, 16, 23), thickness=-1)
    for index, text in enumerate([title, *lines]):
        cv2.putText(
            panel,
            text,
            (pad, pad + ((index + 1) * line_height) - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale if index == 0 else scale * 0.68,
            (255, 255, 255) if index == 0 else (210, 220, 232),
            thickness if index == 0 else max(1, thickness - 1),
            cv2.LINE_AA,
        )
    return panel


def _comparison(
    prepared: dict[str, object],
    gated_owner: np.ndarray,
    lane_scores: dict[str, float],
    gated_scores: dict[str, float],
    *,
    threshold: float,
) -> np.ndarray:
    image = np.asarray(prepared["image"], dtype=np.uint8)
    prediction = np.asarray(prepared["prediction"], dtype=np.uint8)
    target = prepared["target"]
    panels = [
        _label_panel(
            _semantic_overlay(image, prediction),
            "Hybrid semantic mask",
            ["orange primary | blue lateral | pink shoot"],
        ),
        _label_panel(
            _owner_overlay(image, prepared["lane_owner"]),
            "Existing spatial lanes",
            [
                f"full owner Dice {lane_scores['visible_label_full_macro_dice']:.3f}",
                f"length MAPE {lane_scores['visible_label_root_length_mape']:.3f}",
            ],
        ),
        _label_panel(
            _owner_overlay(image, gated_owner),
            "Confidence-gated learned owner",
            [
                f"margin threshold {threshold:.2f}",
                f"full owner Dice {gated_scores['visible_label_full_macro_dice']:.3f}",
                f"length MAPE {gated_scores['visible_label_root_length_mape']:.3f}",
            ],
        ),
        _label_panel(
            _owner_overlay(image, target.owner_mask),
            "Hand-labelled visible ownership",
            ["overlap conflicts excluded | bacterial gaps unknown"],
        ),
    ]
    target_height = 900
    resized: list[np.ndarray] = []
    for panel in panels:
        width = max(1, int(round(panel.shape[1] * target_height / panel.shape[0])))
        resized.append(cv2.resize(panel, (width, target_height), interpolation=cv2.INTER_AREA))
    return np.concatenate(resized, axis=1)


def _per_owner_rows(
    prepared: dict[str, object],
    gated_owner: np.ndarray,
) -> list[dict[str, object]]:
    target = prepared["target"]
    prediction = np.asarray(prepared["prediction"], dtype=np.uint8)
    rows: list[dict[str, object]] = []
    for slot in range(1, 6):
        truth = ((target.owner_mask == slot) & (target.valid_owner_pixels > 0)).astype(np.uint8)
        for method, owner_map in (
            ("spatial_lane", np.asarray(prepared["lane_owner"], dtype=np.uint8)),
            ("confidence_gated_learned", np.asarray(gated_owner, dtype=np.uint8)),
        ):
            owned = owner_map == slot
            total = (owned & np.isin(prediction, (1, 3))).astype(np.uint8)
            primary = (owned & (prediction == 1)).astype(np.uint8)
            lateral = (owned & (prediction == 3)).astype(np.uint8)
            truth_length, truth_area, _perimeter = _measure_mask_length_px(truth)
            total_length, total_area, _perimeter = _measure_mask_length_px(total)
            primary_length, _area, _perimeter = _measure_mask_length_px(primary)
            lateral_length, _area, _perimeter = _measure_mask_length_px(lateral)
            rows.append(
                {
                    "sample_id": prepared["sample"].sample_id,
                    "lane_id": f"lane_{slot:02d}",
                    "method": method,
                    "ownership_claim": False,
                    "ground_truth_visible_root_length_px": float(truth_length),
                    "ground_truth_visible_root_area_px": int(truth_area),
                    "predicted_total_root_length_px": float(total_length),
                    "predicted_root_area_px": int(total_area),
                    "predicted_primary_root_length_px": float(primary_length),
                    "predicted_lateral_root_length_px": float(lateral_length),
                }
            )
    return rows


def _write_contact_sheet(
    paths: Iterable[Path],
    output_path: Path,
    *,
    target_width: int = 1800,
) -> None:
    rows: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read comparison image: {path}")
        width = max(1, int(target_width))
        height = max(1, int(round(image.shape[0] * width / image.shape[1])))
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        rows.append(resized)
        rows.append(np.full((12, width, 3), 20, dtype=np.uint8))
    if not rows:
        return
    contact_sheet = np.concatenate(rows[:-1], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path),
        contact_sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    ):
        raise OSError(f"Unable to write contact sheet: {output_path}")


def run_retrial(
    corpus: Path,
    output_dir: Path,
    shoot_profile_path: Path,
    *,
    count: int,
    seed: int,
    max_pixels_per_owner: int,
    chunk_size: int,
    reuse_predictions: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_five_seedling_corpus(corpus)
    by_holdout_group, shoot_profile = _shoot_holdout(samples, shoot_profile_path)
    challenge, calibration = _split_evaluation_groups(
        by_holdout_group,
        count=count,
        seed=seed,
    )
    evaluation_groups = set(by_holdout_group)
    ranker, ranker_training = _train_owner_ranker(
        samples,
        excluded_groups=evaluation_groups,
        max_pixels_per_owner=max_pixels_per_owner,
    )
    (output_dir / "owner_ranker.json").write_text(
        json.dumps(
            {
                "feature_set": FEATURE_SET,
                "ranker": ranker.to_payload(),
                "training": ranker_training,
                "excluded_shoot_holdout_groups": sorted(evaluation_groups),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    calibration_prepared: list[dict[str, object]] = []
    for index, sample in enumerate(calibration, start=1):
        print(f"[calibration {index}/{len(calibration)}] {sample.sample_id}", flush=True)
        calibration_prepared.append(
            _prepare_case(
                sample,
                output_dir / "calibration",
                ranker,
                reuse_predictions=reuse_predictions,
                chunk_size=chunk_size,
            )
        )
    threshold, threshold_rows = _select_threshold(calibration_prepared)
    _write_csv(output_dir / "calibration_threshold_sweep.csv", threshold_rows)

    challenge_prepared: list[dict[str, object]] = []
    for index, sample in enumerate(challenge, start=1):
        print(f"[challenge {index}/{len(challenge)}] {sample.sample_id}", flush=True)
        challenge_prepared.append(
            _prepare_case(
                sample,
                output_dir / "challenge",
                ranker,
                reuse_predictions=reuse_predictions,
                chunk_size=chunk_size,
            )
        )

    case_rows: list[dict[str, object]] = []
    per_owner_rows: list[dict[str, object]] = []
    comparison_paths: list[Path] = []
    for prepared in challenge_prepared:
        sample = prepared["sample"]
        lane_owner = np.asarray(prepared["lane_owner"], dtype=np.uint8)
        gated_owner, switched = _gated_owner(
            lane_owner,
            prepared["learned_owner"],
            prepared["margin"],
            threshold=threshold,
        )
        lane_scores = _owner_scores(lane_owner, prepared["target"])
        gated_scores = _owner_scores(gated_owner, prepared["target"])
        root_pixels = int(np.count_nonzero(lane_owner))
        case_row: dict[str, object] = {
            "sample_id": sample.sample_id,
            "plate_group": sample.plate_group,
            "selected_margin_threshold": threshold,
            "switched_root_fraction": float(np.count_nonzero(switched) / max(root_pixels, 1)),
        }
        case_row.update({f"lane_{key}": value for key, value in lane_scores.items()})
        case_row.update({f"gated_{key}": value for key, value in gated_scores.items()})
        case_row["full_owner_dice_delta"] = float(
            gated_scores["visible_label_full_macro_dice"]
            - lane_scores["visible_label_full_macro_dice"]
        )
        case_row["root_length_mape_delta"] = float(
            gated_scores["visible_label_root_length_mape"]
            - lane_scores["visible_label_root_length_mape"]
        )
        case_rows.append(case_row)
        per_owner_rows.extend(_per_owner_rows(prepared, gated_owner))

        _write_index_mask(
            output_dir / "challenge" / "ownership" / f"{sample.sample_id}__spatial_lane.png",
            lane_owner,
        )
        _write_index_mask(
            output_dir / "challenge" / "ownership" / f"{sample.sample_id}__gated_learned.png",
            gated_owner,
        )
        _write_index_mask(
            output_dir / "challenge" / "ownership" / f"{sample.sample_id}__ground_truth.png",
            prepared["target"].owner_mask,
        )
        comparison_path = (
            output_dir / "challenge" / "comparisons" / f"{sample.sample_id}__comparison.jpg"
        )
        _write_rgb(
            comparison_path,
            _comparison(
                prepared,
                gated_owner,
                lane_scores,
                gated_scores,
                threshold=threshold,
            ),
            quality=92,
        )
        comparison_paths.append(comparison_path)

    _write_csv(output_dir / "five_plate_case_metrics.csv", case_rows)
    _write_csv(output_dir / "per_seedling_measurements.csv", per_owner_rows)
    contact_sheet_path = output_dir / "five_plate_owner_comparison_contact_sheet.jpg"
    _write_contact_sheet(comparison_paths, contact_sheet_path)
    lane_mean = _mean_metrics(case_rows, "lane")
    gated_mean = _mean_metrics(case_rows, "gated")
    selected_threshold_row = next(
        row for row in threshold_rows if float(row["threshold"]) == float(threshold)
    )
    report = {
        "title": "Yang Song five-plate seedling-ownership diagnostic retrial",
        "provenance": {
            "person": "Yang Song",
            "project": "DroughtFighters",
            "group": "Plant Microbe Interaction",
            "institution": "Utrecht University",
            "legacy_identifier": "yang",
        },
        "corpus": str(corpus),
        "output_dir": str(output_dir),
        "selection": {
            "seed": int(seed),
            "source_partition": "shoot-model plate-group holdout",
            "challenge_samples": [sample.sample_id for sample in challenge],
            "challenge_groups": [sample.plate_group for sample in challenge],
            "calibration_samples": [sample.sample_id for sample in calibration],
            "calibration_groups": sorted({sample.plate_group for sample in calibration}),
            "owner_training_excludes_all_evaluation_groups": True,
            "challenge_not_used_for_threshold_selection": True,
        },
        "models": {
            "root": str(_hybrid_config().root_model_path),
            "root_sha256": _sha256_file(Path(_hybrid_config().root_model_path)),
            "shoot": str(_hybrid_config().shoot_model_path),
            "shoot_sha256": _sha256_file(Path(_hybrid_config().shoot_model_path)),
            "shoot_training_profile": str(shoot_profile_path),
            "shoot_profile_holdout": dict((shoot_profile.get("split") or {}).get("holdout") or {}),
            "owner_ranker": str(output_dir / "owner_ranker.json"),
        },
        "owner_training": ranker_training,
        "confidence_gate": {
            "score": "raw top-two shared-ranker margin; diagnostic, not calibrated probability",
            "selected_threshold": threshold,
            "selection_objective": (
                "maximize calibration mean visible-label full owner Dice; require mean skeleton-"
                "length MAPE and WAPE <= lane + 0.01 and worst plate Dice delta >= -0.02"
            ),
            "selected_calibration_result": selected_threshold_row,
            "threshold_sweep_csv": str(output_dir / "calibration_threshold_sweep.csv"),
        },
        "challenge_mean": {
            "spatial_lane": lane_mean,
            "confidence_gated_learned": gated_mean,
            "delta": {
                key: float(gated_mean[key] - lane_mean[key])
                for key in lane_mean
            },
        },
        "scientific_scope": {
            "ownership_claim": False,
            "visible_label_full_metrics": (
                "Predictions are scored against visible hand labels over the full image. "
                "Unlabelled bacterial gaps can contain hidden roots and are not biological negatives."
            ),
            "conditional_metrics": "Ownership only on pixels with one unambiguous hand-labelled owner.",
            "conflicting_instance_pixels": "excluded",
            "bacterial_gaps": "unknown/background; no hidden-root pixels are invented",
            "primary_lateral": "semantic topology decomposition remains separate from owner assignment",
        },
        "limitations": [
            "Five diagnostic plates are too few for production promotion.",
            "The raw ranker margin is not a calibrated probability.",
            "Root crossings with merged pixels have no uniquely recoverable provenance from one frame.",
            "The gate changes ownership allocation only; it does not repair missed root segmentation.",
        ],
        "case_metrics_csv": str(output_dir / "five_plate_case_metrics.csv"),
        "per_seedling_measurements_csv": str(output_dir / "per_seedling_measurements.csv"),
        "comparison_images": [str(path) for path in comparison_paths],
        "comparison_contact_sheet": str(contact_sheet_path),
    }
    report_path = output_dir / "retrial_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    relative_dice_gain = (
        (gated_mean["visible_label_full_macro_dice"] - lane_mean["visible_label_full_macro_dice"])
        / max(lane_mean["visible_label_full_macro_dice"], 1e-9)
    )
    relative_mape_reduction = (
        (lane_mean["visible_label_root_length_mape"] - gated_mean["visible_label_root_length_mape"])
        / max(lane_mean["visible_label_root_length_mape"], 1e-9)
    )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(
            [
                "# Yang Song Five-Plate Seedling-Ownership Retrial",
                "",
                f"- Challenge samples: {', '.join(sample.sample_id for sample in challenge)}",
                f"- Selected owner margin threshold: {threshold:.2f}",
                (
                    "- Visible-label full owner Dice: "
                    f"{lane_mean['visible_label_full_macro_dice']:.3f} -> "
                    f"{gated_mean['visible_label_full_macro_dice']:.3f} "
                    f"({100.0 * relative_dice_gain:+.1f}% relative)"
                ),
                (
                    "- Per-seedling skeleton-length MAPE: "
                    f"{lane_mean['visible_label_root_length_mape']:.3f} -> "
                    f"{gated_mean['visible_label_root_length_mape']:.3f} "
                    f"({100.0 * relative_mape_reduction:.1f}% relative reduction)"
                ),
                "",
                "This is a five-plate diagnostic, not a production qualification. "
                "Ownership claims remain disabled. Bacterial gaps remain unknown.",
                "",
                f"Contact sheet: {contact_sheet_path}",
                f"Case metrics: {output_dir / 'five_plate_case_metrics.csv'}",
                f"Per-seedling measurements: {output_dir / 'per_seedling_measurements.csv'}",
                f"Full report: {report_path}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Leakage-aware five-plate Yang Song ownership diagnostic retrial."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shoot-profile", type=Path, default=DEFAULT_SHOOT_PROFILE)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-pixels-per-owner", type=int, default=350)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()
    run_retrial(
        args.corpus.expanduser(),
        args.output_dir.expanduser(),
        args.shoot_profile.expanduser(),
        count=max(1, int(args.count)),
        seed=int(args.seed),
        max_pixels_per_owner=max(25, int(args.max_pixels_per_owner)),
        chunk_size=max(1_000, int(args.chunk_size)),
        reuse_predictions=bool(args.reuse_predictions),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
