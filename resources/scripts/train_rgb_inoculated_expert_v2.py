from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.five_seedling_ownership import (  # noqa: E402
    FiveSeedlingSample,
    discover_five_seedling_corpus,
    grouped_split,
    load_semantic_masks,
)


DEFAULT_CORPUS = REPO_ROOT / "data" / "yang_ground_truth"
BUILTIN_DIR = REPO_ROOT / "resources" / "builtin_models" / "rgb_inoculated"
DEFAULT_BASE_MODEL = BUILTIN_DIR / "rgb_inoculated_multiclass.keras"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "rgb_inoculated_training_v2"
INSTALL_MODEL_NAME = "rgb_inoculated_multiclass_v2.keras"
INSTALL_PROFILE_NAME = "rgb_inoculated_multiclass_v2.profile.json"

CLASS_NAMES = ("background", "seed", "shoot", "primary_root", "lateral_root")
NUM_CLASSES = len(CLASS_NAMES)
FOREGROUND_IDS = (1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class PatchSpec:
    sample_index: int
    y: int
    x: int
    centered_class: int


def _semantic_target(sample: FiveSeedlingSample) -> np.ndarray:
    masks = load_semantic_masks(sample)
    shape = next(iter(masks.values())).shape[:2]
    target = np.zeros(shape, dtype=np.uint8)
    target[masks["root"] > 0] = 3
    target[masks["lateral"] > 0] = 4
    target[masks["shoot"] > 0] = 2
    target[masks["seed"] > 0] = 1
    return target


def _assert_group_exclusive(
    train: Sequence[FiveSeedlingSample],
    validation: Sequence[FiveSeedlingSample],
    holdout: Sequence[FiveSeedlingSample],
) -> None:
    group_sets = [
        {sample.plate_group for sample in split}
        for split in (train, validation, holdout)
    ]
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise RuntimeError("Plate-group leakage detected between train, validation, and holdout.")
    if not all(group_sets):
        raise RuntimeError("Each split must contain at least one plate group.")


def _three_way_grouped_split(
    samples: Sequence[FiveSeedlingSample],
    *,
    validation_fraction: float,
    holdout_fraction: float,
) -> tuple[list[FiveSeedlingSample], list[FiveSeedlingSample], list[FiveSeedlingSample]]:
    train_validation, holdout = grouped_split(
        samples,
        holdout_fraction=float(holdout_fraction),
        salt="npec-rgb-inoculated-v2-holdout",
    )
    remaining_fraction = max(1e-6, 1.0 - float(holdout_fraction))
    relative_validation = min(0.8, float(validation_fraction) / remaining_fraction)
    train, validation = grouped_split(
        train_validation,
        holdout_fraction=relative_validation,
        salt="npec-rgb-inoculated-v2-validation",
    )
    _assert_group_exclusive(train, validation, holdout)
    return train, validation, holdout


def _split_summary(samples: Sequence[FiveSeedlingSample]) -> dict[str, object]:
    return {
        "sample_count": len(samples),
        "plate_group_count": len({sample.plate_group for sample in samples}),
        "sample_ids": [sample.sample_id for sample in samples],
        "plate_groups": sorted({sample.plate_group for sample in samples}),
    }


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _patch_specs(
    samples: Sequence[FiveSeedlingSample],
    *,
    patch_size: int,
    patches_per_image: int,
    background_patches_per_image: int,
    seed: int,
) -> list[PatchSpec]:
    specs: list[PatchSpec] = []
    half = int(patch_size) // 2
    for sample_index, sample in enumerate(samples):
        target = _semantic_target(sample)
        height, width = target.shape
        if height < patch_size or width < patch_size:
            raise ValueError(f"Image {sample.sample_id} is smaller than the {patch_size}px patch size.")
        coordinates = {
            class_id: np.argwhere(target == np.uint8(class_id))
            for class_id in FOREGROUND_IDS
        }
        available = [class_id for class_id, points in coordinates.items() if points.size > 0]
        if not available:
            raise ValueError(f"No labeled foreground found for sample {sample.sample_id}.")
        rng = np.random.default_rng(_stable_seed(sample.sample_id, seed))
        seen: set[tuple[int, int, int]] = set()
        attempts = max(64, int(patches_per_image) * 12)
        for attempt in range(attempts):
            if len(seen) >= int(patches_per_image):
                break
            class_id = available[attempt % len(available)]
            points = coordinates[class_id]
            center_y, center_x = points[int(rng.integers(0, len(points)))]
            jitter = max(1, int(patch_size) // 8)
            center_y += int(rng.integers(-jitter, jitter + 1))
            center_x += int(rng.integers(-jitter, jitter + 1))
            y = int(np.clip(int(center_y) - half, 0, height - patch_size))
            x = int(np.clip(int(center_x) - half, 0, width - patch_size))
            key = (y, x, class_id)
            if key in seen:
                continue
            seen.add(key)
            specs.append(PatchSpec(sample_index, y, x, class_id))
        if not seen:
            raise RuntimeError(f"Unable to sample foreground patches from {sample.sample_id}.")
        background_count = max(0, int(background_patches_per_image))
        if background_count <= 0:
            continue
        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Unable to read image while sampling background: {sample.image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        local_contrast = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 3.0)).astype(np.float32)
        chroma = (
            np.max(image_rgb, axis=2).astype(np.float32)
            - np.min(image_rgb, axis=2).astype(np.float32)
        )
        activity = local_contrast + (0.5 * chroma)
        activity_integral = cv2.integral(activity)
        foreground_integral = cv2.integral((target > 0).astype(np.uint8))

        def region_sum(integral: np.ndarray, y: int, x: int) -> float:
            y1 = y + int(patch_size)
            x1 = x + int(patch_size)
            return float(integral[y1, x1] - integral[y, x1] - integral[y1, x] + integral[y, x])

        stride = max(32, int(patch_size) // 2)
        y_positions = list(range(0, height - patch_size + 1, stride))
        x_positions = list(range(0, width - patch_size + 1, stride))
        if y_positions[-1] != height - patch_size:
            y_positions.append(height - patch_size)
        if x_positions[-1] != width - patch_size:
            x_positions.append(width - patch_size)
        candidate_background: list[tuple[float, int, int]] = []
        maximum_foreground = max(4, int(round((patch_size * patch_size) * 0.002)))
        for y in y_positions:
            for x in x_positions:
                foreground_pixels = int(round(region_sum(foreground_integral, y, x)))
                if foreground_pixels > maximum_foreground:
                    continue
                score = region_sum(activity_integral, y, x) / float(patch_size * patch_size)
                candidate_background.append((float(score), int(y), int(x)))
        if not candidate_background:
            raise RuntimeError(f"Unable to sample full-plate background from {sample.sample_id}.")
        candidate_background.sort(reverse=True)
        hard_count = min(len(candidate_background), max(1, background_count // 2))
        selected_background = list(candidate_background[:hard_count])
        remaining = candidate_background[hard_count:]
        rng.shuffle(remaining)
        selected_background.extend(remaining[: max(0, background_count - len(selected_background))])
        if len(selected_background) < background_count:
            selected_keys = {(y, x) for _score, y, x in selected_background}
            selected_background.extend(
                candidate
                for candidate in candidate_background
                if (candidate[1], candidate[2]) not in selected_keys
            )
        for _score, y, x in selected_background[:background_count]:
            specs.append(PatchSpec(sample_index, int(y), int(x), 0))
    return specs


def _class_pixel_counts(
    samples: Sequence[FiveSeedlingSample],
    specs: Sequence[PatchSpec] | None = None,
    *,
    patch_size: int = 256,
) -> np.ndarray:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    if specs is None:
        for sample in samples:
            target = _semantic_target(sample)
            counts += np.bincount(target.reshape(-1), minlength=NUM_CLASSES)[:NUM_CLASSES]
        return counts

    by_sample: dict[int, list[PatchSpec]] = {}
    for spec in specs:
        by_sample.setdefault(spec.sample_index, []).append(spec)
    for sample_index, sample_specs in by_sample.items():
        target = _semantic_target(samples[sample_index])
        for spec in sample_specs:
            patch = target[spec.y : spec.y + patch_size, spec.x : spec.x + patch_size]
            counts += np.bincount(patch.reshape(-1), minlength=NUM_CLASSES)[:NUM_CLASSES]
    return counts


def _class_weights(counts: np.ndarray, *, background_weight: float = 1.0) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    positive_fg = counts[1:][counts[1:] > 0]
    if positive_fg.size < 3:
        raise ValueError("Training patches do not cover enough foreground classes.")
    reference = float(np.median(positive_fg))
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    weights[0] = float(max(0.01, background_weight))
    for class_id in FOREGROUND_IDS:
        if counts[class_id] <= 0:
            raise ValueError(f"No training pixels found for class {class_id} ({CLASS_NAMES[class_id]}).")
        weights[class_id] = float(np.clip(math.sqrt(reference / counts[class_id]), 0.5, 8.0))
    return weights


class _PatchCache:
    def __init__(self, samples: Sequence[FiveSeedlingSample], max_items: int = 2) -> None:
        self.samples = list(samples)
        self.max_items = max(1, int(max_items))
        self._items: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, sample_index: int) -> tuple[np.ndarray, np.ndarray]:
        if sample_index in self._items:
            self._items.move_to_end(sample_index)
            return self._items[sample_index]
        sample = self.samples[sample_index]
        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Unable to read image: {sample.image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        target = _semantic_target(sample)
        if image.shape[:2] != target.shape:
            raise ValueError(f"Image/mask shape mismatch for {sample.sample_id}.")
        self._items[sample_index] = (image, target)
        self._items.move_to_end(sample_index)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return image, target


def _confusion_update(confusion: np.ndarray, truth: np.ndarray, prediction: np.ndarray) -> None:
    valid = (truth >= 0) & (truth < NUM_CLASSES)
    encoded = truth[valid].astype(np.int64) * NUM_CLASSES + prediction[valid].astype(np.int64)
    confusion += np.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES).reshape(
        NUM_CLASSES, NUM_CLASSES
    )


def _metrics_from_confusion(confusion: np.ndarray) -> dict[str, object]:
    confusion = np.asarray(confusion, dtype=np.float64)
    per_class: dict[str, dict[str, float | int | None]] = {}
    ious: list[float] = []
    for class_id, name in enumerate(CLASS_NAMES):
        true_positive = confusion[class_id, class_id]
        false_negative = confusion[class_id, :].sum() - true_positive
        false_positive = confusion[:, class_id].sum() - true_positive
        union = true_positive + false_negative + false_positive
        denominator = 2.0 * true_positive + false_negative + false_positive
        iou = None if union <= 0 else float(true_positive / union)
        dice = None if denominator <= 0 else float((2.0 * true_positive) / denominator)
        per_class[name] = {
            "iou": iou,
            "dice": dice,
            "truth_pixels": int(confusion[class_id, :].sum()),
            "predicted_pixels": int(confusion[:, class_id].sum()),
        }
        if class_id in FOREGROUND_IDS and iou is not None:
            ious.append(iou)

    root_ids = (3, 4)
    root_tp = confusion[np.ix_(root_ids, root_ids)].sum()
    root_truth = confusion[list(root_ids), :].sum()
    root_predicted = confusion[:, list(root_ids)].sum()
    root_union = root_truth + root_predicted - root_tp
    foreground_truth = confusion[1:, :].sum()
    foreground_predicted = confusion[:, 1:].sum()
    ratio = None if foreground_truth <= 0 else float(foreground_predicted / foreground_truth)
    return {
        "foreground_miou": None if not ious else float(np.mean(ious)),
        "root_union_iou": None if root_union <= 0 else float(root_tp / root_union),
        "shoot_iou": per_class["shoot"]["iou"],
        "predicted_to_truth_foreground_ratio": ratio,
        "per_class": per_class,
    }


def _load_tensorflow_cpu():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except (RuntimeError, ValueError):
        pass
    tf.config.threading.set_inter_op_parallelism_threads(max(1, min(4, os.cpu_count() or 1)))
    tf.config.threading.set_intra_op_parallelism_threads(max(1, min(8, os.cpu_count() or 1)))
    return tf


def _build_five_class_model(tf, base_model_path: Path):
    base = tf.keras.models.load_model(str(base_model_path), compile=False)
    if len(base.inputs) != 1 or len(base.outputs) != 1:
        raise ValueError("The base model must have one input and one output.")
    if int(base.output_shape[-1]) != 6:
        raise ValueError(f"Expected a six-channel base head, got {base.output_shape}.")
    old_head = base.layers[-1]
    if not isinstance(old_head, tf.keras.layers.Conv2D) or tuple(old_head.kernel_size) != (1, 1):
        raise ValueError("The base model must end in a 1x1 Conv2D semantic head.")

    new_head = tf.keras.layers.Conv2D(
        NUM_CLASSES,
        (1, 1),
        padding="same",
        activation="softmax",
        name="semantic_5class",
    )
    output = new_head(old_head.input)
    model = tf.keras.Model(base.input, output, name="rgb_inoculated_multiclass_v2")

    old_kernel, old_bias = old_head.get_weights()
    new_kernel = np.zeros((*old_kernel.shape[:-1], NUM_CLASSES), dtype=old_kernel.dtype)
    new_bias = np.zeros((NUM_CLASSES,), dtype=old_bias.dtype)
    new_kernel[..., 0] = 0.5 * (old_kernel[..., 0] + old_kernel[..., 5])
    new_bias[0] = np.logaddexp(old_bias[0], old_bias[5])
    new_kernel[..., 1:] = old_kernel[..., 1:5]
    new_bias[1:] = old_bias[1:5]
    new_head.set_weights([new_kernel, new_bias])
    return model, new_head


def _set_trainable_layers(model, head, *, trunk_layers: int) -> None:
    for layer in model.layers:
        layer.trainable = False
    head.trainable = True
    trunk = [layer for layer in model.layers if layer is not head]
    count = max(0, int(trunk_layers))
    if count > 0:
        for layer in trunk[-count:]:
            layer.trainable = True


def _profile_payload(
    *,
    model_path: Path,
    base_model_path: Path,
    corpus: Path,
    split: dict[str, object],
    benchmark: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "name": "rgb_inoculated_multiclass_v2",
        "mode": "multiclass",
        "input_mode": "rgb",
        "flip_horizontal": False,
        "return_original_coords": True,
        "class_names": list(CLASS_NAMES),
        "class_ids": {
            "background": 0,
            "seed": 1,
            "shoot": 2,
            "primary_root": 3,
            "lateral_root": 4,
        },
        "root_label_ids": [3, 4],
        "primary_root_label_ids": [3],
        "lateral_label_ids": [4],
        "shoot_label_ids": [2],
        "seed_label_ids": [1],
        "include_seed_in_shoot": False,
        "model_path": str(model_path),
        "initialized_from": str(base_model_path),
        "training_corpus": str(corpus),
        "split": split,
        "holdout_benchmark": benchmark,
        "gap_policy": {
            "handwritten_bacterial_gaps": "background_or_unknown",
            "trained_as_visible_root": False,
            "note": (
                "Occluded root recovery must be emitted separately with provenance and confidence; "
                "this visible-semantic model does not fill bacterial gaps."
            ),
        },
    }


def _qualification(
    metrics: dict[str, object],
    *,
    min_foreground_miou: float,
    min_root_iou: float,
    min_shoot_iou: float,
    min_foreground_ratio: float,
    max_foreground_ratio: float,
) -> dict[str, object]:
    checks = {
        "foreground_miou": float(metrics.get("foreground_miou") or 0.0) >= min_foreground_miou,
        "root_union_iou": float(metrics.get("root_union_iou") or 0.0) >= min_root_iou,
        "shoot_iou": float(metrics.get("shoot_iou") or 0.0) >= min_shoot_iou,
        "foreground_not_collapsed": (
            metrics.get("predicted_to_truth_foreground_ratio") is not None
            and min_foreground_ratio
            <= float(metrics["predicted_to_truth_foreground_ratio"])
            <= max_foreground_ratio
        ),
    }
    per_class = metrics.get("per_class", {})
    checks["all_foreground_classes_present"] = all(
        int(per_class.get(name, {}).get("predicted_pixels") or 0) > 0
        for name in ("seed", "shoot", "primary_root", "lateral_root")
    )
    return {
        "qualified": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "min_foreground_miou": min_foreground_miou,
            "min_root_union_iou": min_root_iou,
            "min_shoot_iou": min_shoot_iou,
            "min_predicted_to_truth_foreground_ratio": min_foreground_ratio,
            "max_predicted_to_truth_foreground_ratio": max_foreground_ratio,
        },
    }


def _install_qualified(candidate_model: Path, candidate_profile: Path) -> tuple[Path, Path]:
    BUILTIN_DIR.mkdir(parents=True, exist_ok=True)
    destination_model = BUILTIN_DIR / INSTALL_MODEL_NAME
    destination_profile = BUILTIN_DIR / INSTALL_PROFILE_NAME
    temporary_model = destination_model.with_suffix(destination_model.suffix + ".tmp")
    temporary_profile = destination_profile.with_suffix(destination_profile.suffix + ".tmp")
    shutil.copy2(candidate_model, temporary_model)
    shutil.copy2(candidate_profile, temporary_profile)
    os.replace(temporary_model, destination_model)
    os.replace(temporary_profile, destination_profile)
    return destination_model, destination_profile


def _dry_run(
    samples: Sequence[FiveSeedlingSample],
    train: Sequence[FiveSeedlingSample],
    validation: Sequence[FiveSeedlingSample],
    holdout: Sequence[FiveSeedlingSample],
    *,
    patch_size: int,
    train_patches_per_image: int,
    train_background_patches_per_image: int,
    validation_patches_per_image: int,
    validation_background_patches_per_image: int,
    background_weight: float,
    seed: int,
) -> None:
    train_specs = _patch_specs(
        train,
        patch_size=patch_size,
        patches_per_image=train_patches_per_image,
        background_patches_per_image=train_background_patches_per_image,
        seed=seed,
    )
    validation_specs = _patch_specs(
        validation,
        patch_size=patch_size,
        patches_per_image=validation_patches_per_image,
        background_patches_per_image=validation_background_patches_per_image,
        seed=seed + 1,
    )
    full_counts = _class_pixel_counts(samples)
    train_patch_counts = _class_pixel_counts(train, train_specs, patch_size=patch_size)
    payload = {
        "dry_run_data": True,
        "corpus": str(Path(samples[0].image_path).parents[1]),
        "classes": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "full_corpus_pixel_counts": full_counts.tolist(),
        "train_patch_pixel_counts": train_patch_counts.tolist(),
        "per_pixel_class_weights": _class_weights(
            train_patch_counts,
            background_weight=background_weight,
        ).tolist(),
        "train_patch_count": len(train_specs),
        "validation_patch_count": len(validation_specs),
        "train": _split_summary(train),
        "validation": _split_summary(validation),
        "holdout": _split_summary(holdout),
        "gap_policy": (
            "Handwritten bacterial gaps remain class 0 background/unknown. No gap pixels are filled "
            "or trained as visible root."
        ),
    }
    print(json.dumps(payload, indent=2), flush=True)


def _tiled_predict(model, image_rgb: np.ndarray, *, patch_size: int, halo: int, batch_size: int) -> np.ndarray:
    if halo < 0 or 2 * halo >= patch_size:
        raise ValueError("Tile halo must be non-negative and smaller than half the patch size.")
    output_size = patch_size - 2 * halo
    height, width = image_rgb.shape[:2]
    rows = int(math.ceil(height / output_size))
    columns = int(math.ceil(width / output_size))
    extra_bottom = rows * output_size - height
    extra_right = columns * output_size - width
    border_mode = cv2.BORDER_REFLECT_101 if min(height, width) > 1 else cv2.BORDER_REPLICATE
    padded = cv2.copyMakeBorder(
        image_rgb,
        halo,
        halo + extra_bottom,
        halo,
        halo + extra_right,
        border_mode,
    )
    prediction = np.zeros((height, width), dtype=np.uint8)
    pending: list[np.ndarray] = []
    locations: list[tuple[int, int, int, int]] = []

    def flush() -> None:
        if not pending:
            return
        batch = np.asarray(pending, dtype=np.float32) / 255.0
        probabilities = model.predict_on_batch(batch)
        for probability, (y, x, block_height, block_width) in zip(probabilities, locations):
            labels = np.argmax(probability, axis=-1).astype(np.uint8)
            prediction[y : y + block_height, x : x + block_width] = labels[
                halo : halo + block_height,
                halo : halo + block_width,
            ]
        pending.clear()
        locations.clear()

    for row in range(rows):
        y = row * output_size
        for column in range(columns):
            x = column * output_size
            pending.append(padded[y : y + patch_size, x : x + patch_size])
            locations.append((y, x, min(output_size, height - y), min(output_size, width - x)))
            if len(pending) >= max(1, int(batch_size)):
                flush()
    flush()
    return prediction


def _full_image_benchmark(
    model,
    holdout: Sequence[FiveSeedlingSample],
    *,
    patch_size: int,
    tile_halo: int,
    batch_size: int,
) -> dict[str, object]:
    aggregate = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cases: list[dict[str, object]] = []
    for index, sample in enumerate(holdout, start=1):
        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Unable to read holdout image: {sample.image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        truth = _semantic_target(sample)
        predicted = _tiled_predict(
            model,
            image_rgb,
            patch_size=patch_size,
            halo=tile_halo,
            batch_size=batch_size,
        )
        case_confusion = np.zeros_like(aggregate)
        _confusion_update(case_confusion, truth, predicted)
        aggregate += case_confusion
        cases.append({"sample_id": sample.sample_id, **_metrics_from_confusion(case_confusion)})
        print(f"[holdout] {index}/{len(holdout)} {sample.sample_id}", flush=True)
    return {
        "scope": "full_image_tiled_holdout",
        "image_count": len(holdout),
        "aggregate": _metrics_from_confusion(aggregate),
        "cases": cases,
    }


def _train(args, train, validation, holdout) -> dict[str, object]:
    tf = _load_tensorflow_cpu()
    tf.keras.utils.set_random_seed(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    base_model_path = Path(args.base_model).expanduser().resolve()
    if not base_model_path.is_file():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    model, head = _build_five_class_model(tf, base_model_path)
    expected_patch = tuple(int(value) for value in model.input_shape[1:3])
    if expected_patch != (int(args.patch_size), int(args.patch_size)):
        raise ValueError(f"Base model expects {expected_patch}; requested patch size is {args.patch_size}.")

    train_specs = _patch_specs(
        train,
        patch_size=args.patch_size,
        patches_per_image=args.train_patches_per_image,
        background_patches_per_image=args.train_background_patches_per_image,
        seed=args.seed,
    )
    validation_specs = _patch_specs(
        validation,
        patch_size=args.patch_size,
        patches_per_image=args.validation_patches_per_image,
        background_patches_per_image=args.validation_background_patches_per_image,
        seed=args.seed + 1,
    )
    patch_counts = _class_pixel_counts(train, train_specs, patch_size=args.patch_size)
    class_weights = _class_weights(patch_counts, background_weight=args.background_weight)

    class PatchSequence(tf.keras.utils.Sequence):
        def __init__(self, samples, specs, *, augment, shuffle, seed):
            super().__init__()
            self.samples = list(samples)
            self.specs = list(specs)
            self.augment = bool(augment)
            self.shuffle = bool(shuffle)
            self.rng = np.random.default_rng(seed)
            self.indices = np.arange(len(self.specs), dtype=np.int64)
            self.cache = _PatchCache(self.samples)
            self.on_epoch_end()

        def __len__(self):
            return int(math.ceil(len(self.specs) / max(1, args.batch_size)))

        def __getitem__(self, batch_index):
            begin = batch_index * args.batch_size
            selected = self.indices[begin : begin + args.batch_size]
            images: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            for selected_index in selected:
                spec = self.specs[int(selected_index)]
                image, target = self.cache.get(spec.sample_index)
                image_patch = image[
                    spec.y : spec.y + args.patch_size,
                    spec.x : spec.x + args.patch_size,
                ].copy()
                target_patch = target[
                    spec.y : spec.y + args.patch_size,
                    spec.x : spec.x + args.patch_size,
                ].copy()
                if self.augment and self.rng.random() < 0.5:
                    image_patch = np.ascontiguousarray(image_patch[:, ::-1])
                    target_patch = np.ascontiguousarray(target_patch[:, ::-1])
                if self.augment:
                    scale = float(self.rng.uniform(0.90, 1.10))
                    offset = float(self.rng.uniform(-8.0, 8.0))
                    image_patch = np.clip(image_patch.astype(np.float32) * scale + offset, 0, 255)
                images.append(image_patch.astype(np.float32) / 255.0)
                targets.append(target_patch.astype(np.uint8))
            x = np.asarray(images, dtype=np.float32)
            y = np.asarray(targets, dtype=np.uint8)
            sample_weight = class_weights[y]
            return x, y, sample_weight.astype(np.float32)

        def on_epoch_end(self):
            if self.shuffle:
                by_sample: dict[int, list[int]] = {}
                for spec_index, spec in enumerate(self.specs):
                    by_sample.setdefault(int(spec.sample_index), []).append(int(spec_index))
                sample_order = np.asarray(list(by_sample), dtype=np.int64)
                self.rng.shuffle(sample_order)
                ordered: list[int] = []
                for sample_index in sample_order:
                    sample_indices = np.asarray(by_sample[int(sample_index)], dtype=np.int64)
                    self.rng.shuffle(sample_indices)
                    ordered.extend(int(value) for value in sample_indices)
                self.indices = np.asarray(ordered, dtype=np.int64)

    train_sequence = PatchSequence(train, train_specs, augment=True, shuffle=True, seed=args.seed)
    validation_sequence = PatchSequence(
        validation,
        validation_specs,
        augment=False,
        shuffle=False,
        seed=args.seed + 1,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_model = output_dir / INSTALL_MODEL_NAME
    candidate_profile = output_dir / INSTALL_PROFILE_NAME
    benchmark_path = output_dir / "rgb_inoculated_multiclass_v2.benchmark.json"
    training_path = output_dir / "rgb_inoculated_multiclass_v2.training.json"

    class ForegroundMiouCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = -1.0
            self.best_epoch = 0
            self.epoch_offset = 0

        def on_epoch_end(self, epoch, logs=None):
            confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            for batch_index in range(len(validation_sequence)):
                x, y, _weights = validation_sequence[batch_index]
                predicted = np.argmax(self.model.predict_on_batch(x), axis=-1).astype(np.uint8)
                _confusion_update(confusion, y, predicted)
            metrics = _metrics_from_confusion(confusion)
            score = float(metrics["foreground_miou"] or 0.0)
            if logs is not None:
                logs["val_foreground_miou"] = score
                logs["val_root_union_iou"] = float(metrics["root_union_iou"] or 0.0)
                logs["val_shoot_iou"] = float(metrics["shoot_iou"] or 0.0)
            absolute_epoch = self.epoch_offset + int(epoch) + 1
            print(
                "[epoch] "
                + json.dumps(
                    {
                        "epoch": absolute_epoch,
                        "val_foreground_miou": score,
                        "val_root_union_iou": metrics["root_union_iou"],
                        "val_shoot_iou": metrics["shoot_iou"],
                    }
                ),
                flush=True,
            )
            if score > self.best:
                self.best = score
                self.best_epoch = absolute_epoch
                self.model.save(str(candidate_model))

    monitor = ForegroundMiouCallback()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    histories: list[dict[str, list[float]]] = []
    if args.freeze_trunk_epochs > 0:
        _set_trainable_layers(model, head, trunk_layers=0)
        model.compile(optimizer=tf.keras.optimizers.Adam(args.head_learning_rate), loss=loss)
        history = model.fit(
            train_sequence,
            validation_data=validation_sequence,
            epochs=args.freeze_trunk_epochs,
            callbacks=[monitor],
            verbose=0,
        )
        histories.append(history.history)
        monitor.epoch_offset = int(args.freeze_trunk_epochs)

    _set_trainable_layers(model, head, trunk_layers=args.unfreeze_last_layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), loss=loss)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_foreground_miou",
        mode="max",
        patience=args.patience,
        restore_best_weights=False,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_foreground_miou",
        mode="max",
        patience=max(1, args.patience // 2),
        factor=0.5,
        min_lr=1e-6,
    )
    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=args.epochs,
        callbacks=[monitor, early_stopping, reduce_lr],
        verbose=0,
    )
    histories.append(history.history)
    if not candidate_model.is_file():
        raise RuntimeError("Training did not produce a validation-selected candidate model.")

    candidate = tf.keras.models.load_model(str(candidate_model), compile=False)
    benchmark = _full_image_benchmark(
        candidate,
        holdout,
        patch_size=args.patch_size,
        tile_halo=args.tile_halo,
        batch_size=args.benchmark_batch_size,
    )
    qualification = _qualification(
        benchmark["aggregate"],
        min_foreground_miou=args.min_holdout_foreground_miou,
        min_root_iou=args.min_holdout_root_iou,
        min_shoot_iou=args.min_holdout_shoot_iou,
        min_foreground_ratio=args.min_foreground_ratio,
        max_foreground_ratio=args.max_foreground_ratio,
    )
    benchmark["qualification"] = qualification
    benchmark_path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")

    split = {
        "strategy": "plate_group_exclusive",
        "train": _split_summary(train),
        "validation": _split_summary(validation),
        "holdout": _split_summary(holdout),
    }
    profile = _profile_payload(
        model_path=candidate_model,
        base_model_path=base_model_path,
        corpus=Path(args.corpus).expanduser().resolve(),
        split=split,
        benchmark=benchmark["aggregate"],
    )
    profile["qualification"] = qualification
    candidate_profile.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    installed: dict[str, str] | None = None
    if args.install_if_qualified:
        if qualification["qualified"]:
            installed_model, installed_profile = _install_qualified(candidate_model, candidate_profile)
            installed = {"model": str(installed_model), "profile": str(installed_profile)}
        else:
            print("Installation refused: holdout qualification checks did not pass.", flush=True)

    training_summary = {
        "candidate_model": str(candidate_model),
        "candidate_profile": str(candidate_profile),
        "benchmark": str(benchmark_path),
        "initialized_from": str(base_model_path),
        "best_validation_foreground_miou": monitor.best,
        "best_epoch": monitor.best_epoch,
        "train_patch_count": len(train_specs),
        "validation_patch_count": len(validation_specs),
        "train_patch_pixel_counts": patch_counts.tolist(),
        "per_pixel_class_weights": class_weights.tolist(),
        "histories": histories,
        "qualification": qualification,
        "installed": installed,
        "gap_policy": "No bacterial gap filling; gaps remain background/unknown.",
    }
    training_path.write_text(json.dumps(training_summary, indent=2), encoding="utf-8")
    print(json.dumps(training_summary, indent=2), flush=True)
    return training_summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and gate a five-class RGB inoculated Arabidopsis segmentation expert."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run-data", action="store_true")
    parser.add_argument("--install-if-qualified", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--train-patches-per-image", type=int, default=16)
    parser.add_argument("--train-background-patches-per-image", type=int, default=16)
    parser.add_argument("--validation-patches-per-image", type=int, default=8)
    parser.add_argument("--validation-background-patches-per-image", type=int, default=8)
    parser.add_argument("--background-weight", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--freeze-trunk-epochs", type=int, default=1)
    parser.add_argument("--unfreeze-last-layers", type=int, default=16)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--tile-halo", type=int, default=32)
    parser.add_argument("--benchmark-batch-size", type=int, default=8)
    parser.add_argument("--min-holdout-foreground-miou", type=float, default=0.25)
    parser.add_argument("--min-holdout-root-iou", type=float, default=0.40)
    parser.add_argument("--min-holdout-shoot-iou", type=float, default=0.45)
    parser.add_argument("--min-foreground-ratio", type=float, default=0.25)
    parser.add_argument("--max-foreground-ratio", type=float, default=4.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not 0.0 < args.validation_fraction < 0.5 or not 0.0 < args.holdout_fraction < 0.5:
        raise ValueError("Validation and holdout fractions must each be between 0 and 0.5.")
    if args.validation_fraction + args.holdout_fraction >= 0.8:
        raise ValueError("Validation plus holdout must leave a substantial training split.")
    corpus = Path(args.corpus).expanduser().resolve()
    discovered_samples = discover_five_seedling_corpus(corpus)
    samples = [sample for sample in discovered_samples if re.fullmatch(r"\d+-\d+", sample.sample_id)]
    excluded = [sample.sample_id for sample in discovered_samples if sample not in samples]
    if excluded:
        print(f"Excluded ambiguous sample IDs: {', '.join(excluded)}", flush=True)
    train, validation, holdout = _three_way_grouped_split(
        samples,
        validation_fraction=args.validation_fraction,
        holdout_fraction=args.holdout_fraction,
    )
    if args.dry_run_data:
        _dry_run(
            samples,
            train,
            validation,
            holdout,
            patch_size=args.patch_size,
            train_patches_per_image=args.train_patches_per_image,
            train_background_patches_per_image=args.train_background_patches_per_image,
            validation_patches_per_image=args.validation_patches_per_image,
            validation_background_patches_per_image=args.validation_background_patches_per_image,
            background_weight=args.background_weight,
            seed=args.seed,
        )
        return
    _train(args, train, validation, holdout)


if __name__ == "__main__":
    main()
