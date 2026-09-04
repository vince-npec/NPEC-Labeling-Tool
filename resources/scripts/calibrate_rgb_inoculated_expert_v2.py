from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.five_seedling_ownership import discover_five_seedling_corpus  # noqa: E402
from resources.scripts.train_rgb_inoculated_expert_v2 import (  # noqa: E402
    DEFAULT_CORPUS,
    DEFAULT_OUTPUT_DIR,
    NUM_CLASSES,
    _confusion_update,
    _load_tensorflow_cpu,
    _metrics_from_confusion,
    _qualification,
    _semantic_target,
    _three_way_grouped_split,
)


DEFAULT_MODEL = DEFAULT_OUTPUT_DIR / "rgb_inoculated_multiclass_v2.keras"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "rgb_inoculated_multiclass_v2.calibration.json"
DEFAULT_BACKGROUND_SCALES = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)


def _parse_scales(raw: str) -> tuple[float, ...]:
    values = sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or values[0] <= 0.0:
        raise ValueError("Background scales must be positive numbers.")
    return tuple(values)


def _benchmark_background_scales(
    model,
    samples: Sequence,
    *,
    scales: Sequence[float],
    patch_size: int,
    halo: int,
    batch_size: int,
) -> dict[str, dict[str, object]]:
    if halo < 0 or 2 * halo >= patch_size:
        raise ValueError("Tile halo must be non-negative and smaller than half the patch size.")
    scales = tuple(float(value) for value in scales)
    confusions = {
        scale: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for scale in scales
    }
    output_size = patch_size - 2 * halo

    for image_index, sample in enumerate(samples, start=1):
        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Unable to read image: {sample.image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        truth = _semantic_target(sample)
        height, width = truth.shape
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
        pending: list[np.ndarray] = []
        locations: list[tuple[int, int, int, int]] = []

        def flush() -> None:
            if not pending:
                return
            probabilities = model.predict_on_batch(np.asarray(pending, dtype=np.float32) / 255.0)
            for probability, (y, x, block_height, block_width) in zip(probabilities, locations):
                core = probability[
                    halo : halo + block_height,
                    halo : halo + block_width,
                ]
                foreground_class = np.argmax(core[..., 1:], axis=-1).astype(np.uint8) + 1
                foreground_probability = np.max(core[..., 1:], axis=-1)
                background_probability = core[..., 0]
                truth_block = truth[y : y + block_height, x : x + block_width]
                for scale in scales:
                    labels = np.where(
                        background_probability * scale >= foreground_probability,
                        0,
                        foreground_class,
                    ).astype(np.uint8)
                    _confusion_update(confusions[scale], truth_block, labels)
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
        print(f"[calibration] {image_index}/{len(samples)} {sample.sample_id}", flush=True)

    return {str(scale): _metrics_from_confusion(confusions[scale]) for scale in scales}


def _selection_score(metrics: dict[str, object]) -> float:
    ratio = metrics.get("predicted_to_truth_foreground_ratio")
    if ratio is None or not 0.25 <= float(ratio) <= 4.0:
        return -1.0
    return (
        float(metrics.get("foreground_miou") or 0.0)
        + float(metrics.get("root_union_iou") or 0.0)
        + float(metrics.get("shoot_iou") or 0.0)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the five-class model background prior on full validation plates."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--background-scales", default=",".join(str(value) for value in DEFAULT_BACKGROUND_SCALES))
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--tile-halo", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()

    discovered = discover_five_seedling_corpus(Path(args.corpus).expanduser().resolve())
    samples = [sample for sample in discovered if re.fullmatch(r"\d+-\d+", sample.sample_id)]
    _train, validation, holdout = _three_way_grouped_split(
        samples,
        validation_fraction=float(args.validation_fraction),
        holdout_fraction=float(args.holdout_fraction),
    )
    tf = _load_tensorflow_cpu()
    model = tf.keras.models.load_model(str(Path(args.model).expanduser().resolve()), compile=False)
    scales = _parse_scales(str(args.background_scales))
    validation_results = _benchmark_background_scales(
        model,
        validation,
        scales=scales,
        patch_size=int(args.patch_size),
        halo=int(args.tile_halo),
        batch_size=int(args.batch_size),
    )
    selected_scale = max(scales, key=lambda value: _selection_score(validation_results[str(value)]))
    selected_validation = validation_results[str(selected_scale)]
    if _selection_score(selected_validation) < 0.0:
        selected_scale = min(
            scales,
            key=lambda value: abs(
                math.log(max(1e-9, float(validation_results[str(value)].get("predicted_to_truth_foreground_ratio") or 1e9)))
            ),
        )
        selected_validation = validation_results[str(selected_scale)]

    holdout_results = _benchmark_background_scales(
        model,
        holdout,
        scales=(selected_scale,),
        patch_size=int(args.patch_size),
        halo=int(args.tile_halo),
        batch_size=int(args.batch_size),
    )
    selected_holdout = holdout_results[str(selected_scale)]
    qualification = _qualification(
        selected_holdout,
        min_foreground_miou=0.25,
        min_root_iou=0.40,
        min_shoot_iou=0.45,
        min_foreground_ratio=0.25,
        max_foreground_ratio=4.0,
    )
    payload = {
        "selection_scope": "full_resolution_plate_group_exclusive_validation",
        "selection_metric": "foreground_miou + root_union_iou + shoot_iou, foreground ratio constrained",
        "selected_background_probability_scale": selected_scale,
        "validation_image_count": len(validation),
        "holdout_image_count": len(holdout),
        "validation_grid": validation_results,
        "selected_validation": selected_validation,
        "selected_holdout": selected_holdout,
        "holdout_qualification": qualification,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
