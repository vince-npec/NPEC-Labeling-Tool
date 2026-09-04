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
from resources.scripts import train_rgb_inoculated_expert_v2 as trainer  # noqa: E402
from resources.scripts import train_rgb_inoculated_root_union_v3 as root_union  # noqa: E402


DEFAULT_MODEL = root_union.OUTPUT_DIR / root_union.MODEL_NAME
DEFAULT_OUTPUT = root_union.OUTPUT_DIR / "rgb_inoculated_root_union_v3.calibration.json"
DEFAULT_ROOT_PENALTIES = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)


def _parse_values(raw: str) -> tuple[float, ...]:
    values = sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or values[0] <= 0.0:
        raise ValueError("Root penalties must be positive numbers.")
    return tuple(values)


def _benchmark_penalties(
    model,
    samples: Sequence,
    *,
    penalties: Sequence[float],
    patch_size: int,
    halo: int,
    batch_size: int,
) -> dict[str, dict[str, object]]:
    if halo < 0 or 2 * halo >= patch_size:
        raise ValueError("Tile halo must be non-negative and smaller than half the patch size.")
    penalties = tuple(float(value) for value in penalties)
    confusions = {
        penalty: np.zeros((root_union.NUM_CLASSES, root_union.NUM_CLASSES), dtype=np.int64)
        for penalty in penalties
    }
    output_size = patch_size - 2 * halo
    for image_index, sample in enumerate(samples, start=1):
        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Unable to read image: {sample.image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        truth = root_union._semantic_target(sample)
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
                truth_block = truth[y : y + block_height, x : x + block_width]
                for penalty in penalties:
                    adjusted = core.copy()
                    adjusted[..., 3] /= penalty
                    labels = np.argmax(adjusted, axis=-1).astype(np.uint8)
                    trainer._confusion_update(confusions[penalty], truth_block, labels)
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
        print(f"[root-calibration] {image_index}/{len(samples)} {sample.sample_id}", flush=True)
    return {
        str(penalty): root_union._metrics_from_confusion(confusions[penalty])
        for penalty in penalties
    }


def _selection_score(metrics: dict[str, object]) -> float:
    ratio = metrics.get("predicted_to_truth_foreground_ratio")
    if ratio is None or not 0.25 <= float(ratio) <= 4.0:
        return -1.0
    return (
        2.0 * float(metrics.get("root_union_iou") or 0.0)
        + float(metrics.get("shoot_iou") or 0.0)
        + float(metrics.get("foreground_miou") or 0.0)
    )


def main() -> None:
    root_union._configure_trainer()
    parser = argparse.ArgumentParser(
        description="Fit a root-only probability penalty on full validation plates."
    )
    parser.add_argument("--corpus", type=Path, default=trainer.DEFAULT_CORPUS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--root-penalties", default=",".join(str(value) for value in DEFAULT_ROOT_PENALTIES))
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--tile-halo", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    discovered = discover_five_seedling_corpus(Path(args.corpus).expanduser().resolve())
    samples = [sample for sample in discovered if re.fullmatch(r"\d+-\d+", sample.sample_id)]
    _train, validation, holdout = trainer._three_way_grouped_split(
        samples,
        validation_fraction=0.20,
        holdout_fraction=0.20,
    )
    tf = trainer._load_tensorflow_cpu()
    model = tf.keras.models.load_model(str(Path(args.model).expanduser().resolve()), compile=False)
    penalties = _parse_values(str(args.root_penalties))
    validation_grid = _benchmark_penalties(
        model,
        validation,
        penalties=penalties,
        patch_size=int(args.patch_size),
        halo=int(args.tile_halo),
        batch_size=int(args.batch_size),
    )
    selected_penalty = max(penalties, key=lambda value: _selection_score(validation_grid[str(value)]))
    selected_validation = validation_grid[str(selected_penalty)]
    holdout_grid = _benchmark_penalties(
        model,
        holdout,
        penalties=(selected_penalty,),
        patch_size=int(args.patch_size),
        halo=int(args.tile_halo),
        batch_size=int(args.batch_size),
    )
    selected_holdout = holdout_grid[str(selected_penalty)]
    qualification = root_union._qualification(
        selected_holdout,
        min_foreground_miou=0.25,
        min_root_iou=0.40,
        min_shoot_iou=0.45,
        min_foreground_ratio=0.25,
        max_foreground_ratio=4.0,
    )
    payload = {
        "selection_scope": "full_resolution_plate_group_exclusive_validation",
        "selected_root_probability_penalty": selected_penalty,
        "class_probability_scales": [1.0, 1.0, 1.0, 1.0 / selected_penalty],
        "validation_image_count": len(validation),
        "holdout_image_count": len(holdout),
        "validation_grid": validation_grid,
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
