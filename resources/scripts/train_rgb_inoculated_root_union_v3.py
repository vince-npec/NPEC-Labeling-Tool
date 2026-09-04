from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.five_seedling_ownership import (  # noqa: E402
    discover_five_seedling_corpus,
    load_semantic_masks,
)
from resources.scripts import train_rgb_inoculated_expert_v2 as trainer  # noqa: E402


CLASS_NAMES = ("background", "seed", "shoot", "root")
FOREGROUND_IDS = (1, 2, 3)
NUM_CLASSES = len(CLASS_NAMES)
OUTPUT_DIR = REPO_ROOT / "resources" / "output" / "rgb_inoculated_root_union_v3"
MODEL_NAME = "rgb_inoculated_root_union_v3.keras"
PROFILE_NAME = "rgb_inoculated_root_union_v3.profile.json"
BASE_CLASS_WEIGHTS = trainer._class_weights


def _semantic_target(sample) -> np.ndarray:
    masks = load_semantic_masks(sample)
    shape = next(iter(masks.values())).shape[:2]
    target = np.zeros(shape, dtype=np.uint8)
    target[np.logical_or(masks["root"] > 0, masks["lateral"] > 0)] = 3
    target[masks["shoot"] > 0] = 2
    target[masks["seed"] > 0] = 1
    return target


def _metrics_from_confusion(confusion: np.ndarray) -> dict[str, object]:
    confusion = np.asarray(confusion, dtype=np.float64)
    per_class: dict[str, dict[str, float | int | None]] = {}
    foreground_ious: list[float] = []
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
            foreground_ious.append(iou)
    root_tp = confusion[3, 3]
    root_truth = confusion[3, :].sum()
    root_predicted = confusion[:, 3].sum()
    root_union = root_truth + root_predicted - root_tp
    foreground_truth = confusion[1:, :].sum()
    foreground_predicted = confusion[:, 1:].sum()
    return {
        "foreground_miou": None if not foreground_ious else float(np.mean(foreground_ious)),
        "root_union_iou": None if root_union <= 0 else float(root_tp / root_union),
        "shoot_iou": per_class["shoot"]["iou"],
        "predicted_to_truth_foreground_ratio": (
            None if foreground_truth <= 0 else float(foreground_predicted / foreground_truth)
        ),
        "per_class": per_class,
    }


def _build_root_union_model(tf, base_model_path: Path):
    base = tf.keras.models.load_model(str(base_model_path), compile=False)
    if len(base.inputs) != 1 or len(base.outputs) != 1 or int(base.output_shape[-1]) != 6:
        raise ValueError("The root-union trainer requires the bundled six-channel base model.")
    old_head = base.layers[-1]
    if not isinstance(old_head, tf.keras.layers.Conv2D) or tuple(old_head.kernel_size) != (1, 1):
        raise ValueError("The base model must end in a 1x1 Conv2D semantic head.")
    new_head = tf.keras.layers.Conv2D(
        NUM_CLASSES,
        (1, 1),
        padding="same",
        activation="softmax",
        name="semantic_root_union_4class",
    )
    model = tf.keras.Model(
        base.input,
        new_head(old_head.input),
        name="rgb_inoculated_root_union_v3",
    )
    old_kernel, old_bias = old_head.get_weights()
    new_kernel = np.zeros((*old_kernel.shape[:-1], NUM_CLASSES), dtype=old_kernel.dtype)
    new_bias = np.zeros((NUM_CLASSES,), dtype=old_bias.dtype)
    new_kernel[..., 0] = 0.5 * (old_kernel[..., 0] + old_kernel[..., 5])
    new_bias[0] = np.logaddexp(old_bias[0], old_bias[5])
    new_kernel[..., 1] = old_kernel[..., 1]
    new_bias[1] = old_bias[1]
    new_kernel[..., 2] = old_kernel[..., 2]
    new_bias[2] = old_bias[2]
    new_kernel[..., 3] = 0.5 * (old_kernel[..., 3] + old_kernel[..., 4])
    new_bias[3] = np.logaddexp(old_bias[3], old_bias[4])
    new_head.set_weights([new_kernel, new_bias])
    return model, new_head


def _profile_payload(
    *,
    model_path: Path,
    base_model_path: Path,
    corpus: Path,
    split: dict[str, object],
    benchmark: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "name": "rgb_inoculated_root_union_v3",
        "mode": "multiclass",
        "input_mode": "rgb",
        "flip_horizontal": False,
        "return_original_coords": True,
        "class_names": list(CLASS_NAMES),
        "class_ids": {"background": 0, "seed": 1, "shoot": 2, "root": 3},
        "root_label_ids": [3],
        "primary_root_label_ids": [],
        "lateral_label_ids": [],
        "shoot_label_ids": [2],
        "seed_label_ids": [1],
        "include_seed_in_shoot": False,
        "tile_halo": 32,
        "model_path": str(model_path),
        "initialized_from": str(base_model_path),
        "training_corpus": str(corpus),
        "split": split,
        "holdout_benchmark": benchmark,
        "root_class_policy": (
            "The learned class is visible root union. Primary/lateral separation is performed "
            "after inference from whole-root topology."
        ),
        "gap_policy": {
            "handwritten_bacterial_gaps": "background_or_unknown",
            "trained_as_visible_root": False,
            "note": (
                "Occluded-root recovery is separate provenance-bearing output and is never "
                "merged into visible-root ground truth."
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
    per_class = metrics.get("per_class", {})
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
        "all_foreground_classes_present": all(
            int(per_class.get(name, {}).get("predicted_pixels") or 0) > 0
            for name in ("seed", "shoot", "root")
        ),
    }
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


def _root_union_class_weights(counts: np.ndarray, *, background_weight: float = 1.0) -> np.ndarray:
    weights = BASE_CLASS_WEIGHTS(counts, background_weight=background_weight)
    weights[3] = max(float(weights[3]), 8.0)
    return weights


def _configure_trainer() -> None:
    trainer.CLASS_NAMES = CLASS_NAMES
    trainer.FOREGROUND_IDS = FOREGROUND_IDS
    trainer.NUM_CLASSES = NUM_CLASSES
    trainer.DEFAULT_OUTPUT_DIR = OUTPUT_DIR
    trainer.INSTALL_MODEL_NAME = MODEL_NAME
    trainer.INSTALL_PROFILE_NAME = PROFILE_NAME
    trainer._semantic_target = _semantic_target
    trainer._metrics_from_confusion = _metrics_from_confusion
    trainer._build_five_class_model = _build_root_union_model
    trainer._profile_payload = _profile_payload
    trainer._qualification = _qualification
    trainer._class_weights = _root_union_class_weights


def main() -> None:
    _configure_trainer()
    parser = trainer._parser()
    parser.description = "Train and gate a root-union/shoot/seed RGB inoculated Arabidopsis expert."
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 0.5 or not 0.0 < args.holdout_fraction < 0.5:
        raise ValueError("Validation and holdout fractions must each be between 0 and 0.5.")
    if args.validation_fraction + args.holdout_fraction >= 0.8:
        raise ValueError("Validation plus holdout must leave a substantial training split.")
    corpus = Path(args.corpus).expanduser().resolve()
    discovered = discover_five_seedling_corpus(corpus)
    samples = [sample for sample in discovered if re.fullmatch(r"\d+-\d+", sample.sample_id)]
    excluded = [sample.sample_id for sample in discovered if sample not in samples]
    if excluded:
        print(f"Excluded ambiguous sample IDs: {', '.join(excluded)}", flush=True)
    train, validation, holdout = trainer._three_way_grouped_split(
        samples,
        validation_fraction=args.validation_fraction,
        holdout_fraction=args.holdout_fraction,
    )
    if args.dry_run_data:
        trainer._dry_run(
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
    trainer._train(args, train, validation, holdout)


if __name__ == "__main__":
    main()
