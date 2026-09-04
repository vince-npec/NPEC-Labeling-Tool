from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.general_root_starter import iou_score, normalize_annotation_targets  # noqa: E402
from resources.models import LabelClass  # noqa: E402
from resources.project_io import load_project  # noqa: E402
from resources.pyphenotyper_adapter import (  # noqa: E402
    PyPhenotyperConfig,
    PyPhenotyperExpertConfig,
    run_pyphenotyper_dataset,
)
from resources.training import (  # noqa: E402
    TrainingConfig,
    _apply_training_augmentations,
    _build_unet_model,
    _collect_patch_arrays,
    _configure_device,
)
from resources.scripts.build_mxlab_dark_rgb_project import (  # noqa: E402
    FULL_PROJECT_PATH,
    HOLDOUT_PROJECT_PATH,
    MANIFEST_PATH,
    OUTPUT_ROOT,
    TRAIN_PROJECT_PATH,
    build_projects,
)


PIPELINE_DIR = REPO_ROOT / "resources" / "npec_pyphenotyper"
HADES_DIR = REPO_ROOT / "resources" / "builtin_models" / "hades_lucifer"
POTATO_DIR = REPO_ROOT / "resources" / "builtin_models" / "potato"
BUILTIN_DIR = REPO_ROOT / "resources" / "builtin_models" / "mxlab_dark_rgb"
MODEL_PATH = BUILTIN_DIR / "mxlab_dark_rgb_multiclass.keras"
TRAINING_SUMMARY_PATH = OUTPUT_ROOT / "mxlab_dark_rgb_training_summary.json"
BENCHMARK_PATH = OUTPUT_ROOT / "mxlab_dark_rgb_benchmark.json"

BASE_PIPELINE_OVERRIDES = {
    "shoot_postprocess_mode": "legacy",
    "adapter_shoot_guard_mode": "off",
}

DARK_RGB_PROFILE_OVERRIDES = {
    "name": "mxlab_dark_rgb_multiclass",
    "mode": "multiclass",
    "input_mode": "rgb",
    "flip_horizontal": False,
    "root_label_ids": (3,),
    "shoot_label_ids": (2,),
    "seed_label_ids": (1,),
    "include_seed_in_shoot": False,
}

DARK_RGB_PIPELINE_OVERRIDES = {
    "shoot_postprocess_mode": "simple",
    "adapter_shoot_guard_mode": "off",
    "shoot_min_area": 1,
    "root_min_area": 8,
    "root_edge_margin_fraction": 0.0,
    "shoot_edge_margin_fraction": 0.0,
}


def _pick_class_id(classes: list[LabelClass], names: tuple[str, ...], fallback: int) -> int:
    lowered = {cls.class_id: cls.name.strip().lower() for cls in classes}
    for token in names:
        for class_id, name in lowered.items():
            if token in name:
                return int(class_id)
    return int(fallback)


def _current_general_starter_experts() -> tuple[PyPhenotyperExpertConfig, ...]:
    return (
        PyPhenotyperExpertConfig(
            key="plate_bw_hades",
            label="BW Arabidopsis (Hades)",
            family="arabidopsis_plate_bw",
            image_mode="grayscale_only",
            priority=0.30,
            pipeline_dir=PIPELINE_DIR,
            root_model_path=HADES_DIR / "model_root_14.h5",
            shoot_model_path=HADES_DIR / "model_shoot_10.h5",
            pipeline_overrides=dict(BASE_PIPELINE_OVERRIDES),
            router_hints={
                "prefer_low_res": True,
                "prefer_dark_top": True,
                "root_area_range": (0.0001, 0.20),
                "shoot_area_range": (0.00002, 0.08),
            },
        ),
        PyPhenotyperExpertConfig(
            key="plate_rgb_lucifer",
            label="RGB Arabidopsis (Lucifer)",
            family="arabidopsis_plate_rgb",
            image_mode="rgb_only",
            priority=0.22,
            pipeline_dir=PIPELINE_DIR,
            root_model_path=HADES_DIR / "best_root_model_patch_256_max_f10.8_max_IoU0.905.h5",
            shoot_model_path=HADES_DIR / "best_shoot_model_patch_256_max_f10.834_max_IoU0.968.h5",
            pipeline_overrides=dict(BASE_PIPELINE_OVERRIDES),
            router_hints={
                "prefer_high_res": True,
                "prefer_green": True,
                "root_area_range": (0.0001, 0.18),
                "shoot_area_range": (0.00003, 0.08),
            },
        ),
        PyPhenotyperExpertConfig(
            key="potato_rgb",
            label="Potato RGB",
            family="potato_rgb",
            image_mode="rgb_only",
            priority=0.18,
            pipeline_dir=PIPELINE_DIR,
            root_model_path=POTATO_DIR / "best_potato_root_june_10_model_patch_256_max_f10.815_max_IoU0.915.h5",
            shoot_model_path=POTATO_DIR / "best_potato_shoot_june_10_model_patch_256_max_f10.811_max_IoU0.969.h5",
            pipeline_overrides=dict(BASE_PIPELINE_OVERRIDES),
            router_hints={
                "prefer_high_res": True,
                "prefer_green": True,
                "root_area_range": (0.0001, 0.22),
                "shoot_area_range": (0.00004, 0.12),
            },
        ),
    )


def _dark_rgb_expert() -> PyPhenotyperExpertConfig:
    return PyPhenotyperExpertConfig(
        key="arabidopsis_dark_rgb",
        label="RGB Arabidopsis (Dark Background)",
        family="arabidopsis_plate_rgb_dark",
        image_mode="rgb_only",
        priority=0.26,
        pipeline_dir=PIPELINE_DIR,
        root_model_path=MODEL_PATH,
        shoot_model_path=MODEL_PATH,
        profile_overrides=dict(DARK_RGB_PROFILE_OVERRIDES),
        pipeline_overrides=dict(DARK_RGB_PIPELINE_OVERRIDES),
        router_hints={
            "prefer_high_res": True,
            "prefer_dark_top": True,
            "root_area_range": (0.00002, 0.015),
            "shoot_area_range": (0.0, 0.003),
        },
    )


def _run_training(epochs: int, patch_size: int, patch_stride: int, batch_size: int, max_patches_per_image: int) -> dict[str, object]:
    train_payload = load_project(TRAIN_PROJECT_PATH)
    dataset_items = list(train_payload["dataset_items"])
    annotations = train_payload["annotations"]
    classes = train_payload["classes"]
    config = TrainingConfig(
        patch_size=int(patch_size),
        patch_stride=int(patch_stride),
        batch_size=int(batch_size),
        epochs=int(epochs),
        learning_rate=1e-3,
        val_split=0.2,
        max_patches_per_image=int(max_patches_per_image),
        min_labeled_ratio=0.0,
        base_filters=24,
        dropout=0.10,
        device_mode="cpu",
        output_path=MODEL_PATH,
        augment_enabled=True,
        augment_replicas=1,
        external_python_path=None,
        export_tflite=False,
        export_hef=False,
    )

    log = lambda message: print(message, flush=True)
    x_arr, y_arr, stats = _collect_patch_arrays(dataset_items, annotations, classes, config, log)
    n = int(x_arr.shape[0])
    val_split = max(0.0, min(0.9, float(config.val_split)))
    val_count = int(round(n * val_split)) if n > 1 else 0
    if val_count >= n:
        val_count = max(0, n - 1)

    rng = np.random.default_rng(config.random_seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    if train_idx.size == 0:
        train_idx = indices
        val_idx = np.array([], dtype=np.int64)

    x_train = x_arr[train_idx]
    y_train = y_arr[train_idx]
    x_val = x_arr[val_idx] if val_idx.size > 0 else None
    y_val = y_arr[val_idx] if val_idx.size > 0 else None

    x_train, y_train, aug_stats = _apply_training_augmentations(x_train, y_train, config, log)
    stats["augmented_patches"] = int(aug_stats.get("augmented_patches", 0))
    stats["train_patches_final"] = int(x_train.shape[0])
    stats["val_patches"] = 0 if x_val is None else int(x_val.shape[0])

    patch_size = int(x_train.shape[1])
    num_classes = max(2, int(stats["max_class_id"]) + 1)

    import tensorflow as tf

    _configure_device(tf, str(config.device_mode), log)
    model = _build_unet_model(
        input_shape=(patch_size, patch_size, 3),
        num_classes=num_classes,
        base_filters=max(8, int(config.base_filters)),
        dropout=max(0.0, min(0.6, float(config.dropout))),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(config.learning_rate)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            tf.keras.metrics.MeanIoU(num_classes=num_classes, sparse_y_true=True, sparse_y_pred=False, name="miou"),
        ],
    )

    total_epochs = max(1, int(config.epochs))
    best_state = {"epoch": 0, "val_miou": None}

    class EpochCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            payload = {"epoch": int(epoch) + 1, "total_epochs": total_epochs}
            if logs:
                for key, value in logs.items():
                    try:
                        payload[str(key)] = float(value)
                    except Exception:
                        pass
                current_val = logs.get("val_miou")
                if current_val is not None:
                    current_val = float(current_val)
                    if best_state["val_miou"] is None or current_val > float(best_state["val_miou"]):
                        best_state["epoch"] = int(epoch) + 1
                        best_state["val_miou"] = current_val
            print(f"[epoch] {json.dumps(payload)}", flush=True)
            print(f"[train] {int(epoch) + 1}/{total_epochs}", flush=True)

    with tempfile.TemporaryDirectory(prefix="mxlab_dark_rgb_best_") as tmp_dir_raw:
        best_model_path = Path(tmp_dir_raw) / "mxlab_dark_rgb_best.keras"
        callbacks: list[tf.keras.callbacks.Callback] = [
            EpochCallback(),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(best_model_path),
                monitor="val_miou",
                mode="max",
                save_best_only=True,
                save_weights_only=False,
                verbose=0,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_miou",
                mode="max",
                patience=4,
                restore_best_weights=True,
                verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_miou",
                mode="max",
                factor=0.5,
                patience=2,
                min_lr=1e-5,
                verbose=0,
            ),
        ]

        train_kwargs = {
            "x": x_train,
            "y": y_train,
            "batch_size": max(1, int(config.batch_size)),
            "epochs": total_epochs,
            "shuffle": True,
            "verbose": 0,
            "callbacks": callbacks,
        }
        if x_val is not None and y_val is not None and x_val.shape[0] > 0:
            train_kwargs["validation_data"] = (x_val, y_val)
            log(f"Training start: train patches={x_train.shape[0]}, val patches={x_val.shape[0]}, batch={train_kwargs['batch_size']}, epochs={total_epochs}")
        else:
            log(f"Training start: train patches={x_train.shape[0]}, val patches=0, batch={train_kwargs['batch_size']}, epochs={total_epochs}")

        history = model.fit(**train_kwargs)

        if best_model_path.exists():
            model = tf.keras.models.load_model(str(best_model_path), compile=False)

    model.save(str(MODEL_PATH))
    log(f"Saved trained model: {MODEL_PATH}")

    history_map = history.history if hasattr(history, "history") else {}
    final = {}
    for key, values in history_map.items():
        if isinstance(values, list) and values:
            try:
                final[key] = float(values[-1])
            except Exception:
                pass

    summary = {
        "cancelled": False,
        "output_model": str(MODEL_PATH),
        "history_final": final,
        "best_epoch": int(best_state["epoch"]),
        "best_val_miou": None if best_state["val_miou"] is None else float(best_state["val_miou"]),
        "runtime": "local_best_checkpoint",
        "stats": stats,
    }
    TRAINING_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _benchmark_config(name: str, config: PyPhenotyperConfig, holdout_project: Path) -> dict[str, object]:
    payload = load_project(holdout_project)
    items = list(payload["dataset_items"])
    annotations = payload["annotations"]
    classes = payload["classes"]
    predictions, metadata = run_pyphenotyper_dataset(items, config)

    routing_counts: dict[str, int] = {}
    root_scores: list[float] = []
    shoot_scores: list[float] = []
    combined_scores: list[float] = []
    cases: list[dict[str, object]] = []

    for item in items:
        pred = predictions.get(item.uid)
        if pred is None:
            continue
        meta = metadata.get(item.uid, {})
        branch = str(meta.get("routing_variant") or meta.get("routing_family") or name)
        routing_counts[branch] = routing_counts.get(branch, 0) + 1
        targets = normalize_annotation_targets(annotations.get(item.uid), classes, item.image.shape[:2])
        root_iou = iou_score(pred == np.uint8(config.root_class_id), targets["root_binary"])
        shoot_iou = iou_score(pred == np.uint8(config.shoot_class_id), targets["shoot"])
        parts = [score for score in (root_iou, shoot_iou) if score is not None]
        combined = float(np.mean(parts)) if parts else None
        if root_iou is not None:
            root_scores.append(float(root_iou))
        if shoot_iou is not None:
            shoot_scores.append(float(shoot_iou))
        if combined is not None:
            combined_scores.append(float(combined))
        cases.append(
            {
                "uid": item.uid,
                "name": item.name,
                "branch": branch,
                "root_iou": None if root_iou is None else float(root_iou),
                "shoot_iou": None if shoot_iou is None else float(shoot_iou),
                "combined_iou": combined,
                "uncertain": bool(meta.get("routing_uncertain")),
            }
        )
    return {
        "name": name,
        "image_count": len(items),
        "routing_counts": routing_counts,
        "root_mean_iou": float(np.mean(root_scores)) if root_scores else None,
        "shoot_mean_iou": float(np.mean(shoot_scores)) if shoot_scores else None,
        "combined_mean_iou": float(np.mean(combined_scores)) if combined_scores else None,
        "cases": cases,
    }


def _run_benchmarks() -> dict[str, object]:
    holdout_payload = load_project(HOLDOUT_PROJECT_PATH)
    classes = holdout_payload["classes"]
    root_class_id = _pick_class_id(classes, ("root",), 3)
    shoot_class_id = _pick_class_id(classes, ("shoot", "hypocotyl"), 2)

    baseline_experts = _current_general_starter_experts()
    dark_expert = _dark_rgb_expert()

    baseline_cfg = PyPhenotyperConfig(
        pipeline_dir=PIPELINE_DIR,
        root_model_path=baseline_experts[0].root_model_path,
        shoot_model_path=baseline_experts[0].shoot_model_path,
        patch_size=256,
        refinement_steps=2,
        root_class_id=root_class_id,
        shoot_class_id=shoot_class_id,
        expert_variants=baseline_experts,
        router_mode="general_root_starter",
    )
    dark_cfg = PyPhenotyperConfig(
        pipeline_dir=PIPELINE_DIR,
        root_model_path=MODEL_PATH,
        shoot_model_path=MODEL_PATH,
        patch_size=256,
        refinement_steps=2,
        root_class_id=root_class_id,
        shoot_class_id=shoot_class_id,
        profile_overrides=dict(DARK_RGB_PROFILE_OVERRIDES),
        pipeline_overrides=dict(DARK_RGB_PIPELINE_OVERRIDES),
    )
    expanded_cfg = PyPhenotyperConfig(
        pipeline_dir=PIPELINE_DIR,
        root_model_path=baseline_experts[0].root_model_path,
        shoot_model_path=baseline_experts[0].shoot_model_path,
        patch_size=256,
        refinement_steps=2,
        root_class_id=root_class_id,
        shoot_class_id=shoot_class_id,
        expert_variants=baseline_experts + (dark_expert,),
        router_mode="general_root_starter",
    )

    report = {
        "holdout_project": str(HOLDOUT_PROJECT_PATH),
        "baseline_general_starter": _benchmark_config("baseline_general_starter", baseline_cfg, HOLDOUT_PROJECT_PATH),
        "dark_rgb_expert_only": _benchmark_config("dark_rgb_expert_only", dark_cfg, HOLDOUT_PROJECT_PATH),
        "expanded_general_starter": _benchmark_config("expanded_general_starter", expanded_cfg, HOLDOUT_PROJECT_PATH),
    }
    BENCHMARK_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and benchmark the MultipleXLab dark-background RGB expert.")
    parser.add_argument("--rebuild-project", action="store_true", help="Rebuild the normalized training projects first.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patch-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-patches-per-image", type=int, default=48)
    args = parser.parse_args()

    if args.rebuild_project or not FULL_PROJECT_PATH.exists() or not TRAIN_PROJECT_PATH.exists() or not HOLDOUT_PROJECT_PATH.exists():
        manifest = build_projects()
        print(json.dumps({"rebuilt_project": manifest}, indent=2))
    elif MANIFEST_PATH.exists():
        print(MANIFEST_PATH.read_text(encoding="utf-8"))

    BUILTIN_DIR.mkdir(parents=True, exist_ok=True)
    training_summary = _run_training(
        epochs=int(args.epochs),
        patch_size=int(args.patch_size),
        patch_stride=int(args.patch_stride),
        batch_size=int(args.batch_size),
        max_patches_per_image=int(args.max_patches_per_image),
    )
    benchmark_report = _run_benchmarks()
    print(
        json.dumps(
            {
                "training_summary_path": str(TRAINING_SUMMARY_PATH),
                "benchmark_path": str(BENCHMARK_PATH),
                "output_model": str(MODEL_PATH),
                "training_runtime": training_summary.get("runtime"),
                "benchmark": benchmark_report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
