from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT_DIR = _repo_root()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from resources.general_root_starter import default_moe_training_plan, iou_score, normalize_annotation_targets
from resources.project_io import load_project
from resources.pyphenotyper_adapter import (
    PyPhenotyperConfig,
    PyPhenotyperExpertConfig,
    run_pyphenotyper_dataset,
)


def _pick_class_id(classes: list, keywords: tuple[str, ...], fallback: int) -> int:
    for cls in classes:
        name = str(getattr(cls, "name", "")).strip().lower()
        if any(keyword in name for keyword in keywords):
            return int(getattr(cls, "class_id"))
    return int(fallback)


def _built_in_general_starter_experts() -> tuple[PyPhenotyperExpertConfig, ...]:
    root = _repo_root() / "resources"
    pipeline_dir = root / "npec_pyphenotyper"
    hades_dir = root / "builtin_models" / "hades_lucifer"
    potato_dir = root / "builtin_models" / "potato"
    dark_dir = root / "builtin_models" / "mxlab_dark_rgb"
    overrides = {
        "shoot_postprocess_mode": "legacy",
        "adapter_shoot_guard_mode": "off",
    }
    experts: list[PyPhenotyperExpertConfig] = [
        PyPhenotyperExpertConfig(
            key="plate_bw_hades",
            label="BW Arabidopsis (Hades)",
            family="arabidopsis_plate_bw",
            image_mode="grayscale_only",
            priority=0.30,
            pipeline_dir=pipeline_dir,
            root_model_path=hades_dir / "model_root_14.h5",
            shoot_model_path=hades_dir / "model_shoot_10.h5",
            pipeline_overrides=dict(overrides),
            router_hints={"prefer_low_res": True, "prefer_dark_top": True},
        ),
        PyPhenotyperExpertConfig(
            key="plate_rgb_lucifer",
            label="RGB Arabidopsis (Lucifer)",
            family="arabidopsis_plate_rgb",
            image_mode="rgb_only",
            priority=0.22,
            pipeline_dir=pipeline_dir,
            root_model_path=hades_dir / "best_root_model_patch_256_max_f10.8_max_IoU0.905.h5",
            shoot_model_path=hades_dir / "best_shoot_model_patch_256_max_f10.834_max_IoU0.968.h5",
            pipeline_overrides=dict(overrides),
            router_hints={"prefer_high_res": True, "prefer_green": True},
        ),
        PyPhenotyperExpertConfig(
            key="potato_rgb",
            label="Potato RGB",
            family="potato_rgb",
            image_mode="rgb_only",
            priority=0.18,
            pipeline_dir=pipeline_dir,
            root_model_path=potato_dir / "best_potato_root_june_10_model_patch_256_max_f10.815_max_IoU0.915.h5",
            shoot_model_path=potato_dir / "best_potato_shoot_june_10_model_patch_256_max_f10.811_max_IoU0.969.h5",
            pipeline_overrides=dict(overrides),
            router_hints={"prefer_high_res": True, "prefer_green": True},
        ),
    ]
    dark_root_model = dark_dir / "mxlab_dark_rgb_root_original.hdf5"
    dark_shoot_model = dark_dir / "mxlab_dark_rgb_multiclass.keras"
    if dark_root_model.exists() and dark_shoot_model.exists():
        experts.append(
            PyPhenotyperExpertConfig(
                key="arabidopsis_dark_rgb",
                label="RGB Arabidopsis (Dark Background)",
                family="arabidopsis_plate_rgb_dark",
                image_mode="rgb_only",
                priority=0.26,
                pipeline_dir=pipeline_dir,
                root_model_path=dark_root_model,
                shoot_model_path=dark_shoot_model,
                root_profile_overrides={
                    "name": "mxlab_dark_rgb_root_original",
                    "mode": "binary_pair",
                    "input_mode": "rgb",
                    "flip_horizontal": False,
                    "return_original_coords": True,
                    "root_threshold": 0.5,
                    "shoot_threshold": 0.5,
                    "batch_size": 12,
                    "predictor_kind": "mxlab_legacy_patch256",
                    "predictor_patch_size": 256,
                },
                shoot_profile_overrides={
                    "name": "mxlab_dark_rgb_multiclass",
                    "mode": "multiclass",
                    "input_mode": "rgb",
                    "flip_horizontal": False,
                    "root_label_ids": (3,),
                    "shoot_label_ids": (2,),
                    "seed_label_ids": (1,),
                    "include_seed_in_shoot": False,
                },
                pipeline_overrides={
                    "shoot_postprocess_mode": "simple",
                    "adapter_shoot_guard_mode": "off",
                    "shoot_min_area": 1,
                    "root_min_area": 8,
                    "root_edge_margin_fraction": 0.0,
                    "shoot_edge_margin_fraction": 0.0,
                },
                router_hints={
                    "prefer_high_res": True,
                    "prefer_dark_top": True,
                    "root_area_range": (0.00002, 0.015),
                    "shoot_area_range": (0.0, 0.003),
                },
            )
        )
    elif dark_shoot_model.exists():
        experts.append(
            PyPhenotyperExpertConfig(
                key="arabidopsis_dark_rgb",
                label="RGB Arabidopsis (Dark Background)",
                family="arabidopsis_plate_rgb_dark",
                image_mode="rgb_only",
                priority=0.26,
                pipeline_dir=pipeline_dir,
                root_model_path=dark_shoot_model,
                shoot_model_path=dark_shoot_model,
                profile_overrides={
                    "name": "mxlab_dark_rgb_multiclass",
                    "mode": "multiclass",
                    "input_mode": "rgb",
                    "flip_horizontal": False,
                    "root_label_ids": (3,),
                    "shoot_label_ids": (2,),
                    "seed_label_ids": (1,),
                    "include_seed_in_shoot": False,
                },
                pipeline_overrides={
                    "shoot_postprocess_mode": "simple",
                    "adapter_shoot_guard_mode": "off",
                    "shoot_min_area": 1,
                    "root_min_area": 8,
                    "root_edge_margin_fraction": 0.0,
                    "shoot_edge_margin_fraction": 0.0,
                },
                router_hints={
                    "prefer_high_res": True,
                    "prefer_dark_top": True,
                    "root_area_range": (0.00002, 0.015),
                    "shoot_area_range": (0.0, 0.003),
                },
            )
        )
    return tuple(experts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the built-in General Root Starter preset on an NPEC project.")
    parser.add_argument("project", type=Path, help="Path to .oclp project with images and annotations")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    payload = load_project(args.project)
    items = list(payload["dataset_items"])
    if int(args.max_images) > 0:
        items = items[: int(args.max_images)]

    classes = payload["classes"]
    root_class_id = _pick_class_id(classes, ("root", "primary"), 1)
    shoot_class_id = _pick_class_id(classes, ("shoot", "leaf", "rosette"), 2)
    config = PyPhenotyperConfig(
        pipeline_dir=_repo_root() / "resources" / "npec_pyphenotyper",
        root_model_path=_repo_root() / "resources" / "builtin_models" / "hades_lucifer" / "model_root_14.h5",
        shoot_model_path=_repo_root() / "resources" / "builtin_models" / "hades_lucifer" / "model_shoot_10.h5",
        refinement_steps=2,
        root_class_id=root_class_id,
        shoot_class_id=shoot_class_id,
        expert_variants=_built_in_general_starter_experts(),
        router_mode="general_root_starter",
    )

    predictions, metadata = run_pyphenotyper_dataset(items, config)
    annotations = payload["annotations"]

    routing_counts: dict[str, int] = {}
    root_scores: list[float] = []
    shoot_scores: list[float] = []
    cases: list[dict[str, object]] = []
    for item in items:
        pred = predictions.get(item.uid)
        if pred is None:
            continue
        meta = metadata.get(item.uid, {})
        branch = str(meta.get("routing_variant") or "unknown")
        routing_counts[branch] = routing_counts.get(branch, 0) + 1
        targets = normalize_annotation_targets(annotations.get(item.uid), classes, item.image.shape[:2])
        root_iou = iou_score(pred == np.uint8(config.root_class_id), targets["root_binary"])
        shoot_iou = iou_score(pred == np.uint8(config.shoot_class_id), targets["shoot"])
        if root_iou is not None:
            root_scores.append(float(root_iou))
        if shoot_iou is not None:
            shoot_scores.append(float(shoot_iou))
        cases.append(
            {
                "uid": item.uid,
                "name": item.name,
                "branch": branch,
                "uncertain": bool(meta.get("routing_uncertain")),
                "root_iou": float(root_iou) if root_iou is not None else None,
                "shoot_iou": float(shoot_iou) if shoot_iou is not None else None,
            }
        )

    summary = {
        "project": str(args.project),
        "image_count": len(items),
        "routing_counts": routing_counts,
        "root_mean_iou": float(np.mean(root_scores)) if root_scores else None,
        "shoot_mean_iou": float(np.mean(shoot_scores)) if shoot_scores else None,
        "training_plan": default_moe_training_plan(),
        "cases": cases,
    }

    output_path = args.output if args.output is not None else (_repo_root() / "resources" / "output" / "general_root_starter_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
