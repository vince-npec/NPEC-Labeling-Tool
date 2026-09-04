from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs" / "rgb_inoculated_root_union_v3"
SOURCE_MODEL = OUTPUT_DIR / "rgb_inoculated_root_union_v3.keras"
CALIBRATION = OUTPUT_DIR / "rgb_inoculated_root_union_v3.calibration.json"
DESTINATION_DIR = REPO_ROOT / "resources" / "builtin_models" / "rgb_inoculated"
DESTINATION_MODEL = DESTINATION_DIR / "rgb_inoculated_shoot_v3.keras"
DESTINATION_PROFILE = DESTINATION_DIR / "rgb_inoculated_shoot_v3.profile.json"


def main() -> None:
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    validation = payload.get("selected_validation", {})
    holdout = payload.get("selected_holdout", {})
    validation_shoot_iou = float(validation.get("shoot_iou") or 0.0)
    holdout_shoot_iou = float(holdout.get("shoot_iou") or 0.0)
    holdout_shoot = holdout.get("per_class", {}).get("shoot", {})
    truth_pixels = int(holdout_shoot.get("truth_pixels") or 0)
    predicted_pixels = int(holdout_shoot.get("predicted_pixels") or 0)
    pixel_ratio = float(predicted_pixels / truth_pixels) if truth_pixels > 0 else 0.0
    checks = {
        "validation_shoot_iou_at_least_0_45": validation_shoot_iou >= 0.45,
        "holdout_shoot_iou_at_least_0_45": holdout_shoot_iou >= 0.45,
        "holdout_shoot_pixel_ratio_between_0_5_and_2": 0.5 <= pixel_ratio <= 2.0,
        "source_model_present": SOURCE_MODEL.is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Shoot-only promotion refused: {checks}")

    profile = {
        "name": "rgb_inoculated_shoot_v3",
        "purpose": "shoot_only",
        "prohibited_targets": ["root", "primary_root", "lateral_root", "seed"],
        "mode": "multiclass",
        "input_mode": "rgb",
        "flip_horizontal": False,
        "return_original_coords": True,
        "class_names": ["background", "seed", "shoot", "root"],
        "root_label_ids": [3],
        "lateral_label_ids": [],
        "shoot_label_ids": [2],
        "seed_label_ids": [1],
        "include_seed_in_shoot": False,
        "tile_halo": 32,
        "class_probability_scales": [1.0, 1.0, 1.0, 1.0],
        "qualification": {
            "qualified": True,
            "scope": "shoot_only",
            "checks": checks,
            "validation_shoot_iou": validation_shoot_iou,
            "holdout_shoot_iou": holdout_shoot_iou,
            "holdout_shoot_predicted_to_truth_ratio": pixel_ratio,
            "root_channel_qualified": False,
            "seed_channel_qualified": False,
        },
        "training_corpus": "external:yang-ground-truth-feb-2026",
        "provenance": {
            "person": "Yang Song",
            "project": "DroughtFighters",
            "group": "Plant Microbe Interaction",
            "institution": "Utrecht University",
            "legacy_identifier": "yang",
        },
        "gap_policy": (
            "Handwritten bacterial gaps remain background/unknown. This shoot-only model does not "
            "perform root-gap repair."
        ),
    }

    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
    temporary_model = DESTINATION_MODEL.with_suffix(DESTINATION_MODEL.suffix + ".tmp")
    temporary_profile = DESTINATION_PROFILE.with_suffix(DESTINATION_PROFILE.suffix + ".tmp")
    shutil.copy2(SOURCE_MODEL, temporary_model)
    temporary_profile.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    os.replace(temporary_model, DESTINATION_MODEL)
    os.replace(temporary_profile, DESTINATION_PROFILE)
    print(json.dumps({"installed_model": str(DESTINATION_MODEL), "profile": str(DESTINATION_PROFILE), **profile["qualification"]}, indent=2))


if __name__ == "__main__":
    main()
