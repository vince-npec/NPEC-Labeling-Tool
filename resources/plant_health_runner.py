from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
ROOT_PATH = SCRIPT_PATH.parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from resources.plant_health import (  # noqa: E402
    builtin_health_model_path,
    build_leaf_mask,
    classify_health_crop,
    compute_stress_cue_mask,
    compute_yellowing_mask,
    crop_mask_region,
    load_health_checkpoint,
)


def _load_binary_mask(path: str | None, shape_hw: tuple[int, int]) -> np.ndarray | None:
    text = str(path or "").strip()
    if not text:
        return None
    file_path = Path(text)
    if not file_path.is_file():
        return None
    mask = np.asarray(Image.open(file_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    h, w = shape_hw
    if mask.shape != (h, w):
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST))
    return np.asarray(mask > 0, dtype=bool)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run built-in PlantVillage leaf-health inference.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta-out", required=True)
    parser.add_argument("--model", default=str(builtin_health_model_path()))
    parser.add_argument("--positive", default="")
    parser.add_argument("--negative", default="")
    parser.add_argument("--box", nargs=4, type=int, default=None)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    h, w = image_rgb.shape[:2]

    positive = _load_binary_mask(args.positive, (h, w))
    negative = _load_binary_mask(args.negative, (h, w))
    box = tuple(int(v) for v in args.box) if args.box is not None else None

    leaf_mask, leaf_meta = build_leaf_mask(
        image_rgb,
        box=box,
        positive_prompt=positive,
        negative_prompt=negative,
    )
    yellow_mask, yellow_meta = compute_yellowing_mask(image_rgb, leaf_mask)
    stress_mask, stress_meta = compute_stress_cue_mask(image_rgb, leaf_mask)
    crop_rgb, crop_box = crop_mask_region(image_rgb, leaf_mask, box=box)

    model, model_meta = load_health_checkpoint(args.model, device=args.device or None)
    health_meta = classify_health_crop(
        model,
        crop_rgb,
        class_names=model_meta.get("class_names"),
        device=model_meta.get("device"),
    )
    stress_probability = float(health_meta.get("stress_probability", 0.0))
    yellowing_score = float(yellow_meta.get("yellowing_score", 0.0))

    if stress_probability < 0.40:
        out_mask = yellow_mask
    elif stress_probability < 0.70:
        out_mask = stress_mask
    else:
        out_mask = np.logical_or(stress_mask, yellow_mask)
    out_mask &= leaf_mask

    if positive is not None:
        out_mask |= positive
    if negative is not None:
        out_mask[negative] = False

    output_path = Path(args.output).expanduser()
    meta_path = Path(args.meta_out).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray((out_mask.astype(np.uint8) * 255)).save(output_path)
    payload = {
        "backend": "plant_health_assist",
        "engine": "plantvillage-binary-mobilenet",
        "model_path": str(Path(args.model).expanduser()),
        "leaf_box": [int(v) for v in crop_box],
        "leaf_pixels": int(leaf_meta.get("leaf_pixels", 0)),
        "leaf_coverage": float(leaf_meta.get("leaf_coverage", 0.0)),
        "yellow_pixels": int(yellow_meta.get("yellow_pixels", 0)),
        "yellowing_score": float(yellowing_score),
        "stress_pixels": int(stress_meta.get("stress_pixels", 0)),
        "stress_cue_score": float(stress_meta.get("stress_cue_score", 0.0)),
        "predicted_label": str(health_meta.get("predicted_label", "unknown")),
        "stress_probability": float(stress_probability),
        "class_names": list(health_meta.get("class_names", [])),
        "probabilities": list(health_meta.get("probabilities", [])),
        "predicted_pixels": int(np.count_nonzero(out_mask)),
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
