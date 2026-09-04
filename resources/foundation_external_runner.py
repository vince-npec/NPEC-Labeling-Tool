from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import cv2
import numpy as np


def _load_gray_mask(path: Path | None, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if path is None:
        return np.zeros((h, w), dtype=np.uint8)
    if not path.exists():
        return np.zeros((h, w), dtype=np.uint8)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return np.zeros((h, w), dtype=np.uint8)
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    if raw.shape[:2] != (h, w):
        raw = cv2.resize(raw.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return np.asarray(raw > 0, dtype=np.uint8)


def _clamp_box(box: tuple[int, int, int, int] | None, shape_hw: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    h, w = int(shape_hw[0]), int(shape_hw[1])
    x, y, bw, bh = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    bw = max(1, min(w - x, bw))
    bh = max(1, min(h - y, bh))
    return (x, y, bw, bh)


def _run_fallback_segmenter(
    image_rgb: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    box: tuple[int, int, int, int] | None,
    prior: np.ndarray | None,
    sigma: float,
    iters: int,
    smooth: int,
) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    src = np.asarray(image_rgb, dtype=np.uint8)
    blur = cv2.GaussianBlur(src, (0, 0), sigmaX=max(0.0, float(sigma)), sigmaY=max(0.0, float(sigma)))
    sharp = cv2.addWeighted(src, 1.35, blur, -0.35, 0.0)

    work = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    if prior is not None:
        work[np.asarray(prior > 0)] = cv2.GC_PR_FGD
    work[np.asarray(positive > 0)] = cv2.GC_FGD
    work[np.asarray(negative > 0)] = cv2.GC_BGD
    rect = _clamp_box(box, (h, w))
    if rect is not None:
        x, y, bw, bh = rect
        outside = np.ones((h, w), dtype=bool)
        outside[y : y + bh, x : x + bw] = False
        work[outside] = cv2.GC_BGD

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    hard = int(np.count_nonzero((work == cv2.GC_BGD) | (work == cv2.GC_FGD)))
    mode = cv2.GC_INIT_WITH_MASK if hard > 0 else cv2.GC_INIT_WITH_RECT
    if rect is None:
        rect = (0, 0, max(1, w), max(1, h))
    try:
        cv2.grabCut(sharp, work, rect, bg_model, fg_model, iterCount=max(1, int(iters)), mode=mode)
        out = np.logical_or(work == cv2.GC_PR_FGD, work == cv2.GC_FGD)
    except Exception:
        gray = cv2.cvtColor(sharp, cv2.COLOR_RGB2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out = otsu > 0

    if rect is not None:
        x, y, bw, bh = rect
        scoped = np.zeros((h, w), dtype=bool)
        scoped[y : y + bh, x : x + bw] = out[y : y + bh, x : x + bw]
        out = scoped
    out[np.asarray(positive > 0)] = True
    out[np.asarray(negative > 0)] = False

    k = max(1, int(smooth))
    if k > 1:
        kernel = np.ones((k, k), dtype=np.uint8)
        out_u8 = (np.asarray(out, dtype=np.uint8) * 255).astype(np.uint8)
        out_u8 = cv2.morphologyEx(out_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        out_u8 = cv2.morphologyEx(out_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        out = out_u8 > 0
    return np.asarray(out, dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="NPEC Foundation external runner adapter.")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--positive", required=True)
    parser.add_argument("--negative", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta-out", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--prior", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--box", nargs=4, type=int)
    args, _unknown = parser.parse_known_args()

    image_path = Path(args.image)
    output_path = Path(args.output)
    meta_path = Path(args.meta_out)
    config_path = Path(args.config) if args.config else None
    prior_path = Path(args.prior) if args.prior else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    meta: dict[str, object] = {
        "backend": str(args.backend).strip().lower(),
        "engine": "fallback-opencv-adapter",
        "checkpoint": str(args.checkpoint).strip() or None,
    }

    try:
        raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        if raw.ndim == 2:
            image_rgb = cv2.cvtColor(raw.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(raw[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        positive = _load_gray_mask(Path(args.positive), (h, w))
        negative = _load_gray_mask(Path(args.negative), (h, w))
        prior = _load_gray_mask(prior_path, (h, w)) if prior_path is not None else None

        cfg = {}
        if config_path is not None and config_path.exists():
            try:
                loaded = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    cfg = loaded
            except Exception:
                cfg = {}

        sigma = float(cfg.get("sigma", 1.1))
        iters = int(cfg.get("grabcut_iters", cfg.get("focal_refine_iters", 3)))
        smooth = int(cfg.get("smooth_kernel", 3))
        box = tuple(int(v) for v in args.box) if args.box is not None and len(args.box) == 4 else None

        out = _run_fallback_segmenter(
            image_rgb=image_rgb,
            positive=positive,
            negative=negative,
            box=box,
            prior=prior,
            sigma=sigma,
            iters=iters,
            smooth=smooth,
        )
        if not cv2.imwrite(str(output_path), (np.asarray(out > 0, dtype=np.uint8) * 255)):
            raise RuntimeError(f"Failed to write output mask: {output_path}")

        meta.update(
            {
                "status": "ok",
                "prompt_positive_px": int(np.count_nonzero(positive)),
                "prompt_negative_px": int(np.count_nonzero(negative)),
                "predicted_pixels": int(np.count_nonzero(out)),
                "text_prompt": str(cfg.get("text_prompt", "")).strip(),
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        meta.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        try:
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
