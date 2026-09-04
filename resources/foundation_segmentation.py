from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Callable

import cv2
import numpy as np

from .runtime_python_utils import common_external_python_candidates, safe_current_python_executable


@dataclass(slots=True)
class FoundationSegmentationConfig:
    sigma: float = 1.1
    grabcut_iters: int = 2
    focal_refine_iters: int = 3
    smooth_kernel: int = 3
    external_python: str = ""
    external_runner: str = ""
    external_checkpoint: str = ""
    external_extra_args: str = ""
    external_timeout_sec: int = 240
    text_prompt: str = ""


class FoundationSegmentationCancelled(RuntimeError):
    """Raised when a foundation inference run is cancelled by the caller."""


def available_foundation_backends() -> list[tuple[str, str]]:
    return [
        ("Plant Health Assist (PlantVillage, built-in)", "plant_health_assist"),
        ("Colony SAM-2 (round petri dishes, built-in)", "colony_sam2"),
        ("SAM-2 (promptable, built-in)", "sam2"),
        ("FocalClick-XL (refine, built-in)", "focalclick"),
        ("X-SAM (external adapter)", "xsam_external"),
        ("SAM-2 (external adapter)", "sam2_external"),
        ("FocalClick (external adapter)", "focalclick_external"),
    ]


_AUTO_PROMPTLESS_BACKENDS = {"colony_sam2", "plant_health_assist"}


def _as_bool_mask(mask: np.ndarray | None, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if mask is None:
        return np.zeros((h, w), dtype=bool)
    arr = np.asarray(mask)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D prompt mask, got shape {tuple(arr.shape)}")
    if arr.shape != (h, w):
        arr = cv2.resize(arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return np.asarray(arr > 0, dtype=bool)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return np.asarray(arr[:, :, :3], dtype=np.uint8)
    raise ValueError(f"Unsupported image shape: {tuple(arr.shape)}")


def is_external_foundation_backend(backend: str) -> bool:
    key = str(backend).strip().lower()
    return key in {"xsam_external", "sam2_external", "focalclick_external"}


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


def _enhance_for_edges(image_rgb: np.ndarray, sigma: float) -> np.ndarray:
    s = max(0.0, float(sigma))
    if s <= 0.0:
        return np.asarray(image_rgb, dtype=np.uint8)
    src = np.asarray(image_rgb, dtype=np.uint8)
    blur = cv2.GaussianBlur(src, (0, 0), sigmaX=s, sigmaY=s)
    sharpened = cv2.addWeighted(src, 1.35, blur, -0.35, 0.0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _grabcut_from_prompts(
    image_rgb: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    box: tuple[int, int, int, int] | None,
    prior_mask: np.ndarray | None,
    iters: int,
) -> np.ndarray:
    image = _as_rgb(image_rgb)
    h, w = image.shape[:2]
    work = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    if prior_mask is not None:
        prior = _as_bool_mask(prior_mask, (h, w))
        work[prior] = cv2.GC_PR_FGD

    pos = _as_bool_mask(positive, (h, w))
    neg = _as_bool_mask(negative, (h, w))
    work[pos] = cv2.GC_FGD
    work[neg] = cv2.GC_BGD

    rect = _clamp_box(box, (h, w))
    if rect is not None:
        x, y, bw, bh = rect
        outside = np.ones((h, w), dtype=bool)
        outside[y : y + bh, x : x + bw] = False
        work[outside] = cv2.GC_BGD

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    hard_prompt_pixels = int(np.count_nonzero((work == cv2.GC_FGD) | (work == cv2.GC_BGD)))
    mode = cv2.GC_INIT_WITH_MASK if hard_prompt_pixels > 0 else cv2.GC_INIT_WITH_RECT
    if rect is None:
        rect = (0, 0, max(1, w), max(1, h))

    try:
        cv2.grabCut(
            image,
            work,
            rect,
            bg_model,
            fg_model,
            iterCount=max(1, int(iters)),
            mode=mode,
        )
        pred = np.logical_or(work == cv2.GC_FGD, work == cv2.GC_PR_FGD)
    except Exception:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pred = otsu > 0
        if rect is not None:
            x, y, bw, bh = rect
            scoped = np.zeros((h, w), dtype=bool)
            scoped[y : y + bh, x : x + bw] = pred[y : y + bh, x : x + bw]
            pred = scoped

    pred[pos] = True
    pred[neg] = False
    if rect is not None:
        x, y, bw, bh = rect
        scoped = np.zeros((h, w), dtype=bool)
        scoped[y : y + bh, x : x + bw] = pred[y : y + bh, x : x + bw]
        pred = scoped
    return np.asarray(pred, dtype=bool)


def _seed_intensity_fallback(
    image_rgb: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    image = _as_rgb(image_rgb)
    h, w = image.shape[:2]
    pos = _as_bool_mask(positive, (h, w))
    neg = _as_bool_mask(negative, (h, w))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    if int(np.count_nonzero(pos)) <= 0:
        return np.zeros((h, w), dtype=bool)

    pos_vals = gray[pos]
    neg_vals = gray[neg] if int(np.count_nonzero(neg)) > 0 else np.array([], dtype=np.float32)

    if neg_vals.size > 0:
        mean_pos = float(np.mean(pos_vals))
        mean_neg = float(np.mean(neg_vals))
        candidate = np.abs(gray - mean_pos) <= np.abs(gray - mean_neg)
    else:
        med = float(np.median(pos_vals))
        mad = float(np.median(np.abs(pos_vals - med)))
        spread = max(8.0, mad * 4.5)
        lo = med - spread
        hi = med + spread
        candidate = np.logical_and(gray >= lo, gray <= hi)

    if box is not None:
        x, y, bw, bh = _clamp_box(box, (h, w)) or (0, 0, w, h)
        scoped = np.zeros((h, w), dtype=bool)
        scoped[y : y + bh, x : x + bw] = candidate[y : y + bh, x : x + bw]
        candidate = scoped

    candidate[neg] = False
    candidate[pos] = True

    cc_src = candidate.astype(np.uint8)
    cc_count, labels = cv2.connectedComponents(cc_src)
    if cc_count > 1:
        keep = np.zeros((h, w), dtype=bool)
        seed_labels = labels[pos]
        for lbl in np.unique(seed_labels):
            lid = int(lbl)
            if lid <= 0:
                continue
            keep |= labels == lid
        if int(np.count_nonzero(keep)) > 0:
            candidate = keep

    candidate = _post_smooth(candidate, 3)
    candidate[pos] = True
    candidate[neg] = False
    return np.asarray(candidate, dtype=bool)


def _post_smooth(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    k = max(1, int(kernel_size))
    if k <= 1:
        return np.asarray(mask > 0, dtype=bool)
    kernel = np.ones((k, k), dtype=np.uint8)
    src = (np.asarray(mask, dtype=np.uint8) * 255).astype(np.uint8)
    closed = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened > 0


def _center_scored_circle(
    circles: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[int, int, int] | None:
    h, w = int(image_shape[0]), int(image_shape[1])
    cx = (float(w) - 1.0) * 0.5
    cy = (float(h) - 1.0) * 0.5
    best: tuple[int, int, int] | None = None
    best_score = -1e18
    for raw in circles.reshape(-1, 3):
        x, y, radius = (int(round(float(raw[0]))), int(round(float(raw[1]))), int(round(float(raw[2]))))
        if radius <= 0:
            continue
        dist = float(np.hypot(float(x) - cx, float(y) - cy))
        score = float(radius) - (0.55 * dist)
        if score > best_score:
            best = (x, y, radius)
            best_score = score
    return best


def _score_round_plate_circle(
    gray: np.ndarray,
    grad_mag: np.ndarray,
    circle: tuple[int, int, int],
    *,
    box_prompt: tuple[int, int, int, int] | None = None,
) -> float:
    h, w = gray.shape[:2]
    cx, cy, radius = (int(circle[0]), int(circle[1]), int(circle[2]))
    if radius <= 8:
        return -1e18

    yy, xx = np.indices((h, w), dtype=np.float32)
    dist = np.sqrt((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2)
    band = max(3.0, float(radius) * 0.025)

    interior = dist <= max(1.0, float(radius) - (2.5 * band))
    ring = np.logical_and(dist >= float(radius) - band, dist <= float(radius) + band)
    inner_rim = np.logical_and(dist >= float(radius) - (2.0 * band), dist <= float(radius) - (0.4 * band))
    outer_rim = np.logical_and(dist >= float(radius) + (0.4 * band), dist <= float(radius) + (2.0 * band))

    if box_prompt is not None:
        x0, y0, bw, bh = _clamp_box(box_prompt, (h, w)) or (0, 0, w, h)
        roi = np.zeros((h, w), dtype=bool)
        roi[y0 : y0 + bh, x0 : x0 + bw] = True
        interior &= roi
        ring &= roi
        inner_rim &= roi
        outer_rim &= roi

    if int(np.count_nonzero(interior)) < 256 or int(np.count_nonzero(ring)) < 128:
        return -1e18

    edge_strength = float(np.mean(grad_mag[ring])) / 255.0
    rim_contrast = 0.0
    if int(np.count_nonzero(inner_rim)) > 0 and int(np.count_nonzero(outer_rim)) > 0:
        rim_contrast = abs(float(np.mean(gray[inner_rim])) - float(np.mean(gray[outer_rim]))) / 255.0
    interior_std = float(np.std(gray[interior])) / 64.0

    cx_img = (float(w) - 1.0) * 0.5
    cy_img = (float(h) - 1.0) * 0.5
    center_dist = float(np.hypot(float(cx) - cx_img, float(cy) - cy_img)) / max(1.0, float(min(h, w)))

    touch_overflow = max(
        0.0,
        float(radius) - float(cx),
        float(radius) - float(cy),
        float(cx + radius) - float(w - 1),
        float(cy + radius) - float(h - 1),
    ) / max(1.0, float(radius))

    return (
        (2.8 * edge_strength)
        + (2.2 * rim_contrast)
        + (0.25 * (float(radius) / max(1.0, float(min(h, w)))))
        - (0.85 * interior_std)
        - (0.65 * center_dist)
        - (2.5 * touch_overflow)
    )


def _detect_round_plate_circle(
    image_rgb: np.ndarray,
    box_prompt: tuple[int, int, int, int] | None = None,
) -> tuple[tuple[int, int, int], dict[str, object]]:
    image = _as_rgb(image_rgb)
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_sigma = max(1.5, float(min(h, w)) * 0.008)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.clip(cv2.magnitude(grad_x, grad_y), 0.0, 255.0).astype(np.float32)

    min_dim = float(min(h, w))
    min_radius = max(24, int(round(min_dim * 0.24)))
    max_radius = max(min_radius + 4, int(round(min_dim * 0.52)))
    min_dist = max(32, int(round(min_dim * 0.40)))

    candidate_circles: list[tuple[tuple[int, int, int], str]] = []

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=90,
        param2=28,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None and circles.size > 0:
        for raw in np.asarray(circles).reshape(-1, 3):
            circle = (int(round(float(raw[0]))), int(round(float(raw[1]))), int(round(float(raw[2]))))
            if circle[2] > 0:
                candidate_circles.append((circle, "hough_circle"))

    edges = cv2.Canny(blurred, 60, 140)
    contours, _hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0.0:
                continue
            perimeter = float(cv2.arcLength(contour, closed=True))
            if perimeter <= 0.0:
                continue
            circularity = (4.0 * np.pi * area) / max(1.0, perimeter * perimeter)
            (xc, yc), radius_f = cv2.minEnclosingCircle(contour)
            radius = int(round(radius_f))
            if radius < min_radius or radius > max_radius:
                continue
            if circularity < 0.42:
                continue
            candidate_circles.append(((int(round(xc)), int(round(yc)), int(radius)), "contour_circle"))

    best_circle: tuple[int, int, int] | None = None
    best_meta: dict[str, object] | None = None
    best_score = -1e18
    seen: set[tuple[int, int, int]] = set()
    for circle, strategy in candidate_circles:
        dedupe_key = (
            int(round(circle[0] / 4.0)),
            int(round(circle[1] / 4.0)),
            int(round(circle[2] / 4.0)),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        score = _score_round_plate_circle(gray, grad_mag, circle, box_prompt=box_prompt)
        if score > best_score:
            best_score = score
            best_circle = circle
            best_meta = {"plate_strategy": strategy, "plate_score": float(score), "plate_candidate_count": int(len(candidate_circles))}
    if best_circle is not None and best_meta is not None:
        return best_circle, best_meta

    fallback = (w // 2, h // 2, max(16, int(round(min_dim * 0.45))))
    fallback_score = _score_round_plate_circle(gray, grad_mag, fallback, box_prompt=box_prompt)
    return fallback, {"plate_strategy": "image_center_fallback", "plate_score": float(fallback_score), "plate_candidate_count": 0}


def _circle_mask(shape_hw: tuple[int, int], circle: tuple[int, int, int], inset: float = 0.0) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    cx, cy, radius = (int(circle[0]), int(circle[1]), int(circle[2]))
    rr = max(1.0, float(radius) - float(inset))
    yy, xx = np.indices((h, w), dtype=np.float32)
    return ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) <= (rr * rr)


def _colony_candidate_response(image_rgb: np.ndarray, plate_mask: np.ndarray) -> np.ndarray:
    image = _as_rgb(image_rgb)
    h, w = image.shape[:2]
    plate_bool = np.asarray(plate_mask, dtype=bool)
    if int(np.count_nonzero(plate_bool)) <= 0:
        return np.zeros((h, w), dtype=np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_chan = np.asarray(lab[:, :, 0], dtype=np.uint8)
    a_chan = np.asarray(lab[:, :, 1], dtype=np.uint8)
    b_chan = np.asarray(lab[:, :, 2], dtype=np.uint8)

    l_work = np.array(l_chan, copy=True)
    a_work = np.array(a_chan, copy=True)
    b_work = np.array(b_chan, copy=True)
    l_fill = int(np.median(l_chan[plate_bool]))
    a_fill = int(np.median(a_chan[plate_bool]))
    b_fill = int(np.median(b_chan[plate_bool]))
    l_work[~plate_bool] = np.uint8(l_fill)
    a_work[~plate_bool] = np.uint8(a_fill)
    b_work[~plate_bool] = np.uint8(b_fill)

    feature_kernel = max(5, int(round(min(h, w) * 0.018)))
    if feature_kernel % 2 == 0:
        feature_kernel += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feature_kernel, feature_kernel))

    top_hat = cv2.morphologyEx(l_work, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(l_work, cv2.MORPH_BLACKHAT, kernel)

    local_sigma = max(3.0, float(min(h, w)) * 0.012)
    l_bg = cv2.GaussianBlur(l_work, (0, 0), sigmaX=local_sigma, sigmaY=local_sigma)
    a_bg = cv2.GaussianBlur(a_work, (0, 0), sigmaX=local_sigma, sigmaY=local_sigma)
    b_bg = cv2.GaussianBlur(b_work, (0, 0), sigmaX=local_sigma, sigmaY=local_sigma)
    l_contrast = cv2.absdiff(l_work, l_bg)
    chroma = cv2.add(cv2.absdiff(a_work, a_bg), cv2.absdiff(b_work, b_bg))

    dog_small_sigma = max(1.0, float(min(h, w)) * 0.004)
    dog_large_sigma = max(dog_small_sigma + 2.0, float(min(h, w)) * 0.018)
    dog_small = cv2.GaussianBlur(l_work, (0, 0), sigmaX=dog_small_sigma, sigmaY=dog_small_sigma)
    dog_large = cv2.GaussianBlur(l_work, (0, 0), sigmaX=dog_large_sigma, sigmaY=dog_large_sigma)
    dog = cv2.absdiff(dog_small, dog_large)

    response = np.maximum.reduce(
        [
            np.asarray(top_hat, dtype=np.uint8),
            np.asarray(black_hat, dtype=np.uint8),
            np.asarray(l_contrast, dtype=np.uint8),
            np.asarray(dog, dtype=np.uint8),
            np.asarray(np.clip(chroma * 0.6, 0, 255), dtype=np.uint8),
        ]
    )
    response = cv2.GaussianBlur(response, (0, 0), sigmaX=1.2, sigmaY=1.2)
    out = np.zeros_like(response, dtype=np.uint8)
    out[plate_bool] = response[plate_bool]
    return out


def _detect_colony_candidate_boxes(
    image_rgb: np.ndarray,
    plate_circle: tuple[int, int, int],
    box_prompt: tuple[int, int, int, int] | None = None,
) -> tuple[list[dict[str, object]], np.ndarray, dict[str, object]]:
    image = _as_rgb(image_rgb)
    h, w = image.shape[:2]
    plate_mask = _circle_mask((h, w), plate_circle, inset=max(6.0, float(plate_circle[2]) * 0.05))
    if box_prompt is not None:
        x0, y0, bw, bh = _clamp_box(box_prompt, (h, w)) or (0, 0, w, h)
        roi_mask = np.zeros((h, w), dtype=bool)
        roi_mask[y0 : y0 + bh, x0 : x0 + bw] = True
        plate_mask = np.logical_and(plate_mask, roi_mask)

    response = _colony_candidate_response(image, plate_mask)
    plate_values = response[np.asarray(plate_mask, dtype=bool)]
    if plate_values.size <= 0:
        return [], np.zeros((h, w), dtype=bool), {"candidate_strategy": "empty_plate_mask"}

    otsu_src = np.asarray(plate_values, dtype=np.uint8).reshape(-1, 1)
    otsu_thr, _otsu_mask = cv2.threshold(otsu_src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    perc_thr = float(np.percentile(plate_values, 95.0))
    threshold_value = max(12.0, float(otsu_thr), perc_thr)

    binary = np.zeros((h, w), dtype=np.uint8)
    binary[np.logical_and(plate_mask, response >= threshold_value)] = 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)

    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    plate_area = float(max(1, int(np.count_nonzero(plate_mask))))
    min_area = max(10, int(round(plate_area * 0.00002)))
    max_area = max(min_area + 1, int(round(plate_area * 0.02)))

    candidates: list[dict[str, object]] = []
    candidate_mask = np.zeros((h, w), dtype=bool)
    for lid in range(1, int(n_labels)):
        x, y, bw, bh, area = [int(v) for v in stats[lid].tolist()]
        if area < min_area or area > max_area:
            continue
        component = labels == lid
        component_pixels = int(np.count_nonzero(component))
        if component_pixels <= 0:
            continue
        mean_response = float(np.mean(response[component]))
        extent = float(component_pixels) / float(max(1, bw * bh))
        contours, _hier = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circularity = 0.0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(contour, closed=True))
            area_f = float(cv2.contourArea(contour))
            if perimeter > 0.0 and area_f > 0.0:
                circularity = (4.0 * np.pi * area_f) / max(1.0, perimeter * perimeter)
        if mean_response < threshold_value and extent < 0.12:
            continue
        if circularity < 0.08 and extent < 0.18:
            continue
        candidate_mask |= component
        pad = max(4, int(round(max(bw, bh) * 0.22)))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        local_mask = component[y1:y2, x1:x2]
        candidates.append(
            {
                "box": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                "local_mask": np.asarray(local_mask, dtype=np.uint8),
                "area": int(component_pixels),
                "mean_response": float(mean_response),
                "extent": float(extent),
            }
        )

    candidates.sort(key=lambda row: (float(row["mean_response"]), float(row["area"])), reverse=True)
    return candidates, candidate_mask, {"candidate_strategy": "response_threshold", "threshold": float(threshold_value)}


def detect_colonies_round_petri_dish(
    image_rgb: np.ndarray,
    *,
    box_prompt: tuple[int, int, int, int] | None = None,
    config: FoundationSegmentationConfig | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if cancel_check is not None and bool(cancel_check()):
        raise FoundationSegmentationCancelled("Foundation batch cancelled.")
    image = _as_rgb(image_rgb)
    h, w = image.shape[:2]
    cfg = config if config is not None else FoundationSegmentationConfig()

    plate_circle, plate_meta = _detect_round_plate_circle(image, box_prompt=box_prompt)
    plate_mask = _circle_mask((h, w), plate_circle, inset=max(3.0, float(plate_circle[2]) * 0.02))
    candidates, candidate_mask, candidate_meta = _detect_colony_candidate_boxes(
        image,
        plate_circle=plate_circle,
        box_prompt=box_prompt,
    )

    enhanced = _enhance_for_edges(image, max(0.4, float(cfg.sigma)))
    final_mask = np.zeros((h, w), dtype=bool)
    refined_components = 0
    for row in candidates[:256]:
        if cancel_check is not None and bool(cancel_check()):
            raise FoundationSegmentationCancelled("Foundation batch cancelled.")
        box = tuple(int(v) for v in row["box"])
        x, y, bw, bh = box
        component = np.zeros((h, w), dtype=bool)
        component[y : y + bh, x : x + bw] = np.asarray(row.get("local_mask"), dtype=bool)
        pos_seed = cv2.erode(component.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
        if int(np.count_nonzero(pos_seed)) <= 0:
            pos_seed = component.copy()
        neg_seed = np.zeros((h, w), dtype=bool)
        dilated = cv2.dilate(component.astype(np.uint8), np.ones((7, 7), dtype=np.uint8), iterations=1) > 0
        neg_seed[np.logical_and(dilated, ~component)] = True
        neg_seed[~plate_mask] = True
        refined = _grabcut_from_prompts(
            enhanced,
            positive=pos_seed.astype(np.uint8),
            negative=neg_seed.astype(np.uint8),
            box=box,
            prior_mask=component.astype(np.uint8),
            iters=max(1, int(cfg.grabcut_iters)),
        )
        refined = np.logical_and(refined, plate_mask)
        if int(np.count_nonzero(refined)) <= 0:
            refined = component
        final_mask |= refined
        refined_components += 1

    if int(np.count_nonzero(final_mask)) <= 0:
        final_mask = np.logical_and(candidate_mask, plate_mask)

    final_mask = _post_smooth(final_mask, max(1, int(cfg.smooth_kernel)))
    final_mask = np.logical_and(final_mask, plate_mask)
    meta = {
        "backend": "colony_sam2",
        "engine": "paper_inspired_detector_then_segment",
        "plate_circle": [int(v) for v in plate_circle],
        "candidate_count": int(len(candidates)),
        "refined_components": int(refined_components),
        "predicted_pixels": int(np.count_nonzero(final_mask)),
    }
    meta.update(plate_meta)
    meta.update(candidate_meta)
    return np.asarray(final_mask, dtype=np.uint8), meta


def _run_backend_colony_sam2(
    image_rgb: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    box: tuple[int, int, int, int] | None,
    prior_mask: np.ndarray | None,
    cfg: FoundationSegmentationConfig,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    pred, meta = detect_colonies_round_petri_dish(
        image_rgb,
        box_prompt=box,
        config=cfg,
        cancel_check=cancel_check,
    )
    out = np.asarray(pred > 0, dtype=bool)
    if prior_mask is not None:
        out |= _as_bool_mask(prior_mask, out.shape[:2])
    out[np.asarray(positive, dtype=bool)] = True
    out[np.asarray(negative, dtype=bool)] = False
    out = np.logical_and(out, _circle_mask(out.shape[:2], tuple(meta.get("plate_circle", [out.shape[1] // 2, out.shape[0] // 2, min(out.shape[:2]) // 2]))))
    out = _post_smooth(out, max(1, int(cfg.smooth_kernel)))
    meta = dict(meta)
    meta.update(
        {
            "iters": int(cfg.grabcut_iters),
            "sigma": float(cfg.sigma),
            "smooth_kernel": int(cfg.smooth_kernel),
        }
    )
    return np.asarray(out, dtype=np.uint8), meta


def _run_backend_plant_health_assist(
    image_rgb: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    box: tuple[int, int, int, int] | None,
    prior_mask: np.ndarray | None,
    cfg: FoundationSegmentationConfig,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if cancel_check is not None and bool(cancel_check()):
        raise FoundationSegmentationCancelled("Foundation batch cancelled.")

    python_exe = _resolve_external_python(cfg)
    resources_root = Path(__file__).resolve().parent
    runner_path = resources_root / "plant_health_runner.py"
    model_path = resources_root / "builtin_models" / "plant_health" / "plantvillage_leaf_health_binary_mobilenet_v1.pt"
    if not runner_path.is_file():
        raise RuntimeError(f"Built-in plant health runner is missing: {runner_path}")
    if not model_path.is_file():
        raise RuntimeError(
            "Built-in PlantVillage health model is missing. Train or bundle "
            f"{model_path.name} first."
        )

    image = _as_rgb(image_rgb)
    h, w = image.shape[:2]
    pos_u8 = (_as_bool_mask(positive, (h, w))).astype(np.uint8) * 255
    neg_u8 = (_as_bool_mask(negative, (h, w))).astype(np.uint8) * 255

    with tempfile.TemporaryDirectory(prefix="npec_plant_health_") as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "image.png"
        pos_path = tmp_path / "positive.png"
        neg_path = tmp_path / "negative.png"
        output_path = tmp_path / "output_mask.png"
        meta_path = tmp_path / "meta.json"

        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(pos_path), pos_u8)
        cv2.imwrite(str(neg_path), neg_u8)

        command = [
            str(python_exe),
            str(runner_path),
            "--image",
            str(image_path),
            "--positive",
            str(pos_path),
            "--negative",
            str(neg_path),
            "--output",
            str(output_path),
            "--meta-out",
            str(meta_path),
            "--model",
            str(model_path),
        ]
        if box is not None:
            command.extend(["--box", str(int(box[0])), str(int(box[1])), str(int(box[2])), str(int(box[3]))])

        timeout_sec = max(10, int(cfg.external_timeout_sec))
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Plant health backend launch failed: {exc}") from exc

        stdout_text = ""
        stderr_text = ""
        start_time = time.monotonic()
        while True:
            if cancel_check is not None and bool(cancel_check()):
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=2)
                    except Exception:
                        pass
                raise FoundationSegmentationCancelled("Foundation batch cancelled.")
            remaining = timeout_sec - (time.monotonic() - start_time)
            if remaining <= 0:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=2)
                    except Exception:
                        pass
                raise RuntimeError(f"Plant health backend timed out after {timeout_sec}s.")
            try:
                stdout_text, stderr_text = proc.communicate(timeout=min(0.20, max(0.05, remaining)))
                break
            except subprocess.TimeoutExpired:
                continue

        if int(proc.returncode) != 0:
            stderr_tail = (stderr_text or "").strip()[-1600:]
            stdout_tail = (stdout_text or "").strip()[-800:]
            message = f"Plant health backend failed, exit={proc.returncode}."
            if stderr_tail:
                message += f"\nStderr:\n{stderr_tail}"
            if stdout_tail:
                message += f"\nStdout:\n{stdout_tail}"
            raise RuntimeError(message)

        pred = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        if pred is None:
            raise RuntimeError(f"Plant health backend produced no output mask: {output_path}")
        if pred.ndim == 3:
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        out = np.asarray(pred > 0, dtype=bool)
        if out.shape[:2] != (h, w):
            out = cv2.resize(out.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

        if prior_mask is not None:
            out |= _as_bool_mask(prior_mask, (h, w))
        out[np.asarray(positive, dtype=bool)] = True
        out[np.asarray(negative, dtype=bool)] = False
        if box is not None:
            x, y, bw, bh = box
            scoped = np.zeros((h, w), dtype=bool)
            scoped[y : y + bh, x : x + bw] = out[y : y + bh, x : x + bw]
            out = scoped

        meta: dict[str, object] = {}
        if meta_path.exists():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta = loaded
            except Exception:
                meta = {}
        meta.update(
            {
                "backend": "plant_health_assist",
                "engine": str(meta.get("engine") or "plantvillage-binary-mobilenet"),
                "runner": str(runner_path),
                "python": str(python_exe),
                "model": str(model_path),
            }
        )
        return np.asarray(out, dtype=np.uint8), meta


def _run_backend_sam2(
    image_rgb: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    box: tuple[int, int, int, int] | None,
    prior_mask: np.ndarray | None,
    cfg: FoundationSegmentationConfig,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if cancel_check is not None and bool(cancel_check()):
        raise FoundationSegmentationCancelled("Foundation batch cancelled.")
    enhanced = _enhance_for_edges(image_rgb, cfg.sigma)
    pred = _grabcut_from_prompts(
        enhanced,
        positive=positive,
        negative=negative,
        box=box,
        prior_mask=prior_mask,
        iters=cfg.grabcut_iters,
    )
    prompt_count = int(np.count_nonzero(positive))
    if prompt_count > 0:
        min_expected = max(8, int(prompt_count * 3))
        if int(np.count_nonzero(pred)) < min_expected:
            fallback = _seed_intensity_fallback(enhanced, positive, negative, box)
            if int(np.count_nonzero(fallback)) > int(np.count_nonzero(pred)):
                pred = fallback
    pred = _post_smooth(pred, cfg.smooth_kernel)
    pred[np.asarray(positive, dtype=bool)] = True
    pred[np.asarray(negative, dtype=bool)] = False
    meta = {
        "backend": "sam2",
        "engine": "opencv-grabcut-promptable",
        "iters": int(cfg.grabcut_iters),
        "sigma": float(cfg.sigma),
        "smooth_kernel": int(cfg.smooth_kernel),
    }
    return np.asarray(pred, dtype=np.uint8), meta


def _run_backend_focalclick(
    image_rgb: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    box: tuple[int, int, int, int] | None,
    prior_mask: np.ndarray | None,
    cfg: FoundationSegmentationConfig,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if cancel_check is not None and bool(cancel_check()):
        raise FoundationSegmentationCancelled("Foundation batch cancelled.")
    enhanced = _enhance_for_edges(image_rgb, max(0.4, float(cfg.sigma)))
    pred = _grabcut_from_prompts(
        enhanced,
        positive=positive,
        negative=negative,
        box=box,
        prior_mask=prior_mask,
        iters=max(1, int(cfg.focal_refine_iters)),
    )
    prompt_count = int(np.count_nonzero(positive))
    if prompt_count > 0:
        min_expected = max(8, int(prompt_count * 3))
        if int(np.count_nonzero(pred)) < min_expected:
            fallback = _seed_intensity_fallback(enhanced, positive, negative, box)
            if int(np.count_nonzero(fallback)) > int(np.count_nonzero(pred)):
                pred = fallback
    pred = _post_smooth(pred, max(1, int(cfg.smooth_kernel)))
    pred[np.asarray(positive, dtype=bool)] = True
    pred[np.asarray(negative, dtype=bool)] = False
    meta = {
        "backend": "focalclick",
        "engine": "opencv-grabcut-refine",
        "iters": int(max(1, cfg.focal_refine_iters)),
        "sigma": float(max(0.4, cfg.sigma)),
        "smooth_kernel": int(max(1, cfg.smooth_kernel)),
    }
    return np.asarray(pred, dtype=np.uint8), meta


def _resolve_external_python(config: FoundationSegmentationConfig) -> str:
    candidates: list[str] = []
    if str(config.external_python).strip():
        candidates.append(str(config.external_python).strip())
    for env_name in (
        "NPEC_FOUNDATION_PYTHON",
        "NPEC_EXTERNAL_TF_PYTHON",
        "NPEC_TF_PYTHON",
        "NPEC_EXTERNAL_TORCH_PYTHON",
        "NPEC_PYTORCH_PYTHON",
    ):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            candidates.append(raw)
    for candidate in common_external_python_candidates():
        candidates.append(str(candidate))
    current_python = safe_current_python_executable()
    if current_python is not None:
        candidates.append(str(current_python))
    # Fallback for shell resolution.
    candidates.extend(["python3", "python"])

    for raw in candidates:
        text = str(raw).strip()
        if not text:
            continue
        maybe_path = Path(text).expanduser()
        if maybe_path.exists() and maybe_path.is_file():
            return str(maybe_path.absolute())
        if os.path.sep not in text and "/" not in text:
            return text
    raise RuntimeError("No usable python executable found for external foundation backend.")


def _run_backend_external(
    image_rgb: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    box: tuple[int, int, int, int] | None,
    prior_mask: np.ndarray | None,
    cfg: FoundationSegmentationConfig,
    backend_key: str,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if cancel_check is not None and bool(cancel_check()):
        raise FoundationSegmentationCancelled("Foundation batch cancelled.")
    runner_text = str(cfg.external_runner).strip()
    if not runner_text:
        raise RuntimeError(
            "External backend requires a runner script path. Set it in the Foundation tab (External Runner)."
        )

    runner_path = Path(runner_text).expanduser()
    if not runner_path.exists() or not runner_path.is_file():
        raise RuntimeError(f"External runner not found: {runner_path}")

    python_exe = _resolve_external_python(cfg)

    image = _as_rgb(image_rgb)
    h, w = image.shape[:2]
    pos_u8 = (_as_bool_mask(positive, (h, w))).astype(np.uint8) * 255
    neg_u8 = (_as_bool_mask(negative, (h, w))).astype(np.uint8) * 255
    prior_u8 = (_as_bool_mask(prior_mask, (h, w))).astype(np.uint8) * 255 if prior_mask is not None else None

    with tempfile.TemporaryDirectory(prefix="npec_foundation_") as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "image.png"
        pos_path = tmp_path / "positive.png"
        neg_path = tmp_path / "negative.png"
        prior_path = tmp_path / "prior.png"
        output_path = tmp_path / "output_mask.png"
        meta_path = tmp_path / "meta.json"
        config_path = tmp_path / "config.json"

        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(pos_path), pos_u8)
        cv2.imwrite(str(neg_path), neg_u8)
        if prior_u8 is not None:
            cv2.imwrite(str(prior_path), prior_u8)

        payload = {
            "backend": str(backend_key),
            "sigma": float(cfg.sigma),
            "grabcut_iters": int(cfg.grabcut_iters),
            "focal_refine_iters": int(cfg.focal_refine_iters),
            "smooth_kernel": int(cfg.smooth_kernel),
            "box_prompt": [int(v) for v in box] if box is not None else None,
            "text_prompt": str(cfg.text_prompt),
            "checkpoint": str(cfg.external_checkpoint).strip() or None,
        }
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        command = [
            str(python_exe),
            str(runner_path),
            "--backend",
            str(backend_key),
            "--image",
            str(image_path),
            "--positive",
            str(pos_path),
            "--negative",
            str(neg_path),
            "--output",
            str(output_path),
            "--meta-out",
            str(meta_path),
            "--config",
            str(config_path),
        ]
        if prior_u8 is not None:
            command.extend(["--prior", str(prior_path)])
        if box is not None:
            command.extend(["--box", str(int(box[0])), str(int(box[1])), str(int(box[2])), str(int(box[3]))])
        checkpoint = str(cfg.external_checkpoint).strip()
        if checkpoint:
            command.extend(["--checkpoint", checkpoint])
        extra_args = str(cfg.external_extra_args).strip()
        if extra_args:
            command.extend(shlex.split(extra_args))

        timeout_sec = max(10, int(cfg.external_timeout_sec))
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            raise RuntimeError(f"External backend launch failed ({backend_key}): {exc}") from exc

        stdout_text = ""
        stderr_text = ""
        start_time = time.monotonic()
        while True:
            if cancel_check is not None and bool(cancel_check()):
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=2)
                    except Exception:
                        pass
                raise FoundationSegmentationCancelled("Foundation batch cancelled.")

            remaining = timeout_sec - (time.monotonic() - start_time)
            if remaining <= 0:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=2)
                    except Exception:
                        pass
                raise RuntimeError(f"External backend timed out after {timeout_sec}s ({backend_key}).")

            try:
                stdout_text, stderr_text = proc.communicate(timeout=min(0.20, max(0.05, remaining)))
                break
            except subprocess.TimeoutExpired:
                continue

        if int(proc.returncode) != 0:
            stderr_tail = (stderr_text or "").strip()[-1600:]
            stdout_tail = (stdout_text or "").strip()[-800:]
            message = f"External backend failed ({backend_key}), exit={proc.returncode}."
            if stderr_tail:
                message += f"\nStderr:\n{stderr_tail}"
            if stdout_tail:
                message += f"\nStdout:\n{stdout_tail}"
            raise RuntimeError(message)

        pred = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        if pred is None:
            raise RuntimeError(f"External backend produced no output mask: {output_path}")
        if pred.ndim == 3:
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        out = np.asarray(pred > 0, dtype=bool)
        if out.shape[:2] != (h, w):
            out = cv2.resize(out.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

        if box is not None:
            x, y, bw, bh = box
            scoped = np.zeros((h, w), dtype=bool)
            scoped[y : y + bh, x : x + bw] = out[y : y + bh, x : x + bw]
            out = scoped
        out[np.asarray(positive, dtype=bool)] = True
        out[np.asarray(negative, dtype=bool)] = False

        meta: dict[str, object] = {}
        if meta_path.exists():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta = loaded
            except Exception:
                meta = {}
        meta.update(
            {
                "backend": str(backend_key),
                "engine": "external-runner",
                "runner": str(runner_path),
                "python": str(python_exe),
            }
        )
        return np.asarray(out, dtype=np.uint8), meta


_BACKENDS: dict[str, Callable[..., tuple[np.ndarray, dict[str, object]]]] = {
    "plant_health_assist": _run_backend_plant_health_assist,
    "colony_sam2": _run_backend_colony_sam2,
    "sam2": _run_backend_sam2,
    "focalclick": _run_backend_focalclick,
}


def run_foundation_segmentation(
    image_rgb: np.ndarray,
    *,
    backend: str,
    positive_prompt: np.ndarray | None,
    negative_prompt: np.ndarray | None,
    box_prompt: tuple[int, int, int, int] | None,
    prior_mask: np.ndarray | None,
    config: FoundationSegmentationConfig | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if cancel_check is not None and bool(cancel_check()):
        raise FoundationSegmentationCancelled("Foundation batch cancelled.")
    image = _as_rgb(image_rgb)
    h, w = image.shape[:2]
    key = str(backend).strip().lower()
    positive = _as_bool_mask(positive_prompt, (h, w))
    negative = _as_bool_mask(negative_prompt, (h, w))
    prior = _as_bool_mask(prior_mask, (h, w)) if prior_mask is not None else None
    box = _clamp_box(box_prompt, (h, w))

    if (
        key not in _AUTO_PROMPTLESS_BACKENDS
        and int(np.count_nonzero(positive)) <= 0
        and int(np.count_nonzero(negative)) <= 0
        and box is None
        and prior is None
    ):
        raise ValueError("Provide at least one prompt (positive/negative/box) or prior mask.")

    cfg = config if config is not None else FoundationSegmentationConfig()
    if key in _BACKENDS:
        runner = _BACKENDS[key]
        pred, meta = runner(
            image,
            positive=positive,
            negative=negative,
            box=box,
            prior_mask=prior,
            cfg=cfg,
            cancel_check=cancel_check,
        )
    elif is_external_foundation_backend(key):
        pred, meta = _run_backend_external(
            image,
            positive=positive,
            negative=negative,
            box=box,
            prior_mask=prior,
            cfg=cfg,
            backend_key=key,
            cancel_check=cancel_check,
        )
    else:
        supported = ", ".join(sorted(list(_BACKENDS.keys()) + ["xsam_external", "sam2_external", "focalclick_external"]))
        raise ValueError(f"Unsupported foundation backend '{backend}'. Supported: {supported}")
    out = (np.asarray(pred) > 0).astype(np.uint8)
    meta = dict(meta)
    if cancel_check is not None and bool(cancel_check()):
        raise FoundationSegmentationCancelled("Foundation batch cancelled.")
    meta.update(
        {
            "backend_requested": str(key),
            "prompt_positive_px": int(np.count_nonzero(positive)),
            "prompt_negative_px": int(np.count_nonzero(negative)),
            "box_prompt": [int(v) for v in box] if box is not None else None,
            "used_prior": bool(prior is not None),
            "predicted_pixels": int(np.count_nonzero(out)),
            "predicted_pct": (float(np.count_nonzero(out)) / float(max(1, out.size))) * 100.0,
            "text_prompt": str(cfg.text_prompt).strip(),
        }
    )
    return out, meta


def build_rosette_prior_prompts(
    image_rgb: np.ndarray,
    box_prompt: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    image = _as_rgb(image_rgb)
    h, w = image.shape[:2]
    box = _clamp_box(box_prompt, (h, w))

    rgb_f = image.astype(np.float32)
    vegetation = (1.45 * rgb_f[:, :, 1]) - (0.70 * rgb_f[:, :, 0]) - (0.55 * rgb_f[:, :, 2])
    vegetation = cv2.normalize(vegetation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    yy, xx = np.indices((h, w), dtype=np.float32)
    cx = (float(w) - 1.0) * 0.5
    cy = (float(h) - 1.0) * 0.5
    nx = (xx - cx) / float(max(1.0, w * 0.5))
    ny = (yy - cy) / float(max(1.0, h * 0.5))
    center_bias = np.exp(-(nx * nx + ny * ny) / 0.85)
    center_u8 = np.clip(center_bias * 255.0, 0, 255).astype(np.uint8)
    score = cv2.addWeighted(vegetation, 0.7, center_u8, 0.3, 0.0)

    if box is not None:
        x, y, bw, bh = box
        scoped = np.zeros((h, w), dtype=np.uint8)
        scoped[y : y + bh, x : x + bw] = score[y : y + bh, x : x + bw]
        score = scoped

    _, binary = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.medianBlur(binary, 5)
    k = max(3, int(round(min(h, w) * 0.010)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    chosen = np.zeros((h, w), dtype=np.uint8)
    if int(n_labels) > 1:
        best_label = 1
        best_score = -1e9
        for lid in range(1, int(n_labels)):
            x, y, bw, bh, area = stats[lid].tolist()
            if int(area) <= 0:
                continue
            ccx = x + bw * 0.5
            ccy = y + bh * 0.5
            dist = float(np.hypot(ccx - cx, ccy - cy))
            quality = float(area) - (dist * 0.45)
            if quality > best_score:
                best_score = quality
                best_label = int(lid)
        chosen[labels == best_label] = 255
    else:
        chosen = binary.copy()

    erode_k = max(1, int(round(min(h, w) * 0.012)))
    positive = cv2.erode((chosen > 0).astype(np.uint8), np.ones((erode_k, erode_k), dtype=np.uint8), iterations=1)
    if int(np.count_nonzero(positive)) <= 0:
        positive = np.zeros((h, w), dtype=np.uint8)
        radius = max(3, int(round(min(h, w) * 0.05)))
        cv2.circle(positive, (int(round(cx)), int(round(cy))), radius, color=1, thickness=-1, lineType=cv2.LINE_AA)

    dilate_k = max(3, int(round(min(h, w) * 0.03)))
    if dilate_k % 2 == 0:
        dilate_k += 1
    dilated = cv2.dilate((chosen > 0).astype(np.uint8), np.ones((dilate_k, dilate_k), dtype=np.uint8), iterations=1) > 0
    negative = np.zeros((h, w), dtype=np.uint8)
    margin = max(5, int(round(min(h, w) * 0.08)))
    negative[:margin, :] = 1
    negative[-margin:, :] = 1
    negative[:, :margin] = 1
    negative[:, -margin:] = 1
    negative[dilated] = 0
    negative[(positive > 0)] = 0
    if box is not None:
        x, y, bw, bh = box
        scoped_neg = np.zeros((h, w), dtype=np.uint8)
        scoped_neg[y : y + bh, x : x + bw] = negative[y : y + bh, x : x + bw]
        negative = scoped_neg

    meta = {
        "strategy": "one_rosette_style_prior",
        "positive_seed_px": int(np.count_nonzero(positive)),
        "negative_seed_px": int(np.count_nonzero(negative)),
    }
    return positive.astype(np.uint8), negative.astype(np.uint8), meta


def suggest_next_click(
    image_rgb: np.ndarray,
    *,
    current_mask: np.ndarray | None,
    positive_prompt: np.ndarray | None,
    negative_prompt: np.ndarray | None,
    box_prompt: tuple[int, int, int, int] | None,
) -> dict[str, object]:
    image = _as_rgb(image_rgb)
    h, w = image.shape[:2]
    current = _as_bool_mask(current_mask, (h, w)) if current_mask is not None else np.zeros((h, w), dtype=bool)
    pos = _as_bool_mask(positive_prompt, (h, w))
    neg = _as_bool_mask(negative_prompt, (h, w))
    box = _clamp_box(box_prompt, (h, w))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    gmax = float(np.max(grad))
    if gmax > 1e-6:
        grad = grad / gmax

    scores = np.asarray(grad, dtype=np.float32)

    if np.any(current):
        boundary = cv2.morphologyEx(current.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
        boundary_wide = cv2.dilate(boundary, np.ones((11, 11), dtype=np.uint8), iterations=1) > 0
        scores = scores * (0.35 + 0.65 * boundary_wide.astype(np.float32))

    if box is not None:
        x, y, bw, bh = box
        in_box = np.zeros((h, w), dtype=bool)
        in_box[y : y + bh, x : x + bw] = True
        scores[~in_box] = 0.0

    occupied = np.logical_or(pos, neg).astype(np.uint8)
    if np.any(occupied):
        exclusion = cv2.dilate(occupied, np.ones((13, 13), dtype=np.uint8), iterations=1) > 0
        scores[exclusion] = 0.0

    flat_idx = int(np.argmax(scores))
    best_score = float(scores.reshape(-1)[flat_idx]) if scores.size > 0 else 0.0
    if best_score <= 1e-6:
        if box is not None:
            x, y, bw, bh = box
            px = int(x + bw // 2)
            py = int(y + bh // 2)
        else:
            px = int(w // 2)
            py = int(h // 2)
        suggestion_label = 1 if not bool(current[py, px]) else -1
        return {"x": px, "y": py, "label": suggestion_label, "score": 0.0}

    py, px = np.unravel_index(flat_idx, scores.shape)
    suggestion_label = 1 if not bool(current[py, px]) else -1
    return {"x": int(px), "y": int(py), "label": int(suggestion_label), "score": float(best_score)}
