from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(slots=True)
class TileLayout:
    original_shape_hw: tuple[int, int]
    padded_shape_hw: tuple[int, int]
    patch_size: int
    top_pad: int
    bottom_pad: int
    left_pad: int
    right_pad: int


def load_uint8_image(image_path: str | Path, mode: str = "gray") -> np.ndarray:
    path = Path(image_path)
    pil = Image.open(path)
    if mode == "gray":
        return np.asarray(pil.convert("L"), dtype=np.uint8)
    if mode == "rgb":
        return np.asarray(pil.convert("RGB"), dtype=np.uint8)
    if mode == "bgr":
        return np.asarray(pil.convert("RGB"), dtype=np.uint8)[:, :, ::-1]
    raise ValueError(f"Unsupported image mode: {mode}")


def pad_image_to_patch_multiple(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, TileLayout]:
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"Unsupported image rank: {array.ndim}")
    height, width = array.shape[:2]
    height_pad = ((height + patch_size - 1) // patch_size) * patch_size - height
    width_pad = ((width + patch_size - 1) // patch_size) * patch_size - width
    top_pad = height_pad // 2
    bottom_pad = height_pad - top_pad
    left_pad = width_pad // 2
    right_pad = width_pad - left_pad
    if array.ndim == 2:
        padded = np.pad(array, ((top_pad, bottom_pad), (left_pad, right_pad)), mode="edge")
    else:
        padded = np.pad(array, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode="edge")
    layout = TileLayout(
        original_shape_hw=(height, width),
        padded_shape_hw=padded.shape[:2],
        patch_size=int(patch_size),
        top_pad=int(top_pad),
        bottom_pad=int(bottom_pad),
        left_pad=int(left_pad),
        right_pad=int(right_pad),
    )
    return padded, layout


def remove_padding(image: np.ndarray, layout: TileLayout) -> np.ndarray:
    array = np.asarray(image)
    height, width = array.shape[:2]
    y0 = int(layout.top_pad)
    y1 = height - int(layout.bottom_pad) if layout.bottom_pad else height
    x0 = int(layout.left_pad)
    x1 = width - int(layout.right_pad) if layout.right_pad else width
    return array[y0:y1, x0:x1]


def collect_tiles(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, list[tuple[int, int]], TileLayout]:
    padded, layout = pad_image_to_patch_multiple(image, patch_size)
    coords: list[tuple[int, int]] = []
    tiles: list[np.ndarray] = []
    padded_height, padded_width = padded.shape[:2]
    for y in range(0, padded_height, patch_size):
        for x in range(0, padded_width, patch_size):
            coords.append((int(y), int(x)))
            tiles.append(padded[y : y + patch_size, x : x + patch_size])
    return np.asarray(tiles), coords, layout


def stitch_predictions(
    predictions: np.ndarray,
    coords: list[tuple[int, int]],
    layout: TileLayout,
) -> np.ndarray:
    preds = np.asarray(predictions)
    if preds.ndim not in (3, 4):
        raise ValueError(f"Unsupported prediction rank: {preds.ndim}")
    padded_height, padded_width = layout.padded_shape_hw
    if preds.ndim == 3:
        stitched = np.zeros((padded_height, padded_width), dtype=preds.dtype)
    else:
        stitched = np.zeros((padded_height, padded_width, preds.shape[-1]), dtype=preds.dtype)
    patch_size = int(layout.patch_size)
    for patch, (y, x) in zip(preds, coords, strict=False):
        stitched[y : y + patch_size, x : x + patch_size] = patch
    return remove_padding(stitched, layout)


def filter_connected_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if min_area <= 1 or np.count_nonzero(binary) == 0:
        return binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for label_id in range(1, int(count)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            out[labels == label_id] = 1
    return out


def fill_enclosed_holes(mask: np.ndarray, max_hole_area: int | None = None) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return binary

    inverse = (binary == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    if count <= 1:
        return binary

    border_labels = set(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[-1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, -1]).tolist())

    filled = binary.copy()
    for label_id in range(1, int(count)):
        if label_id in border_labels:
            continue
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if max_hole_area is not None and area > int(max_hole_area):
            continue
        filled[labels == label_id] = 1
    return filled


def remove_noise(
    mask: np.ndarray,
    min_area: int = 50,
    apply_closing: bool = True,
    closing_iterations: int = 2,
    fill_holes: bool = True,
    max_hole_area: int | None = None,
) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return binary

    kernel_open = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    cleaned = filter_connected_components(cleaned, min_area=int(min_area))

    if apply_closing and int(closing_iterations) > 0:
        kernel_close = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=int(closing_iterations))

    if fill_holes:
        cleaned = fill_enclosed_holes(cleaned, max_hole_area=max_hole_area)
    return cleaned.astype(np.uint8)


def _largest_centered_component(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return binary
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape[:2]
    center_x = width * 0.5
    center_y = height * 0.5
    best_label = 0
    best_score = -1e18
    for label_id in range(1, int(count)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_id]
        score = float(area) - (0.75 * abs(float(cx) - center_x)) - (0.35 * abs(float(cy) - center_y))
        if x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height:
            score -= float(max(width, height))
        if score > best_score:
            best_score = score
            best_label = int(label_id)
    out = np.zeros_like(binary)
    if best_label > 0:
        out[labels == best_label] = 1
    return out


def estimate_dish_interior_mask(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(image, dtype=np.uint8)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 5.0)
    _thr, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    interior = _largest_centered_component(thresh > 0)
    if np.count_nonzero(interior) == 0:
        interior = _largest_centered_component(gray > 40)
    if np.count_nonzero(interior) == 0:
        return np.ones(gray.shape[:2], dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    interior = cv2.morphologyEx(interior.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
    interior = cv2.morphologyEx(interior.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    interior = cv2.dilate(interior.astype(np.uint8), kernel, iterations=1)
    return (interior > 0).astype(np.uint8)


def adaptive_side_margin(width: int, fraction: float = 0.08, max_px: int = 320) -> int:
    return max(0, min(int(round(float(width) * float(fraction))), int(max_px)))
