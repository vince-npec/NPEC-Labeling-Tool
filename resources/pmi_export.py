from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PMI_CORE_COLUMNS = ["round", "plate", "plant_id", "parameter", "value"]

PMI_TRAIT_COLUMN_MAP: tuple[tuple[str, str], ...] = (
    ("pixel_size_mm", "pixel_size_mm"),
    ("root_length_px", "root_length_px"),
    ("root_length_px_clean", "root_length_px_clean"),
    ("root_length_mm_raw", "root_length_mm_raw"),
    ("root_length_mm_clean", "root_length_mm_clean"),
    ("root_area_px", "root_area_px"),
    ("root_area_mm2", "root_area_mm2"),
    ("primary_root_length_px", "primary_root_length_px"),
    ("primary_root_length_px_clean", "primary_root_length_px_clean"),
    ("primary_root_length_mm_raw", "primary_root_length_mm_raw"),
    ("primary_root_length_mm_clean", "primary_root_length_mm_clean"),
    ("primary_root_area_px", "primary_root_area_px"),
    ("primary_root_area_mm2", "primary_root_area_mm2"),
    ("total_root_length_px", "total_root_length_px"),
    ("total_root_length_px_clean", "total_root_length_px_clean"),
    ("total_root_length_mm_raw", "total_root_length_mm_raw"),
    ("total_root_length_mm_clean", "total_root_length_mm_clean"),
    ("total_root_area_px", "total_root_area_px"),
    ("total_root_area_mm2", "total_root_area_mm2"),
    ("root_perimeter_px", "root_perimeter_px"),
    ("root_perimeter_mm", "root_perimeter_mm"),
    ("tips_count", "tips"),
    ("branches_count", "branches"),
    ("tip_count_raw", "tip_count_raw"),
    ("base_tip_angle_deg", "base_tip_angle_deg"),
    ("emergence_angle_deg", "emergence_angle_deg"),
    ("convex_hull_area_px2", "convex_hull_area_px2"),
    ("convex_hull_area_mm2", "convex_hull_area_mm2"),
    ("aspect_ratio", "aspect_ratio"),
    ("lateral_count", "lateral_count"),
    ("lateral_total_length_px", "lateral_total_length_px"),
    ("lateral_total_length_mm", "lateral_total_length_mm"),
    ("lateral_mean_length_px", "lateral_mean_length_px"),
    ("lateral_mean_length_mm", "lateral_mean_length_mm"),
    ("lateral_mean_diameter_px", "lateral_mean_diameter_px"),
    ("lateral_mean_diameter_mm", "lateral_mean_diameter_mm"),
    ("lateral_max_length_px", "lateral_max_length_px"),
    ("lateral_max_length_mm", "lateral_max_length_mm"),
    ("shoot_area_px", "shoot_area_px"),
    ("shoot_area_mm2", "shoot_area_mm2"),
    ("combined_root_length_px", "combined_root_length_px"),
    ("combined_root_length_mm", "combined_root_length_mm"),
)

PMI_FLUORESCENCE_PLACEHOLDER_PARAMETERS: tuple[str, ...] = (
    "mean_fluorescence_main_root",
    "mean_fluorescence_lateral_root",
    "mean_fluorescence_main_root_tip",
    "mean_fluorescence_other_tip",
    "mean_fluorescence_node",
    "mean_fluorescence_dilated_root_exclusive",
    "mean_fluorescence_dilated_root",
    "mean_fluorescence_shoot",
    "n_pixels_main_root",
    "n_pixels_lateral_root",
    "n_pixels_main_root_tip",
    "n_pixels_other_tip",
    "n_pixels_node",
    "n_pixels_dilated_root_exclusive",
    "n_pixels_dilated_root",
    "n_pixels_shoot",
    "sum_fluorescence_main_root",
    "sum_fluorescence_lateral_root",
    "sum_fluorescence_main_root_tip",
    "sum_fluorescence_other_tip",
    "sum_fluorescence_node",
    "sum_fluorescence_dilated_root_exclusive",
    "sum_fluorescence_dilated_root",
    "sum_fluorescence_shoot",
    "mean_fluorescence_unknown",
    "n_pixels_unknown",
    "sum_fluorescence_unknown",
)

PMI_MASK_DERIVED_PARAMETERS: tuple[str, ...] = (
    "root_area_px",
    "root_area_mm2",
    "root_length_px",
    "root_length_mm_raw",
    "root_length_px_clean",
    "root_length_mm_clean",
    "root_perimeter_px",
    "root_perimeter_mm",
    "primary_root_length_px",
    "primary_root_length_mm_raw",
    "primary_root_length_px_clean",
    "primary_root_length_mm_clean",
    "primary_root_area_px",
    "primary_root_area_mm2",
    "total_root_length_px",
    "total_root_length_mm_raw",
    "total_root_length_px_clean",
    "total_root_length_mm_clean",
    "total_root_area_px",
    "total_root_area_mm2",
    "shoot_area_px",
    "shoot_area_mm2",
    "tips_count",
    "tip_count_raw",
    "branches_count",
    "convex_hull_area_px2",
    "convex_hull_area_mm2",
    "aspect_ratio",
    "lateral_count",
    "lateral_total_length_px",
    "lateral_total_length_mm",
    "lateral_mean_length_px",
    "lateral_mean_length_mm",
    "lateral_max_length_px",
    "lateral_max_length_mm",
    "n_pixels_main_root",
    "n_pixels_lateral_root",
    "n_pixels_main_root_tip",
    "n_pixels_other_tip",
    "n_pixels_node",
    "n_pixels_dilated_root_exclusive",
    "n_pixels_dilated_root",
    "n_pixels_shoot",
    "n_pixels_unknown",
)

DEFAULT_CLASS_IDS = {
    "seed": 1,
    "shoot": 2,
    "root": 3,
    "lateral": 4,
}

ROOT_DILATION_RADIUS_PX = 8
TIP_ZONE_RADIUS_PX = 3
NODE_ZONE_RADIUS_PX = 5


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _is_valid_ownership_measurement(value: object) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        return bool(value)
    except Exception:
        return False


def _normalize_round_value(value: object, fallback: int) -> int:
    try:
        numeric = float(value)
    except Exception:
        return int(fallback)
    if np.isfinite(numeric):
        return int(round(numeric))
    return int(fallback)


def _normalize_plate_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        if text.isdigit():
            return int(text)
        return text
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if np.isfinite(numeric) and float(numeric).is_integer():
        return int(numeric)
    if np.isfinite(numeric):
        return float(numeric)
    return ""


def _composite_plate_labels(master_df: pd.DataFrame) -> pd.Series:
    petri_values = master_df.get("PetriDish", pd.Series([""] * len(master_df), index=master_df.index))
    series_values = master_df.get("Series", pd.Series([""] * len(master_df), index=master_df.index))
    image_values = master_df.get("image_name", pd.Series([""] * len(master_df), index=master_df.index))
    frame_values = pd.to_numeric(master_df.get("FrameIndex"), errors="coerce")
    plant_values = master_df.get("plant_id", pd.Series([""] * len(master_df), index=master_df.index)).astype(str)

    use_composite = False
    if "Series" in master_df.columns and master_df["Series"].nunique(dropna=True) > 1:
        dedupe_probe = pd.DataFrame(
            {
                "FrameIndex": frame_values,
                "PetriDish": petri_values.astype(str),
                "plant_id": plant_values,
            },
            index=master_df.index,
        )
        use_composite = bool(dedupe_probe.duplicated(subset=["FrameIndex", "PetriDish", "plant_id"], keep=False).any())
    if not use_composite:
        resolved: list[object] = []
        for idx in master_df.index:
            petri_raw = petri_values.get(idx, "")
            petri_text = "" if _is_missing(petri_raw) else str(petri_raw).strip()
            if petri_text.lower() == "unknown":
                petri_text = ""
            if petri_text:
                resolved.append(petri_raw)
                continue
            image_text = str(image_values.get(idx, "")).strip()
            if image_text:
                resolved.append(image_text)
                continue
            series_text = str(series_values.get(idx, "")).strip()
            if series_text:
                resolved.append(series_text)
            else:
                resolved.append("Unknown")
        return pd.Series(resolved, index=master_df.index)

    combined = []
    for idx in master_df.index:
        series_text = str(series_values.get(idx, "")).strip()
        petri_text = str(petri_values.get(idx, "")).strip()
        if petri_text.lower() == "unknown":
            petri_text = ""
        image_text = str(image_values.get(idx, "")).strip()
        if series_text and petri_text:
            combined.append(f"{series_text}::{petri_text}")
        elif petri_text:
            combined.append(petri_text)
        elif image_text:
            combined.append(image_text)
        elif series_text:
            combined.append(series_text)
        else:
            combined.append("Unknown")
    return pd.Series(combined, index=master_df.index)


def _numeric_or_default(value: object, default: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    if np.isfinite(numeric):
        return float(numeric)
    return float(default)


def _int_or_default(value: object, default: int) -> int:
    try:
        numeric = float(value)
    except Exception:
        return int(default)
    if np.isfinite(numeric):
        return int(round(numeric))
    return int(default)


def _normalize_mask_path(row: pd.Series) -> Path | None:
    for key in ("OutputMaskPath", "output_mask"):
        raw = row.get(key)
        if _is_missing(raw):
            continue
        path = Path(str(raw)).expanduser()
        if path.exists():
            return path
    return None


def _load_index_mask(mask_path: Path, mask_cache: dict[Path, np.ndarray]) -> np.ndarray | None:
    cached = mask_cache.get(mask_path)
    if cached is not None:
        return cached
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        mask = np.asarray(Image.open(mask_path))
    except Exception:
        return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = np.asarray(mask, dtype=np.uint8)
    mask_cache[mask_path] = mask
    return mask


def _crop_for_row(mask: np.ndarray, row: pd.Series) -> np.ndarray:
    bbox_w = _numeric_or_default(row.get("bbox_w"), 0.0)
    bbox_h = _numeric_or_default(row.get("bbox_h"), 0.0)
    if bbox_w <= 1.0 or bbox_h <= 1.0:
        return mask
    x0 = max(0, _int_or_default(row.get("bbox_x"), 0))
    y0 = max(0, _int_or_default(row.get("bbox_y"), 0))
    x1 = min(mask.shape[1], x0 + max(1, _int_or_default(bbox_w, 0)))
    y1 = min(mask.shape[0], y0 + max(1, _int_or_default(bbox_h, 0)))
    if x1 <= x0 or y1 <= y0:
        return mask
    return mask[y0:y1, x0:x1]


def _ellipse_kernel(radius_px: int) -> np.ndarray:
    size = max(1, radius_px * 2 + 1)
    yy, xx = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return (xx * xx + yy * yy <= radius_px * radius_px).astype(np.uint8)


def _dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0 or mask.size == 0:
        return np.asarray(mask, dtype=bool)
    try:
        from scipy import ndimage
    except Exception:
        return np.asarray(mask, dtype=bool)
    return ndimage.binary_dilation(mask.astype(bool), structure=_ellipse_kernel(radius_px).astype(bool))


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0 or not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    try:
        from skimage.morphology import skeletonize
    except Exception:
        return np.asarray(mask, dtype=bool)
    return skeletonize(mask > 0)


def _neighbor_count(binary: np.ndarray) -> np.ndarray:
    if binary.size == 0:
        return np.zeros_like(binary, dtype=np.uint8)
    try:
        from scipy import ndimage
    except Exception:
        padded = np.pad(binary.astype(np.uint8), 1)
        out = np.zeros_like(binary, dtype=np.uint8)
        for y in range(binary.shape[0]):
            for x in range(binary.shape[1]):
                region = padded[y : y + 3, x : x + 3]
                out[y, x] = max(0, int(region.sum()) - int(padded[y + 1, x + 1]))
        return out
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = ndimage.convolve(binary.astype(np.uint8), kernel, mode="constant", cval=0)
    neighbors = neighbors.astype(np.int16) - binary.astype(np.int16)
    neighbors[neighbors < 0] = 0
    return neighbors.astype(np.uint8)


def _skeleton_length_px(skeleton: np.ndarray) -> float:
    if skeleton.size == 0 or not np.any(skeleton):
        return 0.0
    binary = skeleton.astype(bool)
    horizontal = np.logical_and(binary[:, :-1], binary[:, 1:]).sum(dtype=np.int64)
    vertical = np.logical_and(binary[:-1, :], binary[1:, :]).sum(dtype=np.int64)
    diag_dr = np.logical_and(binary[:-1, :-1], binary[1:, 1:]).sum(dtype=np.int64)
    diag_dl = np.logical_and(binary[:-1, 1:], binary[1:, :-1]).sum(dtype=np.int64)
    return float(horizontal + vertical + math.sqrt(2.0) * (diag_dr + diag_dl))


def _component_pixel_lengths(mask: np.ndarray) -> list[float]:
    if mask.size == 0 or not np.any(mask):
        return []
    try:
        from scipy import ndimage
    except Exception:
        return [_skeleton_length_px(_skeletonize(mask))]
    labels, components = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    lengths: list[float] = []
    for label_id in range(1, int(components) + 1):
        component_mask = labels == label_id
        length_px = _skeleton_length_px(_skeletonize(component_mask))
        if length_px > 0.0:
            lengths.append(length_px)
    return lengths


def _convex_hull_area_px(binary: np.ndarray) -> float:
    if binary.size == 0 or not np.any(binary):
        return 0.0
    try:
        from skimage.morphology import convex_hull_image
    except Exception:
        ys, xs = np.nonzero(binary)
        if xs.size == 0:
            return 0.0
        return float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    return float(np.count_nonzero(convex_hull_image(binary.astype(bool))))


def _perimeter_px(binary: np.ndarray) -> float:
    if binary.size == 0 or not np.any(binary):
        return 0.0
    try:
        from skimage.measure import perimeter
    except Exception:
        return float(np.count_nonzero(binary))
    return float(perimeter(binary.astype(bool), neighborhood=8))


def _derive_mask_metrics_for_row(row: pd.Series, mask_cache: dict[Path, np.ndarray]) -> dict[str, float]:
    mask_path = _normalize_mask_path(row)
    if mask_path is None:
        return {}
    mask_full = _load_index_mask(mask_path, mask_cache)
    if mask_full is None or mask_full.size == 0:
        return {}
    mask = _crop_for_row(mask_full, row)
    if mask.size == 0:
        return {}

    root_id = _int_or_default(row.get("root_class_id"), DEFAULT_CLASS_IDS["root"])
    lateral_id = _int_or_default(row.get("lateral_class_id"), DEFAULT_CLASS_IDS["lateral"])
    shoot_id = _int_or_default(row.get("shoot_class_id"), DEFAULT_CLASS_IDS["shoot"])
    seed_id = _int_or_default(row.get("seed_class_id"), DEFAULT_CLASS_IDS["seed"])
    pixel_size_mm = _numeric_or_default(row.get("pixel_size_mm"), 0.0)
    shoot_source = str(row.get("shoot_measurement_source", "") or "").strip().lower()
    rescue_enabled_raw = row.get("shoot_rgb_rescue_enabled", False)
    rescue_enabled = (
        rescue_enabled_raw is True
        or str(rescue_enabled_raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}
    )
    green_only_shoot_source = shoot_source == "rgb_green_only" or rescue_enabled
    green_only_shoot_px = _numeric_or_default(
        row.get("shoot_rgb_rescue_green_only_px", row.get("shoot_area_px")),
        -1.0,
    )
    green_only_shoot_mm2 = _numeric_or_default(row.get("shoot_area_mm2"), -1.0)

    main_root_mask = mask == np.uint8(root_id)
    lateral_mask = mask == np.uint8(lateral_id)
    shoot_mask = mask == np.uint8(shoot_id)
    if green_only_shoot_source and green_only_shoot_px >= 0.0:
        shoot_mask = np.zeros_like(mask, dtype=bool)
    known_ids = {0, root_id, lateral_id, shoot_id}
    if seed_id >= 0:
        known_ids.add(seed_id)
    unknown_mask = np.logical_and(mask > 0, ~np.isin(mask, list(known_ids)))
    root_binary = np.logical_or(main_root_mask, lateral_mask)
    if not np.any(root_binary) and not np.any(shoot_mask) and not np.any(unknown_mask):
        return {}

    root_dilated = _dilate(root_binary, ROOT_DILATION_RADIUS_PX)
    root_dilated_exclusive = np.logical_and(root_dilated, ~root_binary)

    skeleton = _skeletonize(root_binary)
    skeleton_main = _skeletonize(main_root_mask)
    neighbor_counts = _neighbor_count(skeleton)
    endpoints = np.logical_and(skeleton, neighbor_counts == 1)
    branchpoints = np.logical_and(skeleton, neighbor_counts >= 3)
    main_endpoints = np.logical_and(endpoints, skeleton_main)

    main_tip_seed = np.zeros_like(skeleton, dtype=bool)
    endpoint_coords = np.argwhere(main_endpoints)
    if endpoint_coords.size == 0:
        endpoint_coords = np.argwhere(endpoints)
    if endpoint_coords.size > 0:
        y, x = endpoint_coords[np.argmax(endpoint_coords[:, 0])]
        main_tip_seed[y, x] = True
    other_tip_seed = np.logical_and(endpoints, ~main_tip_seed)

    main_tip_zone = np.logical_and(_dilate(main_tip_seed, TIP_ZONE_RADIUS_PX), root_dilated)
    other_tip_zone = np.logical_and(_dilate(other_tip_seed, NODE_ZONE_RADIUS_PX), root_dilated)
    other_tip_zone = np.logical_and(other_tip_zone, ~main_tip_zone)
    node_zone = np.logical_and(_dilate(branchpoints, NODE_ZONE_RADIUS_PX), root_dilated)
    node_zone = np.logical_and(node_zone, ~main_tip_zone)
    node_zone = np.logical_and(node_zone, ~other_tip_zone)

    root_area_px = float(np.count_nonzero(root_binary))
    main_root_area_px = float(np.count_nonzero(main_root_mask))
    lateral_area_px = float(np.count_nonzero(lateral_mask))
    shoot_area_px = (
        float(green_only_shoot_px)
        if green_only_shoot_source and green_only_shoot_px >= 0.0
        else float(np.count_nonzero(shoot_mask))
    )
    root_length_px = _skeleton_length_px(skeleton)
    main_root_length_px = _skeleton_length_px(skeleton_main)
    lateral_lengths_px = _component_pixel_lengths(lateral_mask)
    lateral_total_length_px = float(sum(lateral_lengths_px))
    total_root_length_px = float(main_root_length_px + lateral_total_length_px)
    lateral_count = int(len(lateral_lengths_px))
    lateral_mean_length_px = float(np.mean(lateral_lengths_px)) if lateral_lengths_px else 0.0
    lateral_max_length_px = float(np.max(lateral_lengths_px)) if lateral_lengths_px else 0.0
    tips_count = int(np.count_nonzero(endpoints))
    branches_count = int(np.count_nonzero(branchpoints))
    convex_hull_area_px2 = _convex_hull_area_px(root_binary)
    root_perimeter_px = _perimeter_px(root_binary)

    ys, xs = np.nonzero(root_binary)
    if xs.size > 0 and ys.size > 0:
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
        aspect_ratio = float(height / max(width, 1.0))
    else:
        aspect_ratio = 0.0

    metrics: dict[str, float] = {
        "n_pixels_main_root": float(np.count_nonzero(main_root_mask)),
        "n_pixels_lateral_root": float(np.count_nonzero(lateral_mask)),
        "n_pixels_main_root_tip": float(np.count_nonzero(main_tip_zone)),
        "n_pixels_other_tip": float(np.count_nonzero(other_tip_zone)),
        "n_pixels_node": float(np.count_nonzero(node_zone)),
        "n_pixels_dilated_root_exclusive": float(np.count_nonzero(root_dilated_exclusive)),
        "n_pixels_dilated_root": float(np.count_nonzero(root_dilated)),
        "n_pixels_shoot": shoot_area_px,
        "n_pixels_unknown": float(np.count_nonzero(unknown_mask)),
        "root_area_px": root_area_px,
        "primary_root_area_px": main_root_area_px,
        "total_root_area_px": float(main_root_area_px + lateral_area_px),
        "shoot_area_px": shoot_area_px,
        "root_length_px": root_length_px,
        "root_length_px_clean": root_length_px,
        "primary_root_length_px": main_root_length_px,
        "primary_root_length_px_clean": main_root_length_px,
        "total_root_length_px": total_root_length_px,
        "total_root_length_px_clean": total_root_length_px,
        "root_perimeter_px": root_perimeter_px,
        "tips_count": float(tips_count),
        "tip_count_raw": float(tips_count),
        "branches_count": float(branches_count),
        "convex_hull_area_px2": convex_hull_area_px2,
        "aspect_ratio": aspect_ratio,
        "lateral_count": float(lateral_count),
        "lateral_total_length_px": lateral_total_length_px,
        "lateral_mean_length_px": lateral_mean_length_px,
        "lateral_max_length_px": lateral_max_length_px,
    }

    if pixel_size_mm > 0.0:
        metrics["pixel_size_mm"] = pixel_size_mm
        metrics["root_area_mm2"] = root_area_px * pixel_size_mm * pixel_size_mm
        metrics["primary_root_area_mm2"] = main_root_area_px * pixel_size_mm * pixel_size_mm
        metrics["total_root_area_mm2"] = (main_root_area_px + lateral_area_px) * pixel_size_mm * pixel_size_mm
        metrics["shoot_area_mm2"] = (
            float(green_only_shoot_mm2)
            if green_only_shoot_source and green_only_shoot_mm2 >= 0.0
            else shoot_area_px * pixel_size_mm * pixel_size_mm
        )
        metrics["root_length_mm_raw"] = root_length_px * pixel_size_mm
        metrics["root_length_mm_clean"] = root_length_px * pixel_size_mm
        metrics["primary_root_length_mm_raw"] = main_root_length_px * pixel_size_mm
        metrics["primary_root_length_mm_clean"] = main_root_length_px * pixel_size_mm
        metrics["total_root_length_mm_raw"] = total_root_length_px * pixel_size_mm
        metrics["total_root_length_mm_clean"] = total_root_length_px * pixel_size_mm
        metrics["root_perimeter_mm"] = root_perimeter_px * pixel_size_mm
        metrics["convex_hull_area_mm2"] = convex_hull_area_px2 * pixel_size_mm * pixel_size_mm
        metrics["lateral_total_length_mm"] = lateral_total_length_px * pixel_size_mm
        metrics["lateral_mean_length_mm"] = lateral_mean_length_px * pixel_size_mm
        metrics["lateral_max_length_mm"] = lateral_max_length_px * pixel_size_mm
    return metrics


def build_pmi_style_rows(
    master_df: pd.DataFrame | None,
    *,
    include_fluorescence_placeholders: bool = True,
    derive_mask_metrics: bool = True,
) -> pd.DataFrame:
    if master_df is None or master_df.empty:
        return pd.DataFrame(columns=PMI_CORE_COLUMNS)

    df = master_df.copy()
    base_group_cols = [col for col in ("Series", "PetriDish") if col in df.columns]
    if base_group_cols:
        fallback_round = df.groupby(base_group_cols, dropna=False).cumcount() + 1
    else:
        fallback_round = pd.Series(np.arange(1, len(df) + 1), index=df.index)
    frame_values = pd.to_numeric(df.get("FrameIndex"), errors="coerce") if "FrameIndex" in df.columns else pd.Series(np.nan, index=df.index)
    plate_values = _composite_plate_labels(df)
    mask_cache: dict[Path, np.ndarray] = {}
    ownership_validity_known = "ownership_measurement_valid" in df.columns

    rows: list[dict[str, object]] = []
    seen_placeholder_keys: set[tuple[int, object, str]] = set()
    available_map = [(parameter, column) for parameter, column in PMI_TRAIT_COLUMN_MAP if column in df.columns]

    for idx, row in df.iterrows():
        round_value = _normalize_round_value(frame_values.get(idx), int(fallback_round.get(idx, idx + 1)))
        plate_value = _normalize_plate_value(plate_values.get(idx))
        plant_id = str(row.get("plant_id", "")).strip()
        if not plant_id:
            plant_id = f"plant_{idx + 1}"
        ownership_measurement_valid = (
            _is_valid_ownership_measurement(row.get("ownership_measurement_valid"))
            if ownership_validity_known
            else True
        )
        derived_map: dict[str, float] | None = None

        def _derived_value(parameter: str) -> object:
            nonlocal derived_map
            if not ownership_measurement_valid or not bool(derive_mask_metrics):
                return None
            if derived_map is None:
                derived_map = _derive_mask_metrics_for_row(row, mask_cache)
            return derived_map.get(parameter, None)

        emitted_parameters: set[str] = set()

        for parameter, column in available_map:
            value = row.get(column) if ownership_measurement_valid else None
            if _is_missing(value):
                value = _derived_value(parameter)
            if _is_missing(value):
                continue
            rows.append(
                {
                    "round": round_value,
                    "plate": plate_value,
                    "plant_id": plant_id,
                    "parameter": parameter,
                    "value": value,
                }
            )
            emitted_parameters.add(parameter)

        for parameter in PMI_MASK_DERIVED_PARAMETERS:
            if parameter in emitted_parameters:
                continue
            value = row.get(parameter) if ownership_measurement_valid and parameter in df.columns else None
            if _is_missing(value):
                value = _derived_value(parameter)
            if _is_missing(value):
                continue
            rows.append(
                {
                    "round": round_value,
                    "plate": plate_value,
                    "plant_id": plant_id,
                    "parameter": parameter,
                    "value": value,
                }
            )
            emitted_parameters.add(parameter)

        if not include_fluorescence_placeholders:
            continue

        placeholder_key = (round_value, plate_value, plant_id)
        if placeholder_key in seen_placeholder_keys:
            continue
        seen_placeholder_keys.add(placeholder_key)
        for parameter in PMI_FLUORESCENCE_PLACEHOLDER_PARAMETERS:
            if parameter in emitted_parameters:
                continue
            value = _derived_value(parameter)
            if _is_missing(value):
                value = None
            rows.append(
                {
                    "round": round_value,
                    "plate": plate_value,
                    "plant_id": plant_id,
                    "parameter": parameter,
                    "value": value,
                }
            )

    pmi_df = pd.DataFrame(rows, columns=PMI_CORE_COLUMNS)
    if not pmi_df.empty:
        pmi_df.sort_values(by=["round", "plate", "plant_id", "parameter"], inplace=True, kind="mergesort")
        pmi_df.reset_index(drop=True, inplace=True)
    return pmi_df


def export_pmi_style_rows(
    master_df: pd.DataFrame | None,
    output_dir: Path,
    *,
    stem: str = "npec_root_traits_pmi_style",
    include_fluorescence_placeholders: bool = True,
    derive_mask_metrics: bool = True,
) -> tuple[pd.DataFrame | None, Path | None, Path | None]:
    pmi_df = build_pmi_style_rows(
        master_df,
        include_fluorescence_placeholders=include_fluorescence_placeholders,
        derive_mask_metrics=derive_mask_metrics,
    )
    if pmi_df.empty:
        return (None, None, None)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    pmi_df.to_csv(csv_path, index=False)

    xlsx_path = output_dir / f"{stem}.xlsx"
    try:
        pmi_df.to_excel(xlsx_path, index=False, sheet_name="Sheet1")
    except Exception:
        xlsx_path = None

    return (pmi_df, csv_path, xlsx_path)
