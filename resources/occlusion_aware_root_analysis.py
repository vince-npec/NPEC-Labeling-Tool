from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .analytics_engine import open_mp4_video_writer
from .mask_total_root_analysis import (
    _read_index_mask,
    _skeleton_edge_length_px,
    _skeletonize,
    natural_sort_key,
)
from .pyphenotyper_adapter import build_lucifer_green_shoot_mask
from .root_growth_video import apply_accepted_shoots_to_frame_rows


ROOT_PRIMARY_COLOR = (255, 132, 42)
ROOT_LATERAL_COLOR = (112, 158, 238)
SHOOT_COLOR = (255, 44, 190)
CARRIED_COLOR = (154, 86, 255)
BACKGROUND = (255, 255, 255)
PANEL_DARK = (20, 21, 22)
TEXT_DARK = (24, 28, 34)
DEFAULT_ROOT_CLASS_IDS = (1, 3)
DEFAULT_SHOOT_CLASS_IDS = (2,)
DEFAULT_PLANT_COUNT = 5


@dataclass(slots=True)
class OcclusionAwareOutput:
    output_dir: Path
    whole_plate_detail_csv: Path
    whole_plate_summary_csv: Path
    whole_plate_timelapse_csv: Path
    compartment_detail_csv: Path
    compartment_summary_csv: Path
    compartment_timelapse_csv: Path
    workbook_path: Path
    metadata_json: Path
    whole_plate_video: Path | None = None
    compartment_video: Path | None = None


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT_TITLE = _font(25, True)
FONT_SUBTITLE = _font(17, False)
FONT_SMALL = _font(13, False)
FONT_SMALL_BOLD = _font(13, True)
FONT_VALUE = _font(21, True)
FONT_TINY = _font(11, False)


def _cell_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _cell_int(value: object, default: int = 0) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        return int(default)
    return int(parsed)


def _cell_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            if not math.isfinite(float(value)):
                return bool(default)
        except Exception:
            return bool(default)
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "1.0", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "0.0", "no", "n", "off"}:
        return False
    return bool(default)


def _display_timestamp(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def _sort_detail(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Series" not in out.columns:
        out["Series"] = out.get("PetriDish", "Unknown")
    if "PetriDish" not in out.columns:
        out["PetriDish"] = "Unknown"
    out["Timestamp"] = pd.to_datetime(out.get("Timestamp"), errors="coerce")
    out["FrameIndex"] = pd.to_numeric(out.get("FrameIndex", -1), errors="coerce").fillna(-1).astype(int)
    out["_sort_series"] = out["Series"].fillna("Unknown").astype(str).map(natural_sort_key)
    out["_sort_petri"] = out["PetriDish"].fillna("Unknown").astype(str).map(natural_sort_key)
    out.sort_values(["_sort_series", "_sort_petri", "Timestamp", "FrameIndex"], inplace=True, kind="mergesort")
    return out.drop(columns=["_sort_series", "_sort_petri"], errors="ignore").reset_index(drop=True)


def _read_rgb(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray | None:
    try:
        with Image.open(path) as pil:
            rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None
    if shape_hw is not None and rgb.shape[:2] != shape_hw:
        try:
            rgb = np.asarray(Image.fromarray(rgb).resize((shape_hw[1], shape_hw[0]), Image.Resampling.BILINEAR), dtype=np.uint8)
        except Exception:
            return None
    return rgb


def _green_shoot_mask(image_path: Path, shape_hw: tuple[int, int], root_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, object]]:
    rgb = _read_rgb(image_path, shape_hw)
    if rgb is None:
        return np.zeros(shape_hw, dtype=bool), {"enabled": True, "error": "source_image_unreadable", "green_only_pixels": 0}
    try:
        color_mask, meta = build_lucifer_green_shoot_mask(rgb, shape_hw, root_mask=root_mask)
    except Exception as exc:
        return np.zeros(shape_hw, dtype=bool), {"enabled": True, "error": str(exc), "green_only_pixels": 0}
    out = np.asarray(color_mask, dtype=np.uint8) > 0
    if root_mask is not None:
        out &= ~(np.asarray(root_mask, dtype=bool))
    meta = dict(meta)
    meta["green_only_pixels"] = int(np.count_nonzero(out))
    return out, meta


def _row_green_shoot_measurement(row: pd.Series) -> tuple[int, float, int, str]:
    px = _cell_int(
        row.get(
            "shoot_area_green_only_px",
            row.get("shoot_rgb_rescue_green_only_px", row.get("shoot_area_px", 0)),
        ),
        0,
    )
    if px <= 0:
        px = _cell_int(row.get("shoot_area_px", 0), 0)
    area = _cell_float(row.get("shoot_area_green_only_mm2", row.get("shoot_area_mm2", 0.0)), 0.0)
    if area <= 0.0 and px > 0:
        pixel_size = _cell_float(row.get("pixel_size_mm", 0.0), 0.0)
        area = float(px * (pixel_size ** 2)) if pixel_size > 0 else 0.0
    components = _cell_int(row.get("shoot_rgb_rescue_components", row.get("shoot_green_filter_components", 0)), 0)
    source = str(row.get("shoot_measurement_source", "rgb_green_only") or "rgb_green_only")
    if source == "mask_class" and bool(row.get("shoot_rgb_rescue_enabled", False)):
        source = "rgb_green_only"
    return int(px), float(area), int(components), source


def _shoot_source_label(value: object) -> str:
    source = str(value or "").strip().lower()
    if "hades_bw_temporal_visual_reacquisition" in source:
        return "Hades BW model + temporal visual tracking"
    if "hades_bw_temporal_crown_track" in source:
        return "Hades BW model + temporal crown tracking"
    if "hades_bw_crown_local_cv" in source or "crown-local" in source:
        return "Hades BW model + crown-local CV"
    if "mask" in source or "class" in source:
        return "shoot mask class"
    return "RGB green-only shoot"


def _display_mask_path(row: pd.Series) -> Path:
    for column in ("OwnershipDisplayMaskPath", "OutputMaskPath"):
        value = str(row.get(column, "") or "").strip()
        if value and value.lower() != "nan":
            path = Path(value)
            if path.exists():
                return path
    return Path(str(row.get("OutputMaskPath", "") or ""))


def _ownership_shoot_lookup(ownership_df: pd.DataFrame | None) -> dict[tuple[str, str, pd.Timestamp, int, str], tuple[int, float]]:
    lookup: dict[tuple[str, str, pd.Timestamp, int, str], tuple[int, float]] = {}
    if ownership_df is None or ownership_df.empty:
        return lookup
    src = ownership_df.copy()
    for column, default in (("Series", "Unknown"), ("PetriDish", "Unknown"), ("plant_id", "")):
        if column not in src.columns:
            src[column] = default
    timestamp_source = src["Timestamp"] if "Timestamp" in src.columns else pd.Series([pd.NaT] * len(src), index=src.index)
    frame_source = (
        src["FrameIndex"]
        if "FrameIndex" in src.columns
        else src["frame_index"]
        if "frame_index" in src.columns
        else pd.Series([-1] * len(src), index=src.index)
    )
    src["_TimestampParsed"] = pd.to_datetime(timestamp_source, errors="coerce")
    src["_FrameIndex"] = pd.to_numeric(frame_source, errors="coerce").fillna(-1).astype(int)
    for _, row in src.iterrows():
        ts = row.get("_TimestampParsed")
        if pd.isna(ts):
            continue
        pixel_size = _cell_float(row.get("pixel_size_mm", 0.0), 0.0)
        px = _cell_int(row.get("shoot_area_green_only_px", row.get("shoot_area_px", 0)), 0)
        area = _cell_float(row.get("shoot_area_green_only_mm2", row.get("shoot_area_mm2", 0.0)), 0.0)
        if area <= 0.0 and px > 0 and pixel_size > 0.0:
            area = float(px * (pixel_size ** 2))
        key = (
            str(row.get("Series", "Unknown")),
            str(row.get("PetriDish", "Unknown")),
            pd.Timestamp(ts),
            int(row.get("_FrameIndex", -1)),
            str(row.get("plant_id", "")),
        )
        prev_px, prev_area = lookup.get(key, (0, 0.0))
        lookup[key] = (int(prev_px + px), float(prev_area + area))
    return lookup


def _dilate_bool(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if int(np.count_nonzero(mask_u8)) <= 0 or iterations <= 0:
        return mask_u8.astype(bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(mask_u8, kernel, iterations=int(iterations)).astype(bool)


def _skeleton_trace(mask: np.ndarray) -> np.ndarray:
    return np.asarray(_skeletonize(np.asarray(mask, dtype=np.uint8) > 0), dtype=bool)


def _trace_length_px(trace: np.ndarray) -> tuple[float, float, int]:
    trace_bool = np.asarray(trace, dtype=bool)
    px = float(np.count_nonzero(trace_bool))
    weighted = float(_skeleton_edge_length_px(trace_bool))
    return px, weighted, int(px)


def _class_masks(mask: np.ndarray, root_class_ids: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(mask, dtype=np.uint8)
    primary_id = int(root_class_ids[0]) if root_class_ids else 1
    lateral_id = int(root_class_ids[1]) if len(root_class_ids) > 1 else 0
    primary = arr == np.uint8(primary_id)
    lateral = arr == np.uint8(lateral_id) if lateral_id > 0 else np.zeros(arr.shape, dtype=bool)
    total = primary | lateral
    return primary, lateral, total


def _numeric_delta_by_group(df: pd.DataFrame, group_cols: list[str], columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        out[f"delta_{column}"] = values.groupby([out[col] for col in group_cols], dropna=False).diff().fillna(0.0)
    return out


def build_whole_plate_occlusion_dataframe(
    mask_total_df: pd.DataFrame,
    *,
    root_class_ids: tuple[int, ...] = DEFAULT_ROOT_CLASS_IDS,
) -> pd.DataFrame:
    """Add an occlusion-aware root length estimate that carries forward prior observations."""
    src = _sort_detail(mask_total_df)
    rows: list[dict[str, object]] = []
    group_cols = ["Series", "PetriDish"] if "Series" in src.columns else ["PetriDish"]
    for _key, group in src.groupby(group_cols, sort=False, dropna=False):
        carried_primary_mm = 0.0
        carried_lateral_mm = 0.0
        carried_total_mm = 0.0
        carried_primary_weighted_mm = 0.0
        carried_lateral_weighted_mm = 0.0
        carried_total_weighted_mm = 0.0
        for _, row in group.iterrows():
            out_row = row.to_dict()
            visible_primary_mm = _cell_float(row.get("primary_root_length_mm"))
            visible_lateral_mm = _cell_float(row.get("lateral_root_length_mm"))
            visible_total_mm = _cell_float(row.get("total_root_length_mm"))
            visible_primary_weighted_mm = _cell_float(row.get("primary_root_length_weighted_mm", visible_primary_mm))
            visible_lateral_weighted_mm = _cell_float(row.get("lateral_root_length_weighted_mm", visible_lateral_mm))
            visible_total_weighted_mm = _cell_float(row.get("total_root_length_weighted_mm", visible_total_mm))
            carried_primary_mm = max(carried_primary_mm, visible_primary_mm)
            carried_lateral_mm = max(carried_lateral_mm, visible_lateral_mm)
            carried_total_mm = max(carried_total_mm, visible_total_mm, carried_primary_mm + carried_lateral_mm)
            carried_primary_weighted_mm = max(carried_primary_weighted_mm, visible_primary_weighted_mm)
            carried_lateral_weighted_mm = max(carried_lateral_weighted_mm, visible_lateral_weighted_mm)
            carried_total_weighted_mm = max(
                carried_total_weighted_mm,
                visible_total_weighted_mm,
                carried_primary_weighted_mm + carried_lateral_weighted_mm,
            )
            shoot_green_px, shoot_area_mm2, shoot_components, shoot_source = _row_green_shoot_measurement(row)
            out_row.update(
                {
                    "occlusion_analysis_status": "ok",
                    "occlusion_trace_mode": "monotonic_length_carry_forward",
                    "visible_total_root_length_mm": visible_total_mm,
                    "visible_primary_root_length_mm": visible_primary_mm,
                    "visible_lateral_root_length_mm": visible_lateral_mm,
                    "visible_total_root_length_weighted_mm": visible_total_weighted_mm,
                    "visible_primary_root_length_weighted_mm": visible_primary_weighted_mm,
                    "visible_lateral_root_length_weighted_mm": visible_lateral_weighted_mm,
                    "occlusion_aware_primary_root_length_px": 0.0,
                    "occlusion_aware_primary_root_length_mm": float(carried_primary_mm),
                    "occlusion_aware_primary_root_length_weighted_px": 0.0,
                    "occlusion_aware_primary_root_length_weighted_mm": float(carried_primary_weighted_mm),
                    "occlusion_aware_lateral_root_length_px": 0.0,
                    "occlusion_aware_lateral_root_length_mm": float(carried_lateral_mm),
                    "occlusion_aware_lateral_root_length_weighted_px": 0.0,
                    "occlusion_aware_lateral_root_length_weighted_mm": float(carried_lateral_weighted_mm),
                    "occlusion_aware_total_root_length_px": 0.0,
                    "occlusion_aware_total_root_length_mm": float(carried_total_mm),
                    "occlusion_aware_total_root_length_weighted_px": 0.0,
                    "occlusion_aware_total_root_length_weighted_mm": float(carried_total_weighted_mm),
                    "occlusion_inferred_primary_root_length_mm": float(max(0.0, carried_primary_mm - visible_primary_mm)),
                    "occlusion_inferred_lateral_root_length_mm": float(max(0.0, carried_lateral_mm - visible_lateral_mm)),
                    "occlusion_inferred_total_root_length_mm": float(max(0.0, carried_total_mm - visible_total_mm)),
                    "occlusion_inferred_primary_trace_pixels": 0,
                    "occlusion_inferred_lateral_trace_pixels": 0,
                    "occlusion_inferred_total_trace_pixels": 0,
                    "occlusion_new_primary_trace_pixels": 0,
                    "occlusion_new_lateral_trace_pixels": 0,
                    "occlusion_carried_primary_trace_pixels": 0,
                    "occlusion_carried_lateral_trace_pixels": 0,
                    "shoot_area_green_only_px": int(shoot_green_px),
                    "shoot_area_green_only_mm2": float(shoot_area_mm2),
                    "shoot_green_filter_components": int(shoot_components),
                    "shoot_green_filter_source": str(shoot_source),
                }
            )
            rows.append(out_row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _sort_detail(out)
    group_cols = ["Series", "PetriDish"] if "Series" in out.columns else ["PetriDish"]
    return _numeric_delta_by_group(
        out,
        group_cols,
        (
            "occlusion_aware_total_root_length_mm",
            "occlusion_aware_primary_root_length_mm",
            "occlusion_aware_lateral_root_length_mm",
            "occlusion_inferred_total_root_length_mm",
            "shoot_area_green_only_mm2",
        ),
    )


def _derive_lane_centers(
    whole_df: pd.DataFrame,
    ownership_df: pd.DataFrame | None,
    *,
    plant_count: int = DEFAULT_PLANT_COUNT,
) -> dict[tuple[str, str], list[float]]:
    centers: dict[tuple[str, str], list[float]] = {}
    if ownership_df is not None and not ownership_df.empty:
        src = ownership_df.copy()
        for required in ("Series", "PetriDish", "plant_id"):
            if required not in src.columns:
                src[required] = "Unknown"
        for key, group in src.groupby(["Series", "PetriDish"], sort=False, dropna=False):
            values: list[float] = []
            for plant_index in range(1, plant_count + 1):
                plant = f"plant_{plant_index:02d}"
                plant_rows = group[group["plant_id"].astype(str) == plant]
                x_candidates: list[float] = []
                if {"shoot_bbox_x", "shoot_bbox_w"}.issubset(plant_rows.columns):
                    x = pd.to_numeric(plant_rows["shoot_bbox_x"], errors="coerce")
                    w = pd.to_numeric(plant_rows["shoot_bbox_w"], errors="coerce")
                    valid = (w > 0) & np.isfinite(x) & np.isfinite(w)
                    x_candidates.extend((x[valid] + w[valid] / 2.0).astype(float).tolist())
                if {"bbox_x", "bbox_w"}.issubset(plant_rows.columns):
                    x = pd.to_numeric(plant_rows["bbox_x"], errors="coerce")
                    w = pd.to_numeric(plant_rows["bbox_w"], errors="coerce")
                    valid = (w > 0) & np.isfinite(x) & np.isfinite(w)
                    x_candidates.extend((x[valid] + w[valid] / 2.0).astype(float).tolist())
                values.append(float(np.nanmedian(x_candidates)) if x_candidates else float("nan"))
            if any(math.isfinite(v) for v in values):
                centers[(str(key[0]), str(key[1]))] = _fill_lane_centers(values)

    for key, group in whole_df.groupby(["Series", "PetriDish"], sort=False, dropna=False):
        key_tuple = (str(key[0]), str(key[1]))
        if key_tuple in centers:
            continue
        width = int(pd.to_numeric(group.get("mask_width", pd.Series(dtype=float)), errors="coerce").dropna().median() or 1)
        centers[key_tuple] = [(idx + 1) * width / float(plant_count + 1) for idx in range(plant_count)]
    return centers


def _fill_lane_centers(values: list[float]) -> list[float]:
    n = len(values)
    finite = [(idx, val) for idx, val in enumerate(values) if math.isfinite(float(val))]
    if len(finite) >= 2:
        finite_values = np.asarray([val for _idx, val in finite], dtype=np.float64)
        left = float(np.nanmin(finite_values))
        right = float(np.nanmax(finite_values))
    elif len(finite) == 1:
        only = float(finite[0][1])
        spacing = max(80.0, only / max(1, finite[0][0] + 1))
        left = only - finite[0][0] * spacing
        right = only + (n - finite[0][0] - 1) * spacing
    else:
        return [float(idx) for idx in range(n)]
    if right <= left:
        right = left + max(1.0, float(n - 1) * 80.0)
    filled = np.linspace(left, right, n).astype(float).tolist()
    for idx, val in finite:
        filled[idx] = float(val)
    return sorted(float(v) for v in filled)


def _lane_bounds(centers: list[float], width: int) -> list[tuple[int, int]]:
    if not centers:
        return [(0, int(width))]
    centers = sorted(float(v) for v in centers)
    bounds: list[tuple[int, int]] = []
    left = 0
    for idx, center in enumerate(centers):
        if idx < len(centers) - 1:
            right = int(round((center + centers[idx + 1]) / 2.0))
        else:
            right = int(width)
        bounds.append((max(0, min(int(width), left)), max(0, min(int(width), right))))
        left = right
    return bounds


def _ownership_root_values(row: pd.Series) -> tuple[float, float, float]:
    primary = _cell_float(
        row.get(
            "primary_root_length_mm_clean",
            row.get("primary_root_length_mm_raw", row.get("root_length_mm_clean", row.get("root_length_mm_raw", row.get("root_length_mm", 0.0)))),
        ),
        0.0,
    )
    lateral = _cell_float(row.get("lateral_total_length_mm", row.get("lateral_root_length_mm", 0.0)), 0.0)
    total_candidates = (
        row.get("total_root_length_mm_clean"),
        row.get("combined_root_length_mm"),
        row.get("root_length_mm_clean"),
        row.get("root_length_mm_raw"),
    )
    total = 0.0
    for value in total_candidates:
        total = _cell_float(value, 0.0)
        if total > 0.0:
            break
    total = max(total, primary + lateral)
    return float(primary), float(lateral), float(total)


def _build_compartmentalized_from_ownership(
    whole_df: pd.DataFrame,
    ownership_df: pd.DataFrame,
    *,
    plant_count: int,
) -> pd.DataFrame:
    src = ownership_df.copy()
    for column, default in (("Series", "Unknown"), ("PetriDish", "Unknown"), ("plant_id", "")):
        if column not in src.columns:
            src[column] = default
    if "Timestamp" not in src.columns:
        src["Timestamp"] = pd.NaT
    if "FrameIndex" not in src.columns:
        src["FrameIndex"] = src["frame_index"] if "frame_index" in src.columns else -1
    src = _sort_detail(src)
    centers_by_plate = _derive_lane_centers(whole_df, src, plant_count=plant_count)
    rows: list[dict[str, object]] = []
    for _key, group in src.groupby(["Series", "PetriDish", "plant_id"], sort=False, dropna=False):
        carried_primary_mm = 0.0
        carried_lateral_mm = 0.0
        carried_total_mm = 0.0
        for _, row in group.iterrows():
            primary_mm, lateral_mm, total_mm = _ownership_root_values(row)
            ownership_measurement_valid = _cell_bool(row.get("ownership_measurement_valid", True), True)
            previous_carry = (carried_primary_mm, carried_lateral_mm, carried_total_mm)
            if ownership_measurement_valid:
                carried_primary_mm = max(carried_primary_mm, primary_mm)
                carried_lateral_mm = max(carried_lateral_mm, lateral_mm)
                carried_total_mm = max(carried_total_mm, total_mm, carried_primary_mm + carried_lateral_mm)
            if not ownership_measurement_valid:
                carry_forward_status = "frozen_invalid_ownership_measurement"
            elif previous_carry != (carried_primary_mm, carried_lateral_mm, carried_total_mm):
                carry_forward_status = "advanced_valid_individual_measurement"
            else:
                carry_forward_status = "retained_valid_individual_maximum"
            pixel_size = _cell_float(row.get("pixel_size_mm", 0.0), 0.0)
            shoot_px = _cell_int(row.get("shoot_area_green_only_px", row.get("shoot_area_px", 0)), 0)
            shoot_area = _cell_float(row.get("shoot_area_green_only_mm2", row.get("shoot_area_mm2", 0.0)), 0.0)
            if shoot_area <= 0.0 and shoot_px > 0 and pixel_size > 0.0:
                shoot_area = float(shoot_px * (pixel_size ** 2))
            plate_key = (str(row.get("Series", "Unknown")), str(row.get("PetriDish", "Unknown")))
            centers = centers_by_plate.get(plate_key, [])
            plant_text = str(row.get("plant_id", ""))
            plant_index = 0
            try:
                plant_index = max(0, int(plant_text.split("_")[-1]) - 1)
            except Exception:
                plant_index = 0
            lane_center = float(centers[plant_index]) if plant_index < len(centers) else _cell_float(row.get("bbox_x", 0.0)) + _cell_float(row.get("bbox_w", 0.0)) / 2.0
            rows.append(
                {
                    "Series": row.get("Series", "Unknown"),
                    "PetriDish": row.get("PetriDish", "Unknown"),
                    "Timestamp": row.get("Timestamp", ""),
                    "FrameIndex": row.get("FrameIndex", -1),
                    "plant_id": plant_text,
                    "lane_x0": "",
                    "lane_x1": "",
                    "lane_center_x": float(lane_center),
                    "lane_center_source": "ownership_track",
                    "SourceFile": row.get("SourceFile", ""),
                    "OutputMaskPath": row.get("OutputMaskPath", ""),
                    "OwnershipDisplayMaskPath": row.get("OwnershipDisplayMaskPath", ""),
                    "pixel_size_mm": float(pixel_size),
                    "visible_primary_root_length_px": _cell_float(row.get("root_length_px_clean", row.get("root_length_px", 0.0)), 0.0),
                    "visible_primary_root_length_weighted_px": _cell_float(row.get("root_length_px_clean", row.get("root_length_px", 0.0)), 0.0),
                    "visible_primary_root_length_mm": float(primary_mm),
                    "visible_lateral_root_length_px": _cell_float(row.get("lateral_total_length_px", 0.0), 0.0),
                    "visible_lateral_root_length_weighted_px": _cell_float(row.get("lateral_total_length_px", 0.0), 0.0),
                    "visible_lateral_root_length_mm": float(lateral_mm),
                    "visible_total_root_length_mm": float(total_mm),
                    "occlusion_aware_primary_root_length_px": 0.0,
                    "occlusion_aware_primary_root_length_weighted_px": 0.0,
                    "occlusion_aware_primary_root_length_mm": float(carried_primary_mm),
                    "occlusion_aware_lateral_root_length_px": 0.0,
                    "occlusion_aware_lateral_root_length_weighted_px": 0.0,
                    "occlusion_aware_lateral_root_length_mm": float(carried_lateral_mm),
                    "occlusion_aware_total_root_length_mm": float(carried_total_mm),
                    "occlusion_inferred_total_root_length_mm": float(max(0.0, carried_total_mm - total_mm)),
                    "visible_primary_trace_pixels": _cell_int(row.get("root_area_px", 0), 0),
                    "visible_lateral_trace_pixels": 0,
                    "occlusion_carried_primary_trace_pixels": 0,
                    "occlusion_carried_lateral_trace_pixels": 0,
                    "occlusion_inferred_primary_trace_pixels": 0,
                    "occlusion_inferred_lateral_trace_pixels": 0,
                    "shoot_area_green_only_px": int(shoot_px),
                    "shoot_area_green_only_mm2": float(shoot_area),
                    "shoot_green_filter_components": 0,
                    "shoot_green_filter_source": str(
                        row.get(
                            "shoot_green_filter_source",
                            row.get("shoot_measurement_source", "rgb_green_only"),
                        )
                    ),
                    "measurement_tier": row.get("measurement_tier", ""),
                    "ownership_measurement_valid": bool(ownership_measurement_valid),
                    "conflict_group_id": row.get("conflict_group_id", ""),
                    "conflict_group_members": row.get("conflict_group_members", []),
                    "conflict_group_size": row.get("conflict_group_size", 1),
                    "conflict_start_frame": row.get("conflict_start_frame", ""),
                    "conflict_touching_now": row.get("conflict_touching_now", ""),
                    "combined_root_length_mm": row.get("combined_root_length_mm"),
                    "combined_root_length_px": row.get("combined_root_length_px"),
                    "combined_area_px": row.get("combined_area_px"),
                    "combined_area_mm2": row.get("combined_area_mm2"),
                    "combined_group_bbox_x": row.get("combined_group_bbox_x"),
                    "combined_group_bbox_y": row.get("combined_group_bbox_y"),
                    "combined_group_bbox_w": row.get("combined_group_bbox_w"),
                    "combined_group_bbox_h": row.get("combined_group_bbox_h"),
                    "occlusion_analysis_status": "ok" if ownership_measurement_valid else "individual_state_frozen",
                    "occlusion_trace_mode": "ownership_validity_gated_monotonic_length_carry_forward",
                    "occlusion_individual_state_frozen": bool(not ownership_measurement_valid),
                    "occlusion_carry_forward_status": carry_forward_status,
                    "analysis_mode": "ownership_track_compartment_occlusion_carry_forward",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _sort_detail(out)
    return _numeric_delta_by_group(
        out,
        ["Series", "PetriDish", "plant_id"],
        (
            "occlusion_aware_total_root_length_mm",
            "visible_total_root_length_mm",
            "shoot_area_green_only_mm2",
        ),
    )


def build_compartmentalized_occlusion_dataframe(
    whole_df: pd.DataFrame,
    *,
    ownership_df: pd.DataFrame | None = None,
    root_class_ids: tuple[int, ...] = DEFAULT_ROOT_CLASS_IDS,
    plant_count: int = DEFAULT_PLANT_COUNT,
) -> pd.DataFrame:
    """Best-effort lane-based per-seedling occlusion-aware measurements."""
    src = _sort_detail(whole_df)
    if ownership_df is not None and not ownership_df.empty:
        return _build_compartmentalized_from_ownership(src, ownership_df, plant_count=int(plant_count))
    centers_by_plate = _derive_lane_centers(src, ownership_df, plant_count=plant_count)
    shoot_lookup = _ownership_shoot_lookup(ownership_df)
    rows: list[dict[str, object]] = []
    for key, group in src.groupby(["Series", "PetriDish"], sort=False, dropna=False):
        key_tuple = (str(key[0]), str(key[1]))
        lane_centers = centers_by_plate.get(key_tuple, [])
        primary_memory: list[np.ndarray | None] = [None for _ in range(plant_count)]
        lateral_memory: list[np.ndarray | None] = [None for _ in range(plant_count)]
        for _, frame_row in group.iterrows():
            mask_path = Path(str(frame_row.get("OutputMaskPath", "")))
            image_path = Path(str(frame_row.get("SourceFile", frame_row.get("PreviewImagePath", ""))))
            if not mask_path.exists():
                continue
            arr = _read_index_mask(mask_path)
            h, w = arr.shape[:2]
            bounds = _lane_bounds(lane_centers, w)
            primary_mask, lateral_mask, root_mask = _class_masks(arr, root_class_ids)
            primary_trace = _skeleton_trace(primary_mask)
            lateral_trace = _skeleton_trace(lateral_mask)
            pixel_size = _cell_float(frame_row.get("pixel_size_mm"), 1.0)
            for plant_idx in range(plant_count):
                x0, x1 = bounds[plant_idx] if plant_idx < len(bounds) else (0, w)
                lane_primary = np.asarray(primary_trace[:, x0:x1], dtype=bool)
                lane_lateral = np.asarray(lateral_trace[:, x0:x1], dtype=bool)
                if primary_memory[plant_idx] is None or primary_memory[plant_idx].shape != lane_primary.shape:
                    primary_memory[plant_idx] = np.zeros(lane_primary.shape, dtype=bool)
                    lateral_memory[plant_idx] = np.zeros(lane_lateral.shape, dtype=bool)
                assert primary_memory[plant_idx] is not None
                assert lateral_memory[plant_idx] is not None
                primary_memory[plant_idx] |= lane_primary
                lateral_memory[plant_idx] |= lane_lateral
                visible_primary_px, visible_primary_weighted_px, visible_primary_trace_px = _trace_length_px(lane_primary)
                visible_lateral_px, visible_lateral_weighted_px, visible_lateral_trace_px = _trace_length_px(lane_lateral)
                carried_primary_px, carried_primary_weighted_px, carried_primary_trace_px = _trace_length_px(primary_memory[plant_idx])
                carried_lateral_px, carried_lateral_weighted_px, carried_lateral_trace_px = _trace_length_px(lateral_memory[plant_idx])
                plant_id = f"plant_{plant_idx + 1:02d}"
                ts = pd.to_datetime(frame_row.get("Timestamp"), errors="coerce")
                shoot_key = (
                    str(frame_row.get("Series", "Unknown")),
                    str(frame_row.get("PetriDish", "Unknown")),
                    pd.Timestamp(ts) if not pd.isna(ts) else pd.Timestamp("1970-01-01"),
                    _cell_int(frame_row.get("FrameIndex", -1), -1),
                    plant_id,
                )
                shoot_px, shoot_area_mm2 = shoot_lookup.get(shoot_key, (0, 0.0))
                if shoot_px <= 0:
                    whole_shoot_px = _cell_int(frame_row.get("shoot_area_green_only_px", frame_row.get("shoot_area_px", 0)), 0)
                    shoot_px = int(round(whole_shoot_px / max(1, int(plant_count))))
                    shoot_area_mm2 = float(shoot_px * (pixel_size ** 2))
                visible_total_mm = float((visible_primary_px + visible_lateral_px) * pixel_size)
                occl_primary_mm = float(max(visible_primary_px, carried_primary_px) * pixel_size)
                occl_lateral_mm = float(max(visible_lateral_px, carried_lateral_px) * pixel_size)
                occl_total_mm = float(occl_primary_mm + occl_lateral_mm)
                rows.append(
                    {
                        "Series": frame_row.get("Series", "Unknown"),
                        "PetriDish": frame_row.get("PetriDish", "Unknown"),
                        "Timestamp": frame_row.get("Timestamp", ""),
                        "FrameIndex": frame_row.get("FrameIndex", -1),
                        "plant_id": plant_id,
                        "lane_x0": int(x0),
                        "lane_x1": int(x1),
                        "lane_center_x": float(lane_centers[plant_idx]) if plant_idx < len(lane_centers) else float((x0 + x1) / 2.0),
                        "lane_center_source": "ownership_shoot_bbox_or_root_bbox" if key_tuple in centers_by_plate else "even_spacing",
                        "SourceFile": str(image_path),
                        "OutputMaskPath": str(mask_path),
                        "OwnershipDisplayMaskPath": frame_row.get("OwnershipDisplayMaskPath", ""),
                        "pixel_size_mm": float(pixel_size),
                        "visible_primary_root_length_px": float(visible_primary_px),
                        "visible_primary_root_length_weighted_px": float(visible_primary_weighted_px),
                        "visible_primary_root_length_mm": float(visible_primary_px * pixel_size),
                        "visible_lateral_root_length_px": float(visible_lateral_px),
                        "visible_lateral_root_length_weighted_px": float(visible_lateral_weighted_px),
                        "visible_lateral_root_length_mm": float(visible_lateral_px * pixel_size),
                        "visible_total_root_length_mm": float(visible_total_mm),
                        "occlusion_aware_primary_root_length_px": float(carried_primary_px),
                        "occlusion_aware_primary_root_length_weighted_px": float(carried_primary_weighted_px),
                        "occlusion_aware_primary_root_length_mm": float(occl_primary_mm),
                        "occlusion_aware_lateral_root_length_px": float(carried_lateral_px),
                        "occlusion_aware_lateral_root_length_weighted_px": float(carried_lateral_weighted_px),
                        "occlusion_aware_lateral_root_length_mm": float(occl_lateral_mm),
                        "occlusion_aware_total_root_length_mm": float(occl_total_mm),
                        "occlusion_inferred_total_root_length_mm": float(max(0.0, occl_total_mm - visible_total_mm)),
                        "visible_primary_trace_pixels": int(visible_primary_trace_px),
                        "visible_lateral_trace_pixels": int(visible_lateral_trace_px),
                        "occlusion_carried_primary_trace_pixels": int(carried_primary_trace_px),
                        "occlusion_carried_lateral_trace_pixels": int(carried_lateral_trace_px),
                        "occlusion_inferred_primary_trace_pixels": int(np.count_nonzero(primary_memory[plant_idx] & ~lane_primary)),
                        "occlusion_inferred_lateral_trace_pixels": int(np.count_nonzero(lateral_memory[plant_idx] & ~lane_lateral)),
                        "shoot_area_green_only_px": int(shoot_px),
                        "shoot_area_green_only_mm2": float(shoot_area_mm2),
                        "shoot_green_filter_components": 0,
                        "shoot_green_filter_source": str(
                            frame_row.get("shoot_green_filter_source", "rgb_green_only")
                        ),
                        "analysis_mode": "best_effort_lane_compartment_occlusion_carry_forward",
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _sort_detail(out)
    return _numeric_delta_by_group(
        out,
        ["Series", "PetriDish", "plant_id"],
        (
            "occlusion_aware_total_root_length_mm",
            "visible_total_root_length_mm",
            "shoot_area_green_only_mm2",
        ),
    )


def build_whole_plate_summary(whole_df: pd.DataFrame) -> pd.DataFrame:
    if whole_df is None or whole_df.empty:
        return pd.DataFrame()
    src = whole_df.copy()
    return (
        src.groupby(["Series", "PetriDish"], dropna=False)
        .agg(
            frames=("FrameIndex", "count"),
            first_timestamp=("Timestamp", "first"),
            last_timestamp=("Timestamp", "last"),
            final_visible_total_root_length_mm=("visible_total_root_length_mm", "last"),
            final_occlusion_aware_total_root_length_mm=("occlusion_aware_total_root_length_mm", "last"),
            max_occlusion_aware_total_root_length_mm=("occlusion_aware_total_root_length_mm", "max"),
            final_occlusion_aware_primary_root_length_mm=("occlusion_aware_primary_root_length_mm", "last"),
            final_occlusion_aware_lateral_root_length_mm=("occlusion_aware_lateral_root_length_mm", "last"),
            max_occlusion_inferred_total_root_length_mm=("occlusion_inferred_total_root_length_mm", "max"),
            final_shoot_area_green_only_mm2=("shoot_area_green_only_mm2", "last"),
            frames_with_green_shoot_pixels=("shoot_area_green_only_px", lambda values: int((pd.to_numeric(values, errors="coerce").fillna(0) > 0).sum())),
        )
        .reset_index()
    )


def build_compartment_summary(compartment_df: pd.DataFrame) -> pd.DataFrame:
    if compartment_df is None or compartment_df.empty:
        return pd.DataFrame()
    src = compartment_df.copy()
    return (
        src.groupby(["Series", "PetriDish", "plant_id"], dropna=False)
        .agg(
            frames=("FrameIndex", "count"),
            first_timestamp=("Timestamp", "first"),
            last_timestamp=("Timestamp", "last"),
            lane_center_x=("lane_center_x", "median"),
            final_visible_total_root_length_mm=("visible_total_root_length_mm", "last"),
            final_occlusion_aware_total_root_length_mm=("occlusion_aware_total_root_length_mm", "last"),
            max_occlusion_aware_total_root_length_mm=("occlusion_aware_total_root_length_mm", "max"),
            final_occlusion_aware_primary_root_length_mm=("occlusion_aware_primary_root_length_mm", "last"),
            final_occlusion_aware_lateral_root_length_mm=("occlusion_aware_lateral_root_length_mm", "last"),
            max_occlusion_inferred_total_root_length_mm=("occlusion_inferred_total_root_length_mm", "max"),
            final_shoot_area_green_only_mm2=("shoot_area_green_only_mm2", "last"),
            frames_with_green_shoot_pixels=("shoot_area_green_only_px", lambda values: int((pd.to_numeric(values, errors="coerce").fillna(0) > 0).sum())),
        )
        .reset_index()
    )


def build_compartment_timelapse(compartment_df: pd.DataFrame) -> pd.DataFrame:
    if compartment_df is None or compartment_df.empty:
        return pd.DataFrame()
    src = compartment_df.copy()
    grouped = (
        src.groupby(["Series", "PetriDish", "Timestamp", "FrameIndex"], dropna=False)
        .agg(
            plants=("plant_id", "nunique"),
            visible_total_root_length_mm=("visible_total_root_length_mm", "sum"),
            occlusion_aware_total_root_length_mm=("occlusion_aware_total_root_length_mm", "sum"),
            occlusion_aware_primary_root_length_mm=("occlusion_aware_primary_root_length_mm", "sum"),
            occlusion_aware_lateral_root_length_mm=("occlusion_aware_lateral_root_length_mm", "sum"),
            occlusion_inferred_total_root_length_mm=("occlusion_inferred_total_root_length_mm", "sum"),
            shoot_area_green_only_mm2=("shoot_area_green_only_mm2", "sum"),
        )
        .reset_index()
    )
    grouped = _sort_detail(grouped)
    return _numeric_delta_by_group(
        grouped,
        ["Series", "PetriDish"],
        ("visible_total_root_length_mm", "occlusion_aware_total_root_length_mm", "shoot_area_green_only_mm2"),
    )


def _fit_dims(src_w: int, src_h: int, box_w: int, box_h: int) -> tuple[int, int]:
    scale = min(box_w / max(1.0, float(src_w)), box_h / max(1.0, float(src_h)))
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def _blend_overlay(base: Image.Image, masks: list[tuple[np.ndarray, tuple[int, int, int], float]], size: tuple[int, int]) -> Image.Image:
    resized = base.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32)
    for mask, color, alpha in masks:
        m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST), dtype=np.uint8) > 0
        if not np.any(m):
            continue
        color_arr = np.array(color, dtype=np.float32)
        arr[m] = arr[m] * (1.0 - float(alpha)) + color_arr * float(alpha)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _draw_line_chart(
    draw: ImageDraw.ImageDraw,
    values_a: list[float],
    values_b: list[float],
    current_idx: int,
    box: tuple[int, int, int, int],
    *,
    label_a: str,
    label_b: str,
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle(box, outline=(95, 95, 95), width=2)
    n = max(1, len(values_a), len(values_b))
    max_val = max(10.0, *(float(v) for v in values_a if math.isfinite(float(v))), *(float(v) for v in values_b if math.isfinite(float(v))))
    y_max = max_val * 1.12
    for i in range(6):
        y = y2 - int(round((y2 - y1) * i / 5.0))
        draw.line((x1, y, x2, y), fill=(225, 229, 232), width=1)
        draw.text((x1 - 58, y - 7), f"{y_max * i / 5.0:.0f}", fill=(80, 80, 80), font=FONT_TINY)
    x_positions = [(x1 + x2) // 2] if n <= 1 else [x1 + int(round((x2 - x1) * i / (n - 1))) for i in range(n)]

    def points(values: list[float]) -> list[tuple[int, int]]:
        pts: list[tuple[int, int]] = []
        for idx, value in enumerate(values[: current_idx + 1]):
            y = y2 - int(round((float(value) / max(1.0, y_max)) * (y2 - y1)))
            pts.append((x_positions[idx], y))
        return pts

    for vals, color in ((values_a, color_a), (values_b, color_b)):
        pts = points(vals)
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=3)
        for x, y in pts:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    legend_x = x2 - 235
    legend_y = y1 + 12
    draw.rounded_rectangle((legend_x - 10, legend_y - 10, x2 - 10, legend_y + 58), radius=8, fill=(255, 255, 255), outline=(216, 216, 216))
    draw.line((legend_x, legend_y + 8, legend_x + 30, legend_y + 8), fill=color_a, width=3)
    draw.text((legend_x + 40, legend_y), label_a, fill=TEXT_DARK, font=FONT_SMALL)
    draw.line((legend_x, legend_y + 34, legend_x + 30, legend_y + 34), fill=color_b, width=3)
    draw.text((legend_x + 40, legend_y + 26), label_b, fill=TEXT_DARK, font=FONT_SMALL)


def _whole_plate_frame(
    row: pd.Series,
    plate_df: pd.DataFrame,
    current_idx: int,
    *,
    primary_memory: np.ndarray,
    lateral_memory: np.ndarray,
    width: int,
    height: int,
    root_class_ids: tuple[int, ...],
) -> Image.Image:
    frame = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(frame)
    left_w = width // 2
    draw.rectangle((0, 0, left_w, height), fill=PANEL_DARK)
    image_path = Path(str(row.get("SourceFile", row.get("PreviewImagePath", ""))))
    mask_path = _display_mask_path(row)
    if image_path.exists():
        image = Image.open(image_path).convert("RGB")
    elif mask_path.exists():
        arr = _read_index_mask(mask_path)
        image = Image.fromarray(np.full((arr.shape[0], arr.shape[1], 3), 245, dtype=np.uint8), "RGB")
    else:
        draw.text((48, height // 2), "Missing source image and mask", fill=(245, 245, 245), font=FONT_TITLE)
        return frame
    src_w, src_h = image.size
    panel_w, panel_h = left_w - 32, height - 36
    dst_w, dst_h = _fit_dims(src_w, src_h, panel_w, panel_h)
    x0, y0 = 16 + (panel_w - dst_w) // 2, 18 + (panel_h - dst_h) // 2
    arr = _read_index_mask(mask_path) if mask_path.exists() else np.zeros((src_h, src_w), dtype=np.uint8)
    primary_mask, lateral_mask, root_mask = _class_masks(arr, root_class_ids)
    shoot_mask = arr == np.uint8(DEFAULT_SHOOT_CLASS_IDS[0])
    if int(np.count_nonzero(shoot_mask)) <= 0:
        shoot_mask, _shoot_meta = _green_shoot_mask(image_path, tuple(int(v) for v in arr.shape[:2]), root_mask=root_mask)
    carried_mask = (primary_memory | lateral_memory) & ~(primary_mask | lateral_mask)
    panel = _blend_overlay(
        image,
        [
            (_dilate_bool(primary_mask, 1), ROOT_PRIMARY_COLOR, 0.62),
            (_dilate_bool(lateral_mask, 1), ROOT_LATERAL_COLOR, 0.62),
            (_dilate_bool(carried_mask, 2), CARRIED_COLOR, 0.54),
            (_dilate_bool(shoot_mask, 1), SHOOT_COLOR, 0.62),
        ],
        (dst_w, dst_h),
    )
    frame.paste(panel, (x0, y0))
    label = f"{row.get('Series', '')} / {row.get('PetriDish', '')} | {_display_timestamp(row.get('Timestamp'))}"
    draw.rounded_rectangle((24, 26, min(left_w - 24, 38 + len(label) * 8), 58), radius=8, fill=(0, 0, 0))
    draw.text((34, 34), label, fill=(250, 250, 250), font=FONT_SMALL_BOLD)
    legend_y = height - 144
    draw.rounded_rectangle((24, legend_y, 450, height - 26), radius=8, fill=(0, 0, 0))
    for idx, (name, color) in enumerate(
        (
            ("visible primary", ROOT_PRIMARY_COLOR),
            ("visible lateral", ROOT_LATERAL_COLOR),
            ("carried historical root", CARRIED_COLOR),
            (_shoot_source_label(row.get("shoot_green_filter_source")), SHOOT_COLOR),
        )
    ):
        y = legend_y + 16 + idx * 26
        draw.rectangle((40, y + 3, 58, y + 17), fill=color)
        draw.text((68, y), name, fill=(238, 238, 238), font=FONT_SMALL)

    draw.rectangle((left_w, 0, width, height), fill=BACKGROUND)
    draw.line((left_w, 0, left_w, height), fill=(18, 18, 18), width=4)
    draw.text((left_w + 105, 38), "Visible vs Occlusion-Aware Root Length", fill=TEXT_DARK, font=FONT_TITLE)
    draw.text((left_w + 225, 72), str(row.get("PetriDish", "")), fill=(60, 65, 74), font=FONT_SUBTITLE)
    visible = pd.to_numeric(plate_df["visible_total_root_length_mm"], errors="coerce").fillna(0.0).astype(float).tolist()
    occl = pd.to_numeric(plate_df["occlusion_aware_total_root_length_mm"], errors="coerce").fillna(0.0).astype(float).tolist()
    _draw_line_chart(
        draw,
        visible,
        occl,
        current_idx,
        (left_w + 104, 130, width - 84, height - 248),
        label_a="visible root",
        label_b="carried root",
        color_a=(42, 75, 122),
        color_b=CARRIED_COLOR,
    )
    current_visible = _cell_float(row.get("visible_total_root_length_mm"))
    current_occl = _cell_float(row.get("occlusion_aware_total_root_length_mm"))
    inferred = _cell_float(row.get("occlusion_inferred_total_root_length_mm"))
    shoot_area = _cell_float(row.get("shoot_area_green_only_mm2"))
    stats = [
        ("Visible root", f"{current_visible:.1f} mm"),
        ("Carried root", f"{current_occl:.1f} mm"),
        ("Inferred hidden", f"{inferred:.1f} mm"),
        ("Green shoot", f"{shoot_area:.2f} mm2"),
        ("Frame", f"{current_idx + 1}/{len(plate_df)}"),
    ]
    stats_y = height - 150
    draw.rounded_rectangle((left_w + 60, stats_y, width - 60, height - 36), radius=12, fill=(246, 248, 250), outline=(218, 224, 230))
    col_w = (width - left_w - 140) // len(stats)
    for idx, (name, value) in enumerate(stats):
        sx = left_w + 80 + idx * col_w
        draw.text((sx, stats_y + 18), name, fill=(92, 98, 106), font=FONT_SMALL)
        draw.text((sx, stats_y + 48), value, fill=TEXT_DARK, font=FONT_VALUE)
    return frame


def generate_whole_plate_occlusion_video(
    whole_df: pd.DataFrame,
    output_path: Path,
    *,
    root_class_ids: tuple[int, ...] = DEFAULT_ROOT_CLASS_IDS,
    fps: int = 10,
    width: int = 1920,
    height: int = 1080,
    sample_frame_path: Path | None = None,
    contact_sheet_path: Path | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_frame_path = sample_frame_path or output_path.with_name(output_path.stem + "_sample_frame.png")
    contact_sheet_path = contact_sheet_path or output_path.with_name(output_path.stem + "_contact_sheet.jpg")
    df = _sort_detail(whole_df)
    writer = open_mp4_video_writer(output_path, int(width), int(height), fps=int(fps))
    sample_frames: list[Image.Image] = []
    rendered = 0
    try:
        for _key, group in df.groupby(["Series", "PetriDish"], sort=False, dropna=False):
            primary_memory: np.ndarray | None = None
            lateral_memory: np.ndarray | None = None
            group = group.reset_index(drop=True)
            capture_indices = {0, len(group) // 2, max(0, len(group) - 1)}
            for idx, row in group.iterrows():
                mask_path = _display_mask_path(row)
                if mask_path.exists():
                    arr = _read_index_mask(mask_path)
                    primary_mask, lateral_mask, _root_mask = _class_masks(arr, root_class_ids)
                    if primary_memory is None or primary_memory.shape != primary_mask.shape:
                        primary_memory = np.zeros(primary_mask.shape, dtype=bool)
                        lateral_memory = np.zeros(lateral_mask.shape, dtype=bool)
                    assert lateral_memory is not None
                    primary_memory |= primary_mask
                    lateral_memory |= lateral_mask
                if primary_memory is None or lateral_memory is None:
                    continue
                frame = _whole_plate_frame(
                    row,
                    group,
                    int(idx),
                    primary_memory=primary_memory,
                    lateral_memory=lateral_memory,
                    width=int(width),
                    height=int(height),
                    root_class_ids=root_class_ids,
                )
                if rendered == 0:
                    frame.save(sample_frame_path)
                if idx in capture_indices and len(sample_frames) < 12:
                    sample_frames.append(frame.copy().resize((480, 270), Image.Resampling.LANCZOS))
                writer.append(np.asarray(frame, dtype=np.uint8))
                rendered += 1
    finally:
        writer.close()
    if sample_frames:
        cols = min(3, len(sample_frames))
        rows = int(math.ceil(len(sample_frames) / cols))
        sheet = Image.new("RGB", (cols * 480, rows * 270), BACKGROUND)
        for idx, img in enumerate(sample_frames):
            sheet.paste(img, ((idx % cols) * 480, (idx // cols) * 270))
        sheet.save(contact_sheet_path, quality=92)
    return output_path


def _compartment_frame(
    rows: pd.DataFrame,
    plate_df: pd.DataFrame,
    current_idx: int,
    *,
    width: int,
    height: int,
    root_class_ids: tuple[int, ...],
) -> Image.Image:
    first = rows.iloc[0]
    frame = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(frame)
    left_w = width // 2
    draw.rectangle((0, 0, left_w, height), fill=PANEL_DARK)
    image_path = Path(str(first.get("SourceFile", "")))
    mask_path = _display_mask_path(first)
    if image_path.exists():
        image = Image.open(image_path).convert("RGB")
    elif mask_path.exists():
        arr = _read_index_mask(mask_path)
        image = Image.fromarray(np.full((arr.shape[0], arr.shape[1], 3), 245, dtype=np.uint8), "RGB")
    else:
        draw.text((48, height // 2), "Missing source image and mask", fill=(245, 245, 245), font=FONT_TITLE)
        return frame
    arr = _read_index_mask(mask_path) if mask_path.exists() else np.zeros((image.height, image.width), dtype=np.uint8)
    primary_mask, lateral_mask, root_mask = _class_masks(arr, root_class_ids)
    shoot_mask = arr == np.uint8(DEFAULT_SHOOT_CLASS_IDS[0])
    if int(np.count_nonzero(shoot_mask)) <= 0:
        shoot_mask, _meta = _green_shoot_mask(image_path, tuple(int(v) for v in arr.shape[:2]), root_mask=root_mask)
    src_w, src_h = image.size
    dst_w, dst_h = _fit_dims(src_w, src_h, left_w - 32, height - 36)
    x0, y0 = 16 + (left_w - 32 - dst_w) // 2, 18 + (height - 36 - dst_h) // 2
    panel = _blend_overlay(
        image,
        [
            (_dilate_bool(primary_mask, 1), ROOT_PRIMARY_COLOR, 0.50),
            (_dilate_bool(lateral_mask, 1), ROOT_LATERAL_COLOR, 0.50),
            (_dilate_bool(shoot_mask, 1), SHOOT_COLOR, 0.60),
        ],
        (dst_w, dst_h),
    )
    frame.paste(panel, (x0, y0))
    scale_x = dst_w / max(1.0, float(src_w))
    for _, row in rows.iterrows():
        lane_x = int(round(x0 + _cell_float(row.get("lane_x0")) * scale_x))
        draw.line((lane_x, y0, lane_x, y0 + dst_h), fill=(255, 255, 255), width=2)
    if not rows.empty:
        lane_x = int(round(x0 + _cell_float(rows.iloc[-1].get("lane_x1"), src_w) * scale_x))
        draw.line((lane_x, y0, lane_x, y0 + dst_h), fill=(255, 255, 255), width=2)
    label = f"{first.get('Series', '')} / {first.get('PetriDish', '')} | {_display_timestamp(first.get('Timestamp'))}"
    draw.rounded_rectangle((24, 26, min(left_w - 24, 38 + len(label) * 8), 58), radius=8, fill=(0, 0, 0))
    draw.text((34, 34), label, fill=(250, 250, 250), font=FONT_SMALL_BOLD)
    draw.rounded_rectangle((24, height - 118, 475, height - 26), radius=8, fill=(0, 0, 0))
    draw.text((40, height - 101), "Best-effort seedling compartments", fill=(238, 238, 238), font=FONT_SMALL_BOLD)
    draw.text((40, height - 74), "White lanes assign roots/shoots by stable plate position", fill=(238, 238, 238), font=FONT_SMALL)
    shoot_source = (
        rows["shoot_green_filter_source"].iloc[0]
        if "shoot_green_filter_source" in rows.columns and not rows.empty
        else "rgb_green_only"
    )
    draw.text(
        (40, height - 48),
        f"Pink is {_shoot_source_label(shoot_source)}",
        fill=(238, 238, 238),
        font=FONT_SMALL,
    )

    draw.rectangle((left_w, 0, width, height), fill=BACKGROUND)
    draw.line((left_w, 0, left_w, height), fill=(18, 18, 18), width=4)
    draw.text((left_w + 108, 38), "Compartmentalized Seedling Root Length", fill=TEXT_DARK, font=FONT_TITLE)
    draw.text((left_w + 250, 72), str(first.get("PetriDish", "")), fill=(60, 65, 74), font=FONT_SUBTITLE)
    chart_box = (left_w + 104, 130, width - 84, height - 248)
    x1, y1, x2, y2 = chart_box
    draw.rectangle(chart_box, outline=(95, 95, 95), width=2)
    max_val = max(10.0, float(pd.to_numeric(plate_df["occlusion_aware_total_root_length_mm"], errors="coerce").fillna(0.0).max()))
    y_max = max_val * 1.12
    n = max(1, plate_df[["Timestamp", "FrameIndex"]].drop_duplicates().shape[0])
    x_positions = [(x1 + x2) // 2] if n <= 1 else [x1 + int(round((x2 - x1) * i / (n - 1))) for i in range(n)]
    for i in range(6):
        y = y2 - int(round((y2 - y1) * i / 5.0))
        draw.line((x1, y, x2, y), fill=(225, 229, 232), width=1)
        draw.text((x1 - 58, y - 7), f"{y_max * i / 5.0:.0f}", fill=(80, 80, 80), font=FONT_TINY)
    colors = [(245, 142, 69), (94, 144, 255), (225, 88, 120), (199, 107, 219), (82, 186, 105)]
    plant_ids = sorted(plate_df["plant_id"].astype(str).unique())
    for plant_idx, plant_id in enumerate(plant_ids):
        plant = plate_df[plate_df["plant_id"].astype(str) == plant_id].reset_index(drop=True)
        vals = pd.to_numeric(plant["occlusion_aware_total_root_length_mm"], errors="coerce").fillna(0.0).astype(float).tolist()
        pts = []
        for idx, value in enumerate(vals[: current_idx + 1]):
            if idx >= len(x_positions):
                break
            pts.append((x_positions[idx], y2 - int(round((float(value) / max(1.0, y_max)) * (y2 - y1)))))
        color = colors[plant_idx % len(colors)]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=3)
        for x, y in pts:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    legend_w = 258
    legend_h = 44 + 24 * min(len(plant_ids), len(colors))
    legend_x = x2 - legend_w - 14
    legend_y = y1 + 14
    draw.rounded_rectangle(
        (legend_x, legend_y, legend_x + legend_w, legend_y + legend_h),
        radius=8,
        fill=(255, 255, 255),
        outline=(214, 219, 225),
    )
    draw.text((legend_x + 14, legend_y + 10), "Colored lines: per-seedling carried root", fill=TEXT_DARK, font=FONT_SMALL_BOLD)
    for plant_idx, plant_id in enumerate(plant_ids[: len(colors)]):
        y = legend_y + 38 + plant_idx * 24
        color = colors[plant_idx % len(colors)]
        draw.line((legend_x + 16, y + 8, legend_x + 48, y + 8), fill=color, width=4)
        draw.ellipse((legend_x + 29, y + 4, legend_x + 37, y + 12), fill=color)
        draw.text((legend_x + 58, y), plant_id, fill=(45, 50, 58), font=FONT_SMALL)
    draw.text((x1, y2 + 12), "X-axis: frame time   Y-axis: occlusion-aware root length (mm)", fill=(92, 98, 106), font=FONT_SMALL)
    stats_y = height - 150
    total_occl = float(pd.to_numeric(rows["occlusion_aware_total_root_length_mm"], errors="coerce").fillna(0.0).sum())
    total_visible = float(pd.to_numeric(rows["visible_total_root_length_mm"], errors="coerce").fillna(0.0).sum())
    total_shoot = float(pd.to_numeric(rows["shoot_area_green_only_mm2"], errors="coerce").fillna(0.0).sum())
    stats = [
        ("Visible sum", f"{total_visible:.1f} mm"),
        ("Carried sum", f"{total_occl:.1f} mm"),
        ("Hidden estimate", f"{max(0.0, total_occl - total_visible):.1f} mm"),
        ("Shoot area", f"{total_shoot:.2f} mm2"),
        ("Frame", f"{current_idx + 1}/{n}"),
    ]
    draw.rounded_rectangle((left_w + 60, stats_y, width - 60, height - 36), radius=12, fill=(246, 248, 250), outline=(218, 224, 230))
    col_w = (width - left_w - 140) // len(stats)
    for idx, (name, value) in enumerate(stats):
        sx = left_w + 80 + idx * col_w
        draw.text((sx, stats_y + 18), name, fill=(92, 98, 106), font=FONT_SMALL)
        draw.text((sx, stats_y + 48), value, fill=TEXT_DARK, font=FONT_VALUE)
    return frame


def generate_compartment_occlusion_video(
    compartment_df: pd.DataFrame,
    output_path: Path,
    *,
    root_class_ids: tuple[int, ...] = DEFAULT_ROOT_CLASS_IDS,
    fps: int = 10,
    width: int = 1920,
    height: int = 1080,
    sample_frame_path: Path | None = None,
    contact_sheet_path: Path | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_frame_path = sample_frame_path or output_path.with_name(output_path.stem + "_sample_frame.png")
    contact_sheet_path = contact_sheet_path or output_path.with_name(output_path.stem + "_contact_sheet.jpg")
    df = _sort_detail(compartment_df)
    writer = open_mp4_video_writer(output_path, int(width), int(height), fps=int(fps))
    sample_frames: list[Image.Image] = []
    rendered = 0
    try:
        for _key, plate_group in df.groupby(["Series", "PetriDish"], sort=False, dropna=False):
            unique_frames = (
                plate_group[["Timestamp", "FrameIndex"]]
                .drop_duplicates()
                .sort_values(["Timestamp", "FrameIndex"], kind="mergesort")
                .reset_index(drop=True)
            )
            capture_indices = {0, len(unique_frames) // 2, max(0, len(unique_frames) - 1)}
            for idx, frame_key in unique_frames.iterrows():
                rows = plate_group[
                    (plate_group["Timestamp"] == frame_key["Timestamp"])
                    & (pd.to_numeric(plate_group["FrameIndex"], errors="coerce").fillna(-1).astype(int) == int(frame_key["FrameIndex"]))
                ]
                if rows.empty:
                    continue
                frame = _compartment_frame(
                    rows.sort_values("plant_id"),
                    plate_group,
                    int(idx),
                    width=int(width),
                    height=int(height),
                    root_class_ids=root_class_ids,
                )
                if rendered == 0:
                    frame.save(sample_frame_path)
                if idx in capture_indices and len(sample_frames) < 12:
                    sample_frames.append(frame.copy().resize((480, 270), Image.Resampling.LANCZOS))
                writer.append(np.asarray(frame, dtype=np.uint8))
                rendered += 1
    finally:
        writer.close()
    if sample_frames:
        cols = min(3, len(sample_frames))
        rows = int(math.ceil(len(sample_frames) / cols))
        sheet = Image.new("RGB", (cols * 480, rows * 270), BACKGROUND)
        for idx, img in enumerate(sample_frames):
            sheet.paste(img, ((idx % cols) * 480, (idx // cols) * 270))
        sheet.save(contact_sheet_path, quality=92)
    return output_path


def write_occlusion_aware_outputs(
    mask_total_df: pd.DataFrame,
    *,
    output_dir: Path,
    ownership_df: pd.DataFrame | None = None,
    root_class_ids: tuple[int, ...] = DEFAULT_ROOT_CLASS_IDS,
    plant_count: int = DEFAULT_PLANT_COUNT,
    generate_videos: bool = True,
    fps: int = 10,
    width: int = 1920,
    height: int = 1080,
) -> OcclusionAwareOutput:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_frame_df = apply_accepted_shoots_to_frame_rows(mask_total_df, ownership_df)
    whole = build_whole_plate_occlusion_dataframe(accepted_frame_df, root_class_ids=root_class_ids)
    comp = build_compartmentalized_occlusion_dataframe(
        whole,
        ownership_df=ownership_df,
        root_class_ids=root_class_ids,
        plant_count=int(plant_count),
    )
    whole_summary = build_whole_plate_summary(whole)
    comp_summary = build_compartment_summary(comp)
    comp_timelapse = build_compartment_timelapse(comp)

    whole_csv = output_dir / "npec_occlusion_aware_whole_plate_roots_and_shoots.csv"
    whole_summary_csv = output_dir / "npec_occlusion_aware_whole_plate_summary.csv"
    whole_timelapse_csv = output_dir / "npec_occlusion_aware_whole_plate_timelapse.csv"
    comp_csv = output_dir / "npec_compartmentalized_seedling_occlusion_detail.csv"
    comp_summary_csv = output_dir / "npec_compartmentalized_seedling_occlusion_summary.csv"
    comp_timelapse_csv = output_dir / "npec_compartmentalized_seedling_occlusion_timelapse.csv"
    workbook_path = output_dir / "npec_occlusion_aware_all_assets.xlsx"
    metadata_json = output_dir / "npec_occlusion_aware_metadata.json"

    whole.to_csv(whole_csv, index=False)
    whole_summary.to_csv(whole_summary_csv, index=False)
    whole.to_csv(whole_timelapse_csv, index=False)
    comp.to_csv(comp_csv, index=False)
    comp_summary.to_csv(comp_summary_csv, index=False)
    comp_timelapse.to_csv(comp_timelapse_csv, index=False)
    with pd.ExcelWriter(workbook_path) as writer:
        whole.to_excel(writer, index=False, sheet_name="whole_plate_detail")
        whole_summary.to_excel(writer, index=False, sheet_name="whole_plate_summary")
        comp.to_excel(writer, index=False, sheet_name="compartment_detail")
        comp_summary.to_excel(writer, index=False, sheet_name="compartment_summary")
        comp_timelapse.to_excel(writer, index=False, sheet_name="compartment_timelapse")

    whole_video: Path | None = None
    comp_video: Path | None = None
    if bool(generate_videos):
        whole_video = generate_whole_plate_occlusion_video(
            whole,
            output_dir / "npec_occlusion_aware_whole_plate_timelapse_video.mp4",
            root_class_ids=root_class_ids,
            fps=int(fps),
            width=int(width),
            height=int(height),
        )
        comp_video = generate_compartment_occlusion_video(
            comp,
            output_dir / "npec_compartmentalized_seedling_occlusion_timelapse_video.mp4",
            root_class_ids=root_class_ids,
            fps=int(fps),
            width=int(width),
            height=int(height),
        )

    frozen_rows = 0
    combined_conflict_rows = 0
    if not comp.empty:
        if "occlusion_individual_state_frozen" in comp.columns:
            frozen_rows = int(comp["occlusion_individual_state_frozen"].map(lambda value: _cell_bool(value, False)).sum())
        if "combined_root_length_mm" in comp.columns:
            combined_conflict_rows = int(pd.to_numeric(comp["combined_root_length_mm"], errors="coerce").notna().sum())
    shoot_sources = (
        sorted(set(whole["shoot_green_filter_source"].dropna().astype(str)))
        if "shoot_green_filter_source" in whole.columns
        else []
    )
    shoot_method = (
        "Shoot area uses the segmentation mask class."
        if shoot_sources and all("mask" in value.lower() for value in shoot_sources)
        else "Shoot area uses literal RGB green-only pixels when available."
    )
    metadata = {
        "analysis_mode": "occlusion_aware_root_carry_forward",
        "root_class_ids": [int(v) for v in root_class_ids],
        "plant_count": int(plant_count),
        "input_rows": int(len(mask_total_df)),
        "whole_plate_rows": int(len(whole)),
        "compartment_rows": int(len(comp)),
        "video_generated": bool(generate_videos),
        "ownership_carry_forward": {
            "policy": "Only valid individual ownership measurements may advance a per-plant monotonic maximum.",
            "invalid_individual_rows_frozen": int(frozen_rows),
            "combined_conflict_measurement_rows_preserved": int(combined_conflict_rows),
            "row_status_column": "occlusion_carry_forward_status",
        },
        "method": (
            "Visible primary/lateral root masks are skeletonized by frame. "
            "Each plate and lane keeps a monotonic skeleton memory so roots observed before bacterial occlusion "
            "remain measurable in later frames. Ownership-backed per-plant maxima freeze on invalid conflict rows "
            "while combined conflict-group measurements remain in the detail output. "
            f"{shoot_method}"
        ),
        "shoot_measurement_sources": shoot_sources,
        "outputs": {
            "whole_plate_detail_csv": str(whole_csv),
            "whole_plate_summary_csv": str(whole_summary_csv),
            "whole_plate_timelapse_csv": str(whole_timelapse_csv),
            "compartment_detail_csv": str(comp_csv),
            "compartment_summary_csv": str(comp_summary_csv),
            "compartment_timelapse_csv": str(comp_timelapse_csv),
            "workbook": str(workbook_path),
            "whole_plate_video": str(whole_video) if whole_video is not None else None,
            "compartment_video": str(comp_video) if comp_video is not None else None,
        },
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return OcclusionAwareOutput(
        output_dir=output_dir,
        whole_plate_detail_csv=whole_csv,
        whole_plate_summary_csv=whole_summary_csv,
        whole_plate_timelapse_csv=whole_timelapse_csv,
        compartment_detail_csv=comp_csv,
        compartment_summary_csv=comp_summary_csv,
        compartment_timelapse_csv=comp_timelapse_csv,
        workbook_path=workbook_path,
        metadata_json=metadata_json,
        whole_plate_video=whole_video,
        compartment_video=comp_video,
    )


def _parse_classes(text: str | None, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not text:
        return fallback
    out: list[int] = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if value > 0 and value not in out:
            out.append(value)
    return tuple(out) if out else fallback


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate occlusion-aware NPEC root/shoot analysis assets.")
    parser.add_argument("--mask-total-csv", required=True, type=Path)
    parser.add_argument("--ownership-csv", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--root-classes", default="1,3")
    parser.add_argument("--plant-count", default=DEFAULT_PLANT_COUNT, type=int)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fps", default=10, type=int)
    parser.add_argument("--width", default=1920, type=int)
    parser.add_argument("--height", default=1080, type=int)
    args = parser.parse_args(argv)

    if not args.mask_total_csv.exists():
        raise FileNotFoundError(args.mask_total_csv)
    mask_df = pd.read_csv(args.mask_total_csv)
    ownership_df = None
    if args.ownership_csv and args.ownership_csv.exists():
        ownership_df = pd.read_csv(args.ownership_csv)
    result = write_occlusion_aware_outputs(
        mask_df,
        output_dir=args.output_dir,
        ownership_df=ownership_df,
        root_class_ids=_parse_classes(args.root_classes, DEFAULT_ROOT_CLASS_IDS),
        plant_count=int(args.plant_count),
        generate_videos=not bool(args.no_video),
        fps=int(args.fps),
        width=int(args.width),
        height=int(args.height),
    )
    latest = args.output_dir.parent / "npec_latest_occlusion_aware_reanalysis"
    if latest.exists() or latest.is_symlink():
        try:
            if latest.is_symlink() or latest.is_file():
                latest.unlink()
            else:
                shutil.rmtree(latest)
        except Exception:
            latest = None
    if latest is not None:
        try:
            latest.symlink_to(result.output_dir, target_is_directory=True)
        except Exception:
            pass
    print(json.dumps({"output_dir": str(result.output_dir), "metadata": str(result.metadata_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
