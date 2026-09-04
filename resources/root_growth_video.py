from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .analytics_engine import open_mp4_video_writer


PLANT_COLORS = {
    "plant_01": (245, 142, 69),
    "plant_02": (94, 144, 255),
    "plant_03": (225, 88, 120),
    "plant_04": (199, 107, 219),
    "plant_05": (82, 186, 105),
}
MASK_COLORS = {
    1: (255, 132, 42),
    2: (255, 44, 190),
    3: (112, 158, 238),
}
EXTRA_ROOT_COLORS = (
    (123, 203, 145),
    (196, 112, 218),
    (250, 189, 72),
)
DEFAULT_MASK_TOTAL_ROOT_CLASS_IDS = (1, 3)
DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS = (2,)


@dataclass(slots=True)
class LazyRootGrowthVideoConfig:
    fps: int = 1
    width: int = 1920
    height: int = 1080
    mask_only: bool = False
    mask_display_dilation_px: int = 1
    root_metric_mode: str = "total"
    root_class_ids: tuple[int, ...] = DEFAULT_MASK_TOTAL_ROOT_CLASS_IDS
    shoot_class_ids: tuple[int, ...] = DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS
    show_rgb_shoot_rescue: bool = False
    show_seedling_ownership_boxes: bool = True


def apply_accepted_shoots_to_frame_rows(
    frame_df: pd.DataFrame,
    ownership_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Use ownership-accepted shoot masks and areas for frame-level videos."""
    out = frame_df.copy()
    if out.empty or ownership_df is None or ownership_df.empty:
        return out

    key_column = next(
        (
            column
            for column in ("SourceFile", "PreviewImagePath")
            if column in out.columns and column in ownership_df.columns
        ),
        None,
    )
    if key_column is None or "OwnershipDisplayMaskPath" not in ownership_df.columns:
        return out

    ownership = ownership_df.copy()
    ownership["_frame_key"] = ownership[key_column].fillna("").astype(str).str.strip()
    ownership["_display_path"] = (
        ownership["OwnershipDisplayMaskPath"].fillna("").astype(str).str.strip()
    )
    ownership = ownership[
        ownership["_frame_key"].ne("")
        & ownership["_display_path"].ne("")
        & ownership["_display_path"].str.lower().ne("nan")
    ].copy()
    if ownership.empty:
        return out

    shoot_px_source = (
        ownership["shoot_area_px"]
        if "shoot_area_px" in ownership.columns
        else pd.Series(0.0, index=ownership.index)
    )
    shoot_mm2_source = (
        ownership["shoot_area_mm2"]
        if "shoot_area_mm2" in ownership.columns
        else pd.Series(0.0, index=ownership.index)
    )
    ownership["_shoot_px"] = pd.to_numeric(shoot_px_source, errors="coerce").fillna(0.0)
    ownership["_shoot_mm2"] = pd.to_numeric(shoot_mm2_source, errors="coerce").fillna(0.0)

    summary_rows: list[dict[str, object]] = []
    for frame_key, rows in ownership.groupby("_frame_key", sort=False, dropna=False):
        display_paths = [
            value
            for value in rows["_display_path"].astype(str).tolist()
            if value and value.lower() != "nan"
        ]
        if not display_paths:
            continue
        sources: list[str] = []
        if "shoot_measurement_source" in rows.columns:
            for value in rows["shoot_measurement_source"].fillna("").astype(str):
                value = value.strip()
                if value and value.lower() != "nan" and value not in sources:
                    sources.append(value)
        if not sources:
            sources.append("ownership_accepted_shoots")
        summary_rows.append(
            {
                "_frame_key": str(frame_key),
                "OwnershipDisplayMaskPath": display_paths[0],
                "shoot_area_px": int(round(float(rows["_shoot_px"].sum()))),
                "shoot_area_mm2": float(rows["_shoot_mm2"].sum()),
                "shoot_accepted_component_count": int((rows["_shoot_px"] > 0).sum()),
                "shoot_measurement_source": ",".join(sources),
            }
        )
    if not summary_rows:
        return out

    summary = pd.DataFrame(summary_rows).set_index("_frame_key")
    frame_keys = out[key_column].fillna("").astype(str).str.strip()
    matched = frame_keys.isin(summary.index)
    if not bool(matched.any()):
        return out

    existing_display = (
        out["OwnershipDisplayMaskPath"].copy()
        if "OwnershipDisplayMaskPath" in out.columns
        else pd.Series("", index=out.index, dtype=object)
    )
    mapped_display = frame_keys.map(summary["OwnershipDisplayMaskPath"])
    out["OwnershipDisplayMaskPath"] = mapped_display.where(matched, existing_display)

    for column in ("shoot_area_px", "shoot_area_mm2", "shoot_accepted_component_count"):
        mapped = frame_keys.map(summary[column])
        if column not in out.columns:
            out[column] = 0
        out.loc[matched, column] = mapped.loc[matched].to_numpy()

    # Occlusion-aware renderers use the historical green-only names. They now
    # represent the accepted shoot union for BW plates as well.
    green_px_source = (
        out["shoot_area_green_only_px"]
        if "shoot_area_green_only_px" in out.columns
        else out["shoot_area_px"]
        if "shoot_area_px" in out.columns
        else pd.Series(0.0, index=out.index)
    )
    green_mm2_source = (
        out["shoot_area_green_only_mm2"]
        if "shoot_area_green_only_mm2" in out.columns
        else out["shoot_area_mm2"]
        if "shoot_area_mm2" in out.columns
        else pd.Series(0.0, index=out.index)
    )
    out["shoot_area_green_only_px"] = pd.to_numeric(green_px_source, errors="coerce").fillna(0.0)
    out["shoot_area_green_only_mm2"] = pd.to_numeric(green_mm2_source, errors="coerce").fillna(0.0)
    out.loc[matched, "shoot_area_green_only_px"] = frame_keys.map(
        summary["shoot_area_px"]
    ).loc[matched].to_numpy()
    out.loc[matched, "shoot_area_green_only_mm2"] = frame_keys.map(
        summary["shoot_area_mm2"]
    ).loc[matched].to_numpy()

    mapped_source = frame_keys.map(summary["shoot_measurement_source"])
    for column in ("shoot_measurement_source", "shoot_green_filter_source"):
        if column not in out.columns:
            out[column] = ""
        out.loc[matched, column] = mapped_source.loc[matched].to_numpy()
    out["shoot_accepted_ownership_applied"] = matched.astype(bool)
    return out


MASK_TOTAL_ROOT_METRIC_SPECS = {
    "total": {
        "column": "total_root_length_mm",
        "delta_column": "delta_total_root_length_mm",
        "label": "Total Root",
        "axis_label": "Total Root Length (mm)",
        "color": (16, 16, 16),
    },
    "weighted_total": {
        "column": "total_root_length_weighted_mm",
        "delta_column": "delta_total_root_length_weighted_mm",
        "label": "Weighted Total",
        "axis_label": "Weighted Root Length (mm)",
        "color": (16, 16, 16),
    },
    "primary": {
        "column": "primary_root_length_mm",
        "delta_column": "delta_primary_root_length_mm",
        "label": "Primary Root",
        "axis_label": "Primary Root Length (mm)",
        "color": MASK_COLORS[1],
    },
    "lateral": {
        "column": "lateral_root_length_mm",
        "delta_column": "delta_lateral_root_length_mm",
        "label": "Lateral Root",
        "axis_label": "Lateral Root Length (mm)",
        "color": MASK_COLORS[3],
    },
}

OWNERSHIP_ROOT_METRIC_SPECS = {
    "total": {
        "column": "total_root_length_mm",
        "mean_column": "mean_root_length_mm",
        "per_plant_candidates": ("total_root_length_mm_clean", "total_root_length_mm_raw", "root_length_mm_clean", "root_length_mm_raw"),
        "label": "Total Roots",
        "axis_label": "Root Length (mm)",
        "color": (16, 16, 16),
    },
    "weighted_total": {
        "column": "total_root_length_weighted_mm",
        "mean_column": "mean_root_length_weighted_mm",
        "per_plant_candidates": (
            "total_root_length_weighted_mm_clean",
            "total_root_length_weighted_mm_raw",
            "total_root_length_weighted_mm",
            "ownership_total_root_length_weighted_mm",
            "total_root_length_mm_clean",
            "total_root_length_mm_raw",
            "root_length_mm_clean",
            "root_length_mm_raw",
        ),
        "label": "Weighted Total",
        "axis_label": "Weighted Root Length (mm)",
        "color": (16, 16, 16),
    },
    "primary": {
        "column": "total_primary_root_length_mm",
        "mean_column": "mean_primary_root_length_mm",
        "per_plant_candidates": ("primary_root_length_mm_clean", "primary_root_length_mm_raw", "root_length_mm_clean", "root_length_mm_raw"),
        "label": "Primary Roots",
        "axis_label": "Primary Root Length (mm)",
        "color": MASK_COLORS[1],
    },
    "lateral": {
        "column": "total_lateral_root_length_mm",
        "mean_column": "mean_lateral_root_length_mm",
        "per_plant_candidates": ("lateral_total_length_mm",),
        "label": "Lateral Roots",
        "axis_label": "Lateral Root Length (mm)",
        "color": MASK_COLORS[3],
    },
}


def _normalize_mask_total_root_metric_mode(value: object) -> str:
    mode = str(value or "total").strip().lower()
    return mode if mode in MASK_TOTAL_ROOT_METRIC_SPECS else "total"


def _mask_total_root_metric_spec(config: LazyRootGrowthVideoConfig) -> tuple[str, dict[str, object]]:
    mode = _normalize_mask_total_root_metric_mode(config.root_metric_mode)
    return mode, MASK_TOTAL_ROOT_METRIC_SPECS[mode]


def _ownership_root_metric_spec(config: LazyRootGrowthVideoConfig) -> tuple[str, dict[str, object]]:
    mode = _normalize_mask_total_root_metric_mode(config.root_metric_mode)
    return mode, OWNERSHIP_ROOT_METRIC_SPECS[mode]


def _first_existing_name(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _cell_is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return value is None


def _cell_bool(value: object, default: bool = False) -> bool:
    if _cell_is_missing(value):
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    return bool(default)


def _cell_float(value: object, default: float = 0.0) -> float:
    if _cell_is_missing(value):
        return float(default)
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _row_first_text(row: pd.Series, columns: tuple[str, ...], default: str = "") -> str:
    for column in columns:
        if column not in row.index:
            continue
        value = row.get(column)
        if _cell_is_missing(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return str(default)


def _series_any_truthy(df: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    for column in columns:
        if column not in df.columns:
            continue
        if any(_cell_bool(value, default=False) for value in df[column].tolist()):
            return True
    return False


def _series_unique_text(df: pd.DataFrame, columns: tuple[str, ...], default: str = "") -> str:
    values: list[str] = []
    for column in columns:
        if column not in df.columns:
            continue
        for value in df[column].tolist():
            if _cell_is_missing(value):
                continue
            text = str(value).strip()
            if text and text.lower() != "nan" and text not in values:
                values.append(text)
    return ",".join(values) if values else str(default)


def _friendly_source_label(source: object, *, green_enabled: bool = False) -> str:
    text = str(source or "").strip().lower()
    if bool(green_enabled) or text == "rgb_green_only":
        return "RGB green-only"
    source_parts = {
        part.strip()
        for part in text.split(",")
        if part.strip()
    }
    temporal_visual = "hades_bw_temporal_visual_reacquisition" in source_parts
    temporal_area_floor = "hades_bw_temporal_crown_track" in source_parts
    if temporal_visual or temporal_area_floor:
        tracking_label = (
            "temporal visual tracking" if temporal_visual else "temporal crown tracking"
        )
        return (
            f"Hades BW model + crown-local CV + {tracking_label}"
            if "hades_bw_crown_local_cv" in source_parts
            else f"Hades BW model + {tracking_label}"
        )
    if "hades_bw_crown_local_cv" in source_parts:
        model_parts = source_parts - {"hades_bw_crown_local_cv"}
        return (
            "Hades BW model + crown-local CV"
            if model_parts
            else "Hades BW crown-local CV"
        )
    if text in {"explicit_shoot_class", "mask_class", "requested_classes"}:
        return "mask class"
    if not text:
        return "not recorded"
    return text.replace("_", " ")


def _ownership_row_review(row: pd.Series) -> tuple[str, list[str]]:
    explicit = _row_first_text(row, ("ownership_frame_review_status",), default="").strip().lower()
    explicit_reasons = _row_first_text(row, ("ownership_frame_review_reasons",), default="")
    reasons = [part for part in explicit_reasons.split(";") if part]
    if explicit in {"ok", "review", "fail"}:
        return explicit, reasons

    valid = _cell_bool(row.get("ownership_measurement_valid", row.get("ownership_valid", False)), default=False)
    root_present = True
    if any(column in row.index for column in ("ownership_root_present", "total_root_area_px", "root_area_px", "root_length_mm_clean", "root_length_mm_raw")):
        root_present = _cell_bool(row.get("ownership_root_present", True), default=True)
        root_present = root_present or _cell_float(row.get("total_root_area_px", row.get("root_area_px", 0.0))) > 0.0
        root_present = root_present or _cell_float(row.get("total_root_length_mm_clean", row.get("root_length_mm_clean", 0.0))) > 0.0
        root_present = root_present or _cell_float(row.get("total_root_length_mm_raw", row.get("root_length_mm_raw", 0.0))) > 0.0
    shoot_present = True
    if any(column in row.index for column in ("ownership_shoot_present", "shoot_area_px", "shoot_area_mm2")):
        shoot_present = _cell_bool(row.get("ownership_shoot_present", False), default=False)
        shoot_present = shoot_present or _cell_float(row.get("shoot_area_px", 0.0)) > 0.0
        shoot_present = shoot_present or _cell_float(row.get("shoot_area_mm2", 0.0)) > 0.0
    conflict = _cell_bool(row.get("ownership_conflict", False), default=False)
    conflict = conflict or _cell_float(row.get("conflict_group_size", 1.0), default=1.0) > 1.0
    conflict = conflict or _cell_bool(row.get("conflict_touching_now", False), default=False)

    if not valid:
        reasons.append("invalid_ownership_measurement")
    if not root_present:
        reasons.append("root_missing")
    if not shoot_present:
        reasons.append("shoot_missing")
    if conflict:
        reasons.append("seedling_conflict")
    if (not valid) or (not root_present):
        return "fail", reasons
    if reasons:
        return "review", reasons
    return "ok", reasons


def _ownership_frame_qa(rows: pd.DataFrame) -> dict[str, object]:
    if rows is None or rows.empty:
        return {"status": "ok", "fail": 0, "review": 0, "ok": 0, "reasons": "", "shoot_source": "not recorded", "green_only": False}
    fail = 0
    review = 0
    ok = 0
    reasons: list[str] = []
    for _, row in rows.iterrows():
        status, row_reasons = _ownership_row_review(row)
        if status == "fail":
            fail += 1
        elif status == "review":
            review += 1
        else:
            ok += 1
        for reason in row_reasons:
            if reason and reason not in reasons:
                reasons.append(reason)
    green_only = _series_any_truthy(rows, ("shoot_rgb_green_only_enabled", "ownership_anchor_rgb_green_only_enabled"))
    shoot_source = _series_unique_text(rows, ("shoot_measurement_source", "ownership_shoot_mask_source"), default="")
    status = "fail" if fail > 0 else "review" if review > 0 else "ok"
    return {
        "status": status,
        "fail": int(fail),
        "review": int(review),
        "ok": int(ok),
        "reasons": ";".join(reasons),
        "shoot_source": _friendly_source_label(shoot_source, green_enabled=green_only),
        "green_only": bool(green_only),
    }


def _ownership_plate_qa(rows: pd.DataFrame) -> dict[str, object]:
    if rows is None or rows.empty:
        return {"frames_with_review": 0, "frames_with_fail": 0, "tracks_with_review": 0, "shoot_source": "not recorded", "green_only": False}
    frame_columns = [column for column in ("TimestampParsed", "FrameIndex") if column in rows.columns]
    frames_with_review = 0
    frames_with_fail = 0
    if frame_columns:
        for _key, group in rows.groupby(frame_columns, sort=False, dropna=False):
            qa = _ownership_frame_qa(group)
            if qa["status"] in {"review", "fail"}:
                frames_with_review += 1
            if qa["status"] == "fail":
                frames_with_fail += 1
    else:
        qa = _ownership_frame_qa(rows)
        frames_with_review = int(qa["status"] in {"review", "fail"})
        frames_with_fail = int(qa["status"] == "fail")

    tracks_with_review = 0
    if "plant_id" in rows.columns:
        for _track_id, group in rows.groupby("plant_id", sort=False, dropna=False):
            statuses = [_ownership_row_review(row)[0] for _, row in group.iterrows()]
            if any(status in {"review", "fail"} for status in statuses):
                tracks_with_review += 1
    green_only = _series_any_truthy(rows, ("shoot_rgb_green_only_enabled", "ownership_anchor_rgb_green_only_enabled"))
    shoot_source = _series_unique_text(rows, ("shoot_measurement_source", "ownership_shoot_mask_source"), default="")
    return {
        "frames_with_review": int(frames_with_review),
        "frames_with_fail": int(frames_with_fail),
        "tracks_with_review": int(tracks_with_review),
        "shoot_source": _friendly_source_label(shoot_source, green_enabled=green_only),
        "green_only": bool(green_only),
    }


def _ownership_rows_shoot_source(
    rows: pd.DataFrame,
    config: LazyRootGrowthVideoConfig,
) -> tuple[str, bool]:
    source_columns = ("shoot_measurement_source", "ownership_shoot_mask_source")
    flag_columns = (
        "shoot_rgb_green_only_enabled",
        "ownership_anchor_rgb_green_only_enabled",
    )
    source = _series_unique_text(rows, source_columns, default="")
    has_source = bool(str(source).strip())
    has_flag = any(
        column in rows.columns
        and rows[column].map(lambda value: not _cell_is_missing(value)).any()
        for column in flag_columns
    )
    green_only = _series_any_truthy(rows, flag_columns)
    green_only = green_only or str(source).strip().lower() == "rgb_green_only"
    if not has_source and not has_flag:
        green_only = bool(config.show_rgb_shoot_rescue)
    return _friendly_source_label(source, green_enabled=green_only), bool(green_only)


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    status: str = "ok",
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont = None,  # type: ignore[assignment]
) -> tuple[int, int, int, int]:
    font = font or FONT_SMALL_BOLD
    palette = {
        "ok": (30, 122, 70),
        "review": (172, 111, 18),
        "fail": (181, 54, 54),
        "info": (28, 80, 132),
    }
    fill = palette.get(str(status).lower(), palette["info"])
    x, y = int(xy[0]), int(xy[1])
    bbox = draw.textbbox((0, 0), text, font=font)
    w = int(bbox[2] - bbox[0])
    h = int(bbox[3] - bbox[1])
    rect = (x, y, x + w + 18, y + h + 12)
    draw.rounded_rectangle(rect, radius=7, fill=fill)
    draw.text((x + 9, y + 6), text, fill=(255, 255, 255), font=font)
    return rect


def _mask_total_row_shoot_source(row: pd.Series, config: LazyRootGrowthVideoConfig) -> tuple[str, bool]:
    source = _row_first_text(row, ("shoot_measurement_source",), default="")
    has_source = bool(str(source).strip())
    has_flag = (
        "shoot_rgb_rescue_enabled" in row.index
        and not _cell_is_missing(row.get("shoot_rgb_rescue_enabled"))
    )
    green_only = _cell_bool(row.get("shoot_rgb_rescue_enabled", False), default=False)
    green_only = green_only or str(source).strip().lower() == "rgb_green_only"
    if not has_source and not has_flag:
        green_only = bool(config.show_rgb_shoot_rescue)
    return _friendly_source_label(source, green_enabled=green_only), bool(green_only)


def _mask_total_plate_shoot_source(rows: pd.DataFrame, config: LazyRootGrowthVideoConfig) -> tuple[str, bool]:
    if rows is None or rows.empty:
        return _friendly_source_label("", green_enabled=bool(config.show_rgb_shoot_rescue)), bool(config.show_rgb_shoot_rescue)
    source = _series_unique_text(rows, ("shoot_measurement_source",), default="")
    has_source = bool(str(source).strip())
    has_flag = "shoot_rgb_rescue_enabled" in rows.columns
    green_only = _series_any_truthy(rows, ("shoot_rgb_rescue_enabled",))
    green_only = green_only or any(str(value).strip().lower() == "rgb_green_only" for value in rows.get("shoot_measurement_source", pd.Series(dtype=object)).tolist())
    if not has_source and not has_flag:
        green_only = bool(config.show_rgb_shoot_rescue)
    return _friendly_source_label(source, green_enabled=green_only), bool(green_only)


@dataclass(slots=True)
class LazyRootGrowthVideoResult:
    video_path: Path
    sample_frame_path: Path
    summary_csv_path: Path
    frames_rendered: int
    plates_rendered: int


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


FONT_TITLE = _font(26, True)
FONT_SUBTITLE = _font(19, False)
FONT_AXIS = _font(15, False)
FONT_SMALL = _font(13, False)
FONT_SMALL_BOLD = _font(13, True)
FONT_TINY = _font(11, False)
FONT_VALUE = _font(22, True)
FONT_VALUE_COMPACT = _font(18, True)


def _natural_sort_key(value: object) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(\d+)", str(value))
    key: list[tuple[int, object]] = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        elif part:
            key.append((1, part.lower()))
    return tuple(key)


def _parse_timestamp(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    return pd.Timestamp("1970-01-01") if pd.isna(ts) else pd.Timestamp(ts)


def _display_timestamp(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "Endpoint (no timestamp)"
    return pd.Timestamp(parsed).strftime("%Y-%m-%d %H:%M")


def _short_timestamp(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "endpoint"
    return pd.Timestamp(parsed).strftime("%m-%d")


def _display_timeline_position(
    value: object,
    position: int,
    frame_count: int,
) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if not pd.isna(parsed):
        return pd.Timestamp(parsed).strftime("%Y-%m-%d %H:%M")
    if int(frame_count) > 1:
        return f"Frame {int(position) + 1}/{int(frame_count)}"
    return "Endpoint (no timestamp)"


def _short_timeline_position(
    value: object,
    position: int,
    frame_count: int,
) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if not pd.isna(parsed):
        return pd.Timestamp(parsed).strftime("%m-%d")
    if int(frame_count) > 1:
        return f"F{int(position) + 1}"
    return "endpoint"


def _timeline_axis_title(values: object) -> str:
    series = pd.Series(list(values) if values is not None else [], dtype=object)
    if series.empty or pd.to_datetime(series, errors="coerce").isna().all():
        return "Frame"
    return "Timestamp"


def _parse_points(value: object) -> list[tuple[float, float]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    points: list[tuple[float, float]] = []
    for item in parsed if isinstance(parsed, list) else []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                points.append((float(item[0]), float(item[1])))
            except Exception:
                continue
    return points


def _fit_dims(src_w: int, src_h: int, box_w: int, box_h: int) -> tuple[int, int, float]:
    scale = min(box_w / float(src_w), box_h / float(src_h))
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    return dst_w, dst_h, scale


def _row_bbox(
    row: pd.Series,
    prefixes: tuple[str, ...],
) -> tuple[int, int, int, int] | None:
    for prefix in prefixes:
        values = tuple(
            int(round(_cell_float(row.get(f"{prefix}{name}", 0.0))))
            for name in ("x", "y", "w", "h")
        )
        if values[2] > 0 and values[3] > 0:
            return values
    return None


def _scaled_bbox(
    bbox: tuple[int, int, int, int],
    x0: int,
    y0: int,
    scale: float,
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    return (
        int(round(x0 + (float(x) * scale))),
        int(round(y0 + (float(y) * scale))),
        int(round(x0 + (float(x + width) * scale))),
        int(round(y0 + (float(y + height) * scale))),
    )


def _short_plant_label(plant_id: str) -> str:
    match = re.search(r"(\d+)$", str(plant_id))
    if match:
        return f"P{int(match.group(1))}"
    return str(plant_id) or "plant"


def _dashed_line(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    width: int = 2,
    dash: int = 8,
) -> None:
    x1, y1, x2, y2 = xy
    length = max(1.0, float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    distance = 0.0
    while distance < length:
        start = distance
        end = min(length, distance + dash)
        draw.line(
            (
                int(round(x1 + dx * start)),
                int(round(y1 + dy * start)),
                int(round(x1 + dx * end)),
                int(round(y1 + dy * end)),
            ),
            fill=fill,
            width=width,
        )
        distance += dash * 1.8


def _dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    width: int = 2,
    dash: int = 8,
) -> None:
    x0, y0, x1, y1 = bbox
    _dashed_line(draw, (x0, y0, x1, y0), fill, width=width, dash=dash)
    _dashed_line(draw, (x1, y0, x1, y1), fill, width=width, dash=dash)
    _dashed_line(draw, (x1, y1, x0, y1), fill, width=width, dash=dash)
    _dashed_line(draw, (x0, y1, x0, y0), fill, width=width, dash=dash)


def _resolve_image_path(group: pd.DataFrame) -> Path | None:
    for column in ("SourceFile", "PreviewImagePath"):
        if column in group.columns:
            for value in group[column].dropna().astype(str):
                path = Path(value)
                if path.exists() and not path.name.endswith("_mask.png"):
                    return path
    return None


def _resolve_mask_path(group: pd.DataFrame) -> Path | None:
    for column in ("OwnershipDisplayMaskPath", "OutputMaskPath"):
        if column not in group.columns:
            continue
        for value in group[column].dropna().astype(str):
            path = Path(value)
            if path.exists():
                return path
    return None


def _resolve_mask_path_from_row(row: pd.Series) -> Path | None:
    for column in ("OwnershipDisplayMaskPath", "OutputMaskPath"):
        value = str(row.get(column, "") or "").strip()
        if not value or value.lower() == "nan":
            continue
        path = Path(value)
        if path.is_file():
            return path
    return None


def _read_index_mask(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return np.asarray(mask, dtype=np.uint8)


def _dilate_bool(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, min(6, int(iterations)))):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(out, dtype=bool)
        for y_offset in range(3):
            for x_offset in range(3):
                expanded |= padded[y_offset : y_offset + out.shape[0], x_offset : x_offset + out.shape[1]]
        out = expanded
    return out


def _normalize_class_id_tuple(values: object, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return tuple(int(v) for v in fallback)
    raw_values = values
    if isinstance(values, str):
        raw_values = re.split(r"[,;\s]+", values.strip())
    out: list[int] = []
    try:
        iterator = iter(raw_values)  # type: ignore[arg-type]
    except TypeError:
        iterator = iter((raw_values,))
    for value in iterator:
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed <= 0 or parsed > 255 or parsed in out:
            continue
        out.append(parsed)
    return tuple(out) if out else tuple(int(v) for v in fallback)


def _mask_total_class_colors(
    root_class_ids: object = DEFAULT_MASK_TOTAL_ROOT_CLASS_IDS,
    shoot_class_ids: object = DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS,
) -> dict[int, tuple[int, int, int]]:
    root_ids = _normalize_class_id_tuple(root_class_ids, DEFAULT_MASK_TOTAL_ROOT_CLASS_IDS)
    shoot_ids = _normalize_class_id_tuple(shoot_class_ids, DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS)
    color_by_class: dict[int, tuple[int, int, int]] = {}
    for index, cls in enumerate(root_ids):
        if index == 0:
            color = MASK_COLORS[1]
        elif index == 1:
            color = MASK_COLORS[3]
        else:
            color = EXTRA_ROOT_COLORS[(index - 2) % len(EXTRA_ROOT_COLORS)]
        color_by_class[int(cls)] = color
    for cls in shoot_ids:
        color_by_class[int(cls)] = MASK_COLORS[2]
    return color_by_class


def _mask_total_legend_items(
    root_class_ids: object = DEFAULT_MASK_TOTAL_ROOT_CLASS_IDS,
    shoot_class_ids: object = DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS,
    *,
    show_rgb_shoot_rescue: bool = False,
) -> list[tuple[tuple[int, int, int], str]]:
    root_ids = _normalize_class_id_tuple(root_class_ids, DEFAULT_MASK_TOTAL_ROOT_CLASS_IDS)
    shoot_ids = _normalize_class_id_tuple(shoot_class_ids, DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS)
    colors = _mask_total_class_colors(root_ids, shoot_ids)
    items: list[tuple[tuple[int, int, int], str]] = []
    if root_ids:
        items.append((colors[int(root_ids[0])], f"primary/root class {root_ids[0]}"))
    if len(root_ids) >= 2:
        items.append((colors[int(root_ids[1])], f"lateral/root class {root_ids[1]}"))
    if len(root_ids) > 2:
        extra_label = ",".join(str(v) for v in root_ids[2:])
        items.append((colors[int(root_ids[2])], f"extra root classes {extra_label}"))
    shoot_label = ",".join(str(v) for v in shoot_ids)
    shoot_name = f"shoot classes {shoot_label}"
    if bool(show_rgb_shoot_rescue):
        shoot_name += " as RGB green-only"
    items.append((MASK_COLORS[2], shoot_name))
    return items


def _mask_total_dataframe_requests_rgb_shoot_rescue(total_df: pd.DataFrame) -> bool:
    if total_df is None or total_df.empty:
        return False
    if "shoot_rgb_rescue_enabled" in total_df.columns:
        enabled = total_df["shoot_rgb_rescue_enabled"]
        if enabled.dtype == bool:
            if bool(enabled.fillna(False).any()):
                return True
        else:
            normalized = enabled.fillna("").astype(str).str.strip().str.lower()
            if bool(normalized.isin({"1", "true", "yes", "on", "enabled"}).any()):
                return True
    if "shoot_measurement_source" in total_df.columns:
        sources = total_df["shoot_measurement_source"].fillna("").astype(str).str.strip().str.lower()
        if bool(sources.eq("rgb_green_only").any()):
            return True
    return False


def _colorized_mask(
    mask: np.ndarray,
    dilation_px: int = 0,
    *,
    root_class_ids: object = DEFAULT_MASK_TOTAL_ROOT_CLASS_IDS,
    shoot_class_ids: object = DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS,
) -> Image.Image:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    rgb = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 3), dtype=np.uint8)
    for cls, color in _mask_total_class_colors(root_class_ids, shoot_class_ids).items():
        pixels = mask_u8 == np.uint8(cls)
        if dilation_px > 0:
            pixels = _dilate_bool(pixels, dilation_px)
        rgb[pixels] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _mask_array_resized(
    mask: np.ndarray,
    size: tuple[int, int],
    dilation_px: int = 0,
    *,
    root_class_ids: object = DEFAULT_MASK_TOTAL_ROOT_CLASS_IDS,
    shoot_class_ids: object = DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS,
) -> Image.Image:
    resized = Image.fromarray(mask, mode="L").resize(size, Image.Resampling.NEAREST)
    return _colorized_mask(
        np.asarray(resized, dtype=np.uint8),
        dilation_px=dilation_px,
        root_class_ids=root_class_ids,
        shoot_class_ids=shoot_class_ids,
    )


def _mask_only_resized(
    mask_path: Path,
    size: tuple[int, int],
    dilation_px: int = 0,
    *,
    root_class_ids: object = DEFAULT_MASK_TOTAL_ROOT_CLASS_IDS,
    shoot_class_ids: object = DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS,
) -> Image.Image:
    return _mask_array_resized(
        _read_index_mask(mask_path),
        size,
        dilation_px=dilation_px,
        root_class_ids=root_class_ids,
        shoot_class_ids=shoot_class_ids,
    )


def _display_mask_with_rgb_shoot_rescue(
    mask: np.ndarray,
    image_path: Path,
    *,
    root_class_ids: object = DEFAULT_MASK_TOTAL_ROOT_CLASS_IDS,
    shoot_class_ids: object = DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS,
) -> np.ndarray:
    shoot_ids = _normalize_class_id_tuple(shoot_class_ids, DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS)
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if not shoot_ids:
        return mask_u8

    root_ids = _normalize_class_id_tuple(root_class_ids, DEFAULT_MASK_TOTAL_ROOT_CLASS_IDS)
    base_shoot = np.isin(mask_u8, np.asarray(shoot_ids, dtype=np.uint8))
    root_mask = np.isin(mask_u8, np.asarray(root_ids, dtype=np.uint8))

    def _without_model_shoot() -> np.ndarray:
        display = mask_u8.copy()
        display[base_shoot] = 0
        return display

    if not image_path.exists():
        return _without_model_shoot()

    try:
        image = Image.open(image_path).convert("RGB")
        if image.size != (mask_u8.shape[1], mask_u8.shape[0]):
            image = image.resize((mask_u8.shape[1], mask_u8.shape[0]), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.uint8)
        from .pyphenotyper_adapter import build_lucifer_green_shoot_mask

        color_mask, _meta = build_lucifer_green_shoot_mask(
            rgb,
            tuple(int(v) for v in mask_u8.shape[:2]),
            root_mask=root_mask.astype(np.uint8),
        )
    except Exception:
        return _without_model_shoot()

    candidate_pixels = int(np.count_nonzero(color_mask))
    min_pixels = max(20, int(round(float(mask_u8.shape[0] * mask_u8.shape[1]) * 0.000001)))
    if candidate_pixels < min_pixels:
        display = mask_u8.copy()
        display[base_shoot] = 0
        return display

    green_only = (np.asarray(color_mask, dtype=np.uint8) > 0) & (~root_mask)
    display = mask_u8.copy()
    display[base_shoot] = 0
    display[green_only] = np.uint8(shoot_ids[0])
    return display


def _overlay_mask(image: Image.Image, mask_path: Path | None, alpha: int = 82) -> Image.Image:
    if mask_path is None or not mask_path.exists():
        return image
    mask = _read_index_mask(mask_path)
    if mask.shape[:2] != (image.height, image.width):
        mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST))
    base = np.array(image.convert("RGB")).astype(np.float32)
    color_layer = np.zeros_like(base)
    alpha_layer = np.zeros(mask.shape, dtype=np.float32)
    for cls, color in MASK_COLORS.items():
        cls_pixels = mask == cls
        if np.any(cls_pixels):
            color_layer[cls_pixels] = np.array(color, dtype=np.float32)
            alpha_layer[cls_pixels] = alpha / 255.0
    blended = base * (1.0 - alpha_layer[..., None]) + color_layer * alpha_layer[..., None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def _overlay_mask_array_resized(
    image: Image.Image,
    mask: np.ndarray,
    size: tuple[int, int],
    alpha: int = 92,
    *,
    root_class_ids: object = DEFAULT_MASK_TOTAL_ROOT_CLASS_IDS,
    shoot_class_ids: object = DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS,
) -> Image.Image:
    resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(size, Image.Resampling.NEAREST), dtype=np.uint8)
    base = np.array(resized, dtype=np.float32)
    color_layer = np.zeros_like(base)
    alpha_layer = np.zeros(mask.shape, dtype=np.float32)
    for cls, color in _mask_total_class_colors(root_class_ids, shoot_class_ids).items():
        cls_pixels = mask == cls
        if np.any(cls_pixels):
            color_layer[cls_pixels] = np.array(color, dtype=np.float32)
            alpha_layer[cls_pixels] = alpha / 255.0
    blended = base * (1.0 - alpha_layer[..., None]) + color_layer * alpha_layer[..., None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def _overlay_mask_resized(
    image: Image.Image,
    mask_path: Path | None,
    size: tuple[int, int],
    alpha: int = 92,
    *,
    root_class_ids: object = DEFAULT_MASK_TOTAL_ROOT_CLASS_IDS,
    shoot_class_ids: object = DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS,
) -> Image.Image:
    if mask_path is None or not mask_path.exists():
        return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    return _overlay_mask_array_resized(
        image,
        _read_index_mask(mask_path),
        size,
        alpha=alpha,
        root_class_ids=root_class_ids,
        shoot_class_ids=shoot_class_ids,
    )


def _draw_plate_panel(
    frame: Image.Image,
    current_rows: pd.DataFrame,
    timeline_row: pd.Series,
    current_idx: int,
    frame_count: int,
    left_w: int,
    height: int,
    config: LazyRootGrowthVideoConfig,
) -> None:
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, left_w, height), fill=(20, 21, 22))
    image_path = _resolve_image_path(current_rows)
    mask_path = _resolve_mask_path(current_rows)
    plate = str(timeline_row["PetriDish"])
    timestamp = _display_timeline_position(
        timeline_row["Timestamp"],
        current_idx,
        frame_count,
    )
    qa = _ownership_frame_qa(current_rows)
    shoot_source, green_only = _ownership_rows_shoot_source(current_rows, config)
    qa["shoot_source"] = shoot_source
    qa["green_only"] = green_only
    mask_only = bool(config.mask_only)
    if image_path is None and not mask_only:
        draw.text((48, height // 2), f"Missing source image for {plate}", fill=(240, 240, 240), font=FONT_TITLE)
        return
    if mask_only and (mask_path is None or not mask_path.exists()):
        draw.text((48, height // 2), f"Missing mask for {plate}", fill=(240, 240, 240), font=FONT_TITLE)
        return

    if mask_only:
        assert mask_path is not None
        display_mask = _read_index_mask(mask_path)
        if green_only and image_path is not None and image_path.exists():
            display_mask = _display_mask_with_rgb_shoot_rescue(
                display_mask,
                image_path,
                root_class_ids=config.root_class_ids,
                shoot_class_ids=config.shoot_class_ids,
            )
        src_h, src_w = display_mask.shape[:2]
        display_dilation_px = 0 if green_only else max(0, int(config.mask_display_dilation_px))
        image = _colorized_mask(
            display_mask,
            dilation_px=display_dilation_px,
            root_class_ids=config.root_class_ids,
            shoot_class_ids=config.shoot_class_ids,
        )
    else:
        assert image_path is not None
        image = Image.open(image_path).convert("RGB")
        src_w, src_h = image.size
        if green_only and mask_path is not None and mask_path.exists():
            display_mask = _display_mask_with_rgb_shoot_rescue(
                _read_index_mask(mask_path),
                image_path,
                root_class_ids=config.root_class_ids,
                shoot_class_ids=config.shoot_class_ids,
            )
            image = _overlay_mask_array_resized(
                image,
                display_mask,
                (src_w, src_h),
                alpha=82,
                root_class_ids=config.root_class_ids,
                shoot_class_ids=config.shoot_class_ids,
            )
        else:
            image = _overlay_mask_resized(
                image,
                mask_path,
                (src_w, src_h),
                alpha=82,
                root_class_ids=config.root_class_ids,
                shoot_class_ids=config.shoot_class_ids,
            )
    box_x, box_y, box_w, box_h = 16, 18, left_w - 32, height - 36
    dst_w, dst_h, scale = _fit_dims(src_w, src_h, box_w, box_h)
    x0 = box_x + (box_w - dst_w) // 2
    y0 = box_y + (box_h - dst_h) // 2

    frame.paste(image.resize((dst_w, dst_h), Image.Resampling.LANCZOS), (x0, y0))

    if bool(config.show_seedling_ownership_boxes):
        crown_rows: list[tuple[str, float, float]] = []
        for _, crown_row in current_rows.iterrows():
            crown_x = _cell_float(
                crown_row.get("ownership_crown_center_x"),
                default=float("nan"),
            )
            crown_y = _cell_float(
                crown_row.get("ownership_crown_center_y"),
                default=float("nan"),
            )
            if np.isfinite(crown_x) and np.isfinite(crown_y):
                crown_rows.append((str(crown_row.get("plant_id", "")), crown_x, crown_y))
        crown_rows.sort(key=lambda item: item[1])
        expected_lane_boxes: dict[str, tuple[int, int, int, int]] = {}
        for position, (plant_id, crown_x, crown_y) in enumerate(crown_rows):
            left = (
                0
                if position == 0
                else int(round(0.5 * (crown_rows[position - 1][1] + crown_x)))
            )
            right = (
                src_w
                if position == len(crown_rows) - 1
                else int(round(0.5 * (crown_x + crown_rows[position + 1][1])))
            )
            lane_top = max(0, int(round(crown_y - (0.06 * float(src_h)))))
            lane_bottom = min(src_h, int(round(crown_y + (0.72 * float(src_h)))))
            expected_lane_boxes[plant_id] = (
                max(0, left),
                lane_top,
                max(1, min(src_w, right) - max(0, left)),
                max(1, lane_bottom - lane_top),
            )

        drawn_groups: set[tuple[object, ...]] = set()
        for _, row in current_rows.sort_values("plant_id").iterrows():
            plant_id = str(row.get("plant_id", ""))
            color = PLANT_COLORS.get(plant_id, (255, 255, 255))
            bbox = _row_bbox(row, ("seedling_bbox_", "bbox_"))
            if bbox is not None:
                scaled_bbox = _scaled_bbox(bbox, x0, y0, scale)
                draw.rectangle(scaled_bbox, outline=color, width=3)
                valid = _cell_bool(
                    row.get(
                        "ownership_measurement_valid",
                        row.get("ownership_valid", True),
                    ),
                    default=True,
                )
                label = _short_plant_label(plant_id)
                if not valid:
                    label += " ?"
                label_bbox = draw.textbbox((0, 0), label, font=FONT_SMALL_BOLD)
                label_w = int(label_bbox[2] - label_bbox[0]) + 10
                label_h = int(label_bbox[3] - label_bbox[1]) + 8
                label_x = max(x0, min(x0 + dst_w - label_w, scaled_bbox[0]))
                label_y = max(y0, min(y0 + dst_h - label_h, scaled_bbox[1]))
                draw.rectangle(
                    (label_x, label_y, label_x + label_w, label_y + label_h),
                    fill=color,
                )
                draw.text(
                    (label_x + 5, label_y + 4),
                    label,
                    fill=(12, 12, 12),
                    font=FONT_SMALL_BOLD,
                )
            else:
                expected_box = expected_lane_boxes.get(plant_id)
                if expected_box is not None:
                    scaled_expected_box = _scaled_bbox(
                        expected_box,
                        x0,
                        y0,
                        scale,
                    )
                    _dashed_rectangle(
                        draw,
                        scaled_expected_box,
                        color,
                        width=2,
                        dash=7,
                    )
                    label = f"{_short_plant_label(plant_id)} expected"
                    label_bbox = draw.textbbox(
                        (0, 0),
                        label,
                        font=FONT_SMALL_BOLD,
                    )
                    label_w = int(label_bbox[2] - label_bbox[0]) + 10
                    label_h = int(label_bbox[3] - label_bbox[1]) + 8
                    label_x = max(
                        x0,
                        min(x0 + dst_w - label_w, scaled_expected_box[0]),
                    )
                    label_y = max(
                        y0,
                        min(y0 + dst_h - label_h, scaled_expected_box[1]),
                    )
                    draw.rectangle(
                        (label_x, label_y, label_x + label_w, label_y + label_h),
                        fill=(28, 30, 34),
                        outline=color,
                        width=1,
                    )
                    draw.text(
                        (label_x + 5, label_y + 4),
                        label,
                        fill=color,
                        font=FONT_SMALL_BOLD,
                    )

            crown_x = _cell_float(row.get("ownership_crown_center_x"), default=float("nan"))
            crown_y = _cell_float(row.get("ownership_crown_center_y"), default=float("nan"))
            if np.isfinite(crown_x) and np.isfinite(crown_y):
                cx = int(round(x0 + crown_x * scale))
                cy = int(round(y0 + crown_y * scale))
                radius = 5
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    fill=color,
                    outline=(12, 12, 12),
                    width=1,
                )

            group_bbox = _row_bbox(row, ("combined_group_bbox_",))
            group_size = int(round(_cell_float(row.get("conflict_group_size", 1.0), default=1.0)))
            if group_bbox is None or group_size <= 1:
                continue
            group_id = str(row.get("conflict_group_id", "")).strip()
            group_key: tuple[object, ...] = (
                group_id,
                *group_bbox,
            )
            if group_key in drawn_groups:
                continue
            drawn_groups.add(group_key)
            scaled_group_bbox = _scaled_bbox(group_bbox, x0, y0, scale)
            draw.rectangle(
                scaled_group_bbox,
                outline=(232, 68, 68),
                width=5,
            )
            group_label = f"unresolved group ({group_size})"
            group_label_bbox = draw.textbbox(
                (0, 0),
                group_label,
                font=FONT_SMALL_BOLD,
            )
            group_label_w = int(group_label_bbox[2] - group_label_bbox[0]) + 12
            group_label_h = int(group_label_bbox[3] - group_label_bbox[1]) + 8
            group_label_x = max(
                x0,
                min(x0 + dst_w - group_label_w, scaled_group_bbox[0]),
            )
            group_label_y = max(
                y0,
                min(
                    y0 + dst_h - group_label_h,
                    scaled_group_bbox[1] - group_label_h,
                ),
            )
            draw.rectangle(
                (
                    group_label_x,
                    group_label_y,
                    group_label_x + group_label_w,
                    group_label_y + group_label_h,
                ),
                fill=(232, 68, 68),
            )
            draw.text(
                (group_label_x + 6, group_label_y + 4),
                group_label,
                fill=(255, 255, 255),
                font=FONT_SMALL_BOLD,
            )

    for _, row in current_rows.sort_values("plant_id").iterrows():
        plant_id = str(row.get("plant_id", ""))
        points = _parse_points(row.get("path_points"))
        if len(points) < 2:
            continue
        color = PLANT_COLORS.get(plant_id, (255, 255, 255))
        scaled = [(int(round(x0 + x * scale)), int(round(y0 + y * scale))) for x, y in points]
        draw.line(scaled, fill=color, width=3, joint="curve")
        end = scaled[-1]
        draw.ellipse((end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4), fill=color)

    label = f"{plate}  |  {timestamp}"
    text_box = draw.textbbox((0, 0), label, font=FONT_SMALL_BOLD)
    draw.rounded_rectangle((24, 26, 34 + text_box[2], 56), radius=8, fill=(0, 0, 0))
    draw.text((30, 32), label, fill=(250, 250, 250), font=FONT_SMALL_BOLD)
    qa_status = str(qa.get("status", "ok"))
    qa_label = f"QA {qa_status.upper()}  fail {int(qa.get('fail', 0))}  review {int(qa.get('review', 0))}"
    qa_rect = _draw_badge(draw, (24, 62), qa_label, status=qa_status, font=FONT_SMALL_BOLD)
    source_label = f"Shoots: {qa.get('shoot_source', 'not recorded')}"
    source_bbox = draw.textbbox((0, 0), source_label, font=FONT_SMALL_BOLD)
    source_w = int(source_bbox[2] - source_bbox[0]) + 18
    source_x = qa_rect[2] + 8
    source_y = 62
    if source_x + source_w > left_w - 24:
        source_x = 24
        source_y = qa_rect[3] + 6
    _draw_badge(draw, (source_x, source_y), source_label, status="info", font=FONT_SMALL_BOLD)

    legend_items = _mask_total_legend_items(
        config.root_class_ids,
        config.shoot_class_ids,
        show_rgb_shoot_rescue=green_only,
    )
    ownership_legend_height = 46 if bool(config.show_seedling_ownership_boxes) else 0
    legend_y = max(
        24,
        height - (88 + ownership_legend_height + (22 * len(legend_items))),
    )
    draw.rounded_rectangle((24, legend_y, 388, height - 26), radius=8, fill=(0, 0, 0))
    legend_title = (
        "Masks, paths + ownership"
        if bool(config.show_seedling_ownership_boxes)
        else "Mask classes + tracked paths"
    )
    draw.text((38, legend_y + 12), legend_title, fill=(245, 245, 245), font=FONT_SMALL_BOLD)
    for i, (color, name) in enumerate(legend_items):
        y = legend_y + 42 + i * 22
        draw.rectangle((40, y + 2, 56, y + 14), fill=color)
        draw.text((66, y - 1), name, fill=(236, 236, 236), font=FONT_SMALL)
    path_y = legend_y + 42 + len(legend_items) * 22
    for i, plant_id in enumerate(sorted(PLANT_COLORS)):
        x = 40 + i * 14
        draw.rectangle((x, path_y + 3, x + 10, path_y + 13), fill=PLANT_COLORS[plant_id])
    draw.text((118, path_y - 1), "centerline = plant ID", fill=(236, 236, 236), font=FONT_SMALL)
    if bool(config.show_seedling_ownership_boxes):
        box_y = path_y + 22
        draw.rectangle((40, box_y + 1, 58, box_y + 15), outline=(232, 68, 68), width=2)
        draw.text(
            (68, box_y - 1),
            "box = seedling ID; red = unresolved group",
            fill=(236, 236, 236),
            font=FONT_SMALL,
        )
        expected_y = box_y + 22
        _dashed_rectangle(
            draw,
            (40, expected_y + 1, 58, expected_y + 15),
            (170, 190, 215),
            width=2,
            dash=4,
        )
        draw.text(
            (68, expected_y - 1),
            "dashed = expected lane; no measured box",
            fill=(236, 236, 236),
            font=FONT_SMALL,
        )


def _plot_xy(
    values: list[float],
    x_positions: list[int],
    y_min: float,
    y_max: float,
    chart_box: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    x1, y1, x2, y2 = chart_box
    span = max(1.0, y_max - y_min)
    points: list[tuple[int, int]] = []
    for idx, value in enumerate(values):
        x = x_positions[idx]
        if not np.isfinite(float(value)):
            points.append((x, y2))
            continue
        y = y2 - int(round((float(value) - y_min) / span * (y2 - y1)))
        points.append((x, y))
    return points


def _draw_chart_panel(
    frame: Image.Image,
    plate_df: pd.DataFrame,
    plate_master: pd.DataFrame,
    current_idx: int,
    left_w: int,
    width: int,
    height: int,
    config: LazyRootGrowthVideoConfig,
) -> None:
    draw = ImageDraw.Draw(frame)
    draw.rectangle((left_w, 0, width, height), fill=(255, 255, 255))
    draw.line((left_w, 0, left_w, height), fill=(18, 18, 18), width=4)
    plate = str(plate_df["PetriDish"].iloc[0])
    root_metric_mode, root_metric_spec = _ownership_root_metric_spec(config)
    selected_root_column = str(root_metric_spec["column"])
    if selected_root_column not in plate_df.columns:
        selected_root_column = "selected_root_length_mm" if "selected_root_length_mm" in plate_df.columns else "total_root_length_mm"
    selected_delta_column = f"delta_{selected_root_column}"
    selected_mean_column = str(root_metric_spec["mean_column"])
    if selected_mean_column not in plate_df.columns:
        selected_mean_column = "mean_root_length_mm"
    selected_label = str(root_metric_spec["label"])
    selected_color = tuple(int(v) for v in root_metric_spec["color"])
    per_plant_metric_column = _first_existing_name(
        plate_master,
        tuple(str(v) for v in root_metric_spec["per_plant_candidates"]),
    ) or "root_length_mm_clean"
    chart_box = (left_w + 118, 135, width - 82, height - 300)
    cx1, cy1, cx2, cy2 = chart_box

    per_plant: dict[str, list[float]] = {}
    for plant_id in sorted(PLANT_COLORS):
        values: list[float] = []
        for _, row in plate_df.iterrows():
            rows = plate_master[
                (plate_master["TimestampParsed"] == row["TimestampParsed"])
                & (plate_master["FrameIndex"] == int(row["FrameIndex"]))
                & (plate_master["plant_id"] == plant_id)
            ]
            if rows.empty:
                values.append(float("nan"))
            else:
                valid_rows = rows
                validity_column = _first_existing_name(
                    rows,
                    ("ownership_measurement_valid", "ownership_valid"),
                )
                if validity_column is None:
                    valid_rows = rows.iloc[0:0]
                else:
                    valid_mask = rows[validity_column].map(lambda value: _cell_bool(value, default=False))
                    valid_rows = rows[valid_mask]
                numeric = pd.to_numeric(valid_rows[per_plant_metric_column], errors="coerce").dropna()
                values.append(float(numeric.max()) if not numeric.empty else float("nan"))
        per_plant[plant_id] = values
    selected_values = [float(v) for v in pd.to_numeric(plate_df[selected_root_column], errors="coerce")]
    finite_selected = [value for value in selected_values if np.isfinite(value)]
    max_val = max([0.0, *finite_selected, *[v for vals in per_plant.values() for v in vals if np.isfinite(v)]])
    y_min = 0.0
    y_max = max(10.0, max_val * 1.12)

    draw.text((left_w + 220, 38), f"{selected_label} Growth Over Time", fill=(18, 22, 28), font=FONT_TITLE)
    draw.text((left_w + 390, 72), f"Petridish: {plate}", fill=(48, 54, 62), font=FONT_SUBTITLE)
    draw.rectangle(chart_box, outline=(95, 95, 95), width=2)
    for i in range(6):
        y = cy2 - int(round((cy2 - cy1) * i / 5))
        value = y_min + (y_max - y_min) * i / 5
        draw.line((cx1, y, cx2, y), fill=(225, 229, 232), width=1)
        label = f"{value:.0f}"
        text_w = draw.textbbox((0, 0), label, font=FONT_AXIS)[2]
        draw.text((cx1 - text_w - 12, y - 8), label, fill=(80, 80, 80), font=FONT_AXIS)

    n = len(plate_df)
    x_positions = [(cx1 + cx2) // 2] if n <= 1 else [cx1 + int(round((cx2 - cx1) * i / (n - 1))) for i in range(n)]
    for x in x_positions:
        draw.line((x, cy1, x, cy2), fill=(237, 239, 241), width=1)

    for plant_id, values in per_plant.items():
        xy = _plot_xy(values[: current_idx + 1], x_positions[: current_idx + 1], y_min, y_max, chart_box)
        segment: list[tuple[int, int]] = []
        for point, value in zip(xy, values[: current_idx + 1]):
            if not np.isfinite(value):
                if len(segment) >= 2:
                    draw.line(segment, fill=PLANT_COLORS[plant_id], width=3)
                segment = []
                continue
            segment.append(point)
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=PLANT_COLORS[plant_id])
        if len(segment) >= 2:
            draw.line(segment, fill=PLANT_COLORS[plant_id], width=3)

    total_xy = _plot_xy(selected_values[: current_idx + 1], x_positions[: current_idx + 1], y_min, y_max, chart_box)
    previous_point: tuple[int, int] | None = None
    for point, value in zip(total_xy, selected_values[: current_idx + 1]):
        if not np.isfinite(value):
            previous_point = None
            continue
        if previous_point is not None:
            _dashed_line(
                draw,
                (previous_point[0], previous_point[1], point[0], point[1]),
                fill=selected_color,
                width=3,
                dash=10,
            )
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=selected_color)
        previous_point = point

    timestamps = plate_df["Timestamp"].tolist()
    for i, value in enumerate(timestamps):
        label_img = Image.new("RGBA", (70, 24), (255, 255, 255, 0))
        ld = ImageDraw.Draw(label_img)
        ld.text(
            (0, 2),
            _short_timeline_position(value, i, len(plate_df)),
            fill=(75, 75, 75),
            font=FONT_TINY,
        )
        label_img = label_img.rotate(36, expand=True, resample=Image.Resampling.BICUBIC)
        frame.paste(label_img, (x_positions[i] - 22, cy2 + 14), label_img)

    y_label = Image.new("RGBA", (220, 28), (255, 255, 255, 0))
    yd = ImageDraw.Draw(y_label)
    yd.text((0, 2), str(root_metric_spec["axis_label"]), fill=(40, 40, 40), font=FONT_AXIS)
    y_label = y_label.rotate(270, expand=True)
    frame.paste(y_label, (left_w + 28, (cy1 + cy2) // 2 - y_label.height // 2), y_label)
    axis_title = _timeline_axis_title(timestamps)
    axis_width = draw.textbbox((0, 0), axis_title, font=FONT_AXIS)[2]
    draw.text(
        ((cx1 + cx2 - axis_width) // 2, cy2 + 78),
        axis_title,
        fill=(40, 40, 40),
        font=FONT_AXIS,
    )

    legend_x = width - 218
    legend_y = 132
    draw.rounded_rectangle((legend_x - 12, legend_y - 12, width - 30, legend_y + 156), radius=8, fill=(255, 255, 255), outline=(216, 216, 216))
    for i, plant_id in enumerate(sorted(PLANT_COLORS)):
        y = legend_y + i * 24
        color = PLANT_COLORS[plant_id]
        draw.line((legend_x, y + 8, legend_x + 28, y + 8), fill=color, width=3)
        draw.ellipse((legend_x + 10, y + 4, legend_x + 18, y + 12), fill=color)
        draw.text((legend_x + 38, y), f"Plant {int(plant_id[-2:])}", fill=(36, 36, 36), font=FONT_SMALL)
    y = legend_y + 5 * 24
    _dashed_line(draw, (legend_x, y + 8, legend_x + 28, y + 8), fill=selected_color, width=3, dash=7)
    draw.text((legend_x + 38, y), selected_label, fill=(36, 36, 36), font=FONT_SMALL)

    current_row = plate_df.iloc[current_idx]
    current_frame_rows = plate_master[
        (plate_master["TimestampParsed"] == current_row["TimestampParsed"])
        & (plate_master["FrameIndex"] == int(current_row["FrameIndex"]))
    ]
    qa = _ownership_frame_qa(current_frame_rows)
    shoot_source, green_only = _ownership_rows_shoot_source(
        current_frame_rows,
        config,
    )
    qa["shoot_source"] = shoot_source
    qa["green_only"] = green_only
    first_total = next((float(value) for value in selected_values if np.isfinite(value)), float("nan"))
    current_total = float(current_row[selected_root_column])
    if selected_delta_column in current_row.index:
        growth_delta = float(current_row[selected_delta_column])
    else:
        growth_delta = current_total - first_total
    selected_mean = float(pd.to_numeric(pd.Series([current_row.get(selected_mean_column, np.nan)]), errors="coerce").fillna(np.nan).iloc[0])
    if not np.isfinite(selected_mean):
        plants = max(1.0, float(current_row.get("plants_detected", 1) or 1))
        selected_mean = current_total / plants

    def _metric_text(value: float, *, signed: bool = False) -> str:
        if not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value):+.1f} mm" if signed else f"{float(value):.1f} mm"

    stats = [
        ("Current", _metric_text(current_total)),
        ("Growth delta", _metric_text(growth_delta, signed=True)),
        ("Mean plant", _metric_text(selected_mean)),
        ("Plants detected", str(int(current_row["plants_detected"]))),
        ("Frame", f"{current_idx + 1}/{len(plate_df)}"),
    ]
    stats_y = height - 150
    qa_status = str(qa.get("status", "ok"))
    qa_color = {"ok": (30, 122, 70), "review": (172, 111, 18), "fail": (181, 54, 54)}.get(qa_status, (28, 80, 132))
    qa_text = (
        f"Frame QA: {qa_status.upper()}  |  fail {int(qa.get('fail', 0))}, "
        f"review {int(qa.get('review', 0))}  |  shoots {qa.get('shoot_source', 'not recorded')}"
    )
    draw.text((left_w + 90, stats_y - 32), qa_text, fill=qa_color, font=FONT_SMALL_BOLD)
    draw.rounded_rectangle((left_w + 80, stats_y, width - 80, height - 36), radius=12, fill=(246, 248, 250), outline=(218, 224, 230))
    col_w = (width - left_w - 180) // len(stats)
    for i, (name, value) in enumerate(stats):
        sx = left_w + 100 + i * col_w
        draw.text((sx, stats_y + 18), name, fill=(92, 98, 106), font=FONT_SMALL)
        draw.text((sx, stats_y + 48), value, fill=(20, 25, 31), font=FONT_VALUE)


def _prepare_timeline_df(total_df: pd.DataFrame) -> pd.DataFrame:
    df = total_df.copy()
    if "PetriDish" not in df.columns:
        raise ValueError("Total-root timelapse dataframe must contain a PetriDish column.")
    if "Timestamp" not in df.columns:
        raise ValueError("Total-root timelapse dataframe must contain a Timestamp column.")
    for required in ("FrameIndex", "total_root_length_mm", "mean_root_length_mm", "plants_detected"):
        if required not in df.columns:
            raise ValueError(f"Total-root timelapse dataframe is missing {required}.")
    df["TimestampParsed"] = df["Timestamp"].map(_parse_timestamp)
    df["FrameIndex"] = pd.to_numeric(df["FrameIndex"], errors="coerce").fillna(-1).astype(int)
    for column in ("total_root_length_mm", "mean_root_length_mm"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["plants_detected"] = pd.to_numeric(df["plants_detected"], errors="coerce").fillna(0.0)
    if "total_root_length_weighted_mm" not in df.columns:
        df["total_root_length_weighted_mm"] = df["total_root_length_mm"]
    if "mean_root_length_weighted_mm" not in df.columns:
        df["mean_root_length_weighted_mm"] = df["mean_root_length_mm"]
    for column in (
        "total_root_length_weighted_mm",
        "mean_root_length_weighted_mm",
        "selected_root_length_mm",
        "delta_total_root_length_weighted_mm",
        "delta_selected_root_length_mm",
        "total_primary_root_length_mm",
        "mean_primary_root_length_mm",
        "total_lateral_root_length_mm",
        "mean_lateral_root_length_mm",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["_sort_series"] = df.get("Series", df["PetriDish"]).fillna("Unknown").astype(str).map(_natural_sort_key)
    df["_sort_petri"] = df["PetriDish"].fillna("Unknown").astype(str).map(_natural_sort_key)
    df.sort_values(["_sort_series", "_sort_petri", "TimestampParsed", "FrameIndex"], inplace=True, kind="mergesort")
    return df.drop(columns=["_sort_series", "_sort_petri"], errors="ignore").reset_index(drop=True)


def _prepare_mask_total_timeline_df(total_df: pd.DataFrame) -> pd.DataFrame:
    df = total_df.copy()
    for required in ("PetriDish", "Timestamp", "FrameIndex", "total_root_length_mm", "SourceFile", "OutputMaskPath"):
        if required not in df.columns:
            raise ValueError(f"Mask-total timelapse dataframe is missing {required}.")
    df["TimestampParsed"] = df["Timestamp"].map(_parse_timestamp)
    df["FrameIndex"] = pd.to_numeric(df["FrameIndex"], errors="coerce").fillna(-1).astype(int)
    if "shoot_area_px" not in df.columns and "shoot_seed_pixels_class_2" in df.columns:
        df["shoot_area_px"] = df["shoot_seed_pixels_class_2"]
    if "shoot_pixels_class_2" not in df.columns and "shoot_area_px" in df.columns:
        df["shoot_pixels_class_2"] = df["shoot_area_px"]
    if "shoot_area_mm2" not in df.columns:
        if "pixel_size_mm" in df.columns and "shoot_area_px" in df.columns:
            px_size = pd.to_numeric(df["pixel_size_mm"], errors="coerce").fillna(0.0)
            shoot_px = pd.to_numeric(df["shoot_area_px"], errors="coerce").fillna(0.0)
            df["shoot_area_mm2"] = shoot_px * (px_size ** 2)
        else:
            df["shoot_area_mm2"] = 0.0
    if "total_root_pixels_selected_classes" not in df.columns:
        if "measured_mask_pixels" in df.columns:
            df["total_root_pixels_selected_classes"] = df["measured_mask_pixels"]
        elif "total_root_pixels_1_plus_3" in df.columns:
            df["total_root_pixels_selected_classes"] = df["total_root_pixels_1_plus_3"]
        else:
            df["total_root_pixels_selected_classes"] = 0.0
    if "primary_root_pixels_selected_class" not in df.columns:
        df["primary_root_pixels_selected_class"] = df["root_pixels_class_1"] if "root_pixels_class_1" in df.columns else 0.0
    if "lateral_root_pixels_selected_class" not in df.columns:
        df["lateral_root_pixels_selected_class"] = df["lateral_pixels_class_3"] if "lateral_pixels_class_3" in df.columns else 0.0
    numeric_columns = (
        "total_root_length_mm",
        "total_root_length_weighted_mm",
        "primary_root_length_mm",
        "primary_root_length_weighted_mm",
        "lateral_root_length_mm",
        "lateral_root_length_weighted_mm",
        "delta_total_root_length_mm",
        "delta_total_root_length_weighted_mm",
        "delta_primary_root_length_mm",
        "delta_primary_root_length_weighted_mm",
        "delta_lateral_root_length_mm",
        "delta_lateral_root_length_weighted_mm",
        "root_pixels_class_1",
        "lateral_pixels_class_3",
        "total_root_pixels_selected_classes",
        "primary_root_pixels_selected_class",
        "lateral_root_pixels_selected_class",
        "primary_root_class_id",
        "lateral_root_class_id",
        "shoot_seed_pixels_class_2",
        "shoot_pixels_class_2",
        "shoot_area_px",
        "shoot_area_mm2",
        "delta_shoot_area_mm2",
        "root_to_shoot_area_ratio_px",
        "root_length_mm_per_shoot_area_mm2",
        "measured_mask_pixels",
        "component_count",
    )
    for column in numeric_columns:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    if "mean_root_length_mm" not in df.columns:
        df["mean_root_length_mm"] = df["total_root_length_mm"]
    if "plants_detected" not in df.columns:
        df["plants_detected"] = 1
    df["_sort_series"] = df.get("Series", df["PetriDish"]).fillna("Unknown").astype(str).map(_natural_sort_key)
    df["_sort_petri"] = df["PetriDish"].fillna("Unknown").astype(str).map(_natural_sort_key)
    df.sort_values(["_sort_series", "_sort_petri", "TimestampParsed", "FrameIndex"], inplace=True, kind="mergesort")
    group_columns = ["Series", "PetriDish"] if "Series" in df.columns else ["PetriDish"]
    for source_column, delta_column in (
        ("total_root_length_mm", "delta_total_root_length_mm"),
        ("total_root_length_weighted_mm", "delta_total_root_length_weighted_mm"),
        ("primary_root_length_mm", "delta_primary_root_length_mm"),
        ("primary_root_length_weighted_mm", "delta_primary_root_length_weighted_mm"),
        ("lateral_root_length_mm", "delta_lateral_root_length_mm"),
        ("lateral_root_length_weighted_mm", "delta_lateral_root_length_weighted_mm"),
    ):
        if delta_column not in total_df.columns:
            df[delta_column] = df.groupby(group_columns, dropna=False)[source_column].diff().fillna(0.0)
    if "delta_shoot_area_mm2" not in total_df.columns:
        df["delta_shoot_area_mm2"] = df.groupby(group_columns, dropna=False)["shoot_area_mm2"].diff().fillna(0.0)
    return df.drop(columns=["_sort_series", "_sort_petri"], errors="ignore").reset_index(drop=True)


def _draw_mask_total_plate_panel(
    frame: Image.Image,
    timeline_row: pd.Series,
    current_idx: int,
    frame_count: int,
    left_w: int,
    height: int,
    config: LazyRootGrowthVideoConfig,
) -> None:
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, left_w, height), fill=(20, 21, 22))
    image_path = Path(str(timeline_row.get("SourceFile", "")))
    mask_path = _resolve_mask_path_from_row(timeline_row)
    plate = str(timeline_row["PetriDish"])
    series = str(timeline_row.get("Series", "") or "").strip()
    timestamp = _display_timeline_position(
        timeline_row["Timestamp"],
        current_idx,
        frame_count,
    )
    shoot_source, green_only = _mask_total_row_shoot_source(timeline_row, config)
    if mask_path is None or (not bool(config.mask_only) and not image_path.exists()):
        draw.text((48, height // 2), f"Missing source image or mask for {plate}", fill=(240, 240, 240), font=FONT_TITLE)
        return

    image: Image.Image | None = None
    mask: np.ndarray | None = None
    if bool(config.mask_only):
        mask = _read_index_mask(mask_path)
        mask_shape = mask.shape
        src_h, src_w = int(mask_shape[0]), int(mask_shape[1])
    elif image_path.exists():
        image = Image.open(image_path).convert("RGB")
        src_w, src_h = image.size
    else:
        mask = _read_index_mask(mask_path)
        mask_shape = mask.shape
        src_h, src_w = int(mask_shape[0]), int(mask_shape[1])
    box_x, box_y, box_w, box_h = 16, 18, left_w - 32, height - 36
    dst_w, dst_h, _scale = _fit_dims(src_w, src_h, box_w, box_h)
    x0 = box_x + (box_w - dst_w) // 2
    y0 = box_y + (box_h - dst_h) // 2
    if bool(config.mask_only):
        display_mask = np.asarray(mask if mask is not None else _read_index_mask(mask_path), dtype=np.uint8)
        if green_only:
            display_mask = _display_mask_with_rgb_shoot_rescue(
                display_mask,
                image_path,
                root_class_ids=config.root_class_ids,
                shoot_class_ids=config.shoot_class_ids,
            )
        display_dilation_px = 0 if green_only else max(0, int(config.mask_display_dilation_px))
        panel = _mask_array_resized(
            display_mask,
            (dst_w, dst_h),
            dilation_px=display_dilation_px,
            root_class_ids=config.root_class_ids,
            shoot_class_ids=config.shoot_class_ids,
        )
    else:
        if image is not None:
            display_mask = _read_index_mask(mask_path)
            if green_only:
                display_mask = _display_mask_with_rgb_shoot_rescue(
                    display_mask,
                    image_path,
                    root_class_ids=config.root_class_ids,
                    shoot_class_ids=config.shoot_class_ids,
                )
            panel = _overlay_mask_array_resized(
                image,
                display_mask,
                (dst_w, dst_h),
                root_class_ids=config.root_class_ids,
                shoot_class_ids=config.shoot_class_ids,
            )
        else:
            panel = _mask_only_resized(
                mask_path,
                (dst_w, dst_h),
                root_class_ids=config.root_class_ids,
                shoot_class_ids=config.shoot_class_ids,
            )
    frame.paste(panel, (x0, y0))

    label = f"{series} / {plate}  |  {timestamp}" if series and series != plate else f"{plate}  |  {timestamp}"
    text_box = draw.textbbox((0, 0), label, font=FONT_SMALL_BOLD)
    draw.rounded_rectangle((24, 26, 34 + text_box[2], 56), radius=8, fill=(0, 0, 0))
    draw.text((30, 32), label, fill=(250, 250, 250), font=FONT_SMALL_BOLD)
    _draw_badge(
        draw,
        (24, 62),
        f"Shoots: {shoot_source}",
        status="info" if green_only else "review",
        font=FONT_SMALL_BOLD,
    )

    legend_items = _mask_total_legend_items(
        config.root_class_ids,
        config.shoot_class_ids,
        show_rgb_shoot_rescue=green_only,
    )
    legend_height = max(108, 52 + 22 * len(legend_items))
    legend_y = height - legend_height - 26
    draw.rounded_rectangle((24, legend_y, 386, height - 26), radius=8, fill=(0, 0, 0))
    draw.text((38, legend_y + 12), "Mask-only view" if bool(config.mask_only) else "Mask overlay", fill=(245, 245, 245), font=FONT_SMALL_BOLD)
    for i, (color, name) in enumerate(legend_items):
        y = legend_y + 42 + i * 22
        draw.rectangle((40, y + 2, 56, y + 14), fill=color)
        draw.text((66, y - 1), name, fill=(236, 236, 236), font=FONT_SMALL)


def _draw_mask_total_chart_panel(
    frame: Image.Image,
    plate_df: pd.DataFrame,
    current_idx: int,
    left_w: int,
    width: int,
    height: int,
    config: LazyRootGrowthVideoConfig,
) -> None:
    draw = ImageDraw.Draw(frame)
    draw.rectangle((left_w, 0, width, height), fill=(255, 255, 255))
    draw.line((left_w, 0, left_w, height), fill=(18, 18, 18), width=4)
    plate = str(plate_df["PetriDish"].iloc[0])
    series = str(plate_df["Series"].iloc[0]) if "Series" in plate_df.columns else ""
    plate_label = f"{series} / {plate}" if series and series != plate else plate
    chart_box = (left_w + 118, 135, width - 92, height - 310)
    cx1, cy1, cx2, cy2 = chart_box

    metric_mode, metric_spec = _mask_total_root_metric_spec(config)
    root_column = str(metric_spec["column"])
    root_label = str(metric_spec["label"])
    root_axis_label = str(metric_spec["axis_label"])
    root_color = tuple(int(v) for v in metric_spec["color"])  # type: ignore[arg-type]
    total_values = [float(v) for v in pd.to_numeric(plate_df[root_column], errors="coerce").fillna(0.0)]
    shoot_values = [float(v) for v in pd.to_numeric(plate_df["shoot_area_mm2"], errors="coerce").fillna(0.0)]
    max_val = max([0.0, *total_values])
    max_shoot = max([0.0, *shoot_values])
    y_min = 0.0
    y_max = max(10.0, max_val * 1.12)
    shoot_y_max = max(1.0, max_shoot * 1.12)

    draw.text((left_w + 145, 38), f"Mask {root_label} and Shoot Area Over Time", fill=(18, 22, 28), font=FONT_TITLE)
    draw.text((left_w + 360, 72), f"Petridish: {plate_label}", fill=(48, 54, 62), font=FONT_SUBTITLE)
    draw.rectangle(chart_box, outline=(95, 95, 95), width=2)
    for i in range(6):
        y = cy2 - int(round((cy2 - cy1) * i / 5))
        value = y_min + (y_max - y_min) * i / 5
        draw.line((cx1, y, cx2, y), fill=(225, 229, 232), width=1)
        label = f"{value:.0f}"
        text_w = draw.textbbox((0, 0), label, font=FONT_AXIS)[2]
        draw.text((cx1 - text_w - 12, y - 8), label, fill=(80, 80, 80), font=FONT_AXIS)
        shoot_label = f"{(shoot_y_max * i / 5):.1f}"
        draw.text((cx2 + 10, y - 8), shoot_label, fill=MASK_COLORS[2], font=FONT_AXIS)

    n = len(plate_df)
    x_positions = [(cx1 + cx2) // 2] if n <= 1 else [cx1 + int(round((cx2 - cx1) * i / (n - 1))) for i in range(n)]
    for x in x_positions:
        draw.line((x, cy1, x, cy2), fill=(237, 239, 241), width=1)

    total_xy = _plot_xy(total_values[: current_idx + 1], x_positions[: current_idx + 1], y_min, y_max, chart_box)
    if len(total_xy) >= 2:
        for a, b in zip(total_xy[:-1], total_xy[1:]):
            _dashed_line(draw, (a[0], a[1], b[0], b[1]), fill=root_color, width=3, dash=10)
    for point in total_xy:
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=root_color)

    shoot_xy = _plot_xy(shoot_values[: current_idx + 1], x_positions[: current_idx + 1], y_min, shoot_y_max, chart_box)
    if len(shoot_xy) >= 2:
        draw.line(shoot_xy, fill=MASK_COLORS[2], width=3)
    for point in shoot_xy:
        draw.rectangle((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=MASK_COLORS[2])

    timestamps = plate_df["Timestamp"].tolist()
    for i, value in enumerate(timestamps):
        label_img = Image.new("RGBA", (70, 24), (255, 255, 255, 0))
        ld = ImageDraw.Draw(label_img)
        ld.text(
            (0, 2),
            _short_timeline_position(value, i, len(plate_df)),
            fill=(75, 75, 75),
            font=FONT_TINY,
        )
        label_img = label_img.rotate(36, expand=True, resample=Image.Resampling.BICUBIC)
        frame.paste(label_img, (x_positions[i] - 22, cy2 + 14), label_img)

    y_label = Image.new("RGBA", (220, 28), (255, 255, 255, 0))
    yd = ImageDraw.Draw(y_label)
    yd.text((0, 2), root_axis_label, fill=root_color, font=FONT_AXIS)
    y_label = y_label.rotate(270, expand=True)
    frame.paste(y_label, (left_w + 28, (cy1 + cy2) // 2 - y_label.height // 2), y_label)
    y2_label = Image.new("RGBA", (190, 28), (255, 255, 255, 0))
    y2d = ImageDraw.Draw(y2_label)
    y2d.text((0, 2), "Shoot Area (mm2)", fill=MASK_COLORS[2], font=FONT_AXIS)
    y2_label = y2_label.rotate(90, expand=True)
    frame.paste(y2_label, (width - 48, (cy1 + cy2) // 2 - y2_label.height // 2), y2_label)
    axis_title = _timeline_axis_title(timestamps)
    axis_width = draw.textbbox((0, 0), axis_title, font=FONT_AXIS)[2]
    draw.text(
        ((cx1 + cx2 - axis_width) // 2, cy2 + 78),
        axis_title,
        fill=(40, 40, 40),
        font=FONT_AXIS,
    )

    legend_x = width - 270
    legend_y = 132
    draw.rounded_rectangle((legend_x - 12, legend_y - 12, width - 30, legend_y + 78), radius=8, fill=(255, 255, 255), outline=(216, 216, 216))
    _dashed_line(draw, (legend_x, legend_y + 18, legend_x + 36, legend_y + 18), fill=root_color, width=3, dash=7)
    draw.text((legend_x + 46, legend_y + 10), root_label, fill=(36, 36, 36), font=FONT_SMALL)
    draw.line((legend_x, legend_y + 48, legend_x + 36, legend_y + 48), fill=MASK_COLORS[2], width=3)
    draw.rectangle((legend_x + 14, legend_y + 44, legend_x + 22, legend_y + 52), fill=MASK_COLORS[2])
    draw.text((legend_x + 46, legend_y + 40), "Shoot area", fill=(36, 36, 36), font=FONT_SMALL)

    current_row = plate_df.iloc[current_idx]
    first_total = float(total_values[0]) if total_values else 0.0
    current_total = float(current_row[root_column])
    first_shoot = float(shoot_values[0]) if shoot_values else 0.0
    current_shoot = float(current_row.get("shoot_area_mm2", 0.0))
    shoot_source, green_only = _mask_total_row_shoot_source(current_row, config)
    root_ids = _normalize_class_id_tuple(config.root_class_ids, DEFAULT_MASK_TOTAL_ROOT_CLASS_IDS)
    lateral_class_label = f"Class {root_ids[1]} px" if len(root_ids) > 1 else "Second root px"
    lateral_px_value = current_row.get("lateral_root_pixels_selected_class", current_row.get("lateral_pixels_class_3", 0))
    stats = [
        (root_label, f"{current_total:.1f} mm"),
        ("Root delta", f"{current_total - first_total:+.1f} mm"),
        ("Shoot area", f"{current_shoot:.2f} mm2"),
        ("Shoot delta", f"{current_shoot - first_shoot:+.2f} mm2"),
        (lateral_class_label, f"{int(lateral_px_value):,}"),
        ("Shoot px", f"{int(current_row.get('shoot_area_px', current_row.get('shoot_seed_pixels_class_2', 0))):,}"),
        ("Frame", f"{current_idx + 1}/{len(plate_df)}"),
    ]
    stats_y = height - 180
    shoot_source_color = (28, 80, 132) if green_only else (172, 111, 18)
    draw.text((left_w + 70, stats_y - 32), f"Shoot measurement source: {shoot_source}", fill=shoot_source_color, font=FONT_SMALL_BOLD)
    draw.rounded_rectangle((left_w + 60, stats_y, width - 60, height - 24), radius=12, fill=(246, 248, 250), outline=(218, 224, 230))
    stat_columns = 4
    col_w = (width - left_w - 140) // stat_columns
    for i, (name, value) in enumerate(stats):
        stat_row = i // stat_columns
        stat_col = i % stat_columns
        sx = left_w + 80 + stat_col * col_w
        sy = stats_y + 12 + stat_row * 62
        draw.text((sx, sy), name, fill=(92, 98, 106), font=FONT_SMALL)
        draw.text((sx, sy + 24), value, fill=(20, 25, 31), font=FONT_VALUE_COMPACT)


def _make_mask_total_frame(
    plate_df: pd.DataFrame,
    current_idx: int,
    config: LazyRootGrowthVideoConfig,
) -> Image.Image:
    width = int(config.width)
    height = int(config.height)
    left_w = width // 2
    frame = Image.new("RGB", (width, height), (255, 255, 255))
    _draw_mask_total_plate_panel(
        frame,
        plate_df.iloc[current_idx],
        current_idx=current_idx,
        frame_count=len(plate_df),
        left_w=left_w,
        height=height,
        config=config,
    )
    _draw_mask_total_chart_panel(frame, plate_df, current_idx, left_w=left_w, width=width, height=height, config=config)
    return frame


def _prepare_master_df(master_df: pd.DataFrame) -> pd.DataFrame:
    df = master_df.copy()
    if "PetriDish" not in df.columns:
        raise ValueError("Master measurements dataframe must contain a PetriDish column.")
    if "Timestamp" not in df.columns:
        raise ValueError("Master measurements dataframe must contain a Timestamp column.")
    if "FrameIndex" not in df.columns:
        raise ValueError("Master measurements dataframe must contain a FrameIndex column.")
    if "root_length_mm_clean" not in df.columns:
        fallback = _first_existing_name(
            df,
            (
                "total_root_length_mm_clean",
                "total_root_length_mm_raw",
                "root_length_mm_raw",
            ),
        )
        if fallback is None:
            raise ValueError("Master measurements dataframe must contain root length measurements.")
        df["root_length_mm_clean"] = df[fallback]
    df["TimestampParsed"] = df["Timestamp"].map(_parse_timestamp)
    df["FrameIndex"] = pd.to_numeric(df["FrameIndex"], errors="coerce").fillna(-1).astype(int)
    for column in (
        "root_length_mm_clean",
        "total_root_length_mm_clean",
        "total_root_length_mm_raw",
        "primary_root_length_mm_clean",
        "primary_root_length_mm_raw",
        "primary_root_length_weighted_mm_clean",
        "primary_root_length_weighted_mm_raw",
        "total_root_length_weighted_mm_clean",
        "total_root_length_weighted_mm_raw",
        "total_root_length_weighted_mm",
        "lateral_total_length_mm",
        "lateral_total_length_weighted_mm",
        "lateral_root_length_weighted_mm",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _make_frame(
    current_rows: pd.DataFrame,
    plate_df: pd.DataFrame,
    plate_master: pd.DataFrame,
    current_idx: int,
    config: LazyRootGrowthVideoConfig,
) -> Image.Image:
    width = int(config.width)
    height = int(config.height)
    left_w = width // 2
    frame = Image.new("RGB", (width, height), (255, 255, 255))
    _draw_plate_panel(
        frame,
        current_rows,
        plate_df.iloc[current_idx],
        current_idx=current_idx,
        frame_count=len(plate_df),
        left_w=left_w,
        height=height,
        config=config,
    )
    _draw_chart_panel(frame, plate_df, plate_master, current_idx, left_w=left_w, width=width, height=height, config=config)
    return frame


def generate_lazy_root_growth_video(
    master_df: pd.DataFrame,
    total_df: pd.DataFrame,
    output_path: Path,
    *,
    summary_csv_path: Path | None = None,
    sample_frame_path: Path | None = None,
    config: LazyRootGrowthVideoConfig | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> LazyRootGrowthVideoResult:
    cfg = config or LazyRootGrowthVideoConfig()
    timeline = _prepare_timeline_df(total_df)
    master = _prepare_master_df(master_df)
    if not bool(cfg.show_rgb_shoot_rescue) and (
        _mask_total_dataframe_requests_rgb_shoot_rescue(timeline)
        or _mask_total_dataframe_requests_rgb_shoot_rescue(master)
    ):
        cfg.show_rgb_shoot_rescue = True
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path = summary_csv_path or output_path.with_name("npec_root_growth_video_summary.csv")
    sample_frame_path = sample_frame_path or output_path.with_name("npec_root_growth_video_sample_frame.png")
    try:
        sample_frame_path.unlink(missing_ok=True)
    except Exception:
        pass

    if "Series" not in timeline.columns:
        timeline["Series"] = "Unknown"
    if "Series" not in master.columns:
        master["Series"] = "Unknown"
    plate_groups = [
        (str(series), str(plate))
        for series, plate in timeline[["Series", "PetriDish"]].dropna().drop_duplicates().itertuples(index=False, name=None)
        if str(plate) != "_stabilized_input"
    ]
    root_metric_mode, root_metric_spec = _ownership_root_metric_spec(cfg)
    root_metric_column = str(root_metric_spec["column"])
    if root_metric_column not in timeline.columns:
        root_metric_column = "selected_root_length_mm" if "selected_root_length_mm" in timeline.columns else "total_root_length_mm"
    writer = open_mp4_video_writer(output_path=output_path, width=int(cfg.width), height=int(cfg.height), fps=max(1, int(cfg.fps)))
    summary_rows: list[dict[str, object]] = []
    frames_rendered = 0
    plates_rendered = 0
    total_frames = int(len(timeline))
    try:
        for series, plate in plate_groups:
            if cancel_callback is not None and bool(cancel_callback()):
                raise RuntimeError("Root-growth video rendering canceled.")
            plate_df = timeline[
                (timeline["Series"].astype(str) == series) & (timeline["PetriDish"].astype(str) == plate)
            ].copy().reset_index(drop=True)
            plate_master = master[
                (master["Series"].astype(str) == series) & (master["PetriDish"].astype(str) == plate)
            ].copy().reset_index(drop=True)
            if plate_df.empty or plate_master.empty:
                continue
            plates_rendered += 1
            first = plate_df.iloc[0]
            final = plate_df.iloc[-1]
            first_selected_root = float(first.get(root_metric_column, 0.0))
            final_selected_root = float(final.get(root_metric_column, 0.0))
            plate_qa = _ownership_plate_qa(plate_master)
            summary_rows.append(
                {
                    "Series": series,
                    "PetriDish": plate,
                    "frames": len(plate_df),
                    "first_timestamp": first["Timestamp"],
                    "last_timestamp": final["Timestamp"],
                    "root_metric_mode": root_metric_mode,
                    "selected_root_metric_column": root_metric_column,
                    "final_selected_root_length_mm": final_selected_root,
                    "delta_selected_root_length_mm": float(final_selected_root - first_selected_root),
                    "final_total_root_length_mm": float(final["total_root_length_mm"]),
                    "final_total_root_length_weighted_mm": float(
                        final.get("total_root_length_weighted_mm", final["total_root_length_mm"])
                    ),
                    "final_mean_root_length_mm": float(final["mean_root_length_mm"]),
                    "final_mean_root_length_weighted_mm": float(
                        final.get("mean_root_length_weighted_mm", final["mean_root_length_mm"])
                    ),
                    "plants_detected_final": int(final["plants_detected"]),
                    "frames_needing_review": int(plate_qa.get("frames_with_review", 0)),
                    "frames_failed_review": int(plate_qa.get("frames_with_fail", 0)),
                    "tracks_needing_review": int(plate_qa.get("tracks_with_review", 0)),
                    "shoot_measurement_source": str(plate_qa.get("shoot_source", "not recorded")),
                    "shoot_rgb_green_only_enabled": bool(plate_qa.get("green_only", False)),
                    "delta_total_root_length_mm": float(final["total_root_length_mm"] - first["total_root_length_mm"]),
                    "delta_total_root_length_weighted_mm": float(
                        final.get("total_root_length_weighted_mm", final["total_root_length_mm"])
                        - first.get("total_root_length_weighted_mm", first["total_root_length_mm"])
                    ),
                }
            )
            for idx, row in plate_df.iterrows():
                if cancel_callback is not None and bool(cancel_callback()):
                    raise RuntimeError("Root-growth video rendering canceled.")
                current_rows = plate_master[
                    (plate_master["TimestampParsed"] == row["TimestampParsed"])
                    & (plate_master["FrameIndex"] == int(row["FrameIndex"]))
                ].copy()
                if current_rows.empty:
                    continue
                frame = _make_frame(current_rows, plate_df, plate_master, idx, cfg)
                if not sample_frame_path.exists():
                    frame.save(sample_frame_path)
                writer.append(np.asarray(frame, dtype=np.uint8))
                frames_rendered += 1
                if progress_callback is not None:
                    progress_callback(frames_rendered, total_frames, f"{plate} {idx + 1}/{len(plate_df)}")
    finally:
        writer.close()

    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    return LazyRootGrowthVideoResult(
        video_path=output_path,
        sample_frame_path=sample_frame_path,
        summary_csv_path=summary_csv_path,
        frames_rendered=int(frames_rendered),
        plates_rendered=int(plates_rendered),
    )


def generate_lazy_mask_total_root_growth_video(
    total_df: pd.DataFrame,
    output_path: Path,
    *,
    summary_csv_path: Path | None = None,
    sample_frame_path: Path | None = None,
    config: LazyRootGrowthVideoConfig | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> LazyRootGrowthVideoResult:
    cfg = config or LazyRootGrowthVideoConfig()
    timeline = _prepare_mask_total_timeline_df(total_df)
    if not bool(cfg.show_rgb_shoot_rescue) and _mask_total_dataframe_requests_rgb_shoot_rescue(timeline):
        cfg.show_rgb_shoot_rescue = True
    root_metric_mode, root_metric_spec = _mask_total_root_metric_spec(cfg)
    root_metric_column = str(root_metric_spec["column"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path = summary_csv_path or output_path.with_name("npec_root_growth_video_summary.csv")
    sample_frame_path = sample_frame_path or output_path.with_name("npec_root_growth_video_sample_frame.png")
    try:
        sample_frame_path.unlink(missing_ok=True)
    except Exception:
        pass

    if "Series" in timeline.columns:
        group_iter = timeline.groupby(["Series", "PetriDish"], sort=False, dropna=False)
    else:
        group_iter = timeline.groupby(["PetriDish"], sort=False, dropna=False)
    writer = open_mp4_video_writer(output_path=output_path, width=int(cfg.width), height=int(cfg.height), fps=max(1, int(cfg.fps)))
    summary_rows: list[dict[str, object]] = []
    frames_rendered = 0
    plates_rendered = 0
    total_frames = int(len(timeline))
    try:
        for _group_key, group_df in group_iter:
            if cancel_callback is not None and bool(cancel_callback()):
                raise RuntimeError("Root-growth video rendering canceled.")
            plate_df = group_df.copy().reset_index(drop=True)
            if plate_df.empty:
                continue
            plate = str(plate_df["PetriDish"].iloc[0])
            if plate == "_stabilized_input":
                continue
            series = str(plate_df["Series"].iloc[0]) if "Series" in plate_df.columns else ""
            plates_rendered += 1
            first = plate_df.iloc[0]
            final = plate_df.iloc[-1]
            first_selected_root = float(first.get(root_metric_column, 0.0))
            final_selected_root = float(final.get(root_metric_column, 0.0))
            shoot_source, green_only = _mask_total_plate_shoot_source(plate_df, cfg)
            summary_row = {
                "PetriDish": plate,
                "frames": len(plate_df),
                "first_timestamp": first["Timestamp"],
                "last_timestamp": final["Timestamp"],
                "root_metric_mode": root_metric_mode,
                "root_metric_column": root_metric_column,
                "final_selected_root_length_mm": final_selected_root,
                "delta_selected_root_length_mm": float(final_selected_root - first_selected_root),
                "final_total_root_length_mm": float(final["total_root_length_mm"]),
                "final_total_root_length_weighted_mm": float(final.get("total_root_length_weighted_mm", 0.0)),
                "final_primary_root_length_mm": float(final.get("primary_root_length_mm", 0.0)),
                "final_primary_root_length_weighted_mm": float(final.get("primary_root_length_weighted_mm", 0.0)),
                "final_lateral_root_length_mm": float(final.get("lateral_root_length_mm", 0.0)),
                "final_lateral_root_length_weighted_mm": float(final.get("lateral_root_length_weighted_mm", 0.0)),
                "final_shoot_area_mm2": float(final.get("shoot_area_mm2", 0.0)),
                "max_shoot_area_mm2": float(pd.to_numeric(plate_df["shoot_area_mm2"], errors="coerce").fillna(0.0).max()),
                "final_shoot_area_px": int(final.get("shoot_area_px", final.get("shoot_seed_pixels_class_2", 0))),
                "final_measured_mask_pixels": int(final.get("measured_mask_pixels", 0)),
                "final_total_root_pixels_selected_classes": int(final.get("total_root_pixels_selected_classes", 0)),
                "final_primary_root_pixels_selected_class": int(final.get("primary_root_pixels_selected_class", 0)),
                "final_lateral_root_pixels_selected_class": int(final.get("lateral_root_pixels_selected_class", 0)),
                "plants_detected_final": int(final.get("plants_detected", 1)),
                "shoot_measurement_source": str(shoot_source),
                "shoot_rgb_green_only_enabled": bool(green_only),
                "delta_total_root_length_mm": float(final["total_root_length_mm"] - first["total_root_length_mm"]),
                "delta_primary_root_length_mm": float(final.get("primary_root_length_mm", 0.0) - first.get("primary_root_length_mm", 0.0)),
                "delta_lateral_root_length_mm": float(final.get("lateral_root_length_mm", 0.0) - first.get("lateral_root_length_mm", 0.0)),
                "delta_shoot_area_mm2": float(final.get("shoot_area_mm2", 0.0) - first.get("shoot_area_mm2", 0.0)),
            }
            if series:
                summary_row = {"Series": series, **summary_row}
            summary_rows.append(summary_row)
            for idx, _row in plate_df.iterrows():
                if cancel_callback is not None and bool(cancel_callback()):
                    raise RuntimeError("Root-growth video rendering canceled.")
                frame = _make_mask_total_frame(plate_df, idx, cfg)
                if not sample_frame_path.exists():
                    frame.save(sample_frame_path)
                writer.append(np.asarray(frame, dtype=np.uint8))
                frames_rendered += 1
                if progress_callback is not None:
                    progress_callback(frames_rendered, total_frames, f"{plate} {idx + 1}/{len(plate_df)}")
    finally:
        writer.close()

    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    return LazyRootGrowthVideoResult(
        video_path=output_path,
        sample_frame_path=sample_frame_path,
        summary_csv_path=summary_csv_path,
        frames_rendered=int(frames_rendered),
        plates_rendered=int(plates_rendered),
    )
