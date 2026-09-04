from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OWNERSHIP_TOTAL_METRIC_CANDIDATES = (
    "total_root_length_mm_clean",
    "total_root_length_mm_raw",
    "root_length_mm_clean",
    "root_length_mm_raw",
)
OWNERSHIP_TOTAL_WEIGHTED_METRIC_CANDIDATES = (
    "total_root_length_weighted_mm_clean",
    "total_root_length_weighted_mm_raw",
    "total_root_length_weighted_mm",
    "root_length_weighted_mm_clean",
    "root_length_weighted_mm_raw",
    "root_length_weighted_mm",
)
OWNERSHIP_PRIMARY_METRIC_CANDIDATES = (
    "primary_root_length_mm_clean",
    "primary_root_length_mm_raw",
    "root_length_mm_clean",
    "root_length_mm_raw",
)
OWNERSHIP_PRIMARY_WEIGHTED_METRIC_CANDIDATES = (
    "primary_root_length_weighted_mm_clean",
    "primary_root_length_weighted_mm_raw",
    "primary_root_length_weighted_mm",
    "primary_root_weighted_length_mm_clean",
    "primary_root_weighted_length_mm_raw",
    "primary_root_weighted_length_mm",
    "root_length_weighted_mm_clean",
    "root_length_weighted_mm_raw",
    "root_length_weighted_mm",
)
OWNERSHIP_LATERAL_METRIC_CANDIDATES = ("lateral_total_length_mm",)
OWNERSHIP_LATERAL_WEIGHTED_METRIC_CANDIDATES = (
    "lateral_total_length_weighted_mm",
    "lateral_root_length_weighted_mm",
    "lateral_total_weighted_length_mm",
    "lateral_root_weighted_length_mm",
)
OWNERSHIP_SHOOT_AREA_CANDIDATES = ("shoot_area_mm2",)
OWNERSHIP_SHOOT_PIXELS_CANDIDATES = ("shoot_area_px",)
OWNERSHIP_TIMESTEP_HOURS_CANDIDATES = (
    "timestep_hours",
    "time_step_hours",
    "frame_interval_hours",
    "interval_hours",
)
DERIVED_TOTAL_ROOT_COLUMN = "ownership_total_root_length_mm"
DERIVED_TOTAL_ROOT_WEIGHTED_COLUMN = "ownership_total_root_length_weighted_mm"
DERIVED_PRIMARY_ROOT_COLUMN = "ownership_primary_root_length_mm"
DERIVED_PRIMARY_ROOT_WEIGHTED_COLUMN = "ownership_primary_root_length_weighted_mm"
DERIVED_LATERAL_ROOT_COLUMN = "ownership_lateral_root_length_mm"
DERIVED_LATERAL_ROOT_WEIGHTED_COLUMN = "ownership_lateral_root_length_weighted_mm"
DERIVED_SHOOT_AREA_COLUMN = "ownership_shoot_area_mm2"
DERIVED_SHOOT_PIXELS_COLUMN = "ownership_shoot_area_px"
DERIVED_PRIMARY_PLUS_LATERAL_ROOT_COLUMN = "ownership_primary_plus_lateral_root_length_mm"
DERIVED_SELECTED_ROOT_MODE_COLUMN = "ownership_selected_root_metric_mode"
DERIVED_SELECTED_ROOT_COLUMN_COLUMN = "ownership_selected_root_metric_column"
DERIVED_SELECTED_ROOT_LENGTH_COLUMN = "ownership_selected_root_length_mm"
DERIVED_ELAPSED_HOURS_COLUMN = "ownership_elapsed_hours"
DERIVED_ELAPSED_DAYS_COLUMN = "ownership_elapsed_days"
DERIVED_TIME_DELTA_SOURCE_COLUMN = "ownership_time_delta_source"
DERIVED_TOTAL_ROOT_RATE_COLUMN = "ownership_total_root_growth_rate_mm_per_day"
DERIVED_TOTAL_ROOT_WEIGHTED_RATE_COLUMN = "ownership_total_root_length_weighted_growth_rate_mm_per_day"
DERIVED_PRIMARY_ROOT_RATE_COLUMN = "ownership_primary_root_growth_rate_mm_per_day"
DERIVED_PRIMARY_ROOT_WEIGHTED_RATE_COLUMN = "ownership_primary_root_length_weighted_growth_rate_mm_per_day"
DERIVED_LATERAL_ROOT_RATE_COLUMN = "ownership_lateral_root_growth_rate_mm_per_day"
DERIVED_LATERAL_ROOT_WEIGHTED_RATE_COLUMN = "ownership_lateral_root_length_weighted_growth_rate_mm_per_day"
DERIVED_PRIMARY_PLUS_LATERAL_ROOT_RATE_COLUMN = "ownership_primary_plus_lateral_root_growth_rate_mm_per_day"
DERIVED_SELECTED_ROOT_RATE_COLUMN = "ownership_selected_root_growth_rate_mm_per_day"
DERIVED_SHOOT_AREA_RATE_COLUMN = "ownership_shoot_area_growth_rate_mm2_per_day"
DERIVED_SHOOT_PIXELS_RATE_COLUMN = "ownership_shoot_pixel_growth_rate_px_per_day"
DERIVED_FRAME_REVIEW_STATUS_COLUMN = "ownership_frame_review_status"
DERIVED_FRAME_REVIEW_REASONS_COLUMN = "ownership_frame_review_reasons"
DERIVED_FRAME_REVIEW_REASON_COUNT_COLUMN = "ownership_frame_review_reason_count"
DERIVED_GROWTH_RATE_COLUMNS = (
    DERIVED_TOTAL_ROOT_RATE_COLUMN,
    DERIVED_TOTAL_ROOT_WEIGHTED_RATE_COLUMN,
    DERIVED_PRIMARY_ROOT_RATE_COLUMN,
    DERIVED_PRIMARY_ROOT_WEIGHTED_RATE_COLUMN,
    DERIVED_LATERAL_ROOT_RATE_COLUMN,
    DERIVED_LATERAL_ROOT_WEIGHTED_RATE_COLUMN,
    DERIVED_PRIMARY_PLUS_LATERAL_ROOT_RATE_COLUMN,
    DERIVED_SELECTED_ROOT_RATE_COLUMN,
    DERIVED_SHOOT_AREA_RATE_COLUMN,
    DERIVED_SHOOT_PIXELS_RATE_COLUMN,
)
DERIVED_FRAME_REVIEW_COLUMNS = (
    DERIVED_FRAME_REVIEW_STATUS_COLUMN,
    DERIVED_FRAME_REVIEW_REASON_COUNT_COLUMN,
    DERIVED_FRAME_REVIEW_REASONS_COLUMN,
)
OWNERSHIP_ROOT_METRIC_COLUMNS = {
    "total": DERIVED_TOTAL_ROOT_COLUMN,
    "weighted_total": DERIVED_TOTAL_ROOT_WEIGHTED_COLUMN,
    "primary": DERIVED_PRIMARY_ROOT_COLUMN,
    "lateral": DERIVED_LATERAL_ROOT_COLUMN,
}
EXCEL_CELL_CHAR_LIMIT = 32767
EXCEL_SAFE_TEXT_LIMIT = 32000
OWNERSHIP_REVIEW_THRESHOLDS = {
    "warn_stability_score_below": 70.0,
    "fail_stability_score_below": 45.0,
    "warn_observed_frame_fraction_below": 0.70,
    "fail_observed_frame_fraction_below": 0.40,
    "warn_valid_ownership_frame_fraction_below": 0.75,
    "fail_valid_ownership_frame_fraction_below": 0.50,
    "warn_root_present_frame_fraction_below": 0.75,
    "fail_root_present_frame_fraction_below": 0.40,
    "warn_shoot_present_frame_fraction_below": 0.50,
    "fail_shoot_present_frame_fraction_below": 0.20,
    "warn_conflict_frame_fraction_above": 0.20,
    "fail_conflict_frame_fraction_above": 0.50,
    "warn_max_root_length_drop_mm_above": 5.0,
    "fail_max_root_length_drop_mm_above": 15.0,
    "warn_max_bbox_center_jump_px_above": 350.0,
    "fail_max_bbox_center_jump_px_above": 750.0,
    "warn_max_shoot_center_jump_px_above": 250.0,
    "fail_max_shoot_center_jump_px_above": 600.0,
}
OWNERSHIP_REVIEW_COLUMNS = (
    "Series",
    "PetriDish",
    "plant_id",
    "ownership_review_status",
    "ownership_review_reason_count",
    "ownership_review_reasons",
    "ownership_stability_score_0_100",
    "frames_observed",
    "expected_frames",
    "observed_frame_fraction",
    "valid_ownership_frame_fraction",
    "root_present_frame_fraction",
    "shoot_present_frame_fraction",
    "conflict_frame_fraction",
    "invalid_ownership_frames",
    "root_missing_frames",
    "shoot_missing_frames",
    "conflict_frames",
    "frame_review_frames",
    "frame_fail_frames",
    "frame_review_fraction",
    "frame_fail_fraction",
    "frame_review_reasons",
    "total_root_length_drop_frames",
    "max_total_root_length_drop_mm",
    "max_bbox_center_jump_px",
    "max_shoot_center_jump_px",
    "max_seedling_center_jump_px",
    "selected_root_metric_mode",
    "selected_root_metric_column",
    "final_selected_root_length_mm",
    "delta_selected_root_length_mm",
    "mean_selected_root_growth_rate_mm_per_day",
    "max_selected_root_growth_rate_mm_per_day",
    "final_total_root_length_mm",
    "mean_total_root_growth_rate_mm_per_day",
    "max_total_root_growth_rate_mm_per_day",
    "final_primary_root_length_mm",
    "final_lateral_root_length_mm",
    "final_shoot_area_mm2",
    "mean_shoot_area_growth_rate_mm2_per_day",
    "max_shoot_area_growth_rate_mm2_per_day",
    "final_shoot_area_px",
)
OWNERSHIP_FRAME_REVIEW_COLUMNS = (
    "Series",
    "PetriDish",
    "plant_id",
    "Timestamp",
    "FrameIndex",
    "frame_index",
    DERIVED_FRAME_REVIEW_STATUS_COLUMN,
    DERIVED_FRAME_REVIEW_REASON_COUNT_COLUMN,
    DERIVED_FRAME_REVIEW_REASONS_COLUMN,
    "ownership_valid",
    "ownership_root_present",
    "ownership_shoot_present",
    "ownership_conflict",
    DERIVED_SELECTED_ROOT_MODE_COLUMN,
    DERIVED_SELECTED_ROOT_COLUMN_COLUMN,
    DERIVED_SELECTED_ROOT_LENGTH_COLUMN,
    DERIVED_TOTAL_ROOT_COLUMN,
    DERIVED_TOTAL_ROOT_WEIGHTED_COLUMN,
    DERIVED_PRIMARY_ROOT_COLUMN,
    DERIVED_PRIMARY_ROOT_WEIGHTED_COLUMN,
    DERIVED_LATERAL_ROOT_COLUMN,
    DERIVED_LATERAL_ROOT_WEIGHTED_COLUMN,
    DERIVED_SHOOT_AREA_COLUMN,
    DERIVED_SHOOT_PIXELS_COLUMN,
    "shoot_measurement_source",
    "shoot_rgb_green_only_enabled",
    "ownership_shoot_mask_source",
    "ownership_anchor_mask_source",
    "ownership_anchor_rgb_green_only_enabled",
    "delta_ownership_total_root_length_mm",
    "delta_ownership_total_root_length_weighted_mm",
    "delta_ownership_primary_root_length_mm",
    "delta_ownership_primary_root_length_weighted_mm",
    "delta_ownership_lateral_root_length_mm",
    "delta_ownership_lateral_root_length_weighted_mm",
    "ownership_bbox_center_jump_px",
    "ownership_shoot_center_jump_px",
    "ownership_seedling_center_jump_px",
    DERIVED_ELAPSED_HOURS_COLUMN,
    DERIVED_TIME_DELTA_SOURCE_COLUMN,
    "seedling_bbox_x",
    "seedling_bbox_y",
    "seedling_bbox_w",
    "seedling_bbox_h",
    "seedling_center_x",
    "seedling_center_y",
    "SourceFile",
    "OutputMaskPath",
    "PreviewImagePath",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "shoot_bbox_x",
    "shoot_bbox_y",
    "shoot_bbox_w",
    "shoot_bbox_h",
)

OWNERSHIP_QUALITY_GATED_SOURCE_COLUMNS = (
    "root_length_px",
    "root_length_px_clean",
    "root_length_mm_raw",
    "root_length_mm_clean",
    "root_length_weighted_px",
    "root_length_weighted_px_clean",
    "root_length_weighted_mm_raw",
    "root_length_weighted_mm_clean",
    "root_area_px",
    "root_area_mm2",
    "root_perimeter_px",
    "root_perimeter_mm",
    "tips",
    "branches",
    "tip_count_raw",
    "base_tip_angle_deg",
    "emergence_angle_deg",
    "convex_hull_area_px2",
    "convex_hull_area_mm2",
    "aspect_ratio",
    "primary_root_length_px",
    "primary_root_length_px_clean",
    "primary_root_length_mm_raw",
    "primary_root_length_mm_clean",
    "primary_root_length_weighted_px",
    "primary_root_length_weighted_px_clean",
    "primary_root_length_weighted_mm_raw",
    "primary_root_length_weighted_mm_clean",
    "primary_root_area_px",
    "primary_root_area_mm2",
    "total_root_length_px",
    "total_root_length_px_clean",
    "total_root_length_mm_raw",
    "total_root_length_mm_clean",
    "total_root_length_weighted_px",
    "total_root_length_weighted_px_clean",
    "total_root_length_weighted_mm_raw",
    "total_root_length_weighted_mm_clean",
    "total_root_area_px",
    "total_root_area_mm2",
    "lateral_count",
    "lateral_total_length_px",
    "lateral_total_length_mm",
    "lateral_total_length_weighted_px",
    "lateral_total_length_weighted_mm",
    "lateral_root_length_weighted_mm",
    "lateral_mean_length_px",
    "lateral_mean_length_mm",
    "lateral_mean_diameter_px",
    "lateral_mean_diameter_mm",
    "lateral_max_length_px",
    "lateral_max_length_mm",
    "shoot_area_px",
    "shoot_area_mm2",
)


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def normalize_ownership_root_metric_mode(value: object) -> str:
    mode = str(value or "total").strip().lower()
    return mode if mode in OWNERSHIP_ROOT_METRIC_COLUMNS else "total"


def _sort_ownership_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Timestamp" in out.columns:
        out["_sort_timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
    else:
        out["_sort_timestamp"] = pd.NaT
    if "FrameIndex" in out.columns:
        out["_sort_frame"] = pd.to_numeric(out["FrameIndex"], errors="coerce").fillna(1.0e12)
    elif "frame_index" in out.columns:
        out["_sort_frame"] = pd.to_numeric(out["frame_index"], errors="coerce").fillna(1.0e12)
    else:
        out["_sort_frame"] = 1.0e12
    sort_columns = [column for column in ("Series", "PetriDish", "plant_id") if column in out.columns]
    out.sort_values(sort_columns + ["_sort_timestamp", "_sort_frame"], inplace=True, kind="mergesort")
    out.drop(columns=["_sort_timestamp", "_sort_frame"], inplace=True, errors="ignore")
    out.reset_index(drop=True, inplace=True)
    return out


def _numeric_values(group: pd.DataFrame, column: str | None) -> pd.Series:
    if not column or column not in group.columns:
        return pd.Series(dtype=np.float64)
    return pd.to_numeric(group[column], errors="coerce").dropna()


def _numeric_column(df: pd.DataFrame, column: str | None, default: float = 0.0) -> pd.Series:
    if not column or column not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _first_numeric_column(df: pd.DataFrame, candidates: tuple[str, ...], default: float = 0.0) -> tuple[str | None, pd.Series]:
    column = _first_existing_column(df, candidates)
    values = _numeric_column(df, column, default=default)
    diagnostic_column = f"{column}_diagnostic_pre_qa" if column else ""
    if diagnostic_column and diagnostic_column in df.columns:
        diagnostics = pd.to_numeric(df[diagnostic_column], errors="coerce")
        values = pd.to_numeric(values, errors="coerce").combine_first(diagnostics)
    return column, values


def _safe_bool_column(df: pd.DataFrame, column: str, default: bool) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    values = df[column]
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    normalized = values.fillna(default).astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def quality_gate_ownership_measurement_dataframe(master_df: pd.DataFrame | None) -> pd.DataFrame:
    if master_df is None or master_df.empty:
        return pd.DataFrame() if master_df is None else master_df.copy()

    df = master_df.copy()
    ownership_valid = _safe_bool_column(
        df,
        "ownership_measurement_valid",
        default=False,
    )
    new_diagnostic_columns: dict[str, pd.Series] = {}
    for column in OWNERSHIP_QUALITY_GATED_SOURCE_COLUMNS:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        diagnostic_column = f"{column}_diagnostic_pre_qa"
        if diagnostic_column in df.columns:
            diagnostics = pd.to_numeric(df[diagnostic_column], errors="coerce").combine_first(values)
            df[diagnostic_column] = diagnostics
        else:
            diagnostics = values
            new_diagnostic_columns[diagnostic_column] = diagnostics
        df[column] = values.where(ownership_valid, np.nan)
    if new_diagnostic_columns:
        df = pd.concat([df, pd.DataFrame(new_diagnostic_columns, index=df.index)], axis=1)
    return df


def _bbox_center(df: pd.DataFrame, prefix: str = "") -> tuple[pd.Series, pd.Series]:
    x_col = f"{prefix}bbox_x"
    y_col = f"{prefix}bbox_y"
    w_col = f"{prefix}bbox_w"
    h_col = f"{prefix}bbox_h"
    if not all(column in df.columns for column in (x_col, y_col, w_col, h_col)):
        nan = pd.Series(np.nan, index=df.index, dtype=np.float64)
        return nan, nan
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    w = pd.to_numeric(df[w_col], errors="coerce")
    h = pd.to_numeric(df[h_col], errors="coerce")
    valid = (w.fillna(0.0) > 0.0) & (h.fillna(0.0) > 0.0)
    cx = x + w / 2.0
    cy = y + h / 2.0
    return cx.where(valid), cy.where(valid)


def _assign_seedling_bbox_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    seedling_cols = ("seedling_bbox_x", "seedling_bbox_y", "seedling_bbox_w", "seedling_bbox_h")
    if not all(column in df.columns for column in seedling_cols):
        root_x = _numeric_column(df, "bbox_x", default=0.0)
        root_y = _numeric_column(df, "bbox_y", default=0.0)
        root_w = _numeric_column(df, "bbox_w", default=0.0)
        root_h = _numeric_column(df, "bbox_h", default=0.0)
        shoot_x = _numeric_column(df, "shoot_bbox_x", default=0.0)
        shoot_y = _numeric_column(df, "shoot_bbox_y", default=0.0)
        shoot_w = _numeric_column(df, "shoot_bbox_w", default=0.0)
        shoot_h = _numeric_column(df, "shoot_bbox_h", default=0.0)
        root_valid = (root_w > 0.0) & (root_h > 0.0)
        shoot_valid = (shoot_w > 0.0) & (shoot_h > 0.0)
        any_valid = root_valid | shoot_valid
        root_x0 = root_x.where(root_valid, np.inf)
        root_y0 = root_y.where(root_valid, np.inf)
        root_x1 = (root_x + root_w).where(root_valid, -np.inf)
        root_y1 = (root_y + root_h).where(root_valid, -np.inf)
        shoot_x0 = shoot_x.where(shoot_valid, np.inf)
        shoot_y0 = shoot_y.where(shoot_valid, np.inf)
        shoot_x1 = (shoot_x + shoot_w).where(shoot_valid, -np.inf)
        shoot_y1 = (shoot_y + shoot_h).where(shoot_valid, -np.inf)
        x0 = pd.Series(np.minimum(root_x0.to_numpy(), shoot_x0.to_numpy()), index=df.index).where(any_valid, 0.0)
        y0 = pd.Series(np.minimum(root_y0.to_numpy(), shoot_y0.to_numpy()), index=df.index).where(any_valid, 0.0)
        x1 = pd.Series(np.maximum(root_x1.to_numpy(), shoot_x1.to_numpy()), index=df.index).where(any_valid, 0.0)
        y1 = pd.Series(np.maximum(root_y1.to_numpy(), shoot_y1.to_numpy()), index=df.index).where(any_valid, 0.0)
        df["seedling_bbox_x"] = np.floor(x0).astype(int)
        df["seedling_bbox_y"] = np.floor(y0).astype(int)
        df["seedling_bbox_w"] = np.maximum(0.0, np.ceil(x1 - x0)).astype(int)
        df["seedling_bbox_h"] = np.maximum(0.0, np.ceil(y1 - y0)).astype(int)

    seedling_cx, seedling_cy = _bbox_center(df, prefix="seedling_")
    df["seedling_center_x"] = df.get("seedling_center_x", seedling_cx)
    df["seedling_center_y"] = df.get("seedling_center_y", seedling_cy)
    df["ownership_seedling_center_x"] = seedling_cx
    df["ownership_seedling_center_y"] = seedling_cy
    return df


def _group_delta(df: pd.DataFrame, group_columns: list[str], column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    if not group_columns:
        delta = values.diff()
        if not values.empty and pd.notna(values.iloc[0]):
            delta.iloc[0] = 0.0
        return delta
    keys = [df[group_column] for group_column in group_columns]
    delta = values.groupby(keys, dropna=False).diff()
    first_rows = df.groupby(group_columns, dropna=False).cumcount() == 0
    delta.loc[first_rows & values.notna()] = 0.0
    return delta


def _group_diff(df: pd.DataFrame, group_columns: list[str], values: pd.Series) -> pd.Series:
    if not group_columns:
        return values.diff()
    return values.groupby([df[group_column] for group_column in group_columns], dropna=False).diff()


def _group_ordinal(df: pd.DataFrame, group_columns: list[str]) -> pd.Series:
    if not group_columns:
        return pd.Series(np.arange(len(df)), index=df.index, dtype=np.int64)
    return df.groupby([df[group_column] for group_column in group_columns], dropna=False).cumcount()


def _group_elapsed_hours(df: pd.DataFrame, group_columns: list[str]) -> tuple[pd.Series, pd.Series]:
    elapsed = pd.Series(np.nan, index=df.index, dtype=np.float64)
    source = pd.Series("", index=df.index, dtype=object)
    ordinal = _group_ordinal(df, group_columns)
    not_first = ordinal > 0

    if "Timestamp" in df.columns:
        timestamps = pd.to_datetime(df["Timestamp"], errors="coerce")
        timestamp_delta = _group_diff(df, group_columns, timestamps).dt.total_seconds() / 3600.0
        timestamp_mask = not_first & (timestamp_delta > 0.0)
        elapsed.loc[timestamp_mask] = timestamp_delta.loc[timestamp_mask].astype(float)
        source.loc[timestamp_mask] = "timestamp"

    frame_col = "FrameIndex" if "FrameIndex" in df.columns else "frame_index" if "frame_index" in df.columns else None
    frame_delta = None
    if frame_col:
        frame_delta = _group_diff(df, group_columns, pd.to_numeric(df[frame_col], errors="coerce"))

    timestep_col = _first_existing_column(df, OWNERSHIP_TIMESTEP_HOURS_CANDIDATES)
    if timestep_col:
        timestep_hours = pd.to_numeric(df[timestep_col], errors="coerce")
        timestep_hours = timestep_hours.where(timestep_hours > 0.0, np.nan)
        if frame_delta is None:
            timestep_fallback = timestep_hours
        else:
            frame_multiplier = frame_delta.where(frame_delta > 0.0, 1.0).fillna(1.0)
            timestep_fallback = frame_multiplier * timestep_hours
        timestep_mask = not_first & elapsed.isna() & (timestep_fallback > 0.0)
        elapsed.loc[timestep_mask] = timestep_fallback.loc[timestep_mask].astype(float)
        source.loc[timestep_mask] = str(timestep_col)

    if frame_delta is not None:
        default_frame_fallback = frame_delta.where(frame_delta > 0.0, 1.0).fillna(1.0)
        frame_mask = not_first & elapsed.isna() & (default_frame_fallback > 0.0)
        elapsed.loc[frame_mask] = default_frame_fallback.loc[frame_mask].astype(float)
        source.loc[frame_mask] = "frame_index_default_1h"

    row_order_mask = not_first & elapsed.isna()
    elapsed.loc[row_order_mask] = 1.0
    source.loc[row_order_mask] = "row_order_default_1h"
    first_mask = ~not_first
    elapsed.loc[first_mask] = 0.0
    source.loc[first_mask] = "first_frame"
    return elapsed.fillna(0.0).astype(float), source.fillna("unknown").astype(str)


def _growth_rate_per_day(delta: pd.Series, elapsed_hours: pd.Series) -> pd.Series:
    delta_values = pd.to_numeric(delta, errors="coerce").astype(float)
    hours = pd.to_numeric(elapsed_hours, errors="coerce").fillna(0.0).astype(float)
    days = hours / 24.0
    rate = pd.Series(np.nan, index=delta.index, dtype=np.float64)
    valid = (days > 0.0) & delta_values.notna()
    rate.loc[valid] = delta_values.loc[valid] / days.loc[valid]
    first = (days <= 0.0) & delta_values.eq(0.0)
    rate.loc[first] = 0.0
    return rate.replace([np.inf, -np.inf], np.nan)


def _add_reason(reasons: pd.Series, mask: pd.Series, reason: str) -> None:
    if mask.empty:
        return
    for idx in mask[mask.fillna(False)].index:
        reasons.at[idx].append(reason)


def _assign_frame_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    thresholds = OWNERSHIP_REVIEW_THRESHOLDS
    reasons = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)

    valid = df["ownership_valid"].fillna(False).astype(bool) if "ownership_valid" in df.columns else pd.Series(False, index=df.index)
    root_present = (
        df["ownership_root_present"].fillna(False).astype(bool)
        if "ownership_root_present" in df.columns
        else pd.Series(False, index=df.index)
    )
    shoot_present = (
        df["ownership_shoot_present"].fillna(False).astype(bool)
        if "ownership_shoot_present" in df.columns
        else pd.Series(False, index=df.index)
    )
    conflict = (
        df["ownership_conflict"].fillna(False).astype(bool)
        if "ownership_conflict" in df.columns
        else pd.Series(False, index=df.index)
    )
    _add_reason(reasons, ~valid, "invalid_ownership_measurement")
    _add_reason(reasons, ~root_present, "root_missing")
    _add_reason(reasons, ~shoot_present, "shoot_missing")
    _add_reason(reasons, conflict, "seedling_conflict")

    root_drop_column = (
        "diagnostic_delta_ownership_total_root_length_mm"
        if "diagnostic_delta_ownership_total_root_length_mm" in df.columns
        else "delta_ownership_total_root_length_mm"
    )
    root_drop = -_numeric_column(df, root_drop_column, default=0.0)
    _add_reason(
        reasons,
        root_drop > float(thresholds["fail_max_root_length_drop_mm_above"]),
        "large_negative_root_length_drop",
    )
    _add_reason(
        reasons,
        (root_drop > float(thresholds["warn_max_root_length_drop_mm_above"]))
        & (root_drop <= float(thresholds["fail_max_root_length_drop_mm_above"])),
        "negative_root_length_drop",
    )

    root_jump = _numeric_column(df, "ownership_bbox_center_jump_px", default=0.0)
    _add_reason(
        reasons,
        root_jump > float(thresholds["fail_max_bbox_center_jump_px_above"]),
        "large_root_bbox_jump",
    )
    _add_reason(
        reasons,
        (root_jump > float(thresholds["warn_max_bbox_center_jump_px_above"]))
        & (root_jump <= float(thresholds["fail_max_bbox_center_jump_px_above"])),
        "root_bbox_jump",
    )

    shoot_jump = _numeric_column(df, "ownership_shoot_center_jump_px", default=0.0)
    _add_reason(
        reasons,
        shoot_jump > float(thresholds["fail_max_shoot_center_jump_px_above"]),
        "large_shoot_bbox_jump",
    )
    _add_reason(
        reasons,
        (shoot_jump > float(thresholds["warn_max_shoot_center_jump_px_above"]))
        & (shoot_jump <= float(thresholds["fail_max_shoot_center_jump_px_above"])),
        "shoot_bbox_jump",
    )

    seedling_jump = _numeric_column(df, "ownership_seedling_center_jump_px", default=0.0)
    _add_reason(
        reasons,
        seedling_jump > float(thresholds["fail_max_bbox_center_jump_px_above"]),
        "large_seedling_bbox_jump",
    )
    _add_reason(
        reasons,
        (seedling_jump > float(thresholds["warn_max_bbox_center_jump_px_above"]))
        & (seedling_jump <= float(thresholds["fail_max_bbox_center_jump_px_above"])),
        "seedling_bbox_jump",
    )

    if DERIVED_TIME_DELTA_SOURCE_COLUMN in df.columns:
        time_source = df[DERIVED_TIME_DELTA_SOURCE_COLUMN].fillna("").astype(str)
        elapsed = _numeric_column(df, DERIVED_ELAPSED_HOURS_COLUMN, default=0.0)
        default_time = time_source.str.contains("default_1h", regex=False) & (elapsed > 0.0)
        _add_reason(reasons, default_time, "default_time_delta_used")

    fail_reasons = {
        "invalid_ownership_measurement",
        "root_missing",
        "large_negative_root_length_drop",
        "large_root_bbox_jump",
        "large_shoot_bbox_jump",
        "large_seedling_bbox_jump",
    }
    reason_text = reasons.map(lambda values: ";".join(str(value) for value in values))
    reason_count = reasons.map(len).astype(int)
    status = reasons.map(
        lambda values: "fail" if any(value in fail_reasons for value in values) else "review" if values else "ok"
    )
    df[DERIVED_FRAME_REVIEW_STATUS_COLUMN] = status.astype(str)
    df[DERIVED_FRAME_REVIEW_REASON_COUNT_COLUMN] = reason_count
    df[DERIVED_FRAME_REVIEW_REASONS_COLUMN] = reason_text.astype(str)
    return df


def _group_center_jump(df: pd.DataFrame, group_columns: list[str], x_col: str, y_col: str) -> pd.Series:
    if x_col not in df.columns or y_col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=np.float64)
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    if group_columns:
        dx = x.groupby([df[group_column] for group_column in group_columns], dropna=False).diff()
        dy = y.groupby([df[group_column] for group_column in group_columns], dropna=False).diff()
    else:
        dx = x.diff()
        dy = y.diff()
    jump = np.sqrt((dx.astype(float) ** 2) + (dy.astype(float) ** 2))
    return pd.Series(jump, index=df.index).fillna(0.0)


def build_lazy_ownership_detail_dataframe(
    master_df: pd.DataFrame | None,
    root_metric_mode: object = "total",
) -> pd.DataFrame:
    if master_df is None or master_df.empty:
        return pd.DataFrame()

    df = _sort_ownership_rows(master_df)
    validity_known = "ownership_measurement_valid" in df.columns
    df["ownership_validity_known"] = bool(validity_known)
    df["ownership_valid"] = _safe_bool_column(
        df,
        "ownership_measurement_valid",
        default=False,
    )
    ownership_valid = df["ownership_valid"].fillna(False).astype(bool)
    selected_mode = normalize_ownership_root_metric_mode(root_metric_mode)
    selected_col = OWNERSHIP_ROOT_METRIC_COLUMNS[selected_mode]
    explicit_total_col, explicit_total = _first_numeric_column(
        df,
        ("total_root_length_mm_clean", "total_root_length_mm_raw"),
        default=np.nan,
    )
    explicit_total_weighted_col, explicit_total_weighted = _first_numeric_column(
        df,
        OWNERSHIP_TOTAL_WEIGHTED_METRIC_CANDIDATES,
        default=np.nan,
    )
    _primary_col, primary = _first_numeric_column(df, OWNERSHIP_PRIMARY_METRIC_CANDIDATES, default=np.nan)
    primary_weighted_col, primary_weighted = _first_numeric_column(
        df,
        OWNERSHIP_PRIMARY_WEIGHTED_METRIC_CANDIDATES,
        default=np.nan,
    )
    _lateral_col, lateral = _first_numeric_column(df, OWNERSHIP_LATERAL_METRIC_CANDIDATES, default=np.nan)
    lateral_weighted_col, lateral_weighted = _first_numeric_column(
        df,
        OWNERSHIP_LATERAL_WEIGHTED_METRIC_CANDIDATES,
        default=np.nan,
    )
    _shoot_col, shoot_area = _first_numeric_column(df, OWNERSHIP_SHOOT_AREA_CANDIDATES, default=np.nan)
    _shoot_px_col, shoot_px = _first_numeric_column(df, OWNERSHIP_SHOOT_PIXELS_CANDIDATES, default=np.nan)

    df[DERIVED_PRIMARY_ROOT_COLUMN] = primary.astype(float)
    df[DERIVED_LATERAL_ROOT_COLUMN] = lateral.astype(float)
    if explicit_total_col:
        df[DERIVED_TOTAL_ROOT_COLUMN] = explicit_total.astype(float)
    else:
        df[DERIVED_TOTAL_ROOT_COLUMN] = (primary + lateral).astype(float)
    df[DERIVED_PRIMARY_ROOT_WEIGHTED_COLUMN] = (
        primary_weighted.astype(float) if primary_weighted_col else df[DERIVED_PRIMARY_ROOT_COLUMN].astype(float)
    )
    df[DERIVED_LATERAL_ROOT_WEIGHTED_COLUMN] = (
        lateral_weighted.astype(float) if lateral_weighted_col else df[DERIVED_LATERAL_ROOT_COLUMN].astype(float)
    )
    if explicit_total_weighted_col:
        df[DERIVED_TOTAL_ROOT_WEIGHTED_COLUMN] = explicit_total_weighted.astype(float)
        weighted_source = str(explicit_total_weighted_col)
    elif primary_weighted_col or lateral_weighted_col:
        df[DERIVED_TOTAL_ROOT_WEIGHTED_COLUMN] = (
            df[DERIVED_PRIMARY_ROOT_WEIGHTED_COLUMN] + df[DERIVED_LATERAL_ROOT_WEIGHTED_COLUMN]
        ).astype(float)
        weighted_source = "+".join(
            column
            for column in (
                str(primary_weighted_col) if primary_weighted_col else DERIVED_PRIMARY_ROOT_COLUMN,
                str(lateral_weighted_col) if lateral_weighted_col else DERIVED_LATERAL_ROOT_COLUMN,
            )
            if column
        )
    else:
        df[DERIVED_TOTAL_ROOT_WEIGHTED_COLUMN] = df[DERIVED_TOTAL_ROOT_COLUMN].astype(float)
        weighted_source = DERIVED_TOTAL_ROOT_COLUMN
    df["ownership_total_root_length_weighted_source_column"] = weighted_source
    df[DERIVED_SHOOT_AREA_COLUMN] = shoot_area.astype(float)
    df[DERIVED_SHOOT_PIXELS_COLUMN] = shoot_px.astype(float)
    df[DERIVED_PRIMARY_PLUS_LATERAL_ROOT_COLUMN] = (
        df[DERIVED_PRIMARY_ROOT_COLUMN] + df[DERIVED_LATERAL_ROOT_COLUMN]
    ).astype(float)
    df[DERIVED_SELECTED_ROOT_MODE_COLUMN] = selected_mode
    df[DERIVED_SELECTED_ROOT_COLUMN_COLUMN] = selected_col
    df[DERIVED_SELECTED_ROOT_LENGTH_COLUMN] = pd.to_numeric(df[selected_col], errors="coerce")

    analysis_columns = (
        DERIVED_TOTAL_ROOT_COLUMN,
        DERIVED_TOTAL_ROOT_WEIGHTED_COLUMN,
        DERIVED_PRIMARY_ROOT_COLUMN,
        DERIVED_PRIMARY_ROOT_WEIGHTED_COLUMN,
        DERIVED_LATERAL_ROOT_COLUMN,
        DERIVED_LATERAL_ROOT_WEIGHTED_COLUMN,
        DERIVED_SHOOT_AREA_COLUMN,
        DERIVED_SHOOT_PIXELS_COLUMN,
        DERIVED_PRIMARY_PLUS_LATERAL_ROOT_COLUMN,
        DERIVED_SELECTED_ROOT_LENGTH_COLUMN,
    )
    for column in analysis_columns:
        df[f"diagnostic_{column}"] = pd.to_numeric(df[column], errors="coerce")
        df[column] = pd.to_numeric(df[column], errors="coerce").where(ownership_valid, np.nan)
    df = quality_gate_ownership_measurement_dataframe(df)

    denom = df[DERIVED_PRIMARY_PLUS_LATERAL_ROOT_COLUMN].replace(0.0, np.nan)
    df["ownership_primary_fraction_of_root_length"] = (
        pd.to_numeric(df[DERIVED_PRIMARY_ROOT_COLUMN], errors="coerce") / denom
    )
    df["ownership_lateral_fraction_of_root_length"] = (
        pd.to_numeric(df[DERIVED_LATERAL_ROOT_COLUMN], errors="coerce") / denom
    )
    shoot_denom = df[DERIVED_SHOOT_AREA_COLUMN].replace(0.0, np.nan)
    df["ownership_selected_root_length_mm_per_shoot_area_mm2"] = (
        pd.to_numeric(df[DERIVED_SELECTED_ROOT_LENGTH_COLUMN], errors="coerce") / shoot_denom
    )

    root_area_candidates = ("total_root_area_px", "root_area_px")
    _root_area_col, root_area = _first_numeric_column(
        df,
        root_area_candidates,
        default=0.0,
    )
    diagnostic_total_root = pd.to_numeric(
        df.get(
            f"diagnostic_{DERIVED_TOTAL_ROOT_COLUMN}",
            df[DERIVED_TOTAL_ROOT_COLUMN],
        ),
        errors="coerce",
    ).fillna(0.0)
    df["ownership_root_present"] = (
        (diagnostic_total_root > 0.0) | (root_area > 0.0)
    )
    df["ownership_shoot_present"] = (df[DERIVED_SHOOT_AREA_COLUMN] > 0.0) | (df[DERIVED_SHOOT_PIXELS_COLUMN] > 0.0)
    conflict_size = _numeric_column(df, "conflict_group_size", default=1.0)
    conflict_touching = _safe_bool_column(df, "conflict_touching_now", default=False)
    df["ownership_conflict"] = (conflict_size > 1.0) | conflict_touching
    root_cx, root_cy = _bbox_center(df)
    shoot_cx, shoot_cy = _bbox_center(df, prefix="shoot_")
    df["ownership_bbox_center_x"] = root_cx
    df["ownership_bbox_center_y"] = root_cy
    df["ownership_shoot_center_x"] = shoot_cx
    df["ownership_shoot_center_y"] = shoot_cy
    df = _assign_seedling_bbox_columns(df)

    group_columns = [column for column in ("Series", "PetriDish", "plant_id") if column in df.columns]
    for source, delta in (
        (DERIVED_TOTAL_ROOT_COLUMN, "delta_ownership_total_root_length_mm"),
        (DERIVED_TOTAL_ROOT_WEIGHTED_COLUMN, "delta_ownership_total_root_length_weighted_mm"),
        (DERIVED_PRIMARY_ROOT_COLUMN, "delta_ownership_primary_root_length_mm"),
        (DERIVED_PRIMARY_ROOT_WEIGHTED_COLUMN, "delta_ownership_primary_root_length_weighted_mm"),
        (DERIVED_LATERAL_ROOT_COLUMN, "delta_ownership_lateral_root_length_mm"),
        (DERIVED_LATERAL_ROOT_WEIGHTED_COLUMN, "delta_ownership_lateral_root_length_weighted_mm"),
        (DERIVED_PRIMARY_PLUS_LATERAL_ROOT_COLUMN, "delta_ownership_primary_plus_lateral_root_length_mm"),
        (DERIVED_SELECTED_ROOT_LENGTH_COLUMN, "delta_ownership_selected_root_length_mm"),
        (DERIVED_SHOOT_AREA_COLUMN, "delta_ownership_shoot_area_mm2"),
        (DERIVED_SHOOT_PIXELS_COLUMN, "delta_ownership_shoot_area_px"),
    ):
        df[delta] = _group_delta(df, group_columns, source)
        diagnostic_source = f"diagnostic_{source}"
        if diagnostic_source in df.columns:
            df[f"diagnostic_{delta}"] = _group_delta(df, group_columns, diagnostic_source)
    elapsed_hours, time_delta_source = _group_elapsed_hours(df, group_columns)
    df[DERIVED_ELAPSED_HOURS_COLUMN] = elapsed_hours
    df[DERIVED_ELAPSED_DAYS_COLUMN] = elapsed_hours / 24.0
    df[DERIVED_TIME_DELTA_SOURCE_COLUMN] = time_delta_source
    for delta, rate in (
        ("delta_ownership_total_root_length_mm", DERIVED_TOTAL_ROOT_RATE_COLUMN),
        ("delta_ownership_total_root_length_weighted_mm", DERIVED_TOTAL_ROOT_WEIGHTED_RATE_COLUMN),
        ("delta_ownership_primary_root_length_mm", DERIVED_PRIMARY_ROOT_RATE_COLUMN),
        ("delta_ownership_primary_root_length_weighted_mm", DERIVED_PRIMARY_ROOT_WEIGHTED_RATE_COLUMN),
        ("delta_ownership_lateral_root_length_mm", DERIVED_LATERAL_ROOT_RATE_COLUMN),
        ("delta_ownership_lateral_root_length_weighted_mm", DERIVED_LATERAL_ROOT_WEIGHTED_RATE_COLUMN),
        ("delta_ownership_primary_plus_lateral_root_length_mm", DERIVED_PRIMARY_PLUS_LATERAL_ROOT_RATE_COLUMN),
        ("delta_ownership_selected_root_length_mm", DERIVED_SELECTED_ROOT_RATE_COLUMN),
        ("delta_ownership_shoot_area_mm2", DERIVED_SHOOT_AREA_RATE_COLUMN),
        ("delta_ownership_shoot_area_px", DERIVED_SHOOT_PIXELS_RATE_COLUMN),
    ):
        df[rate] = _growth_rate_per_day(df[delta], elapsed_hours)
    df["ownership_bbox_center_jump_px"] = _group_center_jump(
        df,
        group_columns,
        "ownership_bbox_center_x",
        "ownership_bbox_center_y",
    )
    df["ownership_shoot_center_jump_px"] = _group_center_jump(
        df,
        group_columns,
        "ownership_shoot_center_x",
        "ownership_shoot_center_y",
    )
    df["ownership_seedling_center_jump_px"] = _group_center_jump(
        df,
        group_columns,
        "ownership_seedling_center_x",
        "ownership_seedling_center_y",
    )
    df = _assign_frame_review_columns(df)
    return df


def _metric_summary(row: dict[str, object], group: pd.DataFrame, column: str | None, prefix: str) -> None:
    values = _numeric_values(group, column)
    row[f"{prefix}_metric_column"] = str(column or "")
    if values.empty:
        row[f"first_{prefix}"] = np.nan
        row[f"final_{prefix}"] = np.nan
        row[f"max_{prefix}"] = np.nan
        row[f"delta_{prefix}"] = np.nan
        return
    first = float(values.iloc[0])
    final = float(values.iloc[-1])
    row[f"first_{prefix}"] = first
    row[f"final_{prefix}"] = final
    row[f"max_{prefix}"] = float(values.max())
    row[f"delta_{prefix}"] = final - first


def _rate_summary(row: dict[str, object], group: pd.DataFrame, column: str | None, prefix: str) -> None:
    row[f"{prefix}_metric_column"] = str(column or "")
    if not column or column not in group.columns:
        row[f"mean_{prefix}"] = np.nan
        row[f"max_{prefix}"] = np.nan
        row[f"min_{prefix}"] = np.nan
        return
    values = pd.to_numeric(group[column], errors="coerce")
    if DERIVED_ELAPSED_HOURS_COLUMN in group.columns:
        elapsed = pd.to_numeric(group[DERIVED_ELAPSED_HOURS_COLUMN], errors="coerce").fillna(0.0)
        values = values[elapsed > 0.0]
    values = values.dropna()
    if values.empty:
        row[f"mean_{prefix}"] = np.nan
        row[f"max_{prefix}"] = np.nan
        row[f"min_{prefix}"] = np.nan
        return
    row[f"mean_{prefix}"] = float(round(float(values.mean()), 4))
    row[f"max_{prefix}"] = float(round(float(values.max()), 4))
    row[f"min_{prefix}"] = float(round(float(values.min()), 4))


def _fraction(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, float(numerator) / denominator)))


def _ownership_stability_score(row: dict[str, object]) -> float:
    observed_fraction = float(row.get("observed_frame_fraction", 0.0) or 0.0)
    valid_fraction = float(row.get("valid_ownership_frame_fraction", 0.0) or 0.0)
    root_fraction = float(row.get("root_present_frame_fraction", 0.0) or 0.0)
    shoot_fraction = float(row.get("shoot_present_frame_fraction", 0.0) or 0.0)
    conflict_fraction = float(row.get("conflict_frame_fraction", 0.0) or 0.0)
    max_drop = float(row.get("max_total_root_length_drop_mm", 0.0) or 0.0)
    max_jump = float(row.get("max_bbox_center_jump_px", 0.0) or 0.0)
    base_score = 100.0 * (
        0.25 * observed_fraction
        + 0.30 * valid_fraction
        + 0.20 * root_fraction
        + 0.10 * shoot_fraction
        + 0.15 * (1.0 - conflict_fraction)
    )
    drop_penalty = min(20.0, max(0.0, max_drop) * 2.0)
    jump_penalty = min(15.0, max(0.0, max_jump - 250.0) / 25.0)
    return float(round(max(0.0, min(100.0, base_score - drop_penalty - jump_penalty)), 3))


def _row_float(row: dict[str, object] | pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
    except AttributeError:
        value = default
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _ownership_review_reasons(row: dict[str, object] | pd.Series) -> list[str]:
    thresholds = OWNERSHIP_REVIEW_THRESHOLDS
    reasons: list[str] = []

    score = _row_float(row, "ownership_stability_score_0_100", 100.0)
    observed_fraction = _row_float(row, "observed_frame_fraction", 1.0)
    valid_fraction = _row_float(row, "valid_ownership_frame_fraction", 1.0)
    root_fraction = _row_float(row, "root_present_frame_fraction", 1.0)
    shoot_fraction = _row_float(row, "shoot_present_frame_fraction", 1.0)
    conflict_fraction = _row_float(row, "conflict_frame_fraction", 0.0)
    max_root_drop = _row_float(row, "max_total_root_length_drop_mm", 0.0)
    max_bbox_jump = _row_float(row, "max_bbox_center_jump_px", 0.0)
    max_shoot_jump = _row_float(row, "max_shoot_center_jump_px", 0.0)

    if score < thresholds["fail_stability_score_below"]:
        reasons.append("very_low_stability_score")
    elif score < thresholds["warn_stability_score_below"]:
        reasons.append("low_stability_score")

    if observed_fraction < thresholds["fail_observed_frame_fraction_below"]:
        reasons.append("very_low_frame_coverage")
    elif observed_fraction < thresholds["warn_observed_frame_fraction_below"]:
        reasons.append("low_frame_coverage")

    if valid_fraction < thresholds["fail_valid_ownership_frame_fraction_below"]:
        reasons.append("many_invalid_ownership_frames")
    elif valid_fraction < thresholds["warn_valid_ownership_frame_fraction_below"]:
        reasons.append("some_invalid_ownership_frames")

    if root_fraction < thresholds["fail_root_present_frame_fraction_below"]:
        reasons.append("root_missing_most_frames")
    elif root_fraction < thresholds["warn_root_present_frame_fraction_below"]:
        reasons.append("root_missing_many_frames")

    if shoot_fraction < thresholds["fail_shoot_present_frame_fraction_below"]:
        reasons.append("shoot_missing_most_frames")
    elif shoot_fraction < thresholds["warn_shoot_present_frame_fraction_below"]:
        reasons.append("shoot_missing_many_frames")

    if conflict_fraction > thresholds["fail_conflict_frame_fraction_above"]:
        reasons.append("severe_seedling_conflicts")
    elif conflict_fraction > thresholds["warn_conflict_frame_fraction_above"]:
        reasons.append("seedling_conflicts")

    if max_root_drop > thresholds["fail_max_root_length_drop_mm_above"]:
        reasons.append("large_negative_root_length_drop")
    elif max_root_drop > thresholds["warn_max_root_length_drop_mm_above"]:
        reasons.append("negative_root_length_drop")

    if max_bbox_jump > thresholds["fail_max_bbox_center_jump_px_above"]:
        reasons.append("large_root_bbox_jump")
    elif max_bbox_jump > thresholds["warn_max_bbox_center_jump_px_above"]:
        reasons.append("root_bbox_jump")

    if max_shoot_jump > thresholds["fail_max_shoot_center_jump_px_above"]:
        reasons.append("large_shoot_bbox_jump")
    elif max_shoot_jump > thresholds["warn_max_shoot_center_jump_px_above"]:
        reasons.append("shoot_bbox_jump")

    return reasons


def _ownership_review_status(row: dict[str, object] | pd.Series, reasons: list[str]) -> str:
    if not reasons:
        return "ok"
    thresholds = OWNERSHIP_REVIEW_THRESHOLDS
    fail_checks = (
        _row_float(row, "ownership_stability_score_0_100", 100.0) < thresholds["fail_stability_score_below"],
        _row_float(row, "observed_frame_fraction", 1.0) < thresholds["fail_observed_frame_fraction_below"],
        _row_float(row, "valid_ownership_frame_fraction", 1.0) < thresholds["fail_valid_ownership_frame_fraction_below"],
        _row_float(row, "root_present_frame_fraction", 1.0) < thresholds["fail_root_present_frame_fraction_below"],
        _row_float(row, "shoot_present_frame_fraction", 1.0) < thresholds["fail_shoot_present_frame_fraction_below"],
        _row_float(row, "conflict_frame_fraction", 0.0) > thresholds["fail_conflict_frame_fraction_above"],
        _row_float(row, "max_total_root_length_drop_mm", 0.0) > thresholds["fail_max_root_length_drop_mm_above"],
        _row_float(row, "max_bbox_center_jump_px", 0.0) > thresholds["fail_max_bbox_center_jump_px_above"],
        _row_float(row, "max_shoot_center_jump_px", 0.0) > thresholds["fail_max_shoot_center_jump_px_above"],
    )
    return "fail" if any(fail_checks) else "review"


def build_lazy_ownership_track_summary(
    master_df: pd.DataFrame | None,
    root_metric_mode: object = "total",
) -> pd.DataFrame:
    if master_df is None or master_df.empty:
        return pd.DataFrame()

    selected_mode = normalize_ownership_root_metric_mode(root_metric_mode)
    selected_col = OWNERSHIP_ROOT_METRIC_COLUMNS[selected_mode]
    df = build_lazy_ownership_detail_dataframe(master_df, root_metric_mode=selected_mode)
    group_columns = [column for column in ("Series", "PetriDish", "plant_id") if column in df.columns]
    if "plant_id" not in group_columns:
        return pd.DataFrame()

    total_col = DERIVED_TOTAL_ROOT_COLUMN
    total_weighted_col = DERIVED_TOTAL_ROOT_WEIGHTED_COLUMN
    primary_col = DERIVED_PRIMARY_ROOT_COLUMN
    primary_weighted_col = DERIVED_PRIMARY_ROOT_WEIGHTED_COLUMN
    lateral_col = DERIVED_LATERAL_ROOT_COLUMN
    lateral_weighted_col = DERIVED_LATERAL_ROOT_WEIGHTED_COLUMN
    selected_length_col = DERIVED_SELECTED_ROOT_LENGTH_COLUMN
    shoot_col = DERIVED_SHOOT_AREA_COLUMN
    shoot_px_col = DERIVED_SHOOT_PIXELS_COLUMN
    frame_col = "FrameIndex" if "FrameIndex" in df.columns else "frame_index" if "frame_index" in df.columns else None
    plate_columns = [column for column in ("Series", "PetriDish") if column in df.columns]
    expected_frame_lookup: dict[tuple[object, ...], int] = {}
    if frame_col and plate_columns:
        for plate_key, plate_group in df.groupby(plate_columns, dropna=False, sort=False):
            if not isinstance(plate_key, tuple):
                plate_key = (plate_key,)
            expected_frame_lookup[tuple(plate_key)] = int(pd.to_numeric(plate_group[frame_col], errors="coerce").dropna().nunique())

    rows: list[dict[str, object]] = []
    for key, group in df.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: key[index] for index, column in enumerate(group_columns)}
        row["frames_observed"] = int(len(group))
        plate_key = tuple(row[column] for column in plate_columns)
        expected_frames = expected_frame_lookup.get(plate_key, int(len(group)))
        row["expected_frames"] = int(max(1, int(expected_frames), int(len(group))))
        row["observed_frame_fraction"] = _fraction(row["frames_observed"], row["expected_frames"])
        if "Timestamp" in group.columns:
            timestamps = pd.to_datetime(group["Timestamp"], errors="coerce").dropna()
            row["first_timestamp"] = timestamps.iloc[0].isoformat() if not timestamps.empty else ""
            row["final_timestamp"] = timestamps.iloc[-1].isoformat() if not timestamps.empty else ""
        if frame_col and frame_col in group.columns:
            frames = pd.to_numeric(group[frame_col], errors="coerce").dropna()
            row["first_frame_index"] = int(frames.iloc[0]) if not frames.empty else -1
            row["final_frame_index"] = int(frames.iloc[-1]) if not frames.empty else -1

        valid = group["ownership_valid"].fillna(False).astype(bool) if "ownership_valid" in group.columns else pd.Series(False, index=group.index)
        root_present = (
            group["ownership_root_present"].fillna(False).astype(bool)
            if "ownership_root_present" in group.columns
            else pd.Series(False, index=group.index)
        )
        shoot_present = (
            group["ownership_shoot_present"].fillna(False).astype(bool)
            if "ownership_shoot_present" in group.columns
            else pd.Series(False, index=group.index)
        )
        conflict = (
            group["ownership_conflict"].fillna(False).astype(bool)
            if "ownership_conflict" in group.columns
            else pd.Series(False, index=group.index)
        )
        observed = max(1, int(len(group)))
        row["valid_ownership_frames"] = int(valid.sum())
        row["invalid_ownership_frames"] = int((~valid).sum())
        row["valid_ownership_frame_fraction"] = _fraction(row["valid_ownership_frames"], observed)
        row["root_present_frames"] = int(root_present.sum())
        row["root_missing_frames"] = int((~root_present).sum())
        row["root_present_frame_fraction"] = _fraction(row["root_present_frames"], observed)
        row["shoot_present_frames"] = int(shoot_present.sum())
        row["shoot_missing_frames"] = int((~shoot_present).sum())
        row["shoot_present_frame_fraction"] = _fraction(row["shoot_present_frames"], observed)
        row["conflict_frames"] = int(conflict.sum())
        row["conflict_frame_fraction"] = _fraction(row["conflict_frames"], observed)
        if DERIVED_FRAME_REVIEW_STATUS_COLUMN in group.columns:
            frame_status = group[DERIVED_FRAME_REVIEW_STATUS_COLUMN].fillna("ok").astype(str).str.strip().str.lower()
        else:
            frame_status = pd.Series("ok", index=group.index, dtype=object)
        row["frame_review_frames"] = int((frame_status != "ok").sum())
        row["frame_fail_frames"] = int((frame_status == "fail").sum())
        row["frame_review_fraction"] = _fraction(row["frame_review_frames"], observed)
        row["frame_fail_fraction"] = _fraction(row["frame_fail_frames"], observed)
        frame_reasons: list[str] = []
        if DERIVED_FRAME_REVIEW_REASONS_COLUMN in group.columns:
            for text in group[DERIVED_FRAME_REVIEW_REASONS_COLUMN].dropna().astype(str):
                for reason in text.split(";"):
                    reason = reason.strip()
                    if reason and reason not in frame_reasons:
                        frame_reasons.append(reason)
        row["frame_review_reasons"] = ";".join(frame_reasons)

        row["selected_root_metric_mode"] = selected_mode
        row["selected_root_metric_column"] = selected_col
        _metric_summary(row, group, total_col, "total_root_length_mm")
        _metric_summary(row, group, total_weighted_col, "total_root_length_weighted_mm")
        _metric_summary(row, group, primary_col, "primary_root_length_mm")
        _metric_summary(row, group, primary_weighted_col, "primary_root_length_weighted_mm")
        _metric_summary(row, group, lateral_col, "lateral_root_length_mm")
        _metric_summary(row, group, lateral_weighted_col, "lateral_root_length_weighted_mm")
        _metric_summary(row, group, selected_length_col, "selected_root_length_mm")
        _metric_summary(row, group, shoot_col, "shoot_area_mm2")
        _metric_summary(row, group, shoot_px_col, "shoot_area_px")
        elapsed_hours = (
            pd.to_numeric(group[DERIVED_ELAPSED_HOURS_COLUMN], errors="coerce").fillna(0.0)
            if DERIVED_ELAPSED_HOURS_COLUMN in group.columns
            else pd.Series(0.0, index=group.index, dtype=np.float64)
        )
        row["track_elapsed_hours"] = float(round(float(elapsed_hours.sum()) if not elapsed_hours.empty else 0.0, 4))
        row["track_elapsed_days"] = float(round(row["track_elapsed_hours"] / 24.0, 4))
        if DERIVED_TIME_DELTA_SOURCE_COLUMN in group.columns:
            sources = [str(value) for value in group[DERIVED_TIME_DELTA_SOURCE_COLUMN].dropna().unique().tolist()]
            row["time_delta_sources"] = ";".join(sources)
        else:
            row["time_delta_sources"] = ""
        _rate_summary(row, group, DERIVED_TOTAL_ROOT_RATE_COLUMN, "total_root_growth_rate_mm_per_day")
        _rate_summary(row, group, DERIVED_TOTAL_ROOT_WEIGHTED_RATE_COLUMN, "total_root_length_weighted_growth_rate_mm_per_day")
        _rate_summary(row, group, DERIVED_PRIMARY_ROOT_RATE_COLUMN, "primary_root_growth_rate_mm_per_day")
        _rate_summary(row, group, DERIVED_PRIMARY_ROOT_WEIGHTED_RATE_COLUMN, "primary_root_length_weighted_growth_rate_mm_per_day")
        _rate_summary(row, group, DERIVED_LATERAL_ROOT_RATE_COLUMN, "lateral_root_growth_rate_mm_per_day")
        _rate_summary(row, group, DERIVED_LATERAL_ROOT_WEIGHTED_RATE_COLUMN, "lateral_root_length_weighted_growth_rate_mm_per_day")
        _rate_summary(
            row,
            group,
            DERIVED_PRIMARY_PLUS_LATERAL_ROOT_RATE_COLUMN,
            "primary_plus_lateral_root_growth_rate_mm_per_day",
        )
        _rate_summary(row, group, DERIVED_SELECTED_ROOT_RATE_COLUMN, "selected_root_growth_rate_mm_per_day")
        _rate_summary(row, group, DERIVED_SHOOT_AREA_RATE_COLUMN, "shoot_area_growth_rate_mm2_per_day")
        _rate_summary(row, group, DERIVED_SHOOT_PIXELS_RATE_COLUMN, "shoot_pixel_growth_rate_px_per_day")

        drop_column = (
            "diagnostic_delta_ownership_total_root_length_mm"
            if "diagnostic_delta_ownership_total_root_length_mm" in group.columns
            else "delta_ownership_total_root_length_mm"
        )
        drops = (
            pd.to_numeric(group[drop_column], errors="coerce").fillna(0.0)
            if drop_column in group.columns
            else pd.Series(0.0, index=group.index, dtype=np.float64)
        )
        negative_drops = drops[drops < -1.0e-6]
        row["total_root_length_drop_frames"] = int(len(negative_drops))
        row["max_total_root_length_drop_mm"] = float(round(abs(float(negative_drops.min())) if not negative_drops.empty else 0.0, 4))
        bbox_jump = (
            pd.to_numeric(group["ownership_bbox_center_jump_px"], errors="coerce").fillna(0.0)
            if "ownership_bbox_center_jump_px" in group.columns
            else pd.Series(0.0, index=group.index, dtype=np.float64)
        )
        shoot_jump = (
            pd.to_numeric(group["ownership_shoot_center_jump_px"], errors="coerce").fillna(0.0)
            if "ownership_shoot_center_jump_px" in group.columns
            else pd.Series(0.0, index=group.index, dtype=np.float64)
        )
        seedling_jump = (
            pd.to_numeric(group["ownership_seedling_center_jump_px"], errors="coerce").fillna(0.0)
            if "ownership_seedling_center_jump_px" in group.columns
            else pd.Series(0.0, index=group.index, dtype=np.float64)
        )
        row["mean_bbox_center_jump_px"] = float(round(float(bbox_jump.mean()) if not bbox_jump.empty else 0.0, 4))
        row["max_bbox_center_jump_px"] = float(round(float(bbox_jump.max()) if not bbox_jump.empty else 0.0, 4))
        row["mean_shoot_center_jump_px"] = float(round(float(shoot_jump.mean()) if not shoot_jump.empty else 0.0, 4))
        row["max_shoot_center_jump_px"] = float(round(float(shoot_jump.max()) if not shoot_jump.empty else 0.0, 4))
        row["mean_seedling_center_jump_px"] = float(round(float(seedling_jump.mean()) if not seedling_jump.empty else 0.0, 4))
        row["max_seedling_center_jump_px"] = float(round(float(seedling_jump.max()) if not seedling_jump.empty else 0.0, 4))
        row["ownership_stability_score_0_100"] = _ownership_stability_score(row)
        review_reasons = _ownership_review_reasons(row)
        row["ownership_review_status"] = _ownership_review_status(row, review_reasons)
        row["ownership_review_reasons"] = ";".join(review_reasons)
        row["ownership_review_reason_count"] = int(len(review_reasons))
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    sort_columns = [column for column in ("Series", "PetriDish", "plant_id") if column in summary.columns]
    summary.sort_values(sort_columns, inplace=True, kind="mergesort")
    summary.reset_index(drop=True, inplace=True)
    return summary


def build_lazy_ownership_review_dataframe(track_summary: pd.DataFrame | None) -> pd.DataFrame:
    if track_summary is None or track_summary.empty:
        return pd.DataFrame(columns=list(OWNERSHIP_REVIEW_COLUMNS))

    out = track_summary.copy()
    if "ownership_review_reasons" not in out.columns or "ownership_review_status" not in out.columns:
        reasons_series = out.apply(_ownership_review_reasons, axis=1)
        out["ownership_review_reasons"] = reasons_series.apply(lambda reasons: ";".join(reasons))
        out["ownership_review_reason_count"] = reasons_series.apply(len).astype(int)
        out["ownership_review_status"] = [
            _ownership_review_status(row, reasons)
            for (_, row), reasons in zip(out.iterrows(), reasons_series, strict=False)
        ]
    status = out["ownership_review_status"].fillna("ok").astype(str).str.lower()
    review_df = out[status != "ok"].copy()
    if review_df.empty:
        return pd.DataFrame(columns=[column for column in OWNERSHIP_REVIEW_COLUMNS if column in out.columns])

    status_order = {"fail": 0, "review": 1, "ok": 2}
    review_df["_review_sort_status"] = review_df["ownership_review_status"].map(status_order).fillna(3)
    review_df["_review_sort_score"] = _numeric_column(
        review_df,
        "ownership_stability_score_0_100",
        default=100.0,
    )
    sort_columns = ["_review_sort_status", "_review_sort_score"]
    for column in ("Series", "PetriDish", "plant_id"):
        if column in review_df.columns:
            sort_columns.append(column)
    review_df.sort_values(sort_columns, inplace=True, kind="mergesort")
    review_df.drop(columns=["_review_sort_status", "_review_sort_score"], inplace=True, errors="ignore")
    columns = [column for column in OWNERSHIP_REVIEW_COLUMNS if column in review_df.columns]
    return review_df[columns].reset_index(drop=True)


def build_lazy_ownership_frame_review_dataframe(detail_df: pd.DataFrame | None) -> pd.DataFrame:
    if detail_df is None or detail_df.empty:
        return pd.DataFrame(columns=list(OWNERSHIP_FRAME_REVIEW_COLUMNS))

    out = detail_df.copy()
    if DERIVED_FRAME_REVIEW_STATUS_COLUMN not in out.columns:
        out = build_lazy_ownership_detail_dataframe(out)
    status = out.get(DERIVED_FRAME_REVIEW_STATUS_COLUMN)
    if status is None:
        return pd.DataFrame(columns=[column for column in OWNERSHIP_FRAME_REVIEW_COLUMNS if column in out.columns])

    normalized_status = status.fillna("ok").astype(str).str.strip().str.lower()
    review_df = out[normalized_status != "ok"].copy()
    if review_df.empty:
        return pd.DataFrame(columns=[column for column in OWNERSHIP_FRAME_REVIEW_COLUMNS if column in out.columns])

    status_order = {"fail": 0, "review": 1, "ok": 2}
    review_df["_frame_review_sort_status"] = (
        review_df[DERIVED_FRAME_REVIEW_STATUS_COLUMN].fillna("").astype(str).str.lower().map(status_order).fillna(3)
    )
    sort_columns = ["_frame_review_sort_status"]
    for column in ("Series", "PetriDish", "plant_id"):
        if column in review_df.columns:
            sort_columns.append(column)
    if "FrameIndex" in review_df.columns:
        review_df["_frame_review_sort_frame"] = pd.to_numeric(review_df["FrameIndex"], errors="coerce").fillna(1.0e12)
        sort_columns.append("_frame_review_sort_frame")
    elif "frame_index" in review_df.columns:
        review_df["_frame_review_sort_frame"] = pd.to_numeric(review_df["frame_index"], errors="coerce").fillna(1.0e12)
        sort_columns.append("_frame_review_sort_frame")
    if "Timestamp" in review_df.columns:
        review_df["_frame_review_sort_timestamp"] = pd.to_datetime(review_df["Timestamp"], errors="coerce")
        sort_columns.append("_frame_review_sort_timestamp")
    review_df.sort_values(sort_columns, inplace=True, kind="mergesort")
    review_df.drop(
        columns=[
            "_frame_review_sort_status",
            "_frame_review_sort_frame",
            "_frame_review_sort_timestamp",
        ],
        inplace=True,
        errors="ignore",
    )
    columns = [column for column in OWNERSHIP_FRAME_REVIEW_COLUMNS if column in review_df.columns]
    return review_df[columns].reset_index(drop=True)


def _json_safe_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (bool, str, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _json_number(value: object, *, decimals: int | None = None) -> float | int | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    if decimals is not None:
        parsed = round(parsed, int(decimals))
    return float(parsed)


def _json_int(value: object) -> int | None:
    parsed = _json_number(value)
    if parsed is None:
        return None
    return int(parsed)


def _row_json_value(row: pd.Series, column: str) -> object:
    if column not in row.index:
        return None
    return _json_safe_value(row.get(column))


def _row_json_bool(row: pd.Series, column: str) -> bool | None:
    if column not in row.index:
        return None
    value = row.get(column)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _row_bbox_state(row: pd.Series, prefix: str = "") -> dict[str, int] | None:
    x = _json_int(row.get(f"{prefix}bbox_x"))
    y = _json_int(row.get(f"{prefix}bbox_y"))
    w = _json_int(row.get(f"{prefix}bbox_w"))
    h = _json_int(row.get(f"{prefix}bbox_h"))
    if x is None or y is None or w is None or h is None or w <= 0 or h <= 0:
        return None
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def _row_center_state(row: pd.Series, x_column: str, y_column: str) -> dict[str, float] | None:
    x = _json_number(row.get(x_column), decimals=3)
    y = _json_number(row.get(y_column), decimals=3)
    if x is None or y is None:
        return None
    return {"x": float(x), "y": float(y)}


def _identity_frame_state(row: pd.Series) -> dict[str, object]:
    return {
        "frame_index": _json_int(row.get("FrameIndex", row.get("frame_index"))),
        "timestamp": _row_json_value(row, "Timestamp"),
        "source_file": _row_json_value(row, "SourceFile"),
        "output_mask_path": _row_json_value(row, "OutputMaskPath"),
        "preview_image_path": _row_json_value(row, "PreviewImagePath"),
        "root_bbox": _row_bbox_state(row),
        "shoot_bbox": _row_bbox_state(row, "shoot_"),
        "seedling_bbox": _row_bbox_state(row, "seedling_"),
        "root_center": _row_center_state(row, "ownership_bbox_center_x", "ownership_bbox_center_y"),
        "shoot_center": _row_center_state(row, "ownership_shoot_center_x", "ownership_shoot_center_y"),
        "seedling_center": _row_center_state(row, "ownership_seedling_center_x", "ownership_seedling_center_y"),
        "selected_root_length_mm": _json_number(row.get(DERIVED_SELECTED_ROOT_LENGTH_COLUMN), decimals=6),
        "diagnostic_selected_root_length_mm": _json_number(
            row.get(f"diagnostic_{DERIVED_SELECTED_ROOT_LENGTH_COLUMN}"),
            decimals=6,
        ),
        "total_root_length_mm": _json_number(row.get(DERIVED_TOTAL_ROOT_COLUMN), decimals=6),
        "total_root_length_weighted_mm": _json_number(row.get(DERIVED_TOTAL_ROOT_WEIGHTED_COLUMN), decimals=6),
        "primary_root_length_mm": _json_number(row.get(DERIVED_PRIMARY_ROOT_COLUMN), decimals=6),
        "lateral_root_length_mm": _json_number(row.get(DERIVED_LATERAL_ROOT_COLUMN), decimals=6),
        "shoot_area_mm2": _json_number(row.get(DERIVED_SHOOT_AREA_COLUMN), decimals=6),
        "shoot_area_px": _json_number(row.get(DERIVED_SHOOT_PIXELS_COLUMN), decimals=3),
        "ownership_valid": _row_json_bool(row, "ownership_valid"),
        "root_present": _row_json_bool(row, "ownership_root_present"),
        "shoot_present": _row_json_bool(row, "ownership_shoot_present"),
        "conflict": _row_json_bool(row, "ownership_conflict"),
        "frame_review_status": _row_json_value(row, DERIVED_FRAME_REVIEW_STATUS_COLUMN),
        "frame_review_reasons": _row_json_value(row, DERIVED_FRAME_REVIEW_REASONS_COLUMN),
    }


def build_lazy_seedling_identity_state(
    detail_df: pd.DataFrame | None,
    track_summary: pd.DataFrame | None = None,
    *,
    root_metric_mode: object = "total",
) -> dict[str, object]:
    selected_mode = normalize_ownership_root_metric_mode(root_metric_mode)
    if detail_df is None or detail_df.empty:
        return {
            "schema_version": 2,
            "analysis_mode": "ownership",
            "selected_root_metric_mode": selected_mode,
            "track_count": 0,
            "tracks": [],
        }

    detail = build_lazy_ownership_detail_dataframe(detail_df, root_metric_mode=selected_mode)
    summary = (
        build_lazy_ownership_track_summary(detail, root_metric_mode=selected_mode)
        if track_summary is None or track_summary.empty
        else track_summary.copy()
    )
    group_columns = [column for column in ("Series", "PetriDish", "plant_id") if column in detail.columns]
    summary_lookup: dict[tuple[object, ...], pd.Series] = {}
    if not summary.empty:
        summary_group_columns = [column for column in group_columns if column in summary.columns]
        for _, summary_row in summary.iterrows():
            key = tuple(_json_safe_value(summary_row.get(column)) for column in summary_group_columns)
            summary_lookup[key] = summary_row

    tracks: list[dict[str, object]] = []
    grouped_detail = detail.groupby(group_columns, dropna=False, sort=False) if group_columns else [((), detail)]
    for key, group in grouped_detail:
        if not isinstance(key, tuple):
            key = (key,)
        group = _sort_ownership_rows(group).reset_index(drop=True)
        first = group.iloc[0]
        final = group.iloc[-1]
        valid_group = (
            group[group["ownership_valid"].fillna(False).astype(bool)]
            if "ownership_valid" in group.columns
            else group.iloc[0:0]
        )
        final_valid = valid_group.iloc[-1] if not valid_group.empty else None
        lookup_key = tuple(_json_safe_value(value) for value in key)
        summary_row = summary_lookup.get(lookup_key)
        track: dict[str, object] = {
            "series": _row_json_value(final, "Series"),
            "petri_dish": _row_json_value(final, "PetriDish"),
            "plant_id": _row_json_value(final, "plant_id"),
            "selected_root_metric_mode": selected_mode,
            "selected_root_metric_column": _row_json_value(final, DERIVED_SELECTED_ROOT_COLUMN_COLUMN),
            "frames_observed": int(len(group)),
            "first_observation": _identity_frame_state(first),
            "final_observation": _identity_frame_state(final),
            "final_valid_observation": _identity_frame_state(final_valid) if final_valid is not None else None,
            "root_class_id": _json_int(final.get("ownership_root_class_id")),
            "lateral_class_id": _json_int(final.get("ownership_lateral_class_id")),
            "shoot_class_id": _json_int(final.get("ownership_shoot_class_id")),
            "max_root_center_jump_px": _json_number(group["ownership_bbox_center_jump_px"].max(), decimals=4)
            if "ownership_bbox_center_jump_px" in group.columns
            else None,
            "max_shoot_center_jump_px": _json_number(group["ownership_shoot_center_jump_px"].max(), decimals=4)
            if "ownership_shoot_center_jump_px" in group.columns
            else None,
            "max_seedling_center_jump_px": _json_number(group["ownership_seedling_center_jump_px"].max(), decimals=4)
            if "ownership_seedling_center_jump_px" in group.columns
            else None,
        }
        if summary_row is not None:
            for source, target in (
                ("expected_frames", "expected_frames"),
                ("observed_frame_fraction", "observed_frame_fraction"),
                ("valid_ownership_frame_fraction", "valid_ownership_frame_fraction"),
                ("root_present_frame_fraction", "root_present_frame_fraction"),
                ("shoot_present_frame_fraction", "shoot_present_frame_fraction"),
                ("conflict_frames", "conflict_frames"),
                ("conflict_frame_fraction", "conflict_frame_fraction"),
                ("frame_review_frames", "frame_review_frames"),
                ("frame_fail_frames", "frame_fail_frames"),
                ("ownership_stability_score_0_100", "ownership_stability_score_0_100"),
                ("ownership_review_status", "ownership_review_status"),
                ("ownership_review_reasons", "ownership_review_reasons"),
                ("first_frame_index", "first_frame_index"),
                ("final_frame_index", "final_frame_index"),
                ("first_timestamp", "first_timestamp"),
                ("final_timestamp", "final_timestamp"),
                ("final_selected_root_length_mm", "final_selected_root_length_mm"),
                ("delta_selected_root_length_mm", "delta_selected_root_length_mm"),
                ("final_primary_root_length_mm", "final_primary_root_length_mm"),
                ("final_lateral_root_length_mm", "final_lateral_root_length_mm"),
                ("final_shoot_area_mm2", "final_shoot_area_mm2"),
                ("final_shoot_area_px", "final_shoot_area_px"),
            ):
                track[target] = _row_json_value(summary_row, source)
        tracks.append(track)

    tracks.sort(
        key=lambda row: (
            str(row.get("series") or ""),
            str(row.get("petri_dish") or ""),
            str(row.get("plant_id") or ""),
        )
    )
    return {
        "schema_version": 2,
        "analysis_mode": "ownership",
        "selected_root_metric_mode": selected_mode,
        "selected_root_metric_column": OWNERSHIP_ROOT_METRIC_COLUMNS[selected_mode],
        "track_key_columns": group_columns,
        "detail_rows": int(len(detail)),
        "track_count": int(len(tracks)),
        "tracks": tracks,
        "intended_use": (
            "Compact per-seedling continuity state for rerun anchoring, track review, and downstream data panels. "
            "Detailed per-frame geometry remains in npec_per_plant_detail.csv."
        ),
    }


def _metadata_rows(metadata: dict[str, object]) -> pd.DataFrame:
    rows = []
    for key in sorted(metadata.keys()):
        value = metadata.get(key)
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, sort_keys=True)
        rows.append({"key": key, "value": value})
    return pd.DataFrame(rows, columns=["key", "value"])


def _excel_safe_dataframe(df: pd.DataFrame | None) -> tuple[pd.DataFrame | None, int]:
    if df is None or df.empty:
        return (df, 0)
    out = df.copy()
    truncated = 0
    suffix = "... [truncated for Excel cell limit; use the CSV/JSON export for full geometry]"
    budget = max(1, min(EXCEL_SAFE_TEXT_LIMIT, EXCEL_CELL_CHAR_LIMIT - len(suffix)))
    for column in out.columns:
        if not (pd.api.types.is_object_dtype(out[column]) or pd.api.types.is_string_dtype(out[column])):
            continue
        mask = out[column].notna()
        if not bool(mask.any()):
            continue
        text_values = out.loc[mask, column].astype(str)
        long_mask = text_values.str.len() > EXCEL_SAFE_TEXT_LIMIT
        if not bool(long_mask.any()):
            continue
        long_index = text_values.index[long_mask]
        out.loc[long_index, column] = text_values.loc[long_index].str.slice(0, budget) + suffix
        truncated += int(len(long_index))
    return (out, int(truncated))


def _unique_positive_ints(df: pd.DataFrame, column: str) -> list[int]:
    if column not in df.columns:
        return []
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    result: list[int] = []
    for value in values:
        parsed = int(value)
        if parsed > 0 and parsed not in result:
            result.append(parsed)
    return result


def write_lazy_ownership_all_metrics_workbook(
    master_df: pd.DataFrame | None,
    output_dir: Path,
    *,
    timelapse_df: pd.DataFrame | None = None,
    pmi_df: pd.DataFrame | None = None,
    root_metric_mode: object = "total",
) -> tuple[Path | None, Path | None]:
    if master_df is None or master_df.empty:
        return (None, None)

    selected_mode = normalize_ownership_root_metric_mode(root_metric_mode)
    selected_col = OWNERSHIP_ROOT_METRIC_COLUMNS[selected_mode]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = output_dir / "npec_per_plant_all_metrics.xlsx"
    metadata_path = output_dir / "npec_per_plant_all_metrics_metadata.json"
    detail_csv_path = output_dir / "npec_per_plant_detail.csv"
    track_summary_csv_path = output_dir / "npec_per_plant_track_summary.csv"
    review_csv_path = output_dir / "npec_tracks_needing_review.csv"
    frame_review_csv_path = output_dir / "npec_frames_needing_review.csv"
    identity_state_path = output_dir / "npec_seedling_identity_state.json"
    detail_df = build_lazy_ownership_detail_dataframe(master_df, root_metric_mode=selected_mode)
    track_summary = build_lazy_ownership_track_summary(detail_df, root_metric_mode=selected_mode)
    review_df = build_lazy_ownership_review_dataframe(track_summary)
    frame_review_df = build_lazy_ownership_frame_review_dataframe(detail_df)
    identity_state = build_lazy_seedling_identity_state(
        detail_df,
        track_summary,
        root_metric_mode=selected_mode,
    )

    total_col = _first_existing_column(master_df, OWNERSHIP_TOTAL_METRIC_CANDIDATES)
    total_weighted_col = _first_existing_column(master_df, OWNERSHIP_TOTAL_WEIGHTED_METRIC_CANDIDATES)
    primary_col = _first_existing_column(master_df, OWNERSHIP_PRIMARY_METRIC_CANDIDATES)
    primary_weighted_col = _first_existing_column(master_df, OWNERSHIP_PRIMARY_WEIGHTED_METRIC_CANDIDATES)
    lateral_col = _first_existing_column(master_df, OWNERSHIP_LATERAL_METRIC_CANDIDATES)
    lateral_weighted_col = _first_existing_column(master_df, OWNERSHIP_LATERAL_WEIGHTED_METRIC_CANDIDATES)
    shoot_col = _first_existing_column(master_df, OWNERSHIP_SHOOT_AREA_CANDIDATES)
    sheet_names = [
        "per_plant_detail",
        "track_summary",
        "tracks_needing_review",
        "frames_needing_review",
        "run_config",
    ]
    if timelapse_df is not None and not timelapse_df.empty:
        sheet_names.insert(4, "plate_timelapse")
    if pmi_df is not None and not pmi_df.empty:
        sheet_names.insert(-1, "pmi_style")

    metadata = {
        "analysis_mode": "ownership",
        "detail_rows": int(len(master_df)),
        "tracks": int(
            master_df[[column for column in ("Series", "PetriDish", "plant_id") if column in master_df.columns]]
            .drop_duplicates()
            .shape[0]
        )
        if "plant_id" in master_df.columns
        else 0,
        "petri_dishes": int(master_df["PetriDish"].nunique()) if "PetriDish" in master_df.columns else 0,
        "series": int(master_df["Series"].nunique()) if "Series" in master_df.columns else 0,
        "total_root_metric_column": str(total_col or ""),
        "total_root_weighted_metric_column": str(total_weighted_col or ""),
        "primary_root_metric_column": str(primary_col or ""),
        "primary_root_weighted_metric_column": str(primary_weighted_col or ""),
        "lateral_root_metric_column": str(lateral_col or ""),
        "lateral_root_weighted_metric_column": str(lateral_weighted_col or ""),
        "shoot_area_metric_column": str(shoot_col or ""),
        "selected_root_metric_mode": selected_mode,
        "selected_root_metric_column": selected_col,
        "selected_root_metric_definition": (
            "Per-plant ownership selected root length. weighted_total uses weighted skeleton root-length columns "
            "when available, otherwise it falls back to the derived total root length."
        ),
        "derived_total_root_metric_column": DERIVED_TOTAL_ROOT_COLUMN,
        "derived_total_root_weighted_metric_column": DERIVED_TOTAL_ROOT_WEIGHTED_COLUMN,
        "derived_primary_root_metric_column": DERIVED_PRIMARY_ROOT_COLUMN,
        "derived_primary_root_weighted_metric_column": DERIVED_PRIMARY_ROOT_WEIGHTED_COLUMN,
        "derived_lateral_root_metric_column": DERIVED_LATERAL_ROOT_COLUMN,
        "derived_lateral_root_weighted_metric_column": DERIVED_LATERAL_ROOT_WEIGHTED_COLUMN,
        "derived_primary_plus_lateral_root_metric_column": DERIVED_PRIMARY_PLUS_LATERAL_ROOT_COLUMN,
        "derived_selected_root_metric_mode_column": DERIVED_SELECTED_ROOT_MODE_COLUMN,
        "derived_selected_root_metric_column_column": DERIVED_SELECTED_ROOT_COLUMN_COLUMN,
        "derived_selected_root_length_metric_column": DERIVED_SELECTED_ROOT_LENGTH_COLUMN,
        "derived_shoot_area_metric_column": DERIVED_SHOOT_AREA_COLUMN,
        "derived_elapsed_hours_column": DERIVED_ELAPSED_HOURS_COLUMN,
        "derived_elapsed_days_column": DERIVED_ELAPSED_DAYS_COLUMN,
        "derived_time_delta_source_column": DERIVED_TIME_DELTA_SOURCE_COLUMN,
        "derived_growth_rate_columns": DERIVED_GROWTH_RATE_COLUMNS,
        "derived_frame_review_columns": DERIVED_FRAME_REVIEW_COLUMNS,
        "growth_rate_definition": (
            "Per-plant frame-to-frame delta divided by elapsed days. Elapsed time uses Timestamp differences when "
            "available, then configured timestep_hours-style columns, then a conservative one-hour-per-frame fallback."
        ),
        "frame_review_definition": (
            "Per-frame ownership QA status and reasons. fail marks invalid ownership, missing roots, large negative "
            "root-length drops, or large root/shoot bbox jumps; review marks softer warnings such as smaller jumps, "
            "shoot gaps, seedling conflicts, or default one-hour time fallback."
        ),
        "ownership_root_class_ids": _unique_positive_ints(master_df, "ownership_root_class_id"),
        "ownership_lateral_class_ids": _unique_positive_ints(master_df, "ownership_lateral_class_id"),
        "ownership_shoot_class_ids": _unique_positive_ints(master_df, "ownership_shoot_class_id"),
        "tracks_needing_review": int(len(review_df)),
        "tracks_failed_review": int(
            (track_summary["ownership_review_status"].fillna("").astype(str).str.lower() == "fail").sum()
        )
        if "ownership_review_status" in track_summary.columns
        else 0,
        "frames_needing_review": int(len(frame_review_df)),
        "frames_failed_review": int(
            (frame_review_df[DERIVED_FRAME_REVIEW_STATUS_COLUMN].fillna("").astype(str).str.lower() == "fail").sum()
        )
        if DERIVED_FRAME_REVIEW_STATUS_COLUMN in frame_review_df.columns
        else 0,
        "per_plant_detail_csv": str(detail_csv_path),
        "track_summary_csv": str(track_summary_csv_path),
        "tracks_needing_review_csv": str(review_csv_path),
        "frames_needing_review_csv": str(frame_review_csv_path),
        "seedling_identity_state_json": str(identity_state_path),
        "csv_outputs": {
            "per_plant_detail": str(detail_csv_path),
            "track_summary": str(track_summary_csv_path),
            "tracks_needing_review": str(review_csv_path),
            "frames_needing_review": str(frame_review_csv_path),
        },
        "json_outputs": {
            "identity_state": str(identity_state_path),
        },
        "ownership_review_thresholds": OWNERSHIP_REVIEW_THRESHOLDS,
        "ownership_review_sheet": "tracks_needing_review",
        "ownership_quality_definition": (
            "Track-level score from observed frame coverage, valid ownership frames, root/shoot presence, conflict frames, "
            "large negative root-length drops, bbox center jumps, and shoot center jumps. Tracks marked review/fail are "
            "listed in the tracks_needing_review sheet and npec_tracks_needing_review.csv; individual bad frames are "
            "listed in the frames_needing_review sheet and npec_frames_needing_review.csv."
        ),
        "sheets": sheet_names,
    }

    try:
        detail_df.to_csv(detail_csv_path, index=False)
        track_summary.to_csv(track_summary_csv_path, index=False)
        review_df.to_csv(review_csv_path, index=False)
        frame_review_df.to_csv(frame_review_csv_path, index=False)
        identity_state_path.write_text(json.dumps(identity_state, indent=2) + "\n", encoding="utf-8")
        excel_detail_df, detail_truncated = _excel_safe_dataframe(detail_df)
        excel_review_df, review_truncated = _excel_safe_dataframe(review_df)
        excel_frame_review_df, frame_review_truncated = _excel_safe_dataframe(frame_review_df)
        excel_pmi_df, pmi_truncated = _excel_safe_dataframe(pmi_df)
        excel_timelapse_df, timelapse_truncated = _excel_safe_dataframe(timelapse_df)
        metadata["excel_truncated_cells"] = int(
            detail_truncated + review_truncated + frame_review_truncated + pmi_truncated + timelapse_truncated
        )
        metadata["excel_truncation_limit"] = int(EXCEL_SAFE_TEXT_LIMIT)
        with pd.ExcelWriter(workbook_path) as writer:
            if excel_detail_df is not None:
                excel_detail_df.to_excel(writer, index=False, sheet_name="per_plant_detail")
            track_summary.to_excel(writer, index=False, sheet_name="track_summary")
            if excel_review_df is not None:
                excel_review_df.to_excel(writer, index=False, sheet_name="tracks_needing_review")
            if excel_frame_review_df is not None:
                excel_frame_review_df.to_excel(writer, index=False, sheet_name="frames_needing_review")
            if excel_timelapse_df is not None and not excel_timelapse_df.empty:
                excel_timelapse_df.to_excel(writer, index=False, sheet_name="plate_timelapse")
            if excel_pmi_df is not None and not excel_pmi_df.empty:
                excel_pmi_df.to_excel(writer, index=False, sheet_name="pmi_style")
            _metadata_rows(metadata).to_excel(writer, index=False, sheet_name="run_config")
    except Exception:
        try:
            workbook_path.unlink(missing_ok=True)
        except Exception:
            pass
        return (None, None)

    try:
        metadata["all_metrics_workbook"] = str(workbook_path)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except Exception:
        metadata_path = None

    return (workbook_path, metadata_path)
