from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from . import __version__ as NPEC_APP_VERSION
from .analytics_engine import (
    AnalyticsConfig,
    OWNERSHIP_ANALYTICS_SCHEMA_VERSION,
    _effectively_grayscale_u8,
    _prune_small_skeleton_components,
    _recover_grayscale_shoot_near_crown,
    _skeletonize,
    run_temporal_analytics,
)
from .lazy_ownership_export import (
    quality_gate_ownership_measurement_dataframe,
    write_lazy_ownership_all_metrics_workbook,
)
from .models import DatasetImageItem
from .pmi_export import export_pmi_style_rows
from .root_growth_video import (
    LazyRootGrowthVideoConfig,
    generate_lazy_root_growth_video,
)


PIPELINE_VERSION = (
    f"existing-mask-ownership-v20-visual-temporal-shoot-memory-"
    f"schema-{OWNERSHIP_ANALYTICS_SCHEMA_VERSION}"
)
MANIFEST_COLUMNS = (
    "Series",
    "PetriDish",
    "Timestamp",
    "FrameIndex",
    "RelativeFolder",
    "SourceFile",
    "OutputMaskPath",
    "PreviewImagePath",
)


def _safe_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    try:
        if bool(pd.isna(value)):
            return default
    except Exception:
        pass
    return str(value)


def _safe_frame_index(value: object, default: int) -> int:
    try:
        parsed = float(value)
        if np.isfinite(parsed):
            return int(parsed)
    except Exception:
        pass
    return int(default)


def _file_signature(path: Path) -> str:
    try:
        stat = path.stat()
        return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        return f"{path}:missing"


def _plate_fingerprint(task: dict[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(PIPELINE_VERSION.encode("ascii"))
    digest.update(json.dumps(task.get("config", {}), sort_keys=True).encode("utf-8"))
    for frame in task.get("frames", []):
        if not isinstance(frame, dict):
            continue
        digest.update(_file_signature(Path(str(frame.get("SourceFile", "")))).encode("utf-8"))
        digest.update(_file_signature(Path(str(frame.get("OutputMaskPath", "")))).encode("utf-8"))
        digest.update(json.dumps(frame, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _analytics_config(values: dict[str, object]) -> AnalyticsConfig:
    return AnalyticsConfig(
        root_class_id=int(values["root_class_id"]),
        lateral_class_id=int(values["lateral_class_id"]) if values.get("lateral_class_id") else None,
        shoot_class_id=int(values["shoot_class_id"]) if values.get("shoot_class_id") else None,
        shoot_rgb_green_only_enabled=bool(values.get("shoot_rgb_green_only_enabled", True)),
        tracking_mode="arabidopsis_crown_lanes",
        expected_track_count=int(values["expected_plants"]),
        pixel_size_mm=float(values["pixel_size_mm"]),
        timestep_hours=float(values.get("timestep_hours", 1.0)),
        min_component_area=int(values.get("min_component_area", 30)),
        prune_branch_px=int(values.get("prune_branch_px", 5)),
        temporal_smoothing_enabled=True,
        temporal_alpha=0.60,
        shoot_crown_lock_enabled=True,
        shoot_tracking_enabled=True,
        shoot_temporal_crown_memory_enabled=bool(
            values.get("shoot_temporal_crown_memory_enabled", True)
        ),
        shoot_temporal_monotonic_area_enabled=bool(
            values.get("shoot_temporal_monotonic_area_enabled", True)
        ),
        shoot_temporal_max_crown_shift_px=float(
            values.get("shoot_temporal_max_crown_shift_px", 240.0)
        ),
        shoot_temporal_visual_reacquisition_enabled=bool(
            values.get("shoot_temporal_visual_reacquisition_enabled", True)
        ),
        shoot_temporal_visual_search_half_width_px=int(
            values.get("shoot_temporal_visual_search_half_width_px", 320)
        ),
        shoot_temporal_visual_search_above_px=int(
            values.get("shoot_temporal_visual_search_above_px", 360)
        ),
        shoot_temporal_visual_search_below_px=int(
            values.get("shoot_temporal_visual_search_below_px", 520)
        ),
        shoot_temporal_visual_max_shift_px=float(
            values.get("shoot_temporal_visual_max_shift_px", 520.0)
        ),
        shoot_temporal_visual_max_root_distance_px=float(
            values.get("shoot_temporal_visual_max_root_distance_px", 120.0)
        ),
        shoot_temporal_visual_root_vertical_tolerance_px=int(
            values.get("shoot_temporal_visual_root_vertical_tolerance_px", 220)
        ),
        tip_tracking_enabled=True,
        learned_owner_enabled=bool(values.get("learned_owner_enabled", False)),
        learned_owner_ranker_payload=(
            dict(values["learned_owner_ranker_payload"])
            if isinstance(values.get("learned_owner_ranker_payload"), dict)
            else None
        ),
        learned_owner_feature_set=str(
            values.get("learned_owner_feature_set", "crown_coordinates_orientation")
        ),
        learned_owner_support_margin=float(
            values.get("learned_owner_support_margin", 0.50)
        ),
        learned_owner_prior_weight_px=float(
            values.get("learned_owner_prior_weight_px", 4.0)
        ),
        learned_owner_temporal_score_bonus=float(
            values.get("learned_owner_temporal_score_bonus", 0.18)
        ),
        learned_owner_qc_margin=float(values.get("learned_owner_qc_margin", 0.10)),
    )


def _load_rgb(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    try:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        if image.shape[:2] == shape_hw:
            return image
    except Exception:
        pass
    return np.zeros((shape_hw[0], shape_hw[1], 3), dtype=np.uint8)


def _validate_plate_conservation(
    rows: list[dict[str, object]],
    masks: list[np.ndarray],
    config: AnalyticsConfig,
) -> dict[str, object]:
    max_length_error = 0.0
    max_area_error = 0
    max_class_error = 0.0
    for frame_index, mask in enumerate(masks):
        frame_rows = [row for row in rows if int(row.get("frame_index", -1)) == frame_index]
        if len(frame_rows) != int(config.expected_track_count):
            raise RuntimeError(
                f"frame {frame_index}: expected {config.expected_track_count} ownership rows, got {len(frame_rows)}"
            )
        class_ids = [int(config.root_class_id)]
        if config.lateral_class_id is not None:
            class_ids.append(int(config.lateral_class_id))
        root_union = np.isin(mask, np.asarray(class_ids, dtype=np.uint8)).astype(np.uint8)
        skeleton = _prune_small_skeleton_components(_skeletonize(root_union), config.prune_branch_px)
        expected_length = float(np.count_nonzero(skeleton))
        measured_length = float(sum(float(row.get("total_root_length_px", 0.0)) for row in frame_rows))
        expected_area = int(np.count_nonzero(root_union))
        measured_area = int(sum(int(row.get("total_root_area_px", 0)) for row in frame_rows))
        class_length = float(
            sum(
                float(row.get("primary_root_length_px", 0.0))
                + float(row.get("lateral_total_length_px", 0.0))
                for row in frame_rows
            )
        )
        length_error = abs(measured_length - expected_length)
        area_error = abs(measured_area - expected_area)
        class_error = abs(class_length - measured_length)
        max_length_error = max(max_length_error, length_error)
        max_area_error = max(max_area_error, area_error)
        max_class_error = max(max_class_error, class_error)
        if length_error > 1.0e-6 or area_error != 0 or class_error > 1.0e-6:
            raise RuntimeError(
                f"frame {frame_index}: conservation failed "
                f"(skeleton {measured_length}/{expected_length}, area {measured_area}/{expected_area}, "
                f"classes {class_length}/{measured_length})"
            )
    return {
        "frames": int(len(masks)),
        "max_length_error_px": float(max_length_error),
        "max_area_error_px": int(max_area_error),
        "max_class_error_px": float(max_class_error),
    }


def _translate_binary_mask(mask: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if binary.ndim != 2 or not np.any(binary):
        return np.zeros(binary.shape[:2], dtype=np.uint8)
    matrix = np.asarray(
        [[1.0, 0.0, float(shift_x)], [0.0, 1.0, float(shift_y)]],
        dtype=np.float32,
    )
    return (
        cv2.warpAffine(
            binary,
            matrix,
            (int(binary.shape[1]), int(binary.shape[0])),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    ).astype(np.uint8)


def _fill_shoot_area_from_temporal_memory(
    current_mask: np.ndarray,
    previous_mask: np.ndarray,
    *,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    allowed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fill a shoot dropout to the prior area using aligned same-track pixels."""
    current = (np.asarray(current_mask, dtype=np.uint8) > 0).astype(np.uint8)
    previous = (np.asarray(previous_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if current.shape != previous.shape:
        previous = (
            cv2.resize(
                previous,
                (int(current.shape[1]), int(current.shape[0])),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        ).astype(np.uint8)
    if allowed_mask is None:
        allowed = np.ones(current.shape, dtype=bool)
    else:
        allowed_arr = np.asarray(allowed_mask, dtype=np.uint8)
        if allowed_arr.shape != current.shape:
            allowed_arr = cv2.resize(
                allowed_arr,
                (int(current.shape[1]), int(current.shape[0])),
                interpolation=cv2.INTER_NEAREST,
            )
        allowed = allowed_arr > 0
    current = ((current > 0) & allowed).astype(np.uint8)
    current_area = int(np.count_nonzero(current))
    previous_area = int(np.count_nonzero(previous))
    metadata: dict[str, object] = {
        "observed_area_px": current_area,
        "previous_area_px": previous_area,
        "tracked_area_px": current_area,
        "carried_pixels": 0,
        "status": "no_previous_shoot" if previous_area <= 0 else "observed_nondecreasing",
    }
    if previous_area <= 0 or current_area >= previous_area:
        return current, metadata

    aligned = _translate_binary_mask(previous, shift_x, shift_y)
    aligned = ((aligned > 0) & allowed).astype(np.uint8)
    candidates = aligned > 0
    output = current > 0
    missing = int(previous_area - current_area)
    candidate_only = candidates & ~output
    candidate_indices = np.flatnonzero(candidate_only)
    if candidate_indices.size > missing:
        if current_area > 0:
            distance_to_current = cv2.distanceTransform(
                (~output).astype(np.uint8),
                cv2.DIST_L2,
                3,
            ).reshape(-1)
            candidate_scores = distance_to_current[candidate_indices]
            chosen_positions = np.argpartition(candidate_scores, missing - 1)[:missing]
            candidate_indices = candidate_indices[chosen_positions]
        else:
            candidate_indices = candidate_indices[:missing]
    output_flat = output.reshape(-1)
    output_flat[candidate_indices] = True
    tracked_bool = output_flat.reshape(output.shape)
    remaining = max(0, previous_area - int(np.count_nonzero(tracked_bool)))
    if remaining > 0 and np.any(tracked_bool | candidates):
        source = tracked_bool | candidates
        distance = cv2.distanceTransform(
            (~source).astype(np.uint8),
            cv2.DIST_L2,
            3,
        ).reshape(-1)
        expansion_indices = np.flatnonzero(allowed.reshape(-1) & ~tracked_bool.reshape(-1))
        if expansion_indices.size > remaining:
            scores = distance[expansion_indices]
            selected = np.argpartition(scores, remaining - 1)[:remaining]
            expansion_indices = expansion_indices[selected]
        tracked_flat = tracked_bool.reshape(-1)
        tracked_flat[expansion_indices] = True
        tracked_bool = tracked_flat.reshape(current.shape)
    tracked = tracked_bool.astype(np.uint8)
    tracked_area = int(np.count_nonzero(tracked))
    carried_pixels = max(0, tracked_area - current_area)
    metadata.update(
        {
            "tracked_area_px": tracked_area,
            "carried_pixels": carried_pixels,
            "status": "crown_aligned_previous_mask_carried",
        }
    )
    return tracked, metadata


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=np.uint8) > 0)
    if xs.size <= 0 or ys.size <= 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def _recover_grayscale_shoot_from_temporal_lane(
    image: np.ndarray,
    root_mask: np.ndarray,
    previous_mask: np.ndarray,
    *,
    lane_left: int,
    lane_right: int,
    config: AnalyticsConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reacquire a moved BW rosette using prior position and current root support."""
    root_u8 = (np.asarray(root_mask, dtype=np.uint8) > 0).astype(np.uint8)
    empty = np.zeros(root_u8.shape[:2], dtype=np.uint8)
    meta: dict[str, object] = {
        "status": "not_attempted",
        "candidate_components": 0,
        "accepted_components": 0,
        "accepted_pixels": 0,
    }
    gray = _effectively_grayscale_u8(image)
    previous_center = _mask_centroid(previous_mask)
    if gray is None or gray.shape[:2] != root_u8.shape[:2]:
        meta["status"] = "not_grayscale"
        return empty, meta
    if previous_center is None:
        meta["status"] = "no_previous_shoot"
        return empty, meta

    h, w = root_u8.shape[:2]
    half_width = max(
        32,
        int(getattr(config, "shoot_temporal_visual_search_half_width_px", 320)),
    )
    above = max(
        32,
        int(getattr(config, "shoot_temporal_visual_search_above_px", 360)),
    )
    below = max(
        32,
        int(getattr(config, "shoot_temporal_visual_search_below_px", 520)),
    )
    previous_x, previous_y = previous_center
    x0 = max(0, int(lane_left), int(np.floor(previous_x - half_width)))
    x1 = min(w, int(lane_right), int(np.ceil(previous_x + half_width)))
    y0 = max(0, int(np.floor(previous_y - above)))
    y1 = min(h, int(np.ceil(previous_y + below)))
    if x1 <= x0 or y1 <= y0:
        meta["status"] = "empty_roi"
        return empty, meta

    roi_gray = gray[y0:y1, x0:x1]
    threshold_ratio = float(
        getattr(config, "shoot_tracking_grayscale_crown_rescue_threshold_ratio", 0.56)
    )
    threshold_min = int(
        getattr(config, "shoot_tracking_grayscale_crown_rescue_threshold_min", 58)
    )
    threshold_max = int(
        getattr(config, "shoot_tracking_grayscale_crown_rescue_threshold_max", 88)
    )
    threshold = int(
        round(
            max(
                threshold_min,
                min(threshold_max, float(np.median(roi_gray)) * threshold_ratio),
            )
        )
    )
    candidate_roi = (roi_gray <= np.uint8(threshold)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate_roi = cv2.morphologyEx(candidate_roi, cv2.MORPH_OPEN, kernel)
    candidate_roi = cv2.morphologyEx(candidate_roi, cv2.MORPH_CLOSE, kernel)
    root_exclusion = cv2.dilate(root_u8, kernel, iterations=1)
    candidate_roi[root_exclusion[y0:y1, x0:x1] > 0] = 0

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate_roi,
        connectivity=8,
    )
    meta["candidate_components"] = int(max(0, component_count - 1))
    min_area = max(
        1,
        int(getattr(config, "shoot_tracking_grayscale_crown_rescue_min_component_area", 80)),
    )
    max_area = max(
        min_area,
        int(getattr(config, "shoot_tracking_grayscale_crown_rescue_max_component_area", 28000)),
    )
    max_shift = max(
        1.0,
        float(getattr(config, "shoot_temporal_visual_max_shift_px", 520.0)),
    )
    max_root_distance = max(
        1.0,
        float(getattr(config, "shoot_temporal_visual_max_root_distance_px", 120.0)),
    )
    vertical_tolerance = max(
        16,
        int(getattr(config, "shoot_temporal_visual_root_vertical_tolerance_px", 220)),
    )
    root_distance = cv2.distanceTransform(
        (root_u8 == 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    accepted_roi = np.zeros_like(candidate_roi, dtype=np.uint8)
    plausible_components: list[dict[str, object]] = []
    for component_id in range(1, int(component_count)):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        local_component = labels == component_id
        rows, cols = np.where(local_component)
        if rows.size <= 0:
            continue
        global_rows = rows + y0
        global_cols = cols + x0
        center_x = float(centroids[component_id][0]) + float(x0)
        center_y = float(centroids[component_id][1]) + float(y0)
        temporal_distance = float(
            np.hypot(center_x - previous_x, center_y - previous_y)
        )
        if temporal_distance > max_shift:
            continue
        component_root_distance = float(
            np.min(root_distance[global_rows, global_cols])
        )
        if component_root_distance > max_root_distance:
            continue

        component_left = int(stats[component_id, cv2.CC_STAT_LEFT]) + x0
        component_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        component_top = int(stats[component_id, cv2.CC_STAT_TOP]) + y0
        component_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        root_margin = int(np.ceil(max_root_distance))
        strip_left = max(int(lane_left), component_left - root_margin)
        strip_right = min(
            int(lane_right),
            component_left + component_width + root_margin,
        )
        root_rows, _root_cols = np.nonzero(root_u8[:, strip_left:strip_right])
        if root_rows.size <= 0:
            continue
        root_top = int(root_rows.min())
        component_bottom = component_top + component_height
        if component_top > root_top + vertical_tolerance:
            continue
        if component_bottom < root_top - vertical_tolerance:
            continue
        plausible_components.append(
            {
                "mask": local_component,
                "center_x": center_x,
                "center_y": center_y,
                "score": component_root_distance + 0.25 * temporal_distance,
            }
        )

    accepted_components = 0
    if plausible_components:
        plausible_components.sort(key=lambda component: float(component["score"]))
        primary = plausible_components[0]
        primary_x = float(primary["center_x"])
        primary_y = float(primary["center_y"])
        cluster_radius = max(180.0, 1.5 * max_root_distance)
        for component in plausible_components:
            cluster_distance = float(
                np.hypot(
                    float(component["center_x"]) - primary_x,
                    float(component["center_y"]) - primary_y,
                )
            )
            if component is not primary and cluster_distance > cluster_radius:
                continue
            accepted_roi[np.asarray(component["mask"], dtype=bool)] = 1
            accepted_components += 1

    recovered = np.zeros(root_u8.shape[:2], dtype=np.uint8)
    recovered[y0:y1, x0:x1] = accepted_roi
    recovered[root_u8 > 0] = 0
    accepted_pixels = int(np.count_nonzero(recovered))
    meta.update(
        {
            "status": "recovered" if accepted_pixels > 0 else "no_candidate",
            "threshold": int(threshold),
            "roi": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
            "accepted_components": int(accepted_components),
            "accepted_pixels": int(accepted_pixels),
        }
    )
    return recovered, meta


def _binary_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=np.uint8) > 0)
    if xs.size <= 0 or ys.size <= 0:
        return (0, 0, 0, 0)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def _append_source_token(value: object, token: str) -> str:
    parts = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if token not in parts:
        parts.append(token)
    return ",".join(parts)


def _plant_order_value(row: dict[str, object]) -> tuple[int, float, str]:
    track_id = str(row.get("plant_id", ""))
    digits = "".join(character for character in track_id if character.isdigit())
    if digits:
        return (0, float(int(digits)), track_id)
    try:
        crown_x = float(row.get("ownership_crown_center_x", float("nan")))
    except (TypeError, ValueError, OverflowError):
        crown_x = float("nan")
    if np.isfinite(crown_x):
        return (1, crown_x, track_id)
    return (2, 0.0, track_id)


def _update_row_from_temporal_shoot_mask(
    row: dict[str, object],
    shoot_mask: np.ndarray,
    metadata: dict[str, object],
    config: AnalyticsConfig,
    *,
    shift_x: float,
    shift_y: float,
    motion_source: str = "crown",
    visual_reacquisition: dict[str, object] | None = None,
) -> None:
    area_px = int(np.count_nonzero(shoot_mask))
    x, y, width, height = _binary_mask_bbox(shoot_mask)
    row["shoot_area_px"] = area_px
    row["shoot_area_mm2"] = float(area_px * (float(config.pixel_size_mm) ** 2))
    row["shoot_bbox_x"] = x
    row["shoot_bbox_y"] = y
    row["shoot_bbox_w"] = width
    row["shoot_bbox_h"] = height
    if area_px > 0:
        ys, xs = np.nonzero(np.asarray(shoot_mask, dtype=np.uint8) > 0)
        row["shoot_center_x"] = float(xs.mean())
        row["shoot_center_y"] = float(ys.mean())
    else:
        row["shoot_center_x"] = 0.0
        row["shoot_center_y"] = 0.0

    root_bbox = (
        int(row.get("bbox_x", 0) or 0),
        int(row.get("bbox_y", 0) or 0),
        int(row.get("bbox_w", 0) or 0),
        int(row.get("bbox_h", 0) or 0),
    )
    root_valid = root_bbox[2] > 0 and root_bbox[3] > 0
    shoot_valid = area_px > 0 and width > 0 and height > 0
    if root_valid and shoot_valid:
        sx0, sy0, sw, sh = x, y, width, height
        rx0, ry0, rw, rh = root_bbox
        ux0 = min(sx0, rx0)
        uy0 = min(sy0, ry0)
        ux1 = max(sx0 + sw, rx0 + rw)
        uy1 = max(sy0 + sh, ry0 + rh)
    elif root_valid:
        ux0, uy0, rw, rh = root_bbox
        ux1 = ux0 + rw
        uy1 = uy0 + rh
    elif shoot_valid:
        ux0, uy0 = x, y
        ux1 = x + width
        uy1 = y + height
    else:
        ux0 = uy0 = ux1 = uy1 = 0
    if root_valid or shoot_valid:
        row["seedling_bbox_x"] = ux0
        row["seedling_bbox_y"] = uy0
        row["seedling_bbox_w"] = ux1 - ux0
        row["seedling_bbox_h"] = uy1 - uy0
        row["seedling_center_x"] = float(0.5 * (ux0 + ux1))
        row["seedling_center_y"] = float(0.5 * (uy0 + uy1))
    else:
        row["seedling_bbox_x"] = 0
        row["seedling_bbox_y"] = 0
        row["seedling_bbox_w"] = 0
        row["seedling_bbox_h"] = 0
        row["seedling_center_x"] = 0.0
        row["seedling_center_y"] = 0.0

    carried_pixels = int(metadata.get("carried_pixels", 0) or 0)
    visual_meta = visual_reacquisition or {}
    visual_applied = bool(visual_meta.get("applied", False))
    row["shoot_temporal_tracking_applied"] = bool(carried_pixels > 0 or visual_applied)
    row["shoot_temporal_area_floor_applied"] = bool(carried_pixels > 0)
    row["shoot_temporal_observed_area_px"] = int(metadata.get("observed_area_px", area_px) or 0)
    row["shoot_temporal_previous_area_px"] = int(metadata.get("previous_area_px", 0) or 0)
    row["shoot_temporal_tracked_area_px"] = area_px
    row["shoot_temporal_carried_pixels"] = carried_pixels
    row["shoot_temporal_tracking_status"] = str(metadata.get("status", ""))
    row["shoot_temporal_crown_shift_x"] = float(shift_x)
    row["shoot_temporal_crown_shift_y"] = float(shift_y)
    row["shoot_temporal_motion_source"] = str(motion_source)
    row["shoot_temporal_visual_reacquisition_applied"] = bool(visual_applied)
    row["shoot_temporal_visual_reacquisition_status"] = str(
        visual_meta.get("status", "not_attempted")
    )
    row["shoot_temporal_visual_reacquisition_pixels"] = int(
        visual_meta.get("accepted_pixels", 0) or 0
    )
    row["shoot_temporal_visual_reacquisition_components"] = int(
        visual_meta.get("accepted_components", 0) or 0
    )
    if visual_applied:
        row["shoot_measurement_source"] = _append_source_token(
            row.get("shoot_measurement_source", "explicit_shoot_class"),
            "hades_bw_temporal_visual_reacquisition",
        )
    if carried_pixels > 0:
        row["shoot_measurement_source"] = _append_source_token(
            row.get("shoot_measurement_source", "explicit_shoot_class"),
            "hades_bw_temporal_crown_track",
        )


def _write_accepted_shoot_display_masks(
    payload_rows: list[dict[str, object]],
    frames: list[dict[str, object]],
    items: list[DatasetImageItem],
    masks: list[np.ndarray],
    config: AnalyticsConfig,
    output_root: Path,
) -> dict[int, str]:
    """Write video-only masks containing accepted or rescued shoot pixels."""
    if bool(config.shoot_rgb_green_only_enabled) or config.shoot_class_id is None:
        return {}
    shoot_class_id = int(config.shoot_class_id)
    if shoot_class_id <= 0:
        return {}

    rows_by_frame: dict[int, list[dict[str, object]]] = {}
    for row in payload_rows:
        if not isinstance(row, dict):
            continue
        frame_index = int(row.get("frame_index", -1))
        if 0 <= frame_index < len(masks):
            rows_by_frame.setdefault(frame_index, []).append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    output_paths: dict[int, str] = {}
    root_class_ids = {int(config.root_class_id)}
    if config.lateral_class_id is not None:
        root_class_ids.add(int(config.lateral_class_id))
    shoot_memory_by_track: dict[str, np.ndarray] = {}
    crown_memory_by_track: dict[str, tuple[float, float]] = {}

    for frame_index, original_mask in enumerate(masks):
        frame_rows = rows_by_frame.get(frame_index, [])
        if not frame_rows:
            continue
        mask_u8 = np.asarray(original_mask, dtype=np.uint8)
        accepted_shoot = np.zeros(mask_u8.shape[:2], dtype=np.uint8)
        root_union = np.isin(mask_u8, list(root_class_ids)).astype(np.uint8)

        ordered_rows = sorted(frame_rows, key=_plant_order_value)
        crown_xs: list[float] = []
        width = int(mask_u8.shape[1])
        for position, row in enumerate(ordered_rows):
            track_id = str(row.get("plant_id", f"plant_{position + 1:02d}"))
            try:
                crown_x = float(row.get("ownership_crown_center_x", float("nan")))
            except (TypeError, ValueError, OverflowError):
                crown_x = float("nan")
            previous_crown = crown_memory_by_track.get(track_id)
            if not np.isfinite(crown_x) and previous_crown is not None:
                crown_x = float(previous_crown[0])
            if not np.isfinite(crown_x):
                shoot_x = float(row.get("shoot_bbox_x", 0) or 0)
                shoot_w = float(row.get("shoot_bbox_w", 0) or 0)
                root_x = float(row.get("bbox_x", 0) or 0)
                root_w = float(row.get("bbox_w", 0) or 0)
                if shoot_w > 0:
                    crown_x = shoot_x + 0.5 * shoot_w
                elif root_w > 0:
                    crown_x = root_x + 0.5 * root_w
                else:
                    crown_x = float(width) * float(position + 0.5) / float(len(ordered_rows))
            crown_xs.append(float(max(0.0, min(float(width - 1), crown_x))))
        if len(crown_xs) > 1 and np.any(np.diff(np.asarray(crown_xs)) <= 0.0):
            crown_xs = [
                float(width) * float(position + 0.5) / float(len(ordered_rows))
                for position in range(len(ordered_rows))
            ]
        crown_x_by_track = {
            str(row.get("plant_id", f"plant_{position + 1:02d}")): crown_xs[position]
            for position, row in enumerate(ordered_rows)
        }
        lane_bounds: list[tuple[int, int]] = []
        for position, crown_x in enumerate(crown_xs):
            left = (
                0
                if position == 0
                else int(round(0.5 * (crown_xs[position - 1] + crown_x)))
            )
            right = (
                width
                if position == len(crown_xs) - 1
                else int(round(0.5 * (crown_x + crown_xs[position + 1])))
            )
            lane_bounds.append((max(0, left), min(width, max(left + 1, right))))

        frame_track_area_sum = 0
        root_distance = (
            cv2.distanceTransform((root_union == 0).astype(np.uint8), cv2.DIST_L2, 3)
            if np.any(root_union)
            else None
        )
        for position, row in enumerate(ordered_rows):
            track_id = str(row.get("plant_id", f"plant_{position + 1:02d}"))
            lane_left, lane_right = lane_bounds[position]
            lane_allowed = np.zeros(mask_u8.shape[:2], dtype=np.uint8)
            lane_allowed[:, lane_left:lane_right] = 1
            lane_allowed[root_union > 0] = 0

            crown_x = float(crown_x_by_track[track_id])
            previous_crown = crown_memory_by_track.get(track_id)
            try:
                crown_y = float(row.get("ownership_crown_center_y", float("nan")))
            except (TypeError, ValueError, OverflowError):
                crown_y = float("nan")
            if not np.isfinite(crown_y) and previous_crown is not None:
                crown_y = float(previous_crown[1])
            if not np.isfinite(crown_y):
                shoot_y = float(row.get("shoot_bbox_y", 0) or 0)
                shoot_h = float(row.get("shoot_bbox_h", 0) or 0)
                root_y = float(row.get("bbox_y", 0) or 0)
                crown_y = shoot_y + 0.5 * shoot_h if shoot_h > 0 else root_y
            crown_y = float(max(0.0, min(float(mask_u8.shape[0] - 1), crown_y)))

            observed_shoot = np.zeros(mask_u8.shape[:2], dtype=np.uint8)
            rescued = bool(row.get("shoot_grayscale_crown_rescue_applied", False))
            if rescued and frame_index < len(items):
                recovered, _meta = _recover_grayscale_shoot_near_crown(
                    items[frame_index].image,
                    root_union,
                    {
                        "crown_center_x": crown_x,
                        "crown_center_y": crown_y,
                        "lane_center_x": crown_x,
                        "fixed_top_y": crown_y,
                        "lane_left": int(lane_left),
                        "lane_right": int(lane_right),
                        "dynamic_lane_geometry": True,
                    },
                    config,
                )
                observed_shoot = np.maximum(observed_shoot, recovered)
            else:
                x = int(row.get("shoot_bbox_x", 0) or 0)
                y = int(row.get("shoot_bbox_y", 0) or 0)
                box_width = int(row.get("shoot_bbox_w", 0) or 0)
                box_height = int(row.get("shoot_bbox_h", 0) or 0)
                if box_width > 0 and box_height > 0:
                    x0 = max(lane_left, min(lane_right, x))
                    y0 = max(0, min(mask_u8.shape[0], y))
                    x1 = max(x0, min(lane_right, x + box_width))
                    y1 = max(y0, min(mask_u8.shape[0], y + box_height))
                    observed_crop = observed_shoot[y0:y1, x0:x1]
                    observed_crop[mask_u8[y0:y1, x0:x1] == shoot_class_id] = 1
            observed_shoot = ((observed_shoot > 0) & (lane_allowed > 0)).astype(np.uint8)

            previous_shoot = shoot_memory_by_track.get(track_id)
            previous_area = int(np.count_nonzero(previous_shoot)) if previous_shoot is not None else 0
            observed_area = int(np.count_nonzero(observed_shoot))
            visual_meta: dict[str, object] = {
                "status": "not_attempted",
                "accepted_components": 0,
                "accepted_pixels": 0,
                "applied": False,
            }
            if (
                bool(getattr(config, "shoot_temporal_visual_reacquisition_enabled", True))
                and previous_shoot is not None
                and previous_area > 0
                and observed_area < previous_area
                and frame_index < len(items)
            ):
                visual_candidate, visual_meta = _recover_grayscale_shoot_from_temporal_lane(
                    items[frame_index].image,
                    root_union,
                    previous_shoot,
                    lane_left=lane_left,
                    lane_right=lane_right,
                    config=config,
                )
                visual_area = int(np.count_nonzero(visual_candidate))
                if visual_area > 0:
                    current_center = _mask_centroid(observed_shoot)
                    visual_center = _mask_centroid(visual_candidate)
                    merge_distance = float(
                        getattr(config, "shoot_temporal_visual_max_root_distance_px", 120.0)
                    ) * 2.0
                    if observed_area <= 0:
                        observed_shoot = visual_candidate
                        visual_meta["applied"] = True
                    elif (
                        current_center is not None
                        and visual_center is not None
                        and float(
                            np.hypot(
                                current_center[0] - visual_center[0],
                                current_center[1] - visual_center[1],
                            )
                        )
                        <= merge_distance
                    ):
                        observed_shoot = np.maximum(observed_shoot, visual_candidate)
                        visual_meta["applied"] = True
                    elif visual_area > observed_area:
                        observed_shoot = visual_candidate
                        visual_meta["applied"] = True
                    observed_area = int(np.count_nonzero(observed_shoot))

            previous_center = _mask_centroid(previous_shoot) if previous_shoot is not None else None
            observed_center = _mask_centroid(observed_shoot)
            max_visual_shift = max(
                1.0,
                float(getattr(config, "shoot_temporal_visual_max_shift_px", 520.0)),
            )
            visual_shift_valid = False
            visual_shift_x = 0.0
            visual_shift_y = 0.0
            if previous_center is not None and observed_center is not None:
                visual_shift_x = float(observed_center[0] - previous_center[0])
                visual_shift_y = float(observed_center[1] - previous_center[1])
                visual_shift_valid = (
                    float(np.hypot(visual_shift_x, visual_shift_y)) <= max_visual_shift
                )
            if previous_area > 0 and observed_area > 0 and not visual_shift_valid:
                observed_shoot.fill(0)
                observed_area = 0
                visual_meta["status"] = "observed_rejected_excess_motion"
                visual_meta["applied"] = False
            elif (
                previous_area > 0
                and observed_area > 0
                and root_distance is not None
                and float(np.min(root_distance[observed_shoot > 0]))
                > float(getattr(config, "shoot_temporal_visual_max_root_distance_px", 120.0))
            ):
                observed_shoot.fill(0)
                observed_area = 0
                visual_meta["status"] = "observed_rejected_without_root_support"
                visual_meta["applied"] = False

            shift_x = 0.0
            shift_y = 0.0
            motion_source = "none"
            crown_jump_rejected = False
            if visual_shift_valid and observed_area > 0:
                shift_x = visual_shift_x
                shift_y = visual_shift_y
                motion_source = "shoot_visual_centroid"
            elif previous_crown is not None:
                shift_x = float(crown_x - previous_crown[0])
                shift_y = float(crown_y - previous_crown[1])
                motion_source = "root_crown"
                max_shift = max(
                    0.0,
                    float(getattr(config, "shoot_temporal_max_crown_shift_px", 240.0)),
                )
                if max_shift > 0.0 and float(np.hypot(shift_x, shift_y)) > max_shift:
                    shift_x = 0.0
                    shift_y = 0.0
                    motion_source = "unreliable_crown_jump"
                    crown_jump_rejected = True

            tracked_shoot = observed_shoot
            temporal_meta: dict[str, object] = {
                "observed_area_px": int(np.count_nonzero(observed_shoot)),
                "previous_area_px": previous_area,
                "tracked_area_px": int(np.count_nonzero(observed_shoot)),
                "carried_pixels": 0,
                "status": "temporal_memory_seeded"
                if np.any(observed_shoot)
                else "no_shoot_evidence",
            }
            can_carry_spatially = not (
                crown_jump_rejected and not visual_shift_valid and observed_area <= 0
            )
            if (
                bool(getattr(config, "shoot_temporal_crown_memory_enabled", True))
                and bool(getattr(config, "shoot_temporal_monotonic_area_enabled", True))
                and previous_shoot is not None
                and np.any(previous_shoot)
                and can_carry_spatially
            ):
                tracked_shoot, temporal_meta = _fill_shoot_area_from_temporal_memory(
                    observed_shoot,
                    previous_shoot,
                    shift_x=shift_x,
                    shift_y=shift_y,
                    allowed_mask=lane_allowed,
                )
                if bool(visual_meta.get("applied", False)):
                    if int(temporal_meta.get("carried_pixels", 0) or 0) > 0:
                        temporal_meta["status"] = "visual_reacquisition_plus_area_floor"
                    else:
                        temporal_meta["status"] = "visual_reacquisition_observed"
            elif crown_jump_rejected and previous_area > observed_area:
                temporal_meta["status"] = "spatial_carry_suspended_unreliable_motion"

            tracked_shoot = ((tracked_shoot > 0) & (lane_allowed > 0)).astype(np.uint8)
            tracked_shoot[accepted_shoot > 0] = 0
            _update_row_from_temporal_shoot_mask(
                row,
                tracked_shoot,
                temporal_meta,
                config,
                shift_x=shift_x,
                shift_y=shift_y,
                motion_source=motion_source,
                visual_reacquisition=visual_meta,
            )
            frame_track_area_sum += int(np.count_nonzero(tracked_shoot))
            accepted_shoot = np.maximum(accepted_shoot, tracked_shoot)
            if np.any(tracked_shoot):
                shoot_memory_by_track[track_id] = tracked_shoot.copy()
            crown_memory_by_track[track_id] = (crown_x, crown_y)

        accepted_area = int(np.count_nonzero(accepted_shoot))
        if accepted_area != frame_track_area_sum:
            raise RuntimeError(
                f"frame {frame_index}: accepted shoot ownership is not exclusive "
                f"({frame_track_area_sum} track pixels versus {accepted_area} union pixels)"
            )

        display_mask = mask_u8.copy()
        display_mask[display_mask == shoot_class_id] = 0
        display_mask[(accepted_shoot > 0) & (root_union == 0)] = np.uint8(shoot_class_id)
        source_stem = Path(str(frames[frame_index].get("SourceFile", frame_index))).stem
        output_path = output_root / f"{frame_index:04d}_{source_stem}_ownership_display_mask.png"
        Image.fromarray(display_mask, mode="L").save(output_path)
        output_paths[frame_index] = str(output_path.resolve())
    return output_paths


def _process_plate(task: dict[str, object]) -> dict[str, object]:
    cv2.setNumThreads(1)
    started = time.monotonic()
    series = str(task["series"])
    petri = str(task["petri"])
    checkpoint_path = Path(str(task["checkpoint_path"]))
    fingerprint = str(task["fingerprint"])
    summary_path = checkpoint_path.with_suffix(".summary.json")
    if checkpoint_path.exists() and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if str(summary.get("fingerprint", "")) == fingerprint:
                return {**summary, "status": "resumed"}
        except Exception:
            pass

    config = _analytics_config(dict(task["config"]))
    frames = [frame for frame in task["frames"] if isinstance(frame, dict)]
    items: list[DatasetImageItem] = []
    predictions: dict[str, np.ndarray] = {}
    masks: list[np.ndarray] = []
    for frame_position, frame in enumerate(frames):
        source_path = Path(str(frame["SourceFile"]))
        mask_path = Path(str(frame["OutputMaskPath"]))
        mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.ndim != 2:
            raise RuntimeError(f"Unreadable class-index mask: {mask_path}")
        uid = f"{series}:{petri}:{frame_position}"
        image = _load_rgb(source_path, mask.shape[:2])
        items.append(DatasetImageItem(uid=uid, name=source_path.name, path=source_path, image=image))
        predictions[uid] = mask
        masks.append(mask)

    payload = run_temporal_analytics(items, predictions, {}, config)
    payload_rows = payload.get("rows", [])
    if not isinstance(payload_rows, list):
        raise RuntimeError("Analytics payload did not contain rows")
    analytics_summary = payload.get("summary", {})
    analytics_summary = analytics_summary if isinstance(analytics_summary, dict) else {}
    identity = analytics_summary.get("identity_initialization", {})
    identity = identity if isinstance(identity, dict) else {}
    ownership = analytics_summary.get("tip_guided_ownership", {})
    ownership = ownership if isinstance(ownership, dict) else {}
    tip_refinement = analytics_summary.get("tip_ownership_refinement", {})
    tip_refinement = tip_refinement if isinstance(tip_refinement, dict) else {}
    shoot_tracking = analytics_summary.get("shoot_tracking", {})
    shoot_tracking = shoot_tracking if isinstance(shoot_tracking, dict) else {}
    safe_series = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in series
    ).strip("_") or "series"
    display_mask_paths = _write_accepted_shoot_display_masks(
        [row for row in payload_rows if isinstance(row, dict)],
        frames,
        items,
        masks,
        config,
        checkpoint_path.parent.parent
        / "ownership_display_masks"
        / safe_series,
    )

    output_rows: list[dict[str, object]] = []
    for row in payload_rows:
        if not isinstance(row, dict):
            continue
        frame_position = int(row.get("frame_index", -1))
        if frame_position < 0 or frame_position >= len(frames):
            continue
        frame = frames[frame_position]
        out = dict(row)
        out["Series"] = series
        out["PetriDish"] = petri
        out["Timestamp"] = frame.get("Timestamp", "")
        out["FrameIndex"] = frame.get("FrameIndex", frame_position)
        out["RelativeFolder"] = frame.get("RelativeFolder", ".")
        out["SourceFile"] = frame.get("SourceFile", "")
        out["OutputMaskPath"] = frame.get("OutputMaskPath", "")
        out["OwnershipDisplayMaskPath"] = display_mask_paths.get(
            frame_position,
            "",
        )
        out["PreviewImagePath"] = frame.get("PreviewImagePath", frame.get("SourceFile", ""))
        out["ownership_root_class_id"] = int(config.root_class_id)
        out["ownership_lateral_class_id"] = int(config.lateral_class_id or 0)
        out["ownership_shoot_class_id"] = int(config.shoot_class_id or 0)
        out["ownership_shoot_rgb_green_only_enabled"] = bool(config.shoot_rgb_green_only_enabled)
        out["ownership_tracking_mode"] = str(analytics_summary.get("tracking_mode", config.tracking_mode))
        out["ownership_identity_initialization_status"] = str(identity.get("status", ""))
        out["ownership_identity_centers_x"] = json.dumps(identity.get("centers_x", []), separators=(",", ":"))
        out["ownership_measurement_scope"] = str(
            (analytics_summary.get("mask_source") or {}).get("measurement_scope", "")
            if isinstance(analytics_summary.get("mask_source"), dict)
            else ""
        )
        out["ownership_detached_recovery_pixels"] = int(ownership.get("root_detached_recovery_pixels", 0) or 0)
        out["ownership_detached_recovery_components"] = int(
            ownership.get("root_detached_recovery_components", 0) or 0
        )
        out["ownership_tip_refinement_status"] = str(tip_refinement.get("status", ""))
        out["ownership_learned_owner_enabled"] = bool(
            ownership.get("learned_owner_enabled", False)
        )
        out["ownership_learned_owner_status"] = str(
            ownership.get("learned_owner_status", "")
        )
        output_rows.append(out)

    conservation = _validate_plate_conservation(output_rows, masks, config)
    alignment_statuses = identity.get("per_frame_alignment_status", [])
    alignment_statuses = alignment_statuses if isinstance(alignment_statuses, list) else []
    alignment_accepteds = identity.get("per_frame_alignment_accepted", [])
    alignment_accepteds = alignment_accepteds if isinstance(alignment_accepteds, list) else []
    motion_sources = identity.get("per_frame_motion_source", [])
    motion_sources = motion_sources if isinstance(motion_sources, list) else []
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".csv.tmp")
    pd.DataFrame(output_rows).to_csv(temporary, index=False)
    os.replace(temporary, checkpoint_path)
    temporal_rows = [
        row for row in output_rows if bool(row.get("shoot_temporal_tracking_applied", False))
    ]
    visual_temporal_rows = [
        row
        for row in output_rows
        if bool(row.get("shoot_temporal_visual_reacquisition_applied", False))
    ]
    temporal_tracks = sorted(
        {
            str(row.get("plant_id", ""))
            for row in temporal_rows
            if str(row.get("plant_id", ""))
        }
    )
    temporal_drop_violations = 0
    output_frame = pd.DataFrame(output_rows)
    if not output_frame.empty and {"plant_id", "frame_index", "shoot_area_px"}.issubset(
        output_frame.columns
    ):
        for _track_id, track_rows in output_frame.groupby("plant_id", sort=False):
            areas = pd.to_numeric(
                track_rows.sort_values("frame_index", kind="mergesort")["shoot_area_px"],
                errors="coerce",
            ).fillna(0.0)
            temporal_drop_violations += int((areas.diff().fillna(0.0) < 0.0).sum())
    result = {
        "status": "complete",
        "series": series,
        "petri": petri,
        "fingerprint": fingerprint,
        "checkpoint_path": str(checkpoint_path),
        "frames": int(len(frames)),
        "rows": int(len(output_rows)),
        "valid_rows": int(sum(bool(row.get("ownership_measurement_valid", False)) for row in output_rows)),
        "elapsed_seconds": float(round(time.monotonic() - started, 3)),
        "conservation": conservation,
        "identity_status": str(identity.get("status", "")),
        "identity_centers_x": identity.get("centers_x", []),
        "alignment_accepted_frames": int(sum(bool(value) for value in alignment_accepteds)),
        "alignment_total_frames": int(len(alignment_statuses)),
        "root_crown_fallback_frames": int(
            sum(str(value) == "root_crown_fallback" for value in motion_sources)
        ),
        "measurement_conservation_recovery_frames": int(
            ownership.get("root_measurement_conservation_recovery_frames", 0) or 0
        ),
        "measurement_conservation_recovery_pixels": int(
            ownership.get("root_measurement_conservation_recovery_pixels", 0) or 0
        ),
        "accepted_shoot_display_masks": int(len(display_mask_paths)),
        "shoot_crown_gate_rejected_components": int(
            shoot_tracking.get("crown_component_gate_rejected_components", 0)
            or 0
        ),
        "shoot_crown_gate_rejected_pixels": int(
            shoot_tracking.get("crown_component_gate_rejected_pixels", 0) or 0
        ),
        "shoot_grayscale_crown_rescue_track_frames": int(
            shoot_tracking.get("grayscale_crown_rescue_frames", 0) or 0
        ),
        "shoot_grayscale_crown_rescue_pixels": int(
            shoot_tracking.get("grayscale_crown_rescue_pixels", 0) or 0
        ),
        "shoot_grayscale_crown_rescue_tracks": list(
            shoot_tracking.get("grayscale_crown_rescue_tracks", [])
        ),
        "shoot_temporal_tracking_frames": int(len(temporal_rows)),
        "shoot_temporal_carried_pixels": int(
            sum(int(row.get("shoot_temporal_carried_pixels", 0) or 0) for row in temporal_rows)
        ),
        "shoot_temporal_visual_reacquisition_track_frames": int(
            len(visual_temporal_rows)
        ),
        "shoot_temporal_visual_reacquisition_pixels": int(
            sum(
                int(row.get("shoot_temporal_visual_reacquisition_pixels", 0) or 0)
                for row in visual_temporal_rows
            )
        ),
        "shoot_temporal_tracks": temporal_tracks,
        "shoot_temporal_area_drop_violations": int(temporal_drop_violations),
        "learned_owner_status": str(ownership.get("learned_owner_status", "")),
        "learned_owner_frames": int(ownership.get("learned_owner_frames", 0) or 0),
        "learned_owner_support_pixels": int(
            ownership.get("learned_owner_support_pixels", 0) or 0
        ),
        "learned_owner_graph_agreement_pixels": int(
            ownership.get("learned_owner_graph_agreement_pixels", 0) or 0
        ),
        "learned_owner_low_confidence_pixels": int(
            ownership.get("learned_owner_low_confidence_pixels", 0) or 0
        ),
    }
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary_summary, summary_path)
    return result


def _load_frame_manifest(path: Path, selected_plates: set[str] | None) -> list[dict[str, object]]:
    columns = set(pd.read_csv(path, nrows=0).columns)
    usecols = [column for column in MANIFEST_COLUMNS if column in columns]
    required = {"PetriDish", "SourceFile", "OutputMaskPath"}
    missing = required.difference(usecols)
    if missing:
        raise RuntimeError(f"Manifest is missing required columns: {sorted(missing)}")
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    if "Series" not in df.columns:
        df["Series"] = df["PetriDish"]
    if "Timestamp" not in df.columns:
        df["Timestamp"] = ""
    if "FrameIndex" not in df.columns:
        df["FrameIndex"] = df.groupby(["Series", "PetriDish"]).cumcount()
    if "RelativeFolder" not in df.columns:
        df["RelativeFolder"] = "."
    if "PreviewImagePath" not in df.columns:
        df["PreviewImagePath"] = df["SourceFile"]
    df["Series"] = df["Series"].fillna(df["PetriDish"]).astype(str)
    df["PetriDish"] = df["PetriDish"].astype(str)
    if selected_plates:
        df = df[df["PetriDish"].isin(selected_plates)].copy()
    dedupe_columns = [
        column
        for column in ("Series", "PetriDish", "Timestamp", "FrameIndex", "SourceFile", "OutputMaskPath")
        if column in df.columns
    ]
    df.drop_duplicates(subset=dedupe_columns, keep="first", inplace=True)
    df["_timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["_frame"] = pd.to_numeric(df["FrameIndex"], errors="coerce")
    df.sort_values(["Series", "PetriDish", "_timestamp", "_frame", "SourceFile"], inplace=True, kind="mergesort")
    df.drop(columns=["_timestamp", "_frame"], inplace=True)
    return df.to_dict(orient="records")


def _apply_baseline_validity_safeguard(
    master: pd.DataFrame,
    baseline_path: Path | None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if baseline_path is None:
        result = master.copy()
        result["ownership_plate_result_source"] = "learned_v11"
        return result, []

    baseline = pd.read_csv(baseline_path, low_memory=False)
    keys = ["PetriDish", "FrameIndex", "plant_id"]
    missing_current = [column for column in keys if column not in master.columns]
    missing_baseline = [column for column in keys if column not in baseline.columns]
    if missing_current or missing_baseline:
        raise RuntimeError(
            "Baseline validity safeguard requires PetriDish, FrameIndex, and "
            f"plant_id keys (current missing={missing_current}, "
            f"baseline missing={missing_baseline})."
        )

    current = master.copy()
    selected_plates = set(current["PetriDish"].astype(str))
    baseline = baseline[baseline["PetriDish"].astype(str).isin(selected_plates)].copy()
    current_valid_column = (
        "ownership_measurement_valid"
        if "ownership_measurement_valid" in current.columns
        else "ownership_valid"
    )
    baseline_valid_column = (
        "ownership_measurement_valid"
        if "ownership_measurement_valid" in baseline.columns
        else "ownership_valid"
    )
    if current_valid_column not in current.columns or baseline_valid_column not in baseline.columns:
        raise RuntimeError("Baseline validity safeguard could not find an ownership-valid column.")

    current_keyed = current[keys + [current_valid_column]].copy()
    baseline_keyed = baseline[keys + [baseline_valid_column]].copy()
    current_keyed["_current_valid"] = (
        current_keyed[current_valid_column].fillna(False).astype(bool)
    )
    baseline_keyed["_baseline_valid"] = (
        baseline_keyed[baseline_valid_column].fillna(False).astype(bool)
    )
    comparison = baseline_keyed[keys + ["_baseline_valid"]].merge(
        current_keyed[keys + ["_current_valid"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    comparison["_regression"] = (
        comparison["_baseline_valid"] & ~comparison["_current_valid"]
    )
    rollback_plates = sorted(
        comparison.loc[comparison["_regression"], "PetriDish"].astype(str).unique()
    )
    audit: list[dict[str, object]] = []
    if not rollback_plates:
        current["ownership_plate_result_source"] = "learned_v11"
        return current, audit

    current["ownership_plate_result_source"] = "learned_v11"
    output_parts: list[pd.DataFrame] = [
        current[~current["PetriDish"].astype(str).isin(rollback_plates)].copy()
    ]
    all_columns = list(dict.fromkeys([*current.columns, *baseline.columns]))
    for plate in rollback_plates:
        current_plate = current[current["PetriDish"].astype(str) == plate]
        baseline_plate = baseline[baseline["PetriDish"].astype(str) == plate].copy()
        current_keys = set(
            map(tuple, current_plate[keys].itertuples(index=False, name=None))
        )
        baseline_keys = set(
            map(tuple, baseline_plate[keys].itertuples(index=False, name=None))
        )
        if current_keys != baseline_keys:
            raise RuntimeError(
                f"Cannot safely roll back {plate}: baseline/current row keys differ."
            )
        baseline_plate = baseline_plate.reindex(columns=all_columns)
        baseline_plate["ownership_plate_result_source"] = (
            "baseline_v8_non_regression_rollback"
        )
        for status_column in (
            "learned_owner_status",
            "ownership_learned_owner_status",
        ):
            if status_column in baseline_plate.columns:
                baseline_plate[status_column] = "plate_rollback_to_v8"
        output_parts.append(baseline_plate)
        plate_comparison = comparison[
            comparison["PetriDish"].astype(str) == plate
        ]
        audit.append(
            {
                "plate": plate,
                "regressed_valid_rows": int(
                    plate_comparison["_regression"].sum()
                ),
                "baseline_valid_rows": int(
                    plate_comparison["_baseline_valid"].sum()
                ),
                "learned_valid_rows": int(
                    plate_comparison["_current_valid"].sum()
                ),
                "action": "entire_plate_rolled_back_to_v8",
            }
        )

    output = pd.concat(output_parts, ignore_index=True, sort=False)
    output["Timestamp"] = pd.to_datetime(output["Timestamp"], errors="coerce")
    output["FrameIndex"] = pd.to_numeric(output["FrameIndex"], errors="coerce")
    output.sort_values(
        ["Series", "PetriDish", "Timestamp", "FrameIndex", "plant_id"],
        inplace=True,
        kind="mergesort",
    )
    output.reset_index(drop=True, inplace=True)
    return output, audit


def _build_timelapse(master: pd.DataFrame, expected_plants: int) -> pd.DataFrame:
    df = master.copy()
    valid = df.get("ownership_measurement_valid", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    df["_valid"] = valid
    df["_valid_plant"] = df["plant_id"].where(valid, pd.NA)
    metric_sources = {
        "total_root_length_mm": "total_root_length_mm_clean",
        "total_root_length_weighted_mm": "total_root_length_weighted_mm_clean",
        "total_primary_root_length_mm": "primary_root_length_mm_clean",
        "total_primary_root_length_weighted_mm": "primary_root_length_weighted_mm_clean",
        "total_lateral_root_length_mm": "lateral_total_length_mm",
        "total_lateral_root_length_weighted_mm": "lateral_total_length_weighted_mm",
        "total_shoot_area_mm2": "shoot_area_mm2",
        "total_shoot_area_px": "shoot_area_px",
    }
    plate_mask_metric_sources = {
        "plate_mask_total_root_length_mm": "total_root_length_mm_raw",
        "plate_mask_total_root_length_weighted_mm": "total_root_length_weighted_mm_raw",
        "plate_mask_primary_root_length_mm": "primary_root_length_mm_raw",
        "plate_mask_primary_root_length_weighted_mm": "primary_root_length_weighted_mm_raw",
        "plate_mask_lateral_root_length_mm": "lateral_total_length_mm",
        "plate_mask_lateral_root_length_weighted_mm": "lateral_total_length_weighted_mm",
        "plate_mask_total_root_area_mm2": "total_root_area_mm2",
        "plate_mask_primary_root_area_mm2": "primary_root_area_mm2",
    }
    for temporary, source in metric_sources.items():
        df[f"_{temporary}"] = pd.to_numeric(df.get(source), errors="coerce").where(valid)
    for output_column, source in plate_mask_metric_sources.items():
        df[f"_{output_column}"] = pd.to_numeric(df.get(source), errors="coerce")
    keys = ["Series", "PetriDish", "Timestamp", "FrameIndex"]
    grouped = df.groupby(keys, dropna=False, sort=False)
    rows: list[dict[str, object]] = []
    for key, group in grouped:
        row = dict(zip(keys, key, strict=True))
        row["ownership_observed_owners"] = int(group["plant_id"].nunique())
        row["ownership_valid_owners"] = int(group["_valid_plant"].nunique())
        row["ownership_expected_owners"] = int(expected_plants)
        row["ownership_complete"] = bool(row["ownership_valid_owners"] == int(expected_plants))
        row["plants_detected"] = int(row["ownership_valid_owners"])
        row["plate_mask_root_totals_available"] = True
        row["plate_mask_root_totals_ownership_independent"] = True
        for output_column in plate_mask_metric_sources:
            values = pd.to_numeric(group[f"_{output_column}"], errors="coerce").dropna()
            row[output_column] = float(values.sum()) if not values.empty else np.nan
        for output_column in metric_sources:
            values = pd.to_numeric(group[f"_{output_column}"], errors="coerce").dropna()
            row[output_column] = float(values.sum()) if not values.empty and row["ownership_complete"] else np.nan
            row[output_column.replace("total_", "mean_", 1)] = (
                float(values.mean()) if not values.empty and row["ownership_complete"] else np.nan
            )
        row["root_length_metric_source"] = "total_root_length_mm_clean"
        row["root_length_weighted_metric_source"] = "total_root_length_weighted_mm_clean"
        row["selected_root_metric_mode"] = "total"
        row["selected_root_metric_column"] = "total_root_length_mm"
        row["selected_root_length_mm"] = row["total_root_length_mm"]
        row["primary_plus_lateral_root_length_mm"] = (
            row["total_primary_root_length_mm"] + row["total_lateral_root_length_mm"]
            if row["ownership_complete"]
            else np.nan
        )
        denom = row["primary_plus_lateral_root_length_mm"]
        row["primary_fraction_of_root_length"] = (
            row["total_primary_root_length_mm"] / denom if np.isfinite(denom) and denom > 0.0 else np.nan
        )
        row["lateral_fraction_of_root_length"] = (
            row["total_lateral_root_length_mm"] / denom if np.isfinite(denom) and denom > 0.0 else np.nan
        )
        shoot_area = row["total_shoot_area_mm2"]
        row["root_length_mm_per_shoot_area_mm2"] = (
            row["selected_root_length_mm"] / shoot_area
            if np.isfinite(row["selected_root_length_mm"]) and np.isfinite(shoot_area) and shoot_area > 0.0
            else np.nan
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["_timestamp"] = pd.to_datetime(summary["Timestamp"], errors="coerce")
    summary["_frame"] = pd.to_numeric(summary["FrameIndex"], errors="coerce")
    summary.sort_values(["Series", "PetriDish", "_timestamp", "_frame"], inplace=True, kind="mergesort")
    summary.drop(columns=["_timestamp", "_frame"], inplace=True)
    for value_column in (
        "total_root_length_mm",
        "selected_root_length_mm",
        "total_root_length_weighted_mm",
        "total_primary_root_length_mm",
        "total_primary_root_length_weighted_mm",
        "total_lateral_root_length_mm",
        "total_lateral_root_length_weighted_mm",
        "primary_plus_lateral_root_length_mm",
        "total_shoot_area_mm2",
        "total_shoot_area_px",
        "plants_detected",
        *plate_mask_metric_sources,
    ):
        summary[f"delta_{value_column}"] = summary.groupby(["Series", "PetriDish"])[value_column].diff()
    return summary.reset_index(drop=True)


def _promote_endpoint_raw_metrics(master: pd.DataFrame) -> pd.DataFrame:
    """Do not apply temporal cleaning to groups that contain only one frame."""
    if master.empty:
        return master
    result = master.copy()
    group_columns = [column for column in ("Series", "PetriDish") if column in result.columns]
    if not group_columns:
        return result
    if "FrameIndex" in result.columns:
        frame_counts = result.groupby(group_columns, dropna=False)["FrameIndex"].transform("nunique")
    else:
        frame_counts = result.groupby(group_columns, dropna=False)["SourceFile"].transform("nunique")
    endpoint = frame_counts <= 1
    pixel_size = pd.to_numeric(result.get("pixel_size_mm"), errors="coerce").fillna(0.0)
    metric_pairs = (
        ("root_length_mm_raw", "root_length_mm_clean"),
        ("primary_root_length_mm_raw", "primary_root_length_mm_clean"),
        ("total_root_length_mm_raw", "total_root_length_mm_clean"),
        ("primary_root_length_weighted_mm_raw", "primary_root_length_weighted_mm_clean"),
        ("total_root_length_weighted_mm_raw", "total_root_length_weighted_mm_clean"),
    )
    for source, target in metric_pairs:
        if source not in result.columns:
            continue
        result.loc[endpoint, target] = pd.to_numeric(
            result.loc[endpoint, source],
            errors="coerce",
        )
        px_target = target.replace("_mm_", "_px_")
        if px_target in result.columns:
            values = pd.to_numeric(result.loc[endpoint, source], errors="coerce")
            scale = pixel_size.loc[endpoint].replace(0.0, np.nan)
            result.loc[endpoint, px_target] = values / scale
    result["ownership_endpoint_raw_metrics_promoted"] = endpoint.astype(bool)
    return result


def _apply_presence_validity_gate(
    master: pd.DataFrame,
    *,
    enabled: bool,
    min_root_area_px: int,
    min_shoot_area_px: int,
) -> pd.DataFrame:
    """Mark empty expected lanes invalid without discarding their diagnostic rows."""
    if master.empty:
        return master
    result = master.copy()
    current_valid = result.get(
        "ownership_measurement_valid",
        pd.Series(False, index=result.index),
    ).fillna(False).astype(bool)
    root_area_column = next(
        (
            column
            for column in ("total_root_area_px", "root_area_px", "primary_root_area_px")
            if column in result.columns
        ),
        None,
    )
    root_area = (
        pd.to_numeric(result[root_area_column], errors="coerce").fillna(0.0)
        if root_area_column is not None
        else pd.Series(0.0, index=result.index)
    )
    root_length_column = next(
        (
            column
            for column in ("total_root_length_mm_raw", "root_length_mm_raw")
            if column in result.columns
        ),
        None,
    )
    root_length = (
        pd.to_numeric(result[root_length_column], errors="coerce").fillna(0.0)
        if root_length_column is not None
        else pd.Series(0.0, index=result.index)
    )
    shoot_area_column = next(
        (
            column
            for column in ("shoot_area_px", "n_pixels_shoot")
            if column in result.columns
        ),
        None,
    )
    shoot_area = (
        pd.to_numeric(result[shoot_area_column], errors="coerce").fillna(0.0)
        if shoot_area_column is not None
        else pd.Series(0.0, index=result.index)
    )
    root_present = (root_area >= int(max(1, min_root_area_px))) | (root_length > 0.0)
    shoot_present = shoot_area >= int(max(1, min_shoot_area_px))
    presence = root_present | shoot_present
    track_root_present = root_present.copy()
    persistent_shoot_only = pd.Series(False, index=result.index, dtype=bool)
    track_columns = [
        column
        for column in ("Series", "PetriDish", "plant_id")
        if column in result.columns
    ]
    if "plant_id" in track_columns:
        track_keys = [result[column] for column in track_columns]
        grouped_root = root_present.groupby(track_keys, dropna=False)
        track_frames = grouped_root.transform("size")
        root_frames = grouped_root.transform("sum")
        max_root_area = root_area.groupby(track_keys, dropna=False).transform("max")
        max_root_length = root_length.groupby(track_keys, dropna=False).transform("max")
        minimum_root_frames = np.maximum(2.0, np.ceil(track_frames.astype(float) * 0.15))
        track_root_present = (
            (root_frames >= minimum_root_frames)
            | (max_root_area >= float(max(150, int(min_root_area_px) * 5)))
            | (max_root_length >= 5.0)
        )
        persistent_shoot_only = (track_frames > 1) & ~track_root_present
        presence = presence & ~persistent_shoot_only
    result["ownership_presence_gate_enabled"] = bool(enabled)
    result["ownership_root_presence_evidence"] = root_present.astype(bool)
    result["ownership_shoot_presence_evidence"] = shoot_present.astype(bool)
    result["ownership_seedling_presence_evidence"] = presence.astype(bool)
    result["ownership_track_root_presence_evidence"] = track_root_present.astype(bool)
    result["ownership_presence_status"] = np.select(
        [persistent_shoot_only, presence],
        ["no_root_evidence_across_timeseries", "seedling_evidence_present"],
        default="no_seedling_evidence",
    )
    if bool(enabled):
        newly_invalid = current_valid & ~presence
        result["ownership_measurement_valid"] = current_valid & presence
        if "measurement_tier" in result.columns:
            result.loc[newly_invalid, "measurement_tier"] = "no_seedling_detected"
            result.loc[persistent_shoot_only, "measurement_tier"] = "no_root_track_detected"
    return result


def _write_plate_mask_timelapse(timelapse: pd.DataFrame, output_path: Path) -> None:
    identity_columns = ["Series", "PetriDish", "Timestamp", "FrameIndex"]
    status_columns = [
        "ownership_observed_owners",
        "ownership_valid_owners",
        "ownership_expected_owners",
        "ownership_complete",
        "plate_mask_root_totals_available",
        "plate_mask_root_totals_ownership_independent",
    ]
    metric_columns = [
        column
        for column in timelapse.columns
        if column.startswith("plate_mask_") or column.startswith("delta_plate_mask_")
    ]
    columns = list(dict.fromkeys(identity_columns + status_columns + metric_columns))
    timelapse.loc[:, [column for column in columns if column in timelapse.columns]].to_csv(
        output_path,
        index=False,
    )


def _write_run_report(
    output_dir: Path,
    args: argparse.Namespace,
    results: list[dict[str, object]],
    master: pd.DataFrame,
    timelapse: pd.DataFrame,
) -> None:
    valid = master.get("ownership_measurement_valid", pd.Series(False, index=master.index)).fillna(False).astype(bool)
    temporal_applied = master.get(
        "shoot_temporal_tracking_applied",
        pd.Series(False, index=master.index),
    ).fillna(False).astype(bool)
    visual_applied = master.get(
        "shoot_temporal_visual_reacquisition_applied",
        pd.Series(False, index=master.index),
    ).fillna(False).astype(bool)
    temporal_carried = pd.to_numeric(
        master.get("shoot_temporal_carried_pixels", pd.Series(0, index=master.index)),
        errors="coerce",
    ).fillna(0)
    visual_pixels = pd.to_numeric(
        master.get(
            "shoot_temporal_visual_reacquisition_pixels",
            pd.Series(0, index=master.index),
        ),
        errors="coerce",
    ).fillna(0)
    temporal_tracks = (
        master.loc[temporal_applied, ["PetriDish", "plant_id"]].drop_duplicates()
        if {"PetriDish", "plant_id"}.issubset(master.columns)
        else pd.DataFrame()
    )
    temporal_drop_violations = 0
    if {"PetriDish", "plant_id", "FrameIndex", "shoot_area_px"}.issubset(master.columns):
        for _key, rows in master.groupby(["PetriDish", "plant_id"], sort=False):
            areas = pd.to_numeric(
                rows.sort_values("FrameIndex", kind="mergesort")["shoot_area_px"],
                errors="coerce",
            ).fillna(0.0)
            temporal_drop_violations += int((areas.diff().fillna(0.0) < 0.0).sum())
    display_paths = master.get(
        "OwnershipDisplayMaskPath",
        pd.Series(dtype=object),
    ).fillna("").astype(str).str.strip()
    report = {
        "npec_app_version": NPEC_APP_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "source_manifest": str(args.manifest),
        "output_dir": str(output_dir),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "plates": int(master["PetriDish"].nunique()),
        "frames": int(master[["PetriDish", "FrameIndex"]].drop_duplicates().shape[0]),
        "rows": int(len(master)),
        "valid_rows": int(valid.sum()),
        "invalid_rows": int((~valid).sum()),
        "ownership_complete_frames": int(timelapse["ownership_complete"].fillna(False).astype(bool).sum()),
        "workers": int(args.workers),
        "pixel_size_mm": float(args.pixel_size_mm),
        "expected_plants": int(args.expected_plants),
        "root_classes": [int(args.root_class_id), int(args.lateral_class_id)],
        "shoot_class": int(args.shoot_class_id),
        "shoot_rgb_green_only_enabled": bool(args.shoot_rgb_green_only_enabled),
        "presence_gate_enabled": bool(args.presence_gate_enabled),
        "presence_min_root_area_px": int(args.presence_min_root_area_px),
        "presence_min_shoot_area_px": int(args.presence_min_shoot_area_px),
        "measurement_scope": "current_frame_prediction",
        "shoot_measurement_scope": "same_seedling_visual_then_crown_temporal_area_floor",
        "identity_mode": "arabidopsis_crown_lanes",
        "accepted_shoot_display_masks": int(display_paths[display_paths.ne("")].nunique()),
        "shoot_crown_gate_rejected_components": int(
            sum(
                int(result.get("shoot_crown_gate_rejected_components", 0) or 0)
                for result in results
            )
        ),
        "shoot_crown_gate_rejected_pixels": int(
            sum(
                int(result.get("shoot_crown_gate_rejected_pixels", 0) or 0)
                for result in results
            )
        ),
        "shoot_grayscale_crown_rescue_track_frames": int(
            sum(
                int(result.get("shoot_grayscale_crown_rescue_track_frames", 0) or 0)
                for result in results
            )
        ),
        "shoot_grayscale_crown_rescue_pixels": int(
            sum(
                int(result.get("shoot_grayscale_crown_rescue_pixels", 0) or 0)
                for result in results
            )
        ),
        "shoot_temporal_tracking_frames": int(temporal_applied.sum()),
        "shoot_temporal_carried_pixels": int(temporal_carried.sum()),
        "shoot_temporal_track_count": int(len(temporal_tracks)),
        "shoot_temporal_visual_reacquisition_track_frames": int(visual_applied.sum()),
        "shoot_temporal_visual_reacquisition_pixels": int(
            visual_pixels.loc[visual_applied].sum()
        ),
        "shoot_temporal_area_drop_violations": int(temporal_drop_violations),
        "owner_ranker_profile": (
            str(args.owner_ranker_profile)
            if getattr(args, "owner_ranker_profile", None)
            else None
        ),
        "owner_ranker_profile_sha256": (
            hashlib.sha256(Path(args.owner_ranker_profile).read_bytes()).hexdigest()
            if getattr(args, "owner_ranker_profile", None)
            else None
        ),
        "baseline_master": (
            str(args.baseline_master)
            if getattr(args, "baseline_master", None)
            else None
        ),
        "baseline_validity_safeguard": list(
            getattr(args, "_baseline_validity_safeguard", [])
        ),
        "plate_results": results,
    }
    (output_dir / "npec_corrected_run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _render_video(master: pd.DataFrame, timelapse: pd.DataFrame, output_dir: Path) -> None:
    video_path = output_dir / "npec_root_growth_analysis_video.mp4"
    result = generate_lazy_root_growth_video(
        master,
        timelapse,
        video_path,
        summary_csv_path=output_dir / "npec_root_growth_video_summary.csv",
        sample_frame_path=output_dir / "npec_root_growth_video_sample_frame.png",
        config=LazyRootGrowthVideoConfig(
            fps=1,
            width=1920,
            height=1080,
            mask_only=False,
            mask_display_dilation_px=1,
            root_metric_mode="total",
            root_class_ids=(1, 3),
            shoot_class_ids=(2,),
            show_rgb_shoot_rescue=True,
            show_seedling_ownership_boxes=True,
        ),
        progress_callback=lambda done, total, label: print(
            f"VIDEO {done}/{total} {label}",
            flush=True,
        ) if done == 1 or done == total or done % 50 == 0 else None,
    )
    shutil.copy2(result.video_path, output_dir / "npec_root_and_shoot_masks_analysis_video.mp4")


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "_ownership_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    selected_plates = set(args.plates) if args.plates else None
    manifest_rows = _load_frame_manifest(Path(args.manifest), selected_plates)
    if not manifest_rows:
        raise RuntimeError("No matching source/mask frames were found")

    owner_profile: dict[str, object] | None = None
    if args.owner_ranker_profile:
        profile_path = Path(args.owner_ranker_profile).expanduser().resolve()
        args.owner_ranker_profile = profile_path
        owner_profile_obj = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(owner_profile_obj, dict) or not isinstance(
            owner_profile_obj.get("ranker"), dict
        ):
            raise RuntimeError(f"Invalid learned owner ranker profile: {profile_path}")
        owner_profile = owner_profile_obj
        shutil.copy2(profile_path, output_dir / "npec_owner_ranker_profile.json")

    config_values = {
        "root_class_id": int(args.root_class_id),
        "lateral_class_id": int(args.lateral_class_id),
        "shoot_class_id": int(args.shoot_class_id),
        "expected_plants": int(args.expected_plants),
        "pixel_size_mm": float(args.pixel_size_mm),
        "timestep_hours": float(args.timestep_hours),
        "shoot_rgb_green_only_enabled": bool(args.shoot_rgb_green_only_enabled),
        "shoot_temporal_crown_memory_enabled": True,
        "shoot_temporal_monotonic_area_enabled": True,
        "shoot_temporal_max_crown_shift_px": 240.0,
        "shoot_temporal_visual_reacquisition_enabled": True,
        "shoot_temporal_visual_search_half_width_px": 320,
        "shoot_temporal_visual_search_above_px": 360,
        "shoot_temporal_visual_search_below_px": 520,
        "shoot_temporal_visual_max_shift_px": 520.0,
        "shoot_temporal_visual_max_root_distance_px": 120.0,
        "shoot_temporal_visual_root_vertical_tolerance_px": 220,
        "min_component_area": 30,
        "prune_branch_px": 5,
        "learned_owner_enabled": bool(owner_profile),
        "learned_owner_ranker_payload": owner_profile,
        "learned_owner_feature_set": str(
            (owner_profile or {}).get(
                "feature_set",
                "crown_coordinates_orientation",
            )
        ),
        "learned_owner_support_margin": float(
            (owner_profile or {}).get("support_margin", 0.50)
        ),
        "learned_owner_prior_weight_px": float(
            (owner_profile or {}).get("prior_weight_px", 4.0)
        ),
        "learned_owner_temporal_score_bonus": float(
            (owner_profile or {}).get("temporal_score_bonus", 0.18)
        ),
        "learned_owner_qc_margin": float(
            (owner_profile or {}).get("qc_margin", 0.10)
        ),
    }
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for position, frame in enumerate(manifest_rows):
        series = _safe_text(frame.get("Series"), _safe_text(frame.get("PetriDish"), "Unknown"))
        petri = _safe_text(frame.get("PetriDish"), "Unknown")
        normalized = {column: frame.get(column, "") for column in MANIFEST_COLUMNS}
        normalized["Series"] = series
        normalized["PetriDish"] = petri
        normalized["FrameIndex"] = _safe_frame_index(frame.get("FrameIndex"), position)
        normalized["Timestamp"] = _safe_text(frame.get("Timestamp"))
        grouped.setdefault((series, petri), []).append(normalized)

    tasks: list[dict[str, object]] = []
    for series, petri in sorted(grouped, key=lambda key: (key[0], key[1])):
        frames = grouped[(series, petri)]
        frames.sort(key=lambda row: (pd.to_datetime(row.get("Timestamp"), errors="coerce"), int(row["FrameIndex"])))
        slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in f"{series}__{petri}")[:120]
        task = {
            "series": series,
            "petri": petri,
            "frames": frames,
            "config": config_values,
        }
        fingerprint = _plate_fingerprint(task)
        task["fingerprint"] = fingerprint
        task["checkpoint_path"] = str(checkpoint_dir / f"{slug}__{fingerprint[:16]}.csv")
        tasks.append(task)

    print(f"START plates={len(tasks)} frames={len(manifest_rows)} workers={args.workers}", flush=True)
    results: list[dict[str, object]] = []
    failures: list[str] = []
    if int(args.workers) <= 1:
        for completed, task in enumerate(tasks, start=1):
            try:
                result = _process_plate(task)
                results.append(result)
                print(
                    f"PLATE {completed}/{len(tasks)} {result['petri']} {result['status']} "
                    f"rows={result['rows']} valid={result['valid_rows']} seconds={result.get('elapsed_seconds', 0)}",
                    flush=True,
                )
            except Exception as exc:
                message = f"{task['petri']}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"ERROR {message}", file=sys.stderr, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            future_map = {executor.submit(_process_plate, task): task for task in tasks}
            for completed, future in enumerate(as_completed(future_map), start=1):
                task = future_map[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(
                        f"PLATE {completed}/{len(tasks)} {result['petri']} {result['status']} "
                        f"rows={result['rows']} valid={result['valid_rows']} "
                        f"seconds={result.get('elapsed_seconds', 0)}",
                        flush=True,
                    )
                except Exception as exc:
                    message = f"{task['petri']}: {type(exc).__name__}: {exc}"
                    failures.append(message)
                    print(f"ERROR {message}", file=sys.stderr, flush=True)
    if failures:
        (output_dir / "npec_failed_plates.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(failures)} plate(s) failed; checkpoints were retained")
    (output_dir / "npec_failed_plates.txt").unlink(missing_ok=True)

    checkpoint_paths = [Path(str(result["checkpoint_path"])) for result in results]
    frames = [pd.read_csv(path, low_memory=False) for path in checkpoint_paths]
    master = pd.concat(frames, ignore_index=True)
    master["Timestamp"] = pd.to_datetime(master["Timestamp"], errors="coerce")
    master["FrameIndex"] = pd.to_numeric(master["FrameIndex"], errors="coerce")
    master.sort_values(["Series", "PetriDish", "Timestamp", "FrameIndex", "plant_id"], inplace=True, kind="mergesort")
    master.reset_index(drop=True, inplace=True)
    master = _promote_endpoint_raw_metrics(master)
    master = _apply_presence_validity_gate(
        master,
        enabled=bool(args.presence_gate_enabled),
        min_root_area_px=int(args.presence_min_root_area_px),
        min_shoot_area_px=int(args.presence_min_shoot_area_px),
    )
    baseline_path = (
        Path(args.baseline_master).expanduser().resolve()
        if getattr(args, "baseline_master", None)
        else None
    )
    master, baseline_audit = _apply_baseline_validity_safeguard(
        master,
        baseline_path,
    )
    args._baseline_validity_safeguard = baseline_audit
    rollback_by_plate = {
        str(item["plate"]): item for item in baseline_audit
    }
    for result in results:
        plate = str(result.get("petri", ""))
        if plate in rollback_by_plate:
            result["final_result_source"] = "baseline_v8_non_regression_rollback"
            result["valid_rows_before_rollback"] = int(
                result.get("valid_rows", 0) or 0
            )
            result["valid_rows"] = int(
                rollback_by_plate[plate]["baseline_valid_rows"]
            )
        else:
            result["final_result_source"] = "learned_v11"
        plate_valid = master.loc[
            master["PetriDish"].astype(str).eq(plate),
            "ownership_measurement_valid",
        ].fillna(False).astype(bool)
        result["valid_rows"] = int(plate_valid.sum())
        result["invalid_rows"] = int((~plate_valid).sum())
    master_path = output_dir / "npec_root_measurements_master.csv"
    master = quality_gate_ownership_measurement_dataframe(master)
    master.to_csv(master_path, index=False)

    timelapse = _build_timelapse(master, int(args.expected_plants))
    timelapse_path = output_dir / "npec_total_root_length_timelapse.csv"
    timelapse.to_csv(timelapse_path, index=False)
    _write_plate_mask_timelapse(
        timelapse,
        output_dir / "npec_plate_mask_root_totals_timelapse.csv",
    )
    pmi_df, _pmi_csv, _pmi_xlsx = export_pmi_style_rows(
        master,
        output_dir,
        derive_mask_metrics=False,
    )
    write_lazy_ownership_all_metrics_workbook(
        master,
        output_dir,
        timelapse_df=timelapse,
        pmi_df=pmi_df,
        root_metric_mode="total",
    )
    _write_run_report(output_dir, args, results, master, timelapse)
    if not args.skip_video:
        _render_video(master, timelapse, output_dir)
    if not args.keep_checkpoints:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    print(f"DONE output={output_dir}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-run per-seedling ownership from existing NPEC class-index masks without model inference."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--expected-plants", type=int, default=5)
    parser.add_argument("--pixel-size-mm", type=float, default=111.88 / 4200.0)
    parser.add_argument("--timestep-hours", type=float, default=1.0)
    parser.add_argument("--root-class-id", type=int, default=1)
    parser.add_argument("--shoot-class-id", type=int, default=2)
    parser.add_argument("--lateral-class-id", type=int, default=3)
    parser.add_argument(
        "--shoot-rgb-green-only-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use strict RGB-green shoot measurement during ownership. Disable "
            "for grayscale Hades and dark-background MXLab images."
        ),
    )
    parser.add_argument(
        "--presence-gate-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mark expected lanes without root or substantial shoot evidence invalid.",
    )
    parser.add_argument("--presence-min-root-area-px", type=int, default=30)
    parser.add_argument("--presence-min-shoot-area-px", type=int, default=500)
    parser.add_argument(
        "--owner-ranker-profile",
        type=Path,
        default=None,
        help="Optional learned five-seedling owner ranker JSON profile.",
    )
    parser.add_argument(
        "--baseline-master",
        type=Path,
        default=None,
        help=(
            "Optional prior per-seedling master CSV. A plate is rolled back in "
            "full if the new run loses any previously valid owner row."
        ),
    )
    parser.add_argument("--plates", nargs="*", default=[])
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
