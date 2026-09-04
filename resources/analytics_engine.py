from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import csv
import heapq
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image

from .learned_owner_assignment import SharedOwnerRanker, predict_owner_map
from .models import DatasetImageItem
from .stabilization import StabilizationConfig, next_stabilization_reference, stabilize_against_reference
from .tracker_2dt import TwoDTTrackerConfig, build_arabidopsis_2dt_tracking_seed


BBox = tuple[int, int, int, int]
Point = tuple[int, int]
OWNERSHIP_ANALYTICS_SCHEMA_VERSION = 9


@dataclass(slots=True)
class AnalyticsConfig:
    root_class_id: int = 1
    lateral_class_id: int | None = None
    seed_class_id: int | None = None
    shoot_class_id: int | None = None
    external_tracking_seed: dict[str, object] | None = None
    external_tip_priors_by_frame: dict[int, dict[str, list[Point]]] | None = None
    shoot_rgb_green_only_enabled: bool = False
    tracking_mode: str = "auto"  # auto | manual_roi | arabidopsis_2dt | arabidopsis_crown_lanes
    temporal_smoothing_enabled: bool = True
    temporal_alpha: float = 0.60
    pixel_size_mm: float = 0.05
    timestep_hours: float = 1.0
    min_component_area: int = 30
    bbox_padding: int = 12
    tracking_search_margin: int = 24
    prune_branch_px: int = 5
    persistence_frames: int = 2
    enforce_monotonic_growth: bool = True
    emergence_distance_mm: float = 2.0
    qc_overlay_alpha: float = 0.35
    qc_overlay_color_rgb: tuple[int, int, int] = (46, 196, 182)
    qc_bbox_color_rgb: tuple[int, int, int] = (108, 220, 255)
    qc_conflict_bbox_color_rgb: tuple[int, int, int] = (255, 96, 96)
    qc_path_color_rgb: tuple[int, int, int] = (255, 198, 88)
    qc_text_color_rgb: tuple[int, int, int] = (255, 255, 255)
    qc_line_thickness: int = 2
    qc_path_thickness: int = 2
    qc_font_scale: float = 0.55
    qc_contrast_gain: float = 1.0
    # Crossing-aware root-tip tracking inspired by 2D+t graph reconnection.
    tip_tracking_enabled: bool = True
    tip_track_max_link_distance_px: float = 42.0
    tip_track_max_gap_frames: int = 2
    tip_track_direction_weight: float = 0.35
    tip_track_max_link_cost: float = 1.35
    tip_track_allow_cross_plant: bool = False
    tip_track_cross_plant_max_distance_scale: float = 0.35
    tip_track_cross_plant_bbox_penalty: float = 0.9
    tip_track_cross_plant_penalty: float = 1.2
    tip_track_min_frames: int = 1
    tip_track_min_dominant_vote_share: float = 0.65
    # Reuse the prior distal tip location so ownership does not jump between
    # neighboring plants when roots touch or cross.
    root_temporal_seed_enabled: bool = True
    root_temporal_seed_radius_px: int = 18
    root_previous_mask_assignment_weight_px: float = 2.0
    # Resolve shared root components on the skeleton graph. This preserves
    # crown-to-tip continuity when neighboring roots touch and keeps a legacy
    # image-plane fallback for old projects and quick comparisons.
    root_ownership_assignment_mode: str = "temporal_graph"  # temporal_graph | euclidean
    root_ownership_ambiguity_margin_px: float = 2.5
    root_ownership_freeze_ambiguous_state: bool = True
    # A close graph decision remains visible for QA, but only material
    # ambiguity blocks ownership and freezes the temporal state. Tiny shared
    # junctions are common in thin masks and should not discard a whole plant.
    root_ownership_blocking_ambiguity_min_pixels: int = 128
    root_ownership_blocking_ambiguity_min_fraction: float = 0.005
    # Stabilize root assignment near shoot/seed crown to suppress false growth
    # caused by shoot movement/fall-over at late timepoints.
    shoot_crown_lock_enabled: bool = True
    shoot_crown_lock_radius_px: int = 24
    shoot_crown_lock_ref_frames: int = 3
    shoot_crown_lock_start_frame: int = 6
    shoot_crown_lock_top_extra_px: int = 56
    # Track moving shoot/seed anchor mass after early establishment so fallen
    # shoots remain associated with the correct sibling and can be excluded from
    # root/lateral ownership when they overlap the root system.
    shoot_tracking_enabled: bool = True
    shoot_tracking_start_frame: int = 6
    shoot_tracking_motion_radius_px: int = 56
    shoot_tracking_exclusion_radius_px: int = 10
    # In dynamic Arabidopsis crown lanes, reject detached shoot components that
    # are too far from the root crown. This prevents fixed camera/detector
    # artifacts near an outer image edge from becoming a seedling's shoot.
    shoot_tracking_crown_gate_enabled: bool = True
    shoot_tracking_crown_component_max_distance_px: float = 180.0
    # The legacy Hades BW model sometimes misses an otherwise clear dark
    # rosette. Recover only an empty model result, inside the tracked crown
    # lane, and only when the root mask confirms that a plant is present.
    shoot_tracking_grayscale_crown_rescue_enabled: bool = True
    shoot_tracking_grayscale_crown_rescue_threshold_ratio: float = 0.56
    shoot_tracking_grayscale_crown_rescue_threshold_min: int = 58
    shoot_tracking_grayscale_crown_rescue_threshold_max: int = 88
    shoot_tracking_grayscale_crown_rescue_roi_half_width_px: int = 220
    shoot_tracking_grayscale_crown_rescue_roi_above_px: int = 220
    shoot_tracking_grayscale_crown_rescue_roi_below_px: int = 120
    shoot_tracking_grayscale_crown_rescue_root_radius_px: int = 80
    shoot_tracking_grayscale_crown_rescue_min_root_pixels: int = 20
    shoot_tracking_grayscale_crown_rescue_min_component_area: int = 80
    shoot_tracking_grayscale_crown_rescue_max_component_area: int = 28000
    shoot_tracking_grayscale_crown_rescue_max_distance_px: float = 160.0
    # Preserve accepted BW shoot evidence over time. A previous per-seedling
    # mask is translated with its tracked crown and can fill only a later area
    # deficit in the same lane, preventing model dropout from shrinking shoots.
    shoot_temporal_crown_memory_enabled: bool = True
    shoot_temporal_monotonic_area_enabled: bool = True
    shoot_temporal_max_crown_shift_px: float = 240.0
    # When a BW shoot shrinks or disappears, reacquire its dark rosette from the
    # current image inside the same seedling lane. This handles plant motion that
    # is not represented by the root-derived crown coordinate.
    shoot_temporal_visual_reacquisition_enabled: bool = True
    shoot_temporal_visual_search_half_width_px: int = 320
    shoot_temporal_visual_search_above_px: int = 360
    shoot_temporal_visual_search_below_px: int = 520
    shoot_temporal_visual_max_shift_px: float = 520.0
    shoot_temporal_visual_max_root_distance_px: float = 120.0
    shoot_temporal_visual_root_vertical_tolerance_px: int = 220
    # Compartmentalize each seedling into a stable left-right territory and
    # prefer only root pixels connected to that seedling's crown region.
    track_lane_partition_enabled: bool = True
    track_lane_padding_px: int = 0
    track_seed_connectivity_enabled: bool = True
    # When explicit lateral masks are available, keep them out of the primary
    # root path/tip inference so long laterals do not hijack the tracked tip.
    primary_root_tracking_excludes_lateral: bool = True
    primary_root_tracking_fallback_gap_px: int = 96
    # Optional cap to keep only likely true plants during auto-tracking.
    # Set from UI "Expected plants" when available (0 disables capping).
    expected_track_count: int = 0
    # Optional learned five-crown owner assignment. The ranker only reallocates
    # pixels already present in the semantic root mask; it never repairs or
    # invents roots, so plate-level area and length remain conserved.
    learned_owner_enabled: bool = False
    learned_owner_ranker_payload: dict[str, object] | None = None
    learned_owner_feature_set: str = "crown_coordinates_orientation"
    learned_owner_support_margin: float = 0.50
    learned_owner_prior_weight_px: float = 4.0
    learned_owner_temporal_score_bonus: float = 0.18
    learned_owner_qc_margin: float = 0.10
    learned_owner_chunk_size: int = 180_000


def _clip_bbox(bbox: BBox, shape_hw: tuple[int, int]) -> BBox:
    h, w = shape_hw
    x, y, bw, bh = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    if h <= 0 or w <= 0:
        return (0, 0, 0, 0)
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    bw = max(0, min(w - x, bw))
    bh = max(0, min(h - y, bh))
    return (x, y, bw, bh)


def _expand_bbox(bbox: BBox, margin: int, shape_hw: tuple[int, int]) -> BBox:
    x, y, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return (0, 0, 0, 0)
    x0 = x - int(margin)
    y0 = y - int(margin)
    x1 = x + bw + int(margin)
    y1 = y + bh + int(margin)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(shape_hw[1], x1)
    y1 = min(shape_hw[0], y1)
    return _clip_bbox((x0, y0, x1 - x0, y1 - y0), shape_hw)


def _union_bbox(a: BBox, b: BBox, shape_hw: tuple[int, int]) -> BBox:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0:
        return _clip_bbox(b, shape_hw)
    if bw <= 0 or bh <= 0:
        return _clip_bbox(a, shape_hw)
    x0 = min(int(ax), int(bx))
    y0 = min(int(ay), int(by))
    x1 = max(int(ax + aw), int(bx + bw))
    y1 = max(int(ay + ah), int(by + bh))
    return _clip_bbox((x0, y0, max(0, x1 - x0), max(0, y1 - y0)), shape_hw)


def _bbox_intersects(a: BBox, b: BBox) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return False
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    iw = max(0, x1 - x0)
    ih = max(0, y1 - y0)
    inter = float(iw * ih)
    union = float((aw * ah) + (bw * bh) - inter)
    if union <= 0:
        return 0.0
    return inter / union


def _bbox_center_distance(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx = float(ax + aw * 0.5)
    acy = float(ay + ah * 0.5)
    bcx = float(bx + bw * 0.5)
    bcy = float(by + bh * 0.5)
    return float(np.hypot(acx - bcx, acy - bcy))


def _bbox_center_or_none(bbox: BBox) -> tuple[float, float] | None:
    x, y, bw, bh = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    if bw <= 0 or bh <= 0:
        return None
    return (float(x + (0.5 * bw)), float(y + (0.5 * bh)))


def _union_valid_bboxes(a: BBox, b: BBox, shape_hw: tuple[int, int]) -> BBox:
    if a[2] <= 0 or a[3] <= 0:
        return _clip_bbox(b, shape_hw)
    if b[2] <= 0 or b[3] <= 0:
        return _clip_bbox(a, shape_hw)
    return _union_bbox(a, b, shape_hw)


def _center_jump_series(centers: list[tuple[float, float] | None]) -> list[float]:
    jumps: list[float] = []
    previous: tuple[float, float] | None = None
    for center in centers:
        if center is None:
            jumps.append(0.0)
            continue
        if previous is None:
            jumps.append(0.0)
        else:
            jumps.append(float(np.hypot(float(center[0] - previous[0]), float(center[1] - previous[1]))))
        previous = center
    return jumps


def _tracking_assignment_allowed(
    previous: BBox,
    candidate: BBox,
    shape_hw: tuple[int, int],
    config: AnalyticsConfig,
    *,
    anchor_x: float | None = None,
    anchor_overlap: float = 0.0,
) -> bool:
    if previous[2] <= 0 or previous[3] <= 0 or candidate[2] <= 0 or candidate[3] <= 0:
        return False
    h, w = shape_hw
    if h <= 0 or w <= 0:
        return False
    prev_center = _bbox_center_or_none(previous)
    cand_center = _bbox_center_or_none(candidate)
    if prev_center is None or cand_center is None:
        return False

    search_margin = float(max(1, int(config.tracking_search_margin)))
    prev_span = float(max(1, previous[2], previous[3]))
    cand_span = float(max(1, candidate[2], candidate[3]))
    center_limit = min(
        float(np.hypot(float(w), float(h)) * 0.28),
        max(search_margin * 3.0, prev_span * 2.8, cand_span * 1.35),
    )
    if _bbox_center_distance(previous, candidate) > center_limit and float(anchor_overlap) < 0.03:
        return False

    x_jump = abs(float(cand_center[0] - prev_center[0]))
    width_limit = min(
        max(96.0, float(w) * 0.20),
        max(search_margin * 3.0, float(max(previous[2], candidate[2], 32)) * 3.0, 96.0),
    )
    if x_jump > width_limit and float(anchor_overlap) < 0.03:
        return False

    if anchor_x is not None:
        anchor_jump = abs(float(cand_center[0]) - float(anchor_x))
        anchor_limit = min(
            max(128.0, float(w) * 0.25),
            max(search_margin * 4.0, float(max(previous[2], candidate[2], 32)) * 4.0, 128.0),
        )
        if anchor_jump > anchor_limit and float(anchor_overlap) < 0.01:
            return False
    return True


def _track_pair(a: str, b: str) -> tuple[str, str]:
    left = str(a).strip()
    right = str(b).strip()
    if left <= right:
        return (left, right)
    return (right, left)


def _connected_track_groups(track_ids: list[str], pairs: set[tuple[str, str]]) -> list[list[str]]:
    normalized_ids = sorted({str(track_id).strip() for track_id in track_ids if str(track_id).strip()})
    if not normalized_ids:
        return []
    adjacency: dict[str, set[str]] = {track_id: set() for track_id in normalized_ids}
    for left, right in pairs:
        if left not in adjacency or right not in adjacency or left == right:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
    groups: list[list[str]] = []
    seen: set[str] = set()
    for track_id in normalized_ids:
        if track_id in seen:
            continue
        queue = deque([track_id])
        component: list[str] = []
        seen.add(track_id)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        groups.append(sorted(component))
    groups.sort(key=lambda members: (0 if len(members) > 1 else 1, members))
    return groups


def _row_bbox(row: dict[str, object]) -> BBox:
    for prefix in ("seedling_bbox_", "bbox_"):
        try:
            x = int(row.get(f"{prefix}x", 0))
            y = int(row.get(f"{prefix}y", 0))
            w = int(row.get(f"{prefix}w", 0))
            h = int(row.get(f"{prefix}h", 0))
        except Exception:
            continue
        if w > 0 and h > 0:
            return (x, y, max(0, w), max(0, h))
    try:
        x = int(row.get("bbox_x", 0))
        y = int(row.get("bbox_y", 0))
        w = int(row.get("bbox_w", 0))
        h = int(row.get("bbox_h", 0))
    except Exception:
        return (0, 0, 0, 0)
    return (x, y, max(0, w), max(0, h))


def _extract_components(mask: np.ndarray, min_area: int) -> list[dict[str, object]]:
    bin_mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    out: list[dict[str, object]] = []
    for label_id in range(1, int(num_labels)):
        x, y, w, h, area = stats[label_id].tolist()
        if int(area) < int(min_area):
            continue
        cx, cy = centroids[label_id]
        out.append(
            {
                "label": int(label_id),
                "bbox": (int(x), int(y), int(w), int(h)),
                "area": int(area),
                "centroid": (float(cx), float(cy)),
            }
        )
    out.sort(key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))
    return out


def _smooth_binary_masks(masks: list[np.ndarray], alpha: float) -> list[np.ndarray]:
    if not masks:
        return []
    alpha = float(max(0.0, min(0.99, alpha)))
    acc = masks[0].astype(np.float32, copy=True)
    smoothed: list[np.ndarray] = [(acc > 0.5).astype(np.uint8)]
    for mask in masks[1:]:
        acc = mask.astype(np.float32) + (alpha * acc)
        smoothed.append((acc > 0.5).astype(np.uint8))
    return smoothed


def _prediction_to_index_mask(prediction: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction)
    if pred.ndim == 2:
        return pred.astype(np.uint8, copy=False)
    if pred.ndim >= 3:
        # Handle probability/logit volumes by choosing class argmax.
        if np.issubdtype(pred.dtype, np.floating):
            if pred.shape[-1] <= 16:
                return np.argmax(pred, axis=-1).astype(np.uint8)
            if pred.shape[0] <= 16:
                return np.argmax(pred, axis=0).astype(np.uint8)
        # Fallback for unexpected multi-channel masks.
        return np.asarray(pred[..., 0], dtype=np.uint8)
    squeezed = np.squeeze(pred)
    if squeezed.ndim != 2:
        return np.zeros((0, 0), dtype=np.uint8)
    return squeezed.astype(np.uint8, copy=False)


def _resolve_tracking_masks(
    timeline: list[DatasetImageItem],
    predictions: dict[str, np.ndarray],
    config: AnalyticsConfig,
) -> tuple[list[np.ndarray], dict[str, object]]:
    pred_masks: list[np.ndarray] = []
    class_hist = np.zeros(256, dtype=np.int64)
    for item in timeline:
        index_mask = _prediction_to_index_mask(predictions[item.uid])
        if index_mask.ndim != 2 or index_mask.size == 0:
            pred_masks.append(np.zeros((0, 0), dtype=np.uint8))
            continue
        pred_u8 = np.asarray(index_mask, dtype=np.uint8)
        pred_masks.append(pred_u8)
        class_hist += np.bincount(pred_u8.reshape(-1), minlength=256)

    requested_ids: list[int] = [int(config.root_class_id)]
    if config.lateral_class_id is not None:
        lat_id = int(config.lateral_class_id)
        if lat_id > 0 and lat_id not in requested_ids:
            requested_ids.append(lat_id)

    requested_pixels = int(sum(int(class_hist[cid]) for cid in requested_ids if 0 <= cid < class_hist.size))
    effective_ids: list[int] = [cid for cid in requested_ids if 0 <= cid < class_hist.size]
    selection_mode = "requested_classes"

    if requested_pixels <= 0:
        non_zero_ids = [int(cid) for cid in range(1, class_hist.size) if int(class_hist[cid]) > 0]
        if non_zero_ids:
            dominant = max(non_zero_ids, key=lambda cid: int(class_hist[cid]))
            effective_ids = [int(dominant)]
            selection_mode = "dominant_nonzero_fallback"
        else:
            effective_ids = []
            selection_mode = "empty_predictions"

    masks: list[np.ndarray] = []
    if effective_ids:
        ids_u8 = np.asarray(effective_ids, dtype=np.uint8)
        for pred in pred_masks:
            if pred.ndim != 2 or pred.size == 0:
                masks.append(np.zeros((0, 0), dtype=np.uint8))
                continue
            masks.append(np.isin(pred, ids_u8).astype(np.uint8))
    else:
        for pred in pred_masks:
            if pred.ndim != 2 or pred.size == 0:
                masks.append(np.zeros((0, 0), dtype=np.uint8))
                continue
            masks.append((pred > 0).astype(np.uint8))

    nonzero_top = sorted(
        [(int(cid), int(class_hist[cid])) for cid in range(1, class_hist.size) if int(class_hist[cid]) > 0],
        key=lambda kv: kv[1],
        reverse=True,
    )[:8]
    meta = {
        "selection_mode": selection_mode,
        "requested_class_ids": [int(cid) for cid in requested_ids],
        "effective_class_ids": [int(cid) for cid in effective_ids],
        "class_hist_top": nonzero_top,
    }
    return masks, meta


def _resolve_anchor_masks(
    timeline: list[DatasetImageItem],
    predictions: dict[str, np.ndarray],
    config: AnalyticsConfig,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Resolve a binary anchor mask used to suppress root-only particle noise.

    Anchors prioritize explicit seed/crown classes. If no seed/crown class is
    available, the configured shoot class is used as a backward-compatible
    anchor, then an automatically inferred top-band class is used as a final
    fallback.
    """
    if not timeline:
        return [], {"selection_mode": "no_timeline", "anchor_class_ids": []}

    pred_masks: list[np.ndarray] = []
    class_hist = np.zeros(256, dtype=np.int64)
    top_hist = np.zeros(256, dtype=np.int64)

    for item in timeline:
        idx = _prediction_to_index_mask(predictions[item.uid])
        pred_masks.append(idx)
        if idx.ndim != 2 or idx.size == 0:
            continue
        pred_u8 = np.asarray(idx, dtype=np.uint8)
        class_hist += np.bincount(pred_u8.reshape(-1), minlength=256)
        top_rows = max(1, int(round(pred_u8.shape[0] * 0.38)))
        top_hist += np.bincount(pred_u8[:top_rows, :].reshape(-1), minlength=256)

    root_id = int(config.root_class_id)
    lateral_id = int(config.lateral_class_id) if config.lateral_class_id is not None else -1
    excluded_ids = {0, root_id}
    if lateral_id > 0:
        excluded_ids.add(lateral_id)

    anchor_ids: list[int] = []
    seed_id = int(config.seed_class_id) if config.seed_class_id is not None else -1
    shoot_id = int(config.shoot_class_id) if config.shoot_class_id is not None else -1
    explicit_seed_pixels = int(class_hist[seed_id]) if 0 <= seed_id < class_hist.size else 0
    explicit_shoot_pixels = int(class_hist[shoot_id]) if 0 <= shoot_id < class_hist.size else 0
    explicit_anchor_kind = "none"
    if seed_id > 0 and seed_id not in excluded_ids and explicit_seed_pixels > 0:
        anchor_ids.append(seed_id)
        explicit_anchor_kind = "seed"
    elif shoot_id > 0 and shoot_id not in excluded_ids and explicit_shoot_pixels > 0:
        anchor_ids.append(shoot_id)
        explicit_anchor_kind = "shoot_fallback"

    candidates = [cid for cid in range(1, class_hist.size) if cid not in excluded_ids and int(class_hist[cid]) > 0]
    inferred_anchor = -1
    if not anchor_ids and candidates:
        inferred_anchor = max(candidates, key=lambda cid: int(top_hist[cid]))
        top_hits = int(top_hist[inferred_anchor])
        total_hits = int(class_hist[inferred_anchor])
        top_ratio = float(top_hits / max(1, total_hits))
        if top_hits > 0 and top_ratio >= 0.30 and inferred_anchor not in anchor_ids:
            anchor_ids.append(int(inferred_anchor))

    rgb_green_only = bool(getattr(config, "shoot_rgb_green_only_enabled", False))
    if rgb_green_only:
        masks_out: list[np.ndarray] = []
        candidate_pixels_total = 0
        frames_with_candidates = 0
        components_total = 0
        min_pixels_default = 20
        for item, pred in zip(timeline, pred_masks, strict=False):
            shape = pred.shape[:2] if pred.ndim == 2 and pred.size > 0 else item.image.shape[:2]
            green_mask = np.zeros(shape, dtype=np.uint8)
            try:
                from .pyphenotyper_adapter import build_lucifer_green_shoot_mask

                root_like_context = np.zeros(shape, dtype=np.uint8)
                if pred.ndim == 2 and pred.size > 0:
                    pred_u8_context = np.asarray(pred, dtype=np.uint8)
                    root_like_context = (pred_u8_context == np.uint8(root_id)).astype(np.uint8)
                    if lateral_id > 0:
                        root_like_context = np.maximum(root_like_context, (pred_u8_context == np.uint8(lateral_id)).astype(np.uint8))
                color_mask, color_meta = build_lucifer_green_shoot_mask(
                    np.asarray(item.image, dtype=np.uint8),
                    tuple(int(v) for v in shape),
                    root_mask=root_like_context,
                )
            except Exception:
                color_mask = np.zeros(shape, dtype=np.uint8)
                color_meta = {}
            candidate_pixels = int(np.count_nonzero(color_mask))
            min_pixels = max(min_pixels_default, int(round(float(shape[0] * shape[1]) * 0.000001)))
            if candidate_pixels >= min_pixels:
                root_like = np.zeros(shape, dtype=bool)
                if pred.ndim == 2 and pred.size > 0:
                    pred_u8 = np.asarray(pred, dtype=np.uint8)
                    root_like = pred_u8 == np.uint8(root_id)
                    if lateral_id > 0:
                        root_like |= pred_u8 == np.uint8(lateral_id)
                green_mask = ((np.asarray(color_mask, dtype=np.uint8) > 0) & (~root_like)).astype(np.uint8)
            green_pixels = int(np.count_nonzero(green_mask))
            candidate_pixels_total += green_pixels
            if green_pixels > 0:
                frames_with_candidates += 1
            try:
                components_total += int(color_meta.get("components_kept", 0) or 0)
            except Exception:
                pass
            masks_out.append(green_mask)
        return masks_out, {
            "selection_mode": "rgb_green_only",
            "anchor_class_ids": [int(v) for v in anchor_ids],
            "seed_class_id": int(seed_id) if seed_id > 0 else None,
            "shoot_class_id": int(shoot_id) if shoot_id > 0 else None,
            "inferred_anchor_class_id": int(inferred_anchor) if inferred_anchor > 0 else None,
            "rgb_green_only_enabled": True,
            "green_only_pixels": int(candidate_pixels_total),
            "frames_with_green_only_pixels": int(frames_with_candidates),
            "components_kept": int(components_total),
        }

    masks_out: list[np.ndarray] = []
    if anchor_ids:
        ids_u8 = np.asarray(anchor_ids, dtype=np.uint8)
        for pred in pred_masks:
            if pred.ndim != 2 or pred.size == 0:
                masks_out.append(np.zeros((0, 0), dtype=np.uint8))
                continue
            masks_out.append(np.isin(pred, ids_u8).astype(np.uint8))
        if explicit_anchor_kind == "seed":
            mode = "explicit_seed_class"
        elif explicit_anchor_kind == "shoot_fallback":
            mode = "explicit_shoot_anchor_fallback"
        else:
            mode = "auto_topclass"
    else:
        for pred in pred_masks:
            if pred.ndim != 2 or pred.size == 0:
                masks_out.append(np.zeros((0, 0), dtype=np.uint8))
            else:
                masks_out.append(np.zeros_like(np.asarray(pred, dtype=np.uint8), dtype=np.uint8))
        mode = "none"

    return masks_out, {
        "selection_mode": mode,
        "anchor_class_ids": [int(v) for v in anchor_ids],
        "seed_class_id": int(seed_id) if seed_id > 0 else None,
        "shoot_class_id": int(shoot_id) if shoot_id > 0 else None,
        "inferred_anchor_class_id": int(inferred_anchor) if inferred_anchor > 0 else None,
    }


def _configured_shoot_class_id(config: AnalyticsConfig) -> int | None:
    """Return the shoot mask class while preserving older seed-as-shoot projects."""
    for value in (getattr(config, "shoot_class_id", None), getattr(config, "seed_class_id", None)):
        try:
            cid = int(value) if value is not None else -1
        except Exception:
            cid = -1
        if cid > 0 and cid not in {int(config.root_class_id), int(config.lateral_class_id) if config.lateral_class_id is not None else -1}:
            return int(cid)
    return None


def _resolve_shoot_masks(
    timeline: list[DatasetImageItem],
    predictions: dict[str, np.ndarray],
    config: AnalyticsConfig,
    *,
    fallback_masks: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], dict[str, object]]:
    if not timeline:
        return [], {"selection_mode": "no_timeline", "shoot_class_ids": []}

    pred_masks: list[np.ndarray] = []
    class_hist = np.zeros(256, dtype=np.int64)
    for item in timeline:
        idx = _prediction_to_index_mask(predictions[item.uid])
        pred_masks.append(idx)
        if idx.ndim == 2 and idx.size > 0:
            class_hist += np.bincount(np.asarray(idx, dtype=np.uint8).reshape(-1), minlength=256)

    root_id = int(config.root_class_id)
    lateral_id = int(config.lateral_class_id) if config.lateral_class_id is not None else -1
    excluded_ids = {0, root_id}
    if lateral_id > 0:
        excluded_ids.add(lateral_id)

    requested_shoot_id = int(config.shoot_class_id) if config.shoot_class_id is not None else -1
    fallback_seed_id = int(config.seed_class_id) if config.seed_class_id is not None else -1
    selected_id: int | None = None
    selection_mode = "none"

    for candidate, mode in (
        (requested_shoot_id, "explicit_shoot_class"),
        (fallback_seed_id, "seed_class_shoot_fallback"),
    ):
        if candidate <= 0 or candidate in excluded_ids:
            continue
        if 0 <= candidate < class_hist.size and int(class_hist[candidate]) > 0:
            selected_id = int(candidate)
            selection_mode = mode
            break

    rgb_green_only = bool(getattr(config, "shoot_rgb_green_only_enabled", False))
    if rgb_green_only:
        masks_out: list[np.ndarray] = []
        green_pixels_total = 0
        output_pixels_total = 0
        frames_with_green = 0
        components_total = 0
        model_fallback_frames = 0
        for item, pred in zip(timeline, pred_masks, strict=False):
            shape = pred.shape[:2] if pred.ndim == 2 and pred.size > 0 else item.image.shape[:2]
            green_mask = np.zeros(shape, dtype=np.uint8)
            try:
                from .pyphenotyper_adapter import build_lucifer_green_shoot_mask

                root_like_context = np.zeros(shape, dtype=np.uint8)
                if pred.ndim == 2 and pred.size > 0:
                    pred_u8_context = np.asarray(pred, dtype=np.uint8)
                    root_like_context = (pred_u8_context == np.uint8(root_id)).astype(np.uint8)
                    if lateral_id > 0:
                        root_like_context = np.maximum(root_like_context, (pred_u8_context == np.uint8(lateral_id)).astype(np.uint8))
                color_mask, color_meta = build_lucifer_green_shoot_mask(
                    np.asarray(item.image, dtype=np.uint8),
                    tuple(int(v) for v in shape),
                    root_mask=root_like_context,
                )
            except Exception:
                color_mask = np.zeros(shape, dtype=np.uint8)
                color_meta = {}
            min_pixels = max(20, int(round(float(shape[0] * shape[1]) * 0.000001)))
            if int(np.count_nonzero(color_mask)) >= min_pixels:
                root_like = np.zeros(shape, dtype=bool)
                if pred.ndim == 2 and pred.size > 0:
                    pred_u8 = np.asarray(pred, dtype=np.uint8)
                    root_like = pred_u8 == np.uint8(root_id)
                    if lateral_id > 0:
                        root_like |= pred_u8 == np.uint8(lateral_id)
                green_mask = ((np.asarray(color_mask, dtype=np.uint8) > 0) & (~root_like)).astype(np.uint8)
            detector_green_pixels = int(np.count_nonzero(green_mask))
            if detector_green_pixels <= 0 and selected_id is not None and pred.ndim == 2 and pred.size > 0:
                green_mask = (np.asarray(pred, dtype=np.uint8) == np.uint8(selected_id)).astype(np.uint8)
                model_fallback_frames += 1
            green_pixels = int(np.count_nonzero(green_mask))
            green_pixels_total += int(detector_green_pixels)
            output_pixels_total += int(green_pixels)
            if detector_green_pixels > 0:
                frames_with_green += 1
            try:
                components_total += int(color_meta.get("components_kept", 0) or 0)
            except Exception:
                pass
            masks_out.append(green_mask)
        return masks_out, {
            "selection_mode": "rgb_green_only",
            "requested_shoot_class_id": int(requested_shoot_id) if requested_shoot_id > 0 else None,
            "fallback_seed_class_id": int(fallback_seed_id) if fallback_seed_id > 0 else None,
            "shoot_class_ids": [int(selected_id)] if selected_id is not None else [],
            "rgb_green_only_enabled": True,
            "green_only_pixels": int(green_pixels_total),
            "output_shoot_pixels": int(output_pixels_total),
            "frames_with_green_only_pixels": int(frames_with_green),
            "model_fallback_frames": int(model_fallback_frames),
            "failure_policy": "retain_model_on_detector_abstention",
            "components_kept": int(components_total),
        }

    masks_out: list[np.ndarray] = []
    if selected_id is not None:
        for pred in pred_masks:
            if pred.ndim != 2 or pred.size == 0:
                masks_out.append(np.zeros((0, 0), dtype=np.uint8))
                continue
            masks_out.append((np.asarray(pred, dtype=np.uint8) == np.uint8(selected_id)).astype(np.uint8))
    elif fallback_masks is not None:
        for idx, item in enumerate(timeline):
            if idx < len(fallback_masks):
                fb = np.asarray(fallback_masks[idx], dtype=np.uint8)
                if fb.ndim == 2 and fb.size > 0:
                    masks_out.append((fb > 0).astype(np.uint8))
                    continue
            shape = item.image.shape[:2] if getattr(item, "image", None) is not None else (0, 0)
            masks_out.append(np.zeros(shape, dtype=np.uint8))
        selection_mode = "anchor_fallback"
    else:
        for item in timeline:
            shape = item.image.shape[:2] if getattr(item, "image", None) is not None else (0, 0)
            masks_out.append(np.zeros(shape, dtype=np.uint8))

    return masks_out, {
        "selection_mode": selection_mode,
        "requested_shoot_class_id": int(requested_shoot_id) if requested_shoot_id > 0 else None,
        "fallback_seed_class_id": int(fallback_seed_id) if fallback_seed_id > 0 else None,
        "shoot_class_ids": [int(selected_id)] if selected_id is not None else [],
    }


def _build_crown_zone_from_anchor_masks(
    anchor_masks: list[np.ndarray],
    shape_hw: tuple[int, int],
    config: AnalyticsConfig,
) -> tuple[np.ndarray, int, str]:
    h, w = shape_hw
    anchor_union = np.zeros((h, w), dtype=np.uint8)
    anchor_bottoms: list[int] = []
    for anchor in anchor_masks:
        a = np.asarray(anchor, dtype=np.uint8)
        if a.ndim != 2 or a.shape != (h, w):
            continue
        aa = (a > 0).astype(np.uint8)
        if np.count_nonzero(aa) <= 0:
            continue
        anchor_union = np.maximum(anchor_union, aa)
        ys = np.where(aa > 0)[0]
        if ys.size > 0:
            anchor_bottoms.append(int(np.max(ys)))
    if np.count_nonzero(anchor_union) <= 0:
        return np.zeros((h, w), dtype=np.uint8), 0, "empty_anchor_union"

    radius = int(max(4, min(128, int(getattr(config, "shoot_crown_lock_radius_px", 24)))))
    ksz = int((2 * radius) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    crown_zone = cv2.dilate(anchor_union, kernel, iterations=1)

    top_extra = int(max(8, min(256, int(getattr(config, "shoot_crown_lock_top_extra_px", 56)))))
    if anchor_bottoms:
        top_limit = int(min(h, max(1, int(np.percentile(np.asarray(anchor_bottoms, dtype=np.float64), 95.0)) + top_extra)))
    else:
        top_limit = int(min(h, max(1, int(round(0.35 * float(h))))))
    top_band = np.zeros((h, w), dtype=np.uint8)
    top_band[:top_limit, :] = 1
    crown_zone = (crown_zone > 0).astype(np.uint8) * top_band
    if np.count_nonzero(crown_zone) <= 0:
        return np.zeros((h, w), dtype=np.uint8), 0, "empty_crown_zone"
    return crown_zone.astype(np.uint8, copy=False), int(top_limit), "ok"


def _dilate_binary(mask: np.ndarray, radius_px: int) -> np.ndarray:
    src = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if int(np.count_nonzero(src)) <= 0:
        return np.zeros_like(src, dtype=np.uint8)
    radius = int(max(0, min(128, int(radius_px))))
    if radius <= 0:
        return src.astype(np.uint8, copy=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(src, kernel, iterations=1).astype(np.uint8, copy=False)


def _apply_shoot_crown_lock(
    root_masks: list[np.ndarray],
    anchor_masks: list[np.ndarray],
    config: AnalyticsConfig,
) -> tuple[list[np.ndarray], dict[str, object]]:
    if not root_masks:
        return root_masks, {"enabled": False, "status": "no_masks"}
    if not bool(getattr(config, "shoot_crown_lock_enabled", True)):
        return root_masks, {"enabled": False, "status": "disabled"}
    if not isinstance(anchor_masks, list) or len(anchor_masks) != len(root_masks):
        return root_masks, {"enabled": False, "status": "no_anchor_masks"}

    first = np.asarray(root_masks[0], dtype=np.uint8)
    if first.ndim != 2 or first.size == 0:
        return root_masks, {"enabled": False, "status": "invalid_shape"}
    h, w = first.shape[:2]
    crown_zone, top_limit, crown_status = _build_crown_zone_from_anchor_masks(anchor_masks, (h, w), config)
    if np.count_nonzero(crown_zone) <= 0:
        return root_masks, {"enabled": False, "status": str(crown_status)}

    ref_frames = int(max(1, min(len(root_masks), int(getattr(config, "shoot_crown_lock_ref_frames", 3)))))
    ref_idxs = list(range(ref_frames))
    ref_crops: list[np.ndarray] = []
    for idx in ref_idxs:
        rm = np.asarray(root_masks[idx], dtype=np.uint8)
        if rm.ndim != 2 or rm.shape != (h, w):
            continue
        ref_crops.append(((rm > 0).astype(np.uint8) * crown_zone).astype(np.uint8))
    if not ref_crops:
        return root_masks, {"enabled": False, "status": "empty_reference"}

    stack = np.stack(ref_crops, axis=0).astype(np.uint8)
    vote = np.sum(stack, axis=0)
    threshold = int(math.ceil(0.5 * float(stack.shape[0])))
    crown_baseline = (vote >= threshold).astype(np.uint8)
    crown_baseline = cv2.morphologyEx(crown_baseline, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    locked_masks: list[np.ndarray] = []
    zone = crown_zone > 0
    start_frame = int(max(0, min(len(root_masks) - 1, int(getattr(config, "shoot_crown_lock_start_frame", 7)))))
    for rm in root_masks:
        src = (np.asarray(rm, dtype=np.uint8) > 0).astype(np.uint8)
        if len(locked_masks) >= start_frame:
            src[zone] = crown_baseline[zone]
        locked_masks.append(src.astype(np.uint8))
    return locked_masks, {
        "enabled": True,
        "status": "applied",
        "radius_px": int(max(4, min(128, int(getattr(config, "shoot_crown_lock_radius_px", 24))))),
        "ref_frames": int(ref_frames),
        "start_frame": int(start_frame),
        "top_limit_px": int(top_limit),
        "zone_pixels": int(np.count_nonzero(zone)),
    }


def _component_anchor_overlap(comp: dict[str, object], anchor_mask: np.ndarray | None) -> float:
    if anchor_mask is None or not isinstance(anchor_mask, np.ndarray) or anchor_mask.ndim != 2 or anchor_mask.size == 0:
        return 0.0
    bbox_obj = comp.get("bbox")
    if not isinstance(bbox_obj, tuple) or len(bbox_obj) != 4:
        return 0.0
    x, y, w, h = (int(bbox_obj[0]), int(bbox_obj[1]), int(bbox_obj[2]), int(bbox_obj[3]))
    if w <= 0 or h <= 0:
        return 0.0
    ah, aw = anchor_mask.shape[:2]
    x0 = max(0, min(aw - 1, x))
    y0 = max(0, min(ah - 1, y))
    x1 = max(x0, min(aw, x + w))
    y1 = max(y0, min(ah, y + h))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = np.asarray(anchor_mask[y0:y1, x0:x1], dtype=np.uint8)
    if crop.size == 0:
        return 0.0
    return float(np.count_nonzero(crop)) / float(max(1, crop.shape[0] * crop.shape[1]))


def _component_anchor_nearby(comp: dict[str, object], anchor_mask: np.ndarray | None, margin_px: int = 28) -> bool:
    if anchor_mask is None or not isinstance(anchor_mask, np.ndarray) or anchor_mask.ndim != 2 or anchor_mask.size == 0:
        return False
    bbox_obj = comp.get("bbox")
    if not isinstance(bbox_obj, tuple) or len(bbox_obj) != 4:
        return False
    x, y, w, h = (int(bbox_obj[0]), int(bbox_obj[1]), int(bbox_obj[2]), int(bbox_obj[3]))
    if w <= 0 or h <= 0:
        return False
    ah, aw = anchor_mask.shape[:2]
    margin = max(0, int(margin_px))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(aw, x + w + margin)
    y1 = min(ah, y + h + margin)
    if x1 <= x0 or y1 <= y0:
        return False
    crop = np.asarray(anchor_mask[y0:y1, x0:x1], dtype=np.uint8)
    return bool(np.count_nonzero(crop) > 0)


def _component_is_obvious_artifact(
    comp: dict[str, object],
    shape_hw: tuple[int, int],
    min_area: int,
    anchor_mask: np.ndarray | None,
) -> bool:
    h_img, w_img = shape_hw
    bbox_obj = comp.get("bbox")
    if not isinstance(bbox_obj, tuple) or len(bbox_obj) != 4:
        return True
    x, y, w, h = (int(bbox_obj[0]), int(bbox_obj[1]), int(bbox_obj[2]), int(bbox_obj[3]))
    if w <= 0 or h <= 0:
        return True
    area = int(comp.get("area", 0))
    anchor_overlap = _component_anchor_overlap(comp, anchor_mask)
    anchor_near = _component_anchor_nearby(comp, anchor_mask, margin_px=32)

    # Static tray rim / top bar style artifact.
    if (
        y <= int(round(0.16 * float(h_img)))
        and w >= int(round(0.42 * float(w_img)))
        and h <= int(round(0.09 * float(h_img)))
        and anchor_overlap < 0.01
    ):
        return True

    # Bottom dust or isolated tiny particles not connected to shoot/seed anchors.
    if (
        y >= int(round(0.52 * float(h_img)))
        and area <= int(max(1, min_area * 10))
        and h <= int(round(0.14 * float(h_img)))
        and not anchor_near
    ):
        return True

    # Very horizontal small component far from anchor is unlikely to be a root.
    if (
        h < max(10, int(round(0.22 * float(h_img))))
        and float(h) < (0.68 * float(max(1, w)))
        and area <= int(max(1, min_area * 14))
        and anchor_overlap < 0.01
        and not anchor_near
    ):
        return True

    return False


def _component_seed_score(
    comp: dict[str, object],
    shape_hw: tuple[int, int],
    min_area: int,
    anchor_mask: np.ndarray | None,
) -> float:
    h_img, w_img = shape_hw
    bbox_obj = comp.get("bbox")
    if not isinstance(bbox_obj, tuple) or len(bbox_obj) != 4:
        return -1e9
    x, y, w, h = (int(bbox_obj[0]), int(bbox_obj[1]), int(bbox_obj[2]), int(bbox_obj[3]))
    if w <= 0 or h <= 0:
        return -1e9
    area = int(comp.get("area", 0))

    top_pref = 1.0 - min(1.0, float(y) / max(1.0, float(h_img) * 0.90))
    elong = min(2.5, float(h) / max(1.0, float(w))) / 2.5
    area_term = min(1.0, float(area) / float(max(1, int(min_area) * 8)))
    anchor_overlap = _component_anchor_overlap(comp, anchor_mask)
    anchor_term = min(1.0, anchor_overlap * 6.0)
    anchor_near_term = 1.0 if _component_anchor_nearby(comp, anchor_mask, margin_px=32) else 0.0

    wide_bar_penalty = 1.0 if (w >= int(round(0.55 * float(w_img))) and h <= int(round(0.16 * float(h_img)))) else 0.0
    bottom_speck_penalty = 1.0 if (y >= int(round(0.55 * float(h_img))) and h <= int(round(0.10 * float(h_img))) and area <= int(max(1, min_area * 4))) else 0.0
    no_anchor_penalty = 1.0 if (anchor_term < 0.05 and anchor_near_term < 0.5 and y >= int(round(0.45 * float(h_img)))) else 0.0

    return (
        0.42 * top_pref
        + 0.24 * elong
        + 0.16 * area_term
        + 0.38 * anchor_term
        + 0.20 * anchor_near_term
        - 0.90 * wide_bar_penalty
        - 0.45 * bottom_speck_penalty
        - 0.42 * no_anchor_penalty
    )


def _filter_tracking_components(
    comps: list[dict[str, object]],
    shape_hw: tuple[int, int],
    min_area: int,
    anchor_mask: np.ndarray | None,
) -> list[dict[str, object]]:
    if not comps:
        return []
    scored: list[tuple[float, dict[str, object]]] = []
    for comp in comps:
        if _component_is_obvious_artifact(comp, shape_hw, min_area, anchor_mask):
            continue
        score = _component_seed_score(comp, shape_hw, min_area, anchor_mask)
        scored.append((float(score), comp))
    if not scored:
        return []
    # Keep all plausible candidates; if everything scores poorly, keep top few
    # to avoid hard failure on unusual imaging conditions.
    kept = [comp for score, comp in scored if score >= -0.20]
    if kept:
        kept.sort(key=lambda c: (int(c["bbox"][0]), int(c["bbox"][1])))
        return kept
    fallback = [comp for _score, comp in sorted(scored, key=lambda kv: kv[0], reverse=True)[: max(1, min(6, len(scored)))]]
    fallback.sort(key=lambda c: (int(c["bbox"][0]), int(c["bbox"][1])))
    return fallback


def _extract_components_per_frame_adaptive(masks: list[np.ndarray], min_area: int) -> tuple[list[list[dict[str, object]]], int]:
    if not masks:
        return [], max(1, int(min_area))
    thresholds = [
        max(1, int(min_area)),
        max(1, int(min_area) // 2),
        max(1, int(min_area) // 4),
        1,
    ]
    unique_thresholds: list[int] = []
    for value in thresholds:
        if value not in unique_thresholds:
            unique_thresholds.append(value)

    fallback_per_frame: list[list[dict[str, object]]] = [[] for _ in masks]
    fallback_threshold = max(1, int(min_area))
    for threshold in unique_thresholds:
        per_frame = [_extract_components(mask, threshold) for mask in masks]
        if any(bool(comps) for comps in per_frame):
            return per_frame, int(threshold)
        fallback_per_frame = per_frame
        fallback_threshold = int(threshold)
    return fallback_per_frame, fallback_threshold


def _track_auto(
    masks: list[np.ndarray],
    config: AnalyticsConfig,
    anchor_masks: list[np.ndarray] | None = None,
) -> dict[str, object]:
    if not masks:
        return {"track_ids": [], "track_bboxes": {}, "overlap_frames": {}, "seed_frame": -1, "effective_min_component_area": int(max(1, config.min_component_area))}
    shape_hw = masks[0].shape[:2]
    per_frame_raw, effective_min_area = _extract_components_per_frame_adaptive(masks, config.min_component_area)
    if not isinstance(anchor_masks, list) or len(anchor_masks) != len(masks):
        anchor_masks = [np.zeros_like(mask, dtype=np.uint8) for mask in masks]

    per_frame: list[list[dict[str, object]]] = []
    for frame_idx, comps in enumerate(per_frame_raw):
        anchor = anchor_masks[frame_idx] if frame_idx < len(anchor_masks) else None
        filtered = _filter_tracking_components(comps, shape_hw, int(effective_min_area), anchor)
        per_frame.append(filtered)

    expected_count = int(max(0, int(getattr(config, "expected_track_count", 0))))
    seed_frame = -1
    best_seed_key = (-1, -1.0, 10**9)
    for i, comps in enumerate(per_frame):
        if not comps:
            continue
        anchor = anchor_masks[i] if i < len(anchor_masks) else None
        scores = [_component_seed_score(comp, shape_hw, int(effective_min_area), anchor) for comp in comps]
        if expected_count > 0:
            ranked = sorted(scores, reverse=True)
            quality = float(sum(ranked[: min(len(ranked), expected_count)]))
            coverage = int(min(len(comps), expected_count))
        else:
            quality = float(sum(scores))
            coverage = int(len(comps))
        key = (coverage, quality, -i)
        if key > best_seed_key:
            best_seed_key = key
            seed_frame = i
    if seed_frame < 0:
        return {
            "track_ids": [],
            "track_bboxes": {},
            "overlap_frames": {},
            "seed_frame": -1,
            "effective_min_component_area": int(effective_min_area),
        }

    seed_anchor = anchor_masks[seed_frame] if seed_frame < len(anchor_masks) else None
    seed_scored = [
        (
            _component_seed_score(comp, shape_hw, int(effective_min_area), seed_anchor),
            comp,
        )
        for comp in per_frame[seed_frame]
    ]
    seed_scored.sort(key=lambda kv: kv[0], reverse=True)

    seed: list[dict[str, object]] = []
    if expected_count > 0:
        # Anchor-aware seed selection: keep one plausible root per shoot/seed anchor
        # when anchor classes are available. This suppresses isolated particle tracks.
        anchor_components: list[dict[str, object]] = []
        if isinstance(seed_anchor, np.ndarray) and seed_anchor.ndim == 2 and seed_anchor.size > 0 and np.count_nonzero(seed_anchor) > 0:
            anchor_min_area = max(6, int(effective_min_area) // 3)
            anchor_components = _extract_components(seed_anchor, anchor_min_area)
        anchor_components.sort(key=lambda comp: float(comp.get("centroid", (0.0, 0.0))[0]))
        anchor_x_targets = [float(comp.get("centroid", (0.0, 0.0))[0]) for comp in anchor_components]

        if anchor_x_targets:
            candidate_pool = [comp for _score, comp in seed_scored[: max(expected_count * 6, expected_count)]]
            used: set[int] = set()
            selected: list[dict[str, object]] = []
            target_count = min(expected_count, len(candidate_pool))
            for ax in anchor_x_targets[: target_count]:
                best_idx = -1
                best_cost = float("inf")
                for idx, comp in enumerate(candidate_pool):
                    if idx in used:
                        continue
                    centroid = comp.get("centroid")
                    if not isinstance(centroid, tuple) or len(centroid) != 2:
                        continue
                    cx = float(centroid[0])
                    score = _component_seed_score(comp, shape_hw, int(effective_min_area), seed_anchor)
                    overlap = _component_anchor_overlap(comp, seed_anchor)
                    cost = abs(cx - ax) - (58.0 * overlap) - (14.0 * score)
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = idx
                if best_idx >= 0:
                    used.add(best_idx)
                    selected.append(candidate_pool[best_idx])
            if len(selected) < target_count:
                for idx, comp in enumerate(candidate_pool):
                    if idx in used:
                        continue
                    selected.append(comp)
                    used.add(idx)
                    if len(selected) >= target_count:
                        break
            seed = list(selected[:target_count])

    if not seed:
        if expected_count > 0:
            seed_scored = seed_scored[: min(len(seed_scored), expected_count)]
        seed = [comp for _score, comp in seed_scored]

    seed.sort(key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))
    track_ids = [f"plant_{i:02d}" for i in range(1, len(seed) + 1)]
    track_bboxes: dict[str, list[BBox]] = {k: [(0, 0, 0, 0) for _ in masks] for k in track_ids}
    overlap_frames: dict[str, int | None] = {k: None for k in track_ids}
    track_miss_streak: dict[str, int] = {k: 0 for k in track_ids}
    track_anchor_x: dict[str, float] = {}
    for track_id, comp in zip(track_ids, seed):
        track_bboxes[track_id][seed_frame] = _expand_bbox(comp["bbox"], config.bbox_padding, shape_hw)
        centroid = comp.get("centroid")
        if isinstance(centroid, tuple) and len(centroid) == 2:
            track_anchor_x[track_id] = float(centroid[0])

    for frame in range(seed_frame + 1, len(masks)):
        frame_anchor = anchor_masks[frame] if frame < len(anchor_masks) else None
        frame_components = per_frame[frame]
        comp_boxes = [_expand_bbox(comp["bbox"], config.bbox_padding, shape_hw) for comp in frame_components]
        assigned_tracks: set[str] = set()
        assigned_comps: set[int] = set()
        candidates: list[tuple[float, str, int]] = []
        for track_id in track_ids:
            prev = track_bboxes[track_id][frame - 1]
            if prev[2] <= 0 or prev[3] <= 0:
                continue
            adaptive_margin = int(max(config.tracking_search_margin, round(max(prev[2], prev[3]) * 1.10)))
            search = _expand_bbox(prev, adaptive_margin, shape_hw)
            for ci, cb in enumerate(comp_boxes):
                if not _bbox_intersects(search, cb):
                    continue
                iou = _bbox_iou(prev, cb)
                dist = _bbox_center_distance(prev, cb)
                span = float(max(1, prev[2], prev[3], adaptive_margin))
                prev_cx = float(prev[0] + (0.5 * prev[2]))
                cb_cx = float(cb[0] + (0.5 * cb[2]))
                x_pen = abs(cb_cx - prev_cx) / float(max(1.0, prev[2] * 2.5))
                anchor_x = track_anchor_x.get(track_id)
                anchor_x_pen = (
                    abs(cb_cx - float(anchor_x)) / float(max(1.0, prev[2] * 3.0))
                    if anchor_x is not None
                    else 0.0
                )
                prev_bottom = float(prev[1] + prev[3])
                cb_bottom = float(cb[1] + cb[3])
                down_growth = max(-1.0, min(1.0, (cb_bottom - prev_bottom) / float(max(1.0, prev[3]))))
                anchor_boost = _component_anchor_overlap(frame_components[ci], frame_anchor)
                if not _tracking_assignment_allowed(
                    prev,
                    cb,
                    shape_hw,
                    config,
                    anchor_x=anchor_x,
                    anchor_overlap=float(anchor_boost),
                ):
                    continue
                score = (
                    (2.3 * iou)
                    + max(0.0, 1.0 - (dist / (2.2 * span)))
                    + (0.35 * anchor_boost)
                    + (0.25 * max(0.0, down_growth))
                    - (0.20 * x_pen)
                    - (0.22 * anchor_x_pen)
                )
                candidates.append((score, track_id, ci))

        assignments: dict[str, int] = {}
        for score, track_id, ci in sorted(candidates, key=lambda x: x[0], reverse=True):
            if score <= 0 or track_id in assigned_tracks or ci in assigned_comps:
                continue
            assignments[track_id] = ci
            assigned_tracks.add(track_id)
            assigned_comps.add(ci)

        for track_id in track_ids:
            if track_id in assignments:
                continue
            prev = track_bboxes[track_id][frame - 1]
            best_ci = -1
            best_dist = float("inf")
            best_score = -1e9
            for ci, cb in enumerate(comp_boxes):
                if ci in assigned_comps:
                    continue
                d = _bbox_center_distance(prev, cb)
                prev_cx = float(prev[0] + (0.5 * prev[2]))
                cb_cx = float(cb[0] + (0.5 * cb[2]))
                x_pen = abs(cb_cx - prev_cx) / float(max(1.0, prev[2] * 2.5))
                anchor_x = track_anchor_x.get(track_id)
                anchor_x_pen = (
                    abs(cb_cx - float(anchor_x)) / float(max(1.0, prev[2] * 3.0))
                    if anchor_x is not None
                    else 0.0
                )
                anchor_boost = _component_anchor_overlap(frame_components[ci], frame_anchor)
                s = (
                    (1.0 - min(1.0, d / float(max(1.0, prev[3] * 3.0))))
                    - (0.25 * x_pen)
                    - (0.22 * anchor_x_pen)
                    + (0.25 * anchor_boost)
                )
                if s > best_score or (abs(s - best_score) < 1.0e-6 and d < best_dist):
                    best_score = s
                    best_dist = d
                    best_ci = ci
            if best_ci >= 0:
                adaptive_margin = int(max(config.tracking_search_margin, round(max(prev[2], prev[3]) * 1.10)))
                max_jump = float(max(prev[2], prev[3], adaptive_margin) * 5.5)
                best_anchor_boost = _component_anchor_overlap(frame_components[best_ci], frame_anchor)
                best_anchor_x = track_anchor_x.get(track_id)
                if best_dist <= max_jump and _tracking_assignment_allowed(
                    prev,
                    comp_boxes[best_ci],
                    shape_hw,
                    config,
                    anchor_x=best_anchor_x,
                    anchor_overlap=float(best_anchor_boost),
                ):
                    assignments[track_id] = best_ci
                    assigned_comps.add(best_ci)

        for track_id in track_ids:
            if track_id in assignments:
                prev = track_bboxes[track_id][frame - 1]
                assigned_box = comp_boxes[assignments[track_id]]
                # Keep continuity by preserving a union envelope between previous
                # and current detections; this reduces premature bbox truncation.
                track_bboxes[track_id][frame] = _union_bbox(prev, assigned_box, shape_hw)
                track_miss_streak[track_id] = 0
                centroid = frame_components[assignments[track_id]].get("centroid")
                if isinstance(centroid, tuple) and len(centroid) == 2:
                    cx = float(centroid[0])
                    if track_id in track_anchor_x:
                        track_anchor_x[track_id] = (0.78 * float(track_anchor_x[track_id])) + (0.22 * cx)
                    else:
                        track_anchor_x[track_id] = cx
            else:
                prev = track_bboxes[track_id][frame - 1]
                if prev[2] > 0 and prev[3] > 0:
                    search_probe = _expand_bbox(prev, max(config.tracking_search_margin // 2, 8), shape_hw)
                    sx, sy, sw, sh = search_probe
                    probe_crop = masks[frame][sy : sy + sh, sx : sx + sw] if sw > 0 and sh > 0 else np.zeros((0, 0), dtype=np.uint8)
                    signal_px = int(np.count_nonzero(np.asarray(probe_crop, dtype=np.uint8))) if probe_crop.size > 0 else 0
                    if signal_px <= max(2, int(effective_min_area // 8)):
                        track_miss_streak[track_id] = int(track_miss_streak.get(track_id, 0)) + 1
                    else:
                        track_miss_streak[track_id] = max(0, int(track_miss_streak.get(track_id, 0)) - 1)

                    if int(track_miss_streak.get(track_id, 0)) >= 3:
                        track_bboxes[track_id][frame] = (0, 0, 0, 0)
                    else:
                        grow_x = int(max(0, round(prev[2] * 0.03)))
                        grow_y = int(max(4, round(prev[3] * 0.09)))
                        max_h = int(round(shape_hw[0] * 0.95))
                        new_w = min(int(prev[2] + (2 * grow_x)), shape_hw[1])
                        new_h = min(int(prev[3] + grow_y), max_h)
                        track_bboxes[track_id][frame] = _clip_bbox(
                            (prev[0] - grow_x, prev[1], new_w, new_h),
                            shape_hw,
                        )
                else:
                    track_miss_streak[track_id] = int(track_miss_streak.get(track_id, 0)) + 1
                    track_bboxes[track_id][frame] = prev

        for i, left in enumerate(track_ids):
            for right in track_ids[i + 1 :]:
                if _bbox_intersects(track_bboxes[left][frame], track_bboxes[right][frame]):
                    if overlap_frames[left] is None:
                        overlap_frames[left] = frame
                    if overlap_frames[right] is None:
                        overlap_frames[right] = frame
                    # Keep boxes live to avoid growth freeze on temporary overlap.
                    continue

    # Backfill boxes before the seed frame so the preview and exports retain
    # valid tracking overlays across the whole analyzed timeline. The previous
    # implementation only propagated from the seed frame forward, leaving every
    # earlier frame at (0, 0, 0, 0) even when matching components existed.
    track_anchor_x_backward = dict(track_anchor_x)
    track_miss_streak_backward: dict[str, int] = {k: 0 for k in track_ids}
    for frame in range(seed_frame - 1, -1, -1):
        frame_anchor = anchor_masks[frame] if frame < len(anchor_masks) else None
        frame_components = per_frame[frame]
        comp_boxes = [_expand_bbox(comp["bbox"], config.bbox_padding, shape_hw) for comp in frame_components]
        assigned_tracks: set[str] = set()
        assigned_comps: set[int] = set()
        candidates: list[tuple[float, str, int]] = []
        for track_id in track_ids:
            nxt = track_bboxes[track_id][frame + 1]
            if nxt[2] <= 0 or nxt[3] <= 0:
                continue
            adaptive_margin = int(max(config.tracking_search_margin, round(max(nxt[2], nxt[3]) * 1.10)))
            search = _expand_bbox(nxt, adaptive_margin, shape_hw)
            for ci, cb in enumerate(comp_boxes):
                if not _bbox_intersects(search, cb):
                    continue
                iou = _bbox_iou(nxt, cb)
                dist = _bbox_center_distance(nxt, cb)
                span = float(max(1, nxt[2], nxt[3], adaptive_margin))
                nxt_cx = float(nxt[0] + (0.5 * nxt[2]))
                cb_cx = float(cb[0] + (0.5 * cb[2]))
                x_pen = abs(cb_cx - nxt_cx) / float(max(1.0, nxt[2] * 2.5))
                anchor_x = track_anchor_x_backward.get(track_id)
                anchor_x_pen = (
                    abs(cb_cx - float(anchor_x)) / float(max(1.0, nxt[2] * 3.0))
                    if anchor_x is not None
                    else 0.0
                )
                nxt_top = float(nxt[1])
                cb_top = float(cb[1])
                top_consistency = max(0.0, 1.0 - (abs(cb_top - nxt_top) / float(max(1.0, nxt[3]))))
                anchor_boost = _component_anchor_overlap(frame_components[ci], frame_anchor)
                if not _tracking_assignment_allowed(
                    nxt,
                    cb,
                    shape_hw,
                    config,
                    anchor_x=anchor_x,
                    anchor_overlap=float(anchor_boost),
                ):
                    continue
                score = (
                    (2.3 * iou)
                    + max(0.0, 1.0 - (dist / (2.2 * span)))
                    + (0.35 * anchor_boost)
                    + (0.18 * top_consistency)
                    - (0.20 * x_pen)
                    - (0.22 * anchor_x_pen)
                )
                candidates.append((score, track_id, ci))

        assignments: dict[str, int] = {}
        for score, track_id, ci in sorted(candidates, key=lambda x: x[0], reverse=True):
            if score <= 0 or track_id in assigned_tracks or ci in assigned_comps:
                continue
            assignments[track_id] = ci
            assigned_tracks.add(track_id)
            assigned_comps.add(ci)

        for track_id in track_ids:
            if track_id in assignments:
                continue
            nxt = track_bboxes[track_id][frame + 1]
            best_ci = -1
            best_dist = float("inf")
            best_score = -1e9
            for ci, cb in enumerate(comp_boxes):
                if ci in assigned_comps:
                    continue
                d = _bbox_center_distance(nxt, cb)
                nxt_cx = float(nxt[0] + (0.5 * nxt[2]))
                cb_cx = float(cb[0] + (0.5 * cb[2]))
                x_pen = abs(cb_cx - nxt_cx) / float(max(1.0, nxt[2] * 2.5))
                anchor_x = track_anchor_x_backward.get(track_id)
                anchor_x_pen = (
                    abs(cb_cx - float(anchor_x)) / float(max(1.0, nxt[2] * 3.0))
                    if anchor_x is not None
                    else 0.0
                )
                anchor_boost = _component_anchor_overlap(frame_components[ci], frame_anchor)
                s = (
                    (1.0 - min(1.0, d / float(max(1.0, nxt[3] * 3.0))))
                    - (0.25 * x_pen)
                    - (0.22 * anchor_x_pen)
                    + (0.25 * anchor_boost)
                )
                if s > best_score or (abs(s - best_score) < 1.0e-6 and d < best_dist):
                    best_score = s
                    best_dist = d
                    best_ci = ci
            if best_ci >= 0:
                adaptive_margin = int(max(config.tracking_search_margin, round(max(nxt[2], nxt[3]) * 1.10)))
                max_jump = float(max(nxt[2], nxt[3], adaptive_margin) * 5.5)
                best_anchor_boost = _component_anchor_overlap(frame_components[best_ci], frame_anchor)
                best_anchor_x = track_anchor_x_backward.get(track_id)
                if best_dist <= max_jump and _tracking_assignment_allowed(
                    nxt,
                    comp_boxes[best_ci],
                    shape_hw,
                    config,
                    anchor_x=best_anchor_x,
                    anchor_overlap=float(best_anchor_boost),
                ):
                    assignments[track_id] = best_ci
                    assigned_comps.add(best_ci)

        for track_id in track_ids:
            if track_id in assignments:
                assigned_box = comp_boxes[assignments[track_id]]
                track_bboxes[track_id][frame] = _clip_bbox(assigned_box, shape_hw)
                track_miss_streak_backward[track_id] = 0
                centroid = frame_components[assignments[track_id]].get("centroid")
                if isinstance(centroid, tuple) and len(centroid) == 2:
                    cx = float(centroid[0])
                    if track_id in track_anchor_x_backward:
                        track_anchor_x_backward[track_id] = (0.78 * float(track_anchor_x_backward[track_id])) + (0.22 * cx)
                    else:
                        track_anchor_x_backward[track_id] = cx
            else:
                nxt = track_bboxes[track_id][frame + 1]
                if nxt[2] > 0 and nxt[3] > 0:
                    search_probe = _expand_bbox(nxt, max(config.tracking_search_margin // 2, 8), shape_hw)
                    sx, sy, sw, sh = search_probe
                    probe_crop = masks[frame][sy : sy + sh, sx : sx + sw] if sw > 0 and sh > 0 else np.zeros((0, 0), dtype=np.uint8)
                    signal_px = int(np.count_nonzero(np.asarray(probe_crop, dtype=np.uint8))) if probe_crop.size > 0 else 0
                    if signal_px <= max(2, int(effective_min_area // 8)):
                        track_miss_streak_backward[track_id] = int(track_miss_streak_backward.get(track_id, 0)) + 1
                    else:
                        track_miss_streak_backward[track_id] = max(0, int(track_miss_streak_backward.get(track_id, 0)) - 1)

                    if int(track_miss_streak_backward.get(track_id, 0)) >= 2:
                        track_bboxes[track_id][frame] = (0, 0, 0, 0)
                    else:
                        shrink_x = int(max(0, round(nxt[2] * 0.03)))
                        shrink_y = int(max(4, round(nxt[3] * 0.08)))
                        track_bboxes[track_id][frame] = _clip_bbox(
                            (
                                int(nxt[0] + shrink_x),
                                int(nxt[1]),
                                int(max(0, nxt[2] - (2 * shrink_x))),
                                int(max(0, nxt[3] - shrink_y)),
                            ),
                            shape_hw,
                        )
                else:
                    track_bboxes[track_id][frame] = (0, 0, 0, 0)

        for i, left in enumerate(track_ids):
            for right in track_ids[i + 1 :]:
                if _bbox_intersects(track_bboxes[left][frame], track_bboxes[right][frame]):
                    if overlap_frames[left] is None:
                        overlap_frames[left] = frame
                    if overlap_frames[right] is None:
                        overlap_frames[right] = frame
                    continue

    return {
        "track_ids": track_ids,
        "track_bboxes": track_bboxes,
        "overlap_frames": overlap_frames,
        "seed_frame": seed_frame,
        "effective_min_component_area": int(effective_min_area),
    }


def _projection_weighted_median(
    weights: np.ndarray,
    left: int,
    right: int,
    fallback: float,
) -> float:
    projection = np.asarray(weights, dtype=np.float64).reshape(-1)
    lo = int(max(0, min(projection.size, int(left))))
    hi = int(max(lo, min(projection.size, int(right))))
    if hi <= lo:
        return float(fallback)
    lane = np.nan_to_num(projection[lo:hi], nan=0.0, posinf=0.0, neginf=0.0)
    lane = np.maximum(lane, 0.0)
    support = float(np.sum(lane))
    if support <= 0.0:
        return float(fallback)
    offset = int(np.searchsorted(np.cumsum(lane), 0.5 * support, side="left"))
    return float(lo + max(0, min(hi - lo - 1, offset)))


def _robust_motion_median(values: list[float], fallback: float, limit: float) -> float:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size <= 0:
        return float(fallback)
    median = float(np.median(finite))
    if finite.size >= 3:
        deviations = np.abs(finite - median)
        mad = float(np.median(deviations))
        tolerance = max(3.0, 3.5 * mad)
        inliers = finite[deviations <= tolerance]
        if inliers.size > 0:
            median = float(np.median(inliers))
    return float(max(-abs(float(limit)), min(abs(float(limit)), median)))


def _crown_lane_frame_observation(
    root_mask: np.ndarray,
    anchor_mask: np.ndarray | None,
    shape_hw: tuple[int, int],
    top_limit: int,
    left: int,
    right: int,
    min_support: int,
) -> dict[str, object] | None:
    h, w = shape_hw
    lo = int(max(0, min(w - 1, int(left))))
    hi = int(max(lo + 1, min(w, int(right))))
    top = int(max(1, min(h, int(top_limit))))

    root = np.asarray(root_mask, dtype=np.uint8)
    root_valid = root.ndim == 2 and root.shape == (h, w)
    anchor = np.asarray(anchor_mask, dtype=np.uint8) if isinstance(anchor_mask, np.ndarray) else None
    anchor_valid = isinstance(anchor, np.ndarray) and anchor.ndim == 2 and anchor.shape == (h, w)

    anchor_observation: dict[str, object] | None = None
    if anchor_valid:
        anchor_crop = (anchor[:top, lo:hi] > 0).astype(np.uint8)
        anchor_support = int(np.count_nonzero(anchor_crop))
        if anchor_support >= int(max(1, min_support)):
            anchor_ys, anchor_xs = np.where(anchor_crop > 0)
            base_y = float(np.percentile(anchor_ys, 78.0))
            base_pixels = anchor_ys >= int(round(base_y))
            if int(np.count_nonzero(base_pixels)) >= int(max(1, min_support // 2)):
                anchor_observation = {
                    "center_x": float(lo + np.median(anchor_xs[base_pixels])),
                    "center_y": float(np.median(anchor_ys[base_pixels])),
                    "source": "anchor_base",
                    "support_px": int(anchor_support),
                }

    # Root-colored plate rims and tape can appear above the plants. When shoot
    # evidence exists, accept a root crown only near that shoot base.
    if root_valid:
        root_crop = (root[:top, lo:hi] > 0).astype(np.uint8)
        support = int(np.count_nonzero(root_crop))
        if support >= int(max(1, min_support)):
            ys, xs = np.where(root_crop > 0)
            if anchor_observation is not None:
                anchor_x = float(anchor_observation["center_x"]) - float(lo)
                anchor_y = float(anchor_observation["center_y"])
                x_radius = max(12.0, 0.22 * float(max(1, hi - lo)))
                y_radius_above = max(12.0, 0.08 * float(h))
                y_radius_below = max(18.0, 0.12 * float(h))
                near_anchor = (
                    (np.abs(xs.astype(np.float64) - anchor_x) <= x_radius)
                    & (ys.astype(np.float64) >= anchor_y - y_radius_above)
                    & (ys.astype(np.float64) <= anchor_y + y_radius_below)
                )
                if int(np.count_nonzero(near_anchor)) >= int(max(1, min_support // 2)):
                    local_xs = xs[near_anchor].astype(np.float64)
                    local_ys = ys[near_anchor].astype(np.float64)
                    normalized_distance = np.hypot(
                        (local_xs - anchor_x) / max(1.0, x_radius),
                        (local_ys - anchor_y)
                        / max(1.0, max(y_radius_above, y_radius_below)),
                    )
                    nearest = int(np.argmin(normalized_distance))
                    nearest_x = float(local_xs[nearest])
                    nearest_y = float(local_ys[nearest])
                    crown_radius = max(3.0, 0.012 * float(h))
                    crown_pixels = (
                        np.hypot(local_xs - nearest_x, local_ys - nearest_y)
                        <= crown_radius
                    )
                    return {
                        "center_x": float(lo + np.median(local_xs[crown_pixels])),
                        "center_y": float(np.median(local_ys[crown_pixels])),
                        "source": "anchor_guided_root",
                        "support_px": int(np.count_nonzero(near_anchor)),
                    }
                return anchor_observation

            crown_y = float(np.percentile(ys, 2.0))
            crown_band_height = int(max(12, min(round(0.15 * float(top)), round(0.045 * float(h)))))
            in_crown_band = ys <= int(round(crown_y)) + crown_band_height
            if int(np.count_nonzero(in_crown_band)) >= int(max(1, min_support // 2)):
                return {
                    "center_x": float(lo + np.median(xs[in_crown_band])),
                    "center_y": float(crown_y),
                    "source": "top_root",
                    "support_px": int(support),
                }

    return anchor_observation


def _estimate_crown_frame_alignment(
    timeline: list[DatasetImageItem] | None,
    frame_count: int,
    shape_hw: tuple[int, int],
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    warps = [identity.copy() for _ in range(max(0, int(frame_count)))]
    metadata: list[dict[str, object]] = [
        {
            "accepted": False,
            "usable": False,
            "method": "unavailable",
            "status": "missing_image",
            "score": 0.0,
            "shift_x": 0.0,
            "shift_y": 0.0,
            "rotation_deg": 0.0,
            "reference_frame_index": 0,
            "warp_matrix": identity.astype(float).tolist(),
        }
        for _ in range(max(0, int(frame_count)))
    ]
    if frame_count <= 0 or not isinstance(timeline, list) or len(timeline) < frame_count:
        return warps, metadata

    h, w = shape_hw

    def _image_at(index: int) -> np.ndarray | None:
        if index < 0 or index >= len(timeline):
            return None
        raw_image = getattr(timeline[index], "image", None)
        if raw_image is None:
            return None
        try:
            image = np.asarray(raw_image, dtype=np.uint8)
        except (TypeError, ValueError, OverflowError):
            return None
        if image.ndim not in (2, 3) or image.size <= 0:
            return None
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        return image

    images = [_image_at(index) for index in range(frame_count)]
    reference = images[0]
    if reference is None:
        return warps, metadata
    if float(np.std(reference.astype(np.float32))) < 1.0:
        for frame_index, image in enumerate(images):
            metadata[frame_index]["status"] = (
                "insufficient_texture" if image is not None else "missing_image"
            )
        return warps, metadata

    metadata[0] = {
        "accepted": False,
        "usable": True,
        "method": "reference_frame",
        "status": "reference",
        "score": 1.0,
        "shift_x": 0.0,
        "shift_y": 0.0,
        "rotation_deg": 0.0,
        "reference_frame_index": 0,
        "warp_matrix": identity.astype(float).tolist(),
    }
    registration = StabilizationConfig(
        reference_mode="previous",
        target_mode="dish_frame",
        downscale_max_dim=1200,
    )
    for frame_index in range(1, frame_count):
        moving = images[frame_index]
        if moving is None:
            continue
        if float(np.std(moving.astype(np.float32))) < 1.0:
            metadata[frame_index]["status"] = "insufficient_texture"
            continue
        result = stabilize_against_reference(reference, moving, registration)
        matrix = np.asarray(result.warp_matrix, dtype=np.float32).reshape(2, 3)
        metadata[frame_index] = {
            "accepted": bool(result.accepted),
            "usable": bool(result.accepted),
            "method": str(result.method),
            "status": str(result.status),
            "score": float(result.score),
            "shift_x": float(result.shift_x),
            "shift_y": float(result.shift_y),
            "rotation_deg": float(result.rotation_deg),
            "reference_frame_index": 0,
            "warp_matrix": matrix.astype(float).tolist(),
        }
        if not result.accepted:
            continue
        warps[frame_index] = matrix
        reference = next_stabilization_reference(reference, result, registration)
    return warps, metadata


def _track_arabidopsis_crown_lanes(
    masks: list[np.ndarray],
    config: AnalyticsConfig,
    anchor_masks: list[np.ndarray] | None = None,
    timeline: list[DatasetImageItem] | None = None,
) -> dict[str, object]:
    """Track permanent left-to-right identities from per-frame crown evidence."""
    expected_count = int(max(0, int(getattr(config, "expected_track_count", 0))))
    if not masks or expected_count <= 1:
        return _track_auto(masks, config, anchor_masks=anchor_masks)

    first = np.asarray(masks[0], dtype=np.uint8)
    if first.ndim != 2 or first.size == 0:
        return _track_auto(masks, config, anchor_masks=anchor_masks)
    h, w = first.shape[:2]
    if h <= 0 or w <= 0:
        return _track_auto(masks, config, anchor_masks=anchor_masks)

    top_limit = int(max(1, min(h, round(0.42 * float(h)))))
    ref_frames = int(
        max(
            1,
            min(
                len(masks),
                int(max(1, getattr(config, "shoot_crown_lock_ref_frames", 3))),
            ),
        )
    )
    root_projection = np.zeros(w, dtype=np.float64)
    anchor_projection = np.zeros(w, dtype=np.float64)
    for frame_idx in range(ref_frames):
        root = np.asarray(masks[frame_idx], dtype=np.uint8)
        if root.ndim == 2 and root.shape == (h, w):
            root_projection += np.count_nonzero(root[:top_limit, :] > 0, axis=0)
        if isinstance(anchor_masks, list) and frame_idx < len(anchor_masks):
            anchor = np.asarray(anchor_masks[frame_idx], dtype=np.uint8)
            if anchor.ndim == 2 and anchor.shape == (h, w):
                anchor_projection += np.count_nonzero(anchor[:top_limit, :] > 0, axis=0)

    detected_layout: np.ndarray | None = None
    detected_layout_source = "full_frame_layout"
    minimum_layout_area = int(max(3, int(getattr(config, "min_component_area", 30)) // 3))
    merge_distance = float(max(8.0, 0.035 * float(w)))
    minimum_spacing = float(max(6.0, 0.025 * float(w)))
    for source_name, source_masks in (
        ("anchor_components", anchor_masks),
        ("top_root_components", masks),
    ):
        if not isinstance(source_masks, list):
            continue
        direct_layout_candidates: list[tuple[float, int, np.ndarray, str]] = []
        inferred_layout_candidates: list[tuple[float, int, np.ndarray, str]] = []
        for frame_idx in range(min(ref_frames, len(source_masks))):
            candidate_mask = np.asarray(source_masks[frame_idx], dtype=np.uint8)
            if candidate_mask.ndim != 2 or candidate_mask.shape != (h, w):
                continue
            components = _extract_components(
                candidate_mask[:top_limit, :],
                minimum_layout_area,
            )
            filtered_components: list[tuple[float, float]] = []
            for component in components:
                bbox = component.get("bbox")
                centroid = component.get("centroid")
                if not isinstance(bbox, tuple) or len(bbox) != 4:
                    continue
                if not isinstance(centroid, tuple) or len(centroid) != 2:
                    continue
                bbox_width = int(bbox[2])
                if bbox_width >= int(round(0.24 * float(w))):
                    continue
                area = float(max(0, int(component.get("area", 0))))
                center_x = float(centroid[0])
                if area <= 0.0 or center_x < 0.02 * float(w) or center_x > 0.98 * float(w):
                    continue
                filtered_components.append((center_x, area))
            if not filtered_components:
                continue

            clusters: list[list[tuple[float, float]]] = []
            for center_x, area in sorted(filtered_components):
                if not clusters:
                    clusters.append([(center_x, area)])
                    continue
                cluster_area = sum(value[1] for value in clusters[-1])
                cluster_center = sum(value[0] * value[1] for value in clusters[-1]) / max(
                    1.0,
                    cluster_area,
                )
                if abs(center_x - cluster_center) <= merge_distance:
                    clusters[-1].append((center_x, area))
                else:
                    clusters.append([(center_x, area)])
            cluster_values = [
                (
                    sum(value[0] * value[1] for value in cluster) / max(1.0, sum(value[1] for value in cluster)),
                    sum(value[1] for value in cluster),
                )
                for cluster in clusters
            ]
            if len(cluster_values) < expected_count:
                continue
            ranked = sorted(cluster_values, key=lambda value: value[1], reverse=True)[: max(expected_count * 2, expected_count)]
            from itertools import combinations

            for selected in combinations(ranked, expected_count):
                centers_candidate = np.asarray(sorted(value[0] for value in selected), dtype=np.float64)
                spacings_candidate = np.diff(centers_candidate)
                if spacings_candidate.size and float(np.min(spacings_candidate)) < minimum_spacing:
                    continue
                span = float(centers_candidate[-1] - centers_candidate[0])
                if expected_count >= 3 and span < 0.20 * float(w):
                    continue
                spacing_mean = float(np.mean(spacings_candidate)) if spacings_candidate.size else 1.0
                spacing_cv = (
                    float(np.std(spacings_candidate)) / max(1.0, spacing_mean)
                    if spacings_candidate.size
                    else 0.0
                )
                # A complete set containing an edge artifact and one missing crown can
                # still have the expected component count. Reject strongly irregular
                # layouts so the missing-lane fit below can preserve physical IDs.
                if expected_count >= 4 and spacing_cv > 0.32:
                    continue
                selected_area = float(sum(value[1] for value in selected))
                score = (
                    math.log1p(selected_area)
                    + (2.0 * span / max(1.0, float(w)))
                    - (1.5 * spacing_cv)
                    - (0.05 * float(frame_idx))
                )
                direct_layout_candidates.append(
                    (float(score), int(frame_idx), centers_candidate, source_name)
                )

            # Fit a regular lane lattice from one fewer reliable crowns. This handles
            # plates where a seedling is absent while a small rim/label component would
            # otherwise be mistaken for the final plant.
            inferred_count = expected_count - 1
            if expected_count >= 4 and len(ranked) >= inferred_count:
                for selected in combinations(ranked, inferred_count):
                    selected_sorted = sorted(selected, key=lambda value: value[0])
                    selected_centers = np.asarray(
                        [value[0] for value in selected_sorted],
                        dtype=np.float64,
                    )
                    selected_area = float(sum(value[1] for value in selected_sorted))
                    for lane_indices_tuple in combinations(range(expected_count), inferred_count):
                        lane_indices = np.asarray(lane_indices_tuple, dtype=np.float64)
                        lane_mean = float(np.mean(lane_indices))
                        center_mean = float(np.mean(selected_centers))
                        denominator = float(np.sum((lane_indices - lane_mean) ** 2))
                        if denominator <= 0.0:
                            continue
                        spacing = float(
                            np.sum(
                                (lane_indices - lane_mean)
                                * (selected_centers - center_mean)
                            )
                            / denominator
                        )
                        if spacing < 0.04 * float(w) or spacing > 0.24 * float(w):
                            continue
                        offset = float(center_mean - spacing * lane_mean)
                        layout = offset + spacing * np.arange(expected_count, dtype=np.float64)
                        if layout[0] < 0.02 * float(w) or layout[-1] > 0.98 * float(w):
                            continue
                        fitted_selected = offset + spacing * lane_indices
                        residual_ratio = float(
                            np.sqrt(np.mean((selected_centers - fitted_selected) ** 2))
                            / max(1.0, spacing)
                        )
                        if residual_ratio > 0.22:
                            continue
                        score = (
                            math.log1p(selected_area)
                            + (2.0 * float(layout[-1] - layout[0]) / max(1.0, float(w)))
                            - (4.0 * residual_ratio)
                            - 1.0
                            - (0.05 * float(frame_idx))
                        )
                        inferred_layout_candidates.append(
                            (
                                float(score),
                                int(frame_idx),
                                layout,
                                f"{source_name}_inferred_missing_lane",
                            )
                        )
        layout_candidates = direct_layout_candidates or inferred_layout_candidates
        if layout_candidates:
            layout_candidates.sort(key=lambda value: (-value[0], value[1]))
            detected_layout = layout_candidates[0][2]
            detected_layout_source = layout_candidates[0][3]
            break

    defaults = (
        detected_layout
        if detected_layout is not None
        else np.linspace(0.14 * float(w), 0.86 * float(w), expected_count, dtype=np.float64)
    )
    boundaries = [0]
    for idx in range(expected_count - 1):
        boundaries.append(int(round(0.5 * float(defaults[idx] + defaults[idx + 1]))))
    boundaries.append(int(w))

    min_support = int(max(3, int(getattr(config, "min_component_area", 30)) // 3))
    centers: list[float] = []
    center_sources: list[str] = []
    center_support: list[int] = []
    for idx, default_x in enumerate(defaults.tolist()):
        left = int(max(0, min(w - 1, boundaries[idx])))
        right = int(max(left + 1, min(w, boundaries[idx + 1])))
        anchor_support = int(round(float(np.sum(anchor_projection[left:right]))))
        root_support = int(round(float(np.sum(root_projection[left:right]))))
        if anchor_support >= min_support:
            projection = anchor_projection
            source = "anchor"
            support = anchor_support
        elif root_support >= min_support:
            projection = root_projection
            source = "top_root"
            support = root_support
        else:
            projection = np.zeros(w, dtype=np.float64)
            source = "layout_fallback"
            support = max(anchor_support, root_support)
        detected_x = _projection_weighted_median(projection, left, right, float(default_x))
        lane_width = float(max(1, right - left))
        max_shift = max(8.0, 0.30 * lane_width)
        center_x = max(float(default_x) - max_shift, min(float(default_x) + max_shift, detected_x))
        center_x = max(float(left), min(float(right - 1), center_x))
        centers.append(float(center_x))
        center_sources.append(str(source))
        center_support.append(int(support))

    center_boundaries = [0]
    for idx in range(expected_count - 1):
        center_boundaries.append(int(round(0.5 * float(centers[idx] + centers[idx + 1]))))
    center_boundaries.append(int(w))
    spacings = np.diff(np.asarray(centers, dtype=np.float64))
    typical_spacing = float(np.median(spacings)) if spacings.size > 0 else float(w) / float(expected_count)
    typical_spacing = max(8.0, typical_spacing)
    search_margin = int(max(4, round(0.10 * typical_spacing)))
    frame_min_support = int(max(2, math.ceil(float(min_support) / float(max(1, ref_frames)))))

    frame_observations: list[list[dict[str, object] | None]] = []
    for frame_idx, raw_root in enumerate(masks):
        root = np.asarray(raw_root, dtype=np.uint8)
        anchor = None
        if isinstance(anchor_masks, list) and frame_idx < len(anchor_masks):
            candidate_anchor = np.asarray(anchor_masks[frame_idx], dtype=np.uint8)
            if candidate_anchor.ndim == 2 and candidate_anchor.shape == (h, w):
                anchor = candidate_anchor
        observations: list[dict[str, object] | None] = []
        for idx in range(expected_count):
            left = int(max(0, center_boundaries[idx] - search_margin))
            right = int(min(w, center_boundaries[idx + 1] + search_margin))
            observations.append(
                _crown_lane_frame_observation(
                    root,
                    anchor,
                    (h, w),
                    top_limit,
                    left,
                    right,
                    frame_min_support,
                )
            )
        frame_observations.append(observations)

    alignment_warps, alignment_metadata = _estimate_crown_frame_alignment(
        timeline,
        len(masks),
        (h, w),
    )
    baseline_y: list[float] = []
    baseline_x: list[float] = []
    for track_idx in range(expected_count):
        first_observation = frame_observations[0][track_idx] if frame_observations else None
        first_valid_entry = next(
            (
                (frame_index, frame[track_idx])
                for frame_index, frame in enumerate(frame_observations)
                if isinstance(frame[track_idx], dict)
            ),
            None,
        )
        if isinstance(first_observation, dict):
            baseline_x.append(float(first_observation["center_x"]))
            baseline_y.append(float(first_observation["center_y"]))
            continue
        if first_valid_entry is None:
            baseline_x.append(float(centers[track_idx]))
            baseline_y.append(0.18 * float(top_limit))
            continue
        first_valid_index, first_valid = first_valid_entry
        baseline_point = np.array(
            [float(first_valid["center_x"]), float(first_valid["center_y"]), 1.0],
            dtype=np.float64,
        )
        first_valid_meta = (
            alignment_metadata[first_valid_index]
            if first_valid_index < len(alignment_metadata)
            else {}
        )
        if bool(first_valid_meta.get("usable", False)) and first_valid_index < len(alignment_warps):
            reference_point = (
                np.asarray(alignment_warps[first_valid_index], dtype=np.float64).reshape(2, 3)
                @ baseline_point
            )
            baseline_x.append(float(reference_point[0]))
            baseline_y.append(float(reference_point[1]))
        else:
            baseline_x.append(float(centers[track_idx]))
            baseline_y.append(float(first_valid["center_y"]))

    mask_shift_x: list[float] = []
    mask_shift_y: list[float] = []
    observed_counts: list[int] = []
    prior_shift_x = 0.0
    prior_shift_y = 0.0
    fallback_motion_limit = max(
        24.0,
        1.5 * float(max(1, int(config.tracking_search_margin))),
        0.02 * float(w),
    )
    for observations in frame_observations:
        root_dx = [
            float(obs["center_x"]) - float(baseline_x[idx])
            for idx, obs in enumerate(observations)
            if isinstance(obs, dict) and str(obs.get("source", "")) == "top_root"
        ]
        all_dx = [
            float(obs["center_x"]) - float(baseline_x[idx])
            for idx, obs in enumerate(observations)
            if isinstance(obs, dict)
        ]
        root_dy = [
            float(obs["center_y"]) - float(baseline_y[idx])
            for idx, obs in enumerate(observations)
            if isinstance(obs, dict) and str(obs.get("source", "")) == "top_root"
        ]
        all_dy = [
            float(obs["center_y"]) - float(baseline_y[idx])
            for idx, obs in enumerate(observations)
            if isinstance(obs, dict)
        ]
        dx_values = root_dx if len(root_dx) >= 2 else all_dx
        dy_values = root_dy if len(root_dy) >= 2 else all_dy
        shift_x = _robust_motion_median(dx_values, prior_shift_x, fallback_motion_limit)
        shift_y = _robust_motion_median(dy_values, prior_shift_y, fallback_motion_limit)
        mask_shift_x.append(float(shift_x))
        mask_shift_y.append(float(shift_y))
        observed_counts.append(int(len(all_dx)))
        prior_shift_x = float(shift_x)
        prior_shift_y = float(shift_y)

    frame_centers_x: list[list[float]] = []
    frame_centers_y: list[list[float]] = []
    frame_center_sources: list[list[str]] = []
    frame_motion_sources: list[str] = []
    global_shift_x: list[float] = []
    global_shift_y: list[float] = []
    residual_x = [0.0 for _ in range(expected_count)]
    residual_y = [0.0 for _ in range(expected_count)]
    residual_x_limit = max(4.0, min(12.0, 0.025 * typical_spacing, 0.5 * fallback_motion_limit))
    residual_y_limit = max(4.0, min(10.0, 0.25 * fallback_motion_limit))
    for frame_idx, observations in enumerate(frame_observations):
        centers_x_frame: list[float] = []
        centers_y_frame: list[float] = []
        sources_frame: list[str] = []
        predicted_centers: list[tuple[float, float]] = []
        alignment_meta = alignment_metadata[frame_idx] if frame_idx < len(alignment_metadata) else {}
        alignment_usable = bool(alignment_meta.get("usable", False))
        if alignment_usable and frame_idx < len(alignment_warps):
            inverse = cv2.invertAffineTransform(np.asarray(alignment_warps[frame_idx], dtype=np.float32).reshape(2, 3))
            for track_idx in range(expected_count):
                reference_point = np.array(
                    [float(baseline_x[track_idx]), float(baseline_y[track_idx]), 1.0],
                    dtype=np.float64,
                )
                moved = np.asarray(inverse, dtype=np.float64) @ reference_point
                predicted_centers.append((float(moved[0]), float(moved[1])))
            motion_source = "image_registration"
        else:
            predicted_centers = [
                (
                    float(baseline_x[track_idx]) + float(mask_shift_x[frame_idx]),
                    float(baseline_y[track_idx]) + float(mask_shift_y[frame_idx]),
                )
                for track_idx in range(expected_count)
            ]
            motion_source = "root_crown_fallback"
        frame_motion_sources.append(motion_source)
        global_shift_x.append(
            float(np.median([predicted_centers[idx][0] - float(baseline_x[idx]) for idx in range(expected_count)]))
        )
        global_shift_y.append(
            float(np.median([predicted_centers[idx][1] - float(baseline_y[idx]) for idx in range(expected_count)]))
        )
        for track_idx, obs in enumerate(observations):
            predicted_x, predicted_y = predicted_centers[track_idx]
            source = "image_registration" if alignment_usable else "motion_fallback"
            if isinstance(obs, dict):
                candidate_residual_x = float(obs["center_x"]) - predicted_x
                candidate_residual_y = float(obs["center_y"]) - predicted_y
                if abs(candidate_residual_x) <= max(fallback_motion_limit, 0.18 * typical_spacing):
                    source = str(obs.get("source", "evidence"))
                    weight = 0.46 if source == "top_root" else 0.22
                    residual_x[track_idx] = (
                        ((1.0 - weight) * float(residual_x[track_idx]))
                        + (weight * max(-residual_x_limit, min(residual_x_limit, candidate_residual_x)))
                    )
                    residual_y[track_idx] = (
                        ((1.0 - weight) * float(residual_y[track_idx]))
                        + (weight * max(-residual_y_limit, min(residual_y_limit, candidate_residual_y)))
                    )
                else:
                    residual_x[track_idx] *= 0.82
                    residual_y[track_idx] *= 0.82
            else:
                residual_x[track_idx] *= 0.82
                residual_y[track_idx] *= 0.82
            center_x = predicted_x + residual_x[track_idx]
            center_y = predicted_y + residual_y[track_idx]
            if not alignment_usable:
                center_x = max(
                    float(baseline_x[track_idx]) - fallback_motion_limit,
                    min(float(baseline_x[track_idx]) + fallback_motion_limit, center_x),
                )
                center_y = max(
                    float(baseline_y[track_idx]) - fallback_motion_limit,
                    min(float(baseline_y[track_idx]) + fallback_motion_limit, center_y),
                )
            centers_x_frame.append(float(max(0.0, min(float(w - 1), center_x))))
            centers_y_frame.append(float(max(0.0, min(float(top_limit - 1), center_y))))
            sources_frame.append(str(source))
        frame_centers_x.append(centers_x_frame)
        frame_centers_y.append(centers_y_frame)
        frame_center_sources.append(sources_frame)

    track_ids = [f"plant_{idx + 1:02d}" for idx in range(expected_count)]
    track_bboxes: dict[str, list[BBox]] = {track_id: [] for track_id in track_ids}
    for frame_idx in range(len(masks)):
        for idx, track_id in enumerate(track_ids):
            lane_width = int(max(1, center_boundaries[idx + 1] - center_boundaries[idx]))
            crown_width = int(max(24, min(lane_width, round(0.46 * float(lane_width)))))
            crown_height = int(max(16, min(top_limit, round(0.46 * float(top_limit)))))
            x = int(round(float(frame_centers_x[frame_idx][idx]) - (0.5 * float(crown_width))))
            y = int(round(float(frame_centers_y[frame_idx][idx]) - (0.5 * float(crown_height))))
            track_bboxes[track_id].append(_clip_bbox((x, y, crown_width, crown_height), (h, w)))

    return {
        "track_ids": track_ids,
        "track_bboxes": track_bboxes,
        "overlap_frames": {track_id: None for track_id in track_ids},
        "seed_frame": 0,
        "effective_min_component_area": int(max(1, int(config.min_component_area))),
        "source": "arabidopsis_crown_lanes",
        "identity_initialization": {
            "status": "crown_evidence" if all(source != "layout_fallback" for source in center_sources) else "layout_fallback",
            "top_limit_px": int(top_limit),
            "reference_frames": int(ref_frames),
            "centers_x": [float(round(value, 3)) for value in centers],
            "layout_source": detected_layout_source,
            "center_sources": center_sources,
            "center_support_px": center_support,
            "per_frame_centers_x": [
                [float(round(value, 3)) for value in values]
                for values in frame_centers_x
            ],
            "per_frame_centers_y": [
                [float(round(value, 3)) for value in values]
                for values in frame_centers_y
            ],
            "per_frame_center_sources": frame_center_sources,
            "per_frame_global_shift_x": [float(round(value, 3)) for value in global_shift_x],
            "per_frame_global_shift_y": [float(round(value, 3)) for value in global_shift_y],
            "per_frame_motion_source": frame_motion_sources,
            "per_frame_alignment_method": [str(meta.get("method", "")) for meta in alignment_metadata],
            "per_frame_alignment_status": [str(meta.get("status", "")) for meta in alignment_metadata],
            "per_frame_alignment_score": [float(round(float(meta.get("score", 0.0)), 4)) for meta in alignment_metadata],
            "per_frame_alignment_accepted": [bool(meta.get("accepted", False)) for meta in alignment_metadata],
            "per_frame_alignment_usable": [bool(meta.get("usable", False)) for meta in alignment_metadata],
            "per_frame_alignment_shift_x": [float(round(float(meta.get("shift_x", 0.0)), 4)) for meta in alignment_metadata],
            "per_frame_alignment_shift_y": [float(round(float(meta.get("shift_y", 0.0)), 4)) for meta in alignment_metadata],
            "per_frame_alignment_rotation_deg": [float(round(float(meta.get("rotation_deg", 0.0)), 4)) for meta in alignment_metadata],
            "per_frame_alignment_reference_frame_index": [int(meta.get("reference_frame_index", 0)) for meta in alignment_metadata],
            "per_frame_alignment_warp_matrix": [
                meta.get("warp_matrix", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                for meta in alignment_metadata
            ],
            "per_frame_observed_count": observed_counts,
        },
    }


def _track_manual_roi(
    masks: list[np.ndarray],
    config: AnalyticsConfig,
    anchor_masks: list[np.ndarray] | None = None,
) -> dict[str, object]:
    # Uses first detectable frame ROIs and keeps them stable across time.
    out = _track_auto(masks, config, anchor_masks=anchor_masks)
    track_ids = list(out.get("track_ids", []))
    track_bboxes = dict(out.get("track_bboxes", {}))
    seed_frame = int(out.get("seed_frame", -1))
    if seed_frame < 0:
        return out
    for track_id in track_ids:
        seed_box = track_bboxes[track_id][seed_frame]
        for frame in range(len(masks)):
            track_bboxes[track_id][frame] = seed_box
    out["track_bboxes"] = track_bboxes
    return out


def _normalize_external_tracking_seed(
    tracking_seed: dict[str, object] | None,
    frame_count: int,
    shape_hw: tuple[int, int],
    fallback_min_area: int,
) -> dict[str, object] | None:
    if not isinstance(tracking_seed, dict):
        return None
    raw_track_bboxes = tracking_seed.get("track_bboxes")
    if not isinstance(raw_track_bboxes, dict) or not raw_track_bboxes:
        return None
    raw_track_ids = tracking_seed.get("track_ids")
    track_ids = [str(track_id).strip() for track_id in raw_track_ids] if isinstance(raw_track_ids, list) else []
    if not track_ids:
        track_ids = [str(track_id).strip() for track_id in raw_track_bboxes.keys() if str(track_id).strip()]
    normalized_bboxes: dict[str, list[BBox]] = {}
    valid_track_ids: list[str] = []
    for track_id in track_ids:
        series_obj = raw_track_bboxes.get(track_id)
        if not isinstance(series_obj, list):
            continue
        series: list[BBox] = []
        for frame_index in range(frame_count):
            if frame_index < len(series_obj):
                bbox_obj = series_obj[frame_index]
                if isinstance(bbox_obj, (list, tuple)) and len(bbox_obj) == 4:
                    series.append(_clip_bbox((int(bbox_obj[0]), int(bbox_obj[1]), int(bbox_obj[2]), int(bbox_obj[3])), shape_hw))
                    continue
            series.append((0, 0, 0, 0))
        if any(int(bbox[2]) > 0 and int(bbox[3]) > 0 for bbox in series):
            normalized_bboxes[track_id] = series
            valid_track_ids.append(track_id)
    if not valid_track_ids:
        return None

    raw_overlap = tracking_seed.get("overlap_frames")
    overlap_frames: dict[str, int | None] = {}
    for track_id in valid_track_ids:
        overlap_val = raw_overlap.get(track_id) if isinstance(raw_overlap, dict) else None
        overlap_frames[track_id] = int(overlap_val) if isinstance(overlap_val, (int, np.integer)) else None

    raw_seed_frame = tracking_seed.get("seed_frame")
    seed_frame = int(raw_seed_frame) if isinstance(raw_seed_frame, (int, np.integer)) else -1
    if not (0 <= seed_frame < frame_count):
        best_idx = -1
        best_count = -1
        for frame_index in range(frame_count):
            count = sum(1 for track_id in valid_track_ids if normalized_bboxes[track_id][frame_index][2] > 0 and normalized_bboxes[track_id][frame_index][3] > 0)
            if count > best_count:
                best_count = count
                best_idx = frame_index
        seed_frame = best_idx

    return {
        "track_ids": valid_track_ids,
        "track_bboxes": normalized_bboxes,
        "overlap_frames": overlap_frames,
        "seed_frame": int(seed_frame),
        "effective_min_component_area": int(max(1, int(tracking_seed.get("effective_min_component_area", fallback_min_area)))),
        "source": str(tracking_seed.get("source", "external")),
    }


def build_external_tracking_seed_from_measurement_rows(
    items: list[DatasetImageItem],
    measurement_rows: list[dict[str, object]],
    track_order: list[str] | None = None,
    *,
    timeline: list[dict[str, object]] | None = None,
    seed_frame: int | None = None,
    overlap_frames: dict[str, object] | None = None,
    source: str = "external",
) -> dict[str, object] | None:
    if not items or not measurement_rows:
        return None

    uid_to_frame: dict[str, int] = {}
    for frame_index, item in enumerate(items):
        uid = str(getattr(item, "uid", "")).strip()
        if uid and uid not in uid_to_frame:
            uid_to_frame[uid] = int(frame_index)
    if not uid_to_frame:
        return None

    def _coerce_int(value: object, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return int(default)

    raw_track_bboxes: dict[str, list[BBox]] = {}
    row_overlap_frames: dict[str, int | None] = {}
    frame_count = len(items)
    for row in measurement_rows:
        if not isinstance(row, dict):
            continue
        uid = str(row.get("uid", "")).strip()
        if uid not in uid_to_frame:
            continue
        track_id = str(row.get("plant_id", "")).strip()
        if not track_id:
            continue
        series = raw_track_bboxes.setdefault(track_id, [(0, 0, 0, 0) for _ in range(frame_count)])
        series[uid_to_frame[uid]] = (
            max(0, _coerce_int(row.get("bbox_x", 0))),
            max(0, _coerce_int(row.get("bbox_y", 0))),
            max(0, _coerce_int(row.get("bbox_w", 0))),
            max(0, _coerce_int(row.get("bbox_h", 0))),
        )
        overlap_raw = row.get("overlap_frame")
        if track_id not in row_overlap_frames and overlap_raw not in (None, ""):
            row_overlap_frames[track_id] = _coerce_int(overlap_raw)

    if not raw_track_bboxes:
        return None

    ordered_track_ids: list[str] = []
    if isinstance(track_order, list):
        for track_id in track_order:
            key = str(track_id).strip()
            if key and key in raw_track_bboxes and key not in ordered_track_ids:
                ordered_track_ids.append(key)
    for track_id in sorted(raw_track_bboxes.keys()):
        if track_id not in ordered_track_ids:
            ordered_track_ids.append(track_id)

    normalized_track_bboxes: dict[str, list[BBox]] = {}
    valid_track_ids: list[str] = []
    for track_id in ordered_track_ids:
        series = raw_track_bboxes.get(track_id)
        if not isinstance(series, list):
            continue
        if not any(int(bbox[2]) > 0 and int(bbox[3]) > 0 for bbox in series):
            continue
        normalized_track_bboxes[track_id] = series
        valid_track_ids.append(track_id)
    if not valid_track_ids:
        return None

    resolved_overlap_frames: dict[str, int | None] = {}
    for track_id in valid_track_ids:
        raw_overlap = overlap_frames.get(track_id) if isinstance(overlap_frames, dict) else None
        if raw_overlap in (None, ""):
            raw_overlap = row_overlap_frames.get(track_id)
        resolved_overlap_frames[track_id] = (
            _coerce_int(raw_overlap)
            if raw_overlap not in (None, "")
            else None
        )

    mapped_seed_frame = -1
    if isinstance(seed_frame, (int, np.integer)):
        seed_idx = int(seed_frame)
        if isinstance(timeline, list) and 0 <= seed_idx < len(timeline):
            entry = timeline[seed_idx]
            entry_uid = str(entry.get("uid", "")).strip() if isinstance(entry, dict) else ""
            mapped_seed_frame = int(uid_to_frame.get(entry_uid, -1))
        elif 0 <= seed_idx < frame_count:
            mapped_seed_frame = seed_idx

    return {
        "track_ids": valid_track_ids,
        "track_bboxes": normalized_track_bboxes,
        "overlap_frames": resolved_overlap_frames,
        "seed_frame": int(mapped_seed_frame),
        "source": str(source or "external"),
    }


def _resolve_tracking(
    root_masks: list[np.ndarray],
    config: AnalyticsConfig,
    anchor_masks: list[np.ndarray] | None = None,
    timeline: list[DatasetImageItem] | None = None,
    progress_callback: Callable[[str, float], bool] | None = None,
) -> dict[str, object]:
    if not root_masks:
        return {
            "track_ids": [],
            "track_bboxes": {},
            "overlap_frames": {},
            "seed_frame": -1,
            "effective_min_component_area": int(max(1, int(config.min_component_area))),
        }

    shape_hw = tuple(int(v) for v in root_masks[0].shape[:2])
    external_tracking = _normalize_external_tracking_seed(
        getattr(config, "external_tracking_seed", None),
        frame_count=len(root_masks),
        shape_hw=shape_hw,
        fallback_min_area=int(max(1, int(config.min_component_area))),
    )
    if external_tracking is not None:
        return external_tracking
    if config.tracking_mode == "manual_roi":
        return _track_manual_roi(root_masks, config, anchor_masks=anchor_masks)
    if config.tracking_mode == "arabidopsis_crown_lanes":
        return _track_arabidopsis_crown_lanes(
            root_masks,
            config,
            anchor_masks=anchor_masks,
            timeline=timeline,
        )
    if config.tracking_mode == "arabidopsis_2dt":
        return build_arabidopsis_2dt_tracking_seed(
            timeline or [],
            root_masks,
            anchor_masks,
            TwoDTTrackerConfig(
                expected_track_count=int(getattr(config, "expected_track_count", 0)),
                min_component_area=int(max(1, int(config.min_component_area))),
                bbox_padding=int(max(0, int(config.bbox_padding))),
            ),
            progress_callback=progress_callback,
        )
    return _track_auto(root_masks, config, anchor_masks=anchor_masks)


def _keep_components_touching_seed(mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    seed_u8 = (np.asarray(seed_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if mask_u8.shape != seed_u8.shape or mask_u8.size == 0:
        return mask_u8
    if np.count_nonzero(mask_u8) <= 0 or np.count_nonzero(seed_u8) <= 0:
        return mask_u8
    num_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if int(num_labels) <= 1:
        return mask_u8
    keep_labels = np.unique(labels[seed_u8 > 0])
    keep_labels = keep_labels[keep_labels > 0]
    if keep_labels.size <= 0:
        return mask_u8
    out = np.isin(labels, keep_labels).astype(np.uint8)
    return out


def _primary_root_view(
    root_mask: np.ndarray,
    lateral_mask: np.ndarray | None,
    config: AnalyticsConfig,
) -> np.ndarray:
    root_u8 = (np.asarray(root_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if not bool(getattr(config, "primary_root_tracking_excludes_lateral", True)):
        return root_u8
    if lateral_mask is None:
        return root_u8
    lateral_u8 = (np.asarray(lateral_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if lateral_u8.shape != root_u8.shape or int(np.count_nonzero(lateral_u8)) <= 0:
        return root_u8
    removable = lateral_u8.copy()
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(lateral_u8, connectivity=8)
    if int(num_labels) > 1:
        for label_id in range(1, int(num_labels)):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            if width <= 0 or height <= 0:
                continue
            comp = labels[y : y + height, x : x + width] == int(label_id)
            ys, xs = np.where(comp)
            if ys.size <= 0:
                continue
            top_y = int(np.min(ys))
            # Keep the branch attachment zone so subtracting the lateral does
            # not sever the primary backbone at the branch junction.
            keep = ys <= (top_y + 1)
            removable_crop = removable[y : y + height, x : x + width]
            removable_crop[ys[keep], xs[keep]] = 0
    primary = root_u8.copy()
    primary[removable > 0] = 0
    if int(np.count_nonzero(primary)) > 0:
        return primary
    return root_u8


def _anchor_lane_centers_from_reference_frames(
    anchor_masks: list[np.ndarray],
    shape_hw: tuple[int, int],
    top_limit: int,
    ref_frames: int,
    min_area: int,
) -> list[tuple[float, int]]:
    h, w = shape_hw
    if h <= 0 or w <= 0 or not anchor_masks:
        return []
    top = int(max(1, min(h, int(top_limit))))
    union = np.zeros((h, w), dtype=np.uint8)
    for frame_idx in range(int(max(1, ref_frames))):
        if frame_idx >= len(anchor_masks):
            break
        anchor = np.asarray(anchor_masks[frame_idx], dtype=np.uint8)
        if anchor.ndim != 2 or anchor.shape != (h, w):
            continue
        crop = (anchor[:top, :] > 0).astype(np.uint8)
        if int(np.count_nonzero(crop)) <= 0:
            continue
        union[:top, :] = np.maximum(union[:top, :], crop)
    if int(np.count_nonzero(union)) <= 0:
        return []

    union = cv2.morphologyEx(
        union,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    component_min_area = max(3, int(min_area))
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(union, connectivity=8)
    centers: list[tuple[float, int]] = []
    for label_id in range(1, int(num_labels)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < component_min_area:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0:
            continue
        # Ignore broad merged anchor sheets; stable plant anchors should be
        # compact compared with the plate width.
        if bw >= int(round(0.42 * float(w))) and area > component_min_area:
            continue
        cx = float(centroids[label_id][0]) if np.isfinite(float(centroids[label_id][0])) else float(x + (0.5 * bw))
        centers.append((float(max(0.0, min(float(w - 1), cx))), int(area)))
    centers.sort(key=lambda item: item[0])
    return centers


def _replace_lane_centers_with_anchors_when_complete(
    bbox_centers: list[tuple[str, float]],
    anchor_centers: list[tuple[float, int]],
    image_width: int,
) -> tuple[list[tuple[str, float, str, float | None]], dict[str, object]]:
    ordered = sorted([(str(track_id), float(cx)) for track_id, cx in bbox_centers], key=lambda kv: kv[1])
    meta: dict[str, object] = {
        "center_source": "bbox",
        "anchor_center_count": int(len(anchor_centers)),
        "track_count": int(len(ordered)),
    }
    if not ordered:
        return [], meta
    if len(anchor_centers) < len(ordered):
        return [(track_id, cx, "bbox", None) for track_id, cx in ordered], meta

    anchor_x = [float(cx) for cx, _area in sorted(anchor_centers, key=lambda item: item[0])]
    if len(anchor_x) > len(ordered):
        chosen: list[float] = []
        used: set[int] = set()
        for _track_id, cx in ordered:
            best_idx = -1
            best_dist = float("inf")
            for idx, ax in enumerate(anchor_x):
                if idx in used:
                    continue
                dist = abs(float(ax) - float(cx))
                if dist < best_dist:
                    best_idx = idx
                    best_dist = dist
            if best_idx < 0:
                return [(track_id, cx, "bbox", None) for track_id, cx in ordered], meta
            used.add(best_idx)
            chosen.append(float(anchor_x[best_idx]))
        anchor_x = sorted(chosen)
    else:
        anchor_x = anchor_x[: len(ordered)]

    if len(anchor_x) != len(ordered):
        return [(track_id, cx, "bbox", None) for track_id, cx in ordered], meta

    width = float(max(1, int(image_width)))
    tolerance = max(18.0, 0.28 * width)
    distances = [abs(float(ax) - float(root_cx)) for ax, (_track_id, root_cx) in zip(anchor_x, ordered)]
    if distances and max(distances) > tolerance:
        meta["status"] = "anchor_centers_too_far"
        meta["max_anchor_bbox_distance_px"] = float(max(distances))
        return [(track_id, cx, "bbox", None) for track_id, cx in ordered], meta

    meta["center_source"] = "anchor"
    meta["status"] = "anchor_centers_applied"
    meta["max_anchor_bbox_distance_px"] = float(max(distances) if distances else 0.0)
    return [
        (track_id, float(anchor_x[idx]), "anchor", float(root_cx))
        for idx, (track_id, root_cx) in enumerate(ordered)
    ], meta


def _translate_binary_mask(mask: np.ndarray, shift_x: int, shift_y: int, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    src = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    out = np.zeros((h, w), dtype=np.uint8)
    if src.shape != (h, w) or int(np.count_nonzero(src)) <= 0:
        return out
    dx = int(shift_x)
    dy = int(shift_y)
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(w, w - dx)
    src_y1 = min(h, h - dy)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out
    dst_x0 = src_x0 + dx
    dst_y0 = src_y0 + dy
    dst_x1 = src_x1 + dx
    dst_y1 = src_y1 + dy
    out[dst_y0:dst_y1, dst_x0:dst_x1] = src[src_y0:src_y1, src_x0:src_x1]
    return out


def _resolve_track_compartment_hint(
    hint: dict[str, object] | None,
    frame_index: int,
    shape_hw: tuple[int, int],
) -> dict[str, object] | None:
    if not isinstance(hint, dict):
        return None
    if bool(hint.get("_per_frame_resolved", False)):
        return hint
    frame_geometry = hint.get("per_frame")
    if not isinstance(frame_geometry, list) or not frame_geometry:
        return hint
    index = int(max(0, min(len(frame_geometry) - 1, int(frame_index))))
    geometry = frame_geometry[index]
    if not isinstance(geometry, dict):
        return hint
    h, w = shape_hw
    resolved = {key: value for key, value in hint.items() if key != "per_frame"}
    resolved.update(geometry)
    left = int(max(0, min(w, int(resolved.get("lane_left", 0)))))
    right = int(max(left + 1, min(w, int(resolved.get("lane_right", w)))))
    resolved["lane_left"] = int(left)
    resolved["lane_right"] = int(right)
    resolved["fixed_top_y"] = int(max(0, min(h - 1, int(resolved.get("fixed_top_y", 0)))))
    resolved["seed_top_limit"] = int(max(1, min(h, int(resolved.get("seed_top_limit", h)))))
    resolved["_per_frame_resolved"] = True
    return resolved


def _hint_motion_xy(hint: dict[str, object] | None) -> tuple[float, float]:
    if not isinstance(hint, dict):
        return (0.0, 0.0)
    try:
        motion_x = float(hint.get("motion_x", 0.0) or 0.0)
        motion_y = float(hint.get("motion_y", 0.0) or 0.0)
    except (TypeError, ValueError, OverflowError):
        return (0.0, 0.0)
    if not math.isfinite(motion_x) or not math.isfinite(motion_y):
        return (0.0, 0.0)
    return (motion_x, motion_y)


def _seed_mask_for_frame_source(
    source_mask: np.ndarray,
    hint: dict[str, object] | None,
    seed_key: str,
    shape_hw: tuple[int, int],
    *,
    translate_fallback: bool = True,
) -> np.ndarray:
    h, w = shape_hw
    if not isinstance(hint, dict):
        return np.zeros((h, w), dtype=np.uint8)
    base_obj = hint.get(seed_key)
    base = (
        (np.asarray(base_obj, dtype=np.uint8) > 0).astype(np.uint8)
        if isinstance(base_obj, np.ndarray) and np.asarray(base_obj).shape == (h, w)
        else np.zeros((h, w), dtype=np.uint8)
    )
    if not bool(hint.get("dynamic_lane_geometry", False)):
        return base
    if seed_key == "seed_mask" and bool(hint.get("_current_seed_mask_resolved", False)):
        return base

    src = (np.asarray(source_mask, dtype=np.uint8) > 0).astype(np.uint8)
    left = int(max(0, min(w, int(hint.get("lane_left", 0)))))
    right = int(max(left + 1, min(w, int(hint.get("lane_right", w)))))
    top = int(max(1, min(h, int(hint.get("seed_top_limit", h)))))
    direct = np.zeros((h, w), dtype=np.uint8)
    if src.shape == (h, w):
        direct[:top, left:right] = src[:top, left:right]
    if int(np.count_nonzero(direct)) > 0:
        return direct
    if not bool(translate_fallback):
        return direct
    return _translate_binary_mask(
        base,
        int(round(float(hint.get("motion_x", 0.0)))),
        int(round(float(hint.get("motion_y", 0.0)))),
        shape_hw,
    )


def _build_track_compartment_hints(
    root_masks: list[np.ndarray],
    anchor_masks: list[np.ndarray],
    track_ids: list[str],
    track_bboxes: dict[str, list[BBox]],
    seed_frame: int,
    config: AnalyticsConfig,
) -> dict[str, dict[str, object]]:
    if not root_masks or not track_ids:
        return {}
    if not bool(getattr(config, "track_lane_partition_enabled", True)):
        return {}
    first = np.asarray(root_masks[0], dtype=np.uint8)
    if first.ndim != 2 or first.size == 0:
        return {}
    h, w = first.shape[:2]

    def _nearest_valid_bbox(series: object) -> tuple[BBox, int] | None:
        if not isinstance(series, list) or not series:
            return None
        candidates: list[tuple[int, BBox]] = []
        for frame_idx, raw_bbox in enumerate(series):
            if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
                continue
            bbox = _clip_bbox((int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3])), (h, w))
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            candidates.append((int(frame_idx), bbox))
        if not candidates:
            return None
        if 0 <= int(seed_frame) < len(series):
            for frame_idx, bbox in candidates:
                if int(frame_idx) == int(seed_frame):
                    return bbox, int(frame_idx)
            best_frame, best_bbox = min(candidates, key=lambda item: (abs(int(item[0]) - int(seed_frame)), int(item[0])))
            return best_bbox, int(best_frame)
        best_frame, best_bbox = min(candidates, key=lambda item: int(item[0]))
        return best_bbox, int(best_frame)

    centers: list[tuple[str, float]] = []
    reference_bboxes: dict[str, BBox] = {}
    reference_frames: dict[str, int] = {}
    for track_id in track_ids:
        resolved = _nearest_valid_bbox(track_bboxes.get(track_id))
        if resolved is None:
            continue
        bbox, reference_frame = resolved
        reference_bboxes[str(track_id)] = bbox
        reference_frames[str(track_id)] = int(reference_frame)
        centers.append((track_id, float(bbox[0] + (0.5 * bbox[2]))))
    if not centers:
        return {}
    centers.sort(key=lambda kv: kv[1])
    pad = int(max(0, min(256, int(getattr(config, "track_lane_padding_px", 0)))))

    crown_zone, top_limit, _status = _build_crown_zone_from_anchor_masks(anchor_masks, (h, w), config)
    if np.count_nonzero(crown_zone) <= 0:
        crown_zone = np.zeros((h, w), dtype=np.uint8)
        top_limit = int(min(h, max(1, int(round(0.35 * float(h))))))
        crown_zone[:top_limit, :] = 1

    ref_frames = int(max(1, min(len(root_masks), int(getattr(config, "shoot_crown_lock_ref_frames", 3)))))
    lock_start = int(max(0, min(len(root_masks) - 1, int(getattr(config, "shoot_crown_lock_start_frame", 7)))))
    anchor_lane_centers = _anchor_lane_centers_from_reference_frames(
        anchor_masks,
        (h, w),
        int(top_limit),
        int(ref_frames),
        max(3, int(getattr(config, "min_component_area", 30)) // 3),
    )
    dynamic_crown_lanes = bool(
        str(getattr(config, "tracking_mode", "auto")) == "arabidopsis_crown_lanes"
        and len(centers) > 1
    )
    if dynamic_crown_lanes:
        resolved_centers = [(track_id, cx, "crown_lane", None) for track_id, cx in centers]
        center_meta = {
            "center_source": "crown_lane",
            "anchor_center_count": int(len(anchor_lane_centers)),
            "track_count": int(len(centers)),
            "status": "per_frame_crown_lanes",
        }
    else:
        resolved_centers, center_meta = _replace_lane_centers_with_anchors_when_complete(centers, anchor_lane_centers, w)
    centers = [(track_id, cx) for track_id, cx, _source, _root_cx in resolved_centers]
    center_sources = {track_id: source for track_id, _cx, source, _root_cx in resolved_centers}
    root_center_lookup = {track_id: root_cx for track_id, _cx, _source, root_cx in resolved_centers}
    hints: dict[str, dict[str, object]] = {}

    x_bounds: dict[str, tuple[int, int]] = {}
    for idx, (track_id, cx) in enumerate(centers):
        left = 0 if idx == 0 else int(round((centers[idx - 1][1] + cx) * 0.5))
        right = w if idx == len(centers) - 1 else int(round((cx + centers[idx + 1][1]) * 0.5))
        left = max(0, left - pad)
        right = min(w, right + pad)
        if right <= left:
            right = min(w, left + 1)
        x_bounds[track_id] = (int(left), int(right))

    per_frame_geometry: dict[str, list[dict[str, object]]] = {
        str(track_id): [] for track_id, _cx in centers
    }
    if dynamic_crown_lanes:
        reference_y = {
            str(track_id): float(bbox[1] + (0.5 * bbox[3]))
            for track_id, bbox in reference_bboxes.items()
        }
        center_lookup = {str(track_id): float(cx) for track_id, cx in centers}
        for frame_idx in range(len(root_masks)):
            frame_centers: list[tuple[str, float, float, BBox]] = []
            for track_id, static_cx in centers:
                bbox = reference_bboxes.get(str(track_id), (0, 0, 0, 0))
                series = track_bboxes.get(track_id)
                if isinstance(series, list) and frame_idx < len(series):
                    raw_bbox = series[frame_idx]
                    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                        candidate = _clip_bbox(
                            (int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3])),
                            (h, w),
                        )
                        if candidate[2] > 0 and candidate[3] > 0:
                            bbox = candidate
                cx = float(bbox[0] + (0.5 * bbox[2])) if bbox[2] > 0 else float(static_cx)
                cy = (
                    float(bbox[1] + (0.5 * bbox[3]))
                    if bbox[3] > 0
                    else float(reference_y.get(str(track_id), 0.0))
                )
                frame_centers.append((str(track_id), cx, cy, bbox))

            frame_bounds: dict[str, tuple[int, int]] = {}
            for idx, (track_id, cx, _cy, _bbox) in enumerate(frame_centers):
                left = 0 if idx == 0 else int(round(0.5 * float(frame_centers[idx - 1][1] + cx)))
                right = w if idx == len(frame_centers) - 1 else int(round(0.5 * float(cx + frame_centers[idx + 1][1])))
                left = max(0, left - pad)
                right = min(w, right + pad)
                if right <= left:
                    right = min(w, left + 1)
                frame_bounds[track_id] = (int(left), int(right))

            root = np.asarray(root_masks[frame_idx], dtype=np.uint8)
            for track_id, cx, cy, bbox in frame_centers:
                left, right = frame_bounds[track_id]
                detected_top: int | None = None
                if root.ndim == 2 and root.shape == (h, w):
                    rows = np.where(root[:top_limit, left:right] > 0)[0]
                    if rows.size > 0:
                        detected_top = int(np.min(rows))
                per_frame_geometry[track_id].append(
                    {
                        "frame_index": int(frame_idx),
                        "lane_left": int(left),
                        "lane_right": int(right),
                        "lane_center_x": float(cx),
                        "crown_center_x": float(cx),
                        "crown_center_y": float(cy),
                        "motion_x": float(cx - center_lookup[track_id]),
                        "motion_y": float(cy - float(reference_y.get(track_id, cy))),
                        "bbox": tuple(int(value) for value in bbox),
                        "_detected_fixed_top_y": detected_top,
                        "dynamic_lane_geometry": True,
                    }
                )

    for track_id, _cx in centers:
        left, right = x_bounds[track_id]
        seed_mask = np.zeros((h, w), dtype=np.uint8)
        shoot_seed_mask = np.zeros((h, w), dtype=np.uint8)
        for frame_idx in range(ref_frames):
            rm = np.asarray(root_masks[frame_idx], dtype=np.uint8)
            if rm.ndim != 2 or rm.shape != (h, w):
                continue
            sample = np.zeros((h, w), dtype=np.uint8)
            sample[:top_limit, left:right] = rm[:top_limit, left:right]
            if np.count_nonzero(crown_zone) > 0:
                sample = (sample > 0).astype(np.uint8) * crown_zone
            seed_mask = np.maximum(seed_mask, sample.astype(np.uint8))
            if frame_idx < len(anchor_masks):
                anchor = np.asarray(anchor_masks[frame_idx], dtype=np.uint8)
                if anchor.ndim == 2 and anchor.shape == (h, w):
                    shoot_sample = np.zeros((h, w), dtype=np.uint8)
                    shoot_sample[:top_limit, left:right] = (anchor[:top_limit, left:right] > 0).astype(np.uint8)
                    shoot_seed_mask = np.maximum(shoot_seed_mask, shoot_sample.astype(np.uint8))
        if np.count_nonzero(seed_mask) <= 0:
            bbox = reference_bboxes.get(str(track_id), (0, 0, 0, 0))
            x, y, bw, bh = bbox
            if bw > 0 and bh > 0:
                seed_mask[y : min(h, y + bh), max(0, left) : min(w, right)] = 1
        if np.count_nonzero(shoot_seed_mask) <= 0:
            shoot_seed_mask = seed_mask.copy()
        ys = np.where(seed_mask > 0)[0]
        fixed_top_y = int(np.min(ys)) if ys.size > 0 else 0
        frame_geometry = per_frame_geometry.get(str(track_id), []) if dynamic_crown_lanes else []
        for geometry in frame_geometry:
            detected_top = geometry.pop("_detected_fixed_top_y", None)
            if isinstance(detected_top, int):
                frame_top = int(detected_top)
            else:
                frame_top = int(round(float(fixed_top_y) + float(geometry.get("motion_y", 0.0))))
            geometry["fixed_top_y"] = int(max(0, min(h - 1, frame_top)))
            geometry["seed_top_limit"] = int(top_limit)
        reference_bbox_for_track = reference_bboxes.get(
            str(track_id),
            (0, fixed_top_y, 0, 0),
        )
        hints[track_id] = {
            "lane_left": int(left),
            "lane_right": int(right),
            "seed_mask": seed_mask.astype(np.uint8),
            "shoot_seed_mask": shoot_seed_mask.astype(np.uint8),
            "fixed_top_y": int(max(0, min(h - 1, fixed_top_y))),
            "lock_start_frame": int(lock_start),
            "lane_center_x": float(_cx),
            "crown_center_x": float(_cx),
            "crown_center_y": float(
                reference_bbox_for_track[1]
                + (0.5 * reference_bbox_for_track[3])
            ),
            "lane_center_source": str(center_sources.get(track_id, "bbox")),
            "root_seed_center_x": root_center_lookup.get(track_id),
            "anchor_lane_center_count": int(center_meta.get("anchor_center_count", 0) or 0),
            "lane_reference_frame": int(reference_frames.get(str(track_id), seed_frame)),
            "lane_reference_source": "seed_frame" if int(reference_frames.get(str(track_id), seed_frame)) == int(seed_frame) else "nearest_valid_bbox",
            "dynamic_lane_geometry": bool(dynamic_crown_lanes),
            "per_frame": frame_geometry,
        }
    return hints


def _extract_track_mask_from_hint(
    frame_mask: np.ndarray,
    hint: dict[str, object] | None,
    frame_index: int,
    shape_hw: tuple[int, int],
    config: AnalyticsConfig,
) -> tuple[np.ndarray, BBox]:
    h, w = shape_hw
    src = (np.asarray(frame_mask, dtype=np.uint8) > 0).astype(np.uint8)
    hint = _resolve_track_compartment_hint(hint, frame_index, shape_hw)
    if src.ndim != 2 or src.shape != (h, w) or not isinstance(hint, dict):
        ys, xs = np.where(src > 0)
        if ys.size <= 0 or xs.size <= 0:
            return np.zeros((h, w), dtype=np.uint8), (0, 0, 0, 0)
        x0 = int(np.min(xs))
        y0 = int(np.min(ys))
        x1 = int(np.max(xs) + 1)
        y1 = int(np.max(ys) + 1)
        return src, _clip_bbox((x0, y0, x1 - x0, y1 - y0), shape_hw)

    left = int(max(0, min(w, int(hint.get("lane_left", 0)))))
    right = int(max(left + 1, min(w, int(hint.get("lane_right", w)))))
    seed_mask = _seed_mask_for_frame_source(src, hint, "seed_mask", shape_hw)
    fixed_top_y = int(max(0, min(h - 1, int(hint.get("fixed_top_y", 0)))))
    lock_start = int(max(0, int(hint.get("lock_start_frame", 0))))

    lane_mask = np.zeros((h, w), dtype=np.uint8)
    lane_mask[:, left:right] = src[:, left:right]
    if bool(getattr(config, "track_seed_connectivity_enabled", True)):
        connected = _keep_components_touching_seed(lane_mask, seed_mask)
    else:
        connected = lane_mask
    ys, xs = np.where(connected > 0)
    if ys.size <= 0 or xs.size <= 0:
        ys, xs = np.where(lane_mask > 0)
        if ys.size <= 0 or xs.size <= 0:
            return np.zeros((h, w), dtype=np.uint8), (0, 0, 0, 0)
        connected = lane_mask

    x0 = int(np.min(xs))
    y0 = int(np.min(ys))
    x1 = int(np.max(xs) + 1)
    y1 = int(np.max(ys) + 1)
    padding = int(max(0, min(256, int(config.bbox_padding))))
    x0 = max(left, x0 - padding)
    x1 = min(right, x1 + padding)
    y1 = min(h, y1 + padding)
    if frame_index >= lock_start:
        y0 = max(0, min(y0, h - 1))
        y0 = max(fixed_top_y, 0)
    else:
        y0 = max(0, y0)
    bbox = _clip_bbox((x0, y0, max(1, x1 - x0), max(1, y1 - y0)), shape_hw)
    return connected.astype(np.uint8), bbox


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    m = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if np.count_nonzero(m) == 0:
        return m.astype(bool)
    try:
        from skimage.morphology import skeletonize  # type: ignore

        return skeletonize(m > 0)
    except Exception:
        image = (m * 255).astype(np.uint8)
        skel = np.zeros_like(image, dtype=np.uint8)
        elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(image, elem)
            opened = cv2.dilate(eroded, elem)
            temp = cv2.subtract(image, opened)
            skel = cv2.bitwise_or(skel, temp)
            image = eroded
            if cv2.countNonZero(image) == 0:
                break
        return skel > 0


def _prune_small_skeleton_components(skeleton: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return skeleton
    bin_mask = (np.asarray(skeleton, dtype=np.uint8) > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    keep = np.zeros_like(bin_mask, dtype=np.uint8)
    for label_id in range(1, int(num_labels)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= int(min_size):
            keep[labels == label_id] = 1
    return keep > 0


def _skeleton_graph_path(
    skeleton: np.ndarray,
    base_hint_xy: tuple[float, float] | None = None,
) -> tuple[list[Point], int, int]:
    skeleton_u8 = (np.asarray(skeleton, dtype=np.uint8) > 0).astype(np.uint8)
    coords = np.argwhere(skeleton_u8 > 0)
    if coords.size == 0:
        return [], 0, 0

    hint_x = float("nan")
    hint_y = float("nan")
    if base_hint_xy is not None:
        try:
            hint_x = float(base_hint_xy[0])
            hint_y = float(base_hint_xy[1])
        except (TypeError, ValueError, IndexError):
            hint_x = float("nan")
            hint_y = float("nan")
        if math.isfinite(hint_x) and math.isfinite(hint_y):
            component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
                skeleton_u8,
                connectivity=8,
            )
            if int(component_count) > 2:
                component_scores: list[tuple[float, int]] = []
                for component_id in range(1, int(component_count)):
                    component_coords = np.argwhere(component_labels == component_id)
                    if component_coords.size <= 0:
                        continue
                    rows = component_coords[:, 0].astype(np.float64)
                    cols = component_coords[:, 1].astype(np.float64)
                    vertical_span = float(np.max(rows) - np.min(rows) + 1.0)
                    pixel_count = float(component_stats[component_id, cv2.CC_STAT_AREA])
                    distance = float(
                        np.min(np.hypot(cols - hint_x, rows - hint_y))
                    )
                    capped_support = min(pixel_count, 4.0 * vertical_span)
                    score = (
                        (2.0 * vertical_span)
                        + (0.15 * capped_support)
                        - (1.5 * distance)
                    )
                    component_scores.append((float(score), int(component_id)))
                if component_scores:
                    selected_component = max(
                        component_scores,
                        key=lambda item: (float(item[0]), -int(item[1])),
                    )[1]
                    coords = np.argwhere(component_labels == int(selected_component))

    points = [(int(r), int(c)) for r, c in coords]
    point_set = set(points)
    neighbors: dict[Point, list[Point]] = {}
    tips = 0
    branches = 0
    for r, c in points:
        local: list[Point] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                q = (r + dr, c + dc)
                if q in point_set:
                    local.append(q)
        neighbors[(r, c)] = local
        deg = len(local)
        if deg == 1:
            tips += 1
        elif deg >= 3:
            branches += 1
    endpoint_nodes = [point for point in points if len(neighbors.get(point, [])) == 1]
    base_candidates = endpoint_nodes if endpoint_nodes else points
    if math.isfinite(hint_x) and math.isfinite(hint_y):
        base = min(
            base_candidates,
            key=lambda point: (
                math.hypot(float(point[1]) - hint_x, float(point[0]) - hint_y),
                int(point[0]),
                int(point[1]),
            ),
        )
    else:
        base = min(base_candidates, key=lambda p: (p[0], p[1]))
    directions = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    direction_index = {direction: idx for idx, direction in enumerate(directions)}
    start_state = (base, -1)
    state_cost: dict[tuple[Point, int], float] = {start_state: 0.0}
    state_length: dict[tuple[Point, int], float] = {start_state: 0.0}
    parent: dict[tuple[Point, int], tuple[Point, int]] = {}
    best_state: dict[Point, tuple[Point, int]] = {base: start_state}
    heap: list[tuple[float, int, int, int]] = [(0.0, int(base[0]), int(base[1]), -1)]
    while heap:
        current_cost, row, col, previous_direction = heapq.heappop(heap)
        point = (int(row), int(col))
        state = (point, int(previous_direction))
        if current_cost > float(state_cost.get(state, float("inf"))) + 1.0e-9:
            continue
        for nxt in neighbors[point]:
            dr = int(nxt[0] - point[0])
            dc = int(nxt[1] - point[1])
            next_direction = int(direction_index[(dr, dc)])
            step_length = math.sqrt(2.0) if dr != 0 and dc != 0 else 1.0
            turn_penalty = 0.0
            if previous_direction >= 0:
                pdr, pdc = directions[int(previous_direction)]
                cosine = float((pdr * dr) + (pdc * dc)) / float(
                    max(1.0e-9, math.hypot(float(pdr), float(pdc)) * math.hypot(float(dr), float(dc)))
                )
                turn_penalty = 1.65 * float(max(0.0, 1.0 - max(-1.0, min(1.0, cosine))))
            next_state = (nxt, next_direction)
            next_cost = float(current_cost + step_length + turn_penalty)
            if next_cost + 1.0e-9 >= float(state_cost.get(next_state, float("inf"))):
                continue
            state_cost[next_state] = next_cost
            state_length[next_state] = float(state_length.get(state, 0.0) + step_length)
            parent[next_state] = state
            heapq.heappush(heap, (next_cost, int(nxt[0]), int(nxt[1]), next_direction))
            current_best = best_state.get(nxt)
            if current_best is None or next_cost < float(state_cost.get(current_best, float("inf"))):
                best_state[nxt] = next_state
    if not best_state:
        return [], tips, branches
    tips_nodes = [p for p in best_state.keys() if len(neighbors[p]) == 1 and p != base]

    def _endpoint_score(point: Point) -> float:
        state = best_state[point]
        vertical_gain = max(0, int(point[0]) - int(base[0]))
        horizontal_drift = abs(int(point[1]) - int(base[1]))
        path_length = float(state_length.get(state, 0.0))
        curvature = max(0.0, float(state_cost.get(state, path_length)) - path_length)
        return (
            (1.55 * float(vertical_gain))
            + (0.30 * path_length)
            - (0.42 * float(horizontal_drift))
            - (0.90 * curvature)
        )

    endpoint = max(tips_nodes if tips_nodes else list(best_state.keys()), key=_endpoint_score)
    state_path = [best_state[endpoint]]
    while state_path[-1] != start_state:
        previous = parent.get(state_path[-1])
        if previous is None:
            break
        state_path.append(previous)
    state_path.reverse()
    path_rc: list[Point] = []
    for point, _direction in state_path:
        if not path_rc or point != path_rc[-1]:
            path_rc.append(point)
    path_xy = [(int(c), int(r)) for r, c in path_rc]
    return path_xy, tips, branches


def _path_length_px(path_xy: list[Point]) -> float:
    if len(path_xy) < 2:
        return float(len(path_xy))
    acc = 0.0
    for i in range(1, len(path_xy)):
        x0, y0 = path_xy[i - 1]
        x1, y1 = path_xy[i]
        acc += float(np.hypot(x1 - x0, y1 - y0))
    return acc


def _skeleton_edge_length_px(skeleton: np.ndarray) -> float:
    skel = np.asarray(skeleton, dtype=bool)
    pixels = int(np.count_nonzero(skel))
    if pixels <= 1:
        return float(pixels)
    horizontal = int(np.count_nonzero(skel[:, :-1] & skel[:, 1:]))
    vertical = int(np.count_nonzero(skel[:-1, :] & skel[1:, :]))
    diagonal_down = int(np.count_nonzero(skel[:-1, :-1] & skel[1:, 1:]))
    diagonal_up = int(np.count_nonzero(skel[1:, :-1] & skel[:-1, 1:]))
    return float(horizontal + vertical + math.sqrt(2.0) * (diagonal_down + diagonal_up))


def _skeleton_edge_length_by_label(
    skeleton: np.ndarray,
    label_map: np.ndarray,
    label_count: int,
) -> dict[int, float]:
    """Allocate every skeleton edge exactly once across positive labels."""
    skel = np.asarray(skeleton, dtype=bool)
    labels = np.asarray(label_map, dtype=np.int32)
    count = int(max(0, label_count))
    totals = np.zeros(count + 1, dtype=np.float64)
    if skel.shape != labels.shape or count <= 0:
        return {label: 0.0 for label in range(1, count + 1)}
    pixel_count = int(np.count_nonzero(skel))
    if pixel_count == 1:
        label = int(labels[skel][0])
        if 1 <= label <= count:
            totals[label] = 1.0
        return {label: float(totals[label]) for label in range(1, count + 1)}

    def _accumulate(left: np.ndarray, right: np.ndarray, edge_mask: np.ndarray, weight: float) -> None:
        if not bool(np.any(edge_mask)):
            return
        left_labels = left[edge_mask].astype(np.int32, copy=False)
        right_labels = right[edge_mask].astype(np.int32, copy=False)
        valid = (
            (left_labels > 0)
            & (left_labels <= count)
            & (right_labels > 0)
            & (right_labels <= count)
        )
        if not bool(np.any(valid)):
            return
        left_labels = left_labels[valid]
        right_labels = right_labels[valid]
        same = left_labels == right_labels
        if bool(np.any(same)):
            totals[:] += np.bincount(left_labels[same], minlength=count + 1)[: count + 1] * float(weight)
        different = ~same
        if bool(np.any(different)):
            half_weight = 0.5 * float(weight)
            totals[:] += np.bincount(left_labels[different], minlength=count + 1)[: count + 1] * half_weight
            totals[:] += np.bincount(right_labels[different], minlength=count + 1)[: count + 1] * half_weight

    _accumulate(labels[:, :-1], labels[:, 1:], skel[:, :-1] & skel[:, 1:], 1.0)
    _accumulate(labels[:-1, :], labels[1:, :], skel[:-1, :] & skel[1:, :], 1.0)
    diagonal_weight = math.sqrt(2.0)
    _accumulate(labels[:-1, :-1], labels[1:, 1:], skel[:-1, :-1] & skel[1:, 1:], diagonal_weight)
    _accumulate(labels[1:, :-1], labels[:-1, 1:], skel[1:, :-1] & skel[:-1, 1:], diagonal_weight)
    return {label: float(totals[label]) for label in range(1, count + 1)}


def _point_at_distance(path_xy: list[Point], distance_px: float) -> Point | None:
    if not path_xy:
        return None
    if distance_px <= 0:
        return path_xy[0]
    acc = 0.0
    for i in range(1, len(path_xy)):
        x0, y0 = path_xy[i - 1]
        x1, y1 = path_xy[i]
        seg = float(np.hypot(x1 - x0, y1 - y0))
        if acc + seg >= distance_px and seg > 1e-9:
            t = (distance_px - acc) / seg
            return (int(round(x0 + t * (x1 - x0))), int(round(y0 + t * (y1 - y0))))
        acc += seg
    return path_xy[-1]


def _angle_to_vertical(base: Point | None, tip: Point | None) -> float:
    if base is None or tip is None:
        return 0.0
    dx = float(tip[0] - base[0])
    dy = float(tip[1] - base[1])
    norm = float(np.hypot(dx, dy))
    if norm <= 1e-9:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dy / norm))
    return float(np.degrees(np.arccos(cos_theta)))


def _angular_difference_deg(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 180.0
    if not (math.isfinite(float(a)) and math.isfinite(float(b))):
        return 180.0
    diff = abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)
    return float(diff)


def _extract_tip_candidates(skeleton: np.ndarray, path_xy: list[Point]) -> list[dict[str, object]]:
    coords = np.argwhere(skeleton > 0)
    if coords.size == 0:
        return []

    points = [(int(r), int(c)) for r, c in coords]
    point_set = set(points)
    neighbors: dict[Point, list[Point]] = {}
    for r, c in points:
        local: list[Point] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                q = (r + dr, c + dc)
                if q in point_set:
                    local.append(q)
        neighbors[(r, c)] = local

    endpoints_rc: list[Point] = [p for p, local in neighbors.items() if len(local) == 1]
    if not endpoints_rc:
        return []

    base_xy = path_xy[0] if path_xy else None
    out: list[dict[str, object]] = []
    for r, c in endpoints_rc:
        nbr = neighbors[(r, c)][0]
        nr, nc = int(nbr[0]), int(nbr[1])
        dx = float(c - nc)
        dy = float(r - nr)
        angle_deg = float(np.degrees(np.arctan2(dy, dx))) if (abs(dx) + abs(dy)) > 1e-9 else 0.0
        x = int(c)
        y = int(r)
        if base_xy is None:
            dist_base = float("inf")
        else:
            dist_base = float(np.hypot(float(x - base_xy[0]), float(y - base_xy[1])))
        out.append(
            {
                "x": int(x),
                "y": int(y),
                "angle_deg": float(angle_deg),
                "distance_from_base_px": float(dist_base),
                "is_base_endpoint": bool(dist_base <= 2.0),
            }
        )

    # Remove basal endpoint when other distal tips exist.
    distal = [tip for tip in out if not bool(tip.get("is_base_endpoint", False))]
    if distal:
        out = distal
    out.sort(key=lambda tip: (int(tip.get("y", 0)), int(tip.get("x", 0))))
    return out


def _measure_crop(
    mask_crop: np.ndarray,
    config: AnalyticsConfig,
    *,
    base_hint_xy: tuple[float, float] | None = None,
) -> dict[str, object]:
    bin_mask = (np.asarray(mask_crop, dtype=np.uint8) > 0).astype(np.uint8)
    area_px = int(np.count_nonzero(bin_mask))
    if area_px <= 0:
        return {
            "length_px": 0.0,
            "skeleton_length_px": 0.0,
            "weighted_length_px": 0.0,
            "area_px": 0,
            "perimeter_px": 0.0,
            "tips": 0,
            "branches": 0,
            "path_xy": [],
            "base_tip_angle_deg": 0.0,
            "emergence_angle_deg": 0.0,
            "convex_hull_area_px2": 0.0,
            "convex_hull_width_px": 0.0,
            "convex_hull_height_px": 0.0,
            "aspect_ratio": 0.0,
            "tip_candidates": [],
        }
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter_px = float(sum(cv2.arcLength(c, True) for c in contours))
    skeleton = _prune_small_skeleton_components(_skeletonize(bin_mask), config.prune_branch_px)
    path_xy, tips, branches = _skeleton_graph_path(
        skeleton,
        base_hint_xy=base_hint_xy,
    )
    skeleton_length_px = float(np.count_nonzero(skeleton))
    weighted_length_px = _skeleton_edge_length_px(skeleton)
    length_px = _path_length_px(path_xy)
    if length_px <= 0.0:
        length_px = float(np.count_nonzero(skeleton))
    if length_px <= 0.0 and perimeter_px > 0:
        length_px = float((2.0 * area_px) / perimeter_px)

    pts = np.column_stack(np.nonzero(bin_mask))
    if pts.shape[0] >= 3:
        pts_xy = pts[:, ::-1].astype(np.int32)
        hull = cv2.convexHull(pts_xy)
        hull_area_px2 = float(cv2.contourArea(hull))
        x, y, w, h = cv2.boundingRect(hull)
        hull_w = float(w)
        hull_h = float(h)
    else:
        hull_area_px2 = float(area_px)
        ys, xs = np.nonzero(bin_mask)
        hull_w = float(xs.max() - xs.min() + 1) if xs.size else 0.0
        hull_h = float(ys.max() - ys.min() + 1) if ys.size else 0.0
    aspect_ratio = float(hull_h / hull_w) if hull_w > 1e-9 else 0.0

    base = path_xy[0] if path_xy else None
    tip = path_xy[-1] if path_xy else None
    emergence_point = _point_at_distance(path_xy, float(config.emergence_distance_mm / max(config.pixel_size_mm, 1e-6)))
    base_tip_angle = _angle_to_vertical(base, tip)
    emergence_angle = _angle_to_vertical(base, emergence_point)
    tip_candidates = _extract_tip_candidates(skeleton, path_xy)
    return {
        "length_px": float(length_px),
        "skeleton_length_px": float(skeleton_length_px),
        "weighted_length_px": float(weighted_length_px),
        "area_px": int(area_px),
        "perimeter_px": float(perimeter_px),
        "tips": int(tips),
        "branches": int(branches),
        "path_xy": path_xy,
        "base_tip_angle_deg": float(base_tip_angle),
        "emergence_angle_deg": float(emergence_angle),
        "convex_hull_area_px2": float(hull_area_px2),
        "convex_hull_width_px": float(hull_w),
        "convex_hull_height_px": float(hull_h),
        "aspect_ratio": float(aspect_ratio),
        "tip_candidates": tip_candidates,
    }


def _derive_lateral_mask_from_root(root_crop: np.ndarray, root_metrics: dict[str, object]) -> np.ndarray:
    """Approximate lateral-root mask when a dedicated lateral class is unavailable.

    The primary path is estimated from the root skeleton path and removed from
    the full root mask. Remaining connected structures are treated as candidate
    lateral roots.
    """
    root_bin = (np.asarray(root_crop, dtype=np.uint8) > 0).astype(np.uint8)
    if root_bin.size == 0 or int(np.count_nonzero(root_bin)) <= 0:
        return np.zeros_like(root_bin, dtype=np.uint8)

    path_xy = root_metrics.get("path_xy")
    if not isinstance(path_xy, list) or not path_xy:
        return np.zeros_like(root_bin, dtype=np.uint8)

    try:
        root_area = float(root_metrics.get("area_px", 0.0))
        root_length = float(root_metrics.get("length_px", 0.0))
    except Exception:
        root_area = 0.0
        root_length = 0.0
    avg_diameter_px = (root_area / root_length) if root_length > 1e-6 else 1.0
    main_thickness = max(1, int(round(max(1.0, avg_diameter_px * 1.35))))

    main_mask = np.zeros_like(root_bin, dtype=np.uint8)
    points = np.asarray(path_xy, dtype=np.int32).reshape(-1, 1, 2)
    if points.shape[0] == 1:
        cx, cy = int(points[0, 0, 0]), int(points[0, 0, 1])
        cv2.circle(main_mask, (cx, cy), max(1, main_thickness // 2), 1, -1)
    else:
        cv2.polylines(main_mask, [points], False, 1, max(1, main_thickness), cv2.LINE_AA)
    main_mask = cv2.dilate(main_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

    lateral = root_bin.copy()
    lateral[main_mask > 0] = 0
    return lateral.astype(np.uint8, copy=False)


def _measure_lateral_segments(
    lateral_crop: np.ndarray,
    config: AnalyticsConfig,
    skeleton_crop: np.ndarray | None = None,
) -> list[dict[str, object]]:
    mask = (np.asarray(lateral_crop, dtype=np.uint8) > 0).astype(np.uint8)
    if mask.size == 0 or int(np.count_nonzero(mask)) <= 0:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(3, int(config.min_component_area // 4))
    segments: list[dict[str, object]] = []

    for label_id in range(1, int(num_labels)):
        area_px = int(stats[label_id, cv2.CC_STAT_AREA])
        if area_px < min_area:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if width <= 0 or height <= 0:
            continue
        component_roi = (labels[y : y + height, x : x + width] == label_id).astype(np.uint8)
        component = np.pad(component_roi, 1, mode="constant", constant_values=0)
        metrics = _measure_crop(component, config)
        path_length_px = float(metrics.get("length_px", 0.0))
        supplied_skeleton = None
        if isinstance(skeleton_crop, np.ndarray) and skeleton_crop.shape == mask.shape:
            supplied_skeleton_roi = (
                (np.asarray(skeleton_crop[y : y + height, x : x + width], dtype=np.uint8) > 0)
                & (component_roi > 0)
            ).astype(np.uint8)
            supplied_skeleton = np.pad(supplied_skeleton_roi, 1, mode="constant", constant_values=0)
        length_px = (
            float(np.count_nonzero(supplied_skeleton))
            if supplied_skeleton is not None
            else float(metrics.get("skeleton_length_px", path_length_px))
        )
        if length_px <= 0.0 and supplied_skeleton is None:
            continue
        weighted_length_px = (
            float(_skeleton_edge_length_px(supplied_skeleton))
            if supplied_skeleton is not None
            else float(metrics.get("weighted_length_px", length_px))
        )
        diameter_px = float(area_px) / max(length_px, 1e-6)
        segments.append(
            {
                "segment_id": int(label_id),
                "length_px": float(length_px),
                "length_mm": float(length_px * config.pixel_size_mm),
                "path_length_px": float(path_length_px),
                "path_length_mm": float(path_length_px * config.pixel_size_mm),
                "weighted_length_px": float(weighted_length_px),
                "weighted_length_mm": float(weighted_length_px * config.pixel_size_mm),
                "diameter_px": float(diameter_px),
                "diameter_mm": float(diameter_px * config.pixel_size_mm),
                "area_px": int(area_px),
                "area_mm2": float(float(area_px) * (float(config.pixel_size_mm) ** 2)),
                "tips": int(metrics.get("tips", 0)),
                "branches": int(metrics.get("branches", 0)),
            }
        )

    segments.sort(key=lambda seg: float(seg.get("length_mm", 0.0)), reverse=True)
    return segments


def _mask_bbox(mask: np.ndarray, padding: int, shape_hw: tuple[int, int]) -> BBox:
    bin_mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if bin_mask.size <= 0 or int(np.count_nonzero(bin_mask)) <= 0:
        return (0, 0, 0, 0)
    x, y, width, height = cv2.boundingRect(bin_mask)
    if width <= 0 or height <= 0:
        return (0, 0, 0, 0)
    return _expand_bbox(
        (int(x), int(y), int(width), int(height)),
        int(max(0, padding)),
        shape_hw,
    )


def _pack_binary_mask(mask: np.ndarray) -> dict[str, object]:
    mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if mask_u8.size == 0 or int(np.count_nonzero(mask_u8)) <= 0:
        return {"bbox": (0, 0, 0, 0), "shape": (0, 0), "bits": np.zeros((0, 0), dtype=np.uint8)}
    x, y, w, h = _mask_bbox(mask_u8, 0, mask_u8.shape[:2])
    if w <= 0 or h <= 0:
        return {"bbox": (0, 0, 0, 0), "shape": (0, 0), "bits": np.zeros((0, 0), dtype=np.uint8)}
    crop = mask_u8[y : y + h, x : x + w]
    return {
        "bbox": (int(x), int(y), int(w), int(h)),
        "shape": (int(h), int(w)),
        "bits": np.packbits(crop, axis=1),
    }


def _unpack_binary_mask(packed: object, shape_hw: tuple[int, int]) -> np.ndarray:
    if isinstance(packed, np.ndarray):
        mask = (np.asarray(packed, dtype=np.uint8) > 0).astype(np.uint8)
        if mask.shape == shape_hw:
            return mask
        out = np.zeros(shape_hw, dtype=np.uint8)
        h = min(shape_hw[0], mask.shape[0])
        w = min(shape_hw[1], mask.shape[1])
        out[:h, :w] = mask[:h, :w]
        return out
    if not isinstance(packed, dict):
        return np.zeros(shape_hw, dtype=np.uint8)
    bbox_raw = packed.get("bbox", (0, 0, 0, 0))
    shape_raw = packed.get("shape", (0, 0))
    bits = packed.get("bits")
    try:
        x, y, w, h = (int(bbox_raw[0]), int(bbox_raw[1]), int(bbox_raw[2]), int(bbox_raw[3]))
        ch, cw = (int(shape_raw[0]), int(shape_raw[1]))
    except Exception:
        return np.zeros(shape_hw, dtype=np.uint8)
    if w <= 0 or h <= 0 or ch <= 0 or cw <= 0 or not isinstance(bits, np.ndarray):
        return np.zeros(shape_hw, dtype=np.uint8)
    crop = np.unpackbits(np.asarray(bits, dtype=np.uint8), axis=1, count=int(cw))[: int(ch), : int(cw)]
    out = np.zeros(shape_hw, dtype=np.uint8)
    x = max(0, min(shape_hw[1], x))
    y = max(0, min(shape_hw[0], y))
    x1 = max(x, min(shape_hw[1], x + int(cw)))
    y1 = max(y, min(shape_hw[0], y + int(ch)))
    if x1 <= x or y1 <= y:
        return out
    out[y:y1, x:x1] = crop[: y1 - y, : x1 - x].astype(np.uint8, copy=False)
    return out


def _packed_mask_bbox_and_crop(packed: object) -> tuple[BBox, np.ndarray]:
    if isinstance(packed, np.ndarray):
        mask = (np.asarray(packed, dtype=np.uint8) > 0).astype(np.uint8)
        bbox = _mask_bbox(mask, 0, mask.shape[:2])
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return (0, 0, 0, 0), np.zeros((0, 0), dtype=np.uint8)
        return bbox, mask[y : y + h, x : x + w]
    if not isinstance(packed, dict):
        return (0, 0, 0, 0), np.zeros((0, 0), dtype=np.uint8)
    bbox_raw = packed.get("bbox", (0, 0, 0, 0))
    shape_raw = packed.get("shape", (0, 0))
    bits = packed.get("bits")
    try:
        x, y, w, h = (int(bbox_raw[0]), int(bbox_raw[1]), int(bbox_raw[2]), int(bbox_raw[3]))
        ch, cw = (int(shape_raw[0]), int(shape_raw[1]))
    except Exception:
        return (0, 0, 0, 0), np.zeros((0, 0), dtype=np.uint8)
    if w <= 0 or h <= 0 or ch <= 0 or cw <= 0 or not isinstance(bits, np.ndarray):
        return (0, 0, 0, 0), np.zeros((0, 0), dtype=np.uint8)
    crop = np.unpackbits(np.asarray(bits, dtype=np.uint8), axis=1, count=int(cw))[: int(ch), : int(cw)]
    return (int(x), int(y), int(w), int(h)), crop.astype(np.uint8, copy=False)


def _measure_total_mask_length(mask: np.ndarray, config: AnalyticsConfig) -> dict[str, object]:
    bin_mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    area_px = int(np.count_nonzero(bin_mask))
    if area_px <= 0:
        return {
            "length_px": 0.0,
            "length_mm": 0.0,
            "weighted_length_px": 0.0,
            "weighted_length_mm": 0.0,
            "area_px": 0,
            "components": 0,
        }

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    total_length_px = 0.0
    total_weighted_length_px = 0.0
    component_count = 0
    for label_id in range(1, int(num_labels)):
        component_area = int(stats[label_id, cv2.CC_STAT_AREA])
        if component_area <= 0:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if width <= 0 or height <= 0:
            continue
        component_roi = (labels[y : y + height, x : x + width] == label_id).astype(np.uint8)
        component_mask = np.pad(component_roi, 1, mode="constant", constant_values=0)
        metrics = _measure_crop(component_mask, config)
        total_length_px += float(metrics.get("skeleton_length_px", metrics.get("length_px", 0.0)))
        total_weighted_length_px += float(metrics.get("weighted_length_px", 0.0))
        component_count += 1
    return {
        "length_px": float(total_length_px),
        "length_mm": float(total_length_px * config.pixel_size_mm),
        "weighted_length_px": float(total_weighted_length_px),
        "weighted_length_mm": float(total_weighted_length_px * config.pixel_size_mm),
        "area_px": int(area_px),
        "components": int(component_count),
    }


def _packed_masks_touch(
    left_packed: object,
    right_packed: object,
    *,
    radius_px: int = 1,
) -> bool:
    left_bbox, left_crop = _packed_mask_bbox_and_crop(left_packed)
    right_bbox, right_crop = _packed_mask_bbox_and_crop(right_packed)
    if left_crop.size <= 0 or right_crop.size <= 0:
        return False
    lx, ly, lw, lh = left_bbox
    rx, ry, rw, rh = right_bbox
    radius = int(max(0, min(8, radius_px)))
    x0 = min(lx, rx) - radius
    y0 = min(ly, ry) - radius
    x1 = max(lx + lw, rx + rw) + radius
    y1 = max(ly + lh, ry + rh) + radius
    canvas_width = int(max(1, x1 - x0))
    canvas_height = int(max(1, y1 - y0))
    left = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    right = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    left[ly - y0 : ly - y0 + lh, lx - x0 : lx - x0 + lw] = left_crop[:lh, :lw]
    right[ry - y0 : ry - y0 + rh, rx - x0 : rx - x0 + rw] = right_crop[:rh, :rw]
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        left = cv2.dilate(left, kernel, iterations=1)
    return bool(np.any((left > 0) & (right > 0)))


def _build_conflict_tier_timeline(
    frames: list[dict[str, object]],
    root_masks_by_frame: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    pair_first_touch: dict[tuple[str, str], int] = {}
    live_pairs_by_frame: list[set[tuple[str, str]]] = []
    present_by_frame: list[list[str]] = []
    all_track_ids: set[str] = set()

    for frame_idx, frame in enumerate(frames):
        tracks_obj = frame.get("tracks") if isinstance(frame, dict) else {}
        tracks = tracks_obj if isinstance(tracks_obj, dict) else {}
        present: list[str] = []
        present_boxes: dict[str, BBox] = {}
        for track_id, row in tracks.items():
            tid = str(track_id).strip()
            if not tid or not isinstance(row, dict):
                continue
            bbox = _row_bbox(row)
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            present.append(tid)
            present_boxes[tid] = bbox
            all_track_ids.add(tid)
        present = sorted(set(present))
        live_pairs: set[tuple[str, str]] = set()
        packed_frame = (
            root_masks_by_frame[frame_idx]
            if isinstance(root_masks_by_frame, list) and frame_idx < len(root_masks_by_frame)
            else None
        )
        for idx, left in enumerate(present):
            left_box = present_boxes.get(left, (0, 0, 0, 0))
            for right in present[idx + 1 :]:
                right_box = present_boxes.get(right, (0, 0, 0, 0))
                if not _bbox_intersects(left_box, right_box):
                    continue
                if isinstance(packed_frame, dict) and not _packed_masks_touch(
                    packed_frame.get(left),
                    packed_frame.get(right),
                    radius_px=1,
                ):
                    continue
                left_row = tracks.get(left)
                right_row = tracks.get(right)
                graph_mode = bool(
                    isinstance(left_row, dict)
                    and isinstance(right_row, dict)
                    and str(left_row.get("root_ownership_assignment_mode", "")) == "temporal_graph"
                    and str(right_row.get("root_ownership_assignment_mode", "")) == "temporal_graph"
                )
                graph_ambiguous = bool(
                    (
                        isinstance(left_row, dict)
                        and left_row.get(
                            "root_ownership_blocking_ambiguous",
                            left_row.get("root_ownership_ambiguous", False),
                        )
                    )
                    or (
                        isinstance(right_row, dict)
                        and right_row.get(
                            "root_ownership_blocking_ambiguous",
                            right_row.get("root_ownership_ambiguous", False),
                        )
                    )
                )
                if graph_mode and not graph_ambiguous:
                    continue
                pair = _track_pair(left, right)
                live_pairs.add(pair)
                pair_first_touch.setdefault(pair, int(frame_idx))
        present_by_frame.append(present)
        live_pairs_by_frame.append(live_pairs)

    all_track_ids_sorted = sorted(all_track_ids)
    per_frame: list[dict[str, object]] = []
    first_conflict_frame: dict[str, int | None] = {track_id: None for track_id in all_track_ids_sorted}
    group_ids_seen: set[str] = set()
    for frame_idx, present in enumerate(present_by_frame):
        live_pairs = live_pairs_by_frame[frame_idx]
        previous_live_pairs = live_pairs_by_frame[frame_idx - 1] if frame_idx > 0 else set()
        reacquiring_pairs = {
            pair
            for pair in previous_live_pairs
            if pair in pair_first_touch
            and pair not in live_pairs
            and pair[0] in present
            and pair[1] in present
        }
        active_pairs = set(live_pairs) | set(reacquiring_pairs)
        active_groups = _connected_track_groups(present, active_pairs)
        track_meta: dict[str, dict[str, object]] = {
            track_id: {
                "measurement_tier": "individual",
                "ownership_measurement_valid": True,
                "conflict_group_id": None,
                "conflict_group_members": [],
                "conflict_group_size": 1,
                "conflict_start_frame": None,
                "conflict_touching_now": False,
                "conflict_ever_touched": False,
                "conflict_reacquiring": False,
            }
            for track_id in all_track_ids_sorted
        }
        groups_out: list[dict[str, object]] = []
        for members in active_groups:
            if len(members) <= 1:
                continue
            member_pairs = {
                _track_pair(left, right)
                for idx, left in enumerate(members)
                for right in members[idx + 1 :]
                if _track_pair(left, right) in pair_first_touch
            }
            if not member_pairs:
                continue
            start_frame = min(int(pair_first_touch[pair]) for pair in member_pairs)
            group_id = "combined::" + "+".join(members)
            touching_now = any(pair in live_pairs for pair in member_pairs)
            reacquiring = bool(not touching_now and any(pair in reacquiring_pairs for pair in member_pairs))
            groups_out.append(
                {
                    "group_id": group_id,
                    "members": list(members),
                    "size": int(len(members)),
                    "start_frame": int(start_frame),
                    "touching_now": bool(touching_now),
                    "reacquiring": bool(reacquiring),
                }
            )
            group_ids_seen.add(group_id)
            for track_id in members:
                if first_conflict_frame.get(track_id) is None:
                    first_conflict_frame[track_id] = int(start_frame)
                track_meta[track_id] = {
                    "measurement_tier": "combined_overlap" if touching_now else "identity_reacquiring",
                    "ownership_measurement_valid": False,
                    "conflict_group_id": group_id,
                    "conflict_group_members": list(members),
                    "conflict_group_size": int(len(members)),
                    "conflict_start_frame": int(start_frame),
                    "conflict_touching_now": bool(touching_now),
                    "conflict_ever_touched": True,
                    "conflict_reacquiring": bool(reacquiring),
                }

        per_frame.append(
            {
                "frame_index": int(frame_idx),
                "tracks": track_meta,
                "groups": groups_out,
            }
        )

    return {
        "frames": per_frame,
        "track_first_conflict_frame": first_conflict_frame,
        "groups_seen": sorted(group_ids_seen),
    }


def _nearest_skeleton_point(coords_rc: np.ndarray, x: int, y: int, max_dist: float = 96.0) -> Point | None:
    if coords_rc.ndim != 2 or coords_rc.shape[1] != 2 or coords_rc.shape[0] <= 0:
        return None
    delta = coords_rc.astype(np.float32) - np.asarray([[float(y), float(x)]], dtype=np.float32)
    dist2 = np.sum(delta * delta, axis=1)
    idx = int(np.argmin(dist2))
    best = float(dist2[idx])
    if not math.isfinite(best) or best > float(max_dist * max_dist):
        return None
    return (int(coords_rc[idx, 0]), int(coords_rc[idx, 1]))


def _build_skeleton_neighbor_map(skeleton: np.ndarray) -> tuple[list[Point], dict[Point, list[Point]]]:
    coords = np.argwhere(np.asarray(skeleton, dtype=np.uint8) > 0)
    if coords.size == 0:
        return [], {}
    points = [(int(r), int(c)) for r, c in coords]
    point_set = set(points)
    neighbors: dict[Point, list[Point]] = {}
    for r, c in points:
        local: list[Point] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                q = (r + dr, c + dc)
                if q in point_set:
                    local.append(q)
        neighbors[(r, c)] = local
    return points, neighbors


def _infer_distal_tip_point(mask: np.ndarray) -> Point | None:
    bin_mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if int(np.count_nonzero(bin_mask)) <= 0:
        return None
    skeleton = _prune_small_skeleton_components(_skeletonize(bin_mask), 1)
    points, neighbors = _build_skeleton_neighbor_map(skeleton)
    if points:
        endpoints = [p for p, local in neighbors.items() if len(local) == 1]
        target_rc = max(endpoints if endpoints else points, key=lambda p: (int(p[0]), int(p[1])))
        return (int(target_rc[1]), int(target_rc[0]))
    ys, xs = np.where(bin_mask > 0)
    if xs.size <= 0 or ys.size <= 0:
        return None
    idx = int(np.argmax(ys))
    return (int(xs[idx]), int(ys[idx]))


def _trace_tip_guided_path_seed(
    src_mask: np.ndarray,
    hint: dict[str, object] | None,
    frame_idx: int,
    shape_hw: tuple[int, int],
    config: AnalyticsConfig,
    tip_points_xy: list[Point] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    hint = _resolve_track_compartment_hint(hint, frame_idx, shape_hw)
    connected_root_full, _ = _extract_track_mask_from_hint(src_mask, hint, frame_idx, shape_hw, config)
    bin_mask = (np.asarray(connected_root_full, dtype=np.uint8) > 0).astype(np.uint8)
    if int(np.count_nonzero(bin_mask)) <= 0:
        return bin_mask, np.zeros_like(bin_mask, dtype=np.uint8)

    skeleton = _prune_small_skeleton_components(_skeletonize(bin_mask), max(1, int(config.prune_branch_px)))
    coords = np.argwhere(np.asarray(skeleton, dtype=np.uint8) > 0)
    if coords.size == 0:
        return bin_mask, bin_mask.copy()

    points, neighbors = _build_skeleton_neighbor_map(skeleton)
    if not points:
        return bin_mask, bin_mask.copy()

    starts: list[Point] = []
    seed_mask = None
    fixed_top_y = 0
    if isinstance(hint, dict):
        seed_mask_obj = _seed_mask_for_frame_source(src_mask, hint, "seed_mask", shape_hw)
        if seed_mask_obj.shape == bin_mask.shape:
            seed_mask = (np.asarray(seed_mask_obj, dtype=np.uint8) > 0).astype(np.uint8)
        fixed_top_y = int(max(0, min(shape_hw[0] - 1, int(hint.get("fixed_top_y", 0)))))
    if isinstance(seed_mask, np.ndarray):
        seed_overlap = np.argwhere((seed_mask > 0) & (np.asarray(skeleton, dtype=np.uint8) > 0))
        starts.extend((int(r), int(c)) for r, c in seed_overlap)
    if not starts and fixed_top_y > 0:
        starts.extend((int(r), int(c)) for r, c in coords if int(r) <= int(fixed_top_y))
    if not starts:
        top_idx = int(np.argmin(coords[:, 0]))
        starts.append((int(coords[top_idx, 0]), int(coords[top_idx, 1])))
    starts = sorted({(int(r), int(c)) for r, c in starts}, key=lambda p: (p[0], p[1]))

    dist: dict[Point, int] = {}
    parent: dict[Point, Point] = {}
    q = deque(starts)
    for start in starts:
        dist[start] = 0
    while q:
        cur = q.popleft()
        for nxt in neighbors.get(cur, []):
            if nxt in dist:
                continue
            dist[nxt] = int(dist[cur] + 1)
            parent[nxt] = cur
            q.append(nxt)
    if not dist:
        return bin_mask, bin_mask.copy()

    candidate_targets: list[Point] = []
    for tip in tip_points_xy or []:
        target = _nearest_skeleton_point(coords, int(tip[0]), int(tip[1]), max_dist=96.0)
        if target is not None:
            candidate_targets.append(target)
    if not candidate_targets:
        endpoints = [p for p, local in neighbors.items() if len(local) == 1 and p in dist]
        if endpoints:
            candidate_targets = endpoints
        else:
            candidate_targets = [max(dist.keys(), key=lambda p: int(dist[p]))]

    best_target = max(candidate_targets, key=lambda p: int(dist.get(p, -1)))
    if best_target not in dist:
        return bin_mask, bin_mask.copy()

    path_rc = [best_target]
    while path_rc[-1] not in starts:
        nxt = parent.get(path_rc[-1])
        if nxt is None:
            break
        path_rc.append(nxt)
    path_rc.reverse()
    if not path_rc:
        return bin_mask, bin_mask.copy()

    path_seed = np.zeros_like(bin_mask, dtype=np.uint8)
    for r, c in path_rc:
        if 0 <= int(r) < shape_hw[0] and 0 <= int(c) < shape_hw[1]:
            path_seed[int(r), int(c)] = 1
    radius = max(1, min(9, int(round(max(1.0, float(config.prune_branch_px) * 0.6)))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    path_seed = cv2.dilate(path_seed, kernel, iterations=1)
    path_seed[bin_mask <= 0] = 0
    if isinstance(seed_mask, np.ndarray):
        path_seed[seed_mask > 0] = 1
    return bin_mask, path_seed.astype(np.uint8, copy=False)


def _assign_pixels_to_track_seeds_euclidean(
    source_mask: np.ndarray,
    track_seed_masks: dict[str, np.ndarray],
    compartment_hints: dict[str, dict[str, object]],
    *,
    track_prior_masks: dict[str, np.ndarray] | None = None,
    prior_weight_px: float = 0.0,
) -> dict[str, np.ndarray]:
    src = (np.asarray(source_mask, dtype=np.uint8) > 0).astype(np.uint8)
    h, w = src.shape[:2]
    track_ids = [str(track_id) for track_id, seed in track_seed_masks.items() if isinstance(seed, np.ndarray) and seed.shape == src.shape]
    if int(np.count_nonzero(src)) <= 0 or not track_ids:
        return {track_id: np.zeros((h, w), dtype=np.uint8) for track_id in track_seed_masks.keys()}

    best_dist = np.full((h, w), fill_value=np.float32(1.0e9), dtype=np.float32)
    best_idx = np.full((h, w), fill_value=-1, dtype=np.int16)
    x_coords = np.arange(w, dtype=np.int32)[None, :]
    label_count, source_labels, _stats, _centroids = cv2.connectedComponentsWithStats(src, connectivity=8)

    for idx, track_id in enumerate(track_ids):
        seed = (np.asarray(track_seed_masks.get(track_id), dtype=np.uint8) > 0).astype(np.uint8)
        if seed.shape != src.shape or int(np.count_nonzero(seed)) <= 0:
            continue
        touched = np.unique(source_labels[(seed > 0) & (src > 0)])
        touched = touched[touched > 0]
        if touched.size <= 0:
            near_seed = (_dilate_binary(seed, 2) > 0) & (src > 0)
            touched = np.unique(source_labels[near_seed])
            touched = touched[touched > 0]
        inv = np.where(seed > 0, 0, 1).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        if track_prior_masks and float(prior_weight_px) > 0.0:
            prior = track_prior_masks.get(track_id)
            if isinstance(prior, np.ndarray) and prior.shape == src.shape and int(np.count_nonzero(prior)) > 0:
                prior_u8 = (np.asarray(prior, dtype=np.uint8) > 0).astype(np.uint8)
                prior_inv = np.where(prior_u8 > 0, 0, 1).astype(np.uint8)
                prior_dist = cv2.distanceTransform(prior_inv, cv2.DIST_L2, 3)
                discount = np.maximum(0.0, float(prior_weight_px) - prior_dist.astype(np.float32))
                dist = dist.astype(np.float32) - discount.astype(np.float32)
        valid = src > 0
        hint = compartment_hints.get(track_id)
        if isinstance(hint, dict):
            left = int(max(0, min(w, int(hint.get("lane_left", 0)))))
            right = int(max(left + 1, min(w, int(hint.get("lane_right", w)))))
            lane_valid = (x_coords >= left) & (x_coords < right)
            if touched.size > 0 and int(label_count) > 1:
                component_valid = np.isin(source_labels, touched)
                valid = valid & (lane_valid | component_valid)
            else:
                valid = valid & lane_valid
        update = valid & (dist < best_dist)
        best_dist[update] = dist[update]
        best_idx[update] = int(idx)

    out: dict[str, np.ndarray] = {}
    for idx, track_id in enumerate(track_ids):
        owned = ((best_idx == int(idx)) & (src > 0)).astype(np.uint8)
        hint = compartment_hints.get(track_id)
        seed_touch = (np.asarray(track_seed_masks.get(track_id), dtype=np.uint8) > 0).astype(np.uint8)
        if isinstance(hint, dict):
            seed_mask_obj = hint.get("seed_mask")
            if isinstance(seed_mask_obj, np.ndarray) and seed_mask_obj.shape == src.shape:
                seed_touch = np.maximum(seed_touch, (np.asarray(seed_mask_obj, dtype=np.uint8) > 0).astype(np.uint8))
        if int(np.count_nonzero(owned)) > 0 and int(np.count_nonzero(seed_touch)) > 0:
            prior = track_prior_masks.get(track_id) if track_prior_masks and float(prior_weight_px) > 0.0 else None
            if isinstance(prior, np.ndarray) and prior.shape == src.shape and int(np.count_nonzero(prior)) > 0:
                seed_touch = np.maximum(seed_touch, (np.asarray(prior, dtype=np.uint8) > 0).astype(np.uint8))
            owned = _keep_components_touching_seed(owned, seed_touch)
        out[track_id] = owned.astype(np.uint8, copy=False)
    for track_id in track_seed_masks.keys():
        out.setdefault(str(track_id), np.zeros((h, w), dtype=np.uint8))
    return out


def _skeleton_geodesic_distance(
    skeleton: np.ndarray,
    source_nodes: np.ndarray,
) -> np.ndarray:
    """Return 8-neighbor distance constrained to a one-pixel skeleton."""
    skeleton_u8 = (np.asarray(skeleton, dtype=np.uint8) > 0).astype(np.uint8)
    source_u8 = (
        (np.asarray(source_nodes, dtype=np.uint8) > 0) & (skeleton_u8 > 0)
    ).astype(np.uint8)
    shape_hw = skeleton_u8.shape[:2]
    out = np.full(shape_hw, np.float32(np.inf), dtype=np.float32)
    coords = np.argwhere(skeleton_u8 > 0)
    if coords.size <= 0 or int(np.count_nonzero(source_u8)) <= 0:
        return out

    node_index = np.full(shape_hw, -1, dtype=np.int32)
    node_index[coords[:, 0], coords[:, 1]] = np.arange(coords.shape[0], dtype=np.int32)
    distances = np.full(coords.shape[0], np.float32(np.inf), dtype=np.float32)
    heap: list[tuple[float, int]] = []
    for row, col in np.argwhere(source_u8 > 0):
        idx = int(node_index[int(row), int(col)])
        if idx < 0 or float(distances[idx]) <= 0.0:
            continue
        distances[idx] = np.float32(0.0)
        heapq.heappush(heap, (0.0, idx))

    diagonal_cost = math.sqrt(2.0)
    height, width = shape_hw
    while heap:
        current_dist, idx = heapq.heappop(heap)
        if current_dist > float(distances[idx]) + 1.0e-6:
            continue
        row = int(coords[idx, 0])
        col = int(coords[idx, 1])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = row + dr
                nc = col + dc
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                next_idx = int(node_index[nr, nc])
                if next_idx < 0:
                    continue
                step = diagonal_cost if dr != 0 and dc != 0 else 1.0
                candidate = float(current_dist + step)
                if candidate + 1.0e-6 >= float(distances[next_idx]):
                    continue
                distances[next_idx] = np.float32(candidate)
                heapq.heappush(heap, (candidate, next_idx))

    out[coords[:, 0], coords[:, 1]] = distances
    return out


def _nearest_skeleton_seed_support(
    skeleton: np.ndarray,
    seed_mask: np.ndarray,
    *,
    max_distance_px: float = 96.0,
) -> np.ndarray:
    skeleton_u8 = (np.asarray(skeleton, dtype=np.uint8) > 0).astype(np.uint8)
    seed_u8 = (np.asarray(seed_mask, dtype=np.uint8) > 0).astype(np.uint8)
    support = ((skeleton_u8 > 0) & (_dilate_binary(seed_u8, 2) > 0)).astype(np.uint8)
    if int(np.count_nonzero(support)) > 0:
        return support
    if int(np.count_nonzero(seed_u8)) <= 0 or int(np.count_nonzero(skeleton_u8)) <= 0:
        return support
    distance_to_seed = cv2.distanceTransform(
        np.where(seed_u8 > 0, 0, 1).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    skeleton_coords = np.argwhere(skeleton_u8 > 0)
    values = distance_to_seed[skeleton_coords[:, 0], skeleton_coords[:, 1]]
    if values.size <= 0:
        return support
    best_index = int(np.argmin(values))
    if not math.isfinite(float(values[best_index])) or float(values[best_index]) > float(max_distance_px):
        return support
    row = int(skeleton_coords[best_index, 0])
    col = int(skeleton_coords[best_index, 1])
    support[row, col] = 1
    return support


def _assign_pixels_to_track_seeds_graph(
    source_mask: np.ndarray,
    track_seed_masks: dict[str, np.ndarray],
    compartment_hints: dict[str, dict[str, object]],
    *,
    track_prior_masks: dict[str, np.ndarray] | None = None,
    prior_weight_px: float = 0.0,
    track_unary_masks: dict[str, np.ndarray] | None = None,
    unary_weight_px: float = 0.0,
    ambiguity_margin_px: float = 2.5,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    src = (np.asarray(source_mask, dtype=np.uint8) > 0).astype(np.uint8)
    height, width = src.shape[:2]
    track_ids = sorted(
        str(track_id)
        for track_id, seed in track_seed_masks.items()
        if isinstance(seed, np.ndarray) and seed.shape == src.shape
    )
    empty_out = {
        str(track_id): np.zeros((height, width), dtype=np.uint8)
        for track_id in track_seed_masks.keys()
    }
    empty_meta: dict[str, object] = {
        "mode": "temporal_graph",
        "ambiguous_pixels": 0,
        "shared_components": 0,
        "ambiguity_mask": np.zeros((height, width), dtype=np.uint8),
        "tracks": {
            str(track_id): {
                "assigned_pixels": 0,
                "ambiguous_pixels": 0,
                "shared_components": 0,
                "ambiguous": False,
            }
            for track_id in track_seed_masks.keys()
        },
    }
    if int(np.count_nonzero(src)) <= 0 or not track_ids:
        return empty_out, empty_meta

    fallback = _assign_pixels_to_track_seeds_euclidean(
        src,
        track_seed_masks,
        compartment_hints,
        track_prior_masks=track_prior_masks,
        prior_weight_px=prior_weight_px,
    )
    out = {track_id: np.zeros((height, width), dtype=np.uint8) for track_id in track_ids}
    ambiguity_mask = np.zeros((height, width), dtype=np.uint8)
    track_meta: dict[str, dict[str, object]] = {
        track_id: {
            "assigned_pixels": 0,
            "ambiguous_pixels": 0,
            "shared_components": 0,
            "ambiguous": False,
        }
        for track_id in track_ids
    }
    shared_components = 0
    component_count, component_labels, component_stats, _centroids = cv2.connectedComponentsWithStats(
        src,
        connectivity=8,
    )
    margin_threshold = float(max(0.0, ambiguity_margin_px))

    for component_id in range(1, int(component_count)):
        x = int(component_stats[component_id, cv2.CC_STAT_LEFT])
        y = int(component_stats[component_id, cv2.CC_STAT_TOP])
        component_width = int(component_stats[component_id, cv2.CC_STAT_WIDTH])
        component_height = int(component_stats[component_id, cv2.CC_STAT_HEIGHT])
        if component_width <= 0 or component_height <= 0:
            continue
        x0 = max(0, x - 2)
        y0 = max(0, y - 2)
        x1 = min(width, x + component_width + 2)
        y1 = min(height, y + component_height + 2)
        component = (component_labels[y0:y1, x0:x1] == int(component_id)).astype(np.uint8)
        if int(np.count_nonzero(component)) <= 0:
            continue

        prior_candidate_ids: list[str] = []
        seed_exact_candidate_ids: list[str] = []
        nearby_candidate_ids: list[str] = []
        seed_crops: dict[str, np.ndarray] = {}
        prior_crops: dict[str, np.ndarray] = {}
        unary_crops: dict[str, np.ndarray] = {}
        for track_id in track_ids:
            seed_full = (np.asarray(track_seed_masks[track_id], dtype=np.uint8) > 0).astype(np.uint8)
            seed_crop = seed_full[y0:y1, x0:x1]
            prior_full = track_prior_masks.get(track_id) if isinstance(track_prior_masks, dict) else None
            if isinstance(prior_full, np.ndarray) and prior_full.shape == src.shape:
                prior_crop = (np.asarray(prior_full[y0:y1, x0:x1], dtype=np.uint8) > 0).astype(np.uint8)
                has_prior = int(np.count_nonzero(prior_full)) > 0
            else:
                prior_crop = np.zeros_like(component, dtype=np.uint8)
                has_prior = False
            unary_full = (
                track_unary_masks.get(track_id)
                if isinstance(track_unary_masks, dict)
                else None
            )
            unary_crop = (
                (np.asarray(unary_full[y0:y1, x0:x1], dtype=np.uint8) > 0).astype(np.uint8)
                if isinstance(unary_full, np.ndarray) and unary_full.shape == src.shape
                else np.zeros_like(component, dtype=np.uint8)
            )
            seed_crops[track_id] = seed_crop
            prior_crops[track_id] = prior_crop
            unary_crops[track_id] = unary_crop
            touches_seed_exact = bool(np.any((component > 0) & (seed_crop > 0)))
            touches_seed_nearby = bool(np.any((component > 0) & (_dilate_binary(seed_crop, 2) > 0)))
            touches_prior = bool(np.any((component > 0) & (prior_crop > 0)))
            touches_prior_nearby = bool(np.any((component > 0) & (_dilate_binary(prior_crop, 2) > 0)))
            if has_prior and (touches_prior or touches_prior_nearby):
                prior_candidate_ids.append(track_id)
            elif touches_seed_exact:
                seed_exact_candidate_ids.append(track_id)
            elif touches_seed_nearby:
                nearby_candidate_ids.append(track_id)

        # A previous owned mask is stronger identity evidence than a lane seed.
        # When at least one temporal prior reaches this component, do not let a
        # neighboring track's current lane seed claim it. Multiple touching
        # priors still enter the shared-component ambiguity calculation below.
        if prior_candidate_ids:
            candidate_ids = prior_candidate_ids
        elif seed_exact_candidate_ids:
            candidate_ids = seed_exact_candidate_ids
        else:
            candidate_ids = nearby_candidate_ids

        if not candidate_ids:
            component_full = component_labels == int(component_id)
            fallback_ids: list[str] = []
            for track_id in track_ids:
                fallback_mask = fallback.get(track_id)
                if isinstance(fallback_mask, np.ndarray) and fallback_mask.shape == src.shape:
                    assigned = component_full & (fallback_mask > 0)
                    out[track_id][assigned] = 1
                    if bool(np.any(assigned)):
                        fallback_ids.append(track_id)
            has_temporal_history = bool(
                isinstance(track_prior_masks, dict)
                and any(
                    isinstance(prior, np.ndarray) and int(np.count_nonzero(prior)) > 0
                    for prior in track_prior_masks.values()
                )
            )
            if has_temporal_history and fallback_ids:
                ambiguity_mask[component_full] = 1
                for track_id in fallback_ids:
                    track_meta[track_id]["ambiguous"] = True
                    track_meta[track_id]["ambiguous_pixels"] = int(track_meta[track_id]["ambiguous_pixels"]) + int(
                        np.count_nonzero(component_full)
                    )
            continue
        if len(candidate_ids) == 1:
            track_id = candidate_ids[0]
            out[track_id][component_labels == int(component_id)] = 1
            continue

        shared_components += 1
        skeleton = (_skeletonize(component) > 0).astype(np.uint8)
        if int(np.count_nonzero(skeleton)) <= 0:
            component_full = component_labels == int(component_id)
            for track_id in candidate_ids:
                fallback_mask = fallback.get(track_id)
                if isinstance(fallback_mask, np.ndarray) and fallback_mask.shape == src.shape:
                    out[track_id][component_full & (fallback_mask > 0)] = 1
            continue

        support_by_track: dict[str, np.ndarray] = {}
        support_stack: list[np.ndarray] = []
        for track_id in candidate_ids:
            prior_on_component = (
                (prior_crops[track_id] > 0) & (component > 0)
            ).astype(np.uint8)
            prior_support = _nearest_skeleton_seed_support(skeleton, prior_on_component)
            if int(np.count_nonzero(prior_support)) > 0:
                support = prior_support
            else:
                support = _nearest_skeleton_seed_support(skeleton, seed_crops[track_id])
            support_by_track[track_id] = support
            support_stack.append(support)
        support_count = np.sum(np.stack(support_stack, axis=0).astype(np.uint8), axis=0)

        baseline_graph_costs: list[np.ndarray] = []
        fused_graph_costs: list[np.ndarray] = []
        for track_id in candidate_ids:
            support = support_by_track[track_id]
            unique_support = ((support > 0) & (support_count == 1)).astype(np.uint8)
            if int(np.count_nonzero(unique_support)) <= 0:
                unique_support = support
            graph_cost = _skeleton_geodesic_distance(skeleton, unique_support)
            finite = np.isfinite(graph_cost)
            if not bool(np.any(finite)):
                seed = seed_crops[track_id]
                graph_cost = cv2.distanceTransform(
                    np.where(seed > 0, 0, 1).astype(np.uint8),
                    cv2.DIST_L2,
                    5,
                ).astype(np.float32)
                graph_cost[skeleton <= 0] = np.float32(np.inf)
            prior = prior_crops[track_id]
            if float(prior_weight_px) > 0.0 and int(np.count_nonzero(prior)) > 0:
                prior_distance = cv2.distanceTransform(
                    np.where(prior > 0, 0, 1).astype(np.uint8),
                    cv2.DIST_L2,
                    5,
                ).astype(np.float32)
                graph_cost = graph_cost - np.maximum(
                    0.0,
                    float(prior_weight_px) - prior_distance,
                ).astype(np.float32)
            hint = compartment_hints.get(track_id)
            if isinstance(hint, dict):
                lane_left = int(max(0, min(width, int(hint.get("lane_left", 0))))) - x0
                lane_right = int(max(0, min(width, int(hint.get("lane_right", width))))) - x0
                local_x = np.arange(component.shape[1], dtype=np.float32)[None, :]
                outside_distance = np.maximum(
                    np.maximum(float(lane_left) - local_x, local_x - float(max(lane_left, lane_right - 1))),
                    0.0,
                )
                graph_cost = graph_cost + (0.08 * outside_distance).astype(np.float32)
            graph_cost[skeleton <= 0] = np.float32(np.inf)
            baseline_cost = graph_cost.astype(np.float32, copy=True)
            baseline_graph_costs.append(baseline_cost)

            fused_cost = baseline_cost.copy()
            unary = unary_crops[track_id]
            if float(unary_weight_px) > 0.0 and int(np.count_nonzero(unary)) > 0:
                unary_distance = cv2.distanceTransform(
                    np.where(unary > 0, 0, 1).astype(np.uint8),
                    cv2.DIST_L2,
                    5,
                ).astype(np.float32)
                fused_cost = fused_cost - np.maximum(
                    0.0,
                    float(unary_weight_px) - unary_distance,
                ).astype(np.float32)
            fused_cost[skeleton <= 0] = np.float32(np.inf)
            fused_graph_costs.append(fused_cost.astype(np.float32, copy=False))

        baseline_cost_stack = np.stack(baseline_graph_costs, axis=0)
        baseline_sorted_costs = np.sort(baseline_cost_stack, axis=0, kind="stable")
        baseline_cost_order = np.argsort(
            baseline_cost_stack,
            axis=0,
            kind="stable",
        )
        baseline_owner_index = np.argmin(baseline_cost_stack, axis=0)
        baseline_best_cost = baseline_sorted_costs[0]
        baseline_second_cost = baseline_sorted_costs[1]
        finite_best = np.isfinite(baseline_best_cost) & (skeleton > 0)
        baseline_finite_pair = finite_best & np.isfinite(baseline_second_cost)
        baseline_skeleton_margin = np.full(
            baseline_best_cost.shape,
            np.float32(np.inf),
            dtype=np.float32,
        )
        baseline_skeleton_margin[baseline_finite_pair] = (
            baseline_second_cost[baseline_finite_pair]
            - baseline_best_cost[baseline_finite_pair]
        )
        baseline_ambiguous_skeleton = (
            baseline_finite_pair
            & (baseline_skeleton_margin <= float(margin_threshold))
        )

        fused_cost_stack = np.stack(fused_graph_costs, axis=0)
        fused_sorted_costs = np.sort(fused_cost_stack, axis=0, kind="stable")
        fused_owner_index = np.argmin(fused_cost_stack, axis=0)
        fused_best_cost = fused_sorted_costs[0]
        fused_second_cost = fused_sorted_costs[1]
        fused_finite_pair = (
            np.isfinite(fused_best_cost)
            & np.isfinite(fused_second_cost)
            & (skeleton > 0)
        )
        fused_skeleton_margin = np.full(
            fused_best_cost.shape,
            np.float32(np.inf),
            dtype=np.float32,
        )
        fused_skeleton_margin[fused_finite_pair] = (
            fused_second_cost[fused_finite_pair]
            - fused_best_cost[fused_finite_pair]
        )
        fused_ambiguous_skeleton = (
            fused_finite_pair
            & (fused_skeleton_margin <= float(margin_threshold))
        )

        unary_stack = np.stack(
            [(unary_crops[track_id] > 0).astype(np.uint8) for track_id in candidate_ids],
            axis=0,
        )
        prior_stack = np.stack(
            [(prior_crops[track_id] > 0).astype(np.uint8) for track_id in candidate_ids],
            axis=0,
        )
        fused_has_direct_support = np.take_along_axis(
            unary_stack,
            fused_owner_index[None, ...],
            axis=0,
        )[0] > 0
        fused_is_baseline_top_two = (
            (fused_owner_index == baseline_cost_order[0])
            | (fused_owner_index == baseline_cost_order[1])
        )
        unique_temporal_prior = np.sum(prior_stack, axis=0) == 1
        learned_resolution = (
            baseline_ambiguous_skeleton
            & ~fused_ambiguous_skeleton
            & fused_has_direct_support
            & fused_is_baseline_top_two
            & ~unique_temporal_prior
        )

        # Learned evidence is a constrained tie-breaker. It may resolve an
        # existing graph close call, but cannot alter a confident decision,
        # contradict a unique temporal prior, or act through remote support.
        owner_index = baseline_owner_index.copy()
        owner_index[learned_resolution] = fused_owner_index[
            learned_resolution
        ]
        ambiguous_skeleton = (
            baseline_ambiguous_skeleton & ~learned_resolution
        )

        owned_skeletons: list[np.ndarray] = []
        for track_index, _track_id in enumerate(candidate_ids):
            owned_skeletons.append(
                ((owner_index == int(track_index)) & finite_best).astype(np.uint8)
            )
        baseline_owned_skeletons: list[np.ndarray] = []
        for track_index, _track_id in enumerate(candidate_ids):
            baseline_owned_skeletons.append(
                ((baseline_owner_index == int(track_index)) & finite_best).astype(np.uint8)
            )

        baseline_pixel_distances: list[np.ndarray] = []
        for owned_skeleton in baseline_owned_skeletons:
            if int(np.count_nonzero(owned_skeleton)) <= 0:
                baseline_pixel_distances.append(
                    np.full(component.shape, np.float32(np.inf), dtype=np.float32)
                )
                continue
            baseline_pixel_distances.append(
                cv2.distanceTransform(
                    np.where(owned_skeleton > 0, 0, 1).astype(np.uint8),
                    cv2.DIST_L2,
                    5,
                ).astype(np.float32)
            )
        baseline_pixel_cost_stack = np.stack(baseline_pixel_distances, axis=0)
        baseline_pixel_owner_index = np.argmin(
            baseline_pixel_cost_stack,
            axis=0,
        )
        baseline_sorted_pixel_costs = np.sort(
            baseline_pixel_cost_stack,
            axis=0,
            kind="stable",
        )
        baseline_finite_pixel_pair = (
            np.isfinite(baseline_sorted_pixel_costs[0])
            & np.isfinite(baseline_sorted_pixel_costs[1])
        )
        baseline_pixel_margin = np.full(
            component.shape,
            np.float32(np.inf),
            dtype=np.float32,
        )
        baseline_pixel_margin[baseline_finite_pixel_pair] = (
            baseline_sorted_pixel_costs[1][baseline_finite_pixel_pair]
            - baseline_sorted_pixel_costs[0][baseline_finite_pixel_pair]
        )
        baseline_ambiguous_near_graph = cv2.dilate(
            baseline_ambiguous_skeleton.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ) > 0
        baseline_ambiguous_component = (
            (component > 0)
            & (
                baseline_ambiguous_near_graph
                | (
                    baseline_finite_pixel_pair
                    & (
                        baseline_pixel_margin
                        <= float(max(1.0, 0.5 * margin_threshold))
                    )
                )
            )
        )

        pixel_distances: list[np.ndarray] = []
        for owned_skeleton in owned_skeletons:
            if int(np.count_nonzero(owned_skeleton)) <= 0:
                pixel_distances.append(
                    np.full(component.shape, np.float32(np.inf), dtype=np.float32)
                )
                continue
            pixel_distances.append(
                cv2.distanceTransform(
                    np.where(owned_skeleton > 0, 0, 1).astype(np.uint8),
                    cv2.DIST_L2,
                    5,
                ).astype(np.float32)
            )
        pixel_cost_stack = np.stack(pixel_distances, axis=0)
        proposed_pixel_owner_index = np.argmin(pixel_cost_stack, axis=0)
        sorted_pixel_costs = np.sort(pixel_cost_stack, axis=0, kind="stable")
        finite_pixel_pair = np.isfinite(sorted_pixel_costs[0]) & np.isfinite(sorted_pixel_costs[1])
        pixel_margin = np.full(component.shape, np.float32(np.inf), dtype=np.float32)
        pixel_margin[finite_pixel_pair] = (
            sorted_pixel_costs[1][finite_pixel_pair] - sorted_pixel_costs[0][finite_pixel_pair]
        )
        ambiguous_near_graph = cv2.dilate(
            ambiguous_skeleton.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ) > 0
        fused_ambiguous_component = (
            (component > 0)
            & (
                ambiguous_near_graph
                | (
                    finite_pixel_pair
                    & (pixel_margin <= float(max(1.0, 0.5 * margin_threshold)))
                )
            )
        )
        ambiguous_component = (
            baseline_ambiguous_component & fused_ambiguous_component
        )
        pixel_owner_index = baseline_pixel_owner_index.copy()
        pixel_owner_index[baseline_ambiguous_component] = (
            proposed_pixel_owner_index[baseline_ambiguous_component]
        )
        starved_track = any(
            int(
                np.count_nonzero(
                    (component > 0)
                    & (baseline_pixel_owner_index == int(track_index))
                )
            )
            > 0
            and int(
                np.count_nonzero(
                    (component > 0)
                    & (pixel_owner_index == int(track_index))
                )
            )
            <= 0
            for track_index, _track_id in enumerate(candidate_ids)
        )
        if starved_track:
            pixel_owner_index = baseline_pixel_owner_index
            ambiguous_component = baseline_ambiguous_component
        ambiguity_mask[y0:y1, x0:x1][ambiguous_component] = 1

        ambiguous_count = int(np.count_nonzero(ambiguous_component))
        for track_index, track_id in enumerate(candidate_ids):
            local_owned = ((component > 0) & (pixel_owner_index == int(track_index))).astype(np.uint8)
            target = out[track_id][y0:y1, x0:x1]
            target[local_owned > 0] = 1
            track_meta[track_id]["shared_components"] = int(track_meta[track_id]["shared_components"]) + 1
            if ambiguous_count > 0:
                track_meta[track_id]["ambiguous"] = True
                track_meta[track_id]["ambiguous_pixels"] = int(track_meta[track_id]["ambiguous_pixels"]) + ambiguous_count

    for track_id in track_ids:
        out[track_id] = ((out[track_id] > 0) & (src > 0)).astype(np.uint8)
        track_meta[track_id]["assigned_pixels"] = int(np.count_nonzero(out[track_id]))
    for track_id in track_seed_masks.keys():
        out.setdefault(str(track_id), np.zeros((height, width), dtype=np.uint8))
        track_meta.setdefault(
            str(track_id),
            {"assigned_pixels": 0, "ambiguous_pixels": 0, "shared_components": 0, "ambiguous": False},
        )
    return out, {
        "mode": "temporal_graph",
        "ambiguous_pixels": int(np.count_nonzero(ambiguity_mask)),
        "shared_components": int(shared_components),
        "ambiguity_mask": ambiguity_mask.astype(np.uint8, copy=False),
        "tracks": track_meta,
    }


def _assign_pixels_to_track_seeds(
    source_mask: np.ndarray,
    track_seed_masks: dict[str, np.ndarray],
    compartment_hints: dict[str, dict[str, object]],
    *,
    track_prior_masks: dict[str, np.ndarray] | None = None,
    prior_weight_px: float = 0.0,
    track_unary_masks: dict[str, np.ndarray] | None = None,
    unary_weight_px: float = 0.0,
    assignment_mode: str = "euclidean",
    ambiguity_margin_px: float = 2.5,
    return_metadata: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, object]]:
    mode = str(assignment_mode or "euclidean").strip().lower()
    if mode in {"temporal_graph", "graph", "geodesic", "skeleton_graph"}:
        masks, meta = _assign_pixels_to_track_seeds_graph(
            source_mask,
            track_seed_masks,
            compartment_hints,
            track_prior_masks=track_prior_masks,
            prior_weight_px=prior_weight_px,
            track_unary_masks=track_unary_masks,
            unary_weight_px=unary_weight_px,
            ambiguity_margin_px=ambiguity_margin_px,
        )
    else:
        masks = _assign_pixels_to_track_seeds_euclidean(
            source_mask,
            track_seed_masks,
            compartment_hints,
            track_prior_masks=track_prior_masks,
            prior_weight_px=prior_weight_px,
        )
        shape_hw = np.asarray(source_mask).shape[:2]
        meta = {
            "mode": "euclidean",
            "ambiguous_pixels": 0,
            "shared_components": 0,
            "ambiguity_mask": np.zeros(shape_hw, dtype=np.uint8),
            "tracks": {
                str(track_id): {
                    "assigned_pixels": int(np.count_nonzero(mask)),
                    "ambiguous_pixels": 0,
                    "shared_components": 0,
                    "ambiguous": False,
                }
                for track_id, mask in masks.items()
            },
        }
    if return_metadata:
        return masks, meta
    return masks


def _recover_unassigned_crown_lane_components(
    source_mask: np.ndarray,
    owned_masks: dict[str, np.ndarray],
    track_seed_masks: dict[str, np.ndarray],
    compartment_hints: dict[str, dict[str, object]],
    *,
    track_prior_masks: dict[str, np.ndarray] | None = None,
    ambiguity_margin_px: float = 2.5,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Assign detached root fragments without inventing or duplicating pixels.

    Lucifer masks often contain short gaps that disconnect a genuine branch
    from its crown-connected component. The graph pass intentionally abstains
    on those fragments. In the dedicated five-crown mode, recover each whole
    component using its distance to the current path/prior seeds plus a small
    stable-lane penalty. Close calls remain explicitly ambiguous.
    """
    src = (np.asarray(source_mask, dtype=np.uint8) > 0).astype(np.uint8)
    height, width = src.shape[:2]
    track_ids = [
        str(track_id)
        for track_id in track_seed_masks.keys()
        if str(track_id) in compartment_hints
    ]
    out = {
        str(track_id): (
            (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
            if isinstance(mask, np.ndarray) and mask.shape == src.shape
            else np.zeros((height, width), dtype=np.uint8)
        )
        for track_id, mask in owned_masks.items()
    }
    for track_id in track_seed_masks.keys():
        out.setdefault(str(track_id), np.zeros((height, width), dtype=np.uint8))

    assigned = np.zeros((height, width), dtype=np.uint8)
    for mask in out.values():
        assigned = np.maximum(assigned, (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8))
    unassigned = ((src > 0) & (assigned <= 0)).astype(np.uint8)
    unassigned_pixels = int(np.count_nonzero(unassigned))
    empty_meta: dict[str, object] = {
        "enabled": True,
        "components": 0,
        "pixels": 0,
        "ambiguous_components": 0,
        "ambiguous_pixels": 0,
        "ambiguous_tracks": [],
        "ambiguity_mask": np.zeros((height, width), dtype=np.uint8),
        "tracks": {},
    }
    if unassigned_pixels <= 0 or not track_ids:
        return out, empty_meta

    component_count, component_labels, component_stats, component_centroids = cv2.connectedComponentsWithStats(
        unassigned,
        connectivity=8,
    )
    if int(component_count) <= 1:
        return out, empty_meta

    component_ids = list(range(1, int(component_count)))
    scores = np.full((len(component_ids), len(track_ids)), np.inf, dtype=np.float64)
    for track_index, track_id in enumerate(track_ids):
        support = np.zeros((height, width), dtype=np.uint8)
        seed = track_seed_masks.get(track_id)
        if isinstance(seed, np.ndarray) and seed.shape == src.shape:
            support = np.maximum(support, (np.asarray(seed, dtype=np.uint8) > 0).astype(np.uint8))
        prior = track_prior_masks.get(track_id) if isinstance(track_prior_masks, dict) else None
        if isinstance(prior, np.ndarray) and prior.shape == src.shape:
            support = np.maximum(support, (np.asarray(prior, dtype=np.uint8) > 0).astype(np.uint8))

        if int(np.count_nonzero(support)) > 0:
            distance = cv2.distanceTransform(
                np.where(support > 0, 0, 1).astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
        else:
            distance = None

        hint = compartment_hints.get(track_id, {})
        lane_left = int(max(0, min(width, int(hint.get("lane_left", 0)))))
        lane_right = int(max(lane_left + 1, min(width, int(hint.get("lane_right", width)))))
        lane_center = float(hint.get("lane_center_x", 0.5 * float(lane_left + lane_right)))
        for component_index, component_id in enumerate(component_ids):
            x = int(component_stats[component_id, cv2.CC_STAT_LEFT])
            y = int(component_stats[component_id, cv2.CC_STAT_TOP])
            component_width = int(component_stats[component_id, cv2.CC_STAT_WIDTH])
            component_height = int(component_stats[component_id, cv2.CC_STAT_HEIGHT])
            roi_labels = component_labels[y : y + component_height, x : x + component_width]
            component_roi = roi_labels == int(component_id)
            if distance is not None:
                support_distance = float(
                    np.min(distance[y : y + component_height, x : x + component_width][component_roi])
                )
            else:
                support_distance = abs(float(component_centroids[component_id, 0]) - lane_center)

            proximal_height = max(2, min(32, int(round(0.15 * float(component_height)))))
            proximal_labels = component_labels[y : min(height, y + proximal_height), x : x + component_width]
            proximal_x_local = np.where(proximal_labels == int(component_id))[1]
            if proximal_x_local.size > 0:
                proximal_x = float(x + np.median(proximal_x_local.astype(np.float64)))
            else:
                proximal_x = float(component_centroids[component_id, 0])
            outside_lane = max(float(lane_left) - proximal_x, proximal_x - float(max(lane_left, lane_right - 1)), 0.0)
            center_tiebreak = 0.002 * abs(proximal_x - lane_center)
            scores[component_index, track_index] = support_distance + (0.12 * outside_lane) + center_tiebreak

    ambiguity_mask = np.zeros((height, width), dtype=np.uint8)
    ambiguous_tracks: set[str] = set()
    track_pixels = {track_id: 0 for track_id in track_ids}
    track_components = {track_id: 0 for track_id in track_ids}
    ambiguous_components = 0
    ambiguous_pixels = 0
    close_margin = float(max(8.0, 2.0 * max(0.0, float(ambiguity_margin_px))))
    for component_index, component_id in enumerate(component_ids):
        order = np.argsort(scores[component_index], kind="stable")
        winner_index = int(order[0])
        winner = track_ids[winner_index]
        x = int(component_stats[component_id, cv2.CC_STAT_LEFT])
        y = int(component_stats[component_id, cv2.CC_STAT_TOP])
        component_width = int(component_stats[component_id, cv2.CC_STAT_WIDTH])
        component_height = int(component_stats[component_id, cv2.CC_STAT_HEIGHT])
        component = (
            component_labels[y : y + component_height, x : x + component_width] == int(component_id)
        )
        winner_crop = out[winner][y : y + component_height, x : x + component_width]
        winner_crop[component] = 1
        component_pixels = int(component_stats[component_id, cv2.CC_STAT_AREA])
        track_pixels[winner] += component_pixels
        track_components[winner] += 1

        best_score = float(scores[component_index, winner_index])
        second_score = float(scores[component_index, int(order[1])]) if len(order) > 1 else float("inf")
        if math.isfinite(second_score) and (second_score - best_score) <= close_margin:
            ambiguity_crop = ambiguity_mask[y : y + component_height, x : x + component_width]
            ambiguity_crop[component] = 1
            ambiguous_components += 1
            ambiguous_pixels += component_pixels
            ambiguous_tracks.add(winner)
            ambiguous_tracks.add(track_ids[int(order[1])])

    return out, {
        "enabled": True,
        "components": int(len(component_ids)),
        "pixels": int(unassigned_pixels),
        "ambiguous_components": int(ambiguous_components),
        "ambiguous_pixels": int(ambiguous_pixels),
        "ambiguous_tracks": sorted(ambiguous_tracks),
        "ambiguity_mask": ambiguity_mask,
        "tracks": {
            track_id: {
                "components": int(track_components.get(track_id, 0)),
                "pixels": int(track_pixels.get(track_id, 0)),
                "ambiguous": bool(track_id in ambiguous_tracks),
            }
            for track_id in track_ids
        },
    }


def _filter_shoot_seed_components_by_crown(
    seed_mask: np.ndarray,
    hint: dict[str, object],
    max_distance_px: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Keep only detached shoot components supported by the track crown."""
    seed_u8 = (np.asarray(seed_mask, dtype=np.uint8) > 0).astype(np.uint8)
    empty_meta = {
        "candidate_components": 0,
        "rejected_components": 0,
        "rejected_pixels": 0,
    }
    if int(np.count_nonzero(seed_u8)) <= 0 or float(max_distance_px) <= 0.0:
        return seed_u8, empty_meta
    try:
        crown_x = float(hint.get("crown_center_x", hint.get("lane_center_x")))
        crown_y = float(hint.get("crown_center_y", hint.get("fixed_top_y")))
    except (TypeError, ValueError, OverflowError):
        return seed_u8, empty_meta
    if not math.isfinite(crown_x) or not math.isfinite(crown_y):
        return seed_u8, empty_meta

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        seed_u8,
        connectivity=8,
    )
    keep = np.zeros_like(seed_u8, dtype=np.uint8)
    max_distance_sq = float(max_distance_px) ** 2
    rejected_components = 0
    rejected_pixels = 0
    for component_id in range(1, int(component_count)):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        component = labels[y : y + height, x : x + width] == int(component_id)
        rows, cols = np.where(component)
        if rows.size > 0:
            distance_sq = np.min(
                ((cols.astype(np.float64) + float(x)) - crown_x) ** 2
                + ((rows.astype(np.float64) + float(y)) - crown_y) ** 2
            )
        else:
            distance_sq = float("inf")
        if float(distance_sq) <= max_distance_sq:
            keep_crop = keep[y : y + height, x : x + width]
            keep_crop[component] = 1
        else:
            rejected_components += 1
            rejected_pixels += area
    return keep, {
        "candidate_components": int(component_count - 1),
        "rejected_components": int(rejected_components),
        "rejected_pixels": int(rejected_pixels),
    }


def _effectively_grayscale_u8(image: np.ndarray) -> np.ndarray | None:
    """Return an 8-bit gray view only when the source is genuinely grayscale."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = np.asarray(arr[..., :3], dtype=np.float32)
        sample = rgb[::4, ::4]
        if sample.size <= 0:
            return None
        channel_span = np.max(sample, axis=2) - np.min(sample, axis=2)
        if float(np.percentile(channel_span, 95.0)) > 6.0:
            return None
        gray = np.mean(rgb, axis=2)
    else:
        return None

    gray_float = np.asarray(gray, dtype=np.float32)
    if gray_float.size <= 0 or not np.isfinite(gray_float).any():
        return None
    finite = gray_float[np.isfinite(gray_float)]
    if finite.size <= 0:
        return None
    if float(np.max(finite)) <= 1.0:
        gray_float = gray_float * 255.0
    return np.clip(gray_float, 0.0, 255.0).astype(np.uint8)


def _recover_grayscale_shoot_near_crown(
    image: np.ndarray,
    root_mask: np.ndarray,
    hint: dict[str, object],
    config: AnalyticsConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Recover a missing Hades rosette from dark pixels local to its crown."""
    root_u8 = (np.asarray(root_mask, dtype=np.uint8) > 0).astype(np.uint8)
    shape_hw = root_u8.shape[:2]
    empty = np.zeros(shape_hw, dtype=np.uint8)
    meta: dict[str, object] = {
        "status": "not_attempted",
        "root_support_pixels": 0,
        "candidate_components": 0,
        "accepted_components": 0,
        "accepted_pixels": 0,
    }
    gray = _effectively_grayscale_u8(image)
    if gray is None or gray.shape[:2] != shape_hw:
        meta["status"] = "not_grayscale"
        return empty, meta
    try:
        crown_x = float(hint.get("crown_center_x", hint.get("lane_center_x")))
        crown_y = float(hint.get("crown_center_y", hint.get("fixed_top_y")))
        lane_left = int(hint.get("lane_left", 0))
        lane_right = int(hint.get("lane_right", shape_hw[1]))
    except (TypeError, ValueError, OverflowError):
        meta["status"] = "invalid_crown"
        return empty, meta
    if not math.isfinite(crown_x) or not math.isfinite(crown_y):
        meta["status"] = "invalid_crown"
        return empty, meta

    h, w = shape_hw
    root_radius = int(
        max(
            1,
            getattr(
                config,
                "shoot_tracking_grayscale_crown_rescue_root_radius_px",
                80,
            ),
        )
    )
    crown_support = np.zeros(shape_hw, dtype=np.uint8)
    cv2.circle(
        crown_support,
        (int(round(crown_x)), int(round(crown_y))),
        root_radius,
        1,
        thickness=-1,
    )
    root_support_pixels = int(np.count_nonzero((root_u8 > 0) & (crown_support > 0)))
    meta["root_support_pixels"] = int(root_support_pixels)
    min_root_pixels = int(
        max(
            1,
            getattr(
                config,
                "shoot_tracking_grayscale_crown_rescue_min_root_pixels",
                20,
            ),
        )
    )
    if root_support_pixels < min_root_pixels:
        meta["status"] = "insufficient_root_support"
        return empty, meta

    half_width = int(
        max(
            16,
            getattr(
                config,
                "shoot_tracking_grayscale_crown_rescue_roi_half_width_px",
                220,
            ),
        )
    )
    above = int(
        max(
            16,
            getattr(
                config,
                "shoot_tracking_grayscale_crown_rescue_roi_above_px",
                220,
            ),
        )
    )
    below = int(
        max(
            8,
            getattr(
                config,
                "shoot_tracking_grayscale_crown_rescue_roi_below_px",
                120,
            ),
        )
    )
    x0 = max(0, lane_left, int(math.floor(crown_x - float(half_width))))
    x1 = min(w, lane_right, int(math.ceil(crown_x + float(half_width))))
    y0 = max(0, int(math.floor(crown_y - float(above))))
    y1 = min(h, int(math.ceil(crown_y + float(below))))
    if x1 <= x0 or y1 <= y0:
        meta["status"] = "empty_roi"
        return empty, meta

    roi_gray = gray[y0:y1, x0:x1]
    roi_median = float(np.median(roi_gray))
    threshold_ratio = float(
        max(
            0.05,
            min(
                0.95,
                getattr(
                    config,
                    "shoot_tracking_grayscale_crown_rescue_threshold_ratio",
                    0.56,
                ),
            ),
        )
    )
    threshold_min = int(
        max(
            0,
            min(
                255,
                getattr(
                    config,
                    "shoot_tracking_grayscale_crown_rescue_threshold_min",
                    58,
                ),
            ),
        )
    )
    threshold_max = int(
        max(
            threshold_min,
            min(
                255,
                getattr(
                    config,
                    "shoot_tracking_grayscale_crown_rescue_threshold_max",
                    88,
                ),
            ),
        )
    )
    threshold = int(round(max(threshold_min, min(threshold_max, roi_median * threshold_ratio))))
    candidate_roi = (roi_gray <= np.uint8(threshold)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate_roi = cv2.morphologyEx(candidate_roi, cv2.MORPH_OPEN, kernel, iterations=1)
    candidate_roi = cv2.morphologyEx(candidate_roi, cv2.MORPH_CLOSE, kernel, iterations=1)

    # A dark root is part of the source image too. Remove a small neighborhood
    # around the predicted root so it cannot inflate shoot area or bridge into
    # a bacterial streak below the crown.
    root_exclusion = cv2.dilate(root_u8, kernel, iterations=1)
    candidate_roi[root_exclusion[y0:y1, x0:x1] > 0] = 0

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate_roi,
        connectivity=8,
    )
    min_area = int(
        max(
            1,
            getattr(
                config,
                "shoot_tracking_grayscale_crown_rescue_min_component_area",
                80,
            ),
        )
    )
    max_area = int(
        max(
            min_area,
            getattr(
                config,
                "shoot_tracking_grayscale_crown_rescue_max_component_area",
                28000,
            ),
        )
    )
    max_distance = float(
        max(
            1.0,
            getattr(
                config,
                "shoot_tracking_grayscale_crown_rescue_max_distance_px",
                160.0,
            ),
        )
    )
    max_distance_sq = max_distance**2
    accepted_roi = np.zeros_like(candidate_roi, dtype=np.uint8)
    accepted_components = 0
    for component_id in range(1, int(component_count)):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        component = labels == int(component_id)
        rows, cols = np.where(component)
        if rows.size <= 0:
            continue
        distance_sq = np.min(
            ((cols.astype(np.float64) + float(x0)) - crown_x) ** 2
            + ((rows.astype(np.float64) + float(y0)) - crown_y) ** 2
        )
        component_top = int(stats[component_id, cv2.CC_STAT_TOP]) + y0
        centroid_y = float(centroids[component_id][1]) + float(y0)
        if float(distance_sq) > max_distance_sq:
            continue
        if component_top > int(round(crown_y + 40.0)) or centroid_y > crown_y + 90.0:
            continue
        accepted_roi[component] = 1
        accepted_components += 1

    recovered = np.zeros(shape_hw, dtype=np.uint8)
    recovered[y0:y1, x0:x1] = accepted_roi
    accepted_pixels = int(np.count_nonzero(recovered))
    meta.update(
        {
            "status": "recovered" if accepted_pixels > 0 else "no_candidate",
            "threshold": int(threshold),
            "roi": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
            "candidate_components": int(component_count - 1),
            "accepted_components": int(accepted_components),
            "accepted_pixels": int(accepted_pixels),
        }
    )
    return recovered, meta


def _build_owned_shoot_masks_by_frame(
    shoot_masks: list[np.ndarray],
    track_ids: list[str],
    compartment_hints: dict[str, dict[str, object]],
    config: AnalyticsConfig,
    root_masks: list[np.ndarray] | None = None,
    frame_images: list[np.ndarray] | None = None,
) -> tuple[list[dict[str, np.ndarray]], dict[str, object]]:
    owned_shoot_frames: list[dict[str, np.ndarray]] = []
    if not shoot_masks or not track_ids:
        return owned_shoot_frames, {"enabled": False, "status": "no_shoot_tracking"}

    start_frame = int(max(0, min(len(shoot_masks) - 1, int(getattr(config, "shoot_tracking_start_frame", 6)))))
    motion_radius = int(max(0, min(128, int(getattr(config, "shoot_tracking_motion_radius_px", 56)))))
    requested = bool(getattr(config, "shoot_tracking_enabled", True))
    mode = str(getattr(config, "tracking_mode", "auto") or "auto").strip().lower()
    enabled = bool(requested and mode != "arabidopsis_2dt")

    prev_owned: dict[str, np.ndarray] = {}
    prev_motion_by_track: dict[str, tuple[float, float]] = {}
    tracks_with_pixels: set[str] = set()
    crown_gate_tracks: set[str] = set()
    active_frames = 0
    motion_compensated_prior_frames = 0
    crown_gate_candidate_components = 0
    crown_gate_rejected_components = 0
    crown_gate_rejected_pixels = 0
    grayscale_rescue_attempts = 0
    grayscale_rescue_frames = 0
    grayscale_rescue_pixels = 0
    grayscale_rescue_tracks: set[str] = set()
    grayscale_rescue_frames_by_track: dict[str, list[int]] = {}
    crown_gate_enabled = bool(
        getattr(config, "shoot_tracking_crown_gate_enabled", True)
    )
    crown_gate_max_distance = float(
        max(
            0.0,
            getattr(
                config,
                "shoot_tracking_crown_component_max_distance_px",
                180.0,
            ),
        )
    )

    for frame_idx, shoot_mask_frame in enumerate(shoot_masks):
        shoot_u8 = (np.asarray(shoot_mask_frame, dtype=np.uint8) > 0).astype(np.uint8)
        shape_hw = shoot_u8.shape[:2]
        seed_masks: dict[str, np.ndarray] = {}
        frame_hints: dict[str, dict[str, object]] = {}
        frame_motion_by_track: dict[str, tuple[float, float]] = {}
        frame_used_motion_compensation = False
        for track_id in track_ids:
            resolved_hint = _resolve_track_compartment_hint(
                compartment_hints.get(track_id),
                frame_idx,
                shape_hw,
            )
            hint = dict(resolved_hint) if isinstance(resolved_hint, dict) else None
            seed_mask = np.zeros(shape_hw, dtype=np.uint8)
            if isinstance(hint, dict):
                seed_mask = _seed_mask_for_frame_source(
                    shoot_u8,
                    hint,
                    "shoot_seed_mask",
                    shape_hw,
                    translate_fallback=False,
                )
                if int(np.count_nonzero(seed_mask)) <= 0:
                    seed_mask = _seed_mask_for_frame_source(
                        shoot_u8,
                        hint,
                        "seed_mask",
                        shape_hw,
                        translate_fallback=False,
                    )
                hint["shoot_seed_mask"] = seed_mask
                if bool(hint.get("dynamic_lane_geometry", False)):
                    hint["seed_mask"] = seed_mask
                frame_hints[str(track_id)] = hint
            frame_motion_by_track[str(track_id)] = _hint_motion_xy(hint)
            if enabled and frame_idx >= start_frame:
                tid = str(track_id)
                prev_mask = prev_owned.get(tid)
                if isinstance(prev_mask, np.ndarray) and prev_mask.shape == shape_hw and int(np.count_nonzero(prev_mask)) > 0:
                    current_motion = frame_motion_by_track.get(tid, (0.0, 0.0))
                    previous_motion = prev_motion_by_track.get(tid, current_motion)
                    shift_x = int(round(float(current_motion[0] - previous_motion[0])))
                    shift_y = int(round(float(current_motion[1] - previous_motion[1])))
                    aligned_prev = _translate_binary_mask(prev_mask, shift_x, shift_y, shape_hw)
                    seed_mask = np.maximum(seed_mask, _dilate_binary(aligned_prev, motion_radius))
                    frame_used_motion_compensation = frame_used_motion_compensation or shift_x != 0 or shift_y != 0
            seed_masks[str(track_id)] = (np.asarray(seed_mask, dtype=np.uint8) > 0).astype(np.uint8)

        owned = _assign_pixels_to_track_seeds(shoot_u8, seed_masks, frame_hints)
        if crown_gate_enabled and crown_gate_max_distance > 0.0:
            for track_id in track_ids:
                tid = str(track_id)
                hint = frame_hints.get(tid)
                if not isinstance(hint, dict) or not bool(
                    hint.get("dynamic_lane_geometry", False)
                ):
                    continue
                filtered, gate_meta = _filter_shoot_seed_components_by_crown(
                    owned.get(tid, np.zeros(shape_hw, dtype=np.uint8)),
                    hint,
                    crown_gate_max_distance,
                )
                owned[tid] = filtered
                crown_gate_candidate_components += int(
                    gate_meta.get("candidate_components", 0) or 0
                )
                rejected_components = int(
                    gate_meta.get("rejected_components", 0) or 0
                )
                crown_gate_rejected_components += rejected_components
                crown_gate_rejected_pixels += int(
                    gate_meta.get("rejected_pixels", 0) or 0
                )
                if rejected_components > 0:
                    crown_gate_tracks.add(tid)
        grayscale_rescue_enabled = bool(
            getattr(
                config,
                "shoot_tracking_grayscale_crown_rescue_enabled",
                True,
            )
            and mode == "arabidopsis_crown_lanes"
            and isinstance(root_masks, list)
            and frame_idx < len(root_masks)
            and isinstance(frame_images, list)
            and frame_idx < len(frame_images)
        )
        if grayscale_rescue_enabled:
            root_frame = np.asarray(root_masks[frame_idx], dtype=np.uint8)
            frame_image = np.asarray(frame_images[frame_idx])
            for track_id in track_ids:
                tid = str(track_id)
                current_owned = owned.get(
                    tid,
                    np.zeros(shape_hw, dtype=np.uint8),
                )
                if int(np.count_nonzero(current_owned)) > 0:
                    continue
                hint = frame_hints.get(tid)
                if not isinstance(hint, dict) or not bool(
                    hint.get("dynamic_lane_geometry", False)
                ):
                    continue
                grayscale_rescue_attempts += 1
                recovered, rescue_meta = _recover_grayscale_shoot_near_crown(
                    frame_image,
                    root_frame,
                    hint,
                    config,
                )
                if int(np.count_nonzero(recovered)) <= 0:
                    continue
                owned[tid] = recovered
                grayscale_rescue_frames += 1
                grayscale_rescue_pixels += int(
                    rescue_meta.get("accepted_pixels", 0) or 0
                )
                grayscale_rescue_tracks.add(tid)
                grayscale_rescue_frames_by_track.setdefault(tid, []).append(
                    int(frame_idx)
                )
        if enabled and frame_idx >= start_frame:
            active_frames += 1
        for track_id, mask in owned.items():
            mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
            if int(np.count_nonzero(mask_u8)) > 0:
                tid = str(track_id)
                tracks_with_pixels.add(tid)
                prev_owned[tid] = mask_u8
                prev_motion_by_track[tid] = frame_motion_by_track.get(tid, (0.0, 0.0))
        owned_shoot_frames.append({str(track_id): (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) for track_id, mask in owned.items()})
        if frame_used_motion_compensation:
            motion_compensated_prior_frames += 1

    return owned_shoot_frames, {
        "enabled": bool(enabled),
        "status": (
            "tracked"
            if enabled
            else ("disabled_for_arabidopsis_2dt" if requested and mode == "arabidopsis_2dt" else "disabled")
        ),
        "start_frame": int(start_frame),
        "motion_radius_px": int(motion_radius),
        "tracks_with_pixels": int(len(tracks_with_pixels)),
        "active_frames": int(active_frames),
        "motion_compensated_prior_frames": int(motion_compensated_prior_frames),
        "crown_component_gate_enabled": bool(crown_gate_enabled),
        "crown_component_gate_max_distance_px": float(crown_gate_max_distance),
        "crown_component_gate_candidate_components": int(
            crown_gate_candidate_components
        ),
        "crown_component_gate_rejected_components": int(
            crown_gate_rejected_components
        ),
        "crown_component_gate_rejected_pixels": int(crown_gate_rejected_pixels),
        "crown_component_gate_tracks": sorted(crown_gate_tracks),
        "grayscale_crown_rescue_enabled": bool(
            getattr(
                config,
                "shoot_tracking_grayscale_crown_rescue_enabled",
                True,
            )
        ),
        "grayscale_crown_rescue_attempts": int(grayscale_rescue_attempts),
        "grayscale_crown_rescue_frames": int(grayscale_rescue_frames),
        "grayscale_crown_rescue_pixels": int(grayscale_rescue_pixels),
        "grayscale_crown_rescue_tracks": sorted(grayscale_rescue_tracks),
        "grayscale_crown_rescue_frames_by_track": {
            str(track_id): [int(frame_index) for frame_index in frame_indices]
            for track_id, frame_indices in sorted(
                grayscale_rescue_frames_by_track.items()
            )
        },
    }


def _build_tip_priors_by_frame(
    tip_tracks: dict[str, dict[str, object]],
    frame_count: int,
    track_ids: list[str],
    config: AnalyticsConfig,
) -> dict[int, dict[str, list[Point]]]:
    if frame_count <= 0 or not track_ids or not isinstance(tip_tracks, dict):
        return {}
    valid_tracks = {str(track_id) for track_id in track_ids}
    min_vote_share = float(max(0.0, min(1.0, getattr(config, "tip_track_min_dominant_vote_share", 0.65))))
    priors: dict[int, dict[str, list[Point]]] = {}
    for tip_track in tip_tracks.values():
        if not isinstance(tip_track, dict):
            continue
        dominant_track = str(tip_track.get("dominant_plant_id", "")).strip()
        if dominant_track not in valid_tracks:
            continue
        dominant_vote_share = float(tip_track.get("dominant_vote_share", 0.0) or 0.0)
        if dominant_vote_share < min_vote_share:
            continue
        obs_obj = tip_track.get("observations")
        observations = obs_obj if isinstance(obs_obj, list) else []
        for obs in observations:
            if not isinstance(obs, dict):
                continue
            if str(obs.get("plant_id", "")).strip() != dominant_track:
                continue
            frame_idx = int(obs.get("frame_index", -1))
            if frame_idx < 0 or frame_idx >= frame_count:
                continue
            x = int(obs.get("x", 0))
            y = int(obs.get("y", 0))
            priors.setdefault(frame_idx, {}).setdefault(dominant_track, []).append((x, y))
    for frame_idx, frame_map in priors.items():
        for track_id, pts in list(frame_map.items()):
            deduped = sorted({(int(x), int(y)) for x, y in pts}, key=lambda p: (-int(p[1]), int(p[0])))
            frame_map[track_id] = deduped[:6]
    return priors


def _merge_tip_priors_by_frame(
    primary: dict[int, dict[str, list[Point]]] | None,
    secondary: dict[int, dict[str, list[Point]]] | None,
) -> dict[int, dict[str, list[Point]]]:
    out: dict[int, dict[str, list[Point]]] = {}
    for source in (primary, secondary):
        if not isinstance(source, dict):
            continue
        for frame_idx, frame_map in source.items():
            if not isinstance(frame_map, dict):
                continue
            dst_frame = out.setdefault(int(frame_idx), {})
            for track_id, pts_obj in frame_map.items():
                points = pts_obj if isinstance(pts_obj, list) else []
                dst_points = dst_frame.setdefault(str(track_id), [])
                for pt in points:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        dst_points.append((int(pt[0]), int(pt[1])))
    for frame_idx, frame_map in list(out.items()):
        for track_id, pts in list(frame_map.items()):
            deduped = sorted({(int(x), int(y)) for x, y in pts}, key=lambda p: (-int(p[1]), int(p[0])))
            frame_map[track_id] = deduped[:6]
        if not frame_map:
            out.pop(frame_idx, None)
    return out


def _ordered_learned_owner_crowns(
    track_ids: list[str],
    frame_hints: dict[str, dict[str, object]],
    root_source: np.ndarray,
) -> tuple[list[str], np.ndarray, int] | None:
    if len(track_ids) != 5:
        return None
    shape_hw = np.asarray(root_source).shape[:2]
    candidates: list[tuple[str, float, dict[str, object]]] = []
    for track_id in track_ids:
        tid = str(track_id)
        hint = frame_hints.get(tid)
        if not isinstance(hint, dict):
            return None
        center_x = float(hint.get("lane_center_x", 0.0) or 0.0)
        candidates.append((tid, center_x, hint))
    candidates.sort(key=lambda value: (float(value[1]), str(value[0])))
    if any(
        float(candidates[index][1]) >= float(candidates[index + 1][1])
        for index in range(len(candidates) - 1)
    ):
        return None

    root = (np.asarray(root_source, dtype=np.uint8) > 0)
    crowns: list[tuple[float, float]] = []
    fallback_count = 0
    for _tid, lane_center_x, hint in candidates:
        seed_obj = hint.get("seed_mask")
        seed = (
            (np.asarray(seed_obj, dtype=np.uint8) > 0)
            if isinstance(seed_obj, np.ndarray) and np.asarray(seed_obj).shape == shape_hw
            else np.zeros(shape_hw, dtype=bool)
        )
        ys, xs = np.where(seed & root)
        if xs.size > 0:
            top_y = float(np.quantile(ys.astype(np.float64), 0.015))
            keep = ys <= top_y + 35.0
            if int(np.count_nonzero(keep)) < 2:
                keep = np.ones_like(ys, dtype=bool)
            crowns.append((float(np.median(xs[keep])), float(np.median(ys[keep]))))
            continue
        fallback_count += 1
        crowns.append(
            (
                float(lane_center_x),
                float(max(0, min(shape_hw[0] - 1, int(hint.get("fixed_top_y", 0) or 0)))),
            )
        )
    crown_array = np.asarray(crowns, dtype=np.float64)
    if crown_array.shape != (5, 2) or np.any(~np.isfinite(crown_array)):
        return None
    return [candidate[0] for candidate in candidates], crown_array, int(fallback_count)


def _owner_map_from_track_masks(
    track_masks: dict[str, np.ndarray],
    ordered_track_ids: list[str],
    shape_hw: tuple[int, int],
) -> np.ndarray:
    labels = np.zeros(shape_hw, dtype=np.uint8)
    collision = np.zeros(shape_hw, dtype=bool)
    for slot, track_id in enumerate(ordered_track_ids, start=1):
        mask_obj = track_masks.get(str(track_id))
        if not isinstance(mask_obj, np.ndarray) or mask_obj.shape != shape_hw:
            continue
        mask = np.asarray(mask_obj, dtype=np.uint8) > 0
        collision |= mask & (labels > 0)
        labels[mask & (labels == 0)] = np.uint8(slot)
    labels[collision] = 0
    return labels


def _build_learned_owner_supports(
    learned_owner: np.ndarray,
    learned_margin: np.ndarray,
    ordered_track_ids: list[str],
    root_source: np.ndarray,
    config: AnalyticsConfig,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    shape_hw = np.asarray(root_source).shape[:2]
    root = np.asarray(root_source, dtype=np.uint8) > 0
    learned = np.asarray(learned_owner, dtype=np.uint8)
    margin = np.asarray(learned_margin, dtype=np.float32)
    if learned.shape != shape_hw or margin.shape != shape_hw:
        return {}, {"applied": False, "status": "shape_mismatch"}

    support_margin = float(max(0.0, getattr(config, "learned_owner_support_margin", 0.50)))
    support = root & (learned >= 1) & (learned <= 5) & (margin >= support_margin)
    out: dict[str, np.ndarray] = {
        str(track_id): np.zeros(shape_hw, dtype=np.uint8)
        for track_id in ordered_track_ids
    }
    if int(np.count_nonzero(support)) <= 0:
        return out, {
            "applied": False,
            "status": "no_high_margin_support",
            "support_pixels": 0,
        }
    per_track_support: dict[str, int] = {}
    for slot, track_id in enumerate(ordered_track_ids, start=1):
        tid = str(track_id)
        own_support = support & (learned == slot)
        out[tid][own_support] = 1
        per_track_support[tid] = int(np.count_nonzero(own_support))
    return out, {
        "applied": True,
        "status": "graph_support_ready",
        "support_pixels": int(np.count_nonzero(support)),
        "per_track_support_pixels": per_track_support,
    }


def _summarize_learned_owner_graph_result(
    owned_root: dict[str, np.ndarray],
    learned_owner: np.ndarray,
    learned_margin: np.ndarray,
    ordered_track_ids: list[str],
    root_source: np.ndarray,
    support_meta: dict[str, object],
    root_assignment_meta: dict[str, object],
    config: AnalyticsConfig,
) -> dict[str, object]:
    shape_hw = np.asarray(root_source).shape[:2]
    root = np.asarray(root_source, dtype=np.uint8) > 0
    learned = np.asarray(learned_owner, dtype=np.uint8)
    margin = np.asarray(learned_margin, dtype=np.float32)
    graph_owner = _owner_map_from_track_masks(owned_root, ordered_track_ids, shape_hw)
    if learned.shape != shape_hw or margin.shape != shape_hw:
        return {"applied": False, "status": "shape_mismatch"}
    qc_margin = float(max(0.0, getattr(config, "learned_owner_qc_margin", 0.10)))
    low_confidence = root & (margin < qc_margin)
    graph_assigned = root & (graph_owner > 0)
    graph_agreement = graph_assigned & (graph_owner == learned)
    track_meta_obj = root_assignment_meta.get("tracks")
    track_meta = track_meta_obj if isinstance(track_meta_obj, dict) else {}
    per_track: dict[str, dict[str, object]] = {}
    support_by_track_obj = support_meta.get("per_track_support_pixels")
    support_by_track = (
        support_by_track_obj if isinstance(support_by_track_obj, dict) else {}
    )
    for slot, track_id in enumerate(ordered_track_ids, start=1):
        tid = str(track_id)
        owned = graph_owner == slot
        assigned_pixels = int(np.count_nonzero(owned))
        low_confidence_pixels = int(np.count_nonzero(owned & low_confidence))
        low_confidence_fraction = float(low_confidence_pixels / max(assigned_pixels, 1))
        raw_meta = track_meta.get(tid)
        current = raw_meta if isinstance(raw_meta, dict) else {}
        current["learned_owner_low_confidence_fraction"] = float(low_confidence_fraction)
        current["learned_owner_support_pixels"] = int(support_by_track.get(tid, 0) or 0)
        track_meta[tid] = current
        per_track[tid] = {
            "assigned_pixels": int(assigned_pixels),
            "support_pixels": int(support_by_track.get(tid, 0) or 0),
            "low_confidence_pixels": int(low_confidence_pixels),
            "low_confidence_fraction": float(low_confidence_fraction),
            "ambiguous": bool(current.get("ambiguous", False)),
        }
    root_assignment_meta["tracks"] = track_meta
    return {
        "applied": bool(support_meta.get("applied", False)),
        "status": (
            "graph_fused"
            if bool(support_meta.get("applied", False))
            else str(support_meta.get("status", "no_support"))
        ),
        "support_pixels": int(support_meta.get("support_pixels", 0) or 0),
        "graph_agreement_pixels": int(np.count_nonzero(graph_agreement)),
        "graph_assigned_pixels": int(np.count_nonzero(graph_assigned)),
        "low_confidence_pixels": int(np.count_nonzero(low_confidence)),
        "per_track": per_track,
    }


def _build_owned_masks_by_frame(
    root_masks: list[np.ndarray],
    lateral_masks: list[np.ndarray],
    owned_shoot_frames: list[dict[str, np.ndarray]] | None,
    track_ids: list[str],
    compartment_hints: dict[str, dict[str, object]],
    config: AnalyticsConfig,
    tip_priors_by_frame: dict[int, dict[str, list[Point]]] | None = None,
    compact: bool = False,
    root_seed_masks: list[np.ndarray] | None = None,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]], dict[str, object]]:
    owned_root_frames: list[dict[str, np.ndarray]] = []
    owned_lateral_frames: list[dict[str, np.ndarray]] = []
    tracks_with_tip_priors: set[str] = set()
    tracks_with_temporal_seeds: set[str] = set()
    shoot_block_frames = 0
    temporal_seed_frames = 0
    shoot_exclusion_radius = int(max(0, min(128, int(getattr(config, "shoot_tracking_exclusion_radius_px", 10)))))
    shoot_tracking_start = int(max(0, int(getattr(config, "shoot_tracking_start_frame", 6))))
    temporal_seed_enabled = bool(getattr(config, "root_temporal_seed_enabled", True))
    temporal_seed_radius = int(max(0, min(128, int(getattr(config, "root_temporal_seed_radius_px", 18)))))
    prev_tip_points_by_track: dict[str, Point] = {}
    prev_owned_root_by_track: dict[str, np.ndarray] = {}
    prev_state_motion_by_track: dict[str, tuple[float, float]] = {}
    previous_mask_seed_radius = int(max(1, min(8, temporal_seed_radius if temporal_seed_radius > 0 else 1)))
    tracks_with_previous_mask_seeds: set[str] = set()
    tracks_with_previous_exact_seed_reserve: set[str] = set()
    tracks_with_assignment_priors: set[str] = set()
    previous_mask_seed_frames = 0
    previous_exact_seed_reserve_frames = 0
    assignment_prior_frames = 0
    motion_compensated_prior_frames = 0
    motion_compensated_prior_tracks: set[str] = set()
    fallback_recovered_tracks = 0
    fallback_suppressed_tracks = 0
    detached_recovery_components = 0
    detached_recovery_pixels = 0
    detached_recovery_ambiguous_components = 0
    detached_recovery_ambiguous_pixels = 0
    unanchored_seed_frames = 0
    unanchored_seed_tracks: set[str] = set()
    assignment_prior_weight = float(
        max(0.0, min(128.0, float(getattr(config, "root_previous_mask_assignment_weight_px", 2.0))))
    )
    assignment_mode = str(
        getattr(config, "root_ownership_assignment_mode", "temporal_graph") or "temporal_graph"
    ).strip().lower()
    if assignment_mode not in {"temporal_graph", "euclidean"}:
        assignment_mode = "temporal_graph"
    ambiguity_margin = float(
        max(0.0, min(64.0, float(getattr(config, "root_ownership_ambiguity_margin_px", 2.5))))
    )
    freeze_ambiguous_state = bool(getattr(config, "root_ownership_freeze_ambiguous_state", True))
    blocking_ambiguity_min_pixels = int(
        max(
            1,
            min(
                1_000_000,
                int(
                    getattr(
                        config,
                        "root_ownership_blocking_ambiguity_min_pixels",
                        128,
                    )
                ),
            ),
        )
    )
    blocking_ambiguity_min_fraction = float(
        max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        config,
                        "root_ownership_blocking_ambiguity_min_fraction",
                        0.005,
                    )
                ),
            ),
        )
    )
    ambiguity_frames = 0
    blocking_ambiguity_frames = 0
    frozen_state_updates = 0
    ambiguous_tracks_seen: set[str] = set()
    blocking_ambiguous_tracks_seen: set[str] = set()
    assignment_frames: list[dict[str, object]] = []
    learned_owner_requested = bool(getattr(config, "learned_owner_enabled", False))
    learned_owner_ranker: SharedOwnerRanker | None = None
    learned_owner_status = "disabled"
    learned_profile = getattr(config, "learned_owner_ranker_payload", None)
    learned_feature_set = str(
        getattr(config, "learned_owner_feature_set", "crown_coordinates_orientation")
        or "crown_coordinates_orientation"
    )
    if learned_owner_requested:
        learned_owner_status = "invalid_profile"
        try:
            profile = learned_profile if isinstance(learned_profile, dict) else {}
            ranker_payload_obj = profile.get("ranker", profile)
            if not isinstance(ranker_payload_obj, dict):
                raise ValueError("Owner ranker payload is not a mapping.")
            learned_owner_ranker = SharedOwnerRanker.from_payload(ranker_payload_obj)
            learned_feature_set = str(profile.get("feature_set", learned_feature_set) or learned_feature_set)
            learned_owner_status = "ready" if len(track_ids) == 5 else "requires_five_tracks"
        except Exception:
            learned_owner_ranker = None
    learned_owner_active = bool(
        learned_owner_requested
        and learned_owner_ranker is not None
        and len(track_ids) == 5
    )
    learned_owner_frames = 0
    learned_owner_support_pixels = 0
    learned_owner_graph_agreement_pixels = 0
    learned_owner_low_confidence_pixels = 0
    learned_owner_crown_fallbacks = 0
    learned_owner_failed_frames = 0
    measurement_conservation_recovery_frames = 0
    measurement_conservation_recovery_pixels = 0

    for frame_idx, root_mask_frame in enumerate(root_masks):
        shape_hw = root_mask_frame.shape[:2]
        frame_tip_priors = tip_priors_by_frame.get(frame_idx, {}) if isinstance(tip_priors_by_frame, dict) else {}
        frame_owned_shoot = owned_shoot_frames[frame_idx] if isinstance(owned_shoot_frames, list) and frame_idx < len(owned_shoot_frames) else {}
        measurement_root_source_frame = (
            np.asarray(root_mask_frame, dtype=np.uint8) > 0
        ).astype(np.uint8)
        root_source_frame = measurement_root_source_frame.copy()
        if isinstance(root_seed_masks, list) and frame_idx < len(root_seed_masks):
            candidate_seed_source = np.asarray(root_seed_masks[frame_idx], dtype=np.uint8)
            root_seed_source_frame = (
                (candidate_seed_source > 0).astype(np.uint8)
                if candidate_seed_source.shape == root_source_frame.shape
                else root_source_frame.copy()
            )
        else:
            root_seed_source_frame = root_source_frame.copy()
        lateral_source = lateral_masks[frame_idx] if frame_idx < len(lateral_masks) else np.zeros_like(root_mask_frame, dtype=np.uint8)
        lateral_source = (np.asarray(lateral_source, dtype=np.uint8) > 0).astype(np.uint8)
        lateral_source = ((lateral_source > 0) & (root_source_frame > 0)).astype(np.uint8)
        measurement_lateral_source = lateral_source.copy()
        if bool(getattr(config, "shoot_tracking_enabled", True)) and frame_idx >= shoot_tracking_start and isinstance(frame_owned_shoot, dict):
            shoot_union = np.zeros(shape_hw, dtype=np.uint8)
            for mask in frame_owned_shoot.values():
                mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
                if mask_u8.shape != shape_hw or int(np.count_nonzero(mask_u8)) <= 0:
                    continue
                shoot_union = np.maximum(shoot_union, _dilate_binary(mask_u8, shoot_exclusion_radius))
            if int(np.count_nonzero(shoot_union)) > 0:
                root_source_frame[shoot_union > 0] = 0
                root_seed_source_frame[shoot_union > 0] = 0
                lateral_source[shoot_union > 0] = 0
                shoot_block_frames += 1
        root_source_distance = cv2.distanceTransform(np.where(root_source_frame > 0, 0, 1).astype(np.uint8), cv2.DIST_L2, 3)
        seed_masks: dict[str, np.ndarray] = {}
        assignment_prior_masks: dict[str, np.ndarray] = {}
        fallback_root_masks: dict[str, np.ndarray] = {}
        frame_hints: dict[str, dict[str, object]] = {}
        frame_motion_by_track: dict[str, tuple[float, float]] = {}
        for track_id in track_ids:
            resolved_hint = _resolve_track_compartment_hint(
                compartment_hints.get(track_id),
                frame_idx,
                shape_hw,
            )
            if not isinstance(resolved_hint, dict):
                continue
            frame_hint = dict(resolved_hint)
            if bool(frame_hint.get("dynamic_lane_geometry", False)):
                frame_hint["seed_mask"] = _seed_mask_for_frame_source(
                    root_seed_source_frame,
                    frame_hint,
                    "seed_mask",
                    shape_hw,
                )
                frame_hint["_current_seed_mask_resolved"] = True
            frame_hints[str(track_id)] = frame_hint
            frame_motion_by_track[str(track_id)] = _hint_motion_xy(frame_hint)
        frame_used_temporal_seed = False
        frame_used_previous_mask_seed = False
        frame_used_previous_exact_seed_reserve = False
        frame_used_assignment_prior = False
        frame_used_motion_compensation = False
        frame_had_unanchored_seed = False
        for track_id in track_ids:
            tid = str(track_id)
            hint = frame_hints.get(tid, compartment_hints.get(track_id))
            tip_points = list(frame_tip_priors.get(track_id, [])) if isinstance(frame_tip_priors, dict) else []
            if tip_points:
                tracks_with_tip_priors.add(tid)
            current_motion = frame_motion_by_track.get(tid, (0.0, 0.0))
            previous_motion = prev_state_motion_by_track.get(tid, current_motion)
            state_shift_x = int(round(float(current_motion[0] - previous_motion[0])))
            state_shift_y = int(round(float(current_motion[1] - previous_motion[1])))
            prev_tip_raw = prev_tip_points_by_track.get(tid)
            prev_tip = (
                (int(prev_tip_raw[0]) + state_shift_x, int(prev_tip_raw[1]) + state_shift_y)
                if prev_tip_raw is not None
                else None
            )
            if (
                temporal_seed_enabled
                and prev_tip is not None
                and 0 <= int(prev_tip[0]) < shape_hw[1]
                and 0 <= int(prev_tip[1]) < shape_hw[0]
                and float(root_source_distance[int(prev_tip[1]), int(prev_tip[0])]) <= float(max(1, temporal_seed_radius))
            ):
                tip_points.append((int(prev_tip[0]), int(prev_tip[1])))
                tracks_with_temporal_seeds.add(tid)
                frame_used_temporal_seed = True
            connected_root, path_seed = _trace_tip_guided_path_seed(
                root_source_frame,
                hint,
                frame_idx,
                shape_hw,
                config,
                tip_points_xy=tip_points,
            )
            current_seed_obj = hint.get("seed_mask") if isinstance(hint, dict) else None
            current_seed = (
                (np.asarray(current_seed_obj, dtype=np.uint8) > 0).astype(np.uint8)
                if isinstance(current_seed_obj, np.ndarray)
                and np.asarray(current_seed_obj).shape == shape_hw
                else np.zeros(shape_hw, dtype=np.uint8)
            )
            current_seed_touches_root = bool(
                np.any((_dilate_binary(current_seed, 2) > 0) & (root_source_frame > 0))
            )
            previous_owned = prev_owned_root_by_track.get(tid)
            has_previous_owner = bool(
                isinstance(previous_owned, np.ndarray)
                and previous_owned.shape == shape_hw
                and int(np.count_nonzero(previous_owned)) > 0
            )
            if not current_seed_touches_root and not tip_points and not has_previous_owner:
                connected_root = np.zeros(shape_hw, dtype=np.uint8)
                path_seed = np.zeros(shape_hw, dtype=np.uint8)
                unanchored_seed_tracks.add(tid)
                frame_had_unanchored_seed = True
            fallback_root_masks[tid] = connected_root.astype(np.uint8, copy=False)
            if int(np.count_nonzero(path_seed)) <= 0:
                path_seed = connected_root
            if temporal_seed_enabled:
                prev_owned = prev_owned_root_by_track.get(tid)
                if isinstance(prev_owned, np.ndarray) and prev_owned.shape == shape_hw and int(np.count_nonzero(prev_owned)) > 0:
                    aligned_prev_owned = _translate_binary_mask(
                        prev_owned,
                        state_shift_x,
                        state_shift_y,
                        shape_hw,
                    )
                    assignment_prior_masks[tid] = aligned_prev_owned
                    tracks_with_assignment_priors.add(tid)
                    frame_used_assignment_prior = True
                    prior_seed = _dilate_binary(aligned_prev_owned, previous_mask_seed_radius)
                    prior_seed = ((prior_seed > 0) & (root_source_frame > 0)).astype(np.uint8)
                    if int(np.count_nonzero(prior_seed)) > 0:
                        path_seed = np.maximum((np.asarray(path_seed, dtype=np.uint8) > 0).astype(np.uint8), prior_seed)
                        tracks_with_previous_mask_seeds.add(tid)
                        frame_used_previous_mask_seed = True
                    if state_shift_x != 0 or state_shift_y != 0:
                        frame_used_motion_compensation = True
                        motion_compensated_prior_tracks.add(tid)
            seed_masks[tid] = (np.asarray(path_seed, dtype=np.uint8) > 0).astype(np.uint8)

        if temporal_seed_enabled and assignment_prior_masks:
            exact_prior_stack: list[np.ndarray] = []
            prior_track_ids: list[str] = []
            for tid in track_ids:
                prior = assignment_prior_masks.get(str(tid))
                if isinstance(prior, np.ndarray) and prior.shape == shape_hw:
                    exact = ((np.asarray(prior, dtype=np.uint8) > 0) & (root_source_frame > 0)).astype(np.uint8)
                else:
                    exact = np.zeros(shape_hw, dtype=np.uint8)
                exact_prior_stack.append(exact)
                prior_track_ids.append(str(tid))
            if exact_prior_stack:
                prior_counts = np.sum(np.stack(exact_prior_stack, axis=0), axis=0)
                for idx, tid in enumerate(prior_track_ids):
                    own_exact = (exact_prior_stack[idx] > 0) & (prior_counts == 1)
                    if not bool(np.any(own_exact)):
                        continue
                    other_exact = (prior_counts == 1) & (~own_exact)
                    seed = (np.asarray(seed_masks.get(tid, np.zeros(shape_hw, dtype=np.uint8)), dtype=np.uint8) > 0).astype(np.uint8)
                    seed[other_exact] = 0
                    seed[own_exact] = 1
                    seed_masks[tid] = seed.astype(np.uint8, copy=False)
                    tracks_with_previous_exact_seed_reserve.add(tid)
                    frame_used_previous_exact_seed_reserve = True

        learned_owner_candidate: np.ndarray | None = None
        learned_owner_margin: np.ndarray | None = None
        learned_owner_track_order: list[str] = []
        learned_owner_unary_masks: dict[str, np.ndarray] = {}
        learned_owner_support_meta: dict[str, object] = {
            "applied": False,
            "status": "not_attempted",
        }
        learned_owner_frame_meta: dict[str, object] = {
            "applied": False,
            "status": learned_owner_status,
        }
        if learned_owner_active and learned_owner_ranker is not None:
            learned_crowns = _ordered_learned_owner_crowns(
                track_ids,
                frame_hints,
                root_source_frame,
            )
            if learned_crowns is None:
                learned_owner_failed_frames += 1
                learned_owner_frame_meta["status"] = "crown_resolution_failed"
            else:
                learned_owner_track_order, crowns_xy, crown_fallbacks = learned_crowns
                learned_owner_crown_fallbacks += int(crown_fallbacks)
                prior_owner_map = _owner_map_from_track_masks(
                    assignment_prior_masks,
                    learned_owner_track_order,
                    shape_hw,
                )
                try:
                    learned_owner_candidate, learned_owner_margin = predict_owner_map(
                        root_source_frame,
                        lateral_source,
                        crowns_xy,
                        learned_owner_ranker,
                        feature_set=learned_feature_set,
                        chunk_size=max(
                            1,
                            int(getattr(config, "learned_owner_chunk_size", 180_000)),
                        ),
                        prior_owner_map=prior_owner_map,
                        temporal_score_bonus=float(
                            max(
                                0.0,
                                getattr(config, "learned_owner_temporal_score_bonus", 0.18),
                            )
                        ),
                    )
                    learned_owner_frame_meta.update(
                        {
                            "status": "candidate_ready",
                            "crown_fallbacks": int(crown_fallbacks),
                            "crowns_xy": [
                                [float(round(float(x), 3)), float(round(float(y), 3))]
                                for x, y in crowns_xy.tolist()
                            ],
                        }
                    )
                    learned_owner_unary_masks, learned_owner_support_meta = (
                        _build_learned_owner_supports(
                            learned_owner_candidate,
                            learned_owner_margin,
                            learned_owner_track_order,
                            root_source_frame,
                            config,
                        )
                    )
                except Exception:
                    learned_owner_candidate = None
                    learned_owner_margin = None
                    learned_owner_failed_frames += 1
                    learned_owner_frame_meta["status"] = "prediction_failed"

        owned_root_result = _assign_pixels_to_track_seeds(
            root_source_frame,
            seed_masks,
            frame_hints,
            track_prior_masks=assignment_prior_masks,
            prior_weight_px=assignment_prior_weight if temporal_seed_enabled else 0.0,
            track_unary_masks=learned_owner_unary_masks,
            unary_weight_px=float(
                max(
                    0.0,
                    min(
                        128.0,
                        getattr(config, "learned_owner_prior_weight_px", 4.0),
                    ),
                )
            ),
            assignment_mode=assignment_mode,
            ambiguity_margin_px=ambiguity_margin,
            return_metadata=True,
        )
        owned_root, root_assignment_meta = owned_root_result
        detached_recovery_meta: dict[str, object] = {
            "enabled": False,
            "components": 0,
            "pixels": 0,
            "ambiguous_components": 0,
            "ambiguous_pixels": 0,
            "ambiguous_tracks": [],
        }
        if str(getattr(config, "tracking_mode", "auto") or "auto").strip().lower() == "arabidopsis_crown_lanes":
            owned_root, detached_recovery_meta = _recover_unassigned_crown_lane_components(
                measurement_root_source_frame,
                owned_root,
                seed_masks,
                frame_hints,
                track_prior_masks=assignment_prior_masks,
                ambiguity_margin_px=ambiguity_margin,
            )
            restored_measurement_pixels = int(
                np.count_nonzero(
                    (measurement_root_source_frame > 0) & (root_source_frame <= 0)
                )
            )
            if restored_measurement_pixels > 0:
                measurement_conservation_recovery_frames += 1
                measurement_conservation_recovery_pixels += restored_measurement_pixels
            detached_recovery_components += int(detached_recovery_meta.get("components", 0) or 0)
            detached_recovery_pixels += int(detached_recovery_meta.get("pixels", 0) or 0)
            detached_recovery_ambiguous_components += int(
                detached_recovery_meta.get("ambiguous_components", 0) or 0
            )
            detached_recovery_ambiguous_pixels += int(detached_recovery_meta.get("ambiguous_pixels", 0) or 0)

            recovery_ambiguity_mask = detached_recovery_meta.get("ambiguity_mask")
            base_ambiguity_mask = root_assignment_meta.get("ambiguity_mask")
            recovery_added_ambiguity_pixels = 0
            if isinstance(recovery_ambiguity_mask, np.ndarray) and recovery_ambiguity_mask.shape == shape_hw:
                if isinstance(base_ambiguity_mask, np.ndarray) and base_ambiguity_mask.shape == shape_hw:
                    recovery_added_ambiguity_pixels = int(
                        np.count_nonzero(
                            (np.asarray(recovery_ambiguity_mask, dtype=np.uint8) > 0)
                            & (np.asarray(base_ambiguity_mask, dtype=np.uint8) <= 0)
                        )
                    )
                    merged_ambiguity = np.maximum(
                        (np.asarray(base_ambiguity_mask, dtype=np.uint8) > 0).astype(np.uint8),
                        (np.asarray(recovery_ambiguity_mask, dtype=np.uint8) > 0).astype(np.uint8),
                    )
                else:
                    merged_ambiguity = (np.asarray(recovery_ambiguity_mask, dtype=np.uint8) > 0).astype(np.uint8)
                    recovery_added_ambiguity_pixels = int(np.count_nonzero(merged_ambiguity))
                root_assignment_meta["ambiguity_mask"] = merged_ambiguity
                root_assignment_meta["ambiguous_pixels"] = int(np.count_nonzero(merged_ambiguity))

            recovery_tracks = detached_recovery_meta.get("tracks")
            assignment_tracks = root_assignment_meta.get("tracks")
            if not isinstance(assignment_tracks, dict):
                assignment_tracks = {}
                root_assignment_meta["tracks"] = assignment_tracks
            if isinstance(recovery_tracks, dict):
                for track_id, recovery_track_meta in recovery_tracks.items():
                    tid = str(track_id)
                    track_meta = assignment_tracks.setdefault(tid, {})
                    if not isinstance(track_meta, dict) or not isinstance(recovery_track_meta, dict):
                        continue
                    track_meta["assigned_pixels"] = int(
                        np.count_nonzero(owned_root.get(tid, np.zeros(shape_hw, dtype=np.uint8)))
                    )
                    track_meta["detached_recovery_components"] = int(recovery_track_meta.get("components", 0) or 0)
                    track_meta["detached_recovery_pixels"] = int(recovery_track_meta.get("pixels", 0) or 0)
                    if bool(recovery_track_meta.get("ambiguous", False)):
                        track_meta["ambiguous"] = True
                        track_meta["ambiguous_pixels"] = int(
                            track_meta.get("ambiguous_pixels", 0) or 0
                        ) + int(recovery_added_ambiguity_pixels)
        owned_lateral: dict[str, np.ndarray] = {}

        frame_track_meta = root_assignment_meta.get("tracks") if isinstance(root_assignment_meta, dict) else {}
        frame_track_meta = frame_track_meta if isinstance(frame_track_meta, dict) else {}
        frame_ambiguous_tracks = {
            str(track_id)
            for track_id, raw_meta in frame_track_meta.items()
            if isinstance(raw_meta, dict) and bool(raw_meta.get("ambiguous", False))
        }
        frame_track_ambiguity: dict[str, dict[str, object]] = {}
        frame_blocking_ambiguous_tracks: set[str] = set()
        for track_id in track_ids:
            tid = str(track_id)
            raw_track_meta = frame_track_meta.get(tid)
            track_meta = raw_track_meta if isinstance(raw_track_meta, dict) else {}
            assigned_pixels = int(max(0, int(track_meta.get("assigned_pixels", 0) or 0)))
            ambiguous_pixels = int(max(0, int(track_meta.get("ambiguous_pixels", 0) or 0)))
            ambiguous_fraction = float(ambiguous_pixels / max(1, assigned_pixels))
            ambiguous = bool(track_meta.get("ambiguous", False) or ambiguous_pixels > 0)
            blocking = bool(
                ambiguous
                and ambiguous_pixels >= blocking_ambiguity_min_pixels
                and ambiguous_fraction >= blocking_ambiguity_min_fraction
            )
            frame_track_ambiguity[tid] = {
                "assigned_pixels": int(assigned_pixels),
                "ambiguous_pixels": int(ambiguous_pixels),
                "ambiguous_fraction": float(ambiguous_fraction),
                "ambiguous": bool(ambiguous),
                "blocking": bool(blocking),
            }
            if blocking:
                frame_blocking_ambiguous_tracks.add(tid)
        frame_ambiguous_pixels = int(root_assignment_meta.get("ambiguous_pixels", 0) or 0)
        if frame_ambiguous_pixels > 0:
            ambiguity_frames += 1
            ambiguous_tracks_seen.update(frame_ambiguous_tracks)
        if frame_blocking_ambiguous_tracks:
            blocking_ambiguity_frames += 1
            blocking_ambiguous_tracks_seen.update(frame_blocking_ambiguous_tracks)

        for track_id in track_ids:
            tid = str(track_id)
            if int(np.count_nonzero(owned_root.get(tid, np.zeros(shape_hw, dtype=np.uint8)))) <= 0:
                fallback = fallback_root_masks.get(tid)
                if isinstance(fallback, np.ndarray) and fallback.shape == root_mask_frame.shape:
                    occupied = np.zeros(shape_hw, dtype=np.uint8)
                    for other_tid, other_mask in owned_root.items():
                        if str(other_tid) == tid:
                            continue
                        other_u8 = (np.asarray(other_mask, dtype=np.uint8) > 0).astype(np.uint8)
                        if other_u8.shape == shape_hw and int(np.count_nonzero(other_u8)) > 0:
                            occupied = np.maximum(occupied, other_u8)
                    fallback_unique = ((np.asarray(fallback, dtype=np.uint8) > 0) & (occupied <= 0)).astype(np.uint8)
                    if int(np.count_nonzero(fallback_unique)) > 0:
                        owned_root[tid] = fallback_unique.astype(np.uint8, copy=False)
                        fallback_recovered_tracks += 1
                    else:
                        owned_root[tid] = np.zeros(shape_hw, dtype=np.uint8)
                        fallback_suppressed_tracks += 1
            owned_root_mask = (np.asarray(owned_root.get(tid, np.zeros(shape_hw, dtype=np.uint8)), dtype=np.uint8) > 0).astype(np.uint8)
            # Lateral pixels inherit the root-union owner. Assigning the two
            # masks independently can give one physical pixel two seedlings.
            owned_lateral[tid] = (
                (owned_root_mask > 0) & (measurement_lateral_source > 0)
            ).astype(np.uint8)
            state_update_allowed = not (
                freeze_ambiguous_state and tid in frame_blocking_ambiguous_tracks
            )
            if temporal_seed_enabled and int(np.count_nonzero(owned_root_mask)) > 0 and state_update_allowed:
                prev_owned_root_by_track[tid] = owned_root_mask.copy()
                prev_state_motion_by_track[tid] = frame_motion_by_track.get(tid, (0.0, 0.0))
                owned_lateral_mask = (np.asarray(owned_lateral.get(tid, np.zeros(shape_hw, dtype=np.uint8)), dtype=np.uint8) > 0).astype(np.uint8)
                primary_root_mask = _primary_root_view(owned_root_mask, owned_lateral_mask, config)
                tip_point = _infer_distal_tip_point(primary_root_mask)
                union_tip_point = _infer_distal_tip_point(owned_root_mask)
                fallback_gap = int(max(0, int(getattr(config, "primary_root_tracking_fallback_gap_px", 96))))
                if tip_point is None:
                    tip_point = union_tip_point
                elif (
                    union_tip_point is not None
                    and fallback_gap > 0
                    and int(union_tip_point[1]) - int(tip_point[1]) > fallback_gap
                ):
                    tip_point = union_tip_point
                if tip_point is not None:
                    prev_tip_points_by_track[tid] = tip_point
            elif temporal_seed_enabled and int(np.count_nonzero(owned_root_mask)) > 0:
                frozen_state_updates += 1

        final_track_meta_obj = root_assignment_meta.get("tracks")
        final_track_meta = (
            final_track_meta_obj if isinstance(final_track_meta_obj, dict) else {}
        )
        for track_id in track_ids:
            tid = str(track_id)
            current_meta = final_track_meta.setdefault(tid, {})
            if not isinstance(current_meta, dict):
                current_meta = {}
                final_track_meta[tid] = current_meta
            current_meta["assigned_pixels"] = int(
                np.count_nonzero(
                    owned_root.get(tid, np.zeros(shape_hw, dtype=np.uint8))
                )
            )
        root_assignment_meta["tracks"] = final_track_meta

        if (
            learned_owner_candidate is not None
            and learned_owner_margin is not None
            and len(learned_owner_track_order) == 5
        ):
            learned_owner_frame_meta = _summarize_learned_owner_graph_result(
                owned_root,
                learned_owner_candidate,
                learned_owner_margin,
                learned_owner_track_order,
                root_source_frame,
                learned_owner_support_meta,
                root_assignment_meta,
                config,
            )
            if bool(learned_owner_frame_meta.get("applied", False)):
                learned_owner_frames += 1
                learned_owner_support_pixels += int(
                    learned_owner_frame_meta.get("support_pixels", 0) or 0
                )
                learned_owner_graph_agreement_pixels += int(
                    learned_owner_frame_meta.get("graph_agreement_pixels", 0) or 0
                )
                learned_owner_low_confidence_pixels += int(
                    learned_owner_frame_meta.get("low_confidence_pixels", 0) or 0
                )

        assignment_frames.append(
            {
                "frame_index": int(frame_idx),
                "ambiguous_pixels": int(frame_ambiguous_pixels),
                "shared_components": int(root_assignment_meta.get("shared_components", 0) or 0),
                "ambiguous_tracks": sorted(frame_ambiguous_tracks),
                "blocking_ambiguous_tracks": sorted(frame_blocking_ambiguous_tracks),
                "track_ambiguity": frame_track_ambiguity,
                "detached_recovery_components": int(detached_recovery_meta.get("components", 0) or 0),
                "detached_recovery_pixels": int(detached_recovery_meta.get("pixels", 0) or 0),
                "detached_recovery_ambiguous_components": int(
                    detached_recovery_meta.get("ambiguous_components", 0) or 0
                ),
                "learned_owner_applied": bool(
                    learned_owner_frame_meta.get("applied", False)
                ),
                "learned_owner_status": str(
                    learned_owner_frame_meta.get("status", learned_owner_status)
                ),
                "learned_owner_support_pixels": int(
                    learned_owner_frame_meta.get("support_pixels", 0) or 0
                ),
                "learned_owner_graph_agreement_pixels": int(
                    learned_owner_frame_meta.get("graph_agreement_pixels", 0) or 0
                ),
                "learned_owner_low_confidence_pixels": int(
                    learned_owner_frame_meta.get("low_confidence_pixels", 0) or 0
                ),
                "learned_owner_tracks": dict(
                    learned_owner_frame_meta.get("per_track", {})
                )
                if isinstance(learned_owner_frame_meta.get("per_track"), dict)
                else {},
            }
        )

        if compact:
            owned_root_frames.append({str(track_id): _pack_binary_mask(owned_root.get(str(track_id), np.zeros(shape_hw, dtype=np.uint8))) for track_id in track_ids})
            owned_lateral_frames.append(
                {str(track_id): _pack_binary_mask(owned_lateral.get(str(track_id), np.zeros(shape_hw, dtype=np.uint8))) for track_id in track_ids}
            )
        else:
            owned_root_frames.append(owned_root)
            owned_lateral_frames.append(owned_lateral)
        if frame_used_temporal_seed:
            temporal_seed_frames += 1
        if frame_used_previous_mask_seed:
            previous_mask_seed_frames += 1
        if frame_used_previous_exact_seed_reserve:
            previous_exact_seed_reserve_frames += 1
        if frame_used_assignment_prior:
            assignment_prior_frames += 1
        if frame_used_motion_compensation:
            motion_compensated_prior_frames += 1
        if frame_had_unanchored_seed:
            unanchored_seed_frames += 1

    meta = {
        "enabled": True,
        "tracks_with_tip_priors": int(len(tracks_with_tip_priors)),
        "tip_prior_frames": int(len(tip_priors_by_frame or {})),
        "root_temporal_seed_enabled": bool(temporal_seed_enabled),
        "root_temporal_seed_radius_px": int(temporal_seed_radius),
        "tracks_with_temporal_seeds": int(len(tracks_with_temporal_seeds)),
        "root_temporal_seed_frames": int(temporal_seed_frames),
        "root_previous_mask_seed_radius_px": int(previous_mask_seed_radius),
        "tracks_with_previous_mask_seeds": int(len(tracks_with_previous_mask_seeds)),
        "root_previous_mask_seed_frames": int(previous_mask_seed_frames),
        "tracks_with_previous_exact_seed_reserve": int(len(tracks_with_previous_exact_seed_reserve)),
        "root_previous_exact_seed_reserve_frames": int(previous_exact_seed_reserve_frames),
        "root_previous_mask_assignment_weight_px": float(assignment_prior_weight),
        "root_ownership_assignment_mode": str(assignment_mode),
        "root_ownership_ambiguity_margin_px": float(ambiguity_margin),
        "root_ownership_freeze_ambiguous_state": bool(freeze_ambiguous_state),
        "root_ownership_blocking_ambiguity_min_pixels": int(
            blocking_ambiguity_min_pixels
        ),
        "root_ownership_blocking_ambiguity_min_fraction": float(
            blocking_ambiguity_min_fraction
        ),
        "root_ownership_ambiguity_frames": int(ambiguity_frames),
        "root_ownership_ambiguous_tracks": sorted(ambiguous_tracks_seen),
        "root_ownership_blocking_ambiguity_frames": int(
            blocking_ambiguity_frames
        ),
        "root_ownership_blocking_ambiguous_tracks": sorted(
            blocking_ambiguous_tracks_seen
        ),
        "root_ownership_frozen_state_updates": int(frozen_state_updates),
        "root_ownership_assignment_frames": assignment_frames,
        "tracks_with_previous_assignment_priors": int(len(tracks_with_assignment_priors)),
        "root_previous_assignment_prior_frames": int(assignment_prior_frames),
        "root_motion_compensated_prior_frames": int(motion_compensated_prior_frames),
        "root_motion_compensated_prior_tracks": int(len(motion_compensated_prior_tracks)),
        "root_fallback_recovered_tracks": int(fallback_recovered_tracks),
        "root_fallback_suppressed_duplicate_tracks": int(fallback_suppressed_tracks),
        "root_detached_recovery_enabled": bool(
            str(getattr(config, "tracking_mode", "auto") or "auto").strip().lower()
            == "arabidopsis_crown_lanes"
        ),
        "root_detached_recovery_components": int(detached_recovery_components),
        "root_detached_recovery_pixels": int(detached_recovery_pixels),
        "root_detached_recovery_ambiguous_components": int(detached_recovery_ambiguous_components),
        "root_detached_recovery_ambiguous_pixels": int(detached_recovery_ambiguous_pixels),
        "root_measurement_conservation_recovery_frames": int(
            measurement_conservation_recovery_frames
        ),
        "root_measurement_conservation_recovery_pixels": int(
            measurement_conservation_recovery_pixels
        ),
        "root_unanchored_seed_frames": int(unanchored_seed_frames),
        "root_unanchored_seed_tracks": sorted(unanchored_seed_tracks),
        "shoot_block_frames": int(shoot_block_frames),
        "shoot_tracking_start_frame": int(shoot_tracking_start),
        "learned_owner_requested": bool(learned_owner_requested),
        "learned_owner_enabled": bool(learned_owner_active),
        "learned_owner_status": (
            "applied" if learned_owner_frames > 0 else learned_owner_status
        ),
        "learned_owner_feature_set": str(learned_feature_set),
        "learned_owner_frames": int(learned_owner_frames),
        "learned_owner_failed_frames": int(learned_owner_failed_frames),
        "learned_owner_support_pixels": int(learned_owner_support_pixels),
        "learned_owner_graph_agreement_pixels": int(
            learned_owner_graph_agreement_pixels
        ),
        "learned_owner_low_confidence_pixels": int(learned_owner_low_confidence_pixels),
        "learned_owner_crown_fallbacks": int(learned_owner_crown_fallbacks),
        "learned_owner_support_margin": float(
            max(0.0, getattr(config, "learned_owner_support_margin", 0.50))
        ),
        "learned_owner_prior_weight_px": float(
            max(0.0, getattr(config, "learned_owner_prior_weight_px", 4.0))
        ),
        "learned_owner_temporal_score_bonus": float(
            max(0.0, getattr(config, "learned_owner_temporal_score_bonus", 0.18))
        ),
        "learned_owner_qc_margin": float(
            max(0.0, getattr(config, "learned_owner_qc_margin", 0.10))
        ),
    }
    return owned_root_frames, owned_lateral_frames, meta


def _pack_owned_mask_frames(frames: list[dict[str, np.ndarray]]) -> list[dict[str, dict[str, object]]]:
    packed_frames: list[dict[str, dict[str, object]]] = []
    for frame in frames:
        packed_frames.append({str(track_id): _pack_binary_mask(mask) for track_id, mask in frame.items()})
    return packed_frames


def _assemble_payload_from_owned_masks(
    timeline: list[DatasetImageItem],
    owned_root_frames: list[dict[str, dict[str, object]]],
    owned_lateral_frames: list[dict[str, dict[str, object]]],
    owned_shoot_frames: list[dict[str, np.ndarray]] | None,
    anchor_masks: list[np.ndarray],
    track_ids: list[str],
    overlap_frames: dict[str, int | None],
    config: AnalyticsConfig,
    mask_source: dict[str, object],
    anchor_source: dict[str, object],
    shoot_crown_lock_meta: dict[str, object],
    shoot_tracking_meta: dict[str, object],
    compartment_hints: dict[str, dict[str, object]],
    ownership_meta: dict[str, object],
    lateral_mode: str,
    shoot_source: dict[str, object] | None = None,
) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    series_raw: dict[str, list[float]] = {k: [] for k in track_ids}
    series_primary_root_raw: dict[str, list[float]] = {k: [] for k in track_ids}
    series_total_root_raw: dict[str, list[float]] = {k: [] for k in track_ids}
    series_primary_root_weighted_raw: dict[str, list[float]] = {k: [] for k in track_ids}
    series_total_root_weighted_raw: dict[str, list[float]] = {k: [] for k in track_ids}
    series_lateral_count: dict[str, list[int]] = {k: [] for k in track_ids}
    series_lateral_total_length_mm: dict[str, list[float]] = {k: [] for k in track_ids}
    series_lateral_total_weighted_length_mm: dict[str, list[float]] = {k: [] for k in track_ids}
    series_lateral_mean_diameter_mm: dict[str, list[float]] = {k: [] for k in track_ids}
    series_anchor_overlap: dict[str, list[float]] = {k: [] for k in track_ids}
    series_bbox_h: dict[str, list[int]] = {k: [] for k in track_ids}
    series_bbox_bottom: dict[str, list[int]] = {k: [] for k in track_ids}
    series_bbox_centers: dict[str, list[tuple[float, float] | None]] = {k: [] for k in track_ids}
    series_root_area_px: dict[str, list[int]] = {k: [] for k in track_ids}
    series_total_root_area_px: dict[str, list[int]] = {k: [] for k in track_ids}
    series_shoot_area_px: dict[str, list[int]] = {k: [] for k in track_ids}
    series_shoot_centers: dict[str, list[tuple[float, float] | None]] = {k: [] for k in track_ids}
    series_seedling_centers: dict[str, list[tuple[float, float] | None]] = {k: [] for k in track_ids}
    root_masks_by_frame: list[dict[str, dict[str, object]]] = []
    conflict_root_masks_by_frame: list[dict[str, dict[str, object]]] = []
    lateral_masks_by_frame: list[dict[str, dict[str, object]]] = []
    configured_shoot_class_id = _configured_shoot_class_id(config)
    shoot_source = shoot_source if isinstance(shoot_source, dict) else {}
    anchor_selection_mode = str(anchor_source.get("selection_mode", ""))
    shoot_selection_mode = str(shoot_source.get("selection_mode", ""))
    anchor_green_only_enabled = bool(anchor_source.get("rgb_green_only_enabled", False))
    shoot_green_only_enabled = bool(shoot_source.get("rgb_green_only_enabled", False))
    shoot_measurement_source = "rgb_green_only" if shoot_green_only_enabled else (shoot_selection_mode or "mask_class")
    grayscale_rescue_frames_obj = shoot_tracking_meta.get(
        "grayscale_crown_rescue_frames_by_track"
    )
    grayscale_rescue_frames_by_track = (
        {
            str(track_id): {
                int(frame_index)
                for frame_index in frame_indices
                if isinstance(frame_index, (int, np.integer))
            }
            for track_id, frame_indices in grayscale_rescue_frames_obj.items()
            if isinstance(frame_indices, list)
        }
        if isinstance(grayscale_rescue_frames_obj, dict)
        else {}
    )
    assignment_frames_obj = ownership_meta.get("root_ownership_assignment_frames") if isinstance(ownership_meta, dict) else None
    assignment_frames = assignment_frames_obj if isinstance(assignment_frames_obj, list) else []

    for frame_idx, item in enumerate(timeline):
        assignment_frame = (
            assignment_frames[frame_idx]
            if frame_idx < len(assignment_frames) and isinstance(assignment_frames[frame_idx], dict)
            else {}
        )
        ambiguous_track_ids = {
            str(track_id)
            for track_id in assignment_frame.get("ambiguous_tracks", [])
        } if isinstance(assignment_frame.get("ambiguous_tracks"), list) else set()
        blocking_ambiguous_track_ids = {
            str(track_id)
            for track_id in assignment_frame.get("blocking_ambiguous_tracks", [])
        } if isinstance(assignment_frame.get("blocking_ambiguous_tracks"), list) else set()
        track_ambiguity_obj = assignment_frame.get("track_ambiguity")
        track_ambiguity = (
            track_ambiguity_obj
            if isinstance(track_ambiguity_obj, dict)
            else {}
        )
        learned_frame_tracks_obj = assignment_frame.get("learned_owner_tracks")
        learned_frame_tracks = (
            learned_frame_tracks_obj
            if isinstance(learned_frame_tracks_obj, dict)
            else {}
        )
        root_owned_frame = owned_root_frames[frame_idx] if frame_idx < len(owned_root_frames) else {}
        lateral_owned_frame = owned_lateral_frames[frame_idx] if frame_idx < len(owned_lateral_frames) else {}
        anchor_mask_frame = anchor_masks[frame_idx] if frame_idx < len(anchor_masks) else np.zeros_like(item.image[..., 0], dtype=np.uint8)
        fh, fw = item.image.shape[:2]
        frame_tracks: dict[str, dict[str, object]] = {}
        frame_root_masks: dict[str, dict[str, object]] = {}
        frame_conflict_root_masks: dict[str, dict[str, object]] = {}
        frame_lateral_masks: dict[str, dict[str, object]] = {}
        root_full_by_track: dict[str, np.ndarray] = {}
        lateral_full_by_track: dict[str, np.ndarray] = {}
        root_union_full = np.zeros((fh, fw), dtype=np.uint8)
        for track_id in track_ids:
            root_mask = _unpack_binary_mask(root_owned_frame.get(track_id), (fh, fw))
            lateral_mask = _unpack_binary_mask(lateral_owned_frame.get(track_id), (fh, fw))
            lateral_mask = ((lateral_mask > 0) & (root_mask > 0)).astype(np.uint8)
            root_full_by_track[str(track_id)] = root_mask
            lateral_full_by_track[str(track_id)] = lateral_mask
            root_union_full = np.maximum(root_union_full, root_mask)

        global_root_skeleton = _prune_small_skeleton_components(
            _skeletonize(root_union_full),
            config.prune_branch_px,
        )
        owner_label_map = np.zeros((fh, fw), dtype=np.int16)
        owner_class_label_map = np.zeros((fh, fw), dtype=np.int16)
        owner_label_by_track = {str(track_id): idx + 1 for idx, track_id in enumerate(track_ids)}
        for track_id in track_ids:
            tid = str(track_id)
            owner_label = int(owner_label_by_track[tid])
            owned = root_full_by_track[tid] > 0
            owner_label_map[owned] = owner_label
            if lateral_mode == "class_mask":
                lateral_owned = owned & (lateral_full_by_track[tid] > 0)
            else:
                lateral_owned = np.zeros((fh, fw), dtype=bool)
            owner_class_label_map[owned & (~lateral_owned)] = (2 * owner_label) - 1
            owner_class_label_map[lateral_owned] = 2 * owner_label

        owner_weighted_lengths = _skeleton_edge_length_by_label(
            global_root_skeleton,
            owner_label_map,
            len(track_ids),
        )
        owner_class_weighted_lengths = _skeleton_edge_length_by_label(
            global_root_skeleton,
            owner_class_label_map,
            2 * len(track_ids),
        )
        owner_total_skeleton_px: dict[str, float] = {}
        owner_primary_skeleton_px: dict[str, float] = {}
        owner_lateral_skeleton_px: dict[str, float] = {}
        for track_id in track_ids:
            tid = str(track_id)
            owner_label = int(owner_label_by_track[tid])
            owner_skeleton = (global_root_skeleton > 0) & (owner_label_map == owner_label)
            if lateral_mode == "class_mask":
                lateral_skeleton = owner_skeleton & (lateral_full_by_track[tid] > 0)
            else:
                lateral_skeleton = np.zeros((fh, fw), dtype=bool)
            owner_total_skeleton_px[tid] = float(np.count_nonzero(owner_skeleton))
            owner_lateral_skeleton_px[tid] = float(np.count_nonzero(lateral_skeleton))
            owner_primary_skeleton_px[tid] = float(
                owner_total_skeleton_px[tid] - owner_lateral_skeleton_px[tid]
            )

        for track_id in track_ids:
            learned_track_meta_obj = learned_frame_tracks.get(str(track_id))
            learned_track_meta = (
                learned_track_meta_obj
                if isinstance(learned_track_meta_obj, dict)
                else {}
            )
            root_full = root_full_by_track[str(track_id)]
            lateral_full = lateral_full_by_track[str(track_id)]
            owner_label = int(owner_label_by_track[str(track_id)])
            shoot_full = np.zeros((fh, fw), dtype=np.uint8)
            if isinstance(owned_shoot_frames, list) and frame_idx < len(owned_shoot_frames):
                frame_shoot = owned_shoot_frames[frame_idx]
                if isinstance(frame_shoot, dict):
                    shoot_full = (np.asarray(frame_shoot.get(track_id, np.zeros((fh, fw), dtype=np.uint8)), dtype=np.uint8) > 0).astype(np.uint8)
            bbox_mask = root_full.copy()
            if lateral_mode == "class_mask":
                bbox_mask = np.maximum(bbox_mask, lateral_full)
            bbox = _mask_bbox(bbox_mask, config.bbox_padding, (fh, fw))
            shoot_bbox = _mask_bbox(shoot_full, 0, (fh, fw))
            seedling_bbox = _union_valid_bboxes(bbox, shoot_bbox, (fh, fw))
            x, y, bw, bh = bbox
            sx, sy, sbw, sbh = shoot_bbox
            se_x, se_y, se_bw, se_bh = seedling_bbox
            bbox_center = _bbox_center_or_none(bbox)
            shoot_center = _bbox_center_or_none(shoot_bbox)
            seedling_center = _bbox_center_or_none(seedling_bbox)
            tracking_primary_full = _primary_root_view(
                root_full,
                lateral_full if lateral_mode == "class_mask" else None,
                config,
            )
            measurement_primary_full = tracking_primary_full.copy()
            if lateral_mode == "class_mask":
                measurement_primary_full = root_full.copy()
                measurement_primary_full[lateral_full > 0] = 0
            frame_conflict_root_masks[str(track_id)] = _pack_binary_mask(
                measurement_primary_full
            )
            if bw > 0 and bh > 0:
                crop = tracking_primary_full[y : y + bh, x : x + bw]
                primary_measurement_crop = measurement_primary_full[y : y + bh, x : x + bw]
                union_crop = root_full[y : y + bh, x : x + bw]
                owner_skeleton_crop = (
                    (global_root_skeleton[y : y + bh, x : x + bw] > 0)
                    & (owner_label_map[y : y + bh, x : x + bw] == owner_label)
                ).astype(np.uint8)
                lateral_skeleton_crop = (
                    (owner_skeleton_crop > 0)
                    & (lateral_full[y : y + bh, x : x + bw] > 0)
                ).astype(np.uint8)
            else:
                crop = np.zeros((0, 0), dtype=np.uint8)
                primary_measurement_crop = np.zeros((0, 0), dtype=np.uint8)
                union_crop = np.zeros((0, 0), dtype=np.uint8)
                owner_skeleton_crop = np.zeros((0, 0), dtype=np.uint8)
                lateral_skeleton_crop = np.zeros((0, 0), dtype=np.uint8)
            path_base_hint_crop: tuple[float, float] | None = None
            resolved_hint = _resolve_track_compartment_hint(
                compartment_hints.get(str(track_id)),
                frame_idx,
                (fh, fw),
            )
            if isinstance(resolved_hint, dict) and bw > 0 and bh > 0:
                try:
                    hint_x = float(
                        resolved_hint.get(
                            "lane_center_x",
                            float(x) + (0.5 * float(bw)),
                        )
                    )
                    hint_y = float(resolved_hint.get("fixed_top_y", y))
                except (TypeError, ValueError, OverflowError):
                    hint_x = float("nan")
                    hint_y = float("nan")
                if math.isfinite(hint_x) and math.isfinite(hint_y):
                    path_base_hint_crop = (
                        float(hint_x - float(x)),
                        float(hint_y - float(y)),
                    )
            tracking_metrics = _measure_crop(
                crop,
                config,
                base_hint_xy=path_base_hint_crop,
            )
            primary_metrics = _measure_crop(
                primary_measurement_crop,
                config,
                base_hint_xy=path_base_hint_crop,
            )
            union_metrics = _measure_crop(
                union_crop,
                config,
                base_hint_xy=path_base_hint_crop,
            )
            metrics = tracking_metrics
            root_length_measurement_mode = "primary_root"
            if lateral_mode == "class_mask" and bool(getattr(config, "primary_root_tracking_excludes_lateral", True)):
                primary_path = metrics.get("path_xy") if isinstance(metrics, dict) else []
                union_path = union_metrics.get("path_xy") if isinstance(union_metrics, dict) else []
                if isinstance(primary_path, list) and primary_path and isinstance(union_path, list) and union_path:
                    primary_tip_y = int(primary_path[-1][1]) if len(primary_path[-1]) >= 2 else -1
                    union_tip_y = int(union_path[-1][1]) if len(union_path[-1]) >= 2 else -1
                    fallback_gap = int(max(0, int(getattr(config, "primary_root_tracking_fallback_gap_px", 96))))
                    if fallback_gap > 0 and (union_tip_y - primary_tip_y) > fallback_gap:
                        metrics = union_metrics
                        root_length_measurement_mode = "total_root_fallback"
            path_abs = [[int(x + p[0]), int(y + p[1])] for p in metrics["path_xy"]]
            length_px = float(metrics["length_px"])
            length_mm_raw = float(length_px * config.pixel_size_mm)
            anchor_overlap = _component_anchor_overlap({"bbox": bbox}, anchor_mask_frame)
            series_anchor_overlap[track_id].append(float(anchor_overlap))
            series_bbox_h[track_id].append(int(max(0, bh)))
            series_bbox_bottom[track_id].append(int(max(0, y + bh)))
            series_bbox_centers[track_id].append(bbox_center)
            series_root_area_px[track_id].append(int(metrics["area_px"]))
            series_shoot_area_px[track_id].append(int(np.count_nonzero(shoot_full)))
            series_shoot_centers[track_id].append(shoot_center)
            series_seedling_centers[track_id].append(seedling_center)

            if lateral_mode == "class_mask":
                if bw > 0 and bh > 0:
                    lateral_crop = lateral_full[y : y + bh, x : x + bw]
                else:
                    lateral_crop = np.zeros((0, 0), dtype=np.uint8)
            else:
                    lateral_crop = _derive_lateral_mask_from_root(union_crop, union_metrics)
            lateral_segments = _measure_lateral_segments(
                lateral_crop,
                config,
                skeleton_crop=lateral_skeleton_crop if lateral_mode == "class_mask" else None,
            )
            frame_root_masks[track_id] = root_owned_frame.get(track_id, {"bbox": (0, 0, 0, 0), "shape": (0, 0), "bits": np.zeros((0, 0), dtype=np.uint8)})
            frame_lateral_masks[track_id] = lateral_owned_frame.get(track_id, {"bbox": (0, 0, 0, 0), "shape": (0, 0), "bits": np.zeros((0, 0), dtype=np.uint8)})
            lateral_count = int(len(lateral_segments))
            if lateral_mode == "class_mask":
                lateral_total_length_px = float(owner_lateral_skeleton_px[str(track_id)])
                lateral_total_weighted_length_px = float(
                    owner_class_weighted_lengths.get(2 * owner_label, 0.0)
                )
                primary_root_length_px = float(owner_primary_skeleton_px[str(track_id)])
                primary_root_weighted_length_px = float(
                    owner_class_weighted_lengths.get((2 * owner_label) - 1, 0.0)
                )
            else:
                lateral_total_length_px = float(sum(float(seg.get("length_px", 0.0)) for seg in lateral_segments))
                lateral_total_weighted_length_px = float(
                    sum(float(seg.get("weighted_length_px", seg.get("length_px", 0.0))) for seg in lateral_segments)
                )
                primary_root_length_px = float(
                    primary_metrics.get("skeleton_length_px", primary_metrics.get("length_px", 0.0))
                )
                primary_root_weighted_length_px = float(
                    primary_metrics.get("weighted_length_px", primary_root_length_px)
                )
            lateral_total_length_mm = float(lateral_total_length_px * config.pixel_size_mm)
            lateral_total_weighted_length_mm = float(
                lateral_total_weighted_length_px * config.pixel_size_mm
            )
            primary_root_length_mm_raw = float(primary_root_length_px * config.pixel_size_mm)
            primary_root_weighted_length_mm_raw = float(primary_root_weighted_length_px * config.pixel_size_mm)
            primary_root_area_px = int(primary_metrics.get("area_px", 0))
            total_root_length_px = float(owner_total_skeleton_px[str(track_id)])
            total_root_length_mm = float(total_root_length_px * config.pixel_size_mm)
            total_root_weighted_length_px = float(owner_weighted_lengths.get(owner_label, 0.0))
            total_root_weighted_length_mm = float(total_root_weighted_length_px * config.pixel_size_mm)
            total_root_area_px = int(np.count_nonzero(union_crop))
            total_root_area_mm2 = float(float(total_root_area_px) * (float(config.pixel_size_mm) ** 2))
            total_root_measurement_mode = "owned_union_skeleton"
            series_raw[track_id].append(length_mm_raw)
            series_primary_root_raw[track_id].append(primary_root_length_mm_raw)
            series_total_root_raw[track_id].append(total_root_length_mm)
            series_primary_root_weighted_raw[track_id].append(primary_root_weighted_length_mm_raw)
            series_total_root_weighted_raw[track_id].append(total_root_weighted_length_mm)
            series_total_root_area_px[track_id].append(total_root_area_px)
            lateral_mean_length_px = (
                float(lateral_total_length_px / float(max(1, lateral_count))) if lateral_count > 0 else 0.0
            )
            lateral_mean_length_mm = (
                float(lateral_total_length_mm / float(max(1, lateral_count))) if lateral_count > 0 else 0.0
            )
            lateral_diams_px = [
                float(seg.get("diameter_px", 0.0)) for seg in lateral_segments if float(seg.get("diameter_px", 0.0)) > 0.0
            ]
            lateral_mean_diameter_px = float(np.mean(lateral_diams_px)) if lateral_diams_px else 0.0
            lateral_diams = [float(seg.get("diameter_mm", 0.0)) for seg in lateral_segments if float(seg.get("diameter_mm", 0.0)) > 0.0]
            lateral_mean_diameter_mm = float(np.mean(lateral_diams)) if lateral_diams else 0.0
            lateral_max_length_px = (
                float(max(float(seg.get("length_px", 0.0)) for seg in lateral_segments)) if lateral_segments else 0.0
            )
            lateral_max_length_mm = (
                float(max(float(seg.get("length_mm", 0.0)) for seg in lateral_segments)) if lateral_segments else 0.0
            )
            shoot_area_px = int(np.count_nonzero(shoot_full))
            grayscale_crown_rescue_applied = bool(
                frame_idx
                in grayscale_rescue_frames_by_track.get(str(track_id), set())
            )
            row_shoot_measurement_source = (
                "hades_bw_crown_local_cv"
                if grayscale_crown_rescue_applied
                else shoot_measurement_source
            )
            lateral_segments_out = [
                {
                    "segment_id": int(seg.get("segment_id", i + 1)),
                    "length_px": float(round(float(seg.get("length_px", 0.0)), 4)),
                    "length_mm": float(round(float(seg.get("length_mm", 0.0)), 4)),
                    "path_length_px": float(round(float(seg.get("path_length_px", 0.0)), 4)),
                    "path_length_mm": float(round(float(seg.get("path_length_mm", 0.0)), 4)),
                    "weighted_length_px": float(round(float(seg.get("weighted_length_px", seg.get("length_px", 0.0))), 4)),
                    "weighted_length_mm": float(round(float(seg.get("weighted_length_mm", seg.get("length_mm", 0.0))), 4)),
                    "diameter_px": float(round(float(seg.get("diameter_px", 0.0)), 4)),
                    "diameter_mm": float(round(float(seg.get("diameter_mm", 0.0)), 4)),
                    "area_px": int(seg.get("area_px", 0)),
                    "area_mm2": float(round(float(seg.get("area_mm2", 0.0)), 4)),
                    "tips": int(seg.get("tips", 0)),
                    "branches": int(seg.get("branches", 0)),
                }
                for i, seg in enumerate(lateral_segments)
            ]
            series_lateral_count[track_id].append(int(lateral_count))
            series_lateral_total_length_mm[track_id].append(float(lateral_total_length_mm))
            series_lateral_total_weighted_length_mm[track_id].append(float(lateral_total_weighted_length_mm))
            series_lateral_mean_diameter_mm[track_id].append(float(lateral_mean_diameter_mm))
            tip_candidates_abs: list[dict[str, object]] = []
            tip_candidates_obj = metrics.get("tip_candidates")
            if isinstance(tip_candidates_obj, list):
                for tip in tip_candidates_obj:
                    if not isinstance(tip, dict):
                        continue
                    tx = int(x + int(tip.get("x", 0)))
                    ty = int(y + int(tip.get("y", 0)))
                    tx = max(0, min(max(0, fw - 1), tx))
                    ty = max(0, min(max(0, fh - 1), ty))
                    tip_candidates_abs.append(
                        {
                            "x": int(tx),
                            "y": int(ty),
                            "angle_deg": float(tip.get("angle_deg", 0.0)),
                            "distance_from_base_px": float(tip.get("distance_from_base_px", 0.0)),
                            "is_base_endpoint": bool(tip.get("is_base_endpoint", False)),
                        }
                    )

            raw_track_ambiguity = track_ambiguity.get(str(track_id))
            track_ambiguity_meta = (
                raw_track_ambiguity
                if isinstance(raw_track_ambiguity, dict)
                else {}
            )
            row = {
                "uid": item.uid,
                "image_name": item.name,
                "frame_index": int(frame_idx),
                "plant_id": track_id,
                "bbox_x": int(x),
                "bbox_y": int(y),
                "bbox_w": int(bw),
                "bbox_h": int(bh),
                "bbox_center_x": float(round(float(bbox_center[0]), 3)) if bbox_center is not None else None,
                "bbox_center_y": float(round(float(bbox_center[1]), 3)) if bbox_center is not None else None,
                "pixel_size_mm": float(round(float(config.pixel_size_mm), 8)),
                "root_class_id": int(config.root_class_id),
                "lateral_class_id": int(config.lateral_class_id) if config.lateral_class_id is not None else None,
                "seed_class_id": int(config.seed_class_id) if config.seed_class_id is not None else None,
                "shoot_class_id": int(configured_shoot_class_id) if configured_shoot_class_id is not None else None,
                "ownership_anchor_mask_source": anchor_selection_mode,
                "ownership_anchor_rgb_green_only_enabled": bool(anchor_green_only_enabled),
                "ownership_shoot_mask_source": shoot_selection_mode,
                "shoot_measurement_source": row_shoot_measurement_source,
                "shoot_grayscale_crown_rescue_applied": bool(
                    grayscale_crown_rescue_applied
                ),
                "shoot_rgb_green_only_enabled": bool(shoot_green_only_enabled),
                "root_length_measurement_mode": root_length_measurement_mode,
                "root_length_px": float(round(length_px, 4)),
                "root_length_mm_raw": float(round(length_mm_raw, 4)),
                "root_area_px": int(metrics["area_px"]),
                "root_area_mm2": float(round(float(metrics["area_px"]) * (config.pixel_size_mm**2), 4)),
                "root_perimeter_px": float(round(float(metrics["perimeter_px"]), 4)),
                "root_perimeter_mm": float(round(float(metrics["perimeter_px"]) * config.pixel_size_mm, 4)),
                "tips": int(metrics["tips"]),
                "branches": int(metrics["branches"]),
                "base_tip_angle_deg": float(round(float(metrics["base_tip_angle_deg"]), 3)),
                "emergence_angle_deg": float(round(float(metrics["emergence_angle_deg"]), 3)),
                "convex_hull_area_px2": int(metrics["convex_hull_area_px2"]),
                "convex_hull_area_mm2": float(round(float(metrics["convex_hull_area_px2"]) * (config.pixel_size_mm**2), 4)),
                "aspect_ratio": float(round(float(metrics["aspect_ratio"]), 4)),
                "overlap_frame": overlap_frames.get(track_id),
                "path_points": path_abs,
                "tip_candidates": tip_candidates_abs,
                "tip_count_raw": int(len(tip_candidates_abs)),
                "primary_root_length_px": float(round(primary_root_length_px, 4)),
                "primary_root_length_mm_raw": float(round(primary_root_length_mm_raw, 4)),
                "primary_root_length_weighted_px": float(round(primary_root_weighted_length_px, 4)),
                "primary_root_length_weighted_mm_raw": float(round(primary_root_weighted_length_mm_raw, 4)),
                "primary_root_area_px": int(primary_root_area_px),
                "primary_root_area_mm2": float(round(float(primary_root_area_px) * (config.pixel_size_mm**2), 4)),
                "total_root_length_px": float(round(total_root_length_px, 4)),
                "total_root_length_mm_raw": float(round(total_root_length_mm, 4)),
                "total_root_length_weighted_px": float(round(total_root_weighted_length_px, 4)),
                "total_root_length_weighted_mm_raw": float(round(total_root_weighted_length_mm, 4)),
                "total_root_area_px": int(total_root_area_px),
                "total_root_area_mm2": float(round(total_root_area_mm2, 4)),
                "total_root_measurement_mode": total_root_measurement_mode,
                "lateral_count": int(lateral_count),
                "lateral_total_length_px": float(round(lateral_total_length_px, 4)),
                "lateral_total_length_mm": float(round(lateral_total_length_mm, 4)),
                "lateral_total_length_weighted_px": float(round(lateral_total_weighted_length_px, 4)),
                "lateral_total_length_weighted_mm": float(round(lateral_total_weighted_length_mm, 4)),
                "lateral_mean_length_px": float(round(lateral_mean_length_px, 4)),
                "lateral_mean_length_mm": float(round(lateral_mean_length_mm, 4)),
                "lateral_mean_diameter_px": float(round(lateral_mean_diameter_px, 4)),
                "lateral_mean_diameter_mm": float(round(lateral_mean_diameter_mm, 4)),
                "lateral_max_length_px": float(round(lateral_max_length_px, 4)),
                "lateral_max_length_mm": float(round(lateral_max_length_mm, 4)),
                "anchor_overlap": float(round(float(anchor_overlap), 6)),
                "lateral_segments": lateral_segments_out,
                "shoot_area_px": int(shoot_area_px),
                "shoot_area_mm2": float(round(float(shoot_area_px) * (config.pixel_size_mm**2), 4)),
                "shoot_bbox_x": int(sx),
                "shoot_bbox_y": int(sy),
                "shoot_bbox_w": int(sbw),
                "shoot_bbox_h": int(sbh),
                "shoot_center_x": float(round(float(shoot_center[0]), 3)) if shoot_center is not None else None,
                "shoot_center_y": float(round(float(shoot_center[1]), 3)) if shoot_center is not None else None,
                "seedling_bbox_x": int(se_x),
                "seedling_bbox_y": int(se_y),
                "seedling_bbox_w": int(se_bw),
                "seedling_bbox_h": int(se_bh),
                "seedling_center_x": float(round(float(seedling_center[0]), 3)) if seedling_center is not None else None,
                "seedling_center_y": float(round(float(seedling_center[1]), 3)) if seedling_center is not None else None,
                "measurement_tier": "individual",
                "ownership_measurement_valid": True,
                "root_ownership_assignment_mode": str(
                    ownership_meta.get("root_ownership_assignment_mode", "euclidean")
                ),
                "root_ownership_ambiguous": bool(track_id in ambiguous_track_ids),
                "root_ownership_blocking_ambiguous": bool(
                    track_id in blocking_ambiguous_track_ids
                ),
                "root_ownership_ambiguous_pixels": int(
                    track_ambiguity_meta.get("ambiguous_pixels", 0) or 0
                ),
                "root_ownership_ambiguous_fraction": float(
                    track_ambiguity_meta.get("ambiguous_fraction", 0.0) or 0.0
                ),
                "root_ownership_frame_ambiguous_pixels": int(
                    assignment_frame.get("ambiguous_pixels", 0) or 0
                ),
                "root_ownership_shared_components": int(assignment_frame.get("shared_components", 0) or 0),
                "root_ownership_state_frozen": bool(
                    ownership_meta.get("root_ownership_freeze_ambiguous_state", False)
                    and track_id in blocking_ambiguous_track_ids
                ),
                "learned_owner_applied": bool(
                    assignment_frame.get("learned_owner_applied", False)
                ),
                "learned_owner_status": str(
                    assignment_frame.get("learned_owner_status", "")
                ),
                "learned_owner_support_pixels": int(
                    learned_track_meta.get("support_pixels", 0) or 0
                ),
                "learned_owner_graph_agreement_pixels": int(
                    assignment_frame.get("learned_owner_graph_agreement_pixels", 0)
                    or 0
                ),
                "learned_owner_low_confidence_pixels": int(
                    learned_track_meta.get("low_confidence_pixels", 0) or 0
                ),
                "learned_owner_low_confidence_fraction": float(
                    learned_track_meta.get("low_confidence_fraction", 0.0) or 0.0
                ),
                "conflict_group_id": None,
                "conflict_group_members": [],
                "conflict_group_size": 1,
                "conflict_start_frame": None,
                "conflict_touching_now": False,
                "conflict_ever_touched": False,
                "conflict_reacquiring": False,
                "combined_root_length_mm": None,
                "combined_root_length_px": None,
                "combined_area_px": None,
                "combined_area_mm2": None,
            }
            rows.append(row)
            frame_tracks[track_id] = row
        frames.append({"uid": item.uid, "name": item.name, "frame_index": frame_idx, "tracks": frame_tracks, "combined_groups": []})
        root_masks_by_frame.append(frame_root_masks)
        conflict_root_masks_by_frame.append(frame_conflict_root_masks)
        lateral_masks_by_frame.append(frame_lateral_masks)

    frame_count = len(frames)
    series_bbox_center_jump: dict[str, list[float]] = {
        track_id: _center_jump_series(series_bbox_centers.get(track_id, [])) for track_id in track_ids
    }
    series_shoot_center_jump: dict[str, list[float]] = {
        track_id: _center_jump_series(series_shoot_centers.get(track_id, [])) for track_id in track_ids
    }
    series_seedling_center_jump: dict[str, list[float]] = {
        track_id: _center_jump_series(series_seedling_centers.get(track_id, [])) for track_id in track_ids
    }
    tracks_out: dict[str, dict[str, object]] = {}
    for track_id in track_ids:
        raw = series_raw[track_id]
        cleaned = _remove_short_runs(raw, config.persistence_frames)
        if config.enforce_monotonic_growth:
            cleaned = np.maximum.accumulate(np.asarray(cleaned, dtype=np.float64)).tolist()
        primary_raw = series_primary_root_raw[track_id]
        primary_cleaned = _remove_short_runs(primary_raw, config.persistence_frames)
        if config.enforce_monotonic_growth:
            primary_cleaned = np.maximum.accumulate(np.asarray(primary_cleaned, dtype=np.float64)).tolist()
        total_raw = series_total_root_raw[track_id]
        total_cleaned = _remove_short_runs(total_raw, config.persistence_frames)
        if config.enforce_monotonic_growth:
            total_cleaned = np.maximum.accumulate(np.asarray(total_cleaned, dtype=np.float64)).tolist()
        primary_weighted_raw = series_primary_root_weighted_raw[track_id]
        primary_weighted_cleaned = _remove_short_runs(primary_weighted_raw, config.persistence_frames)
        if config.enforce_monotonic_growth:
            primary_weighted_cleaned = np.maximum.accumulate(np.asarray(primary_weighted_cleaned, dtype=np.float64)).tolist()
        total_weighted_raw = series_total_root_weighted_raw[track_id]
        total_weighted_cleaned = _remove_short_runs(total_weighted_raw, config.persistence_frames)
        if config.enforce_monotonic_growth:
            total_weighted_cleaned = np.maximum.accumulate(np.asarray(total_weighted_cleaned, dtype=np.float64)).tolist()
        if cleaned:
            speed = np.diff(np.asarray(cleaned, dtype=np.float64), prepend=cleaned[0]) / float(max(1e-6, config.timestep_hours))
            detrended = speed - _rolling_median(speed, window=5)
            periods = _dominant_periods(detrended, config.timestep_hours, top_n=3)
        else:
            speed = np.asarray([], dtype=np.float64)
            detrended = np.asarray([], dtype=np.float64)
            periods = []
        germ_frame = -1
        for i, v in enumerate(cleaned):
            if float(v) > 0.0:
                germ_frame = i
                break
        tracks_out[track_id] = {
            "length_mm_raw": [float(v) for v in raw],
            "length_mm_clean": [float(v) for v in cleaned],
            "primary_root_length_mm_raw": [float(v) for v in primary_raw],
            "primary_root_length_mm_clean": [float(v) for v in primary_cleaned],
            "total_root_length_mm_raw": [float(v) for v in total_raw],
            "total_root_length_mm_clean": [float(v) for v in total_cleaned],
            "primary_root_length_weighted_mm_raw": [float(v) for v in primary_weighted_raw],
            "primary_root_length_weighted_mm_clean": [float(v) for v in primary_weighted_cleaned],
            "total_root_length_weighted_mm_raw": [float(v) for v in total_weighted_raw],
            "total_root_length_weighted_mm_clean": [float(v) for v in total_weighted_cleaned],
            "growth_speed_mm_h": [float(v) for v in speed.tolist()],
            "detrended_speed_mm_h": [float(v) for v in detrended.tolist()],
            "dominant_period_hours": [float(v) for v in periods],
            "germination_frame": int(germ_frame),
            "overlap_frame": overlap_frames.get(track_id),
            "lateral_count_per_frame": [int(v) for v in series_lateral_count[track_id]],
            "lateral_total_length_mm_per_frame": [float(v) for v in series_lateral_total_length_mm[track_id]],
            "lateral_total_length_weighted_mm_per_frame": [
                float(v) for v in series_lateral_total_weighted_length_mm[track_id]
            ],
            "lateral_mean_diameter_mm_per_frame": [float(v) for v in series_lateral_mean_diameter_mm[track_id]],
            "lateral_count_mean": float(np.mean(series_lateral_count[track_id])) if series_lateral_count[track_id] else 0.0,
            "lateral_total_length_mm_mean": (
                float(np.mean(series_lateral_total_length_mm[track_id])) if series_lateral_total_length_mm[track_id] else 0.0
            ),
            "lateral_total_length_weighted_mm_mean": (
                float(np.mean(series_lateral_total_weighted_length_mm[track_id]))
                if series_lateral_total_weighted_length_mm[track_id]
                else 0.0
            ),
            "lateral_mean_diameter_mm_mean": (
                float(np.mean(series_lateral_mean_diameter_mm[track_id])) if series_lateral_mean_diameter_mm[track_id] else 0.0
            ),
            "anchor_overlap_per_frame": [float(v) for v in series_anchor_overlap[track_id]],
            "anchor_overlap_mean": float(np.mean(series_anchor_overlap[track_id])) if series_anchor_overlap[track_id] else 0.0,
            "anchor_overlap_max": float(np.max(np.asarray(series_anchor_overlap[track_id], dtype=np.float64)))
            if series_anchor_overlap[track_id]
            else 0.0,
            "bbox_height_max_px": int(max(series_bbox_h[track_id])) if series_bbox_h[track_id] else 0,
            "bbox_bottom_max_px": int(max(series_bbox_bottom[track_id])) if series_bbox_bottom[track_id] else 0,
            "bbox_center_jump_px_per_frame": [float(v) for v in series_bbox_center_jump.get(track_id, [])],
            "bbox_center_jump_px_max": (
                float(np.max(np.asarray(series_bbox_center_jump[track_id], dtype=np.float64)))
                if series_bbox_center_jump.get(track_id)
                else 0.0
            ),
            "bbox_center_jump_px_mean": (
                float(np.mean(np.asarray(series_bbox_center_jump[track_id], dtype=np.float64)))
                if series_bbox_center_jump.get(track_id)
                else 0.0
            ),
            "root_area_peak_px": int(max(series_root_area_px[track_id])) if series_root_area_px[track_id] else 0,
            "total_root_area_peak_px": int(max(series_total_root_area_px[track_id])) if series_total_root_area_px[track_id] else 0,
            "shoot_area_peak_px": int(max(series_shoot_area_px[track_id])) if series_shoot_area_px[track_id] else 0,
            "shoot_area_mean_px": float(np.mean(series_shoot_area_px[track_id])) if series_shoot_area_px[track_id] else 0.0,
            "shoot_center_jump_px_per_frame": [float(v) for v in series_shoot_center_jump.get(track_id, [])],
            "shoot_center_jump_px_max": (
                float(np.max(np.asarray(series_shoot_center_jump[track_id], dtype=np.float64)))
                if series_shoot_center_jump.get(track_id)
                else 0.0
            ),
            "shoot_center_jump_px_mean": (
                float(np.mean(np.asarray(series_shoot_center_jump[track_id], dtype=np.float64)))
                if series_shoot_center_jump.get(track_id)
                else 0.0
            ),
            "seedling_center_jump_px_per_frame": [float(v) for v in series_seedling_center_jump.get(track_id, [])],
            "seedling_center_jump_px_max": (
                float(np.max(np.asarray(series_seedling_center_jump[track_id], dtype=np.float64)))
                if series_seedling_center_jump.get(track_id)
                else 0.0
            ),
            "seedling_center_jump_px_mean": (
                float(np.mean(np.asarray(series_seedling_center_jump[track_id], dtype=np.float64)))
                if series_seedling_center_jump.get(track_id)
                else 0.0
            ),
        }

    row_index: dict[tuple[int, str], int] = {}
    for i, row in enumerate(rows):
        row_index[(int(row["frame_index"]), str(row["plant_id"]))] = i
    for track_id in track_ids:
        cleaned = tracks_out[track_id]["length_mm_clean"]
        primary_cleaned = tracks_out[track_id].get("primary_root_length_mm_clean", [])
        total_cleaned = tracks_out[track_id].get("total_root_length_mm_clean", [])
        primary_weighted_cleaned = tracks_out[track_id].get("primary_root_length_weighted_mm_clean", [])
        total_weighted_cleaned = tracks_out[track_id].get("total_root_length_weighted_mm_clean", [])
        lateral_weighted_values = tracks_out[track_id].get("lateral_total_length_weighted_mm_per_frame", [])
        bbox_jumps = series_bbox_center_jump.get(track_id, [])
        shoot_jumps = series_shoot_center_jump.get(track_id, [])
        seedling_jumps = series_seedling_center_jump.get(track_id, [])
        for frame_idx, length_clean in enumerate(cleaned):
            idx = row_index.get((frame_idx, track_id))
            if idx is not None:
                if frame_idx < len(bbox_jumps):
                    rows[idx]["bbox_center_jump_px"] = float(round(float(bbox_jumps[frame_idx]), 4))
                if frame_idx < len(shoot_jumps):
                    rows[idx]["shoot_center_jump_px"] = float(round(float(shoot_jumps[frame_idx]), 4))
                if frame_idx < len(seedling_jumps):
                    rows[idx]["seedling_center_jump_px"] = float(round(float(seedling_jumps[frame_idx]), 4))
                rows[idx]["root_length_mm_clean"] = float(round(float(length_clean), 4))
                rows[idx]["root_length_px_clean"] = float(
                    round(float(length_clean) / max(float(config.pixel_size_mm), 1.0e-9), 4)
                )
                if isinstance(primary_cleaned, list) and frame_idx < len(primary_cleaned):
                    primary_clean_mm = float(primary_cleaned[frame_idx])
                    rows[idx]["primary_root_length_mm_clean"] = float(round(primary_clean_mm, 4))
                    rows[idx]["primary_root_length_px_clean"] = float(
                        round(primary_clean_mm / max(float(config.pixel_size_mm), 1.0e-9), 4)
                    )
                if isinstance(total_cleaned, list) and frame_idx < len(total_cleaned):
                    total_clean_mm = float(total_cleaned[frame_idx])
                    rows[idx]["total_root_length_mm_clean"] = float(round(total_clean_mm, 4))
                    rows[idx]["total_root_length_px_clean"] = float(
                        round(total_clean_mm / max(float(config.pixel_size_mm), 1.0e-9), 4)
                    )
                if isinstance(primary_weighted_cleaned, list) and frame_idx < len(primary_weighted_cleaned):
                    primary_weighted_mm = float(primary_weighted_cleaned[frame_idx])
                    rows[idx]["primary_root_length_weighted_mm_clean"] = float(round(primary_weighted_mm, 4))
                    rows[idx]["primary_root_length_weighted_px_clean"] = float(
                        round(primary_weighted_mm / max(float(config.pixel_size_mm), 1.0e-9), 4)
                    )
                if isinstance(total_weighted_cleaned, list) and frame_idx < len(total_weighted_cleaned):
                    total_weighted_mm = float(total_weighted_cleaned[frame_idx])
                    rows[idx]["total_root_length_weighted_mm_clean"] = float(round(total_weighted_mm, 4))
                    rows[idx]["total_root_length_weighted_px_clean"] = float(
                        round(total_weighted_mm / max(float(config.pixel_size_mm), 1.0e-9), 4)
                    )
                if isinstance(lateral_weighted_values, list) and frame_idx < len(lateral_weighted_values):
                    rows[idx]["lateral_total_length_weighted_mm"] = float(round(float(lateral_weighted_values[frame_idx]), 4))
                    rows[idx]["lateral_root_length_weighted_mm"] = float(round(float(lateral_weighted_values[frame_idx]), 4))

    conflict_tiers = _build_conflict_tier_timeline(
        frames,
        conflict_root_masks_by_frame,
    )
    combined_rows: list[dict[str, object]] = []
    for frame_idx, frame in enumerate(frames):
        frame_meta = (
            conflict_tiers.get("frames", [])[frame_idx]
            if isinstance(conflict_tiers.get("frames"), list) and frame_idx < len(conflict_tiers.get("frames", []))
            else {}
        )
        track_meta = frame_meta.get("tracks") if isinstance(frame_meta, dict) else {}
        group_meta = frame_meta.get("groups") if isinstance(frame_meta, dict) else {}
        track_meta = track_meta if isinstance(track_meta, dict) else {}
        group_meta = group_meta if isinstance(group_meta, list) else []
        tracks_obj = frame.get("tracks")
        frame_tracks = tracks_obj if isinstance(tracks_obj, dict) else {}
        frame_root_masks = root_masks_by_frame[frame_idx] if frame_idx < len(root_masks_by_frame) else {}
        frame_lateral_masks = lateral_masks_by_frame[frame_idx] if frame_idx < len(lateral_masks_by_frame) else {}
        frame_combined_groups: list[dict[str, object]] = []

        for track_id, row in frame_tracks.items():
            if not isinstance(row, dict):
                continue
            meta = track_meta.get(str(track_id), {}) if isinstance(track_meta, dict) else {}
            measurement_tier = str(meta.get("measurement_tier", "individual"))
            ownership_valid = bool(meta.get("ownership_measurement_valid", True))
            members = [str(member) for member in meta.get("conflict_group_members", [])] if isinstance(meta.get("conflict_group_members"), list) else []
            row["measurement_tier"] = measurement_tier
            row["ownership_measurement_valid"] = ownership_valid
            row["conflict_group_id"] = meta.get("conflict_group_id")
            row["conflict_group_members"] = members
            row["conflict_group_size"] = int(meta.get("conflict_group_size", 1))
            row["conflict_start_frame"] = meta.get("conflict_start_frame")
            row["conflict_touching_now"] = bool(meta.get("conflict_touching_now", False))
            row["conflict_ever_touched"] = bool(meta.get("conflict_ever_touched", False))
            row["conflict_reacquiring"] = bool(meta.get("conflict_reacquiring", False))

        for group in group_meta:
            if not isinstance(group, dict):
                continue
            members = [str(member).strip() for member in group.get("members", []) if str(member).strip()]
            if len(members) <= 1:
                continue
            union_root = None
            union_total = None
            for member in members:
                root_bbox, root_crop = _packed_mask_bbox_and_crop(frame_root_masks.get(member))
                lateral_bbox, lateral_crop = _packed_mask_bbox_and_crop(frame_lateral_masks.get(member))
                if root_crop.size <= 0:
                    continue
                if union_root is None:
                    combined_bbox = root_bbox
                    if lateral_crop.size > 0:
                        combined_bbox = _union_bbox(combined_bbox, lateral_bbox, (fh, fw))
                    union_root = np.zeros((combined_bbox[3], combined_bbox[2]), dtype=np.uint8)
                    union_total = np.zeros((combined_bbox[3], combined_bbox[2]), dtype=np.uint8)
                else:
                    combined_bbox = _union_bbox(combined_bbox, root_bbox, (fh, fw))
                    if lateral_crop.size > 0:
                        combined_bbox = _union_bbox(combined_bbox, lateral_bbox, (fh, fw))
                    prev_root = union_root
                    prev_total = union_total
                    union_root = np.zeros((combined_bbox[3], combined_bbox[2]), dtype=np.uint8)
                    union_total = np.zeros((combined_bbox[3], combined_bbox[2]), dtype=np.uint8)
                    px, py, pw, ph = prev_bbox
                    ox = px - combined_bbox[0]
                    oy = py - combined_bbox[1]
                    union_root[oy : oy + ph, ox : ox + pw] = np.maximum(
                        union_root[oy : oy + ph, ox : ox + pw],
                        prev_root,
                    )
                    union_total[oy : oy + ph, ox : ox + pw] = np.maximum(
                        union_total[oy : oy + ph, ox : ox + pw],
                        prev_total,
                    )
                rx, ry, rw, rh = root_bbox
                rox = rx - combined_bbox[0]
                roy = ry - combined_bbox[1]
                union_root[roy : roy + rh, rox : rox + rw] = np.maximum(
                    union_root[roy : roy + rh, rox : rox + rw],
                    root_crop,
                )
                union_total[roy : roy + rh, rox : rox + rw] = np.maximum(
                    union_total[roy : roy + rh, rox : rox + rw],
                    root_crop,
                )
                if lateral_crop.size > 0:
                    lx, ly, lw, lh = lateral_bbox
                    lox = lx - combined_bbox[0]
                    loy = ly - combined_bbox[1]
                    union_total[loy : loy + lh, lox : lox + lw] = np.maximum(
                        union_total[loy : loy + lh, lox : lox + lw],
                        lateral_crop,
                    )
                prev_bbox = combined_bbox
            if union_total is None or union_root is None or int(np.count_nonzero(union_total)) <= 0:
                continue
            combined_bbox = _expand_bbox(prev_bbox, config.bbox_padding, (fh, fw))
            combined_metrics = _measure_total_mask_length(union_total, config)
            group_row = {
                "uid": frame.get("uid"),
                "image_name": frame.get("name"),
                "frame_index": int(frame_idx),
                "group_id": str(group.get("group_id", "")),
                "members": members,
                "group_size": int(len(members)),
                "bbox_x": int(combined_bbox[0]),
                "bbox_y": int(combined_bbox[1]),
                "bbox_w": int(combined_bbox[2]),
                "bbox_h": int(combined_bbox[3]),
                "pixel_size_mm": float(round(float(config.pixel_size_mm), 8)),
                "combined_root_length_mm": float(round(float(combined_metrics.get("length_mm", 0.0)), 4)),
                "combined_root_length_px": float(round(float(combined_metrics.get("length_px", 0.0)), 4)),
                "combined_area_px": int(combined_metrics.get("area_px", 0)),
                "combined_area_mm2": float(
                    round(float(combined_metrics.get("area_px", 0.0)) * (float(config.pixel_size_mm) ** 2), 4)
                ),
                "component_count": int(combined_metrics.get("components", 0)),
                "touching_now": bool(group.get("touching_now", False)),
                "reacquiring": bool(group.get("reacquiring", False)),
                "start_frame": int(group.get("start_frame", frame_idx)),
                "measurement_tier": (
                    "combined_overlap" if bool(group.get("touching_now", False)) else "identity_reacquiring"
                ),
            }
            frame_combined_groups.append(group_row)
            combined_rows.append(dict(group_row))
            for member in members:
                row = frame_tracks.get(member)
                if not isinstance(row, dict):
                    continue
                row["combined_root_length_mm"] = float(group_row["combined_root_length_mm"])
                row["combined_root_length_px"] = float(group_row["combined_root_length_px"])
                row["combined_area_px"] = int(group_row["combined_area_px"])
                row["combined_area_mm2"] = float(group_row["combined_area_mm2"])
                row["combined_group_bbox_x"] = int(group_row["bbox_x"])
                row["combined_group_bbox_y"] = int(group_row["bbox_y"])
                row["combined_group_bbox_w"] = int(group_row["bbox_w"])
                row["combined_group_bbox_h"] = int(group_row["bbox_h"])
        frame["combined_groups"] = frame_combined_groups

    for track_id in track_ids:
        conflict_start_frame = conflict_tiers.get("track_first_conflict_frame", {}).get(track_id)
        measurement_tiers: list[str] = []
        ownership_valid_series: list[bool] = []
        combined_series: list[float] = []
        for frame_idx in range(frame_count):
            idx = row_index.get((frame_idx, track_id))
            row = rows[idx] if idx is not None else None
            if isinstance(row, dict):
                measurement_tiers.append(str(row.get("measurement_tier", "individual")))
                ownership_valid_series.append(bool(row.get("ownership_measurement_valid", True)))
                combined_value = row.get("combined_root_length_mm")
                try:
                    combined_series.append(float(combined_value) if combined_value not in (None, "") else 0.0)
                except Exception:
                    combined_series.append(0.0)
            else:
                measurement_tiers.append("individual")
                ownership_valid_series.append(True)
                combined_series.append(0.0)
        tracks_out[track_id]["measurement_tier_per_frame"] = measurement_tiers
        tracks_out[track_id]["ownership_measurement_valid_per_frame"] = ownership_valid_series
        tracks_out[track_id]["combined_root_length_mm_per_frame"] = combined_series
        tracks_out[track_id]["conflict_start_frame"] = (
            int(conflict_start_frame)
            if isinstance(conflict_start_frame, (int, np.integer))
            else None
        )
        tracks_out[track_id]["combined_conflict_frames"] = int(sum(1 for tier in measurement_tiers if tier == "combined_overlap"))
        tracks_out[track_id]["identity_reacquisition_frames"] = int(
            sum(1 for tier in measurement_tiers if tier == "identity_reacquiring")
        )

    def _quality_gated_clean_series(raw_values: object, valid_flags: list[bool]) -> list[float]:
        raw_list = raw_values if isinstance(raw_values, list) else []
        prepared: list[float] = []
        for frame_idx in range(frame_count):
            try:
                value = float(raw_list[frame_idx]) if frame_idx < len(raw_list) else 0.0
            except Exception:
                value = 0.0
            valid = frame_idx < len(valid_flags) and bool(valid_flags[frame_idx]) and np.isfinite(value)
            prepared.append(float(value) if valid else 0.0)
        filtered = _remove_short_runs(prepared, config.persistence_frames)
        out: list[float] = []
        previous_valid: float | None = None
        for frame_idx, value in enumerate(filtered):
            if frame_idx >= len(valid_flags) or not bool(valid_flags[frame_idx]):
                out.append(float("nan"))
                continue
            current = float(value)
            if bool(config.enforce_monotonic_growth) and previous_valid is not None:
                current = max(float(previous_valid), current)
            previous_valid = float(current)
            out.append(float(current))
        return out

    metric_specs = (
        ("length_mm_raw", "length_mm_clean", "root_length_mm_clean", "root_length_px_clean"),
        (
            "primary_root_length_mm_raw",
            "primary_root_length_mm_clean",
            "primary_root_length_mm_clean",
            "primary_root_length_px_clean",
        ),
        (
            "total_root_length_mm_raw",
            "total_root_length_mm_clean",
            "total_root_length_mm_clean",
            "total_root_length_px_clean",
        ),
        (
            "primary_root_length_weighted_mm_raw",
            "primary_root_length_weighted_mm_clean",
            "primary_root_length_weighted_mm_clean",
            "primary_root_length_weighted_px_clean",
        ),
        (
            "total_root_length_weighted_mm_raw",
            "total_root_length_weighted_mm_clean",
            "total_root_length_weighted_mm_clean",
            "total_root_length_weighted_px_clean",
        ),
    )
    for track_id in track_ids:
        track = tracks_out[track_id]
        validity_obj = track.get("ownership_measurement_valid_per_frame")
        validity = [bool(value) for value in validity_obj] if isinstance(validity_obj, list) else [False] * frame_count
        for raw_key, clean_key, row_mm_key, row_px_key in metric_specs:
            diagnostic_key = f"{clean_key}_diagnostic_pre_qa"
            previous_clean = track.get(clean_key)
            track[diagnostic_key] = list(previous_clean) if isinstance(previous_clean, list) else []
            clean_values = _quality_gated_clean_series(track.get(raw_key), validity)
            track[clean_key] = clean_values
            for frame_idx, clean_value in enumerate(clean_values):
                row_idx = row_index.get((frame_idx, track_id))
                if row_idx is None:
                    continue
                row = rows[row_idx]
                row_diagnostic_key = f"{row_mm_key}_diagnostic_pre_qa"
                if row_diagnostic_key not in row:
                    row[row_diagnostic_key] = row.get(row_mm_key)
                if np.isfinite(float(clean_value)):
                    row[row_mm_key] = float(round(float(clean_value), 4))
                    row[row_px_key] = float(
                        round(float(clean_value) / max(float(config.pixel_size_mm), 1.0e-9), 4)
                    )
                else:
                    row[row_mm_key] = None
                    row[row_px_key] = None

        clean_arr = np.asarray(track.get("length_mm_clean", []), dtype=np.float64)
        speed = np.full(clean_arr.shape, np.nan, dtype=np.float64)
        last_valid_idx: int | None = None
        last_valid_value = 0.0
        for frame_idx, value in enumerate(clean_arr.tolist()):
            if not np.isfinite(float(value)):
                continue
            if last_valid_idx is None:
                speed[frame_idx] = 0.0
            else:
                elapsed_h = float(max(1, frame_idx - last_valid_idx)) * float(max(1.0e-6, config.timestep_hours))
                speed[frame_idx] = float(float(value) - last_valid_value) / elapsed_h
            last_valid_idx = int(frame_idx)
            last_valid_value = float(value)
        speed_for_filter = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
        detrended = speed_for_filter - _rolling_median(speed_for_filter, window=5)
        detrended[~np.isfinite(speed)] = np.nan
        track["growth_speed_mm_h"] = [float(value) for value in speed.tolist()]
        track["detrended_speed_mm_h"] = [float(value) for value in detrended.tolist()]
        track["dominant_period_hours"] = [
            float(value) for value in _dominant_periods(speed_for_filter, config.timestep_hours, top_n=3)
        ]
        track["germination_frame"] = next(
            (idx for idx, value in enumerate(clean_arr.tolist()) if np.isfinite(value) and float(value) > 0.0),
            -1,
        )

        valid_indices = [idx for idx, valid in enumerate(validity) if bool(valid)]

        def _valid_series_values(key: str) -> list[float]:
            values_obj = track.get(key)
            values = values_obj if isinstance(values_obj, list) else []
            out: list[float] = []
            for idx in valid_indices:
                if idx >= len(values):
                    continue
                try:
                    value = float(values[idx])
                except Exception:
                    continue
                if np.isfinite(value):
                    out.append(value)
            return out

        lateral_counts = _valid_series_values("lateral_count_per_frame")
        lateral_lengths = _valid_series_values("lateral_total_length_mm_per_frame")
        lateral_weighted_lengths = _valid_series_values("lateral_total_length_weighted_mm_per_frame")
        lateral_diameters = _valid_series_values("lateral_mean_diameter_mm_per_frame")
        track["lateral_count_mean"] = float(np.mean(lateral_counts)) if lateral_counts else None
        track["lateral_total_length_mm_mean"] = float(np.mean(lateral_lengths)) if lateral_lengths else None
        track["lateral_total_length_weighted_mm_mean"] = (
            float(np.mean(lateral_weighted_lengths)) if lateral_weighted_lengths else None
        )
        track["lateral_mean_diameter_mm_mean"] = float(np.mean(lateral_diameters)) if lateral_diameters else None
        track["ownership_valid_frames"] = int(len(valid_indices))
        track["ownership_valid_fraction"] = float(len(valid_indices) / max(1, frame_count))

    tip_tracks: dict[str, dict[str, object]] = {}
    tip_track_summary = {"observations": 0, "cross_plant_links": 0, "assignment_method": "none"}
    if bool(config.tip_tracking_enabled):
        tip_rows = [row for row in rows if bool(row.get("ownership_measurement_valid", True))]
        tip_tracks, tip_track_summary = _build_tip_tracks(tip_rows, frame_count, config)

    germinated_per_frame = []
    for frame_idx in range(frame_count):
        germinated = 0
        for track_id in track_ids:
            if tracks_out[track_id]["length_mm_clean"][frame_idx] > 0.0:
                germinated += 1
        germinated_per_frame.append(germinated)
    total_plants = max(1, len(track_ids))
    germ_pct = [(100.0 * g / total_plants) for g in germinated_per_frame]
    t50_frame = -1
    for i, p in enumerate(germ_pct):
        if p >= 50.0:
            t50_frame = i
            break
    if len(germ_pct) >= 2:
        grad = np.diff(np.asarray(germ_pct, dtype=np.float64), prepend=germ_pct[0]) / float(max(1e-6, config.timestep_hours))
        tmgr_idx = int(np.argmax(grad))
    else:
        tmgr_idx = -1

    valid_rows = [
        row for row in rows if isinstance(row, dict) and bool(row.get("ownership_measurement_valid", False))
    ]
    lateral_count_values = [int(row.get("lateral_count", 0)) for row in valid_rows]
    lateral_total_length_values = [float(row.get("lateral_total_length_mm", 0.0)) for row in valid_rows]
    lateral_diam_values = [
        float(row.get("lateral_mean_diameter_mm", 0.0))
        for row in valid_rows
        if float(row.get("lateral_mean_diameter_mm", 0.0)) > 0.0
    ]
    primary_root_length_values = [float(row.get("primary_root_length_mm_raw", 0.0)) for row in valid_rows]
    total_root_length_values = [float(row.get("total_root_length_mm_raw", 0.0)) for row in valid_rows]
    total_root_area_values = [int(row.get("total_root_area_px", 0)) for row in valid_rows]
    shoot_area_values = [int(row.get("shoot_area_px", 0)) for row in valid_rows]

    fpca = _compute_fpca({k: v["length_mm_clean"] for k, v in tracks_out.items()}, config.timestep_hours)
    summary = {
        "status": "ok",
        "tracking_mode": config.tracking_mode,
        "tracks": len(track_ids),
        "frames": frame_count,
        "pixel_size_mm": float(config.pixel_size_mm),
        "timestep_hours": float(config.timestep_hours),
        "t50_frame": int(t50_frame),
        "t50_hours": float(t50_frame * config.timestep_hours) if t50_frame >= 0 else None,
        "tmgr_frame": int(tmgr_idx),
        "tmgr_hours": float(tmgr_idx * config.timestep_hours) if tmgr_idx >= 0 else None,
        "germination_percent_per_frame": [float(v) for v in germ_pct],
        "fpca": fpca,
        "mask_source": dict(mask_source),
        "anchor_source": dict(anchor_source),
        "shoot_source": dict(shoot_source or {}),
        "shoot_crown_lock": dict(shoot_crown_lock_meta),
        "shoot_tracking": dict(shoot_tracking_meta),
        "compartmentalized_tracking": {
            "enabled": bool(compartment_hints),
            "lane_partition": bool(getattr(config, "track_lane_partition_enabled", True)),
            "seed_connectivity": bool(getattr(config, "track_seed_connectivity_enabled", True)),
            "tracks_with_hints": int(len(compartment_hints)),
        },
        "tip_guided_ownership": dict(ownership_meta),
        "expected_track_count": int(max(0, int(getattr(config, "expected_track_count", 0)))),
        "lateral_mode": str(lateral_mode),
        "lateral_total_count": int(sum(lateral_count_values)),
        "lateral_mean_count_per_track_frame": float(np.mean(lateral_count_values)) if lateral_count_values else 0.0,
        "lateral_total_length_mm": float(sum(lateral_total_length_values)),
        "lateral_mean_diameter_mm": float(np.mean(lateral_diam_values)) if lateral_diam_values else 0.0,
        "primary_root_total_length_mm": float(sum(primary_root_length_values)),
        "total_root_total_length_mm": float(sum(total_root_length_values)),
        "total_root_area_peak_px": int(max(total_root_area_values)) if total_root_area_values else 0,
        "shoot_area_peak_px": int(max(shoot_area_values)) if shoot_area_values else 0,
        "shoot_area_mean_px": float(np.mean(shoot_area_values)) if shoot_area_values else 0.0,
        "tip_tracking_enabled": bool(config.tip_tracking_enabled),
        "tip_tracks": int(len(tip_tracks)),
        "external_tip_prior_frames": int(len(getattr(config, "external_tip_priors_by_frame", {}) or {})),
        "external_tip_prior_tracks": int(
            len(
                {
                    str(track_id)
                    for frame_map in (getattr(config, "external_tip_priors_by_frame", {}) or {}).values()
                    if isinstance(frame_map, dict)
                    for track_id in frame_map.keys()
                }
            )
        ),
        "tip_track_observations": int(tip_track_summary.get("observations", 0)),
        "tip_cross_plant_links": int(tip_track_summary.get("cross_plant_links", 0)),
        "tip_assignment_method": str(tip_track_summary.get("assignment_method", "none")),
        "tip_vector_speed_mm_day_mean": float(tip_track_summary.get("vector_speed_mm_day_mean", 0.0)),
        "tip_vector_speed_mm_day_p90": float(tip_track_summary.get("vector_speed_mm_day_p90", 0.0)),
        "tip_vector_speed_mm_day_max": float(tip_track_summary.get("vector_speed_mm_day_max", 0.0)),
        "measurement_tiers": {
            "combined_overlap_tracks": int(
                sum(
                    1
                    for track_id in track_ids
                    if tracks_out.get(track_id, {}).get("conflict_start_frame") is not None
                )
            ),
            "combined_overlap_groups": int(len(conflict_tiers.get("groups_seen", []))),
            "combined_overlap_rows": int(sum(1 for row in rows if str(row.get("measurement_tier", "")) == "combined_overlap")),
            "identity_reacquisition_rows": int(
                sum(1 for row in rows if str(row.get("measurement_tier", "")) == "identity_reacquiring")
            ),
        },
    }
    return {
        "frames": frames,
        "rows": rows,
        "combined_rows": combined_rows,
        "tracks": tracks_out,
        "tip_tracks": tip_tracks,
        "summary": summary,
    }


def _assign_tip_links(
    active_states: list[dict[str, object]],
    detections: list[dict[str, object]],
    config: AnalyticsConfig,
) -> tuple[list[tuple[int, int]], str]:
    if not active_states or not detections:
        return [], "none"

    n_rows = len(active_states)
    n_cols = len(detections)
    valid = np.zeros((n_rows, n_cols), dtype=bool)
    cost = np.full((n_rows, n_cols), fill_value=1.0e6, dtype=np.float64)

    for i, state in enumerate(active_states):
        last = state.get("last_detection")
        if not isinstance(last, dict):
            continue
        gap = int(max(1, int(state.get("gap_frames", 1))))
        base_max_dist = float(max(1.0, float(config.tip_track_max_link_distance_px)))
        max_dist = base_max_dist * (1.0 + 0.45 * float(max(0, gap - 1)))
        prev_x = float(last.get("x", 0.0))
        prev_y = float(last.get("y", 0.0))
        prev_angle = float(last.get("angle_deg", 0.0))
        prev_plant_id = str(last.get("plant_id", ""))
        prev_bbox_obj = last.get("bbox", (0, 0, 0, 0))
        prev_bbox = prev_bbox_obj if isinstance(prev_bbox_obj, tuple) else (0, 0, 0, 0)
        if len(prev_bbox) != 4:
            prev_bbox = (0, 0, 0, 0)

        for j, det in enumerate(detections):
            dx = float(det.get("x", 0.0)) - prev_x
            dy = float(det.get("y", 0.0)) - prev_y
            dist = float(np.hypot(dx, dy))
            curr_angle = float(det.get("angle_deg", 0.0))
            ang_term = float(_angular_difference_deg(prev_angle, curr_angle) / 180.0)
            dir_weight = float(max(0.0, min(1.0, config.tip_track_direction_weight)))

            det_plant_id = str(det.get("plant_id", ""))
            plant_penalty = 0.0
            link_max_dist = float(max_dist)
            if det_plant_id != prev_plant_id:
                if not bool(config.tip_track_allow_cross_plant):
                    continue
                det_bbox_obj = det.get("bbox", (0, 0, 0, 0))
                det_bbox = det_bbox_obj if isinstance(det_bbox_obj, tuple) else (0, 0, 0, 0)
                if len(det_bbox) != 4:
                    det_bbox = (0, 0, 0, 0)
                cross_scale = float(max(0.1, min(1.0, getattr(config, "tip_track_cross_plant_max_distance_scale", 0.35))))
                link_max_dist = float(max(1.0, max_dist * cross_scale))
                plant_penalty = (
                    float(max(0.0, getattr(config, "tip_track_cross_plant_bbox_penalty", 0.9)))
                    if _bbox_intersects(prev_bbox, det_bbox)
                    else float(max(0.0, getattr(config, "tip_track_cross_plant_penalty", 1.2)))
                )
            if dist > link_max_dist:
                continue

            dist_term = float(dist / link_max_dist)
            gap_penalty = 0.06 * float(max(0, gap - 1))
            c = ((1.0 - dir_weight) * dist_term) + (dir_weight * ang_term) + plant_penalty + gap_penalty
            if c > float(config.tip_track_max_link_cost):
                continue
            valid[i, j] = True
            cost[i, j] = c

    if not np.any(valid):
        return [], "none"

    assignment: list[tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows.tolist(), cols.tolist()):
            if i < 0 or i >= n_rows or j < 0 or j >= n_cols:
                continue
            if not bool(valid[i, j]):
                continue
            c = float(cost[i, j])
            if math.isfinite(c) and c <= float(config.tip_track_max_link_cost):
                assignment.append((int(i), int(j)))
        return assignment, "hungarian"
    except Exception:
        pass

    # Fallback greedy bipartite matching if SciPy is unavailable.
    candidates: list[tuple[float, int, int]] = []
    for i in range(n_rows):
        for j in range(n_cols):
            if bool(valid[i, j]):
                candidates.append((float(cost[i, j]), int(i), int(j)))
    candidates.sort(key=lambda t: t[0])
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for c, i, j in candidates:
        if i in used_rows or j in used_cols:
            continue
        if not math.isfinite(c) or c > float(config.tip_track_max_link_cost):
            continue
        assignment.append((i, j))
        used_rows.add(i)
        used_cols.add(j)
    return assignment, "greedy"


def _build_tip_tracks(
    rows: list[dict[str, object]],
    frame_count: int,
    config: AnalyticsConfig,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    if frame_count <= 0 or not rows:
        return {}, {"observations": 0, "cross_plant_links": 0, "assignment_method": "none"}

    rows_by_frame: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        frame_idx = int(row.get("frame_index", -1))
        if frame_idx < 0:
            continue
        rows_by_frame.setdefault(frame_idx, []).append(row)
        row["tip_track_ids"] = []
        row["tip_track_count"] = 0
        row["tip_annotations"] = []

    active: dict[str, dict[str, object]] = {}
    tracks_store: dict[str, dict[str, object]] = {}
    next_track_num = 1
    assignment_methods: set[str] = set()
    cross_plant_links = 0
    observations = 0
    link_speeds_mm_day: list[float] = []

    for frame_idx in range(frame_count):
        frame_rows = rows_by_frame.get(frame_idx, [])
        detections: list[dict[str, object]] = []
        for row in frame_rows:
            plant_id = str(row.get("plant_id", ""))
            bbox = (
                int(row.get("bbox_x", 0)),
                int(row.get("bbox_y", 0)),
                int(row.get("bbox_w", 0)),
                int(row.get("bbox_h", 0)),
            )
            tip_candidates_obj = row.get("tip_candidates")
            if not isinstance(tip_candidates_obj, list):
                continue
            for local_idx, tip in enumerate(tip_candidates_obj):
                if not isinstance(tip, dict):
                    continue
                x = int(tip.get("x", 0))
                y = int(tip.get("y", 0))
                angle_deg = float(tip.get("angle_deg", 0.0))
                detections.append(
                    {
                        "frame_index": int(frame_idx),
                        "plant_id": plant_id,
                        "x": int(x),
                        "y": int(y),
                        "angle_deg": float(angle_deg),
                        "local_idx": int(local_idx),
                        "bbox": bbox,
                        "row_ref": row,
                    }
                )

        active_ids = sorted(active.keys())
        active_states: list[dict[str, object]] = []
        for tip_id in active_ids:
            state = active.get(tip_id)
            if not isinstance(state, dict):
                continue
            gap = frame_idx - int(state.get("last_frame", frame_idx - 1))
            state["gap_frames"] = int(max(1, gap))
            if int(gap) > int(config.tip_track_max_gap_frames) + 1:
                continue
            active_states.append({"tip_id": tip_id, **state})

        assignments, method = _assign_tip_links(active_states, detections, config)
        if method != "none":
            assignment_methods.add(method)

        matched_active_idx: set[int] = set()
        matched_det_idx: set[int] = set()

        for active_idx, det_idx in assignments:
            if active_idx < 0 or active_idx >= len(active_states):
                continue
            if det_idx < 0 or det_idx >= len(detections):
                continue
            tip_id = str(active_states[active_idx]["tip_id"])
            det = detections[det_idx]
            track = tracks_store.get(tip_id)
            if not isinstance(track, dict):
                continue
            last_det = track.get("last_detection")
            det_x = int(det.get("x", 0))
            det_y = int(det.get("y", 0))
            det_frame = int(det.get("frame_index", frame_idx))
            speed_mm_h = 0.0
            speed_mm_day = 0.0
            vector_dx_px = 0.0
            vector_dy_px = 0.0
            if isinstance(last_det, dict):
                if str(last_det.get("plant_id", "")) != str(det.get("plant_id", "")):
                    cross_plant_links += 1
                    track["cross_plant_switches"] = int(track.get("cross_plant_switches", 0)) + 1
                prev_x = float(last_det.get("x", det_x))
                prev_y = float(last_det.get("y", det_y))
                prev_frame = int(last_det.get("frame_index", det_frame - 1))
                delta_frames = max(1, int(det_frame - prev_frame))
                vector_dx_px = float(det_x - prev_x)
                vector_dy_px = float(det_y - prev_y)
                dist_px = float(np.hypot(vector_dx_px, vector_dy_px))
                dt_hours = float(max(1.0e-6, delta_frames * float(config.timestep_hours)))
                speed_mm_h = float((dist_px * float(config.pixel_size_mm)) / dt_hours)
                speed_mm_day = float(speed_mm_h * 24.0)
                if speed_mm_day > 0.0 and math.isfinite(speed_mm_day):
                    link_speeds_mm_day.append(float(speed_mm_day))

            bbox_obj = det.get("bbox", (0, 0, 0, 0))
            if isinstance(bbox_obj, (list, tuple)) and len(bbox_obj) >= 4:
                bbox_tuple = (int(bbox_obj[0]), int(bbox_obj[1]), int(bbox_obj[2]), int(bbox_obj[3]))
            else:
                bbox_tuple = (0, 0, 0, 0)
            det_record = {
                "frame_index": int(det_frame),
                "plant_id": str(det.get("plant_id", "")),
                "x": int(det_x),
                "y": int(det_y),
                "angle_deg": float(det.get("angle_deg", 0.0)),
                "bbox": bbox_tuple,
                "speed_mm_h_from_prev": float(speed_mm_h),
                "speed_mm_day_from_prev": float(speed_mm_day),
                "vector_dx_px_from_prev": float(vector_dx_px),
                "vector_dy_px_from_prev": float(vector_dy_px),
            }
            detections_list = track.get("detections")
            if not isinstance(detections_list, list):
                detections_list = []
                track["detections"] = detections_list
            detections_list.append(det_record)
            track["last_detection"] = det_record
            votes = track.get("parent_votes")
            if not isinstance(votes, dict):
                votes = {}
                track["parent_votes"] = votes
            parent_id = str(det.get("plant_id", ""))
            votes[parent_id] = int(votes.get(parent_id, 0)) + 1

            row_ref = det.get("row_ref")
            if isinstance(row_ref, dict):
                tip_ids = row_ref.get("tip_track_ids")
                if not isinstance(tip_ids, list):
                    tip_ids = []
                    row_ref["tip_track_ids"] = tip_ids
                tip_ids.append(tip_id)
                ann = row_ref.get("tip_annotations")
                if not isinstance(ann, list):
                    ann = []
                    row_ref["tip_annotations"] = ann
                ann.append(
                    {
                        "id": tip_id,
                        "x": int(det_record["x"]),
                        "y": int(det_record["y"]),
                        "angle_deg": float(det_record["angle_deg"]),
                        "plant_id": parent_id,
                        "speed_mm_day": float(round(speed_mm_day, 4)),
                        "vector_dx_px": float(round(vector_dx_px, 4)),
                        "vector_dy_px": float(round(vector_dy_px, 4)),
                    }
                )
                observations += 1

            active[tip_id] = {
                "last_detection": det_record,
                "last_frame": int(frame_idx),
            }
            matched_active_idx.add(int(active_idx))
            matched_det_idx.add(int(det_idx))

        # Create new tracks for unmatched detections.
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_idx:
                continue
            tip_id = f"tip_{next_track_num:03d}"
            next_track_num += 1
            bbox_obj = det.get("bbox", (0, 0, 0, 0))
            if isinstance(bbox_obj, (list, tuple)) and len(bbox_obj) >= 4:
                bbox_tuple = (int(bbox_obj[0]), int(bbox_obj[1]), int(bbox_obj[2]), int(bbox_obj[3]))
            else:
                bbox_tuple = (0, 0, 0, 0)
            det_record = {
                "frame_index": int(det.get("frame_index", frame_idx)),
                "plant_id": str(det.get("plant_id", "")),
                "x": int(det.get("x", 0)),
                "y": int(det.get("y", 0)),
                "angle_deg": float(det.get("angle_deg", 0.0)),
                "bbox": bbox_tuple,
                "speed_mm_h_from_prev": 0.0,
                "speed_mm_day_from_prev": 0.0,
                "vector_dx_px_from_prev": 0.0,
                "vector_dy_px_from_prev": 0.0,
            }
            tracks_store[tip_id] = {
                "tip_track_id": tip_id,
                "detections": [det_record],
                "cross_plant_switches": 0,
                "parent_votes": {str(det.get("plant_id", "")): 1},
                "last_detection": det_record,
            }
            active[tip_id] = {
                "last_detection": det_record,
                "last_frame": int(frame_idx),
            }

            row_ref = det.get("row_ref")
            if isinstance(row_ref, dict):
                tip_ids = row_ref.get("tip_track_ids")
                if not isinstance(tip_ids, list):
                    tip_ids = []
                    row_ref["tip_track_ids"] = tip_ids
                tip_ids.append(tip_id)
                ann = row_ref.get("tip_annotations")
                if not isinstance(ann, list):
                    ann = []
                    row_ref["tip_annotations"] = ann
                ann.append(
                    {
                        "id": tip_id,
                        "x": int(det_record["x"]),
                        "y": int(det_record["y"]),
                        "angle_deg": float(det_record["angle_deg"]),
                        "plant_id": str(det_record["plant_id"]),
                        "speed_mm_day": 0.0,
                        "vector_dx_px": 0.0,
                        "vector_dy_px": 0.0,
                    }
                )
                observations += 1

        # Retire stale active tracks.
        to_remove: list[str] = []
        for tip_id, state in active.items():
            last_frame = int(state.get("last_frame", frame_idx))
            if (frame_idx - last_frame) > int(config.tip_track_max_gap_frames):
                to_remove.append(tip_id)
        for tip_id in to_remove:
            active.pop(tip_id, None)

    for row in rows:
        tip_ids_obj = row.get("tip_track_ids")
        tip_ids = tip_ids_obj if isinstance(tip_ids_obj, list) else []
        deduped_ids = sorted({str(tid) for tid in tip_ids})
        row["tip_track_ids"] = deduped_ids
        row["tip_track_count"] = int(len(deduped_ids))
        ann_obj = row.get("tip_annotations")
        ann_list = ann_obj if isinstance(ann_obj, list) else []
        ann_list.sort(key=lambda e: str(e.get("id", "")) if isinstance(e, dict) else "")
        row["tip_annotations"] = ann_list

    tip_tracks_out: dict[str, dict[str, object]] = {}
    min_frames_required = int(max(1, config.tip_track_min_frames))
    for tip_id, track in sorted(tracks_store.items(), key=lambda kv: str(kv[0])):
        detections_obj = track.get("detections")
        detections = detections_obj if isinstance(detections_obj, list) else []
        if len(detections) < min_frames_required:
            continue
        detections = sorted(
            [d for d in detections if isinstance(d, dict)],
            key=lambda d: (int(d.get("frame_index", 0)), int(d.get("x", 0)), int(d.get("y", 0))),
        )
        if not detections:
            continue
        votes_obj = track.get("parent_votes")
        votes = votes_obj if isinstance(votes_obj, dict) else {}
        dominant_parent = max(votes.items(), key=lambda kv: int(kv[1]))[0] if votes else ""
        dominant_votes = int(votes.get(dominant_parent, 0)) if dominant_parent else 0
        total_votes = int(sum(int(v) for v in votes.values())) if votes else 0
        dominant_vote_share = (float(dominant_votes) / float(total_votes)) if total_votes > 0 else 0.0

        traj_len_px = 0.0
        for i in range(1, len(detections)):
            x0 = float(detections[i - 1].get("x", 0.0))
            y0 = float(detections[i - 1].get("y", 0.0))
            x1 = float(detections[i].get("x", 0.0))
            y1 = float(detections[i].get("y", 0.0))
            traj_len_px += float(np.hypot(x1 - x0, y1 - y0))

        frame_start = int(detections[0].get("frame_index", 0))
        frame_end = int(detections[-1].get("frame_index", frame_start))
        frame_span = max(0, frame_end - frame_start)
        traj_len_mm = float(traj_len_px * config.pixel_size_mm)
        velocity_mm_h = traj_len_mm / float(max(1.0e-6, frame_span * config.timestep_hours)) if frame_span > 0 else 0.0

        tip_tracks_out[tip_id] = {
            "tip_track_id": tip_id,
            "dominant_plant_id": str(dominant_parent),
            "dominant_vote_share": float(round(dominant_vote_share, 4)),
            "frames_present": int(len(detections)),
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
            "cross_plant_switches": int(track.get("cross_plant_switches", 0)),
            "parent_votes": {str(k): int(v) for k, v in votes.items()},
            "trajectory_length_px": float(round(traj_len_px, 4)),
            "trajectory_length_mm": float(round(traj_len_mm, 4)),
            "mean_velocity_mm_h": float(round(velocity_mm_h, 4)),
            "observations": detections,
        }

    method_label = "none"
    if "hungarian" in assignment_methods:
        method_label = "hungarian"
    elif "greedy" in assignment_methods:
        method_label = "greedy"

    summary = {
        "observations": int(observations),
        "cross_plant_links": int(cross_plant_links),
        "assignment_method": str(method_label),
        "vector_speed_mm_day_mean": float(np.mean(link_speeds_mm_day)) if link_speeds_mm_day else 0.0,
        "vector_speed_mm_day_p90": float(np.percentile(np.asarray(link_speeds_mm_day, dtype=np.float64), 90.0))
        if link_speeds_mm_day
        else 0.0,
        "vector_speed_mm_day_max": float(np.max(np.asarray(link_speeds_mm_day, dtype=np.float64))) if link_speeds_mm_day else 0.0,
    }
    return tip_tracks_out, summary


def _remove_short_runs(values: list[float], min_run: int) -> list[float]:
    out = np.asarray(values, dtype=np.float64).copy()
    if min_run <= 1:
        return out.tolist()
    pos = out > 0
    i = 0
    n = out.shape[0]
    while i < n:
        if not pos[i]:
            i += 1
            continue
        j = i + 1
        while j < n and pos[j]:
            j += 1
        if (j - i) < min_run:
            out[i:j] = 0.0
        i = j
    return out.tolist()


def _rolling_median(signal: np.ndarray, window: int = 5) -> np.ndarray:
    if signal.size == 0:
        return signal
    window = max(1, int(window))
    half = window // 2
    out = np.zeros_like(signal, dtype=np.float64)
    for i in range(signal.size):
        lo = max(0, i - half)
        hi = min(signal.size, i + half + 1)
        out[i] = float(np.median(signal[lo:hi]))
    return out


def _dominant_periods(signal: np.ndarray, timestep_h: float, top_n: int = 3) -> list[float]:
    if signal.size < 4:
        return []
    centered = signal.astype(np.float64) - float(np.mean(signal))
    fft_vals = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=float(max(1e-6, timestep_h)))
    amps = np.abs(fft_vals)
    amps[freqs <= 1e-9] = 0.0
    order = np.argsort(amps)[::-1]
    periods: list[float] = []
    for idx in order:
        if len(periods) >= top_n:
            break
        f = float(freqs[idx])
        if f <= 1e-9 or float(amps[idx]) <= 1e-9:
            continue
        periods.append(float(1.0 / f))
    return periods


def _compute_fpca(series_by_track: dict[str, list[float]], timestep_h: float, degree: int = 5, n_components: int = 3) -> dict[str, object] | None:
    candidate_keys = sorted(series_by_track.keys())
    if len(candidate_keys) < 2:
        return None
    lengths = {len(series_by_track[key]) for key in candidate_keys}
    if len(lengths) != 1 or not lengths:
        return None
    n_frames = int(next(iter(lengths)))
    x = np.arange(n_frames, dtype=np.float64)
    prepared_rows: list[np.ndarray] = []
    keys: list[str] = []
    for key in candidate_keys:
        row = np.asarray(series_by_track[key], dtype=np.float64)
        finite = np.isfinite(row)
        if int(np.count_nonzero(finite)) < 2:
            continue
        prepared_rows.append(np.interp(x, x[finite], row[finite]))
        keys.append(key)
    if len(keys) < 2:
        return None
    matrix = np.asarray(prepared_rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 4:
        return None
    n_frames = int(matrix.shape[1])
    degree = int(max(1, min(degree, n_frames - 1)))
    t = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    basis = np.vander(t, N=degree + 1, increasing=True)
    coeffs = []
    for row in matrix:
        c, _, _, _ = np.linalg.lstsq(basis, row, rcond=None)
        coeffs.append(c)
    coeff_arr = np.asarray(coeffs, dtype=np.float64)
    centered = coeff_arr - np.mean(coeff_arr, axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    if s.size == 0:
        return None
    n_components = int(max(1, min(n_components, vt.shape[0])))
    comp = vt[:n_components]
    eigen_vals = (s**2) / max(1.0, float(centered.shape[0] - 1))
    ratio = eigen_vals / max(1e-9, float(np.sum(eigen_vals)))
    scores = centered @ comp.T
    comp_curves = basis @ comp.T
    return {
        "track_ids": keys,
        "explained_variance_ratio": [float(r) for r in ratio[:n_components]],
        "scores": {keys[i]: [float(v) for v in scores[i, :]] for i in range(len(keys))},
        "component_curves": [[float(v) for v in comp_curves[:, j]] for j in range(n_components)],
        "time_hours": [float(i * timestep_h) for i in range(n_frames)],
    }


def run_temporal_analytics(
    items: list[DatasetImageItem],
    predictions: dict[str, np.ndarray],
    annotations: dict[str, dict[int, np.ndarray]],
    config: AnalyticsConfig,
    progress_callback: Callable[[int, int], bool] | None = None,
) -> dict[str, object]:
    total_steps = 12

    def _tick(step: int) -> bool:
        if progress_callback is None:
            return True
        return bool(progress_callback(int(max(0, min(step, total_steps))), total_steps))

    def _tracker_progress(_stage: str, fraction: float) -> bool:
        base = 4
        span = 3
        scaled = base + int(round(max(0.0, min(1.0, float(fraction))) * span))
        return _tick(scaled)

    timeline = [item for item in items if item.uid in predictions]
    if not timeline:
        return {"frames": [], "rows": [], "tracks": {}, "tip_tracks": {}, "summary": {"status": "no_predictions"}}

    if not _tick(0):
        return {"frames": [], "rows": [], "tracks": {}, "tip_tracks": {}, "summary": {"status": "cancelled", "frames": 0, "tracks": 0}}
    measurement_root_masks, mask_source = _resolve_tracking_masks(timeline, predictions, config)
    measurement_primary_root_masks: list[np.ndarray] = []
    for frame_idx, item in enumerate(timeline):
        pred_idx = _prediction_to_index_mask(predictions[item.uid])
        if pred_idx.ndim == 2 and pred_idx.shape == measurement_root_masks[frame_idx].shape:
            measurement_primary_root_masks.append(
                (pred_idx == np.uint8(int(config.root_class_id))).astype(np.uint8)
            )
        else:
            measurement_primary_root_masks.append(
                np.zeros_like(measurement_root_masks[frame_idx], dtype=np.uint8)
            )
    primary_seed_available = any(
        int(np.count_nonzero(mask)) > 0 for mask in measurement_primary_root_masks
    )
    if not primary_seed_available:
        measurement_primary_root_masks = [
            np.asarray(mask, dtype=np.uint8).copy() for mask in measurement_root_masks
        ]
    if not _tick(1):
        return {"frames": [], "rows": [], "tracks": {}, "tip_tracks": {}, "summary": {"status": "cancelled", "frames": 0, "tracks": 0}}
    anchor_masks, anchor_source = _resolve_anchor_masks(timeline, predictions, config)
    shoot_masks, shoot_source = _resolve_shoot_masks(timeline, predictions, config, fallback_masks=anchor_masks)
    if not _tick(2):
        return {"frames": [], "rows": [], "tracks": {}, "tip_tracks": {}, "summary": {"status": "cancelled", "frames": 0, "tracks": 0}}
    # Temporal preprocessing is useful for maintaining plant identities, but it
    # must never add historical pixels to the mask used for phenotype metrics.
    tracking_root_masks = [
        np.asarray(mask, dtype=np.uint8).copy() for mask in measurement_root_masks
    ]
    tracking_primary_seed_masks = [
        np.asarray(mask, dtype=np.uint8).copy() for mask in measurement_primary_root_masks
    ]
    if config.temporal_smoothing_enabled:
        tracking_root_masks = _smooth_binary_masks(tracking_root_masks, config.temporal_alpha)
        tracking_primary_seed_masks = _smooth_binary_masks(
            tracking_primary_seed_masks,
            config.temporal_alpha,
        )
    if not _tick(3):
        return {"frames": [], "rows": [], "tracks": {}, "tip_tracks": {}, "summary": {"status": "cancelled", "frames": 0, "tracks": 0}}
    tracking_root_masks, shoot_crown_lock_meta = _apply_shoot_crown_lock(tracking_root_masks, anchor_masks, config)
    tracking_primary_seed_masks, _primary_seed_crown_lock_meta = _apply_shoot_crown_lock(
        tracking_primary_seed_masks,
        anchor_masks,
        config,
    )
    shoot_crown_lock_meta = {**shoot_crown_lock_meta, "scope": "tracking_only"}
    mask_source = {
        **mask_source,
        "measurement_scope": "current_frame_prediction",
        "tracking_temporal_smoothing_enabled": bool(config.temporal_smoothing_enabled),
        "tracking_temporal_alpha": float(config.temporal_alpha),
        "ownership_seed_scope": (
            "primary_root_class" if primary_seed_available else "root_union_fallback"
        ),
    }

    lateral_mode = "derived_from_root"
    lateral_masks: list[np.ndarray] = [np.zeros_like(mask, dtype=np.uint8) for mask in measurement_root_masks]
    lateral_class_id = int(config.lateral_class_id) if config.lateral_class_id is not None else None
    if lateral_class_id is not None and lateral_class_id > 0 and lateral_class_id != int(config.root_class_id):
        lateral_mode = "class_mask"
        for frame_idx, item in enumerate(timeline):
            pred_idx = _prediction_to_index_mask(predictions[item.uid])
            if pred_idx.ndim != 2 or pred_idx.size == 0:
                lateral_masks[frame_idx] = np.zeros_like(measurement_root_masks[frame_idx], dtype=np.uint8)
                continue
            lateral_masks[frame_idx] = (pred_idx == np.uint8(lateral_class_id)).astype(np.uint8)

    tracking = _resolve_tracking(
        tracking_root_masks,
        config,
        anchor_masks=anchor_masks,
        timeline=timeline,
        progress_callback=_tracker_progress,
    )
    two_dt_summary = tracking.get("two_dt_summary") if isinstance(tracking, dict) else None
    if isinstance(two_dt_summary, dict) and str(two_dt_summary.get("status", "")).lower() == "cancelled":
        return {"frames": [], "rows": [], "tracks": {}, "tip_tracks": {}, "summary": {"status": "cancelled", "frames": 0, "tracks": 0}}
    if not _tick(7):
        return {"frames": [], "rows": [], "tracks": {}, "tip_tracks": {}, "summary": {"status": "cancelled", "frames": 0, "tracks": 0}}

    track_ids = list(tracking.get("track_ids", []))
    track_bboxes: dict[str, list[BBox]] = dict(tracking.get("track_bboxes", {}))
    overlap_frames = dict(tracking.get("overlap_frames", {}))
    compartment_hints = _build_track_compartment_hints(
        root_masks=tracking_primary_seed_masks,
        anchor_masks=anchor_masks,
        track_ids=track_ids,
        track_bboxes=track_bboxes,
        seed_frame=int(tracking.get("seed_frame", -1)),
        config=config,
    )
    owned_shoot_frames, shoot_tracking_meta = _build_owned_shoot_masks_by_frame(
        shoot_masks=shoot_masks,
        track_ids=track_ids,
        compartment_hints=compartment_hints,
        config=config,
        root_masks=tracking_root_masks,
        frame_images=[np.asarray(item.image) for item in timeline],
    )
    if not _tick(8):
        return {"frames": [], "rows": [], "tracks": {}, "tip_tracks": {}, "summary": {"status": "cancelled", "frames": 0, "tracks": 0}}
    owned_root_frames, owned_lateral_frames, ownership_meta = _build_owned_masks_by_frame(
        measurement_root_masks,
        lateral_masks,
        owned_shoot_frames,
        track_ids,
        compartment_hints,
        config,
        tip_priors_by_frame=(
            getattr(config, "external_tip_priors_by_frame", None)
            if isinstance(getattr(config, "external_tip_priors_by_frame", None), dict)
            else None
        ),
        compact=True,
        root_seed_masks=measurement_primary_root_masks,
    )
    payload = _assemble_payload_from_owned_masks(
        timeline,
        owned_root_frames,
        owned_lateral_frames,
        owned_shoot_frames,
        anchor_masks,
        track_ids,
        overlap_frames,
        config,
        mask_source,
        anchor_source,
        shoot_crown_lock_meta,
        shoot_tracking_meta,
        compartment_hints,
        {
            **ownership_meta,
            "refined_pass": False,
            "effective_min_component_area": int(tracking.get("effective_min_component_area", max(1, int(config.min_component_area)))),
        },
        lateral_mode,
        shoot_source,
    )
    if not _tick(9):
        return {"frames": [], "rows": [], "tracks": {}, "tip_tracks": {}, "summary": {"status": "cancelled", "frames": 0, "tracks": 0}}
    tracking_mode = str(getattr(config, "tracking_mode", "auto") or "auto").strip().lower()
    tip_ownership_refinement_enabled = bool(
        config.tip_tracking_enabled and tracking_mode != "arabidopsis_crown_lanes"
    )
    tip_ownership_refinement_applied = False
    if tip_ownership_refinement_enabled:
        tip_priors_by_frame = _merge_tip_priors_by_frame(
            _build_tip_priors_by_frame(
                payload.get("tip_tracks", {}) if isinstance(payload.get("tip_tracks"), dict) else {},
                len(timeline),
                track_ids,
                config,
            ),
            getattr(config, "external_tip_priors_by_frame", None),
        )
        if tip_priors_by_frame:
            tip_ownership_refinement_applied = True
            owned_root_frames, owned_lateral_frames, ownership_meta = _build_owned_masks_by_frame(
                measurement_root_masks,
                lateral_masks,
                owned_shoot_frames,
                track_ids,
                compartment_hints,
                config,
                tip_priors_by_frame=tip_priors_by_frame,
                compact=True,
                root_seed_masks=measurement_primary_root_masks,
            )
            payload = _assemble_payload_from_owned_masks(
                timeline,
                owned_root_frames,
                owned_lateral_frames,
                owned_shoot_frames,
                anchor_masks,
                track_ids,
                overlap_frames,
                config,
                mask_source,
                anchor_source,
                shoot_crown_lock_meta,
                shoot_tracking_meta,
                compartment_hints,
                {
                    **ownership_meta,
                    "refined_pass": True,
                    "effective_min_component_area": int(tracking.get("effective_min_component_area", max(1, int(config.min_component_area)))),
                },
                lateral_mode,
                shoot_source,
            )
    payload_summary = payload.get("summary")
    if isinstance(payload_summary, dict):
        payload_summary["tip_ownership_refinement"] = {
            "enabled": bool(tip_ownership_refinement_enabled),
            "applied": bool(tip_ownership_refinement_applied),
            "status": (
                "applied"
                if tip_ownership_refinement_applied
                else (
                    "skipped_for_stable_crown_lanes"
                    if bool(config.tip_tracking_enabled) and tracking_mode == "arabidopsis_crown_lanes"
                    else ("no_tip_priors" if tip_ownership_refinement_enabled else "disabled")
                )
            ),
        }
    if not _tick(11):
        return {"frames": [], "rows": [], "tracks": {}, "tip_tracks": {}, "summary": {"status": "cancelled", "frames": 0, "tracks": 0}}
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if isinstance(summary, dict):
        summary["tracking_source"] = str(tracking.get("source", "internal"))
        if isinstance(tracking.get("identity_initialization"), dict):
            summary["identity_initialization"] = dict(tracking.get("identity_initialization", {}))
        if isinstance(tracking.get("two_dt_summary"), dict):
            summary["two_dt_tracker"] = dict(tracking.get("two_dt_summary", {}))
    identity = tracking.get("identity_initialization") if isinstance(tracking, dict) else None
    payload_rows = payload.get("rows") if isinstance(payload, dict) else None
    if isinstance(identity, dict) and isinstance(payload_rows, list):
        track_positions = {str(track_id): idx for idx, track_id in enumerate(track_ids)}
        frame_centers_x = identity.get("per_frame_centers_x")
        frame_centers_y = identity.get("per_frame_centers_y")
        frame_center_sources = identity.get("per_frame_center_sources")
        frame_motion_sources = identity.get("per_frame_motion_source")
        frame_alignment_methods = identity.get("per_frame_alignment_method")
        frame_alignment_statuses = identity.get("per_frame_alignment_status")
        frame_alignment_scores = identity.get("per_frame_alignment_score")
        frame_alignment_accepteds = identity.get("per_frame_alignment_accepted")
        frame_alignment_usables = identity.get("per_frame_alignment_usable")
        frame_alignment_shift_x = identity.get("per_frame_alignment_shift_x")
        frame_alignment_shift_y = identity.get("per_frame_alignment_shift_y")
        frame_alignment_rotation = identity.get("per_frame_alignment_rotation_deg")
        frame_alignment_reference = identity.get("per_frame_alignment_reference_frame_index")
        frame_alignment_warps = identity.get("per_frame_alignment_warp_matrix")

        def _frame_value(values: object, frame_index: int, default: object = "") -> object:
            if not isinstance(values, list) or frame_index < 0 or frame_index >= len(values):
                return default
            return values[frame_index]

        def _track_frame_value(
            values: object,
            frame_index: int,
            track_index: int,
            default: object = np.nan,
        ) -> object:
            frame_values = _frame_value(values, frame_index, default=[])
            if not isinstance(frame_values, list) or track_index < 0 or track_index >= len(frame_values):
                return default
            return frame_values[track_index]

        for row in payload_rows:
            if not isinstance(row, dict):
                continue
            frame_index = int(row.get("frame_index", -1))
            track_index = int(track_positions.get(str(row.get("plant_id", "")), -1))
            alignment_status = str(_frame_value(frame_alignment_statuses, frame_index, default=""))
            row["ownership_crown_center_x"] = _track_frame_value(frame_centers_x, frame_index, track_index)
            row["ownership_crown_center_y"] = _track_frame_value(frame_centers_y, frame_index, track_index)
            row["ownership_crown_center_source"] = str(
                _track_frame_value(frame_center_sources, frame_index, track_index, default="")
            )
            row["ownership_frame_motion_source"] = str(
                _frame_value(frame_motion_sources, frame_index, default="")
            )
            row["ownership_frame_alignment_method"] = str(
                _frame_value(frame_alignment_methods, frame_index, default="")
            )
            row["ownership_frame_alignment_status"] = alignment_status
            row["ownership_frame_alignment_score"] = float(
                _frame_value(frame_alignment_scores, frame_index, default=0.0) or 0.0
            )
            row["ownership_frame_alignment_accepted"] = bool(
                _frame_value(frame_alignment_accepteds, frame_index, default=False)
            )
            row["ownership_frame_alignment_usable"] = bool(
                _frame_value(frame_alignment_usables, frame_index, default=False)
            )
            row["ownership_frame_alignment_shift_x"] = float(
                _frame_value(frame_alignment_shift_x, frame_index, default=0.0) or 0.0
            )
            row["ownership_frame_alignment_shift_y"] = float(
                _frame_value(frame_alignment_shift_y, frame_index, default=0.0) or 0.0
            )
            row["ownership_frame_alignment_rotation_deg"] = float(
                _frame_value(frame_alignment_rotation, frame_index, default=0.0) or 0.0
            )
            row["ownership_frame_alignment_reference_frame_index"] = int(
                _frame_value(frame_alignment_reference, frame_index, default=0) or 0
            )
            row["ownership_frame_alignment_warp_matrix"] = json.dumps(
                _frame_value(
                    frame_alignment_warps,
                    frame_index,
                    default=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                ),
                separators=(",", ":"),
            )
    _tick(total_steps)
    return payload


def _infer_timestep_hours_from_summary(summary: dict[str, object]) -> float:
    raw_step = summary.get("timestep_hours")
    try:
        step = float(raw_step)
        if step > 0.0:
            return step
    except Exception:
        pass

    for frame_key, hour_key in (("t50_frame", "t50_hours"), ("tmgr_frame", "tmgr_hours")):
        try:
            frame = int(summary.get(frame_key, -1))
            hours = float(summary.get(hour_key, 0.0))
        except Exception:
            continue
        if frame > 0 and hours > 0.0:
            inferred = hours / float(frame)
            if inferred > 0.0:
                return float(inferred)
    return 1.0


def filter_analytics_payload_tracks(payload: dict[str, object], included_track_ids: set[str] | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {"frames": [], "rows": [], "tracks": {}, "tip_tracks": {}, "summary": {"status": "no_payload"}}
    if included_track_ids is None:
        return payload

    include = {str(track_id).strip() for track_id in included_track_ids if str(track_id).strip()}
    tracks_in = payload.get("tracks")
    rows_in = payload.get("rows")
    combined_rows_in = payload.get("combined_rows")
    frames_in = payload.get("frames")
    tip_tracks_in = payload.get("tip_tracks")
    summary_in = payload.get("summary")

    tracks_src = tracks_in if isinstance(tracks_in, dict) else {}
    rows_src = rows_in if isinstance(rows_in, list) else []
    combined_rows_src = combined_rows_in if isinstance(combined_rows_in, list) else []
    frames_src = frames_in if isinstance(frames_in, list) else []
    tip_tracks_src = tip_tracks_in if isinstance(tip_tracks_in, dict) else {}
    summary_src = summary_in if isinstance(summary_in, dict) else {}

    tracks_out: dict[str, dict[str, object]] = {}
    for track_id, track_data in tracks_src.items():
        tid = str(track_id).strip()
        if tid and tid in include and isinstance(track_data, dict):
            tracks_out[tid] = dict(track_data)

    rows_out: list[dict[str, object]] = []
    for row in rows_src:
        if not isinstance(row, dict):
            continue
        tid = str(row.get("plant_id", "")).strip()
        if tid in include:
            rows_out.append(dict(row))

    combined_rows_out: list[dict[str, object]] = []
    for row in combined_rows_src:
        if not isinstance(row, dict):
            continue
        members = [str(member).strip() for member in row.get("members", [])] if isinstance(row.get("members"), list) else []
        if members and all(member in include for member in members):
            combined_rows_out.append(dict(row))

    frames_out: list[dict[str, object]] = []
    for frame in frames_src:
        if not isinstance(frame, dict):
            continue
        tracks = frame.get("tracks")
        tracks_dict = tracks if isinstance(tracks, dict) else {}
        filtered_tracks = {
            str(track_id): dict(track_row)
            for track_id, track_row in tracks_dict.items()
            if str(track_id).strip() in include and isinstance(track_row, dict)
        }
        frame_copy = dict(frame)
        frame_copy["tracks"] = filtered_tracks
        combined_groups_obj = frame.get("combined_groups")
        combined_groups = combined_groups_obj if isinstance(combined_groups_obj, list) else []
        frame_copy["combined_groups"] = [
            dict(group)
            for group in combined_groups
            if isinstance(group, dict)
            and isinstance(group.get("members"), list)
            and all(str(member).strip() in include for member in group.get("members", []))
        ]
        frames_out.append(frame_copy)

    tip_tracks_out: dict[str, dict[str, object]] = {}
    for tip_id, tip_data in tip_tracks_src.items():
        if not isinstance(tip_data, dict):
            continue
        dominant_parent = str(tip_data.get("dominant_plant_id", "")).strip()
        if dominant_parent and dominant_parent in include:
            tip_tracks_out[str(tip_id)] = dict(tip_data)

    frame_count = len(frames_out)
    track_ids = sorted(tracks_out.keys())
    track_count = len(track_ids)
    timestep_hours = _infer_timestep_hours_from_summary(summary_src)

    germ_pct: list[float] = []
    if frame_count > 0 and track_count > 0:
        for frame_idx in range(frame_count):
            germinated = 0
            for track_id in track_ids:
                series = tracks_out.get(track_id, {}).get("length_mm_clean")
                values = series if isinstance(series, list) else []
                value = float(values[frame_idx]) if frame_idx < len(values) else 0.0
                if value > 0.0:
                    germinated += 1
            germ_pct.append((100.0 * germinated) / float(max(1, track_count)))
    else:
        germ_pct = [0.0 for _ in range(frame_count)]

    t50_frame = -1
    for idx, value in enumerate(germ_pct):
        if value >= 50.0:
            t50_frame = idx
            break

    if len(germ_pct) >= 2:
        grad = np.diff(np.asarray(germ_pct, dtype=np.float64), prepend=germ_pct[0]) / float(max(1e-6, timestep_hours))
        tmgr_frame = int(np.argmax(grad))
    else:
        tmgr_frame = -1

    fpca = _compute_fpca(
        {
            track_id: [
                float(v)
                for v in (tracks_out.get(track_id, {}).get("length_mm_clean") or [])
            ]
            for track_id in track_ids
        },
        timestep_h=float(max(1e-6, timestep_hours)),
    )

    summary_out = dict(summary_src)
    summary_out["tracks"] = int(track_count)
    summary_out["frames"] = int(frame_count)
    summary_out["timestep_hours"] = float(timestep_hours)
    summary_out["t50_frame"] = int(t50_frame)
    summary_out["t50_hours"] = float(t50_frame * timestep_hours) if t50_frame >= 0 else None
    summary_out["tmgr_frame"] = int(tmgr_frame)
    summary_out["tmgr_hours"] = float(tmgr_frame * timestep_hours) if tmgr_frame >= 0 else None
    summary_out["germination_percent_per_frame"] = [float(v) for v in germ_pct]
    summary_out["fpca"] = fpca
    summary_out["filtered_track_ids"] = track_ids
    summary_out["excluded_track_ids"] = sorted(set(str(k).strip() for k in tracks_src.keys()) - set(track_ids))
    lateral_count_values = [int(row.get("lateral_count", 0)) for row in rows_out if isinstance(row, dict)]
    lateral_total_length_values = [float(row.get("lateral_total_length_mm", 0.0)) for row in rows_out if isinstance(row, dict)]
    lateral_diam_values = [
        float(row.get("lateral_mean_diameter_mm", 0.0))
        for row in rows_out
        if isinstance(row, dict) and float(row.get("lateral_mean_diameter_mm", 0.0)) > 0.0
    ]
    primary_root_length_values = [float(row.get("primary_root_length_mm_raw", 0.0)) for row in rows_out if isinstance(row, dict)]
    total_root_length_values = [float(row.get("total_root_length_mm_raw", 0.0)) for row in rows_out if isinstance(row, dict)]
    total_root_area_values = [int(row.get("total_root_area_px", 0)) for row in rows_out if isinstance(row, dict)]
    shoot_area_values = [int(row.get("shoot_area_px", 0)) for row in rows_out if isinstance(row, dict)]
    summary_out["lateral_total_count"] = int(sum(lateral_count_values))
    summary_out["lateral_mean_count_per_track_frame"] = float(np.mean(lateral_count_values)) if lateral_count_values else 0.0
    summary_out["lateral_total_length_mm"] = float(sum(lateral_total_length_values))
    summary_out["lateral_mean_diameter_mm"] = float(np.mean(lateral_diam_values)) if lateral_diam_values else 0.0
    summary_out["primary_root_total_length_mm"] = float(sum(primary_root_length_values))
    summary_out["total_root_total_length_mm"] = float(sum(total_root_length_values))
    summary_out["total_root_area_peak_px"] = int(max(total_root_area_values)) if total_root_area_values else 0
    summary_out["shoot_area_peak_px"] = int(max(shoot_area_values)) if shoot_area_values else 0
    summary_out["shoot_area_mean_px"] = float(np.mean(shoot_area_values)) if shoot_area_values else 0.0
    summary_out["tip_tracks"] = int(len(tip_tracks_out))
    summary_out["tip_track_observations"] = int(
        sum(len(t.get("observations", [])) for t in tip_tracks_out.values() if isinstance(t, dict))
    )
    summary_out["tip_cross_plant_links"] = int(
        sum(int(t.get("cross_plant_switches", 0)) for t in tip_tracks_out.values() if isinstance(t, dict))
    )
    tip_link_speeds: list[float] = []
    for tip_track in tip_tracks_out.values():
        if not isinstance(tip_track, dict):
            continue
        obs_obj = tip_track.get("observations")
        obs_list = obs_obj if isinstance(obs_obj, list) else []
        for obs in obs_list:
            if not isinstance(obs, dict):
                continue
            try:
                speed = float(obs.get("speed_mm_day_from_prev", 0.0))
            except Exception:
                continue
            if speed > 0.0 and math.isfinite(speed):
                tip_link_speeds.append(speed)
    summary_out["tip_vector_speed_mm_day_mean"] = float(np.mean(tip_link_speeds)) if tip_link_speeds else 0.0
    summary_out["tip_vector_speed_mm_day_p90"] = (
        float(np.percentile(np.asarray(tip_link_speeds, dtype=np.float64), 90.0)) if tip_link_speeds else 0.0
    )
    summary_out["tip_vector_speed_mm_day_max"] = float(np.max(np.asarray(tip_link_speeds, dtype=np.float64))) if tip_link_speeds else 0.0
    measurement_tiers = summary_out.get("measurement_tiers")
    if isinstance(measurement_tiers, dict):
        summary_out["measurement_tiers"] = {
            **measurement_tiers,
            "combined_overlap_tracks": int(
                sum(1 for track_id in track_ids if tracks_out.get(track_id, {}).get("conflict_start_frame") is not None)
            ),
            "combined_overlap_groups": int(len({str(row.get("group_id", "")) for row in combined_rows_out if isinstance(row, dict)})),
            "combined_overlap_rows": int(sum(1 for row in rows_out if str(row.get("measurement_tier", "")) == "combined_overlap")),
        }

    return {
        "frames": frames_out,
        "rows": rows_out,
        "combined_rows": combined_rows_out,
        "tracks": tracks_out,
        "tip_tracks": tip_tracks_out,
        "summary": summary_out,
    }


def _build_gt_index(shape_hw: tuple[int, int], annotations_per_uid: dict[int, np.ndarray], class_ids: list[int]) -> np.ndarray:
    h, w = shape_hw
    gt = np.zeros((h, w), dtype=np.uint8)
    for class_id in class_ids:
        mask = annotations_per_uid.get(int(class_id))
        if mask is None:
            continue
        if mask.shape != (h, w):
            resized = np.array(Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST), dtype=np.uint8)
            gt[resized > 0] = np.uint8(class_id)
        else:
            gt[mask > 0] = np.uint8(class_id)
    return gt


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if np.count_nonzero(m) == 0:
        return m
    er = cv2.erode(m, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return cv2.subtract(m, er)


def _hausdorff_distance_mm(pred: np.ndarray, gt: np.ndarray, px_mm: float) -> float | None:
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    if np.count_nonzero(p) == 0 and np.count_nonzero(g) == 0:
        return 0.0
    if np.count_nonzero(p) == 0 or np.count_nonzero(g) == 0:
        return None
    pb = _boundary(p)
    gb = _boundary(g)
    if np.count_nonzero(pb) == 0 or np.count_nonzero(gb) == 0:
        return None
    dt_g = cv2.distanceTransform((1 - gb).astype(np.uint8), cv2.DIST_L2, 3)
    dt_p = cv2.distanceTransform((1 - pb).astype(np.uint8), cv2.DIST_L2, 3)
    d1 = float(np.max(dt_g[pb > 0])) if np.any(pb > 0) else 0.0
    d2 = float(np.max(dt_p[gb > 0])) if np.any(gb > 0) else 0.0
    return float(max(d1, d2) * px_mm)


def benchmark_predictions(
    items: list[DatasetImageItem],
    predictions: dict[str, np.ndarray],
    annotations: dict[str, dict[int, np.ndarray]],
    class_ids: list[int],
    pixel_size_mm: float = 1.0,
) -> dict[str, object]:
    class_ids = [int(cid) for cid in class_ids]
    stats = {cid: {"tp": 0, "fp": 0, "fn": 0, "hd": []} for cid in class_ids}
    evaluated = 0
    for item in items:
        pred = predictions.get(item.uid)
        ann = annotations.get(item.uid)
        if pred is None or ann is None:
            continue
        gt = _build_gt_index(item.image.shape[:2], ann, class_ids)
        evaluated += 1
        for cid in class_ids:
            pred_c = pred == cid
            gt_c = gt == cid
            tp = int(np.logical_and(pred_c, gt_c).sum())
            fp = int(np.logical_and(pred_c, np.logical_not(gt_c)).sum())
            fn = int(np.logical_and(np.logical_not(pred_c), gt_c).sum())
            stats[cid]["tp"] += tp
            stats[cid]["fp"] += fp
            stats[cid]["fn"] += fn
            hd = _hausdorff_distance_mm(pred_c.astype(np.uint8), gt_c.astype(np.uint8), pixel_size_mm)
            if hd is not None:
                stats[cid]["hd"].append(float(hd))
    per_class: dict[str, dict[str, float | None]] = {}
    dices: list[float] = []
    ious: list[float] = []
    for cid in class_ids:
        tp = float(stats[cid]["tp"])
        fp = float(stats[cid]["fp"])
        fn = float(stats[cid]["fn"])
        dice = (2.0 * tp) / max(1.0, (2.0 * tp + fp + fn))
        iou = tp / max(1.0, (tp + fp + fn))
        dices.append(dice)
        ious.append(iou)
        hd_vals = stats[cid]["hd"]
        per_class[str(cid)] = {
            "dice": float(dice),
            "iou": float(iou),
            "hausdorff_mm_mean": float(np.mean(hd_vals)) if hd_vals else None,
        }
    return {
        "evaluated_images": int(evaluated),
        "macro_dice": float(np.mean(dices)) if dices else 0.0,
        "macro_iou": float(np.mean(ious)) if ious else 0.0,
        "per_class": per_class,
    }


def export_rows_csv(rows: list[dict[str, object]], output_path: Path) -> Path:
    if not rows:
        raise ValueError("No analytics rows available.")
    keys = sorted({str(k) for row in rows for k in row.keys() if k != "path_points"})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            safe: dict[str, object] = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    safe[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                else:
                    safe[key] = value
            writer.writerow(safe)
    return output_path


def export_summary_json(payload: dict[str, object], output_path: Path) -> Path:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def export_rsml(payload: dict[str, object], output_path: Path) -> Path:
    tracks = payload.get("tracks", {})
    frames = payload.get("frames", [])
    root = ET.Element("rsml")
    metadata = ET.SubElement(root, "metadata")
    metadata.set("unit", "mm")
    scene = ET.SubElement(root, "scene")
    for track_id, track_data in sorted(tracks.items()):
        plant = ET.SubElement(scene, "plant", id=str(track_id), label=str(track_id))
        for frame in frames:
            frame_index = int(frame.get("frame_index", 0))
            frame_tracks = frame.get("tracks", {})
            if not isinstance(frame_tracks, dict) or track_id not in frame_tracks:
                continue
            row = frame_tracks[track_id]
            path_points = row.get("path_points", [])
            if not isinstance(path_points, list) or len(path_points) < 2:
                continue
            root_elem = ET.SubElement(plant, "root", id=f"{track_id}_t{frame_index}")
            props = ET.SubElement(root_elem, "properties")
            ET.SubElement(props, "property", name="frame_index", value=str(frame_index))
            ET.SubElement(props, "property", name="length_mm", value=str(row.get("root_length_mm_clean", row.get("root_length_mm_raw", 0.0))))
            ET.SubElement(
                props,
                "property",
                name="primary_root_length_mm",
                value=str(row.get("primary_root_length_mm_clean", row.get("primary_root_length_mm_raw", ""))),
            )
            ET.SubElement(
                props,
                "property",
                name="total_root_length_mm",
                value=str(row.get("total_root_length_mm_clean", row.get("total_root_length_mm_raw", ""))),
            )
            ET.SubElement(props, "property", name="lateral_root_length_mm", value=str(row.get("lateral_total_length_mm", "")))
            ET.SubElement(props, "property", name="shoot_area_mm2", value=str(row.get("shoot_area_mm2", "")))
            geom = ET.SubElement(root_elem, "geometry")
            poly = ET.SubElement(geom, "polyline")
            for p in path_points:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                ET.SubElement(poly, "point", x=str(float(p[0])), y=str(float(p[1])))
        if isinstance(track_data, dict):
            props = ET.SubElement(plant, "properties")
            for k in ("germination_frame", "overlap_frame"):
                ET.SubElement(props, "property", name=k, value=str(track_data.get(k)))

    tree = ET.ElementTree(root)
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)
    return output_path


def _fit_to_1080p(frame: np.ndarray) -> np.ndarray:
    target_w, target_h = 1920, 1080
    h, w = frame.shape[:2]
    scale = min(target_w / float(max(1, w)), target_h / float(max(1, h)))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = np.array(Image.fromarray(frame).resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    canvas = np.full((target_h, target_w, 3), 10, dtype=np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _apply_contrast_gain(image_rgb: np.ndarray, gain: float) -> np.ndarray:
    gain = float(max(0.25, min(4.0, gain)))
    if abs(gain - 1.0) <= 1.0e-3:
        return np.asarray(image_rgb, dtype=np.uint8)
    image = np.asarray(image_rgb, dtype=np.float32)
    adjusted = (image - 127.5) * gain + 127.5
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _row_float(row: dict[str, object], keys: tuple[str, ...], default: float = 0.0) -> float:
    for key in keys:
        if key not in row:
            continue
        try:
            value = float(row.get(key, default))
        except Exception:
            continue
        if np.isfinite(value):
            return float(value)
    return float(default)


def analytics_track_overlay_label(track_id: object, row: dict[str, object]) -> str:
    """Compact per-plant label for analytics previews and QC MP4s."""

    track_label = str(track_id).strip() or str(row.get("plant_id", "")).strip() or "plant"
    total_mm = _row_float(
        row,
        (
            "total_root_length_mm_clean",
            "total_root_length_mm_raw",
            "root_length_mm_clean",
            "root_length_mm_raw",
        ),
    )
    primary_mm = _row_float(
        row,
        (
            "primary_root_length_mm_clean",
            "primary_root_length_mm_raw",
        ),
    )
    lateral_mm = _row_float(row, ("lateral_total_length_mm",))
    shoot_px = int(round(_row_float(row, ("shoot_area_px",), default=0.0)))
    return f"{track_label} T={total_mm:.2f}mm P={primary_mm:.2f} L={lateral_mm:.2f} S={shoot_px}px"


class Mp4VideoWriter:
    """Robust MP4 writer for packaged apps (PyInstaller/macOS/Windows)."""

    def __init__(self, output_path: Path, width: int, height: int, fps: int = 6) -> None:
        self.output_path = prepare_mp4_output_path(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self._backend = ""
        self._writer = None
        self._errors: list[str] = []

        # Primary backend: imageio-ffmpeg (does not depend on imageio metadata).
        try:
            import imageio_ffmpeg

            gen = imageio_ffmpeg.write_frames(
                str(self.output_path),
                size=(self.width, self.height),
                fps=self.fps,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                macro_block_size=1,
                ffmpeg_log_level="error",
            )
            gen.send(None)
            self._backend = "imageio_ffmpeg"
            self._writer = gen
            return
        except Exception as exc:
            self._errors.append(f"imageio_ffmpeg: {exc}")

        # Fallback: OpenCV MP4 encoder (less ideal, but keeps export available).
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(self.output_path), fourcc, float(self.fps), (self.width, self.height))
            if not writer.isOpened():
                raise RuntimeError("VideoWriter failed to open mp4v encoder.")
            self._backend = "opencv"
            self._writer = writer
            return
        except Exception as exc:
            self._errors.append(f"opencv: {exc}")

        detail = "; ".join(self._errors) if self._errors else "No backend available."
        raise RuntimeError(
            "Unable to initialize MP4 export backend. Install FFmpeg and restart the app, or set "
            "NPEC_FFMPEG_EXE to an executable FFmpeg path. "
            f"Details: {detail}"
        )

    def append(self, frame_rgb: np.ndarray) -> None:
        frame = np.asarray(frame_rgb)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("MP4 frame must be RGB (H, W, 3).")
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise ValueError(
                f"Frame size mismatch: expected {(self.height, self.width)} got {frame.shape[:2]}."
            )
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

        if self._backend == "imageio_ffmpeg":
            self._writer.send(frame)
        elif self._backend == "opencv":
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self._writer.write(bgr)
        else:
            raise RuntimeError("MP4 writer backend is not initialized.")

    def close(self) -> None:
        writer = self._writer
        self._writer = None
        if writer is None:
            return
        try:
            if self._backend == "imageio_ffmpeg":
                writer.close()
            elif self._backend == "opencv":
                writer.release()
        except Exception:
                pass


def prepare_mp4_output_path(output_path: Path) -> Path:
    path = Path(output_path).expanduser()
    parent = path.parent if str(path.parent) else Path.cwd()
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise OSError(f'Cannot create MP4 output folder "{parent}": {exc}') from exc
    if not parent.is_dir():
        raise NotADirectoryError(f'MP4 output folder is not a directory: "{parent}"')
    if path.exists() and not os.access(path, os.W_OK):
        raise PermissionError(f'MP4 output file is not writable: "{path}"')

    probe_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name or 'mp4'}.write-test-",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as probe:
            probe_path = Path(probe.name)
    except Exception as exc:
        raise PermissionError(
            f'Cannot write MP4 output in "{parent}": {exc}. Choose a writable folder such as Movies or Documents.'
        ) from exc
    finally:
        if probe_path is not None:
            try:
                probe_path.unlink(missing_ok=True)
            except Exception:
                pass
    return path


def open_mp4_video_writer(output_path: Path, width: int, height: int, fps: int = 6) -> Mp4VideoWriter:
    return Mp4VideoWriter(output_path=output_path, width=width, height=height, fps=fps)


def export_qc_video(
    payload: dict[str, object],
    items: list[DatasetImageItem],
    predictions: dict[str, np.ndarray],
    config: AnalyticsConfig,
    output_path: Path,
    fps: int = 6,
    show_rgb_shoot_rescue: bool = False,
) -> Path:
    by_uid = {item.uid: item for item in items}
    frames = payload.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("No analytics frames available.")
    writer = open_mp4_video_writer(
        output_path=Path(output_path),
        width=1920,
        height=1080,
        fps=max(1, int(fps)),
    )
    line_thickness = max(1, int(config.qc_line_thickness))
    path_thickness = max(1, int(config.qc_path_thickness))
    text_thickness = max(1, int(round(line_thickness * 0.8)))
    text_scale = float(max(0.20, min(3.0, config.qc_font_scale)))
    overlay_color = np.array(
        [
            float(int(config.qc_overlay_color_rgb[0])),
            float(int(config.qc_overlay_color_rgb[1])),
            float(int(config.qc_overlay_color_rgb[2])),
        ],
        dtype=np.float32,
    )
    bbox_color = tuple(int(v) for v in config.qc_bbox_color_rgb[:3])
    conflict_bbox_color = tuple(int(v) for v in config.qc_conflict_bbox_color_rgb[:3])
    path_color = tuple(int(v) for v in config.qc_path_color_rgb[:3])
    text_color = tuple(int(v) for v in config.qc_text_color_rgb[:3])
    try:
        for frame in frames:
            uid = str(frame.get("uid", ""))
            item = by_uid.get(uid)
            if item is None:
                continue
            image = _apply_contrast_gain(item.image.copy(), config.qc_contrast_gain)
            pred = predictions.get(uid)
            if pred is not None:
                pred_idx = _prediction_to_index_mask(pred)
                overlay = image.astype(np.float32)
                if pred_idx.ndim == 2 and pred_idx.shape[:2] == image.shape[:2]:
                    shoot_overlay_id = _configured_shoot_class_id(config)
                    display_pred_idx = pred_idx
                    if bool(show_rgb_shoot_rescue) and shoot_overlay_id is not None and int(shoot_overlay_id) > 0:
                        display_pred_idx = pred_idx.copy()
                        model_shoot = display_pred_idx == np.uint8(int(shoot_overlay_id))
                        display_pred_idx[model_shoot] = 0
                        try:
                            from .pyphenotyper_adapter import build_lucifer_green_shoot_mask

                            color_mask, color_meta = build_lucifer_green_shoot_mask(
                                item.image,
                                tuple(int(v) for v in pred_idx.shape[:2]),
                            )
                            candidate_pixels = int(np.count_nonzero(color_mask))
                            min_pixels = max(20, int(round(float(pred_idx.shape[0] * pred_idx.shape[1]) * 0.000001)))
                            if candidate_pixels >= min_pixels and bool(color_meta.get("rgb_like", False)):
                                root_mask = pred_idx == np.uint8(int(config.root_class_id))
                                if config.lateral_class_id is not None and int(config.lateral_class_id) > 0:
                                    root_mask = np.logical_or(
                                        root_mask,
                                        pred_idx == np.uint8(int(config.lateral_class_id)),
                                    )
                                green_only = (np.asarray(color_mask, dtype=np.uint8) > 0) & (~root_mask)
                                display_pred_idx[green_only] = np.uint8(int(shoot_overlay_id))
                        except Exception:
                            pass
                    class_layers: list[tuple[np.ndarray, np.ndarray]] = [
                        ((display_pred_idx == np.uint8(config.root_class_id)), overlay_color),
                    ]
                    if config.lateral_class_id is not None and int(config.lateral_class_id) > 0:
                        class_layers.append(
                            (
                                (display_pred_idx == np.uint8(int(config.lateral_class_id))),
                                np.asarray([126.0, 161.0, 255.0], dtype=np.float32),
                            )
                        )
                    seed_overlay_id = int(config.seed_class_id) if config.seed_class_id is not None else -1
                    if seed_overlay_id > 0 and seed_overlay_id != shoot_overlay_id:
                        class_layers.append(
                            (
                                (display_pred_idx == np.uint8(seed_overlay_id)),
                                np.asarray([255.0, 214.0, 112.0], dtype=np.float32),
                            )
                        )
                    if shoot_overlay_id is not None and int(shoot_overlay_id) > 0:
                        class_layers.append(
                            (
                                (display_pred_idx == np.uint8(int(shoot_overlay_id))),
                                np.asarray([255.0, 112.0, 214.0], dtype=np.float32),
                            )
                        )
                    for mask, color in class_layers:
                        if np.any(mask):
                            overlay[mask] = overlay[mask] * (1.0 - config.qc_overlay_alpha) + color * config.qc_overlay_alpha
                image = np.clip(overlay, 0, 255).astype(np.uint8)
            tracks = frame.get("tracks", {})
            if isinstance(tracks, dict):
                for track_id, row in tracks.items():
                    if not isinstance(row, dict):
                        continue
                    x = int(row.get("bbox_x", 0))
                    y = int(row.get("bbox_y", 0))
                    w = int(row.get("bbox_w", 0))
                    h = int(row.get("bbox_h", 0))
                    in_conflict = not bool(row.get("ownership_measurement_valid", True))
                    row_bbox_color = conflict_bbox_color if in_conflict else bbox_color
                    if w > 0 and h > 0:
                        cv2.rectangle(image, (x, y), (x + w, y + h), row_bbox_color, line_thickness, cv2.LINE_AA)
                    if not in_conflict:
                        label = analytics_track_overlay_label(track_id, row)
                        cv2.putText(
                            image,
                            label,
                            (x + 2, max(14, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            text_scale,
                            text_color,
                            text_thickness,
                            cv2.LINE_AA,
                        )
                        path_points = row.get("path_points", [])
                        if isinstance(path_points, list) and len(path_points) >= 2:
                            pts = np.asarray(path_points, dtype=np.int32).reshape(-1, 1, 2)
                            cv2.polylines(image, [pts], False, path_color, path_thickness, cv2.LINE_AA)
            combined_groups_obj = frame.get("combined_groups", [])
            combined_groups = combined_groups_obj if isinstance(combined_groups_obj, list) else []
            for group in combined_groups:
                if not isinstance(group, dict):
                    continue
                x = int(group.get("bbox_x", 0))
                y = int(group.get("bbox_y", 0))
                w = int(group.get("bbox_w", 0))
                h = int(group.get("bbox_h", 0))
                if w > 0 and h > 0:
                    cv2.rectangle(image, (x, y), (x + w, y + h), conflict_bbox_color, line_thickness, cv2.LINE_AA)
                members = [str(member) for member in group.get("members", [])] if isinstance(group.get("members"), list) else []
                length_mm = float(group.get("combined_root_length_mm", 0.0))
                label = f"{'+'.join(members)} combined {length_mm:.2f}mm"
                cv2.putText(
                    image,
                    label,
                    (x + 2, max(14, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale,
                    text_color,
                    text_thickness,
                    cv2.LINE_AA,
                )
            frame1080 = _fit_to_1080p(image)
            writer.append(frame1080)
    finally:
        writer.close()
    return output_path
