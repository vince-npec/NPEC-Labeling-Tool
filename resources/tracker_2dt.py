from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import heapq
import math

import cv2
import numpy as np

from .models import DatasetImageItem


BBox = tuple[int, int, int, int]


class TwoDTTrackerCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class TwoDTTrackerConfig:
    expected_track_count: int = 0
    min_component_area: int = 30
    bbox_padding: int = 12
    appearance_mean_shift_threshold: float = 12.0
    appearance_drop_threshold: float = 6.0
    top_band_ratio: float = 0.40
    reconnect_max_time_gap: int = 4
    reconnect_max_cost: float = 2.5
    registration_max_dim: int = 1024


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
    x0 = max(0, int(x - margin))
    y0 = max(0, int(y - margin))
    x1 = min(shape_hw[1], int(x + bw + margin))
    y1 = min(shape_hw[0], int(y + bh + margin))
    return _clip_bbox((x0, y0, x1 - x0, y1 - y0), shape_hw)


def _bbox_intersects(a: BBox, b: BBox) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return False
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _bbox_union(a: BBox, b: BBox) -> BBox:
    if int(a[2]) <= 0 or int(a[3]) <= 0:
        return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    if int(b[2]) <= 0 or int(b[3]) <= 0:
        return (int(a[0]), int(a[1]), int(a[2]), int(a[3]))
    x0 = min(int(a[0]), int(b[0]))
    y0 = min(int(a[1]), int(b[1]))
    x1 = max(int(a[0] + a[2]), int(b[0] + b[2]))
    y1 = max(int(a[1] + a[3]), int(b[1] + b[3]))
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def _mask_bbox(mask: np.ndarray) -> BBox:
    src = np.asarray(mask, dtype=np.uint8)
    if src.ndim != 2 or src.size == 0 or int(np.count_nonzero(src)) <= 0:
        return (0, 0, 0, 0)
    ys, xs = np.where(src > 0)
    if xs.size <= 0 or ys.size <= 0:
        return (0, 0, 0, 0)
    x0 = int(np.min(xs))
    y0 = int(np.min(ys))
    x1 = int(np.max(xs)) + 1
    y1 = int(np.max(ys)) + 1
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def _to_gray(image: np.ndarray) -> np.ndarray:
    src = np.asarray(image)
    if src.ndim == 2:
        return src.astype(np.uint8, copy=False)
    if src.ndim == 3 and src.shape[2] >= 3:
        rgb = src[..., :3].astype(np.uint8, copy=False)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.zeros(src.shape[:2], dtype=np.uint8)


def _resize_for_registration(gray: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    src = np.asarray(gray, dtype=np.uint8)
    if src.ndim != 2 or src.size == 0:
        return np.zeros((0, 0), dtype=np.uint8), 1.0
    max_side = max(int(src.shape[0]), int(src.shape[1]))
    limit = max(64, int(max_dim))
    if max_side <= limit:
        return src, 1.0
    scale = float(limit) / float(max_side)
    resized = cv2.resize(
        src,
        (max(1, int(round(src.shape[1] * scale))), max(1, int(round(src.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, float(scale)


def _resize_binary_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    src = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if src.ndim != 2 or src.size == 0:
        return np.zeros(shape_hw, dtype=np.uint8)
    if src.shape == shape_hw:
        return src
    resized = cv2.resize(src, (int(shape_hw[1]), int(shape_hw[0])), interpolation=cv2.INTER_NEAREST)
    return (np.asarray(resized, dtype=np.uint8) > 0).astype(np.uint8)


def _warp_binary_mask(mask: np.ndarray, shift_xy: tuple[float, float], shape_hw: tuple[int, int]) -> np.ndarray:
    src = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if src.ndim != 2 or src.size == 0:
        return np.zeros(shape_hw, dtype=np.uint8)
    warp = np.float32([[1.0, 0.0, float(shift_xy[0])], [0.0, 1.0, float(shift_xy[1])]])
    aligned = cv2.warpAffine(
        src,
        warp,
        (int(shape_hw[1]), int(shape_hw[0])),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (np.asarray(aligned, dtype=np.uint8) > 0).astype(np.uint8)


def _inverse_shift_bbox(bbox: BBox, shift_xy: tuple[float, float], shape_hw: tuple[int, int]) -> BBox:
    x, y, bw, bh = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    if bw <= 0 or bh <= 0:
        return (0, 0, 0, 0)
    dx = int(round(float(shift_xy[0])))
    dy = int(round(float(shift_xy[1])))
    return _clip_bbox((x - dx, y - dy, bw, bh), shape_hw)


def _scale_bbox_to_shape(bbox: BBox, src_shape_hw: tuple[int, int], dst_shape_hw: tuple[int, int]) -> BBox:
    x, y, bw, bh = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    if bw <= 0 or bh <= 0:
        return (0, 0, 0, 0)
    src_h, src_w = max(1, int(src_shape_hw[0])), max(1, int(src_shape_hw[1]))
    dst_h, dst_w = max(1, int(dst_shape_hw[0])), max(1, int(dst_shape_hw[1]))
    scale_x = float(dst_w) / float(src_w)
    scale_y = float(dst_h) / float(src_h)
    scaled = (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        max(1, int(round(bw * scale_x))),
        max(1, int(round(bh * scale_y))),
    )
    return _clip_bbox(scaled, dst_shape_hw)


def _register_to_last_frame(
    items: list[DatasetImageItem],
    max_dim: int,
    progress_callback: Callable[[str, float], bool] | None = None,
) -> tuple[list[np.ndarray], list[tuple[float, float]], dict[str, object]]:
    if not items:
        return [], [], {"mode": "none", "mean_abs_shift_px": 0.0, "max_abs_shift_px": 0.0}
    grays = [_to_gray(item.image) for item in items]
    ref = grays[-1]
    ref_small, ref_scale = _resize_for_registration(ref, max_dim)
    registered: list[np.ndarray] = [ref_small]
    shifts_rev: list[tuple[float, float]] = [(0.0, 0.0)]
    cumulative_dx = 0.0
    cumulative_dy = 0.0
    window = cv2.createHanningWindow((ref_small.shape[1], ref_small.shape[0]), cv2.CV_32F) if ref_small.size > 0 else None
    denom = max(1, len(grays) - 1)
    for idx in range(len(grays) - 2, -1, -1):
        moving_small, moving_scale = _resize_for_registration(grays[idx], max_dim)
        fixed_small, fixed_scale = _resize_for_registration(grays[idx + 1], max_dim)
        moving = moving_small.astype(np.float32, copy=False)
        fixed = fixed_small.astype(np.float32, copy=False)
        if moving.shape != fixed.shape or moving.size == 0:
            shifts_rev.append((cumulative_dx, cumulative_dy))
            registered.append(moving_small if moving_small.size > 0 else np.zeros_like(ref_small))
            if progress_callback is not None and not progress_callback("register", float(len(grays) - 1 - idx) / float(denom)):
                raise TwoDTTrackerCancelled("cancelled during registration")
            continue
        try:
            (dx, dy), _ = cv2.phaseCorrelate(moving, fixed, window)
        except Exception:
            dx, dy = 0.0, 0.0
        if not np.isfinite(dx):
            dx = 0.0
        if not np.isfinite(dy):
            dy = 0.0
        scale = float(min(moving_scale, fixed_scale, ref_scale))
        if scale <= 0.0:
            scale = 1.0
        cumulative_dx += float(dx) / scale
        cumulative_dy += float(dy) / scale
        warp = np.float32([[1.0, 0.0, cumulative_dx * ref_scale], [0.0, 1.0, cumulative_dy * ref_scale]])
        aligned = cv2.warpAffine(
            moving_small,
            warp,
            (ref_small.shape[1], ref_small.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        shifts_rev.append((cumulative_dx, cumulative_dy))
        registered.append(aligned)
        if progress_callback is not None and not progress_callback("register", float(len(grays) - 1 - idx) / float(denom)):
            raise TwoDTTrackerCancelled("cancelled during registration")
    registered.reverse()
    shifts_rev.reverse()
    magnitudes = [float(math.hypot(dx, dy)) for dx, dy in shifts_rev]
    meta = {
        "mode": "translation_phase_correlation",
        "mean_abs_shift_px": float(np.mean(magnitudes)) if magnitudes else 0.0,
        "max_abs_shift_px": float(np.max(magnitudes)) if magnitudes else 0.0,
        "reference_frame": int(max(0, len(items) - 1)),
        "frame_shifts_xy": [[float(dx), float(dy)] for dx, dy in shifts_rev],
        "registration_max_dim": int(max_dim),
        "registration_scale": float(ref_scale),
        "registered_shape_hw": [int(ref_small.shape[0]), int(ref_small.shape[1])],
    }
    return registered, shifts_rev, meta


def _clean_label_components(label_map: np.ndarray, min_area: int) -> np.ndarray:
    src = np.asarray(label_map, dtype=np.int32)
    if src.ndim != 2 or src.size == 0:
        return np.zeros((0, 0), dtype=np.int32)
    positive = (src > 0).astype(np.uint8)
    if int(np.count_nonzero(positive)) <= 0:
        return np.zeros_like(src, dtype=np.int32)
    min_pixels = int(max(1, min_area))
    num_labels, cc_map, stats, _ = cv2.connectedComponentsWithStats(positive, connectivity=8)
    out = src.copy()
    for component_id in range(1, int(num_labels)):
        component_mask = cc_map == component_id
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_pixels:
            out[component_mask] = 0
            continue
        component_values = src[component_mask]
        if component_values.size <= 0:
            continue
        counts = np.bincount(component_values.astype(np.int32, copy=False))
        for label_value in range(1, int(counts.shape[0])):
            if int(counts[label_value]) >= min_pixels:
                continue
            out[component_mask & (src == label_value)] = 0
    return out


def _component_list(mask: np.ndarray, min_area: int) -> list[dict[str, object]]:
    src = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if src.ndim != 2 or src.size == 0 or int(np.count_nonzero(src)) <= 0:
        return []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(src, connectivity=8)
    out: list[dict[str, object]] = []
    for component_id in range(1, int(num_labels)):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < int(max(1, min_area)):
            continue
        component = (labels == component_id).astype(np.uint8)
        bbox = _mask_bbox(component)
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        out.append(
            {
                "mask": component,
                "bbox": bbox,
                "area": area,
                "centroid": (float(centroids[component_id][0]), float(centroids[component_id][1])),
            }
        )
    out.sort(key=lambda comp: (float(comp["centroid"][0]), float(comp["centroid"][1])))
    return out


def _compute_apparition_map(
    registered_grays: list[np.ndarray],
    root_masks: list[np.ndarray],
    config: TwoDTTrackerConfig,
) -> np.ndarray:
    if not registered_grays:
        return np.zeros((0, 0), dtype=np.int32)
    shape_hw = registered_grays[0].shape[:2]
    root_union = np.zeros(shape_hw, dtype=np.uint8)
    first_presence = np.zeros(shape_hw, dtype=np.int32)
    for frame_idx, mask in enumerate(root_masks):
        mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
        if mask_u8.shape != shape_hw:
            continue
        new_pixels = (mask_u8 > 0) & (first_presence == 0)
        first_presence[new_pixels] = int(frame_idx + 1)
        root_union = np.maximum(root_union, mask_u8)
    if int(np.count_nonzero(root_union)) <= 0:
        return np.zeros(shape_hw, dtype=np.int32)

    out = first_presence.astype(np.int32, copy=True)
    stack = np.stack([np.asarray(frame, dtype=np.float32) for frame in registered_grays], axis=0)
    prefix = np.cumsum(stack, axis=0)
    total = prefix[-1]
    best_score = np.full(shape_hw, -1.0e9, dtype=np.float32)
    best_step = np.zeros(shape_hw, dtype=np.int32)
    best_drop = np.zeros(shape_hw, dtype=np.float32)
    nt = int(stack.shape[0])
    if nt >= 2:
        for split in range(0, nt - 1):
            left_sum = prefix[split]
            right_sum = total - left_sum
            left_mean = left_sum / float(split + 1)
            right_mean = right_sum / float(max(1, nt - split - 1))
            score = left_mean - right_mean
            drop = stack[split] - stack[split + 1]
            better = score > best_score
            best_score[better] = score[better]
            best_step[better] = int(split + 1)
            best_drop[better] = drop[better]
        valid = (
            (root_union > 0)
            & (best_score >= float(config.appearance_mean_shift_threshold))
            & (best_drop >= float(config.appearance_drop_threshold))
        )
        agree = valid & ((out == 0) | (np.abs(best_step - out) <= 1))
        out[agree] = best_step[agree]

    late_fill = (root_union > 0) & (out == 0)
    out[late_fill] = first_presence[late_fill]
    return _clean_label_components(out, max(4, int(config.min_component_area // 2)))


def _build_region_adjacency_graph(label_map: np.ndarray) -> dict[str, object]:
    src = np.asarray(label_map, dtype=np.int32)
    if src.ndim != 2 or src.size == 0:
        return {
            "vertices": [],
            "edges": [],
            "vertex_count": 0,
            "edge_count": 0,
            "component_owner": np.zeros((0, 0), dtype=np.int32),
        }
    h, w = src.shape[:2]
    component_owner = np.zeros((h, w), dtype=np.int32)
    vertices: list[dict[str, object]] = []
    next_vertex_id = 1
    for time_label in sorted(int(v) for v in np.unique(src) if int(v) > 0):
        cc_count, cc_map, stats, centroids = cv2.connectedComponentsWithStats(
            (src == time_label).astype(np.uint8),
            connectivity=8,
        )
        for comp_idx in range(1, int(cc_count)):
            area_px = int(stats[comp_idx, cv2.CC_STAT_AREA])
            if area_px <= 0:
                continue
            component_owner[cc_map == comp_idx] = int(next_vertex_id)
            x0 = int(stats[comp_idx, cv2.CC_STAT_LEFT])
            y0 = int(stats[comp_idx, cv2.CC_STAT_TOP])
            bw = int(stats[comp_idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[comp_idx, cv2.CC_STAT_HEIGHT])
            vertices.append(
                {
                    "id": int(next_vertex_id),
                    "time_label": int(time_label),
                    "area_px": int(area_px),
                    "centroid": (float(centroids[comp_idx][0]), float(centroids[comp_idx][1])),
                    "bbox": (x0, y0, bw, bh),
                }
            )
            next_vertex_id += 1
    if not vertices:
        return {
            "vertices": [],
            "edges": [],
            "vertex_count": 0,
            "edge_count": 0,
            "component_owner": component_owner,
        }

    vertex_by_id = {int(v["id"]): v for v in vertices}
    candidate_arrays: list[np.ndarray] = []
    neighbor_slices = (
        ((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
        ((slice(0, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(0, -1), slice(0, -1)), (slice(1, None), slice(1, None))),
        ((slice(0, -1), slice(1, None)), (slice(1, None), slice(0, -1))),
    )
    for (src_rows, src_cols), (dst_rows, dst_cols) in neighbor_slices:
        src_view = component_owner[src_rows, src_cols]
        dst_view = component_owner[dst_rows, dst_cols]
        valid = (src_view > 0) & (dst_view > 0) & (src_view != dst_view)
        if not np.any(valid):
            continue
        pairs = np.stack((src_view[valid], dst_view[valid]), axis=1).astype(np.int32, copy=False)
        candidate_arrays.append(pairs)
    candidate_pairs: set[tuple[int, int]] = set()
    if candidate_arrays:
        merged_pairs = np.unique(np.concatenate(candidate_arrays, axis=0), axis=0)
        candidate_pairs = {(int(pair[0]), int(pair[1])) for pair in merged_pairs.tolist()}
    edges: list[dict[str, object]] = []
    seen_edges: set[tuple[int, int]] = set()
    for left, right in sorted(candidate_pairs):
        a = vertex_by_id.get(int(left))
        b = vertex_by_id.get(int(right))
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        if int(a["time_label"]) == int(b["time_label"]):
            continue
        src_v, dst_v = (a, b) if int(a["time_label"]) < int(b["time_label"]) else (b, a)
        key = (int(src_v["id"]), int(dst_v["id"]))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        dx = float(dst_v["centroid"][0] - src_v["centroid"][0])
        dy = float(dst_v["centroid"][1] - src_v["centroid"][1])
        distance = float(max(1.0e-6, math.hypot(dx, dy)))
        downward_alignment = max(0.0, min(1.0, dy / distance))
        area_src = float(max(1, int(src_v["area_px"])))
        area_dst = float(max(1, int(dst_v["area_px"])))
        weight = (
            abs(area_src - area_dst) / max(1.0, area_src + area_dst)
            + max(0, int(dst_v["time_label"]) - int(src_v["time_label"]) - 1)
            + (1.0 - downward_alignment)
        )
        edges.append(
            {
                "src": int(src_v["id"]),
                "dst": int(dst_v["id"]),
                "src_time_label": int(src_v["time_label"]),
                "dst_time_label": int(dst_v["time_label"]),
                "weight": float(weight),
            }
        )
    return {
        "vertices": vertices,
        "edges": edges,
        "vertex_count": int(len(vertices)),
        "edge_count": int(len(edges)),
        "component_owner": component_owner,
    }


def _anchor_seed_mask(anchor_masks: list[np.ndarray], shape_hw: tuple[int, int]) -> np.ndarray:
    for mask in anchor_masks:
        mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
        if mask_u8.shape == shape_hw and int(np.count_nonzero(mask_u8)) > 0:
            return mask_u8
    return np.zeros(shape_hw, dtype=np.uint8)


def _component_anchor_overlap(component: dict[str, object], anchor_mask: np.ndarray) -> float:
    comp_mask = component.get("mask")
    if not isinstance(comp_mask, np.ndarray):
        return 0.0
    mask_u8 = (np.asarray(comp_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if mask_u8.shape != anchor_mask.shape or int(np.count_nonzero(mask_u8)) <= 0:
        return 0.0
    overlap = np.count_nonzero((mask_u8 > 0) & (anchor_mask > 0))
    return float(overlap) / float(max(1, np.count_nonzero(mask_u8)))


def _select_seed_components(
    apparition_map: np.ndarray,
    anchor_mask: np.ndarray,
    config: TwoDTTrackerConfig,
) -> tuple[list[dict[str, object]], int]:
    positive = [int(v) for v in np.unique(apparition_map) if int(v) > 0]
    if not positive:
        return [], -1
    h, _w = apparition_map.shape[:2]
    top_limit = max(1, int(round(h * float(config.top_band_ratio))))
    anchor_count = len(_component_list(anchor_mask, max(4, int(config.min_component_area // 2))))
    target = int(config.expected_track_count) if int(config.expected_track_count) > 0 else int(anchor_count)
    best_components: list[dict[str, object]] = []
    best_label = int(positive[0])
    for label_limit in positive:
        cumulative = ((apparition_map > 0) & (apparition_map <= int(label_limit))).astype(np.uint8)
        cumulative[top_limit:, :] = 0
        candidates = _component_list(cumulative, max(4, int(config.min_component_area // 2)))
        if not candidates:
            continue
        if int(np.count_nonzero(anchor_mask)) > 0:
            candidates.sort(
                key=lambda comp: (
                    _component_anchor_overlap(comp, anchor_mask),
                    float(comp["area"]),
                    -float(comp["centroid"][1]),
                ),
                reverse=True,
            )
        else:
            candidates.sort(
                key=lambda comp: (
                    float(comp["area"]),
                    -float(comp["centroid"][1]),
                ),
                reverse=True,
            )
        if target > 0 and len(candidates) >= target:
            chosen = sorted(candidates[:target], key=lambda comp: float(comp["centroid"][0]))
            return chosen, int(label_limit - 1)
        if len(candidates) > len(best_components):
            best_components = list(candidates)
            best_label = int(label_limit)
    best_components = sorted(best_components, key=lambda comp: float(comp["centroid"][0]))
    return best_components[:target] if target > 0 else best_components, int(best_label - 1)


def _lane_bounds(seed_components: list[dict[str, object]], width: int) -> dict[str, tuple[int, int]]:
    if not seed_components:
        return {}
    centers = [float(comp["centroid"][0]) for comp in seed_components]
    bounds: dict[str, tuple[int, int]] = {}
    for idx, _component in enumerate(seed_components):
        left = 0 if idx == 0 else int(round((centers[idx - 1] + centers[idx]) * 0.5))
        right = width if idx == len(seed_components) - 1 else int(round((centers[idx] + centers[idx + 1]) * 0.5))
        bounds[f"plant_{idx + 1:02d}"] = (max(0, left), min(width, right))
    return bounds


def _seed_vertices_from_components(graph: dict[str, object], seed_components: list[dict[str, object]]) -> dict[str, int]:
    component_owner = graph.get("component_owner")
    vertices = graph.get("vertices", [])
    if not isinstance(component_owner, np.ndarray) or component_owner.ndim != 2 or not isinstance(vertices, list):
        return {}
    vertex_by_id = {int(v["id"]): v for v in vertices if isinstance(v, dict) and "id" in v}
    out: dict[str, int] = {}
    for idx, comp in enumerate(seed_components, start=1):
        track_id = f"plant_{idx:02d}"
        mask = comp.get("mask")
        if not isinstance(mask, np.ndarray) or mask.shape != component_owner.shape:
            continue
        overlap_ids, counts = np.unique(component_owner[mask > 0], return_counts=True)
        positive = [(int(vid), int(cnt)) for vid, cnt in zip(overlap_ids.tolist(), counts.tolist()) if int(vid) > 0]
        chosen = -1
        if positive:
            chosen = max(positive, key=lambda item: item[1])[0]
        else:
            cx = float(comp["centroid"][0])
            cy = float(comp["centroid"][1])
            best_dist = float("inf")
            for vertex in vertices:
                if not isinstance(vertex, dict):
                    continue
                dist = math.hypot(float(vertex["centroid"][0]) - cx, float(vertex["centroid"][1]) - cy)
                if dist < best_dist:
                    best_dist = dist
                    chosen = int(vertex["id"])
        if chosen > 0 and chosen in vertex_by_id:
            out[track_id] = int(chosen)
    return out


def _augment_graph_with_reconnects(
    graph: dict[str, object],
    seed_vertex_ids: dict[str, int],
    lane_bounds: dict[str, tuple[int, int]],
    seed_components: list[dict[str, object]],
    config: TwoDTTrackerConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    vertices = graph.get("vertices", [])
    base_edges = list(graph.get("edges", [])) if isinstance(graph.get("edges"), list) else []
    if not isinstance(vertices, list) or not vertices:
        return base_edges, {"reconnect_edge_count": 0, "reconnect_candidate_count": 0}
    vertex_by_id = {int(v["id"]): v for v in vertices if isinstance(v, dict)}
    indegree: dict[int, int] = {int(v["id"]): 0 for v in vertices if isinstance(v, dict)}
    outdegree: dict[int, int] = {int(v["id"]): 0 for v in vertices if isinstance(v, dict)}
    for edge in base_edges:
        src = int(edge.get("src", -1))
        dst = int(edge.get("dst", -1))
        if src in outdegree:
            outdegree[src] += 1
        if dst in indegree:
            indegree[dst] += 1
    seed_vertex_set = {int(v) for v in seed_vertex_ids.values()}
    seed_center_by_track = {f"plant_{idx + 1:02d}": float(comp["centroid"][0]) for idx, comp in enumerate(seed_components)}
    candidates: list[tuple[float, dict[str, object]]] = []
    track_starts: dict[str, list[int]] = {track_id: [] for track_id in lane_bounds}
    track_stops: dict[str, list[int]] = {track_id: [] for track_id in lane_bounds}
    for vertex in vertices:
        if not isinstance(vertex, dict):
            continue
        vid = int(vertex["id"])
        cx = float(vertex["centroid"][0])
        assigned_track = None
        for track_id, (left, right) in lane_bounds.items():
            if left <= cx < right:
                assigned_track = track_id
                break
        if assigned_track is None:
            continue
        if vid not in seed_vertex_set and indegree.get(vid, 0) == 0:
            track_starts[assigned_track].append(vid)
        if outdegree.get(vid, 0) == 0:
            track_stops[assigned_track].append(vid)

    for track_id, starts in track_starts.items():
        stops = track_stops.get(track_id, [])
        lane = lane_bounds.get(track_id, (0, 0))
        seed_center = float(seed_center_by_track.get(track_id, sum(lane) * 0.5))
        for start_id in starts:
            start_v = vertex_by_id.get(int(start_id))
            if not isinstance(start_v, dict):
                continue
            for stop_id in stops:
                if int(stop_id) == int(start_id):
                    continue
                stop_v = vertex_by_id.get(int(stop_id))
                if not isinstance(stop_v, dict):
                    continue
                time_gap = int(start_v["time_label"]) - int(stop_v["time_label"])
                if time_gap <= 0 or time_gap > int(max(1, config.reconnect_max_time_gap)):
                    continue
                dx = float(start_v["centroid"][0] - stop_v["centroid"][0])
                dy = float(start_v["centroid"][1] - stop_v["centroid"][1])
                if dy < -6.0:
                    continue
                dist = float(math.hypot(dx, dy))
                lane_penalty = abs(float(start_v["centroid"][0]) - seed_center) / float(max(20.0, (lane[1] - lane[0]) * 0.5))
                cost = (0.55 * float(time_gap - 1)) + (dist / 80.0) + max(0.0, -dy / 40.0) + (0.20 * lane_penalty)
                if cost <= float(config.reconnect_max_cost):
                    candidates.append(
                        (
                            float(cost),
                            {
                                "src": int(stop_id),
                                "dst": int(start_id),
                                "src_time_label": int(stop_v["time_label"]),
                                "dst_time_label": int(start_v["time_label"]),
                                "weight": float(cost),
                                "reconnect": True,
                                "track_id": str(track_id),
                            },
                        )
                    )
    candidates.sort(key=lambda item: item[0])
    used_stops: set[int] = set()
    used_starts: set[int] = set()
    reconnect_edges: list[dict[str, object]] = []
    for _cost, edge in candidates:
        src = int(edge["src"])
        dst = int(edge["dst"])
        if src in used_stops or dst in used_starts:
            continue
        used_stops.add(src)
        used_starts.add(dst)
        reconnect_edges.append(edge)
    return base_edges + reconnect_edges, {
        "reconnect_edge_count": int(len(reconnect_edges)),
        "reconnect_candidate_count": int(len(candidates)),
    }


def _assign_vertices_to_tracks(
    graph: dict[str, object],
    edges: list[dict[str, object]],
    seed_vertex_ids: dict[str, int],
    lane_bounds: dict[str, tuple[int, int]],
    seed_components: list[dict[str, object]],
) -> tuple[dict[int, str], dict[str, float]]:
    vertices = graph.get("vertices", [])
    if not isinstance(vertices, list) or not vertices or not seed_vertex_ids:
        return {}, {}
    vertex_by_id = {int(v["id"]): v for v in vertices if isinstance(v, dict)}
    outgoing: dict[int, list[dict[str, object]]] = {vid: [] for vid in vertex_by_id}
    for edge in edges:
        src = int(edge.get("src", -1))
        if src in outgoing:
            outgoing[src].append(edge)
    seed_center_by_track = {f"plant_{idx + 1:02d}": float(comp["centroid"][0]) for idx, comp in enumerate(seed_components)}
    best_cost_by_track: dict[str, dict[int, float]] = {}
    for track_id, seed_vertex_id in seed_vertex_ids.items():
        lane = lane_bounds.get(track_id, (0, 0))
        lane_center = float(seed_center_by_track.get(track_id, sum(lane) * 0.5))
        pq: list[tuple[float, int]] = [(0.0, int(seed_vertex_id))]
        best: dict[int, float] = {int(seed_vertex_id): 0.0}
        while pq:
            cost, vid = heapq.heappop(pq)
            if cost > best.get(int(vid), float("inf")) + 1.0e-6:
                continue
            for edge in outgoing.get(int(vid), []):
                dst = int(edge.get("dst", -1))
                vertex = vertex_by_id.get(dst)
                if not isinstance(vertex, dict):
                    continue
                cx = float(vertex["centroid"][0])
                lane_penalty = 0.0
                if not (lane[0] <= cx < lane[1]):
                    lane_penalty = abs(cx - lane_center) / float(max(15.0, (lane[1] - lane[0]) * 0.35))
                anchor_penalty = abs(cx - lane_center) / float(max(20.0, (lane[1] - lane[0]) * 0.5))
                next_cost = float(cost) + float(edge.get("weight", 1.0)) + (0.85 * lane_penalty) + (0.10 * anchor_penalty)
                if next_cost + 1.0e-6 < best.get(dst, float("inf")):
                    best[dst] = next_cost
                    heapq.heappush(pq, (next_cost, dst))
        best_cost_by_track[track_id] = best

    assignments: dict[int, str] = {}
    winning_costs: dict[str, float] = {}
    for vid, vertex in vertex_by_id.items():
        best_track = None
        best_cost = float("inf")
        cx = float(vertex["centroid"][0])
        for track_id, costs in best_cost_by_track.items():
            lane = lane_bounds.get(track_id, (0, 0))
            lane_center = float(seed_center_by_track.get(track_id, sum(lane) * 0.5))
            if vid in costs:
                cost = float(costs[vid])
            else:
                lane_penalty = abs(cx - lane_center) / float(max(20.0, (lane[1] - lane[0]) * 0.5))
                cost = 50.0 + lane_penalty
            if cost < best_cost:
                best_cost = cost
                best_track = track_id
        if best_track is not None:
            assignments[int(vid)] = str(best_track)
    for track_id, costs in best_cost_by_track.items():
        finite = [float(v) for v in costs.values() if np.isfinite(float(v))]
        winning_costs[track_id] = float(min(finite)) if finite else float("inf")
    return assignments, winning_costs


def _track_bboxes_from_vertex_assignments(
    graph: dict[str, object],
    track_ids: list[str],
    assignments: dict[int, str],
    shifts: list[tuple[float, float]],
    config: TwoDTTrackerConfig,
    frame_count: int,
    shape_hw: tuple[int, int],
    registered_shape_hw: tuple[int, int],
) -> tuple[dict[str, list[BBox]], dict[str, int | None]]:
    vertices = graph.get("vertices", [])
    if not isinstance(vertices, list):
        return ({track_id: [(0, 0, 0, 0) for _ in range(frame_count)] for track_id in track_ids}, {track_id: None for track_id in track_ids})
    per_track_by_time: dict[str, list[BBox]] = {
        track_id: [(0, 0, 0, 0) for _ in range(frame_count)]
        for track_id in track_ids
    }
    for vertex in vertices:
        if not isinstance(vertex, dict):
            continue
        vid = int(vertex["id"])
        track_id = assignments.get(vid)
        if track_id not in per_track_by_time:
            continue
        frame_idx = max(0, min(frame_count - 1, int(vertex["time_label"]) - 1))
        per_track_by_time[track_id][frame_idx] = _bbox_union(
            per_track_by_time[track_id][frame_idx],
            _scale_bbox_to_shape(
                tuple(int(v) for v in vertex.get("bbox", (0, 0, 0, 0))),
                registered_shape_hw,
                shape_hw,
            ),
        )
    track_bboxes: dict[str, list[BBox]] = {track_id: [(0, 0, 0, 0) for _ in range(frame_count)] for track_id in track_ids}
    overlap_frames: dict[str, int | None] = {track_id: None for track_id in track_ids}
    cumulative_by_track: dict[str, BBox] = {track_id: (0, 0, 0, 0) for track_id in track_ids}
    for frame_idx in range(frame_count):
        for track_id in track_ids:
            cumulative_by_track[track_id] = _bbox_union(cumulative_by_track[track_id], per_track_by_time[track_id][frame_idx])
            reg_bbox = _expand_bbox(cumulative_by_track[track_id], int(config.bbox_padding), shape_hw)
            track_bboxes[track_id][frame_idx] = _inverse_shift_bbox(reg_bbox, shifts[frame_idx], shape_hw)
        for left_idx, left in enumerate(track_ids):
            for right in track_ids[left_idx + 1 :]:
                if _bbox_intersects(track_bboxes[left][frame_idx], track_bboxes[right][frame_idx]):
                    if overlap_frames[left] is None:
                        overlap_frames[left] = frame_idx
                    if overlap_frames[right] is None:
                        overlap_frames[right] = frame_idx
    return track_bboxes, overlap_frames


def build_arabidopsis_2dt_tracking_seed(
    timeline: list[DatasetImageItem],
    root_masks: list[np.ndarray],
    anchor_masks: list[np.ndarray] | None,
    config: TwoDTTrackerConfig,
    progress_callback: Callable[[str, float], bool] | None = None,
) -> dict[str, object]:
    if not timeline or not root_masks:
        return {
            "track_ids": [],
            "track_bboxes": {},
            "overlap_frames": {},
            "seed_frame": -1,
            "effective_min_component_area": int(max(1, int(config.min_component_area))),
            "source": "arabidopsis_2dt",
            "two_dt_summary": {"status": "empty"},
        }

    anchor_masks = anchor_masks if isinstance(anchor_masks, list) else [np.zeros_like(root_masks[0], dtype=np.uint8) for _ in root_masks]
    def _report(stage: str, fraction: float) -> None:
        if progress_callback is not None and not progress_callback(stage, max(0.0, min(1.0, float(fraction)))):
            raise TwoDTTrackerCancelled(f"cancelled during {stage}")

    try:
        _report("setup", 0.02)
        registered, shifts, registration_meta = _register_to_last_frame(
            timeline,
            int(max(256, int(config.registration_max_dim))),
            progress_callback=progress_callback,
        )
        shape_hw = timeline[0].image.shape[:2]
        registered_shape_hw = tuple(int(v) for v in registration_meta.get("registered_shape_hw", list(shape_hw)))
        registration_scale = float(registration_meta.get("registration_scale", 1.0))
        _report("warp_masks", 0.18)
        registered_root_masks = [
            _warp_binary_mask(
                _resize_binary_mask(mask, registered_shape_hw),
                (float(shifts[idx][0]) * registration_scale, float(shifts[idx][1]) * registration_scale),
                registered_shape_hw,
            )
            for idx, mask in enumerate(root_masks)
        ]
        registered_anchor_masks = [
            _warp_binary_mask(
                _resize_binary_mask(mask, registered_shape_hw),
                (float(shifts[idx][0]) * registration_scale, float(shifts[idx][1]) * registration_scale),
                registered_shape_hw,
            )
            for idx, mask in enumerate(anchor_masks)
        ]
        _report("apparition_map", 0.34)
        apparition_map = _compute_apparition_map(registered, registered_root_masks, config)
        _report("region_graph", 0.52)
        graph = _build_region_adjacency_graph(apparition_map)
        _report("seed_components", 0.66)
        anchor_seed = _anchor_seed_mask(registered_anchor_masks, apparition_map.shape[:2])
        seed_components, seed_frame = _select_seed_components(apparition_map, anchor_seed, config)
    except TwoDTTrackerCancelled:
        return {
            "track_ids": [],
            "track_bboxes": {},
            "overlap_frames": {},
            "seed_frame": -1,
            "effective_min_component_area": int(max(1, int(config.min_component_area))),
            "source": "arabidopsis_2dt",
            "two_dt_summary": {"status": "cancelled"},
        }
    if not seed_components:
        return {
            "track_ids": [],
            "track_bboxes": {},
            "overlap_frames": {},
            "seed_frame": -1,
            "effective_min_component_area": int(max(1, int(config.min_component_area))),
            "source": "arabidopsis_2dt",
            "two_dt_summary": {
                "status": "no_seed_components",
                "registration": registration_meta,
                "graph_vertex_count": int(graph.get("vertex_count", 0)),
                "graph_edge_count": int(graph.get("edge_count", 0)),
                "positive_apparition_labels": [int(v) for v in np.unique(apparition_map) if int(v) > 0],
            },
        }

    lanes = _lane_bounds(seed_components, shape_hw[1])
    track_ids = [f"plant_{idx + 1:02d}" for idx in range(len(seed_components))]
    min_area = max(1, int(config.min_component_area))
    _report("seed_vertices", 0.74)
    seed_vertex_ids = _seed_vertices_from_components(graph, seed_components)
    _report("reconnect", 0.82)
    edges_with_reconnect, reconnect_meta = _augment_graph_with_reconnects(graph, seed_vertex_ids, lanes, seed_components, config)
    _report("assign_tracks", 0.90)
    vertex_assignments, winning_costs = _assign_vertices_to_tracks(graph, edges_with_reconnect, seed_vertex_ids, lanes, seed_components)
    _report("track_bboxes", 0.97)
    track_bboxes, overlap_frames = _track_bboxes_from_vertex_assignments(
        graph,
        track_ids,
        vertex_assignments,
        shifts,
        config,
        len(timeline),
        shape_hw,
        registered_shape_hw,
    )
    _report("done", 1.0)

    summary = {
        "status": "ok",
        "registration": registration_meta,
        "seed_frame": int(max(-1, seed_frame)),
        "track_count": int(len(track_ids)),
        "graph_vertex_count": int(graph.get("vertex_count", 0)),
        "graph_edge_count": int(graph.get("edge_count", 0)),
        "graph_edge_count_augmented": int(len(edges_with_reconnect)),
        "seed_vertex_count": int(len(seed_vertex_ids)),
        "assigned_vertex_count": int(len(vertex_assignments)),
        "unassigned_vertex_count": int(max(0, int(graph.get("vertex_count", 0)) - len(vertex_assignments))),
        "winning_track_costs": {str(k): float(v) for k, v in winning_costs.items()},
        **reconnect_meta,
        "positive_apparition_labels": [int(v) for v in np.unique(apparition_map) if int(v) > 0],
        "apparition_component_count": int(len(_component_list((apparition_map > 0).astype(np.uint8), max(4, int(min_area // 2))))),
        "mode": "arabidopsis_2dt_graph_tracking",
    }
    return {
        "track_ids": track_ids,
        "track_bboxes": track_bboxes,
        "overlap_frames": overlap_frames,
        "seed_frame": int(max(-1, seed_frame)),
        "effective_min_component_area": int(min_area),
        "source": "arabidopsis_2dt",
        "two_dt_summary": summary,
    }
