from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np


PointXY = tuple[float, float]


@dataclass(frozen=True, slots=True)
class GuidedPolyline:
    """A user guide whose recovered pixels belong to one label class."""

    class_id: int
    points_xy: Sequence[PointXY]


@dataclass(frozen=True, slots=True)
class GuidedRootRescueConfig:
    """Routing and rasterization controls for guided root recovery."""

    line_width_px: int = 5
    corridor_px: int = 36
    frangi_sigmas: tuple[float, ...] = (0.8, 1.2, 1.8, 2.5)
    vesselness_weight: float = 1.0
    darkness_weight: float = 0.8
    existing_support_weight: float = 0.95
    guide_distance_weight: float = 0.65
    signal_cost_weight: float = 4.0
    minimum_step_cost: float = 0.05
    outside_corridor_cost: float = 1_000_000.0
    max_route_length_ratio: float = 4.0
    fully_connected: bool = True

    def __post_init__(self) -> None:
        if int(self.line_width_px) < 1:
            raise ValueError("line_width_px must be at least 1.")
        if int(self.corridor_px) < 0:
            raise ValueError("corridor_px cannot be negative.")
        if not self.frangi_sigmas or any(float(value) <= 0.0 for value in self.frangi_sigmas):
            raise ValueError("frangi_sigmas must contain positive values.")
        for name in (
            "vesselness_weight",
            "darkness_weight",
            "existing_support_weight",
            "guide_distance_weight",
            "signal_cost_weight",
            "minimum_step_cost",
            "outside_corridor_cost",
        ):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be a finite non-negative value.")
        if not math.isfinite(float(self.max_route_length_ratio)) or float(self.max_route_length_ratio) < 1.0:
            raise ValueError("max_route_length_ratio must be finite and at least 1.")


@dataclass(slots=True)
class GuidedRootRescueResult:
    addition_masks: dict[int, np.ndarray]
    metadata: dict[str, object]


def _optional_frangi() -> Callable[..., np.ndarray] | None:
    try:
        from skimage.filters import frangi  # type: ignore

        return frangi
    except Exception:
        return None


def _optional_route_through_array() -> Callable[..., tuple[list[tuple[int, int]], float]] | None:
    try:
        from skimage.graph import route_through_array  # type: ignore

        return route_through_array
    except Exception:
        return None


def _as_rgb_u8(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("image_rgb must have shape (height, width, 3).")
    image = image[:, :, :3]
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    values = np.asarray(image, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0)
    observed_max = float(np.max(values)) if values.size else 0.0
    if observed_max <= 1.0:
        values *= 255.0
    elif observed_max > 255.0:
        values *= 255.0 / observed_max
    return np.ascontiguousarray(np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8))


def _root_support_maps(
    existing_root_support: np.ndarray | None,
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if existing_root_support is None:
        empty = np.zeros(shape_hw, dtype=np.float32)
        return empty, empty.astype(bool)

    source = np.asarray(existing_root_support)
    if source.ndim == 3 and source.shape[2] == 1:
        source = source[:, :, 0]
    if source.ndim != 2 or source.shape != shape_hw:
        raise ValueError("existing_root_support must match the RGB image height and width.")

    finite = np.nan_to_num(np.asarray(source, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    occupied = finite > 0.0
    if np.issubdtype(source.dtype, np.floating) and float(np.max(finite, initial=0.0)) <= 1.0:
        support = np.clip(finite, 0.0, 1.0)
    else:
        support = occupied.astype(np.float32)
    return np.ascontiguousarray(support), np.ascontiguousarray(occupied)


def _robust_unit_interval(values: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    array = np.maximum(array, 0.0)
    positive = array[array > 0.0]
    if positive.size == 0:
        return np.zeros(array.shape, dtype=np.float32)
    scale = float(np.percentile(positive, 99.5))
    if not math.isfinite(scale) or scale <= 1.0e-12:
        scale = float(np.max(positive))
    if scale <= 1.0e-12:
        return np.zeros(array.shape, dtype=np.float32)
    return np.asarray(np.clip(array / scale, 0.0, 1.0), dtype=np.float32)


def _dark_thin_root_evidence(
    image_rgb: np.ndarray,
    config: GuidedRootRescueConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    gray_u8 = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_float = np.asarray(gray_u8, dtype=np.float32) / 255.0

    largest_sigma = max(float(value) for value in config.frangi_sigmas)
    radius = max(2, int(math.ceil(largest_sigma * 2.5)))
    kernel_size = (radius * 2) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, kernel)
    background = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=max(1.0, largest_sigma * 2.0))
    local_dark = np.maximum(
        np.asarray(blackhat, dtype=np.float32),
        np.maximum(np.asarray(background, dtype=np.float32) - np.asarray(gray_u8, dtype=np.float32), 0.0),
    )
    darkness = _robust_unit_interval(local_dark)

    frangi_fn = _optional_frangi()
    vesselness = np.zeros(gray_u8.shape, dtype=np.float32)
    frangi_error = ""
    if frangi_fn is not None:
        try:
            response = frangi_fn(
                gray_float,
                sigmas=tuple(float(value) for value in config.frangi_sigmas),
                black_ridges=True,
            )
            vesselness = _robust_unit_interval(np.asarray(response, dtype=np.float32))
        except Exception as exc:
            frangi_error = f"{type(exc).__name__}: {exc}"

    if frangi_fn is not None and not frangi_error:
        method = "frangi_dark_ridges_plus_morphological_darkness"
    elif frangi_fn is None:
        method = "morphological_darkness_fallback"
    else:
        method = "morphological_darkness_after_frangi_error"
    metadata: dict[str, object] = {
        "method": method,
        "frangi_available": bool(frangi_fn is not None),
        "frangi_used": bool(frangi_fn is not None and not frangi_error),
        "frangi_error": frangi_error,
        "darkness_nonzero_pixels": int(np.count_nonzero(darkness > 0.0)),
        "vesselness_nonzero_pixels": int(np.count_nonzero(vesselness > 0.0)),
    }
    return vesselness, darkness, metadata


def _round_and_clip_point(point: Sequence[float], shape_hw: tuple[int, int]) -> tuple[tuple[int, int], bool]:
    if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 2:
        raise ValueError("Each guide point must contain x and y coordinates.")
    x_raw = float(point[0])
    y_raw = float(point[1])
    if not math.isfinite(x_raw) or not math.isfinite(y_raw):
        raise ValueError("Guide coordinates must be finite.")
    x_rounded = int(math.floor(x_raw + 0.5))
    y_rounded = int(math.floor(y_raw + 0.5))
    height, width = shape_hw
    x = max(0, min(width - 1, x_rounded))
    y = max(0, min(height - 1, y_rounded))
    return (x, y), bool(x != x_rounded or y != y_rounded)


def _coerce_guides(
    guided_polylines: Sequence[GuidedPolyline | Mapping[str, object]],
    shape_hw: tuple[int, int],
) -> tuple[list[tuple[int, list[tuple[int, int]]]], int]:
    if not isinstance(guided_polylines, Sequence) or isinstance(guided_polylines, (str, bytes)):
        raise ValueError("guided_polylines must be a sequence of class-assigned polylines.")
    if not guided_polylines:
        raise ValueError("At least one guided polyline is required.")

    parsed: list[tuple[int, list[tuple[int, int]]]] = []
    clipped_count = 0
    for guide_index, raw_guide in enumerate(guided_polylines):
        if isinstance(raw_guide, GuidedPolyline):
            class_id_raw = raw_guide.class_id
            points_raw = raw_guide.points_xy
        elif isinstance(raw_guide, Mapping):
            class_id_raw = raw_guide.get("class_id")
            points_raw = raw_guide.get("points_xy", raw_guide.get("points"))
        else:
            raise ValueError(f"Guide {guide_index} must be GuidedPolyline or a mapping.")

        try:
            class_id = int(class_id_raw)
        except Exception as exc:
            raise ValueError(f"Guide {guide_index} has an invalid class_id.") from exc
        if class_id < 1 or class_id > 255:
            raise ValueError(f"Guide {guide_index} class_id must be between 1 and 255.")
        if not isinstance(points_raw, Sequence) or isinstance(points_raw, (str, bytes)):
            raise ValueError(f"Guide {guide_index} points must be a sequence.")

        points: list[tuple[int, int]] = []
        for point in points_raw:
            rounded, clipped = _round_and_clip_point(point, shape_hw)
            clipped_count += int(clipped)
            if not points or rounded != points[-1]:
                points.append(rounded)
        if len(points) < 2:
            raise ValueError(f"Guide {guide_index} must contain at least two distinct points.")
        parsed.append((class_id, points))
    return parsed, int(clipped_count)


def _bresenham_path_rc(start_xy: tuple[int, int], end_xy: tuple[int, int]) -> np.ndarray:
    x0, y0 = start_xy
    x1, y1 = end_xy
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    points: list[tuple[int, int]] = []
    while True:
        points.append((int(y0), int(x0)))
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy
    return np.asarray(points, dtype=np.int32)


def _path_length_px(path_rc: np.ndarray) -> float:
    if path_rc.shape[0] < 2:
        return 0.0
    deltas = np.diff(np.asarray(path_rc, dtype=np.float64), axis=0)
    return float(np.hypot(deltas[:, 0], deltas[:, 1]).sum())


def _route_segment(
    start_xy: tuple[int, int],
    end_xy: tuple[int, int],
    vesselness: np.ndarray,
    darkness: np.ndarray,
    root_support: np.ndarray,
    route_fn: Callable[..., tuple[list[tuple[int, int]], float]] | None,
    config: GuidedRootRescueConfig,
) -> tuple[np.ndarray, str, str, float | None]:
    straight_path = _bresenham_path_rc(start_xy, end_xy)
    if route_fn is None:
        return straight_path, "straight_fallback", "shortest_path_unavailable", None

    height, width = vesselness.shape
    margin = int(config.corridor_px) + int(math.ceil(config.line_width_px / 2.0)) + 2
    x0 = max(0, min(start_xy[0], end_xy[0]) - margin)
    x1 = min(width, max(start_xy[0], end_xy[0]) + margin + 1)
    y0 = max(0, min(start_xy[1], end_xy[1]) - margin)
    y1 = min(height, max(start_xy[1], end_xy[1]) + margin + 1)
    local_shape = (y1 - y0, x1 - x0)
    local_start = (int(start_xy[1] - y0), int(start_xy[0] - x0))
    local_end = (int(end_xy[1] - y0), int(end_xy[0] - x0))

    guide_line = np.zeros(local_shape, dtype=np.uint8)
    cv2.line(
        guide_line,
        (local_start[1], local_start[0]),
        (local_end[1], local_end[0]),
        1,
        thickness=1,
        lineType=cv2.LINE_8,
    )
    distance = cv2.distanceTransform((guide_line == 0).astype(np.uint8), cv2.DIST_L2, 3)
    corridor_limit = float(config.corridor_px) + 0.5
    inside_corridor = distance <= corridor_limit

    vessel_local = vesselness[y0:y1, x0:x1]
    dark_local = darkness[y0:y1, x0:x1]
    support_local = root_support[y0:y1, x0:x1]
    signal = np.maximum(
        np.asarray(vessel_local, dtype=np.float64) * float(config.vesselness_weight),
        np.asarray(dark_local, dtype=np.float64) * float(config.darkness_weight),
    )
    signal = np.maximum(signal, np.asarray(support_local, dtype=np.float64) * float(config.existing_support_weight))
    signal = np.clip(signal, 0.0, 1.0)
    distance_scale = max(1.0, float(config.corridor_px))
    distance_penalty = np.square(np.clip(np.asarray(distance, dtype=np.float64) / distance_scale, 0.0, 1.0))
    cost = (
        float(config.minimum_step_cost)
        + (float(config.signal_cost_weight) * (1.0 - signal))
        + (float(config.guide_distance_weight) * distance_penalty)
    )
    cost[~inside_corridor] = float(config.outside_corridor_cost)
    cost[local_start] = min(float(cost[local_start]), float(config.minimum_step_cost))
    cost[local_end] = min(float(cost[local_end]), float(config.minimum_step_cost))

    try:
        indices, total_cost = route_fn(
            np.ascontiguousarray(cost, dtype=np.float64),
            local_start,
            local_end,
            fully_connected=bool(config.fully_connected),
            geometric=True,
        )
        local_path = np.asarray(indices, dtype=np.int32)
    except Exception as exc:
        reason = f"routing_error:{type(exc).__name__}"
        return straight_path, "straight_fallback", reason, None

    if local_path.ndim != 2 or local_path.shape[0] < 2 or local_path.shape[1] != 2:
        return straight_path, "straight_fallback", "invalid_route_shape", None
    rows = local_path[:, 0]
    cols = local_path[:, 1]
    if (
        np.any(rows < 0)
        or np.any(cols < 0)
        or np.any(rows >= local_shape[0])
        or np.any(cols >= local_shape[1])
        or np.any(~inside_corridor[rows, cols])
    ):
        return straight_path, "straight_fallback", "route_left_corridor", None

    path = local_path.copy()
    path[:, 0] += int(y0)
    path[:, 1] += int(x0)
    straight_distance = max(1.0, float(math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])))
    if _path_length_px(path) > straight_distance * float(config.max_route_length_ratio):
        return straight_path, "straight_fallback", "route_exceeded_length_limit", None
    return path, "shortest_path", "", float(total_cost)


def _paint_path(mask: np.ndarray, path_rc: np.ndarray, width_px: int) -> None:
    if path_rc.size == 0:
        return
    points_xy = np.ascontiguousarray(path_rc[:, ::-1].reshape((-1, 1, 2)), dtype=np.int32)
    if points_xy.shape[0] == 1:
        radius = max(0, int(width_px) // 2)
        cv2.circle(mask, tuple(int(value) for value in points_xy[0, 0]), radius, 1, thickness=-1, lineType=cv2.LINE_8)
        return
    cv2.polylines(mask, [points_xy], False, 1, thickness=int(width_px), lineType=cv2.LINE_8)


def _mean_along_path(values: np.ndarray, path_rc: np.ndarray) -> float:
    if path_rc.size == 0:
        return 0.0
    rows = path_rc[:, 0]
    cols = path_rc[:, 1]
    return float(np.mean(np.asarray(values[rows, cols], dtype=np.float64)))


def _config_metadata(config: GuidedRootRescueConfig) -> dict[str, object]:
    return {
        "line_width_px": int(config.line_width_px),
        "corridor_px": int(config.corridor_px),
        "frangi_sigmas": [float(value) for value in config.frangi_sigmas],
        "vesselness_weight": float(config.vesselness_weight),
        "darkness_weight": float(config.darkness_weight),
        "existing_support_weight": float(config.existing_support_weight),
        "guide_distance_weight": float(config.guide_distance_weight),
        "signal_cost_weight": float(config.signal_cost_weight),
        "minimum_step_cost": float(config.minimum_step_cost),
        "outside_corridor_cost": float(config.outside_corridor_cost),
        "max_route_length_ratio": float(config.max_route_length_ratio),
        "fully_connected": bool(config.fully_connected),
    }


def compute_guided_root_rescue(
    image_rgb: np.ndarray,
    existing_root_support: np.ndarray | None,
    guided_polylines: Sequence[GuidedPolyline | Mapping[str, object]],
    config: GuidedRootRescueConfig | None = None,
) -> GuidedRootRescueResult:
    """Recover guided root paths without modifying the existing root mask.

    Returned masks contain additions only: any pixel already present in
    ``existing_root_support`` is removed from every class result.
    """

    effective_config = config or GuidedRootRescueConfig()
    image = _as_rgb_u8(image_rgb)
    shape_hw = (int(image.shape[0]), int(image.shape[1]))
    support, occupied_support = _root_support_maps(existing_root_support, shape_hw)
    guides, clipped_prompt_points = _coerce_guides(guided_polylines, shape_hw)
    vesselness, darkness, evidence_metadata = _dark_thin_root_evidence(image, effective_config)
    route_fn = _optional_route_through_array()

    class_ids = sorted({int(class_id) for class_id, _points in guides})
    routed_masks = {class_id: np.zeros(shape_hw, dtype=np.uint8) for class_id in class_ids}
    guide_counts = {class_id: 0 for class_id in class_ids}
    segment_counts = {class_id: 0 for class_id in class_ids}
    segment_metadata: list[dict[str, object]] = []
    routed_segment_count = 0
    fallback_segment_count = 0

    for guide_index, (class_id, points) in enumerate(guides):
        guide_counts[class_id] += 1
        for segment_index, (start_xy, end_xy) in enumerate(zip(points[:-1], points[1:])):
            path, method, fallback_reason, route_cost = _route_segment(
                start_xy,
                end_xy,
                vesselness,
                darkness,
                support,
                route_fn,
                effective_config,
            )
            _paint_path(routed_masks[class_id], path, int(effective_config.line_width_px))
            segment_counts[class_id] += 1
            if method == "shortest_path":
                routed_segment_count += 1
            else:
                fallback_segment_count += 1
            segment_metadata.append(
                {
                    "guide_index": int(guide_index),
                    "segment_index": int(segment_index),
                    "class_id": int(class_id),
                    "start_xy": [int(start_xy[0]), int(start_xy[1])],
                    "end_xy": [int(end_xy[0]), int(end_xy[1])],
                    "method": method,
                    "fallback_reason": fallback_reason,
                    "path_points": int(path.shape[0]),
                    "path_length_px": float(_path_length_px(path)),
                    "route_cost": route_cost,
                    "mean_vesselness": _mean_along_path(vesselness, path),
                    "mean_darkness": _mean_along_path(darkness, path),
                    "mean_existing_support": _mean_along_path(support, path),
                }
            )

    class_metadata: dict[str, dict[str, int]] = {}
    addition_masks: dict[int, np.ndarray] = {}
    for class_id in class_ids:
        routed = (routed_masks[class_id] > 0).astype(np.uint8)
        overlap = int(np.count_nonzero((routed > 0) & occupied_support))
        addition = routed.copy()
        addition[occupied_support] = 0
        addition_masks[class_id] = np.ascontiguousarray(addition, dtype=np.uint8)
        class_metadata[str(class_id)] = {
            "guide_count": int(guide_counts[class_id]),
            "segment_count": int(segment_counts[class_id]),
            "routed_pixels": int(np.count_nonzero(routed)),
            "existing_support_overlap_removed_px": overlap,
            "addition_pixels": int(np.count_nonzero(addition)),
        }

    sum_addition_pixels = int(sum(np.count_nonzero(mask) for mask in addition_masks.values()))
    if addition_masks:
        union = np.maximum.reduce(list(addition_masks.values()))
        union_addition_pixels = int(np.count_nonzero(union))
    else:  # Defensive; _coerce_guides currently prevents this case.
        union_addition_pixels = 0
    metadata: dict[str, object] = {
        "algorithm": "guided_root_rescue",
        "algorithm_version": 1,
        "image_shape_hw": [int(shape_hw[0]), int(shape_hw[1])],
        "config": _config_metadata(effective_config),
        "evidence": evidence_metadata,
        "shortest_path_available": bool(route_fn is not None),
        "guide_count": int(len(guides)),
        "segment_count": int(len(segment_metadata)),
        "routed_segment_count": int(routed_segment_count),
        "fallback_segment_count": int(fallback_segment_count),
        "clipped_prompt_points": int(clipped_prompt_points),
        "existing_support_pixels": int(np.count_nonzero(occupied_support)),
        "sum_class_addition_pixels": sum_addition_pixels,
        "union_addition_pixels": union_addition_pixels,
        "interclass_overlap_pixels": int(sum_addition_pixels - union_addition_pixels),
        "classes": class_metadata,
        "segments": segment_metadata,
    }
    return GuidedRootRescueResult(addition_masks=addition_masks, metadata=metadata)


__all__ = [
    "GuidedPolyline",
    "GuidedRootRescueConfig",
    "GuidedRootRescueResult",
    "compute_guided_root_rescue",
]
