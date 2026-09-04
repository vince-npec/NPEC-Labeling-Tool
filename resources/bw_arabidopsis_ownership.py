from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .models import DatasetImageItem

if TYPE_CHECKING:
    from .pyphenotyper_adapter import PyPhenotyperConfig


BBox = tuple[int, int, int, int]


def _coerce_positive_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed <= 0.0:
        return None
    return parsed


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def _largest_centered_component(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if np.count_nonzero(binary) <= 0:
        return binary
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape[:2]
    center_x = float(width) * 0.5
    center_y = float(height) * 0.5
    best_label = 0
    best_score = -1.0e18
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


def _estimate_dish_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    gray = _ensure_gray(image)
    blurred = cv2.GaussianBlur(gray, (0, 0), 5.0)
    _thr, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    interior = _largest_centered_component(thresh > 0)
    if np.count_nonzero(interior) <= 0:
        interior = _largest_centered_component(gray > 40)
    ys, xs = np.where(interior > 0)
    if xs.size <= 0 or ys.size <= 0:
        return (0, 0, int(gray.shape[1]), int(gray.shape[0]))
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _default_expected_centers(
    shape_hw: tuple[int, int],
    plant_count: int,
    dish_bbox: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int, int]]:
    height, width = int(shape_hw[0]), int(shape_hw[1])
    if dish_bbox is None:
        x0, y0, x1, y1 = (0, 0, width, height)
    else:
        x0, y0, x1, y1 = dish_bbox
    dish_width = max(1, int(x1 - x0))
    dish_height = max(1, int(y1 - y0))
    # Match the Hades five-lane layout while adapting to the actual
    # detected dish footprint.
    xs = np.linspace(x0 + (dish_width * 0.14), x0 + (dish_width * 0.86), max(1, int(plant_count)))
    cy = int(round(y0 + (dish_height * 0.18)))
    centers: list[tuple[int, int, int]] = []
    for idx, cx in enumerate(xs, start=1):
        centers.append((int(round(cx)), int(cy), int(idx)))
    return centers


def _yx_box_to_xywh(box: tuple[int, int, int, int], shape_hw: tuple[int, int]) -> BBox:
    ymin, ymax, xmin, xmax = [int(v) for v in box]
    height, width = shape_hw
    ymin = max(0, min(ymin, height))
    ymax = max(ymin, min(ymax, height))
    xmin = max(0, min(xmin, width))
    xmax = max(xmin, min(xmax, width))
    return (int(xmin), int(ymin), int(max(0, xmax - xmin)), int(max(0, ymax - ymin)))


def _bbox_intersects(a: BBox, b: BBox) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return False
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _nonoverlapping_lane_boxes(
    shape_hw: tuple[int, int],
    expected_centers: list[tuple[int, int, int]],
    dish_bbox: tuple[int, int, int, int],
) -> list[BBox]:
    height, width = int(shape_hw[0]), int(shape_hw[1])
    x0, y0, x1, y1 = [int(value) for value in dish_bbox]
    x0 = max(0, min(x0, width))
    x1 = max(x0, min(x1, width))
    y0 = max(0, min(y0, height))
    y1 = max(y0, min(y1, height))
    centers = sorted(int(center[0]) for center in expected_centers)
    if not centers:
        return []
    boundaries = [x0]
    boundaries.extend(
        int(round((float(left) + float(right)) * 0.5))
        for left, right in zip(centers[:-1], centers[1:], strict=False)
    )
    boundaries.append(x1)
    boxes: list[BBox] = []
    for index in range(len(centers)):
        left = max(x0, min(int(boundaries[index]), x1))
        right = max(left, min(int(boundaries[index + 1]), x1))
        boxes.append((left, y0, max(0, right - left), max(0, y1 - y0)))
    return boxes


def _crown_aligned_centers(
    shoot_mask: np.ndarray,
    expected_centers: list[tuple[int, int, int]],
    dish_bbox: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int, int]], str]:
    """Refine lane centers from the lower shoot pixels without claiming ownership."""
    if not expected_centers:
        return [], "unavailable"
    binary = (np.asarray(shoot_mask, dtype=np.uint8) > 0).astype(np.uint8)
    preliminary_boxes = _nonoverlapping_lane_boxes(binary.shape[:2], expected_centers, dish_bbox)
    measured_x: list[int | None] = []
    for bbox in preliminary_boxes:
        x, y, w, h = bbox
        crop = binary[y : y + h, x : x + w]
        ys, xs = np.where(crop > 0)
        if xs.size < 20:
            measured_x.append(None)
            continue
        lower_cutoff = float(np.quantile(ys, 0.70))
        crown_xs = xs[ys >= lower_cutoff]
        if crown_xs.size < 5:
            crown_xs = xs
        measured_x.append(int(round(float(x) + float(np.median(crown_xs)))))

    valid_indices = [index for index, value in enumerate(measured_x) if value is not None]
    minimum_valid = max(2, int(np.ceil(len(expected_centers) * 0.5)))
    if len(valid_indices) < minimum_valid:
        return list(expected_centers), "dish_geometry"

    offsets = [
        int(measured_x[index]) - int(expected_centers[index][0])
        for index in valid_indices
    ]
    global_offset = int(round(float(np.median(offsets))))
    x0, _y0, x1, _y1 = [int(value) for value in dish_bbox]
    dish_width = max(1, x1 - x0)
    minimum_separation = max(4, int(round(dish_width * 0.045)))
    refined_x = [
        int(measured_x[index])
        if measured_x[index] is not None
        else int(expected_centers[index][0]) + global_offset
        for index in range(len(expected_centers))
    ]
    refined_x[0] = int(np.clip(refined_x[0], x0, max(x0, x1 - 1)))
    for index in range(1, len(refined_x)):
        refined_x[index] = max(refined_x[index], refined_x[index - 1] + minimum_separation)
    if refined_x[-1] >= x1:
        shift = refined_x[-1] - max(x0, x1 - 1)
        refined_x = [value - shift for value in refined_x]
    for index in range(len(refined_x) - 2, -1, -1):
        refined_x[index] = min(refined_x[index], refined_x[index + 1] - minimum_separation)
    refined_x = [int(np.clip(value, x0, max(x0, x1 - 1))) for value in refined_x]
    return [
        (refined_x[index], int(expected_centers[index][1]), int(expected_centers[index][2]))
        for index in range(len(expected_centers))
    ], "shoot_crown_pixels"


def _crop_to_bbox(mask: np.ndarray, bbox: BBox) -> np.ndarray:
    x, y, w, h = [int(value) for value in bbox]
    if w <= 0 or h <= 0:
        return np.zeros((0, 0), dtype=np.uint8)
    return (np.asarray(mask[y : y + h, x : x + w], dtype=np.uint8) > 0).astype(np.uint8)


def _lane_overlap_risk(
    root_union: np.ndarray,
    lane_boxes: list[BBox],
    *,
    boundary_radius: int = 4,
) -> list[bool]:
    root_binary = np.asarray(root_union, dtype=np.uint8) > 0
    _height, width = root_binary.shape[:2]
    risk = [False for _ in lane_boxes]
    radius = max(1, int(boundary_radius))
    for boundary_index in range(len(lane_boxes) - 1):
        left_box = lane_boxes[boundary_index]
        boundary_x = int(left_box[0] + left_box[2])
        strip_x0 = max(0, boundary_x - radius)
        strip_x1 = min(width, boundary_x + radius + 1)
        if strip_x1 <= strip_x0:
            continue
        if np.any(root_binary[:, strip_x0:strip_x1]):
            risk[boundary_index] = True
            risk[boundary_index + 1] = True
    return risk


def _mask_perimeter_px(mask: np.ndarray) -> float:
    bin_mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(sum(float(cv2.arcLength(contour, True)) for contour in contours))


def _skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    mask_bin = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if np.count_nonzero(mask_bin) == 0:
        return mask_bin.astype(bool)
    image = (mask_bin * 255).astype(np.uint8)
    skeleton = np.zeros_like(image, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(image, element)
        opened = cv2.dilate(eroded, element)
        edge = cv2.subtract(image, opened)
        skeleton = cv2.bitwise_or(skeleton, edge)
        image = eroded
        if cv2.countNonZero(image) == 0:
            break
    return skeleton > 0


def _measure_mask_length_px(mask: np.ndarray) -> tuple[float, int, float]:
    bin_mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    area_px = int(np.count_nonzero(bin_mask))
    if area_px <= 0:
        return 0.0, 0, 0.0
    skeleton = _skeletonize_binary(bin_mask)
    length_px = float(np.count_nonzero(skeleton))
    perimeter_px = float(_mask_perimeter_px(bin_mask))
    if length_px <= 0.0:
        if perimeter_px > 0.0:
            length_px = float((2.0 * area_px) / perimeter_px)
        else:
            length_px = float(np.sqrt(float(area_px)))
    return length_px, area_px, perimeter_px


def _grow_bounding_boxes(
    binary_image: np.ndarray,
    expected_centers: list[tuple[int, int, int]],
    *,
    initial_box_halfsize_x: int,
    initial_box_halfsize_y: int,
    expansion_step: int = 2,
    max_empty_expansions: int = 120,
    stop_on_overlap: bool = True,
) -> list[tuple[int, int, int, int]]:
    binary = (np.asarray(binary_image, dtype=np.uint8) > 0).astype(np.uint8)
    if np.count_nonzero(binary) > 0:
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
    height, width = binary.shape[:2]

    final_boxes: list[list[int]] = []
    empty_counters: list[dict[str, int]] = []
    can_expand: list[dict[str, bool]] = []

    for cx, cy, _ in expected_centers:
        ymin_raw = max(int(cy) - int(initial_box_halfsize_y), 0)
        ymax_raw = min(int(cy) + int(initial_box_halfsize_y), height)
        xmin_raw = max(int(cx) - int(initial_box_halfsize_x), 0)
        xmax_raw = min(int(cx) + int(initial_box_halfsize_x), width)

        region = binary[ymin_raw:ymax_raw, xmin_raw:xmax_raw]
        ys, xs = np.where(region > 0)
        ymin, ymax, xmin, xmax = ymin_raw, ymax_raw, xmin_raw, xmax_raw
        if xs.size > 0 and ys.size > 0:
            ymin = max(ymin_raw + int(ys.min()) - 1, 0)
            ymax = min(ymin_raw + int(ys.max()) + 2, height)
            xmin = max(xmin_raw + int(xs.min()) - 1, 0)
            xmax = min(xmin_raw + int(xs.max()) + 2, width)

        final_boxes.append([ymin, ymax, xmin, xmax])
        empty_counters.append({"up": 0, "down": 0, "left": 0, "right": 0})
        can_expand.append({"up": True, "down": True, "left": True, "right": True})

    def overlaps_any(proposed_box: list[int], current_idx: int) -> bool:
        pymin, pymax, pxmin, pxmax = proposed_box
        for idx, (oymin, oymax, oxmin, oxmax) in enumerate(final_boxes):
            if idx == current_idx:
                continue
            if not (pxmax <= oxmin or pxmin >= oxmax or pymax <= oymin or pymin >= oymax):
                return True
        return False

    still_expanding = True
    while still_expanding:
        still_expanding = False
        for idx in range(len(final_boxes)):
            ymin, ymax, xmin, xmax = final_boxes[idx]
            for direction in ("up", "down", "left", "right"):
                if not can_expand[idx][direction]:
                    continue
                if direction == "up":
                    new_ymin = max(ymin - expansion_step, 0)
                    region = binary[new_ymin:ymin, xmin:xmax]
                    proposed = [new_ymin, ymax, xmin, xmax]
                elif direction == "down":
                    new_ymax = min(ymax + expansion_step, height)
                    region = binary[ymax:new_ymax, xmin:xmax]
                    proposed = [ymin, new_ymax, xmin, xmax]
                elif direction == "left":
                    new_xmin = max(xmin - expansion_step, 0)
                    region = binary[ymin:ymax, new_xmin:xmin]
                    proposed = [ymin, ymax, new_xmin, xmax]
                else:
                    new_xmax = min(xmax + expansion_step, width)
                    region = binary[ymin:ymax, xmax:new_xmax]
                    proposed = [ymin, ymax, xmin, new_xmax]

                if stop_on_overlap and overlaps_any(proposed, idx):
                    can_expand[idx][direction] = False
                    continue

                if np.any(region):
                    final_boxes[idx] = proposed
                    empty_counters[idx][direction] = 0
                    still_expanding = True
                else:
                    empty_counters[idx][direction] += 1
                    if empty_counters[idx][direction] >= max_empty_expansions:
                        can_expand[idx][direction] = False

    return [(int(y0), int(y1), int(x0), int(x1)) for y0, y1, x0, x1 in final_boxes]


def _find_shoot_regions(
    shoot_mask: np.ndarray,
    expected_centers: list[tuple[int, int, int]],
    *,
    initial_box_halfsize_x: int,
    initial_box_halfsize_y: int,
    expansion_step: int = 1,
    max_empty_expansions: int = 200,
) -> list[np.ndarray]:
    binary = (np.asarray(shoot_mask, dtype=np.uint8) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label_id in range(1, int(count)):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= 50:
            cleaned[labels == label_id] = 1
    binary = cleaned
    height, width = binary.shape[:2]
    final_boxes: list[list[int]] = []
    shoot_masks: list[np.ndarray] = []

    def overlaps_any(new_box: list[int], existing_boxes: list[list[int]]) -> bool:
        y0, y1, x0, x1 = new_box
        for ey0, ey1, ex0, ex1 in existing_boxes:
            if not (x1 <= ex0 or x0 >= ex1 or y1 <= ey0 or y0 >= ey1):
                return True
        return False

    for cx, cy, _ in expected_centers:
        ymin = max(int(cy) - int(initial_box_halfsize_y), 0)
        ymax = min(int(cy) + int(initial_box_halfsize_y), height)
        xmin = max(int(cx) - int(initial_box_halfsize_x), 0)
        xmax = min(int(cx) + int(initial_box_halfsize_x), width)
        empty_counts = {"up": 0, "down": 0, "left": 0, "right": 0}
        can_expand = {"up": True, "down": True, "left": True, "right": True}
        while any(can_expand.values()):
            for direction in ("up", "down", "left", "right"):
                if not can_expand[direction]:
                    continue
                if direction == "up":
                    y0 = max(ymin - expansion_step, 0)
                    new_strip = binary[y0:ymin, xmin:xmax]
                    proposed = [y0, ymax, xmin, xmax]
                elif direction == "down":
                    y1 = min(ymax + expansion_step, height)
                    new_strip = binary[ymax:y1, xmin:xmax]
                    proposed = [ymin, y1, xmin, xmax]
                elif direction == "left":
                    x0 = max(xmin - expansion_step, 0)
                    new_strip = binary[ymin:ymax, x0:xmin]
                    proposed = [ymin, ymax, x0, xmax]
                else:
                    x1 = min(xmax + expansion_step, width)
                    new_strip = binary[ymin:ymax, xmax:x1]
                    proposed = [ymin, ymax, xmin, x1]
                if np.any(new_strip) and not overlaps_any(proposed, final_boxes):
                    ymin, ymax, xmin, xmax = proposed
                    empty_counts[direction] = 0
                else:
                    empty_counts[direction] += 1
                    if empty_counts[direction] >= max_empty_expansions:
                        can_expand[direction] = False

        final_boxes.append([ymin, ymax, xmin, xmax])
        region = binary[ymin:ymax, xmin:xmax]
        region_mask = np.zeros_like(binary)
        region_mask[ymin:ymax, xmin:xmax] = region
        shoot_masks.append(region_mask.astype(np.uint8))
    return shoot_masks


def _shoot_mask_is_plausible(shoot_mask: np.ndarray, expected_count: int) -> bool:
    binary = (np.asarray(shoot_mask, dtype=np.uint8) > 0).astype(np.uint8)
    area = int(np.count_nonzero(binary))
    if area <= 0:
        return False
    height, width = binary.shape[:2]
    image_area = max(1, int(height * width))
    min_area = max(20, int(round(float(image_area) * 0.00001)))
    max_area = max(min_area + 1, int(round(float(image_area) * 0.22)))
    if area < min_area or area > max_area:
        return False
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_areas = [
        int(stats[label_id, cv2.CC_STAT_AREA])
        for label_id in range(1, int(count))
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= min_area
    ]
    if not component_areas:
        return False
    return len(component_areas) <= max(1, int(expected_count) * 4)


def _extract_owned_root_component(
    root_mask: np.ndarray,
    shoot_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    center_x: int,
) -> np.ndarray:
    ymin, ymax, xmin, xmax = bbox
    if ymax <= ymin or xmax <= xmin:
        return np.zeros_like(root_mask, dtype=np.uint8)
    root_crop = (np.asarray(root_mask[ymin:ymax, xmin:xmax], dtype=np.uint8) > 0).astype(np.uint8)
    if np.count_nonzero(root_crop) <= 0:
        return np.zeros_like(root_mask, dtype=np.uint8)
    shoot_crop = (np.asarray(shoot_mask[ymin:ymax, xmin:xmax], dtype=np.uint8) > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(root_crop, connectivity=8)
    if count <= 1:
        out = np.zeros_like(root_mask, dtype=np.uint8)
        out[ymin:ymax, xmin:xmax] = root_crop
        return out

    selected_label = 0
    best_score = -1.0e18
    local_center_x = float(center_x - xmin)
    shoot_touch = cv2.dilate(shoot_crop, np.ones((5, 5), np.uint8), iterations=1).astype(bool) if np.count_nonzero(shoot_crop) > 0 else None
    for label_id in range(1, int(count)):
        component = labels == label_id
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        score = float(area)
        cx, cy = centroids[label_id]
        score -= abs(float(cx) - local_center_x) * 0.45
        ys, xs = np.where(component)
        if ys.size > 0:
            top_y = float(ys.min())
            score -= top_y * 0.02
        if shoot_touch is not None:
            overlap = int(np.count_nonzero(component & shoot_touch))
            score += float(overlap) * 8.0
        if score > best_score:
            best_score = score
            selected_label = int(label_id)
    owned_local = (labels == selected_label).astype(np.uint8)
    out = np.zeros_like(root_mask, dtype=np.uint8)
    out[ymin:ymax, xmin:xmax] = owned_local
    return out


def build_bw_arabidopsis_measurements(
    items: list[DatasetImageItem],
    predictions: dict[str, np.ndarray],
    config: PyPhenotyperConfig,
    metadata: dict[str, dict[str, object]] | None = None,
    progress_callback: Callable[[int, int], bool] | None = None,
) -> dict[str, object]:
    timeline_items: list[DatasetImageItem] = [item for item in items if item.uid in predictions]
    if not timeline_items:
        return {"summary": {}, "per_uid": {}}

    overrides = config.pipeline_overrides if isinstance(config.pipeline_overrides, dict) else {}
    plant_count = max(1, int(overrides.get("expected_plant_count", 5) or 5))
    lateral_class_id_raw = overrides.get("lateral_class_id")
    try:
        lateral_class_id = int(lateral_class_id_raw) if lateral_class_id_raw not in (None, "") else None
    except (TypeError, ValueError):
        lateral_class_id = None
    if lateral_class_id is not None and lateral_class_id <= 0:
        lateral_class_id = None
    track_ids = [f"lane_{index:02d}" for index in range(1, plant_count + 1)]

    per_uid: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    overlap_frames: dict[str, int | None] = {track_id: None for track_id in track_ids}
    dataset_pixel_sizes: dict[str, float] = {}

    for frame_index, item in enumerate(timeline_items):
        pred = np.asarray(predictions[item.uid], dtype=np.uint8)
        root_mask = (pred == np.uint8(int(config.root_class_id))).astype(np.uint8)
        lateral_mask = (
            (pred == np.uint8(int(lateral_class_id))).astype(np.uint8)
            if lateral_class_id is not None and int(lateral_class_id) != int(config.root_class_id)
            else np.zeros_like(root_mask, dtype=np.uint8)
        )
        root_union = np.logical_or(root_mask > 0, lateral_mask > 0).astype(np.uint8)
        shoot_mask = (pred == np.uint8(int(config.shoot_class_id))).astype(np.uint8)
        shape_hw = root_mask.shape[:2]
        item_meta = metadata.get(item.uid, {}) if isinstance(metadata, dict) else {}
        pixel_size_mm = _coerce_positive_float(item_meta.get("pixel_size_mm")) or float(config.pixel_size_mm)
        dataset_pixel_sizes[item.uid] = float(pixel_size_mm)
        shoot_mask_plausible = _shoot_mask_is_plausible(shoot_mask, plant_count)
        shoot_for_support = shoot_mask if shoot_mask_plausible else np.zeros_like(shoot_mask, dtype=np.uint8)

        dish_bbox = _estimate_dish_bbox(item.image)
        expected_centers = _default_expected_centers(shape_hw, plant_count, dish_bbox=dish_bbox)
        expected_centers, lane_center_source = _crown_aligned_centers(
            shoot_for_support,
            expected_centers,
            dish_bbox,
        )
        dish_width = max(1, int(dish_bbox[2] - dish_bbox[0]))
        dish_height = max(1, int(dish_bbox[3] - dish_bbox[1]))
        root_support = np.logical_or(root_union > 0, shoot_for_support > 0).astype(np.uint8)
        root_boxes_yx = _grow_bounding_boxes(
            root_support,
            expected_centers,
            initial_box_halfsize_x=max(80, int(round(dish_width * 0.05))),
            initial_box_halfsize_y=max(120, int(round(dish_height * 0.08))),
            expansion_step=2,
            max_empty_expansions=max(60, int(getattr(config, "tracking_search_margin", 26) * 2)),
            stop_on_overlap=True,
        )
        shoot_masks = _find_shoot_regions(
            shoot_for_support,
            expected_centers,
            initial_box_halfsize_x=max(80, int(round(dish_width * 0.05))),
            initial_box_halfsize_y=max(120, int(round(dish_height * 0.12))),
        )
        lane_boxes = _nonoverlapping_lane_boxes(shape_hw, expected_centers, dish_bbox)
        frame_overlap_risk = _lane_overlap_risk(root_union, lane_boxes)

        frame_bboxes: dict[str, list[int]] = {}
        frame_shoot_bboxes: dict[str, list[int]] = {}
        frame_lengths_mm: dict[str, float] = {}
        frame_lengths_px: dict[str, float] = {}
        frame_root_area_px: dict[str, int] = {}
        frame_primary_lengths_mm: dict[str, float] = {}
        frame_lateral_lengths_mm: dict[str, float] = {}
        frame_shoot_area_px: dict[str, int] = {}

        boxes_xywh: list[BBox] = []
        for lane_index, (track_id, _center, root_box_yx, _shoot_owned) in enumerate(
            zip(track_ids, expected_centers, root_boxes_yx, shoot_masks, strict=False)
        ):
            dynamic_bbox_xywh = _yx_box_to_xywh(root_box_yx, shape_hw)
            bbox_xywh = lane_boxes[lane_index] if lane_index < len(lane_boxes) else dynamic_bbox_xywh
            boxes_xywh.append(bbox_xywh)
            x, y, w, h = bbox_xywh
            primary_crop = _crop_to_bbox(root_mask, bbox_xywh)
            lateral_crop = _crop_to_bbox(lateral_mask, bbox_xywh)
            owned_crop = np.logical_or(primary_crop > 0, lateral_crop > 0).astype(np.uint8)
            primary_length_px, primary_area_px, primary_perimeter_px = _measure_mask_length_px(primary_crop)
            lateral_length_px, lateral_area_px, lateral_perimeter_px = _measure_mask_length_px(lateral_crop)
            length_px, area_px, perimeter_px = _measure_mask_length_px(owned_crop)
            length_mm = float(length_px * pixel_size_mm)
            primary_length_mm = float(primary_length_px * pixel_size_mm)
            lateral_length_mm = float(lateral_length_px * pixel_size_mm)

            shoot_binary = np.zeros_like(shoot_mask, dtype=np.uint8)
            if w > 0 and h > 0:
                shoot_binary[y : y + h, x : x + w] = shoot_mask[y : y + h, x : x + w]
            ys, xs = np.where(shoot_binary > 0)
            if xs.size > 0 and ys.size > 0:
                sx0 = int(xs.min())
                sy0 = int(ys.min())
                sx1 = int(xs.max()) + 1
                sy1 = int(ys.max()) + 1
                shoot_bbox_xywh = (sx0, sy0, max(0, sx1 - sx0), max(0, sy1 - sy0))
            else:
                shoot_bbox_xywh = (0, 0, 0, 0)

            frame_bboxes[track_id] = [int(v) for v in bbox_xywh]
            frame_shoot_bboxes[track_id] = [int(v) for v in shoot_bbox_xywh]
            frame_lengths_mm[track_id] = float(round(length_mm, 4))
            frame_lengths_px[track_id] = float(round(length_px, 4))
            frame_root_area_px[track_id] = int(area_px)
            frame_primary_lengths_mm[track_id] = float(round(primary_length_mm, 4))
            frame_lateral_lengths_mm[track_id] = float(round(lateral_length_mm, 4))
            frame_shoot_area_px[track_id] = int(np.count_nonzero(shoot_binary))

            frame_boundary_crossing = bool(
                frame_overlap_risk[lane_index]
                if lane_index < len(frame_overlap_risk)
                else False
            )
            if frame_boundary_crossing and overlap_frames.get(track_id) is None:
                overlap_frames[track_id] = int(frame_index)
            overlap_risk = bool(overlap_frames.get(track_id) is not None)
            shoot_area_px = int(np.count_nonzero(shoot_binary))
            measurement_available = bool(area_px > 0)
            spatial_lane_valid = bool(measurement_available and not overlap_risk)
            qc_tier = (
                "unavailable"
                if not measurement_available
                else "overlap_unknown"
                if overlap_risk
                else "no_crossing_detected"
            )

            rows.append(
                {
                    "uid": item.uid,
                    "image_name": item.name,
                    "frame_index": int(frame_index),
                    "plant_id": track_id,
                    "lane_id": track_id,
                    "pixel_size_mm": float(round(pixel_size_mm, 8)),
                    "bbox_x": int(x),
                    "bbox_y": int(y),
                    "bbox_w": int(w),
                    "bbox_h": int(h),
                    "shoot_bbox_x": int(shoot_bbox_xywh[0]),
                    "shoot_bbox_y": int(shoot_bbox_xywh[1]),
                    "shoot_bbox_w": int(shoot_bbox_xywh[2]),
                    "shoot_bbox_h": int(shoot_bbox_xywh[3]),
                    "root_area_px": int(area_px),
                    "root_area_mm2": float(round(float(area_px) * (pixel_size_mm ** 2), 4)),
                    "root_perimeter_px": float(round(perimeter_px, 4)),
                    "root_perimeter_mm": float(round(perimeter_px * pixel_size_mm, 4)),
                    "root_length_px": float(round(length_px, 4)),
                    "root_length_mm": float(round(length_mm, 4)),
                    "primary_root_area_px": int(primary_area_px),
                    "primary_root_perimeter_px": float(round(primary_perimeter_px, 4)),
                    "primary_root_length_px": float(round(primary_length_px, 4)),
                    "primary_root_length_mm": float(round(primary_length_mm, 4)),
                    "lateral_root_area_px": int(lateral_area_px),
                    "lateral_root_perimeter_px": float(round(lateral_perimeter_px, 4)),
                    "lateral_root_length_px": float(round(lateral_length_px, 4)),
                    "lateral_root_length_mm": float(round(lateral_length_mm, 4)),
                    "total_root_area_px": int(area_px),
                    "total_root_length_px": float(round(length_px, 4)),
                    "total_root_length_mm": float(round(length_mm, 4)),
                    "shoot_area_px": int(shoot_area_px),
                    "shoot_area_mm2": float(round(float(shoot_area_px) * (pixel_size_mm ** 2), 4)),
                    "dynamic_bbox_x": int(dynamic_bbox_xywh[0]),
                    "dynamic_bbox_y": int(dynamic_bbox_xywh[1]),
                    "dynamic_bbox_w": int(dynamic_bbox_xywh[2]),
                    "dynamic_bbox_h": int(dynamic_bbox_xywh[3]),
                    "overlap_frame": overlap_frames.get(track_id),
                    "ownership_backend": "spatial_lane_fallback",
                    "measurement_scope": "crown_aligned_spatial_lane",
                    "ownership_claim": False,
                    "lane_center_source": lane_center_source,
                    "lane_center_x": int(expected_centers[lane_index][0]),
                    "frame_boundary_crossing_detected": bool(frame_boundary_crossing),
                    "neighbor_overlap_risk": bool(overlap_risk),
                    "bbox_measurement_available": bool(measurement_available),
                    "spatial_lane_measurement_valid": bool(spatial_lane_valid),
                    "ownership_measurement_valid": False,
                    "spatial_lane_qc_tier": qc_tier,
                    "lateral_class_id": int(lateral_class_id) if lateral_class_id is not None else "",
                    "shoot_mask_plausible": bool(shoot_mask_plausible),
                }
            )

        for left_index, left_track_id in enumerate(track_ids):
            left_bbox = boxes_xywh[left_index] if left_index < len(boxes_xywh) else (0, 0, 0, 0)
            for right_index, right_track_id in enumerate(track_ids[left_index + 1 :], start=left_index + 1):
                right_bbox = boxes_xywh[right_index] if right_index < len(boxes_xywh) else (0, 0, 0, 0)
                if _bbox_intersects(left_bbox, right_bbox):
                    if overlap_frames[left_track_id] is None:
                        overlap_frames[left_track_id] = int(frame_index)
                    if overlap_frames[right_track_id] is None:
                        overlap_frames[right_track_id] = int(frame_index)

        per_uid[item.uid] = {
            "enabled": True,
            "frame_index": int(frame_index),
            "plant_count": int(plant_count),
            "seed_frame_index": 0,
            "bboxes_xywh": frame_bboxes,
            "shoot_bboxes_xywh": frame_shoot_bboxes,
            "lengths_px": frame_lengths_px,
            "lengths_mm": frame_lengths_mm,
            "primary_lengths_mm": frame_primary_lengths_mm,
            "lateral_lengths_mm": frame_lateral_lengths_mm,
            "root_area_px": frame_root_area_px,
            "shoot_area_px": frame_shoot_area_px,
            "pixel_size_mm": float(pixel_size_mm),
            "overlap_frames": {track_id: overlap_frames.get(track_id) for track_id in track_ids},
            "ownership_backend": "spatial_lane_fallback",
            "measurement_scope": "crown_aligned_spatial_lane",
            "ownership_claim": False,
            "lane_center_source": lane_center_source,
            "shoot_mask_plausible": bool(shoot_mask_plausible),
        }
        if progress_callback is not None and not bool(progress_callback(frame_index + 1, len(timeline_items))):
            break

    summary = {
        "backend": "pyphenotyper",
        "bbox_tracking_enabled": True,
        "seed_frame_index": 0,
        "pixel_size_mm": float(config.pixel_size_mm),
        "pixel_size_mm_by_uid": {uid: float(value) for uid, value in dataset_pixel_sizes.items()},
        "plant_ids": track_ids,
        "lane_ids": track_ids,
        "timeline": [{"uid": item.uid, "name": item.name} for item in timeline_items],
        "measurements": rows,
        "overlap_frames": {k: overlap_frames.get(k) for k in track_ids},
        "ownership_backend": "spatial_lane_fallback",
        "measurement_scope": "crown_aligned_spatial_lane",
        "ownership_claim": False,
        "lateral_class_id": int(lateral_class_id) if lateral_class_id is not None else None,
    }
    return {"summary": summary, "per_uid": per_uid}
