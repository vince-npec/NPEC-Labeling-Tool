from __future__ import annotations

import cv2
import numpy as np

from .models import LabelClass, hex_to_rgb


def ensure_layer_map(
    layer_store: dict[str, dict[int, np.ndarray]],
    image_uid: str,
    classes: list[LabelClass],
    shape_hw: tuple[int, int],
) -> dict[int, np.ndarray]:
    h, w = shape_hw
    class_ids = [cls.class_id for cls in classes]
    per_image = layer_store.setdefault(image_uid, {})

    for class_id in class_ids:
        if class_id not in per_image or per_image[class_id].shape != (h, w):
            per_image[class_id] = np.zeros((h, w), dtype=np.uint8)

    stale_ids = [class_id for class_id in per_image if class_id not in class_ids]
    for class_id in stale_ids:
        del per_image[class_id]

    return per_image


def index_mask_from_layers(layers: dict[int, np.ndarray], classes: list[LabelClass], shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    index_mask = np.zeros((h, w), dtype=np.uint8)
    for cls in classes:
        class_id = cls.class_id
        layer = layers.get(class_id)
        if layer is None:
            continue
        index_mask[layer > 0] = np.uint8(min(class_id, 255))
    return index_mask


def blend_layers_on_image(
    image_rgb: np.ndarray,
    layers: dict[int, np.ndarray],
    classes: list[LabelClass],
    alpha: float = 0.45,
    active_class_id: int | None = None,
    edge_thickness: int = 0,
    edge_alpha: float = 0.95,
) -> np.ndarray:
    output = image_rgb.astype(np.float32).copy()
    edge_thickness = max(0, int(edge_thickness))
    edge_alpha = max(0.0, min(1.0, float(edge_alpha)))
    kernel = np.ones((3, 3), dtype=np.uint8) if edge_thickness > 0 else None
    for cls in classes:
        class_id = cls.class_id
        layer = layers.get(class_id)
        if layer is None:
            continue
        pixels = layer > 0
        if not np.any(pixels):
            continue
        color = np.array(hex_to_rgb(cls.color_hex), dtype=np.float32)
        local_alpha = alpha
        if active_class_id is not None and class_id != active_class_id:
            local_alpha = max(0.12, alpha * 0.5)
        output[pixels] = output[pixels] * (1.0 - local_alpha) + color * local_alpha
        if edge_thickness > 0 and kernel is not None:
            mask_u8 = pixels.astype(np.uint8)
            eroded = cv2.erode(mask_u8, kernel, iterations=edge_thickness)
            edge = np.logical_and(pixels, eroded == 0)
            if np.any(edge):
                output[edge] = output[edge] * (1.0 - edge_alpha) + color * edge_alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def colorize_index_mask(index_mask: np.ndarray, classes: list[LabelClass]) -> np.ndarray:
    h, w = index_mask.shape
    colorized = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in classes:
        colorized[index_mask == cls.class_id] = hex_to_rgb(cls.color_hex)
    return colorized


def blend_index_mask(
    image_rgb: np.ndarray,
    index_mask: np.ndarray,
    classes: list[LabelClass],
    alpha: float = 0.45,
    edge_thickness: int = 0,
    edge_alpha: float = 0.95,
) -> np.ndarray:
    colorized = colorize_index_mask(index_mask, classes)
    output = image_rgb.astype(np.float32).copy()
    labeled = index_mask > 0
    output[labeled] = output[labeled] * (1.0 - alpha) + colorized[labeled].astype(np.float32) * alpha
    edge_thickness = max(0, int(edge_thickness))
    edge_alpha = max(0.0, min(1.0, float(edge_alpha)))
    if edge_thickness > 0 and np.any(labeled):
        kernel = np.ones((3, 3), dtype=np.uint8)
        for cls in classes:
            pixels = index_mask == cls.class_id
            if not np.any(pixels):
                continue
            color = np.asarray(hex_to_rgb(cls.color_hex), dtype=np.float32)
            mask_u8 = pixels.astype(np.uint8)
            eroded = cv2.erode(mask_u8, kernel, iterations=edge_thickness)
            edge = np.logical_and(pixels, eroded == 0)
            if np.any(edge):
                output[edge] = output[edge] * (1.0 - edge_alpha) + color * edge_alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def disagreement_overlay(image_rgb: np.ndarray, disagreement: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    output = image_rgb.astype(np.float32).copy()
    red = np.array([255.0, 32.0, 32.0], dtype=np.float32)
    output[disagreement] = output[disagreement] * (1.0 - alpha) + red * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def compute_metrics(gt: np.ndarray, pred: np.ndarray, class_ids: list[int]) -> dict[str, float]:
    if gt.shape != pred.shape:
        return {}

    metrics: dict[str, float] = {}
    metrics["pixel_accuracy"] = float(np.mean(gt == pred))

    ious: list[float] = []
    dices: list[float] = []
    for class_id in class_ids:
        gt_c = gt == class_id
        pred_c = pred == class_id
        inter = float(np.logical_and(gt_c, pred_c).sum())
        union = float(np.logical_or(gt_c, pred_c).sum())
        denom = float(gt_c.sum() + pred_c.sum())

        if union > 0:
            iou = inter / union
            ious.append(iou)
            metrics[f"class_{class_id}_iou"] = iou
        if denom > 0:
            dice = (2.0 * inter) / denom
            dices.append(dice)
            metrics[f"class_{class_id}_dice"] = dice

    if ious:
        metrics["mean_iou"] = float(np.mean(ious))
    if dices:
        metrics["mean_dice"] = float(np.mean(dices))
    return metrics


def paint_disk(mask: np.ndarray, x: int, y: int, radius: int, value: bool = True) -> None:
    h, w = mask.shape
    if h == 0 or w == 0:
        return

    r = max(0, int(radius))
    if r == 0:
        if 0 <= x < w and 0 <= y < h:
            mask[y, x] = bool(value)
        return

    x0 = max(0, x - r)
    x1 = min(w - 1, x + r)
    y0 = max(0, y - r)
    y1 = min(h - 1, y + r)
    if x0 > x1 or y0 > y1:
        return

    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    circle = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    if value:
        mask[y0 : y1 + 1, x0 : x1 + 1][circle] = True
    else:
        mask[y0 : y1 + 1, x0 : x1 + 1][circle] = False


def rasterize_stroke(
    shape_hw: tuple[int, int],
    start_xy: tuple[int, int],
    end_xy: tuple[int, int],
    radius: int,
) -> dict[str, object]:
    h, w = shape_hw
    if h <= 0 or w <= 0:
        return {"bbox": (0, 0, -1, -1), "mask": np.zeros((0, 0), dtype=bool)}

    sx, sy = int(start_xy[0]), int(start_xy[1])
    ex, ey = int(end_xy[0]), int(end_xy[1])
    r = max(0, int(radius))

    x0 = max(0, min(sx, ex) - r)
    y0 = max(0, min(sy, ey) - r)
    x1 = min(w - 1, max(sx, ex) + r)
    y1 = min(h - 1, max(sy, ey) + r)

    if x1 < x0 or y1 < y0:
        return {"bbox": (0, 0, -1, -1), "mask": np.zeros((0, 0), dtype=bool)}

    stroke = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)

    dx = abs(ex - sx)
    dy = abs(ey - sy)
    steps = max(dx, dy, 1)
    xs = np.linspace(sx, ex, steps + 1)
    ys = np.linspace(sy, ey, steps + 1)

    for x_float, y_float in zip(xs, ys):
        lx = int(round(x_float)) - x0
        ly = int(round(y_float)) - y0
        paint_disk(stroke, lx, ly, r, value=True)

    return {"bbox": (x0, y0, x1, y1), "mask": stroke}


def build_wand_stroke_from_prediction(
    index_mask: np.ndarray,
    target_class_id: int,
    seed_xy: tuple[int, int],
    *,
    search_radius: int = 8,
    min_component_area: int = 1,
) -> dict[str, object] | None:
    arr = np.asarray(index_mask, dtype=np.uint8)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.size == 0:
        return None

    h, w = arr.shape[:2]
    sx = int(seed_xy[0])
    sy = int(seed_xy[1])
    if sx < 0 or sy < 0 or sx >= w or sy >= h:
        return None

    target_class_id = int(target_class_id)
    if target_class_id <= 0:
        return None

    target_mask = (arr == np.uint8(target_class_id))
    if not np.any(target_mask):
        return None

    if not bool(target_mask[sy, sx]):
        best_point: tuple[int, int] | None = None
        best_dist2: int | None = None
        max_radius = max(0, int(search_radius))
        for radius in range(1, max_radius + 1):
            x0 = max(0, sx - radius)
            x1 = min(w - 1, sx + radius)
            y0 = max(0, sy - radius)
            y1 = min(h - 1, sy + radius)
            local = target_mask[y0 : y1 + 1, x0 : x1 + 1]
            if not np.any(local):
                continue
            yy, xx = np.nonzero(local)
            for ly, lx in zip(yy.tolist(), xx.tolist()):
                px = int(x0 + lx)
                py = int(y0 + ly)
                dist2 = int((px - sx) * (px - sx) + (py - sy) * (py - sy))
                if best_dist2 is None or dist2 < best_dist2:
                    best_point = (px, py)
                    best_dist2 = dist2
            if best_point is not None:
                sx, sy = best_point
                break
        if best_point is None:
            return None

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(target_mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None
    component_id = int(labels[sy, sx])
    if component_id <= 0:
        return None

    area_px = int(stats[component_id, cv2.CC_STAT_AREA])
    if area_px < max(1, int(min_component_area)):
        return None

    x = int(stats[component_id, cv2.CC_STAT_LEFT])
    y = int(stats[component_id, cv2.CC_STAT_TOP])
    width = int(stats[component_id, cv2.CC_STAT_WIDTH])
    height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
    if width <= 0 or height <= 0:
        return None

    local_mask = labels[y : y + height, x : x + width] == component_id
    return {
        "bbox": (x, y, x + width - 1, y + height - 1),
        "mask": np.asarray(local_mask, dtype=bool),
        "point": (int(sx), int(sy)),
        "wand_class_id": int(target_class_id),
        "area_px": int(area_px),
    }


def translate_mask(mask: np.ndarray, dx: int, dy: int, *, fill_value: int = 0) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        return np.asarray(arr)
    shift_x = int(dx)
    shift_y = int(dy)
    if shift_x == 0 and shift_y == 0:
        return np.asarray(arr).copy()

    h, w = arr.shape[:2]
    out = np.full((h, w), fill_value, dtype=arr.dtype)

    src_x0 = max(0, -shift_x)
    src_x1 = min(w, w - shift_x) if shift_x >= 0 else w
    dst_x0 = max(0, shift_x)
    dst_x1 = min(w, w + shift_x) if shift_x <= 0 else w

    src_y0 = max(0, -shift_y)
    src_y1 = min(h, h - shift_y) if shift_y >= 0 else h
    dst_y0 = max(0, shift_y)
    dst_y1 = min(h, h + shift_y) if shift_y <= 0 else h

    if src_x1 <= src_x0 or src_y1 <= src_y0 or dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return out

    out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def translate_selected_layers(
    layers: dict[int, np.ndarray],
    class_ids: list[int] | tuple[int, ...],
    dx: int,
    dy: int,
) -> int:
    moved_pixels = 0
    target_ids = [int(class_id) for class_id in class_ids]
    if not target_ids:
        return moved_pixels
    for class_id in target_ids:
        layer = layers.get(int(class_id))
        if layer is None:
            continue
        shifted = translate_mask(layer, int(dx), int(dy), fill_value=0)
        moved_pixels += int(np.count_nonzero(shifted))
        layer[:] = np.asarray(shifted, dtype=layer.dtype)
    return int(moved_pixels)


def rotate_image_90(image: np.ndarray, turns: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim < 2 or arr.size == 0:
        return np.asarray(arr).copy()
    k = int(turns) % 4
    if k == 0:
        return np.asarray(arr).copy()
    return np.rot90(arr, k=k, axes=(0, 1)).copy()


def rotate_point_xy(point_xy: tuple[int, int], shape_hw: tuple[int, int], turns: int) -> tuple[int, int]:
    x = int(point_xy[0])
    y = int(point_xy[1])
    h, w = int(shape_hw[0]), int(shape_hw[1])
    k = int(turns) % 4
    for _ in range(k):
        x, y = y, (w - 1 - x)
        h, w = w, h
    return (int(x), int(y))


def rotate_rect_xywh(rect_xywh: tuple[int, int, int, int], shape_hw: tuple[int, int], turns: int) -> tuple[int, int, int, int]:
    x, y, width, height = (int(rect_xywh[0]), int(rect_xywh[1]), int(rect_xywh[2]), int(rect_xywh[3]))
    if width <= 0 or height <= 0:
        return (0, 0, 0, 0)
    corners = (
        (x, y),
        (x + width - 1, y),
        (x, y + height - 1),
        (x + width - 1, y + height - 1),
    )
    rotated = [rotate_point_xy(corner, shape_hw, turns) for corner in corners]
    xs = [point[0] for point in rotated]
    ys = [point[1] for point in rotated]
    x0 = int(min(xs))
    y0 = int(min(ys))
    x1 = int(max(xs))
    y1 = int(max(ys))
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def apply_tool_on_layers(
    layers: dict[int, np.ndarray],
    stroke_mask: object,
    tool: str,
    target_class_id: int,
    restrict_mask: np.ndarray | None = None,
    exclusive_brush: bool = True,
) -> None:
    normalized_tool = tool.lower()
    if normalized_tool not in {"brush", "eraser", "razor"}:
        return

    if isinstance(stroke_mask, dict) and "bbox" in stroke_mask and "mask" in stroke_mask:
        bbox = stroke_mask["bbox"]
        local_mask = np.asarray(stroke_mask["mask"], dtype=bool)
        if local_mask.size == 0:
            return
        x0, y0, x1, y1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        if x1 < x0 or y1 < y0:
            return

        if restrict_mask is not None:
            restrict_view = restrict_mask[y0 : y1 + 1, x0 : x1 + 1]
            if restrict_view.shape != local_mask.shape:
                h = min(restrict_view.shape[0], local_mask.shape[0])
                w = min(restrict_view.shape[1], local_mask.shape[1])
                if h <= 0 or w <= 0:
                    return
                local_mask = local_mask[:h, :w]
                restrict_view = restrict_view[:h, :w]
            local_mask = np.logical_and(local_mask, restrict_view)
            if not np.any(local_mask):
                return

        if normalized_tool == "brush":
            if exclusive_brush:
                for class_id in list(layers.keys()):
                    patch = layers[class_id][y0 : y0 + local_mask.shape[0], x0 : x0 + local_mask.shape[1]]
                    patch[local_mask] = 0
            target = layers.get(target_class_id)
            if target is not None:
                target_patch = target[y0 : y0 + local_mask.shape[0], x0 : x0 + local_mask.shape[1]]
                target_patch[local_mask] = 1
            return

        if normalized_tool == "eraser":
            target = layers.get(target_class_id)
            if target is None:
                return
            patch = target[y0 : y0 + local_mask.shape[0], x0 : x0 + local_mask.shape[1]]
            patch[local_mask] = 0
            return

        for class_id in list(layers.keys()):
            patch = layers[class_id][y0 : y0 + local_mask.shape[0], x0 : x0 + local_mask.shape[1]]
            patch[local_mask] = 0
        return

    if not isinstance(stroke_mask, np.ndarray) or stroke_mask.size == 0:
        return

    edit_mask = stroke_mask.astype(bool)
    if restrict_mask is not None:
        edit_mask = np.logical_and(edit_mask, restrict_mask)
    if not np.any(edit_mask):
        return

    if normalized_tool == "brush":
        if exclusive_brush:
            for class_id in list(layers.keys()):
                layers[class_id][edit_mask] = 0
        target = layers.get(target_class_id)
        if target is not None:
            target[edit_mask] = 1
    elif normalized_tool == "eraser":
        target = layers.get(target_class_id)
        if target is not None:
            target[edit_mask] = 0
    else:
        for class_id in list(layers.keys()):
            layers[class_id][edit_mask] = 0
