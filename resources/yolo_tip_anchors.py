from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import shutil

import cv2
import numpy as np
from PIL import Image

from .models import DatasetImageItem
from .project_io import load_project


BBox = tuple[int, int, int, int]
Point = tuple[int, int]

DETECT_CLASS_NAMES = ["crown", "primary_tip"]
POSE_CLASS_NAMES = ["seedling"]
CROWN_NAME_TOKENS = ("shoot", "seed", "rosette", "leaf")
ROOT_NAME_TOKENS = ("root", "primary")
LATERAL_NAME_TOKENS = ("lateral",)
SEED_NAME_TOKENS = ("seed",)
YOLO_CROWN_CLASS_NAMES = {"crown", "shoot_crown", "seedling", "plant"}
YOLO_TIP_CLASS_NAMES = {"primary_tip", "root_tip", "tip"}


@dataclass(slots=True)
class AnchorTarget:
    plant_id: str
    crown_bbox: BBox
    seedling_bbox: BBox
    primary_tip_xy: Point
    crown_center_xy: Point
    lane_bounds: tuple[int, int]


def _normalize_name(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _mask_bbox(mask: np.ndarray) -> BBox:
    src = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
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


def _clip_bbox(bbox: BBox, shape_hw: tuple[int, int]) -> BBox:
    h, w = (int(shape_hw[0]), int(shape_hw[1]))
    x, y, bw, bh = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    if h <= 0 or w <= 0:
        return (0, 0, 0, 0)
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    bw = max(0, min(w - x, bw))
    bh = max(0, min(h - y, bh))
    return (x, y, bw, bh)


def _union_bbox(a: BBox, b: BBox) -> BBox:
    if int(a[2]) <= 0 or int(a[3]) <= 0:
        return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    if int(b[2]) <= 0 or int(b[3]) <= 0:
        return (int(a[0]), int(a[1]), int(a[2]), int(a[3]))
    x0 = min(int(a[0]), int(b[0]))
    y0 = min(int(a[1]), int(b[1]))
    x1 = max(int(a[0] + a[2]), int(b[0] + b[2]))
    y1 = max(int(a[1] + a[3]), int(b[1] + b[3]))
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def _bbox_center(bbox: BBox) -> Point:
    x, y, bw, bh = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    return (int(round(x + bw * 0.5)), int(round(y + bh * 0.5)))


def _connected_components(mask: np.ndarray, min_area: int = 4) -> list[dict[str, object]]:
    src = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if src.ndim != 2 or src.size == 0 or int(np.count_nonzero(src)) <= 0:
        return []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(src, connectivity=8)
    out: list[dict[str, object]] = []
    for component_id in range(1, int(num_labels)):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < int(max(1, min_area)):
            continue
        component_mask = (labels == component_id).astype(np.uint8)
        bbox = _mask_bbox(component_mask)
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        out.append(
            {
                "mask": component_mask,
                "bbox": bbox,
                "area": area,
                "centroid": (float(centroids[component_id][0]), float(centroids[component_id][1])),
            }
        )
    out.sort(key=lambda comp: (float(comp["centroid"][0]), float(comp["centroid"][1])))
    return out


def _lane_bounds(components: list[dict[str, object]], width: int) -> list[tuple[int, int]]:
    if not components:
        return []
    centers = [float(comp["centroid"][0]) for comp in components]
    out: list[tuple[int, int]] = []
    for idx, _ in enumerate(components):
        left = 0 if idx == 0 else int(round((centers[idx - 1] + centers[idx]) * 0.5))
        right = int(width) if idx == len(components) - 1 else int(round((centers[idx] + centers[idx + 1]) * 0.5))
        out.append((max(0, left), max(0, right)))
    return out


def _bottommost_point(mask: np.ndarray) -> Point | None:
    ys, xs = np.where(np.asarray(mask, dtype=np.uint8) > 0)
    if xs.size <= 0 or ys.size <= 0:
        return None
    y_max = int(np.max(ys))
    row_xs = xs[ys == y_max]
    if row_xs.size <= 0:
        return None
    x_med = int(np.median(row_xs))
    return (x_med, y_max)


def _union_masks(masks: list[np.ndarray], shape_hw: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape_hw, dtype=np.uint8)
    for mask in masks:
        src = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
        if src.shape != shape_hw:
            continue
        out = np.maximum(out, src)
    return out


def _class_mask_bundle(
    annotation_masks: dict[int, np.ndarray],
    class_name_by_id: dict[int, str],
    shape_hw: tuple[int, int],
) -> dict[str, np.ndarray]:
    shoot_masks: list[np.ndarray] = []
    seed_masks: list[np.ndarray] = []
    root_masks: list[np.ndarray] = []
    lateral_masks: list[np.ndarray] = []
    for class_id, mask in annotation_masks.items():
        normalized = _normalize_name(class_name_by_id.get(int(class_id), ""))
        if any(token in normalized for token in CROWN_NAME_TOKENS):
            shoot_masks.append(mask)
        if any(token in normalized for token in SEED_NAME_TOKENS):
            seed_masks.append(mask)
        if any(token in normalized for token in LATERAL_NAME_TOKENS):
            lateral_masks.append(mask)
        elif any(token in normalized for token in ROOT_NAME_TOKENS):
            root_masks.append(mask)
    shoot = _union_masks(shoot_masks, shape_hw)
    seed = _union_masks(seed_masks, shape_hw)
    root = _union_masks(root_masks, shape_hw)
    lateral = _union_masks(lateral_masks, shape_hw)
    root_total = np.maximum(root, lateral)
    crown = np.maximum(shoot, seed)
    return {
        "shoot": shoot,
        "seed": seed,
        "root": root,
        "lateral": lateral,
        "root_total": root_total,
        "crown": crown,
    }


def derive_seedling_anchor_targets(
    image_shape_hw: tuple[int, int],
    annotation_masks: dict[int, np.ndarray],
    class_name_by_id: dict[int, str],
    *,
    expected_track_count: int = 0,
    top_band_ratio: float = 0.40,
) -> list[AnchorTarget]:
    shape_hw = (int(image_shape_hw[0]), int(image_shape_hw[1]))
    bundles = _class_mask_bundle(annotation_masks, class_name_by_id, shape_hw)
    crown_mask = bundles["crown"]
    root_mask = bundles["root"]
    root_total_mask = bundles["root_total"]
    if int(np.count_nonzero(crown_mask)) <= 0:
        fallback = root_total_mask.copy()
        cutoff = max(1, int(round(shape_hw[0] * float(top_band_ratio))))
        fallback[cutoff:, :] = 0
        crown_components = _connected_components(fallback, min_area=4)
    else:
        crown_components = _connected_components(crown_mask, min_area=4)
    if expected_track_count > 0 and len(crown_components) > expected_track_count:
        crown_components = crown_components[:expected_track_count]
    if not crown_components:
        return []
    lanes = _lane_bounds(crown_components, shape_hw[1])
    targets: list[AnchorTarget] = []
    for idx, component in enumerate(crown_components, start=1):
        lane_left, lane_right = lanes[idx - 1]
        lane_mask = np.zeros(shape_hw, dtype=np.uint8)
        lane_mask[:, lane_left:lane_right] = 1
        crown_local = np.where(lane_mask > 0, crown_mask, 0).astype(np.uint8)
        if int(np.count_nonzero(crown_local)) <= 0:
            crown_local = np.asarray(component["mask"], dtype=np.uint8)
        root_local = np.where(lane_mask > 0, root_mask, 0).astype(np.uint8)
        if int(np.count_nonzero(root_local)) <= 0:
            root_local = np.where(lane_mask > 0, root_total_mask, 0).astype(np.uint8)
        seedling_local = np.maximum(crown_local, np.where(lane_mask > 0, root_total_mask, 0).astype(np.uint8))
        crown_bbox = _mask_bbox(crown_local)
        seedling_bbox = _mask_bbox(seedling_local)
        if seedling_bbox[2] <= 0 or seedling_bbox[3] <= 0:
            seedling_bbox = crown_bbox
        if crown_bbox[2] <= 0 or crown_bbox[3] <= 0:
            crown_bbox = seedling_bbox
        if crown_bbox[2] <= 0 or crown_bbox[3] <= 0:
            continue
        tip_point = _bottommost_point(root_local)
        if tip_point is None:
            tip_point = _bottommost_point(seedling_local)
        if tip_point is None:
            tip_point = _bbox_center(seedling_bbox)
        targets.append(
            AnchorTarget(
                plant_id=f"plant_{idx:02d}",
                crown_bbox=_clip_bbox(crown_bbox, shape_hw),
                seedling_bbox=_clip_bbox(seedling_bbox, shape_hw),
                primary_tip_xy=(int(tip_point[0]), int(tip_point[1])),
                crown_center_xy=_bbox_center(crown_bbox),
                lane_bounds=(int(lane_left), int(lane_right)),
            )
        )
    return targets


def _normalized_bbox_line(class_id: int, bbox: BBox, shape_hw: tuple[int, int]) -> str | None:
    h, w = (max(1, int(shape_hw[0])), max(1, int(shape_hw[1])))
    x, y, bw, bh = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    if bw <= 0 or bh <= 0:
        return None
    xc = (x + bw * 0.5) / float(w)
    yc = (y + bh * 0.5) / float(h)
    wn = bw / float(w)
    hn = bh / float(h)
    return f"{int(class_id)} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def _tip_box(point_xy: Point, shape_hw: tuple[int, int], size_px: int) -> BBox:
    half = max(2, int(round(size_px * 0.5)))
    x = int(point_xy[0]) - half
    y = int(point_xy[1]) - half
    bbox = (x, y, half * 2, half * 2)
    return _clip_bbox(bbox, shape_hw)


def _pose_line(class_id: int, bbox: BBox, shape_hw: tuple[int, int], keypoints_xy: list[Point]) -> str | None:
    base = _normalized_bbox_line(class_id, bbox, shape_hw)
    if base is None:
        return None
    h, w = (max(1, int(shape_hw[0])), max(1, int(shape_hw[1])))
    kp_parts: list[str] = []
    for x, y in keypoints_xy:
        kp_parts.extend([f"{float(x) / float(w):.6f}", f"{float(y) / float(h):.6f}", "2"])
    return f"{base} {' '.join(kp_parts)}"


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_yaml(path: Path, lines: list[str]) -> None:
    _write_text(path, lines)


def export_yolo26_anchor_dataset(
    project_path: Path,
    output_dir: Path,
    *,
    val_stride: int = 5,
    tip_box_size_px: int = 24,
    expected_track_count: int = 0,
) -> dict[str, object]:
    loaded = load_project(Path(project_path))
    items: list[DatasetImageItem] = list(loaded.get("dataset_items", []))
    annotations: dict[str, dict[int, np.ndarray]] = dict(loaded.get("annotations", {}))
    classes = list(loaded.get("classes", []))
    class_name_by_id = {int(cls.class_id): str(cls.name) for cls in classes}
    detect_root = Path(output_dir) / "detect"
    pose_root = Path(output_dir) / "pose"
    if detect_root.exists():
        shutil.rmtree(detect_root)
    if pose_root.exists():
        shutil.rmtree(pose_root)
    splits = {"train": [], "val": []}
    sorted_items = sorted(items, key=lambda item: (str(item.name).lower(), str(item.uid)))
    for idx, item in enumerate(sorted_items):
        split = "val" if int(max(2, val_stride)) > 0 and (idx % int(max(2, val_stride)) == 0) else "train"
        splits[split].append(item)

    manifest_images: list[dict[str, object]] = []
    detect_counts = {"train": 0, "val": 0}
    pose_counts = {"train": 0, "val": 0}

    for split, split_items in splits.items():
        for item in split_items:
            image = np.asarray(item.image, dtype=np.uint8)
            shape_hw = image.shape[:2]
            targets = derive_seedling_anchor_targets(
                shape_hw,
                annotations.get(item.uid, {}),
                class_name_by_id,
                expected_track_count=int(max(0, expected_track_count)),
            )
            detect_lines: list[str] = []
            pose_lines: list[str] = []
            for target in targets:
                crown_line = _normalized_bbox_line(0, target.crown_bbox, shape_hw)
                if crown_line is not None:
                    detect_lines.append(crown_line)
                    detect_counts[split] += 1
                tip_line = _normalized_bbox_line(1, _tip_box(target.primary_tip_xy, shape_hw, tip_box_size_px), shape_hw)
                if tip_line is not None:
                    detect_lines.append(tip_line)
                    detect_counts[split] += 1
                pose_line = _pose_line(0, target.seedling_bbox, shape_hw, [target.crown_center_xy, target.primary_tip_xy])
                if pose_line is not None:
                    pose_lines.append(pose_line)
                    pose_counts[split] += 1

            detect_image_path = detect_root / "images" / split / item.name
            pose_image_path = pose_root / "images" / split / item.name
            detect_label_path = detect_root / "labels" / split / f"{Path(item.name).stem}.txt"
            pose_label_path = pose_root / "labels" / split / f"{Path(item.name).stem}.txt"
            detect_image_path.parent.mkdir(parents=True, exist_ok=True)
            pose_image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(detect_image_path)
            Image.fromarray(image).save(pose_image_path)
            _write_text(detect_label_path, detect_lines)
            _write_text(pose_label_path, pose_lines)
            manifest_images.append(
                {
                    "uid": item.uid,
                    "name": item.name,
                    "split": split,
                    "targets": [
                        {
                            "plant_id": target.plant_id,
                            "crown_bbox": list(target.crown_bbox),
                            "seedling_bbox": list(target.seedling_bbox),
                            "primary_tip_xy": list(target.primary_tip_xy),
                            "crown_center_xy": list(target.crown_center_xy),
                            "lane_bounds": list(target.lane_bounds),
                        }
                        for target in targets
                    ],
                }
            )

    _write_yaml(
        detect_root / "dataset.yaml",
        [
            f"path: {detect_root}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: crown",
            "  1: primary_tip",
        ],
    )
    _write_yaml(
        pose_root / "dataset.yaml",
        [
            f"path: {pose_root}",
            "train: images/train",
            "val: images/val",
            "kpt_shape: [2, 3]",
            "flip_idx: [0, 1]",
            "names:",
            "  0: seedling",
        ],
    )
    summary = {
        "project_path": str(project_path),
        "output_dir": str(output_dir),
        "image_count": int(len(sorted_items)),
        "train_count": int(len(splits["train"])),
        "val_count": int(len(splits["val"])),
        "detect_label_count": {key: int(value) for key, value in detect_counts.items()},
        "pose_label_count": {key: int(value) for key, value in pose_counts.items()},
        "tip_box_size_px": int(max(4, tip_box_size_px)),
        "expected_track_count": int(max(0, expected_track_count)),
        "images": manifest_images,
    }
    manifest_path = Path(output_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _frame_lookup(items: list[DatasetImageItem]) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, item in enumerate(items):
        keys = {
            str(item.uid).strip(),
            str(item.name).strip(),
            Path(str(item.name)).stem,
            str(item.path).strip(),
            Path(str(item.path)).name,
            Path(str(item.path)).stem,
        }
        for key in keys:
            normalized = str(key).strip()
            if normalized:
                out[normalized] = idx
    return out


def _parse_bbox_xyxy(raw: object) -> BBox | None:
    if isinstance(raw, dict):
        if all(key in raw for key in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = (float(raw["x1"]), float(raw["y1"]), float(raw["x2"]), float(raw["y2"]))
            return (int(round(x1)), int(round(y1)), max(0, int(round(x2 - x1))), max(0, int(round(y2 - y1))))
        if all(key in raw for key in ("x", "y", "w", "h")):
            return (int(round(float(raw["x"]))), int(round(float(raw["y"]))), max(0, int(round(float(raw["w"])))), max(0, int(round(float(raw["h"])))))
        if "xyxy" in raw:
            return _parse_bbox_xyxy(raw.get("xyxy"))
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        x1, y1, x2, y2 = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        if x2 >= x1 and y2 >= y1:
            return (int(round(x1)), int(round(y1)), max(0, int(round(x2 - x1))), max(0, int(round(y2 - y1))))
        return (int(round(x1)), int(round(y1)), max(0, int(round(x2))), max(0, int(round(y2))))
    return None


def _extract_keypoints(raw: object) -> list[Point]:
    out: list[Point] = []
    if isinstance(raw, dict):
        if "primary_tip" in raw:
            tip = raw.get("primary_tip")
            if isinstance(tip, (list, tuple)) and len(tip) >= 2:
                out.append((int(round(float(tip[0]))), int(round(float(tip[1])))))
        if "crown_center" in raw:
            crown = raw.get("crown_center")
            if isinstance(crown, (list, tuple)) and len(crown) >= 2:
                out.insert(0, (int(round(float(crown[0]))), int(round(float(crown[1])))))
        if "xy" in raw:
            return _extract_keypoints(raw.get("xy"))
    elif isinstance(raw, (list, tuple)):
        if raw and isinstance(raw[0], (list, tuple)):
            for pt in raw:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    out.append((int(round(float(pt[0]))), int(round(float(pt[1])))))
        elif len(raw) >= 2:
            out.append((int(round(float(raw[0]))), int(round(float(raw[1])))))
    return out


def _best_track_ids_from_boxes(frame_boxes: list[list[dict[str, object]]]) -> tuple[list[str], dict[int, dict[str, BBox]]]:
    explicit_ids = {
        str(det.get("track_id")).strip()
        for detections in frame_boxes
        for det in detections
        if str(det.get("track_id", "")).strip()
    }
    if explicit_ids:
        track_ids = sorted(explicit_ids)
        per_frame: dict[int, dict[str, BBox]] = {}
        for frame_idx, detections in enumerate(frame_boxes):
            for det in detections:
                track_id = str(det.get("track_id", "")).strip()
                bbox = det.get("bbox")
                if track_id and isinstance(bbox, tuple):
                    per_frame.setdefault(frame_idx, {})[track_id] = bbox
        return track_ids, per_frame

    reference_idx = max(range(len(frame_boxes)), key=lambda idx: len(frame_boxes[idx]), default=-1)
    if reference_idx < 0 or not frame_boxes[reference_idx]:
        return [], {}
    ref_boxes = sorted(frame_boxes[reference_idx], key=lambda det: _bbox_center(det["bbox"])[0])  # type: ignore[index]
    track_ids = [f"plant_{idx + 1:02d}" for idx in range(len(ref_boxes))]
    per_frame: dict[int, dict[str, BBox]] = {reference_idx: {track_id: det["bbox"] for track_id, det in zip(track_ids, ref_boxes)}}
    previous_boxes = dict(per_frame[reference_idx])
    for frame_idx, detections in enumerate(frame_boxes):
        if frame_idx == reference_idx:
            continue
        remaining = sorted((det for det in detections if isinstance(det.get("bbox"), tuple)), key=lambda det: _bbox_center(det["bbox"])[0])  # type: ignore[index]
        assigned: dict[str, BBox] = {}
        used: set[int] = set()
        for track_id in track_ids:
            prev_bbox = previous_boxes.get(track_id) or per_frame[reference_idx].get(track_id)
            if prev_bbox is None:
                continue
            prev_center = _bbox_center(prev_bbox)
            best_idx = None
            best_cost = float("inf")
            for det_idx, det in enumerate(remaining):
                if det_idx in used:
                    continue
                bbox = det["bbox"]  # type: ignore[index]
                center = _bbox_center(bbox)
                cost = abs(float(center[0]) - float(prev_center[0])) + 0.15 * abs(float(center[1]) - float(prev_center[1]))
                if cost < best_cost:
                    best_cost = cost
                    best_idx = det_idx
            if best_idx is not None:
                used.add(best_idx)
                assigned[track_id] = remaining[best_idx]["bbox"]  # type: ignore[index]
        per_frame[frame_idx] = assigned
        if assigned:
            previous_boxes.update(assigned)
    return track_ids, per_frame


def build_external_seed_and_tip_priors_from_yolo_json(
    json_path: Path,
    frame_items: list[DatasetImageItem],
) -> tuple[dict[str, object] | None, dict[int, dict[str, list[Point]]], dict[str, object]]:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    frames_raw = payload.get("frames") if isinstance(payload, dict) else payload
    frames = frames_raw if isinstance(frames_raw, list) else []
    lookup = _frame_lookup(frame_items)
    frame_boxes: list[list[dict[str, object]]] = [[] for _ in frame_items]
    raw_tip_points: dict[int, list[dict[str, object]]] = {}
    matched_frames = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_key_candidates = [
            frame.get("uid"),
            frame.get("name"),
            frame.get("image"),
            frame.get("file"),
            frame.get("path"),
            Path(str(frame.get("image", ""))).name,
            Path(str(frame.get("image", ""))).stem,
        ]
        frame_idx = None
        for candidate in frame_key_candidates:
            key = str(candidate or "").strip()
            if key and key in lookup:
                frame_idx = int(lookup[key])
                break
        if frame_idx is None:
            raw_index = frame.get("frame_index")
            if isinstance(raw_index, int) and 0 <= raw_index < len(frame_items):
                frame_idx = int(raw_index)
        if frame_idx is None:
            continue
        matched_frames += 1
        detections = frame.get("detections") or frame.get("predictions") or frame.get("results") or []
        if not isinstance(detections, list):
            continue
        for det in detections:
            if not isinstance(det, dict):
                continue
            class_name = _normalize_name(det.get("class_name") or det.get("name") or det.get("class"))
            track_id = str(det.get("track_id") or det.get("track") or det.get("id") or "").strip()
            bbox = (
                _parse_bbox_xyxy(det.get("bbox_xyxy"))
                or _parse_bbox_xyxy(det.get("bbox"))
                or _parse_bbox_xyxy(det.get("box"))
            )
            keypoints = _extract_keypoints(det.get("keypoints"))
            if class_name in YOLO_CROWN_CLASS_NAMES and isinstance(bbox, tuple):
                frame_boxes[frame_idx].append({"track_id": track_id, "bbox": bbox})
                if keypoints:
                    raw_tip_points.setdefault(frame_idx, []).append(
                        {
                            "track_id": track_id,
                            "point": keypoints[-1],
                            "bbox": bbox,
                        }
                    )
            elif class_name in YOLO_TIP_CLASS_NAMES:
                point = keypoints[0] if keypoints else (_bbox_center(bbox) if isinstance(bbox, tuple) else None)
                if point is not None:
                    raw_tip_points.setdefault(frame_idx, []).append(
                        {
                            "track_id": track_id,
                            "point": point,
                            "bbox": bbox,
                        }
                    )

    track_ids, per_frame_boxes = _best_track_ids_from_boxes(frame_boxes)
    if not track_ids:
        return None, {}, {"status": "no_tracks", "matched_frames": int(matched_frames), "track_count": 0}

    track_bboxes: dict[str, list[BBox]] = {
        track_id: [(0, 0, 0, 0) for _ in frame_items]
        for track_id in track_ids
    }
    for frame_idx, frame_map in per_frame_boxes.items():
        for track_id, bbox in frame_map.items():
            if track_id in track_bboxes:
                track_bboxes[track_id][int(frame_idx)] = tuple(int(v) for v in bbox)

    seed_frame = max(range(len(frame_items)), key=lambda idx: sum(1 for track_id in track_ids if track_bboxes[track_id][idx][2] > 0), default=0)
    tip_priors_by_frame: dict[int, dict[str, list[Point]]] = {}
    for frame_idx, tips in raw_tip_points.items():
        frame_assignment: dict[str, list[Point]] = {}
        frame_track_boxes = per_frame_boxes.get(frame_idx, {})
        for tip in tips:
            track_id = str(tip.get("track_id", "")).strip()
            point = tip.get("point")
            if not (isinstance(point, tuple) and len(point) >= 2):
                continue
            if not track_id or track_id not in track_bboxes:
                containing = [
                    (candidate_id, bbox)
                    for candidate_id, bbox in frame_track_boxes.items()
                    if bbox[2] > 0
                    and bbox[3] > 0
                    and int(point[0]) >= int(bbox[0])
                    and int(point[0]) < int(bbox[0] + bbox[2])
                    and int(point[1]) >= int(bbox[1])
                    and int(point[1]) < int(bbox[1] + bbox[3])
                ]
                if containing:
                    track_id = containing[0][0]
                else:
                    best_track = None
                    best_cost = float("inf")
                    for candidate_id, bbox in frame_track_boxes.items():
                        center = _bbox_center(bbox)
                        cost = float(math.hypot(float(center[0]) - float(point[0]), float(center[1]) - float(point[1])))
                        if cost < best_cost:
                            best_cost = cost
                            best_track = candidate_id
                    track_id = str(best_track or "")
            if track_id and track_id in track_bboxes:
                frame_assignment.setdefault(track_id, []).append((int(point[0]), int(point[1])))
        if frame_assignment:
            tip_priors_by_frame[int(frame_idx)] = {
                track_id: sorted({(int(x), int(y)) for x, y in points}, key=lambda p: (-int(p[1]), int(p[0])))[:6]
                for track_id, points in frame_assignment.items()
            }

    external_tracking_seed = {
        "track_ids": track_ids,
        "track_bboxes": track_bboxes,
        "overlap_frames": {track_id: None for track_id in track_ids},
        "seed_frame": int(seed_frame),
        "effective_min_component_area": 1,
        "source": "yolo26_anchor_tracking",
    }
    summary = {
        "status": "ok",
        "matched_frames": int(matched_frames),
        "track_count": int(len(track_ids)),
        "tip_prior_frames": int(len(tip_priors_by_frame)),
        "source": "yolo26_anchor_tracking",
    }
    return external_tracking_seed, tip_priors_by_frame, summary
