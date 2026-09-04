from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_MASK_TOTAL_CLASS_IDS = (1, 3)
DEFAULT_MASK_TOTAL_SHOOT_CLASS_ID = 2
DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS = (DEFAULT_MASK_TOTAL_SHOOT_CLASS_ID,)
MASK_TOTAL_ROOT_METRIC_COLUMNS = {
    "total": "total_root_length_mm",
    "weighted_total": "total_root_length_weighted_mm",
    "primary": "primary_root_length_mm",
    "lateral": "lateral_root_length_mm",
}
MASK_TOTAL_FRAME_REVIEW_FILENAME = "npec_mask_total_frames_needing_review.csv"
MASK_TOTAL_FRAME_REVIEW_STATUS_COLUMN = "mask_total_review_status"
MASK_TOTAL_FRAME_REVIEW_REASON_COLUMN = "mask_total_review_reasons"
MASK_TOTAL_FRAME_REVIEW_COLUMNS = [
    "Series",
    "PetriDish",
    "Timestamp",
    "FrameIndex",
    "SourceFile",
    "OutputMaskPath",
    "PreviewImagePath",
    MASK_TOTAL_FRAME_REVIEW_STATUS_COLUMN,
    MASK_TOTAL_FRAME_REVIEW_REASON_COLUMN,
    "total_root_length_mm",
    "delta_total_root_length_mm",
    "root_length_drop_fraction",
    "primary_root_length_mm",
    "lateral_root_length_mm",
    "total_root_pixels_selected_classes",
    "primary_root_pixels_selected_class",
    "lateral_root_pixels_selected_class",
    "shoot_measurement_source",
    "shoot_rgb_rescue_enabled",
    "shoot_rgb_rescue_abstained",
    "shoot_rgb_rescue_fallback_status",
    "shoot_area_px",
    "shoot_area_model_px",
    "shoot_rgb_rescue_removed_model_px",
    "shoot_rgb_rescue_green_only_px",
    "shoot_model_rejected_fraction",
    "shoot_green_fraction_of_model",
    "shoot_rgb_rescue_components",
]


@dataclass(slots=True)
class MaskTotalMeasurement:
    Series: str
    PetriDish: str
    Timestamp: object
    FrameIndex: int
    RelativeFolder: str
    SourceFile: str
    OutputMaskPath: str
    PreviewImagePath: str
    mask_height: int
    mask_width: int
    pixel_size_mm: float
    measurement_class_ids: str
    shoot_class_ids: str
    primary_root_class_id: int
    lateral_root_class_id: int
    total_root_pixels_selected_classes: int
    primary_root_pixels_selected_class: int
    lateral_root_pixels_selected_class: int
    total_root_pixels_1_plus_3: int
    root_pixels_class_1: int
    lateral_pixels_class_3: int
    shoot_seed_pixels_class_2: int
    shoot_pixels_class_2: int
    shoot_pixels_selected_classes: int
    measured_mask_pixels: int
    shoot_area_px: int
    shoot_area_mm2: float
    shoot_area_selected_px: int
    shoot_area_selected_mm2: float
    shoot_area_model_px: int
    shoot_area_model_mm2: float
    shoot_rgb_rescue_enabled: bool
    shoot_rgb_rescue_applied: bool
    shoot_rgb_rescue_abstained: bool
    shoot_rgb_rescue_fallback_status: str
    shoot_rgb_rescue_error: str
    shoot_rgb_rescue_candidate_px: int
    shoot_rgb_rescue_added_px: int
    shoot_rgb_rescue_removed_model_px: int
    shoot_rgb_rescue_green_only_px: int
    shoot_rgb_rescue_components: int
    shoot_measurement_source: str
    component_count: int
    largest_component_area_px: int
    primary_root_length_px: float
    primary_root_length_mm: float
    primary_root_length_weighted_px: float
    primary_root_length_weighted_mm: float
    lateral_root_length_px: float
    lateral_root_length_mm: float
    lateral_root_length_weighted_px: float
    lateral_root_length_weighted_mm: float
    total_root_length_px: float
    total_root_length_mm: float
    total_root_length_weighted_px: float
    total_root_length_weighted_mm: float
    root_to_shoot_area_ratio_px: float
    root_length_mm_per_shoot_area_mm2: float
    mean_root_length_mm: float
    plants_detected: int
    analysis_mode: str


@dataclass(frozen=True, slots=True)
class LazyOwnershipClassSelection:
    root_class_id: int
    lateral_class_id: int | None
    shoot_class_id: int | None
    ignored_root_class_ids: tuple[int, ...] = ()
    ignored_shoot_class_ids: tuple[int, ...] = ()


def parse_class_id_text(text: object, fallback: Iterable[int] = DEFAULT_MASK_TOTAL_CLASS_IDS) -> list[int]:
    raw = str(text or "").strip()
    values: list[int] = []
    for part in re.split(r"[,;\s]+", raw):
        token = part.strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if value <= 0 or value > 255 or value in values:
            continue
        values.append(value)
    if values:
        return values
    return [int(v) for v in fallback if int(v) > 0]


def _resolve_class_ids(value: object, fallback: Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return parse_class_id_text(value, fallback=fallback)
    try:
        text = ",".join(str(int(v)) for v in value)  # type: ignore[union-attr]
    except Exception:
        text = ""
    return parse_class_id_text(text, fallback=fallback)


def resolve_lazy_ownership_class_selection(
    root_class_ids: object,
    shoot_class_ids: object,
    *,
    root_fallback: Iterable[int] = DEFAULT_MASK_TOTAL_CLASS_IDS,
    shoot_fallback: Iterable[int] = DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS,
) -> LazyOwnershipClassSelection:
    resolved_roots = _resolve_class_ids(root_class_ids, root_fallback)
    if not resolved_roots:
        resolved_roots = [int(v) for v in root_fallback if int(v) > 0]
    resolved_shoots = _resolve_class_ids(shoot_class_ids, shoot_fallback)
    if not resolved_shoots:
        resolved_shoots = [int(v) for v in shoot_fallback if int(v) > 0]

    root_class_id = int(resolved_roots[0]) if resolved_roots else 1
    lateral_class_id: int | None = None
    if len(resolved_roots) >= 2 and int(resolved_roots[1]) != int(root_class_id):
        lateral_class_id = int(resolved_roots[1])

    ignored_roots_start = 2 if lateral_class_id is not None else 1
    ignored_root_class_ids = tuple(int(v) for v in resolved_roots[ignored_roots_start:])
    shoot_class_id = int(resolved_shoots[0]) if resolved_shoots else None
    ignored_shoot_class_ids = tuple(int(v) for v in resolved_shoots[1:])

    return LazyOwnershipClassSelection(
        root_class_id=int(root_class_id),
        lateral_class_id=lateral_class_id,
        shoot_class_id=shoot_class_id,
        ignored_root_class_ids=ignored_root_class_ids,
        ignored_shoot_class_ids=ignored_shoot_class_ids,
    )


def _primary_lateral_class_ids(class_ids: Iterable[int]) -> tuple[int, int]:
    resolved = _resolve_class_ids(class_ids, DEFAULT_MASK_TOTAL_CLASS_IDS)
    if not resolved:
        resolved = list(DEFAULT_MASK_TOTAL_CLASS_IDS)
    primary_class_id = int(resolved[0])
    lateral_class_id = int(resolved[1]) if len(resolved) > 1 else 0
    return primary_class_id, lateral_class_id


def _positive_int_or_zero(value: object) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        return 0
    return parsed if parsed > 0 else 0


def _primary_lateral_class_ids_from_row(row: pd.Series) -> tuple[int, int]:
    primary_class_id = _positive_int_or_zero(row.get("primary_root_class_id", 0))
    lateral_class_id = _positive_int_or_zero(row.get("lateral_root_class_id", 0))
    if primary_class_id > 0:
        return primary_class_id, lateral_class_id
    parsed = parse_class_id_text(row.get("measurement_class_ids", ""), fallback=DEFAULT_MASK_TOTAL_CLASS_IDS)
    if not parsed:
        parsed = list(DEFAULT_MASK_TOTAL_CLASS_IDS)
    primary_class_id = int(parsed[0])
    lateral_class_id = int(parsed[1]) if len(parsed) > 1 else 0
    return primary_class_id, lateral_class_id


def natural_sort_key(value: object) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(\d+)", str(value))
    key: list[tuple[int, object]] = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        elif part:
            key.append((1, part.lower()))
    return tuple(key)


def _read_index_mask(mask_path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D index mask, got shape {arr.shape}: {mask_path}")
    return np.asarray(arr, dtype=np.uint8)


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    mask_bin = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if int(np.count_nonzero(mask_bin)) <= 0:
        return mask_bin.astype(bool)
    try:
        from skimage.morphology import skeletonize  # type: ignore

        return np.asarray(skeletonize(mask_bin > 0), dtype=bool)
    except Exception:
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


def _measure_combined_mask(root_mask: np.ndarray, min_component_area: int) -> tuple[float, float, int, int]:
    root_u8 = (np.asarray(root_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if int(np.count_nonzero(root_u8)) <= 0:
        return 0.0, 0.0, 0, 0

    label_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(root_u8, connectivity=8)
    total_px = 0.0
    total_weighted_px = 0.0
    component_count = 0
    largest_area = 0
    for label_id in range(1, int(label_count)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < int(min_component_area):
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        crop = labels[y : y + h, x : x + w] == label_id
        if not np.any(crop):
            continue
        skeleton = _skeletonize(crop)
        total_px += float(np.count_nonzero(skeleton))
        total_weighted_px += float(_skeleton_edge_length_px(skeleton))
        component_count += 1
        largest_area = max(largest_area, area)
    return total_px, total_weighted_px, component_count, largest_area


def _class_metric_prefix(class_id: int) -> str:
    return f"class_{int(class_id)}"


def _normalize_root_metric_mode(value: object) -> str:
    mode = str(value or "total").strip().lower()
    return mode if mode in MASK_TOTAL_ROOT_METRIC_COLUMNS else "total"


def _record_pixel_size_mm(record: dict[str, object], fallback_pixel_size_mm: float) -> float:
    details = record.get("details")
    if isinstance(details, dict):
        for key in ("pixel_size_mm", "detected_pixel_size_mm", "analytics_pixel_size_mm"):
            value = details.get(key)
            try:
                parsed = float(value)
            except Exception:
                continue
            if math.isfinite(parsed) and parsed > 0:
                return float(parsed)
    return float(fallback_pixel_size_mm)


def _read_rgb_image(path: Path, shape_hw: tuple[int, int]) -> np.ndarray | None:
    try:
        with Image.open(path) as pil:
            rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None
    if rgb.shape[:2] != shape_hw:
        try:
            rgb = np.asarray(Image.fromarray(rgb).resize((shape_hw[1], shape_hw[0]), Image.BILINEAR), dtype=np.uint8)
        except Exception:
            return None
    return rgb


def _measurement_only_green_shoot_rescue(
    image_path: Path,
    *,
    shape_hw: tuple[int, int],
    class_shoot_mask: np.ndarray,
    measured_root_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    base_shoot = np.asarray(class_shoot_mask, dtype=bool)
    root_bin = np.asarray(measured_root_mask, dtype=bool)
    meta: dict[str, object] = {
        "enabled": True,
        "applied": False,
        "rgb_like": False,
        "candidate_pixels": 0,
        "added_pixels": 0,
        "removed_model_pixels": 0,
        "green_only_pixels": 0,
        "components_kept": 0,
        "mode": "green_only",
    }

    def _model_fallback_result(error: str | None = None) -> tuple[np.ndarray, dict[str, object]]:
        if error:
            meta["error"] = error
        meta["abstained"] = True
        meta["fallback_status"] = "retained_model_mask"
        meta["retained_model_pixels"] = int(np.count_nonzero(base_shoot))
        meta["green_only_pixels"] = 0
        return base_shoot.astype(bool, copy=True), meta

    rgb = _read_rgb_image(image_path, shape_hw)
    if rgb is None:
        return _model_fallback_result("source_image_unreadable")
    try:
        from .pyphenotyper_adapter import build_lucifer_green_shoot_mask

        color_mask, color_meta = build_lucifer_green_shoot_mask(rgb, shape_hw, root_mask=measured_root_mask)
    except Exception as exc:
        return _model_fallback_result(str(exc))

    meta.update(color_meta)
    candidate_pixels = int(np.count_nonzero(color_mask))
    meta["candidate_pixels"] = candidate_pixels
    min_pixels = max(20, int(round(float(shape_hw[0] * shape_hw[1]) * 0.000001)))
    if candidate_pixels < min_pixels:
        return _model_fallback_result()

    green_only = (np.asarray(color_mask, dtype=np.uint8) > 0) & (~root_bin)
    added_pixels = int(np.count_nonzero(green_only & (~base_shoot)))
    removed_pixels = int(np.count_nonzero(base_shoot & (~green_only)))
    meta["added_pixels"] = int(added_pixels)
    meta["removed_model_pixels"] = int(removed_pixels)
    meta["green_only_pixels"] = int(np.count_nonzero(green_only))
    meta["applied"] = bool(added_pixels > 0 or removed_pixels > 0)
    return green_only.astype(bool), meta


def build_mask_total_dataframe(
    segmentation_records: list[dict[str, object]],
    *,
    class_ids: Iterable[int],
    shoot_class_ids: Iterable[int] = DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS,
    fallback_pixel_size_mm: float,
    min_component_area: int = 1,
    shoot_rgb_rescue: bool = False,
    progress_callback: Callable[[int, int, str], bool] | None = None,
) -> pd.DataFrame:
    resolved_class_ids = _resolve_class_ids(class_ids, DEFAULT_MASK_TOTAL_CLASS_IDS)
    if not resolved_class_ids:
        resolved_class_ids = list(DEFAULT_MASK_TOTAL_CLASS_IDS)
    resolved_shoot_class_ids = _resolve_class_ids(shoot_class_ids, DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS)
    if not resolved_shoot_class_ids:
        resolved_shoot_class_ids = list(DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS)

    rows: list[dict[str, object]] = []
    class_array = np.asarray(resolved_class_ids, dtype=np.uint8)
    shoot_class_array = np.asarray(resolved_shoot_class_ids, dtype=np.uint8)
    primary_class_id, lateral_class_id = _primary_lateral_class_ids(resolved_class_ids)
    class_id_label = ",".join(str(int(v)) for v in resolved_class_ids)
    shoot_class_id_label = ",".join(str(int(v)) for v in resolved_shoot_class_ids)
    total_records = len(segmentation_records)
    for record_index, record in enumerate(segmentation_records, start=1):
        if progress_callback is not None:
            label = str(record.get("image_name") or record.get("uid") or record_index)
            if not bool(progress_callback(int(record_index), int(max(1, total_records)), label)):
                break
        meta = record.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        mask_path_raw = record.get("output_mask")
        image_path_raw = record.get("image_path")
        if not isinstance(mask_path_raw, (str, Path)) or not isinstance(image_path_raw, (str, Path)):
            continue
        mask_path = Path(mask_path_raw)
        if not mask_path.exists():
            continue
        arr = _read_index_mask(mask_path)
        measured_mask = np.isin(arr, class_array)
        length_px, weighted_px, component_count, largest_area = _measure_combined_mask(
            measured_mask,
            min_component_area=int(min_component_area),
        )
        primary_length_px, primary_weighted_px, _primary_components, _primary_largest = _measure_combined_mask(
            arr == np.uint8(primary_class_id),
            min_component_area=int(min_component_area),
        )
        lateral_mask = arr == np.uint8(lateral_class_id) if lateral_class_id > 0 else np.zeros_like(arr, dtype=bool)
        lateral_length_px, lateral_weighted_px, _lateral_components, _lateral_largest = _measure_combined_mask(
            lateral_mask,
            min_component_area=int(min_component_area),
        )
        class_measure_cache: dict[int, tuple[float, float, int, int]] = {
            int(primary_class_id): (
                float(primary_length_px),
                float(primary_weighted_px),
                int(_primary_components),
                int(_primary_largest),
            )
        }
        if int(lateral_class_id) > 0:
            class_measure_cache[int(lateral_class_id)] = (
                float(lateral_length_px),
                float(lateral_weighted_px),
                int(_lateral_components),
                int(_lateral_largest),
            )
        pixel_size_mm = _record_pixel_size_mm(record, fallback_pixel_size_mm)
        total_mm = float(length_px * pixel_size_mm)
        primary_mm = float(primary_length_px * pixel_size_mm)
        lateral_mm = float(lateral_length_px * pixel_size_mm)
        shoot_class_mask = np.isin(arr, shoot_class_array)
        shoot_model_px = int(np.count_nonzero(shoot_class_mask))
        class_2_shoot_px = int(np.count_nonzero(arr == DEFAULT_MASK_TOTAL_SHOOT_CLASS_ID))
        selected_root_area_px = int(np.count_nonzero(measured_mask))
        legacy_root_area_px = int(np.count_nonzero((arr == 1) | (arr == 3)))
        primary_selected_px = int(np.count_nonzero(arr == np.uint8(primary_class_id)))
        lateral_selected_px = int(np.count_nonzero(lateral_mask))
        root_area_px = int(selected_root_area_px)
        rescue_meta: dict[str, object] = {
            "enabled": bool(shoot_rgb_rescue),
            "applied": False,
            "candidate_pixels": 0,
            "added_pixels": 0,
            "removed_model_pixels": 0,
            "green_only_pixels": 0,
            "components_kept": 0,
            "mode": "mask_class",
        }
        shoot_measurement_mask = shoot_class_mask
        if bool(shoot_rgb_rescue):
            shoot_measurement_mask, rescue_meta = _measurement_only_green_shoot_rescue(
                Path(image_path_raw),
                shape_hw=tuple(int(v) for v in arr.shape[:2]),
                class_shoot_mask=shoot_class_mask,
                measured_root_mask=measured_mask,
            )
        shoot_px = int(np.count_nonzero(shoot_measurement_mask))
        shoot_area_mm2 = float(shoot_px * (pixel_size_mm ** 2))
        shoot_model_area_mm2 = float(shoot_model_px * (pixel_size_mm ** 2))
        shoot_rescue_added_px = int(rescue_meta.get("added_pixels", 0) or 0)
        shoot_rescue_removed_model_px = int(rescue_meta.get("removed_model_pixels", 0) or 0)
        shoot_rescue_green_only_px = int(rescue_meta.get("green_only_pixels", 0) or 0)
        root_to_shoot_area_ratio_px = float(root_area_px / shoot_px) if shoot_px > 0 else 0.0
        root_length_mm_per_shoot_area_mm2 = float(total_mm / shoot_area_mm2) if shoot_area_mm2 > 0 else 0.0
        per_class_metrics: dict[str, object] = {}
        for class_id in resolved_class_ids:
            class_mask = arr == np.uint8(class_id)
            cached_measure = class_measure_cache.get(int(class_id))
            if cached_measure is None:
                class_length_px, class_weighted_px, class_components, class_largest = _measure_combined_mask(
                    class_mask,
                    min_component_area=int(min_component_area),
                )
            else:
                class_length_px, class_weighted_px, class_components, class_largest = cached_measure
            prefix = _class_metric_prefix(int(class_id))
            per_class_metrics[f"{prefix}_pixels"] = int(np.count_nonzero(class_mask))
            per_class_metrics[f"{prefix}_length_px"] = float(class_length_px)
            per_class_metrics[f"{prefix}_length_mm"] = float(class_length_px * pixel_size_mm)
            per_class_metrics[f"{prefix}_weighted_length_px"] = float(class_weighted_px)
            per_class_metrics[f"{prefix}_weighted_length_mm"] = float(class_weighted_px * pixel_size_mm)
            per_class_metrics[f"{prefix}_component_count"] = int(class_components)
            per_class_metrics[f"{prefix}_largest_component_area_px"] = int(class_largest)
        for class_id in resolved_shoot_class_ids:
            shoot_class_mask = arr == np.uint8(class_id)
            per_class_metrics[f"shoot_class_{int(class_id)}_pixels"] = int(np.count_nonzero(shoot_class_mask))

        try:
            frame_index_value = int(meta.get("frame_index", len(rows)))
        except (TypeError, ValueError):
            frame_index_value = int(len(rows))
        measurement = MaskTotalMeasurement(
                Series=str(meta.get("series", "Unknown")),
                PetriDish=str(meta.get("petri", "Unknown")),
                Timestamp=meta.get("timestamp"),
                FrameIndex=frame_index_value,
                RelativeFolder=str(meta.get("relative_folder", ".")),
                SourceFile=str(image_path_raw),
                OutputMaskPath=str(mask_path),
                PreviewImagePath=str(image_path_raw),
                mask_height=int(arr.shape[0]),
                mask_width=int(arr.shape[1]),
                pixel_size_mm=float(pixel_size_mm),
                measurement_class_ids=class_id_label,
                shoot_class_ids=shoot_class_id_label,
                primary_root_class_id=int(primary_class_id),
                lateral_root_class_id=int(lateral_class_id),
                total_root_pixels_selected_classes=int(selected_root_area_px),
                primary_root_pixels_selected_class=int(primary_selected_px),
                lateral_root_pixels_selected_class=int(lateral_selected_px),
                total_root_pixels_1_plus_3=int(legacy_root_area_px),
                root_pixels_class_1=int(np.count_nonzero(arr == 1)),
                lateral_pixels_class_3=int(np.count_nonzero(arr == 3)),
                shoot_seed_pixels_class_2=int(class_2_shoot_px),
                shoot_pixels_class_2=int(class_2_shoot_px),
                shoot_pixels_selected_classes=int(shoot_model_px),
                measured_mask_pixels=int(np.count_nonzero(measured_mask)),
                shoot_area_px=int(shoot_px),
                shoot_area_mm2=float(shoot_area_mm2),
                shoot_area_selected_px=int(shoot_model_px),
                shoot_area_selected_mm2=float(shoot_model_area_mm2),
                shoot_area_model_px=int(shoot_model_px),
                shoot_area_model_mm2=float(shoot_model_area_mm2),
                shoot_rgb_rescue_enabled=bool(shoot_rgb_rescue),
                shoot_rgb_rescue_applied=bool(rescue_meta.get("applied", False)),
                shoot_rgb_rescue_abstained=bool(rescue_meta.get("abstained", False)),
                shoot_rgb_rescue_fallback_status=str(rescue_meta.get("fallback_status", "") or ""),
                shoot_rgb_rescue_error=str(rescue_meta.get("error", "") or ""),
                shoot_rgb_rescue_candidate_px=int(rescue_meta.get("candidate_pixels", 0) or 0),
                shoot_rgb_rescue_added_px=int(shoot_rescue_added_px),
                shoot_rgb_rescue_removed_model_px=int(shoot_rescue_removed_model_px),
                shoot_rgb_rescue_green_only_px=int(shoot_rescue_green_only_px),
                shoot_rgb_rescue_components=int(rescue_meta.get("components_kept", 0) or 0),
                shoot_measurement_source=(
                    "rgb_green_only" if bool(shoot_rgb_rescue) else "mask_class"
                ),
                component_count=int(component_count),
                largest_component_area_px=int(largest_area),
                primary_root_length_px=float(primary_length_px),
                primary_root_length_mm=float(primary_mm),
                primary_root_length_weighted_px=float(primary_weighted_px),
                primary_root_length_weighted_mm=float(primary_weighted_px * pixel_size_mm),
                lateral_root_length_px=float(lateral_length_px),
                lateral_root_length_mm=float(lateral_mm),
                lateral_root_length_weighted_px=float(lateral_weighted_px),
                lateral_root_length_weighted_mm=float(lateral_weighted_px * pixel_size_mm),
                total_root_length_px=float(length_px),
                total_root_length_mm=float(total_mm),
                total_root_length_weighted_px=float(weighted_px),
                total_root_length_weighted_mm=float(weighted_px * pixel_size_mm),
                root_to_shoot_area_ratio_px=float(root_to_shoot_area_ratio_px),
                root_length_mm_per_shoot_area_mm2=float(root_length_mm_per_shoot_area_mm2),
                mean_root_length_mm=float(total_mm),
                plants_detected=1,
                analysis_mode="mask_total",
        )
        row = asdict(measurement)
        row.update(
            {
                "OriginalSourceFile": str(meta.get("original_source_file", image_path_raw)),
                "OriginalFileName": str(meta.get("original_file_name", Path(str(image_path_raw)).name)),
                "AllocatedFileName": str(meta.get("allocated_file_name", Path(str(image_path_raw)).name)),
                "PlateIdentityStatus": str(meta.get("plate_identity_status", "")),
                "PlateIdentityConfidence": meta.get("plate_identity_confidence", ""),
            }
        )
        row.update(per_class_metrics)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["FrameIndex"] = pd.to_numeric(df["FrameIndex"], errors="coerce").fillna(-1).astype(int)
    df["_sort_series"] = df["Series"].fillna("Unknown").astype(str).map(natural_sort_key)
    df["_sort_petri"] = df["PetriDish"].fillna("Unknown").astype(str).map(natural_sort_key)
    df["_sort_frame"] = pd.to_numeric(df["FrameIndex"], errors="coerce").fillna(1.0e12)
    df.sort_values(["_sort_series", "_sort_petri", "Timestamp", "_sort_frame"], inplace=True, kind="mergesort")
    df.drop(columns=["_sort_series", "_sort_petri", "_sort_frame"], inplace=True, errors="ignore")
    df.reset_index(drop=True, inplace=True)
    df["delta_total_root_length_mm"] = df.groupby(["Series", "PetriDish"], dropna=False)["total_root_length_mm"].diff().fillna(0.0)
    df["delta_total_root_length_weighted_mm"] = (
        df.groupby(["Series", "PetriDish"], dropna=False)["total_root_length_weighted_mm"].diff().fillna(0.0)
    )
    df["delta_primary_root_length_mm"] = (
        df.groupby(["Series", "PetriDish"], dropna=False)["primary_root_length_mm"].diff().fillna(0.0)
    )
    df["delta_primary_root_length_weighted_mm"] = (
        df.groupby(["Series", "PetriDish"], dropna=False)["primary_root_length_weighted_mm"].diff().fillna(0.0)
    )
    df["delta_lateral_root_length_mm"] = (
        df.groupby(["Series", "PetriDish"], dropna=False)["lateral_root_length_mm"].diff().fillna(0.0)
    )
    df["delta_lateral_root_length_weighted_mm"] = (
        df.groupby(["Series", "PetriDish"], dropna=False)["lateral_root_length_weighted_mm"].diff().fillna(0.0)
    )
    df["delta_shoot_area_px"] = df.groupby(["Series", "PetriDish"], dropna=False)["shoot_area_px"].diff().fillna(0).astype(int)
    df["delta_shoot_area_mm2"] = df.groupby(["Series", "PetriDish"], dropna=False)["shoot_area_mm2"].diff().fillna(0.0)
    return df


def build_mask_total_timelapse_dataframe(mask_df: pd.DataFrame, root_metric_mode: object = "total") -> pd.DataFrame:
    """Return a frame-level table with explicit root/shoot metric aliases."""
    if mask_df is None or mask_df.empty:
        return pd.DataFrame()
    timeline = mask_df.copy()
    mode = _normalize_root_metric_mode(root_metric_mode)
    selected_col = MASK_TOTAL_ROOT_METRIC_COLUMNS[mode]
    if selected_col not in timeline.columns:
        mode = "total"
        selected_col = MASK_TOTAL_ROOT_METRIC_COLUMNS[mode]

    for column in (
        "total_root_length_mm",
        "total_root_length_weighted_mm",
        "primary_root_length_mm",
        "primary_root_length_weighted_mm",
        "lateral_root_length_mm",
        "lateral_root_length_weighted_mm",
        "shoot_area_px",
        "shoot_area_mm2",
        "shoot_area_model_px",
        "shoot_area_model_mm2",
        "shoot_rgb_rescue_candidate_px",
        "shoot_rgb_rescue_added_px",
        "shoot_rgb_rescue_removed_model_px",
        "shoot_rgb_rescue_green_only_px",
        "shoot_rgb_rescue_components",
        "shoot_pixels_selected_classes",
        "shoot_area_selected_px",
        "shoot_area_selected_mm2",
        "total_root_pixels_selected_classes",
        "primary_root_pixels_selected_class",
        "lateral_root_pixels_selected_class",
        "primary_root_class_id",
        "lateral_root_class_id",
        "plants_detected",
    ):
        if column in timeline.columns:
            timeline[column] = pd.to_numeric(timeline[column], errors="coerce").fillna(0.0)

    timeline["selected_root_metric_mode"] = mode
    timeline["selected_root_metric_column"] = selected_col
    timeline["selected_root_length_mm"] = pd.to_numeric(timeline[selected_col], errors="coerce").fillna(0.0)

    if {"primary_root_length_mm", "lateral_root_length_mm"}.issubset(timeline.columns):
        timeline["primary_plus_lateral_root_length_mm"] = (
            pd.to_numeric(timeline["primary_root_length_mm"], errors="coerce").fillna(0.0)
            + pd.to_numeric(timeline["lateral_root_length_mm"], errors="coerce").fillna(0.0)
        )
        denom = timeline["primary_plus_lateral_root_length_mm"].replace(0.0, np.nan)
        timeline["primary_fraction_of_root_length"] = (
            pd.to_numeric(timeline["primary_root_length_mm"], errors="coerce").fillna(0.0) / denom
        ).fillna(0.0)
        timeline["lateral_fraction_of_root_length"] = (
            pd.to_numeric(timeline["lateral_root_length_mm"], errors="coerce").fillna(0.0) / denom
        ).fillna(0.0)
    else:
        timeline["primary_plus_lateral_root_length_mm"] = 0.0
        timeline["primary_fraction_of_root_length"] = 0.0
        timeline["lateral_fraction_of_root_length"] = 0.0

    if "shoot_area_mm2" in timeline.columns:
        shoot_denom = pd.to_numeric(timeline["shoot_area_mm2"], errors="coerce").replace(0.0, np.nan)
        timeline["selected_root_length_mm_per_shoot_area_mm2"] = (
            pd.to_numeric(timeline["selected_root_length_mm"], errors="coerce").fillna(0.0) / shoot_denom
        ).fillna(0.0)

    group_cols = [column for column in ("Series", "PetriDish") if column in timeline.columns]
    if not group_cols:
        group_cols = ["PetriDish"] if "PetriDish" in timeline.columns else []
    for value_col, delta_col in (
        ("selected_root_length_mm", "delta_selected_root_length_mm"),
        ("primary_plus_lateral_root_length_mm", "delta_primary_plus_lateral_root_length_mm"),
        ("shoot_area_model_px", "delta_shoot_area_model_px"),
        ("shoot_area_model_mm2", "delta_shoot_area_model_mm2"),
        ("shoot_rgb_rescue_added_px", "delta_shoot_rgb_rescue_added_px"),
        ("shoot_rgb_rescue_removed_model_px", "delta_shoot_rgb_rescue_removed_model_px"),
        ("shoot_rgb_rescue_green_only_px", "delta_shoot_rgb_rescue_green_only_px"),
        ("plants_detected", "delta_plants_detected"),
    ):
        if value_col not in timeline.columns:
            continue
        if group_cols:
            timeline[delta_col] = timeline.groupby(group_cols, dropna=False)[value_col].diff().fillna(0.0)
        else:
            timeline[delta_col] = pd.to_numeric(timeline[value_col], errors="coerce").diff().fillna(0.0)

    if "Series" in timeline.columns:
        timeline["_sort_series"] = timeline["Series"].fillna("Unknown").astype(str).map(natural_sort_key)
    if "PetriDish" in timeline.columns:
        timeline["_sort_petri"] = timeline["PetriDish"].fillna("Unknown").astype(str).map(natural_sort_key)
    if "FrameIndex" in timeline.columns:
        timeline["_sort_frame"] = pd.to_numeric(timeline["FrameIndex"], errors="coerce").fillna(1.0e12)
    sort_cols = [col for col in ("_sort_series", "_sort_petri", "Timestamp", "_sort_frame") if col in timeline.columns]
    if sort_cols:
        timeline.sort_values(sort_cols, inplace=True, kind="mergesort")
    timeline.drop(columns=["_sort_series", "_sort_petri", "_sort_frame"], inplace=True, errors="ignore")
    timeline.reset_index(drop=True, inplace=True)
    return timeline


def _first_shoot_class_id(mask_df: pd.DataFrame, shoot_class_ids: Iterable[int] | None = None) -> int:
    if shoot_class_ids is not None:
        parsed = _resolve_class_ids(shoot_class_ids, DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS)
        if parsed:
            return int(parsed[0])
    if mask_df is not None and not mask_df.empty and "shoot_class_ids" in mask_df.columns:
        for value in mask_df["shoot_class_ids"].dropna().astype(str):
            parsed = parse_class_id_text(value, fallback=DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS)
            if parsed:
                return int(parsed[0])
    return int(DEFAULT_MASK_TOTAL_SHOOT_CLASS_ID)


def build_mask_total_pmi_dataframe(
    mask_df: pd.DataFrame,
    *,
    shoot_class_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Adapt plate-total mask measurements to the PMI long-table exporter input."""
    if mask_df is None or mask_df.empty:
        return pd.DataFrame()
    out = mask_df.copy()
    if "plant_id" not in out.columns:
        out["plant_id"] = "plate_total"
    else:
        out["plant_id"] = out["plant_id"].fillna("").astype(str).replace("", "plate_total")

    class_pairs = [_primary_lateral_class_ids_from_row(row) for _, row in out.iterrows()]
    out["root_class_id"] = [int(pair[0]) for pair in class_pairs]
    out["lateral_class_id"] = [int(pair[1]) for pair in class_pairs]
    resolved_shoot_class_id = _first_shoot_class_id(out, shoot_class_ids=shoot_class_ids)
    out["shoot_class_id"] = int(resolved_shoot_class_id)
    out["seed_class_id"] = int(resolved_shoot_class_id)

    alias_map = {
        "root_length_px": "total_root_length_px",
        "root_length_px_clean": "total_root_length_px",
        "root_length_mm_raw": "total_root_length_mm",
        "root_length_mm_clean": "total_root_length_mm",
        "primary_root_length_px_clean": "primary_root_length_px",
        "primary_root_length_mm_raw": "primary_root_length_mm",
        "primary_root_length_mm_clean": "primary_root_length_mm",
        "total_root_length_px_clean": "total_root_length_px",
        "total_root_length_mm_raw": "total_root_length_mm",
        "total_root_length_mm_clean": "total_root_length_mm",
        "lateral_total_length_px": "lateral_root_length_px",
        "lateral_total_length_mm": "lateral_root_length_mm",
        "combined_root_length_px": "total_root_length_px",
        "combined_root_length_mm": "total_root_length_mm",
        "root_area_px": ("total_root_pixels_selected_classes", "total_root_pixels_1_plus_3"),
        "primary_root_area_px": ("primary_root_pixels_selected_class", "root_pixels_class_1"),
        "total_root_area_px": ("total_root_pixels_selected_classes", "total_root_pixels_1_plus_3"),
        "n_pixels_main_root": ("primary_root_pixels_selected_class", "root_pixels_class_1"),
        "n_pixels_lateral_root": ("lateral_root_pixels_selected_class", "lateral_pixels_class_3"),
        "n_pixels_shoot": "shoot_area_px",
    }
    for target, sources in alias_map.items():
        if isinstance(sources, str):
            sources = (sources,)
        for source in sources:
            if source in out.columns:
                out[target] = pd.to_numeric(out[source], errors="coerce").fillna(0.0)
                break
    if "n_pixels_shoot" not in out.columns and "shoot_area_px" in out.columns:
        out["n_pixels_shoot"] = pd.to_numeric(out["shoot_area_px"], errors="coerce").fillna(0.0)

    if "pixel_size_mm" in out.columns:
        pixel_size = pd.to_numeric(out["pixel_size_mm"], errors="coerce").fillna(0.0)
        for area_target, pixel_source in (
            ("root_area_mm2", "root_area_px"),
            ("primary_root_area_mm2", "primary_root_area_px"),
            ("total_root_area_mm2", "total_root_area_px"),
        ):
            if pixel_source in out.columns:
                out[area_target] = pd.to_numeric(out[pixel_source], errors="coerce").fillna(0.0) * (pixel_size ** 2)

    lateral_pixel_column = "lateral_root_pixels_selected_class" if "lateral_root_pixels_selected_class" in out.columns else "lateral_pixels_class_3"
    if lateral_pixel_column in out.columns:
        lateral_px = pd.to_numeric(out[lateral_pixel_column], errors="coerce").fillna(0.0)
        out["lateral_count"] = np.where(lateral_px > 0.0, 1, 0)
    dilated_root_column = (
        "total_root_pixels_selected_classes"
        if "total_root_pixels_selected_classes" in out.columns
        else "total_root_pixels_1_plus_3"
    )
    if dilated_root_column in out.columns:
        out["n_pixels_dilated_root"] = pd.to_numeric(out[dilated_root_column], errors="coerce").fillna(0.0)
        out["n_pixels_dilated_root_exclusive"] = 0.0
    out["n_pixels_main_root_tip"] = 0.0
    out["n_pixels_other_tip"] = 0.0
    out["n_pixels_node"] = 0.0
    out["n_pixels_unknown"] = 0.0
    if {"lateral_total_length_px", "lateral_count"}.issubset(out.columns):
        lateral_count = pd.to_numeric(out["lateral_count"], errors="coerce").replace(0.0, np.nan)
        out["lateral_mean_length_px"] = (
            pd.to_numeric(out["lateral_total_length_px"], errors="coerce").fillna(0.0) / lateral_count
        ).fillna(0.0)
        out["lateral_max_length_px"] = pd.to_numeric(out["lateral_total_length_px"], errors="coerce").fillna(0.0)
    if {"lateral_total_length_mm", "lateral_count"}.issubset(out.columns):
        lateral_count = pd.to_numeric(out["lateral_count"], errors="coerce").replace(0.0, np.nan)
        out["lateral_mean_length_mm"] = (
            pd.to_numeric(out["lateral_total_length_mm"], errors="coerce").fillna(0.0) / lateral_count
        ).fillna(0.0)
        out["lateral_max_length_mm"] = pd.to_numeric(out["lateral_total_length_mm"], errors="coerce").fillna(0.0)

    out["mask_total_export_mode"] = "plate_total_as_single_pmi_entity"
    return out


def build_mask_total_summary_dataframe(mask_df: pd.DataFrame) -> pd.DataFrame:
    if mask_df is None or mask_df.empty:
        return pd.DataFrame()
    summary_source = mask_df.copy()
    if "total_root_pixels_selected_classes" not in summary_source.columns:
        fallback = "measured_mask_pixels" if "measured_mask_pixels" in summary_source.columns else "total_root_pixels_1_plus_3"
        if fallback in summary_source.columns:
            summary_source["total_root_pixels_selected_classes"] = summary_source[fallback]
        else:
            summary_source["total_root_pixels_selected_classes"] = 0
    if "primary_root_pixels_selected_class" not in summary_source.columns:
        summary_source["primary_root_pixels_selected_class"] = summary_source.get("root_pixels_class_1", 0)
    if "lateral_root_pixels_selected_class" not in summary_source.columns:
        summary_source["lateral_root_pixels_selected_class"] = summary_source.get("lateral_pixels_class_3", 0)
    if "primary_root_class_id" not in summary_source.columns:
        summary_source["primary_root_class_id"] = int(DEFAULT_MASK_TOTAL_CLASS_IDS[0])
    if "lateral_root_class_id" not in summary_source.columns:
        summary_source["lateral_root_class_id"] = int(DEFAULT_MASK_TOTAL_CLASS_IDS[1])
    for column in (
        "total_root_pixels_selected_classes",
        "primary_root_pixels_selected_class",
        "lateral_root_pixels_selected_class",
        "shoot_area_px",
        "shoot_area_mm2",
    ):
        if column not in summary_source.columns:
            summary_source[column] = 0
    for column in (
        "shoot_area_model_px",
        "shoot_area_model_mm2",
        "shoot_rgb_rescue_candidate_px",
        "shoot_rgb_rescue_added_px",
        "shoot_rgb_rescue_removed_model_px",
        "shoot_rgb_rescue_green_only_px",
        "shoot_rgb_rescue_components",
    ):
        if column not in summary_source.columns:
            summary_source[column] = 0
    if "shoot_rgb_rescue_applied" not in summary_source.columns:
        summary_source["shoot_rgb_rescue_applied"] = False
    if "shoot_measurement_source" not in summary_source.columns:
        summary_source["shoot_measurement_source"] = "mask_class"

    group_columns = ["Series", "PetriDish"] if "Series" in summary_source.columns else ["PetriDish"]
    summary = (
        summary_source.groupby(group_columns, dropna=False)
        .agg(
            frames=("FrameIndex", "count"),
            first_timestamp=("Timestamp", "first"),
            last_timestamp=("Timestamp", "last"),
            primary_root_class_id=("primary_root_class_id", "first"),
            lateral_root_class_id=("lateral_root_class_id", "first"),
            first_total_root_length_mm=("total_root_length_mm", "first"),
            final_total_root_length_mm=("total_root_length_mm", "last"),
            max_total_root_length_mm=("total_root_length_mm", "max"),
            final_total_root_length_weighted_mm=("total_root_length_weighted_mm", "last"),
            first_primary_root_length_mm=("primary_root_length_mm", "first"),
            final_primary_root_length_mm=("primary_root_length_mm", "last"),
            max_primary_root_length_mm=("primary_root_length_mm", "max"),
            final_primary_root_length_weighted_mm=("primary_root_length_weighted_mm", "last"),
            first_lateral_root_length_mm=("lateral_root_length_mm", "first"),
            final_lateral_root_length_mm=("lateral_root_length_mm", "last"),
            max_lateral_root_length_mm=("lateral_root_length_mm", "max"),
            final_lateral_root_length_weighted_mm=("lateral_root_length_weighted_mm", "last"),
            final_total_root_pixels_selected_classes=("total_root_pixels_selected_classes", "last"),
            final_primary_root_pixels_selected_class=("primary_root_pixels_selected_class", "last"),
            final_lateral_root_pixels_selected_class=("lateral_root_pixels_selected_class", "last"),
            frames_with_selected_root_pixels=(
                "total_root_pixels_selected_classes",
                lambda values: int((pd.to_numeric(values, errors="coerce").fillna(0) > 0).sum()),
            ),
            frames_with_primary_root_pixels=(
                "primary_root_pixels_selected_class",
                lambda values: int((pd.to_numeric(values, errors="coerce").fillna(0) > 0).sum()),
            ),
            frames_with_lateral_root_pixels=(
                "lateral_root_pixels_selected_class",
                lambda values: int((pd.to_numeric(values, errors="coerce").fillna(0) > 0).sum()),
            ),
            final_root_pixels_class_1=("root_pixels_class_1", "last"),
            final_lateral_pixels_class_3=("lateral_pixels_class_3", "last"),
            first_shoot_area_mm2=("shoot_area_mm2", "first"),
            final_shoot_area_mm2=("shoot_area_mm2", "last"),
            max_shoot_area_mm2=("shoot_area_mm2", "max"),
            final_shoot_area_px=("shoot_area_px", "last"),
            final_shoot_area_model_mm2=("shoot_area_model_mm2", "last"),
            final_shoot_area_model_px=("shoot_area_model_px", "last"),
            total_shoot_rgb_rescue_added_px=("shoot_rgb_rescue_added_px", "sum"),
            max_shoot_rgb_rescue_added_px=("shoot_rgb_rescue_added_px", "max"),
            total_shoot_rgb_rescue_removed_model_px=("shoot_rgb_rescue_removed_model_px", "sum"),
            max_shoot_rgb_rescue_green_only_px=("shoot_rgb_rescue_green_only_px", "max"),
            frames_with_shoot_rgb_rescue=("shoot_rgb_rescue_applied", "sum"),
            frames_with_shoot_model_pixels=(
                "shoot_area_model_px",
                lambda values: int((pd.to_numeric(values, errors="coerce").fillna(0) > 0).sum()),
            ),
            frames_with_final_shoot_pixels=(
                "shoot_area_px",
                lambda values: int((pd.to_numeric(values, errors="coerce").fillna(0) > 0).sum()),
            ),
            final_shoot_measurement_source=("shoot_measurement_source", "last"),
            final_shoot_pixels_class_2=("shoot_pixels_class_2", "last"),
            final_measured_mask_pixels=("measured_mask_pixels", "last"),
            final_component_count=("component_count", "last"),
            final_root_to_shoot_area_ratio_px=("root_to_shoot_area_ratio_px", "last"),
            final_root_length_mm_per_shoot_area_mm2=("root_length_mm_per_shoot_area_mm2", "last"),
        )
        .reset_index()
    )
    summary["delta_total_root_length_mm"] = summary["final_total_root_length_mm"] - summary["first_total_root_length_mm"]
    summary["delta_primary_root_length_mm"] = summary["final_primary_root_length_mm"] - summary["first_primary_root_length_mm"]
    summary["delta_lateral_root_length_mm"] = summary["final_lateral_root_length_mm"] - summary["first_lateral_root_length_mm"]
    summary["delta_shoot_area_mm2"] = summary["final_shoot_area_mm2"] - summary["first_shoot_area_mm2"]
    if "Series" in summary.columns:
        summary["_sort_series"] = summary["Series"].map(natural_sort_key)
    summary["_sort_petri"] = summary["PetriDish"].map(natural_sort_key)
    sort_columns = ["_sort_petri"]
    if "_sort_series" in summary.columns:
        sort_columns.insert(0, "_sort_series")
    summary.sort_values(sort_columns, inplace=True, kind="mergesort")
    summary.drop(columns=["_sort_series", "_sort_petri"], inplace=True, errors="ignore")
    summary.reset_index(drop=True, inplace=True)
    return summary


def _review_numeric(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _review_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def build_mask_total_frame_review_dataframe(mask_df: pd.DataFrame | None) -> pd.DataFrame:
    """Return non-ok mask-total frames that deserve visual review."""
    if mask_df is None or mask_df.empty:
        return pd.DataFrame(columns=MASK_TOTAL_FRAME_REVIEW_COLUMNS)

    src = mask_df.copy()
    if "Series" not in src.columns:
        src["Series"] = "Unknown"
    if "PetriDish" not in src.columns:
        src["PetriDish"] = "Unknown"
    if "Timestamp" in src.columns:
        src["_review_timestamp"] = pd.to_datetime(src["Timestamp"], errors="coerce")
    else:
        src["_review_timestamp"] = pd.NaT
    if "FrameIndex" in src.columns:
        src["_review_frame"] = pd.to_numeric(src["FrameIndex"], errors="coerce").fillna(1.0e12)
    else:
        src["_review_frame"] = np.arange(len(src), dtype=np.float64)
        src["FrameIndex"] = np.arange(len(src), dtype=np.int64)
    src["_review_series"] = src["Series"].fillna("Unknown").astype(str).map(natural_sort_key)
    src["_review_petri"] = src["PetriDish"].fillna("Unknown").astype(str).map(natural_sort_key)
    src.sort_values(
        ["_review_series", "_review_petri", "_review_timestamp", "_review_frame"],
        inplace=True,
        kind="mergesort",
    )

    if "total_root_length_mm" in src.columns:
        root_lengths = pd.to_numeric(src["total_root_length_mm"], errors="coerce")
        src["_review_prev_total_root_length_mm"] = root_lengths.groupby([src["Series"], src["PetriDish"]]).shift(1)
    else:
        src["_review_prev_total_root_length_mm"] = np.nan

    rows: list[dict[str, object]] = []
    for _idx, row in src.iterrows():
        reasons: list[str] = []
        severity = 0

        selected_root_px = _review_numeric(
            row.get("total_root_pixels_selected_classes", row.get("measured_mask_pixels", 0)),
            default=0.0,
        )
        total_root_length = _review_numeric(row.get("total_root_length_mm", 0.0), default=0.0)
        if selected_root_px <= 0 or total_root_length <= 0:
            severity = max(severity, 2)
            reasons.append("no selected root pixels or zero total root length")

        prev_total_root_length = _review_numeric(row.get("_review_prev_total_root_length_mm", np.nan), default=np.nan)
        delta_total_root_length = _review_numeric(row.get("delta_total_root_length_mm", np.nan), default=np.nan)
        if not math.isfinite(delta_total_root_length) and math.isfinite(prev_total_root_length):
            delta_total_root_length = float(total_root_length - prev_total_root_length)
        root_drop_fraction = 0.0
        if math.isfinite(prev_total_root_length) and prev_total_root_length > 0 and math.isfinite(delta_total_root_length):
            root_drop_fraction = float(abs(min(0.0, delta_total_root_length)) / max(1.0e-9, prev_total_root_length))
            if delta_total_root_length < -max(20.0, 0.25 * prev_total_root_length):
                severity = max(severity, 2)
                reasons.append("large negative total-root length drop")
            elif delta_total_root_length < -max(8.0, 0.12 * prev_total_root_length):
                severity = max(severity, 1)
                reasons.append("moderate negative total-root length drop")

        source = str(row.get("shoot_measurement_source", "mask_class") or "mask_class").strip().lower()
        green_only = _review_bool(row.get("shoot_rgb_rescue_enabled", False)) or source == "rgb_green_only"
        shoot_px = _review_numeric(row.get("shoot_area_px", 0), default=0.0)
        shoot_model_px = _review_numeric(row.get("shoot_area_model_px", row.get("shoot_pixels_selected_classes", 0)), default=0.0)
        removed_model_px = _review_numeric(row.get("shoot_rgb_rescue_removed_model_px", 0), default=0.0)
        green_px = _review_numeric(row.get("shoot_rgb_rescue_green_only_px", shoot_px), default=0.0)
        components = _review_numeric(row.get("shoot_rgb_rescue_components", 0), default=0.0)
        rescue_abstained = _review_bool(row.get("shoot_rgb_rescue_abstained", False))
        rejected_fraction = float(removed_model_px / shoot_model_px) if shoot_model_px > 0 else 0.0
        green_fraction = float(green_px / shoot_model_px) if shoot_model_px > 0 else 0.0

        if green_only:
            if rescue_abstained:
                severity = max(severity, 1)
                reasons.append("RGB crown-green detector abstained; retained model shoot mask")
            if shoot_model_px > 0 and removed_model_px >= shoot_model_px and shoot_px <= 0:
                severity = max(severity, 1)
                reasons.append("model shoot pixels rejected as non-green and no green shoot pixels remained")
            if shoot_px <= 0:
                severity = max(severity, 1)
                reasons.append("no RGB green-only shoot pixels detected")
            if components >= 18:
                severity = max(severity, 1)
                reasons.append("many disconnected RGB green shoot components")
        elif shoot_px <= 0:
            severity = max(severity, 1)
            reasons.append("no shoot pixels detected")

        if severity <= 0:
            continue

        out_row = {
            "Series": row.get("Series", "Unknown"),
            "PetriDish": row.get("PetriDish", "Unknown"),
            "Timestamp": row.get("Timestamp", ""),
            "FrameIndex": row.get("FrameIndex", ""),
            "SourceFile": row.get("SourceFile", ""),
            "OutputMaskPath": row.get("OutputMaskPath", ""),
            "PreviewImagePath": row.get("PreviewImagePath", ""),
            MASK_TOTAL_FRAME_REVIEW_STATUS_COLUMN: "fail" if severity >= 2 else "review",
            MASK_TOTAL_FRAME_REVIEW_REASON_COLUMN: "; ".join(dict.fromkeys(reasons)),
            "total_root_length_mm": total_root_length,
            "delta_total_root_length_mm": delta_total_root_length if math.isfinite(delta_total_root_length) else "",
            "root_length_drop_fraction": root_drop_fraction,
            "primary_root_length_mm": _review_numeric(row.get("primary_root_length_mm", 0), default=0.0),
            "lateral_root_length_mm": _review_numeric(row.get("lateral_root_length_mm", 0), default=0.0),
            "total_root_pixels_selected_classes": int(round(selected_root_px)),
            "primary_root_pixels_selected_class": int(round(_review_numeric(row.get("primary_root_pixels_selected_class", 0), default=0.0))),
            "lateral_root_pixels_selected_class": int(round(_review_numeric(row.get("lateral_root_pixels_selected_class", 0), default=0.0))),
            "shoot_measurement_source": source or "mask_class",
            "shoot_rgb_rescue_enabled": bool(green_only),
            "shoot_area_px": int(round(shoot_px)),
            "shoot_area_model_px": int(round(shoot_model_px)),
            "shoot_rgb_rescue_removed_model_px": int(round(removed_model_px)),
            "shoot_rgb_rescue_green_only_px": int(round(green_px)),
            "shoot_rgb_rescue_abstained": bool(rescue_abstained),
            "shoot_rgb_rescue_fallback_status": row.get("shoot_rgb_rescue_fallback_status", ""),
            "shoot_model_rejected_fraction": rejected_fraction,
            "shoot_green_fraction_of_model": green_fraction,
            "shoot_rgb_rescue_components": int(round(components)),
        }
        rows.append(out_row)

    if not rows:
        return pd.DataFrame(columns=MASK_TOTAL_FRAME_REVIEW_COLUMNS)
    out = pd.DataFrame(rows)
    for column in MASK_TOTAL_FRAME_REVIEW_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    return out[MASK_TOTAL_FRAME_REVIEW_COLUMNS]


def write_mask_total_frame_review_csv(mask_df: pd.DataFrame | None, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_review_path = output_dir / MASK_TOTAL_FRAME_REVIEW_FILENAME
    build_mask_total_frame_review_dataframe(mask_df).to_csv(frame_review_path, index=False)
    return frame_review_path


def _metadata_rows(metadata: dict[str, object]) -> pd.DataFrame:
    rows = []
    for key in sorted(metadata.keys()):
        value = metadata.get(key)
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, sort_keys=True)
        rows.append({"key": key, "value": value})
    return pd.DataFrame(rows, columns=["key", "value"])


def _metadata_column_int_sum(df: pd.DataFrame, column: str) -> int:
    if df is None or column not in df.columns:
        return 0
    return int(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def _metadata_column_positive_frame_count(df: pd.DataFrame, column: str) -> int:
    if df is None or column not in df.columns:
        return 0
    values = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return int((values > 0).sum())


def _mask_total_class_coverage_metadata(
    mask_df: pd.DataFrame,
    root_class_ids: Iterable[int],
    shoot_class_ids: Iterable[int],
) -> dict[str, object]:
    frames = int(len(mask_df)) if mask_df is not None else 0

    def _class_stats(class_ids: Iterable[int], *, prefix: str) -> tuple[dict[str, int], dict[str, int], list[int]]:
        total_pixels: dict[str, int] = {}
        frame_counts: dict[str, int] = {}
        missing: list[int] = []
        for raw_class_id in class_ids:
            class_id = int(raw_class_id)
            if class_id <= 0:
                continue
            column = f"shoot_class_{class_id}_pixels" if prefix == "shoot" else f"{_class_metric_prefix(class_id)}_pixels"
            total = _metadata_column_int_sum(mask_df, column)
            count = _metadata_column_positive_frame_count(mask_df, column)
            total_pixels[str(class_id)] = int(total)
            frame_counts[str(class_id)] = int(count)
            if total <= 0:
                missing.append(int(class_id))
        return total_pixels, frame_counts, missing

    root_total_pixels, root_frame_counts, missing_root_ids = _class_stats(root_class_ids, prefix="root")
    shoot_total_pixels, shoot_frame_counts, missing_shoot_ids = _class_stats(shoot_class_ids, prefix="shoot")
    warnings: list[str] = []
    if missing_root_ids:
        warnings.append("Selected root classes absent from all masks: " + ",".join(str(v) for v in missing_root_ids))
    if missing_shoot_ids:
        warnings.append("Selected shoot classes absent from all masks: " + ",".join(str(v) for v in missing_shoot_ids))

    return {
        "frames_measured": int(frames),
        "root_class_total_pixels": root_total_pixels,
        "root_class_frame_counts": root_frame_counts,
        "missing_root_class_ids": missing_root_ids,
        "shoot_class_total_pixels": shoot_total_pixels,
        "shoot_class_frame_counts": shoot_frame_counts,
        "missing_shoot_class_ids": missing_shoot_ids,
        "frames_missing_selected_root_pixels": int(frames - _metadata_column_positive_frame_count(mask_df, "total_root_pixels_selected_classes")),
        "frames_missing_primary_root_pixels": int(frames - _metadata_column_positive_frame_count(mask_df, "primary_root_pixels_selected_class")),
        "frames_missing_lateral_root_pixels": int(frames - _metadata_column_positive_frame_count(mask_df, "lateral_root_pixels_selected_class")),
        "frames_missing_model_shoot_pixels": int(frames - _metadata_column_positive_frame_count(mask_df, "shoot_area_model_px")),
        "frames_missing_final_shoot_pixels": int(frames - _metadata_column_positive_frame_count(mask_df, "shoot_area_px")),
        "warnings": warnings,
    }


def build_mask_total_class_coverage_metadata(
    mask_df: pd.DataFrame,
    root_class_ids: Iterable[int],
    shoot_class_ids: Iterable[int],
) -> dict[str, object]:
    return _mask_total_class_coverage_metadata(mask_df, root_class_ids, shoot_class_ids)


def write_mask_total_all_metrics_workbook(
    mask_df: pd.DataFrame,
    output_dir: Path,
    *,
    class_ids: Iterable[int],
    shoot_class_ids: Iterable[int] = DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS,
    root_metric_mode: object = "total",
    pmi_df: pd.DataFrame | None = None,
    metadata_json_path: Path | None = None,
) -> Path | None:
    if mask_df is None or mask_df.empty:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = output_dir / "npec_total_root_and_shoot_all_metrics.xlsx"
    resolved_class_ids = _resolve_class_ids(class_ids, DEFAULT_MASK_TOTAL_CLASS_IDS)
    resolved_shoot_class_ids = _resolve_class_ids(shoot_class_ids, DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS)
    primary_class_id, lateral_class_id = _primary_lateral_class_ids(resolved_class_ids)
    timelapse_df = build_mask_total_timelapse_dataframe(mask_df, root_metric_mode=root_metric_mode)
    summary_df = build_mask_total_summary_dataframe(mask_df)
    frame_review_df = build_mask_total_frame_review_dataframe(mask_df)
    class_coverage = _mask_total_class_coverage_metadata(mask_df, resolved_class_ids, resolved_shoot_class_ids)
    metadata = {
        "analysis_mode": "mask_total",
        "frames_measured": int(len(mask_df)),
        "petri_dishes": int(mask_df["PetriDish"].nunique()) if "PetriDish" in mask_df.columns else 0,
        "measurement_class_ids": [int(v) for v in resolved_class_ids],
        "shoot_class_ids": [int(v) for v in resolved_shoot_class_ids],
        "primary_root_class_id": int(primary_class_id),
        "lateral_root_class_id": int(lateral_class_id) if lateral_class_id > 0 else None,
        "selected_root_metric_mode": _normalize_root_metric_mode(root_metric_mode),
        "selected_root_metric_column": MASK_TOTAL_ROOT_METRIC_COLUMNS[_normalize_root_metric_mode(root_metric_mode)],
        "shoot_rgb_rescue_enabled": bool(mask_df.get("shoot_rgb_rescue_enabled", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
        "frames_with_shoot_rgb_rescue": _metadata_column_int_sum(mask_df, "shoot_rgb_rescue_applied"),
        "total_shoot_rgb_rescue_added_px": _metadata_column_int_sum(mask_df, "shoot_rgb_rescue_added_px"),
        "total_shoot_rgb_rescue_removed_model_px": _metadata_column_int_sum(mask_df, "shoot_rgb_rescue_removed_model_px"),
        "total_shoot_rgb_rescue_green_only_px": _metadata_column_int_sum(mask_df, "shoot_rgb_rescue_green_only_px"),
        "frames_needing_review": int(len(frame_review_df)),
        "frames_failed_review": int(
            (frame_review_df[MASK_TOTAL_FRAME_REVIEW_STATUS_COLUMN].fillna("").astype(str).str.lower() == "fail").sum()
        )
        if MASK_TOTAL_FRAME_REVIEW_STATUS_COLUMN in frame_review_df.columns
        else 0,
        "frame_review_csv": str(output_dir / MASK_TOTAL_FRAME_REVIEW_FILENAME),
        "frame_review_definition": (
            "Mask-total frame review flags missing selected roots, negative total-root length jumps, missing shoots, "
            "and frames where RGB green-only shoot measurement rejected most model shoot pixels as non-green."
        ),
        "class_coverage": class_coverage,
        "class_coverage_warnings": class_coverage.get("warnings", []),
    }
    try:
        frame_review_df.to_csv(output_dir / MASK_TOTAL_FRAME_REVIEW_FILENAME, index=False)
        with pd.ExcelWriter(workbook_path) as writer:
            mask_df.to_excel(writer, index=False, sheet_name="detail")
            summary_df.to_excel(writer, index=False, sheet_name="summary")
            timelapse_df.to_excel(writer, index=False, sheet_name="timelapse")
            frame_review_df.to_excel(writer, index=False, sheet_name="mask_total_frame_review")
            _metadata_rows(metadata).to_excel(writer, index=False, sheet_name="run_config")
            if pmi_df is not None and not pmi_df.empty:
                pmi_df.to_excel(writer, index=False, sheet_name="pmi_style")
    except Exception:
        try:
            workbook_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    if metadata_json_path is not None:
        try:
            path = Path(metadata_json_path)
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            if not isinstance(payload, dict):
                payload = {}
            payload["all_metrics_workbook"] = str(workbook_path)
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
    return workbook_path


def write_mask_total_outputs(
    mask_df: pd.DataFrame,
    output_dir: Path,
    *,
    class_ids: Iterable[int],
    shoot_class_ids: Iterable[int] = DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS,
    root_metric_mode: object = "total",
    write_compat_timelapse: bool = True,
) -> tuple[Path, Path, Path | None, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = output_dir / "npec_total_root_and_shoot_from_masks.csv"
    summary_csv = output_dir / "npec_total_root_and_shoot_from_masks_summary.csv"
    timelapse_csv = output_dir / "npec_total_root_and_shoot_timelapse.csv"
    frame_review_csv = output_dir / MASK_TOTAL_FRAME_REVIEW_FILENAME
    compat_csv = output_dir / "npec_total_root_length_timelapse.csv" if write_compat_timelapse else None
    metadata_json = output_dir / "npec_total_root_and_shoot_from_masks_metadata.json"
    legacy_detail_csv = output_dir / "npec_total_root_length_from_masks.csv"
    legacy_summary_csv = output_dir / "npec_total_root_length_from_masks_summary.csv"
    legacy_metadata_json = output_dir / "npec_total_root_length_from_masks_metadata.json"
    resolved_class_ids = _resolve_class_ids(class_ids, DEFAULT_MASK_TOTAL_CLASS_IDS)
    resolved_shoot_class_ids = _resolve_class_ids(shoot_class_ids, DEFAULT_MASK_TOTAL_SHOOT_CLASS_IDS)
    primary_class_id, lateral_class_id = _primary_lateral_class_ids(resolved_class_ids)
    class_coverage = _mask_total_class_coverage_metadata(mask_df, resolved_class_ids, resolved_shoot_class_ids)

    timelapse_df = build_mask_total_timelapse_dataframe(mask_df, root_metric_mode=root_metric_mode)
    frame_review_df = build_mask_total_frame_review_dataframe(mask_df)
    mask_df.to_csv(detail_csv, index=False)
    mask_df.to_csv(legacy_detail_csv, index=False)
    timelapse_df.to_csv(timelapse_csv, index=False)
    frame_review_df.to_csv(frame_review_csv, index=False)
    group_columns = ["Series", "PetriDish"] if "Series" in mask_df.columns else ["PetriDish"]
    summary = build_mask_total_summary_dataframe(mask_df)
    summary.to_csv(summary_csv, index=False)
    summary.to_csv(legacy_summary_csv, index=False)

    if compat_csv is not None:
        timelapse_df.to_csv(compat_csv, index=False)

    metadata = {
        "analysis_mode": "mask_total",
        "output_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
        "timelapse_csv": str(timelapse_csv),
        "frame_review_csv": str(frame_review_csv),
        "compat_timelapse_csv": str(compat_csv) if compat_csv is not None else None,
        "legacy_output_csv": str(legacy_detail_csv),
        "legacy_summary_csv": str(legacy_summary_csv),
        "legacy_metadata_json": str(legacy_metadata_json),
        "frames_measured": int(len(mask_df)),
        "petri_dishes": int(mask_df["PetriDish"].nunique()) if "PetriDish" in mask_df.columns else 0,
        "series_petri_groups": (
            int(mask_df[group_columns].drop_duplicates().shape[0])
            if all(column in mask_df.columns for column in group_columns)
            else 0
        ),
        "measurement_class_ids": [int(v) for v in resolved_class_ids],
        "shoot_class_ids": [int(v) for v in resolved_shoot_class_ids],
        "primary_root_class_id": int(primary_class_id),
        "lateral_root_class_id": int(lateral_class_id) if lateral_class_id > 0 else None,
        "selected_root_metric_mode": _normalize_root_metric_mode(root_metric_mode),
        "selected_root_metric_column": MASK_TOTAL_ROOT_METRIC_COLUMNS[_normalize_root_metric_mode(root_metric_mode)],
        "shoot_class_id": int(_first_shoot_class_id(mask_df, shoot_class_ids=resolved_shoot_class_ids)),
        "shoot_rgb_rescue_enabled": bool(mask_df.get("shoot_rgb_rescue_enabled", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
        "frames_with_shoot_rgb_rescue": _metadata_column_int_sum(mask_df, "shoot_rgb_rescue_applied"),
        "total_shoot_rgb_rescue_added_px": _metadata_column_int_sum(mask_df, "shoot_rgb_rescue_added_px"),
        "total_shoot_rgb_rescue_removed_model_px": _metadata_column_int_sum(mask_df, "shoot_rgb_rescue_removed_model_px"),
        "total_shoot_rgb_rescue_green_only_px": _metadata_column_int_sum(mask_df, "shoot_rgb_rescue_green_only_px"),
        "frames_needing_review": int(len(frame_review_df)),
        "frames_failed_review": int(
            (frame_review_df[MASK_TOTAL_FRAME_REVIEW_STATUS_COLUMN].fillna("").astype(str).str.lower() == "fail").sum()
        )
        if MASK_TOTAL_FRAME_REVIEW_STATUS_COLUMN in frame_review_df.columns
        else 0,
        "frame_review_definition": (
            "Mask-total frame review flags missing selected roots, negative total-root length jumps, missing shoots, "
            "and frames where RGB green-only shoot measurement rejected most model shoot pixels as non-green."
        ),
        "class_coverage": class_coverage,
        "class_coverage_warnings": class_coverage.get("warnings", []),
        "per_selected_class_columns": [
            _class_metric_prefix(int(v))
            for v in resolved_class_ids
            if int(v) > 0
        ],
        "length_definition": (
            "total_root_length_px is the skeleton pixel count over the combined requested mask classes; "
            "primary_root_length_px is measured from the first requested root class; "
            "lateral_root_length_px is measured from the second requested root class when present; "
            "root_pixels_class_1 and lateral_pixels_class_3 remain literal legacy class-count columns; "
            "the *_weighted_px columns are 8-neighbor skeleton edge lengths."
        ),
        "shoot_definition": (
            "shoot_area_mm2 is the final measured shoot area multiplied by pixel_size_mm squared. "
            "When RGB-green shoot rescue is enabled, shoot_area_model_* preserves the class-only model area, "
            "shoot_area_* is measured from strict green RGB pixels only, and "
            "shoot_rgb_rescue_removed_model_px records non-green model shoot pixels ignored by measurement."
        ),
    }
    payload = json.dumps(metadata, indent=2) + "\n"
    metadata_json.write_text(payload, encoding="utf-8")
    legacy_metadata_json.write_text(payload, encoding="utf-8")
    return detail_csv, summary_csv, compat_csv, metadata_json
