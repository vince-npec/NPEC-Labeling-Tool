from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Callable, Iterable, Mapping, Sequence

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.analytics_engine import (  # noqa: E402
    _path_length_px,
    _skeleton_edge_length_px,
    _skeleton_graph_path,
    _skeletonize,
)
from resources.npec_pyphenotyper.pyphenotyper.data.data_processing import (  # noqa: E402
    collect_tiles,
    stitch_predictions,
)
from resources.pyphenotyper_adapter import _load_keras_model_cached  # noqa: E402
from resources.plate_identity import (  # noqa: E402
    acquisition_datetime,
    read_plate_identity_manifest,
    sanitize_plate_id,
)


PIPELINE_VERSION = "npec-potato-endpoint-v5"
LUCIFER_PIXEL_SIZE_MM = 111.88 / 4200.0
RAW_CROP_BOX = (1728, 720, 5895, 4855)
EXPECTED_CROP_SIZE = (4167, 4135)
ROOT_THRESHOLD = 0.60
ROOT_GRAPH_THRESHOLD = 0.80
MODEL_MASK_SOURCE = "dario_finetuned_root_plus_green_shoot"
FORCED_MASK_SOURCE = "forced_crown_anchored_frangi_rescue"
POTATO_MODEL_DIR = REPO_ROOT / "resources/builtin_models/potato"
DEFAULT_ROOT_MODEL = POTATO_MODEL_DIR / "best_potato_root_june_10_model_patch_256_max_f10.815_max_IoU0.915.h5"
DEFAULT_ROOT_WEIGHTS = POTATO_MODEL_DIR / "dario_potato_root_final_all16.weights.h5"
DEFAULT_VALIDATION = POTATO_MODEL_DIR / "npec_potato_reference_validation.json"

CLASS_COLORS = {
    "primary_root": (255, 142, 43),
    "lateral_root": (76, 141, 255),
    "adventitious_root": (28, 210, 220),
    "shoot": (236, 55, 125),
}
CLASS_IDS = {
    "primary_root": 1,
    "shoot": 2,
    "lateral_root": 3,
    "adventitious_root": 4,
}

KNOWN_NON_PLANT_CAPTURES: dict[str, str] = {}

PSD_LAYER_NAMES = {
    "shoot": "shoot",
    "shott": "shoot",
    "primary_root": "primary_root",
    "primay_root": "primary_root",
    "lateral_root": "lateral_root",
    "adventitious_root": "adventitious_root",
    "adventitious_roots": "adventitious_root",
    "adventituos_root": "adventitious_root",
    "adventitous_root": "adventitious_root",
    "seed_coat": "seed_coat",
}


# Dataset-specific hand traces are deliberately not part of the public engine.
FORCED_VESSEL_RESCUES: dict[str, dict[str, object]] = {}


@dataclass(slots=True)
class ManualReference:
    image_rgb: np.ndarray
    masks: dict[str, np.ndarray]
    psd_path: Path


class PotatoEndpointCancelled(RuntimeError):
    """Raised when an interactive endpoint-package run is canceled."""


def _parse_args() -> argparse.Namespace:
    default_dataset = Path.cwd()
    parser = argparse.ArgumentParser(
        description="Phenotype the single-time-point Lucifer potato dataset."
    )
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dataset / "npec_potato_endpoint_analysis_2026-07-20",
    )
    parser.add_argument(
        "--root-weights",
        type=Path,
        default=DEFAULT_ROOT_WEIGHTS,
    )
    parser.add_argument("--root-model", type=Path, default=DEFAULT_ROOT_MODEL)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Folder containing endpoint images. Defaults to <dataset>/raw when present.",
    )
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument("--identity-manifest", type=Path, default=None)
    parser.add_argument(
        "--generic-dataset",
        action="store_true",
        default=True,
        help="Use the generic endpoint workflow without private dataset-specific rules.",
    )
    parser.add_argument("--pixel-size-mm", type=float, default=LUCIFER_PIXEL_SIZE_MM)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--stems",
        type=str,
        default="",
        help="Optional comma-separated full filename stems for a targeted QA run.",
    )
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--no-manual-labels", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_psd_stem(path: Path) -> str:
    return re.sub(r"_recheck$", "", path.stem, flags=re.IGNORECASE)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
        if bold
        else Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Helvetica.ttc"),
    )
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default(size=max(10, size))


def _load_manual_references(
    labels_dir: Path,
    selected_stems: set[str] | None = None,
) -> dict[str, ManualReference]:
    try:
        from psd_tools import PSDImage
    except Exception as exc:
        raise RuntimeError(
            "psd-tools is required to consume layered reference labels."
        ) from exc

    references: dict[str, ManualReference] = {}
    for psd_path in sorted(labels_dir.glob("*.psd")):
        normalized_stem = _normalized_psd_stem(psd_path)
        if selected_stems is not None and normalized_stem not in selected_stems:
            continue
        psd = PSDImage.open(psd_path)
        background = psd[0].composite(
            viewport=psd.viewbox,
            force=True,
            color=1.0,
            alpha=0.0,
        )
        if background is None:
            continue
        image_rgb = np.asarray(background.convert("RGB"), dtype=np.uint8)
        masks: dict[str, np.ndarray] = {}
        for layer in psd[1:]:
            normalized = re.sub(r"\s+", "_", str(layer.name).strip().lower())
            class_name = PSD_LAYER_NAMES.get(normalized)
            if class_name is None:
                continue
            composite = layer.composite(viewport=psd.viewbox, force=True)
            if composite is None or "A" not in composite.getbands():
                continue
            alpha = np.asarray(composite.getchannel("A"), dtype=np.uint8)
            mask = (alpha > 0).astype(np.uint8)
            current = masks.get(class_name)
            masks[class_name] = mask if current is None else np.maximum(current, mask)
        references[normalized_stem] = ManualReference(
            image_rgb=image_rgb,
            masks=masks,
            psd_path=psd_path,
        )
    return references


def _load_normalized_crop(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with Image.open(path) as source:
        exif_orientation = int(source.getexif().get(274, 1) or 1)
        if source.size[0] < RAW_CROP_BOX[2] or source.size[1] < RAW_CROP_BOX[3]:
            raise ValueError(f"Unexpected source size {source.size} for {path.name}")
        crop = source.convert("RGB").crop(RAW_CROP_BOX).transpose(Image.Transpose.ROTATE_180)
    if crop.size != EXPECTED_CROP_SIZE:
        crop = crop.resize(EXPECTED_CROP_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(crop, dtype=np.uint8), {
        "source_exif_orientation": exif_orientation,
        "raw_crop_box": list(RAW_CROP_BOX),
        "normalized_rotation_deg": 180,
        "normalized_size": list(crop.size),
    }


def _forced_rescue_spec(stem: str) -> dict[str, object] | None:
    return FORCED_VESSEL_RESCUES.get(stem)


def _forced_rescue_spec_sha256(spec: dict[str, object]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _forced_crown_anchored_rescue(
    image_rgb: np.ndarray,
    shoot_mask: np.ndarray,
    spec: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object], list[dict[str, object]]]:
    from scipy.interpolate import PchipInterpolator
    from skimage.filters import frangi
    from skimage.graph import route_through_array

    image = np.asarray(image_rgb, dtype=np.uint8)
    height, width = image.shape[:2]
    expected_size = [width, height]
    if list(spec.get("normalized_size", [])) != expected_size:
        raise ValueError(
            f"Forced rescue expects {spec.get('normalized_size')}, got {expected_size}."
        )

    primary_controls = np.asarray(spec["primary_root"], dtype=np.float64)
    if len(primary_controls) < 2 or np.any(np.diff(primary_controls[:, 1]) <= 0):
        raise ValueError("Forced primary-root controls must have strictly increasing y coordinates.")
    dense_y = np.arange(
        int(primary_controls[0, 1]),
        int(primary_controls[-1, 1]) + 1,
        dtype=np.int32,
    )
    dense_x = PchipInterpolator(primary_controls[:, 1], primary_controls[:, 0])(dense_y)
    primary_path = np.column_stack((np.rint(dense_x).astype(np.int32), dense_y))

    primary = np.zeros((height, width), dtype=np.uint8)
    lateral = np.zeros_like(primary)
    adventitious = np.zeros_like(primary)
    cv2.polylines(
        primary,
        [primary_path.reshape((-1, 1, 2))],
        False,
        1,
        thickness=9,
        lineType=cv2.LINE_AA,
    )
    primary = (primary > 0).astype(np.uint8)

    roi_x0, roi_y0, roi_x1, roi_y1 = (int(value) for value in spec["vessel_roi"])
    if not (0 <= roi_x0 < roi_x1 <= width and 0 <= roi_y0 < roi_y1 <= height):
        raise ValueError("Forced vesselness ROI is outside the normalized image.")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    vesselness = frangi(
        gray[roi_y0:roi_y1, roi_x0:roi_x1].astype(np.float64) / 255.0,
        sigmas=(1, 2, 3, 4),
        black_ridges=True,
    ).astype(np.float32)

    def snap_to_vessel(point: list[int], radius: int = 8) -> tuple[int, int]:
        local_x = int(round(point[0] - roi_x0))
        local_y = int(round(point[1] - roi_y0))
        x_start = max(0, local_x - radius)
        x_stop = min(vesselness.shape[1], local_x + radius + 1)
        y_start = max(0, local_y - radius)
        y_stop = min(vesselness.shape[0], local_y + radius + 1)
        patch = vesselness[y_start:y_stop, x_start:x_stop]
        if patch.size <= 0:
            return local_x, local_y
        patch_y, patch_x = np.unravel_index(int(np.argmax(patch)), patch.shape)
        return x_start + int(patch_x), y_start + int(patch_y)

    def route_segment(start: list[int], stop: list[int]) -> np.ndarray:
        start_x, start_y = snap_to_vessel(start)
        stop_x, stop_y = snap_to_vessel(stop)
        margin = 55
        x_start = max(0, min(start_x, stop_x) - margin)
        x_stop = min(vesselness.shape[1], max(start_x, stop_x) + margin + 1)
        y_start = max(0, min(start_y, stop_y) - margin)
        y_stop = min(vesselness.shape[0], max(start_y, stop_y) + margin + 1)
        response = vesselness[y_start:y_stop, x_start:x_stop]
        guide = np.zeros(response.shape, dtype=np.uint8)
        cv2.line(
            guide,
            (start_x - x_start, start_y - y_start),
            (stop_x - x_start, stop_y - y_start),
            1,
            1,
            cv2.LINE_AA,
        )
        distance = cv2.distanceTransform((guide == 0).astype(np.uint8), cv2.DIST_L2, 3)
        scale = float(np.percentile(response, 99.9)) if np.any(response) else 1.0
        normalized = np.clip(response / max(scale, 1.0e-8), 0.0, 1.0)
        cost = 1.0 + (16.0 * (1.0 - normalized)) + (0.20 * np.minimum(distance, 80.0))
        route, _weight = route_through_array(
            cost,
            (start_y - y_start, start_x - x_start),
            (stop_y - y_start, stop_x - x_start),
            fully_connected=True,
            geometric=True,
        )
        return np.asarray(
            [
                (column + x_start + roi_x0, row + y_start + roi_y0)
                for row, column in route
            ],
            dtype=np.int32,
        )

    def draw_routes(mask: np.ndarray, routes: object) -> list[np.ndarray]:
        routed_paths: list[np.ndarray] = []
        for raw_controls in routes:
            controls = [list(map(int, point)) for point in raw_controls]
            routed: list[list[int]] = [controls[0]]
            for start, stop in zip(controls[:-1], controls[1:]):
                segment = route_segment(start, stop)
                if routed and len(segment):
                    segment = segment[1:]
                routed.extend(segment.tolist())
            if len(routed) > 1:
                routed_path = np.asarray(routed, dtype=np.int32)
                cv2.polylines(
                    mask,
                    [routed_path.reshape((-1, 1, 2))],
                    False,
                    1,
                    thickness=6,
                    lineType=cv2.LINE_AA,
                )
                routed_paths.append(routed_path)
        mask[:] = (mask > 0).astype(np.uint8)
        return routed_paths

    lateral_paths = draw_routes(lateral, spec["lateral_root"])
    adventitious_paths = draw_routes(adventitious, spec["adventitious_root"])
    lateral[primary > 0] = 0
    adventitious[np.logical_or(primary > 0, lateral > 0)] = 0
    shoot = (np.asarray(shoot_mask, dtype=np.uint8) > 0).astype(np.uint8)
    root_union = np.maximum.reduce((primary, lateral, adventitious))
    shoot[cv2.dilate(root_union, np.ones((9, 9), np.uint8), iterations=1) > 0] = 0

    masks = {
        "primary_root": primary,
        "lateral_root": lateral,
        "adventitious_root": adventitious,
        "shoot": shoot,
    }
    root_skeleton = _skeletonize(root_union)
    neighbor_count = (
        cv2.filter2D(root_skeleton.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
        - root_skeleton.astype(np.uint8)
    )
    route_xy = [(int(x), int(y)) for x, y in primary_path]
    route_length = float(_path_length_px(route_xy))
    branch_pixels = ((root_skeleton > 0) & (neighbor_count >= 3)).astype(np.uint8)
    graph_meta: dict[str, object] = {
        "crown_x_px": int(primary_path[0, 0]),
        "crown_y_px": int(primary_path[0, 1]),
        "primary_route_gap_length_px": 0.0,
        "primary_detected_seed_length_px": route_length,
        "primary_path_length_px": route_length,
        "primary_base_x_px": int(primary_path[0, 0]),
        "primary_base_y_px": int(primary_path[0, 1]),
        "primary_tip_x_px": int(primary_path[-1, 0]),
        "primary_tip_y_px": int(primary_path[-1, 1]),
        "primary_path_points": len(primary_path),
        "root_graph_tips_raw": int(np.count_nonzero((root_skeleton > 0) & (neighbor_count == 1))),
        "root_graph_branches_raw": max(
            0,
            int(cv2.connectedComponents(branch_pixels, connectivity=8)[0]) - 1,
        ),
        "high_confidence_root_area_px": int(np.count_nonzero(root_union)),
        "decomposition_mode": "reviewed_crown_anchored_frangi_routes",
        "root_components_before_topology_filter": max(
            0,
            int(cv2.connectedComponents(root_union, connectivity=8)[0]) - 1,
        ),
        "root_components_after_topology_filter": max(
            0,
            int(cv2.connectedComponents(root_union, connectivity=8)[0]) - 1,
        ),
        "root_pixels_before_topology_filter": int(np.count_nonzero(root_union)),
        "root_pixels_removed_by_topology_filter": 0,
        "topology_filter_applied": False,
        "topology_filter_trigger": "",
        "prefilter_root_graph_tips_raw": 0,
        "prefilter_primary_route_gap_length_px": 0.0,
        "rescue_applied": True,
        "rescue_method": str(spec["version"]),
        "rescue_spec_sha256": _forced_rescue_spec_sha256(spec),
        "rescue_normalized_size": f"{width}x{height}",
    }
    branch_rows: list[dict[str, object]] = []
    for branch_class, paths in (
        ("lateral_root", lateral_paths),
        ("adventitious_root", adventitious_paths),
    ):
        for path in paths:
            start = path[0]
            distances = np.hypot(
                primary_path[:, 0] - int(start[0]),
                primary_path[:, 1] - int(start[1]),
            )
            attachment_index = int(np.argmin(distances))
            attachment = primary_path[attachment_index]
            tip = path[-1]
            length_px = float(_path_length_px([(int(x), int(y)) for x, y in path]))
            branch_rows.append(
                {
                    "branch_class": branch_class,
                    "attachment_x_px": int(attachment[0]),
                    "attachment_y_px": int(attachment[1]),
                    "attachment_distance_px": float(distances[attachment_index]),
                    "tip_x_px": int(tip[0]),
                    "tip_y_px": int(tip[1]),
                    "branch_length_px": length_px,
                    "branch_angle_deg": float(
                        np.degrees(
                            np.arctan2(
                                int(tip[1]) - int(attachment[1]),
                                int(tip[0]) - int(attachment[0]),
                            )
                        )
                    ),
                    "branch_area_seed_px": int(len(path) * 6),
                }
            )
    return masks, graph_meta, branch_rows


def _component_filter(mask: np.ndarray, min_area: int, min_extent: int = 0) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for label_id in range(1, int(count)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if area >= int(min_area) and max(width, height) >= int(min_extent):
            out[labels == label_id] = 1
    return out


def _green_shoot_mask(image_rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image_rgb, dtype=np.uint8)
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    excess_green = (2 * green) - red - blue
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[: int(round(0.32 * height)), int(round(0.18 * width)) : int(round(0.82 * width))] = 1
    mask = (
        (excess_green >= 2)
        & (hsv[:, :, 1] >= 20)
        & (gray < 225)
        & (roi > 0)
    ).astype(np.uint8)
    mask = _component_filter(mask, min_area=20, min_extent=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1).astype(np.uint8)

    # Colored noise on the printed plate label can satisfy a permissive ExG rule.
    # Keep detached leaf pieces near the main shoot, but reject distant speckles.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    largest_id = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x0 = int(stats[largest_id, cv2.CC_STAT_LEFT])
    y0 = int(stats[largest_id, cv2.CC_STAT_TOP])
    x1 = x0 + int(stats[largest_id, cv2.CC_STAT_WIDTH])
    y1 = y0 + int(stats[largest_id, cv2.CC_STAT_HEIGHT])
    retained = np.zeros_like(mask)
    retained[labels == largest_id] = 1
    for label_id in range(1, int(count)):
        if label_id == largest_id:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        component_x1 = x + int(stats[label_id, cv2.CC_STAT_WIDTH])
        component_y1 = y + int(stats[label_id, cv2.CC_STAT_HEIGHT])
        dx = max(0, max(x - x1, x0 - component_x1))
        dy = max(0, max(y - y1, y0 - component_y1))
        if math.hypot(dx, dy) <= 220.0:
            retained[labels == label_id] = 1
    return retained


def _root_roi(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[400 : max(401, height - 250), 500 : max(501, width - 500)] = 1
    return roi


def _predict_probability(model: object, image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    tiles, coordinates, layout = collect_tiles(gray, 256)
    batch = tiles[:, :, :, None].astype(np.float32) / 255.0
    predictions = np.asarray(
        model.predict(batch, batch_size=12, verbose=0),
        dtype=np.float32,
    )
    return np.asarray(stitch_predictions(predictions[:, :, :, 0], coordinates, layout), dtype=np.float32)


def _load_isolated_keras_model(model_path: Path) -> object:
    """Clone the cached base model so endpoint weights cannot leak into Lazy inference."""

    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("TensorFlow is required for the Potato endpoint pipeline.") from exc
    cached = _load_keras_model_cached(str(Path(model_path).resolve()))
    isolated = tf.keras.models.clone_model(cached)
    isolated.set_weights(cached.get_weights())
    return isolated


def _clean_root_mask(probability: np.ndarray, threshold: float) -> np.ndarray:
    roi = _root_roi(probability.shape[:2])
    mask = ((probability >= float(threshold)) & (roi > 0)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return _component_filter(mask, min_area=12, min_extent=8)


def _select_primary_component(skeleton: np.ndarray) -> np.ndarray:
    binary = (np.asarray(skeleton, dtype=np.uint8) > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    height, width = binary.shape[:2]
    best_id = 0
    best_score = -float("inf")
    for label_id in range(1, int(count)):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        component_height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        center_x = float(centroids[label_id][0])
        score = (
            (3.0 * component_height)
            + area
            - (0.20 * y)
            - (0.12 * abs(center_x - (0.5 * width)))
        )
        if y > int(0.65 * height):
            score -= float(height)
        if score > best_score:
            best_score = score
            best_id = label_id
    return (labels == best_id).astype(np.uint8) if best_id > 0 else binary


def _nearest_attachment(component: np.ndarray, primary_skeleton: np.ndarray) -> tuple[int, int, float]:
    ys, xs = np.where(component > 0)
    if ys.size == 0 or not np.any(primary_skeleton > 0):
        return 0, 0, float("inf")
    distance = cv2.distanceTransform((primary_skeleton == 0).astype(np.uint8), cv2.DIST_L2, 3)
    local = distance[ys, xs]
    index = int(np.argmin(local))
    return int(xs[index]), int(ys[index]), float(local[index])


def _crown_point(shoot_mask: np.ndarray, root_mask: np.ndarray) -> tuple[int, int]:
    shoot_y, shoot_x = np.where(shoot_mask > 0)
    if shoot_y.size:
        crown_y = int(np.percentile(shoot_y, 96))
        lower = shoot_y >= int(np.percentile(shoot_y, 85))
        crown_x = int(np.median(shoot_x[lower])) if np.any(lower) else int(np.median(shoot_x))
        return crown_x, crown_y
    root_y, root_x = np.where(root_mask > 0)
    if root_y.size:
        top_band = root_y <= int(np.percentile(root_y, 5))
        return int(np.median(root_x[top_band])), int(np.min(root_y))
    height, width = root_mask.shape[:2]
    return width // 2, int(round(0.15 * height))


def _retain_crown_connected_root_system(
    root_mask: np.ndarray,
    shoot_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Remove detached model fragments while retaining the one crown-anchored plant."""
    root = (np.asarray(root_mask, dtype=np.uint8) > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(root, connectivity=8)
    before_components = max(0, int(component_count) - 1)
    before_pixels = int(np.count_nonzero(root))
    empty_meta = {
        "root_components_before_topology_filter": before_components,
        "root_components_after_topology_filter": before_components,
        "root_pixels_before_topology_filter": before_pixels,
        "root_pixels_removed_by_topology_filter": 0,
    }
    if component_count <= 2 or before_pixels == 0:
        return root, empty_meta

    crown_x, crown_y = _crown_point(shoot_mask, root)
    root_y, root_x = np.where(root > 0)
    nearest_index = int(np.argmin(np.hypot(root_x - crown_x, root_y - crown_y)))
    start_component = int(labels[root_y[nearest_index], root_x[nearest_index]])

    supported_ids = np.zeros(int(component_count), dtype=bool)
    supported_ids[start_component] = True
    for component_id in range(1, int(component_count)):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        component_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        if area >= 40 and component_height >= 60:
            supported_ids[component_id] = True

    supported = supported_ids[labels]
    factor = 4
    small_size = (
        int(math.ceil(root.shape[1] / factor)),
        int(math.ceil(root.shape[0] / factor)),
    )
    small_supported = cv2.resize(
        supported.astype(np.uint8),
        small_size,
        interpolation=cv2.INTER_AREA,
    ) > 0
    link_gap_px = max(24, int(round(0.0435 * root.shape[0])))
    link_radius = max(1, int(round(link_gap_px / (2 * factor))))
    link_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * link_radius + 1, 2 * link_radius + 1),
    )
    linked = cv2.dilate(small_supported.astype(np.uint8), link_kernel)
    _network_count, network_labels = cv2.connectedComponents(linked, connectivity=8)

    start_y, start_x = np.where(labels == start_component)
    start_network_values = network_labels[start_y // factor, start_x // factor]
    start_network_values = start_network_values[start_network_values > 0]
    if start_network_values.size == 0:
        connected_ids = np.zeros(int(component_count), dtype=bool)
        connected_ids[start_component] = True
    else:
        start_network = int(np.bincount(start_network_values).argmax())
        supported_y, supported_x = np.where(supported)
        in_start_network = (
            network_labels[supported_y // factor, supported_x // factor] == start_network
        )
        connected_ids = np.zeros(int(component_count), dtype=bool)
        connected_ids[np.unique(labels[supported_y[in_start_network], supported_x[in_start_network]])] = True
        connected_ids[0] = False

    connected_system = connected_ids[labels]
    distance_to_system = cv2.distanceTransform(
        (~connected_system).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    minimum_distance = np.full(int(component_count), np.inf, dtype=np.float32)
    np.minimum.at(minimum_distance, labels[root_y, root_x], distance_to_system[root_y, root_x])
    attachment_gap_px = max(12, int(round(0.012 * root.shape[0])))
    retained_ids = connected_ids | (minimum_distance <= float(attachment_gap_px))
    retained_ids[0] = False
    retained = retained_ids[labels].astype(np.uint8)
    after_components = max(
        0,
        int(cv2.connectedComponents(retained, connectivity=8)[0]) - 1,
    )
    after_pixels = int(np.count_nonzero(retained))
    return retained, {
        "root_components_before_topology_filter": before_components,
        "root_components_after_topology_filter": after_components,
        "root_pixels_before_topology_filter": before_pixels,
        "root_pixels_removed_by_topology_filter": max(0, before_pixels - after_pixels),
    }


def _primary_route_skeleton(
    root_mask: np.ndarray,
    shoot_mask: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]], dict[str, float | int]]:
    root = (np.asarray(root_mask, dtype=np.uint8) > 0).astype(np.uint8)
    all_skeleton = _skeletonize(root).astype(np.uint8)
    crown_x, crown_y = _crown_point(shoot_mask, root)
    ys, xs = np.where(all_skeleton > 0)
    if ys.size == 0:
        return np.zeros_like(root), [], {
            "crown_x_px": crown_x,
            "crown_y_px": crown_y,
            "primary_route_gap_length_px": 0.0,
            "primary_detected_seed_length_px": 0.0,
        }

    crown_distances = np.hypot(xs - crown_x, ys - crown_y)
    start_index = int(np.argmin(crown_distances))
    start_x = int(xs[start_index])
    start_y = int(ys[start_index])
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        all_skeleton,
        connectivity=8,
    )
    supported_components = np.zeros(component_count, dtype=bool)
    for component_id in range(1, int(component_count)):
        area = int(component_stats[component_id, cv2.CC_STAT_AREA])
        component_height = int(component_stats[component_id, cv2.CC_STAT_HEIGHT])
        if area >= 40 and component_height >= 60:
            supported_components[component_id] = True
    supported_pixels = supported_components[component_labels[ys, xs]]
    eligible = (ys >= start_y + max(40, int(round(0.02 * root.shape[0])))) & supported_pixels
    if not np.any(eligible):
        eligible = ys >= start_y + max(40, int(round(0.02 * root.shape[0])))
    if not np.any(eligible):
        selected = _select_primary_component(all_skeleton)
        path_xy, _tips, _branches = _skeleton_graph_path(selected > 0)
        return selected, path_xy, {
            "crown_x_px": crown_x,
            "crown_y_px": crown_y,
            "primary_base_x_px": start_x,
            "primary_base_y_px": start_y,
            "primary_tip_x_px": int(path_xy[-1][0]) if path_xy else start_x,
            "primary_tip_y_px": int(path_xy[-1][1]) if path_xy else start_y,
            "primary_route_gap_length_px": 0.0,
            "primary_detected_seed_length_px": float(_skeleton_edge_length_px(selected > 0)),
        }

    candidate_indices = np.flatnonzero(eligible)
    # Prefer depth while suppressing isolated lower specks far from the crown axis.
    target_scores = ys[candidate_indices] - (0.35 * np.abs(xs[candidate_indices] - crown_x))
    target_index = int(candidate_indices[int(np.argmax(target_scores))])
    target_x = int(xs[target_index])
    target_y = int(ys[target_index])

    factor = 4
    height, width = root.shape[:2]
    small_width = int(math.ceil(width / factor))
    small_height = int(math.ceil(height / factor))
    small_root = cv2.resize(
        root,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    ) > 0
    small_corridor = cv2.dilate(
        small_root.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ) > 0
    distance = cv2.distanceTransform((~small_corridor).astype(np.uint8), cv2.DIST_L2, 3)
    cost = 1.0 + (0.90 * np.minimum(distance, 25.0))
    start_rc = (
        min(small_height - 1, max(0, start_y // factor)),
        min(small_width - 1, max(0, start_x // factor)),
    )
    target_rc = (
        min(small_height - 1, max(0, target_y // factor)),
        min(small_width - 1, max(0, target_x // factor)),
    )
    try:
        from skimage.graph import route_through_array  # type: ignore

        route_rc, _route_cost = route_through_array(
            cost,
            start_rc,
            target_rc,
            fully_connected=True,
            geometric=True,
        )
    except Exception:
        selected = _select_primary_component(all_skeleton)
        path_xy, _tips, _branches = _skeleton_graph_path(selected > 0)
        return selected, path_xy, {
            "crown_x_px": crown_x,
            "crown_y_px": crown_y,
            "primary_base_x_px": start_x,
            "primary_base_y_px": start_y,
            "primary_tip_x_px": int(path_xy[-1][0]) if path_xy else target_x,
            "primary_tip_y_px": int(path_xy[-1][1]) if path_xy else target_y,
            "primary_route_gap_length_px": 0.0,
            "primary_detected_seed_length_px": float(_skeleton_edge_length_px(selected > 0)),
        }

    route_array = np.asarray(route_rc, dtype=np.int32)
    route_xy = [
        (
            min(width - 1, int(col * factor + (factor // 2))),
            min(height - 1, int(row * factor + (factor // 2))),
        )
        for row, col in route_array
    ]
    route_mask = np.zeros_like(root)
    if len(route_xy) >= 2:
        cv2.polylines(
            route_mask,
            [np.asarray(route_xy, dtype=np.int32)],
            isClosed=False,
            color=1,
            thickness=33,
        )
    elif route_xy:
        cv2.circle(route_mask, route_xy[0], 9, 1, thickness=-1)
    primary_seed = np.logical_and(all_skeleton > 0, route_mask > 0).astype(np.uint8)
    if np.count_nonzero(primary_seed) < 10:
        primary_seed = _select_primary_component(all_skeleton)

    route_length_px = float(_path_length_px(route_xy))
    gap_length_px = 0.0
    if len(route_array) >= 2:
        for route_index in range(1, len(route_array)):
            row, col = route_array[route_index]
            previous_row, previous_col = route_array[route_index - 1]
            if not small_corridor[int(row), int(col)]:
                gap_length_px += factor * float(
                    math.hypot(int(row) - int(previous_row), int(col) - int(previous_col))
                )
    return primary_seed, route_xy, {
        "crown_x_px": crown_x,
        "crown_y_px": crown_y,
        "primary_base_x_px": start_x,
        "primary_base_y_px": start_y,
        "primary_tip_x_px": target_x,
        "primary_tip_y_px": target_y,
        "primary_path_length_px": route_length_px,
        "primary_route_gap_length_px": float(gap_length_px),
        "primary_detected_seed_length_px": float(_skeleton_edge_length_px(primary_seed > 0)),
    }


def _single_primary_decomposition(
    total_root_mask: np.ndarray,
    high_confidence_root_mask: np.ndarray,
    shoot_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object], list[dict[str, object]]]:
    root = (np.asarray(total_root_mask, dtype=np.uint8) > 0).astype(np.uint8)
    high = (np.asarray(high_confidence_root_mask, dtype=np.uint8) > 0).astype(np.uint8)
    all_skeleton = _skeletonize(root)
    primary_skeleton, primary_path_xy, route_meta = _primary_route_skeleton(root, shoot_mask)
    _unused_path, tips, branches = _skeleton_graph_path(all_skeleton)
    side_skeleton = np.logical_and(all_skeleton > 0, primary_skeleton == 0).astype(np.uint8)
    side_skeleton = _component_filter(side_skeleton, min_area=6, min_extent=4)

    crown_x = int(route_meta["crown_x_px"])
    crown_y = int(route_meta["crown_y_px"])

    branch_rows: list[dict[str, object]] = []
    adventitious_skeleton = np.zeros_like(root, dtype=np.uint8)
    lateral_skeleton = np.zeros_like(root, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(side_skeleton, connectivity=8)
    for label_id in range(1, int(count)):
        component = (labels == label_id).astype(np.uint8)
        attachment_x, attachment_y, attachment_distance = _nearest_attachment(component, primary_skeleton)
        ys, xs = np.where(component > 0)
        if ys.size == 0:
            continue
        distances = np.hypot(xs - attachment_x, ys - attachment_y)
        tip_index = int(np.argmax(distances))
        tip_x = int(xs[tip_index])
        tip_y = int(ys[tip_index])
        length_px = float(_skeleton_edge_length_px(component > 0))
        angle_deg = float(np.degrees(np.arctan2(tip_y - attachment_y, tip_x - attachment_x)))
        near_crown = attachment_y <= crown_y + max(70, int(round(0.02 * root.shape[0])))
        branch_class = "adventitious_root" if near_crown else "lateral_root"
        if branch_class == "adventitious_root":
            adventitious_skeleton[component > 0] = 1
        else:
            lateral_skeleton[component > 0] = 1
        branch_rows.append(
            {
                "branch_class": branch_class,
                "attachment_x_px": attachment_x,
                "attachment_y_px": attachment_y,
                "attachment_distance_px": attachment_distance,
                "tip_x_px": tip_x,
                "tip_y_px": tip_y,
                "branch_length_px": length_px,
                "branch_angle_deg": angle_deg,
                "branch_area_seed_px": int(stats[label_id, cv2.CC_STAT_AREA]),
            }
        )

    if not np.any(lateral_skeleton) and not np.any(adventitious_skeleton):
        primary = root.copy()
        lateral = np.zeros_like(root)
        adventitious = np.zeros_like(root)
    else:
        distance_primary = cv2.distanceTransform((primary_skeleton == 0).astype(np.uint8), cv2.DIST_L2, 3)
        distance_lateral = (
            cv2.distanceTransform((lateral_skeleton == 0).astype(np.uint8), cv2.DIST_L2, 3)
            if np.any(lateral_skeleton)
            else np.full(root.shape, 1.0e6, dtype=np.float32)
        )
        distance_adventitious = (
            cv2.distanceTransform((adventitious_skeleton == 0).astype(np.uint8), cv2.DIST_L2, 3)
            if np.any(adventitious_skeleton)
            else np.full(root.shape, 1.0e6, dtype=np.float32)
        )
        stack = np.stack((distance_primary, distance_lateral, distance_adventitious), axis=0)
        assignment = np.argmin(stack, axis=0)
        primary = np.logical_and(root > 0, assignment == 0).astype(np.uint8)
        lateral = np.logical_and(root > 0, assignment == 1).astype(np.uint8)
        adventitious = np.logical_and(root > 0, assignment == 2).astype(np.uint8)

    masks = {
        "primary_root": primary,
        "lateral_root": lateral,
        "adventitious_root": adventitious,
        "shoot": (shoot_mask > 0).astype(np.uint8),
    }
    graph_meta = {
        **route_meta,
        "primary_path_points": len(primary_path_xy),
        "root_graph_tips_raw": int(tips),
        "root_graph_branches_raw": int(branches),
        "high_confidence_root_area_px": int(np.count_nonzero(high)),
        "decomposition_mode": "single_primary_crown_to_deepest_minimum_cost_route",
    }
    return masks, graph_meta, branch_rows


def _mask_geometry(mask: np.ndarray, pixel_size_mm: float) -> dict[str, float | int]:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    area_px = int(np.count_nonzero(binary))
    if area_px <= 0:
        return {
            "area_px": 0,
            "area_mm2": 0.0,
            "length_px": 0.0,
            "length_mm": 0.0,
            "perimeter_px": 0.0,
            "perimeter_mm": 0.0,
            "width_px": 0,
            "height_px": 0,
            "convex_hull_area_px2": 0.0,
            "tips": 0,
            "branch_points": 0,
            "mean_diameter_px": 0.0,
            "max_diameter_px": 0.0,
            "centroid_x_px": 0.0,
            "centroid_y_px": 0.0,
        }
    ys, xs = np.where(binary > 0)
    width_px = int(xs.max() - xs.min() + 1)
    height_px = int(ys.max() - ys.min() + 1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter_px = float(sum(cv2.arcLength(contour, True) for contour in contours))
    points = np.column_stack((xs, ys)).astype(np.int32)
    hull_area = float(cv2.contourArea(cv2.convexHull(points))) if len(points) >= 3 else 0.0
    skeleton = _skeletonize(binary)
    length_px = float(_skeleton_edge_length_px(skeleton))
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), np.uint8)) - skeleton.astype(np.uint8)
    tips = int(np.count_nonzero((skeleton > 0) & (neighbor_count == 1)))
    branch_pixels = ((skeleton > 0) & (neighbor_count >= 3)).astype(np.uint8)
    branch_count = max(0, int(cv2.connectedComponents(branch_pixels, connectivity=8)[0]) - 1)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    mean_diameter = float(area_px / max(length_px, 1.0))
    max_diameter = float(2.0 * np.max(distance))
    return {
        "area_px": area_px,
        "area_mm2": float(area_px * pixel_size_mm * pixel_size_mm),
        "length_px": length_px,
        "length_mm": float(length_px * pixel_size_mm),
        "perimeter_px": perimeter_px,
        "perimeter_mm": float(perimeter_px * pixel_size_mm),
        "width_px": width_px,
        "height_px": height_px,
        "convex_hull_area_px2": hull_area,
        "tips": tips,
        "branch_points": branch_count,
        "mean_diameter_px": mean_diameter,
        "max_diameter_px": max_diameter,
        "centroid_x_px": float(np.mean(xs)),
        "centroid_y_px": float(np.mean(ys)),
    }


def _prefix_metrics(prefix: str, metrics: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _primary_shape_metrics(
    primary_mask: np.ndarray,
    pixel_size_mm: float,
    route_meta: dict[str, object] | None = None,
) -> dict[str, float]:
    if route_meta and float(route_meta.get("primary_path_length_px", 0.0) or 0.0) > 0.0:
        path_length = float(route_meta["primary_path_length_px"])
        base_x = float(route_meta.get("primary_base_x_px", 0.0) or 0.0)
        base_y = float(route_meta.get("primary_base_y_px", 0.0) or 0.0)
        tip_x = float(route_meta.get("primary_tip_x_px", 0.0) or 0.0)
        tip_y = float(route_meta.get("primary_tip_y_px", 0.0) or 0.0)
        euclidean = float(np.hypot(tip_x - base_x, tip_y - base_y))
        angle = float(np.degrees(np.arctan2(abs(tip_x - base_x), max(1.0, abs(tip_y - base_y)))))
        return {
            "primary_path_length_px": path_length,
            "primary_path_length_mm": path_length * pixel_size_mm,
            "primary_euclidean_length_px": euclidean,
            "primary_tortuosity": path_length / max(euclidean, 1.0),
            "primary_angle_from_vertical_deg": angle,
            "primary_base_x_px": base_x,
            "primary_base_y_px": base_y,
            "primary_tip_x_px": tip_x,
            "primary_tip_y_px": tip_y,
        }
    skeleton = _skeletonize(primary_mask)
    path_xy, _tips, _branches = _skeleton_graph_path(skeleton)
    if len(path_xy) < 2:
        return {
            "primary_path_length_px": float(len(path_xy)),
            "primary_path_length_mm": float(len(path_xy) * pixel_size_mm),
            "primary_euclidean_length_px": 0.0,
            "primary_tortuosity": 0.0,
            "primary_angle_from_vertical_deg": 0.0,
            "primary_base_x_px": float(path_xy[0][0]) if path_xy else 0.0,
            "primary_base_y_px": float(path_xy[0][1]) if path_xy else 0.0,
            "primary_tip_x_px": float(path_xy[-1][0]) if path_xy else 0.0,
            "primary_tip_y_px": float(path_xy[-1][1]) if path_xy else 0.0,
        }
    path_length = float(_path_length_px(path_xy))
    base_x, base_y = path_xy[0]
    tip_x, tip_y = path_xy[-1]
    euclidean = float(np.hypot(tip_x - base_x, tip_y - base_y))
    angle = float(np.degrees(np.arctan2(abs(tip_x - base_x), max(1.0, tip_y - base_y))))
    return {
        "primary_path_length_px": path_length,
        "primary_path_length_mm": path_length * pixel_size_mm,
        "primary_euclidean_length_px": euclidean,
        "primary_tortuosity": path_length / max(euclidean, 1.0),
        "primary_angle_from_vertical_deg": angle,
        "primary_base_x_px": float(base_x),
        "primary_base_y_px": float(base_y),
        "primary_tip_x_px": float(tip_x),
        "primary_tip_y_px": float(tip_y),
    }


def _shoot_color_metrics(image_rgb: np.ndarray, shoot_mask: np.ndarray) -> dict[str, float]:
    pixels = np.asarray(image_rgb, dtype=np.uint8)[shoot_mask > 0]
    if pixels.size == 0:
        return {
            "shoot_mean_red": 0.0,
            "shoot_mean_green": 0.0,
            "shoot_mean_blue": 0.0,
            "shoot_excess_green_mean": 0.0,
            "shoot_hue_mean": 0.0,
            "shoot_saturation_mean": 0.0,
            "shoot_value_mean": 0.0,
        }
    hsv = cv2.cvtColor(pixels.reshape((-1, 1, 3)), cv2.COLOR_RGB2HSV).reshape((-1, 3))
    excess = (2.0 * pixels[:, 1]) - pixels[:, 0] - pixels[:, 2]
    return {
        "shoot_mean_red": float(np.mean(pixels[:, 0])),
        "shoot_mean_green": float(np.mean(pixels[:, 1])),
        "shoot_mean_blue": float(np.mean(pixels[:, 2])),
        "shoot_excess_green_mean": float(np.mean(excess)),
        "shoot_hue_mean": float(np.mean(hsv[:, 0])),
        "shoot_saturation_mean": float(np.mean(hsv[:, 1])),
        "shoot_value_mean": float(np.mean(hsv[:, 2])),
    }


def _exclusive_index_mask(masks: dict[str, np.ndarray]) -> np.ndarray:
    shape = next(iter(masks.values())).shape
    indexed = np.zeros(shape, dtype=np.uint8)
    for name in ("primary_root", "lateral_root", "adventitious_root", "shoot"):
        indexed[np.asarray(masks.get(name, np.zeros(shape)), dtype=np.uint8) > 0] = CLASS_IDS[name]
    return indexed


def _render_overlay(image_rgb: np.ndarray, masks: dict[str, np.ndarray], display_dilation: int = 1) -> np.ndarray:
    output = np.asarray(image_rgb, dtype=np.uint8).astype(np.float32)
    for name in ("primary_root", "lateral_root", "adventitious_root", "shoot"):
        mask = (np.asarray(masks.get(name, np.zeros(output.shape[:2])), dtype=np.uint8) > 0).astype(np.uint8)
        if display_dilation > 0 and name != "shoot":
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=display_dilation)
        active = mask > 0
        output[active] = (0.30 * output[active]) + (0.70 * np.asarray(CLASS_COLORS[name], dtype=np.float32))
    return np.clip(output, 0, 255).astype(np.uint8)


def _mask_rgb(indexed_mask: np.ndarray) -> np.ndarray:
    output = np.zeros((*indexed_mask.shape, 3), dtype=np.uint8)
    for name, class_id in CLASS_IDS.items():
        output[indexed_mask == class_id] = CLASS_COLORS[name]
    return output


def _save_overlay_with_legend(path: Path, overlay_rgb: np.ndarray, row: dict[str, object]) -> None:
    image = Image.fromarray(overlay_rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(38)
    font_bold = _font(42, bold=True)
    x0, y0 = 35, image.height - 335
    draw.rounded_rectangle((x0, y0, x0 + 1010, image.height - 35), radius=12, fill=(0, 0, 0, 205))
    draw.text((x0 + 28, y0 + 20), str(row["sample_stem"]), font=font_bold, fill=(255, 255, 255, 255))
    legend_y = y0 + 88
    for name in ("primary_root", "lateral_root", "adventitious_root", "shoot"):
        color = CLASS_COLORS[name]
        draw.rectangle((x0 + 28, legend_y + 4, x0 + 58, legend_y + 34), fill=(*color, 255))
        draw.text((x0 + 75, legend_y), name.replace("_", " "), font=font, fill=(245, 245, 245, 255))
        legend_y += 48
    source = str(row.get("mask_source", ""))
    draw.text((x0 + 510, y0 + 98), f"Source: {source}", font=font, fill=(235, 235, 235, 255))
    draw.text(
        (x0 + 510, y0 + 154),
        f"Total root: {float(row.get('total_root_length_mm', 0.0)):.2f} mm",
        font=font,
        fill=(235, 235, 235, 255),
    )
    draw.text(
        (x0 + 510, y0 + 210),
        f"Shoot area: {float(row.get('shoot_area_mm2', 0.0)):.2f} mm²",
        font=font,
        fill=(235, 235, 235, 255),
    )
    image.save(path, quality=92, subsampling=0)


def _video_frame(image_rgb: np.ndarray, row: dict[str, object], index: int, total: int, mask_only: bool) -> np.ndarray:
    canvas = Image.new("RGB", (1920, 1080), (12, 14, 18))
    image = Image.fromarray(image_rgb)
    image.thumbnail((1080, 1040), Image.Resampling.LANCZOS)
    canvas.paste(image, ((1100 - image.width) // 2, (1080 - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(34, bold=True)
    body_font = _font(25)
    small_font = _font(20)
    x = 1135
    draw.text((x, 34), "NPEC Potato Endpoint Phenotyping", font=title_font, fill=(255, 255, 255))
    draw.text((x, 84), f"Plate {index}/{total}", font=body_font, fill=(180, 190, 205))
    draw.text((x, 126), str(row.get("sample_stem", "")), font=small_font, fill=(225, 228, 235))
    draw.text((x, 165), f"View: {'mask only' if mask_only else 'image overlay'}", font=small_font, fill=(165, 175, 190))
    y = 220
    for name in ("primary_root", "lateral_root", "adventitious_root", "shoot"):
        color = CLASS_COLORS[name]
        draw.rectangle((x, y + 4, x + 24, y + 28), fill=color)
        draw.text((x + 38, y), name.replace("_", " "), font=body_font, fill=(235, 238, 242))
        y += 42
    y += 16
    metrics = (
        ("Total root length", "total_root_length_mm", "mm"),
        ("Primary length", "primary_root_length_mm", "mm"),
        ("Lateral length", "lateral_root_length_mm", "mm"),
        ("Adventitious length", "adventitious_root_length_mm", "mm"),
        ("Root area", "total_root_area_mm2", "mm²"),
        ("Shoot area", "shoot_area_mm2", "mm²"),
        ("Root depth", "total_root_height_px", "px"),
        ("Root width", "total_root_width_px", "px"),
        ("Root tips", "total_root_tips", ""),
        ("Lateral branches", "lateral_branch_count", ""),
        ("Primary angle", "primary_angle_from_vertical_deg", "°"),
    )
    for label, key, unit in metrics:
        value = float(row.get(key, 0.0) or 0.0)
        rendered = f"{value:.2f}" if unit not in {"", "px"} else f"{value:.0f}"
        draw.text((x, y), label, font=small_font, fill=(175, 184, 198))
        draw.text((1670, y), f"{rendered} {unit}".strip(), font=small_font, fill=(246, 248, 250))
        y += 35
    y += 16
    source = str(row.get("mask_source", ""))
    draw.text((x, y), f"Mask source: {source}", font=small_font, fill=(180, 190, 205))
    draw.text((x, y + 34), f"Scale: {float(row.get('pixel_size_mm', 0.0)):.6f} mm/px", font=small_font, fill=(180, 190, 205))
    draw.text((x, 1035), "Single endpoint per plate: no growth-rate inference", font=small_font, fill=(245, 176, 85))
    return np.asarray(canvas, dtype=np.uint8)


def _write_video(
    output_path: Path,
    rows: list[dict[str, object]],
    overlay_paths: dict[str, Path],
    mask_paths: dict[str, Path],
    fps: float,
    mask_only: bool,
) -> None:
    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        (1920, 1080),
        fps=float(fps),
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "23", "-preset", "medium", "-movflags", "+faststart"],
        macro_block_size=1,
    )
    writer.send(None)
    try:
        total = len(rows)
        for index, row in enumerate(rows, start=1):
            stem = str(row["sample_stem"])
            if mask_only:
                indexed = np.asarray(Image.open(mask_paths[stem]), dtype=np.uint8)
                image_rgb = _mask_rgb(indexed)
            else:
                image_rgb = np.asarray(Image.open(overlay_paths[stem]).convert("RGB"), dtype=np.uint8)
            frame = _video_frame(image_rgb, row, index, total, mask_only)
            writer.send(frame.tobytes())
            if index % 25 == 0 or index == total:
                print(f"video {output_path.name}: {index}/{total}", flush=True)
    finally:
        writer.close()


def _trait_dictionary() -> pd.DataFrame:
    rows = [
        ("*_length_mm", "Skeleton edge length converted with the Lucifer pixel scale."),
        ("*_area_mm2", "Foreground pixel area converted with the squared Lucifer pixel scale."),
        ("primary_tortuosity", "Primary path length divided by base-to-tip Euclidean distance."),
        ("primary_angle_from_vertical_deg", "Absolute primary base-to-tip deviation from image vertical."),
        ("tips", "Skeleton endpoints with one 8-connected neighbor."),
        ("branch_points", "Connected clusters of skeleton pixels with at least three neighbors."),
        ("adventitious_root", "Model-only crown-emerging side-root candidate; manual PSD class when available."),
        ("topology_filter_applied", "True when a noisy model mask was restricted to the one crown-connected root system."),
        (
            "mask_source",
            "manual_psd, dario_finetuned_root_plus_green_shoot, or an explicitly QC-flagged forced rescue.",
        ),
        ("rescue_applied", "True only for a reviewed, deterministic per-capture rescue specification."),
    ]
    return pd.DataFrame(rows, columns=["field", "definition"])


def _write_workbook(
    output_path: Path,
    phenotype_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    provenance: dict[str, object],
    validation: dict[str, object] | None,
    plate_label_map_df: pd.DataFrame | None = None,
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        phenotype_df.to_excel(writer, sheet_name="Endpoint phenotypes", index=False)
        branch_df.to_excel(writer, sheet_name="Branch traits", index=False)
        qc_df.to_excel(writer, sheet_name="QC", index=False)
        excluded_df.to_excel(writer, sheet_name="Excluded captures", index=False)
        summary_df.to_excel(writer, sheet_name="Dataset summary", index=False)
        _trait_dictionary().to_excel(writer, sheet_name="Trait dictionary", index=False)
        if validation:
            aggregate_rows = [
                {"class": class_name, **dict(metrics)}
                for class_name, metrics in dict(validation.get("aggregate", {})).items()
            ]
            pd.DataFrame(aggregate_rows).to_excel(
                writer,
                sheet_name="Validation aggregate",
                index=False,
            )
            pd.DataFrame(validation.get("per_image", [])).to_excel(
                writer,
                sheet_name="Validation plates",
                index=False,
            )
        pd.DataFrame(
            [{"key": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in provenance.items()]
        ).to_excel(writer, sheet_name="Provenance", index=False)
        if plate_label_map_df is not None and not plate_label_map_df.empty:
            plate_label_map_df.to_excel(writer, sheet_name="Plate label map", index=False)


def _dataset_summary(phenotype_df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        column
        for column in phenotype_df.columns
        if column.endswith(("_length_mm", "_area_mm2", "_width_px", "_height_px"))
        or column
        in {
            "root_to_shoot_area_ratio",
            "primary_tortuosity",
            "primary_angle_from_vertical_deg",
            "total_root_tips",
            "total_root_branch_points",
            "lateral_branch_count",
            "adventitious_branch_count",
        }
    ]
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("all", phenotype_df)]
    groups.extend(
        (str(mask_source), group)
        for mask_source, group in phenotype_df.groupby("mask_source", dropna=False)
    )
    for group_name, group in groups:
        for metric in preferred:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "group": group_name,
                    "metric": metric,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "q25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q75": float(values.quantile(0.75)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def _stem_metadata(stem: str) -> dict[str, object]:
    match = re.fullmatch(
        r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d+)",
        stem,
    )
    if match is None:
        return {"acquisition_datetime": "", "capture_sequence": 0}
    year, month, day, hour, minute, second, sequence = match.groups()
    return {
        "acquisition_datetime": f"{year}-{month}-{day}T{hour}:{minute}:{second}",
        "capture_sequence": int(sequence),
    }


def _manual_branch_rows(masks: dict[str, np.ndarray], pixel_size_mm: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    primary_skeleton = _skeletonize(masks.get("primary_root", np.zeros_like(next(iter(masks.values())))))
    for class_name in ("lateral_root", "adventitious_root"):
        skeleton = _skeletonize(masks.get(class_name, np.zeros_like(primary_skeleton)))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton.astype(np.uint8), connectivity=8)
        for label_id in range(1, int(count)):
            component = (labels == label_id).astype(np.uint8)
            if int(stats[label_id, cv2.CC_STAT_AREA]) < 4:
                continue
            attachment_x, attachment_y, attachment_distance = _nearest_attachment(component, primary_skeleton)
            ys, xs = np.where(component > 0)
            distances = np.hypot(xs - attachment_x, ys - attachment_y)
            tip_index = int(np.argmax(distances))
            length_px = float(_skeleton_edge_length_px(component > 0))
            rows.append(
                {
                    "branch_class": class_name,
                    "attachment_x_px": attachment_x,
                    "attachment_y_px": attachment_y,
                    "attachment_distance_px": attachment_distance,
                    "tip_x_px": int(xs[tip_index]),
                    "tip_y_px": int(ys[tip_index]),
                    "branch_length_px": length_px,
                    "branch_length_mm": length_px * pixel_size_mm,
                    "branch_angle_deg": float(
                        np.degrees(np.arctan2(ys[tip_index] - attachment_y, xs[tip_index] - attachment_x))
                    ),
                    "branch_area_seed_px": int(stats[label_id, cv2.CC_STAT_AREA]),
                }
            )
    return rows


def _arg(args: argparse.Namespace, name: str, default: object = None) -> object:
    return getattr(args, name, default)


def _path_key(path: Path) -> str:
    try:
        return str(Path(path).expanduser().resolve()).casefold()
    except OSError:
        return str(Path(path).expanduser().absolute()).casefold()


def _load_identity_records(args: argparse.Namespace) -> list[dict[str, object]]:
    supplied = _arg(args, "identity_records", None)
    if supplied is not None:
        return [dict(row) for row in supplied if isinstance(row, Mapping)]
    manifest = _arg(args, "identity_manifest", None)
    if manifest:
        return read_plate_identity_manifest(Path(manifest))
    return []


def _identity_evidence_text(record: Mapping[str, object]) -> str:
    evidence = record.get("evidence", [])
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        return ""
    parts: list[str] = []
    for item in evidence:
        if isinstance(item, Mapping):
            text = str(item.get("text", "") or "").strip()
            if text and text not in parts:
                parts.append(text)
    return " | ".join(parts)


def _plate_fields(plate_id: str) -> tuple[object, str, str]:
    tokens = [token for token in sanitize_plate_id(plate_id).split("_") if token]
    plate_number: object = ""
    genotype = ""
    treatment = ""
    if tokens and re.fullmatch(r"\d{1,6}", tokens[0]):
        plate_number = int(tokens[0])
        tokens = tokens[1:]
    if tokens:
        genotype = tokens[0]
    if len(tokens) > 1:
        treatment = "_".join(tokens[1:])
    return plate_number, genotype, treatment


def _prepare_image_entries(
    images: Sequence[Path],
    identity_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_path: dict[str, dict[str, object]] = {}
    by_name: dict[str, list[dict[str, object]]] = {}
    for raw_record in identity_records:
        record = dict(raw_record)
        source_path = str(record.get("source_path", "") or "").strip()
        if source_path:
            by_path[_path_key(Path(source_path))] = record
        source_name = str(record.get("source_name", "") or "").strip().casefold()
        if source_name:
            by_name.setdefault(source_name, []).append(record)

    selected: list[tuple[Path, dict[str, object]]] = []
    for image_path in images:
        record = by_path.get(_path_key(image_path), {})
        if not record:
            name_records = by_name.get(image_path.name.casefold(), [])
            if len(name_records) == 1:
                record = name_records[0]
        selected.append((image_path, dict(record)))
    selected.sort(
        key=lambda item: (
            str(item[1].get("source_date", "") or acquisition_datetime(item[0])[0].isoformat()),
            str(item[0]).casefold(),
        )
    )

    plate_totals: dict[str, int] = {}
    for image_path, record in selected:
        plate_id = sanitize_plate_id(record.get("plate_id", ""))
        plate_key = plate_id.casefold() if plate_id else f"~{_path_key(image_path)}"
        plate_totals[plate_key] = plate_totals.get(plate_key, 0) + 1

    used_stems: set[str] = set()
    plate_indices: dict[str, int] = {}
    entries: list[dict[str, object]] = []
    for image_path, record in selected:
        plate_id = sanitize_plate_id(record.get("plate_id", ""))
        plate_key = plate_id.casefold() if plate_id else f"~{_path_key(image_path)}"
        plate_indices[plate_key] = plate_indices.get(plate_key, 0) + 1
        capture_index = int(record.get("plate_frame_index", 0) or 0)
        if capture_index <= 0:
            capture_index = plate_indices[plate_key]

        plate_count = int(plate_totals.get(plate_key, 1))
        allocated_name = str(record.get("allocated_name", "") or "").strip()
        if plate_id:
            export_stem = (
                f"{plate_id}__capture{capture_index:02d}"
                if plate_count > 1
                else plate_id
            )
        else:
            export_stem = sanitize_plate_id(Path(allocated_name).stem) if allocated_name else ""
            if not export_stem:
                export_stem = sanitize_plate_id(image_path.stem) or f"endpoint_{len(entries) + 1:04d}"
        candidate = export_stem
        suffix_index = 1
        while candidate.casefold() in used_stems:
            suffix_index += 1
            candidate = f"{export_stem}__capture{suffix_index:02d}"
        export_stem = candidate
        used_stems.add(export_stem.casefold())

        acquired_at, date_source = acquisition_datetime(image_path)
        acquisition_text = str(record.get("source_date", "") or "").strip()
        if not acquisition_text:
            acquisition_text = acquired_at.isoformat(timespec="seconds")
        source_metadata = _stem_metadata(image_path.stem)
        plate_number, genotype, treatment = _plate_fields(plate_id)
        ocr_text = _identity_evidence_text(record)
        confidence = float(record.get("confidence", 0.0) or 0.0)
        metadata = {
            "sample_stem": export_stem,
            "source_image_stem": image_path.stem,
            "plate_name": plate_id or export_stem,
            "plate_number": plate_number,
            "genotype": genotype,
            "treatment": treatment,
            "plate_capture_index": capture_index,
            "plate_capture_count": plate_count,
            "is_duplicate_plate_capture": plate_count > 1,
            "acquisition_datetime": acquisition_text,
            "capture_sequence": source_metadata.get("capture_sequence", 0),
            "plate_number_parse_method": str(record.get("status", "") or "unidentified"),
            "plate_number_resolution_score": round(confidence * 100.0, 4),
            "ocr_plate_text": plate_id,
            "ocr_plate_confidence": confidence,
            "ocr_plate_x": "",
            "ocr_genotype_text": genotype,
            "ocr_treatment_text": treatment,
            "ocr_treatment_candidates": treatment,
            "ocr_text": ocr_text,
            "ocr_candidate_audit_json": json.dumps(record.get("evidence", []), separators=(",", ":")),
            "source_date_source": str(record.get("source_date_source", "") or date_source),
            "plate_identity_status": str(record.get("status", "") or "unidentified"),
        }
        entries.append({"path": image_path, "record": record, "metadata": metadata})
    return entries


def _ordered_columns(frame: pd.DataFrame, preferred: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [column for column in preferred if column in frame.columns]
    columns.extend(column for column in frame.columns if column not in columns)
    return frame.loc[:, columns]


def _plate_label_map(
    entries: Sequence[Mapping[str, object]],
    output: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_index, entry in enumerate(entries, start=1):
        image_path = Path(entry["path"])
        metadata = dict(entry["metadata"])
        export_stem = str(metadata["sample_stem"])
        rows.append(
            {
                "sample_index": sample_index,
                "source_image_stem": metadata["source_image_stem"],
                "acquisition_datetime": metadata["acquisition_datetime"],
                "capture_sequence": metadata["capture_sequence"],
                "plate_number": metadata["plate_number"],
                "genotype": metadata["genotype"],
                "treatment": metadata["treatment"],
                "plate_name": metadata["plate_name"],
                "plate_capture_index": metadata["plate_capture_index"],
                "plate_capture_count": metadata["plate_capture_count"],
                "is_duplicate_plate_capture": metadata["is_duplicate_plate_capture"],
                "export_stem": export_stem,
                "plate_number_parse_method": metadata["plate_number_parse_method"],
                "plate_number_resolution_score": metadata["plate_number_resolution_score"],
                "ocr_plate_text": metadata["ocr_plate_text"],
                "ocr_plate_confidence": metadata["ocr_plate_confidence"],
                "ocr_plate_x": metadata["ocr_plate_x"],
                "ocr_genotype_text": metadata["ocr_genotype_text"],
                "ocr_treatment_text": metadata["ocr_treatment_text"],
                "ocr_treatment_candidates": metadata["ocr_treatment_candidates"],
                "ocr_text": metadata["ocr_text"],
                "ocr_candidate_audit_json": metadata["ocr_candidate_audit_json"],
                "source_label_crop_path": str(output / "label_crops" / f"{export_stem}__label.jpg"),
                "export_crop_path": str(output / "crops" / f"{export_stem}.jpg"),
                "export_label_crop_path": str(output / "label_crops" / f"{export_stem}__label.jpg"),
            }
        )
    return pd.DataFrame(rows)


def _iter_selected_images(args: argparse.Namespace) -> tuple[list[Path], list[dict[str, str]]]:
    supplied_paths = _arg(args, "image_paths", None)
    if supplied_paths is not None:
        images = [Path(path).expanduser().resolve() for path in supplied_paths]
    else:
        dataset = Path(args.dataset).expanduser().resolve()
        requested_input = _arg(args, "input_dir", None)
        if requested_input is not None:
            raw_dir = Path(requested_input).expanduser().resolve()
        else:
            raw_dir = dataset / "raw" if (dataset / "raw").is_dir() else dataset
        images = [
            path.resolve()
            for path in raw_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg"}
        ]
    images = sorted(set(images), key=lambda path: str(path).casefold())
    selected_stems = {value.strip() for value in str(_arg(args, "stems", "")).split(",") if value.strip()}
    if selected_stems:
        images = [path for path in images if path.stem in selected_stems]
    excluded: list[dict[str, str]] = []
    use_dario_exclusions = not bool(_arg(args, "generic_dataset", False))
    if use_dario_exclusions and not bool(_arg(args, "include_calibration", False)):
        kept: list[Path] = []
        for path in images:
            reason = KNOWN_NON_PLANT_CAPTURES.get(path.stem)
            if reason is None:
                kept.append(path)
            else:
                excluded.append({"sample_stem": path.stem, "source_path": str(path), "reason": reason})
        images = kept
    if int(_arg(args, "limit", 0) or 0) > 0:
        images = images[: int(_arg(args, "limit", 0) or 0)]
    return images, excluded


def _write_readme(
    path: Path,
    phenotype_rows: list[dict[str, object]],
    validation: dict[str, object] | None,
    pixel_size_mm: float = LUCIFER_PIXEL_SIZE_MM,
) -> None:
    manual_count = sum(row["mask_source"] == "manual_psd" for row in phenotype_rows)
    model_count = sum(row["mask_source"] == MODEL_MASK_SOURCE for row in phenotype_rows)
    rescue_count = sum(row["mask_source"] == FORCED_MASK_SOURCE for row in phenotype_rows)
    validation_lines = ["No strict holdout validation report was available."]
    if validation:
        validation_lines = []
        for class_name, metrics in dict(validation.get("aggregate", {})).items():
            validation_lines.append(
                f"- {class_name}: Dice {float(metrics['dice']):.3f}, "
                f"precision {float(metrics['precision']):.3f}, "
                f"recall {float(metrics['recall']):.3f}"
            )
    path.write_text(
        "\n".join(
            [
                "# NPEC Potato Endpoint Phenotyping Package",
                "",
                f"Analyzed plates: {len(phenotype_rows)} ({manual_count} manual PSD masks; {model_count} model-derived masks; {rescue_count} reviewed forced rescue).",
                "",
                "This dataset contains one independent final time point per plate. The two MP4 files are endpoint review reels, not biological timelapses. Growth rates and temporal deltas cannot be estimated from these captures.",
                "",
                "## Class colors",
                "",
                "- Orange: primary root",
                "- Blue: lateral root",
                "- Cyan: adventitious-root candidate",
                "- Pink: green shoot",
                "",
                "## Main files",
                "",
                "- npec_potato_endpoint_all_metrics.xlsx: complete workbook",
                "- npec_potato_endpoint_phenotypes.csv: one row per plate",
                "- npec_potato_branch_traits.csv: one row per detected branch component",
                "- npec_potato_qc.csv: review flags and confidence fields",
                "- npec_potato_endpoint_overlay_reel.mp4: source image with masks and measurements",
                "- npec_potato_endpoint_mask_reel.mp4: mask-only review reel",
                "- masks/: indexed and per-class lossless PNG masks",
                "- overlays/: full-resolution review overlays",
                "- label_crops/: printed plate-label crops for metadata reconciliation",
                "- models/: exact architecture and fine-tuned weights used for model-derived plates",
                "",
                "## Validation and interpretation",
                "",
                *validation_lines,
                "",
                "Total-root, primary-root, and shoot measurements are the strongest automated outputs. Lateral-root predictions are conservative (higher precision than recall), so fine laterals can be missed. Adventitious-root output is a topology-based candidate class with weaker validation and should be reviewed before inferential use. Noisy model masks with excessive tips or implausible route gaps are restricted to the one crown-connected root system; those rows remain flagged with topology_cleanup_applied for review.",
                "",
                f"Manual PSD masks retained exactly: {manual_count}. Reviewed one-off forced rescues: {rescue_count}.",
                "",
                f"Pixel scale: {pixel_size_mm:.9f} mm/px (Lucifer endpoint calibration).",
                "",
                "## Plate identity",
                "",
                "Exported crops, masks, overlays, and video labels use the detected or reviewed physical plate ID. Repeated captures add __captureNN in acquisition order. Original source stems and acquisition dates remain in the phenotype and plate-map tables; raw source files are not renamed.",
                "",
                "See npec_potato_plate_label_map.csv. npec_dario_plate_label_map.csv is retained as a compatibility alias for the original Dario package contract.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    started = time.time()
    dataset = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    overwrite = bool(_arg(args, "overwrite", False))
    generic_dataset = bool(_arg(args, "generic_dataset", False))
    enable_forced_rescues = bool(_arg(args, "enable_forced_rescues", not generic_dataset))
    pixel_size_mm = float(_arg(args, "pixel_size_mm", LUCIFER_PIXEL_SIZE_MM))
    skip_videos = bool(_arg(args, "skip_videos", False))
    video_fps = float(_arg(args, "video_fps", 2.0))
    progress_callback = _arg(args, "progress_callback", None)
    cancel_callback = _arg(args, "cancel_callback", None)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output}. Use --overwrite to replace it.")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for folder_name in ("crops", "masks", "overlays", "label_crops", "models"):
            shutil.rmtree(output / folder_name, ignore_errors=True)

    crops_dir = output / "crops"
    masks_dir = output / "masks"
    overlays_dir = output / "overlays"
    label_crops_dir = output / "label_crops"
    models_dir = output / "models"
    for directory in (crops_dir, masks_dir, overlays_dir, label_crops_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    images, excluded_rows = _iter_selected_images(args)
    if not images:
        raise ValueError("No JPEG captures were selected.")
    identity_records = _load_identity_records(args)
    image_entries = _prepare_image_entries(images, identity_records)
    image_entries = sorted(
        image_entries,
        key=lambda entry: (
            str(entry["metadata"].get("acquisition_datetime", "")),
            int(entry["metadata"].get("capture_sequence", 0) or 0),
            str(entry["path"]).casefold(),
        ),
    )
    selected_stems = {Path(entry["path"]).stem for entry in image_entries}
    labels_dir_value = _arg(args, "labels_dir", None)
    labels_dir = (
        Path(labels_dir_value).expanduser().resolve()
        if labels_dir_value is not None
        else dataset / "labeled"
    )
    manual_references = (
        _load_manual_references(labels_dir, selected_stems)
        if not bool(_arg(args, "no_manual_labels", False)) and labels_dir.is_dir()
        else {}
    )

    root_model_path = Path(_arg(args, "root_model", DEFAULT_ROOT_MODEL)).expanduser().resolve()
    root_weights_path = Path(_arg(args, "root_weights", DEFAULT_ROOT_WEIGHTS)).expanduser().resolve()
    if not root_model_path.exists() or not root_weights_path.exists():
        raise FileNotFoundError(
            f"Missing potato endpoint root model or fine-tuned weights: {root_model_path}, {root_weights_path}"
        )
    root_model = _load_isolated_keras_model(root_model_path)
    root_model.load_weights(root_weights_path)
    shutil.copy2(root_model_path, models_dir / root_model_path.name)
    shutil.copy2(root_weights_path, models_dir / root_weights_path.name)
    validation_value = _arg(args, "validation_path", None)
    validation_source = (
        Path(validation_value).expanduser().resolve()
        if validation_value is not None
        else DEFAULT_VALIDATION
    )
    legacy_validation = dataset / "analysis_work/holdout_eval/final_v3_topology_metrics.json"
    if validation_value is None and legacy_validation.exists():
        validation_source = legacy_validation
    validation: dict[str, object] | None = None
    if validation_source.exists():
        validation = json.loads(validation_source.read_text(encoding="utf-8"))
        shutil.copy2(validation_source, output / "npec_potato_reference_validation.json")

    phenotype_rows: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    overlay_paths: dict[str, Path] = {}
    mask_paths: dict[str, Path] = {}

    for index, entry in enumerate(image_entries, start=1):
        if callable(cancel_callback) and bool(cancel_callback()):
            raise PotatoEndpointCancelled("Potato endpoint package generation was canceled.")
        image_path = Path(entry["path"])
        identity_metadata = dict(entry["metadata"])
        source_stem = image_path.stem
        stem = str(identity_metadata["sample_stem"])
        reference = manual_references.get(source_stem)
        crop_meta: dict[str, object]
        if reference is not None:
            image_rgb = reference.image_rgb.copy()
            masks = {
                name: (reference.masks.get(name, np.zeros(image_rgb.shape[:2], dtype=np.uint8)) > 0).astype(np.uint8)
                for name in ("primary_root", "lateral_root", "adventitious_root", "shoot")
            }
            mask_source = "manual_psd"
            crop_meta = {
                "crop_source": "psd_background",
                "manual_psd_path": str(reference.psd_path),
                "normalized_size": [int(image_rgb.shape[1]), int(image_rgb.shape[0])],
            }
            graph_meta = {
                "decomposition_mode": "manual_psd_layers",
                "crown_x_px": 0,
                "crown_y_px": 0,
                "primary_path_points": 0,
                "root_graph_tips_raw": 0,
                "root_graph_branches_raw": 0,
                "topology_filter_applied": False,
                "topology_filter_trigger": "",
                "prefilter_root_graph_tips_raw": 0,
                "prefilter_primary_route_gap_length_px": 0.0,
                "root_components_before_topology_filter": 0,
                "root_components_after_topology_filter": 0,
                "root_pixels_before_topology_filter": 0,
                "root_pixels_removed_by_topology_filter": 0,
            }
            item_branches = _manual_branch_rows(masks, pixel_size_mm)
            probability_mean = 1.0
            low_confidence_fraction = 0.0
        else:
            image_rgb, crop_meta = _load_normalized_crop(image_path)
            crop_meta["crop_source"] = "raw_jpeg_fixed_lucifer_geometry"
            shoot_mask = _green_shoot_mask(image_rgb)
            rescue_spec = _forced_rescue_spec(source_stem) if enable_forced_rescues else None
            if rescue_spec is not None:
                masks, graph_meta, item_branches = _forced_crown_anchored_rescue(
                    image_rgb,
                    shoot_mask,
                    rescue_spec,
                )
                graph_meta["rescue_source_jpeg_sha256"] = _sha256(image_path)
                mask_source = FORCED_MASK_SOURCE
                probability_mean = float("nan")
                low_confidence_fraction = float("nan")
            else:
                root_probability = _predict_probability(root_model, image_rgb)
                total_root = _clean_root_mask(root_probability, ROOT_THRESHOLD)
                high_confidence = _clean_root_mask(root_probability, ROOT_GRAPH_THRESHOLD)
                total_root[shoot_mask > 0] = 0
                high_confidence[shoot_mask > 0] = 0
                masks, graph_meta, item_branches = _single_primary_decomposition(
                    total_root,
                    high_confidence,
                    shoot_mask,
                )
                prefilter_tips = int(graph_meta.get("root_graph_tips_raw", 0) or 0)
                prefilter_gap = float(graph_meta.get("primary_route_gap_length_px", 0.0) or 0.0)
                prefilter_route_length = float(graph_meta.get("primary_path_length_px", 0.0) or 0.0)
                cleanup_triggers: list[str] = []
                if prefilter_tips > 120:
                    cleanup_triggers.append("excess_root_tips")
                if prefilter_gap > max(300.0, 0.15 * prefilter_route_length):
                    cleanup_triggers.append("large_primary_route_gap")
                topology_meta = {
                    "root_components_before_topology_filter": max(
                        0,
                        int(cv2.connectedComponents(total_root, connectivity=8)[0]) - 1,
                    ),
                    "root_components_after_topology_filter": max(
                        0,
                        int(cv2.connectedComponents(total_root, connectivity=8)[0]) - 1,
                    ),
                    "root_pixels_before_topology_filter": int(np.count_nonzero(total_root)),
                    "root_pixels_removed_by_topology_filter": 0,
                }
                topology_applied = False
                if cleanup_triggers:
                    filtered_root, topology_meta = _retain_crown_connected_root_system(
                        total_root,
                        shoot_mask,
                    )
                    if np.any(filtered_root) and np.count_nonzero(filtered_root) < np.count_nonzero(total_root):
                        total_root = filtered_root
                        high_confidence = np.logical_and(high_confidence > 0, total_root > 0).astype(np.uint8)
                        masks, graph_meta, item_branches = _single_primary_decomposition(
                            total_root,
                            high_confidence,
                            shoot_mask,
                        )
                        topology_applied = True
                graph_meta.update(topology_meta)
                graph_meta.update(
                    {
                        "topology_filter_applied": topology_applied,
                        "topology_filter_trigger": ";".join(cleanup_triggers),
                        "prefilter_root_graph_tips_raw": prefilter_tips,
                        "prefilter_primary_route_gap_length_px": prefilter_gap,
                    }
                )
                mask_source = MODEL_MASK_SOURCE
                root_pixels = root_probability[total_root > 0]
                probability_mean = float(np.mean(root_pixels)) if root_pixels.size else 0.0
                low_confidence_fraction = (
                    float(np.mean(root_pixels < ROOT_GRAPH_THRESHOLD)) if root_pixels.size else 1.0
                )

        graph_meta.setdefault("rescue_applied", False)
        graph_meta.setdefault("rescue_method", "")
        graph_meta.setdefault("rescue_spec_sha256", "")
        graph_meta.setdefault("rescue_source_jpeg_sha256", "")
        graph_meta.setdefault("rescue_normalized_size", "")

        total_root_mask = np.maximum.reduce(
            [masks["primary_root"], masks["lateral_root"], masks["adventitious_root"]]
        ).astype(np.uint8)
        indexed = _exclusive_index_mask(masks)
        crop_path = crops_dir / f"{stem}.jpg"
        Image.fromarray(image_rgb).save(crop_path, quality=94, subsampling=0)
        indexed_path = masks_dir / f"{stem}__classes.png"
        Image.fromarray(indexed).save(indexed_path, optimize=True)
        mask_paths[stem] = indexed_path
        for name, mask in masks.items():
            Image.fromarray((mask > 0).astype(np.uint8) * 255).save(
                masks_dir / f"{stem}__{name}.png",
                optimize=True,
            )
        Image.fromarray(total_root_mask * 255).save(
            masks_dir / f"{stem}__total_root.png",
            optimize=True,
        )

        # The printed plate label is retained as a separate image for manual metadata mapping.
        label_crop = Image.fromarray(image_rgb).crop((200, 0, 1500, 650))
        label_crop_path = label_crops_dir / f"{stem}__label.jpg"
        label_crop.save(label_crop_path, quality=94)

        primary_metrics = _mask_geometry(masks["primary_root"], pixel_size_mm)
        lateral_metrics = _mask_geometry(masks["lateral_root"], pixel_size_mm)
        adventitious_metrics = _mask_geometry(masks["adventitious_root"], pixel_size_mm)
        total_metrics = _mask_geometry(total_root_mask, pixel_size_mm)
        shoot_metrics = _mask_geometry(masks["shoot"], pixel_size_mm)
        primary_shape = _primary_shape_metrics(
            masks["primary_root"],
            pixel_size_mm,
            graph_meta if mask_source != "manual_psd" else None,
        )
        shoot_color = _shoot_color_metrics(image_rgb, masks["shoot"])

        row: dict[str, object] = {
            "sample_index": index,
            **{
                key: identity_metadata[key]
                for key in (
                    "sample_stem",
                    "source_image_stem",
                    "plate_name",
                    "plate_number",
                    "genotype",
                    "treatment",
                    "plate_capture_index",
                    "plate_capture_count",
                    "is_duplicate_plate_capture",
                    "acquisition_datetime",
                    "capture_sequence",
                )
            },
            "source_jpeg": str(image_path),
            "source_cr3": str(image_path.with_suffix(".CR3")),
            "source_cr3_exists": image_path.with_suffix(".CR3").exists(),
            "crop_path": str(crop_path),
            "plate_label_crop_path": str(label_crop_path),
            "indexed_mask_path": str(indexed_path),
            "mask_source": mask_source,
            "single_timepoint": True,
            "expected_plants": 1,
            "pixel_size_mm": pixel_size_mm,
            "root_threshold": ROOT_THRESHOLD,
            "root_graph_threshold": ROOT_GRAPH_THRESHOLD,
            "root_probability_mean": probability_mean,
            "root_low_confidence_fraction": low_confidence_fraction,
            "lateral_branch_count": sum(1 for branch in item_branches if branch["branch_class"] == "lateral_root"),
            "adventitious_branch_count": sum(
                1 for branch in item_branches if branch["branch_class"] == "adventitious_root"
            ),
            **_prefix_metrics("primary_root", primary_metrics),
            **_prefix_metrics("lateral_root", lateral_metrics),
            **_prefix_metrics("adventitious_root", adventitious_metrics),
            **_prefix_metrics("total_root", total_metrics),
            **_prefix_metrics("shoot", shoot_metrics),
            **primary_shape,
            **shoot_color,
            **graph_meta,
            "export_stem": stem,
            "plate_number_parse_method": identity_metadata["plate_number_parse_method"],
            "plate_number_resolution_score": identity_metadata["plate_number_resolution_score"],
            "ocr_plate_text": identity_metadata["ocr_plate_text"],
            "ocr_plate_confidence": identity_metadata["ocr_plate_confidence"],
            "ocr_text": identity_metadata["ocr_text"],
        }
        row["root_to_shoot_area_ratio"] = float(
            float(row["total_root_area_px"]) / max(1.0, float(row["shoot_area_px"]))
        )
        phenotype_rows.append(row)

        for branch_index, branch in enumerate(item_branches, start=1):
            enriched = dict(branch)
            enriched.update(
                {
                    "sample_index": index,
                    "sample_stem": stem,
                    "source_image_stem": source_stem,
                    "plate_name": identity_metadata["plate_name"],
                    "plate_number": identity_metadata["plate_number"],
                    "genotype": identity_metadata["genotype"],
                    "treatment": identity_metadata["treatment"],
                    "plate_capture_index": identity_metadata["plate_capture_index"],
                    "plate_capture_count": identity_metadata["plate_capture_count"],
                    "is_duplicate_plate_capture": identity_metadata["is_duplicate_plate_capture"],
                    "branch_index": branch_index,
                    "mask_source": mask_source,
                    "pixel_size_mm": pixel_size_mm,
                    "export_stem": stem,
                    "plate_number_parse_method": identity_metadata["plate_number_parse_method"],
                    "plate_number_resolution_score": identity_metadata["plate_number_resolution_score"],
                    "ocr_plate_text": identity_metadata["ocr_plate_text"],
                    "ocr_plate_confidence": identity_metadata["ocr_plate_confidence"],
                    "ocr_text": identity_metadata["ocr_text"],
                }
            )
            enriched.setdefault(
                "branch_length_mm",
                float(enriched.get("branch_length_px", 0.0)) * pixel_size_mm,
            )
            branch_rows.append(enriched)

        qc_flags: list[str] = []
        if int(total_metrics["area_px"]) <= 0:
            qc_flags.append("empty_root_mask")
        if int(shoot_metrics["area_px"]) <= 0:
            qc_flags.append("empty_shoot_mask")
        if bool(graph_meta.get("topology_filter_applied", False)):
            qc_flags.append("topology_cleanup_applied")
        if bool(graph_meta.get("rescue_applied", False)):
            qc_flags.append("forced_segmentation_rescue")
        if mask_source == MODEL_MASK_SOURCE and float(low_confidence_fraction) > 0.65:
            qc_flags.append("low_root_confidence")
        route_length = float(graph_meta.get("primary_path_length_px", 0.0) or 0.0)
        route_gap = float(graph_meta.get("primary_route_gap_length_px", 0.0) or 0.0)
        if mask_source == MODEL_MASK_SOURCE and route_gap > max(300.0, 0.15 * route_length):
            qc_flags.append("large_primary_route_gap")
        if int(total_metrics["tips"]) > 120:
            qc_flags.append("excess_root_tips")
        qc_rows.append(
            {
                "sample_index": index,
                "sample_stem": stem,
                "source_image_stem": source_stem,
                "plate_name": identity_metadata["plate_name"],
                "plate_number": identity_metadata["plate_number"],
                "genotype": identity_metadata["genotype"],
                "treatment": identity_metadata["treatment"],
                "plate_capture_index": identity_metadata["plate_capture_index"],
                "plate_capture_count": identity_metadata["plate_capture_count"],
                "is_duplicate_plate_capture": identity_metadata["is_duplicate_plate_capture"],
                "mask_source": mask_source,
                "review_required": bool(qc_flags),
                "qc_flags": ";".join(qc_flags),
                "root_probability_mean": probability_mean,
                "root_low_confidence_fraction": low_confidence_fraction,
                "total_root_area_px": int(total_metrics["area_px"]),
                "shoot_area_px": int(shoot_metrics["area_px"]),
                "rescue_applied": bool(graph_meta.get("rescue_applied", False)),
                "rescue_method": str(graph_meta.get("rescue_method", "")),
                "rescue_spec_sha256": str(graph_meta.get("rescue_spec_sha256", "")),
                "rescue_source_jpeg_sha256": str(
                    graph_meta.get("rescue_source_jpeg_sha256", "")
                ),
                "crop_metadata": json.dumps(crop_meta, separators=(",", ":")),
                "export_stem": stem,
                "plate_number_parse_method": identity_metadata["plate_number_parse_method"],
                "plate_number_resolution_score": identity_metadata["plate_number_resolution_score"],
                "ocr_plate_text": identity_metadata["ocr_plate_text"],
                "ocr_plate_confidence": identity_metadata["ocr_plate_confidence"],
                "ocr_text": identity_metadata["ocr_text"],
            }
        )

        overlay = _render_overlay(image_rgb, masks, display_dilation=1)
        overlay_path = overlays_dir / f"{stem}__overlay.jpg"
        _save_overlay_with_legend(overlay_path, overlay, row)
        overlay_paths[stem] = overlay_path
        if index % 10 == 0 or index == len(images):
            print(f"processed {index}/{len(images)}: {stem}", flush=True)
        if callable(progress_callback):
            keep_running = progress_callback(index, len(image_entries), stem)
            if keep_running is False:
                raise PotatoEndpointCancelled("Potato endpoint package generation was canceled.")

    identity_leading = [
        "sample_index",
        "sample_stem",
        "source_image_stem",
        "plate_name",
        "plate_number",
        "genotype",
        "treatment",
        "plate_capture_index",
        "plate_capture_count",
        "is_duplicate_plate_capture",
    ]
    phenotype_df = _ordered_columns(pd.DataFrame(phenotype_rows), identity_leading)
    phenotype_tail = [
        "root_to_shoot_area_ratio",
        "rescue_applied",
        "rescue_method",
        "rescue_spec_sha256",
        "rescue_normalized_size",
        "rescue_source_jpeg_sha256",
        "export_stem",
        "plate_number_parse_method",
        "plate_number_resolution_score",
        "ocr_plate_text",
        "ocr_plate_confidence",
        "ocr_text",
    ]
    phenotype_df = phenotype_df.loc[
        :,
        [column for column in phenotype_df.columns if column not in phenotype_tail]
        + [column for column in phenotype_tail if column in phenotype_df.columns],
    ]
    branch_df = _ordered_columns(
        pd.DataFrame(branch_rows),
        identity_leading
        + [
            "branch_class",
            "attachment_x_px",
            "attachment_y_px",
            "attachment_distance_px",
            "tip_x_px",
            "tip_y_px",
            "branch_length_px",
            "branch_angle_deg",
            "branch_area_seed_px",
            "branch_index",
            "mask_source",
            "pixel_size_mm",
            "branch_length_mm",
            "export_stem",
            "plate_number_parse_method",
            "plate_number_resolution_score",
            "ocr_plate_text",
            "ocr_plate_confidence",
            "ocr_text",
        ],
    )
    qc_df = _ordered_columns(
        pd.DataFrame(qc_rows),
        identity_leading
        + [
            "mask_source",
            "review_required",
            "qc_flags",
            "root_probability_mean",
            "root_low_confidence_fraction",
            "total_root_area_px",
            "shoot_area_px",
            "crop_metadata",
            "rescue_applied",
            "rescue_method",
            "rescue_spec_sha256",
            "rescue_source_jpeg_sha256",
            "export_stem",
            "plate_number_parse_method",
            "plate_number_resolution_score",
            "ocr_plate_text",
            "ocr_plate_confidence",
            "ocr_text",
        ],
    )
    excluded_df = pd.DataFrame(
        excluded_rows,
        columns=["sample_stem", "source_path", "reason"],
    )
    plate_label_map_df = _plate_label_map(image_entries, output)
    summary_df = _dataset_summary(phenotype_df)
    phenotype_csv = output / "npec_potato_endpoint_phenotypes.csv"
    branch_csv = output / "npec_potato_branch_traits.csv"
    qc_csv = output / "npec_potato_qc.csv"
    excluded_csv = output / "npec_potato_excluded_captures.csv"
    summary_csv = output / "npec_potato_dataset_summary.csv"
    plate_map_csv = output / "npec_potato_plate_label_map.csv"
    dario_compat_plate_map_csv = output / "npec_dario_plate_label_map.csv"
    phenotype_df.to_csv(phenotype_csv, index=False)
    branch_df.to_csv(branch_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)
    excluded_df.to_csv(excluded_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    plate_label_map_df.to_csv(plate_map_csv, index=False)
    plate_label_map_df.to_csv(dario_compat_plate_map_csv, index=False)

    provenance = {
        "pipeline_version": PIPELINE_VERSION,
        "dataset": str(dataset),
        "output": str(output),
        "input_jpeg_count": len(image_entries),
        "manual_reference_count": sum(row["mask_source"] == "manual_psd" for row in phenotype_rows),
        "model_inference_count": sum(row["mask_source"] == MODEL_MASK_SOURCE for row in phenotype_rows),
        "forced_rescue_count": sum(row["mask_source"] == FORCED_MASK_SOURCE for row in phenotype_rows),
        "forced_rescues": {
            stem: {
                "method": str(spec["version"]),
                "spec_sha256": _forced_rescue_spec_sha256(spec),
                "normalized_size": spec["normalized_size"],
            }
            for stem, spec in FORCED_VESSEL_RESCUES.items()
            if enable_forced_rescues and stem in selected_stems
        },
        "one_off_forced_rescues_enabled": enable_forced_rescues,
        "generic_dataset_mode": generic_dataset,
        "excluded_capture_count": len(excluded_rows),
        "pixel_size_mm": pixel_size_mm,
        "pixel_scale_basis": "Lucifer app constant 111.88 mm / 4200 px",
        "root_model_path": str(root_model_path),
        "root_model_sha256": _sha256(root_model_path),
        "root_weights_path": str(root_weights_path),
        "root_weights_sha256": _sha256(root_weights_path),
        "root_threshold": ROOT_THRESHOLD,
        "root_graph_threshold": ROOT_GRAPH_THRESHOLD,
        "topology_cleanup_method": "conditional one-plant crown-connected component network; triggered by >120 tips or an implausible route gap",
        "shoot_method": "crown-restricted excess-green RGB mask with components retained near the main shoot",
        "primary_root_method": "crown-to-deepest minimum-cost route with measured-pixel-only class assignment",
        "crop_method": "stored JPEG pixels crop x=1728,y=720,w=4167,h=4135 then 180-degree rotation",
        "manual_classes": ["primary_root", "lateral_root", "adventitious_root", "shoot"],
        "single_timepoint_notice": "Each plate is an independent endpoint; growth rates and temporal deltas are not inferred.",
        "reference_validation": validation.get("aggregate", {}) if validation else {},
        "plate_identity_method": "NPEC plate identity scanner with reviewed OCR, QR, barcode, handwriting, filename, and folder evidence",
        "source_raw_filenames_preserved": True,
        "export_assets_use_physical_plate_names": True,
        "plate_label_map": str(plate_map_csv),
        "dario_compat_plate_label_map": str(dario_compat_plate_map_csv),
        "unique_physical_plates": int(plate_label_map_df["plate_name"].nunique()),
        "duplicate_capture_counts": {
            str(plate_name): int(group["plate_capture_count"].max())
            for plate_name, group in plate_label_map_df.groupby("plate_name", dropna=False)
            if int(group["plate_capture_count"].max()) > 1
        },
        "elapsed_seconds": float(time.time() - started),
    }
    provenance_path = output / "npec_potato_analysis_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    workbook_path = output / "npec_potato_endpoint_all_metrics.xlsx"
    _write_workbook(
        workbook_path,
        phenotype_df,
        branch_df,
        qc_df,
        excluded_df,
        summary_df,
        provenance,
        validation,
        plate_label_map_df,
    )
    _write_readme(output / "README.md", phenotype_rows, validation, pixel_size_mm)

    if not skip_videos:
        if callable(progress_callback):
            progress_callback(len(image_entries), len(image_entries), "Rendering overlay review reel")
        _write_video(
            output / "npec_potato_endpoint_overlay_reel.mp4",
            phenotype_rows,
            overlay_paths,
            mask_paths,
            video_fps,
            mask_only=False,
        )
        if callable(progress_callback):
            progress_callback(len(image_entries), len(image_entries), "Rendering mask-only review reel")
        _write_video(
            output / "npec_potato_endpoint_mask_reel.mp4",
            phenotype_rows,
            overlay_paths,
            mask_paths,
            video_fps,
            mask_only=True,
        )
        first_row = phenotype_rows[0]
        first_stem = str(first_row["sample_stem"])
        sample_overlay = np.asarray(Image.open(overlay_paths[first_stem]).convert("RGB"), dtype=np.uint8)
        Image.fromarray(_video_frame(sample_overlay, first_row, 1, len(phenotype_rows), False)).save(
            output / "npec_potato_endpoint_video_sample_frame.png"
        )

    provenance["elapsed_seconds"] = float(time.time() - started)
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    finalization_summary = {
        "package": str(output),
        "mapped_images": len(image_entries),
        "unique_physical_plates": provenance["unique_physical_plates"],
        "duplicate_capture_counts": provenance["duplicate_capture_counts"],
        "validated_assets": {
            "crops": len(list(crops_dir.glob("*.jpg"))),
            "label_crops": len(list(label_crops_dir.glob("*.jpg"))),
            "overlays": len(list(overlays_dir.glob("*.jpg"))),
            "masks": len(list(masks_dir.glob("*.png"))),
        },
        "forced_rescue_count": provenance["forced_rescue_count"],
        "one_off_forced_rescues_enabled": enable_forced_rescues,
        "elapsed_seconds": provenance["elapsed_seconds"],
    }
    (output / "npec_potato_finalization_summary.json").write_text(
        json.dumps(finalization_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2), flush=True)
    return 0


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
