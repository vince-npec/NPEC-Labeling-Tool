from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import math
import re
import tarfile

import cv2
import numpy as np

from .sensors_processing import load_fluorcam_tar_preview


FRAME_KEY_RE = re.compile(r"^(\d+_\d+)")
FILTER_RE = re.compile(r"^\s*Filter\s*=\s*([A-Za-z0-9]+)\s*$", re.MULTILINE)
ROOT_CANVAS_SHAPE = (3006, 4202)


def _require_skimage_function(module_name: str, attr_name: str):
    try:
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    except Exception as exc:  # pragma: no cover - exercised indirectly in frozen builds
        raise RuntimeError(
            "This FluorCam feature needs the optional scikit-image/SciPy runtime, "
            "which is currently unavailable in this build."
        ) from exc


@dataclass(slots=True)
class FluorCamTarMetadata:
    filter_name: str | None
    filter_offsets: dict[str, tuple[int, int]]
    fish_eye: dict[str, float] | None = None


@dataclass(slots=True)
class FluorCamAlignment:
    target_size: tuple[int, int]
    resized_shape: tuple[int, int]
    offset_x: int
    offset_y: int
    root_type: str
    fluor_channel: str
    filter_name: str | None
    source: str
    mirror_horizontal: bool = False
    mirror_source: str = "unspecified"


@dataclass(slots=True)
class FluorCamPair:
    frame_key: str
    root_image_path: Path
    fluorescence_path: Path


@dataclass(slots=True)
class GapBridge:
    start_xy: tuple[int, int]
    end_xy: tuple[int, int]
    distance_px: float
    mean_support: float
    min_support: float
    path_pixels: int
    score: float


KNOWN_ALIGNMENT_PRESETS: dict[tuple[str, str, str], FluorCamAlignment] = {
    ("ROOT1", "FC1", "F483"): FluorCamAlignment(
        target_size=ROOT_CANVAS_SHAPE,
        resized_shape=(3005, 4201),
        offset_x=-1,
        offset_y=9,
        root_type="ROOT1",
        fluor_channel="FC1",
        filter_name="F483",
        source="hades-script-preset",
    ),
    ("ROOT1", "FC1", "F513"): FluorCamAlignment(
        target_size=ROOT_CANVAS_SHAPE,
        resized_shape=(3005, 4201),
        offset_x=0,
        offset_y=0,
        root_type="ROOT1",
        fluor_channel="FC1",
        filter_name="F513",
        source="hades-script-preset",
    ),
    ("ROOT2", "FC2", "F635"): FluorCamAlignment(
        target_size=ROOT_CANVAS_SHAPE,
        resized_shape=(3020, 4222),
        offset_x=-79,
        offset_y=-59,
        root_type="ROOT2",
        fluor_channel="FC2",
        filter_name="F635",
        source="hades-script-preset",
    ),
}


BASE_ALIGNMENT_PRESETS: dict[tuple[str, str], FluorCamAlignment] = {
    ("ROOT1", "FC1"): KNOWN_ALIGNMENT_PRESETS[("ROOT1", "FC1", "F513")],
    ("ROOT2", "FC2"): KNOWN_ALIGNMENT_PRESETS[("ROOT2", "FC2", "F635")],
}


def extract_frame_key(path: Path) -> str | None:
    match = FRAME_KEY_RE.match(path.name)
    return match.group(1) if match else None


def infer_root_type(path: Path) -> str:
    name = str(path).upper()
    if "ROOT2" in name:
        return "ROOT2"
    return "ROOT1"


def infer_fluorcam_channel(path: Path) -> str:
    name = str(path).upper()
    if "FC2" in name:
        return "FC2"
    return "FC1"


def _read_tar_text(path: Path, suffix: str) -> str | None:
    source = Path(path)
    suffix_lower = suffix.lower()
    with tarfile.open(source, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            if member.name.lower().endswith(suffix_lower):
                handle = tf.extractfile(member)
                if handle is None:
                    return None
                raw = handle.read()
                return raw.decode("utf-8", errors="ignore")
    return None


def _read_tar_json(path: Path, suffix: str) -> object | None:
    text = _read_tar_text(path, suffix)
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def read_fluorcam_tar_metadata(path: Path) -> FluorCamTarMetadata:
    protocol = _read_tar_text(path, "protocol") or ""
    match = FILTER_RE.search(protocol)
    filter_name = match.group(1).upper() if match else None

    filter_names_raw = _read_tar_json(path, "filter-names.json")
    filter_offsets_raw = _read_tar_json(path, "filter-offsets.json")
    fish_eye_raw = _read_tar_json(path, "fish-eye.json")

    filter_names: list[str] = []
    if isinstance(filter_names_raw, list):
        filter_names = [str(item).upper() for item in filter_names_raw]

    offsets: dict[str, tuple[int, int]] = {}
    if isinstance(filter_offsets_raw, list):
        for entry in filter_offsets_raw:
            if not isinstance(entry, dict):
                continue
            try:
                idx = int(entry.get("Index", -1))
            except Exception:
                continue
            if idx < 0 or idx >= len(filter_names):
                continue
            key = filter_names[idx]
            try:
                offsets[key] = (int(entry.get("OffsetX", 0)), int(entry.get("OffsetY", 0)))
            except Exception:
                offsets[key] = (0, 0)

    fish_eye: dict[str, float] | None = None
    if isinstance(fish_eye_raw, dict):
        parsed_fish_eye: dict[str, float] = {}
        for key, value in fish_eye_raw.items():
            try:
                parsed_fish_eye[str(key)] = float(value)
            except Exception:
                continue
        fish_eye = parsed_fish_eye or None

    return FluorCamTarMetadata(filter_name=filter_name, filter_offsets=offsets, fish_eye=fish_eye)


def pair_root_and_fluorcam_series(root_dir: Path, fluorescence_dir: Path) -> list[FluorCamPair]:
    root_dir = Path(root_dir)
    fluorescence_dir = Path(fluorescence_dir)
    roots_by_key: dict[str, Path] = {}
    fluor_by_key: dict[str, Path] = {}
    for path in sorted(root_dir.iterdir()):
        if path.is_file():
            key = extract_frame_key(path)
            if key:
                roots_by_key[key] = path
    for path in sorted(fluorescence_dir.iterdir()):
        if path.is_file():
            key = extract_frame_key(path)
            if key:
                fluor_by_key[key] = path
    shared = sorted(set(roots_by_key) & set(fluor_by_key), key=_natural_frame_sort_key)
    return [FluorCamPair(frame_key=key, root_image_path=roots_by_key[key], fluorescence_path=fluor_by_key[key]) for key in shared]


def _natural_frame_sort_key(frame_key: str) -> tuple[int, int]:
    left, _, right = frame_key.partition("_")
    try:
        return (int(left), int(right))
    except Exception:
        return (0, 0)


def _to_grayscale(base_image: np.ndarray) -> np.ndarray:
    image = np.asarray(base_image)
    if image.ndim == 2:
        return np.asarray(image, dtype=np.uint8)
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(np.asarray(image[:, :, :3], dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Unsupported base image shape: {image.shape}")


def _normalize01(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros_like(data, dtype=np.float32)
    lo = float(np.min(data[finite]))
    hi = float(np.max(data[finite]))
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    out = (data - lo) / (hi - lo)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0)


def align_mask_to_fixed_canvas(
    input_image: np.ndarray,
    *,
    target_size: tuple[int, int] = ROOT_CANVAS_SHAPE,
    resized_shape: tuple[int, int] = ROOT_CANVAS_SHAPE,
    offset_x: int = 0,
    offset_y: int = 0,
) -> np.ndarray:
    resize = _require_skimage_function("skimage.transform", "resize")
    resized_img = resize(
        np.asarray(input_image, dtype=np.float32),
        resized_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.asarray(input_image).dtype)

    canvas = np.zeros(target_size, dtype=resized_img.dtype)
    h_resized, w_resized = resized_shape
    h_canvas, w_canvas = target_size

    start_y = max(0, offset_y)
    start_x = max(0, offset_x)
    end_y = min(offset_y + h_resized, h_canvas)
    end_x = min(offset_x + w_resized, w_canvas)

    src_y = 0 if offset_y >= 0 else -offset_y
    src_x = 0 if offset_x >= 0 else -offset_x
    cropped_h = max(0, end_y - start_y)
    cropped_w = max(0, end_x - start_x)
    if cropped_h <= 0 or cropped_w <= 0:
        return canvas

    canvas[start_y:end_y, start_x:end_x] = resized_img[src_y : src_y + cropped_h, src_x : src_x + cropped_w]
    return canvas


def _load_root_gray_image(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return np.asarray(image, dtype=np.uint8)
    if image.ndim == 3:
        return cv2.cvtColor(np.asarray(image[:, :, :3], dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    return None


def _score_aligned_fluorcam_against_root_gray(root_gray: np.ndarray, aligned_fluor: np.ndarray) -> float:
    gray = np.asarray(root_gray, dtype=np.uint8)
    aligned = np.asarray(aligned_fluor, dtype=np.float32)
    if gray.shape != aligned.shape:
        return float("-inf")
    plate_mask = estimate_plate_region(gray)
    if not np.any(plate_mask):
        return float("-inf")
    signal = aligned.copy()
    signal[~plate_mask] = 0.0
    active = signal[plate_mask]
    if active.size == 0:
        return float("-inf")
    lo, hi = np.percentile(active, [90, 99.9])
    if hi <= lo:
        return float("-inf")
    support = np.clip((signal - float(lo)) / float(hi - lo), 0.0, 1.0)
    root_darkness = (255.0 - gray.astype(np.float32)) / 255.0
    weighted = support * root_darkness * plate_mask.astype(np.float32)
    denom = float(np.sum(support * plate_mask.astype(np.float32)))
    if denom <= 1e-6:
        return float("-inf")
    return float(np.sum(weighted) / denom)


def choose_fluorcam_mirror_orientation(
    frame: np.ndarray,
    *,
    alignment: FluorCamAlignment,
    root_gray: np.ndarray | None,
) -> tuple[bool, str]:
    if root_gray is None:
        return True, "legacy-default"
    candidates = (
        (False, "auto-no-flip", np.asarray(frame, dtype=np.float32)),
        (True, "auto-flip-horizontal", np.fliplr(frame)),
    )
    best_mirror = True
    best_source = "legacy-default"
    best_score = float("-inf")
    for mirror_horizontal, source, candidate in candidates:
        aligned = align_mask_to_fixed_canvas(
            candidate,
            target_size=alignment.target_size,
            resized_shape=alignment.resized_shape,
            offset_x=alignment.offset_x,
            offset_y=alignment.offset_y,
        )
        score = _score_aligned_fluorcam_against_root_gray(root_gray, aligned)
        if score > best_score:
            best_score = score
            best_mirror = mirror_horizontal
            best_source = source
    return best_mirror, best_source


def resolve_fluorcam_alignment(
    *,
    root_type: str,
    fluor_channel: str,
    metadata: FluorCamTarMetadata,
) -> FluorCamAlignment:
    filter_name = (metadata.filter_name or "").upper() or None
    exact = KNOWN_ALIGNMENT_PRESETS.get((root_type, fluor_channel, filter_name or ""))
    if exact is not None:
        return exact

    base = BASE_ALIGNMENT_PRESETS.get((root_type, fluor_channel))
    if base is None:
        return FluorCamAlignment(
            target_size=ROOT_CANVAS_SHAPE,
            resized_shape=ROOT_CANVAS_SHAPE,
            offset_x=0,
            offset_y=0,
            root_type=root_type,
            fluor_channel=fluor_channel,
            filter_name=filter_name,
            source="fallback-full-canvas",
        )

    derived_x = int(base.offset_x)
    derived_y = int(base.offset_y)
    source = "base-root-channel-preset"
    anchor_filter = base.filter_name
    if (
        filter_name
        and anchor_filter
        and filter_name in metadata.filter_offsets
        and anchor_filter in metadata.filter_offsets
    ):
        cur_x, cur_y = metadata.filter_offsets[filter_name]
        anchor_x, anchor_y = metadata.filter_offsets[anchor_filter]
        derived_x += int(cur_x - anchor_x)
        derived_y += int(cur_y - anchor_y)
        source = "preset-plus-filter-offset-delta"

    return FluorCamAlignment(
        target_size=base.target_size,
        resized_shape=base.resized_shape,
        offset_x=derived_x,
        offset_y=derived_y,
        root_type=root_type,
        fluor_channel=fluor_channel,
        filter_name=filter_name,
        source=source,
    )


def load_aligned_fluorcam_frame(
    fluorescence_path: Path,
    *,
    root_image_path: Path | None = None,
    mirror_horizontal: bool | None = None,
) -> tuple[np.ndarray, FluorCamTarMetadata, FluorCamAlignment]:
    source = Path(fluorescence_path)
    metadata = read_fluorcam_tar_metadata(source)
    root_type = infer_root_type(root_image_path or source)
    fluor_channel = infer_fluorcam_channel(source)
    alignment = resolve_fluorcam_alignment(root_type=root_type, fluor_channel=fluor_channel, metadata=metadata)

    frame = np.asarray(load_fluorcam_tar_preview(source), dtype=np.float32)
    target_w = int(alignment.target_size[1])
    if frame.shape[1] < target_w:
        pad = np.zeros((frame.shape[0], target_w - frame.shape[1]), dtype=frame.dtype)
        frame = np.concatenate([frame, pad], axis=1)
    root_gray = _load_root_gray_image(root_image_path)
    if mirror_horizontal is None:
        mirror_horizontal, mirror_source = choose_fluorcam_mirror_orientation(
            frame,
            alignment=alignment,
            root_gray=root_gray,
        )
    else:
        mirror_source = "manual-override"
    if mirror_horizontal:
        frame = np.fliplr(frame)
    aligned = align_mask_to_fixed_canvas(
        frame,
        target_size=alignment.target_size,
        resized_shape=alignment.resized_shape,
        offset_x=alignment.offset_x,
        offset_y=alignment.offset_y,
    )
    resolved_alignment = replace(
        alignment,
        mirror_horizontal=bool(mirror_horizontal),
        mirror_source=str(mirror_source),
    )
    return aligned.astype(np.float32), metadata, resolved_alignment


def _largest_component(binary_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(binary_mask > 0, dtype=np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask.astype(bool)
    idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
    return labels == idx


def estimate_plate_region(base_gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(base_gray, dtype=np.uint8)
    threshold = int(np.percentile(gray, 25))
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    largest = _largest_component(closed)
    largest_u8 = np.asarray(largest, dtype=np.uint8) * 255
    largest_u8 = cv2.morphologyEx(largest_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    return largest_u8 > 0


def compute_fluorescence_support_map(
    base_image: np.ndarray,
    fluorescence_frame: np.ndarray,
    *,
    plate_mask: np.ndarray | None = None,
) -> np.ndarray:
    frangi = _require_skimage_function("skimage.filters", "frangi")
    base_gray = _to_grayscale(base_image)
    fluor = np.asarray(fluorescence_frame, dtype=np.float32)
    if fluor.shape != base_gray.shape:
        fluor = cv2.resize(fluor, (base_gray.shape[1], base_gray.shape[0]), interpolation=cv2.INTER_LINEAR)

    if plate_mask is None:
        plate_mask = estimate_plate_region(base_gray)
    plate_mask_bool = np.asarray(plate_mask > 0, dtype=bool)
    if np.any(plate_mask_bool):
        interior_fill = float(np.median(fluor[plate_mask_bool]))
        fluor = np.where(plate_mask_bool, fluor, interior_fill)

    dark_response = cv2.morphologyEx(base_gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    dark_response_f = _normalize01(dark_response)

    fluor_u8 = _normalize01(fluor) * 255.0
    fluor_u8 = np.asarray(fluor_u8, dtype=np.uint8)
    fluor_tophat = cv2.morphologyEx(fluor_u8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    fluor_ridge = frangi(np.asarray(fluor_tophat, dtype=np.float32) / 255.0, sigmas=range(1, 5), black_ridges=False)
    fluor_ridge_f = _normalize01(fluor_ridge)

    dark_ridge = frangi(np.asarray(dark_response, dtype=np.float32) / 255.0, sigmas=range(1, 4), black_ridges=False)
    dark_ridge_f = _normalize01(dark_ridge)

    support = np.maximum(0.65 * dark_ridge_f + 0.35 * fluor_ridge_f, dark_response_f * 0.55 + fluor_ridge_f * 0.45)
    support = np.where(plate_mask_bool, support, 0.0)
    return np.asarray(np.clip(support, 0.0, 1.0), dtype=np.float32)


def load_pair_support_map(pair: FluorCamPair) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = cv2.imread(str(pair.root_image_path), cv2.IMREAD_UNCHANGED)
    if root is None:
        raise ValueError(f"Could not read root image: {pair.root_image_path}")
    fluor, _metadata, _alignment = load_aligned_fluorcam_frame(
        pair.fluorescence_path,
        root_image_path=pair.root_image_path,
    )
    support = compute_fluorescence_support_map(root, fluor)
    return root, fluor, support


def _skeleton_endpoints(skeleton: np.ndarray) -> list[tuple[int, int]]:
    skel = np.asarray(skeleton > 0, dtype=np.uint8)
    neighbor_kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.int16,
    )
    neighbor_count = cv2.filter2D(skel.astype(np.int16), cv2.CV_16S, neighbor_kernel, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.where((skel > 0) & (neighbor_count == 1))
    return [(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())]


def _endpoint_direction(skeleton_mask: np.ndarray, endpoint_rc: tuple[int, int], max_steps: int = 12) -> np.ndarray:
    y, x = endpoint_rc
    current = (int(y), int(x))
    previous: tuple[int, int] | None = None
    points = [current]
    for _ in range(max_steps):
        cy, cx = current
        neighbors: list[tuple[int, int]] = []
        for ny in range(max(0, cy - 1), min(skeleton_mask.shape[0], cy + 2)):
            for nx in range(max(0, cx - 1), min(skeleton_mask.shape[1], cx + 2)):
                if (ny, nx) == current or (previous is not None and (ny, nx) == previous):
                    continue
                if skeleton_mask[ny, nx]:
                    neighbors.append((ny, nx))
        if not neighbors:
            break
        next_point = neighbors[0]
        points.append(next_point)
        previous = current
        current = next_point
    if len(points) < 2:
        return np.zeros(2, dtype=np.float32)
    py, px = points[1]
    direction = np.array([float(x - px), float(y - py)], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return np.zeros(2, dtype=np.float32)
    return direction / norm


def _component_endpoint_map(skeleton_mask: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    labels_count, labels = cv2.connectedComponents(np.asarray(skeleton_mask > 0, dtype=np.uint8), connectivity=8)
    endpoints = _skeleton_endpoints(skeleton_mask)
    out: dict[int, list[tuple[int, int]]] = {}
    for y, x in endpoints:
        label = int(labels[y, x])
        if label <= 0:
            continue
        out.setdefault(label, []).append((y, x))
    if labels_count <= 1:
        return {}
    return out


def _bridge_path(
    support_map: np.ndarray,
    start_rc: tuple[int, int],
    end_rc: tuple[int, int],
    max_gap_px: float,
) -> tuple[np.ndarray, float, float]:
    route_through_array = _require_skimage_function("skimage.graph", "route_through_array")
    y0, x0 = start_rc
    y1, x1 = end_rc
    margin = max(12, int(max_gap_px * 0.35))
    y_min = max(0, min(y0, y1) - margin)
    y_max = min(support_map.shape[0], max(y0, y1) + margin + 1)
    x_min = max(0, min(x0, x1) - margin)
    x_max = min(support_map.shape[1], max(x0, x1) + margin + 1)
    local = np.asarray(support_map[y_min:y_max, x_min:x_max], dtype=np.float32)
    cost = 1.0 - np.clip(local, 0.0, 1.0)
    cost = np.maximum(cost, 0.02)
    indices, total_cost = route_through_array(
        cost,
        (int(y0 - y_min), int(x0 - x_min)),
        (int(y1 - y_min), int(x1 - x_min)),
        fully_connected=True,
    )
    path = np.asarray([(int(r + y_min), int(c + x_min)) for r, c in indices], dtype=np.int32)
    supports = support_map[path[:, 0], path[:, 1]]
    return path, float(np.mean(supports)), float(np.min(supports))


def repair_root_gaps_with_fluorescence(
    root_mask: np.ndarray,
    support_map: np.ndarray,
    *,
    max_gap_px: int = 140,
    min_mean_support: float = 0.18,
    min_endpoint_facing: float = 0.45,
    max_bridges: int = 8,
) -> tuple[np.ndarray, list[GapBridge]]:
    skeletonize = _require_skimage_function("skimage.morphology", "skeletonize")
    repaired = np.asarray(root_mask > 0, dtype=np.uint8)
    support = np.asarray(np.clip(support_map, 0.0, 1.0), dtype=np.float32)
    accepted: list[GapBridge] = []

    for _ in range(int(max_bridges)):
        skeleton = np.asarray(skeletonize(repaired > 0), dtype=np.uint8)
        endpoint_map = _component_endpoint_map(skeleton)
        if len(endpoint_map) < 2:
            break

        candidates: list[tuple[float, np.ndarray, GapBridge]] = []
        labels = sorted(endpoint_map)
        for i, label_a in enumerate(labels):
            for label_b in labels[i + 1 :]:
                for start_rc in endpoint_map.get(label_a, []):
                    dir_a = _endpoint_direction(skeleton, start_rc)
                    if not np.any(dir_a):
                        continue
                    for end_rc in endpoint_map.get(label_b, []):
                        dist = float(math.hypot(end_rc[1] - start_rc[1], end_rc[0] - start_rc[0]))
                        if dist <= 1.0 or dist > float(max_gap_px):
                            continue
                        dir_b = _endpoint_direction(skeleton, end_rc)
                        if not np.any(dir_b):
                            continue
                        vec_ab = np.array([float(end_rc[1] - start_rc[1]), float(end_rc[0] - start_rc[0])], dtype=np.float32)
                        vec_ba = -vec_ab
                        vec_ab /= max(np.linalg.norm(vec_ab), 1e-6)
                        vec_ba /= max(np.linalg.norm(vec_ba), 1e-6)
                        facing = float(np.dot(dir_a, vec_ab) + np.dot(dir_b, vec_ba)) * 0.5
                        if facing < float(min_endpoint_facing):
                            continue

                        try:
                            path, mean_support, min_support = _bridge_path(support, start_rc, end_rc, float(max_gap_px))
                        except Exception:
                            continue
                        if mean_support < float(min_mean_support):
                            continue
                        length_penalty = float(len(path)) / max(dist, 1.0)
                        score = mean_support + 0.20 * facing - 0.05 * max(0.0, length_penalty - 1.2)
                        bridge = GapBridge(
                            start_xy=(int(start_rc[1]), int(start_rc[0])),
                            end_xy=(int(end_rc[1]), int(end_rc[0])),
                            distance_px=dist,
                            mean_support=mean_support,
                            min_support=min_support,
                            path_pixels=int(len(path)),
                            score=score,
                        )
                        candidates.append((score, path, bridge))

        if not candidates:
            break

        candidates.sort(key=lambda item: item[0], reverse=True)
        _best_score, best_path, best_bridge = candidates[0]
        for y, x in best_path.tolist():
            repaired[int(y), int(x)] = 1
        accepted.append(best_bridge)

    return repaired.astype(np.uint8), accepted


def assign_repaired_pixels_to_root_classes(
    prediction_index: np.ndarray,
    repaired_root_mask: np.ndarray,
    *,
    root_class_id: int,
    lateral_class_id: int | None = None,
) -> tuple[np.ndarray, int, int]:
    pred_u8 = np.asarray(prediction_index, dtype=np.uint8)
    class_ids = [int(root_class_id)]
    if lateral_class_id is not None and int(lateral_class_id) > 0 and int(lateral_class_id) != int(root_class_id):
        class_ids.append(int(lateral_class_id))

    existing_rootish = np.isin(pred_u8, np.asarray(class_ids, dtype=np.uint8))
    repaired_bin = np.asarray(repaired_root_mask > 0, dtype=bool)
    added_mask = np.logical_and(repaired_bin, ~existing_rootish)
    if not np.any(added_mask):
        return pred_u8.copy(), 0, 0

    out = pred_u8.copy()
    root_mask = pred_u8 == np.uint8(root_class_id)
    lateral_mask = (
        pred_u8 == np.uint8(lateral_class_id)
        if lateral_class_id is not None and int(lateral_class_id) > 0 and int(lateral_class_id) != int(root_class_id)
        else np.zeros_like(pred_u8, dtype=bool)
    )

    lateral_added = 0
    if np.any(lateral_mask) and np.any(root_mask):
        root_dist = cv2.distanceTransform((~root_mask).astype(np.uint8), cv2.DIST_L2, 5)
        lateral_dist = cv2.distanceTransform((~lateral_mask).astype(np.uint8), cv2.DIST_L2, 5)
        assign_lateral = np.logical_and(added_mask, lateral_dist < root_dist)
        assign_root = np.logical_and(added_mask, ~assign_lateral)
        out[assign_root] = np.uint8(root_class_id)
        out[assign_lateral] = np.uint8(lateral_class_id)
        lateral_added = int(np.count_nonzero(assign_lateral))
    else:
        out[added_mask] = np.uint8(root_class_id)

    root_added = int(np.count_nonzero(added_mask) - lateral_added)
    return out, root_added, lateral_added


def render_support_overlay(base_image: np.ndarray, support_map: np.ndarray) -> np.ndarray:
    base = np.asarray(base_image)
    if base.ndim == 2:
        rgb = cv2.cvtColor(np.asarray(base, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        rgb = np.asarray(base[:, :, :3], dtype=np.uint8)
    support_u8 = np.asarray(np.clip(support_map, 0.0, 1.0) * 255.0, dtype=np.uint8)
    overlay = rgb.copy()
    overlay[:, :, 1] = np.maximum(overlay[:, :, 1], support_u8)
    return overlay
