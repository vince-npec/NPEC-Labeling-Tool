from __future__ import annotations

from dataclasses import dataclass, replace
import math

import cv2
import numpy as np


_DEFAULT_MIN_RESPONSE = 0.08
_DEFAULT_MAX_SHIFT_FRACTION = 0.20
_DEFAULT_MAX_ROTATION_DEG = 4.0
_DEFAULT_MAX_SCALE_DEVIATION = 0.04


@dataclass(slots=True)
class StabilizationConfig:
    reference_mode: str = "previous"
    target_mode: str = "plant_top"
    downscale_max_dim: int = 1600
    border_fraction: float = 0.14
    top_fraction: float = 0.08
    plant_top_fraction: float = 0.055
    plant_bottom_fraction: float = 0.58
    plant_side_margin_fraction: float = 0.065
    blur_sigma: float = 1.2
    min_response: float = _DEFAULT_MIN_RESPONSE
    max_shift_fraction: float = _DEFAULT_MAX_SHIFT_FRACTION
    enable_feature_fallback: bool = True
    feature_max_keypoints: int = 1800
    feature_min_matches: int = 10
    feature_min_inlier_ratio: float = 0.45
    feature_ratio_test: float = 0.78
    max_rotation_deg: float = _DEFAULT_MAX_ROTATION_DEG
    max_scale_deviation: float = _DEFAULT_MAX_SCALE_DEVIATION
    min_affine_alignment_score: float = 0.32
    min_affine_alignment_gain: float = 0.018
    affine_rotation_selection_deg: float = 0.30
    enable_stable_frame_fallback: bool = True
    enable_stable_frame_affine_fallback: bool = True
    stable_frame_min_response: float = 0.12
    stable_frame_min_alignment_score: float = 0.60
    stable_frame_min_alignment_gain: float = 0.015
    stable_frame_biological_tolerance: float = 0.003
    stable_frame_max_shift_fraction: float = 0.08
    stable_frame_max_rotation_deg: float = 2.0
    enable_biological_motion_veto: bool = True
    biological_motion_veto_min_identity_score: float = 0.72
    biological_motion_veto_min_score_drop: float = 0.035
    biological_motion_veto_min_displacement_px: float = 1.5


@dataclass(slots=True)
class StabilizationResult:
    warped_image: np.ndarray
    warp_matrix: np.ndarray
    shift_x: float
    shift_y: float
    rotation_deg: float
    score: float
    method: str
    status: str

    @property
    def accepted(self) -> bool:
        return self.status == "ok"


@dataclass(slots=True)
class _AffineCandidate:
    matrix: np.ndarray
    score: float
    rotation_deg: float
    scale: float
    inlier_count: int
    inlier_ratio: float
    median_error: float


@dataclass(slots=True)
class _StableFrameCandidate:
    matrix: np.ndarray
    score: float
    rotation_deg: float
    method: str


def _identity_result(
    moving_image: np.ndarray,
    *,
    status: str,
    score: float = 0.0,
) -> StabilizationResult:
    safe_score = float(score)
    if not math.isfinite(safe_score):
        safe_score = 0.0
    return StabilizationResult(
        warped_image=np.asarray(moving_image, dtype=np.uint8).copy(),
        warp_matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        shift_x=0.0,
        shift_y=0.0,
        rotation_deg=0.0,
        score=safe_score,
        method="identity",
        status=status,
    )


def next_stabilization_reference(
    current_reference: np.ndarray,
    result: StabilizationResult,
    config: StabilizationConfig,
) -> np.ndarray:
    """Keep the last accepted reference when chaining previous-frame registration."""
    reference_mode = str(getattr(config, "reference_mode", "previous") or "previous").strip().lower()
    if reference_mode == "previous" and result.accepted:
        return result.warped_image
    return current_reference


def build_registration_mask(shape_hw: tuple[int, int], config: StabilizationConfig) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    if h <= 0 or w <= 0:
        return mask
    target_mode = str(getattr(config, "target_mode", "plant_top") or "plant_top").strip().lower()
    if target_mode in {"plant", "plants", "plant_top", "top_roots", "biological", "bio"}:
        margin_x = max(24, min(w // 3, int(round(w * float(max(0.02, min(0.24, config.plant_side_margin_fraction)))))))
        y0 = max(0, min(h - 1, int(round(h * float(max(0.0, min(0.30, config.plant_top_fraction)))))))
        y1 = max(y0 + 1, min(h, int(round(h * float(max(0.18, min(0.90, config.plant_bottom_fraction)))))))
        x0 = max(0, min(w, margin_x))
        x1 = max(x0, min(w, w - margin_x))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
        return mask

    margin_x = max(24, int(round(w * float(max(0.04, min(0.30, config.border_fraction))))))
    margin_y = max(24, int(round(h * float(max(0.04, min(0.30, config.border_fraction))))))
    top_h = max(margin_y, int(round(h * float(max(0.04, min(0.25, config.top_fraction))))))
    mask[:margin_y, :] = 255
    mask[max(0, h - margin_y) :, :] = 255
    mask[:, :margin_x] = 255
    mask[:, max(0, w - margin_x) :] = 255
    mask[:top_h, :] = 255
    return mask


def _normalize_float01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.5))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        hi = float(np.max(finite))
        lo = float(np.min(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / float(hi - lo), 0.0, 1.0).astype(np.float32)


def _preprocess_frame_registration_image(image_rgb: np.ndarray, config: StabilizationConfig) -> np.ndarray:
    image_u8 = np.asarray(image_rgb, dtype=np.uint8)
    if image_u8.ndim == 3 and image_u8.shape[2] >= 3:
        gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.asarray(image_u8, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    sigma = float(max(0.0, min(4.0, config.blur_sigma)))
    if sigma > 0.05:
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap = np.abs(lap)
    lap = cv2.normalize(lap, None, 0.0, 1.0, cv2.NORM_MINMAX)
    gray_f = gray.astype(np.float32) / 255.0
    return (0.55 * gray_f + 0.45 * lap).astype(np.float32)


def _preprocess_plant_registration_image(image_rgb: np.ndarray, config: StabilizationConfig) -> np.ndarray:
    image_u8 = np.asarray(image_rgb, dtype=np.uint8)
    if image_u8.ndim == 2:
        rgb = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2RGB)
    elif image_u8.ndim == 3 and image_u8.shape[2] >= 3:
        rgb = image_u8[:, :, :3]
    else:
        return _preprocess_frame_registration_image(image_u8, config)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sigma = float(max(0.0, min(4.0, config.blur_sigma)))
    if sigma > 0.05:
        gray_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    else:
        gray_blur = gray

    lap = np.abs(cv2.Laplacian(gray_blur, cv2.CV_32F, ksize=3))
    lap = _normalize_float01(lap)

    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    total = np.maximum(1, r + g + b).astype(np.float32)
    green_fraction = g.astype(np.float32) / total
    red_fraction = r.astype(np.float32) / total
    blue_fraction = b.astype(np.float32) / total
    green_chroma = green_fraction - np.maximum(red_fraction, blue_fraction)
    exg = ((2 * g) - r - b).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.int16)
    saturation = hsv[..., 1].astype(np.float32)
    value = hsv[..., 2].astype(np.float32)
    green_like = (
        (hue >= 24)
        & (hue <= 96)
        & (saturation >= 26)
        & (value >= 24)
        & (value <= 220)
        & (g >= b + 6)
        & (green_fraction >= 0.340)
        & (green_chroma >= 0.000)
        & (exg >= 8)
    ).astype(np.uint8)
    green_like = cv2.morphologyEx(
        green_like,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    green_score = cv2.GaussianBlur(green_like.astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=2.0)
    green_score = _normalize_float01(green_score)

    dark_plant_texture = _normalize_float01(np.maximum(0.0, 190.0 - gray.astype(np.float32)))
    signal = (0.58 * green_score) + (0.28 * lap) + (0.14 * dark_plant_texture)
    return np.clip(signal, 0.0, 1.0).astype(np.float32)


def _preprocess_registration_image(image_rgb: np.ndarray, config: StabilizationConfig) -> np.ndarray:
    target_mode = str(getattr(config, "target_mode", "plant_top") or "plant_top").strip().lower()
    if target_mode in {"plant", "plants", "plant_top", "top_roots", "biological", "bio"}:
        return _preprocess_plant_registration_image(image_rgb, config)
    return _preprocess_frame_registration_image(image_rgb, config)


def _downscale_for_registration(
    image_f32: np.ndarray,
    mask_u8: np.ndarray,
    max_dim: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    h, w = image_f32.shape[:2]
    longest = max(h, w)
    scale = 1.0
    if longest > int(max(64, max_dim)):
        scale = float(max_dim) / float(longest)
        new_size = (max(64, int(round(w * scale))), max(64, int(round(h * scale))))
        image_f32 = cv2.resize(image_f32, new_size, interpolation=cv2.INTER_AREA)
        mask_u8 = cv2.resize(mask_u8, new_size, interpolation=cv2.INTER_NEAREST)
    return image_f32, mask_u8, float(scale)


def _config_float(config: StabilizationConfig, name: str, default: float) -> float:
    try:
        value = float(getattr(config, name, default))
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _is_plant_target(config: StabilizationConfig) -> bool:
    target_mode = str(getattr(config, "target_mode", "plant_top") or "plant_top").strip().lower()
    return target_mode in {"plant", "plants", "plant_top", "top_roots", "biological", "bio"}


def _feature_detection_image(image_rgb: np.ndarray, config: StabilizationConfig) -> np.ndarray:
    image_u8 = np.asarray(image_rgb, dtype=np.uint8)
    if image_u8.ndim == 3 and image_u8.shape[2] >= 3:
        gray = cv2.cvtColor(image_u8[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = np.asarray(image_u8, dtype=np.uint8)
    gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    if not _is_plant_target(config):
        return gray

    biological = _preprocess_plant_registration_image(image_u8, config)
    biological_u8 = np.clip(np.rint(biological * 255.0), 0, 255).astype(np.uint8)
    return cv2.addWeighted(gray, 0.42, biological_u8, 0.58, 0.0)


def _masked_texture_std(image_f32: np.ndarray, mask_u8: np.ndarray) -> float:
    active = mask_u8 > 0
    if int(np.count_nonzero(active)) < 64:
        return 0.0
    values = np.asarray(image_f32, dtype=np.float32)[active]
    if values.size < 64:
        return 0.0
    return float(np.std(values))


def _alignment_score(
    reference_f32: np.ndarray,
    moving_f32: np.ndarray,
    mask_u8: np.ndarray,
    matrix: np.ndarray,
) -> float:
    h, w = reference_f32.shape[:2]
    matrix_f32 = np.asarray(matrix, dtype=np.float32).reshape(2, 3)
    warped = cv2.warpAffine(
        np.asarray(moving_f32, dtype=np.float32),
        matrix_f32,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    valid = cv2.warpAffine(
        np.ones((h, w), dtype=np.uint8),
        matrix_f32,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    active = (mask_u8 > 0) & (valid > 0)
    if int(np.count_nonzero(active)) < 128:
        return 0.0
    reference_values = np.asarray(reference_f32, dtype=np.float32)[active]
    moving_values = warped[active]
    reference_values = reference_values - float(np.mean(reference_values))
    moving_values = moving_values - float(np.mean(moving_values))
    denominator = float(np.linalg.norm(reference_values) * np.linalg.norm(moving_values))
    if not math.isfinite(denominator) or denominator <= 1.0e-8:
        return 0.0
    correlation = float(np.dot(reference_values, moving_values) / denominator)
    if not math.isfinite(correlation):
        return 0.0
    return float(max(0.0, min(1.0, correlation)))


def _overlap_fraction(mask_u8: np.ndarray, matrix: np.ndarray) -> float:
    h, w = mask_u8.shape[:2]
    active_count = int(np.count_nonzero(mask_u8))
    if active_count <= 0:
        return 0.0
    valid = cv2.warpAffine(
        np.ones((h, w), dtype=np.uint8),
        np.asarray(matrix, dtype=np.float32).reshape(2, 3),
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return float(np.count_nonzero((mask_u8 > 0) & (valid > 0))) / float(active_count)


def _affine_geometry(matrix: np.ndarray) -> tuple[float, float, float]:
    transform = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    scale_x = math.hypot(float(transform[0, 0]), float(transform[1, 0]))
    scale_y = math.hypot(float(transform[0, 1]), float(transform[1, 1]))
    scale = 0.5 * (scale_x + scale_y)
    anisotropy = abs(scale_x - scale_y)
    rotation_deg = math.degrees(math.atan2(float(transform[1, 0]), float(transform[0, 0])))
    return float(rotation_deg), float(scale), float(anisotropy)


def _transform_rejection_status(
    matrix: np.ndarray,
    shape_hw: tuple[int, int],
    config: StabilizationConfig,
) -> str | None:
    transform = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    if not np.all(np.isfinite(transform)):
        return "rejected_non_finite"

    rotation_deg, scale, anisotropy = _affine_geometry(transform)
    max_rotation = max(0.0, _config_float(config, "max_rotation_deg", _DEFAULT_MAX_ROTATION_DEG))
    if abs(rotation_deg) > max_rotation:
        return "rejected_large_rotation"
    max_scale_deviation = max(
        0.0,
        _config_float(config, "max_scale_deviation", _DEFAULT_MAX_SCALE_DEVIATION),
    )
    if abs(scale - 1.0) > max_scale_deviation or anisotropy > max_scale_deviation:
        return "rejected_scale_change"

    h, w = int(shape_hw[0]), int(shape_hw[1])
    max_shift_fraction = _config_float(
        config,
        "max_shift_fraction",
        _DEFAULT_MAX_SHIFT_FRACTION,
    )
    if max_shift_fraction < 0.0:
        max_shift_fraction = _DEFAULT_MAX_SHIFT_FRACTION
    points = np.array(
        [[0.0, 0.0], [float(w), 0.0], [0.0, float(h)], [float(w), float(h)], [0.5 * w, 0.5 * h]],
        dtype=np.float64,
    )
    moved = (points @ transform[:, :2].T) + transform[:, 2]
    displacement = moved - points
    if (
        float(np.max(np.abs(displacement[:, 0]))) > float(w) * max_shift_fraction
        or float(np.max(np.abs(displacement[:, 1]))) > float(h) * max_shift_fraction
    ):
        return "rejected_large_shift"
    return None


def _estimate_feature_affine(
    reference_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    reference_f32: np.ndarray,
    moving_f32: np.ndarray,
    mask_u8: np.ndarray,
    config: StabilizationConfig,
) -> tuple[_AffineCandidate | None, str]:
    max_keypoints = int(max(200, min(6000, getattr(config, "feature_max_keypoints", 1800))))
    min_matches = int(max(6, min(100, getattr(config, "feature_min_matches", 10))))
    ratio_test = max(0.50, min(0.95, _config_float(config, "feature_ratio_test", 0.78)))
    min_inlier_ratio = max(
        0.10,
        min(0.95, _config_float(config, "feature_min_inlier_ratio", 0.45)),
    )

    orb = cv2.ORB_create(
        nfeatures=max_keypoints,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        fastThreshold=7,
    )
    reference_features = _feature_detection_image(reference_rgb, config)
    moving_features = _feature_detection_image(moving_rgb, config)
    reference_keypoints, reference_descriptors = orb.detectAndCompute(reference_features, mask_u8)
    moving_keypoints, moving_descriptors = orb.detectAndCompute(moving_features, mask_u8)
    if (
        reference_descriptors is None
        or moving_descriptors is None
        or len(reference_keypoints) < min_matches
        or len(moving_keypoints) < min_matches
    ):
        return None, "rejected_insufficient_features"

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(moving_descriptors, reference_descriptors, k=2)
    ratio_matches = [
        first
        for pair in raw_matches
        if len(pair) >= 2
        for first, second in [pair[:2]]
        if float(first.distance) < ratio_test * float(second.distance)
    ]
    unique_matches = []
    used_query: set[int] = set()
    used_train: set[int] = set()
    for match in sorted(ratio_matches, key=lambda item: float(item.distance)):
        if int(match.queryIdx) in used_query or int(match.trainIdx) in used_train:
            continue
        used_query.add(int(match.queryIdx))
        used_train.add(int(match.trainIdx))
        unique_matches.append(match)
    if len(unique_matches) < min_matches:
        return None, "rejected_insufficient_matches"

    moving_points = np.float32([moving_keypoints[item.queryIdx].pt for item in unique_matches]).reshape(-1, 1, 2)
    reference_points = np.float32([reference_keypoints[item.trainIdx].pt for item in unique_matches]).reshape(-1, 1, 2)
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        moving_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=3000,
        confidence=0.995,
        refineIters=15,
    )
    if matrix is None or inlier_mask is None:
        return None, "rejected_affine_estimation"
    matrix = np.asarray(matrix, dtype=np.float32).reshape(2, 3)
    if not np.all(np.isfinite(matrix)):
        return None, "rejected_non_finite"

    inliers = np.asarray(inlier_mask, dtype=np.uint8).reshape(-1) > 0
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = float(inlier_count) / float(max(1, len(unique_matches)))
    if inlier_count < max(6, min_matches // 2) or inlier_ratio < min_inlier_ratio:
        return None, "rejected_low_inlier_support"

    source = moving_points.reshape(-1, 2).astype(np.float64)
    target = reference_points.reshape(-1, 2).astype(np.float64)
    projected = (source @ matrix[:, :2].astype(np.float64).T) + matrix[:, 2].astype(np.float64)
    errors = np.linalg.norm(projected - target, axis=1)
    median_error = float(np.median(errors[inliers]))
    if not math.isfinite(median_error) or median_error > 3.0:
        return None, "rejected_high_reprojection_error"

    rotation_deg, scale, _anisotropy = _affine_geometry(matrix)
    alignment_score = _alignment_score(reference_f32, moving_f32, mask_u8, matrix)
    minimum_score = max(
        0.0,
        min(1.0, _config_float(config, "min_affine_alignment_score", 0.32)),
    )
    if alignment_score < minimum_score:
        return None, "rejected_low_affine_score"
    if _overlap_fraction(mask_u8, matrix) < 0.72:
        return None, "rejected_low_overlap"

    return (
        _AffineCandidate(
            matrix=matrix,
            score=float(alignment_score),
            rotation_deg=float(rotation_deg),
            scale=float(scale),
            inlier_count=int(inlier_count),
            inlier_ratio=float(inlier_ratio),
            median_error=float(median_error),
        ),
        "ok",
    )


def _stable_frame_gate_config(config: StabilizationConfig) -> StabilizationConfig:
    configured_shift = _config_float(config, "max_shift_fraction", _DEFAULT_MAX_SHIFT_FRACTION)
    if configured_shift < 0.0:
        configured_shift = _DEFAULT_MAX_SHIFT_FRACTION
    fallback_shift = max(
        0.0,
        _config_float(config, "stable_frame_max_shift_fraction", 0.08),
    )
    configured_rotation = max(
        0.0,
        _config_float(config, "max_rotation_deg", _DEFAULT_MAX_ROTATION_DEG),
    )
    fallback_rotation = max(
        0.0,
        _config_float(config, "stable_frame_max_rotation_deg", 2.0),
    )
    return replace(
        config,
        target_mode="dish_frame",
        min_response=max(
            0.0,
            min(1.0, _config_float(config, "stable_frame_min_response", 0.12)),
        ),
        max_shift_fraction=min(configured_shift, fallback_shift),
        max_rotation_deg=min(configured_rotation, fallback_rotation),
        min_affine_alignment_score=max(
            0.0,
            min(1.0, _config_float(config, "stable_frame_min_alignment_score", 0.60)),
        ),
        min_affine_alignment_gain=max(
            0.0,
            _config_float(config, "stable_frame_min_alignment_gain", 0.015),
        ),
    )


def _plant_candidate_conflicts_with_stable_frame(
    reference_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    candidate_matrix: np.ndarray,
    config: StabilizationConfig,
) -> bool:
    """Veto plant-only motion when a reliable dish frame is already stationary."""
    if not bool(getattr(config, "enable_biological_motion_veto", True)):
        return False

    frame_config = _stable_frame_gate_config(config)
    frame_mask = build_registration_mask(reference_rgb.shape[:2], frame_config)
    reference_frame = _preprocess_frame_registration_image(reference_rgb, frame_config)
    moving_frame = _preprocess_frame_registration_image(moving_rgb, frame_config)
    if min(
        _masked_texture_std(reference_frame, frame_mask),
        _masked_texture_std(moving_frame, frame_mask),
    ) < 0.008:
        return False

    transform = np.asarray(candidate_matrix, dtype=np.float32).reshape(2, 3)
    h, w = reference_frame.shape[:2]
    points = np.array(
        [[0.0, 0.0], [float(w), 0.0], [0.0, float(h)], [float(w), float(h)], [0.5 * w, 0.5 * h]],
        dtype=np.float32,
    )
    moved = (points @ transform[:, :2].T) + transform[:, 2]
    maximum_displacement = float(np.max(np.linalg.norm(moved - points, axis=1)))
    minimum_displacement = max(
        0.0,
        _config_float(config, "biological_motion_veto_min_displacement_px", 1.5),
    )
    if maximum_displacement < minimum_displacement:
        return False

    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    identity_score = _alignment_score(
        reference_frame,
        moving_frame,
        frame_mask,
        identity,
    )
    minimum_identity_score = max(
        0.0,
        min(
            1.0,
            _config_float(config, "biological_motion_veto_min_identity_score", 0.72),
        ),
    )
    if identity_score < minimum_identity_score:
        return False

    candidate_score = _alignment_score(
        reference_frame,
        moving_frame,
        frame_mask,
        transform,
    )
    minimum_score_drop = max(
        0.0,
        _config_float(config, "biological_motion_veto_min_score_drop", 0.035),
    )
    return bool(candidate_score + minimum_score_drop < identity_score)


def _stable_frame_candidate_passes(
    *,
    matrix_small: np.ndarray,
    matrix_full: np.ndarray,
    reference_frame_f32: np.ndarray,
    moving_frame_f32: np.ndarray,
    frame_mask_u8: np.ndarray,
    reference_biological_f32: np.ndarray,
    moving_biological_f32: np.ndarray,
    biological_mask_u8: np.ndarray,
    frame_identity_score: float,
    biological_identity_score: float,
    full_shape_hw: tuple[int, int],
    config: StabilizationConfig,
) -> tuple[bool, str, float]:
    rejection = _transform_rejection_status(matrix_full, full_shape_hw, config)
    if rejection is not None:
        return False, rejection, 0.0

    frame_score = _alignment_score(
        reference_frame_f32,
        moving_frame_f32,
        frame_mask_u8,
        matrix_small,
    )
    minimum_score = max(
        0.0,
        min(1.0, _config_float(config, "stable_frame_min_alignment_score", 0.60)),
    )
    minimum_gain = max(
        0.0,
        _config_float(config, "stable_frame_min_alignment_gain", 0.015),
    )
    if frame_score < minimum_score:
        return False, "rejected_stable_frame_low_score", frame_score
    if frame_score < frame_identity_score + minimum_gain:
        return False, "rejected_stable_frame_low_gain", frame_score
    if _overlap_fraction(frame_mask_u8, matrix_small) < 0.84:
        return False, "rejected_stable_frame_low_overlap", frame_score

    biological_score = _alignment_score(
        reference_biological_f32,
        moving_biological_f32,
        biological_mask_u8,
        matrix_small,
    )
    tolerance = max(
        0.0,
        _config_float(config, "stable_frame_biological_tolerance", 0.003),
    )
    if biological_score + tolerance < biological_identity_score:
        return False, "rejected_stable_frame_biological_conflict", frame_score
    return True, "ok", frame_score


def _estimate_stable_frame_fallback(
    reference_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    reference_biological_f32: np.ndarray,
    moving_biological_f32: np.ndarray,
    biological_mask_u8: np.ndarray,
    biological_identity_score: float,
    full_shape_hw: tuple[int, int],
    registration_scale: float,
    config: StabilizationConfig,
) -> tuple[_StableFrameCandidate | None, str]:
    frame_config = _stable_frame_gate_config(config)
    frame_mask = build_registration_mask(reference_rgb.shape[:2], frame_config)
    reference_frame = _preprocess_frame_registration_image(reference_rgb, frame_config)
    moving_frame = _preprocess_frame_registration_image(moving_rgb, frame_config)
    if min(
        _masked_texture_std(reference_frame, frame_mask),
        _masked_texture_std(moving_frame, frame_mask),
    ) < 0.008:
        return None, "rejected_stable_frame_insufficient_texture"

    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    frame_identity_score = _alignment_score(
        reference_frame,
        moving_frame,
        frame_mask,
        identity,
    )
    active = frame_mask > 0
    mask_norm = frame_mask.astype(np.float32) / 255.0
    reference_masked = (reference_frame - float(np.mean(reference_frame[active]))) * mask_norm
    moving_masked = (moving_frame - float(np.mean(moving_frame[active]))) * mask_norm
    shift, response = cv2.phaseCorrelate(reference_masked, moving_masked)
    try:
        raw_shift_x = float(shift[0])
        raw_shift_y = float(shift[1])
        response_score = float(response)
    except (IndexError, TypeError, ValueError, OverflowError):
        raw_shift_x = raw_shift_y = response_score = float("nan")

    phase_rejection = "rejected_stable_frame_non_finite"
    if all(math.isfinite(value) for value in (raw_shift_x, raw_shift_y, response_score)):
        phase_small = np.array(
            [[1.0, 0.0, -raw_shift_x], [0.0, 1.0, -raw_shift_y]],
            dtype=np.float32,
        )
        phase_full = phase_small.copy()
        phase_full[:, 2] /= max(1.0e-6, float(registration_scale))
        minimum_response = max(
            0.0,
            min(1.0, _config_float(config, "stable_frame_min_response", 0.12)),
        )
        if response_score >= minimum_response:
            passes, phase_rejection, _frame_score = _stable_frame_candidate_passes(
                matrix_small=phase_small,
                matrix_full=phase_full,
                reference_frame_f32=reference_frame,
                moving_frame_f32=moving_frame,
                frame_mask_u8=frame_mask,
                reference_biological_f32=reference_biological_f32,
                moving_biological_f32=moving_biological_f32,
                biological_mask_u8=biological_mask_u8,
                frame_identity_score=frame_identity_score,
                biological_identity_score=biological_identity_score,
                full_shape_hw=full_shape_hw,
                config=frame_config,
            )
            if passes:
                return (
                    _StableFrameCandidate(
                        matrix=phase_full,
                        score=float(response_score),
                        rotation_deg=0.0,
                        method="dish_frame_phase_fallback",
                    ),
                    "ok",
                )
        else:
            phase_rejection = "rejected_stable_frame_low_response"

    if not bool(getattr(config, "enable_stable_frame_affine_fallback", True)):
        return None, phase_rejection

    affine, affine_status = _estimate_feature_affine(
        reference_rgb,
        moving_rgb,
        reference_frame,
        moving_frame,
        frame_mask,
        frame_config,
    )
    if affine is None:
        return None, affine_status if affine_status != "ok" else phase_rejection
    affine_full = np.asarray(affine.matrix, dtype=np.float32).copy()
    affine_full[:, 2] /= max(1.0e-6, float(registration_scale))
    passes, affine_status, frame_score = _stable_frame_candidate_passes(
        matrix_small=affine.matrix,
        matrix_full=affine_full,
        reference_frame_f32=reference_frame,
        moving_frame_f32=moving_frame,
        frame_mask_u8=frame_mask,
        reference_biological_f32=reference_biological_f32,
        moving_biological_f32=moving_biological_f32,
        biological_mask_u8=biological_mask_u8,
        frame_identity_score=frame_identity_score,
        biological_identity_score=biological_identity_score,
        full_shape_hw=full_shape_hw,
        config=frame_config,
    )
    if not passes:
        return None, affine_status
    return (
        _StableFrameCandidate(
            matrix=affine_full,
            score=float(frame_score),
            rotation_deg=float(affine.rotation_deg),
            method="dish_frame_orb_ransac_affine_fallback",
        ),
        "ok",
    )


def stabilize_against_reference(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    config: StabilizationConfig,
) -> StabilizationResult:
    moving_rgb = np.asarray(moving_image, dtype=np.uint8)
    reference_rgb = np.asarray(reference_image, dtype=np.uint8)
    h, w = moving_rgb.shape[:2]
    if h <= 0 or w <= 0:
        return _identity_result(moving_rgb, status="empty")

    if reference_rgb.shape[:2] != moving_rgb.shape[:2]:
        reference_rgb = cv2.resize(reference_rgb, (w, h), interpolation=cv2.INTER_AREA)

    scale = 1.0
    ref_for_registration = reference_rgb
    mov_for_registration = moving_rgb
    longest = max(h, w)
    max_dim = int(max(64, config.downscale_max_dim))
    if longest > max_dim:
        scale = float(max_dim) / float(longest)
        new_size = (max(64, int(round(w * scale))), max(64, int(round(h * scale))))
        ref_for_registration = cv2.resize(reference_rgb, new_size, interpolation=cv2.INTER_AREA)
        mov_for_registration = cv2.resize(moving_rgb, new_size, interpolation=cv2.INTER_AREA)

    mask_small = build_registration_mask(ref_for_registration.shape[:2], config)
    ref_small = _preprocess_registration_image(ref_for_registration, config)
    mov_small = _preprocess_registration_image(mov_for_registration, config)

    mask_norm = mask_small.astype(np.float32) / 255.0
    active_mask = mask_small > 0
    if np.any(active_mask):
        ref_masked = (ref_small - float(np.mean(ref_small[active_mask]))) * mask_norm
        mov_masked = (mov_small - float(np.mean(mov_small[active_mask]))) * mask_norm
    else:
        ref_masked = np.zeros_like(ref_small, dtype=np.float32)
        mov_masked = np.zeros_like(mov_small, dtype=np.float32)

    shift, response = cv2.phaseCorrelate(ref_masked, mov_masked)
    try:
        raw_shift_x = float(shift[0])
        raw_shift_y = float(shift[1])
        response_score = float(response)
    except (IndexError, TypeError, ValueError, OverflowError):
        return _identity_result(moving_rgb, status="rejected_non_finite")

    if not all(math.isfinite(value) for value in (raw_shift_x, raw_shift_y, response_score)):
        return _identity_result(moving_rgb, status="rejected_non_finite")

    min_response = _config_float(config, "min_response", _DEFAULT_MIN_RESPONSE)
    min_response = max(0.0, min(1.0, min_response))
    dx = -raw_shift_x / max(1.0e-6, float(scale))
    dy = -raw_shift_y / max(1.0e-6, float(scale))
    if not math.isfinite(dx) or not math.isfinite(dy):
        return _identity_result(moving_rgb, status="rejected_non_finite", score=response_score)

    phase_small = np.array(
        [[1.0, 0.0, -raw_shift_x], [0.0, 1.0, -raw_shift_y]],
        dtype=np.float32,
    )
    phase_full = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    phase_status = "ok"
    if response_score < min_response:
        phase_status = "rejected_low_response"
    else:
        phase_status = _transform_rejection_status(phase_full, (h, w), config) or "ok"
    phase_alignment_score = _alignment_score(ref_small, mov_small, mask_small, phase_small)
    identity_alignment_score = _alignment_score(
        ref_small,
        mov_small,
        mask_small,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    )

    affine_candidate: _AffineCandidate | None = None
    affine_full: np.ndarray | None = None
    affine_status = "disabled"
    feature_enabled = bool(getattr(config, "enable_feature_fallback", True))
    has_texture = min(
        _masked_texture_std(ref_small, mask_small),
        _masked_texture_std(mov_small, mask_small),
    ) >= 0.008
    if feature_enabled and has_texture:
        affine_candidate, affine_status = _estimate_feature_affine(
            ref_for_registration,
            mov_for_registration,
            ref_small,
            mov_small,
            mask_small,
            config,
        )
        if affine_candidate is not None:
            affine_full = np.asarray(affine_candidate.matrix, dtype=np.float32).copy()
            affine_full[:, 2] /= max(1.0e-6, float(scale))
            affine_status = _transform_rejection_status(affine_full, (h, w), config) or "ok"
            minimum_gain = max(
                0.0,
                _config_float(config, "min_affine_alignment_gain", 0.018),
            )
            if (
                affine_status == "ok"
                and affine_candidate.score < identity_alignment_score + minimum_gain
            ):
                affine_status = "rejected_no_alignment_gain"
    elif feature_enabled:
        affine_status = "rejected_insufficient_texture"

    use_affine = False
    if affine_candidate is not None and affine_full is not None and affine_status == "ok":
        if phase_status != "ok":
            use_affine = True
        else:
            minimum_gain = max(
                0.0,
                _config_float(config, "min_affine_alignment_gain", 0.018),
            )
            rotation_threshold = max(
                0.0,
                _config_float(config, "affine_rotation_selection_deg", 0.30),
            )
            rotation_improves = (
                abs(float(affine_candidate.rotation_deg)) >= rotation_threshold
                and affine_candidate.score >= phase_alignment_score + (0.5 * minimum_gain)
            )
            substantial_improvement = (
                affine_candidate.score >= phase_alignment_score + max(0.06, 3.0 * minimum_gain)
            )
            use_affine = bool(rotation_improves or substantial_improvement)

    stable_frame_candidate: _StableFrameCandidate | None = None
    if (
        _is_plant_target(config)
        and phase_status != "ok"
        and not use_affine
        and bool(getattr(config, "enable_stable_frame_fallback", True))
    ):
        stable_frame_candidate, _stable_frame_status = _estimate_stable_frame_fallback(
            ref_for_registration,
            mov_for_registration,
            ref_small,
            mov_small,
            mask_small,
            identity_alignment_score,
            (h, w),
            scale,
            config,
        )

    direct_plant_matrix_small: np.ndarray | None = None
    if use_affine and affine_candidate is not None and affine_full is not None:
        warp = affine_full
        method = "plant_top_orb_ransac_affine" if _is_plant_target(config) else "orb_ransac_affine"
        result_score = float(affine_candidate.score)
        rotation_deg = float(affine_candidate.rotation_deg)
        if _is_plant_target(config):
            direct_plant_matrix_small = affine_candidate.matrix
    elif phase_status == "ok":
        warp = phase_full
        method = "plant_top_phase_correlation" if _is_plant_target(config) else "phase_correlation"
        result_score = float(response_score)
        rotation_deg = 0.0
        if _is_plant_target(config):
            direct_plant_matrix_small = phase_small
    elif stable_frame_candidate is not None:
        warp = stable_frame_candidate.matrix
        method = stable_frame_candidate.method
        result_score = float(stable_frame_candidate.score)
        rotation_deg = float(stable_frame_candidate.rotation_deg)
    else:
        diagnostic_status = phase_status
        if affine_status in {
            "rejected_large_rotation",
            "rejected_scale_change",
            "rejected_large_shift",
            "rejected_non_finite",
        }:
            diagnostic_status = affine_status
        return _identity_result(moving_rgb, status=diagnostic_status, score=response_score)

    if direct_plant_matrix_small is not None and _plant_candidate_conflicts_with_stable_frame(
        ref_for_registration,
        mov_for_registration,
        direct_plant_matrix_small,
        config,
    ):
        return _identity_result(
            moving_rgb,
            status="rejected_biological_motion_conflict",
            score=result_score,
        )

    warped = cv2.warpAffine(
        moving_rgb,
        warp,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return StabilizationResult(
        warped_image=warped,
        warp_matrix=warp,
        shift_x=float(warp[0, 2]),
        shift_y=float(warp[1, 2]),
        rotation_deg=float(rotation_deg),
        score=float(result_score),
        method=method,
        status="ok",
    )


def warp_summary(transform: np.ndarray) -> tuple[float, float, float]:
    matrix = np.asarray(transform, dtype=np.float32).reshape(2, 3)
    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    angle = float(math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))))
    return dx, dy, angle
