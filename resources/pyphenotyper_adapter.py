from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from functools import lru_cache
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

from .bw_arabidopsis_ownership import build_bw_arabidopsis_measurements
from .general_root_starter import (
    build_calibration_profile,
    describe_image_features,
    expert_prior_score,
    features_as_dict,
    iou_score,
    mask_confidence,
    normalize_annotation_targets,
)
from .lucifer_gan_gap_repair import apply_lucifer_gan_gap_repair
from .models import DatasetImageItem


BBox = tuple[int, int, int, int]


@lru_cache(maxsize=64)
def _artifact_sha256(path_text: str) -> str | None:
    path = Path(path_text).expanduser()
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(slots=True)
class PyPhenotyperVariantConfig:
    pipeline_dir: Path
    root_model_path: Path
    shoot_model_path: Path
    profile_overrides: dict[str, object] | None = None
    root_profile_overrides: dict[str, object] | None = None
    shoot_profile_overrides: dict[str, object] | None = None
    pipeline_overrides: dict[str, object] | None = None
    label: str = "variant"
    pixel_size_mm: float | None = None


@dataclass(slots=True)
class PyPhenotyperExpertConfig:
    key: str
    pipeline_dir: Path
    root_model_path: Path
    shoot_model_path: Path
    profile_overrides: dict[str, object] | None = None
    root_profile_overrides: dict[str, object] | None = None
    shoot_profile_overrides: dict[str, object] | None = None
    pipeline_overrides: dict[str, object] | None = None
    label: str = "expert"
    image_mode: str = "any"  # any | rgb_only | grayscale_only
    family: str = "generic"
    priority: float = 0.0
    router_hints: dict[str, object] | None = None
    pixel_size_mm: float | None = None


@dataclass(slots=True)
class PyPhenotyperConfig:
    pipeline_dir: Path
    root_model_path: Path
    shoot_model_path: Path
    seed_model_path: Path | None = None
    patch_size: int = 256
    refinement_steps: int = 1
    root_class_id: int = 1
    shoot_class_id: int = 2
    lateral_class_id: int | None = None
    seed_class_id: int | None = None
    include_occlusion: bool = True
    enable_bbox_tracking: bool = True
    min_component_area: int = 40
    bbox_padding: int = 12
    tracking_search_margin: int = 26
    pixel_size_mm: float = 0.05
    profile_overrides: dict[str, object] | None = None
    root_profile_overrides: dict[str, object] | None = None
    shoot_profile_overrides: dict[str, object] | None = None
    seed_profile_overrides: dict[str, object] | None = None
    pipeline_overrides: dict[str, object] | None = None
    grayscale_variant: PyPhenotyperVariantConfig | None = None
    rgb_variant: PyPhenotyperVariantConfig | None = None
    expert_variants: tuple[PyPhenotyperExpertConfig, ...] = ()
    router_mode: str = "legacy"
    router_uncertainty_margin: float = 0.08
    calibration_profile: dict[str, object] | None = None
    lucifer_gan_gap_repair_enabled: bool = False
    lucifer_gan_gap_repair_dir: Path | None = None


_SCALE_METADATA_FILENAMES = (
    "pyphenotyper_scale.json",
    "scale_metadata.json",
    "npec_scale.json",
    "pipeline_metadata.json",
)

_ADAPTER_ONLY_PIPELINE_OVERRIDE_KEYS = {
    "adapter_shoot_guard_mode",
    "expected_plant_count",
    "lucifer_green_shoot_rescue",
    "lateral_class_id",
    "ownership_backend",
    "shoot_color_rescue_eligibility",
    "shoot_color_rescue_eligible",
    "shoot_color_rescue_failure_policy",
    "shoot_color_rescue_model_locked",
    "shoot_color_rescue_mode",
    "shoot_priority_over_root",
}


@contextmanager
def _prepend_sys_path(path: Path):
    value = str(path)
    inserted = False
    if value not in sys.path:
        sys.path.insert(0, value)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(value)
            except ValueError:
                pass


def _normalize_pipeline_dir(pipeline_dir: Path) -> tuple[Path, Path]:
    pipeline_dir = Path(pipeline_dir).expanduser().resolve()
    if not pipeline_dir.exists():
        raise FileNotFoundError(f"PyPhenotyper path does not exist: {pipeline_dir}")
    if pipeline_dir.name == "pyphenotyper":
        package_root = pipeline_dir.parent
        package_dir = pipeline_dir
    else:
        package_dir = pipeline_dir / "pyphenotyper"
        if not package_dir.exists():
            raise FileNotFoundError(
                "PyPhenotyper package folder not found. Expected either '<path>/pyphenotyper' "
                f"or path ending with 'pyphenotyper'. Got: {pipeline_dir}"
            )
        package_root = pipeline_dir
    return package_root, package_dir


def _safe_load_image_for_pipeline(image_path: str, verbose: bool = True) -> np.ndarray:
    _ = verbose
    # Keep horizontal flip to match pyphenotyper model conventions.
    image = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)
    return np.fliplr(image)


def _coerce_positive_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed <= 0.0:
        return None
    return parsed


def _extract_pixel_size_mm_from_mapping(payload: object) -> float | None:
    if not isinstance(payload, dict):
        return None

    direct_keys = (
        "pixel_size_mm",
        "mm_per_px",
        "mm_per_pixel",
    )
    inverse_keys = (
        "px_per_mm",
        "pixels_per_mm",
        "pixel_per_mm",
    )
    for key in direct_keys:
        value = _coerce_positive_float(payload.get(key))
        if value is not None:
            return value
    for key in inverse_keys:
        value = _coerce_positive_float(payload.get(key))
        if value is not None:
            return 1.0 / value
    for nested_key in ("scale", "pixel_scale", "measurement", "metadata"):
        nested_value = _extract_pixel_size_mm_from_mapping(payload.get(nested_key))
        if nested_value is not None:
            return nested_value
    return None


def _extract_pixel_size_from_text(path: Path) -> float | None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    plate_mm_match = re.search(r"plate_size_mm\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
    plate_px_match = re.search(r"plate_size_pixels\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
    if plate_mm_match and plate_px_match:
        plate_mm = _coerce_positive_float(plate_mm_match.group(1))
        plate_px = _coerce_positive_float(plate_px_match.group(1))
        if plate_mm is not None and plate_px is not None:
            return plate_mm / plate_px

    px_per_mm_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*px/mm", text, flags=re.IGNORECASE)
    if px_per_mm_match:
        px_per_mm = _coerce_positive_float(px_per_mm_match.group(1))
        if px_per_mm is not None:
            return 1.0 / px_per_mm

    mm_per_px_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mm/px", text, flags=re.IGNORECASE)
    if mm_per_px_match:
        mm_per_px = _coerce_positive_float(mm_per_px_match.group(1))
        if mm_per_px is not None:
            return mm_per_px
    return None


def detect_pyphenotyper_pixel_size_mm(pipeline_dir: Path | str | None) -> tuple[float | None, str | None]:
    if pipeline_dir is None:
        return None, None
    try:
        package_root, package_dir = _normalize_pipeline_dir(Path(pipeline_dir).expanduser())
    except Exception:
        return None, None

    candidate_dirs = (package_root, package_dir)
    for base_dir in candidate_dirs:
        for filename in _SCALE_METADATA_FILENAMES:
            metadata_path = base_dir / filename
            if not metadata_path.exists():
                continue
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            value = _extract_pixel_size_mm_from_mapping(payload)
            if value is not None:
                return value, str(metadata_path)

    legacy_files = (
        package_dir / "features" / "roots_segmentation.py",
        package_dir / "features" / "roots_segmentation_old.py",
    )
    for legacy_path in legacy_files:
        if not legacy_path.exists():
            continue
        value = _extract_pixel_size_from_text(legacy_path)
        if value is not None:
            return value, str(legacy_path)

    return None, None


def _is_rgb_like_image(image: np.ndarray) -> bool:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return False
    rgb = arr[..., :3].astype(np.int16, copy=False)
    channel_delta = max(
        float(np.mean(np.abs(rgb[..., 0] - rgb[..., 1]))),
        float(np.mean(np.abs(rgb[..., 1] - rgb[..., 2]))),
        float(np.mean(np.abs(rgb[..., 0] - rgb[..., 2]))),
    )
    pixel_chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    strong_chroma_fraction = float(np.mean(pixel_chroma >= 8))
    moderate_chroma_fraction = float(np.mean(pixel_chroma >= 3))
    # RGB-triplet grayscale files often acquire sub-pixel JPEG chroma noise.
    # Treat that as grayscale so a Hades series cannot hop routes frame to
    # frame. Sparse but strongly colored roots/leaves still count as RGB.
    return bool(
        channel_delta > 1.5
        or strong_chroma_fraction >= 0.0005
        or (channel_delta > 0.5 and moderate_chroma_fraction >= 0.02)
    )


def _merge_overrides(
    base: dict[str, object] | None,
    variant: dict[str, object] | None,
) -> dict[str, object] | None:
    merged: dict[str, object] = {}
    if isinstance(base, dict):
        merged.update(base)
    if isinstance(variant, dict):
        merged.update(variant)
    return merged or None


def _pipeline_tuning_overrides(overrides: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(overrides, dict):
        return None
    filtered = {str(key): value for key, value in overrides.items() if str(key) not in _ADAPTER_ONLY_PIPELINE_OVERRIDE_KEYS}
    return filtered or None


def _config_from_variant(base_config: PyPhenotyperConfig, variant: PyPhenotyperVariantConfig) -> PyPhenotyperConfig:
    pixel_size_mm = (
        float(variant.pixel_size_mm)
        if variant.pixel_size_mm is not None and float(variant.pixel_size_mm) > 0.0
        else float(base_config.pixel_size_mm)
    )
    return PyPhenotyperConfig(
        pipeline_dir=Path(variant.pipeline_dir).expanduser(),
        root_model_path=Path(variant.root_model_path).expanduser(),
        shoot_model_path=Path(variant.shoot_model_path).expanduser(),
        seed_model_path=Path(base_config.seed_model_path).expanduser() if base_config.seed_model_path is not None else None,
        patch_size=int(base_config.patch_size),
        refinement_steps=int(base_config.refinement_steps),
        root_class_id=int(base_config.root_class_id),
        shoot_class_id=int(base_config.shoot_class_id),
        lateral_class_id=int(base_config.lateral_class_id) if base_config.lateral_class_id is not None else None,
        seed_class_id=int(base_config.seed_class_id) if base_config.seed_class_id is not None else None,
        include_occlusion=bool(base_config.include_occlusion),
        enable_bbox_tracking=bool(base_config.enable_bbox_tracking),
        min_component_area=int(base_config.min_component_area),
        bbox_padding=int(base_config.bbox_padding),
        tracking_search_margin=int(base_config.tracking_search_margin),
        pixel_size_mm=float(pixel_size_mm),
        profile_overrides=_merge_overrides(base_config.profile_overrides, variant.profile_overrides),
        root_profile_overrides=_merge_overrides(base_config.root_profile_overrides, variant.root_profile_overrides),
        shoot_profile_overrides=_merge_overrides(base_config.shoot_profile_overrides, variant.shoot_profile_overrides),
        seed_profile_overrides=base_config.seed_profile_overrides,
        pipeline_overrides=_merge_overrides(base_config.pipeline_overrides, variant.pipeline_overrides),
        router_mode="legacy",
        calibration_profile=base_config.calibration_profile,
        lucifer_gan_gap_repair_enabled=bool(base_config.lucifer_gan_gap_repair_enabled),
        lucifer_gan_gap_repair_dir=(
            Path(base_config.lucifer_gan_gap_repair_dir).expanduser()
            if base_config.lucifer_gan_gap_repair_dir is not None
            else None
        ),
    )


def _resolve_item_config(item: DatasetImageItem, config: PyPhenotyperConfig) -> tuple[PyPhenotyperConfig, str, str]:
    family = "rgb" if _is_rgb_like_image(item.image) else "grayscale"
    variant = config.rgb_variant if family == "rgb" else config.grayscale_variant
    if variant is None:
        return config, "default", family
    return _config_from_variant(config, variant), str(variant.label or family), family


def _config_from_expert(base_config: PyPhenotyperConfig, expert: PyPhenotyperExpertConfig) -> PyPhenotyperConfig:
    pixel_size_mm = (
        float(expert.pixel_size_mm)
        if expert.pixel_size_mm is not None and float(expert.pixel_size_mm) > 0.0
        else float(base_config.pixel_size_mm)
    )
    return PyPhenotyperConfig(
        pipeline_dir=Path(expert.pipeline_dir).expanduser(),
        root_model_path=Path(expert.root_model_path).expanduser(),
        shoot_model_path=Path(expert.shoot_model_path).expanduser(),
        seed_model_path=Path(base_config.seed_model_path).expanduser() if base_config.seed_model_path is not None else None,
        patch_size=int(base_config.patch_size),
        refinement_steps=int(base_config.refinement_steps),
        root_class_id=int(base_config.root_class_id),
        shoot_class_id=int(base_config.shoot_class_id),
        lateral_class_id=int(base_config.lateral_class_id) if base_config.lateral_class_id is not None else None,
        seed_class_id=int(base_config.seed_class_id) if base_config.seed_class_id is not None else None,
        include_occlusion=bool(base_config.include_occlusion),
        enable_bbox_tracking=bool(base_config.enable_bbox_tracking),
        min_component_area=int(base_config.min_component_area),
        bbox_padding=int(base_config.bbox_padding),
        tracking_search_margin=int(base_config.tracking_search_margin),
        pixel_size_mm=float(pixel_size_mm),
        profile_overrides=_merge_overrides(base_config.profile_overrides, expert.profile_overrides),
        root_profile_overrides=_merge_overrides(base_config.root_profile_overrides, expert.root_profile_overrides),
        shoot_profile_overrides=_merge_overrides(base_config.shoot_profile_overrides, expert.shoot_profile_overrides),
        seed_profile_overrides=base_config.seed_profile_overrides,
        pipeline_overrides=_merge_overrides(base_config.pipeline_overrides, expert.pipeline_overrides),
        router_mode="legacy",
        calibration_profile=base_config.calibration_profile,
        lucifer_gan_gap_repair_enabled=bool(base_config.lucifer_gan_gap_repair_enabled),
        lucifer_gan_gap_repair_dir=(
            Path(base_config.lucifer_gan_gap_repair_dir).expanduser()
            if base_config.lucifer_gan_gap_repair_dir is not None
            else None
        ),
    )


def _profile_payload(profile: object) -> dict[str, object] | None:
    if profile is None:
        return None
    if is_dataclass(profile):
        try:
            return dict(asdict(profile))
        except Exception:
            return None
    if isinstance(profile, dict):
        return dict(profile)
    return None


def _legacy_patch_maker(height: int, width: int, patch_size: int) -> list[tuple[int, int, int, int]]:
    h_adjusted = int(height - (height % patch_size))
    w_adjusted = int(width - (width % patch_size))
    h_vals = range(0, h_adjusted, patch_size)
    w_vals = range(0, w_adjusted, patch_size)
    patches = [(int(y), int(x), int(patch_size), int(patch_size)) for y in h_vals for x in w_vals]
    remainder_h = int(height - patch_size)
    remainder_w = int(width - patch_size)
    patches.extend((remainder_h, int(x), int(patch_size), int(patch_size)) for x in w_vals)
    patches.extend((int(y), remainder_w, int(patch_size), int(patch_size)) for y in h_vals)
    patches.append((remainder_h, remainder_w, int(patch_size), int(patch_size)))
    deduped: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for patch in patches:
        if patch not in seen:
            deduped.append(patch)
            seen.add(patch)
    return deduped


def _legacy_patch_predict_mask(
    *,
    image_path: Path,
    model: object,
    patch_size: int,
    threshold: float,
) -> np.ndarray:
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    model_input_size = int(input_shape[1]) if input_shape is not None and len(input_shape) >= 3 else int(patch_size)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image for legacy MultipleXLab predictor: {image_path}")
    patches = _legacy_patch_maker(int(image.shape[0]), int(image.shape[1]), int(patch_size))
    stitched = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    patch_images: list[np.ndarray] = []
    for patch_y, patch_x, patch_h, patch_w in patches:
        patch_image = image[patch_y : patch_y + patch_h, patch_x : patch_x + patch_w]
        patch_image = cv2.resize(patch_image, (model_input_size, model_input_size))
        patch_images.append(np.asarray(patch_image, dtype=np.float32) / 255.0)
    batch = np.asarray(patch_images, dtype=np.float32)
    preds = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
    if preds.ndim == 4 and preds.shape[-1] == 1:
        preds = preds[..., 0]
    for index, (patch_y, patch_x, patch_h, patch_w) in enumerate(patches):
        pred_patch = cv2.resize(preds[index].reshape(model_input_size, model_input_size), (patch_size, patch_size))
        stitched[patch_y : patch_y + patch_h, patch_x : patch_x + patch_w] = pred_patch[:patch_h, :patch_w]
    return (stitched >= float(threshold)).astype(np.uint8)


def _mixed_profile_enabled(config: PyPhenotyperConfig) -> bool:
    return bool(config.root_profile_overrides or config.shoot_profile_overrides or config.seed_profile_overrides or config.seed_model_path)


def _seed_mask_from_multiclass(labels: np.ndarray, profile: object) -> np.ndarray:
    seed_label_ids = tuple(int(value) for value in (getattr(profile, "seed_label_ids", ()) or ()) if int(value) >= 0)
    if not seed_label_ids:
        return np.zeros(labels.shape[:2], dtype=np.uint8)
    return np.isin(labels, seed_label_ids).astype(np.uint8)


def _predict_single_target_mask(
    *,
    features_module,
    model: object,
    image_path: Path,
    patch_size: int,
    refinement_steps: int,
    tuning,
    profile_overrides: dict[str, object] | None,
    target: str,
) -> tuple[np.ndarray, object, dict[str, object] | None]:
    infer_profile = getattr(features_module, "infer_model_profile", None)
    if not callable(infer_profile):
        raise RuntimeError("Selected PyPhenotyper pipeline does not expose infer_model_profile for mixed expert routing.")
    profile_input_overrides = dict(profile_overrides or {})
    predictor_kind = str(profile_input_overrides.pop("predictor_kind", "") or "").strip().lower()
    predictor_patch_size = int(profile_input_overrides.pop("predictor_patch_size", patch_size) or patch_size)
    profile = infer_profile(model, profile_overrides=profile_input_overrides or None)
    if predictor_kind in {"mxlab_legacy_patch", "mxlab_legacy_patch256"}:
        threshold_name = "root_threshold" if target == "root" else "shoot_threshold"
        threshold = float(getattr(profile, threshold_name, 0.5) or 0.5)
        mask = _legacy_patch_predict_mask(
            image_path=image_path,
            model=model,
            patch_size=int(predictor_patch_size),
            threshold=threshold,
        )
        return np.asarray(mask, dtype=np.uint8), profile, _profile_payload(profile)
    mode = str(getattr(profile, "mode", "") or "").strip().lower()
    if mode == "multiclass":
        predict_multiclass = getattr(features_module, "_predict_multiclass_labels", None)
        masks_from_multiclass = getattr(features_module, "_masks_from_multiclass", None)
        if not callable(predict_multiclass) or not callable(masks_from_multiclass):
            raise RuntimeError("Selected PyPhenotyper pipeline is missing multiclass helpers required for mixed expert routing.")
        labels = predict_multiclass(model, image_path, int(patch_size), profile)
        root_mask, shoot_mask = masks_from_multiclass(labels, profile)
        seed_mask = _seed_mask_from_multiclass(labels, profile)
        if target == "root":
            mask = root_mask
        elif target == "shoot":
            mask = shoot_mask
        elif target == "seed":
            mask = seed_mask
        else:
            raise ValueError(f"Unsupported mixed-profile target '{target}'.")
    else:
        predict_binary = getattr(features_module, "_predict_binary_probability_map", None)
        if not callable(predict_binary):
            raise RuntimeError("Selected PyPhenotyper pipeline is missing binary helpers required for mixed expert routing.")
        prob = predict_binary(
            model,
            image_path=image_path,
            patch_size=int(patch_size),
            profile=profile,
            refinement_steps=int(refinement_steps),
            tuning=tuning,
        )
        threshold_name = "root_threshold" if target == "root" else "shoot_threshold"
        threshold = float(getattr(profile, threshold_name, 0.5) or 0.5)
        mask = (np.asarray(prob, dtype=np.float32) >= threshold).astype(np.uint8)
    return np.asarray(mask, dtype=np.uint8), profile, _profile_payload(profile)


def _predict_masks_for_config_mixed(item: DatasetImageItem, effective_config: PyPhenotyperConfig) -> dict[str, object]:
    features_module = _load_pyphenotyper_features_module(str(Path(effective_config.pipeline_dir).expanduser().resolve()))
    resolve_tuning = getattr(features_module, "resolve_pipeline_tuning", None)
    if not callable(resolve_tuning):
        raise RuntimeError("Selected PyPhenotyper pipeline does not expose resolve_pipeline_tuning for mixed expert routing.")
    tuning_overrides = _pipeline_tuning_overrides(effective_config.pipeline_overrides)
    tuning = resolve_tuning(tuning_overrides or None)
    root_model = _load_keras_model_cached(str(effective_config.root_model_path.expanduser().resolve()))
    shoot_model = _load_keras_model_cached(str(effective_config.shoot_model_path.expanduser().resolve()))
    seed_model = (
        _load_keras_model_cached(str(effective_config.seed_model_path.expanduser().resolve()))
        if effective_config.seed_model_path is not None
        and effective_config.seed_class_id is not None
        and int(effective_config.seed_class_id) > 0
        else None
    )

    image_path, temp_ctx = _ensure_item_path(item)
    try:
        root_mask_raw, root_profile_obj, root_profile = _predict_single_target_mask(
            features_module=features_module,
            model=root_model,
            image_path=image_path,
            patch_size=int(effective_config.patch_size),
            refinement_steps=int(effective_config.refinement_steps),
            tuning=tuning,
            profile_overrides=effective_config.root_profile_overrides or effective_config.profile_overrides,
            target="root",
        )
        shoot_mask_raw, shoot_profile_obj, shoot_profile = _predict_single_target_mask(
            features_module=features_module,
            model=shoot_model,
            image_path=image_path,
            patch_size=int(effective_config.patch_size),
            refinement_steps=int(effective_config.refinement_steps),
            tuning=tuning,
            profile_overrides=effective_config.shoot_profile_overrides or effective_config.profile_overrides,
            target="shoot",
        )
        seed_mask_raw = np.zeros(item.image.shape[:2], dtype=np.uint8)
        seed_profile_obj = None
        seed_profile = None
        if effective_config.seed_class_id is not None and int(effective_config.seed_class_id) > 0:
            seed_source_model = seed_model if seed_model is not None else shoot_model
            seed_profile_overrides = (
                effective_config.seed_profile_overrides
                or effective_config.shoot_profile_overrides
                or effective_config.profile_overrides
            )
            seed_mask_raw, seed_profile_obj, seed_profile = _predict_single_target_mask(
                features_module=features_module,
                model=seed_source_model,
                image_path=image_path,
                patch_size=int(effective_config.patch_size),
                refinement_steps=int(effective_config.refinement_steps),
                tuning=tuning,
                profile_overrides=seed_profile_overrides,
                target="seed",
            )
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

    processing_image = np.asarray(item.image)
    load_processing_rgb = getattr(features_module, "_load_processing_rgb", None)
    if callable(load_processing_rgb):
        processing_profile = shoot_profile_obj if shoot_profile is not None else root_profile_obj
        if processing_profile is not None:
            image_path, temp_ctx = _ensure_item_path(item)
            try:
                processing_image = np.asarray(load_processing_rgb(image_path, processing_profile), dtype=np.uint8)
            finally:
                if temp_ctx is not None:
                    temp_ctx.cleanup()

    postprocess_root = getattr(features_module, "_postprocess_root_mask", None)
    postprocess_shoot = getattr(features_module, "_postprocess_shoot_mask", None)
    build_occlusion = getattr(features_module, "_build_occlusion_mask", None)
    if not callable(postprocess_root) or not callable(postprocess_shoot) or not callable(build_occlusion):
        raise RuntimeError("Selected PyPhenotyper pipeline is missing postprocess helpers required for mixed expert routing.")

    shape_hw = item.image.shape[:2]
    root_mask = _ensure_mask_shape(np.asarray(root_mask_raw, dtype=np.uint8), shape_hw)
    shoot_mask = _ensure_mask_shape(np.asarray(shoot_mask_raw, dtype=np.uint8), shape_hw)
    seed_mask = _ensure_mask_shape(np.asarray(seed_mask_raw, dtype=np.uint8), shape_hw)
    root_predictor_kind = str((effective_config.root_profile_overrides or {}).get("predictor_kind") or "").strip().lower()
    if root_predictor_kind not in {"mxlab_legacy_patch", "mxlab_legacy_patch256"}:
        root_mask = _ensure_mask_shape(postprocess_root(root_mask, image=processing_image, tuning=tuning), shape_hw)
    shoot_mask = _ensure_mask_shape(postprocess_shoot(shoot_mask, root_mask=root_mask, image=processing_image, tuning=tuning), shape_hw)
    occlusion_mask = _ensure_mask_shape(build_occlusion(root_mask, refinement_steps=int(effective_config.refinement_steps)), shape_hw)

    root_top_pixels = int(np.count_nonzero(root_mask[: max(1, root_mask.shape[0] // 2), :]))
    shoot_top_pixels = int(np.count_nonzero(shoot_mask[: max(1, shoot_mask.shape[0] // 2), :]))
    flag = bool(root_top_pixels <= 25 or shoot_top_pixels <= 75)
    return {
        "root_mask": root_mask.astype(np.uint8),
        "shoot_mask": shoot_mask.astype(np.uint8),
        "seed_mask": seed_mask.astype(np.uint8),
        "occlusion_mask": occlusion_mask.astype(np.uint8),
        "flagged": flag,
        "profile": {
            "mode": "mixed",
            "root_profile": root_profile,
            "shoot_profile": shoot_profile,
            "seed_profile": seed_profile,
        },
        "tuning": dict(asdict(tuning)) if is_dataclass(tuning) else None,
    }


def _routing_enabled(config: PyPhenotyperConfig) -> bool:
    return bool(config.expert_variants or config.grayscale_variant is not None or config.rgb_variant is not None)


def _predict_masks_for_config(item: DatasetImageItem, effective_config: PyPhenotyperConfig) -> dict[str, object]:
    if _mixed_profile_enabled(effective_config):
        return _predict_masks_for_config_mixed(item, effective_config)
    features_module = _load_pyphenotyper_features_module(str(Path(effective_config.pipeline_dir).expanduser().resolve()))
    root_model = _load_keras_model_cached(str(effective_config.root_model_path.expanduser().resolve()))
    shoot_model = _load_keras_model_cached(str(effective_config.shoot_model_path.expanduser().resolve()))
    profile_payload = None
    describe_profile = getattr(features_module, "describe_model_profile", None)
    if callable(describe_profile):
        try:
            profile_payload = describe_profile(root_model, shoot_model, profile_overrides=effective_config.profile_overrides)
        except Exception:
            profile_payload = None

    image_path, temp_ctx = _ensure_item_path(item)
    try:
        raw_output = _call_model_create_masks_compat(
            features_module=features_module,
            image_path=image_path,
            patch_size=int(effective_config.patch_size),
            root_model=root_model,
            shoot_model=shoot_model,
            refinement_steps=int(effective_config.refinement_steps),
            profile_overrides=effective_config.profile_overrides,
            pipeline_overrides=_pipeline_tuning_overrides(effective_config.pipeline_overrides),
        )
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

    if not isinstance(raw_output, (tuple, list)) or len(raw_output) < 2:
        raise RuntimeError(
            "PyPhenotyper model_create_masks returned an unexpected output shape. "
            f"Expected at least root_mask and shoot_mask, got: {type(raw_output).__name__}"
        )

    root_mask = np.asarray(raw_output[0], dtype=np.uint8)
    shoot_mask = np.asarray(raw_output[1], dtype=np.uint8)
    semantic_labels_raw = getattr(raw_output, "semantic_labels", None)
    semantic_labels = (
        np.asarray(semantic_labels_raw, dtype=np.uint8)
        if semantic_labels_raw is not None
        else None
    )
    if len(raw_output) >= 3:
        occlusion_mask = np.asarray(raw_output[2], dtype=np.uint8)
    else:
        occlusion_mask = np.zeros_like(root_mask, dtype=np.uint8)
    flagged = bool(raw_output[3]) if len(raw_output) >= 4 else False

    returns_original_coords = _returns_original_coords(features_module, profile_payload)
    if not returns_original_coords:
        root_mask = np.fliplr(root_mask)
        shoot_mask = np.fliplr(shoot_mask)
        occlusion_mask = np.fliplr(occlusion_mask)
        if semantic_labels is not None:
            semantic_labels = np.fliplr(semantic_labels)

    shape_hw = item.image.shape[:2]
    root_mask = _ensure_mask_shape(root_mask, shape_hw)
    if _should_apply_adapter_shoot_guard(effective_config):
        shoot_mask = _hard_guard_shoot_mask(shoot_mask, root_mask=root_mask, shape_hw=shape_hw, image=item.image)
    else:
        shoot_mask = _ensure_mask_shape(shoot_mask, shape_hw)
    occlusion_mask = _ensure_mask_shape(occlusion_mask, shape_hw)
    if semantic_labels is not None:
        semantic_labels = _ensure_mask_shape(semantic_labels, shape_hw)

    lateral_label_ids: tuple[int, ...] = ()
    seed_label_ids: tuple[int, ...] = ()
    if isinstance(profile_payload, dict):
        lateral_label_ids = tuple(
            int(value)
            for value in (profile_payload.get("lateral_label_ids") or ())
            if int(value) >= 0
        )
        seed_label_ids = tuple(
            int(value)
            for value in (profile_payload.get("seed_label_ids") or ())
            if int(value) >= 0
        )
    direct_lateral_labels = bool(semantic_labels is not None and lateral_label_ids)
    if direct_lateral_labels:
        lateral_root_mask = np.logical_and(
            np.isin(semantic_labels, lateral_label_ids),
            root_mask > 0,
        ).astype(np.uint8)
    else:
        lateral_root_mask = np.zeros_like(root_mask, dtype=np.uint8)
    primary_root_mask = np.logical_and(root_mask > 0, lateral_root_mask == 0).astype(np.uint8)
    seed_mask = (
        np.isin(semantic_labels, seed_label_ids).astype(np.uint8)
        if semantic_labels is not None and seed_label_ids
        else np.zeros_like(root_mask, dtype=np.uint8)
    )

    tuning_payload = None
    describe_tuning = getattr(features_module, "describe_pipeline_tuning", None)
    if callable(describe_tuning):
        try:
            tuning_payload = describe_tuning(_pipeline_tuning_overrides(effective_config.pipeline_overrides))
        except Exception:
            tuning_payload = None

    return {
        "root_mask": root_mask,
        "primary_root_mask": primary_root_mask,
        "lateral_root_mask": lateral_root_mask,
        "shoot_mask": shoot_mask,
        "seed_mask": seed_mask,
        "occlusion_mask": occlusion_mask,
        "flagged": bool(flagged),
        "profile": profile_payload,
        "tuning": tuning_payload,
        "semantic_labels": semantic_labels,
        "direct_lateral_labels": direct_lateral_labels,
    }


def _run_pyphenotyper_item_single(
    item: DatasetImageItem,
    *,
    config: PyPhenotyperConfig,
    effective_config: PyPhenotyperConfig,
    variant_label: str,
    routing_family: str,
    routing_payload: dict[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    mask_payload = _predict_masks_for_config(item, effective_config)
    lucifer_gap_repair_meta: dict[str, object] | None = None
    if bool(effective_config.lucifer_gan_gap_repair_enabled) and effective_config.lucifer_gan_gap_repair_dir is not None:
        gap_result = apply_lucifer_gan_gap_repair(
            item,
            np.asarray(mask_payload["root_mask"], dtype=np.uint8),
            effective_config.lucifer_gan_gap_repair_dir,
        )
        mask_payload["root_mask"] = np.asarray(gap_result.root_mask, dtype=np.uint8)
        lateral_root_mask = np.asarray(
            mask_payload.get("lateral_root_mask", np.zeros_like(mask_payload["root_mask"])),
            dtype=np.uint8,
        )
        lateral_root_mask = np.logical_and(lateral_root_mask > 0, mask_payload["root_mask"] > 0).astype(np.uint8)
        mask_payload["lateral_root_mask"] = lateral_root_mask
        mask_payload["primary_root_mask"] = np.logical_and(
            mask_payload["root_mask"] > 0,
            lateral_root_mask == 0,
        ).astype(np.uint8)
        lucifer_gap_repair_meta = {
            "enabled": True,
            "applied": bool(gap_result.applied),
            "mode": str(gap_result.mode),
            "stem": str(gap_result.stem) if gap_result.stem else None,
            "case_dir": str(gap_result.case_dir) if gap_result.case_dir is not None else None,
            "added_pixels": int(gap_result.added_pixels),
            "used_full_repaired_mask": bool(gap_result.used_full_repaired_mask),
            "message": str(gap_result.message or ""),
        }
    lucifer_green_shoot_rescue_meta: dict[str, object] | None = None
    if _should_apply_lucifer_green_shoot_rescue(effective_config):
        rescued_shoot_mask, lucifer_green_shoot_rescue_meta = _apply_lucifer_green_shoot_rescue(
            item.image,
            np.asarray(mask_payload["root_mask"], dtype=np.uint8),
            np.asarray(mask_payload["shoot_mask"], dtype=np.uint8),
            config=effective_config,
        )
        mask_payload["shoot_mask"] = rescued_shoot_mask
    pred = masks_to_index_prediction(
        root_mask=np.asarray(mask_payload["root_mask"], dtype=np.uint8),
        shoot_mask=np.asarray(mask_payload["shoot_mask"], dtype=np.uint8),
        seed_mask=np.asarray(mask_payload.get("seed_mask", np.zeros(item.image.shape[:2], dtype=np.uint8)), dtype=np.uint8),
        occlusion_mask=np.asarray(mask_payload["occlusion_mask"], dtype=np.uint8),
        shape_hw=item.image.shape[:2],
        root_class_id=effective_config.root_class_id,
        shoot_class_id=effective_config.shoot_class_id,
        lateral_root_mask=np.asarray(
            mask_payload.get("lateral_root_mask", np.zeros(item.image.shape[:2], dtype=np.uint8)),
            dtype=np.uint8,
        ),
        lateral_class_id=effective_config.lateral_class_id,
        seed_class_id=effective_config.seed_class_id,
        include_occlusion=effective_config.include_occlusion,
        shoot_priority_over_root=_shoot_priority_over_root(effective_config),
    )
    details: dict[str, object] = {
        "backend": "pyphenotyper",
        "pipeline_dir": str(Path(effective_config.pipeline_dir).expanduser().resolve()),
        "root_model": str(effective_config.root_model_path),
        "root_model_sha256": _artifact_sha256(str(effective_config.root_model_path)),
        "shoot_model": str(effective_config.shoot_model_path),
        "shoot_model_sha256": _artifact_sha256(str(effective_config.shoot_model_path)),
        "seed_model": str(effective_config.seed_model_path) if effective_config.seed_model_path is not None else None,
        "seed_model_sha256": (
            _artifact_sha256(str(effective_config.seed_model_path))
            if effective_config.seed_model_path is not None
            else None
        ),
        "patch_size": int(effective_config.patch_size),
        "refinement_steps": int(effective_config.refinement_steps),
        "root_class_id": int(effective_config.root_class_id),
        "shoot_class_id": int(effective_config.shoot_class_id),
        "lateral_class_id": (
            int(effective_config.lateral_class_id)
            if effective_config.lateral_class_id is not None
            else None
        ),
        "seed_class_id": int(effective_config.seed_class_id) if effective_config.seed_class_id is not None else None,
        "include_occlusion": bool(effective_config.include_occlusion),
        "bbox_tracking_enabled": bool(effective_config.enable_bbox_tracking),
        "pixel_size_mm": float(effective_config.pixel_size_mm),
        "routing_variant": variant_label,
        "routing_family": routing_family,
        "hybrid_routing_enabled": _routing_enabled(config),
        "flagged": bool(mask_payload["flagged"]),
        "output_shape": [int(pred.shape[0]), int(pred.shape[1])],
        "output_dtype": str(pred.dtype),
        "profile": mask_payload["profile"],
        "tuning": mask_payload["tuning"],
        "lucifer_gan_gap_repair": lucifer_gap_repair_meta,
        "lucifer_green_shoot_rescue": lucifer_green_shoot_rescue_meta,
        "direct_lateral_labels": bool(mask_payload.get("direct_lateral_labels", False)),
        "direct_lateral_pixels": int(
            np.count_nonzero(
                np.asarray(
                    mask_payload.get("lateral_root_mask", np.zeros(item.image.shape[:2], dtype=np.uint8)),
                    dtype=np.uint8,
                )
            )
        ),
    }
    details.update(
        _mask_qc_summary(
            np.asarray(mask_payload["root_mask"], dtype=np.uint8),
            np.asarray(mask_payload["shoot_mask"], dtype=np.uint8),
            np.asarray(mask_payload.get("seed_mask", np.zeros(item.image.shape[:2], dtype=np.uint8)), dtype=np.uint8),
        )
    )
    if isinstance(routing_payload, dict):
        details.update(routing_payload)
    return pred, details, mask_payload


def _expert_calibration_bias(config: PyPhenotyperConfig, expert_key: str) -> float:
    profile = config.calibration_profile if isinstance(config.calibration_profile, dict) else {}
    raw_bias = profile.get("expert_score_bias", {})
    if not isinstance(raw_bias, dict):
        return 0.0
    try:
        return float(raw_bias.get(str(expert_key), 0.0) or 0.0)
    except Exception:
        return 0.0


def _candidate_experts_for_item(
    item: DatasetImageItem,
    config: PyPhenotyperConfig,
) -> tuple[object, list[dict[str, object]]]:
    features = describe_image_features(item.image)
    image_arr = np.asarray(item.image)
    has_rgb_channels = bool(image_arr.ndim == 3 and image_arr.shape[2] >= 3)
    name_hint = f"{item.name} {item.path}".lower() if getattr(item, "path", None) is not None else str(item.name).lower()
    candidates: list[dict[str, object]] = []
    for expert in config.expert_variants:
        image_mode = str(expert.image_mode or "any").strip().lower()
        if image_mode == "rgb_only" and not has_rgb_channels:
            continue
        if image_mode == "grayscale_only" and features.is_rgb_like and has_rgb_channels:
            continue
        prior_mode = image_mode
        if image_mode == "rgb_only" and has_rgb_channels and not features.is_rgb_like:
            prior_mode = "any"
        prior = expert_prior_score(
            features,
            expert_key=expert.key,
            image_mode=prior_mode,
            family=expert.family,
            priority=expert.priority,
            hints=expert.router_hints,
            calibration_bias=_expert_calibration_bias(config, expert.key),
        )
        expert_key = str(expert.key or "").lower()
        if "potato" in name_hint and "potato" in expert_key:
            prior += 1.30
        elif "potato" in name_hint and "potato" not in expert_key:
            prior -= 0.25
        if "lucifer" in name_hint and "lucifer" in expert_key:
            prior += 0.30
        if "hades" in name_hint and "hades" in expert_key:
            prior += 0.30
        candidates.append(
            {
                "expert": expert,
                "prior_score": float(prior),
            }
        )
    if not candidates:
        for expert in config.expert_variants:
            prior = expert_prior_score(
                features,
                expert_key=expert.key,
                image_mode="any",
                family=expert.family,
                priority=expert.priority,
                hints=expert.router_hints,
                calibration_bias=_expert_calibration_bias(config, expert.key),
            )
            candidates.append({"expert": expert, "prior_score": float(prior)})
    candidates.sort(key=lambda row: float(row["prior_score"]), reverse=True)
    return features, candidates


@lru_cache(maxsize=4)
def _load_keras_model_cached(model_path: str):
    def _annotate(model):
        try:
            setattr(model, "_npec_source_path", model_path)
        except Exception:
            pass
        return model

    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("TensorFlow is required to run PyPhenotyper models.") from exc
    try:
        return _annotate(tf.keras.models.load_model(model_path, compile=False))
    except Exception as exc:
        text = str(exc)
        if "Conv2DTranspose" not in text or "groups" not in text:
            raise

        class CompatConv2DTranspose(tf.keras.layers.Conv2DTranspose):
            def __init__(self, *args, **kwargs):
                groups = kwargs.pop("groups", 1)
                if groups not in (None, 1):
                    raise ValueError(f"Unsupported Conv2DTranspose groups value: {groups}")
                super().__init__(*args, **kwargs)

            @classmethod
            def from_config(cls, config):
                cfg = dict(config)
                groups = cfg.pop("groups", 1)
                if groups not in (None, 1):
                    raise ValueError(f"Unsupported Conv2DTranspose groups value in config: {groups}")
                return super().from_config(cfg)

        custom_objects = {
            "Conv2DTranspose": CompatConv2DTranspose,
            "keras.layers.Conv2DTranspose": CompatConv2DTranspose,
            "tf.keras.layers.Conv2DTranspose": CompatConv2DTranspose,
        }
        try:
            return _annotate(tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects))
        except Exception as compat_exc:
            raise RuntimeError(
                "PyPhenotyper model deserialization failed and compatibility fallback did not recover it. "
                f"Original error: {exc} | Fallback error: {compat_exc}"
            ) from compat_exc


@lru_cache(maxsize=4)
def _load_pyphenotyper_features_module(pipeline_dir: str):
    package_root, _ = _normalize_pipeline_dir(Path(pipeline_dir))
    with _prepend_sys_path(package_root):
        try:
            module = importlib.import_module("pyphenotyper.features.features")
        except ModuleNotFoundError as exc:
            missing = str(getattr(exc, "name", "") or "").strip()
            if missing.startswith("skimage"):
                raise RuntimeError(
                    "scikit-image is required by the selected PyPhenotyper pipeline but was not found in this runtime. "
                    "Install `scikit-image` in your environment or rebuild the packaged app with scikit-image included."
                ) from exc
            raise
    # Patch the module-level loader used by model_create_masks/model_predict_*
    module.load_image = _safe_load_image_for_pipeline
    return module


@lru_cache(maxsize=1)
def _load_skimage_skeletonize():
    try:
        from skimage.morphology import skeletonize  # type: ignore
    except Exception:
        return None
    return skeletonize


def _ensure_item_path(item: DatasetImageItem) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if item.path is not None and item.path.exists():
        return item.path, None
    tmp_dir = tempfile.TemporaryDirectory(prefix="npec_pyphenotyper_")
    temp_path = Path(tmp_dir.name) / f"{item.uid}.png"
    Image.fromarray(item.image).save(temp_path)
    return temp_path, tmp_dir


def _ensure_mask_shape(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask.astype(np.uint8, copy=False)
    resized = np.array(Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST), dtype=np.uint8)
    return resized


def _ensure_gray_image(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.shape != shape_hw:
        arr = np.asarray(Image.fromarray(arr).resize((shape_hw[1], shape_hw[0]), Image.BILINEAR), dtype=np.uint8)
    return arr


def _filter_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return binary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for label_id in range(1, int(num_labels)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            out[labels == label_id] = 1
    return out


def _fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return binary
    inverse = (binary == 0).astype(np.uint8)
    count, labels, _stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    if count <= 1:
        return binary
    border_labels = set(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[-1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, -1]).tolist())
    out = binary.copy()
    for label_id in range(1, int(count)):
        if label_id in border_labels:
            continue
        out[labels == label_id] = 1
    return out


def _largest_centered_component(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if np.count_nonzero(binary) <= 0:
        return binary
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape[:2]
    center_x = float(width) * 0.5
    center_y = float(height) * 0.5
    best_label = 0
    best_score = -1e18
    for label_id in range(1, int(num_labels)):
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


def _estimate_dish_interior_mask(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    gray = _ensure_gray_image(image, shape_hw)
    if gray.size <= 0:
        return np.ones(shape_hw, dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray, (0, 0), 5.0)
    _thr, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    interior = _largest_centered_component(thresh > 0)
    if np.count_nonzero(interior) <= 0:
        interior = _largest_centered_component(gray > 40)
    if np.count_nonzero(interior) <= 0:
        return np.ones(shape_hw, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    interior = cv2.morphologyEx(interior.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
    interior = cv2.morphologyEx(interior.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    interior = cv2.dilate(interior.astype(np.uint8), kernel, iterations=1)
    return (interior > 0).astype(np.uint8)


def _ensure_rgb_image(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[..., :3]
    else:
        arr = np.zeros((shape_hw[0], shape_hw[1], 3), dtype=np.uint8)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[:2] != shape_hw:
        arr = np.asarray(Image.fromarray(arr).resize((shape_hw[1], shape_hw[0]), Image.BILINEAR), dtype=np.uint8)
    return arr


def _override_enabled(raw_value: object) -> bool:
    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if value in {"", "0", "false", "off", "no", "none", "disabled"}:
            return False
        if value in {"1", "true", "on", "yes", "enabled", "lucifer", "lucifer_green", "green"}:
            return True
    return bool(raw_value)


def _should_apply_lucifer_green_shoot_rescue(config: PyPhenotyperConfig) -> bool:
    overrides = config.pipeline_overrides if isinstance(config.pipeline_overrides, dict) else {}
    mode = str(overrides.get("shoot_color_rescue_mode", "") or "").strip().lower()
    if mode in {"off", "none", "disabled", "false", "0"}:
        return False
    if mode in {"lucifer_green", "lucifer", "green", "rgb_green", "vegetation", "green_only", "rgb_green_only"}:
        return True
    return _override_enabled(overrides.get("lucifer_green_shoot_rescue", False))


def _lucifer_green_shoot_rescue_mode(config: PyPhenotyperConfig) -> str:
    overrides = config.pipeline_overrides if isinstance(config.pipeline_overrides, dict) else {}
    mode = str(overrides.get("shoot_color_rescue_mode", "") or "").strip().lower()
    if mode in {"merge", "rgb_green_merge", "lucifer_green_merge"}:
        return "merge"
    return "green_only"


def _shoot_color_rescue_eligibility(config: PyPhenotyperConfig) -> tuple[bool, str]:
    overrides = config.pipeline_overrides if isinstance(config.pipeline_overrides, dict) else {}
    if _override_enabled(overrides.get("shoot_color_rescue_model_locked", False)):
        return False, "model_locked"

    raw_eligibility: object | None = None
    if "shoot_color_rescue_eligible" in overrides:
        raw_eligibility = overrides.get("shoot_color_rescue_eligible")
    elif "shoot_color_rescue_eligibility" in overrides:
        raw_eligibility = overrides.get("shoot_color_rescue_eligibility")
    if raw_eligibility is None:
        return True, "default_eligible"

    if isinstance(raw_eligibility, str):
        normalized = raw_eligibility.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {
            "0",
            "disabled",
            "false",
            "ineligible",
            "locked",
            "model_locked",
            "model_only",
            "no",
            "off",
        }:
            return False, normalized or "override_ineligible"
        if normalized in {"1", "eligible", "enabled", "true", "yes", "on", "rgb_crown"}:
            return True, normalized
    eligible = bool(raw_eligibility)
    return eligible, "override_eligible" if eligible else "override_ineligible"


def _shoot_color_rescue_failure_policy(config: PyPhenotyperConfig) -> str:
    overrides = config.pipeline_overrides if isinstance(config.pipeline_overrides, dict) else {}
    raw_policy = str(overrides.get("shoot_color_rescue_failure_policy", "retain_model") or "retain_model")
    normalized = raw_policy.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {
        "fallback_to_model",
        "keep_model",
        "keep_model_mask",
        "model_fallback",
        "preserve_model",
        "preserve_model_mask",
        "retain_model",
        "retain_model_mask",
    }:
        return "retain_model"
    # An empty color result is an abstention, never evidence that the model mask
    # should be deleted. Unknown policies therefore fail closed to the model.
    return "retain_model"


def _build_lucifer_green_shoot_mask(
    image: np.ndarray,
    shape_hw: tuple[int, int],
    root_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    meta: dict[str, object] = {
        "rgb_like": bool(_is_rgb_like_image(image)),
        "candidate_pixels": 0,
        "components_kept": 0,
        "filter": "arabidopsis_rosette_rgb_chroma",
        "root_context_enabled": False,
        "root_context_supported_components": 0,
        "olive_root_supported_components": 0,
        "green_chain_supported_components": 0,
        "large_root_supported_components": 0,
        "root_crown_gate_rejections": 0,
    }
    if not bool(meta["rgb_like"]):
        return np.zeros(shape_hw, dtype=np.uint8), meta

    height, width = shape_hw
    top_cut = int(round(height * 0.055))
    bottom_cut = int(round(height * 0.43))
    edge_margin = max(20, min(260, int(round(width * 0.055))))
    x0 = max(0, min(width, edge_margin))
    x1 = max(x0, min(width, width - edge_margin))
    y0 = max(0, min(height, top_cut))
    y1 = max(y0, min(height, bottom_cut))
    if x1 <= x0 or y1 <= y0:
        return np.zeros(shape_hw, dtype=np.uint8), meta

    rgb = _ensure_rgb_image(image, shape_hw)
    rgb_roi = rgb[y0:y1, x0:x1]
    r = rgb_roi[..., 0].astype(np.int16)
    g = rgb_roi[..., 1].astype(np.int16)
    b = rgb_roi[..., 2].astype(np.int16)
    exg_roi = (2 * g) - r - b
    max_channel = np.maximum(np.maximum(r, g), b)
    min_channel = np.minimum(np.minimum(r, g), b)
    channel_spread_roi = max_channel - min_channel
    channel_total_roi = np.maximum(1, r + g + b).astype(np.float32)
    green_fraction_roi = g.astype(np.float32) / channel_total_roi
    red_fraction_roi = r.astype(np.float32) / channel_total_roi
    blue_fraction_roi = b.astype(np.float32) / channel_total_roi
    green_chroma_roi = green_fraction_roi - np.maximum(red_fraction_roi, blue_fraction_roi)
    hsv_roi = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2HSV)
    hue_roi = hsv_roi[..., 0].astype(np.int16)
    saturation_roi = hsv_roi[..., 1].astype(np.int16)
    value_roi = hsv_roi[..., 2].astype(np.int16)
    broad_green_hue_roi = (hue_roi >= 24) & (hue_roi <= 96)
    strict_green_hue_roi = (hue_roi >= 32) & (hue_roi <= 92)
    olive_green_hue_roi = (hue_roi >= 27) & (hue_roi <= 60)

    broad_candidate = (
        broad_green_hue_roi
        & (saturation_roi >= 34)
        & (value_roi >= 28)
        & (value_roi <= 205)
        & (g >= b + 10)
        & (g >= r + 3)
        & (exg_roi >= 18)
        & (green_fraction_roi >= 0.350)
        & (green_chroma_roi >= 0.006)
    )
    strict_candidate = (
        strict_green_hue_roi
        & (saturation_roi >= 48)
        & (g >= r + 11)
        & (g >= b + 24)
        & (exg_roi >= 58)
        & (channel_spread_roi >= 34)
        & (green_fraction_roi >= 0.380)
        & (green_chroma_roi >= 0.026)
        & (blue_fraction_roi <= 0.32)
        & (g >= 36)
        & (max_channel <= 210)
    )
    olive_candidate = (
        olive_green_hue_roi
        & (saturation_roi >= 88)
        & (value_roi >= 24)
        & (value_roi <= 195)
        & (g >= r + 2)
        & (g >= b + 28)
        & (exg_roi >= 30)
        & (channel_spread_roi >= 35)
        & (green_fraction_roi >= 0.370)
        & (green_chroma_roi >= 0.010)
    )

    raw_candidate = (broad_candidate | olive_candidate).astype(np.uint8)
    candidate_roi = cv2.morphologyEx(raw_candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    # Keep neighboring leaves and greenish colonies separate while scoring color;
    # a wider close is applied only after accepted components have been selected.
    candidate_roi = cv2.morphologyEx(candidate_roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_roi, connectivity=8)
    kept_roi = np.zeros_like(candidate_roi, dtype=np.uint8)
    green_minus_red_roi = g - r
    green_minus_blue_roi = g - b
    root_distance_roi: np.ndarray | None = None
    root_coord_y: np.ndarray | None = None
    root_coord_x: np.ndarray | None = None
    if root_mask is not None:
        try:
            root_arr = _ensure_mask_shape((np.asarray(root_mask, dtype=np.uint8) > 0).astype(np.uint8), shape_hw)
            root_roi = root_arr[y0:y1, x0:x1] > 0
            if int(np.count_nonzero(root_roi)) > 0:
                root_distance_roi = cv2.distanceTransform((~root_roi).astype(np.uint8), cv2.DIST_L2, 5)
                root_coord_y, root_coord_x = np.nonzero(root_roi)
                meta["root_context_enabled"] = True
        except Exception:
            root_distance_roi = None

    plate_area = float(height * width)
    min_area = max(25, int(round(plate_area * 0.000003)))
    max_unanchored_area = max(min_area + 1, 12000, int(round(plate_area * 0.008)))
    max_root_supported_area = max(max_unanchored_area, 18000, int(round(plate_area * 0.025)))
    root_support_radius = max(16, min(300, int(round(float(max(height, width)) * 0.055))))
    olive_root_support_radius = max(24, min(300, int(round(float(max(height, width)) * 0.075))))
    crown_vertical_tolerance = max(4, min(48, int(round(float(height) * 0.008))))
    components_kept = 0
    root_supported_components = 0
    olive_root_supported_components = 0
    green_chain_supported_components = 0
    large_root_supported_components = 0
    root_crown_gate_rejections = 0
    component_candidates: list[tuple[float, int, int]] = []
    deferred_olive_candidates: list[tuple[float, int, int]] = []
    for label_id in range(1, int(count)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area or area > max_root_supported_area:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT]) + x0
        y = int(stats[label_id, cv2.CC_STAT_TOP]) + y0
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if y + h <= int(round(height * 0.06)):
            continue
        component = labels == label_id
        strict_fraction = float(np.count_nonzero(strict_candidate[component])) / float(max(1, area))
        olive_fraction = float(np.count_nonzero(olive_candidate[component])) / float(max(1, area))
        mean_exg = float(np.mean(exg_roi[component]))
        mean_green_minus_red = float(np.mean(green_minus_red_roi[component]))
        mean_green_minus_blue = float(np.mean(green_minus_blue_roi[component]))
        mean_channel_spread = float(np.mean(channel_spread_roi[component]))
        mean_saturation = float(np.mean(saturation_roi[component]))
        mean_green_fraction = float(np.mean(green_fraction_roi[component]))
        mean_blue_fraction = float(np.mean(blue_fraction_roi[component]))
        mean_green_chroma = float(np.mean(green_chroma_roi[component]))
        root_supported = False
        olive_root_supported = False
        if root_distance_roi is not None:
            try:
                min_root_distance = float(np.min(root_distance_roi[component]))
                root_supported = min_root_distance <= float(root_support_radius)
                olive_root_supported = min_root_distance <= float(olive_root_support_radius)
            except Exception:
                root_supported = False
                olive_root_supported = False
        component_above_crown = False
        if root_coord_y is not None and root_coord_x is not None:
            local_left = int(stats[label_id, cv2.CC_STAT_LEFT])
            local_right = int(local_left + stats[label_id, cv2.CC_STAT_WIDTH])
            x_padding = int(max(root_support_radius, min(olive_root_support_radius, w)))
            nearby_root = (root_coord_x >= local_left - x_padding) & (root_coord_x <= local_right + x_padding)
            if int(np.count_nonzero(nearby_root)) > 0:
                local_crown_y = float(np.percentile(root_coord_y[nearby_root], 5.0))
                component_center_y = float(centroids[label_id][1])
                component_above_crown = component_center_y <= local_crown_y + float(crown_vertical_tolerance)
        if (root_supported or olive_root_supported) and not component_above_crown:
            root_crown_gate_rejections += 1
        area_root_supported = bool((root_supported or olive_root_supported) and component_above_crown)
        if area > (max_root_supported_area if area_root_supported else max_unanchored_area):
            continue
        max_component_width = int(round(width * (0.62 if area_root_supported else 0.34)))
        max_component_height = int(round(height * (0.36 if area_root_supported else 0.30)))
        if w > max_component_width or h > max_component_height:
            continue

        conservative_color = (
            strict_fraction >= 0.08
            and mean_exg >= 45
            and mean_green_minus_red >= 8
            and mean_green_minus_blue >= 24
            and mean_saturation >= 48
            and mean_green_fraction >= 0.370
            and mean_blue_fraction <= 0.34
            and mean_green_chroma >= 0.020
        )
        root_context_color = (
            root_supported
            and component_above_crown
            and mean_exg >= 40
            and mean_green_minus_red >= 3.5
            and mean_green_minus_blue >= 32
            and mean_channel_spread >= 32
            and mean_saturation >= 68
            and mean_green_fraction >= 0.360
            and mean_blue_fraction <= 0.36
            and mean_green_chroma >= 0.014
        )
        olive_root_context_color = (
            olive_root_supported
            and component_above_crown
            and olive_fraction >= 0.50
            and mean_exg >= 38
            and mean_green_minus_red >= 2.5
            and mean_green_minus_blue >= 34
            and mean_channel_spread >= 38
            and mean_saturation >= 88
            and mean_green_fraction >= 0.370
            and mean_blue_fraction <= 0.34
            and mean_green_chroma >= 0.012
        )
        olive_chain_color = (
            olive_fraction >= 0.35
            and mean_exg >= 34
            and mean_green_minus_red >= 2.5
            and mean_green_minus_blue >= 30
            and mean_channel_spread >= 34
            and mean_saturation >= 78
            and mean_green_fraction >= 0.365
            and mean_blue_fraction <= 0.35
            and mean_green_chroma >= 0.010
        )
        if (
            not conservative_color
            and not root_context_color
            and not olive_root_context_color
        ):
            if olive_chain_color:
                deferred_olive_candidates.append((float(olive_fraction), int(area), int(label_id)))
            continue
        if root_context_color:
            root_supported_components += 1
        if olive_root_context_color:
            olive_root_supported_components += 1
        if area_root_supported and area > max_unanchored_area:
            large_root_supported_components += 1
        priority = float(
            strict_fraction
            + (0.35 if root_context_color else 0.0)
            + (0.30 if olive_root_context_color else 0.0)
        )
        component_candidates.append((priority, int(area), int(label_id)))

    accepted_label_ids = {int(entry[2]) for entry in component_candidates}
    green_chain_radius = max(12, min(360, int(round(float(max(height, width)) * 0.070))))
    remaining_deferred = list(deferred_olive_candidates)
    for _pass in range(3):
        if not accepted_label_ids or not remaining_deferred:
            break
        accepted_support = np.isin(labels, tuple(sorted(accepted_label_ids)))
        accepted_distance = cv2.distanceTransform((~accepted_support).astype(np.uint8), cv2.DIST_L2, 5)
        accepted_this_pass: list[tuple[float, int, int]] = []
        still_deferred: list[tuple[float, int, int]] = []
        for olive_fraction, area, label_id in remaining_deferred:
            component = labels == int(label_id)
            try:
                nearby_green = float(np.min(accepted_distance[component])) <= float(green_chain_radius)
            except Exception:
                nearby_green = False
            if nearby_green:
                accepted_this_pass.append((0.20 + float(olive_fraction), int(area), int(label_id)))
                accepted_label_ids.add(int(label_id))
                green_chain_supported_components += 1
            else:
                still_deferred.append((float(olive_fraction), int(area), int(label_id)))
        component_candidates.extend(accepted_this_pass)
        remaining_deferred = still_deferred
        if not accepted_this_pass:
            break

    max_total_pixels = max(6000, int(round(plate_area * 0.060)))
    selected_total = 0
    for _strong_fraction, _area, label_id in sorted(component_candidates, key=lambda entry: (-entry[0], -entry[1], entry[2])):
        component = labels == label_id
        component_pixels = int(np.count_nonzero(raw_candidate[component]))
        if selected_total > 0 and selected_total + component_pixels > max_total_pixels:
            continue
        kept_roi[component] = raw_candidate[component]
        selected_total += int(component_pixels)
        components_kept += 1

    kept_roi = (kept_roi > 0).astype(np.uint8)
    kept_roi = cv2.morphologyEx(kept_roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    kept = np.zeros(shape_hw, dtype=np.uint8)
    kept[y0:y1, x0:x1] = kept_roi
    meta["candidate_pixels"] = int(np.count_nonzero(kept))
    meta["components_kept"] = int(components_kept)
    meta["root_context_supported_components"] = int(root_supported_components)
    meta["olive_root_supported_components"] = int(olive_root_supported_components)
    meta["green_chain_supported_components"] = int(green_chain_supported_components)
    meta["large_root_supported_components"] = int(large_root_supported_components)
    meta["root_crown_gate_rejections"] = int(root_crown_gate_rejections)
    meta["filter"] = "arabidopsis_rosette_rgb_chroma_root_context"
    meta["max_component_area_px"] = int(max_root_supported_area)
    meta["max_unanchored_component_area_px"] = int(max_unanchored_area)
    meta["max_root_supported_component_area_px"] = int(max_root_supported_area)
    meta["max_total_pixels"] = int(max_total_pixels)
    meta["root_support_radius_px"] = int(root_support_radius)
    meta["olive_root_support_radius_px"] = int(olive_root_support_radius)
    meta["green_chain_radius_px"] = int(green_chain_radius)
    meta["root_crown_vertical_tolerance_px"] = int(crown_vertical_tolerance)
    return kept.astype(np.uint8), meta


def build_lucifer_green_shoot_mask(
    image: np.ndarray,
    shape_hw: tuple[int, int],
    root_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return a green/olive RGB shoot candidate mask without merging it into model masks."""
    return _build_lucifer_green_shoot_mask(image, shape_hw, root_mask=root_mask)


def _apply_lucifer_green_shoot_rescue(
    image: np.ndarray,
    root_mask: np.ndarray,
    shoot_mask: np.ndarray,
    *,
    config: PyPhenotyperConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    shape_hw = tuple(int(value) for value in np.asarray(shoot_mask).shape[:2])
    shoot_bin = _ensure_mask_shape((np.asarray(shoot_mask, dtype=np.uint8) > 0).astype(np.uint8), shape_hw)
    model_pixels = int(np.count_nonzero(shoot_bin))
    eligible, eligibility_reason = _shoot_color_rescue_eligibility(config)
    meta: dict[str, object] = {
        "enabled": bool(_should_apply_lucifer_green_shoot_rescue(config)),
        "eligible": bool(eligible),
        "eligibility_reason": str(eligibility_reason),
        "failure_policy": _shoot_color_rescue_failure_policy(config),
        "applied": False,
        "abstained": False,
        "status": "pending",
        "detector_status": "not_run",
        "fallback_status": "not_needed",
        "candidate_pixels": 0,
        "added_pixels": 0,
        "removed_model_pixels": 0,
        "retained_model_pixels": 0,
        "mode": _lucifer_green_shoot_rescue_mode(config),
        "output_pixels": model_pixels,
    }
    if not bool(meta["enabled"]):
        meta["status"] = "disabled"
        return shoot_bin, meta
    if not bool(meta["eligible"]):
        meta["status"] = "ineligible_route"
        meta["detector_status"] = "not_run_ineligible"
        meta["fallback_status"] = "retained_model_mask"
        meta["retained_model_pixels"] = model_pixels
        return shoot_bin, meta

    color_mask, color_meta = _build_lucifer_green_shoot_mask(image, shape_hw, root_mask=root_mask)
    meta.update(color_meta)
    candidate_pixels = int(np.count_nonzero(color_mask))
    minimum_candidate_pixels = max(20, int(round(float(shape_hw[0] * shape_hw[1]) * 0.000001)))
    meta["candidate_pixels"] = candidate_pixels
    meta["minimum_candidate_pixels"] = int(minimum_candidate_pixels)
    if candidate_pixels < minimum_candidate_pixels:
        abstention_reason = "no_candidates" if bool(color_meta.get("rgb_like", False)) else "not_rgb_like"
        meta["abstained"] = True
        meta["status"] = f"abstained_{abstention_reason}"
        meta["detector_status"] = f"abstained_{abstention_reason}"
        meta["fallback_status"] = "retained_model_mask"
        meta["retained_model_pixels"] = model_pixels
        return shoot_bin, meta

    meta["detector_status"] = "candidates_accepted"
    if str(meta["mode"]) == "green_only":
        green_only = (np.asarray(color_mask, dtype=np.uint8) > 0).astype(np.uint8)
        meta["added_pixels"] = int(np.count_nonzero((green_only > 0) & (shoot_bin == 0)))
        meta["removed_model_pixels"] = int(np.count_nonzero((shoot_bin > 0) & (green_only == 0)))
        meta["retained_model_pixels"] = int(np.count_nonzero((shoot_bin > 0) & (green_only > 0)))
        meta["output_pixels"] = int(np.count_nonzero(green_only))
        meta["applied"] = bool(meta["added_pixels"] or meta["removed_model_pixels"])
        meta["status"] = "applied" if bool(meta["applied"]) else "unchanged"
        return green_only.astype(np.uint8), meta

    added_from_color = int(np.count_nonzero((color_mask > 0) & (shoot_bin == 0)))
    combined = np.logical_or(shoot_bin > 0, color_mask > 0).astype(np.uint8)
    combined = _fill_enclosed_holes(combined)
    meta["added_pixels"] = added_from_color
    meta["filled_pixels"] = int(max(0, np.count_nonzero((combined > 0) & (shoot_bin == 0)) - added_from_color))
    meta["output_pixels"] = int(np.count_nonzero(combined))
    meta["applied"] = bool(added_from_color > 0)
    meta["status"] = "applied" if bool(meta["applied"]) else "unchanged"
    return combined.astype(np.uint8), meta


def _build_fixed_shoot_slots(shape_hw: tuple[int, int], top_limit: int | None = None) -> np.ndarray:
    height, width = int(shape_hw[0]), int(shape_hw[1])
    guard = np.zeros((height, width), dtype=np.uint8)
    slot_half_width = max(110, min(420, int(round(width * 0.06))))
    slot_bottom = min(height, int(top_limit)) if top_limit is not None else min(height, max(180, int(round(height * 0.22))))
    slot_bottom = max(80, min(slot_bottom, height))
    for center_x in np.linspace(width * 0.12, width * 0.88, 5):
        cx = int(round(center_x))
        x0 = max(0, cx - slot_half_width)
        x1 = min(width, cx + slot_half_width)
        guard[:slot_bottom, x0:x1] = 1
    edge_margin = max(0, min(int(round(width * 0.1)), 420))
    if edge_margin > 0:
        guard[:, :edge_margin] = 0
        guard[:, width - edge_margin :] = 0
    return guard


def _build_root_anchored_shoot_guard(
    shape_hw: tuple[int, int],
    root_mask: np.ndarray | None,
    dish_mask: np.ndarray | None = None,
) -> np.ndarray:
    height, width = int(shape_hw[0]), int(shape_hw[1])
    guard = np.zeros((height, width), dtype=np.uint8)
    top_limit = max(120, min(height, int(round(height * 0.26))))
    dish_bbox: tuple[int, int, int, int] | None = None
    if dish_mask is not None and np.count_nonzero(dish_mask) > 0:
        ys, xs = np.where(dish_mask > 0)
        if xs.size > 0 and ys.size > 0:
            dish_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    if root_mask is not None:
        root_bin = (np.asarray(root_mask, dtype=np.uint8) > 0).astype(np.uint8)
        root_bin = _ensure_mask_shape(root_bin, shape_hw)
        min_area = max(120, int(round(float(height * width) * 0.00002)))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(root_bin, connectivity=8)
        components: list[tuple[int, int, int, int, int]] = []
        for label_id in range(1, int(num_labels)):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            if y >= max(160, int(round(height * 0.45))):
                continue
            if dish_bbox is not None:
                dish_x0, _dish_y0, dish_x1, _dish_y1 = dish_bbox
                if x <= dish_x0 or (x + w) >= dish_x1:
                    continue
            components.append((x, y, w, h, label_id))
        components.sort(key=lambda item: item[0])
        pad_x = max(120, min(420, int(round(width * 0.07))))
        pad_up = max(90, min(240, int(round(height * 0.06))))
        pad_down = max(120, min(380, int(round(height * 0.12))))
        for _x, _y, _w, _h, label_id in components[:5]:
            ys, xs = np.where(labels == label_id)
            if xs.size <= 0:
                continue
            top_index = int(np.argmin(ys))
            anchor_x = int(xs[top_index])
            anchor_y = int(ys[top_index])
            x0 = max(0, anchor_x - pad_x)
            x1 = min(width, anchor_x + pad_x)
            y0 = max(0, anchor_y - pad_up)
            y1 = min(top_limit, anchor_y + pad_down)
            guard[y0:y1, x0:x1] = 1
    if np.count_nonzero(guard) > 0:
        fixed_guard = _build_fixed_shoot_slots(shape_hw, top_limit=top_limit)
        guard = np.logical_or(guard > 0, fixed_guard > 0).astype(np.uint8)
    else:
        guard = _build_fixed_shoot_slots(shape_hw, top_limit=top_limit)
    if dish_mask is not None and np.count_nonzero(dish_mask) > 0:
        guard = np.logical_and(guard > 0, dish_mask > 0).astype(np.uint8)
    return guard.astype(np.uint8)


def _hard_guard_shoot_mask(
    shoot_mask: np.ndarray,
    root_mask: np.ndarray | None,
    shape_hw: tuple[int, int],
    image: np.ndarray | None = None,
) -> np.ndarray:
    shoot_bin = _ensure_mask_shape((np.asarray(shoot_mask, dtype=np.uint8) > 0).astype(np.uint8), shape_hw)
    if np.count_nonzero(shoot_bin) == 0:
        return shoot_bin

    height, width = shape_hw
    min_area = max(40, int(round(float(height * width) * 0.00002)))
    filtered = _filter_components(shoot_bin, min_area=min_area)

    edge_margin = max(0, min(int(round(width * 0.1)), 420))
    if edge_margin > 0:
        filtered[:, :edge_margin] = 0
        filtered[:, width - edge_margin :] = 0

    lower_start = min(height, max(0, int(round(height * 0.72))))
    if lower_start < height:
        filtered[lower_start:, :] = 0

    dish_mask = None
    if image is not None:
        dish_mask = _estimate_dish_interior_mask(image, shape_hw)
        filtered = np.logical_and(filtered > 0, dish_mask > 0).astype(np.uint8)

    guard = _build_root_anchored_shoot_guard(shape_hw, root_mask=root_mask, dish_mask=dish_mask)
    guarded = np.logical_and(filtered > 0, guard > 0).astype(np.uint8)
    guarded = _filter_components(guarded, min_area=min_area)

    if np.count_nonzero(guarded) >= max(20, int(round(np.count_nonzero(filtered) * 0.08))):
        target = guarded.astype(np.uint8)
    else:
        target = filtered.astype(np.uint8)
    return _fill_enclosed_holes(target).astype(np.uint8)


def _returns_original_coords(features_module, profile_payload: dict[str, object] | None) -> bool:
    if isinstance(profile_payload, dict) and "return_original_coords" in profile_payload:
        try:
            return bool(profile_payload.get("return_original_coords"))
        except Exception:
            pass
    return bool(getattr(features_module, "RETURNS_ORIGINAL_COORDS", False))


def _should_apply_adapter_shoot_guard(config: PyPhenotyperConfig) -> bool:
    overrides = config.pipeline_overrides if isinstance(config.pipeline_overrides, dict) else {}
    guard_mode = str(overrides.get("adapter_shoot_guard_mode", "") or "").strip().lower()
    if guard_mode in {"off", "none", "disabled", "legacy"}:
        return False
    postprocess_mode = str(overrides.get("shoot_postprocess_mode", "") or "").strip().lower()
    if postprocess_mode in {"legacy", "old", "legacy_mask", "legacy_shoot"} and not guard_mode:
        return False
    return True


def _shoot_priority_over_root(config: PyPhenotyperConfig) -> bool:
    overrides = config.pipeline_overrides if isinstance(config.pipeline_overrides, dict) else {}
    raw_value = overrides.get("shoot_priority_over_root", True)
    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if value in {"0", "false", "off", "no", "disabled"}:
            return False
        if value in {"1", "true", "on", "yes", "enabled"}:
            return True
    return bool(raw_value)


def _mask_qc_summary(root_mask: np.ndarray, shoot_mask: np.ndarray, seed_mask: np.ndarray | None = None) -> dict[str, object]:
    root_bin = np.asarray(root_mask, dtype=np.uint8) > 0
    shoot_bin = np.asarray(shoot_mask, dtype=np.uint8) > 0
    seed_bin = np.asarray(seed_mask, dtype=np.uint8) > 0 if seed_mask is not None else np.zeros_like(root_bin, dtype=bool)
    root_pixels = int(np.count_nonzero(root_bin))
    shoot_pixels = int(np.count_nonzero(shoot_bin))
    seed_pixels = int(np.count_nonzero(seed_bin))
    overlap_pixels = int(np.count_nonzero(root_bin & shoot_bin))
    image_pixels = max(1, int(root_bin.size))
    shoot_to_root = float(shoot_pixels / max(1, root_pixels))
    overlap_to_root = float(overlap_pixels / max(1, root_pixels))
    shoot_area_fraction = float(shoot_pixels / image_pixels)
    flags: list[str] = []
    if root_pixels <= 0:
        flags.append("empty_root")
    if shoot_pixels <= 0:
        flags.append("empty_shoot")
    if shoot_area_fraction > 0.22:
        flags.append("shoot_area_too_large")
    if shoot_to_root > 2.5:
        flags.append("shoot_root_ratio_high")
    if overlap_to_root > 0.18:
        flags.append("root_shoot_overlap_high")
    return {
        "root_pixels_raw": int(root_pixels),
        "shoot_pixels_raw": int(shoot_pixels),
        "seed_pixels_raw": int(seed_pixels),
        "root_shoot_overlap_pixels": int(overlap_pixels),
        "shoot_to_root_area_ratio": float(round(shoot_to_root, 6)),
        "root_shoot_overlap_fraction": float(round(overlap_to_root, 6)),
        "shoot_area_fraction": float(round(shoot_area_fraction, 6)),
        "mask_qc_flags": flags,
    }


def _clip_bbox_xywh(bbox: BBox, shape_hw: tuple[int, int]) -> BBox:
    x, y, w, h = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    h_img, w_img = shape_hw
    if h_img <= 0 or w_img <= 0:
        return (0, 0, 0, 0)
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(0, min(w, w_img - x))
    h = max(0, min(h, h_img - y))
    return (x, y, w, h)


def _expand_bbox_xywh(bbox: BBox, padding: int, shape_hw: tuple[int, int]) -> BBox:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return (0, 0, 0, 0)
    x0 = x - int(padding)
    y0 = y - int(padding)
    x1 = x + w + int(padding)
    y1 = y + h + int(padding)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(shape_hw[1], x1)
    y1 = min(shape_hw[0], y1)
    return _clip_bbox_xywh((x0, y0, x1 - x0, y1 - y0), shape_hw)


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
    inter_x0 = max(ax, bx)
    inter_y0 = max(ay, by)
    inter_x1 = min(ax + aw, bx + bw)
    inter_y1 = min(ay + ah, by + bh)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter_area = float(inter_w * inter_h)
    union_area = float((aw * ah) + (bw * bh) - inter_area)
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _bbox_center_distance(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx = float(ax + (aw * 0.5))
    acy = float(ay + (ah * 0.5))
    bcx = float(bx + (bw * 0.5))
    bcy = float(by + (bh * 0.5))
    return float(np.hypot(acx - bcx, acy - bcy))


def _extract_components(mask: np.ndarray, min_area: int) -> list[dict[str, object]]:
    bin_mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    components: list[dict[str, object]] = []
    for label_id in range(1, int(num_labels)):
        x, y, w, h, area = stats[label_id].tolist()
        if int(area) < int(min_area):
            continue
        cx, cy = centroids[label_id]
        components.append(
            {
                "label_id": int(label_id),
                "bbox": (int(x), int(y), int(w), int(h)),
                "area": int(area),
                "centroid": (float(cx), float(cy)),
            }
        )
    components.sort(key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))
    return components


def _track_bboxes_from_root_masks(root_masks: list[np.ndarray], config: PyPhenotyperConfig) -> dict[str, object]:
    frame_count = len(root_masks)
    if frame_count == 0:
        return {"track_ids": [], "track_bboxes": {}, "overlap_frames": {}, "seed_frame_index": -1}

    shape_hw = root_masks[0].shape[:2]
    frame_components = [_extract_components(mask, config.min_component_area) for mask in root_masks]

    seed_frame_index = -1
    for idx, components in enumerate(frame_components):
        if components:
            seed_frame_index = idx
            break
    if seed_frame_index < 0:
        return {"track_ids": [], "track_bboxes": {}, "overlap_frames": {}, "seed_frame_index": -1}

    seed_components = sorted(frame_components[seed_frame_index], key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    track_ids = [f"plant_{index:02d}" for index in range(1, len(seed_components) + 1)]
    track_bboxes: dict[str, list[BBox]] = {
        track_id: [(0, 0, 0, 0) for _ in range(frame_count)] for track_id in track_ids
    }
    overlap_frames: dict[str, int | None] = {track_id: None for track_id in track_ids}

    for track_id, comp in zip(track_ids, seed_components):
        seed_bbox = _expand_bbox_xywh(comp["bbox"], config.bbox_padding, shape_hw)
        track_bboxes[track_id][seed_frame_index] = seed_bbox

    for frame_index in range(seed_frame_index + 1, frame_count):
        components = frame_components[frame_index]
        component_bboxes = [_expand_bbox_xywh(comp["bbox"], config.bbox_padding, shape_hw) for comp in components]
        assignments: dict[str, int] = {}
        candidate_scores: list[tuple[float, str, int]] = []

        for track_id in track_ids:
            previous_bbox = track_bboxes[track_id][frame_index - 1]
            if previous_bbox[2] <= 0 or previous_bbox[3] <= 0:
                continue
            search_bbox = _expand_bbox_xywh(previous_bbox, config.tracking_search_margin, shape_hw)
            for comp_index, comp_bbox in enumerate(component_bboxes):
                if not _bbox_intersects(search_bbox, comp_bbox):
                    continue
                iou = _bbox_iou(previous_bbox, comp_bbox)
                distance = _bbox_center_distance(previous_bbox, comp_bbox)
                span = float(max(1, previous_bbox[2], previous_bbox[3], config.tracking_search_margin))
                score = (2.5 * iou) + max(0.0, 1.0 - (distance / (2.0 * span)))
                candidate_scores.append((score, track_id, comp_index))

        assigned_tracks: set[str] = set()
        assigned_components: set[int] = set()
        for score, track_id, comp_index in sorted(candidate_scores, key=lambda item: item[0], reverse=True):
            if score <= 0:
                continue
            if track_id in assigned_tracks or comp_index in assigned_components:
                continue
            assignments[track_id] = comp_index
            assigned_tracks.add(track_id)
            assigned_components.add(comp_index)

        for track_id in track_ids:
            if track_id in assignments:
                continue
            previous_bbox = track_bboxes[track_id][frame_index - 1]
            if previous_bbox[2] <= 0 or previous_bbox[3] <= 0:
                continue
            best_index = -1
            best_distance = float("inf")
            for comp_index, comp_bbox in enumerate(component_bboxes):
                if comp_index in assigned_components:
                    continue
                distance = _bbox_center_distance(previous_bbox, comp_bbox)
                if distance < best_distance:
                    best_distance = distance
                    best_index = comp_index
            if best_index >= 0:
                max_jump = float(max(previous_bbox[2], previous_bbox[3], config.tracking_search_margin) * 2.5)
                if best_distance <= max_jump:
                    assignments[track_id] = best_index
                    assigned_components.add(best_index)

        for track_id in track_ids:
            if track_id in assignments:
                track_bboxes[track_id][frame_index] = component_bboxes[assignments[track_id]]
            else:
                track_bboxes[track_id][frame_index] = track_bboxes[track_id][frame_index - 1]

        for left_index, left_track_id in enumerate(track_ids):
            left_bbox = track_bboxes[left_track_id][frame_index]
            if left_bbox[2] <= 0 or left_bbox[3] <= 0:
                continue
            for right_track_id in track_ids[left_index + 1 :]:
                right_bbox = track_bboxes[right_track_id][frame_index]
                if right_bbox[2] <= 0 or right_bbox[3] <= 0:
                    continue
                if not _bbox_intersects(left_bbox, right_bbox):
                    continue
                if overlap_frames[left_track_id] is None:
                    overlap_frames[left_track_id] = int(frame_index)
                if overlap_frames[right_track_id] is None:
                    overlap_frames[right_track_id] = int(frame_index)
                track_bboxes[left_track_id][frame_index] = track_bboxes[left_track_id][frame_index - 1]
                track_bboxes[right_track_id][frame_index] = track_bboxes[right_track_id][frame_index - 1]

    return {
        "track_ids": track_ids,
        "track_bboxes": track_bboxes,
        "overlap_frames": overlap_frames,
        "seed_frame_index": int(seed_frame_index),
    }


def _skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    mask_bin = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if np.count_nonzero(mask_bin) == 0:
        return mask_bin.astype(bool)

    skeletonize = _load_skimage_skeletonize()
    if skeletonize is not None:
        return skeletonize(mask_bin > 0)

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


def _mask_perimeter_px(mask: np.ndarray) -> float:
    bin_mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(sum(float(cv2.arcLength(contour, True)) for contour in contours))


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


def _build_bbox_measurements(
    items: list[DatasetImageItem],
    predictions: dict[str, np.ndarray],
    config: PyPhenotyperConfig,
    metadata: dict[str, dict[str, object]] | None = None,
    progress_callback: Callable[[int, int], bool] | None = None,
) -> dict[str, object]:
    overrides = config.pipeline_overrides if isinstance(config.pipeline_overrides, dict) else {}
    ownership_backend = str(overrides.get("ownership_backend", "") or "").strip().lower()
    if ownership_backend in {"bw_arabidopsis", "bw_arabidopsis_seed_centers", "seed_center_bboxes"}:
        return build_bw_arabidopsis_measurements(
            items,
            predictions,
            config,
            metadata=metadata,
            progress_callback=progress_callback,
        )

    timeline_items: list[DatasetImageItem] = [item for item in items if item.uid in predictions]
    if not timeline_items:
        return {"summary": {}, "per_uid": {}}

    root_masks: list[np.ndarray] = [
        (np.asarray(predictions[item.uid], dtype=np.uint8) == np.uint8(config.root_class_id)).astype(np.uint8)
        for item in timeline_items
    ]
    tracking = _track_bboxes_from_root_masks(root_masks, config)
    track_ids: list[str] = list(tracking.get("track_ids", []))
    track_bboxes: dict[str, list[BBox]] = dict(tracking.get("track_bboxes", {}))
    overlap_frames: dict[str, int | None] = dict(tracking.get("overlap_frames", {}))

    per_uid: dict[str, dict] = {}
    rows: list[dict[str, object]] = []
    dataset_pixel_sizes: dict[str, float] = {}

    measurement_total = len(timeline_items)
    for frame_index, item in enumerate(timeline_items):
        root_mask = root_masks[frame_index]
        h_img, w_img = root_mask.shape[:2]
        frame_bboxes: dict[str, list[int]] = {}
        frame_lengths_mm: dict[str, float] = {}
        frame_lengths_px: dict[str, float] = {}
        item_meta = metadata.get(item.uid, {}) if isinstance(metadata, dict) else {}
        pixel_size_mm = _coerce_positive_float(item_meta.get("pixel_size_mm")) or float(config.pixel_size_mm)
        dataset_pixel_sizes[item.uid] = float(pixel_size_mm)

        for track_id in track_ids:
            bbox_series = track_bboxes.get(track_id, [])
            bbox = bbox_series[frame_index] if frame_index < len(bbox_series) else (0, 0, 0, 0)
            x, y, w, h = _clip_bbox_xywh(bbox, (h_img, w_img))

            if w <= 0 or h <= 0:
                length_px = 0.0
                length_mm = 0.0
                area_px = 0
                perimeter_px = 0.0
            else:
                crop = root_mask[y : y + h, x : x + w]
                length_px, area_px, perimeter_px = _measure_mask_length_px(crop)
                length_mm = float(length_px * pixel_size_mm)

            frame_bboxes[track_id] = [int(x), int(y), int(w), int(h)]
            frame_lengths_mm[track_id] = float(round(length_mm, 4))
            rows.append(
                {
                    "uid": item.uid,
                    "image_name": item.name,
                    "frame_index": int(frame_index),
                    "plant_id": track_id,
                    "pixel_size_mm": float(round(pixel_size_mm, 8)),
                    "bbox_x": int(x),
                    "bbox_y": int(y),
                    "bbox_w": int(w),
                    "bbox_h": int(h),
                    "root_area_px": int(area_px),
                    "root_area_mm2": float(round(float(area_px) * (pixel_size_mm ** 2), 4)),
                    "root_perimeter_px": float(round(perimeter_px, 4)),
                    "root_perimeter_mm": float(round(perimeter_px * pixel_size_mm, 4)),
                    "root_length_px": float(round(length_px, 4)),
                    "root_length_mm": float(round(length_mm, 4)),
                    "overlap_frame": overlap_frames.get(track_id),
                }
            )
            frame_lengths_px[track_id] = float(round(length_px, 4))

        per_uid[item.uid] = {
            "enabled": True,
            "frame_index": int(frame_index),
            "plant_count": int(len(track_ids)),
            "seed_frame_index": int(tracking.get("seed_frame_index", -1)),
            "bboxes_xywh": frame_bboxes,
            "lengths_px": frame_lengths_px,
            "lengths_mm": frame_lengths_mm,
            "pixel_size_mm": float(pixel_size_mm),
            "overlap_frames": {k: overlap_frames.get(k) for k in track_ids},
        }
        if progress_callback is not None and not bool(progress_callback(frame_index + 1, measurement_total)):
            break

    summary = {
        "backend": "pyphenotyper",
        "bbox_tracking_enabled": True,
        "seed_frame_index": int(tracking.get("seed_frame_index", -1)),
        "pixel_size_mm": float(config.pixel_size_mm),
        "pixel_size_mm_by_uid": {uid: float(value) for uid, value in dataset_pixel_sizes.items()},
        "plant_ids": track_ids,
        "timeline": [{"uid": item.uid, "name": item.name} for item in timeline_items],
        "measurements": rows,
        "overlap_frames": {k: overlap_frames.get(k) for k in track_ids},
    }
    return {"summary": summary, "per_uid": per_uid}


def _adapt_tracking_to_ui_metadata(
    metadata: dict[str, dict],
    items: list[DatasetImageItem],
    tracking_payload: dict[str, object],
) -> None:
    per_uid = tracking_payload.get("per_uid")
    if isinstance(per_uid, dict):
        for item in items:
            frame_payload = per_uid.get(item.uid)
            if frame_payload is None:
                continue
            if item.uid not in metadata:
                metadata[item.uid] = {}
            metadata[item.uid]["bbox_tracking"] = frame_payload

    summary = tracking_payload.get("summary")
    if isinstance(summary, dict) and summary:
        metadata["__dataset__"] = summary


def masks_to_index_prediction(
    root_mask: np.ndarray,
    shoot_mask: np.ndarray,
    seed_mask: np.ndarray,
    occlusion_mask: np.ndarray,
    shape_hw: tuple[int, int],
    root_class_id: int,
    shoot_class_id: int,
    seed_class_id: int | None = None,
    include_occlusion: bool = True,
    shoot_priority_over_root: bool = True,
    lateral_root_mask: np.ndarray | None = None,
    lateral_class_id: int | None = None,
) -> np.ndarray:
    root_mask = _ensure_mask_shape(root_mask, shape_hw)
    shoot_mask = _ensure_mask_shape(shoot_mask, shape_hw)
    seed_mask = _ensure_mask_shape(seed_mask, shape_hw)
    occlusion_mask = _ensure_mask_shape(occlusion_mask, shape_hw)
    lateral_mask = _ensure_mask_shape(
        lateral_root_mask if lateral_root_mask is not None else np.zeros(shape_hw, dtype=np.uint8),
        shape_hw,
    )

    root_bin = root_mask > 0
    if include_occlusion:
        root_bin = np.logical_or(root_bin, occlusion_mask > 0)
    shoot_bin = shoot_mask > 0
    seed_bin = seed_mask > 0
    lateral_bin = np.logical_and(lateral_mask > 0, root_mask > 0)

    index = np.zeros(shape_hw, dtype=np.uint8)
    shoot_class_id = max(0, min(255, int(shoot_class_id)))
    root_class_id = max(0, min(255, int(root_class_id)))
    seed_class_id = max(0, min(255, int(seed_class_id or 0)))
    lateral_class_id = max(0, min(255, int(lateral_class_id or 0)))

    if root_class_id > 0:
        index[root_bin] = np.uint8(root_class_id)
    if lateral_class_id > 0 and lateral_class_id != root_class_id:
        index[lateral_bin] = np.uint8(lateral_class_id)
    if shoot_class_id > 0:
        if shoot_priority_over_root:
            # Keep shoot priority over root at the crown / hypocotyl junction.
            # The pyphenotyper root mask can legitimately bleed into the shoot
            # region, and allowing root to overwrite shoot causes visually
            # "missing" shoots in the UI.
            index[shoot_bin] = np.uint8(shoot_class_id)
        else:
            # Some legacy dark-background experts produce a tighter root mask and
            # a looser shoot mask. Preserve root ownership and let shoot fill
            # only the remaining pixels.
            index[np.logical_and(shoot_bin, ~root_bin)] = np.uint8(shoot_class_id)
    if seed_class_id > 0:
        index[seed_bin] = np.uint8(seed_class_id)
    return index


def validate_pyphenotyper_setup(config: PyPhenotyperConfig) -> None:
    def _validate_variant_paths(pipeline_dir: Path, root_model_path: Path, shoot_model_path: Path, label: str) -> None:
        _, package_dir = _normalize_pipeline_dir(pipeline_dir)
        if not (package_dir / "features" / "features.py").exists():
            raise FileNotFoundError(f"Missing features module for {label} at: {package_dir / 'features' / 'features.py'}")
        if not root_model_path.exists():
            raise FileNotFoundError(f"Root model not found for {label}: {root_model_path}")
        if not shoot_model_path.exists():
            raise FileNotFoundError(f"Shoot model not found for {label}: {shoot_model_path}")

    _validate_variant_paths(config.pipeline_dir, config.root_model_path, config.shoot_model_path, "default")
    if config.seed_model_path is not None and not Path(config.seed_model_path).exists():
        raise FileNotFoundError(f"Seed model not found for default: {config.seed_model_path}")
    if config.lucifer_gan_gap_repair_enabled:
        if config.lucifer_gan_gap_repair_dir is None:
            raise FileNotFoundError("Lucifer GAN gap repair is enabled, but no GAN folder was selected.")
        gap_dir = Path(config.lucifer_gan_gap_repair_dir).expanduser()
        if not gap_dir.exists():
            raise FileNotFoundError(f"Lucifer GAN gap repair folder not found: {gap_dir}")
    if config.grayscale_variant is not None:
        _validate_variant_paths(
            Path(config.grayscale_variant.pipeline_dir).expanduser(),
            Path(config.grayscale_variant.root_model_path).expanduser(),
            Path(config.grayscale_variant.shoot_model_path).expanduser(),
            str(config.grayscale_variant.label or "grayscale"),
        )
    if config.rgb_variant is not None:
        _validate_variant_paths(
            Path(config.rgb_variant.pipeline_dir).expanduser(),
            Path(config.rgb_variant.root_model_path).expanduser(),
            Path(config.rgb_variant.shoot_model_path).expanduser(),
            str(config.rgb_variant.label or "rgb"),
        )
    for expert in config.expert_variants:
        _validate_variant_paths(
            Path(expert.pipeline_dir).expanduser(),
            Path(expert.root_model_path).expanduser(),
            Path(expert.shoot_model_path).expanduser(),
            str(expert.label or expert.key or "expert"),
        )
    if config.patch_size <= 0:
        raise ValueError("Patch size must be > 0.")
    if config.refinement_steps < 0:
        raise ValueError("Refinement steps must be >= 0.")
    if config.min_component_area <= 0:
        raise ValueError("Minimum component area must be > 0.")
    if config.bbox_padding < 0:
        raise ValueError("Bounding-box padding must be >= 0.")
    if config.tracking_search_margin < 0:
        raise ValueError("Tracking search margin must be >= 0.")
    if config.pixel_size_mm <= 0:
        raise ValueError("Pixel size (mm/px) must be > 0.")
    if config.router_uncertainty_margin < 0.0:
        raise ValueError("Router uncertainty margin must be >= 0.")


def _call_model_create_masks_compat(
    features_module,
    image_path: Path,
    patch_size: int,
    root_model,
    shoot_model,
    refinement_steps: int,
    profile_overrides: dict[str, object] | None = None,
    pipeline_overrides: dict[str, object] | None = None,
):
    """Call pyphenotyper.features.model_create_masks across signature variants.

    Some pipeline versions require additional positional/keyword args such as
    `traditional_cv_shoot`. This helper introspects the callable and applies
    conservative defaults so packaged runtime stays compatible.
    """
    if not hasattr(features_module, "model_create_masks"):
        raise AttributeError("PyPhenotyper features module does not expose `model_create_masks`.")
    fn = features_module.model_create_masks

    attempts: list[tuple[dict[str, object], str]] = []
    base_kwargs: dict[str, object] = {"refinement_steps": int(refinement_steps), "verbose": False}
    attempts.append((base_kwargs, "base"))

    try:
        sig = inspect.signature(fn)
    except Exception:
        sig = None

    if sig is not None:
        params = sig.parameters
        accepts_varkw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
        compat_kwargs: dict[str, object] = {}
        if "refinement_steps" in params:
            compat_kwargs["refinement_steps"] = int(refinement_steps)
        if "verbose" in params:
            compat_kwargs["verbose"] = False
        if profile_overrides is not None and ("profile_overrides" in params or accepts_varkw):
            compat_kwargs["profile_overrides"] = dict(profile_overrides)
        if pipeline_overrides is not None and ("pipeline_overrides" in params or accepts_varkw):
            compat_kwargs["pipeline_overrides"] = dict(pipeline_overrides)
        # Known legacy/variant parameters encountered in external pipelines.
        for name in (
            "traditional_cv_shoot",
            "traditional_cv_root",
            "traditional_cv",
            "use_traditional_cv_shoot",
            "use_traditional_cv_root",
            "cv_shoot",
            "cv_root",
        ):
            if name in params:
                compat_kwargs[name] = False
        for name, param in params.items():
            if name in compat_kwargs:
                continue
            if name in {
                "image_path",
                "patch_size",
                "segmentation_model",
                "root_model",
                "shoot_model",
                "refinement_steps",
                "verbose",
            }:
                continue
            if param.default is not inspect._empty and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            if param.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            # Required-but-unknown args: keep conservative defaults to avoid runtime break.
            name_lower = str(name).lower()
            if "cv" in name_lower or "traditional" in name_lower:
                compat_kwargs[name] = False
            else:
                compat_kwargs[name] = None
        attempts.insert(0, (compat_kwargs, "signature"))

    errors: list[str] = []
    for kwargs, label in attempts:
        try:
            return fn(str(image_path), int(patch_size), root_model, shoot_model, **kwargs)
        except TypeError as exc:
            errors.append(f"{label}: {exc}")
            continue
    # Positional fallback for strict legacy signatures:
    # model_create_masks(image, patch, root_model, shoot_model, traditional_cv_shoot, refinement_steps, verbose)
    try:
        return fn(str(image_path), int(patch_size), root_model, shoot_model, False, int(refinement_steps), False)
    except Exception as exc:
        details = "; ".join(errors) if errors else "no captured TypeError details"
        raise RuntimeError(
            "PyPhenotyper model_create_masks signature mismatch. "
            f"Tried compatibility adapters but failed. Details: {details}. Final error: {exc}"
        ) from exc


def run_pyphenotyper_item(item: DatasetImageItem, config: PyPhenotyperConfig) -> tuple[np.ndarray, dict]:
    validate_pyphenotyper_setup(config)
    if config.expert_variants:
        image_features, routed_candidates = _candidate_experts_for_item(item, config)
        evaluated: list[dict[str, object]] = []
        for row in routed_candidates:
            expert = row["expert"]
            effective_config = _config_from_expert(config, expert)
            pred, details, mask_payload = _run_pyphenotyper_item_single(
                item,
                config=config,
                effective_config=effective_config,
                variant_label=str(expert.label or expert.key),
                routing_family=str(expert.family or expert.key or "expert"),
            )
            confidence_payload = mask_confidence(
                root_mask=np.asarray(mask_payload["root_mask"], dtype=np.uint8),
                shoot_mask=np.asarray(mask_payload["shoot_mask"], dtype=np.uint8),
                image_shape_hw=item.image.shape[:2],
                family=str(expert.family or expert.key or "expert"),
                hints=expert.router_hints,
            )
            total_score = float(row["prior_score"]) + float(confidence_payload["score"])
            evaluated.append(
                {
                    "expert": expert,
                    "pred": pred,
                    "details": details,
                    "prior_score": float(row["prior_score"]),
                    "confidence_score": float(confidence_payload["score"]),
                    "confidence_payload": confidence_payload,
                    "total_score": float(total_score),
                }
            )
        evaluated.sort(key=lambda row: float(row["total_score"]), reverse=True)
        best = evaluated[0]
        second = evaluated[1] if len(evaluated) > 1 else None
        margin = float(best["total_score"]) - float(second["total_score"]) if second is not None else 1.0
        uncertain = bool(second is not None and margin < float(config.router_uncertainty_margin))
        alternatives: list[dict[str, object]] = []
        for row in evaluated[1:4]:
            expert = row["expert"]
            alternatives.append(
                {
                    "key": str(expert.key),
                    "label": str(expert.label or expert.key),
                    "family": str(expert.family or expert.key),
                    "score": float(row["total_score"]),
                }
            )
        best_details = dict(best["details"])
        best_details.update(
            {
                "routing_mode": str(config.router_mode or "expert_router"),
                "routing_expert_key": str(best["expert"].key),
                "routing_confidence": float(best["total_score"]),
                "routing_prior_score": float(best["prior_score"]),
                "routing_mask_score": float(best["confidence_score"]),
                "routing_uncertain": uncertain,
                "routing_margin": float(margin),
                "routing_features": features_as_dict(image_features),
                "routing_alternatives": alternatives,
                "routing_selected_by": "image_statistics_plus_mask_confidence",
                "routing_calibrated": bool(config.calibration_profile),
                "routing_calibration_preference": (
                    str(config.calibration_profile.get("preferred_expert_key") or "")
                    if isinstance(config.calibration_profile, dict)
                    else ""
                ),
            }
        )
        return np.asarray(best["pred"], dtype=np.uint8), best_details

    effective_config, variant_label, routing_family = _resolve_item_config(item, config)
    pred, details, _mask_payload = _run_pyphenotyper_item_single(
        item,
        config=config,
        effective_config=effective_config,
        variant_label=variant_label,
        routing_family=routing_family,
    )
    return pred, details


def _dataset_route_lock_key(item: DatasetImageItem) -> str:
    path = getattr(item, "path", None)
    if isinstance(path, Path):
        try:
            return str(path.expanduser().resolve().parent)
        except Exception:
            return str(path.parent)
    return "__dataset__"


def _run_pyphenotyper_item_with_route_lock(
    item: DatasetImageItem,
    config: PyPhenotyperConfig,
    route_lock: dict[str, str],
) -> tuple[np.ndarray, dict[str, object]]:
    kind = str(route_lock.get("kind", "") or "")
    value = str(route_lock.get("value", "") or "")
    if kind == "expert":
        expert = next(
            (candidate for candidate in config.expert_variants if str(candidate.key) == value),
            None,
        )
        if expert is not None:
            effective_config = _config_from_expert(config, expert)
            pred, details, _payload = _run_pyphenotyper_item_single(
                item,
                config=config,
                effective_config=effective_config,
                variant_label=str(expert.label or expert.key),
                routing_family=str(expert.family or expert.key or "expert"),
            )
            details = dict(details)
            details["routing_expert_key"] = str(expert.key)
            return np.asarray(pred, dtype=np.uint8), details
    if kind == "variant":
        family = "rgb" if value == "rgb" else "grayscale"
        variant = config.rgb_variant if family == "rgb" else config.grayscale_variant
        if variant is not None:
            effective_config = _config_from_variant(config, variant)
            pred, details, _payload = _run_pyphenotyper_item_single(
                item,
                config=config,
                effective_config=effective_config,
                variant_label=str(variant.label or family),
                routing_family=family,
            )
            return np.asarray(pred, dtype=np.uint8), dict(details)
    pred, details = run_pyphenotyper_item(item, config)
    return np.asarray(pred, dtype=np.uint8), dict(details)


def run_pyphenotyper_segmentation_pipeline(
    items: list[DatasetImageItem],
    config: PyPhenotyperConfig,
    progress_callback: Callable[[int, int], bool] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    predictions: dict[str, np.ndarray] = {}
    metadata: dict[str, dict] = {}
    total = len(items)
    if total == 0:
        return predictions, metadata

    tracking_steps = len(items) if config.enable_bbox_tracking else 0
    total_steps = total + tracking_steps
    route_locks: dict[str, dict[str, str]] = {}

    for index, item in enumerate(items, start=1):
        route_key = _dataset_route_lock_key(item)
        route_lock = route_locks.get(route_key)
        if route_lock is not None:
            pred, details = _run_pyphenotyper_item_with_route_lock(item, config, route_lock)
            details["routing_locked"] = True
            details["routing_selected_by"] = "series_route_lock"
        else:
            pred, details = run_pyphenotyper_item(item, config)
            selected_lock: dict[str, str] | None = None
            expert_key = str(details.get("routing_expert_key", "") or "").strip()
            if config.expert_variants and expert_key:
                selected_lock = {"kind": "expert", "value": expert_key}
            elif config.rgb_variant is not None or config.grayscale_variant is not None:
                family = str(details.get("routing_family", "") or "").strip().lower()
                if family in {"rgb", "grayscale"}:
                    selected_lock = {"kind": "variant", "value": family}
            if selected_lock is not None:
                route_locks[route_key] = selected_lock
                details["routing_locked"] = True
                details["routing_selected_by"] = "series_first_frame_lock"
            else:
                details["routing_locked"] = False
        details["routing_series_key"] = route_key
        predictions[item.uid] = pred
        metadata[item.uid] = details
        if progress_callback is not None and not bool(progress_callback(index, total_steps)):
            break

    processed_items = items[: len(predictions)]
    if config.enable_bbox_tracking and processed_items:
        def _tracking_progress(frame_index: int, measurement_total: int) -> bool:
            if progress_callback is None:
                return True
            return bool(
                progress_callback(
                    len(processed_items) + int(frame_index),
                    len(processed_items) + int(measurement_total),
                )
            )

        tracking_payload = _build_bbox_measurements(
            processed_items,
            predictions,
            config,
            metadata=metadata,
            progress_callback=_tracking_progress if progress_callback is not None else None,
        )
        _adapt_tracking_to_ui_metadata(metadata, processed_items, tracking_payload)

    return predictions, metadata


def run_pyphenotyper_dataset(
    items: list[DatasetImageItem],
    config: PyPhenotyperConfig,
    progress_callback: Callable[[int, int], bool] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    return run_pyphenotyper_segmentation_pipeline(items, config, progress_callback=progress_callback)


def calibrate_pyphenotyper_experts(
    items: list[DatasetImageItem],
    annotations: dict[str, dict[int, np.ndarray]],
    classes: list,
    config: PyPhenotyperConfig,
    progress_callback: Callable[[int, int, str], bool] | None = None,
) -> dict[str, object]:
    validate_pyphenotyper_setup(config)
    if not config.expert_variants:
        raise ValueError("General starter calibration requires expert variants.")

    selected_items: list[DatasetImageItem] = []
    normalized_targets: dict[str, dict[str, np.ndarray]] = {}
    for item in items:
        targets = normalize_annotation_targets(annotations.get(item.uid), classes, item.image.shape[:2])
        if np.count_nonzero(targets["root_binary"]) <= 0 and np.count_nonzero(targets["shoot"]) <= 0:
            continue
        selected_items.append(item)
        normalized_targets[item.uid] = targets
    if not selected_items:
        raise ValueError("No labeled images matched the normalized general-root schema.")

    total_steps = max(1, len(selected_items) * len(config.expert_variants))
    step = 0
    expert_rows: list[dict[str, object]] = []
    for expert in config.expert_variants:
        effective_config = _config_from_expert(config, expert)
        case_rows: list[dict[str, object]] = []
        combined_scores: list[float] = []
        root_scores: list[float] = []
        shoot_scores: list[float] = []
        for item in selected_items:
            step += 1
            if progress_callback is not None and not bool(
                progress_callback(step, total_steps, f"Scoring {expert.label or expert.key} on {item.name}")
            ):
                raise RuntimeError("General starter calibration cancelled.")
            pred, _details, _mask_payload = _run_pyphenotyper_item_single(
                item,
                config=config,
                effective_config=effective_config,
                variant_label=str(expert.label or expert.key),
                routing_family=str(expert.family or expert.key or "expert"),
            )
            targets = normalized_targets[item.uid]
            root_iou = iou_score(pred == np.uint8(config.root_class_id), targets["root_binary"])
            shoot_iou = iou_score(pred == np.uint8(config.shoot_class_id), targets["shoot"])
            parts = [score for score in (root_iou, shoot_iou) if score is not None]
            combined = float(np.mean(parts)) if parts else 0.0
            if root_iou is not None:
                root_scores.append(float(root_iou))
            if shoot_iou is not None:
                shoot_scores.append(float(shoot_iou))
            combined_scores.append(float(combined))
            case_rows.append(
                {
                    "uid": item.uid,
                    "name": item.name,
                    "root_iou": float(root_iou) if root_iou is not None else None,
                    "shoot_iou": float(shoot_iou) if shoot_iou is not None else None,
                    "combined_iou": float(combined),
                }
            )
        expert_rows.append(
            {
                "expert_key": str(expert.key),
                "label": str(expert.label or expert.key),
                "family": str(expert.family or expert.key),
                "image_mode": str(expert.image_mode or "any"),
                "combined_mean_iou": float(np.mean(combined_scores)) if combined_scores else 0.0,
                "root_mean_iou": float(np.mean(root_scores)) if root_scores else None,
                "shoot_mean_iou": float(np.mean(shoot_scores)) if shoot_scores else None,
                "cases": case_rows,
            }
        )
    profile = build_calibration_profile(expert_rows, selected_uids=[item.uid for item in selected_items])
    return {
        "selected_uids": [item.uid for item in selected_items],
        "image_count": len(selected_items),
        "expert_scores": expert_rows,
        "profile": profile,
    }
