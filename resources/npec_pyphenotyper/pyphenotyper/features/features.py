from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pyphenotyper.data.data_processing import (
    adaptive_side_margin,
    collect_tiles,
    estimate_dish_interior_mask,
    filter_connected_components,
    load_uint8_image,
    remove_noise,
    stitch_predictions,
)
from pyphenotyper.logger_config import logger


RETURNS_ORIGINAL_COORDS = True


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    mode: str
    input_mode: str
    flip_horizontal: bool
    return_original_coords: bool
    include_seed_in_shoot: bool
    root_threshold: float
    shoot_threshold: float
    root_label_ids: tuple[int, ...]
    lateral_label_ids: tuple[int, ...]
    shoot_label_ids: tuple[int, ...]
    seed_label_ids: tuple[int, ...]
    batch_size: int
    tile_halo: int
    class_probability_scales: tuple[float, ...]


DEFAULT_BINARY_PROFILE = ModelProfile(
    name="legacy_binary_pair",
    mode="binary_pair",
    input_mode="gray",
    flip_horizontal=True,
    return_original_coords=True,
    include_seed_in_shoot=False,
    root_threshold=0.5,
    shoot_threshold=0.5,
    root_label_ids=(3, 4),
    lateral_label_ids=(4,),
    shoot_label_ids=(2,),
    seed_label_ids=(1,),
    batch_size=12,
    tile_halo=0,
    class_probability_scales=(),
)

DEFAULT_MULTICLASS_PROFILE = ModelProfile(
    name="multiclass_semantic",
    mode="multiclass",
    input_mode="rgb",
    flip_horizontal=True,
    return_original_coords=True,
    include_seed_in_shoot=False,
    root_threshold=0.5,
    shoot_threshold=0.5,
    root_label_ids=(3, 4),
    lateral_label_ids=(4,),
    shoot_label_ids=(2,),
    seed_label_ids=(1,),
    batch_size=8,
    tile_halo=0,
    class_probability_scales=(),
)


@dataclass(frozen=True, slots=True)
class PipelineTuning:
    root_min_area: int
    shoot_min_area: int
    shoot_postprocess_mode: str
    root_edge_margin_fraction: float
    root_edge_margin_max_px: int
    shoot_edge_margin_fraction: float
    shoot_edge_margin_max_px: int
    shoot_lower_start_fraction: float
    shoot_guard_top_limit_fraction: float
    shoot_guard_min_top_limit_px: int
    shoot_guard_component_min_area: int
    shoot_guard_component_area_fraction: float
    shoot_guard_component_max_y_fraction: float
    shoot_guard_component_min_y_px: int
    shoot_guard_pad_x_fraction: float
    shoot_guard_pad_x_min_px: int
    shoot_guard_pad_x_max_px: int
    shoot_guard_pad_up_fraction: float
    shoot_guard_pad_up_min_px: int
    shoot_guard_pad_up_max_px: int
    shoot_guard_pad_down_fraction: float
    shoot_guard_pad_down_min_px: int
    shoot_guard_pad_down_max_px: int
    shoot_guard_slot_half_width_fraction: float
    shoot_guard_slot_half_width_min_px: int
    shoot_guard_slot_half_width_max_px: int
    shoot_guard_slot_count: int
    shoot_guard_edge_margin_fraction: float
    shoot_guard_edge_margin_max_px: int
    shoot_guard_accept_ratio: float
    root_prediction_offsets: tuple[int, ...]
    shoot_prediction_refinement_steps: int


DEFAULT_PIPELINE_TUNING = PipelineTuning(
    root_min_area=24,
    shoot_min_area=32,
    shoot_postprocess_mode="guarded",
    root_edge_margin_fraction=0.08,
    root_edge_margin_max_px=320,
    shoot_edge_margin_fraction=0.08,
    shoot_edge_margin_max_px=320,
    shoot_lower_start_fraction=0.72,
    shoot_guard_top_limit_fraction=0.28,
    shoot_guard_min_top_limit_px=120,
    shoot_guard_component_min_area=100,
    shoot_guard_component_area_fraction=0.00002,
    shoot_guard_component_max_y_fraction=0.45,
    shoot_guard_component_min_y_px=180,
    shoot_guard_pad_x_fraction=0.085,
    shoot_guard_pad_x_min_px=140,
    shoot_guard_pad_x_max_px=520,
    shoot_guard_pad_up_fraction=0.06,
    shoot_guard_pad_up_min_px=90,
    shoot_guard_pad_up_max_px=240,
    shoot_guard_pad_down_fraction=0.12,
    shoot_guard_pad_down_min_px=120,
    shoot_guard_pad_down_max_px=380,
    shoot_guard_slot_half_width_fraction=0.06,
    shoot_guard_slot_half_width_min_px=110,
    shoot_guard_slot_half_width_max_px=420,
    shoot_guard_slot_count=5,
    shoot_guard_edge_margin_fraction=0.10,
    shoot_guard_edge_margin_max_px=420,
    shoot_guard_accept_ratio=0.08,
    root_prediction_offsets=(),
    shoot_prediction_refinement_steps=0,
)

MODEL_PROFILE_OVERRIDES: dict[str, dict[str, object]] = {
    "npec_trained_model-hades-lucifer_workstation.keras": {
        "name": "hades_lucifer_multiclass",
        "mode": "multiclass",
        "input_mode": "rgb",
        "flip_horizontal": True,
    },
}


def _coerce_int_tuple(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        values = [int(value) for value in raw]
    else:
        values = [int(raw)]
    return tuple(max(0, int(value)) for value in values)


def _coerce_positive_float_tuple(raw: Any) -> tuple[float, ...]:
    if raw is None:
        return ()
    values = raw if isinstance(raw, (list, tuple)) else (raw,)
    parsed = tuple(float(value) for value in values)
    if any(not np.isfinite(value) or value <= 0.0 for value in parsed):
        raise ValueError("Class probability scales must be finite positive values.")
    return parsed


def resolve_pipeline_tuning(pipeline_overrides: dict[str, object] | None = None) -> PipelineTuning:
    if not pipeline_overrides:
        return DEFAULT_PIPELINE_TUNING
    overrides = dict(pipeline_overrides)
    if "root_prediction_offsets" in overrides:
        overrides["root_prediction_offsets"] = _coerce_int_tuple(overrides.get("root_prediction_offsets"))
    if "shoot_postprocess_mode" in overrides:
        raw_mode = str(overrides.get("shoot_postprocess_mode") or "guarded").strip().lower()
        if raw_mode in {"legacy", "old", "legacy_mask", "legacy_shoot"}:
            overrides["shoot_postprocess_mode"] = "legacy"
        elif raw_mode in {"bw_arabidopsis", "bw_arabidopsis_solid", "solid_bw", "solid_shoot"}:
            overrides["shoot_postprocess_mode"] = "bw_arabidopsis"
        elif raw_mode in {"simple", "plain", "filter_only", "minimal"}:
            overrides["shoot_postprocess_mode"] = "simple"
        else:
            overrides["shoot_postprocess_mode"] = "guarded"
    return replace(DEFAULT_PIPELINE_TUNING, **overrides)


def describe_pipeline_tuning(pipeline_overrides: dict[str, object] | None = None) -> dict[str, object]:
    tuning = resolve_pipeline_tuning(pipeline_overrides)
    payload = asdict(tuning)
    payload["root_prediction_offsets"] = list(tuning.root_prediction_offsets)
    return payload


def _shape_last(shape: Any) -> int | None:
    if shape is None:
        return None
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    if not isinstance(shape, (list, tuple)) or not shape:
        return None
    try:
        value = shape[-1]
        return int(value) if value is not None else None
    except Exception:
        return None


def _model_source_name(model: Any) -> str:
    raw = str(getattr(model, "_npec_source_path", "") or "")
    return Path(raw).name.lower()


def infer_model_profile(segmentation_model: Any, profile_overrides: dict[str, object] | None = None) -> ModelProfile:
    input_channels = _shape_last(getattr(segmentation_model, "input_shape", None)) or 1
    output_channels = _shape_last(getattr(segmentation_model, "output_shape", None)) or 1
    source_name = _model_source_name(segmentation_model)
    base = DEFAULT_MULTICLASS_PROFILE if output_channels > 1 or input_channels > 1 else DEFAULT_BINARY_PROFILE
    overrides: dict[str, object] = {}
    if source_name in MODEL_PROFILE_OVERRIDES:
        overrides.update(MODEL_PROFILE_OVERRIDES[source_name])
    if profile_overrides:
        overrides.update(profile_overrides)
    if "class_probability_scales" in overrides:
        overrides["class_probability_scales"] = _coerce_positive_float_tuple(
            overrides.get("class_probability_scales")
        )
    return replace(base, **overrides) if overrides else base


def describe_model_profile(
    segmentation_model: Any,
    shoot_model: Any | None = None,
    profile_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    profile = infer_model_profile(segmentation_model, profile_overrides=profile_overrides)
    return {
        **asdict(profile),
        "segmentation_input_channels": _shape_last(getattr(segmentation_model, "input_shape", None)),
        "segmentation_output_channels": _shape_last(getattr(segmentation_model, "output_shape", None)),
        "shoot_input_channels": _shape_last(getattr(shoot_model, "input_shape", None)) if shoot_model is not None else None,
        "shoot_output_channels": _shape_last(getattr(shoot_model, "output_shape", None)) if shoot_model is not None else None,
        "segmentation_model": _model_source_name(segmentation_model),
        "shoot_model": _model_source_name(shoot_model) if shoot_model is not None else None,
    }


def _load_image_for_profile(image_path: str | Path, profile: ModelProfile) -> np.ndarray:
    image = load_uint8_image(image_path, profile.input_mode)
    if profile.flip_horizontal:
        image = np.fliplr(image)
    return image


def _load_original_rgb(image_path: str | Path) -> np.ndarray:
    return load_uint8_image(image_path, "rgb")


def _load_processing_rgb(image_path: str | Path, profile: ModelProfile) -> np.ndarray:
    image = _load_original_rgb(image_path)
    if profile.flip_horizontal and not bool(profile.return_original_coords):
        image = np.fliplr(image)
    return image


def _normalize_model_batch(tiles: np.ndarray, input_mode: str) -> np.ndarray:
    batch = np.asarray(tiles, dtype=np.float32)
    if batch.ndim == 3:
        batch = batch[..., np.newaxis]
    if input_mode == "gray" and batch.shape[-1] != 1:
        batch = batch[..., :1]
    return batch / 255.0


def _predict_tiled_output(
    model: Any,
    image: np.ndarray,
    patch_size: int,
    batch_size: int,
    input_mode: str,
    tile_halo: int = 0,
) -> np.ndarray:
    halo = max(0, int(tile_halo))
    if halo > 0:
        if 2 * halo >= int(patch_size):
            raise ValueError("Tile halo must be smaller than half the patch size.")
        output_size = int(patch_size) - (2 * halo)
        height, width = image.shape[:2]
        rows = int(np.ceil(height / output_size))
        columns = int(np.ceil(width / output_size))
        extra_bottom = rows * output_size - height
        extra_right = columns * output_size - width
        border_mode = cv2.BORDER_REFLECT_101 if min(height, width) > 1 else cv2.BORDER_REPLICATE
        padded = cv2.copyMakeBorder(
            image,
            halo,
            halo + extra_bottom,
            halo,
            halo + extra_right,
            border_mode,
        )
        tiles: list[np.ndarray] = []
        locations: list[tuple[int, int, int, int]] = []
        for row in range(rows):
            y = row * output_size
            for column in range(columns):
                x = column * output_size
                tiles.append(padded[y : y + patch_size, x : x + patch_size])
                locations.append((y, x, min(output_size, height - y), min(output_size, width - x)))
        stitched: np.ndarray | None = None
        stride = max(1, int(batch_size))
        for start in range(0, len(tiles), stride):
            stop = min(len(tiles), start + stride)
            batch = _normalize_model_batch(np.asarray(tiles[start:stop]), input_mode=input_mode)
            predicted = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
            if predicted.ndim == 3:
                predicted = predicted[..., np.newaxis]
            if stitched is None:
                stitched = np.zeros((height, width, predicted.shape[-1]), dtype=predicted.dtype)
            for probability, (y, x, block_height, block_width) in zip(
                predicted,
                locations[start:stop],
                strict=False,
            ):
                stitched[y : y + block_height, x : x + block_width] = probability[
                    halo : halo + block_height,
                    halo : halo + block_width,
                ]
        if stitched is None:
            raise RuntimeError("No tiles were generated for model prediction.")
        return stitched
    tiles, coords, layout = collect_tiles(image, patch_size)
    batch = _normalize_model_batch(tiles, input_mode=input_mode)
    predicted_batches: list[np.ndarray] = []
    stride = max(1, int(batch_size))
    for start in range(0, len(batch), stride):
        stop = start + stride
        pred = np.asarray(model.predict(batch[start:stop], verbose=0), dtype=np.float32)
        if pred.ndim == 3:
            pred = pred[..., np.newaxis]
        predicted_batches.append(pred)
    stitched = stitch_predictions(np.concatenate(predicted_batches, axis=0), coords, layout)
    return stitched


def _apply_shift(image: np.ndarray, vertical_offset: int = 0, horizontal_offset: int = 0) -> np.ndarray:
    array = np.asarray(image)
    shifted = array.copy()
    if vertical_offset > 0:
        if array.ndim == 2:
            shifted = np.concatenate(
                (array[vertical_offset:, :], np.zeros((vertical_offset, array.shape[1]), dtype=array.dtype)),
                axis=0,
            )
        else:
            shifted = np.concatenate(
                (
                    array[vertical_offset:, :, :],
                    np.zeros((vertical_offset, array.shape[1], array.shape[2]), dtype=array.dtype),
                ),
                axis=0,
            )
    if horizontal_offset > 0:
        if shifted.ndim == 2:
            shifted = np.concatenate(
                (shifted[:, horizontal_offset:], np.zeros((shifted.shape[0], horizontal_offset), dtype=shifted.dtype)),
                axis=1,
            )
        else:
            shifted = np.concatenate(
                (
                    shifted[:, horizontal_offset:, :],
                    np.zeros((shifted.shape[0], horizontal_offset, shifted.shape[2]), dtype=shifted.dtype),
                ),
                axis=1,
            )
    return shifted


def _restore_shift(mask: np.ndarray, vertical_offset: int = 0, horizontal_offset: int = 0) -> np.ndarray:
    restored = np.asarray(mask)
    if vertical_offset > 0:
        if restored.ndim == 2:
            restored = np.concatenate(
                (
                    np.zeros((vertical_offset, restored.shape[1]), dtype=restored.dtype),
                    restored[:-vertical_offset, :],
                ),
                axis=0,
            )
        else:
            restored = np.concatenate(
                (
                    np.zeros((vertical_offset, restored.shape[1], restored.shape[2]), dtype=restored.dtype),
                    restored[:-vertical_offset, :, :],
                ),
                axis=0,
            )
    if horizontal_offset > 0:
        if restored.ndim == 2:
            restored = np.concatenate(
                (
                    np.zeros((restored.shape[0], horizontal_offset), dtype=restored.dtype),
                    restored[:, :-horizontal_offset],
                ),
                axis=1,
            )
        else:
            restored = np.concatenate(
                (
                    np.zeros((restored.shape[0], horizontal_offset, restored.shape[2]), dtype=restored.dtype),
                    restored[:, :-horizontal_offset, :],
                ),
                axis=1,
            )
    return restored


def _prediction_offsets(patch_size: int, refinement_steps: int, tuning: PipelineTuning) -> list[int]:
    if tuning.root_prediction_offsets:
        offsets = [max(0, int(value)) for value in tuning.root_prediction_offsets]
        return list(dict.fromkeys(offsets))
    offsets = [0]
    candidates = [patch_size // 2, patch_size // 4, patch_size // 8]
    for value in candidates[: max(0, int(refinement_steps))]:
        if value > 0 and value not in offsets:
            offsets.append(int(value))
    return offsets


def _predict_binary_probability_map(
    model: Any,
    image_path: str | Path,
    patch_size: int,
    profile: ModelProfile,
    refinement_steps: int,
    tuning: PipelineTuning,
) -> np.ndarray:
    image = _load_image_for_profile(image_path, profile)
    probabilities: list[np.ndarray] = []
    for vertical_offset in _prediction_offsets(patch_size, refinement_steps, tuning):
        shifted = _apply_shift(image, vertical_offset=vertical_offset, horizontal_offset=0)
        prob = _predict_tiled_output(
            model,
            shifted,
            patch_size=patch_size,
            batch_size=profile.batch_size,
            input_mode=profile.input_mode,
            tile_halo=profile.tile_halo,
        )
        if prob.ndim == 3:
            prob = prob[..., 0]
        prob = _restore_shift(prob, vertical_offset=vertical_offset, horizontal_offset=0)
        probabilities.append(prob)
    merged = np.mean(probabilities, axis=0)
    if profile.flip_horizontal and profile.return_original_coords:
        merged = np.fliplr(merged)
    return merged


def _predict_multiclass_labels(
    model: Any,
    image_path: str | Path,
    patch_size: int,
    profile: ModelProfile,
) -> np.ndarray:
    image = _load_image_for_profile(image_path, profile)
    logits = _predict_tiled_output(
        model,
        image,
        patch_size=patch_size,
        batch_size=profile.batch_size,
        input_mode=profile.input_mode,
        tile_halo=profile.tile_halo,
    )
    scales = tuple(float(value) for value in profile.class_probability_scales)
    if scales:
        if len(scales) != int(logits.shape[-1]):
            raise ValueError(
                f"Profile supplies {len(scales)} class probability scales for {logits.shape[-1]} outputs."
            )
        logits = logits * np.asarray(scales, dtype=np.float32).reshape((1, 1, -1))
    labels = np.argmax(logits, axis=-1).astype(np.uint8)
    if profile.flip_horizontal and profile.return_original_coords:
        labels = np.fliplr(labels)
    return labels


def _build_shoot_guard_mask(
    shape_hw: tuple[int, int],
    root_mask: np.ndarray | None,
    dish_mask: np.ndarray,
    tuning: PipelineTuning,
) -> np.ndarray:
    height, width = int(shape_hw[0]), int(shape_hw[1])
    guard = np.zeros((height, width), dtype=np.uint8)
    top_limit = max(int(tuning.shoot_guard_min_top_limit_px), min(height, int(round(height * float(tuning.shoot_guard_top_limit_fraction)))))
    comps: list[tuple[int, int, int, int]] = []
    if root_mask is not None and np.count_nonzero(root_mask) > 0:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats((root_mask > 0).astype(np.uint8), connectivity=8)
        min_area = max(int(tuning.shoot_guard_component_min_area), int(round(float(height * width) * float(tuning.shoot_guard_component_area_fraction))))
        for label_id in range(1, int(count)):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            if y >= max(int(tuning.shoot_guard_component_min_y_px), int(round(height * float(tuning.shoot_guard_component_max_y_fraction)))):
                continue
            comps.append((x, y, w, h))
    if comps:
        pad_x = max(int(tuning.shoot_guard_pad_x_min_px), min(int(tuning.shoot_guard_pad_x_max_px), int(round(width * float(tuning.shoot_guard_pad_x_fraction)))))
        pad_up = max(int(tuning.shoot_guard_pad_up_min_px), min(int(tuning.shoot_guard_pad_up_max_px), int(round(height * float(tuning.shoot_guard_pad_up_fraction)))))
        pad_down = max(int(tuning.shoot_guard_pad_down_min_px), min(int(tuning.shoot_guard_pad_down_max_px), int(round(height * float(tuning.shoot_guard_pad_down_fraction)))))
        for x, y, w, _h in comps:
            cx = int(round(x + (w * 0.5)))
            x0 = max(0, cx - pad_x)
            x1 = min(width, cx + pad_x)
            y0 = max(0, y - pad_up)
            y1 = min(top_limit, y + pad_down)
            guard[y0:y1, x0:x1] = 1
    slot_guard = np.zeros_like(guard)
    slot_half_width = max(
        int(tuning.shoot_guard_slot_half_width_min_px),
        min(int(tuning.shoot_guard_slot_half_width_max_px), int(round(width * float(tuning.shoot_guard_slot_half_width_fraction)))),
    )
    slot_bottom = max(80, min(top_limit, height))
    for center_x in np.linspace(width * 0.12, width * 0.88, max(1, int(tuning.shoot_guard_slot_count))):
        cx = int(round(center_x))
        x0 = max(0, cx - slot_half_width)
        x1 = min(width, cx + slot_half_width)
        slot_guard[:slot_bottom, x0:x1] = 1
    guard = np.logical_or(guard > 0, slot_guard > 0).astype(np.uint8)
    edge_margin = adaptive_side_margin(width, fraction=float(tuning.shoot_guard_edge_margin_fraction), max_px=int(tuning.shoot_guard_edge_margin_max_px))
    if edge_margin > 0:
        guard[:, :edge_margin] = 0
        guard[:, width - edge_margin :] = 0
    return np.logical_and(guard > 0, dish_mask > 0).astype(np.uint8)


def _postprocess_root_mask(root_mask: np.ndarray, image: np.ndarray, tuning: PipelineTuning) -> np.ndarray:
    mask = filter_connected_components(root_mask, min_area=int(tuning.root_min_area))
    if np.count_nonzero(mask) == 0:
        return mask
    height, width = mask.shape[:2]
    edge_margin = adaptive_side_margin(
        width,
        fraction=float(tuning.root_edge_margin_fraction),
        max_px=int(tuning.root_edge_margin_max_px),
    )
    if edge_margin > 0:
        mask[:, :edge_margin] = 0
        mask[:, width - edge_margin :] = 0
    dish_mask = estimate_dish_interior_mask(image)
    mask = np.logical_and(mask > 0, dish_mask > 0).astype(np.uint8)
    return filter_connected_components(mask, min_area=int(tuning.root_min_area))


def _postprocess_shoot_mask_legacy(shoot_mask: np.ndarray) -> np.ndarray:
    mask = (np.asarray(shoot_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if np.count_nonzero(mask) == 0:
        return mask
    height, width = mask.shape[:2]
    exclusion = np.ones((height, width), dtype=np.uint8)
    center_x, center_y = width // 2, height // 2
    cv2.circle(exclusion, (center_x, center_y - 250), max(1, height // 2), 0, thickness=-1)
    cv2.rectangle(exclusion, (0, 0), (min(width, 700), height), 255, thickness=-1)
    cv2.rectangle(exclusion, (max(0, width - 800), 0), (width, height), 255, thickness=-1)
    mask[exclusion != 0] = 0
    # Match the original BW shoot cleanup more closely: remove dust, close small
    # gaps, and fill enclosed leaf holes so older/senescing rosettes stay solid.
    mask = remove_noise(mask, min_area=50, apply_closing=True, closing_iterations=2, fill_holes=True)
    return mask.astype(np.uint8)


def _postprocess_shoot_mask_bw_arabidopsis(
    shoot_mask: np.ndarray,
    root_mask: np.ndarray | None = None,
    image: np.ndarray | None = None,
    tuning: PipelineTuning | None = None,
) -> np.ndarray:
    # Use the same dish/root-aware guard as the default pipeline. The old BW
    # cleanup used fixed center/edge geometry that can erase side seedlings on
    # five-lane Lucifer plates.
    tuning = tuning or DEFAULT_PIPELINE_TUNING
    mask = filter_connected_components(shoot_mask, min_area=int(tuning.shoot_min_area))
    if np.count_nonzero(mask) == 0:
        return mask
    height, width = mask.shape[:2]
    edge_margin = adaptive_side_margin(
        width,
        fraction=min(float(tuning.shoot_edge_margin_fraction), 0.05),
        max_px=min(int(tuning.shoot_edge_margin_max_px), 220),
    )
    if edge_margin > 0:
        mask[:, :edge_margin] = 0
        mask[:, width - edge_margin :] = 0
    lower_start = min(height, max(0, int(round(height * float(tuning.shoot_lower_start_fraction)))))
    if lower_start < height:
        mask[lower_start:, :] = 0

    if image is not None:
        dish_mask = estimate_dish_interior_mask(image)
        if dish_mask.shape[:2] != mask.shape[:2]:
            dish_mask = cv2.resize(dish_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = np.logical_and(mask > 0, dish_mask > 0).astype(np.uint8)
    else:
        dish_mask = np.ones_like(mask, dtype=np.uint8)

    guard = _build_shoot_guard_mask(mask.shape[:2], root_mask=root_mask, dish_mask=dish_mask, tuning=tuning)
    guarded = np.logical_and(mask > 0, guard > 0).astype(np.uint8)
    guarded = filter_connected_components(guarded, min_area=int(tuning.shoot_min_area))
    if np.count_nonzero(guarded) >= max(20, int(round(np.count_nonzero(mask) * 0.04))):
        mask = guarded
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = remove_noise(mask, min_area=50, apply_closing=True, closing_iterations=3, fill_holes=True)
    return mask.astype(np.uint8)


def _postprocess_shoot_mask_simple(shoot_mask: np.ndarray, tuning: PipelineTuning) -> np.ndarray:
    mask = (np.asarray(shoot_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if np.count_nonzero(mask) == 0:
        return mask
    return remove_noise(
        mask,
        min_area=max(1, int(tuning.shoot_min_area)),
        apply_closing=True,
        closing_iterations=2,
        fill_holes=True,
    )


def _postprocess_shoot_mask(shoot_mask: np.ndarray, root_mask: np.ndarray, image: np.ndarray, tuning: PipelineTuning) -> np.ndarray:
    mode = str(tuning.shoot_postprocess_mode).strip().lower()
    if mode in {"native", "passthrough"}:
        return filter_connected_components(shoot_mask, min_area=max(1, int(tuning.shoot_min_area)))
    if mode == "legacy":
        return _postprocess_shoot_mask_legacy(shoot_mask)
    if mode == "bw_arabidopsis":
        return _postprocess_shoot_mask_bw_arabidopsis(shoot_mask, root_mask=root_mask, image=image, tuning=tuning)
    if mode == "simple":
        return _postprocess_shoot_mask_simple(shoot_mask, tuning)
    mask = filter_connected_components(shoot_mask, min_area=int(tuning.shoot_min_area))
    if np.count_nonzero(mask) == 0:
        return mask
    height, width = mask.shape[:2]
    edge_margin = adaptive_side_margin(
        width,
        fraction=float(tuning.shoot_edge_margin_fraction),
        max_px=int(tuning.shoot_edge_margin_max_px),
    )
    if edge_margin > 0:
        mask[:, :edge_margin] = 0
        mask[:, width - edge_margin :] = 0
    lower_start = min(height, max(0, int(round(height * float(tuning.shoot_lower_start_fraction)))))
    if lower_start < height:
        mask[lower_start:, :] = 0
    mask = filter_connected_components(mask, min_area=int(tuning.shoot_min_area))
    dish_mask = estimate_dish_interior_mask(image)
    mask = np.logical_and(mask > 0, dish_mask > 0).astype(np.uint8)
    guard = _build_shoot_guard_mask(mask.shape[:2], root_mask=root_mask, dish_mask=dish_mask, tuning=tuning)
    guarded = np.logical_and(mask > 0, guard > 0).astype(np.uint8)
    guarded = filter_connected_components(guarded, min_area=int(tuning.shoot_min_area))
    if np.count_nonzero(guarded) >= max(20, int(round(np.count_nonzero(mask) * float(tuning.shoot_guard_accept_ratio)))):
        target = guarded
    else:
        target = mask
    return remove_noise(
        target,
        min_area=max(1, int(tuning.shoot_min_area)),
        apply_closing=True,
        closing_iterations=2,
        fill_holes=True,
    )


def _inpainter_correcter(original_mask: np.ndarray, expanded_mask: np.ndarray) -> np.ndarray:
    base = (original_mask > 0).astype(np.uint8)
    expanded = (expanded_mask > 0).astype(np.uint8)
    diff = (expanded > base).astype(np.uint8)
    fill_count, fill_labels, fill_stats, _ = cv2.connectedComponentsWithStats(diff, connectivity=8)
    _orig_count, orig_labels = cv2.connectedComponents(base, connectivity=8)
    corrected = base.copy()
    height, width = base.shape[:2]
    kernel = np.ones((3, 3), np.uint8)
    for label_id in range(1, int(fill_count)):
        x, y, w, h, area = fill_stats[label_id]
        if int(area) <= 0:
            continue
        x0 = max(int(x) - 1, 0)
        y0 = max(int(y) - 1, 0)
        x1 = min(int(x + w) + 1, width)
        y1 = min(int(y + h) + 1, height)
        roi_comp = (fill_labels[y0:y1, x0:x1] == label_id).astype(np.uint8)
        if roi_comp.max() == 0:
            continue
        roi_neigh = cv2.dilate(roi_comp, kernel, iterations=1).astype(bool)
        touching = orig_labels[y0:y1, x0:x1][roi_neigh]
        touching = touching[touching != 0]
        if touching.size and np.unique(touching).size >= 2:
            corrected[y0:y1, x0:x1][roi_comp.astype(bool)] = 1
    return corrected.astype(np.uint8)


def _build_occlusion_mask(root_mask: np.ndarray, refinement_steps: int) -> np.ndarray:
    iterations = max(1, int(refinement_steps))
    kernel = np.ones((5, 2), np.uint8)
    dilated = cv2.dilate((root_mask > 0).astype(np.uint8), kernel, iterations=iterations)
    return _inpainter_correcter(root_mask, dilated)


def _masks_from_multiclass(labels: np.ndarray, profile: ModelProfile) -> tuple[np.ndarray, np.ndarray]:
    root_mask = np.isin(labels, profile.root_label_ids).astype(np.uint8)
    shoot_labels = set(profile.shoot_label_ids)
    if profile.include_seed_in_shoot:
        shoot_labels.update(profile.seed_label_ids)
    shoot_mask = np.isin(labels, tuple(sorted(shoot_labels))).astype(np.uint8)
    return root_mask, shoot_mask


class ModelMaskResult(tuple):
    """Four-item legacy result with optional native multiclass labels attached."""

    def __new__(
        cls,
        root_mask: np.ndarray,
        shoot_mask: np.ndarray,
        occlusion_mask: np.ndarray,
        flagged: bool,
        *,
        semantic_labels: np.ndarray | None = None,
    ):
        result = super().__new__(cls, (root_mask, shoot_mask, occlusion_mask, flagged))
        result.semantic_labels = semantic_labels
        return result


def model_create_masks(
    image_path: str,
    patch_size: int,
    segmentation_model: Any,
    shoot_model: Any | None,
    refinement_steps: int = 1,
    verbose: bool = True,
    profile_overrides: dict[str, object] | None = None,
    pipeline_overrides: dict[str, object] | None = None,
    **_kwargs,
):
    profile = infer_model_profile(segmentation_model, profile_overrides=profile_overrides)
    tuning = resolve_pipeline_tuning(pipeline_overrides)
    if verbose:
        logger.info("Running NPEC PyPhenotyper profile: %s", profile.name)
    processing_image = _load_processing_rgb(image_path, profile)
    if profile.mode == "multiclass":
        labels = _predict_multiclass_labels(segmentation_model, image_path, int(patch_size), profile)
        root_mask, shoot_mask = _masks_from_multiclass(labels, profile)
    else:
        effective_shoot_model = shoot_model if shoot_model is not None else segmentation_model
        root_prob = _predict_binary_probability_map(
            segmentation_model,
            image_path=image_path,
            patch_size=int(patch_size),
            profile=profile,
            refinement_steps=int(refinement_steps),
            tuning=tuning,
        )
        shoot_prob = _predict_binary_probability_map(
            effective_shoot_model,
            image_path=image_path,
            patch_size=int(patch_size),
            profile=profile,
            refinement_steps=int(tuning.shoot_prediction_refinement_steps),
            tuning=tuning,
        )
        root_mask = (root_prob >= float(profile.root_threshold)).astype(np.uint8)
        shoot_mask = (shoot_prob >= float(profile.shoot_threshold)).astype(np.uint8)
    root_mask = _postprocess_root_mask(root_mask, image=processing_image, tuning=tuning)
    shoot_mask = _postprocess_shoot_mask(shoot_mask, root_mask=root_mask, image=processing_image, tuning=tuning)
    occlusion_mask = _build_occlusion_mask(root_mask, refinement_steps=int(refinement_steps))
    root_top_pixels = int(np.count_nonzero(root_mask[: max(1, root_mask.shape[0] // 2), :]))
    shoot_top_pixels = int(np.count_nonzero(shoot_mask[: max(1, shoot_mask.shape[0] // 2), :]))
    flag = bool(root_top_pixels <= 25 or shoot_top_pixels <= 75)
    return ModelMaskResult(
        root_mask.astype(np.uint8),
        shoot_mask.astype(np.uint8),
        occlusion_mask.astype(np.uint8),
        flag,
        semantic_labels=(labels.astype(np.uint8) if profile.mode == "multiclass" else None),
    )
