from __future__ import annotations

from dataclasses import dataclass
import json
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from ..models import LabelClass


DATASET_REGISTRY_PATH = Path(__file__).resolve().parent / "public_dataset_registry.json"
LABEL_SCHEMA_PATH = Path(__file__).resolve().parent / "normalized_label_schema.json"
GENERAL_ROOT_STARTER_VERSION = 1
SCHEMA_KEYS = ("shoot", "primary_root", "lateral_root", "root_binary", "seed_crown")


@dataclass(slots=True)
class RoutingImageFeatures:
    width: int
    height: int
    megapixels: float
    is_rgb_like: bool
    channel_delta_mean: float
    green_dominance: float
    mean_intensity: float
    intensity_std: float
    top_dark_ratio: float


def _json_load(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


@lru_cache(maxsize=1)
def load_public_dataset_registry() -> list[dict[str, object]]:
    return _json_load(DATASET_REGISTRY_PATH)


@lru_cache(maxsize=1)
def load_normalized_label_schema() -> list[dict[str, object]]:
    return _json_load(LABEL_SCHEMA_PATH)


def default_moe_training_plan() -> dict[str, object]:
    return {
        "version": GENERAL_ROOT_STARTER_VERSION,
        "name": "general_root_starter",
        "normalized_schema": list(SCHEMA_KEYS),
        "encoder": {
            "family": "shared_unet_encoder",
            "input_channels": 3,
            "notes": "Shared encoder for plate grayscale, RGB plate, and future public-root experts.",
        },
        "heads": [
            {
                "key": "plate_bw_hades",
                "status": "active",
                "tasks": ["root_binary", "shoot"],
                "notes": "Current bundled grayscale Arabidopsis expert.",
            },
            {
                "key": "plate_rgb_lucifer",
                "status": "active",
                "tasks": ["root_binary", "shoot"],
                "notes": "Current bundled RGB Arabidopsis expert.",
            },
            {
                "key": "potato_rgb",
                "status": "active",
                "tasks": ["root_binary", "shoot"],
                "notes": "Current bundled potato expert.",
            },
            {
                "key": "arabidopsis_dark_rgb",
                "status": "active",
                "tasks": ["root_binary", "shoot"],
                "notes": "Bundled dark-background RGB Arabidopsis expert trained from MultipleXLab labels.",
            },
            {
                "key": "minirhizotron_public",
                "status": "experimental",
                "tasks": ["root_binary"],
                "notes": "RootPainter-style / public minirhizotron branch. First corpus and checkpoint exist; extend with more public-root data before promoting it to the default starter stack.",
            },
        ],
        "router": {
            "stages": [
                "image statistics prior",
                "expert prediction confidence",
                "3-10 image calibration reranking",
            ],
            "image_statistics": [
                "rgb/grayscale family",
                "resolution",
                "channel delta",
                "green dominance",
                "top-region darkness",
            ],
            "prediction_confidence": [
                "root area ratio",
                "shoot area ratio",
                "root vertical span",
                "shoot top occupancy",
                "root below shoot consistency",
            ],
        },
    }


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def _schema_entry(key: str) -> dict[str, object]:
    for entry in load_normalized_label_schema():
        if str(entry.get("key", "")).strip() == key:
            return entry
    return {}


def _matches_synonyms(name: str, key: str) -> bool:
    normalized = _normalized_name(name)
    entry = _schema_entry(key)
    synonyms = entry.get("synonyms", [])
    if isinstance(synonyms, list):
        for raw in synonyms:
            token = _normalized_name(str(raw))
            if token and (token == normalized or token in normalized):
                return True
    return False


def _to_grayscale_u8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.uint8)
    if cv2 is not None:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    rgb_f = arr.astype(np.float32)
    gray = (0.299 * rgb_f[..., 0]) + (0.587 * rgb_f[..., 1]) + (0.114 * rgb_f[..., 2])
    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


def _resize_mask_nearest(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    arr = np.asarray(mask, dtype=np.uint8)
    if arr.shape == (h, w):
        return arr
    if cv2 is not None:
        return np.array(
            cv2.resize(arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST),
            dtype=np.uint8,
        )
    image = Image.fromarray(arr)
    return np.asarray(image.resize((w, h), Image.Resampling.NEAREST), dtype=np.uint8)


def describe_image_features(image: np.ndarray) -> RoutingImageFeatures:
    arr = np.asarray(image)
    if arr.ndim == 2:
        rgb = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3]
    else:
        rgb = np.repeat(np.asarray(arr)[..., None], 3, axis=2)
    rgb = np.asarray(rgb, dtype=np.uint8)
    height, width = rgb.shape[:2]
    megapixels = float(height * width) / 1_000_000.0
    rgb_f = rgb.astype(np.float32)
    channel_delta_mean = max(
        float(np.mean(np.abs(rgb_f[..., 0] - rgb_f[..., 1]))),
        float(np.mean(np.abs(rgb_f[..., 1] - rgb_f[..., 2]))),
        float(np.mean(np.abs(rgb_f[..., 0] - rgb_f[..., 2]))),
    )
    pixel_chroma = np.max(rgb_f, axis=2) - np.min(rgb_f, axis=2)
    strong_chroma_fraction = float(np.mean(pixel_chroma >= 8.0))
    moderate_chroma_fraction = float(np.mean(pixel_chroma >= 3.0))
    is_rgb_like = bool(
        channel_delta_mean > 1.5
        or strong_chroma_fraction >= 0.0005
        or (channel_delta_mean > 0.5 and moderate_chroma_fraction >= 0.02)
    )
    green_dominance = float(np.mean(np.maximum(0.0, rgb_f[..., 1] - np.maximum(rgb_f[..., 0], rgb_f[..., 2])) / 255.0))
    gray = _to_grayscale_u8(rgb)
    top_band = gray[: max(1, int(round(height * 0.2))), :]
    return RoutingImageFeatures(
        width=int(width),
        height=int(height),
        megapixels=megapixels,
        is_rgb_like=is_rgb_like,
        channel_delta_mean=float(channel_delta_mean),
        green_dominance=float(green_dominance),
        mean_intensity=float(np.mean(gray) / 255.0),
        intensity_std=float(np.std(gray) / 255.0),
        top_dark_ratio=float(np.mean(top_band < 96)) if top_band.size > 0 else 0.0,
    )


def features_as_dict(features: RoutingImageFeatures) -> dict[str, object]:
    return {
        "width": int(features.width),
        "height": int(features.height),
        "megapixels": float(features.megapixels),
        "is_rgb_like": bool(features.is_rgb_like),
        "channel_delta_mean": float(features.channel_delta_mean),
        "green_dominance": float(features.green_dominance),
        "mean_intensity": float(features.mean_intensity),
        "intensity_std": float(features.intensity_std),
        "top_dark_ratio": float(features.top_dark_ratio),
    }


def expert_prior_score(
    features: RoutingImageFeatures,
    *,
    expert_key: str,
    image_mode: str,
    family: str,
    priority: float = 0.0,
    hints: dict[str, object] | None = None,
    calibration_bias: float = 0.0,
) -> float:
    mode = str(image_mode or "any").strip().lower()
    family_key = str(family or expert_key or "generic").strip().lower()
    score = float(priority) + float(calibration_bias)
    if mode == "rgb_only":
        score += 0.4 if features.is_rgb_like else -4.0
    elif mode == "grayscale_only":
        score += 0.4 if not features.is_rgb_like else -4.0

    if family_key in {"arabidopsis_plate_bw", "plate_bw_hades"}:
        score += 0.25 if not features.is_rgb_like else -0.35
        score += min(0.20, features.top_dark_ratio * 0.35)
        score += min(0.18, max(0.0, 0.9 - features.megapixels) * 0.18)
    elif family_key in {"arabidopsis_plate_rgb", "plate_rgb_lucifer"}:
        score += 0.20 if features.is_rgb_like else -0.50
        score += min(0.25, features.green_dominance * 2.2)
        score += min(0.15, max(0.0, features.megapixels - 0.6) * 0.10)
    elif family_key in {"potato_rgb", "potato_plate_rgb"}:
        score += 0.18 if features.is_rgb_like else -0.50
        score += min(0.22, features.green_dominance * 1.8)
        score += 0.08 if features.megapixels >= 0.6 else 0.0
    elif family_key in {"arabidopsis_dark_rgb", "arabidopsis_plate_rgb_dark", "plate_rgb_dark_mxlab"}:
        score += 0.18 if features.is_rgb_like else -0.55
        score += min(0.30, features.top_dark_ratio * 0.32)
        score += min(0.18, max(0.0, 0.18 - features.mean_intensity) * 1.6)
        score += min(0.10, max(0.0, features.megapixels - 1.0) * 0.02)
        score -= min(0.16, features.green_dominance * 12.0)
    elif family_key in {"minirhizotron_public", "minirhizotron"}:
        score += 0.12
        score += 0.10 if features.mean_intensity < 0.55 else -0.05

    if isinstance(hints, dict):
        if bool(hints.get("prefer_high_res")):
            score += min(0.12, max(0.0, features.megapixels - 0.8) * 0.08)
        if bool(hints.get("prefer_low_res")):
            score += min(0.12, max(0.0, 1.0 - features.megapixels) * 0.08)
        if bool(hints.get("prefer_green")):
            score += min(0.12, features.green_dominance * 1.5)
        if bool(hints.get("prefer_dark_top")):
            score += min(0.12, features.top_dark_ratio * 0.25)
    return float(score)


def _range_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    if low <= value <= high:
        return 1.0
    span = max(1e-6, high - low)
    distance = (low - value) if value < low else (value - high)
    return max(0.0, 1.0 - (distance / (span * 2.0)))


def mask_confidence(
    *,
    root_mask: np.ndarray,
    shoot_mask: np.ndarray,
    image_shape_hw: tuple[int, int],
    family: str,
    hints: dict[str, object] | None = None,
) -> dict[str, object]:
    h, w = int(image_shape_hw[0]), int(image_shape_hw[1])
    total_pixels = float(max(1, h * w))
    root_bin = np.asarray(root_mask, dtype=np.uint8) > 0
    shoot_bin = np.asarray(shoot_mask, dtype=np.uint8) > 0
    root_area_ratio = float(np.count_nonzero(root_bin) / total_pixels)
    shoot_area_ratio = float(np.count_nonzero(shoot_bin) / total_pixels)

    root_span_y = 0.0
    root_below_shoot_ratio = 0.5
    if np.any(root_bin):
        ys_root, _xs_root = np.where(root_bin)
        root_span_y = float((ys_root.max() - ys_root.min() + 1) / max(1, h))
    shoot_top_ratio = 0.0
    if np.any(shoot_bin):
        ys_shoot, _xs_shoot = np.where(shoot_bin)
        shoot_top_ratio = float(np.mean(ys_shoot < max(1, int(round(h * 0.35)))))
        if np.any(root_bin):
            root_below_shoot_ratio = float(np.mean(ys_root >= float(np.mean(ys_shoot))))

    defaults = {
        "root_area_range": (0.0001, 0.18),
        "shoot_area_range": (0.00002, 0.08),
        "root_span_range": (0.12, 0.98),
    }
    family_key = str(family or "").strip().lower()
    if family_key in {"potato_rgb", "potato_plate_rgb"}:
        defaults["shoot_area_range"] = (0.00004, 0.12)
    if family_key in {"minirhizotron_public", "minirhizotron"}:
        defaults["shoot_area_range"] = (0.0, 0.005)
        defaults["root_span_range"] = (0.08, 1.0)

    if isinstance(hints, dict):
        for key, value in hints.items():
            if key in defaults and isinstance(value, (tuple, list)) and len(value) == 2:
                defaults[key] = (float(value[0]), float(value[1]))

    root_area_score = _range_score(root_area_ratio, *defaults["root_area_range"])
    shoot_area_score = _range_score(shoot_area_ratio, *defaults["shoot_area_range"])
    root_span_score = _range_score(root_span_y, *defaults["root_span_range"])
    shoot_top_score = _range_score(shoot_top_ratio, 0.70, 1.0) if shoot_area_ratio > 0 else 0.5
    root_below_shoot_score = _range_score(root_below_shoot_ratio, 0.60, 1.0) if shoot_area_ratio > 0 and root_area_ratio > 0 else 0.5

    confidence = (
        (0.26 * root_area_score)
        + (0.18 * shoot_area_score)
        + (0.24 * root_span_score)
        + (0.18 * shoot_top_score)
        + (0.14 * root_below_shoot_score)
    )
    return {
        "score": float(confidence),
        "root_area_ratio": float(root_area_ratio),
        "shoot_area_ratio": float(shoot_area_ratio),
        "root_span_y_ratio": float(root_span_y),
        "shoot_top_ratio": float(shoot_top_ratio),
        "root_below_shoot_ratio": float(root_below_shoot_ratio),
    }


def normalize_annotation_targets(
    layers: dict[int, np.ndarray] | None,
    classes: list[LabelClass],
    shape_hw: tuple[int, int],
) -> dict[str, np.ndarray]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    blank = np.zeros((h, w), dtype=np.uint8)
    normalized = {key: blank.copy() for key in SCHEMA_KEYS}
    per_image = layers if isinstance(layers, dict) else {}

    for cls in classes:
        mask = per_image.get(int(cls.class_id))
        if mask is None:
            continue
        layer = np.asarray(mask, dtype=np.uint8)
        if layer.shape != (h, w):
            layer = _resize_mask_nearest(layer, (h, w))
        layer = (layer > 0).astype(np.uint8)
        name = cls.name
        if _matches_synonyms(name, "shoot"):
            normalized["shoot"] = np.maximum(normalized["shoot"], layer)
        if _matches_synonyms(name, "primary_root"):
            normalized["primary_root"] = np.maximum(normalized["primary_root"], layer)
            normalized["root_binary"] = np.maximum(normalized["root_binary"], layer)
        if _matches_synonyms(name, "lateral_root"):
            normalized["lateral_root"] = np.maximum(normalized["lateral_root"], layer)
            normalized["root_binary"] = np.maximum(normalized["root_binary"], layer)
        if _matches_synonyms(name, "root_binary"):
            normalized["root_binary"] = np.maximum(normalized["root_binary"], layer)
        if _matches_synonyms(name, "seed_crown"):
            normalized["seed_crown"] = np.maximum(normalized["seed_crown"], layer)

    if np.count_nonzero(normalized["root_binary"]) <= 0:
        normalized["root_binary"] = np.maximum(normalized["root_binary"], normalized["primary_root"])
        normalized["root_binary"] = np.maximum(normalized["root_binary"], normalized["lateral_root"])
    if np.count_nonzero(normalized["primary_root"]) <= 0 and np.count_nonzero(normalized["root_binary"]) > 0:
        normalized["primary_root"] = normalized["root_binary"].copy()
    return normalized


def iou_score(prediction: np.ndarray, target: np.ndarray) -> float | None:
    pred = np.asarray(prediction, dtype=np.uint8) > 0
    truth = np.asarray(target, dtype=np.uint8) > 0
    union = float(np.count_nonzero(np.logical_or(pred, truth)))
    if union <= 0.0:
        return None
    inter = float(np.count_nonzero(np.logical_and(pred, truth)))
    return inter / union


def build_calibration_profile(
    expert_scores: list[dict[str, object]],
    *,
    selected_uids: list[str],
    schema_keys: tuple[str, ...] = ("shoot", "root_binary"),
) -> dict[str, object]:
    if not expert_scores:
        raise ValueError("No expert scores were provided for calibration.")
    mean_scores = [float(row.get("combined_mean_iou") or 0.0) for row in expert_scores]
    baseline = float(np.mean(mean_scores)) if mean_scores else 0.0
    sorted_rows = sorted(expert_scores, key=lambda row: float(row.get("combined_mean_iou") or -1.0), reverse=True)
    best = sorted_rows[0]
    second = sorted_rows[1] if len(sorted_rows) > 1 else None
    expert_score_bias = {
        str(row.get("expert_key")): float(row.get("combined_mean_iou") or 0.0) - baseline
        for row in expert_scores
        if row.get("expert_key") is not None
    }
    best_delta = float(best.get("combined_mean_iou") or 0.0) - float(second.get("combined_mean_iou") or 0.0) if second else 0.0
    return {
        "version": GENERAL_ROOT_STARTER_VERSION,
        "mode": "rerank",
        "selected_uids": list(selected_uids),
        "schema_keys": list(schema_keys),
        "preferred_expert_key": str(best.get("expert_key") or ""),
        "preferred_expert_label": str(best.get("label") or best.get("expert_key") or ""),
        "preferred_margin": float(best_delta),
        "expert_score_bias": expert_score_bias,
        "expert_scores": expert_scores,
        "notes": (
            "Quick calibration reranks built-in experts from 3-10 labeled images. "
            "Use this when the auto router is unsure or the dataset is outside the standard Arabidopsis/Potato families."
        ),
    }
