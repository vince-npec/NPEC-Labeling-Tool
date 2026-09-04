from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from .models import DatasetImageItem, LabelClass


def _clamp_unit(value: object, fallback: float = 0.0) -> float:
    try:
        resolved = float(value)
    except Exception:
        resolved = float(fallback)
    return max(0.0, min(1.0, resolved))


def _mask_union_pixels(per_class: dict[int, np.ndarray] | None) -> int:
    if not per_class:
        return 0
    masks = [np.asarray(mask, dtype=np.uint8) > 0 for mask in per_class.values() if isinstance(mask, np.ndarray)]
    if not masks:
        return 0
    union = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        if mask.shape != union.shape:
            continue
        union |= mask
    return int(np.count_nonzero(union))


def _resize_mask_nearest(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.uint8)
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if arr.shape == (h, w):
        return arr
    pil = Image.fromarray(arr, mode="L")
    return np.asarray(pil.resize((w, h), Image.NEAREST), dtype=np.uint8)


def _index_mask_to_layers(
    index_mask: np.ndarray,
    classes: list[LabelClass],
    shape_hw: tuple[int, int],
) -> dict[int, np.ndarray]:
    arr = _resize_mask_nearest(np.asarray(index_mask, dtype=np.uint8), shape_hw)
    return {
        int(cls.class_id): (arr == int(cls.class_id)).astype(np.uint8)
        for cls in classes
    }


def annotation_coverage_fraction(per_class: dict[int, np.ndarray] | None, shape_hw: tuple[int, int]) -> float:
    h, w = shape_hw
    total = max(1, int(h) * int(w))
    return float(_mask_union_pixels(per_class) / total)


def prediction_coverage_fraction(index_mask: np.ndarray | None) -> float:
    if not isinstance(index_mask, np.ndarray) or index_mask.size <= 0:
        return 0.0
    total = max(1, int(index_mask.size))
    return float(np.count_nonzero(np.asarray(index_mask, dtype=np.uint8) > 0) / total)


def derive_uncertainty_payload(
    prediction: np.ndarray | None,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = dict(details) if isinstance(details, dict) else {}
    coverage = prediction_coverage_fraction(prediction)
    unique_labels = 0
    if isinstance(prediction, np.ndarray) and prediction.size > 0:
        unique_labels = int(len(np.unique(np.asarray(prediction, dtype=np.uint8))))

    confidence = None
    uncertainty = None
    margin = None

    if "confidence_score" in payload:
        confidence = _clamp_unit(payload.get("confidence_score"))
    elif "routing_mask_score" in payload:
        confidence = _clamp_unit(payload.get("routing_mask_score"))
    elif "routing_confidence" in payload:
        confidence = _clamp_unit(payload.get("routing_confidence"))

    if "uncertainty_score" in payload:
        uncertainty = _clamp_unit(payload.get("uncertainty_score"))

    if "margin_score" in payload:
        margin = _clamp_unit(payload.get("margin_score"))
    elif "routing_margin" in payload:
        margin = _clamp_unit(payload.get("routing_margin"))

    if confidence is None and margin is not None:
        confidence = margin
    if uncertainty is None and confidence is not None:
        uncertainty = float(1.0 - confidence)
    if confidence is None and uncertainty is not None:
        confidence = float(1.0 - uncertainty)

    if confidence is None:
        if isinstance(prediction, np.ndarray) and prediction.size > 0:
            # Fallback: prefer predictions with moderate coverage over empty/all-on masks.
            confidence = float(max(0.0, 1.0 - min(1.0, abs(coverage - 0.18) / 0.18)))
        else:
            confidence = 0.0
    if uncertainty is None:
        uncertainty = float(1.0 - confidence)

    payload["confidence_score"] = float(confidence)
    payload["uncertainty_score"] = float(uncertainty)
    if margin is not None:
        payload["margin_score"] = float(margin)
    payload["prediction_coverage_fraction"] = float(coverage)
    payload["prediction_nonzero_pixels"] = int(np.count_nonzero(np.asarray(prediction, dtype=np.uint8) > 0)) if isinstance(prediction, np.ndarray) else 0
    payload["prediction_unique_labels"] = int(unique_labels)
    return payload


def build_hitl_candidate_rows(
    dataset_items: list[DatasetImageItem],
    annotations: dict[str, dict[int, np.ndarray]],
    predictions: dict[str, np.ndarray],
    prediction_metadata_by_uid: dict[str, dict[str, object]] | None = None,
    pseudo_labels: dict[str, np.ndarray] | None = None,
    review_state: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    meta_store = prediction_metadata_by_uid or {}
    pseudo_store = pseudo_labels or {}
    review_store = review_state or {}
    rows: list[dict[str, object]] = []

    for index, item in enumerate(dataset_items):
        layers = annotations.get(item.uid, {})
        labeled_fraction = annotation_coverage_fraction(layers, item.image.shape[:2])
        is_labeled = bool(labeled_fraction > 0.0)
        prediction = predictions.get(item.uid)
        pseudo = pseudo_store.get(item.uid)
        review = review_store.get(item.uid, {}) if isinstance(review_store.get(item.uid, {}), dict) else {}
        summary = derive_uncertainty_payload(prediction, meta_store.get(item.uid))
        confidence = float(summary.get("confidence_score", 0.0))
        uncertainty = float(summary.get("uncertainty_score", 1.0))
        coverage = float(summary.get("prediction_coverage_fraction", 0.0))
        status = str(review.get("status", "none") or "none")
        if status == "none" and pseudo is not None:
            status = "pending"
        has_prediction = isinstance(prediction, np.ndarray)
        has_pseudo = isinstance(pseudo, np.ndarray)

        if not has_prediction:
            recommendation = "Run prediction"
            priority = 3.0
        elif is_labeled:
            recommendation = "Already labeled"
            priority = 0.0
        elif has_pseudo and status in {"pending", "opened"}:
            recommendation = "Review pseudo-label"
            priority = 4.0 + uncertainty
        elif has_pseudo and status == "auto-accepted":
            recommendation = "Auto-accepted for retraining"
            priority = 1.5 + uncertainty
        elif has_pseudo and status in {"accepted", "refined"}:
            recommendation = "Accepted for training"
            priority = 1.0
        elif confidence >= 0.80 and coverage > 0.0:
            recommendation = "Good pseudo-label candidate"
            priority = 2.5 + uncertainty
        else:
            recommendation = "Prioritize for human labeling"
            priority = 3.5 + uncertainty

        rows.append(
            {
                "uid": item.uid,
                "dataset_index": int(index),
                "image_name": item.name,
                "labeled": bool(is_labeled),
                "labeled_fraction": float(labeled_fraction),
                "has_prediction": bool(has_prediction),
                "has_pseudo_label": bool(has_pseudo),
                "pseudo_status": status,
                "confidence_score": float(confidence),
                "uncertainty_score": float(uncertainty),
                "prediction_coverage_fraction": float(coverage),
                "prediction_unique_labels": int(summary.get("prediction_unique_labels", 0)),
                "recommendation": recommendation,
                "priority_score": float(priority),
                "metadata": summary,
            }
        )

    rows.sort(
        key=lambda row: (
            0 if not bool(row.get("labeled")) else 1,
            -float(row.get("priority_score", 0.0)),
            str(row.get("image_name", "")),
        )
    )
    return rows


def stage_pseudo_labels(
    dataset_items: list[DatasetImageItem],
    annotations: dict[str, dict[int, np.ndarray]],
    predictions: dict[str, np.ndarray],
    prediction_metadata_by_uid: dict[str, dict[str, object]] | None,
    pseudo_labels: dict[str, np.ndarray],
    review_state: dict[str, dict[str, object]],
    *,
    min_confidence: float = 0.80,
) -> dict[str, object]:
    staged = 0
    skipped_labeled = 0
    skipped_low_conf = 0
    skipped_missing = 0
    staged_uids: list[str] = []
    threshold = _clamp_unit(min_confidence, fallback=0.8)

    for item in dataset_items:
        if annotation_coverage_fraction(annotations.get(item.uid, {}), item.image.shape[:2]) > 0.0:
            skipped_labeled += 1
            continue
        prediction = predictions.get(item.uid)
        if not isinstance(prediction, np.ndarray):
            skipped_missing += 1
            continue
        summary = derive_uncertainty_payload(prediction, (prediction_metadata_by_uid or {}).get(item.uid))
        confidence = float(summary.get("confidence_score", 0.0))
        coverage = float(summary.get("prediction_coverage_fraction", 0.0))
        if confidence < threshold or coverage <= 0.0:
            skipped_low_conf += 1
            continue
        pseudo_labels[item.uid] = np.asarray(prediction, dtype=np.uint8).copy()
        review_state[item.uid] = {
            "status": "pending",
            "confidence_score": float(confidence),
            "uncertainty_score": float(summary.get("uncertainty_score", 1.0)),
            "prediction_coverage_fraction": float(coverage),
            "source": str(summary.get("backend", "prediction")),
        }
        staged += 1
        staged_uids.append(item.uid)

    return {
        "staged": int(staged),
        "skipped_labeled": int(skipped_labeled),
        "skipped_low_conf": int(skipped_low_conf),
        "skipped_missing": int(skipped_missing),
        "staged_uids": staged_uids,
    }


def auto_accept_pseudo_labels(
    dataset_items: list[DatasetImageItem],
    annotations: dict[str, dict[int, np.ndarray]],
    pseudo_labels: dict[str, np.ndarray],
    review_state: dict[str, dict[str, object]],
    *,
    min_confidence: float = 0.80,
) -> dict[str, object]:
    accepted = 0
    skipped_labeled = 0
    skipped_missing = 0
    skipped_low_conf = 0
    accepted_uids: list[str] = []
    threshold = _clamp_unit(min_confidence, fallback=0.8)

    for item in dataset_items:
        if annotation_coverage_fraction(annotations.get(item.uid, {}), item.image.shape[:2]) > 0.0:
            skipped_labeled += 1
            continue
        pseudo = pseudo_labels.get(item.uid)
        if not isinstance(pseudo, np.ndarray):
            skipped_missing += 1
            continue
        state = dict(review_state.get(item.uid, {}))
        confidence = _clamp_unit(state.get("confidence_score"), fallback=0.0)
        if confidence < threshold:
            skipped_low_conf += 1
            continue
        state["status"] = "auto-accepted"
        state["accepted_for_training"] = True
        state["confidence_score"] = float(confidence)
        review_state[item.uid] = state
        accepted += 1
        accepted_uids.append(item.uid)

    return {
        "auto_accepted": int(accepted),
        "skipped_labeled": int(skipped_labeled),
        "skipped_missing": int(skipped_missing),
        "skipped_low_conf": int(skipped_low_conf),
        "accepted_uids": accepted_uids,
    }


def build_retrain_snapshot(
    dataset_items: list[DatasetImageItem],
    classes: list[LabelClass],
    annotations: dict[str, dict[int, np.ndarray]],
    pseudo_labels: dict[str, np.ndarray],
    review_state: dict[str, dict[str, object]],
    *,
    include_review_statuses: tuple[str, ...] = ("accepted", "refined", "auto-accepted", "synthetic"),
) -> tuple[list[DatasetImageItem], dict[str, dict[int, np.ndarray]], dict[str, object]]:
    allowed = {str(status).strip().lower() for status in include_review_statuses if str(status).strip()}
    snapshot_items: list[DatasetImageItem] = []
    snapshot_annotations: dict[str, dict[int, np.ndarray]] = {}
    manual_count = 0
    pseudo_count = 0
    synthetic_count = 0

    for item in dataset_items:
        uid = str(item.uid)
        state = review_state.get(uid, {})
        status = str(state.get("status", "") or "").strip().lower()
        layers = annotations.get(uid, {})
        manual_fraction = annotation_coverage_fraction(layers, item.image.shape[:2])

        if manual_fraction > 0.0:
            snapshot_items.append(
                DatasetImageItem(uid=item.uid, name=item.name, path=item.path, image=np.asarray(item.image, dtype=np.uint8).copy())
            )
            snapshot_annotations[uid] = {
                int(cls.class_id): np.asarray(layers.get(cls.class_id, np.zeros(item.image.shape[:2], dtype=np.uint8)), dtype=np.uint8).copy()
                for cls in classes
            }
            if status == "synthetic":
                synthetic_count += 1
            else:
                manual_count += 1
            continue

        if status not in allowed:
            continue
        pseudo = pseudo_labels.get(uid)
        if not isinstance(pseudo, np.ndarray):
            continue
        snapshot_items.append(
            DatasetImageItem(uid=item.uid, name=item.name, path=item.path, image=np.asarray(item.image, dtype=np.uint8).copy())
        )
        snapshot_annotations[uid] = _index_mask_to_layers(np.asarray(pseudo, dtype=np.uint8), classes, item.image.shape[:2])
        pseudo_count += 1

    summary = {
        "total_items": int(len(snapshot_items)),
        "manual_items": int(manual_count),
        "pseudo_items": int(pseudo_count),
        "synthetic_items": int(synthetic_count),
        "included_review_statuses": sorted(allowed),
        "uids": [str(item.uid) for item in snapshot_items],
    }
    return snapshot_items, snapshot_annotations, summary


def _apply_geom_image(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "hflip":
        return np.ascontiguousarray(image[:, ::-1])
    if mode == "vflip":
        return np.ascontiguousarray(image[::-1, :])
    if mode == "rot90":
        return np.ascontiguousarray(np.rot90(image, 1))
    if mode == "rot180":
        return np.ascontiguousarray(np.rot90(image, 2))
    if mode == "rot270":
        return np.ascontiguousarray(np.rot90(image, 3))
    return np.ascontiguousarray(image)


def _apply_geom_mask(mask: np.ndarray, mode: str) -> np.ndarray:
    return _apply_geom_image(mask, mode)


def generate_synthetic_variants(
    image_rgb: np.ndarray,
    layers: dict[int, np.ndarray],
    *,
    count: int,
    rng_seed: int | None = None,
) -> list[tuple[np.ndarray, dict[int, np.ndarray], dict[str, object]]]:
    variants: list[tuple[np.ndarray, dict[int, np.ndarray], dict[str, object]]] = []
    if int(count) <= 0:
        return variants

    rng = np.random.default_rng(rng_seed)
    geom_modes = ("identity", "hflip", "vflip", "rot90", "rot180", "rot270")

    for index in range(int(count)):
        geom_mode = geom_modes[int(rng.integers(0, len(geom_modes)))]
        brightness = float(rng.uniform(0.88, 1.18))
        contrast = float(rng.uniform(0.88, 1.22))
        gamma = float(rng.uniform(0.92, 1.10))
        noise_std = float(rng.uniform(0.0, 10.0))

        transformed = _apply_geom_image(np.asarray(image_rgb, dtype=np.uint8), geom_mode)
        pil_image = Image.fromarray(transformed.astype(np.uint8), mode="RGB")
        pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness)
        pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast)
        transformed = np.asarray(pil_image, dtype=np.float32) / 255.0
        transformed = np.power(np.clip(transformed, 0.0, 1.0), gamma)
        transformed = transformed * 255.0
        if noise_std > 0.0:
            transformed += rng.normal(0.0, noise_std, size=transformed.shape)
        transformed = np.clip(transformed, 0, 255).astype(np.uint8)

        transformed_layers: dict[int, np.ndarray] = {}
        for class_id, mask in layers.items():
            transformed_layers[int(class_id)] = np.asarray(_apply_geom_mask(np.asarray(mask, dtype=np.uint8), geom_mode), dtype=np.uint8)

        variants.append(
            (
                transformed,
                transformed_layers,
                {
                    "geom_mode": geom_mode,
                    "brightness": float(brightness),
                    "contrast": float(contrast),
                    "gamma": float(gamma),
                    "noise_std": float(noise_std),
                    "variant_index": int(index),
                },
            )
        )
    return variants


def default_synthetic_output_dir() -> Path:
    return Path.home() / "npec_hitl_synthetic"
