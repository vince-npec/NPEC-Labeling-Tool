from __future__ import annotations

import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from PIL import Image

from .models import DatasetImageItem, LabelClass, normalize_color_hex


PROJECT_VERSION = 3
DEFAULT_SERIES_NAME = "Ungrouped"
DEFAULT_SERIES_FILTER = "__all_timeseries__"
DEFAULT_ROOT_CLASS_FALLBACK = (
    LabelClass(1, "Root", "#ffd166"),
    LabelClass(2, "Shoot", "#56f39a"),
    LabelClass(3, "Lateral Root", "#3fc1ff"),
)


def _resize_nearest(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask.astype(np.uint8)
    resized = Image.fromarray(mask.astype(np.uint8), mode="L").resize((w, h), Image.NEAREST)
    return np.array(resized, dtype=np.uint8)


def _serialize_npy(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def _deserialize_npy(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)


def _safe_member_name(value: str, fallback: str = "image") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text or fallback


def _serialize_image_payload(item: DatasetImageItem) -> tuple[str, bytes]:
    # Always serialize the current in-memory image so project saves preserve
    # app-side edits such as stabilization or rotation instead of silently
    # reverting to the original file bytes from disk.
    buffer = io.BytesIO()
    try:
        image_u8 = np.asarray(item.image, dtype=np.uint8)
    except (TypeError, ValueError):
        if item.path is None or not Path(item.path).exists():
            raise
        with Image.open(item.path) as source_image:
            image_u8 = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
    if image_u8.ndim == 2:
        Image.fromarray(image_u8, mode="L").save(buffer, format="PNG")
    else:
        Image.fromarray(image_u8, mode="RGB").save(buffer, format="PNG")
    fallback_name = Path(item.name).with_suffix(".png").name if str(item.name).strip() else f"{item.uid}.png"
    return fallback_name, buffer.getvalue()


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(entry) for entry in value]
    return str(value)


def _positive_int_or_none(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _path_tail(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    normalized = raw.replace("\\", "/").rstrip("/")
    if not normalized:
        return ""
    return normalized.split("/")[-1]


def _resolve_dataset_image_path(raw_path: str, project_base_dir: Path, fallback_name: str) -> Path | None:
    raw = str(raw_path or "").strip()
    fallback = _path_tail(fallback_name)
    raw_basename = _path_tail(raw)
    candidates: list[Path] = []
    if raw:
        path_obj = Path(raw)
        candidates.append(path_obj)
        if not path_obj.is_absolute():
            candidates.append(project_base_dir / path_obj)
        else:
            parts = [part for part in raw.replace("\\", "/").split("/") if part]
            parts_lower = [part.lower() for part in parts]
            for marker in ("images", "layers_original", "masks_binary", "masks_instances"):
                if marker in parts_lower:
                    idx = parts_lower.index(marker)
                    if idx < len(parts):
                        candidates.append(project_base_dir / Path(*parts[idx:]))
            if raw_basename:
                candidates.append(project_base_dir / raw_basename)
                candidates.append(project_base_dir / "images" / raw_basename)
            candidates.append(project_base_dir / path_obj.name)
    if fallback:
        candidates.append(project_base_dir / "images" / fallback)
        candidates.append(project_base_dir / fallback)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate.exists():
                return candidate.resolve()
        except Exception:
            continue

    # Last-resort lookup by filename inside the project root.
    search_names: list[str] = []
    if fallback:
        search_names.append(fallback)
    if raw_basename and raw_basename not in search_names:
        search_names.append(raw_basename)
    images_dir = project_base_dir / "images"
    for filename in search_names:
        if images_dir.exists():
            matches = list(images_dir.rglob(filename))
            if len(matches) == 1:
                return matches[0].resolve()
        matches = list(project_base_dir.rglob(filename))
        if len(matches) == 1:
            return matches[0].resolve()
    return None


def save_project(
    project_path: Path,
    dataset_items: list[DatasetImageItem],
    classes: list[LabelClass],
    annotations: dict[str, dict[int, np.ndarray]],
    predictions: dict[str, np.ndarray],
    current_index: int,
    active_class_id: int,
    patch_target_class_id: int,
    next_class_id: int,
    model_path: Path | None,
    inference_backend: str = "standard",
    class_visibility: dict[int, bool] | None = None,
    external_tf_python_path: Path | None = None,
    hailo_hef_path: Path | None = None,
    hailo_infer_command: str | None = None,
    training_hailo_compile_command: str | None = None,
    training_hef_output_path: Path | None = None,
    series_by_uid: dict[str, str] | None = None,
    series_order: list[str] | None = None,
    series_filter_key: str | None = None,
    foundation_external_python_path: Path | None = None,
    foundation_external_runner_path: Path | None = None,
    foundation_external_checkpoint_path: Path | None = None,
    foundation_external_extra_args: str | None = None,
    foundation_text_prompt: str | None = None,
    foundation_interactions: dict[str, list[dict[str, object]]] | None = None,
    foundation_run_history: dict[str, list[dict[str, object]]] | None = None,
    exclusive_class_paint: bool = True,
    timeline_show_thumbnails: bool = True,
    store_relative_paths: bool = False,
    pyphenotyper_preset_key: str | None = None,
    pyphenotyper_pipeline_dir: Path | None = None,
    pyphenotyper_root_model_path: Path | None = None,
    pyphenotyper_shoot_model_path: Path | None = None,
    pyphenotyper_lucifer_gan_gap_repair_enabled: bool | None = None,
    pyphenotyper_lucifer_gan_gap_repair_dir: Path | None = None,
    pyphenotyper_preserve_root_on_shoot_overlap: bool | None = None,
    pyphenotyper_shoot_color_rescue_mode: str | None = None,
    pyphenotyper_pixel_size_mm: float | None = None,
    analytics_pixel_size_mm: float | None = None,
    analytics_root_class_id: int | None = None,
    analytics_lateral_class_id: int | None = None,
    analytics_seed_class_id: int | None = None,
    analytics_shoot_class_id: int | None = None,
    analytics_root_ownership_mode: str | None = None,
    analytics_freeze_ambiguous_ownership: bool | None = None,
    pipeline_temporal_primary_lock_enabled: bool | None = None,
    pipeline_lazy_analysis_mode: str | None = None,
    pipeline_lazy_standard_expected_plants: int | None = None,
    pipeline_lazy_measure_class_ids: str | None = None,
    pipeline_lazy_shoot_class_ids: str | None = None,
    pipeline_lazy_video_mask_only: bool | None = None,
    pipeline_lazy_root_metric_mode: str | None = None,
    pipeline_lazy_shoot_rgb_rescue_enabled: bool | None = None,
    pipeline_lazy_plate_identity_enabled: bool | None = None,
    pipeline_stabilization_target_mode: str | None = None,
    pyphenotyper_general_starter_calibration: dict[str, object] | None = None,
    prediction_metadata_by_uid: dict[str, dict[str, object]] | None = None,
    image_identity_by_uid: dict[str, dict[str, object]] | None = None,
    hitl_pseudo_labels: dict[str, np.ndarray] | None = None,
    hitl_review_state: dict[str, dict[str, object]] | None = None,
    hitl_session_input_folder: Path | None = None,
    hitl_session_template_preset_key: str | None = None,
    hitl_synthetic_output_dir: Path | None = None,
    hitl_long_session_cycle_count: int | None = None,
) -> None:
    safe_series_by_uid = dict(series_by_uid or {})
    safe_series_order = [str(name).strip() for name in (series_order or []) if str(name).strip()]
    safe_visibility: dict[str, bool] = {}
    for cls in classes:
        safe_visibility[str(cls.class_id)] = bool((class_visibility or {}).get(cls.class_id, True))
    metadata = {
        "version": PROJECT_VERSION,
        "stores_embedded_images": True,
        "classes": [{"id": cls.class_id, "name": cls.name, "color": normalize_color_hex(cls.color_hex)} for cls in classes],
        "class_visibility": safe_visibility,
        "dataset": [],
        "current_index": int(current_index),
        "active_class_id": int(active_class_id),
        "patch_target_class_id": int(patch_target_class_id),
        "next_class_id": int(next_class_id),
        "model_path": str(model_path) if model_path is not None else None,
        "inference_backend": str(inference_backend or "standard"),
        "external_tf_python_path": str(external_tf_python_path) if external_tf_python_path is not None else None,
        "hailo_hef_path": str(hailo_hef_path) if hailo_hef_path is not None else None,
        "hailo_infer_command": str(hailo_infer_command or ""),
        "training_hailo_compile_command": str(training_hailo_compile_command or ""),
        "training_hef_output_path": str(training_hef_output_path) if training_hef_output_path is not None else None,
        "foundation_external_python_path": str(foundation_external_python_path) if foundation_external_python_path is not None else None,
        "foundation_external_runner_path": str(foundation_external_runner_path) if foundation_external_runner_path is not None else None,
        "foundation_external_checkpoint_path": str(foundation_external_checkpoint_path) if foundation_external_checkpoint_path is not None else None,
        "foundation_external_extra_args": str(foundation_external_extra_args or ""),
        "foundation_text_prompt": str(foundation_text_prompt or ""),
        "foundation_interactions": foundation_interactions or {},
        "foundation_run_history": foundation_run_history or {},
        "series_by_uid": safe_series_by_uid,
        "series_order": safe_series_order,
        "series_filter_key": str(series_filter_key or DEFAULT_SERIES_FILTER),
        "exclusive_class_paint": bool(exclusive_class_paint),
        "timeline_show_thumbnails": bool(timeline_show_thumbnails),
        "pyphenotyper_preset_key": str(pyphenotyper_preset_key or "custom"),
        "pyphenotyper_pipeline_dir": str(pyphenotyper_pipeline_dir) if pyphenotyper_pipeline_dir is not None else None,
        "pyphenotyper_root_model_path": (
            str(pyphenotyper_root_model_path) if pyphenotyper_root_model_path is not None else None
        ),
        "pyphenotyper_shoot_model_path": (
            str(pyphenotyper_shoot_model_path) if pyphenotyper_shoot_model_path is not None else None
        ),
        "pyphenotyper_lucifer_gan_gap_repair_enabled": bool(pyphenotyper_lucifer_gan_gap_repair_enabled),
        "pyphenotyper_lucifer_gan_gap_repair_dir": (
            str(pyphenotyper_lucifer_gan_gap_repair_dir) if pyphenotyper_lucifer_gan_gap_repair_dir is not None else None
        ),
        "pyphenotyper_preserve_root_on_shoot_overlap": bool(
            True if pyphenotyper_preserve_root_on_shoot_overlap is None else pyphenotyper_preserve_root_on_shoot_overlap
        ),
        "pyphenotyper_shoot_color_rescue_mode": str(pyphenotyper_shoot_color_rescue_mode or "auto"),
        "pyphenotyper_pixel_size_mm": (
            float(pyphenotyper_pixel_size_mm) if pyphenotyper_pixel_size_mm is not None else None
        ),
        "analytics_pixel_size_mm": (
            float(analytics_pixel_size_mm) if analytics_pixel_size_mm is not None else None
        ),
        "analytics_root_class_id": _positive_int_or_none(analytics_root_class_id),
        "analytics_lateral_class_id": _positive_int_or_none(analytics_lateral_class_id),
        "analytics_seed_class_id": _positive_int_or_none(analytics_seed_class_id),
        "analytics_shoot_class_id": _positive_int_or_none(analytics_shoot_class_id),
        "analytics_root_ownership_mode": str(analytics_root_ownership_mode or "temporal_graph"),
        "analytics_freeze_ambiguous_ownership": bool(
            True if analytics_freeze_ambiguous_ownership is None else analytics_freeze_ambiguous_ownership
        ),
        "pipeline_temporal_primary_lock_enabled": bool(
            True if pipeline_temporal_primary_lock_enabled is None else pipeline_temporal_primary_lock_enabled
        ),
        "pipeline_lazy_analysis_mode": str(pipeline_lazy_analysis_mode or "combined"),
        "pipeline_lazy_standard_expected_plants": max(
            1,
            min(128, int(pipeline_lazy_standard_expected_plants or 5)),
        ),
        "pipeline_lazy_measure_class_ids": str(pipeline_lazy_measure_class_ids or "1,3"),
        "pipeline_lazy_shoot_class_ids": str(pipeline_lazy_shoot_class_ids or "2"),
        "pipeline_lazy_video_mask_only": bool(False if pipeline_lazy_video_mask_only is None else pipeline_lazy_video_mask_only),
        "pipeline_lazy_root_metric_mode": str(pipeline_lazy_root_metric_mode or "total"),
        "pipeline_lazy_shoot_rgb_rescue_enabled": bool(
            False if pipeline_lazy_shoot_rgb_rescue_enabled is None else pipeline_lazy_shoot_rgb_rescue_enabled
        ),
        "pipeline_lazy_plate_identity_enabled": bool(
            True if pipeline_lazy_plate_identity_enabled is None else pipeline_lazy_plate_identity_enabled
        ),
        "pipeline_stabilization_target_mode": str(pipeline_stabilization_target_mode or "plant_top"),
        "pyphenotyper_general_starter_calibration": (
            pyphenotyper_general_starter_calibration if isinstance(pyphenotyper_general_starter_calibration, dict) else None
        ),
        "prediction_metadata_by_uid": _json_safe(prediction_metadata_by_uid or {}),
        "image_identity_by_uid": _json_safe(image_identity_by_uid or {}),
        "hitl_review_state": _json_safe(hitl_review_state or {}),
        "hitl_session_input_folder": str(hitl_session_input_folder) if hitl_session_input_folder is not None else None,
        "hitl_session_template_preset_key": str(hitl_session_template_preset_key or "current"),
        "hitl_synthetic_output_dir": str(hitl_synthetic_output_dir) if hitl_synthetic_output_dir is not None else None,
        "hitl_long_session_cycle_count": (
            int(hitl_long_session_cycle_count) if hitl_long_session_cycle_count not in (None, "") else 1
        ),
    }

    with zipfile.ZipFile(project_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in dataset_items:
            item_path = Path(item.path)
            item_path_str = str(item_path)
            if store_relative_paths:
                try:
                    item_path_str = str(item_path.resolve().relative_to(project_path.parent.resolve()))
                except Exception:
                    item_path_str = str(item_path)
            image_shape = list(item.image.shape[:2])
            entry = {
                "uid": item.uid,
                "name": item.name,
                "path": item_path_str,
                "shape": image_shape,
            }
            bundled_name, bundled_payload = _serialize_image_payload(item)
            asset_member = f"images/{item.uid}/{_safe_member_name(bundled_name, fallback=f'{item.uid}.png')}"
            entry["embedded_image_member"] = asset_member
            metadata["dataset"].append(entry)
            zf.writestr(asset_member, bundled_payload)

            image_layers = annotations.get(item.uid, {})
            for cls in classes:
                class_id = cls.class_id
                layer = image_layers.get(class_id)
                if layer is None:
                    continue
                zf.writestr(f"masks/{item.uid}/class_{class_id}.npy", _serialize_npy(layer.astype(np.uint8)))

            pred = predictions.get(item.uid)
            if pred is not None:
                zf.writestr(f"predictions/{item.uid}.npy", _serialize_npy(pred.astype(np.uint8)))
            pseudo = (hitl_pseudo_labels or {}).get(item.uid)
            if pseudo is not None:
                zf.writestr(f"pseudo_labels/{item.uid}.npy", _serialize_npy(np.asarray(pseudo, dtype=np.uint8)))

        zf.writestr("project.json", json.dumps(metadata, indent=2))


def load_project(project_path: Path) -> dict:
    with zipfile.ZipFile(project_path, "r") as zf:
        if "project.json" not in zf.namelist():
            raise ValueError("Invalid project file: missing project.json.")
        metadata = json.loads(zf.read("project.json").decode("utf-8"))
        zip_members = set(zf.namelist())

        class_entries = metadata.get("classes", [])
        classes: list[LabelClass] = []
        for raw in class_entries:
            class_id = int(raw.get("id", 0))
            if class_id <= 0 or class_id > 255:
                continue
            classes.append(
                LabelClass(
                    class_id=class_id,
                    name=str(raw.get("name", f"Class {class_id}")),
                    color_hex=normalize_color_hex(str(raw.get("color", "#ff5a5f"))),
                )
            )
        if not classes:
            classes = [LabelClass(cls.class_id, cls.name, cls.color_hex) for cls in DEFAULT_ROOT_CLASS_FALLBACK]

        class_ids = [cls.class_id for cls in classes]
        raw_class_visibility = metadata.get("class_visibility", {})
        class_visibility: dict[int, bool] = {}
        if isinstance(raw_class_visibility, dict):
            for key, value in raw_class_visibility.items():
                try:
                    class_id = int(key)
                except Exception:
                    continue
                if class_id in class_ids:
                    class_visibility[class_id] = bool(value)
        for class_id in class_ids:
            class_visibility.setdefault(class_id, True)

        dataset_items: list[DatasetImageItem] = []
        annotations: dict[str, dict[int, np.ndarray]] = {}
        predictions: dict[str, np.ndarray] = {}
        hitl_pseudo_labels: dict[str, np.ndarray] = {}
        missing_paths: list[str] = []

        for raw_item in metadata.get("dataset", []):
            uid = str(raw_item.get("uid", "")).strip()
            raw_path = str(raw_item.get("path", ""))
            name_hint = Path(raw_path).name
            name = str(raw_item.get("name", name_hint if name_hint else "image"))
            embedded_member = str(raw_item.get("embedded_image_member", "") or "").strip()
            if not uid:
                continue
            try:
                image_path = None
                if embedded_member and embedded_member in zip_members:
                    with Image.open(io.BytesIO(zf.read(embedded_member))) as pil:
                        image = np.array(pil.convert("RGB"), dtype=np.uint8)
                    image_path = project_path.parent / "__embedded_project_assets__" / uid / Path(name).name
                else:
                    path = _resolve_dataset_image_path(raw_path, project_path.parent, name)
                    if path is None or not path.exists():
                        missing_paths.append(raw_path if raw_path else name)
                        continue
                    with Image.open(path) as pil:
                        image = np.array(pil.convert("RGB"), dtype=np.uint8)
                    image_path = path
            except Exception:
                missing_paths.append(raw_path if raw_path else name)
                continue

            item = DatasetImageItem(uid=uid, name=name, path=image_path, image=image)
            dataset_items.append(item)

            h, w = image.shape[:2]
            per_image: dict[int, np.ndarray] = {}
            for cls in classes:
                mask_member = f"masks/{uid}/class_{cls.class_id}.npy"
                if mask_member in zip_members:
                    raw_mask = _deserialize_npy(zf.read(mask_member))
                    mask = _resize_nearest(raw_mask, (h, w))
                    per_image[cls.class_id] = (mask > 0).astype(np.uint8)
                else:
                    per_image[cls.class_id] = np.zeros((h, w), dtype=np.uint8)
            annotations[uid] = per_image

            pred_member = f"predictions/{uid}.npy"
            if pred_member in zip_members:
                raw_pred = _deserialize_npy(zf.read(pred_member))
                pred = _resize_nearest(raw_pred, (h, w))
                predictions[uid] = pred.astype(np.uint8)

            pseudo_member = f"pseudo_labels/{uid}.npy"
            if pseudo_member in zip_members:
                raw_pseudo = _deserialize_npy(zf.read(pseudo_member))
                pseudo = _resize_nearest(raw_pseudo, (h, w))
                hitl_pseudo_labels[uid] = pseudo.astype(np.uint8)

    active_class_id = int(metadata.get("active_class_id", class_ids[0]))
    patch_target_class_id = int(metadata.get("patch_target_class_id", active_class_id))
    if active_class_id not in class_ids:
        active_class_id = class_ids[0]
    if patch_target_class_id not in class_ids:
        patch_target_class_id = active_class_id

    current_index = int(metadata.get("current_index", 0))
    current_index = max(0, min(current_index, max(0, len(dataset_items) - 1)))

    model_path_raw = metadata.get("model_path")
    model_path = Path(model_path_raw) if model_path_raw else None
    inference_backend = str(metadata.get("inference_backend", "standard") or "standard")
    external_tf_python_raw = metadata.get("external_tf_python_path")
    external_tf_python_path = Path(external_tf_python_raw) if external_tf_python_raw else None
    has_external_tf_python_path = "external_tf_python_path" in metadata
    hailo_hef_path_raw = metadata.get("hailo_hef_path")
    hailo_hef_path = Path(hailo_hef_path_raw) if hailo_hef_path_raw else None
    hailo_infer_command = str(metadata.get("hailo_infer_command", "") or "")
    training_hailo_compile_command = str(metadata.get("training_hailo_compile_command", "") or "")
    training_hef_output_path_raw = metadata.get("training_hef_output_path")
    training_hef_output_path = Path(training_hef_output_path_raw) if training_hef_output_path_raw else None
    pyphenotyper_preset_key = str(metadata.get("pyphenotyper_preset_key", "custom") or "custom")
    pyphenotyper_pipeline_dir_raw = metadata.get("pyphenotyper_pipeline_dir")
    pyphenotyper_pipeline_dir = Path(pyphenotyper_pipeline_dir_raw) if pyphenotyper_pipeline_dir_raw else None
    pyphenotyper_root_model_raw = metadata.get("pyphenotyper_root_model_path")
    pyphenotyper_root_model_path = Path(pyphenotyper_root_model_raw) if pyphenotyper_root_model_raw else None
    pyphenotyper_shoot_model_raw = metadata.get("pyphenotyper_shoot_model_path")
    pyphenotyper_shoot_model_path = Path(pyphenotyper_shoot_model_raw) if pyphenotyper_shoot_model_raw else None
    pyphenotyper_lucifer_gan_gap_repair_enabled = bool(metadata.get("pyphenotyper_lucifer_gan_gap_repair_enabled", False))
    pyphenotyper_lucifer_gan_gap_repair_dir_raw = metadata.get("pyphenotyper_lucifer_gan_gap_repair_dir")
    pyphenotyper_lucifer_gan_gap_repair_dir = (
        Path(pyphenotyper_lucifer_gan_gap_repair_dir_raw) if pyphenotyper_lucifer_gan_gap_repair_dir_raw else None
    )
    pyphenotyper_preserve_root_on_shoot_overlap = bool(
        metadata.get("pyphenotyper_preserve_root_on_shoot_overlap", True)
    )
    pyphenotyper_shoot_color_rescue_mode = str(metadata.get("pyphenotyper_shoot_color_rescue_mode", "auto") or "auto")
    if pyphenotyper_shoot_color_rescue_mode not in {"auto", "lucifer_green", "off"}:
        pyphenotyper_shoot_color_rescue_mode = "auto"
    pyphenotyper_pixel_size_mm_raw = metadata.get("pyphenotyper_pixel_size_mm")
    pyphenotyper_pixel_size_mm = float(pyphenotyper_pixel_size_mm_raw) if pyphenotyper_pixel_size_mm_raw not in (None, "") else None
    analytics_pixel_size_mm_raw = metadata.get("analytics_pixel_size_mm")
    analytics_pixel_size_mm = float(analytics_pixel_size_mm_raw) if analytics_pixel_size_mm_raw not in (None, "") else None
    analytics_root_ownership_mode = str(metadata.get("analytics_root_ownership_mode", "temporal_graph") or "temporal_graph")
    if analytics_root_ownership_mode not in {"temporal_graph", "euclidean"}:
        analytics_root_ownership_mode = "temporal_graph"
    analytics_freeze_ambiguous_ownership = bool(metadata.get("analytics_freeze_ambiguous_ownership", True))
    pipeline_temporal_primary_lock_enabled = bool(metadata.get("pipeline_temporal_primary_lock_enabled", True))
    valid_class_ids = set(int(class_id) for class_id in class_ids)
    analytics_class_settings_present = any(
        key in metadata
        for key in (
            "analytics_root_class_id",
            "analytics_lateral_class_id",
            "analytics_seed_class_id",
            "analytics_shoot_class_id",
        )
    )
    analytics_root_class_id = _positive_int_or_none(metadata.get("analytics_root_class_id"))
    analytics_lateral_class_id = _positive_int_or_none(metadata.get("analytics_lateral_class_id"))
    analytics_seed_class_id = _positive_int_or_none(metadata.get("analytics_seed_class_id"))
    analytics_shoot_class_id = _positive_int_or_none(metadata.get("analytics_shoot_class_id"))
    if analytics_root_class_id not in valid_class_ids:
        analytics_root_class_id = None
    if analytics_lateral_class_id not in valid_class_ids:
        analytics_lateral_class_id = None
    if analytics_seed_class_id not in valid_class_ids:
        analytics_seed_class_id = None
    if analytics_shoot_class_id not in valid_class_ids:
        analytics_shoot_class_id = None
    pipeline_lazy_analysis_mode = str(metadata.get("pipeline_lazy_analysis_mode", "combined") or "combined")
    if pipeline_lazy_analysis_mode not in {"combined", "mask_total", "ownership", "potato_endpoint"}:
        pipeline_lazy_analysis_mode = "combined"
    try:
        pipeline_lazy_standard_expected_plants = max(
            1,
            min(128, int(metadata.get("pipeline_lazy_standard_expected_plants", 5) or 5)),
        )
    except (TypeError, ValueError):
        pipeline_lazy_standard_expected_plants = 5
    pipeline_lazy_measure_class_ids = str(metadata.get("pipeline_lazy_measure_class_ids", "1,3") or "1,3")
    pipeline_lazy_shoot_class_ids = str(metadata.get("pipeline_lazy_shoot_class_ids", "2") or "2")
    pipeline_lazy_video_mask_only = bool(metadata.get("pipeline_lazy_video_mask_only", False))
    pipeline_lazy_root_metric_mode = str(metadata.get("pipeline_lazy_root_metric_mode", "total") or "total")
    if pipeline_lazy_root_metric_mode not in {"total", "weighted_total", "primary", "lateral"}:
        pipeline_lazy_root_metric_mode = "total"
    pipeline_lazy_shoot_rgb_rescue_enabled = bool(metadata.get("pipeline_lazy_shoot_rgb_rescue_enabled", False))
    pipeline_lazy_plate_identity_enabled = bool(metadata.get("pipeline_lazy_plate_identity_enabled", True))
    pipeline_stabilization_target_mode = str(metadata.get("pipeline_stabilization_target_mode", "plant_top") or "plant_top")
    if pipeline_stabilization_target_mode not in {"plant_top", "dish_frame"}:
        pipeline_stabilization_target_mode = "plant_top"
    pyphenotyper_general_starter_calibration = metadata.get("pyphenotyper_general_starter_calibration")
    if not isinstance(pyphenotyper_general_starter_calibration, dict):
        pyphenotyper_general_starter_calibration = None
    prediction_metadata_by_uid = metadata.get("prediction_metadata_by_uid", {})
    if not isinstance(prediction_metadata_by_uid, dict):
        prediction_metadata_by_uid = {}
    else:
        prediction_metadata_by_uid = {
            str(key): value
            for key, value in prediction_metadata_by_uid.items()
            if isinstance(value, dict)
        }
    image_identity_by_uid = metadata.get("image_identity_by_uid", {})
    if not isinstance(image_identity_by_uid, dict):
        image_identity_by_uid = {}
    else:
        image_identity_by_uid = {
            str(key): value
            for key, value in image_identity_by_uid.items()
            if isinstance(value, dict)
        }
    hitl_review_state = metadata.get("hitl_review_state", {})
    if not isinstance(hitl_review_state, dict):
        hitl_review_state = {}
    else:
        hitl_review_state = {
            str(key): value
            for key, value in hitl_review_state.items()
            if isinstance(value, dict)
        }
    hitl_session_input_folder_raw = metadata.get("hitl_session_input_folder")
    hitl_session_input_folder = Path(hitl_session_input_folder_raw) if hitl_session_input_folder_raw else None
    hitl_session_template_preset_key = str(metadata.get("hitl_session_template_preset_key", "current") or "current")
    hitl_synthetic_output_dir_raw = metadata.get("hitl_synthetic_output_dir")
    hitl_synthetic_output_dir = Path(hitl_synthetic_output_dir_raw) if hitl_synthetic_output_dir_raw else None
    hitl_long_session_cycle_count_raw = metadata.get("hitl_long_session_cycle_count", 1)
    try:
        hitl_long_session_cycle_count = max(1, int(hitl_long_session_cycle_count_raw))
    except Exception:
        hitl_long_session_cycle_count = 1
    foundation_external_python_raw = metadata.get("foundation_external_python_path")
    foundation_external_python_path = Path(foundation_external_python_raw) if foundation_external_python_raw else None
    foundation_external_runner_raw = metadata.get("foundation_external_runner_path")
    foundation_external_runner_path = Path(foundation_external_runner_raw) if foundation_external_runner_raw else None
    foundation_external_checkpoint_raw = metadata.get("foundation_external_checkpoint_path")
    foundation_external_checkpoint_path = Path(foundation_external_checkpoint_raw) if foundation_external_checkpoint_raw else None
    foundation_external_extra_args = str(metadata.get("foundation_external_extra_args", "") or "")
    foundation_text_prompt = str(metadata.get("foundation_text_prompt", "") or "")
    foundation_interactions = metadata.get("foundation_interactions", {})
    if not isinstance(foundation_interactions, dict):
        foundation_interactions = {}
    foundation_run_history = metadata.get("foundation_run_history", {})
    if not isinstance(foundation_run_history, dict):
        foundation_run_history = {}

    raw_series_by_uid = metadata.get("series_by_uid", {})
    series_by_uid: dict[str, str] = {}
    if isinstance(raw_series_by_uid, dict):
        for key, value in raw_series_by_uid.items():
            uid = str(key).strip()
            name = str(value).strip() if value is not None else ""
            if not uid:
                continue
            series_by_uid[uid] = name if name else DEFAULT_SERIES_NAME

    raw_series_order = metadata.get("series_order", [])
    series_order: list[str] = []
    if isinstance(raw_series_order, list):
        for value in raw_series_order:
            name = str(value).strip()
            if name and name not in series_order:
                series_order.append(name)

    for item in dataset_items:
        if item.uid not in series_by_uid:
            series_by_uid[item.uid] = DEFAULT_SERIES_NAME
        if series_by_uid[item.uid] not in series_order:
            series_order.append(series_by_uid[item.uid])

    series_filter_key = str(metadata.get("series_filter_key", DEFAULT_SERIES_FILTER))
    if not series_filter_key:
        series_filter_key = DEFAULT_SERIES_FILTER
    exclusive_class_paint = bool(metadata.get("exclusive_class_paint", True))
    timeline_show_thumbnails = bool(metadata.get("timeline_show_thumbnails", True))

    return {
        "classes": classes,
        "dataset_items": dataset_items,
        "annotations": annotations,
        "predictions": predictions,
        "current_index": current_index,
        "active_class_id": active_class_id,
        "patch_target_class_id": patch_target_class_id,
        "next_class_id": int(metadata.get("next_class_id", max(class_ids) + 1)),
        "class_visibility": class_visibility,
        "model_path": model_path if model_path and model_path.exists() else None,
        "inference_backend": inference_backend,
        "external_tf_python_path": external_tf_python_path,
        "has_external_tf_python_path": has_external_tf_python_path,
        "hailo_hef_path": hailo_hef_path if hailo_hef_path and hailo_hef_path.exists() else hailo_hef_path,
        "hailo_infer_command": hailo_infer_command,
        "training_hailo_compile_command": training_hailo_compile_command,
        "training_hef_output_path": training_hef_output_path,
        "pyphenotyper_preset_key": pyphenotyper_preset_key,
        "pyphenotyper_pipeline_dir": pyphenotyper_pipeline_dir,
        "pyphenotyper_root_model_path": pyphenotyper_root_model_path,
        "pyphenotyper_shoot_model_path": pyphenotyper_shoot_model_path,
        "pyphenotyper_lucifer_gan_gap_repair_enabled": pyphenotyper_lucifer_gan_gap_repair_enabled,
        "pyphenotyper_lucifer_gan_gap_repair_dir": pyphenotyper_lucifer_gan_gap_repair_dir,
        "pyphenotyper_preserve_root_on_shoot_overlap": pyphenotyper_preserve_root_on_shoot_overlap,
        "pyphenotyper_shoot_color_rescue_mode": pyphenotyper_shoot_color_rescue_mode,
        "pyphenotyper_pixel_size_mm": pyphenotyper_pixel_size_mm,
        "analytics_pixel_size_mm": analytics_pixel_size_mm,
        "analytics_root_class_id": analytics_root_class_id,
        "analytics_lateral_class_id": analytics_lateral_class_id,
        "analytics_seed_class_id": analytics_seed_class_id,
        "analytics_shoot_class_id": analytics_shoot_class_id,
        "analytics_root_ownership_mode": analytics_root_ownership_mode,
        "analytics_freeze_ambiguous_ownership": analytics_freeze_ambiguous_ownership,
        "pipeline_temporal_primary_lock_enabled": pipeline_temporal_primary_lock_enabled,
        "analytics_class_settings_present": analytics_class_settings_present,
        "pipeline_lazy_analysis_mode": pipeline_lazy_analysis_mode,
        "pipeline_lazy_standard_expected_plants": pipeline_lazy_standard_expected_plants,
        "pipeline_lazy_measure_class_ids": pipeline_lazy_measure_class_ids,
        "pipeline_lazy_shoot_class_ids": pipeline_lazy_shoot_class_ids,
        "pipeline_lazy_video_mask_only": pipeline_lazy_video_mask_only,
        "pipeline_lazy_root_metric_mode": pipeline_lazy_root_metric_mode,
        "pipeline_lazy_shoot_rgb_rescue_enabled": pipeline_lazy_shoot_rgb_rescue_enabled,
        "pipeline_lazy_plate_identity_enabled": pipeline_lazy_plate_identity_enabled,
        "pipeline_stabilization_target_mode": pipeline_stabilization_target_mode,
        "pyphenotyper_general_starter_calibration": pyphenotyper_general_starter_calibration,
        "prediction_metadata_by_uid": prediction_metadata_by_uid,
        "image_identity_by_uid": image_identity_by_uid,
        "hitl_pseudo_labels": hitl_pseudo_labels,
        "hitl_review_state": hitl_review_state,
        "hitl_session_input_folder": hitl_session_input_folder,
        "hitl_session_template_preset_key": hitl_session_template_preset_key,
        "hitl_synthetic_output_dir": hitl_synthetic_output_dir,
        "hitl_long_session_cycle_count": hitl_long_session_cycle_count,
        "foundation_external_python_path": foundation_external_python_path,
        "foundation_external_runner_path": foundation_external_runner_path,
        "foundation_external_checkpoint_path": foundation_external_checkpoint_path,
        "foundation_external_extra_args": foundation_external_extra_args,
        "foundation_text_prompt": foundation_text_prompt,
        "foundation_interactions": foundation_interactions,
        "foundation_run_history": foundation_run_history,
        "series_by_uid": series_by_uid,
        "series_order": series_order,
        "series_filter_key": series_filter_key,
        "exclusive_class_paint": exclusive_class_paint,
        "timeline_show_thumbnails": timeline_show_thumbnails,
        "missing_paths": missing_paths,
    }
