from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

from .mask_ops import index_mask_from_layers
from .models import DatasetImageItem, LabelClass


def export_dataset_zip(
    dataset_items: list[DatasetImageItem],
    classes: list[LabelClass],
    annotations: dict[str, dict[int, np.ndarray]],
    output_path: Path,
) -> None:
    metadata = {
        "classes": [
            {"id": cls.class_id, "name": cls.name, "color": cls.color_hex}
            for cls in classes
        ],
        "items": [],
    }

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in dataset_items:
            image = item.image
            h, w = image.shape[:2]
            layers = annotations.get(item.uid, {})
            index_mask = index_mask_from_layers(layers, classes, (h, w))
            stem = Path(item.name).stem

            indexed_buf = io.BytesIO()
            Image.fromarray(index_mask, mode="L").save(indexed_buf, format="PNG")
            zf.writestr(f"masks/indexed/{stem}.png", indexed_buf.getvalue())

            for cls in classes:
                class_mask = (layers.get(cls.class_id, np.zeros((h, w), dtype=np.uint8)) > 0).astype(np.uint8) * 255
                class_buf = io.BytesIO()
                Image.fromarray(class_mask, mode="L").save(class_buf, format="PNG")
                zf.writestr(f"masks/binary/{stem}/class_{cls.class_id}.png", class_buf.getvalue())

            metadata["items"].append(
                {
                    "uid": item.uid,
                    "image_name": item.name,
                    "source_path": str(item.path),
                    "shape": [h, w],
                    "indexed_mask": f"masks/indexed/{stem}.png",
                    "binary_masks": f"masks/binary/{stem}/",
                }
            )

        zf.writestr("metadata.json", json.dumps(metadata, indent=2))


def _downsample_stack_nearest(stack: np.ndarray) -> np.ndarray:
    arr = np.asarray(stack)
    if arr.ndim != 3:
        return arr
    # [t, y, x] -> [t, y/2, x/2]
    return arr[:, ::2, ::2]


def _ensure_hw(arr: np.ndarray, shape_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    data = np.asarray(arr)
    if data.ndim != 2:
        data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {tuple(data.shape)}")
    if data.shape != (h, w):
        data = cv2.resize(data, (w, h), interpolation=interpolation)
    return data


def _zarr_write_dataset(group, name: str, data: np.ndarray) -> None:
    chunks = (
        max(1, min(8, int(data.shape[0]))),
        max(1, min(512, int(data.shape[1]))),
        max(1, min(512, int(data.shape[2]))),
    )
    if hasattr(group, "create_dataset"):
        group.create_dataset(name, data=data, chunks=chunks, overwrite=True)
        return
    if hasattr(group, "array"):
        group.array(name, data=data, chunks=chunks)
        return
    raise RuntimeError("Unsupported zarr group API: cannot create dataset.")


def export_ome_zarr_predictions(
    dataset_items: list[DatasetImageItem],
    predictions: dict[str, np.ndarray],
    output_path: Path,
    *,
    probabilities: dict[str, np.ndarray] | None = None,
    include_segmentation: bool = True,
    include_probabilities: bool = False,
    foreground_class_id: int = 1,
    probability_threshold: float = 0.5,
    max_levels: int = 4,
) -> dict[str, object]:
    if not include_segmentation and not include_probabilities:
        raise ValueError("Select at least one export layer: segmentation and/or probabilities.")
    if not dataset_items:
        raise ValueError("No dataset items available for OME-Zarr export.")

    ordered_items: list[DatasetImageItem] = []
    for item in dataset_items:
        if include_segmentation and item.uid in predictions:
            ordered_items.append(item)
        elif include_probabilities and probabilities is not None and item.uid in probabilities:
            ordered_items.append(item)
    if not ordered_items:
        raise ValueError("No RF outputs found for the selected images. Run RF first.")

    base_h, base_w = ordered_items[0].image.shape[:2]
    seg_planes: list[np.ndarray] = []
    prob_planes: list[np.ndarray] = []
    frame_names: list[str] = []

    for item in ordered_items:
        frame_names.append(str(item.name))
        pred = predictions.get(item.uid)
        prob = probabilities.get(item.uid) if probabilities is not None else None

        if include_segmentation:
            if pred is None:
                seg = np.zeros((base_h, base_w), dtype=np.uint16)
            else:
                seg = _ensure_hw(np.asarray(pred, dtype=np.uint16), (base_h, base_w), interpolation=cv2.INTER_NEAREST).astype(np.uint16)
            seg_planes.append(seg)

        if include_probabilities:
            if prob is None:
                if pred is None:
                    prob_arr = np.zeros((base_h, base_w), dtype=np.float32)
                else:
                    pred_u8 = _ensure_hw(np.asarray(pred, dtype=np.uint8), (base_h, base_w), interpolation=cv2.INTER_NEAREST)
                    prob_arr = (pred_u8 == int(foreground_class_id)).astype(np.float32)
            else:
                prob_arr = _ensure_hw(np.asarray(prob, dtype=np.float32), (base_h, base_w), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                prob_arr = np.clip(prob_arr, 0.0, 1.0)
            prob_planes.append(prob_arr)

    seg_stack = np.stack(seg_planes, axis=0) if seg_planes else None
    prob_stack = np.stack(prob_planes, axis=0) if prob_planes else None

    try:
        import zarr  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "OME-Zarr export requires the 'zarr' package. Install it in your runtime: pip install zarr"
        ) from exc

    target = Path(output_path).expanduser()
    if str(target).lower().endswith(".ome.zarr"):
        pass
    elif target.suffix:
        target = target.with_suffix(".ome.zarr")
    else:
        target = Path(str(target) + ".ome.zarr")
    target.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(target), mode="w")
    root.attrs["creator"] = "NPEC Labeling Tool"
    root.attrs["format"] = "OME-Zarr-like multiscale export"
    root.attrs["foreground_class_id"] = int(foreground_class_id)
    root.attrs["probability_threshold"] = float(np.clip(probability_threshold, 0.0, 1.0))
    root.attrs["frame_names"] = frame_names

    levels = max(1, int(max_levels))

    if seg_stack is not None:
        seg_group = root.require_group("segmentation")
        datasets = []
        curr = np.asarray(seg_stack, dtype=np.uint16)
        for level in range(levels):
            name = str(level)
            _zarr_write_dataset(seg_group, name, curr)
            datasets.append({"path": name})
            if min(curr.shape[1], curr.shape[2]) < 16:
                break
            next_arr = _downsample_stack_nearest(curr)
            if next_arr.shape == curr.shape:
                break
            curr = next_arr
        seg_group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": "segmentation",
                "axes": [{"name": "t", "type": "time"}, {"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
                "datasets": datasets,
            }
        ]

    if prob_stack is not None:
        prob_group = root.require_group("probabilities")
        datasets = []
        curr_f = np.asarray(prob_stack, dtype=np.float32)
        for level in range(levels):
            name = str(level)
            _zarr_write_dataset(prob_group, name, curr_f)
            datasets.append({"path": name})
            if min(curr_f.shape[1], curr_f.shape[2]) < 16:
                break
            next_arr = _downsample_stack_nearest(curr_f)
            if next_arr.shape == curr_f.shape:
                break
            curr_f = next_arr
        prob_group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": "probabilities",
                "axes": [{"name": "t", "type": "time"}, {"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
                "datasets": datasets,
            }
        ]

    return {
        "output_path": str(target),
        "frames": int(len(frame_names)),
        "shape_hw": [int(base_h), int(base_w)],
        "segmentation_exported": bool(seg_stack is not None),
        "probabilities_exported": bool(prob_stack is not None),
    }
