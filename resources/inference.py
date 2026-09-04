from __future__ import annotations

from functools import lru_cache
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np
from PIL import Image


def _get_interpreter_class():
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter
    except Exception:
        pass

    try:
        from tensorflow.lite import Interpreter

        return Interpreter
    except Exception:
        return None


def _dequantize(values: np.ndarray, details: dict) -> np.ndarray:
    scale, zero = details.get("quantization", (0.0, 0))
    if scale is not None and scale > 0:
        return scale * (values.astype(np.float32) - float(zero))
    return values.astype(np.float32)


def _stable_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=axis, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return exp / denom


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -32.0, 32.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _prediction_uncertainty_summary(raw: np.ndarray, output_details: dict | None = None) -> dict[str, float | int | str]:
    quant = output_details if output_details is not None else {"quantization": (0.0, 0)}
    summary: dict[str, float | int | str] = {}

    if raw.ndim == 4:
        out = raw[0]
        if out.shape[0] in (1, 2, 3, 4, 5, 6, 7, 8) and out.shape[-1] not in (1, 2, 3, 4, 5, 6, 7, 8):
            out = np.transpose(out, (1, 2, 0))
        if out.ndim == 3 and out.shape[-1] > 1:
            vals = _dequantize(out, quant)
            if vals.min() >= 0.0 and vals.max() <= 1.0 and np.allclose(
                np.sum(vals, axis=-1),
                1.0,
                atol=0.15,
            ):
                probs = vals.astype(np.float32)
            else:
                probs = _stable_softmax(vals.astype(np.float32), axis=-1)
            top1 = np.max(probs, axis=-1)
            sorted_probs = np.sort(probs, axis=-1)
            top2 = sorted_probs[:, :, -2] if probs.shape[-1] > 1 else np.zeros_like(top1)
            summary["confidence_score"] = float(np.mean(top1))
            summary["margin_score"] = float(np.mean(top1 - top2))
            summary["uncertainty_score"] = float(1.0 - np.mean(top1 - top2))
            summary["probability_mode"] = "multiclass"
            return summary
        if out.ndim == 3 and out.shape[-1] == 1:
            vals = _dequantize(out[:, :, 0], quant).astype(np.float32)
            probs = vals if vals.min() >= 0.0 and vals.max() <= 1.0 else _sigmoid(vals)
            confidence = np.abs(probs - 0.5) * 2.0
            summary["confidence_score"] = float(np.mean(confidence))
            summary["uncertainty_score"] = float(1.0 - np.mean(confidence))
            summary["probability_mode"] = "binary"
            return summary
        vals = _dequantize(out.squeeze(), quant).astype(np.float32)
        probs = vals if vals.min() >= 0.0 and vals.max() <= 1.0 else _sigmoid(vals)
        confidence = np.abs(probs - 0.5) * 2.0
        summary["confidence_score"] = float(np.mean(confidence))
        summary["uncertainty_score"] = float(1.0 - np.mean(confidence))
        summary["probability_mode"] = "binary"
        return summary

    if raw.ndim == 3:
        out = raw[0]
        if out.ndim == 2:
            vals = _dequantize(out, quant).astype(np.float32)
            probs = vals if vals.min() >= 0.0 and vals.max() <= 1.0 else _sigmoid(vals)
            confidence = np.abs(probs - 0.5) * 2.0
            summary["confidence_score"] = float(np.mean(confidence))
            summary["uncertainty_score"] = float(1.0 - np.mean(confidence))
            summary["probability_mode"] = "binary"
            return summary
        vals = _dequantize(out, quant).astype(np.float32)
        probs = _stable_softmax(vals, axis=-1)
        top1 = np.max(probs, axis=-1)
        sorted_probs = np.sort(probs, axis=-1)
        top2 = sorted_probs[:, -2] if probs.shape[-1] > 1 else np.zeros_like(top1)
        summary["confidence_score"] = float(np.mean(top1))
        summary["margin_score"] = float(np.mean(top1 - top2))
        summary["uncertainty_score"] = float(1.0 - np.mean(top1 - top2))
        summary["probability_mode"] = "multiclass"
        return summary

    summary["confidence_score"] = 0.0
    summary["uncertainty_score"] = 1.0
    summary["probability_mode"] = "unknown"
    return summary


def _decode_segmentation_output(raw: np.ndarray, output_details: dict | None = None) -> np.ndarray:
    quant = output_details if output_details is not None else {"quantization": (0.0, 0)}

    if raw.ndim == 4:
        out = raw[0]
        if out.shape[0] in (1, 2, 3, 4, 5, 6, 7, 8) and out.shape[-1] not in (1, 2, 3, 4, 5, 6, 7, 8):
            out = np.transpose(out, (1, 2, 0))
        if out.ndim == 3 and out.shape[-1] > 1:
            pred_small = np.argmax(out, axis=-1).astype(np.uint8)
        elif out.ndim == 3 and out.shape[-1] == 1:
            vals = _dequantize(out[:, :, 0], quant)
            threshold = 0.5 if vals.min() >= 0.0 and vals.max() <= 1.0 else 0.0
            pred_small = (vals > threshold).astype(np.uint8)
        else:
            vals = _dequantize(out.squeeze(), quant)
            threshold = 0.5 if vals.min() >= 0.0 and vals.max() <= 1.0 else 0.0
            pred_small = (vals > threshold).astype(np.uint8)
    elif raw.ndim == 3:
        out = raw[0]
        if out.ndim == 2:
            vals = _dequantize(out, quant)
            threshold = 0.5 if vals.min() >= 0.0 and vals.max() <= 1.0 else 0.0
            pred_small = (vals > threshold).astype(np.uint8)
        else:
            pred_small = np.argmax(out, axis=-1).astype(np.uint8)
    elif raw.ndim == 2:
        out = raw[0]
        if out.ndim == 1:
            side = int(math.sqrt(out.shape[0]))
            if side * side != out.shape[0]:
                raise ValueError(f"Unsupported flattened output shape: {raw.shape}")
            pred_small = (out.reshape(side, side) > 0.0).astype(np.uint8)
        else:
            pred_small = np.argmax(out, axis=-1).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported output tensor shape: {raw.shape}")

    return pred_small.astype(np.uint8)


def _resize_image_channels(image_rgb: np.ndarray, width: int, height: int, channels: int) -> np.ndarray:
    if channels <= 0:
        raise ValueError("Channel count must be positive.")

    if channels == 1:
        resized = np.array(Image.fromarray(image_rgb).convert("L").resize((width, height), Image.BILINEAR), dtype=np.uint8)
        return resized[:, :, None]

    resized = np.array(Image.fromarray(image_rgb).resize((width, height), Image.BILINEAR), dtype=np.uint8)
    if resized.ndim == 2:
        resized = np.repeat(resized[:, :, None], 3, axis=2)
    if channels == 3 and resized.shape[2] > 3:
        return resized[:, :, :3]
    if resized.shape[2] >= channels:
        return resized[:, :, :channels]
    pad = np.zeros((resized.shape[0], resized.shape[1], channels - resized.shape[2]), dtype=np.uint8)
    return np.concatenate([resized, pad], axis=2)


def run_tflite_mask(image_rgb: np.ndarray, model_bytes: bytes) -> tuple[np.ndarray, dict]:
    interpreter_class = _get_interpreter_class()
    if interpreter_class is None:
        raise RuntimeError("TFLite runtime is not available. Install tflite-runtime or tensorflow.")

    interpreter = interpreter_class(model_content=model_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    input_shape = [int(v) for v in input_details["shape"]]
    input_dtype = np.dtype(input_details["dtype"])
    input_index = input_details["index"]

    if len(input_shape) != 4:
        raise ValueError(f"Expected 4D input tensor, got {input_shape}")

    nchw_input = False
    if input_shape[1] in (1, 3, 4) and input_shape[-1] not in (1, 3, 4):
        nchw_input = True
        _, channels, in_h, in_w = input_shape
    else:
        _, in_h, in_w, channels = input_shape

    in_h = int(in_h)
    in_w = int(in_w)
    channels = int(channels)
    if in_h <= 0 or in_w <= 0:
        raise ValueError(f"Model has dynamic or invalid input size: {input_shape}")

    resized = _resize_image_channels(image_rgb, in_w, in_h, channels)

    if input_dtype == np.float32:
        tensor = (resized.astype(np.float32) / 255.0).astype(np.float32)
    elif input_dtype == np.uint8:
        tensor = resized.astype(np.uint8)
    elif input_dtype == np.int8:
        scale, zero = input_details.get("quantization", (0.0, 0))
        normalized = resized.astype(np.float32) / 255.0
        if scale and scale > 0:
            tensor = np.round(normalized / scale + zero)
        else:
            tensor = np.round(normalized * 127.0)
        tensor = np.clip(tensor, -128, 127).astype(np.int8)
    else:
        tensor = resized.astype(input_dtype)

    if nchw_input:
        tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    try:
        interpreter.resize_tensor_input(input_index, tensor.shape, strict=False)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
    except Exception:
        pass

    interpreter.set_tensor(input_details["index"], tensor)
    interpreter.invoke()

    output_details = interpreter.get_output_details()[0]
    raw = interpreter.get_tensor(output_details["index"])

    pred_small = _decode_segmentation_output(raw, output_details)

    h, w = image_rgb.shape[:2]
    pred = np.array(Image.fromarray(pred_small).resize((w, h), Image.NEAREST), dtype=np.uint8)
    details = {
        "input_shape": [int(v) for v in input_details["shape"]],
        "output_shape": [int(v) for v in raw.shape],
        "input_dtype": str(input_dtype),
        "output_dtype": str(output_details["dtype"]),
    }
    details.update(_prediction_uncertainty_summary(raw, output_details))
    details["prediction_nonzero_pixels"] = int(np.count_nonzero(pred > 0))
    details["prediction_coverage_fraction"] = float(np.count_nonzero(pred > 0) / max(1, pred.size))
    return pred, details


@lru_cache(maxsize=2)
def _load_keras_model_cached(model_path: str):
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required to load .h5/.keras models. "
            "Install TensorFlow in this runtime or configure an external TensorFlow Python path."
        ) from exc

    try:
        return tf.keras.models.load_model(model_path, compile=False)
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
            return tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        except Exception as compat_exc:
            raise RuntimeError(
                "Model deserialization failed and compatibility fallback did not recover it. "
                f"Original error: {exc} | Fallback error: {compat_exc}"
            ) from compat_exc


def run_keras_mask(image_rgb: np.ndarray, model_path: Path) -> tuple[np.ndarray, dict]:
    model = _load_keras_model_cached(str(model_path.resolve()))
    input_tensor = model.inputs[0]
    input_shape = [dim if dim is None else int(dim) for dim in input_tensor.shape]
    if len(input_shape) != 4:
        raise ValueError(f"Expected NHWC input for Keras model, got {input_shape}")

    _, in_h, in_w, channels = input_shape
    in_h = int(in_h) if in_h is not None else image_rgb.shape[0]
    in_w = int(in_w) if in_w is not None else image_rgb.shape[1]
    channels = int(channels) if channels is not None else 3
    if in_h <= 0 or in_w <= 0:
        raise ValueError(f"Invalid model input shape: {input_shape}")

    resized = _resize_image_channels(image_rgb, in_w, in_h, channels)
    dtype = np.dtype(str(input_tensor.dtype).replace("<dtype: '", "").replace("'>", ""))
    if np.issubdtype(dtype, np.floating):
        model_input = resized.astype(np.float32) / 255.0
    else:
        model_input = resized.astype(dtype)

    batch = np.expand_dims(model_input, axis=0)

    prediction = model(batch, training=False)
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    if hasattr(prediction, "numpy"):
        raw = prediction.numpy()
    else:
        raw = np.array(prediction)

    pred_small = _decode_segmentation_output(raw, None)
    h, w = image_rgb.shape[:2]
    pred = np.array(Image.fromarray(pred_small).resize((w, h), Image.NEAREST), dtype=np.uint8)

    details = {
        "backend": "keras",
        "input_shape": [int(v) if v is not None else None for v in input_shape],
        "output_shape": [int(v) for v in raw.shape],
        "input_dtype": str(dtype),
        "output_dtype": str(raw.dtype),
    }
    details.update(_prediction_uncertainty_summary(np.asarray(raw), None))
    details["prediction_nonzero_pixels"] = int(np.count_nonzero(pred > 0))
    details["prediction_coverage_fraction"] = float(np.count_nonzero(pred > 0) / max(1, pred.size))
    return pred, details


def _build_pytorch_unet(torch_module, in_channels: int, num_classes: int, base_filters: int, dropout: float):
    nn = torch_module.nn

    class _DoubleConv(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, drop: float):
            super().__init__()
            layers: list[nn.Module] = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            if float(drop) > 0.0:
                layers.append(nn.Dropout2d(float(drop)))
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            return self.block(x)

    class _TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            f1 = max(8, int(base_filters))
            f2 = int(f1 * 2)
            f3 = int(f1 * 4)
            f4 = int(f1 * 8)
            self.enc1 = _DoubleConv(in_channels, f1, float(dropout) * 0.2)
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = _DoubleConv(f1, f2, float(dropout) * 0.4)
            self.pool2 = nn.MaxPool2d(2)
            self.enc3 = _DoubleConv(f2, f3, float(dropout) * 0.6)
            self.pool3 = nn.MaxPool2d(2)
            self.bottleneck = _DoubleConv(f3, f4, float(dropout))
            self.up3 = nn.ConvTranspose2d(f4, f3, kernel_size=2, stride=2)
            self.dec3 = _DoubleConv(f3 + f3, f3, float(dropout) * 0.6)
            self.up2 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)
            self.dec2 = _DoubleConv(f2 + f2, f2, float(dropout) * 0.4)
            self.up1 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)
            self.dec1 = _DoubleConv(f1 + f1, f1, float(dropout) * 0.2)
            self.head = nn.Conv2d(f1, num_classes, kernel_size=1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            b = self.bottleneck(self.pool3(e3))
            d3 = self.up3(b)
            d3 = self.dec3(torch_module.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch_module.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch_module.cat([d1, e1], dim=1))
            return self.head(d1)

    return _TinyUNet()


def _normalize_patch_size(value, default: int = 256) -> int:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        resolved = int(value)
    except Exception:
        resolved = int(default)
    return max(16, resolved)


def _load_pytorch_checkpoint(torch_module, model_path: Path) -> tuple[dict, dict]:
    path = str(model_path)
    try:
        payload = torch_module.load(path, map_location="cpu")
    except TypeError:
        payload = torch_module.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state_dict = payload["state_dict"]
        metadata: dict = {}
        nested_metadata = payload.get("metadata")
        if isinstance(nested_metadata, dict):
            metadata.update(nested_metadata)
        for key, value in payload.items():
            if key in {"state_dict", "metadata"}:
                continue
            metadata[key] = value
        return state_dict, metadata
    if isinstance(payload, dict):
        tensor_values = [value for value in payload.values() if hasattr(value, "shape")]
        if tensor_values and len(tensor_values) == len(payload):
            return payload, {}
    raise RuntimeError(
        "Unsupported PyTorch checkpoint format. Expected a training artifact with "
        "'state_dict' metadata or a plain state_dict checkpoint."
    )


def _infer_pytorch_checkpoint_config(state_dict: dict, metadata: dict) -> dict[str, int | float]:
    num_classes = metadata.get("num_classes")
    if num_classes is None:
        head_weight = state_dict.get("head.weight")
        if head_weight is None or not hasattr(head_weight, "shape") or len(head_weight.shape) < 1:
            raise RuntimeError("PyTorch checkpoint is missing 'num_classes' and head.weight shape is unavailable.")
        num_classes = int(head_weight.shape[0])

    base_filters = metadata.get("base_filters")
    if base_filters is None:
        enc_weight = state_dict.get("enc1.block.0.weight")
        if enc_weight is None or not hasattr(enc_weight, "shape") or len(enc_weight.shape) < 1:
            raise RuntimeError("PyTorch checkpoint is missing 'base_filters' and enc1.block.0.weight shape is unavailable.")
        base_filters = int(enc_weight.shape[0])

    patch_size = _normalize_patch_size(metadata.get("patch_size", 256))
    dropout = metadata.get("dropout", 0.0)
    try:
        dropout_value = float(dropout)
    except Exception:
        dropout_value = 0.0

    return {
        "num_classes": int(num_classes),
        "base_filters": int(base_filters),
        "patch_size": int(patch_size),
        "dropout": float(dropout_value),
    }


def _is_unified_general_root_starter_metadata(metadata: dict) -> bool:
    return str(metadata.get("model_family") or "").strip().lower() == "general_root_starter_unified"


def _compose_unified_task_class_mask(task_masks: dict[str, np.ndarray]) -> np.ndarray:
    if not task_masks:
        raise RuntimeError("Unified General Root Starter inference did not return any task masks.")
    sample_mask = np.asarray(next(iter(task_masks.values())), dtype=np.uint8)
    shape = sample_mask.shape[:2]

    def _mask(name: str) -> np.ndarray:
        value = task_masks.get(name)
        if value is None:
            return np.zeros(shape, dtype=bool)
        return np.asarray(value, dtype=np.uint8) > 0

    root_binary = _mask("root_binary")
    primary_root = _mask("primary_root")
    lateral_root = _mask("lateral_root")
    shoot = _mask("shoot")
    seed = _mask("seed_crown")

    pred = np.zeros(shape, dtype=np.uint8)
    root_mask = np.logical_or(root_binary, primary_root)
    root_mask = np.logical_and(root_mask, np.logical_not(lateral_root))
    root_mask = np.logical_and(root_mask, np.logical_not(shoot))
    root_mask = np.logical_and(root_mask, np.logical_not(seed))
    pred[root_mask] = 1
    pred[shoot] = 2
    pred[lateral_root] = 3
    pred[seed] = 4
    return pred


def _select_pytorch_device(torch_module):
    if bool(getattr(torch_module.cuda, "is_available", lambda: False)()):
        return torch_module.device("cuda"), "cuda"
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps_backend is not None and bool(getattr(mps_backend, "is_available", lambda: False)()):
        return torch_module.device("mps"), "mps"
    return torch_module.device("cpu"), "cpu"


def run_pytorch_mask(image_rgb: np.ndarray, model_path: Path) -> tuple[np.ndarray, dict]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required to load .pt/.pth models. "
            "Install torch in this runtime or configure an external Python runtime."
        ) from exc

    state_dict, metadata = _load_pytorch_checkpoint(torch, model_path)
    if _is_unified_general_root_starter_metadata(metadata):
        from .general_root_starter.unified_model import run_unified_general_root_starter_on_image

        task_masks, details = run_unified_general_root_starter_on_image(image_rgb, model_path)
        pred = _compose_unified_task_class_mask(task_masks)
        details = dict(details)
        details["model_family"] = "general_root_starter_unified"
        details["prediction_nonzero_pixels"] = int(np.count_nonzero(pred > 0))
        details["prediction_coverage_fraction"] = float(np.count_nonzero(pred > 0) / max(1, pred.size))
        return pred, details

    checkpoint_cfg = _infer_pytorch_checkpoint_config(state_dict, metadata)
    patch_size = int(checkpoint_cfg["patch_size"])
    resized = _resize_image_channels(image_rgb, patch_size, patch_size, 3).astype(np.float32) / 255.0
    batch = np.expand_dims(np.transpose(resized, (2, 0, 1)), axis=0)

    device, device_label = _select_pytorch_device(torch)
    model = _build_pytorch_unet(
        torch,
        in_channels=3,
        num_classes=int(checkpoint_cfg["num_classes"]),
        base_filters=int(checkpoint_cfg["base_filters"]),
        dropout=float(checkpoint_cfg["dropout"]),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        input_tensor = torch.from_numpy(batch).float().to(device)
        logits = model(input_tensor)
        pred_small = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    h, w = image_rgb.shape[:2]
    pred = np.array(Image.fromarray(pred_small).resize((w, h), Image.NEAREST), dtype=np.uint8)
    details = {
        "backend": "pytorch",
        "framework": str(metadata.get("framework", "pytorch")),
        "checkpoint_path": str(model_path),
        "input_shape": [1, 3, int(patch_size), int(patch_size)],
        "output_shape": [1, int(checkpoint_cfg["num_classes"]), int(patch_size), int(patch_size)],
        "patch_size": int(patch_size),
        "num_classes": int(checkpoint_cfg["num_classes"]),
        "base_filters": int(checkpoint_cfg["base_filters"]),
        "dropout": float(checkpoint_cfg["dropout"]),
        "device": device_label,
    }
    details.update(_prediction_uncertainty_summary(logits.detach().cpu().numpy(), None))
    details["prediction_nonzero_pixels"] = int(np.count_nonzero(pred > 0))
    details["prediction_coverage_fraction"] = float(np.count_nonzero(pred > 0) / max(1, pred.size))
    return pred, details


_EXTERNAL_KERAS_INFERENCE_SCRIPT = r"""
import argparse
import json
import math

import numpy as np
from PIL import Image
import tensorflow as tf


def _load_model_with_compat(path: str):
    try:
        return tf.keras.models.load_model(path, compile=False)
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

        custom_objects = {"Conv2DTranspose": CompatConv2DTranspose}
        return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)


def _decode_segmentation_output(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 4:
        out = raw[0]
        if out.shape[0] in (1, 2, 3, 4, 5, 6, 7, 8) and out.shape[-1] not in (1, 2, 3, 4, 5, 6, 7, 8):
            out = np.transpose(out, (1, 2, 0))
        if out.ndim == 3 and out.shape[-1] > 1:
            return np.argmax(out, axis=-1).astype(np.uint8)
        if out.ndim == 3 and out.shape[-1] == 1:
            vals = out[:, :, 0].astype(np.float32)
            threshold = 0.5 if vals.min() >= 0.0 and vals.max() <= 1.0 else 0.0
            return (vals > threshold).astype(np.uint8)
        vals = out.squeeze().astype(np.float32)
        threshold = 0.5 if vals.min() >= 0.0 and vals.max() <= 1.0 else 0.0
        return (vals > threshold).astype(np.uint8)

    if raw.ndim == 3:
        out = raw[0]
        if out.ndim == 2:
            vals = out.astype(np.float32)
            threshold = 0.5 if vals.min() >= 0.0 and vals.max() <= 1.0 else 0.0
            return (vals > threshold).astype(np.uint8)
        return np.argmax(out, axis=-1).astype(np.uint8)

    if raw.ndim == 2:
        out = raw[0]
        if out.ndim == 1:
            side = int(math.sqrt(out.shape[0]))
            if side * side != out.shape[0]:
                raise ValueError(f"Unsupported flattened output shape: {raw.shape}")
            return (out.reshape(side, side) > 0.0).astype(np.uint8)
        return np.argmax(out, axis=-1).astype(np.uint8)

    raise ValueError(f"Unsupported output tensor shape: {raw.shape}")


def _stable_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    denom = np.maximum(np.sum(exp, axis=axis, keepdims=True), 1e-8)
    return exp / denom


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -32.0, 32.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _prediction_uncertainty_summary(raw: np.ndarray) -> dict:
    if raw.ndim == 4:
        out = raw[0]
        if out.shape[0] in (1, 2, 3, 4, 5, 6, 7, 8) and out.shape[-1] not in (1, 2, 3, 4, 5, 6, 7, 8):
            out = np.transpose(out, (1, 2, 0))
        if out.ndim == 3 and out.shape[-1] > 1:
            vals = out.astype(np.float32)
            probs = vals if vals.min() >= 0.0 and vals.max() <= 1.0 and np.allclose(np.sum(vals, axis=-1), 1.0, atol=0.15) else _stable_softmax(vals, axis=-1)
            top1 = np.max(probs, axis=-1)
            sorted_probs = np.sort(probs, axis=-1)
            top2 = sorted_probs[:, :, -2] if probs.shape[-1] > 1 else np.zeros_like(top1)
            return {
                "confidence_score": float(np.mean(top1)),
                "margin_score": float(np.mean(top1 - top2)),
                "uncertainty_score": float(1.0 - np.mean(top1 - top2)),
                "probability_mode": "multiclass",
            }
        vals = out.squeeze().astype(np.float32)
        probs = vals if vals.min() >= 0.0 and vals.max() <= 1.0 else _sigmoid(vals)
        confidence = np.abs(probs - 0.5) * 2.0
        return {
            "confidence_score": float(np.mean(confidence)),
            "uncertainty_score": float(1.0 - np.mean(confidence)),
            "probability_mode": "binary",
        }
    return {"confidence_score": 0.0, "uncertainty_score": 1.0, "probability_mode": "unknown"}


def _resize_image_channels(image_rgb: np.ndarray, width: int, height: int, channels: int) -> np.ndarray:
    if channels <= 0:
        raise ValueError("Channel count must be positive.")
    if channels == 1:
        resized = np.array(Image.fromarray(image_rgb).convert("L").resize((width, height), Image.BILINEAR), dtype=np.uint8)
        return resized[:, :, None]

    resized = np.array(Image.fromarray(image_rgb).resize((width, height), Image.BILINEAR), dtype=np.uint8)
    if resized.ndim == 2:
        resized = np.repeat(resized[:, :, None], 3, axis=2)
    if channels == 3 and resized.shape[2] > 3:
        return resized[:, :, :3]
    if resized.shape[2] >= channels:
        return resized[:, :, :channels]
    pad = np.zeros((resized.shape[0], resized.shape[1], channels - resized.shape[2]), dtype=np.uint8)
    return np.concatenate([resized, pad], axis=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--meta", required=True)
    args = parser.parse_args()

    image_rgb = np.asarray(np.load(args.image, allow_pickle=False), dtype=np.uint8)
    model = _load_model_with_compat(args.model)
    input_tensor = model.inputs[0]
    input_shape = [dim if dim is None else int(dim) for dim in input_tensor.shape]
    if len(input_shape) != 4:
        raise ValueError(f"Expected NHWC input for Keras model, got {input_shape}")

    _, in_h, in_w, channels = input_shape
    in_h = int(in_h) if in_h is not None else image_rgb.shape[0]
    in_w = int(in_w) if in_w is not None else image_rgb.shape[1]
    channels = int(channels) if channels is not None else 3
    if in_h <= 0 or in_w <= 0:
        raise ValueError(f"Invalid model input shape: {input_shape}")

    resized = _resize_image_channels(image_rgb, in_w, in_h, channels)
    dtype = np.dtype(str(input_tensor.dtype).replace("<dtype: '", "").replace("'>", ""))
    if np.issubdtype(dtype, np.floating):
        model_input = resized.astype(np.float32) / 255.0
    else:
        model_input = resized.astype(dtype)

    batch = np.expand_dims(model_input, axis=0)
    prediction = model(batch, training=False)
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    raw = prediction.numpy() if hasattr(prediction, "numpy") else np.array(prediction)

    pred_small = _decode_segmentation_output(np.asarray(raw))
    h, w = image_rgb.shape[:2]
    pred = np.array(Image.fromarray(pred_small).resize((w, h), Image.NEAREST), dtype=np.uint8)

    details = {
        "backend": "keras-external",
        "input_shape": [int(v) if v is not None else None for v in input_shape],
        "output_shape": [int(v) for v in raw.shape],
        "input_dtype": str(dtype),
        "output_dtype": str(raw.dtype),
    }
    details.update(_prediction_uncertainty_summary(np.asarray(raw)))
    details["prediction_nonzero_pixels"] = int(np.count_nonzero(pred > 0))
    details["prediction_coverage_fraction"] = float(np.count_nonzero(pred > 0) / max(1, pred.size))
    np.save(args.pred, pred.astype(np.uint8), allow_pickle=False)
    with open(args.meta, "w", encoding="utf-8") as handle:
        json.dump(details, handle)


if __name__ == "__main__":
    main()
"""


def _external_pytorch_inference_script() -> str:
    return "\n".join(
        [
            "import argparse",
            "import json",
            "import numpy as np",
            "from PIL import Image",
            "import torch",
            "",
            "def _resize_image_channels(image_rgb: np.ndarray, width: int, height: int, channels: int) -> np.ndarray:",
            "    if channels <= 0:",
            "        raise ValueError('Channel count must be positive.')",
            "    if channels == 1:",
            "        resized = np.array(Image.fromarray(image_rgb).convert('L').resize((width, height), Image.BILINEAR), dtype=np.uint8)",
            "        return resized[:, :, None]",
            "    resized = np.array(Image.fromarray(image_rgb).resize((width, height), Image.BILINEAR), dtype=np.uint8)",
            "    if resized.ndim == 2:",
            "        resized = np.repeat(resized[:, :, None], 3, axis=2)",
            "    if channels == 3 and resized.shape[2] > 3:",
            "        return resized[:, :, :3]",
            "    if resized.shape[2] >= channels:",
            "        return resized[:, :, :channels]",
            "    pad = np.zeros((resized.shape[0], resized.shape[1], channels - resized.shape[2]), dtype=np.uint8)",
            "    return np.concatenate([resized, pad], axis=2)",
            "",
            "def _normalize_patch_size(value, default: int = 256) -> int:",
            "    if isinstance(value, (list, tuple)) and value:",
            "        value = value[0]",
            "    try:",
            "        resolved = int(value)",
            "    except Exception:",
            "        resolved = int(default)",
            "    return max(16, resolved)",
            "",
            "def _build_model(in_channels: int, num_classes: int, base_filters: int, dropout: float):",
            "    nn = torch.nn",
            "    class _DoubleConv(nn.Module):",
            "        def __init__(self, in_ch: int, out_ch: int, drop: float):",
            "            super().__init__()",
            "            layers = [",
            "                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),",
            "                nn.ReLU(inplace=True),",
            "                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),",
            "                nn.ReLU(inplace=True),",
            "            ]",
            "            if float(drop) > 0.0:",
            "                layers.append(nn.Dropout2d(float(drop)))",
            "            self.block = nn.Sequential(*layers)",
            "        def forward(self, x):",
            "            return self.block(x)",
            "    class _TinyUNet(nn.Module):",
            "        def __init__(self):",
            "            super().__init__()",
            "            f1 = max(8, int(base_filters))",
            "            f2 = int(f1 * 2)",
            "            f3 = int(f1 * 4)",
            "            f4 = int(f1 * 8)",
            "            self.enc1 = _DoubleConv(in_channels, f1, float(dropout) * 0.2)",
            "            self.pool1 = nn.MaxPool2d(2)",
            "            self.enc2 = _DoubleConv(f1, f2, float(dropout) * 0.4)",
            "            self.pool2 = nn.MaxPool2d(2)",
            "            self.enc3 = _DoubleConv(f2, f3, float(dropout) * 0.6)",
            "            self.pool3 = nn.MaxPool2d(2)",
            "            self.bottleneck = _DoubleConv(f3, f4, float(dropout))",
            "            self.up3 = nn.ConvTranspose2d(f4, f3, kernel_size=2, stride=2)",
            "            self.dec3 = _DoubleConv(f3 + f3, f3, float(dropout) * 0.6)",
            "            self.up2 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)",
            "            self.dec2 = _DoubleConv(f2 + f2, f2, float(dropout) * 0.4)",
            "            self.up1 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)",
            "            self.dec1 = _DoubleConv(f1 + f1, f1, float(dropout) * 0.2)",
            "            self.head = nn.Conv2d(f1, num_classes, kernel_size=1)",
            "        def forward(self, x):",
            "            e1 = self.enc1(x)",
            "            e2 = self.enc2(self.pool1(e1))",
            "            e3 = self.enc3(self.pool2(e2))",
            "            b = self.bottleneck(self.pool3(e3))",
            "            d3 = self.up3(b)",
            "            d3 = self.dec3(torch.cat([d3, e3], dim=1))",
            "            d2 = self.up2(d3)",
            "            d2 = self.dec2(torch.cat([d2, e2], dim=1))",
            "            d1 = self.up1(d2)",
            "            d1 = self.dec1(torch.cat([d1, e1], dim=1))",
            "            return self.head(d1)",
            "    return _TinyUNet()",
            "",
            "def _load_checkpoint(path: str):",
            "    try:",
            "        payload = torch.load(path, map_location='cpu')",
            "    except TypeError:",
            "        payload = torch.load(path, map_location='cpu', weights_only=False)",
            "    if isinstance(payload, dict) and 'state_dict' in payload and isinstance(payload['state_dict'], dict):",
            "        state_dict = payload['state_dict']",
            "        meta = {}",
            "        nested = payload.get('metadata')",
            "        if isinstance(nested, dict):",
            "            meta.update(nested)",
            "        for key, value in payload.items():",
            "            if key in {'state_dict', 'metadata'}:",
            "                continue",
            "            meta[key] = value",
            "        return state_dict, meta",
            "    if isinstance(payload, dict):",
            "        tensor_values = [value for value in payload.values() if hasattr(value, 'shape')]",
            "        if tensor_values and len(tensor_values) == len(payload):",
            "            return payload, {}",
            "    raise RuntimeError('Unsupported PyTorch checkpoint format.')",
            "",
            "def _is_unified_meta(metadata: dict) -> bool:",
            "    return str(metadata.get('model_family') or '').strip().lower() == 'general_root_starter_unified'",
            "",
            "def _infer_cfg(state_dict: dict, metadata: dict):",
            "    num_classes = metadata.get('num_classes')",
            "    if num_classes is None:",
            "        head_weight = state_dict.get('head.weight')",
            "        if head_weight is None:",
            "            raise RuntimeError(\"Checkpoint missing 'num_classes' and head.weight.\")",
            "        num_classes = int(head_weight.shape[0])",
            "    base_filters = metadata.get('base_filters')",
            "    if base_filters is None:",
            "        enc_weight = state_dict.get('enc1.block.0.weight')",
            "        if enc_weight is None:",
            "            raise RuntimeError(\"Checkpoint missing 'base_filters' and enc1.block.0.weight.\")",
            "        base_filters = int(enc_weight.shape[0])",
            "    patch_size = _normalize_patch_size(metadata.get('patch_size', 256))",
            "    try:",
            "        dropout = float(metadata.get('dropout', 0.0))",
            "    except Exception:",
            "        dropout = 0.0",
            "    return int(num_classes), int(base_filters), int(patch_size), float(dropout)",
            "",
            "def _build_unified_model(tasks, families, base_filters: int, dropout: float):",
            "    nn = torch.nn",
            "    class ConvBlock(nn.Module):",
            "        def __init__(self, in_channels: int, out_channels: int, dropout_value: float = 0.0):",
            "            super().__init__()",
            "            layers = [",
            "                nn.Conv2d(in_channels, out_channels, 3, padding=1),",
            "                nn.BatchNorm2d(out_channels),",
            "                nn.ReLU(inplace=True),",
            "                nn.Conv2d(out_channels, out_channels, 3, padding=1),",
            "                nn.BatchNorm2d(out_channels),",
            "                nn.ReLU(inplace=True),",
            "            ]",
            "            if float(dropout_value) > 0.0:",
            "                layers.append(nn.Dropout2d(float(dropout_value)))",
            "            self.block = nn.Sequential(*layers)",
            "        def forward(self, x):",
            "            return self.block(x)",
            "    class UnifiedRootStarterNet(nn.Module):",
            "        def __init__(self):",
            "            super().__init__()",
            "            bf = max(8, int(base_filters))",
            "            self.enc1 = ConvBlock(3, bf, dropout * 0.25)",
            "            self.pool1 = nn.MaxPool2d(2)",
            "            self.enc2 = ConvBlock(bf, bf * 2, dropout * 0.5)",
            "            self.pool2 = nn.MaxPool2d(2)",
            "            self.enc3 = ConvBlock(bf * 2, bf * 4, dropout * 0.75)",
            "            self.pool3 = nn.MaxPool2d(2)",
            "            self.bottleneck = ConvBlock(bf * 4, bf * 8, dropout)",
            "            self.up3 = nn.ConvTranspose2d(bf * 8, bf * 4, 2, stride=2)",
            "            self.dec3 = ConvBlock(bf * 8, bf * 4, dropout * 0.75)",
            "            self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)",
            "            self.dec2 = ConvBlock(bf * 4, bf * 2, dropout * 0.5)",
            "            self.up1 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)",
            "            self.dec1 = ConvBlock(bf * 2, bf, dropout * 0.25)",
            "            self.heads = nn.ModuleDict({task: nn.Conv2d(bf, 1, 1) for task in tasks})",
            "            self.family_head = nn.Linear(bf * 8, len(families)) if len(families) > 1 else None",
            "        def forward(self, x):",
            "            e1 = self.enc1(x)",
            "            e2 = self.enc2(self.pool1(e1))",
            "            e3 = self.enc3(self.pool2(e2))",
            "            b = self.bottleneck(self.pool3(e3))",
            "            d3 = self.up3(b)",
            "            d3 = self.dec3(torch.cat([d3, e3], dim=1))",
            "            d2 = self.up2(d3)",
            "            d2 = self.dec2(torch.cat([d2, e2], dim=1))",
            "            d1 = self.up1(d2)",
            "            d1 = self.dec1(torch.cat([d1, e1], dim=1))",
            "            outputs = {task: self.heads[task](d1) for task in tasks}",
            "            if self.family_head is not None:",
            "                pooled = torch.mean(b, dim=(2, 3))",
            "                outputs['family_logits'] = self.family_head(pooled)",
            "            return outputs",
            "    return UnifiedRootStarterNet()",
            "",
            "def _select_device():",
            "    if bool(getattr(torch.cuda, 'is_available', lambda: False)()):",
            "        return torch.device('cuda'), 'cuda'",
            "    mps_backend = getattr(getattr(torch, 'backends', None), 'mps', None)",
            "    if mps_backend is not None and bool(getattr(mps_backend, 'is_available', lambda: False)()):",
            "        return torch.device('mps'), 'mps'",
            "    return torch.device('cpu'), 'cpu'",
            "",
            "def _stable_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:",
            "    shifted = values - np.max(values, axis=axis, keepdims=True)",
            "    exp = np.exp(shifted)",
            "    denom = np.maximum(np.sum(exp, axis=axis, keepdims=True), 1e-8)",
            "    return exp / denom",
            "",
            "def _prediction_uncertainty_summary(raw: np.ndarray) -> dict:",
            "    if raw.ndim == 4:",
            "        out = raw[0]",
            "        if out.shape[0] > 1:",
            "            logits = np.transpose(out, (1, 2, 0)).astype(np.float32)",
            "            probs = _stable_softmax(logits, axis=-1)",
            "            top1 = np.max(probs, axis=-1)",
            "            sorted_probs = np.sort(probs, axis=-1)",
            "            top2 = sorted_probs[:, :, -2] if probs.shape[-1] > 1 else np.zeros_like(top1)",
            "            return {",
            "                'confidence_score': float(np.mean(top1)),",
            "                'margin_score': float(np.mean(top1 - top2)),",
            "                'uncertainty_score': float(1.0 - np.mean(top1 - top2)),",
            "                'probability_mode': 'multiclass',",
            "            }",
            "        vals = out[0].astype(np.float32)",
            "        probs = 1.0 / (1.0 + np.exp(-np.clip(vals, -32.0, 32.0)))",
            "        confidence = np.abs(probs - 0.5) * 2.0",
            "        return {",
            "            'confidence_score': float(np.mean(confidence)),",
            "            'uncertainty_score': float(1.0 - np.mean(confidence)),",
            "            'probability_mode': 'binary',",
            "        }",
            "    return {'confidence_score': 0.0, 'uncertainty_score': 1.0, 'probability_mode': 'unknown'}",
            "",
            "def _compose_unified_mask(task_masks: dict):",
            "    sample = np.asarray(next(iter(task_masks.values())), dtype=np.uint8)",
            "    shape = sample.shape[:2]",
            "    def _mask(name: str):",
            "        value = task_masks.get(name)",
            "        if value is None:",
            "            return np.zeros(shape, dtype=bool)",
            "        return np.asarray(value, dtype=np.uint8) > 0",
            "    root_binary = _mask('root_binary')",
            "    primary_root = _mask('primary_root')",
            "    lateral_root = _mask('lateral_root')",
            "    shoot = _mask('shoot')",
            "    seed = _mask('seed_crown')",
            "    pred = np.zeros(shape, dtype=np.uint8)",
            "    root_mask = np.logical_or(root_binary, primary_root)",
            "    root_mask = np.logical_and(root_mask, np.logical_not(lateral_root))",
            "    root_mask = np.logical_and(root_mask, np.logical_not(shoot))",
            "    root_mask = np.logical_and(root_mask, np.logical_not(seed))",
            "    pred[root_mask] = 1",
            "    pred[shoot] = 2",
            "    pred[lateral_root] = 3",
            "    pred[seed] = 4",
            "    return pred",
            "",
            "def main() -> None:",
            "    parser = argparse.ArgumentParser()",
            "    parser.add_argument('--image', required=True)",
            "    parser.add_argument('--model', required=True)",
            "    parser.add_argument('--pred', required=True)",
            "    parser.add_argument('--meta', required=True)",
            "    args = parser.parse_args()",
            "    image_rgb = np.asarray(np.load(args.image, allow_pickle=False), dtype=np.uint8)",
            "    state_dict, metadata = _load_checkpoint(args.model)",
            "    device, device_label = _select_device()",
            "    if _is_unified_meta(metadata):",
            "        tasks = tuple(metadata.get('tasks') or ('root_binary', 'shoot', 'primary_root', 'lateral_root', 'seed_crown'))",
            "        families = tuple(metadata.get('families') or ())",
            "        base_filters = int(metadata.get('base_filters') or 24)",
            "        dropout = float(metadata.get('dropout') or 0.10)",
            "        patch_size = _normalize_patch_size(metadata.get('patch_size', 256))",
            "        model = _build_unified_model(tasks, families, base_filters, dropout).to(device)",
            "        model.load_state_dict(state_dict, strict=False)",
            "        model.eval()",
            "        resized = _resize_image_channels(image_rgb, patch_size, patch_size, 3).astype(np.float32) / 255.0",
            "        batch = np.expand_dims(np.transpose(resized, (2, 0, 1)), axis=0)",
            "        with torch.no_grad():",
            "            tensor = torch.from_numpy(batch).float().to(device)",
            "            outputs = model(tensor)",
            "        h, w = image_rgb.shape[:2]",
            "        task_masks = {}",
            "        certainty_scores = []",
            "        task_confidences = {}",
            "        for task in tasks:",
            "            probs = torch.sigmoid(outputs[task])[0, 0].detach().cpu().numpy().astype(np.float32)",
            "            full = np.asarray(Image.fromarray((probs * 255.0).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.uint8).astype(np.float32) / 255.0",
            "            task_masks[str(task)] = (full >= 0.5).astype(np.uint8)",
            "            certainty = float(np.mean(np.abs(full - 0.5) * 2.0))",
            "            task_confidences[str(task)] = certainty",
            "            certainty_scores.append(certainty)",
            "        pred = _compose_unified_mask(task_masks)",
            "        details = {",
            "            'backend': 'pytorch-external',",
            "            'framework': str(metadata.get('framework', 'pytorch')),",
            "            'checkpoint_path': str(args.model),",
            "            'patch_size': int(patch_size),",
            "            'base_filters': int(base_filters),",
            "            'dropout': float(dropout),",
            "            'device': device_label,",
            "            'tasks': list(tasks),",
            "            'families': list(families),",
            "            'model_family': 'general_root_starter_unified',",
            "            'task_confidences': task_confidences,",
            "            'confidence_score': float(np.mean(certainty_scores)) if certainty_scores else 0.0,",
            "            'uncertainty_score': float(1.0 - np.mean(certainty_scores)) if certainty_scores else 1.0,",
            "        }",
            "    else:",
            "        num_classes, base_filters, patch_size, dropout = _infer_cfg(state_dict, metadata)",
            "        model = _build_model(3, num_classes, base_filters, dropout).to(device)",
            "        model.load_state_dict(state_dict)",
            "        model.eval()",
            "        resized = _resize_image_channels(image_rgb, patch_size, patch_size, 3).astype(np.float32) / 255.0",
            "        batch = np.expand_dims(np.transpose(resized, (2, 0, 1)), axis=0)",
            "        with torch.no_grad():",
            "            tensor = torch.from_numpy(batch).float().to(device)",
            "            logits = model(tensor)",
            "            logits_np = logits.detach().cpu().numpy()",
            "            pred_small = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)",
            "        h, w = image_rgb.shape[:2]",
            "        pred = np.array(Image.fromarray(pred_small).resize((w, h), Image.NEAREST), dtype=np.uint8)",
            "        details = {",
            "            'backend': 'pytorch-external',",
            "            'framework': str(metadata.get('framework', 'pytorch')),",
            "            'checkpoint_path': str(args.model),",
            "            'patch_size': int(patch_size),",
            "            'num_classes': int(num_classes),",
            "            'base_filters': int(base_filters),",
            "            'dropout': float(dropout),",
            "            'device': device_label,",
            "            'input_shape': [1, 3, int(patch_size), int(patch_size)],",
            "            'output_shape': [1, int(num_classes), int(patch_size), int(patch_size)],",
            "        }",
            "        details.update(_prediction_uncertainty_summary(logits_np))",
            "    details['prediction_nonzero_pixels'] = int(np.count_nonzero(pred > 0))",
            "    details['prediction_coverage_fraction'] = float(np.count_nonzero(pred > 0) / max(1, pred.size))",
            "    np.save(args.pred, pred.astype(np.uint8), allow_pickle=False)",
            "    with open(args.meta, 'w', encoding='utf-8') as handle:",
            "        json.dump(details, handle)",
            "",
            "if __name__ == '__main__':",
            "    main()",
        ]
    )


def run_keras_mask_external(image_rgb: np.ndarray, model_path: Path, external_python: Path) -> tuple[np.ndarray, dict]:
    python_path = Path(external_python).expanduser().resolve()
    if not python_path.exists():
        raise RuntimeError(f"External TensorFlow python not found: {python_path}")

    model_resolved = model_path.expanduser().resolve()
    if not model_resolved.exists():
        raise FileNotFoundError(f"Model file not found: {model_resolved}")

    with tempfile.TemporaryDirectory(prefix="npec_tf_bridge_") as tmp_dir:
        temp_root = Path(tmp_dir)
        script_path = temp_root / "keras_bridge_worker.py"
        image_path = temp_root / "input_image.npy"
        pred_path = temp_root / "prediction.npy"
        meta_path = temp_root / "details.json"
        script_path.write_text(_EXTERNAL_KERAS_INFERENCE_SCRIPT, encoding="utf-8")
        np.save(image_path, np.asarray(image_rgb, dtype=np.uint8), allow_pickle=False)

        cmd = [
            str(python_path),
            str(script_path),
            "--image",
            str(image_path),
            "--model",
            str(model_resolved),
            "--pred",
            str(pred_path),
            "--meta",
            str(meta_path),
        ]
        env = os.environ.copy()
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or "Unknown TensorFlow bridge error."
            raise RuntimeError(f"External TensorFlow inference failed ({python_path}): {detail}")

        if not pred_path.exists():
            raise RuntimeError("External TensorFlow inference did not produce prediction output.")
        pred = np.asarray(np.load(pred_path, allow_pickle=False), dtype=np.uint8)
        if pred.shape[:2] != image_rgb.shape[:2]:
            h, w = image_rgb.shape[:2]
            pred = np.array(Image.fromarray(pred).resize((w, h), Image.NEAREST), dtype=np.uint8)

        details: dict = {}
        if meta_path.exists():
            try:
                details = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                details = {}
        details["backend"] = "keras-external"
        details["external_python"] = str(python_path)
        return pred.astype(np.uint8), details


def run_pytorch_mask_external(image_rgb: np.ndarray, model_path: Path, external_python: Path) -> tuple[np.ndarray, dict]:
    python_path = Path(external_python).expanduser().resolve()
    if not python_path.exists():
        raise RuntimeError(f"External PyTorch python not found: {python_path}")

    model_resolved = model_path.expanduser().resolve()
    if not model_resolved.exists():
        raise FileNotFoundError(f"Model file not found: {model_resolved}")

    with tempfile.TemporaryDirectory(prefix="npec_torch_bridge_") as tmp_dir:
        temp_root = Path(tmp_dir)
        script_path = temp_root / "pytorch_bridge_worker.py"
        image_path = temp_root / "input_image.npy"
        pred_path = temp_root / "prediction.npy"
        meta_path = temp_root / "details.json"
        script_path.write_text(_external_pytorch_inference_script(), encoding="utf-8")
        np.save(image_path, np.asarray(image_rgb, dtype=np.uint8), allow_pickle=False)

        cmd = [
            str(python_path),
            str(script_path),
            "--image",
            str(image_path),
            "--model",
            str(model_resolved),
            "--pred",
            str(pred_path),
            "--meta",
            str(meta_path),
        ]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or "Unknown PyTorch bridge error."
            raise RuntimeError(f"External PyTorch inference failed ({python_path}): {detail}")

        if not pred_path.exists():
            raise RuntimeError("External PyTorch inference did not produce prediction output.")
        pred = np.asarray(np.load(pred_path, allow_pickle=False), dtype=np.uint8)
        if pred.shape[:2] != image_rgb.shape[:2]:
            h, w = image_rgb.shape[:2]
            pred = np.array(Image.fromarray(pred).resize((w, h), Image.NEAREST), dtype=np.uint8)

        details: dict = {}
        if meta_path.exists():
            try:
                details = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                details = {}
        details["backend"] = "pytorch-external"
        details["external_python"] = str(python_path)
        return pred.astype(np.uint8), details


def _format_external_command(template: str, placeholders: dict[str, str]) -> list[str]:
    raw = str(template or "").strip()
    if not raw:
        raise RuntimeError("Command template is empty.")
    try:
        rendered = raw.format(**placeholders)
    except KeyError as exc:
        missing = str(exc).strip("'")
        raise RuntimeError(
            f"Command template is missing placeholder '{missing}'. "
            "Use placeholders like {hef}, {image}, {pred}, and {meta}."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Invalid command template: {exc}") from exc
    try:
        cmd = shlex.split(rendered, posix=(os.name != "nt"))
    except Exception as exc:
        raise RuntimeError(f"Unable to parse command template: {exc}") from exc
    if not cmd:
        raise RuntimeError("Command template rendered to an empty command.")
    return cmd


def run_hef_mask_external(image_rgb: np.ndarray, model_path: Path, infer_command: str | None) -> tuple[np.ndarray, dict]:
    model_resolved = model_path.expanduser().resolve()
    if not model_resolved.exists():
        raise FileNotFoundError(f"HEF file not found: {model_resolved}")

    command_template = str(infer_command or "").strip()
    if not command_template:
        raise RuntimeError(
            "Hailo inference command is not configured. "
            "Set a command template in Segmentation Pipeline (Hailo mode). "
            "Required placeholders: {hef}, {image}, {pred}, {meta}."
        )

    with tempfile.TemporaryDirectory(prefix="npec_hailo_bridge_") as tmp_dir:
        temp_root = Path(tmp_dir)
        image_path = temp_root / "input_image.npy"
        pred_path = temp_root / "prediction.npy"
        meta_path = temp_root / "details.json"
        np.save(image_path, np.asarray(image_rgb, dtype=np.uint8), allow_pickle=False)

        cmd = _format_external_command(
            command_template,
            {
                "hef": str(model_resolved),
                "image": str(image_path),
                "pred": str(pred_path),
                "meta": str(meta_path),
            },
        )

        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or "Unknown Hailo execution error."
            raise RuntimeError(f"Hailo inference command failed (exit {completed.returncode}): {detail}")

        if not pred_path.exists():
            raise RuntimeError(
                "Hailo inference command finished but no prediction file was written. "
                "Expected output at: " + str(pred_path)
            )
        pred = np.asarray(np.load(pred_path, allow_pickle=False), dtype=np.uint8)
        pred = np.squeeze(pred)
        if pred.ndim != 2:
            raise RuntimeError(f"Hailo prediction must be 2D index mask, got shape {tuple(pred.shape)}")

        if pred.shape[:2] != image_rgb.shape[:2]:
            h, w = image_rgb.shape[:2]
            pred = np.array(Image.fromarray(pred).resize((w, h), Image.NEAREST), dtype=np.uint8)

        details: dict = {}
        if meta_path.exists():
            try:
                parsed = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    details.update(parsed)
            except Exception:
                details = {}
        details.setdefault("backend", "hailo")
        details["hef_path"] = str(model_resolved)
        details["command"] = " ".join(cmd[:8]) + (" ..." if len(cmd) > 8 else "")
        return pred.astype(np.uint8), details


def run_model_mask(
    image_rgb: np.ndarray,
    model_path: Path,
    model_bytes: bytes | None = None,
    external_runtime_python: Path | None = None,
    external_keras_python: Path | None = None,
    hailo_infer_command: str | None = None,
) -> tuple[np.ndarray, dict]:
    resolved_external_python = external_runtime_python if external_runtime_python is not None else external_keras_python
    suffix = model_path.suffix.lower()
    if suffix == ".tflite":
        payload = model_bytes if model_bytes is not None else model_path.read_bytes()
        pred, details = run_tflite_mask(image_rgb, payload)
        details["backend"] = "tflite"
        return pred, details
    if suffix in {".h5", ".keras"}:
        if resolved_external_python is not None:
            return run_keras_mask_external(image_rgb, model_path, resolved_external_python)
        return run_keras_mask(image_rgb, model_path)
    if suffix in {".pt", ".pth"}:
        if resolved_external_python is not None:
            return run_pytorch_mask_external(image_rgb, model_path, resolved_external_python)
        return run_pytorch_mask(image_rgb, model_path)
    if suffix == ".hef":
        return run_hef_mask_external(image_rgb, model_path, infer_command=hailo_infer_command)
    raise ValueError(f"Unsupported model format: {model_path.suffix}. Use .tflite, .h5, .keras, .pt, .pth, or .hef.")
