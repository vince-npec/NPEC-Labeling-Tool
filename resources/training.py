from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import traceback

import numpy as np
try:
    from PySide6.QtCore import QObject, Signal, Slot
except Exception:
    class QObject:  # type: ignore[override]
        pass

    class Signal:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs

        def emit(self, *args, **kwargs):
            return None

        def connect(self, *args, **kwargs):
            return None

    def Slot(*args, **kwargs):  # type: ignore[override]
        def _decorator(func):
            return func

        return _decorator

from .mask_ops import ensure_layer_map, index_mask_from_layers
from .models import DatasetImageItem, LabelClass
from .runtime_python_utils import is_probably_python_executable, safe_current_python_executable


def _external_tf_probe_source() -> str:
    return "\n".join(
        [
            "import json",
            "info = {",
            "    'available': False,",
            "    'backend': 'tensorflow',",
            "    'version': None,",
            "    'gpu_count': 0,",
            "    'gpu_names': [],",
            "    'cpu_count': 0,",
            "    'cuda_built': False,",
            "    'error': None,",
            "}",
            "try:",
            "    import tensorflow as tf",
            "    g = tf.config.list_physical_devices('GPU')",
            "    c = tf.config.list_physical_devices('CPU')",
            "    info['available'] = True",
            "    info['version'] = str(getattr(tf, '__version__', 'unknown'))",
            "    info['gpu_count'] = len(g)",
            "    info['gpu_names'] = [str(d.name) for d in g]",
            "    info['cpu_count'] = len(c)",
            "    info['cuda_built'] = bool(getattr(tf.test, 'is_built_with_cuda', lambda: False)())",
            "except Exception as exc:",
            "    info['error'] = str(exc)",
            "print(json.dumps(info))",
        ]
    )


def _external_torch_probe_source() -> str:
    return "\n".join(
        [
            "import json",
            "import os",
            "info = {",
            "    'available': False,",
            "    'backend': 'pytorch',",
            "    'version': None,",
            "    'gpu_count': 0,",
            "    'gpu_names': [],",
            "    'cpu_count': int(os.cpu_count() or 0),",
            "    'cuda_available': False,",
            "    'mps_available': False,",
            "    'error': None,",
            "}",
            "try:",
            "    import torch",
            "    info['available'] = True",
            "    info['version'] = str(getattr(torch, '__version__', 'unknown'))",
            "    if bool(getattr(torch.cuda, 'is_available', lambda: False)()):",
            "        count = int(getattr(torch.cuda, 'device_count', lambda: 0)() or 0)",
            "        info['cuda_available'] = True",
            "        info['gpu_count'] = count",
            "        info['gpu_names'] = [str(torch.cuda.get_device_name(i)) for i in range(count)]",
            "    else:",
            "        mps_backend = getattr(getattr(torch, 'backends', None), 'mps', None)",
            "        if mps_backend is not None and bool(getattr(mps_backend, 'is_available', lambda: False)()):",
            "            info['mps_available'] = True",
            "            info['gpu_count'] = 1",
            "            info['gpu_names'] = ['Apple Metal (MPS)']",
            "except Exception as exc:",
            "    info['error'] = str(exc)",
            "print(json.dumps(info))",
        ]
    )


def _probe_external_tf_runtime(python_executable: Path) -> dict[str, object]:
    probe = _external_tf_probe_source()
    info: dict[str, object] = {
        "available": False,
        "backend": "tensorflow",
        "version": None,
        "gpu_count": 0,
        "gpu_names": [],
        "cpu_count": 0,
        "cuda_built": False,
        "diagnostics": None,
        "error": None,
    }
    try:
        proc = subprocess.run(
            [str(python_executable), "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=25,
        )
    except Exception as exc:
        info["error"] = f"External probe failed: {exc}"
        return info

    stderr_txt = (proc.stderr or "").strip()
    output = (proc.stdout or "").strip().splitlines()
    if not output:
        info["error"] = stderr_txt if stderr_txt else f"External probe failed (exit {proc.returncode})."
        return info

    try:
        payload = json.loads(output[-1])
    except Exception:
        stderr_txt = (proc.stderr or "").strip()
        info["error"] = stderr_txt or output[-1]
        return info

    if isinstance(payload, dict):
        info.update(payload)
    if stderr_txt:
        info["diagnostics"] = stderr_txt
        gpu_count = int(info.get("gpu_count") or 0)
        has_error = bool(str(info.get("error") or "").strip())
        if not has_error and gpu_count <= 0:
            info["error"] = stderr_txt
    return info


def _probe_external_torch_runtime(python_executable: Path) -> dict[str, object]:
    probe = _external_torch_probe_source()
    info: dict[str, object] = {
        "available": False,
        "backend": "pytorch",
        "version": None,
        "gpu_count": 0,
        "gpu_names": [],
        "cpu_count": 0,
        "cuda_available": False,
        "mps_available": False,
        "diagnostics": None,
        "error": None,
    }
    try:
        proc = subprocess.run(
            [str(python_executable), "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=25,
        )
    except Exception as exc:
        info["error"] = f"External probe failed: {exc}"
        return info

    stderr_txt = (proc.stderr or "").strip()
    output = (proc.stdout or "").strip().splitlines()
    if not output:
        info["error"] = stderr_txt if stderr_txt else f"External probe failed (exit {proc.returncode})."
        return info

    try:
        payload = json.loads(output[-1])
    except Exception:
        info["error"] = stderr_txt or output[-1]
        return info

    if isinstance(payload, dict):
        info.update(payload)
    if stderr_txt:
        info["diagnostics"] = stderr_txt
    return info


def _condense_runtime_diagnostics(text: str, *, max_lines: int = 4) -> str:
    lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "] " in line and line[:4].isdigit():
            line = line.split("] ", 1)[1].strip()
        if line not in lines:
            lines.append(line)
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return " | ".join(lines)


def _external_gpu_preflight_issue(python_executable: Path, *, framework: str = "tensorflow") -> str | None:
    normalized = str(framework or "tensorflow").strip().lower()
    probe = (
        _probe_external_torch_runtime(python_executable)
        if normalized == "pytorch"
        else _probe_external_tf_runtime(python_executable)
    )
    if bool(probe.get("available")) and int(probe.get("gpu_count") or 0) > 0:
        return None

    parts: list[str] = []
    version = str(probe.get("version") or "").strip()
    if version:
        runtime_name = "PyTorch" if normalized == "pytorch" else "TensorFlow"
        parts.append(f"{runtime_name} {version}")
    if bool(probe.get("available")):
        parts.append("no GPU device detected")
    else:
        parts.append("runtime unavailable")

    runtime_label = "PyTorch" if normalized == "pytorch" else "TensorFlow"
    message = f"External {runtime_label} runtime at {python_executable} is not ready for GPU training"
    if parts:
        message += f" ({'; '.join(parts)})"

    diagnostics = _condense_runtime_diagnostics(
        str(probe.get("error") or probe.get("diagnostics") or "")
    )
    if diagnostics:
        message += f". Details: {diagnostics}"
    return message


def _guard_standard_streams():
    stack = ExitStack()
    current_stdout = getattr(sys, "stdout", None)
    current_stderr = getattr(sys, "stderr", None)
    if current_stdout is None or not hasattr(current_stdout, "write"):
        stack.enter_context(redirect_stdout(io.StringIO()))
    if current_stderr is None or not hasattr(current_stderr, "write"):
        stack.enter_context(redirect_stderr(io.StringIO()))
    return stack


def _materialize_saved_model(tf_module, model, export_dir: Path, log_fn) -> tuple[bool, list[tuple[str, str]]]:
    attempts: list[tuple[str, str]] = []
    export_dir = Path(export_dir)

    exporters: list[tuple[str, object]] = [
        ("tf.saved_model.save", lambda: tf_module.saved_model.save(model, str(export_dir))),
    ]
    if hasattr(model, "export"):
        def _keras_export():
            try:
                model.export(str(export_dir), verbose=False)
            except TypeError:
                model.export(str(export_dir))

        exporters.append(("keras.model.export", _keras_export))

    for name, exporter in exporters:
        try:
            with _guard_standard_streams():
                exporter()
            log_fn(f"TFLite export fallback: materialized SavedModel at {export_dir} via {name}")
            return True, attempts
        except Exception as exc:
            details = traceback.format_exc(limit=12)
            attempts.append((f"saved_model_export:{name}", f"{exc}\n{details}"))
            log_fn(f"TFLite export fallback failed during SavedModel export ({name}): {exc}")
    return False, attempts


def detect_training_backends(external_python: Path | None = None, framework: str = "tensorflow") -> dict[str, object]:
    normalized = str(framework or "tensorflow").strip().lower()
    info: dict[str, object] = {
        "available": False,
        "backend": normalized,
        "version": None,
        "gpu_count": 0,
        "gpu_names": [],
        "cpu_count": 0,
        "cuda_built": False,
        "cuda_available": False,
        "mps_available": False,
        "error": None,
        "runtime_kind": "bundled",
        "runtime_path": None,
    }
    if external_python is not None:
        try:
            candidate = Path(external_python).expanduser().absolute()
        except Exception:
            candidate = Path(external_python).expanduser()
        info["runtime_kind"] = "external"
        info["runtime_path"] = str(candidate)
        if not candidate.exists() or not candidate.is_file():
            info["error"] = f"External python executable not found: {candidate}"
            return info
        if not is_probably_python_executable(candidate):
            info["error"] = (
                "Selected runtime does not look like a Python executable. "
                "Choose a python/python.exe interpreter explicitly."
            )
            return info
        if normalized == "pytorch":
            info.update(_probe_external_torch_runtime(candidate))
        else:
            info.update(_probe_external_tf_runtime(candidate))
        return info

    if normalized == "pytorch":
        try:
            import torch
        except Exception as exc:
            info["error"] = str(exc)
            return info

        info["available"] = True
        info["version"] = str(getattr(torch, "__version__", "unknown"))
        info["cpu_count"] = int(os.cpu_count() or 0)
        if bool(getattr(torch.cuda, "is_available", lambda: False)()):
            count = int(getattr(torch.cuda, "device_count", lambda: 0)() or 0)
            info["cuda_available"] = True
            info["gpu_count"] = count
            info["gpu_names"] = [str(torch.cuda.get_device_name(i)) for i in range(count)]
        else:
            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            if mps_backend is not None and bool(getattr(mps_backend, "is_available", lambda: False)()):
                info["mps_available"] = True
                info["gpu_count"] = 1
                info["gpu_names"] = ["Apple Metal (MPS)"]
        return info

    try:
        import tensorflow as tf
    except Exception as exc:
        info["error"] = str(exc)
        return info

    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    info["available"] = True
    info["version"] = str(getattr(tf, "__version__", "unknown"))
    info["gpu_count"] = len(gpus)
    info["gpu_names"] = [str(dev.name) for dev in gpus]
    info["cpu_count"] = len(cpus)
    info["cuda_built"] = bool(getattr(tf.test, "is_built_with_cuda", lambda: False)())
    info["cuda_available"] = bool(len(gpus) > 0)
    return info


def _load_keras_model_for_finetune(tf_module, model_path: Path):
    model_path = Path(model_path).expanduser()
    try:
        return tf_module.keras.models.load_model(str(model_path), compile=False)
    except Exception as exc:
        text = str(exc)
        if "Conv2DTranspose" not in text or "groups" not in text:
            raise

        class CompatConv2DTranspose(tf_module.keras.layers.Conv2DTranspose):
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
            return tf_module.keras.models.load_model(str(model_path), compile=False, custom_objects=custom_objects)
        except Exception as compat_exc:
            raise RuntimeError(
                "Fine-tune model deserialization failed and compatibility fallback did not recover it. "
                f"Original error: {exc} | Fallback error: {compat_exc}"
            ) from compat_exc


@dataclass(slots=True)
class TrainingConfig:
    framework: str = "tensorflow"  # tensorflow | pytorch
    patch_size: int = 256
    patch_stride: int = 128
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 1e-3
    val_split: float = 0.2
    max_patches_per_image: int = 96
    min_labeled_ratio: float = 0.0
    base_filters: int = 24
    dropout: float = 0.10
    device_mode: str = "auto"  # auto | gpu | cpu
    output_path: Path | None = None
    fine_tune_path: Path | None = None
    freeze_fraction: float = 0.0
    export_tflite: bool = False
    export_hef: bool = False
    hef_output_path: Path | None = None
    hailo_compile_command: str = ""
    random_seed: int = 1337
    augment_enabled: bool = True
    augment_replicas: int = 1
    aug_hflip_prob: float = 0.5
    aug_vflip_prob: float = 0.1
    aug_rotate90_prob: float = 0.5
    aug_brightness_delta: float = 0.15
    aug_contrast_range: float = 0.20
    aug_noise_std: float = 0.03
    aug_gamma_range: float = 0.20
    aug_blur_prob: float = 0.12
    external_python_path: Path | None = None
    external_fallback_cpu: bool = True


def _render_command_tokens(template: str, placeholders: dict[str, str]) -> list[str]:
    raw = str(template or "").strip()
    if not raw:
        raise RuntimeError("Hailo compile command is empty.")
    try:
        rendered = raw.format(**placeholders)
    except KeyError as exc:
        missing = str(exc).strip("'")
        raise RuntimeError(
            f"Hailo compile command is missing placeholder '{missing}'. "
            "Use placeholders like {tflite}, {hef}, {keras}, {output}."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Invalid Hailo compile command template: {exc}") from exc
    try:
        tokens = shlex.split(rendered, posix=(os.name != "nt"))
    except Exception as exc:
        raise RuntimeError(f"Unable to parse Hailo compile command: {exc}") from exc
    if not tokens:
        raise RuntimeError("Hailo compile command rendered to an empty command.")
    return tokens


def _with_last_index(limit: int, patch: int, stride: int) -> list[int]:
    if limit <= patch:
        return [0]
    stride = max(1, int(stride))
    starts = list(range(0, limit - patch + 1, stride))
    last = limit - patch
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def _ensure_min_shape(image: np.ndarray, mask: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h == 0 and pad_w == 0:
        return image, mask

    img_pad = np.pad(
        image,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect" if h > 1 and w > 1 else "edge",
    )
    mask_pad = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return img_pad, mask_pad


def _collect_patch_arrays(
    dataset_items: list[DatasetImageItem],
    annotations: dict[str, dict[int, np.ndarray]],
    classes: list[LabelClass],
    config: TrainingConfig,
    log_fn,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if not dataset_items:
        raise RuntimeError("No images available. Load and label a dataset first.")
    if not classes:
        raise RuntimeError("No classes available for training.")

    patch_size = max(32, int(config.patch_size))
    patch_stride = max(1, int(config.patch_stride))
    max_patches_per_image = max(1, int(config.max_patches_per_image))
    min_labeled_ratio = max(0.0, min(1.0, float(config.min_labeled_ratio)))
    rng = np.random.default_rng(config.random_seed)

    x_patches: list[np.ndarray] = []
    y_patches: list[np.ndarray] = []
    total_candidates = 0
    used_candidates = 0
    image_count = 0

    class_ids = [int(cls.class_id) for cls in classes]
    max_class_id = max([0] + class_ids)
    if max_class_id > 255:
        raise RuntimeError("Max class id > 255 is not supported.")

    for item in dataset_items:
        image = item.image
        h, w = image.shape[:2]
        layers = ensure_layer_map(annotations, item.uid, classes, (h, w))
        index_mask = index_mask_from_layers(layers, classes, (h, w))
        image, index_mask = _ensure_min_shape(image, index_mask, patch_size)
        hh, ww = image.shape[:2]

        ys = _with_last_index(hh, patch_size, patch_stride)
        xs = _with_last_index(ww, patch_size, patch_stride)
        coords = [(yy, xx) for yy in ys for xx in xs]
        total_candidates += len(coords)

        if len(coords) > max_patches_per_image:
            selected_idx = rng.choice(len(coords), size=max_patches_per_image, replace=False)
            coords = [coords[int(i)] for i in selected_idx]

        local_used = 0
        for y0, x0 in coords:
            mask_patch = index_mask[y0 : y0 + patch_size, x0 : x0 + patch_size]
            labeled_ratio = float((mask_patch > 0).mean())
            if labeled_ratio < min_labeled_ratio:
                continue
            img_patch = image[y0 : y0 + patch_size, x0 : x0 + patch_size]
            if img_patch.shape[0] != patch_size or img_patch.shape[1] != patch_size:
                continue

            x_patches.append(img_patch.astype(np.float32) / 255.0)
            y_patches.append(mask_patch.astype(np.uint8))
            local_used += 1

        if local_used == 0:
            cy = max(0, (hh - patch_size) // 2)
            cx = max(0, (ww - patch_size) // 2)
            x_patches.append(image[cy : cy + patch_size, cx : cx + patch_size].astype(np.float32) / 255.0)
            y_patches.append(index_mask[cy : cy + patch_size, cx : cx + patch_size].astype(np.uint8))
            local_used = 1

        used_candidates += local_used
        image_count += 1
        log_fn(f"Prepared patches for {item.name}: {local_used} patch(es).")

    if not x_patches:
        raise RuntimeError("No training patches generated. Lower min labeled ratio or check annotations.")

    x_arr = np.stack(x_patches, axis=0).astype(np.float32, copy=False)
    y_arr = np.stack(y_patches, axis=0).astype(np.uint8, copy=False)
    stats = {
        "num_images": image_count,
        "num_patches": int(x_arr.shape[0]),
        "patch_size": patch_size,
        "max_class_id": max_class_id,
        "total_candidates": total_candidates,
        "used_candidates": used_candidates,
    }
    return x_arr, y_arr, stats


def _box_blur3(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
        return image
    pad = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    out = (
        pad[:-2, :-2]
        + pad[:-2, 1:-1]
        + pad[:-2, 2:]
        + pad[1:-1, :-2]
        + pad[1:-1, 1:-1]
        + pad[1:-1, 2:]
        + pad[2:, :-2]
        + pad[2:, 1:-1]
        + pad[2:, 2:]
    ) / 9.0
    return out.astype(np.float32, copy=False)


def _augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    config: TrainingConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    img = image
    msk = mask

    if config.aug_hflip_prob > 0.0 and rng.random() < config.aug_hflip_prob:
        img = np.flip(img, axis=1)
        msk = np.flip(msk, axis=1)
    if config.aug_vflip_prob > 0.0 and rng.random() < config.aug_vflip_prob:
        img = np.flip(img, axis=0)
        msk = np.flip(msk, axis=0)
    if config.aug_rotate90_prob > 0.0 and rng.random() < config.aug_rotate90_prob:
        k = int(rng.integers(1, 4))
        img = np.rot90(img, k, axes=(0, 1))
        msk = np.rot90(msk, k, axes=(0, 1))

    img = img.astype(np.float32, copy=False)

    delta = float(config.aug_brightness_delta)
    if delta > 0.0:
        img = img + np.float32(rng.uniform(-delta, delta))

    contrast = float(config.aug_contrast_range)
    if contrast > 0.0:
        factor = np.float32(rng.uniform(max(0.1, 1.0 - contrast), 1.0 + contrast))
        mean = np.mean(img, axis=(0, 1), keepdims=True, dtype=np.float32)
        img = (img - mean) * factor + mean

    gamma_range = float(config.aug_gamma_range)
    if gamma_range > 0.0:
        gamma = float(rng.uniform(max(0.2, 1.0 - gamma_range), 1.0 + gamma_range))
        img = np.power(np.clip(img, 0.0, 1.0), np.float32(gamma))

    noise_std = float(config.aug_noise_std)
    if noise_std > 0.0:
        img = img + rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)

    if config.aug_blur_prob > 0.0 and rng.random() < config.aug_blur_prob:
        img = _box_blur3(img)

    img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
    return np.ascontiguousarray(img), np.ascontiguousarray(msk.astype(np.uint8, copy=False))


def _apply_training_augmentations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainingConfig,
    log_fn,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if not bool(config.augment_enabled):
        return x_train, y_train, {"augmented_patches": 0}

    replicas = max(0, int(config.augment_replicas))
    if replicas <= 0:
        return x_train, y_train, {"augmented_patches": 0}

    n = int(x_train.shape[0])
    if n <= 0:
        return x_train, y_train, {"augmented_patches": 0}

    rng = np.random.default_rng(int(config.random_seed) + 17)
    x_parts = [x_train]
    y_parts = [y_train]
    augmented_total = 0

    for _ in range(replicas):
        perm = rng.permutation(n)
        x_aug = np.empty_like(x_train)
        y_aug = np.empty_like(y_train)
        for i, src_idx in enumerate(perm):
            img_aug, mask_aug = _augment_pair(x_train[int(src_idx)], y_train[int(src_idx)], config, rng)
            x_aug[i] = img_aug
            y_aug[i] = mask_aug
        x_parts.append(x_aug)
        y_parts.append(y_aug)
        augmented_total += n

    x_out = np.concatenate(x_parts, axis=0).astype(np.float32, copy=False)
    y_out = np.concatenate(y_parts, axis=0).astype(np.uint8, copy=False)
    log_fn(
        f"Augmentation enabled: replicas={replicas}, train patches {n} -> {int(x_out.shape[0])} "
        f"(+{augmented_total})."
    )
    return x_out, y_out, {"augmented_patches": int(augmented_total)}


def _double_conv(x, filters: int, dropout: float):
    import tensorflow as tf

    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    return x


def _build_unet_model(input_shape: tuple[int, int, int], num_classes: int, base_filters: int, dropout: float):
    import tensorflow as tf

    inp = tf.keras.Input(shape=input_shape, name="image")

    c1 = _double_conv(inp, base_filters, dropout * 0.2)
    p1 = tf.keras.layers.MaxPool2D()(c1)

    c2 = _double_conv(p1, base_filters * 2, dropout * 0.4)
    p2 = tf.keras.layers.MaxPool2D()(c2)

    c3 = _double_conv(p2, base_filters * 4, dropout * 0.6)
    p3 = tf.keras.layers.MaxPool2D()(c3)

    bottleneck = _double_conv(p3, base_filters * 8, dropout)

    u3 = tf.keras.layers.UpSampling2D()(bottleneck)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c6 = _double_conv(u3, base_filters * 4, dropout * 0.6)

    u2 = tf.keras.layers.UpSampling2D()(c6)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c7 = _double_conv(u2, base_filters * 2, dropout * 0.4)

    u1 = tf.keras.layers.UpSampling2D()(c7)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c8 = _double_conv(u1, base_filters, dropout * 0.2)

    out = tf.keras.layers.Conv2D(num_classes, 1, padding="same", activation="softmax", name="mask")(c8)
    model = tf.keras.Model(inputs=inp, outputs=out, name="npec_unet")
    return model


def _try_export_tflite(tf_module, model, output_path: Path, log_fn) -> tuple[Path | None, str | None]:
    attempts: list[tuple[str, str]] = []
    output_path = Path(output_path).expanduser()
    tflite_path = output_path.with_suffix(".tflite")

    def _configure_converter(converter):
        try:
            converter.target_spec.supported_ops = [
                tf_module.lite.OpsSet.TFLITE_BUILTINS,
                tf_module.lite.OpsSet.SELECT_TF_OPS,
            ]
        except Exception:
            pass
        try:
            converter._experimental_lower_tensor_list_ops = False
        except Exception:
            pass
        try:
            converter.experimental_enable_resource_variables = True
        except Exception:
            pass
        try:
            converter.optimizations = []
        except Exception:
            pass
        return converter

    def _attempt(name: str, factory):
        try:
            converter = _configure_converter(factory())
            payload = converter.convert()
            tflite_path.write_bytes(payload)
            log_fn(f"Saved TFLite model: {tflite_path} ({name})")
            return tflite_path, None
        except Exception as exc:
            details = traceback.format_exc(limit=12)
            attempts.append((name, f"{exc}\n{details}"))
            log_fn(f"TFLite export attempt failed ({name}): {exc}")
            return None, str(exc)

    with tempfile.TemporaryDirectory(prefix="npec_tflite_export_") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        export_dir = tmp_dir / "saved_model"
        export_ok, export_attempts = _materialize_saved_model(tf_module, model, export_dir, log_fn)
        attempts.extend(export_attempts)

        if export_ok:
            path, _ = _attempt(
                "from_saved_model",
                lambda: tf_module.lite.TFLiteConverter.from_saved_model(str(export_dir)),
            )
            if path is not None:
                return path, None

            path, _ = _attempt(
                "from_keras_model",
                lambda: tf_module.lite.TFLiteConverter.from_keras_model(model),
            )
            if path is not None:
                return path, None

    error_lines = ["TFLite export failed after all converter attempts:"]
    for name, details in attempts:
        error_lines.append(f"[{name}] {details.strip()}")
    return None, "\n\n".join(error_lines)


def _configure_device(tf, mode: str, log_fn):
    mode = mode.lower().strip()
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    if mode == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        log_fn("Device mode: CPU only.")
        return "cpu"

    if mode == "gpu":
        if not gpus:
            raise RuntimeError("GPU mode selected, but no GPU device was detected by TensorFlow.")
        log_fn(f"Device mode: GPU only ({len(gpus)} GPU device(s)).")
        return "gpu"

    if gpus:
        log_fn(f"Device mode: AUTO -> using GPU ({len(gpus)} device(s)).")
        return "gpu"
    log_fn("Device mode: AUTO -> no GPU detected, using CPU.")
    return "cpu"


def _external_training_runner_source() -> str:
    return """import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


def _emit_log(message: str) -> None:
    print("__NPEC_LOG__" + str(message), flush=True)


def _emit_epoch(payload: dict) -> None:
    print("__NPEC_EPOCH__" + json.dumps(payload), flush=True)


def _load_keras_model_for_finetune(tf_module, model_path: Path):
    model_path = Path(model_path).expanduser()
    try:
        return tf_module.keras.models.load_model(str(model_path), compile=False)
    except Exception as exc:
        text = str(exc)
        if "Conv2DTranspose" not in text or "groups" not in text:
            raise

        class CompatConv2DTranspose(tf_module.keras.layers.Conv2DTranspose):
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
        return tf_module.keras.models.load_model(str(model_path), compile=False, custom_objects=custom_objects)


def _double_conv(tf_module, x, filters: int, dropout: float):
    x = tf_module.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf_module.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    if dropout > 0:
        x = tf_module.keras.layers.Dropout(dropout)(x)
    return x


def _build_unet_model(tf_module, input_shape: tuple[int, int, int], num_classes: int, base_filters: int, dropout: float):
    inp = tf_module.keras.Input(shape=input_shape, name="image")
    c1 = _double_conv(tf_module, inp, base_filters, dropout * 0.2)
    p1 = tf_module.keras.layers.MaxPool2D()(c1)
    c2 = _double_conv(tf_module, p1, base_filters * 2, dropout * 0.4)
    p2 = tf_module.keras.layers.MaxPool2D()(c2)
    c3 = _double_conv(tf_module, p2, base_filters * 4, dropout * 0.6)
    p3 = tf_module.keras.layers.MaxPool2D()(c3)
    bottleneck = _double_conv(tf_module, p3, base_filters * 8, dropout)
    u3 = tf_module.keras.layers.UpSampling2D()(bottleneck)
    u3 = tf_module.keras.layers.Concatenate()([u3, c3])
    c6 = _double_conv(tf_module, u3, base_filters * 4, dropout * 0.6)
    u2 = tf_module.keras.layers.UpSampling2D()(c6)
    u2 = tf_module.keras.layers.Concatenate()([u2, c2])
    c7 = _double_conv(tf_module, u2, base_filters * 2, dropout * 0.4)
    u1 = tf_module.keras.layers.UpSampling2D()(c7)
    u1 = tf_module.keras.layers.Concatenate()([u1, c1])
    c8 = _double_conv(tf_module, u1, base_filters, dropout * 0.2)
    out = tf_module.keras.layers.Conv2D(num_classes, 1, padding="same", activation="softmax", name="mask")(c8)
    return tf_module.keras.Model(inputs=inp, outputs=out, name="npec_unet")


def _guard_standard_streams():
    stack = ExitStack()
    current_stdout = getattr(sys, "stdout", None)
    current_stderr = getattr(sys, "stderr", None)
    if current_stdout is None or not hasattr(current_stdout, "write"):
        stack.enter_context(redirect_stdout(io.StringIO()))
    if current_stderr is None or not hasattr(current_stderr, "write"):
        stack.enter_context(redirect_stderr(io.StringIO()))
    return stack


def _materialize_saved_model(tf_module, model, export_dir: Path):
    attempts = []

    exporters = [
        ("tf.saved_model.save", lambda: tf_module.saved_model.save(model, str(export_dir))),
    ]
    if hasattr(model, "export"):
        def _keras_export():
            try:
                model.export(str(export_dir), verbose=False)
            except TypeError:
                model.export(str(export_dir))

        exporters.append(("keras.model.export", _keras_export))

    for name, exporter in exporters:
        try:
            with _guard_standard_streams():
                exporter()
            _emit_log(f"TFLite export fallback: materialized SavedModel at {export_dir} via {name}")
            return True, attempts
        except Exception as exc:
            details = traceback.format_exc(limit=12)
            attempts.append((f"saved_model_export:{name}", f"{exc}\\n{details}"))
            _emit_log(f"TFLite export fallback failed during SavedModel export ({name}): {exc}")
    return False, attempts


def _try_export_tflite(tf_module, model, output_path: Path):
    attempts = []
    output_path = Path(output_path).expanduser()
    tflite_path = output_path.with_suffix(".tflite")

    def _configure_converter(converter):
        try:
            converter.target_spec.supported_ops = [
                tf_module.lite.OpsSet.TFLITE_BUILTINS,
                tf_module.lite.OpsSet.SELECT_TF_OPS,
            ]
        except Exception:
            pass
        try:
            converter._experimental_lower_tensor_list_ops = False
        except Exception:
            pass
        try:
            converter.experimental_enable_resource_variables = True
        except Exception:
            pass
        return converter

    def _attempt(name, factory):
        try:
            converter = _configure_converter(factory())
            payload = converter.convert()
            tflite_path.write_bytes(payload)
            _emit_log(f"Saved TFLite model: {tflite_path} ({name})")
            return tflite_path, None
        except Exception as exc:
            details = traceback.format_exc(limit=12)
            attempts.append((name, f"{exc}\\n{details}"))
            _emit_log(f"TFLite export attempt failed ({name}): {exc}")
            return None, str(exc)

    with tempfile.TemporaryDirectory(prefix="npec_tflite_export_") as tmp_dir_raw:
        export_dir = Path(tmp_dir_raw) / "saved_model"
        export_ok, export_attempts = _materialize_saved_model(tf_module, model, export_dir)
        attempts.extend(export_attempts)

        if export_ok:
            path, _ = _attempt(
                "from_saved_model",
                lambda: tf_module.lite.TFLiteConverter.from_saved_model(str(export_dir)),
            )
            if path is not None:
                return path, None

            path, _ = _attempt(
                "from_keras_model",
                lambda: tf_module.lite.TFLiteConverter.from_keras_model(model),
            )
            if path is not None:
                return path, None

    error_lines = ["TFLite export failed after all converter attempts:"]
    for name, details in attempts:
        error_lines.append(f"[{name}] {details.strip()}")
    return None, "\\n\\n".join(error_lines)


def _configure_device(tf_module, mode: str):
    mode = str(mode).lower().strip()
    gpus = tf_module.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf_module.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    if mode == "cpu":
        try:
            tf_module.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        _emit_log("Device mode: CPU only.")
        return "cpu"
    if mode == "gpu":
        if not gpus:
            raise RuntimeError("GPU mode selected, but no GPU device was detected by TensorFlow.")
        _emit_log(f"Device mode: GPU only ({len(gpus)} GPU device(s)).")
        return "gpu"
    if gpus:
        _emit_log(f"Device mode: AUTO -> using GPU ({len(gpus)} device(s)).")
        return "gpu"
    _emit_log("Device mode: AUTO -> no GPU detected, using CPU.")
    return "cpu"


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--arrays", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    payload_path = Path(args.payload)
    arrays_path = Path(args.arrays)
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import tensorflow as tf

        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        arrays = np.load(arrays_path, allow_pickle=False)
        x_train = np.asarray(arrays["x_train"], dtype=np.float32)
        y_train = np.asarray(arrays["y_train"], dtype=np.uint8)
        has_val = bool(payload.get("has_val", False))
        x_val = np.asarray(arrays["x_val"], dtype=np.float32) if has_val else None
        y_val = np.asarray(arrays["y_val"], dtype=np.uint8) if has_val else None

        patch_size = int(payload["patch_size"])
        num_classes = int(payload["num_classes"])
        _configure_device(tf, str(payload.get("device_mode", "auto")))

        fine_tune_path = str(payload.get("fine_tune_path", "")).strip()
        if fine_tune_path:
            _emit_log(f"Loading base model for fine-tuning: {fine_tune_path}")
            model = _load_keras_model_for_finetune(tf, Path(fine_tune_path))
            out_ch = int(model.output_shape[-1])
            if out_ch != num_classes:
                raise RuntimeError(
                    f"Fine-tune model output channels ({out_ch}) do not match required channels ({num_classes})."
                )
            freeze_fraction = max(0.0, min(0.95, float(payload.get("freeze_fraction", 0.0))))
            if freeze_fraction > 0:
                freeze_count = int(round(len(model.layers) * freeze_fraction))
                for layer in model.layers[:freeze_count]:
                    layer.trainable = False
                _emit_log(f"Frozen first {freeze_count} layer(s) ({freeze_fraction:.0%}).")
        else:
            _emit_log("Building a new U-Net model.")
            model = _build_unet_model(
                tf,
                input_shape=(patch_size, patch_size, 3),
                num_classes=num_classes,
                base_filters=max(8, int(payload.get("base_filters", 24))),
                dropout=max(0.0, min(0.6, float(payload.get("dropout", 0.1)))),
            )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(payload.get("learning_rate", 1e-3))),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
                tf.keras.metrics.MeanIoU(num_classes=num_classes, sparse_y_true=True, sparse_y_pred=False, name="miou"),
            ],
        )

        total_epochs = max(1, int(payload.get("epochs", 1)))

        class EpochCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                out = {"epoch": int(epoch) + 1, "total_epochs": total_epochs}
                if logs:
                    for key, value in logs.items():
                        try:
                            out[str(key)] = float(value)
                        except Exception:
                            pass
                _emit_epoch(out)

        train_kwargs = {
            "x": x_train,
            "y": y_train,
            "batch_size": max(1, int(payload.get("batch_size", 8))),
            "epochs": total_epochs,
            "shuffle": True,
            "verbose": 0,
            "callbacks": [EpochCallback()],
        }
        if x_val is not None and y_val is not None and x_val.shape[0] > 0:
            train_kwargs["validation_data"] = (x_val, y_val)

        _emit_log(
            f"Training start: train patches={x_train.shape[0]}, val patches={0 if x_val is None else x_val.shape[0]}, "
            f"batch={train_kwargs['batch_size']}, epochs={train_kwargs['epochs']}"
        )
        history = model.fit(**train_kwargs)

        output_path = Path(str(payload.get("output_path", "npec_trained_model.keras"))).expanduser()
        if output_path.suffix.lower() not in {".keras", ".h5"}:
            output_path = output_path.with_suffix(".keras")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output_path))
        _emit_log(f"Saved trained model: {output_path}")

        tflite_path = None
        tflite_error = None
        if bool(payload.get("export_tflite", False)):
            tflite_path, tflite_error = _try_export_tflite(tf, model, output_path)
            if tflite_error:
                _emit_log(f"TFLite export failed: {tflite_error}")

        history_map = history.history if hasattr(history, "history") else {}
        final = {}
        for key, values in history_map.items():
            if isinstance(values, list) and values:
                try:
                    final[key] = float(values[-1])
                except Exception:
                    pass

        result = {
            "cancelled": False,
            "output_model": str(output_path),
            "output_tflite": None if tflite_path is None else str(tflite_path),
            "tflite_error": tflite_error,
            "history_final": final,
            "runtime": "external",
            "runtime_python": str(Path(os.sys.executable)),
        }
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return 0
    except Exception as exc:
        details = {"error": str(exc), "traceback": traceback.format_exc(limit=8)}
        try:
            result_path.write_text(json.dumps(details), encoding="utf-8")
        except Exception:
            pass
        _emit_log(str(exc))
        return 1


raise SystemExit(_main())
"""


def _external_pytorch_training_runner_source() -> str:
    return """import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np


def _emit_log(message: str) -> None:
    print("__NPEC_LOG__" + str(message), flush=True)


def _emit_epoch(payload: dict) -> None:
    print("__NPEC_EPOCH__" + json.dumps(payload), flush=True)


def _resolve_device(torch_module, mode: str):
    mode = str(mode).lower().strip()
    if mode == "cpu":
        _emit_log("Device mode: CPU only.")
        return torch_module.device("cpu"), "cpu"
    if bool(getattr(torch_module.cuda, "is_available", lambda: False)()):
        count = int(getattr(torch_module.cuda, "device_count", lambda: 0)() or 0)
        if mode in {"auto", "gpu"}:
            _emit_log(f"Device mode: {'GPU only' if mode == 'gpu' else 'AUTO -> using GPU'} ({count} CUDA device(s)).")
            return torch_module.device("cuda"), "cuda"
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps_backend is not None and bool(getattr(mps_backend, "is_available", lambda: False)()):
        if mode in {"auto", "gpu"}:
            _emit_log(f"Device mode: {'GPU only' if mode == 'gpu' else 'AUTO -> using GPU'} (Apple Metal MPS).")
            return torch_module.device("mps"), "mps"
    if mode == "gpu":
        raise RuntimeError("GPU mode selected, but no CUDA or Metal (MPS) device was detected by PyTorch.")
    _emit_log("Device mode: AUTO -> no GPU detected, using CPU.")
    return torch_module.device("cpu"), "cpu"


def _build_model(torch_module, in_channels: int, num_classes: int, base_filters: int, dropout: float):
    nn = torch_module.nn

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch, drop):
            super().__init__()
            layers = [
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

    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            bf = max(8, int(base_filters))
            self.enc1 = DoubleConv(in_channels, bf, dropout * 0.2)
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = DoubleConv(bf, bf * 2, dropout * 0.4)
            self.pool2 = nn.MaxPool2d(2)
            self.enc3 = DoubleConv(bf * 2, bf * 4, dropout * 0.6)
            self.pool3 = nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(bf * 4, bf * 8, dropout)
            self.up3 = nn.ConvTranspose2d(bf * 8, bf * 4, kernel_size=2, stride=2)
            self.dec3 = DoubleConv(bf * 8, bf * 4, dropout * 0.6)
            self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, kernel_size=2, stride=2)
            self.dec2 = DoubleConv(bf * 4, bf * 2, dropout * 0.4)
            self.up1 = nn.ConvTranspose2d(bf * 2, bf, kernel_size=2, stride=2)
            self.dec1 = DoubleConv(bf * 2, bf, dropout * 0.2)
            self.head = nn.Conv2d(bf, num_classes, kernel_size=1)

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

    return SmallUNet()


def _evaluate_model(torch_module, model, loader, criterion, device):
    if loader is None:
        return None, None
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    total_items = 0
    with torch_module.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.detach().cpu().item()) * int(xb.shape[0])
            preds = torch_module.argmax(logits, dim=1)
            total_correct += int((preds == yb).sum().detach().cpu().item())
            total_pixels += int(yb.numel())
            total_items += int(xb.shape[0])
    if total_items <= 0:
        return None, None
    return float(total_loss / total_items), (float(total_correct) / float(total_pixels)) if total_pixels > 0 else None


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--arrays", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    payload_path = Path(args.payload)
    arrays_path = Path(args.arrays)
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch

        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        arrays = np.load(arrays_path, allow_pickle=False)
        x_train = np.asarray(arrays["x_train"], dtype=np.float32)
        y_train = np.asarray(arrays["y_train"], dtype=np.int64)
        has_val = bool(payload.get("has_val", False))
        x_val = np.asarray(arrays["x_val"], dtype=np.float32) if has_val else None
        y_val = np.asarray(arrays["y_val"], dtype=np.int64) if has_val else None

        output_path = Path(str(payload.get("output_path", "npec_trained_model.pt"))).expanduser()
        if output_path.suffix.lower() not in {".pt", ".pth"}:
            output_path = output_path.with_suffix(".pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if str(payload.get("fine_tune_path", "")).strip():
            raise RuntimeError("PyTorch fine-tuning is not supported yet in the app runner.")
        if bool(payload.get("export_tflite", False)):
            _emit_log("PyTorch runner ignores TFLite/HEF export requests.")

        patch_size = int(payload["patch_size"])
        num_classes = int(payload["num_classes"])
        base_filters = max(8, int(payload.get("base_filters", 24)))
        dropout = max(0.0, min(0.6, float(payload.get("dropout", 0.1))))
        total_epochs = max(1, int(payload.get("epochs", 1)))
        batch_size = max(1, int(payload.get("batch_size", 8)))
        learning_rate = float(payload.get("learning_rate", 1e-3))

        device, device_label = _resolve_device(torch, str(payload.get("device_mode", "auto")))
        model = _build_model(torch, 3, num_classes, base_filters, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        x_train_t = torch.from_numpy(np.transpose(x_train, (0, 3, 1, 2))).float()
        y_train_t = torch.from_numpy(y_train).long()
        train_ds = torch.utils.data.TensorDataset(x_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = None
        if x_val is not None and y_val is not None and int(x_val.shape[0]) > 0:
            x_val_t = torch.from_numpy(np.transpose(x_val, (0, 3, 1, 2))).float()
            y_val_t = torch.from_numpy(y_val).long()
            val_ds = torch.utils.data.TensorDataset(x_val_t, y_val_t)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        _emit_log(
            f"Training start: train patches={x_train.shape[0]}, val patches={0 if x_val is None else x_val.shape[0]}, "
            f"batch={batch_size}, epochs={total_epochs}, runtime={device_label}"
        )

        final = {}
        for epoch in range(total_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_pixels = 0
            epoch_items = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu().item()) * int(xb.shape[0])
                preds = torch.argmax(logits.detach(), dim=1)
                epoch_correct += int((preds == yb).sum().detach().cpu().item())
                epoch_pixels += int(yb.numel())
                epoch_items += int(xb.shape[0])

            train_loss = float(epoch_loss / epoch_items) if epoch_items > 0 else 0.0
            train_acc = (float(epoch_correct) / float(epoch_pixels)) if epoch_pixels > 0 else 0.0
            val_loss, val_acc = _evaluate_model(torch, model, val_loader, criterion, device)

            out = {
                "epoch": int(epoch) + 1,
                "total_epochs": total_epochs,
                "loss": train_loss,
                "acc": train_acc,
            }
            if val_loss is not None:
                out["val_loss"] = float(val_loss)
            if val_acc is not None:
                out["val_acc"] = float(val_acc)
            _emit_epoch(out)
            final = out

        torch.save(
            {
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "patch_size": patch_size,
                "base_filters": base_filters,
                "dropout": dropout,
                "framework": "pytorch",
            },
            str(output_path),
        )
        _emit_log(f"Saved trained model: {output_path}")

        result = {
            "cancelled": False,
            "output_model": str(output_path),
            "output_tflite": None,
            "tflite_error": None,
            "history_final": final,
            "runtime": "external_pytorch",
            "runtime_python": str(Path(os.sys.executable)),
        }
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return 0
    except Exception as exc:
        details = {"error": str(exc), "traceback": traceback.format_exc(limit=8)}
        try:
            result_path.write_text(json.dumps(details), encoding="utf-8")
        except Exception:
            pass
        _emit_log(str(exc))
        return 1


raise SystemExit(_main())
"""


class TrainingWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    epoch_finished = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        dataset_items: list[DatasetImageItem],
        annotations: dict[str, dict[int, np.ndarray]],
        classes: list[LabelClass],
        config: TrainingConfig,
    ) -> None:
        super().__init__()
        self.dataset_items = dataset_items
        self.annotations = annotations
        self.classes = classes
        self.config = config
        self.cancel_requested = False

    @Slot()
    def cancel(self) -> None:
        self.cancel_requested = True
        self.log.emit("Cancellation requested. Stopping at next safe checkpoint...")

    @Slot()
    def run(self) -> None:
        try:
            summary = self._run_impl()
            self.finished.emit(summary)
        except Exception as exc:
            details = f"{exc}\n\n{traceback.format_exc(limit=6)}"
            self.failed.emit(details)

    def _prepare_training_arrays(self) -> tuple[dict[str, object], dict[str, object]]:
        x_arr, y_arr, stats = _collect_patch_arrays(
            self.dataset_items,
            self.annotations,
            self.classes,
            self.config,
            self.log.emit,
        )
        n = int(x_arr.shape[0])
        val_split = max(0.0, min(0.9, float(self.config.val_split)))
        val_count = int(round(n * val_split)) if n > 1 else 0
        if val_count >= n:
            val_count = max(0, n - 1)

        rng = np.random.default_rng(self.config.random_seed)
        indices = np.arange(n)
        rng.shuffle(indices)
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]
        if train_idx.size == 0:
            train_idx = indices
            val_idx = np.array([], dtype=np.int64)

        x_train = x_arr[train_idx]
        y_train = y_arr[train_idx]
        x_val = x_arr[val_idx] if val_idx.size > 0 else None
        y_val = y_arr[val_idx] if val_idx.size > 0 else None

        x_train, y_train, aug_stats = _apply_training_augmentations(
            x_train,
            y_train,
            self.config,
            self.log.emit,
        )
        stats["augmented_patches"] = int(aug_stats.get("augmented_patches", 0))
        stats["train_patches_final"] = int(x_train.shape[0])

        patch_size = int(x_train.shape[1])
        num_classes = int(stats["max_class_id"]) + 1
        if num_classes < 2:
            num_classes = 2

        if stats["max_class_id"] > 64:
            self.log.emit(
                f"Warning: max class id is {stats['max_class_id']}, so output channels = {num_classes}. "
                "Consider keeping class IDs compact for faster training."
            )

        prepared = {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "patch_size": patch_size,
            "num_classes": num_classes,
        }
        return prepared, stats

    def _resolve_external_python(self) -> Path | None:
        candidate = self.config.external_python_path
        if candidate is None:
            return None
        try:
            path = Path(candidate).expanduser().absolute()
        except Exception:
            path = Path(candidate).expanduser()
        if path.exists() and path.is_file():
            return path
        self.log.emit(f"External training python path is invalid ({path}). Using bundled runtime.")
        return None

    def _resolve_hef_output_path(self, output_model_path: Path | None) -> Path:
        candidate = self.config.hef_output_path
        if candidate is None:
            if output_model_path is None:
                return Path.cwd() / "npec_trained_model.hef"
            candidate = output_model_path.with_suffix(".hef")
        path = Path(candidate).expanduser()
        if path.suffix.lower() != ".hef":
            path = path.with_suffix(".hef")
        return path

    def _compile_hef_artifact(
        self,
        output_model_path: Path | None,
        tflite_path: Path | None,
    ) -> tuple[Path | None, str | None]:
        if not bool(self.config.export_hef):
            return None, None

        if tflite_path is None:
            return None, "HEF export requested but no TFLite artifact was produced."
        tflite_path = Path(tflite_path).expanduser()
        if not tflite_path.exists():
            return None, f"HEF export requested but TFLite file is missing: {tflite_path}"

        compile_template = str(self.config.hailo_compile_command or "").strip()
        if not compile_template:
            return (
                None,
                "HEF export requested but no Hailo compile command is configured. "
                "Set it in Training tab (example: hailomz compile ... {tflite} ... {hef}).",
            )

        hef_path = self._resolve_hef_output_path(output_model_path)
        hef_path.parent.mkdir(parents=True, exist_ok=True)
        keras_path = Path(output_model_path).expanduser() if output_model_path is not None else tflite_path.with_suffix(".keras")
        try:
            cmd = _render_command_tokens(
                compile_template,
                {
                    "tflite": str(tflite_path),
                    "hef": str(hef_path),
                    "output": str(hef_path),
                    "keras": str(keras_path),
                },
            )
        except Exception as exc:
            return None, str(exc)

        self.log.emit("Running Hailo compile command...")
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=dict(os.environ),
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or f"Hailo compile command failed with exit code {completed.returncode}."
            return None, detail
        if not hef_path.exists():
            return None, f"Hailo compile command finished but HEF file was not created: {hef_path}"
        self.log.emit(f"Saved HEF model: {hef_path}")
        return hef_path, None

    def _run_external_training(
        self,
        prepared: dict[str, object],
        stats: dict[str, object],
        external_python: Path,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory(prefix="npec_train_external_") as tmp_dir_raw:
            tmp_dir = Path(tmp_dir_raw)
            arrays_path = tmp_dir / "train_arrays.npz"
            payload_path = tmp_dir / "train_payload.json"
            result_path = tmp_dir / "train_result.json"
            runner_path = tmp_dir / "external_train_runner.py"

            x_train = np.asarray(prepared["x_train"], dtype=np.float32)
            y_train = np.asarray(prepared["y_train"], dtype=np.uint8)
            x_val = prepared.get("x_val")
            y_val = prepared.get("y_val")
            has_val = (
                isinstance(x_val, np.ndarray)
                and isinstance(y_val, np.ndarray)
                and x_val.size > 0
                and y_val.size > 0
            )

            np.savez_compressed(
                arrays_path,
                x_train=x_train,
                y_train=y_train,
                x_val=np.asarray(x_val, dtype=np.float32) if has_val else np.zeros((0,), dtype=np.float32),
                y_val=np.asarray(y_val, dtype=np.uint8) if has_val else np.zeros((0,), dtype=np.uint8),
            )

            output_path = self.config.output_path
            if output_path is None:
                output_path = Path.cwd() / "npec_trained_model.keras"
            if output_path.suffix.lower() not in {".keras", ".h5"}:
                output_path = output_path.with_suffix(".keras")

            payload = {
                "patch_size": int(prepared["patch_size"]),
                "num_classes": int(prepared["num_classes"]),
                "learning_rate": float(self.config.learning_rate),
                "batch_size": max(1, int(self.config.batch_size)),
                "epochs": max(1, int(self.config.epochs)),
                "base_filters": max(8, int(self.config.base_filters)),
                "dropout": max(0.0, min(0.6, float(self.config.dropout))),
                "device_mode": str(self.config.device_mode),
                "output_path": str(output_path),
                "fine_tune_path": "" if self.config.fine_tune_path is None else str(self.config.fine_tune_path),
                "freeze_fraction": float(self.config.freeze_fraction),
                "export_tflite": bool(self.config.export_tflite or self.config.export_hef),
                "has_val": bool(has_val),
            }
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
            runner_path.write_text(_external_training_runner_source(), encoding="utf-8")

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [
                str(external_python),
                str(runner_path),
                "--payload",
                str(payload_path),
                "--arrays",
                str(arrays_path),
                "--result",
                str(result_path),
            ]
            self.log.emit(f"Launching external training runtime: {external_python}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            try:
                if proc.stdout is not None:
                    for raw_line in proc.stdout:
                        line = raw_line.strip()
                        if not line:
                            if self.cancel_requested and proc.poll() is None:
                                proc.terminate()
                            continue
                        if line.startswith("__NPEC_EPOCH__"):
                            payload_txt = line[len("__NPEC_EPOCH__") :]
                            try:
                                epoch_payload = json.loads(payload_txt)
                            except Exception:
                                self.log.emit(f"[external] {line}")
                            else:
                                if isinstance(epoch_payload, dict):
                                    self.epoch_finished.emit(epoch_payload)
                                    self.progress.emit(
                                        int(epoch_payload.get("epoch", 0)),
                                        int(epoch_payload.get("total_epochs", max(1, int(self.config.epochs)))),
                                    )
                            if self.cancel_requested and proc.poll() is None:
                                proc.terminate()
                            continue
                        if line.startswith("__NPEC_LOG__"):
                            self.log.emit(line[len("__NPEC_LOG__") :].strip())
                        else:
                            self.log.emit(f"[external] {line}")
                        if self.cancel_requested and proc.poll() is None:
                            proc.terminate()
                rc = proc.wait(timeout=20)
            finally:
                if proc.poll() is None:
                    proc.kill()

            if self.cancel_requested:
                return {"cancelled": True, "stats": stats, "runtime": "external"}

            result_payload: dict[str, object] = {}
            if result_path.exists():
                try:
                    parsed = json.loads(result_path.read_text(encoding="utf-8"))
                    if isinstance(parsed, dict):
                        result_payload = parsed
                except Exception:
                    result_payload = {}

            if rc != 0:
                err_txt = str(result_payload.get("error", "")).strip()
                if not err_txt:
                    err_txt = f"External training failed with exit code {rc}."
                raise RuntimeError(err_txt)

            if result_payload.get("error"):
                raise RuntimeError(str(result_payload.get("error")))

            output_model_path: Path | None = None
            raw_model = result_payload.get("output_model")
            if raw_model:
                try:
                    output_model_path = Path(str(raw_model)).expanduser()
                except Exception:
                    output_model_path = None

            output_tflite_path: Path | None = None
            raw_tflite = result_payload.get("output_tflite")
            if raw_tflite:
                try:
                    output_tflite_path = Path(str(raw_tflite)).expanduser()
                except Exception:
                    output_tflite_path = None

            hef_path, hef_error = self._compile_hef_artifact(output_model_path, output_tflite_path)
            result_payload["output_hef"] = None if hef_path is None else str(hef_path)
            result_payload["hef_error"] = hef_error
            result_payload["stats"] = stats
            result_payload["cancelled"] = bool(result_payload.get("cancelled", False))
            return result_payload

    def _run_external_pytorch_training(
        self,
        prepared: dict[str, object],
        stats: dict[str, object],
        external_python: Path,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory(prefix="npec_train_external_torch_") as tmp_dir_raw:
            tmp_dir = Path(tmp_dir_raw)
            arrays_path = tmp_dir / "train_arrays.npz"
            payload_path = tmp_dir / "train_payload.json"
            result_path = tmp_dir / "train_result.json"
            runner_path = tmp_dir / "external_train_runner_pytorch.py"

            x_train = np.asarray(prepared["x_train"], dtype=np.float32)
            y_train = np.asarray(prepared["y_train"], dtype=np.uint8)
            x_val = prepared.get("x_val")
            y_val = prepared.get("y_val")
            has_val = (
                isinstance(x_val, np.ndarray)
                and isinstance(y_val, np.ndarray)
                and x_val.size > 0
                and y_val.size > 0
            )

            np.savez_compressed(
                arrays_path,
                x_train=x_train,
                y_train=y_train,
                x_val=np.asarray(x_val, dtype=np.float32) if has_val else np.zeros((0,), dtype=np.float32),
                y_val=np.asarray(y_val, dtype=np.uint8) if has_val else np.zeros((0,), dtype=np.uint8),
            )

            output_path = self.config.output_path
            if output_path is None:
                output_path = Path.cwd() / "npec_trained_model.pt"
            if output_path.suffix.lower() not in {".pt", ".pth"}:
                output_path = output_path.with_suffix(".pt")

            payload = {
                "patch_size": int(prepared["patch_size"]),
                "num_classes": int(prepared["num_classes"]),
                "learning_rate": float(self.config.learning_rate),
                "batch_size": max(1, int(self.config.batch_size)),
                "epochs": max(1, int(self.config.epochs)),
                "base_filters": max(8, int(self.config.base_filters)),
                "dropout": max(0.0, min(0.6, float(self.config.dropout))),
                "device_mode": str(self.config.device_mode),
                "output_path": str(output_path),
                "fine_tune_path": "" if self.config.fine_tune_path is None else str(self.config.fine_tune_path),
                "export_tflite": False,
                "has_val": bool(has_val),
            }
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
            runner_path.write_text(_external_pytorch_training_runner_source(), encoding="utf-8")

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [
                str(external_python),
                str(runner_path),
                "--payload",
                str(payload_path),
                "--arrays",
                str(arrays_path),
                "--result",
                str(result_path),
            ]
            self.log.emit(f"Launching external PyTorch runtime: {external_python}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            try:
                if proc.stdout is not None:
                    for raw_line in proc.stdout:
                        line = raw_line.strip()
                        if not line:
                            continue
                        if line.startswith("__NPEC_EPOCH__"):
                            payload_txt = line[len("__NPEC_EPOCH__") :]
                            try:
                                epoch_payload = json.loads(payload_txt)
                            except Exception:
                                self.log.emit(f"[external] {line}")
                            else:
                                if isinstance(epoch_payload, dict):
                                    self.epoch_finished.emit(epoch_payload)
                                    self.progress.emit(
                                        int(epoch_payload.get("epoch", 0)),
                                        int(epoch_payload.get("total_epochs", max(1, int(self.config.epochs)))),
                                    )
                            continue
                        if line.startswith("__NPEC_LOG__"):
                            self.log.emit(line[len("__NPEC_LOG__") :].strip())
                        else:
                            self.log.emit(f"[external] {line}")
                rc = proc.wait(timeout=20)
            finally:
                if proc.poll() is None:
                    proc.kill()

            result_payload: dict[str, object] = {}
            if result_path.exists():
                try:
                    parsed = json.loads(result_path.read_text(encoding="utf-8"))
                    if isinstance(parsed, dict):
                        result_payload = parsed
                except Exception:
                    result_payload = {}

            if rc != 0:
                err_txt = str(result_payload.get("error", "")).strip()
                if not err_txt:
                    err_txt = f"External PyTorch training failed with exit code {rc}."
                raise RuntimeError(err_txt)

            if result_payload.get("error"):
                raise RuntimeError(str(result_payload.get("error")))

            result_payload["output_tflite"] = None
            result_payload["tflite_error"] = None
            result_payload["output_hef"] = None
            result_payload["hef_error"] = None
            result_payload["stats"] = stats
            result_payload["cancelled"] = bool(result_payload.get("cancelled", False))
            return result_payload

    def _run_local_tf_training(
        self,
        prepared: dict[str, object],
        stats: dict[str, object],
        device_mode_override: str | None = None,
    ) -> dict[str, object]:
        try:
            import tensorflow as tf
        except Exception as exc:
            raise RuntimeError("TensorFlow is required for training. Check your installation.") from exc

        x_train = np.asarray(prepared["x_train"], dtype=np.float32)
        y_train = np.asarray(prepared["y_train"], dtype=np.uint8)
        x_val = prepared.get("x_val")
        y_val = prepared.get("y_val")
        patch_size = int(prepared["patch_size"])
        num_classes = int(prepared["num_classes"])

        _configure_device(tf, device_mode_override or self.config.device_mode, self.log.emit)

        model = None
        if self.config.fine_tune_path is not None:
            self.log.emit(f"Loading base model for fine-tuning: {self.config.fine_tune_path}")
            model = _load_keras_model_for_finetune(tf, self.config.fine_tune_path)
            out_ch = int(model.output_shape[-1])
            if out_ch != num_classes:
                raise RuntimeError(
                    f"Fine-tune model output channels ({out_ch}) do not match required channels ({num_classes})."
                )
            freeze_fraction = max(0.0, min(0.95, float(self.config.freeze_fraction)))
            if freeze_fraction > 0:
                freeze_count = int(round(len(model.layers) * freeze_fraction))
                for layer in model.layers[:freeze_count]:
                    layer.trainable = False
                self.log.emit(f"Frozen first {freeze_count} layer(s) ({freeze_fraction:.0%}).")
        else:
            self.log.emit("Building a new U-Net model.")
            model = _build_unet_model(
                input_shape=(patch_size, patch_size, 3),
                num_classes=num_classes,
                base_filters=max(8, int(self.config.base_filters)),
                dropout=max(0.0, min(0.6, float(self.config.dropout))),
            )

        optimizer = tf.keras.optimizers.Adam(learning_rate=float(self.config.learning_rate))
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
                tf.keras.metrics.MeanIoU(num_classes=num_classes, sparse_y_true=True, sparse_y_pred=False, name="miou"),
            ],
        )

        worker = self
        total_epochs = max(1, int(self.config.epochs))

        class QtTrainingCallback(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                if worker.cancel_requested:
                    self.model.stop_training = True

            def on_epoch_end(self, epoch, logs=None):
                payload = {"epoch": int(epoch) + 1, "total_epochs": total_epochs}
                if logs:
                    for key, value in logs.items():
                        try:
                            payload[str(key)] = float(value)
                        except Exception:
                            continue
                worker.epoch_finished.emit(payload)
                worker.progress.emit(int(epoch) + 1, total_epochs)

        callbacks = [QtTrainingCallback()]
        train_kwargs = {
            "x": x_train,
            "y": y_train,
            "batch_size": max(1, int(self.config.batch_size)),
            "epochs": max(1, int(self.config.epochs)),
            "shuffle": True,
            "verbose": 0,
            "callbacks": callbacks,
        }
        if x_val is not None and y_val is not None and x_val.shape[0] > 0:
            train_kwargs["validation_data"] = (x_val, y_val)

        self.log.emit(
            f"Training start: train patches={x_train.shape[0]}, val patches={0 if x_val is None else x_val.shape[0]}, "
            f"batch={train_kwargs['batch_size']}, epochs={train_kwargs['epochs']}"
        )

        history = model.fit(**train_kwargs)

        output_path = self.config.output_path
        if output_path is None:
            output_path = Path.cwd() / "npec_trained_model.keras"
        if output_path.suffix.lower() not in {".keras", ".h5"}:
            output_path = output_path.with_suffix(".keras")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output_path))
        self.log.emit(f"Saved trained model: {output_path}")

        tflite_path: Path | None = None
        tflite_error: str | None = None
        if self.config.export_tflite or self.config.export_hef:
            tflite_path, tflite_error = _try_export_tflite(tf, model, output_path, self.log.emit)
            if tflite_error:
                self.log.emit(f"TFLite export failed: {tflite_error}")

        hef_path, hef_error = self._compile_hef_artifact(output_path, tflite_path)
        if hef_error:
            self.log.emit(f"HEF export failed: {hef_error}")

        hist = history.history if hasattr(history, "history") else {}
        final = {}
        for key, values in hist.items():
            if isinstance(values, list) and values:
                try:
                    final[key] = float(values[-1])
                except Exception:
                    continue

        return {
            "cancelled": bool(self.cancel_requested),
            "stats": stats,
            "output_model": str(output_path),
            "output_tflite": None if tflite_path is None else str(tflite_path),
            "tflite_error": tflite_error,
            "output_hef": None if hef_path is None else str(hef_path),
            "hef_error": hef_error,
            "history_final": final,
            "runtime": "bundled",
            "runtime_python": str(Path(os.sys.executable)),
        }

    def _run_impl(self) -> dict[str, object]:
        prepared, stats = self._prepare_training_arrays()
        if self.cancel_requested:
            return {"cancelled": True, "stats": stats}

        framework = str(self.config.framework or "tensorflow").strip().lower()
        external_python = self._resolve_external_python()
        if framework == "pytorch":
            if self.config.fine_tune_path is not None:
                raise RuntimeError("PyTorch training in the app does not support fine-tuning from an existing checkpoint yet.")
            if bool(self.config.export_tflite or self.config.export_hef):
                self.log.emit("PyTorch training does not export TFLite/HEF yet; those options will be ignored.")
            if external_python is None:
                local_info = detect_training_backends(framework="pytorch")
                if bool(local_info.get("available")):
                    external_python = safe_current_python_executable()
                    if external_python is not None:
                        self.log.emit(f"Using current Python runtime for PyTorch training: {external_python}")
                    else:
                        raise RuntimeError(
                            "The packaged application executable is not a Python interpreter. "
                            "Set Runtime Python in the Training tab to a python/python.exe environment with PyTorch."
                        )
                else:
                    raise RuntimeError(
                        "PyTorch training requires a Python runtime with torch installed. "
                        "Set Runtime Python in the Training tab to a conda/env python that has PyTorch with CUDA or MPS."
                    )
            if str(self.config.device_mode or "auto").strip().lower() == "gpu":
                gpu_preflight_issue = _external_gpu_preflight_issue(external_python, framework="pytorch")
                if gpu_preflight_issue:
                    if bool(self.config.external_fallback_cpu):
                        self.log.emit(gpu_preflight_issue)
                        self.log.emit("Continuing with PyTorch on CPU because GPU-only runtime was not available.")
                        self.config.device_mode = "cpu"
                    else:
                        raise RuntimeError(gpu_preflight_issue)
            return self._run_external_pytorch_training(prepared, stats, external_python)

        if external_python is not None:
            if (
                str(self.config.device_mode or "auto").strip().lower() == "gpu"
                and bool(self.config.external_fallback_cpu)
            ):
                gpu_preflight_issue = _external_gpu_preflight_issue(external_python, framework="tensorflow")
                if gpu_preflight_issue:
                    self.log.emit(gpu_preflight_issue)
                    self.log.emit("Falling back to bundled TensorFlow runtime on CPU.")
                    return self._run_local_tf_training(prepared, stats, device_mode_override="cpu")
            try:
                return self._run_external_training(prepared, stats, external_python)
            except Exception as exc:
                self.log.emit(f"External TensorFlow training failed: {exc}")
                if not bool(self.config.external_fallback_cpu):
                    raise
                self.log.emit("Falling back to bundled TensorFlow runtime on CPU.")
                return self._run_local_tf_training(prepared, stats, device_mode_override="cpu")

        return self._run_local_tf_training(prepared, stats)
