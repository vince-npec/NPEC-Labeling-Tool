from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color as skcolor

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


PLANT_HEALTH_BINARY_CLASS_NAMES = ["healthy", "stressed"]


def plantvillage_binary_label(class_name: str) -> int:
    lowered = str(class_name).strip().lower()
    return 0 if "healthy" in lowered else 1


def plantvillage_binary_name(class_name: str) -> str:
    return PLANT_HEALTH_BINARY_CLASS_NAMES[plantvillage_binary_label(class_name)]


def find_plantvillage_imagefolder_root(root: str | Path) -> Path:
    base = Path(root).expanduser().resolve()
    candidates = [
        base / "segmented",
        base / "color",
        base / "PlantVillage",
        base,
    ]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        subdirs = [path for path in candidate.iterdir() if path.is_dir()]
        if not subdirs:
            continue
        image_count = 0
        for subdir in subdirs[:6]:
            image_count += sum(1 for child in subdir.iterdir() if child.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if image_count > 0:
            return candidate
    raise FileNotFoundError(f"Could not find a PlantVillage imagefolder root under: {base}")


def builtin_health_model_path() -> Path:
    return Path(__file__).resolve().parent / "builtin_models" / "plant_health" / "plantvillage_leaf_health_binary_mobilenet_v1.pt"


def _resize_nearest(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    image = Image.fromarray((np.asarray(mask, dtype=np.uint8) * 255))
    resized = image.resize((w, h), Image.Resampling.NEAREST)
    return np.asarray(resized > 0, dtype=bool)


def _normalize_u8(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _binary_open_close(mask: np.ndarray, open_size: int = 5, close_size: int = 9) -> np.ndarray:
    work = np.asarray(mask, dtype=bool)
    open_size = max(1, int(open_size))
    close_size = max(1, int(close_size))
    opened = ndi.binary_opening(work, structure=np.ones((open_size, open_size), dtype=bool))
    closed = ndi.binary_closing(opened, structure=np.ones((close_size, close_size), dtype=bool))
    return np.asarray(closed, dtype=bool)


def build_leaf_mask(
    image_rgb: np.ndarray,
    *,
    box: tuple[int, int, int, int] | None = None,
    positive_prompt: np.ndarray | None = None,
    negative_prompt: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Unsupported image shape: {tuple(image.shape)}")
    image = image[:, :, :3]
    h, w = image.shape[:2]

    work_mask = np.ones((h, w), dtype=bool)
    if box is not None:
        x, y, bw, bh = [int(v) for v in box]
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        bw = max(1, min(w - x, bw))
        bh = max(1, min(h - y, bh))
        work_mask[:] = False
        work_mask[y : y + bh, x : x + bw] = True

    rgb_f = image.astype(np.float32)
    r = rgb_f[:, :, 0]
    g = rgb_f[:, :, 1]
    b = rgb_f[:, :, 2]
    exg = (2.0 * g) - r - b
    exg_norm = _normalize_u8(exg)
    hsv = skcolor.rgb2hsv(image.astype(np.float32) / 255.0)
    hch = hsv[:, :, 0].astype(np.float32) * 179.0
    sch = hsv[:, :, 1].astype(np.float32) * 255.0
    vch = hsv[:, :, 2].astype(np.float32) * 255.0

    green_mask = (
        (exg_norm >= max(30, int(np.percentile(exg_norm[work_mask], 62)))) &
        (sch >= 18.0) &
        (vch >= 35.0) &
        (g >= (r * 0.85)) &
        (g >= (b * 0.85))
    )
    yellow_mask = (
        (hch >= 12.0) &
        (hch <= 42.0) &
        (sch >= 18.0) &
        (vch >= 45.0)
    )
    candidate = np.logical_and(np.logical_or(green_mask, yellow_mask), work_mask)
    candidate = _binary_open_close(candidate, 5, 9)

    if positive_prompt is not None:
        pos = np.asarray(positive_prompt, dtype=bool)
        if pos.shape != (h, w):
            pos = _resize_nearest(pos, (h, w))
        candidate = np.logical_or(candidate, pos)
    if negative_prompt is not None:
        neg = np.asarray(negative_prompt, dtype=bool)
        if neg.shape != (h, w):
            neg = _resize_nearest(neg, (h, w))
        candidate[neg] = False
    else:
        neg = np.zeros((h, w), dtype=bool)

    labels, count = ndi.label(candidate)
    if count > 1:
        keep = np.zeros((h, w), dtype=bool)
        if positive_prompt is not None and np.count_nonzero(pos) > 0:
            touched = np.unique(labels[pos])
            for label_id in touched:
                if int(label_id) <= 0:
                    continue
                keep |= labels == int(label_id)
        if np.count_nonzero(keep) <= 0:
            scores: list[tuple[float, int]] = []
            cx = (w - 1) * 0.5
            cy = (h - 1) * 0.5
            objects = ndi.find_objects(labels)
            for label_id in range(1, int(count) + 1):
                obj = objects[label_id - 1]
                if obj is None:
                    continue
                y_slice, x_slice = obj
                y = int(y_slice.start)
                x = int(x_slice.start)
                bh = int(y_slice.stop - y_slice.start)
                bw = int(x_slice.stop - x_slice.start)
                area = int(np.count_nonzero(labels == label_id))
                if int(area) <= 0:
                    continue
                ccx = x + (bw * 0.5)
                ccy = y + (bh * 0.5)
                distance = float(np.hypot(ccx - cx, ccy - cy))
                score = float(area) - (distance * 0.35)
                scores.append((score, int(label_id)))
            for _score, label_id in sorted(scores, reverse=True)[:4]:
                keep |= labels == int(label_id)
        candidate = keep

    candidate[neg] = False
    coverage = float(np.count_nonzero(candidate)) / float(max(1, candidate.size))
    meta = {
        "leaf_pixels": int(np.count_nonzero(candidate)),
        "leaf_coverage": float(coverage),
        "strategy": "color-leaf-mask",
    }
    return candidate.astype(bool), meta


def compute_yellowing_mask(image_rgb: np.ndarray, leaf_mask: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    image = image[:, :, :3]
    leaf = np.asarray(leaf_mask, dtype=bool)
    if leaf.shape != image.shape[:2]:
        leaf = _resize_nearest(leaf, image.shape[:2])
    hsv = skcolor.rgb2hsv(image.astype(np.float32) / 255.0)
    lab = skcolor.rgb2lab(image.astype(np.float32) / 255.0)
    hue = hsv[:, :, 0].astype(np.float32) * 179.0
    sat = hsv[:, :, 1].astype(np.float32) * 255.0
    val = hsv[:, :, 2].astype(np.float32) * 255.0
    bch = lab[:, :, 2].astype(np.float32)
    yellow = (
        (hue >= 12.0) &
        (hue <= 44.0) &
        (sat >= 20.0) &
        (val >= 45.0) &
        (bch >= np.percentile(bch[leaf], 56) if np.count_nonzero(leaf) > 0 else 150.0)
    )
    yellow &= leaf
    score = float(np.count_nonzero(yellow)) / float(max(1, np.count_nonzero(leaf)))
    return yellow.astype(bool), {
        "yellow_pixels": int(np.count_nonzero(yellow)),
        "yellowing_score": float(score),
    }


def compute_stress_cue_mask(
    image_rgb: np.ndarray,
    leaf_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    image = image[:, :, :3]
    leaf = np.asarray(leaf_mask, dtype=bool)
    if leaf.shape != image.shape[:2]:
        leaf = _resize_nearest(leaf, image.shape[:2])
    hsv = skcolor.rgb2hsv(image.astype(np.float32) / 255.0)
    lab = skcolor.rgb2lab(image.astype(np.float32) / 255.0)
    hue = hsv[:, :, 0].astype(np.float32) * 179.0
    sat = hsv[:, :, 1].astype(np.float32) * 255.0
    val = hsv[:, :, 2].astype(np.float32) * 255.0
    ach = lab[:, :, 1].astype(np.float32)
    bch = lab[:, :, 2].astype(np.float32)

    yellow_mask, yellow_meta = compute_yellowing_mask(image, leaf)
    brown_mask = (
        (hue >= 4.0) &
        (hue <= 25.0) &
        (sat >= 18.0) &
        (val >= 20.0) &
        (val <= 180.0)
    ) & leaf
    lesion_mask = (
        (ach >= (np.percentile(ach[leaf], 70) if np.count_nonzero(leaf) > 0 else 150.0)) &
        (bch >= (np.percentile(bch[leaf], 60) if np.count_nonzero(leaf) > 0 else 150.0)) &
        (sat >= 25.0)
    ) & leaf
    cue = yellow_mask | brown_mask | lesion_mask
    cue = _binary_open_close(cue, 3, 5)
    return cue.astype(bool), {
        "stress_pixels": int(np.count_nonzero(cue)),
        "stress_cue_score": float(np.count_nonzero(cue)) / float(max(1, np.count_nonzero(leaf))),
        "yellowing_score": float(yellow_meta["yellowing_score"]),
    }


def crop_mask_region(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    box: tuple[int, int, int, int] | None = None,
    pad_frac: float = 0.08,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    image = image[:, :, :3]
    h, w = image.shape[:2]
    work = np.asarray(mask, dtype=bool)
    if work.shape != (h, w):
        work = _resize_nearest(work, (h, w))
    if np.count_nonzero(work) <= 0:
        if box is not None:
            x, y, bw, bh = [int(v) for v in box]
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            bw = max(1, min(w - x, bw))
            bh = max(1, min(h - y, bh))
            return image[y : y + bh, x : x + bw].copy(), (x, y, bw, bh)
        return image.copy(), (0, 0, w, h)
    ys, xs = np.nonzero(work)
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    pad = int(round(max(4.0, max(x1 - x0, y1 - y0) * float(max(0.0, pad_frac)))))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    return image[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)


def create_plant_health_model(num_classes: int = 2, pretrained: bool = True):
    import torch.nn as nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = int(model.classifier[-1].in_features)
    model.classifier[-1] = nn.Linear(in_features, int(num_classes))
    return model


def build_health_train_transform(image_size: int = 224):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.04),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_health_eval_transform(image_size: int = 224):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def choose_torch_device(device_hint: str | None = None) -> str:
    import torch

    hint = str(device_hint or "").strip().lower()
    if hint in {"mps", "metal"} and torch.backends.mps.is_available():
        return "mps"
    if hint in {"cuda", "gpu"} and torch.cuda.is_available():
        return "cuda"
    if hint in {"cpu"}:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_health_checkpoint(model_path: str | Path, *, device: str | None = None):
    import torch

    resolved_device = choose_torch_device(device)
    checkpoint = torch.load(Path(model_path), map_location=resolved_device)
    num_classes = int(checkpoint.get("num_classes", 2))
    class_names = checkpoint.get("class_names", PLANT_HEALTH_BINARY_CLASS_NAMES)
    model = create_plant_health_model(num_classes=num_classes, pretrained=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    model.to(resolved_device)
    meta = dict(checkpoint) if isinstance(checkpoint, dict) else {}
    meta["class_names"] = list(class_names)
    meta["device"] = resolved_device
    return model, meta


def classify_health_crop(
    model,
    crop_rgb: np.ndarray,
    *,
    class_names: Iterable[str] | None = None,
    device: str | None = None,
) -> dict[str, object]:
    import torch
    from PIL import Image

    resolved_device = choose_torch_device(device)
    image = Image.fromarray(np.asarray(crop_rgb, dtype=np.uint8))
    tensor = build_health_eval_transform()(image).unsqueeze(0).to(resolved_device)
    model = model.to(resolved_device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    labels = list(class_names or PLANT_HEALTH_BINARY_CLASS_NAMES)
    top_index = int(np.argmax(probs))
    return {
        "class_names": labels,
        "probabilities": [float(value) for value in probs.tolist()],
        "predicted_index": int(top_index),
        "predicted_label": str(labels[top_index]) if top_index < len(labels) else str(top_index),
        "stress_probability": float(probs[min(len(labels) - 1, 1)]) if len(probs) > 1 else float(probs[0]),
    }
