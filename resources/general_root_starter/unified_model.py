from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image


TASK_KEYS = ("root_binary", "shoot", "primary_root", "lateral_root", "seed_crown")
DEFAULT_ROOT_ONLY_TASKS = ("root_binary", "primary_root", "lateral_root")
DEFAULT_TASK_LOSS_WEIGHTS = {
    "root_binary": 1.0,
    "shoot": 0.8,
    "primary_root": 0.6,
    "lateral_root": 0.6,
    "seed_crown": 0.4,
}
DEFAULT_FOCUS_TASK_WEIGHTS = {
    "root_binary": 1.0,
    "shoot": 1.6,
    "primary_root": 0.7,
    "lateral_root": 0.5,
    "seed_crown": 0.4,
}


@dataclass(slots=True)
class CurriculumStage:
    name: str
    epochs: int
    families: tuple[str, ...]
    task_loss_weights: dict[str, float]
    focus_tasks: tuple[str, ...] = ("root_binary", "shoot")
    positive_crop_bias: float = 0.95
    repeat_factor: int | None = None


def _resolve_tasks(tasks: Iterable[str] | None) -> tuple[str, ...]:
    if tasks is None:
        return tuple(TASK_KEYS)
    resolved: list[str] = []
    for raw in tasks:
        task = str(raw or "").strip()
        if task in TASK_KEYS and task not in resolved:
            resolved.append(task)
    return tuple(resolved) or tuple(TASK_KEYS)


def _default_focus_tasks(tasks: Iterable[str]) -> tuple[str, ...]:
    resolved = _resolve_tasks(tasks)
    if "shoot" in resolved:
        ordered = ("root_binary", "shoot", "primary_root", "lateral_root", "seed_crown")
    else:
        ordered = ("root_binary", "primary_root", "lateral_root", "seed_crown")
    focus = tuple(task for task in ordered if task in resolved)
    return focus or resolved


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for the unified General Root Starter model. "
            "Run this with a Python runtime that has torch installed."
        ) from exc
    return torch, nn, F, Dataset, DataLoader


def load_corpus_manifest(corpus_dir: Path) -> dict[str, object]:
    manifest_path = Path(corpus_dir).expanduser().resolve() / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _read_masks(path: Path) -> dict[str, np.ndarray]:
    mask_path = Path(path)
    if mask_path.suffix.lower() == ".npz":
        payload = np.load(mask_path)
        return {key: np.asarray(payload[key], dtype=np.uint8) for key in payload.files}
    gray = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    white_fraction = float(np.count_nonzero(gray >= 128)) / float(max(1, gray.size))
    # Public root datasets are not consistent about foreground polarity.
    # Use the minority binary value as the foreground class so PRMI (white roots on black)
    # and future black-on-white exports can share the same direct-mask path.
    if white_fraction > 0.5:
        root_binary = (gray < 128).astype(np.uint8)
    else:
        root_binary = (gray >= 128).astype(np.uint8)
    return {"root_binary": root_binary}


def _select_device(torch_module, mode: str = "auto"):
    normalized = str(mode or "auto").strip().lower()
    if normalized == "cpu":
        return torch_module.device("cpu"), "cpu"
    if bool(getattr(torch_module.cuda, "is_available", lambda: False)()):
        return torch_module.device("cuda"), "cuda"
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps_backend is not None and bool(getattr(mps_backend, "is_available", lambda: False)()):
        return torch_module.device("mps"), "mps"
    if normalized == "gpu":
        raise RuntimeError("GPU mode requested, but neither CUDA nor Apple Metal (MPS) is available.")
    return torch_module.device("cpu"), "cpu"


class _UnifiedRootDatasetBase:
    def __init__(
        self,
        samples: list[dict[str, object]],
        patch_size: int,
        *,
        augment: bool,
        repeat_factor: int = 1,
        random_seed: int = 17,
        focus_tasks: Iterable[str] | None = None,
        positive_crop_bias: float = 0.95,
        focus_task_weights: dict[str, float] | None = None,
    ):
        self.samples = list(samples)
        self.patch_size = max(64, int(patch_size))
        self.augment = bool(augment)
        self.repeat_factor = max(1, int(repeat_factor))
        self.rng = random.Random(int(random_seed))
        self.focus_tasks = tuple(str(task) for task in (focus_tasks or ("root_binary", "shoot")) if str(task))
        self.positive_crop_bias = max(0.0, min(1.0, float(positive_crop_bias)))
        self.focus_task_weights = dict(DEFAULT_FOCUS_TASK_WEIGHTS)
        if isinstance(focus_task_weights, dict):
            for key, value in focus_task_weights.items():
                if key in self.focus_task_weights:
                    try:
                        self.focus_task_weights[key] = max(0.0, float(value))
                    except Exception:
                        continue

    def __len__(self) -> int:
        return len(self.samples) * self.repeat_factor

    def _pick_focus_mask(self, masks: dict[str, np.ndarray]) -> np.ndarray | None:
        weighted: list[tuple[float, np.ndarray]] = []
        for task in self.focus_tasks:
            mask = np.asarray(masks.get(task), dtype=np.uint8)
            if np.count_nonzero(mask) <= 0:
                continue
            weighted.append((float(self.focus_task_weights.get(task, 1.0)), mask))
        if weighted:
            total = sum(weight for weight, _mask in weighted)
            if total > 0:
                pick = self.rng.random() * total
                running = 0.0
                for weight, mask in weighted:
                    running += weight
                    if pick <= running:
                        return mask
        fallback = None
        for task in TASK_KEYS:
            mask = np.asarray(masks.get(task), dtype=np.uint8)
            if np.count_nonzero(mask) <= 0:
                continue
            if fallback is None:
                fallback = (mask > 0).astype(np.uint8)
            else:
                fallback = np.maximum(fallback, (mask > 0).astype(np.uint8))
        return fallback

    def _crop_bounds(self, image_hw: tuple[int, int], masks: dict[str, np.ndarray]) -> tuple[int, int, int, int]:
        h, w = image_hw
        patch = self.patch_size
        focus_mask = self._pick_focus_mask(masks)
        focus_nonzero = 0 if focus_mask is None else int(np.count_nonzero(focus_mask))
        use_positive_focus = focus_nonzero > 0 and self.rng.random() < self.positive_crop_bias
        center_y = None
        center_x = None
        if use_positive_focus and focus_mask is not None:
            ys, xs = np.nonzero(focus_mask > 0)
            if len(ys) > 0:
                pick_idx = self.rng.randrange(len(ys))
                center_y = int(ys[pick_idx])
                center_x = int(xs[pick_idx])
        if h <= patch:
            top = 0
        elif center_y is not None:
            top = max(0, min(h - patch, center_y - patch // 2))
        else:
            top = self.rng.randint(0, h - patch)
        if w <= patch:
            left = 0
        elif center_x is not None:
            left = max(0, min(w - patch, center_x - patch // 2))
        else:
            left = self.rng.randint(0, w - patch)
        bottom = min(h, top + patch)
        right = min(w, left + patch)
        return top, bottom, left, right

    def _pad_crop(self, arr: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
        crop = np.asarray(arr[top:bottom, left:right])
        patch = self.patch_size
        pad_h = max(0, patch - crop.shape[0])
        pad_w = max(0, patch - crop.shape[1])
        if arr.ndim == 3:
            if pad_h > 0 or pad_w > 0:
                crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        else:
            if pad_h > 0 or pad_w > 0:
                crop = np.pad(crop, ((0, pad_h), (0, pad_w)), mode="constant")
        return crop

    def _augment_pair(self, image: np.ndarray, targets: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if not self.augment:
            return image, targets
        if self.rng.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            targets = {key: np.flip(value, axis=1).copy() for key, value in targets.items()}
        if self.rng.random() < 0.1:
            image = np.flip(image, axis=0).copy()
            targets = {key: np.flip(value, axis=0).copy() for key, value in targets.items()}
        if self.rng.random() < 0.2:
            delta = self.rng.uniform(-0.08, 0.08)
            img = image.astype(np.float32) / 255.0
            img = np.clip(img + delta, 0.0, 1.0)
            image = (img * 255.0).astype(np.uint8)
        if self.rng.random() < 0.2:
            gain = self.rng.uniform(0.85, 1.20)
            img = image.astype(np.float32) / 255.0
            img = np.clip((img - 0.5) * gain + 0.5, 0.0, 1.0)
            image = (img * 255.0).astype(np.uint8)
        if self.rng.random() < 0.12:
            noise = self.rng.normalvariate(0.0, 0.02)
            img = image.astype(np.float32) / 255.0
            img = np.clip(img + noise, 0.0, 1.0)
            image = (img * 255.0).astype(np.uint8)
        return image, targets


def _build_dataset_class():
    torch, _nn, _F, Dataset, _DataLoader = _require_torch()

    class GeneralRootStarterDataset(_UnifiedRootDatasetBase, Dataset):
        def __getitem__(self, index: int):
            sample = self.samples[int(index) % len(self.samples)]
            image = _read_rgb(Path(str(sample["image_path"])))
            masks = _read_masks(Path(str(sample["mask_path"])))
            full_masks = {
                key: np.asarray(masks.get(key, np.zeros(image.shape[:2], dtype=np.uint8)), dtype=np.uint8)
                for key in TASK_KEYS
            }
            top, bottom, left, right = self._crop_bounds(image.shape[:2], full_masks)
            image_crop = self._pad_crop(image, top, bottom, left, right)
            target_crops = {
                key: self._pad_crop(np.asarray(full_masks[key], dtype=np.uint8), top, bottom, left, right)
                for key in TASK_KEYS
            }
            image_crop, target_crops = self._augment_pair(image_crop, target_crops)
            family = str(sample.get("family") or "unknown")
            tasks_available = sample.get("tasks_available") if isinstance(sample.get("tasks_available"), dict) else {}
            tensor_image = torch.from_numpy(np.transpose(image_crop.astype(np.float32) / 255.0, (2, 0, 1))).float()
            targets = {key: torch.from_numpy((value > 0).astype(np.float32))[None, ...] for key, value in target_crops.items()}
            availability = {
                key: torch.tensor(float(bool(tasks_available.get(key, False))), dtype=torch.float32)
                for key in TASK_KEYS
            }
            return {
                "image": tensor_image,
                "targets": targets,
                "availability": availability,
                "family": family,
                "sample_id": str(sample.get("sample_id") or ""),
            }

    return GeneralRootStarterDataset


def _build_unified_model_class():
    torch, nn, _F, _Dataset, _DataLoader = _require_torch()

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
            super().__init__()
            layers = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            return self.block(x)

    class UnifiedRootStarterNet(nn.Module):
        def __init__(self, tasks: Iterable[str], families: Iterable[str], in_channels: int = 3, base_filters: int = 24, dropout: float = 0.10):
            super().__init__()
            tasks = tuple(str(task) for task in tasks)
            families = tuple(str(family) for family in families)
            self.tasks = tasks
            self.families = families
            bf = max(8, int(base_filters))
            self.enc1 = ConvBlock(in_channels, bf, dropout=dropout * 0.25)
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = ConvBlock(bf, bf * 2, dropout=dropout * 0.5)
            self.pool2 = nn.MaxPool2d(2)
            self.enc3 = ConvBlock(bf * 2, bf * 4, dropout=dropout * 0.75)
            self.pool3 = nn.MaxPool2d(2)
            self.bottleneck = ConvBlock(bf * 4, bf * 8, dropout=dropout)
            self.up3 = nn.ConvTranspose2d(bf * 8, bf * 4, 2, stride=2)
            self.dec3 = ConvBlock(bf * 8, bf * 4, dropout=dropout * 0.75)
            self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
            self.dec2 = ConvBlock(bf * 4, bf * 2, dropout=dropout * 0.5)
            self.up1 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
            self.dec1 = ConvBlock(bf * 2, bf, dropout=dropout * 0.25)
            self.heads = nn.ModuleDict({task: nn.Conv2d(bf, 1, 1) for task in tasks})
            self.family_head = nn.Linear(bf * 8, len(families)) if len(families) > 1 else None

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            b = self.bottleneck(self.pool3(e3))
            d3 = self.up3(b)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            outputs = {task: self.heads[task](d1) for task in self.tasks}
            if self.family_head is not None:
                pooled = torch.mean(b, dim=(2, 3))
                outputs["family_logits"] = self.family_head(pooled)
            return outputs

    return UnifiedRootStarterNet


def _binary_iou(pred: np.ndarray, truth: np.ndarray) -> float | None:
    pred_b = np.asarray(pred, dtype=np.uint8) > 0
    truth_b = np.asarray(truth, dtype=np.uint8) > 0
    union = float(np.count_nonzero(np.logical_or(pred_b, truth_b)))
    if union <= 0:
        return None
    inter = float(np.count_nonzero(np.logical_and(pred_b, truth_b)))
    return inter / union


def _sample_task_loss(F, logits, target, availability):
    probs = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(1, 2, 3))
    probs_flat = probs.flatten(1)
    target_flat = target.flatten(1)
    intersection = (probs_flat * target_flat).sum(dim=1)
    denom = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0))
    per_sample = (0.5 * bce) + (0.5 * dice)
    weights = availability.view(-1)
    denom = float(weights.sum().item())
    if denom <= 0:
        return None
    return (per_sample * weights).sum() / weights.sum()


def _resolve_task_loss_weights(task_loss_weights: dict[str, float] | None) -> dict[str, float]:
    resolved = dict(DEFAULT_TASK_LOSS_WEIGHTS)
    if isinstance(task_loss_weights, dict):
        for key, value in task_loss_weights.items():
            if key in resolved:
                try:
                    resolved[key] = max(0.0, float(value))
                except Exception:
                    continue
    return resolved


def _family_index_map(samples: list[dict[str, object]]) -> tuple[list[str], dict[str, int]]:
    families = sorted({str(sample.get("family") or "unknown") for sample in samples})
    return families, {family: idx for idx, family in enumerate(families)}


def _selection_score(task_means: dict[str, float], tasks: Iterable[str]) -> float:
    resolved = _resolve_tasks(tasks)
    root_score = float(task_means.get("root_binary", 0.0))
    shoot_score = float(task_means.get("shoot", 0.0))
    if "shoot" in resolved:
        return float((0.7 * root_score) + (0.3 * shoot_score))
    if "root_binary" in resolved:
        return root_score
    scores = [float(task_means.get(task, 0.0)) for task in resolved]
    return float(np.mean(scores)) if scores else 0.0


def _resolve_stage_weights(weights: dict[str, float] | None) -> dict[str, float]:
    return _resolve_task_loss_weights(weights)


def _build_family_balanced_sampler(torch_module, stage_train_samples: list[dict[str, object]], dataset_length: int):
    if not stage_train_samples or dataset_length <= 0:
        return None
    family_counts = Counter(str(sample.get("family") or "unknown") for sample in stage_train_samples)
    if len(family_counts) <= 1:
        return None
    base_weights = [
        1.0 / float(family_counts[str(sample.get("family") or "unknown")])
        for sample in stage_train_samples
    ]
    repeated_weights = [float(base_weights[idx % len(base_weights)]) for idx in range(dataset_length)]
    return torch_module.utils.data.WeightedRandomSampler(
        weights=torch_module.tensor(repeated_weights, dtype=torch_module.double),
        num_samples=int(dataset_length),
        replacement=True,
    )


def _build_default_curriculum_stages(
    epochs: int,
    families: list[str],
    *,
    tasks: tuple[str, ...],
    repeat_factor: int,
    task_loss_weights: dict[str, float],
    positive_crop_bias: float,
) -> list[CurriculumStage]:
    total_epochs = max(1, int(epochs))
    all_families = tuple(str(family) for family in families)
    plate_families = tuple(family for family in all_families if family != "minirhizotron_public") or all_families
    resolved_tasks = _resolve_tasks(tasks)
    focus_tasks = _default_focus_tasks(resolved_tasks)
    core_weights = {
        "root_binary": max(1.0, float(task_loss_weights.get("root_binary", 1.0))) if "root_binary" in resolved_tasks else 0.0,
        "shoot": max(0.9, float(task_loss_weights.get("shoot", 0.8))) if "shoot" in resolved_tasks else 0.0,
        "primary_root": 0.0,
        "lateral_root": 0.0,
        "seed_crown": 0.0,
    }
    aux_warmup_weights = {
        "root_binary": max(1.0, float(task_loss_weights.get("root_binary", 1.0))) if "root_binary" in resolved_tasks else 0.0,
        "shoot": max(0.9, float(task_loss_weights.get("shoot", 0.8))) if "shoot" in resolved_tasks else 0.0,
        "primary_root": min(float(task_loss_weights.get("primary_root", 0.6)), 0.18) if "primary_root" in resolved_tasks else 0.0,
        "lateral_root": min(float(task_loss_weights.get("lateral_root", 0.6)), 0.18) if "lateral_root" in resolved_tasks else 0.0,
        "seed_crown": min(float(task_loss_weights.get("seed_crown", 0.4)), 0.08) if "seed_crown" in resolved_tasks else 0.0,
    }
    full_weights = {
        task: (float(task_loss_weights.get(task, 0.0)) if task in resolved_tasks else 0.0)
        for task in TASK_KEYS
    }

    if total_epochs == 1:
        return [
            CurriculumStage(
                name="all_domain_core",
                epochs=1,
                families=all_families,
                task_loss_weights=core_weights,
                focus_tasks=focus_tasks,
                positive_crop_bias=positive_crop_bias,
                repeat_factor=repeat_factor,
            )
        ]
    if total_epochs == 2:
        return [
            CurriculumStage(
                name="plate_core",
                epochs=1,
                families=plate_families,
                task_loss_weights=core_weights,
                focus_tasks=focus_tasks,
                positive_crop_bias=positive_crop_bias,
                repeat_factor=repeat_factor,
            ),
            CurriculumStage(
                name="all_domain_full",
                epochs=1,
                families=all_families,
                task_loss_weights=full_weights,
                focus_tasks=focus_tasks,
                positive_crop_bias=positive_crop_bias,
                repeat_factor=repeat_factor,
            ),
        ]
    if total_epochs == 3:
        return [
            CurriculumStage(
                name="plate_core",
                epochs=1,
                families=plate_families,
                task_loss_weights=core_weights,
                focus_tasks=focus_tasks,
                positive_crop_bias=positive_crop_bias,
                repeat_factor=repeat_factor,
            ),
            CurriculumStage(
                name="all_domain_core",
                epochs=1,
                families=all_families,
                task_loss_weights=core_weights,
                focus_tasks=focus_tasks,
                positive_crop_bias=positive_crop_bias,
                repeat_factor=repeat_factor,
            ),
            CurriculumStage(
                name="all_domain_full",
                epochs=1,
                families=all_families,
                task_loss_weights=full_weights,
                focus_tasks=focus_tasks,
                positive_crop_bias=positive_crop_bias,
                repeat_factor=repeat_factor,
            ),
        ]

    fractions = [0.30, 0.25, 0.25, 0.20]
    raw_counts = [max(1, int(round(total_epochs * frac))) for frac in fractions]
    while sum(raw_counts) > total_epochs:
        reducible = [idx for idx, count in enumerate(raw_counts) if count > 1]
        if not reducible:
            break
        largest_idx = max(reducible, key=lambda idx: raw_counts[idx])
        raw_counts[largest_idx] -= 1
    while sum(raw_counts) < total_epochs:
        raw_counts[-1] += 1

    return [
        CurriculumStage(
            name="plate_core",
            epochs=int(raw_counts[0]),
            families=plate_families,
            task_loss_weights=core_weights,
            focus_tasks=focus_tasks,
            positive_crop_bias=positive_crop_bias,
            repeat_factor=repeat_factor,
        ),
        CurriculumStage(
            name="all_domain_core",
            epochs=int(raw_counts[1]),
            families=all_families,
            task_loss_weights=core_weights,
            focus_tasks=focus_tasks,
            positive_crop_bias=positive_crop_bias,
            repeat_factor=repeat_factor,
        ),
        CurriculumStage(
            name="all_domain_aux_warmup",
            epochs=int(raw_counts[2]),
            families=all_families,
            task_loss_weights=aux_warmup_weights,
            focus_tasks=focus_tasks,
            positive_crop_bias=positive_crop_bias,
            repeat_factor=repeat_factor,
        ),
        CurriculumStage(
            name="all_domain_full",
            epochs=int(raw_counts[3]),
            families=all_families,
            task_loss_weights=full_weights,
            focus_tasks=focus_tasks,
            positive_crop_bias=positive_crop_bias,
            repeat_factor=repeat_factor,
        ),
    ]


def train_unified_general_root_starter(
    corpus_dir: Path,
    output_path: Path,
    *,
    epochs: int = 24,
    batch_size: int = 8,
    patch_size: int = 256,
    base_filters: int = 24,
    dropout: float = 0.10,
    learning_rate: float = 1e-3,
    device_mode: str = "auto",
    repeat_factor: int = 4,
    random_seed: int = 17,
    weight_decay: float = 1e-4,
    family_loss_weight: float = 0.05,
    early_stopping_patience: int = 3,
    early_stopping_min_delta: float = 1e-4,
    task_loss_weights: dict[str, float] | None = None,
    curriculum_mode: str = "auto",
    positive_crop_bias: float = 0.95,
    tasks: Iterable[str] | None = None,
) -> dict[str, object]:
    torch, _nn, F, _Dataset, DataLoader = _require_torch()
    GeneralRootStarterDataset = _build_dataset_class()
    UnifiedRootStarterNet = _build_unified_model_class()

    manifest = load_corpus_manifest(corpus_dir)
    samples = list(manifest.get("samples") or [])
    train_samples = [sample for sample in samples if str(sample.get("split")) == "train"]
    val_samples = [sample for sample in samples if str(sample.get("split")) == "val"]
    if not train_samples:
        raise ValueError("Corpus manifest has no train split samples.")
    if not val_samples:
        val_samples = train_samples[: min(len(train_samples), max(1, len(train_samples) // 5))]
    families, family_to_idx = _family_index_map(samples)
    resolved_task_loss_weights = _resolve_task_loss_weights(task_loss_weights)
    resolved_tasks = _resolve_tasks(tasks)
    normalized_curriculum_mode = str(curriculum_mode or "auto").strip().lower()
    if normalized_curriculum_mode == "none":
        curriculum_stages = [
            CurriculumStage(
                name="all_domain_full",
                epochs=max(1, int(epochs)),
                families=tuple(families),
                task_loss_weights=dict(resolved_task_loss_weights),
                focus_tasks=_default_focus_tasks(resolved_tasks),
                positive_crop_bias=float(positive_crop_bias),
                repeat_factor=int(repeat_factor),
            )
        ]
    else:
        curriculum_stages = _build_default_curriculum_stages(
            int(epochs),
            families,
            tasks=resolved_tasks,
            repeat_factor=int(repeat_factor),
            task_loss_weights=resolved_task_loss_weights,
            positive_crop_bias=float(positive_crop_bias),
        )

    device, device_label = _select_device(torch, device_mode)
    model = UnifiedRootStarterNet(resolved_tasks, families, in_channels=3, base_filters=base_filters, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))

    best = {"epoch": 0, "selection_score": -1.0, "val_root_iou": -1.0, "val_shoot_iou": -1.0}
    history: list[dict[str, object]] = []
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    early_stopping_patience = max(0, int(early_stopping_patience))
    no_improvement_epochs = 0
    stopped_early = False
    early_stopping_start_epoch = max(
        1,
        1 + sum(int(stage.epochs) for stage in curriculum_stages[:-1]),
    )

    epoch_index = 0
    for stage in curriculum_stages:
        stage_families = set(stage.families)
        stage_train_samples = [sample for sample in train_samples if str(sample.get("family") or "unknown") in stage_families]
        if not stage_train_samples:
            continue
        stage_train_ds = GeneralRootStarterDataset(
            stage_train_samples,
            patch_size,
            augment=True,
            repeat_factor=int(stage.repeat_factor or repeat_factor),
            random_seed=random_seed + epoch_index,
            focus_tasks=stage.focus_tasks,
            positive_crop_bias=float(stage.positive_crop_bias),
        )
        stage_sampler = _build_family_balanced_sampler(torch, stage_train_samples, len(stage_train_ds))
        train_loader = DataLoader(
            stage_train_ds,
            batch_size=max(1, int(batch_size)),
            shuffle=stage_sampler is None,
            sampler=stage_sampler,
        )
        stage_weights = _resolve_stage_weights(stage.task_loss_weights)

        for _stage_epoch in range(max(1, int(stage.epochs))):
            epoch_index += 1
            model.train()
            train_losses: list[float] = []
            for batch in train_loader:
                images = batch["image"].to(device)
                outputs = model(images)
                losses = []
                for task in resolved_tasks:
                    task_weight = float(stage_weights.get(task, 0.0))
                    if task_weight <= 0.0:
                        continue
                    task_loss = _sample_task_loss(
                        F,
                        outputs[task],
                        batch["targets"][task].to(device),
                        batch["availability"][task].to(device),
                    )
                    if task_loss is not None:
                        losses.append(task_loss * task_weight)
                family_logits = outputs.get("family_logits")
                if family_logits is not None and float(family_loss_weight) > 0.0:
                    family_targets = torch.tensor([family_to_idx[str(name)] for name in batch["family"]], dtype=torch.long, device=device)
                    losses.append(F.cross_entropy(family_logits, family_targets) * float(family_loss_weight))
                if not losses:
                    continue
                total_loss = sum(losses)
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()
                train_losses.append(float(total_loss.detach().cpu().item()))

            eval_rows = evaluate_unified_general_root_starter(model, val_samples, device=device)
            task_metric_means: dict[str, float] = {}
            for task in resolved_tasks:
                key = f"{task}_iou"
                scores = [float(row[key]) for row in eval_rows if row.get(key) is not None]
                task_metric_means[task] = float(np.mean(scores)) if scores else 0.0
            val_root_iou = float(task_metric_means.get("root_binary", 0.0))
            val_shoot_iou = float(task_metric_means.get("shoot", 0.0))
            selection_score = _selection_score(task_metric_means, resolved_tasks)
            epoch_row = {
                "epoch": int(epoch_index),
                "curriculum_stage": str(stage.name),
                "train_loss": float(np.mean(train_losses)) if train_losses else None,
                "val_root_iou": val_root_iou,
                "val_shoot_iou": val_shoot_iou,
                "val_task_ious": dict(task_metric_means),
                "val_selection_score": selection_score,
            }
            history.append(epoch_row)
            improvement = float(selection_score) - float(best["selection_score"])
            if improvement >= float(early_stopping_min_delta):
                best = {
                    "epoch": int(epoch_index),
                    "selection_score": selection_score,
                    "val_root_iou": val_root_iou,
                    "val_shoot_iou": val_shoot_iou,
                }
                no_improvement_epochs = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "metadata": {
                            "framework": "pytorch",
                            "model_family": "general_root_starter_unified",
                            "tasks": list(resolved_tasks),
                            "families": list(families),
                            "patch_size": int(patch_size),
                            "base_filters": int(base_filters),
                            "dropout": float(dropout),
                            "device_label": device_label,
                            "weight_decay": float(weight_decay),
                            "family_loss_weight": float(family_loss_weight),
                            "task_loss_weights": dict(resolved_task_loss_weights),
                            "curriculum_mode": normalized_curriculum_mode,
                            "curriculum_stages": [
                                {
                                    "name": str(curr.name),
                                    "epochs": int(curr.epochs),
                                    "families": list(curr.families),
                                    "focus_tasks": list(curr.focus_tasks),
                                    "positive_crop_bias": float(curr.positive_crop_bias),
                                    "repeat_factor": int(curr.repeat_factor or repeat_factor),
                                    "task_loss_weights": dict(curr.task_loss_weights),
                                }
                                for curr in curriculum_stages
                            ],
                            "history": history,
                            "best_epoch": int(epoch_index),
                            "best_selection_score": float(selection_score),
                        },
                    },
                    output_path,
                )
            elif epoch_index >= early_stopping_start_epoch:
                no_improvement_epochs += 1
                if early_stopping_patience > 0 and no_improvement_epochs >= early_stopping_patience:
                    stopped_early = True
                    break
        if stopped_early:
            break

    return {
        "corpus_dir": str(Path(corpus_dir).expanduser().resolve()),
        "output_path": str(output_path),
        "device": device_label,
        "families": families,
        "tasks": list(resolved_tasks),
        "best": best,
        "history": history,
        "stopped_early": bool(stopped_early),
        "weight_decay": float(weight_decay),
        "family_loss_weight": float(family_loss_weight),
        "task_loss_weights": dict(resolved_task_loss_weights),
        "early_stopping_patience": int(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "curriculum_mode": normalized_curriculum_mode,
        "curriculum_stages": [
            {
                "name": str(stage.name),
                "epochs": int(stage.epochs),
                "families": list(stage.families),
                "focus_tasks": list(stage.focus_tasks),
                "positive_crop_bias": float(stage.positive_crop_bias),
                "repeat_factor": int(stage.repeat_factor or repeat_factor),
                "task_loss_weights": dict(stage.task_loss_weights),
            }
            for stage in curriculum_stages
        ],
        "early_stopping_start_epoch": int(early_stopping_start_epoch),
        "positive_crop_bias": float(positive_crop_bias),
    }


def load_unified_checkpoint(checkpoint_path: Path):
    torch, _nn, _F, _Dataset, _DataLoader = _require_torch()
    UnifiedRootStarterNet = _build_unified_model_class()
    payload = torch.load(str(Path(checkpoint_path).expanduser().resolve()), map_location="cpu")
    state_dict = payload["state_dict"]
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    tasks = tuple(metadata.get("tasks") or TASK_KEYS)
    families = tuple(metadata.get("families") or [])
    base_filters = int(metadata.get("base_filters") or 24)
    dropout = float(metadata.get("dropout") or 0.10)
    model = UnifiedRootStarterNet(tasks, families, in_channels=3, base_filters=base_filters, dropout=dropout)
    model.load_state_dict(state_dict)
    return model, metadata


def run_unified_general_root_starter_on_image(image_rgb: np.ndarray, checkpoint_path: Path, *, device_mode: str = "auto") -> tuple[dict[str, np.ndarray], dict[str, object]]:
    torch, _nn, F, _Dataset, _DataLoader = _require_torch()
    model, metadata = load_unified_checkpoint(checkpoint_path)
    device, device_label = _select_device(torch, device_mode)
    model = model.to(device)
    model.eval()
    image = np.asarray(image_rgb, dtype=np.uint8)
    h, w = image.shape[:2]
    patch_size = int(metadata.get("patch_size") or 256)
    resized = np.asarray(Image.fromarray(image).resize((patch_size, patch_size), Image.BILINEAR), dtype=np.uint8)
    tensor = torch.from_numpy(np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]).float().to(device)
    with torch.no_grad():
        outputs = model(tensor)
    task_masks: dict[str, np.ndarray] = {}
    task_confidences: dict[str, float] = {}
    for task in metadata.get("tasks") or TASK_KEYS:
        logits = outputs[task]
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
        full = np.asarray(Image.fromarray((probs * 255.0).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.uint8).astype(np.float32) / 255.0
        task_masks[str(task)] = (full >= 0.5).astype(np.uint8)
        task_confidences[str(task)] = float(np.mean(full))
    details = {
        "backend": "pytorch-unified",
        "device": device_label,
        "tasks": list(metadata.get("tasks") or TASK_KEYS),
        "families": list(metadata.get("families") or []),
        "task_confidences": task_confidences,
    }
    return task_masks, details


def evaluate_unified_general_root_starter(model_or_checkpoint, samples: list[dict[str, object]], *, device=None, device_mode: str = "auto") -> list[dict[str, object]]:
    torch, _nn, _F, _Dataset, _DataLoader = _require_torch()
    if isinstance(model_or_checkpoint, (str, Path)):
        model, metadata = load_unified_checkpoint(Path(model_or_checkpoint))
    else:
        model = model_or_checkpoint
        metadata = {"tasks": list(getattr(model, "tasks", TASK_KEYS)), "patch_size": 256}
    if device is None:
        device, _device_label = _select_device(torch, device_mode)
    model = model.to(device)
    model.eval()
    rows: list[dict[str, object]] = []
    for sample in samples:
        image = _read_rgb(Path(str(sample["image_path"])))
        # Inline inference avoids reloading the checkpoint for every sample.
        h, w = image.shape[:2]
        patch_size = int(metadata.get("patch_size") or 256)
        resized = np.asarray(Image.fromarray(image).resize((patch_size, patch_size), Image.BILINEAR), dtype=np.uint8)
        tensor = torch.from_numpy(np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]).float().to(device)
        with torch.no_grad():
            outputs = model(tensor)
        target_masks = _read_masks(Path(str(sample["mask_path"])))
        row: dict[str, object] = {
            "sample_id": str(sample.get("sample_id") or ""),
            "family": str(sample.get("family") or "unknown"),
        }
        available = sample.get("tasks_available") if isinstance(sample.get("tasks_available"), dict) else {}
        for task in metadata.get("tasks") or TASK_KEYS:
            probs = torch.sigmoid(outputs[task])[0, 0].detach().cpu().numpy().astype(np.float32)
            full = np.asarray(Image.fromarray((probs * 255.0).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.uint8).astype(np.float32) / 255.0
            pred = (full >= 0.5).astype(np.uint8)
            if bool(available.get(task, False)):
                row[f"{task}_iou"] = _binary_iou(pred, np.asarray(target_masks.get(task, np.zeros((h, w), dtype=np.uint8)), dtype=np.uint8))
            else:
                row[f"{task}_iou"] = None
        rows.append(row)
    return rows
