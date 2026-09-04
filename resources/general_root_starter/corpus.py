from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image

from ..models import LabelClass
from ..project_io import load_project
from . import SCHEMA_KEYS, _matches_synonyms, describe_image_features, normalize_annotation_targets


@dataclass(slots=True)
class CorpusSourceSpec:
    project_path: Path
    family: str
    name: str | None = None


def infer_family_from_path(project_path: Path) -> str:
    token = str(project_path).lower()
    if "rootpainter" in token or "minirhizotron" in token:
        return "minirhizotron_public"
    if "multiplex" in token or "dark" in token or "mxlab" in token:
        return "arabidopsis_dark_rgb"
    if "potato" in token:
        return "potato_rgb"
    if "lucifer" in token or "rgb" in token:
        return "plate_rgb_lucifer"
    if "hades" in token or "bw" in token or "grayscale" in token:
        return "plate_bw_hades"
    return "generic_root"


def _normalized_task_coverage(classes: list[LabelClass]) -> dict[str, bool]:
    coverage = {key: False for key in SCHEMA_KEYS}
    for cls in classes:
        name = str(cls.name or "")
        for key in SCHEMA_KEYS:
            if _matches_synonyms(name, key):
                coverage[key] = True
    if not coverage["root_binary"] and (coverage["primary_root"] or coverage["lateral_root"]):
        coverage["root_binary"] = True
    # In most NPEC plate datasets, "Root" and "Lateral" mean primary root vs lateral root.
    # Promote that combination so the unified model can actually learn a primary-root head
    # instead of treating it as perpetually unlabeled auxiliary supervision.
    if not coverage["primary_root"] and coverage["root_binary"] and coverage["lateral_root"]:
        coverage["primary_root"] = True
    return coverage


def _safe_stem(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    return token.strip("_") or "sample"


def _write_png(path: Path, image_rgb: np.ndarray) -> None:
    arr = np.asarray(image_rgb, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    Image.fromarray(arr).save(path)


def _split_names(train_fraction: float, val_fraction: float) -> tuple[float, float, float]:
    train = max(0.0, min(1.0, float(train_fraction)))
    val = max(0.0, min(1.0 - train, float(val_fraction)))
    test = max(0.0, 1.0 - train - val)
    return train, val, test


def _assign_split(index: int, total: int, train_fraction: float, val_fraction: float) -> str:
    if total <= 1:
        return "train"
    position = (index + 0.5) / float(total)
    train_cut = float(train_fraction)
    val_cut = float(train_fraction + val_fraction)
    if position <= train_cut:
        return "train"
    if position <= val_cut:
        return "val"
    return "test"


def build_general_root_corpus(
    sources: Iterable[CorpusSourceSpec],
    output_dir: Path,
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    random_seed: int = 17,
    max_images_per_source: int = 0,
) -> dict[str, object]:
    train_fraction, val_fraction, _ = _split_names(train_fraction, val_fraction)
    output_root = Path(output_dir).expanduser().resolve()
    images_dir = output_root / "images"
    masks_dir = output_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(random_seed))
    manifest_entries: list[dict[str, object]] = []
    family_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    source_rows: list[dict[str, object]] = []

    for source_idx, source in enumerate(list(sources)):
        project_path = Path(source.project_path).expanduser().resolve()
        payload = load_project(project_path)
        items = list(payload["dataset_items"])
        annotations = payload["annotations"]
        classes = payload["classes"]
        source_name = str(source.name or project_path.stem)
        family = str(source.family or infer_family_from_path(project_path))
        coverage = _normalized_task_coverage(classes)

        if max_images_per_source > 0 and len(items) > int(max_images_per_source):
            items = list(items)
            rng.shuffle(items)
            items = items[: int(max_images_per_source)]

        shuffled = list(items)
        rng.shuffle(shuffled)
        total = len(shuffled)
        source_rows.append(
            {
                "project_path": str(project_path),
                "source_name": source_name,
                "family": family,
                "image_count": int(total),
                "task_coverage": dict(coverage),
            }
        )

        for item_idx, item in enumerate(shuffled):
            image = np.asarray(item.image, dtype=np.uint8)
            normalized = normalize_annotation_targets(annotations.get(item.uid), classes, image.shape[:2])
            sample_id = f"{source_idx:02d}_{item_idx:05d}_{_safe_stem(item.uid)}"
            image_path = images_dir / f"{sample_id}.png"
            mask_path = masks_dir / f"{sample_id}.npz"
            _write_png(image_path, image)
            np.savez_compressed(
                mask_path,
                shoot=np.asarray(normalized["shoot"], dtype=np.uint8),
                primary_root=np.asarray(normalized["primary_root"], dtype=np.uint8),
                lateral_root=np.asarray(normalized["lateral_root"], dtype=np.uint8),
                root_binary=np.asarray(normalized["root_binary"], dtype=np.uint8),
                seed_crown=np.asarray(normalized["seed_crown"], dtype=np.uint8),
            )
            features = describe_image_features(image)
            split = _assign_split(item_idx, total, train_fraction, val_fraction)
            split_counts[split] = int(split_counts.get(split, 0) + 1)
            family_counts[family] = int(family_counts.get(family, 0) + 1)
            manifest_entries.append(
                {
                    "sample_id": sample_id,
                    "uid": str(item.uid),
                    "name": str(item.name),
                    "source_name": source_name,
                    "source_project_path": str(project_path),
                    "family": family,
                    "split": split,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "tasks_available": dict(coverage),
                    "image_features": {
                        "is_rgb_like": bool(features.is_rgb_like),
                        "megapixels": float(features.megapixels),
                        "mean_intensity": float(features.mean_intensity),
                        "top_dark_ratio": float(features.top_dark_ratio),
                        "green_dominance": float(features.green_dominance),
                    },
                }
            )

    manifest = {
        "name": "general_root_starter_corpus",
        "version": 1,
        "schema_keys": list(SCHEMA_KEYS),
        "random_seed": int(random_seed),
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
        "sources": source_rows,
        "family_counts": family_counts,
        "split_counts": split_counts,
        "sample_count": len(manifest_entries),
        "samples": manifest_entries,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
