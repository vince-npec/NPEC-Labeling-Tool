from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np


ORGAN_NAMES = ("seed", "shoot", "root", "lateral")
DEFAULT_SEMANTIC_IDS = {
    "background": 0,
    "seed": 1,
    "shoot": 2,
    "root": 3,
    "lateral": 4,
}
_INSTANCE_ID_RE = re.compile(r"(?P<owner>\d+)$")


@dataclass(frozen=True, slots=True)
class FiveSeedlingSample:
    sample_id: str
    plate_group: str
    image_path: Path
    semantic_dir: Path
    instance_dir: Path


@dataclass(frozen=True, slots=True)
class OwnershipTargets:
    owner_mask: np.ndarray
    visible_foreground: np.ndarray
    valid_owner_pixels: np.ndarray
    conflict_pixels: np.ndarray
    semantic_mask: np.ndarray
    owner_boxes_xywh: dict[int, tuple[int, int, int, int]]
    source_owner_to_slot: dict[int, int]


def _natural_key(text: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", str(text).lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _plate_group(sample_id: str) -> str:
    clean = str(sample_id).strip()
    match = re.match(r"^(?P<plate>\d+)-\d+$", clean)
    if match:
        return str(match.group("plate"))
    return re.sub(r"[^a-z0-9]+", "-", clean.lower()).strip("-") or "unknown"


def discover_five_seedling_corpus(root: Path | str) -> list[FiveSeedlingSample]:
    corpus_root = Path(root).expanduser()
    image_root = corpus_root / "images"
    semantic_root = corpus_root / "masks_binary"
    instance_root = corpus_root / "masks_instances"
    if not image_root.is_dir() or not semantic_root.is_dir() or not instance_root.is_dir():
        raise FileNotFoundError(
            "Five-seedling ground truth requires images/, masks_binary/, and masks_instances/ folders."
        )

    samples: list[FiveSeedlingSample] = []
    for image_path in sorted(image_root.glob("*.png"), key=lambda path: _natural_key(path.stem)):
        sample_id = image_path.stem
        semantic_dir = semantic_root / sample_id
        sample_instance_dir = instance_root / sample_id
        if not semantic_dir.is_dir() or not sample_instance_dir.is_dir():
            continue
        samples.append(
            FiveSeedlingSample(
                sample_id=sample_id,
                plate_group=_plate_group(sample_id),
                image_path=image_path,
                semantic_dir=semantic_dir,
                instance_dir=sample_instance_dir,
            )
        )
    if not samples:
        raise FileNotFoundError(f"No matched five-seedling samples were found under {corpus_root}.")
    return samples


def _read_binary(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Unable to read mask: {path}")
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if shape_hw is not None and tuple(binary.shape[:2]) != tuple(shape_hw):
        raise ValueError(f"Mask shape {binary.shape[:2]} does not match expected {shape_hw}: {path}")
    return binary


def load_semantic_masks(sample: FiveSeedlingSample) -> dict[str, np.ndarray]:
    indexed_path = sample.semantic_dir / "indexed_mask.png"
    indexed = cv2.imread(str(indexed_path), cv2.IMREAD_GRAYSCALE)
    if indexed is None:
        raise ValueError(f"Unable to read indexed mask: {indexed_path}")
    shape_hw = tuple(int(value) for value in indexed.shape[:2])
    masks: dict[str, np.ndarray] = {}
    for organ in (*ORGAN_NAMES, "background"):
        path = sample.semantic_dir / f"{organ}.png"
        if path.exists():
            masks[organ] = _read_binary(path, shape_hw)
        else:
            masks[organ] = np.zeros(shape_hw, dtype=np.uint8)
    return masks


def load_instance_masks(
    sample: FiveSeedlingSample,
    organ: str,
) -> dict[int, np.ndarray]:
    normalized = str(organ).strip().lower()
    if normalized not in ORGAN_NAMES:
        raise ValueError(f"Unsupported instance organ: {organ}")
    organ_dir = sample.instance_dir / normalized
    if not organ_dir.is_dir():
        return {}
    masks: dict[int, np.ndarray] = {}
    shape_hw: tuple[int, int] | None = None
    for path in sorted(organ_dir.glob("*.png"), key=lambda candidate: _natural_key(candidate.stem)):
        match = _INSTANCE_ID_RE.search(path.stem)
        if match is None:
            continue
        owner = int(match.group("owner"))
        mask = _read_binary(path, shape_hw)
        if shape_hw is None:
            shape_hw = tuple(int(value) for value in mask.shape[:2])
        masks[owner] = np.maximum(masks.get(owner, np.zeros_like(mask)), mask)
    return masks


def _mask_center_x(mask: np.ndarray) -> float | None:
    _ys, xs = np.where(np.asarray(mask, dtype=np.uint8) > 0)
    if xs.size <= 0:
        return None
    return float(np.median(xs.astype(np.float64)))


def shoot_ordered_slot_mapping(
    sample: FiveSeedlingSample,
    expected_count: int = 5,
) -> dict[int, int]:
    shoot_masks = load_instance_masks(sample, "shoot")
    root_masks = load_instance_masks(sample, "root")
    candidate_ids = sorted(set(shoot_masks) | set(root_masks))
    ranked: list[tuple[float, int]] = []
    missing: list[int] = []
    for owner in candidate_ids:
        center = _mask_center_x(shoot_masks.get(owner, np.zeros((1, 1), dtype=np.uint8)))
        if center is None:
            center = _mask_center_x(root_masks.get(owner, np.zeros((1, 1), dtype=np.uint8)))
        if center is None:
            missing.append(owner)
        else:
            ranked.append((float(center), int(owner)))
    ranked.sort(key=lambda item: (item[0], item[1]))
    ordered_ids = [owner for _center, owner in ranked] + sorted(missing)
    mapping = {int(owner): int(slot) for slot, owner in enumerate(ordered_ids, start=1)}
    if len(mapping) > int(max(1, expected_count)):
        mapping = {owner: slot for owner, slot in mapping.items() if slot <= int(expected_count)}
    return mapping


def _bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(np.asarray(mask, dtype=np.uint8) > 0)
    if xs.size <= 0 or ys.size <= 0:
        return (0, 0, 0, 0)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def build_ownership_targets(
    sample: FiveSeedlingSample,
    *,
    expected_count: int = 5,
    owner_organs: Iterable[str] = ("root", "lateral", "shoot"),
    semantic_ids: dict[str, int] | None = None,
) -> OwnershipTargets:
    semantic_ids = dict(DEFAULT_SEMANTIC_IDS if semantic_ids is None else semantic_ids)
    semantic_masks = load_semantic_masks(sample)
    shape_hw = next(iter(semantic_masks.values())).shape[:2]
    semantic_mask = np.zeros(shape_hw, dtype=np.uint8)
    for organ in ("seed", "shoot", "root", "lateral"):
        class_id = int(semantic_ids.get(organ, 0))
        if class_id > 0:
            semantic_mask[semantic_masks[organ] > 0] = np.uint8(class_id)

    source_to_slot = shoot_ordered_slot_mapping(sample, expected_count=expected_count)
    owner_hits = np.zeros((int(expected_count), *shape_hw), dtype=np.uint8)
    owner_union = np.zeros_like(owner_hits)
    for organ in owner_organs:
        normalized = str(organ).strip().lower()
        for source_owner, mask in load_instance_masks(sample, normalized).items():
            slot = source_to_slot.get(int(source_owner))
            if slot is None or slot < 1 or slot > int(expected_count):
                continue
            owner_union[slot - 1] = np.maximum(owner_union[slot - 1], mask.astype(np.uint8))

    owner_hits[:] = (owner_union > 0).astype(np.uint8)
    hit_count = np.sum(owner_hits, axis=0, dtype=np.uint8)
    valid = hit_count == 1
    conflict = hit_count > 1
    visible = hit_count > 0
    owner_mask = np.zeros(shape_hw, dtype=np.uint8)
    if np.any(valid):
        winner = np.argmax(owner_hits, axis=0).astype(np.uint8) + np.uint8(1)
        owner_mask[valid] = winner[valid]

    boxes = {
        slot: _bbox_xywh(owner_union[slot - 1])
        for slot in range(1, int(expected_count) + 1)
    }
    return OwnershipTargets(
        owner_mask=owner_mask,
        visible_foreground=visible.astype(np.uint8),
        valid_owner_pixels=valid.astype(np.uint8),
        conflict_pixels=conflict.astype(np.uint8),
        semantic_mask=semantic_mask,
        owner_boxes_xywh=boxes,
        source_owner_to_slot=source_to_slot,
    )


def grouped_split(
    samples: Iterable[FiveSeedlingSample],
    *,
    holdout_fraction: float = 0.20,
    salt: str = "npec-five-seedling-v1",
) -> tuple[list[FiveSeedlingSample], list[FiveSeedlingSample]]:
    sample_list = list(samples)
    fraction = max(0.0, min(0.8, float(holdout_fraction)))
    groups = sorted({sample.plate_group for sample in sample_list}, key=_natural_key)
    holdout_groups: set[str] = set()
    for group in groups:
        digest = hashlib.sha1(f"{salt}:{group}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        if value < fraction:
            holdout_groups.add(group)
    if fraction > 0.0 and groups and not holdout_groups:
        holdout_groups.add(groups[-1])
    if len(holdout_groups) == len(groups) and len(groups) > 1:
        holdout_groups.remove(groups[0])
    train = [sample for sample in sample_list if sample.plate_group not in holdout_groups]
    holdout = [sample for sample in sample_list if sample.plate_group in holdout_groups]
    return train, holdout
