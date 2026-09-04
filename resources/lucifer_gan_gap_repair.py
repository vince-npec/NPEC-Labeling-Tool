from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np

from .models import DatasetImageItem


_STEM_TOKEN_RE = re.compile(r"(?<!\d)(\d{1,4}-\d{1,4})(?!\d)")
_PRECOMPUTED_SUMMARY_NAME = "summary.json"
_PRECOMPUTED_ORIGINAL_MASK_NAME = "02_original_mask.png"
_PRECOMPUTED_REPAIRED_MASK_NAME = "06_inpainted_after_cleanup.png"


@dataclass(slots=True)
class LuciferGanGapRepairResult:
    root_mask: np.ndarray
    applied: bool
    mode: str
    stem: str | None = None
    case_dir: Path | None = None
    added_pixels: int = 0
    used_full_repaired_mask: bool = False
    message: str = ""


def _ensure_mask_shape(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    arr = np.asarray(mask, dtype=np.uint8)
    if arr.shape[:2] == (h, w):
        return (arr > 0).astype(np.uint8)
    resized = cv2.resize((arr > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return (np.asarray(resized, dtype=np.uint8) > 0).astype(np.uint8)


def _load_binary_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot read Lucifer GAN mask: {path}")
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    return _ensure_mask_shape(raw, shape_hw)


def _candidate_stems_for_item(item: DatasetImageItem) -> list[str]:
    candidates: list[str] = []
    for raw in (
        Path(str(item.name or "")).stem,
        Path(str(item.path or "")).stem,
        str(item.uid or ""),
        str(item.name or ""),
        str(item.path or ""),
    ):
        text = str(raw or "").strip()
        if not text:
            continue
        if text not in candidates:
            candidates.append(text)
        for match in _STEM_TOKEN_RE.findall(text):
            if match not in candidates:
                candidates.append(match)
    return candidates


def _detect_precomputed_root(root: Path) -> Path | None:
    candidate = Path(root).expanduser()
    if not candidate.exists():
        return None
    if (candidate / _PRECOMPUTED_SUMMARY_NAME).exists():
        return candidate
    for child in candidate.iterdir():
        if child.is_dir() and (child / _PRECOMPUTED_SUMMARY_NAME).exists():
            return child
    return None


def _find_case_dir(precomputed_root: Path, item: DatasetImageItem) -> tuple[Path | None, str | None]:
    for stem in _candidate_stems_for_item(item):
        case_dir = precomputed_root / stem
        if case_dir.is_dir() and (case_dir / _PRECOMPUTED_REPAIRED_MASK_NAME).exists():
            return case_dir, stem
    return None, None


def _apply_precomputed_case(
    *,
    item: DatasetImageItem,
    root_mask: np.ndarray,
    case_dir: Path,
    stem: str,
) -> LuciferGanGapRepairResult:
    shape_hw = item.image.shape[:2]
    repaired_path = case_dir / _PRECOMPUTED_REPAIRED_MASK_NAME
    repaired_mask = _load_binary_mask(repaired_path, shape_hw)
    original_path = case_dir / _PRECOMPUTED_ORIGINAL_MASK_NAME
    current_root = _ensure_mask_shape(root_mask, shape_hw)

    if not original_path.exists():
        added_pixels = int(np.count_nonzero(np.logical_and(repaired_mask > 0, current_root == 0)))
        return LuciferGanGapRepairResult(
            root_mask=repaired_mask.astype(np.uint8),
            applied=True,
            mode="precomputed_replace",
            stem=stem,
            case_dir=case_dir,
            added_pixels=added_pixels,
            used_full_repaired_mask=True,
            message="Used the full repaired Lucifer GAN mask because no original pre-GAN mask was present.",
        )

    original_mask = _load_binary_mask(original_path, shape_hw)
    delta_add = np.logical_and(repaired_mask > 0, original_mask == 0)
    current_root_bin = current_root > 0
    if int(np.count_nonzero(current_root_bin)) <= 0:
        added_pixels = int(np.count_nonzero(np.logical_and(repaired_mask > 0, ~current_root_bin)))
        return LuciferGanGapRepairResult(
            root_mask=repaired_mask.astype(np.uint8),
            applied=True,
            mode="precomputed_replace",
            stem=stem,
            case_dir=case_dir,
            added_pixels=added_pixels,
            used_full_repaired_mask=True,
            message="Used the full repaired Lucifer GAN mask because the current root prediction was empty.",
        )

    updated = np.logical_or(current_root_bin, delta_add)
    added_pixels = int(np.count_nonzero(np.logical_and(updated, ~current_root_bin)))
    return LuciferGanGapRepairResult(
        root_mask=np.asarray(updated, dtype=np.uint8),
        applied=bool(added_pixels > 0),
        mode="precomputed_delta",
        stem=stem,
        case_dir=case_dir,
        added_pixels=added_pixels,
        used_full_repaired_mask=False,
        message=(
            "Applied only the GAN-added repair pixels on top of the current root mask."
            if added_pixels > 0
            else "A matching Lucifer GAN case was found, but it did not add any new root pixels."
        ),
    )


def apply_lucifer_gan_gap_repair(
    item: DatasetImageItem,
    root_mask: np.ndarray,
    gap_repair_root: Path | str | None,
) -> LuciferGanGapRepairResult:
    shape_hw = item.image.shape[:2]
    base_mask = _ensure_mask_shape(root_mask, shape_hw)
    if gap_repair_root is None or not str(gap_repair_root).strip():
        return LuciferGanGapRepairResult(
            root_mask=base_mask,
            applied=False,
            mode="disabled",
            message="Lucifer GAN gap repair is disabled.",
        )

    root_dir = Path(gap_repair_root).expanduser()
    if not root_dir.exists():
        raise FileNotFoundError(f"Lucifer GAN gap repair folder does not exist: {root_dir}")

    precomputed_root = _detect_precomputed_root(root_dir)
    if precomputed_root is None:
        return LuciferGanGapRepairResult(
            root_mask=base_mask,
            applied=False,
            mode="unavailable",
            message=(
                "The selected Lucifer GAN folder does not look like a precomputed inference_output folder yet. "
                "Point it at a folder that contains summary.json and per-image outputs."
            ),
        )

    case_dir, stem = _find_case_dir(precomputed_root, item)
    if case_dir is None:
        return LuciferGanGapRepairResult(
            root_mask=base_mask,
            applied=False,
            mode="no_match",
            message=(
                "No matching Lucifer GAN case folder was found for this image. "
                f"Tried stems: {', '.join(_candidate_stems_for_item(item)[:8])}"
            ),
        )
    return _apply_precomputed_case(item=item, root_mask=base_mask, case_dir=case_dir, stem=str(stem))
