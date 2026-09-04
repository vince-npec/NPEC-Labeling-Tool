from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np


PROVENANCE_NONE = np.uint8(0)
PROVENANCE_SYNTHETIC_BACTERIAL_OCCLUSION = np.uint8(1)


@dataclass(frozen=True)
class RootGapSupervision:
    """Disjoint visible and hidden targets for synthetic gap-repair training.

    ``visible_root_mask`` is the only root mask suitable for visible-root
    measurement. ``hidden_root_target`` is supervision for a separate repair
    head and must not be merged into visible measurements. The caller's
    authoritative mask is never retained by reference or modified.
    """

    visible_root_mask: np.ndarray
    hidden_root_target: np.ndarray
    hidden_root_provenance: np.ndarray
    synthetic_occluder_mask: np.ndarray
    seed: int
    requested_occlusion_count: int
    applied_occlusion_count: int


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    parsed = int(parsed)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _root_mask_copy(root_mask: np.ndarray) -> np.ndarray:
    array = np.asarray(root_mask)
    if array.ndim != 2:
        raise ValueError("root_mask must be a two-dimensional array")
    if array.size == 0 or 0 in array.shape:
        raise ValueError("root_mask must have non-zero height and width")
    if not (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError("root_mask must contain boolean or numeric values")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError("root_mask must not contain NaN or infinite values")
    return np.array(array != 0, dtype=bool, copy=True)


def _ellipse_mask(
    shape: tuple[int, int],
    center_y: int,
    center_x: int,
    radius_y: int,
    radius_x: int,
    angle_radians: float,
) -> np.ndarray:
    height, width = shape
    extent = max(radius_x, radius_y) + 1
    y0 = max(0, center_y - extent)
    y1 = min(height, center_y + extent + 1)
    x0 = max(0, center_x - extent)
    x1 = min(width, center_x + extent + 1)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    dy = yy.astype(np.float64) - float(center_y)
    dx = xx.astype(np.float64) - float(center_x)
    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)
    rotated_x = (cosine * dx) + (sine * dy)
    rotated_y = (-sine * dx) + (cosine * dy)
    local = ((rotated_x / radius_x) ** 2 + (rotated_y / radius_y) ** 2) <= 1.0

    result = np.zeros(shape, dtype=bool)
    result[y0:y1, x0:x1] = local
    return result


def create_synthetic_bacterial_occlusions(
    root_mask: np.ndarray,
    *,
    seed: int = 0,
    occlusion_count: int = 3,
    min_radius: int = 3,
    max_radius: int = 12,
) -> RootGapSupervision:
    """Create deterministic occlusions without changing authoritative labels.

    Ellipses are centered on currently unhidden root pixels, so every applied
    occlusion intersects the supplied root. Background covered by an ellipse is
    retained only in ``synthetic_occluder_mask`` and never becomes a root target.
    Empty root content and ``occlusion_count=0`` are safe no-op cases.
    """

    original = _root_mask_copy(root_mask)
    parsed_seed = _integer(seed, "seed")
    parsed_count = _integer(occlusion_count, "occlusion_count", minimum=0)
    parsed_min_radius = _integer(min_radius, "min_radius", minimum=1)
    parsed_max_radius = _integer(max_radius, "max_radius", minimum=1)
    if parsed_max_radius < parsed_min_radius:
        raise ValueError("max_radius must be greater than or equal to min_radius")

    hidden = np.zeros_like(original)
    occluder = np.zeros_like(original)
    applied = 0
    rng = np.random.default_rng(parsed_seed)

    for _ in range(parsed_count):
        remaining_coordinates = np.argwhere(original & ~hidden)
        if remaining_coordinates.size == 0:
            break
        center_index = int(rng.integers(0, len(remaining_coordinates)))
        center_y, center_x = (int(value) for value in remaining_coordinates[center_index])
        radius_y = int(rng.integers(parsed_min_radius, parsed_max_radius + 1))
        radius_x = int(rng.integers(parsed_min_radius, parsed_max_radius + 1))
        angle = float(rng.uniform(0.0, np.pi))
        candidate = _ellipse_mask(
            original.shape,
            center_y,
            center_x,
            radius_y,
            radius_x,
            angle,
        )
        newly_hidden = candidate & original & ~hidden
        if not np.any(newly_hidden):
            continue
        occluder |= candidate
        hidden |= candidate & original
        applied += 1

    visible = original & ~hidden
    provenance = np.full(original.shape, PROVENANCE_NONE, dtype=np.uint8)
    provenance[hidden] = PROVENANCE_SYNTHETIC_BACTERIAL_OCCLUSION

    return RootGapSupervision(
        visible_root_mask=visible,
        hidden_root_target=hidden,
        hidden_root_provenance=provenance,
        synthetic_occluder_mask=occluder,
        seed=parsed_seed,
        requested_occlusion_count=parsed_count,
        applied_occlusion_count=applied,
    )


__all__ = [
    "PROVENANCE_NONE",
    "PROVENANCE_SYNTHETIC_BACTERIAL_OCCLUSION",
    "RootGapSupervision",
    "create_synthetic_bacterial_occlusions",
]
