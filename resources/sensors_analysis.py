from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import cv2
import numpy as np
from PIL import Image


REGION_COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "main_root": (255, 0, 0),
    "lateral_root": (0, 255, 0),
    "main_root_tip": (0, 0, 255),
    "other_tip": (0, 255, 255),
    "node": (255, 255, 0),
    "dilated_root_exclusive": (255, 128, 0),
    "shoot": (255, 0, 255),
}


def _safe_skeletonize(binary: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize  # type: ignore
    except Exception:
        return np.asarray(binary > 0, dtype=bool)
    return np.asarray(skeletonize(np.asarray(binary) > 0), dtype=bool)


@dataclass(slots=True)
class PlantRegionAnalysis:
    plant_id: str
    plant_mask: np.ndarray
    region_masks: dict[str, np.ndarray]


def _mask_u8(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8)


def _component_centroid(mask: np.ndarray) -> tuple[float, float]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return (0.0, 0.0)
    y_mean = float(np.mean(coords[:, 0]))
    x_mean = float(np.mean(coords[:, 1]))
    return (x_mean, y_mean)


def _connected_regions(mask: np.ndarray, dilation_iters: int = 0) -> list[np.ndarray]:
    binary = _mask_u8(mask)
    if dilation_iters > 0 and np.any(binary):
        kernel = np.ones((3, 3), dtype=np.uint8)
        grown = cv2.dilate(binary, kernel, iterations=int(dilation_iters))
    else:
        grown = binary
    count, labels = cv2.connectedComponents(grown, connectivity=8)
    regions: list[np.ndarray] = []
    for label in range(1, int(count)):
        region = np.logical_and(binary > 0, labels == label)
        if np.any(region):
            regions.append(region.astype(np.uint8))
    return regions


def _split_occupied_by_shoot_anchors(occupied: np.ndarray, shoot_mask: np.ndarray) -> list[np.ndarray]:
    anchors = _connected_regions(shoot_mask, dilation_iters=0)
    if len(anchors) <= 1:
        if np.any(occupied):
            return [_mask_u8(occupied)]
        return []

    centroids = np.asarray([_component_centroid(anchor) for anchor in anchors], dtype=np.float32)
    coords = np.argwhere(occupied > 0)
    if coords.size == 0:
        return []

    points_xy = coords[:, ::-1].astype(np.float32)
    diff = points_xy[:, None, :] - centroids[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    assignments = np.argmin(dist2, axis=1)

    regions: list[np.ndarray] = []
    for idx in range(len(anchors)):
        region = np.zeros_like(occupied, dtype=np.uint8)
        chosen = coords[assignments == idx]
        if chosen.size > 0:
            region[chosen[:, 0], chosen[:, 1]] = 1
        # Always keep the shoot anchor inside the assigned plant.
        region = np.logical_or(region > 0, anchors[idx] > 0).astype(np.uint8)
        if np.any(region):
            regions.append(region)
    return regions


def _derive_plant_masks(root_binary: np.ndarray, shoot_mask: np.ndarray) -> list[np.ndarray]:
    occupied = np.logical_or(root_binary > 0, shoot_mask > 0).astype(np.uint8)
    if not np.any(occupied):
        return []
    if np.any(shoot_mask):
        regions = _split_occupied_by_shoot_anchors(occupied, shoot_mask)
    else:
        regions = _connected_regions(occupied, dilation_iters=6)

    if not regions:
        regions = [_mask_u8(occupied)]

    regions = [region for region in regions if np.any(region)]
    regions.sort(key=lambda region: _component_centroid(region)[0])
    return regions


def _skeleton_endpoints_and_nodes(root_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary = np.asarray(root_mask) > 0
    if not np.any(binary):
        shape = binary.shape
        return (np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8))

    skeleton = _safe_skeletonize(binary)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = np.logical_and(skeleton, neighbors == 2)
    nodes = np.logical_and(skeleton, neighbors >= 4)
    return (endpoints.astype(np.uint8), nodes.astype(np.uint8))


def derive_plant_region_analyses(
    main_root_mask: np.ndarray,
    lateral_root_mask: np.ndarray,
    shoot_mask: np.ndarray | None = None,
    dilated_root_iterations: int = 50,
) -> list[PlantRegionAnalysis]:
    main_root = _mask_u8(main_root_mask)
    lateral_root = _mask_u8(lateral_root_mask)
    shoot = _mask_u8(shoot_mask) if shoot_mask is not None else np.zeros_like(main_root, dtype=np.uint8)
    root_binary = np.logical_or(main_root > 0, lateral_root > 0).astype(np.uint8)
    if not np.any(root_binary) and not np.any(shoot):
        return []

    plant_masks = _derive_plant_masks(root_binary, shoot)
    analyses: list[PlantRegionAnalysis] = []
    kernel = np.ones((3, 3), dtype=np.uint8)

    for idx, plant_mask in enumerate(plant_masks, start=1):
        plant_binary = _mask_u8(plant_mask)
        plant_main = np.logical_and(main_root > 0, plant_binary > 0).astype(np.uint8)
        plant_lateral = np.logical_and(lateral_root > 0, plant_binary > 0).astype(np.uint8)
        plant_shoot = np.logical_and(shoot > 0, plant_binary > 0).astype(np.uint8)
        plant_root = np.logical_or(plant_main > 0, plant_lateral > 0).astype(np.uint8)

        if not np.any(plant_root) and not np.any(plant_shoot):
            continue

        dilated_root = cv2.dilate(plant_root, kernel, iterations=max(0, int(dilated_root_iterations)))
        if np.any(plant_shoot):
            shoot_coords = np.argwhere(plant_shoot > 0)
            if shoot_coords.size > 0:
                shoot_bottom_y = int(np.max(shoot_coords[:, 0]))
                dilated_root[:shoot_bottom_y, :] = 0
        dilated_root = _mask_u8(dilated_root)
        dilated_exclusive = np.logical_and(dilated_root > 0, plant_root == 0).astype(np.uint8)

        endpoints, nodes = _skeleton_endpoints_and_nodes(plant_root)
        endpoints_main = np.logical_and(endpoints > 0, plant_main > 0)
        endpoint_coords = np.argwhere(endpoints_main)
        if endpoint_coords.size == 0:
            endpoint_coords = np.argwhere(endpoints > 0)

        main_tip = np.zeros_like(plant_root, dtype=np.uint8)
        other_tip = np.zeros_like(plant_root, dtype=np.uint8)
        if endpoint_coords.size > 0:
            endpoint_coords = endpoint_coords[np.argsort(endpoint_coords[:, 0])]
            main_tip_coord = endpoint_coords[-1]
            main_tip[main_tip_coord[0], main_tip_coord[1]] = 1
            remaining = np.argwhere(endpoints > 0)
            if remaining.size > 0:
                for y, x in remaining:
                    if int(y) == int(main_tip_coord[0]) and int(x) == int(main_tip_coord[1]):
                        continue
                    other_tip[y, x] = 1

        main_tip = cv2.dilate(main_tip, kernel, iterations=5)
        other_tip = cv2.dilate(other_tip, kernel, iterations=5)
        node_mask = cv2.dilate(nodes.astype(np.uint8), kernel, iterations=5)

        analyses.append(
            PlantRegionAnalysis(
                plant_id=f"plant_{idx}",
                plant_mask=plant_binary,
                region_masks={
                    "main_root": plant_main,
                    "lateral_root": plant_lateral,
                    "main_root_tip": _mask_u8(main_tip),
                    "other_tip": _mask_u8(other_tip),
                    "node": _mask_u8(node_mask),
                    "dilated_root_exclusive": _mask_u8(dilated_exclusive),
                    "dilated_root": dilated_root,
                    "shoot": plant_shoot,
                },
            )
        )

    return analyses


def summarize_fluorescence_regions(
    signal_image: np.ndarray,
    analyses: list[PlantRegionAnalysis],
    fc_file: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    signal = np.asarray(signal_image, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"Fluorescence analysis expects a 2D image, got {signal.shape}")

    summary_rows: list[dict[str, object]] = []
    pixel_rows: list[dict[str, object]] = []
    for analysis in analyses:
        for region_name, region_mask in analysis.region_masks.items():
            mask = np.asarray(region_mask) > 0
            coords = np.argwhere(mask)
            values = signal[mask] if np.any(mask) else np.zeros((0,), dtype=np.float32)
            region_mean = float(np.mean(values)) if values.size > 0 else float("nan")
            region_sum = float(np.sum(values)) if values.size > 0 else float("nan")
            pixel_count = int(values.size)

            for y, x in coords:
                pixel_rows.append(
                    {
                        "fc_file": fc_file,
                        "plant_id": analysis.plant_id,
                        "region": region_name,
                        "y": int(y),
                        "x": int(x),
                        "fluorescence": float(signal[int(y), int(x)]),
                    }
                )

            summary_rows.append(
                {"fc_file": fc_file, "plant_id": analysis.plant_id, "parameter": f"mean_fluorescence_{region_name}", "value": region_mean}
            )
            summary_rows.append(
                {"fc_file": fc_file, "plant_id": analysis.plant_id, "parameter": f"n_pixels_{region_name}", "value": pixel_count}
            )
            summary_rows.append(
                {"fc_file": fc_file, "plant_id": analysis.plant_id, "parameter": f"sum_fluorescence_{region_name}", "value": region_sum}
            )
    return summary_rows, pixel_rows


def summarize_hyperspectral_regions(
    sensor_cube: np.ndarray,
    analyses: list[PlantRegionAnalysis],
    source_name: str,
    wavelengths: list[float] | None = None,
    mask_source_label: str = "",
) -> list[dict[str, object]]:
    cube = np.asarray(sensor_cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Hyperspectral analysis expects a 3D cube, got {cube.shape}")
    band_count = int(cube.shape[2])
    if wavelengths is None or len(wavelengths) != band_count:
        wavelengths = [float(i) for i in range(band_count)]

    rows: list[dict[str, object]] = []
    for analysis in analyses:
        for region_name, region_mask in analysis.region_masks.items():
            mask = np.asarray(region_mask) > 0
            if not np.any(mask):
                continue
            pixel_values = cube[mask]
            n_pixels = int(pixel_values.shape[0])
            stats = {
                "mean": np.nanmean(pixel_values, axis=0),
                "sum": np.nansum(pixel_values, axis=0),
                "std": np.nanstd(pixel_values, axis=0),
            }
            for stat_name, values in stats.items():
                row = {
                    "bil_file": source_name,
                    "root_mask_path": mask_source_label,
                    "plant_id": analysis.plant_id,
                    "region": region_name,
                    "n_pixels": n_pixels,
                    "stat": stat_name,
                }
                for wavelength, value in zip(wavelengths, np.asarray(values, dtype=np.float32)):
                    row[f"{float(wavelength):.1f}nm"] = float(value)
                rows.append(row)
    return rows


def render_region_overlay(
    image_rgb: np.ndarray,
    analyses: list[PlantRegionAnalysis],
    alpha: float = 0.55,
) -> np.ndarray:
    base = np.asarray(image_rgb, dtype=np.uint8)
    if base.ndim == 2:
        base = np.repeat(base[:, :, None], 3, axis=2)
    overlay = base.astype(np.float32).copy()
    a = float(max(0.0, min(1.0, alpha)))
    for analysis in analyses:
        for region_name, region_mask in analysis.region_masks.items():
            color = REGION_COLOR_MAP.get(region_name)
            if color is None:
                continue
            pixels = np.asarray(region_mask) > 0
            if not np.any(pixels):
                continue
            overlay[pixels] = overlay[pixels] * (1.0 - a) + np.asarray(color, dtype=np.float32) * a
    return np.clip(overlay, 0, 255).astype(np.uint8)


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(str(key))
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_overlay_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(path)
