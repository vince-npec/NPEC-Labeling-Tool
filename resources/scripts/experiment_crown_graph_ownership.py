from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
from typing import Iterable

import cv2
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.five_seedling_ownership import (  # noqa: E402
    FiveSeedlingSample,
    build_ownership_targets,
    discover_five_seedling_corpus,
    grouped_split,
    load_instance_masks,
    load_semantic_masks,
    shoot_ordered_slot_mapping,
)


DEFAULT_CORPUS = REPO_ROOT / "data" / "yang_ground_truth"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "crown_graph_ownership_experiment"
OWNER_COLORS_BGR = np.asarray(
    [
        (0, 0, 0),
        (52, 152, 219),
        (46, 204, 113),
        (241, 196, 15),
        (155, 89, 182),
        (230, 126, 34),
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True, slots=True)
class GraphAssignment:
    owner_mask: np.ndarray
    skeleton_owner: np.ndarray
    uncertain_mask: np.ndarray
    seeded_owners: tuple[int, ...]
    node_count: int


@dataclass(frozen=True, slots=True)
class CrownGraphParameters:
    name: str
    ambiguity_absolute_px: float
    ambiguity_relative: float
    primary_position_weight: float
    lateral_position_weight: float
    primary_gap_px: float
    primary_owner_margin_px: float
    lateral_gap_px: float
    lateral_owner_margin_px: float
    lane_component_consensus: float


PARAMETER_CANDIDATES = (
    CrownGraphParameters(
        name="strict",
        ambiguity_absolute_px=30.0,
        ambiguity_relative=0.12,
        primary_position_weight=0.030,
        lateral_position_weight=0.020,
        primary_gap_px=90.0,
        primary_owner_margin_px=60.0,
        lateral_gap_px=160.0,
        lateral_owner_margin_px=80.0,
        lane_component_consensus=0.995,
    ),
    CrownGraphParameters(
        name="balanced",
        ambiguity_absolute_px=24.0,
        ambiguity_relative=0.08,
        primary_position_weight=0.025,
        lateral_position_weight=0.015,
        primary_gap_px=150.0,
        primary_owner_margin_px=45.0,
        lateral_gap_px=240.0,
        lateral_owner_margin_px=60.0,
        lane_component_consensus=0.985,
    ),
    CrownGraphParameters(
        name="coverage",
        ambiguity_absolute_px=16.0,
        ambiguity_relative=0.05,
        primary_position_weight=0.018,
        lateral_position_weight=0.010,
        primary_gap_px=240.0,
        primary_owner_margin_px=30.0,
        lateral_gap_px=360.0,
        lateral_owner_margin_px=40.0,
        lane_component_consensus=0.970,
    ),
)
DEFAULT_PARAMETERS = PARAMETER_CANDIDATES[1]
CHALLENGE_SAMPLE_IDS = ("78-3", "73-4", "79-3", "77-4", "1-4")


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _crown_points(sample: FiveSeedlingSample) -> dict[int, tuple[float, float]]:
    """Return left-to-right crown points from instance shoots.

    The per-shoot instances are used only as oracle crown detections. Root and
    lateral instance labels are never exposed to the ownership algorithm.
    """
    source_to_slot = shoot_ordered_slot_mapping(sample, expected_count=5)
    shoots = load_instance_masks(sample, "shoot")
    points: dict[int, tuple[float, float]] = {}
    for source_owner, mask in shoots.items():
        slot = source_to_slot.get(int(source_owner))
        ys, xs = np.where(np.asarray(mask, dtype=np.uint8) > 0)
        if slot is None or xs.size <= 0:
            continue
        lower_cutoff = float(np.quantile(ys.astype(np.float64), 0.85))
        lower = ys >= lower_cutoff
        if int(np.count_nonzero(lower)) < 5:
            lower = np.ones_like(ys, dtype=bool)
        points[int(slot)] = (
            float(np.median(ys[lower].astype(np.float64))),
            float(np.median(xs[lower].astype(np.float64))),
        )
    return points


def _skeleton_graph(skeleton: np.ndarray) -> tuple[np.ndarray, csr_matrix]:
    ys, xs = np.where(np.asarray(skeleton, dtype=bool))
    coords = np.column_stack((ys, xs)).astype(np.int32, copy=False)
    node_count = int(coords.shape[0])
    if node_count <= 0:
        return coords, csr_matrix((0, 0), dtype=np.float32)

    height, width = skeleton.shape[:2]
    node_map = np.full((height, width), -1, dtype=np.int32)
    node_map[ys, xs] = np.arange(node_count, dtype=np.int32)
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    source_ids = np.arange(node_count, dtype=np.int32)
    for dy, dx, weight in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0**0.5), (1, -1, 2.0**0.5)):
        ny = ys + int(dy)
        nx = xs + int(dx)
        inside = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
        targets = np.full(node_count, -1, dtype=np.int32)
        targets[inside] = node_map[ny[inside], nx[inside]]
        valid = targets >= 0
        src = source_ids[valid]
        dst = targets[valid]
        weights = np.full(src.shape, float(weight), dtype=np.float32)
        row_parts.extend((src, dst))
        col_parts.extend((dst, src))
        data_parts.extend((weights, weights))
    if not row_parts:
        return coords, csr_matrix((node_count, node_count), dtype=np.float32)
    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    data = np.concatenate(data_parts)
    graph = coo_matrix((data, (rows, cols)), shape=(node_count, node_count)).tocsr()
    return coords, graph


def _nearest_node_indices(
    coords: np.ndarray,
    points_by_owner: dict[int, Iterable[tuple[float, float]]],
    *,
    max_distance: float,
) -> dict[int, np.ndarray]:
    if coords.size <= 0:
        return {}
    tree = cKDTree(coords.astype(np.float64))
    seeds: dict[int, np.ndarray] = {}
    for owner, raw_points in points_by_owner.items():
        points = np.asarray(list(raw_points), dtype=np.float64)
        if points.size <= 0:
            continue
        distances, indices = tree.query(points, k=1)
        keep = np.asarray(distances, dtype=np.float64) <= float(max_distance)
        accepted = np.unique(np.asarray(indices, dtype=np.int64)[keep])
        if accepted.size > 0:
            seeds[int(owner)] = accepted
    return seeds


def _multi_owner_distances(
    graph: csr_matrix,
    seeds_by_owner: dict[int, np.ndarray],
    *,
    owner_count: int = 5,
) -> np.ndarray:
    node_count = int(graph.shape[0])
    if node_count <= 0:
        return np.full((owner_count, 0), np.inf, dtype=np.float64)
    extra_rows: list[int] = []
    extra_cols: list[int] = []
    for owner in range(1, owner_count + 1):
        super_node = node_count + owner - 1
        for seed in np.asarray(seeds_by_owner.get(owner, ()), dtype=np.int64):
            extra_rows.extend((super_node, int(seed)))
            extra_cols.extend((int(seed), super_node))
    base = graph.tocoo()
    rows = np.concatenate(
        (base.row.astype(np.int64), np.asarray(extra_rows, dtype=np.int64))
    )
    cols = np.concatenate(
        (base.col.astype(np.int64), np.asarray(extra_cols, dtype=np.int64))
    )
    data = np.concatenate(
        (base.data.astype(np.float32), np.zeros(len(extra_rows), dtype=np.float32))
    )
    extended = coo_matrix(
        (data, (rows, cols)),
        shape=(node_count + owner_count, node_count + owner_count),
    ).tocsr()
    source_nodes = np.arange(node_count, node_count + owner_count, dtype=np.int64)
    return np.asarray(
        dijkstra(extended, directed=False, indices=source_nodes)[:, :node_count],
        dtype=np.float64,
    )


def _project_skeleton_labels(
    binary_mask: np.ndarray,
    skeleton: np.ndarray,
    node_labels: np.ndarray,
    node_uncertain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    owner = np.zeros(binary_mask.shape[:2], dtype=np.uint8)
    uncertain = np.zeros_like(owner)
    skeleton_coords = np.column_stack(np.where(skeleton))
    foreground_coords = np.column_stack(np.where(np.asarray(binary_mask, dtype=bool)))
    if skeleton_coords.size <= 0 or foreground_coords.size <= 0:
        return owner, uncertain
    nearest = cKDTree(skeleton_coords.astype(np.float64)).query(
        foreground_coords.astype(np.float64),
        k=1,
    )[1]
    owner[foreground_coords[:, 0], foreground_coords[:, 1]] = np.asarray(
        node_labels, dtype=np.uint8
    )[nearest]
    uncertain[foreground_coords[:, 0], foreground_coords[:, 1]] = np.asarray(
        node_uncertain, dtype=np.uint8
    )[nearest]
    return owner, uncertain


def _assign_graph(
    binary_mask: np.ndarray,
    points_by_owner: dict[int, Iterable[tuple[float, float]]],
    crown_x: dict[int, float],
    *,
    max_seed_distance: float,
    ambiguity_absolute_px: float,
    ambiguity_relative: float,
    lateral_position_weight: float,
) -> GraphAssignment:
    binary = np.asarray(binary_mask, dtype=bool)
    skeleton = skeletonize(binary)
    coords, graph = _skeleton_graph(skeleton)
    seeds = _nearest_node_indices(
        coords,
        points_by_owner,
        max_distance=max_seed_distance,
    )
    if coords.size <= 0 or not seeds:
        empty = np.zeros(binary.shape[:2], dtype=np.uint8)
        return GraphAssignment(empty, empty, binary.astype(np.uint8), tuple(), int(coords.shape[0]))

    distances = _multi_owner_distances(graph, seeds)
    for owner in range(1, 6):
        if owner not in crown_x:
            distances[owner - 1, :] = np.inf
            continue
        distances[owner - 1, :] += (
            float(lateral_position_weight)
            * np.abs(coords[:, 1].astype(np.float64) - float(crown_x[owner]))
        )
    order = np.argsort(distances, axis=0)
    best_owner = order[0] + 1
    best = np.take_along_axis(distances, order[0:1], axis=0)[0]
    second = np.take_along_axis(distances, order[1:2], axis=0)[0]
    finite = np.isfinite(best)
    gap = np.full(best.shape, np.inf, dtype=np.float64)
    finite_pair = np.isfinite(best) & np.isfinite(second)
    gap[finite_pair] = second[finite_pair] - best[finite_pair]
    threshold = np.maximum(
        float(ambiguity_absolute_px),
        float(ambiguity_relative) * np.maximum(best, 1.0),
    )
    ambiguous = finite & np.isfinite(second) & (gap <= threshold)
    node_labels = np.where(finite & ~ambiguous, best_owner, 0).astype(np.uint8)
    node_uncertain = (~finite | ambiguous).astype(np.uint8)
    owner, uncertain = _project_skeleton_labels(
        binary,
        skeleton,
        node_labels,
        node_uncertain,
    )
    skeleton_owner = np.zeros(binary.shape[:2], dtype=np.uint8)
    skeleton_owner[coords[:, 0], coords[:, 1]] = node_labels
    return GraphAssignment(
        owner_mask=owner,
        skeleton_owner=skeleton_owner,
        uncertain_mask=uncertain,
        seeded_owners=tuple(sorted(seeds)),
        node_count=int(coords.shape[0]),
    )


def _attachment_points(
    child_mask: np.ndarray,
    parent_owner: np.ndarray,
    *,
    max_distance: float,
    owner_count: int = 5,
) -> dict[int, list[tuple[float, float]]]:
    child_coords = np.column_stack(np.where(np.asarray(child_mask, dtype=bool)))
    parent_coords = np.column_stack(np.where(np.asarray(parent_owner, dtype=np.uint8) > 0))
    if child_coords.size <= 0 or parent_coords.size <= 0:
        return {}
    parent_labels = parent_owner[parent_coords[:, 0], parent_coords[:, 1]]
    distances, nearest = cKDTree(parent_coords.astype(np.float64)).query(
        child_coords.astype(np.float64),
        k=1,
    )
    points: dict[int, list[tuple[float, float]]] = {}
    for owner in range(1, owner_count + 1):
        selected = (
            (np.asarray(distances, dtype=np.float64) <= float(max_distance))
            & (parent_labels[np.asarray(nearest, dtype=np.int64)] == owner)
        )
        coords = child_coords[selected]
        if coords.size <= 0:
            continue
        # Dense contact regions do not need thousands of equivalent super-source edges.
        stride = max(1, int(np.ceil(coords.shape[0] / 128)))
        points[owner] = [
            (float(y), float(x))
            for y, x in coords[::stride]
        ]
    return points


def _bridge_unassigned_components(
    binary_mask: np.ndarray,
    owner_mask: np.ndarray,
    reference_owner: np.ndarray,
    *,
    max_gap: float,
    owner_margin: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Assign wholly unseeded components only when one owner is uniquely nearby."""
    binary = np.asarray(binary_mask, dtype=np.uint8) > 0
    owner = np.asarray(owner_mask, dtype=np.uint8).copy()
    recovered = np.zeros_like(owner)
    trees: dict[int, cKDTree] = {}
    for candidate_owner in range(1, 6):
        coords = np.column_stack(
            np.where(np.asarray(reference_owner, dtype=np.uint8) == candidate_owner)
        )
        if coords.size > 0:
            trees[candidate_owner] = cKDTree(coords.astype(np.float64))
    if not trees:
        return owner, recovered, 0, 0

    component_count, components = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    recovered_components = 0
    recovered_pixels = 0
    for component_id in range(1, int(component_count)):
        component = components == component_id
        if np.any(owner[component] > 0):
            continue
        coords = np.column_stack(np.where(component))
        if coords.size <= 0:
            continue
        stride = max(1, int(np.ceil(coords.shape[0] / 1024)))
        sampled = coords[::stride].astype(np.float64)
        candidates: list[tuple[float, int]] = []
        for candidate_owner, tree in trees.items():
            distances = tree.query(sampled, k=1)[0]
            candidates.append((float(np.min(distances)), int(candidate_owner)))
        candidates.sort()
        best_distance, best_owner = candidates[0]
        second_distance = candidates[1][0] if len(candidates) > 1 else np.inf
        if best_distance > float(max_gap):
            continue
        if np.isfinite(second_distance) and (second_distance - best_distance) < float(owner_margin):
            continue
        owner[component] = np.uint8(best_owner)
        recovered[component] = 1
        recovered_components += 1
        recovered_pixels += int(coords.shape[0])
    return owner, recovered, recovered_components, recovered_pixels


def _fill_lane_contained_components(
    binary_mask: np.ndarray,
    owner_mask: np.ndarray,
    crowns: dict[int, tuple[float, float]],
    *,
    minimum_consensus: float = 0.985,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Fill only disconnected fragments lying almost entirely in one crown lane."""
    binary = np.asarray(binary_mask, dtype=np.uint8) > 0
    owner = np.asarray(owner_mask, dtype=np.uint8).copy()
    recovered = np.zeros_like(owner)
    slots = np.asarray(sorted(crowns), dtype=np.uint8)
    if slots.size <= 0:
        return owner, recovered, 0, 0
    crown_x = np.asarray([crowns[int(slot)][1] for slot in slots], dtype=np.float64)
    component_count, components = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    recovered_components = 0
    recovered_pixels = 0
    for component_id in range(1, int(component_count)):
        component = components == component_id
        if np.any(owner[component] > 0):
            continue
        ys, xs = np.where(component)
        if xs.size <= 0:
            continue
        nearest = np.argmin(
            np.abs(xs[:, None].astype(np.float64) - crown_x[None, :]),
            axis=1,
        )
        counts = np.bincount(nearest, minlength=int(slots.size))
        winner_index = int(np.argmax(counts))
        consensus = float(counts[winner_index] / max(1, xs.size))
        if consensus < float(minimum_consensus):
            continue
        winner = int(slots[winner_index])
        owner[component] = np.uint8(winner)
        recovered[component] = 1
        recovered_components += 1
        recovered_pixels += int(xs.size)
    return owner, recovered, recovered_components, recovered_pixels


def crown_graph_ownership(
    primary_mask: np.ndarray,
    lateral_mask: np.ndarray,
    crowns: dict[int, tuple[float, float]],
    parameters: CrownGraphParameters = DEFAULT_PARAMETERS,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    shape_hw = primary_mask.shape[:2]
    crown_x = {owner: float(point[1]) for owner, point in crowns.items()}
    primary_points = {owner: [point] for owner, point in crowns.items()}
    scale = float(max(shape_hw)) / 4267.0
    primary = _assign_graph(
        primary_mask,
        primary_points,
        crown_x,
        max_seed_distance=max(80.0, 260.0 * scale),
        ambiguity_absolute_px=max(6.0, parameters.ambiguity_absolute_px * scale),
        ambiguity_relative=parameters.ambiguity_relative,
        lateral_position_weight=parameters.primary_position_weight,
    )
    primary_owner, primary_recovered, primary_recovered_components, primary_recovered_pixels = (
        _bridge_unassigned_components(
            primary_mask,
            primary.owner_mask,
            primary.owner_mask,
            max_gap=max(20.0, parameters.primary_gap_px * scale),
            owner_margin=max(10.0, parameters.primary_owner_margin_px * scale),
        )
    )
    primary_owner, primary_lane_recovered, primary_lane_components, primary_lane_pixels = (
        _fill_lane_contained_components(
            primary_mask,
            primary_owner,
            crowns,
            minimum_consensus=parameters.lane_component_consensus,
        )
    )
    lateral_points = _attachment_points(
        lateral_mask,
        primary_owner,
        max_distance=max(8.0, 36.0 * scale),
    )
    lateral = _assign_graph(
        lateral_mask,
        lateral_points,
        crown_x,
        max_seed_distance=max(8.0, 18.0 * scale),
        ambiguity_absolute_px=max(5.0, parameters.ambiguity_absolute_px * 0.75 * scale),
        ambiguity_relative=parameters.ambiguity_relative,
        lateral_position_weight=parameters.lateral_position_weight,
    )
    lateral_owner, lateral_recovered, lateral_recovered_components, lateral_recovered_pixels = (
        _bridge_unassigned_components(
            lateral_mask,
            lateral.owner_mask,
            primary_owner,
            max_gap=max(30.0, parameters.lateral_gap_px * scale),
            owner_margin=max(12.0, parameters.lateral_owner_margin_px * scale),
        )
    )
    lateral_owner, lateral_lane_recovered, lateral_lane_components, lateral_lane_pixels = (
        _fill_lane_contained_components(
            lateral_mask,
            lateral_owner,
            crowns,
            minimum_consensus=parameters.lane_component_consensus,
        )
    )
    owner = np.zeros(shape_hw, dtype=np.uint8)
    owner[np.asarray(primary_mask, dtype=bool)] = primary_owner[
        np.asarray(primary_mask, dtype=bool)
    ]
    owner[np.asarray(lateral_mask, dtype=bool)] = lateral_owner[
        np.asarray(lateral_mask, dtype=bool)
    ]
    uncertain = np.zeros(shape_hw, dtype=np.uint8)
    uncertain[np.asarray(primary_mask, dtype=bool)] = primary.uncertain_mask[
        np.asarray(primary_mask, dtype=bool)
    ]
    uncertain[np.asarray(lateral_mask, dtype=bool)] = lateral.uncertain_mask[
        np.asarray(lateral_mask, dtype=bool)
    ]
    uncertain[primary_recovered > 0] = 0
    uncertain[lateral_recovered > 0] = 0
    uncertain[primary_lane_recovered > 0] = 0
    uncertain[lateral_lane_recovered > 0] = 0
    metadata = {
        "primary_seeded_owners": list(primary.seeded_owners),
        "lateral_seeded_owners": list(lateral.seeded_owners),
        "primary_graph_nodes": int(primary.node_count),
        "lateral_graph_nodes": int(lateral.node_count),
        "primary_gap_recovered_components": int(primary_recovered_components),
        "primary_gap_recovered_pixels": int(primary_recovered_pixels),
        "lateral_gap_recovered_components": int(lateral_recovered_components),
        "lateral_gap_recovered_pixels": int(lateral_recovered_pixels),
        "primary_lane_recovered_components": int(primary_lane_components),
        "primary_lane_recovered_pixels": int(primary_lane_pixels),
        "lateral_lane_recovered_components": int(lateral_lane_components),
        "lateral_lane_recovered_pixels": int(lateral_lane_pixels),
        "parameter_set": parameters.name,
    }
    return owner, uncertain, metadata


def _spatial_lane_owner(
    root_union: np.ndarray,
    crowns: dict[int, tuple[float, float]],
) -> np.ndarray:
    owner = np.zeros(root_union.shape[:2], dtype=np.uint8)
    if not crowns:
        return owner
    root_coords = np.column_stack(np.where(np.asarray(root_union, dtype=bool)))
    if root_coords.size <= 0:
        return owner
    slots = np.asarray(sorted(crowns), dtype=np.uint8)
    crown_x = np.asarray([crowns[int(slot)][1] for slot in slots], dtype=np.float64)
    nearest = np.argmin(
        np.abs(root_coords[:, 1:2].astype(np.float64) - crown_x[None, :]),
        axis=1,
    )
    owner[root_coords[:, 0], root_coords[:, 1]] = slots[nearest]
    return owner


def _score_owner(
    predicted: np.ndarray,
    expected: np.ndarray,
    valid_pixels: np.ndarray,
) -> dict[str, float]:
    valid = np.asarray(valid_pixels, dtype=bool)
    pred = np.asarray(predicted, dtype=np.uint8)
    truth = np.asarray(expected, dtype=np.uint8)
    valid_count = int(np.count_nonzero(valid))
    assigned = valid & (pred > 0)
    assigned_count = int(np.count_nonzero(assigned))
    correct = valid & (pred == truth)
    dice: list[float] = []
    length_errors: list[float] = []
    for owner in range(1, 6):
        pred_owner = valid & (pred == owner)
        truth_owner = valid & (truth == owner)
        intersection = int(np.count_nonzero(pred_owner & truth_owner))
        total = int(np.count_nonzero(pred_owner)) + int(np.count_nonzero(truth_owner))
        if int(np.count_nonzero(truth_owner)) > 0:
            dice.append(0.0 if total <= 0 else float(2 * intersection / total))
            truth_length = int(np.count_nonzero(skeletonize(truth_owner)))
            pred_length = int(np.count_nonzero(skeletonize(pred_owner)))
            if truth_length > 0:
                length_errors.append(float(abs(pred_length - truth_length) / truth_length))
    return {
        "pixel_accuracy": 0.0 if valid_count <= 0 else float(np.count_nonzero(correct) / valid_count),
        "macro_dice": float(np.mean(dice)) if dice else 0.0,
        "coverage": 0.0 if valid_count <= 0 else float(assigned_count / valid_count),
        "selective_accuracy": (
            0.0
            if assigned_count <= 0
            else float(np.count_nonzero(correct & assigned) / assigned_count)
        ),
        "wrong_assignment_fraction": (
            0.0
            if valid_count <= 0
            else float(np.count_nonzero(assigned & (pred != truth)) / valid_count)
        ),
        "unassigned_fraction": (
            0.0
            if valid_count <= 0
            else float(np.count_nonzero(valid & (pred == 0)) / valid_count)
        ),
        "root_length_mape": float(np.mean(length_errors)) if length_errors else 0.0,
    }


def _comparison_visual(
    image_rgb: np.ndarray,
    truth: np.ndarray,
    lane: np.ndarray,
    graph: np.ndarray,
    uncertain: np.ndarray,
    output_path: Path,
) -> None:
    panels: list[np.ndarray] = []
    for title, owner in (("Ground truth", truth), ("Spatial lanes", lane), ("Crown graph", graph)):
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        colors = OWNER_COLORS_BGR[np.asarray(owner, dtype=np.uint8)]
        mask = owner > 0
        bgr[mask] = cv2.addWeighted(bgr[mask], 0.25, colors[mask], 0.75, 0.0)
        if title == "Crown graph":
            unknown = np.asarray(uncertain, dtype=bool)
            bgr[unknown] = (64, 64, 64)
        cv2.putText(
            bgr,
            title,
            (35, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (255, 255, 255),
            5,
            cv2.LINE_AA,
        )
        panels.append(bgr)
    target_height = 1000
    resized = [
        cv2.resize(
            panel,
            (int(round(panel.shape[1] * target_height / panel.shape[0])), target_height),
            interpolation=cv2.INTER_AREA,
        )
        for panel in panels
    ]
    cv2.imwrite(str(output_path), np.concatenate(resized, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 92])


def _deterministic_visual_samples(
    samples: list[FiveSeedlingSample],
    *,
    count: int,
    seed: int,
) -> set[str]:
    by_plate: dict[str, list[FiveSeedlingSample]] = {}
    for sample in samples:
        by_plate.setdefault(sample.plate_group, []).append(sample)
    groups = sorted(by_plate)
    selected_groups = random.Random(int(seed)).sample(groups, k=min(int(count), len(groups)))
    return {
        sorted(by_plate[group], key=lambda sample: sample.sample_id)[0].sample_id
        for group in selected_groups
    }


def run_experiment(
    corpus: Path,
    output_dir: Path,
    *,
    samples_override: list[FiveSeedlingSample] | None = None,
    parameters: CrownGraphParameters = DEFAULT_PARAMETERS,
    limit: int | None = None,
    visual_count: int = 5,
    seed: int = 20260723,
) -> dict[str, object]:
    samples = (
        list(samples_override)
        if samples_override is not None
        else discover_five_seedling_corpus(corpus)
    )
    if limit is not None and int(limit) > 0:
        samples = samples[: int(limit)]
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "comparisons"
    visual_dir.mkdir(parents=True, exist_ok=True)
    visual_ids = _deterministic_visual_samples(samples, count=visual_count, seed=seed)

    rows: list[dict[str, object]] = []
    for index, sample in enumerate(samples, start=1):
        semantic = load_semantic_masks(sample)
        primary = semantic["root"] > 0
        lateral = semantic["lateral"] > 0
        root_union = primary | lateral
        target = build_ownership_targets(sample, owner_organs=("root", "lateral"))
        crowns = _crown_points(sample)
        lane_owner = _spatial_lane_owner(root_union, crowns)
        graph_owner, uncertain, metadata = crown_graph_ownership(
            primary,
            lateral,
            crowns,
            parameters,
        )
        lane_scores = _score_owner(lane_owner, target.owner_mask, target.valid_owner_pixels)
        graph_scores = _score_owner(graph_owner, target.owner_mask, target.valid_owner_pixels)
        row: dict[str, object] = {
            "sample_id": sample.sample_id,
            "plate_group": sample.plate_group,
            "crown_count": len(crowns),
            "valid_owner_pixels": int(np.count_nonzero(target.valid_owner_pixels)),
            "instance_conflict_pixels_ignored": int(np.count_nonzero(target.conflict_pixels)),
            **{f"lane_{key}": value for key, value in lane_scores.items()},
            **{f"graph_{key}": value for key, value in graph_scores.items()},
            "graph_accuracy_delta": float(
                graph_scores["pixel_accuracy"] - lane_scores["pixel_accuracy"]
            ),
            "graph_dice_delta": float(graph_scores["macro_dice"] - lane_scores["macro_dice"]),
            **metadata,
        }
        rows.append(row)
        if sample.sample_id in visual_ids:
            _comparison_visual(
                _read_rgb(sample.image_path),
                target.owner_mask,
                lane_owner,
                graph_owner,
                uncertain,
                visual_dir / f"{sample.sample_id}__comparison.jpg",
            )
        print(
            f"[{index:02d}/{len(samples):02d}] {sample.sample_id}: "
            f"lane={lane_scores['pixel_accuracy']:.3f}, "
            f"graph={graph_scores['pixel_accuracy']:.3f}, "
            f"coverage={graph_scores['coverage']:.3f}",
            flush=True,
        )

    csv_path = output_dir / "crown_graph_ownership_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)

    metric_names = (
        "pixel_accuracy",
        "macro_dice",
        "coverage",
        "selective_accuracy",
        "wrong_assignment_fraction",
        "unassigned_fraction",
        "root_length_mape",
    )
    means = {
        method: {
            metric: (
                float(np.mean([float(row[f"{method}_{metric}"]) for row in rows]))
                if rows
                else None
            )
            for metric in metric_names
        }
        for method in ("lane", "graph")
    }
    report = {
        "corpus": str(corpus),
        "sample_count": len(rows),
        "plate_group_count": len({sample.plate_group for sample in samples}),
        "deterministic_seed": int(seed),
        "parameter_set": {
            field: getattr(parameters, field)
            for field in parameters.__dataclass_fields__
        },
        "visualized_samples": sorted(visual_ids),
        "input_scope": {
            "semantic_masks": "perfect hand-labelled visible primary and lateral masks",
            "crown_seeds": "oracle per-shoot instance masks, ordered left-to-right",
            "owner_truth": "per-root and per-lateral instances used only for scoring",
            "bacterial_gaps": "left disconnected; never filled as measured root",
            "crossing_policy": "near-tied graph territories are owner 0 (unknown)",
        },
        "methods": {
            "lane": "nearest oracle crown x for every visible root pixel",
            "graph": (
                "primary skeleton geodesics from crowns; lateral skeleton geodesics "
                "from confident primary attachments; uniquely attributable short gaps and "
                "lane-contained fragments recovered; ambiguous crossings remain unassigned"
            ),
        },
        "mean": means,
        "graph_minus_lane": {
            metric: (
                None
                if means["graph"][metric] is None
                else float(means["graph"][metric] - means["lane"][metric])
            )
            for metric in metric_names
        },
        "graph_better_accuracy_cases": int(
            sum(float(row["graph_accuracy_delta"]) > 0.0 for row in rows)
        ),
        "graph_better_dice_cases": int(
            sum(float(row["graph_dice_delta"]) > 0.0 for row in rows)
        ),
        "case_csv": str(csv_path),
        "comparison_dir": str(visual_dir),
    }
    report_path = output_dir / "crown_graph_ownership_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _training_tuning_samples(
    training: list[FiveSeedlingSample],
    *,
    group_count: int,
    seed: int,
) -> list[FiveSeedlingSample]:
    by_group: dict[str, list[FiveSeedlingSample]] = {}
    for sample in training:
        by_group.setdefault(sample.plate_group, []).append(sample)
    groups = sorted(by_group)
    selected_groups = random.Random(int(seed)).sample(
        groups,
        k=min(max(1, int(group_count)), len(groups)),
    )
    return [
        sorted(by_group[group], key=lambda sample: sample.sample_id)[0]
        for group in sorted(selected_groups)
    ]


def run_grouped_protocol(
    corpus: Path,
    output_dir: Path,
    *,
    seed: int,
    tuning_group_count: int = 12,
) -> dict[str, object]:
    all_samples = discover_five_seedling_corpus(corpus)
    training, holdout = grouped_split(all_samples)
    challenge_lookup = {sample.sample_id: sample for sample in holdout}
    missing_challenge = [
        sample_id for sample_id in CHALLENGE_SAMPLE_IDS if sample_id not in challenge_lookup
    ]
    if missing_challenge:
        raise ValueError(
            "Challenge samples are not wholly contained in grouped_split holdout: "
            + ", ".join(missing_challenge)
        )
    tuning_samples = _training_tuning_samples(
        training,
        group_count=tuning_group_count,
        seed=seed,
    )
    tuning_root = output_dir / "tuning"
    candidate_reports: dict[str, dict[str, object]] = {}
    candidate_objectives: dict[str, float] = {}
    for parameters in PARAMETER_CANDIDATES:
        report = run_experiment(
            corpus,
            tuning_root / parameters.name,
            samples_override=tuning_samples,
            parameters=parameters,
            visual_count=0,
            seed=seed,
        )
        candidate_reports[parameters.name] = report
        graph_mean = dict(report["mean"]["graph"])
        objective = float(graph_mean["macro_dice"]) - (
            0.25 * float(graph_mean["wrong_assignment_fraction"])
        )
        candidate_objectives[parameters.name] = objective

    selected_name = max(
        candidate_objectives,
        key=lambda name: (candidate_objectives[name], name),
    )
    selected = next(
        parameters for parameters in PARAMETER_CANDIDATES if parameters.name == selected_name
    )
    holdout_report = run_experiment(
        corpus,
        output_dir / "broader_holdout",
        samples_override=holdout,
        parameters=selected,
        visual_count=0,
        seed=seed,
    )
    challenge_samples = [challenge_lookup[sample_id] for sample_id in CHALLENGE_SAMPLE_IDS]
    challenge_report = run_experiment(
        corpus,
        output_dir / "challenge_five",
        samples_override=challenge_samples,
        parameters=selected,
        visual_count=5,
        seed=seed,
    )
    report = {
        "protocol": "grouped_split_train_tuning_then_frozen_holdout",
        "corpus": str(corpus),
        "seed": int(seed),
        "split": {
            "training_samples": len(training),
            "training_plate_groups": len({sample.plate_group for sample in training}),
            "holdout_samples": len(holdout),
            "holdout_plate_groups": len({sample.plate_group for sample in holdout}),
            "tuning_samples": [sample.sample_id for sample in tuning_samples],
            "tuning_plate_groups": [sample.plate_group for sample in tuning_samples],
            "challenge_samples": list(CHALLENGE_SAMPLE_IDS),
        },
        "selection": {
            "objective": "training macro_dice - 0.25 * training wrong_assignment_fraction",
            "candidate_objectives": candidate_objectives,
            "selected_parameter_set": selected_name,
            "selected_parameters": {
                field: getattr(selected, field)
                for field in selected.__dataclass_fields__
            },
            "challenge_used_for_tuning": False,
        },
        "tuning_candidate_means": {
            name: report["mean"] for name, report in candidate_reports.items()
        },
        "broader_holdout": holdout_report,
        "challenge_five": challenge_report,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "grouped_protocol_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experimental crown-seeded graph ownership benchmark."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--visual-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--protocol",
        choices=("grouped", "all"),
        default="grouped",
        help="Use leakage-safe grouped tuning/holdout protocol or a single all-sample run.",
    )
    parser.add_argument("--tuning-groups", type=int, default=12)
    args = parser.parse_args()
    if args.protocol == "grouped":
        run_grouped_protocol(
            args.corpus,
            args.output,
            seed=args.seed,
            tuning_group_count=max(1, args.tuning_groups),
        )
        return 0
    report = run_experiment(
        args.corpus,
        args.output,
        limit=None if args.limit <= 0 else args.limit,
        visual_count=max(0, args.visual_count),
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
