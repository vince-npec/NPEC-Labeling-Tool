from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import json

import cv2
import numpy as np

from .analytics_engine import (
    AnalyticsConfig,
    _apply_shoot_crown_lock,
    _assemble_payload_from_owned_masks,
    _build_owned_masks_by_frame,
    _build_owned_shoot_masks_by_frame,
    _build_tip_priors_by_frame,
    _build_track_compartment_hints,
    _derive_lateral_mask_from_root,
    _mask_bbox,
    _measure_crop,
    _prediction_to_index_mask,
    _resolve_anchor_masks,
    _resolve_shoot_masks,
    _resolve_tracking,
    _resolve_tracking_masks,
    _smooth_binary_masks,
)
from .models import DatasetImageItem


Point = tuple[int, int]


@dataclass(slots=True)
class ArchitectureTrackerConfig:
    branch_link_max_distance_px: float = 42.0
    branch_link_max_gap_frames: int = 2
    branch_base_weight: float = 0.65
    branch_tip_weight: float = 0.35
    branch_length_weight: float = 0.10
    min_branch_length_px: float = 6.0
    min_branch_frames: int = 2


def _nearest_distance(points: list[Point], target: Point) -> float:
    if not points:
        return float("inf")
    tx, ty = int(target[0]), int(target[1])
    pts = np.asarray(points, dtype=np.float32)
    dx = pts[:, 0] - float(tx)
    dy = pts[:, 1] - float(ty)
    dist = np.sqrt((dx * dx) + (dy * dy))
    if dist.size <= 0:
        return float("inf")
    return float(np.min(dist))


def _path_length(points: list[Point]) -> float:
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float32)
    diffs = arr[1:] - arr[:-1]
    seg = np.sqrt(np.sum(diffs * diffs, axis=1))
    return float(np.sum(seg))


def _component_branch_candidates(
    lateral_mask_full: np.ndarray,
    main_path_points: list[Point],
    analytics_config: AnalyticsConfig,
    arch_config: ArchitectureTrackerConfig,
) -> list[dict[str, object]]:
    mask = (np.asarray(lateral_mask_full, dtype=np.uint8) > 0).astype(np.uint8)
    if mask.size == 0 or int(np.count_nonzero(mask)) <= 0:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(3, int(analytics_config.min_component_area // 4))
    out: list[dict[str, object]] = []
    for label_id in range(1, int(num_labels)):
        area_px = int(stats[label_id, cv2.CC_STAT_AREA])
        if area_px < min_area:
            continue
        component = (labels == label_id).astype(np.uint8)
        bbox = _mask_bbox(component, 1, mask.shape[:2])
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            continue
        crop = component[y : y + h, x : x + w]
        metrics = _measure_crop(crop, analytics_config)
        path_xy_obj = metrics.get("path_xy")
        path_xy = path_xy_obj if isinstance(path_xy_obj, list) else []
        path_abs = [(int(x + int(pt[0])), int(y + int(pt[1]))) for pt in path_xy if isinstance(pt, (list, tuple)) and len(pt) >= 2]
        if not path_abs:
            ys, xs = np.nonzero(component > 0)
            if xs.size <= 0:
                continue
            cx = int(np.round(np.mean(xs)))
            cy = int(np.round(np.mean(ys)))
            path_abs = [(cx, cy)]
        length_px = max(float(metrics.get("length_px", 0.0)), _path_length(path_abs))
        if length_px < float(max(1.0, arch_config.min_branch_length_px)):
            continue
        if len(path_abs) == 1:
            endpoint_a = path_abs[0]
            endpoint_b = path_abs[0]
        else:
            endpoint_a = path_abs[0]
            endpoint_b = path_abs[-1]
        dist_a = _nearest_distance(main_path_points, endpoint_a)
        dist_b = _nearest_distance(main_path_points, endpoint_b)
        if dist_b < dist_a:
            path_abs = list(reversed(path_abs))
            endpoint_a, endpoint_b = endpoint_b, endpoint_a
            dist_a, dist_b = dist_b, dist_a
        base_point = endpoint_a
        tip_point = endpoint_b
        diameter_px = float(area_px) / max(length_px, 1.0e-6)
        out.append(
            {
                "bbox": bbox,
                "area_px": int(area_px),
                "length_px": float(length_px),
                "length_mm": float(length_px * analytics_config.pixel_size_mm),
                "diameter_px": float(diameter_px),
                "diameter_mm": float(diameter_px * analytics_config.pixel_size_mm),
                "area_mm2": float(float(area_px) * (float(analytics_config.pixel_size_mm) ** 2)),
                "path_points": [[int(pt[0]), int(pt[1])] for pt in path_abs],
                "base_point": [int(base_point[0]), int(base_point[1])],
                "tip_point": [int(tip_point[0]), int(tip_point[1])],
                "base_distance_px": float(dist_a),
                "tip_distance_px": float(dist_b),
                "tips": int(metrics.get("tips", 0)),
                "branches": int(metrics.get("branches", 0)),
            }
        )
    out.sort(key=lambda cand: (int(cand["base_point"][1]), int(cand["base_point"][0]), -float(cand["length_px"])))
    return out


def _branch_link_cost(
    prev_obs: dict[str, object],
    candidate: dict[str, object],
    gap_frames: int,
    config: ArchitectureTrackerConfig,
) -> float:
    prev_base = prev_obs.get("base_point") if isinstance(prev_obs, dict) else None
    prev_tip = prev_obs.get("tip_point") if isinstance(prev_obs, dict) else None
    cand_base = candidate.get("base_point")
    cand_tip = candidate.get("tip_point")
    if not (isinstance(prev_base, list) and len(prev_base) >= 2 and isinstance(cand_base, list) and len(cand_base) >= 2):
        return float("inf")
    base_dist = float(np.hypot(float(prev_base[0]) - float(cand_base[0]), float(prev_base[1]) - float(cand_base[1])))
    tip_dist = 0.0
    if isinstance(prev_tip, list) and len(prev_tip) >= 2 and isinstance(cand_tip, list) and len(cand_tip) >= 2:
        tip_dist = float(np.hypot(float(prev_tip[0]) - float(cand_tip[0]), float(prev_tip[1]) - float(cand_tip[1])))
    prev_len = float(prev_obs.get("length_px", 0.0))
    cand_len = float(candidate.get("length_px", 0.0))
    len_delta = abs(prev_len - cand_len)
    gap_scale = float(max(1, int(gap_frames)))
    max_dist = float(max(1.0, config.branch_link_max_distance_px)) * gap_scale
    norm_base = base_dist / max_dist
    norm_tip = tip_dist / max_dist
    norm_len = len_delta / max(8.0, prev_len, cand_len)
    return (
        float(config.branch_base_weight) * norm_base
        + float(config.branch_tip_weight) * norm_tip
        + float(config.branch_length_weight) * norm_len
    )


def _link_branch_candidates(
    per_frame_candidates: list[dict[str, list[dict[str, object]]]],
    analytics_config: AnalyticsConfig,
    arch_config: ArchitectureTrackerConfig,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]], dict[str, object]]:
    tracks: dict[str, dict[str, object]] = {}
    next_idx_by_plant: dict[str, int] = defaultdict(int)
    frames_out: list[dict[str, object]] = []

    for frame_idx, candidates_by_plant in enumerate(per_frame_candidates):
        frame_branches: list[dict[str, object]] = []
        for plant_id, candidates in sorted(candidates_by_plant.items(), key=lambda kv: str(kv[0])):
            available_track_ids = []
            for track_id, track in tracks.items():
                if str(track.get("plant_id", "")) != str(plant_id):
                    continue
                observations_obj = track.get("observations")
                observations = observations_obj if isinstance(observations_obj, list) else []
                if not observations:
                    continue
                last_obs = observations[-1]
                try:
                    last_frame = int(last_obs.get("frame_index", -999))
                except Exception:
                    last_frame = -999
                gap = int(frame_idx - last_frame)
                if gap <= 0 or gap > int(max(1, arch_config.branch_link_max_gap_frames)):
                    continue
                available_track_ids.append(track_id)

            candidate_pairs: list[tuple[float, str, int]] = []
            for track_id in available_track_ids:
                observations = tracks[track_id].get("observations", [])
                if not isinstance(observations, list) or not observations:
                    continue
                prev_obs = observations[-1]
                try:
                    gap = int(frame_idx - int(prev_obs.get("frame_index", frame_idx)))
                except Exception:
                    gap = 1
                for cand_idx, cand in enumerate(candidates):
                    cost = _branch_link_cost(prev_obs, cand, gap, arch_config)
                    if cost <= 1.50:
                        candidate_pairs.append((float(cost), str(track_id), int(cand_idx)))
            candidate_pairs.sort(key=lambda item: item[0])

            assigned_candidates: set[int] = set()
            assigned_tracks: set[str] = set()
            matches: dict[int, str] = {}
            for cost, track_id, cand_idx in candidate_pairs:
                if track_id in assigned_tracks or cand_idx in assigned_candidates:
                    continue
                matches[int(cand_idx)] = str(track_id)
                assigned_tracks.add(str(track_id))
                assigned_candidates.add(int(cand_idx))

            for cand_idx, candidate in enumerate(candidates):
                if int(cand_idx) in matches:
                    track_id = matches[int(cand_idx)]
                    track = tracks[track_id]
                else:
                    next_idx_by_plant[str(plant_id)] += 1
                    track_id = f"{plant_id}:branch_{next_idx_by_plant[str(plant_id)]:03d}"
                    track = {
                        "branch_id": track_id,
                        "plant_id": str(plant_id),
                        "birth_frame": int(frame_idx),
                        "death_frame": int(frame_idx),
                        "observations": [],
                    }
                    tracks[track_id] = track

                observations = track.get("observations")
                obs_list = observations if isinstance(observations, list) else []
                prev_obs = obs_list[-1] if obs_list else None
                speed_px_day = 0.0
                speed_mm_day = 0.0
                if isinstance(prev_obs, dict):
                    prev_tip = prev_obs.get("tip_point")
                    cand_tip = candidate.get("tip_point")
                    if isinstance(prev_tip, list) and len(prev_tip) >= 2 and isinstance(cand_tip, list) and len(cand_tip) >= 2:
                        dist_px = float(np.hypot(float(prev_tip[0]) - float(cand_tip[0]), float(prev_tip[1]) - float(cand_tip[1])))
                        prev_frame = int(prev_obs.get("frame_index", frame_idx - 1))
                        delta_frames = max(1, int(frame_idx - prev_frame))
                        delta_h = max(1.0e-6, float(analytics_config.timestep_hours) * float(delta_frames))
                        speed_px_day = (dist_px / delta_h) * 24.0
                        speed_mm_day = (dist_px * float(analytics_config.pixel_size_mm) / delta_h) * 24.0
                obs = {
                    **candidate,
                    "branch_id": track_id,
                    "plant_id": str(plant_id),
                    "frame_index": int(frame_idx),
                    "speed_px_day": float(max(0.0, speed_px_day)),
                    "speed_mm_day": float(max(0.0, speed_mm_day)),
                    "status": "continued" if obs_list else "born",
                }
                obs_list.append(obs)
                track["observations"] = obs_list
                track["death_frame"] = int(frame_idx)
                track["max_length_px"] = float(max(float(track.get("max_length_px", 0.0)), float(candidate.get("length_px", 0.0))))
                track["max_length_mm"] = float(max(float(track.get("max_length_mm", 0.0)), float(candidate.get("length_mm", 0.0))))
                track["max_speed_px_day"] = float(max(float(track.get("max_speed_px_day", 0.0)), float(speed_px_day)))
                track["max_speed_mm_day"] = float(max(float(track.get("max_speed_mm_day", 0.0)), float(speed_mm_day)))
                frame_branches.append(obs)

        frames_out.append({"frame_index": int(frame_idx), "branches": frame_branches})

    tracks_out: dict[str, dict[str, object]] = {}
    branch_rows: list[dict[str, object]] = []
    for track_id, track in tracks.items():
        observations_obj = track.get("observations")
        observations = observations_obj if isinstance(observations_obj, list) else []
        if len(observations) < int(max(1, arch_config.min_branch_frames)):
            continue
        speeds_px = [float(obs.get("speed_px_day", 0.0)) for obs in observations if float(obs.get("speed_px_day", 0.0)) > 0.0]
        speeds = [float(obs.get("speed_mm_day", 0.0)) for obs in observations if float(obs.get("speed_mm_day", 0.0)) > 0.0]
        mean_speed_px = float(np.mean(speeds_px)) if speeds_px else 0.0
        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        track_out = {
            **track,
            "frames": int(len(observations)),
            "mean_speed_px_day": float(mean_speed_px),
            "mean_speed_mm_day": float(mean_speed),
            "observations": observations,
        }
        tracks_out[str(track_id)] = track_out
        branch_rows.append(
            {
                "branch_id": str(track_id),
                "plant_id": str(track.get("plant_id", "")),
                "frames": int(len(observations)),
                "birth_frame": int(track.get("birth_frame", 0)),
                "death_frame": int(track.get("death_frame", 0)),
                "pixel_size_mm": float(analytics_config.pixel_size_mm),
                "max_length_px": float(track.get("max_length_px", 0.0)),
                "max_length_mm": float(track.get("max_length_mm", 0.0)),
                "mean_speed_px_day": float(mean_speed_px),
                "mean_speed_mm_day": float(mean_speed),
                "max_speed_px_day": float(track.get("max_speed_px_day", 0.0)),
                "max_speed_mm_day": float(track.get("max_speed_mm_day", 0.0)),
            }
        )

    kept_ids = set(tracks_out.keys())
    for frame in frames_out:
        branches_obj = frame.get("branches")
        branches = branches_obj if isinstance(branches_obj, list) else []
        frame["branches"] = [obs for obs in branches if str(obs.get("branch_id", "")) in kept_ids]

    speed_values = [float(row.get("max_speed_mm_day", 0.0)) for row in branch_rows if float(row.get("max_speed_mm_day", 0.0)) > 0.0]
    summary = {
        "status": "ok" if branch_rows else "no_branches",
        "branch_tracks": int(len(branch_rows)),
        "branch_observations": int(sum(int(row.get("frames", 0)) for row in branch_rows)),
        "plants": int(len({str(row.get("plant_id", "")) for row in branch_rows})),
        "speed_ref_mm_day": float(np.percentile(np.asarray(speed_values, dtype=np.float64), 95.0)) if speed_values else 8.0,
    }
    branch_rows.sort(key=lambda row: (str(row.get("plant_id", "")), int(row.get("birth_frame", 0)), str(row.get("branch_id", ""))))
    return tracks_out, branch_rows, summary


def run_root_architecture_tracker(
    items: list[DatasetImageItem],
    predictions: dict[str, np.ndarray],
    annotations: dict[str, dict[int, np.ndarray]],
    analytics_config: AnalyticsConfig,
    architecture_config: ArchitectureTrackerConfig,
) -> dict[str, object]:
    timeline = [item for item in items if item.uid in predictions]
    if not timeline:
        return {"frames": [], "tracks": {}, "rows": [], "summary": {"status": "no_predictions"}}

    root_masks, mask_source = _resolve_tracking_masks(timeline, predictions, analytics_config)
    anchor_masks, anchor_source = _resolve_anchor_masks(timeline, predictions, analytics_config)
    shoot_masks, shoot_source = _resolve_shoot_masks(
        timeline,
        predictions,
        analytics_config,
        fallback_masks=anchor_masks,
    )
    if analytics_config.temporal_smoothing_enabled:
        root_masks = _smooth_binary_masks(root_masks, analytics_config.temporal_alpha)
    root_masks, shoot_crown_lock_meta = _apply_shoot_crown_lock(root_masks, anchor_masks, analytics_config)

    lateral_mode = "derived_from_root"
    lateral_masks: list[np.ndarray] = [np.zeros_like(mask, dtype=np.uint8) for mask in root_masks]
    lateral_class_id = int(analytics_config.lateral_class_id) if analytics_config.lateral_class_id is not None else None
    if lateral_class_id is not None and lateral_class_id > 0 and lateral_class_id != int(analytics_config.root_class_id):
        lateral_mode = "class_mask"
        for frame_idx, item in enumerate(timeline):
            pred_idx = _prediction_to_index_mask(predictions[item.uid])
            if pred_idx.ndim != 2 or pred_idx.size == 0:
                lateral_masks[frame_idx] = np.zeros_like(root_masks[frame_idx], dtype=np.uint8)
                continue
            lateral_masks[frame_idx] = (pred_idx == np.uint8(lateral_class_id)).astype(np.uint8)

    tracking = _resolve_tracking(root_masks, analytics_config, anchor_masks=anchor_masks, timeline=timeline)
    track_ids = list(tracking.get("track_ids", []))
    track_bboxes = dict(tracking.get("track_bboxes", {}))
    overlap_frames = dict(tracking.get("overlap_frames", {}))

    compartment_hints = _build_track_compartment_hints(
        root_masks=root_masks,
        anchor_masks=anchor_masks,
        track_ids=track_ids,
        track_bboxes=track_bboxes,
        seed_frame=int(tracking.get("seed_frame", -1)),
        config=analytics_config,
    )
    owned_shoot_frames, shoot_tracking_meta = _build_owned_shoot_masks_by_frame(
        shoot_masks=shoot_masks,
        track_ids=track_ids,
        compartment_hints=compartment_hints,
        config=analytics_config,
    )
    owned_root_frames, owned_lateral_frames, ownership_meta = _build_owned_masks_by_frame(
        root_masks,
        lateral_masks,
        owned_shoot_frames,
        track_ids,
        compartment_hints,
        analytics_config,
        tip_priors_by_frame=None,
    )
    analytics_payload = _assemble_payload_from_owned_masks(
        timeline,
        owned_root_frames,
        owned_lateral_frames,
        owned_shoot_frames,
        anchor_masks,
        track_ids,
        overlap_frames,
        analytics_config,
        mask_source,
        anchor_source,
        shoot_crown_lock_meta,
        shoot_tracking_meta,
        compartment_hints,
        {**ownership_meta, "refined_pass": False, "effective_min_component_area": int(tracking.get("effective_min_component_area", max(1, int(analytics_config.min_component_area))))},
        lateral_mode,
        shoot_source,
    )
    if bool(analytics_config.tip_tracking_enabled):
        tip_tracks = analytics_payload.get("tip_tracks", {}) if isinstance(analytics_payload.get("tip_tracks"), dict) else {}
        tip_priors_by_frame = _build_tip_priors_by_frame(tip_tracks, len(timeline), track_ids, analytics_config)
        if tip_priors_by_frame:
            owned_root_frames, owned_lateral_frames, ownership_meta = _build_owned_masks_by_frame(
                root_masks,
                lateral_masks,
                owned_shoot_frames,
                track_ids,
                compartment_hints,
                analytics_config,
                tip_priors_by_frame=tip_priors_by_frame,
            )
            analytics_payload = _assemble_payload_from_owned_masks(
                timeline,
                owned_root_frames,
                owned_lateral_frames,
                owned_shoot_frames,
                anchor_masks,
                track_ids,
                overlap_frames,
                analytics_config,
                mask_source,
                anchor_source,
                shoot_crown_lock_meta,
                shoot_tracking_meta,
                compartment_hints,
                {**ownership_meta, "refined_pass": True, "effective_min_component_area": int(tracking.get("effective_min_component_area", max(1, int(analytics_config.min_component_area))))},
                lateral_mode,
                shoot_source,
            )

    analytics_frames_obj = analytics_payload.get("frames") if isinstance(analytics_payload, dict) else []
    analytics_frames = analytics_frames_obj if isinstance(analytics_frames_obj, list) else []

    per_frame_candidates: list[dict[str, list[dict[str, object]]]] = []
    out_frames: list[dict[str, object]] = []
    for frame_idx, item in enumerate(timeline):
        frame_candidates: dict[str, list[dict[str, object]]] = {}
        plants: dict[str, dict[str, object]] = {}
        analytics_frame = analytics_frames[frame_idx] if frame_idx < len(analytics_frames) and isinstance(analytics_frames[frame_idx], dict) else {}
        frame_tracks = analytics_frame.get("tracks") if isinstance(analytics_frame, dict) else {}
        frame_tracks = frame_tracks if isinstance(frame_tracks, dict) else {}
        owned_lateral_frame = owned_lateral_frames[frame_idx] if frame_idx < len(owned_lateral_frames) else {}
        for track_id in track_ids:
            row = frame_tracks.get(track_id)
            if isinstance(row, dict):
                plants[str(track_id)] = row
                if not bool(row.get("ownership_measurement_valid", True)):
                    frame_candidates[str(track_id)] = []
                    continue
            lateral_full = np.asarray(owned_lateral_frame.get(track_id, np.zeros(item.image.shape[:2], dtype=np.uint8)), dtype=np.uint8)
            if lateral_mode != "class_mask" and isinstance(row, dict):
                x = int(row.get("bbox_x", 0))
                y = int(row.get("bbox_y", 0))
                w = int(row.get("bbox_w", 0))
                h = int(row.get("bbox_h", 0))
                if w > 0 and h > 0:
                    root_full = np.asarray(owned_root_frames[frame_idx].get(track_id, np.zeros(item.image.shape[:2], dtype=np.uint8)), dtype=np.uint8)
                    crop = root_full[y : y + h, x : x + w]
                    root_metrics = _measure_crop(crop, analytics_config)
                    derived = _derive_lateral_mask_from_root(crop, root_metrics)
                    local = np.zeros_like(root_full, dtype=np.uint8)
                    local[y : y + h, x : x + w] = derived
                    lateral_full = local
            main_path_obj = row.get("path_points") if isinstance(row, dict) else []
            main_path = [
                (int(pt[0]), int(pt[1]))
                for pt in main_path_obj
                if isinstance(pt, (list, tuple)) and len(pt) >= 2
            ] if isinstance(main_path_obj, list) else []
            frame_candidates[str(track_id)] = _component_branch_candidates(lateral_full, main_path, analytics_config, architecture_config)
        per_frame_candidates.append(frame_candidates)
        out_frames.append(
            {
                "uid": item.uid,
                "name": item.name,
                "frame_index": int(frame_idx),
                "plants": plants,
                "branches": [],
                "combined_groups": analytics_frame.get("combined_groups", []) if isinstance(analytics_frame, dict) else [],
            }
        )

    branch_tracks, branch_rows, branch_summary = _link_branch_candidates(per_frame_candidates, analytics_config, architecture_config)
    for frame in out_frames:
        idx = int(frame.get("frame_index", 0))
        if 0 <= idx < len(per_frame_candidates):
            frame["branches"] = [
                obs
                for track in branch_tracks.values()
                for obs in (track.get("observations") if isinstance(track.get("observations"), list) else [])
                if int(obs.get("frame_index", -1)) == idx
            ]

    summary = {
        **branch_summary,
        "analytics_tracking_mode": str(analytics_config.tracking_mode),
        "tracking_source": str(tracking.get("source", "internal")),
        "pixel_size_mm": float(analytics_config.pixel_size_mm),
        "timestep_hours": float(analytics_config.timestep_hours),
        "lateral_mode": str(lateral_mode),
        "tip_guided_ownership": bool((analytics_payload.get("summary") or {}).get("tip_guided_ownership", {}).get("refined_pass", False)) if isinstance(analytics_payload, dict) else False,
        "combined_overlap_tracks": int(
            ((analytics_payload.get("summary") or {}).get("measurement_tiers", {}) or {}).get("combined_overlap_tracks", 0)
        ) if isinstance(analytics_payload, dict) else 0,
    }
    return {
        "frames": out_frames,
        "tracks": branch_tracks,
        "rows": branch_rows,
        "summary": summary,
        "analytics_payload": analytics_payload,
    }


def export_architecture_rows_csv(payload: dict[str, object], output_path: Path) -> Path:
    rows_obj = payload.get("rows") if isinstance(payload, dict) else []
    rows = rows_obj if isinstance(rows_obj, list) else []
    fieldnames = [
        "branch_id",
        "plant_id",
        "frames",
        "birth_frame",
        "death_frame",
        "pixel_size_mm",
        "max_length_px",
        "max_length_mm",
        "mean_speed_px_day",
        "mean_speed_mm_day",
        "max_speed_px_day",
        "max_speed_mm_day",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, dict):
                continue
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output_path


def export_architecture_summary_json(payload: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path
