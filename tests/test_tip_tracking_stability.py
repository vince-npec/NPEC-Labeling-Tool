from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from resources.analytics_engine import (
    AnalyticsConfig,
    _assemble_payload_from_owned_masks,
    _assign_pixels_to_track_seeds,
    _assign_tip_links,
    _build_owned_masks_by_frame,
    _build_tip_priors_by_frame,
    _skeleton_graph_path,
    _track_auto,
    run_temporal_analytics,
)
from resources.models import DatasetImageItem


class TipTrackingStabilityTests(unittest.TestCase):
    def test_crown_anchored_path_prefers_long_root_below_small_top_fragment(self) -> None:
        skeleton = np.zeros((150, 80), dtype=np.uint8)
        skeleton[2:11, 8] = 1
        skeleton[20:140, 42] = 1

        path, _tips, _branches = _skeleton_graph_path(
            skeleton,
            base_hint_xy=(42.0, 5.0),
        )

        self.assertGreater(len(path), 100)
        self.assertEqual(path[0][0], 42)
        self.assertLessEqual(path[0][1], 22)
        self.assertGreaterEqual(path[-1][1], 138)

    def test_auto_tracker_rejects_large_cross_lane_jump(self) -> None:
        h, w = 220, 220
        frame0 = np.zeros((h, w), dtype=np.uint8)
        frame1 = np.zeros((h, w), dtype=np.uint8)
        frame0[20:180, 15] = 1
        frame1[20:180, 170] = 1

        tracking = _track_auto(
            [frame0, frame1],
            AnalyticsConfig(
                min_component_area=1,
                bbox_padding=0,
                tracking_search_margin=24,
                expected_track_count=1,
            ),
            anchor_masks=[np.zeros_like(frame0), np.zeros_like(frame1)],
        )

        bbox = tracking["track_bboxes"]["plant_01"][1]
        self.assertLess(int(bbox[0]), 60)
        self.assertLess(int(bbox[2]), 20)

    def test_empty_track_fallback_does_not_duplicate_sibling_root_pixels(self) -> None:
        h, w = 28, 28
        root = np.zeros((h, w), dtype=np.uint8)
        root[4:24, 6] = 1
        lateral = np.zeros_like(root)
        seed_1 = np.zeros_like(root)
        seed_1[4, 6] = 1
        seed_2 = np.zeros_like(root)
        seed_2[4, 20] = 1
        hints = {
            "plant_01": {
                "lane_left": 0,
                "lane_right": 12,
                "seed_mask": seed_1,
                "fixed_top_y": 4,
                "lock_start_frame": 0,
            },
            "plant_02": {
                "lane_left": 0,
                "lane_right": 12,
                "seed_mask": seed_2,
                "fixed_top_y": 4,
                "lock_start_frame": 0,
            },
        }

        owned_root_frames, _owned_lateral_frames, meta = _build_owned_masks_by_frame(
            [root],
            [lateral],
            None,
            ["plant_01", "plant_02"],
            hints,
            AnalyticsConfig(
                min_component_area=1,
                prune_branch_px=1,
                bbox_padding=0,
                root_temporal_seed_enabled=False,
                track_lane_partition_enabled=True,
                track_seed_connectivity_enabled=True,
            ),
        )

        owned = owned_root_frames[0]
        plant_01 = owned["plant_01"]
        plant_02 = owned["plant_02"]
        self.assertGreater(int(np.count_nonzero(plant_01)), 0)
        self.assertEqual(int(np.count_nonzero(plant_02)), 0)
        self.assertEqual(int(np.count_nonzero((plant_01 > 0) & (plant_02 > 0))), 0)
        self.assertEqual(int(meta["root_fallback_suppressed_duplicate_tracks"]), 1)

    def test_assign_tip_links_rejects_cross_plant_hops_by_default(self) -> None:
        config = AnalyticsConfig(tip_track_max_link_distance_px=20.0)
        active_states = [
            {
                "tip_id": "tip_001",
                "gap_frames": 1,
                "last_detection": {
                    "x": 10,
                    "y": 10,
                    "angle_deg": 90.0,
                    "plant_id": "plant_01",
                    "bbox": (0, 0, 20, 20),
                },
            }
        ]
        detections = [
            {
                "x": 12,
                "y": 12,
                "angle_deg": 92.0,
                "plant_id": "plant_02",
                "bbox": (10, 0, 20, 20),
            }
        ]

        assignments, method = _assign_tip_links(active_states, detections, config)

        self.assertEqual(assignments, [])
        self.assertEqual(method, "none")

    def test_tip_priors_ignore_cross_plant_observations_and_unstable_tracks(self) -> None:
        config = AnalyticsConfig(tip_track_min_dominant_vote_share=0.65)
        tip_tracks = {
            "tip_001": {
                "dominant_plant_id": "plant_01",
                "dominant_vote_share": 0.75,
                "observations": [
                    {"frame_index": 0, "plant_id": "plant_01", "x": 10, "y": 18},
                    {"frame_index": 1, "plant_id": "plant_02", "x": 26, "y": 31},
                ],
            },
            "tip_002": {
                "dominant_plant_id": "plant_02",
                "dominant_vote_share": 0.50,
                "observations": [
                    {"frame_index": 0, "plant_id": "plant_02", "x": 20, "y": 22},
                ],
            },
        }

        priors = _build_tip_priors_by_frame(tip_tracks, 2, ["plant_01", "plant_02"], config)

        self.assertEqual(priors, {0: {"plant_01": [(10, 18)]}})

    def test_previous_owner_prior_biases_ambiguous_pixel_assignment(self) -> None:
        h, w = 24, 24
        source = np.zeros((h, w), dtype=np.uint8)
        source[12, 5:18] = 1
        source[5:13, 5] = 1
        source[5:13, 17] = 1

        left_seed = np.zeros((h, w), dtype=np.uint8)
        left_seed[5, 5] = 1
        right_seed = np.zeros((h, w), dtype=np.uint8)
        right_seed[5, 17] = 1
        seeds = {"left": left_seed, "right": right_seed}
        hints = {
            "left": {"lane_left": 0, "lane_right": w, "seed_mask": left_seed},
            "right": {"lane_left": 0, "lane_right": w, "seed_mask": right_seed},
        }

        without_prior = _assign_pixels_to_track_seeds(source, seeds, hints)

        left_prior = np.zeros((h, w), dtype=np.uint8)
        left_prior[12, 14] = 1
        with_prior = _assign_pixels_to_track_seeds(
            source,
            seeds,
            hints,
            track_prior_masks={"left": left_prior},
            prior_weight_px=6.0,
        )

        self.assertEqual(int(without_prior["right"][12, 14]), 1)
        self.assertEqual(int(with_prior["left"][12, 14]), 1)
        self.assertEqual(int(with_prior["right"][12, 14]), 0)

    def test_previous_owner_prior_keeps_disconnected_gap_segment(self) -> None:
        h, w = 24, 24
        source = np.zeros((h, w), dtype=np.uint8)
        source[2:8, 8] = 1
        source[14:21, 8] = 1
        source[14:21, 18] = 1

        seed = np.zeros((h, w), dtype=np.uint8)
        seed[2:4, 7:10] = 1
        hints = {
            "plant_01": {
                "lane_left": 0,
                "lane_right": 14,
                "seed_mask": seed,
            }
        }
        prior = np.zeros((h, w), dtype=np.uint8)
        prior[15:20, 8] = 1

        without_prior = _assign_pixels_to_track_seeds(
            source,
            {"plant_01": seed},
            hints,
        )
        with_prior = _assign_pixels_to_track_seeds(
            source,
            {"plant_01": seed},
            hints,
            track_prior_masks={"plant_01": prior},
            prior_weight_px=4.0,
        )

        self.assertEqual(int(without_prior["plant_01"][16, 8]), 0)
        self.assertEqual(int(with_prior["plant_01"][16, 8]), 1)
        self.assertEqual(int(with_prior["plant_01"][16, 18]), 0)

    def test_temporal_tip_seed_preserves_identity_through_crossing_component(self) -> None:
        h, w = 12, 12
        frame0 = np.zeros((h, w), dtype=np.uint8)
        frame0[:, 2] = 1
        frame0[:, 9] = 1
        frame1 = frame0.copy()
        frame1[5, 2:10] = 1

        root_masks = [frame0, frame1]
        lateral_masks = [np.zeros_like(frame0), np.zeros_like(frame1)]
        seed_left = np.zeros((h, w), dtype=np.uint8)
        seed_left[0:2, 1:4] = 1
        seed_right = np.zeros((h, w), dtype=np.uint8)
        seed_right[0:2, 8:11] = 1
        hints = {
            "left": {"lane_left": 0, "lane_right": w, "seed_mask": seed_left, "fixed_top_y": 0},
            "right": {"lane_left": 0, "lane_right": w, "seed_mask": seed_right, "fixed_top_y": 0},
        }

        disabled_config = AnalyticsConfig(
            min_component_area=1,
            prune_branch_px=1,
            root_temporal_seed_enabled=False,
            root_temporal_seed_radius_px=2,
        )
        disabled_owned_root, _disabled_owned_lateral, _disabled_meta = _build_owned_masks_by_frame(
            root_masks,
            lateral_masks,
            None,
            ["left", "right"],
            hints,
            disabled_config,
            tip_priors_by_frame=None,
        )

        enabled_config = AnalyticsConfig(
            min_component_area=1,
            prune_branch_px=1,
            root_temporal_seed_enabled=True,
            root_temporal_seed_radius_px=2,
        )
        enabled_owned_root, _enabled_owned_lateral, enabled_meta = _build_owned_masks_by_frame(
            root_masks,
            lateral_masks,
            None,
            ["left", "right"],
            hints,
            enabled_config,
            tip_priors_by_frame=None,
        )

        disabled_left_bottom = np.where(disabled_owned_root[1]["left"][10] > 0)[0].tolist()
        disabled_right_bottom = np.where(disabled_owned_root[1]["right"][10] > 0)[0].tolist()
        enabled_left_bottom = np.where(enabled_owned_root[1]["left"][10] > 0)[0].tolist()
        enabled_right_bottom = np.where(enabled_owned_root[1]["right"][10] > 0)[0].tolist()

        self.assertEqual(disabled_left_bottom, [9])
        self.assertEqual(disabled_right_bottom, [2])
        self.assertEqual(enabled_left_bottom, [2])
        self.assertEqual(enabled_right_bottom, [9])
        self.assertEqual(enabled_meta["tracks_with_temporal_seeds"], 2)
        self.assertEqual(enabled_meta["root_temporal_seed_frames"], 1)
        self.assertEqual(enabled_meta["tracks_with_previous_mask_seeds"], 2)
        self.assertEqual(enabled_meta["root_previous_mask_seed_frames"], 1)
        self.assertEqual(enabled_meta["tracks_with_previous_assignment_priors"], 2)
        self.assertEqual(enabled_meta["root_previous_assignment_prior_frames"], 1)

    def test_overlapping_temporal_seed_dilation_preserves_exact_previous_owner(self) -> None:
        h, w = 16, 16
        frame0 = np.zeros((h, w), dtype=np.uint8)
        frame0[1:15, 5] = 1
        frame0[1:15, 8] = 1

        frame1 = frame0.copy()
        frame1[7, 5:9] = 1

        seed_left = np.zeros((h, w), dtype=np.uint8)
        seed_left[0:2, 4:7] = 1
        seed_right = np.zeros((h, w), dtype=np.uint8)
        seed_right[0:2, 7:10] = 1
        hints = {
            "left": {"lane_left": 0, "lane_right": w, "seed_mask": seed_left, "fixed_top_y": 0},
            "right": {"lane_left": 0, "lane_right": w, "seed_mask": seed_right, "fixed_top_y": 0},
        }

        owned_root, _owned_lateral, meta = _build_owned_masks_by_frame(
            [frame0, frame1],
            [np.zeros_like(frame0), np.zeros_like(frame1)],
            None,
            ["left", "right"],
            hints,
            AnalyticsConfig(
                min_component_area=1,
                prune_branch_px=1,
                root_temporal_seed_enabled=True,
                root_temporal_seed_radius_px=4,
            ),
            tip_priors_by_frame=None,
        )

        self.assertEqual(int(owned_root[1]["left"][14, 5]), 1)
        self.assertEqual(int(owned_root[1]["right"][14, 8]), 1)
        overlap = (owned_root[1]["left"] > 0) & (owned_root[1]["right"] > 0)
        self.assertEqual(int(np.count_nonzero(overlap)), 0)
        self.assertEqual(meta["tracks_with_previous_exact_seed_reserve"], 2)
        self.assertEqual(meta["root_previous_exact_seed_reserve_frames"], 1)

    def test_previous_owned_mask_prior_can_follow_root_across_static_lane(self) -> None:
        h, w = 14, 14
        frame0 = np.zeros((h, w), dtype=np.uint8)
        frame0[1:13, 2] = 1
        frame0[1:13, 11] = 1

        frame1 = np.zeros((h, w), dtype=np.uint8)
        frame1[1:7, 2] = 1
        frame1[6, 2:9] = 1
        frame1[6:13, 8] = 1
        frame1[1:13, 11] = 1

        seed_left = np.zeros((h, w), dtype=np.uint8)
        seed_left[0:2, 1:4] = 1
        seed_right = np.zeros((h, w), dtype=np.uint8)
        seed_right[0:2, 10:13] = 1
        hints = {
            "left": {"lane_left": 0, "lane_right": 7, "seed_mask": seed_left, "fixed_top_y": 0},
            "right": {"lane_left": 7, "lane_right": w, "seed_mask": seed_right, "fixed_top_y": 0},
        }

        owned_root, _owned_lateral, meta = _build_owned_masks_by_frame(
            [frame0, frame1],
            [np.zeros_like(frame0), np.zeros_like(frame1)],
            None,
            ["left", "right"],
            hints,
            AnalyticsConfig(
                min_component_area=1,
                prune_branch_px=1,
                root_temporal_seed_enabled=True,
                root_temporal_seed_radius_px=8,
            ),
            tip_priors_by_frame=None,
        )

        self.assertEqual(np.where(owned_root[1]["left"][12] > 0)[0].tolist(), [8])
        self.assertEqual(np.where(owned_root[1]["right"][12] > 0)[0].tolist(), [11])
        self.assertEqual(meta["tracks_with_previous_mask_seeds"], 2)
        self.assertEqual(meta["root_previous_mask_seed_frames"], 1)
        self.assertEqual(meta["tracks_with_previous_assignment_priors"], 2)
        self.assertEqual(meta["root_previous_assignment_prior_frames"], 1)

    def test_explicit_lateral_does_not_hijack_main_path_tip(self) -> None:
        h, w = 20, 20
        root = np.zeros((h, w), dtype=np.uint8)
        root[1:15, 5] = 1
        for offset in range(10):
            root[7 + offset, 5 + offset] = 1

        lateral = np.zeros((h, w), dtype=np.uint8)
        for offset in range(10):
            lateral[7 + offset, 5 + offset] = 1

        item = DatasetImageItem(
            uid="frame_000",
            name="frame_000.png",
            path=Path("/tmp/frame_000.png"),
            image=np.zeros((h, w), dtype=np.uint8),
        )
        hint = {
            "plant_01": {
                "lane_left": 0,
                "lane_right": w,
                "seed_mask": np.pad(np.ones((2, 2), dtype=np.uint8), ((0, h - 2), (4, w - 6))),
                "fixed_top_y": 0,
            }
        }

        legacy_payload = _assemble_payload_from_owned_masks(
            [item],
            [{"plant_01": root}],
            [{"plant_01": lateral}],
            None,
            [np.zeros((h, w), dtype=np.uint8)],
            ["plant_01"],
            {"plant_01": None},
            AnalyticsConfig(min_component_area=1, prune_branch_px=1, pixel_size_mm=1.0, primary_root_tracking_excludes_lateral=False),
            {},
            {},
            {},
            {},
            hint,
            {"enabled": True},
            "class_mask",
        )
        fixed_payload = _assemble_payload_from_owned_masks(
            [item],
            [{"plant_01": root}],
            [{"plant_01": lateral}],
            None,
            [np.zeros((h, w), dtype=np.uint8)],
            ["plant_01"],
            {"plant_01": None},
            AnalyticsConfig(min_component_area=1, prune_branch_px=1, pixel_size_mm=1.0, primary_root_tracking_excludes_lateral=True),
            {},
            {},
            {},
            {},
            hint,
            {"enabled": True},
            "class_mask",
        )

        legacy_tip = legacy_payload["frames"][0]["tracks"]["plant_01"]["path_points"][-1]
        fixed_tip = fixed_payload["frames"][0]["tracks"]["plant_01"]["path_points"][-1]

        self.assertEqual(legacy_tip, [14, 16])
        self.assertEqual(fixed_tip, [5, 14])

    def test_payload_exposes_class_resolved_root_and_shoot_metrics(self) -> None:
        h, w = 24, 24
        item = DatasetImageItem(
            uid="frame_000",
            name="frame_000.png",
            path=Path("/tmp/frame_000.png"),
            image=np.zeros((h, w), dtype=np.uint8),
        )

        root_1 = np.zeros((h, w), dtype=np.uint8)
        root_1[3:16, 5] = 1
        lateral_1 = np.zeros((h, w), dtype=np.uint8)
        lateral_1[7, 5:10] = 1
        lateral_1[7:13, 9] = 1
        shoot_1 = np.zeros((h, w), dtype=np.uint8)
        shoot_1[1:3, 3:6] = 1

        root_2 = np.zeros((h, w), dtype=np.uint8)
        root_2[4:18, 16] = 1
        lateral_2 = np.zeros((h, w), dtype=np.uint8)
        lateral_2[9, 12:17] = 1
        lateral_2[9:15, 12] = 1
        shoot_2 = np.zeros((h, w), dtype=np.uint8)
        shoot_2[1:4, 15:19] = 1

        payload = _assemble_payload_from_owned_masks(
            [item],
            [{"plant_01": np.maximum(root_1, lateral_1), "plant_02": np.maximum(root_2, lateral_2)}],
            [{"plant_01": lateral_1, "plant_02": lateral_2}],
            [{"plant_01": shoot_1, "plant_02": shoot_2}],
            [np.maximum(shoot_1, shoot_2)],
            ["plant_01", "plant_02"],
            {"plant_01": None, "plant_02": None},
            AnalyticsConfig(min_component_area=1, prune_branch_px=1, pixel_size_mm=0.5, bbox_padding=0),
            {},
            {},
            {},
            {},
            {},
            {"enabled": True},
            "class_mask",
        )

        rows = {str(row["plant_id"]): row for row in payload["rows"]}
        row_1 = rows["plant_01"]
        frame_row_1 = payload["frames"][0]["tracks"]["plant_01"]

        self.assertEqual(row_1, frame_row_1)
        self.assertEqual(int(row_1["shoot_area_px"]), 6)
        self.assertAlmostEqual(float(row_1["shoot_area_mm2"]), 1.5, places=4)
        self.assertGreater(float(row_1["primary_root_length_px"]), 0.0)
        self.assertGreater(float(row_1["lateral_total_length_px"]), 0.0)
        self.assertAlmostEqual(
            float(row_1["total_root_length_px"]),
            float(row_1["primary_root_length_px"]) + float(row_1["lateral_total_length_px"]),
            places=4,
        )
        self.assertAlmostEqual(
            float(row_1["total_root_length_mm_raw"]),
            float(row_1["total_root_length_px"]) * 0.5,
            places=4,
        )
        self.assertEqual(payload["tracks"]["plant_01"]["shoot_area_peak_px"], 6)
        self.assertEqual(int(row_1["seedling_bbox_x"]), 3)
        self.assertEqual(int(row_1["seedling_bbox_y"]), 1)
        self.assertEqual(int(row_1["seedling_bbox_w"]), 7)
        self.assertEqual(int(row_1["seedling_bbox_h"]), 15)
        self.assertAlmostEqual(float(row_1["seedling_center_x"]), 6.5)
        self.assertAlmostEqual(float(row_1["seedling_center_y"]), 8.5)
        self.assertAlmostEqual(
            float(payload["tracks"]["plant_01"]["total_root_length_mm_raw"][0]),
            float(row_1["total_root_length_mm_raw"]),
            places=4,
        )
        self.assertAlmostEqual(
            float(payload["summary"]["total_root_total_length_mm"]),
            sum(float(row["total_root_length_mm_raw"]) for row in rows.values()),
            places=4,
        )

    def test_payload_exposes_owned_mask_center_jump_diagnostics(self) -> None:
        h, w = 40, 60
        items = [
            DatasetImageItem(
                uid="frame_000",
                name="frame_000.png",
                path=Path("/tmp/frame_000.png"),
                image=np.zeros((h, w), dtype=np.uint8),
            ),
            DatasetImageItem(
                uid="frame_001",
                name="frame_001.png",
                path=Path("/tmp/frame_001.png"),
                image=np.zeros((h, w), dtype=np.uint8),
            ),
        ]
        root0 = np.zeros((h, w), dtype=np.uint8)
        root1 = np.zeros((h, w), dtype=np.uint8)
        root0[5:25, 8] = 1
        root1[5:25, 38] = 1
        shoot0 = np.zeros((h, w), dtype=np.uint8)
        shoot1 = np.zeros((h, w), dtype=np.uint8)
        shoot0[2:5, 7:10] = 1
        shoot1[2:5, 37:40] = 1

        payload = _assemble_payload_from_owned_masks(
            items,
            [{"plant_01": root0}, {"plant_01": root1}],
            [{"plant_01": np.zeros_like(root0)}, {"plant_01": np.zeros_like(root1)}],
            [{"plant_01": shoot0}, {"plant_01": shoot1}],
            [np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)],
            ["plant_01"],
            {"plant_01": None},
            AnalyticsConfig(min_component_area=1, prune_branch_px=1, pixel_size_mm=1.0, bbox_padding=0),
            {},
            {},
            {},
            {},
            {"plant_01": {"lane_left": 0, "lane_right": w, "seed_mask": np.ones((h, w), dtype=np.uint8), "fixed_top_y": 0}},
            {"enabled": True},
            "class_mask",
        )

        row = payload["rows"][1]
        track = payload["tracks"]["plant_01"]
        self.assertAlmostEqual(float(row["bbox_center_jump_px"]), 30.0)
        self.assertAlmostEqual(float(row["shoot_center_jump_px"]), 30.0)
        self.assertAlmostEqual(float(row["seedling_center_jump_px"]), 30.0)
        self.assertAlmostEqual(float(track["bbox_center_jump_px_max"]), 30.0)
        self.assertAlmostEqual(float(track["shoot_center_jump_px_max"]), 30.0)
        self.assertAlmostEqual(float(track["seedling_center_jump_px_max"]), 30.0)

    def test_shoot_only_bbox_overlap_does_not_invalidate_root_ownership(self) -> None:
        h, w = 24, 30
        item = DatasetImageItem(
            uid="frame_000",
            name="frame_000.png",
            path=Path("/tmp/frame_000.png"),
            image=np.zeros((h, w), dtype=np.uint8),
        )
        root_1 = np.zeros((h, w), dtype=np.uint8)
        root_2 = np.zeros((h, w), dtype=np.uint8)
        root_1[8:18, 3:7] = 1
        root_2[8:18, 17:21] = 1
        shoot_1 = np.zeros((h, w), dtype=np.uint8)
        shoot_2 = np.zeros((h, w), dtype=np.uint8)
        shoot_1[2:8, 8:15] = 1
        shoot_2[2:8, 14:21] = 1

        payload = _assemble_payload_from_owned_masks(
            [item],
            [{"plant_01": root_1, "plant_02": root_2}],
            [{"plant_01": np.zeros((h, w), dtype=np.uint8), "plant_02": np.zeros((h, w), dtype=np.uint8)}],
            [{"plant_01": shoot_1, "plant_02": shoot_2}],
            [np.maximum(shoot_1, shoot_2)],
            ["plant_01", "plant_02"],
            {"plant_01": None, "plant_02": None},
            AnalyticsConfig(min_component_area=1, prune_branch_px=1, pixel_size_mm=1.0, bbox_padding=0),
            {},
            {},
            {},
            {},
            {},
            {"enabled": True},
            "class_mask",
        )

        frame_rows = payload["frames"][0]["tracks"]
        self.assertEqual(int(frame_rows["plant_01"]["bbox_x"]), 3)
        self.assertEqual(int(frame_rows["plant_02"]["bbox_x"]), 17)
        self.assertLess(
            int(frame_rows["plant_01"]["bbox_x"]) + int(frame_rows["plant_01"]["bbox_w"]),
            int(frame_rows["plant_02"]["bbox_x"]),
        )
        self.assertEqual(frame_rows["plant_01"]["measurement_tier"], "individual")
        self.assertEqual(frame_rows["plant_02"]["measurement_tier"], "individual")
        self.assertTrue(frame_rows["plant_01"]["ownership_measurement_valid"])
        self.assertEqual(len(payload["frames"][0]["combined_groups"]), 0)
        self.assertEqual(payload["summary"]["measurement_tiers"]["combined_overlap_rows"], 0)

    def test_temporal_analytics_uses_separate_seed_anchor_and_shoot_class(self) -> None:
        h, w = 40, 40
        items: list[DatasetImageItem] = []
        predictions: dict[str, np.ndarray] = {}
        for frame_idx in range(2):
            uid = f"frame_{frame_idx:03d}"
            pred = np.zeros((h, w), dtype=np.uint8)
            pred[5:8, 18:22] = 2
            pred[1:5, 12:18] = 5
            pred[8:30, 20] = 1
            pred[18, 20:25] = 3
            items.append(
                DatasetImageItem(
                    uid=uid,
                    name=f"{uid}.png",
                    path=Path(f"/tmp/{uid}.png"),
                    image=np.zeros((h, w, 3), dtype=np.uint8),
                )
            )
            predictions[uid] = pred

        payload = run_temporal_analytics(
            items,
            predictions,
            {},
            AnalyticsConfig(
                root_class_id=1,
                lateral_class_id=3,
                seed_class_id=2,
                shoot_class_id=5,
                min_component_area=1,
                bbox_padding=0,
                tracking_search_margin=4,
                prune_branch_px=1,
                persistence_frames=1,
                temporal_smoothing_enabled=False,
                shoot_crown_lock_enabled=False,
                shoot_tracking_start_frame=0,
                expected_track_count=1,
                pixel_size_mm=1.0,
            ),
        )

        summary = payload["summary"]
        self.assertEqual(summary["anchor_source"]["anchor_class_ids"], [2])
        self.assertEqual(summary["anchor_source"]["selection_mode"], "explicit_seed_class")
        self.assertEqual(summary["shoot_source"]["shoot_class_ids"], [5])
        self.assertEqual(summary["shoot_source"]["selection_mode"], "explicit_shoot_class")
        self.assertTrue(payload["rows"])
        for row in payload["rows"]:
            self.assertEqual(row["seed_class_id"], 2)
            self.assertEqual(row["shoot_class_id"], 5)
            self.assertEqual(row["shoot_area_px"], 24)

    def test_touching_boxes_switch_rows_into_combined_overlap_tier(self) -> None:
        h, w = 24, 24
        items = [
            DatasetImageItem(
                uid=f"frame_{idx:03d}",
                name=f"frame_{idx:03d}.png",
                path=Path(f"/tmp/frame_{idx:03d}.png"),
                image=np.zeros((h, w), dtype=np.uint8),
            )
            for idx in range(4)
        ]

        def _mask(x0: int, x1: int) -> np.ndarray:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[2:18, x0:x1] = 1
            return mask

        owned_root_frames = [
            {"plant_01": _mask(3, 7), "plant_02": _mask(15, 19)},
            {"plant_01": _mask(3, 10), "plant_02": _mask(8, 15)},
            {"plant_01": _mask(3, 7), "plant_02": _mask(15, 19)},
            {"plant_01": _mask(3, 7), "plant_02": _mask(15, 19)},
        ]
        owned_lateral_frames = [
            {"plant_01": np.zeros((h, w), dtype=np.uint8), "plant_02": np.zeros((h, w), dtype=np.uint8)}
            for _ in range(4)
        ]
        payload = _assemble_payload_from_owned_masks(
            items,
            owned_root_frames,
            owned_lateral_frames,
            None,
            [np.zeros((h, w), dtype=np.uint8) for _ in range(4)],
            ["plant_01", "plant_02"],
            {"plant_01": None, "plant_02": None},
            AnalyticsConfig(min_component_area=1, prune_branch_px=1, pixel_size_mm=1.0, bbox_padding=0),
            {},
            {},
            {},
            {},
            {},
            {"enabled": True},
            "class_mask",
        )

        frame0_rows = payload["frames"][0]["tracks"]
        frame1_rows = payload["frames"][1]["tracks"]
        frame2_rows = payload["frames"][2]["tracks"]
        frame3_rows = payload["frames"][3]["tracks"]

        self.assertEqual(frame0_rows["plant_01"]["measurement_tier"], "individual")
        self.assertEqual(frame1_rows["plant_01"]["measurement_tier"], "combined_overlap")
        self.assertEqual(frame2_rows["plant_01"]["measurement_tier"], "identity_reacquiring")
        self.assertEqual(frame3_rows["plant_01"]["measurement_tier"], "individual")
        self.assertFalse(frame1_rows["plant_01"]["ownership_measurement_valid"])
        self.assertFalse(frame2_rows["plant_02"]["ownership_measurement_valid"])
        self.assertTrue(frame3_rows["plant_02"]["ownership_measurement_valid"])
        self.assertEqual(frame1_rows["plant_01"]["conflict_start_frame"], 1)
        self.assertGreater(float(frame1_rows["plant_01"]["combined_root_length_mm"] or 0.0), 0.0)
        self.assertEqual(len(payload["frames"][1]["combined_groups"]), 1)
        self.assertEqual(payload["tracks"]["plant_01"]["conflict_start_frame"], 1)
        self.assertEqual(payload["tracks"]["plant_02"]["combined_conflict_frames"], 1)
        self.assertEqual(payload["tracks"]["plant_02"]["identity_reacquisition_frames"], 1)
        self.assertEqual(payload["summary"]["measurement_tiers"]["combined_overlap_tracks"], 2)
        self.assertEqual(payload["summary"]["measurement_tiers"]["combined_overlap_groups"], 1)

    def test_lateral_contact_does_not_invalidate_separate_primary_roots(self) -> None:
        h, w = 28, 28
        item = DatasetImageItem(
            uid="frame_000",
            name="frame_000.png",
            path=Path("/tmp/frame_000.png"),
            image=np.zeros((h, w), dtype=np.uint8),
        )
        root_1 = np.zeros((h, w), dtype=np.uint8)
        root_2 = np.zeros((h, w), dtype=np.uint8)
        lateral_1 = np.zeros((h, w), dtype=np.uint8)
        lateral_2 = np.zeros((h, w), dtype=np.uint8)
        root_1[2:24, 5] = 1
        root_2[2:24, 20] = 1
        lateral_1[10, 5:17] = 1
        lateral_2[11, 10:21] = 1
        root_1 = np.maximum(root_1, lateral_1)
        root_2 = np.maximum(root_2, lateral_2)

        payload = _assemble_payload_from_owned_masks(
            [item],
            [{"plant_01": root_1, "plant_02": root_2}],
            [{"plant_01": lateral_1, "plant_02": lateral_2}],
            None,
            [np.zeros((h, w), dtype=np.uint8)],
            ["plant_01", "plant_02"],
            {"plant_01": None, "plant_02": None},
            AnalyticsConfig(
                min_component_area=1,
                prune_branch_px=1,
                pixel_size_mm=1.0,
                bbox_padding=0,
            ),
            {},
            {},
            {},
            {},
            {},
            {"enabled": True},
            "class_mask",
        )

        rows = payload["frames"][0]["tracks"]
        self.assertTrue(rows["plant_01"]["ownership_measurement_valid"])
        self.assertTrue(rows["plant_02"]["ownership_measurement_valid"])
        self.assertEqual(rows["plant_01"]["measurement_tier"], "individual")

    def test_small_graph_ambiguity_is_reviewable_without_blocking_measurement(self) -> None:
        h, w = 28, 28
        item = DatasetImageItem(
            uid="frame_000",
            name="frame_000.png",
            path=Path("/tmp/frame_000.png"),
            image=np.zeros((h, w), dtype=np.uint8),
        )
        root_1 = np.zeros((h, w), dtype=np.uint8)
        root_2 = np.zeros((h, w), dtype=np.uint8)
        root_1[2:24, 5] = 1
        root_1[10, 5:17] = 1
        root_2[2:24, 20] = 1
        root_2[11, 10:21] = 1
        ownership_meta = {
            "enabled": True,
            "root_ownership_assignment_mode": "temporal_graph",
            "root_ownership_freeze_ambiguous_state": True,
            "root_ownership_assignment_frames": [
                {
                    "frame_index": 0,
                    "ambiguous_pixels": 12,
                    "shared_components": 1,
                    "ambiguous_tracks": ["plant_01", "plant_02"],
                    "blocking_ambiguous_tracks": [],
                    "track_ambiguity": {
                        "plant_01": {
                            "ambiguous_pixels": 6,
                            "ambiguous_fraction": 0.01,
                            "ambiguous": True,
                            "blocking": False,
                        },
                        "plant_02": {
                            "ambiguous_pixels": 6,
                            "ambiguous_fraction": 0.01,
                            "ambiguous": True,
                            "blocking": False,
                        },
                    },
                }
            ],
        }

        payload = _assemble_payload_from_owned_masks(
            [item],
            [{"plant_01": root_1, "plant_02": root_2}],
            [
                {
                    "plant_01": np.zeros((h, w), dtype=np.uint8),
                    "plant_02": np.zeros((h, w), dtype=np.uint8),
                }
            ],
            None,
            [np.zeros((h, w), dtype=np.uint8)],
            ["plant_01", "plant_02"],
            {"plant_01": None, "plant_02": None},
            AnalyticsConfig(
                min_component_area=1,
                prune_branch_px=1,
                pixel_size_mm=1.0,
                bbox_padding=0,
            ),
            {},
            {},
            {},
            {},
            {},
            ownership_meta,
            "class_mask",
        )

        rows = payload["frames"][0]["tracks"]
        self.assertTrue(rows["plant_01"]["root_ownership_ambiguous"])
        self.assertFalse(rows["plant_01"]["root_ownership_blocking_ambiguous"])
        self.assertTrue(rows["plant_01"]["ownership_measurement_valid"])
        self.assertFalse(rows["plant_01"]["root_ownership_state_frozen"])


if __name__ == "__main__":
    unittest.main()
