from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from resources.analytics_engine import (
    AnalyticsConfig,
    _assign_pixels_to_track_seeds,
    _build_owned_masks_by_frame,
    _derive_lateral_mask_from_root,
    _measure_crop,
    run_temporal_analytics,
)
from resources.models import DatasetImageItem


class TemporalGraphOwnershipTests(unittest.TestCase):
    @staticmethod
    def _crossing_fixture() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict[str, object]]]:
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[2:22, 6] = 1
        mask[2:22, 17] = 1
        mask[11, 6:18] = 1
        left = np.zeros_like(mask)
        right = np.zeros_like(mask)
        left[2:4, 5:8] = 1
        right[2:4, 16:19] = 1
        hints = {
            "left": {"lane_left": 0, "lane_right": 24, "seed_mask": left},
            "right": {"lane_left": 0, "lane_right": 24, "seed_mask": right},
        }
        return mask, {"left": left, "right": right}, hints

    def test_symmetric_crossing_is_independent_of_track_iteration_order(self) -> None:
        mask, seeds, hints = self._crossing_fixture()

        first, first_meta = _assign_pixels_to_track_seeds(
            mask,
            seeds,
            hints,
            assignment_mode="temporal_graph",
            return_metadata=True,
        )
        reversed_seeds = {"right": seeds["right"], "left": seeds["left"]}
        second, second_meta = _assign_pixels_to_track_seeds(
            mask,
            reversed_seeds,
            hints,
            assignment_mode="temporal_graph",
            return_metadata=True,
        )

        np.testing.assert_array_equal(first["left"], second["left"])
        np.testing.assert_array_equal(first["right"], second["right"])
        self.assertGreater(int(first_meta["ambiguous_pixels"]), 0)
        self.assertEqual(int(first_meta["ambiguous_pixels"]), int(second_meta["ambiguous_pixels"]))

    def test_exact_previous_owners_preserve_both_distal_branches(self) -> None:
        mask, seeds, hints = self._crossing_fixture()
        left_prior = np.zeros_like(mask)
        right_prior = np.zeros_like(mask)
        left_prior[2:22, 6] = 1
        right_prior[2:22, 17] = 1

        owned, meta = _assign_pixels_to_track_seeds(
            mask,
            seeds,
            hints,
            track_prior_masks={"left": left_prior, "right": right_prior},
            prior_weight_px=6.0,
            assignment_mode="temporal_graph",
            return_metadata=True,
        )

        self.assertEqual(int(owned["left"][20, 6]), 1)
        self.assertEqual(int(owned["right"][20, 17]), 1)
        self.assertEqual(int(np.count_nonzero((owned["left"] > 0) & (owned["right"] > 0))), 0)
        self.assertEqual(str(meta["mode"]), "temporal_graph")

    def test_ambiguous_crossing_does_not_advance_identity_memory(self) -> None:
        crossing, seeds, hints = self._crossing_fixture()
        separate = crossing.copy()
        separate[11, 7:17] = 0

        owned, _lateral, meta = _build_owned_masks_by_frame(
            [separate, crossing],
            [np.zeros_like(separate), np.zeros_like(crossing)],
            None,
            ["left", "right"],
            hints,
            AnalyticsConfig(
                min_component_area=1,
                prune_branch_px=1,
                root_temporal_seed_enabled=True,
                root_ownership_assignment_mode="temporal_graph",
                root_ownership_freeze_ambiguous_state=True,
                root_ownership_blocking_ambiguity_min_pixels=1,
                root_ownership_blocking_ambiguity_min_fraction=0.0,
            ),
        )

        self.assertEqual(int(owned[1]["left"][20, 6]), 1)
        self.assertEqual(int(owned[1]["right"][20, 17]), 1)
        self.assertGreaterEqual(int(meta["root_ownership_ambiguity_frames"]), 1)
        self.assertGreaterEqual(int(meta["root_ownership_frozen_state_updates"]), 2)

    def test_tiny_graph_ambiguity_does_not_freeze_default_identity_memory(self) -> None:
        crossing, _seeds, hints = self._crossing_fixture()
        separate = crossing.copy()
        separate[11, 7:17] = 0

        _owned, _lateral, meta = _build_owned_masks_by_frame(
            [separate, crossing],
            [np.zeros_like(separate), np.zeros_like(crossing)],
            None,
            ["left", "right"],
            hints,
            AnalyticsConfig(
                min_component_area=1,
                prune_branch_px=1,
                root_temporal_seed_enabled=True,
                root_ownership_assignment_mode="temporal_graph",
                root_ownership_freeze_ambiguous_state=True,
            ),
        )

        self.assertGreaterEqual(int(meta["root_ownership_ambiguity_frames"]), 1)
        self.assertEqual(int(meta["root_ownership_blocking_ambiguity_frames"]), 0)
        self.assertEqual(int(meta["root_ownership_frozen_state_updates"]), 0)

    def test_per_seedling_decomposition_keeps_long_side_branch_lateral(self) -> None:
        root = np.zeros((64, 128), dtype=np.uint8)
        root[4:34, 20:23] = 1
        root[16:19, 23:112] = 1
        config = AnalyticsConfig(min_component_area=1, prune_branch_px=1)

        metrics = _measure_crop(root, config)
        lateral = _derive_lateral_mask_from_root(root, metrics)

        self.assertGreater(int(lateral[17, 100]), 0)
        path = metrics.get("path_xy")
        self.assertIsInstance(path, list)
        self.assertGreater(int(path[-1][1]), 28)

    def test_lateral_pixels_inherit_the_root_union_owner(self) -> None:
        root = np.zeros((36, 56), dtype=np.uint8)
        root[3:32, 12] = 1
        root[3:32, 43] = 1
        root[17, 5:13] = 1
        root[22, 43:51] = 1
        lateral = np.zeros_like(root)
        lateral[17, 5:13] = 1
        lateral[22, 43:51] = 1
        left_seed = np.zeros_like(root)
        right_seed = np.zeros_like(root)
        left_seed[3:6, 11:14] = 1
        right_seed[3:6, 42:45] = 1
        hints = {
            "left": {"lane_left": 0, "lane_right": 28, "seed_mask": left_seed},
            "right": {"lane_left": 28, "lane_right": 56, "seed_mask": right_seed},
        }

        owned_root, owned_lateral, _meta = _build_owned_masks_by_frame(
            [root],
            [lateral],
            None,
            ["left", "right"],
            hints,
            AnalyticsConfig(
                min_component_area=1,
                prune_branch_px=1,
                root_temporal_seed_enabled=False,
                root_ownership_assignment_mode="temporal_graph",
            ),
        )

        root_union = np.maximum(owned_root[0]["left"], owned_root[0]["right"])
        lateral_union = np.maximum(owned_lateral[0]["left"], owned_lateral[0]["right"])
        np.testing.assert_array_equal(root_union, root)
        np.testing.assert_array_equal(lateral_union, lateral)
        for track_id in ("left", "right"):
            self.assertEqual(
                int(np.count_nonzero((owned_lateral[0][track_id] > 0) & (owned_root[0][track_id] <= 0))),
                0,
            )

    def test_crossing_lateral_cannot_create_a_neighboring_current_seed(self) -> None:
        shape = (48, 80)
        primary = np.zeros(shape, dtype=np.uint8)
        primary[4:44, 20] = 1
        lateral = np.zeros(shape, dtype=np.uint8)
        lateral[12, 20:62] = 1
        lateral[12:44, 61] = 1
        root_union = np.maximum(primary, lateral)
        left_seed = np.zeros(shape, dtype=np.uint8)
        right_seed = np.zeros(shape, dtype=np.uint8)
        left_seed[4:9, 19:22] = 1
        right_seed[4:9, 60:63] = 1
        hints = {
            "plant_01": {
                "lane_left": 0,
                "lane_right": 40,
                "seed_mask": left_seed,
                "dynamic_lane_geometry": True,
                "per_frame": [{"lane_left": 0, "lane_right": 40, "seed_top_limit": 20, "dynamic_lane_geometry": True}],
            },
            "plant_02": {
                "lane_left": 40,
                "lane_right": 80,
                "seed_mask": right_seed,
                "dynamic_lane_geometry": True,
                "per_frame": [{"lane_left": 40, "lane_right": 80, "seed_top_limit": 20, "dynamic_lane_geometry": True}],
            },
        }

        owned_root, owned_lateral, _meta = _build_owned_masks_by_frame(
            [root_union],
            [lateral],
            None,
            ["plant_01", "plant_02"],
            hints,
            AnalyticsConfig(
                tracking_mode="arabidopsis_crown_lanes",
                expected_track_count=2,
                min_component_area=1,
                prune_branch_px=1,
                root_temporal_seed_enabled=False,
            ),
            root_seed_masks=[primary],
        )

        self.assertEqual(int(np.count_nonzero(owned_root[0]["plant_02"])), 0)
        np.testing.assert_array_equal(owned_root[0]["plant_01"], root_union)
        np.testing.assert_array_equal(owned_lateral[0]["plant_01"], lateral)

    def test_crown_lane_mode_recovers_detached_root_fragments_without_duplication(self) -> None:
        root = np.zeros((48, 80), dtype=np.uint8)
        root[3:18, 18] = 1
        root[24:44, 19] = 1
        root[3:20, 61] = 1
        root[27:45, 60] = 1
        left_seed = np.zeros_like(root)
        right_seed = np.zeros_like(root)
        left_seed[3:7, 17:20] = 1
        right_seed[3:7, 60:63] = 1
        hints = {
            "plant_01": {
                "lane_left": 0,
                "lane_right": 40,
                "lane_center_x": 18.0,
                "seed_mask": left_seed,
            },
            "plant_02": {
                "lane_left": 40,
                "lane_right": 80,
                "lane_center_x": 61.0,
                "seed_mask": right_seed,
            },
        }

        owned_root, _owned_lateral, meta = _build_owned_masks_by_frame(
            [root],
            [np.zeros_like(root)],
            None,
            ["plant_01", "plant_02"],
            hints,
            AnalyticsConfig(
                tracking_mode="arabidopsis_crown_lanes",
                expected_track_count=2,
                min_component_area=1,
                prune_branch_px=1,
                root_temporal_seed_enabled=False,
                root_ownership_assignment_mode="temporal_graph",
            ),
        )

        recovered_union = np.maximum(owned_root[0]["plant_01"], owned_root[0]["plant_02"])
        np.testing.assert_array_equal(recovered_union, root)
        self.assertEqual(
            int(np.count_nonzero((owned_root[0]["plant_01"] > 0) & (owned_root[0]["plant_02"] > 0))),
            0,
        )
        self.assertEqual(int(owned_root[0]["plant_01"][40, 19]), 1)
        self.assertEqual(int(owned_root[0]["plant_02"][40, 60]), 1)
        self.assertEqual(int(meta["root_detached_recovery_components"]), 2)
        self.assertGreater(int(meta["root_detached_recovery_pixels"]), 0)

    def test_class_measurements_are_disjoint_and_total_uses_owned_union(self) -> None:
        prediction = np.zeros((64, 64), dtype=np.uint8)
        prediction[4:56, 32] = 1
        prediction[20, 14:33] = 3
        prediction[36, 32:52] = 3
        item = DatasetImageItem(
            uid="frame_000",
            name="frame_000.png",
            path=Path("frame_000.png"),
            image=np.zeros((64, 64, 3), dtype=np.uint8),
        )
        config = AnalyticsConfig(
            root_class_id=1,
            lateral_class_id=3,
            expected_track_count=1,
            min_component_area=1,
            prune_branch_px=1,
            temporal_smoothing_enabled=False,
            shoot_crown_lock_enabled=False,
            shoot_tracking_enabled=False,
            tip_tracking_enabled=False,
            external_tracking_seed={
                "track_ids": ["plant_01"],
                "track_bboxes": {"plant_01": [(0, 0, 64, 64)]},
                "seed_frame": 0,
            },
        )

        payload = run_temporal_analytics(
            [item],
            {item.uid: prediction},
            annotations={},
            config=config,
        )
        row = payload["rows"][0]
        union = (prediction > 0).astype(np.uint8)
        expected_union_length = float(_measure_crop(union, config)["skeleton_length_px"])

        self.assertEqual(int(row["primary_root_area_px"]), int(np.count_nonzero(prediction == 1)))
        self.assertEqual(int(row["total_root_area_px"]), int(np.count_nonzero(union)))
        self.assertEqual(
            int(row["primary_root_area_px"]) + sum(int(segment["area_px"]) for segment in row["lateral_segments"]),
            int(row["total_root_area_px"]),
        )
        self.assertAlmostEqual(float(row["total_root_length_px"]), expected_union_length)
        self.assertEqual(str(row["total_root_measurement_mode"]), "owned_union_skeleton")

    def test_temporal_smoothing_never_carries_old_pixels_into_measurements(self) -> None:
        first = np.zeros((64, 64), dtype=np.uint8)
        second = np.zeros((64, 64), dtype=np.uint8)
        first[5:55, 18] = 1
        second[5:55, 44] = 1
        items = [
            DatasetImageItem(
                uid=f"frame_{idx:03d}",
                name=f"frame_{idx:03d}.png",
                path=Path(f"frame_{idx:03d}.png"),
                image=np.zeros((64, 64, 3), dtype=np.uint8),
            )
            for idx in range(2)
        ]
        config = AnalyticsConfig(
            root_class_id=1,
            expected_track_count=1,
            min_component_area=1,
            prune_branch_px=1,
            temporal_smoothing_enabled=True,
            temporal_alpha=0.60,
            enforce_monotonic_growth=False,
            shoot_crown_lock_enabled=False,
            shoot_tracking_enabled=False,
            tip_tracking_enabled=False,
            external_tracking_seed={
                "track_ids": ["plant_01"],
                "track_bboxes": {"plant_01": [(0, 0, 64, 64), (0, 0, 64, 64)]},
                "seed_frame": 0,
            },
        )

        payload = run_temporal_analytics(
            items,
            {items[0].uid: first, items[1].uid: second},
            annotations={},
            config=config,
        )

        self.assertEqual(len(payload["rows"]), 2)
        self.assertEqual(int(payload["rows"][0]["total_root_area_px"]), int(np.count_nonzero(first)))
        self.assertEqual(int(payload["rows"][1]["total_root_area_px"]), int(np.count_nonzero(second)))
        self.assertEqual(str(payload["summary"]["mask_source"]["measurement_scope"]), "current_frame_prediction")
        self.assertEqual(str(payload["summary"]["shoot_crown_lock"]["scope"]), "tracking_only")

    def test_temporal_root_priors_follow_dynamic_lane_motion(self) -> None:
        shape = (72, 96)
        first = np.zeros(shape, dtype=np.uint8)
        second = np.zeros(shape, dtype=np.uint8)
        first[4:68, 20] = 1
        second[4:68, 50] = 1
        base_seed = np.zeros(shape, dtype=np.uint8)
        base_seed[4:14, 20] = 1
        hints = {
            "plant_01": {
                "lane_left": 0,
                "lane_right": 48,
                "lane_center_x": 20.0,
                "fixed_top_y": 4,
                "seed_top_limit": 16,
                "lock_start_frame": 0,
                "seed_mask": base_seed,
                "shoot_seed_mask": base_seed,
                "dynamic_lane_geometry": True,
                "per_frame": [
                    {
                        "frame_index": 0,
                        "lane_left": 0,
                        "lane_right": 48,
                        "lane_center_x": 20.0,
                        "fixed_top_y": 4,
                        "seed_top_limit": 16,
                        "motion_x": 0.0,
                        "motion_y": 0.0,
                        "dynamic_lane_geometry": True,
                    },
                    {
                        "frame_index": 1,
                        "lane_left": 24,
                        "lane_right": 80,
                        "lane_center_x": 50.0,
                        "fixed_top_y": 4,
                        "seed_top_limit": 16,
                        "motion_x": 30.0,
                        "motion_y": 0.0,
                        "dynamic_lane_geometry": True,
                    },
                ],
            }
        }
        config = AnalyticsConfig(
            root_class_id=1,
            tracking_mode="arabidopsis_crown_lanes",
            expected_track_count=1,
            min_component_area=1,
            prune_branch_px=1,
            temporal_smoothing_enabled=False,
            shoot_tracking_enabled=False,
            tip_tracking_enabled=False,
            root_temporal_seed_enabled=True,
            root_temporal_seed_radius_px=8,
        )

        owned, _laterals, metadata = _build_owned_masks_by_frame(
            [first, second],
            [np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8)],
            [{}, {}],
            ["plant_01"],
            hints,
            config,
        )

        self.assertEqual(int(np.count_nonzero(owned[0]["plant_01"])), int(np.count_nonzero(first)))
        self.assertEqual(int(np.count_nonzero(owned[1]["plant_01"])), int(np.count_nonzero(second)))
        self.assertEqual(int(metadata["root_motion_compensated_prior_frames"]), 1)
        self.assertEqual(int(metadata["root_motion_compensated_prior_tracks"]), 1)
        self.assertEqual(int(metadata["root_previous_mask_seed_frames"]), 1)
        self.assertEqual(int(metadata["root_temporal_seed_frames"]), 1)

    def test_crown_lane_mode_tracks_tips_without_repeating_ownership_solve(self) -> None:
        prediction = np.zeros((64, 96), dtype=np.uint8)
        prediction[5:58, 18] = 1
        prediction[5:58, 76] = 1
        image = np.full((64, 96, 3), 224, dtype=np.uint8)
        image[3:61, 3:6] = 70
        image[3:6, 3:93] = 70
        items = [
            DatasetImageItem(
                uid="frame_000",
                name="frame_000.png",
                path=Path("frame_000.png"),
                image=image,
            )
        ]
        config = AnalyticsConfig(
            root_class_id=1,
            tracking_mode="arabidopsis_crown_lanes",
            expected_track_count=2,
            min_component_area=1,
            prune_branch_px=1,
            temporal_smoothing_enabled=False,
            shoot_crown_lock_enabled=False,
            shoot_tracking_enabled=False,
            tip_tracking_enabled=True,
        )

        payload = run_temporal_analytics(items, {items[0].uid: prediction}, {}, config)

        self.assertTrue(bool(payload["summary"]["tip_tracking_enabled"]))
        self.assertEqual(
            str(payload["summary"]["tip_ownership_refinement"]["status"]),
            "skipped_for_stable_crown_lanes",
        )
        self.assertFalse(bool(payload["summary"]["tip_ownership_refinement"]["applied"]))
        for row in payload["rows"]:
            self.assertTrue(np.isfinite(float(row["ownership_crown_center_x"])))
            self.assertTrue(np.isfinite(float(row["ownership_crown_center_y"])))
            self.assertEqual(str(row["ownership_frame_motion_source"]), "image_registration")
            self.assertEqual(str(row["ownership_frame_alignment_method"]), "reference_frame")
            self.assertEqual(str(row["ownership_frame_alignment_status"]), "reference")
            self.assertFalse(bool(row["ownership_frame_alignment_accepted"]))
            self.assertTrue(bool(row["ownership_frame_alignment_usable"]))
            self.assertEqual(float(row["ownership_frame_alignment_shift_x"]), 0.0)
            self.assertEqual(float(row["ownership_frame_alignment_shift_y"]), 0.0)
            self.assertEqual(int(row["ownership_frame_alignment_reference_frame_index"]), 0)
            self.assertEqual(
                str(row["ownership_frame_alignment_warp_matrix"]),
                "[[1.0,0.0,0.0],[0.0,1.0,0.0]]",
            )

    def test_crown_lane_layout_uses_detected_crowns_inside_dark_margins(self) -> None:
        shape = (100, 400)
        crown_centers = [105, 150, 205, 260, 310]
        prediction = np.zeros(shape, dtype=np.uint8)
        for center_x in crown_centers:
            prediction[14:88, center_x] = 1
            prediction[8:16, center_x - 5 : center_x + 6] = 2
        item = DatasetImageItem(
            uid="dark_margin_frame",
            name="dark_margin_frame.png",
            path=Path("dark_margin_frame.png"),
            image=np.full((shape[0], shape[1], 3), 180, dtype=np.uint8),
        )
        config = AnalyticsConfig(
            root_class_id=1,
            shoot_class_id=2,
            shoot_rgb_green_only_enabled=False,
            tracking_mode="arabidopsis_crown_lanes",
            expected_track_count=5,
            min_component_area=1,
            prune_branch_px=1,
            temporal_smoothing_enabled=False,
            shoot_crown_lock_enabled=True,
            shoot_tracking_enabled=True,
            tip_tracking_enabled=False,
        )

        payload = run_temporal_analytics(
            [item],
            {item.uid: prediction},
            {},
            config,
        )

        identity = payload["summary"]["identity_initialization"]
        self.assertEqual(str(identity["layout_source"]), "anchor_components")
        self.assertEqual(len(identity["centers_x"]), 5)
        np.testing.assert_allclose(identity["centers_x"], crown_centers, atol=2.0)
        self.assertEqual(len(payload["rows"]), 5)

    def test_crown_lane_layout_preserves_missing_lane_and_rejects_edge_artifact(self) -> None:
        shape = (100, 400)
        expected_centers = [105, 155, 205, 260, 310]
        observed_centers = [105, 205, 260, 310]
        prediction = np.zeros(shape, dtype=np.uint8)
        for center_x in observed_centers:
            prediction[14:88, center_x] = 1
            prediction[8:16, center_x - 5 : center_x + 6] = 2
        prediction[2:6, 346:350] = 2
        item = DatasetImageItem(
            uid="missing_lane_frame",
            name="missing_lane_frame.png",
            path=Path("missing_lane_frame.png"),
            image=np.full((shape[0], shape[1], 3), 180, dtype=np.uint8),
        )
        config = AnalyticsConfig(
            root_class_id=1,
            shoot_class_id=2,
            shoot_rgb_green_only_enabled=False,
            tracking_mode="arabidopsis_crown_lanes",
            expected_track_count=5,
            min_component_area=1,
            prune_branch_px=1,
            temporal_smoothing_enabled=False,
            shoot_crown_lock_enabled=True,
            shoot_tracking_enabled=True,
            tip_tracking_enabled=False,
        )

        payload = run_temporal_analytics([item], {item.uid: prediction}, {}, config)

        identity = payload["summary"]["identity_initialization"]
        self.assertEqual(
            str(identity["layout_source"]),
            "anchor_components_inferred_missing_lane",
        )
        np.testing.assert_allclose(identity["centers_x"], expected_centers, atol=6.0)
        self.assertLess(float(identity["centers_x"][-1]), 330.0)


if __name__ == "__main__":
    unittest.main()
