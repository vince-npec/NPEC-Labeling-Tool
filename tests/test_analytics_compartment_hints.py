from __future__ import annotations

import unittest

import numpy as np

from resources.analytics_engine import (
    AnalyticsConfig,
    _build_owned_shoot_masks_by_frame,
    _build_track_compartment_hints,
    _extract_track_mask_from_hint,
    _track_arabidopsis_crown_lanes,
)


class AnalyticsCompartmentHintTests(unittest.TestCase):
    def _root_masks(self) -> list[np.ndarray]:
        root = np.zeros((80, 120), dtype=np.uint8)
        root[14:72, 29:32] = 1
        root[30:33, 32:84] = 1
        root[14:72, 89:92] = 1
        return [root.copy(), root.copy()]

    def _track_bboxes(self) -> dict[str, list[tuple[int, int, int, int]]]:
        return {
            "plant_01": [(28, 10, 56, 64), (28, 10, 56, 64)],
            "plant_02": [(88, 10, 4, 64), (88, 10, 4, 64)],
        }

    def test_compartment_lanes_use_complete_shoot_anchor_centers(self) -> None:
        anchors = [np.zeros((80, 120), dtype=np.uint8) for _ in range(2)]
        for anchor in anchors:
            anchor[4:10, 27:34] = 1
            anchor[4:10, 87:94] = 1

        hints = _build_track_compartment_hints(
            self._root_masks(),
            anchors,
            ["plant_01", "plant_02"],
            self._track_bboxes(),
            seed_frame=0,
            config=AnalyticsConfig(min_component_area=12, track_lane_padding_px=0),
        )

        self.assertEqual(str(hints["plant_01"]["lane_center_source"]), "anchor")
        self.assertEqual(str(hints["plant_02"]["lane_center_source"]), "anchor")
        self.assertAlmostEqual(float(hints["plant_01"]["lane_center_x"]), 30.0)
        self.assertAlmostEqual(float(hints["plant_02"]["lane_center_x"]), 90.0)
        self.assertEqual(int(hints["plant_01"]["lane_right"]), 60)
        self.assertEqual(int(hints["plant_02"]["lane_left"]), 60)
        self.assertAlmostEqual(float(hints["plant_01"]["root_seed_center_x"]), 56.0)

    def test_compartment_lanes_fall_back_to_root_boxes_when_anchors_are_incomplete(self) -> None:
        anchors = [np.zeros((80, 120), dtype=np.uint8) for _ in range(2)]
        for anchor in anchors:
            anchor[4:10, 27:34] = 1

        hints = _build_track_compartment_hints(
            self._root_masks(),
            anchors,
            ["plant_01", "plant_02"],
            self._track_bboxes(),
            seed_frame=0,
            config=AnalyticsConfig(min_component_area=12, track_lane_padding_px=0),
        )

        self.assertEqual(str(hints["plant_01"]["lane_center_source"]), "bbox")
        self.assertEqual(str(hints["plant_02"]["lane_center_source"]), "bbox")
        self.assertAlmostEqual(float(hints["plant_01"]["lane_center_x"]), 56.0)
        self.assertAlmostEqual(float(hints["plant_02"]["lane_center_x"]), 90.0)
        self.assertEqual(int(hints["plant_01"]["lane_right"]), 73)
        self.assertEqual(int(hints["plant_02"]["lane_left"]), 73)

    def test_compartment_hints_use_nearest_valid_bbox_when_track_absent_on_seed_frame(self) -> None:
        root0 = np.zeros((80, 120), dtype=np.uint8)
        root0[14:72, 29:32] = 1
        root1 = root0.copy()
        root1[14:72, 89:92] = 1
        anchors = [np.zeros((80, 120), dtype=np.uint8) for _ in range(2)]
        track_bboxes = {
            "plant_01": [(28, 10, 4, 64), (28, 10, 4, 64)],
            "plant_02": [(0, 0, 0, 0), (88, 10, 4, 64)],
        }

        hints = _build_track_compartment_hints(
            [root0, root1],
            anchors,
            ["plant_01", "plant_02"],
            track_bboxes,
            seed_frame=0,
            config=AnalyticsConfig(min_component_area=12, track_lane_padding_px=0),
        )

        self.assertEqual(set(hints.keys()), {"plant_01", "plant_02"})
        self.assertEqual(int(hints["plant_01"]["lane_reference_frame"]), 0)
        self.assertEqual(str(hints["plant_01"]["lane_reference_source"]), "seed_frame")
        self.assertEqual(int(hints["plant_02"]["lane_reference_frame"]), 1)
        self.assertEqual(str(hints["plant_02"]["lane_reference_source"]), "nearest_valid_bbox")
        self.assertEqual(int(hints["plant_01"]["lane_right"]), 60)
        self.assertEqual(int(hints["plant_02"]["lane_left"]), 60)
        self.assertGreater(int(np.count_nonzero(hints["plant_02"]["seed_mask"])), 0)

    def test_crown_lane_hints_move_root_and_shoot_seeds_with_each_frame(self) -> None:
        h, w = 140, 240
        centers = [32, 76, 120, 164, 208]
        translations = [(0, 0), (10, 4), (18, 7)]
        roots: list[np.ndarray] = []
        anchors: list[np.ndarray] = []
        for frame_idx, (dx, dy) in enumerate(translations):
            root = np.zeros((h, w), dtype=np.uint8)
            anchor = np.zeros((h, w), dtype=np.uint8)
            for plant_idx, center in enumerate(centers):
                x = center + dx
                if frame_idx == 2 and plant_idx == 2:
                    root[78 + dy : 132, x] = 1
                    continue
                root[27 + dy : 132, x] = 1
                anchor[9 + dy : 19 + dy, x - 6 : x + 7] = 1
            roots.append(root)
            anchors.append(anchor)

        config = AnalyticsConfig(
            expected_track_count=5,
            min_component_area=1,
            track_lane_padding_px=0,
            tracking_mode="arabidopsis_crown_lanes",
            shoot_tracking_enabled=False,
        )
        tracking = _track_arabidopsis_crown_lanes(roots, config, anchor_masks=anchors)
        hints = _build_track_compartment_hints(
            roots,
            anchors,
            tracking["track_ids"],
            tracking["track_bboxes"],
            seed_frame=int(tracking["seed_frame"]),
            config=config,
        )

        third_geometry = hints["plant_03"]["per_frame"]
        self.assertEqual(len(third_geometry), len(roots))
        self.assertTrue(bool(hints["plant_03"]["dynamic_lane_geometry"]))
        observed_motion = float(third_geometry[2]["lane_center_x"]) - float(third_geometry[0]["lane_center_x"])
        self.assertLess(abs(observed_motion - float(translations[2][0])), 3.0)
        for frame_idx in range(len(roots)):
            frame_centers = [float(hints[track_id]["per_frame"][frame_idx]["lane_center_x"]) for track_id in tracking["track_ids"]]
            self.assertEqual(frame_centers, sorted(frame_centers))

        shifted_root, shifted_bbox = _extract_track_mask_from_hint(
            roots[1],
            hints["plant_04"],
            1,
            (h, w),
            config,
        )
        shifted_x = centers[3] + translations[1][0]
        self.assertGreater(int(np.count_nonzero(shifted_root)), 0)
        self.assertEqual(int(shifted_root[50, shifted_x]), 1)
        self.assertLessEqual(int(shifted_bbox[0]), shifted_x)
        self.assertGreater(int(shifted_bbox[0] + shifted_bbox[2]), shifted_x)

        missing_root, _missing_bbox = _extract_track_mask_from_hint(
            roots[2],
            hints["plant_03"],
            2,
            (h, w),
            config,
        )
        missing_x = centers[2] + translations[2][0]
        self.assertEqual(int(missing_root[100, missing_x]), 1)

        owned_shoots, _meta = _build_owned_shoot_masks_by_frame(
            anchors,
            tracking["track_ids"],
            hints,
            config,
        )
        for plant_idx, track_id in enumerate(tracking["track_ids"]):
            shoot_x = centers[plant_idx] + translations[1][0]
            self.assertEqual(int(owned_shoots[1][track_id][15, shoot_x]), 1)
        self.assertEqual(int(np.count_nonzero(owned_shoots[2]["plant_03"])), 0)

    def test_crown_lane_shoot_assignment_rejects_far_corner_component(self) -> None:
        h, w = 160, 320
        root_seed = np.zeros((h, w), dtype=np.uint8)
        root_seed[88:132, 219:222] = 1
        true_shoot = np.zeros((h, w), dtype=np.uint8)
        true_shoot[66:91, 205:237] = 1
        corner_artifact = np.zeros((h, w), dtype=np.uint8)
        corner_artifact[4:22, 258:276] = 1
        shoots = [
            np.maximum(true_shoot, corner_artifact),
            corner_artifact.copy(),
        ]
        hints = {
            "plant_05": {
                "lane_left": 180,
                "lane_right": w,
                "lane_center_x": 220.0,
                "fixed_top_y": 10,
                "crown_center_x": 220.0,
                "crown_center_y": 90.0,
                "seed_top_limit": 140,
                "seed_mask": root_seed,
                "shoot_seed_mask": true_shoot,
                "dynamic_lane_geometry": True,
            }
        }
        config = AnalyticsConfig(
            tracking_mode="arabidopsis_crown_lanes",
            shoot_tracking_enabled=True,
            shoot_tracking_start_frame=0,
            shoot_tracking_motion_radius_px=24,
            shoot_tracking_crown_gate_enabled=True,
            shoot_tracking_crown_component_max_distance_px=48.0,
        )

        owned, meta = _build_owned_shoot_masks_by_frame(
            shoots,
            ["plant_05"],
            hints,
            config,
        )

        self.assertEqual(
            int(np.count_nonzero(owned[0]["plant_05"])),
            int(np.count_nonzero(true_shoot)),
        )
        self.assertEqual(int(np.count_nonzero(owned[1]["plant_05"])), 0)
        self.assertEqual(
            int(np.count_nonzero(owned[0]["plant_05"] & corner_artifact)),
            0,
        )
        self.assertEqual(int(meta["crown_component_gate_rejected_components"]), 2)
        self.assertEqual(
            int(meta["crown_component_gate_rejected_pixels"]),
            2 * int(np.count_nonzero(corner_artifact)),
        )
        self.assertEqual(meta["crown_component_gate_tracks"], ["plant_05"])

    def test_hades_bw_empty_shoot_is_recovered_only_at_supported_crown(self) -> None:
        h, w = 220, 360
        crown_x, crown_y = 250, 112
        root = np.zeros((h, w), dtype=np.uint8)
        root[crown_y : 205, crown_x - 1 : crown_x + 2] = 1
        corner_artifact = np.zeros((h, w), dtype=np.uint8)
        corner_artifact[8:24, 330:346] = 1

        gray = np.full((h, w), 220, dtype=np.uint8)
        gray[75:108, 218:282] = 34
        gray[62:90, 228:247] = 34
        gray[62:90, 258:277] = 34
        gray[crown_y:205, crown_x - 2 : crown_x + 3] = 42
        gray[corner_artifact > 0] = 20
        image = np.repeat(gray[..., None], 3, axis=2)

        hints = {
            "plant_05": {
                "lane_left": 180,
                "lane_right": w,
                "lane_center_x": float(crown_x),
                "fixed_top_y": crown_y,
                "crown_center_x": float(crown_x),
                "crown_center_y": float(crown_y),
                "seed_top_limit": 190,
                "seed_mask": root,
                "shoot_seed_mask": root,
                "dynamic_lane_geometry": True,
            }
        }
        config = AnalyticsConfig(
            tracking_mode="arabidopsis_crown_lanes",
            shoot_tracking_enabled=True,
            shoot_tracking_start_frame=0,
            shoot_tracking_crown_component_max_distance_px=56.0,
            shoot_tracking_grayscale_crown_rescue_enabled=True,
            shoot_tracking_grayscale_crown_rescue_roi_half_width_px=100,
            shoot_tracking_grayscale_crown_rescue_roi_above_px=90,
            shoot_tracking_grayscale_crown_rescue_roi_below_px=80,
            shoot_tracking_grayscale_crown_rescue_root_radius_px=30,
            shoot_tracking_grayscale_crown_rescue_min_root_pixels=8,
            shoot_tracking_grayscale_crown_rescue_max_distance_px=72.0,
        )

        owned, meta = _build_owned_shoot_masks_by_frame(
            [corner_artifact],
            ["plant_05"],
            hints,
            config,
            root_masks=[root],
            frame_images=[image],
        )

        recovered = owned[0]["plant_05"]
        self.assertGreater(int(np.count_nonzero(recovered[62:108, 218:282])), 1000)
        self.assertEqual(int(np.count_nonzero(recovered & corner_artifact)), 0)
        self.assertEqual(meta["grayscale_crown_rescue_tracks"], ["plant_05"])
        self.assertEqual(
            meta["grayscale_crown_rescue_frames_by_track"],
            {"plant_05": [0]},
        )

    def test_hades_bw_crown_rescue_requires_root_support_and_grayscale(self) -> None:
        h, w = 160, 260
        crown_x, crown_y = 180, 90
        empty_root = np.zeros((h, w), dtype=np.uint8)
        image = np.full((h, w, 3), 220, dtype=np.uint8)
        image[45:88, 150:212] = 30
        hint = {
            "plant_05": {
                "lane_left": 120,
                "lane_right": w,
                "lane_center_x": float(crown_x),
                "fixed_top_y": crown_y,
                "crown_center_x": float(crown_x),
                "crown_center_y": float(crown_y),
                "seed_mask": empty_root,
                "shoot_seed_mask": empty_root,
                "dynamic_lane_geometry": True,
            }
        }
        config = AnalyticsConfig(
            tracking_mode="arabidopsis_crown_lanes",
            shoot_tracking_start_frame=0,
            shoot_tracking_grayscale_crown_rescue_enabled=True,
            shoot_tracking_grayscale_crown_rescue_min_root_pixels=5,
        )
        owned, meta = _build_owned_shoot_masks_by_frame(
            [np.zeros((h, w), dtype=np.uint8)],
            ["plant_05"],
            hint,
            config,
            root_masks=[empty_root],
            frame_images=[image],
        )
        self.assertEqual(int(np.count_nonzero(owned[0]["plant_05"])), 0)
        self.assertEqual(int(meta["grayscale_crown_rescue_frames"]), 0)

        supported_root = empty_root.copy()
        supported_root[crown_y : 145, crown_x] = 1
        color_image = image.copy()
        color_image[45:88, 150:212] = (20, 180, 20)
        owned_color, color_meta = _build_owned_shoot_masks_by_frame(
            [np.zeros((h, w), dtype=np.uint8)],
            ["plant_05"],
            hint,
            config,
            root_masks=[supported_root],
            frame_images=[color_image],
        )
        self.assertEqual(int(np.count_nonzero(owned_color[0]["plant_05"])), 0)
        self.assertEqual(int(color_meta["grayscale_crown_rescue_frames"]), 0)


if __name__ == "__main__":
    unittest.main()
