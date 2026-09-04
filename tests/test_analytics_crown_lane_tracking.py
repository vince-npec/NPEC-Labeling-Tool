from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from resources.analytics_engine import AnalyticsConfig, _track_arabidopsis_crown_lanes
from resources.models import DatasetImageItem


class ArabidopsisCrownLaneTrackingTests(unittest.TestCase):
    def test_shoot_base_rejects_root_colored_plate_edge_above_crown(self) -> None:
        h, w = 140, 240
        centers = [42, 120, 198]
        root = np.zeros((h, w), dtype=np.uint8)
        anchor = np.zeros((h, w), dtype=np.uint8)
        for center in centers:
            root[42:132, center] = 1
            anchor[24:40, center - 7 : center + 8] = 1
        root[2:30, 3:6] = 1

        result = _track_arabidopsis_crown_lanes(
            [root],
            AnalyticsConfig(
                expected_track_count=3,
                min_component_area=1,
                tracking_mode="arabidopsis_crown_lanes",
            ),
            anchor_masks=[anchor],
        )

        initialization = result["identity_initialization"]
        centers_x = initialization["per_frame_centers_x"][0]
        centers_y = initialization["per_frame_centers_y"][0]
        sources = initialization["per_frame_center_sources"][0]
        self.assertEqual(sources, ["anchor_guided_root"] * 3)
        for expected, actual_x, actual_y in zip(centers, centers_x, centers_y):
            self.assertLess(abs(float(expected) - float(actual_x)), 3.0)
            self.assertGreater(float(actual_y), 35.0)

    def test_fragmented_leaves_still_initialize_five_ordered_crowns(self) -> None:
        h, w = 120, 200
        centers = [28, 64, 100, 136, 172]
        masks: list[np.ndarray] = []
        anchors: list[np.ndarray] = []
        for frame_idx in range(3):
            root = np.zeros((h, w), dtype=np.uint8)
            anchor = np.zeros((h, w), dtype=np.uint8)
            for center in centers:
                root[28:112, center + frame_idx % 2] = 1
                anchor[14:21, center - 5 : center] = 1
                anchor[10:18, center + 2 : center + 7] = 1
            for offset in range(-10, 11, 4):
                anchor[5 + (offset % 3) : 8 + (offset % 3), 28 + offset : 30 + offset] = 1
            masks.append(root)
            anchors.append(anchor)

        result = _track_arabidopsis_crown_lanes(
            masks,
            AnalyticsConfig(expected_track_count=5, min_component_area=1),
            anchor_masks=anchors,
        )

        self.assertEqual(result["track_ids"], [f"plant_{idx:02d}" for idx in range(1, 6)])
        initialized = result["identity_initialization"]
        initialized_centers = [float(value) for value in initialized["centers_x"]]
        self.assertEqual(initialized_centers, sorted(initialized_centers))
        self.assertEqual(len(initialized_centers), 5)
        for expected, actual in zip(centers, initialized_centers):
            self.assertLess(abs(float(expected) - float(actual)), 14.0)
        for track_id in result["track_ids"]:
            self.assertEqual(len(result["track_bboxes"][track_id]), len(masks))
            bbox_centers = [float(x + (0.5 * bw)) for x, _y, bw, _bh in result["track_bboxes"][track_id]]
            self.assertLessEqual(max(bbox_centers) - min(bbox_centers), 1.0)

    def test_plate_translation_and_missing_crown_keep_permanent_ordered_identities(self) -> None:
        h, w = 140, 240
        centers = [32, 76, 120, 164, 208]
        translations = [(0, 0), (9, 3), (17, 6), (12, 4)]
        missing_frame = 2
        missing_plant = 2
        masks: list[np.ndarray] = []
        anchors: list[np.ndarray] = []
        for frame_idx, (dx, dy) in enumerate(translations):
            root = np.zeros((h, w), dtype=np.uint8)
            anchor = np.zeros((h, w), dtype=np.uint8)
            for plant_idx, center in enumerate(centers):
                x = center + dx
                if frame_idx == missing_frame and plant_idx == missing_plant:
                    root[78 + dy : 132, x] = 1
                    continue
                root[28 + dy : 132, x] = 1
                anchor[10 + dy : 19 + dy, x - 7 : x - 1] = 1
                anchor[7 + dy : 17 + dy, x + 2 : x + 8] = 1
            masks.append(root)
            anchors.append(anchor)

        result = _track_arabidopsis_crown_lanes(
            masks,
            AnalyticsConfig(
                expected_track_count=5,
                min_component_area=1,
                tracking_mode="arabidopsis_crown_lanes",
            ),
            anchor_masks=anchors,
        )

        self.assertEqual(result["track_ids"], [f"plant_{idx:02d}" for idx in range(1, 6)])
        initialization = result["identity_initialization"]
        per_frame_centers = initialization["per_frame_centers_x"]
        per_frame_sources = initialization["per_frame_center_sources"]
        self.assertEqual(per_frame_sources[missing_frame][missing_plant], "motion_fallback")
        for frame_centers in per_frame_centers:
            self.assertEqual(frame_centers, sorted(frame_centers))

        for frame_idx, (dx, _dy) in enumerate(translations):
            expected_motion = float(dx - translations[0][0])
            observed_motion = float(
                np.median(
                    np.asarray(per_frame_centers[frame_idx], dtype=np.float64)
                    - np.asarray(per_frame_centers[0], dtype=np.float64)
                )
            )
            self.assertLess(abs(observed_motion - expected_motion), 3.0)

        missing_track = result["track_bboxes"]["plant_03"]
        missing_centers = [float(x + (0.5 * bw)) for x, _y, bw, _bh in missing_track]
        self.assertLess(
            abs((missing_centers[missing_frame] - missing_centers[0]) - float(translations[missing_frame][0])),
            3.0,
        )

    def test_single_plant_configuration_keeps_existing_tracker(self) -> None:
        root = np.zeros((80, 80), dtype=np.uint8)
        root[8:70, 40] = 1
        result = _track_arabidopsis_crown_lanes(
            [root],
            AnalyticsConfig(expected_track_count=1, min_component_area=1),
            anchor_masks=[np.zeros_like(root)],
        )

        self.assertNotEqual(str(result.get("source", "")), "arabidopsis_crown_lanes")

    def test_image_registration_separates_camera_motion_from_rosette_growth(self) -> None:
        h, w = 260, 360
        centers = [48, 114, 180, 246, 312]
        translations = [(0, 0), (14, -6), (20, -4)]
        base = np.full((h, w, 3), 224, dtype=np.uint8)
        cv2.rectangle(base, (8, 8), (w - 9, h - 9), (75, 78, 82), 3)
        rng = np.random.default_rng(20260722)
        for x, y in zip(rng.integers(18, w - 18, 90), rng.integers(18, h - 18, 90)):
            cv2.circle(base, (int(x), int(y)), 1, (130, 134, 128), -1)

        masks: list[np.ndarray] = []
        anchors: list[np.ndarray] = []
        timeline: list[DatasetImageItem] = []
        for frame_idx, (dx, dy) in enumerate(translations):
            transform = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
            image = cv2.warpAffine(base, transform, (w, h), borderMode=cv2.BORDER_REPLICATE)
            root = np.zeros((h, w), dtype=np.uint8)
            anchor = np.zeros((h, w), dtype=np.uint8)
            for center in centers:
                x = center + dx
                root[62 + dy : 238 + dy, x] = 1
                anchor[42 + dy : 61 + dy, x - 7 : x + 8] = 1
                if frame_idx > 0:
                    anchor[28 + dy : 58 + dy, x + 4 : min(w, x + 30 + (8 * frame_idx))] = 1
                cv2.line(image, (x, 62 + dy), (x, 42 + dy), (48, 122, 38), 3)
                cv2.circle(image, (x + 8 + (7 * frame_idx), 38 + dy), 10 + frame_idx, (45, 136, 42), -1)
            masks.append(root)
            anchors.append(anchor)
            timeline.append(
                DatasetImageItem(
                    uid=f"frame-{frame_idx}",
                    name=f"frame-{frame_idx}.png",
                    path=Path(f"/tmp/frame-{frame_idx}.png"),
                    image=image,
                )
            )

        result = _track_arabidopsis_crown_lanes(
            masks,
            AnalyticsConfig(
                expected_track_count=5,
                min_component_area=1,
                tracking_mode="arabidopsis_crown_lanes",
            ),
            anchor_masks=anchors,
            timeline=timeline,
        )

        initialization = result["identity_initialization"]
        per_frame_centers = np.asarray(initialization["per_frame_centers_x"], dtype=np.float64)
        self.assertEqual(initialization["per_frame_motion_source"], ["image_registration"] * len(translations))
        for frame_idx, (dx, _dy) in enumerate(translations):
            observed = float(np.median(per_frame_centers[frame_idx] - per_frame_centers[0]))
            self.assertLess(abs(observed - float(dx)), 3.0)

    def test_missing_first_frame_crown_is_mapped_back_to_reference_coordinates(self) -> None:
        h, w = 260, 360
        centers = [48, 114, 180, 246, 312]
        translations = [(0, 0), (18, 4), (26, 7)]
        base = np.full((h, w, 3), 228, dtype=np.uint8)
        cv2.rectangle(base, (8, 8), (w - 9, h - 9), (68, 72, 76), 4)
        rng = np.random.default_rng(73)
        for x, y in zip(rng.integers(14, w - 14, 110), rng.integers(14, h - 14, 110)):
            cv2.circle(base, (int(x), int(y)), 1, (115, 121, 118), -1)

        masks: list[np.ndarray] = []
        anchors: list[np.ndarray] = []
        timeline: list[DatasetImageItem] = []
        for frame_idx, (dx, dy) in enumerate(translations):
            transform = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
            image = cv2.warpAffine(base, transform, (w, h), borderMode=cv2.BORDER_REPLICATE)
            root = np.zeros((h, w), dtype=np.uint8)
            anchor = np.zeros((h, w), dtype=np.uint8)
            for plant_idx, center in enumerate(centers):
                x = center + dx
                if frame_idx == 0 and plant_idx == 2:
                    continue
                root[62 + dy : 238 + dy, x] = 1
                anchor[42 + dy : 61 + dy, x - 7 : x + 8] = 1
            masks.append(root)
            anchors.append(anchor)
            timeline.append(
                DatasetImageItem(
                    uid=f"missing-{frame_idx}",
                    name=f"missing-{frame_idx}.png",
                    path=Path(f"/tmp/missing-{frame_idx}.png"),
                    image=image,
                )
            )

        result = _track_arabidopsis_crown_lanes(
            masks,
            AnalyticsConfig(
                expected_track_count=5,
                min_component_area=1,
                tracking_mode="arabidopsis_crown_lanes",
            ),
            anchor_masks=anchors,
            timeline=timeline,
        )

        per_frame = np.asarray(
            result["identity_initialization"]["per_frame_centers_x"],
            dtype=np.float64,
        )
        plant_three_motion = per_frame[:, 2] - per_frame[0, 2]
        for frame_idx, (dx, _dy) in enumerate(translations):
            self.assertLess(abs(float(plant_three_motion[frame_idx]) - float(dx)), 3.0)

    def test_stationary_dish_does_not_follow_moving_green_shoots(self) -> None:
        h, w = 260, 360
        centers = [48, 114, 180, 246, 312]
        base = np.full((h, w, 3), 226, dtype=np.uint8)
        cv2.rectangle(base, (7, 7), (w - 8, h - 8), (70, 74, 79), 4)
        rng = np.random.default_rng(91)
        for x, y in zip(rng.integers(14, w - 14, 120), rng.integers(14, h - 14, 120)):
            cv2.circle(base, (int(x), int(y)), 1, (120, 124, 119), -1)

        masks: list[np.ndarray] = []
        anchors: list[np.ndarray] = []
        timeline: list[DatasetImageItem] = []
        for frame_idx, shoot_dx in enumerate((0, 18, 30)):
            image = base.copy()
            root = np.zeros((h, w), dtype=np.uint8)
            anchor = np.zeros((h, w), dtype=np.uint8)
            for center in centers:
                root[62:238, center] = 1
                anchor[42:61, center - 7 : center + 8] = 1
                cv2.line(image, (center, 62), (center + shoot_dx, 38), (45, 128, 38), 4)
                cv2.circle(image, (center + shoot_dx, 34), 11 + frame_idx, (44, 138, 40), -1)
            masks.append(root)
            anchors.append(anchor)
            timeline.append(
                DatasetImageItem(
                    uid=f"growth-{frame_idx}",
                    name=f"growth-{frame_idx}.png",
                    path=Path(f"/tmp/growth-{frame_idx}.png"),
                    image=image,
                )
            )

        result = _track_arabidopsis_crown_lanes(
            masks,
            AnalyticsConfig(
                expected_track_count=5,
                min_component_area=1,
                tracking_mode="arabidopsis_crown_lanes",
            ),
            anchor_masks=anchors,
            timeline=timeline,
        )

        per_frame = np.asarray(
            result["identity_initialization"]["per_frame_centers_x"],
            dtype=np.float64,
        )
        for frame_idx in range(1, len(per_frame)):
            observed = float(np.median(per_frame[frame_idx] - per_frame[0]))
            self.assertLess(abs(observed), 2.0)


if __name__ == "__main__":
    unittest.main()
