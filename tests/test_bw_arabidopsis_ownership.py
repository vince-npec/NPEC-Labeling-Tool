from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from resources.bw_arabidopsis_ownership import build_bw_arabidopsis_measurements
from resources.models import DatasetImageItem
from resources.pyphenotyper_adapter import PyPhenotyperConfig


class BwArabidopsisOwnershipTests(unittest.TestCase):
    def test_seed_center_measurements_return_stable_five_track_layout(self) -> None:
        shape = (300, 500)
        root_class_id = 3
        shoot_class_id = 2
        image = np.full(shape, 220, dtype=np.uint8)
        prediction = np.zeros(shape, dtype=np.uint8)

        centers_x = [70, 160, 250, 340, 430]
        for center_x in centers_x:
            prediction[35:55, center_x - 12 : center_x + 12] = np.uint8(shoot_class_id)
            prediction[55:220, center_x - 2 : center_x + 2] = np.uint8(root_class_id)
            prediction[95:110, center_x - 35 : center_x - 2] = np.uint8(root_class_id)
            prediction[145:160, center_x + 2 : center_x + 32] = np.uint8(root_class_id)

        item = DatasetImageItem(
            uid="frame_001",
            name="frame_001.png",
            path=Path("/tmp/frame_001.png"),
            image=image,
        )
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            root_class_id=root_class_id,
            shoot_class_id=shoot_class_id,
            pixel_size_mm=0.05,
            pipeline_overrides={
                "ownership_backend": "bw_arabidopsis_seed_centers",
                "expected_plant_count": 5,
            },
        )
        payload = build_bw_arabidopsis_measurements(
            [item],
            {"frame_001": prediction},
            config,
            metadata={"frame_001": {"pixel_size_mm": 0.05}},
        )
        summary = payload["summary"]
        per_uid = payload["per_uid"]["frame_001"]
        self.assertEqual(summary["ownership_backend"], "spatial_lane_fallback")
        self.assertFalse(bool(summary["ownership_claim"]))
        self.assertEqual(len(summary["plant_ids"]), 5)
        self.assertEqual(int(per_uid["plant_count"]), 5)
        self.assertEqual(len(per_uid["bboxes_xywh"]), 5)
        self.assertTrue(all(float(v) > 0.0 for v in per_uid["lengths_px"].values()))

    def test_implausible_shoot_mask_is_ignored_for_root_ownership(self) -> None:
        shape = (300, 500)
        root_class_id = 3
        shoot_class_id = 2
        image = np.full(shape, 220, dtype=np.uint8)
        prediction = np.zeros(shape, dtype=np.uint8)
        prediction[:260, :] = np.uint8(shoot_class_id)

        centers_x = [70, 160, 250, 340, 430]
        for center_x in centers_x:
            prediction[55:220, center_x - 2 : center_x + 2] = np.uint8(root_class_id)

        item = DatasetImageItem(
            uid="frame_bad_shoot",
            name="frame_bad_shoot.png",
            path=Path("/tmp/frame_bad_shoot.png"),
            image=image,
        )
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            root_class_id=root_class_id,
            shoot_class_id=shoot_class_id,
            pixel_size_mm=0.05,
            pipeline_overrides={
                "ownership_backend": "bw_arabidopsis_seed_centers",
                "expected_plant_count": 5,
            },
        )

        payload = build_bw_arabidopsis_measurements(
            [item],
            {"frame_bad_shoot": prediction},
            config,
            metadata={"frame_bad_shoot": {"pixel_size_mm": 0.05}},
        )
        per_uid = payload["per_uid"]["frame_bad_shoot"]

        self.assertFalse(bool(per_uid["shoot_mask_plausible"]))
        self.assertTrue(all(float(v) > 0.0 for v in per_uid["lengths_px"].values()))

    def test_lane_fallback_exports_primary_lateral_shoot_and_overlap_risk(self) -> None:
        shape = (240, 500)
        prediction = np.zeros(shape, dtype=np.uint8)
        centers_x = [70, 160, 250, 340, 430]
        for center_x in centers_x:
            prediction[20:42, center_x - 8 : center_x + 8] = 2
            prediction[42:210, center_x - 1 : center_x + 2] = 1
            prediction[90:92, center_x : min(shape[1], center_x + 35)] = 3
        prediction[120:122, 110:205] = 3

        item = DatasetImageItem(
            uid="frame_classes",
            name="frame_classes.png",
            path=Path("/tmp/frame_classes.png"),
            image=np.full(shape, 220, dtype=np.uint8),
        )
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            root_class_id=1,
            shoot_class_id=2,
            pixel_size_mm=0.05,
            pipeline_overrides={
                "ownership_backend": "bw_arabidopsis_seed_centers",
                "expected_plant_count": 5,
                "lateral_class_id": 3,
            },
        )

        payload = build_bw_arabidopsis_measurements(
            [item],
            {"frame_classes": prediction},
            config,
        )
        rows = payload["summary"]["measurements"]
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["measurement_scope"] == "crown_aligned_spatial_lane" for row in rows))
        self.assertTrue(all(row["ownership_claim"] is False for row in rows))
        self.assertTrue(all(str(row["lane_id"]).startswith("lane_") for row in rows))
        self.assertTrue(all(float(row["primary_root_length_mm"]) > 0.0 for row in rows))
        self.assertTrue(all(float(row["lateral_root_length_mm"]) > 0.0 for row in rows))
        self.assertTrue(all(int(row["shoot_area_px"]) > 0 for row in rows))
        self.assertTrue(any(bool(row["neighbor_overlap_risk"]) for row in rows))
        self.assertTrue(all(bool(row["bbox_measurement_available"]) for row in rows))
        self.assertTrue(
            all(
                not bool(row["spatial_lane_measurement_valid"])
                for row in rows
                if bool(row["neighbor_overlap_risk"])
            )
        )
        self.assertTrue(all(not bool(row["ownership_measurement_valid"]) for row in rows))
        boxes = [(int(row["bbox_x"]), int(row["bbox_w"])) for row in rows]
        self.assertTrue(all(x + w <= next_x for (x, w), (next_x, _next_w) in zip(boxes[:-1], boxes[1:])))

    def test_lane_overlap_risk_persists_after_bacterial_gap(self) -> None:
        shape = (240, 500)
        crossing = np.zeros(shape, dtype=np.uint8)
        gap = np.zeros(shape, dtype=np.uint8)
        for prediction in (crossing, gap):
            for center_x in (70, 160, 250, 340, 430):
                prediction[20:42, center_x - 8 : center_x + 8] = 2
                prediction[42:210, center_x - 1 : center_x + 2] = 1
        crossing[120:122, 110:205] = 3
        items = [
            DatasetImageItem(
                uid=uid,
                name=f"{uid}.png",
                path=Path(f"/tmp/{uid}.png"),
                image=np.full(shape, 220, dtype=np.uint8),
            )
            for uid in ("crossing", "gap")
        ]
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            root_class_id=1,
            shoot_class_id=2,
            pixel_size_mm=0.05,
            pipeline_overrides={
                "ownership_backend": "bw_arabidopsis_seed_centers",
                "expected_plant_count": 5,
                "lateral_class_id": 3,
            },
        )
        payload = build_bw_arabidopsis_measurements(
            items,
            {"crossing": crossing, "gap": gap},
            config,
        )
        rows = payload["summary"]["measurements"]
        risky_lanes = {
            row["lane_id"]
            for row in rows
            if row["frame_index"] == 0 and row["frame_boundary_crossing_detected"]
        }
        self.assertTrue(risky_lanes)
        second_frame = [row for row in rows if row["frame_index"] == 1 and row["lane_id"] in risky_lanes]
        self.assertTrue(second_frame)
        self.assertTrue(all(bool(row["neighbor_overlap_risk"]) for row in second_frame))
        self.assertTrue(all(not bool(row["frame_boundary_crossing_detected"]) for row in second_frame))
        self.assertTrue(all(not bool(row["spatial_lane_measurement_valid"]) for row in second_frame))

    def test_lane_centers_follow_uneven_shoot_crowns(self) -> None:
        shape = (260, 520)
        crown_positions = (55, 135, 255, 375, 465)
        prediction = np.zeros(shape, dtype=np.uint8)
        for center_x in crown_positions:
            prediction[25:55, center_x - 10 : center_x + 10] = 2
            prediction[55:225, center_x - 1 : center_x + 2] = 1
        item = DatasetImageItem(
            uid="uneven_crowns",
            name="uneven_crowns.png",
            path=Path("/tmp/uneven_crowns.png"),
            image=np.full(shape, 220, dtype=np.uint8),
        )
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            root_class_id=1,
            shoot_class_id=2,
            pixel_size_mm=0.05,
            pipeline_overrides={"expected_plant_count": 5},
        )
        payload = build_bw_arabidopsis_measurements(
            [item],
            {"uneven_crowns": prediction},
            config,
        )
        rows = payload["summary"]["measurements"]
        self.assertEqual([row["lane_center_source"] for row in rows], ["shoot_crown_pixels"] * 5)
        detected = [int(row["lane_center_x"]) for row in rows]
        for actual, expected in zip(detected, crown_positions, strict=True):
            self.assertLessEqual(abs(actual - expected), 3)

    def test_empty_lane_is_unavailable_not_valid_zero(self) -> None:
        shape = (240, 500)
        prediction = np.zeros(shape, dtype=np.uint8)
        for center_x in (70, 160, 250, 340, 430):
            prediction[20:42, center_x - 8 : center_x + 8] = 2
        item = DatasetImageItem(
            uid="shoot_only",
            name="shoot_only.png",
            path=Path("/tmp/shoot_only.png"),
            image=np.full(shape, 220, dtype=np.uint8),
        )
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            root_class_id=1,
            shoot_class_id=2,
            pixel_size_mm=0.05,
            pipeline_overrides={"expected_plant_count": 5},
        )
        rows = build_bw_arabidopsis_measurements(
            [item],
            {"shoot_only": prediction},
            config,
        )["summary"]["measurements"]
        self.assertTrue(all(not bool(row["bbox_measurement_available"]) for row in rows))
        self.assertTrue(all(not bool(row["spatial_lane_measurement_valid"]) for row in rows))
        self.assertTrue(all(row["spatial_lane_qc_tier"] == "unavailable" for row in rows))


if __name__ == "__main__":
    unittest.main()
