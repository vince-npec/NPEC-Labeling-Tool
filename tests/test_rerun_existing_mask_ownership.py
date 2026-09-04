from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from resources.rerun_existing_mask_ownership import (
    _apply_baseline_validity_safeguard,
    _build_timelapse,
    _fill_shoot_area_from_temporal_memory,
    _plate_fingerprint,
    _process_plate,
    _recover_grayscale_shoot_from_temporal_lane,
    _write_accepted_shoot_display_masks,
)
from resources.analytics_engine import AnalyticsConfig
from resources.models import DatasetImageItem


class ExistingMaskOwnershipRerunTests(unittest.TestCase):
    def test_accepted_shoot_display_mask_removes_unowned_class_two_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.full((90, 140, 3), 220, dtype=np.uint8)
            mask = np.zeros((90, 140), dtype=np.uint8)
            mask[30:84, 34] = 1
            mask[18:31, 25:45] = 2
            mask[4:10, 126:134] = 2
            item = DatasetImageItem(
                uid="frame-0",
                name="frame.png",
                path=root / "frame.png",
                image=image,
            )
            rows = [
                {
                    "frame_index": 0,
                    "plant_id": "plant_01",
                    "shoot_bbox_x": 25,
                    "shoot_bbox_y": 18,
                    "shoot_bbox_w": 20,
                    "shoot_bbox_h": 13,
                    "shoot_grayscale_crown_rescue_applied": False,
                    "ownership_crown_center_x": 34.0,
                    "ownership_crown_center_y": 30.0,
                }
            ]
            paths = _write_accepted_shoot_display_masks(
                rows,
                [{"SourceFile": str(item.path)}],
                [item],
                [mask],
                AnalyticsConfig(
                    root_class_id=1,
                    lateral_class_id=3,
                    shoot_class_id=2,
                    shoot_rgb_green_only_enabled=False,
                ),
                root / "display",
            )

            display = np.asarray(Image.open(paths[0]), dtype=np.uint8)
            self.assertEqual(int(np.count_nonzero(display[18:31, 25:45] == 2)), 260)
            self.assertEqual(int(np.count_nonzero(display[4:10, 126:134] == 2)), 0)
            self.assertEqual(
                int(np.count_nonzero(display == 1)),
                int(np.count_nonzero(mask == 1)),
            )

    def test_temporal_shoot_memory_recovers_complete_dropout(self) -> None:
        previous = np.zeros((24, 32), dtype=np.uint8)
        previous[5:9, 8:14] = 1
        current = np.zeros_like(previous)

        tracked, metadata = _fill_shoot_area_from_temporal_memory(
            current,
            previous,
        )

        self.assertTrue(np.array_equal(tracked, previous))
        self.assertEqual(int(metadata["observed_area_px"]), 0)
        self.assertEqual(int(metadata["previous_area_px"]), 24)
        self.assertEqual(int(metadata["tracked_area_px"]), 24)
        self.assertEqual(int(metadata["carried_pixels"]), 24)
        self.assertEqual(
            metadata["status"],
            "crown_aligned_previous_mask_carried",
        )

    def test_temporal_shoot_memory_follows_crown_translation(self) -> None:
        previous = np.zeros((40, 60), dtype=np.uint8)
        previous[10:15, 10:15] = 1
        expected = np.zeros_like(previous)
        expected[12:17, 18:23] = 1
        current = np.zeros_like(previous)
        current[12:15, 18:23] = 1

        tracked, metadata = _fill_shoot_area_from_temporal_memory(
            current,
            previous,
            shift_x=8.0,
            shift_y=2.0,
        )

        self.assertTrue(np.array_equal(tracked, expected))
        self.assertEqual(int(metadata["observed_area_px"]), 15)
        self.assertEqual(int(metadata["previous_area_px"]), 25)
        self.assertEqual(int(metadata["tracked_area_px"]), 25)
        self.assertEqual(int(metadata["carried_pixels"]), 10)

    def test_temporal_shoot_memory_preserves_observed_growth_without_carry(self) -> None:
        previous = np.zeros((24, 32), dtype=np.uint8)
        previous[8:11, 10:14] = 1
        current = np.zeros_like(previous)
        current[6:11, 8:14] = 1

        tracked, metadata = _fill_shoot_area_from_temporal_memory(
            current,
            previous,
            shift_x=3.0,
            shift_y=2.0,
        )

        self.assertTrue(np.array_equal(tracked, current))
        self.assertEqual(int(metadata["observed_area_px"]), 30)
        self.assertEqual(int(metadata["previous_area_px"]), 12)
        self.assertEqual(int(metadata["tracked_area_px"]), 30)
        self.assertEqual(int(metadata["carried_pixels"]), 0)
        self.assertEqual(metadata["status"], "observed_nondecreasing")

    def test_temporal_shoot_memory_is_isolated_by_plant_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.full((50, 100, 3), 220, dtype=np.uint8)
            masks = [
                np.zeros((50, 100), dtype=np.uint8),
                np.zeros((50, 100), dtype=np.uint8),
            ]
            masks[0][8:11, 12:16] = 2
            masks[0][8:12, 70:76] = 2
            masks[1][8:13, 70:76] = 2
            items = [
                DatasetImageItem(
                    uid=f"frame-{frame_index}",
                    name=f"frame_{frame_index}.png",
                    path=root / f"frame_{frame_index}.png",
                    image=image,
                )
                for frame_index in range(2)
            ]
            rows = [
                {
                    "frame_index": 0,
                    "plant_id": "plant_01",
                    "shoot_bbox_x": 12,
                    "shoot_bbox_y": 8,
                    "shoot_bbox_w": 4,
                    "shoot_bbox_h": 3,
                    "shoot_grayscale_crown_rescue_applied": False,
                    "ownership_crown_center_x": 14.0,
                    "ownership_crown_center_y": 11.0,
                    "shoot_measurement_source": "explicit_shoot_class",
                },
                {
                    "frame_index": 0,
                    "plant_id": "plant_02",
                    "shoot_bbox_x": 70,
                    "shoot_bbox_y": 8,
                    "shoot_bbox_w": 6,
                    "shoot_bbox_h": 4,
                    "shoot_grayscale_crown_rescue_applied": False,
                    "ownership_crown_center_x": 73.0,
                    "ownership_crown_center_y": 12.0,
                    "shoot_measurement_source": "explicit_shoot_class",
                },
                {
                    "frame_index": 1,
                    "plant_id": "plant_01",
                    "shoot_bbox_x": 0,
                    "shoot_bbox_y": 0,
                    "shoot_bbox_w": 0,
                    "shoot_bbox_h": 0,
                    "shoot_grayscale_crown_rescue_applied": False,
                    "ownership_crown_center_x": 14.0,
                    "ownership_crown_center_y": 11.0,
                    "shoot_measurement_source": "explicit_shoot_class",
                },
                {
                    "frame_index": 1,
                    "plant_id": "plant_02",
                    "shoot_bbox_x": 70,
                    "shoot_bbox_y": 8,
                    "shoot_bbox_w": 6,
                    "shoot_bbox_h": 5,
                    "shoot_grayscale_crown_rescue_applied": False,
                    "ownership_crown_center_x": 73.0,
                    "ownership_crown_center_y": 12.0,
                    "shoot_measurement_source": "explicit_shoot_class",
                },
            ]

            paths = _write_accepted_shoot_display_masks(
                rows,
                [
                    {"SourceFile": str(root / "frame_0.png")},
                    {"SourceFile": str(root / "frame_1.png")},
                ],
                items,
                masks,
                AnalyticsConfig(
                    root_class_id=1,
                    lateral_class_id=3,
                    shoot_class_id=2,
                    shoot_rgb_green_only_enabled=False,
                ),
                root / "display",
            )

            display = np.asarray(Image.open(paths[1]), dtype=np.uint8)
            plant_one = rows[2]
            plant_two = rows[3]
            self.assertEqual(int(plant_one["shoot_area_px"]), 12)
            self.assertEqual(int(plant_one["shoot_temporal_carried_pixels"]), 12)
            self.assertTrue(bool(plant_one["shoot_temporal_tracking_applied"]))
            self.assertIn(
                "hades_bw_temporal_crown_track",
                str(plant_one["shoot_measurement_source"]),
            )
            self.assertEqual(int(plant_two["shoot_area_px"]), 30)
            self.assertEqual(int(plant_two["shoot_temporal_carried_pixels"]), 0)
            self.assertFalse(bool(plant_two["shoot_temporal_tracking_applied"]))
            self.assertEqual(int(np.count_nonzero(display[8:11, 12:16] == 2)), 12)
            self.assertEqual(int(np.count_nonzero(display[8:13, 70:76] == 2)), 30)
            self.assertEqual(int(np.count_nonzero(display == 2)), 42)

    def test_temporal_visual_reacquisition_follows_moved_bw_rosette(self) -> None:
        image = np.full((120, 120, 3), 220, dtype=np.uint8)
        image[52:68, 62:82] = 20
        root = np.zeros((120, 120), dtype=np.uint8)
        root[64:110, 72] = 1
        previous = np.zeros((120, 120), dtype=np.uint8)
        previous[20:36, 52:72] = 1

        recovered, metadata = _recover_grayscale_shoot_from_temporal_lane(
            image,
            root,
            previous,
            lane_left=40,
            lane_right=100,
            config=AnalyticsConfig(
                shoot_temporal_visual_max_shift_px=80.0,
                shoot_temporal_visual_max_root_distance_px=24.0,
            ),
        )

        ys, xs = np.nonzero(recovered)
        self.assertGreater(int(xs.size), 0)
        self.assertGreater(float(ys.mean()), 50.0)
        self.assertEqual(metadata["status"], "recovered")

    def test_overlapping_shoot_boxes_are_exclusive_by_seedling_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.full((60, 100, 3), 220, dtype=np.uint8)
            mask = np.zeros((60, 100), dtype=np.uint8)
            mask[10:20, 40:60] = 2
            rows = [
                {
                    "frame_index": 0,
                    "plant_id": "plant_01",
                    "shoot_bbox_x": 35,
                    "shoot_bbox_y": 8,
                    "shoot_bbox_w": 30,
                    "shoot_bbox_h": 14,
                    "ownership_crown_center_x": 30.0,
                    "ownership_crown_center_y": 20.0,
                },
                {
                    "frame_index": 0,
                    "plant_id": "plant_02",
                    "shoot_bbox_x": 35,
                    "shoot_bbox_y": 8,
                    "shoot_bbox_w": 30,
                    "shoot_bbox_h": 14,
                    "ownership_crown_center_x": 70.0,
                    "ownership_crown_center_y": 20.0,
                },
            ]
            item = DatasetImageItem("f0", "f0.png", root / "f0.png", image)
            paths = _write_accepted_shoot_display_masks(
                rows,
                [{"SourceFile": str(item.path)}],
                [item],
                [mask],
                AnalyticsConfig(shoot_class_id=2, shoot_rgb_green_only_enabled=False),
                root / "display",
            )

            display = np.asarray(Image.open(paths[0]), dtype=np.uint8)
            row_sum = sum(int(row["shoot_area_px"]) for row in rows)
            self.assertEqual(row_sum, 200)
            self.assertEqual(row_sum, int(np.count_nonzero(display == 2)))

    def test_temporal_shoot_never_overwrites_current_root_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.full((50, 70, 3), 220, dtype=np.uint8)
            masks = [np.zeros((50, 70), dtype=np.uint8) for _ in range(2)]
            masks[0][8:18, 25:35] = 2
            masks[1][8:18, 25:35] = 1
            items = [
                DatasetImageItem(f"f{i}", f"f{i}.png", root / f"f{i}.png", image)
                for i in range(2)
            ]
            rows = [
                {
                    "frame_index": frame_index,
                    "plant_id": "plant_01",
                    "shoot_bbox_x": 25 if frame_index == 0 else 0,
                    "shoot_bbox_y": 8 if frame_index == 0 else 0,
                    "shoot_bbox_w": 10 if frame_index == 0 else 0,
                    "shoot_bbox_h": 10 if frame_index == 0 else 0,
                    "ownership_crown_center_x": 30.0,
                    "ownership_crown_center_y": 18.0,
                }
                for frame_index in range(2)
            ]
            paths = _write_accepted_shoot_display_masks(
                rows,
                [{"SourceFile": str(item.path)} for item in items],
                items,
                masks,
                AnalyticsConfig(shoot_class_id=2, shoot_rgb_green_only_enabled=False),
                root / "display",
            )

            display = np.asarray(Image.open(paths[1]), dtype=np.uint8)
            self.assertEqual(
                int(np.count_nonzero(display == 1)),
                int(np.count_nonzero(masks[1] == 1)),
            )
            self.assertEqual(int(np.count_nonzero((display == 2) & (masks[1] == 1))), 0)

    def test_distant_shoot_spike_does_not_advance_temporal_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.full((60, 120, 3), 220, dtype=np.uint8)
            masks = [np.zeros((60, 120), dtype=np.uint8) for _ in range(2)]
            masks[0][10:15, 10:15] = 2
            masks[1][20:30, 90:100] = 2
            items = [
                DatasetImageItem(f"f{i}", f"f{i}.png", root / f"f{i}.png", image)
                for i in range(2)
            ]
            rows = [
                {
                    "frame_index": 0,
                    "plant_id": "plant_01",
                    "shoot_bbox_x": 10,
                    "shoot_bbox_y": 10,
                    "shoot_bbox_w": 5,
                    "shoot_bbox_h": 5,
                    "ownership_crown_center_x": 12.0,
                    "ownership_crown_center_y": 15.0,
                },
                {
                    "frame_index": 1,
                    "plant_id": "plant_01",
                    "shoot_bbox_x": 90,
                    "shoot_bbox_y": 20,
                    "shoot_bbox_w": 10,
                    "shoot_bbox_h": 10,
                    "ownership_crown_center_x": 12.0,
                    "ownership_crown_center_y": 15.0,
                },
            ]
            _write_accepted_shoot_display_masks(
                rows,
                [{"SourceFile": str(item.path)} for item in items],
                items,
                masks,
                AnalyticsConfig(
                    shoot_class_id=2,
                    shoot_rgb_green_only_enabled=False,
                    shoot_temporal_visual_max_shift_px=30.0,
                ),
                root / "display",
            )

            self.assertEqual(int(rows[1]["shoot_area_px"]), 25)
            self.assertEqual(
                rows[1]["shoot_temporal_tracking_status"],
                "crown_aligned_previous_mask_carried",
            )

    def test_plate_worker_conserves_masks_and_resumes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = []
            for frame_index in range(2):
                image_path = root / f"frame_{frame_index}.png"
                mask_path = root / f"frame_{frame_index}_mask.png"
                Image.fromarray(np.full((64, 96, 3), 235, dtype=np.uint8)).save(image_path)
                mask = np.zeros((64, 96), dtype=np.uint8)
                mask[5:58, 18] = 1
                mask[5:58, 76] = 1
                mask[24, 10:19] = 3
                mask[36, 76:86] = 3
                Image.fromarray(mask).save(mask_path)
                frames.append(
                    {
                        "Series": "plate_a",
                        "PetriDish": "plate_a",
                        "Timestamp": f"2026-01-0{frame_index + 1}T00:00:00",
                        "FrameIndex": frame_index,
                        "RelativeFolder": ".",
                        "SourceFile": str(image_path),
                        "OutputMaskPath": str(mask_path),
                        "PreviewImagePath": str(image_path),
                    }
                )
            task = {
                "series": "plate_a",
                "petri": "plate_a",
                "frames": frames,
                "config": {
                    "root_class_id": 1,
                    "lateral_class_id": 3,
                    "shoot_class_id": 2,
                    "expected_plants": 2,
                    "pixel_size_mm": 0.05,
                    "timestep_hours": 24.0,
                    "shoot_rgb_green_only_enabled": False,
                    "min_component_area": 1,
                    "prune_branch_px": 1,
                },
            }
            task["fingerprint"] = _plate_fingerprint(task)
            task["checkpoint_path"] = str(root / "checkpoint.csv")

            result = _process_plate(task)
            resumed = _process_plate(task)
            checkpoint = pd.read_csv(task["checkpoint_path"])

            self.assertEqual(result["status"], "complete")
            self.assertEqual(resumed["status"], "resumed")
            self.assertEqual(int(result["rows"]), 4)
            self.assertEqual(float(result["conservation"]["max_length_error_px"]), 0.0)
            self.assertEqual(int(result["conservation"]["max_area_error_px"]), 0)
            self.assertEqual(int(result["alignment_accepted_frames"]), 0)
            self.assertEqual(int(result["alignment_total_frames"]), 2)
            self.assertEqual(int(result["root_crown_fallback_frames"]), 2)
            self.assertEqual(set(checkpoint["ownership_measurement_scope"]), {"current_frame_prediction"})
            self.assertEqual(set(checkpoint["ownership_frame_motion_source"]), {"root_crown_fallback"})
            self.assertEqual(set(checkpoint["ownership_frame_alignment_status"]), {"insufficient_texture"})
            self.assertEqual(int(result["accepted_shoot_display_masks"]), 2)
            self.assertTrue(
                all(
                    Path(value).exists()
                    for value in checkpoint["OwnershipDisplayMaskPath"].astype(str)
                )
            )

    def test_plate_worker_conserves_root_pixels_when_shoot_exclusion_starts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = []
            for frame_index in range(7):
                image_path = root / f"frame_{frame_index}.png"
                mask_path = root / f"frame_{frame_index}_mask.png"
                Image.fromarray(np.full((64, 96, 3), 235, dtype=np.uint8)).save(image_path)
                mask = np.zeros((64, 96), dtype=np.uint8)
                mask[10:58, 18] = 1
                mask[10:58, 76] = 1
                mask[3:9, 14:23] = 2
                mask[3:9, 72:81] = 2
                Image.fromarray(mask).save(mask_path)
                frames.append(
                    {
                        "Series": "plate_with_shoots",
                        "PetriDish": "plate_with_shoots",
                        "Timestamp": f"2026-01-{frame_index + 1:02d}T00:00:00",
                        "FrameIndex": frame_index,
                        "RelativeFolder": ".",
                        "SourceFile": str(image_path),
                        "OutputMaskPath": str(mask_path),
                        "PreviewImagePath": str(image_path),
                    }
                )
            task = {
                "series": "plate_with_shoots",
                "petri": "plate_with_shoots",
                "frames": frames,
                "config": {
                    "root_class_id": 1,
                    "lateral_class_id": 3,
                    "shoot_class_id": 2,
                    "expected_plants": 2,
                    "pixel_size_mm": 0.05,
                    "timestep_hours": 24.0,
                    "shoot_rgb_green_only_enabled": False,
                    "min_component_area": 1,
                    "prune_branch_px": 1,
                },
            }
            task["fingerprint"] = _plate_fingerprint(task)
            task["checkpoint_path"] = str(root / "checkpoint.csv")

            result = _process_plate(task)

            self.assertEqual(result["status"], "complete")
            self.assertEqual(float(result["conservation"]["max_length_error_px"]), 0.0)
            self.assertEqual(int(result["conservation"]["max_area_error_px"]), 0)
            self.assertGreaterEqual(
                int(result["measurement_conservation_recovery_frames"]),
                1,
            )
            self.assertGreater(
                int(result["measurement_conservation_recovery_pixels"]),
                0,
            )

    def test_timelapse_blanks_quantitative_values_when_any_owner_is_invalid(self) -> None:
        master = pd.DataFrame(
            [
                {
                    "Series": "s",
                    "PetriDish": "p",
                    "Timestamp": "2026-01-01",
                    "FrameIndex": 0,
                    "plant_id": "plant_01",
                    "ownership_measurement_valid": True,
                    "total_root_length_mm_raw": 9.5,
                    "total_root_length_weighted_mm_raw": 10.5,
                    "primary_root_length_mm_raw": 6.5,
                    "primary_root_length_weighted_mm_raw": 7.5,
                    "total_root_area_mm2": 1.5,
                    "primary_root_area_mm2": 1.0,
                    "total_root_length_mm_clean": 10.0,
                    "total_root_length_weighted_mm_clean": 11.0,
                    "primary_root_length_mm_clean": 7.0,
                    "primary_root_length_weighted_mm_clean": 8.0,
                    "lateral_total_length_mm": 3.0,
                    "lateral_total_length_weighted_mm": 3.0,
                    "shoot_area_mm2": 2.0,
                    "shoot_area_px": 20,
                },
                {
                    "Series": "s",
                    "PetriDish": "p",
                    "Timestamp": "2026-01-01",
                    "FrameIndex": 0,
                    "plant_id": "plant_02",
                    "ownership_measurement_valid": False,
                    "total_root_length_mm_raw": 11.5,
                    "total_root_length_weighted_mm_raw": 12.5,
                    "primary_root_length_mm_raw": 7.5,
                    "primary_root_length_weighted_mm_raw": 8.5,
                    "total_root_area_mm2": 2.5,
                    "primary_root_area_mm2": 1.5,
                    "total_root_length_mm_clean": 12.0,
                    "total_root_length_weighted_mm_clean": 13.0,
                    "primary_root_length_mm_clean": 8.0,
                    "primary_root_length_weighted_mm_clean": 9.0,
                    "lateral_total_length_mm": 4.0,
                    "lateral_total_length_weighted_mm": 4.0,
                    "shoot_area_mm2": 2.0,
                    "shoot_area_px": 20,
                },
            ]
        )

        timelapse = _build_timelapse(master, expected_plants=2)

        self.assertFalse(bool(timelapse.iloc[0]["ownership_complete"]))
        self.assertTrue(pd.isna(timelapse.iloc[0]["total_root_length_mm"]))
        self.assertEqual(int(timelapse.iloc[0]["ownership_valid_owners"]), 1)
        self.assertEqual(float(timelapse.iloc[0]["plate_mask_total_root_length_mm"]), 21.0)
        self.assertEqual(float(timelapse.iloc[0]["plate_mask_primary_root_length_mm"]), 14.0)
        self.assertEqual(float(timelapse.iloc[0]["plate_mask_lateral_root_length_mm"]), 7.0)
        self.assertTrue(bool(timelapse.iloc[0]["plate_mask_root_totals_ownership_independent"]))

    def test_baseline_safeguard_rolls_back_entire_regressing_plate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.csv"
            shared = {
                "Series": "s",
                "Timestamp": "2026-01-01",
                "FrameIndex": 0,
                "plant_id": "plant_01",
            }
            baseline = pd.DataFrame(
                [
                    {
                        **shared,
                        "PetriDish": "plate_a",
                        "ownership_measurement_valid": True,
                        "total_root_length_mm_clean": 10.0,
                    },
                    {
                        **shared,
                        "PetriDish": "plate_b",
                        "ownership_measurement_valid": True,
                        "total_root_length_mm_clean": 20.0,
                    },
                ]
            )
            current = baseline.copy()
            current.loc[current["PetriDish"] == "plate_a", "ownership_measurement_valid"] = False
            current.loc[:, "total_root_length_mm_clean"] = [999.0, 21.0]
            baseline.to_csv(baseline_path, index=False)

            guarded, audit = _apply_baseline_validity_safeguard(
                current,
                baseline_path,
            )

            plate_a = guarded[guarded["PetriDish"] == "plate_a"].iloc[0]
            plate_b = guarded[guarded["PetriDish"] == "plate_b"].iloc[0]
            self.assertTrue(bool(plate_a["ownership_measurement_valid"]))
            self.assertEqual(float(plate_a["total_root_length_mm_clean"]), 10.0)
            self.assertEqual(
                plate_a["ownership_plate_result_source"],
                "baseline_v8_non_regression_rollback",
            )
            self.assertEqual(float(plate_b["total_root_length_mm_clean"]), 21.0)
            self.assertEqual(plate_b["ownership_plate_result_source"], "learned_v11")
            self.assertEqual(len(audit), 1)
            self.assertEqual(audit[0]["plate"], "plate_a")
            self.assertEqual(int(audit[0]["regressed_valid_rows"]), 1)


if __name__ == "__main__":
    unittest.main()
