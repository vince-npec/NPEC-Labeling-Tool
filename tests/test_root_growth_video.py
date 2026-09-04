from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from resources.root_growth_video import (
    LazyRootGrowthVideoConfig,
    _colorized_mask,
    _display_timeline_position,
    _display_mask_with_rgb_shoot_rescue,
    _friendly_source_label,
    _mask_total_dataframe_requests_rgb_shoot_rescue,
    _mask_total_row_shoot_source,
    _resolve_mask_path,
    _short_timeline_position,
    _timeline_axis_title,
    apply_accepted_shoots_to_frame_rows,
    generate_lazy_mask_total_root_growth_video,
    generate_lazy_root_growth_video,
)


class RootGrowthVideoTests(unittest.TestCase):
    def test_accepted_shoots_replace_raw_frame_video_mask_and_metrics(self) -> None:
        frame_df = pd.DataFrame(
            [
                {
                    "SourceFile": "/images/plate_01.png",
                    "OutputMaskPath": "/masks/raw.png",
                    "shoot_area_px": 999,
                    "shoot_area_mm2": 9.99,
                    "shoot_measurement_source": "mask_class",
                },
                {
                    "SourceFile": "/images/unmatched.png",
                    "OutputMaskPath": "/masks/unmatched.png",
                    "shoot_area_px": 77,
                    "shoot_area_mm2": 0.77,
                    "shoot_measurement_source": "mask_class",
                },
            ]
        )
        ownership_df = pd.DataFrame(
            [
                {
                    "SourceFile": "/images/plate_01.png",
                    "OwnershipDisplayMaskPath": "/masks/accepted.png",
                    "shoot_area_px": 10,
                    "shoot_area_mm2": 0.10,
                    "shoot_measurement_source": "mask_class",
                },
                {
                    "SourceFile": "/images/plate_01.png",
                    "OwnershipDisplayMaskPath": "/masks/accepted.png",
                    "shoot_area_px": 20,
                    "shoot_area_mm2": 0.20,
                    "shoot_measurement_source": "hades_bw_crown_local_cv",
                },
            ]
        )

        merged = apply_accepted_shoots_to_frame_rows(frame_df, ownership_df)

        self.assertEqual(merged.loc[0, "OwnershipDisplayMaskPath"], "/masks/accepted.png")
        self.assertEqual(int(merged.loc[0, "shoot_area_px"]), 30)
        self.assertAlmostEqual(float(merged.loc[0, "shoot_area_mm2"]), 0.30)
        self.assertEqual(int(merged.loc[0, "shoot_area_green_only_px"]), 30)
        self.assertEqual(int(merged.loc[0, "shoot_accepted_component_count"]), 2)
        self.assertEqual(
            merged.loc[0, "shoot_measurement_source"],
            "mask_class,hades_bw_crown_local_cv",
        )
        self.assertTrue(bool(merged.loc[0, "shoot_accepted_ownership_applied"]))
        self.assertEqual(int(merged.loc[1, "shoot_area_px"]), 77)
        self.assertFalse(bool(merged.loc[1, "shoot_accepted_ownership_applied"]))

    def test_hades_shoot_source_label_explains_mixed_model_and_cv_frames(self) -> None:
        self.assertEqual(
            _friendly_source_label(
                "explicit_shoot_class,hades_bw_crown_local_cv"
            ),
            "Hades BW model + crown-local CV",
        )

    def test_hades_shoot_source_label_explains_temporal_tracking(self) -> None:
        self.assertEqual(
            _friendly_source_label(
                "explicit_shoot_class,hades_bw_crown_local_cv,"
                "hades_bw_temporal_crown_track"
            ),
            "Hades BW model + crown-local CV + temporal crown tracking",
        )
        self.assertEqual(
            _friendly_source_label(
                "explicit_shoot_class,hades_bw_crown_local_cv,"
                "hades_bw_temporal_visual_reacquisition"
            ),
            "Hades BW model + crown-local CV + temporal visual tracking",
        )

    def test_ownership_display_mask_is_preferred_over_raw_model_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            root = Path(tmp_dir_name)
            raw = root / "raw.png"
            accepted = root / "accepted.png"
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(raw)
            Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(accepted)
            resolved = _resolve_mask_path(
                pd.DataFrame(
                    {
                        "OutputMaskPath": [str(raw)],
                        "OwnershipDisplayMaskPath": [str(accepted)],
                    }
                )
            )
            self.assertEqual(resolved, accepted)

    def test_missing_timestamps_use_frame_labels_for_real_sequences(self) -> None:
        self.assertEqual(
            _display_timeline_position(float("nan"), position=1, frame_count=3),
            "Frame 2/3",
        )
        self.assertEqual(
            _short_timeline_position(float("nan"), position=1, frame_count=3),
            "F2",
        )
        self.assertEqual(
            _timeline_axis_title(pd.Series([float("nan"), float("nan")])),
            "Frame",
        )
        self.assertEqual(
            _display_timeline_position(float("nan"), position=0, frame_count=1),
            "Endpoint (no timestamp)",
        )

    def test_row_shoot_provenance_overrides_mixed_video_global_flag(self) -> None:
        config = LazyRootGrowthVideoConfig(show_rgb_shoot_rescue=True)
        source, green_only = _mask_total_row_shoot_source(
            pd.Series(
                {
                    "shoot_measurement_source": "mask_class",
                    "shoot_rgb_rescue_enabled": False,
                }
            ),
            config,
        )
        self.assertEqual(source, "mask class")
        self.assertFalse(green_only)

        source, green_only = _mask_total_row_shoot_source(
            pd.Series(
                {
                    "shoot_measurement_source": "rgb_green_only",
                    "shoot_rgb_rescue_enabled": True,
                }
            ),
            LazyRootGrowthVideoConfig(show_rgb_shoot_rescue=False),
        )
        self.assertEqual(source, "RGB green-only")
        self.assertTrue(green_only)

    def test_generates_lazy_root_growth_video_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_paths: list[Path] = []
            mask_paths: list[Path] = []
            for index in range(2):
                image = np.zeros((90, 120, 3), dtype=np.uint8)
                image[:, :, 0] = 42 + index * 18
                image[:, :, 1] = 58
                image[:, :, 2] = 66
                image_path = tmp_dir / f"plate_01_t{index + 1}.png"
                Image.fromarray(image).save(image_path)
                image_paths.append(image_path)

                mask = np.zeros((90, 120), dtype=np.uint8)
                mask[18:68, 42 + index * 2] = 1
                mask[22:72, 78 + index * 2] = 1
                mask[14:21, 38:47] = 2
                mask[18:25, 74:83] = 2
                mask_path = tmp_dir / f"plate_01_t{index + 1}_mask.png"
                Image.fromarray(mask, mode="L").save(mask_path)
                mask_paths.append(mask_path)

            timestamps = ["2026-07-01 08:00:00", "2026-07-02 08:00:00"]
            master_rows: list[dict[str, object]] = []
            for frame_index, (image_path, mask_path, timestamp) in enumerate(zip(image_paths, mask_paths, timestamps)):
                for plant_index, plant_id in enumerate(("plant_01", "plant_02")):
                    x = 42 + plant_index * 36 + frame_index * 2
                    master_rows.append(
                        {
                            "Series": "synthetic",
                            "PetriDish": "plate_01",
                            "Timestamp": timestamp,
                            "FrameIndex": frame_index,
                            "plant_id": plant_id,
                            "root_length_mm_clean": 15.0 + frame_index * 5.0 + plant_index,
                            "shoot_area_px": 64,
                            "shoot_area_mm2": 0.64,
                            "shoot_measurement_source": "rgb_green_only",
                            "shoot_rgb_green_only_enabled": True,
                            "ownership_measurement_valid": not (frame_index == 1 and plant_id == "plant_02"),
                            "conflict_group_size": 1,
                            "path_points": str([(x, 18), (x, 44), (x + 3, 68)]),
                            "SourceFile": str(image_path),
                            "OutputMaskPath": str(mask_path),
                        }
                    )

            total_df = pd.DataFrame(
                [
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_01",
                        "Timestamp": timestamps[0],
                        "FrameIndex": 0,
                        "total_root_length_mm": 31.0,
                        "mean_root_length_mm": 15.5,
                        "plants_detected": 2,
                        "delta_total_root_length_mm": 0.0,
                    },
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_01",
                        "Timestamp": timestamps[1],
                        "FrameIndex": 1,
                        "total_root_length_mm": 41.0,
                        "mean_root_length_mm": 20.5,
                        "plants_detected": 2,
                        "delta_total_root_length_mm": 10.0,
                    },
                ]
            )
            output_path = tmp_dir / "growth.mp4"

            result = generate_lazy_root_growth_video(
                master_df=pd.DataFrame(master_rows),
                total_df=total_df,
                output_path=output_path,
                config=LazyRootGrowthVideoConfig(fps=1, width=960, height=540),
            )

            self.assertEqual(result.frames_rendered, 2)
            self.assertEqual(result.plates_rendered, 1)
            self.assertTrue(result.video_path.exists())
            self.assertTrue(result.sample_frame_path.exists())
            self.assertTrue(result.summary_csv_path.exists())

            summary_df = pd.read_csv(result.summary_csv_path)
            self.assertEqual(summary_df["PetriDish"].tolist(), ["plate_01"])
            self.assertEqual(summary_df["root_metric_mode"].tolist(), ["total"])
            self.assertEqual(summary_df["selected_root_metric_column"].tolist(), ["total_root_length_mm"])
            self.assertAlmostEqual(float(summary_df["final_selected_root_length_mm"].iloc[0]), 41.0)
            self.assertAlmostEqual(float(summary_df["delta_selected_root_length_mm"].iloc[0]), 10.0)
            self.assertAlmostEqual(float(summary_df["delta_total_root_length_mm"].iloc[0]), 10.0)
            self.assertEqual(int(summary_df["frames_needing_review"].iloc[0]), 1)
            self.assertEqual(int(summary_df["frames_failed_review"].iloc[0]), 1)
            self.assertEqual(int(summary_df["tracks_needing_review"].iloc[0]), 1)
            self.assertEqual(str(summary_df["shoot_measurement_source"].iloc[0]), "RGB green-only")
            self.assertTrue(bool(summary_df["shoot_rgb_green_only_enabled"].iloc[0]))

            capture = cv2.VideoCapture(str(result.video_path))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 960)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 540)
                decoded_frames = 0
                while True:
                    ok, _frame = capture.read()
                    if not ok:
                        break
                    decoded_frames += 1
                self.assertEqual(decoded_frames, 2)
            finally:
                capture.release()

    def test_ownership_video_draws_seedling_and_unresolved_group_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "plate_box_t1.png"
            mask_path = tmp_dir / "plate_box_t1_mask.png"
            Image.fromarray(np.full((100, 100, 3), 210, dtype=np.uint8)).save(
                image_path
            )
            Image.fromarray(np.zeros((100, 100), dtype=np.uint8), mode="L").save(
                mask_path
            )
            master_df = pd.DataFrame(
                [
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_box",
                        "Timestamp": "2026-07-01 08:00:00",
                        "FrameIndex": 0,
                        "plant_id": "plant_01",
                        "root_length_mm_clean": 1.0,
                        "ownership_measurement_valid": False,
                        "conflict_group_id": "combined::plant_01+plant_02",
                        "conflict_group_size": 2,
                        "seedling_bbox_x": 12,
                        "seedling_bbox_y": 14,
                        "seedling_bbox_w": 28,
                        "seedling_bbox_h": 62,
                        "combined_group_bbox_x": 8,
                        "combined_group_bbox_y": 10,
                        "combined_group_bbox_w": 72,
                        "combined_group_bbox_h": 76,
                        "path_points": "[]",
                        "SourceFile": str(image_path),
                        "OutputMaskPath": str(mask_path),
                    }
                ]
            )
            total_df = pd.DataFrame(
                [
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_box",
                        "Timestamp": "2026-07-01 08:00:00",
                        "FrameIndex": 0,
                        "total_root_length_mm": 1.0,
                        "mean_root_length_mm": 1.0,
                        "plants_detected": 1,
                        "delta_total_root_length_mm": 0.0,
                    }
                ]
            )

            result = generate_lazy_root_growth_video(
                master_df=master_df,
                total_df=total_df,
                output_path=tmp_dir / "ownership_boxes.mp4",
                config=LazyRootGrowthVideoConfig(
                    fps=1,
                    width=960,
                    height=540,
                ),
            )
            sample = np.asarray(
                Image.open(result.sample_frame_path).convert("RGB")
            )
            plant_color = np.array([245, 142, 69], dtype=np.uint8)
            group_color = np.array([232, 68, 68], dtype=np.uint8)
            self.assertTrue(np.array_equal(sample[200, 70], plant_color))
            self.assertTrue(np.array_equal(sample[91, 52], group_color))

            hidden_result = generate_lazy_root_growth_video(
                master_df=master_df,
                total_df=total_df,
                output_path=tmp_dir / "ownership_boxes_hidden.mp4",
                config=LazyRootGrowthVideoConfig(
                    fps=1,
                    width=960,
                    height=540,
                    show_seedling_ownership_boxes=False,
                ),
            )
            hidden_sample = np.asarray(
                Image.open(hidden_result.sample_frame_path).convert("RGB")
            )[:, :480]
            self.assertFalse(np.array_equal(hidden_sample[200, 70], plant_color))
            self.assertFalse(np.array_equal(hidden_sample[91, 52], group_color))

    def test_lazy_root_growth_video_mask_only_honors_custom_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image = np.full((70, 90, 3), 180, dtype=np.uint8)
            image_path = tmp_dir / "plate_custom_t1.png"
            Image.fromarray(image).save(image_path)

            mask = np.zeros((70, 90), dtype=np.uint8)
            mask[16:58, 35] = 4
            mask[10:22, 52:65] = 7
            mask_path = tmp_dir / "plate_custom_t1_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)

            master_df = pd.DataFrame(
                [
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_custom",
                        "Timestamp": "2026-07-01 08:00:00",
                        "FrameIndex": 0,
                        "plant_id": "plant_01",
                        "root_length_mm_clean": 12.0,
                        "path_points": str([(35, 16), (35, 58)]),
                        "SourceFile": str(image_path),
                        "OutputMaskPath": str(mask_path),
                    }
                ]
            )
            total_df = pd.DataFrame(
                [
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_custom",
                        "Timestamp": "2026-07-01 08:00:00",
                        "FrameIndex": 0,
                        "total_root_length_mm": 12.0,
                        "mean_root_length_mm": 12.0,
                        "plants_detected": 1,
                    }
                ]
            )

            result = generate_lazy_root_growth_video(
                master_df=master_df,
                total_df=total_df,
                output_path=tmp_dir / "custom_ownership_mask_only.mp4",
                config=LazyRootGrowthVideoConfig(
                    fps=1,
                    width=960,
                    height=540,
                    mask_only=True,
                    root_class_ids=(4,),
                    shoot_class_ids=(7,),
                ),
            )

            self.assertEqual(result.frames_rendered, 1)
            sample = np.asarray(Image.open(result.sample_frame_path).convert("RGB"))
            pink_pixels = np.all(sample == np.array([255, 44, 190], dtype=np.uint8), axis=2)
            orange_pixels = np.all(sample == np.array([255, 132, 42], dtype=np.uint8), axis=2)
            self.assertGreater(int(np.count_nonzero(pink_pixels)), 0)
            self.assertGreater(int(np.count_nonzero(orange_pixels)), 0)

    def test_lazy_root_growth_video_auto_uses_green_only_shoot_display(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image = np.full((100, 100, 3), 235, dtype=np.uint8)
            image[20:36, 42:58] = [72, 132, 38]
            image[68:86, 70:90] = [118, 126, 96]
            image_path = tmp_dir / "plate_green_and_colony.png"
            Image.fromarray(image).save(image_path)

            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[15:82, 15] = 1
            mask[68:86, 70:90] = 2
            mask_path = tmp_dir / "plate_green_and_colony_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)

            master_df = pd.DataFrame(
                [
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_green_and_colony",
                        "Timestamp": "2026-07-01 08:00:00",
                        "FrameIndex": 0,
                        "plant_id": "plant_01",
                        "root_length_mm_clean": 12.0,
                        "path_points": str([(15, 15), (15, 82)]),
                        "SourceFile": str(image_path),
                        "OutputMaskPath": str(mask_path),
                    }
                ]
            )
            total_df = pd.DataFrame(
                [
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_green_and_colony",
                        "Timestamp": "2026-07-01 08:00:00",
                        "FrameIndex": 0,
                        "total_root_length_mm": 12.0,
                        "mean_root_length_mm": 12.0,
                        "plants_detected": 1,
                        "shoot_rgb_rescue_enabled": True,
                        "shoot_measurement_source": "rgb_green_only",
                    }
                ]
            )

            result = generate_lazy_root_growth_video(
                master_df=master_df,
                total_df=total_df,
                output_path=tmp_dir / "ownership_green_only.mp4",
                config=LazyRootGrowthVideoConfig(
                    fps=1,
                    width=960,
                    height=540,
                    mask_only=True,
                    mask_display_dilation_px=4,
                ),
            )

            sample = np.asarray(Image.open(result.sample_frame_path).convert("RGB"))
            pink = np.array([255, 44, 190], dtype=np.uint8)
            self.assertTrue(np.array_equal(sample[171, 240], pink))
            self.assertFalse(np.array_equal(sample[171, 289], pink))
            self.assertFalse(np.array_equal(sample[390, 374], pink))

    def test_generates_lazy_root_growth_video_summary_uses_selected_lateral_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_paths: list[Path] = []
            mask_paths: list[Path] = []
            for index in range(2):
                image = np.full((60, 80, 3), 92 + index * 12, dtype=np.uint8)
                image_path = tmp_dir / f"plate_01_lateral_t{index + 1}.png"
                Image.fromarray(image).save(image_path)
                image_paths.append(image_path)

                mask = np.zeros((60, 80), dtype=np.uint8)
                mask[14:48, 35 + index] = 1
                mask[30, 35 + index : 52 + index] = 3
                mask_path = tmp_dir / f"plate_01_lateral_t{index + 1}_mask.png"
                Image.fromarray(mask, mode="L").save(mask_path)
                mask_paths.append(mask_path)

            timestamps = ["2026-07-01 08:00:00", "2026-07-02 08:00:00"]
            master_rows: list[dict[str, object]] = []
            for frame_index, (image_path, mask_path, timestamp) in enumerate(zip(image_paths, mask_paths, timestamps)):
                master_rows.append(
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_01",
                        "Timestamp": timestamp,
                        "FrameIndex": frame_index,
                        "plant_id": "plant_01",
                        "root_length_mm_clean": 20.0 + frame_index * 10.0,
                        "total_root_length_weighted_mm_clean": 22.0 + frame_index * 14.0,
                        "primary_root_length_mm_raw": 14.0 + frame_index * 4.0,
                        "primary_root_length_weighted_mm_raw": 15.0 + frame_index * 6.0,
                        "lateral_total_length_mm": 6.0 + frame_index * 6.0,
                        "lateral_total_length_weighted_mm": 7.0 + frame_index * 8.0,
                        "path_points": str([(35 + frame_index, 14), (35 + frame_index, 30), (52 + frame_index, 30)]),
                        "SourceFile": str(image_path),
                        "OutputMaskPath": str(mask_path),
                    }
                )

            total_df = pd.DataFrame(
                [
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_01",
                        "Timestamp": timestamps[0],
                        "FrameIndex": 0,
                        "total_root_length_mm": 20.0,
                        "total_root_length_weighted_mm": 22.0,
                        "mean_root_length_mm": 20.0,
                        "mean_root_length_weighted_mm": 22.0,
                        "total_primary_root_length_mm": 14.0,
                        "mean_primary_root_length_mm": 14.0,
                        "total_lateral_root_length_mm": 6.0,
                        "mean_lateral_root_length_mm": 6.0,
                        "selected_root_metric_mode": "lateral",
                        "selected_root_metric_column": "total_lateral_root_length_mm",
                        "selected_root_length_mm": 6.0,
                        "plants_detected": 1,
                        "delta_total_root_length_mm": 0.0,
                        "delta_total_root_length_weighted_mm": 0.0,
                        "delta_total_lateral_root_length_mm": 0.0,
                        "delta_selected_root_length_mm": 0.0,
                    },
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_01",
                        "Timestamp": timestamps[1],
                        "FrameIndex": 1,
                        "total_root_length_mm": 30.0,
                        "total_root_length_weighted_mm": 36.0,
                        "mean_root_length_mm": 30.0,
                        "mean_root_length_weighted_mm": 36.0,
                        "total_primary_root_length_mm": 18.0,
                        "mean_primary_root_length_mm": 18.0,
                        "total_lateral_root_length_mm": 12.0,
                        "mean_lateral_root_length_mm": 12.0,
                        "selected_root_metric_mode": "lateral",
                        "selected_root_metric_column": "total_lateral_root_length_mm",
                        "selected_root_length_mm": 12.0,
                        "plants_detected": 1,
                        "delta_total_root_length_mm": 10.0,
                        "delta_total_root_length_weighted_mm": 14.0,
                        "delta_total_lateral_root_length_mm": 6.0,
                        "delta_selected_root_length_mm": 6.0,
                    },
                ]
            )

            result = generate_lazy_root_growth_video(
                master_df=pd.DataFrame(master_rows),
                total_df=total_df,
                output_path=tmp_dir / "lateral_growth.mp4",
                config=LazyRootGrowthVideoConfig(fps=1, width=960, height=540, root_metric_mode="lateral"),
            )

            self.assertEqual(result.frames_rendered, 2)
            summary_df = pd.read_csv(result.summary_csv_path)
            self.assertEqual(summary_df["root_metric_mode"].tolist(), ["lateral"])
            self.assertEqual(summary_df["selected_root_metric_column"].tolist(), ["total_lateral_root_length_mm"])
            self.assertAlmostEqual(float(summary_df["final_selected_root_length_mm"].iloc[0]), 12.0)
            self.assertAlmostEqual(float(summary_df["delta_selected_root_length_mm"].iloc[0]), 6.0)
            self.assertAlmostEqual(float(summary_df["delta_total_root_length_mm"].iloc[0]), 10.0)

            weighted_result = generate_lazy_root_growth_video(
                master_df=pd.DataFrame(master_rows),
                total_df=total_df,
                output_path=tmp_dir / "weighted_growth.mp4",
                config=LazyRootGrowthVideoConfig(fps=1, width=960, height=540, root_metric_mode="weighted_total"),
            )

            self.assertEqual(weighted_result.frames_rendered, 2)
            weighted_summary_df = pd.read_csv(weighted_result.summary_csv_path)
            self.assertEqual(weighted_summary_df["root_metric_mode"].tolist(), ["weighted_total"])
            self.assertEqual(weighted_summary_df["selected_root_metric_column"].tolist(), ["total_root_length_weighted_mm"])
            self.assertAlmostEqual(float(weighted_summary_df["final_selected_root_length_mm"].iloc[0]), 36.0)
            self.assertAlmostEqual(float(weighted_summary_df["delta_selected_root_length_mm"].iloc[0]), 14.0)
            self.assertAlmostEqual(float(weighted_summary_df["delta_total_root_length_weighted_mm"].iloc[0]), 14.0)

    def test_generates_mask_total_root_growth_video_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            rows: list[dict[str, object]] = []
            for frame_index in range(2):
                image = np.zeros((90, 120, 3), dtype=np.uint8)
                image[:, :, 0] = 52 + frame_index * 12
                image[:, :, 1] = 64
                image[:, :, 2] = 70
                image_path = tmp_dir / f"plate_01_t{frame_index + 1}.png"
                Image.fromarray(image).save(image_path)

                mask = np.zeros((90, 120), dtype=np.uint8)
                mask[18:70, 44 + frame_index] = 1
                mask[34, 44 + frame_index : 68 + frame_index] = 3
                mask[14:21, 38:47] = 2
                mask_path = tmp_dir / f"plate_01_t{frame_index + 1}_mask.png"
                Image.fromarray(mask, mode="L").save(mask_path)
                rows.append(
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_01",
                        "Timestamp": f"2026-07-0{frame_index + 1} 08:00:00",
                        "FrameIndex": frame_index,
                        "SourceFile": str(image_path),
                        "OutputMaskPath": str(mask_path),
                        "total_root_length_mm": 25.0 + frame_index * 12.0,
                        "total_root_length_weighted_mm": 27.0 + frame_index * 12.0,
                        "primary_root_length_mm": 18.0 + frame_index * 6.0,
                        "primary_root_length_weighted_mm": 19.0 + frame_index * 6.0,
                        "lateral_root_length_mm": 7.0 + frame_index * 6.0,
                        "lateral_root_length_weighted_mm": 8.0 + frame_index * 6.0,
                        "root_pixels_class_1": 52,
                        "lateral_pixels_class_3": 24,
                        "shoot_seed_pixels_class_2": 63,
                        "shoot_pixels_class_2": 63,
                        "shoot_area_px": 63,
                        "shoot_area_mm2": 0.063 + frame_index * 0.01,
                        "delta_shoot_area_mm2": 0.0 if frame_index == 0 else 0.01,
                        "measured_mask_pixels": 76,
                        "component_count": 1,
                        "plants_detected": 1,
                    }
                )

            output_path = tmp_dir / "mask_total_growth.mp4"
            result = generate_lazy_mask_total_root_growth_video(
                total_df=pd.DataFrame(rows),
                output_path=output_path,
                config=LazyRootGrowthVideoConfig(fps=1, width=960, height=540, mask_only=True, root_metric_mode="lateral"),
            )

            self.assertEqual(result.frames_rendered, 2)
            self.assertEqual(result.plates_rendered, 1)
            self.assertTrue(result.video_path.exists())
            self.assertTrue(result.sample_frame_path.exists())
            self.assertTrue(result.summary_csv_path.exists())

            summary_df = pd.read_csv(result.summary_csv_path)
            self.assertEqual(summary_df["PetriDish"].tolist(), ["plate_01"])
            self.assertEqual(summary_df["root_metric_mode"].tolist(), ["lateral"])
            self.assertEqual(summary_df["root_metric_column"].tolist(), ["lateral_root_length_mm"])
            self.assertAlmostEqual(float(summary_df["final_selected_root_length_mm"].iloc[0]), 13.0)
            self.assertAlmostEqual(float(summary_df["delta_selected_root_length_mm"].iloc[0]), 6.0)
            self.assertAlmostEqual(float(summary_df["delta_total_root_length_mm"].iloc[0]), 12.0)
            self.assertAlmostEqual(float(summary_df["final_primary_root_length_mm"].iloc[0]), 24.0)
            self.assertAlmostEqual(float(summary_df["final_lateral_root_length_mm"].iloc[0]), 13.0)
            self.assertAlmostEqual(float(summary_df["final_shoot_area_mm2"].iloc[0]), 0.073)
            self.assertAlmostEqual(float(summary_df["delta_shoot_area_mm2"].iloc[0]), 0.01)
            sample = np.asarray(Image.open(result.sample_frame_path).convert("RGB"))
            pink_pixels = np.all(sample == np.array([255, 44, 190], dtype=np.uint8), axis=2)
            self.assertGreater(int(np.count_nonzero(pink_pixels)), 0)

            capture = cv2.VideoCapture(str(result.video_path))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 960)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 540)
            finally:
                capture.release()

    def test_mask_total_video_colors_configured_shoot_class_pink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image = np.zeros((80, 100, 3), dtype=np.uint8)
            image_path = tmp_dir / "plate_01_t1.png"
            Image.fromarray(image).save(image_path)

            mask = np.zeros((80, 100), dtype=np.uint8)
            mask[20:65, 40] = 1
            mask[12:22, 52:64] = 4
            mask_path = tmp_dir / "plate_01_t1_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)

            result = generate_lazy_mask_total_root_growth_video(
                total_df=pd.DataFrame(
                    [
                        {
                            "Series": "synthetic",
                            "PetriDish": "plate_01",
                            "Timestamp": "2026-07-01 08:00:00",
                            "FrameIndex": 0,
                            "SourceFile": str(image_path),
                            "OutputMaskPath": str(mask_path),
                            "total_root_length_mm": 10.0,
                            "primary_root_length_mm": 10.0,
                            "lateral_root_length_mm": 0.0,
                            "shoot_class_ids": "4",
                            "shoot_area_px": 120,
                            "shoot_area_mm2": 1.2,
                            "measured_mask_pixels": 45,
                            "component_count": 1,
                            "plants_detected": 1,
                        }
                    ]
                ),
                output_path=tmp_dir / "custom_shoot_growth.mp4",
                config=LazyRootGrowthVideoConfig(
                    fps=1,
                    width=960,
                    height=540,
                    mask_only=True,
                    root_class_ids=(1, 3),
                    shoot_class_ids=(4,),
                ),
            )

            sample = np.asarray(Image.open(result.sample_frame_path).convert("RGB"))
            pink_pixels = np.all(sample == np.array([255, 44, 190], dtype=np.uint8), axis=2)
            self.assertGreater(int(np.count_nonzero(pink_pixels)), 0)

    def test_mask_total_video_prefers_temporally_accepted_shoot_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "plate_accepted_t1.png"
            Image.fromarray(np.full((100, 100, 3), 210, dtype=np.uint8)).save(image_path)

            raw_mask = np.zeros((100, 100), dtype=np.uint8)
            raw_mask[35:85, 50] = 1
            raw_path = tmp_dir / "plate_accepted_t1_raw.png"
            Image.fromarray(raw_mask, mode="L").save(raw_path)

            accepted_mask = raw_mask.copy()
            accepted_mask[20:36, 18:38] = 2
            accepted_path = tmp_dir / "plate_accepted_t1_tracked.png"
            Image.fromarray(accepted_mask, mode="L").save(accepted_path)

            result = generate_lazy_mask_total_root_growth_video(
                total_df=pd.DataFrame(
                    [
                        {
                            "Series": "synthetic",
                            "PetriDish": "plate_accepted",
                            "Timestamp": "2026-07-01 08:00:00",
                            "FrameIndex": 0,
                            "SourceFile": str(image_path),
                            "OutputMaskPath": str(raw_path),
                            "OwnershipDisplayMaskPath": str(accepted_path),
                            "total_root_length_mm": 10.0,
                            "primary_root_length_mm": 10.0,
                            "lateral_root_length_mm": 0.0,
                            "shoot_area_px": 320,
                            "shoot_area_mm2": 3.2,
                            "measured_mask_pixels": 50,
                            "component_count": 1,
                            "plants_detected": 1,
                        }
                    ]
                ),
                output_path=tmp_dir / "accepted_shoot_video.mp4",
                config=LazyRootGrowthVideoConfig(
                    fps=1,
                    width=960,
                    height=540,
                    mask_only=True,
                    root_class_ids=(1, 3),
                    shoot_class_ids=(2,),
                ),
            )

            sample = np.asarray(Image.open(result.sample_frame_path).convert("RGB"))
            pink = np.array([255, 44, 190], dtype=np.uint8)
            accepted_roi = sample[125:220, 75:210]
            self.assertGreater(
                int(np.count_nonzero(np.all(accepted_roi == pink, axis=2))),
                0,
            )

    def test_rgb_shoot_rescue_is_added_to_video_display_mask_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image = np.full((90, 120, 3), 220, dtype=np.uint8)
            image[14:32, 54:72] = np.array([74, 126, 32], dtype=np.uint8)
            image_path = tmp_dir / "plate_green.png"
            Image.fromarray(image).save(image_path)

            mask = np.zeros((90, 120), dtype=np.uint8)
            mask[18:60, 60] = 1
            mask[68:82, 88:105] = 2
            display = _display_mask_with_rgb_shoot_rescue(
                mask,
                image_path,
                root_class_ids=(1, 3),
                shoot_class_ids=(2,),
            )

            self.assertEqual(int(mask[22, 55]), 0)
            self.assertEqual(int(display[22, 55]), 2)
            self.assertEqual(int(display[22, 60]), 1)
            self.assertEqual(int(mask[72, 92]), 2)
            self.assertEqual(int(display[72, 92]), 0)

    def test_rgb_shoot_rescue_removes_model_shoot_when_source_image_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            mask = np.zeros((90, 120), dtype=np.uint8)
            mask[18:60, 60] = 1
            mask[68:82, 88:105] = 2

            display = _display_mask_with_rgb_shoot_rescue(
                mask,
                tmp_dir / "missing_source.png",
                root_class_ids=(1, 3),
                shoot_class_ids=(2,),
            )

            self.assertEqual(int(display[30, 60]), 1)
            self.assertEqual(int(mask[72, 92]), 2)
            self.assertEqual(int(display[72, 92]), 0)

    def test_mask_total_video_rgb_rescue_does_not_display_gray_colony_as_shoot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image = np.full((100, 100, 3), 235, dtype=np.uint8)
            image[20:36, 42:58] = [72, 132, 38]
            image[32:50, 62:78] = [118, 126, 96]
            image_path = tmp_dir / "plate_green_and_colony.png"
            Image.fromarray(image).save(image_path)

            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[20:80, 12] = 1
            mask[32:50, 62:78] = 2
            mask_path = tmp_dir / "plate_green_and_colony_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)

            result = generate_lazy_mask_total_root_growth_video(
                total_df=pd.DataFrame(
                    [
                        {
                            "Series": "synthetic",
                            "PetriDish": "plate_green_and_colony",
                            "Timestamp": "2026-07-01 08:00:00",
                            "FrameIndex": 0,
                            "SourceFile": str(image_path),
                            "OutputMaskPath": str(mask_path),
                            "total_root_length_mm": 10.0,
                            "primary_root_length_mm": 10.0,
                            "lateral_root_length_mm": 0.0,
                            "shoot_area_px": 256,
                            "shoot_area_mm2": 2.56,
                            "shoot_rgb_rescue_enabled": True,
                            "shoot_measurement_source": "rgb_green_only",
                            "measured_mask_pixels": 60,
                            "component_count": 1,
                            "plants_detected": 1,
                        }
                    ]
                ),
                output_path=tmp_dir / "green_only_video.mp4",
                config=LazyRootGrowthVideoConfig(
                    fps=1,
                    width=960,
                    height=540,
                    mask_only=True,
                    mask_display_dilation_px=4,
                    root_class_ids=(1, 3),
                    shoot_class_ids=(2,),
                    show_rgb_shoot_rescue=True,
                ),
            )

            sample = np.asarray(Image.open(result.sample_frame_path).convert("RGB"))
            pink = np.array([255, 44, 190], dtype=np.uint8)
            self.assertTrue(np.array_equal(sample[171, 240], pink))
            self.assertFalse(np.array_equal(sample[230, 330], pink))
            summary_df = pd.read_csv(result.summary_csv_path)
            self.assertEqual(str(summary_df["shoot_measurement_source"].iloc[0]), "RGB green-only")
            self.assertTrue(bool(summary_df["shoot_rgb_green_only_enabled"].iloc[0]))

    def test_rgb_shoot_rescue_display_rejects_lower_gray_green_colony(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image = np.full((100, 100, 3), 235, dtype=np.uint8)
            image[20:36, 42:58] = [72, 132, 38]
            cv2.ellipse(image, (68, 56), (14, 8), 0, 0, 360, (95, 104, 48), -1)
            image_path = tmp_dir / "plate_lower_colony.png"
            Image.fromarray(image).save(image_path)

            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[20:80, 12] = 1
            cv2.ellipse(mask, (68, 56), (14, 8), 0, 0, 360, 2, -1)

            display = _display_mask_with_rgb_shoot_rescue(
                mask,
                image_path,
                root_class_ids=(1, 3),
                shoot_class_ids=(2,),
            )

            self.assertEqual(int(display[28, 50]), 2)
            self.assertEqual(int(mask[56, 68]), 2)
            self.assertEqual(int(display[56, 68]), 0)

    def test_mask_total_video_auto_detects_rgb_green_shoot_tables(self) -> None:
        self.assertTrue(
            _mask_total_dataframe_requests_rgb_shoot_rescue(
                pd.DataFrame({"shoot_rgb_rescue_enabled": [False, True]})
            )
        )
        self.assertTrue(
            _mask_total_dataframe_requests_rgb_shoot_rescue(
                pd.DataFrame({"shoot_measurement_source": ["mask_class", "rgb_green_only"]})
            )
        )
        self.assertFalse(
            _mask_total_dataframe_requests_rgb_shoot_rescue(
                pd.DataFrame({"shoot_rgb_rescue_enabled": [False], "shoot_measurement_source": ["mask_class"]})
            )
        )

    def test_mask_total_colorized_mask_uses_selected_root_and_shoot_classes(self) -> None:
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[2:8, 4] = 4
        mask[10, 5:12] = 5
        mask[2:5, 10:14] = 6

        image = np.asarray(
            _colorized_mask(
                mask,
                root_class_ids=(4, 5),
                shoot_class_ids=(6,),
            ).convert("RGB")
        )

        orange_pixels = np.all(image == np.array([255, 132, 42], dtype=np.uint8), axis=2)
        blue_pixels = np.all(image == np.array([112, 158, 238], dtype=np.uint8), axis=2)
        pink_pixels = np.all(image == np.array([255, 44, 190], dtype=np.uint8), axis=2)
        self.assertEqual(int(np.count_nonzero(orange_pixels)), 6)
        self.assertEqual(int(np.count_nonzero(blue_pixels)), 7)
        self.assertEqual(int(np.count_nonzero(pink_pixels)), 12)


if __name__ == "__main__":
    unittest.main()
