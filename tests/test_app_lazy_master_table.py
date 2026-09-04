from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from resources.analytics_engine import AnalyticsConfig
from resources.app import NpecLabelingMainWindow, _normalize_lazy_source_rgb
from resources.models import LabelClass


class LazyMasterTableSourceImageTests(unittest.TestCase):
    def test_normalize_lazy_source_rgb_preserves_and_resizes_source_image(self) -> None:
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        image[:, :, 0] = 20
        image[:, :, 1] = 140
        image[:, :, 2] = 210

        normalized = _normalize_lazy_source_rgb(image, (8, 12))

        self.assertEqual(normalized.shape, (8, 12, 3))
        self.assertGreater(int(normalized[:, :, 1].mean()), 100)
        self.assertGreater(int(normalized[:, :, 2].mean()), 180)
        self.assertLess(int(normalized[:, :, 0].mean()), 50)

    def test_lazy_master_table_uses_original_source_image_for_analytics_and_preview(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lazy_master_source_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            output_dir = tmp_dir / "segmentation_masks"
            output_dir.mkdir()

            image = np.zeros((12, 18, 3), dtype=np.uint8)
            image[:, :, 0] = 30
            image[:, :, 1] = 160
            image[:, :, 2] = 220
            image_path = tmp_dir / "frame_001.png"
            Image.fromarray(image).save(image_path)

            mask = np.zeros((12, 18), dtype=np.uint8)
            mask[2:10, 8:10] = 3
            mask[1:4, 7:12] = 2
            mask_path = output_dir / "frame_001_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)

            class FakeWindow:
                classes = [LabelClass(3, "Root", "#ff9f1c")]
                plant_track_expected_count = 1

                @staticmethod
                def _natural_sort_key(text: str):
                    return ((1, text),)

                @staticmethod
                def _build_analytics_config_from_ui() -> AnalyticsConfig:
                    return AnalyticsConfig(root_class_id=3, shoot_class_id=2, lateral_class_id=4, expected_track_count=1)

                @staticmethod
                def _read_index_mask_from_path(path: Path) -> np.ndarray | None:
                    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    return np.asarray(arr, dtype=np.uint8) if arr is not None else None

                @staticmethod
                def _decode_image_path(path: Path) -> np.ndarray | None:
                    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

            records = [
                {
                    "uid": "frame_001",
                    "image_name": "frame_001.png",
                    "image_path": str(image_path),
                    "output_mask": str(mask_path),
                    "meta": {
                        "series": "synthetic",
                        "petri": "plate_01",
                        "timestamp": "2026-07-07 09:00:00",
                        "frame_index": 1,
                        "relative_folder": ".",
                    },
                }
            ]
            captured: dict[str, object] = {}

            def fake_run_temporal_analytics(items, predictions, annotations, config, progress_callback=None):
                del annotations, config, progress_callback
                captured["image_mean"] = float(items[0].image.mean())
                captured["image_shape"] = tuple(int(v) for v in items[0].image.shape)
                captured["prediction_shape"] = tuple(int(v) for v in predictions["frame_001"].shape)
                return {
                    "rows": [
                        {
                            "frame_index": 0,
                            "plant_id": "plant_01",
                            "root_length_mm_raw": 1.25,
                            "total_root_length_mm_raw": 1.25,
                            "shoot_area_mm2": 0.5,
                        }
                    ]
                }

            with mock.patch("resources.app.run_temporal_analytics", side_effect=fake_run_temporal_analytics):
                master_df, master_csv, total_csv, warnings = NpecLabelingMainWindow._build_lazy_segmentation_master_table(
                    FakeWindow(),
                    tmp_dir,
                    output_dir,
                    records,
                )

            self.assertEqual(warnings, [])
            self.assertIsNotNone(master_df)
            assert master_df is not None
            self.assertIsNotNone(master_csv)
            self.assertIsNotNone(total_csv)
            self.assertEqual(captured["image_shape"], (12, 18, 3))
            self.assertEqual(captured["prediction_shape"], (12, 18))
            self.assertGreater(float(captured["image_mean"]), 100.0)
            row = master_df.iloc[0]
            self.assertEqual(str(row["SourceFile"]), str(image_path))
            self.assertEqual(str(row["OutputMaskPath"]), str(mask_path))
            self.assertEqual(str(row["PreviewImagePath"]), str(image_path))

    def test_lazy_master_table_enables_rgb_green_shoot_masks_for_ownership(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lazy_master_green_ownership_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            output_dir = tmp_dir / "segmentation_masks"
            output_dir.mkdir()

            image = np.full((100, 120, 3), 225, dtype=np.uint8)
            image[18:34, 46:62] = [72, 132, 38]
            image_path = tmp_dir / "frame_001.png"
            Image.fromarray(image).save(image_path)

            mask = np.zeros((100, 120), dtype=np.uint8)
            mask[20:80, 12] = 3
            mask[60:82, 70:94] = 2
            mask_path = output_dir / "frame_001_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)

            class FakeWindow:
                classes = [LabelClass(3, "Root", "#ff9f1c")]
                plant_track_expected_count = 1
                pipeline_lazy_shoot_rgb_rescue_enabled = True

                @staticmethod
                def _natural_sort_key(text: str):
                    return ((1, text),)

                @staticmethod
                def _build_analytics_config_from_ui() -> AnalyticsConfig:
                    return AnalyticsConfig(root_class_id=3, shoot_class_id=2, lateral_class_id=4, expected_track_count=1)

                @staticmethod
                def _read_index_mask_from_path(path: Path) -> np.ndarray | None:
                    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    return np.asarray(arr, dtype=np.uint8) if arr is not None else None

                @staticmethod
                def _decode_image_path(path: Path) -> np.ndarray | None:
                    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

            records = [
                {
                    "uid": "frame_001",
                    "image_name": "frame_001.png",
                    "image_path": str(image_path),
                    "output_mask": str(mask_path),
                    "meta": {
                        "series": "synthetic",
                        "petri": "plate_01",
                        "timestamp": "2026-07-07 09:00:00",
                        "frame_index": 1,
                        "relative_folder": ".",
                    },
                }
            ]
            captured: dict[str, object] = {}

            def fake_run_temporal_analytics(items, predictions, annotations, config, progress_callback=None):
                del items, predictions, annotations, progress_callback
                captured["green_only_enabled"] = bool(config.shoot_rgb_green_only_enabled)
                return {
                    "rows": [
                        {
                            "frame_index": 0,
                            "plant_id": "plant_01",
                            "root_length_mm_raw": 1.25,
                            "total_root_length_mm_raw": 1.25,
                            "shoot_area_mm2": 0.5,
                            "shoot_area_px": 50,
                        }
                    ]
                }

            with mock.patch("resources.app.run_temporal_analytics", side_effect=fake_run_temporal_analytics):
                master_df, _master_csv, _total_csv, warnings = NpecLabelingMainWindow._build_lazy_segmentation_master_table(
                    FakeWindow(),
                    tmp_dir,
                    output_dir,
                    records,
                )

            self.assertTrue(bool(captured["green_only_enabled"]))
            self.assertTrue(any("RGB green-only" in warning for warning in warnings))
            self.assertIsNotNone(master_df)
            assert master_df is not None
            self.assertTrue(bool(master_df.iloc[0]["ownership_shoot_rgb_green_only_enabled"]))

    def test_lazy_master_table_tracks_hades_shoot_area_and_exports_accepted_masks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lazy_master_temporal_shoot_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            output_dir = tmp_dir / "segmentation_masks"
            output_dir.mkdir()
            records: list[dict[str, object]] = []
            masks: list[np.ndarray] = []
            for frame_index in range(2):
                image = np.full((80, 80, 3), 225, dtype=np.uint8)
                image_path = tmp_dir / f"frame_{frame_index:03d}.png"
                Image.fromarray(image).save(image_path)

                mask = np.zeros((80, 80), dtype=np.uint8)
                mask[22:70, 39:42] = 1
                if frame_index == 0:
                    mask[10:22, 34:46] = 2
                else:
                    mask[14:20, 38:42] = 2
                mask_path = output_dir / f"frame_{frame_index:03d}_mask.png"
                Image.fromarray(mask, mode="L").save(mask_path)
                masks.append(mask)
                records.append(
                    {
                        "uid": f"frame_{frame_index:03d}",
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "output_mask": str(mask_path),
                        "meta": {
                            "series": "hades_synthetic",
                            "petri": "plate_01",
                            "timestamp": f"2026-07-07 09:0{frame_index}:00",
                            "frame_index": frame_index,
                            "relative_folder": ".",
                        },
                    }
                )

            class FakeWindow:
                classes = [LabelClass(1, "Root", "#ff9f1c")]
                plant_track_expected_count = 1

                @staticmethod
                def _natural_sort_key(text: str):
                    return ((1, text),)

                @staticmethod
                def _build_analytics_config_from_ui() -> AnalyticsConfig:
                    return AnalyticsConfig(
                        root_class_id=1,
                        shoot_class_id=2,
                        lateral_class_id=3,
                        expected_track_count=1,
                        pixel_size_mm=0.1,
                    )

                @staticmethod
                def _read_index_mask_from_path(path: Path) -> np.ndarray | None:
                    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    return np.asarray(arr, dtype=np.uint8) if arr is not None else None

                @staticmethod
                def _decode_image_path(path: Path) -> np.ndarray | None:
                    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

            def fake_run_temporal_analytics(items, predictions, annotations, config, progress_callback=None):
                del items, predictions, annotations, config, progress_callback
                return {
                    "rows": [
                        {
                            "frame_index": frame_index,
                            "plant_id": "plant_01",
                            "bbox_x": 39,
                            "bbox_y": 22,
                            "bbox_w": 3,
                            "bbox_h": 48,
                            "shoot_bbox_x": 34 if frame_index == 0 else 38,
                            "shoot_bbox_y": 10 if frame_index == 0 else 14,
                            "shoot_bbox_w": 12 if frame_index == 0 else 4,
                            "shoot_bbox_h": 12 if frame_index == 0 else 6,
                            "ownership_crown_center_x": 40.0,
                            "ownership_crown_center_y": 22.0,
                            "total_root_length_mm_raw": 4.8,
                            "root_length_mm_raw": 4.8,
                            "shoot_area_px": int(np.count_nonzero(masks[frame_index] == 2)),
                            "shoot_area_mm2": float(np.count_nonzero(masks[frame_index] == 2)) * 0.01,
                            "ownership_measurement_valid": True,
                        }
                        for frame_index in range(2)
                    ]
                }

            with mock.patch("resources.app.run_temporal_analytics", side_effect=fake_run_temporal_analytics):
                master_df, _master_csv, _total_csv, warnings = (
                    NpecLabelingMainWindow._build_lazy_segmentation_master_table(
                        FakeWindow(),
                        tmp_dir,
                        output_dir,
                        records,
                    )
                )

            self.assertEqual(warnings, [])
            self.assertIsNotNone(master_df)
            assert master_df is not None
            rows = master_df.sort_values("FrameIndex", kind="mergesort")
            areas = pd.to_numeric(rows["shoot_area_px"], errors="raise").astype(int).tolist()
            self.assertEqual(areas, [144, 144])
            self.assertTrue(bool(rows.iloc[1]["shoot_temporal_area_floor_applied"]))
            accepted_paths = [Path(value) for value in rows["OwnershipDisplayMaskPath"].astype(str)]
            self.assertTrue(all(path.is_file() for path in accepted_paths))
            accepted_last = np.asarray(Image.open(accepted_paths[-1]), dtype=np.uint8)
            self.assertEqual(int(np.count_nonzero(accepted_last == 2)), 144)

    def test_lazy_master_table_falls_back_when_source_image_shape_differs_from_mask(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lazy_master_mismatch_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            output_dir = tmp_dir / "segmentation_masks"
            output_dir.mkdir()

            image = np.full((20, 30, 3), 180, dtype=np.uint8)
            image_path = tmp_dir / "frame_001.png"
            Image.fromarray(image).save(image_path)

            mask = np.zeros((12, 18), dtype=np.uint8)
            mask[2:10, 8:10] = 3
            mask_path = output_dir / "frame_001_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)

            class FakeWindow:
                classes = [LabelClass(3, "Root", "#ff9f1c")]
                plant_track_expected_count = 1

                @staticmethod
                def _natural_sort_key(text: str):
                    return ((1, text),)

                @staticmethod
                def _build_analytics_config_from_ui() -> AnalyticsConfig:
                    return AnalyticsConfig(root_class_id=3, shoot_class_id=2, lateral_class_id=4, expected_track_count=1)

                @staticmethod
                def _read_index_mask_from_path(path: Path) -> np.ndarray | None:
                    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    return np.asarray(arr, dtype=np.uint8) if arr is not None else None

                @staticmethod
                def _decode_image_path(path: Path) -> np.ndarray | None:
                    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

            records = [
                {
                    "uid": "frame_001",
                    "image_name": "frame_001.png",
                    "image_path": str(image_path),
                    "output_mask": str(mask_path),
                    "meta": {"series": "synthetic", "petri": "plate_01", "frame_index": 1},
                }
            ]
            captured: dict[str, object] = {}

            def fake_run_temporal_analytics(items, predictions, annotations, config, progress_callback=None):
                del predictions, annotations, config, progress_callback
                captured["image_mean"] = float(items[0].image.mean())
                captured["image_shape"] = tuple(int(v) for v in items[0].image.shape)
                return {"rows": [{"frame_index": 0, "plant_id": "plant_01", "root_length_mm_raw": 1.0}]}

            with mock.patch("resources.app.run_temporal_analytics", side_effect=fake_run_temporal_analytics):
                master_df, _master_csv, _total_csv, warnings = NpecLabelingMainWindow._build_lazy_segmentation_master_table(
                    FakeWindow(),
                    tmp_dir,
                    output_dir,
                    records,
                )

            self.assertIsNotNone(master_df)
            self.assertEqual(captured["image_shape"], (12, 18, 3))
            self.assertEqual(float(captured["image_mean"]), 0.0)
            self.assertTrue(any("differs from mask size" in warning for warning in warnings))

    def test_lazy_master_table_timelapse_summary_exports_primary_lateral_shoot_and_deltas(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lazy_master_metrics_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            output_dir = tmp_dir / "segmentation_masks"
            output_dir.mkdir()
            records: list[dict[str, object]] = []
            for frame_index in range(2):
                image = np.full((12, 18, 3), 120 + (frame_index * 20), dtype=np.uint8)
                image_path = tmp_dir / f"frame_{frame_index:03d}.png"
                Image.fromarray(image).save(image_path)
                mask = np.zeros((12, 18), dtype=np.uint8)
                mask[2:10, 8 + frame_index : 10 + frame_index] = 3
                mask_path = output_dir / f"frame_{frame_index:03d}_mask.png"
                Image.fromarray(mask, mode="L").save(mask_path)
                records.append(
                    {
                        "uid": f"frame_{frame_index:03d}",
                        "image_name": f"frame_{frame_index:03d}.png",
                        "image_path": str(image_path),
                        "output_mask": str(mask_path),
                        "meta": {
                            "series": "synthetic",
                            "petri": "plate_01",
                            "timestamp": f"2026-07-07 09:0{frame_index}:00",
                            "frame_index": frame_index,
                            "relative_folder": ".",
                        },
                    }
                )

            class FakeWindow:
                classes = [LabelClass(3, "Root", "#ff9f1c")]
                plant_track_expected_count = 2

                @staticmethod
                def _natural_sort_key(text: str):
                    return ((1, text),)

                @staticmethod
                def _build_analytics_config_from_ui() -> AnalyticsConfig:
                    return AnalyticsConfig(root_class_id=3, shoot_class_id=2, lateral_class_id=4, expected_track_count=2)

                @staticmethod
                def _read_index_mask_from_path(path: Path) -> np.ndarray | None:
                    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    return np.asarray(arr, dtype=np.uint8) if arr is not None else None

                @staticmethod
                def _decode_image_path(path: Path) -> np.ndarray | None:
                    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

                @staticmethod
                def _current_pipeline_lazy_root_metric_mode() -> str:
                    return "lateral"

            def fake_run_temporal_analytics(items, predictions, annotations, config, progress_callback=None):
                del items, predictions, annotations, config, progress_callback
                return {
                    "rows": [
                        {
                            "frame_index": 0,
                            "plant_id": "plant_01",
                            "total_root_length_mm_raw": 10.0,
                            "total_root_length_weighted_mm_raw": 12.0,
                            "primary_root_length_mm_raw": 7.0,
                            "primary_root_length_weighted_mm_raw": 8.0,
                            "lateral_total_length_mm": 3.0,
                            "lateral_total_length_weighted_mm": 4.0,
                            "shoot_area_mm2": 1.0,
                            "shoot_area_px": 100,
                            "ownership_measurement_valid": True,
                        },
                        {
                            "frame_index": 0,
                            "plant_id": "plant_02",
                            "total_root_length_mm_raw": 20.0,
                            "total_root_length_weighted_mm_raw": 22.0,
                            "primary_root_length_mm_raw": 12.0,
                            "primary_root_length_weighted_mm_raw": 13.0,
                            "lateral_total_length_mm": 8.0,
                            "lateral_total_length_weighted_mm": 9.0,
                            "shoot_area_mm2": 2.0,
                            "shoot_area_px": 200,
                            "ownership_measurement_valid": True,
                        },
                        {
                            "frame_index": 1,
                            "plant_id": "plant_01",
                            "total_root_length_mm_raw": 15.0,
                            "total_root_length_weighted_mm_raw": 18.0,
                            "primary_root_length_mm_raw": 9.0,
                            "primary_root_length_weighted_mm_raw": 10.0,
                            "lateral_total_length_mm": 6.0,
                            "lateral_total_length_weighted_mm": 8.0,
                            "shoot_area_mm2": 1.5,
                            "shoot_area_px": 150,
                            "ownership_measurement_valid": True,
                        },
                        {
                            "frame_index": 1,
                            "plant_id": "plant_02",
                            "total_root_length_mm_raw": 25.0,
                            "total_root_length_weighted_mm_raw": 30.0,
                            "primary_root_length_mm_raw": 14.0,
                            "primary_root_length_weighted_mm_raw": 17.0,
                            "lateral_total_length_mm": 11.0,
                            "lateral_total_length_weighted_mm": 13.0,
                            "shoot_area_mm2": 2.5,
                            "shoot_area_px": 250,
                            "ownership_measurement_valid": True,
                        },
                    ]
                }

            with mock.patch("resources.app.run_temporal_analytics", side_effect=fake_run_temporal_analytics):
                _master_df, _master_csv, total_csv, warnings = NpecLabelingMainWindow._build_lazy_segmentation_master_table(
                    FakeWindow(),
                    tmp_dir,
                    output_dir,
                    records,
                )

            self.assertEqual(warnings, [])
            self.assertIsNotNone(total_csv)
            assert total_csv is not None
            summary = pd.read_csv(total_csv)
            self.assertEqual(len(summary), 2)
            frame1 = summary.sort_values("FrameIndex").iloc[1]
            self.assertAlmostEqual(float(frame1["total_root_length_mm"]), 40.0)
            self.assertAlmostEqual(float(frame1["total_root_length_weighted_mm"]), 48.0)
            self.assertAlmostEqual(float(frame1["total_primary_root_length_weighted_mm"]), 27.0)
            self.assertAlmostEqual(float(frame1["total_lateral_root_length_weighted_mm"]), 21.0)
            self.assertAlmostEqual(float(frame1["total_primary_root_length_mm"]), 23.0)
            self.assertAlmostEqual(float(frame1["total_lateral_root_length_mm"]), 17.0)
            self.assertAlmostEqual(float(frame1["primary_plus_lateral_root_length_mm"]), 40.0)
            self.assertAlmostEqual(float(frame1["total_shoot_area_mm2"]), 4.0)
            self.assertAlmostEqual(float(frame1["total_shoot_area_px"]), 400.0)
            self.assertAlmostEqual(float(frame1["delta_total_root_length_mm"]), 10.0)
            self.assertAlmostEqual(float(frame1["delta_total_root_length_weighted_mm"]), 14.0)
            self.assertAlmostEqual(float(frame1["delta_total_primary_root_length_mm"]), 4.0)
            self.assertAlmostEqual(float(frame1["delta_total_primary_root_length_weighted_mm"]), 6.0)
            self.assertAlmostEqual(float(frame1["delta_total_lateral_root_length_mm"]), 6.0)
            self.assertAlmostEqual(float(frame1["delta_total_lateral_root_length_weighted_mm"]), 8.0)
            self.assertAlmostEqual(float(frame1["delta_total_shoot_area_mm2"]), 1.0)
            self.assertAlmostEqual(float(frame1["delta_total_shoot_area_px"]), 100.0)
            self.assertAlmostEqual(float(frame1["primary_fraction_of_root_length"]), 23.0 / 40.0)
            self.assertAlmostEqual(float(frame1["lateral_fraction_of_root_length"]), 17.0 / 40.0)
            self.assertAlmostEqual(float(frame1["root_length_mm_per_shoot_area_mm2"]), 17.0 / 4.0)
            self.assertEqual(str(frame1["selected_root_metric_mode"]), "lateral")
            self.assertEqual(str(frame1["selected_root_metric_column"]), "total_lateral_root_length_mm")
            self.assertAlmostEqual(float(frame1["selected_root_length_mm"]), 17.0)
            self.assertAlmostEqual(float(frame1["delta_selected_root_length_mm"]), 6.0)
            self.assertAlmostEqual(float(frame1["plate_mask_total_root_length_mm"]), 40.0)
            self.assertAlmostEqual(float(frame1["plate_mask_primary_root_length_mm"]), 23.0)
            self.assertAlmostEqual(float(frame1["plate_mask_lateral_root_length_mm"]), 17.0)
            self.assertAlmostEqual(float(frame1["delta_plate_mask_total_root_length_mm"]), 10.0)
            self.assertTrue(bool(frame1["plate_mask_root_totals_ownership_independent"]))
            self.assertTrue((output_dir / "npec_plate_mask_root_totals_timelapse.csv").is_file())

            class FakeWeightedWindow(FakeWindow):
                @staticmethod
                def _current_pipeline_lazy_root_metric_mode() -> str:
                    return "weighted_total"

            with mock.patch("resources.app.run_temporal_analytics", side_effect=fake_run_temporal_analytics):
                _master_df, _master_csv, weighted_total_csv, warnings = NpecLabelingMainWindow._build_lazy_segmentation_master_table(
                    FakeWeightedWindow(),
                    tmp_dir,
                    output_dir,
                    records,
                )

            self.assertEqual(warnings, [])
            self.assertIsNotNone(weighted_total_csv)
            assert weighted_total_csv is not None
            weighted_summary = pd.read_csv(weighted_total_csv)
            weighted_frame1 = weighted_summary.sort_values("FrameIndex").iloc[1]
            self.assertEqual(str(weighted_frame1["selected_root_metric_mode"]), "weighted_total")
            self.assertEqual(str(weighted_frame1["selected_root_metric_column"]), "total_root_length_weighted_mm")
            self.assertAlmostEqual(float(weighted_frame1["selected_root_length_mm"]), 48.0)
            self.assertAlmostEqual(float(weighted_frame1["delta_selected_root_length_mm"]), 14.0)


if __name__ == "__main__":
    unittest.main()
