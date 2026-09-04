from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from resources.mask_total_root_analysis import (
    build_mask_total_dataframe,
    build_mask_total_class_coverage_metadata,
    build_mask_total_frame_review_dataframe,
    build_mask_total_pmi_dataframe,
    build_mask_total_timelapse_dataframe,
    parse_class_id_text,
    resolve_lazy_ownership_class_selection,
    write_mask_total_all_metrics_workbook,
    write_mask_total_outputs,
)
from resources.pmi_export import build_pmi_style_rows


class MaskTotalRootAnalysisTests(unittest.TestCase):
    def test_parse_class_ids_uses_unique_positive_ids(self) -> None:
        self.assertEqual(parse_class_id_text("1, 3; 3 bad 0 256 2"), [1, 3, 2])
        self.assertEqual(parse_class_id_text("", fallback=(4, 5)), [4, 5])

    def test_resolve_lazy_ownership_class_selection_uses_first_root_lateral_and_shoot(self) -> None:
        selection = resolve_lazy_ownership_class_selection("4,5,7", "6,8")

        self.assertEqual(selection.root_class_id, 4)
        self.assertEqual(selection.lateral_class_id, 5)
        self.assertEqual(selection.shoot_class_id, 6)
        self.assertEqual(selection.ignored_root_class_ids, (7,))
        self.assertEqual(selection.ignored_shoot_class_ids, (8,))

    def test_resolve_lazy_ownership_class_selection_falls_back_to_defaults(self) -> None:
        selection = resolve_lazy_ownership_class_selection("", "")

        self.assertEqual(selection.root_class_id, 1)
        self.assertEqual(selection.lateral_class_id, 3)
        self.assertEqual(selection.shoot_class_id, 2)
        self.assertEqual(selection.ignored_root_class_ids, ())
        self.assertEqual(selection.ignored_shoot_class_ids, ())

    def test_build_mask_total_dataframe_combines_selected_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "plate_01.png"
            Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8)).save(image_path)

            mask = np.zeros((24, 24), dtype=np.uint8)
            mask[2:8, 4] = 1
            mask[12, 10:15] = 3
            mask[18:23, 18] = 2
            mask_path = tmp_dir / "plate_01_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)

            df = build_mask_total_dataframe(
                [
                    {
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "output_mask": str(mask_path),
                        "meta": {
                            "series": "synthetic",
                            "petri": "plate_01",
                            "timestamp": "2026-07-01 08:00:00",
                            "frame_index": 0,
                            "relative_folder": ".",
                        },
                        "details": {"pixel_size_mm": 0.5},
                    }
                ],
                class_ids=[1, 3],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
            )

            self.assertEqual(len(df), 1)
            row = df.iloc[0]
            self.assertEqual(int(row["root_pixels_class_1"]), 6)
            self.assertEqual(int(row["lateral_pixels_class_3"]), 5)
            self.assertEqual(int(row["shoot_seed_pixels_class_2"]), 5)
            self.assertEqual(int(row["shoot_area_px"]), 5)
            self.assertAlmostEqual(float(row["shoot_area_mm2"]), 5 * (0.5 ** 2))
            self.assertEqual(int(row["measured_mask_pixels"]), 11)
            self.assertEqual(str(row["PreviewImagePath"]), str(image_path))
            self.assertEqual(float(row["pixel_size_mm"]), 0.5)
            self.assertEqual(int(row["class_1_pixels"]), 6)
            self.assertEqual(int(row["class_3_pixels"]), 5)
            self.assertGreater(float(row["class_1_length_mm"]), 0.0)
            self.assertGreater(float(row["class_3_length_mm"]), 0.0)
            self.assertGreater(float(row["primary_root_length_mm"]), 0.0)
            self.assertGreater(float(row["lateral_root_length_mm"]), 0.0)
            self.assertAlmostEqual(float(row["primary_root_length_mm"]), float(row["primary_root_length_px"]) * 0.5)
            self.assertAlmostEqual(float(row["lateral_root_length_mm"]), float(row["lateral_root_length_px"]) * 0.5)
            self.assertAlmostEqual(float(row["total_root_length_mm"]), float(row["total_root_length_px"]) * 0.5)

            detail_csv, summary_csv, compat_csv, metadata_json = write_mask_total_outputs(
                df,
                tmp_dir,
                class_ids=[1, 3],
                root_metric_mode="lateral",
            )
            self.assertTrue(detail_csv.exists())
            self.assertEqual(detail_csv.name, "npec_total_root_and_shoot_from_masks.csv")
            self.assertTrue(summary_csv.exists())
            self.assertIsNotNone(compat_csv)
            self.assertTrue(compat_csv.exists())
            self.assertTrue(metadata_json.exists())
            self.assertTrue((tmp_dir / "npec_mask_total_frames_needing_review.csv").exists())
            self.assertTrue((tmp_dir / "npec_total_root_length_from_masks.csv").exists())
            compat = pd.read_csv(compat_csv)
            self.assertEqual(compat["analysis_mode"].tolist(), ["mask_total"])
            self.assertEqual(compat["selected_root_metric_mode"].tolist(), ["lateral"])
            self.assertEqual(compat["selected_root_metric_column"].tolist(), ["lateral_root_length_mm"])
            self.assertAlmostEqual(float(compat["selected_root_length_mm"].iloc[0]), float(row["lateral_root_length_mm"]))
            self.assertIn("shoot_area_mm2", compat.columns)
            self.assertIn("primary_root_length_mm", compat.columns)
            self.assertIn("lateral_root_length_mm", compat.columns)
            self.assertTrue((tmp_dir / "npec_total_root_and_shoot_timelapse.csv").exists())
            summary = pd.read_csv(summary_csv)
            self.assertIn("final_primary_root_length_mm", summary.columns)
            self.assertIn("final_lateral_root_length_mm", summary.columns)
            self.assertIn("frames_with_selected_root_pixels", summary.columns)
            self.assertIn("frames_with_shoot_model_pixels", summary.columns)
            metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
            self.assertIn("frame_review_csv", metadata)
            self.assertIn("frames_needing_review", metadata)

    def test_build_mask_total_dataframe_reports_record_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            records: list[dict[str, object]] = []
            for index in range(2):
                image_path = tmp_dir / f"plate_{index + 1}.png"
                Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(image_path)
                mask = np.zeros((12, 12), dtype=np.uint8)
                mask[2:8, 4] = 1
                mask_path = tmp_dir / f"plate_{index + 1}_mask.png"
                Image.fromarray(mask, mode="L").save(mask_path)
                records.append(
                    {
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "output_mask": str(mask_path),
                        "meta": {
                            "series": "synthetic",
                            "petri": f"plate_{index + 1}",
                            "frame_index": index,
                            "relative_folder": ".",
                        },
                        "details": {"pixel_size_mm": 0.5},
                    }
                )

            progress_calls: list[tuple[int, int, str]] = []
            df = build_mask_total_dataframe(
                records,
                class_ids=[1, 3],
                shoot_class_ids=[2],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
                progress_callback=lambda done, total, label: progress_calls.append((done, total, label)) or True,
            )

            self.assertEqual(len(df), 2)
            self.assertEqual(progress_calls, [(1, 2, "plate_1.png"), (2, 2, "plate_2.png")])

    def test_mask_total_class_coverage_reports_missing_selected_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "plate_missing_classes.png"
            Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(image_path)

            mask = np.zeros((20, 20), dtype=np.uint8)
            mask[2:10, 4] = 1
            mask_path = tmp_dir / "plate_missing_classes_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)

            df = build_mask_total_dataframe(
                [
                    {
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "output_mask": str(mask_path),
                        "meta": {
                            "series": "synthetic",
                            "petri": "plate_missing",
                            "timestamp": "2026-07-01 08:00:00",
                            "frame_index": 0,
                            "relative_folder": ".",
                        },
                        "details": {"pixel_size_mm": 0.5},
                    }
                ],
                class_ids=[1, 3],
                shoot_class_ids=[2],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
            )

            coverage = build_mask_total_class_coverage_metadata(df, [1, 3], [2])
            self.assertEqual(coverage["root_class_total_pixels"], {"1": 8, "3": 0})
            self.assertEqual(coverage["root_class_frame_counts"], {"1": 1, "3": 0})
            self.assertEqual(coverage["missing_root_class_ids"], [3])
            self.assertEqual(coverage["missing_shoot_class_ids"], [2])
            self.assertEqual(int(coverage["frames_missing_lateral_root_pixels"]), 1)
            self.assertEqual(int(coverage["frames_missing_model_shoot_pixels"]), 1)

            _detail_csv, summary_csv, _compat_csv, metadata_json = write_mask_total_outputs(
                df,
                tmp_dir,
                class_ids=[1, 3],
                shoot_class_ids=[2],
            )
            metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
            self.assertEqual(metadata["class_coverage"]["missing_root_class_ids"], [3])
            self.assertEqual(metadata["class_coverage"]["missing_shoot_class_ids"], [2])
            self.assertGreaterEqual(len(metadata["class_coverage_warnings"]), 2)
            self.assertTrue(any("Selected root classes absent" in warning for warning in metadata["class_coverage_warnings"]))
            self.assertTrue(any("Selected shoot classes absent" in warning for warning in metadata["class_coverage_warnings"]))
            summary = pd.read_csv(summary_csv)
            self.assertEqual(int(summary["frames_with_selected_root_pixels"].iloc[0]), 1)
            self.assertEqual(int(summary["frames_with_lateral_root_pixels"].iloc[0]), 0)
            self.assertEqual(int(summary["frames_with_shoot_model_pixels"].iloc[0]), 0)

    def test_timelapse_dataframe_exports_selected_metric_and_root_shoot_deltas(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-01 08:00:00",
                    "FrameIndex": 0,
                    "total_root_length_mm": 10.0,
                    "total_root_length_weighted_mm": 11.0,
                    "primary_root_length_mm": 7.0,
                    "lateral_root_length_mm": 3.0,
                    "shoot_area_mm2": 2.0,
                    "plants_detected": 1,
                    "analysis_mode": "mask_total",
                },
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-02 08:00:00",
                    "FrameIndex": 1,
                    "total_root_length_mm": 16.0,
                    "total_root_length_weighted_mm": 17.0,
                    "primary_root_length_mm": 9.0,
                    "lateral_root_length_mm": 7.0,
                    "shoot_area_mm2": 4.0,
                    "plants_detected": 1,
                    "analysis_mode": "mask_total",
                },
            ]
        )

        timeline = build_mask_total_timelapse_dataframe(df, root_metric_mode="primary")

        self.assertEqual(timeline["selected_root_metric_mode"].tolist(), ["primary", "primary"])
        self.assertEqual(timeline["selected_root_metric_column"].tolist(), ["primary_root_length_mm", "primary_root_length_mm"])
        self.assertAlmostEqual(float(timeline["selected_root_length_mm"].iloc[1]), 9.0)
        self.assertAlmostEqual(float(timeline["delta_selected_root_length_mm"].iloc[1]), 2.0)
        self.assertAlmostEqual(float(timeline["primary_plus_lateral_root_length_mm"].iloc[1]), 16.0)
        self.assertAlmostEqual(float(timeline["delta_primary_plus_lateral_root_length_mm"].iloc[1]), 6.0)
        self.assertAlmostEqual(float(timeline["primary_fraction_of_root_length"].iloc[1]), 9.0 / 16.0)
        self.assertAlmostEqual(float(timeline["lateral_fraction_of_root_length"].iloc[1]), 7.0 / 16.0)
        self.assertAlmostEqual(float(timeline["selected_root_length_mm_per_shoot_area_mm2"].iloc[1]), 9.0 / 4.0)

    def test_build_mask_total_dataframe_uses_configurable_shoot_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "plate_custom_shoot.png"
            Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(image_path)

            mask = np.zeros((20, 20), dtype=np.uint8)
            mask[2:10, 6] = 1
            mask[3:7, 12] = 2
            mask[10:16, 12] = 4
            mask_path = tmp_dir / "plate_custom_shoot_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)

            df = build_mask_total_dataframe(
                [
                    {
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "output_mask": str(mask_path),
                        "meta": {
                            "series": "synthetic",
                            "petri": "plate_custom",
                            "timestamp": "2026-07-01 08:00:00",
                            "frame_index": 0,
                            "relative_folder": ".",
                        },
                        "details": {"pixel_size_mm": 0.25},
                    }
                ],
                class_ids=[1, 3],
                shoot_class_ids=[4],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
            )

            row = df.iloc[0]
            self.assertEqual(str(row["shoot_class_ids"]), "4")
            self.assertEqual(int(row["shoot_pixels_class_2"]), 4)
            self.assertEqual(int(row["shoot_pixels_selected_classes"]), 6)
            self.assertEqual(int(row["shoot_area_px"]), 6)
            self.assertAlmostEqual(float(row["shoot_area_mm2"]), 6 * (0.25 ** 2))
            self.assertEqual(int(row["shoot_class_4_pixels"]), 6)

            detail_csv, _summary_csv, _compat_csv, metadata_json = write_mask_total_outputs(
                df,
                tmp_dir,
                class_ids=[1, 3],
                shoot_class_ids=[4],
            )
            detail = pd.read_csv(detail_csv)
            self.assertEqual(detail["shoot_class_ids"].tolist(), [4])
            metadata = metadata_json.read_text(encoding="utf-8")
            self.assertIn('"shoot_class_ids": [', metadata)
            self.assertIn("4", metadata)

            pmi_source = build_mask_total_pmi_dataframe(df, shoot_class_ids=[4])
            self.assertEqual(pmi_source["shoot_class_id"].tolist(), [4])
            self.assertEqual(pmi_source["n_pixels_shoot"].tolist(), [6])
            workbook_path = write_mask_total_all_metrics_workbook(
                df,
                tmp_dir,
                class_ids=[1, 3],
                shoot_class_ids=[4],
                root_metric_mode="total",
                pmi_df=pmi_source,
                metadata_json_path=metadata_json,
            )
            self.assertIsNotNone(workbook_path)
            self.assertTrue(workbook_path.exists())
            with zipfile.ZipFile(workbook_path, "r") as zf:
                workbook_xml = zf.read("xl/workbook.xml").decode("utf-8")
            for sheet_name in ("detail", "summary", "timelapse", "mask_total_frame_review", "run_config", "pmi_style"):
                self.assertIn(sheet_name, workbook_xml)
            metadata_payload = metadata_json.read_text(encoding="utf-8")
            self.assertIn("all_metrics_workbook", metadata_payload)
            self.assertIn(workbook_path.name, metadata_payload)

    def test_rgb_green_shoot_rescue_adds_shoot_measurement_without_changing_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "plate_green_shoot.png"
            image = np.full((100, 100, 3), 235, dtype=np.uint8)
            image[20:36, 42:58] = [72, 132, 38]
            Image.fromarray(image).save(image_path)

            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[20:80, 12] = 1
            mask_path = tmp_dir / "plate_green_shoot_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)
            records = [
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "output_mask": str(mask_path),
                    "meta": {
                        "series": "synthetic",
                        "petri": "plate_green",
                        "timestamp": "2026-07-01 08:00:00",
                        "frame_index": 0,
                        "relative_folder": ".",
                    },
                    "details": {"pixel_size_mm": 0.1},
                }
            ]

            class_only = build_mask_total_dataframe(
                records,
                class_ids=[1, 3],
                shoot_class_ids=[2],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
            )
            rescued = build_mask_total_dataframe(
                records,
                class_ids=[1, 3],
                shoot_class_ids=[2],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
                shoot_rgb_rescue=True,
            )

            plain_row = class_only.iloc[0]
            rescued_row = rescued.iloc[0]
            self.assertEqual(int(plain_row["shoot_area_px"]), 0)
            self.assertEqual(int(rescued_row["shoot_area_model_px"]), 0)
            self.assertGreater(int(rescued_row["shoot_rgb_rescue_added_px"]), 0)
            self.assertEqual(int(rescued_row["shoot_rgb_rescue_removed_model_px"]), 0)
            self.assertEqual(int(rescued_row["shoot_area_px"]), int(rescued_row["shoot_rgb_rescue_added_px"]))
            self.assertEqual(str(rescued_row["shoot_measurement_source"]), "rgb_green_only")

            frame_review = build_mask_total_frame_review_dataframe(rescued)
            self.assertEqual(len(frame_review), 0)
            self.assertAlmostEqual(float(rescued_row["total_root_length_mm"]), float(plain_row["total_root_length_mm"]))
            self.assertEqual(int(rescued_row["total_root_pixels_selected_classes"]), int(plain_row["total_root_pixels_selected_classes"]))

            _detail_csv, summary_csv, _compat_csv, metadata_json = write_mask_total_outputs(
                rescued,
                tmp_dir,
                class_ids=[1, 3],
                shoot_class_ids=[2],
            )
            metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
            self.assertTrue(bool(metadata["shoot_rgb_rescue_enabled"]))
            self.assertGreater(int(metadata["total_shoot_rgb_rescue_added_px"]), 0)
            summary = pd.read_csv(summary_csv)
            self.assertGreater(int(summary["total_shoot_rgb_rescue_added_px"].iloc[0]), 0)

    def test_rgb_green_shoot_rescue_removes_non_green_model_shoot_colony(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "plate_gray_colony.png"
            image = np.full((100, 100, 3), 235, dtype=np.uint8)
            image[20:36, 42:58] = [72, 132, 38]
            image[64:88, 64:88] = [118, 126, 96]
            Image.fromarray(image).save(image_path)

            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[20:80, 12] = 1
            mask[64:88, 64:88] = 2
            mask_path = tmp_dir / "plate_gray_colony_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)
            records = [
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "output_mask": str(mask_path),
                    "meta": {
                        "series": "synthetic",
                        "petri": "plate_gray_colony",
                        "timestamp": "2026-07-01 08:00:00",
                        "frame_index": 0,
                        "relative_folder": ".",
                    },
                    "details": {"pixel_size_mm": 0.1},
                }
            ]

            class_only = build_mask_total_dataframe(
                records,
                class_ids=[1, 3],
                shoot_class_ids=[2],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
            )
            rescued = build_mask_total_dataframe(
                records,
                class_ids=[1, 3],
                shoot_class_ids=[2],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
                shoot_rgb_rescue=True,
            )

            plain_row = class_only.iloc[0]
            rescued_row = rescued.iloc[0]
            self.assertEqual(int(plain_row["shoot_area_px"]), 24 * 24)
            self.assertEqual(int(rescued_row["shoot_area_model_px"]), 24 * 24)
            self.assertGreater(int(rescued_row["shoot_rgb_rescue_added_px"]), 0)
            self.assertEqual(int(rescued_row["shoot_rgb_rescue_removed_model_px"]), 24 * 24)
            self.assertEqual(int(rescued_row["shoot_area_px"]), int(rescued_row["shoot_rgb_rescue_green_only_px"]))
            self.assertLess(int(rescued_row["shoot_area_px"]), int(rescued_row["shoot_area_model_px"]))
            self.assertEqual(str(rescued_row["shoot_measurement_source"]), "rgb_green_only")

            frame_review = build_mask_total_frame_review_dataframe(rescued)
            self.assertEqual(len(frame_review), 0)

    def test_rgb_green_shoot_rescue_rejects_lower_gray_green_colony(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "plate_lower_colony.png"
            image = np.full((100, 100, 3), 235, dtype=np.uint8)
            image[20:36, 42:58] = [72, 132, 38]
            cv2.ellipse(image, (68, 56), (14, 8), 0, 0, 360, (95, 104, 48), -1)
            Image.fromarray(image).save(image_path)

            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[20:80, 12] = 1
            cv2.ellipse(mask, (68, 56), (14, 8), 0, 0, 360, 2, -1)
            mask_path = tmp_dir / "plate_lower_colony_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)
            records = [
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "output_mask": str(mask_path),
                    "meta": {
                        "series": "synthetic",
                        "petri": "plate_lower_colony",
                        "timestamp": "2026-07-01 08:00:00",
                        "frame_index": 0,
                        "relative_folder": ".",
                    },
                    "details": {"pixel_size_mm": 0.1},
                }
            ]

            rescued = build_mask_total_dataframe(
                records,
                class_ids=[1, 3],
                shoot_class_ids=[2],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
                shoot_rgb_rescue=True,
            )

            row = rescued.iloc[0]
            self.assertGreater(int(row["shoot_rgb_rescue_added_px"]), 0)
            self.assertGreater(int(row["shoot_rgb_rescue_removed_model_px"]), 0)
            self.assertLess(int(row["shoot_area_px"]), int(row["shoot_area_model_px"]))
            self.assertLess(int(row["shoot_area_px"]), 400)
            self.assertEqual(str(row["shoot_measurement_source"]), "rgb_green_only")

    def test_rgb_green_shoot_rescue_retains_model_when_source_image_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "missing_lucifer_source.png"

            mask = np.zeros((60, 80), dtype=np.uint8)
            mask[10:52, 30] = 1
            mask[36:50, 44:60] = 2
            mask_path = tmp_dir / "missing_lucifer_source_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)

            rescued = build_mask_total_dataframe(
                [
                    {
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "output_mask": str(mask_path),
                        "meta": {
                            "series": "synthetic",
                            "petri": "missing_lucifer_source",
                            "timestamp": "2026-07-01 08:00:00",
                            "frame_index": 0,
                            "relative_folder": ".",
                        },
                        "details": {"pixel_size_mm": 0.1},
                    }
                ],
                class_ids=[1, 3],
                shoot_class_ids=[2],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
                shoot_rgb_rescue=True,
            )

            row = rescued.iloc[0]
            self.assertEqual(int(row["shoot_area_model_px"]), 14 * 16)
            self.assertEqual(int(row["shoot_rgb_rescue_removed_model_px"]), 0)
            self.assertEqual(int(row["shoot_area_px"]), 14 * 16)
            self.assertTrue(bool(row["shoot_rgb_rescue_abstained"]))
            self.assertEqual(str(row["shoot_rgb_rescue_fallback_status"]), "retained_model_mask")
            self.assertEqual(str(row["shoot_measurement_source"]), "rgb_green_only")

            frame_review = build_mask_total_frame_review_dataframe(rescued)
            self.assertEqual(len(frame_review), 1)
            self.assertEqual(str(frame_review.iloc[0]["mask_total_review_status"]), "review")
            self.assertIn("detector abstained", str(frame_review.iloc[0]["mask_total_review_reasons"]))
            self.assertEqual(float(frame_review.iloc[0]["shoot_model_rejected_fraction"]), 0.0)

    def test_selected_root_classes_drive_primary_lateral_and_pmi_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "plate_custom_roots.png"
            Image.fromarray(np.zeros((28, 28, 3), dtype=np.uint8)).save(image_path)

            mask = np.zeros((28, 28), dtype=np.uint8)
            mask[3:13, 5] = 4
            mask[16, 8:16] = 5
            mask[3:7, 20:24] = 6
            mask_path = tmp_dir / "plate_custom_roots_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)

            df = build_mask_total_dataframe(
                [
                    {
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "output_mask": str(mask_path),
                        "meta": {
                            "series": "synthetic",
                            "petri": "plate_custom_roots",
                            "timestamp": "2026-07-01 08:00:00",
                            "frame_index": 0,
                            "relative_folder": ".",
                        },
                        "details": {"pixel_size_mm": 0.4},
                    }
                ],
                class_ids=[4, 5],
                shoot_class_ids=[6],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
            )

            row = df.iloc[0]
            self.assertEqual(str(row["measurement_class_ids"]), "4,5")
            self.assertEqual(int(row["primary_root_class_id"]), 4)
            self.assertEqual(int(row["lateral_root_class_id"]), 5)
            self.assertEqual(int(row["root_pixels_class_1"]), 0)
            self.assertEqual(int(row["lateral_pixels_class_3"]), 0)
            self.assertEqual(int(row["total_root_pixels_1_plus_3"]), 0)
            self.assertEqual(int(row["primary_root_pixels_selected_class"]), 10)
            self.assertEqual(int(row["lateral_root_pixels_selected_class"]), 8)
            self.assertEqual(int(row["total_root_pixels_selected_classes"]), 18)
            self.assertEqual(int(row["shoot_pixels_selected_classes"]), 16)
            self.assertAlmostEqual(float(row["primary_root_length_mm"]), float(row["class_4_length_mm"]))
            self.assertAlmostEqual(float(row["lateral_root_length_mm"]), float(row["class_5_length_mm"]))
            self.assertAlmostEqual(float(row["root_to_shoot_area_ratio_px"]), 18 / 16)

            pmi_source = build_mask_total_pmi_dataframe(df, shoot_class_ids=[6])
            self.assertEqual(pmi_source["root_class_id"].tolist(), [4])
            self.assertEqual(pmi_source["lateral_class_id"].tolist(), [5])
            self.assertEqual(pmi_source["shoot_class_id"].tolist(), [6])
            self.assertEqual(pmi_source["n_pixels_main_root"].tolist(), [10])
            self.assertEqual(pmi_source["n_pixels_lateral_root"].tolist(), [8])
            self.assertEqual(pmi_source["n_pixels_dilated_root"].tolist(), [18])
            self.assertEqual(pmi_source["root_area_px"].tolist(), [18])

            _detail_csv, summary_csv, _compat_csv, metadata_json = write_mask_total_outputs(
                df,
                tmp_dir,
                class_ids=[4, 5],
                shoot_class_ids=[6],
            )
            summary = pd.read_csv(summary_csv)
            self.assertEqual(summary["primary_root_class_id"].tolist(), [4])
            self.assertEqual(summary["lateral_root_class_id"].tolist(), [5])
            self.assertEqual(summary["final_total_root_pixels_selected_classes"].tolist(), [18])
            metadata = metadata_json.read_text(encoding="utf-8")
            self.assertIn('"primary_root_class_id": 4', metadata)
            self.assertIn('"lateral_root_class_id": 5', metadata)

    def test_mask_total_pmi_adapter_exports_primary_lateral_total_and_shoot_traits(self) -> None:
        mask_df = pd.DataFrame(
            [
                {
                    "Series": "lucifer",
                    "PetriDish": "plate_01",
                    "FrameIndex": 3,
                    "pixel_size_mm": 0.2,
                    "root_pixels_class_1": 100,
                    "lateral_pixels_class_3": 40,
                    "total_root_pixels_1_plus_3": 140,
                    "primary_root_length_px": 80.0,
                    "primary_root_length_mm": 16.0,
                    "lateral_root_length_px": 35.0,
                    "lateral_root_length_mm": 7.0,
                    "total_root_length_px": 115.0,
                    "total_root_length_mm": 23.0,
                    "shoot_area_px": 50,
                    "shoot_area_mm2": 2.0,
                }
            ]
        )

        pmi_source = build_mask_total_pmi_dataframe(mask_df)
        pmi_rows = build_pmi_style_rows(
            pmi_source,
            include_fluorescence_placeholders=False,
            derive_mask_metrics=False,
        )

        def value_for(parameter: str) -> float:
            row = pmi_rows[pmi_rows["parameter"] == parameter]
            self.assertEqual(len(row), 1, parameter)
            return float(row.iloc[0]["value"])

        self.assertEqual(pmi_rows["plant_id"].unique().tolist(), ["plate_total"])
        self.assertAlmostEqual(value_for("primary_root_length_mm_clean"), 16.0)
        self.assertAlmostEqual(value_for("lateral_total_length_mm"), 7.0)
        self.assertAlmostEqual(value_for("total_root_length_mm_clean"), 23.0)
        self.assertAlmostEqual(value_for("combined_root_length_mm"), 23.0)
        self.assertAlmostEqual(value_for("shoot_area_mm2"), 2.0)
        self.assertAlmostEqual(value_for("n_pixels_main_root"), 100.0)
        self.assertAlmostEqual(value_for("n_pixels_lateral_root"), 40.0)
        self.assertAlmostEqual(value_for("n_pixels_shoot"), 50.0)
        self.assertAlmostEqual(value_for("primary_root_area_mm2"), 100 * (0.2 ** 2))
        self.assertAlmostEqual(value_for("total_root_area_mm2"), 140 * (0.2 ** 2))

    def test_summary_keeps_duplicate_petri_names_separate_by_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            records: list[dict[str, object]] = []
            for series, x in (("series_a", 4), ("series_b", 12)):
                image_path = tmp_dir / f"{series}_plate_01.png"
                Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(image_path)
                mask = np.zeros((20, 20), dtype=np.uint8)
                mask[2:10, x] = 1
                mask[12:16, x] = 2
                mask_path = tmp_dir / f"{series}_plate_01_mask.png"
                Image.fromarray(mask, mode="L").save(mask_path)
                records.append(
                    {
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "output_mask": str(mask_path),
                        "meta": {
                            "series": series,
                            "petri": "plate_01",
                            "timestamp": "2026-07-01 08:00:00",
                            "frame_index": 0,
                            "relative_folder": ".",
                        },
                        "details": {"pixel_size_mm": 1.0},
                    }
                )

            df = build_mask_total_dataframe(
                records,
                class_ids=[1, 3],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
            )
            _detail_csv, summary_csv, _compat_csv, _metadata_json = write_mask_total_outputs(
                df,
                tmp_dir,
                class_ids=[1, 3],
            )

            summary = pd.read_csv(summary_csv)
            self.assertEqual(summary["Series"].tolist(), ["series_a", "series_b"])
            self.assertEqual(summary["PetriDish"].tolist(), ["plate_01", "plate_01"])


if __name__ == "__main__":
    unittest.main()
