from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import pandas as pd

from resources.lazy_ownership_export import (
    build_lazy_ownership_detail_dataframe,
    build_lazy_ownership_frame_review_dataframe,
    build_lazy_ownership_review_dataframe,
    build_lazy_ownership_track_summary,
    build_lazy_seedling_identity_state,
    quality_gate_ownership_measurement_dataframe,
    write_lazy_ownership_all_metrics_workbook,
)


class LazyOwnershipExportTests(unittest.TestCase):
    def test_invalid_source_traits_are_retained_only_as_pre_qa_diagnostics(self) -> None:
        master = pd.DataFrame(
            [
                {
                    "plant_id": "plant_01",
                    "ownership_measurement_valid": True,
                    "lateral_total_length_mm": 4.0,
                    "shoot_area_px": 20,
                },
                {
                    "plant_id": "plant_02",
                    "ownership_measurement_valid": False,
                    "lateral_total_length_mm": 99.0,
                    "shoot_area_px": 500,
                },
            ]
        )

        gated = quality_gate_ownership_measurement_dataframe(master)

        self.assertEqual(float(gated.iloc[0]["lateral_total_length_mm"]), 4.0)
        self.assertTrue(pd.isna(gated.iloc[1]["lateral_total_length_mm"]))
        self.assertTrue(pd.isna(gated.iloc[1]["shoot_area_px"]))
        self.assertEqual(float(gated.iloc[1]["lateral_total_length_mm_diagnostic_pre_qa"]), 99.0)
        self.assertEqual(float(gated.iloc[1]["shoot_area_px_diagnostic_pre_qa"]), 500.0)

    def test_invalid_individual_row_preserves_valid_combined_group_measurement(self) -> None:
        master = pd.DataFrame(
            [
                {
                    "plant_id": "plant_02",
                    "ownership_measurement_valid": False,
                    "total_root_length_mm_clean": 12.0,
                    "combined_root_length_mm": 31.5,
                    "combined_root_length_px": 1182.0,
                    "combined_area_px": 4200,
                    "combined_area_mm2": 2.98,
                }
            ]
        )

        gated = quality_gate_ownership_measurement_dataframe(master)

        self.assertTrue(pd.isna(gated.iloc[0]["total_root_length_mm_clean"]))
        self.assertAlmostEqual(float(gated.iloc[0]["combined_root_length_mm"]), 31.5)
        self.assertAlmostEqual(float(gated.iloc[0]["combined_root_length_px"]), 1182.0)
        self.assertEqual(int(gated.iloc[0]["combined_area_px"]), 4200)
        self.assertAlmostEqual(float(gated.iloc[0]["combined_area_mm2"]), 2.98)

    def test_invalid_identity_with_root_pixels_is_not_reported_as_root_missing(self) -> None:
        detail = build_lazy_ownership_detail_dataframe(
            pd.DataFrame(
                [
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_01",
                        "Timestamp": "2026-07-07 09:00:00",
                        "FrameIndex": 0,
                        "plant_id": "plant_02",
                        "total_root_length_mm_clean": 14.0,
                        "primary_root_length_mm_clean": 10.0,
                        "lateral_total_length_mm": 4.0,
                        "total_root_area_px": 900,
                        "shoot_area_px": 50,
                        "ownership_measurement_valid": False,
                        "conflict_group_size": 2,
                    }
                ]
            )
        )

        row = detail.iloc[0]
        self.assertTrue(bool(row["ownership_root_present"]))
        reasons = set(str(row["ownership_frame_review_reasons"]).split(";"))
        self.assertIn("invalid_ownership_measurement", reasons)
        self.assertIn("seedling_conflict", reasons)
        self.assertNotIn("root_missing", reasons)

    def test_missing_ownership_validity_fails_closed(self) -> None:
        detail = build_lazy_ownership_detail_dataframe(
            pd.DataFrame(
                [
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_01",
                        "Timestamp": "2026-07-07 09:00:00",
                        "FrameIndex": 0,
                        "plant_id": "plant_01",
                        "total_root_length_mm_raw": 12.5,
                        "primary_root_length_mm_raw": 10.0,
                        "lateral_total_length_mm": 2.5,
                    }
                ]
            )
        )

        row = detail.iloc[0]
        self.assertFalse(bool(row["ownership_validity_known"]))
        self.assertFalse(bool(row["ownership_valid"]))
        self.assertTrue(pd.isna(row["ownership_total_root_length_mm"]))
        self.assertAlmostEqual(float(row["diagnostic_ownership_total_root_length_mm"]), 12.5)
        self.assertEqual(str(row["ownership_frame_review_status"]), "fail")

    def test_frame_review_preserves_shoot_source_columns(self) -> None:
        master_df = pd.DataFrame(
            [
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-08 09:00:00",
                    "FrameIndex": 0,
                    "plant_id": "plant_01",
                    "total_root_length_mm_clean": 0.0,
                    "primary_root_length_mm_clean": 0.0,
                    "lateral_total_length_mm": 0.0,
                    "total_root_area_px": 0,
                    "shoot_area_mm2": 1.25,
                    "shoot_area_px": 250,
                    "shoot_measurement_source": "rgb_green_only",
                    "shoot_rgb_green_only_enabled": True,
                    "ownership_shoot_mask_source": "rgb_green_only",
                    "ownership_anchor_mask_source": "rgb_green_only",
                    "ownership_anchor_rgb_green_only_enabled": True,
                    "ownership_measurement_valid": True,
                    "conflict_group_size": 1,
                },
            ]
        )

        detail = build_lazy_ownership_detail_dataframe(master_df)
        frame_review = build_lazy_ownership_frame_review_dataframe(detail)

        self.assertEqual(len(frame_review), 1)
        self.assertIn("shoot_measurement_source", frame_review.columns)
        self.assertIn("shoot_rgb_green_only_enabled", frame_review.columns)
        self.assertIn("ownership_shoot_mask_source", frame_review.columns)
        self.assertIn("ownership_anchor_mask_source", frame_review.columns)
        self.assertEqual(str(frame_review.iloc[0]["shoot_measurement_source"]), "rgb_green_only")
        self.assertTrue(bool(frame_review.iloc[0]["shoot_rgb_green_only_enabled"]))

    def test_track_summary_and_workbook_include_primary_lateral_shoot_metrics(self) -> None:
        master_df = pd.DataFrame(
            [
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-07 09:00:00",
                    "FrameIndex": 0,
                    "plant_id": "plant_01",
                    "total_root_length_mm_clean": 10.0,
                    "primary_root_length_mm_clean": 7.0,
                    "lateral_total_length_mm": 3.0,
                    "shoot_area_mm2": 2.0,
                    "shoot_area_px": 200,
                    "ownership_measurement_valid": True,
                    "conflict_group_size": 1,
                    "ownership_root_class_id": 4,
                    "ownership_lateral_class_id": 5,
                    "ownership_shoot_class_id": 6,
                },
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-07 09:05:00",
                    "FrameIndex": 1,
                    "plant_id": "plant_01",
                    "total_root_length_mm_clean": 14.0,
                    "primary_root_length_mm_clean": 8.5,
                    "lateral_total_length_mm": 5.5,
                    "shoot_area_mm2": 3.25,
                    "shoot_area_px": 325,
                    "ownership_measurement_valid": False,
                    "conflict_group_size": 2,
                    "ownership_root_class_id": 4,
                    "ownership_lateral_class_id": 5,
                    "ownership_shoot_class_id": 6,
                },
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-07 09:00:00",
                    "FrameIndex": 0,
                    "plant_id": "plant_02",
                    "total_root_length_mm_clean": 4.0,
                    "primary_root_length_mm_clean": 4.0,
                    "lateral_total_length_mm": 0.0,
                    "shoot_area_mm2": 1.0,
                    "shoot_area_px": 100,
                    "ownership_measurement_valid": True,
                    "conflict_group_size": 1,
                    "ownership_root_class_id": 4,
                    "ownership_lateral_class_id": 5,
                    "ownership_shoot_class_id": 6,
                },
            ]
        )
        timelapse_df = pd.DataFrame(
            [
                {"Series": "synthetic", "PetriDish": "plate_01", "FrameIndex": 0, "total_root_length_mm": 14.0},
                {"Series": "synthetic", "PetriDish": "plate_01", "FrameIndex": 1, "total_root_length_mm": 14.0},
            ]
        )
        pmi_df = pd.DataFrame(
            [
                {"round": "2026-07-07 09:00:00", "plate": "plate_01", "plant_id": "plant_01", "parameter": "root.length", "value": 10.0}
            ]
        )

        summary = build_lazy_ownership_track_summary(master_df)
        plant_01 = summary[summary["plant_id"] == "plant_01"].iloc[0]
        self.assertEqual(int(plant_01["frames_observed"]), 2)
        self.assertEqual(int(plant_01["invalid_ownership_frames"]), 1)
        self.assertEqual(int(plant_01["conflict_frames"]), 1)
        self.assertAlmostEqual(float(plant_01["delta_total_root_length_mm"]), 0.0)
        self.assertAlmostEqual(float(plant_01["delta_primary_root_length_mm"]), 0.0)
        self.assertAlmostEqual(float(plant_01["delta_lateral_root_length_mm"]), 0.0)
        self.assertAlmostEqual(float(plant_01["delta_shoot_area_mm2"]), 0.0)
        self.assertAlmostEqual(float(plant_01["delta_shoot_area_px"]), 0.0)
        self.assertIn(str(plant_01["ownership_review_status"]), {"review", "fail"})
        self.assertIn("seedling_conflicts", str(plant_01["ownership_review_reasons"]))

        with tempfile.TemporaryDirectory(prefix="lazy_ownership_export_") as tmp_dir_name:
            output_dir = Path(tmp_dir_name)
            workbook_path, metadata_path = write_lazy_ownership_all_metrics_workbook(
                master_df,
                output_dir,
                timelapse_df=timelapse_df,
                pmi_df=pmi_df,
            )

            self.assertIsNotNone(workbook_path)
            self.assertIsNotNone(metadata_path)
            assert workbook_path is not None
            assert metadata_path is not None
            self.assertTrue(workbook_path.exists())
            self.assertTrue(metadata_path.exists())
            with zipfile.ZipFile(workbook_path) as zf:
                workbook_xml = zf.read("xl/workbook.xml").decode("utf-8")
            for sheet_name in (
                "per_plant_detail",
                "track_summary",
                "tracks_needing_review",
                "frames_needing_review",
                "plate_timelapse",
                "pmi_style",
                "run_config",
            ):
                self.assertIn(f'name="{sheet_name}"', workbook_xml)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            detail_csv_path = Path(metadata["per_plant_detail_csv"])
            track_summary_csv_path = Path(metadata["track_summary_csv"])
            review_csv_path = Path(metadata["tracks_needing_review_csv"])
            frame_review_csv_path = Path(metadata["frames_needing_review_csv"])
            identity_state_path = Path(metadata["seedling_identity_state_json"])
            self.assertTrue(detail_csv_path.exists())
            self.assertTrue(track_summary_csv_path.exists())
            self.assertTrue(review_csv_path.exists())
            self.assertTrue(frame_review_csv_path.exists())
            self.assertTrue(identity_state_path.exists())
            detail_csv = pd.read_csv(detail_csv_path)
            track_summary_csv = pd.read_csv(track_summary_csv_path)
            review_csv = pd.read_csv(review_csv_path)
            frame_review_csv = pd.read_csv(frame_review_csv_path)
            identity_state = json.loads(identity_state_path.read_text(encoding="utf-8"))
            self.assertIn("ownership_selected_root_growth_rate_mm_per_day", detail_csv.columns)
            self.assertIn("ownership_frame_review_status", detail_csv.columns)
            self.assertIn("mean_selected_root_growth_rate_mm_per_day", track_summary_csv.columns)
            self.assertIn("frame_fail_frames", track_summary_csv.columns)
            self.assertIn("frame_review_reasons", track_summary_csv.columns)
            self.assertIn("ownership_review_status", review_csv.columns)
            self.assertIn("ownership_frame_review_status", frame_review_csv.columns)
            self.assertIn("ownership_frame_review_reasons", frame_review_csv.columns)
            self.assertGreaterEqual(len(review_csv), 1)
            self.assertGreaterEqual(len(frame_review_csv), 1)
            self.assertEqual(metadata["analysis_mode"], "ownership")
            self.assertEqual(metadata["detail_rows"], 3)
            self.assertEqual(metadata["tracks"], 2)
            self.assertEqual(metadata["petri_dishes"], 1)
            self.assertEqual(metadata["total_root_metric_column"], "total_root_length_mm_clean")
            self.assertEqual(metadata["primary_root_metric_column"], "primary_root_length_mm_clean")
            self.assertEqual(metadata["lateral_root_metric_column"], "lateral_total_length_mm")
            self.assertEqual(metadata["shoot_area_metric_column"], "shoot_area_mm2")
            self.assertEqual(metadata["selected_root_metric_mode"], "total")
            self.assertEqual(metadata["selected_root_metric_column"], "ownership_total_root_length_mm")
            self.assertEqual(metadata["derived_total_root_metric_column"], "ownership_total_root_length_mm")
            self.assertEqual(metadata["derived_selected_root_length_metric_column"], "ownership_selected_root_length_mm")
            self.assertEqual(metadata["ownership_root_class_ids"], [4])
            self.assertEqual(metadata["ownership_lateral_class_ids"], [5])
            self.assertEqual(metadata["ownership_shoot_class_ids"], [6])
            self.assertGreaterEqual(metadata["tracks_needing_review"], 1)
            self.assertGreaterEqual(metadata["tracks_failed_review"], 0)
            self.assertGreaterEqual(metadata["frames_needing_review"], 1)
            self.assertGreaterEqual(metadata["frames_failed_review"], 0)
            self.assertEqual(metadata["ownership_review_sheet"], "tracks_needing_review")
            self.assertEqual(metadata["csv_outputs"]["per_plant_detail"], str(detail_csv_path))
            self.assertEqual(metadata["csv_outputs"]["track_summary"], str(track_summary_csv_path))
            self.assertEqual(metadata["csv_outputs"]["tracks_needing_review"], str(review_csv_path))
            self.assertEqual(metadata["csv_outputs"]["frames_needing_review"], str(frame_review_csv_path))
            self.assertEqual(metadata["json_outputs"]["identity_state"], str(identity_state_path))
            self.assertEqual(identity_state["analysis_mode"], "ownership")
            self.assertEqual(identity_state["selected_root_metric_mode"], "total")
            self.assertEqual(identity_state["track_count"], 2)
            plant_state = next(track for track in identity_state["tracks"] if track["plant_id"] == "plant_01")
            self.assertEqual(int(plant_state["frames_observed"]), 2)
            self.assertEqual(int(plant_state["root_class_id"]), 4)
            self.assertEqual(int(plant_state["lateral_class_id"]), 5)
            self.assertEqual(int(plant_state["shoot_class_id"]), 6)
            self.assertEqual(plant_state["final_observation"]["frame_index"], 1)
            self.assertIsNone(plant_state["final_observation"]["selected_root_length_mm"])
            self.assertAlmostEqual(
                float(plant_state["final_observation"]["diagnostic_selected_root_length_mm"]),
                14.0,
            )
            self.assertEqual(plant_state["final_valid_observation"]["frame_index"], 0)
            self.assertAlmostEqual(
                float(plant_state["final_valid_observation"]["selected_root_length_mm"]),
                10.0,
            )
            self.assertIn("ownership_review_thresholds", metadata)
            self.assertIn("ownership_quality_definition", metadata)
            self.assertIn("derived_growth_rate_columns", metadata)
            self.assertIn("derived_frame_review_columns", metadata)
            self.assertIn("frame_review_definition", metadata)
            self.assertEqual(metadata["derived_elapsed_hours_column"], "ownership_elapsed_hours")
            self.assertEqual(metadata["excel_truncated_cells"], 0)
            self.assertEqual(metadata["all_metrics_workbook"], str(workbook_path))

    def test_detail_and_summary_include_irregular_timestamp_growth_rates(self) -> None:
        master_df = pd.DataFrame(
            [
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-07 00:00:00",
                    "FrameIndex": 0,
                    "plant_id": "plant_01",
                    "total_root_length_mm_clean": 10.0,
                    "primary_root_length_mm_clean": 7.0,
                    "lateral_total_length_mm": 3.0,
                    "shoot_area_mm2": 2.0,
                    "shoot_area_px": 200,
                    "ownership_measurement_valid": True,
                    "conflict_group_size": 1,
                },
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-07 12:00:00",
                    "FrameIndex": 1,
                    "plant_id": "plant_01",
                    "total_root_length_mm_clean": 16.0,
                    "primary_root_length_mm_clean": 10.0,
                    "lateral_total_length_mm": 6.0,
                    "shoot_area_mm2": 3.0,
                    "shoot_area_px": 260,
                    "ownership_measurement_valid": True,
                    "conflict_group_size": 1,
                },
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-09 00:00:00",
                    "FrameIndex": 2,
                    "plant_id": "plant_01",
                    "total_root_length_mm_clean": 25.0,
                    "primary_root_length_mm_clean": 13.0,
                    "lateral_total_length_mm": 12.0,
                    "shoot_area_mm2": 6.0,
                    "shoot_area_px": 500,
                    "ownership_measurement_valid": True,
                    "conflict_group_size": 1,
                },
            ]
        )

        detail = build_lazy_ownership_detail_dataframe(master_df)
        plant = detail[detail["plant_id"] == "plant_01"].reset_index(drop=True)
        self.assertEqual(plant["ownership_time_delta_source"].tolist(), ["first_frame", "timestamp", "timestamp"])
        self.assertEqual(plant["ownership_elapsed_hours"].tolist(), [0.0, 12.0, 36.0])
        self.assertEqual(plant["ownership_elapsed_days"].tolist(), [0.0, 0.5, 1.5])
        self.assertEqual(plant["delta_ownership_total_root_length_mm"].tolist(), [0.0, 6.0, 9.0])
        self.assertEqual(plant["ownership_total_root_growth_rate_mm_per_day"].tolist(), [0.0, 12.0, 6.0])
        self.assertEqual(plant["ownership_primary_root_growth_rate_mm_per_day"].tolist(), [0.0, 6.0, 2.0])
        self.assertEqual(plant["ownership_lateral_root_growth_rate_mm_per_day"].tolist(), [0.0, 6.0, 4.0])
        self.assertEqual(plant["ownership_shoot_area_growth_rate_mm2_per_day"].tolist(), [0.0, 2.0, 2.0])
        self.assertEqual(plant["ownership_shoot_pixel_growth_rate_px_per_day"].tolist(), [0.0, 120.0, 160.0])

        summary = build_lazy_ownership_track_summary(master_df)
        plant_summary = summary.iloc[0]
        self.assertAlmostEqual(float(plant_summary["track_elapsed_hours"]), 48.0)
        self.assertAlmostEqual(float(plant_summary["track_elapsed_days"]), 2.0)
        self.assertEqual(str(plant_summary["time_delta_sources"]), "first_frame;timestamp")
        self.assertAlmostEqual(float(plant_summary["mean_selected_root_growth_rate_mm_per_day"]), 9.0)
        self.assertAlmostEqual(float(plant_summary["max_selected_root_growth_rate_mm_per_day"]), 12.0)
        self.assertAlmostEqual(float(plant_summary["mean_total_root_growth_rate_mm_per_day"]), 9.0)
        self.assertAlmostEqual(float(plant_summary["max_total_root_growth_rate_mm_per_day"]), 12.0)
        self.assertAlmostEqual(float(plant_summary["mean_shoot_area_growth_rate_mm2_per_day"]), 2.0)
        self.assertAlmostEqual(float(plant_summary["max_shoot_area_growth_rate_mm2_per_day"]), 2.0)

    def test_detail_derives_seedling_bbox_from_root_and_shoot_bboxes(self) -> None:
        master_df = pd.DataFrame(
            [
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-07 09:00:00",
                    "FrameIndex": 0,
                    "plant_id": "plant_01",
                    "total_root_length_mm_clean": 3.0,
                    "primary_root_length_mm_clean": 3.0,
                    "lateral_total_length_mm": 0.0,
                    "shoot_area_mm2": 1.0,
                    "shoot_area_px": 10,
                    "bbox_x": 10,
                    "bbox_y": 20,
                    "bbox_w": 4,
                    "bbox_h": 30,
                    "shoot_bbox_x": 6,
                    "shoot_bbox_y": 8,
                    "shoot_bbox_w": 8,
                    "shoot_bbox_h": 6,
                    "ownership_measurement_valid": True,
                    "conflict_group_size": 1,
                },
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-07 10:00:00",
                    "FrameIndex": 1,
                    "plant_id": "plant_01",
                    "total_root_length_mm_clean": 4.0,
                    "primary_root_length_mm_clean": 4.0,
                    "lateral_total_length_mm": 0.0,
                    "shoot_area_mm2": 1.0,
                    "shoot_area_px": 10,
                    "bbox_x": 14,
                    "bbox_y": 20,
                    "bbox_w": 4,
                    "bbox_h": 30,
                    "shoot_bbox_x": 10,
                    "shoot_bbox_y": 8,
                    "shoot_bbox_w": 8,
                    "shoot_bbox_h": 6,
                    "ownership_measurement_valid": True,
                    "conflict_group_size": 1,
                },
            ]
        )

        detail = build_lazy_ownership_detail_dataframe(master_df)
        row0 = detail.iloc[0]
        self.assertEqual(int(row0["seedling_bbox_x"]), 6)
        self.assertEqual(int(row0["seedling_bbox_y"]), 8)
        self.assertEqual(int(row0["seedling_bbox_w"]), 8)
        self.assertEqual(int(row0["seedling_bbox_h"]), 42)
        self.assertAlmostEqual(float(row0["ownership_seedling_center_x"]), 10.0)
        self.assertAlmostEqual(float(row0["ownership_seedling_center_y"]), 29.0)
        self.assertAlmostEqual(float(detail.iloc[1]["ownership_seedling_center_jump_px"]), 4.0)

        summary = build_lazy_ownership_track_summary(master_df)
        self.assertAlmostEqual(float(summary.iloc[0]["max_seedling_center_jump_px"]), 4.0)
        identity_state = build_lazy_seedling_identity_state(detail, summary)
        self.assertEqual(identity_state["track_count"], 1)
        track = identity_state["tracks"][0]
        self.assertEqual(track["first_observation"]["seedling_bbox"], {"x": 6, "y": 8, "w": 8, "h": 42})
        self.assertEqual(track["first_observation"]["seedling_center"], {"x": 10.0, "y": 29.0})
        self.assertAlmostEqual(float(track["max_seedling_center_jump_px"]), 4.0)

    def test_legacy_ownership_rows_derive_total_primary_and_quality_metrics(self) -> None:
        master_df = pd.DataFrame(
            [
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-07 09:00:00",
                    "FrameIndex": 0,
                    "plant_id": "plant_01",
                    "root_length_mm_clean": 4.0,
                    "total_root_length_weighted_mm_clean": 5.4,
                    "primary_root_length_weighted_mm_clean": 4.2,
                    "lateral_total_length_weighted_mm": 1.2,
                    "lateral_total_length_mm": 1.0,
                    "shoot_area_mm2": 2.0,
                    "shoot_area_px": 200,
                    "bbox_x": 10,
                    "bbox_y": 10,
                    "bbox_w": 10,
                    "bbox_h": 20,
                    "shoot_bbox_x": 8,
                    "shoot_bbox_y": 3,
                    "shoot_bbox_w": 12,
                    "shoot_bbox_h": 7,
                    "ownership_measurement_valid": True,
                    "conflict_group_size": 1,
                },
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-07 09:05:00",
                    "FrameIndex": 1,
                    "plant_id": "plant_01",
                    "root_length_mm_clean": 6.0,
                    "total_root_length_weighted_mm_clean": 10.8,
                    "primary_root_length_weighted_mm_clean": 6.2,
                    "lateral_total_length_weighted_mm": 4.6,
                    "lateral_total_length_mm": 4.0,
                    "shoot_area_mm2": 0.0,
                    "shoot_area_px": 0,
                    "bbox_x": 13,
                    "bbox_y": 14,
                    "bbox_w": 10,
                    "bbox_h": 20,
                    "shoot_bbox_x": 0,
                    "shoot_bbox_y": 0,
                    "shoot_bbox_w": 0,
                    "shoot_bbox_h": 0,
                    "ownership_measurement_valid": False,
                    "conflict_group_size": 2,
                },
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-07 09:00:00",
                    "FrameIndex": 0,
                    "plant_id": "plant_02",
                    "root_length_mm_clean": 3.0,
                    "total_root_length_weighted_mm_clean": 3.2,
                    "primary_root_length_weighted_mm_clean": 3.2,
                    "lateral_total_length_weighted_mm": 0.0,
                    "lateral_total_length_mm": 0.0,
                    "shoot_area_mm2": 1.0,
                    "shoot_area_px": 100,
                    "bbox_x": 100,
                    "bbox_y": 10,
                    "bbox_w": 10,
                    "bbox_h": 20,
                    "ownership_measurement_valid": True,
                    "conflict_group_size": 1,
                },
            ]
        )

        detail = build_lazy_ownership_detail_dataframe(master_df)
        plant_01 = detail[detail["plant_id"] == "plant_01"].reset_index(drop=True)
        self.assertEqual(float(plant_01["ownership_primary_root_length_mm"].iloc[0]), 4.0)
        self.assertTrue(pd.isna(plant_01["ownership_primary_root_length_mm"].iloc[1]))
        self.assertEqual(float(plant_01["ownership_lateral_root_length_mm"].iloc[0]), 1.0)
        self.assertTrue(pd.isna(plant_01["ownership_lateral_root_length_mm"].iloc[1]))
        self.assertEqual(float(plant_01["ownership_total_root_length_mm"].iloc[0]), 5.0)
        self.assertTrue(pd.isna(plant_01["ownership_total_root_length_mm"].iloc[1]))
        self.assertEqual(float(plant_01["delta_ownership_total_root_length_mm"].iloc[0]), 0.0)
        self.assertTrue(pd.isna(plant_01["delta_ownership_total_root_length_mm"].iloc[1]))
        self.assertEqual(
            plant_01["diagnostic_ownership_total_root_length_mm"].tolist(),
            [5.0, 10.0],
        )
        self.assertEqual(
            plant_01["diagnostic_delta_ownership_total_root_length_mm"].tolist(),
            [0.0, 5.0],
        )
        self.assertEqual(plant_01["ownership_selected_root_metric_mode"].tolist(), ["total", "total"])
        self.assertEqual(float(plant_01["ownership_selected_root_length_mm"].iloc[0]), 5.0)
        self.assertTrue(pd.isna(plant_01["ownership_selected_root_length_mm"].iloc[1]))
        self.assertEqual(plant_01["ownership_shoot_present"].tolist(), [True, False])
        self.assertEqual(plant_01["ownership_conflict"].tolist(), [False, True])
        self.assertAlmostEqual(float(plant_01["ownership_bbox_center_jump_px"].iloc[1]), 5.0)

        lateral_detail = build_lazy_ownership_detail_dataframe(master_df, root_metric_mode="lateral")
        lateral_plant_01 = lateral_detail[lateral_detail["plant_id"] == "plant_01"].reset_index(drop=True)
        self.assertEqual(lateral_plant_01["ownership_selected_root_metric_mode"].tolist(), ["lateral", "lateral"])
        self.assertEqual(
            lateral_plant_01["ownership_selected_root_metric_column"].tolist(),
            ["ownership_lateral_root_length_mm", "ownership_lateral_root_length_mm"],
        )
        self.assertEqual(float(lateral_plant_01["ownership_selected_root_length_mm"].iloc[0]), 1.0)
        self.assertTrue(pd.isna(lateral_plant_01["ownership_selected_root_length_mm"].iloc[1]))
        self.assertEqual(float(lateral_plant_01["delta_ownership_selected_root_length_mm"].iloc[0]), 0.0)
        self.assertTrue(pd.isna(lateral_plant_01["delta_ownership_selected_root_length_mm"].iloc[1]))
        weighted_detail = build_lazy_ownership_detail_dataframe(master_df, root_metric_mode="weighted_total")
        weighted_plant_01 = weighted_detail[weighted_detail["plant_id"] == "plant_01"].reset_index(drop=True)
        self.assertEqual(weighted_plant_01["ownership_selected_root_metric_mode"].tolist(), ["weighted_total", "weighted_total"])
        self.assertEqual(
            weighted_plant_01["ownership_selected_root_metric_column"].tolist(),
            ["ownership_total_root_length_weighted_mm", "ownership_total_root_length_weighted_mm"],
        )
        self.assertEqual(float(weighted_plant_01["ownership_total_root_length_weighted_mm"].iloc[0]), 5.4)
        self.assertTrue(pd.isna(weighted_plant_01["ownership_total_root_length_weighted_mm"].iloc[1]))
        self.assertEqual(float(weighted_plant_01["ownership_selected_root_length_mm"].iloc[0]), 5.4)
        self.assertTrue(pd.isna(weighted_plant_01["ownership_selected_root_length_mm"].iloc[1]))
        self.assertEqual(float(weighted_plant_01["delta_ownership_selected_root_length_mm"].iloc[0]), 0.0)
        self.assertTrue(pd.isna(weighted_plant_01["delta_ownership_selected_root_length_mm"].iloc[1]))

        summary = build_lazy_ownership_track_summary(master_df)
        plant_01_summary = summary[summary["plant_id"] == "plant_01"].iloc[0]
        plant_02_summary = summary[summary["plant_id"] == "plant_02"].iloc[0]
        self.assertEqual(int(plant_01_summary["expected_frames"]), 2)
        self.assertEqual(int(plant_02_summary["expected_frames"]), 2)
        self.assertAlmostEqual(float(plant_01_summary["final_total_root_length_mm"]), 5.0)
        self.assertAlmostEqual(float(plant_01_summary["delta_total_root_length_mm"]), 0.0)
        self.assertAlmostEqual(float(plant_01_summary["final_primary_root_length_mm"]), 4.0)
        self.assertAlmostEqual(float(plant_01_summary["final_lateral_root_length_mm"]), 1.0)
        self.assertEqual(int(plant_01_summary["shoot_missing_frames"]), 1)
        self.assertEqual(int(plant_01_summary["conflict_frames"]), 1)
        self.assertAlmostEqual(float(plant_01_summary["observed_frame_fraction"]), 1.0)
        self.assertAlmostEqual(float(plant_02_summary["observed_frame_fraction"]), 0.5)
        self.assertLess(float(plant_01_summary["ownership_stability_score_0_100"]), 100.0)
        self.assertLess(float(plant_02_summary["ownership_stability_score_0_100"]), 100.0)

        lateral_summary = build_lazy_ownership_track_summary(master_df, root_metric_mode="lateral")
        lateral_plant_01_summary = lateral_summary[lateral_summary["plant_id"] == "plant_01"].iloc[0]
        self.assertEqual(str(lateral_plant_01_summary["selected_root_metric_mode"]), "lateral")
        self.assertEqual(
            str(lateral_plant_01_summary["selected_root_metric_column"]),
            "ownership_lateral_root_length_mm",
        )
        self.assertAlmostEqual(float(lateral_plant_01_summary["final_selected_root_length_mm"]), 1.0)
        self.assertAlmostEqual(float(lateral_plant_01_summary["delta_selected_root_length_mm"]), 0.0)

        weighted_summary = build_lazy_ownership_track_summary(master_df, root_metric_mode="weighted_total")
        weighted_plant_01_summary = weighted_summary[weighted_summary["plant_id"] == "plant_01"].iloc[0]
        self.assertEqual(str(weighted_plant_01_summary["selected_root_metric_mode"]), "weighted_total")
        self.assertEqual(
            str(weighted_plant_01_summary["selected_root_metric_column"]),
            "ownership_total_root_length_weighted_mm",
        )
        self.assertAlmostEqual(float(weighted_plant_01_summary["final_selected_root_length_mm"]), 5.4)
        self.assertAlmostEqual(float(weighted_plant_01_summary["delta_selected_root_length_mm"]), 0.0)
        self.assertAlmostEqual(float(weighted_plant_01_summary["final_total_root_length_weighted_mm"]), 5.4)

        with tempfile.TemporaryDirectory(prefix="lazy_ownership_lateral_") as tmp_dir_name:
            _workbook_path, metadata_path = write_lazy_ownership_all_metrics_workbook(
                master_df,
                Path(tmp_dir_name),
                root_metric_mode="lateral",
            )
            self.assertIsNotNone(metadata_path)
            assert metadata_path is not None
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["selected_root_metric_mode"], "lateral")
            self.assertEqual(metadata["selected_root_metric_column"], "ownership_lateral_root_length_mm")
            self.assertEqual(metadata["derived_total_root_weighted_metric_column"], "ownership_total_root_length_weighted_mm")
            self.assertNotIn("aliases total", metadata["selected_root_metric_definition"])

    def test_review_dataframe_flags_bad_ownership_tracks(self) -> None:
        master_df = pd.DataFrame(
            [
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-07 09:00:00",
                    "FrameIndex": 0,
                    "plant_id": "plant_bad",
                    "total_root_length_mm_clean": 24.0,
                    "primary_root_length_mm_clean": 20.0,
                    "lateral_total_length_mm": 4.0,
                    "shoot_area_mm2": 2.0,
                    "shoot_area_px": 200,
                    "bbox_x": 0,
                    "bbox_y": 0,
                    "bbox_w": 10,
                    "bbox_h": 20,
                    "shoot_bbox_x": 0,
                    "shoot_bbox_y": 0,
                    "shoot_bbox_w": 10,
                    "shoot_bbox_h": 10,
                    "ownership_measurement_valid": True,
                    "conflict_group_size": 1,
                },
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-07 09:05:00",
                    "FrameIndex": 1,
                    "plant_id": "plant_bad",
                    "total_root_length_mm_clean": 5.0,
                    "primary_root_length_mm_clean": 5.0,
                    "lateral_total_length_mm": 0.0,
                    "shoot_area_mm2": 0.0,
                    "shoot_area_px": 0,
                    "bbox_x": 900,
                    "bbox_y": 900,
                    "bbox_w": 10,
                    "bbox_h": 20,
                    "shoot_bbox_x": 850,
                    "shoot_bbox_y": 850,
                    "shoot_bbox_w": 10,
                    "shoot_bbox_h": 10,
                    "ownership_measurement_valid": False,
                    "conflict_group_size": 4,
                },
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-07 09:10:00",
                    "FrameIndex": 2,
                    "plant_id": "plant_bad",
                    "total_root_length_mm_clean": 0.0,
                    "primary_root_length_mm_clean": 0.0,
                    "lateral_total_length_mm": 0.0,
                    "shoot_area_mm2": 0.0,
                    "shoot_area_px": 0,
                    "bbox_x": 920,
                    "bbox_y": 920,
                    "bbox_w": 0,
                    "bbox_h": 0,
                    "shoot_bbox_x": 0,
                    "shoot_bbox_y": 0,
                    "shoot_bbox_w": 0,
                    "shoot_bbox_h": 0,
                    "ownership_measurement_valid": False,
                    "conflict_group_size": 4,
                },
            ]
        )

        summary = build_lazy_ownership_track_summary(master_df)
        bad_track = summary.iloc[0]
        self.assertEqual(bad_track["ownership_review_status"], "fail")
        reasons = set(str(bad_track["ownership_review_reasons"]).split(";"))
        self.assertIn("many_invalid_ownership_frames", reasons)
        self.assertIn("shoot_missing_many_frames", reasons)
        self.assertIn("severe_seedling_conflicts", reasons)
        self.assertIn("large_negative_root_length_drop", reasons)
        self.assertIn("large_root_bbox_jump", reasons)
        detail = build_lazy_ownership_detail_dataframe(master_df)
        self.assertEqual(detail["ownership_frame_review_status"].tolist(), ["ok", "fail", "fail"])
        frame_1_reasons = set(str(detail.iloc[1]["ownership_frame_review_reasons"]).split(";"))
        self.assertIn("invalid_ownership_measurement", frame_1_reasons)
        self.assertIn("shoot_missing", frame_1_reasons)
        self.assertIn("seedling_conflict", frame_1_reasons)
        self.assertIn("large_negative_root_length_drop", frame_1_reasons)
        self.assertIn("large_root_bbox_jump", frame_1_reasons)
        frame_2_reasons = set(str(detail.iloc[2]["ownership_frame_review_reasons"]).split(";"))
        self.assertIn("root_missing", frame_2_reasons)
        self.assertIn("shoot_missing", frame_2_reasons)
        self.assertEqual(int(bad_track["frame_review_frames"]), 2)
        self.assertEqual(int(bad_track["frame_fail_frames"]), 2)
        self.assertIn("large_negative_root_length_drop", str(bad_track["frame_review_reasons"]))

        review_df = build_lazy_ownership_review_dataframe(summary)
        self.assertEqual(len(review_df), 1)
        self.assertEqual(review_df.iloc[0]["plant_id"], "plant_bad")
        self.assertEqual(review_df.iloc[0]["ownership_review_status"], "fail")
        frame_review_df = build_lazy_ownership_frame_review_dataframe(detail)
        self.assertEqual(len(frame_review_df), 2)
        self.assertEqual(frame_review_df["ownership_frame_review_status"].tolist(), ["fail", "fail"])
        self.assertIn("large_negative_root_length_drop", str(frame_review_df.iloc[0]["ownership_frame_review_reasons"]))

        with tempfile.TemporaryDirectory(prefix="lazy_ownership_review_") as tmp_dir_name:
            workbook_path, metadata_path = write_lazy_ownership_all_metrics_workbook(master_df, Path(tmp_dir_name))
            self.assertIsNotNone(workbook_path)
            self.assertIsNotNone(metadata_path)
            assert workbook_path is not None
            assert metadata_path is not None
            with zipfile.ZipFile(workbook_path) as zf:
                workbook_xml = zf.read("xl/workbook.xml").decode("utf-8")
            self.assertIn('name="tracks_needing_review"', workbook_xml)
            self.assertIn('name="frames_needing_review"', workbook_xml)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["tracks_needing_review"], 1)
            self.assertEqual(metadata["tracks_failed_review"], 1)
            self.assertEqual(metadata["frames_needing_review"], 2)
            self.assertEqual(metadata["frames_failed_review"], 2)
            review_csv = pd.read_csv(Path(metadata["tracks_needing_review_csv"]))
            self.assertEqual(len(review_csv), 1)
            self.assertEqual(review_csv.iloc[0]["plant_id"], "plant_bad")
            self.assertEqual(review_csv.iloc[0]["ownership_review_status"], "fail")
            frame_review_csv = pd.read_csv(Path(metadata["frames_needing_review_csv"]))
            self.assertEqual(len(frame_review_csv), 2)
            self.assertEqual(frame_review_csv.iloc[0]["plant_id"], "plant_bad")
            self.assertIn("ownership_frame_review_reasons", frame_review_csv.columns)

    def test_workbook_truncates_excel_unsafe_long_geometry_cells(self) -> None:
        master_df = pd.DataFrame(
            [
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_01",
                    "Timestamp": "2026-07-07 09:00:00",
                    "FrameIndex": 0,
                    "plant_id": "plant_01",
                    "root_length_mm_clean": 4.0,
                    "lateral_total_length_mm": 1.0,
                    "shoot_area_mm2": 2.0,
                    "shoot_area_px": 200,
                    "path_points": "x" * 40000,
                    "ownership_measurement_valid": True,
                    "conflict_group_size": 1,
                }
            ]
        )

        with tempfile.TemporaryDirectory(prefix="lazy_ownership_long_excel_") as tmp_dir_name:
            workbook_path, metadata_path = write_lazy_ownership_all_metrics_workbook(master_df, Path(tmp_dir_name))
            self.assertIsNotNone(workbook_path)
            self.assertIsNotNone(metadata_path)
            assert metadata_path is not None
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["excel_truncated_cells"], 1)
            self.assertEqual(metadata["excel_truncation_limit"], 32000)


if __name__ == "__main__":
    unittest.main()
