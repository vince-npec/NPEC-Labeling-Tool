from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

from resources.pmi_export import build_pmi_style_rows, export_pmi_style_rows


class PmiExportTests(unittest.TestCase):
    def test_build_pmi_style_rows_preserves_traits_and_fluorescence_placeholders(self) -> None:
        master_df = pd.DataFrame(
            [
                {
                    "Series": "Exp81",
                    "PetriDish": "1",
                    "FrameIndex": 10,
                    "plant_id": "plant_1",
                    "root_length_mm_clean": 12.5,
                    "root_length_px_clean": 250.0,
                    "root_area_px": 100,
                    "shoot_area_px": 20,
                }
            ]
        )

        pmi_df = build_pmi_style_rows(master_df)
        self.assertFalse(pmi_df.empty)

        row = pmi_df[
            (pmi_df["round"] == 10)
            & (pmi_df["plate"] == 1)
            & (pmi_df["plant_id"] == "plant_1")
            & (pmi_df["parameter"] == "root_length_mm_clean")
        ]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["value"]), 12.5, places=6)

        placeholder = pmi_df[
            (pmi_df["round"] == 10)
            & (pmi_df["plate"] == 1)
            & (pmi_df["plant_id"] == "plant_1")
            & (pmi_df["parameter"] == "mean_fluorescence_main_root")
        ]
        self.assertEqual(len(placeholder), 1)
        self.assertTrue(pd.isna(placeholder.iloc[0]["value"]))

    def test_build_pmi_style_rows_uses_composite_plate_when_series_would_collide(self) -> None:
        master_df = pd.DataFrame(
            [
                {
                    "Series": "Series_A",
                    "PetriDish": "1",
                    "FrameIndex": 1,
                    "plant_id": "plant_1",
                    "root_length_mm_clean": 10.0,
                },
                {
                    "Series": "Series_B",
                    "PetriDish": "1",
                    "FrameIndex": 1,
                    "plant_id": "plant_1",
                    "root_length_mm_clean": 11.0,
                },
            ]
        )

        pmi_df = build_pmi_style_rows(master_df, include_fluorescence_placeholders=False)
        plates = set(pmi_df["plate"].astype(str).tolist())
        self.assertEqual(plates, {"Series_A::1", "Series_B::1"})

    def test_export_pmi_style_rows_writes_csv_and_optional_xlsx(self) -> None:
        master_df = pd.DataFrame(
            [
                {
                    "Series": "Exp81",
                    "PetriDish": "2",
                    "FrameIndex": 3,
                    "plant_id": "plant_2",
                    "root_length_mm_clean": 5.75,
                }
            ]
        )
        with tempfile.TemporaryDirectory(prefix="pmi_export_test_") as tmp_dir:
            output_dir = Path(tmp_dir)
            pmi_df, csv_path, xlsx_path = export_pmi_style_rows(master_df, output_dir)
            self.assertIsNotNone(pmi_df)
            self.assertIsNotNone(csv_path)
            self.assertTrue(csv_path.exists())
            if xlsx_path is not None:
                self.assertTrue(xlsx_path.exists())

    def test_build_pmi_style_rows_derives_pmi_pixel_counts_from_output_mask(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pmi_export_mask_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[1:4, 1:4] = 2
            mask[4:12, 7] = 3
            mask[8, 7:12] = 4
            mask_path = tmp_path / "sample_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)

            master_df = pd.DataFrame(
                [
                    {
                        "image_name": "sample.png",
                        "FrameIndex": 1,
                        "plant_id": "plant_1",
                        "pixel_size_mm": 0.05,
                        "bbox_x": 0,
                        "bbox_y": 0,
                        "bbox_w": 16,
                        "bbox_h": 16,
                        "OutputMaskPath": str(mask_path),
                    }
                ]
            )

            pmi_df = build_pmi_style_rows(master_df)

            def value_for(parameter: str) -> float:
                row = pmi_df[pmi_df["parameter"] == parameter]
                self.assertEqual(len(row), 1, parameter)
                return float(row.iloc[0]["value"])

            self.assertEqual(value_for("n_pixels_main_root"), 7.0)
            self.assertEqual(value_for("n_pixels_lateral_root"), 5.0)
            self.assertEqual(value_for("n_pixels_shoot"), 9.0)
            self.assertEqual(value_for("root_area_px"), 12.0)
            self.assertEqual(value_for("primary_root_area_px"), 7.0)
            self.assertEqual(value_for("total_root_area_px"), 12.0)
            self.assertEqual(value_for("shoot_area_px"), 9.0)
            self.assertEqual(value_for("n_pixels_unknown"), 0.0)
            self.assertGreaterEqual(value_for("n_pixels_dilated_root"), value_for("root_area_px"))
            self.assertGreater(value_for("total_root_length_px"), value_for("primary_root_length_px"))
            self.assertAlmostEqual(
                value_for("total_root_length_mm_raw"),
                value_for("total_root_length_px") * 0.05,
                places=6,
            )
            self.assertEqual(str(pmi_df.iloc[0]["plate"]), "sample.png")

    def test_build_pmi_style_rows_uses_green_only_shoot_measurement_over_raw_mask(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pmi_export_green_shoot_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[4:22, 10] = 3
            mask[8:24, 14:28] = 2
            mask_path = tmp_path / "sample_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)

            master_df = pd.DataFrame(
                [
                    {
                        "image_name": "sample.png",
                        "FrameIndex": 1,
                        "plant_id": "plant_1",
                        "pixel_size_mm": 0.1,
                        "OutputMaskPath": str(mask_path),
                        "shoot_measurement_source": "rgb_green_only",
                        "shoot_rgb_rescue_enabled": True,
                        "shoot_rgb_rescue_green_only_px": 5,
                        "shoot_area_px": 5,
                        "shoot_area_mm2": 0.05,
                    }
                ]
            )

            pmi_df = build_pmi_style_rows(master_df)

            def value_for(parameter: str) -> float:
                row = pmi_df[pmi_df["parameter"] == parameter]
                self.assertEqual(len(row), 1, parameter)
                return float(row.iloc[0]["value"])

            self.assertEqual(value_for("shoot_area_px"), 5.0)
            self.assertEqual(value_for("n_pixels_shoot"), 5.0)
            self.assertAlmostEqual(value_for("shoot_area_mm2"), 0.05, places=6)

    def test_per_seedling_export_does_not_duplicate_shared_plate_mask_traits(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pmi_export_ownership_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            mask = np.zeros((24, 24), dtype=np.uint8)
            mask[2:22, 12] = 1
            mask_path = tmp_path / "shared_plate_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)
            master_df = pd.DataFrame(
                [
                    {
                        "Series": "plate_a",
                        "PetriDish": "plate_a",
                        "FrameIndex": 1,
                        "plant_id": "plant_01",
                        "total_root_length_mm_clean": 3.0,
                        "OutputMaskPath": str(mask_path),
                    },
                    {
                        "Series": "plate_a",
                        "PetriDish": "plate_a",
                        "FrameIndex": 1,
                        "plant_id": "plant_02",
                        "total_root_length_mm_clean": 7.0,
                        "OutputMaskPath": str(mask_path),
                    },
                ]
            )

            pmi_df = build_pmi_style_rows(master_df, derive_mask_metrics=False)

            root_lengths = pmi_df[pmi_df["parameter"] == "total_root_length_mm_clean"]
            self.assertEqual(root_lengths.set_index("plant_id")["value"].to_dict(), {"plant_01": 3.0, "plant_02": 7.0})
            self.assertFalse((pmi_df["parameter"] == "root_area_px").any())
            main_root_pixels = pmi_df[pmi_df["parameter"] == "n_pixels_main_root"]
            self.assertTrue(main_root_pixels["value"].isna().all())

    def test_invalid_ownership_row_cannot_leak_traits_or_shared_mask_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pmi_export_invalid_ownership_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            mask = np.zeros((24, 24), dtype=np.uint8)
            mask[2:22, 12] = 1
            mask_path = tmp_path / "shared_plate_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)
            master_df = pd.DataFrame(
                [
                    {
                        "Series": "plate_a",
                        "PetriDish": "plate_a",
                        "FrameIndex": 1,
                        "plant_id": "plant_01",
                        "ownership_measurement_valid": False,
                        "lateral_total_length_mm": 99.0,
                        "root_area_px": 999,
                        "OutputMaskPath": str(mask_path),
                    }
                ]
            )

            pmi_df = build_pmi_style_rows(master_df, derive_mask_metrics=True)

            values = pd.to_numeric(pmi_df["value"], errors="coerce")
            self.assertTrue(values.isna().all())


if __name__ == "__main__":
    unittest.main()
