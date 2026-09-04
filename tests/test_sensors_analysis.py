from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from resources.sensors_analysis import (
    derive_plant_region_analyses,
    render_region_overlay,
    summarize_fluorescence_regions,
    summarize_hyperspectral_regions,
)
from resources.sensors_processing import apply_optical_correction_array, load_envi_wavelengths, OpticalCorrectionConfig


class SensorsAnalysisTests(unittest.TestCase):
    def test_fluorescence_summary_derives_two_plants_and_template_rows(self) -> None:
        main_root = np.zeros((40, 60), dtype=np.uint8)
        lateral = np.zeros_like(main_root)
        shoot = np.zeros_like(main_root)

        main_root[10:32, 10:12] = 1
        main_root[8:28, 42:44] = 1
        lateral[18:24, 12:18] = 1
        lateral[16:22, 36:42] = 1
        shoot[4:9, 8:15] = 1
        shoot[4:9, 40:47] = 1

        analyses = derive_plant_region_analyses(main_root, lateral, shoot, dilated_root_iterations=8)
        self.assertEqual(len(analyses), 2)
        self.assertEqual([entry.plant_id for entry in analyses], ["plant_1", "plant_2"])

        signal = np.zeros((40, 60), dtype=np.float32)
        signal[main_root > 0] = 3.0
        signal[lateral > 0] = 6.0
        signal[shoot > 0] = 2.0

        summary_rows, pixel_rows = summarize_fluorescence_regions(signal, analyses, "frame_fc.tar")
        parameters = {str(row["parameter"]) for row in summary_rows}
        self.assertIn("mean_fluorescence_main_root", parameters)
        self.assertIn("n_pixels_lateral_root", parameters)
        self.assertIn("sum_fluorescence_shoot", parameters)
        self.assertGreater(len(pixel_rows), 0)

        overlay = render_region_overlay(np.zeros((40, 60, 3), dtype=np.uint8), analyses, alpha=0.7)
        self.assertEqual(overlay.shape, (40, 60, 3))
        self.assertGreater(int(np.sum(overlay)), 0)

    def test_hyperspectral_summary_uses_region_and_wavelength_columns(self) -> None:
        main_root = np.zeros((20, 20), dtype=np.uint8)
        lateral = np.zeros_like(main_root)
        shoot = np.zeros_like(main_root)
        main_root[6:16, 8:10] = 1
        shoot[3:6, 6:12] = 1

        analyses = derive_plant_region_analyses(main_root, lateral, shoot, dilated_root_iterations=6)
        cube = np.zeros((20, 20, 3), dtype=np.float32)
        cube[:, :, 0] = 1.0
        cube[:, :, 1] = 2.0
        cube[:, :, 2] = 4.0

        rows = summarize_hyperspectral_regions(
            cube,
            analyses,
            "sample_Data.bil",
            wavelengths=[450.0, 550.0, 650.0],
            mask_source_label="predictions:sample.png",
        )
        self.assertGreater(len(rows), 0)
        first = rows[0]
        self.assertIn("region", first)
        self.assertIn("450.0nm", first)
        self.assertIn("550.0nm", first)
        self.assertIn("650.0nm", first)

    def test_apply_optical_correction_array_and_wavelength_parser(self) -> None:
        cfg = OpticalCorrectionConfig(shift_x=1.0, shift_y=-1.0)
        cube = np.zeros((12, 10, 2), dtype=np.float32)
        cube[4:8, 3:5, 0] = 2.0
        cube[5:9, 5:8, 1] = 7.0
        corrected = apply_optical_correction_array(cube, (14, 16), cfg)
        self.assertEqual(corrected.shape, (14, 16, 2))

        with tempfile.TemporaryDirectory() as tmpdir:
            hdr = Path(tmpdir) / "sample.hdr"
            bil = Path(tmpdir) / "sample.bil"
            hdr.write_text(
                "ENVI\n"
                "samples = 2\n"
                "lines = 2\n"
                "bands = 3\n"
                "interleave = bil\n"
                "wavelength = {\n"
                "450.0, 550.0,\n"
                "650.0}\n",
                encoding="utf-8",
            )
            bil.write_bytes(b"\x00" * 12)
            self.assertEqual(load_envi_wavelengths(bil), [450.0, 550.0, 650.0])


if __name__ == "__main__":
    unittest.main()
