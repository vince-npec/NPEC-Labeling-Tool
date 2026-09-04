from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from resources.architecture_tracker import export_architecture_rows_csv


class ArchitectureTrackerExportTests(unittest.TestCase):
    def test_export_architecture_rows_csv_includes_px_and_mm_columns(self) -> None:
        payload = {
            "rows": [
                {
                    "branch_id": "plant_01:branch_001",
                    "plant_id": "plant_01",
                    "frames": 4,
                    "birth_frame": 0,
                    "death_frame": 3,
                    "pixel_size_mm": 0.04118616144975288,
                    "max_length_px": 120.0,
                    "max_length_mm": 4.9423,
                    "mean_speed_px_day": 18.5,
                    "mean_speed_mm_day": 0.7619,
                    "max_speed_px_day": 22.0,
                    "max_speed_mm_day": 0.9061,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            output_path = Path(tmp_dir_name) / "architecture.csv"
            export_architecture_rows_csv(payload, output_path)
            text = output_path.read_text(encoding="utf-8")
            self.assertIn("pixel_size_mm", text)
            self.assertIn("max_length_px", text)
            self.assertIn("max_length_mm", text)
            self.assertIn("mean_speed_px_day", text)
            self.assertIn("mean_speed_mm_day", text)
            self.assertIn("max_speed_px_day", text)
            self.assertIn("max_speed_mm_day", text)


if __name__ == "__main__":
    unittest.main()
