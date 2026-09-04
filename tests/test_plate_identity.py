from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from resources.plate_identity import (
    allocate_plate_names,
    infer_plate_identity,
    read_plate_identity_manifest,
    scan_plate_identities,
    write_plate_identity_manifest,
)


def _observation(text: str, confidence: float, x: float, y: float, width: float, height: float) -> dict:
    return {
        "candidates": [{"text": text, "confidence": confidence}],
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


class PlateIdentityTests(unittest.TestCase):
    def test_ocr_label_is_combined_and_filename_date_drives_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "2024_07_15_09_34_10_0007.jpg"
            Image.fromarray(np.full((120, 160, 3), 240, dtype=np.uint8)).save(path)
            ocr = {
                str(path): {
                    "path": str(path),
                    "observations": [
                        _observation("RP043", 0.98, 0.20, 0.91, 0.08, 0.03),
                        _observation("163", 0.99, 0.12, 0.89, 0.07, 0.05),
                        _observation("g25", 0.96, 0.21, 0.86, 0.05, 0.03),
                    ],
                }
            }
            records = scan_plate_identities([path], ocr_results=ocr, decode_codes=False)

            self.assertEqual(records[0]["plate_id"], "163_RP043_g25")
            self.assertEqual(records[0]["source_date_source"], "filename")
            self.assertEqual(records[0]["allocated_name"], "163_RP043_g25__20240715_093410.jpg")
            self.assertEqual(records[0]["status"], "accepted")

    def test_qr_has_priority_and_known_prefix_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "capture.png"
            Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(path)
            record = infer_plate_identity(
                path,
                ocr_result={"observations": [_observation("wrong99", 1.0, 0.1, 0.8, 0.2, 0.1)]},
                qr_values=["URL:S37-1"],
            )
            self.assertEqual(record["plate_id"], "S37-1")
            self.assertEqual(record["evidence"][0]["kind"], "qr")
            self.assertEqual(record["confidence"], 0.99)

    def test_low_confidence_text_requires_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "capture.png"
            Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(path)
            record = infer_plate_identity(
                path,
                ocr_result={"observations": [_observation("B3-3", 0.31, 0.45, 0.45, 0.1, 0.05)]},
            )
            self.assertEqual(record["plate_id"], "B3-3")
            self.assertEqual(record["status"], "review")

    def test_collision_names_are_deterministic_and_source_files_unchanged(self) -> None:
        records = [
            {
                "source_name": "a.JPG",
                "source_path": "/source/a.JPG",
                "source_date": "2024-07-15T09:34:10",
                "plate_id": "P1",
            },
            {
                "source_name": "b.JPG",
                "source_path": "/source/b.JPG",
                "source_date": "2024-07-15T09:34:10",
                "plate_id": "P1",
            },
        ]
        first = allocate_plate_names(records)
        second = allocate_plate_names(records)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["allocated_name"], "P1__20240715_093410.jpg")
        self.assertEqual(first[1]["allocated_name"], "P1__20240715_093410__capture02.jpg")
        self.assertEqual([record["source_path"] for record in first], ["/source/a.JPG", "/source/b.JPG"])

    def test_manifest_writes_csv_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            output = Path(tmp_name) / "npec_plate_identity_manifest.csv"
            result = write_plate_identity_manifest(
                [
                    {
                        "source_name": "a.png",
                        "source_path": "/source/a.png",
                        "source_date": "2024-01-01T10:00:00",
                        "plate_id": "A1",
                        "allocated_name": "A1__20240101_100000.png",
                        "status": "accepted",
                        "confidence": 0.99,
                        "evidence": [],
                    }
                ],
                output,
            )
            self.assertEqual(result, output)
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".json").exists())
            loaded = read_plate_identity_manifest(output)
            self.assertEqual(loaded[0]["plate_id"], "A1")
            self.assertEqual(loaded[0]["allocated_name"], "A1__20240101_100000.png")
            self.assertAlmostEqual(float(loaded[0]["confidence"]), 0.99)


if __name__ == "__main__":
    unittest.main()
