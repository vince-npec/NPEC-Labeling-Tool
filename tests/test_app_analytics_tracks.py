from __future__ import annotations

import unittest

from resources.app import NpecLabelingMainWindow


class _FakeAnalyticsTrackWindow:
    @staticmethod
    def _natural_sort_key(text: str):
        return ((1, str(text)),)

    _analytics_tracks_from_rows = NpecLabelingMainWindow._analytics_tracks_from_rows
    _analytics_track_rows_from_payload = NpecLabelingMainWindow._analytics_track_rows_from_payload


class AppAnalyticsTrackRowsTests(unittest.TestCase):
    def test_track_rows_surface_total_primary_lateral_and_shoot_metrics(self) -> None:
        payload = {
            "tracks": {
                "plant_01": {
                    "total_root_length_mm_clean": [10.0, 15.0, 19.0],
                    "total_root_length_mm_raw": [10.0, 16.0, 20.0],
                    "primary_root_length_mm_clean": [8.0, 11.0, 12.5],
                    "lateral_total_length_mm_per_frame": [2.0, 4.0, 6.5],
                    "shoot_area_peak_px": 420,
                    "shoot_area_mean_px": 240.0,
                    "root_area_peak_px": 1500,
                    "bbox_height_max_px": 90,
                    "bbox_center_jump_px_max": 410.0,
                    "bbox_center_jump_px_mean": 95.0,
                    "shoot_center_jump_px_max": 120.0,
                }
            },
            "rows": [
                {"plant_id": "plant_01", "tips": 2, "branches": 1},
                {"plant_id": "plant_01", "tips": 3, "branches": 2},
            ],
            "summary": {},
        }

        rows = _FakeAnalyticsTrackWindow()._analytics_track_rows_from_payload(payload)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["plant_id"], "plant_01")
        self.assertAlmostEqual(float(row["total_end_mm"]), 19.0)
        self.assertAlmostEqual(float(row["total_delta_mm"]), 9.0)
        self.assertAlmostEqual(float(row["primary_end_mm"]), 12.5)
        self.assertAlmostEqual(float(row["lateral_end_mm"]), 6.5)
        self.assertAlmostEqual(float(row["primary_peak_mm"]), 12.5)
        self.assertAlmostEqual(float(row["lateral_peak_mm"]), 6.5)
        self.assertEqual(int(row["shoot_area_peak_px"]), 420)
        self.assertAlmostEqual(float(row["shoot_area_mean_px"]), 240.0)
        self.assertTrue(bool(row["needs_review"]))
        self.assertIn("root jump", str(row["review_reasons"]))
        self.assertIn("Review", str(row["status"]))
        self.assertAlmostEqual(float(row["bbox_center_jump_px_max"]), 410.0)
        self.assertEqual(int(row["tips_max"]), 3)
        self.assertEqual(int(row["branches_max"]), 2)


if __name__ == "__main__":
    unittest.main()
