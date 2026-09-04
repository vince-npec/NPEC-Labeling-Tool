from __future__ import annotations

import unittest

from resources.analytics_engine import analytics_track_overlay_label


class AnalyticsOverlayLabelTests(unittest.TestCase):
    def test_overlay_label_includes_total_primary_lateral_and_shoot(self) -> None:
        label = analytics_track_overlay_label(
            "plant_01",
            {
                "total_root_length_mm_clean": 19.25,
                "primary_root_length_mm_clean": 12.5,
                "lateral_total_length_mm": 6.75,
                "shoot_area_px": 420,
            },
        )

        self.assertEqual(label, "plant_01 T=19.25mm P=12.50 L=6.75 S=420px")

    def test_overlay_label_falls_back_to_raw_root_values(self) -> None:
        label = analytics_track_overlay_label(
            "plant_02",
            {
                "root_length_mm_raw": 4.0,
                "shoot_area_px": 12.4,
            },
        )

        self.assertEqual(label, "plant_02 T=4.00mm P=0.00 L=0.00 S=12px")


if __name__ == "__main__":
    unittest.main()
