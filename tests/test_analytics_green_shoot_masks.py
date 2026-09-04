from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from resources.analytics_engine import AnalyticsConfig, _resolve_anchor_masks, _resolve_shoot_masks, run_temporal_analytics
from resources.models import DatasetImageItem


class AnalyticsGreenShootMaskTests(unittest.TestCase):
    def test_rgb_green_only_shoot_masks_replace_raw_model_class_two(self) -> None:
        image = np.full((100, 120, 3), 225, dtype=np.uint8)
        image[18:34, 46:62] = [72, 132, 38]
        image[60:82, 70:94] = [118, 126, 96]

        pred = np.zeros((100, 120), dtype=np.uint8)
        pred[20:80, 12] = 1
        pred[60:82, 70:94] = 2

        item = DatasetImageItem(
            uid="frame_001",
            name="frame_001.png",
            path=Path("frame_001.png"),
            image=image,
        )
        masks, meta = _resolve_shoot_masks(
            [item],
            {"frame_001": pred},
            AnalyticsConfig(
                root_class_id=1,
                lateral_class_id=3,
                shoot_class_id=2,
                shoot_rgb_green_only_enabled=True,
                min_component_area=1,
            ),
        )

        self.assertEqual(str(meta["selection_mode"]), "rgb_green_only")
        self.assertEqual(len(masks), 1)
        self.assertGreater(int(masks[0][24, 52]), 0)
        self.assertEqual(int(masks[0][70, 82]), 0)

    def test_run_temporal_analytics_rows_report_green_only_shoot_sources(self) -> None:
        items: list[DatasetImageItem] = []
        predictions: dict[str, np.ndarray] = {}
        for frame_idx in range(3):
            image = np.full((120, 140, 3), 230, dtype=np.uint8)
            image[20:36, 58:74] = [72, 132, 38]
            image[68:92, 88:116] = [118, 126, 96]

            pred = np.zeros((120, 140), dtype=np.uint8)
            pred[34 : 92 + (frame_idx * 6), 65] = 1
            pred[68:92, 88:116] = 2

            uid = f"frame_{frame_idx}"
            items.append(
                DatasetImageItem(
                    uid=uid,
                    name=f"{uid}.png",
                    path=Path(f"{uid}.png"),
                    image=image,
                )
            )
            predictions[uid] = pred

        payload = run_temporal_analytics(
            items,
            predictions,
            {},
            AnalyticsConfig(
                root_class_id=1,
                lateral_class_id=3,
                shoot_class_id=2,
                shoot_rgb_green_only_enabled=True,
                min_component_area=1,
                bbox_padding=2,
                expected_track_count=1,
                temporal_smoothing_enabled=False,
                pixel_size_mm=1.0,
            ),
        )

        rows = payload.get("rows")
        self.assertIsInstance(rows, list)
        self.assertEqual(len(rows), 3)
        first = rows[0]
        self.assertEqual(first["ownership_anchor_mask_source"], "rgb_green_only")
        self.assertEqual(first["ownership_shoot_mask_source"], "rgb_green_only")
        self.assertEqual(first["shoot_measurement_source"], "rgb_green_only")
        self.assertTrue(bool(first["shoot_rgb_green_only_enabled"]))
        self.assertTrue(bool(first["ownership_anchor_rgb_green_only_enabled"]))

    def test_rgb_green_only_anchor_masks_do_not_require_shoot_metadata_locals(self) -> None:
        image = np.full((100, 120, 3), 225, dtype=np.uint8)
        image[18:34, 46:62] = [72, 132, 38]
        image[60:82, 70:94] = [118, 126, 96]

        pred = np.zeros((100, 120), dtype=np.uint8)
        pred[20:80, 12] = 1
        pred[60:82, 70:94] = 2

        item = DatasetImageItem(
            uid="frame_001",
            name="frame_001.png",
            path=Path("frame_001.png"),
            image=image,
        )
        masks, meta = _resolve_anchor_masks(
            [item],
            {"frame_001": pred},
            AnalyticsConfig(
                root_class_id=1,
                lateral_class_id=3,
                shoot_class_id=2,
                shoot_rgb_green_only_enabled=True,
                min_component_area=1,
            ),
        )

        self.assertEqual(str(meta["selection_mode"]), "rgb_green_only")
        self.assertTrue(bool(meta["rgb_green_only_enabled"]))
        self.assertEqual(meta["shoot_class_id"], 2)
        self.assertEqual(len(masks), 1)
        self.assertGreater(int(masks[0][24, 52]), 0)
        self.assertEqual(int(masks[0][70, 82]), 0)


if __name__ == "__main__":
    unittest.main()
