from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from resources.analytics_engine import (
    AnalyticsConfig,
    _resolve_tracking,
    build_external_tracking_seed_from_measurement_rows,
    run_temporal_analytics,
)
from resources.models import DatasetImageItem


class AnalyticsExternalTrackingTests(unittest.TestCase):
    def _item(self, uid: str) -> DatasetImageItem:
        return DatasetImageItem(
            uid=uid,
            name=uid,
            path=Path(f"/tmp/{uid}.png"),
            image=np.zeros((40, 40), dtype=np.uint8),
        )

    def test_build_external_tracking_seed_maps_selected_scope(self) -> None:
        items = [self._item("frame_b"), self._item("frame_c")]
        rows = [
            {"uid": "frame_a", "plant_id": "plant_01", "bbox_x": 1, "bbox_y": 1, "bbox_w": 4, "bbox_h": 5},
            {"uid": "frame_b", "plant_id": "plant_01", "bbox_x": 11, "bbox_y": 1, "bbox_w": 4, "bbox_h": 5},
            {"uid": "frame_c", "plant_id": "plant_01", "bbox_x": 21, "bbox_y": 2, "bbox_w": 4, "bbox_h": 5},
            {"uid": "frame_b", "plant_id": "plant_02", "bbox_x": 31, "bbox_y": 3, "bbox_w": 6, "bbox_h": 7, "overlap_frame": 2},
        ]

        seed = build_external_tracking_seed_from_measurement_rows(
            items,
            rows,
            track_order=["plant_02", "plant_01"],
            timeline=[{"uid": "frame_a"}, {"uid": "frame_b"}, {"uid": "frame_c"}],
            seed_frame=0,
            overlap_frames={"plant_02": 2},
            source="pyphenotyper",
        )

        self.assertIsNotNone(seed)
        assert seed is not None
        self.assertEqual(seed["track_ids"], ["plant_02", "plant_01"])
        self.assertEqual(seed["seed_frame"], -1)
        self.assertEqual(seed["source"], "pyphenotyper")
        self.assertEqual(seed["overlap_frames"], {"plant_02": 2, "plant_01": None})
        self.assertEqual(seed["track_bboxes"]["plant_02"], [(31, 3, 6, 7), (0, 0, 0, 0)])
        self.assertEqual(seed["track_bboxes"]["plant_01"], [(11, 1, 4, 5), (21, 2, 4, 5)])

    def test_resolve_tracking_prefers_external_seed(self) -> None:
        items = [self._item("frame_b"), self._item("frame_c")]
        rows = [
            {"uid": "frame_b", "plant_id": "plant_01", "bbox_x": 11, "bbox_y": 1, "bbox_w": 4, "bbox_h": 5},
            {"uid": "frame_c", "plant_id": "plant_01", "bbox_x": 21, "bbox_y": 2, "bbox_w": 4, "bbox_h": 5},
            {"uid": "frame_b", "plant_id": "plant_02", "bbox_x": 31, "bbox_y": 3, "bbox_w": 6, "bbox_h": 7},
        ]
        external_seed = build_external_tracking_seed_from_measurement_rows(
            items,
            rows,
            track_order=["plant_02", "plant_01"],
            source="pyphenotyper",
        )

        masks = [np.zeros((40, 40), dtype=np.uint8), np.zeros((40, 40), dtype=np.uint8)]
        masks[0][0:2, 0:2] = 1
        masks[1][0:2, 0:2] = 1
        anchor_masks = [np.zeros_like(mask) for mask in masks]
        tracking = _resolve_tracking(
            masks,
            AnalyticsConfig(min_component_area=1, external_tracking_seed=external_seed),
            anchor_masks=anchor_masks,
        )

        self.assertEqual(tracking["track_ids"], ["plant_02", "plant_01"])
        self.assertEqual(tracking["seed_frame"], 0)
        self.assertEqual(tracking["source"], "pyphenotyper")
        self.assertEqual(tracking["track_bboxes"]["plant_02"], [(31, 3, 6, 7), (0, 0, 0, 0)])
        self.assertEqual(tracking["track_bboxes"]["plant_01"], [(11, 1, 4, 5), (21, 2, 4, 5)])

    def test_run_temporal_analytics_surfaces_external_tip_priors(self) -> None:
        items = [self._item("frame_001"), self._item("frame_002")]
        rows = [
            {"uid": "frame_001", "plant_id": "plant_01", "bbox_x": 10, "bbox_y": 2, "bbox_w": 6, "bbox_h": 6},
            {"uid": "frame_002", "plant_id": "plant_01", "bbox_x": 10, "bbox_y": 2, "bbox_w": 6, "bbox_h": 6},
        ]
        external_seed = build_external_tracking_seed_from_measurement_rows(
            items,
            rows,
            track_order=["plant_01"],
            source="external",
        )
        predictions = {
            "frame_001": np.zeros((40, 40), dtype=np.uint8),
            "frame_002": np.zeros((40, 40), dtype=np.uint8),
        }
        predictions["frame_001"][8:20, 12] = 1
        predictions["frame_002"][8:24, 12] = 1
        payload = run_temporal_analytics(
            items,
            predictions,
            annotations={},
            config=AnalyticsConfig(
                root_class_id=1,
                tracking_mode="auto",
                external_tracking_seed=external_seed,
                external_tip_priors_by_frame={
                    0: {"plant_01": [(12, 19)]},
                    1: {"plant_01": [(12, 23)]},
                },
                tip_tracking_enabled=False,
                expected_track_count=1,
                min_component_area=1,
            ),
        )
        summary = payload.get("summary", {})
        self.assertEqual(summary.get("external_tip_prior_frames"), 2)
        self.assertEqual(summary.get("external_tip_prior_tracks"), 1)


if __name__ == "__main__":
    unittest.main()
