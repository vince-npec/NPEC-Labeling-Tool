from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from resources.analytics_engine import AnalyticsConfig, _resolve_tracking, run_temporal_analytics
from resources.models import DatasetImageItem
from resources.tracker_2dt import TwoDTTrackerConfig, build_arabidopsis_2dt_tracking_seed


class Tracker2DTTests(unittest.TestCase):
    def _timeline(self) -> tuple[list[DatasetImageItem], list[np.ndarray], list[np.ndarray]]:
        items: list[DatasetImageItem] = []
        root_masks: list[np.ndarray] = []
        anchor_masks: list[np.ndarray] = []
        left_depths = [6, 10, 14]
        right_depths = [4, 8, 12]
        for idx, (left_depth, right_depth) in enumerate(zip(left_depths, right_depths)):
            image = np.full((32, 32), 220, dtype=np.uint8)
            root = np.zeros((32, 32), dtype=np.uint8)
            anchor = np.zeros((32, 32), dtype=np.uint8)
            root[4 : 4 + left_depth, 8] = 1
            root[4 : 4 + right_depth, 24] = 1
            anchor[1:4, 7:10] = 1
            anchor[1:4, 23:26] = 1
            image[root > 0] = 60
            image[anchor > 0] = 90
            items.append(
                DatasetImageItem(
                    uid=f"frame_{idx:03d}",
                    name=f"frame_{idx:03d}.png",
                    path=Path(f"/tmp/frame_{idx:03d}.png"),
                    image=image,
                )
            )
            root_masks.append(root)
            anchor_masks.append(anchor)
        return items, root_masks, anchor_masks

    def test_build_arabidopsis_2dt_tracking_seed_generates_seeded_tracks(self) -> None:
        timeline, root_masks, anchor_masks = self._timeline()

        tracking = build_arabidopsis_2dt_tracking_seed(
            timeline,
            root_masks,
            anchor_masks,
            TwoDTTrackerConfig(expected_track_count=2, min_component_area=1, bbox_padding=0),
        )

        self.assertEqual(tracking["source"], "arabidopsis_2dt")
        self.assertEqual(tracking["track_ids"], ["plant_01", "plant_02"])
        self.assertGreaterEqual(int(tracking["seed_frame"]), 0)
        self.assertTrue(any(v > 0 for v in tracking["two_dt_summary"]["positive_apparition_labels"]))
        self.assertGreater(int(tracking["two_dt_summary"]["graph_vertex_count"]), 0)
        self.assertEqual(tracking["track_bboxes"]["plant_01"][-1][0], 8)
        self.assertEqual(tracking["track_bboxes"]["plant_02"][-1][0], 24)

    def test_resolve_tracking_supports_arabidopsis_2dt_mode(self) -> None:
        timeline, root_masks, anchor_masks = self._timeline()

        tracking = _resolve_tracking(
            root_masks,
            AnalyticsConfig(
                tracking_mode="arabidopsis_2dt",
                min_component_area=1,
                bbox_padding=0,
                expected_track_count=2,
            ),
            anchor_masks=anchor_masks,
            timeline=timeline,
        )

        self.assertEqual(tracking["source"], "arabidopsis_2dt")
        self.assertEqual(tracking["track_ids"], ["plant_01", "plant_02"])
        self.assertIn("two_dt_summary", tracking)

    def test_run_temporal_analytics_surfaces_two_dt_summary(self) -> None:
        timeline, root_masks, anchor_masks = self._timeline()
        predictions = {
            item.uid: root_mask.astype(np.uint8, copy=True)
            for item, root_mask in zip(timeline, root_masks)
        }

        payload = run_temporal_analytics(
            timeline,
            predictions,
            annotations={},
            config=AnalyticsConfig(
                tracking_mode="arabidopsis_2dt",
                min_component_area=1,
                bbox_padding=0,
                root_class_id=1,
                expected_track_count=2,
                lateral_class_id=None,
                seed_class_id=1,
            ),
        )

        summary = payload.get("summary", {})
        self.assertEqual(summary.get("tracking_source"), "arabidopsis_2dt")
        self.assertIsInstance(summary.get("two_dt_tracker"), dict)
        self.assertEqual(summary.get("two_dt_tracker", {}).get("mode"), "arabidopsis_2dt_graph_tracking")

    def test_run_temporal_analytics_can_cancel_during_two_dt_tracking(self) -> None:
        timeline, root_masks, _anchor_masks = self._timeline()
        predictions = {
            item.uid: root_mask.astype(np.uint8, copy=True)
            for item, root_mask in zip(timeline, root_masks)
        }
        calls: list[tuple[int, int]] = []

        def _cancel_after_first_update(step: int, total: int) -> bool:
            calls.append((int(step), int(total)))
            return len(calls) < 3

        payload = run_temporal_analytics(
            timeline,
            predictions,
            annotations={},
            config=AnalyticsConfig(
                tracking_mode="arabidopsis_2dt",
                min_component_area=1,
                bbox_padding=0,
                root_class_id=1,
                expected_track_count=2,
                lateral_class_id=None,
                seed_class_id=1,
            ),
            progress_callback=_cancel_after_first_update,
        )

        self.assertEqual(payload.get("summary", {}).get("status"), "cancelled")
        self.assertGreaterEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
