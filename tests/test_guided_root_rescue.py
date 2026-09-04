from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from resources.guided_root_rescue import (
    GuidedPolyline,
    GuidedRootRescueConfig,
    compute_guided_root_rescue,
)


class GuidedRootRescueTests(unittest.TestCase):
    def test_addition_masks_are_deterministic_uint8_and_exclude_existing_support(self) -> None:
        image = np.full((90, 120, 3), 245, dtype=np.uint8)
        cv2.line(image, (8, 30), (108, 30), (35, 35, 35), thickness=2)
        cv2.line(image, (60, 45), (60, 82), (45, 45, 45), thickness=2)
        support = np.zeros((90, 120), dtype=np.uint8)
        cv2.line(support, (8, 30), (35, 30), 1, thickness=3)
        guides = [
            GuidedPolyline(class_id=1, points_xy=((8, 30), (108, 30))),
            GuidedPolyline(class_id=3, points_xy=((60, 45), (60, 82))),
        ]
        config = GuidedRootRescueConfig(line_width_px=3, corridor_px=12)

        first = compute_guided_root_rescue(image, support, guides, config)
        second = compute_guided_root_rescue(image, support, guides, config)

        self.assertEqual(set(first.addition_masks), {1, 3})
        for class_id, mask in first.addition_masks.items():
            self.assertEqual(mask.dtype, np.uint8)
            self.assertEqual(mask.shape, image.shape[:2])
            self.assertEqual(set(np.unique(mask)).difference({0, 1}), set())
            self.assertEqual(int(np.count_nonzero(mask[support > 0])), 0)
            np.testing.assert_array_equal(mask, second.addition_masks[class_id])
        self.assertGreater(int(np.count_nonzero(first.addition_masks[1])), 100)
        self.assertGreater(int(np.count_nonzero(first.addition_masks[3])), 50)
        self.assertEqual(first.metadata, second.metadata)
        self.assertEqual(first.metadata["existing_support_pixels"], int(np.count_nonzero(support)))

    def test_shortest_path_follows_dark_thin_root_inside_corridor(self) -> None:
        image = np.full((80, 110, 3), 250, dtype=np.uint8)
        dark_route = np.asarray([(10, 55), (55, 20), (100, 55)], dtype=np.int32)
        cv2.polylines(image, [dark_route.reshape((-1, 1, 2))], False, (20, 20, 20), thickness=2)
        config = GuidedRootRescueConfig(
            line_width_px=1,
            corridor_px=38,
            darkness_weight=1.0,
            guide_distance_weight=0.05,
            signal_cost_weight=8.0,
        )
        result = compute_guided_root_rescue(
            image,
            None,
            [GuidedPolyline(class_id=1, points_xy=((10, 55), (100, 55)))],
            config,
        )

        if not result.metadata["shortest_path_available"]:
            self.skipTest("scikit-image shortest-path runtime is unavailable")
        segment = result.metadata["segments"][0]
        self.assertEqual(segment["method"], "shortest_path")
        self.assertEqual(segment["fallback_reason"], "")
        recovered = result.addition_masks[1]
        self.assertGreater(int(np.count_nonzero(recovered[17:25, 50:61])), 0)
        recovered_y = np.where(recovered > 0)[0]
        self.assertGreaterEqual(int(recovered_y.min()), 16)
        self.assertLessEqual(int(recovered_y.max()), 56)

    def test_unavailable_router_uses_deterministic_straight_polyline_fallback(self) -> None:
        image = np.full((60, 90, 3), 230, dtype=np.uint8)
        guides = [{"class_id": 7, "points": [(8, 12), (42, 40), (80, 20)]}]
        config = GuidedRootRescueConfig(line_width_px=3, corridor_px=8)

        with (
            patch("resources.guided_root_rescue._optional_route_through_array", return_value=None),
            patch("resources.guided_root_rescue._optional_frangi", return_value=None),
        ):
            result = compute_guided_root_rescue(image, None, guides, config)

        mask = result.addition_masks[7]
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(int(mask[12, 8]), 1)
        self.assertEqual(int(mask[40, 42]), 1)
        self.assertEqual(int(mask[20, 80]), 1)
        self.assertEqual(result.metadata["shortest_path_available"], False)
        self.assertEqual(result.metadata["routed_segment_count"], 0)
        self.assertEqual(result.metadata["fallback_segment_count"], 2)
        self.assertEqual(
            [segment["fallback_reason"] for segment in result.metadata["segments"]],
            ["shortest_path_unavailable", "shortest_path_unavailable"],
        )
        self.assertEqual(result.metadata["evidence"]["method"], "morphological_darkness_fallback")

    def test_router_error_falls_back_without_losing_the_guide(self) -> None:
        image = np.full((45, 70, 3), 240, dtype=np.uint8)

        def failing_router(*_args, **_kwargs):
            raise RuntimeError("synthetic routing failure")

        with patch(
            "resources.guided_root_rescue._optional_route_through_array",
            return_value=failing_router,
        ):
            result = compute_guided_root_rescue(
                image,
                None,
                [GuidedPolyline(class_id=2, points_xy=((5, 20), (65, 20)))],
                GuidedRootRescueConfig(line_width_px=1, corridor_px=6),
            )

        self.assertTrue(bool(np.all(result.addition_masks[2][20, 5:66] > 0)))
        segment = result.metadata["segments"][0]
        self.assertEqual(segment["method"], "straight_fallback")
        self.assertEqual(segment["fallback_reason"], "routing_error:RuntimeError")

    def test_invalid_inputs_and_config_are_rejected(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            GuidedRootRescueConfig(line_width_px=0)
        with self.assertRaises(ValueError):
            compute_guided_root_rescue(image[:, :, 0], None, [GuidedPolyline(1, ((1, 1), (2, 2)))])
        with self.assertRaises(ValueError):
            compute_guided_root_rescue(image, np.zeros((19, 30), dtype=np.uint8), [GuidedPolyline(1, ((1, 1), (2, 2)))])
        with self.assertRaises(ValueError):
            compute_guided_root_rescue(image, None, [GuidedPolyline(0, ((1, 1), (2, 2)))])
        with self.assertRaises(ValueError):
            compute_guided_root_rescue(image, None, [GuidedPolyline(1, ((3, 3), (3, 3)))])


if __name__ == "__main__":
    unittest.main()
