from __future__ import annotations

import unittest

import numpy as np

from resources.mask_ops import (
    build_wand_stroke_from_prediction,
    rotate_image_90,
    rotate_point_xy,
    rotate_rect_xywh,
    translate_mask,
    translate_selected_layers,
)


class MaskOpsWandTests(unittest.TestCase):
    def test_wand_selects_connected_component_under_seed(self) -> None:
        pred = np.zeros((8, 8), dtype=np.uint8)
        pred[2:5, 3:6] = 3
        stroke = build_wand_stroke_from_prediction(pred, 3, (4, 3))
        self.assertIsNotNone(stroke)
        assert stroke is not None
        self.assertEqual(tuple(stroke["bbox"]), (3, 2, 5, 4))
        self.assertEqual(int(stroke["area_px"]), 9)
        self.assertEqual(np.asarray(stroke["mask"], dtype=bool).shape, (3, 3))
        self.assertTrue(bool(np.all(np.asarray(stroke["mask"], dtype=bool))))

    def test_wand_can_snap_to_nearby_target_component(self) -> None:
        pred = np.zeros((10, 10), dtype=np.uint8)
        pred[4:7, 4:7] = 2
        stroke = build_wand_stroke_from_prediction(pred, 2, (2, 2), search_radius=4)
        self.assertIsNotNone(stroke)
        assert stroke is not None
        self.assertEqual(tuple(stroke["bbox"]), (4, 4, 6, 6))
        self.assertEqual(tuple(stroke["point"]), (4, 4))

    def test_translate_mask_shifts_without_wraparound(self) -> None:
        mask = np.zeros((5, 6), dtype=np.uint8)
        mask[1:3, 2:4] = 1
        shifted = translate_mask(mask, 2, 1)
        expected = np.zeros((5, 6), dtype=np.uint8)
        expected[2:4, 4:6] = 1
        np.testing.assert_array_equal(shifted, expected)

    def test_translate_selected_layers_moves_only_requested_classes(self) -> None:
        layers = {
            1: np.zeros((4, 4), dtype=np.uint8),
            2: np.zeros((4, 4), dtype=np.uint8),
        }
        layers[1][1, 1] = 1
        layers[2][2, 2] = 1
        moved = translate_selected_layers(layers, [1], 1, 0)
        self.assertGreater(moved, 0)
        self.assertEqual(int(layers[1][1, 2]), 1)
        self.assertEqual(int(layers[1][1, 1]), 0)
        self.assertEqual(int(layers[2][2, 2]), 1)

    def test_rotate_image_90_rotates_rgb_image(self) -> None:
        image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        rotated = rotate_image_90(image, 1)
        self.assertEqual(rotated.shape, (3, 2, 3))
        np.testing.assert_array_equal(rotated[2, 0], image[0, 0])

    def test_rotate_point_xy_matches_np_rot90_geometry(self) -> None:
        self.assertEqual(rotate_point_xy((0, 0), (2, 3), 1), (0, 2))
        self.assertEqual(rotate_point_xy((0, 0), (2, 3), 3), (1, 0))
        self.assertEqual(rotate_point_xy((2, 1), (2, 3), 2), (0, 0))

    def test_rotate_rect_xywh_rotates_bbox_without_losing_extent(self) -> None:
        rect = (1, 0, 2, 2)
        rotated_ccw = rotate_rect_xywh(rect, (3, 4), 1)
        rotated_cw = rotate_rect_xywh(rect, (3, 4), 3)
        self.assertEqual(rotated_ccw, (0, 1, 2, 2))
        self.assertEqual(rotated_cw, (1, 1, 2, 2))


if __name__ == "__main__":
    unittest.main()
