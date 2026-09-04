from __future__ import annotations

import unittest

import cv2
import numpy as np

from resources.foundation_segmentation import (
    FoundationSegmentationConfig,
    _detect_round_plate_circle,
    available_foundation_backends,
    detect_colonies_round_petri_dish,
    run_foundation_segmentation,
)


def _synthetic_colony_plate() -> np.ndarray:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(image, (128, 128), 112, (208, 208, 208), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (128, 128), 114, (160, 160, 160), thickness=3, lineType=cv2.LINE_AA)

    cv2.circle(image, (86, 92), 13, (245, 245, 245), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (156, 112), 11, (70, 70, 70), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (122, 168), 10, (88, 132, 92), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (182, 164), 8, (230, 214, 180), thickness=-1, lineType=cv2.LINE_AA)

    return image


def _synthetic_colony_plate_on_carrier() -> tuple[np.ndarray, tuple[int, int, int]]:
    image = np.full((320, 320, 3), 242, dtype=np.uint8)

    # Square carrier behind the dish.
    cv2.rectangle(image, (58, 54), (234, 246), (230, 230, 232), thickness=-1, lineType=cv2.LINE_AA)
    cv2.rectangle(image, (58, 54), (234, 246), (204, 206, 210), thickness=4, lineType=cv2.LINE_AA)

    # Corner markers and side label that should not end up as colonies.
    for cx, cy in [(50, 278), (88, 304), (214, 28), (246, 60)]:
        cv2.circle(image, (cx, cy), 13, (24, 24, 24), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(image, (cx, cy), 9, (8, 8, 8), thickness=-1, lineType=cv2.LINE_AA)
    cv2.rectangle(image, (246, 105), (292, 248), (210, 212, 214), thickness=-1, lineType=cv2.LINE_AA)
    cv2.rectangle(image, (246, 105), (292, 248), (168, 170, 174), thickness=2, lineType=cv2.LINE_AA)

    plate_circle = (152, 150, 82)
    cv2.circle(image, (plate_circle[0], plate_circle[1]), plate_circle[2], (222, 214, 192), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (plate_circle[0], plate_circle[1]), plate_circle[2] + 2, (168, 168, 168), thickness=4, lineType=cv2.LINE_AA)

    for center, radius, color in [
        ((96, 86), 5, (190, 166, 120)),
        ((88, 98), 4, (184, 168, 132)),
        ((83, 110), 4, (174, 154, 120)),
        ((90, 122), 3, (170, 150, 116)),
        ((102, 132), 3, (176, 156, 122)),
        ((188, 126), 4, (162, 144, 112)),
        ((172, 178), 5, (164, 146, 116)),
    ]:
        cv2.circle(image, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)

    return image, plate_circle


class ColonyFoundationSegmentationTests(unittest.TestCase):
    def test_backend_is_listed(self) -> None:
        keys = [key for _title, key in available_foundation_backends()]
        self.assertIn("colony_sam2", keys)

    def test_detect_colonies_round_petri_dish_returns_mask_and_plate_meta(self) -> None:
        image = _synthetic_colony_plate()
        mask, meta = detect_colonies_round_petri_dish(
            image,
            config=FoundationSegmentationConfig(sigma=1.0, grabcut_iters=2, smooth_kernel=3),
        )
        self.assertEqual(mask.shape, image.shape[:2])
        self.assertGreater(int(np.count_nonzero(mask)), 350)
        self.assertGreaterEqual(int(meta.get("candidate_count", 0)), 1)
        self.assertIn("plate_circle", meta)
        plate_circle = meta.get("plate_circle")
        self.assertIsInstance(plate_circle, list)
        assert isinstance(plate_circle, list)
        self.assertEqual(len(plate_circle), 3)

    def test_run_foundation_segmentation_allows_promptless_colony_backend(self) -> None:
        image = _synthetic_colony_plate()
        mask, meta = run_foundation_segmentation(
            image,
            backend="colony_sam2",
            positive_prompt=None,
            negative_prompt=None,
            box_prompt=None,
            prior_mask=None,
            config=FoundationSegmentationConfig(sigma=1.0, grabcut_iters=2, smooth_kernel=3),
        )
        self.assertEqual(mask.shape, image.shape[:2])
        self.assertGreater(int(np.count_nonzero(mask)), 350)
        self.assertEqual(str(meta.get("backend_requested")), "colony_sam2")
        self.assertEqual(str(meta.get("engine")), "paper_inspired_detector_then_segment")

    def test_round_plate_detector_prefers_actual_dish_over_square_carrier(self) -> None:
        image, true_circle = _synthetic_colony_plate_on_carrier()
        circle, meta = _detect_round_plate_circle(image)
        self.assertEqual(str(meta.get("plate_strategy")), "hough_circle")
        self.assertLess(abs(int(circle[0]) - int(true_circle[0])), 14)
        self.assertLess(abs(int(circle[1]) - int(true_circle[1])), 14)
        self.assertLess(abs(int(circle[2]) - int(true_circle[2])), 16)

    def test_detect_colonies_round_petri_dish_stays_inside_plate_on_realistic_carrier(self) -> None:
        image, true_circle = _synthetic_colony_plate_on_carrier()
        mask, meta = detect_colonies_round_petri_dish(
            image,
            config=FoundationSegmentationConfig(sigma=1.0, grabcut_iters=2, smooth_kernel=3),
        )
        self.assertEqual(mask.shape, image.shape[:2])
        self.assertGreater(int(np.count_nonzero(mask)), 50)
        yy, xx = np.indices(mask.shape, dtype=np.float32)
        cx, cy, radius = true_circle
        true_plate = ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) <= float(radius * radius)
        outside_pixels = int(np.count_nonzero(np.logical_and(mask > 0, ~true_plate)))
        inside_pixels = int(np.count_nonzero(np.logical_and(mask > 0, true_plate)))
        self.assertLess(outside_pixels, 120)
        self.assertGreater(inside_pixels, outside_pixels)
        self.assertLess(int(np.count_nonzero(mask)), int(np.count_nonzero(true_plate)) // 4)


if __name__ == "__main__":
    unittest.main()
