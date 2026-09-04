from __future__ import annotations

import unittest

import cv2
import numpy as np

from resources.plant_health import (
    build_leaf_mask,
    compute_stress_cue_mask,
    compute_yellowing_mask,
    plantvillage_binary_label,
)


def _synthetic_rosette_with_yellow_patch() -> np.ndarray:
    image = np.full((240, 240, 3), 245, dtype=np.uint8)
    cv2.circle(image, (120, 120), 62, color=(84, 170, 76), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (140, 100), 18, color=(212, 198, 88), thickness=-1, lineType=cv2.LINE_AA)
    return image


class TestPlantHealthHelpers(unittest.TestCase):
    def test_binary_mapping_marks_healthy(self) -> None:
        self.assertEqual(plantvillage_binary_label("Tomato___healthy"), 0)

    def test_binary_mapping_marks_nonhealthy_as_stressed(self) -> None:
        self.assertEqual(plantvillage_binary_label("Tomato___Target_Spot"), 1)

    def test_leaf_mask_finds_green_and_yellow_rosette(self) -> None:
        image = _synthetic_rosette_with_yellow_patch()
        mask, meta = build_leaf_mask(image)
        self.assertGreater(int(np.count_nonzero(mask)), 8000)
        self.assertGreater(float(meta["leaf_coverage"]), 0.12)

    def test_yellowing_mask_finds_patch_inside_leaf(self) -> None:
        image = _synthetic_rosette_with_yellow_patch()
        leaf_mask, _ = build_leaf_mask(image)
        yellow_mask, meta = compute_yellowing_mask(image, leaf_mask)
        self.assertGreater(int(np.count_nonzero(yellow_mask)), 300)
        self.assertGreater(float(meta["yellowing_score"]), 0.01)

    def test_stress_mask_includes_yellowing_patch(self) -> None:
        image = _synthetic_rosette_with_yellow_patch()
        leaf_mask, _ = build_leaf_mask(image)
        stress_mask, meta = compute_stress_cue_mask(image, leaf_mask)
        self.assertGreater(int(np.count_nonzero(stress_mask)), 300)
        self.assertGreater(float(meta["stress_cue_score"]), 0.01)


if __name__ == "__main__":
    unittest.main()
