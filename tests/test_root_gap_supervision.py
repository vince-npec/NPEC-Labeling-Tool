from __future__ import annotations

import unittest

import numpy as np

from resources.root_gap_supervision import (
    PROVENANCE_NONE,
    PROVENANCE_SYNTHETIC_BACTERIAL_OCCLUSION,
    create_synthetic_bacterial_occlusions,
)


class RootGapSupervisionTests(unittest.TestCase):
    @staticmethod
    def _branching_root() -> np.ndarray:
        root = np.zeros((96, 128), dtype=np.uint8)
        root[8:89, 62:66] = 255
        for y, direction in ((24, -1), (40, 1), (58, -1), (73, 1)):
            for step in range(34):
                x = 64 + (direction * step)
                root[y + (step // 12), x] = 255
        return root

    def test_hidden_is_original_subset_and_reconstructs_original(self) -> None:
        original = self._branching_root()
        authoritative_before = original.copy()

        result = create_synthetic_bacterial_occlusions(
            original,
            seed=41,
            occlusion_count=5,
            min_radius=4,
            max_radius=10,
        )
        original_bool = original != 0

        self.assertGreater(result.applied_occlusion_count, 0)
        self.assertTrue(np.all(result.hidden_root_target <= original_bool))
        self.assertFalse(np.any(result.visible_root_mask & result.hidden_root_target))
        np.testing.assert_array_equal(
            result.visible_root_mask | result.hidden_root_target,
            original_bool,
        )
        np.testing.assert_array_equal(original, authoritative_before)
        self.assertFalse(np.shares_memory(original, result.visible_root_mask))
        self.assertFalse(np.shares_memory(original, result.hidden_root_target))

    def test_provenance_marks_only_synthetic_hidden_root_pixels(self) -> None:
        original = self._branching_root()
        result = create_synthetic_bacterial_occlusions(original, seed=7)

        expected = np.full(original.shape, PROVENANCE_NONE, dtype=np.uint8)
        expected[result.hidden_root_target] = PROVENANCE_SYNTHETIC_BACTERIAL_OCCLUSION
        np.testing.assert_array_equal(result.hidden_root_provenance, expected)
        self.assertTrue(np.all(result.hidden_root_target <= result.synthetic_occluder_mask))
        self.assertTrue(np.any(result.synthetic_occluder_mask & ~(original != 0)))

    def test_seed_behavior_is_deterministic(self) -> None:
        original = self._branching_root()
        first = create_synthetic_bacterial_occlusions(original, seed=123, occlusion_count=4)
        repeated = create_synthetic_bacterial_occlusions(original, seed=123, occlusion_count=4)
        different = create_synthetic_bacterial_occlusions(original, seed=124, occlusion_count=4)

        np.testing.assert_array_equal(first.visible_root_mask, repeated.visible_root_mask)
        np.testing.assert_array_equal(first.hidden_root_target, repeated.hidden_root_target)
        np.testing.assert_array_equal(first.synthetic_occluder_mask, repeated.synthetic_occluder_mask)
        self.assertFalse(np.array_equal(first.hidden_root_target, different.hidden_root_target))

    def test_no_op_is_safe_for_empty_content_and_zero_count(self) -> None:
        empty_content = np.zeros((20, 30), dtype=np.uint8)
        empty_result = create_synthetic_bacterial_occlusions(empty_content, seed=9)
        self.assertEqual(empty_result.applied_occlusion_count, 0)
        self.assertFalse(np.any(empty_result.hidden_root_target))
        self.assertFalse(np.any(empty_result.synthetic_occluder_mask))
        np.testing.assert_array_equal(empty_result.visible_root_mask, empty_content.astype(bool))

        original = self._branching_root()
        original_before = original.copy()
        zero_result = create_synthetic_bacterial_occlusions(
            original,
            seed=9,
            occlusion_count=0,
        )
        self.assertEqual(zero_result.applied_occlusion_count, 0)
        self.assertFalse(np.any(zero_result.hidden_root_target))
        np.testing.assert_array_equal(zero_result.visible_root_mask, original != 0)
        np.testing.assert_array_equal(original, original_before)

    def test_validation_rejects_ambiguous_or_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            create_synthetic_bacterial_occlusions(np.zeros((4, 4, 1), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "non-zero"):
            create_synthetic_bacterial_occlusions(np.zeros((0, 4), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "NaN"):
            create_synthetic_bacterial_occlusions(np.array([[np.nan]], dtype=float))
        with self.assertRaisesRegex(ValueError, "max_radius"):
            create_synthetic_bacterial_occlusions(np.ones((4, 4)), min_radius=3, max_radius=2)
        with self.assertRaises(TypeError):
            create_synthetic_bacterial_occlusions(np.ones((4, 4)), occlusion_count=True)


if __name__ == "__main__":
    unittest.main()
