from __future__ import annotations

import unittest

import numpy as np

from resources.app import NpecLabelingMainWindow


class _PostprocessHarness:
    dataset_items: list[object] = []

    @staticmethod
    def _series_name_for_uid(_uid: str) -> str:
        return "plate"

    @staticmethod
    def _pipeline_temporal_primary_lock_active() -> bool:
        return False


class AppLateralPostprocessTests(unittest.TestCase):
    def test_native_multiclass_lateral_pixels_are_not_redecomposed(self) -> None:
        prediction = np.array([[1, 1, 3], [0, 1, 3]], dtype=np.uint8)
        predictions = {"frame": prediction.copy()}
        metadata = {"frame": {"direct_lateral_labels": True}}

        changed = NpecLabelingMainWindow._apply_lateral_root_class_postprocess(
            _PostprocessHarness(),
            predictions=predictions,
            metadata=metadata,
            root_class_id=1,
            lateral_class_id=3,
        )

        self.assertEqual(changed, 0)
        np.testing.assert_array_equal(predictions["frame"], prediction)
        self.assertEqual(metadata["frame"]["lateral_postprocess"], "preserved_native_multiclass")
        self.assertEqual(metadata["frame"]["lateral_pixels"], 2)


if __name__ == "__main__":
    unittest.main()
