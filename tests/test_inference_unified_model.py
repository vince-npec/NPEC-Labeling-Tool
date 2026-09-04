import unittest

import numpy as np

from resources.inference import _compose_unified_task_class_mask


class UnifiedInferenceTests(unittest.TestCase):
    def test_compose_unified_task_class_mask_uses_expected_priority(self) -> None:
        shape = (4, 4)
        task_masks = {
            "root_binary": np.zeros(shape, dtype=np.uint8),
            "primary_root": np.zeros(shape, dtype=np.uint8),
            "lateral_root": np.zeros(shape, dtype=np.uint8),
            "shoot": np.zeros(shape, dtype=np.uint8),
            "seed_crown": np.zeros(shape, dtype=np.uint8),
        }
        task_masks["root_binary"][0, 0] = 1
        task_masks["primary_root"][0, 1] = 1
        task_masks["lateral_root"][1, 0] = 1
        task_masks["shoot"][1, 1] = 1
        task_masks["seed_crown"][2, 2] = 1
        task_masks["root_binary"][1, 0] = 1
        task_masks["root_binary"][1, 1] = 1
        task_masks["root_binary"][2, 2] = 1

        pred = _compose_unified_task_class_mask(task_masks)

        self.assertEqual(int(pred[0, 0]), 1)
        self.assertEqual(int(pred[0, 1]), 1)
        self.assertEqual(int(pred[1, 0]), 3)
        self.assertEqual(int(pred[1, 1]), 2)
        self.assertEqual(int(pred[2, 2]), 4)


if __name__ == "__main__":
    unittest.main()
