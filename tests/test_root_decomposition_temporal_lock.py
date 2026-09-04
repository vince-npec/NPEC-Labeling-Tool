from __future__ import annotations

import unittest

import numpy as np

from resources.app import (
    _decompose_main_lateral_root_masks,
    _decompose_main_lateral_root_sequence,
)


class RootDecompositionTemporalLockTests(unittest.TestCase):
    def test_temporal_lock_keeps_top_primary_pixels_main(self) -> None:
        h, w = 28, 28
        frame0 = np.zeros((h, w), dtype=np.uint8)
        frame0[2:24, 12] = 1

        frame1 = frame0.copy()
        for offset in range(12):
            frame1[8 + offset, 12 + offset] = 1

        unlocked_main, unlocked_lateral, _stats = _decompose_main_lateral_root_masks(frame1)
        outputs = _decompose_main_lateral_root_sequence(
            [frame0, frame1],
            ref_frames=1,
            start_frame=1,
            top_band_px=8,
            dilate_px=1,
            min_lock_pixels=2,
        )
        locked_main, locked_lateral, _locked_stats = outputs[1]

        self.assertGreater(int(unlocked_lateral.sum()), 0)
        self.assertEqual(int(locked_main[2, 12]), 1)
        self.assertEqual(int(locked_lateral[2, 12]), 0)
        self.assertEqual(int(locked_main[3, 12]), 1)

    def test_short_skeleton_burrs_are_not_promoted_to_lateral(self) -> None:
        h, w = 64, 64
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[6:56, 31:34] = 1
        mask[22:24, 34:37] = 1

        main, lateral, stats = _decompose_main_lateral_root_masks(mask)

        self.assertGreater(int(main.sum()), 0)
        self.assertEqual(int(lateral.sum()), 0)
        self.assertGreaterEqual(int(stats.get("lateral_rejected_components", 0)), 1)

    def test_clear_side_branch_is_kept_as_lateral(self) -> None:
        h, w = 96, 96
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[6:86, 47:50] = 1
        mask[42:45, 50:78] = 1

        main, lateral, stats = _decompose_main_lateral_root_masks(mask)

        self.assertGreater(int(main.sum()), 0)
        self.assertGreater(int(lateral.sum()), 0)
        self.assertGreaterEqual(int(stats.get("lateral_accepted_components", 0)), 1)

    def test_merged_seedlings_keep_multiple_primary_axes(self) -> None:
        h, w = 112, 112
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[6:96, 30:33] = 1
        mask[6:96, 78:81] = 1
        mask[46:49, 33:78] = 1

        main, lateral, stats = _decompose_main_lateral_root_masks(mask)

        self.assertEqual(int(main[90, 31]), 1)
        self.assertEqual(int(main[90, 79]), 1)
        self.assertGreater(int(lateral[:, 40:70].sum()), 0)
        self.assertGreaterEqual(int(stats.get("lateral_accepted_components", 0)), 1)

    def test_long_lateral_is_not_reclassified_as_primary_by_length(self) -> None:
        mask = np.zeros((64, 128), dtype=np.uint8)
        mask[4:34, 20:23] = 1
        mask[16:19, 23:112] = 1

        main, lateral, stats = _decompose_main_lateral_root_masks(mask)

        self.assertEqual(int(main[30, 21]), 1)
        self.assertEqual(int(lateral[17, 100]), 1)
        self.assertGreaterEqual(int(stats.get("lateral_accepted_components", 0)), 1)

    def test_x_crossing_prefers_tangent_continuity_and_reports_ambiguity(self) -> None:
        mask = np.zeros((80, 104), dtype=np.uint8)
        for offset in range(61):
            mask[6 + offset, 20 + offset] = 1
            mask[6 + offset, 80 - offset] = 1

        main, lateral, stats = _decompose_main_lateral_root_masks(mask)

        self.assertEqual(int(main[64, 78]), 1)
        self.assertEqual(int(main[64, 22]), 1)
        self.assertEqual(int(lateral.sum()), 0)
        self.assertGreaterEqual(int(stats.get("crossing_clusters", 0)), 1)


if __name__ == "__main__":
    unittest.main()
