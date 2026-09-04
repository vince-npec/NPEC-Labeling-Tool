from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from resources.lucifer_gan_gap_repair import apply_lucifer_gan_gap_repair
from resources.models import DatasetImageItem


class LuciferGanGapRepairTests(unittest.TestCase):
    def _write_mask(self, path: Path, mask: np.ndarray) -> None:
        Image.fromarray((np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255, mode="L").save(path)

    def test_applies_delta_pixels_from_precomputed_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            precomputed_root = tmp_dir / "inference_output"
            case_dir = precomputed_root / "48-4"
            case_dir.mkdir(parents=True)
            (precomputed_root / "summary.json").write_text("[]", encoding="utf-8")

            original = np.zeros((8, 8), dtype=np.uint8)
            original[2:6, 3] = 1
            repaired = original.copy()
            repaired[4, 4:6] = 1
            self._write_mask(case_dir / "02_original_mask.png", original)
            self._write_mask(case_dir / "06_inpainted_after_cleanup.png", repaired)

            item = DatasetImageItem(
                uid="u1",
                name="48-4.png",
                path=tmp_dir / "48-4.png",
                image=np.zeros((8, 8, 3), dtype=np.uint8),
            )
            current = original.copy()
            result = apply_lucifer_gan_gap_repair(item, current, precomputed_root)

            self.assertTrue(result.applied)
            self.assertEqual(result.mode, "precomputed_delta")
            self.assertEqual(result.stem, "48-4")
            self.assertEqual(result.added_pixels, 2)
            np.testing.assert_array_equal(result.root_mask, repaired)

    def test_uses_full_repaired_mask_when_current_root_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            precomputed_root = tmp_dir / "inference_output"
            case_dir = precomputed_root / "29-3"
            case_dir.mkdir(parents=True)
            (precomputed_root / "summary.json").write_text("[]", encoding="utf-8")

            original = np.zeros((6, 6), dtype=np.uint8)
            original[1:5, 2] = 1
            repaired = original.copy()
            repaired[3, 3:5] = 1
            self._write_mask(case_dir / "02_original_mask.png", original)
            self._write_mask(case_dir / "06_inpainted_after_cleanup.png", repaired)

            item = DatasetImageItem(
                uid="u2",
                name="some_29-3_frame.png",
                path=tmp_dir / "some_29-3_frame.png",
                image=np.zeros((6, 6, 3), dtype=np.uint8),
            )
            current = np.zeros((6, 6), dtype=np.uint8)
            result = apply_lucifer_gan_gap_repair(item, current, precomputed_root)

            self.assertTrue(result.applied)
            self.assertTrue(result.used_full_repaired_mask)
            self.assertEqual(result.mode, "precomputed_replace")
            np.testing.assert_array_equal(result.root_mask, repaired)

    def test_returns_no_match_when_case_folder_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            precomputed_root = tmp_dir / "inference_output"
            precomputed_root.mkdir(parents=True)
            (precomputed_root / "summary.json").write_text("[]", encoding="utf-8")

            item = DatasetImageItem(
                uid="u3",
                name="99-9.png",
                path=tmp_dir / "99-9.png",
                image=np.zeros((5, 5, 3), dtype=np.uint8),
            )
            current = np.zeros((5, 5), dtype=np.uint8)
            result = apply_lucifer_gan_gap_repair(item, current, precomputed_root)

            self.assertFalse(result.applied)
            self.assertEqual(result.mode, "no_match")
            np.testing.assert_array_equal(result.root_mask, current)


if __name__ == "__main__":
    unittest.main()
