from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from resources.hades_data import ensure_hades_series_folders


class HadesDataTests(unittest.TestCase):
    def test_ensure_hades_series_folders_moves_flat_files_into_dish_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in (
                "139_1_2026-03-02_22-02-15_exp79_01_ROOT1_Original.png",
                "139_2_2026-03-03_12-47-20_exp79_01_ROOT1_Original.png",
                "139_1_2026-03-02_22-04-09_exp79_02_ROOT1_Original.png",
                "notes.txt",
            ):
                (root / name).write_bytes(b"demo")

            result = ensure_hades_series_folders(root)

            self.assertEqual(result.moved_files, 3)
            self.assertEqual(result.series_names, ["exp79_01", "exp79_02"])
            self.assertEqual(result.series_counts["exp79_01"], 2)
            self.assertEqual(result.series_counts["exp79_02"], 1)
            self.assertEqual([path.name for path in result.unmatched_root_files], ["notes.txt"])
            self.assertTrue((root / "exp79_01" / "139_1_2026-03-02_22-02-15_exp79_01_ROOT1_Original.png").exists())
            self.assertTrue((root / "exp79_02" / "139_1_2026-03-02_22-04-09_exp79_02_ROOT1_Original.png").exists())

    def test_ensure_hades_series_folders_detects_existing_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            series_dir = root / "exp79_01"
            series_dir.mkdir()
            (series_dir / "139_1_2026-03-02_22-02-15_exp79_01_ROOT1_Original.png").write_bytes(b"demo")

            result = ensure_hades_series_folders(root)

            self.assertTrue(result.already_structured)
            self.assertEqual(result.moved_files, 0)
            self.assertEqual(result.series_names, ["exp79_01"])


if __name__ == "__main__":
    unittest.main()
