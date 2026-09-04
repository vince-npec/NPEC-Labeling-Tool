from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from resources.scripts.import_prmi_official import build_prmi_public_corpus


class ImportPrmiOfficialTests(unittest.TestCase):
    def test_build_prmi_public_corpus_preserves_splits_and_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="prmi_import_test_") as tmp_dir:
            root = Path(tmp_dir) / "PRMI_official"
            for split in ("train", "val", "test"):
                (root / split / "images" / "Peanut_10x8_DPI120").mkdir(parents=True, exist_ok=True)
                (root / split / "masks_pixel_gt" / "Peanut_10x8_DPI120").mkdir(parents=True, exist_ok=True)
                (root / split / "labels_image_gt").mkdir(parents=True, exist_ok=True)

            rows = {
                "train": [
                    {
                        "image_name": "train_a.jpg",
                        "binary_mask": "GT_train_a.png",
                        "crop": "Peanut",
                        "has_root": 1,
                        "location": "X",
                        "tube_num": "T1",
                        "date": "2022.01.01",
                        "depth": "L001",
                        "dpi": "120",
                    }
                ],
                "val": [
                    {
                        "image_name": "val_a.jpg",
                        "binary_mask": "GT_val_a.png",
                        "crop": "Peanut",
                        "has_root": 0,
                        "location": "X",
                        "tube_num": "T1",
                        "date": "2022.01.02",
                        "depth": "L002",
                        "dpi": "120",
                    }
                ],
                "test": [
                    {
                        "image_name": "test_a.jpg",
                        "binary_mask": "GT_test_a.png",
                        "crop": "Peanut",
                        "has_root": 1,
                        "location": "X",
                        "tube_num": "T1",
                        "date": "2022.01.03",
                        "depth": "L003",
                        "dpi": "120",
                    }
                ],
            }

            for split, split_rows in rows.items():
                folder = root / split / "images" / "Peanut_10x8_DPI120"
                mask_folder = root / split / "masks_pixel_gt" / "Peanut_10x8_DPI120"
                for row in split_rows:
                    image = np.zeros((8, 10, 3), dtype=np.uint8)
                    image[..., 1] = 80
                    mask = np.zeros((8, 10), dtype=np.uint8)
                    if int(row["has_root"]) == 1:
                        mask[2:6, 4:6] = 255
                    Image.fromarray(image).save(folder / row["image_name"])
                    Image.fromarray(mask).save(mask_folder / row["binary_mask"])
                json_path = root / split / "labels_image_gt" / f"Peanut_10x8_DPI120_{split}.json"
                json_path.write_text(json.dumps(split_rows), encoding="utf-8")

            output_dir = Path(tmp_dir) / "out"
            summary = build_prmi_public_corpus(root, output_dir)
            self.assertEqual(summary["sample_count"], 3)
            self.assertEqual(summary["split_counts"]["train"], 1)
            self.assertEqual(summary["split_counts"]["val"], 1)
            self.assertEqual(summary["split_counts"]["test"], 1)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["family_counts"]["minirhizotron_public"], 3)
            train_sample = next(sample for sample in manifest["samples"] if sample["split"] == "train")
            self.assertTrue(train_sample["tasks_available"]["root_binary"])
            self.assertFalse(train_sample["tasks_available"]["shoot"])
            self.assertTrue(Path(train_sample["image_path"]).exists())
            self.assertTrue(Path(train_sample["mask_path"]).exists())


if __name__ == "__main__":
    unittest.main()
