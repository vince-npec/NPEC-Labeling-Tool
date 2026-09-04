from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from resources.project_io import load_project
from resources.scripts.import_rootpainter_demo import import_rootpainter_demo


class ImportRootPainterDemoTests(unittest.TestCase):
    def test_import_rootpainter_demo_builds_root_only_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_root = root / "images"
            masks_root = root / "masks"
            images_root.mkdir()
            masks_root.mkdir()

            image = np.zeros((6, 8, 3), dtype=np.uint8)
            image[..., 0] = 110
            image[..., 1] = 90
            image[..., 2] = 70
            mask = np.full((6, 8), 255, dtype=np.uint8)
            mask[1:5, 3:5] = 0

            Image.fromarray(image).save(images_root / "Image_1.jpg")
            Image.fromarray(mask).save(masks_root / "Image_1.png")

            output_project = root / "rootpainter_demo.oclp"
            summary = import_rootpainter_demo(images_root, masks_root, output_project)
            self.assertEqual(summary["image_count"], 1)

            payload = load_project(output_project)
            self.assertEqual(len(payload["dataset_items"]), 1)
            self.assertEqual([cls.name for cls in payload["classes"]], ["Root"])
            imported = payload["annotations"]["Image_1"][1]
            self.assertEqual(int(np.count_nonzero(imported)), 8)
            self.assertAlmostEqual(float(payload["pyphenotyper_pixel_size_mm"]), 1.0 / 148.0, places=8)
            self.assertAlmostEqual(float(payload["analytics_pixel_size_mm"]), 1.0 / 148.0, places=8)


if __name__ == "__main__":
    unittest.main()
