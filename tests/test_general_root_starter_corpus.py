from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from resources.general_root_starter.corpus import CorpusSourceSpec, build_general_root_corpus
from resources.models import DatasetImageItem, LabelClass
from resources.project_io import save_project


class GeneralRootStarterCorpusTests(unittest.TestCase):
    def test_builds_manifest_with_normalized_availability(self):
        with tempfile.TemporaryDirectory(prefix="grs_corpus_test_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[..., 1] = 120
            image_path = tmp_root / "sample_a.png"
            Image.fromarray(image).save(image_path)
            item = DatasetImageItem(uid="sample_a", name="sample_a.png", path=image_path, image=image)
            classes = [
                LabelClass(1, "Root", "#ffcc33"),
                LabelClass(2, "Shoot", "#55dd77"),
            ]
            annotations = {
                "sample_a": {
                    1: np.pad(np.ones((16, 4), dtype=np.uint8), ((8, 8), (14, 14))),
                    2: np.pad(np.ones((6, 10), dtype=np.uint8), ((2, 24), (11, 11))),
                }
            }
            project_path = tmp_root / "mini.oclp"
            save_project(
                project_path,
                [item],
                classes,
                annotations,
                predictions={},
                current_index=0,
                active_class_id=1,
                patch_target_class_id=1,
                next_class_id=3,
                model_path=None,
            )
            manifest = build_general_root_corpus(
                [CorpusSourceSpec(project_path=project_path, family="plate_bw_hades")],
                tmp_root / "corpus",
                random_seed=3,
            )
            self.assertEqual(manifest["sample_count"], 1)
            sample = manifest["samples"][0]
            self.assertEqual(sample["family"], "plate_bw_hades")
            self.assertTrue(sample["tasks_available"]["root_binary"])
            self.assertTrue(sample["tasks_available"]["shoot"])
            self.assertFalse(sample["tasks_available"]["seed_crown"])
            self.assertTrue(Path(sample["image_path"]).exists())
            self.assertTrue(Path(sample["mask_path"]).exists())

    def test_root_plus_lateral_promotes_primary_root_availability(self):
        with tempfile.TemporaryDirectory(prefix="grs_corpus_test_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[..., 0] = 80
            image_path = tmp_root / "sample_b.png"
            Image.fromarray(image).save(image_path)
            item = DatasetImageItem(uid="sample_b", name="sample_b.png", path=image_path, image=image)
            classes = [
                LabelClass(1, "Root", "#ffcc33"),
                LabelClass(2, "Lateral", "#33a1ff"),
            ]
            annotations = {
                "sample_b": {
                    1: np.pad(np.ones((16, 4), dtype=np.uint8), ((8, 8), (14, 14))),
                    2: np.pad(np.ones((8, 6), dtype=np.uint8), ((10, 14), (2, 24))),
                }
            }
            project_path = tmp_root / "mini_root_lateral.oclp"
            save_project(
                project_path,
                [item],
                classes,
                annotations,
                predictions={},
                current_index=0,
                active_class_id=1,
                patch_target_class_id=1,
                next_class_id=3,
                model_path=None,
            )
            manifest = build_general_root_corpus(
                [CorpusSourceSpec(project_path=project_path, family="plate_bw_hades")],
                tmp_root / "corpus",
                random_seed=3,
            )
            sample = manifest["samples"][0]
            self.assertTrue(sample["tasks_available"]["root_binary"])
            self.assertTrue(sample["tasks_available"]["lateral_root"])
            self.assertTrue(sample["tasks_available"]["primary_root"])


if __name__ == "__main__":
    unittest.main()
