from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from resources.general_root_starter.corpus import CorpusSourceSpec, build_general_root_corpus
from resources.models import DatasetImageItem, LabelClass
from resources.project_io import save_project


class GeneralRootStarterUnifiedTests(unittest.TestCase):
    def test_unified_mask_reader_supports_direct_png_masks(self):
        from resources.general_root_starter.unified_model import _read_masks

        with tempfile.TemporaryDirectory(prefix="grs_mask_reader_test_") as tmp_dir:
            mask_path = Path(tmp_dir) / "mask.png"
            mask = np.zeros((6, 8), dtype=np.uint8)
            mask[1:5, 3:5] = 255
            Image.fromarray(mask).save(mask_path)
            payload = _read_masks(mask_path)
            self.assertIn("root_binary", payload)
            self.assertEqual(int(np.count_nonzero(payload["root_binary"])), 8)

    def test_unified_checkpoint_round_trip(self):
        try:
            from resources.general_root_starter.unified_model import (
                load_corpus_manifest,
                load_unified_checkpoint,
                train_unified_general_root_starter,
            )
        except Exception as exc:
            if "torch" in str(exc).lower():
                self.skipTest("PyTorch not available in this runtime.")
            raise

        with tempfile.TemporaryDirectory(prefix="grs_unified_test_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            items = []
            annotations = {}
            classes = [
                LabelClass(1, "Root", "#ffcc33"),
                LabelClass(2, "Shoot", "#55dd77"),
            ]
            for idx in range(3):
                image = np.zeros((64, 64, 3), dtype=np.uint8)
                image[..., 1] = 80 + idx * 20
                uid = f"sample_{idx}"
                image_path = tmp_root / f"{uid}.png"
                Image.fromarray(image).save(image_path)
                item = DatasetImageItem(uid=uid, name=f"{uid}.png", path=image_path, image=image)
                root = np.zeros((64, 64), dtype=np.uint8)
                root[10:58, 30:34] = 1
                shoot = np.zeros((64, 64), dtype=np.uint8)
                shoot[4:14, 24:40] = 1
                annotations[uid] = {1: root, 2: shoot}
                items.append(item)
            project_path = tmp_root / "mini.oclp"
            save_project(
                project_path,
                items,
                classes,
                annotations,
                predictions={},
                current_index=0,
                active_class_id=1,
                patch_target_class_id=1,
                next_class_id=3,
                model_path=None,
            )
            corpus_dir = tmp_root / "corpus"
            build_general_root_corpus(
                [CorpusSourceSpec(project_path=project_path, family="plate_bw_hades")],
                corpus_dir,
                train_fraction=0.67,
                val_fraction=0.33,
                random_seed=2,
            )
            checkpoint_path = tmp_root / "general_root_starter_unified.pt"
            try:
                summary = train_unified_general_root_starter(
                    corpus_dir,
                    checkpoint_path,
                    epochs=1,
                    batch_size=1,
                    patch_size=64,
                    base_filters=8,
                    repeat_factor=1,
                    device_mode="cpu",
                )
            except RuntimeError as exc:
                if "PyTorch is required" in str(exc):
                    self.skipTest("PyTorch not available in this runtime.")
                raise
            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(summary["tasks"][0], "root_binary")
            self.assertIn("stopped_early", summary)
            self.assertIn("task_loss_weights", summary)
            self.assertIn("curriculum_stages", summary)
            self.assertIn("curriculum_mode", summary)
            _model, metadata = load_unified_checkpoint(checkpoint_path)
            manifest = load_corpus_manifest(corpus_dir)
            self.assertIn("tasks", metadata)
            self.assertIn("weight_decay", metadata)
            self.assertIn("curriculum_stages", metadata)
            self.assertEqual(len(manifest["samples"]), 3)

    def test_unified_training_supports_root_only_tasks(self):
        try:
            from resources.general_root_starter.unified_model import (
                load_unified_checkpoint,
                train_unified_general_root_starter,
            )
        except Exception as exc:
            if "torch" in str(exc).lower():
                self.skipTest("PyTorch not available in this runtime.")
            raise

        with tempfile.TemporaryDirectory(prefix="grs_unified_root_only_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            items = []
            annotations = {}
            classes = [LabelClass(1, "Root", "#ffcc33"), LabelClass(2, "Lateral Root", "#25c7ff")]
            for idx in range(2):
                image = np.zeros((48, 48, 3), dtype=np.uint8)
                image[..., 0] = 40 + idx * 30
                uid = f"root_only_{idx}"
                image_path = tmp_root / f"{uid}.png"
                Image.fromarray(image).save(image_path)
                item = DatasetImageItem(uid=uid, name=f"{uid}.png", path=image_path, image=image)
                root = np.zeros((48, 48), dtype=np.uint8)
                root[8:40, 22:26] = 1
                lateral = np.zeros((48, 48), dtype=np.uint8)
                lateral[20:24, 8:22] = 1
                annotations[uid] = {1: root, 2: lateral}
                items.append(item)
            project_path = tmp_root / "root_only.oclp"
            save_project(
                project_path,
                items,
                classes,
                annotations,
                predictions={},
                current_index=0,
                active_class_id=1,
                patch_target_class_id=1,
                next_class_id=3,
                model_path=None,
            )
            corpus_dir = tmp_root / "corpus"
            build_general_root_corpus(
                [CorpusSourceSpec(project_path=project_path, family="plate_bw_hades")],
                corpus_dir,
                train_fraction=0.5,
                val_fraction=0.5,
                random_seed=2,
            )
            checkpoint_path = tmp_root / "general_root_starter_unified_root_only.pt"
            try:
                summary = train_unified_general_root_starter(
                    corpus_dir,
                    checkpoint_path,
                    epochs=1,
                    batch_size=1,
                    patch_size=48,
                    base_filters=8,
                    repeat_factor=1,
                    device_mode="cpu",
                    tasks=("root_binary", "primary_root", "lateral_root"),
                )
            except RuntimeError as exc:
                if "PyTorch is required" in str(exc):
                    self.skipTest("PyTorch not available in this runtime.")
                raise
            self.assertEqual(summary["tasks"], ["root_binary", "primary_root", "lateral_root"])
            _model, metadata = load_unified_checkpoint(checkpoint_path)
            self.assertEqual(metadata.get("tasks"), ["root_binary", "primary_root", "lateral_root"])


if __name__ == "__main__":
    unittest.main()
