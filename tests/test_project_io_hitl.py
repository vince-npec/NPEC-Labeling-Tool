from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from resources.models import DatasetImageItem, LabelClass
from resources.project_io import load_project, save_project


class ProjectIOHitlTests(unittest.TestCase):
    def test_round_trip_preserves_hitl_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "img.png"
            Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(image_path)
            item = DatasetImageItem(uid="u1", name="img.png", path=image_path, image=np.zeros((10, 10, 3), dtype=np.uint8))
            classes = [
                LabelClass(1, "Root", "#ffd166"),
                LabelClass(2, "Shoot", "#56f39a"),
            ]
            annotations = {
                "u1": {
                    1: np.zeros((10, 10), dtype=np.uint8),
                    2: np.zeros((10, 10), dtype=np.uint8),
                }
            }
            predictions = {"u1": np.ones((10, 10), dtype=np.uint8)}
            prediction_metadata = {
                "u1": {
                    "confidence_score": 0.91,
                    "uncertainty_score": 0.09,
                    "prediction_coverage_fraction": 0.25,
                    "backend": "keras",
                }
            }
            pseudo = {"u1": np.ones((10, 10), dtype=np.uint8)}
            review = {"u1": {"status": "pending", "confidence_score": 0.91}}
            project_path = tmp_dir / "test_hitl.oclp"
            save_project(
                project_path=project_path,
                dataset_items=[item],
                classes=classes,
                annotations=annotations,
                predictions=predictions,
                current_index=0,
                active_class_id=1,
                patch_target_class_id=1,
                next_class_id=3,
                model_path=None,
                prediction_metadata_by_uid=prediction_metadata,
                hitl_pseudo_labels=pseudo,
                hitl_review_state=review,
                hitl_session_input_folder=tmp_dir,
                hitl_session_template_preset_key="hades_lucifer_builtin",
                hitl_synthetic_output_dir=tmp_dir / "hitl_output",
                hitl_long_session_cycle_count=4,
            )
            loaded = load_project(project_path)
            self.assertIn("u1", loaded["prediction_metadata_by_uid"])
            self.assertAlmostEqual(float(loaded["prediction_metadata_by_uid"]["u1"]["confidence_score"]), 0.91, places=6)
            self.assertIn("u1", loaded["hitl_pseudo_labels"])
            self.assertEqual(int(np.count_nonzero(loaded["hitl_pseudo_labels"]["u1"])), 100)
            self.assertEqual(loaded["hitl_review_state"]["u1"]["status"], "pending")
            self.assertEqual(Path(loaded["hitl_session_input_folder"]), tmp_dir)
            self.assertEqual(loaded["hitl_session_template_preset_key"], "hades_lucifer_builtin")
            self.assertEqual(Path(loaded["hitl_synthetic_output_dir"]), tmp_dir / "hitl_output")
            self.assertEqual(int(loaded["hitl_long_session_cycle_count"]), 4)


if __name__ == "__main__":
    unittest.main()
