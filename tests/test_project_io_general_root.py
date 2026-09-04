from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from resources.models import DatasetImageItem, LabelClass
from resources.project_io import load_project, save_project


class ProjectIOGeneralRootTests(unittest.TestCase):
    def test_round_trip_preserves_combined_lazy_mode_and_expected_seedlings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "img.png"
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            Image.fromarray(image).save(image_path)
            item = DatasetImageItem(uid="u1", name=image_path.name, path=image_path, image=image)
            project_path = tmp_dir / "combined.oclp"

            save_project(
                project_path=project_path,
                dataset_items=[item],
                classes=[LabelClass(1, "Root", "#ffd166")],
                annotations={"u1": {1: np.zeros((4, 4), dtype=np.uint8)}},
                predictions={},
                current_index=0,
                active_class_id=1,
                patch_target_class_id=1,
                next_class_id=2,
                model_path=None,
                pipeline_lazy_analysis_mode="combined",
                pipeline_lazy_standard_expected_plants=3,
            )

            loaded = load_project(project_path)

            self.assertEqual(loaded["pipeline_lazy_analysis_mode"], "combined")
            self.assertEqual(loaded["pipeline_lazy_standard_expected_plants"], 3)

    def test_round_trip_preserves_general_root_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "img.png"
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
            item = DatasetImageItem(uid="u1", name="img.png", path=image_path, image=np.zeros((8, 8, 3), dtype=np.uint8))
            classes = [
                LabelClass(1, "Seed", "#f59e0b"),
                LabelClass(2, "Shoot", "#56f39a"),
                LabelClass(3, "Root", "#ffd166"),
                LabelClass(4, "Lateral Root", "#3fc1ff"),
            ]
            annotations = {
                "u1": {
                    1: np.zeros((8, 8), dtype=np.uint8),
                    2: np.zeros((8, 8), dtype=np.uint8),
                    3: np.ones((8, 8), dtype=np.uint8),
                    4: np.zeros((8, 8), dtype=np.uint8),
                }
            }
            project_path = tmp_dir / "test.oclp"
            calibration = {
                "version": 1,
                "preferred_expert_key": "potato_rgb",
                "expert_score_bias": {"potato_rgb": 0.2, "plate_bw_hades": -0.1},
                "selected_uids": ["u1"],
            }
            identity = {
                "source_name": "img.png",
                "source_path": str(image_path),
                "source_date": "2024-07-15T09:34:10",
                "source_date_source": "filename",
                "plate_id": "152_RP043_g23",
                "allocated_name": "152_RP043_g23__20240715_093410.png",
                "confidence": 0.99,
                "status": "accepted",
                "evidence": [{"kind": "vision_text", "text": "152 RP043 g23"}],
            }
            save_project(
                project_path=project_path,
                dataset_items=[item],
                classes=classes,
                annotations=annotations,
                predictions={},
                current_index=0,
                active_class_id=2,
                patch_target_class_id=2,
                next_class_id=5,
                model_path=None,
                inference_backend="pyphenotyper",
                pyphenotyper_preset_key="general_root_starter_auto",
                pyphenotyper_pipeline_dir=tmp_dir / "pipeline",
                pyphenotyper_root_model_path=tmp_dir / "root.h5",
                pyphenotyper_shoot_model_path=tmp_dir / "shoot.h5",
                pyphenotyper_lucifer_gan_gap_repair_enabled=True,
                pyphenotyper_lucifer_gan_gap_repair_dir=tmp_dir / "lucifer_gan",
                pyphenotyper_preserve_root_on_shoot_overlap=True,
                pyphenotyper_shoot_color_rescue_mode="lucifer_green",
                pyphenotyper_pixel_size_mm=0.04118616144975288,
                analytics_pixel_size_mm=0.02663809523809524,
                analytics_root_class_id=3,
                analytics_lateral_class_id=4,
                analytics_seed_class_id=1,
                analytics_shoot_class_id=2,
                analytics_root_ownership_mode="temporal_graph",
                analytics_freeze_ambiguous_ownership=True,
                pipeline_temporal_primary_lock_enabled=True,
                pipeline_lazy_analysis_mode="potato_endpoint",
                pipeline_lazy_standard_expected_plants=5,
                pipeline_lazy_measure_class_ids="1,3",
                pipeline_lazy_shoot_class_ids="2,4",
                pipeline_lazy_video_mask_only=False,
                pipeline_lazy_root_metric_mode="primary",
                pipeline_lazy_shoot_rgb_rescue_enabled=True,
                pipeline_lazy_plate_identity_enabled=False,
                pyphenotyper_general_starter_calibration=calibration,
                image_identity_by_uid={"u1": identity},
            )
            loaded = load_project(project_path)
            self.assertEqual(loaded["pyphenotyper_preset_key"], "general_root_starter_auto")
            self.assertEqual(loaded["pyphenotyper_general_starter_calibration"], calibration)
            self.assertEqual(loaded["pyphenotyper_pipeline_dir"], tmp_dir / "pipeline")
            self.assertEqual(loaded["pyphenotyper_root_model_path"], tmp_dir / "root.h5")
            self.assertEqual(loaded["pyphenotyper_shoot_model_path"], tmp_dir / "shoot.h5")
            self.assertTrue(bool(loaded["pyphenotyper_lucifer_gan_gap_repair_enabled"]))
            self.assertEqual(loaded["pyphenotyper_lucifer_gan_gap_repair_dir"], tmp_dir / "lucifer_gan")
            self.assertTrue(bool(loaded["pyphenotyper_preserve_root_on_shoot_overlap"]))
            self.assertEqual(loaded["pyphenotyper_shoot_color_rescue_mode"], "lucifer_green")
            self.assertAlmostEqual(float(loaded["pyphenotyper_pixel_size_mm"]), 0.04118616144975288, places=10)
            self.assertAlmostEqual(float(loaded["analytics_pixel_size_mm"]), 0.02663809523809524, places=10)
            self.assertTrue(bool(loaded["analytics_class_settings_present"]))
            self.assertEqual(loaded["analytics_root_class_id"], 3)
            self.assertEqual(loaded["analytics_lateral_class_id"], 4)
            self.assertEqual(loaded["analytics_seed_class_id"], 1)
            self.assertEqual(loaded["analytics_shoot_class_id"], 2)
            self.assertEqual(loaded["analytics_root_ownership_mode"], "temporal_graph")
            self.assertTrue(bool(loaded["analytics_freeze_ambiguous_ownership"]))
            self.assertTrue(bool(loaded["pipeline_temporal_primary_lock_enabled"]))
            self.assertEqual(loaded["pipeline_lazy_analysis_mode"], "potato_endpoint")
            self.assertEqual(loaded["pipeline_lazy_standard_expected_plants"], 5)
            self.assertEqual(loaded["pipeline_lazy_measure_class_ids"], "1,3")
            self.assertEqual(loaded["pipeline_lazy_shoot_class_ids"], "2,4")
            self.assertFalse(bool(loaded["pipeline_lazy_video_mask_only"]))
            self.assertEqual(loaded["pipeline_lazy_root_metric_mode"], "primary")
            self.assertTrue(bool(loaded["pipeline_lazy_shoot_rgb_rescue_enabled"]))
            self.assertFalse(bool(loaded["pipeline_lazy_plate_identity_enabled"]))
            self.assertEqual(loaded["image_identity_by_uid"], {"u1": identity})

    def test_load_old_project_without_analytics_class_metadata_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            project_path = tmp_dir / "old_project.oclp"
            metadata = {
                "version": 2,
                "classes": [
                    {"id": 1, "name": "Root", "color": "#ffd166"},
                    {"id": 2, "name": "Shoot", "color": "#56f39a"},
                ],
                "class_visibility": {"1": True, "2": True},
                "dataset": [],
                "current_index": 0,
                "active_class_id": 1,
                "patch_target_class_id": 1,
                "next_class_id": 3,
                "model_path": None,
            }
            with zipfile.ZipFile(project_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("project.json", json.dumps(metadata))

            loaded = load_project(project_path)
            self.assertFalse(bool(loaded["analytics_class_settings_present"]))
            self.assertEqual(loaded["pipeline_lazy_standard_expected_plants"], 5)
            self.assertIsNone(loaded["analytics_root_class_id"])
            self.assertIsNone(loaded["analytics_lateral_class_id"])
            self.assertIsNone(loaded["analytics_seed_class_id"])
            self.assertIsNone(loaded["analytics_shoot_class_id"])
            self.assertEqual(loaded["analytics_root_ownership_mode"], "temporal_graph")
            self.assertTrue(bool(loaded["analytics_freeze_ambiguous_ownership"]))
            self.assertTrue(bool(loaded["pipeline_temporal_primary_lock_enabled"]))
            self.assertEqual(loaded["pipeline_lazy_analysis_mode"], "combined")
            self.assertEqual(loaded["pipeline_lazy_shoot_class_ids"], "2")
            self.assertFalse(bool(loaded["pipeline_lazy_shoot_rgb_rescue_enabled"]))
            self.assertTrue(bool(loaded["pipeline_lazy_plate_identity_enabled"]))
            self.assertEqual(loaded["image_identity_by_uid"], {})

    def test_round_trip_uses_embedded_image_when_original_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_array = np.zeros((6, 7, 3), dtype=np.uint8)
            image_array[..., 1] = 120
            image_path = tmp_dir / "portable.png"
            Image.fromarray(image_array).save(image_path)
            item = DatasetImageItem(uid="portable_u1", name="portable.png", path=image_path, image=image_array.copy())
            classes = [
                LabelClass(1, "Root", "#ffd166"),
                LabelClass(2, "Shoot", "#56f39a"),
            ]
            annotations = {
                "portable_u1": {
                    1: np.ones((6, 7), dtype=np.uint8),
                    2: np.zeros((6, 7), dtype=np.uint8),
                }
            }
            project_path = tmp_dir / "portable.oclp"
            save_project(
                project_path=project_path,
                dataset_items=[item],
                classes=classes,
                annotations=annotations,
                predictions={},
                current_index=0,
                active_class_id=1,
                patch_target_class_id=1,
                next_class_id=3,
                model_path=None,
            )

            image_path.unlink()
            loaded = load_project(project_path)
            self.assertEqual(len(loaded["dataset_items"]), 1)
            restored_item = loaded["dataset_items"][0]
            self.assertEqual(restored_item.uid, "portable_u1")
            self.assertEqual(restored_item.image.shape, image_array.shape)
            np.testing.assert_array_equal(restored_item.image, image_array)
            self.assertEqual(int(np.count_nonzero(loaded["annotations"]["portable_u1"][1])), 42)
            self.assertEqual(loaded["missing_paths"], [])

    def test_round_trip_preserves_in_memory_image_state_over_source_file_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            disk_image = np.zeros((4, 5, 3), dtype=np.uint8)
            disk_image[0, 0] = [255, 0, 0]
            image_path = tmp_dir / "edited.png"
            Image.fromarray(disk_image).save(image_path)

            edited_image = np.rot90(disk_image, 1).copy()
            item = DatasetImageItem(uid="edited_u1", name="edited.png", path=image_path, image=edited_image)
            classes = [LabelClass(1, "Root", "#ffd166")]
            annotations = {"edited_u1": {1: np.zeros(edited_image.shape[:2], dtype=np.uint8)}}
            project_path = tmp_dir / "edited.oclp"

            save_project(
                project_path=project_path,
                dataset_items=[item],
                classes=classes,
                annotations=annotations,
                predictions={},
                current_index=0,
                active_class_id=1,
                patch_target_class_id=1,
                next_class_id=2,
                model_path=None,
            )

            loaded = load_project(project_path)
            restored_item = loaded["dataset_items"][0]
            np.testing.assert_array_equal(restored_item.image, edited_image)
            self.assertNotEqual(restored_item.image.shape, disk_image.shape)


if __name__ == "__main__":
    unittest.main()
