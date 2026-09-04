from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from resources.models import DatasetImageItem, LabelClass
from resources.project_io import save_project
from resources.yolo_tip_anchors import (
    build_external_seed_and_tip_priors_from_yolo_json,
    export_yolo26_anchor_dataset,
)


class YoloTipAnchorsTests(unittest.TestCase):
    def _image_and_annotations(self) -> tuple[np.ndarray, dict[int, np.ndarray], list[LabelClass]]:
        image = np.full((32, 32, 3), 235, dtype=np.uint8)
        root = np.zeros((32, 32), dtype=np.uint8)
        shoot = np.zeros((32, 32), dtype=np.uint8)
        seed = np.zeros((32, 32), dtype=np.uint8)
        lateral = np.zeros((32, 32), dtype=np.uint8)
        # Plant 1
        shoot[2:5, 6:10] = 1
        seed[5:7, 7:9] = 1
        root[7:25, 8] = 1
        lateral[14, 6:8] = 1
        # Plant 2
        shoot[2:5, 22:26] = 1
        seed[5:7, 23:25] = 1
        root[7:27, 24] = 1
        lateral[16, 24:27] = 1
        image[root > 0] = (40, 120, 220)
        image[shoot > 0] = (40, 180, 60)
        image[seed > 0] = (210, 120, 40)
        annotations = {1: root, 2: shoot, 3: seed, 4: lateral}
        classes = [
            LabelClass(1, "Root", "#ffcc00"),
            LabelClass(2, "Shoot", "#00ff66"),
            LabelClass(3, "Seed", "#ff5a5f"),
            LabelClass(4, "Lateral", "#55ccff"),
        ]
        return image, annotations, classes

    def test_export_yolo26_anchor_dataset_writes_detect_and_pose_labels(self) -> None:
        image, annotations, classes = self._image_and_annotations()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_path = root / "toy.oclp"
            item = DatasetImageItem(uid="frame_001", name="frame_001.png", path=root / "frame_001.png", image=image)
            save_project(
                project_path,
                dataset_items=[item],
                classes=classes,
                annotations={"frame_001": annotations},
                predictions={},
                current_index=0,
                active_class_id=1,
                patch_target_class_id=1,
                next_class_id=5,
                model_path=None,
            )

            summary = export_yolo26_anchor_dataset(project_path, root / "yolo_out", val_stride=5, expected_track_count=2)
            self.assertEqual(summary["image_count"], 1)
            detect_label = root / "yolo_out" / "detect" / "labels" / "val" / "frame_001.txt"
            pose_label = root / "yolo_out" / "pose" / "labels" / "val" / "frame_001.txt"
            self.assertTrue(detect_label.exists())
            self.assertTrue(pose_label.exists())
            detect_lines = [line for line in detect_label.read_text(encoding="utf-8").splitlines() if line.strip()]
            pose_lines = [line for line in pose_label.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(detect_lines), 4)
            self.assertEqual(len(pose_lines), 2)
            self.assertTrue((root / "yolo_out" / "detect" / "dataset.yaml").exists())
            self.assertTrue((root / "yolo_out" / "pose" / "dataset.yaml").exists())

    def test_build_external_seed_and_tip_priors_from_yolo_json_maps_tracks(self) -> None:
        items = [
            DatasetImageItem(uid="frame_001", name="frame_001.png", path=Path("/tmp/frame_001.png"), image=np.zeros((32, 32, 3), dtype=np.uint8)),
            DatasetImageItem(uid="frame_002", name="frame_002.png", path=Path("/tmp/frame_002.png"), image=np.zeros((32, 32, 3), dtype=np.uint8)),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "anchors.json"
            json_path.write_text(
                """
{
  "frames": [
    {
      "uid": "frame_001",
      "detections": [
        {"class_name": "crown", "track_id": "plant_01", "bbox_xyxy": [4, 2, 10, 8]},
        {"class_name": "crown", "track_id": "plant_02", "bbox_xyxy": [20, 2, 26, 8]},
        {"class_name": "primary_tip", "bbox_xyxy": [7, 22, 9, 24]},
        {"class_name": "primary_tip", "bbox_xyxy": [23, 24, 25, 26]}
      ]
    },
    {
      "uid": "frame_002",
      "detections": [
        {"class_name": "crown", "track_id": "plant_01", "bbox_xyxy": [5, 2, 11, 8]},
        {"class_name": "crown", "track_id": "plant_02", "bbox_xyxy": [19, 2, 25, 8]},
        {"class_name": "primary_tip", "bbox_xyxy": [8, 24, 10, 26]},
        {"class_name": "primary_tip", "bbox_xyxy": [22, 26, 24, 28]}
      ]
    }
  ]
}
""".strip(),
                encoding="utf-8",
            )
            seed, tip_priors, summary = build_external_seed_and_tip_priors_from_yolo_json(json_path, items)
            self.assertIsNotNone(seed)
            assert seed is not None
            self.assertEqual(summary["track_count"], 2)
            self.assertEqual(seed["track_ids"], ["plant_01", "plant_02"])
            self.assertEqual(seed["track_bboxes"]["plant_01"][0], (4, 2, 6, 6))
            self.assertEqual(seed["track_bboxes"]["plant_02"][1], (19, 2, 6, 6))
            self.assertIn(0, tip_priors)
            self.assertIn("plant_01", tip_priors[0])
            self.assertIn("plant_02", tip_priors[1])


if __name__ == "__main__":
    unittest.main()
