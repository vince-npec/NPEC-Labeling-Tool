from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from resources.consolidation_engine import (
    ConsolidationConfig,
    _build_metric_owned_regions,
    _overlay_metric_decomposition_on_image,
    build_metric_plot_preview,
    consolidate_measurements,
    generate_metric_timelapse,
)


class ConsolidationEngineTests(unittest.TestCase):
    def _make_preview_image(self, root: Path, name: str, value: int) -> str:
        image = np.full((96, 72, 3), value, dtype=np.uint8)
        image[:, :, 1] = np.clip(value + 30, 0, 255)
        path = root / name
        self.assertTrue(cv2.imwrite(str(path), image))
        return str(path)

    def _sample_df(self, root: Path) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "uid": ["a", "b", "c", "d"],
                "FrameIndex": [1, 1, 2, 2],
                "Timestamp": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
                "plant_id": ["plant_01", "plant_02", "plant_01", "plant_02"],
                "pixel_size_mm": [0.04119, 0.04119, 0.04119, 0.04119],
                "root_length_px": [120.0, 98.0, 141.0, 109.0],
                "PreviewImagePath": [
                    self._make_preview_image(root, "frame_001.png", 48),
                    self._make_preview_image(root, "frame_001.png", 48),
                    self._make_preview_image(root, "frame_002.png", 92),
                    self._make_preview_image(root, "frame_002.png", 92),
                ],
            }
        )

    def test_preview_uses_frame_index_when_timestamps_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df = self._sample_df(Path(tmp))
            frame = build_metric_plot_preview(df, metric_col="pixel_size_mm", width=640, height=360)
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.shape, (360, 640, 3))
            self.assertGreater(float(frame.mean()), 0.0)
            self.assertLess(float(frame.mean()), 255.0)

    def test_timelapse_uses_frame_index_when_timestamps_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            df = self._sample_df(root)
            output = root / "metric_timelapse.mp4"
            path = generate_metric_timelapse(
                consolidated_df=df,
                metric_col="pixel_size_mm",
                output_path=output,
                config=ConsolidationConfig(fps=2, width=640, height=360),
            )
            self.assertEqual(path, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_consolidation_prefers_original_image_over_mask_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks_dir = root / "segmentation_masks"
            masks_dir.mkdir()

            original = np.full((80, 120, 3), 180, dtype=np.uint8)
            original[:, :, 1] = 200
            original_path = root / "frame_001.png"
            self.assertTrue(cv2.imwrite(str(original_path), original))

            mask = np.zeros((80, 120), dtype=np.uint8)
            mask[15:60, 30:45] = 3
            mask[10:25, 25:50] = 2
            mask_path = masks_dir / "frame_001_mask.png"
            self.assertTrue(cv2.imwrite(str(mask_path), mask))

            lazy = pd.DataFrame(
                [
                    {
                        "uid": "path:test",
                        "image_name": "frame_001.png",
                        "output_mask": str(mask_path),
                        "pixel_size_mm": 0.04119,
                        "root_class_id": 3,
                        "shoot_class_id": 2,
                        "lateral_class_id": 4,
                    }
                ]
            )
            lazy.to_csv(masks_dir / "lazy_segmentation_run_details.csv", index=False)

            df, warnings = consolidate_measurements(masks_dir)
            self.assertFalse(df.empty)
            self.assertIn("PreviewImagePath", df.columns)
            self.assertIn("MaskPath", df.columns)
            self.assertTrue(str(df.iloc[0]["PreviewImagePath"]).endswith("frame_001.png"))
            self.assertTrue(str(df.iloc[0]["MaskPath"]).endswith("frame_001_mask.png"))
            self.assertLess(len(warnings), 5)

    def test_consolidation_can_resolve_preview_from_source_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "images"
            source_dir.mkdir()
            masks_dir = root / "segmentation_masks"
            masks_dir.mkdir()

            original = np.full((80, 120, 3), 140, dtype=np.uint8)
            original[:, :, 2] = 210
            original_path = source_dir / "frame_002.png"
            self.assertTrue(cv2.imwrite(str(original_path), original))

            mask = np.zeros((80, 120), dtype=np.uint8)
            mask[15:60, 30:45] = 3
            mask[16:34, 28:47] = 2
            mask_path = masks_dir / "frame_002_mask.png"
            self.assertTrue(cv2.imwrite(str(mask_path), mask))

            lazy = pd.DataFrame(
                [
                    {
                        "uid": "path:test2",
                        "image_name": "frame_002.png",
                        "SourceFile": str(original_path),
                        "output_mask": str(mask_path),
                        "pixel_size_mm": 0.04119,
                    }
                ]
            )
            lazy.to_csv(masks_dir / "lazy_segmentation_run_details.csv", index=False)

            df, warnings = consolidate_measurements(masks_dir, preview_search_roots=[source_dir])
            self.assertFalse(df.empty)
            self.assertTrue(str(df.iloc[0]["PreviewImagePath"]).endswith("frame_002.png"))
            self.assertIn("/images/", str(df.iloc[0]["PreviewImagePath"]))
            self.assertLess(len(warnings), 5)

    def test_lateral_metric_overlay_uses_plant_track_decomposition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.full((80, 80, 3), 180, dtype=np.uint8)
            mask = np.zeros((80, 80), dtype=np.uint8)
            mask[18:58, 10:24] = 4
            mask[18:58, 56:70] = 4
            mask_path = root / "lateral_mask.png"
            self.assertTrue(cv2.imwrite(str(mask_path), mask))

            current_rows = pd.DataFrame(
                [
                    {
                        "plant_id": "plant_01",
                        "bbox_x": 6,
                        "bbox_y": 12,
                        "bbox_w": 24,
                        "bbox_h": 52,
                        "lateral_total_length_mm": 10.0,
                    },
                    {
                        "plant_id": "plant_02",
                        "bbox_x": 50,
                        "bbox_y": 12,
                        "bbox_w": 24,
                        "bbox_h": 52,
                        "lateral_total_length_mm": 12.0,
                    },
                ]
            )

            overlaid = _overlay_metric_decomposition_on_image(
                image_rgb=image,
                mask_path=str(mask_path),
                current_rows=current_rows,
                metric_col="lateral_total_length_mm",
                cache={},
            )
            self.assertIsNotNone(overlaid)
            assert overlaid is not None
            left_px = overlaid[30, 16].astype(int)
            right_px = overlaid[30, 62].astype(int)
            self.assertGreater(int(left_px[0]), int(left_px[1]))
            self.assertGreater(int(left_px[0]), int(left_px[2]))
            self.assertGreater(int(right_px[1]), int(right_px[0]))
            self.assertGreaterEqual(int(right_px[1]), int(right_px[2]))
            self.assertFalse(np.array_equal(left_px, right_px))

    def test_metric_owned_regions_use_bbox_seedling_demarcation(self) -> None:
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[8:14, 18:28] = 2
        mask[8:14, 58:68] = 2
        mask[14:86, 22:24] = 3
        mask[14:86, 62:64] = 3
        mask[78:82, 24:63] = 3
        mask[36:38, 10:22] = 4
        mask[36:38, 64:82] = 4

        current_rows = pd.DataFrame(
            [
                {
                    "plant_id": "plant_01",
                    "bbox_x": 12,
                    "bbox_y": 8,
                    "bbox_w": 26,
                    "bbox_h": 80,
                    "shoot_bbox_x": 18,
                    "shoot_bbox_y": 8,
                    "shoot_bbox_w": 10,
                    "shoot_bbox_h": 6,
                    "root_class_id": 3,
                    "shoot_class_id": 2,
                    "lateral_class_id": 4,
                },
                {
                    "plant_id": "plant_02",
                    "bbox_x": 52,
                    "bbox_y": 8,
                    "bbox_w": 26,
                    "bbox_h": 80,
                    "shoot_bbox_x": 58,
                    "shoot_bbox_y": 8,
                    "shoot_bbox_w": 10,
                    "shoot_bbox_h": 6,
                    "root_class_id": 3,
                    "shoot_class_id": 2,
                    "lateral_class_id": 4,
                },
            ]
        )

        root_regions = _build_metric_owned_regions(mask, current_rows, "root_length_mm_clean")
        self.assertIsNotNone(root_regions)
        assert root_regions is not None
        self.assertTrue(root_regions["plant_01"][50, 23])
        self.assertFalse(root_regions["plant_02"][50, 23])
        self.assertTrue(root_regions["plant_02"][50, 63])
        self.assertFalse(root_regions["plant_01"][50, 63])

        lateral_regions = _build_metric_owned_regions(mask, current_rows, "lateral_total_length_mm")
        self.assertIsNotNone(lateral_regions)
        assert lateral_regions is not None
        self.assertTrue(lateral_regions["plant_01"][37, 14])
        self.assertFalse(lateral_regions["plant_02"][37, 14])
        self.assertTrue(lateral_regions["plant_02"][37, 76])
        self.assertFalse(lateral_regions["plant_01"][37, 76])

    def test_consolidation_refreshes_measurements_from_bbox_owned_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks_dir = root / "segmentation_masks"
            masks_dir.mkdir()

            original = np.full((100, 120, 3), 200, dtype=np.uint8)
            original_path = root / "frame_003.png"
            self.assertTrue(cv2.imwrite(str(original_path), original))

            mask = np.zeros((100, 120), dtype=np.uint8)
            mask[8:14, 18:28] = 2
            mask[8:14, 58:68] = 2
            mask[14:86, 22:24] = 3
            mask[14:86, 62:64] = 3
            mask[36:38, 10:22] = 4
            mask[36:38, 64:82] = 4
            mask_path = masks_dir / "frame_003_mask.png"
            self.assertTrue(cv2.imwrite(str(mask_path), mask))

            lazy = pd.DataFrame(
                [
                    {
                        "uid": "path:a",
                        "image_name": "frame_003.png",
                        "plant_id": "plant_01",
                        "bbox_x": 0,
                        "bbox_y": 0,
                        "bbox_w": 5,
                        "bbox_h": 5,
                        "shoot_bbox_x": 18,
                        "shoot_bbox_y": 8,
                        "shoot_bbox_w": 10,
                        "shoot_bbox_h": 6,
                        "pixel_size_mm": 0.05,
                        "root_length_px": 1.0,
                        "root_length_mm_raw": 0.05,
                        "lateral_total_length_mm": 0.0,
                        "root_class_id": 3,
                        "shoot_class_id": 2,
                        "lateral_class_id": 4,
                        "output_mask": str(mask_path),
                        "SourceFile": str(original_path),
                        "FrameIndex": 1,
                    },
                    {
                        "uid": "path:b",
                        "image_name": "frame_003.png",
                        "plant_id": "plant_02",
                        "bbox_x": 0,
                        "bbox_y": 0,
                        "bbox_w": 5,
                        "bbox_h": 5,
                        "shoot_bbox_x": 58,
                        "shoot_bbox_y": 8,
                        "shoot_bbox_w": 10,
                        "shoot_bbox_h": 6,
                        "pixel_size_mm": 0.05,
                        "root_length_px": 1.0,
                        "root_length_mm_raw": 0.05,
                        "lateral_total_length_mm": 0.0,
                        "root_class_id": 3,
                        "shoot_class_id": 2,
                        "lateral_class_id": 4,
                        "output_mask": str(mask_path),
                        "SourceFile": str(original_path),
                        "FrameIndex": 1,
                    },
                ]
            )
            lazy.to_csv(masks_dir / "lazy_segmentation_run_details.csv", index=False)

            df, _warnings = consolidate_measurements(masks_dir, preview_search_roots=[root])
            self.assertFalse(df.empty)
            self.assertIn("ownership_recomputed_from_bbox_masks", df.columns)
            self.assertTrue(bool(df["ownership_recomputed_from_bbox_masks"].all()))
            self.assertTrue((pd.to_numeric(df["root_length_px"], errors="coerce") > 1.0).all())
            self.assertIn("primary_root_length_px", df.columns)
            self.assertIn("total_root_length_px", df.columns)
            self.assertIn("total_root_length_mm_clean", df.columns)
            self.assertTrue((pd.to_numeric(df["primary_root_length_px"], errors="coerce") > 1.0).all())
            self.assertTrue(
                (
                    pd.to_numeric(df["total_root_length_px"], errors="coerce")
                    > pd.to_numeric(df["primary_root_length_px"], errors="coerce")
                ).all()
            )
            self.assertTrue((pd.to_numeric(df["shoot_area_px"], errors="coerce") == 60).all())
            self.assertTrue((pd.to_numeric(df["bbox_w"], errors="coerce") > 5).all())


if __name__ == "__main__":
    unittest.main()
