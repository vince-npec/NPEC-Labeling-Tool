from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from resources.mask_total_root_analysis import build_mask_total_dataframe
from resources.occlusion_aware_root_analysis import (
    build_compartmentalized_occlusion_dataframe,
    build_whole_plate_occlusion_dataframe,
    write_occlusion_aware_outputs,
)


class OcclusionAwareRootAnalysisTests(unittest.TestCase):
    def _write_image(self, path: Path) -> None:
        image = np.full((100, 100, 3), 235, dtype=np.uint8)
        image[20:36, 42:58] = [72, 132, 38]
        Image.fromarray(image).save(path)

    def test_whole_plate_carries_forward_roots_hidden_after_occlusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            records: list[dict[str, object]] = []
            for frame_index in range(2):
                image_path = tmp_dir / f"plate_t{frame_index}.png"
                self._write_image(image_path)
                mask = np.zeros((100, 100), dtype=np.uint8)
                if frame_index == 0:
                    mask[30:80, 48] = 1
                    mask[56, 48:72] = 3
                mask_path = tmp_dir / f"plate_t{frame_index}_mask.png"
                Image.fromarray(mask, mode="L").save(mask_path)
                records.append(
                    {
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "output_mask": str(mask_path),
                        "meta": {
                            "series": "synthetic",
                            "petri": "plate_01",
                            "timestamp": f"2026-07-0{frame_index + 1} 08:00:00",
                            "frame_index": frame_index,
                            "relative_folder": ".",
                        },
                        "details": {"pixel_size_mm": 1.0},
                    }
                )

            mask_df = build_mask_total_dataframe(
                records,
                class_ids=[1, 3],
                shoot_class_ids=[2],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
                shoot_rgb_rescue=True,
            )
            enriched = build_whole_plate_occlusion_dataframe(mask_df)

            first = enriched.iloc[0]
            second = enriched.iloc[1]
            self.assertGreater(float(first["visible_total_root_length_mm"]), 0.0)
            self.assertEqual(float(second["visible_total_root_length_mm"]), 0.0)
            self.assertGreater(float(second["occlusion_aware_total_root_length_mm"]), 0.0)
            self.assertAlmostEqual(
                float(second["occlusion_aware_total_root_length_mm"]),
                float(first["occlusion_aware_total_root_length_mm"]),
            )
            self.assertGreater(float(second["occlusion_inferred_total_root_length_mm"]), 0.0)
            self.assertGreater(int(first["shoot_area_green_only_px"]), 0)

    def test_occlusion_outputs_use_ownership_accepted_shoot_mask_and_area(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "plate.png"
            self._write_image(image_path)

            raw_mask = np.zeros((100, 100), dtype=np.uint8)
            raw_mask[30:80, 48] = 1
            raw_mask[2:8, 90:98] = 2
            raw_path = tmp_dir / "raw.png"
            Image.fromarray(raw_mask, mode="L").save(raw_path)

            accepted_mask = raw_mask.copy()
            accepted_mask[accepted_mask == 2] = 0
            accepted_mask[18:24, 45:51] = 2
            accepted_path = tmp_dir / "accepted.png"
            Image.fromarray(accepted_mask, mode="L").save(accepted_path)

            mask_df = build_mask_total_dataframe(
                [
                    {
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "output_mask": str(raw_path),
                        "meta": {
                            "series": "synthetic",
                            "petri": "plate_accepted",
                            "timestamp": "2026-07-01 08:00:00",
                            "frame_index": 0,
                            "relative_folder": ".",
                        },
                        "details": {"pixel_size_mm": 1.0},
                    }
                ],
                class_ids=[1, 3],
                shoot_class_ids=[2],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
                shoot_rgb_rescue=False,
            )
            ownership_df = pd.DataFrame(
                [
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_accepted",
                        "Timestamp": "2026-07-01 08:00:00",
                        "FrameIndex": 0,
                        "plant_id": "plant_01",
                        "SourceFile": str(image_path),
                        "OutputMaskPath": str(raw_path),
                        "OwnershipDisplayMaskPath": str(accepted_path),
                        "shoot_area_px": 12,
                        "shoot_area_mm2": 12.0,
                        "shoot_measurement_source": "mask_class",
                        "ownership_measurement_valid": True,
                    },
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_accepted",
                        "Timestamp": "2026-07-01 08:00:00",
                        "FrameIndex": 0,
                        "plant_id": "plant_02",
                        "SourceFile": str(image_path),
                        "OutputMaskPath": str(raw_path),
                        "OwnershipDisplayMaskPath": str(accepted_path),
                        "shoot_area_px": 24,
                        "shoot_area_mm2": 24.0,
                        "shoot_measurement_source": "hades_bw_crown_local_cv",
                        "ownership_measurement_valid": True,
                    },
                ]
            )

            result = write_occlusion_aware_outputs(
                mask_df,
                output_dir=tmp_dir / "out",
                ownership_df=ownership_df,
                plant_count=2,
                generate_videos=False,
            )

            whole = pd.read_csv(result.whole_plate_detail_csv)
            compartment = pd.read_csv(result.compartment_detail_csv)
            self.assertEqual(whole.loc[0, "OwnershipDisplayMaskPath"], str(accepted_path))
            self.assertEqual(int(whole.loc[0, "shoot_area_green_only_px"]), 36)
            self.assertEqual(
                whole.loc[0, "shoot_green_filter_source"],
                "mask_class,hades_bw_crown_local_cv",
            )
            self.assertTrue(
                compartment["OwnershipDisplayMaskPath"].astype(str).eq(str(accepted_path)).all()
            )

    def test_compartmentalized_carry_forward_is_lane_specific(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            records: list[dict[str, object]] = []
            for frame_index in range(2):
                image_path = tmp_dir / f"lane_t{frame_index}.png"
                self._write_image(image_path)
                mask = np.zeros((100, 100), dtype=np.uint8)
                mask[30:80, 25] = 1
                if frame_index == 0:
                    mask[30:80, 75] = 1
                mask_path = tmp_dir / f"lane_t{frame_index}_mask.png"
                Image.fromarray(mask, mode="L").save(mask_path)
                records.append(
                    {
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "output_mask": str(mask_path),
                        "meta": {
                            "series": "synthetic",
                            "petri": "plate_lanes",
                            "timestamp": f"2026-07-0{frame_index + 1} 08:00:00",
                            "frame_index": frame_index,
                            "relative_folder": ".",
                        },
                        "details": {"pixel_size_mm": 1.0},
                    }
                )

            mask_df = build_mask_total_dataframe(
                records,
                class_ids=[1, 3],
                shoot_class_ids=[2],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
                shoot_rgb_rescue=True,
            )
            whole = build_whole_plate_occlusion_dataframe(mask_df)
            ownership = pd.DataFrame(
                [
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_lanes",
                        "Timestamp": "2026-07-01 08:00:00",
                        "FrameIndex": 0,
                        "plant_id": "plant_01",
                        "shoot_bbox_x": 18,
                        "shoot_bbox_w": 14,
                        "root_length_mm_clean": 50.0,
                        "lateral_total_length_mm": 0.0,
                    },
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_lanes",
                        "Timestamp": "2026-07-02 08:00:00",
                        "FrameIndex": 1,
                        "plant_id": "plant_01",
                        "shoot_bbox_x": 18,
                        "shoot_bbox_w": 14,
                        "root_length_mm_clean": 50.0,
                        "lateral_total_length_mm": 0.0,
                    },
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_lanes",
                        "Timestamp": "2026-07-01 08:00:00",
                        "FrameIndex": 0,
                        "plant_id": "plant_02",
                        "shoot_bbox_x": 68,
                        "shoot_bbox_w": 14,
                        "root_length_mm_clean": 50.0,
                        "lateral_total_length_mm": 0.0,
                    },
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_lanes",
                        "Timestamp": "2026-07-02 08:00:00",
                        "FrameIndex": 1,
                        "plant_id": "plant_02",
                        "shoot_bbox_x": 68,
                        "shoot_bbox_w": 14,
                        "root_length_mm_clean": 0.0,
                        "lateral_total_length_mm": 0.0,
                    },
                ]
            )
            compartment = build_compartmentalized_occlusion_dataframe(
                whole,
                ownership_df=ownership,
                plant_count=2,
            )

            plant_01 = compartment[compartment["plant_id"] == "plant_01"].reset_index(drop=True)
            plant_02 = compartment[compartment["plant_id"] == "plant_02"].reset_index(drop=True)
            self.assertGreater(float(plant_01.loc[1, "visible_total_root_length_mm"]), 0.0)
            self.assertGreater(float(plant_02.loc[0, "visible_total_root_length_mm"]), 0.0)
            self.assertEqual(float(plant_02.loc[1, "visible_total_root_length_mm"]), 0.0)
            self.assertGreater(float(plant_02.loc[1, "occlusion_aware_total_root_length_mm"]), 0.0)
            self.assertGreater(float(plant_02.loc[1, "occlusion_inferred_total_root_length_mm"]), 0.0)

    def test_invalid_ownership_length_cannot_poison_later_occlusion_carry_forward(self) -> None:
        whole = pd.DataFrame(
            [
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_conflict",
                    "Timestamp": "2026-07-01 08:00:00",
                    "FrameIndex": 0,
                    "mask_width": 100,
                }
            ]
        )
        ownership = pd.DataFrame(
            [
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_conflict",
                    "Timestamp": "2026-07-01 08:00:00",
                    "FrameIndex": 0,
                    "plant_id": "plant_01",
                    "shoot_bbox_x": 42,
                    "shoot_bbox_w": 16,
                    "primary_root_length_mm_clean": 8.0,
                    "lateral_total_length_mm": 2.0,
                    "total_root_length_mm_clean": 10.0,
                    "measurement_tier": "individual",
                    "ownership_measurement_valid": True,
                },
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_conflict",
                    "Timestamp": "2026-07-02 08:00:00",
                    "FrameIndex": 1,
                    "plant_id": "plant_01",
                    "shoot_bbox_x": 42,
                    "shoot_bbox_w": 16,
                    "primary_root_length_mm_clean": 800.0,
                    "lateral_total_length_mm": 200.0,
                    "total_root_length_mm_clean": 1000.0,
                    "measurement_tier": "combined_overlap",
                    "ownership_measurement_valid": False,
                    "conflict_group_id": "combined::plant_01+plant_02",
                    "conflict_group_members": ["plant_01", "plant_02"],
                    "conflict_group_size": 2,
                    "combined_root_length_mm": 27.5,
                    "combined_root_length_px": 55.0,
                    "combined_area_px": 120,
                    "combined_area_mm2": 30.0,
                },
                {
                    "Series": "synthetic",
                    "PetriDish": "plate_conflict",
                    "Timestamp": "2026-07-03 08:00:00",
                    "FrameIndex": 2,
                    "plant_id": "plant_01",
                    "shoot_bbox_x": 42,
                    "shoot_bbox_w": 16,
                    "primary_root_length_mm_clean": 0.0,
                    "lateral_total_length_mm": 0.0,
                    "total_root_length_mm_clean": 0.0,
                    "measurement_tier": "combined_overlap",
                    "ownership_measurement_valid": False,
                    "conflict_group_id": "combined::plant_01+plant_02",
                    "conflict_group_members": ["plant_01", "plant_02"],
                    "conflict_group_size": 2,
                },
            ]
        )

        compartment = build_compartmentalized_occlusion_dataframe(
            whole,
            ownership_df=ownership,
            plant_count=1,
        ).reset_index(drop=True)

        self.assertEqual(compartment["occlusion_aware_total_root_length_mm"].tolist(), [10.0, 10.0, 10.0])
        self.assertEqual(float(compartment.loc[1, "visible_total_root_length_mm"]), 1000.0)
        self.assertEqual(float(compartment.loc[1, "combined_root_length_mm"]), 27.5)
        self.assertEqual(float(compartment.loc[1, "combined_root_length_px"]), 55.0)
        self.assertEqual(int(compartment.loc[1, "combined_area_px"]), 120)
        self.assertEqual(float(compartment.loc[2, "occlusion_inferred_total_root_length_mm"]), 10.0)
        self.assertEqual(compartment.loc[1, "occlusion_analysis_status"], "individual_state_frozen")
        self.assertTrue(bool(compartment.loc[1, "occlusion_individual_state_frozen"]))
        self.assertEqual(
            compartment.loc[1, "occlusion_carry_forward_status"],
            "frozen_invalid_ownership_measurement",
        )

    def test_writes_all_occlusion_assets_without_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "plate.png"
            self._write_image(image_path)
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[30:80, 48] = 1
            mask_path = tmp_dir / "plate_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)
            mask_df = build_mask_total_dataframe(
                [
                    {
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "output_mask": str(mask_path),
                        "meta": {
                            "series": "synthetic",
                            "petri": "plate_assets",
                            "timestamp": "2026-07-01 08:00:00",
                            "frame_index": 0,
                            "relative_folder": ".",
                        },
                        "details": {"pixel_size_mm": 1.0},
                    }
                ],
                class_ids=[1, 3],
                shoot_class_ids=[2],
                fallback_pixel_size_mm=1.0,
                min_component_area=1,
                shoot_rgb_rescue=True,
            )
            ownership_df = pd.DataFrame(
                [
                    {
                        "Series": "synthetic",
                        "PetriDish": "plate_assets",
                        "Timestamp": "2026-07-01 08:00:00",
                        "FrameIndex": 0,
                        "plant_id": "plant_01",
                        "pixel_size_mm": 1.0,
                        "ownership_primary_root_length_mm": 50.0,
                        "ownership_lateral_root_length_mm": 0.0,
                        "ownership_total_root_length_mm": 50.0,
                        "ownership_measurement_valid": False,
                        "combined_root_length_mm": 50.0,
                        "combined_root_length_px": 50.0,
                        "combined_area_px": 50,
                    }
                ]
            )
            result = write_occlusion_aware_outputs(
                mask_df,
                output_dir=tmp_dir / "out",
                ownership_df=ownership_df,
                plant_count=2,
                generate_videos=False,
            )

            self.assertTrue(result.whole_plate_detail_csv.exists())
            self.assertTrue(result.whole_plate_summary_csv.exists())
            self.assertTrue(result.compartment_detail_csv.exists())
            self.assertTrue(result.compartment_summary_csv.exists())
            self.assertTrue(result.workbook_path.exists())
            self.assertTrue(result.metadata_json.exists())
            metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
            carry_forward = metadata["ownership_carry_forward"]
            self.assertEqual(carry_forward["invalid_individual_rows_frozen"], 1)
            self.assertEqual(carry_forward["combined_conflict_measurement_rows_preserved"], 1)


if __name__ == "__main__":
    unittest.main()
