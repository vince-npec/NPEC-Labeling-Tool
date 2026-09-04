from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from resources.scripts import analyze_dario_potato_dataset as endpoint
from resources import potato_endpoint_pipeline as package


class PotatoEndpointPipelineTests(unittest.TestCase):
    def test_dataset_service_pins_models_calibration_and_disables_one_off_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "endpoint.JPG"
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
            output_dir = tmp_dir / "output"
            model_path = tmp_dir / "best_potato_root_june_10_model_patch_256_max_f10.815_max_IoU0.915.h5"
            weights_path = tmp_dir / "dario_potato_root_final_all16.weights.h5"
            validation_path = tmp_dir / "npec_potato_reference_validation.json"
            model_path.write_bytes(b"test-model")
            weights_path.write_bytes(b"test-weights")
            validation_path.write_text("{}", encoding="utf-8")
            captured: list[Namespace] = []

            def fake_run(args: Namespace) -> int:
                captured.append(args)
                output_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([{"sample_stem": "101_RP001_Control"}]).to_csv(
                    output_dir / "npec_potato_endpoint_phenotypes.csv",
                    index=False,
                )
                pd.DataFrame([{"branch_class": "lateral_root"}]).to_csv(
                    output_dir / "npec_potato_branch_traits.csv",
                    index=False,
                )
                pd.DataFrame([{"review_required": True}]).to_csv(
                    output_dir / "npec_potato_qc.csv",
                    index=False,
                )
                (output_dir / "npec_potato_finalization_summary.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                return 0

            config = package.PotatoEndpointPackageConfig(
                input_dir=tmp_dir,
                output_dir=output_dir,
                image_paths=(image_path,),
            )
            with (
                patch.object(
                    package,
                    "validate_bundled_models",
                    return_value=(model_path, weights_path, validation_path),
                ),
                patch.object(package.analyzer, "run", side_effect=fake_run),
            ):
                result = package.run_potato_endpoint_package(config)

        self.assertEqual(result.analyzed_images, 1)
        self.assertEqual(result.branch_rows, 1)
        self.assertEqual(result.review_rows, 1)
        self.assertEqual(len(captured), 1)
        args = captured[0]
        self.assertFalse(bool(args.enable_forced_rescues))
        self.assertTrue(bool(args.generic_dataset))
        self.assertAlmostEqual(float(args.pixel_size_mm), 111.88 / 4200.0, places=12)
        self.assertEqual(Path(args.root_weights).name, "dario_potato_root_final_all16.weights.h5")

    def test_generic_package_emits_reference_artifact_contract_without_forced_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            input_dir = tmp_dir / "raw"
            output_dir = tmp_dir / "package"
            input_dir.mkdir()
            image_path = input_dir / "2024_07_15_09_34_10_0007.JPG"
            Image.fromarray(np.full((32, 32, 3), 230, dtype=np.uint8)).save(image_path)

            image_rgb = np.full((900, 1200, 3), 235, dtype=np.uint8)
            image_rgb[250:415, 540:665] = (55, 145, 45)
            probability = np.zeros((900, 1200), dtype=np.float32)
            cv2.line(probability, (600, 395), (600, 645), 0.95, thickness=7)
            cv2.line(probability, (600, 500), (530, 545), 0.92, thickness=5)
            cv2.line(probability, (600, 555), (675, 600), 0.91, thickness=5)

            model_path = tmp_dir / endpoint.DEFAULT_ROOT_MODEL.name
            weights_path = tmp_dir / endpoint.DEFAULT_ROOT_WEIGHTS.name
            model_path.write_bytes(b"test-model")
            weights_path.write_bytes(b"test-weights")
            validation_path = endpoint.DEFAULT_VALIDATION
            args = Namespace(
                dataset=tmp_dir,
                input_dir=input_dir,
                output=output_dir,
                image_paths=[image_path],
                labels_dir=tmp_dir / "labeled",
                identity_records=[
                    {
                        "source_name": image_path.name,
                        "source_path": str(image_path),
                        "source_date": "2026-01-02T03:04:05",
                        "plate_id": "101_RP001_Control",
                        "plate_frame_index": 1,
                        "confidence": 0.99,
                        "status": "accepted",
                        "evidence": [{"kind": "vision_text", "text": "101 RP001 Control"}],
                    }
                ],
                identity_manifest=None,
                root_model=model_path,
                root_weights=weights_path,
                validation_path=validation_path,
                pixel_size_mm=endpoint.LUCIFER_PIXEL_SIZE_MM,
                limit=0,
                stems="",
                include_calibration=False,
                no_manual_labels=True,
                skip_videos=False,
                video_fps=2.0,
                overwrite=True,
                generic_dataset=False,
                enable_forced_rescues=False,
                progress_callback=None,
                cancel_callback=None,
            )

            def fake_video(path, *_args, **_kwargs):
                Path(path).write_bytes(b"test-video")

            with (
                patch.object(endpoint, "_load_isolated_keras_model", return_value=Mock()),
                patch.object(
                    endpoint,
                    "_load_normalized_crop",
                    return_value=(
                        image_rgb,
                        {
                            "source_exif_orientation": 1,
                            "raw_crop_box": list(endpoint.RAW_CROP_BOX),
                            "normalized_rotation_deg": 180,
                            "normalized_size": [1200, 900],
                        },
                    ),
                ),
                patch.object(endpoint, "_predict_probability", return_value=probability),
                patch.object(endpoint, "_write_video", side_effect=fake_video),
                patch.object(
                    endpoint,
                    "_forced_crown_anchored_rescue",
                    side_effect=AssertionError("the plate-163 one-off rescue must stay disabled"),
                ),
            ):
                self.assertEqual(endpoint.run(args), 0)

            stem = "101_RP001_Control"
            masks = sorted((output_dir / "masks").glob(f"{stem}__*.png"))
            self.assertEqual(len(masks), 6)
            for path in (
                output_dir / "crops" / f"{stem}.jpg",
                output_dir / "label_crops" / f"{stem}__label.jpg",
                output_dir / "overlays" / f"{stem}__overlay.jpg",
                output_dir / "npec_potato_endpoint_all_metrics.xlsx",
                output_dir / "npec_potato_endpoint_overlay_reel.mp4",
                output_dir / "npec_potato_endpoint_mask_reel.mp4",
                output_dir / "npec_potato_endpoint_video_sample_frame.png",
                output_dir / "npec_potato_analysis_provenance.json",
                output_dir / "npec_potato_finalization_summary.json",
                output_dir / "npec_potato_plate_label_map.csv",
                output_dir / "npec_dario_plate_label_map.csv",
                output_dir / "README.md",
            ):
                self.assertTrue(path.exists(), path)

            phenotypes = pd.read_csv(output_dir / "npec_potato_endpoint_phenotypes.csv")
            self.assertEqual(len(phenotypes), 1)
            self.assertEqual(len(phenotypes.columns), 148)
            self.assertEqual(
                list(phenotypes.columns[:10]),
                [
                    "sample_index",
                    "sample_stem",
                    "source_image_stem",
                    "plate_name",
                    "plate_number",
                    "genotype",
                    "treatment",
                    "plate_capture_index",
                    "plate_capture_count",
                    "is_duplicate_plate_capture",
                ],
            )
            self.assertEqual(
                list(phenotypes.columns[-12:]),
                [
                    "root_to_shoot_area_ratio",
                    "rescue_applied",
                    "rescue_method",
                    "rescue_spec_sha256",
                    "rescue_normalized_size",
                    "rescue_source_jpeg_sha256",
                    "export_stem",
                    "plate_number_parse_method",
                    "plate_number_resolution_score",
                    "ocr_plate_text",
                    "ocr_plate_confidence",
                    "ocr_text",
                ],
            )
            self.assertEqual(str(phenotypes.loc[0, "sample_stem"]), stem)
            self.assertEqual(str(phenotypes.loc[0, "source_image_stem"]), image_path.stem)
            self.assertEqual(int(phenotypes.loc[0, "expected_plants"]), 1)
            self.assertIn("adventitious_root_length_mm", phenotypes.columns)
            self.assertIn("shoot_excess_green_mean", phenotypes.columns)

            branches = pd.read_csv(output_dir / "npec_potato_branch_traits.csv")
            qc = pd.read_csv(output_dir / "npec_potato_qc.csv")
            plate_map = pd.read_csv(output_dir / "npec_potato_plate_label_map.csv")
            self.assertEqual(len(branches.columns), 29)
            self.assertEqual(len(qc.columns), 28)
            self.assertEqual(len(plate_map.columns), 25)
            with pd.ExcelFile(output_dir / "npec_potato_endpoint_all_metrics.xlsx") as workbook:
                self.assertEqual(
                    workbook.sheet_names,
                    [
                        "Endpoint phenotypes",
                        "Branch traits",
                        "QC",
                        "Excluded captures",
                        "Dataset summary",
                        "Trait dictionary",
                        "Validation aggregate",
                        "Validation plates",
                        "Provenance",
                        "Plate label map",
                    ],
                )

            provenance = json.loads(
                (output_dir / "npec_potato_analysis_provenance.json").read_text(encoding="utf-8")
            )
            self.assertFalse(bool(provenance["one_off_forced_rescues_enabled"]))
            self.assertEqual(int(provenance["forced_rescue_count"]), 0)
            self.assertTrue(bool(provenance["export_assets_use_physical_plate_names"]))


if __name__ == "__main__":
    unittest.main()
