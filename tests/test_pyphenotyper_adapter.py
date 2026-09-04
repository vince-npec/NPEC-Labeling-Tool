from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from resources.pyphenotyper_adapter import (
    PyPhenotyperConfig,
    PyPhenotyperExpertConfig,
    PyPhenotyperVariantConfig,
    _artifact_sha256,
    _apply_lucifer_green_shoot_rescue,
    _is_rgb_like_image,
    _mask_qc_summary,
    _mixed_profile_enabled,
    _pipeline_tuning_overrides,
    _routing_enabled,
    _resolve_item_config,
    _returns_original_coords,
    _should_apply_lucifer_green_shoot_rescue,
    _shoot_priority_over_root,
    _should_apply_adapter_shoot_guard,
    build_lucifer_green_shoot_mask,
    detect_pyphenotyper_pixel_size_mm,
    masks_to_index_prediction,
    run_pyphenotyper_segmentation_pipeline,
)
from resources.models import DatasetImageItem


class _DummyFeaturesModule:
    RETURNS_ORIGINAL_COORDS = True


class PyPhenotyperAdapterTests(unittest.TestCase):
    def test_artifact_sha256_records_exact_model_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            model_path = Path(tmp_dir_name) / "model.h5"
            payload = b"npec-model-artifact"
            model_path.write_bytes(payload)
            self.assertEqual(
                _artifact_sha256(str(model_path)),
                hashlib.sha256(payload).hexdigest(),
            )

    def test_detect_pixel_size_from_json_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            pipeline_dir = Path(tmp_dir_name) / "pyphenotyper"
            features_dir = pipeline_dir / "features"
            features_dir.mkdir(parents=True)
            metadata_path = pipeline_dir / "pyphenotyper_scale.json"
            metadata_path.write_text(json.dumps({"pixel_size_mm": 0.04118616144975288}), encoding="utf-8")
            value, source = detect_pyphenotyper_pixel_size_mm(pipeline_dir)
            self.assertAlmostEqual(value or 0.0, 0.04118616144975288, places=10)
            self.assertEqual(Path(source or "").resolve(), metadata_path.resolve())

    def test_detect_pixel_size_from_legacy_roots_segmentation_constants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            pipeline_dir = Path(tmp_dir_name) / "pyphenotyper"
            features_dir = pipeline_dir / "features"
            features_dir.mkdir(parents=True)
            legacy_path = features_dir / "roots_segmentation.py"
            legacy_path.write_text(
                "plate_size_mm = 111.88\nplate_size_pixels = 4200\n",
                encoding="utf-8",
            )
            value, source = detect_pyphenotyper_pixel_size_mm(pipeline_dir)
            self.assertAlmostEqual(value or 0.0, 111.88 / 4200.0, places=10)
            self.assertEqual(Path(source or "").resolve(), legacy_path.resolve())

    def test_rgb_like_detection_rejects_grayscale_rgb_triplets(self) -> None:
        rgb_gray = np.repeat(np.arange(16, dtype=np.uint8).reshape(4, 4, 1), 3, axis=2)
        self.assertFalse(_is_rgb_like_image(rgb_gray))

    def test_rgb_like_detection_rejects_small_jpeg_chroma_noise(self) -> None:
        base = np.full((32, 32, 3), 140, dtype=np.int16)
        base[::2, ::2, 0] += 1
        base[1::2, 1::2, 2] -= 1
        self.assertFalse(_is_rgb_like_image(np.clip(base, 0, 255).astype(np.uint8)))

    def test_rgb_like_detection_accepts_real_color_image(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[..., 1] = 40
        self.assertTrue(_is_rgb_like_image(image))

    def test_dataset_pipeline_locks_routed_expert_per_series_folder(self) -> None:
        expert = PyPhenotyperExpertConfig(
            key="plate_bw_hades",
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
        )
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            expert_variants=(expert,),
            enable_bbox_tracking=False,
        )
        items = [
            DatasetImageItem("f0", "f0.png", Path("/tmp/plate_a/f0.png"), np.zeros((8, 8, 3), dtype=np.uint8)),
            DatasetImageItem("f1", "f1.png", Path("/tmp/plate_a/f1.png"), np.zeros((8, 8, 3), dtype=np.uint8)),
        ]
        prediction = np.zeros((8, 8), dtype=np.uint8)
        first_details = {"routing_expert_key": "plate_bw_hades", "routing_family": "arabidopsis_plate_bw"}
        locked_details = {"routing_expert_key": "plate_bw_hades", "routing_family": "arabidopsis_plate_bw"}

        with patch(
            "resources.pyphenotyper_adapter.run_pyphenotyper_item",
            return_value=(prediction, first_details),
        ) as initial_route, patch(
            "resources.pyphenotyper_adapter._run_pyphenotyper_item_with_route_lock",
            return_value=(prediction, locked_details),
        ) as locked_route:
            _predictions, metadata = run_pyphenotyper_segmentation_pipeline(items, config)

        self.assertEqual(initial_route.call_count, 1)
        self.assertEqual(locked_route.call_count, 1)
        self.assertEqual(metadata["f0"]["routing_selected_by"], "series_first_frame_lock")
        self.assertEqual(metadata["f1"]["routing_selected_by"], "series_route_lock")

    def test_profile_payload_overrides_returns_original_coords(self) -> None:
        module = _DummyFeaturesModule()
        self.assertFalse(_returns_original_coords(module, {"return_original_coords": False}))
        self.assertTrue(_returns_original_coords(module, {"return_original_coords": True}))

    def test_legacy_shoot_mode_disables_adapter_guard_by_default(self) -> None:
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            pipeline_overrides={"shoot_postprocess_mode": "legacy"},
        )
        self.assertFalse(_should_apply_adapter_shoot_guard(config))

    def test_adapter_guard_can_be_forced_back_on(self) -> None:
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            pipeline_overrides={
                "shoot_postprocess_mode": "legacy",
                "adapter_shoot_guard_mode": "default",
            },
        )
        self.assertTrue(_should_apply_adapter_shoot_guard(config))

    def test_pipeline_tuning_overrides_strip_adapter_only_keys(self) -> None:
        cleaned = _pipeline_tuning_overrides(
            {
                "shoot_postprocess_mode": "bw_arabidopsis",
                "adapter_shoot_guard_mode": "default",
                "ownership_backend": "bw_arabidopsis_seed_centers",
                "expected_plant_count": 5,
                "shoot_color_rescue_eligibility": "model_locked",
                "shoot_color_rescue_eligible": False,
                "shoot_color_rescue_failure_policy": "retain_model",
                "shoot_color_rescue_model_locked": True,
                "shoot_color_rescue_mode": "lucifer_green",
                "shoot_priority_over_root": False,
            }
        )

        self.assertEqual(cleaned, {"shoot_postprocess_mode": "bw_arabidopsis"})

    def test_lucifer_green_shoot_rescue_recovers_olive_rgb_leaves(self) -> None:
        image = np.full((600, 900, 3), 214, dtype=np.uint8)
        root_mask = np.zeros((600, 900), dtype=np.uint8)
        shoot_mask = np.zeros_like(root_mask)
        for center_x in (160, 320, 500, 680):
            cv2.line(root_mask, (center_x, 190), (center_x + 25, 540), 1, 3)
            cv2.ellipse(image, (center_x - 24, 150), (38, 18), -15, 0, 360, (75, 93, 34), -1)
            cv2.ellipse(image, (center_x + 28, 145), (34, 17), 18, 0, 360, (78, 98, 38), -1)
            cv2.line(image, (center_x, 180), (center_x + 2, 120), (80, 96, 40), 6)
        image[470:560, 50:180] = 35
        shoot_mask[470:560, 50:180] = 1

        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            pipeline_overrides={"shoot_color_rescue_mode": "lucifer_green"},
        )
        rescued, meta = _apply_lucifer_green_shoot_rescue(image, root_mask, shoot_mask, config=config)

        self.assertTrue(_should_apply_lucifer_green_shoot_rescue(config))
        self.assertTrue(meta["applied"])
        self.assertEqual(meta["mode"], "green_only")
        self.assertGreater(int(meta["removed_model_pixels"]), 0)
        self.assertGreater(int(rescued.sum()), 1500)
        self.assertEqual(int(rescued[520, 100]), 0)
        self.assertEqual(int(rescued[450, 500]), 0)

    def test_lucifer_green_shoot_rescue_rejects_gray_colony_texture(self) -> None:
        image = np.full((600, 900, 3), 214, dtype=np.uint8)
        root_mask = np.zeros((600, 900), dtype=np.uint8)
        shoot_mask = np.zeros_like(root_mask)
        cv2.ellipse(image, (450, 245), (95, 48), 0, 0, 360, (138, 140, 111), -1)
        cv2.ellipse(shoot_mask, (450, 245), (95, 48), 0, 0, 360, 1, -1)
        cv2.ellipse(image, (250, 150), (42, 20), -15, 0, 360, (75, 96, 35), -1)
        cv2.ellipse(image, (306, 145), (36, 18), 18, 0, 360, (78, 99, 38), -1)
        cv2.line(image, (280, 182), (282, 122), (80, 97, 40), 6)

        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            pipeline_overrides={"shoot_color_rescue_mode": "lucifer_green"},
        )
        rescued, meta = _apply_lucifer_green_shoot_rescue(image, root_mask, shoot_mask, config=config)

        self.assertTrue(meta["applied"])
        self.assertEqual(meta["mode"], "green_only")
        self.assertGreater(int(meta["removed_model_pixels"]), 0)
        self.assertGreater(int(rescued.sum()), 1200)
        self.assertEqual(int(rescued[245, 450]), 0)

    def test_lucifer_green_shoot_rescue_rejects_muted_green_gray_colony_texture(self) -> None:
        image = np.full((600, 900, 3), 214, dtype=np.uint8)
        root_mask = np.zeros((600, 900), dtype=np.uint8)
        shoot_mask = np.zeros_like(root_mask)
        cv2.ellipse(image, (450, 245), (95, 48), 0, 0, 360, (118, 126, 96), -1)
        cv2.ellipse(shoot_mask, (450, 245), (95, 48), 0, 0, 360, 1, -1)
        cv2.ellipse(image, (250, 150), (42, 20), -15, 0, 360, (75, 96, 35), -1)
        cv2.ellipse(image, (306, 145), (36, 18), 18, 0, 360, (78, 99, 38), -1)
        cv2.line(image, (280, 182), (282, 122), (80, 97, 40), 6)

        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            pipeline_overrides={"shoot_color_rescue_mode": "lucifer_green"},
        )
        rescued, meta = _apply_lucifer_green_shoot_rescue(image, root_mask, shoot_mask, config=config)

        self.assertTrue(meta["applied"])
        self.assertEqual(meta["mode"], "green_only")
        self.assertGreater(int(meta["removed_model_pixels"]), 0)
        self.assertGreater(int(rescued.sum()), 1200)
        self.assertEqual(int(rescued[245, 450]), 0)

    def test_lucifer_green_shoot_rescue_preserves_model_mask_for_ineligible_route(self) -> None:
        image = np.full((300, 400, 3), 214, dtype=np.uint8)
        cv2.ellipse(image, (150, 90), (34, 18), 0, 0, 360, (50, 130, 20), -1)
        root_mask = np.zeros((300, 400), dtype=np.uint8)
        shoot_mask = np.zeros_like(root_mask)
        shoot_mask[190:240, 280:340] = 1
        color_mask, _color_meta = build_lucifer_green_shoot_mask(image, image.shape[:2])
        self.assertGreater(int(color_mask.sum()), 20)

        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            pipeline_overrides={
                "shoot_color_rescue_mode": "lucifer_green",
                "shoot_color_rescue_eligible": False,
                "shoot_color_rescue_failure_policy": "retain_model",
            },
        )
        rescued, meta = _apply_lucifer_green_shoot_rescue(image, root_mask, shoot_mask, config=config)

        np.testing.assert_array_equal(rescued, shoot_mask)
        self.assertFalse(bool(meta["eligible"]))
        self.assertEqual(meta["status"], "ineligible_route")
        self.assertEqual(meta["detector_status"], "not_run_ineligible")
        self.assertEqual(meta["fallback_status"], "retained_model_mask")
        self.assertEqual(int(meta["retained_model_pixels"]), int(shoot_mask.sum()))

    def test_lucifer_green_shoot_rescue_falls_back_to_model_mask_when_detector_abstains(self) -> None:
        image = np.empty((240, 320, 3), dtype=np.uint8)
        image[...] = np.array([180, 120, 80], dtype=np.uint8)
        root_mask = np.zeros((240, 320), dtype=np.uint8)
        shoot_mask = np.zeros_like(root_mask)
        shoot_mask[120:180, 130:190] = 1
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            pipeline_overrides={
                "shoot_color_rescue_mode": "lucifer_green",
                "shoot_color_rescue_eligible": True,
                "shoot_color_rescue_failure_policy": "retain_model",
            },
        )

        rescued, meta = _apply_lucifer_green_shoot_rescue(image, root_mask, shoot_mask, config=config)

        np.testing.assert_array_equal(rescued, shoot_mask)
        self.assertTrue(bool(meta["eligible"]))
        self.assertTrue(bool(meta["rgb_like"]))
        self.assertTrue(bool(meta["abstained"]))
        self.assertFalse(bool(meta["applied"]))
        self.assertEqual(int(meta["candidate_pixels"]), 0)
        self.assertEqual(meta["status"], "abstained_no_candidates")
        self.assertEqual(meta["fallback_status"], "retained_model_mask")
        self.assertEqual(int(meta["retained_model_pixels"]), int(shoot_mask.sum()))
        self.assertEqual(int(meta["output_pixels"]), int(shoot_mask.sum()))

    def test_lucifer_green_shoot_mask_does_not_fill_gray_colony_interior(self) -> None:
        image = np.full((300, 300, 3), 214, dtype=np.uint8)
        cv2.ellipse(image, (150, 90), (28, 16), 0, 0, 360, (126, 128, 116), -1)
        cv2.ellipse(image, (150, 90), (28, 16), 0, 0, 360, (70, 125, 32), 4)

        mask, meta = build_lucifer_green_shoot_mask(image, image.shape[:2])

        self.assertGreater(int(mask.sum()), 100)
        self.assertEqual(int(mask[90, 150]), 0)
        self.assertGreater(int(mask[90, 122]), 0)
        self.assertEqual(meta["filter"], "arabidopsis_rosette_rgb_chroma_root_context")

    def test_lucifer_green_shoot_mask_rejects_lower_gray_green_colony_band(self) -> None:
        image = np.full((600, 900, 3), 214, dtype=np.uint8)
        cv2.ellipse(image, (250, 150), (42, 20), -15, 0, 360, (75, 96, 35), -1)
        cv2.ellipse(image, (306, 145), (36, 18), 18, 0, 360, (78, 99, 38), -1)
        cv2.line(image, (280, 182), (282, 122), (80, 97, 40), 6)
        cv2.ellipse(image, (450, 315), (118, 36), 0, 0, 360, (95, 104, 48), -1)

        mask, meta = build_lucifer_green_shoot_mask(image, image.shape[:2])

        self.assertTrue(bool(meta["rgb_like"]))
        self.assertGreater(int(mask[150, 250]), 0)
        self.assertGreater(int(mask.sum()), 400)
        self.assertEqual(int(mask[315, 450]), 0)

    def test_lucifer_green_shoot_mask_rejects_low_chroma_gray_green_colony(self) -> None:
        image = np.full((600, 900, 3), 214, dtype=np.uint8)
        cv2.ellipse(image, (250, 150), (42, 20), -15, 0, 360, (75, 96, 35), -1)
        cv2.ellipse(image, (450, 172), (88, 30), 0, 0, 360, (122, 128, 92), -1)

        mask, _meta = build_lucifer_green_shoot_mask(image, image.shape[:2])

        self.assertGreater(int(mask[150, 250]), 0)
        self.assertEqual(int(mask[172, 450]), 0)

    def test_lucifer_green_shoot_mask_rejects_top_greenish_gray_colony(self) -> None:
        image = np.full((600, 900, 3), 214, dtype=np.uint8)
        cv2.ellipse(image, (250, 150), (42, 20), -15, 0, 360, (75, 96, 35), -1)
        cv2.ellipse(image, (450, 172), (88, 30), 0, 0, 360, (95, 104, 48), -1)

        mask, _meta = build_lucifer_green_shoot_mask(image, image.shape[:2])

        self.assertGreater(int(mask[150, 250]), 0)
        self.assertEqual(int(mask[172, 450]), 0)

    def test_lucifer_green_shoot_mask_rejects_greenish_colony_below_root_crown(self) -> None:
        image = np.full((600, 900, 3), 214, dtype=np.uint8)
        root_mask = np.zeros((600, 900), dtype=np.uint8)
        cv2.line(root_mask, (450, 190), (450, 540), 1, 5)
        cv2.ellipse(image, (450, 245), (88, 30), 0, 0, 360, (95, 104, 48), -1)

        mask, meta = build_lucifer_green_shoot_mask(image, image.shape[:2], root_mask=root_mask)

        self.assertEqual(int(mask[245, 450]), 0)
        self.assertEqual(int(mask.sum()), 0)
        self.assertGreater(int(meta["root_crown_gate_rejections"]), 0)

    def test_lucifer_green_shoot_mask_uses_root_context_for_muted_rosettes(self) -> None:
        image = np.full((420, 620, 3), 214, dtype=np.uint8)
        cv2.ellipse(image, (260, 130), (44, 20), -12, 0, 360, (92, 101, 50), -1)
        cv2.ellipse(image, (320, 132), (36, 18), 16, 0, 360, (96, 104, 56), -1)
        root_mask = np.zeros((420, 620), dtype=np.uint8)
        cv2.line(root_mask, (292, 150), (296, 340), 1, 5)

        strict_mask, strict_meta = build_lucifer_green_shoot_mask(image, image.shape[:2])
        contextual_mask, contextual_meta = build_lucifer_green_shoot_mask(image, image.shape[:2], root_mask=root_mask)

        self.assertEqual(int(strict_mask.sum()), 0)
        self.assertGreater(int(contextual_mask.sum()), 1200)
        self.assertTrue(bool(contextual_meta["root_context_enabled"]))
        self.assertGreater(int(contextual_meta["root_context_supported_components"]), 0)
        self.assertFalse(bool(strict_meta["root_context_enabled"]))

    def test_lucifer_green_shoot_mask_uses_loose_root_context_for_olive_leaf(self) -> None:
        image = np.full((600, 900, 3), 214, dtype=np.uint8)
        root_mask = np.zeros((600, 900), dtype=np.uint8)
        cv2.line(root_mask, (450, 220), (455, 560), 1, 5)
        cv2.ellipse(image, (450, 145), (54, 28), 0, 0, 360, (98, 107, 48), -1)

        unanchored_mask, _unanchored_meta = build_lucifer_green_shoot_mask(image, image.shape[:2])
        anchored_mask, anchored_meta = build_lucifer_green_shoot_mask(
            image,
            image.shape[:2],
            root_mask=root_mask,
        )

        self.assertEqual(int(unanchored_mask.sum()), 0)
        self.assertGreater(int(anchored_mask.sum()), 3500)
        self.assertGreater(int(anchored_meta["olive_root_supported_components"]), 0)
        self.assertGreater(
            int(anchored_meta["olive_root_support_radius_px"]),
            int(anchored_meta["root_support_radius_px"]),
        )

    def test_lucifer_green_shoot_mask_keeps_large_root_anchored_rosette(self) -> None:
        image = np.full((900, 1200, 3), 214, dtype=np.uint8)
        root_mask = np.zeros((900, 1200), dtype=np.uint8)
        cv2.line(root_mask, (600, 280), (605, 820), 1, 5)
        cv2.ellipse(image, (535, 215), (80, 40), -10, 0, 360, (72, 104, 34), -1)
        cv2.ellipse(image, (665, 215), (80, 40), 10, 0, 360, (76, 108, 36), -1)
        cv2.line(image, (600, 290), (600, 165), (74, 105, 35), 18)

        unanchored_mask, unanchored_meta = build_lucifer_green_shoot_mask(image, image.shape[:2])
        anchored_mask, anchored_meta = build_lucifer_green_shoot_mask(
            image,
            image.shape[:2],
            root_mask=root_mask,
        )

        self.assertEqual(int(unanchored_mask.sum()), 0)
        self.assertGreater(int(anchored_mask.sum()), 15000)
        self.assertGreater(int(anchored_meta["large_root_supported_components"]), 0)
        self.assertGreater(
            int(anchored_meta["max_root_supported_component_area_px"]),
            int(anchored_meta["max_unanchored_component_area_px"]),
        )
        self.assertFalse(bool(unanchored_meta["root_context_enabled"]))

    def test_shoot_color_rescue_off_overrides_green_flag(self) -> None:
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            pipeline_overrides={
                "shoot_color_rescue_mode": "off",
                "lucifer_green_shoot_rescue": True,
            },
        )

        self.assertFalse(_should_apply_lucifer_green_shoot_rescue(config))

    def test_shoot_priority_over_root_defaults_true(self) -> None:
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
        )
        self.assertTrue(_shoot_priority_over_root(config))

    def test_shoot_priority_over_root_can_be_disabled(self) -> None:
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/pipeline"),
            root_model_path=Path("/tmp/root.h5"),
            shoot_model_path=Path("/tmp/shoot.h5"),
            pipeline_overrides={"shoot_priority_over_root": False},
        )
        self.assertFalse(_shoot_priority_over_root(config))

    def test_masks_to_index_prediction_can_preserve_root_priority(self) -> None:
        root_mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        shoot_mask = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        occlusion_mask = np.zeros((2, 2), dtype=np.uint8)
        pred = masks_to_index_prediction(
            root_mask=root_mask,
            shoot_mask=shoot_mask,
            seed_mask=np.zeros((2, 2), dtype=np.uint8),
            occlusion_mask=occlusion_mask,
            shape_hw=(2, 2),
            root_class_id=3,
            shoot_class_id=2,
            include_occlusion=True,
            shoot_priority_over_root=False,
        )
        expected = np.array([[3, 3], [2, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(pred, expected)

    def test_seed_class_can_override_root_and_shoot_pixels(self) -> None:
        root_mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        shoot_mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        seed_mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        occlusion_mask = np.zeros((2, 2), dtype=np.uint8)
        pred = masks_to_index_prediction(
            root_mask=root_mask,
            shoot_mask=shoot_mask,
            seed_mask=seed_mask,
            occlusion_mask=occlusion_mask,
            shape_hw=(2, 2),
            root_class_id=3,
            shoot_class_id=2,
            seed_class_id=1,
            include_occlusion=True,
            shoot_priority_over_root=True,
        )
        expected = np.array([[1, 2], [0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(pred, expected)

    def test_native_lateral_class_is_preserved_without_inflating_occlusion(self) -> None:
        root_mask = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.uint8)
        lateral_mask = np.array([[0, 1, 0], [0, 0, 0]], dtype=np.uint8)
        occlusion_mask = np.array([[0, 0, 1], [0, 0, 0]], dtype=np.uint8)
        pred = masks_to_index_prediction(
            root_mask=root_mask,
            lateral_root_mask=lateral_mask,
            shoot_mask=np.zeros((2, 3), dtype=np.uint8),
            seed_mask=np.zeros((2, 3), dtype=np.uint8),
            occlusion_mask=occlusion_mask,
            shape_hw=(2, 3),
            root_class_id=1,
            lateral_class_id=3,
            shoot_class_id=2,
            include_occlusion=True,
        )
        expected = np.array([[1, 3, 1], [0, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(pred, expected)

    def test_mask_qc_flags_high_root_shoot_overlap(self) -> None:
        root_mask = np.zeros((10, 10), dtype=np.uint8)
        shoot_mask = np.zeros((10, 10), dtype=np.uint8)
        root_mask[2:8, 2:8] = 1
        shoot_mask[2:8, 2:8] = 1

        qc = _mask_qc_summary(root_mask, shoot_mask)

        self.assertEqual(int(qc["root_shoot_overlap_pixels"]), 36)
        self.assertIn("root_shoot_overlap_high", qc["mask_qc_flags"])

    def test_item_config_routes_to_rgb_variant(self) -> None:
        item = DatasetImageItem(
            uid="rgb",
            name="rgb.png",
            path=Path("/tmp/rgb.png"),
            image=np.dstack(
                (
                    np.zeros((4, 4), dtype=np.uint8),
                    np.full((4, 4), 20, dtype=np.uint8),
                    np.zeros((4, 4), dtype=np.uint8),
                )
            ),
        )
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/default_pipeline"),
            root_model_path=Path("/tmp/default_root.h5"),
            shoot_model_path=Path("/tmp/default_shoot.h5"),
            grayscale_variant=PyPhenotyperVariantConfig(
                pipeline_dir=Path("/tmp/gray_pipeline"),
                root_model_path=Path("/tmp/gray_root.h5"),
                shoot_model_path=Path("/tmp/gray_shoot.h5"),
                label="grayscale_hades",
            ),
            rgb_variant=PyPhenotyperVariantConfig(
                pipeline_dir=Path("/tmp/rgb_pipeline"),
                root_model_path=Path("/tmp/rgb_root.h5"),
                shoot_model_path=Path("/tmp/rgb_shoot.h5"),
                label="lucifer_rgb",
            ),
        )
        effective, variant_label, routing_family = _resolve_item_config(item, config)
        self.assertEqual(variant_label, "lucifer_rgb")
        self.assertEqual(routing_family, "rgb")
        self.assertEqual(effective.pipeline_dir, Path("/tmp/rgb_pipeline"))

    def test_item_config_routes_to_grayscale_variant(self) -> None:
        item = DatasetImageItem(
            uid="gray",
            name="gray.png",
            path=Path("/tmp/gray.png"),
            image=np.zeros((4, 4), dtype=np.uint8),
        )
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/default_pipeline"),
            root_model_path=Path("/tmp/default_root.h5"),
            shoot_model_path=Path("/tmp/default_shoot.h5"),
            grayscale_variant=PyPhenotyperVariantConfig(
                pipeline_dir=Path("/tmp/gray_pipeline"),
                root_model_path=Path("/tmp/gray_root.h5"),
                shoot_model_path=Path("/tmp/gray_shoot.h5"),
                label="grayscale_hades",
            ),
            rgb_variant=PyPhenotyperVariantConfig(
                pipeline_dir=Path("/tmp/rgb_pipeline"),
                root_model_path=Path("/tmp/rgb_root.h5"),
                shoot_model_path=Path("/tmp/rgb_shoot.h5"),
                label="lucifer_rgb",
            ),
        )
        effective, variant_label, routing_family = _resolve_item_config(item, config)
        self.assertEqual(variant_label, "grayscale_hades")
        self.assertEqual(routing_family, "grayscale")
        self.assertEqual(effective.pipeline_dir, Path("/tmp/gray_pipeline"))

    def test_expert_variants_enable_router(self) -> None:
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/default_pipeline"),
            root_model_path=Path("/tmp/default_root.h5"),
            shoot_model_path=Path("/tmp/default_shoot.h5"),
            expert_variants=(
                PyPhenotyperExpertConfig(
                    key="plate_bw_hades",
                    pipeline_dir=Path("/tmp/gray_pipeline"),
                    root_model_path=Path("/tmp/gray_root.h5"),
                    shoot_model_path=Path("/tmp/gray_shoot.h5"),
                ),
            ),
        )
        self.assertTrue(_routing_enabled(config))

    def test_mixed_profile_enabled_when_root_or_shoot_overrides_are_set(self) -> None:
        config = PyPhenotyperConfig(
            pipeline_dir=Path("/tmp/default_pipeline"),
            root_model_path=Path("/tmp/default_root.h5"),
            shoot_model_path=Path("/tmp/default_shoot.h5"),
            root_profile_overrides={"mode": "binary_pair", "input_mode": "rgb"},
            shoot_profile_overrides={"mode": "multiclass", "input_mode": "rgb"},
        )
        self.assertTrue(_mixed_profile_enabled(config))


if __name__ == "__main__":
    unittest.main()
