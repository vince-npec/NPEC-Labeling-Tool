from __future__ import annotations

import unittest

import numpy as np

from resources.general_root_starter import (
    build_calibration_profile,
    describe_image_features,
    expert_prior_score,
    load_normalized_label_schema,
    load_public_dataset_registry,
    normalize_annotation_targets,
)
from resources.models import LabelClass


class GeneralRootStarterTests(unittest.TestCase):
    def test_public_registry_contains_requested_sources(self) -> None:
        rows = load_public_dataset_registry()
        keys = {str(row.get("key")) for row in rows}
        self.assertIn("rootnav2", keys)
        self.assertIn("rootpainter_minirhizotron_demo", keys)
        self.assertIn("plantcv_dataset_catalog", keys)
        rootnav = next(row for row in rows if str(row.get("key")) == "rootnav2")
        rootpainter = next(row for row in rows if str(row.get("key")) == "rootpainter_minirhizotron_demo")
        self.assertEqual(str(rootnav.get("license")), "BSD-3-Clause")
        self.assertEqual(str(rootpainter.get("status")), "ingested")

    def test_label_schema_contains_core_keys(self) -> None:
        keys = {str(row.get("key")) for row in load_normalized_label_schema()}
        self.assertEqual(keys, {"shoot", "primary_root", "lateral_root", "root_binary", "seed_crown"})

    def test_normalize_targets_builds_root_binary_from_primary_and_lateral(self) -> None:
        classes = [
            LabelClass(1, "Shoot", "#56f39a"),
            LabelClass(2, "Primary Root", "#ffd166"),
            LabelClass(3, "Lateral Root", "#3fc1ff"),
        ]
        layers = {
            1: np.array(
                [
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
            2: np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
            3: np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
        }
        normalized = normalize_annotation_targets(layers, classes, (5, 5))
        self.assertEqual(int(np.count_nonzero(normalized["shoot"])), 2)
        self.assertEqual(int(np.count_nonzero(normalized["primary_root"])), 3)
        self.assertEqual(int(np.count_nonzero(normalized["lateral_root"])), 3)
        self.assertEqual(int(np.count_nonzero(normalized["root_binary"])), 5)

    def test_image_features_detect_rgb_and_gray(self) -> None:
        gray = np.zeros((32, 32), dtype=np.uint8)
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[..., 1] = 80
        gray_features = describe_image_features(gray)
        rgb_features = describe_image_features(rgb)
        self.assertFalse(gray_features.is_rgb_like)
        self.assertTrue(rgb_features.is_rgb_like)
        self.assertGreater(rgb_features.green_dominance, gray_features.green_dominance)

    def test_expert_prior_favors_matching_family(self) -> None:
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[..., 1] = 100
        features = describe_image_features(rgb)
        lucifer = expert_prior_score(
            features,
            expert_key="plate_rgb_lucifer",
            image_mode="rgb_only",
            family="arabidopsis_plate_rgb",
            priority=0.2,
            hints={"prefer_high_res": True, "prefer_green": True},
        )
        hades = expert_prior_score(
            features,
            expert_key="plate_bw_hades",
            image_mode="grayscale_only",
            family="arabidopsis_plate_bw",
            priority=0.2,
            hints={"prefer_low_res": True},
        )
        self.assertGreater(lucifer, hades)

    def test_expert_prior_favors_dark_rgb_family_for_dark_rgb_images(self) -> None:
        rgb = np.zeros((96, 96, 3), dtype=np.uint8)
        rgb[..., 0] = 20
        rgb[..., 1] = 22
        rgb[..., 2] = 24
        rgb[:12, 30:36, :] = (230, 210, 160)
        features = describe_image_features(rgb)
        dark_rgb = expert_prior_score(
            features,
            expert_key="arabidopsis_dark_rgb",
            image_mode="rgb_only",
            family="arabidopsis_plate_rgb_dark",
            priority=0.26,
            hints={"prefer_high_res": True, "prefer_dark_top": True},
        )
        lucifer = expert_prior_score(
            features,
            expert_key="plate_rgb_lucifer",
            image_mode="rgb_only",
            family="arabidopsis_plate_rgb",
            priority=0.22,
            hints={"prefer_high_res": True, "prefer_green": True},
        )
        self.assertGreater(dark_rgb, lucifer)

    def test_build_calibration_profile_prefers_best_expert(self) -> None:
        profile = build_calibration_profile(
            [
                {"expert_key": "plate_bw_hades", "label": "BW Arabidopsis (Hades)", "combined_mean_iou": 0.42},
                {"expert_key": "plate_rgb_lucifer", "label": "RGB Arabidopsis (Lucifer)", "combined_mean_iou": 0.61},
                {"expert_key": "potato_rgb", "label": "Potato RGB", "combined_mean_iou": 0.38},
            ],
            selected_uids=["a", "b", "c"],
        )
        self.assertEqual(profile["preferred_expert_key"], "plate_rgb_lucifer")
        self.assertGreater(float(profile["expert_score_bias"]["plate_rgb_lucifer"]), 0.0)
        self.assertLess(float(profile["expert_score_bias"]["potato_rgb"]), 0.0)

    def test_expert_prior_handles_public_minirhizotron_family(self) -> None:
        rgb = np.zeros((96, 96, 3), dtype=np.uint8)
        rgb[:] = (70, 65, 60)
        features = describe_image_features(rgb)
        public_score = expert_prior_score(
            features,
            expert_key="minirhizotron_public",
            image_mode="rgb_only",
            family="minirhizotron_public",
            priority=0.1,
            hints={"prefer_dark_top": True},
        )
        hades_score = expert_prior_score(
            features,
            expert_key="plate_bw_hades",
            image_mode="grayscale_only",
            family="plate_bw_hades",
            priority=0.1,
        )
        self.assertGreater(public_score, hades_score)


if __name__ == "__main__":
    unittest.main()
