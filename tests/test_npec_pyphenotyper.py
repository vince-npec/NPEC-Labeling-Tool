from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "npec_pyphenotyper"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pyphenotyper.data.data_processing import collect_tiles, stitch_predictions
from pyphenotyper.features.features import (
    _predict_tiled_output,
    _postprocess_shoot_mask_bw_arabidopsis,
    _postprocess_shoot_mask_legacy,
    infer_model_profile,
    resolve_pipeline_tuning,
)


class _DummyModel:
    def __init__(self, input_shape, output_shape, source_path: str = ""):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self._npec_source_path = source_path


class _IdentityPredictModel:
    def predict(self, batch, verbose=0):
        return np.asarray(batch)[..., :1]


class NpecPyPhenotyperTests(unittest.TestCase):
    def test_collect_and_stitch_roundtrip_for_single_channel_tiles(self):
        image = np.arange(15 * 23, dtype=np.float32).reshape(15, 23)
        tiles, coords, layout = collect_tiles(image, patch_size=8)
        rebuilt = stitch_predictions(tiles, coords, layout)
        np.testing.assert_array_equal(rebuilt, image)

    def test_infer_binary_profile_from_model_shape(self):
        model = _DummyModel((None, 256, 256, 1), (None, 256, 256, 1), source_path="model_root_15.h5")
        profile = infer_model_profile(model)
        self.assertEqual(profile.mode, "binary_pair")
        self.assertEqual(profile.input_mode, "gray")

    def test_infer_multiclass_profile_from_model_shape(self):
        model = _DummyModel(
            (None, 256, 256, 3),
            (None, 256, 256, 6),
            source_path="npec_trained_model-hades-lucifer_Workstation.keras",
        )
        profile = infer_model_profile(model)
        self.assertEqual(profile.mode, "multiclass")
        self.assertEqual(profile.input_mode, "rgb")

    def test_profile_overrides_can_disable_original_coords(self):
        model = _DummyModel((None, 256, 256, 1), (None, 256, 256, 1), source_path="model_root_14.h5")
        profile = infer_model_profile(model, profile_overrides={"return_original_coords": False})
        self.assertFalse(profile.return_original_coords)

    def test_profile_overrides_enable_halo_tiling(self):
        model = _DummyModel((None, 256, 256, 3), (None, 256, 256, 4))
        profile = infer_model_profile(model, profile_overrides={"tile_halo": 32})
        self.assertEqual(profile.tile_halo, 32)

    def test_profile_normalizes_class_probability_scales(self):
        model = _DummyModel((None, 256, 256, 3), (None, 256, 256, 4))
        profile = infer_model_profile(
            model,
            profile_overrides={"class_probability_scales": [1, 1, 1, 0.25]},
        )
        self.assertEqual(profile.class_probability_scales, (1.0, 1.0, 1.0, 0.25))

    def test_halo_tiling_reconstructs_original_coordinates(self):
        image = np.arange(13 * 17, dtype=np.uint8).reshape(13, 17)
        predicted = _predict_tiled_output(
            _IdentityPredictModel(),
            image,
            patch_size=8,
            batch_size=3,
            input_mode="gray",
            tile_halo=2,
        )
        self.assertEqual(predicted.shape, (13, 17, 1))
        np.testing.assert_allclose(predicted[..., 0], image.astype(np.float32) / 255.0)

    def test_resolve_pipeline_tuning_normalizes_legacy_shoot_mode(self):
        tuning = resolve_pipeline_tuning({"shoot_postprocess_mode": "legacy_shoot"})
        self.assertEqual(tuning.shoot_postprocess_mode, "legacy")

    def test_resolve_pipeline_tuning_normalizes_bw_arabidopsis_mode(self):
        tuning = resolve_pipeline_tuning({"shoot_postprocess_mode": "bw_arabidopsis_solid"})
        self.assertEqual(tuning.shoot_postprocess_mode, "bw_arabidopsis")

    def test_legacy_shoot_postprocess_keeps_center_and_clears_edges(self):
        shoot_mask = np.ones((1200, 2000), dtype=np.uint8)
        processed = _postprocess_shoot_mask_legacy(shoot_mask)
        self.assertEqual(int(processed[:, :650].sum()), 0)
        self.assertEqual(int(processed[:, 1300:].sum()), 0)
        self.assertGreater(int(processed[:250, 900:1100].sum()), 0)
        self.assertGreater(int(processed[900:, 900:1100].sum()), 0)

    def test_legacy_shoot_postprocess_fills_enclosed_leaf_holes(self):
        shoot_mask = np.zeros((1200, 2000), dtype=np.uint8)
        shoot_mask[120:260, 860:1140] = 1
        shoot_mask[165:215, 955:1045] = 0
        processed = _postprocess_shoot_mask_legacy(shoot_mask)
        self.assertEqual(int(processed[190, 1000]), 1)

    def test_bw_arabidopsis_shoot_postprocess_strengthens_solid_leaf_fill(self):
        shoot_mask = np.zeros((1200, 2000), dtype=np.uint8)
        shoot_mask[120:320, 820:1180] = 1
        shoot_mask[170:270, 940:1060] = 0
        processed = _postprocess_shoot_mask_bw_arabidopsis(shoot_mask)
        self.assertEqual(int(processed[220, 1000]), 1)
        self.assertGreater(int(processed.sum()), 0)

    def test_bw_arabidopsis_shoot_postprocess_keeps_five_lanes(self):
        shoot_mask = np.zeros((1200, 2000), dtype=np.uint8)
        root_mask = np.zeros_like(shoot_mask)
        for center_x in (240, 620, 1000, 1380, 1760):
            shoot_mask[120:260, center_x - 45 : center_x + 45] = 1
            root_mask[245:800, center_x - 2 : center_x + 2] = 1
        processed = _postprocess_shoot_mask_bw_arabidopsis(shoot_mask, root_mask=root_mask)
        for center_x in (240, 620, 1000, 1380, 1760):
            self.assertGreater(int(processed[120:260, center_x - 45 : center_x + 45].sum()), 0)


if __name__ == "__main__":
    unittest.main()
