from __future__ import annotations

import unittest

from resources.analytics_engine import AnalyticsConfig
from resources.app import NpecLabelingMainWindow


class _OwnerProfileWindow:
    _apply_builtin_five_seedling_owner_profile = (
        NpecLabelingMainWindow._apply_builtin_five_seedling_owner_profile
    )

    @staticmethod
    def _detect_builtin_five_seedling_owner_profile() -> dict[str, object]:
        return {
            "feature_set": "crown_coordinates_orientation",
            "ranker": {
                "model_type": "shared_owner_linear_ranker",
                "mean": [0.0],
                "scale": [1.0],
                "weights": [0.0],
            },
            "support_margin": 0.75,
            "prior_weight_px": 3.0,
            "temporal_score_bonus": 0.15,
            "qc_margin": 0.2,
        }


class AppLearnedOwnerProfileTests(unittest.TestCase):
    def test_profile_is_enabled_only_for_five_crown_graph_tracking(self) -> None:
        window = _OwnerProfileWindow()
        config = AnalyticsConfig(
            root_class_id=1,
            expected_track_count=5,
            tracking_mode="arabidopsis_crown_lanes",
            root_ownership_assignment_mode="temporal_graph",
        )

        applied = window._apply_builtin_five_seedling_owner_profile(config)

        self.assertTrue(applied)
        self.assertTrue(config.learned_owner_enabled)
        self.assertIsInstance(config.learned_owner_ranker_payload, dict)
        self.assertEqual(config.learned_owner_support_margin, 0.75)
        self.assertEqual(config.learned_owner_prior_weight_px, 3.0)

        config.expected_track_count = 1
        applied = window._apply_builtin_five_seedling_owner_profile(config)
        self.assertFalse(applied)
        self.assertFalse(config.learned_owner_enabled)
        self.assertIsNone(config.learned_owner_ranker_payload)


if __name__ == "__main__":
    unittest.main()
