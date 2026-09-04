from __future__ import annotations

import unittest

import numpy as np

from resources.analytics_engine import (
    AnalyticsConfig,
    _assign_pixels_to_track_seeds,
    _build_learned_owner_supports,
)
from resources.learned_owner_assignment import (
    SharedOwnerRanker,
    predict_owner_map,
)


def _nearest_crown_ranker() -> SharedOwnerRanker:
    weights = np.zeros(13, dtype=np.float64)
    weights[1] = -1.0
    return SharedOwnerRanker(
        mean_=np.zeros(13, dtype=np.float64),
        scale_=np.ones(13, dtype=np.float64),
        weights_=weights,
    )


class LearnedOwnerAssignmentTests(unittest.TestCase):
    def test_predict_owner_map_assigns_nearest_ordered_crown(self) -> None:
        root = np.zeros((80, 100), dtype=np.uint8)
        root[15:75, 9:12] = 1
        root[15:75, 29:32] = 1
        root[15:75, 49:52] = 1
        root[15:75, 69:72] = 1
        root[15:75, 89:92] = 1
        crowns = np.asarray(
            [[10, 10], [30, 10], [50, 10], [70, 10], [90, 10]],
            dtype=np.float64,
        )

        owner, margin = predict_owner_map(
            root,
            np.zeros_like(root),
            crowns,
            _nearest_crown_ranker(),
            feature_set="crown_coordinates",
        )

        for slot, x in enumerate((10, 30, 50, 70, 90), start=1):
            self.assertTrue(np.all(owner[15:75, x] == slot))
        self.assertTrue(np.all(owner[root == 0] == 0))
        self.assertGreater(float(np.min(margin[root > 0])), 0.0)

    def test_temporal_score_bonus_can_preserve_close_previous_owner(self) -> None:
        root = np.zeros((30, 100), dtype=np.uint8)
        root[10:20, 18:20] = 1
        crowns = np.asarray(
            [[10, 5], [30, 5], [50, 5], [70, 5], [90, 5]],
            dtype=np.float64,
        )
        prior = np.zeros_like(root)
        prior[root > 0] = 2

        without_prior, _ = predict_owner_map(
            root,
            np.zeros_like(root),
            crowns,
            _nearest_crown_ranker(),
            feature_set="crown_coordinates",
        )
        with_prior, _ = predict_owner_map(
            root,
            np.zeros_like(root),
            crowns,
            _nearest_crown_ranker(),
            feature_set="crown_coordinates",
            prior_owner_map=prior,
            temporal_score_bonus=0.3,
        )

        self.assertTrue(np.all(without_prior[root > 0] == 1))
        self.assertTrue(np.all(with_prior[root > 0] == 2))

    def test_graph_supports_are_unique_and_preserve_low_margin_history(self) -> None:
        root = np.zeros((24, 50), dtype=np.uint8)
        root[4:20, 2:48] = 1
        learned = np.zeros_like(root)
        learned[:, :10] = 1
        learned[:, 10:20] = 2
        learned[:, 20:30] = 3
        learned[:, 30:40] = 4
        learned[:, 40:] = 5
        learned[root == 0] = 0
        margin = np.ones(root.shape, dtype=np.float32)
        margin[:, 20:30] = 0.1
        result, learned_meta = _build_learned_owner_supports(
            learned,
            margin,
            [f"lane_{slot:02d}" for slot in range(1, 6)],
            root,
            AnalyticsConfig(
                learned_owner_enabled=True,
                learned_owner_support_margin=0.5,
            ),
        )

        summed = np.zeros_like(root, dtype=np.uint8)
        for mask in result.values():
            summed += (mask > 0).astype(np.uint8)
        expected_support = (root > 0) & ~(
            np.indices(root.shape)[1] >= 20
        )
        expected_support |= (root > 0) & (np.indices(root.shape)[1] >= 30)
        self.assertTrue(np.array_equal(summed > 0, expected_support))
        self.assertEqual(int(np.max(summed)), 1)
        self.assertTrue(bool(learned_meta["applied"]))
        self.assertEqual(
            int(np.count_nonzero(summed[:, 20:30])),
            0,
        )
        self.assertGreater(int(learned_meta["support_pixels"]), 0)

    def test_unary_evidence_does_not_create_a_new_graph_candidate(self) -> None:
        root = np.zeros((40, 40), dtype=np.uint8)
        root[5:35, 18:22] = 1
        seed_masks = {
            "lane_01": np.zeros_like(root),
            "lane_02": np.zeros_like(root),
        }
        seed_masks["lane_01"][5:8, 18:22] = 1
        hints = {
            "lane_01": {"lane_left": 0, "lane_right": 20},
            "lane_02": {"lane_left": 20, "lane_right": 40},
        }
        unary = {
            "lane_01": np.zeros_like(root),
            "lane_02": root.copy(),
        }

        owned, metadata = _assign_pixels_to_track_seeds(
            root,
            seed_masks,
            hints,
            assignment_mode="temporal_graph",
            track_unary_masks=unary,
            unary_weight_px=8.0,
            return_metadata=True,
        )

        self.assertEqual(int(np.count_nonzero(owned["lane_01"])), int(np.count_nonzero(root)))
        self.assertEqual(int(np.count_nonzero(owned["lane_02"])), 0)
        self.assertEqual(int(metadata["shared_components"]), 0)

    def test_unary_fusion_cannot_increase_graph_ambiguity(self) -> None:
        root = np.zeros((50, 90), dtype=np.uint8)
        root[23:28, 8:82] = 1
        seed_masks = {
            "lane_01": np.zeros_like(root),
            "lane_02": np.zeros_like(root),
        }
        seed_masks["lane_01"][23:28, 8:12] = 1
        seed_masks["lane_02"][23:28, 78:82] = 1
        hints = {
            "lane_01": {"lane_left": 0, "lane_right": 90},
            "lane_02": {"lane_left": 0, "lane_right": 90},
        }

        baseline, baseline_meta = _assign_pixels_to_track_seeds(
            root,
            seed_masks,
            hints,
            assignment_mode="temporal_graph",
            ambiguity_margin_px=3.0,
            return_metadata=True,
        )
        unary = {
            "lane_01": root.copy(),
            "lane_02": np.zeros_like(root),
        }
        fused, fused_meta = _assign_pixels_to_track_seeds(
            root,
            seed_masks,
            hints,
            assignment_mode="temporal_graph",
            track_unary_masks=unary,
            unary_weight_px=8.0,
            ambiguity_margin_px=3.0,
            return_metadata=True,
        )

        self.assertGreater(int(baseline_meta["ambiguous_pixels"]), 0)
        self.assertLessEqual(
            int(fused_meta["ambiguous_pixels"]),
            int(baseline_meta["ambiguous_pixels"]),
        )
        baseline_owner = np.zeros_like(root, dtype=np.uint8)
        fused_owner = np.zeros_like(root, dtype=np.uint8)
        for owner_id, track_id in enumerate(("lane_01", "lane_02"), start=1):
            baseline_owner[baseline[track_id] > 0] = owner_id
            fused_owner[fused[track_id] > 0] = owner_id
        baseline_ambiguity = np.asarray(
            baseline_meta["ambiguity_mask"],
            dtype=np.uint8,
        ) > 0
        self.assertTrue(
            np.array_equal(
                baseline_owner[~baseline_ambiguity],
                fused_owner[~baseline_ambiguity],
            )
        )
        self.assertEqual(
            int(np.count_nonzero(fused_owner > 0)),
            int(np.count_nonzero(root)),
        )


if __name__ == "__main__":
    unittest.main()
