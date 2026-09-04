from __future__ import annotations

import unittest

import numpy as np

from resources.human_in_the_loop import (
    auto_accept_pseudo_labels,
    build_hitl_candidate_rows,
    build_retrain_snapshot,
    stage_pseudo_labels,
)
from resources.models import DatasetImageItem, LabelClass


class HumanInTheLoopTests(unittest.TestCase):
    def test_build_rows_prioritizes_pending_unlabeled_images(self) -> None:
        image = np.zeros((6, 6, 3), dtype=np.uint8)
        items = [
            DatasetImageItem(uid="u1", name="img1.png", path=None, image=image),  # type: ignore[arg-type]
            DatasetImageItem(uid="u2", name="img2.png", path=None, image=image),  # type: ignore[arg-type]
        ]
        annotations = {
            "u1": {1: np.zeros((6, 6), dtype=np.uint8)},
            "u2": {1: np.ones((6, 6), dtype=np.uint8)},
        }
        predictions = {
            "u1": np.ones((6, 6), dtype=np.uint8),
            "u2": np.ones((6, 6), dtype=np.uint8),
        }
        metadata = {
            "u1": {"confidence_score": 0.92, "uncertainty_score": 0.08},
            "u2": {"confidence_score": 0.40, "uncertainty_score": 0.60},
        }
        pseudo = {"u1": np.ones((6, 6), dtype=np.uint8)}
        review = {"u1": {"status": "pending"}}
        rows = build_hitl_candidate_rows(items, annotations, predictions, metadata, pseudo, review)
        self.assertEqual(rows[0]["uid"], "u1")
        self.assertEqual(rows[0]["recommendation"], "Review pseudo-label")
        self.assertTrue(bool(rows[0]["has_pseudo_label"]))
        self.assertTrue(bool(rows[1]["labeled"]))

    def test_stage_pseudo_labels_skips_labeled_and_low_confidence(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        items = [
            DatasetImageItem(uid="u1", name="img1.png", path=None, image=image),  # type: ignore[arg-type]
            DatasetImageItem(uid="u2", name="img2.png", path=None, image=image),  # type: ignore[arg-type]
            DatasetImageItem(uid="u3", name="img3.png", path=None, image=image),  # type: ignore[arg-type]
        ]
        annotations = {
            "u1": {1: np.zeros((8, 8), dtype=np.uint8)},
            "u2": {1: np.ones((8, 8), dtype=np.uint8)},
            "u3": {1: np.zeros((8, 8), dtype=np.uint8)},
        }
        predictions = {
            "u1": np.ones((8, 8), dtype=np.uint8),
            "u2": np.ones((8, 8), dtype=np.uint8),
            "u3": np.ones((8, 8), dtype=np.uint8),
        }
        metadata = {
            "u1": {"confidence_score": 0.90, "uncertainty_score": 0.10},
            "u2": {"confidence_score": 0.95, "uncertainty_score": 0.05},
            "u3": {"confidence_score": 0.40, "uncertainty_score": 0.60},
        }
        pseudo: dict[str, np.ndarray] = {}
        review: dict[str, dict[str, object]] = {}
        result = stage_pseudo_labels(
            items,
            annotations,
            predictions,
            metadata,
            pseudo,
            review,
            min_confidence=0.80,
        )
        self.assertEqual(int(result["staged"]), 1)
        self.assertIn("u1", pseudo)
        self.assertNotIn("u2", pseudo)
        self.assertNotIn("u3", pseudo)
        self.assertEqual(str(review["u1"]["status"]), "pending")

    def test_auto_accept_and_snapshot_include_manual_and_pseudo_training_items(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        items = [
            DatasetImageItem(uid="manual", name="manual.png", path=None, image=image),  # type: ignore[arg-type]
            DatasetImageItem(uid="pseudo", name="pseudo.png", path=None, image=image),  # type: ignore[arg-type]
            DatasetImageItem(uid="skip", name="skip.png", path=None, image=image),  # type: ignore[arg-type]
        ]
        classes = [
            LabelClass(1, "Root", "#ffd166"),
            LabelClass(2, "Shoot", "#56f39a"),
            LabelClass(3, "Lateral Root", "#3fc1ff"),
        ]
        annotations = {
            "manual": {
                1: np.ones((8, 8), dtype=np.uint8),
                2: np.zeros((8, 8), dtype=np.uint8),
                3: np.zeros((8, 8), dtype=np.uint8),
            },
            "pseudo": {
                1: np.zeros((8, 8), dtype=np.uint8),
                2: np.zeros((8, 8), dtype=np.uint8),
                3: np.zeros((8, 8), dtype=np.uint8),
            },
            "skip": {
                1: np.zeros((8, 8), dtype=np.uint8),
                2: np.zeros((8, 8), dtype=np.uint8),
                3: np.zeros((8, 8), dtype=np.uint8),
            },
        }
        pseudo_labels = {
            "pseudo": np.full((8, 8), 2, dtype=np.uint8),
            "skip": np.full((8, 8), 1, dtype=np.uint8),
        }
        review = {
            "pseudo": {"status": "pending", "confidence_score": 0.95},
            "skip": {"status": "pending", "confidence_score": 0.40},
        }

        result = auto_accept_pseudo_labels(
            items,
            annotations,
            pseudo_labels,
            review,
            min_confidence=0.80,
        )
        self.assertEqual(int(result["auto_accepted"]), 1)
        self.assertEqual(str(review["pseudo"]["status"]), "auto-accepted")
        self.assertEqual(str(review["skip"]["status"]), "pending")

        snapshot_items, snapshot_annotations, summary = build_retrain_snapshot(
            items,
            classes,
            annotations,
            pseudo_labels,
            review,
        )
        self.assertEqual(int(summary["total_items"]), 2)
        self.assertIn("manual", summary["uids"])
        self.assertIn("pseudo", summary["uids"])
        self.assertNotIn("skip", summary["uids"])
        self.assertEqual(int(np.count_nonzero(snapshot_annotations["pseudo"][2])), 64)
        self.assertEqual(int(np.count_nonzero(snapshot_annotations["pseudo"][1])), 0)


if __name__ == "__main__":
    unittest.main()
