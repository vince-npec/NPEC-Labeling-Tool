from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from resources.five_seedling_ownership import (
    build_ownership_targets,
    discover_five_seedling_corpus,
    grouped_split,
)
from resources.scripts.benchmark_five_seedling_ground_truth import _owner_map_from_boxes
from resources.scripts.experiment_learned_owner_assignment import SharedOwnerRanker
from resources.scripts.run_yang_five_plate_ownership_retrial import (
    _gated_owner,
    _owner_scores,
    _select_threshold,
)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), (np.asarray(mask, dtype=np.uint8) * 255)):
        raise RuntimeError(f"Unable to write test mask: {path}")


class FiveSeedlingOwnershipTests(unittest.TestCase):
    def _build_sample(self, root: Path, sample_id: str, reverse_ids: bool = False) -> None:
        shape = (80, 120)
        image = np.full((*shape, 3), 220, dtype=np.uint8)
        image_path = root / "images" / f"{sample_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        self.assertTrue(cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)))

        semantic = {
            "seed": np.zeros(shape, dtype=np.uint8),
            "shoot": np.zeros(shape, dtype=np.uint8),
            "root": np.zeros(shape, dtype=np.uint8),
            "lateral": np.zeros(shape, dtype=np.uint8),
        }
        centers = [12, 36, 60, 84, 108]
        source_ids = list(reversed(range(1, 6))) if reverse_ids else list(range(1, 6))
        for source_owner, center_x in zip(source_ids, centers, strict=True):
            shoot = np.zeros(shape, dtype=np.uint8)
            shoot[8:18, center_x - 3 : center_x + 4] = 1
            primary = np.zeros(shape, dtype=np.uint8)
            primary[18:70, center_x : center_x + 2] = 1
            lateral = np.zeros(shape, dtype=np.uint8)
            lateral[38:40, max(0, center_x - 8) : center_x] = 1
            _write_mask(
                root / "masks_instances" / sample_id / "shoot" / f"shoot-{source_owner:02d}.png",
                shoot,
            )
            _write_mask(
                root / "masks_instances" / sample_id / "root" / f"root-{source_owner:02d}.png",
                primary,
            )
            _write_mask(
                root / "masks_instances" / sample_id / "lateral" / f"lateral-{source_owner:02d}.png",
                lateral,
            )
            semantic["shoot"] = np.maximum(semantic["shoot"], shoot)
            semantic["root"] = np.maximum(semantic["root"], primary)
            semantic["lateral"] = np.maximum(semantic["lateral"], lateral)

        background = np.logical_not(
            (semantic["seed"] > 0)
            | (semantic["shoot"] > 0)
            | (semantic["root"] > 0)
            | (semantic["lateral"] > 0)
        ).astype(np.uint8)
        indexed = np.full(shape, 5, dtype=np.uint8)
        for organ, class_id in (("seed", 1), ("shoot", 2), ("root", 3), ("lateral", 4)):
            indexed[semantic[organ] > 0] = np.uint8(class_id)
        for organ, mask in (*semantic.items(), ("background", background)):
            _write_mask(root / "masks_binary" / sample_id / f"{organ}.png", mask)
        indexed_path = root / "masks_binary" / sample_id / "indexed_mask.png"
        indexed_path.parent.mkdir(parents=True, exist_ok=True)
        self.assertTrue(cv2.imwrite(str(indexed_path), indexed))

    def test_shoot_position_normalizes_source_instance_numbers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="five_seedling_gt_") as tmp_name:
            root = Path(tmp_name)
            self._build_sample(root, "12-1", reverse_ids=True)
            sample = discover_five_seedling_corpus(root)[0]
            targets = build_ownership_targets(sample)

            self.assertEqual(targets.source_owner_to_slot[5], 1)
            self.assertEqual(targets.source_owner_to_slot[1], 5)
            self.assertEqual(int(targets.owner_mask[30, 12]), 1)
            self.assertEqual(int(targets.owner_mask[30, 108]), 5)
            self.assertEqual(int(np.count_nonzero(targets.conflict_pixels)), 0)
            self.assertEqual(int(targets.semantic_mask[30, 12]), 3)

    def test_conflicting_owner_pixels_are_excluded_from_supervision(self) -> None:
        with tempfile.TemporaryDirectory(prefix="five_seedling_conflict_") as tmp_name:
            root = Path(tmp_name)
            self._build_sample(root, "18-2")
            conflict_path = root / "masks_instances" / "18-2" / "lateral" / "lateral-02.png"
            conflict = cv2.imread(str(conflict_path), cv2.IMREAD_GRAYSCALE)
            assert conflict is not None
            conflict[30, 12] = 255
            self.assertTrue(cv2.imwrite(str(conflict_path), conflict))

            targets = build_ownership_targets(discover_five_seedling_corpus(root)[0])
            self.assertEqual(int(targets.conflict_pixels[30, 12]), 1)
            self.assertEqual(int(targets.valid_owner_pixels[30, 12]), 0)
            self.assertEqual(int(targets.owner_mask[30, 12]), 0)

    def test_grouped_split_keeps_plate_variants_together(self) -> None:
        with tempfile.TemporaryDirectory(prefix="five_seedling_split_") as tmp_name:
            root = Path(tmp_name)
            for sample_id in ("1-1", "1-2", "2-1", "3-1", "4-1", "5-1"):
                self._build_sample(root, sample_id)
            samples = discover_five_seedling_corpus(root)
            train, holdout = grouped_split(samples, holdout_fraction=0.4)
            train_groups = {sample.plate_group for sample in train}
            holdout_groups = {sample.plate_group for sample in holdout}
            self.assertFalse(train_groups & holdout_groups)
            self.assertTrue(train)
            self.assertTrue(holdout)

    def test_benchmark_owner_map_accepts_current_lane_identifiers(self) -> None:
        root_union = np.ones((4, 10), dtype=np.uint8)
        bboxes = {
            "lane_01": [0, 0, 2, 4],
            "lane_02": [2, 0, 2, 4],
            "lane_03": [4, 0, 2, 4],
            "lane_04": [6, 0, 2, 4],
            "lane_05": [8, 0, 2, 4],
        }

        owner = _owner_map_from_boxes(root_union.shape, bboxes, root_union)

        self.assertEqual(owner[0].tolist(), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    def test_shared_owner_ranker_payload_round_trip_preserves_scores(self) -> None:
        ranker = SharedOwnerRanker(l2=0.002)
        ranker.mean_ = np.array([1.0, 2.0], dtype=np.float64)
        ranker.scale_ = np.array([2.0, 4.0], dtype=np.float64)
        ranker.weights_ = np.array([0.3, -0.7], dtype=np.float64)
        features = np.arange(20, dtype=np.float64).reshape(2, 5, 2)

        restored = SharedOwnerRanker.from_payload(ranker.to_payload())

        np.testing.assert_allclose(
            restored.decision_scores(features),
            ranker.decision_scores(features),
        )

    def test_confidence_gate_keeps_lane_below_margin_threshold(self) -> None:
        lane = np.array([[1, 1, 2, 2]], dtype=np.uint8)
        learned = np.array([[1, 2, 1, 2]], dtype=np.uint8)
        margin = np.array([[0.0, 0.8, 0.6, 0.0]], dtype=np.float32)

        gated, switched = _gated_owner(lane, learned, margin, threshold=0.75)

        self.assertEqual(gated.tolist(), [[1, 2, 2, 2]])
        self.assertEqual(switched.tolist(), [[0, 1, 0, 0]])

    def test_owner_scores_penalize_full_image_false_positive_roots(self) -> None:
        owner_mask = np.zeros((8, 8), dtype=np.uint8)
        owner_mask[1:7, 2] = 1
        predicted = owner_mask.copy()
        predicted[3:6, 3] = 1

        class Target:
            valid_owner_pixels = (owner_mask > 0).astype(np.uint8)
            conflict_pixels = np.zeros_like(owner_mask, dtype=np.uint8)

        Target.owner_mask = owner_mask
        scores = _owner_scores(predicted, Target())

        self.assertEqual(scores["conditional_pixel_accuracy"], 1.0)
        self.assertLess(scores["visible_label_full_macro_dice"], 1.0)

    def test_threshold_selection_rejects_large_plate_regression(self) -> None:
        owner_mask = np.zeros((8, 8), dtype=np.uint8)
        owner_mask[1:7, 2] = 1
        owner_mask[1:7, 5] = 2
        learned_owner = owner_mask.copy()
        learned_owner[1:7, 2] = 2
        margin = np.zeros((8, 8), dtype=np.float32)
        margin[1:7, 2] = 1.0

        class Target:
            valid_owner_pixels = (owner_mask > 0).astype(np.uint8)
            conflict_pixels = np.zeros_like(owner_mask, dtype=np.uint8)

        Target.owner_mask = owner_mask

        prepared = [
            {
                "sample": type("Sample", (), {"sample_id": "1-1"})(),
                "lane_owner": owner_mask,
                "learned_owner": learned_owner,
                "margin": margin,
                "target": Target(),
            }
        ]

        selected, rows = _select_threshold(prepared, thresholds=(0.0, 2.0))

        self.assertEqual(selected, 2.0)
        self.assertFalse(rows[0]["passes_constraints"])
        self.assertTrue(rows[1]["passes_constraints"])


if __name__ == "__main__":
    unittest.main()
