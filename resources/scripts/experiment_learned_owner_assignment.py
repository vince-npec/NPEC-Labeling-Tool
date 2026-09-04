from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.five_seedling_ownership import (  # noqa: E402
    FiveSeedlingSample,
    build_ownership_targets,
    discover_five_seedling_corpus,
    grouped_split,
    load_instance_masks,
    load_semantic_masks,
    shoot_ordered_slot_mapping,
)


DEFAULT_CORPUS = REPO_ROOT / "data" / "yang_ground_truth"
CHALLENGE_SAMPLE_IDS = ("78-3", "73-4", "79-3", "77-4", "1-4")
SPLIT_SALT = "npec-owner-feasibility-v1"
FEATURE_SETS = ("crown_coordinates", "crown_coordinates_orientation")
_VALID_SAMPLE_RE = re.compile(r"^\d+-\d+$")


def _stable_seed(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _mask_crown_point(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(np.asarray(mask, dtype=np.uint8) > 0)
    if xs.size <= 0:
        return None
    lower_cutoff = float(np.quantile(ys.astype(np.float64), 0.82))
    keep = ys >= lower_cutoff
    if int(np.count_nonzero(keep)) < 8:
        keep = np.ones_like(ys, dtype=bool)
    return float(np.median(xs[keep])), float(np.median(ys[keep]))


def _ordered_crowns(sample: FiveSeedlingSample) -> np.ndarray:
    source_to_slot = shoot_ordered_slot_mapping(sample, expected_count=5)
    shoot_masks = load_instance_masks(sample, "shoot")
    crowns = np.full((5, 2), np.nan, dtype=np.float64)
    for source_owner, slot in source_to_slot.items():
        point = _mask_crown_point(shoot_masks.get(source_owner, np.zeros((1, 1), dtype=np.uint8)))
        if point is not None and 1 <= slot <= 5:
            crowns[slot - 1] = point
    if np.any(~np.isfinite(crowns)):
        raise ValueError(f"Five ordered shoot crowns are required for sample {sample.sample_id}.")
    return crowns


def _orientation_channels(root_union: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(root_union, dtype=np.uint8)
    small_w = max(32, int(round(mask.shape[1] / 4.0)))
    small_h = max(32, int(round(mask.shape[0] / 4.0)))
    small = cv2.resize(mask.astype(np.float32), (small_w, small_h), interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(small, (0, 0), sigmaX=1.35, sigmaY=1.35)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    tangent_x = -gy
    tangent_y = gx
    magnitude = np.sqrt(tangent_x * tangent_x + tangent_y * tangent_y)
    valid = magnitude > 1e-6
    tangent_x[valid] /= magnitude[valid]
    tangent_y[valid] /= magnitude[valid]
    tangent_y_negative = tangent_y < 0
    tangent_x[tangent_y_negative] *= -1
    tangent_y[tangent_y_negative] *= -1
    return tangent_x, tangent_y


def _sample_orientation(
    channels: tuple[np.ndarray, np.ndarray],
    ys: np.ndarray,
    xs: np.ndarray,
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    tangent_x, tangent_y = channels
    scaled_x = np.clip(
        np.rint(xs.astype(np.float64) * tangent_x.shape[1] / shape_hw[1]).astype(np.int64),
        0,
        tangent_x.shape[1] - 1,
    )
    scaled_y = np.clip(
        np.rint(ys.astype(np.float64) * tangent_y.shape[0] / shape_hw[0]).astype(np.int64),
        0,
        tangent_y.shape[0] - 1,
    )
    return tangent_x[scaled_y, scaled_x].astype(np.float64), tangent_y[scaled_y, scaled_x].astype(
        np.float64
    )


def _pair_features(
    xs: np.ndarray,
    ys: np.ndarray,
    crowns_xy: np.ndarray,
    shape_hw: tuple[int, int],
    *,
    feature_set: str,
    tangent_xy: tuple[np.ndarray, np.ndarray] | None,
    lateral_flags: np.ndarray,
) -> np.ndarray:
    height, width = shape_hw
    x = xs.astype(np.float64)[:, None] / max(float(width - 1), 1.0)
    y = ys.astype(np.float64)[:, None] / max(float(height - 1), 1.0)
    crown_x = crowns_xy[:, 0][None, :] / max(float(width - 1), 1.0)
    crown_y = crowns_xy[:, 1][None, :] / max(float(height - 1), 1.0)
    dx = x - crown_x
    dy = y - crown_y
    abs_dx = np.abs(dx)
    slot = np.linspace(-1.0, 1.0, 5, dtype=np.float64)[None, :]
    left_gap = np.empty(5, dtype=np.float64)
    right_gap = np.empty(5, dtype=np.float64)
    crown_x_flat = crown_x[0]
    left_gap[0] = crown_x_flat[1] - crown_x_flat[0]
    left_gap[1:] = np.diff(crown_x_flat)
    right_gap[-1] = crown_x_flat[-1] - crown_x_flat[-2]
    right_gap[:-1] = np.diff(crown_x_flat)
    local_scale = np.maximum((left_gap + right_gap) * 0.5, 0.025)[None, :]
    lane_dx = dx / local_scale

    columns = [
        dx,
        abs_dx,
        dx * dx,
        abs_dx * y,
        dx * y,
        lane_dx,
        np.abs(lane_dx),
        lane_dx * lane_dx,
        dy,
        dy * dy,
        np.broadcast_to(slot, dx.shape),
        np.broadcast_to(slot, dx.shape) * x,
        np.broadcast_to(slot, dx.shape) * y,
    ]
    if feature_set == "crown_coordinates_orientation":
        if tangent_xy is None:
            raise ValueError("Orientation features require tangent channels.")
        tangent_x, tangent_y = tangent_xy
        tx = tangent_x[:, None]
        ty = tangent_y[:, None]
        distance = np.sqrt(dx * dx + dy * dy) + 1e-6
        cross_alignment = np.abs(dx * ty - dy * tx) / distance
        dot_alignment = np.abs(dx * tx + dy * ty) / distance
        safe_ty = np.where(np.abs(ty) >= 0.12, ty, np.where(ty >= 0, 0.12, -0.12))
        projected_crown_dx = np.clip(dx - dy * tx / safe_ty, -2.0, 2.0)
        lateral = lateral_flags.astype(np.float64)[:, None]
        columns.extend(
            [
                cross_alignment,
                dot_alignment,
                np.abs(projected_crown_dx),
                projected_crown_dx * projected_crown_dx,
                lateral * abs_dx,
                lateral * np.abs(lane_dx),
                lateral * cross_alignment,
            ]
        )
    return np.stack(columns, axis=2).astype(np.float64)


def _sample_training_pixels(
    sample: FiveSeedlingSample,
    *,
    max_pixels_per_owner: int,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray]:
    target = build_ownership_targets(sample, owner_organs=("root", "lateral"))
    semantic = load_semantic_masks(sample)
    root_union = np.logical_or(semantic["root"] > 0, semantic["lateral"] > 0)
    crowns = _ordered_crowns(sample)
    orientation = (
        _orientation_channels(root_union.astype(np.uint8))
        if feature_set == "crown_coordinates_orientation"
        else None
    )
    rng = np.random.default_rng(_stable_seed(f"{sample.sample_id}:{feature_set}"))
    selected_y: list[np.ndarray] = []
    selected_x: list[np.ndarray] = []
    selected_owner: list[np.ndarray] = []
    for owner in range(1, 6):
        ys, xs = np.where(
            (target.owner_mask == owner)
            & (target.valid_owner_pixels > 0)
            & root_union
        )
        if xs.size > max_pixels_per_owner:
            indices = rng.choice(xs.size, size=max_pixels_per_owner, replace=False)
            ys = ys[indices]
            xs = xs[indices]
        selected_y.append(ys)
        selected_x.append(xs)
        selected_owner.append(np.full(xs.size, owner - 1, dtype=np.int64))
    ys = np.concatenate(selected_y)
    xs = np.concatenate(selected_x)
    owners = np.concatenate(selected_owner)
    tangent = None if orientation is None else _sample_orientation(orientation, ys, xs, root_union.shape)
    lateral = semantic["lateral"][ys, xs] > 0
    features = _pair_features(
        xs,
        ys,
        crowns,
        root_union.shape,
        feature_set=feature_set,
        tangent_xy=tangent,
        lateral_flags=lateral,
    )
    return features, owners


class SharedOwnerRanker:
    def __init__(self, *, l2: float = 1e-3) -> None:
        self.l2 = float(l2)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.optimization_: dict[str, object] = {}

    def fit(self, features: np.ndarray, owners: np.ndarray) -> "SharedOwnerRanker":
        from scipy.optimize import minimize

        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(owners, dtype=np.int64)
        flat = x.reshape(-1, x.shape[-1])
        mean = np.mean(flat, axis=0)
        scale = np.std(flat, axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (x - mean[None, None, :]) / scale[None, None, :]

        def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
            scores = np.tensordot(standardized, weights, axes=([2], [0]))
            scores -= np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            row = np.arange(y.size)
            loss = -float(np.mean(np.log(probabilities[row, y] + 1e-12)))
            loss += 0.5 * self.l2 * float(np.dot(weights, weights))
            probabilities[row, y] -= 1.0
            gradient = np.mean(
                np.sum(probabilities[:, :, None] * standardized, axis=1),
                axis=0,
            )
            gradient += self.l2 * weights
            return loss, gradient

        result = minimize(
            objective,
            np.zeros(x.shape[-1], dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 180, "ftol": 1e-10, "gtol": 1e-7},
        )
        self.mean_ = mean
        self.scale_ = scale
        self.weights_ = np.asarray(result.x, dtype=np.float64)
        self.optimization_ = {
            "success": bool(result.success),
            "status": int(result.status),
            "iterations": int(result.nit),
            "loss": float(result.fun),
            "message": str(result.message),
        }
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        scores = self.decision_scores(features)
        return np.argmax(scores, axis=1).astype(np.uint8) + np.uint8(1)

    def decision_scores(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.weights_ is None:
            raise RuntimeError("Ranker has not been fitted.")
        x = (np.asarray(features, dtype=np.float64) - self.mean_[None, None, :]) / self.scale_[
            None, None, :
        ]
        return np.tensordot(x, self.weights_, axes=([2], [0]))

    def to_payload(self) -> dict[str, object]:
        if self.mean_ is None or self.scale_ is None or self.weights_ is None:
            raise RuntimeError("Ranker has not been fitted.")
        return {
            "model_type": "shared_owner_linear_ranker",
            "l2": float(self.l2),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "weights": self.weights_.tolist(),
            "optimization": dict(self.optimization_),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "SharedOwnerRanker":
        model = cls(l2=float(payload.get("l2", 1e-3)))
        model.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        model.scale_ = np.asarray(payload["scale"], dtype=np.float64)
        model.weights_ = np.asarray(payload["weights"], dtype=np.float64)
        model.optimization_ = dict(payload.get("optimization") or {})
        if (
            model.mean_.ndim != 1
            or model.scale_.shape != model.mean_.shape
            or model.weights_.shape != model.mean_.shape
        ):
            raise ValueError("Invalid shared owner ranker payload dimensions.")
        return model


def _predict_sample(
    sample: FiveSeedlingSample,
    *,
    method: str,
    ranker: SharedOwnerRanker | None,
    feature_set: str | None,
    chunk_size: int,
) -> tuple[np.ndarray, object]:
    target = build_ownership_targets(sample, owner_organs=("root", "lateral"))
    semantic = load_semantic_masks(sample)
    root_union = np.logical_or(semantic["root"] > 0, semantic["lateral"] > 0)
    crowns = _ordered_crowns(sample)
    ys, xs = np.where(root_union)
    predicted = np.zeros(root_union.shape, dtype=np.uint8)
    if method == "nearest_crown_x":
        distances = np.abs(xs.astype(np.float64)[:, None] - crowns[:, 0][None, :])
        predicted[ys, xs] = np.argmin(distances, axis=1).astype(np.uint8) + np.uint8(1)
        return predicted, target
    if ranker is None or feature_set is None:
        raise ValueError("Learned prediction requires a ranker and feature set.")
    orientation = (
        _orientation_channels(root_union.astype(np.uint8))
        if feature_set == "crown_coordinates_orientation"
        else None
    )
    for start in range(0, xs.size, max(1, int(chunk_size))):
        stop = min(xs.size, start + max(1, int(chunk_size)))
        chunk_y = ys[start:stop]
        chunk_x = xs[start:stop]
        tangent = (
            None
            if orientation is None
            else _sample_orientation(orientation, chunk_y, chunk_x, root_union.shape)
        )
        features = _pair_features(
            chunk_x,
            chunk_y,
            crowns,
            root_union.shape,
            feature_set=feature_set,
            tangent_xy=tangent,
            lateral_flags=semantic["lateral"][chunk_y, chunk_x] > 0,
        )
        predicted[chunk_y, chunk_x] = ranker.predict(features)
    return predicted, target


def _score_prediction(predicted: np.ndarray, target: object) -> dict[str, float | int]:
    expected = np.asarray(target.owner_mask, dtype=np.uint8)
    valid = np.asarray(target.valid_owner_pixels, dtype=np.uint8) > 0
    valid_count = int(np.count_nonzero(valid))
    correct = int(np.count_nonzero((predicted == expected) & valid))
    dice: list[float] = []
    owner_pixel_ape: list[float] = []
    for owner in range(1, 6):
        truth = (expected == owner) & valid
        pred = (predicted == owner) & valid
        truth_count = int(np.count_nonzero(truth))
        pred_count = int(np.count_nonzero(pred))
        if truth_count <= 0:
            continue
        intersection = int(np.count_nonzero(truth & pred))
        dice.append(float((2 * intersection) / max(truth_count + pred_count, 1)))
        owner_pixel_ape.append(float(abs(pred_count - truth_count) / truth_count))
    return {
        "valid_owner_pixels": valid_count,
        "conflict_pixels_ignored": int(np.count_nonzero(target.conflict_pixels)),
        "pixel_accuracy": float(correct / max(valid_count, 1)),
        "macro_owner_dice": float(np.mean(dice)) if dice else 0.0,
        "owner_pixel_count_mape": float(np.mean(owner_pixel_ape)) if owner_pixel_ape else 0.0,
    }


def _evaluate(
    samples: Iterable[FiveSeedlingSample],
    *,
    method: str,
    ranker: SharedOwnerRanker | None = None,
    feature_set: str | None = None,
    chunk_size: int = 50_000,
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    total_valid = 0
    total_correct = 0.0
    for sample in samples:
        prediction, target = _predict_sample(
            sample,
            method=method,
            ranker=ranker,
            feature_set=feature_set,
            chunk_size=chunk_size,
        )
        metrics = _score_prediction(prediction, target)
        valid = int(metrics["valid_owner_pixels"])
        total_valid += valid
        total_correct += float(metrics["pixel_accuracy"]) * valid
        cases.append({"sample_id": sample.sample_id, "plate_group": sample.plate_group, **metrics})
    return {
        "sample_count": len(cases),
        "micro_pixel_accuracy": float(total_correct / max(total_valid, 1)),
        "mean_pixel_accuracy": float(np.mean([case["pixel_accuracy"] for case in cases]))
        if cases
        else 0.0,
        "mean_macro_owner_dice": float(np.mean([case["macro_owner_dice"] for case in cases]))
        if cases
        else 0.0,
        "mean_owner_pixel_count_mape": float(
            np.mean([case["owner_pixel_count_mape"] for case in cases])
        )
        if cases
        else 0.0,
        "cases": cases,
    }


def _audit(samples: list[FiveSeedlingSample]) -> dict[str, object]:
    instance_counts: dict[str, Counter[int]] = {
        organ: Counter() for organ in ("seed", "shoot", "root", "lateral")
    }
    total_valid = 0
    total_conflict = 0
    for sample in samples:
        for organ in instance_counts:
            instance_counts[organ][len(load_instance_masks(sample, organ))] += 1
        target = build_ownership_targets(sample, owner_organs=("root", "lateral"))
        total_valid += int(np.count_nonzero(target.valid_owner_pixels))
        total_conflict += int(np.count_nonzero(target.conflict_pixels))
    return {
        "sample_count": len(samples),
        "plate_group_count": len({sample.plate_group for sample in samples}),
        "image_shape": [4218, 4267],
        "instance_count_histograms": {
            organ: {str(count): int(frequency) for count, frequency in sorted(histogram.items())}
            for organ, histogram in instance_counts.items()
        },
        "valid_visible_root_pixels": total_valid,
        "conflicting_owner_pixels_ignored": total_conflict,
        "scientific_label_scope": {
            "root_and_shoot_instances": "complete five-owner layers in all discovered samples",
            "lateral_layers": "absent layers can mean no labelled visible lateral; absence is not a hidden-root target",
            "bacterial_occlusion_gaps": "background/unknown, excluded from supervision",
            "crown_points": "oracle points derived from hand-labelled shoot instances",
        },
    }


def run_experiment(
    corpus: Path,
    *,
    max_pixels_per_owner: int,
    chunk_size: int,
) -> dict[str, object]:
    discovered = discover_five_seedling_corpus(corpus)
    challenge_lookup = {sample.sample_id: sample for sample in discovered}
    missing = [sample_id for sample_id in CHALLENGE_SAMPLE_IDS if sample_id not in challenge_lookup]
    if missing:
        raise ValueError(f"Missing required challenge samples: {missing}")
    challenge = [challenge_lookup[sample_id] for sample_id in CHALLENGE_SAMPLE_IDS]
    challenge_groups = {sample.plate_group for sample in challenge}
    ambiguous = [sample for sample in discovered if not _VALID_SAMPLE_RE.match(sample.sample_id)]
    eligible = [
        sample
        for sample in discovered
        if sample.plate_group not in challenge_groups and _VALID_SAMPLE_RE.match(sample.sample_id)
    ]
    train, development = grouped_split(
        eligible,
        holdout_fraction=0.20,
        salt=SPLIT_SALT,
    )
    train_groups = {sample.plate_group for sample in train}
    development_groups = {sample.plate_group for sample in development}
    if train_groups & development_groups or train_groups & challenge_groups or development_groups & challenge_groups:
        raise RuntimeError("Plate-group leakage detected in ownership experiment split.")

    baseline_development = _evaluate(
        development,
        method="nearest_crown_x",
        chunk_size=chunk_size,
    )
    models: dict[str, SharedOwnerRanker] = {}
    development_results: dict[str, dict[str, object]] = {
        "nearest_crown_x": baseline_development
    }
    training_rows: dict[str, int] = {}
    for feature_set in FEATURE_SETS:
        feature_batches: list[np.ndarray] = []
        owner_batches: list[np.ndarray] = []
        for sample in train:
            features, owners = _sample_training_pixels(
                sample,
                max_pixels_per_owner=max_pixels_per_owner,
                feature_set=feature_set,
            )
            feature_batches.append(features)
            owner_batches.append(owners)
        train_features = np.concatenate(feature_batches, axis=0)
        train_owners = np.concatenate(owner_batches, axis=0)
        ranker = SharedOwnerRanker(l2=1e-3).fit(train_features, train_owners)
        models[feature_set] = ranker
        training_rows[feature_set] = int(train_owners.size)
        development_results[feature_set] = _evaluate(
            development,
            method="learned",
            ranker=ranker,
            feature_set=feature_set,
            chunk_size=chunk_size,
        )

    selected_feature_set = max(
        FEATURE_SETS,
        key=lambda name: (
            float(development_results[name]["mean_macro_owner_dice"]),
            float(development_results[name]["micro_pixel_accuracy"]),
        ),
    )
    selected_model = models[selected_feature_set]

    # The challenge is touched only after feature-set selection on development data.
    challenge_baseline = _evaluate(
        challenge,
        method="nearest_crown_x",
        chunk_size=chunk_size,
    )
    challenge_selected = _evaluate(
        challenge,
        method="learned",
        ranker=selected_model,
        feature_set=selected_feature_set,
        chunk_size=chunk_size,
    )
    return {
        "experiment": "five-seedling learned owner assignment feasibility",
        "corpus": str(corpus),
        "audit": _audit(discovered),
        "split": {
            "split_function": "resources.five_seedling_ownership.grouped_split",
            "salt": SPLIT_SALT,
            "train_samples": len(train),
            "train_groups": len(train_groups),
            "development_samples": len(development),
            "development_groups": len(development_groups),
            "development_sample_ids": [sample.sample_id for sample in development],
            "challenge_sample_ids": list(CHALLENGE_SAMPLE_IDS),
            "challenge_groups": sorted(challenge_groups),
            "all_samples_excluded_for_challenge_groups": sorted(
                sample.sample_id for sample in discovered if sample.plate_group in challenge_groups
            ),
            "ambiguous_identity_samples_excluded": [sample.sample_id for sample in ambiguous],
            "leakage_check": "passed",
        },
        "input_scope": {
            "root_pixels": "oracle visible hand-labelled primary+lateral union",
            "crowns": "oracle lower shoot-instance points, ordered left-to-right",
            "coordinates": "normalized x/y and adaptive crown spacing",
            "orientation": "quarter-resolution local root-mask tangent",
            "organ_context": "oracle primary-versus-lateral flag",
            "raw_rgb": "not used",
            "occluded_hidden_roots": "not inferred or scored",
        },
        "training": {
            "sampled_pixels_per_owner_per_image_max": int(max_pixels_per_owner),
            "training_pixel_rows": training_rows,
            "models": {
                feature_set: models[feature_set].optimization_ for feature_set in FEATURE_SETS
            },
        },
        "development": development_results,
        "selection": {
            "criterion": "highest development mean_macro_owner_dice; micro accuracy tie-break",
            "selected_feature_set": selected_feature_set,
            "challenge_not_used_for_selection": True,
        },
        "selected_model": {
            "feature_set": selected_feature_set,
            "ranker": selected_model.to_payload(),
        },
        "challenge": {
            "nearest_crown_x": challenge_baseline,
            "selected_learned_model": challenge_selected,
        },
        "limitations": [
            "This is an oracle-mask ownership experiment, not end-to-end RGB segmentation.",
            "Oracle hand-labelled shoot instances provide crown locations; production must detect five crowns.",
            "A pixel classifier cannot recover biological provenance after two roots merge at a crossing.",
            "Bacterial gaps remain unknown; synthetic closing must be trained as a separate hidden-root target.",
            "Fifty-nine images across forty-nine groups are enough for feasibility, not a robust owner network.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Leakage-safe Yang five-seedling owner-assignment feasibility experiment."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--max-pixels-per-owner", type=int, default=350)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    report = run_experiment(
        args.corpus.expanduser(),
        max_pixels_per_owner=max(25, int(args.max_pixels_per_owner)),
        chunk_size=max(1_000, int(args.chunk_size)),
    )
    rendered = json.dumps(report, indent=2)
    if args.output_json is not None:
        output_path = args.output_json.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
