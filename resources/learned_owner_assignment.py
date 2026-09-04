from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


FEATURE_SETS = ("crown_coordinates", "crown_coordinates_orientation")


def _bfgs_minimize(
    objective: Any,
    initial: np.ndarray,
    *,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, object]]:
    weights = np.asarray(initial, dtype=np.float64).copy()
    inverse_hessian = np.eye(weights.size, dtype=np.float64)
    loss, gradient = objective(weights)
    success = False
    message = "maximum iterations reached"
    iterations = 0
    for iteration in range(max(1, int(max_iterations))):
        iterations = iteration + 1
        if float(np.max(np.abs(gradient))) <= 1.0e-7:
            success = True
            message = "gradient tolerance reached"
            break
        direction = -(inverse_hessian @ gradient)
        directional_derivative = float(np.dot(gradient, direction))
        if directional_derivative >= -1.0e-12:
            inverse_hessian = np.eye(weights.size, dtype=np.float64)
            direction = -gradient
            directional_derivative = -float(np.dot(gradient, gradient))

        step = 1.0
        candidate_loss = float("inf")
        candidate_gradient = gradient
        candidate_weights = weights
        while step >= 1.0e-8:
            proposed = weights + (step * direction)
            proposed_loss, proposed_gradient = objective(proposed)
            if (
                np.isfinite(proposed_loss)
                and proposed_loss <= loss + (1.0e-4 * step * directional_derivative)
            ):
                candidate_weights = proposed
                candidate_loss = float(proposed_loss)
                candidate_gradient = np.asarray(proposed_gradient, dtype=np.float64)
                break
            step *= 0.5
        if step < 1.0e-8:
            message = "line search failed"
            break

        delta_weights = candidate_weights - weights
        delta_gradient = candidate_gradient - gradient
        curvature = float(np.dot(delta_gradient, delta_weights))
        if curvature > 1.0e-12:
            rho = 1.0 / curvature
            identity = np.eye(weights.size, dtype=np.float64)
            left = identity - (rho * np.outer(delta_weights, delta_gradient))
            right = identity - (rho * np.outer(delta_gradient, delta_weights))
            inverse_hessian = (
                left @ inverse_hessian @ right
            ) + (rho * np.outer(delta_weights, delta_weights))
        else:
            inverse_hessian = np.eye(weights.size, dtype=np.float64)

        relative_change = abs(float(loss) - candidate_loss) / max(1.0, abs(float(loss)))
        weights = candidate_weights
        loss = candidate_loss
        gradient = candidate_gradient
        if relative_change <= 1.0e-10:
            success = True
            message = "relative loss tolerance reached"
            break
    return weights, {
        "success": bool(success),
        "status": 0 if success else 1,
        "iterations": int(iterations),
        "loss": float(loss),
        "message": str(message),
        "optimizer": "numpy_bfgs",
    }


def orientation_channels(root_union: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (np.asarray(root_union, dtype=np.uint8) > 0).astype(np.uint8)
    small_w = max(32, int(round(mask.shape[1] / 4.0)))
    small_h = max(32, int(round(mask.shape[0] / 4.0)))
    small = cv2.resize(mask.astype(np.float32), (small_w, small_h), interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(small, (0, 0), sigmaX=1.35, sigmaY=1.35)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    tangent_x = -gy
    tangent_y = gx
    magnitude = np.sqrt(tangent_x * tangent_x + tangent_y * tangent_y)
    valid = magnitude > 1.0e-6
    tangent_x[valid] /= magnitude[valid]
    tangent_y[valid] /= magnitude[valid]
    upward = tangent_y < 0
    tangent_x[upward] *= -1
    tangent_y[upward] *= -1
    return tangent_x, tangent_y


def sample_orientation(
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
    return (
        tangent_x[scaled_y, scaled_x].astype(np.float64),
        tangent_y[scaled_y, scaled_x].astype(np.float64),
    )


def pair_features(
    xs: np.ndarray,
    ys: np.ndarray,
    crowns_xy: np.ndarray,
    shape_hw: tuple[int, int],
    *,
    feature_set: str,
    tangent_xy: tuple[np.ndarray, np.ndarray] | None,
    lateral_flags: np.ndarray,
) -> np.ndarray:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unsupported owner feature set: {feature_set}")
    crowns = np.asarray(crowns_xy, dtype=np.float64)
    if crowns.shape != (5, 2) or np.any(~np.isfinite(crowns)):
        raise ValueError("Learned five-seedling ownership requires five finite crown points.")

    height, width = shape_hw
    x = xs.astype(np.float64)[:, None] / max(float(width - 1), 1.0)
    y = ys.astype(np.float64)[:, None] / max(float(height - 1), 1.0)
    crown_x = crowns[:, 0][None, :] / max(float(width - 1), 1.0)
    crown_y = crowns[:, 1][None, :] / max(float(height - 1), 1.0)
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
        distance = np.sqrt(dx * dx + dy * dy) + 1.0e-6
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


@dataclass(slots=True)
class SharedOwnerRanker:
    l2: float = 1.0e-3
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    weights_: np.ndarray | None = None
    optimization_: dict[str, Any] = field(default_factory=dict)

    def fit(self, features: np.ndarray, owners: np.ndarray) -> "SharedOwnerRanker":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(owners, dtype=np.int64)
        if x.ndim != 3 or x.shape[1] != 5 or y.shape != (x.shape[0],):
            raise ValueError("Owner training expects features shaped (pixels, 5, features).")
        flat = x.reshape(-1, x.shape[-1])
        mean = np.mean(flat, axis=0)
        scale = np.std(flat, axis=0)
        scale[scale < 1.0e-8] = 1.0
        standardized = (x - mean[None, None, :]) / scale[None, None, :]

        def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
            scores = np.tensordot(standardized, weights, axes=([2], [0]))
            scores -= np.max(scores, axis=1, keepdims=True)
            probabilities = np.exp(scores)
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
            row = np.arange(y.size)
            loss = -float(np.mean(np.log(probabilities[row, y] + 1.0e-12)))
            loss += 0.5 * float(self.l2) * float(np.dot(weights, weights))
            probabilities[row, y] -= 1.0
            gradient = np.mean(
                np.sum(probabilities[:, :, None] * standardized, axis=1),
                axis=0,
            )
            gradient += float(self.l2) * weights
            return loss, gradient

        result = None
        try:
            from scipy.optimize import minimize

            result = minimize(
                objective,
                np.zeros(x.shape[-1], dtype=np.float64),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 180, "ftol": 1.0e-10, "gtol": 1.0e-7},
            )
        except (ImportError, AttributeError):
            result = None
        self.mean_ = mean
        self.scale_ = scale
        if result is not None:
            self.weights_ = np.asarray(result.x, dtype=np.float64)
            self.optimization_ = {
                "success": bool(result.success),
                "status": int(result.status),
                "iterations": int(result.nit),
                "loss": float(result.fun),
                "message": str(result.message),
                "optimizer": "scipy_lbfgsb",
            }
        else:
            self.weights_, self.optimization_ = _bfgs_minimize(
                objective,
                np.zeros(x.shape[-1], dtype=np.float64),
                max_iterations=180,
            )
        return self

    def decision_scores(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.weights_ is None:
            raise RuntimeError("Owner ranker has not been fitted.")
        values = np.asarray(features, dtype=np.float64)
        standardized = (
            values - self.mean_[None, None, :]
        ) / self.scale_[None, None, :]
        return np.tensordot(standardized, self.weights_, axes=([2], [0]))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.decision_scores(features), axis=1).astype(np.uint8) + np.uint8(1)

    def to_payload(self) -> dict[str, object]:
        if self.mean_ is None or self.scale_ is None or self.weights_ is None:
            raise RuntimeError("Owner ranker has not been fitted.")
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
        model = cls(l2=float(payload.get("l2", 1.0e-3)))
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


def predict_owner_map(
    root_union: np.ndarray,
    lateral_mask: np.ndarray,
    crowns_xy: np.ndarray,
    ranker: SharedOwnerRanker,
    *,
    feature_set: str = "crown_coordinates_orientation",
    chunk_size: int = 180_000,
    prior_owner_map: np.ndarray | None = None,
    temporal_score_bonus: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    root = (np.asarray(root_union, dtype=np.uint8) > 0)
    lateral = (np.asarray(lateral_mask, dtype=np.uint8) > 0)
    if root.shape != lateral.shape:
        raise ValueError("Root and lateral masks must have the same shape.")
    prior = (
        np.asarray(prior_owner_map, dtype=np.uint8)
        if isinstance(prior_owner_map, np.ndarray) and prior_owner_map.shape == root.shape
        else np.zeros(root.shape, dtype=np.uint8)
    )
    ys, xs = np.where(root)
    owner = np.zeros(root.shape, dtype=np.uint8)
    margin = np.zeros(root.shape, dtype=np.float32)
    if xs.size <= 0:
        return owner, margin
    orientation = (
        orientation_channels(root.astype(np.uint8))
        if feature_set == "crown_coordinates_orientation"
        else None
    )
    batch_size = max(1, int(chunk_size))
    score_bonus = float(max(0.0, temporal_score_bonus))
    for start in range(0, xs.size, batch_size):
        stop = min(xs.size, start + batch_size)
        chunk_y = ys[start:stop]
        chunk_x = xs[start:stop]
        tangent = (
            None
            if orientation is None
            else sample_orientation(orientation, chunk_y, chunk_x, root.shape)
        )
        features = pair_features(
            chunk_x,
            chunk_y,
            crowns_xy,
            root.shape,
            feature_set=feature_set,
            tangent_xy=tangent,
            lateral_flags=lateral[chunk_y, chunk_x],
        )
        scores = ranker.decision_scores(features)
        if score_bonus > 0.0:
            prior_slots = prior[chunk_y, chunk_x].astype(np.int64) - 1
            valid_prior = (prior_slots >= 0) & (prior_slots < 5)
            if bool(np.any(valid_prior)):
                rows = np.flatnonzero(valid_prior)
                scores[rows, prior_slots[valid_prior]] += score_bonus
        top_two = np.partition(scores, -2, axis=1)
        owner[chunk_y, chunk_x] = np.argmax(scores, axis=1).astype(np.uint8) + np.uint8(1)
        margin[chunk_y, chunk_x] = (top_two[:, -1] - top_two[:, -2]).astype(np.float32)
    return owner, margin
