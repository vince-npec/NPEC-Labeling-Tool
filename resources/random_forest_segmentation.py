from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class RandomForestSegmentationConfig:
    sigma: float = 1.2
    n_trees: int = 80
    max_depth: int = 18
    min_sample_count: int = 3
    max_train_samples: int = 150000
    refinement_iters: int = 2
    enhancer_mode: str = "classic"  # classic | tda_root_v1
    tda_mode: bool = False
    foreground_class_id: int = 1
    probability_threshold: float = 0.5
    return_probabilities: bool = False


def _enhance_for_separation(image_rgb: np.ndarray, sigma: float, enhancer_mode: str = "classic") -> tuple[np.ndarray, np.ndarray]:
    sigma = max(0.0, float(sigma))
    base = np.asarray(image_rgb, dtype=np.uint8)
    mode = str(enhancer_mode or "classic").strip().lower()
    if sigma > 0.0 and mode != "tda_root_v1":
        enhanced = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma, sigmaY=sigma)  # Classic mode.
    else:
        enhanced = base
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    if mode == "tda_root_v1":
        # Root-focused enhancer: CLAHE + light denoising + DoG contrast boost.
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        smooth = cv2.GaussianBlur(gray_eq, (0, 0), sigmaX=max(0.4, sigma * 0.6 + 0.4), sigmaY=max(0.4, sigma * 0.6 + 0.4))
        wide = cv2.GaussianBlur(gray_eq, (0, 0), sigmaX=max(1.0, sigma * 2.2 + 1.0), sigmaY=max(1.0, sigma * 2.2 + 1.0))
        dog = cv2.subtract(smooth, wide)
        dog_u8 = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        enhanced = cv2.cvtColor(dog_u8, cv2.COLOR_GRAY2RGB)
        gray = dog_u8
    else:
        gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    # Local edge contrast to improve seed-class separation.
    edge = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge = np.abs(edge)
    edge = cv2.normalize(edge, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return enhanced, edge


def _feature_stack(image_rgb: np.ndarray, sigma: float, enhancer_mode: str = "classic") -> np.ndarray:
    enhanced, edge = _enhance_for_separation(image_rgb, sigma=sigma, enhancer_mode=enhancer_mode)
    h, w = enhanced.shape[:2]
    rgb = enhanced.astype(np.float32) / 255.0
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_mag = cv2.normalize(grad_mag, None, 0.0, 1.0, cv2.NORM_MINMAX)
    yy, xx = np.indices((h, w), dtype=np.float32)
    xx = xx / max(1.0, float(w - 1))
    yy = yy / max(1.0, float(h - 1))
    feats_list: list[np.ndarray] = [
        rgb[:, :, 0],
        rgb[:, :, 1],
        rgb[:, :, 2],
        gray,
        grad_mag,
        edge.astype(np.float32),
        xx,
        yy,
    ]
    if str(enhancer_mode or "classic").strip().lower() == "tda_root_v1":
        gray_u8 = np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray_u8).astype(np.float32) / 255.0
        top_hat = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        top_hat = top_hat.astype(np.float32) / 255.0
        blur_small = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(0.4, sigma + 0.4), sigmaY=max(0.4, sigma + 0.4))
        blur_wide = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1.2, sigma * 2.4 + 0.8), sigmaY=max(1.2, sigma * 2.4 + 0.8))
        dog = cv2.normalize(blur_small - blur_wide, None, 0.0, 1.0, cv2.NORM_MINMAX)
        feats_list.extend([gray_eq, top_hat, dog.astype(np.float32)])
    feats = np.dstack(feats_list)
    return feats.astype(np.float32)


def _balanced_seed_indices(seed_map: np.ndarray, max_train_samples: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(seed_map, dtype=np.int32).reshape(-1)
    seeded_idx = np.flatnonzero(labels > 0)
    if seeded_idx.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int32)

    # Per-class cap keeps training balanced even when one class has much denser strokes.
    classes = sorted(int(v) for v in np.unique(labels[seeded_idx]).tolist() if int(v) > 0)
    per_class_cap = max(1, int(max_train_samples // max(1, len(classes))))

    selected_parts: list[np.ndarray] = []
    rng = np.random.default_rng(12345)
    for cls in classes:
        cls_idx = seeded_idx[labels[seeded_idx] == cls]
        if cls_idx.size > per_class_cap:
            cls_idx = rng.choice(cls_idx, size=per_class_cap, replace=False)
        selected_parts.append(np.asarray(cls_idx, dtype=np.int64))

    selected = np.concatenate(selected_parts, axis=0) if selected_parts else seeded_idx
    rng.shuffle(selected)
    return selected, labels[selected]


def _class_separability_score(features: np.ndarray, labels: np.ndarray) -> float:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        return 0.0
    classes = [int(v) for v in np.unique(y).tolist() if int(v) > 0]
    if len(classes) < 2:
        return 0.0

    overall = x.mean(axis=0)
    between = 0.0
    within = 0.0
    for cls in classes:
        cls_x = x[y == cls]
        if cls_x.size == 0:
            continue
        mu = cls_x.mean(axis=0)
        between += float(cls_x.shape[0]) * float(np.mean((mu - overall) ** 2))
        within += float(np.mean((cls_x - mu) ** 2))
    return float(between / (within + 1e-8))


def optimize_sigma_from_seeds(
    image_rgb: np.ndarray,
    seed_index_mask: np.ndarray,
    sigma_values: list[float] | None = None,
    enhancer_mode: str = "classic",
) -> tuple[float, list[tuple[float, float]]]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Sigma optimization expects an RGB image.")
    if seed_index_mask.shape[:2] != image_rgb.shape[:2]:
        raise ValueError("Seed mask shape does not match image shape.")

    if sigma_values is None:
        sigma_values = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.6, 3.2, 4.0]
    candidates = [float(max(0.0, s)) for s in sigma_values]
    if not candidates:
        raise ValueError("No sigma candidates provided.")

    train_idx, train_labels = _balanced_seed_indices(seed_index_mask, max_train_samples=180000)
    if train_idx.size == 0:
        raise ValueError("No seeded pixels found. Add seed strokes before optimizing sigma.")
    if len(set(int(v) for v in train_labels.tolist())) < 2:
        raise ValueError("Sigma optimization needs seed strokes for at least two classes.")

    scores: list[tuple[float, float]] = []
    for sigma in candidates:
        feats = _feature_stack(image_rgb, sigma=sigma, enhancer_mode=enhancer_mode)
        feats_flat = feats.reshape((-1, feats.shape[-1]))
        sample = feats_flat[train_idx]
        score = _class_separability_score(sample, train_labels)
        scores.append((sigma, float(score)))

    scores.sort(key=lambda pair: pair[1], reverse=True)
    best_sigma = float(scores[0][0])
    return best_sigma, scores


def _train_rtrees(features_flat: np.ndarray, train_idx: np.ndarray, train_labels: np.ndarray, cfg: RandomForestSegmentationConfig):
    if train_idx.size == 0:
        raise ValueError("No seeded pixels were provided. Add brush strokes before running Random Forest segmentation.")

    X = np.asarray(features_flat[train_idx], dtype=np.float32)
    y = np.asarray(train_labels, dtype=np.int32).reshape(-1, 1)

    model = cv2.ml.RTrees_create()
    model.setMaxDepth(max(2, int(cfg.max_depth)))
    model.setMinSampleCount(max(1, int(cfg.min_sample_count)))
    model.setRegressionAccuracy(0.0)
    model.setUseSurrogates(False)
    model.setMaxCategories(128)
    model.setCalculateVarImportance(False)
    model.setActiveVarCount(0)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, max(10, int(cfg.n_trees)), 0.0))
    ok = model.train(X, cv2.ml.ROW_SAMPLE, y)
    if not ok:
        raise RuntimeError("Random Forest training failed.")
    return model


def _predict_chunked(model, features_flat: np.ndarray, total_pixels: int, chunk_size: int = 250000) -> np.ndarray:
    pred = np.zeros((total_pixels,), dtype=np.int32)
    for start in range(0, total_pixels, chunk_size):
        end = min(total_pixels, start + chunk_size)
        _, out = model.predict(np.asarray(features_flat[start:end], dtype=np.float32))
        pred[start:end] = out.reshape(-1).astype(np.int32)
    return pred


def _predict_with_votes_chunked(
    model,
    features_flat: np.ndarray,
    total_pixels: int,
    class_ids: list[int],
    chunk_size: int = 250000,
) -> tuple[np.ndarray, dict[int, np.ndarray] | None]:
    pred = _predict_chunked(model, features_flat, total_pixels=total_pixels, chunk_size=chunk_size)
    if not hasattr(model, "getVotes"):
        return pred, None
    if not class_ids:
        return pred, None

    votes_parts: list[np.ndarray] = []
    for start in range(0, total_pixels, chunk_size):
        end = min(total_pixels, start + chunk_size)
        chunk = np.asarray(features_flat[start:end], dtype=np.float32)
        try:
            votes = model.getVotes(chunk, 0)
        except Exception:
            return pred, None
        votes_arr = np.asarray(votes, dtype=np.float32)
        if votes_arr.ndim != 2:
            return pred, None
        rows = int(end - start)
        if votes_arr.shape[0] != rows and votes_arr.shape[1] == rows:
            votes_arr = votes_arr.T
        if votes_arr.shape[0] != rows:
            return pred, None

        if votes_arr.shape[1] == len(class_ids) + 1:
            votes_arr = votes_arr[:, 1:]
        elif votes_arr.shape[1] > len(class_ids):
            votes_arr = votes_arr[:, : len(class_ids)]
        elif votes_arr.shape[1] < len(class_ids):
            return pred, None
        votes_parts.append(votes_arr)

    if not votes_parts:
        return pred, None
    all_votes = np.concatenate(votes_parts, axis=0)
    if all_votes.shape[0] != total_pixels:
        return pred, None
    vote_sum = all_votes.sum(axis=1, keepdims=True)
    vote_sum[vote_sum <= 0.0] = 1.0
    probs = all_votes / vote_sum
    prob_maps: dict[int, np.ndarray] = {}
    for idx, class_id in enumerate(class_ids):
        prob_maps[int(class_id)] = np.asarray(probs[:, idx], dtype=np.float32)
    return pred, prob_maps


def _refine_mask(index_mask: np.ndarray, class_ids: list[int], iters: int) -> np.ndarray:
    if iters <= 0:
        return index_mask
    mask = np.asarray(index_mask, dtype=np.uint8).copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(int(iters)):
        refined = np.zeros_like(mask, dtype=np.uint8)
        for class_id in class_ids:
            if class_id <= 0:
                continue
            binary = (mask == class_id).astype(np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            refined[binary > 0] = np.uint8(class_id)
        # Keep background where no class remains.
        mask = refined
    return mask


def run_random_forest_segmentation(
    image_rgb: np.ndarray,
    seed_index_mask: np.ndarray,
    class_ids: list[int],
    config: RandomForestSegmentationConfig | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    cfg = config or RandomForestSegmentationConfig()
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Random Forest segmentation expects an RGB image.")
    if seed_index_mask.shape[:2] != image_rgb.shape[:2]:
        raise ValueError("Seed mask shape does not match image shape.")

    enhancer_mode = str(cfg.enhancer_mode or "classic").strip().lower()
    features = _feature_stack(image_rgb, sigma=cfg.sigma, enhancer_mode=enhancer_mode)
    h, w = features.shape[:2]
    features_flat = features.reshape((-1, features.shape[-1]))
    train_idx, train_labels = _balanced_seed_indices(seed_index_mask, max_train_samples=max(1000, int(cfg.max_train_samples)))
    if train_idx.size == 0:
        raise ValueError("No seeded pixels found. Add strokes for at least one class.")

    trained_class_ids = [int(v) for v in sorted(set(int(x) for x in train_labels.tolist())) if int(v) > 0]
    model = _train_rtrees(features_flat, train_idx, train_labels, cfg)
    pred_flat, vote_prob_by_class = _predict_with_votes_chunked(
        model,
        features_flat,
        total_pixels=h * w,
        class_ids=trained_class_ids,
    )

    fg_class_id = int(cfg.foreground_class_id)
    if fg_class_id not in trained_class_ids and trained_class_ids:
        fg_class_id = int(trained_class_ids[0])

    fg_prob_flat: np.ndarray | None = None
    probability_source = "none"
    if vote_prob_by_class is not None and fg_class_id in vote_prob_by_class:
        fg_prob_flat = np.asarray(vote_prob_by_class[fg_class_id], dtype=np.float32)
        probability_source = "votes"
    else:
        fg_prob_flat = (pred_flat == fg_class_id).astype(np.float32)
        probability_source = "hard_label_fallback"

    pred = pred_flat.reshape((h, w)).astype(np.uint8)
    if bool(cfg.tda_mode):
        thr = float(np.clip(cfg.probability_threshold, 0.0, 1.0))
        fg_prob_map = np.asarray(fg_prob_flat, dtype=np.float32).reshape((h, w))
        pred = np.where(fg_prob_map >= thr, np.uint8(fg_class_id), np.uint8(0))
        pred = _refine_mask(pred, [int(fg_class_id)], iters=int(cfg.refinement_iters))
    else:
        pred = _refine_mask(pred, [int(c) for c in class_ids], iters=int(cfg.refinement_iters))

    details: dict[str, object] = {
        "backend": "random_forest_opencv",
        "sigma": float(cfg.sigma),
        "n_trees": int(cfg.n_trees),
        "max_depth": int(cfg.max_depth),
        "min_sample_count": int(cfg.min_sample_count),
        "max_train_samples": int(cfg.max_train_samples),
        "refinement_iters": int(cfg.refinement_iters),
        "enhancer_mode": enhancer_mode,
        "tda_mode": bool(cfg.tda_mode),
        "foreground_class_id": int(fg_class_id),
        "probability_threshold": float(np.clip(cfg.probability_threshold, 0.0, 1.0)),
        "probability_source": probability_source,
        "seeded_pixels": int(train_idx.size),
        "classes_seeded": [int(v) for v in sorted(set(int(x) for x in train_labels.tolist()))],
        "feature_count": int(features.shape[-1]),
        "foreground_probability_mean": float(np.mean(fg_prob_flat)) if fg_prob_flat is not None else None,
    }
    if bool(cfg.return_probabilities) and fg_prob_flat is not None:
        details["_foreground_probability"] = np.asarray(fg_prob_flat, dtype=np.float32).reshape((h, w))
    return pred, details
