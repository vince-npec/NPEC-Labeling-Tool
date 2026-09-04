from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.general_root_starter.unified_model import evaluate_unified_general_root_starter, load_corpus_manifest  # noqa: E402
from resources.models import DatasetImageItem  # noqa: E402
from resources.pyphenotyper_adapter import PyPhenotyperConfig, run_pyphenotyper_dataset  # noqa: E402
from resources.scripts.benchmark_general_root_starter import _built_in_general_starter_experts  # noqa: E402


def _load_items(samples: list[dict[str, object]]) -> list[DatasetImageItem]:
    items: list[DatasetImageItem] = []
    for sample in samples:
        path = Path(str(sample["image_path"])).expanduser().resolve()
        from PIL import Image

        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        items.append(
            DatasetImageItem(
                uid=str(sample.get("sample_id") or path.stem),
                name=str(sample.get("name") or path.name),
                path=path,
                image=image,
            )
        )
    return items


def _mean(values: list[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    return float(np.mean(nums)) if nums else None


def _iou(pred: np.ndarray, truth: np.ndarray) -> float | None:
    pred_b = np.asarray(pred, dtype=np.uint8) > 0
    truth_b = np.asarray(truth, dtype=np.uint8) > 0
    union = float(np.count_nonzero(np.logical_or(pred_b, truth_b)))
    if union <= 0:
        return None
    inter = float(np.count_nonzero(np.logical_and(pred_b, truth_b)))
    return inter / union


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the unified General Root Starter checkpoint against the current routed-expert baseline.")
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_corpus_manifest(args.corpus_dir)
    split = str(args.split or "test")
    samples = [sample for sample in list(manifest.get("samples") or []) if str(sample.get("split")) == split]
    if not samples:
        raise SystemExit(f"No samples found for split={split!r}")

    unified_rows = evaluate_unified_general_root_starter(args.checkpoint, samples, device_mode="auto")

    items = _load_items(samples)
    config = PyPhenotyperConfig(
        pipeline_dir=REPO_ROOT / "resources" / "npec_pyphenotyper",
        root_model_path=REPO_ROOT / "resources" / "builtin_models" / "hades_lucifer" / "model_root_14.h5",
        shoot_model_path=REPO_ROOT / "resources" / "builtin_models" / "hades_lucifer" / "model_shoot_10.h5",
        refinement_steps=2,
        root_class_id=1,
        shoot_class_id=2,
        expert_variants=_built_in_general_starter_experts(),
        router_mode="general_root_starter",
    )
    routed_predictions, routed_metadata = run_pyphenotyper_dataset(items, config)
    routed_rows: list[dict[str, object]] = []
    for sample, item in zip(samples, items):
        target_masks = np.load(Path(str(sample["mask_path"])))
        available = sample.get("tasks_available") if isinstance(sample.get("tasks_available"), dict) else {}
        pred = routed_predictions.get(item.uid)
        if pred is None:
            routed_rows.append({"sample_id": item.uid, "root_binary_iou": None, "shoot_iou": None, "routing_variant": None})
            continue
        routed_rows.append(
            {
                "sample_id": item.uid,
                "routing_variant": str(routed_metadata.get(item.uid, {}).get("routing_variant") or "unknown"),
                "root_binary_iou": _iou(pred == np.uint8(config.root_class_id), np.asarray(target_masks["root_binary"], dtype=np.uint8))
                if bool(available.get("root_binary", False))
                else None,
                "shoot_iou": _iou(pred == np.uint8(config.shoot_class_id), np.asarray(target_masks["shoot"], dtype=np.uint8))
                if bool(available.get("shoot", False))
                else None,
            }
        )

    summary = {
        "corpus_dir": str(Path(args.corpus_dir).expanduser().resolve()),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "split": split,
        "sample_count": len(samples),
        "unified": {
            "root_mean_iou": _mean([row.get("root_binary_iou") for row in unified_rows]),
            "shoot_mean_iou": _mean([row.get("shoot_iou") for row in unified_rows]),
            "primary_mean_iou": _mean([row.get("primary_root_iou") for row in unified_rows]),
            "lateral_mean_iou": _mean([row.get("lateral_root_iou") for row in unified_rows]),
            "seed_mean_iou": _mean([row.get("seed_crown_iou") for row in unified_rows]),
        },
        "routed_experts": {
            "root_mean_iou": _mean([row.get("root_binary_iou") for row in routed_rows]),
            "shoot_mean_iou": _mean([row.get("shoot_iou") for row in routed_rows]),
            "routing_counts": {
                key: sum(1 for row in routed_rows if row.get("routing_variant") == key)
                for key in sorted({str(row.get("routing_variant") or "unknown") for row in routed_rows})
            },
        },
        "unified_cases": unified_rows,
        "routed_cases": routed_rows,
    }
    output_path = args.output or (Path(args.corpus_dir).expanduser().resolve() / "benchmark_unified_vs_routed.json")
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
