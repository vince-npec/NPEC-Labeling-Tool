from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.general_root_starter.unified_model import train_unified_general_root_starter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the unified shared-backbone General Root Starter model from a normalized corpus.")
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Output .pt/.pth checkpoint path")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--base-filters", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto", help="auto | gpu | cpu")
    parser.add_argument("--repeat-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--family-loss-weight", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--curriculum", type=str, default="auto", help="auto | none")
    parser.add_argument("--positive-crop-bias", type=float, default=0.95)
    parser.add_argument("--tasks", type=str, default="", help="Comma-separated task list. Defaults to all tasks.")
    parser.add_argument("--root-only", action="store_true", help="Train only root tasks (root_binary, primary_root, lateral_root).")
    args = parser.parse_args()

    tasks: list[str] | None = None
    if args.root_only:
        tasks = ["root_binary", "primary_root", "lateral_root"]
    elif str(args.tasks or "").strip():
        tasks = [token.strip() for token in str(args.tasks).split(",") if token.strip()]

    summary = train_unified_general_root_starter(
        args.corpus_dir,
        args.output,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        patch_size=int(args.patch_size),
        base_filters=int(args.base_filters),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        device_mode=str(args.device),
        repeat_factor=int(args.repeat_factor),
        random_seed=int(args.seed),
        weight_decay=float(args.weight_decay),
        family_loss_weight=float(args.family_loss_weight),
        early_stopping_patience=int(args.early_stopping_patience),
        early_stopping_min_delta=float(args.early_stopping_min_delta),
        curriculum_mode=str(args.curriculum),
        positive_crop_bias=float(args.positive_crop_bias),
        tasks=tasks,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
