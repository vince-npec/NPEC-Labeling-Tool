from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Sequence


SCRIPT_PATH = Path(__file__).resolve()
ROOT_PATH = SCRIPT_PATH.parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from resources.plant_health import (  # noqa: E402
    PLANT_HEALTH_BINARY_CLASS_NAMES,
    build_health_eval_transform,
    build_health_train_transform,
    choose_torch_device,
    create_plant_health_model,
    find_plantvillage_imagefolder_root,
    plantvillage_binary_label,
)


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def _discover_items(dataset_root: Path) -> tuple[list[tuple[Path, int, str]], dict[str, int]]:
    items: list[tuple[Path, int, str]] = []
    counts = {"healthy": 0, "stressed": 0}
    for class_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        binary_index = int(plantvillage_binary_label(class_dir.name))
        binary_name = PLANT_HEALTH_BINARY_CLASS_NAMES[binary_index]
        for image_path in sorted(path for path in class_dir.iterdir() if path.is_file() and _is_image(path)):
            items.append((image_path, binary_index, class_dir.name))
            counts[binary_name] += 1
    if not items:
        raise RuntimeError(f"No images found under {dataset_root}")
    return items, counts


def _split_items(
    items: Sequence[tuple[Path, int, str]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[tuple[Path, int, str]], list[tuple[Path, int, str]]]:
    grouped: dict[int, list[tuple[Path, int, str]]] = {0: [], 1: []}
    for row in items:
        grouped[int(row[1])].append(row)
    rng = random.Random(int(seed))
    train_rows: list[tuple[Path, int, str]] = []
    val_rows: list[tuple[Path, int, str]] = []
    for label, rows in grouped.items():
        rows = list(rows)
        rng.shuffle(rows)
        val_count = max(1, int(round(len(rows) * float(val_fraction))))
        val_rows.extend(rows[:val_count])
        train_rows.extend(rows[val_count:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def _build_dataset(rows: Sequence[tuple[Path, int, str]], transform):
    from PIL import Image
    from torch.utils.data import Dataset

    class _LeafDataset(Dataset):
        def __init__(self, samples, transform_):
            self.samples = list(samples)
            self.transform = transform_

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            path, label, class_name = self.samples[index]
            image = Image.open(path).convert("RGB")
            tensor = self.transform(image)
            return tensor, int(label)

    return _LeafDataset(rows, transform)


def _evaluate(model, loader, device):
    import torch

    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    true_positive = 0
    false_positive = 0
    false_negative = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            total += int(labels.numel())
            correct += int((pred == labels).sum().item())
            total_loss += float(loss.item()) * int(labels.numel())
            true_positive += int(((pred == 1) & (labels == 1)).sum().item())
            false_positive += int(((pred == 1) & (labels == 0)).sum().item())
            false_negative += int(((pred == 0) & (labels == 1)).sum().item())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = (2.0 * precision * recall) / max(1e-8, precision + recall)
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "f1_stressed": f1,
        "precision_stressed": precision,
        "recall_stressed": recall,
        "samples": total,
    }


def main() -> int:
    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Train a lightweight PlantVillage-derived plant-health model.")
    parser.add_argument("--dataset-root", default="resources/output/public_datasets/plantvillage/PlantVillage-Dataset")
    parser.add_argument("--variant", default="segmented", choices=["segmented", "color"])
    parser.add_argument("--output-dir", default="resources/output/plant_health_training")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    raw_root = Path(args.dataset_root).expanduser().resolve()
    imagefolder_base = find_plantvillage_imagefolder_root(raw_root)
    dataset_root = imagefolder_base / args.variant if (imagefolder_base / args.variant).is_dir() else imagefolder_base
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items, counts = _discover_items(dataset_root)
    train_rows, val_rows = _split_items(items, val_fraction=float(args.val_fraction), seed=int(args.seed))

    train_ds = _build_dataset(train_rows, build_health_train_transform(int(args.image_size)))
    val_ds = _build_dataset(val_rows, build_health_eval_transform(int(args.image_size)))
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    device = choose_torch_device(None if str(args.device).strip().lower() == "auto" else args.device)
    model = create_plant_health_model(num_classes=2, pretrained=True).to(device)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(args.learning_rate), weight_decay=1e-4)

    history: list[dict[str, float | int]] = []
    best_acc = -1.0
    best_path = output_dir / "plantvillage_leaf_health_binary_mobilenet_v1.pt"
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(labels.numel())
            seen += int(labels.numel())

        val_metrics = _evaluate(model, val_loader, device)
        row = {
            "epoch": int(epoch),
            "train_loss": float(running_loss / max(1, seen)),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_f1_stressed": float(val_metrics["f1_stressed"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{int(args.epochs)} "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} "
            f"val_f1_stressed={row['val_f1_stressed']:.4f}",
            flush=True,
        )
        if float(val_metrics["accuracy"]) >= best_acc:
            best_acc = float(val_metrics["accuracy"])
            torch.save(
                {
                    "model_arch": "mobilenet_v3_small",
                    "num_classes": 2,
                    "class_names": list(PLANT_HEALTH_BINARY_CLASS_NAMES),
                    "state_dict": model.state_dict(),
                    "variant": str(args.variant),
                    "device": str(device),
                    "train_count": len(train_rows),
                    "val_count": len(val_rows),
                },
                best_path,
            )

    summary = {
        "dataset_root": str(dataset_root),
        "variant": str(args.variant),
        "counts": counts,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "epochs": int(args.epochs),
        "device": str(device),
        "best_val_accuracy": float(best_acc),
        "history": history,
        "model_path": str(best_path),
    }
    summary_path = output_dir / "plantvillage_leaf_health_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(best_path))
    print(str(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
