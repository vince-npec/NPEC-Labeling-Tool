from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.models import DatasetImageItem, LabelClass  # noqa: E402
from resources.five_seedling_ownership import (  # noqa: E402
    discover_five_seedling_corpus,
    grouped_split,
    load_semantic_masks,
)
from resources.project_io import save_project  # noqa: E402


LABEL_ROOT = REPO_ROOT / "data" / "yang_ground_truth"
ORIGINAL_ROOT = LABEL_ROOT / "images"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "rgb_inoculated_training"
IMAGES_ROOT = OUTPUT_ROOT / "images"
FULL_PROJECT_PATH = OUTPUT_ROOT / "rgb_inoculated_full.oclp"
TRAIN_PROJECT_PATH = OUTPUT_ROOT / "rgb_inoculated_train.oclp"
HOLDOUT_PROJECT_PATH = OUTPUT_ROOT / "rgb_inoculated_holdout.oclp"
MANIFEST_PATH = OUTPUT_ROOT / "rgb_inoculated_manifest.json"
README_PATH = OUTPUT_ROOT / "README.txt"

CLASS_SPECS: list[tuple[int, str, str]] = [
    (1, "Seed", "#ff5a8a"),
    (2, "Shoot", "#35d07f"),
    (3, "Root", "#ffb347"),
    (4, "Lateral", "#33a1ff"),
]

FRAME_RE = re.compile(r"^(?P<base>.+?)_\d+_(?P<label>[^_]+)$")
PLATE_RE = re.compile(r"^(?P<plate>\d+-\d+)[-_](?P<rest>.+)$")


def _canonicalize_layers(
    *,
    seed: np.ndarray,
    shoot: np.ndarray,
    root: np.ndarray,
    lateral: np.ndarray,
    include_background: bool,
) -> dict[str, np.ndarray]:
    seed = (seed > 0).astype(np.uint8)
    shoot = ((shoot > 0) & (~seed.astype(bool))).astype(np.uint8)
    lateral = ((lateral > 0) & (~seed.astype(bool)) & (~shoot.astype(bool))).astype(np.uint8)
    root = (
        (root > 0)
        & (~seed.astype(bool))
        & (~shoot.astype(bool))
        & (~lateral.astype(bool))
    ).astype(np.uint8)
    foreground = (seed > 0) | (shoot > 0) | (root > 0) | (lateral > 0)
    background = (~foreground).astype(np.uint8) if include_background else np.zeros_like(seed, dtype=np.uint8)
    return {
        "Seed": seed,
        "Shoot": shoot,
        "Root": root,
        "Lateral": lateral,
        "Background": background,
    }


def _extract_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        if img.mode == "P":
            indexed = np.array(img, dtype=np.uint16)
            transparency = img.info.get("transparency")
            if isinstance(transparency, int):
                return (indexed != transparency).astype(np.uint8)
        rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3]
    if np.unique(alpha).size > 1 or int(alpha.min()) < 250:
        mask = alpha > 0
        if mask.any():
            return mask.astype(np.uint8)
    rgb = rgba[..., :3]
    return np.any(rgb < 245, axis=2).astype(np.uint8)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def _iter_original_candidates(base_name: str) -> list[Path]:
    candidates: list[str] = []
    for name in (base_name, f"{base_name}.png", f"{base_name}.png.png"):
        if name not in candidates:
            candidates.append(name)
    match = PLATE_RE.match(base_name)
    if match:
        variants = [
            f"{match.group('plate')}_{match.group('rest')}",
            f"{match.group('plate')}-{match.group('rest')}",
        ]
        for variant in variants:
            for name in (variant, f"{variant}.png", f"{variant}.png.png"):
                if name not in candidates:
                    candidates.append(name)
    return [ORIGINAL_ROOT / name for name in candidates]


def _resolve_original(base_name: str) -> Path | None:
    for candidate in _iter_original_candidates(base_name):
        if candidate.exists():
            return candidate
    return None


def _copy_image(source_path: Path, folder_name: str) -> Path:
    dest_name = f"rgb_inoculated_{folder_name}_{source_path.name}"
    dest_path = IMAGES_ROOT / dest_name
    if not dest_path.exists():
        shutil.copy2(source_path, dest_path)
    return dest_path


def _subset_payload(
    dataset_items: list[DatasetImageItem],
    annotations: dict[str, dict[int, np.ndarray]],
    series_by_uid: dict[str, str],
    wanted_uids: set[str],
) -> tuple[list[DatasetImageItem], dict[str, dict[int, np.ndarray]], dict[str, str]]:
    subset_items = [item for item in dataset_items if item.uid in wanted_uids]
    subset_annotations = {
        item.uid: {int(class_id): np.asarray(mask, dtype=np.uint8) for class_id, mask in annotations[item.uid].items()}
        for item in subset_items
    }
    subset_series = {item.uid: str(series_by_uid.get(item.uid, "")) for item in subset_items}
    return subset_items, subset_annotations, subset_series


def _save_subset_project(
    project_path: Path,
    dataset_items: list[DatasetImageItem],
    classes: list[LabelClass],
    annotations: dict[str, dict[int, np.ndarray]],
    series_by_uid: dict[str, str],
) -> None:
    series_order = sorted(dict.fromkeys(series_by_uid.values()))
    save_project(
        project_path=project_path,
        dataset_items=dataset_items,
        classes=classes,
        annotations=annotations,
        predictions={},
        current_index=0,
        active_class_id=3,
        patch_target_class_id=3,
        next_class_id=max(cls.class_id for cls in classes) + 1,
        model_path=None,
        series_by_uid=series_by_uid,
        series_order=series_order,
        store_relative_paths=True,
    )


def _split_holdout_uids(dataset_items: list[DatasetImageItem]) -> tuple[set[str], set[str]]:
    ordered = sorted(dataset_items, key=lambda item: item.name.lower())
    holdout: set[str] = set()
    train: set[str] = set()
    for idx, item in enumerate(ordered):
        if idx % 5 == 0:
            holdout.add(item.uid)
        else:
            train.add(item.uid)
    if not holdout and ordered:
        holdout.add(ordered[0].uid)
        train.discard(ordered[0].uid)
    train.update({item.uid for item in ordered if item.uid not in holdout})
    return train, holdout


def build_projects() -> dict[str, object]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGES_ROOT.mkdir(parents=True, exist_ok=True)

    classes = [LabelClass(class_id, name, color) for class_id, name, color in CLASS_SPECS]
    class_id_by_name = {name: class_id for class_id, name, _ in CLASS_SPECS}

    dataset_items: list[DatasetImageItem] = []
    annotations: dict[str, dict[int, np.ndarray]] = {}
    series_by_uid: dict[str, str] = {}
    manifest_rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    discovered_samples = discover_five_seedling_corpus(LABEL_ROOT)
    samples = [sample for sample in discovered_samples if re.fullmatch(r"\d+-\d+", sample.sample_id)]
    skipped.extend(
        {"folder": sample.sample_id, "reason": "ambiguous_sample_id"}
        for sample in discovered_samples
        if sample not in samples
    )
    train_samples, holdout_samples = grouped_split(samples, holdout_fraction=0.20, salt="rgb-inoculated-v2-holdout")
    train_sample_ids = {sample.sample_id for sample in train_samples}
    holdout_sample_ids = {sample.sample_id for sample in holdout_samples}
    for sample in samples:
        original_path = sample.image_path
        image = _load_rgb(original_path)
        masks = load_semantic_masks(sample)
        per_name = _canonicalize_layers(
            seed=masks["seed"],
            shoot=masks["shoot"],
            root=masks["root"],
            lateral=masks["lateral"],
            include_background=False,
        )
        if not per_name["Root"].any() or not per_name["Shoot"].any():
            skipped.append({"folder": sample.sample_id, "reason": "missing_root_or_shoot"})
            continue

        uid = f"rgb_inoculated_{sample.sample_id}"
        dest_path = _copy_image(original_path, sample.sample_id)
        dataset_items.append(DatasetImageItem(uid=uid, name=dest_path.name, path=dest_path, image=image))
        annotations[uid] = {
            class_id_by_name[name]: np.asarray(mask, dtype=np.uint8)
            for name, mask in per_name.items()
            if name in class_id_by_name and np.count_nonzero(mask) > 0
        }
        series_by_uid[uid] = str(sample.plate_group)
        manifest_rows.append(
            {
                "uid": uid,
                "folder": sample.sample_id,
                "image": original_path.name,
                "seed_pixels": int(np.count_nonzero(per_name["Seed"])),
                "shoot_pixels": int(np.count_nonzero(per_name["Shoot"])),
                "root_pixels": int(np.count_nonzero(per_name["Root"])),
                "lateral_pixels": int(np.count_nonzero(per_name["Lateral"])),
                "has_seed": bool(np.count_nonzero(per_name["Seed"])),
                "has_lateral": bool(np.count_nonzero(per_name["Lateral"])),
                "plate_group": str(sample.plate_group),
                "split": "holdout" if sample.sample_id in holdout_sample_ids else "train",
            }
        )

    _save_subset_project(FULL_PROJECT_PATH, dataset_items, classes, annotations, series_by_uid)
    train_uids = {f"rgb_inoculated_{sample_id}" for sample_id in train_sample_ids}
    holdout_uids = {f"rgb_inoculated_{sample_id}" for sample_id in holdout_sample_ids}
    train_items, train_annotations, train_series = _subset_payload(dataset_items, annotations, series_by_uid, train_uids)
    holdout_items, holdout_annotations, holdout_series = _subset_payload(dataset_items, annotations, series_by_uid, holdout_uids)
    _save_subset_project(TRAIN_PROJECT_PATH, train_items, classes, train_annotations, train_series)
    _save_subset_project(HOLDOUT_PROJECT_PATH, holdout_items, classes, holdout_annotations, holdout_series)

    manifest = {
        "label_root": str(LABEL_ROOT),
        "original_root": str(ORIGINAL_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "projects": {
            "full": str(FULL_PROJECT_PATH),
            "train": str(TRAIN_PROJECT_PATH),
            "holdout": str(HOLDOUT_PROJECT_PATH),
        },
        "total_folders_seen": len(samples),
        "usable_images": len(dataset_items),
        "train_count": len(train_items),
        "holdout_count": len(holdout_items),
        "with_seed_count": int(sum(1 for row in manifest_rows if row["has_seed"])),
        "with_lateral_count": int(sum(1 for row in manifest_rows if row["has_lateral"])),
        "skipped": skipped,
        "items": manifest_rows,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    README_PATH.write_text(
        "\n".join(
            [
                "RGB/Inoculated Lucifer-style training project",
                "",
                f"Full project: {FULL_PROJECT_PATH.name}",
                f"Train project: {TRAIN_PROJECT_PATH.name}",
                f"Holdout project: {HOLDOUT_PROJECT_PATH.name}",
                f"Usable images: {len(dataset_items)}",
                f"Train / holdout: {len(train_items)} / {len(holdout_items)}",
                "",
                "Classes:",
                "1 Seed",
                "2 Shoot",
                "3 Root",
                "4 Lateral",
                "",
                "Notes:",
                "- Built from the Yang Song ground-truth corpus for the DroughtFighters project,",
                "  Plant Microbe Interaction group, Utrecht University.",
                "- Background is implicit class 0; visible organs use classes 1-4.",
                "- Hand-labelled bacterial occlusions remain unknown gaps, not foreground labels.",
                "- Holdout is plate-group exclusive to prevent related samples leaking across splits.",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    manifest = build_projects()
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
