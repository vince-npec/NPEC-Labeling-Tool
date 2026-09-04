from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.models import DatasetImageItem, LabelClass  # noqa: E402
from resources.project_io import save_project  # noqa: E402


PIXEL_SIZE_MM = 1.0 / 148.0
CLASS_SPECS = [
    (1, "Root", "#ffd24d"),
]


class _ShapeOnly:
    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.shape = shape


def _find_files(root: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    hits: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        stem = path.stem
        hits.setdefault(stem, path)
    return hits


def _decode_root_mask(mask_path: Path) -> np.ndarray:
    gray = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return (gray < 128).astype(np.uint8)


def import_rootpainter_demo(images_root: Path, masks_root: Path, output_project: Path) -> dict[str, object]:
    images_root = Path(images_root).expanduser().resolve()
    masks_root = Path(masks_root).expanduser().resolve()
    output_project = Path(output_project).expanduser().resolve()
    output_project.parent.mkdir(parents=True, exist_ok=True)

    image_map = _find_files(images_root, (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    mask_map = _find_files(masks_root, (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    if not image_map:
        raise FileNotFoundError(f"No source images found under {images_root}")
    if not mask_map:
        raise FileNotFoundError(f"No mask images found under {masks_root}")

    shared_ids = sorted(set(image_map) & set(mask_map))
    if not shared_ids:
        raise RuntimeError("No matching RootPainter image/mask stems were found.")

    classes = [LabelClass(class_id, name, color) for class_id, name, color in CLASS_SPECS]
    dataset_items: list[DatasetImageItem] = []
    annotations: dict[str, dict[int, np.ndarray]] = {}
    rows: list[dict[str, object]] = []

    for stem in shared_ids:
        image_path = image_map[stem]
        mask_path = mask_map[stem]
        with Image.open(image_path) as pil:
            width, height = pil.size
        root_mask = _decode_root_mask(mask_path)
        if root_mask.shape != (height, width):
            raise ValueError(
                f"Mask shape mismatch for {stem}: image {(height, width)} vs mask {tuple(root_mask.shape)}"
            )
        dataset_items.append(
            DatasetImageItem(
                uid=stem,
                name=image_path.name,
                path=image_path,
                image=_ShapeOnly((height, width, 3)),  # type: ignore[arg-type]
            )
        )
        annotations[stem] = {1: root_mask}
        rows.append(
            {
                "uid": stem,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "width": int(width),
                "height": int(height),
                "root_pixels": int(np.count_nonzero(root_mask)),
            }
        )

    save_project(
        output_project,
        dataset_items,
        classes,
        annotations,
        predictions={},
        current_index=0,
        active_class_id=1,
        patch_target_class_id=1,
        next_class_id=2,
        model_path=None,
        pyphenotyper_pixel_size_mm=PIXEL_SIZE_MM,
        analytics_pixel_size_mm=PIXEL_SIZE_MM,
    )

    summary = {
        "project_path": str(output_project),
        "images_root": str(images_root),
        "masks_root": str(masks_root),
        "pixel_size_mm": float(PIXEL_SIZE_MM),
        "image_count": len(shared_ids),
        "rows": rows,
    }
    output_project.with_suffix(".import_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Import the RootPainter minirhizotron demo into an NPEC .oclp project.")
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--masks-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = import_rootpainter_demo(args.images_root, args.masks_root, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
