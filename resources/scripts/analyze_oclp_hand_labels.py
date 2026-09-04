from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.mask_total_root_analysis import (  # noqa: E402
    build_mask_total_dataframe,
    build_mask_total_pmi_dataframe,
    write_mask_total_outputs,
)
from resources.pmi_export import build_pmi_style_rows  # noqa: E402
from resources.pyphenotyper_adapter import build_lucifer_green_shoot_mask  # noqa: E402
from resources.root_growth_video import (  # noqa: E402
    LazyRootGrowthVideoConfig,
    generate_lazy_mask_total_root_growth_video,
)


ROOT_CLASS_IDS = (1, 3)
SHOOT_CLASS_IDS = (2,)
MASK_COLORS = {
    1: (255, 132, 42),
    2: (255, 44, 190),
    3: (112, 158, 238),
}
FILENAME_RE = re.compile(
    r"^(?P<petri>.+?)_(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_"
    r"(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})_(?P<frame>\d+)$"
)


def _safe_member(member: str) -> str:
    normalized = str(member).replace("\\", "/").strip("/")
    if not normalized or normalized.startswith("/") or ".." in Path(normalized).parts:
        raise ValueError(f"Unsafe project archive member: {member!r}")
    return normalized


def _read_json_member(archive: zipfile.ZipFile, member: str) -> dict[str, object]:
    payload = json.loads(archive.read(_safe_member(member)).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {member}")
    return payload


def _read_npy_member(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    array = np.load(io.BytesIO(archive.read(_safe_member(member))), allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mask in {member}, got {array.shape}")
    return (np.asarray(array) > 0).astype(np.uint8)


def _filename_metadata(name: str, fallback_index: int) -> tuple[str, str, int]:
    match = FILENAME_RE.match(Path(name).stem)
    if match is None:
        return Path(name).stem, "", int(fallback_index)
    values = match.groupdict()
    timestamp = (
        f"{values['year']}-{values['month']}-{values['day']} "
        f"{values['hour']}:{values['minute']}:{values['second']}"
    )
    return str(values["petri"]), timestamp, int(values["frame"])


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf") if bold else Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    )
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _overlay_image(image: Image.Image, mask: np.ndarray, *, display_dilation: int = 3) -> Image.Image:
    base = image.convert("RGB")
    index_image = Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L")
    if display_dilation > 1:
        index_image = index_image.filter(ImageFilter.MaxFilter(display_dilation))
    index = np.asarray(index_image, dtype=np.uint8)
    base_array = np.asarray(base, dtype=np.uint8).copy()
    for class_id, color in MASK_COLORS.items():
        selected = index == int(class_id)
        if not np.any(selected):
            continue
        color_array = np.asarray(color, dtype=np.float32)
        base_array[selected] = np.clip(
            (base_array[selected].astype(np.float32) * 0.30) + (color_array * 0.70),
            0,
            255,
        ).astype(np.uint8)
    return Image.fromarray(base_array, mode="RGB")


def _write_contact_sheet(entries: list[dict[str, object]], output_path: Path) -> Path:
    columns = 4
    tile_width = 480
    tile_image_height = 475
    label_height = 34
    title_height = 92
    rows = max(1, int(np.ceil(len(entries) / columns)))
    canvas = Image.new(
        "RGB",
        (columns * tile_width, title_height + rows * (tile_image_height + label_height)),
        (244, 246, 248),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(25, bold=True)
    body_font = _font(17)
    label_font = _font(15, bold=True)
    draw.text((18, 13), "Vince hand labels + improved root-anchored RGB shoot masks", fill=(26, 31, 38), font=title_font)
    legend_x = 20
    legend_y = 54
    for class_id, label in ((1, "Primary/root class 1"), (3, "Lateral/root class 3"), (2, "RGB shoot class 2")):
        draw.rectangle((legend_x, legend_y, legend_x + 22, legend_y + 16), fill=MASK_COLORS[class_id])
        draw.text((legend_x + 29, legend_y - 2), label, fill=(38, 43, 50), font=body_font)
        legend_x += 250

    for index, entry in enumerate(entries):
        row = index // columns
        col = index % columns
        x = col * tile_width
        y = title_height + row * (tile_image_height + label_height)
        image = Image.open(Path(str(entry["image_path"]))).convert("RGB")
        mask = np.asarray(Image.open(Path(str(entry["analysis_mask_path"]))).convert("L"), dtype=np.uint8)
        overlay = _overlay_image(image, mask)
        overlay.thumbnail((tile_width, tile_image_height), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, tile_image_height), (18, 20, 24))
        tile.paste(overlay, ((tile_width - overlay.width) // 2, (tile_image_height - overlay.height) // 2))
        canvas.paste(tile, (x, y))
        label = f"{entry['petri']} | shoots {int(entry['shoot_pixels']):,} px"
        draw.rectangle((x, y + tile_image_height, x + tile_width, y + tile_image_height + label_height), fill=(18, 20, 24))
        draw.text((x + 9, y + tile_image_height + 7), label, fill=(247, 249, 252), font=label_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92, optimize=True)
    return output_path


def _compare_projects(current_project: Path, baseline_project: Path, output_path: Path) -> Path:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(current_project, "r") as current_archive, zipfile.ZipFile(baseline_project, "r") as baseline_archive:
        current = _read_json_member(current_archive, "project.json")
        baseline = _read_json_member(baseline_archive, "project.json")
        current_items = {str(item["uid"]): item for item in current.get("dataset", []) if isinstance(item, dict)}
        baseline_items = {str(item["uid"]): item for item in baseline.get("dataset", []) if isinstance(item, dict)}
        for uid in sorted(set(current_items) & set(baseline_items)):
            name = str(current_items[uid].get("name", uid))
            petri, _timestamp, _frame = _filename_metadata(name, 0)
            for class_id in (1, 2, 3):
                member = f"masks/{uid}/class_{class_id}.npy"
                current_mask = _read_npy_member(current_archive, member) > 0
                baseline_mask = _read_npy_member(baseline_archive, member) > 0
                added = int(np.count_nonzero(current_mask & (~baseline_mask)))
                removed = int(np.count_nonzero(baseline_mask & (~current_mask)))
                rows.append(
                    {
                        "uid": uid,
                        "image_name": name,
                        "petri": petri,
                        "class_id": class_id,
                        "baseline_pixels": int(np.count_nonzero(baseline_mask)),
                        "current_pixels": int(np.count_nonzero(current_mask)),
                        "added_pixels": added,
                        "removed_pixels": removed,
                        "changed_positions": int(added + removed),
                    }
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def analyze_project(
    project_path: Path,
    output_dir: Path,
    *,
    pixel_size_mm: float,
    comparison_project: Path | None = None,
    fps: int = 1,
) -> dict[str, object]:
    project_path = project_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = output_dir / "source_images"
    manual_mask_root = output_dir / "manual_root_lateral_masks"
    analysis_mask_root = output_dir / "analysis_masks_improved_rgb_shoots"
    for path in (source_root, manual_mask_root, analysis_mask_root):
        path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    shoot_rows: list[dict[str, object]] = []
    class_totals = {1: 0, 2: 0, 3: 0}
    manual_shoot_total = 0
    root_overlap_total = 0

    with zipfile.ZipFile(project_path, "r") as archive:
        project = _read_json_member(archive, "project.json")
        dataset = project.get("dataset", [])
        if not isinstance(dataset, list) or not dataset:
            raise ValueError("The .oclp project does not contain a dataset")
        classes = project.get("classes", [])
        for item_index, item in enumerate(dataset):
            if not isinstance(item, dict):
                continue
            uid = str(item.get("uid", ""))
            name = str(item.get("name", ""))
            image_member = str(item.get("embedded_image_member", ""))
            if not uid or not name or not image_member:
                raise ValueError(f"Dataset item {item_index} is missing embedded image metadata")
            image_bytes = archive.read(_safe_member(image_member))
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_array = np.asarray(image, dtype=np.uint8)
            shape_hw = tuple(int(value) for value in image_array.shape[:2])
            petri, timestamp, frame_index = _filename_metadata(name, item_index)
            safe_petri = petri.replace("/", "_").replace("\\", "_")

            class_masks: dict[int, np.ndarray] = {}
            for class_id in (1, 2, 3):
                class_masks[class_id] = _read_npy_member(archive, f"masks/{uid}/class_{class_id}.npy")
                if class_masks[class_id].shape != shape_hw:
                    raise ValueError(
                        f"Mask shape {class_masks[class_id].shape} does not match image {name} shape {shape_hw}"
                    )
            manual_shoot_px = int(np.count_nonzero(class_masks[2]))
            manual_shoot_total += manual_shoot_px
            root_overlap = int(np.count_nonzero((class_masks[1] > 0) & (class_masks[3] > 0)))
            root_overlap_total += root_overlap

            manual_mask = np.zeros(shape_hw, dtype=np.uint8)
            manual_mask[class_masks[1] > 0] = 1
            manual_mask[class_masks[3] > 0] = 3
            measured_root_mask = np.isin(manual_mask, ROOT_CLASS_IDS).astype(np.uint8)
            shoot_candidate, shoot_meta = build_lucifer_green_shoot_mask(
                image_array,
                shape_hw,
                root_mask=measured_root_mask,
            )
            shoot_mask = (np.asarray(shoot_candidate, dtype=np.uint8) > 0) & (measured_root_mask == 0)
            analysis_mask = manual_mask.copy()
            analysis_mask[shoot_mask] = 2
            shoot_pixels = int(np.count_nonzero(shoot_mask))
            candidate_pixels = int(np.count_nonzero(shoot_candidate))
            shoot_fraction = float(shoot_pixels / max(1, shape_hw[0] * shape_hw[1]))
            qc_status = "ok"
            qc_reason = ""
            if shoot_pixels == 0:
                qc_status = "fail"
                qc_reason = "no green shoot pixels detected"
            elif shoot_fraction < 0.004:
                qc_status = "review"
                qc_reason = "unusually small shoot fraction"
            elif shoot_fraction > 0.065:
                qc_status = "review"
                qc_reason = "unusually large shoot fraction"

            image_path = source_root / safe_petri / name
            manual_mask_path = manual_mask_root / safe_petri / f"{Path(name).stem}_manual_mask.png"
            analysis_mask_path = analysis_mask_root / safe_petri / f"{Path(name).stem}_mask.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            manual_mask_path.parent.mkdir(parents=True, exist_ok=True)
            analysis_mask_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(image_bytes)
            Image.fromarray(manual_mask, mode="L").save(manual_mask_path, optimize=True)
            Image.fromarray(analysis_mask, mode="L").save(analysis_mask_path, optimize=True)

            records.append(
                {
                    "image_name": name,
                    "image_path": str(image_path),
                    "output_mask": str(analysis_mask_path),
                    "meta": {
                        "series": "manual_hand_labels_vince",
                        "petri": petri,
                        "timestamp": timestamp,
                        "frame_index": frame_index,
                        "relative_folder": safe_petri,
                    },
                    "details": {"pixel_size_mm": float(pixel_size_mm)},
                }
            )
            class_counts = {class_id: int(np.count_nonzero(analysis_mask == class_id)) for class_id in (1, 2, 3)}
            for class_id, count in class_counts.items():
                class_totals[class_id] += count
            entries.append(
                {
                    "uid": uid,
                    "image_name": name,
                    "petri": petri,
                    "image_path": str(image_path),
                    "manual_mask_path": str(manual_mask_path),
                    "analysis_mask_path": str(analysis_mask_path),
                    "shoot_pixels": shoot_pixels,
                    "shoot_meta": dict(shoot_meta),
                }
            )
            manifest_rows.append(
                {
                    "uid": uid,
                    "image_name": name,
                    "petri": petri,
                    "timestamp": timestamp,
                    "frame_index": frame_index,
                    "height": shape_hw[0],
                    "width": shape_hw[1],
                    "class_1_root_pixels": class_counts[1],
                    "class_2_shoot_pixels": class_counts[2],
                    "class_3_lateral_root_pixels": class_counts[3],
                    "manual_class_2_pixels": manual_shoot_px,
                    "manual_root_lateral_overlap_pixels": root_overlap,
                    "source_image": str(image_path),
                    "manual_root_lateral_mask": str(manual_mask_path),
                    "analysis_mask": str(analysis_mask_path),
                }
            )
            shoot_rows.append(
                {
                    "uid": uid,
                    "image_name": name,
                    "petri": petri,
                    "manual_shoot_px": manual_shoot_px,
                    "rgb_green_candidate_px": candidate_pixels,
                    "root_overlap_removed_px": int(candidate_pixels - shoot_pixels),
                    "rgb_green_baked_shoot_px": shoot_pixels,
                    "shoot_fraction_of_image": shoot_fraction,
                    "rgb_green_components": int(shoot_meta.get("components_kept", 0) or 0),
                    "root_supported_components": int(shoot_meta.get("root_context_supported_components", 0) or 0),
                    "olive_root_supported_components": int(shoot_meta.get("olive_root_supported_components", 0) or 0),
                    "green_chain_supported_components": int(shoot_meta.get("green_chain_supported_components", 0) or 0),
                    "large_root_supported_components": int(shoot_meta.get("large_root_supported_components", 0) or 0),
                    "root_crown_gate_rejections": int(shoot_meta.get("root_crown_gate_rejections", 0) or 0),
                    "root_context_enabled": bool(shoot_meta.get("root_context_enabled", False)),
                    "shoot_filter": str(shoot_meta.get("filter", "")),
                    "qc_status": qc_status,
                    "qc_reason": qc_reason,
                }
            )

    if not records:
        raise ValueError("No valid embedded images were found")

    mask_df = build_mask_total_dataframe(
        records,
        class_ids=ROOT_CLASS_IDS,
        shoot_class_ids=SHOOT_CLASS_IDS,
        fallback_pixel_size_mm=float(pixel_size_mm),
        min_component_area=1,
        shoot_rgb_rescue=False,
    )
    if mask_df.empty:
        raise RuntimeError("The mask measurement table is empty")

    entry_by_name = {str(entry["image_name"]): entry for entry in entries}
    shoot_by_name = {str(row["image_name"]): row for row in shoot_rows}
    mask_df["manual_label_project"] = str(project_path)
    mask_df["root_measurement_source"] = "student_hand_label_masks_class_1_and_3"
    mask_df["shoot_measurement_source"] = "improved_root_anchored_rgb_green_mask_class_2"
    mask_df["shoot_mask_baked_from_rgb_green"] = True
    mask_df["manual_shoot_masks_present"] = bool(manual_shoot_total > 0)
    mask_df["full_resolution_source_image"] = mask_df["SourceFile"].astype(str)
    mask_df["manual_root_lateral_mask"] = mask_df["SourceFile"].map(
        lambda value: str(entry_by_name[Path(str(value)).name]["manual_mask_path"])
    )
    mask_df["shoot_detection_qc"] = mask_df["SourceFile"].map(
        lambda value: str(shoot_by_name[Path(str(value)).name]["qc_status"])
    )
    mask_df["shoot_rgb_rescue_candidate_px"] = mask_df["SourceFile"].map(
        lambda value: int(shoot_by_name[Path(str(value)).name]["rgb_green_candidate_px"])
    )
    mask_df["shoot_rgb_rescue_green_only_px"] = mask_df["shoot_area_px"].astype(int)
    mask_df["shoot_rgb_rescue_components"] = mask_df["SourceFile"].map(
        lambda value: int(shoot_by_name[Path(str(value)).name]["rgb_green_components"])
    )

    detail_csv, summary_csv, compat_csv, measurement_metadata = write_mask_total_outputs(
        mask_df,
        output_dir,
        class_ids=ROOT_CLASS_IDS,
        shoot_class_ids=SHOOT_CLASS_IDS,
        root_metric_mode="total",
    )
    manifest_csv = output_dir / "npec_hand_label_project_manifest.csv"
    shoot_csv = output_dir / "npec_improved_rgb_green_shoot_summary.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    pd.DataFrame(shoot_rows).to_csv(shoot_csv, index=False)

    pmi_source = build_mask_total_pmi_dataframe(mask_df, shoot_class_ids=SHOOT_CLASS_IDS)
    pmi_df = build_pmi_style_rows(
        pmi_source,
        include_fluorescence_placeholders=True,
        derive_mask_metrics=True,
    )
    pmi_csv = output_dir / "npec_root_traits_pmi_style.csv"
    pmi_df.to_csv(pmi_csv, index=False)

    overlay_video = output_dir / "npec_manual_labels_root_shoot_overlay_video.mp4"
    overlay_sample = output_dir / "npec_manual_labels_root_shoot_overlay_video_sample_frame.png"
    overlay_summary = output_dir / "npec_manual_labels_root_shoot_overlay_video_summary.csv"
    generate_lazy_mask_total_root_growth_video(
        mask_df,
        overlay_video,
        summary_csv_path=overlay_summary,
        sample_frame_path=overlay_sample,
        config=LazyRootGrowthVideoConfig(
            fps=max(1, int(fps)),
            width=1920,
            height=1080,
            mask_only=False,
            mask_display_dilation_px=1,
            root_metric_mode="total",
            root_class_ids=ROOT_CLASS_IDS,
            shoot_class_ids=SHOOT_CLASS_IDS,
            show_rgb_shoot_rescue=False,
        ),
    )
    mask_only_video = output_dir / "npec_manual_labels_mask_only_video.mp4"
    mask_only_sample = output_dir / "npec_manual_labels_mask_only_video_sample_frame.png"
    mask_only_summary = output_dir / "npec_manual_labels_mask_only_video_summary.csv"
    generate_lazy_mask_total_root_growth_video(
        mask_df,
        mask_only_video,
        summary_csv_path=mask_only_summary,
        sample_frame_path=mask_only_sample,
        config=LazyRootGrowthVideoConfig(
            fps=max(1, int(fps)),
            width=1920,
            height=1080,
            mask_only=True,
            mask_display_dilation_px=1,
            root_metric_mode="total",
            root_class_ids=ROOT_CLASS_IDS,
            shoot_class_ids=SHOOT_CLASS_IDS,
            show_rgb_shoot_rescue=False,
        ),
    )
    contact_sheet = _write_contact_sheet(entries, output_dir / "npec_manual_labels_overlay_contact_sheet.jpg")

    comparison_csv: Path | None = None
    if comparison_project is not None and comparison_project.expanduser().exists():
        comparison_csv = _compare_projects(
            project_path,
            comparison_project.expanduser().resolve(),
            output_dir / "npec_vince_project_mask_changes.csv",
        )

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            (
                "# NPEC Vince hand-label analysis",
                "",
                f"Source project: `{project_path}`",
                f"Measurement calibration: `{pixel_size_mm:.6f} mm/px`.",
                "",
                "Roots are measured from the student's class 1 (primary/root) and class 3 (lateral root) masks.",
                "Shoots are measured from strict RGB green pixels in the upper plate region, with large rosettes",
                "accepted only when supported by the manual root masks. Neutral grey bacterial material is rejected.",
                "The generated class 2 masks are stored in `analysis_masks_improved_rgb_shoots`.",
                "",
                "The source project contains no manual class 2 shoot pixels, so shoot precision and recall cannot be",
                "calculated from this package. Use `npec_manual_labels_overlay_contact_sheet.jpg` for visual QA.",
                "The crown-direction guard rejects green/olive candidate components whose center lies below the",
                "local top of the root trace, reducing bacterial-colony leakage along inoculated roots.",
                "",
                "Compared with the earlier project, Vince changed only B59-1: class 1 gained 62 pixels, while",
                "class 3 gained 304 pixels and removed 7,035 pixels (6,731 net pixels removed).",
                "",
                "Each embedded image is a different petri dish, so this package contains one analyzed frame per dish.",
                "Raw pixel counts are retained alongside calibrated length and area measurements.",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    project_pixel_sizes = {
        "analytics_pixel_size_mm": project.get("analytics_pixel_size_mm"),
        "pyphenotyper_pixel_size_mm": project.get("pyphenotyper_pixel_size_mm"),
    }
    metadata = {
        "source_project": str(project_path),
        "comparison_project": str(comparison_project.expanduser().resolve()) if comparison_project is not None else None,
        "output_dir": str(output_dir),
        "frames_measured": int(len(mask_df)),
        "petri_dishes": int(mask_df["PetriDish"].nunique()),
        "classes": classes,
        "root_class_ids": list(ROOT_CLASS_IDS),
        "shoot_class_ids": list(SHOOT_CLASS_IDS),
        "root_measurement_source": "student hand labels class 1 Root + class 3 Lateral Root",
        "shoot_measurement_source": "improved root-anchored RGB green Arabidopsis detector baked into class 2",
        "shoot_detector_revision": "large_rosette_local_crown_gate_v3",
        "shoot_detector_safety": "strict RGB chroma plus local crown direction; neutral grey and below-crown bacteria rejected",
        "pixel_size_mm_used_for_main_mm_outputs": float(pixel_size_mm),
        "project_metadata_pixel_sizes": project_pixel_sizes,
        "manual_class_2_shoot_pixels_in_project": int(manual_shoot_total),
        "manual_root_lateral_overlap_pixels": int(root_overlap_total),
        "class_pixel_totals_after_shoot_bake": {str(key): int(value) for key, value in class_totals.items()},
        "rgb_green_shoot_pixels_baked_total": int(class_totals[2]),
        "shoot_qc_counts": pd.DataFrame(shoot_rows)["qc_status"].value_counts().to_dict(),
        "outputs": {
            "detail_csv": str(detail_csv),
            "summary_csv": str(summary_csv),
            "timelapse_csv": str(output_dir / "npec_total_root_and_shoot_timelapse.csv"),
            "legacy_total_root_length_timelapse_csv": str(compat_csv) if compat_csv is not None else None,
            "measurement_metadata_json": str(measurement_metadata),
            "manifest_csv": str(manifest_csv),
            "shoot_detection_csv": str(shoot_csv),
            "pmi_style_csv": str(pmi_csv),
            "overlay_video": str(overlay_video),
            "overlay_video_sample_frame": str(overlay_sample),
            "mask_only_video": str(mask_only_video),
            "mask_only_video_sample_frame": str(mask_only_sample),
            "overlay_contact_sheet": str(contact_sheet),
            "comparison_csv": str(comparison_csv) if comparison_csv is not None else None,
            "all_metrics_workbook": str(output_dir / "npec_total_root_and_shoot_all_metrics.xlsx"),
            "readme": str(readme_path),
        },
    }
    metadata_path = output_dir / "npec_manual_hand_label_analysis_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze embedded hand labels from an NPEC .oclp project")
    parser.add_argument("project", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixel-size-mm", type=float, default=0.026638)
    parser.add_argument("--comparison-project", type=Path)
    parser.add_argument("--fps", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = analyze_project(
        args.project,
        args.output_dir,
        pixel_size_mm=float(args.pixel_size_mm),
        comparison_project=args.comparison_project,
        fps=max(1, int(args.fps)),
    )
    print(json.dumps(metadata, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
