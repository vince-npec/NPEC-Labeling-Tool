from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.scripts.analyze_jason_mixed_dataset import (  # noqa: E402
    HADES_PIXEL_SIZE_MM,
    MODEL_ROUTE_HADES,
    build_model_config,
    run_segmentation,
)
from resources.stabilization import (  # noqa: E402
    StabilizationConfig,
    next_stabilization_reference,
    stabilize_against_reference,
)


SOURCE_RE = re.compile(
    r"^(?P<run>\d+)_(?P<frame>\d+)_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_"
    r"(?P<plate>exp\d+_\d+)_ROOT\d+_Original\.png$",
    re.IGNORECASE,
)
AUDIT_FILENAME = "npec_hades_inoculated_dataset_manifest.csv"
STABILIZATION_FILENAME = "npec_stabilization_transforms.csv"
SEGMENTATION_FILENAME = "npec_segmentation_manifest.csv"


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as pil:
        return np.asarray(pil.convert("RGB"), dtype=np.uint8)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path, optimize=True)


def _source_rows(input_dir: Path, selected_plates: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    paths = [
        path
        for plate_dir in input_dir.iterdir()
        if plate_dir.is_dir() and re.fullmatch(r"exp\d+_\d+", plate_dir.name, re.IGNORECASE)
        for path in plate_dir.glob("*.png")
    ]
    for path in sorted(paths, key=lambda value: str(value).lower()):
        if path.name.startswith("._"):
            continue
        match = SOURCE_RE.match(path.name)
        if match is None:
            continue
        plate = match.group("plate").lower()
        if selected_plates and plate not in selected_plates:
            continue
        timestamp = pd.to_datetime(
            f"{match.group('date')} {match.group('time').replace('-', ':')}",
            errors="raise",
        )
        frame_number = int(match.group("frame"))
        rows.append(
            {
                "SourceFile": str(path.resolve()),
                "OriginalSourceFile": str(path.resolve()),
                "SourceName": path.name,
                "DatasetPrefix": match.group("run"),
                "SampleIndex": frame_number,
                "Cohort": "hades_heavily_inoculated_timeseries",
                "AcquisitionSystem": "Hades",
                "ModelRoute": MODEL_ROUTE_HADES,
                "PixelSizeMm": float(HADES_PIXEL_SIZE_MM),
                "ScaleStatus": "validated_hades_plate_calibration",
                "Series": plate,
                "PetriDish": plate,
                "Timestamp": timestamp.isoformat(sep=" "),
                "FrameIndex": frame_number - 1,
                "RawFrameIndex": frame_number,
                "IsTimeSeries": True,
                "ExpectedPlants": 5,
                "GroupingConfidence": 1.0,
                "GroupingEvidence": "folder and filename agree on exp79 plate ID",
                "SeriesKind": "time_series",
                "SeriesConfidence": "definite",
                "TimeAxis": "acquisition_timestamp",
                "NominalLayout": "horizontal_five_seedling_lanes",
                "OwnershipMode": "five_lane_temporal_occlusion_aware",
                "RelativeFolder": plate,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No Hades expNN_NN time-series PNGs found in {input_dir}")
    duplicates = frame.duplicated(["Series", "FrameIndex"], keep=False)
    if bool(duplicates.any()):
        names = frame.loc[duplicates, "SourceName"].tolist()
        raise RuntimeError(f"Duplicate plate/frame identities: {names}")
    frame.sort_values(
        ["Series", "Timestamp", "FrameIndex", "SourceName"],
        inplace=True,
        kind="mergesort",
    )
    frame.reset_index(drop=True, inplace=True)
    return frame


def _limit_per_plate(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if int(limit) <= 0:
        return frame
    limited = (
        frame.groupby("Series", sort=False, group_keys=False)
        .head(int(limit))
        .copy()
    )
    limited.reset_index(drop=True, inplace=True)
    return limited


def _stabilize(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    enabled: bool,
    overwrite: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = StabilizationConfig(
        reference_mode="previous",
        target_mode="plant_top",
        plant_top_fraction=0.04,
        plant_bottom_fraction=0.70,
        max_shift_fraction=0.08,
        max_rotation_deg=2.0,
        max_scale_deviation=0.025,
        enable_biological_motion_veto=True,
    )
    updated_rows: list[dict[str, object]] = []
    transform_rows: list[dict[str, object]] = []
    for plate, group in frame.groupby("Series", sort=True):
        ordered = group.sort_values(["Timestamp", "FrameIndex"], kind="mergesort")
        reference: np.ndarray | None = None
        for position, (_, source_row) in enumerate(ordered.iterrows(), start=1):
            row = source_row.to_dict()
            source_path = Path(str(row["OriginalSourceFile"]))
            source_image = _load_rgb(source_path)
            output_path = output_dir / "stabilized_input" / str(plate) / source_path.name
            if not enabled:
                stabilized = source_image
                status = "disabled"
                method = "identity"
                score = 1.0
                shift_x = 0.0
                shift_y = 0.0
                rotation = 0.0
                accepted = False
                matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            elif reference is None:
                stabilized = source_image
                status = "reference"
                method = "identity"
                score = 1.0
                shift_x = 0.0
                shift_y = 0.0
                rotation = 0.0
                accepted = True
                matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
                reference = stabilized
            else:
                result = stabilize_against_reference(reference, source_image, config)
                stabilized = result.warped_image if result.accepted else source_image
                status = result.status
                method = result.method
                score = float(result.score)
                shift_x = float(result.shift_x)
                shift_y = float(result.shift_y)
                rotation = float(result.rotation_deg)
                accepted = bool(result.accepted)
                matrix = np.asarray(result.warp_matrix, dtype=np.float32)
                reference = next_stabilization_reference(reference, result, config)
            if enabled:
                if overwrite or not output_path.exists():
                    _save_rgb(output_path, stabilized)
                row["SourceFile"] = str(output_path.resolve())
                row["PreviewImagePath"] = str(output_path.resolve())
            else:
                row["SourceFile"] = str(source_path.resolve())
                row["PreviewImagePath"] = str(source_path.resolve())
            row["StabilizationEnabled"] = bool(enabled)
            row["StabilizationAccepted"] = bool(accepted)
            row["StabilizationStatus"] = status
            row["StabilizationMethod"] = method
            row["StabilizationScore"] = score
            row["StabilizationShiftX"] = shift_x
            row["StabilizationShiftY"] = shift_y
            row["StabilizationRotationDeg"] = rotation
            updated_rows.append(row)
            transform_rows.append(
                {
                    "Series": plate,
                    "PetriDish": plate,
                    "FrameIndex": int(row["FrameIndex"]),
                    "Timestamp": row["Timestamp"],
                    "OriginalSourceFile": str(source_path.resolve()),
                    "StabilizedSourceFile": str(row["SourceFile"]),
                    "accepted": bool(accepted),
                    "status": status,
                    "method": method,
                    "score": score,
                    "shift_x_px": shift_x,
                    "shift_y_px": shift_y,
                    "rotation_deg": rotation,
                    "warp_matrix": json.dumps(matrix.tolist(), separators=(",", ":")),
                }
            )
            print(
                f"STABILIZE {plate} {position}/{len(ordered)} "
                f"{status} dx={shift_x:.2f} dy={shift_y:.2f} rot={rotation:.3f}",
                flush=True,
            )
    updated = pd.DataFrame(updated_rows)
    updated.sort_values(["Series", "FrameIndex"], inplace=True, kind="mergesort")
    transforms = pd.DataFrame(transform_rows)
    transforms.sort_values(["Series", "FrameIndex"], inplace=True, kind="mergesort")
    return updated.reset_index(drop=True), transforms.reset_index(drop=True)


def _write_audit(frame: pd.DataFrame, transforms: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / AUDIT_FILENAME, index=False)
    transforms.to_csv(output_dir / STABILIZATION_FILENAME, index=False)
    counts = frame.groupby("Series")["FrameIndex"].nunique().sort_index()
    summary = {
        "schema_version": "npec-hades-inoculated-timeseries/1",
        "images": int(len(frame)),
        "plates": int(frame["Series"].nunique()),
        "frames_per_plate": {str(key): int(value) for key, value in counts.items()},
        "expected_plants_per_plate": 5,
        "model_route": MODEL_ROUTE_HADES,
        "pixel_size_mm": float(HADES_PIXEL_SIZE_MM),
        "stabilization": {
            "enabled": bool(frame["StabilizationEnabled"].iloc[0]),
            "accepted_frames_including_references": int(transforms["accepted"].sum()),
            "total_frames": int(len(transforms)),
            "target": "plant_top_and_upper_roots",
            "reference": "previous_accepted_frame",
        },
    }
    (output_dir / "npec_hades_inoculated_dataset_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _run_command(command: Iterable[str]) -> None:
    env = dict(os.environ)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("PYTHONUNBUFFERED", "1")
    print("COMMAND " + " ".join(str(value) for value in command), flush=True)
    subprocess.run([str(value) for value in command], cwd=REPO_ROOT, env=env, check=True)


def _package(output_dir: Path, *, workers: int, skip_videos: bool) -> None:
    package_command = [
        sys.executable,
        "-m",
        "resources.scripts.package_jason_mixed_dataset",
        "--segmentation-manifest",
        str(output_dir / SEGMENTATION_FILENAME),
        "--output-dir",
        str(output_dir),
        "--workers",
        str(max(1, int(workers))),
    ]
    if skip_videos:
        package_command.append("--skip-videos")
    _run_command(package_command)

    plate_csv = output_dir / "plate_total" / "npec_total_root_and_shoot_from_masks.csv"
    ownership_csv = (
        output_dir
        / "per_seedling"
        / "hades_bw"
        / "npec_root_measurements_master.csv"
    )
    occlusion_command = [
        sys.executable,
        "-m",
        "resources.occlusion_aware_root_analysis",
        "--mask-total-csv",
        str(plate_csv),
        "--ownership-csv",
        str(ownership_csv),
        "--output-dir",
        str(output_dir / "occlusion_aware"),
        "--root-classes",
        "1,3",
        "--plant-count",
        "5",
        "--fps",
        "2",
    ]
    if skip_videos:
        occlusion_command.append("--no-video")
    _run_command(occlusion_command)


def _write_provenance(output_dir: Path, input_dir: Path) -> None:
    segmentation = pd.read_csv(output_dir / SEGMENTATION_FILENAME, low_memory=False)
    ownership_master_path = (
        output_dir
        / "per_seedling"
        / "hades_bw"
        / "npec_root_measurements_master.csv"
    )
    ownership_master = (
        pd.read_csv(ownership_master_path, low_memory=False)
        if ownership_master_path.exists()
        else pd.DataFrame()
    )
    ownership_valid_rows = 0
    expected_lane_tracks = 0
    root_bearing_tracks = 0
    tracks_without_root_evidence: list[dict[str, str]] = []
    if not ownership_master.empty:
        valid = ownership_master.get(
            "ownership_measurement_valid",
            pd.Series(False, index=ownership_master.index),
        ).map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})
        ownership_valid_rows = int(valid.sum())
        track_columns = ["Series", "PetriDish", "plant_id"]
        tracks = ownership_master.groupby(track_columns, dropna=False, sort=True)
        expected_lane_tracks = int(tracks.ngroups)
        root_evidence = ownership_master.get(
            "ownership_track_root_presence_evidence",
            ownership_master.get(
                "ownership_root_presence_evidence",
                pd.Series(False, index=ownership_master.index),
            ),
        ).map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})
        root_by_track = root_evidence.groupby(
            [ownership_master[column] for column in track_columns],
            dropna=False,
        ).any()
        root_bearing_tracks = int(root_by_track.sum())
        for (series, petri, plant_id), has_root in root_by_track.items():
            if bool(has_root):
                continue
            tracks_without_root_evidence.append(
                {
                    "Series": str(series),
                    "PetriDish": str(petri),
                    "plant_id": str(plant_id),
                }
            )
    report_path = (
        output_dir
        / "per_seedling"
        / "hades_bw"
        / "npec_corrected_run_report.json"
    )
    ownership_report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    report = {
        "schema_version": "npec-hades-inoculated-analysis/1",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "images": int(len(segmentation)),
        "plates": int(segmentation["PetriDish"].nunique()),
        "successful_segmentations": int(
            segmentation["SegmentationStatus"].astype(str).str.lower().eq("ok").sum()
        ),
        "expected_plants_per_plate": 5,
        "acquisition_system": "Hades BW",
        "pixel_size_mm": float(HADES_PIXEL_SIZE_MM),
        "root_model": "resources/builtin_models/hades_lucifer/model_root_14.h5",
        "shoot_model": "resources/builtin_models/hades_lucifer/model_shoot_10.h5",
        "ownership_pipeline": ownership_report.get("pipeline_version", ""),
        "ownership_complete_frames": ownership_report.get("ownership_complete_frames", 0),
        "ownership_invalid_rows": ownership_report.get("invalid_rows", 0),
        "ownership_rows": int(len(ownership_master)),
        "ownership_valid_rows": int(ownership_valid_rows),
        "expected_lane_tracks": int(expected_lane_tracks),
        "root_bearing_tracks": int(root_bearing_tracks),
        "tracks_without_root_evidence": tracks_without_root_evidence,
        "interpretation": {
            "plate_total": "ownership-independent segmented root total",
            "per_seedling": (
                "crown-aligned five-lane ownership; use only rows where "
                "ownership_measurement_valid is true"
            ),
            "occlusion_aware": (
                "monotonic carry-forward of previously visible root skeleton; "
                "invalid individual ownership rows remain frozen"
            ),
            "primary_lateral": (
                "topology-derived decomposition of the segmented root union, "
                "not direct semantic ground truth"
            ),
        },
    }
    (output_dir / "npec_hades_inoculated_analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    readme = "\n".join(
        (
            "# NPEC Hades inoculated time-series analysis",
            "",
            f"- Input images: {report['images']}",
            f"- Plates: {report['plates']}",
            "- Expected seedlings per plate: 5",
            f"- Root-bearing tracks: {root_bearing_tracks}/{expected_lane_tracks}",
            f"- Valid plant-frame rows: {ownership_valid_rows}/{len(ownership_master)}",
            f"- Hades scale: {HADES_PIXEL_SIZE_MM:.12f} mm/px",
            "",
            "Dataset-specific QA:",
            *(
                f"- {item['PetriDish']} {item['plant_id']}: expected lane has no persistent root evidence"
                for item in tracks_without_root_evidence
            ),
            "",
            "The plate-total tables are ownership-independent. Per-seedling values are",
            "reported with validity/confidence fields. Use npec_per_seedling_measurements_qa_accepted.csv",
            "for the convenience subset; retain the full detail table for audit and manual review.",
            "",
            "The occlusion-aware folder carries forward root skeleton that was visible",
            "before bacterial obstruction. Individual carry-forward freezes during an",
            "ownership conflict; plate totals remain available for every frame.",
            "",
            "Primary/lateral classes are topology-derived from the segmented root union.",
            "Crossings and complete bacterial occlusion remain intrinsically ambiguous.",
            "",
        )
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stabilize, segment, quantify, and track five-seedling Hades "
            "time series with conservative bacterial-occlusion handling."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--stage",
        choices=("audit", "segment", "package", "all"),
        default="all",
    )
    parser.add_argument("--plates", nargs="*", default=[])
    parser.add_argument(
        "--frames",
        nargs="*",
        type=int,
        default=[],
        help="Optional one-based source frame numbers used for focused validation runs.",
    )
    parser.add_argument("--limit-per-plate", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-stabilize", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    selected_plates = {str(value).strip().lower() for value in args.plates if str(value).strip()}
    source = _source_rows(input_dir, selected_plates)
    selected_frames = {int(value) for value in args.frames if int(value) > 0}
    if selected_frames:
        source = source[source["RawFrameIndex"].astype(int).isin(selected_frames)].copy()
        if source.empty:
            raise RuntimeError(f"No rows matched source frames {sorted(selected_frames)}")
    source = _limit_per_plate(source, int(args.limit_per_plate))
    audit, transforms = _stabilize(
        source,
        output_dir,
        enabled=not bool(args.no_stabilize),
        overwrite=bool(args.overwrite),
    )
    _write_audit(audit, transforms, output_dir)
    print(
        f"AUDIT images={len(audit)} plates={audit['Series'].nunique()} "
        f"manifest={output_dir / AUDIT_FILENAME}",
        flush=True,
    )
    if args.stage in {"segment", "all"}:
        hades_config = build_model_config(MODEL_ROUTE_HADES)
        hades_config.refinement_steps = 2
        hades_config.enable_bbox_tracking = True
        segmented = run_segmentation(
            audit,
            output_dir,
            routes=(MODEL_ROUTE_HADES,),
            overwrite=bool(args.overwrite),
            config_overrides={MODEL_ROUTE_HADES: hades_config},
        )
        ok = int(segmented["SegmentationStatus"].astype(str).str.lower().eq("ok").sum())
        if ok != len(audit):
            raise RuntimeError(f"Only {ok}/{len(audit)} frames segmented successfully")
    if args.stage in {"package", "all"}:
        if not (output_dir / SEGMENTATION_FILENAME).exists():
            raise RuntimeError(f"Missing segmentation manifest: {output_dir / SEGMENTATION_FILENAME}")
        _package(
            output_dir,
            workers=max(1, int(args.workers)),
            skip_videos=bool(args.skip_videos),
        )
        _write_provenance(output_dir, input_dir)
    print(f"DONE output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
