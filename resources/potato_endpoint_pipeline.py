from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd

from .scripts import analyze_dario_potato_dataset as analyzer


POTATO_ENDPOINT_ANALYSIS_MODE = "potato_endpoint"
PotatoEndpointCancelled = analyzer.PotatoEndpointCancelled


@dataclass(frozen=True, slots=True)
class PotatoEndpointPackageConfig:
    input_dir: Path
    output_dir: Path
    image_paths: tuple[Path, ...]
    identity_records: tuple[Mapping[str, object], ...] = ()
    labels_dir: Path | None = None
    pixel_size_mm: float = analyzer.LUCIFER_PIXEL_SIZE_MM
    video_fps: float = 2.0
    overwrite: bool = True


@dataclass(frozen=True, slots=True)
class PotatoEndpointPackageResult:
    output_dir: Path
    phenotype_csv: Path
    branch_csv: Path
    qc_csv: Path
    plate_map_csv: Path
    workbook: Path
    overlay_video: Path
    mask_video: Path
    sample_frame: Path
    provenance_json: Path
    finalization_json: Path
    analyzed_images: int
    branch_rows: int
    review_rows: int


def bundled_model_paths() -> tuple[Path, Path, Path]:
    model_dir = Path(__file__).resolve().parent / "builtin_models" / "potato"
    return (
        model_dir / analyzer.DEFAULT_ROOT_MODEL.name,
        model_dir / analyzer.DEFAULT_ROOT_WEIGHTS.name,
        model_dir / analyzer.DEFAULT_VALIDATION.name,
    )


def validate_bundled_models() -> tuple[Path, Path, Path]:
    paths = bundled_model_paths()
    missing = [path for path in paths[:2] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The Potato endpoint package is missing bundled model files: "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def _dataset_root(input_dir: Path) -> Path:
    input_path = Path(input_dir).expanduser().resolve()
    return input_path.parent if input_path.name.casefold() == "raw" else input_path


def run_potato_endpoint_package(
    config: PotatoEndpointPackageConfig,
    *,
    progress_callback: Callable[[int, int, str], object] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> PotatoEndpointPackageResult:
    root_model, root_weights, validation_path = validate_bundled_models()
    input_dir = Path(config.input_dir).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    dataset_root = _dataset_root(input_dir)
    labels_dir = (
        Path(config.labels_dir).expanduser().resolve()
        if config.labels_dir is not None
        else dataset_root / "labeled"
    )
    image_paths = tuple(Path(path).expanduser().resolve() for path in config.image_paths)
    image_stems = {path.stem for path in image_paths}
    known_capture_stems = set(analyzer.KNOWN_NON_PLANT_CAPTURES)
    is_original_dario_capture_set = bool(known_capture_stems) and known_capture_stems.issubset(image_stems)
    args = Namespace(
        dataset=dataset_root,
        input_dir=input_dir,
        output=output_dir,
        image_paths=image_paths,
        labels_dir=labels_dir,
        identity_records=[dict(record) for record in config.identity_records],
        identity_manifest=None,
        root_model=root_model,
        root_weights=root_weights,
        validation_path=validation_path,
        pixel_size_mm=float(config.pixel_size_mm),
        limit=0,
        stems="",
        include_calibration=False,
        no_manual_labels=not labels_dir.is_dir(),
        skip_videos=False,
        video_fps=float(config.video_fps),
        overwrite=bool(config.overwrite),
        generic_dataset=not is_original_dario_capture_set,
        enable_forced_rescues=False,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
    analyzer.run(args)

    phenotype_csv = output_dir / "npec_potato_endpoint_phenotypes.csv"
    branch_csv = output_dir / "npec_potato_branch_traits.csv"
    qc_csv = output_dir / "npec_potato_qc.csv"
    phenotype_df = pd.read_csv(phenotype_csv)
    branch_df = pd.read_csv(branch_csv)
    qc_df = pd.read_csv(qc_csv)
    finalization_path = output_dir / "npec_potato_finalization_summary.json"
    if finalization_path.exists():
        json.loads(finalization_path.read_text(encoding="utf-8"))
    return PotatoEndpointPackageResult(
        output_dir=output_dir,
        phenotype_csv=phenotype_csv,
        branch_csv=branch_csv,
        qc_csv=qc_csv,
        plate_map_csv=output_dir / "npec_potato_plate_label_map.csv",
        workbook=output_dir / "npec_potato_endpoint_all_metrics.xlsx",
        overlay_video=output_dir / "npec_potato_endpoint_overlay_reel.mp4",
        mask_video=output_dir / "npec_potato_endpoint_mask_reel.mp4",
        sample_frame=output_dir / "npec_potato_endpoint_video_sample_frame.png",
        provenance_json=output_dir / "npec_potato_analysis_provenance.json",
        finalization_json=finalization_path,
        analyzed_images=len(phenotype_df),
        branch_rows=len(branch_df),
        review_rows=int(pd.to_numeric(qc_df["review_required"], errors="coerce").fillna(0).astype(bool).sum()),
    )


__all__ = [
    "POTATO_ENDPOINT_ANALYSIS_MODE",
    "PotatoEndpointCancelled",
    "PotatoEndpointPackageConfig",
    "PotatoEndpointPackageResult",
    "analyzer",
    "bundled_model_paths",
    "run_potato_endpoint_package",
    "validate_bundled_models",
]
