from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd

from .analytics_engine import (
    AnalyticsConfig,
    _build_owned_masks_by_frame,
    _build_owned_shoot_masks_by_frame,
    _build_track_compartment_hints,
    _mask_bbox,
    _measure_crop,
    _measure_lateral_segments,
    _primary_root_view,
    _remove_short_runs,
    open_mp4_video_writer,
)


FOLDER_PATTERNS = [
    re.compile(r"^(?P<pos>.+?)__+(?P<ts>20\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})_(?P<seq>\d{1,6})$"),
    re.compile(r"^(?P<pos>.+?)_(?P<ts>20\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})_(?P<seq>\d{1,6})$"),
    re.compile(r"^(?P<pos>.+?)(?P<ts>20\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})_(?P<seq>\d{1,6})$"),
]
TIMESTAMP_PATTERNS = [
    re.compile(r"(20\d{2}[._-]\d{2}[._-]\d{2}[ _-]\d{2}[._-]\d{2}[._-]\d{2})"),
    re.compile(r"(20\d{2}\d{2}\d{2}\d{2}\d{2}\d{2})"),
]
FRAME_PATTERNS = [
    re.compile(r"(?:frame|img|image|seq|index|timepoint|t)(?:[_-]?)(\d{1,7})$", re.IGNORECASE),
    re.compile(r"(?:^|[_-])(\d{1,7})$", re.IGNORECASE),
]
PETRI_PATTERNS = [
    re.compile(r"(?:petri|dish|plate|position|pos|well)[ _-]*([A-Za-z0-9._-]+)", re.IGNORECASE),
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".jxl"}
MEASUREMENT_SUFFIXES = {".xlsx", ".xls", ".csv", ".tsv"}
MEASUREMENT_EXACT_NAMES = {
    "measurements.xlsx",
    "measurements.csv",
    "measurements.tsv",
    "plant_metrics.csv",
    "npec_plant_metrics.csv",
    "lazy_segmentation_run_details.csv",
}
MEASUREMENT_TOKENS = (
    "measurement",
    "metrics",
    "plant_metrics",
    "root_length",
    "length",
)


@dataclass(slots=True)
class ConsolidationConfig:
    fps: int = 1
    width: int = 1920
    height: int = 1080


@dataclass(slots=True)
class PathMetadata:
    petri: str | None
    timestamp: pd.Timestamp | None
    frame_index: int | None
    series: str
    relative_folder: str


def _parse_timestamp_token(token: str) -> pd.Timestamp | None:
    raw = str(token).strip()
    if not raw:
        return None
    if re.fullmatch(r"20\d{12}", raw):
        try:
            ts = pd.to_datetime(raw, format="%Y%m%d%H%M%S", errors="coerce")
            return None if pd.isna(ts) else ts
        except Exception:
            return None
    normalized = re.sub(r"[ .-]+", "_", raw)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not re.fullmatch(r"20\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}", normalized):
        return None
    try:
        ts = pd.to_datetime(normalized, format="%Y_%m_%d_%H_%M_%S", errors="coerce")
    except Exception:
        return None
    return None if pd.isna(ts) else ts


def _parse_folder_metadata_extended(folder_name: str) -> tuple[str | None, pd.Timestamp | None, int | None]:
    raw = re.sub(r"\s+", " ", str(folder_name).strip())
    for pat in FOLDER_PATTERNS:
        match = pat.match(raw)
        if match is None:
            continue
        pos = match.group("pos").strip()
        timestamp = _parse_timestamp_token(match.group("ts"))
        seq_raw = match.group("seq")
        seq = int(seq_raw) if seq_raw.isdigit() else None
        return pos if pos else None, timestamp, seq
    return None, None, None


def parse_folder_metadata(folder_name: str) -> tuple[str | None, pd.Timestamp | None]:
    pos, timestamp, _ = _parse_folder_metadata_extended(folder_name)
    return pos, timestamp


def _normalize_name(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(name).strip().lower())


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    wanted = {_normalize_name(c) for c in candidates}
    for col in df.columns:
        if _normalize_name(str(col)) in wanted:
            return str(col)
    return None


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace("\u200b", "").replace("\xa0", " ") for c in out.columns]
    keep_cols = [c for c in out.columns if not str(c).startswith("Unnamed")]
    return out[keep_cols]


def _read_measurement_file(path: Path) -> pd.DataFrame | None:
    try:
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        else:
            return None
    except Exception:
        return None
    if df is None or df.empty:
        return None
    return _clean_columns(df)


def _is_measurement_filename(name: str) -> bool:
    low = str(name).lower()
    suffix = Path(low).suffix
    if suffix not in MEASUREMENT_SUFFIXES:
        return False
    if low in MEASUREMENT_EXACT_NAMES:
        return True
    return any(token in low for token in MEASUREMENT_TOKENS)


def _natural_sort_key(text: str) -> tuple[tuple[int, object], ...]:
    tokens = re.findall(r"\d+|\D+", str(text).lower())
    key: list[tuple[int, object]] = []
    for token in tokens:
        if token.isdigit():
            key.append((0, int(token)))
        else:
            key.append((1, token))
    return tuple(key)


def _extract_timestamp_from_text(text: str) -> pd.Timestamp | None:
    raw = str(text)
    for pattern in TIMESTAMP_PATTERNS:
        match = pattern.search(raw)
        if match is None:
            continue
        ts = _parse_timestamp_token(match.group(1))
        if ts is not None:
            return ts
    return None


def _extract_frame_index_from_text(text: str) -> int | None:
    raw = str(text).strip()
    if not raw:
        return None
    for pattern in FRAME_PATTERNS:
        match = pattern.search(raw)
        if match is None:
            continue
        token = match.group(1)
        if token.isdigit():
            return int(token)
    return None


def _extract_petri_from_text(text: str) -> str | None:
    raw = str(text).strip()
    if not raw:
        return None
    for pattern in PETRI_PATTERNS:
        match = pattern.search(raw)
        if match is None:
            continue
        token = str(match.group(1)).strip("_- .")
        if token:
            return token
    return None


def _clean_text_series(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.replace({"nan": "", "None": "", "none": "", "<NA>": ""})
    return out


def _pick_preview_image(dir_path: Path, filenames: list[str]) -> str:
    image_names = [name for name in filenames if Path(name).suffix.lower() in IMAGE_SUFFIXES]
    if not image_names:
        return ""

    def _is_mask_like(name: str) -> bool:
        low = str(name).lower()
        return any(token in low for token in ("mask", "prediction", "segmentation", "overlay"))

    def _score(name: str) -> tuple[int, tuple[tuple[int, object], ...]]:
        low = name.lower()
        if "image" in low or "original" in low or "raw" in low:
            return (0, _natural_sort_key(low))
        if not _is_mask_like(low):
            return (1, _natural_sort_key(low))
        if low == "image_mask.png":
            return (4, _natural_sort_key(low))
        if "mask" in low and ("image" in low or "prediction" in low or "seg" in low):
            return (5, _natural_sort_key(low))
        if "mask" in low:
            return (6, _natural_sort_key(low))
        return (7, _natural_sort_key(low))

    selected = sorted(image_names, key=_score)[0]
    return str((dir_path / selected).resolve())


def _candidate_image_search_roots(root: Path, dir_path: Path, extra_roots: list[Path] | None = None) -> list[Path]:
    candidates = [dir_path, root, dir_path.parent, root.parent]
    if extra_roots:
        candidates.extend(extra_roots)
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def _resolve_existing_image_path(candidate: str, search_roots: list[Path]) -> Path | None:
    raw = str(candidate).strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve() if path.exists() and path.suffix.lower() in IMAGE_SUFFIXES else None
    for base in search_roots:
        full = base / raw
        if full.exists() and full.suffix.lower() in IMAGE_SUFFIXES:
            return full.resolve()
    return None


def _resolve_existing_file_path(candidate: str, dir_path: Path) -> Path | None:
    raw = str(candidate).strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve() if path.exists() else None
    full = dir_path / raw
    if full.exists():
        return full.resolve()
    return None


def _strip_preview_suffix(stem: str) -> str:
    text = str(stem)
    for suffix in ("_mask", "_prediction", "_pred", "_annotation", "_annotations", "_label", "_labels", "_overlay"):
        if text.lower().endswith(suffix):
            return text[: -len(suffix)]
    return text


def _resolve_mask_path_for_row(row: pd.Series, dir_path: Path) -> str:
    for key in ("MaskPath", "OutputMaskPath", "output_mask", "mask_path", "mask", "prediction_path", "prediction"):
        if key not in row.index:
            continue
        value = row.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        resolved = _resolve_existing_file_path(str(value), dir_path)
        if resolved is not None:
            return str(resolved)
    return ""


def _resolve_preview_image_for_row(
    row: pd.Series,
    root: Path,
    dir_path: Path,
    fallback_preview: str,
    extra_roots: list[Path] | None = None,
) -> str:
    search_roots = _candidate_image_search_roots(root, dir_path, extra_roots=extra_roots)
    for key in (
        "image_path",
        "source_image",
        "source_path",
        "SourceFile",
        "sourcefile",
        "image_name",
        "filename",
        "file_name",
        "ImageName",
        "Image",
        "PreviewImagePath",
    ):
        if key not in row.index:
            continue
        value = row.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        resolved = _resolve_existing_image_path(str(value), search_roots)
        if resolved is not None and key == "PreviewImagePath":
            low = resolved.name.lower()
            if any(token in low for token in ("mask", "prediction", "segmentation", "overlay")):
                resolved = None
        if resolved is not None:
            return str(resolved)

    mask_path = _resolve_mask_path_for_row(row, dir_path)
    if mask_path:
        mask = Path(mask_path)
        stem = _strip_preview_suffix(mask.stem)
        for base in search_roots:
            for suffix in sorted(IMAGE_SUFFIXES):
                candidate = base / f"{stem}{suffix}"
                if candidate.exists():
                    return str(candidate.resolve())

    if fallback_preview:
        fallback = Path(fallback_preview)
        if fallback.exists():
            return str(fallback.resolve())
    return ""


def _derive_path_metadata(root: Path, directory: Path, file_stem: str) -> PathMetadata:
    try:
        rel = directory.relative_to(root)
    except Exception:
        rel = Path(directory.name)
    rel_parts = [str(part) for part in rel.parts]
    series = rel_parts[0] if rel_parts else root.name
    relative_folder = str(rel) if rel_parts else "."

    petri: str | None = None
    timestamp: pd.Timestamp | None = None
    frame_index: int | None = None

    candidates: list[str] = [file_stem, directory.name]
    candidates.extend(reversed(rel_parts))
    for token in candidates:
        pos, ts, seq = _parse_folder_metadata_extended(token)
        if petri is None and pos:
            petri = pos
        if timestamp is None and ts is not None and not pd.isna(ts):
            timestamp = ts
        if frame_index is None and seq is not None:
            frame_index = seq
        if frame_index is None:
            frame_index = _extract_frame_index_from_text(token)
        if petri is None:
            petri = _extract_petri_from_text(token)

    if timestamp is None:
        joined = " ".join(rel_parts + [file_stem])
        timestamp = _extract_timestamp_from_text(joined)
    if petri is None and rel_parts:
        petri = rel_parts[0]
    if petri is not None:
        petri = str(petri).strip()
        if not petri:
            petri = None

    return PathMetadata(
        petri=petri,
        timestamp=timestamp,
        frame_index=frame_index,
        series=series,
        relative_folder=relative_folder,
    )


def consolidate_measurements(
    root_dir: Path,
    preview_search_roots: list[Path] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")
    extra_preview_roots = [
        Path(path).expanduser().resolve()
        for path in (preview_search_roots or [])
        if path is not None and Path(path).expanduser().exists()
    ]

    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    no_timestamp_files = 0
    no_petri_files = 0
    scanned_measurement_files = 0

    for dirpath, _, filenames in os.walk(root):
        if not filenames:
            continue
        measurement_files = [name for name in filenames if _is_measurement_filename(name)]
        if not measurement_files:
            continue

        dir_path = Path(dirpath)
        preview_image = _pick_preview_image(dir_path, filenames)
        for filename in measurement_files:
            scanned_measurement_files += 1
            file_path = dir_path / filename
            metadata = _derive_path_metadata(root, dir_path, file_path.stem)
            df = _read_measurement_file(file_path)
            if df is None:
                warnings.append(f"Skipped unreadable file: {file_path}")
                continue

            out = df.copy()
            petri_column = _find_column(out, ("PetriDish", "petri", "dish", "plate", "position", "well", "plate_id"))
            timestamp_column = _find_column(out, ("Timestamp", "timestamp", "time", "datetime", "date", "acquisition_time"))
            frame_column = _find_column(out, ("FrameIndex", "frame_index", "frame", "sequence", "seq", "timepoint", "index"))

            if petri_column is not None:
                petri_series = _clean_text_series(out[petri_column])
            else:
                petri_series = pd.Series("", index=out.index, dtype=object)

            petri_fallback = metadata.petri or ""
            petri_series = petri_series.where(petri_series != "", petri_fallback)
            petri_series = petri_series.where(petri_series != "", "Unknown")
            out["PetriDish"] = petri_series
            if (petri_series == "Unknown").all():
                no_petri_files += 1

            if timestamp_column is not None:
                ts_series = pd.to_datetime(out[timestamp_column], errors="coerce")
            else:
                ts_series = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
            if metadata.timestamp is not None:
                ts_series = ts_series.fillna(metadata.timestamp)
            out["Timestamp"] = ts_series
            if bool(ts_series.isna().all()):
                no_timestamp_files += 1

            if frame_column is not None:
                frame_series = pd.to_numeric(out[frame_column], errors="coerce")
            else:
                frame_series = pd.Series(np.nan, index=out.index, dtype=float)
            if metadata.frame_index is not None:
                frame_series = frame_series.fillna(float(metadata.frame_index))
            out["FrameIndex"] = frame_series

            out["Series"] = str(metadata.series)
            out["RelativeFolder"] = str(metadata.relative_folder)
            out["Folder"] = dir_path.name
            out["SourceFile"] = str(file_path.resolve())
            out["MaskPath"] = out.apply(lambda row: _resolve_mask_path_for_row(row, dir_path), axis=1)
            out["PreviewImagePath"] = out.apply(
                lambda row: _resolve_preview_image_for_row(
                    row,
                    root,
                    dir_path,
                    preview_image,
                    extra_roots=extra_preview_roots,
                ),
                axis=1,
            )
            frames.append(out)

    if not frames:
        if scanned_measurement_files <= 0:
            warnings.append("No measurement files found under selected root.")
        return pd.DataFrame(), warnings

    merged = pd.concat(frames, ignore_index=True)
    merged["Timestamp"] = pd.to_datetime(merged.get("Timestamp"), errors="coerce")
    merged["FrameIndex"] = pd.to_numeric(merged.get("FrameIndex"), errors="coerce")
    if "frame_index" in merged.columns:
        merged["FrameIndex"] = merged["FrameIndex"].fillna(pd.to_numeric(merged["frame_index"], errors="coerce"))

    merged["_sort_series"] = merged["Series"].fillna("").astype(str).map(_natural_sort_key)
    merged["_sort_petri"] = merged["PetriDish"].fillna("Unknown").astype(str).map(_natural_sort_key)
    merged["_sort_frame"] = merged["FrameIndex"].fillna(1.0e12).astype(float)
    plant_col = "plant_id" if "plant_id" in merged.columns else ("plant" if "plant" in merged.columns else None)
    if plant_col is not None:
        merged["_sort_plant"] = _clean_text_series(merged[plant_col]).map(_natural_sort_key)
    else:
        merged["_sort_plant"] = pd.Series([tuple()] * len(merged), index=merged.index, dtype=object)

    merged.sort_values(
        by=["_sort_series", "_sort_petri", "Timestamp", "_sort_frame", "_sort_plant"],
        kind="mergesort",
        inplace=True,
    )
    merged.drop(columns=["_sort_series", "_sort_petri", "_sort_frame", "_sort_plant"], inplace=True, errors="ignore")
    merged.reset_index(drop=True, inplace=True)
    merged = _apply_bbox_owned_measurement_refresh(merged)

    if no_petri_files > 0:
        warnings.append(f"Files without petri metadata: {no_petri_files} (set to 'Unknown').")
    if no_timestamp_files > 0:
        warnings.append(f"Files without timestamp metadata: {no_timestamp_files} (timeline ordering may be partial).")
    return merged, warnings


def find_metric_columns(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    block = {
        "Folder",
        "RelativeFolder",
        "Series",
        "PetriDish",
        "Timestamp",
        "FrameIndex",
        "SourceFile",
        "PreviewImagePath",
        "MaskPath",
        "plant",
        "plant_id",
        "uid",
        "image_name",
        "frame_index",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    }
    numeric = []
    for col in df.columns:
        if col in block:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() > 0:
            numeric.append(str(col))
    return numeric


def _metric_group_column(df: pd.DataFrame) -> str:
    if "plant_id" in df.columns:
        return "plant_id"
    if "Plant ID" in df.columns:
        return "Plant ID"
    if "PlantID" in df.columns:
        return "PlantID"
    if "plant" in df.columns:
        return "plant"
    if "Series" in df.columns:
        return "Series"
    return "PetriDish"


def _format_timestamp_label(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return "-"
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _timestamp_key(ts: pd.Timestamp | None) -> int | None:
    if ts is None or pd.isna(ts):
        return None
    try:
        return int(pd.Timestamp(ts).value)
    except Exception:
        return None


def _has_valid_timestamps(df: pd.DataFrame) -> bool:
    if "Timestamp" not in df.columns:
        return False
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    return bool(ts.notna().any())


def _has_valid_frame_index(df: pd.DataFrame) -> bool:
    if "FrameIndex" not in df.columns:
        return False
    frame = pd.to_numeric(df["FrameIndex"], errors="coerce")
    return bool(frame.notna().any())


def _pretty_metric_title(metric_col: str) -> str:
    raw = str(metric_col).strip()
    special = {
        "lateral_root_count": "Lateral Root Count",
        "lateral_count": "Lateral Root Count",
        "primary_length_mm": "Primary Root Length (mm)",
        "lateral_length_mm": "Lateral Root Length (mm)",
        "total_length_mm": "Total Root Length (mm)",
        "root_length_mm": "Root Length (mm)",
        "leaf_size_px": "Leaf Size (px)",
    }
    key = raw.lower()
    if key in special:
        return special[key]
    text = raw.replace("_", " ").replace("-", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.title() if text else raw


def _pretty_track_name(name: str) -> str:
    text = str(name).strip()
    match = re.fullmatch(r"plant[_\s-]*0*([0-9]+)", text, flags=re.IGNORECASE)
    if match is not None:
        return f"Plant {int(match.group(1))}"
    return re.sub(r"[_-]+", " ", text).strip() or text


def _plot_context_label(df: pd.DataFrame) -> str:
    if "PetriDish" in df.columns:
        values = [str(v).strip() for v in df["PetriDish"].dropna().unique().tolist() if str(v).strip() and str(v).strip().lower() != "unknown"]
        if len(values) == 1:
            return f"Petri Dish {values[0]}"
    if "Series" in df.columns:
        values = [str(v).strip() for v in df["Series"].dropna().unique().tolist() if str(v).strip()]
        if len(values) == 1:
            return values[0]
    return "Consolidated measurements"


TRACK_PALETTE: list[tuple[int, int, int]] = [
    (255, 95, 109),
    (75, 192, 192),
    (255, 205, 86),
    (153, 102, 255),
    (255, 159, 64),
    (67, 181, 129),
    (250, 130, 49),
    (116, 185, 255),
    (214, 112, 218),
    (245, 149, 99),
]


def _metric_track_palette() -> list[tuple[int, int, int]]:
    return TRACK_PALETTE


def _metric_target_class_ids(metric_col: str) -> set[int] | None:
    key = str(metric_col).strip().lower()
    if not key:
        return None
    if "lateral" in key:
        return {4}
    if "shoot" in key or "leaf" in key or "rosette" in key:
        return {2}
    if "seed" in key or "colony" in key:
        return {5}
    if "primary" in key or "main_root" in key or "main root" in key:
        return {1, 3}
    if "root" in key:
        return {1, 3, 4}
    return None


def _metric_rows_for_timestamp(df: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    if "Timestamp" not in df.columns:
        return pd.DataFrame()
    subset = df.copy()
    subset["Timestamp"] = pd.to_datetime(subset["Timestamp"], errors="coerce")
    subset = subset.dropna(subset=["Timestamp"])
    if subset.empty:
        return subset.iloc[0:0]
    ts = pd.Timestamp(timestamp)
    exact = subset[subset["Timestamp"] == ts]
    if not exact.empty:
        return exact.copy()
    prior = subset[subset["Timestamp"] <= ts].sort_values("Timestamp")
    if prior.empty:
        return subset.iloc[0:0]
    latest = pd.Timestamp(prior.iloc[-1]["Timestamp"])
    return prior[prior["Timestamp"] == latest].copy()


def _metric_rows_for_frame(df: pd.DataFrame, frame_index: float) -> pd.DataFrame:
    if "FrameIndex" not in df.columns:
        return pd.DataFrame()
    subset = df.copy()
    subset["FrameIndex"] = pd.to_numeric(subset["FrameIndex"], errors="coerce")
    subset = subset.dropna(subset=["FrameIndex"])
    if subset.empty:
        return subset.iloc[0:0]
    frame_value = float(frame_index)
    exact = subset[np.isclose(subset["FrameIndex"].astype(float), frame_value, atol=1e-6)]
    if not exact.empty:
        return exact.copy()
    prior = subset[subset["FrameIndex"] <= frame_value].sort_values("FrameIndex")
    if prior.empty:
        return subset.iloc[0:0]
    latest = float(prior.iloc[-1]["FrameIndex"])
    return prior[np.isclose(prior["FrameIndex"].astype(float), latest, atol=1e-6)].copy()


def _coerce_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _choose_anchor_box(row: pd.Series, image_shape: tuple[int, int]) -> tuple[tuple[float, float], tuple[int, int, int, int]] | None:
    height, width = int(image_shape[0]), int(image_shape[1])
    candidate_sets = [
        ("shoot_bbox_x", "shoot_bbox_y", "shoot_bbox_w", "shoot_bbox_h"),
        ("bbox_x", "bbox_y", "bbox_w", "bbox_h"),
        ("combined_group_bbox_x", "combined_group_bbox_y", "combined_group_bbox_w", "combined_group_bbox_h"),
    ]
    for x_key, y_key, w_key, h_key in candidate_sets:
        if x_key not in row.index or y_key not in row.index or w_key not in row.index or h_key not in row.index:
            continue
        x = _coerce_float(row.get(x_key))
        y = _coerce_float(row.get(y_key))
        w = _coerce_float(row.get(w_key))
        h = _coerce_float(row.get(h_key))
        if x is None or y is None or w is None or h is None or w <= 0 or h <= 0:
            continue
        x0 = max(0, min(width - 1, int(round(x))))
        y0 = max(0, min(height - 1, int(round(y))))
        x1 = max(x0 + 1, min(width, int(round(x + w))))
        y1 = max(y0 + 1, min(height, int(round(y + h))))
        cx = (x0 + x1 - 1) * 0.5
        cy = (y0 + y1 - 1) * 0.5
        return (cx, cy), (x0, y0, x1, y1)
    return None


def _choose_shoot_box(row: pd.Series, image_shape: tuple[int, int]) -> tuple[float, float, tuple[int, int, int, int]] | None:
    height, width = int(image_shape[0]), int(image_shape[1])
    if all(key in row.index for key in ("shoot_bbox_x", "shoot_bbox_y", "shoot_bbox_w", "shoot_bbox_h")):
        x = _coerce_float(row.get("shoot_bbox_x"))
        y = _coerce_float(row.get("shoot_bbox_y"))
        w = _coerce_float(row.get("shoot_bbox_w"))
        h = _coerce_float(row.get("shoot_bbox_h"))
        if x is not None and y is not None and w is not None and h is not None and w > 0 and h > 0:
            x0 = max(0, min(width - 1, int(round(x))))
            y0 = max(0, min(height - 1, int(round(y))))
            x1 = max(x0 + 1, min(width, int(round(x + w))))
            y1 = max(y0 + 1, min(height, int(round(y + h))))
            cx = (x0 + x1 - 1) * 0.5
            cy = (y0 + y1 - 1) * 0.5
            return cx, cy, (x0, y0, x1, y1)
    anchor = _choose_anchor_box(row, image_shape)
    if anchor is None:
        return None
    (cx, cy), box = anchor
    return cx, cy, box


def _box_xyxy_to_bbox(box: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    x0, y0, x1, y1 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _union_bbox_xywh(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None, image_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    height, width = int(image_shape[0]), int(image_shape[1])
    boxes = [box for box in (a, b) if box is not None]
    if not boxes:
        return None
    x0 = min(int(box[0]) for box in boxes)
    y0 = min(int(box[1]) for box in boxes)
    x1 = max(int(box[0] + box[2]) for box in boxes)
    y1 = max(int(box[1] + box[3]) for box in boxes)
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return (x0, y0, x1 - x0, y1 - y0)


def _tracking_bbox_for_row(row: pd.Series, image_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    shoot_anchor = _choose_shoot_box(row, image_shape)
    shoot_box = _box_xyxy_to_bbox(shoot_anchor[2]) if shoot_anchor is not None else None
    root_anchor = _choose_anchor_box(row, image_shape)
    root_box = _box_xyxy_to_bbox(root_anchor[1]) if root_anchor is not None else None
    return _union_bbox_xywh(shoot_box, root_box, image_shape)


def _current_class_id(current_rows: pd.DataFrame, column: str, fallback: int | None) -> int | None:
    if column in current_rows.columns:
        values = pd.to_numeric(current_rows[column], errors="coerce").dropna().tolist()
        for value in values:
            try:
                class_id = int(round(float(value)))
            except Exception:
                continue
            if class_id > 0:
                return class_id
    return fallback


def _build_metric_owned_regions(
    mask: np.ndarray,
    current_rows: pd.DataFrame,
    metric_col: str,
) -> dict[str, np.ndarray] | None:
    owned = _build_owned_masks_for_frame(mask, current_rows)
    if owned is None:
        return None
    owned_root, owned_lateral, owned_shoot, _config = owned
    key = str(metric_col).strip().lower()
    regions: dict[str, np.ndarray] = {}
    for name in owned_root.keys():
        root_mask = (np.asarray(owned_root.get(name, np.zeros_like(mask, dtype=np.uint8)), dtype=np.uint8) > 0)
        lateral_mask = (np.asarray(owned_lateral.get(name, np.zeros_like(mask, dtype=np.uint8)), dtype=np.uint8) > 0)
        shoot_mask = (np.asarray(owned_shoot.get(name, np.zeros_like(mask, dtype=np.uint8)), dtype=np.uint8) > 0)
        if "lateral" in key:
            region = lateral_mask
        elif "shoot" in key or "leaf" in key or "rosette" in key:
            region = shoot_mask
        elif "primary" in key or "main_root" in key or "main root" in key:
            region = root_mask & ~lateral_mask
        else:
            region = root_mask
        regions[name] = region.astype(bool, copy=False)
    return regions


def _build_owned_masks_for_frame(
    mask: np.ndarray,
    current_rows: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], AnalyticsConfig] | None:
    if current_rows.empty:
        return None
    group_col = _metric_group_column(current_rows)
    if group_col not in current_rows.columns:
        return None

    grouped_rows: list[tuple[str, pd.Series]] = []
    for group_name, group_df in current_rows.groupby(group_col):
        if group_df.empty:
            continue
        grouped_rows.append((str(group_name), group_df.iloc[0]))
    grouped_rows.sort(key=lambda item: _natural_sort_key(item[0]))
    if not grouped_rows:
        return None

    root_class_id = _current_class_id(current_rows, "root_class_id", 3)
    lateral_class_id = _current_class_id(current_rows, "lateral_class_id", 4)
    shoot_class_id = _current_class_id(current_rows, "shoot_class_id", 2)
    seed_class_id = _current_class_id(current_rows, "seed_class_id", None)

    valid_root_ids = [int(root_class_id)] if root_class_id and int(root_class_id) > 0 else []
    if lateral_class_id and int(lateral_class_id) > 0 and int(lateral_class_id) not in valid_root_ids:
        valid_root_ids.append(int(lateral_class_id))
    if not valid_root_ids:
        return None

    root_source = np.isin(mask, np.asarray(valid_root_ids, dtype=np.uint8)).astype(np.uint8)
    lateral_source = (
        (mask == np.uint8(int(lateral_class_id))).astype(np.uint8)
        if lateral_class_id and int(lateral_class_id) > 0
        else np.zeros_like(mask, dtype=np.uint8)
    )

    anchor_ids: list[int] = []
    if seed_class_id and int(seed_class_id) > 0:
        anchor_ids.append(int(seed_class_id))
    elif shoot_class_id and int(shoot_class_id) > 0:
        anchor_ids.append(int(shoot_class_id))
    if anchor_ids:
        anchor_source = np.isin(mask, np.asarray(anchor_ids, dtype=np.uint8)).astype(np.uint8)
    else:
        anchor_source = np.zeros_like(mask, dtype=np.uint8)
    shoot_source = (
        (mask == np.uint8(int(shoot_class_id))).astype(np.uint8)
        if shoot_class_id and int(shoot_class_id) > 0
        else anchor_source.copy()
    )

    track_ids = [name for name, _row in grouped_rows]
    track_bboxes: dict[str, list[tuple[int, int, int, int]]] = {}
    for name, row in grouped_rows:
        bbox = _tracking_bbox_for_row(row, mask.shape)
        if bbox is None:
            continue
        track_bboxes[name] = [bbox]
    if not track_bboxes:
        return None

    config = AnalyticsConfig(
        root_class_id=int(root_class_id) if root_class_id else 3,
        lateral_class_id=int(lateral_class_id) if lateral_class_id else None,
        seed_class_id=int(seed_class_id) if seed_class_id else None,
        shoot_class_id=int(shoot_class_id) if shoot_class_id else None,
        track_lane_partition_enabled=True,
        track_seed_connectivity_enabled=True,
        shoot_tracking_enabled=False,
        shoot_crown_lock_enabled=False,
        root_temporal_seed_enabled=False,
        bbox_padding=12,
    )
    if "pixel_size_mm" in current_rows.columns:
        pixel_values = pd.to_numeric(current_rows["pixel_size_mm"], errors="coerce").dropna().tolist()
        if pixel_values:
            config.pixel_size_mm = float(pixel_values[0])
    hints = _build_track_compartment_hints(
        root_masks=[root_source],
        anchor_masks=[anchor_source],
        track_ids=track_ids,
        track_bboxes=track_bboxes,
        seed_frame=0,
        config=config,
    )
    if not hints:
        return None

    owned_shoot_frames, _shoot_meta = _build_owned_shoot_masks_by_frame(
        shoot_masks=[shoot_source],
        track_ids=track_ids,
        compartment_hints=hints,
        config=config,
    )
    owned_root_frames, owned_lateral_frames, _ownership_meta = _build_owned_masks_by_frame(
        root_masks=[root_source],
        lateral_masks=[lateral_source],
        owned_shoot_frames=owned_shoot_frames,
        track_ids=track_ids,
        compartment_hints=hints,
        config=config,
        tip_priors_by_frame=None,
    )
    owned_root = owned_root_frames[0] if owned_root_frames else {}
    owned_lateral = owned_lateral_frames[0] if owned_lateral_frames else {}
    owned_shoot = owned_shoot_frames[0] if owned_shoot_frames else {}
    for name in track_ids:
        owned_root.setdefault(name, np.zeros_like(mask, dtype=np.uint8))
        owned_lateral.setdefault(name, np.zeros_like(mask, dtype=np.uint8))
        owned_shoot.setdefault(name, np.zeros_like(mask, dtype=np.uint8))
    return owned_root, owned_lateral, owned_shoot, config


def _recompute_owned_metrics_for_frame(frame_df: pd.DataFrame, cache: dict[str, np.ndarray | None]) -> pd.DataFrame:
    if frame_df.empty:
        return frame_df
    mask_path = ""
    if "MaskPath" in frame_df.columns:
        for value in frame_df["MaskPath"].tolist():
            text = str(value).strip()
            if text:
                mask_path = text
                break
    mask = _load_index_mask(mask_path, cache)
    if mask is None or mask.size == 0:
        return frame_df

    owned = _build_owned_masks_for_frame(mask, frame_df)
    if owned is None:
        return frame_df
    owned_root, owned_lateral, owned_shoot, config = owned
    group_col = _metric_group_column(frame_df)
    if group_col not in frame_df.columns:
        return frame_df

    out = frame_df.copy()
    shape_hw = mask.shape[:2]
    lateral_mode = "class_mask" if config.lateral_class_id is not None and int(config.lateral_class_id) > 0 else "derived_from_root"
    row_groups = {str(name): group_df for name, group_df in frame_df.groupby(group_col)}

    for group_name, group_df in row_groups.items():
        root_full = (np.asarray(owned_root.get(group_name, np.zeros(shape_hw, dtype=np.uint8)), dtype=np.uint8) > 0).astype(np.uint8)
        lateral_full = (np.asarray(owned_lateral.get(group_name, np.zeros(shape_hw, dtype=np.uint8)), dtype=np.uint8) > 0).astype(np.uint8)
        shoot_full = (np.asarray(owned_shoot.get(group_name, np.zeros(shape_hw, dtype=np.uint8)), dtype=np.uint8) > 0).astype(np.uint8)
        bbox_mask = root_full.copy()
        if lateral_mode == "class_mask":
            bbox_mask = np.maximum(bbox_mask, lateral_full)
        bbox = _mask_bbox(bbox_mask, config.bbox_padding, shape_hw)
        shoot_bbox = _mask_bbox(shoot_full, 0, shape_hw)
        x, y, bw, bh = bbox
        sx, sy, sbw, sbh = shoot_bbox
        primary_root_full = _primary_root_view(root_full, lateral_full if lateral_mode == "class_mask" else None, config)
        if bw > 0 and bh > 0:
            crop = primary_root_full[y : y + bh, x : x + bw]
            union_crop = root_full[y : y + bh, x : x + bw]
            lateral_crop = lateral_full[y : y + bh, x : x + bw] if lateral_mode == "class_mask" else np.zeros((bh, bw), dtype=np.uint8)
        else:
            crop = np.zeros((0, 0), dtype=np.uint8)
            union_crop = np.zeros((0, 0), dtype=np.uint8)
            lateral_crop = np.zeros((0, 0), dtype=np.uint8)
        primary_metrics = _measure_crop(crop, config)
        metrics = primary_metrics
        root_length_measurement_mode = "primary_root"
        if lateral_mode == "class_mask" and bool(getattr(config, "primary_root_tracking_excludes_lateral", True)):
            union_metrics = _measure_crop(union_crop, config)
            primary_path = metrics.get("path_xy") if isinstance(metrics, dict) else []
            union_path = union_metrics.get("path_xy") if isinstance(union_metrics, dict) else []
            if isinstance(primary_path, list) and primary_path and isinstance(union_path, list) and union_path:
                primary_tip_y = int(primary_path[-1][1]) if len(primary_path[-1]) >= 2 else -1
                union_tip_y = int(union_path[-1][1]) if len(union_path[-1]) >= 2 else -1
                if (union_tip_y - primary_tip_y) > int(getattr(config, "primary_root_tracking_fallback_gap_px", 96)):
                    metrics = union_metrics
                    root_length_measurement_mode = "total_root_fallback"
        lateral_segments = _measure_lateral_segments(lateral_crop, config)
        lateral_count = int(len(lateral_segments))
        lateral_total_length_px = float(sum(float(seg.get("length_px", 0.0)) for seg in lateral_segments))
        lateral_total_length_mm = float(sum(float(seg.get("length_mm", 0.0)) for seg in lateral_segments))
        primary_root_length_px = float(primary_metrics.get("length_px", 0.0))
        primary_root_length_mm_raw = float(primary_root_length_px * config.pixel_size_mm)
        primary_root_area_px = int(primary_metrics.get("area_px", 0))
        total_root_length_px = float(primary_root_length_px + lateral_total_length_px)
        total_root_length_mm_raw = float(primary_root_length_mm_raw + lateral_total_length_mm)
        total_root_area_px = int(np.count_nonzero(union_crop))
        total_root_area_mm2 = float(float(total_root_area_px) * (config.pixel_size_mm**2))
        total_root_measurement_mode = (
            "primary_plus_class_lateral" if lateral_mode == "class_mask" else "primary_plus_derived_lateral"
        )
        lateral_mean_length_px = float(lateral_total_length_px / float(max(1, lateral_count))) if lateral_count > 0 else 0.0
        lateral_mean_length_mm = float(lateral_total_length_mm / float(max(1, lateral_count))) if lateral_count > 0 else 0.0
        lateral_diams_px = [float(seg.get("diameter_px", 0.0)) for seg in lateral_segments if float(seg.get("diameter_px", 0.0)) > 0.0]
        lateral_mean_diameter_px = float(np.mean(lateral_diams_px)) if lateral_diams_px else 0.0
        lateral_diams_mm = [float(seg.get("diameter_mm", 0.0)) for seg in lateral_segments if float(seg.get("diameter_mm", 0.0)) > 0.0]
        lateral_mean_diameter_mm = float(np.mean(lateral_diams_mm)) if lateral_diams_mm else 0.0
        lateral_max_length_px = float(max((float(seg.get("length_px", 0.0)) for seg in lateral_segments), default=0.0))
        lateral_max_length_mm = float(max((float(seg.get("length_mm", 0.0)) for seg in lateral_segments), default=0.0))
        shoot_area_px = int(np.count_nonzero(shoot_full))
        length_px = float(metrics.get("length_px", 0.0))
        length_mm_raw = float(length_px * config.pixel_size_mm)

        for idx in group_df.index.tolist():
            out.at[idx, "bbox_x"] = int(x)
            out.at[idx, "bbox_y"] = int(y)
            out.at[idx, "bbox_w"] = int(bw)
            out.at[idx, "bbox_h"] = int(bh)
            out.at[idx, "shoot_bbox_x"] = int(sx)
            out.at[idx, "shoot_bbox_y"] = int(sy)
            out.at[idx, "shoot_bbox_w"] = int(sbw)
            out.at[idx, "shoot_bbox_h"] = int(sbh)
            out.at[idx, "root_length_px"] = float(round(length_px, 4))
            out.at[idx, "root_length_mm_raw"] = float(round(length_mm_raw, 4))
            out.at[idx, "root_area_px"] = int(metrics.get("area_px", 0))
            out.at[idx, "root_area_mm2"] = float(round(float(metrics.get("area_px", 0.0)) * (config.pixel_size_mm**2), 4))
            out.at[idx, "root_perimeter_px"] = float(round(float(metrics.get("perimeter_px", 0.0)), 4))
            out.at[idx, "root_perimeter_mm"] = float(round(float(metrics.get("perimeter_px", 0.0)) * config.pixel_size_mm, 4))
            out.at[idx, "tips"] = int(metrics.get("tips", 0))
            out.at[idx, "branches"] = int(metrics.get("branches", 0))
            out.at[idx, "base_tip_angle_deg"] = float(round(float(metrics.get("base_tip_angle_deg", 0.0)), 3))
            out.at[idx, "emergence_angle_deg"] = float(round(float(metrics.get("emergence_angle_deg", 0.0)), 3))
            out.at[idx, "convex_hull_area_px2"] = int(metrics.get("convex_hull_area_px2", 0))
            out.at[idx, "convex_hull_area_mm2"] = float(round(float(metrics.get("convex_hull_area_px2", 0.0)) * (config.pixel_size_mm**2), 4))
            out.at[idx, "aspect_ratio"] = float(round(float(metrics.get("aspect_ratio", 0.0)), 4))
            out.at[idx, "root_length_measurement_mode"] = str(root_length_measurement_mode)
            out.at[idx, "primary_root_length_px"] = float(round(primary_root_length_px, 4))
            out.at[idx, "primary_root_length_mm_raw"] = float(round(primary_root_length_mm_raw, 4))
            out.at[idx, "primary_root_area_px"] = int(primary_root_area_px)
            out.at[idx, "primary_root_area_mm2"] = float(round(float(primary_root_area_px) * (config.pixel_size_mm**2), 4))
            out.at[idx, "total_root_length_px"] = float(round(total_root_length_px, 4))
            out.at[idx, "total_root_length_mm_raw"] = float(round(total_root_length_mm_raw, 4))
            out.at[idx, "total_root_area_px"] = int(total_root_area_px)
            out.at[idx, "total_root_area_mm2"] = float(round(total_root_area_mm2, 4))
            out.at[idx, "total_root_measurement_mode"] = str(total_root_measurement_mode)
            out.at[idx, "lateral_count"] = int(lateral_count)
            out.at[idx, "lateral_total_length_px"] = float(round(lateral_total_length_px, 4))
            out.at[idx, "lateral_total_length_mm"] = float(round(lateral_total_length_mm, 4))
            out.at[idx, "lateral_mean_length_px"] = float(round(lateral_mean_length_px, 4))
            out.at[idx, "lateral_mean_length_mm"] = float(round(lateral_mean_length_mm, 4))
            out.at[idx, "lateral_mean_diameter_px"] = float(round(lateral_mean_diameter_px, 4))
            out.at[idx, "lateral_mean_diameter_mm"] = float(round(lateral_mean_diameter_mm, 4))
            out.at[idx, "lateral_max_length_px"] = float(round(lateral_max_length_px, 4))
            out.at[idx, "lateral_max_length_mm"] = float(round(lateral_max_length_mm, 4))
            out.at[idx, "shoot_area_px"] = int(shoot_area_px)
            out.at[idx, "shoot_area_mm2"] = float(round(float(shoot_area_px) * (config.pixel_size_mm**2), 4))
            out.at[idx, "ownership_recomputed_from_bbox_masks"] = True

    return out


def _apply_bbox_owned_measurement_refresh(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "SourceFile" not in df.columns or "MaskPath" not in df.columns:
        return df
    out = df.copy()
    image_cache: dict[str, np.ndarray | None] = {}
    frame_group_cols: list[str] = []
    for col in ("Series", "PetriDish", "FrameIndex", "Timestamp", "image_name", "PreviewImagePath", "MaskPath"):
        if col in out.columns:
            frame_group_cols.append(col)
    if not frame_group_cols:
        frame_group_cols = ["SourceFile"]
    grouped_frames = out.groupby(frame_group_cols, dropna=False, sort=False)
    updated_parts: list[pd.DataFrame] = []
    for _frame_key, frame_df in grouped_frames:
        updated_parts.append(_recompute_owned_metrics_for_frame(frame_df, image_cache))
    if not updated_parts:
        return out
    out = pd.concat(updated_parts, axis=0).sort_index()

    group_col = _metric_group_column(out)
    if group_col in out.columns and "root_length_mm_raw" in out.columns:
        group_keys = [col for col in ("Series", "PetriDish") if col in out.columns] + [group_col]
        for _key, group_df in out.groupby(group_keys, dropna=False, sort=False):
            ordered = group_df.copy()
            if "FrameIndex" in ordered.columns and pd.to_numeric(ordered["FrameIndex"], errors="coerce").notna().any():
                ordered["_sort"] = pd.to_numeric(ordered["FrameIndex"], errors="coerce")
            elif "Timestamp" in ordered.columns:
                ordered["_sort"] = pd.to_datetime(ordered["Timestamp"], errors="coerce")
            else:
                ordered["_sort"] = np.arange(len(ordered), dtype=float)
            ordered = ordered.sort_values("_sort", kind="mergesort")
            raw = pd.to_numeric(ordered["root_length_mm_raw"], errors="coerce").fillna(0.0).astype(float).tolist()
            cleaned = _remove_short_runs(raw, 2)
            cleaned = np.maximum.accumulate(np.asarray(cleaned, dtype=np.float64)).tolist() if cleaned else []
            for idx, clean_mm in zip(ordered.index.tolist(), cleaned):
                out.at[idx, "root_length_mm_clean"] = float(round(float(clean_mm), 4))
                pixel_size = _coerce_float(out.at[idx, "pixel_size_mm"]) or 0.0
                if pixel_size > 1.0e-9:
                    out.at[idx, "root_length_px_clean"] = float(round(float(clean_mm) / pixel_size, 4))
            for raw_col, clean_col, clean_px_col in (
                ("primary_root_length_mm_raw", "primary_root_length_mm_clean", "primary_root_length_px_clean"),
                ("total_root_length_mm_raw", "total_root_length_mm_clean", "total_root_length_px_clean"),
            ):
                if raw_col not in ordered.columns:
                    continue
                raw_metric = pd.to_numeric(ordered[raw_col], errors="coerce").fillna(0.0).astype(float).tolist()
                clean_metric = _remove_short_runs(raw_metric, 2)
                clean_metric = np.maximum.accumulate(np.asarray(clean_metric, dtype=np.float64)).tolist() if clean_metric else []
                for idx, clean_mm in zip(ordered.index.tolist(), clean_metric):
                    out.at[idx, clean_col] = float(round(float(clean_mm), 4))
                    pixel_size = _coerce_float(out.at[idx, "pixel_size_mm"]) or 0.0
                    if pixel_size > 1.0e-9:
                        out.at[idx, clean_px_col] = float(round(float(clean_mm) / pixel_size, 4))
    return out


def _track_horizontal_bands(track_rows: list[tuple[str, pd.Series, tuple[int, int, int]]], image_shape: tuple[int, int]) -> dict[str, tuple[int, int]]:
    height, width = int(image_shape[0]), int(image_shape[1])
    del height
    centers: list[tuple[str, float]] = []
    for idx, (name, row, _color) in enumerate(track_rows):
        anchor = _choose_shoot_box(row, image_shape)
        cx = float(anchor[0]) if anchor is not None else float((idx + 0.5) * width / max(1, len(track_rows)))
        centers.append((name, cx))
    centers.sort(key=lambda item: item[1])
    bands: dict[str, tuple[int, int]] = {}
    for i, (name, cx) in enumerate(centers):
        left = 0 if i == 0 else int(round((centers[i - 1][1] + cx) * 0.5))
        right = width if i == len(centers) - 1 else int(round((cx + centers[i + 1][1]) * 0.5))
        bands[name] = (max(0, left), min(width, right))
    return bands


def _extract_primary_owner_masks(
    track_rows: list[tuple[str, pd.Series, tuple[int, int, int]]],
    primary_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    if not track_rows:
        return {}
    bands = _track_horizontal_bands(track_rows, primary_mask.shape)
    owner_masks: dict[str, np.ndarray] = {}
    height, width = primary_mask.shape[:2]

    for name, row, _color in track_rows:
        anchor = _choose_shoot_box(row, primary_mask.shape)
        if anchor is None:
            continue
        cx, cy, shoot_box = anchor
        x0b, x1b = bands.get(name, (0, width))
        shoot_x0, shoot_y0, shoot_x1, shoot_y1 = shoot_box
        base_box = _choose_anchor_box(row, primary_mask.shape)
        root_box = base_box[1] if base_box is not None else (shoot_x0, shoot_y0, shoot_x1, min(height, shoot_y1 + 220))
        root_x0, root_y0, root_x1, root_y1 = root_box
        search_x0 = max(0, max(x0b, min(shoot_x0, root_x0) - 18))
        search_x1 = min(width, min(x1b, max(shoot_x1, root_x1) + 18))
        search_y0 = max(0, min(shoot_y0, root_y0))
        search_y1 = min(height, max(shoot_y1 + 180, root_y1))
        local_primary = primary_mask[search_y0:search_y1, search_x0:search_x1]
        if not np.any(local_primary):
            continue
        corridor_half_w = max(14, int(round((shoot_x1 - shoot_x0) * 0.55)))
        corridor_x0 = max(search_x0, int(round(cx)) - corridor_half_w)
        corridor_x1 = min(search_x1, int(round(cx)) + corridor_half_w)
        corridor_y0 = max(search_y0, shoot_y1 - 4)
        corridor_y1 = min(search_y1, shoot_y1 + max(80, int(round((search_y1 - search_y0) * 0.22))))
        owner = np.zeros_like(primary_mask, dtype=bool)
        owner_region = local_primary.copy()
        corridor = owner_region[(corridor_y0 - search_y0):(corridor_y1 - search_y0), (corridor_x0 - search_x0):(corridor_x1 - search_x0)]
        if np.any(corridor):
            # Keep corridor-connected primary support but do not force a single global component.
            labels = cv2.connectedComponents(owner_region.astype(np.uint8), connectivity=8)[1]
            touched = [int(v) for v in np.unique(labels[(corridor_y0 - search_y0):(corridor_y1 - search_y0), (corridor_x0 - search_x0):(corridor_x1 - search_x0)]) if int(v) > 0]
            if touched:
                owner_region = np.isin(labels, touched)
        owner[search_y0:search_y1, search_x0:search_x1] = owner_region
        owner_masks[name] = owner
    return owner_masks


def _overlay_metric_decomposition_on_image(
    image_rgb: np.ndarray | None,
    mask_path: str | None,
    current_rows: pd.DataFrame,
    metric_col: str,
    cache: dict[str, np.ndarray | None],
) -> np.ndarray | None:
    if image_rgb is None:
        return None
    if current_rows.empty:
        return _overlay_mask_on_image(image_rgb, mask_path, cache)
    target_classes = _metric_target_class_ids(metric_col)
    if not target_classes:
        return _overlay_mask_on_image(image_rgb, mask_path, cache)
    mask = _load_index_mask(mask_path, cache)
    if mask is None or mask.size == 0:
        return image_rgb

    base = np.asarray(image_rgb, dtype=np.uint8).copy()
    if mask.shape[:2] != base.shape[:2]:
        mask = cv2.resize(mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)

    metric_mask = np.isin(mask, list(target_classes))
    if not np.any(metric_mask):
        return _overlay_mask_on_image(image_rgb, mask_path, cache)

    group_col = _metric_group_column(current_rows)
    if group_col not in current_rows.columns:
        return _overlay_mask_on_image(image_rgb, mask_path, cache)

    anchors: list[tuple[str, pd.Series, tuple[int, int, int], float]] = []
    for idx, (group_name, group_df) in enumerate(sorted(((str(name), group_df) for name, group_df in current_rows.groupby(group_col)), key=lambda entry: _natural_sort_key(entry[0]))):
        row = group_df.iloc[0]
        shoot_anchor = _choose_shoot_box(row, metric_mask.shape)
        sort_x = float(shoot_anchor[0]) if shoot_anchor is not None else float(idx)
        anchors.append((str(group_name), row, _metric_track_palette()[idx % len(_metric_track_palette())], sort_x))
    anchors.sort(key=lambda item: item[3])
    track_rows = [(name, row, color) for name, row, color, _ in anchors]
    if not track_rows:
        return _overlay_mask_on_image(image_rgb, mask_path, cache)
    assigned_regions = _build_metric_owned_regions(mask=mask, current_rows=current_rows, metric_col=metric_col)
    if not assigned_regions:
        primary_mask = np.isin(mask, [1, 3])
        owner_masks = _extract_primary_owner_masks(track_rows, primary_mask)
        bands = _track_horizontal_bands(track_rows, metric_mask.shape)
        assigned_regions = {name: np.zeros(metric_mask.shape, dtype=bool) for name, _row, _color in track_rows}

        if target_classes == {4} and owner_masks:
            lateral_labels = cv2.connectedComponents(metric_mask.astype(np.uint8), connectivity=8)[1]
            comp_ids = [int(v) for v in np.unique(lateral_labels) if int(v) > 0]
            dilated_owners = {
                name: cv2.dilate(owner_mask.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1) > 0
                for name, owner_mask in owner_masks.items()
            }
            for comp_id in comp_ids:
                component = lateral_labels == comp_id
                ys, xs = np.where(component)
                if len(xs) == 0:
                    continue
                comp_cx = float(xs.mean())
                comp_y = float(ys.min())
                best_name: str | None = None
                best_score = -1e18
                for name, row, _color in track_rows:
                    if name not in dilated_owners:
                        continue
                    contact = int(np.count_nonzero(component & dilated_owners[name]))
                    band_left, band_right = bands.get(name, (0, metric_mask.shape[1]))
                    band_penalty = 0.0 if (band_left <= comp_cx < band_right) else min(abs(comp_cx - band_left), abs(comp_cx - band_right)) * 2.0
                    shoot_anchor = _choose_shoot_box(row, metric_mask.shape)
                    anchor_x = float(shoot_anchor[0]) if shoot_anchor is not None else comp_cx
                    anchor_y = float(shoot_anchor[2][3]) if shoot_anchor is not None else comp_y
                    score = contact * 5000.0 - abs(comp_cx - anchor_x) * 3.0 - max(0.0, comp_y - anchor_y) * 0.05 - band_penalty
                    if score > best_score:
                        best_score = score
                        best_name = name
                if best_name is not None:
                    assigned_regions[best_name] |= component
        else:
            ys, xs = np.nonzero(metric_mask)
            if len(xs) == 0:
                return _overlay_mask_on_image(image_rgb, mask_path, cache)
            points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
            centers = []
            boxes = []
            names = []
            for name, row, _color in track_rows:
                anchor = _choose_anchor_box(row, metric_mask.shape)
                if anchor is None:
                    continue
                center, box = anchor
                names.append(name)
                centers.append(center)
                boxes.append(box)
            if not centers:
                return _overlay_mask_on_image(image_rgb, mask_path, cache)
            centers_arr = np.asarray(centers, dtype=np.float32)
            boxes_arr = np.asarray(boxes, dtype=np.float32)
            dist2 = ((points[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2)
            inside = (
                (points[:, None, 0] >= boxes_arr[None, :, 0])
                & (points[:, None, 0] < boxes_arr[None, :, 2])
                & (points[:, None, 1] >= boxes_arr[None, :, 1])
                & (points[:, None, 1] < boxes_arr[None, :, 3])
            )
            gated_dist = np.where(inside, dist2, np.inf)
            assigned = np.argmin(np.where(np.isfinite(gated_dist).any(axis=1, keepdims=True), gated_dist, dist2), axis=1)
            for idx, name in enumerate(names):
                region_idx = assigned == idx
                if not np.any(region_idx):
                    continue
                assigned_regions[name][ys[region_idx], xs[region_idx]] = True

    alpha = 0.58
    for name, row, color in track_rows:
        region = assigned_regions.get(name)
        if region is None or not np.any(region):
            continue
        region_ys, region_xs = np.where(region)
        base_region = base[region_ys, region_xs].astype(np.float32)
        overlay_color = np.asarray(color, dtype=np.float32)
        base[region_ys, region_xs] = np.clip(base_region * (1.0 - alpha) + overlay_color * alpha, 0, 255).astype(np.uint8)
        plant_region = np.zeros(metric_mask.shape, dtype=np.uint8)
        plant_region[region_ys, region_xs] = 255
        edge = cv2.morphologyEx(plant_region, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        base[edge] = np.asarray(color, dtype=np.uint8)
        label = _pretty_track_name(name)
        anchor = _choose_shoot_box(row, metric_mask.shape) or _choose_anchor_box(row, metric_mask.shape)
        if anchor is None:
            continue
        center, box = (anchor[0:2], anchor[2]) if isinstance(anchor, tuple) and len(anchor) == 3 and isinstance(anchor[2], tuple) else anchor
        cx, cy = center
        x0, y0, x1, _ = box
        label_x = max(6, min(base.shape[1] - 80, int(round(x0))))
        label_y = max(18, min(base.shape[0] - 8, int(round(min(cy, y0) - 8))))
        cv2.putText(base, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, np.asarray(color, dtype=np.uint8).tolist(), 1, cv2.LINE_AA)
    return base


def _render_metric_plot_frame(
    df: pd.DataFrame,
    metric_col: str,
    upto_timestamp: pd.Timestamp,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 236, dtype=np.uint8)
    valid = df[df["Timestamp"] <= upto_timestamp].copy()
    if valid.empty:
        return canvas

    valid["metric"] = pd.to_numeric(valid[metric_col], errors="coerce")
    valid = valid[np.isfinite(valid["metric"])]
    valid = valid.dropna(subset=["Timestamp"])
    if valid.empty:
        return canvas

    valid = valid.sort_values("Timestamp")
    ts_values = valid["Timestamp"].drop_duplicates().sort_values()
    if ts_values.empty:
        return canvas

    margin_l = 88
    margin_t = 78
    margin_b = 92
    margin_r = 220 if width >= 1200 else 150
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    if chart_w < 220 or chart_h < 180:
        return canvas

    y_min = float(valid["metric"].min())
    y_max = float(valid["metric"].max())
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return canvas
    if y_max <= y_min:
        base = max(abs(y_min), 1.0)
        y_pad = max(1e-6, base * 0.10)
        y_min -= y_pad
        y_max += y_pad
    else:
        y_pad = max(1e-6, (y_max - y_min) * 0.08)
        y_min -= y_pad
        y_max += y_pad

    t0 = pd.Timestamp(ts_values.iloc[0])
    t1 = pd.Timestamp(ts_values.iloc[-1])
    span = max(1.0, float((t1 - t0).total_seconds()))

    def _x(ts: pd.Timestamp) -> int:
        dt = float((pd.Timestamp(ts) - t0).total_seconds())
        ratio = max(0.0, min(1.0, dt / span))
        return int(round(margin_l + ratio * chart_w))

    def _y(v: float) -> int:
        ratio = (float(v) - y_min) / max(1e-9, (y_max - y_min))
        ratio = max(0.0, min(1.0, ratio))
        return int(round(margin_t + (1.0 - ratio) * chart_h))

    title = f"{_pretty_metric_title(metric_col)} Over Time"
    subtitle = _plot_context_label(valid)
    cv2.rectangle(canvas, (margin_l, margin_t), (margin_l + chart_w, margin_t + chart_h), (190, 198, 208), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        title,
        (margin_l, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        subtitle,
        (margin_l, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (72, 72, 72),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Current time: {_format_timestamp_label(upto_timestamp)}",
        (margin_l, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (96, 96, 96),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, "Timestamp", (margin_l + chart_w // 2 - 46, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, _pretty_metric_title(metric_col), (12, margin_t + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (80, 80, 80), 1, cv2.LINE_AA)

    y_ticks = 5
    for i in range(y_ticks + 1):
        val = y_min + (y_max - y_min) * (i / y_ticks)
        py = _y(val)
        cv2.line(canvas, (margin_l, py), (margin_l + chart_w, py), (224, 228, 233), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{val:.2f}", (8, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (92, 92, 92), 1, cv2.LINE_AA)

    x_ticks = min(6, len(ts_values))
    if x_ticks >= 2:
        tick_idx = np.linspace(0, len(ts_values) - 1, x_ticks, dtype=int)
        for idx in sorted(set(int(v) for v in tick_idx.tolist())):
            ts = pd.Timestamp(ts_values.iloc[idx])
            px = _x(ts)
            cv2.line(canvas, (px, margin_t), (px, margin_t + chart_h), (234, 236, 240), 1, cv2.LINE_AA)
            cv2.line(canvas, (px, margin_t + chart_h), (px, margin_t + chart_h + 6), (140, 140, 140), 1, cv2.LINE_AA)
            label = ts.strftime("%m-%d %H:%M")
            cv2.putText(canvas, label, (px - 40, margin_t + chart_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (96, 96, 96), 1, cv2.LINE_AA)

    group_col = _metric_group_column(valid)
    groups: list[tuple[str, pd.DataFrame]] = []
    if group_col in valid.columns:
        for group_name, group_df in valid.groupby(group_col):
            subset = group_df.sort_values("Timestamp")
            if subset.empty:
                continue
            groups.append((str(group_name), subset))
    else:
        groups.append(("Total", valid.sort_values("Timestamp")))
    groups.sort(key=lambda entry: _natural_sort_key(entry[0]))

    palette = _metric_track_palette()

    legend_x = margin_l + chart_w + 14
    legend_y = margin_t + 20
    cv2.putText(canvas, "Tracks", (legend_x, legend_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (72, 72, 72), 1, cv2.LINE_AA)
    for idx, (name, subset) in enumerate(groups[:12]):
        color = palette[idx % len(palette)]
        points: list[tuple[int, int]] = []
        for _, row in subset.iterrows():
            ts = row["Timestamp"]
            val = float(row["metric"])
            if pd.isna(ts):
                continue
            points.append((_x(pd.Timestamp(ts)), _y(val)))
        if len(points) >= 2:
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
        if points:
            cv2.circle(canvas, points[-1], 3, color, -1, cv2.LINE_AA)
        ly = legend_y + idx * 18
        if ly > height - 12:
            break
        cv2.rectangle(canvas, (legend_x, ly - 8), (legend_x + 10, ly + 2), color, -1)
        cv2.putText(canvas, _pretty_track_name(name)[:24], (legend_x + 16, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (64, 64, 64), 1, cv2.LINE_AA)

    sum_series = valid.groupby("Timestamp")["metric"].sum().sort_index()
    total_points = [(_x(pd.Timestamp(ts)), _y(float(v))) for ts, v in sum_series.items()]
    if len(total_points) >= 2:
        cv2.polylines(canvas, [np.asarray(total_points, dtype=np.int32).reshape(-1, 1, 2)], False, (70, 70, 70), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Total", (legend_x, min(height - 14, legend_y + 228)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)

    px = _x(pd.Timestamp(upto_timestamp))
    cv2.line(canvas, (px, margin_t), (px, margin_t + chart_h), (90, 90, 90), 1, cv2.LINE_AA)
    return canvas


def _render_metric_plot_frame_by_frame(
    df: pd.DataFrame,
    metric_col: str,
    upto_frame_index: float,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 236, dtype=np.uint8)
    valid = df.copy()
    valid["FrameIndex"] = pd.to_numeric(valid["FrameIndex"], errors="coerce")
    valid = valid[valid["FrameIndex"].notna()]
    valid = valid[valid["FrameIndex"] <= float(upto_frame_index)]
    if valid.empty:
        return canvas

    valid["metric"] = pd.to_numeric(valid[metric_col], errors="coerce")
    valid = valid[np.isfinite(valid["metric"])]
    if valid.empty:
        return canvas

    valid = valid.sort_values("FrameIndex")
    frame_values = sorted(float(v) for v in valid["FrameIndex"].dropna().unique().tolist())
    if not frame_values:
        return canvas

    margin_l = 88
    margin_t = 78
    margin_b = 92
    margin_r = 220 if width >= 1200 else 150
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    if chart_w < 220 or chart_h < 180:
        return canvas

    y_min = float(valid["metric"].min())
    y_max = float(valid["metric"].max())
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return canvas
    if y_max <= y_min:
        base = max(abs(y_min), 1.0)
        y_pad = max(1e-6, base * 0.10)
        y_min -= y_pad
        y_max += y_pad
    else:
        y_pad = max(1e-6, (y_max - y_min) * 0.08)
        y_min -= y_pad
        y_max += y_pad

    f0 = float(frame_values[0])
    f1 = float(frame_values[-1])
    span = max(1.0, f1 - f0)

    def _x(frame_value: float) -> int:
        ratio = max(0.0, min(1.0, (float(frame_value) - f0) / span))
        return int(round(margin_l + ratio * chart_w))

    def _y(v: float) -> int:
        ratio = (float(v) - y_min) / max(1e-9, (y_max - y_min))
        ratio = max(0.0, min(1.0, ratio))
        return int(round(margin_t + (1.0 - ratio) * chart_h))

    title = f"{_pretty_metric_title(metric_col)} Over Frames"
    subtitle = _plot_context_label(valid)
    cv2.rectangle(canvas, (margin_l, margin_t), (margin_l + chart_w, margin_t + chart_h), (190, 198, 208), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (margin_l, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (margin_l, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (72, 72, 72), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Current frame: {int(round(float(upto_frame_index)))}",
        (margin_l, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (96, 96, 96),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, "Frame Index", (margin_l + chart_w // 2 - 42, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, _pretty_metric_title(metric_col), (12, margin_t + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (80, 80, 80), 1, cv2.LINE_AA)

    y_ticks = 5
    for i in range(y_ticks + 1):
        val = y_min + (y_max - y_min) * (i / y_ticks)
        py = _y(val)
        cv2.line(canvas, (margin_l, py), (margin_l + chart_w, py), (224, 228, 233), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{val:.2f}", (8, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (92, 92, 92), 1, cv2.LINE_AA)

    x_ticks = min(6, len(frame_values))
    if x_ticks >= 2:
        tick_idx = np.linspace(0, len(frame_values) - 1, x_ticks, dtype=int)
        for idx in sorted(set(int(v) for v in tick_idx.tolist())):
            frame_value = float(frame_values[idx])
            px = _x(frame_value)
            cv2.line(canvas, (px, margin_t), (px, margin_t + chart_h), (234, 236, 240), 1, cv2.LINE_AA)
            cv2.line(canvas, (px, margin_t + chart_h), (px, margin_t + chart_h + 6), (140, 140, 140), 1, cv2.LINE_AA)
            cv2.putText(canvas, str(int(round(frame_value))), (px - 18, margin_t + chart_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (96, 96, 96), 1, cv2.LINE_AA)

    group_col = _metric_group_column(valid)
    groups: list[tuple[str, pd.DataFrame]] = []
    if group_col in valid.columns:
        for group_name, group_df in valid.groupby(group_col):
            subset = group_df.sort_values("FrameIndex")
            if subset.empty:
                continue
            groups.append((str(group_name), subset))
    else:
        groups.append(("Total", valid.sort_values("FrameIndex")))
    groups.sort(key=lambda entry: _natural_sort_key(entry[0]))

    palette = _metric_track_palette()

    legend_x = margin_l + chart_w + 14
    legend_y = margin_t + 20
    cv2.putText(canvas, "Tracks", (legend_x, legend_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (72, 72, 72), 1, cv2.LINE_AA)
    for idx, (name, subset) in enumerate(groups[:12]):
        color = palette[idx % len(palette)]
        points: list[tuple[int, int]] = []
        for _, row in subset.iterrows():
            frame_value = row["FrameIndex"]
            val = float(row["metric"])
            if pd.isna(frame_value):
                continue
            points.append((_x(float(frame_value)), _y(val)))
        if len(points) >= 2:
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
        if points:
            cv2.circle(canvas, points[-1], 3, color, -1, cv2.LINE_AA)
        ly = legend_y + idx * 18
        if ly > height - 12:
            break
        cv2.rectangle(canvas, (legend_x, ly - 8), (legend_x + 10, ly + 2), color, -1)
        cv2.putText(canvas, _pretty_track_name(name)[:24], (legend_x + 16, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (64, 64, 64), 1, cv2.LINE_AA)

    sum_series = valid.groupby("FrameIndex")["metric"].sum().sort_index()
    total_points = [(_x(float(frame_idx)), _y(float(v))) for frame_idx, v in sum_series.items()]
    if len(total_points) >= 2:
        cv2.polylines(canvas, [np.asarray(total_points, dtype=np.int32).reshape(-1, 1, 2)], False, (70, 70, 70), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Total", (legend_x, min(height - 14, legend_y + 228)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)

    px = _x(float(upto_frame_index))
    cv2.line(canvas, (px, margin_t), (px, margin_t + chart_h), (90, 90, 90), 1, cv2.LINE_AA)
    return canvas


def _load_preview_image_rgb(path_text: str | None, cache: dict[str, np.ndarray | None]) -> np.ndarray | None:
    if not path_text:
        return None
    key = str(path_text)
    if key in cache:
        return cache[key]
    path = Path(key)
    if not path.exists():
        cache[key] = None
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        cache[key] = None
        return None
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.ndim == 3 and image.shape[2] >= 3:
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        cache[key] = None
        return None
    if rgb.dtype != np.uint8:
        rgb = cv2.normalize(rgb.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cache[key] = rgb
    return rgb


def _fit_rgb_image(image_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    if image_rgb is None or image_rgb.size == 0:
        return canvas
    working = np.asarray(image_rgb, dtype=np.uint8)
    if working.ndim != 3 or working.shape[2] < 3:
        return canvas
    gray = cv2.cvtColor(working[:, :, :3], cv2.COLOR_RGB2GRAY)
    non_dark = gray > 12
    if np.any(non_dark):
        ys, xs = np.where(non_dark)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        crop_w = max(1, x1 - x0 + 1)
        crop_h = max(1, y1 - y0 + 1)
        cover = (crop_w * crop_h) / float(max(1, working.shape[0] * working.shape[1]))
        if 0.18 <= cover < 0.99:
            pad_x = max(6, int(round(crop_w * 0.03)))
            pad_y = max(6, int(round(crop_h * 0.03)))
            x0 = max(0, x0 - pad_x)
            y0 = max(0, y0 - pad_y)
            x1 = min(working.shape[1] - 1, x1 + pad_x)
            y1 = min(working.shape[0] - 1, y1 + pad_y)
            working = working[y0 : y1 + 1, x0 : x1 + 1]

    ih, iw = working.shape[:2]
    if ih <= 0 or iw <= 0:
        return canvas
    scale = min(width / float(max(1, iw)), height / float(max(1, ih)))
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    inter = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(working, (nw, nh), interpolation=inter)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _load_index_mask(path_text: str | None, cache: dict[str, np.ndarray | None]) -> np.ndarray | None:
    if not path_text:
        return None
    key = f"mask::{path_text}"
    if key in cache:
        return cache[key]
    path = Path(str(path_text))
    if not path.exists():
        cache[key] = None
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        cache[key] = None
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = np.asarray(mask, dtype=np.uint8)
    cache[key] = mask
    return mask


def _overlay_mask_on_image(image_rgb: np.ndarray | None, mask_path: str | None, cache: dict[str, np.ndarray | None]) -> np.ndarray | None:
    if image_rgb is None:
        return None
    if not mask_path:
        return image_rgb
    mask = _load_index_mask(mask_path, cache)
    if mask is None or mask.size == 0:
        return image_rgb

    base = np.asarray(image_rgb, dtype=np.uint8).copy()
    if mask.shape[:2] != base.shape[:2]:
        mask = cv2.resize(mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)

    class_colors = {
        1: (255, 90, 138),
        2: (86, 243, 154),
        3: (255, 209, 102),
        4: (63, 193, 255),
        5: (255, 159, 28),
    }
    alpha = 0.52
    for class_id, color in class_colors.items():
        region = mask == int(class_id)
        if not np.any(region):
            continue
        overlay_color = np.asarray(color, dtype=np.float32)
        base_region = base[region].astype(np.float32)
        base[region] = np.clip(base_region * (1.0 - alpha) + overlay_color * alpha, 0, 255).astype(np.uint8)
        edge = cv2.morphologyEx(region.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        base[edge] = np.asarray(color, dtype=np.uint8)
    return base


def _mask_path_for_timestamp(df: pd.DataFrame, timestamp: pd.Timestamp) -> str | None:
    if "MaskPath" not in df.columns:
        if "output_mask" in df.columns:
            subset = df[["Timestamp", "output_mask"]].copy()
            column = "output_mask"
        else:
            return None
    else:
        subset = df[["Timestamp", "MaskPath"]].copy()
        column = "MaskPath"
    subset["Timestamp"] = pd.to_datetime(subset["Timestamp"], errors="coerce")
    subset = subset.dropna(subset=["Timestamp"])
    if subset.empty:
        return None
    exact = subset[subset["Timestamp"] == pd.Timestamp(timestamp)]
    if exact.empty:
        exact = subset[subset["Timestamp"] <= pd.Timestamp(timestamp)].sort_values("Timestamp")
        if exact.empty:
            return None
        value = exact.iloc[-1][column]
    else:
        value = exact.iloc[0][column]
    text = str(value).strip()
    return text if text else None


def _mask_path_for_frame(df: pd.DataFrame, frame_index: float) -> str | None:
    target = int(round(float(frame_index)))
    column = "MaskPath" if "MaskPath" in df.columns else ("output_mask" if "output_mask" in df.columns else None)
    if column is None or "FrameIndex" not in df.columns:
        return None
    subset = df[["FrameIndex", column]].copy()
    subset["FrameIndex"] = pd.to_numeric(subset["FrameIndex"], errors="coerce")
    subset = subset.dropna(subset=["FrameIndex"])
    if subset.empty:
        return None
    exact = subset[subset["FrameIndex"].round().astype(int) == target]
    if exact.empty:
        exact = subset[subset["FrameIndex"] <= float(frame_index)].sort_values("FrameIndex")
        if exact.empty:
            return None
        value = exact.iloc[-1][column]
    else:
        value = exact.iloc[0][column]
    text = str(value).strip()
    return text if text else None


def _render_preview_panel(
    image_rgb: np.ndarray | None,
    width: int,
    height: int,
    timestamp: pd.Timestamp,
    source_name: str,
) -> np.ndarray:
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    header_h = 54
    cv2.putText(panel, "Segmented Plate Preview", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (42, 42, 42), 1, cv2.LINE_AA)
    cv2.putText(panel, _format_timestamp_label(timestamp), (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (92, 92, 92), 1, cv2.LINE_AA)

    body_h = max(10, height - header_h - 14)
    body_w = max(10, width - 12)
    fitted = _fit_rgb_image(image_rgb, body_w, body_h)
    panel[header_h + 7 : header_h + 7 + body_h, 6 : 6 + body_w] = fitted
    cv2.rectangle(panel, (6, header_h + 7), (6 + body_w - 1, header_h + 7 + body_h - 1), (190, 198, 208), 1, cv2.LINE_AA)

    if image_rgb is None:
        cv2.putText(panel, "No preview image found", (18, header_h + body_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 72, 72), 1, cv2.LINE_AA)
    elif source_name:
        label = source_name[:48]
        cv2.putText(panel, label, (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (96, 96, 96), 1, cv2.LINE_AA)
    return panel


def _render_preview_panel_for_frame(
    image_rgb: np.ndarray | None,
    width: int,
    height: int,
    frame_index: float,
    source_name: str,
) -> np.ndarray:
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    header_h = 54
    cv2.putText(panel, "Segmented Plate Preview", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (42, 42, 42), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Frame {int(round(float(frame_index)))}", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (92, 92, 92), 1, cv2.LINE_AA)

    body_h = max(10, height - header_h - 14)
    body_w = max(10, width - 12)
    fitted = _fit_rgb_image(image_rgb, body_w, body_h)
    panel[header_h + 7 : header_h + 7 + body_h, 6 : 6 + body_w] = fitted
    cv2.rectangle(panel, (6, header_h + 7), (6 + body_w - 1, header_h + 7 + body_h - 1), (190, 198, 208), 1, cv2.LINE_AA)

    if image_rgb is None:
        cv2.putText(panel, "No preview image found", (18, header_h + body_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 72, 72), 1, cv2.LINE_AA)
    elif source_name:
        label = source_name[:48]
        cv2.putText(panel, label, (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (96, 96, 96), 1, cv2.LINE_AA)
    return panel


def _build_preview_lookup(df: pd.DataFrame) -> dict[int, str]:
    if "PreviewImagePath" not in df.columns:
        return {}
    subset = df[["Timestamp", "PreviewImagePath"]].copy()
    subset["Timestamp"] = pd.to_datetime(subset["Timestamp"], errors="coerce")
    subset = subset.dropna(subset=["Timestamp"])
    if subset.empty:
        return {}
    subset = subset.sort_values("Timestamp")
    out: dict[int, str] = {}
    for _, row in subset.iterrows():
        ts = pd.Timestamp(row["Timestamp"])
        path_text = str(row["PreviewImagePath"]).strip()
        if not path_text:
            continue
        path = Path(path_text)
        if not path.exists():
            continue
        key = _timestamp_key(ts)
        if key is not None:
            out[key] = str(path.resolve())
    return out


def _build_preview_lookup_by_frame(df: pd.DataFrame) -> dict[int, str]:
    if "PreviewImagePath" not in df.columns or "FrameIndex" not in df.columns:
        return {}
    subset = df[["FrameIndex", "PreviewImagePath"]].copy()
    subset["FrameIndex"] = pd.to_numeric(subset["FrameIndex"], errors="coerce")
    subset = subset.dropna(subset=["FrameIndex"])
    if subset.empty:
        return {}
    subset = subset.sort_values("FrameIndex")
    out: dict[int, str] = {}
    for _, row in subset.iterrows():
        frame_idx = int(round(float(row["FrameIndex"])))
        path_text = str(row["PreviewImagePath"]).strip()
        if not path_text:
            continue
        path = Path(path_text)
        if not path.exists():
            continue
        out[frame_idx] = str(path.resolve())
    return out


def _select_preview_at_or_before(lookup: dict[int, str], timestamp: pd.Timestamp, current_path: str | None) -> str | None:
    key = _timestamp_key(timestamp)
    if key is None:
        return current_path
    if key in lookup:
        return lookup[key]
    prior = [k for k in lookup.keys() if k <= key]
    if not prior:
        return current_path
    return lookup[max(prior)]


def _select_preview_at_or_before_frame(lookup: dict[int, str], frame_index: float, current_path: str | None) -> str | None:
    key = int(round(float(frame_index)))
    if key in lookup:
        return lookup[key]
    prior = [k for k in lookup.keys() if int(k) <= key]
    if not prior:
        return current_path
    return lookup[max(prior)]


def _build_metric_visual_frame(
    df: pd.DataFrame,
    metric_col: str,
    timestamp: pd.Timestamp,
    width: int,
    height: int,
    preview_path: str | None,
    image_cache: dict[str, np.ndarray | None],
) -> np.ndarray:
    if not preview_path:
        return _render_metric_plot_frame(df, metric_col=metric_col, upto_timestamp=timestamp, width=width, height=height)

    image_w = int(round(width * 0.50))
    image_w = max(320, min(image_w, width - 360))
    plot_w = max(320, width - image_w)
    source_name = Path(preview_path).name
    preview_rgb = _load_preview_image_rgb(preview_path, image_cache)
    mask_path = _mask_path_for_timestamp(df, timestamp)
    current_rows = _metric_rows_for_timestamp(df, timestamp)
    preview_rgb = _overlay_metric_decomposition_on_image(preview_rgb, mask_path, current_rows, metric_col, image_cache)
    preview_panel = _render_preview_panel(
        preview_rgb,
        width=image_w,
        height=height,
        timestamp=timestamp,
        source_name=source_name,
    )
    plot_panel = _render_metric_plot_frame(df, metric_col=metric_col, upto_timestamp=timestamp, width=plot_w, height=height)
    return np.concatenate([preview_panel, plot_panel], axis=1)


def _build_metric_visual_frame_by_frame(
    df: pd.DataFrame,
    metric_col: str,
    frame_index: float,
    width: int,
    height: int,
    preview_path: str | None,
    image_cache: dict[str, np.ndarray | None],
) -> np.ndarray:
    if not preview_path:
        return _render_metric_plot_frame_by_frame(df, metric_col=metric_col, upto_frame_index=frame_index, width=width, height=height)

    image_w = int(round(width * 0.50))
    image_w = max(320, min(image_w, width - 360))
    plot_w = max(320, width - image_w)
    source_name = Path(preview_path).name
    preview_rgb = _load_preview_image_rgb(preview_path, image_cache)
    mask_path = _mask_path_for_frame(df, frame_index)
    current_rows = _metric_rows_for_frame(df, frame_index)
    preview_rgb = _overlay_metric_decomposition_on_image(preview_rgb, mask_path, current_rows, metric_col, image_cache)
    panel = _render_preview_panel_for_frame(
        preview_rgb,
        width=image_w,
        height=height,
        frame_index=frame_index,
        source_name=source_name,
    )

    plot_panel = _render_metric_plot_frame_by_frame(df, metric_col=metric_col, upto_frame_index=frame_index, width=plot_w, height=height)
    return np.concatenate([panel, plot_panel], axis=1)


def generate_metric_timelapse(
    consolidated_df: pd.DataFrame,
    metric_col: str,
    output_path: Path,
    config: ConsolidationConfig | None = None,
) -> Path:
    cfg = config or ConsolidationConfig()
    if consolidated_df.empty:
        raise ValueError("Consolidated dataframe is empty.")
    if metric_col not in consolidated_df.columns:
        raise ValueError(f"Metric column not found: {metric_col}")
    df = consolidated_df.copy()
    use_timestamps = _has_valid_timestamps(df)
    use_frames = _has_valid_frame_index(df)
    if not use_timestamps and not use_frames:
        raise ValueError("No valid timestamps or frame indices available after parsing.")

    image_cache: dict[str, np.ndarray | None] = {}
    active_preview: str | None = None
    writer = open_mp4_video_writer(output_path=Path(output_path), width=int(cfg.width), height=int(cfg.height), fps=max(1, int(cfg.fps)))
    try:
        if use_timestamps:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = df.dropna(subset=["Timestamp"])
            if df.empty:
                raise ValueError("No valid timestamps available after parsing.")
            df = df.sort_values("Timestamp")
            timestamps = [pd.Timestamp(ts) for ts in df["Timestamp"].drop_duplicates().sort_values().tolist()]
            if not timestamps:
                raise ValueError("No timestamps available for timelapse.")
            preview_lookup = _build_preview_lookup(df)
            for ts in timestamps:
                active_preview = _select_preview_at_or_before(preview_lookup, ts, active_preview)
                frame = _build_metric_visual_frame(
                    df,
                    metric_col=metric_col,
                    timestamp=ts,
                    width=int(cfg.width),
                    height=int(cfg.height),
                    preview_path=active_preview,
                    image_cache=image_cache,
                )
                writer.append(frame)
        else:
            df["FrameIndex"] = pd.to_numeric(df["FrameIndex"], errors="coerce")
            df = df.dropna(subset=["FrameIndex"])
            if df.empty:
                raise ValueError("No frame indices available for timelapse.")
            df = df.sort_values("FrameIndex")
            frame_values = [float(v) for v in df["FrameIndex"].drop_duplicates().sort_values().tolist()]
            if not frame_values:
                raise ValueError("No frame indices available for timelapse.")
            preview_lookup = _build_preview_lookup_by_frame(df)
            for frame_idx in frame_values:
                active_preview = _select_preview_at_or_before_frame(preview_lookup, frame_idx, active_preview)
                frame = _build_metric_visual_frame_by_frame(
                    df,
                    metric_col=metric_col,
                    frame_index=frame_idx,
                    width=int(cfg.width),
                    height=int(cfg.height),
                    preview_path=active_preview,
                    image_cache=image_cache,
                )
                writer.append(frame)
    finally:
        writer.close()
    return Path(output_path)


def build_metric_plot_preview(
    consolidated_df: pd.DataFrame,
    metric_col: str,
    timestamp: pd.Timestamp | None = None,
    width: int = 1280,
    height: int = 720,
) -> np.ndarray:
    def _placeholder(message: str) -> np.ndarray:
        canvas = np.full((max(10, int(height)), max(10, int(width)), 3), 236, dtype=np.uint8)
        cv2.rectangle(canvas, (10, 10), (canvas.shape[1] - 11, canvas.shape[0] - 11), (182, 190, 202), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Consolidated Measurements Preview", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (48, 56, 72), 1, cv2.LINE_AA)
        cv2.putText(canvas, message, (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (96, 104, 118), 1, cv2.LINE_AA)
        return canvas

    if consolidated_df.empty or metric_col not in consolidated_df.columns:
        return _placeholder("No valid metric data is available for preview yet.")

    df = consolidated_df.copy()
    if _has_valid_timestamps(df):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])
        if df.empty:
            return _placeholder("Timestamps could not be parsed from the consolidated table.")
        if timestamp is None:
            timestamp = pd.Timestamp(df["Timestamp"].max())
        else:
            timestamp = pd.Timestamp(timestamp)

        preview_lookup = _build_preview_lookup(df)
        preview_path = _select_preview_at_or_before(preview_lookup, timestamp, None)
        return _build_metric_visual_frame(
            df=df,
            metric_col=metric_col,
            timestamp=timestamp,
            width=int(width),
            height=int(height),
            preview_path=preview_path,
            image_cache={},
        )

    if _has_valid_frame_index(df):
        df["FrameIndex"] = pd.to_numeric(df["FrameIndex"], errors="coerce")
        df = df.dropna(subset=["FrameIndex"])
        if df.empty:
            return _placeholder("Frame indices could not be parsed from the consolidated table.")
        frame_index = float(df["FrameIndex"].max())
        preview_lookup = _build_preview_lookup_by_frame(df)
        preview_path = _select_preview_at_or_before_frame(preview_lookup, frame_index, None)
        return _build_metric_visual_frame_by_frame(
            df=df,
            metric_col=metric_col,
            frame_index=frame_index,
            width=int(width),
            height=int(height),
            preview_path=preview_path,
            image_cache={},
        )

    return _placeholder("No valid timestamps or frame indices are available for preview yet.")


def export_consolidated_measurements(df: pd.DataFrame, output_path: Path) -> Path:
    path = Path(output_path)
    if path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=False)
    else:
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        df.to_csv(path, index=False)
    return path
