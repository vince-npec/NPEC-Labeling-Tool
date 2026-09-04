from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tarfile

import cv2
import numpy as np
from PIL import Image


@dataclass(slots=True)
class OpticalCorrectionConfig:
    scale_x: float = 1.0
    scale_y: float = 1.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    rotation_deg: float = 0.0
    alpha: float = 0.45
    radial_k1: float = 0.0
    radial_k2: float = 0.0
    perspective_matrix: np.ndarray | None = None
    clip_min_percentile: float = 2.0
    clip_max_percentile: float = 98.0


OPTICAL_PROFILE_KEYS = {
    "scale_x",
    "scale_y",
    "shift_x",
    "shift_y",
    "rotation_deg",
    "alpha",
    "radial_k1",
    "radial_k2",
    "perspective_matrix",
    "clip_min_percentile",
    "clip_max_percentile",
}


def _resolve_existing_pair(path: Path, target_suffix: str) -> Path | None:
    preferred = path.with_suffix(target_suffix)
    if preferred.exists():
        return preferred
    stem = path.stem
    parent = path.parent
    wanted = target_suffix.lower()
    for candidate in parent.iterdir():
        if not candidate.is_file():
            continue
        if candidate.stem != stem:
            continue
        if candidate.suffix.lower() == wanted:
            return candidate
    return None


def _parse_hdr_text(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    meta: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(";"):
            continue
        key = ""
        value = ""
        if "=" in line:
            key, value = line.split("=", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) == 2:
                key, value = parts
            else:
                continue
        key_norm = key.strip().lower().replace(" ", "").replace("_", "")
        value_norm = value.strip().strip("{}")
        if key_norm:
            meta[key_norm] = value_norm
    return meta


def _int_meta(meta: dict[str, str], *keys: str, default: int = 0) -> int:
    for key in keys:
        value = meta.get(key.lower().replace(" ", "").replace("_", ""))
        if value is None:
            continue
        token = str(value).strip()
        if not token:
            continue
        try:
            return int(float(token))
        except Exception:
            continue
    return int(default)


def _dtype_from_meta(meta: dict[str, str]) -> tuple[np.dtype, int]:
    data_type = _int_meta(meta, "datatype", default=0)
    nbits = _int_meta(meta, "nbits", default=0)
    byteorder_raw = meta.get("byteorder", meta.get("byteorder", "i"))
    byteorder = str(byteorder_raw).strip().lower() if byteorder_raw is not None else "i"
    little = byteorder in {"i", "intel", "0", "little", "littleendian", "lsb"}
    endian = "<" if little else ">"

    if data_type:
        table: dict[int, str] = {
            1: "u1",
            2: "i2",
            3: "i4",
            4: "f4",
            5: "f8",
            12: "u2",
            13: "u4",
            14: "i8",
            15: "u8",
        }
        base = table.get(int(data_type))
        if base is not None:
            dtype = np.dtype(base if base.endswith("1") else f"{endian}{base[1:]}")
            return dtype, int(dtype.itemsize)

    if nbits <= 8:
        return np.dtype("u1"), 1
    if nbits <= 16:
        return np.dtype(f"{endian}u2"), 2
    if nbits <= 32:
        return np.dtype(f"{endian}f4"), 4
    return np.dtype(f"{endian}f8"), 8


def _resolve_envi_hdr_path(path: Path) -> Path:
    src = Path(path)
    suffix = src.suffix.lower()
    if suffix == ".hdr":
        return src
    if suffix == ".bil":
        hdr = _resolve_existing_pair(src, ".hdr")
        if hdr is not None and hdr.exists():
            return hdr
    raise FileNotFoundError(f"HDR pair not found for: {src}")


def load_envi_wavelengths(path: Path) -> list[float]:
    hdr_path = _resolve_envi_hdr_path(path)
    text = hdr_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"wavelength\s*=\s*\{(.*?)\}", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    tokens = re.split(r"[\s,]+", match.group(1).strip())
    values: list[float] = []
    for token in tokens:
        if not token:
            continue
        try:
            values.append(float(token))
        except Exception:
            continue
    return values


def load_envi_cube(path: Path) -> np.ndarray:
    src = Path(path)
    suffix = src.suffix.lower()
    if suffix not in {".hdr", ".bil"}:
        raise ValueError(f"Unsupported ENVI file suffix: {suffix}")

    hdr_path: Path | None = src if suffix == ".hdr" else _resolve_existing_pair(src, ".hdr")
    bil_path: Path | None = src if suffix == ".bil" else _resolve_existing_pair(src, ".bil")
    if hdr_path is None or not hdr_path.exists():
        raise FileNotFoundError(f"HDR pair not found for: {src}")
    if bil_path is None or not bil_path.exists():
        raise FileNotFoundError(f"BIL pair not found for: {src}")

    meta = _parse_hdr_text(hdr_path)
    rows = _int_meta(meta, "nrows", "lines", default=0)
    cols = _int_meta(meta, "ncols", "samples", default=0)
    bands = _int_meta(meta, "nbands", "bands", default=1)
    if rows <= 0 or cols <= 0 or bands <= 0:
        raise ValueError(f"Invalid ENVI dimensions in {hdr_path.name}: rows={rows}, cols={cols}, bands={bands}")

    layout = str(meta.get("layout", meta.get("interleave", "bil"))).strip().lower()
    if layout not in {"bil", "bsq", "bip"}:
        layout = "bil"

    dtype, bytes_per_sample = _dtype_from_meta(meta)
    raw = np.fromfile(bil_path, dtype=dtype)

    expected = rows * cols * bands
    band_row_bytes = _int_meta(meta, "bandrowbytes", default=cols * bytes_per_sample)
    total_row_bytes = _int_meta(meta, "totalrowbytes", default=bands * band_row_bytes)
    row_stride_samples = max(1, total_row_bytes // max(1, bytes_per_sample))
    effective_row_samples = cols * bands

    cube: np.ndarray
    if raw.size >= rows * row_stride_samples and row_stride_samples >= effective_row_samples and layout == "bil":
        trimmed = raw[: rows * row_stride_samples].reshape(rows, row_stride_samples)
        core = trimmed[:, :effective_row_samples].reshape(rows, bands, cols)
        cube = np.transpose(core, (0, 2, 1))
    else:
        if raw.size < expected:
            raise ValueError(
                f"ENVI data size mismatch for {bil_path.name}: expected at least {expected} samples, got {raw.size}"
            )
        arr = raw[:expected]
        if layout == "bil":
            cube = np.transpose(arr.reshape(rows, bands, cols), (0, 2, 1))
        elif layout == "bsq":
            cube = np.transpose(arr.reshape(bands, rows, cols), (1, 2, 0))
        else:
            cube = arr.reshape(rows, cols, bands)

    return np.asarray(cube)


def load_envi_preview_rgb(path: Path, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    src = Path(path)
    suffix = src.suffix.lower()
    if suffix not in {".hdr", ".bil"}:
        raise ValueError(f"Unsupported ENVI file suffix: {suffix}")

    hdr_path: Path | None = src if suffix == ".hdr" else _resolve_existing_pair(src, ".hdr")
    bil_path: Path | None = src if suffix == ".bil" else _resolve_existing_pair(src, ".bil")
    if hdr_path is None or not hdr_path.exists():
        raise FileNotFoundError(f"HDR pair not found for: {src}")
    if bil_path is None or not bil_path.exists():
        raise FileNotFoundError(f"BIL pair not found for: {src}")

    meta = _parse_hdr_text(hdr_path)
    rows = _int_meta(meta, "nrows", "lines", default=0)
    cols = _int_meta(meta, "ncols", "samples", default=0)
    bands = _int_meta(meta, "nbands", "bands", default=1)
    if rows <= 0 or cols <= 0 or bands <= 0:
        raise ValueError(f"Invalid ENVI dimensions in {hdr_path.name}: rows={rows}, cols={cols}, bands={bands}")

    layout = str(meta.get("layout", meta.get("interleave", "bil"))).strip().lower()
    if layout not in {"bil", "bsq", "bip"}:
        layout = "bil"

    dtype, bytes_per_sample = _dtype_from_meta(meta)
    expected = rows * cols * bands
    band_row_bytes = _int_meta(meta, "bandrowbytes", default=cols * bytes_per_sample)
    total_row_bytes = _int_meta(meta, "totalrowbytes", default=bands * band_row_bytes)
    row_stride_samples = max(1, total_row_bytes // max(1, bytes_per_sample))
    effective_row_samples = cols * bands

    if bands >= 3:
        band_ids = [max(0, min(bands - 1, int(round((bands - 1) * ratio)))) for ratio in (0.20, 0.50, 0.80)]
    else:
        band_ids = [0]

    mm = np.memmap(bil_path, dtype=dtype, mode="r")

    planes: list[np.ndarray] = []

    if layout == "bil" and mm.size >= rows * row_stride_samples and row_stride_samples >= effective_row_samples:
        view = mm[: rows * row_stride_samples].reshape(rows, row_stride_samples)
        for band in band_ids:
            start = band * cols
            stop = start + cols
            plane = np.asarray(view[:, start:stop], dtype=np.float32)
            planes.append(plane)
    elif layout == "bsq" and mm.size >= expected:
        arr = mm[:expected]
        band_plane_size = rows * cols
        for band in band_ids:
            start = band * band_plane_size
            stop = start + band_plane_size
            plane = np.asarray(arr[start:stop].reshape(rows, cols), dtype=np.float32)
            planes.append(plane)
    elif layout == "bip" and mm.size >= expected:
        arr = mm[:expected].reshape(rows, cols, bands)
        for band in band_ids:
            planes.append(np.asarray(arr[:, :, band], dtype=np.float32))
    else:
        cube = load_envi_cube(path).astype(np.float32)
        return hyperspectral_to_rgb(cube, band_index=max(0, bands // 2 - 1), pmin=pmin, pmax=pmax)

    if not planes:
        cube = load_envi_cube(path).astype(np.float32)
        return hyperspectral_to_rgb(cube, band_index=max(0, bands // 2 - 1), pmin=pmin, pmax=pmax)

    if len(planes) == 1:
        gray = normalize_to_uint8(planes[0], pmin=pmin, pmax=pmax)
        return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[:, :, ::-1]

    channels = [normalize_to_uint8(plane, pmin=pmin, pmax=pmax) for plane in planes[:3]]
    rgb = np.stack(channels, axis=2)
    return np.asarray(rgb, dtype=np.uint8)


def load_fluorcam_tar_preview(path: Path) -> np.ndarray:
    source = Path(path)
    with tarfile.open(source, "r") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        dumm = next((m for m in members if m.name.lower().endswith(".dumm")), None)
        if dumm is None:
            raise ValueError("No .dumm frame found inside FluorCam TAR.")

        payload = tf.extractfile(dumm).read()
        if payload is None:
            raise ValueError("Failed to read .dumm payload from TAR.")
        if len(payload) >= 18 and (len(payload) - 16) % 2 == 0:
            payload = payload[16:]
        elif len(payload) % 2 != 0:
            raise ValueError("Unexpected odd byte length in .dumm payload.")

        values = np.frombuffer(payload, dtype="<u2")
        if values.size == 0:
            raise ValueError("Empty .dumm payload.")

        width = 0
        height = 0
        res_member = next((m for m in members if m.name.lower().endswith("resolution-list.json")), None)
        if res_member is not None:
            try:
                raw = tf.extractfile(res_member).read()
                txt = raw.decode("utf-8", errors="ignore")
                parsed = json.loads(txt)
                if isinstance(parsed, list):
                    for entry in parsed:
                        if not isinstance(entry, dict):
                            continue
                        w = int(entry.get("Width", 0))
                        h = int(entry.get("Height", 0))
                        if w > 0 and h > 0 and (w * h) == int(values.size):
                            width = w
                            height = h
                            break
            except Exception:
                pass

        if width <= 0 or height <= 0:
            n = int(values.size)
            side = int(np.sqrt(n))
            if side > 0 and (n % side) == 0:
                height = side
                width = n // side
            else:
                raise ValueError("Could not infer image dimensions from FluorCam TAR.")

        frame = values.reshape(height, width).astype(np.float32)
    return frame


def discover_optical_profile_roots(extra_roots: Sequence[Path] | None = None) -> list[Path]:
    roots: list[Path] = []

    env_value = os.environ.get("NPEC_OPTICAL_PROFILES_DIR", "").strip()
    if env_value:
        for token in env_value.split(os.pathsep):
            token = token.strip()
            if not token:
                continue
            roots.append(Path(token).expanduser())

    env_hvir = os.environ.get("NPEC_HVIR_PIPELINES_DIR", "").strip()
    if env_hvir:
        roots.append(Path(env_hvir).expanduser())

    roots.extend(
        [
            Path.home() / "Documents" / "Apps" / "NPEC Labels" / "HVIR-FC" / "Pipelines",
        ]
    )

    if extra_roots is not None:
        roots.extend(Path(p).expanduser() for p in extra_roots)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in roots:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists() and resolved.is_dir():
            unique.append(resolved)
    return unique


def _json_looks_like_optical_profile(raw: object) -> bool:
    if not isinstance(raw, dict):
        return False
    keys = {str(k).strip().lower() for k in raw.keys()}
    return bool(keys.intersection(OPTICAL_PROFILE_KEYS))


def discover_optical_correction_profiles(search_roots: Sequence[Path] | None = None) -> list[Path]:
    roots = discover_optical_profile_roots(search_roots)
    profiles: list[Path] = []
    for root in roots:
        for candidate in sorted(root.rglob("*.json")):
            if not candidate.is_file():
                continue
            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
            except Exception:
                continue
            if _json_looks_like_optical_profile(raw):
                profiles.append(candidate)
    return profiles


def create_optical_correction_template(
    directory: Path,
    file_name: str = "optical_correction_template.json",
    overwrite: bool = False,
) -> Path:
    out_dir = Path(directory).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / file_name

    if out_path.exists() and not overwrite:
        return out_path

    payload: dict[str, object] = {
        "_description": "NPEC optical correction profile template",
        "scale_x": 1.0,
        "scale_y": 1.0,
        "shift_x": 0.0,
        "shift_y": 0.0,
        "rotation_deg": 0.0,
        "alpha": 0.45,
        "radial_k1": 0.0,
        "radial_k2": 0.0,
        "perspective_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "clip_min_percentile": 2.0,
        "clip_max_percentile": 98.0,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_optical_correction_config(path: Path | None) -> OpticalCorrectionConfig:
    if path is None:
        return OpticalCorrectionConfig()
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Optical correction JSON not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Optical correction JSON must contain an object.")

    perspective = raw.get("perspective_matrix")
    perspective_matrix: np.ndarray | None = None
    if perspective is not None:
        arr = np.asarray(perspective, dtype=np.float32)
        if arr.size == 9:
            perspective_matrix = arr.reshape(3, 3)
        elif arr.shape == (3, 3):
            perspective_matrix = arr.astype(np.float32)
        else:
            raise ValueError("perspective_matrix must be a 3x3 matrix or flat list of 9 values.")

    return OpticalCorrectionConfig(
        scale_x=float(raw.get("scale_x", 1.0)),
        scale_y=float(raw.get("scale_y", 1.0)),
        shift_x=float(raw.get("shift_x", 0.0)),
        shift_y=float(raw.get("shift_y", 0.0)),
        rotation_deg=float(raw.get("rotation_deg", 0.0)),
        alpha=float(raw.get("alpha", 0.45)),
        radial_k1=float(raw.get("radial_k1", 0.0)),
        radial_k2=float(raw.get("radial_k2", 0.0)),
        perspective_matrix=perspective_matrix,
        clip_min_percentile=float(raw.get("clip_min_percentile", 2.0)),
        clip_max_percentile=float(raw.get("clip_max_percentile", 98.0)),
    )


def load_sensor_data(path: Path) -> np.ndarray:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Sensor file not found: {source}")

    suffix = source.suffix.lower()
    if suffix in {".hdr", ".bil"}:
        return load_envi_cube(source)
    if suffix == ".tar":
        return load_fluorcam_tar_preview(source)
    if suffix == ".npy":
        arr = np.load(source)
        return np.asarray(arr)
    if suffix == ".npz":
        payload = np.load(source)
        if len(payload.files) == 0:
            raise ValueError("NPZ file is empty.")
        key = payload.files[0]
        return np.asarray(payload[key])

    with Image.open(source) as pil:
        arr = np.asarray(pil)
    return np.asarray(arr)


def normalize_to_uint8(array: np.ndarray, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    lo = np.percentile(arr, max(0.0, min(100.0, pmin)))
    hi = np.percentile(arr, max(0.0, min(100.0, pmax)))
    if hi <= lo:
        hi = lo + 1.0
    scaled = (arr - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def hyperspectral_to_rgb(sensor_cube: np.ndarray, band_index: int = 0, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    cube = np.asarray(sensor_cube)
    if cube.ndim == 2:
        gray = normalize_to_uint8(cube, pmin=pmin, pmax=pmax)
        return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[:, :, ::-1]
    if cube.ndim == 3 and cube.shape[2] == 1:
        gray = normalize_to_uint8(cube[:, :, 0], pmin=pmin, pmax=pmax)
        return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[:, :, ::-1]
    if cube.ndim == 3 and cube.shape[2] == 3:
        rgb = cube.astype(np.float32)
        out = np.zeros_like(rgb, dtype=np.uint8)
        for c in range(3):
            out[:, :, c] = normalize_to_uint8(rgb[:, :, c], pmin=pmin, pmax=pmax)
        return out
    if cube.ndim == 3 and cube.shape[2] > 3:
        band = int(max(0, min(cube.shape[2] - 1, band_index)))
        gray = normalize_to_uint8(cube[:, :, band], pmin=pmin, pmax=pmax)
        return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[:, :, ::-1]
    raise ValueError(f"Unsupported hyperspectral array shape: {cube.shape}")


def _apply_radial_correction(image_rgb: np.ndarray, k1: float, k2: float) -> np.ndarray:
    if abs(k1) < 1e-12 and abs(k2) < 1e-12:
        return image_rgb
    h, w = image_rgb.shape[:2]
    fx = float(w)
    fy = float(h)
    cx = float(w) * 0.5
    cy = float(h) * 0.5
    camera = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    dist = np.array([float(k1), float(k2), 0.0, 0.0, 0.0], dtype=np.float32)
    return cv2.undistort(image_rgb, camera, dist)


def apply_optical_correction_array(
    sensor_data: np.ndarray,
    target_hw: tuple[int, int],
    cfg: OpticalCorrectionConfig,
) -> np.ndarray:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError("Target shape must be positive.")

    data = np.asarray(sensor_data)
    if data.ndim not in {2, 3}:
        raise ValueError(f"Unsupported sensor array shape: {data.shape}")

    def _warp_channel(channel: np.ndarray) -> np.ndarray:
        plane = np.asarray(channel, dtype=np.float32)
        plane = cv2.resize(plane, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        plane = _apply_radial_correction(plane, k1=cfg.radial_k1, k2=cfg.radial_k2)

        center = (target_w * 0.5, target_h * 0.5)
        rot = cv2.getRotationMatrix2D(center, cfg.rotation_deg, 1.0)
        rot[0, 0] *= cfg.scale_x
        rot[0, 1] *= cfg.scale_x
        rot[1, 0] *= cfg.scale_y
        rot[1, 1] *= cfg.scale_y
        rot[0, 2] += cfg.shift_x
        rot[1, 2] += cfg.shift_y
        warped = cv2.warpAffine(
            plane,
            rot,
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        if cfg.perspective_matrix is not None:
            warped = cv2.warpPerspective(
                warped,
                cfg.perspective_matrix.astype(np.float32),
                (target_w, target_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return np.asarray(warped, dtype=np.float32)

    if data.ndim == 2:
        return _warp_channel(data)

    channels = [_warp_channel(data[:, :, idx]) for idx in range(int(data.shape[2]))]
    return np.stack(channels, axis=2)


def apply_optical_correction(sensor_rgb: np.ndarray, target_hw: tuple[int, int], cfg: OpticalCorrectionConfig) -> np.ndarray:
    img = np.asarray(sensor_rgb, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Sensor correction expects an RGB image.")
    corrected = apply_optical_correction_array(img, target_hw, cfg)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def overlay_modalities(base_image: np.ndarray, sensor_corrected_rgb: np.ndarray, alpha: float) -> np.ndarray:
    base = np.asarray(base_image)
    sensor = np.asarray(sensor_corrected_rgb, dtype=np.uint8)
    if base.ndim == 2:
        base_rgb = np.repeat(base[:, :, None], 3, axis=2)
    elif base.ndim == 3 and base.shape[2] >= 3:
        base_rgb = base[:, :, :3]
    else:
        raise ValueError("Base image must be grayscale or RGB.")

    if base_rgb.shape[:2] != sensor.shape[:2]:
        sensor = cv2.resize(sensor, (base_rgb.shape[1], base_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    a = float(max(0.0, min(1.0, alpha)))
    blended = base_rgb.astype(np.float32) * (1.0 - a) + sensor.astype(np.float32) * a
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)
