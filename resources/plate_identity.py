from __future__ import annotations

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable, Iterable, Mapping

import cv2
import numpy as np
from PIL import Image


PLATE_IDENTITY_SCANNER_VERSION = 1
_GENERIC_TEXT = {
    "agar",
    "camera",
    "control",
    "date",
    "dish",
    "greiner",
    "lucifer",
    "npec",
    "petri",
    "plate",
    "time",
}
_GENERIC_FOLDER_NAMES = {
    "images",
    "image",
    "input",
    "inputs",
    "jpeg",
    "jpg",
    "png",
    "raw",
    "rgb",
    "stabilized",
    "stabilized_input",
}
_DATE_PATTERNS = (
    re.compile(
        r"(?<!\d)(20\d{2})[._-](\d{2})[._-](\d{2})[ T._-](\d{2})[._:-](\d{2})[._:-](\d{2})(?!\d)"
    ),
    re.compile(r"(?<!\d)(20\d{12})(?!\d)"),
)


def _resource_path(*parts: str) -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root).joinpath(*parts)
    return Path(__file__).resolve().parent.parent.joinpath(*parts)


def sanitize_plate_id(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:96]


def _normalize_code_payload(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^(?:URL|PLATE|DISH|ID)\s*:\s*", "", text, flags=re.IGNORECASE)
    if re.match(r"^https?://", text, flags=re.IGNORECASE):
        text = text.rstrip("/").rsplit("/", 1)[-1]
    return sanitize_plate_id(text)


def _parse_exif_datetime(value: object) -> datetime | None:
    text = str(value or "").strip().replace("\x00", "")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def parse_datetime_from_text(value: object) -> datetime | None:
    text = str(value or "")
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        try:
            if len(match.groups()) == 1:
                return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
            return datetime(*(int(part) for part in match.groups()))
        except (TypeError, ValueError):
            continue
    return None


def acquisition_datetime(path: Path) -> tuple[datetime, str]:
    source = Path(path)
    parsed = parse_datetime_from_text(source.stem)
    if parsed is not None:
        return parsed, "filename"

    try:
        with Image.open(source) as image:
            exif = image.getexif()
            for tag in (36867, 36868, 306):
                parsed = _parse_exif_datetime(exif.get(tag))
                if parsed is not None:
                    return parsed, {36867: "exif_datetime_original", 36868: "exif_datetime_digitized"}.get(
                        tag, "exif_datetime"
                    )
    except Exception:
        pass

    try:
        return datetime.fromtimestamp(source.stat().st_mtime).astimezone(), "filesystem_mtime"
    except OSError:
        return datetime.fromtimestamp(0).astimezone(), "unavailable"


def _candidate_tokens(text: object) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    tokens: list[str] = []
    for token in re.split(r"[\s,;:/|]+", raw):
        token = sanitize_plate_id(token)
        if len(token) < 2 or len(token) > 24:
            continue
        lowered = token.lower().strip("._-")
        if lowered in _GENERIC_TEXT:
            continue
        if parse_datetime_from_text(token) is not None:
            continue
        if re.fullmatch(r"\d+", token):
            if not 2 <= len(token) <= 4:
                continue
        elif not re.search(r"\d", token):
            if not (2 <= len(token) <= 12 and token.upper() == token):
                continue
        tokens.append(token)
    return tokens


def _observation_entries(ocr_result: Mapping[str, object] | None) -> list[dict[str, object]]:
    if not isinstance(ocr_result, Mapping):
        return []
    observations = ocr_result.get("observations", [])
    if not isinstance(observations, list):
        return []
    entries: list[dict[str, object]] = []
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        candidates = observation.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            continue
        best = candidates[0]
        if not isinstance(best, Mapping):
            continue
        tokens = _candidate_tokens(best.get("text", ""))
        if not tokens:
            continue
        try:
            confidence = float(best.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        bbox = tuple(float(observation.get(key, 0.0) or 0.0) for key in ("x", "y", "width", "height"))
        entries.append(
            {
                "tokens": tokens,
                "text": str(best.get("text", "")),
                "confidence": max(0.0, min(1.0, confidence)),
                "bbox": bbox,
            }
        )
    return entries


def _entry_center(entry: Mapping[str, object]) -> tuple[float, float]:
    bbox = entry.get("bbox", (0.0, 0.0, 0.0, 0.0))
    try:
        x, y, width, height = (float(v) for v in bbox)  # type: ignore[arg-type]
    except Exception:
        return 0.0, 0.0
    return x + width * 0.5, y + height * 0.5


def _cluster_ocr_entries(entries: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    clusters: list[list[dict[str, object]]] = []
    remaining = set(range(len(entries)))
    while remaining:
        seed = remaining.pop()
        cluster_indices = {seed}
        queue = [seed]
        while queue:
            current = queue.pop()
            cx, cy = _entry_center(entries[current])
            linked: list[int] = []
            for other in remaining:
                ox, oy = _entry_center(entries[other])
                if abs(cx - ox) <= 0.52 and abs(cy - oy) <= 0.35:
                    linked.append(other)
            for other in linked:
                remaining.remove(other)
                cluster_indices.add(other)
                queue.append(other)
        clusters.append([entries[index] for index in cluster_indices])
    return clusters


def _ocr_cluster_candidate(entries: list[dict[str, object]]) -> tuple[str, float, list[dict[str, object]]]:
    if not entries:
        return "", 0.0, []
    ordered = sorted(
        entries,
        key=lambda entry: (
            0 if any(re.fullmatch(r"\d{2,4}", str(token)) for token in entry.get("tokens", [])) else 1,
            -_entry_center(entry)[1],
            _entry_center(entry)[0],
        ),
    )
    tokens: list[str] = []
    evidence: list[dict[str, object]] = []
    confidences: list[float] = []
    for entry in ordered:
        for token in entry.get("tokens", []):
            clean = sanitize_plate_id(token)
            if clean and clean.lower() not in {existing.lower() for existing in tokens}:
                tokens.append(clean)
        confidence = float(entry.get("confidence", 0.0) or 0.0)
        confidences.append(confidence)
        evidence.append(
            {
                "kind": "vision_text",
                "text": str(entry.get("text", "")),
                "confidence": round(confidence, 4),
                "bbox": [round(float(v), 5) for v in entry.get("bbox", ())],
            }
        )
    if not tokens:
        return "", 0.0, []

    centers = [_entry_center(entry) for entry in ordered]
    mean_x = sum(point[0] for point in centers) / len(centers)
    mean_y = sum(point[1] for point in centers) / len(centers)
    corner_distance = min(
        ((mean_x - x) ** 2 + (mean_y - y) ** 2) ** 0.5
        for x, y in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
    )
    position_bonus = max(0.0, 0.14 - corner_distance * 0.12)
    top_left_bonus = 0.08 if mean_x <= 0.45 and mean_y >= 0.50 else 0.0
    structured_bonus = 0.08 if any(re.search(r"[A-Za-z]", token) and re.search(r"\d", token) for token in tokens) else 0.0
    multi_token_bonus = min(0.08, max(0, len(tokens) - 1) * 0.04)
    average_confidence = sum(confidences) / max(1, len(confidences))
    score = max(0.0, min(0.99, average_confidence * 0.82 + position_bonus + top_left_bonus + structured_bonus + multi_token_bonus))
    return "_".join(tokens), score, evidence


def _filename_candidate(path: Path) -> tuple[str, float, list[dict[str, object]]]:
    stem = str(path.stem)
    for pattern in _DATE_PATTERNS:
        date_match = pattern.search(stem)
        if date_match is None:
            continue
        suffix = stem[date_match.end() :]
        if re.fullmatch(r"[_-]\d{1,7}", suffix) is None:
            continue
        prefix = stem[: date_match.start()].strip("._- ")
        prefix_tokens = _candidate_tokens(prefix)
        if prefix_tokens:
            candidate = "_".join(prefix_tokens[:4])
            return candidate, 0.94, [{"kind": "filename", "text": stem, "confidence": 0.94}]

    without_dates = stem
    for pattern in _DATE_PATTERNS:
        without_dates = pattern.sub(" ", without_dates)
    without_dates = re.sub(r"(?:frame|image|img|capture|seq)[_-]*\d+", " ", without_dates, flags=re.IGNORECASE)
    stem_tokens = _candidate_tokens(without_dates)
    if stem_tokens:
        candidate = "_".join(stem_tokens[:4])
        return candidate, 0.56, [{"kind": "filename", "text": stem, "confidence": 0.56}]

    parent = sanitize_plate_id(path.parent.name)
    if parent and parent.lower() not in _GENERIC_FOLDER_NAMES:
        parent_tokens = _candidate_tokens(parent)
        if parent_tokens:
            candidate = "_".join(parent_tokens[:4])
            return candidate, 0.48, [{"kind": "folder", "text": path.parent.name, "confidence": 0.48}]
    return "", 0.0, []


def infer_structured_filename_identity(path: Path) -> dict[str, object] | None:
    source = Path(path)
    acquired_at = parse_datetime_from_text(source.stem)
    plate_id, confidence, evidence = _filename_candidate(source)
    plate_id = sanitize_plate_id(plate_id)
    if acquired_at is None or not plate_id or confidence < 0.90:
        return None
    return {
        "source_name": source.name,
        "source_path": str(source),
        "source_date": acquired_at.isoformat(timespec="seconds"),
        "source_date_source": "filename",
        "plate_id": plate_id,
        "allocated_name": "",
        "confidence": round(float(confidence), 4),
        "status": "accepted",
        "evidence": evidence,
        "user_override": None,
        "scanner_version": PLATE_IDENTITY_SCANNER_VERSION,
    }


def infer_plate_identity(
    path: Path,
    *,
    ocr_result: Mapping[str, object] | None = None,
    qr_values: Iterable[str] = (),
    barcode_values: Iterable[str] = (),
) -> dict[str, object]:
    source = Path(path)
    structured_filename_record = infer_structured_filename_identity(source)
    acquired_at, date_source = acquisition_datetime(source)
    evidence: list[dict[str, object]] = []

    for kind, values, score in (("qr", qr_values, 0.99), ("barcode", barcode_values, 0.96)):
        for raw in values:
            candidate = _normalize_code_payload(raw)
            if not candidate:
                continue
            evidence.append({"kind": kind, "text": str(raw), "confidence": score})
            return {
                "source_name": source.name,
                "source_path": str(source),
                "source_date": acquired_at.isoformat(timespec="seconds"),
                "source_date_source": date_source,
                "plate_id": candidate,
                "allocated_name": "",
                "confidence": score,
                "status": "accepted",
                "evidence": evidence,
                "user_override": None,
                "scanner_version": PLATE_IDENTITY_SCANNER_VERSION,
            }

    ocr_candidates = [_ocr_cluster_candidate(cluster) for cluster in _cluster_ocr_entries(_observation_entries(ocr_result))]
    ocr_candidates = [candidate for candidate in ocr_candidates if candidate[0]]
    filename_plate_id, filename_confidence, filename_evidence = _filename_candidate(source)
    if structured_filename_record is not None:
        plate_id = str(structured_filename_record["plate_id"])
        confidence = float(structured_filename_record["confidence"])
        evidence.extend(structured_filename_record["evidence"])
    elif ocr_candidates:
        plate_id, confidence, ocr_evidence = max(ocr_candidates, key=lambda candidate: candidate[1])
        evidence.extend(ocr_evidence)
    else:
        plate_id = filename_plate_id
        confidence = filename_confidence
        evidence.extend(filename_evidence)

    plate_id = sanitize_plate_id(plate_id)
    if not plate_id:
        status = "unidentified"
    elif confidence >= 0.72:
        status = "accepted"
    else:
        status = "review"
    return {
        "source_name": source.name,
        "source_path": str(source),
        "source_date": acquired_at.isoformat(timespec="seconds"),
        "source_date_source": date_source,
        "plate_id": plate_id,
        "allocated_name": "",
        "confidence": round(float(confidence), 4),
        "status": status,
        "evidence": evidence,
        "user_override": None,
        "scanner_version": PLATE_IDENTITY_SCANNER_VERSION,
    }


def _identity_sort_key(record: Mapping[str, object]) -> tuple[str, datetime, str]:
    plate_id = sanitize_plate_id(record.get("plate_id", "")) or "~unidentified"
    try:
        acquired = datetime.fromisoformat(str(record.get("source_date", "")))
    except (TypeError, ValueError):
        acquired = datetime.max
    return plate_id.lower(), acquired, str(record.get("source_name", "")).lower()


def allocate_plate_names(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    output = [dict(record) for record in records]
    plate_counts: dict[str, int] = {}
    used_names: set[str] = set()
    for index in sorted(range(len(output)), key=lambda value: _identity_sort_key(output[value])):
        record = output[index]
        plate_id = sanitize_plate_id(record.get("plate_id", ""))
        source_name = str(record.get("source_name", "image.png"))
        extension = Path(source_name).suffix.lower() or ".png"
        try:
            acquired = datetime.fromisoformat(str(record.get("source_date", "")))
            date_token = acquired.strftime("%Y%m%d_%H%M%S")
        except (TypeError, ValueError):
            date_token = "undated"
        if not plate_id:
            fallback = sanitize_plate_id(Path(source_name).stem) or "unidentified"
            base = f"{fallback}__{date_token}"
        else:
            base = f"{plate_id}__{date_token}"
        plate_key = plate_id.lower() or "~unidentified"
        plate_counts[plate_key] = plate_counts.get(plate_key, 0) + 1
        record["plate_frame_index"] = int(plate_counts[plate_key])
        candidate = f"{base}{extension}"
        capture = 1
        while candidate.lower() in used_names:
            capture += 1
            candidate = f"{base}__capture{capture:02d}{extension}"
        used_names.add(candidate.lower())
        record["allocated_name"] = candidate
    return output


def _resolve_ocr_executable(explicit: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env_value = str(os.environ.get("NPEC_PLATE_OCR_HELPER", "") or "").strip()
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(
        [
            _resource_path("resources", "bin", "npec_plate_ocr"),
            Path(__file__).resolve().parent / "bin" / "npec_plate_ocr",
        ]
    )
    for candidate in candidates:
        try:
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate
        except OSError:
            continue
    return None


def run_macos_vision_ocr(paths: Iterable[Path], executable: Path | None = None) -> dict[str, dict[str, object]]:
    helper = _resolve_ocr_executable(executable)
    if helper is None or sys.platform != "darwin":
        return {}
    path_list = [Path(path) for path in paths]
    results: dict[str, dict[str, object]] = {}
    for start in range(0, len(path_list), 64):
        chunk = path_list[start : start + 64]
        try:
            completed = subprocess.run(
                [str(helper), *(str(path) for path in chunk)],
                capture_output=True,
                text=True,
                timeout=max(30, len(chunk) * 8),
                check=False,
            )
        except Exception:
            continue
        for line in completed.stdout.splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("path"):
                results[str(payload["path"])] = payload
    return results


def _decode_qr_and_barcodes(path: Path) -> tuple[list[str], list[str]]:
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception:
        image = None
    if image is None:
        return [], []
    height, width = image.shape[:2]
    scale = min(1.0, 3200.0 / float(max(height, width)))
    if scale < 1.0:
        image = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)

    qr_values: list[str] = []
    try:
        detector = cv2.QRCodeDetector()
        ok, decoded, _points, _straight = detector.detectAndDecodeMulti(image)
        if ok:
            qr_values.extend(str(value).strip() for value in decoded if str(value).strip())
        if not qr_values:
            decoded_one, _points, _straight = detector.detectAndDecode(image)
            if str(decoded_one).strip():
                qr_values.append(str(decoded_one).strip())
    except Exception:
        pass

    barcode_values: list[str] = []
    try:
        barcode_cls = getattr(cv2, "barcode_BarcodeDetector", None)
        if barcode_cls is not None:
            result = barcode_cls().detectAndDecode(image)
            if isinstance(result, tuple):
                for value in result:
                    if isinstance(value, str) and value.strip():
                        barcode_values.append(value.strip())
                    elif isinstance(value, (tuple, list)):
                        barcode_values.extend(str(entry).strip() for entry in value if isinstance(entry, str) and entry.strip())
    except Exception:
        pass
    return list(dict.fromkeys(qr_values)), list(dict.fromkeys(barcode_values))


def scan_plate_identities(
    paths: Iterable[Path],
    *,
    ocr_executable: Path | None = None,
    ocr_results: Mapping[str, Mapping[str, object]] | None = None,
    decode_codes: bool = True,
    progress_callback: Callable[[int, int, str], bool] | None = None,
) -> list[dict[str, object]]:
    path_list = [Path(path) for path in paths]
    resolved_ocr = dict(ocr_results or run_macos_vision_ocr(path_list, executable=ocr_executable))
    records: list[dict[str, object]] = []
    total = len(path_list)
    for index, path in enumerate(path_list, start=1):
        if progress_callback is not None and not bool(progress_callback(index - 1, total, path.name)):
            break
        qr_values, barcode_values = _decode_qr_and_barcodes(path) if decode_codes else ([], [])
        ocr_result = resolved_ocr.get(str(path))
        if ocr_result is None:
            try:
                ocr_result = resolved_ocr.get(str(path.resolve()))
            except OSError:
                pass
        if isinstance(ocr_result, Mapping):
            vision_barcodes = ocr_result.get("barcodes", [])
            if isinstance(vision_barcodes, list):
                for entry in vision_barcodes:
                    if not isinstance(entry, Mapping):
                        continue
                    payload = str(entry.get("payload", "") or "").strip()
                    if not payload:
                        continue
                    symbology = str(entry.get("symbology", "") or "").lower()
                    if "qr" in symbology:
                        qr_values.append(payload)
                    else:
                        barcode_values.append(payload)
        records.append(
            infer_plate_identity(
                path,
                ocr_result=ocr_result,
                qr_values=qr_values,
                barcode_values=barcode_values,
            )
        )
        if progress_callback is not None and not bool(progress_callback(index, total, path.name)):
            break
    return allocate_plate_names(records)


def write_plate_identity_manifest(records: Iterable[Mapping[str, object]], output_path: Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(record) for record in records]
    fields = [
        "source_name",
        "source_path",
        "source_date",
        "source_date_source",
        "plate_id",
        "allocated_name",
        "plate_frame_index",
        "confidence",
        "status",
        "user_override",
        "scanner_version",
        "evidence",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field, []), ensure_ascii=True)
                    if field == "evidence"
                    else row.get(field, "")
                    for field in fields
                }
            )
    path.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_plate_identity_manifest(input_path: Path) -> list[dict[str, object]]:
    """Read a scanner manifest without changing its allocated filenames."""

    path = Path(input_path)
    if not path.exists() or not path.is_file():
        return []
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [dict(row) for row in payload if isinstance(row, Mapping)] if isinstance(payload, list) else []

    rows: list[dict[str, object]] = []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, object] = dict(raw)
                evidence_raw = str(row.get("evidence", "") or "").strip()
                try:
                    evidence = json.loads(evidence_raw) if evidence_raw else []
                except json.JSONDecodeError:
                    evidence = []
                row["evidence"] = evidence if isinstance(evidence, list) else []
                for field in ("plate_frame_index", "scanner_version"):
                    try:
                        row[field] = int(str(row.get(field, "") or ""))
                    except (TypeError, ValueError):
                        row[field] = 0
                try:
                    row["confidence"] = float(str(row.get("confidence", "") or "0"))
                except (TypeError, ValueError):
                    row["confidence"] = 0.0
                rows.append(row)
    except OSError:
        return []
    return rows
