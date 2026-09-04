from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

import resources.app as app_module
from resources.app import NpecLabelingMainWindow
from resources.consolidation_engine import _derive_path_metadata
from resources.plate_identity import (
    allocate_plate_names,
    infer_plate_identity,
    infer_structured_filename_identity,
    scan_plate_identities,
)


LUCIFER_FILENAME_CASES = (
    ("0-1__2025_01_22_13_11_13_0582.png", "0-1", "2025-01-22 13:11:13", 582),
    ("136-4_2025_01_22_13_00_30_0516.png", "136-4", "2025-01-22 13:00:30", 516),
)


def _write_rgb(path: Path, value: int = 0) -> None:
    pixels = np.full((4, 5, 3), value, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _ocr_observation(text: str, confidence: float) -> dict[str, object]:
    return {
        "candidates": [{"text": text, "confidence": confidence}],
        "x": 0.45,
        "y": 0.45,
        "width": 0.1,
        "height": 0.05,
    }


class FlatLuciferFilenameRegressionTests(unittest.TestCase):
    def test_filename_fallback_excludes_capture_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            for filename, expected_plate, expected_timestamp, _ in LUCIFER_FILENAME_CASES:
                with self.subTest(filename=filename):
                    path = root / filename
                    _write_rgb(path)

                    record = infer_plate_identity(path)

                    self.assertEqual(record["plate_id"], expected_plate)
                    self.assertEqual(record["source_date"], expected_timestamp.replace(" ", "T"))
                    self.assertEqual(record["source_date_source"], "filename")
                    self.assertNotIn(Path(filename).stem.rsplit("_", 1)[-1], record["plate_id"])

    def test_structured_filename_identity_needs_no_image_decode(self) -> None:
        path = Path("/does/not/exist") / LUCIFER_FILENAME_CASES[0][0]

        record = infer_structured_filename_identity(path)

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["plate_id"], "0-1")
        self.assertEqual(record["source_date"], "2025-01-22T13:11:13")

    def test_structured_filename_wins_over_conflicting_low_quality_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / LUCIFER_FILENAME_CASES[0][0]
            _write_rgb(path)

            records = scan_plate_identities(
                [path],
                ocr_results={
                    str(path): {
                        "observations": [_ocr_observation("WRONG-99", 0.05)],
                    }
                },
                decode_codes=False,
            )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["plate_id"], "0-1")
            self.assertEqual(record["status"], "accepted")
            self.assertEqual(record["evidence"][0]["kind"], "filename")
            self.assertEqual(record["allocated_name"], "0-1__20250122_131113.png")

    def test_qr_and_barcode_remain_higher_priority_than_filename(self) -> None:
        cases = (
            ({"qr_values": ["URL:QR-7"]}, "QR-7", "qr"),
            ({"barcode_values": ["PLATE:BAR-8"]}, "BAR-8", "barcode"),
        )
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / LUCIFER_FILENAME_CASES[0][0]
            _write_rgb(path)
            for code_kwargs, expected_plate, expected_kind in cases:
                with self.subTest(kind=expected_kind):
                    payload = next(iter(code_kwargs.values()))[0]
                    symbology = "qr" if expected_kind == "qr" else "code128"
                    records = scan_plate_identities(
                        [path],
                        ocr_results={
                            str(path): {
                                "observations": [_ocr_observation("WRONG-99", 0.05)],
                                "barcodes": [
                                    {
                                        "payload": payload,
                                        "symbology": symbology,
                                    }
                                ],
                            }
                        },
                        decode_codes=False,
                    )

                    record = records[0]
                    self.assertEqual(record["plate_id"], expected_plate)
                    self.assertEqual(record["evidence"][0]["kind"], expected_kind)

    def test_consolidation_metadata_is_clean_for_flat_lucifer_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            for filename, expected_plate, expected_timestamp, expected_sequence in LUCIFER_FILENAME_CASES:
                with self.subTest(filename=filename):
                    metadata = _derive_path_metadata(root, root, Path(filename).stem)

                    self.assertEqual(metadata.petri, expected_plate)
                    self.assertEqual(metadata.timestamp, pd.Timestamp(expected_timestamp))
                    self.assertEqual(metadata.frame_index, expected_sequence)
                    self.assertEqual(metadata.relative_folder, ".")


class _FakeProgressDialog:
    def __init__(self, *_args, **_kwargs) -> None:
        self.value = 0

    def setWindowModality(self, _modality) -> None:
        pass

    def setMinimumDuration(self, _duration: int) -> None:
        pass

    def wasCanceled(self) -> bool:
        return False

    def setValue(self, value: int) -> None:
        self.value = value

    def setLabelText(self, _text: str) -> None:
        pass


class _FakeApplication:
    @staticmethod
    def setOverrideCursor(_cursor) -> None:
        pass

    @staticmethod
    def processEvents() -> None:
        pass

    @staticmethod
    def restoreOverrideCursor() -> None:
        pass


class _HeadlessLazyStabilizer:
    _stabilize_pipeline_folder = NpecLabelingMainWindow._stabilize_pipeline_folder
    _load_reusable_pipeline_stabilization = NpecLabelingMainWindow._load_reusable_pipeline_stabilization
    _identity_path_key = staticmethod(NpecLabelingMainWindow._identity_path_key)

    def __init__(self, image_paths: list[Path], stabilized_dir: Path) -> None:
        self.image_paths = image_paths
        self.stabilized_dir = stabilized_dir

    def _collect_image_paths_recursive(self, _input_dir: Path, exclude_roots=None):
        return list(self.image_paths), 0

    def _default_pipeline_stabilized_dir(self, _input_dir: Path, _output_dir: Path | None) -> Path:
        return self.stabilized_dir

    def _build_pipeline_stabilization_config(self):
        return SimpleNamespace(reference_mode="previous", target_mode="plant_top")

    def _decode_image_path(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    def _refresh_pipeline_stabilization_status(self) -> None:
        pass


class _HeadlessLazyIdentityLoader:
    _lazy_plate_identity_records = NpecLabelingMainWindow._lazy_plate_identity_records
    _identity_path_key = staticmethod(NpecLabelingMainWindow._identity_path_key)
    _path_sort_key = NpecLabelingMainWindow._path_sort_key
    _natural_sort_key = staticmethod(NpecLabelingMainWindow._natural_sort_key)

    def __init__(self) -> None:
        self.pipeline_lazy_plate_identity_cache: dict[str, dict[str, object]] = {}
        self.scan_calls = 0

    def _scan_plate_identity_records(self, _paths: list[Path], title: str) -> list[dict[str, object]]:
        self.scan_calls += 1
        raise AssertionError(f"Structured Lucifer names should not invoke OCR: {title}")


class _HeadlessLazyCollector:
    _collect_image_paths_recursive = NpecLabelingMainWindow._collect_image_paths_recursive
    _is_auxiliary_dataset_dirname = staticmethod(NpecLabelingMainWindow._is_auxiliary_dataset_dirname)
    _path_sort_key = NpecLabelingMainWindow._path_sort_key
    _natural_sort_key = staticmethod(NpecLabelingMainWindow._natural_sort_key)


class _HeadlessStabilizationResume:
    _load_reusable_pipeline_stabilization = NpecLabelingMainWindow._load_reusable_pipeline_stabilization
    _identity_path_key = staticmethod(NpecLabelingMainWindow._identity_path_key)


class FlatLuciferStabilizationGroupingTests(unittest.TestCase):
    def test_flat_structured_names_bypass_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            paths = []
            for filename, _, _, _ in LUCIFER_FILENAME_CASES:
                path = root / filename
                _write_rgb(path)
                paths.append(path)
            loader = _HeadlessLazyIdentityLoader()

            records = loader._lazy_plate_identity_records(root, paths)

            self.assertEqual(loader.scan_calls, 0)
            self.assertEqual([record["plate_id"] for record in records], ["0-1", "136-4"])

    def test_explicit_stabilized_input_root_is_collectible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            stabilized_dir = Path(tmp_name) / "_stabilized_input"
            stabilized_dir.mkdir()
            image_path = stabilized_dir / LUCIFER_FILENAME_CASES[0][0]
            _write_rgb(image_path)

            paths, walk_errors = _HeadlessLazyCollector()._collect_image_paths_recursive(stabilized_dir)

            self.assertEqual(paths, [image_path])
            self.assertEqual(walk_errors, 0)

    def test_complete_previous_mode_manifest_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            source_dir = root / "source"
            stabilized_dir = root / "_stabilized_input"
            source_dir.mkdir()
            stabilized_dir.mkdir()
            source_path = source_dir / LUCIFER_FILENAME_CASES[0][0]
            stabilized_path = stabilized_dir / source_path.name
            _write_rgb(source_path, 10)
            _write_rgb(stabilized_path, 11)
            manifest_path = stabilized_dir / "stabilization_manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_path",
                        "stabilized_path",
                        "target_mode",
                        "status",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": str(source_path),
                        "stabilized_path": str(stabilized_path),
                        "target_mode": "plant_top",
                        "status": "ok",
                    }
                )

            source_map = _HeadlessStabilizationResume()._load_reusable_pipeline_stabilization(
                stabilized_dir,
                [source_path],
                SimpleNamespace(reference_mode="previous", target_mode="plant_top"),
            )

            self.assertEqual(
                source_map,
                {
                    NpecLabelingMainWindow._identity_path_key(stabilized_path): str(source_path),
                },
            )

    def test_flat_folder_stabilization_groups_and_sorts_by_plate_identity(self) -> None:
        filenames = (
            ("0-1__2025_01_22_13_11_13_0582.png", 10),
            ("0-1__2025_01_22_13_15_13_0584.png", 11),
            ("136-4_2025_01_22_13_00_30_0516.png", 20),
            ("136-4_2025_01_22_13_05_30_0518.png", 21),
        )
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            input_dir = root / "flat"
            stabilized_dir = root / "stabilized"
            input_dir.mkdir()
            paths: list[Path] = []
            for filename, value in filenames:
                path = input_dir / filename
                _write_rgb(path, value)
                paths.append(path)

            identity_records = allocate_plate_names([infer_plate_identity(path) for path in paths])
            window = _HeadlessLazyStabilizer(
                image_paths=[paths[1], paths[3], paths[0], paths[2]],
                stabilized_dir=stabilized_dir,
            )
            stabilization_pairs: list[tuple[int, int]] = []

            def _stabilize(reference: np.ndarray, current: np.ndarray, _config):
                stabilization_pairs.append((int(reference[0, 0, 0]), int(current[0, 0, 0])))
                return SimpleNamespace(
                    warped_image=current,
                    warp_matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                    status="accepted",
                    score=1.0,
                    method="test_identity",
                    shift_x=0.0,
                    shift_y=0.0,
                    rotation_deg=0.0,
                    accepted=True,
                )

            with (
                patch.object(app_module, "QProgressDialog", _FakeProgressDialog),
                patch.object(app_module, "QApplication", _FakeApplication),
                patch.object(app_module, "stabilize_against_reference", side_effect=_stabilize),
                patch.object(
                    app_module,
                    "next_stabilization_reference",
                    side_effect=lambda _reference, result, _config: result.warped_image,
                ),
            ):
                result_dir, completed, walk_errors = window._stabilize_pipeline_folder(
                    input_dir,
                    root / "unused-output",
                    identity_records=identity_records,
                )

            self.assertEqual(result_dir, stabilized_dir)
            self.assertEqual(completed, 4)
            self.assertEqual(walk_errors, 0)
            self.assertEqual(stabilization_pairs, [(10, 11), (20, 21)])

            with (stabilized_dir / "stabilization_manifest.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                [row["stabilization_group"] for row in rows],
                ["0-1", "0-1", "136-4", "136-4"],
            )
            self.assertEqual(
                [row["status"] for row in rows],
                ["reference", "accepted", "reference", "accepted"],
            )


if __name__ == "__main__":
    unittest.main()
