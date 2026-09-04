from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from resources.app import (
    LAZY_COMBINED_ANALYSIS_MODE,
    LUCIFER_PIXEL_SIZE_MM,
    NpecLabelingMainWindow,
    POTATO_ENDPOINT_ANALYSIS_MODE,
    lazy_analysis_includes_mask_total,
    lazy_analysis_includes_ownership,
)
from resources.plate_identity import write_plate_identity_manifest


class _DummyLabel:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, value: str) -> None:
        self.text = str(value)


class _DummyButton:
    def __init__(self) -> None:
        self.enabled = False
        self.checked = False
        self.signals_blocked = False

    def setEnabled(self, value: bool) -> None:
        self.enabled = bool(value)

    def setChecked(self, value: bool) -> None:
        self.checked = bool(value)

    def isChecked(self) -> bool:
        return self.checked

    def blockSignals(self, value: bool) -> None:
        self.signals_blocked = bool(value)

    def setText(self, value: str) -> None:
        self._text = str(value)

    def text(self) -> str:
        return getattr(self, "_text", "")

    def findData(self, _value: object) -> int:
        return -1

    def setCurrentIndex(self, _value: int) -> None:
        return


class _DummySpin:
    def __init__(self, value: int) -> None:
        self._value = int(value)
        self.enabled = True
        self.signals_blocked = False

    def value(self) -> int:
        return self._value

    def setValue(self, value: int) -> None:
        self._value = int(value)

    def setEnabled(self, value: bool) -> None:
        self.enabled = bool(value)

    def blockSignals(self, value: bool) -> None:
        self.signals_blocked = bool(value)


class _DummyCombo:
    def __init__(self, current_data: str) -> None:
        self.current_data = str(current_data)
        self.enabled = True
        self.signals_blocked = False
        self.items = [
            LAZY_COMBINED_ANALYSIS_MODE,
            "mask_total",
            "ownership",
            POTATO_ENDPOINT_ANALYSIS_MODE,
        ]

    def currentData(self) -> str:
        return self.current_data

    def setEnabled(self, value: bool) -> None:
        self.enabled = bool(value)

    def findData(self, value: object) -> int:
        try:
            return self.items.index(str(value))
        except ValueError:
            return -1

    def setCurrentIndex(self, value: int) -> None:
        if 0 <= int(value) < len(self.items):
            self.current_data = self.items[int(value)]

    def blockSignals(self, value: bool) -> None:
        self.signals_blocked = bool(value)


class _FakeLazyModeWindow:
    _current_pipeline_lazy_analysis_mode = NpecLabelingMainWindow._current_pipeline_lazy_analysis_mode
    _set_shared_expected_plant_count = NpecLabelingMainWindow._set_shared_expected_plant_count
    _on_pipeline_lazy_analysis_mode_changed = NpecLabelingMainWindow._on_pipeline_lazy_analysis_mode_changed
    _on_pipeline_lazy_expected_plants_changed = NpecLabelingMainWindow._on_pipeline_lazy_expected_plants_changed
    _on_plant_tracks_expected_changed = NpecLabelingMainWindow._on_plant_tracks_expected_changed
    _activate_arabidopsis_lazy_defaults = NpecLabelingMainWindow._activate_arabidopsis_lazy_defaults

    def __init__(self) -> None:
        self.pipeline_lazy_analysis_mode = "ownership"
        self.pipeline_lazy_non_endpoint_analysis_mode = "ownership"
        self.pipeline_lazy_standard_expected_plants = 5
        self.plant_track_expected_count = 5
        self.analytics_track_expected_count = 5
        self.pipeline_lazy_analysis_mode_combo = _DummyCombo("ownership")
        self.pipeline_lazy_expected_plants_spin = _DummySpin(5)
        self.plant_tracks_expected_spin = _DummySpin(5)
        self.analytics_expected_tracks_spin = _DummySpin(5)
        self.pipeline_lazy_measure_class_ids_edit = _DummyButton()
        self.pipeline_lazy_shoot_class_ids_edit = _DummyButton()
        self.pipeline_lazy_root_metric_combo = _DummyButton()
        self.pipeline_lazy_shoot_rgb_rescue_cb = _DummyButton()
        self.pipeline_lazy_mask_only_video_cb = _DummyButton()
        self.pipeline_stabilize_before_run_cb = _DummyButton()
        self.pipeline_lazy_analyze_masks_btn = _DummyButton()
        self.pipeline_stabilization_enabled = False
        self.pipeline_stabilization_reference_mode = "first"
        self.pipeline_stabilization_target_mode = "dish_frame"
        self.dirty = False

    def _set_dirty(self, value: bool) -> None:
        self.dirty = bool(value)

    def _refresh_plant_tracks_table(self) -> None:
        return


class _FakeLuciferPresetWindow:
    _apply_pyphenotyper_preset = NpecLabelingMainWindow._apply_pyphenotyper_preset
    _preset_default_pixel_size_mm = NpecLabelingMainWindow._preset_default_pixel_size_mm
    _preset_fixed_patch_size = NpecLabelingMainWindow._preset_fixed_patch_size

    def __init__(self) -> None:
        self.pyphenotyper_preset_key = "custom"
        self.pyphenotyper_pipeline_dir: Path | None = None
        self.pyphenotyper_root_model_path: Path | None = None
        self.pyphenotyper_shoot_model_path: Path | None = None
        self.pyphenotyper_seed_model_path: Path | None = None
        self._wand_prediction_cache: dict[object, object] = {}
        self.pixel_size_mm: float | None = None
        self.dirty = False

    def _detect_builtin_pyphenotyper_pipeline_dir(self) -> Path:
        return Path("/bundled/pipeline")

    def _detect_builtin_lucifer_model_paths(self) -> tuple[Path, Path]:
        return Path("/bundled/lucifer_root.h5"), Path("/bundled/lucifer_shoot.h5")

    def _detect_builtin_rgb_inoculated_model_path(self) -> Path:
        return Path("/bundled/rgb_inoculated_shoot_v3.keras")

    def _set_pixel_size_defaults(self, value, **_kwargs) -> bool:
        self.pixel_size_mm = float(value)
        return True

    def _update_model_info_label(self) -> None:
        return

    def _refresh_general_root_starter_controls(self) -> None:
        return

    def _set_dirty(self, value: bool) -> None:
        self.dirty = bool(value)


class _FakeInoculatedConfigWindow:
    _build_pyphenotyper_config_from_ui = NpecLabelingMainWindow._build_pyphenotyper_config_from_ui
    _preset_fixed_patch_size = NpecLabelingMainWindow._preset_fixed_patch_size

    def __init__(self) -> None:
        self.pyphenotyper_preset_key = "lucifer_inoculated_builtin"
        self.pyphenotyper_pipeline_dir = Path("/selected/pipeline")
        self.pyphenotyper_root_model_path = Path("/selected/root.keras")
        self.pyphenotyper_shoot_model_path = Path("/selected/shoot.keras")
        self.pyphenotyper_seed_model_path = Path("/must/not/be/used.keras")
        self.pyphenotyper_pipeline_edit = _DummyButton()
        self.pyphenotyper_root_model_edit = _DummyButton()
        self.pyphenotyper_shoot_model_edit = _DummyButton()
        self.pyphenotyper_root_class_combo = _DummyCombo("1")
        self.pyphenotyper_shoot_class_combo = _DummyCombo("2")
        self.pyphenotyper_seed_class_combo = _DummyCombo("4")
        self.pyphenotyper_patch_size_spin = _DummySpin(256)
        self.pyphenotyper_refinement_steps_spin = _DummySpin(2)
        self.pyphenotyper_min_component_area_spin = _DummySpin(40)
        self.pyphenotyper_bbox_padding_spin = _DummySpin(12)
        self.pyphenotyper_tracking_margin_spin = _DummySpin(26)
        self.pyphenotyper_pixel_size_mm_spin = _DummySpin(1)
        self.pyphenotyper_include_occlusion_cb = _DummyButton()
        self.pyphenotyper_include_occlusion_cb.setChecked(True)
        self.pyphenotyper_enable_bbox_tracking_cb = _DummyButton()
        self.pyphenotyper_enable_bbox_tracking_cb.setChecked(True)
        self.pyphenotyper_preserve_root_overlap_cb = _DummyButton()
        self.pyphenotyper_preserve_root_on_shoot_overlap = False
        self.pyphenotyper_shoot_color_rescue_mode = "auto"
        self.pyphenotyper_lucifer_gan_gap_repair_dir = None
        self.general_root_starter_calibration = {}

    def _ensure_segmentation_pipeline_classes(self, **_kwargs) -> dict[str, int]:
        return {"root": 1, "shoot": 2, "lateral": 3}

    def _current_pyphenotyper_lucifer_gan_gap_repair_dir(self):
        return None

    def _detect_builtin_pyphenotyper_pipeline_dir(self) -> Path:
        return Path("/bundled/pipeline")

    def _detect_builtin_lucifer_model_paths(self) -> tuple[Path, Path]:
        return Path("/bundled/lucifer_root.h5"), Path("/bundled/lucifer_shoot.h5")

    def _detect_builtin_rgb_inoculated_model_path(self) -> Path:
        return Path("/bundled/rgb_inoculated_shoot_v3.keras")

    def _current_pyphenotyper_shoot_color_rescue_mode(self) -> str:
        return "auto"


class _FakeLazyOutputWindow:
    _natural_sort_key = staticmethod(NpecLabelingMainWindow._natural_sort_key)
    _path_sort_key = NpecLabelingMainWindow._path_sort_key
    _is_auxiliary_dataset_dirname = staticmethod(NpecLabelingMainWindow._is_auxiliary_dataset_dirname)
    _normalize_mask_match_stem = staticmethod(NpecLabelingMainWindow._normalize_mask_match_stem)
    _collect_image_paths_recursive = NpecLabelingMainWindow._collect_image_paths_recursive
    _parse_lazy_timestamp = NpecLabelingMainWindow._parse_lazy_timestamp
    _extract_lazy_frame_index = NpecLabelingMainWindow._extract_lazy_frame_index
    _derive_lazy_segmentation_metadata = NpecLabelingMainWindow._derive_lazy_segmentation_metadata
    _make_uid = NpecLabelingMainWindow._make_uid
    _lazy_source_stem_from_mask_stem = staticmethod(NpecLabelingMainWindow._lazy_source_stem_from_mask_stem)
    _is_lazy_mask_artifact = staticmethod(NpecLabelingMainWindow._is_lazy_mask_artifact)
    _collect_existing_lazy_mask_records = NpecLabelingMainWindow._collect_existing_lazy_mask_records
    _discover_pipeline_lazy_outputs = NpecLabelingMainWindow._discover_pipeline_lazy_outputs
    _set_pipeline_lazy_latest_outputs = NpecLabelingMainWindow._set_pipeline_lazy_latest_outputs
    _refresh_pipeline_lazy_output_actions = NpecLabelingMainWindow._refresh_pipeline_lazy_output_actions
    _current_pipeline_lazy_shoot_rgb_rescue_enabled = NpecLabelingMainWindow._current_pipeline_lazy_shoot_rgb_rescue_enabled
    _current_pyphenotyper_shoot_color_rescue_mode = NpecLabelingMainWindow._current_pyphenotyper_shoot_color_rescue_mode
    _lazy_lucifer_context_suggests_green_shoots = NpecLabelingMainWindow._lazy_lucifer_context_suggests_green_shoots
    _lazy_shoot_rgb_rescue_for_run = NpecLabelingMainWindow._lazy_shoot_rgb_rescue_for_run

    def __init__(self) -> None:
        self.pipeline_lazy_latest_outputs: dict[str, Path] = {}
        self.pipeline_lazy_input_dir: Path | None = None
        self.pipeline_lazy_output_dir: Path | None = None
        self.pipeline_lazy_shoot_rgb_rescue_enabled = False
        self.pyphenotyper_shoot_color_rescue_mode = "auto"
        self.pipeline_lazy_outputs_label = _DummyLabel()
        self.pipeline_lazy_open_output_folder_btn = _DummyButton()
        self.pipeline_lazy_open_workbook_btn = _DummyButton()
        self.pipeline_lazy_open_timelapse_btn = _DummyButton()
        self.pipeline_lazy_open_track_summary_btn = _DummyButton()
        self.pipeline_lazy_open_review_btn = _DummyButton()
        self.pipeline_lazy_open_frame_review_btn = _DummyButton()
        self.pipeline_lazy_open_identity_state_btn = _DummyButton()
        self.pipeline_lazy_open_video_btn = _DummyButton()
        self.pipeline_lazy_open_video_summary_btn = _DummyButton()


class _FakeLazyOwnershipWindow:
    _natural_sort_key = staticmethod(NpecLabelingMainWindow._natural_sort_key)
    _build_lazy_segmentation_master_table = NpecLabelingMainWindow._build_lazy_segmentation_master_table

    def __init__(self) -> None:
        self.plant_track_expected_count = 1

    def _build_analytics_config_from_ui(self) -> SimpleNamespace:
        return SimpleNamespace(
            root_class_id=1,
            lateral_class_id=3,
            shoot_class_id=2,
            pixel_size_mm=0.1,
            expected_track_count=1,
            shoot_rgb_green_only_enabled=False,
        )

    def _current_pipeline_lazy_measure_class_ids(self) -> tuple[int, ...]:
        return (1, 3)

    def _current_pipeline_lazy_shoot_class_ids(self) -> tuple[int, ...]:
        return (2,)

    def _current_pipeline_lazy_root_metric_mode(self) -> str:
        return "total"

    def _lazy_shoot_rgb_rescue_for_run(self, **_kwargs) -> bool:
        return False

    def _read_index_mask_from_path(self, _path: Path) -> np.ndarray:
        return np.array([[0, 1], [3, 2]], dtype=np.uint8)

    def _decode_image_path(self, _path: Path) -> np.ndarray:
        return np.zeros((2, 2, 3), dtype=np.uint8)


class AppLazyOutputsTests(unittest.TestCase):
    def test_lucifer_arabidopsis_defaults_carry_validated_rgb_run_settings(self) -> None:
        window = _FakeLazyModeWindow()
        window._activate_arabidopsis_lazy_defaults(lucifer_rgb=True)

        self.assertEqual(window.pipeline_lazy_analysis_mode, LAZY_COMBINED_ANALYSIS_MODE)
        self.assertEqual(window.pipeline_lazy_non_endpoint_analysis_mode, LAZY_COMBINED_ANALYSIS_MODE)
        self.assertEqual(window.pipeline_lazy_analysis_mode_combo.currentData(), LAZY_COMBINED_ANALYSIS_MODE)
        self.assertEqual(window.pipeline_lazy_standard_expected_plants, 5)
        self.assertEqual(window.pipeline_lazy_measure_class_ids_text, "1,3")
        self.assertEqual(window.pipeline_lazy_shoot_class_ids_text, "2")
        self.assertEqual(window.pipeline_lazy_root_metric_mode, "total")
        self.assertTrue(window.pipeline_lazy_plate_identity_enabled)
        self.assertTrue(window.pipeline_lazy_shoot_rgb_rescue_enabled)
        self.assertFalse(window.pipeline_lazy_video_mask_only)
        self.assertFalse(window.pipeline_lazy_mask_only_video_cb.isChecked())
        self.assertTrue(window.pipeline_stabilization_enabled)
        self.assertEqual(window.pipeline_stabilization_reference_mode, "previous")
        self.assertEqual(window.pipeline_stabilization_target_mode, "plant_top")
        self.assertTrue(window.pipeline_stabilize_before_run_cb.isChecked())

    def test_hades_arabidopsis_defaults_do_not_enable_rgb_shoot_rescue(self) -> None:
        window = _FakeLazyModeWindow()
        window._activate_arabidopsis_lazy_defaults(lucifer_rgb=False)

        self.assertEqual(window.pipeline_lazy_analysis_mode, LAZY_COMBINED_ANALYSIS_MODE)
        self.assertEqual(window.pipeline_lazy_standard_expected_plants, 5)
        self.assertFalse(window.pipeline_lazy_shoot_rgb_rescue_enabled)
        self.assertTrue(window.pipeline_stabilization_enabled)

    def test_combined_lazy_mode_runs_both_metric_families(self) -> None:
        self.assertTrue(lazy_analysis_includes_mask_total(LAZY_COMBINED_ANALYSIS_MODE))
        self.assertTrue(lazy_analysis_includes_ownership(LAZY_COMBINED_ANALYSIS_MODE))
        self.assertTrue(lazy_analysis_includes_mask_total("mask_total"))
        self.assertFalse(lazy_analysis_includes_ownership("mask_total"))
        self.assertFalse(lazy_analysis_includes_mask_total("ownership"))
        self.assertTrue(lazy_analysis_includes_ownership("ownership"))

    def test_both_lazy_entry_points_dispatch_every_combined_postprocessor(self) -> None:
        for method in (
            NpecLabelingMainWindow._analyze_existing_lazy_masks,
            NpecLabelingMainWindow._run_inference_external_folder_lazy,
        ):
            source = inspect.getsource(method)
            self.assertIn("lazy_analysis_includes_mask_total(lazy_analysis_mode)", source)
            self.assertIn("lazy_analysis_includes_ownership(lazy_analysis_mode)", source)
            self.assertIn("write_occlusion_aware_outputs(", source)

    def test_lazy_ownership_checkpoints_resume_completed_plates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lazy_ownership_resume_") as tmp_dir_name:
            root = Path(tmp_dir_name)
            input_root = root / "input"
            output_dir = root / "output"
            input_root.mkdir()
            output_dir.mkdir()

            records: list[dict[str, object]] = []
            for index, plate in enumerate(("Plate-A", "Plate-B")):
                image_path = input_root / f"{plate}.png"
                mask_path = output_dir / f"{plate}_mask.png"
                image_path.write_bytes(b"image")
                mask_path.write_bytes(b"mask")
                records.append(
                    {
                        "uid": f"uid-{index}",
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "output_mask": str(mask_path),
                        "meta": {
                            "series": plate,
                            "petri": plate,
                            "timestamp": f"2026-07-2{index + 1}T09:00:00",
                            "frame_index": 0,
                            "relative_folder": plate,
                        },
                    }
                )

            def _fake_temporal_analytics(*_args, **_kwargs) -> dict[str, object]:
                return {
                    "rows": [
                        {
                            "frame_index": 0,
                            "plant_id": 1,
                            "total_root_length_mm_raw": 1.0,
                            "total_root_length_weighted_mm_raw": 1.0,
                            "primary_root_length_mm_raw": 0.6,
                            "primary_root_length_weighted_mm_raw": 0.6,
                            "lateral_total_length_mm": 0.4,
                            "lateral_total_length_weighted_mm": 0.4,
                            "shoot_area_mm2": 0.2,
                            "shoot_area_px": 20,
                        }
                    ]
                }

            window = _FakeLazyOwnershipWindow()
            with patch("resources.app.run_temporal_analytics", side_effect=_fake_temporal_analytics) as analytics:
                first_result = window._build_lazy_segmentation_master_table(
                    input_root,
                    output_dir,
                    records,
                    progress_callback=lambda done, _total, _label: done < 1,
                )

                self.assertIsNone(first_result[0])
                checkpoint_dir = output_dir / "_ownership_checkpoints"
                self.assertEqual(len(list(checkpoint_dir.glob("*.csv"))), 1)
                self.assertEqual(analytics.call_count, 1)

                resumed_df, master_csv, total_csv, warnings = window._build_lazy_segmentation_master_table(
                    input_root,
                    output_dir,
                    records,
                    progress_callback=lambda _done, _total, _label: True,
                )

            self.assertIsNotNone(resumed_df)
            self.assertEqual(len(resumed_df), 2)
            self.assertEqual(analytics.call_count, 2)
            self.assertTrue(master_csv.is_file())
            self.assertTrue(total_csv.is_file())
            self.assertFalse(checkpoint_dir.exists())
            self.assertEqual(warnings, [])

    def test_potato_mode_does_not_overwrite_arabidopsis_expected_count(self) -> None:
        window = _FakeLazyModeWindow()
        window.pipeline_lazy_analysis_mode_combo.current_data = POTATO_ENDPOINT_ANALYSIS_MODE

        window._on_pipeline_lazy_analysis_mode_changed()

        self.assertEqual(window.pipeline_lazy_analysis_mode, POTATO_ENDPOINT_ANALYSIS_MODE)
        self.assertEqual(window.pipeline_lazy_expected_plants_spin.value(), 1)
        self.assertFalse(window.pipeline_lazy_expected_plants_spin.enabled)
        self.assertEqual(window.pipeline_lazy_standard_expected_plants, 5)
        self.assertEqual(window.plant_track_expected_count, 5)
        self.assertEqual(window.analytics_track_expected_count, 5)

        window.pipeline_lazy_analysis_mode_combo.current_data = "ownership"
        window._on_pipeline_lazy_analysis_mode_changed()

        self.assertEqual(window.pipeline_lazy_analysis_mode, "ownership")
        self.assertEqual(window.pipeline_lazy_expected_plants_spin.value(), 5)
        self.assertTrue(window.pipeline_lazy_expected_plants_spin.enabled)
        self.assertEqual(window.plant_track_expected_count, 5)
        self.assertEqual(window.analytics_track_expected_count, 5)

        window.pipeline_lazy_analysis_mode_combo.current_data = LAZY_COMBINED_ANALYSIS_MODE
        window._on_pipeline_lazy_analysis_mode_changed()

        self.assertEqual(window.pipeline_lazy_analysis_mode, LAZY_COMBINED_ANALYSIS_MODE)
        self.assertEqual(window.pipeline_lazy_non_endpoint_analysis_mode, LAZY_COMBINED_ANALYSIS_MODE)

    def test_custom_arabidopsis_count_survives_a_potato_mode_round_trip(self) -> None:
        window = _FakeLazyModeWindow()
        window._on_pipeline_lazy_expected_plants_changed(7)
        self.assertEqual(window.pipeline_lazy_standard_expected_plants, 7)

        window.pipeline_lazy_analysis_mode_combo.current_data = POTATO_ENDPOINT_ANALYSIS_MODE
        window._on_pipeline_lazy_analysis_mode_changed()
        window._on_plant_tracks_expected_changed(7)
        self.assertEqual(window.pipeline_lazy_expected_plants_spin.value(), 1)

        window.pipeline_lazy_analysis_mode_combo.current_data = "mask_total"
        window._on_pipeline_lazy_analysis_mode_changed()

        self.assertEqual(window.pipeline_lazy_expected_plants_spin.value(), 7)
        self.assertEqual(window.plant_track_expected_count, 7)
        self.assertEqual(window.analytics_track_expected_count, 7)

    def test_explicit_lucifer_arabidopsis_preset_uses_lucifer_scale(self) -> None:
        window = _FakeLuciferPresetWindow()

        applied = window._apply_pyphenotyper_preset("lucifer_builtin", mark_dirty=True)

        self.assertTrue(applied)
        self.assertEqual(window.pyphenotyper_root_model_path, Path("/bundled/lucifer_root.h5"))
        self.assertEqual(window.pyphenotyper_shoot_model_path, Path("/bundled/lucifer_shoot.h5"))
        self.assertAlmostEqual(float(window.pixel_size_mm), float(LUCIFER_PIXEL_SIZE_MM), places=12)
        self.assertEqual(window._preset_fixed_patch_size("lucifer_builtin"), 256)
        self.assertTrue(window.dirty)

    def test_inoculated_preset_applies_validated_shoot_model_only(self) -> None:
        window = _FakeLuciferPresetWindow()

        applied = window._apply_pyphenotyper_preset("lucifer_inoculated_builtin", mark_dirty=True)

        self.assertTrue(applied)
        self.assertEqual(window.pyphenotyper_root_model_path, Path("/bundled/lucifer_root.h5"))
        self.assertEqual(
            window.pyphenotyper_shoot_model_path,
            Path("/bundled/rgb_inoculated_shoot_v3.keras"),
        )
        self.assertIsNone(window.pyphenotyper_seed_model_path)

    def test_inoculated_runtime_config_cannot_use_unqualified_root_or_seed_channels(self) -> None:
        window = _FakeInoculatedConfigWindow()

        with patch("resources.app.validate_pyphenotyper_setup"):
            config = window._build_pyphenotyper_config_from_ui()

        self.assertEqual(config.root_model_path, Path("/bundled/lucifer_root.h5"))
        self.assertEqual(config.shoot_model_path, Path("/bundled/rgb_inoculated_shoot_v3.keras"))
        self.assertIsNone(config.seed_model_path)
        self.assertIsNone(config.seed_class_id)
        self.assertIsNone(config.profile_overrides)
        self.assertIsNone(config.root_profile_overrides)
        self.assertEqual(config.shoot_profile_overrides["name"], "rgb_inoculated_shoot_v3")
        self.assertEqual(config.shoot_profile_overrides["shoot_label_ids"], (2,))
        self.assertEqual(config.shoot_profile_overrides["tile_halo"], 32)
        self.assertFalse(config.include_occlusion)

    def test_auxiliary_dataset_dir_filter_skips_generated_analysis_artifacts(self) -> None:
        self.assertTrue(NpecLabelingMainWindow._is_auxiliary_dataset_dirname("_analysis_backup_before_rerun"))
        self.assertTrue(NpecLabelingMainWindow._is_auxiliary_dataset_dirname("_rgb_green_shoot_display_masks_green_only"))
        self.assertTrue(NpecLabelingMainWindow._is_auxiliary_dataset_dirname("_stabilized_input"))
        self.assertFalse(NpecLabelingMainWindow._is_auxiliary_dataset_dirname("B3-3"))

    def test_latest_lazy_output_actions_follow_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lazy_outputs_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            output_dir = tmp_dir / "segmentation_masks"
            output_dir.mkdir()
            workbook = output_dir / "npec_total_root_and_shoot_all_metrics.xlsx"
            workbook.write_text("workbook", encoding="utf-8")
            timelapse = output_dir / "npec_total_root_length_timelapse.csv"
            timelapse.write_text("frame,total\n", encoding="utf-8")
            track_summary = output_dir / "npec_per_plant_track_summary.csv"
            track_summary.write_text("plant_id,status\n", encoding="utf-8")
            review = output_dir / "npec_tracks_needing_review.csv"
            review.write_text("plant_id,status\n", encoding="utf-8")
            frame_review = output_dir / "npec_frames_needing_review.csv"
            frame_review.write_text("plant_id,FrameIndex,status\n", encoding="utf-8")
            identity_state = output_dir / "npec_seedling_identity_state.json"
            identity_state.write_text('{"tracks":[]}\n', encoding="utf-8")
            missing_video = output_dir / "npec_root_growth_analysis_video.mp4"
            summary = output_dir / "npec_root_growth_video_summary.csv"
            summary.write_text("frames\n", encoding="utf-8")

            window = _FakeLazyOutputWindow()
            window._set_pipeline_lazy_latest_outputs(
                output_dir=output_dir,
                workbook=workbook,
                timelapse_csv=timelapse,
                track_summary_csv=track_summary,
                review_csv=review,
                frame_review_csv=frame_review,
                identity_state_json=identity_state,
                video=missing_video,
                video_summary=summary,
            )

            self.assertEqual(window.pipeline_lazy_outputs_label.text, str(output_dir))
            self.assertTrue(window.pipeline_lazy_open_output_folder_btn.enabled)
            self.assertTrue(window.pipeline_lazy_open_workbook_btn.enabled)
            self.assertTrue(window.pipeline_lazy_open_timelapse_btn.enabled)
            self.assertTrue(window.pipeline_lazy_open_track_summary_btn.enabled)
            self.assertTrue(window.pipeline_lazy_open_review_btn.enabled)
            self.assertTrue(window.pipeline_lazy_open_frame_review_btn.enabled)
            self.assertTrue(window.pipeline_lazy_open_identity_state_btn.enabled)
            self.assertFalse(window.pipeline_lazy_open_video_btn.enabled)
            self.assertTrue(window.pipeline_lazy_open_video_summary_btn.enabled)

    def test_lazy_output_discovery_finds_combined_root_shoot_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lazy_outputs_discovery_") as tmp_dir_name:
            output_dir = Path(tmp_dir_name)
            workbook = output_dir / "npec_total_root_and_shoot_all_metrics.xlsx"
            workbook.write_text("workbook", encoding="utf-8")
            timelapse = output_dir / "npec_total_root_and_shoot_timelapse.csv"
            timelapse.write_text("frame,total\n", encoding="utf-8")
            track_summary = output_dir / "npec_total_root_and_shoot_from_masks_summary.csv"
            track_summary.write_text("PetriDish,total\n", encoding="utf-8")
            frame_review = output_dir / "npec_mask_total_frames_needing_review.csv"
            frame_review.write_text("PetriDish,FrameIndex,mask_total_review_status\n", encoding="utf-8")
            video = output_dir / "npec_root_and_shoot_masks_analysis_video.mp4"
            video.write_bytes(b"mp4")
            summary = output_dir / "npec_root_growth_video_summary.csv"
            summary.write_text("frames\n", encoding="utf-8")

            window = _FakeLazyOutputWindow()
            discovered = window._discover_pipeline_lazy_outputs(output_dir)
            window._set_pipeline_lazy_latest_outputs(**discovered)

            self.assertEqual(discovered["output_dir"], output_dir)
            self.assertEqual(discovered["workbook"], workbook)
            self.assertEqual(discovered["timelapse_csv"], timelapse)
            self.assertEqual(discovered["track_summary_csv"], track_summary)
            self.assertNotIn("review_csv", discovered)
            self.assertEqual(discovered["frame_review_csv"], frame_review)
            self.assertEqual(discovered["video"], video)
            self.assertEqual(discovered["video_summary"], summary)
            self.assertTrue(window.pipeline_lazy_open_workbook_btn.enabled)
            self.assertTrue(window.pipeline_lazy_open_timelapse_btn.enabled)
            self.assertTrue(window.pipeline_lazy_open_track_summary_btn.enabled)
            self.assertFalse(window.pipeline_lazy_open_review_btn.enabled)
            self.assertTrue(window.pipeline_lazy_open_frame_review_btn.enabled)
            self.assertFalse(window.pipeline_lazy_open_identity_state_btn.enabled)
            self.assertTrue(window.pipeline_lazy_open_video_btn.enabled)
            self.assertTrue(window.pipeline_lazy_open_video_summary_btn.enabled)

    def test_lazy_output_discovery_prefers_occlusion_aware_combined_workbook(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lazy_combined_discovery_") as tmp_dir_name:
            output_dir = Path(tmp_dir_name)
            plate_workbook = output_dir / "npec_total_root_and_shoot_all_metrics.xlsx"
            ownership_workbook = output_dir / "npec_per_plant_all_metrics.xlsx"
            combined_workbook = output_dir / "npec_occlusion_aware_all_assets.xlsx"
            for path in (plate_workbook, ownership_workbook, combined_workbook):
                path.write_bytes(b"workbook")
            occlusion_timelapse = output_dir / "npec_occlusion_aware_whole_plate_timelapse.csv"
            occlusion_timelapse.write_text("PetriDish,FrameIndex\n", encoding="utf-8")
            plate_timelapse = output_dir / "npec_total_root_and_shoot_timelapse.csv"
            plate_timelapse.write_text("PetriDish,FrameIndex\n", encoding="utf-8")
            ownership_review = output_dir / "npec_frames_needing_review.csv"
            ownership_review.write_text("plant_id,FrameIndex\n", encoding="utf-8")
            mask_review = output_dir / "npec_mask_total_frames_needing_review.csv"
            mask_review.write_text("PetriDish,FrameIndex\n", encoding="utf-8")

            window = _FakeLazyOutputWindow()
            discovered = window._discover_pipeline_lazy_outputs(output_dir)

            self.assertEqual(discovered["workbook"], combined_workbook)
            self.assertEqual(discovered["timelapse_csv"], occlusion_timelapse)
            self.assertEqual(discovered["frame_review_csv"], ownership_review)

    def test_lazy_output_discovery_finds_potato_endpoint_package(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lazy_potato_endpoint_discovery_") as tmp_dir_name:
            output_dir = Path(tmp_dir_name)
            expected = {
                "workbook": output_dir / "npec_potato_endpoint_all_metrics.xlsx",
                "timelapse_csv": output_dir / "npec_potato_endpoint_phenotypes.csv",
                "track_summary_csv": output_dir / "npec_potato_branch_traits.csv",
                "review_csv": output_dir / "npec_potato_qc.csv",
                "frame_review_csv": output_dir / "npec_potato_plate_label_map.csv",
                "identity_state_json": output_dir / "npec_potato_analysis_provenance.json",
                "video": output_dir / "npec_potato_endpoint_overlay_reel.mp4",
                "video_summary": output_dir / "npec_potato_finalization_summary.json",
            }
            for path in expected.values():
                path.write_bytes(b"endpoint-output")

            window = _FakeLazyOutputWindow()
            discovered = window._discover_pipeline_lazy_outputs(output_dir)

            self.assertEqual(discovered["output_dir"], output_dir)
            for key, path in expected.items():
                self.assertEqual(discovered[key], path)

    def test_root_shoot_video_alias_is_refreshed_from_generated_video(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lazy_video_alias_") as tmp_dir_name:
            output_dir = Path(tmp_dir_name)
            generated = output_dir / "npec_root_growth_analysis_video.mp4"
            alias = output_dir / "npec_root_and_shoot_masks_analysis_video.mp4"
            generated.write_bytes(b"fresh green-only video")
            alias.write_bytes(b"stale raw class video")

            refreshed = NpecLabelingMainWindow._refresh_pipeline_lazy_root_shoot_video_alias(generated)

            self.assertEqual(refreshed, alias)
            self.assertEqual(alias.read_bytes(), b"fresh green-only video")

    def test_lazy_output_discovery_falls_back_to_ownership_and_compat_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lazy_outputs_discovery_fallback_") as tmp_dir_name:
            output_dir = Path(tmp_dir_name)
            workbook = output_dir / "npec_per_plant_all_metrics.xlsx"
            workbook.write_text("workbook", encoding="utf-8")
            timelapse = output_dir / "npec_total_root_length_timelapse.csv"
            timelapse.write_text("frame,total\n", encoding="utf-8")
            track_summary = output_dir / "npec_per_plant_track_summary.csv"
            track_summary.write_text("plant_id,status\n", encoding="utf-8")
            review = output_dir / "npec_tracks_needing_review.csv"
            review.write_text("plant_id,status\n", encoding="utf-8")
            frame_review = output_dir / "npec_frames_needing_review.csv"
            frame_review.write_text("plant_id,FrameIndex,status\n", encoding="utf-8")
            identity_state = output_dir / "npec_seedling_identity_state.json"
            identity_state.write_text('{"tracks":[]}\n', encoding="utf-8")
            video = output_dir / "npec_root_growth_analysis_video.mp4"
            video.write_bytes(b"mp4")

            window = _FakeLazyOutputWindow()
            discovered = window._discover_pipeline_lazy_outputs(output_dir)

            self.assertEqual(discovered["output_dir"], output_dir)
            self.assertEqual(discovered["workbook"], workbook)
            self.assertEqual(discovered["timelapse_csv"], timelapse)
            self.assertEqual(discovered["track_summary_csv"], track_summary)
            self.assertEqual(discovered["review_csv"], review)
            self.assertEqual(discovered["frame_review_csv"], frame_review)
            self.assertEqual(discovered["identity_state_json"], identity_state)
            self.assertEqual(discovered["video"], video)
            self.assertNotIn("video_summary", discovered)

    def test_existing_lazy_mask_records_match_sources_and_fallback_to_mask_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lazy_existing_masks_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            source_root = tmp_dir / "PNGs"
            source_plate = source_root / "B3-3"
            source_plate.mkdir(parents=True)
            source_image = source_plate / "plate_B3-3_t001.png"
            source_image.write_bytes(b"source")

            output_dir = source_root / "segmentation_masks"
            output_plate = output_dir / "B3-3"
            output_plate.mkdir(parents=True)
            matched_mask = output_plate / "plate_B3-3_t001_mask.png"
            matched_mask.write_bytes(b"mask")
            orphan_mask = output_dir / "orphan_t002_mask.png"
            orphan_mask.write_bytes(b"mask")

            window = _FakeLazyOutputWindow()
            records, warnings = window._collect_existing_lazy_mask_records(source_root, output_dir)

            self.assertEqual(len(records), 2)
            by_mask = {Path(str(record["output_mask"])).name: record for record in records}
            matched = by_mask[matched_mask.name]
            orphan = by_mask[orphan_mask.name]
            self.assertEqual(Path(str(matched["image_path"])), source_image)
            self.assertEqual(Path(str(orphan["image_path"])), orphan_mask)
            self.assertIn("B3-3", str(matched["meta"]["petri"]))
            self.assertEqual(int(matched["meta"]["frame_index"]), 1)
            self.assertEqual(int(orphan["meta"]["frame_index"]), 2)
            self.assertTrue(any("mask-only fallback" in warning for warning in warnings))

    def test_existing_lazy_mask_records_follow_plate_identity_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lazy_manifest_masks_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            source_root = tmp_dir / "unrenamed"
            source_root.mkdir()
            source_image = source_root / "IMG_0007.png"
            source_image.write_bytes(b"source")

            output_dir = source_root / "segmentation_masks"
            plate_id = "152_RP043_g23"
            plate_dir = output_dir / plate_id
            plate_dir.mkdir(parents=True)
            allocated_name = f"{plate_id}__20240715_093410.png"
            mask_path = plate_dir / f"{Path(allocated_name).stem}_mask.png"
            mask_path.write_bytes(b"mask")
            write_plate_identity_manifest(
                [
                    {
                        "source_name": source_image.name,
                        "source_path": str(source_image),
                        "source_date": "2024-07-15T09:34:10",
                        "source_date_source": "filename",
                        "plate_id": plate_id,
                        "allocated_name": allocated_name,
                        "plate_frame_index": 1,
                        "confidence": 0.99,
                        "status": "accepted",
                        "user_override": None,
                        "scanner_version": 1,
                        "evidence": [{"kind": "vision_text", "text": "152 RP043 g23"}],
                    }
                ],
                output_dir / "npec_plate_identity_manifest.csv",
            )

            window = _FakeLazyOutputWindow()
            records, warnings = window._collect_existing_lazy_mask_records(source_root, output_dir)

            self.assertEqual(warnings, [])
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(Path(str(record["image_path"])), source_image)
            self.assertEqual(record["image_name"], allocated_name)
            self.assertEqual(record["meta"]["series"], plate_id)
            self.assertEqual(record["meta"]["petri"], plate_id)
            self.assertEqual(record["meta"]["allocated_file_name"], allocated_name)
            self.assertEqual(int(record["meta"]["frame_index"]), 0)

    def test_lucifer_folder_name_does_not_override_segmentation_shoot_source(self) -> None:
        window = _FakeLazyOutputWindow()

        enabled = window._lazy_shoot_rgb_rescue_for_run(
            input_dir=Path("/data/lucifer/plate_images"),
            output_dir=Path("/data/lucifer/plate_images/segmentation_masks"),
        )

        self.assertFalse(enabled)

    def test_explicit_crown_green_mode_enables_lazy_shoot_recompute(self) -> None:
        window = _FakeLazyOutputWindow()
        window.pyphenotyper_shoot_color_rescue_mode = "lucifer_green"

        enabled = window._lazy_shoot_rgb_rescue_for_run(
            input_dir=Path("/tmp/new_rgb_dataset"),
            output_dir=Path("/tmp/new_rgb_masks"),
        )

        self.assertTrue(enabled)

    def test_shoot_rescue_off_overrides_lucifer_lazy_auto_detection(self) -> None:
        window = _FakeLazyOutputWindow()
        window.pyphenotyper_shoot_color_rescue_mode = "off"

        enabled = window._lazy_shoot_rgb_rescue_for_run(
            input_dir=Path("/data/lucifer/plate_images"),
            output_dir=Path("/data/lucifer/plate_images/segmentation_masks"),
        )

        self.assertFalse(enabled)

    def test_latest_lazy_output_actions_show_empty_state_without_paths(self) -> None:
        window = _FakeLazyOutputWindow()
        window._set_pipeline_lazy_latest_outputs()

        self.assertEqual(window.pipeline_lazy_outputs_label.text, "No Lazy run yet")
        self.assertFalse(window.pipeline_lazy_open_output_folder_btn.enabled)
        self.assertFalse(window.pipeline_lazy_open_workbook_btn.enabled)
        self.assertFalse(window.pipeline_lazy_open_timelapse_btn.enabled)
        self.assertFalse(window.pipeline_lazy_open_track_summary_btn.enabled)
        self.assertFalse(window.pipeline_lazy_open_review_btn.enabled)
        self.assertFalse(window.pipeline_lazy_open_frame_review_btn.enabled)
        self.assertFalse(window.pipeline_lazy_open_identity_state_btn.enabled)
        self.assertFalse(window.pipeline_lazy_open_video_btn.enabled)
        self.assertFalse(window.pipeline_lazy_open_video_summary_btn.enabled)


if __name__ == "__main__":
    unittest.main()
