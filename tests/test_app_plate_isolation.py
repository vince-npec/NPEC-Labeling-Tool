from __future__ import annotations

from types import SimpleNamespace
import unittest

from resources.app import NpecLabelingMainWindow, TIME_SERIES_ALL_KEY


class _FakePlateScopeWindow:
    _series_name_for_uid = NpecLabelingMainWindow._series_name_for_uid
    _current_series_filter_key = NpecLabelingMainWindow._current_series_filter_key
    _analytics_items_from_selection = NpecLabelingMainWindow._analytics_items_from_selection
    _timelapse_segment_state = NpecLabelingMainWindow._timelapse_segment_state

    def __init__(self) -> None:
        self.dataset_items = [
            SimpleNamespace(uid="a1"),
            SimpleNamespace(uid="a2"),
            SimpleNamespace(uid="b1"),
            SimpleNamespace(uid="b2"),
        ]
        self.series_by_uid = {"a1": "plate_A", "a2": "plate_A", "b1": "plate_B", "b2": "plate_B"}
        self.series_filter_key = TIME_SERIES_ALL_KEY
        self.current_index = 2
        self.timelapse_index = 1
        self.timelapse_interp_subframe = 2

    def _selected_dataset_items(self):
        return []

    def _current_item(self):
        return self.dataset_items[self.current_index]

    def _timelapse_effective_interp_steps(self):
        return 3


class AppPlateIsolationTests(unittest.TestCase):
    def test_dataset_analytics_uses_current_plate_when_multiple_series_are_loaded(self) -> None:
        window = _FakePlateScopeWindow()

        items, label = window._analytics_items_from_selection()

        self.assertEqual([item.uid for item in items], ["b1", "b2"])
        self.assertIn("plate series plate_B", label)

    def test_series_filter_controls_analytics_scope(self) -> None:
        window = _FakePlateScopeWindow()
        window.series_filter_key = "plate_A"

        items, label = window._analytics_items_from_selection()

        self.assertEqual([item.uid for item in items], ["a1", "a2"])
        self.assertEqual(label, "time series plate_A")

    def test_timelapse_never_interpolates_across_plate_boundary(self) -> None:
        window = _FakePlateScopeWindow()

        base_index, next_index, alpha, steps = window._timelapse_segment_state()

        self.assertEqual(base_index, 1)
        self.assertIsNone(next_index)
        self.assertEqual(alpha, 0.0)
        self.assertEqual(steps, 0)
        self.assertEqual(window.timelapse_interp_subframe, 0)


if __name__ == "__main__":
    unittest.main()
