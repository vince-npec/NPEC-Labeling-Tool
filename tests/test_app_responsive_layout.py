"""Offscreen regression coverage for the responsive main-window layout.

The tests exercise the real main window rather than a reduced UI fixture.  They
intentionally describe the supported layout contract at 1280x720, 1440x900,
and 1920x1080, so a child widget with an oversized minimum size cannot make a
small viewport appear to pass.

Stable lookup contract for future UI refactors:

* Existing ``NpecLabelingMainWindow`` attributes are used first.
* A refactor may instead expose the object names listed in ``WIDGET_LOOKUPS``.
* Resizing the window must be enough to apply responsive layout; the tests do
  not call a private reflow helper.

These are Qt integration tests.  They skip only when PySide6 itself is not
installed; missing application runtime dependencies remain real failures.
"""

from __future__ import annotations

import os
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QPoint, QRect, Qt
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDockWidget,
        QPushButton,
        QStyle,
        QStyleOptionButton,
        QStyleOptionComboBox,
        QTabBar,
        QWidget,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by minimal CI jobs
    raise unittest.SkipTest("PySide6 is required for responsive UI tests") from exc

from resources.app import NpecLabelingMainWindow


VIEWPORTS = ((1024, 700), (1280, 720), (1440, 900), (1920, 1080))

WIDGET_LOOKUPS: dict[str, tuple[str, str, type[QWidget]]] = {
    "dataset_panel": ("_dataset_panel_widget", "datasetPanel", QWidget),
    "layers_dock": ("layers_dock", "layersDock", QDockWidget),
    "module_navigation": ("tabs_nav_widget", "moduleNavigation", QWidget),
    "workflow_sections": ("workflow_tabs", "workflowTabs", QTabBar),
    "module_combo": ("tab_module_combo", "tabModuleCombo", QComboBox),
    "tabs": ("tabs", "moduleTabs", QWidget),
}

LEFT_BUTTONS = (
    ("load_folder_btn", "loadFolderButton"),
    ("load_hades_data_btn", "loadHadesButton"),
    ("load_images_masks_btn", "loadImagesMasksButton"),
    ("load_files_btn", "loadFilesButton"),
    ("rotate_left_btn", "rotateLeftButton"),
    ("rotate_right_btn", "rotateRightButton"),
    ("remove_dataset_btn", "removeDatasetButton"),
    ("clear_dataset_btn", "clearDatasetButton"),
    ("save_project_btn", "saveProjectButton"),
    ("load_project_btn", "loadProjectButton"),
    ("load_template_btn", "loadTemplateButton"),
    ("load_timeseries_btn", "loadTimeSeriesButton"),
    ("scan_plate_identity_btn", "reviewPlateIdsButton"),
)

RIGHT_BUTTONS = (
    ("add_class_btn", "addClassButton"),
    ("remove_class_btn", "removeClassButton"),
    ("rename_class_btn", "renameClassButton"),
    ("color_class_btn", "colorClassButton"),
    ("import_layer_btn", "importLayerMaskButton"),
    ("merge_layers_btn", "mergeLayersButton"),
    ("replicate_layers_btn", "replicateMasksButton"),
)


class ResponsiveMainWindowLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls.window = NpecLabelingMainWindow()
        cls.window.show()
        cls._settle_events()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.window.hide()
        cls.window.deleteLater()
        cls._settle_events()

    @classmethod
    def _settle_events(cls) -> None:
        # Two passes flush resize/layout requests posted during the first pass.
        cls.app.processEvents()
        cls.app.sendPostedEvents()
        cls.app.processEvents()

    def _resize_to(self, width: int, height: int) -> None:
        self.window.resize(width, height)
        self._settle_events()

    def _named_widget(self, key: str) -> QWidget:
        attribute, object_name, widget_type = WIDGET_LOOKUPS[key]
        widget = getattr(self.window, attribute, None)
        if widget is None:
            widget = self.window.findChild(widget_type, object_name)
        self.assertIsNotNone(
            widget,
            f"Responsive layout must expose {attribute!r} or objectName={object_name!r}",
        )
        return widget

    def _button(self, attribute: str, object_name: str) -> QPushButton:
        button = getattr(self.window, attribute, None)
        if button is None:
            button = self.window.findChild(QPushButton, object_name)
        self.assertIsNotNone(
            button,
            f"Responsive layout must expose {attribute!r} or objectName={object_name!r}",
        )
        return button

    def _window_rect_for(self, widget: QWidget) -> QRect:
        origin = widget.mapTo(self.window, QPoint(0, 0))
        return QRect(origin, widget.size())

    def _assert_inside_viewport(self, widget: QWidget, width: int, height: int) -> None:
        rect = self._window_rect_for(widget)
        viewport = QRect(0, 0, width, height)
        self.assertTrue(
            viewport.contains(rect),
            f"{widget.objectName() or type(widget).__name__} lies outside {width}x{height}: {rect}",
        )

    def _assert_button_text_fits(self, button: QPushButton) -> None:
        self.assertTrue(button.isVisible(), f"{button.text()!r} button is hidden")
        self.assertGreater(button.width(), 0, f"{button.text()!r} button has no width")
        option = QStyleOptionButton()
        option.initFrom(button)
        option.text = button.text()
        contents = button.style().subElementRect(
            QStyle.SubElement.SE_PushButtonContents,
            option,
            button,
        )
        required = button.fontMetrics().horizontalAdvance(button.text())
        self.assertLessEqual(
            required,
            contents.width(),
            f"Button text is clipped: {button.text()!r} needs {required}px, has {contents.width()}px",
        )

    def _assert_combo_items_fit(self, combo: QComboBox) -> None:
        option = QStyleOptionComboBox()
        option.initFrom(combo)
        contents = combo.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            option,
            QStyle.SubControl.SC_ComboBoxEditField,
            combo,
        )
        for index in range(combo.count()):
            text = combo.itemText(index)
            required = combo.fontMetrics().horizontalAdvance(text)
            self.assertLessEqual(
                required,
                contents.width(),
                f"Combo item is clipped: {text!r} needs {required}px, has {contents.width()}px",
            )

    def _assert_tab_labels_fit(self, tab_bar: QTabBar) -> None:
        for index in range(tab_bar.count()):
            text = tab_bar.tabText(index)
            available = tab_bar.tabRect(index).width() - 20
            required = tab_bar.fontMetrics().horizontalAdvance(text)
            self.assertLessEqual(
                required,
                available,
                f"Workflow section is clipped: {text!r} needs {required}px, has {available}px",
            )

    def test_window_honors_each_supported_viewport(self) -> None:
        """No child minimum-size hint may silently enlarge the main window."""

        for width, height in VIEWPORTS:
            with self.subTest(viewport=f"{width}x{height}"):
                self._resize_to(width, height)
                actual = self.window.size()
                self.assertLessEqual(
                    actual.width(),
                    width + 8,
                    f"Minimum-width pressure expanded {width}px viewport to {actual.width()}px",
                )
                self.assertLessEqual(
                    actual.height(),
                    height + 8,
                    f"Minimum-height pressure expanded {height}px viewport to {actual.height()}px",
                )

    def test_sidebars_and_controls_stay_inside_each_viewport(self) -> None:
        """Essential sidebar commands must remain inside the requested client area."""

        for width, height in VIEWPORTS:
            with self.subTest(viewport=f"{width}x{height}"):
                self._resize_to(width, height)
                dataset_panel = self._named_widget("dataset_panel")
                layers_dock = self._named_widget("layers_dock")
                self._assert_inside_viewport(dataset_panel, width, height)
                self._assert_inside_viewport(layers_dock, width, height)

                self.assertGreaterEqual(dataset_panel.width(), 220)
                self.assertLessEqual(dataset_panel.width(), int(width * 0.32))

                for attribute, object_name in LEFT_BUTTONS + RIGHT_BUTTONS:
                    with self.subTest(control=attribute):
                        button = self._button(attribute, object_name)
                        self._assert_inside_viewport(button, width, height)

    def test_sidebar_control_labels_are_not_clipped(self) -> None:
        """Text must fit the style-provided content rectangle at every density."""

        for width, height in VIEWPORTS:
            with self.subTest(viewport=f"{width}x{height}"):
                self._resize_to(width, height)
                for attribute, object_name in LEFT_BUTTONS + RIGHT_BUTTONS:
                    with self.subTest(control=attribute):
                        button = self._button(attribute, object_name)
                        self._assert_button_text_fits(button)

                identity_checkbox = getattr(self.window, "auto_plate_identity_cb", None)
                if isinstance(identity_checkbox, QCheckBox):
                    self.assertTrue(identity_checkbox.isVisible())
                    self.assertLessEqual(
                        identity_checkbox.sizeHint().width(),
                        identity_checkbox.width() + 2,
                        "Plate-ID import option is clipped in the dataset sidebar",
                    )

    def test_layers_panel_stays_narrow_and_preserves_workspace(self) -> None:
        """The layers dock is a utility rail, not a second primary workspace."""

        for width, height in VIEWPORTS:
            with self.subTest(viewport=f"{width}x{height}"):
                self._resize_to(width, height)
                dock = self._named_widget("layers_dock")
                dataset_panel = self._named_widget("dataset_panel")
                self.assertLessEqual(
                    dock.width(),
                    int(width * 0.22),
                    f"Layers dock consumes {dock.width()}px of a {width}px viewport",
                )
                remaining = width - dataset_panel.width() - dock.width()
                self.assertGreaterEqual(
                    remaining,
                    int(width * 0.50),
                    "Sidebars leave less than half the viewport for the active module",
                )

    def test_every_module_is_reachable_through_responsive_navigation(self) -> None:
        """Workflow sections and the module selector must reach every module."""

        for width, height in VIEWPORTS:
            with self.subTest(viewport=f"{width}x{height}"):
                self._resize_to(width, height)
                nav = self._named_widget("module_navigation")
                workflow_sections = self._named_widget("workflow_sections")
                module_combo = self._named_widget("module_combo")
                tabs = self._named_widget("tabs")

                self.assertTrue(nav.isVisible(), "Responsive module navigation is hidden")
                self.assertTrue(workflow_sections.isVisible(), "Workflow section navigation is hidden")
                self.assertTrue(module_combo.isVisible(), "Module selector is hidden")
                self._assert_inside_viewport(nav, width, height)
                self._assert_inside_viewport(workflow_sections, width, height)
                self._assert_inside_viewport(module_combo, width, height)
                self._assert_tab_labels_fit(workflow_sections)

                reached: set[int] = set()
                for section_index in range(workflow_sections.count()):
                    workflow_sections.setCurrentIndex(section_index)
                    self._settle_events()
                    self._assert_combo_items_fit(module_combo)
                    for module_index in range(module_combo.count()):
                        target = module_combo.itemData(module_index, Qt.ItemDataRole.UserRole)
                        self.assertIsNotNone(target, "Module selector item has no tab index")
                        module_combo.setCurrentIndex(module_index)
                        self._settle_events()
                        self.assertEqual(
                            tabs.currentIndex(),
                            int(target),
                            f"Selecting module {module_combo.itemText(module_index)!r} did not activate its tab",
                        )
                        reached.add(int(target))

                self.assertEqual(
                    reached,
                    set(range(tabs.count())),
                    "Responsive navigation does not expose every application module",
                )


if __name__ == "__main__":
    unittest.main()
