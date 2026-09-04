from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .canvas import SegmentationCanvas
from .guided_root_rescue import GuidedRootRescueConfig, compute_guided_root_rescue
from .models import LabelClass, hex_to_rgb


def _is_root_class(label_class: LabelClass) -> bool:
    name = str(label_class.name or "").lower()
    return any(token in name for token in ("root", "lateral", "primary", "adventitious", "radicle"))


class GuidedRootRescueDialog(QDialog):
    def __init__(
        self,
        image_rgb: np.ndarray,
        existing_layers: dict[int, np.ndarray],
        classes: list[LabelClass],
        *,
        initial_class_id: int | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Guided Root Rescue")
        self.resize(1280, 860)
        self._image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        self._layers = {
            int(class_id): (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
            for class_id, mask in existing_layers.items()
        }
        self._classes = list(classes)
        self._class_by_id = {int(label_class.class_id): label_class for label_class in self._classes}
        self._routes: list[dict[str, object]] = []
        self._current_points: list[tuple[int, int]] = []
        self._addition_masks: dict[int, np.ndarray] = {}
        self._metadata: dict[str, object] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Class"))
        self.class_combo = QComboBox(self)
        root_classes = [label_class for label_class in self._classes if _is_root_class(label_class)]
        for label_class in root_classes or self._classes:
            self.class_combo.addItem(f"{label_class.class_id}: {label_class.name}", int(label_class.class_id))
        if initial_class_id is not None:
            initial_index = self.class_combo.findData(int(initial_class_id))
            if initial_index >= 0:
                self.class_combo.setCurrentIndex(initial_index)
        controls.addWidget(self.class_combo)
        controls.addWidget(QLabel("Width"))
        self.width_spin = QSpinBox(self)
        self.width_spin.setRange(1, 21)
        self.width_spin.setValue(3)
        self.width_spin.setSuffix(" px")
        controls.addWidget(self.width_spin)
        controls.addWidget(QLabel("Search corridor"))
        self.corridor_spin = QSpinBox(self)
        self.corridor_spin.setRange(4, 240)
        self.corridor_spin.setValue(36)
        self.corridor_spin.setSuffix(" px")
        controls.addWidget(self.corridor_spin)
        self.new_route_btn = QPushButton("Add Route", self)
        self.new_route_btn.setToolTip("Keep the current polyline and start another class-assigned route.")
        self.undo_point_btn = QPushButton("Undo Point", self)
        self.clear_btn = QPushButton("Clear", self)
        self.preview_btn = QPushButton("Preview", self)
        controls.addWidget(self.new_route_btn)
        controls.addWidget(self.undo_point_btn)
        controls.addWidget(self.clear_btn)
        controls.addWidget(self.preview_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.canvas = SegmentationCanvas(editable=False, parent=self)
        self.canvas.set_image(self._base_overlay())
        self.canvas.set_point_select_mode(True)
        self.canvas.point_selected.connect(self._add_point)
        layout.addWidget(self.canvas, 1)

        self.status_label = QLabel("Routes: 0 · Current points: 0", self)
        self.status_label.setStyleSheet("color: #9eb2d8;")
        layout.addWidget(self.status_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel, self)
        self.buttons.button(QDialogButtonBox.Apply).setText("Apply Rescue")
        self.buttons.accepted.connect(self._apply_and_accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.new_route_btn.clicked.connect(self._commit_current_route)
        self.undo_point_btn.clicked.connect(self._undo_point)
        self.clear_btn.clicked.connect(self._clear_routes)
        self.preview_btn.clicked.connect(self._preview)
        self.class_combo.currentIndexChanged.connect(lambda _index=0: self._refresh_guides())

    def _base_overlay(self) -> np.ndarray:
        display = self._image_rgb.astype(np.float32).copy()
        for label_class in self._classes:
            mask = self._layers.get(int(label_class.class_id))
            if mask is None or not np.any(mask):
                continue
            color = np.asarray(hex_to_rgb(label_class.color_hex), dtype=np.float32)
            selected = np.asarray(mask, dtype=bool)
            display[selected] = display[selected] * 0.45 + color * 0.55
        return np.clip(display, 0, 255).astype(np.uint8)

    def _current_class_id(self) -> int:
        value = self.class_combo.currentData()
        return int(value) if value is not None else 1

    def _class_color(self, class_id: int) -> tuple[int, int, int]:
        label_class = self._class_by_id.get(int(class_id))
        return hex_to_rgb(label_class.color_hex) if label_class is not None else (255, 198, 88)

    def _routes_with_current(self, *, require_complete: bool) -> list[dict[str, object]]:
        routes = [dict(route) for route in self._routes]
        if len(self._current_points) >= 2:
            routes.append({"class_id": self._current_class_id(), "points": list(self._current_points)})
        elif require_complete and self._current_points:
            raise ValueError("The current route needs at least two points.")
        return routes

    def _refresh_guides(self) -> None:
        routes = [dict(route) for route in self._routes]
        if self._current_points:
            routes.append({"class_id": self._current_class_id(), "points": list(self._current_points)})
        canvas_routes: list[dict[str, object]] = []
        for route in routes:
            class_id = int(route.get("class_id", 1))
            canvas_routes.append(
                {
                    "points": list(route.get("points", [])),
                    "color": self._class_color(class_id),
                }
            )
        self.canvas.set_guide_routes(canvas_routes)
        self.status_label.setText(
            f"Routes: {len(self._routes)} · Current points: {len(self._current_points)}"
        )

    def _add_point(self, point: object) -> None:
        try:
            x, y = point  # type: ignore[misc]
            parsed = (int(x), int(y))
        except Exception:
            return
        if not self._current_points or self._current_points[-1] != parsed:
            self._current_points.append(parsed)
        self._addition_masks = {}
        self._refresh_guides()

    def _commit_current_route(self) -> None:
        if len(self._current_points) < 2:
            QMessageBox.information(self, "Route incomplete", "Add at least two points before starting another route.")
            return
        self._routes.append({"class_id": self._current_class_id(), "points": list(self._current_points)})
        self._current_points = []
        self._addition_masks = {}
        self._refresh_guides()

    def _undo_point(self) -> None:
        if self._current_points:
            self._current_points.pop()
        elif self._routes:
            restored = self._routes.pop()
            self._current_points = [tuple(point) for point in restored.get("points", [])]  # type: ignore[misc]
            class_index = self.class_combo.findData(int(restored.get("class_id", self._current_class_id())))
            if class_index >= 0:
                self.class_combo.setCurrentIndex(class_index)
        self._addition_masks = {}
        self._refresh_guides()

    def _clear_routes(self) -> None:
        self._routes = []
        self._current_points = []
        self._addition_masks = {}
        self._metadata = {}
        self.canvas.set_image(self._base_overlay())
        self.canvas.set_point_select_mode(True)
        self._refresh_guides()

    def _existing_root_support(self) -> np.ndarray:
        support = np.zeros(self._image_rgb.shape[:2], dtype=np.uint8)
        root_ids = {int(label_class.class_id) for label_class in self._classes if _is_root_class(label_class)}
        if not root_ids:
            root_ids = set(self._layers)
        for class_id in root_ids:
            layer = self._layers.get(class_id)
            if layer is not None:
                support[np.asarray(layer, dtype=np.uint8) > 0] = 1
        return support

    def _preview(self) -> bool:
        try:
            routes = self._routes_with_current(require_complete=True)
        except ValueError as exc:
            QMessageBox.information(self, "Route incomplete", str(exc))
            return False
        if not routes:
            QMessageBox.information(self, "No routes", "Add a crown-to-tip or branch route first.")
            return False
        try:
            result = compute_guided_root_rescue(
                self._image_rgb,
                self._existing_root_support(),
                routes,
                GuidedRootRescueConfig(
                    line_width_px=int(self.width_spin.value()),
                    corridor_px=int(self.corridor_spin.value()),
                ),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Guided rescue failed", str(exc))
            return False
        self._addition_masks = {
            int(class_id): np.asarray(mask, dtype=np.uint8)
            for class_id, mask in result.addition_masks.items()
        }
        self._metadata = dict(result.metadata)
        preview = self._base_overlay().astype(np.float32)
        for class_id, mask in self._addition_masks.items():
            selected = np.asarray(mask, dtype=bool)
            color = np.asarray(self._class_color(class_id), dtype=np.float32)
            preview[selected] = preview[selected] * 0.20 + color * 0.80
        self.canvas.set_image(np.clip(preview, 0, 255).astype(np.uint8))
        self.canvas.set_point_select_mode(True)
        self._refresh_guides()
        added = sum(int(np.count_nonzero(mask)) for mask in self._addition_masks.values())
        self.status_label.setText(
            f"Routes: {len(routes)} · Added pixels: {added} · Routed segments: {result.metadata.get('routed_segment_count', 0)}"
        )
        return added > 0

    def _apply_and_accept(self) -> None:
        if not self._preview():
            return
        self.accept()

    def addition_masks(self) -> dict[int, np.ndarray]:
        return {class_id: mask.copy() for class_id, mask in self._addition_masks.items()}

    def rescue_metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    def route_class_order(self) -> list[int]:
        routes = self._routes_with_current(require_complete=False)
        return [int(route.get("class_id", 1)) for route in routes]
