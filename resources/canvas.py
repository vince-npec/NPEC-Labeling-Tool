from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QTabletEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from .mask_ops import rasterize_stroke


def numpy_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        h, w = image.shape
        data = np.ascontiguousarray(image)
        return QImage(data.data, w, h, data.strides[0], QImage.Format_Grayscale8).copy()

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape for display: {image.shape}")

    h, w, _ = image.shape
    data = np.ascontiguousarray(image)
    return QImage(data.data, w, h, data.strides[0], QImage.Format_RGB888).copy()


class SegmentationCanvas(QGraphicsView):
    stroke_applied = Signal(object, str)
    stroke_finished = Signal()
    mask_nudged = Signal(int, int)
    mask_nudge_finished = Signal()
    brush_size_adjust = Signal(int)
    zoom_changed = Signal(float)
    roi_selected = Signal(object)
    point_selected = Signal(object)

    def __init__(self, editable: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._editable = editable
        self._tool = "brush"
        self._brush_size = 14
        self._brush_color = QColor(255, 90, 95, 230)
        self._image_shape = (0, 0)
        self._drawing = False
        self._move_dragging = False
        self._panning = False
        self._space_held = False
        self._last_point: tuple[int, int] | None = None
        self._move_last_scene_pos: QPointF | None = None
        self._stroke_points: list[tuple[int, int]] = []
        self._smoothed_last_point: tuple[int, int] | None = None
        self._smoothing_alpha = 0.72
        self._tablet_pressure = 1.0
        self._use_tablet_pressure = True
        self._hover_scene_pos: QPointF | None = None
        self._zoom_factor = 1.0
        self._wheel_zoom_base_step = 1.06
        self._auto_fit = True
        self._cursor_cache: dict[tuple[str, int], QCursor] = {}
        self._roi_select_mode = False
        self._roi_dragging = False
        self._roi_start_scene: QPointF | None = None
        self._roi_current_scene: QPointF | None = None
        self._roi_rect: tuple[int, int, int, int] | None = None
        self._point_select_mode = False
        self._guide_routes: list[dict[str, object]] = []

        self.setScene(QGraphicsScene(self))
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setFrameShape(QFrame.NoFrame)
        self.setBackgroundBrush(QColor(14, 17, 26))
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_TabletTracking, True)
        self._update_cursor()

    def set_editable(self, editable: bool) -> None:
        self._editable = editable
        self._update_cursor()
        self.viewport().update()

    def set_tool(self, tool: str) -> None:
        self._tool = tool.lower()
        if self._tool != "move":
            self._move_dragging = False
            self._move_last_scene_pos = None
        self._update_cursor()
        self.viewport().update()

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, int(size))
        self._update_cursor()
        self.viewport().update()

    def set_brush_color(self, color: QColor | tuple[int, int, int] | str) -> None:
        if isinstance(color, QColor):
            next_color = QColor(color)
        elif isinstance(color, tuple) and len(color) >= 3:
            next_color = QColor(int(color[0]), int(color[1]), int(color[2]))
        else:
            next_color = QColor(str(color))
        if not next_color.isValid():
            return
        next_color.setAlpha(230)
        self._brush_color = next_color
        self._update_cursor()
        self.viewport().update()

    def set_tablet_pressure_enabled(self, enabled: bool) -> None:
        self._use_tablet_pressure = bool(enabled)
        if not self._use_tablet_pressure:
            self._tablet_pressure = 1.0
        self.viewport().update()

    def set_roi_select_mode(self, enabled: bool) -> None:
        self._roi_select_mode = bool(enabled)
        if not self._roi_select_mode:
            self._roi_dragging = False
            self._roi_start_scene = None
            self._roi_current_scene = None
        self._update_cursor()
        self.viewport().update()

    def set_roi_rect(self, rect: tuple[int, int, int, int] | None) -> None:
        if rect is None:
            self._roi_rect = None
            self.viewport().update()
            return
        if len(rect) != 4:
            return
        x, y, w, h = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        if w <= 0 or h <= 0:
            self._roi_rect = None
        else:
            self._roi_rect = (x, y, w, h)
        self.viewport().update()

    def set_point_select_mode(self, enabled: bool) -> None:
        self._point_select_mode = bool(enabled)
        self._update_cursor()
        self.viewport().update()

    def set_guide_routes(self, routes: list[dict[str, object]] | None) -> None:
        self._guide_routes = [dict(route) for route in (routes or []) if isinstance(route, dict)]
        self.viewport().update()

    def set_image(self, image_rgb: np.ndarray | None) -> None:
        if image_rgb is None:
            self._pixmap_item.setPixmap(QPixmap())
            self._image_shape = (0, 0)
            self._hover_scene_pos = None
            self._stroke_points = []
            self._smoothed_last_point = None
            self._roi_dragging = False
            self._roi_start_scene = None
            self._roi_current_scene = None
            self.viewport().update()
            return

        h, w = image_rgb.shape[:2]
        self._image_shape = (h, w)
        qimage = numpy_to_qimage(image_rgb)
        pixmap = QPixmap.fromImage(qimage)
        self._pixmap_item.setPixmap(pixmap)
        self.scene().setSceneRect(0, 0, w, h)
        if self._auto_fit:
            self.fit_to_view()

    def update_image_patch(self, patch_rgb: np.ndarray, x0: int, y0: int) -> None:
        if patch_rgb is None or patch_rgb.size == 0:
            return
        if self._pixmap_item.pixmap().isNull():
            return
        if patch_rgb.ndim != 3 or patch_rgb.shape[2] != 3:
            return

        h, w = self._image_shape
        if h <= 0 or w <= 0:
            return

        x0 = int(x0)
        y0 = int(y0)
        x1 = min(w, x0 + int(patch_rgb.shape[1]))
        y1 = min(h, y0 + int(patch_rgb.shape[0]))
        if x0 < 0:
            patch_rgb = patch_rgb[:, -x0:]
            x0 = 0
        if y0 < 0:
            patch_rgb = patch_rgb[-y0:, :]
            y0 = 0
        if x1 <= x0 or y1 <= y0:
            return

        patch_rgb = np.ascontiguousarray(patch_rgb[: y1 - y0, : x1 - x0])
        patch_qimage = numpy_to_qimage(patch_rgb)
        pixmap = self._pixmap_item.pixmap()
        painter = QPainter(pixmap)
        painter.drawImage(x0, y0, patch_qimage)
        painter.end()
        self._pixmap_item.setPixmap(pixmap)
        self.viewport().update()

    def fit_to_view(self) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._zoom_factor = 1.0
        self._auto_fit = True
        self.zoom_changed.emit(self._zoom_factor)

    def reset_zoom(self) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        self.resetTransform()
        self._zoom_factor = 1.0
        self._auto_fit = False
        self.zoom_changed.emit(self._zoom_factor)

    def zoom_in(self) -> None:
        self._apply_zoom(1.15)

    def zoom_out(self) -> None:
        self._apply_zoom(1.0 / 1.15)

    def _apply_zoom(self, factor: float) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        self.scale(factor, factor)
        self._zoom_factor = max(0.05, min(40.0, self._zoom_factor * factor))
        self._auto_fit = False
        self.zoom_changed.emit(self._zoom_factor)

    def _update_cursor(self) -> None:
        if self._move_dragging:
            self.setCursor(Qt.ClosedHandCursor)
            return
        if self._panning:
            self.setCursor(Qt.OpenHandCursor)
            return
        if self._roi_select_mode:
            self.setCursor(Qt.CrossCursor)
            return
        if self._point_select_mode:
            self.setCursor(Qt.CrossCursor)
            return
        if not self._editable:
            self.setCursor(Qt.ArrowCursor)
            return
        if self._tool == "move":
            self.setCursor(Qt.OpenHandCursor)
            return
        if self._tool == "wand":
            diameter = 22
        else:
            diameter = max(18, min(42, int(round(self._brush_size * 1.35))))
        accent = self._tool_color()
        cache_key = (self._tool, diameter, int(accent.rgba()))
        cached = self._cursor_cache.get(cache_key)
        if cached is None:
            pix = QPixmap(diameter, diameter)
            pix.fill(Qt.transparent)
            center = float(diameter // 2)
            radius = max(3.0, center - 3.0)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 230), 1.2))
            painter.drawEllipse(QRectF(center - radius - 0.9, center - radius - 0.9, 2.0 * (radius + 0.9), 2.0 * (radius + 0.9)))
            painter.setPen(QPen(accent, 1.2))
            painter.drawEllipse(QRectF(center - radius, center - radius, 2.0 * radius, 2.0 * radius))
            painter.drawLine(QPointF(center - 5.5, center), QPointF(center + 5.5, center))
            painter.drawLine(QPointF(center, center - 5.5), QPointF(center, center + 5.5))
            if self._tool == "eraser":
                painter.setPen(QPen(QColor(255, 181, 58, 210), 1.4))
                painter.drawLine(QPointF(center - 6.0, center + 6.0), QPointF(center + 6.0, center - 6.0))
            elif self._tool == "razor":
                painter.setPen(QPen(QColor(255, 92, 92, 220), 1.5))
                painter.drawLine(QPointF(center - 6.0, center + 4.5), QPointF(center + 7.0, center - 4.5))
                painter.drawLine(QPointF(center + 4.5, center - 6.0), QPointF(center + 7.0, center - 4.5))
            elif self._tool == "wand":
                painter.setPen(QPen(QColor(98, 214, 255, 230), 1.4))
                painter.drawEllipse(QRectF(center - 3.0, center - 3.0, 6.0, 6.0))
                painter.drawLine(QPointF(center + 3.5, center - 3.5), QPointF(center + 7.2, center - 7.2))
                painter.drawLine(QPointF(center + 5.7, center - 7.8), QPointF(center + 8.0, center - 5.5))
            painter.end()
            cached = QCursor(pix, int(center), int(center))
            self._cursor_cache[cache_key] = cached
        self.setCursor(cached)

    def _tool_color(self) -> QColor:
        if self._tool == "eraser":
            return QColor(255, 181, 58, 220)
        if self._tool == "razor":
            return QColor(255, 92, 92, 230)
        if self._tool == "wand":
            return QColor(98, 214, 255, 235)
        if self._tool == "move":
            return QColor(232, 244, 255, 235)
        return QColor(self._brush_color)

    def _scene_point_to_image_point(self, scene_point: QPointF) -> tuple[int, int] | None:
        if self._image_shape == (0, 0):
            return None
        x = int(round(scene_point.x()))
        y = int(round(scene_point.y()))
        h, w = self._image_shape
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return (x, y)

    def _event_image_point(self, event: QMouseEvent) -> tuple[int, int] | None:
        return self._scene_point_to_image_point(self.mapToScene(event.position().toPoint()))

    def _set_pan_mode(self, enabled: bool) -> None:
        self._panning = enabled
        self.setDragMode(QGraphicsView.ScrollHandDrag if enabled else QGraphicsView.NoDrag)
        self._update_cursor()

    def _effective_radius(self) -> int:
        pressure_scale = self._tablet_pressure if self._use_tablet_pressure else 1.0
        pressure_scale = max(0.10, min(1.75, pressure_scale))
        if self._brush_size <= 1:
            return 0
        base_radius = ((float(self._brush_size) - 1.0) / 2.0) * pressure_scale
        return max(1, int(round(base_radius)))

    def _update_hover_scene_pos_from_scene(self, scene_point: QPointF) -> None:
        if self._image_shape == (0, 0):
            self._hover_scene_pos = None
            return
        h, w = self._image_shape
        if scene_point.x() < 0 or scene_point.y() < 0 or scene_point.x() >= w or scene_point.y() >= h:
            self._hover_scene_pos = None
        else:
            self._hover_scene_pos = scene_point
        self.viewport().update()

    def _update_hover_scene_pos(self, event: QMouseEvent) -> None:
        self._update_hover_scene_pos_from_scene(self.mapToScene(event.position().toPoint()))

    def _smoothed_point(self, raw_point: tuple[int, int]) -> tuple[int, int]:
        self._stroke_points.append(raw_point)
        if len(self._stroke_points) > 24:
            self._stroke_points = self._stroke_points[-24:]
        if self._brush_size <= 3:
            return raw_point
        if self._smoothed_last_point is None:
            return raw_point
        prev_x, prev_y = self._smoothed_last_point
        alpha = self._smoothing_alpha
        sx = (1.0 - alpha) * prev_x + alpha * raw_point[0]
        sy = (1.0 - alpha) * prev_y + alpha * raw_point[1]
        return (int(round(sx)), int(round(sy)))

    def _start_stroke(self, point: tuple[int, int], pressure: float = 1.0) -> None:
        if self._tool == "wand":
            self.stroke_applied.emit(
                {
                    "bbox": (int(point[0]), int(point[1]), int(point[0]), int(point[1])),
                    "mask": np.ones((1, 1), dtype=bool),
                    "point": (int(point[0]), int(point[1])),
                    "tool": "wand",
                },
                self._tool,
            )
            self.stroke_finished.emit()
            return
        self._drawing = True
        self._auto_fit = False
        self._tablet_pressure = float(max(0.05, min(1.75, pressure)))
        self._last_point = point
        self._stroke_points = [point]
        self._smoothed_last_point = point
        radius = self._effective_radius()
        stroke = rasterize_stroke(self._image_shape, point, point, radius)
        self.stroke_applied.emit(stroke, self._tool)

    def _normalized_roi_from_scene_points(
        self,
        start_scene: QPointF,
        end_scene: QPointF,
    ) -> tuple[int, int, int, int] | None:
        if self._image_shape == (0, 0):
            return None
        h, w = self._image_shape
        x0 = int(np.floor(min(start_scene.x(), end_scene.x())))
        y0 = int(np.floor(min(start_scene.y(), end_scene.y())))
        x1 = int(np.ceil(max(start_scene.x(), end_scene.x())))
        y1 = int(np.ceil(max(start_scene.y(), end_scene.y())))

        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        if x1 < x0 or y1 < y0:
            return None

        roi_w = max(1, x1 - x0 + 1)
        roi_h = max(1, y1 - y0 + 1)
        return (x0, y0, roi_w, roi_h)

    def _continue_stroke(self, point: tuple[int, int], pressure: float = 1.0) -> None:
        if not self._drawing:
            return
        self._tablet_pressure = float(max(0.05, min(1.75, pressure)))
        smoothed = self._smoothed_point(point)
        start = self._smoothed_last_point if self._smoothed_last_point is not None else smoothed
        stroke = rasterize_stroke(self._image_shape, start, smoothed, self._effective_radius())
        self.stroke_applied.emit(stroke, self._tool)
        self._last_point = point
        self._smoothed_last_point = smoothed

    def _end_stroke(self) -> None:
        if not self._drawing:
            return
        self._drawing = False
        self._last_point = None
        self._smoothed_last_point = None
        self._stroke_points = []
        self._tablet_pressure = 1.0
        self.stroke_finished.emit()

    def wheelEvent(self, event: QWheelEvent) -> None:
        angle_delta = int(event.angleDelta().y())
        pixel_delta = int(event.pixelDelta().y())
        if angle_delta == 0 and pixel_delta == 0:
            event.ignore()
            return

        if self._editable and bool(event.modifiers() & Qt.AltModifier):
            delta = angle_delta if angle_delta != 0 else pixel_delta
            direction = 1 if delta > 0 else -1
            self.brush_size_adjust.emit(direction)
            event.accept()
            return

        # Use fine-grained wheel/trackpad steps instead of a coarse fixed jump.
        if angle_delta != 0:
            steps = float(angle_delta) / 120.0
        else:
            # Trackpads often report pixel deltas in smaller increments.
            steps = float(pixel_delta) / 240.0
        if abs(steps) < 0.01:
            event.ignore()
            return

        steps = max(-4.0, min(4.0, steps))
        zoom_factor = self._wheel_zoom_base_step ** steps
        self._apply_zoom(float(zoom_factor))
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._update_hover_scene_pos(event)

        if event.button() == Qt.MiddleButton:
            self._set_pan_mode(True)
            fake = QMouseEvent(
                QMouseEvent.MouseButtonPress,
                event.position(),
                Qt.LeftButton,
                Qt.LeftButton,
                event.modifiers(),
            )
            super().mousePressEvent(fake)
            return

        if self._space_held and event.button() == Qt.LeftButton:
            self._set_pan_mode(True)
            fake = QMouseEvent(
                QMouseEvent.MouseButtonPress,
                event.position(),
                Qt.LeftButton,
                Qt.LeftButton,
                event.modifiers(),
            )
            super().mousePressEvent(fake)
            return

        if self._point_select_mode and event.button() == Qt.LeftButton and self._image_shape != (0, 0):
            point = self._event_image_point(event)
            if point is not None:
                self.point_selected.emit(point)
                event.accept()
                return

        if self._roi_select_mode and event.button() == Qt.LeftButton and self._image_shape != (0, 0):
            scene_pos = self.mapToScene(event.position().toPoint())
            image_point = self._scene_point_to_image_point(scene_pos)
            if image_point is not None:
                self._roi_dragging = True
                self._roi_start_scene = QPointF(float(scene_pos.x()), float(scene_pos.y()))
                self._roi_current_scene = QPointF(float(scene_pos.x()), float(scene_pos.y()))
                event.accept()
                self.viewport().update()
                return

        if not self._editable or event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        if self._tool == "move":
            scene_pos = self.mapToScene(event.position().toPoint())
            point = self._scene_point_to_image_point(scene_pos)
            if point is None:
                super().mousePressEvent(event)
                return
            self._move_dragging = True
            self._move_last_scene_pos = QPointF(float(scene_pos.x()), float(scene_pos.y()))
            self._auto_fit = False
            self._update_cursor()
            event.accept()
            return

        point = self._event_image_point(event)
        if point is None:
            super().mousePressEvent(event)
            return

        self._start_stroke(point, pressure=1.0)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._update_hover_scene_pos(event)

        if self._roi_dragging and self._roi_select_mode:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._roi_current_scene = QPointF(float(scene_pos.x()), float(scene_pos.y()))
            self.viewport().update()
            event.accept()
            return

        if self._move_dragging and self._editable and self._tool == "move":
            scene_pos = self.mapToScene(event.position().toPoint())
            if self._move_last_scene_pos is None:
                self._move_last_scene_pos = QPointF(float(scene_pos.x()), float(scene_pos.y()))
            dx = int(round(scene_pos.x() - self._move_last_scene_pos.x()))
            dy = int(round(scene_pos.y() - self._move_last_scene_pos.y()))
            if dx != 0 or dy != 0:
                self.mask_nudged.emit(int(dx), int(dy))
                self._move_last_scene_pos = QPointF(float(scene_pos.x()), float(scene_pos.y()))
            event.accept()
            return

        if self._drawing and self._editable:
            point = self._event_image_point(event)
            if point is not None and self._last_point is not None:
                self._continue_stroke(point, pressure=1.0)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._update_hover_scene_pos(event)

        if event.button() == Qt.MiddleButton:
            fake = QMouseEvent(
                QMouseEvent.MouseButtonRelease,
                event.position(),
                Qt.LeftButton,
                Qt.LeftButton,
                event.modifiers(),
            )
            super().mouseReleaseEvent(fake)
            self._set_pan_mode(False)
            return

        if self._panning and event.button() == Qt.LeftButton:
            fake = QMouseEvent(
                QMouseEvent.MouseButtonRelease,
                event.position(),
                Qt.LeftButton,
                Qt.LeftButton,
                event.modifiers(),
            )
            super().mouseReleaseEvent(fake)
            self._set_pan_mode(False)
            return

        if self._roi_dragging and event.button() == Qt.LeftButton and self._roi_select_mode:
            scene_pos = self.mapToScene(event.position().toPoint())
            if self._roi_start_scene is not None:
                rect = self._normalized_roi_from_scene_points(self._roi_start_scene, scene_pos)
                if rect is not None:
                    self._roi_rect = rect
                    self.roi_selected.emit(rect)
            self._roi_dragging = False
            self._roi_start_scene = None
            self._roi_current_scene = None
            event.accept()
            self.viewport().update()
            return

        if self._move_dragging and event.button() == Qt.LeftButton and self._tool == "move":
            self._move_dragging = False
            self._move_last_scene_pos = None
            self.mask_nudge_finished.emit()
            self._update_cursor()
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            self._end_stroke()

        super().mouseReleaseEvent(event)

    def tabletEvent(self, event: QTabletEvent) -> None:
        scene_point = self.mapToScene(event.position().toPoint())
        self._update_hover_scene_pos_from_scene(scene_point)

        if not self._editable or self._image_shape == (0, 0):
            event.ignore()
            return

        image_point = self._scene_point_to_image_point(scene_point)
        pressure = float(event.pressure()) if self._use_tablet_pressure else 1.0
        pressure = max(0.05, min(1.75, pressure))

        if event.type() == QEvent.Type.TabletPress:
            if image_point is not None:
                self._start_stroke(image_point, pressure=pressure)
            event.accept()
            return

        if event.type() == QEvent.Type.TabletMove:
            if image_point is not None and self._drawing:
                self._continue_stroke(image_point, pressure=pressure)
            event.accept()
            return

        if event.type() == QEvent.Type.TabletRelease:
            if self._drawing:
                self._end_stroke()
            event.accept()
            return

        super().tabletEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover_scene_pos = None
        self.viewport().update()
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            self._space_held = True
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            self._space_held = False
            if self._panning:
                self._set_pan_mode(False)
        super().keyReleaseEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._auto_fit and not self._pixmap_item.pixmap().isNull():
            self.fit_to_view()

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawForeground(painter, rect)

        if self._roi_rect is not None:
            x, y, w, h = self._roi_rect
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(70, 212, 255, 230), 0))
            painter.drawRect(QRectF(float(x), float(y), float(max(1, w)), float(max(1, h))))
            painter.restore()

        if self._roi_dragging and self._roi_select_mode and self._roi_start_scene is not None and self._roi_current_scene is not None:
            drag_rect = self._normalized_roi_from_scene_points(self._roi_start_scene, self._roi_current_scene)
            if drag_rect is not None:
                rx, ry, rw, rh = drag_rect
                painter.save()
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setBrush(Qt.NoBrush)
                pen = QPen(QColor(70, 212, 255, 245), 0)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                painter.drawRect(QRectF(float(rx), float(ry), float(max(1, rw)), float(max(1, rh))))
                painter.restore()

        for route in self._guide_routes:
            points_raw = route.get("points", [])
            if not isinstance(points_raw, (list, tuple)):
                continue
            points: list[QPointF] = []
            for point in points_raw:
                try:
                    x, y = point
                    points.append(QPointF(float(x), float(y)))
                except Exception:
                    continue
            if not points:
                continue
            color_raw = route.get("color", (255, 198, 88))
            try:
                color = QColor(int(color_raw[0]), int(color_raw[1]), int(color_raw[2]), 245)
            except Exception:
                color = QColor(255, 198, 88, 245)
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setBrush(color)
            painter.setPen(QPen(QColor(0, 0, 0, 220), 3.0))
            for start, end in zip(points[:-1], points[1:]):
                painter.drawLine(start, end)
            painter.setPen(QPen(color, 1.6))
            for start, end in zip(points[:-1], points[1:]):
                painter.drawLine(start, end)
            for index, point in enumerate(points, start=1):
                painter.setPen(QPen(QColor(0, 0, 0, 230), 1.2))
                painter.drawEllipse(QRectF(point.x() - 4.0, point.y() - 4.0, 8.0, 8.0))
                painter.setPen(QPen(QColor(255, 255, 255, 245), 0))
                painter.drawText(QRectF(point.x() + 5.0, point.y() - 10.0, 28.0, 20.0), str(index))
            painter.restore()

        if self._roi_select_mode or self._point_select_mode:
            return

        if not self._editable or self._hover_scene_pos is None or self._image_shape == (0, 0):
            return

        if self._tool == "move":
            x = self._hover_scene_pos.x()
            y = self._hover_scene_pos.y()
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(QPen(QColor(232, 244, 255, 220), 0))
            cross = 8.0
            painter.drawLine(QPointF(x - cross, y), QPointF(x + cross, y))
            painter.drawLine(QPointF(x, y - cross), QPointF(x, y + cross))
            label = "Move mask"
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(label) + 10
            th = fm.height() + 6
            tx = x + 14.0
            ty = y - 12.0
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(8, 10, 16, 196))
            painter.drawRoundedRect(QRectF(tx, ty - th, tw, th), 4.0, 4.0)
            painter.setPen(QPen(QColor(220, 231, 255, 245), 0))
            painter.drawText(QRectF(tx + 5.0, ty - th + 3.0, tw - 6.0, th - 4.0), Qt.AlignLeft | Qt.AlignVCenter, label)
            painter.restore()
            return

        x = self._hover_scene_pos.x()
        y = self._hover_scene_pos.y()
        h, w = self._image_shape
        if x < 0 or y < 0 or x >= w or y >= h:
            return

        effective_radius = 0 if self._tool == "wand" else self._effective_radius()
        radius = 0.5 if effective_radius <= 0 else float(effective_radius)
        inner_color = self._tool_color()

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(0, 0, 0, 220), 0))
        painter.drawEllipse(QRectF(x - radius - 0.75, y - radius - 0.75, 2.0 * (radius + 0.75), 2.0 * (radius + 0.75)))
        painter.setPen(QPen(inner_color, 0))
        painter.drawEllipse(QRectF(x - radius, y - radius, 2.0 * radius, 2.0 * radius))

        cross = max(3.0, min(radius * 0.45, 10.0))
        painter.drawLine(QPointF(x - cross, y), QPointF(x + cross, y))
        painter.drawLine(QPointF(x, y - cross), QPointF(x, y + cross))

        label = "Wand" if self._tool == "wand" else f"{self._tool.capitalize()} {self._brush_size}px"
        if self._tool != "wand" and self._use_tablet_pressure and abs(self._tablet_pressure - 1.0) > 0.02:
            label += f"  p:{self._tablet_pressure:.2f}"
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label) + 10
        th = fm.height() + 6
        tx = x + radius + 10.0
        ty = y - radius - 8.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(8, 10, 16, 196))
        painter.drawRoundedRect(QRectF(tx, ty - th, tw, th), 4.0, 4.0)
        painter.setPen(QPen(QColor(220, 231, 255, 245), 0))
        painter.drawText(QRectF(tx + 5.0, ty - th + 3.0, tw - 6.0, th - 4.0), Qt.AlignLeft | Qt.AlignVCenter, label)
        painter.restore()
