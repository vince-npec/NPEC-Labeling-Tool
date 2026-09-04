from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QApplication, QLayout, QLayoutItem, QSizePolicy, QStyle, QTabWidget, QWidget


class FlowLayout(QLayout):
    """A compact wrapping layout for command controls."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        margin: int = 0,
        horizontal_spacing: int = 6,
        vertical_spacing: int = 6,
    ) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._horizontal_spacing = int(horizontal_spacing)
        self._vertical_spacing = int(vertical_spacing)
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientations:
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, max(0, int(width)), 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _spacing(self, item: QLayoutItem, *, horizontal: bool) -> int:
        explicit = self._horizontal_spacing if horizontal else self._vertical_spacing
        if explicit >= 0:
            return explicit
        widget = item.widget()
        style = widget.style() if widget is not None else QApplication.style()
        metric = QStyle.PM_LayoutHorizontalSpacing if horizontal else QStyle.PM_LayoutVerticalSpacing
        return max(0, int(style.pixelMetric(metric)))

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x = effective.x()
        y = effective.y()
        line_height = 0

        for item in self._items:
            widget = item.widget()
            if widget is not None and not widget.isVisible() and not test_only:
                continue
            hint = item.sizeHint()
            width = max(item.minimumSize().width(), hint.width())
            height = max(item.minimumSize().height(), hint.height())
            h_space = self._spacing(item, horizontal=True)
            v_space = self._spacing(item, horizontal=False)
            next_x = x + width
            if x > effective.x() and next_x > effective.right() + 1:
                x = effective.x()
                y += line_height + v_space
                next_x = x + width
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), QSize(min(width, effective.width()), height)))
            x = next_x + h_space
            line_height = max(line_height, height)

        return max(0, y + line_height - rect.y() + margins.bottom())


class ResponsiveTabWidget(QTabWidget):
    """A tab stack whose hidden pages cannot dictate the window minimum size."""

    def minimumSizeHint(self) -> QSize:
        return QSize(460, 320)

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(max(760, min(1280, hint.width())), max(560, min(900, hint.height())))


def place_window_on_available_screen(window: QWidget, *, width_fraction: float = 0.94, height_fraction: float = 0.92) -> None:
    """Choose a useful normal geometry before the main window is maximized."""

    screen = window.screen() or QApplication.primaryScreen()
    if screen is None:
        return
    available = screen.availableGeometry()
    target_width = min(1780, max(1024, int(round(available.width() * float(width_fraction)))))
    target_height = min(1030, max(680, int(round(available.height() * float(height_fraction)))))
    target_width = min(target_width, available.width())
    target_height = min(target_height, available.height())
    window.resize(target_width, target_height)
    frame = window.frameGeometry()
    frame.moveCenter(available.center())
    window.move(frame.topLeft())
    window.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
