from __future__ import annotations

import multiprocessing
import os
import sys

if __name__ == "__main__":
    # Call this before importing Qt or app modules so frozen Windows helper
    # processes do not re-enter the GUI bootstrap.
    multiprocessing.freeze_support()

from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QProgressBar, QVBoxLayout, QWidget

try:
    from resources.app import APP_TITLE, create_main_window
    from resources.responsive_ui import place_window_on_available_screen
except Exception:
    try:
        from NPEC_Labeling_Tool.app import APP_TITLE, create_main_window
        from NPEC_Labeling_Tool.responsive_ui import place_window_on_available_screen
    except Exception as exc:
        raise ModuleNotFoundError(
            "Unable to import application module. "
            "Tried resources.app and NPEC_Labeling_Tool.app."
        ) from exc


SPLASH_ASSET_PATH = Path(__file__).resolve().parent / "assets" / "npec-label.jpeg"
SPLASH_USER_PATH = Path(os.environ["NPEC_LABEL_TOOL_SPLASH_PATH"]).expanduser() if os.environ.get("NPEC_LABEL_TOOL_SPLASH_PATH") else None


def _resolve_splash_path() -> Path | None:
    if SPLASH_USER_PATH is not None and SPLASH_USER_PATH.exists():
        return SPLASH_USER_PATH
    if SPLASH_ASSET_PATH.exists():
        return SPLASH_ASSET_PATH
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        candidate = Path(meipass) / "resources" / "assets" / "npec-label.jpeg"
        if candidate.exists():
            return candidate
    return None


class StartupSplash(QWidget):
    def __init__(self, image_path: Path | None) -> None:
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setObjectName("startupSplash")
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setStyleSheet(
            "#startupSplash { background-color: #111821; border: 1px solid #2b3647; border-radius: 10px; }"
            "QLabel { color: #d8e3f7; font-size: 13px; }"
            "QProgressBar { border: 1px solid #2f3c50; border-radius: 6px; background: #0c121a; height: 14px; }"
            "QProgressBar::chunk { border-radius: 5px; background: #2aa6ff; }"
        )
        self.setMinimumSize(620, 380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(280)
        layout.addWidget(self.image_label, 1)

        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        if image_path is not None and image_path.exists():
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                scaled = pixmap.scaled(980, 540, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled)

        self._center_on_screen()

    def _center_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        rect = screen.availableGeometry()
        self.move(rect.center() - self.rect().center())

    def set_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(max(0, min(100, int(value))))
        self.status_label.setText(message)
        QApplication.processEvents()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    # Use a consistent non-native style so stylesheet behavior is stable across macOS/Windows builds.
    app.setStyle("Fusion")

    splash = StartupSplash(_resolve_splash_path())
    splash.show()
    splash.set_progress(12, "Starting application...")

    window = create_main_window()
    splash.set_progress(78, "Loading interface...")
    place_window_on_available_screen(window)
    if str(os.environ.get("NPEC_WINDOWED_START", "")).strip().lower() in {"1", "true", "yes"}:
        window.show()
    else:
        window.showMaximized()
    splash.set_progress(100, "Ready")

    QTimer.singleShot(220, splash.close)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
