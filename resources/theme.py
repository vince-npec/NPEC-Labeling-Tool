APP_STYLESHEET = """
QMainWindow {
    background-color: #2a2c31;
    color: #d4d7dd;
}

QWidget {
    color: #d4d7dd;
    font-family: "SF Pro Text", "Segoe UI", "Roboto", sans-serif;
}

#NpecMainWindow[uiCompact="true"] QWidget {
    font-size: 12px;
}

#NpecMainWindow[uiUltraCompact="true"] QWidget {
    font-size: 11px;
}

#NpecMainWindow[uiCompact="true"] QToolButton,
#NpecMainWindow[uiCompact="true"] QPushButton {
    padding: 3px 7px;
}

#NpecMainWindow[uiUltraCompact="true"] QToolButton,
#NpecMainWindow[uiUltraCompact="true"] QPushButton {
    padding: 3px 7px;
}

#NpecMainWindow[uiCompact="true"] QTabBar::tab {
    padding: 5px 8px;
}

#NpecMainWindow[uiUltraCompact="true"] QTabBar::tab {
    padding: 4px 6px;
}

QSplitter::handle {
    background: #1d1f23;
    border: 1px solid #111317;
}

#datasetSidebar {
    background: #292c32;
    border-right: 1px solid #181a1f;
}

#workspaceShell {
    background: #2b2e34;
}

#workflowNavigation {
    background: #26292f;
    border: 1px solid #17191d;
    border-radius: 6px;
    padding: 6px;
}

QTabBar#workflowTabs::tab {
    min-height: 24px;
    margin: 0 2px 0 0;
    padding: 6px 10px;
    border: 1px solid #17191d;
    border-radius: 5px;
}

QTabBar#workflowTabs::tab:selected {
    background: #f28a2e;
    color: #191b1f;
    border-color: #9f5d1f;
}

QComboBox#moduleSelector {
    min-height: 24px;
}

QDockWidget#layersInspector {
    color: #cdd2da;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}

QDockWidget#layersInspector::title {
    background: #30343c;
    border-bottom: 1px solid #181a1f;
    padding: 6px 8px;
    text-align: left;
}

QMenuBar {
    background: #2f3137;
    border-bottom: 1px solid #1a1c20;
}

QMenuBar::item {
    background: transparent;
    color: #c7ccd4;
    padding: 5px 10px;
}

QMenuBar::item:selected {
    background: #3a3d45;
    color: #f2f4f8;
}

QMenu {
    background: #32353d;
    border: 1px solid #1a1c20;
    padding: 5px;
}

QMenu::item {
    padding: 5px 12px;
    border-radius: 4px;
}

QMenu::item:selected {
    background: #f28a2e;
    color: #1b1c1f;
}

QTabWidget::pane {
    background: #2b2e34;
    border: 1px solid #17191d;
    border-top: none;
    border-bottom-left-radius: 6px;
    border-bottom-right-radius: 6px;
}

QTabWidget > QWidget,
QTabWidget QStackedWidget > QWidget,
QScrollArea,
QScrollArea > QWidget,
QScrollArea > QWidget > QWidget {
    background: #2b2e34;
}

QTabBar::tab {
    color: #b6bbc4;
    background: #2f3239;
    border: 1px solid #17191d;
    border-bottom: none;
    padding: 7px 11px;
    margin-right: 3px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:hover {
    background: #3b404a;
    color: #e9edf2;
}

QTabBar::tab:selected {
    background: #4a515e;
    color: #ffffff;
}

QToolBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3a3d44, stop:1 #2e3036);
    border-bottom: 1px solid #17191d;
    spacing: 6px;
    padding: 5px;
}

QToolButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4e525b, stop:1 #3a3d45);
    color: #d8dbe2;
    border: 1px solid #1b1d22;
    border-radius: 5px;
    padding: 5px 9px;
}

QToolButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5f646f, stop:1 #464a54);
}

QToolButton:checked,
QToolButton:pressed {
    background: #f28a2e;
    color: #191b1f;
    border: 1px solid #9f5d1f;
}

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4b4f58, stop:1 #3a3d45);
    border: 1px solid #1b1d22;
    border-radius: 6px;
    padding: 6px 11px;
    color: #e0e4eb;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5d626e, stop:1 #454a55);
}

QPushButton:pressed {
    background: #343840;
}

QPushButton:disabled {
    color: #808691;
    background: #2c2f35;
}

QLineEdit, QTextEdit, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #25282e;
    border: 1px solid #13151a;
    border-radius: 6px;
    padding: 4px 6px;
    color: #d9dde5;
    selection-background-color: #f28a2e;
    selection-color: #1a1c20;
}

QAbstractSpinBox {
    padding-right: 32px;
    min-height: 20px;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #17191d;
    border-bottom: 1px solid #17191d;
    background: #32353c;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: padding;
    subcontrol-position: bottom right;
    width: 24px;
    border-left: 1px solid #17191d;
    border-top: 1px solid #17191d;
    background: #32353c;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #424652;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none;
    width: 0px;
    height: 0px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 6px solid #c2c8d2;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    width: 0px;
    height: 0px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #c2c8d2;
}

QComboBox::drop-down {
    border-left: 1px solid #17191d;
    width: 20px;
    background: #32353c;
}

QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #c2c8d2;
    margin-right: 4px;
}

QAbstractScrollArea::viewport {
    background: #2b2e34;
}

QTableWidget, QTableView {
    background: #25282e;
    gridline-color: #17191d;
}

QListWidget {
    outline: none;
}

QListWidget::item {
    border-radius: 5px;
    padding: 6px 4px;
}

QListWidget::item:hover {
    background: #3a3f49;
}

QListWidget::item:selected {
    background: #566074;
    color: #f7f9fc;
    border: 1px solid #78829a;
}

QCheckBox {
    spacing: 6px;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid #17191d;
    background: #2b2e34;
}

QCheckBox::indicator:checked {
    background: #f28a2e;
    border: 1px solid #9f5d1f;
}

QSlider::groove:horizontal {
    border: 1px solid #14161b;
    height: 6px;
    background: #24272d;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #f28a2e;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #d3d8e1;
    border: 1px solid #202227;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #ffffff;
}

QDockWidget {
    color: #d4d7dd;
    border: 1px solid #14161a;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}

QDockWidget::title {
    background: #343740;
    text-align: center;
    padding: 6px;
    border-bottom: 1px solid #17191d;
}

QTableWidget, QTableView {
    background: #25282e;
    alternate-background-color: #2b2f37;
    border: 1px solid #13151a;
    color: #d9dde5;
    gridline-color: #17191d;
    selection-background-color: #566074;
    selection-color: #f7f9fc;
}

QAbstractScrollArea,
QAbstractScrollArea > QWidget,
QAbstractScrollArea QWidget#qt_scrollarea_viewport,
QTreeView,
QListView {
    background: #2b2e34;
}

QHeaderView::section {
    background: #343740;
    border: 1px solid #1f2229;
    padding: 4px 6px;
    color: #d9dde5;
}

QStatusBar {
    background: #2a2c31;
    border-top: 1px solid #17191d;
}

QStatusBar QLabel {
    color: #c6cbd4;
    padding: 0 6px;
}

QScrollBar:vertical {
    background: #1f2126;
    width: 12px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #4f5665;
    border-radius: 6px;
    min-height: 25px;
    border: 1px solid #20242b;
}

QScrollBar::handle:vertical:hover {
    background: #636b7b;
}

QScrollBar:horizontal {
    background: #1f2126;
    height: 12px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background: #4f5665;
    border-radius: 6px;
    min-width: 25px;
    border: 1px solid #20242b;
}

QScrollBar::handle:horizontal:hover {
    background: #636b7b;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal,
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    border: none;
    background: none;
    width: 0px;
    height: 0px;
}
"""


WINDOWS_DIALOG_STYLESHEET = """
QMessageBox,
QProgressDialog,
QInputDialog {
    background-color: #f5f6f8;
    color: #111111;
}

QMessageBox QWidget,
QProgressDialog QWidget,
QInputDialog QWidget {
    background-color: #f5f6f8;
    color: #111111;
}

QMessageBox QLabel,
QProgressDialog QLabel,
QInputDialog QLabel {
    background: transparent;
    color: #111111;
}

QMessageBox QCheckBox,
QProgressDialog QCheckBox,
QInputDialog QCheckBox {
    color: #111111;
}

QMessageBox QTextEdit,
QInputDialog QLineEdit,
QInputDialog QSpinBox,
QInputDialog QDoubleSpinBox,
QInputDialog QComboBox {
    background: #ffffff;
    color: #111111;
    border: 1px solid #b9c3d0;
}

QMessageBox QProgressBar,
QProgressDialog QProgressBar {
    border: 1px solid #b9c3d0;
    border-radius: 4px;
    background: #ffffff;
    color: #111111;
    text-align: center;
}
"""
