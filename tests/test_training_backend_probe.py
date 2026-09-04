from __future__ import annotations

import ast
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import PySide6  # noqa: F401
except ModuleNotFoundError:
    qtcore = types.ModuleType("PySide6.QtCore")

    class _QObject:
        pass

    class _Signal:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def emit(self, *args, **kwargs) -> None:
            pass

    def _slot(*args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    qtcore.QObject = _QObject
    qtcore.Signal = _Signal
    qtcore.Slot = _slot

    pyside6 = types.ModuleType("PySide6")
    pyside6.QtCore = qtcore
    sys.modules.setdefault("PySide6", pyside6)
    sys.modules.setdefault("PySide6.QtCore", qtcore)

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from resources.training import (
    _external_tf_probe_source,
    _external_training_runner_source,
    detect_training_backends,
)


class TrainingBackendProbeTests(unittest.TestCase):
    def test_external_probe_source_is_valid_python(self) -> None:
        source = _external_tf_probe_source()
        ast.parse(source)
        self.assertIn("import tensorflow as tf", source)
        self.assertIn("print(json.dumps(info))", source)

    def test_external_training_runner_source_is_valid_python(self) -> None:
        source = _external_training_runner_source()
        ast.parse(source)
        self.assertIn("def _try_export_tflite", source)
        self.assertIn("tf.saved_model.save", source)

    def test_external_backend_probe_rejects_gui_executable_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            fake_gui_exe = Path(tmp_dir_name) / "NPEC Labeling Tool.exe"
            fake_gui_exe.write_bytes(b"not-a-python-runtime")
            info = detect_training_backends(external_python=fake_gui_exe, framework="tensorflow")
        self.assertFalse(bool(info.get("available", False)))
        self.assertIn("does not look like a Python executable", str(info.get("error", "")))


if __name__ == "__main__":
    unittest.main()
