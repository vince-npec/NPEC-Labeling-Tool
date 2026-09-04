from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import types
import unittest
from unittest.mock import patch

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

from resources.training import _external_gpu_preflight_issue, _try_export_tflite


class _FakeConverter:
    def __init__(self) -> None:
        self.target_spec = SimpleNamespace(supported_ops=None)
        self._experimental_lower_tensor_list_ops = None
        self.experimental_enable_resource_variables = None
        self.optimizations = None

    def convert(self) -> bytes:
        return b"TFL3"


class _FakeSavedModelOK:
    def save(self, model, path: str) -> None:
        export_dir = Path(path)
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "saved_model.pb").write_bytes(b"saved")


class _FakeSavedModelFail:
    def save(self, model, path: str) -> None:
        raise RuntimeError("simulated tf.saved_model.save failure")


class _FakeConverterFactory:
    @staticmethod
    def from_saved_model(path: str) -> _FakeConverter:
        return _FakeConverter()

    @staticmethod
    def from_keras_model(model) -> _FakeConverter:
        return _FakeConverter()


def _fake_tf(saved_model_impl) -> SimpleNamespace:
    return SimpleNamespace(
        saved_model=saved_model_impl,
        lite=SimpleNamespace(
            OpsSet=SimpleNamespace(TFLITE_BUILTINS="builtins", SELECT_TF_OPS="select"),
            TFLiteConverter=_FakeConverterFactory,
        ),
    )


class _ModelWithExportThatMustNotRun:
    def __init__(self) -> None:
        self.export_called = False

    def export(self, path: str, verbose: bool = True) -> None:
        self.export_called = True
        raise AssertionError("model.export should not be called when tf.saved_model.save succeeds")


class _ModelWithStdoutExport:
    def __init__(self) -> None:
        self.export_called = False

    def export(self, path: str, verbose: bool = True) -> None:
        self.export_called = True
        sys.stdout.write("materializing saved model")
        export_dir = Path(path)
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "saved_model.pb").write_bytes(b"saved")


class TrainingExportFallbackTests(unittest.TestCase):
    def test_tflite_export_prefers_tf_saved_model_save_before_model_export(self) -> None:
        tf_module = _fake_tf(_FakeSavedModelOK())
        model = _ModelWithExportThatMustNotRun()
        logs: list[str] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "trained_model.keras"
            with patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
                tflite_path, error = _try_export_tflite(tf_module, model, output_path, logs.append)
            tflite_exists = False if tflite_path is None else Path(tflite_path).exists()

        self.assertIsNone(error)
        self.assertIsNotNone(tflite_path)
        self.assertTrue(tflite_exists)
        self.assertFalse(model.export_called)
        self.assertTrue(any("tf.saved_model.save" in entry for entry in logs))

    def test_tflite_export_falls_back_to_model_export_with_guarded_streams(self) -> None:
        tf_module = _fake_tf(_FakeSavedModelFail())
        model = _ModelWithStdoutExport()
        logs: list[str] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "trained_model.keras"
            with patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
                tflite_path, error = _try_export_tflite(tf_module, model, output_path, logs.append)
            tflite_exists = False if tflite_path is None else Path(tflite_path).exists()

        self.assertIsNone(error)
        self.assertIsNotNone(tflite_path)
        self.assertTrue(tflite_exists)
        self.assertTrue(model.export_called)
        self.assertTrue(any("keras.model.export" in entry for entry in logs))

    def test_external_gpu_preflight_reports_missing_gpu_runtime_cleanly(self) -> None:
        probe_payload = {
            "available": True,
            "version": "2.10.1",
            "gpu_count": 0,
            "error": (
                "Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found\n"
                "Cannot dlopen some GPU libraries."
            ),
        }
        with patch("resources.training._probe_external_tf_runtime", return_value=probe_payload):
            message = _external_gpu_preflight_issue(Path("C:/path/to/NPEC_GPU/python.exe"))

        self.assertIsNotNone(message)
        self.assertIn("not ready for GPU training", str(message))
        self.assertIn("no GPU device detected", str(message))
        self.assertIn("cudart64_110.dll", str(message))


if __name__ == "__main__":
    unittest.main()
