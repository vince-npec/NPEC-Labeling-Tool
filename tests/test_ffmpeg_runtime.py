from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from resources.ffmpeg_runtime import configure_ffmpeg_runtime


class FfmpegRuntimeTests(unittest.TestCase):
    def test_explicit_npec_path_configures_imageio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            executable = Path(tmp_dir_name) / "ffmpeg"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
            with patch.dict(
                os.environ,
                {"NPEC_FFMPEG_EXE": str(executable)},
                clear=True,
            ):
                result = configure_ffmpeg_runtime()
                self.assertEqual(result, executable.resolve())
                self.assertEqual(
                    os.environ.get("IMAGEIO_FFMPEG_EXE"),
                    str(executable.resolve()),
                )

    def test_missing_explicit_path_falls_back_to_path_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            executable = Path(tmp_dir_name) / "ffmpeg"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
            with (
                patch.dict(
                    os.environ,
                    {"NPEC_FFMPEG_EXE": str(Path(tmp_dir_name) / "missing")},
                    clear=True,
                ),
                patch("resources.ffmpeg_runtime.shutil.which", return_value=str(executable)),
            ):
                self.assertEqual(configure_ffmpeg_runtime(), executable.resolve())

    def test_returns_none_when_no_executable_is_available(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("resources.ffmpeg_runtime.shutil.which", return_value=None),
            patch("resources.ffmpeg_runtime._usable_executable", return_value=None),
        ):
            self.assertIsNone(configure_ffmpeg_runtime())


if __name__ == "__main__":
    unittest.main()
