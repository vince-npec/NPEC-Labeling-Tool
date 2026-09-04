from __future__ import annotations

import os
from pathlib import Path
import shutil


def _usable_executable(value: str | os.PathLike[str] | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return path.resolve()
    return None


def configure_ffmpeg_runtime() -> Path | None:
    """Point imageio-ffmpeg at a user-provided FFmpeg executable when available."""

    for variable in ("NPEC_FFMPEG_EXE", "IMAGEIO_FFMPEG_EXE"):
        candidate = _usable_executable(os.environ.get(variable))
        if candidate is not None:
            os.environ["IMAGEIO_FFMPEG_EXE"] = str(candidate)
            return candidate

    candidates = [shutil.which("ffmpeg")]
    if os.name != "nt":
        candidates.extend(("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"))

    for value in candidates:
        candidate = _usable_executable(value)
        if candidate is not None:
            os.environ["IMAGEIO_FFMPEG_EXE"] = str(candidate)
            return candidate
    return None


def ffmpeg_install_hint() -> str:
    return (
        "Install FFmpeg and restart the app, or set NPEC_FFMPEG_EXE to an "
        "executable FFmpeg path. On macOS with Homebrew: brew install ffmpeg"
    )
