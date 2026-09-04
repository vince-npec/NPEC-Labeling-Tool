from __future__ import annotations

import sys
from pathlib import Path


_PYTHON_LIKE_NAMES = {
    "python",
    "python.exe",
    "pythonw",
    "pythonw.exe",
    "python3",
    "python3.exe",
    "py",
    "py.exe",
    "pypy",
    "pypy.exe",
    "pypy3",
    "pypy3.exe",
}


def is_probably_python_executable(path: str | Path | None) -> bool:
    if path is None:
        return False
    try:
        name = Path(path).name.strip().lower()
    except Exception:
        return False
    if not name:
        return False
    if name in _PYTHON_LIKE_NAMES:
        return True
    return name.startswith("python") or name.startswith("pypy")


def safe_current_python_executable() -> Path | None:
    raw_candidates = [
        getattr(sys, "_base_executable", None),
        getattr(sys, "executable", None),
    ]
    seen: set[str] = set()
    for raw in raw_candidates:
        if not raw:
            continue
        try:
            candidate = Path(raw).expanduser().absolute()
        except Exception:
            continue
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists() or not candidate.is_file():
            continue
        if is_probably_python_executable(candidate):
            return candidate
    return None


def common_external_python_candidates() -> list[Path]:
    home = Path.home()
    raw_candidates = [
        home / "anaconda3" / "bin" / "python",
        home / "miniconda3" / "bin" / "python",
        home / "miniforge3" / "bin" / "python",
        home / "mambaforge" / "bin" / "python",
        home / "anaconda3" / "python.exe",
        home / "miniconda3" / "python.exe",
        home / "miniforge3" / "python.exe",
        home / "mambaforge" / "python.exe",
        home / "AppData" / "Local" / "Programs" / "Python" / "Python310" / "python.exe",
        home / "AppData" / "Local" / "Programs" / "Python" / "Python311" / "python.exe",
        home / "AppData" / "Local" / "Programs" / "Python" / "Python312" / "python.exe",
        home / "AppData" / "Local" / "Programs" / "Python" / "Python313" / "python.exe",
        Path("C:/ProgramData/Anaconda3/python.exe"),
        Path("C:/ProgramData/miniconda3/python.exe"),
        Path("C:/ProgramData/miniforge3/python.exe"),
    ]
    current = safe_current_python_executable()
    if current is not None:
        raw_candidates.append(current)

    candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        try:
            normalized = candidate.expanduser().absolute()
        except Exception:
            continue
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(normalized)
    return candidates
