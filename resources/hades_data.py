from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil


HADES_DISH_TOKEN_RE = re.compile(r"^(exp\d+_\d+)$")
HADES_DISH_SEARCH_RE = re.compile(r"(exp\d+_\d+)")


@dataclass(slots=True)
class HadesOrganizationResult:
    root: Path
    series_names: list[str]
    series_counts: dict[str, int]
    moved_files: int
    unmatched_root_files: list[Path]
    conflict_files: list[Path]
    already_structured: bool


def _iter_root_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_file() and not path.name.startswith("._")
    ]


def collect_flat_hades_groups(root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in _iter_root_files(root):
        match = HADES_DISH_SEARCH_RE.search(path.name)
        if not match:
            continue
        grouped.setdefault(match.group(1), []).append(path)
    return grouped


def discover_hades_series_dirs(root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not HADES_DISH_TOKEN_RE.fullmatch(child.name):
            continue
        paths = [path for path in sorted(child.iterdir()) if path.is_file() and not path.name.startswith("._")]
        if paths:
            grouped[child.name] = paths
    return grouped


def ensure_hades_series_folders(root: Path) -> HadesOrganizationResult:
    existing_dirs = discover_hades_series_dirs(root)
    flat_groups = collect_flat_hades_groups(root)

    matched_root_paths = {path for paths in flat_groups.values() for path in paths}
    unmatched_root_files = [path for path in _iter_root_files(root) if path not in matched_root_paths]
    conflict_files: list[Path] = []
    moved_files = 0

    for series_name, paths in flat_groups.items():
        target_dir = root / series_name
        target_dir.mkdir(exist_ok=True)
        for path in paths:
            target_path = target_dir / path.name
            if target_path.exists():
                conflict_files.append(path)
                continue
            shutil.move(str(path), str(target_path))
            moved_files += 1

    final_dirs = discover_hades_series_dirs(root)
    series_names = sorted(final_dirs.keys())
    series_counts = {series_name: len(paths) for series_name, paths in final_dirs.items()}
    already_structured = bool(existing_dirs) and moved_files == 0 and not flat_groups

    return HadesOrganizationResult(
        root=root,
        series_names=series_names,
        series_counts=series_counts,
        moved_files=moved_files,
        unmatched_root_files=unmatched_root_files,
        conflict_files=conflict_files,
        already_structured=already_structured,
    )
