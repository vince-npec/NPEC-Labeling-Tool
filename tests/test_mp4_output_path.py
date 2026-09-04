from __future__ import annotations

from pathlib import Path

import pytest

from resources.analytics_engine import prepare_mp4_output_path


def test_prepare_mp4_output_path_creates_missing_parent(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "timelapse.mp4"

    resolved = prepare_mp4_output_path(output_path)

    assert resolved == output_path
    assert output_path.parent.is_dir()
    assert not output_path.exists()


def test_prepare_mp4_output_path_rejects_file_as_parent(tmp_path: Path) -> None:
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x", encoding="utf-8")

    with pytest.raises(OSError):
        prepare_mp4_output_path(blocker / "timelapse.mp4")
