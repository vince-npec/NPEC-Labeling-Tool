from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_LABEL_COLOR_HEX = "#ff5a5f"


@dataclass(slots=True)
class LabelClass:
    class_id: int
    name: str
    color_hex: str


@dataclass(slots=True)
class DatasetImageItem:
    uid: str
    name: str
    path: Path
    image: np.ndarray


def normalize_color_hex(color_hex: str, fallback: str = DEFAULT_LABEL_COLOR_HEX) -> str:
    raw = str(color_hex).strip().lower()
    if raw.startswith("0x"):
        raw = raw[2:]
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) == 8:
        # Accept AARRGGBB and keep RGB channels.
        raw = raw[2:]
    if len(raw) == 3 and all(ch in "0123456789abcdef" for ch in raw):
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6 or not all(ch in "0123456789abcdef" for ch in raw):
        return fallback
    return f"#{raw}"


def hex_to_rgb(color_hex: str) -> tuple[int, int, int]:
    raw = normalize_color_hex(color_hex)[1:]
    return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"
