from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def _cover_resize(image: Image.Image, size: int) -> Image.Image:
    src_w, src_h = image.size
    scale = max(size / float(src_w), size / float(src_h))
    resized = image.resize((int(round(src_w * scale)), int(round(src_h * scale))), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - size) // 2)
    top = max(0, (resized.height - size) // 2)
    return resized.crop((left, top, left + size, top + size))


def build_master_icon(source_path: Path, out_png: Path, size: int = 1024) -> Image.Image:
    source = Image.open(source_path).convert("RGBA")
    source_cover = _cover_resize(source, size)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # Background gradient
    bg = Image.new("RGBA", (size, size), (20, 30, 46, 255))
    bg_draw = ImageDraw.Draw(bg)
    for i in range(size):
        t = i / max(1, size - 1)
        r = int(20 + (38 - 20) * t)
        g = int(30 + (48 - 30) * t)
        b = int(46 + (68 - 46) * t)
        bg_draw.line((0, i, size, i), fill=(r, g, b, 255))

    radius = int(size * 0.20)
    frame_margin = int(size * 0.075)

    # Shadow
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        (frame_margin + 10, frame_margin + 16, size - frame_margin + 10, size - frame_margin + 16),
        radius=radius,
        fill=(0, 0, 0, 140),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=18))
    canvas.alpha_composite(shadow)

    # Main rounded container
    rounded_mask = Image.new("L", (size, size), 0)
    rounded_draw = ImageDraw.Draw(rounded_mask)
    rounded_draw.rounded_rectangle(
        (frame_margin, frame_margin, size - frame_margin, size - frame_margin),
        radius=radius,
        fill=255,
    )

    panel = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    panel.alpha_composite(bg)
    panel.alpha_composite(source_cover)

    composed = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    composed.paste(panel, (0, 0), rounded_mask)

    border = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    border_draw = ImageDraw.Draw(border)
    border_draw.rounded_rectangle(
        (frame_margin + 1, frame_margin + 1, size - frame_margin - 1, size - frame_margin - 1),
        radius=radius,
        outline=(235, 242, 255, 200),
        width=max(4, size // 180),
    )

    canvas.alpha_composite(composed)
    canvas.alpha_composite(border)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, format="PNG")
    return canvas


def build_ico(icon_image: Image.Image, out_ico: Path) -> None:
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    out_ico.parent.mkdir(parents=True, exist_ok=True)
    icon_image.save(out_ico, format="ICO", sizes=sizes)


def _build_icns_with_iconutil(master_png: Path, out_icns: Path) -> bool:
    iconutil = shutil.which("iconutil")
    sips = shutil.which("sips")
    if not iconutil or not sips:
        return False

    with tempfile.TemporaryDirectory(prefix="npec_iconset_") as tmpdir:
        iconset_dir = Path(tmpdir) / "npec-app.iconset"
        iconset_dir.mkdir(parents=True, exist_ok=True)
        pairs = [16, 32, 128, 256, 512]

        for base in pairs:
            png1x = iconset_dir / f"icon_{base}x{base}.png"
            png2x = iconset_dir / f"icon_{base}x{base}@2x.png"
            subprocess.run([sips, "-z", str(base), str(base), str(master_png), "--out", str(png1x)], check=True)
            subprocess.run([sips, "-z", str(base * 2), str(base * 2), str(master_png), "--out", str(png2x)], check=True)

        subprocess.run([iconutil, "-c", "icns", str(iconset_dir), "-o", str(out_icns)], check=True)
        return out_icns.exists()


def build_icns(icon_image: Image.Image, out_icns: Path, master_png: Path) -> None:
    out_icns.parent.mkdir(parents=True, exist_ok=True)
    try:
        icon_image.save(out_icns, format="ICNS")
        if out_icns.exists() and out_icns.stat().st_size > 0:
            return
    except Exception:
        pass

    if _build_icns_with_iconutil(master_png, out_icns):
        return

    raise RuntimeError("Unable to create .icns icon (Pillow ICNS save unsupported and iconutil unavailable).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate app icon assets from source image.")
    parser.add_argument("--input", required=True, help="Source logo/image file.")
    parser.add_argument("--output-dir", required=True, help="Output directory for icon assets.")
    parser.add_argument("--name", default="npec-app-icon", help="Base output name (default: npec-app-icon).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    base_name = args.name

    if not source_path.exists():
        raise FileNotFoundError(f"Source image not found: {source_path}")

    out_png = output_dir / f"{base_name}.png"
    out_ico = output_dir / f"{base_name}.ico"
    out_icns = output_dir / f"{base_name}.icns"

    icon_image = build_master_icon(source_path, out_png)
    build_ico(icon_image, out_ico)

    try:
        build_icns(icon_image, out_icns, out_png)
    except Exception as exc:
        print(f"[warn] ICNS generation skipped: {exc}")

    print(f"Generated: {out_png}")
    print(f"Generated: {out_ico}")
    if out_icns.exists():
        print(f"Generated: {out_icns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
