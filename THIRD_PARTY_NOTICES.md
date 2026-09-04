# Third-Party Notices

NPEC Labeling Tool depends on third-party packages that retain their own
copyright and license terms. This summary is not a replacement for the license
files shipped by each dependency.

| Component | Release version | License summary |
|---|---:|---|
| PySide6 / Qt for Python | 6.9.3 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only |
| NumPy | 1.26.4 | BSD-3-Clause and bundled notices |
| Pillow | 12.3.0 | HPND |
| pandas | 3.0.5 | BSD-3-Clause |
| OpenCV Python | 4.11.0.86 | Apache-2.0 |
| scikit-image | 0.26.0 | BSD-3-Clause and bundled notices |
| SciPy | 1.17.1 | BSD-3-Clause and bundled notices |
| imageio | 2.37.4 | BSD-2-Clause |
| imageio-ffmpeg | 0.6.0 | BSD-2-Clause wrapper; bundled FFmpeg has separate terms |
| TensorFlow | 2.16.2 | Apache-2.0 |
| PyInstaller | 6.22.2 | GPL-2.0-or-later with bootloader exception |

## FFmpeg

The frozen macOS development environment resolves an `imageio-ffmpeg` FFmpeg
7.1 binary built with `--enable-gpl`, including GPL codecs such as x264/x265.
Its exact build configuration is recorded in
`docs/third-party/ffmpeg-7.1-build.txt`.

The public macOS build defaults to `NPEC_BUNDLE_FFMPEG=0` and removes that
executable from the application bundle. Users provide an FFmpeg installation
separately; the app discovers `ffmpeg` on `PATH`, Homebrew's standard macOS
locations, or the path in `NPEC_FFMPEG_EXE`. A distributor who deliberately
sets `NPEC_BUNDLE_FFMPEG=1` is responsible for the applicable FFmpeg/GPL notices
and corresponding-source obligations.

## OpenCV macOS wheel

The stock macOS `opencv-python-headless` wheel used during development contains
FFmpeg libraries plus `libx264` and `libx265`. The public release build therefore
uses the same OpenCV version rebuilt from source with `WITH_FFMPEG=OFF`; see
`resources/scripts/build_macos_opencv_no_ffmpeg.sh`. The macOS packaging script
refuses a public build when x264/x265 libraries remain in the app bundle.
The qualified macOS 13-compatible ARM64 wheel has SHA-256
`9f282b07f15f6bbb648eea62b3abab8c09effca6c27e6ef8af670a787d036e3c`;
the complete application requires macOS 14 because of the frozen SciPy wheel.

## PyPhenotyper compatibility

`resources/npec_pyphenotyper` is a compatibility implementation for the
PyPhenotyper calling surface and a substantially revised descendant of that
software lineage. The authoritative frozen NPEC source is
[`NPEC-NL/pyphenotyper`](https://github.com/NPEC-NL/pyphenotyper) at commit
`8b5c6f4eb1c5902d943917b129fb246ead9a8a9d`, licensed BSD-3-Clause. Its license
is reproduced in `licenses/PyPhenotyper-BSD-3-Clause.txt`.

PyPhenotyper lineage credit belongs to Alican Noyan, Fedya (Fedor) Chursin,
Francisco Ribeiro Mansilha, Wesley van Gaalen, Borislav Nachev, Vlad Matache,
Valerian Meline, and Danna Shao. Detailed contribution statements are recorded
in the upstream repository's `PROVENANCE.md`.

The exact Yang root and shoot checkpoints documented by this repository do not
match the already-public PyPhenotyper model files. Their release therefore
remains subject to the separate clearance described in `docs/MODELS.md`.

## Complete environment

The exact package inventory used for the macOS ARM64 release candidate is in
`environments/macos-arm64-2026.8.2.2.txt`.
Its reproducible CycloneDX 1.6 software bill of materials is in
`docs/sbom/macos-arm64-2026.8.2.2.cdx.json`.
