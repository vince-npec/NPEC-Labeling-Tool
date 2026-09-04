# Building the macOS Application

Release `2026.8.2.2` targets Apple Silicon and requires macOS 14 or newer. The
minimum is recorded in the bundle as `LSMinimumSystemVersion`.

## Public release build

The public build deliberately excludes bundled FFmpeg/x264/x265 executables.
Build an OpenCV wheel without video-codec support first:

```bash
./resources/scripts/build_macos_opencv_no_ffmpeg.sh "$PWD/dist-opencv"
```

Then build the app with that wheel:

```bash
NPEC_OPENCV_WHEEL="$PWD/dist-opencv/opencv_python_headless-4.11.0.86-...whl" \
  NPEC_BUNDLE_FFMPEG=0 \
  NPEC_MODEL_PROFILE=yang \
  ./resources/build_macos.sh
```

The build aborts if `libx264` or `libx265` remains in the application bundle.
MP4 generation in the resulting app uses a separately installed FFmpeg selected
from `NPEC_FFMPEG_EXE`, `PATH`, `/opt/homebrew/bin/ffmpeg`, or
`/usr/local/bin/ffmpeg`.

`NPEC_MODEL_PROFILE` accepts `none`, `yang`, or `all` and defaults to `none`.
The `yang` profile contains the cleared publication artifacts and
checksum-verifies every file listed in the Yang model card. `all` is intended
for controlled internal builds because the other model families require
separate provenance review.

The build installs the exact tested environment from
`environments/macos-arm64-2026.8.2.2.txt`. Override `NPEC_REQUIREMENTS_FILE`
only when intentionally qualifying a different dependency set.

## Internal GPL-aware build

Setting `NPEC_BUNDLE_FFMPEG=1` retains the video libraries from the build
environment. Do not distribute that artifact unless all applicable license
notices and corresponding source obligations have been satisfied.

## Signing

The default script applies an ad-hoc signature. A broadly distributed macOS
release should instead use an authorized Developer ID certificate and Apple
notarization. The bundle identifier is `nl.npec.labelingtool` and the release
version comes from `NPEC_APP_VERSION`.

Always verify the final artifact:

```bash
codesign --verify --deep --strict --verbose=2 "resources/dist/NPEC Labeling Tool.app"
shasum -a 256 -c "resources/dist/NPEC Labeling Tool.dmg.sha256"
```
