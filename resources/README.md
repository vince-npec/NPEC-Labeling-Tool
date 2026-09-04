# NPEC Labeling Tool (Desktop)

Native-style desktop segmentation labeling app for macOS workflows, built with `PySide6`.

## Key capabilities

- Five workflow tabs:
  - `Labeling`: layered pixel-wise masks with transparent overlays
  - `Model Testing`: model inference + GT/prediction/disagreement comparison + metrics + split-view compare
  - `Patching`: disagreement-focused correction tools before export
  - `Training`: in-app segmentation training/fine-tuning with patch-based settings, GPU/CPU mode selection, and progress logs
  - `Timelapse`: playback and 1080p MP4 export of original + labels (+ optional predictions)
- Pro annotation tools:
  - `Brush` for painting class regions
  - `Eraser` for clearing the active class
  - `Razor` for cutting labels across all classes
- Camera-like canvas controls:
  - mouse wheel zoom
  - live brush perimeter/cursor outline
  - Space + drag pan
  - `B` / `E` / `R` tool switching
  - `[` and `]` brush size shortcuts
  - `A` / `D` timeline navigation
  - `F1` shortcut map
  - tablet pressure hooks (optional toggle)
- Project persistence:
  - save project as `.oclp`
  - load project and continue with classes, masks, predictions, timeline state
  - close prompt asks to save unsaved project changes
- Model support:
  - `.tflite`
  - `.h5`
  - `.keras`
  - optional external TensorFlow runtime bridge (point app to a separate Python executable, e.g. `NPEC_GPU/python.exe`, to run `.h5/.keras` inference outside the packaged app)
- Training support:
  - new model training (U-Net style)
  - fine-tune from existing `.h5` / `.keras`
  - device preference: Auto / GPU-only / CPU-only (uses TensorFlow runtime on the host machine)
  - Training tab runtime selector for external TensorFlow Python (useful for Windows GPU envs such as `NPEC_GPU`)
  - optional `.tflite` export after training
- Export pipeline:
  - indexed PNG masks
  - per-class binary PNG masks
  - 1080p MP4 timelapse video
  - metadata JSON in ZIP archive

## Install and run (dev)

```bash
cd "/path/to/NPEC-Labeling-Tool"
python3.11 -m venv .oc-desktop-venv
source .oc-desktop-venv/bin/activate
pip install -r resources/requirements.txt
python -m resources.main
```

### External TensorFlow environment (recommended for packaged Windows app)

If `.h5/.keras` inference reports missing TensorFlow:

1. Open tab `2) Model Testing`
2. Set `Backend` to `Standard (.tflite/.h5/.keras)`
3. In `External TF Python`, click `Browse` and select your TensorFlow env executable (for example `...\anaconda3\envs\NPEC_GPU\python.exe`)
4. Run inference again

You can also set this before launch with environment variable:

- `NPEC_EXTERNAL_TF_PYTHON`
- or `NPEC_TF_PYTHON`

## Build macOS app bundle and installer

```bash
cd "/path/to/NPEC-Labeling-Tool"
chmod +x resources/build_macos.sh
./resources/build_macos.sh
```

Outputs:

- `.app`: `resources/dist/NPEC Labeling Tool.app`
- `.dmg`: `resources/dist/NPEC Labeling Tool.dmg` (uses `create-dmg` when available, otherwise built with `hdiutil`)

## Build Windows package and installer

Run on a Windows machine (PyInstaller cannot cross-compile Windows binaries from macOS):

```powershell
cd "C:\path\to\New project"
.\resources\build_windows.ps1
```

Or:

```bat
cd C:\path\to\New project
resources\build_windows.cmd
```

Prerequisites for Windows build:

- Python `3.11` (recommended) or `3.12` installed via [python.org](https://www.python.org/downloads/windows/)
- `py` launcher available (`py -0p` should list your installed runtimes)
- Do not use Python `3.13` for packaging this app

If you are in Anaconda `base`, either deactivate first or explicitly call 3.11:

```bat
conda deactivate
py -3.11 resources\build_windows.ps1
```

or:

```bat
resources\build_windows.cmd
```

Outputs:

- onedir app folder: `resources\dist-win\NPEC Labeling Tool\`
- executable: `resources\dist-win\NPEC Labeling Tool\NPEC Labeling Tool.exe`
- zip package: `resources\dist-win\NPEC-Labeling-Tool-windows.zip`
- optional installer (if Inno Setup `iscc.exe` is installed): `resources\dist-win\NPEC-Labeling-Tool-Setup.exe`

Optional flags:

- `-OneFile` to produce a single-file executable build.
- `-SkipTensorFlowCollection` to skip bundling TensorFlow internals (smaller build; training and in-app Keras runtime may be unavailable).

### Launch the Windows app with an external TensorFlow GPU environment

If you want the packaged app to use a separate TensorFlow GPU environment for training or `.h5/.keras` runtime support:

```bat
set NPEC_EXTERNAL_TF_PYTHON=C:\path\to\NPEC_GPU\python.exe
resources\launch_with_external_tf.cmd
```

Or pass the interpreter directly:

```bat
resources\launch_with_external_tf.cmd C:\path\to\NPEC_GPU\python.exe
```

Inside the app, the Training tab should then report `TensorFlow ... (external) · GPU xN ...` after clicking `Refresh Backend`.

## Export structure

- `masks/indexed/<image_stem>.png` (single-channel indexed mask, class IDs as pixel values)
- `masks/binary/<image_stem>/class_<id>.png` (per-class binary masks)
- `metadata.json` (class map + image metadata)
