# NPEC Labeling Tool

NPEC Labeling Tool is a desktop application for image annotation, semantic
plant segmentation, time-series tracking, and phenotyping. It supports Hades
grayscale images, Lucifer RGB images, dark-background RGB experiments, and
single-plant potato endpoint images.

This repository is the publication reproducibility snapshot for release
`2026.8.2.4`, including the five-seedling Lucifer RGB pipeline used for the
Yang Song Arabidopsis analyses in the DroughtFighters project, Plant Microbe
Interaction group, Utrecht University.

The historical technical token `yang` in filenames, paths, profile keys, and
commands refers to Yang Song. Those identifiers remain unchanged for backward
compatibility and checksum-level reproducibility; human-facing provenance uses
her full name and project affiliation.

## Main capabilities

- Layered pixel annotation for roots, shoots, lateral roots, and custom classes.
- Hades/Lucifer PyPhenotyper-compatible model routing.
- Recursive Lazy Folder processing with plate identity and timestamp grouping.
- Plant/top-root or dish-frame stabilization for plate time series.
- Primary/lateral root decomposition from whole-root masks.
- Plate-total and per-seedling root metrics, shoot area, QC, workbooks, and CSVs.
- Temporal five-lane seedling tracking with visible ownership boxes.
- Source-image mask-overlay videos and mask-only review videos.
- Project persistence in the `.oclp` format.
- macOS and Windows packaging scripts.

## Scientific scope

Plate-total root length is independent of seedling ownership and is the most
robust measurement when neighboring roots overlap. Per-seedling allocation is
an inferred, quality-gated measurement. Ambiguous crossings can be marked
invalid, and invalid per-seedling aggregates can be `NaN` while the conserved
plate total remains available.

Bacterial occlusion is treated as unknown visibility. The default Yang Song
workflow profile does not invent hidden root pixels. Optional gap-repair outputs
must be reported separately with their own provenance.

See [the Yang Song pipeline record](docs/YANG_PIPELINE.md) for the exact presets,
model hashes, validation design, limitations, and reproduction commands.
The source-snapshot and verification record is in
[docs/RELEASE_PROVENANCE.md](docs/RELEASE_PROVENANCE.md).

## Install from source

Python 3.11 is recommended.

```bash
git clone https://github.com/vince-npec/NPEC-Labeling-Tool.git
cd NPEC-Labeling-Tool
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r resources/requirements.txt
python -m resources.main
```

The app starts maximized by default. Set `NPEC_WINDOWED_START=1` to start in a
regular window.

MP4 generation requires an FFmpeg executable. The public macOS installer does
not bundle the GPL-enabled development binary. Install FFmpeg separately (for
example, `brew install ffmpeg`) or set `NPEC_FFMPEG_EXE=/path/to/ffmpeg` before
starting the app.

## Yang Song Lucifer RGB recipe

1. Open **Segmentation Pipeline**.
2. Select the PyPhenotyper backend.
3. Select **Arabidopsis - Lucifer RGB inoculated hybrid (5 seedlings)**.
4. Set the source folder and a new output-mask folder.
5. Review plate identities if plate-ID scanning is enabled.
6. Choose **Plate + per-seedling metrics**.
7. Keep expected plants `5`, root classes `1,3`, shoot class `2`, and total-root metric.
8. Enable previous-frame stabilization with **plants/top roots**.
9. Keep source-overlay video and RGB green-shoot recomputation enabled.
10. Leave Lucifer GAN repair disabled unless its precomputed output is part of the documented protocol.
11. Run **Folder (Lazy)** and archive the generated manifests and run report.

The normal combined run writes, among other files:

- `npec_root_measurements_master.csv`
- `npec_total_root_length_timelapse.csv`
- `npec_total_root_and_shoot_from_masks.csv`
- `npec_per_plant_all_metrics.xlsx`
- `npec_root_growth_analysis_video.mp4`
- `npec_root_and_shoot_masks_analysis_video.mp4`
- `npec_root_growth_video_summary.csv`
- `npec_root_growth_video_sample_frame.png`

## Model artifacts

Checkpoint binaries are intentionally kept out of normal Git history. The exact
Yang Song workflow production artifacts have been cleared for redistribution
and are published as a checksum-verified release asset and in the corresponding
macOS installer. The repository records all expected filenames and hashes in
[docs/MODEL_SHA256SUMS](docs/MODEL_SHA256SUMS) and scientific scope in
[docs/MODELS.md](docs/MODELS.md). Other models may be supplied externally through
the application model selectors.

Real images, `.oclp` templates, annotation corpora, paper drafts, and generated
analysis outputs are not part of this source repository.

## Build

macOS:

```bash
NPEC_MODEL_PROFILE=yang ./resources/build_macos.sh
```

Public-release macOS builds use a no-FFmpeg OpenCV wheel; follow
[`docs/BUILD_MACOS.md`](docs/BUILD_MACOS.md) rather than distributing a wheel
that silently carries GPL video libraries.

The `2026.8.2.4` macOS installer is an Apple Silicon build for macOS 14 or
newer.

Windows builds must run on Windows:

```powershell
.\resources\build_windows.ps1
```

Both platform builders default to `none` for bundled model checkpoints. Pass
`NPEC_MODEL_PROFILE=yang` on macOS or `-ModelProfile yang` on Windows to build
the authorized Yang Song workflow release profile. Both builders verify all
publication artifact hashes before packaging.

Installers must be built from a tagged commit. Release `2026.8.2.4` uses the
frozen macOS ARM64 environment in
[environments/macos-arm64-2026.8.2.4.txt](environments/macos-arm64-2026.8.2.4.txt).

## Test

```bash
pip install -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen pytest -q
```

## Citation

Use GitHub's **Cite this repository** control or the metadata in
[CITATION.cff](CITATION.cff). For a paper, cite the tagged release actually used
for analysis rather than the moving default branch.

## Acknowledgements

The project thanks **Jason van Hamond** for extensive testing and feedback that
helped improve the NPEC Labeling Tool's reliability, usability, and release
readiness. Contributor roles and research provenance are recorded in
[AUTHORS.md](AUTHORS.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues should follow
[SECURITY.md](SECURITY.md).

Release changes are recorded in [CHANGELOG.md](CHANGELOG.md).

The original NPEC Labeling Tool code is licensed under Apache-2.0, Copyright
2026 Vinicius Lube. Third-party components retain their own licenses.

The software license does not grant rights to NPEC names or logos; see
[TRADEMARKS.md](TRADEMARKS.md). Third-party components retain their own terms,
summarized in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
