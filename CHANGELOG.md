# Changelog

## 2026.8.2.2 - 2026-09-04

First public-source reproducibility snapshot of NPEC Labeling Tool.

### Included

- Desktop annotation, segmentation, tracking, phenotyping, and project I/O.
- Recursive Lazy Folder processing for Hades and Lucifer datasets.
- Plate-total and quality-gated per-seedling measurements.
- Five-lane temporal ownership with source-overlay videos and bounding boxes.
- The Yang Lucifer RGB inoculated five-seedling preset, ranker profile, model
  checksums, validation records, and Methods-ready pipeline description.
- A separately downloadable, checksum-verified Yang Song model pack and a
  matching model-bearing macOS installer.
- macOS and Windows packaging scripts and an automated test workflow.

### Release qualifications

- Plate-total root length is conserved independently of ownership.
- Per-seedling ownership is inferred and can be invalidated by QC when roots
  overlap beyond reliable assignment.
- Bacteria-occluded root gaps are not imputed by the default Yang preset.
- Binary checkpoints remain outside normal Git history. The exact Yang Song
  release artifacts were authorized for public redistribution on 2026-09-04;
  other model families require separate provenance review.
