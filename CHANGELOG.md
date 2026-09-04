# Changelog

## 2026.8.2.4 - 2026-09-04

Acknowledgement-only correction to release `2026.8.2.3`.

- Corrected the in-application and user-guide testing credit to the full name
  Jason van Hamond.
- Added public acknowledgement of his extensive application testing and
  feedback to the authorship record and README.
- No segmentation algorithm, model weight, measurement formula, validation
  result, or scientific interpretation changed in this patch.

## 2026.8.2.3 - 2026-09-04

Provenance-only correction to release `2026.8.2.2`.

- Expanded the dataset-specific attribution to Yang Song of the DroughtFighters
  project, Plant Microbe Interaction group, Utrecht University.
- Added the full person and project provenance to the model card, validation
  records, Methods text, application message, packaged model documentation, and
  citation metadata.
- Retained legacy `yang` filenames, paths, profile keys, corpus IDs, and command
  values for backward compatibility and checksum-level reproducibility.
- No segmentation algorithm, model weight, measurement formula, or validation
  result changed in this patch.

## 2026.8.2.2 - 2026-09-04

First public-source reproducibility snapshot of NPEC Labeling Tool.

### Included

- Desktop annotation, segmentation, tracking, phenotyping, and project I/O.
- Recursive Lazy Folder processing for Hades and Lucifer datasets.
- Plate-total and quality-gated per-seedling measurements.
- Five-lane temporal ownership with source-overlay videos and bounding boxes.
- The Yang Song Lucifer RGB inoculated five-seedling preset for the
  DroughtFighters project, Plant Microbe Interaction group, Utrecht University,
  including its ranker profile, model checksums, validation records, and
  Methods-ready pipeline description.
- A separately downloadable, checksum-verified Yang Song model pack and a
  matching model-bearing macOS installer.
- macOS and Windows packaging scripts and an automated test workflow.

### Release qualifications

- Plate-total root length is conserved independently of ownership.
- Per-seedling ownership is inferred and can be invalidated by QC when roots
  overlap beyond reliable assignment.
- Bacteria-occluded root gaps are not imputed by the default Yang Song workflow
  preset.
- Binary checkpoints remain outside normal Git history. The exact Yang Song
  release artifacts were authorized for public redistribution on 2026-09-04;
  other model families require separate provenance review.
