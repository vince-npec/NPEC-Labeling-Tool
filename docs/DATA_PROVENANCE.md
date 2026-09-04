# Data Provenance

No raw research images, `.oclp` projects, private masks, or paper drafts are
distributed in this repository.

## Yang Song Arabidopsis corpus

The five-seedling owner ranker and inoculated RGB shoot work used a private
February 2026 annotation corpus associated with Yang Song of the DroughtFighters
project, Plant Microbe Interaction group, Utrecht University. Vinicius Lube
authorized publication of the derived checkpoints, fitted coefficients, split
identifiers, and sample-level validation measurements on 2026-09-04. The source
images and hand labels remain private and are not distributed by this release.

The stable corpus ID `external:yang-ground-truth-feb-2026` and related `yang`
filenames predate this public release. They are retained for reproducibility and
refer to Yang Song and the project affiliation recorded above.

## Public root datasets referenced by the experimental starter

- PRMI Dataset of Minirhizotron Images, Dryad DOI
  `10.5061/dryad.2v6wwpzp4`, recorded as CC0-1.0.
- RootPainter minirhizotron demo, Zenodo record `6992696`, recorded as
  CC-BY-4.0.
- RootNav 2.0 is referenced as a BSD-3-Clause codebase, not bundled as a
  training dataset.

The registry lives at
`resources/general_root_starter/public_dataset_registry.json`. Dataset licenses
must be checked again before publishing any derivative checkpoint.

## Plant health

The optional health checkpoint was trained from PlantVillage segmented images
and initialized from TorchVision/ImageNet weights. Its derivative redistribution
terms are not yet cleared, so the checkpoint is not part of normal Git history.
