# Yang Song Pipeline Model Card

## Scope

This card covers the three artifacts selected by the production preset
`Arabidopsis - Lucifer RGB inoculated hybrid (5 seedlings)` in release
`2026.8.2.4`.

The dataset and workflow provenance is Yang Song, DroughtFighters project,
Plant Microbe Interaction group, Utrecht University. Stable `yang` identifiers
in artifact filenames, paths, and profile keys refer to Yang Song and remain
unchanged for checksum-level reproducibility.

## Visible-root checkpoint

- File: `best_root_model_patch_256_max_f10.8_max_IoU0.905.h5`
- SHA-256: `2034ddeb36a980e2ebdb67871c059c2746ed7fbaba25e4e500d61bcb5a14059e`
- Framework metadata: Keras 2.10 / TensorFlow functional model
- Input: normalized 256 by 256 single-channel tile
- Production role: visible-root probability for the Lucifer RGB hybrid route
- Output use: thresholded root support before primary/lateral decomposition

The score embedded in the historical filename is not treated as a release
validation claim because its original split records are not available in this
snapshot. The checkpoint must not be used to infer roots through opaque
bacterial regions without a separately reported repair protocol.

## Inoculated RGB shoot checkpoint

- File: `rgb_inoculated_shoot_v3.keras`
- SHA-256: `5064497983311af53bc5a070f165fcf8ccb909226e42100d753c3ff059c962ea`
- Production role: shoot-class proposal for Lucifer RGB inoculated plates
- Recorded validation IoU: `0.675085`
- Recorded holdout IoU: `0.598109`
- Predicted/truth shoot-area ratio: `1.260918`

Only the shoot channel is qualified. The profile explicitly disallows using its
root or seed channels as production outputs. Reported shoot measurements may
use the stricter RGB crown-green recomputation selected by the preset.

## Five-seedling owner ranker

- File: `yang_rgb_owner_ranker_v1.json`
- Sanitized-file SHA-256: `a000451852dfd3ea088abeddcf62799e495efd67533a6ac9032890f4779bcb85`
- Original-file SHA-256: `d4d14f84186fd84b218aada81790ea36c3f99e634d32c1684af925931b7059ae`
- Canonical ranker-payload SHA-256: `4ac0f1a962b30d26b6793373b15b8e31dae5f8f7175e3a186495751a659355ff`
- Training record: 58 annotated samples, 48 plate groups, 101,500 sampled rows
- Production role: constrained tie-breaker for ambiguous five-lane ownership

The ranker allocates already segmented visible pixels. It cannot create root
pixels, establish provenance at irrecoverably merged crossings, or override an
otherwise unique temporal assignment. Lateral roots inherit the owner of their
connected root system.

The five-plate challenge excluded evaluation groups while selecting the method,
but the final bundled profile was then refitted on all 58 annotations. Challenge
metrics therefore support the method design and are not an independent test of
the final fitted coefficient vector. Exact metrics and split records are in
`docs/YANG_PIPELINE.md` and `docs/validation/yang_2026-07-23/`.

## Redistribution status

Vinicius Lube, the named release rights holder, authorized public redistribution
of these exact Yang Song artifacts and the sample-level validation records on
2026-09-04 under Apache-2.0. Checkpoints are published outside normal Git
history as a checksum-verified release asset and may be bundled in the matching
installer. This authorization does not include the private source images or
hand labels.
