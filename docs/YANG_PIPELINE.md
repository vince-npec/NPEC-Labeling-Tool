# Yang Lucifer RGB Five-Seedling Pipeline

## Release identity

- Application release: `2026.8.2.2`
- Production interface: Lazy Folder in the Segmentation Pipeline tab
- Preset: `Arabidopsis - Lucifer RGB inoculated hybrid (5 seedlings)`
- Existing-mask ownership pipeline: `existing-mask-ownership-v20-visual-temporal-shoot-memory-schema-9`
- Owner profile: `yang-rgb-five-seedling-owner-v1`
- Pixel scale: `0.02663809523809524 mm/pixel` (`111.88 / 4200`)

The Lazy Folder application path is authoritative for raw-image processing.
Scripts named `run_yang_*` are evaluation tools and are not invoked by the GUI.

## Processing sequence

1. **Discovery and identity.** Supported images are discovered recursively.
   Filenames, timestamps, folder structure, OCR, and QR/barcode evidence can be
   reviewed before the app writes `npec_plate_identity_manifest.csv`.
2. **Stabilization.** Frames are grouped by plate and sorted chronologically.
   The recommended Yang setting is previous-frame registration against the
   plant/top-root region. Registration uses phase correlation with ORB/RANSAC
   fallback and conservative transform limits.
3. **Semantic inference.** RGB is normalized to `[0,1]` and processed in
   overlapping 256-pixel tiles. The hybrid preset uses the Lucifer RGB root
   checkpoint and the inoculated RGB shoot checkpoint listed below.
4. **Indexed classes.** `1` is primary/root, `2` is shoot, and `3` is lateral
   root. The preset now records shoot postprocessing as `guarded`, matching the
   behavior historically executed when the old `native` label resolved to the
   compatibility layer's guarded mode.
5. **Primary/lateral decomposition.** A skeleton graph is rooted near crown
   candidates. Turn-penalized routes identify primary roots; residual eligible
   branches become lateral roots. This step operates on visible root pixels.
6. **Temporal ownership.** Five stable left-to-right crowns define persistent
   lanes. Assignment combines crown connectivity, lane geometry, previous tips,
   previous owner masks, temporal continuity, and the Yang shared linear ranker
   as a constrained tie-breaker for graph ambiguities.
7. **Measurement.** Root length uses weighted 8-neighbor skeleton edges. Shoot
   area uses squared pixel scale. Strict RGB crown-green masks may replace the
   model class for reported shoot measurements and video overlays.
8. **QC and rendering.** Presence and ownership quality gates can invalidate
   uncertain seedlings. The source-image video renders masks, graphs, and owner
   boxes when valid ownership data is present.

The per-frame primary/lateral decomposition call is static. Temporal continuity
is applied later during ownership and reporting, not during raw class-1/class-3
decomposition.

## Models used for Yang

| Component | Release path | Original SHA-256 |
|---|---|---|
| Lucifer RGB root | `resources/builtin_models/hades_lucifer/best_root_model_patch_256_max_f10.8_max_IoU0.905.h5` | `2034ddeb36a980e2ebdb67871c059c2746ed7fbaba25e4e500d61bcb5a14059e` |
| Inoculated RGB shoot v3 | `resources/builtin_models/rgb_inoculated/rgb_inoculated_shoot_v3.keras` | `5064497983311af53bc5a070f165fcf8ccb909226e42100d753c3ff059c962ea` |
| Five-seedling owner ranker v1 | `resources/builtin_models/five_seedling_ownership/yang_rgb_owner_ranker_v1.json` | `d4d14f84186fd84b218aada81790ea36c3f99e634d32c1684af925931b7059ae` |

The public owner JSON has only machine-specific provenance paths sanitized. Its
canonical `ranker` payload SHA-256 remains
`4ac0f1a962b30d26b6793373b15b8e31dae5f8f7175e3a186495751a659355ff`.

The shoot profile reports validation IoU `0.675085`, holdout IoU `0.598109`,
and predicted/truth shoot-area ratio `1.260918`. Only the shoot channel was
qualified; the root and seed channels were not promoted.

## Ownership validation

The final profile was fitted from 58 annotated samples across 48 plate groups
and 101,500 sampled pixel rows. A five-plate challenge used a ranker trained
without the evaluation groups:

| Metric | Lane baseline | Ranker-gated |
|---|---:|---:|
| Conditional pixel accuracy | 0.7588 | 0.8137 |
| Conditional macro Dice | 0.7974 | 0.8536 |
| Visible-label full macro Dice | 0.7197 | 0.7689 |
| Per-seedling length MAPE | 0.1281 | 0.1024 |
| Per-seedling length WAPE | 0.1335 | 0.0968 |

The frozen split, evaluation ranker, threshold sweep, case metrics, and report
are in [`docs/validation/yang_2026-07-23`](validation/yang_2026-07-23).
Private validation images are not distributed.

The challenge supports the method design, not an independent test of the final
parameter vector: after evaluation, the bundled final profile was refitted on
all 58 annotations.

## Interpretation limits

- The ranker allocates existing visible root pixels; it cannot create roots.
- It does not infer biological provenance at an irrecoverably merged crossing.
- Lateral-root pixels inherit the owner of the connected root system.
- The ranker cannot override an unambiguous temporal assignment.
- Bacterial gaps remain unknown/background in the visible-label benchmark.
- GAN gap repair is optional and was not part of the default Yang profile.
- Plate-total measurements are conserved independently of ownership.

## Existing-mask reproduction

The following command reruns ownership and reporting from indexed masks. The
manifest must contain `Series`, `PetriDish`, `Timestamp`, `FrameIndex`,
`SourceFile`, `OutputMaskPath`, and `PreviewImagePath` columns.

```bash
python -m resources.rerun_existing_mask_ownership \
  --manifest /path/to/ownership_manifest.csv \
  --output-dir /path/to/ownership_rerun \
  --expected-plants 5 \
  --pixel-size-mm 0.02663809523809524 \
  --root-class-id 1 \
  --lateral-class-id 3 \
  --shoot-class-id 2 \
  --owner-ranker-profile resources/builtin_models/five_seedling_ownership/yang_rgb_owner_ranker_v1.json
```

This command does not perform raw RGB semantic inference.

## Suggested Methods language

Images were processed with NPEC Labeling Tool release `2026.8.2.2` using the
Lucifer RGB inoculated Arabidopsis five-seedling preset. Frames were grouped by
plate and registered sequentially using the plant/top-root region. Visible root
pixels were segmented by tiled RGB inference and decomposed into primary and
lateral classes by crown-anchored skeleton topology. Per-seedling allocation
used five temporally tracked crown lanes plus a constrained linear ranker to
resolve ambiguous graph assignments; plate-total length remained independent
of ownership. Root length was calculated from weighted 8-neighbor skeleton
edges at `0.02663809523809524 mm/pixel`. Bacteria-occluded gaps were treated as
unknown and were not automatically imputed by the default profile.
