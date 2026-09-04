# Release Provenance

## Snapshot origin

Release `2026.8.2.2` was assembled on 2026-09-04 from the active NPEC Labeling
Tool development working copy. That working copy did not contain prior Git
metadata, so this repository begins with a curated source snapshot rather than
an imported historical commit graph.

The release was assembled with an explicit allowlist. Experimental images,
`.oclp` projects, annotation corpora, generated outputs, paper drafts, local
environments, build products, unrelated laboratory software, and machine paths
were excluded.

## Scientific record

The exact Yang production workflow is documented in `docs/YANG_PIPELINE.md`.
Its tracked record includes:

- the production preset and class mapping;
- pixel scale and measurement algorithm;
- model filenames and SHA-256 checksums;
- the sanitized owner-ranker profile;
- frozen split and challenge metrics;
- validation limitations and suggested Methods language.

Private source images and hand annotations are not distributed. Vinicius Lube
authorized publication of the exact Yang Song checkpoints, owner-ranker, and
sample-level validation records on 2026-09-04.

The bundled PyPhenotyper compatibility layer was reconciled against the public
BSD-3-Clause NPEC lineage at `NPEC-NL/pyphenotyper` commit
`8b5c6f4eb1c5902d943917b129fb246ead9a8a9d`; its notice and contributor credit
are retained in this snapshot and in application bundles.

## Verification

The candidate source tree and a reconstructed public checkout without model
binaries both passed:

- 359 tests;
- 2 optional-backend skips;
- 191 parameterized subtests;
- an offscreen GUI construction and event-processing smoke test.

The macOS ARM64 candidate environment is frozen in `environments/`, with a
reproducible CycloneDX 1.6 SBOM in `docs/sbom/`.

The source-only installer rehearsal used the checksum-qualified no-FFmpeg
OpenCV wheel, passed deep code-signature and DMG verification, launched through
an offscreen Qt smoke test, and contained no bundled checkpoints, FFmpeg
executable, x264/x265 library, or dangling symlink. The release is ad-hoc signed
and not notarized until a Developer ID Application certificate is supplied.

The public checkout is intentionally capable of annotation, review, analytics,
and external-model use without bundled checkpoints. The authorized Yang Song
model pack and model-bearing installer are built from the tagged source with
checksum enforcement.
