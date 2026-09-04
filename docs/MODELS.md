# Model Manifest and Release Status

Model filenames and hashes are recorded in `docs/MODEL_SHA256SUMS`. Binary
checkpoints are excluded from normal Git history and distributed separately as
checksum-verified release assets when their rights have been cleared.

| Family | Intended use | Status |
|---|---|---|
| `hades_lucifer` | Hades grayscale and Lucifer RGB root/shoot segmentation | Exact Yang root artifact cleared; other checkpoints require separate review |
| `rgb_inoculated` | Yang/inoculated Lucifer RGB shoot segmentation | Exact Yang shoot v3 artifact cleared; shoot-only qualification recorded |
| `five_seedling_ownership` | Five-lane Yang root ownership tie-breaking | Yang owner-ranker and validation records cleared |
| `potato` | Single-plant Lucifer potato endpoint segmentation | Provenance clearance required |
| `mxlab_dark_rgb` | Dark-background RGB seed/root/shoot segmentation | Provenance clearance required |
| `general_root_starter_unified` | Experimental cross-domain root starter | Experimental; low validation IoU; not a default production model |
| `plant_health` | Optional PlantVillage-derived health classifier | PlantVillage/ImageNet derivative rights require review |

The NPEC application can use external model paths, so the source release remains
usable for annotation, review, analytics, and user-supplied checkpoints.

The combined card for the exact Yang production artifacts is in
[`docs/model-cards/YANG_PIPELINE_MODELS.md`](model-cards/YANG_PIPELINE_MODELS.md).

## Yang-specific qualification

Vinicius Lube, the named release rights holder, authorized public redistribution
of the exact Yang Song checkpoint, owner-ranker, and sample-level validation
artifacts under Apache-2.0 on 2026-09-04. This grant does not extend to the
private source images or hand-label corpus.

The Yang shoot checkpoint is qualified only for shoot segmentation. Its profile
explicitly prohibits treating its root or seed channels as production outputs.
The owner ranker is an ambiguity resolver over existing visible roots, not a
segmentation model and not a biological ownership oracle.

## Adding a distributable checkpoint

Before a model is attached to a release, add a model card containing:

- copyright owner and redistribution grant;
- training and validation dataset identifiers and licenses;
- architecture and initialization provenance;
- exact training code revision and environment;
- intended use, excluded use, and known failure modes;
- split design and leakage controls;
- metrics with denominators and uncertainty where available;
- SHA-256 hash and expected application preset.

After clearance, an authorized release manager can build the checksum-verified
Yang archive with:

```bash
python -m resources.scripts.package_yang_model_release \
  --confirm-redistribution-rights
```
