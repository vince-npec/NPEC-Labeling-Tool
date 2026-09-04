NPEC PyPhenotyper

Bundled PyPhenotyper-compatible package for the NPEC Labeling Tool.

This compatibility implementation is a substantially revised descendant of
the BSD-3-Clause PyPhenotyper lineage frozen at
`NPEC-NL/pyphenotyper@8b5c6f4eb1c5902d943917b129fb246ead9a8a9d`. See the root
`THIRD_PARTY_NOTICES.md` and `licenses/PyPhenotyper-BSD-3-Clause.txt` for
licensing and contributor credit.

What it does:
- Exposes `pyphenotyper.features.features.model_create_masks(...)` for the app adapter.
- Supports legacy binary root/shoot `.h5` models.
- Supports newer multiclass `.keras` models through the same entrypoint.
- Returns masks in original image coordinates so the app can consume them without extra model-specific hacks.

Recommended starting models:
- Root: `resources/builtin_models/hades_lucifer/best_root_model_patch_256_max_f10.8_max_IoU0.905.h5`
- Shoot: `resources/builtin_models/hades_lucifer/best_shoot_model_patch_256_max_f10.834_max_IoU0.968.h5`

The general root starter in `resources/builtin_models/general_root_starter_unified/`
is experimental and is not the default Hades/Lucifer model.

Validation datasets and model directories must be supplied explicitly; they
are not distributed with the source repository.
