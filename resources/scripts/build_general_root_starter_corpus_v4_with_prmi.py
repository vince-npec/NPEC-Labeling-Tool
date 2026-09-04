from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OUTPUT_ROOT = REPO_ROOT / "resources" / "output" / "general_root_starter_full_push_v4_prmi"
V3_MANIFEST = (
    REPO_ROOT
    / "resources"
    / "output"
    / "general_root_starter_full_push_v3"
    / "corpus_v3"
    / "manifest.json"
)
PRMI_MANIFEST = (
    REPO_ROOT
    / "resources"
    / "output"
    / "prmi_import_v1"
    / "manifest.json"
)


def _merge_manifests(base_manifest: dict[str, object], extra_manifest: dict[str, object]) -> dict[str, object]:
    merged_samples = list(base_manifest.get("samples") or []) + list(extra_manifest.get("samples") or [])
    family_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for sample in merged_samples:
        if not isinstance(sample, dict):
            continue
        family = str(sample.get("family") or "unknown")
        split = str(sample.get("split") or "train")
        family_counts[family] = int(family_counts.get(family, 0) + 1)
        split_counts[split] = int(split_counts.get(split, 0) + 1)
    source_rows = list(base_manifest.get("sources") or []) + list(extra_manifest.get("sources") or [])
    return {
        "name": "general_root_starter_corpus_v4_with_prmi",
        "version": 1,
        "schema_keys": list(base_manifest.get("schema_keys") or extra_manifest.get("schema_keys") or []),
        "random_seed": int(base_manifest.get("random_seed") or 17),
        "train_fraction": float(base_manifest.get("train_fraction") or 0.70),
        "val_fraction": float(base_manifest.get("val_fraction") or 0.15),
        "sources": source_rows,
        "family_counts": family_counts,
        "split_counts": split_counts,
        "sample_count": int(len(merged_samples)),
        "samples": merged_samples,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base_manifest = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))
    extra_manifest = json.loads(PRMI_MANIFEST.read_text(encoding="utf-8"))
    merged = _merge_manifests(base_manifest, extra_manifest)
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    summary = {
        "base_manifest": str(V3_MANIFEST),
        "prmi_manifest": str(PRMI_MANIFEST),
        "manifest_path": str(manifest_path),
        "sample_count": int(merged["sample_count"]),
        "family_counts": merged["family_counts"],
        "split_counts": merged["split_counts"],
    }
    (OUTPUT_ROOT / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
