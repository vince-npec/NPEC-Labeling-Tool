from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.general_root_starter.corpus import CorpusSourceSpec, build_general_root_corpus, infer_family_from_path  # noqa: E402


def _parse_source(raw: str) -> CorpusSourceSpec:
    text = str(raw).strip()
    if "=" not in text:
        path = Path(text).expanduser().resolve()
        return CorpusSourceSpec(project_path=path, family=infer_family_from_path(path), name=path.stem)
    family, raw_path = text.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    return CorpusSourceSpec(project_path=path, family=str(family).strip() or infer_family_from_path(path), name=path.stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multiple labeled NPEC projects into a normalized General Root Starter corpus.")
    parser.add_argument("--source", action="append", default=[], help="Source project spec as family=/path/to/project.oclp. If family is omitted, it will be inferred.")
    parser.add_argument("--output", type=Path, required=True, help="Output corpus directory")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-images-per-source", type=int, default=0)
    args = parser.parse_args()

    if not args.source:
        raise SystemExit("Provide at least one --source family=/path/to/project.oclp")
    sources = [_parse_source(raw) for raw in args.source]
    manifest = build_general_root_corpus(
        sources,
        args.output,
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
        random_seed=int(args.seed),
        max_images_per_source=int(args.max_images_per_source),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
