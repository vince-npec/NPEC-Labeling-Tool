from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources.five_seedling_ownership import (  # noqa: E402
    discover_five_seedling_corpus,
)
from resources.learned_owner_assignment import SharedOwnerRanker  # noqa: E402
from resources.scripts.experiment_learned_owner_assignment import (  # noqa: E402
    _sample_training_pixels,
)


DEFAULT_CORPUS = REPO_ROOT / "data" / "yang_ground_truth"
DEFAULT_VALIDATION_REPORT = (
    REPO_ROOT
    / "docs"
    / "validation"
    / "yang_2026-07-23"
    / "retrial_report.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "resources"
    / "builtin_models"
    / "five_seedling_ownership"
    / "yang_rgb_owner_ranker_v1.json"
)
FEATURE_SET = "crown_coordinates_orientation"
VALID_SAMPLE_RE = re.compile(r"^\d+-\d+$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_profile(
    corpus: Path,
    output_path: Path,
    *,
    validation_report_path: Path | None,
    max_pixels_per_owner: int,
) -> dict[str, object]:
    samples = [
        sample
        for sample in discover_five_seedling_corpus(corpus)
        if VALID_SAMPLE_RE.match(sample.sample_id)
    ]
    if not samples:
        raise RuntimeError("No valid numbered Yang annotation samples were found.")

    feature_batches: list[np.ndarray] = []
    owner_batches: list[np.ndarray] = []
    for index, sample in enumerate(samples, start=1):
        print(f"TRAINING SAMPLE {index}/{len(samples)} {sample.sample_id}", flush=True)
        features, owners = _sample_training_pixels(
            sample,
            max_pixels_per_owner=max(25, int(max_pixels_per_owner)),
            feature_set=FEATURE_SET,
        )
        feature_batches.append(features)
        owner_batches.append(owners)
    train_features = np.concatenate(feature_batches, axis=0)
    train_owners = np.concatenate(owner_batches, axis=0)
    ranker = SharedOwnerRanker(l2=1.0e-3).fit(train_features, train_owners)

    validation: dict[str, object] = {}
    if validation_report_path is not None and validation_report_path.exists():
        report = json.loads(validation_report_path.read_text(encoding="utf-8"))
        confidence_gate = report.get("confidence_gate")
        confidence_gate = confidence_gate if isinstance(confidence_gate, dict) else {}
        validation = {
            "source_report": str(validation_report_path.name),
            "source_report_sha256": _sha256(validation_report_path),
            "selection": report.get("selection", {}),
            "challenge_mean": report.get("challenge_mean", {}),
            "confidence_gate": confidence_gate,
            "scientific_scope": report.get("scientific_scope", {}),
        }

    profile: dict[str, object] = {
        "profile_version": "yang-rgb-five-seedling-owner-v1",
        "model_type": "shared_owner_linear_ranker",
        "feature_set": FEATURE_SET,
        "ranker": ranker.to_payload(),
        "support_margin": 0.50,
        "prior_weight_px": 4.0,
        "temporal_score_bonus": 0.18,
        "qc_margin": 0.10,
        "fusion": {
            "mode": "confidence_preserving_temporal_graph_tiebreak",
            "direct_pixel_override": False,
            "ambiguity_preserved": True,
            "support_margin_note": (
                "0.50 is the conservative calibration-sweep point; learned evidence "
                "may resolve an existing graph ambiguity but cannot alter ownership "
                "outside the baseline ambiguity mask."
            ),
        },
        "training": {
            "corpus": f"external:{corpus.name}",
            "sample_count": int(len(samples)),
            "plate_group_count": int(len({sample.plate_group for sample in samples})),
            "sample_ids": [sample.sample_id for sample in samples],
            "sampled_pixel_rows": int(train_owners.size),
            "max_pixels_per_owner": int(max_pixels_per_owner),
            "optimization": dict(ranker.optimization_),
        },
        "independent_validation": validation,
        "scope": {
            "expected_seedlings": 5,
            "input": "existing visible semantic root/lateral pixels plus five tracked crowns",
            "output": "left-to-right seedling owner slot 1..5",
            "plate_total_conservation": True,
            "hidden_root_repair": False,
            "biological_ownership_claim": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the final Yang five-seedling owner ranker profile."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=DEFAULT_VALIDATION_REPORT,
    )
    parser.add_argument("--max-pixels-per-owner", type=int, default=350)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    profile = train_profile(
        args.corpus.expanduser().resolve(),
        args.output.expanduser().resolve(),
        validation_report_path=(
            args.validation_report.expanduser().resolve()
            if args.validation_report
            else None
        ),
        max_pixels_per_owner=max(25, int(args.max_pixels_per_owner)),
    )
    print(
        f"DONE output={args.output.expanduser().resolve()} "
        f"samples={profile['training']['sample_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
