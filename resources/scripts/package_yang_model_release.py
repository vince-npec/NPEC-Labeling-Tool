from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile


ARTIFACTS = (
    (
        "hades_lucifer/best_root_model_patch_256_max_f10.8_max_IoU0.905.h5",
        "2034ddeb36a980e2ebdb67871c059c2746ed7fbaba25e4e500d61bcb5a14059e",
    ),
    (
        "rgb_inoculated/rgb_inoculated_shoot_v3.keras",
        "5064497983311af53bc5a070f165fcf8ccb909226e42100d753c3ff059c962ea",
    ),
    (
        "rgb_inoculated/rgb_inoculated_shoot_v3.profile.json",
        "db54b5bf54ecccc0e79674351cf61dd00f7c8746abdf02ee1ad24f1b4931fe83",
    ),
    (
        "five_seedling_ownership/yang_rgb_owner_ranker_v1.json",
        "a000451852dfd3ea088abeddcf62799e495efd67533a6ac9032890f4779bcb85",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_reproducible(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(2026, 9, 4, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def build_package(repo_root: Path, output: Path) -> Path:
    model_root = repo_root / "resources" / "builtin_models"
    verified: list[tuple[str, str, Path]] = []
    for relative, expected in ARTIFACTS:
        path = model_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required Yang Song workflow artifact is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {relative}: expected {expected}, got {actual}"
            )
        verified.append((relative, actual, path))

    output.parent.mkdir(parents=True, exist_ok=True)
    readme = (
        "NPEC Labeling Tool Yang Song model pack 2026.8.2.3\n\n"
        "Provenance: Yang Song, DroughtFighters project, Plant Microbe "
        "Interaction group, Utrecht University.\n\n"
        "These are the exact visible-root, inoculated-shoot, and five-seedling "
        "ownership artifacts documented in docs/YANG_PIPELINE.md.\n\n"
        "The artifacts are released under Apache-2.0 by the named rights "
        "holder. The private source images and hand-label corpus are not "
        "included.\n\n"
        "Place the extracted family folders under resources/builtin_models/ or "
        "select the checkpoints through the application's model controls.\n"
    ).encode("utf-8")
    checksums = "".join(
        f"{digest}  models/{relative}\n" for relative, digest, _path in verified
    ).encode("ascii")

    with zipfile.ZipFile(output, mode="w") as archive:
        _write_reproducible(archive, "README.txt", readme)
        _write_reproducible(archive, "MODEL_SHA256SUMS", checksums)
        for relative, _digest, path in verified:
            _write_reproducible(archive, f"models/{relative}", path.read_bytes())
        for relative in (
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "docs/YANG_PIPELINE.md",
            "docs/model-cards/YANG_PIPELINE_MODELS.md",
            "docs/DATA_PROVENANCE.md",
        ):
            path = repo_root / relative
            _write_reproducible(archive, relative, path.read_bytes())

    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(
        f"{_sha256(output)}  {output.name}\n",
        encoding="ascii",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the checksum-verified Yang Song workflow model release archive."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/NPEC-Yang-Song-model-pack-2026.8.2.3.zip"),
    )
    parser.add_argument(
        "--confirm-redistribution-rights",
        action="store_true",
        help="Required acknowledgement that model and training-data rights were cleared.",
    )
    args = parser.parse_args()
    if not args.confirm_redistribution_rights:
        parser.error(
            "Model packaging is blocked until --confirm-redistribution-rights is supplied "
            "by an authorized release manager."
        )
    output = args.output.expanduser().resolve()
    print(build_package(args.repo_root.expanduser().resolve(), output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
