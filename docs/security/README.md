# Dependency Audit

`pip-audit-macos-arm64-2026.8.2.2.json` records the 2026-09-04 audit of the
frozen macOS ARM64 development environment. Provenance-only release
`2026.8.2.3`, acknowledgement-only release `2026.8.2.4`, and documentation and
provenance release `2026.8.2.5` reuse that byte-identical dependency inventory
and SBOM.

The audit reported one advisory: `PYSEC-2026-1805` / `CVE-2026-0994` affects
`protobuf 4.25.9` when an application parses adversarial deeply nested
`google.protobuf.Any` JSON through `ParseDict`.

NPEC Labeling Tool does not expose a protobuf-JSON import surface, which limits
the known protobuf advisory's reachable attack path, but this is not a claim of
general immunity. TensorFlow 2.16 constrains protobuf below version 5, while the
audited fixed protobuf versions begin at 5.29.6. A final installer should be
rebuilt and re-audited when a compatible TensorFlow/protobuf pair is qualified.

The source repository does not vendor these Python packages. The release build
uses updated `pip 26.2.1` and `setuptools 84.0.0`; advisories previously
reported for older build tools are no longer present in the frozen audit.
Security reports should follow `SECURITY.md`.
