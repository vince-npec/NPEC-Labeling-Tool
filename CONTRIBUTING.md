# Contributing

1. Open an issue describing the dataset type, expected behavior, and a minimal
   reproducible example that contains no private research data.
2. Create a focused branch and keep algorithm changes separate from generated
   outputs or model artifacts.
3. Add or update tests for behavioral changes.
4. Run `QT_QPA_PLATFORM=offscreen pytest -q` before opening a pull request.
5. Document model/data provenance for any new learned checkpoint.

Do not commit raw experimental images, `.oclp` projects containing private
images, credentials, patient/person data, or unpublished manuscript material.

By contributing, you certify that you have the right to submit the work under
the repository license.
