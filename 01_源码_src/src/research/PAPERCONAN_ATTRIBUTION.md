# PaperConan integration and attribution

CitrusQA vendors the upstream PaperConan 0.9.0 numeric-audit engine under the
MIT License. The copied engine is in `vendor/paperconan/`; its upstream license
is preserved as `vendor/PAPERCONAN_LICENSE`.

- Project: https://github.com/zixixr/paperconan
- Source revision: `8d116455b23e858651250d8ed23ba41c1a87ab0d`
- License: MIT
- Reviewed documentation: upstream `docs/detectors.md`, `docs/cli.md`, and
  `skills/paperconan/references/detectors.md`.

`paperconan_adapter.py` passes selected request attachments to the engine in a
unique temporary directory under CitrusQA's configured data directory. It
disables image inventory/diagnostics, creates session-bound downloadable
reports, and removes the temporary input directory when the scan returns. The
source workbook is never modified. The integration does not execute model-
generated code.

CitrusQA adds general table-quality checks and presents findings through the
existing Agent and artifact-download flow. Numeric signals are leads for human
review; they do not establish an error, author intent, or misconduct.
