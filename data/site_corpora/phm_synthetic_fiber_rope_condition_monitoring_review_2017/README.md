# PHM synthetic-fiber-rope condition-monitoring review corpus

This package is a manually audited, rights-filtered Markdown distillation of a 2017 PHM Society review about offshore synthetic-fiber-rope condition monitoring. It preserves the authors' qualitative construction hierarchy, monitoring taxonomy, limitations, and research gaps while excluding third-party visual components and unsafe or overgeneralized operational claims.

The package is marked `direct_training_ready: true` because the retained prose is attributable article text or a close factual distillation covered by the article-level CC BY 3.0 US license, every page and visual component was reviewed, and deterministic tests enforce the intended boundary.

Files:

- `CORPUS.md`: dense, claim-cited training text.
- `sources.jsonl`: source identity, rights, retrieval, and checksum record.
- `dispositions.jsonl`: one manual disposition for each source page.
- `components.jsonl`: one exclusion decision for each numbered figure.
- `source_snapshot/provenance.json`: exact first-party body provenance.
- `REPORT.md`: audit method, exclusions, and findings.
- `manifest.json`: checksums for package artifacts.
- `tests/test_corpus.py`: deterministic integrity and content-boundary checks.

The original PDF is not redistributed in this package. Its exact checksum and official download URL are recorded so the reviewed body can be independently verified.

