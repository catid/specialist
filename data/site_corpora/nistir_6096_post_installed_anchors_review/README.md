# NISTIR 6096 historical concrete-anchor review corpus

This package is a manually audited, quotation-light digest of NISTIR 6096, *Post-Installed Anchors—A Literature Review*, by Geraldine S. Cheok and Long T. Phan, published in 1998.

The package is marked `direct_training_ready: true`. It uses only the official NIST publication record, the exact official 109-page NISTIR PDF, and the official NIST technical-series rights page. The rights page is used solely for provenance. Eligible NIST employee-authored synthesis is treated as United States public-domain federal text, with NIST's stated residual foreign-rights grant; all third-party or uncertain components are excluded.

The corpus is limited to historical failure-mode vocabulary, research-variable distinctions, evidence gaps, model limits, internal citation leads, and a current qualified-engineering boundary. It contains no capacities, numerical performance results, operational loads, thresholds, products, standards, figures, tables, equations, procedures, or design claims.

Files:

- `CORPUS.md`: claim-cited, explicitly labeled non-bondage concrete-anchor digest.
- `sources.jsonl`: exact identities, retrieval metadata, rights, and checksums for all three allowed bodies.
- `surfaces.jsonl`: manual dispositions for NIST publication-record and rights-page surfaces.
- `dispositions.jsonl`: one manual disposition for every physical PDF page.
- `components.jsonl`: figure, table, and controlled-block exclusion audit.
- `source_snapshot/provenance.json`: exact retrieval, PDF metadata, rights basis, and audit record.
- `REPORT.md`: review method, retained scope, exclusions, and current-use boundary.
- `manifest.json`: schemas, readiness flag, statistics, and artifact hashes.
- `tests/test_corpus.py`: deterministic integrity, scope, and leakage checks.

The three allowed source bodies are not redistributed. Their official URLs and exact checksums are recorded for independent verification.
