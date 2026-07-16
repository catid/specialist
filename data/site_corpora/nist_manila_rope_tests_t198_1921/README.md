# NBS T198 historical Manila-rope test corpus

This package is a manually audited, manually rewritten, quotation-light digest of NBS Technologic Paper T198, *Results of Some Tests of Manila Rope*, by Ambrose H. Stang and Lory R. Strickenberg, issued in 1921.

The package is marked `direct_training_ready: true`. It uses only the exact official NIST-hosted PDF, the official NIST 1921 Technologic Papers catalog page, and the official NIST technical-series rights page. The rights page is used solely for provenance. Eligible employee-authored federal text is adapted under NIST's technical-series policy; every third-party or uncertain component is excluded.

The retained corpus is limited to historical report identity, rope-construction vocabulary, sampling and test-design distinctions, manufacturer and procurement variability, measurement limitations, negative evidence, and explicit non-transfer boundaries. It contains no raw breaking-load or strength value, formula, coefficient, numerical table or result, procurement rule, product ranking, tolerance, modern recommendation, figure, image, equipment recipe, operational handling instruction, standard, or cited source body.

Files:

- `CORPUS.md`: dense, claim-cited, non-operational historical digest.
- `sources.jsonl`: exact identities, retrieval metadata, rights, and checksums for the three allowed bodies.
- `surfaces.jsonl`: manual dispositions for every material surface of the two HTML bodies.
- `dispositions.jsonl`: one manual disposition for each of the 13 physical PDF pages.
- `components.jsonl`: complete figure, table, formula, numerical-output, procedure, standard, recommendation, and uncertain-component exclusions.
- `source_snapshot/provenance.json`: exact retrieval, robots, DOI resolution, PDF metadata, rights, authorship, and audit record.
- `REPORT.md`: audit method, retained scope, exclusions, and safety boundary.
- `manifest.json`: package schemas, readiness, statistics, and artifact hashes.
- `tests/test_corpus.py`: deterministic integrity, provenance, scope, and leakage checks.

No PDF, HTML, OCR output, rendered page, photograph, plot, table, or source body is redistributed. Only exact official URLs, checksums, and audit metadata are retained for independent verification.

This adaptation was made on 2026-07-16 and materially rewrites, narrows, and safety-bounds the source. Republished courtesy of the National Institute of Standards and Technology.
