# MAIB throw-bag hidden fused-joints corpus

This package is a manually audited, rights-filtered digest of the Crown-authored qualitative failure analysis in MAIB Report No. 2/2019. It focuses on a nominally continuous rescue line assembled from thermally fused segments, the hidden joint failure, comparable defects, limits of random visual quality control, and the distinct roles of traceability, batch control, process validation, output checks, and independent verification.

The package is marked `direct_training_ready: true`. The official GOV.UK endpoints were normally accessible; the exact report body was checksumed; OGL v3 scope was verified for Crown content; all eight pages were audited; and every figure, photograph, table, legal or standards block, external test block, vendor block, and uncertain component was excluded.

Files:

- `CORPUS.md`: dense, claim-cited qualitative synthesis.
- `sources.jsonl`: source identity, rights, retrieval, and exact checksums.
- `dispositions.jsonl`: one manual decision per physical PDF page.
- `components.jsonl`: component-level exclusion audit.
- `source_snapshot/provenance.json`: first-party body and rights provenance.
- `REPORT.md`: audit method, inclusion boundary, exclusions, and statistics.
- `manifest.json`: schema identifiers, readiness flag, and artifact hashes.
- `tests/test_corpus.py`: deterministic integrity and leakage checks.

The official report PDF is not redistributed. Its URL and exact checksum are recorded for independent verification.

