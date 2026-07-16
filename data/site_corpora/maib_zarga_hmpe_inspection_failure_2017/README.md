# MAIB Zarga jacketed-HMPE failure-analysis corpus

This package is a manually audited, rights-filtered Markdown digest of the Crown-authored analysis in MAIB Report No. 13/2017. It preserves the industrial maritime context, construction-specific failure model, monitoring blind spots, system-compatibility analysis, information-flow failures, and limits on transferring offshore test evidence.

The package is marked `direct_training_ready: true`. Both official GOV.UK endpoints were normally accessible, the exact PDF was checksumed, OGL v3 coverage was verified for Crown content, all 116 pages were manually reviewed, and every visual, table, annex listing, and separately credited source was excluded conservatively.

Files:

- `CORPUS.md`: information-dense, claim-cited Crown-authored synthesis.
- `sources.jsonl`: source identity, access, rights, dates, and body checksums.
- `dispositions.jsonl`: one manual decision for each physical PDF page.
- `components.jsonl`: exclusion audit for every figure, table, annex, and credited source block.
- `source_snapshot/provenance.json`: exact first-party body and rights provenance.
- `REPORT.md`: audit method, inclusion boundary, exclusions, and statistics.
- `manifest.json`: schema/version, readiness flag, and artifact hashes.
- `tests/test_corpus.py`: deterministic integrity and content-boundary tests.

The PDF and separate annex bundle are not redistributed. The official URLs and exact checksum of the reviewed report are recorded for independent verification.

