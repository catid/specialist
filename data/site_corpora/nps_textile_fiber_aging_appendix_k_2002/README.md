# NPS museum-textile fiber aging corpus

This package is a manually audited, text-only mechanism digest from the National Park Service *Museum Handbook, Part I*, Appendix K, *Curatorial Care of Textile Objects*. Retained content is limited to the K:2–K:13 scope and explicitly labeled as museum-textile evidence.

The package is marked `direct_training_ready: true`. The exact official NPS PDF was normally accessible and checksumable; the NPS-authored federal text is treated as public domain in the United States; all fifty physical pages were audited; and every numbered figure, photograph, credited or permission-controlled visual, vendor reference, quantitative prescription, procedural passage, and out-of-scope page was excluded.

Files:

- `CORPUS.md`: claim-cited museum-textile mechanism digest.
- `sources.jsonl`: source identity, retrieval, rights, and exact checksums.
- `dispositions.jsonl`: one manual decision for each physical PDF page.
- `components.jsonl`: exclusion audit for all numbered figures and controlled source blocks.
- `source_snapshot/provenance.json`: exact first-party body and public-domain provenance.
- `REPORT.md`: scope, audit method, retained mechanisms, and exclusions.
- `manifest.json`: schema identifiers, readiness flag, and artifact hashes.
- `tests/test_corpus.py`: deterministic integrity, labeling, and leakage checks.

The original PDF is not redistributed. Its official URL and exact checksum are recorded for independent verification.

