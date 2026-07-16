# FHWA adhesive-anchor failure and governance corpus

This package is a manually audited, text-only digest of Federal Highway Administration Technical Advisory T5140.34, *Use and Inspection of Adhesive Anchors in Federal-Aid Projects*. It retains only bounded civil-infrastructure failure and governance lessons.

The package is marked `direct_training_ready: true`. The exact four-page first-party PDF was normally accessible and checksumable. FHWA-authored federal prose is treated as public domain in the United States. Incorporated standards, organization and certification-program text, product and material detail, precise loads and durations, procedures, the schematic, and the reproduced table are excluded.

Files:

- `CORPUS.md`: claim-cited, explicitly labeled non-bondage civil-infrastructure digest.
- `sources.jsonl`: source identity, exact retrieval, rights, and checksums.
- `dispositions.jsonl`: one manual disposition for each physical PDF page.
- `components.jsonl`: categorical component and controlled-block exclusions.
- `source_snapshot/provenance.json`: first-party provenance, PDF metadata, and audit record.
- `REPORT.md`: review method, retained lessons, limitations, and exclusions.
- `manifest.json`: schema identifiers, readiness flag, statistics, and artifact hashes.
- `tests/test_corpus.py`: deterministic integrity, scope, and leakage checks.

The original PDF is not redistributed. Its official URL and exact checksum are recorded for independent verification.
