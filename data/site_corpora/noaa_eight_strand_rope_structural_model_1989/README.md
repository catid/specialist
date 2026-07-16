# NOAA eight-strand plaited-rope structural-model corpus

This bounded package contains a manually written, training-source-only digest of Wang and Backer’s 1989 qualitative structural model for eight-strand plaited synthetic marine rope. It preserves construction hierarchy, geometry, two inter-yarn friction endpoints, load aggregation, inter-strand pressure and relative motion, lateral-contraction sensitivity, and the distinction from the separate 1990 double-braided-rope deterioration report.

Files:

- `CORPUS.md` — claim-cited historical methods digest and public-domain source notice.
- `REPORT.md` — complete 51-page OCR/visual review, evidence audit, relationship audit, rights screen, and exclusions.
- `dispositions.jsonl` — one manual disposition for every PDF page.
- `sources.jsonl` — official NOAA record, checksum-bound main PDF, and relationship-only 1990 source metadata.
- `source_snapshot/provenance.json` — non-content retrieval, OCR, visual-review, rights, and source-relation metadata.
- `manifest.json` — package metadata, split hygiene, statistics, and output hashes.
- `tests/test_corpus.py` — deterministic content-boundary and integrity checks.

Run the bounded tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s data/site_corpora/noaa_eight_strand_rope_structural_model_1989/tests -p 'test_*.py'
```

Do not derive validation, holdout, evaluation, OOD, or probe artifacts from this package. Any future training Q&A must keep the 1989 model attached to eight-strand plaited synthetic rope, pre-damage axial modeling, and the authors’ qualitative-only limitation. It must not become rope-use or human-load guidance.
