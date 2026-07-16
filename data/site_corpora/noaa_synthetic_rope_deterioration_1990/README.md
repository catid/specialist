# NOAA synthetic-rope deterioration corpus

This bounded package contains a manually written, training-source-only digest of Seo, Backer, and Mandell’s 1990 historical mechanism model for synthetic double-braided marine rope. It explains internal fiber-to-fiber abrasion and tensile creep-fatigue without retaining service-life calculations, numerical parameters, figures, or tables.

Files:

- `CORPUS.md` — claim-cited historical methods digest and public-domain source notice.
- `REPORT.md` — complete 41-page OCR/visual review, scientific audit, rights screen, and exclusions.
- `dispositions.jsonl` — one manual disposition for each PDF page.
- `sources.jsonl` — official NOAA record and bound-PDF retrieval metadata.
- `source_snapshot/provenance.json` — non-content retrieval, OCR, review, and rights audit.
- `manifest.json` — package metadata, split hygiene, and output hashes.
- `tests/test_corpus.py` — deterministic content-boundary and integrity checks.

Run the bounded tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s data/site_corpora/noaa_synthetic_rope_deterioration_1990/tests -p 'test_*.py'
```

Do not derive validation, holdout, evaluation, OOD, or probe artifacts from this package. Any future training Q&A must keep predictions attached to the 1990 model’s construction, material, test, and evidence limits and must not become rope-use guidance.
