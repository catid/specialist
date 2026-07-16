# NIST manila-rope statistics corpus

This bounded package contains a manually written, training-source-only digest of Newman and Curtiss’s 1947 RP1847 analysis of three-strand manila-rope procurement tests. It preserves sampling, measurement, statistical assumptions, and anomalies without retaining strength tables, equations, plots, ratings, or procurement rules.

The verified article DOI is `10.6028/jres.039.039`. The task-supplied `.037` DOI is documented as an unrelated-paper binding and is not used as this article’s identifier.

Files:

- `CORPUS.md` — claim-cited historical statistical-methods digest and attribution.
- `REPORT.md` — complete nine-page review, DOI correction, evidence audit, and rights screen.
- `dispositions.jsonl` — one manual disposition per PDF page.
- `sources.jsonl` — official PDF, DOI, and NIST rights-source ledger.
- `source_snapshot/provenance.json` — non-content retrieval, document, review, DOI, and rights audit.
- `manifest.json` — package metadata, split hygiene, and output hashes.
- `tests/test_corpus.py` — deterministic boundary and integrity checks.

Run the bounded tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s data/site_corpora/nist_manila_rope_statistics_1947/tests -p 'test_*.py'
```

Do not derive validation, holdout, evaluation, OOD, or probe artifacts from this package. Any future training Q&A must remain historical, population-bound, and non-operational.
