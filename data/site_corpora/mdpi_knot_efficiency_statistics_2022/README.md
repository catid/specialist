# MDPI knot-efficiency statistics corpus

This bounded package contains a manually written, training-source-only digest of the statistical method in Šimon and Ftorek (2022). It explains knot efficiency as a ratio distribution without retaining the article’s calculators, numerical efficiencies, figures, or tables.

Files:

- `CORPUS.md` — claim-cited training source and CC BY attribution notice.
- `REPORT.md` — review, rights, provenance, and scope audit.
- `dispositions.jsonl` — section/page-level manual inclusion decisions.
- `sources.jsonl` — official source URLs, hashes, and retrieval metadata.
- `source_snapshot/provenance.json` — non-content audit record; source bodies are not stored.
- `manifest.json` — package metadata, split hygiene, statistics, and output hashes.
- `tests/test_corpus.py` — deterministic integrity and boundary checks.

Run the bounded tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s data/site_corpora/mdpi_knot_efficiency_statistics_2022/tests -p 'test_*.py'
```

Do not derive validation, holdout, evaluation, OOD, or probe artifacts from this package. Any future Q&A must be separately reviewed, training-only, and must preserve the distinction between population variability and finite-sample parameter uncertainty.
