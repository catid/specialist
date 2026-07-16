# Manual curation report

## Source decision

Accepted. The University of Twente research record is the official institutional landing page. It names the article, authors, journal, DOI, publication date, final-published-version PDF, and `CC BY` license. Its license link resolves to Creative Commons Attribution 4.0. The PDF was retrieved directly from the University's `ris.utwente.nl` portal, returned as `application/pdf`, was not encrypted, and contained 17 pages.

## Review method

The reviewer:

1. retrieved the official landing HTML and official final PDF;
2. recorded response metadata and SHA-256 checksums for both objects, plus SHA-512 for the PDF;
3. extracted text separately for every page;
4. rendered all 17 pages at 150 pixels per inch;
5. visually inspected every page in contact sheets;
6. inspected every page supporting a retained claim again at the full rendered resolution;
7. assigned one disposition to every page; and
8. manually rewrote the retained evidence into a qualitative, cited corpus.

No secondary copy, unofficial mirror, automated summary, or unreviewed OCR text was used as corpus prose.

## Inclusion policy

Retained:

- the exact Twaron 2200, laid three-strand specimen scope;
- rope → strand → yarn → filament hierarchy;
- distinction between rope twist and strand twist;
- the tensile-test and calibrated pressure-film workflow;
- measurement limitations and replication logic;
- qualitative rope-twist, strand-twist, and filament-count comparisons;
- ideal-helix model assumptions and observed mismatch;
- the limited, empirical nature of the fitted correction;
- data-availability language; and
- full funding, specimen-support, and conflict disclosures.

Removed:

- equations and derivations;
- figures, tables, plots, specimen codes, and numerical settings;
- all numerical forces, pressures, thresholds, dimensions, twist rates, correction values, intervals, and projections;
- performance, durability, calculator, and rating claims;
- product or manufacturer language except neutral specimen and disclosure identification;
- transfer to natural-fiber or other untested rope families;
- unrelated operational domains; and
- the reference list as standalone training content.

## Manual resolution of a source inconsistency

The abstract, detailed results, mechanism discussion, and main conclusion say that increasing strand twist makes the strand stiffer, decreases contact width, and increases pressure for the same contact force. A carry-over sentence on page 16 instead says that increasing strand twist increases both contact width and pressure. Because that isolated phrase contradicts the paper's repeated result and explanation, it was treated as an editorial inconsistency and was not retained as fact. `CORPUS.md` explicitly alerts readers to the discrepancy.

## Overclaim control

The paper reports a fitted helix-angle correction and later describes prediction for any Twaron three-strand rope. The same page states that experiments were limited to a specific material and construction. This corpus retains only that a dataset-fitted adjustment improved agreement for tested specimens. It does not retain the correction value, confidence interval, formula, calculator-like use, or universal-prediction wording.

The paper also extrapolates measured filament-count trends beyond the tested specimens and discusses downstream degradation. Those extrapolations and outcome claims were excluded. The retained comparisons are framed as observations within the tested construction.

## Page audit summary

Pages 1 through 16 received `partial` dispositions because each contains at least one supportable qualitative claim but also contains excluded numerical, graphical, formulaic, speculative, or application material. Page 17 received `exclude` because it contains only the continuation of the reference list. See `dispositions.jsonl` for page-specific notes.

## Corpus statistics

Statistics and integrity hashes are generated from the committed files and checked by `tests/test_corpus.py`. The package contains one curated Markdown corpus, one source record, and exactly 17 page dispositions.
