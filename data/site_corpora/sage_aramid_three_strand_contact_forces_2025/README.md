# SAGE / University of Twente aramid contact-force corpus

This package contains a manually written qualitative adaptation of a rights-verified research article on contact between strands in tensile tests of Twaron 2200 laid three-strand ropes.

## Contents

- `CORPUS.md`: information-dense training text with page-level citations.
- `sources.jsonl`: canonical source, license, bibliographic, retrieval, and checksum metadata.
- `dispositions.jsonl`: exactly one manual disposition for each of the PDF's 17 pages.
- `source_snapshot/provenance.json`: official-host retrieval and rights-verification record.
- `REPORT.md`: audit method, inclusion rules, known source inconsistency, and corpus limits.
- `manifest.json`: byte counts and SHA-256 checksums for package files other than the manifest itself.
- `tests/test_corpus.py`: structural, scope, citation, disclosure, and integrity checks.

## Intended use

The corpus preserves only qualitative research knowledge supported by the tested material and construction. It is not an operating procedure, design calculator, product endorsement, or basis for transferring findings to a different rope family.

## License and attribution

The official University of Twente landing page labels the final published version `CC BY`, links to Creative Commons Attribution 4.0, and serves the final PDF. `CORPUS.md` is a substantially paraphrased and reorganized adaptation. The source authors, title, DOI, official URLs, license URL, and cryptographic checksums are recorded in `sources.jsonl` and `source_snapshot/provenance.json`.

## Reproducibility

From the repository root, run:

```bash
python3 -m unittest discover -s data/site_corpora/sage_aramid_three_strand_contact_forces_2025/tests -v
```

The source PDF is not duplicated in this directory. Its official institutional URL and two checksums are retained so a reviewer can retrieve and verify the same object.
