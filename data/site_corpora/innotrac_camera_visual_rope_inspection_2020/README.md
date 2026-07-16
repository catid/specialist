# innoTRAC camera-based industrial rope-inspection corpus

This package is a manually audited, manually rewritten, quotation-light digest of Gregor Novak's 2020 article, *Camera-based Visual Rope Inspection*, innoTRAC Journal, volume 1, pages 55–63, DOI `10.14464/innotrac.v1i0.462`.

The package is marked `direct_training_ready: true`. It uses only the [official article record](https://www.bibliothek.tu-chemnitz.de/ojs/index.php/innoTRAC/article/view/462), the exact [official final PDF](https://www.bibliothek.tu-chemnitz.de/ojs/index.php/innoTRAC/article/view/462/186), and the [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en). The article record and PDF both explicitly license the article under CC BY 4.0. The legal code is used solely for rights provenance.

The retained corpus is limited to bounded industrial inspection-method evidence: one-sided manual viewing, fatigue and perceptual habituation, absence of a continuous manual record, multi-angle surface imaging, surface-indicator combination, exploratory project and specimen scope, funding and commercial-platform disclosure, surface-versus-internal observability, validation gaps, and the paper's admission that high-modulus fibre-rope discard criteria remained insufficiently researched.

The package excludes all nine figures, all numerical outputs, the logical discard rule, thresholds, equations, camera and equipment setup, processing recipes, algorithms, code, products, trademarks, proposed discard rules, ratings, remaining-life or internal-condition claims, reference bodies, and third-party or uncertain components. The article contains no tables.

Files:

- `CORPUS.md`: dense, claim-cited, non-operational industrial inspection-method digest.
- `sources.jsonl`: exact identities, retrieval metadata, rights, and checksums for the three allowed bodies.
- `surfaces.jsonl`: manual dispositions for all material surfaces of the article record and CC legal-code page.
- `dispositions.jsonl`: one manual disposition for each of the nine physical PDF pages.
- `components.jsonl`: complete figure, logical-rule, numerical-output, setup, processing, trademark, standards, reference, and unsafe-inference exclusions.
- `source_snapshot/provenance.json`: exact retrieval, DOI, robots, PDF metadata, license, authorship, disclosure, and audit record.
- `REPORT.md`: review method, retained scope, exclusions, rights, disclosures, and safety boundary.
- `manifest.json`: package schemas, readiness, statistics, and artifact hashes.
- `tests/test_corpus.py`: deterministic integrity, provenance, scope, and leakage checks.

No PDF, HTML, extraction, rendered page, figure, source image, reference body, or source body is redistributed. Only official URLs, exact checksums, and audit metadata are retained for independent verification.

Attribution: © 2020 Gregor Novak, *Camera-based Visual Rope Inspection*, innoTRAC Journal, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This adaptation was manually rewritten, narrowed, and safety-bounded on 2026-07-16. No endorsement is implied.
