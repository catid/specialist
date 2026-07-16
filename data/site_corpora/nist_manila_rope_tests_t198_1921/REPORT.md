# Manual audit report

## Outcome

This package is accepted as a quotation-light, bounded historical rope-construction and test-design corpus with `direct_training_ready: true`. It uses three exact official NIST bodies: the 1921 Technologic Papers catalog page, the T198 PDF, and the NIST technical-series rights page. No crawl, mirror, archive, search snippet, DOI metadata body, or linked source body was used as evidence.

The corpus retains only eligible employee-authored historical synthesis: document identity, yarn/strand/lay vocabulary, sampling and field-coverage distinctions, high-level specimen and measurement distinctions, manufacturer and procurement variability, uncontrolled variables, measurement limitations, negative evidence, and absolute non-transfer boundaries.

## Verified source identity

- Report: Bureau of Standards Technologic Paper T198, *Results of Some Tests of Manila Rope*
- Authors: Ambrose H. Stang and Lory R. Strickenberg
- Issue date on cover: September 15, 1921
- Report date at close: February 26, 1921
- DOI: `10.6028/nbst.6078`
- Official PDF pages: 13
- PDF bytes: 1207742
- PDF SHA-256: `bceb261d3ac009b71046b67a0ad400c632360bb7a61ba0c88c222ca31beb1f32`
- Retrieved: `2026-07-16T15:50:41Z`

The official NIST catalog gives the title, T198 identifier, abbreviated author names, and DOI. The report cover gives full names, Bureau positions, and issue date. The report closes with an earlier manuscript date. File digitization and hosting dates are retained only as provenance.

## Rights determination

The report title page identifies Stang as an associate physicist and Strickenberg as an assistant mechanical engineer of the Bureau of Standards. The official NIST catalog places T198 in the Technologic Papers series. NIST's technical-series policy states that employee-authored works are not subject to United States copyright, provides broad residual foreign-rights reuse, and warns that NIST publications can contain protected third-party work.

Accordingly, the package adapts only eligible employee-authored federal text. All photographs, plots, tables, visual marks, standards, purchase-specification bodies, manufacturer remarks, and uncertain components are excluded. The rights page is used only for rights provenance. This manually rewritten derivative was made on 2026-07-16 and is republished courtesy of the National Institute of Standards and Technology.

## Robots and retrieval audit

The `www.nist.gov` robots file returned HTTP 200. Neither the catalog path nor the rights-policy path was disallowed. The `nvlpubs.nist.gov` host returned HTTP 404 for `/robots.txt`; therefore no host-root robots policy was available there. Retrieval was limited to the already known exact PDF URL, with no path discovery or crawl. The DOI returned a single redirect to the same official PDF, whose final body exactly matched the direct-PDF checksum.

Exact allowed-body and robots-response checksums are recorded in `sources.jsonl` and `source_snapshot/provenance.json`. Robots bodies and source bodies are not redistributed.

## Complete manual review method

The 13-page PDF has a hidden OCR text layer over scanned page images. Every physical page was extracted for text screening, rendered, and manually inspected in two ordered contact sheets. Every substantive page was then read against the page-resolution rendering. `dispositions.jsonl` records one decision for each physical page.

Nine pages are partially retained and four are excluded. The excluded pages are one blank verso and three pages whose only substantive content is excluded visual matter. The component audit reconciles five figures and three tables individually, then records eight controlled exclusion blocks for branding and uncertain visual matter, numerical performance, formulas and fitted relations, procedures and equipment, standards and procurement rules, manufacturer-specific remarks, historical operational recommendations, and external or uncertain source bodies.

The catalog and rights HTML bodies were manually reviewed by document metadata, visible body, DOM structure, navigation, links, and footer. Fourteen material HTML surfaces are recorded in `surfaces.jsonl`. The catalog's notice that the page is no longer updated and may be out of date is retained as provenance, not as technical evidence.

## Retained historical evidence

- Exact T198 identity, authorship, issue history, and historical status.
- The bounded sample identity: commercial three-strand regular-lay Manila rope, chiefly submitted by manufacturers under government purchase orders around the study period.
- Yarn, strand, rope-lay, strand-lay, regular-lay, hard-laid, and soft-laid construction vocabulary.
- Distinctions among full-pool tensile observations, partial-pool construction and mass observations, and the small special hard-laid elongation subset.
- High-level distinctions among specimen termination, conditioning, machine assignment, tension, mass-per-length, circumference-based size, lay measurement, and constituent counting.
- The methodological warning that frequent termination-region failure can confound interpretation of rope-body performance.
- Manufacturer and purchase-specification variability, lack of an investigational sampling plan, uncontrolled variables, incomplete modern provenance, and incomplete uncertainty reporting.
- Negative evidence that the special hard-laid subset showed no well-defined proportional limit and that a simple fitted yarn-count relation did not capture every specimen closely.
- Absolute exclusion of present-day operational transfer.

## Exclusions

- Every raw breaking-load or strength value, fitted or calculated result, force, dimension, unit-bearing performance value, coefficient, equation, formula, plotted point, numerical table, and output.
- Every working-load rule, safety factor, tolerance, ranking, acceptance threshold, procurement rule, and period claim of safe use.
- All five figures, including two photographs and three plots, with captions, labels, image content, and image-derived claims.
- All three tables, including headings, notes, sample remarks, manufacturer identity, specification values, and layout.
- All specimen-preparation dimensions, soaking duration, fastening method, machine capacity, machine rate, preload rule, cutting or weighing recipe, and operational handling sequence.
- Every incorporated or cited standard, purchase specification, manufacturer direction, and third-party source body.
- Seals, marks, cover design, price, sales directions, commercial labels, and uncertain visual components.
- Any current claim about product selection, strength, service life, care, cleaning, hygiene, drying, storage, inspection, retirement, knots, frictions, working loads, body contact, uplines, anchors, bondage, or human suspension.

## Historical and safety boundary

The source concerns commercial three-strand regular-lay Manila rope obtained chiefly through government purchase-order testing around the early twentieth-century study period. It cannot transfer to modern products, 6 mm jute or hemp, rope care or hygiene, retirement, knots, working loads, body contact, uplines, anchors, bondage, or human suspension.

Nothing in this package approves, selects, sizes, rates, tests, certifies, installs, inspects, retires, or evaluates a rope, knot, upline, ceiling, hardpoint, anchor, bondage arrangement, body-support system, or human-suspension system. It cannot be used to generate an operational calculator or rule.

The three allowed source bodies are not redistributed.
