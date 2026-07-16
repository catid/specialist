# Manual audit report

## Outcome

This package is accepted as a quotation-light, bounded historical-review corpus with `direct_training_ready: true`. It uses only three targeted official NIST bodies: the publication record, the exact NISTIR 6096 PDF, and the NIST technical-series rights page. No crawl, mirror, archive, snippet, or linked source body was used.

The corpus retains only NIST employee-authored historical synthesis concerning failure-mode vocabulary, distinctions among research variables, evidence gaps, model limitations, internal citation leads, and the need for current site-specific qualified engineering verification.

## Verified source identity

- Report: NISTIR 6096, *Post-Installed Anchors—A Literature Review*
- Authors: Geraldine S. Cheok and Long T. Phan
- Publication: January 1998
- Publisher: National Institute of Standards and Technology
- Exact PDF pages: 109
- PDF bytes: 7053078
- PDF SHA-256: `8b6b586e65e61a634b29af302bfa686d4bb357a304f661363f9863028c6c88d7`
- Retrieved: `2026-07-16T14:53:14Z`

The official publication record gives January 1, 1998; the report title page says January 1998. Later file and hosting dates are retained only as provenance.

## Rights determination

The official NIST record and title page identify Cheok and Phan, and the title page places them in NIST's Building and Fire Research Laboratory. NIST's technical-series policy states that NIST employee works are not subject to United States copyright and describes a broad worldwide grant for any residual foreign rights. The same policy warns that NIST publications can contain protected third-party work.

Accordingly, the package uses only eligible NIST employee-authored synthesis. All figures, tables, standards, quotations, third-party study bodies, references as evidence, and uncertain components are excluded. The rights page is used only for rights provenance.

Exact allowed-body checksums and the rights basis are recorded in `sources.jsonl` and `source_snapshot/provenance.json`.

## Manual review method

The 109-page PDF is a scan without a text layer. Every physical page was rendered and OCR-screened. All 109 page images were manually inspected in ten ordered contact sheets, and relevant pages were read at page resolution against the scan. `dispositions.jsonl` records one manual decision for every physical page.

Twenty-six pages are partially retained and eighty-three are excluded. The scan contains six explicit blank pages. The component audit reconciles thirteen figures and four tables individually, then records ten controlled exclusion blocks for visual branding, products and installation descriptions, incorporated standards, individual third-party study summaries, equations and calculations, numerical performance matter, procedures and recommendations, unavailable or ongoing research, references, and calculation appendices.

The NIST publication record and rights page were divided into fourteen material surfaces and manually dispositioned in `surfaces.jsonl`.

## Retained historical evidence

- Failure-mode vocabulary: steel, concrete cone, edge breakout or bursting, splitting, pull-out or slip, bond-interface failure, and combined modes.
- The distinction between failure in the anchor component, surrounding concrete, and one of two bond interfaces.
- Research-variable distinctions: action regime, static or cyclic history, cracked or uncracked substrate, concrete strength, member and embedment geometry, edge and spacing context, single or multiple anchorage, eccentricity, preload, installation condition, and nearby reinforcement.
- Historical evidence gaps for combined action, especially adhesive anchors, reversed-cyclic behavior, and cracked concrete.
- Condition-specific model limits, inconsistent historical predictions, cross-study scatter, omitted proprietary work, and incomplete reporting.
- Internal leads to the report's vocabulary section, discussion sections, summary, conclusions, and bibliography.

The corpus preserves that these are findings and gaps of a 1998 secondary review. It does not claim that they describe the current state of evidence.

## Exclusions

- Every capacity, numerical performance result, force value, threshold, unit, equation, coefficient, curve, and sample calculation.
- Every product, brand, specific anchor configuration, commercial identity, and manufacturer direction.
- Every installation, inspection, testing, proof-load, design, reinforcement, or acceptance recipe.
- Every incorporated standard, code, design guide, third-party quotation, and externally authored study body.
- All thirteen figures, all four tables, captions, plots, diagrams, visual branding, seals, and image-derived claims.
- The complete bibliography as substantive evidence; it remains only an internal navigation lead requiring separate source and rights audits.
- Both numerical calculation appendices in full.
- Ongoing, proprietary, unavailable, and future experimental details.
- Every operational selection, sizing, or certification inference.

## Current-use and safety boundary

NISTIR 6096 is historical secondary-review evidence, not current qualification evidence. Any real installation requires current governing requirements, site information, condition assessment, and current site-specific verification by appropriately qualified engineering professionals.

Nothing in this package approves, selects, sizes, proof-tests, certifies, or evaluates a ceiling, hardpoint, anchor, bondage setup, body-support system, or human-suspension application.

The three allowed bodies are not redistributed.
