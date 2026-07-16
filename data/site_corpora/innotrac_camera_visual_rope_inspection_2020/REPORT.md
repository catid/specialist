# Manual audit report

## Outcome

This package is accepted as a quotation-light, bounded industrial inspection-method corpus with `direct_training_ready: true`. It uses three exact bodies: the official innoTRAC article record, the exact final article PDF, and the CC BY 4.0 legal code. No crawl, mirror, archive, search snippet, citation export, linked reference, project source, standard, product page, or external study body was used as evidence.

The corpus retains only manually rewritten evidence concerning one-sided manual surface viewing, inspector fatigue and perceptual habituation, absent continuous manual records, multi-angle surface acquisition, combining outward indicators, exploratory synthetic-rope project scope, funding and commercial-platform disclosure, validation gaps, and the absolute distinction between surface appearance and internal condition.

## Verified source identity

- Article: *Camera-based Visual Rope Inspection*
- Author: Gregor Novak
- Affiliation: Institute for Mechanical Handling and Logistics, University of Stuttgart, Germany
- Journal: innoTRAC Journal, volume 1
- Printed pages: 55–63
- DOI: `10.14464/innotrac.v1i0.462`
- Official record publication date: December 3, 2020
- PDF online-availability date: December 7, 2020
- Exact PDF pages: 9
- PDF bytes: 670058
- PDF SHA-256: `f7b64101a3d8df7b76c8a8ecb06e631a1a892985e104bd10248f19d923559af2`
- Retrieved: `2026-07-16T16:05:23Z`

The official record supplies title, author, affiliation, DOI, volume, page range, and its publication date. The final PDF confirms the same identity, adds receipt and acceptance history, and gives a later online-availability date. This date difference is recorded rather than silently collapsed.

## Rights and component determination

The official article record states that all articles are published under Creative Commons Attribution 4.0. More importantly, the first page of the exact final PDF identifies © 2020 G. Novak and states that this article is distributed under CC BY 4.0. The legal code permits adaptation and redistribution subject to attribution and change notice, prohibits implied endorsement, and does not license patent or trademark rights.

No figure-specific credit or contrary rights notice appears in the PDF. Nevertheless, all nine figures, the journal logo, photographs, plots, measurement overlays, product depictions, trademarks, patented-system names, fibre-brand names, and uncertain visual components are excluded. This avoids relying on the general article license for components that are unnecessary to the retained evidence.

Attribution is provided in `README.md`, the source manifest, and provenance. This is a manually rewritten, narrowed, component-filtered adaptation made on 2026-07-16. No endorsement is implied.

## Robots and retrieval audit

The TU Chemnitz library host's robots file returned HTTP 200 and allows all paths for the general user agent. The Creative Commons robots file also returned HTTP 200 and contained no general-agent disallow rule. Retrieval used only the known article, PDF, and license URLs. The article's view-PDF route redirected once to its official download route; the DOI redirected once to the official article record, whose final body matched the directly retrieved record.

Exact source-body and robots-response checksums are recorded in `sources.jsonl` and `source_snapshot/provenance.json`. None of those bodies is redistributed.

## Complete manual review method

The nine-page tagged PDF has a native text layer. Every physical page was extracted for text screening, rendered, and manually inspected in two ordered contact sheets. Every page containing visual or retained material was then checked at page resolution. `dispositions.jsonl` records one manual decision for each physical page; all nine are partially retained because each contains at least one eligible contextual or limitation claim.

The component audit reconciles all nine numbered figures individually and confirms that the article has no tables. It then records nine controlled exclusion blocks covering the journal logo and visual branding; numerical results and thresholds; the explicit logical discard rule; camera, illumination, machine, specimen, and acquisition recipes; software, processing, and algorithms; products, patents, and trademarks; standards and external methods; discard, rating, lifetime, and internal-condition claims; and all twelve reference bodies.

The article-record and CC legal-code HTML bodies were reviewed by metadata, visible body, DOM structure, navigation, links, rights statements, and footer. Fifteen material HTML surfaces are recorded in `surfaces.jsonl`.

## Retained industrial method evidence

- A manual viewer normally sees only the facing surface orientation.
- Repetitive long-pass viewing can create fatigue or perceptual habituation, but the article reports no controlled miss-rate study.
- A purely manual pass may lack a continuous record and leave later reconstruction dependent on memory and testimony.
- Multiple camera viewpoints can reduce directional surface blind spots and create a persistent surface record, but the project used one or more cameras and does not prove complete circumferential coverage.
- Apparent geometry, lay-related geometry, colour intensity, visible silhouette area, and protruding fringes are distinct candidate surface indicators.
- One supposed diameter example used an ellipse-axis surrogate rather than a direct diameter, exposing a measurement-construct limitation.
- Combining indicators can reduce reliance on one outward signal, but the combined model still needs independent calibration, validation, and error analysis.
- Project specimens were high-modulus synthetic ropes in braided and laid constructions, with and without covers, under industrial running-rope and staged bending-test contexts.
- The body discloses one publicly funded project, a student project nested in a state-financed project, use of a patented commercial platform, standard commercial software, and separately programmed processing.
- The final article has no separate conflict-of-interest, formal funding, data-availability, or code-availability declaration.
- The conclusion says high-modulus fibre-rope discard criteria were not sufficiently researched.

## Exclusions

- All nine figures, captions, plots, photographs, rope images, overlays, axes, labels, logos, legends, values, and image-derived claims.
- The explicit logical discard rule and every threshold, percentage, turning point, numerical output, region, equation-like expression, and proposed discard point.
- Camera count and placement recipes, illumination, machine configuration, loads, motion, acquisition intervals, specimen staging, break-test sequence, sensor layout, calibration, and equipment directions.
- Commercial software, custom processing, algorithms, code, segmentation steps, data transformations, and spreadsheet workflow.
- Product and system names, patented-platform claims, trademarked fibre brands, logos, and any endorsement or comparative claim.
- All standards, guidelines, external inspection modalities, product literature, cited findings, and twelve reference bodies.
- Claims of inspection-system accuracy, product superiority, operational speed, downtime reduction, safe continuation, emergency shutdown, remaining life, internal condition, strength, working load, discard readiness, rating, acceptance, or certification.
- Claims about finished natural-fibre, jute, or hemp shibari rope; visual bondage-rope inspection; care, hygiene, retirement; knots; body contact; uplines; anchors; bondage; or human suspension.

## Disclosure and validation surprise

The most important surprise is the gap between the paper's system-oriented language and its reported validation. The article provides no complete sample inventory, held-out detection study, sensitivity or specificity, false-positive or false-negative analysis, blinded review, independent replication, or external-site validation. It also moves between a several-camera concept and experiments described as using one or more cameras, so full multi-angle coverage cannot be assumed for every result.

The article's own conclusion that discard criteria were insufficiently researched is therefore controlling. The source is useful for inspection architecture and observability lessons, not for a discard rule.

## Absolute application boundary

Surface imaging does not establish internal condition. Combining multiple surface indicators does not change that observability limit unless the resulting model is independently validated against relevant internal states and failure outcomes.

Nothing in this package approves, selects, rates, calibrates, installs, validates, certifies, or evaluates a camera system, rope, knot, upline, ceiling, hardpoint, anchor, bondage arrangement, body-support system, or human-suspension system. No threshold, discard rule, remaining-life estimator, internal-condition classifier, or operational calculator can be generated from it.

The three allowed source bodies are not redistributed.
