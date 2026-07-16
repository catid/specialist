# Curation, evidence, identity, and rights report

## Outcome

Accepted as a bounded historical statistical-methods corpus. The retained artifact is a manual, claim-cited digest of RP1847, emphasizing population provenance, measurement protocol, a static breaking endpoint, log-scale modeling assumptions, nonrandom sampling, heterogeneous production, anomalies, and the limits of formal precision.

Excluded are all source equations, Tables 1–4, Figures 1–5, numeric strength and rating outputs, numerical tolerance bands, procurement and specification rules, contemporary product claims, page images, scan text, cited third-party material, and transfer to modern natural-fiber bondage rope, knots, wet or aged behavior, care, body contact, uplines, anchors, or human suspension.

## Source identity and DOI correction

The official PDF identifies:

- Sanford B. Newman and J. H. Curtiss;
- “A Statistical Analysis of Some Mechanical Properties of Manila Rope”;
- U.S. Department of Commerce, National Bureau of Standards;
- Research Paper RP1847;
- volume 39, December 1947;
- *Journal of Research of the National Bureau of Standards*, pages 551–559; and
- Washington, May 26, 1947 on the final page.

The task-supplied DOI `10.6028/jres.039.037` is not this paper. A no-follow resolver request returned HTTP 302 to NIST’s `jresv39n1p53_A1b.pdf`; DOI registry metadata names an unrelated uranium article by Andrew I. Dahl and Milton S. Van Dusen on page 53.

Title search and registry metadata identify the manila-rope paper as DOI `10.6028/jres.039.039`. A no-follow request to that DOI returned HTTP 302 to the exact official PDF requested for this corpus, and its registry metadata matches Newman, Curtiss, title, journal, volume, issue, December 1947, and opening page 551. The corpus uses `.039` and records `.037` only as a rejected identity candidate.

## Retrieval and document audit

The official NIST PDF was retrieved successfully on 2026-07-16. It is 9,591,323 bytes with SHA-256 `5429d55bb31652b297703ba861f9b64aa43f3e6d3de6ae811ebc0e847fe67741`. The server reported a 2012 last-modified date, which describes the repository scan file rather than the 1947 publication.

The PDF has nine pages measuring 537 by 732.48 points. Each contains one high-resolution, 8-bit grayscale JPEG page scan at 600 pixels per inch; several pages also contain trivial inline stencil objects. The PDF has a searchable OCR layer. Deterministic layout extraction produced 72,737 bytes and 6,304 whitespace-delimited words with SHA-256 `03842ed99c5d6f7ceafc1600af90b11960e8076108ac0efa03adf1e6f4e1b56a`.

The OCR layer was used for navigation and search, not accepted as an authoritative transcription. All nine pages were reviewed visually in an ordered contact sheet, and prose-heavy identity, methods, results, anomaly, and reference pages were inspected at full rendered resolution. No source PDF, extracted text, render, or contact sheet is retained.

## Complete page review

- PDF page 1 / journal page 551: identity, abstract, introduction, 1938–1941 government-submission provenance, three-strand scope, inadequate four-strand evidence, source heterogeneity, variability causes, and opening conditioning method. Included selectively; numerical sizes and cited specification rules excluded.
- PDF page 2 / journal page 552: circumference/weight procedure, Table 1, breaking-strength procedure, 863-observation count, nonrandom and heterogeneous universe, anomalies warning, and opening model. Methods and sampling cautions included; settings, Table 1, and Equation 1 excluded.
- PDF page 3 / journal page 553: log transformation, unweighted least squares, back-transform correction, log-normality and constant-variance assumptions, punched-card implementation, and rationale. Included in plain language; Equations 2–3 and constants excluded.
- PDF page 4 / journal page 554: curve-family rationale, flexibility check, results interpretation, formal coefficient uncertainty, and Table 2. Model-selection reasoning and caveat included; all equations, coefficients, strength values, and Table 2 excluded.
- PDF page 5 / journal page 555: consequences of nonrandom sampling, curve misspecification, individual dispersion, tolerance construction, rounding, and distinction between individual measurements and means. Included without formulas, coverage numbers, or output values.
- PDF page 6 / journal page 556: Figures 1–4 showing fitted means and tolerance curves. Entire page excluded as source visual/numerical components.
- PDF page 7 / journal page 557: Table 3 of circumference, weight, and strength estimates and ranges. Entire page excluded as tabular numerical output.
- PDF page 8 / journal page 558: fit check, three data irregularities, Table 4, and Figure 5. Anomalies included in qualitative paraphrase; percentages, subgroup values, table, plot, procurement inference as a rule, and strength comparison output excluded.
- PDF page 9 / journal page 559: end of historical strength comparison, references, and dateline. Dateline retained for identity; comparison result and third-party references excluded.

One page-level decision for each page appears in `dispositions.jsonl`.

## Dataset and measurement audit

The abstract’s “more than 800 samples” and the methods section’s exact 863 observations are consistent descriptions at different precision. The sample was not selected randomly by the analysts. It arrived through government acceptance testing and represents an unknown mixture of suppliers, works, contractors, sizes, production periods, and previous treatments.

Only three-strand rope is analyzed. The paper explicitly rejects statistical treatment of its limited four-strand records. The corpus prevents construction transfer and treats fiber quality, fabrication, and previous treatment as unmodeled sources of variation rather than causal coefficients.

Circumference and weight were measured on conditioned, unspliced material under a prescribed preload. Breaking strength used separate eye-spliced specimens, controlled conditioning, temporarily wetted splices, pins, horizontal machines, a slow moving head, and failure at the maximum static load in one or more strands. The corpus preserves these dependencies while omitting operational settings.

Two machine types were selected according to anticipated strength. The paper gives the split and procedures but no cross-machine equivalence analysis. This potential apparatus discontinuity remains a stated limitation rather than an asserted bias.

## Statistical interpretation audit

The paper models four conditional relationships. It explicitly warns that a conditional mean at fixed weight is not obtained by merely eliminating nominal diameter between two other mean curves. The digest preserves this distinction without equations.

Log transformation was chosen because raw spread grew with the mean. The resulting method assumes a linear conditional mean and constant variance in log space plus log-normality for the arithmetic-mean back transformation and individual range construction. The paper reports exploratory support for variance stabilization, not a modern residual or out-of-sample validation program.

Large sample size made formal coefficient errors small under the model. The authors nevertheless reject their usual population interpretation because sampling was not rigorously random, the universe was heterogeneous, and the simple curve might miss the true mean relation. The corpus foregrounds that separation of computational precision from population validity.

The paper’s “tolerance limits” describe individual modeled observations around a curve of means. They are not confidence intervals for a mean, current manufacturing tolerances, strength ratings, procurement rules, or safe-use limits.

## Anomaly audit

All three reported irregularities are preserved:

- some nominal-size labels behaved like particular larger groups, suggesting historical substitution or misclassification;
- observations at fixed nominal diameter appeared in tighter clusters than the pooled dispersion, consistent with nonrandom grouping; and
- one very small size subgroup was poorly represented by the global fit.

The first two effects inflate pooled spread. The third led the authors to prefer global extrapolation over a few anomalous observations for that historical table. The digest does not turn that judgment into a general extrapolation rule.

## Rights and component audit

The NIST Research Library FAQ states that federal NIST publications are generally public domain and not subject to copyright in the United States, and requests original-source credit. NIST’s Technical Series policy states that employee-authored works are not protected in the United States, reserves possible foreign rights, and grants a broad worldwide reprint and derivative permission to the extent NIST may assert those rights.

RP1847 identifies both authors with the National Bureau of Standards federal publication and contains no contrary copyright notice. The corpus includes the NIST-requested credit, a change notice, and no-endorsement statement.

Images and source-layout components remain excluded because their component provenance is unnecessary and potentially ambiguous. Tables and graphs contain numerical property and procurement outputs; references are separate third-party works; equations could become detached calculators. This is a conservative operational rights screen, not legal advice.

## Split hygiene and verification

This package is training-source-only. It must not seed validation, holdout, evaluation, OOD, or probe data. Future training Q&A must preserve the date, construction, population, conditioning, endpoint, sampling caveats, and DOI correction and must not become current rope-use advice.

Deterministic tests cover identity, DOI binding and rejection, rights, page review, source hashes, extraction audit, population provenance, construction scope, measurements, statistical assumptions, anomaly retention, forbidden numerical and operational outputs, source-component exclusion, split hygiene, and manifest integrity.
