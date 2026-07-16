# Curation, evidence, and rights report

## Outcome

Accepted as a bounded statistical-method corpus. The retained artifact is a manual, claim-cited explanation of knot efficiency as the ratio of two heterogeneous breaking-strength random variables. It includes model assumptions, normality evidence, probability summaries, approximation boundaries, and failure modes of common estimators.

The corpus excludes every source equation as an operational calculator, all numerical efficiencies and example ranges, ratings, safety factors, safe-load recipes, knot choices, source figures and tables, vendor claims, third-party text, and cross-domain validation.

## Retrieval and complete review

The DOI resolved to the official MDPI article landing page. Direct command-line access to that HTML landing page returned HTTP 403 during this review, so it was not used as a content source. The official MDPI asset host supplied both the publisher PDF and JATS XML with matching article identity and an embedded CC BY 4.0 notice.

The PDF contains 25 pages. Every page was reviewed:

- PDF pages 1–3: front matter, license, abstract, introduction, ratio definition, prior-data normality statement, two misleading calculation strategies, and objectives.
- PDF pages 4–5: general ratio-density geometry and derivation.
- PDF pages 6–8: normal-input assumptions, physical parameter sector, and exact ratio PDF.
- PDF pages 9–13: PDF/CDF properties, scaled parameterization, nonphysical tails, global versus truncated moments, mode, median, and central equal-tail ranges.
- PDF pages 14–16: solid approximation, closed-form CDF, approximation median, central ranges, and truncated summaries.
- PDF pages 17–19: approximation error bounds and comparison with a normal approximation.
- PDF pages 20–24: inherited worn-rope and new-rope examples, qualitative comparison of skew/dispersion, and conclusions.
- PDF page 25: references.

The official XML section structure and table alignment were used as navigation aids. Text extraction was used only for audit and search. Corpus prose was manually written after reviewing all sections and pages.

## Section-level scientific disposition

Included:

- Article identity, authors, dates, DOI, and embedded license.
- Directly measured knotted and straight breaking strengths versus calculated efficiency.
- The independent-draw/destructive-test assumption.
- Roughly 200 straight and 80 knotted observations reported from earlier work as provenance for normality checks.
- Quantile-plot, Kolmogorov–Smirnov, and Shapiro–Wilk checks, with failure-to-reject distinguished from proof.
- Why one knotted/one straight break and a ratio of small-sample means are misleading.
- General ratio-density construction and the normal-input special case in plain language.
- PDF, CDF, central tendency, input and truncated dispersion, and equal-tail population ranges.
- Nonphysical normal tails and the resulting absence of finite global ratio mean and variance.
- Solid-approximation assumptions, conveniences, residual numerical integration, and explicit error management.
- Case-dependent normal approximation illustrated qualitatively by inherited worn/new examples.
- Finite-sample parameter uncertainty as distinct from specimen-population variability.

Excluded:

- Equations 1–71, proofs, theorem statements as calculators, and parameter recipes.
- All numerical efficiencies, regression or comparison values, example force values, probability ranges, and portable interval endpoints.
- Tables 1–4 and Figures 1–9.
- Broad claims that every knot, rope, or publication behaves one way.
- Knot selection, geometry recommendations, ratings, safety factors, safe-load formulas, and field guidance.
- Vendor identity, material-support statements, trademarks, publisher marks, and product-specific examples.
- Third-party quotations, standards text, cited papers/books, and references as unreviewed sources.
- Dynamic, cyclic, wet, aged, damaged, natural-fiber, bondage, body-contact, upline, and human-suspension transfer.

Detailed decisions appear in `dispositions.jsonl` with section and page locations.

## Statistical interpretation audit

The article derives a population distribution after specifying input distributions and their parameters. This is not the same as deriving a sampling distribution for estimated means and variances. In practice, fitted input parameters carry finite-sample uncertainty in addition to specimen-to-specimen variability. The digest makes that distinction explicit so a smooth fitted ratio PDF is not mistaken for complete inferential certainty.

The article reports that normality checks on earlier data did not find significant departure. That result supports using normal inputs provisionally but does not prove normality. Goodness-of-fit tests can fail to detect departures, especially in tails, and their outcome depends on sample size and test power.

The paper uses “tolerance interval” for equal-tail central quantile ranges of the modeled population. These are not confidence intervals for fitted parameters and do not include the confidence-on-coverage dimension of a classical estimated statistical tolerance interval. They are also not engineering tolerances or safety bands. The digest retains the paper’s term while stating this scope.

## Rights and component provenance

The publisher PDF and XML both state © 2022 by the authors and distribute the article under CC BY 4.0. The Creative Commons deed permits sharing and adaptation with appropriate credit, a license link, and an indication of changes, without additional restrictions. The corpus supplies title, authors, journal, year, DOI, license link, change notice, and no-endorsement notice.

Conservative component exclusions remain in force:

- Figures and tables may include data or visual components associated with prior work.
- References, quoted propositions, and standards-related claims have separate provenance.
- The acknowledgments identify commercial material and technical support.
- Publisher layout, marks, and third-party trademarks are not needed for the adaptation.

Only manual paraphrase and bibliographic facts are retained. This is an operational rights screen, not legal advice.

## Extraction and structure audit

The 25-page PDF is digitally generated. Deterministic extraction yielded 65,137 characters and 11,586 alphanumeric tokens, with extraction-stream SHA-256 `899f8052729f816a1c75cef9f7c275d9d0c041caea97d0aff107b09764a3d29a`. The stream is not stored.

The JATS XML identifies nine numbered figures and four numbered tables. Two raster image XObjects were found in the PDF page resources; other figures are vector or native PDF content. All visual, tabular, formula, and source-layout content is excluded regardless of encoding.

## Split hygiene and verification

This package is training-source-only. It must not seed validation, holdout, evaluation, OOD, or probe material. Any future training Q&A must preserve model assumptions, provenance, and the difference between population spread and parameter-estimation uncertainty.

Deterministic tests cover identity, page and section review, source hashes, license attribution, model assumptions, normality caution, estimator pitfalls, probability summaries, approximation limits, forbidden numerical outputs, component exclusions, split hygiene, README, and manifest integrity.
