# Manual source, rights, component, and leakage audit

## Outcome

This package is accepted as a direct-training-ready historical evidence-and-limitations digest. The retained corpus is intentionally small because the source’s useful lesson is a boundary: Becker and Appel studied evaluation of prepared abaca-fiber color, explicitly not the relationship between color and rope serviceability.

The official PDF was read completely and inspected as rendered pages. All figures, tables, experimental values, scales, instruments, solvents, procedures, specifications, ranks, recommendations, product names, and third-party/reference bodies were excluded. The PDF and official HTML source bodies are not redistributed.

## Official identity audit

The current official NIST Volume 11 page lists the paper in Issue 6, December 1933, at page 811, names Becker and Appel, and links DOI `10.6028/jres.011.057`. The DOI resolves to the same official NIST PDF route and reports the same content length and ETag as the checksum-bound artifact.

The exact PDF independently identifies:

- U.S. Department of Commerce, Bureau of Standards;
- Research Paper RP627;
- *Bureau of Standards Journal of Research*, volume 11, December 1933;
- title *Evaluation of Manila-Rope Fiber for Color*;
- authors Genevieve Becker and William D. Appel;
- printed pages 811–822; and
- closing dateline Washington, September 16, 1933.

The first affiliation note identifies Becker as research associate at the Bureau of Standards for the Cordage Institute. The second identifies Appel as chief of the Bureau’s Textile Section. The introduction says the Cordage Institute established the research-associate position and that cordage manufacturers and the Boston Navy Yard furnished rope.

## Access and exact artifact

The official NIST main-host robots file returned HTTP 200. None of the three reviewed official page paths was disallowed. The NIST publication host returned HTTP 404 at `/robots.txt`; this was recorded rather than interpreted as affirmative permission. The requested PDF was one exact, known official publication URL linked from NIST’s volume and DOI records; no crawl or search endpoint was used.

- URL: `https://nvlpubs.nist.gov/nistpubs/jres/11/jresv11n6p811_a2b.pdf`
- Retrieved: `2026-07-16T16:41:55Z`
- HTTP status: 200
- HTTP Content-Type: `application/pdf`
- Last-Modified: `Tue, 31 Jul 2012 21:21:34 GMT`
- ETag: `"c207176626fcd1:0"`
- Bytes: 1,082,392
- SHA-256: `fc29fe3c20edda1d8372c05083e9154d328b225c6959a9f1b8f4ab6edc5debd1`
- SHA-512: `bec7afc264e1659011a2c9a483be4842d9ba7f6c3805c74204d909c03085a547a124636f191ab0b57a6dbd445e143cf5a06c4ac78eb79cbb0460fa4dd43e2cbd`

The PDF metadata reports 12 pages, PDF version 1.6, no encryption, no JavaScript, no forms, and an untagged document. It identifies the title and authors and says the artifact was digitized by the Internet Archive and recoded by LuraDocument. Those production credits are provenance only; no Internet Archive body was used.

## Rights determination

NIST’s official “About the Journal” page states that journal papers are not subject to copyright in the United States, remain open access, and should receive attribution. NIST’s current technical-series rights page says works authored by NIST employees are not subject to U.S. copyright, foreign rights are reserved, and broad worldwide reprint and derivative rights are granted to the extent NIST may assert them. It also cautions that some works or components written by third parties may be protected.

This package therefore records the source as not subject to copyright in the United States under the journal statement, makes no unqualified worldwide public-domain claim, attributes the paper and NIST, and excludes every figure, table, cited work, product, and separable reference component. It is a manually rewritten and narrowed derivative; no endorsement is implied.

## Complete paper and component review

All 12 physical pages were extracted for text screening, rendered, and manually inspected in two ordered contact sheets. The paper is a scan with an OCR text layer. Each physical page contains two RGB scan-image objects and one mask, for 24 RGB images and 12 masks. All 36 image or mask objects were excluded.

The printed-page mapping is exact: physical pages 1–12 correspond to printed pages 811–822. `dispositions.jsonl` records one manual decision for every page. Four numbered figures and five numbered tables were found and individually reconciled:

- Figure 1 on printed page 814;
- Table 1 on printed page 815;
- Figure 2 on printed page 816;
- Figure 3 on printed page 817;
- Table 2 and Figure 4 on printed page 819;
- Table 3 on printed page 820; and
- Tables 4 and 5 on printed page 821.

Every one is excluded, including captions, axes, labels, plotted order, categories, values, and claims that require reading the visual component.

The article has 21 numbered footnotes. The first two affiliation notes are retained only for author context. Every external publication, administrative order, specification, unpublished report, working drawing, equipment direction, commercial source, and cross-reference contained in the remaining footnotes is excluded. No referenced body was acquired.

## Retained evidence and limitations

The paper distinguishes constituent fiber color from finished-rope surface appearance. It explains that a surface observation is affected by construction, yarn size, strand arrangement, twist, dirt, and dust. Its own analytical target was prepared fiber rather than the intact exterior.

The study also separates prelubrication fiber color from finished-rope appearance. Lubricant can contribute its own color, accumulate dust, and bleach or darken through light and air exposure. The source reports that lubricant removal did not always restore the original response. Abaca could also be bleached. These are confounders, not a procedure for revealing an intrinsic safety signal.

Most importantly, the authors explicitly say they did not study the relationship between fiber color and rope serviceability. Their brief mention of unspecified Navy Department experiments supplies no design, sample description, endpoint, table, or auditable citation. It weakens a categorical historical belief that dark fiber necessarily meant worse rope, but it proves neither that dark rope is safe nor that color has no relationship under every condition.

The sample context remains bounded. Manufacturers and the Boston Navy Yard furnished rope for the color work, and a commercial multi-manufacturer set was examined. The paper does not establish service exposure, present-day representativeness, Navy qualification, or any connection between those specimens and the separate Navy experiments mentioned in passing.

## Novelty check

Candidate phrases were searched only in existing `manifest.json`, `REPORT.md`, and `README.md` files under `data/site_corpora`. Existing `CORPUS.md` bodies were never searched or opened. At audit time, `manila-rope fiber`, `serviceability of rope`, `finished-rope surface`, `dark rope`, `lubricant affects the color`, `Boston Navy Yard`, and `Cordage Institute` each returned zero matches in those allowed metadata/report surfaces.

No QA, trainer, evaluation, holdout, OOD, probe, or experiment artifact was inspected.

## Controlled exclusions

- every scan image, figure, table, caption-derived visual claim, axis, label, color scale, experimental value, formula, and visual rank;
- all instruments, apparatus, products, standards, white references, filters, illuminants, solvents, oils, cutting tools, containers, and preparation materials;
- every specimen-preparation, extraction, drying, viewing, measurement, sampling, procurement, laboratory, and equipment recipe;
- every government specification, grading requirement, threshold, named value, rope grade, ordering, acceptance rule, and cost comparison;
- every external paper, book, bulletin, administrative order, unpublished report, working drawing, and other third-party/reference body;
- all strength, deterioration, decay, breaking, serviceability, and safety implications not directly established by this study;
- every modern recommendation, including inspection, purchase, maintenance, cleaning, hygiene, drying, storage, retirement, or working-load guidance; and
- all transfer to jute, true hemp, body contact, knots, frictions, uplines, anchors, hardpoints, bondage, restraint, body lowering, or human suspension.

## Absolute application boundary

No visible darkness, lightness, color change, stain, surface condition, or laboratory fiber-color result from this paper is a discard or acceptance criterion. Nothing here establishes material identity, internal condition, contamination, hygiene, strength, remaining life, serviceability, working load, or body-support suitability.

The source bodies are not redistributed.
