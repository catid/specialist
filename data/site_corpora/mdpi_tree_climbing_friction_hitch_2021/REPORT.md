# MDPI tree-climbing friction-hitch study corpus report

## Outcome

This package adds one 2,913-whitespace-word, non-Q&A Markdown digest of the 2021 *Forests* article *Tensile Strength of Ropes and Friction Hitch Used in Tree Climbing Work*. The digest was manually written from the complete official article and contains 43 claim-level citations.

The study is represented as a narrow preliminary bench experiment: 36 slow pulls, two synthetic arborist work ropes, two friction hitches, four treatments, and nine runs per treatment. Its central comparative result is that hitch type was associated with load and slip while rope type was associated with diameter deformation.

Limitations appear before detailed methods or results. The document repeatedly rejects dynamic, cyclic, fall, natural-fiber, bondage, human-suspension, safe-rating, best-knot, certification, and field-suitability interpretations.

The 56 disposition records contain 36 included or narrowly included scientific decisions and 20 metadata-only or excluded decisions.

## Source identity and integrity

The primary hashed source is the official MDPI article PDF asset. It returned HTTP 200 as a 17,049,016-byte PDF with SHA-256 `55832e9cb3ed23942315922fad01a5c79fd239e0fcf50b690fddb58bd519910d`. The file has 12 digital pages. Its server `Last-Modified` value is 26 October 2021 07:06:26 GMT.

A deterministic pypdf 6.14.2 audit extraction produced 39,105 bytes and 6,149 words with SHA-256 `78d26d2c0e2b155f421dfc46534143aac1cbbb49929c82c8483d77a09861e01d`. This extraction was used for complete manual review and was not retained or emitted as training prose.

The DOI resolver returned HTTP 302 to the canonical MDPI article URL. Its 167-byte resolution body is hashed in the source ledger. The canonical HTML was available through a read-only browser renderer as a complete 493-line article and was crosschecked against the official PDF. A shell request to the same canonical URL received a 403 access-denial body; the denial digest is recorded explicitly and is not misrepresented as article content.

Two Creative Commons routes—the CC BY 4.0 deed and its linked English legal code—were retrieved and hashed only for licence metadata. The hashed successful bodies total 17,130,331 bytes: 17,049,016 bytes of article PDF, 167 bytes of DOI resolution, and 81,148 bytes of licence metadata.

`sources.jsonl` records exact URLs, access timestamps, status, content types, sizes, hashes, PDF metadata, audit-text metrics, and treatment for all five source-ledger records.

## Access boundary

Only official article, official article-asset, DOI, and Creative Commons licence routes were used. No reference link, standards provider, manufacturer, product page, image route, or other third-party article was accessed.

Several official article variants returned 403, 404, or 429. They contributed no content. Their route, status, and zero-use disposition are recorded in provenance so an unsuccessful request cannot be confused with an unreviewed source.

Raw HTML, the PDF, DOI response, licence pages, and audit extraction are absent from the repository. The only direct-training artifact is the manually transformed `CORPUS.md`.

## Rights

The article states © 2021 by the authors and distribution under Creative Commons Attribution 4.0 International. The adaptation supplies:

- the complete article title;
- all six article authors;
- journal and year;
- linked DOI and canonical article;
- linked CC BY 4.0 licence;
- a description of manual selection, paraphrase, condensation, reorganization, and added boundaries;
- a no-endorsement statement;
- CC BY 4.0 metadata for the adapted Markdown.

The licence deed and legal code are rights metadata only. No legal-code text is direct training material, and this package gives no legal opinion.

## Component and trademark treatment

The PDF contains ten raster-image occurrences, including article graphics and publisher assets. No image, photograph, diagram, chart, logo, or icon is reused. No source table layout is copied. Essential study factors and numerical results were manually rewritten from article text and tables.

The rope-model labels `Axis` and `XTC 16` are retained only because the task requires the exact tested systems. The corpus immediately identifies them as experimental labels and disclaims endorsement. Manufacturer identities, catalog breaking loads, machine and camera brands, connector and plate labels, catalog masses, and promotional claims are excluded.

Standard numbers and standards bodies are excluded. The scientific fact that the authors modified a standardized test method is retained because it materially limits interpretation. The package makes no compliance claim.

The article's reference list and all third-party findings are excluded unless needed only to report that the authors themselves identified an untested domain. No reference was separately fetched or rights-reviewed.

## Exact experimental design

The digest preserves the complete factorial structure:

- System `A`: the Axis model, listed as an 11 mm polyester/polyamide synthetic rope;
- System `X`: the XTC 16 model, listed as a 13 mm polyamide synthetic rope;
- hitch cord: 8 mm polyester/Technora;
- hitch `P`: Prusik;
- hitch `T`: Valdotain tresse;
- treatments: `AP`, `AT`, `XP`, and `XT`;
- nine runs per treatment;
- 36 traction tests total.

The article's results narrative later calls the larger rope 12.7 mm while its specimen table lists 13 mm. The digest reports both and labels the discrepancy rather than silently choosing one.

The source describes three rope pieces and three cords per rope/hitch type, with three traction tests on each described set of sections and cords. This detail remains explicit so nine runs are not casually described as nine independently sourced fresh complete systems.

## Bench method without a construction recipe

The retained method is sufficient to define evidence but not to recreate a life-support system:

- slow pull on semi-static rope;
- traction bench;
- 250 mm total stroke;
- one initial and five later observation positions at 50 mm spacing;
- applied system load, hitch slip, and rope-diameter change measured at those positions;
- near-hitch temperature measured at the beginning and end;
- ambient temperature and humidity recorded.

Terminal-knot construction, plate and connector arrangement, preloading, resetting, attachment dimensions, and equipment brands are excluded as unsafe operational detail.

The system-level nature of the measurement is emphasized. The assembly contained rope, hitch cord, hitch, terminal and connecting components, and machine geometry. Results do not isolate the intrinsic strength of a single component.

## Statistical design

The article's distributional and variance checks, one-factor temperature analysis, factorial multivariate analysis, post-hoc family-wise confidence level, and regression plan are summarized.

The main model used rope and hitch as factors with load, slip, and diameter change as responses. Separate regressions examined pairwise relationships; a multiple model described load using slip and diameter reduction together.

The digest warns that repeated displacement observations and the limited physical specimen structure remain part of the data-generating process. Recorded points are not treated as additional equipment types or field contexts.

## Results retained

The abstract's observed extrema are preserved:

- 18.7 kN maximum recorded load;
- 9.6 cm maximum recorded hitch slip;
- 3 mm maximum recorded diameter change.

Every occurrence labels these as observations within the short bench stroke, not breaking strengths, safe working loads, design limits, acceptance criteria, or recommendations.

The end-stroke mean load trends are retained to distinguish treatment means from the individual maximum: about 11.9 kN (`AP`), 15.1 kN (`XP`), 9.3 kN (`AT`), and 8.6 kN (`XT`). End-stroke slip and diameter summaries are retained as comparative observations, again without thresholds.

Hitch type had a reported significant effect on load and slip at `p < 0.001`. Prusik treatments developed greater grip, higher observed load, and less slip than Valdotain-tresse treatments. The corpus explicitly blocks the shortcut that more grip or less slip is automatically safer: in this experiment, greater grip also coupled to greater load and more deformation in parts of the system.

Rope type had a reported significant effect on diameter change at `p < 0.001`. XTC 16 deformed more than Axis. The digest does not attribute this to diameter, polymer, or braid alone because each named rope bundles several properties.

Adjusted `R²` values are retained:

- load versus diameter reduction: `0.445`;
- load versus slip: `0.418`;
- diameter reduction versus slip: `0.378`;
- combined load model using slip and diameter reduction: `0.535`.

The nonlinear slip terms and incomplete explained variation are noted. Full coefficients are excluded so the paper cannot be turned into an unsupported field-load calculator.

Start-to-end temperature changes were about 6.2 °C (`AP`), 6.8 °C (`AT`), 6.7 °C (`XP`), and 3.1 °C (`XT`), without a reported significant difference among treatments. The digest distinguishes that result from no heating and from dynamic thermal safety.

## Limitations made unusually prominent

The document opens with five limitation paragraphs before the research question. It states that the experiment had:

- two exact synthetic arborist rope systems;
- two hitches;
- 36 slow pulls;
- a short 250 mm bench stroke;
- a modified standardized method;
- no human participant;
- no dynamic, fall, shock, swing, or cyclic loading;
- no continuous or dynamic peak-temperature measurement;
- no natural-fiber rope;
- no bondage, shibari, kinbaku, or recreational human suspension;
- no medical, anatomical, consent, anchor, or emergency-release outcome.

The article itself asks for future falling-body and cyclic tests. The digest uses that request as evidence that those questions remained untested, not as an invitation to conduct unsafe informal experiments.

The machine stopped at its programmed stroke. Consequently, the paper's load observations are not silently relabeled as breaking strengths. This distinction is especially important because of the article title.

## Preliminary and no-best-knot purpose

The authors explicitly said their purpose was not to define the best knot. The digest repeats this at the beginning, in the results interpretation, in supported conclusions, and in the derivation rules.

The conclusion's suggestion that both combinations could be used is not transformed into field or life-support advice. Instead, the corpus retains the measured comparative behaviors and the authors' preliminary framing. Equipment choice, regulatory drafting, certification, and suitability remain excluded.

## Cross-domain boundary

The source involved occupational tree-climbing materials but no live tree climber. It cannot establish:

- field arborist suitability;
- a safe working load or safety factor;
- dynamic fall or rescue performance;
- performance of aged, wet, contaminated, UV-exposed, frozen, or differently tied equipment;
- natural-fiber behavior;
- bondage or human-suspension suitability;
- anatomical, circulatory, nerve, skin, consent, or emergency-response safety.

This is not an implication that all rope systems behave the same. It is the opposite: the paper shows that even within four tightly controlled treatments, rope and hitch factors changed different coupled outcomes. Transfer requires new evidence for the new system and load case.

## Citation and split hygiene

Every source-derived paragraph or compact result list in `CORPUS.md` carries a citation such as `[MDPI-PDF: §3 Results and Discussion]`. The source ledger maps citation IDs to exact provenance. Identity and licence citations are metadata-only.

The official article HTML identity, PDF, DOI record, Markdown, and every derivative must remain in one source-family split before chunking or question generation. Questions may test design, variables, relationships, and limitations. They may not test DOI, URL, author order, publication date, manufacturers, brands, standards, figure/table order, references, or assembly instructions.

QA, dataset review, dataset, trainer, training-projection, validation, evaluation, holdout, OOD, shadow, benchmark, probe, and unrelated corpus artifacts were not accessed.

## Package inventory

- `CORPUS.md`: manually adapted, directly trainable CC BY 4.0 Markdown.
- `sources.jsonl`: five-record source and licence ledger with hashes and timestamps.
- `dispositions.jsonl`: 56 manual inclusion, narrowing, metadata, and exclusion decisions.
- `source_snapshot/provenance.json`: access, rights, component, study-design, result, limitation, and split audit.
- `manifest.json`: deterministic package metadata and output hashes.
- `tests/test_corpus.py`: deterministic source-integrity, content, safety-boundary, and package checks.
