# Curation, OCR, evidence, and rights report

## Outcome

Accepted as a bounded historical-methods corpus. The retained artifact is a manually written, claim-cited digest of a 1990 mechanism model for deterioration of synthetic-fiber double-braided marine rope. It preserves the distinction between tensile creep-fatigue and internal fiber-to-fiber abrasion, the geometrical and failure assumptions, input-data provenance, validation confounds, and reported evidence limits.

The corpus excludes formulas as calculators; every numerical curve, load, pressure, cycle, strength, frequency, and service-life value; Tables 1–2; Figures 1–13d; commercial material identities; third-party recommendations; safety factors; ratings; and operational advice. It makes no transfer to natural fiber, knots, care or retirement, bondage, body contact, uplines, anchors, or human suspension.

## Official record and retrieval

The official NOAA Institutional Repository record and its bound main-document URL were retrieved successfully on 2026-07-16. The record supplied:

- indexed title and personal/corporate authors;
- publication year 1990 and series MITSG 90-18;
- grant `NA86AA-D-SG089` and project `RT-11`;
- document type “Technical Report”;
- rights information “Public Domain”;
- main-document download URL; and
- main-document SHA-256 `7a29a928c08d6257f1558eac2d63a26c8e662240e01ee87400cedc95b70d242f`.

The downloaded PDF is 860,928 bytes and independently hashes to the exact value published in the record. The repository HTML is 46,777 bytes with SHA-256 `e77baf777031dc27a979976b7d827e3439c26f6fee1c0799dc4603cdc26c6191`.

## Image-only scan and OCR audit

The PDF contains 41 pages and no extractable text. Its metadata identifies Acrobat’s scan plug-in as creator and producer, with scan-era creation and modification timestamps in 1999; those timestamps are not the report’s publication date. Each page contains one 1-bit grayscale raster image at 160 pixels per inch. The first page image is 1320 by 1752 pixels; the remaining page images are 1312 by 1744 pixels.

For review, all pages were rendered locally as grayscale images at 300 pixels per inch and OCRed with Tesseract 5.3.4 using the English model. The 41 ordered OCR outputs contain 40,254 bytes and 6,762 whitespace-delimited words. Their ordered concatenation hashes to `272238d64f26759d0437e5fd8529d55e478c101f6392923cef8dfd106fd808f8`.

OCR was a navigation aid, not an authoritative transcription. Every page was checked visually in ordered contact sheets, and key prose pages were inspected at full rendered resolution. Equations, tabular values, plot coordinates, and uncertain OCR tokens were not carried into the corpus. No PDF, render, OCR output, or contact sheet is stored in the repository.

## Complete page review

- PDF page 1: outer technical-report cover; identity only.
- PDF page 2: title page; title spelling, author order, series, institution, grant, and project retained as identity facts.
- PDF page 3: blank separator; excluded.
- PDF page 4 / report page 1: abstract; two-mechanism purpose and claimed regime behavior included with qualification.
- PDF page 5 / report page 2: contents; used for navigation only.
- PDF pages 6–7 / report pages 3–4: introduction, marine context, extrapolation problem, prior microscopy, prior model, and opening structural geometry; selected conceptual claims included.
- PDF pages 8–9 / report pages 5–6: structural calculation, internal-contact mechanism, vibration hypothesis, first assumption set, and opening fiber/yarn evidence; included in manual paraphrase.
- PDF pages 10–11 / report pages 7–8: tensile-fatigue and yarn-wear inputs, formulas, empirical curve segments, and start of rope prediction; concepts included, equations and parameters excluded.
- PDF pages 12–13 / report pages 9–10: second assumption set, internal/external wear distinction, load-transfer discussion, termination failures, and hysteretic-heating statement; selected claims and confounds included.
- PDF page 14 / report page 11: model-input setup and Table 1; comparability assumption included, table and values excluded.
- PDF page 15 / report page 12: wet nylon/PET mid-span comparisons and residual-strength setup; qualitative evidence and uncertainty included.
- PDF page 16 / report page 13: Table 2 and discussion beginning; table excluded, discrepancies summarized without values.
- PDF page 17 / report page 14: residual-strength interpretation and failure-mode ambiguity; included with author-judgment qualification.
- PDF page 18 / report page 15: termination/external-wear model and equations; evidence limits included, equations and geometric recipe excluded.
- PDF pages 19–20 / report pages 16–17: local/uniform external-wear analysis and comparison; retained only as qualitative model/confound discussion.
- PDF page 21 / report page 18: low- versus high-pressure sensitivity discussion and beginning conclusions; conceptual sensitivity and synthetic-parameter caveat included.
- PDF page 22 / report page 19: conclusions and beginning references; conclusions critically summarized, recommendations and reference text excluded.
- PDF pages 23–24 / report pages 20–21: references only; excluded as third-party works not independently audited.
- PDF page 25 / report page 22: Figure 1, strand-path diagram; excluded.
- PDF page 26 / report page 23: Figure 2, square-bundle diagram and formula; excluded.
- PDF page 27 / report page 24: Figure 3, computation flow diagram; excluded.
- PDF page 28 / report page 25: Figure 4, empirical yarn-wear curve; excluded.
- PDF page 29 / report page 26: Figure 5, rope-specimen diagram; excluded.
- PDF page 30 / report page 27: Figure 6, wet-nylon prediction/data plot; excluded.
- PDF page 31 / report page 28: Figure 7, wet-PET prediction/data plot; excluded.
- PDF page 32 / report page 29: Figure 8, modeled material comparison; excluded.
- PDF page 33 / report page 30: Figure 9, modeled residual-strength and life curves; excluded.
- PDF page 34 / report page 31: Figure 10, measured/predicted residual-strength plot; excluded.
- PDF page 35 / report page 32: Figure 11a,b, series-model plots; excluded.
- PDF page 36 / report page 33: Figure 12a, nylon termination comparison; excluded.
- PDF page 37 / report page 34: Figure 12b, PET termination comparison; excluded.
- PDF page 38 / report page 35: Figure 13a, nylon low-pressure sensitivity; excluded.
- PDF page 39 / report page 36: Figure 13b, PET low-pressure sensitivity; excluded.
- PDF page 40 / report page 37: Figure 13c, nylon high-pressure sensitivity; excluded.
- PDF page 41 / report page 38: Figure 13d, PET high-pressure sensitivity; excluded.

Machine-readable page dispositions appear in `dispositions.jsonl` and cover the exact set of PDF pages 1 through 41.

## Scientific interpretation audit

The report’s model is not a single empirical S–N fit. It combines a geometrical double-braid model with two damage inputs: tensile creep-fatigue parameterized by accumulated exposure time and yarn-on-yarn wear parameterized by friction cycles and contact pressure. This supports the corpus explanation that cycling frequency can change which mechanism dominates even when one input is described as frequency-independent.

The model relies on ideal friction endpoints, identical strands and filaments, prescribed bundle geometry, wear restricted to a high-curvature surface, layerwise removal, unchanged curvature, uniform fatigue, limited fatigue/wear interaction, relative movement every cycle, and weakest-location failure. All are presented as assumptions, not facts about every rope.

The lowest-pressure wear region is weakly supported: the report says little data were available there and supplies an extrapolated form. Later low-load sensitivity plots use systematically varied parameters that do not represent measured yarn data. Because the modeled low-tension rope regime depends on this input, the digest makes that evidence gap prominent.

Validation data are heterogeneous and largely inherited. Many reports omitted construction detail; the authors assumed geometrical similarity. Termination failures dominate much of the literature, wet mid-span cases are limited, residual-strength agreement is mixed, and the external-wear extension is qualitative. The source’s claims of reasonable agreement are preserved as author assessments rather than upgraded to universal validation.

The hysteretic-heating statement is supported by one prose passage explaining the choice of wet comparison data. No temperature dataset or thermal submodel appears. The corpus therefore includes heating only as a qualitative dry-test confound.

## Rights and component audit

The NOAA record explicitly labels the bound main document “Public Domain” and publishes the matching checksum. This is strong source-level permission for manual reuse. The corpus nonetheless excludes source scans and components not needed for the educational digest.

Tables and figures package values and observations originating partly in cited studies and named commercial test materials. References are third-party works. Equations and parameter sets could become operational calculators when detached from assumptions. All are excluded despite the report-level public-domain designation. This is a conservative operational screen, not legal advice.

## Split hygiene and verification

This package is training-source-only. It must not seed validation, holdout, evaluation, OOD, or probe data. Any future Q&A must remain training-only, preserve historical provenance and assumptions, distinguish reported evidence from model prediction, and avoid operational transfer.

Deterministic tests cover file inventory, identity, checksum binding, public-domain status, full page review, OCR audit, mechanisms, assumptions, evidence limits, hysteretic-heating scope, numerical and operational exclusions, source-body exclusion, split hygiene, and manifest integrity.
