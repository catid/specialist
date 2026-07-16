# Europe PMC BDSM-fatality review corpus report

## Outcome

This package adds a 2,974-word, non-Q&A Markdown evidence digest of Schori, Jackowski, and Schön's forensic literature review of published BDSM-associated fatalities. The direct-training artifact emphasizes method, denominators, aggregate cause categories, missing data, forensic terminology, and limits on causal or prevalence inference.

The complete authorized EMBL-EBI JATS is retained for provenance under CC BY 4.0 but is not a direct-training artifact. The Markdown excludes every individual case narrative, apparatus description, body position, sequence of failure, relationship profile, and case-to-reference mapping.

## Source and access

- PMCID: PMC8813685
- DOI: 10.1007/s00414-021-02674-0
- PMID: 34383118
- Authorized route: `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8813685/fullTextXML`
- Remote XML: 84,800 bytes; SHA-256 `70bbf0a86d13f9b2b34ef99f3b78bcc7cf3ee87b34dfd0eba78f7941fe7bd0bc`
- Local snapshot: 84,801 bytes after one terminal LF; SHA-256 `5c1e66790925eed475b554fe673c04ba6a42363ba1691aa3e2d20934f6434bcf`

No publisher page, United States PMC page, generated PDF, supplementary route, or other article body was accessed. XML parsing supported navigation and arithmetic checks; the training Markdown was selected, rewritten, and critically reviewed by hand.

## Rights

The JATS identifies the article as © The Author(s) 2021 under Creative Commons Attribution 4.0. The corpus provides article attribution, a license link in the manifest and provenance, and a change notice explaining that the source was reorganized, paraphrased, corrected, narrowed, and supplemented with methodological critique.

No source figure or expressive table layout appears in the training Markdown. One small denominator table is original to the digest. It reorganizes four reported counts to show why 17 published cases, three of 74 selected deaths, and three of 16,437 autopsies answer different questions.

## Manual evidence audit

The source reports 14 case reports plus one study and compiles 17 cases. The training document preserves these numbers as a publication-derived case set. It does not turn 17 into a population, participant, scene, or exposure denominator.

The included German postmortem study contains:

- 16,437 autopsies;
- 74 non-natural deaths associated with sexual activity;
- three BDSM-associated cases.

The paper reports 15 of 17 selected cases as strangulation. That is retained as a signal within the selected publications, not an estimate that 88.2 percent of all BDSM deaths or dangerous incidents have that cause.

The source reports deceased-person toxicology in 13 cases, eight positive and five negative. Four partners reportedly had the same detected substance. These counts are retained as selected-case co-occurrence, not proof of impairment, contribution, correlation, or causation.

## Problems corrected or gated

The review's abstract and conclusion are stronger than its design in several places. The corpus explicitly prevents the following conversions:

- a case-report count becoming incidence or prevalence;
- one institutional autopsy fraction becoming risk per participant or scene;
- heterogeneous autoerotic, natural-death, and partnered-BDSM studies becoming directly comparable rates;
- positive toxicology becoming proof that substances caused a death;
- a fatal-only sample becoming a correlation analysis;
- missing experience data becoming evidence that inexperience caused fatalities;
- community safeguards becoming proven interventions;
- an author recommendation becoming proof that all cases are preventable.

Four source issues receive explicit treatment:

1. The stated 64.3-percent substance-involvement figure has no clear denominator in the prose. It appears to be nine of 14 cases with some participant toxicology or intoxication information, not nine of all 17.
2. Table 1 appears to exchange the descriptions attached to “traumatic” and “positional” asphyxia. The digest flags the anomaly rather than silently teaching it.
3. The results refer to two cases with breath-control instruction and CPR teaching, while the safeguards discussion refers to three cases with instruction and CPR discussion.
4. The conclusion links fatalities to lack of knowledge even though knowledge was not consistently measured and both participants were described as non-novices in all nine cases with sufficient experience reporting.

## Sensitive-content boundary

The source's case table contains enough act, apparatus, body-position, relationship, and toxicology detail to become sensational or instructional. None of it is reproduced in `CORPUS.md`. Aggregate categories are sufficient to support the valid safety lesson.

The corpus contains no method for neck compression, breath control, loss of consciousness, suspension, bondage construction, or emergency release. It states that breath-control risk cannot be eliminated and that a safeword, observer, experience, instruction, resuscitation knowledge, or previous nonfatal outcome does not establish safety.

Forensic reconstruction is also narrowed. The digest retains uncertainty created by emergency intervention, witness stress, stigma, and possible scene changes, but excludes operational search tactics and device reconstruction.

## Split and sampling controls

The direct-training document is one canonical source unit identified by PMCID PMC8813685. Assign the source to one split before chunking or deriving QA, and keep all descendants in that split. No validation, OOD, shadow, sealed-holdout, or protected-QA data were accessed.

Later QA may test:

- what the 17-case denominator represents;
- why the German autopsy ratios are not population incidence;
- what missing toxicology, experience, or safeword data permit;
- the difference between strangulation, suffocation, and broader mechanical asphyxia;
- why positive toxicology does not establish causation;
- why selected case reports cannot compare prevalence.

It must not ask for title, author, DOI, PMCID, PMID, publication year, case number, page, or URL recall, and it must not reconstruct an individual incident.

## Package

- `CORPUS.md`: 2,974-word manually written training digest.
- `dispositions.jsonl`: 22 include, narrow, critique, and exclusion decisions.
- `manifest.json`: source identity, license, training role, hashes, split rules, and safety gates.
- `source_snapshot/fullTextXML.xml`: authorized CC BY 4.0 JATS, provenance only.
- `source_snapshot/provenance.json`: exact access, normalization, rights, evidence audit, and training boundary.
- `tests/test_corpus.py`: offline integrity, arithmetic, anti-trivia, no-technique, no-prevalence, and source-issue tests.

The corpus is deliberately short relative to the source. It retains the information that changes an answer while omitting narrative detail that does not improve inference or safety.
