# Europe PMC entrapment-neuropathy review corpus report

## Outcome

This package adds a 3,002-whitespace-word, non-Q&A Markdown digest of Schmid, Fundaun, and Tampin's 2020 narrative review of entrapment neuropathies. The direct-training artifact preserves clinically useful distinctions among mechanisms, symptoms, examination signs, diagnoses, and investigations while keeping human findings separate from animal and other preclinical evidence.

The article is relevant to rope-related knowledge only as bounded medical background. It did not study rope bondage, suspension, wrap pressure, contact geometry, placement, loading direction, duration, checking schedules, release, or return to rope. The digest therefore provides no numerical pressure, placement, duration, release-timing, recovery, prognosis, treatment, or self-care guidance. It directs neurological symptoms after rope to a qualified medical professional.

The complete authorized EMBL-EBI JATS is retained under CC BY 4.0 for provenance and offline audit. It is not a direct-training artifact.

## Source and access

- PMCID: PMC7382548
- DOI: 10.1097/PR9.0000000000000829
- PMID: 32766466
- Human-facing metadata locator: `https://europepmc.org/article/MED/32766466` (recorded as an identifier; not used as an article-body source)
- Sole article-body route: `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7382548/fullTextXML`
- Remote XML: 220,620 bytes; SHA-256 `9d3244720e670f574c8b203635f54d6c00f7cdcfbcf89d1c8155bba486d086b5`
- Local snapshot: 220,621 bytes after one terminal LF; SHA-256 `da030b3e0a84759e96c4639399f8b41921c277df638213a3d1bd77848e38c0cc`

No publisher page, United States PMC page, PDF, supplementary route, or external figure, thumbnail, or image-table binary was accessed. XML parsing supported navigation, metadata checks, caption review, and integrity tests. The training document itself was selected, rewritten, bounded, and checked by hand.

## Rights

The JATS identifies the article as copyright 2020 by the authors and distributed under Creative Commons Attribution 4.0. The corpus provides article attribution and a change notice. The manifest and provenance record the license link and explain that the source was manually selected, paraphrased, reorganized, narrowed, and supplemented with evidence-design and transfer-limit annotations.

The article contains four external figure images and one image-based table. None was fetched or retained. Figure 2 is explicitly described in its caption as adapted from reference 55, so it is treated as third-party/adapted visual content. The other visuals are also excluded rather than assuming that article-level licensing resolves every component-level question. Captions were read only to identify subject and disposition.

## Review-design audit

The paper describes itself as a contemporary review and narratively synthesizes preclinical models, human observational findings, diagnostic studies, guidelines, trials, and other reviews. It does not report a systematic search strategy, prespecified eligibility criteria, duplicate screening, study-flow diagram, or uniform risk-of-bias assessment.

That design does not make the review unhelpful, but it changes the proper training treatment. Statements remain labeled as one of the following:

- preclinical observation;
- human association;
- human diagnostic-performance finding;
- clinical inference or proposed mechanism;
- guideline or management recommendation.

The digest retains the first four only when relevant to mechanisms, presentation, or assessment. It excludes substantive management and prognosis content. It does not silently flatten cited evidence into one level of certainty.

## Human-versus-preclinical boundary

The source emphasizes that much neuropathic-pain biology comes from acute and severe animal injury models, whereas common human entrapment neuropathies may develop slowly under milder chronic compromise. The digest preserves that warning wherever experimental biology might otherwise look operational.

The source's experimental compression-pressure values are omitted completely. A magnitude observed in one experimental system is not a human no-injury threshold and cannot be transferred across species, tissue, anatomical site, measurement method, pressure distribution, shear, stretch, repeated exposure, individual susceptibility, or rope contact geometry.

The retained mechanism map includes ischemia, edema, proposed fibrosis, demyelination, conduction disturbance, axonal injury, large- and small-fiber changes, neuroinflammation, axonal-transport disruption, central adaptation, and contextual consequences. It does not claim that every mechanism is present in every person or that one sensation identifies a mechanism.

## Clinical distinctions retained

The source makes several distinctions that materially improve the corpus:

- radicular pain can occur without the loss of function that defines radiculopathy;
- pain, a demonstrable nerve lesion, and heightened mechanosensitivity are not interchangeable;
- mechanosensitivity can exist without a demonstrable lesion, and an entrapment neuropathy can exist without heightened mechanosensitivity;
- symptoms may extend beyond a canonical dermatome or named nerve territory;
- nerve compression on imaging can be asymptomatic;
- large- and small-fiber changes can differ, so a coarse touch comparison does not assess every form of nerve function;
- symptom, examination sign, test result, and diagnosis are different levels of evidence.

The article discusses professional clinical assessment as integrated history, examination, differential diagnosis, and additional testing when indicated. The digest does not turn this into a lay checklist. It excludes procedure instructions for provocation maneuvers and excludes the image-based differential-diagnosis table.

## Test limitations retained

The training document keeps the limitations next to each investigation:

- early carpal tunnel involvement may be missed by nerve-conduction studies;
- the value of nerve-conduction testing is less established for some proximal conditions;
- MRI can produce false-positive and false-negative results;
- an imaging level may not match the clinical examination;
- ultrasound nerve enlargement is one finding within a workup, not a standalone diagnosis;
- performance depends on the condition, stage, population, protocol, reference standard, and interpreter.

These limitations block both directions of overclaim: no single positive finding establishes a diagnosis, and no single negative result clears a symptomatic nerve.

## Excluded clinical material

Section 5 contains natural-history estimates, prognostic associations, physiotherapy and occupational-therapy discussion, medication and injection findings, surgical outcomes, complications, and selection criteria. None of those details appears as substantive training content.

Those data concern diagnosed carpal tunnel syndrome and heterogeneous clinical populations grouped under sciatica. They do not answer what to do after an acute rope symptom. They cannot define release timing, recovery, prognosis, home treatment, or return-to-rope timing. The digest mentions the section only to establish that non-transfer boundary.

The conclusion's proposed personalized-management model is similarly excluded. Only the underlying point that patient mechanisms and presentations are heterogeneous is retained.

## Clinical and rope-transfer gate

The direct-training document explicitly rejects derivation of:

- safe rope pressure, tension, or loading magnitude;
- safe placement, body location, wrap width, loading direction, or geometry;
- safe duration, check interval, or post-symptom waiting period;
- a release technique or judgment that release was sufficiently fast;
- injury severity from one symptom or from improvement;
- recovery time, prognosis, treatment, self-care, or return-to-rope timing;
- direct claims that a rope configuration is safe or unsafe based on this paper.

It states that numbness, tingling, altered sensation, electric or burning pain, weakness, loss of coordination, or other neurological symptoms after rope require evaluation by a qualified medical professional. It is not an emergency-triage protocol, diagnosis tool, or substitute for current clinical guidance.

## Split and derivation controls

The direct-training document is one canonical source unit identified by PMCID PMC7382548. Assign the source to one split before chunking or deriving QA, and keep all descendants in that split. No QA artifact, training projection, evaluation set, validation set, OOD set, shadow set, benchmark, probe, sealed holdout, or protected dataset was accessed during the build.

Later QA may test evidence-type distinctions, clinical terminology, mechanism uncertainty, symptom-versus-sign reasoning, and investigation limitations. It must not ask for title, author, DOI, PMCID, PMID, publication date, page, URL, experimental pressure, duration, placement, treatment, or prognosis recall. It must not turn the paper into a self-test or rope-design recipe.

## Package

- `CORPUS.md`: 2,977 Unicode-regex words and 3,002 whitespace-delimited words of manually written direct-training Markdown.
- `dispositions.jsonl`: 25 manual inclusion, narrowing, critique, visual-audit, and exclusion decisions.
- `manifest.json`: source identity, rights, role, hashes, safety boundaries, and split rules.
- `source_snapshot/fullTextXML.xml`: authorized CC BY 4.0 JATS, provenance only.
- `source_snapshot/provenance.json`: exact source access, normalization, evidence audit, visual audit, and training boundary.
- `tests/test_corpus.py`: offline identity, integrity, rights, anti-threshold, anti-treatment, clinical-distinction, visual-exclusion, and split-hygiene checks.

The result is intentionally far shorter than the source. It keeps the content that helps distinguish evidence and clinical concepts while excluding details that could be misread as operational medical or rope advice.
