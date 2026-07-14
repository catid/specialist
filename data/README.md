# Dataset status

## Current training artifact

[`train_qa_curated_v1.jsonl`](train_qa_curated_v1.jsonl) is the current curated
training set. Its
[`deterministic build report`](train_qa_curated_v1.report.json) verifies 1,487
unique questions and fact IDs:

| Validated tranche | Accepted rows | Artifact and audit |
| --- | ---: | --- |
| Leakage-gated and manually audited base | 1,314 | [`train_qa_verified_leakfree_v2.jsonl`](train_qa_verified_leakfree_v2.jsonl) supplied 3,113 rows; 1,797 explicit manual drops and two additional distinctive-answer aliases are excluded during the curated merge. |
| Source-document manual QA | 81 | [`train_qa_manual_v1.jsonl`](train_qa_manual_v1.jsonl), with provenance in its [`report`](train_qa_manual_v1.report.json) |
| Owner-curated resource directory | 29 | [`rope_resource_qa_v1.jsonl`](rope_resource_qa_v1.jsonl): 23 direct resource answers plus six category answers; see its [`report`](rope_resource_qa_v1.report.json) |
| Manually reviewed resource facts | 38 | [`rope_resource_factual_qa_v1.jsonl`](rope_resource_factual_qa_v1.jsonl), with evidence and reviewer provenance summarized in its [`report`](rope_resource_factual_qa_v1.report.json) |
| Additional manually reviewed resource facts | 10 | [`rope_resource_manual_v1.jsonl`](rope_resource_manual_v1.jsonl), audited against live Crash Restraint, Knot Head Nylon, and Shibari Study evidence in [`manual_reviews/resources/additions_audit_v1.jsonl`](manual_reviews/resources/additions_audit_v1.jsonl) |
| Rope-topia resource index | 15 | [`rope_topia_manual_v1.jsonl`](rope_topia_manual_v1.jsonl) contains manually reviewed title-to-canonical-URL mappings only because the live article bodies are demo-gated; see its [`report`](rope_topia_manual_v1.report.json). |
| **Curated v1 total** | **1,487** | [`train_qa_curated_v1.jsonl`](train_qa_curated_v1.jsonl) and [`report`](train_qa_curated_v1.report.json) |

The fact-ID-keyed general
[`curation decisions`](train_qa_curated_v1.curation.jsonl) and complete
[`Kinbaku Today audit`](train_qa_kinbakutoday.curation.jsonl) contain 1,827
reviewed actions: 1,797 drops and 30 evidence-backed edits. The source-grouped
review covered every one of the 1,240 Kinbaku Today, 707 Rope365, 716 Esinem,
and 325 Wikipedia base records; a stricter second Kinbaku pass then reviewed
all 1,174 formerly retained Kinbaku rows and retained 423. It removes
translated/mixed-language copies, unsafe or
medically unsupported snippets, volatile promotion, unrelated encyclopedia
trivia, semantic duplicates, contextless questions, and source misreads. Every
edit pins the expected original Q&A, reviewer, date, reason, source URL, and an
extractive evidence passage; stale or unused decisions make the build fail.

Dataset refreshes use an explicit snapshot boundary. Manual workers write
source-grouped pending ledgers while an A/B run is active. Between runs, the
reviewed ledgers are promoted, the deterministic JSONL and Arrow artifacts are
rebuilt, and their hashes are pinned before either arm starts. The HPO runner
rehashes train/validation Arrow files, manifest, model identity, and trainer
source at every run boundary and refuses to mix a changed snapshot with cached
results. The prior 2,228-row S3 cohort is preserved in
[`../experiments/eggroll_es_hpo/snapshots/s3/`](../experiments/eggroll_es_hpo/snapshots/s3/).
The complete stricter Rope365, Esinem, and Wikipedia passes remain pending
under [`manual_reviews/rope365/`](manual_reviews/rope365/),
[`manual_reviews/esinem_second_pass/`](manual_reviews/esinem_second_pass/), and
[`manual_reviews/wikipedia_second_pass/`](manual_reviews/wikipedia_second_pass/)
for the next promoted snapshot. They are not part of the current 1,487-row
artifact. Rope365 proposes 323 drops and 21 edits across all 601 retained rows;
Esinem proposes 73 drops and four edits across all 150 retained rows; Wikipedia
proposes 175 drops and two edits across all 204 retained rows. Their combined,
replacement-aware temporary build passes twice byte-identically and projects
916 unique rows: 278 Rope365, 77 Esinem, and 29 Wikipedia. Their separation
preserves the completed S4 A/B cohort exactly.

The merge also re-renders every retained record as canonical
`Question: ...\nAnswer: ...` text from its validated structured fields. Legacy
generator instructions and alternate `Q:`/`A:` serializations therefore cannot
become training targets even when their structured Q&A was valid.

The manual tranche covers nine complete source documents and 156 reviewed
candidates. [`manual_qa_candidates_v1.jsonl`](manual_qa_candidates_v1.jsonl) is
the compact candidate subset consumed by those decisions; it is an audit input,
not training data. `eval_qa*.jsonl` contains development evaluations only, not
independent final test sets.

The pilot's reviewers consolidated 104 candidate IDs into 26 corrected facts,
dropped 52 candidate IDs, and manually added 55 useful missing facts. They also
caught semantic evaluation collisions involving Somerville bowline, agura
shibari, `Ipponnawa`/`Ippon nawa`, Demon Tie/`tengu shibari`, the function of a
bend, and a double-/two-half-hitches alias. The transliteration-spacing case is
now a deterministic leakage-gate regression.

## Resource directory and live coverage

The [`owner-curated manifest`](../sources/rope_resources_v1.json) preserves all
23 user-supplied URLs verbatim alongside canonical destinations and bounded
collection rules. The 29 directory QA records preserve all 23 resources in
training; they do not imply that every linked page was copied or endorsed.

The scoped live collection saved 165 relevant public documents. The compact
[`coverage report`](rope_resources_coverage_v1.json) records all 23 resources:
nine produced documents, eleven were blocked by reference-only policy, robots
content signals, access challenges, or an unavailable YouTube adapter, and
three rendered too little static text for safe ingestion. It also retains the
29 blocked outcomes and 564 scope/extraction skips, with no failed fetches.
Raw pages and per-page coverage are intentionally ignored under `data/raw/`;
the tracked report is the reproducible coverage artifact. Detailed collection
boundaries and exclusions are in the
[`source README`](../sources/README.md), while the 38 admitted factual records
come from independently reviewed packets under
[`sources/manual_facts/`](../sources/manual_facts/).

## Training defaults

[`sft_lora.py`](../sft_lora.py) defaults to
`data/train_qa_curated_v1.jsonl`. [`es_train_acc.py`](../es_train_acc.py) keeps
`data/train_qa_verified_leakfree_v2.jsonl` as its legacy default so recorded ES
runs remain reproducible. Select curated v1 explicitly for a new ES run:

```bash
python es_train_acc.py --data data/train_qa_curated_v1.jsonl [other options]
```

## Invalidated and intentionally not tracked

The prior `train_qa_v3_clean_leakfree_v2.jsonl` contained 55,773 rows, but
28,017 `qa_chat` rows had ChatML/`<think>` delimiters leaked into their parsed
question and answer metadata. Correct structural parsing collapses paired raw
and chat renderings and projects 27,756 unique survivors before manual review.
The strengthened transliteration/spacing-alias gate conservatively removes 27
more semantic evaluation collisions, leaving 27,729 future review candidates.

That generated file must not be used for training. A source-grounded manual
curation pass is replacing it: reviewers receive small packets grouped by
source document, make explicit keep/edit/drop decisions, and add better
self-contained Q&A with source evidence.

## Manual review and rebuild

- [`prepare_manual_qa_review.py`](../prepare_manual_qa_review.py) joins
  candidates to one complete source per
  small packet. It performs no generation or judging.
- [`MANUAL_QA_CURATION.md`](../MANUAL_QA_CURATION.md) defines the manual
  keep/edit/drop/add protocol.
- [`build_manual_qa.py`](../build_manual_qa.py) requires complete candidate
  coverage for every reviewed
  document, verifies exact source evidence and extractive answers, reruns the
  leakage gate, rejects duplicates and protocol tokens, and emits deterministic
  `qa_manual` records.

Run the builders from the repository root. Refreshing the live corpus is
optional; the remaining builders deterministically regenerate the tracked
reports and curated dataset from their reviewed inputs:

```bash
.venv/bin/python collect_resource_corpus.py --disable-youtube
.venv/bin/python build_resource_coverage_report.py
.venv/bin/python build_manual_qa.py
.venv/bin/python build_resource_qa.py
.venv/bin/python build_resource_facts.py
.venv/bin/python build_curated_qa.py
```

The corresponding entry points are
[`collect_resource_corpus.py`](../collect_resource_corpus.py),
[`build_resource_coverage_report.py`](../build_resource_coverage_report.py),
[`build_resource_qa.py`](../build_resource_qa.py),
[`build_resource_facts.py`](../build_resource_facts.py), and
[`build_curated_qa.py`](../build_curated_qa.py).
