# Dataset status

## Current training artifact

[`train_qa_curated_v1.jsonl`](train_qa_curated_v1.jsonl) is the current curated
training set. Its
[`deterministic build report`](train_qa_curated_v1.report.json) verifies 3,258
unique questions and fact IDs:

| Validated tranche | Accepted rows | Artifact and audit |
| --- | ---: | --- |
| Leakage-gated base | 3,110 | [`train_qa_verified_leakfree_v2.jsonl`](train_qa_verified_leakfree_v2.jsonl) supplied 3,113 rows; the curated merge report excludes three additional distinctive-answer aliases. |
| Source-document manual QA | 81 | [`train_qa_manual_v1.jsonl`](train_qa_manual_v1.jsonl), with provenance in its [`report`](train_qa_manual_v1.report.json) |
| Owner-curated resource directory | 29 | [`rope_resource_qa_v1.jsonl`](rope_resource_qa_v1.jsonl): 23 direct resource answers plus six category answers; see its [`report`](rope_resource_qa_v1.report.json) |
| Manually reviewed resource facts | 38 | [`rope_resource_factual_qa_v1.jsonl`](rope_resource_factual_qa_v1.jsonl), with evidence and reviewer provenance summarized in its [`report`](rope_resource_factual_qa_v1.report.json) |
| **Curated v1 total** | **3,258** | [`train_qa_curated_v1.jsonl`](train_qa_curated_v1.jsonl) and [`report`](train_qa_curated_v1.report.json) |

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
