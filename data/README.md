# Dataset status

## Active low-regression training QA authority

The `plan.md` Qwen3.6 low-regression program does **not** train from the legacy
784-row curated artifact described below. Its immutable QA authority is the
516-row V440 logical view after removal of the 15 URL-index rows, partitioned
by source document before any derivation:

- training: [`training_inventory/source_group_split_v1/train_v440_qa.jsonl`](training_inventory/source_group_split_v1/train_v440_qa.jsonl), 382 rows;
- development: [`training_inventory/source_group_split_v1/development_v440_qa.jsonl`](training_inventory/source_group_split_v1/development_v440_qa.jsonl), 74 rows;
- final: 60 rows represented only by aggregate counts and commitments, with no
  semantic records emitted.

[`training_inventory/source_group_split_authority_v1.json`](training_inventory/source_group_split_authority_v1.json)
is the authority for those projections. It seals exact, near-duplicate,
source-URL, source-group, and descendant-fact disjointness; the final partition
stays redacted. Verify the already materialized safe projections byte-for-byte
with:

```bash
.venv/bin/python build_source_group_split_authority_v1.py --check
```

This check does not reopen any quarantined legacy evaluation source or emit
final records.

## Retired legacy curated artifact

[`train_qa_curated_v1.jsonl`](train_qa_curated_v1.jsonl) is retained for older
trainer lineage reproduction; it is not an input to the `plan.md` protocol. Its
[`deterministic build report`](train_qa_curated_v1.report.json) verifies 784
unique questions and fact IDs:

| Validated tranche | Accepted rows | Artifact and audit |
| --- | ---: | --- |
| Leakage-gated and manually audited base | 630 | [`train_qa_verified_leakfree_v2.jsonl`](train_qa_verified_leakfree_v2.jsonl) supplied 3,113 rows; the active ledgers make every exclusion explicit. |
| Source-document manual QA | 72 | [`train_qa_manual_v1.jsonl`](train_qa_manual_v1.jsonl), with provenance in its [`report`](train_qa_manual_v1.report.json) |
| Owner-curated resource directory | 27 | [`rope_resource_qa_v1.jsonl`](rope_resource_qa_v1.jsonl): direct and category resource answers; see its [`report`](rope_resource_qa_v1.report.json) |
| Manually reviewed resource facts | 30 | [`rope_resource_factual_qa_v1.jsonl`](rope_resource_factual_qa_v1.jsonl), with evidence and reviewer provenance summarized in its [`report`](rope_resource_factual_qa_v1.report.json) |
| Additional manually reviewed resource facts | 10 | [`rope_resource_manual_v1.jsonl`](rope_resource_manual_v1.jsonl), audited against live Crash Restraint, Knot Head Nylon, and Shibari Study evidence in [`manual_reviews/resources/additions_audit_v1.jsonl`](manual_reviews/resources/additions_audit_v1.jsonl) |
| Rope-topia resource index | 15 | [`rope_topia_manual_v1.jsonl`](rope_topia_manual_v1.jsonl) contains manually reviewed title-to-canonical-URL mappings only because the live article bodies are demo-gated; see its [`report`](rope_topia_manual_v1.report.json). |
| **Curated v1 total** | **784** | [`train_qa_curated_v1.jsonl`](train_qa_curated_v1.jsonl) and [`report`](train_qa_curated_v1.report.json) |

The fact-ID-keyed general
[`curation decisions`](train_qa_curated_v1.curation.jsonl) and complete
[`Kinbaku Today audit`](train_qa_kinbakutoday.curation.jsonl) contain 2,548
reviewed actions: 2,500 drops and 48 evidence-backed edits. The source-grouped
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
The complete stricter Rope365, Esinem, Wikipedia, Anatomie Studio, and resource
second passes are promoted in the current artifact. The deterministic
[`S5 promotion manifest`](train_qa_curated_v1.promotion_s5.json) records 602
unique appended decisions, seven replacements, and two identical cross-ledger
confirmations. The result retains 278 Rope365, 77 Esinem, 29 Wikipedia, 23
Anatomie Studio, and 82 resource-derived rows. The subsequent strict Kinbaku
quality pass manually reviewed 316 rows from 125 documents and removed 124,
leaving 299 Kinbaku rows and 784 rows overall. Its production promotion is
pinned in
[`train_qa_curated_v1.promotion_s5_quality.json`](train_qa_curated_v1.promotion_s5_quality.json).
Another 107 rows were flagged only because they overlap the obsolete S4
development benchmark; they were deliberately not removed for that reason.

The user's malformed, contextless Anatomie Studio “What material...?” / `jute`
example has zero matches in the active curated dataset. The second-pass report
records its legacy fact ID and locations so it cannot be mistaken for a current
training row or reintroduced by the pending edits.

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

## Legacy training defaults

[`sft_lora.py`](../sft_lora.py) defaults to
`data/train_qa_curated_v1.jsonl`. [`es_train_acc.py`](../es_train_acc.py) keeps
`data/train_qa_verified_leakfree_v2.jsonl` as its legacy default so recorded ES
runs remain reproducible. These defaults do not authorize either artifact for
the `plan.md` program, and EGGROLL-ES is halted. The historical explicit
selection was:

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
optional. The retired curated-QA builder deliberately has no no-argument
production mode: a rebuild would require the exact opaque collision manifest and
its independently pinned SHA-256 from an approved collision-authority record.
Do not calculate `OPAQUE_COLLISION_AUTHORIZATION_SHA256` from the manifest in
the same command that consumes it, because that would not preserve the sealed
content-addressed boundary.

The two legacy evaluation inputs that once supplied that collision boundary
are under an irreversible no-reopen quarantine after an earlier access. They
must not be resolved, statted, hashed, counted, or reopened to mint a
replacement authorization. Consequently the command below documents the
fail-closed interface for a future independently authorized manifest; it is
not the rebuild path for current training. Current training uses the
source-split V440 projections above.

Once the protected collision runner has published those two pinned values, the
reproducible production command is:

```bash
OPAQUE_COLLISION_AUTHORIZATION=data/train_qa_curated_v1.opaque_collision_authorization_v1.json
OPAQUE_COLLISION_AUTHORIZATION_SHA256=<sha256-pinned-by-collision-authority>
.venv/bin/python build_curated_qa.py \
  --collision-authorization "$OPAQUE_COLLISION_AUTHORIZATION" \
  --collision-authorization-sha256 "$OPAQUE_COLLISION_AUTHORIZATION_SHA256"
```

The manifest schema permits only fixed schema metadata, ordered input hashes,
an aggregate candidate identity hash/count, an aggregate evaluation identity
hash/count, and the aggregate collision count. It contains no protected text,
questions, answers, URLs, source paths, individual identities, or per-item
metrics. The builder verifies canonical JSON, the independently supplied
manifest hash, every ordered training and curation input hash, the completed
candidate identity, and a zero collision count before writing the output. A
missing, changed, stale, noncanonical, or collision-positive authorization
fails closed. Authorization symlinks, parent aliases, and hard links are
forbidden. Resolved output and report paths must also be disjoint from each
other, every input, every curation ledger, and the authorization itself.
Before opening any training or curation input, the builder rejects evaluation,
protected, holdout, OOD, terminal, incident, and manual-review paths in both
their lexical and symlink-expanded forms. Input and curation symlink aliases
are not permitted.

`--synthetic-empty-eval` exists only for unit/regression fixtures. It rejects
all paths inside this repository, including the canonical defaults, so it
cannot be used to certify or overwrite a production artifact. Synthetic tests
provide explicit temporary inputs, output, report, and an empty curation list;
they never open a real evaluation source. Direct `--eval` loading is not a
production CLI option. The retained internal evaluation-fact helper accepts
only explicitly named synthetic fixtures outside the repository and rejects
symlinks before calling its loader.

The other deterministic builders remain independently runnable as needed:

```bash
.venv/bin/python collect_resource_corpus.py --disable-youtube
.venv/bin/python build_resource_coverage_report.py
.venv/bin/python build_manual_qa.py
.venv/bin/python build_resource_qa.py
.venv/bin/python build_resource_facts.py
```

The corresponding entry points are
[`collect_resource_corpus.py`](../collect_resource_corpus.py),
[`build_resource_coverage_report.py`](../build_resource_coverage_report.py),
[`build_resource_qa.py`](../build_resource_qa.py),
[`build_resource_facts.py`](../build_resource_facts.py), and
[`build_curated_qa.py`](../build_curated_qa.py).
