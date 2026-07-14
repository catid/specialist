# Specialist

Qwen3.6-35B-A3B post-training experiments using SGLang-resident evolutionary
strategies, deterministic FP32 master weights, targeted layer plans, and
source-grounded QA curation.

The main technical audit and experiment results are in
[`ES_AT_SCALE_LISA_REPORT.md`](ES_AT_SCALE_LISA_REPORT.md). The hand-reviewed
paper transcription is in
[`references/papers/2509.24372.md`](references/papers/2509.24372.md).
The latest reproducible layer-location artifacts and exact recipe are in
[`experiments/es_deterministic/README.md`](experiments/es_deterministic/README.md).
The direct EGGROLL/ES-at-Scale replication, pinned upstream submodule, Qwen3.6
adapter, and four-GPU launch recipe are documented in
[`EGGROLL_ES_REPLICATION.md`](EGGROLL_ES_REPLICATION.md).
The cleaned-dataset HPO, final checkpoint recipe, holdout comparison, and GPU
telemetry are in
[`experiments/eggroll_es_hpo/README.md`](experiments/eggroll_es_hpo/README.md).
On the frozen 1,487-row S4 snapshot, the selected six-update EGGROLL treatment
raised validation reward from 0.097673 to 0.104408 and S4 holdout reward
from 0.067436 to 0.074038. The holdout interval still includes zero, so this is
an encouraging one-seed result rather than a conclusive improvement. A later
seed-43 validation replicate scored below baseline and also crossed zero,
showing that the short recipe is seed-sensitive.

Large model files, generated checkpoints, raw source corpora, and server logs
are intentionally excluded. Compact deterministic experiment logs are retained
with their journals so summary artifacts can be regenerated. The nested SGLang
implementation checkpoint is stored as `patches/sglang-specialist.bundle` so it
can be recovered without vendoring the full upstream repository.

## Dataset status

[`data/train_qa_curated_v1.jsonl`](data/train_qa_curated_v1.jsonl) is the current
training dataset. Its
[`build report`](data/train_qa_curated_v1.report.json) verifies 1,487 unique
facts: 1,314 accepted base facts, 81 source-document manual facts, 29
owner-curated resource-directory facts, 48 manually reviewed resource facts,
and 15 manually reviewed Rope-topia resource-index facts. The merge applies
1,827 fact-ID-keyed decisions from the general
[`curation ledger`](data/train_qa_curated_v1.curation.jsonl) and the complete
[`Kinbaku audit`](data/train_qa_kinbakutoday.curation.jsonl)—1,797 drops and 30
source-evidenced edits—and excludes two additional
distinctive-answer aliases from the 3,113-row
[`train_qa_verified_leakfree_v2.jsonl`](data/train_qa_verified_leakfree_v2.jsonl)
base. It also canonicalizes every training `text` field from validated Q&A so
legacy generator instructions cannot leak into training. That base remains the
default in [`es_train_acc.py`](es_train_acc.py) only
to reproduce the recorded ES runs; [`sft_lora.py`](sft_lora.py) defaults to the
curated dataset. New ES runs can select it with
`--data data/train_qa_curated_v1.jsonl`.

The completed 2,228-row S3 EGGROLL A/B cohort remains frozen under
[`experiments/eggroll_es_hpo/snapshots/s3/`](experiments/eggroll_es_hpo/snapshots/s3/).
The completed 1,487-row S4 cohort is likewise frozen under
[`experiments/eggroll_es_hpo/snapshots/s4_final/`](experiments/eggroll_es_hpo/snapshots/s4_final/).
Foreground experiments promote newer curation only between comparisons, then
hash-pin one Arrow snapshot for both arms.

The [`resource manifest`](sources/rope_resources_v1.json) preserves all 23
user-supplied URLs. Its bounded, policy-respecting live collection captured 165
relevant public documents; the
[`coverage report`](data/rope_resources_coverage_v1.json) records every
policy/access challenge and scope exclusion rather than bypassing it. See the
[`source-policy notes`](sources/README.md),
[`manual curation guide`](MANUAL_QA_CURATION.md), and
[`dataset artifact and rebuild guide`](data/README.md) for provenance, reports,
and build commands.

The previous large leak-free artifact remains invalid: a parser-order bug
polluted 28,017 ChatML metadata rows and allowed 270 leakage collisions to
bypass filtering. The structural parser fix and regression tests are included.
The specific malformed Anatomie Studio “What material...?” / `jute` example
has zero matches in the active curated dataset; its legacy locations are
recorded in the pending
[`Anatomie Studio second-pass report`](data/manual_reviews/anatomiestudio_second_pass/report_rows_0001_0023_v1.json).
