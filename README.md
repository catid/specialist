# Specialist

Qwen3.6-35B-A3B post-training experiments using SGLang-resident evolutionary
strategies, deterministic FP32 master weights, targeted layer plans, and
source-grounded QA curation.

The main technical audit and experiment results are in
[`ES_AT_SCALE_LISA_REPORT.md`](ES_AT_SCALE_LISA_REPORT.md). The hand-reviewed
paper transcription is in
[`references/papers/2509.24372.md`](references/papers/2509.24372.md).

Large model files, generated checkpoints, raw source corpora, and runtime logs
are intentionally excluded. The nested SGLang implementation checkpoint is
stored as `patches/sglang-specialist.bundle` so it can be recovered without
vendoring the full upstream repository.

## Dataset status

[`data/train_qa_curated_v1.jsonl`](data/train_qa_curated_v1.jsonl) is the current
training dataset. Its
[`build report`](data/train_qa_curated_v1.report.json) verifies 3,258 unique
facts: 3,110 accepted base facts, 81 source-document manual facts, 29
owner-curated resource-directory facts, and 38 manually reviewed resource
facts. The final merge excluded three additional distinctive-answer aliases
from the 3,113-row
[`train_qa_verified_leakfree_v2.jsonl`](data/train_qa_verified_leakfree_v2.jsonl)
base. That base remains the default in [`es_train_acc.py`](es_train_acc.py) only
to reproduce the recorded ES runs; [`sft_lora.py`](sft_lora.py) defaults to the
curated dataset. New ES runs can select it with
`--data data/train_qa_curated_v1.jsonl`.

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
