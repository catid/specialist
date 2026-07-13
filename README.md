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

`data/train_qa_verified_leakfree_v2.jsonl` is the valid 3,113-fact dataset used
by the recorded ES location experiments. The previous large leak-free artifact
is excluded because a parser-order bug polluted 28,017 ChatML metadata rows and
allowed 270 leakage collisions to bypass filtering. The structural parser fix
is committed, and source-document-level manual curation is in progress before a
replacement large dataset is published.
