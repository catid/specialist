# Dataset status

## Valid

- `train_qa_verified_leakfree_v2.jsonl`: 3,113 unique, leakage-gated facts.
- `eval_qa*.jsonl`: development evaluations only; they are not independent
  final test sets.

## Invalidated and intentionally not tracked

The prior `train_qa_v3_clean_leakfree_v2.jsonl` contained 55,773 rows, but
28,017 `qa_chat` rows had ChatML/`<think>` delimiters leaked into their parsed
question and answer metadata. Correct structural parsing collapses paired raw
and chat renderings and projects 27,756 unique survivors before manual review.

That generated file must not be used for training. A source-grounded manual
curation pass is replacing it: reviewers receive small packets grouped by
source document, make explicit keep/edit/drop decisions, and add better
self-contained Q&A with source evidence.
