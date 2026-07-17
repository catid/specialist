# Rope Connections partial-suspension manual shard

This directory is an additive, manually reviewed public-training shard. It
reviews all 33 generated candidates for one source document and keeps the raw
source copy separate from the six resulting question-and-answer records.

The shard is intentionally not merged into `train_qa_curated_v1.jsonl`. Its
promotion status is `pending_semantic_leakage_audit`: the background curation
lane did not read any development, holdout, OOD, terminal, or other protected
evaluation content. A foreground snapshot owner can perform the separately
authorized semantic collision check and promote the shard between A/B runs.

Files:

- `source.md`: the complete public source text already present in
  `data/train_chunks_v2.jsonl`, rendered as standalone Markdown.
- `candidate_review.jsonl`: hand-written decisions consuming each of the 33
  candidate fact IDs exactly once, plus three manually added facts.
- `qa.jsonl`: six canonical, source-grounded training examples.
- `report.json`: counts, provenance, hashes, and promotion boundary.
- `test_shard.py`: exact-path validation that never opens evaluation data.
