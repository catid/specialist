# Rope Connections rope-kit manual shard

This additive public-training shard reviews every generated candidate for the
Rope Connections rope-kit article. The full source is kept separately from the
manually curated question-and-answer records.

The shard is not merged into `train_qa_curated_v1.jsonl`. Its promotion status
is `pending_semantic_leakage_audit`: this background lane did not read any
development, holdout, OOD, terminal, or other protected evaluation content.
A foreground snapshot owner can run the separately authorized semantic
collision audit and promote it between experiments.

Files:

- `source.md`: complete source text from the two matching public training
  chunks.
- `candidate_review.jsonl`: decisions consuming all 33 candidate IDs once,
  plus five hand-written additions.
- `qa.jsonl`: ten canonical, extractively grounded Q&A records.
- `report.json`: provenance, review counts, hashes, and promotion boundary.
- `test_shard.py`: exact-path validation with no evaluation-data access.
