# Rope Connections rope-dyeing manual shard

This additive public-training shard reviews every generated candidate for one
Rope Connections first-person rope-dyeing account. The complete source remains
separate from the curated Q&A.

The shard is not merged into `train_qa_curated_v1.jsonl`. Promotion remains
`pending_semantic_leakage_audit`: this background lane did not read any
development, holdout, OOD, terminal, or other protected evaluation content.

The verbatim source contains explicit imagery and profanity, vendor and price
mentions, and personal chemical, boiling, wet-treatment, stretching, and rope
conditioning practices. These are not laboratory-validated instructions.
Shopping trivia and the unqualified boiling claim are excluded from Q&A; all
retained process observations are source-attributed. The raw Markdown and Q&A
both require downstream material/content/safety review before promotion.

Files:

- `source.md`: complete source text from three matching public training chunks.
- `candidate_review.jsonl`: decisions consuming all 40 candidates exactly once,
  plus four manually added process facts.
- `qa.jsonl`: nine canonical, extractively grounded Q&A records.
- `report.json`: provenance, counts, hashes, and promotion boundary.
- `test_shard.py`: exact-path validation without evaluation-data access.
