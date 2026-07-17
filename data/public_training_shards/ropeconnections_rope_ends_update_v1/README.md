# Rope Connections rope-end update manual shard

This additive public-training shard reviews all generated candidates for the
Rope Connections follow-up on whipped and knotted rope ends. The complete
source text remains separate from the technical Q&A output.

The shard is not merged into `train_qa_curated_v1.jsonl`. Promotion remains
`pending_semantic_leakage_audit`: this background lane did not read any
development, holdout, OOD, terminal, or other protected evaluation content.

The verbatim source includes explicit first-person scene narrative, profanity,
and anecdotal durability observations. Those details are preserved for source
fidelity, excluded from the Q&A except where a narrowly attributed mechanical
observation is necessary, and require downstream content/safety review before
the raw Markdown itself is promoted.

Files:

- `source.md`: complete source text from the two matching public training
  chunks.
- `candidate_review.jsonl`: decisions consuming all 39 candidates exactly
  once, plus five manually added technical facts.
- `qa.jsonl`: eleven canonical, extractively grounded Q&A records.
- `report.json`: provenance, counts, hashes, and promotion boundary.
- `test_shard.py`: exact-path validation without evaluation-data access.
