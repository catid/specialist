# Rope Connections secondary-column manual shard

This additive public-training shard reviews all generated candidates for one
Rope Connections secondary-column article. Its complete source text remains
separate from the manually curated Q&A records.

The shard is not merged into `train_qa_curated_v1.jsonl`. Promotion remains
`pending_semantic_leakage_audit`: this background lane did not read any
development, holdout, OOD, terminal, or other protected evaluation content.
The authorized snapshot owner can perform semantic collision review and merge
the shard between experiments.

The verbatim source includes a non-medical blog warning about possible injury.
That warning is preserved for source fidelity but excluded from the Q&A output;
promoting the raw Markdown itself therefore also requires the downstream safety
review recorded in `report.json`.

Files:

- `source.md`: complete source text from the two matching public training
  chunks.
- `candidate_review.jsonl`: decisions consuming all 37 candidate IDs exactly
  once, plus eight manually added technical facts.
- `qa.jsonl`: thirteen canonical, extractively grounded Q&A records.
- `report.json`: provenance, counts, hashes, and promotion boundary.
- `test_shard.py`: exact-path validation without evaluation-data access.
