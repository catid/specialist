# Frozen S3 training snapshot

These files are the immutable 2,228-row training cohort used by the final
validation grid, selected retrain, and holdout evaluation. They are retained
separately because `data/train_qa_curated_v1.jsonl` is the moving, manually
curated foreground input for later experiments.

| Artifact | SHA-256 |
| --- | --- |
| `train_qa_curated_v1.jsonl` | `ea178be3d1052000095cde77a5c4b1b8b93130bb3b16fafdd85d0a107a7edf4d` |
| `train_qa_curated_v1.curation.jsonl` | `e6cb9da70241f41dee170dad37b2191a8a48c1945f1c7b72d6e90ba41d3fdb34` |
| `train_qa_curated_v1.report.json` | `5ffdfa6f89748a2a712f7e6fb0c35ae90012677f493d4c1244697a9a9d81d4b3` |
| `eggroll_es_manifest.json` | `68cf1ac18e0f828d7855a4a087a90e85ad329c859d7a2f4d1fc1bd6aa3e5c4a4` |
| `train.arrow` | `bb60372725825f2fc81b46b681899ed8b4ba1af79d10ab1e6905bae5fb660f6f` |
| `train_eggroll_es_specialist.py` | `ebdf5f1be4e9e64af3e234f2359afa571f2322a5eb0724368ae7087ed0a439af` |

The evaluation source remains `data/eval_qa_v2.jsonl`, SHA-256
`62e920a786fb7a0da383aa19253ee6a2e9f63ac93d3d40de6ed93c5ea10b9fd7`.

This directory intentionally preserves executed bytes, including one
Rope-topia row whose curated Q&A target used the correct exact sitemap URL but
whose non-target `evidence_url`/canonical metadata remained stale. The active
builder and S4 source fix that provenance metadata; S3 is not rewritten because
doing so would falsify the experiment hash.

The copied trainer is the exact executed source. An identical root-level copy,
`train_eggroll_es_specialist_s3.py`, is the runnable replay entry point because
the script derives upstream/model paths from its own directory. The moving root
trainer later changed only its seed-zero handling: S3 seed 42 behavior is
unchanged, while a requested seed 0 now remains 0 instead of falling back to
42.
