# Frozen S4 training snapshot

These are the exact 1,487-row inputs used by the hash-guarded S4 transfer
probe. The snapshot promotes the complete Kinbaku re-audit and ten additional
manual resource facts, but predates the still-pending Rope365 second pass.

| Artifact | SHA-256 |
| --- | --- |
| `train_qa_curated_v1.jsonl` | `d371114ca4e2dacf7dfb97adb2f669ee8b3f44455354a15015fd2b56ac982d5f` |
| `train_qa_curated_v1.curation.jsonl` | `3fc033f3a58638567c709f1bba8b94a5eda93d59c36d4d544c2f8c874b9d18ec` |
| `train_qa_kinbakutoday.curation.jsonl` | `8b68caf2a6a4411f9ae26a05759c6ceb6852eaf05c28171b4a11dd93033ebadd` |
| `train_qa_curated_v1.report.json` | `88be349256984078b9310f34038c119cded448bdf72085bc33b35e17a7dc2649` |
| `eggroll_es_manifest.json` | `f78314347895a67a1696e9e936c2710de4b7874833b6e4f28378ff4f960e8df2` |
| `train.arrow` | `ff1b07297f404249adca6000acf8360000017cb2a75412b49f6e234a8082cc7c` |

After the probe, the ten promoted resource rows had their `source_lineage`
label changed from the pending-review path to the active artifact path. That
metadata-only cleanup changes the moving JSONL hash but not its Q&A text or the
Arrow hash; this directory retains the exact pre-cleanup bytes used by S4.
