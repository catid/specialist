# Frozen S4 final dataset snapshot

These are the exact tracked dataset artifacts consumed by the final S4
EGGROLL/ES experiment. The foreground experiment pulled the latest promoted
dataset only between A/B runs, then hash-pinned this snapshot before either arm
started. Pending Rope365, Esinem, and later audit ledgers were not promoted
during the comparison.

| Artifact | SHA-256 |
| --- | --- |
| Training source JSONL | `d9ec2b0fe0a438067b74cb1a168172e0a658d78c17627d4d6e29c1f02dbfef6d` |
| Build report | `3382e6bfef22d8690d7cb697f570db611e8a01696533eb60ec0502181ab60d82` |
| General curation ledger | `3fc033f3a58638567c709f1bba8b94a5eda93d59c36d4d544c2f8c874b9d18ec` |
| Kinbaku Today curation ledger | `8b68caf2a6a4411f9ae26a05759c6ceb6852eaf05c28171b4a11dd93033ebadd` |
| Training Arrow | `ff1b07297f404249adca6000acf8360000017cb2a75412b49f6e234a8082cc7c` |
| Dataset manifest | `8e41b1b4c239035d098ca245735457898da97a768269fadb3733cda3c5c11360` |

The snapshot contains 1,487 training rows. Evaluation remains in the separately
tracked `data/eval_qa_v2.jsonl` (SHA-256
`62e920a786fb7a0da383aa19253ee6a2e9f63ac93d3d40de6ed93c5ea10b9fd7`):
236 validation rows and 169 heldout rows.
