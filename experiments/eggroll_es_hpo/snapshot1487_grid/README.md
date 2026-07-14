# S4 fresh three-step grid

The inherited S3 winner failed on S4, so this grid reevaluated all six fixed
hyperparameter candidates on the same hash-guarded 1,487-row training Arrow.
The holdout was not used.

| Sigma | Alpha | Validation reward | Delta vs baseline |
| ---: | ---: | ---: | ---: |
| 0 (baseline) | 0 | 0.0976728523 | — |
| 0.0003 | 0.00015 | 0.0998501608 | +0.0021773085 |
| 0.0005 | 0.00025 | 0.0979130834 | +0.0002402311 |
| **0.001** | **0.00025** | **0.1031286700** | **+0.0054558177 (+5.59%)** |
| 0.001 | 0.0005 | 0.0980506680 | +0.0003778157 |
| 0.001 | 0.001 | 0.1007300357 | +0.0030571834 |
| 0.002 | 0.001 | 0.0946905038 | -0.0029823486 |

The selected treatment won 12 validation items, lost 10, and tied 214. Exact
matches changed 14 to 15, while nonzero rewards changed 93 to 92. The paired
bootstrap 95% interval is [-0.0008249700, 0.0153114807], so this remains a
promising one-seed result rather than conclusive evidence. Validation also has
known Rope365 document-level overlap with training; the holdout does not
have that overlap in the manually checked tranches.

| Artifact | SHA-256 |
| --- | --- |
| HPO journal | `7aaf26f3da240451836e90113bc1fbbe23b8a83aed88f96b590d479e18c6aa41` |
| Self-contained paired comparison | `6ba375f9aaa939581f50720e9c730533b5b018b77c77ec4cbf14df0f398ecad5` |
| Training Arrow | `ff1b07297f404249adca6000acf8360000017cb2a75412b49f6e234a8082cc7c` |
| Validation Arrow | `3b8b980f4be5060ade3671a53dde03a975382c63ab1e44f0133a19cd741d06cb` |
| Dataset manifest | `8e41b1b4c239035d098ca245735457898da97a768269fadb3733cda3c5c11360` |
| GPU trace | `cdc4a1970b47d52cc1fbe568f082f8ffc9fa1d22702edf24079f4bcf48a07c5c` |

The 571-sample GPU trace is
[`../../gpu_utilization_eggroll_es_s4_grid.jsonl`](../../gpu_utilization_eggroll_es_s4_grid.jsonl).
All four GPUs were at least 20% active together for 124 samples and exactly
100% for 62; peak memory was 96,639 MiB, 82,494 MiB, 84,076 MiB, and 83,106
MiB. Whole-trace means include startup/evaluation/inter-trial idle time and were
44.74%, 22.95%, 23.02%, and 23.25%.
