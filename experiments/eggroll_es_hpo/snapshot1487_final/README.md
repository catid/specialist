# S4 final retrain and holdout

The selected S4 setting—`sigma=0.001`, `alpha=0.00025`, population 8, batch
size 64, seed 42, and six ES updates—was retrained on the frozen 1,487-row
Arrow snapshot. It reproduced the selected validation score exactly before the
169-item holdout was evaluated for S4 final reporting. Holdout was not used for
S4 grid or horizon selection.

| Split | Base | Treatment | Delta | Exact (base → treatment) | Nonzero (base → treatment) | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation (`train`) | 0.097673 | 0.104408 | +0.006735 (+6.90%) | 14 → 15 | 93 → 95 | [+0.000186, +0.016840] |
| Holdout | 0.067436 | 0.074038 | +0.006602 (+9.79%) | 7 → 8 | 54 → 56 | [-0.000913, +0.019884] |

Validation had 15 treatment wins, eight baseline wins, and 213 ties. Holdout
had nine treatment wins, ten baseline wins, and 150 ties. The validation
interval is optimistic because both the setting and horizon were selected on
that split, and validation has known document overlap with training. The
holdout was not used for selection and improved numerically, but its interval
includes zero. This is a useful one-seed result, not conclusive evidence of a
general improvement. The metric is normalized answer-token overlap with
exact-match credit, not binary accuracy.

A later validation-only replication with seed 43 scored 0.093420, below the
0.097673 baseline, and its paired interval crossed zero. It did not access
holdout. See [`../snapshot1487_seed43/`](../snapshot1487_seed43/). This failed
replication strengthens the seed-sensitivity limitation on the seed-42 result.

The exact machine-readable result is [`final_results.json`](final_results.json).
The paired comparison files embed aligned per-example rewards and exactness
keyed by prompt/answer hashes, so their means, wins, ties, and intervals remain
auditable without the ignored raw generation arrays.

| Artifact | SHA-256 |
| --- | --- |
| Final run summary | `96749661a47cc04719a238259f6d56d029e22655f577b394fba89912d9df96ae` |
| Validation comparison | `1c3ffb3b12829875e6a5a34e355628445092d611a637326fd43748169f024209` |
| Holdout comparison | `6a995f208ee5235b701d579d3e8763fbfcdeabb8b1abc54412a2a384182b15c5` |
| Training Arrow | `ff1b07297f404249adca6000acf8360000017cb2a75412b49f6e234a8082cc7c` |
| Dataset manifest | `8e41b1b4c239035d098ca245735457898da97a768269fadb3733cda3c5c11360` |
| GPU trace | `1d4297aff43e2a127df48ce94d673ed405e784678b8bbaeb5cb0c1ca2c756811` |

The final checkpoint is stored locally at
`../runs/final_s4_selected/checkpoint-es_exact_steps_6/pytorch_model.pth`. It
is 69,321,427,447 bytes, has SHA-256
`95189c28318ee394b6ed3e7abc8c05554fe0b74b0ff674f900dc260ef76a355a`,
and is intentionally gitignored.

The 190-sample GPU trace is
[`../../gpu_utilization_eggroll_es_s4_final.jsonl`](../../gpu_utilization_eggroll_es_s4_final.jsonl).
All four GPUs were at least 20% active together for 43 samples and exactly 100%
for 23. Whole-trace means—including startup, centralized evaluation/update,
checkpoint serialization, and teardown—were 43.11%, 21.62%, 21.21%, and
21.12%; peak memory was 96,639 MiB on GPU 0 and 86,482 MiB on GPUs 1–3.
