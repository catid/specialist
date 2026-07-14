# S4 seed-43 validation robustness replicate

This is an independent-training-seed replicate of the selected S4 setting on
the same frozen 1,487-row Arrow dataset. It used Qwen3.6-35B-A3B, `sigma=0.001`,
`alpha=0.00025`, population 8, batch and mini-batch size 64, 32 generated
tokens, and six exact ES updates. The training seed was 43. Only the 236-item
`train` validation split was evaluated; baseline evaluation was skipped during
the run, the held-out split was never opened, and no model checkpoint was
serialized.

The required pre-run gates matched: the frozen and active training Arrow both
had SHA-256
`ff1b07297f404249adca6000acf8360000017cb2a75412b49f6e234a8082cc7c`,
and the executed trainer had SHA-256
`9c5ca268d74808e6d67b8b11f4e2e53ff45e60abfef99400c5ae24d0d0b419b6`.

| Validation result | Retained S4 baseline | Seed-43 treatment | Change |
| --- | ---: | ---: | ---: |
| Mean reward | 0.0976728523 | 0.0934198433 | -0.0042530091 (-4.35%) |
| Exact matches | 14 | 13 | -1 |
| Nonzero rewards | 93 | 92 | -1 |

The paired comparison has 12 treatment wins, 14 baseline wins, and 210 ties.
Its paired bootstrap 95% interval for treatment minus baseline is
[-0.0144604421, +0.0020124196], which includes zero. Thus this seed did not
replicate seed 42's validation improvement, but it also does not establish a
significant regression. Together, the two seeds show that the six-step result
is seed-sensitive and needs more independent replicates before treating it as
robust. Validation also has known document overlap with training and was used
to select the setting and horizon in seed 42.

The one-second GPU trace contains 124 samples from startup through cleanup.
Whole-trace mean utilization was 62.08%, 32.78%, 33.55%, and 33.88% for GPUs
0–3. All four GPUs were at least 20% active together for 43 samples and exactly
100% active together for 22. Peak memory was 96,639 MiB on GPU 0, 82,346 MiB
on GPUs 1–2, and 81,884 MiB on GPU 3. Centralized model updates and evaluation
account for the expected intervals where GPU 0 was active and the other
engines waited.

| Artifact | SHA-256 |
| --- | --- |
| Compact run summary | `5d00e358c2289df4efe4de61e8d9057e46237c29e7f7df05c23588f158cda32d` |
| Self-contained paired comparison | `207e1ca39826c14b847bc8829145262fceaa6f8d033c08fb9ae83fb843b2a488` |
| Raw treatment evaluation (ignored) | `56b24c2933808011726edc2a865a35fc9ef86f41e0ace6df00e5713818d70bec` |
| Retained baseline evaluation (ignored) | `f82b3a373c91108a69324bf80cd486cb4aa557c492c47423b4ad3dd1b8ab975c` |
| GPU trace | `f2cecf04954e2d731860c85c547740de26e8edc6a2c4e267916b95d5d0ed1ea2` |

The tracked [`run_summary.json`](run_summary.json) records the exact setting
and score. The tracked
[`validation_comparison.json`](validation_comparison.json) embeds aligned
per-example rewards and exactness under prompt/answer hashes, so its means,
wins, ties, and interval remain auditable without retaining raw generations.
The telemetry is
[`../../gpu_utilization_eggroll_es_s4_seed43.jsonl`](../../gpu_utilization_eggroll_es_s4_seed43.jsonl).
