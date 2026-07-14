# S4 seed-44 validation robustness replicate

This is a second independent-training-seed replicate of the selected S4
setting on the same frozen 1,487-row Arrow dataset. It used
Qwen3.6-35B-A3B, `sigma=0.001`, `alpha=0.00025`, population 8, batch and
mini-batch size 64, 32 generated tokens, and six exact ES updates. The
training seed was 44. Only the 236-item `train` validation split was
evaluated; baseline evaluation was skipped during the run, the held-out split
was never opened, and no model checkpoint was serialized.

The required pre-run and post-run gates matched: the frozen and active
training Arrow both had SHA-256
`ff1b07297f404249adca6000acf8360000017cb2a75412b49f6e234a8082cc7c`,
and the executed trainer had SHA-256
`9c5ca268d74808e6d67b8b11f4e2e53ff45e60abfef99400c5ae24d0d0b419b6`.

| Validation result | Retained S4 baseline | Seed-44 treatment | Change |
| --- | ---: | ---: | ---: |
| Mean reward | 0.0976728523 | 0.0979182133 | +0.0002453609 (+0.25%) |
| Exact matches | 14 | 14 | 0 |
| Nonzero rewards | 93 | 93 | 0 |

The paired comparison has 17 treatment wins, 13 baseline wins, and 206 ties.
Its paired bootstrap 95% interval for treatment minus baseline is
[-0.0123886265, +0.0125775175], which includes zero. Seed 44 therefore
essentially matched the baseline and did not independently replicate seed
42's validation improvement. Together with seed 43's lower score, the result
reinforces that the six-step recipe is seed-sensitive. Validation also has
known document overlap with training and was used to select the setting and
horizon in seed 42.

The one-second GPU trace contains 122 samples from startup through cleanup.
Whole-trace mean utilization was 64.46%, 32.85%, 32.98%, and 33.14% for GPUs
0–3. All four GPUs were at least 20% active together for 42 samples and
exactly 100% active together for 21. Peak memory was 96,639 MiB on GPU 0 and
86,482 MiB on GPUs 1–3. Centralized model updates and evaluation account for
the expected intervals where GPU 0 was active and the other engines waited.

| Artifact | SHA-256 |
| --- | --- |
| Compact run summary | `0778f0410d51de58bae915d76051de551df15e609ac601a7ba1f551a008e12b6` |
| Self-contained paired comparison | `e0854926f9f197723702f02ddb51d2ed5ff6ec33e7311ca5b85f46633296d17d` |
| Raw treatment evaluation (ignored) | `9a0f2f53fa3b96ed9c5b5cf9d12d0b1f94d1b2114cc932939ab40f9cb1900e6d` |
| Retained baseline evaluation (ignored) | `f82b3a373c91108a69324bf80cd486cb4aa557c492c47423b4ad3dd1b8ab975c` |
| GPU trace | `596d53bea2d82219759bc8334fddea85b655babc1fa73160becc66e43f11cbfa` |

The tracked [`run_summary.json`](run_summary.json) records the exact setting
and score. The tracked
[`validation_comparison.json`](validation_comparison.json) embeds aligned
per-example rewards and exactness under prompt/answer hashes, so its means,
wins, ties, and interval remain auditable without retaining raw generations.
The telemetry is
[`../../gpu_utilization_eggroll_es_s4_seed44.jsonl`](../../gpu_utilization_eggroll_es_s4_seed44.jsonl).
