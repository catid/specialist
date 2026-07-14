# S4 seed-46 validation robustness replicate

This is a fourth independent-training-seed replicate of the selected S4
setting on the same frozen 1,487-row Arrow dataset. It used
Qwen3.6-35B-A3B, `sigma=0.001`, `alpha=0.00025`, population 8, batch and
mini-batch size 64, 32 generated tokens, and six exact ES updates. The
training seed was 46. Only the 236-item `train` validation split was
evaluated; baseline evaluation was skipped during the run, the held-out split
was never opened, and no model checkpoint was serialized.

The required pre-run and post-run gates matched: the frozen and active
training Arrow both had SHA-256
`ff1b07297f404249adca6000acf8360000017cb2a75412b49f6e234a8082cc7c`,
and the executed trainer had SHA-256
`9c5ca268d74808e6d67b8b11f4e2e53ff45e60abfef99400c5ae24d0d0b419b6`.

| Validation result | Retained S4 baseline | Seed-46 treatment | Change |
| --- | ---: | ---: | ---: |
| Mean reward | 0.0976728523 | 0.0970728458 | -0.0006000065 (-0.61%) |
| Exact matches | 14 | 14 | 0 |
| Nonzero rewards | 93 | 91 | -2 |

The paired comparison has 13 treatment wins, 17 baseline wins, and 206 ties.
Its paired bootstrap 95% interval for treatment minus baseline is
[-0.0133161304, +0.0116500335], which includes zero. Seed 46 therefore
essentially matched the baseline and did not independently replicate seed
42's validation improvement. Along with seeds 43–45, this reinforces that the
six-step recipe is seed-sensitive. Validation also has known document overlap
with training and was used to select the setting and horizon in seed 42.

The one-second GPU trace contains 123 samples from startup through cleanup.
Whole-trace mean utilization was 63.11%, 33.98%, 31.77%, and 34.85% for GPUs
0–3. All four GPUs were at least 20% active together for 43 samples and
exactly 100% active together for 21. Peak memory was 96,641 MiB on GPU 0 and
82,346 MiB on GPUs 1–3. Centralized model updates and evaluation account for
the expected intervals where GPU 0 was active and the other engines waited.

| Artifact | SHA-256 |
| --- | --- |
| Compact run summary | `57b1cfa60b34de535849ce0f335bcdeab1414ea6c752d88980c9546335ad4fef` |
| Self-contained paired comparison | `2ba61608075a2fa40e137713a3b22655faa4b3193fb9389a720d6f1feaa1b4a2` |
| Raw treatment evaluation (ignored) | `09f1d406eab2f513f08644eb4247ed0d98037b586c39edd31e0de666d54f59c0` |
| Retained baseline evaluation (ignored) | `f82b3a373c91108a69324bf80cd486cb4aa557c492c47423b4ad3dd1b8ab975c` |
| GPU trace | `5b14a24a8da8d63a790c226dda1a5436df5a527ceea2e6bece6ee0d384fce19a` |

The tracked [`run_summary.json`](run_summary.json) records the exact setting
and score. The tracked
[`validation_comparison.json`](validation_comparison.json) embeds aligned
per-example rewards and exactness under prompt/answer hashes, so its means,
wins, ties, and interval remain auditable without retaining raw generations.
The telemetry is
[`../../gpu_utilization_eggroll_es_s4_seed46.jsonl`](../../gpu_utilization_eggroll_es_s4_seed46.jsonl).
