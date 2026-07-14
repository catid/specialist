# S4 seed-45 validation robustness replicate

This is a third independent-training-seed replicate of the selected S4
setting on the same frozen 1,487-row Arrow dataset. It used
Qwen3.6-35B-A3B, `sigma=0.001`, `alpha=0.00025`, population 8, batch and
mini-batch size 64, 32 generated tokens, and six exact ES updates. The
training seed was 45. Only the 236-item `train` validation split was
evaluated; baseline evaluation was skipped during the run, the held-out split
was never opened, and no model checkpoint was serialized.

The required pre-run and post-run gates matched: the frozen and active
training Arrow both had SHA-256
`ff1b07297f404249adca6000acf8360000017cb2a75412b49f6e234a8082cc7c`,
and the executed trainer had SHA-256
`9c5ca268d74808e6d67b8b11f4e2e53ff45e60abfef99400c5ae24d0d0b419b6`.

| Validation result | Retained S4 baseline | Seed-45 treatment | Change |
| --- | ---: | ---: | ---: |
| Mean reward | 0.0976728523 | 0.0970984390 | -0.0005744133 (-0.59%) |
| Exact matches | 14 | 14 | 0 |
| Nonzero rewards | 93 | 90 | -3 |

The paired comparison has 11 treatment wins, 14 baseline wins, and 211 ties.
Its paired bootstrap 95% interval for treatment minus baseline is
[-0.0133208999, +0.0117376060], which includes zero. Seed 45 therefore
essentially matched the baseline and did not independently replicate seed
42's validation improvement. Along with seeds 43 and 44, this reinforces that
the six-step recipe is seed-sensitive. Validation also has known document
overlap with training and was used to select the setting and horizon in seed
42.

The one-second GPU trace contains 122 samples from startup through cleanup.
Whole-trace mean utilization was 63.58%, 31.57%, 33.46%, and 32.39% for GPUs
0–3. All four GPUs were at least 20% active together for 41 samples and
exactly 100% active together for 22. Peak memory was 96,639 MiB on GPU 0 and
82,346 MiB on GPUs 1–3. Centralized model updates and evaluation account for
the expected intervals where GPU 0 was active and the other engines waited.

| Artifact | SHA-256 |
| --- | --- |
| Compact run summary | `7d47d595700343fd30545e995f48eba8fa1f7d78fa406c676f81b551dcacd8b4` |
| Self-contained paired comparison | `7e6405d85b31623eab7c7dde16b246f5fff560c6bd56829740c1b602dd0490e4` |
| Raw treatment evaluation (ignored) | `d4e02e55a0ac22527166814cef940323da7103c6b064c527a04c695736f1fbf9` |
| Retained baseline evaluation (ignored) | `f82b3a373c91108a69324bf80cd486cb4aa557c492c47423b4ad3dd1b8ab975c` |
| GPU trace | `81fb2342dd07fed8b846a1ce012e71c4988d734f54ed5bb9b50763e588ea87de` |

The tracked [`run_summary.json`](run_summary.json) records the exact setting
and score. The tracked
[`validation_comparison.json`](validation_comparison.json) embeds aligned
per-example rewards and exactness under prompt/answer hashes, so its means,
wins, ties, and interval remain auditable without retaining raw generations.
The telemetry is
[`../../gpu_utilization_eggroll_es_s4_seed45.jsonl`](../../gpu_utilization_eggroll_es_s4_seed45.jsonl).
