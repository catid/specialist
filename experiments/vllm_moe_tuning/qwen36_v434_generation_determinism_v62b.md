# Qwen3.6 V434 repeated-generation determinism probe (V62B)

## Scope and safety

This is a data-free runtime probe. It uses 68 generated synthetic prompts and
persists only generated-token counts and SHA-256 hashes. It opens no training,
validation, OOD, shadow, or terminal-holdout rows; persists no prompt,
reference-answer, or generation text; and performs no adapter update or HPO.

The common projection is Qwen3.6-35B-A3B, V434 rank-32 LoRA, vLLM 0.25.0,
BF16, TP1, synchronous FCFS scheduling, `VLLM_BATCH_INVARIANT=0`, greedy
temperature-zero decoding, 68 requests, and 64 generated tokens. Every actor
receives one unscored full request call and then four identical recorded calls
unless stated otherwise.

Probe implementation SHA-256:
`8dfaa169b3bfff8430d5efd60625d43dee26872693bee9aaa1452ffc0ec0a70a`.

## Primary four-actor comparison

| Runtime | Changed rows within each actor across four identical calls | Mean changed rows | Cross-actor differing rows by call | Mean steady seconds |
|---|---:|---:|---:|---:|
| Eager, max 68 | 15, 15, 16, 13 | 14.75 | 13, 17, 16, 13 | 22.9604 |
| CUDA graphs, max 68 | 15, 19, 22, 21 | 19.25 | 18, 21, 20, 19 | 11.5789 |

CUDA graphs reduced the measured post-load probe time by 49.6%, but increased
the mean number of within-actor changed rows by 30.5%. Neither runtime was
repeatable. Eager remains the V62B calibration identity because it matches the
earlier evaluator and showed less drift; graph mode requires its own separately
sealed calibration before it can be used for HPO comparisons.

Eager receipt file SHA-256 values, actors 0 through 3:

- `30f5237eb29e91257d3982e8b5d216d815e071ff993e94bed083503bd48128fa`
- `2f21dc924575102a3e539384c3c4d8256ca9ac894b2c58071f6fb7cf0099aef5`
- `2703a26f32aa96bbaed9a3de4a0285a113d41b337529ae53f6426d90e6c49790`
- `b18eb0faea1b6d0cef5a8a9d14b24241ed8333acaf3835b55e6a870e8428c536`

Graph receipt file SHA-256 values, actors 0 through 3:

- `4f1601fc93375442fa5ac841cf989c55ae124316c6e06594f23571bd1828e457`
- `cce40b0b62232553ddba12e4d0e10017f1758d8a32a3fa02f5434c3c8508fb40`
- `69205c2084d734ab45f694cf997de17b9ceb7c342ac88e0a241af8cbb01ae4c1`
- `02df94990b4ade94b09f08a3cb68843b7083154d420bbf4b2456edad110a37e0`

## Bounded follow-up controls

Lowering graph-mode `max_num_seqs` to 1, 2, 4, and 8 changed 21, 18, 14,
and 15 rows, respectively. Corresponding post-load times were 195.57, 109.88,
61.61, and 40.62 seconds. Eager limits of 2, 4, and 8 changed 18, 24, and 14
rows while taking 547.94, 280.26, and 152.68 seconds. Even one active sequence
therefore did not remove the drift, and low concurrency imposed a severe
throughput penalty.

Four unscored eager warmup calls followed by the four recorded calls changed 14
and 15 rows in two runs, still inside the original eager range of 13--16. Four
graph warmups changed 20 rows. Warmup is useful for counterbalancing startup
effects but is not treated as a determinism fix.

`torch.use_deterministic_algorithms(True)` still changed 19 rows in eager mode.
Graph mode failed closed during Inductor benchmarking because that benchmarking
path is not vetted for deterministic mode. Installed vLLM batch-invariant mode
also fails closed for the model's GDN attention backend. These safeguards were
not patched around.

A later four-actor run embedded and verified
`CUDA_DEVICE_MAX_CONNECTIONS=1`; it changed 15, 14, 18, and 14 rows (mean
15.25) in a mean 22.59 seconds. A four-actor run with embedded
`CUDA_LAUNCH_BLOCKING=1` changed 17, 16, 12, and 15 rows (mean 15.0) while
increasing mean post-load time to 34.46 seconds. Neither control improved on
the ordinary eager mean of 14.75, so neither is promoted into the evaluator
runtime identity.

## Decision

Greedy token generation is empirically nondeterministic on this exact
Qwen3.6-MoE + vLLM + LoRA path. The evaluator must treat this as measurement
noise: retain four actors, counterbalance label order, average all fixed paired
replicas within each conflict unit, bootstrap conflict units only, and fail
closed on the unchanged confidence-width, zero-containment, and actor-influence
gates. No threshold is relaxed and no favorable repetition may be selected or
dropped after observing its result.
