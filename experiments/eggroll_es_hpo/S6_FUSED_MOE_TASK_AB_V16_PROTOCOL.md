# S6 V16 fused-MoE task-level A/B

V16 is a train-only systems experiment independent of the failed V15B
architecture confirmation. The retained recipe is V13 middle-late layers
20--23. Both arms run the exact V13 five-panel, 32-direction, alpha-zero
diagnostic on Qwen3.6-35B-A3B with four independent TP=1 engines. No model
update, validation, OOD, heldout, benchmark outcome, or data HPO is allowed.

The default arm binds Triton with `VLLM_TUNED_CONFIG_FOLDER` unset. The tuned
arm binds the exact committed directory and device-specific config with SHA256
`a6fbb265df9527d0024d531a1779dc19ecb416f80c873036d9713a2fdce9df2d`.
Everything else—model, plan, panels, basis, seed, generation, engine topology,
and wave order—is identical. Each arm starts in a fresh process because vLLM
loads tuned configuration at initialization.

## Exact equivalence

Promotion requires exact equality of the complete diagnostic content hash,
the dense scored-output manifest hash, the compact task-output hash, and the
compact estimator. Raw outputs, response vectors, rows, prompts, and answers
are never persisted. Every V13 restoration, base-probe, population-boundary,
hardware-coverage, and alpha-zero integrity audit must pass in both arms.

## Generation-only timing

After engine initialization, model load, graph capture, and one unmeasured
combined-five-panel reference warmup, the runtime times only the blocking
four-engine generation resolve for each of the 16 signed waves using
`time.perf_counter_ns`. Perturbation, restoration, scoring, analysis, cleanup,
JIT warmup, initialization, and loading are outside the boundary. Wave order
is the frozen V13 basis order with plus then minus in both arms.

All timing rules are conjunctive:

1. summed generation-call time speedup must be at least `1.05`;
2. median of the 16 paired-wave speedups must be at least `1.05`; and
3. at least 14 of 16 paired waves must be nonregressive (`>=1.0`).

The sealed synthetic evidence measured `1.1705x` median and `1.1647x` worst-GPU
speedup. A 5% task-level floor is a conservative but material fraction of that
gain under task-shape shift. Shapes, repetitions, thresholds, and endpoints
cannot be selected after the run.

Passing authorizes only opt-in use of the tuned directory in a later,
separately preregistered training experiment. It does not authorize a model
update or any evaluation. Failure keeps tuned configuration disabled and
retains default-Triton V13.

This preregistration commit does not authorize GPU launch. A separately
committed fail-closed runtime bundle and exact dry hashes are required first.
