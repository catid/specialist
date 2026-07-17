# Mirrored ES with common random numbers: V66 CPU evidence

Date: 2026-07-17

Bead: `specialist-0j5.2`

Status: CPU implementation, launch harness, preregistration, and deterministic
regression evidence complete; four-GPU Qwen3.6 execution still required before
the Bead can close.

## Implemented contract

- Eight unique perturbation directions form a signed population of 16.
- Each four-engine wave contains two complete `+epsilon/-epsilon` pairs.
- Signs rotate between ranks across waves. Every rank evaluates four
  candidates: two positive and two negative.
- A pair binds one canonical prompt-block hash, decode-parameter hash,
  evaluation seed, and judge hash. The executor passes the same immutable
  evaluation payload to both signs.
- The update sent to the existing V41A FP32 LoRA worker is
  `learning_rate / (2*N*sigma) * sum((Rplus-Rminus)*epsilon)`.
- Materialization and evaluation handles are submitted as full four-actor
  waves and all handles are drained, even after an error.
- Every actor receives an unconditional exact-master restore after every wave,
  including actors whose candidate RPC completion is uncertain.
- The worker records its candidate transaction before the first runtime-slot
  write. A partial write is repaired from the canonical FP32 master before the
  original error escapes. An unverified repair terminally poisons the actor and
  blocks inherited state operations.

## Focused evidence

Command:

```text
es-at-scale/.venv/bin/python -m pytest -q \
  test_eggroll_es_worker_lora_v41a.py \
  test_lora_es_fused_anchor_runtime_v43i.py \
  test_lora_es_transition_microbenchmark_v51.py \
  test_lora_es_nested_population_v52.py \
  test_lora_es_ranking64_alpha_zero_calibration_v65a.py \
  test_lora_es_ranking64_alpha_zero_calibration_v65b.py \
  test_eggroll_es_mirrored_v66.py
```

Result: `164 passed in 16.88s`.

After adding the launch harness and idempotent uncertain-update abort, the
expanded adjacent suite passed `172 tests in 29.73s`.

The focused V66 coverage proves:

- deterministic keyed FP32 noise and exact elementwise sign negation;
- complete pair identity for prompts, decoding, evaluation seed, and judge;
- exact central-difference coefficient algebra and compatibility with the
  current four-rank V41A sharded update endpoint;
- four complete actors per wave, equal candidate count per actor, and equal
  signs per actor;
- successful idempotent restore after an uncertain candidate RPC;
- repair after an injected partial runtime-slot write; and
- terminal poisoning after an injected restore-write failure.

The sealed V63 and V64 suites intentionally require a fresh process in which
CUDA has not already been initialized. They passed separately:

```text
test_lora_es_v59_vs_v434_robust_confirmation_v63.py: 61 passed in 50.44s
test_lora_es_v59_vs_v434_robust_confirmation_v64.py: 63 passed in 61.94s
```

## Remaining calibration

Do not mark `specialist-0j5.2` complete until a four-GPU Qwen3.6 run records:

- one nonzero pair difference on an authorized train-only prompt block;
- distinct and consensus-matched `+epsilon/-epsilon` candidate identities;
- identical prompt/decode/judge contract identities for both signs;
- useful activity from physical GPUs 0, 1, 2, and 3 during every retained
  mirrored wave;
- exact canonical-master restoration on all four actors; and
- a nonzero compiled pair-difference update receipt accepted by the V41A
  four-rank update path.

No GPU process was launched for this CPU evidence because the four GPUs were
owned by another active experiment.

## Sealed four-GPU calibration handoff

Preregistration:

- Path: `experiments/eggroll_es_hpo/preregistrations/lora_es_mirrored_calibration_v66.json`
- File SHA-256: `968d96af4c4f511eda317f0fbeda21c0cf4fedcc70caa07d2e2d02e3db17d411`
- Content SHA-256: `f706b63befbd9da93cdda6ad9e612bf8fccfeda395e573ae59ff3515f24e8eef`

The CPU-only dry run passed with:

```text
es-at-scale/.venv/bin/python run_lora_es_mirrored_calibration_v66.py \
  --preregistration /home/catid/specialist/experiments/eggroll_es_hpo/preregistrations/lora_es_mirrored_calibration_v66.json \
  --preregistration-sha256 968d96af4c4f511eda317f0fbeda21c0cf4fedcc70caa07d2e2d02e3db17d411 \
  --preregistration-content-sha256 f706b63befbd9da93cdda6ad9e612bf8fccfeda395e573ae59ff3515f24e8eef \
  --dry-run
```

After the current four-GPU owner has fully cleaned up and `nvidia-smi` shows no
compute processes, the exact live command is:

```text
source ~/.bashrc
es-at-scale/.venv/bin/python run_lora_es_mirrored_calibration_v66.py \
  --preregistration /home/catid/specialist/experiments/eggroll_es_hpo/preregistrations/lora_es_mirrored_calibration_v66.json \
  --preregistration-sha256 968d96af4c4f511eda317f0fbeda21c0cf4fedcc70caa07d2e2d02e3db17d411 \
  --preregistration-content-sha256 f706b63befbd9da93cdda6ad9e612bf8fccfeda395e573ae59ff3515f24e8eef \
  --execute
```

Fresh output paths are reserved under
`experiments/eggroll_es_hpo/runs/v66_lora_es_mirrored_crn_qwen36_calibration`.
The sibling attempt path is
`experiments/eggroll_es_hpo/runs/.v66_lora_es_mirrored_crn_qwen36_calibration.attempt.json`.

The live harness fails unless all of these are true:

- the exact Qwen3.6 model seal and every model file pass pre-run hashing;
- physical GPUs 0-3 are exclusive and each hosts one TP1 actor;
- every mirrored wave records positive resident activity on every GPU;
- at least one train-only pair difference and the coefficient L2 are nonzero;
- the distributed FP32 candidate and its BF16 runtime differ from the master;
- all four candidate replicas have identical FP32 and BF16 identities;
- the candidate remains uncommitted and every actor exactly aborts to master;
- the final model bytes still match the pre-run seal; and
- strict actor cleanup returns all four GPUs to idle.
