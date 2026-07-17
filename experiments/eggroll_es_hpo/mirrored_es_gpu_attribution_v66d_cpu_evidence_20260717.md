# Mirrored LoRA-ES V66d GPU-attribution evidence

Date: 2026-07-17 UTC

## Outcome

V66c completed all sixteen signed Qwen3.6-35B-A3B evaluations, all four
mirrored waves, a nonzero pair-difference update candidate, and exact
four-actor abort/master restoration. Its final report was withheld because a
sub-second generation phase ended before the 0.5-second NVML monitor emitted
an exact-label row for GPU 0. The immutable V66c population and update are
inputs to the fresh V66d preregistration; V66c paths are not reused.

V66d keeps the live-proven ES and state protocol unchanged and replaces the
ambiguous activity gate with two independent receipts:

1. A phase transition cannot return until the monitor has written and flushed
   one same-epoch, same-timestamp row for each of physical GPUs 0, 1, 2, and 3.
   A partial device batch cannot acknowledge the transition, so generation
   cannot begin before an exact generation label exists in the durable log.
2. Each signed candidate records a `torch.cuda.Event` before generation and a
   synchronized end event after generation. The worker receipt has positive
   elapsed time, exact engine-rank/worker-PID/physical-GPU binding, and the
   controller-observed cardinality of 64 request outputs, 64 samples, 64
   generated tokens, and positive prompt-token coverage.

NVML utilization and PCIe samples remain recorded. A short generation phase
may have zero positive NVML ticks only when its exact actor CUDA-event/output
receipt is valid. Residency alone, device-wide activity alone, a missing
event, zero output/token coverage, a wrong PID/GPU, or a partial wave fails
closed.

## CPU validation

Focused V66d suite: `25 passed in 1.29s`.

The focused adversarial tests cover:

- phase completion attempted before the first full monitor batch;
- an artifact containing only a partial four-device sample batch;
- a resident actor with zero useful output/token cardinality;
- a missing CUDA event receipt;
- a wrong worker PID or physical GPU;
- a partial four-actor wave;
- a valid sub-sample-interval phase with zero positive NVML ticks;
- worker start/end event order and positive elapsed time;
- generation submission only after phase acknowledgement and actor event start;
- fresh V66d attempt schema/self-hash and restoration of legacy hooks.

The expanded adjacent CPU suite covered V41A, V43I, V51, V52, V65A/V65B,
the V66 mirrored protocol and all retry harnesses, plus V66d telemetry/worker:
`212 passed in 19.71s` with CUDA hidden. `py_compile`, preregistration `--check`,
the no-write dry run, and `git diff --check` also passed.

## Sealed V66d identity

- Preregistration:
  `experiments/eggroll_es_hpo/preregistrations/lora_es_mirrored_calibration_v66d.json`
- File SHA-256:
  `3269f7138d74266538cc3b0f31e1a904808f8f3751dde5a7a9456e93b13314b0`
- Content SHA-256:
  `2f8e23b643507b594b05719966da1b9bcc64a2b7f412021066df4c6418144531`
- Attempt and run paths were confirmed absent after the dry run.
- Protected dev/OOD/holdout access, checkpoint creation, candidate commit, and
  promotion remain unauthorized.

The live command, gated on the root coordinator's exact GPU handoff, is:

```text
es-at-scale/.venv/bin/python run_lora_es_mirrored_calibration_v66d.py \
  --preregistration experiments/eggroll_es_hpo/preregistrations/lora_es_mirrored_calibration_v66d.json \
  --preregistration-sha256 3269f7138d74266538cc3b0f31e1a904808f8f3751dde5a7a9456e93b13314b0 \
  --preregistration-content-sha256 2f8e23b643507b594b05719966da1b9bcc64a2b7f412021066df4c6418144531 \
  --execute
```

## Live result

The root coordinator confirmed all four GPUs idle and issued the exact V66d
handoff. The sealed command completed successfully.

- Report file SHA-256:
  `12a5e854856d28bd8439cf3ed004664317086f8d117ae08e78b59f857f6102bb`
- Report content SHA-256:
  `87d1eca139ee0b766b15517c81459becd0369c9d5f7ffb78269fdfce977de684`
- GPU telemetry file SHA-256:
  `a31d9c4cfe6507ca642c061c14cdb40b8ebe35b6ea81783a2199df2bb3c0e475`
- Actor CUDA-work log SHA-256:
  `aa10617c347b7ce5449165580dd4eaa98bb5131cfde5fcf9cda1134b380390e0`
- Population file SHA-256:
  `9d172d15f82a54c697b8b860ff3131733d59006f1e4b790b5b9b87ded679e9d4`
- Update file SHA-256:
  `f958f90b26c5b2afa4a81b03a0ab91c12d9684c2ce236bbb658d674e7a5eeffd`

All sixteen signed candidates have an exact actor receipt. Every receipt has
64 request outputs, 64 samples, 64 generated tokens, positive prompt-token
coverage, positive CUDA-event elapsed time, and the registered worker PID/GPU.
Every generation wave has one complete acknowledged four-GPU monitor batch;
there were no foreign compute processes. Peak NVML utilization for physical
GPUs 0 through 3 was 74%, 100%, 100%, and 100%, respectively, with 84,138 MiB
peak memory used on each GPU.

All eight mirrored pair differences were nonzero. The coefficient L2 norm was
`0.004587680869466419`; both FP32 candidate identity and BF16 runtime identity
differed from the canonical master. The candidate was not committed. The
final canonical master and runtime hashes exactly matched the pinned values.
The run charged `492.405638704` GPU-seconds, created no checkpoint, opened no
protected dev/OOD/holdout data, and performed no promotion.

Cleanup removed all four placement groups, terminated all four engines, and
proved all four compute-process lists empty. A final independent check showed
4 MiB and 0% utilization on every GPU with no EngineCore process. Offline
revalidation recomputed every population/update/report self-hash and reran the
V66d combined telemetry validator successfully.
