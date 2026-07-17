# Qwen3.6 mirrored LoRA-ES phase profile V66c

Date: 2026-07-17 UTC

Status: fail-closed diagnostic evidence. V66c completed its substantive
mirrored population and nonzero-update/abort protocol, but publication was
withheld because the 500 ms NVML monitor missed GPU 0's sub-second wave-1
generation phase. The failure receipt, population, update, and telemetry are
immutable. This is not a promoted training result.

## Substantive result and final gate

The run evaluated 16 signed candidates (eight antithetic directions) in four
four-actor waves. All eight pair differences were nonzero. The compiled update
had coefficient L2 norm `0.0042338077269223435`; its FP32 candidate identity and
BF16 runtime identity both differed from the canonical master. The update was
not committed. All four actors then reconstructed the exact master, returned
consistent abort receipts, opened no protected/dev/OOD data, wrote no
checkpoint, and cleaned all four GPUs back to empty compute-process lists.

The final telemetry validator correctly failed: wave 1 advanced through
generation before a 500 ms monitor tick, leaving GPU 0 with no row under that
exact phase label. Adjacent restore samples showed device activity, and all
state-specific outputs/rewards were complete, but neither is a valid substitute
for phase-attributed actor work. V66d must add an all-device sampling handshake
and actor-side CUDA-event/output receipts rather than relaxing this gate.

## Phase timing and memory

The table uses the first sample of the next phase as the approximate end of the
current phase. Resolution is therefore about 0.5 seconds. Generation windows
that finished between ticks cannot be timed from this artifact.

| Phase | Approximate controller window | Observed GPU memory | Activity interpretation |
|---|---:|---:|---|
| Activate staged LoRA on four actors | 1.0 s | 83,212 MiB | mostly CPU/runtime control |
| Install and audit canonical V434 master | 10.6 s | 83,212 MiB | 9/84 resident samples positive; peak 4% |
| Candidate materialization, each of four waves | 4.0-4.5 s | 83,212-84,138 MiB | mostly 0%; peak 1-4% |
| Generation | less than one monitor interval in two waves | 84,138 MiB | attribution incomplete; final gate failed |
| Exact restore, each of four waves | 3.0-3.5 s | 84,138 MiB | some lagged/restore GPU activity |
| Update prepare | 1.5 s | 84,138 MiB | mostly CPU; peak 3% |
| Nonzero update execute | 4.0 s | 84,138 MiB | peak 34%, only 4/32 samples positive |
| Exact update abort/master reconstruction | 8.1 s | 84,138 MiB | mostly CPU/audit; peak 3% |

The candidate/update plateau was 926 MiB above setup/install. Model residency
began before this monitor started; total failed-run residency was 115.176
seconds per GPU, or 460.703 charged GPU-seconds. The log has no PCIe counters,
so low device utilization plus timing is evidence of transition overhead, not a
measured H2D/D2H byte rate.

## Code-path audit and optimization implications

The transition cost is much larger than the serialized adapter size would
suggest. The current path performs these operations for every candidate:

1. Generate each keyed FP32 perturbation tensor on GPU and form a candidate.
2. Copy every candidate tensor back to CPU and hash the FP32 identity.
3. Reset runtime LoRA modules, convert each logical tensor/view to BF16, and
   copy each view to its GPU slot separately.
4. Copy each view back to CPU for `torch.equal`, then read/hash the GPU view
   again for the materialization certificate.
5. Synchronize and repeat a similar exact-write/audit path for every restore.

The immutable-master and exact-audit guarantees are valuable; the per-view
round trips and synchronization are not. The highest-confidence optimization
is a preallocated single-slot transaction that regenerates seeded noise into a
bounded GPU workspace, writes runtime views in batches, and emits one batched
exact audit receipt. A final full audit and poison-on-uncertain-restore behavior
must remain. Candidate and audit buffers should be capacity-checked before
launch, pinned transfers should be overlapped where a host copy is unavoidable,
and the immutable master identity should not be cryptographically rehashed at
every internal check unless its tensor version/storage metadata changed.

This evidence does not yet justify a fused implementation: it lacks paired
before/after correctness, PCIe bytes, and live V66d receipts. It does establish
that materialize/restore/audit—not the sub-second scoring generation—is the
dominant measured loop overhead for this small calibration.
