# Qwen3.6 V71/V72 live calibration V73 CPU evidence

Status: the additive live runner, sealed preregistration, and adversarial CPU
tests are ready. No GPU was launched, no protected/dev/OOD/holdout content was
opened, and V66, V66d, `es-at-scale`, V71, and V72 were not edited by this
integration work. The four-GPU execution remains an explicit live gate.

## Dependency-ordered integration

V73 retains the accepted V66d four-TP1 train-only schedule, four-row GPU phase
handshake, actor CUDA-event receipts, exact abort, and cleanup-idle behavior.
It substitutes the V72 worker and inserts the V71/V72 APIs in this order:

1. Verify the immutable accepted V66d population, update, report, actor-work,
   and GPU-log file identities.
2. Complete V72 install and prove one unique 18,112,512-byte host master per
   actor.
3. Run the unchanged 16 signed candidates. V71 withholds every actor work
   receipt until its post-generation exact LoRA audit passes.
4. Require four candidate-audit hashes per actor and 16 unique hashes in all.
5. Accept each actor's four provisional rewards at the exact population
   boundary. The four population tokens are rank-local, not broadcast.
6. Only after all four acceptances, compute the pair-difference update. Prepare
   each actor with its own population token and retain its distinct update
   acceptance token.
7. Execute V72's two-bank candidate, compare candidate and runtime identities
   exactly with V66d, and then exact-abort all four actors without commit.
8. Prove one unique host bank again, collect audit traffic, flush/fdatasync host
   RSS/NUMA/fault telemetry before actor teardown, and retain V66d's final
   four-GPU idle certificate.

Unknown or partially completed prepare/execute waves always enter the
four-actor exact-abort path. An invalid abort receipt supersedes the primary
error rather than allowing an unproved rollback to be hidden.

## Equivalence surface and integration findings

The live run requires exact equality with accepted V66d for the 16 candidate
FP32 identities, 16 runtime BF16 identities, 16 floating-point rewards,
pair-difference coefficients, rank shards, executed update candidate, restore
identities, and actor work IDs/output cardinalities. Worker PID, physical GPU
assignment, CUDA timing, NVML values, RSS, faults, and NUMA placement are
expected to be run-specific and are recorded rather than compared bytewise.

Three controller incompatibilities were resolved explicitly:

- V71 population and update acceptance hashes differ by rank because each
  actor owns four different candidates and a different update shard. Reusing
  V66d's broadcast RPC arguments would be incorrect; V73 uses rank-specific
  calls and rejects collapsed tokens.
- V71's stronger candidate receipt no longer repeats the legacy
  `noise_protocol` and `master_unchanged` compact fields. V73 derives only
  those two fields from the sealed V66 noise constant plus the exact cached
  master receipt; all candidate/runtime hashes remain worker-provided.
- V72 begins with reference generation 1 and the inherited explicit capture
  advances it once, so its manifest metadata hash intentionally differs from
  V66d. Manifest equality is not treated as mathematical update equality. The
  differing manifest is recorded, while coefficients, shards, candidate
  bytes, runtime bytes, and exact abort must match.

## Transfer accounting

After the final state certificate, each actor must report exactly:

- H2D LoRA materialization: 137,797,632 bytes (14 fused materializations).
- D2H fused LoRA audit/materialization: 196,853,760 bytes (20 calls).
- D2H exact frozen-base audits: 857,997,312 bytes (population acceptance,
  update acceptance, and exact abort).
- Validation-only FP32 master copies: zero.
- Counted device-transfer total: 1,192,648,704 bytes per actor and
  4,770,594,816 bytes across four actors.

The worker counter intentionally covers V71/V72 audit and materialization, not
all state-path traffic. V73 separately byte-accounts, per actor, the
285,999,104-byte install bootstrap base readback, 72,450,048 bytes of FP32
candidate-master H2D, 72,450,048 bytes of owned-candidate D2H, and the
18,112,512-byte reduced update delta D2H. This outside-counter subtotal is
449,011,712 bytes per actor. The known code-path total is therefore
1,641,660,416 bytes per actor and 6,566,641,664 bytes across four actors;
generation, allocator traffic, and GPU all-reduce traffic remain named
exclusions.

The immutable accepted V66d measured/code-ledger lower bound contains
18,341,068,800 base-plus-LoRA D2H bytes. V73's exact expected base-plus-LoRA
D2H is 4,219,404,288 bytes. The pre-registered reduction is 14,121,664,512
bytes, or 76.9948%. Timing improvement is not claimed by CPU evidence; the live
run records every phase handshake, controller operation, worker
materialization/restore duration, and actor CUDA duration.

## Host telemetry and cleanup

The controller binds every sample to actor rank, worker PID, and physical GPU.
It samples `VmRSS`, `VmHWM`, anonymous/file/shared RSS, `VmLck`, `VmPin`, thread
count, minor and major faults, plus per-node pages from `/proc/PID/numa_maps`.
Lightweight status/fault samples run every 0.5 seconds; NUMA placement is read
at acknowledged phase changes and named install/update/abort/final boundaries.
Fault counters must remain monotonic. The monitor takes and fsyncs its final
sample before actor cleanup; a telemetry failure converts an otherwise clean
run into a failure only after teardown and GPU-idle handling remain possible.

The logical V72 receipts must show one bank after install, two banks while the
executed candidate is retained, and one bank after exact abort/final
certificate. Physical RSS/NUMA/page-fault values are measurements, not inferred
from the logical byte proof.

## Sealed launch

Preregistration:
`experiments/eggroll_es_hpo/preregistrations/lora_es_v71_v72_live_calibration_v73.json`

- File SHA-256: `320b038f07b615622cab0a2a5a9aec86aa06d7649e794201794f382d8ab3783e`
- Content SHA-256: `8bea32f21d33970f1b234cfe59ebaa3eb60fcd47faca1aff1913d81f5dcfe08c`
- Builder SHA-256: `f8fd420c82d61a2df2f02d1e664224dd8cc91941b1358386aaeacac520d17923`
- Runner SHA-256: `b579ac417d88aa9bf9eead5635d1454045451b6ba9fbe383569f4b2e2530260b`

Exact live command (not executed by this work):

```bash
/home/catid/specialist/es-at-scale/.venv/bin/python /home/catid/specialist/run_lora_es_v71_v72_live_calibration_v73.py --preregistration /home/catid/specialist/experiments/eggroll_es_hpo/preregistrations/lora_es_v71_v72_live_calibration_v73.json --preregistration-sha256 320b038f07b615622cab0a2a5a9aec86aa06d7649e794201794f382d8ab3783e --preregistration-content-sha256 8bea32f21d33970f1b234cfe59ebaa3eb60fcd47faca1aff1913d81f5dcfe08c --execute
```

The same command with `--dry-run` completed without model/Ray/GPU loading,
filesystem writes, or protected access. Validation: 12 focused V73 tests and
95 V41/V66/V66d/V71/V72/V73 adjacent tests passed. Contract regeneration,
self-hash validation, and Python compilation passed.
