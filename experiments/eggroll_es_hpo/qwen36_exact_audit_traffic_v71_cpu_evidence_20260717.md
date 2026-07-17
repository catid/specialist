# Qwen3.6 LoRA exact-audit traffic V71 CPU evidence

Status: CPU implementation and adversarial contract accepted; live four-GPU
profiling remains required before promotion or closing `specialist-0j5.21`.
No dataset, protected holdout, dev/OOD content, or GPU was opened by this work.

## Current-path evidence

The immutable V66d run's measured/code-path ledger contains 60 full frozen-base
hashes across four actors. At 285,999,104 bytes each, those hashes moved
17,159,946,240 D2H bytes. Its 60 runtime materializations separately read the
9,842,688-byte BF16 LoRA state for equality and SHA, producing 120 D2H calls and
1,181,122,560 additional D2H bytes. These values are reproduced exactly in the
V71 machine-readable contract.

## V71 design

V71 preserves V66/V66d and adds a new worker surface. Every transition checks
the tensor mapping, object, storage pointer and offset, shape, stride, dtype,
device, size, and PyTorch version without a D2H readback. Exact frozen-base
content checks remain before population reward acceptance, before update
acceptance, after provisional commit state, and after provisional final state
on every actor. Checkpoint creation adds another exact base/master/runtime
boundary and retains the existing serialized-file readback.

The live LoRA views are concatenated once on device, copied to CPU once, and
then compared and SHA-hashed segment-by-segment from that one host buffer. The
expected values are not concatenated on CPU. Deliberate canonical-master
replacement is adopted from one exact no-clone identity; subsequent transition
checks reuse its immutable identity cache. Object, storage, or version drift is
still rejected cheaply, and a version-bypassing content write is rejected at
the next exact boundary.

Commit and final receipts are withheld until their exact boundary passes.
Rollback ownership is retained until final succeeds. Legacy prepare/commit
entry points cannot bypass the V71 population/update acceptance tokens.
The existing V66d actor work receipt is returned byte-for-byte unchanged, but
its pre-generation call now performs the cheap invariant and its end call
withholds that receipt until the exact post-generation LoRA audit passes.
Unknown or partially completed runtime/update RPCs reconstruct the live slot
from the owned FP32 master and verify the reconstruction, or terminally poison
the actor if that proof cannot be completed.

## Byte-accounted projection

For 16 signed candidates distributed over four actors, including one
update/commit/final lifecycle, the safety-equivalent old schedule projects
23,507,615,744 device-transfer bytes. V71 projects 5,678,366,720 bytes, saving
17,829,249,024 bytes (75.84%). Including the old validation-only FP32 master
clones, modeled host-copy-or-device-transfer traffic falls from 25,536,217,088
to 5,678,366,720 bytes, a 77.76% reduction.

The fusion has an explicit cost: each actor needs a 9,842,688-byte temporary
GPU staging buffer. Across the modeled lifecycle, staging reads plus writes add
1,338,605,568 bytes of GPU-memory traffic. That cost is not hidden inside the
PCIe savings and must be measured live; it is expected to be preferable to 68
extra small synchronous D2H calls, but the CPU contract does not claim that
timing result.

## Adversarial validation

The focused CPU suite injects one-element corruption into frozen base, live
LoRA, and FP32 master tensors. Normal writes are caught by the cheap version
guard. `.data` writes deliberately bypass the version counter and are caught
by exact content audits. Object replacement, storage replacement, mapping
replacement, an uncertain partial write with successful exact repair, and a
failed repair that must poison are also covered. Commit/final ordering tests
prove that rollback state is not released early.

Machine-readable contract:
`preregistrations/qwen36_exact_audit_traffic_v71.json`

- File SHA-256: `8747e9ca3c022b593bdfcf445881106d5410c3496f0135bcd2a663f07ca55240`
- Contract content SHA-256: `14c7afe2fd370798a26641f6950e92592be67e5b2e2e5fabfc442b76462c2f99`
- Focused validation: 24 tests passed

## Remaining live acceptance gate

Run the unchanged 16-candidate, four-actor Qwen3.6 flow with V71 and record
phase-attributed H2D/D2H calls and bytes, transition latency, the 9.39 MiB
staging peak, GPU utilization, actor/GPU receipts, and cleanup-idle state.
Candidate identities, rewards, update identity, abort/commit/final receipts,
and exact final runtime/master state must match the sealed control. Until that
profile materially reduces D2H bytes and transition time, V71 is implementation
ready but not production-promoted.
