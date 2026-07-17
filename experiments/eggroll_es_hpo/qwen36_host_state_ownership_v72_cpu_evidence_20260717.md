# Qwen3.6 canonical LoRA host-state ownership V72 CPU evidence

Status: additive CPU implementation and fault-injection evidence accepted.
Four-GPU Qwen RSS, NUMA, transfer, and phase-time confirmation remains required
before `specialist-0j5.19` can close. V66/V66d and `es-at-scale` were not
modified. No dataset or protected evaluation content was accessed, and no GPU
was launched.

## Audit result

The canonical FP32 LoRA state is 70 tensors, 4,528,128 elements, and 18,112,512
bytes per actor. V41 retains a second full tensor bank as `_v41_reference`, but
no code ever reads its tensors: only reference identity, generation, and
freshness are consumed. V41 also creates a rollback clone at prepare even
though the canonical master is immutable, full-state validation clones around
the executed candidate, up to seven full-state copy passes around commit, and
four full-state passes across checkpoint write/readback.

V72 replaces those implicit copies with explicit generation ownership:

- Install copies each source-mapped tensor exactly once into the owned master.
- A reference is identity and generation metadata only; it has no tensor bank.
- Prepare retains a versioned exact-auditable lease that aliases the immutable
  master, so rollback allocates nothing.
- Execute streams one reduced tensor through CPU at a time and owns exactly one
  complete candidate. Validation hashes that bank without cloning it.
- Commit moves candidate ownership to the canonical role and retains the old
  master as rollback. Final releases rollback only after V71's exact final
  boundary passes.
- Abort verifies and re-adopts the old owned bank. An unknown or corrupted
  state still restores exactly or terminally poisons the actor.
- Rank-zero checkpoint writes the immutable master directly into a hidden
  temporary directory, fsyncs and verifies it one tensor at a time, performs a
  post-write exact master audit, and atomically renames the complete directory.
  The canonical topology's largest streamed readback tensor is 1,048,576 bytes;
  no 18,112,512-byte readback bank is built.

The candidate math and identity are tested against the original FP32 update
ordering. V71's base/runtime/master exact-audit lifecycle remains in force.

## Exact byte and logical RSS account

For one install/reference/update/commit/final/checkpoint lifecycle, the old
path performs 26 full-state tensor copy passes per actor. V72 performs one—the
necessary source-mmap-to-owned-master copy at install. This removes 25 passes,
or 452,812,800 copied bytes per actor and 1,811,251,200 bytes across four
actors.

The old commit path's logical tensor-residency peak is seven banks, 126,787,584
bytes per actor. V72's peak is two banks, 36,225,024 bytes, saving 90,562,560
bytes (71.43%) per actor and 362,250,240 bytes in four-actor RSS sum. Quiescent
state falls from two banks to one, saving 18,112,512 bytes per actor and
72,450,048 bytes across four actors. The streamed checkpoint's logical tensor
peak is 19,161,088 bytes rather than the old 72,450,048-byte peak.

These figures count owned tensor storage and explicit full-state tensor copy
passes. Hash serialization, safetensors page-cache residency, per-tensor update
staging, and model/KV/runtime buffers are named exclusions rather than being
silently mixed into the totals.

## Isolated CPU RSS diagnostic

Fresh child processes with GPUs hidden touched either seven or two synthetic
canonical-size FP32 banks. The baseline RSS delta was 128,278,528 bytes; V72's
was 37,695,488 bytes, an observed reduction of 90,583,040 bytes (70.61%). The
20,480-byte difference from the exact 90,562,560-byte tensor saving is allocator
and process accounting overhead.

A separate diagnostic copied 470,925,312 bytes in 26 passes in 19.037 ms versus
18,112,512 bytes in one necessary pass in 4.364 ms. These timings demonstrate
the CPU mechanism only; they are not treated as four-GPU Qwen acceptance or a
stable bandwidth ratio.

## Fault injection and atomicity

Tests cover normal and version-counter-bypassing rollback mutation, a
one-element candidate mutation, an incomplete candidate dictionary, a partial
checkpoint file followed by an exception, and master mutation during snapshot
publication. No corrupted or incomplete candidate becomes canonical. No
failed checkpoint creates the final directory. If the only rollback bank is
corrupted, execution stops before the collective candidate can be accepted and
the actor is terminally poisoned.

Speculative pinning was deliberately not selected. The runtime consumes
converted and split BF16 views rather than the FP32 master directly, so pinning
the master does not make those transfers asynchronous. A future live arm may
test one bounded NUMA-local BF16 staging buffer with CUDA-event completion and
locked-memory accounting; CPU-only evidence cannot safely claim that benefit.

The 143,807,290,816-byte dense FP32 master design is also not implemented here.
Four private dense masters are 575,229,163,264 bytes. Cross-process mmap or
shared-memory ownership still needs capacity qualification, NUMA placement,
page-fault measurement, and a recovery protocol proving that no actor observes
a partial generation. The safe LoRA ownership result is not extrapolated into
that materially different design.

## Sealed evidence

Contract: `preregistrations/qwen36_host_state_ownership_v72.json`

- File SHA-256: `b748d9eb4b7404b753f29f3cc1ff6827e152e70d44b430aac9f17d026679a398`
- Content SHA-256: `95f29e599bfdee505035cf2a4a1182f6f8322d70a52f603a55b79a18c2896ad8`
- Residency content SHA-256: `31097c19e319d3711400cf9a4d13038bbad8ed53e5f0f89e355f0f69329b6fce`

CPU benchmark: `qwen36_host_state_v72_cpu_benchmark_20260717.json`

- File SHA-256: `ce3f5d150c2b0d4aa0ae7fb90fe09ede8e433802d023f2ce8600da940cdf8774`
- Content SHA-256: `ed5247676797c8a209e768e4dd04c0250b3acf49e29565f013fee74af01a39ed`
- Focused validation: 15 tests passed
- Focused plus V41/V66/V66d/V71 adjacent validation: 65 tests passed

## Remaining live gate

Run the unchanged four-actor Qwen flow with V72 and record per-process host RSS
and peak, NUMA node and page placement, locked/pinned bytes, install/update/
abort/final/checkpoint elapsed time, H2D/D2H bytes, candidate/update/final
identity equivalence, exact restore or poison receipts, atomic checkpoint
readback, all-GPU useful-work receipts, and cleanup idle. Dense shared-master or
pinned-staging work remains a separate unapproved arm until those measurements
exist.
