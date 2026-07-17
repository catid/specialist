# Qwen3.6 LoRA scalar exchange and deterministic replay V84A

## Decision

Keep the native exact-FP32 V72 update.  V84A registers a source-only scalar
exchange/replay alternative, but authorizes no live implementation or run.
An accepted V73D profile must first rank the canonical dense collective among
the top three bottlenecks in at least two of three replicates.  Even then, a
paired arm is useful only if the removed collective time exceeds the extra
four-rank replay work.

## Exact network projection

The native 70-call, 4,528,128-element FP32 update projects
27,168,768 ring bus bytes per actor.
Eight antithetic directions collapse to eight uint64-seed/binary64-coefficient
pairs.  Their fixed allgather projects 96
bytes per actor; the four SHA-256 digest consensus projects another
96 bytes.  Total scalar traffic is
192 bytes per actor, nominally
141,504x smaller.  These are ring/
allgather projections, not measurements of the canonical communicator.

## RNG, scratch, and HBM tradeoff

Balanced native work generates two directions per actor; replay generates all
eight.  IID normals therefore rise from
9,056,256
to 36,225,024 per
actor.  The same 4x multiplier applies to structured RNG values at ranks 1,
4, 8, and 16.  At 16,384-element chunks, exact-order scratch ranges from
262,144
bytes for IID to
788,480
bytes for structured rank 16; no full dense noise/update surface is allocated.

The explicit source-equivalent HBM ledger, excluding RNG and collective
internals, is 217,350,144 bytes per
native actor versus 941,850,624 bytes
for exact-order replay, an increment of
724,500,480 bytes.  A fused implementation
could reduce that traffic, but V84A makes no fused-kernel or speed claim.

## Numerical surprise and safety boundary

Globally sorted FP32 accumulation is algebraically correct but exceeded the
accepted two-ULP final-update gate in the synthetic comparison.  The bounded
proof therefore retains each pair's origin rank, sorts seeds canonically for
identity, and replays arithmetic in ascending origin-rank then seed order.
Fake-four-rank tests cover IID and structured ranks 1/4/8/16, all-rank digest
consensus, missing/duplicate/nonfinite rejection, retry identity, provisional
rollback, terminal poison, and finalization.

No model, GPU, Ray, live communicator, dataset, training example, or protected
evaluation source was opened.  No update was committed; quality, speed,
memory-saving, live-arm, and promotion authority are false.

- V84A preregistration content SHA-256:
  `56686192dc5e9e96dd9c7f9b711151e6e3a57aefd693d5bb83197e153006fbe6`
- V84A report body SHA-256: `846854a77ad058be1e42c0b9d69d864df42c492a10a31468a6940605b985b084`
