# Qwen3.6 fused structured runtime V72: CPU evidence

## Outcome

A production-shaped CPU implementation now generates absolute-index IID or
rank-1/4/8/16 structured perturbations in bounded chunks and writes them
directly into the real Qwen3.6 packed BF16 LoRA layout.  It also streams the
weighted ES update into one transaction-owned FP32 pending master without a
whole noise or update surface.

The implementation is correctness infrastructure, not GPU promotion.  It did
not initialize CUDA, open a dataset or protected content, train, score, or
change the V71 worker/audit files.

- Implementation: `eggroll_es_fused_structured_runtime_v72.py`, SHA-256
  `357607f3c16b071f67d2bc3adb0317bbbd29f31f7e1db0cf1aa3030ac997df6e`.
- Machine preregistration:
  `preregistrations/qwen36_fused_structured_runtime_v72.json`, content SHA-256
  `d3144d7f22570951974b8eb366d10e5e4ab9a3d3cc149d44afcb4b2be5c2ee58`,
  file SHA-256
  `ed7d78f2570032d8cd31229f847ce572628ea2347a2bd6b4af4fe28071c1e5df`.
- Production runtime-projection manifest content SHA-256:
  `7ad7c2ec6f55d38915744a6287e1d0bd56b4393f319053c62f3f4c9e36c9dcf5`.

## Production topology bound

The projection manifest is derived from four identical sealed V61 Qwen
installations, not a toy one-to-one layout.  The source report is bound at
`89aa6b70b6150cc5abafa6ebddaffebf1751fb6001c136fdd0dd40dd29ad2878`.
All four installations have assignment identity
`bac008805d7fc7c6279c47255d8d1563b0be978cb21109e8c013114f143e09df`.

The manifest covers:

- 70 canonical FP32 PEFT tensors and 4,528,128 source elements.
- 82 unique packed BF16 runtime views and 4,921,344 runtime elements
  (9,842,688 bytes).
- Full packed-A duplication where a canonical A tensor feeds multiple packed
  slices.
- Disjoint row-aligned packed-B splitting and the production B scale of 2.0.
- Unique runtime keys, exact source/runtime extents, no runtime aliasing, and
  exact B gap/overlap rejection.

No tensor payload from the sealed GPU report was opened.  Only its assignment
certificates and materialization aggregates were consumed.

## Sealed RNG and counter schedule

The implementation imports, rather than redefines, the structured ES oracle
at file identity
`8fca35f89744f292ef0d9327f547196dd26f93268336f3fad4812a065f35f740`.
Each random value is fixed by:

1. Direction seed.
2. Full canonical source-tensor key.
3. IID element or structured left/right/rank domain.
4. Tensor-local absolute element or factor ordinal.

Global work uses source keys in ascending order and half-open shard ranges
`floor(total*rank/world_size)`.  Chunk and shard order cannot change values.
Weighted updates sort unique seed/coefficient pairs by direction seed and use
fixed ascending FP32 reduction.  Structured products retain the oracle's
ascending component reduction and FP32 rounding.  A gap, overlap, duplicate,
or mid-transaction world-size change fails closed.

## Direct candidate lifecycle

The runtime class owns the immutable CPU FP32 master and preallocated BF16
views.  Candidate chunks contain only bounded noise, candidate, optional
scaled projection, and BF16 staging values.  Each source chunk is projected
directly into every intersecting packed runtime span; there is no whole FP32
candidate or noise allocation.

The lifecycle is compatible with the unchanged V71 audit contract, bound at
file identity
`cc80ac0e1bf3c9db83e3275df16ea1479273d92a40240496163543643bd0eaa8`:

- Candidate begin checks the cached immutable-master and runtime
  object/storage/version invariants.
- After exact source/runtime coverage, the deliberate write advances V71
  versions with a fail-closed provisional content sentinel.  The normal cheap
  pre-generation edge therefore works without an early D2H readback; any
  accidental exact audit before post-generation rejects.
- Reward remains provisional.  One post-generation exact BF16 readback checks
  every runtime chunk against its independently stored expected digest,
  detects both versioned and unversioned corruption, constructs the exact
  per-view identity, and rebinds the V71 registry.
- Restore streams the immutable master through all packed projections, then
  exact-audits both the master and the runtime once.  An unknown or partial RPC
  must complete this exact restore or terminally poison the transaction.

## Streamed update lifecycle

The update transaction requires an exact V71 `update_acceptance_sha256` before
work begins.  It accumulates each bounded update chunk in fixed seed order and
writes `master + step*update` directly into one FP32 pending master.  That
18,112,512-byte pending master is persistent transaction output needed for
atomic ownership and rollback; it is not scratch.  No whole noise or update
tensor is allocated.

The pending master is not committed by `finish`.  Commit remains provisional
until an exact V71 commit-boundary identity is supplied, and the rollback
master remains owned until the exact final boundary.  Partial pre-commit work
is discarded only after an exact original-master audit.  A corrupt rollback
or corrupt final identity terminally poisons the transaction.  The registered
final-update gate is at most two FP32 ULP versus the independent CPU oracle.

## Production byte and scratch ledger

The direct path preserves the irreducible 9,842,688-byte BF16 write for each
candidate and restore and preserves one exact 9,842,688-byte readback after
generation and restore.  Compared with the current candidate path, it removes
an 18,112,512-byte whole-candidate FP32 device-to-host copy and a 9,842,688-byte
pre-generation runtime equality/readback per signed candidate.  That is
27,955,200 bytes removed per candidate, or 447,283,200 bytes over a 16-candidate
update, before counting avoided per-view synchronization overhead.

All methods write 157,483,008 BF16 bytes over 16 candidates and the same amount
over 16 restores.  Scratch ceilings at the sealed 16,384-element chunk are:

| Method | Random values/direction | Candidate scratch | Update scratch |
|---|---:|---:|---:|
| IID absolute index | 4,528,128 | 229,376 B | 196,608 B |
| Structured rank 1 | 143,744 | 262,272 B | 229,504 B |
| Structured rank 4 | 574,976 | 360,960 B | 328,192 B |
| Structured rank 8 | 1,149,952 | 492,544 B | 459,776 B |
| Structured rank 16 | 2,299,904 | 755,712 B | 722,944 B |

The candidate ceilings conservatively include full per-tensor structured
factor cache, FP32 noise/candidate/scaled staging, and BF16 projection staging.
The update ceilings match the oracle's bounded factor plus three-FP32-buffer
account.  Whole-surface noise, candidate, and update allocations are zero.

## Adversarial verification

Focused coverage includes:

- Exact IID and rank-1/4/8/16 results across full writes, arbitrary chunks, and
  1/4/7 shards in permuted order.
- Exact packed-A duplication and packed-B split/scale.
- Candidate and update gaps, overlaps, duplicate shards, and world-size drift.
- Versioned and unversioned runtime corruption before reward acceptance.
- Partial candidate repair, injected restore failure, and terminal poison.
- Partial update/generator failure with exact original-master preservation.
- Provisional commit rollback, corrupt rollback poison, commit/final ordering,
  and seed-order invariance.
- Exact streamed update reconstruction and the accept-2/reject-3 ULP boundary.

The adjacent CPU-only run kept CUDA hidden and included the original structured
oracle plus V71 audit-contract and worker suites:

```text
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  test_eggroll_es_fused_structured_runtime_v72.py \
  test_build_fused_structured_runtime_preregistration_v72.py \
  test_structured_es_oracle_v1.py \
  test_build_structured_es_preregistration_v1.py \
  test_eggroll_es_audit_contract_v71.py \
  test_eggroll_es_worker_lora_v71.py
127 passed in 1.99s
```

Focused test identities:

- Runtime tests:
  `b0084612fe2d9b5ea925a2e9640ca147b76735b4df9bf412495885905b7a7ff8`.
- Preregistration tests:
  `992517352e57055e4cfe237d92880ce47f23c950c65550332a433e0c9a8d317a`.
- Preregistration builder:
  `77c9217b2c6d912038f0e46419cfe9180a04f0430b5341ae2c3e78e174620a7e`.

## Remaining blockers

`specialist-0j5.18` remains in progress.  CPU evidence does not satisfy its
live acceptance criteria.  Promotion still requires a CUDA implementation
integrated into the V71 worker without changing the audit schedule, four-GPU
candidate/update/restore receipts, direct PCIe/HBM and peak-scratch counters,
at least three paired throughput replicates, exact candidate/restore and
at-most-two-ULP update certificates, cleanup-idle telemetry, and Qwen reward,
semantic, and OOD noninferiority.
