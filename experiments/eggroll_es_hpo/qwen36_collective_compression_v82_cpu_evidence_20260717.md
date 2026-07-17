# Qwen3.6 EGGROLL-ES collective compression V82 CPU evidence

Status: sealed CPU/source contract; live launch is deliberately blocked.

This work defines a conditional FP32-versus-BF16 update-collective ablation for
`specialist-0j5.28`. It did not initialize a GPU, load a model, open a dataset,
read dev/OOD/protected content, train, or create/promote a checkpoint.

## Current decision

Do not launch the collective-compression GPU probe yet. Beads
`specialist-0j5.14`, `.18`, and `.19` remain in progress, while the new exact
post-optimization profiling gate `.32` is open; the required
post-optimization bottleneck receipt does not exist. V66d observed a 4.202 s
update-execute span (4.622 s upper window) with approximately 8,075.28 MiB
sampled PCIe RX and 1,496.05 MiB TX, but these are phase-aligned NVML sampling
estimates rather than exact NCCL link or HBM bytes. That signal is enough to
motivate a preregistration, not enough to claim that collectives remain a
top-three bottleneck after fused updates and host-copy reduction.

The launch gate requires `.14`, `.18`, `.19`, and `.32` to be closed with accepted
evidence, a profile measured after `.18` and `.19`, exact collective time, link
bytes, HBM bytes, four-GPU attribution, and a descending residual-bottleneck
ranking. If `update_collective` ranks 1--3, the data-free probe becomes
eligible. If it ranks fourth or lower, `.28` closes as not applicable without
using GPU time.

## Local implementation evidence

The builder inspects the installed implementation without importing PyTorch:

- PyTorch `2.11.0+cu130`, git
  `70d99e998b4955e0049d13a98d77ae1b14db1f45`, CUDA build 13.0;
- NCCL wheel/runtime 2.28.9;
- installed PyTorch maps `BFloat16` to `ncclBfloat16`, and the bound NCCL ABI
  declares BF16, so a BF16 arm is registered subject to a live four-GPU
  preflight;
- NCCL also declares FP8 enum values, but the installed PyTorch
  `ProcessGroupNCCL` binary contains the explicit runtime rejection
  `Unsupported Float8 type for NCCL reduction`.

Therefore ordinary FP8 all-reduce is unsupported and launch-forbidden in V82.
An NCCL datatype enum is not treated as end-to-end PyTorch collective support.
No FP8 trial, byte reinterpretation, or silent fallback is permitted.

The V68 same-process evidence remains authoritative for layout: retain the
native 23 tensor boundaries. Its two process medians for native/flat speed were
1.0035265 and 0.9977076, an effective tie, while a flat FP32 shadow costs
571,998,208 bytes (545.5 MiB) per rank. V82 forbids that shadow.

## Exact update and residual contract

Both arms consume the same canonical FP32 antithetic pair-difference update,
the same deterministic strided seed shards, and the same native tensor order.
Candidate inference precision is unchanged.

The FP32 control is the existing in-place FP32 PyNccl sum. It performs no
conversion or rescaling before the collective and allocates no residual. Any
nonzero inherited residual makes the fallback fail closed; a failed compressed
transaction must roll back before the independent exact FP32 control runs.

For the BF16 challenger, each local FP32 element follows:

```text
compensated = fp32(update + old_residual)
q = BF16_RNE(compensated)
next_residual = fp32(compensated - fp32(q))
```

The CPU oracle proves `fp32(q + next_residual)` reconstructs the compensated
FP32 bits exactly for every accepted element and rejects nonfinite or
overflowing values. It records the local BF16 spacing bound and requires the
observed error to remain within it. This local conservation does not fabricate
bit-deterministic NCCL reduction order; live cross-rank result and residual
receipts remain mandatory. Residuals are rank-local and are not required to be
identical across ranks; their manifests and coverage must be complete.

Residual state has explicit tensor order, FP32 bit strings, version, and a
canonical SHA-256. Prepare does not publish. Exact rollback requires the old
and candidate FP32 residual generations to coexist until the candidate master
and residual can commit atomically. Resume restores the exact residual bytes
and must replay identical transmitted and next-residual bits. Unknown
generations, one-element changes, reordered shards, partial collectives, or
master/residual publication uncertainty force exact restore or terminal poison.

## Memory and bandwidth accounting

The production update surface is 142,999,552 FP32 elements in 23 tensors per
rank; the largest tensor has 25,165,824 elements.

| Quantity, per rank | FP32 control | BF16 + FP32 residual |
|---|---:|---:|
| Collective payload | 571,998,208 B | 285,999,104 B |
| Nominal four-rank ring bus bytes | 857,997,312 B | 428,998,656 B |
| Steady FP32 residual | 0 B | 571,998,208 B |
| Transaction peak FP32 residual | 0 B | 1,143,996,416 B |
| Largest native BF16 staging tensor | 0 B | 50,331,648 B |
| Incremental transaction peak | 0 B | 1,194,328,064 B |

The nominal projected ring saving is 428,998,656 bytes per rank, or
1,715,994,624 bytes across four ranks. It is not a measurement. A fused local
prepare must read the FP32 update and old residual, then write the new FP32
residual and BF16 payload: at least 2,001,993,728 HBM bytes per rank per update
before NCCL. The BF16 result is copied to CPU, converted/scaled in FP32, and
applied to the canonical FP32 master; a full FP32 GPU decode staging surface is
forbidden.

This is the central tradeoff: BF16 halves collective payload but adds about
1.11 GiB of transaction residual storage plus a 48 MiB staging tensor per rank
and substantial local HBM traffic. Promotion requires measured end-to-end
update throughput, not payload arithmetic. A replicated payload saving with no
throughput gain retains FP32.

## Prospective data-free probe

`benchmark_eggroll_es_collective_compression_v82.py` is sealed but currently
non-launchable. It reads and validates the post-optimization gate before
importing `torch`, uses all four GPUs, native tensor boundaries, three
registered synthetic seeds (1701/1702/1703), and same-process counterbalanced
AB/BA blocks. It opens no model or data and cannot promote quality. Its exact
future command is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=INFO .venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=4 benchmark_eggroll_es_collective_compression_v82.py --gate experiments/eggroll_es_hpo/gates/qwen36_collective_residual_bottleneck_gate_v82.json --output experiments/eggroll_es_hpo/runs/v82_collective_compression_data_free_preflight/preflight.json
```

Running that command now correctly fails before CUDA because the gate receipt
is absent. Even after it becomes eligible, the preflight's algorithmic byte
ledger is not a link measurement; the profiler receipt required by `.14` must
be joined before any systems decision.

Only a data-free throughput win advances to three-seed recipe confirmation.
The immutable source-disjoint evaluation contract then requires pooled paired
dev 95% LCB at least 0, positive dev deltas on at least two seeds, OOD QA reward
LCB at least -0.02, OOD exact-count delta at least -1, OOD prose logprob LCB at
least -0.02, and every OOD condition passing. Protected holdout remains closed
during HPO.

## Artifacts and validation

Files added:

- `eggroll_es_collective_compression_v82.py`
- `benchmark_eggroll_es_collective_compression_v82.py`
- `build_qwen36_collective_compression_preregistration_v82.py`
- `test_eggroll_es_collective_compression_v82.py`
- `preregistrations/qwen36_collective_compression_v82.json`

Focused validation at the time of sealing:

- `.venv/bin/pytest -q test_eggroll_es_collective_compression_v82.py`: 18 passed;
- focused plus adjacent V68/V71/V72/structured-oracle suites: 109 passed;
- `python3 -m py_compile` over the oracle, builder, preflight, and tests: passed;
- builder `--write` followed by `--check`: passed;
- no-GPU import guards for both the builder and blocked prospective runner:
  passed.

The Bead remains in progress. This milestone is a correctness and measurement
contract, not evidence that BF16 collectives improve this trainer.

Sealed identities: preregistration content SHA-256
`42f96cdaf7ada94f1e696aecc7d8254f25ee3e3b4fbce8400ba8218f4abf0cc3`,
preregistration file SHA-256
`ca727884ceda45652a2d3d67677107e3e8582173bd8c96ee556b96bf6521e1f6`,
CPU oracle SHA-256
`1b205a846dde09da73f5f36477f68e643f0d2f3ae89b16eb7918db46db03022d`,
and prospective preflight SHA-256
`24a2b8d94d79845e54c7608869feca72fe2e5c0dbdf00a46110a1a45f18b3888`.
