# Qwen3.6 LoRA exact-FP32 collective coalescing V83A

## Result

The source-only contract is sealed and its focused synthetic CPU suites pass
22 tests.  It binds the accepted 70-key, 4,528,128-element, 18,112,512-byte
FP32 LoRA surface and four deterministic source-order, no-split bucket plans.
No bucket is selected and no live launch is authorized.  V73D must first show
that canonical collective-call latency is material.

| Plan | Calls | Maximum staging bytes | Calls eliminated |
|---|---:|---:|---:|
| `flat_all_18112512b` | 1 | 18,112,512 | 69 |
| `bounded_8mib` | 3 | 8,372,224 | 67 |
| `bounded_4mib` | 5 | 4,194,304 | 65 |
| `bounded_2mib` | 10 | 2,064,384 | 60 |

The network payload and nominal ring bytes are unchanged for every plan.  The
flat plan needs 18,112,512 bytes of sequential
staging, 17,063,936
bytes above V72's largest single accumulator.  A materialized pack plus GPU
unpack moves at least 72,450,048
local HBM bytes per actor per update, excluding noise generation, D2H, and
NCCL internals.  Therefore the intended future design is direct generation
into bucket views followed by reduced-slice D2H; that path is not measured or
implemented live here.

## Bound runtime semantics

The future canonical call is
`self.inter_pg.all_reduce(bucket, out_tensor=bucket, stream=stream)` in FP32.
Fill, collective, scale, and D2H stay ordered on one stream.  An event is
recorded after scale and synchronized before host consumption or staging
reuse.  Any fill, collective, incompatible return, event, or unpack failure
must preserve the exact original master or terminally poison the transaction;
a partial candidate and stale retry are forbidden.

The fake-four-rank proof is bitwise exact because native and coalesced paths
use the same explicit rank order.  This does not prove live bitwise identity:
a different message size may make PyNccl/NCCL choose another reduction tree.
A future live arm therefore needs its own exact candidate/restore gate plus a
paired unchanged-V72 performance and memory comparison.

Focused command: `.venv/bin/pytest -q test_eggroll_es_fp32_collective_coalescing_v83a.py test_build_qwen36_lora_fp32_collective_coalescing_preregistration_v83a.py`

Observed result: `22 passed` (CPU only; no model, Ray, GPU, dataset, evaluation,
OOD, holdout, shadow, or probe access).

V83A content SHA-256: `a624e22dece1fdbe6287870c9f4596d1134e3daeb9cc4f3ff3598e4551384e57`
