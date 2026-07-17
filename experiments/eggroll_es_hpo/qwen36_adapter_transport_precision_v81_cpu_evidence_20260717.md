# Qwen3.6 LoRA adapter transport precision V81 — CPU/source evidence

Date: 2026-07-17 UTC

Bead: `specialist-0j5.26`

Scope: CPU-only source audit, byte contract, and failure-state tests. No GPU was
initialized, no dataset or protected content was opened, and no vLLM,
`es-at-scale`, runtime-worker, adapter, or model file was changed.

## Result

The production path already makes the right precision split:

- The 70-tensor, 4,528,128-element canonical adapter is CPU FP32
  (18,112,512 bytes). Perturbations, weighted updates, optimizer authority, and
  checkpoints must remain FP32.
- The sealed vLLM execution layout is 82 BF16 views, 4,921,344 elements, and
  9,842,688 persistent device bytes. Packing duplicates some A projections and
  splits/scales B, so the runtime element count is intentionally larger than
  the canonical element count.
- Installed vLLM 0.25.0 accepts only `auto`, `float16`, and `bfloat16` for LoRA.
  Its dense and fused-MoE slot allocations use `lora_config.lora_dtype`.
  FP8/INT8/INT4 LoRA execution views are unsupported. FP16 is supported but is
  byte-neutral versus BF16 and would introduce a semantic change, so it is not
  a memory optimization.

There is therefore no supported dtype change that can reduce persistent LoRA
VRAM or the current 9,842,688-byte H2D payload. The useful remaining
optimization is transport shape, not another precision arm.

## Credible transport challenger

V71 currently creates each projected BF16 value on CPU, calls
`expected_value.to(device=view.device)`, and then synchronously copies that
temporary device tensor into the resident runtime view. Across the sealed
layout this means:

| Per transition | V71 control | Pinned-direct challenger | Exact delta |
|---|---:|---:|---:|
| H2D payload | 9,842,688 B | 9,842,688 B | 0 B |
| H2D calls | 82 | 82 | 0 |
| Temporary device-copy payload | 9,842,688 B | 0 B | -9,842,688 B |
| Materialization HBM read/write lower bound | 29,528,064 B | 9,842,688 B | -19,685,376 B (-66.67%) |
| Largest logical temporary device tensor | 524,288 B | 0 B | -524,288 B |
| Host execution staging | 9,842,688 B fragments | one 9,842,688 B pinned bank | byte-neutral, one storage |

The HBM lower bound counts the V71 H2D write into the temporary plus the
temporary-to-runtime device read and write. The challenger writes each pinned
host segment directly into its already allocated runtime view. It adds no
device staging bank and projects 314,966,016 fewer HBM read/write bytes across
16 candidate installations. These are a byte model, not a measured speedup.

The installed vLLM source independently supports the required primitives: it
pins packed CPU LoRA weights after packing and its dense/fused-MoE `set_lora`
paths use direct `copy_(..., non_blocking=True)` operations. No site-package
modification is needed.

## Safety boundary

The CPU state machine permits one stable pinned BF16 host storage because the
production protocol has one resident slot and serialized candidate
transitions. Double buffering is rejected until a separate overlapping-slot
protocol exists.

Publication order is fail-closed:

1. Project the next FP32 state into the idle pinned bank and bind its version
   and exact projected-value identity.
2. Issue exactly 82 direct nonblocking copies totaling 9,842,688 bytes on the
   bound stream; pageable sources, device staging, and partial byte/view sets
   are rejected.
3. Record and observe the matching completion event. A pending, stale, or
   foreign event cannot advance the adapter.
4. Perform the existing one-readback V71 exact audit only after the event.
5. Publish the candidate for generation only after the exact identity matches.
6. Reuse the host bank only after that candidate is retired.

An unknown or partially observed copy cannot accept a reward or update. It
must restore the exact FP32 master and prove the restored identity; otherwise
the actor becomes terminally poisoned. Final success requires an unpoisoned
idle state.

## Relationship to existing work

This contract intentionally does not duplicate neighboring tasks:

- `.19`/V72 owns the one-copy FP32 host master and checkpoint publication.
- `.18`/V72 owns structured perturbation and weighted-update generation.
- `.21`/V71 owns exact audit cadence, single D2H readback, restore, and poison.
- `.16` owns the sole resident LoRA slot.
- `.14` owns phase and transport telemetry.

The new task depends on `.14`, `.18`, `.19`, and `.21`. It does not block the
production layout task `.20`: synchronous BF16 remains a valid safe default,
and this challenger changes optional transition latency/HBM traffic rather
than the persistent layout or H2D byte volume.

## Preregistered live decision

No GPU run is authorized by the CPU artifact. A later integration must use at
least three counterbalanced four-GPU pairs, with the pinned-direct path as the
only change. Promotion requires:

- exact candidate, post-generation audit, reward, update, abort, and final
  identities;
- no H2D-byte increase and zero device-staging bytes;
- a material transition-time improvement, with H2D/copy trace, HBM activity,
  peak allocated/reserved VRAM, PCIe, pinned-host/NUMA, and cleanup evidence;
- useful-work/PID attribution on all four GPUs; and
- semantic plus source-disjoint OOD non-inferiority.

Machine preregistration:
`experiments/eggroll_es_hpo/preregistrations/qwen36_adapter_transport_precision_v81.json`

- preregistration content SHA-256:
  `db01b11ca0fc1565cc81a00cd0a6426f80ae840f907eb0446fda5fb0908eeb80`
- transport-plan content SHA-256:
  `469d56a33c30501df55f7d599c4b8f4d65d2cf807ca76ca2f467d56e778c5d4c`

## Validation

- 25 focused and 83 combined adjacent CPU tests pass.
- Deterministic preregistration `--check` passes.
- Both Python files compile.
- Scoped `git diff --check` passes.
