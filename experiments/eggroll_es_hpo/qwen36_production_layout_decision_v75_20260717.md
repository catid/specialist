# Qwen3.6 production-layout decision V75 (provisional)

## Outcome

The sealed safe default for confirmation is BF16, eager execution, one GPU LoRA
slot, `max_num_seqs=68`, native 23-tensor collective boundaries, and external
FP32 canonical LoRA state.  Serialized block-FP8 at
`gpu_memory_utilization=0.50` is retained as a capacity-positive challenger,
not promoted.  This document cannot authorize the final recipe benchmark.

Machine-readable decision:
`decisions/qwen36_production_layout_provisional_v75.json`

- Decision content SHA-256: `5dd23d1effbecec2068d8e21d7f8bf9e5afab85a9e8a58d38a913e835c0e0ed5`
- Decision file SHA-256: `f960df73f7082b2e5583692c1649b2057f06e012268530eb4529f2bc616a1ebb`
- Builder file SHA-256: `922c5d15f64591d0ab56eda141a10a02dd4cff78bf82abd6501cc112ab97276f`

This was a CPU-only evidence analysis.  It did not initialize CUDA, open a
dataset, inspect protected content, or authorize training/scored evaluation.

## Bound evidence

The builder fails closed if any sealed input, self hash, aggregate, receipt
coverage, physical-GPU binding, useful-work signal, or cleanup-idle signal
changes.

- Recipe evaluation contract: file
  `04af81499067e2feb0186c0a61e4c1af10f838a8eb7deec6dd41cd192748cacf`,
  content
  `2442c0c2be3ac4c883612f400f8f213ce3bc82ef96e03fad1ef10ec3b7d11fad`.
- V71 full-ES phase analysis: content
  `a4503e18cb6185ee872ad571d24a51fd8a7ac5154e1fd69793fba629842166e6`.
  It binds 492.406 charged GPU-seconds, useful work on GPUs 0-3, cleanup idle,
  84,138 MiB peak per GPU, and 13,749 MiB minimum physical headroom.
- V73 paired precision inventory:
  `ab948ff5be8a9fede83107bbf01b4cde51b5367980b3489141595af3592f1762`.
  Four actors per arm covered every physical GPU over two counterbalanced
  waves and ended idle.
- V74 immutable right-sized run inventory:
  `13dc3991ec440e273359455fdf970f0025f3048deeb74d73219d29dd845ee04c`.
  This inventory binds all four receipts, all four actor logs, NVML samples,
  and actor PID evidence.  The probe implementation is bound at
  `c4434b1c720c9c209b3638abc6202ec864e18d7f5233963f7360a88fedbd5a63`.
- One-slot inventory:
  `dc7aecfc5fd5a84a673cc1d9593c0e733b1e30a4b3fd0f71dd0d55aa1c171b93`;
  dual-slot inventory:
  `aec63377f7fbf35ad18759ea70064cb02020bb6ba5547489184e8ae12ee8a14d`;
  sequence-cap inventory:
  `42f3e5ca3342d70288c175e3d9ea3ad991b75ae1e9fd5787d8230de0086c920e`.
- Same-process native/flat collective artifacts are individually bound.  Their
  per-run native/flat median-speed ratios are 1.00353 and 0.99771.

## Precision and capacity result

At the matched V73 `gpu_memory_utilization=0.82` budget, BF16 had a 45.196 s
combined median and serialized FP8 had a 49.393 s median.  FP8 was 9.286%
slower.  BF16 loaded 68.24 GiB of model weights and exposed 6.87 GiB / 139,264
KV tokens; FP8 loaded 36.93 GiB and exposed 38.14 GiB / 775,372 KV tokens.
The fixed budget converted the 31.31 GiB weight reduction into KV capacity,
not physical VRAM savings: NVML peaks were 83,820 MiB BF16 and 83,878 MiB FP8.

V74 changed only FP8 `gpu_memory_utilization` from 0.82 to 0.50.  Its four
actors produced these sealed results:

- Median runtime 48.958 s (range 48.794-49.292 s), 8.324% slower than the V73
  BF16 median and 0.880% faster than the V73 FP8 median.
- 52,738 MiB steady/peak allocation on every GPU, saving 31,140 MiB versus the
  fixed-0.82 FP8 peak and leaving 45,149 MiB physical headroom per GPU.
- 7.73 GiB KV cache, 157,286 KV tokens, and 76.8x stated concurrency for 2,048
  tokens.  This remains above the BF16 capacity floor of 139,264 tokens.
- All GPUs reached 100% sampled utilization, all four engines shut down, and
  final NVML samples were idle at 4 MiB.

This proves the right-sized layout has useful capacity.  It does not prove it
is precision-safe or production-ready.  Reference repeats still changed 4-5
of 68 token-hash rows, only one capacity replicate exists, no full ES update
was run, and no source-disjoint semantic/OOD evaluation was authorized.  Logs
also bind unresolved execution evidence: the DeepGemm-to-CUTLASS warning is
followed by a DeepGEMM E8M0-enabled message, default MoE and FlashInfer tactics
are used, an inherited BF16 tuning path appears, LoRA kernel configs are
missing, and routed-expert method count is pending.  The capacity gate is true;
the promotion gate is false.

## Layout choices

The BF16 confirmation default and conditional FP8 challenger share the
following choices:

- One resident GPU LoRA slot.  Two slots regressed eager runtime by 73.78% and
  graph runtime by 35.73% while consuming KV capacity.
- `max_num_seqs=68`.  Global caps 48 and 32 regressed graph runtime by 23.09%
  and 35.76%; real-length bucketing remains a separate open experiment.
- Native parameter boundaries.  A flat shadow had no repeatable material speed
  benefit and would reserve another 571,998,208 bytes (545.5 MiB) per rank.
- No dense full-weight master on the external-LoRA path.
- Eager execution.  Compiled/graph execution is deferred because it changed
  token behavior and still needs semantic/OOD and kernel gates.

Current FP8 at utilization 0.82, dual LoRA slots, global caps 48/32, the flat
shadow, and silent INT4 fallback are rejected.  Fused update/audit traffic and
shared or streamed host state are deferred, not rejected, pending their
correctness and fault evidence.

## Headroom reservation

The decision reserves from the measured V71 full-ES profile rather than
assuming the data-free V74 peak carries through training:

- Device total: 97,887 MiB.
- Maximum full-ES peak: 84,138 MiB; minimum reserved full-ES headroom:
  13,749 MiB.
- Candidate/update delta: 926 MiB.
- Before ES generation/update, retain at least 14,675 MiB headroom, implying a
  maximum 83,212 MiB setup/generation peak.
- Preserve at least 139,264 KV tokens and reserve no flat shadow buffer.

## Consumer and finalization contract

A confirmation consumer must bind the exact decision content hash and an exact
layout object.  BF16 confirmation is currently permitted.  Conditional FP8
confirmation additionally requires every registered promotion field: the
sealed V74 capacity identity, three paired replicates, bounded generation and
full-ES peaks, exact reference/candidate/restore/update evidence, zero
unresolved kernel/tactic fallbacks, routed-method attestation, source-disjoint
semantic noninferiority, all registered OOD noninferiority conditions, useful
activity on all GPUs, cleanup idle, and no protected-terminal access.

The provisional artifact rejects `final_benchmark` unconditionally.  It must
be rebuilt with final hashes and consumed unchanged only after blockers
`specialist-0j5.15`, `.18`, `.19`, `.21`, and `.22` close.  Tasks `.16` and
`.17` retain provisional current choices but remain open.

## Verification

Executed with CUDA hidden:

```text
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  test_build_qwen36_production_layout_decision_v75.py \
  test_analyze_v66d_phase_memory_v71.py \
  test_probe_vllm_fp8_rightsized_v74.py \
  test_benchmark_eggroll_es_collective_paired_v68.py
51 passed in 3.46s

CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  build_qwen36_production_layout_decision_v75.py --check
passed=true, status=provisional_not_final_benchmark_authority
```

The focused V75 test file SHA-256 is
`5301c2f983a670d503ec0586d9b8dbdcde696203b900f8fbf7c7984344d0be4e`.
