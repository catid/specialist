# Qwen3.6 FP8 hybrid-KV capacity matching V79 CPU evidence

## Outcome

V79 is implemented and CPU-preregistered, but it has not been launched.  It
reproduces the current V78 FP8 per-token-head hybrid-cache runtime and changes
only `gpu_memory_utilization`, from `0.500` to `0.485`.  The actor fails during
engine construction unless live group-aware capacity is at least 161,792
tokens.  No projection, log parse, semantic score, or prior run can substitute
for that live engine field.

The preregistration is
`experiments/eggroll_es_hpo/preregistrations/qwen36_fp8_kv_capacity_matched_v79.json`.
Its canonical content SHA-256 is
`6c73ac0f6bf4019cdf297546e4315dc99b68d9549a24c03f4eaa9c8ebb589023`;
its file SHA-256 is
`0e195c05fd72e36656ee6536d6656d932aac0028fbbc7983f688df9dc7b18753`.

## Evidence ancestry

The launch ancestry is the post-residency-attestation evidence, not the older
first replication:

- V76 r7 uses auto/BF16 attention KV at utilization 0.500.  All four actors
  report 7.77 GiB, 157,696 tokens, and 77.00 full 2,048-token contexts.  Median
  data-free runtime is 48.5125 seconds.  The immutable nine-file run bundle is
  `46cf5ab3e6d3688de25cfdcf101710a129fdba309a5f11a9404d17344848e5e6`.
- V78 r3 uses `fp8_per_token_head` attention KV at the same utilization and
  cache budget.  All four actors report 198,656 tokens and 97.00 contexts, a
  25.97% capacity increase.  Median runtime is 49.3601 seconds, 1.75% slower
  than V76 r7.  The immutable run bundle is
  `e6df12c976910948c1026249b05fc065932169897aa5a09ff984b6d765385463`.
- V78 r3 resolves `TRITON_ATTN`, dynamic per-token-head attention KV scales,
  no skipped KV layers, and an FP32 Mamba SSM cache.  FP8 KV therefore does
  not mean the whole hybrid cache is FP8.
- Every V78 r3 worker also binds the current named-parameter residency: 813
  language parameters and 35,712,084,096 logical bytes, with no separate
  visible vision or MTP named-parameter component.  V79 requires that exact
  identity.
- V78 r1 remains immutable historical evidence (bundle
  `0897b1c80b8161171736b994e1e5e4a88728a19d39200f280ff7799552838c71`,
  median 49.3861 seconds, 198,656 tokens), but it is not V79's source ancestry.

## Why utilization 0.485

The sealed devices have 97,887 MiB each.  Reducing utilization from 0.500 to
0.485 releases 1,468.305 MiB of vLLM's requested budget.  Scaling from V78's
7.77-GiB/198,656-token observation projects 161,995.6 raw tokens.  Hybrid
capacity is operationally evaluated in complete 2,048-token contexts, so the
conservative aligned projection is 79 contexts or 161,792 tokens.  That leaves
two contexts (4,096 tokens, 2.60%) beyond V76's 77-context floor while removing
18.56% of V78's excess capacity.

On the sealed 0.001 grid, 0.484 releases 1,566.192 MiB and projects only
159,551.6 raw tokens, which rounds down to the 157,696-token V76 floor.  It
does not retain the preregistered two-context safety margin.  Thus 0.485 is the
lowest grid point eligible for a live test.  These calculations select an arm;
they are not a capacity result.

The projected telemetry peak is approximately 49,388 MiB rather than V78's
observed 50,856 MiB, but this is also not a measurement.  V79 must report its
actual peak and retain at least the V78-observed 47,031-MiB physical headroom.

## Wrapper and telemetry contract

The additive wrapper reconstructs the V78 settings directly around the V73
data-free engine call.  This avoids rewriting or falsifying V74's historical
live certificate, which correctly says 0.500.  The V79 actor independently
attests:

- serialized block-FP8 weights, routed TRITON FP8 MoE, eager FCFS execution,
  one GPU LoRA slot, `max_num_seqs=68`, and the unchanged call plan;
- utilization 0.485, `fp8_per_token_head`, no scale calculation or skipped
  layers, Mamba cache `auto` resolving its SSM state to FP32, and at least
  161,792 group-aware cache tokens;
- exact current routed-expert and named-parameter residency identities;
- ten timed generation calls (two warmups and eight measured calls), generated
  tokens per second, and median, p95 nearest-rank, and maximum call latency;
- only token counts and SHA-256 output commitments, never prompts, generated
  text, or token IDs.

The separate NVML monitor samples at 0.5 seconds without creating a CUDA
context.  Every row includes the physical GPU and UUID, launched actor root
PID, attributed descendant compute PIDs, foreign PIDs, GPU utilization, HBM
utilization percentage, VRAM used/total, power, and PCIe RX/TX KiB/s.  PCIe
counter absence fails closed.  Post-run integration must report RX/TX byte
estimates using sampled elapsed time and label them as left-rectangle
estimates.  HBM bytes/s must not be inferred from `utilization.memory`.

## Acceptance and promotion gates

All four actors must self-hash, bind distinct PIDs to physical GPUs 0-3, show
useful GPU and HBM activity, agree between the live cache field and capacity
log, remain below 1.03 times V78 r3 median runtime, and satisfy the measured
VRAM, throughput, latency, PCIe, power, and log gates.  Required log fragments
include TRITON routed MoE, `TRITON_ATTN`, cache GiB, cache tokens, concurrency,
and disabled FlashInfer autotuning.  DeepGemm auto-disable/E8M0, traceback,
and OOM fragments are forbidden.  Known default MoE/LoRA warnings remain
visible rather than being suppressed.

Cleanup requires engine shutdown and process-group destruction in every actor,
no remaining actor or foreign compute PID, and at least two consecutive
post-exit batches at zero GPU utilization and no more than 4 MiB used.

The data-free output gate requires candidate-state separation, exact candidate
repeats, and a V78/V79 token-hash drift matrix that does not hide the known
reference-repeat nondeterminism.  Only after those gates pass may a paired,
source-disjoint semantic check run (mean delta at least -0.001 and paired 95%
lower bound at least -0.002).  The protected OOD check is one-shot after
selection freeze (paired 95% lower bound at least -0.005, worst-stratum point
delta at least -0.01, and no new safety failures).  No retuning is allowed
after protected access.  Scored evaluation, training, checkpointing, and
production-layout promotion remain false by default.

The known non-exact reference restore is still unresolved; V79 capacity alone
cannot promote FP8 KV.

## Exact live command

This command was sealed but not executed by the CPU work:

```bash
RUN=/home/catid/specialist/experiments/eggroll_es_hpo/runs/v79_fp8_kv_capacity_0485_r1 bash /home/catid/specialist/launch_qwen36_fp8_kv_capacity_v79.sh
```

The launcher SHA-256 is
`4ca93e3a171787bb56613bf3648365ae96a355e28ec95f094b12d1982b6772df`;
the telemetry monitor SHA-256 is
`30b67f5665aba0159c4d42c5c99ec15e6939f67fe0b5d065336b60cdc40f2d4e`.

## CPU validation

The validation used no GPU initialization, dataset/protected access, training,
checkpoint write, configuration promotion, or site-package mutation.

```text
es-at-scale/.venv/bin/python -m pytest -q \
  test_probe_vllm_quantized_adapter_switch_v73.py \
  test_probe_vllm_fp8_rightsized_v74.py \
  test_probe_vllm_fp8_attested_v76.py \
  test_probe_vllm_fp8_kv_cache_v78.py \
  test_probe_vllm_fp8_kv_capacity_v79.py \
  test_build_qwen36_fp8_kv_capacity_preregistration_v79.py \
  test_monitor_qwen36_fp8_kv_capacity_v79.py
34 passed in 1.21s

python3 build_qwen36_fp8_kv_capacity_preregistration_v79.py --check
status=cpu_preregistered_live_launch_not_performed

python3 -m py_compile \
  probe_vllm_fp8_kv_capacity_v79.py \
  build_qwen36_fp8_kv_capacity_preregistration_v79.py \
  monitor_qwen36_fp8_kv_capacity_v79.py

bash -n launch_qwen36_fp8_kv_capacity_v79.sh
```
