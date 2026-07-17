# Qwen3.6 fused-MoE-LoRA kernel tuning V77 CPU evidence

## Outcome

The BF16/serialized-FP8 kernel-tuning experiment is fully preregistered but
GPU launch and promotion remain fail-closed under this CPU-only artifact's
authority.  V76 resolves the earlier V75 environment preflight for the bound
baseline without changing installed source: all four actors disable DeepGemm
before vLLM post-init, emit neither the DeepGemm warning nor E8M0-enable log,
and attest the routed FP8 implementation as TRITON.  DeepGemm disable and the
routed-MoE backend are separate gates.

The immutable preregistration is
`experiments/eggroll_es_hpo/preregistrations/qwen36_fused_moe_lora_kernel_tuning_v77.json`.
Its canonical content and file hashes are recorded after the final validation
run: `ad901f701a1e0045ffa9406da02ff7b8149c83e8a9f41b74c346fbf48e76d71a`
and `53a9433ca21fd8fd182126c3a7f523e393def82d08b12536771b0cfde0d17ee8`,
respectively.

## Bound evidence

- V73 has four BF16 and four serialized-FP8 actors.  Median data-free paired
  runtime is 45.1957 seconds for BF16 and 49.3926 seconds for FP8, so FP8 is
  9.286% slower despite its capacity benefit.  The eight actor logs contain
  48 missing-LoRA-table messages: all six required files are absent per actor.
- V74 has four right-sized FP8 actors.  All four emit the DeepGemm accuracy
  warning and all four subsequently log that DeepGemm E8M0 is enabled.  This
  is unsafe evidence and cannot seed a promoted arm.
- V75 has four FP8 actors, each with a fresh empty tuning directory,
  `VLLM_USE_DEEP_GEMM=0`, and FlashInfer autotune disabled.  All four skip
  FlashInfer autotuning and none logs E8M0 enabled, but all four still emit the
  DeepGemm warning.  Installed `config/vllm.py` checks
  `quant_config.use_deep_gemm is None` and architecture compatibility without
  consulting the environment in that warning branch.  Therefore V75 does not
  satisfy a zero-warning environment gate.
- V76 uses the process-local ordering workaround: every actor records
  `VLLM_USE_DEEP_GEMM=0`, `quant_config.use_deep_gemm=false` before post-init,
  and no upstream-source modification.  The four logs contain zero DeepGemm
  accuracy warnings, zero E8M0-enable messages, and four explicit
  `Using TRITON Fp8 MoE backend` messages.
- Each V76 actor's live module audit finds 40 `RoutedExperts` owners using
  `Fp8MoEMethod`, `Fp8MoeBackend` enum `TRITON`, `TritonExperts`, and the
  `FusedMoEModularMethod` runtime wrapper.  It also binds 40
  `FusedMoE3DWithLoRA` runners and 80 LoRA quantization references.  The V76
  run bundle hash is
  `5124652dc91af81de6e55c66d5eaa6b8a6b355a85f50369052d960eb5c028d87`.
- A legacy receipt field says a CUTLASS path was requested.  It is explicitly
  rejected as backend evidence because both the live audit and runtime log
  identify routed MoE as TRITON.  The linear-layer log selecting a CUTLASS FP8
  matmul likewise does not identify the routed-MoE backend.
- Every V75 actor falls back for dense shrink/expand and fused W13/W2
  shrink/expand.  Every actor also JIT-compiles `_lora_shrink_kernel`,
  `_lora_expand_kernel`, `_fused_moe_lora_one_shot_kernel`, and
  `fused_moe_kernel` during inference.
- Installed sources and versions are sealed as vLLM 0.25.0, torch 2.11.0,
  Triton 3.6.0, and safetensors 0.8.0.  The nine-source bundle hash is
  `28608e25464cd7fc6dc69166c3f13c34d43d9a48f9b24d8e8b43ecdb81b06823`.
- The rejected V29 selected FP8 base-MoE table is pinned by path plus file,
  content, and loaded-config hashes.  Its prior full-model test failed all five
  latency endpoints (global speedup 0.98138x) and it is rejected from every
  baseline, search seed, bundle, and production arm.

## Tuning contract

The harness covers six exact vLLM filenames: dense `SHRINK`, dense
`EXPAND_TRUE`, and W13/W2 fused-MoE-LoRA shrink/expand.  Dense cases come from
the exact rank-32 adapter header and Qwen3.5 packed-module map; they cover input
dimensions 512/2048/4096, output dimensions 32/256/512/2048/8192, and one,
two, or three packed slices.  Fused cases bind hidden size 2048, rank 32, MoE
intermediate size 512, 256 experts, top-k 8, W13 two slices, and W2 one slice.

The config hierarchy deliberately reproduces an installed-source quirk.  Only
the operation named exactly `shrink` indexes `(K=hidden, N=rank)`.  All other
names—including fused shrink—index `(K=rank, N=hidden)`.  `num_slices` must
match exactly, while max LoRAs, M, K, N, and MoE intermediate size use vLLM's
nearest-integer-key behavior.  Exact leaves are preregistered for
M = 1, 2, 4, 8, 16, 32, 64, 68, 128, 256, 512, 1024, 2048, 4096, 8192, and
16384.

BF16 and FP8 use separate content-addressed config directories because vLLM's
LoRA filenames do not encode precision.  Each paired baseline starts in a new
empty `VLLM_TUNED_CONFIG_FOLDER`.  The tuned directory contains only the six
LoRA tables; no base-MoE table is allowed, which freezes base-MoE behavior and
prevents V29 leakage.  The base fused-MoE kernel is still required in sealed
warmup so no measurement-phase JIT remains.

With the V76 environment identity held fixed, an independently authorized
selection run proceeds in this order:

1. Attest a fresh child process, explicit DeepGemm disable, zero DeepGemm
   warning, `quant_config.use_deep_gemm=false` before engine start,
   `moe_backend=triton`, the exact V76 routed-FP8 implementation identity, and
   disabled FlashInfer autotune.  DeepGemm state cannot stand in for the
   routed-backend audit.
2. Measure explicit empty/default before search.
3. Tune every operation/shape/M leaf with correctness first and at least 100
   steady iterations per candidate.  Deterministic tie breaks use p95 time,
   workspace, registers, then canonical config order.
4. Freeze the six files and cache, then run at least three counterbalanced
   four-GPU default/tuned pairs per precision on GPUs 0–3.  Causal claims are
   within precision only.
5. Require median aggregate token-throughput speedup >=1.02x and paired 95%
   lower bound >=1.0x; peak-VRAM ratio <=1.01; complete utilization,
   memory-activity, power, VRAM, and tokens/s telemetry; unchanged cache; and
   zero inference JIT, missing-config, fallback, or DeepGemm-warning messages.
6. Report exact token hashes for every pair.  After selection freeze, require
   paired source-disjoint semantic delta >=-0.001 with 95% lower bound
   >=-0.002, then a one-shot protected OOD lower bound >=-0.005, worst-stratum
   point delta >=-0.01, and zero new safety failures.  No retuning follows the
   protected result.

## Validation

Focused validation completed without importing torch or vLLM in the builder
and without GPU, dataset, protected-evaluation, training, checkpoint, config
promotion, or site-package access beyond read-only source inspection.

```text
es-at-scale/.venv/bin/python -m pytest -q test_qwen36_fused_moe_lora_tuning_v77.py
24 passed in 0.49s

python3 build_qwen36_fused_moe_lora_tuning_preregistration_v77.py --check
status=cpu_preregistration_complete_launch_blocked

python3 -m py_compile qwen36_fused_moe_lora_tuning_v77.py \
  build_qwen36_fused_moe_lora_tuning_preregistration_v77.py \
  test_qwen36_fused_moe_lora_tuning_v77.py

git diff --check -- qwen36_fused_moe_lora_tuning_v77.py \
  build_qwen36_fused_moe_lora_tuning_preregistration_v77.py \
  test_qwen36_fused_moe_lora_tuning_v77.py \
  experiments/eggroll_es_hpo/preregistrations/qwen36_fused_moe_lora_kernel_tuning_v77.json
```
