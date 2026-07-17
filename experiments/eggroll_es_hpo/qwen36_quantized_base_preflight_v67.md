# Qwen3.6 frozen-base precision preflight V67

Status: CPU-only preparation passed on 2026-07-17. GPU execution remains
ordered after `specialist-0j5.14`; this document does not authorize scored
validation, OOD access, training, or an INT4 launch.

## Safe next GPU arms

Two arms are safe to take into a fail-closed live preflight after the roofline
baseline finishes:

1. **BF16:** `/home/catid/specialist/models/Qwen3.6-35B-A3B`, explicit
   `dtype=bfloat16`, no quantization argument.
2. **Serialized block FP8:**
   `/home/catid/specialist/models/Qwen3.6-35B-A3B-FP8`, explicit
   `dtype=bfloat16, quantization=fp8`. The checkpoint declares dynamic E4M3
   activation quantization and a `[128, 128]` weight block.

Both use TP1, the same BF16/FP32 LoRA state, the same prompts and generation
settings, and `kv_cache_dtype=auto`. The only allowed engine differences are
the model path and quantization argument. The initial crossover is BF16 on
GPUs 0/2 and FP8 on GPUs 1/3, followed by the reverse assignment, so all four
GPUs are occupied while physical-GPU and order effects remain separable.

Neither arm may be scored until a four-actor runtime receipt proves the exact
resolved format, load/generate success, kernel methods, adapter switching,
candidate output effect, exact restoration, unchanged FP32 master and frozen
base hashes, measurements, consensus, and simultaneous GPU activity.

## What is actually sealed

The CPU preflight checks checkpoint headers without importing torch or vLLM:

| Surface | BF16 | Serialized FP8 |
|---|---:|---:|
| Indexed tensors | 1,045 | 64,196 |
| Shards | 26 | 42 |
| Logical tensor bytes | 71,903,645,408 | 37,454,789,472 |
| BF16 tensors | 1,045 | 32,451 |
| FP8 E4M3 tensors | 0 | 31,745 |
| FP8 routed-expert matrices | 0 | 31,488 |
| BF16 inverse scales | 0 | 31,745 |
| BF16 router matrices | 41 | 41 |

Config, index, shard-size, and tensor metadata manifests are cryptographically
sealed. The installed stack is vLLM 0.25.0, torch 2.11.0, transformers 5.13.1,
and compressed-tensors 0.17.0. Relevant Qwen, FP8, AWQ, GPTQ, bitsandbytes,
compressed-tensors, fused-MoE, and LoRA source files are also pinned by hash.

The Qwen class inherits `SupportsLoRA`, and its MoE wrapper declares 3D MoE
LoRA weights. That is static API evidence only; it does not substitute for the
live switch/restore probe on each weight format.

## VRAM, KV cache, and bandwidth expectation

Serialized FP8 removes 34,448,855,936 logical bytes relative to BF16, or
47.91% of the BF16 checkpoint tensor footprint. Most of the saving is in the
256 routed experts per layer, which is also the surface most likely to be
limited by weight reads. This is therefore a plausible material VRAM and
memory-bandwidth improvement, but logical checkpoint bytes are not CUDA
allocator measurements. Dynamic activation quantization, scales, kernel
workspace, LoRA buffers, and vLLM's allocator all affect the live result.

Weight precision does not change KV precision in this ablation. With ten full
attention layers, two KV heads, a head dimension of 256, and BF16 K/V, the
nominal attention KV payload is 20,480 bytes per token in either arm before
block/page overhead. FP8 should instead turn freed weight memory into more KV
blocks at fixed `gpu_memory_utilization=0.82`. Qwen's 30 linear-attention
layers also carry sequence state, and vLLM aligns/pads hybrid cache pages, so
available KV memory and block counts must be read from the live engine rather
than inferred from the nominal number.

The live comparison records post-load and peak VRAM, available KV memory,
GPU/CPU block counts, prefill/decode throughput, switch/restore latency, DRAM
traffic or the best available bandwidth proxy, and SM/memory-controller
utilization.

## Kernel-table surprise

The prior V29H full-model experiment is bound into the preregistration. Its
synthetically selected FP8 MoE table preserved exact outputs and VRAM but
failed all five latency endpoints; the global tuned-over-default geometric
speedup was 0.98138. The FP8 arm must therefore start with vLLM's empty/default
table. It may not reuse the BF16 table or the rejected selected FP8 table.

## Why INT4 is not yet safe

vLLM 0.25 has real routed-expert code for several 4-bit routes, but the local
environment does not yet have an executable arm:

- No local Qwen3.6 AWQ, GPTQ, compressed-tensors W4A16, or bitsandbytes INT4
  checkpoint exists.
- Bitsandbytes is not installed; this CUDA stack requires at least 0.48.1.
- Its installed MoE implementation dequantizes routed-expert weights before a
  generic `fused_experts` call on the hot path, which is a serious temporary
  memory and bandwidth risk even after installing it.
- AWQ and GPTQ can change MoE backends to WNA16 when Marlin geometry is not
  supported. Compressed-tensors also has multiple W4A16 kernel choices. Those
  may be valid implementations, but accepting an unobserved backend change
  would violate this experiment's fail-closed comparison.
- LoRA plus Qwen's 3D MoE wrapper has not been exercised on any of those INT4
  kernels here.

The preferred first future INT4 candidate is a serialized
compressed-tensors W4A16 checkpoint because its dependency and routed-expert
loader are already present. It still needs its own artifact seal and live
kernel/LoRA preregistration. AWQ and GPTQ are secondary candidates.
Bitsandbytes online NF4/FP4 should be tested last, if at all. MXFP4/NVFP4 is a
different floating-point format and is not silently relabeled as INT4.

## Fail-closed rules

The preflight rejects changed checkpoint manifests, package/source drift,
online conversion in the serialized-FP8 arm, any INT4 artifact that appears
without a new preregistration, fallback log messages, unplanned unquantized
modules, a different resolved quantizer, a different FP8 block shape, missing
measurements, adapter/base hash drift, failed exact restoration, or missing
four-GPU activity. Unit tests inject each major class of bad receipt.

Artifacts:

- `build_qwen36_quantized_base_ablation_preregistration_v67.py`
- `experiments/eggroll_es_hpo/preregistrations/qwen36_quantized_base_ablation_v67.json`
- `test_build_qwen36_quantized_base_ablation_preregistration_v67.py`

CPU verification: 11 focused tests passed. No CUDA context or GPU allocation
was created.
