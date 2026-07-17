# Qwen3.6 TP/EP layout V82 CPU evidence

V82 admits exactly four live arms after the prerequisite Beads close: four replicated TP1 actors, one tensor-sharded TP4 engine, TP-only EP4, and DP4/TP1 wide EP4 with vLLM's built-in allgather/reducescatter backend. The builder did not import vLLM or torch, launch a model, open dataset rows, or use a GPU.

## Support findings

The installed vLLM is source-bound to 0.25.0. Its Qwen3.5-MoE implementation exposes LoRA, TP-aware attention/Mamba state, and FusedMoE expert parallelism. The checkpoint has 256 experts, so EP4 has an exact static partition of 64 experts per rank.

A non-obvious limitation is KV scaling: Qwen has 2 full-attention KV heads. TP4 therefore retains 1 KV head per rank, versus 2 at TP1. Full-attention KV elements per rank are expected to fall by only 2x, not 4x. The live receipt must separately attest hybrid attention and Mamba-state bytes.

vLLM's all-to-all distinction is also important. TP-only EP4 does not activate its all-to-all kernels; it uses TP collectives around rank-local experts. DP4/TP1 wide EP does activate allgather/reducescatter. V82 therefore measures both instead of mislabeling TP-only EP traffic as all-to-all.

The admitted wide-EP arm installs one globally identical candidate on all four ranks and partitions the 64 prompts. The tempting variant that keeps four different one-slot candidates concurrently is rejected: expert routing would mix candidate states unless a separately sealed global multi-LoRA implementation carries adapter identity end to end.

## Rejected configurations

- `dp4_tp1_wide_ep_four_distinct_one_slot`: the sealed recipe concurrently materializes four distinct candidates in one LoRA slot per actor; wide EP routes tokens to ranks holding different candidate values, so candidate identity is not preserved
- `wide_dp4_tp1_ep4_deepep`: DeepEP module is absent; no fallback is permitted
- `wide_dp4_tp1_ep4_nixl`: NIXL module is absent; no fallback is permitted
- `wide_dp4_tp1_ep4_pplx_or_naive`: vLLM 0.25 removes these choices and rewrites them to allgather_reducescatter; silent rewrite is forbidden
- `wide_dp4_tp1_ep4_flashinfer_nvlink`: package presence does not prove topology, FP8-Qwen, LoRA, or runtime-kernel support
- `ep_weight_filter_as_memory_arm`: Qwen3.5-MoE declares 3D fused MoE weights and vLLM documents the pre-read EP filter as ineffective for 3D fused-expert checkpoints; no VRAM or I/O saving may be assumed

## Live comparison contract

All arms consume the same 16 signed V66D candidates, 64 prompts per candidate, decode parameters, tokenized prompt hash, reward, and canonical LoRA master. TP1 evaluates four candidates concurrently; each sharded arm evaluates them serially while all four ranks work. This makes the loss of candidate parallelism part of the measured end-to-end cost rather than hiding it in a per-call microbenchmark.

Three clean timing replicates are required per arm. A separate Nsight Systems replicate records NCCL operation sizes/durations and GPU DRAM activity; NCU HBM byte metrics run only on finalists and never supply timing. Per-rank VRAM, parameter ownership, attention KV, Mamba state, PCIe, NCCL bytes, tokens/s, rollouts/s, and rollouts/GPU-second are all mandatory.

A sharded arm is rejected on any backend rewrite/fallback, incorrect rank ownership, idle GPU, candidate/restore hash mismatch, numerical correctness failure, or point throughput below the replicated TP1 control. A surviving winner still needs the sealed source-disjoint semantic gate and one-shot protected OOD gate before promotion.

Contract content SHA-256: `828bfb4e9a64b0497b12dbce8c840ce9469d7cb6d3d548ec37dfdc048583261e`
