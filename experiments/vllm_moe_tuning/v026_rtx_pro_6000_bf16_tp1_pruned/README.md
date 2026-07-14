# RTX PRO 6000 BF16 fused-MoE pruned configuration

This directory contains an opt-in vLLM 0.25.0 configuration for the
Qwen3.6-35B-A3B routed-expert shape on the exact Max-Q workstation device
name. It is not enabled by default and was not used by V14b.

The search was deliberately bounded around vLLM's generic default and its
official H100 BF16 configuration. Each candidate changed one tile field at a
time. This avoided the pathological tail observed in the exhaustive 1,920
configuration search. All comparisons were paired on the same GPU.

The selected kernels were also run on identical inputs, weights, gates, and
routes at batches 1, 128, 1024, and 4096. Their BF16 outputs matched the
generic kernels element-for-element.

The exact device-specific loader was then verified on four TP=1 workers. An
end-to-end Triton-backend Qwen3.6 comparison used 256 fixed synthetic prompts
and greedy 32-token generation on every physical GPU. Tuned and default runs
produced the same generated text/token payload hash. Median generation time
fell from 2.4645 to 2.1055 seconds, a 1.1705x speedup; the worst physical-GPU
speedup was 1.1647x. Model load, compile, warmup, and graph capture were
outside the timing boundary.

This remains insufficient to activate the directory in an already-frozen
experiment. A future recipe must bind the directory and file hashes, isolate
kernel tuning from model/data HPO, and recheck task-level timing before setting
`VLLM_TUNED_CONFIG_FOLDER`.
