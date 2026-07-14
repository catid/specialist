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
generic kernels element-for-element. This is necessary but not sufficient to
activate the directory in training. Loader selection and end-to-end model
throughput must be checked separately, and a frozen experiment must bind the
directory and file hashes before setting `VLLM_TUNED_CONFIG_FOLDER`.
