#!/bin/bash
# Best batch-throughput config for HyperscaleES rollouts:
# ~8000 out tok/s (~16k total tok/s) at concurrency 256 with 1024in/1024out (2026-07-10)
# DP=4 x TP=1 (zero inter-GPU comms; each replica holds the full FP8 model),
# no MTP (spec decode loses at batch and complicates in-place ES weight updates).
# NOTE: throughput peaks near concurrency 256; beyond that prefill/decode
# contention degrades it. Feed ~64 concurrent rollouts per GPU.
cd /home/catid && source /home/catid/specialist/.venv/bin/activate
MODEL=/home/catid/specialist/models/Qwen3.6-35B-A3B-FP8
exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --port 30000 --host 127.0.0.1 \
  --tp-size 1 --dp-size 4 \
  --max-running-requests 1024 \
  --fp8-gemm-backend cutlass \
  "$@"
