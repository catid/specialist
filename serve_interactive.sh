#!/bin/bash
# Best single-request latency config: 434 tok/s @ 1024in/1024out (2026-07-10)
# TP=4 + MTP speculative decoding + patched symm-mem allreduce + cutlass fp8 gemm
cd /home/catid && source /home/catid/specialist/.venv/bin/activate
MODEL=/home/catid/specialist/models/Qwen3.6-35B-A3B-FP8
exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --port 30000 --host 127.0.0.1 \
  --tp-size 4 \
  --enable-torch-symm-mem \
  --fp8-gemm-backend cutlass \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path "$MODEL" \
  "$@"
