#!/bin/bash
# Launch SGLang server for Qwen3.6-35B-A3B. Extra args appended verbatim.
# Usage: ./launch.sh [fp8|bf16] [extra args...]
set -e
cd /home/catid/specialist
source .venv/bin/activate
VARIANT=${1:-fp8}; shift || true
if [ "$VARIANT" = "bf16" ]; then
  MODEL=/home/catid/specialist/models/Qwen3.6-35B-A3B
else
  MODEL=/home/catid/specialist/models/Qwen3.6-35B-A3B-FP8
fi
exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --port 30000 --host 127.0.0.1 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path "$MODEL" \
  "$@"
