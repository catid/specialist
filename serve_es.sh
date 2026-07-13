#!/bin/bash
# Launch one identical BF16 TP=1 ES population replica per visible GPU.
# Usage: ./serve_es.sh [gpu_index]   (no arg = every detected GPU)
set -euo pipefail
cd /home/catid/specialist
source /home/catid/specialist/.venv/bin/activate
MODEL=${ES_MODEL:-/home/catid/specialist/models/Qwen3.6-35B-A3B}
MEM_FRACTION=${ES_MEM_FRACTION:-0.88}
pids=()
launch_one() {
  local g=$1
  CUDA_VISIBLE_DEVICES=$g python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port $((30001 + g)) --host 127.0.0.1 \
    --tp-size 1 --mem-fraction-static "$MEM_FRACTION" \
    > /home/catid/specialist/server_es$g.log 2>&1 &
  pids+=("$!")
  echo "GPU $g -> port $((30001 + g)) (pid $!, model $MODEL)"
}
trap 'kill "${pids[@]}" 2>/dev/null || true; wait || true' INT TERM EXIT
if [ "${1:-}" != "" ]; then
  launch_one "$1"
else
  mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  if [ "${#gpus[@]}" -eq 0 ]; then
    echo "No NVIDIA GPUs detected" >&2
    exit 1
  fi
  for g in "${gpus[@]}"; do launch_one "$g"; done
fi
wait
