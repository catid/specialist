#!/bin/bash
# 1024in/1024out throughput benchmark against a running SGLang server.
# Usage: ./bench.sh [port] [num_prompts] [concurrency]
PORT=${1:-30000}
N=${2:-5}
CONC=${3:-1}
TOKENIZER=${TOKENIZER:-/home/catid/specialist/models/Qwen3.6-35B-A3B-FP8}
source /home/catid/specialist/.venv/bin/activate
cd "$HOME"  # avoid ~/specialist: the sglang/ checkout there shadows the package
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port "$PORT" \
  --tokenizer "$TOKENIZER" \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 \
  --random-range-ratio 1.0 \
  --num-prompts "$N" \
  --max-concurrency "$CONC"
