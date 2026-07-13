#!/bin/bash
# Hyperparameter sweep: short ES runs from a fixed checkpoint, each config
# measured on domain qa_probe gain vs OOD degradation (logged by es_train).
# Usage: ./hp_sweep.sh <checkpoint_dir> <gens_per_config>
set -e
CKPT=${1:-/home/catid/specialist/models/Qwen3.6-35B-A3B-rope-v2}
GENS=${2:-30}
GPUS=(${SWEEP_GPUS:-0 1 2 3})
PORTS=(); for g in "${GPUS[@]}"; do PORTS+=($((30001 + g))); done
# pairs must be divisible by server count to co-locate antithetic pairs
PAIRS=${SWEEP_PAIRS:-16}
cd /home/catid/specialist

# config format: "tag sigma lr rank"
CONFIGS=(
  "lr4      0.01  0.04  4"    # run-2 setting (control)
  "lr8      0.01  0.08  4"    # 2x step size
  "sig2lr8  0.02  0.08  4"    # bigger noise + matched step
  "rank16   0.01  0.04  16"   # richer noise subspace
)

for cfg in "${CONFIGS[@]}"; do
  read -r TAG SIGMA LR RANK <<< "$cfg"
  echo "=== config $TAG sigma=$SIGMA lr=$LR rank=$RANK ==="
  # fresh servers from the sweep checkpoint (only on the sweep GPUs)
  for p in "${PORTS[@]}"; do
    kill $(pgrep -f "port $p --host" | head -1) 2>/dev/null || true
  done
  sleep 8
  for g in "${GPUS[@]}"; do
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -ge 1000 ]; do sleep 2; done
    ES_MODEL=$CKPT nohup ./serve_es.sh "$g" > /dev/null 2>&1 &
  done
  for p in "${PORTS[@]}"; do
    until curl -sf http://127.0.0.1:$p/health_generate -o /dev/null 2>/dev/null; do sleep 5; done
  done
  rm -f "es_journal_hp_$TAG.jsonl"
  source .venv/bin/activate
  python es_train.py --gens "$GENS" --pairs "$PAIRS" --sigma "$SIGMA" --lr "$LR" \
    --rank "$RANK" --chunks-per-eval 16 --eval-every 5 \
    --ports "${PORTS[@]}" \
    --data data/train_mix.jsonl \
    --journal "es_journal_hp_$TAG.jsonl" > "es_train_hp_$TAG.log" 2>&1
  echo "--- $TAG done; final evals:"
  grep -E "qa_probe" "es_train_hp_$TAG.log" | tail -2
done
echo ALL_CONFIGS_DONE
