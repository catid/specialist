#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/es-at-scale/.venv"

git -C "$ROOT" submodule update --init --recursive es-at-scale
python3.12 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip setuptools wheel
"$VENV/bin/pip" install -r "$ROOT/requirements-eggroll-qwen36.txt"
"$VENV/bin/pip" install --no-deps --editable "$ROOT/es-at-scale"

"$VENV/bin/python" - <<'PY'
import datasets
import ray
import torch
import transformers
import vllm

for module in (datasets, ray, torch, transformers, vllm):
    print(module.__name__, module.__version__)
print("CUDA devices", torch.cuda.device_count())
PY
