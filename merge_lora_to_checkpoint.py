#!/usr/bin/env python
"""Merge LoRA adapter deltas directly into the full HF (VL) checkpoint."""
import json, re, shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="/home/catid/specialist/models/lora-rope-sft/final")
ap.add_argument("--dst", default="/home/catid/specialist/models/Qwen3.6-35B-A3B-rope-sft")
a = ap.parse_args()
SRC = Path("/home/catid/specialist/models/Qwen3.6-35B-A3B")
DST = Path(a.dst)
ADAPTER = Path(a.adapter)

cfg = json.load(open(ADAPTER / "adapter_config.json"))
scaling = cfg["lora_alpha"] / cfg["r"]
print(f"lora r={cfg['r']} alpha={cfg['lora_alpha']} scaling={scaling}")

deltas = {}  # hf tensor name -> (A, B)
with safe_open(ADAPTER / "adapter_model.safetensors", framework="pt") as f:
    keys = list(f.keys())
    for k in keys:
        m = re.match(r"base_model\.model\.model\.(.+)\.lora_A\.weight", k)
        if not m:
            continue
        hf = f"model.language_model.{m.group(1)}.weight"
        A = f.get_tensor(k).float()
        B = f.get_tensor(k.replace("lora_A", "lora_B")).float()
        deltas[hf] = (A, B)
print(f"{len(deltas)} target tensors")

DST.mkdir(parents=True, exist_ok=True)
idx = json.load(open(SRC / "model.safetensors.index.json"))
shards = sorted(set(idx["weight_map"].values()))
n_mod = 0
for si, shard in enumerate(shards):
    tensors = {}
    with safe_open(SRC / shard, framework="pt") as f:
        meta = f.metadata()
        for name in f.keys():
            t = f.get_tensor(name)
            if name in deltas:
                A, B = deltas[name]
                t = (t.float() + scaling * (B @ A)).to(t.dtype)
                n_mod += 1
            tensors[name] = t
    save_file(tensors, DST / shard, metadata=meta or {"format": "pt"})
    print(f"shard {si+1}/{len(shards)} ({n_mod} merged so far)", flush=True)
for f in SRC.iterdir():
    if f.suffix in (".json", ".txt", ".jinja"):
        shutil.copy(f, DST / f.name)
print(f"done: {n_mod} tensors merged -> {DST}")
