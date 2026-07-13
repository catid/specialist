#!/usr/bin/env python
"""Replay an ES journal through persistent FP32 masters into an HF checkpoint.

Each tensor stays FP32 across every journal generation and is cast once at
deployment export, matching the live server's authoritative-master semantics.
Per-generation target regexes are honored; ambiguous legacy targets fail safe.

Usage: python es_make_checkpoint.py --journal es_journal.jsonl \
           --src models/Qwen3.6-35B-A3B --dst models/Qwen3.6-35B-A3B-rope
"""
import argparse, json, shutil, sys
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "/home/catid/specialist/sglang/python")
from sglang.srt.managers.scheduler_components.es_perturb import (
    apply_commits_to_hf_tensor,
)
sys.path.insert(0, "/home/catid/specialist")
from es_replay_to_servers import prepare_replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default="/home/catid/specialist/es_journal.jsonl")
    ap.add_argument("--src", default="/home/catid/specialist/models/Qwen3.6-35B-A3B")
    ap.add_argument("--dst", default="/home/catid/specialist/models/Qwen3.6-35B-A3B-rope")
    legacy = ap.add_mutually_exclusive_group()
    legacy.add_argument("--legacy-include-regex")
    legacy.add_argument("--allow-legacy-full-model", action="store_true")
    args = ap.parse_args()

    prepared = prepare_replay(
        args.journal, legacy_include_regex=args.legacy_include_regex,
        allow_legacy_full_model=args.allow_legacy_full_model)
    gens = [{"gen": generation, "ops": ops, "sigma": sigma, "rank": rank,
             "include_regex": include_regex}
            for generation, ops, sigma, rank, include_regex in prepared]
    print(f"replaying {len(gens)} generations")
    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")
    temporary = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary destination already exists: {temporary}")
    temporary.mkdir(parents=True)

    idx = json.load(open(src / "model.safetensors.index.json"))
    shards = sorted(set(idx["weight_map"].values()))
    n_mod = 0
    try:
        for si, shard in enumerate(shards):
            tensors = {}
            with safe_open(src / shard, framework="pt") as f:
                meta = f.metadata()
                for name in f.keys():
                    t = f.get_tensor(name)
                    if t.dtype == torch.bfloat16 and t.dim() in (2, 3):
                        tg = t.cuda()
                        if apply_commits_to_hf_tensor(name, tg, gens):
                            t = tg.cpu()
                            n_mod += 1
                        del tg
                    tensors[name] = t
            save_file(tensors, temporary / shard,
                      metadata=meta or {"format": "pt"})
            print(f"shard {si+1}/{len(shards)} done "
                  f"({n_mod} tensors modified so far)", flush=True)
        if not n_mod:
            raise ValueError("journal target matched zero checkpoint tensors")
        for f in src.iterdir():
            if (f.suffix in (".json", ".txt", ".jinja") and
                    f.name != "model.safetensors.index.json"):
                shutil.copy2(f, temporary / f.name)
        shutil.copy2(src / "model.safetensors.index.json",
                     temporary / "model.safetensors.index.json")
        temporary.rename(dst)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"done: {n_mod} tensors modified -> {dst}")


if __name__ == "__main__":
    main()
