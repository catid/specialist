#!/usr/bin/env python3
"""Overlay authoritative ES FP32 masters onto a base HF checkpoint.

Masters remain FP32 until this deployment export, then each selected tensor is
cast exactly once to the base tensor's dtype. The output is built in a sibling
temporary directory, verified, and atomically renamed.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masters", required=True)
    parser.add_argument("--src", default="/home/catid/specialist/models/Qwen3.6-35B-A3B")
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    master_path = Path(args.masters)
    masters = {}
    with safe_open(master_path, framework="pt") as source:
        metadata = source.metadata() or {}
        for key in source.keys():
            masters[key] = source.get_tensor(key)
    if not masters:
        raise ValueError("master file contains no tensors")
    print(f"{len(masters)} master tensors ({metadata.get('schema', 'legacy')})")

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")
    temporary = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary destination already exists: {temporary}")
    temporary.mkdir(parents=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    unknown = sorted(set(masters) - set(index["weight_map"]))
    if unknown:
        shutil.rmtree(temporary)
        raise ValueError(f"{len(unknown)} masters are absent from the base: {unknown[:3]}")

    replaced = set()
    try:
        shards = sorted(set(index["weight_map"].values()))
        for shard_index, shard in enumerate(shards, 1):
            tensors = {}
            with safe_open(src / shard, framework="pt") as source:
                shard_metadata = source.metadata() or {"format": "pt"}
                for name in source.keys():
                    base = source.get_tensor(name)
                    if name in masters:
                        master = masters[name]
                        if tuple(master.shape) != tuple(base.shape):
                            raise ValueError(
                                f"shape mismatch for {name}: {tuple(master.shape)} "
                                f"!= {tuple(base.shape)}")
                        tensors[name] = master.to(dtype=base.dtype).contiguous()
                        replaced.add(name)
                    else:
                        tensors[name] = base
            save_file(tensors, temporary / shard, metadata=shard_metadata)
            print(f"shard {shard_index}/{len(shards)} ({len(replaced)} replaced)",
                  flush=True)
        missing = sorted(set(masters) - replaced)
        if missing:
            raise ValueError(f"failed to replace {len(missing)} masters: {missing[:3]}")
        for path in src.iterdir():
            if path.suffix in (".json", ".txt", ".jinja"):
                shutil.copy2(path, temporary / path.name)
        provenance = {
            "schema": "specialist-es-deployment-overlay-v1",
            "masters": str(master_path.resolve()),
            "master_schema": metadata.get("schema", "legacy-bf16"),
            "base": str(src.resolve()),
            "replaced_tensors": len(replaced),
        }
        (temporary / "es_overlay.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        temporary.rename(dst)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"done: {len(replaced)} tensors replaced -> {dst}")


if __name__ == "__main__":
    main()
