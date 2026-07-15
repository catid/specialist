#!/usr/bin/env python3
"""Build the immutable prelaunch contract for the V40A LoRA topology probe."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_lora_topology_probe_v40a as runtime


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "v37_lora_topology_probe_v40a.json"
).resolve()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return result


def build() -> dict:
    config = json.loads((runtime.ADAPTER / "adapter_config.json").read_text())
    tuned = json.loads(runtime.TUNED_FILE.read_text())
    paths = {
        "runtime": Path(runtime.__file__).resolve(),
        "worker": ROOT / "eggroll_es_worker_lora_topology_v40a.py",
        "adapter_weights": runtime.ADAPTER_FILE,
        "adapter_config": runtime.ADAPTER / "adapter_config.json",
        "staged_adapter_weights": runtime.STAGED_ADAPTER_FILE,
        "staged_adapter_config": runtime.STAGED_ADAPTER / "adapter_config.json",
        "stage_manifest": runtime.STAGE_MANIFEST,
        "stage_runtime": ROOT / "stage_v37_adapter_for_vllm_v40a.py",
        "model_config": runtime.MODEL / "config.json",
        "model_index": runtime.MODEL / "model.safetensors.index.json",
        "tuned_table": runtime.TUNED_FILE,
        "base_runtime": ROOT / "train_eggroll_es_specialist.py",
        "cleanup_runtime": ROOT / "run_eggroll_es_equal_unit_v38a.py",
    }
    bindings = {key: runtime.file_sha256(path) for key, path in paths.items()}
    index = json.loads((runtime.MODEL / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    if len(shard_names) != 26 or not all(
        (runtime.MODEL / name).is_file() for name in shard_names
    ):
        raise RuntimeError("v40a base model shard inventory changed")
    # This is the content hash of the 26-shard manifest already sealed before
    # V39's failed pre-data launch; V40 binds that value in its own source hash.
    bindings["model_shards_content_sha256"] = runtime.MODEL_SHARDS_CONTENT_SHA256
    if (
        config.get("r") != 32 or config.get("lora_alpha") != 64
        or config.get("bias") != "none"
    ):
        raise RuntimeError("v40a exact V37 adapter configuration changed")
    value = {
        "schema": "lora-topology-preregistration-v40a",
        "status": "preregistered_before_gpu_launch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Map the exact V37A PEFT tensor surface into resident vLLM LoRA "
            "GPU buffers and prove reversible in-place forward effect."
        ),
        "implementation_bindings": bindings,
        "runtime": {
            "physical_gpu_ids": list(runtime.GPU_IDS),
            "engine_count": 4, "tensor_parallel_size": 1,
            "all_four_actors_concurrent": True,
            "model": str(runtime.MODEL),
            "adapter": str(runtime.ADAPTER),
            "staged_adapter": str(runtime.STAGED_ADAPTER),
            "adapter_rank": 32, "adapter_alpha": 64,
            "enable_lora": True, "max_loras": 1, "max_cpu_loras": 1,
            "lora_dtype": "auto_from_bfloat16_base",
            "tuned_folder": str(runtime.TUNED_FOLDER),
            "tuned_table_content_sha256": runtime.canonical_sha256(tuned),
            "synthetic_prompt_sha256": __import__("hashlib").sha256(
                runtime.SYNTHETIC_PROMPT.encode()
            ).hexdigest(),
        },
        "probe": {
            "peft_tensor_count_expected": 70,
            "peft_elements_expected": 4_528_128,
            "mapping": "PEFT key to wrapper, slot, packed slice and exact GPU view",
            "namespace_stage": (
                "key-only exact-value rename into Qwen3.5 vLLM language_model namespace"
            ),
            "base_audit": "hash every relevant wrapper base_layer.weight before/after",
            "mutation": {
                "target": "model.layers.23.self_attn.o_proj.lora_B[0,0]",
                "requested_delta": 1.0,
                "one_element_only": True,
                "restore_exact_original_value": True,
                "forward_without_adapter_reload_required": True,
            },
        },
        "artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "report": str(runtime.REPORT),
            "gpu_log": str(runtime.GPU_LOG),
        },
        "dataset_or_evaluation_access_authorized": False,
        "dataset_shadow_ood_holdout_or_heldout_paths_bound": False,
        "synthetic_prompt_only": True,
        "no_quality_claim_authorized": True,
    }
    value["content_sha256_before_self_field"] = runtime.canonical_sha256(value)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    runtime.atomic_json(output, build())
    print(json.dumps({
        "path": str(output), "file_sha256": runtime.file_sha256(output),
        "content_sha256": json.loads(output.read_text())[
            "content_sha256_before_self_field"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
