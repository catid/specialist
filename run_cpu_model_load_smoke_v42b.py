#!/usr/bin/env python3
"""Real CPU-only Qwen3.5-MoE model-load smoke for the V42B adapter path."""

from __future__ import annotations

import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# This evidence process is intentionally incapable of enumerating a GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import peft
import torch
import transformers
from transformers import AutoModelForCausalLM

import run_sft_train_only_control_v36a as engine
import sft_lora as base
import sft_lora_equal_unit_matched_init_v42a as contract
import sft_lora_equal_unit_matched_init_v42b as retry


ROOT = Path(__file__).resolve().parent
REPORT = (
    ROOT / "experiments/sft_controls/"
    "v42b_matched_init_equal_unit_fold3_v412_retry_direct_load/"
    "cpu_model_load_smoke_v42b.json"
).resolve()


def run_smoke_v42b() -> dict:
    if torch.cuda.is_initialized() or torch.cuda.device_count() != 0:
        raise RuntimeError("V42B CPU smoke did not begin in a CPU-only process")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        base.BASE,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cpu",
    )
    loaded = retry.ExactCanonicalAdapterLoaderV42B.from_pretrained(
        model,
        contract.INITIAL_ADAPTER_V42A,
        is_trainable=True,
        autocast_adapter_dtype=True,
    )
    state = retry._canonical_adapter_state_v42b(loaded)
    source = contract.validate_initialization_artifact_v42a(
        contract.INITIAL_ADAPTER_V42A
    )
    readback = contract.validate_loaded_adapter_state_v42a(
        state, contract.INITIAL_ADAPTER_V42A, source
    )
    trainable = [
        {"name": name, "elements": parameter.numel()}
        for name, parameter in loaded.named_parameters()
        if parameter.requires_grad
    ]
    inventory = {
        "target_layers": [20, 21, 22, 23],
        "tensor_count": len(trainable),
        "elements": sum(item["elements"] for item in trainable),
        "identity_sha256": contract.canonical_sha256_v42a(trainable),
    }
    if inventory != engine.EXPECTED_TRAINABLE_INVENTORY:
        raise RuntimeError("V42B real CPU smoke trainable inventory changed")
    devices = sorted({parameter.device.type for parameter in loaded.parameters()})
    if (
        devices != ["cpu"]
        or torch.cuda.is_initialized()
        or torch.cuda.device_count() != 0
    ):
        raise RuntimeError("V42B real CPU smoke touched a non-CPU device")
    report = {
        "schema": "specialist-real-cpu-model-load-smoke-v42b",
        "status": "complete_exact_load_verified",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - started,
        "gpu_accessed": False,
        "cuda_initialized": False,
        "cuda_visible_devices": "",
        "visible_cuda_device_count": 0,
        "dataset_or_training_examples_accessed": False,
        "shadow_ood_holdout_or_heldout_accessed": False,
        "evaluation_performed": False,
        "device_types": devices,
        "model": {
            "path": str(Path(base.BASE).resolve()),
            "config_sha256": engine.file_sha256(
                Path(base.BASE) / "config.json"
            ),
            "index_sha256": engine.file_sha256(
                Path(base.BASE) / "model.safetensors.index.json"
            ),
        },
        "source_initialization": source,
        "adapter_loader": retry.expected_loader_audit_v42b(),
        "loaded_readback": readback,
        "trainable_inventory": inventory,
        "implementation": {
            "smoke": str(Path(__file__).resolve()),
            "smoke_sha256": engine.file_sha256(Path(__file__).resolve()),
            "retry_sft": str(Path(retry.__file__).resolve()),
            "retry_sft_sha256": engine.file_sha256(Path(retry.__file__).resolve()),
            "source_contract": str(Path(contract.__file__).resolve()),
            "source_contract_sha256": engine.file_sha256(
                Path(contract.__file__).resolve()
            ),
            "torch_version": torch.__version__,
            "peft_version": peft.__version__,
            "transformers_version": transformers.__version__,
        },
    }
    del state, loaded, model
    gc.collect()
    return engine.self_hashed(report)


def main() -> int:
    if REPORT.exists():
        raise RuntimeError("V42B CPU smoke report already exists")
    report = run_smoke_v42b()
    engine.atomic_write_json(REPORT, report)
    print(json.dumps({
        "report": str(REPORT),
        "file_sha256": engine.file_sha256(REPORT),
        "content_sha256": report["content_sha256_before_self_field"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
