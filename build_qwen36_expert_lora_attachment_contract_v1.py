#!/usr/bin/env python3
"""Build the data-free Qwen3.6 expert-LoRA attachment contract on meta tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models/Qwen3.6-35B-A3B"
ARCHITECTURE_CONTRACT = (
    ROOT / "training_protocol/qwen36_architecture_contract_v1.json"
)
OUTPUT = ROOT / "training_protocol/qwen36_expert_lora_attachment_contract_v1.json"
SCHEMA = "qwen36-expert-lora-attachment-contract-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable_lora_config(config: Any) -> dict[str, Any]:
    value = config.to_dict()
    for key in ("target_modules", "target_parameters"):
        if value.get(key) is not None:
            value[key] = sorted(value[key])
    value["rank_pattern"] = dict(sorted(value.get("rank_pattern", {}).items()))
    value["alpha_pattern"] = dict(sorted(value.get("alpha_pattern", {}).items()))
    return value


def construct() -> dict[str, Any]:
    # This must be set before importing torch/Transformers.  The audit is
    # deliberately incapable of initializing CUDA or loading checkpoint weights.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import qwen36_expert_lora_v1 as expert_lora
    import peft
    import torch
    import transformers
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as modeling

    _require(not torch.cuda.is_available(), "meta attachment audit unexpectedly sees CUDA")
    architecture = expert_lora.load_architecture_contract(ARCHITECTURE_CONTRACT)
    spec = expert_lora.spec_from_contract(architecture)
    config_path = MODEL_ROOT / "config.json"
    config = AutoConfig.from_pretrained(MODEL_ROOT, local_files_only=True)

    # The optional fused norm constructor asks for a live CUDA device even on a
    # meta model.  Norms are outside the adapter target set, so use the native
    # Qwen norm solely for this shape/ownership audit.
    modeling.FusedRMSNormGated = None
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    _require(type(model).__name__ == "Qwen3_5MoeForCausalLM", "text model dispatch changed")
    preattach = expert_lora.validate_preattach_model(model, spec)
    lora_config = expert_lora.make_lora_config(spec)
    adapted = get_peft_model(model, lora_config)
    postattach = expert_lora.audit_postattach_scope(adapted, spec)
    _require(postattach["target_count"] == 200, "LoRA target count changed")
    _require(postattach["trainable_tensor_count"] == 400, "LoRA tensor count changed")
    _require(postattach["trainable_elements"] == 235_601_920, "LoRA element count changed")
    _require(
        {row["dtype"] for row in postattach["trainable_parameters"]} == {"torch.float32"},
        "PEFT adapter dtype policy changed",
    )

    result = {
        "schema": SCHEMA,
        "purpose": "prove exact expert-aware MLP-only PEFT attachment before weight loading or training",
        "status": "complete_data_free_meta_attachment",
        "authority": {
            "cuda_visible": False,
            "checkpoint_config_loaded": True,
            "checkpoint_weights_loaded": False,
            "dataset_or_training_rows_opened": False,
            "protected_holdout_ood_terminal_incident_or_manual_review_opened": False,
            "optimizer_created": False,
            "training_launched": False,
            "weight_or_adapter_update_performed": False,
        },
        "architecture_contract": {
            "path": ARCHITECTURE_CONTRACT.relative_to(ROOT).as_posix(),
            "file_sha256": file_sha256(ARCHITECTURE_CONTRACT),
            "content_sha256": architecture["content_sha256_before_self_field"],
        },
        "checkpoint_config": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "file_sha256": file_sha256(config_path),
            "model_class": type(model).__name__,
            "model_type": config.model_type,
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "meta_device_only": True,
            "fused_norm_replaced_only_for_meta_construction": True,
        },
        "lora_config": _jsonable_lora_config(lora_config),
        "preattach": preattach,
        "postattach": postattach,
        "acceptance": {
            "literal_full_name_targeting": True,
            "shared_expert_target_count": postattach["shared_target_count"],
            "routed_expert_target_count": postattach["routed_target_count"],
            "both_fused_parameters_per_expert_module_attached": True,
            "trainable_tensor_count": postattach["trainable_tensor_count"],
            "trainable_elements": postattach["trainable_elements"],
            "only_lora_a_and_b_tensors_trainable": True,
            "dropout_zero_for_every_wrapper": True,
            "classic_alpha_over_rank_scaling_for_every_wrapper": True,
            "bias_rslora_and_dora_disabled_for_every_wrapper": True,
            "requested_adapter_active_enabled_and_unmerged": True,
            "router_attention_mixing_embedding_head_norm_vision_and_mtp_trainable": False,
            "adapter_parameters_created_in_fp32": True,
        },
        "builder": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "file_sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    result["content_sha256_before_self_field"] = expert_lora.canonical_sha256(result)
    return result


def render() -> str:
    return json.dumps(construct(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    payload = render()
    if arguments.check:
        _require(OUTPUT.read_text(encoding="utf-8") == payload, "attachment contract changed")
    else:
        _atomic_write(OUTPUT, payload.encode("utf-8"))
    value = json.loads(payload)
    print(
        json.dumps(
            {
                "output": OUTPUT.relative_to(ROOT).as_posix(),
                "content_sha256": value["content_sha256_before_self_field"],
                "targets": value["postattach"]["target_count"],
                "trainable_tensors": value["postattach"]["trainable_tensor_count"],
                "trainable_elements": value["postattach"]["trainable_elements"],
                "training_launched": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
