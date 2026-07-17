#!/usr/bin/env python3
"""Data-free BF16/serialized-FP8 LoRA switch preflight for V67.

This is a narrow live diagnostic under the sealed V67 precision contract.  It
reuses the V63 synthetic workload and changes only the frozen model artifact
and resolved quantization.  It does not score data or authorize a precision
arm for training.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import probe_vllm_two_adapter_switch_v63 as base


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_quantized_base_ablation_v67.json"
)
PREREG_FILE_SHA256 = (
    "e1bd808ce102882c0a3c13d4eab337ebce5ed70011ec8eefdbe8a727bbdc2724"
)
PREREG_CONTENT_SHA256 = (
    "cdef17b02c2f77516b17225562edbed2c8faab41cbdb359a6a58ee2fec8b5236"
)
SCHEMA_V73 = "v73-qwen36-precision-lora-switch-preflight"
ARMS = {
    "bf16": {
        "model": ROOT / "models/Qwen3.6-35B-A3B",
        "config_sha256": (
            "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
        ),
        "index_sha256": (
            "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
        ),
        "quantization": None,
        "block_shape": None,
    },
    "fp8_serialized": {
        "model": ROOT / "models/Qwen3.6-35B-A3B-FP8",
        "config_sha256": (
            "570ef7ea45a7e1d3de2b1d3c70c4ac3562d0e768acdc195778cb4f4d95025845"
        ),
        "index_sha256": (
            "6f176f344e41d35b17af12904e33401da5ebff3b49fccb8bfa0185bc2d50f9d6"
        ),
        "quantization": "fp8",
        "block_shape": [128, 128],
    },
}


def file_sha256_v73(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arm_argument_v73(argv: list[str]) -> tuple[str, list[str]]:
    positions = [index for index, value in enumerate(argv) if value == "--precision-arm"]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise RuntimeError("V73 requires exactly one --precision-arm")
    position = positions[0]
    arm = argv[position + 1]
    if arm not in ARMS:
        raise ValueError("V73 precision arm is not sealed")
    if "--graph" in argv:
        raise RuntimeError("V73 sealed preflight requires eager execution")
    cleaned = argv[:position] + argv[position + 2:]
    return arm, cleaned


def output_argument_v73(argv: list[str]) -> Path:
    positions = [index for index, value in enumerate(argv) if value == "--output"]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise RuntimeError("V73 requires exactly one --output")
    return Path(argv[positions[0] + 1]).resolve()


def validate_static_v73(arm: str) -> dict:
    if file_sha256_v73(PREREG) != PREREG_FILE_SHA256:
        raise RuntimeError("V73 preregistration file identity changed")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    compact = dict(prereg)
    claimed = compact.pop("content_sha256_before_self_field", None)
    if (
        claimed != PREREG_CONTENT_SHA256
        or base.base.canonical_sha256(compact) != claimed
        or prereg.get("dependency", {}).get("required_before_gpu_launch") is not True
        or prereg.get("arms", {}).get(arm, {}).get("gpu_preflight_safe_after_dependency")
        is not True
    ):
        raise RuntimeError("V73 preregistration content or authority changed")
    contract = ARMS[arm]
    model = contract["model"]
    if (
        file_sha256_v73(model / "config.json") != contract["config_sha256"]
        or file_sha256_v73(model / "model.safetensors.index.json")
        != contract["index_sha256"]
        or Path(prereg["arms"][arm]["checkpoint"]["path"]).resolve()
        != model.resolve()
    ):
        raise RuntimeError("V73 model artifact identity changed")
    return prereg


def resolved_precision_v73(engine: object, arm: str) -> dict:
    llm_engine = getattr(engine, "llm_engine", None)
    config = getattr(llm_engine, "vllm_config", None)
    model = getattr(config, "model_config", None)
    quant = getattr(config, "quant_config", None)
    resolved = getattr(model, "quantization", None)
    quant_name = quant.get_name() if quant is not None else None
    block_shape = getattr(quant, "weight_block_size", None)
    if block_shape is not None:
        block_shape = list(block_shape)
    expected = ARMS[arm]
    if (
        resolved != expected["quantization"]
        or quant_name != expected["quantization"]
        or block_shape != expected["block_shape"]
    ):
        raise RuntimeError("V73 live engine resolved a different precision format")
    return {
        "model_quantization": resolved,
        "quant_config_name": quant_name,
        "weight_block_size": block_shape,
        "quant_config_class": type(quant).__name__ if quant is not None else None,
        "resolved_from_live_engine": True,
    }


def upgraded_receipt_v73(value: dict, arm: str, resolved: dict) -> dict:
    if (
        not isinstance(value, dict)
        or value.get("schema")
        != "v63-synthetic-two-adapter-switch-feasibility-probe"
        or value.get("runtime", {}).get("enforce_eager") is not True
        or value.get("runtime", {}).get("cuda_graphs_enabled") is not False
        or value.get("engine_shutdown_completed") is not True
        or value.get("adapter_update_or_hpo_performed") is not False
    ):
        raise RuntimeError("V73 underlying V63 receipt changed")
    claimed = value.get("content_sha256_before_self_field")
    original = dict(value)
    original.pop("content_sha256_before_self_field", None)
    if base.base.canonical_sha256(original) != claimed:
        raise RuntimeError("V73 underlying V63 receipt identity changed")
    expected = ARMS[arm]
    if (
        resolved.get("model_quantization") != expected["quantization"]
        or resolved.get("quant_config_name") != expected["quantization"]
        or resolved.get("weight_block_size") != expected["block_shape"]
        or resolved.get("resolved_from_live_engine") is not True
    ):
        raise RuntimeError("V73 precision certificate changed")
    result = dict(original)
    result["schema"] = SCHEMA_V73
    result["precision_arm"] = arm
    result["runtime"] = dict(result["runtime"])
    result["runtime"].update({
        "frozen_model_path": str(expected["model"].resolve()),
        "resolved_quantization": expected["quantization"],
        "serialized_fp8_checkpoint": arm == "fp8_serialized",
    })
    result["model_identity"] = {
        "config_sha256": expected["config_sha256"],
        "index_sha256": expected["index_sha256"],
    }
    result["resolved_precision_certificate"] = dict(resolved)
    result["preflight_gates"] = {
        "load_generate_switch_cleanup_passed": True,
        "candidate_changes_output": result["between_state_differing_rows"] > 0,
        "reference_restore_exact_at_token_hash_level": (
            result["reference_within_state_changed_rows"] == 0
        ),
        "candidate_repeat_exact_at_token_hash_level": (
            result["candidate_within_state_changed_rows"] == 0
        ),
        "scored_evaluation_or_training_authorized": False,
        "routed_expert_method_count_pending_worker_attestation": True,
    }
    result["preregistration"] = {
        "file_sha256": PREREG_FILE_SHA256,
        "content_sha256": PREREG_CONTENT_SHA256,
    }
    result["content_sha256_before_self_field"] = base.base.canonical_sha256(result)
    return result


def publish_v73(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_name(f".{path.name}.v73-{os.getpid()}")
    with temporary.open("x", encoding="ascii") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    arm, cleaned = arm_argument_v73(sys.argv[1:])
    output = output_argument_v73(cleaned)
    validate_static_v73(arm)
    import vllm

    original_argv = list(sys.argv)
    original_model = base.base.MODEL
    original_llm = vllm.LLM
    observed: list[dict] = []
    expected = ARMS[arm]

    def precision_llm(*args, **kwargs):
        if (
            Path(kwargs.get("model", "")).resolve() != expected["model"].resolve()
            or kwargs.get("enforce_eager") is not True
            or "quantization" in kwargs
        ):
            raise RuntimeError("V73 underlying engine contract changed")
        kwargs = dict(kwargs)
        kwargs["quantization"] = expected["quantization"]
        engine = original_llm(*args, **kwargs)
        observed.append(resolved_precision_v73(engine, arm))
        return engine

    base.base.MODEL = expected["model"].resolve()
    vllm.LLM = precision_llm
    sys.argv = [original_argv[0], *cleaned]
    try:
        status = base.main()
    finally:
        sys.argv = original_argv
        base.base.MODEL = original_model
        vllm.LLM = original_llm
    if status != 0 or not output.is_file() or len(observed) != 1:
        raise RuntimeError("V73 underlying precision probe failed")
    upgraded = upgraded_receipt_v73(
        json.loads(output.read_text(encoding="utf-8")), arm, observed[0]
    )
    publish_v73(output, upgraded)
    print(json.dumps({
        "schema": SCHEMA_V73,
        "precision_arm": arm,
        "output": str(output),
        "content_sha256": upgraded["content_sha256_before_self_field"],
        "wall_runtime_seconds": upgraded[
            "wall_runtime_seconds_excluding_model_load_and_cleanup"
        ],
        "preflight_gates": upgraded["preflight_gates"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
