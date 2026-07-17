#!/usr/bin/env python3

import copy
import json
import subprocess
import sys

import pytest

import build_qwen36_quantized_base_ablation_preregistration_v67 as prereg


def _frozen():
    return json.loads(prereg.OUTPUT.read_text(encoding="utf-8"))


def _valid_live_receipt(arm="fp8_serialized"):
    fp8 = arm == "fp8_serialized"
    return {
        "schema": "qwen36-quantized-base-live-preflight-v67",
        "arm": arm,
        "actor_count": 4,
        "physical_gpus": [0, 1, 2, 3],
        "load_and_generate_passed": True,
        "resolved_dtype": "bfloat16",
        "resolved_quantization": "fp8" if fp8 else None,
        "serialized_quantization": fp8,
        "fallback_messages": [],
        "unexpected_unquantized_module_prefixes": [],
        "routed_expert_method_counts": (
            {"Fp8MoEMethod": 40}
            if fp8
            else {"UnquantizedFusedMoEMethod": 40}
        ),
        "fp8_block_shape": [128, 128] if fp8 else None,
        "adapter_state": {
            "load_switch_candidate_restore_passed": True,
            "candidate_changed_output": True,
            "restore_exact_output": True,
            "master_sha256_before": "1" * 64,
            "master_sha256_after": "1" * 64,
            "base_sha256_before": "2" * 64,
            "base_sha256_after": "2" * 64,
        },
        "measurements": {
            "post_load_vram_bytes": 1,
            "peak_vram_bytes": 2,
            "available_kv_cache_memory_bytes": 3,
            "gpu_kv_block_count": 4,
            "prefill_tokens_per_second": 5.0,
            "decode_tokens_per_second": 6.0,
            "candidate_switch_seconds": 0.1,
            "restore_seconds": 0.1,
        },
        "all_four_actor_receipts_consensus": True,
        "all_four_gpus_positive_activity_witness": True,
    }


def test_v67_frozen_preregistration_rebuilds_and_self_validates():
    frozen = _frozen()
    built = prereg.build_preregistration_v67()
    assert built == frozen
    assert prereg.validate_preregistration_v67(frozen) == frozen


def test_v67_builder_does_not_import_torch_vllm_or_initialize_cuda():
    code = """
import json, sys
import build_qwen36_quantized_base_ablation_preregistration_v67 as p
p.build_preregistration_v67()
print(json.dumps({"torch": "torch" in sys.modules, "vllm": "vllm" in sys.modules}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=prereg.ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {"torch": False, "vllm": False}


def test_v67_only_bf16_and_serialized_fp8_are_safe_after_roofline():
    value = _frozen()
    assert value["dependency"]["bead"] == "specialist-0j5.14"
    assert value["arms"]["bf16"]["gpu_preflight_safe_after_dependency"] is True
    assert (
        value["arms"]["fp8_serialized"]["gpu_preflight_safe_after_dependency"]
        is True
    )
    assert value["arms"]["int4"]["gpu_preflight_safe_after_dependency"] is False
    assert value["arms"]["int4"]["launch_authorized"] is False
    assert value["arms"]["int4"]["local_serialized_checkpoint_candidates"] == []
    assert value["authority"]["scored_validation_before_per_arm_live_preflight"] is False


def test_v67_exact_engine_delta_and_no_silent_online_fp8_conversion():
    value = _frozen()
    bf16 = value["arms"]["bf16"]["engine_kwargs"]
    fp8 = value["arms"]["fp8_serialized"]["engine_kwargs"]
    differing = {key for key in bf16 if bf16[key] != fp8[key]}
    assert differing == {"model", "quantization"}
    assert bf16["quantization"] is None
    assert fp8["quantization"] == "fp8"
    assert fp8["model"].endswith("Qwen3.6-35B-A3B-FP8")
    assert value["arms"]["fp8_serialized"][
        "online_requantization_of_bf16_checkpoint_for_this_arm_forbidden"
    ] is True
    assert value["arms"]["fp8_serialized"]["required_starting_moe_tuning_table"] == (
        "empty_default"
    )
    assert value["prior_fp8_runtime_evidence"][
        "all_five_latency_endpoints_failed"
    ] is True
    assert value["prior_fp8_runtime_evidence"][
        "global_geometric_mean_tuned_over_default_speedup"
    ] == pytest.approx(0.9813847810295329)
    for item in (bf16, fp8):
        assert item["dtype"] == "bfloat16"
        assert item["kv_cache_dtype"] == "auto"
        assert item["enable_lora"] is True
        assert item["max_lora_rank"] == 32
        assert item["max_loras"] == 1
        assert item["max_cpu_loras"] == 2
        assert item["tensor_parallel_size"] == 1


def test_v67_checkpoint_dtype_and_memory_contract_is_exact():
    value = _frozen()
    bf16 = value["arms"]["bf16"]["checkpoint"]
    fp8 = value["arms"]["fp8_serialized"]["checkpoint"]
    memory = value["memory_expectations"]
    assert bf16["dtype_tensor_counts"] == {"BF16": 1045}
    assert fp8["dtype_tensor_counts"] == {"BF16": 32451, "F8_E4M3": 31745}
    assert fp8["fp8_expert_weight_count"] == 31488
    assert fp8["fp8_scale_count"] == 31745
    assert fp8["bf16_router_count"] == 41
    assert memory["logical_bytes_saved_by_fp8"] == 34_448_855_936
    assert memory["fraction_of_bf16_logical_bytes_saved"] == pytest.approx(
        0.4790974885978065
    )
    assert memory["nominal_full_attention_bf16_kv_bytes_per_token"] == 20_480
    assert memory["weight_format_does_not_directly_reduce_kv_bytes_per_token"] is True


def test_v67_int4_discovery_detects_but_does_not_authorize_new_checkpoint(tmp_path):
    candidate = tmp_path / "qwen-awq"
    candidate.mkdir()
    (candidate / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "awq", "bits": 4}}),
        encoding="utf-8",
    )
    discovered = prereg.discover_local_int4_checkpoints_v67(tmp_path)
    assert len(discovered) == 1
    assert discovered[0]["quant_method"] == "awq"
    assert discovered[0]["path"] == str(candidate)


def test_v67_fallback_log_classifier_is_case_insensitive_and_deduplicated():
    value = _frozen()
    logs = "\n".join(
        [
            "INFO exact FP8 kernel selected",
            "WARNING Falling Back to Moe WNA16 kernels",
            "WARNING Falling Back to Moe WNA16 kernels",
            "ERROR Please install bitsandbytes>=0.48.1",
        ]
    )
    assert prereg.forbidden_fallback_messages_v67(value, logs) == [
        "WARNING Falling Back to Moe WNA16 kernels",
        "ERROR Please install bitsandbytes>=0.48.1",
    ]


@pytest.mark.parametrize("arm", ["bf16", "fp8_serialized"])
def test_v67_valid_live_receipt_requires_exact_format_state_and_measurements(arm):
    value = _frozen()
    receipt = _valid_live_receipt(arm)
    assert prereg.validate_live_receipt_v67(value, receipt) == receipt


def test_v67_live_receipt_mutations_fail_closed(monkeypatch):
    value = _frozen()
    monkeypatch.setattr(prereg, "validate_preregistration_v67", lambda item: item)
    mutations = (
        lambda item: item.__setitem__("resolved_quantization", None),
        lambda item: item["fallback_messages"].append("falling back"),
        lambda item: item["routed_expert_method_counts"].update(
            {"Fp8MoEMethod": 39}
        ),
        lambda item: item["unexpected_unquantized_module_prefixes"].append(
            "model.layers.0.mlp.experts"
        ),
        lambda item: item["adapter_state"].__setitem__(
            "master_sha256_after", "3" * 64
        ),
        lambda item: item["adapter_state"].__setitem__(
            "candidate_changed_output", False
        ),
        lambda item: item["measurements"].__setitem__(
            "decode_tokens_per_second", float("nan")
        ),
        lambda item: item.__setitem__(
            "all_four_gpus_positive_activity_witness", False
        ),
    )
    for mutate in mutations:
        changed = copy.deepcopy(_valid_live_receipt())
        mutate(changed)
        with pytest.raises(RuntimeError):
            prereg.validate_live_receipt_v67(value, changed)


def test_v67_preregistration_mutations_fail_closed():
    value = _frozen()
    changed = copy.deepcopy(value)
    changed["arms"]["int4"]["launch_authorized"] = True
    changed["content_sha256_before_self_field"] = prereg.canonical_sha256_v67(
        {
            key: item
            for key, item in changed.items()
            if key != "content_sha256_before_self_field"
        }
    )
    with pytest.raises(RuntimeError, match="contract changed"):
        prereg.validate_preregistration_v67(changed)
