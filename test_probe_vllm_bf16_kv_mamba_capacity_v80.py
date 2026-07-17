from pathlib import Path
from types import SimpleNamespace

import pytest

import probe_vllm_bf16_kv_mamba_capacity_v80 as subject


def _parent_receipt() -> dict:
    # The live V73 receipt records resolved fields, but not the requested
    # gpu_memory_utilization.  The engine-call wrapper independently binds it.
    value = {
        "schema": subject.base.SCHEMA_V73,
        "precision_arm": "fp8_serialized",
        "runtime": {
            "resolved_quantization": "fp8",
            "enforce_eager": True,
        },
        "adapter_update_or_hpo_performed": False,
        "source_dataset_rows_opened": 0,
        "protected_ood_shadow_or_terminal_opened": False,
        "preflight_gates": {
            "candidate_changes_output": True,
            "candidate_repeat_exact_at_token_hash_level": True,
            "scored_evaluation_or_training_authorized": False,
        },
    }
    value["content_sha256_before_self_field"] = (
        subject.base.base.base.canonical_sha256(value)
    )
    return value


def _residency() -> dict:
    return {
        "schema": "v76-live-named-parameter-residency",
        "components": {
            "language": {
                "device_counts": {"cuda:0": 813},
                "dtype_counts": {
                    "torch.bfloat16": 303,
                    "torch.float32": 270,
                    "torch.float8_e4m3fn": 240,
                },
                "logical_bytes": 35_712_084_096,
                "parameter_count": 813,
                "parameter_names_sha256": (
                    "a850f55c3f02ef904041d48b29f13af2d29834da200f92dcc9728760cb185b90"
                ),
            }
        },
        "named_parameters_remove_duplicate_default": True,
        "total_logical_bytes": 35_712_084_096,
        "total_parameter_count": 813,
    }


def _audit() -> dict:
    records = [
        {
            "name": f"model.language_model.layers.{index}.mlp.experts",
            "module_class": "RoutedExperts",
            "quant_method_class": "Fp8MoEMethod",
            "runtime_quant_wrapper_class": "FusedMoEModularMethod",
            "weight_block_size": [128, 128],
            "block_quant": True,
            "w13_dtype": "torch.float8_e4m3fn",
            "w2_dtype": "torch.float8_e4m3fn",
        }
        for index in range(40)
    ]
    names_hash = (
        subject.audit_base.base.base.base.base.base.canonical_sha256(
            [row["name"] for row in records]
        )
    )
    return {
        "schema": "v76-fp8-routed-model-runtime-audit",
        "fp8_moe_method_count": 40,
        "fp8_quant_reference_count": 80,
        "fp8_moe_names_sha256": names_hash,
        "fp8_moe_records": records,
        "routed_like_module_count": 40,
        "routed_like_without_fp8_method": [],
        "moe_runner_module_count": 40,
        "moe_runner_modules": [],
        "parameter_residency": _residency(),
    }


def _resolved(tokens: int = 161_792) -> dict:
    return {
        "gpu_memory_utilization": 0.479,
        "cache_dtype": "auto",
        "calculate_kv_scales": False,
        "kv_cache_dtype_skip_layers": [],
        "mamba_cache_dtype": "auto",
        "mamba_ssm_cache_dtype": "bfloat16",
        "kv_cache_size_tokens": tokens,
        "kv_cache_max_concurrency": tokens / 2048,
        "num_gpu_blocks": 79,
        "block_size": 2048,
        "enable_flashinfer_autotune": False,
        "model_quantization": "fp8",
        "resolved_from_live_engine": True,
    }


def _performance() -> dict:
    labels = [
        "reference",
        "candidate",
        "reference",
        "candidate",
        "candidate",
        "reference",
        "reference",
        "candidate",
        "candidate",
        "reference",
    ]
    rows = [
        {
            "call_index": index,
            "label": label,
            "request_count": 68,
            "generated_token_count": 4_352,
            "elapsed_seconds": 1.0 + index / 100,
        }
        for index, label in enumerate(labels)
    ]
    return subject.summarize_generation_timings_v80(rows)


def test_static_preregistration_is_bound_and_authority_false():
    value = subject.validate_static_v80()
    assert value["single_variable_change_from_v78c"] == {
        "gpu_memory_utilization": [0.5, 0.479]
    }
    assert value["authority"]["scored_or_training_authority"] is False


def test_argv_requires_exact_fresh_fp8_eager_surface(tmp_path: Path):
    output = tmp_path / "receipt.json"
    assert subject.validate_argv_v80(
        [
            "--precision-arm",
            "fp8_serialized",
            "--actor-label",
            "gpu-0",
            "--output",
            str(output),
        ]
    ) == output
    with pytest.raises(RuntimeError, match="serialized-FP8"):
        subject.validate_argv_v80(
            ["--precision-arm", "bf16", "--output", str(output)]
        )
    with pytest.raises(RuntimeError, match="eager"):
        subject.validate_argv_v80(
            [
                "--precision-arm",
                "fp8_serialized",
                "--graph",
                "--output",
                str(output),
            ]
        )
    output.write_text("occupied", encoding="ascii")
    with pytest.raises(RuntimeError, match="fresh output"):
        subject.validate_argv_v80(
            ["--precision-arm", "fp8_serialized", "--output", str(output)]
        )


def test_rightsized_kwargs_reconstruct_v78c_and_change_only_utilization():
    original = {
        "quantization": "fp8",
        "gpu_memory_utilization": 0.82,
        "enforce_eager": True,
        "moe_backend": "triton",
        "max_num_seqs": 68,
    }
    result = subject.rightsized_kwargs_v80(original)
    assert original["gpu_memory_utilization"] == 0.82
    assert result == {
        **original,
        "gpu_memory_utilization": 0.479,
        "mamba_ssm_cache_dtype": "bfloat16",
        "enable_flashinfer_autotune": False,
    }
    simulated_v78c = {**result, "gpu_memory_utilization": 0.5}
    changed = {
        key: (simulated_v78c[key], value)
        for key, value in result.items()
        if simulated_v78c[key] != value
    }
    assert changed == {"gpu_memory_utilization": (0.5, 0.479)}


def test_rightsized_kwargs_reject_parent_drift():
    with pytest.raises(RuntimeError, match="underlying V73"):
        subject.rightsized_kwargs_v80(
            {
                "quantization": "fp8",
                "gpu_memory_utilization": 0.82,
                "enforce_eager": False,
                "moe_backend": "triton",
            }
        )


def test_live_certificate_binds_bf16_hybrid_cache_and_capacity():
    resolved = _resolved()
    cache = SimpleNamespace(
        **{
            key: value
            for key, value in resolved.items()
            if key
            not in {
                "enable_flashinfer_autotune",
                "model_quantization",
                "resolved_from_live_engine",
            }
        }
    )
    engine = SimpleNamespace(
        llm_engine=SimpleNamespace(
            vllm_config=SimpleNamespace(
                cache_config=cache,
                kernel_config=SimpleNamespace(
                    enable_flashinfer_autotune=False
                ),
                model_config=SimpleNamespace(quantization="fp8"),
            )
        )
    )
    assert subject.resolved_runtime_v80(engine) == resolved
    cache.kv_cache_size_tokens = 159_744
    with pytest.raises(RuntimeError, match="capacity certificate"):
        subject.resolved_runtime_v80(engine)


def test_upgrade_accepts_real_parent_shape_and_is_fail_closed():
    result = subject.upgraded_receipt_v80(
        _parent_receipt(), _resolved(), _audit(), _performance()
    )
    assert result["runtime"]["gpu_memory_utilization"] == 0.479
    assert result["single_variable_change_from_v78c"] == {
        "gpu_memory_utilization": [0.5, 0.479]
    }
    assert result["resolved_hybrid_cache_certificate"][
        "mamba_ssm_cache_dtype"
    ] == "bfloat16"
    assert result["routed_fp8_runtime_attestation"][
        "parameter_residency"
    ] == _residency()
    assert result["preflight_gates"][
        "scored_evaluation_or_training_authorized"
    ] is False
    claimed = result.pop("content_sha256_before_self_field")
    assert subject.canonical_sha256_v80(result) == claimed


def test_upgrade_rejects_parent_capacity_and_residency_drift():
    parent = _parent_receipt()
    parent["source_dataset_rows_opened"] = 1
    with pytest.raises(RuntimeError, match="underlying V73"):
        subject.upgraded_receipt_v80(
            parent, _resolved(), _audit(), _performance()
        )
    with pytest.raises(RuntimeError, match="resolved runtime"):
        subject.upgraded_receipt_v80(
            _parent_receipt(),
            _resolved(159_744),
            _audit(),
            _performance(),
        )
    audit = _audit()
    audit["parameter_residency"]["total_parameter_count"] = 812
    with pytest.raises(RuntimeError, match="residency identity"):
        subject.upgraded_receipt_v80(
            _parent_receipt(), _resolved(), audit, _performance()
        )


def test_generation_timing_binds_throughput_and_tail_latency():
    value = _performance()
    assert value["warmup_call_count"] == 2
    assert value["measured_call_count"] == 8
    assert value["measured_generated_token_count"] == 8 * 4_352
    assert value["p95_call_latency_seconds_nearest_rank"] == 1.09
    assert value["prompt_or_generation_text_persisted"] is False
    assert value["token_ids_persisted"] is False
    with pytest.raises(RuntimeError, match="timing surface"):
        subject.summarize_generation_timings_v80(value["calls"][:-1])
