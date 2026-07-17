import pytest

import probe_vllm_fp8_kv_cache_v78 as subject


def parent_receipt():
    value = {
        "schema": subject.base.SCHEMA_V76,
        "precision_arm": "fp8_serialized",
        "runtime": {"gpu_memory_utilization": 0.50},
        "routed_fp8_runtime_attestation": {"fp8_moe_method_count": 40},
        "preflight_gates": {
            "scored_evaluation_or_training_authorized": False,
        },
    }
    value["content_sha256_before_self_field"] = (
        subject.base.base.base.base.base.base.canonical_sha256(value)
    )
    return value


def resolved():
    return {
        "cache_dtype": "fp8_per_token_head",
        "calculate_kv_scales": False,
        "kv_cache_dtype_skip_layers": [],
        "mamba_cache_dtype": "auto",
        "mamba_ssm_cache_dtype": "auto",
        "resolved_from_live_engine": True,
    }


def test_upgrade_seals_single_variable_and_keeps_authority_false():
    value = subject.upgraded_receipt_v78(parent_receipt(), resolved())
    assert value["runtime"]["kv_cache_dtype"] == "fp8_per_token_head"
    assert value["single_variable_change_from_v76"] == {
        "kv_cache_dtype": ["auto_resolved_bfloat16", "fp8_per_token_head"]
    }
    assert value["preflight_gates"][
        "scored_evaluation_or_training_authorized"
    ] is False
    claimed = value.pop("content_sha256_before_self_field")
    assert subject.base.base.base.base.base.base.canonical_sha256(value) == claimed


def test_upgrade_rejects_parent_or_resolved_drift():
    parent = parent_receipt()
    parent["runtime"]["gpu_memory_utilization"] = 0.51
    with pytest.raises(RuntimeError, match="underlying V76"):
        subject.upgraded_receipt_v78(parent, resolved())
    changed = resolved()
    changed["calculate_kv_scales"] = True
    with pytest.raises(RuntimeError, match="KV-cache certificate"):
        subject.upgraded_receipt_v78(parent_receipt(), changed)
