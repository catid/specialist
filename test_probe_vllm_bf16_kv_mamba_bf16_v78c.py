import pytest

import probe_vllm_bf16_kv_mamba_bf16_v78c as subject


def parent_receipt():
    value = {
        "schema": subject.base.SCHEMA_V76,
        "runtime": {"gpu_memory_utilization": 0.50},
        "preflight_gates": {
            "scored_evaluation_or_training_authorized": False,
        },
    }
    canonical = subject.base.base.base.base.base.base.canonical_sha256
    value["content_sha256_before_self_field"] = canonical(value)
    return value


def resolved():
    return {
        "cache_dtype": "auto",
        "mamba_ssm_cache_dtype": "bfloat16",
        "resolved_from_live_engine": True,
    }


def test_upgrade_isolates_mamba_ssm_dtype_and_keeps_authority_false():
    value = subject.upgraded_receipt_v78c(parent_receipt(), resolved())
    assert value["single_variable_change_from_v76"] == {
        "mamba_ssm_cache_dtype": ["float32", "bfloat16"]
    }
    assert value["preflight_gates"][
        "scored_evaluation_or_training_authorized"
    ] is False
    claimed = value.pop("content_sha256_before_self_field")
    canonical = subject.base.base.base.base.base.base.canonical_sha256
    assert canonical(value) == claimed


def test_upgrade_rejects_parent_or_resolved_drift():
    parent = parent_receipt()
    parent["runtime"]["gpu_memory_utilization"] = 0.49
    with pytest.raises(RuntimeError, match="underlying V76"):
        subject.upgraded_receipt_v78c(parent, resolved())
    changed = resolved()
    changed["cache_dtype"] = "fp8_per_token_head"
    with pytest.raises(RuntimeError, match="hybrid-cache certificate"):
        subject.upgraded_receipt_v78c(parent_receipt(), changed)
