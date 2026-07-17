import pytest

import probe_vllm_fp8_kv_mamba_bf16_v78b as subject


def parent_receipt():
    value = {
        "schema": subject.base.SCHEMA_V78,
        "runtime": {"kv_cache_dtype": "fp8_per_token_head"},
        "resolved_kv_cache_certificate": {
            "mamba_ssm_cache_dtype": "bfloat16",
        },
        "preflight_gates": {
            "scored_evaluation_or_training_authorized": False,
        },
        "single_variable_change_from_v76": {
            "kv_cache_dtype": [
                "auto_resolved_bfloat16", "fp8_per_token_head"
            ],
        },
    }
    canonical = subject.base.base.base.base.base.base.base.canonical_sha256
    value["content_sha256_before_self_field"] = canonical(value)
    return value


def resolved():
    return {
        "mamba_ssm_cache_dtype": "bfloat16",
        "resolved_from_live_engine": True,
    }


def test_upgrade_seals_only_mamba_ssm_dtype_and_keeps_authority_false():
    value = subject.upgraded_receipt_v78b(parent_receipt(), resolved())
    assert value["single_variable_change_from_v78"] == {
        "mamba_ssm_cache_dtype": ["float32", "bfloat16"]
    }
    assert value["combined_changes_from_v76"] == {
        "kv_cache_dtype": ["auto_resolved_bfloat16", "fp8_per_token_head"],
        "mamba_ssm_cache_dtype": ["float32", "bfloat16"],
    }
    assert "single_variable_change_from_v76" not in value
    assert value["preflight_gates"][
        "scored_evaluation_or_training_authorized"
    ] is False
    claimed = value.pop("content_sha256_before_self_field")
    canonical = subject.base.base.base.base.base.base.base.canonical_sha256
    assert canonical(value) == claimed


def test_upgrade_rejects_parent_or_resolved_drift():
    parent = parent_receipt()
    parent["runtime"]["kv_cache_dtype"] = "auto"
    with pytest.raises(RuntimeError, match="underlying V78"):
        subject.upgraded_receipt_v78b(parent, resolved())
    changed = resolved()
    changed["mamba_ssm_cache_dtype"] = "float16"
    with pytest.raises(RuntimeError, match="Mamba cache certificate"):
        subject.upgraded_receipt_v78b(parent_receipt(), changed)
