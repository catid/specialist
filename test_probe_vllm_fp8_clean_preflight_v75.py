import pytest

import probe_vllm_fp8_clean_preflight_v75 as subject


def v74_receipt():
    value = {
        "schema": subject.base.SCHEMA_V74,
        "runtime": {
            "resolved_quantization": "fp8",
            "gpu_memory_utilization": 0.50,
        },
        "authority": {
            "scored_evaluation_training_checkpoint_or_promotion_allowed": False,
        },
    }
    value["content_sha256_before_self_field"] = (
        subject.base.base.base.base.canonical_sha256(value)
    )
    return value


def environment():
    return {
        "VLLM_USE_DEEP_GEMM": "0",
        "enable_flashinfer_autotune": False,
        "tuned_config_folder_was_fresh_and_empty": True,
        "explicit_cutlass_path_requested_without_runtime_fallback": True,
    }


def test_upgrade_seals_explicit_environment_and_defers_log_gate():
    upgraded = subject.upgraded_receipt_v75(v74_receipt(), environment())
    assert upgraded["runtime"]["starting_moe_tuning_table"] == (
        "fresh_empty_default"
    )
    assert upgraded["runtime"]["enable_flashinfer_autotune"] is False
    assert upgraded["log_gate"]["forbidden_fallback_fragments_passed"] is None
    claimed = upgraded.pop("content_sha256_before_self_field")
    assert subject.base.base.base.base.canonical_sha256(upgraded) == claimed


def test_upgrade_rejects_parent_or_environment_drift():
    changed = v74_receipt()
    changed["runtime"]["gpu_memory_utilization"] = 0.51
    with pytest.raises(RuntimeError, match="receipt changed"):
        subject.upgraded_receipt_v75(changed, environment())
    changed_environment = environment()
    changed_environment["VLLM_USE_DEEP_GEMM"] = "1"
    with pytest.raises(RuntimeError, match="environment changed"):
        subject.upgraded_receipt_v75(v74_receipt(), changed_environment)
