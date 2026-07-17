import copy
from types import SimpleNamespace

import pytest

import probe_vllm_quantized_adapter_switch_v73 as subject


def baseline_receipt(reference_changed=0):
    value = {
        "schema": "v63-synthetic-two-adapter-switch-feasibility-probe",
        "runtime": {"enforce_eager": True, "cuda_graphs_enabled": False},
        "engine_shutdown_completed": True,
        "adapter_update_or_hpo_performed": False,
        "between_state_differing_rows": 7,
        "reference_within_state_changed_rows": reference_changed,
        "candidate_within_state_changed_rows": 0,
    }
    value["content_sha256_before_self_field"] = (
        subject.base.base.canonical_sha256(value)
    )
    return value


def certificate(arm):
    expected = subject.ARMS[arm]
    return {
        "model_quantization": expected["quantization"],
        "quant_config_name": expected["quantization"],
        "weight_block_size": expected["block_shape"],
        "quant_config_class": "Fp8Config" if arm == "fp8_serialized" else None,
        "resolved_from_live_engine": True,
    }


def test_argument_surface_is_sealed_and_removed_for_parent():
    arm, cleaned = subject.arm_argument_v73([
        "--precision-arm", "fp8_serialized", "--output", "x.json"
    ])
    assert arm == "fp8_serialized"
    assert cleaned == ["--output", "x.json"]
    with pytest.raises(RuntimeError, match="eager"):
        subject.arm_argument_v73([
            "--precision-arm", "bf16", "--graph", "--output", "x.json"
        ])
    with pytest.raises(ValueError, match="not sealed"):
        subject.arm_argument_v73([
            "--precision-arm", "int4", "--output", "x.json"
        ])


def test_live_resolution_requires_exact_arm():
    quant = SimpleNamespace(
        get_name=lambda: "fp8", weight_block_size=[128, 128]
    )
    engine = SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(
        model_config=SimpleNamespace(quantization="fp8"), quant_config=quant
    )))
    assert subject.resolved_precision_v73(
        engine, "fp8_serialized"
    )["quant_config_name"] == "fp8"
    engine.llm_engine.vllm_config.model_config.quantization = None
    with pytest.raises(RuntimeError, match="different precision"):
        subject.resolved_precision_v73(engine, "fp8_serialized")


def test_upgrade_records_pass_and_nonpass_without_overclaiming():
    upgraded = subject.upgraded_receipt_v73(
        baseline_receipt(), "fp8_serialized", certificate("fp8_serialized")
    )
    assert upgraded["precision_arm"] == "fp8_serialized"
    assert upgraded["preflight_gates"]["candidate_changes_output"] is True
    assert upgraded["preflight_gates"][
        "reference_restore_exact_at_token_hash_level"
    ] is True
    assert upgraded["preflight_gates"][
        "scored_evaluation_or_training_authorized"
    ] is False
    claimed = upgraded.pop("content_sha256_before_self_field")
    assert subject.base.base.canonical_sha256(upgraded) == claimed

    changed = subject.upgraded_receipt_v73(
        baseline_receipt(reference_changed=1), "bf16", certificate("bf16")
    )
    assert changed["preflight_gates"][
        "reference_restore_exact_at_token_hash_level"
    ] is False


def test_upgrade_rejects_forged_parent_or_precision():
    forged = baseline_receipt()
    forged["between_state_differing_rows"] = 8
    with pytest.raises(RuntimeError, match="identity changed"):
        subject.upgraded_receipt_v73(
            forged, "bf16", certificate("bf16")
        )
    wrong = copy.deepcopy(certificate("fp8_serialized"))
    wrong["weight_block_size"] = None
    with pytest.raises(RuntimeError, match="certificate changed"):
        subject.upgraded_receipt_v73(
            baseline_receipt(), "fp8_serialized", wrong
        )
