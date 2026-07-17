from types import SimpleNamespace

import pytest

import probe_vllm_fp8_rightsized_v74 as subject


def v73_receipt():
    value = {
        "schema": subject.base.SCHEMA_V73,
        "precision_arm": "fp8_serialized",
        "runtime": {"resolved_quantization": "fp8", "enforce_eager": True},
        "preflight_gates": {
            "scored_evaluation_or_training_authorized": False,
        },
    }
    value["content_sha256_before_self_field"] = (
        subject.base.base.base.canonical_sha256(value)
    )
    return value


def test_argv_requires_exact_fp8_eager_surface():
    assert subject.validate_argv_v74([
        "--precision-arm", "fp8_serialized", "--output", "x.json"
    ]).name == "x.json"
    with pytest.raises(RuntimeError, match="serialized-FP8"):
        subject.validate_argv_v74([
            "--precision-arm", "bf16", "--output", "x.json"
        ])
    with pytest.raises(RuntimeError, match="eager"):
        subject.validate_argv_v74([
            "--precision-arm", "fp8_serialized", "--graph", "--output", "x.json"
        ])


def test_live_budget_requires_exact_fraction():
    engine = SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(
        cache_config=SimpleNamespace(gpu_memory_utilization=0.50)
    )))
    assert subject.resolved_budget_v74(engine)["gpu_memory_utilization"] == 0.50
    engine.llm_engine.vllm_config.cache_config.gpu_memory_utilization = 0.51
    with pytest.raises(RuntimeError, match="different memory fraction"):
        subject.resolved_budget_v74(engine)


def test_upgrade_is_single_variable_and_rehashed():
    resolved = {"gpu_memory_utilization": 0.50, "resolved_from_live_engine": True}
    upgraded = subject.upgraded_receipt_v74(v73_receipt(), resolved)
    assert upgraded["runtime"]["gpu_memory_utilization"] == 0.50
    assert upgraded["single_variable_change_from_v73_fp8"] == {
        "gpu_memory_utilization": [0.82, 0.50]
    }
    assert upgraded["authority"][
        "scored_evaluation_training_checkpoint_or_promotion_allowed"
    ] is False
    claimed = upgraded.pop("content_sha256_before_self_field")
    assert subject.base.base.base.canonical_sha256(upgraded) == claimed


def test_upgrade_rejects_forged_parent_or_budget():
    forged = v73_receipt()
    forged["precision_arm"] = "bf16"
    with pytest.raises(RuntimeError, match="receipt changed"):
        subject.upgraded_receipt_v74(
            forged, {"gpu_memory_utilization": 0.50,
                     "resolved_from_live_engine": True}
        )
    with pytest.raises(RuntimeError, match="certificate changed"):
        subject.upgraded_receipt_v74(
            v73_receipt(), {"gpu_memory_utilization": 0.51,
                            "resolved_from_live_engine": True}
        )
