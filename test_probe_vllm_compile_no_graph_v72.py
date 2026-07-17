import copy
from types import SimpleNamespace

import pytest

import probe_vllm_compile_no_graph_v72 as subject


def baseline_receipt():
    value = {
        "schema": "v63-synthetic-two-adapter-switch-feasibility-probe",
        "runtime": {
            "enforce_eager": False,
            "cuda_graphs_enabled": True,
        },
        "engine_shutdown_completed": True,
        "adapter_update_or_hpo_performed": False,
        "wall_runtime_seconds_excluding_model_load_and_cleanup": 1.0,
        "between_state_differing_rows": 2,
    }
    value["content_sha256_before_self_field"] = (
        subject.base.base.canonical_sha256(value)
    )
    return value


def test_resolved_compilation_requires_compile_without_graphs():
    engine = SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=SimpleNamespace(name="VLLM_COMPILE"),
            cudagraph_mode=SimpleNamespace(name="NONE"),
        )
    )))
    assert subject.resolved_compilation_v72(engine)["cudagraph_mode"] == "NONE"
    engine.llm_engine.vllm_config.compilation_config.cudagraph_mode.name = "FULL"
    with pytest.raises(RuntimeError, match="not VLLM_COMPILE"):
        subject.resolved_compilation_v72(engine)


def test_upgrade_corrects_underlying_flag_and_rehashes():
    original = baseline_receipt()
    resolved = {
        "compilation_mode": "VLLM_COMPILE",
        "cudagraph_mode": "NONE",
        "resolved_from_live_engine": True,
    }
    upgraded = subject.upgraded_receipt_v72(copy.deepcopy(original), resolved)
    assert upgraded["schema"] == subject.SCHEMA_V72
    assert upgraded["runtime"]["enforce_eager"] is False
    assert upgraded["runtime"]["cuda_graphs_enabled"] is False
    assert upgraded["runtime"]["compilation_mode"] == "VLLM_COMPILE"
    claimed = upgraded.pop("content_sha256_before_self_field")
    assert subject.base.base.canonical_sha256(upgraded) == claimed
    assert original["runtime"]["cuda_graphs_enabled"] is True


def test_upgrade_rejects_forgery_or_wrong_resolution():
    forged = baseline_receipt()
    forged["between_state_differing_rows"] = 3
    resolved = {
        "compilation_mode": "VLLM_COMPILE",
        "cudagraph_mode": "NONE",
        "resolved_from_live_engine": True,
    }
    with pytest.raises(RuntimeError, match="identity changed"):
        subject.upgraded_receipt_v72(forged, resolved)
    wrong = dict(resolved, cudagraph_mode="PIECEWISE")
    with pytest.raises(RuntimeError, match="certificate changed"):
        subject.upgraded_receipt_v72(baseline_receipt(), wrong)


def test_output_argument_requires_fresh_surface_shape():
    assert subject.output_argument_v72([
        "--graph", "--output", "receipt.json"
    ]).name == "receipt.json"
    with pytest.raises(RuntimeError, match="--graph"):
        subject.output_argument_v72(["--output", "receipt.json"])
    with pytest.raises(RuntimeError, match="exactly one"):
        subject.output_argument_v72([
            "--graph", "--output", "a.json", "--output", "b.json"
        ])
