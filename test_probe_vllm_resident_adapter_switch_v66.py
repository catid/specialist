import copy

import pytest

import probe_vllm_resident_adapter_switch_v66 as subject


def receipt():
    value = {
        "schema": "v63-synthetic-two-adapter-switch-feasibility-probe",
        "runtime": {"max_loras": 1, "max_cpu_loras": 2},
        "engine_shutdown_completed": True,
        "adapter_update_or_hpo_performed": False,
        "wall_runtime_seconds_excluding_model_load_and_cleanup": 1.0,
        "reference_within_state_changed_rows": 0,
        "candidate_within_state_changed_rows": 0,
    }
    value["content_sha256_before_self_field"] = (
        subject.base.base.canonical_sha256(value)
    )
    return value


def test_receipt_upgrade_changes_only_capacity_contract():
    original = receipt()
    upgraded = subject.upgraded_receipt_v66(copy.deepcopy(original))
    assert upgraded["schema"] == subject.SCHEMA_V66
    assert upgraded["runtime"]["max_loras"] == 2
    assert upgraded["runtime"]["max_cpu_loras"] == 2
    assert upgraded["runtime"][
        "both_adapters_have_resident_gpu_slots"
    ] is True
    claimed = upgraded.pop("content_sha256_before_self_field")
    assert subject.base.base.canonical_sha256(upgraded) == claimed
    assert original["runtime"]["max_loras"] == 1


def test_receipt_upgrade_rejects_changed_or_forged_baseline():
    changed = receipt()
    changed["runtime"]["max_loras"] = 3
    with pytest.raises(RuntimeError, match="receipt changed"):
        subject.upgraded_receipt_v66(changed)
    forged = receipt()
    forged["reference_within_state_changed_rows"] = 1
    with pytest.raises(RuntimeError, match="identity changed"):
        subject.upgraded_receipt_v66(forged)


def test_output_argument_requires_exactly_one_value():
    assert subject.output_argument_v66(["--output", "result.json"]).name == (
        "result.json"
    )
    with pytest.raises(RuntimeError, match="one --output"):
        subject.output_argument_v66([])
    with pytest.raises(RuntimeError, match="one --output"):
        subject.output_argument_v66([
            "--output", "a.json", "--output", "b.json",
        ])
