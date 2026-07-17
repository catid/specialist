import copy

import pytest

import probe_vllm_static48_wave_v68 as subject


def receipt():
    value = {
        "schema": "v63-synthetic-two-adapter-switch-feasibility-probe",
        "runtime": {
            "max_num_seqs": 68,
            "max_loras": 1,
            "max_cpu_loras": 2,
        },
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


def test_receipt_upgrade_changes_only_sequence_capacity():
    original = receipt()
    upgraded = subject.upgraded_receipt_v68(copy.deepcopy(original))
    assert upgraded["schema"] == subject.SCHEMA_V68
    assert upgraded["runtime"]["max_num_seqs"] == 48
    assert upgraded["runtime"]["max_loras"] == 1
    assert upgraded["runtime"]["submitted_prompt_count"] == 68
    claimed = upgraded.pop("content_sha256_before_self_field")
    assert subject.base.base.canonical_sha256(upgraded) == claimed
    assert original["runtime"]["max_num_seqs"] == 68


def test_receipt_upgrade_rejects_changed_or_forged_baseline():
    changed = receipt()
    changed["runtime"]["max_num_seqs"] = 48
    with pytest.raises(RuntimeError, match="receipt changed"):
        subject.upgraded_receipt_v68(changed)
    forged = receipt()
    forged["candidate_within_state_changed_rows"] = 1
    with pytest.raises(RuntimeError, match="identity changed"):
        subject.upgraded_receipt_v68(forged)


def test_output_argument_is_inherited_fail_closed():
    assert subject.wave32.output_argument_v67([
        "--output", "result.json",
    ]).name == "result.json"
    with pytest.raises(RuntimeError, match="one --output"):
        subject.wave32.output_argument_v67([])
