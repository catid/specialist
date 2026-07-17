import copy

import pytest

import probe_vllm_static_wave_v67 as subject


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
    upgraded = subject.upgraded_receipt_v67(copy.deepcopy(original))
    assert upgraded["schema"] == subject.SCHEMA_V67
    assert upgraded["runtime"]["max_num_seqs"] == 32
    assert upgraded["runtime"]["max_loras"] == 1
    assert upgraded["runtime"]["max_cpu_loras"] == 2
    assert upgraded["runtime"]["submitted_prompt_count"] == 68
    assert upgraded["runtime"]["bounded_internal_scheduling_waves"] is True
    claimed = upgraded.pop("content_sha256_before_self_field")
    assert subject.base.base.canonical_sha256(upgraded) == claimed
    assert original["runtime"]["max_num_seqs"] == 68


def test_receipt_upgrade_rejects_changed_or_forged_baseline():
    changed = receipt()
    changed["runtime"]["max_num_seqs"] = 31
    with pytest.raises(RuntimeError, match="receipt changed"):
        subject.upgraded_receipt_v67(changed)
    forged = receipt()
    forged["reference_within_state_changed_rows"] = 1
    with pytest.raises(RuntimeError, match="identity changed"):
        subject.upgraded_receipt_v67(forged)


def test_output_argument_requires_exactly_one_value():
    assert subject.output_argument_v67(["--output", "result.json"]).name == (
        "result.json"
    )
    with pytest.raises(RuntimeError, match="one --output"):
        subject.output_argument_v67([])
    with pytest.raises(RuntimeError, match="one --output"):
        subject.output_argument_v67([
            "--output", "a.json", "--output", "b.json",
        ])
