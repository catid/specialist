import copy
import json

import pytest

import build_qwen36_bf16_kv_mamba_confirmation_preregistration_v80b as v80b


def test_v80b_generated_json_is_current_and_self_hashed():
    value = v80b.build_preregistration_v80b()
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field")
    assert claimed == v80b.canonical_sha256_v80b(body)
    assert v80b.OUTPUT.read_text(encoding="ascii") == v80b.render_json_v80b(value)


def test_v80b_is_exactly_two_prospective_confirmation_runs():
    value = v80b.build_preregistration_v80b()
    runs = value["confirmatory_runs"]
    assert [row["run"] for row in runs] == list(v80b.CONFIRMATIONS)
    assert [row["ordinal"] for row in runs] == [2, 3]
    assert len(runs) == 2
    assert all(row["fresh_run_directory_required"] for row in runs)


def test_v80b_exact_commands_use_unchanged_v80_launcher():
    commands = [
        row["exact_command"]
        for row in v80b.build_preregistration_v80b()["confirmatory_runs"]
    ]
    assert commands == [v80b._command_v80b(run) for run in v80b.CONFIRMATIONS]
    assert all("launch_qwen36_bf16_kv_mamba_capacity_v80.sh" in command for command in commands)


def test_v80b_copies_parent_runtime_and_gates_without_edits():
    parent = v80b._load_parent_v80b()
    sealed = v80b.build_preregistration_v80b()["parent_v80"]
    assert sealed["selected_runtime_retained_exactly"] == parent["selected_runtime"]
    assert sealed["live_acceptance_retained_exactly"] == parent["live_acceptance"]
    assert sealed["selected_runtime_canonical_sha256"] == v80b.canonical_sha256_v80b(parent["selected_runtime"])
    assert sealed["live_acceptance_canonical_sha256"] == v80b.canonical_sha256_v80b(parent["live_acceptance"])


def test_v80b_discloses_r1_and_forbids_tuning_or_promotion():
    value = v80b.build_preregistration_v80b()
    disclosure = value["post_observation_disclosure"]
    integrity = value["prospective_integrity"]
    assert disclosure["observed_before_v80b_preregistration"] is True
    assert disclosure["artifact_inventory"]["bundle_sha256"] == v80b.R1_BUNDLE_SHA256
    assert disclosure["observed_capacity_tokens_per_actor"] == [162669] * 4
    assert disclosure["torch_process_group_destroyed_receipt_value_per_actor"] == [False] * 4
    assert integrity["r1_not_used_to_change_any_parent_gate_or_threshold"] is True
    assert integrity["threshold_tuning_after_r1_forbidden"] is True
    assert integrity["promotion_from_r1_r2_or_r3_forbidden"] is True


def test_v80b_keeps_literal_parent_cleanup_gate_and_pending_quality_gates():
    value = v80b.build_preregistration_v80b()
    gates = value["parent_v80"]["live_acceptance_retained_exactly"]
    assert gates["cleanup"]["torch_process_group_destroyed_per_actor"] is True
    assert value["confirmation_analysis"]["literal_torch_process_group_destroyed_true_gate_retained"] is True
    assert value["confirmation_analysis"]["semantic_and_protected_ood_gates_remain_pending"] is True
    assert value["confirmation_analysis"]["promotion_default"] is False


def test_v80b_binds_parent_source_and_r1_hashes():
    value = v80b.build_preregistration_v80b()
    assert value["parent_v80"]["file_sha256"] == v80b.PARENT_FILE_SHA256
    assert value["parent_v80"]["content_sha256"] == v80b.PARENT_CONTENT_SHA256
    assert {row["path"]: row["sha256"] for row in value["sealed_executable_sources"]["files"]} == v80b.SOURCE_SHA256
    assert value["post_observation_disclosure"]["artifact_inventory"]["file_count"] == 11


def test_v80b_builder_is_cpu_only_and_has_no_scoring_authority():
    authority = v80b.build_preregistration_v80b()["authority"]
    assert authority["cpu_file_inspection_only"] is True
    assert authority["dataset_prompt_generated_text_or_protected_data_opened"] is False
    assert authority["gpu_or_model_launch_performed_by_builder"] is False
    assert authority["model_adapter_or_training_update_performed"] is False
    assert authority["scored_training_checkpoint_or_layout_promotion_authorized"] is False


def test_v80b_fails_closed_on_parent_or_r1_identity_change(monkeypatch):
    monkeypatch.setattr(v80b, "PARENT_FILE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="parent file identity changed"):
        v80b.build_preregistration_v80b()


def test_v80b_fails_closed_if_future_run_directory_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(v80b, "RUN_ROOT", tmp_path)
    (tmp_path / v80b.CONFIRMATIONS[0]).mkdir()
    with pytest.raises(RuntimeError, match="fresh r2/r3"):
        v80b.build_preregistration_v80b(require_future_directories_absent=True)


def test_v80b_json_has_no_nonfinite_values():
    encoded = v80b.render_json_v80b(v80b.build_preregistration_v80b())
    assert json.loads(encoded)["schema"] == v80b.SCHEMA
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
