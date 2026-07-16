#!/usr/bin/env python3

import json
import types

import build_sft_v42i_vs_v47c_ood_first_preregistration_v47d as builder
import run_matched_lora_candidate_eval_v44a as core
import run_sft_v42i_vs_v47c_ood_first_v47d as subject


def _metric(reward, exact=10, nonzero=10, teacher=-1.0, leak=0):
    return {
        "generated_equal_unit_mean_reward": reward,
        "generated_exact_count": exact,
        "generated_nonzero_count": nonzero,
        "teacher_forced_equal_unit_mean_answer_logprob": teacher,
        "protocol_leak_counters": {"leak": leak},
    }


def test_two_full_waves_assign_every_gpu_v47d():
    waves = subject.arm_wave_plan_v47d()
    assert len(waves) == 2
    assert tuple(arm for arm, _ in waves[0]) == subject.BASE_ARMS_V47D
    assert tuple(arm for arm, _ in waves[1]) == subject.CANDIDATE_ARMS_V47D
    assert all(tuple(engine for _, engine in wave) == (0, 1, 2, 3) for wave in waves)


def test_each_replica_must_pass_ood_before_mean_shadow_ranking_v47d(monkeypatch):
    shadow = {
        "base_a": _metric(0.10),
        "sft_v42i_a": _metric(0.20), "sft_v42i_b": _metric(0.22),
        "sft_v47c_a": _metric(0.50), "sft_v47c_b": _metric(0.52),
    }
    ood = {arm: {"passed": True} for arm in shadow}
    prose = {arm: {"passed": True} for arm in shadow}
    ood["sft_v47c_b"] = {"passed": False}
    raw = {"ood_qa": {arm: [arm] for arm in shadow}}
    monkeypatch.setattr(core.v39a, "qa_ood_gate", lambda base, item: {
        "passed": item["passed"]
    })
    monkeypatch.setattr(core.v39a, "prose_gate", lambda base, item: {
        "passed": item["passed"]
    })
    monkeypatch.setattr(subject.ood_first, "paired_qa_bootstrap_v45a", lambda *args: {})
    with subject.patched_trusted_v47d():
        result = subject.trusted.trusted.finalize_selection_v46c(
            shadow, ood, prose, raw,
            {"shadow": {"all_four_base_outputs_exact": True}},
        )
    assert result["selected_logical_candidate"] == "sft_v42i"
    assert "sft_v47c" in result["ineligible_logical_candidates"]
    assert result["mean_replicated_shadow_ranking"] is True


def test_builder_reuses_original_shadow_and_is_zero_protected_access(tmp_path, monkeypatch):
    protected = {item["path"] for item in core.PROTECTED_INPUTS_V44A.values()}
    original = core.file_sha256

    def guarded(path):
        assert str(path) not in protected
        return original(path)

    monkeypatch.setattr(core, "file_sha256", guarded)
    value = builder.build_v47d()
    policy = value["validation_label_policy"]
    assert value["heldout_or_holdout_access_authorized"] is False
    assert value["consumed_holdout_artifact_bound"] is False
    assert value["protected_semantics_opened_by_builder"] is False
    assert value["single_access_inputs"] == core.PROTECTED_INPUTS_V44A
    assert policy["accepted_edit_successors_in_v47c_training"] == 42
    assert policy["accepted_shadow_side_edits"] == 9
    assert policy["accepted_shadow_side_edit_units"] == 3
    assert policy["refreshed_shadow_used_as_evaluation_input"] is False
    assert policy["evaluation_shadow_file_sha256"] == (
        core.PROTECTED_INPUTS_V44A["shadow"]["file_sha256"]
    )
    scope = value["edit_scope_audit_binding"]
    assert scope["file_sha256"] == subject.EDIT_SCOPE_AUDIT_FILE_SHA256
    assert scope["content_sha256"] == subject.EDIT_SCOPE_AUDIT_CONTENT_SHA256
    assert scope["edited_rows_entering_train"] == 42
    assert scope["edited_units_entering_train"] == 34
    assert scope["edited_rows_excluded_in_shadow"] == 9
    assert scope["edited_units_excluded_in_shadow"] == 3
    bindings = value["implementation_bindings"]
    assert "sft_v47c_edit_scope_audit_builder" in bindings
    assert "sft_v47c_edit_scope_audit_tests" in bindings
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args = types.SimpleNamespace(
        preregistration=str(path), preregistration_sha256=original(path),
        preregistration_content_sha256=value["content_sha256_before_self_field"],
    )
    subject.load_preregistration_v47d(args)


def test_stage_bindings_and_independent_replica_ids_v47d():
    stages = subject.replica_stage_bindings_v47d()
    for logical, (left, right) in subject.LOGICAL_REPLICAS_V47D.items():
        assert stages[left]["logical_candidate"] == logical
        assert stages[left]["weights_file_sha256"] == stages[right]["weights_file_sha256"]
        assert stages[left]["adapter_id"] != stages[right]["adapter_id"]
