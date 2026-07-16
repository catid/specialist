#!/usr/bin/env python3

import json
import types

import build_sft_v47c_vs_v49b_ood_first_preregistration_v49c as builder
import run_matched_lora_candidate_eval_v44a as core
import run_sft_v47c_vs_v49b_ood_first_v49c as subject


def _metric(reward, exact=10, nonzero=10, teacher=-1.0, leak=0):
    return {
        "generated_equal_unit_mean_reward": reward,
        "generated_exact_count": exact,
        "generated_nonzero_count": nonzero,
        "teacher_forced_equal_unit_mean_answer_logprob": teacher,
        "protocol_leak_counters": {"leak": leak},
    }


def test_two_full_v47d_shape_waves_assign_every_gpu_v49c():
    waves = subject.arm_wave_plan_v49c()
    assert len(waves) == 2
    assert tuple(arm for arm, _ in waves[0]) == subject.BASE_ARMS_V49C
    assert tuple(arm for arm, _ in waves[1]) == subject.CANDIDATE_ARMS_V49C
    assert all(tuple(engine for _, engine in wave) == (0, 1, 2, 3) for wave in waves)
    assert set(subject.LOGICAL_CANDIDATES_V49C) == {"sft_v47c", "sft_v49b"}
    assert "sft_v42i" not in subject.ARMS_V49C


def test_stage_bindings_share_bytes_but_use_independent_replica_ids_v49c():
    stages = subject.replica_stage_bindings_v49c()
    for logical, (left, right) in subject.LOGICAL_REPLICAS_V49C.items():
        assert stages[left]["logical_candidate"] == logical
        assert stages[right]["logical_candidate"] == logical
        assert stages[left]["weights_file_sha256"] == stages[right][
            "weights_file_sha256"
        ]
        assert stages[left]["adapter_id"] != stages[right]["adapter_id"]


def test_each_v49b_replica_must_pass_ood_before_shadow_ranking_v49c(monkeypatch):
    shadow = {
        "base_a": _metric(0.10),
        "sft_v47c_a": _metric(0.20), "sft_v47c_b": _metric(0.22),
        "sft_v49b_a": _metric(0.50), "sft_v49b_b": _metric(0.52),
    }
    ood = {arm: {"passed": True} for arm in shadow}
    prose = {arm: {"passed": True} for arm in shadow}
    ood["sft_v49b_b"] = {"passed": False}
    raw = {"ood_qa": {arm: [arm] for arm in shadow}}
    monkeypatch.setattr(core.v39a, "qa_ood_gate", lambda base, item: {
        "passed": item["passed"]
    })
    monkeypatch.setattr(core.v39a, "prose_gate", lambda base, item: {
        "passed": item["passed"]
    })
    monkeypatch.setattr(subject.ood_first, "paired_qa_bootstrap_v45a", lambda *args: {})
    with subject.patched_trusted_v49c():
        result = subject.trusted.trusted.trusted.finalize_selection_v46c(
            shadow, ood, prose, raw,
            {"shadow": {"all_four_base_outputs_exact": True}},
        )
    assert result["selected_logical_candidate"] == "sft_v47c"
    assert "sft_v49b" in result["ineligible_logical_candidates"]
    assert result["mean_replicated_shadow_ranking"] is True


def test_builder_and_runtime_dry_run_are_zero_protected_access_v49c(
    tmp_path, monkeypatch, capsys,
):
    protected = {item["path"] for item in core.PROTECTED_INPUTS_V44A.values()}
    original = core.file_sha256

    def guarded(path):
        assert str(path) not in protected
        return original(path)

    monkeypatch.setattr(core, "file_sha256", guarded)
    value = builder.build_v49c()
    assert value["heldout_or_holdout_access_authorized"] is False
    assert value["consumed_holdout_artifact_bound"] is False
    assert value["protected_semantics_opened_by_builder"] is False
    assert value["single_access_inputs"] == core.PROTECTED_INPUTS_V44A
    assert value["matched_comparison"]["v42i_role"] == (
        "historical anchor through frozen V47D only"
    )
    assert value["runtime"]["no_partial_third_wave"] is True
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args = types.SimpleNamespace(
        preregistration=str(path), preregistration_sha256=original(path),
        preregistration_content_sha256=value["content_sha256_before_self_field"],
    )
    subject.load_preregistration_v49c(args)

    class ForbiddenFirewall:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dry-run must not construct protected firewall")

    monkeypatch.setattr(core, "SingleSemanticAccessV44A", ForbiddenFirewall)
    argv = [
        "--preregistration", str(path),
        "--preregistration-sha256", original(path),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ]
    assert subject.main(argv) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["protected_semantic_access_count"] == 0
    assert output["heldout_or_holdout_opened"] is False


def test_builder_binds_v49b_completion_and_exact_v49a_weight_identity_v49c():
    value = builder.build_v49c()
    definitions = {
        item["logical_candidate"]: item for item in value["candidate_definitions"]
    }
    assert set(definitions) == {"sft_v47c", "sft_v49b"}
    assert definitions["sft_v49b"]["source_weights_sha256"] == (
        subject.v49b_stage.EXPECTED["weights"]
    )
    assert definitions["sft_v49b"]["weight_identity_sha256"] == (
        subject.v49b_stage.EXPECTED_WEIGHT_IDENTITY
    )
    assert definitions["sft_v49b"]["stage_manifest_file_sha256"] == (
        subject.STAGE_EXPECTED_V49C["sft_v49b"]["manifest_file"]
    )
    assert value["implementation_bindings"] == subject.implementation_bindings_v49c()
