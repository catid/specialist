#!/usr/bin/env python3

import json
import types

import pytest

import build_sft_v42i_vs_v47a_ood_first_preregistration_v47b as builder
import run_matched_lora_candidate_eval_v44a as core
import run_sft_v42i_vs_v47a_ood_first_v47b as subject


def _metric(reward, exact=10, nonzero=10, teacher=-1.0, leak=0):
    return {
        "generated_equal_unit_mean_reward": reward,
        "generated_exact_count": exact,
        "generated_nonzero_count": nonzero,
        "teacher_forced_equal_unit_mean_answer_logprob": teacher,
        "protocol_leak_counters": {"leak": leak},
    }


def test_two_full_waves_assign_every_gpu_v47b():
    waves = subject.arm_wave_plan_v47b()
    assert len(waves) == 2
    assert tuple(arm for arm, _ in waves[0]) == subject.BASE_ARMS_V47B
    assert tuple(arm for arm, _ in waves[1]) == subject.CANDIDATE_ARMS_V47B
    assert all(
        tuple(engine for _, engine in wave) == (0, 1, 2, 3)
        for wave in waves
    )


def test_candidate_replicas_share_staged_bytes_with_unique_ids_v47b():
    stages = subject.replica_stage_bindings_v47b()
    assert tuple(stages) == subject.CANDIDATE_ARMS_V47B
    for logical, (left, right) in subject.LOGICAL_REPLICAS_V47B.items():
        assert stages[left]["logical_candidate"] == logical
        assert stages[right]["logical_candidate"] == logical
        assert stages[left]["weights_file_sha256"] == stages[right][
            "weights_file_sha256"
        ]
        assert stages[left]["adapter_id"] != stages[right]["adapter_id"]


def test_each_replica_must_pass_ood_before_mean_shadow_ranking_v47b(
    monkeypatch,
):
    shadow = {
        "base_a": _metric(0.10),
        "sft_v42i_a": _metric(0.20),
        "sft_v42i_b": _metric(0.22),
        "sft_v47a_a": _metric(0.50),
        "sft_v47a_b": _metric(0.52),
    }
    ood = {arm: {"passed": True} for arm in shadow}
    prose = {arm: {"passed": True} for arm in shadow}
    ood["sft_v47a_b"] = {"passed": False}
    raw = {"ood_qa": {arm: [arm] for arm in shadow}}
    monkeypatch.setattr(core.v39a, "qa_ood_gate", lambda base, item: {
        "passed": item["passed"]
    })
    monkeypatch.setattr(core.v39a, "prose_gate", lambda base, item: {
        "passed": item["passed"]
    })
    monkeypatch.setattr(
        subject.ood_first, "paired_qa_bootstrap_v45a",
        lambda base, item: {},
    )
    with subject.patched_trusted_v47b():
        result = subject.trusted.finalize_selection_v46c(
            shadow, ood, prose, raw, {
                "shadow": {"all_four_base_outputs_exact": True}
            },
        )
    assert result["selected_logical_candidate"] == "sft_v42i"
    assert "sft_v47a" in result["ineligible_logical_candidates"]
    assert result["mean_replicated_shadow_ranking"] is True


def test_builder_and_runtime_dry_run_are_zero_protected_access_v47b(
    tmp_path, monkeypatch, capsys,
):
    protected = {item["path"] for item in core.PROTECTED_INPUTS_V44A.values()}
    original = core.file_sha256

    def guarded(path):
        assert str(path) not in protected
        return original(path)

    monkeypatch.setattr(core, "file_sha256", guarded)
    value = builder.build_v47b()
    assert value["heldout_or_holdout_access_authorized"] is False
    assert value["current_fixed_holdout_cycle_eligible"] is False
    assert value["v46d_or_any_holdout_artifact_bound"] is False
    assert value["protected_semantics_opened_by_v47b_builder"] is False
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args = types.SimpleNamespace(
        preregistration=str(path),
        preregistration_sha256=original(path),
        preregistration_content_sha256=value[
            "content_sha256_before_self_field"
        ],
    )
    subject.load_preregistration_v47b(args)

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


def test_builder_binds_all_source_stage_and_implementation_hashes_v47b():
    value = builder.build_v47b()
    definitions = {
        item["logical_candidate"]: item
        for item in value["candidate_definitions"]
    }
    assert set(definitions) == {"sft_v42i", "sft_v47a"}
    assert definitions["sft_v47a"]["source_weights_sha256"] == (
        subject.v47a_stage.EXPECTED["weights"]
    )
    assert definitions["sft_v47a"]["stage_manifest_file_sha256"] == (
        subject.STAGE_EXPECTED_V47B["sft_v47a"]["manifest_file"]
    )
    assert value["implementation_bindings"] == (
        subject.implementation_bindings_v47b()
    )
