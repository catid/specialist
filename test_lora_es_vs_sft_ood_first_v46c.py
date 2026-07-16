#!/usr/bin/env python3

import json
import types

import pytest

import build_lora_es_vs_sft_ood_first_preregistration_v46c as builder
import run_lora_es_vs_sft_ood_first_v46c as subject
import run_matched_lora_candidate_eval_v44a as core


def _metric(reward, exact=10, nonzero=10, teacher=-1.0, leak=0):
    return {
        "generated_equal_unit_mean_reward": reward,
        "generated_exact_count": exact,
        "generated_nonzero_count": nonzero,
        "teacher_forced_equal_unit_mean_answer_logprob": teacher,
        "protocol_leak_counters": {"leak": leak},
    }


def test_two_full_waves_use_every_gpu_v46c():
    waves = subject.arm_wave_plan_v46c()
    assert len(waves) == 2
    assert all(
        tuple(engine for _, engine in wave) == (0, 1, 2, 3)
        for wave in waves
    )
    assert tuple(arm for wave in waves for arm, _ in wave) == subject.ARMS_V46C


def test_stage_replicas_share_bytes_but_outputs_need_not_be_exact_v46c():
    stages = subject.replica_stage_bindings_v46c()
    assert tuple(stages) == subject.CANDIDATE_ARMS_V46C
    for logical, (left, right) in subject.LOGICAL_REPLICAS_V46C.items():
        assert stages[left]["logical_candidate"] == logical
        assert stages[right]["logical_candidate"] == logical
        assert stages[left]["weights_file_sha256"] == stages[right][
            "weights_file_sha256"
        ]
        assert stages[left]["adapter_id"] != stages[right]["adapter_id"]


def test_mean_replica_ranking_accepts_nonidentical_candidate_outputs_v46c(
    monkeypatch,
):
    shadow = {
        "base_a": _metric(0.10),
        "sft_v42i_a": _metric(0.20),
        "sft_v42i_b": _metric(0.22),
        "lora_es_v43j_a": _metric(0.30),
        "lora_es_v43j_b": _metric(0.34),
    }
    ood = {arm: {"eligible": True} for arm in shadow}
    prose = {arm: {"eligible": True} for arm in shadow}
    raw = {"ood_qa": {arm: [arm] for arm in shadow}}
    monkeypatch.setattr(core.v39a, "qa_ood_gate", lambda base, item: {
        "passed": item["eligible"]
    })
    monkeypatch.setattr(core.v39a, "prose_gate", lambda base, item: {
        "passed": item["eligible"]
    })
    monkeypatch.setattr(subject.ood_first, "paired_qa_bootstrap_v45a",
                        lambda base, item: {"paired_passed": True})
    result = subject.finalize_selection_v46c(
        shadow, ood, prose, raw,
        {"shadow": {"all_four_base_outputs_exact": True}},
    )
    assert result["selected_logical_candidate"] == "lora_es_v43j"
    assert result["candidate_replicas_required_bit_exact"] is False
    assert result["mean_replicated_shadow_ranking"] is True
    assert result["per_logical_candidate_gate_table"]["lora_es_v43j"][
        "mean_replicated_shadow_metrics"
    ]["generated_equal_unit_mean_reward"] == pytest.approx(0.32)


def test_either_replica_ood_failure_excludes_whole_logical_candidate_v46c(
    monkeypatch,
):
    shadow = {
        "base_a": _metric(0.10),
        "sft_v42i_a": _metric(0.20),
        "sft_v42i_b": _metric(0.21),
        "lora_es_v43j_a": _metric(0.50),
        "lora_es_v43j_b": _metric(0.51),
    }
    ood = {arm: {"passed": True} for arm in shadow}
    ood["lora_es_v43j_b"] = {"passed": False}
    prose = {arm: {"passed": True} for arm in shadow}
    raw = {"ood_qa": {arm: [arm] for arm in shadow}}
    monkeypatch.setattr(core.v39a, "qa_ood_gate", lambda base, item: {
        "passed": item["passed"]
    })
    monkeypatch.setattr(core.v39a, "prose_gate", lambda base, item: {
        "passed": item["passed"]
    })
    monkeypatch.setattr(subject.ood_first, "paired_qa_bootstrap_v45a",
                        lambda base, item: {})
    result = subject.finalize_selection_v46c(
        shadow, ood, prose, raw, {}
    )
    assert result["selected_logical_candidate"] == "sft_v42i"
    assert "lora_es_v43j" in result["ineligible_logical_candidates"]


def test_builder_loader_and_dry_run_are_zero_protected_access_v46c(
    tmp_path, monkeypatch, capsys,
):
    protected = {item["path"] for item in core.PROTECTED_INPUTS_V44A.values()}
    original = core.file_sha256

    def guarded(path):
        assert str(path) not in protected
        return original(path)

    monkeypatch.setattr(core, "file_sha256", guarded)
    value = builder.build_v46c()
    assert value["gpu_launch_authorized"] is True
    assert value["heldout_or_holdout_access_authorized"] is False
    assert value["protected_semantics_inspected_during_v46c_revision"] is False
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args = types.SimpleNamespace(
        preregistration=str(path),
        preregistration_sha256=original(path),
        preregistration_content_sha256=value[
            "content_sha256_before_self_field"
        ],
    )
    subject.load_preregistration_v46c(args)

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
