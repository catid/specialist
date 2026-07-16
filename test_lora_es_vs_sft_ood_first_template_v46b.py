import json
import types

import pytest

import build_lora_es_vs_sft_ood_first_template_v46b as builder
import run_lora_es_vs_sft_ood_first_template_v46b as subject
import run_matched_lora_candidate_eval_v44a as core


def _metric(reward, exact=10, nonzero=10, teacher=-1.0):
    return {
        "generated_equal_unit_mean_reward": reward,
        "generated_exact_count": exact,
        "generated_nonzero_count": nonzero,
        "teacher_forced_equal_unit_mean_answer_logprob": teacher,
        "protocol_leak_counters": {"leak": 0},
    }


def test_two_full_waves_use_all_gpus_and_exact_replica_layout_v46b():
    waves = subject.arm_wave_plan_v46b()
    assert len(waves) == 2
    assert all(tuple(engine for _, engine in wave) == (0, 1, 2, 3)
               for wave in waves)
    assert tuple(arm for wave in waves for arm, _ in wave) == subject.ARMS_V46B
    assert len(subject.BASE_ARMS_V46B) == 4
    assert all(len(replicas) == 2
               for replicas in subject.CANDIDATE_FAMILIES_V46B.values())


def test_ood_ineligible_high_shadow_family_cannot_win_v46b():
    shadow = {
        "base_a": _metric(0.1),
        "sft_boundary_winner_a": _metric(0.9),
        "lora_es_v43i_a": _metric(0.2),
    }
    result = subject.choose_eligible_family_v46b(shadow, {
        "sft_boundary_winner": {"eligible": False},
        "lora_es_v43i": {"eligible": True},
    })
    assert result["selected_family"] == "lora_es_v43i"
    assert result["eligible_families"] == ["lora_es_v43i"]
    assert result["ood_eligible_set_constructed_before_shadow_ranking"] is True


def test_no_ood_eligible_family_falls_back_to_base_v46b():
    shadow = {
        "base_a": _metric(0.1),
        "sft_boundary_winner_a": _metric(0.9),
        "lora_es_v43i_a": _metric(0.8),
    }
    result = subject.choose_eligible_family_v46b(shadow, {
        family: {"eligible": False}
        for family in subject.FAMILY_TIE_ORDER_V46B
    })
    assert result["selected_arm"] == "base_a"
    assert result["selected_family"] is None
    assert result["shadow_improvement_gate_passed"] is False


def test_exact_duplicate_checks_fail_closed_v46b():
    values = {arm: {"score": 1} for arm in subject.ARMS_V46B}
    subject._assert_exact_replicas(values, "synthetic")
    values["base_d"] = {"score": 2}
    with pytest.raises(RuntimeError, match="four-base exact equivalence"):
        subject._assert_exact_replicas(values, "synthetic")
    values["base_d"] = {"score": 1}
    values["lora_es_v43i_b"] = {"score": 2}
    with pytest.raises(RuntimeError, match="two-GPU exact equivalence"):
        subject._assert_exact_replicas(values, "synthetic")


def test_builder_and_loader_never_hash_protected_inputs_v46b(
    tmp_path, monkeypatch,
):
    protected = {item["path"] for item in core.PROTECTED_INPUTS_V44A.values()}
    original = core.file_sha256

    def guarded(path):
        assert str(path) not in protected
        return original(path)

    monkeypatch.setattr(core, "file_sha256", guarded)
    value = builder.build_v46b()
    assert value["gpu_launch_authorized"] is False
    assert value["evaluation_launch_authorized"] is False
    assert value["heldout_or_holdout_access_authorized"] is False
    assert value["protected_semantics_inspected_during_v46b_revision"] is False
    assert value["candidate_artifact_seals"] == (
        subject.pending_candidate_seals_v46b()
    )
    path = tmp_path / "template.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args = types.SimpleNamespace(
        preregistration=str(path),
        preregistration_sha256=original(path),
        preregistration_content_sha256=value[
            "content_sha256_before_self_field"
        ],
    )
    loaded = subject.load_preregistration_v46b(args)
    assert loaded["status"].startswith("blocked_pending")


def test_dry_run_is_access_zero_and_non_dry_fails_before_firewall_v46b(
    tmp_path, monkeypatch, capsys,
):
    value = builder.build_v46b()
    path = tmp_path / "template.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    argv = [
        "--preregistration", str(path),
        "--preregistration-sha256", core.file_sha256(path),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
    ]

    class ForbiddenFirewall:
        def __init__(self, *args, **kwargs):
            raise AssertionError("protected firewall must not be constructed")

    monkeypatch.setattr(core, "SingleSemanticAccessV44A", ForbiddenFirewall)
    assert subject.main(argv + ["--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["gpu_launched"] is False
    assert output["protected_semantic_access_count"] == 0
    assert output["protected_paths_opened"] == []
    assert output["heldout_or_holdout_opened"] is False
    with pytest.raises(RuntimeError, match="unlaunchable CPU template"):
        subject.main(argv)


def test_supplying_only_one_candidate_identity_cannot_unlock_template_v46b(
    tmp_path,
):
    value = builder.build_v46b()
    value["candidate_artifact_seals"]["lora_es_v43i"]["status"] = "sealed"
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    value["content_sha256_before_self_field"] = core.canonical_sha256(compact)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    args = types.SimpleNamespace(
        preregistration=str(path),
        preregistration_sha256=core.file_sha256(path),
        preregistration_content_sha256=value[
            "content_sha256_before_self_field"
        ],
    )
    with pytest.raises(RuntimeError, match="fail-closed template contract"):
        subject.load_preregistration_v46b(args)
