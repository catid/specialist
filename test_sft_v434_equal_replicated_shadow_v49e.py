#!/usr/bin/env python3

import json
import types
from pathlib import Path

import build_sft_v434_equal_replicated_shadow_preregistration_v49e as builder
import run_sft_v434_equal_replicated_shadow_only_v49e as subject


def _metric(reward, exact=10, nonzero=10, teacher=-1.0, leak=0):
    return {
        "generated_equal_unit_mean_reward": reward,
        "generated_exact_count": exact,
        "generated_nonzero_count": nonzero,
        "teacher_forced_equal_unit_mean_answer_logprob": teacher,
        "protocol_leak_counters": {"leak": leak},
    }


def test_one_wave_uses_all_four_gpus_and_excludes_source50():
    wave = subject.arm_wave_plan_v49e()
    assert len(wave) == 1
    assert tuple(arm for arm, _ in wave[0]) == (
        "base_a", "base_b", "v434_equal_a", "v434_equal_b"
    )
    assert tuple(engine for _, engine in wave[0]) == (0, 1, 2, 3)
    assert all("source50" not in arm for arm in subject.ARMS)


def test_ood_recovery_freezes_equal_as_only_eligible_candidate():
    proof = subject.ood_recovery_proof_v49e()
    assert proof["eligible_logical_candidates"] == ["v434_equal"]
    assert proof["excluded_ineligible_logical_candidates"] == ["v434_source50"]
    assert proof["shadow_opened_in_ood_phase"] is False
    assert proof["heldout_or_holdout_opened"] is False


def test_builder_and_dryrun_do_not_open_shadow_split_or_gpu(tmp_path, monkeypatch, capsys):
    protected = {item["path"] for item in subject.SHADOW_INPUTS.values()}
    original_hash = subject.core.file_sha256

    def guarded_hash(path):
        assert str(Path(path).resolve()) not in protected
        return original_hash(path)

    monkeypatch.setattr(subject.core, "file_sha256", guarded_hash)
    value = builder.build()
    assert value["single_access_inputs"] == subject.SHADOW_INPUTS
    assert value["protected_semantics_opened_by_builder"] is False
    assert value["heldout_or_holdout_access_authorized"] is False
    path = tmp_path / "prereg.json"
    subject.core.atomic_json(path, value)
    args = types.SimpleNamespace(
        preregistration=str(path), preregistration_sha256=original_hash(path),
        preregistration_content_sha256=value["content_sha256_before_self_field"],
    )
    subject.load_preregistration_v49e(args)
    monkeypatch.setattr(
        subject.topology, "gpu_preflight",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run reached GPU")),
    )
    monkeypatch.setattr(
        subject.core, "SingleSemanticAccessV44A",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("dry-run opened shadow/split")
        ),
    )
    argv = [
        "--preregistration", str(path),
        "--preregistration-sha256", original_hash(path),
        "--preregistration-content-sha256", value["content_sha256_before_self_field"],
        "--dry-run",
    ]
    assert subject.main(argv) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["single_access_labels"] == ["shadow", "split_manifest"]
    assert output["protected_semantic_access_count"] == 0
    assert output["shadow_opened"] is False
    assert output["heldout_or_holdout_opened"] is False
    assert output["gpu_accessed"] is False


def test_replicated_equal_selection_uses_mean_vector_and_protocol_gate():
    metrics = {
        "base_a": _metric(0.10), "base_b": _metric(0.10),
        "v434_equal_a": _metric(0.12), "v434_equal_b": _metric(0.14),
    }
    value = subject.shadow_decision_v49e(metrics)
    assert value["mean_replicated_shadow_metrics"][
        "generated_equal_unit_mean_reward"
    ] == 0.13
    assert value["replicated_equal_vs_base_decision_passed"] is True
    assert value["selected_logical_candidate"] == "v434_equal"
    metrics["v434_equal_b"] = _metric(0.14, leak=1)
    value = subject.shadow_decision_v49e(metrics)
    assert value["shadow_improvement_gate_passed"] is True
    assert value["replicated_equal_vs_base_decision_passed"] is False
    assert value["selected_arm"] == "base_a"


def test_stage_replicas_share_bytes_but_use_distinct_ids():
    stages = subject.replica_stage_bindings_v49e()
    left, right = subject.CANDIDATE_ARMS
    assert stages[left]["weights_file_sha256"] == stages[right]["weights_file_sha256"]
    assert stages[left]["adapter_id"] != stages[right]["adapter_id"]
