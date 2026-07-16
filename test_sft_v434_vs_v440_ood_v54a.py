#!/usr/bin/env python3

from __future__ import annotations

import json
import types
from pathlib import Path

import build_sft_v434_vs_v440_ood_preregistration_v54a as builder
import run_sft_v434_vs_v440_replicated_ood_only_v54a as subject


def test_v54a_sources_and_stages_are_exact_and_train_only():
    for logical in subject.LOGICAL_CANDIDATES:
        seal = subject._source_seal(logical)
        assert seal["completed_steps"] == 48
        assert seal["shadow_ood_holdout_or_heldout_opened"] is False
        binding = subject.canonical_stage_binding_v54a(logical)
        assert binding["weights_file_sha256"] == subject.STAGE_EXPECTED[
            logical
        ]["weights"]
        assert binding["tensor_bytes_preserved_exactly"] is True


def test_v54a_has_exact_two_full_content_matched_waves():
    value = builder.build()
    assert value["single_access_inputs"] == subject.OOD_INPUTS
    assert set(value["single_access_inputs"]) == {"ood_qa", "ood_prose"}
    assert value["shadow_access_authorized"] is False
    assert value["heldout_or_holdout_access_authorized"] is False
    waves = value["runtime"]["two_full_fixed_waves"]
    assert [[item["arm"] for item in wave] for wave in waves] == [
        ["base_a", "base_b", "base_c", "base_d"],
        ["v434_equal_a", "v434_equal_b", "v440_equal_a", "v440_equal_b"],
    ]
    assert all(
        [item["engine_index"] for item in wave] == [0, 1, 2, 3]
        for wave in waves
    )
    assert value["input_scope"][
        "same_content_addressed_inputs_for_every_arm"
    ] is True


def test_v54a_builder_and_dry_run_open_zero_semantics_or_gpu(
    tmp_path, monkeypatch, capsys,
):
    protected = {item["path"] for item in subject.OOD_INPUTS.values()}
    original_hash = subject.core.file_sha256

    def guarded_hash(path):
        assert str(Path(path).resolve()) not in protected
        return original_hash(path)

    monkeypatch.setattr(subject.core, "file_sha256", guarded_hash)
    value = builder.build()
    path = tmp_path / "prereg.json"
    subject.core.atomic_json(path, value)
    args = types.SimpleNamespace(
        preregistration=str(path),
        preregistration_sha256=original_hash(path),
        preregistration_content_sha256=value[
            "content_sha256_before_self_field"
        ],
    )
    subject.load_preregistration_v54a(args)
    monkeypatch.setattr(
        subject.parent.topology,
        "gpu_preflight",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run reached GPU")),
    )
    monkeypatch.setattr(
        subject.core,
        "SingleSemanticAccessV44A",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run opened protected input")
        ),
    )
    argv = [
        "--preregistration", str(path),
        "--preregistration-sha256", original_hash(path),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ]
    assert subject.main(argv) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["single_access_labels"] == ["ood_prose", "ood_qa"]
    assert output["protected_semantic_access_count"] == 0
    assert output["shadow_opened"] is False
    assert output["heldout_or_holdout_opened"] is False
    assert output["gpu_accessed"] is False


def test_v54a_gate_contract_requires_both_replicas_and_direct_prose():
    value = builder.build()
    gates = value["ood_gates"]
    assert gates["each_candidate_replica_independently_gated"] is True
    assert gates["both_replicas_required_for_logical_eligibility"] is True
    assert gates["base_relative_qa_mean_reward_delta_minimum"] == 0.0
    assert gates["base_relative_qa_exact_count_delta_minimum"] == 0
    assert gates["base_relative_prose_document_bootstrap_lcb_minimum"] == 0.0
    direct = value["direct_v440_vs_v434_point_gates"]
    assert direct["mean_reward_delta_minimum"] == 0.0
    assert direct["mean_exact_count_delta_minimum"] == 0
    assert direct["mean_prose_token_logprob_delta_minimum"] == 0.0
    assert direct["all_three_direct_gates_required"] is True


def test_v54a_direct_gate_computation_includes_prose_nonregression():
    qa = {}
    prose = {}
    for arm in subject.ARMS:
        qa[arm] = {
            "generated_equal_unit_mean_reward": 1.0,
            "generated_exact_count": 1,
        }
        prose[arm] = {"mean_token_logprob": -1.0}
    qa["v440_equal_a"]["generated_equal_unit_mean_reward"] = 1.1
    qa["v440_equal_b"]["generated_equal_unit_mean_reward"] = 1.1
    prose["v440_equal_a"]["mean_token_logprob"] = -0.9
    prose["v440_equal_b"]["mean_token_logprob"] = -0.9

    original = subject.parent._replica_gate
    subject.parent._replica_gate = lambda *_args: {"eligible": True}
    try:
        table, direct = subject._gate_table_v54a(qa, prose, {})
    finally:
        subject.parent._replica_gate = original
    assert all(
        item["both_replicas_independently_ood_eligible"]
        for item in table.values()
    )
    assert direct["reward_nonnegative"] is True
    assert direct["exact_nonnegative"] is True
    assert direct["prose_nonregression"] is True
    assert direct["all_direct_point_gates_passed"] is True
