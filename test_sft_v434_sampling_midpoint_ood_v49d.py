#!/usr/bin/env python3

import json
import types
from pathlib import Path

import build_sft_v434_sampling_midpoint_ood_preregistration_v49d as builder
import run_sft_v434_sampling_midpoint_ood_only_v49d as subject
import stage_v49d_adapters_vllm as stage


def test_v49d_sources_and_stages_are_exact_and_train_only():
    for arm in stage.ARMS:
        seal = stage.source_seal_v49d(arm)
        assert seal["completed_steps"] == 48
        assert seal["shadow_ood_holdout_or_heldout_opened"] is False
        binding = subject.canonical_stage_binding_v49d(arm)
        assert binding["weights_file_sha256"] == subject.STAGE_EXPECTED[arm]["weights"]
        assert binding["tensor_bytes_preserved_exactly"] is True


def test_v49d_ood_prereg_has_exact_two_full_waves_and_no_shadow_input():
    value = builder.build()
    assert value["single_access_inputs"] == subject.OOD_INPUTS
    assert set(value["single_access_inputs"]) == {"ood_qa", "ood_prose"}
    assert value["shadow_access_authorized"] is False
    assert value["heldout_or_holdout_access_authorized"] is False
    assert value["input_scope"]["shadow_or_split_manifest_bound"] is False
    assert value["input_scope"]["sealed_holdout_or_heldout_bound"] is False
    waves = value["runtime"]["two_full_fixed_waves"]
    assert [[item["arm"] for item in wave] for wave in waves] == [
        ["base_a", "base_b", "base_c", "base_d"],
        ["v434_equal_a", "v434_equal_b", "v434_source50_a", "v434_source50_b"],
    ]
    assert all([item["engine_index"] for item in wave] == [0, 1, 2, 3]
               for wave in waves)


def test_v49d_builder_and_dry_run_do_not_open_ood_or_gpu(tmp_path, monkeypatch, capsys):
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
        preregistration_content_sha256=value["content_sha256_before_self_field"],
    )
    subject.load_preregistration_v49d(args)
    monkeypatch.setattr(
        subject.topology, "gpu_preflight",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run reached GPU")),
    )
    monkeypatch.setattr(
        subject.core, "SingleSemanticAccessV44A",
        lambda *_a, **_k: (_ for _ in ()).throw(
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


def test_v49d_gate_contract_preserves_replication_and_thresholds():
    value = builder.build()
    gates = value["ood_gates"]
    assert gates["each_candidate_replica_independently_gated"] is True
    assert gates["both_replicas_required_for_logical_eligibility"] is True
    assert gates["base_relative_qa_mean_reward_delta_minimum"] == 0.0
    assert gates["base_relative_qa_exact_count_delta_minimum"] == 0
    assert gates["base_relative_prose_document_bootstrap_lcb_minimum"] == 0.0
    assert gates["paired_qa_bootstrap_ci_role"] == "informational_not_a_gate"
    direct = value["direct_hypothesis_ood_point_gates"]
    assert direct["mean_reward_delta_minimum"] == 0.0
    assert direct["mean_exact_count_delta_minimum"] == 0
    assert direct["shadow_reward_delta_minimum_deferred"] == 0.0008257591
