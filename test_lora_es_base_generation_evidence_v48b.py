#!/usr/bin/env python3

import json
import types

import pytest

import build_lora_es_base_generation_evidence_preregistration_v48b as builder
import lora_es_generation_boundary_sampling_v48a as boundary
import run_lora_es_base_generation_evidence_v48b as subject


class _Completion:
    def __init__(self, text):
        self.text = text


class _Output:
    def __init__(self, text):
        self.outputs = [_Completion(text)]


def test_v48b_train_inputs_are_exact_and_need_no_original_split_runtime():
    rows, membership = subject.load_train_inputs_v48b()
    assert len(rows) == 448
    assert len({row["row_sha256"] for row in rows}) == 448
    assert len({row["unit_identity_sha256"] for row in rows}) == 208
    assert membership["runtime_requires_original_split_commitment"] is False
    assert all(token not in str(subject.TRAIN_DATASET).casefold()
               for token in subject.FORBIDDEN_PATH_TOKENS)


def test_v48b_actor_scoring_and_evidence_persist_no_raw_text():
    rows, membership = subject.load_train_inputs_v48b()
    outputs = [_Output(row["answer"]) for row in rows]
    actors = [subject.score_actor_outputs_v48b(rows, outputs, actor)
              for actor in range(4)]
    evidence = subject.build_evidence_v48b(
        rows, membership, actors,
        {"sha256": subject.v43m.V43I_RESTORED_MASTER_SHA256},
    )
    assert evidence["row_count"] == 448
    assert evidence["actor_count"] == 4
    assert evidence["raw_question_answer_or_generation_text_persisted"] is False
    serialized = json.dumps(evidence, sort_keys=True)
    assert '"question"' not in serialized
    assert '"answer"' not in serialized
    selector_bundle = {
        "row_sha256": [row["row_sha256"] for row in rows],
        "unit_membership_v48a": [{
            "row_sha256": row["row_sha256"],
            "unit_identity_sha256": row["unit_identity_sha256"],
            "row_count": row["row_count"],
        } for row in rows],
        "train_bundle_content_sha256": subject.v43i.TRAIN_BUNDLE_SHA256,
    }
    subset = boundary.build_fragile_subset_v48a(selector_bundle, evidence)
    assert subset["selected_conflict_units"] == 64


def test_v48b_preregistration_and_dry_run_open_zero_semantics_or_gpus(
    tmp_path, monkeypatch, capsys,
):
    value = builder.build_v48b()
    assert value["gpu_launch_authorized"] is True
    assert value["access_contract"]["original_split_manifest_opened_at_runtime"] is False
    assert value["shadow_ood_holdout_or_benchmark_authorized"] is False
    assert value["evidence_contract"]["greedy_completions"] == 1792
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    file_sha = subject.v43i.v40a.file_sha256(path)
    args = types.SimpleNamespace(
        preregistration=str(path), preregistration_sha256=file_sha,
        preregistration_content_sha256=value[
            "content_sha256_before_self_field"
        ],
    )
    subject.load_preregistration_v48b(args)
    monkeypatch.setattr(
        subject, "load_train_inputs_v48b",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run loaded train")),
    )
    monkeypatch.setattr(
        subject, "_make_trainer_v48b",
        lambda prereg: (_ for _ in ()).throw(AssertionError("dry-run made trainer")),
    )
    assert subject.main([
        "--preregistration", str(path),
        "--preregistration-sha256", file_sha,
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["protected_semantic_access_count"] == 0
    assert output["model_or_gpu_loaded"] is False
    assert output["filesystem_writes"] is False


def test_v48b_implementation_hashes_bind_every_runtime_input():
    value = builder.build_v48b()
    assert value["implementation_bindings"] == subject.implementation_bindings_v48b()
    assert value["recipe"]["train_dataset_sha256"] == subject.EXPECTED_TRAIN_SHA256
    assert value["recipe"]["membership_file_sha256"] == (
        subject.EXPECTED_MEMBERSHIP_SHA256
    )


def test_v48c_activates_all_four_lora_slots_without_generation(monkeypatch):
    request = object()
    calls = []
    monkeypatch.setattr(subject.v43i, "_lora_request", lambda: request)
    monkeypatch.setattr(
        subject.v43i.v40a, "_rpc_all",
        lambda trainer, method, args=(): (
            calls.append((trainer, method, args)) or [True, True, True, True]
        ),
    )
    trainer = object()
    receipt = subject.activate_adapter_slots_v48c(trainer)
    assert calls == [(trainer, "add_lora", (request,))]
    assert receipt["actors"] == 4
    assert receipt["generation_completions"] == 0


def test_v48c_rejects_incomplete_lora_slot_activation(monkeypatch):
    monkeypatch.setattr(subject.v43i, "_lora_request", object)
    monkeypatch.setattr(
        subject.v43i.v40a, "_rpc_all",
        lambda trainer, method, args=(): [True, True, False, True],
    )
    with pytest.raises(RuntimeError, match="four-actor LoRA slot activation"):
        subject.activate_adapter_slots_v48c(object())
