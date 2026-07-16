#!/usr/bin/env python3

import json
import types
from pathlib import Path

import build_sft_v434_ood_offline_recovery_preregistration_v49d as builder
import recover_sft_v434_sampling_midpoint_ood_only_v49d as subject


def test_recovery_builder_binds_only_receipts_not_semantic_inputs(monkeypatch):
    protected = {
        str((subject.ROOT / "data/ood_qa_v3.jsonl").resolve()),
        str((subject.ROOT / "data/ood_prose_v3.jsonl").resolve()),
    }
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert str(Path(path).resolve()) not in protected
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    value = builder.build()
    assert value["generation_or_gpu_access_authorized"] is False
    assert value["protected_semantic_input_access_authorized"] is False
    assert value["access_contract"]["semantic_source_files_allowed"] == []
    assert value["expected_receipt_inventory"]["total_arm_rows"] == 320
    assert value["expected_gpu_receipt"]["phase_labels_exact"] == [
        "ood_qa", "ood_prose"
    ]


def test_receipt_inventory_requires_exact_arms_phases_and_two_waves():
    raw = subject._load_json(subject.SOURCE_RAW)
    value = subject.receipt_inventory_v49d(raw)
    assert value["arms"] == list(subject.ARMS)
    assert value["phases"] == ["ood_qa", "ood_prose"]
    assert value["total_arm_rows"] == 320
    for phase in subject.PHASES:
        waves = value["phase_inventory"][phase]["wave_receipts"]
        assert [[item["arm"] for item in wave["engine_arm_map"]] for wave in waves] == [
            list(subject.BASE_ARMS), list(subject.CANDIDATE_ARMS)
        ]
        assert all(
            [item["engine_index"] for item in wave["engine_arm_map"]] == [0, 1, 2, 3]
            for wave in waves
        )


def test_gpu_receipt_requires_only_ood_phases_and_all_four_positive():
    value = subject.gpu_receipt_v49d(subject.SOURCE_GPU_LOG)
    assert value["phase_labels_exact"] == ["ood_qa", "ood_prose"]
    assert value["shadow_phase_present"] is False
    assert value["all_four_attributed_positive_each_ood_phase"] is True
    assert set(value["by_gpu"]) == {"0", "1", "2", "3"}


def test_dry_run_does_not_parse_receipts_or_access_gpu(tmp_path, monkeypatch, capsys):
    value = builder.build()
    value.pop("content_sha256_before_self_field")
    value["content_sha256_before_self_field"] = subject.canonical_sha256(value)
    path = tmp_path / "recovery_prereg.json"
    subject.atomic_json(path, value)
    monkeypatch.setattr(
        subject, "recover_v49d",
        lambda *_: (_ for _ in ()).throw(AssertionError("dry-run parsed receipts")),
    )
    args = [
        "--preregistration", str(path),
        "--preregistration-sha256", subject.file_sha256(path),
        "--preregistration-content-sha256", value["content_sha256_before_self_field"],
        "--dry-run",
    ]
    assert subject.main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["protected_semantic_input_reads"] == 0
    assert output["model_or_generation_accessed"] is False
    assert output["gpu_accessed"] is False
    assert output["shadow_opened"] is False
    assert output["heldout_or_holdout_opened"] is False


def test_metric_contract_on_synthetic_receipts(monkeypatch):
    rows = [{
        "item_index": 0,
        "item_sha256": subject.canonical_sha256({"question": "Q?", "answer": "A"}),
        "question": "Q?", "answer": "A", "response": "A",
        "teacher": {"mean_answer_token_logprob": -0.25},
        "format": "exact", "reward": 1.0,
        "counters": {
            "protocol_token_emission": 0,
            "prompt_echo": 0,
            "empty_extracted_answer": 0,
        },
    }]
    value = subject._qa_aggregate(rows)
    assert value["generated_row_mean_reward"] == 1.0
    assert value["generated_exact_count"] == 1
    assert value["teacher_forced_equal_unit_mean_answer_logprob"] == -0.25
