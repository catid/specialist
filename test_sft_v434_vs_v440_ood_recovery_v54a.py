#!/usr/bin/env python3

import json
from pathlib import Path

import build_sft_v434_vs_v440_ood_recovery_preregistration_v54a as builder
import recover_sft_v434_vs_v440_ood_only_v54a as subject


def test_recovery_builder_reads_receipts_not_semantic_inputs(monkeypatch):
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


def test_receipt_inventory_has_exact_arms_phases_and_two_waves():
    raw = subject._load_json(subject.SOURCE_RAW)
    value = subject.receipt_inventory_v54a(raw)
    assert value["arms"] == list(subject.ARMS)
    assert value["phases"] == ["ood_qa", "ood_prose"]
    assert value["total_arm_rows"] == 320
    for phase in subject.PHASES:
        waves = value["phase_inventory"][phase]["wave_receipts"]
        assert [
            [item["arm"] for item in wave["engine_arm_map"]]
            for wave in waves
        ] == [list(subject.BASE_ARMS), list(subject.CANDIDATE_ARMS)]
        assert all(
            [item["engine_index"] for item in wave["engine_arm_map"]]
            == [0, 1, 2, 3]
            for wave in waves
        )


def test_gpu_receipt_requires_only_ood_phases_and_all_four_positive():
    value = subject.gpu_receipt_v54a(subject.SOURCE_GPU_LOG)
    assert value["phase_labels_exact"] == ["ood_qa", "ood_prose"]
    assert value["shadow_phase_present"] is False
    assert value["all_four_attributed_positive_each_ood_phase"] is True
    assert set(value["by_gpu"]) == {"0", "1", "2", "3"}


def test_dry_run_does_not_parse_raw_or_access_gpu(tmp_path, monkeypatch, capsys):
    value = builder.build()
    value.pop("content_sha256_before_self_field")
    value["content_sha256_before_self_field"] = subject.canonical_sha256(value)
    path = tmp_path / "recovery_prereg.json"
    subject.atomic_json(path, value)
    monkeypatch.setattr(
        subject, "recover_v54a",
        lambda *_: (_ for _ in ()).throw(AssertionError("dry-run parsed raw")),
    )
    args = [
        "--preregistration", str(path),
        "--preregistration-sha256", subject.file_sha256(path),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ]
    assert subject.main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["protected_semantic_input_reads"] == 0
    assert output["model_or_generation_accessed"] is False
    assert output["gpu_accessed"] is False
    assert output["shadow_opened"] is False
    assert output["heldout_or_holdout_opened"] is False


def test_direct_gate_uses_replica_means():
    qa = {
        arm: {
            "generated_equal_unit_mean_reward": reward,
            "generated_exact_count": exact,
        }
        for arm, reward, exact in [
            ("v434_equal_a", 0.4, 8), ("v434_equal_b", 0.6, 10),
            ("v440_equal_a", 0.5, 9), ("v440_equal_b", 0.7, 11),
        ]
    }
    prose = {
        "v434_equal_a": {"mean_token_logprob": -1.2},
        "v434_equal_b": {"mean_token_logprob": -1.0},
        "v440_equal_a": {"mean_token_logprob": -1.1},
        "v440_equal_b": {"mean_token_logprob": -0.9},
    }
    # Exercise the direct calculation without constructing synthetic rows for
    # the inherited independent replica gate.
    original = subject.base._replica_gate
    subject.base._replica_gate = lambda *_: {"eligible": True}
    try:
        _, direct = subject._gate_table_v54a(qa, prose, {})
    finally:
        subject.base._replica_gate = original
    assert abs(direct["v440_minus_v434_mean_reward"] - 0.1) < 1e-12
    assert direct["v440_minus_v434_mean_exact_count"] == 1.0
    assert abs(direct["v440_minus_v434_mean_prose_token_logprob"] - 0.1) < 1e-12
    assert direct["all_direct_point_gates_passed"] is True
