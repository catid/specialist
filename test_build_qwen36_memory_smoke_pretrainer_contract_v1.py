from __future__ import annotations

import copy
import json

import pytest
import build_qwen36_memory_smoke_pretrainer_contract_v1 as contract


def _checked() -> dict:
    value = json.loads(contract.OUTPUT.read_text(encoding="utf-8"))
    unsigned = dict(value)
    declared = unsigned.pop("content_sha256_before_self_field")
    assert contract.canonical_sha256(unsigned) == declared
    return value


def test_contract_is_pretrainer_only_and_synthetic_data_only():
    value = _checked()
    assert value["schema"] == contract.SCHEMA
    assert value["authority"] == {
        "synthetic_token_ids_only": True,
        "dataset_or_evaluation_source_opened": False,
        "actual_checkpoint_weights_loaded": True,
        "actual_expert_lora_adapters_attached": True,
        "one_optimizer_update_per_receipt": True,
        "training_dataset_opened": False,
        "training_launched": False,
        "final_mixed_trainer_validated": False,
        "checkpoint_resume_validated": False,
    }


def test_every_planned_adapter_shape_fits_with_operational_headroom():
    value = _checked()
    receipts = value["receipts"]
    assert [item["physical_gpu_index"] for item in receipts] == [0, 1, 2, 3]
    assert [item["configuration"]["routed_rank"] for item in receipts] == [
        None, 2, 4, 4,
    ]
    assert all(item["configuration"]["sequence_length"] == 2048 for item in receipts)
    assert all(item["configuration"]["dtype"] == "torch.bfloat16" for item in receipts)
    assert all(item["measurements"]["headroom_gib"] >= 8.0 for item in receipts)
    assert value["result"]["minimum_headroom_gib"] >= 8.0
    assert value["result"]["pretrainer_memory_smoke_passed"] is True
    assert value["result"]["final_trainer_memory_gate_pending"] is True


def test_receipts_remain_content_addressed_and_check_rebuild_is_exact():
    value = _checked()
    for item in value["receipts"]:
        path = contract.ROOT / item["artifact"]
        assert path.stat().st_size == item["artifact_bytes"]
        assert contract.file_sha256(path) == item["artifact_sha256"]
    assert contract.build() == value


def _synthetic_receipt_copy() -> dict:
    path = (
        contract.INPUT_DIRECTORY / "gpu2_routed_r4_shared_r16_seq2048.json"
    )
    return copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))


def _validate_synthetic_copy(monkeypatch, tmp_path, value: dict) -> dict:
    path = tmp_path / "gpu2_routed_r4_shared_r16_seq2048.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(contract, "ROOT", tmp_path)
    return contract._validate_receipt(
        path,
        gpu_index=2,
        shared_only=False,
        routed_rank=4,
        shared_rank=16,
        trainable_elements=235_601_920,
    )


def test_receipt_rejects_wrong_gpu_model(monkeypatch, tmp_path):
    value = _synthetic_receipt_copy()
    value["gpu"]["name"] = "Synthetic Lookalike GPU"
    with pytest.raises(RuntimeError, match="wrong GPU model"):
        _validate_synthetic_copy(monkeypatch, tmp_path, value)


def test_receipt_rejects_peak_memory_below_current_memory(monkeypatch, tmp_path):
    value = _synthetic_receipt_copy()
    stage = value["measurement"]["after_step"]
    stage["peak_allocated_gib"] = 50.0
    stage["peak_reserved_gib"] = 51.0
    with pytest.raises(RuntimeError, match="peak allocated below current allocated"):
        _validate_synthetic_copy(monkeypatch, tmp_path, value)


def test_receipt_rejects_unbound_adapter_scope_identity(monkeypatch, tmp_path):
    value = _synthetic_receipt_copy()
    value["scope"]["postattach_identity_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="unexpected postattach_identity_sha256"):
        _validate_synthetic_copy(monkeypatch, tmp_path, value)


def test_receipt_rejects_internally_inconsistent_throughput(monkeypatch, tmp_path):
    value = _synthetic_receipt_copy()
    value["measurement"]["tokens_per_second"] *= 2
    with pytest.raises(RuntimeError, match="throughput does not match"):
        _validate_synthetic_copy(monkeypatch, tmp_path, value)
