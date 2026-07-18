from __future__ import annotations

import json

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
