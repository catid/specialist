from __future__ import annotations

from pathlib import Path

import pytest

import probe_vllm_two_adapter_switch_v63 as probe


def test_fixed_call_plan_is_fully_counterbalanced() -> None:
    assert probe.CALL_PLAN == (
        "reference", "candidate", "candidate", "reference",
        "reference", "candidate", "candidate", "reference",
    )
    assert probe.CALL_PLAN.count("reference") == 4
    assert probe.CALL_PLAN.count("candidate") == 4


def test_changed_rows_counts_hash_disagreement() -> None:
    calls = [
        {"rows": [{"token_ids_sha256": f"same-{row}"} for row in range(68)]},
        {"rows": [{"token_ids_sha256": (
            "different" if row == 7 else f"same-{row}"
        )} for row in range(68)]},
    ]
    assert probe.changed_rows(calls) == 1


def test_adapter_identity_binds_both_files(tmp_path: Path) -> None:
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "adapter_config.json").write_bytes(b"config")
    value = probe.adapter_identity(tmp_path)
    assert set(value) == {"weights_sha256", "config_sha256"}
    assert all(len(item) == 64 for item in value.values())


def test_adapter_identity_fails_closed_on_incomplete_path(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="incomplete"):
        probe.adapter_identity(tmp_path)


def test_source_is_data_free_and_text_free() -> None:
    source = Path(probe.__file__).read_text()
    assert '"source_dataset_rows_opened": 0' in source
    assert '"prompt_or_generation_text_persisted": False' in source
    assert '"token_ids_persisted": False' in source
    assert '"adapter_update_or_hpo_performed": False' in source
    assert "train_qa" not in source


def test_output_preflight_precedes_vllm_import_and_model_load() -> None:
    source = Path(probe.__file__).read_text()
    preflight = source.index("if output.exists():")
    vllm_import = source.index("from vllm import LLM")
    model_load = source.index("engine = LLM(")
    assert preflight < vllm_import < model_load


def test_engine_and_process_group_are_explicitly_cleaned() -> None:
    source = Path(probe.__file__).read_text()
    assert "engine.llm_engine.engine_core.shutdown()" in source
    assert "dist.destroy_process_group()" in source
    assert '"engine_shutdown_completed": True' in source
