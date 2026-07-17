from __future__ import annotations

import json
from pathlib import Path

import pytest

import build_lora_es_mirrored_calibration_preregistration_v66b as builder
import run_lora_es_mirrored_calibration_v66 as v66
import run_lora_es_mirrored_calibration_v66b as runtime
import run_lora_topology_probe_v40a as v40a
import train_eggroll_es_specialist as base


def _write_preregistration(path: Path) -> dict:
    value = builder.build_preregistration_v66b()
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return value


def test_v66b_builder_is_deterministic_and_preserves_failed_v66_receipts():
    first = builder.build_preregistration_v66b()
    second = builder.build_preregistration_v66b()
    assert first == second
    compact = {
        key: item for key, item in first.items()
        if key != "content_sha256_before_self_field"
    }
    assert first["content_sha256_before_self_field"] == (
        v66.mirrored.canonical_sha256_v66(compact)
    )
    assert first["runtime"]["trainer_state_mode"] == "external_worker"
    assert first["runtime"][
        "dense_full_weight_master_install_authorized"
    ] is False
    failed = first["supersedes_failed_v66"]
    assert failed["failure_file_sha256"] == (
        v66.file_sha256_v66(builder.FAILED_RECEIPT)
    )
    assert failed["adapter_install_reached"] is False
    assert failed["final_gpu_idle"] is True
    assert first["artifacts"] == runtime.artifacts_v66b()


def test_v66b_dry_run_resolves_lora_surface_without_writes(tmp_path, capsys):
    prereg = tmp_path / "prereg.json"
    value = _write_preregistration(prereg)
    before = (runtime.ATTEMPT.exists(), runtime.RUN_DIR.exists())
    result = runtime.main([
        "--preregistration", str(prereg),
        "--preregistration-sha256", v66.file_sha256_v66(prereg),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ])
    assert result == 0
    assert (runtime.ATTEMPT.exists(), runtime.RUN_DIR.exists()) == before
    output = json.loads(capsys.readouterr().out)
    assert output["worker_contract"] == {
        "state_mode": "external_worker",
        "worker_extension_cls": v66.WORKER_EXTENSION_V66,
        "required_worker_endpoints": list(
            runtime.REQUIRED_WORKER_ENDPOINTS_V66B
        ),
        "resolved_class": v66.WORKER_EXTENSION_V66,
        "dense_full_weight_master_install_authorized": False,
    }
    assert output["train_semantics_model_ray_or_gpu_loaded"] is False
    assert output["filesystem_writes"] is False


def test_v40a_factory_passes_patched_v66_worker_in_external_mode(monkeypatch):
    captured = {}

    class Parent:
        def __init__(self, *args, **kwargs):
            captured["trainer_args"] = args
            captured["trainer_kwargs"] = kwargs

    def fake_load_trainer(**kwargs):
        captured["factory_kwargs"] = kwargs
        return Parent

    monkeypatch.setattr(v40a.base, "load_trainer", fake_load_trainer)
    monkeypatch.setattr(v40a, "WORKER_EXTENSION", v66.WORKER_EXTENSION_V66)
    trainer = v40a.make_trainer({"runtime": {"tuned_table_content_sha256": "x"}})
    assert isinstance(trainer, Parent)
    assert captured["factory_kwargs"] == {
        "state_mode": base.TRAINER_STATE_MODE_EXTERNAL_WORKER,
        "worker_extension_cls": v66.WORKER_EXTENSION_V66,
        "required_worker_endpoints": v40a.REQUIRED_WORKER_ENDPOINTS,
    }


def test_v66b_namespace_is_fresh_and_restores_v66_globals():
    original = {
        name: getattr(v66, name)
        for name in (
            "RUN_DIR", "ATTEMPT", "GPU_LOG", "POPULATION",
            "UPDATE", "REPORT", "FAILURE",
        )
    }
    assert runtime.RUN_DIR != original["RUN_DIR"]
    with runtime._fresh_namespace_v66b():
        assert v66.RUN_DIR == runtime.RUN_DIR
        assert v66.ATTEMPT == runtime.ATTEMPT
        assert v66.FAILURE == runtime.FAILURE
    assert all(getattr(v66, name) == value for name, value in original.items())


def test_v66b_rejects_missing_lora_surface_before_gpu(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "REQUIRED_WORKER_ENDPOINTS_V66B",
        ("install_adapter_state_v41a", "missing_v66_endpoint"),
    )
    with pytest.raises(RuntimeError, match="missing required endpoints"):
        runtime.validate_lora_worker_contract_v66b()
