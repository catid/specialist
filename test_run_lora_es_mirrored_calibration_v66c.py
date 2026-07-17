from __future__ import annotations

import json
from pathlib import Path

import build_lora_es_mirrored_calibration_preregistration_v66c as builder
import run_lora_es_mirrored_calibration_v66 as v66
import run_lora_es_mirrored_calibration_v66c as runtime


def _write_preregistration(path: Path) -> dict:
    value = builder.build_preregistration_v66c()
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return value


def test_v66c_builder_is_deterministic_and_seals_both_failed_attempts():
    first = builder.build_preregistration_v66c()
    second = builder.build_preregistration_v66c()
    assert first == second
    compact = {
        key: item for key, item in first.items()
        if key != "content_sha256_before_self_field"
    }
    assert first["content_sha256_before_self_field"] == (
        v66.mirrored.canonical_sha256_v66(compact)
    )
    assert set(first["supersedes_failed_attempts"]) == {"v66", "v66b"}
    assert all(
        item["final_gpu_idle"] is True
        for item in first["supersedes_failed_attempts"].values()
    )
    assert first["supersedes_failed_attempts"]["v66b"][
        "trainer_constructed_in_external_lora_mode"
    ] is True
    assert first["fixed_recipe"]["lora_activation"] == {
        "request_name": "matched_lora_initialization_v41b",
        "request_id": 1,
        "request_path": str(builder.v66.STAGED_ADAPTER),
        "four_actor_add_lora_required": True,
        "active_slot_certificate_required_before_state_write": True,
        "active_slot_index": 0,
    }
    assert first["artifacts"] == runtime.artifacts_v66c()


def test_v66c_dry_run_seals_activation_order_and_performs_no_writes(
    tmp_path, capsys,
):
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
    assert output["activation_order"] == [
        "add_lora_all_four",
        "active_slot_certificate_all_four",
        "install_adapter_state_all_four",
        "canonical_certificate_all_four",
    ]
    assert output["worker_contract"]["required_worker_endpoints"] == list(
        runtime.REQUIRED_WORKER_ENDPOINTS_V66C
    )
    assert output["worker_contract"][
        "dense_full_weight_master_install_authorized"
    ] is False
    assert output["train_semantics_model_ray_or_gpu_loaded"] is False
    assert output["filesystem_writes"] is False


def test_v66c_namespace_never_reuses_v66_or_v66b_paths():
    assert "v66c_" in runtime.RUN_DIR.name
    assert runtime.RUN_DIR != v66.RUN_DIR
    assert runtime.RUN_DIR != builder.v66b.RUN
    original = {
        name: getattr(v66, name)
        for name in (
            "RUN_DIR", "ATTEMPT", "GPU_LOG", "POPULATION",
            "UPDATE", "REPORT", "FAILURE",
        )
    }
    with runtime._fresh_namespace_v66c():
        assert v66.RUN_DIR == runtime.RUN_DIR
        assert v66.ATTEMPT == runtime.ATTEMPT
        assert v66.FAILURE == runtime.FAILURE
    assert all(getattr(v66, name) == value for name, value in original.items())


def test_v66c_worker_contract_includes_preinstall_active_slot_endpoint():
    contract = runtime.validate_lora_worker_contract_v66c()
    assert contract["state_mode"] == "external_worker"
    assert contract["resolved_class"] == v66.WORKER_EXTENSION_V66
    assert "active_lora_slot_certificate_v66" in contract[
        "required_worker_endpoints"
    ]
