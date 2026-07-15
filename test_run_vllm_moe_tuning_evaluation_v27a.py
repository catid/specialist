#!/usr/bin/env python3

import inspect
import json
from pathlib import Path

import pytest

import run_vllm_moe_tuning_evaluation_v27a as runtime


def test_v27a_runtime_binds_exact_preoutput_preregistration():
    value = runtime.load_preregistration()
    assert value["content_sha256_before_self_field"] == runtime.PREREG_CONTENT_SHA256
    assert runtime.PREREG_COMMIT == "2572204429cf2b016d0a001086d309b582b97724"


def test_v27a_tuned_config_validation_is_exact_except_commit_check(tmp_path, monkeypatch):
    path = tmp_path / runtime.EXPECTED_TUNED_FILENAME
    value = {"triton_version": "3.6.0"}
    config = {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    }
    value.update({str(batch): config for batch in runtime.BATCH_SIZES})
    path.write_text(json.dumps(value))
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **k: b"tracked\n")
    monkeypatch.setattr(Path, "relative_to", lambda self, other: Path(self.name))
    assert runtime.validate_tuned_config(path, runtime.file_sha256(path)) == value
    value.pop("2048")
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="batch coverage"):
        runtime.validate_tuned_config(path, runtime.file_sha256(path))


def test_v27a_summary_gate_is_exact_and_cannot_adopt_directly():
    records = []
    for repetition in range(5):
        default = []
        tuned = []
        for gpu, batch in enumerate(runtime.BATCH_SIZES):
            default.append({"batch_size": batch, "kernel_time_microseconds": 103.0})
            tuned.append({"batch_size": batch, "kernel_time_microseconds": 100.0})
        records.append({"repetition": repetition, "default": default, "tuned": tuned})
    summary = runtime.summarize(records)
    assert summary["pass"] is True
    assert summary["global_geometric_mean_speedup"] == pytest.approx(1.03)
    assert summary["direct_recipe_adoption_authorized"] is False
    records[0]["tuned"][0]["kernel_time_microseconds"] = 200.0
    records[1]["tuned"][0]["kernel_time_microseconds"] = 200.0
    assert runtime.summarize(records)["pass"] is False


def test_v27a_real_launch_is_fail_closed_and_dry_run_is_gpu_free(capsys):
    with pytest.raises(ValueError, match="requires all exact expected identities"):
        runtime.main([])
    runtime.main(["--dry-run"])
    value = json.loads(capsys.readouterr().out)
    assert value["gpu_launched"] is False
    assert value["evaluation_launched"] is False


def test_v27a_worker_and_report_close_dataset_model_and_nontrain_surfaces():
    source = inspect.getsource(runtime)
    assert 'ray.remote(num_gpus=1)' in source
    assert 'set(by_gpu) != {0, 1, 2, 3}' in source
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3"' in source
    assert 'tuple(item["uuid"] for item in rows) != GPU_UUIDS' in source
    assert '"model_update_applied": False' in source
    assert '"checkpoint_written": False' in source
    assert '"dataset_surface_opened": False' in source
    assert '"nontrain_evaluation_surface_opened": False' in source
    for forbidden in ("load_frozen_train(", "heldout.arrow", "ood_qa.arrow"):
        assert forbidden not in source
