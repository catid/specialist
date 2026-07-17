from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import build_lora_es_mirrored_calibration_preregistration_v66 as builder
import eggroll_es_mirrored_v66 as mirrored
import run_lora_es_mirrored_calibration_v66 as runtime


def _write_preregistration(path: Path) -> dict:
    value = builder.build_preregistration_v66()
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return value


def test_builder_is_deterministic_narrow_and_binds_current_implementation():
    first = builder.build_preregistration_v66()
    second = builder.build_preregistration_v66()
    assert first == second
    compact = {
        key: item for key, item in first.items()
        if key != "content_sha256_before_self_field"
    }
    assert first["content_sha256_before_self_field"] == (
        builder.canonical_sha256_v66(compact)
    )
    assert first["fixed_recipe"]["direction_count"] == 8
    assert first["fixed_recipe"]["signed_population_size"] == 16
    assert first["fixed_recipe"]["engines"] == 4
    assert first["fixed_recipe"]["tensor_parallel_size"] == 1
    assert first["authorization"] == {
        "gpu_launch": True,
        "train_panel_semantic_access": True,
        "nonzero_update_execute_then_exact_abort": True,
        "dev_access": False,
        "ood_access": False,
        "protected_holdout_access": False,
        "checkpoint_snapshot": False,
        "candidate_commit": False,
        "promotion": False,
    }
    assert first["train_only_input"]["protected_dev_ood_or_holdout_paths"] == []
    for binding in first["implementation_bindings"].values():
        path = Path(binding["path"])
        assert path.is_file()
        assert builder.file_sha256_v66(path) == binding["file_sha256"]


def test_dry_run_validates_exact_prereg_and_performs_no_writes(
    tmp_path, capsys,
):
    prereg = tmp_path / "prereg.json"
    value = _write_preregistration(prereg)
    before = (runtime.ATTEMPT.exists(), runtime.RUN_DIR.exists())
    result = runtime.main([
        "--preregistration", str(prereg),
        "--preregistration-sha256", runtime.file_sha256_v66(prereg),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ])
    assert result == 0
    assert (runtime.ATTEMPT.exists(), runtime.RUN_DIR.exists()) == before
    output = json.loads(capsys.readouterr().out)
    assert output["four_tp1_engines"] is True
    assert output["signed_population_size"] == 16
    assert output["train_semantics_model_ray_or_gpu_loaded"] is False
    assert output["filesystem_writes"] is False
    assert output["protected_dev_ood_or_holdout_opened"] is False


def test_dry_run_rejects_prereg_tamper_even_when_cli_hash_matches(tmp_path):
    prereg = tmp_path / "prereg.json"
    value = _write_preregistration(prereg)
    value["authorization"]["candidate_commit"] = True
    value["content_sha256_before_self_field"] = builder.canonical_sha256_v66({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })
    prereg.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(RuntimeError, match="content changed"):
        runtime.main([
            "--preregistration", str(prereg),
            "--preregistration-sha256", runtime.file_sha256_v66(prereg),
            "--preregistration-content-sha256",
            value["content_sha256_before_self_field"],
            "--dry-run",
        ])


def test_preregistered_recipe_constructs_complete_rotating_v66_plan():
    recipe = builder.build_preregistration_v66()["fixed_recipe"]
    payload = mirrored.common_evaluation_payload_v66(
        [{"prompt_token_ids": [1, 2, 3]}],
        {
            "n": 1, "seed": recipe["evaluation_seed"],
            "temperature": 0.0, "max_tokens": 1,
        },
        {"schema": "synthetic-train-only-judge"},
        recipe["evaluation_seed"],
    )
    plan = mirrored.mirrored_population_plan_v66(
        recipe["direction_seeds"], recipe["sigma"], payload
    )
    assert plan["direction_count"] == 8
    assert plan["signed_population_size"] == 16
    assert plan["wave_count"] == 4
    assert plan["candidates_per_engine"] == 4
    assert all(
        {item["engine_rank"] for item in wave} == {0, 1, 2, 3}
        for wave in plan["waves"]
    )


def test_gpu_wave_summary_requires_every_physical_gpu_positive_each_wave(
    tmp_path,
):
    path = tmp_path / "gpu.jsonl"
    pids = {gpu: 9000 + gpu for gpu in range(4)}
    rows = []
    for wave in range(4):
        for gpu in range(4):
            rows.append({
                "phase": f"mirrored_wave_{wave}_generation_all_actors",
                "gpu": gpu,
                "compute_pids": [pids[gpu]],
                "utilization_percent": 25 + gpu,
                "memory_used_mib": 82000 + gpu,
            })
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = runtime._gpu_wave_summary_v66(path, pids)
    assert summary["all_four_physical_gpus_positive_in_every_wave"] is True
    rows[-1]["utilization_percent"] = 0
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lacked useful activity"):
        runtime._gpu_wave_summary_v66(path, pids)


class _Collective:
    def __init__(self, rank):
        self.rank = rank

    def remote(self, method, args=()):
        return self.rank, method, args


class _FakeUpdateTrainer:
    def __init__(self):
        self.engines = [
            SimpleNamespace(collective_rpc=_Collective(rank))
            for rank in range(4)
        ]
        self.calls = []

    def _resolve(self, handles):
        values = []
        for rank, method, args in handles:
            self.calls.append((rank, method, args))
            if method == "prepare_sharded_adapter_update_v41a":
                values.append([{
                    "manifest_sha256": "a" * 64,
                    "rank": rank,
                    "shard_indices": [rank, rank + 4],
                    "shard_seeds": [args[0][rank], args[0][rank + 4]],
                }])
            elif method == "execute_sharded_adapter_update_v41a":
                values.append([{
                    "candidate_identity": {"sha256": "b" * 64},
                    "materialization": {"runtime_values_sha256": "c" * 64},
                    "master_committed": False,
                }])
            elif method == "abort_mirrored_update_if_present_v66":
                values.append([{
                    "restored_identity": {"sha256": runtime.MASTER_SHA256_V66},
                    "terminal_poisoned": False,
                }])
            else:
                raise AssertionError(method)
        return values


def test_nonzero_update_harness_executes_on_four_ranks_then_always_aborts():
    trainer = _FakeUpdateTrainer()
    phase = SimpleNamespace(value="setup")
    update = {
        "direction_seeds": list(builder.DIRECTION_SEEDS_V66),
        "coefficients": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
        "coefficient_sha256": "d" * 64,
        "worker_population_size": 8,
        "worker_alpha": 0.125,
        "effective_noise_scale": 0.015625,
    }
    receipt = runtime._execute_and_abort_nonzero_update_v66(
        trainer,
        update,
        runtime.MASTER_SHA256_V66,
        reference_generation=2,
        phase=phase,
    )
    assert receipt["candidate_differs_from_master"] is True
    assert receipt["candidate_runtime_differs_from_master"] is True
    assert receipt["master_committed"] is False
    assert receipt["all_four_abort_receipts_exact"] is True
    assert receipt["nonzero_pair_differences"] == 8
    methods = [method for _rank, method, _args in trainer.calls]
    assert methods.count("prepare_sharded_adapter_update_v41a") == 4
    assert methods.count("execute_sharded_adapter_update_v41a") == 4
    assert methods.count("abort_mirrored_update_if_present_v66") == 4


def test_nonzero_update_harness_rejects_zero_without_touching_workers():
    trainer = _FakeUpdateTrainer()
    update = {
        "direction_seeds": list(builder.DIRECTION_SEEDS_V66),
        "coefficients": [0.0] * 8,
        "coefficient_sha256": "d" * 64,
        "worker_population_size": 8,
        "worker_alpha": 0.125,
        "effective_noise_scale": 0.015625,
    }
    with pytest.raises(RuntimeError, match="zero pair update"):
        runtime._execute_and_abort_nonzero_update_v66(
            trainer,
            update,
            runtime.MASTER_SHA256_V66,
            reference_generation=2,
            phase=SimpleNamespace(value="setup"),
        )
    assert trainer.calls == []
