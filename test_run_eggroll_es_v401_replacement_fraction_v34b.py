import copy
import json

import pytest

import run_eggroll_es_v401_replacement_fraction_v34b as runner


def test_cpu_dry_run_is_hash_locked_compact_and_launches_nothing():
    value = runner.build_cpu_dry_run()
    assert value["schema"] == "eggroll-es-v401-replacement-fraction-cpu-dry-run-v34b"
    assert value["GPU_launched"] is False
    assert value["train_only_runtime_launched"] is False
    assert value["this_cpu_bundle_launches_or_reserves_no_gpu"] is True
    assert value["content_sha256_before_self_field"] == runner.canonical_sha256(
        runner._without_self(value)
    )
    assert runner.assert_compact(value) is value


def test_execution_contract_requires_four_tp1_engines_all_waves_and_cleanup():
    value = runner.build_cpu_dry_run()["execution_contract"]
    engines = value["engine_contract"]
    assert engines["gpu_ids"] == [0, 1, 2, 3]
    assert engines["engine_count"] == 4
    assert engines["tensor_parallel_per_engine"] == 1
    assert engines["all_four_engines_active_every_signed_wave"] is True
    assert "engine_cleanup_and_final_all_gpu_idle_gate" == value["phase_order"][-1]
    assert value["alpha"] == 0.0


def test_budget_and_fixed_fraction_sequence_are_exact():
    value = runner.build_cpu_dry_run()["execution_contract"]
    budget = value["request_budget"]
    assert budget["perturbed_requests"] == 49_920
    assert budget["full_context_requests"] == 4_680
    assert budget["fraction_specific_requests"] == 0
    assert budget["total_generation_requests"] == 54_600
    assert value["fraction_contract"]["fractions_in_fixed_test_order"] == [
        0.05, 0.1, 0.2, 0.4, 1.0
    ]
    assert value["fixed_sequence_gate"]["stop_at_first_failure"] is True


def test_every_bound_file_is_rechecked():
    identities = runner.verify_bound_files()
    assert set(identities) == set(runner.BOUND_FILES)
    assert all(len(item["file_sha256"]) == 64 for item in identities.values())


def test_forbidden_cli_and_compact_keys_fail_closed():
    with pytest.raises(ValueError):
        runner.main(["--v34b-dry-run", "--validation-path=x"])
    with pytest.raises(RuntimeError):
        runner.assert_compact({"unit_scores": [1.0]})
    with pytest.raises(ValueError):
        runner.main([])


def test_expected_hashes_gate_dry_run(capsys):
    value = runner.build_cpu_dry_run()
    result = runner.main([
        "--v34b-dry-run",
        "--expected-implementation-bundle-sha256",
        value["implementation"]["bundle_sha256"],
        "--expected-execution-contract-sha256",
        value["execution_contract"]["content_sha256_before_self_field"],
    ])
    assert result["GPU_launched"] is False
    json.loads(capsys.readouterr().out)
    with pytest.raises(ValueError):
        runner.main([
            "--v34b-dry-run",
            "--expected-implementation-bundle-sha256",
            "0" * 64,
        ])


def test_schedule_mutation_fails_execution_contract(monkeypatch):
    prereg = runner.mechanics_v34b.load_hardened_preregistration()
    bundle = runner.mechanics_v34b.materialize_paired_panel_bundle()
    implementation = runner.implementation_identity()
    schedule = runner.mechanics_v34b.resident_signed_wave_schedule()
    schedule[0]["engine_direction_indices"] = [0, 1, 2]
    monkeypatch.setattr(
        runner.mechanics_v34b,
        "resident_signed_wave_schedule",
        lambda: copy.deepcopy(schedule),
    )
    with pytest.raises(RuntimeError):
        runner.execution_contract(prereg, bundle, implementation)
