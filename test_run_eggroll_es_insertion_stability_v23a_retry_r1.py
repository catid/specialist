import copy
import inspect
import json
from pathlib import Path

import pytest

import eggroll_es_insertion_stability_preregistration_v23a as prereg_v23a
import run_eggroll_es_insertion_stability_v23a as original_v23a
import run_eggroll_es_insertion_stability_v23a_retry_r1 as retry_r1


MAIN_RUNS = Path("/home/catid/specialist/experiments/eggroll_es_hpo/runs")
MAIN_ATTEMPT = MAIN_RUNS / original_v23a.ATTEMPT_NAME_V23A
MAIN_REPORT = MAIN_RUNS / original_v23a.EXPERIMENT_NAME_V23A / original_v23a.REPORT_NAME_V23A


def _prereg():
    return original_v23a._load_preregistration_v23a()


def test_v23a_r1_revalidates_compact_original_failure_without_copying_traceback():
    value = retry_r1.validate_original_failure_r1(MAIN_ATTEMPT, MAIN_REPORT)
    assert value["failed_attempt"]["status"] == "failed"
    assert value["failed_attempt"]["model_update_applied"] is False
    assert value["failed_attempt"]["nontrain_surface_opened"] is False
    assert value["failed_attempt"]["compact_report_absent"] is True
    assert value["traceback_or_model_repr_persisted"] is False
    assert value["row_or_response_content_persisted"] is False


def test_v23a_r1_recipe_preserves_basis_gate_panels_and_mapping():
    prereg = _prereg()
    evidence = retry_r1.validate_original_failure_r1(MAIN_ATTEMPT, MAIN_REPORT)
    implementation = {"bundle_sha256": "a" * 64}
    retry = retry_r1.recipe_r1(prereg, implementation, evidence)
    original = original_v23a.recipe_v23a(prereg, implementation)
    for key in ("arms", "panel_contract", "fresh_basis", "runtime", "analysis", "authority"):
        assert retry[key] == original[key]
    assert retry["experiment_name"] == retry_r1.EXPERIMENT_NAME_R1
    assert retry["worker_extension"] == retry_r1.WORKER_EXTENSION_R1
    assert retry["same_preregistered_basis_gate_panels_and_arm_mapping"] is True
    assert retry["seed_domain_repair"]["perturbation_basis_unchanged"] is True
    assert retry["seed_domain_repair"]["explicit_torch_generator_receives_full_seed"] is True
    assert retry["seed_domain_repair"]["only_numpy_legacy_seed_is_projected"] is True
    assert retry["content_sha256_before_self_field"] == retry_r1.canonical_sha256(
        retry_r1._without_self(retry)
    )


def test_v23a_r1_paths_are_exclusive_and_disjoint_from_failed_attempt():
    assert retry_r1.ATTEMPT_NAME_R1 != original_v23a.ATTEMPT_NAME_V23A
    assert retry_r1.EXPERIMENT_NAME_R1 != original_v23a.EXPERIMENT_NAME_V23A
    assert retry_r1.REPORT_NAME_R1 != original_v23a.REPORT_NAME_V23A
    retry_paths = {
        retry_r1.OUTPUT_DIRECTORY_R1 / retry_r1.ATTEMPT_NAME_R1,
        retry_r1.OUTPUT_DIRECTORY_R1 / retry_r1.EXPERIMENT_NAME_R1,
        retry_r1.OUTPUT_DIRECTORY_R1 / retry_r1.EXPERIMENT_NAME_R1 / retry_r1.REPORT_NAME_R1,
    }
    original_paths = {
        retry_r1.ORIGINAL_ATTEMPT_PATH_R1,
        retry_r1.ORIGINAL_REPORT_PATH_R1.parent,
        retry_r1.ORIGINAL_REPORT_PATH_R1,
    }
    assert retry_paths.isdisjoint(original_paths)
    source = inspect.getsource(retry_r1.run_exact_r1)
    assert "validate_original_failure_r1()" in source
    assert "_exclusive_write_json_v23a(attempt_path, attempt)" in source


def test_v23a_r1_all_four_worker_certificates_must_match():
    evidence = retry_r1.validate_original_failure_r1(MAIN_ATTEMPT, MAIN_REPORT)
    projection = retry_r1.seed_projection_contract_r1(_prereg(), evidence)
    report = {
        "schema": "eggroll-es-v23a-seed-projection-worker-certificate-r1",
        **{key: projection[key] for key in (
            "direction_count", "direction_seed_list_sha256",
            "full_to_numpy_projection_sha256", "numpy_projection_unique_count",
            "numpy_projection_contains_zero", "python_random_receives_full_seed",
            "torch_global_receives_full_seed", "torch_cuda_all_receives_full_seed",
            "explicit_torch_generator_receives_full_seed",
            "only_numpy_legacy_seed_is_projected",
        )},
    }
    certificate = retry_r1._validate_worker_projection_reports_r1([copy.deepcopy(report)] * 4, projection)
    assert certificate["worker_count"] == 4
    assert certificate["all_four_workers_identical"] is True
    assert certificate["certificate_completed_before_reference_scoring"] is True
    changed = [copy.deepcopy(report) for _ in range(4)]
    changed[-1]["direction_count"] = 31
    with pytest.raises(RuntimeError, match="certificate changed"):
        retry_r1._validate_worker_projection_reports_r1(changed, projection)


def test_v23a_r1_launcher_uses_only_new_worker_extension():
    source = inspect.getsource(retry_r1.load_runtime_trainer_r1)
    assert '"worker_extension_cls": WORKER_EXTENSION_R1' in source
    assert original_v23a.WORKER_EXTENSION_V23A not in source
    assert '"tensor_parallel_size": 1' in source
    assert "for rank, arm in enumerate(prereg_v23a.ARM_ORDER_V23A)" in source


def test_v23a_r1_dry_run_revalidates_failure_and_launches_no_gpu(monkeypatch, capsys):
    monkeypatch.setattr(retry_r1, "ORIGINAL_ATTEMPT_PATH_R1", MAIN_ATTEMPT)
    monkeypatch.setattr(retry_r1, "ORIGINAL_REPORT_PATH_R1", MAIN_REPORT)
    value = retry_r1.main(["--v23a-r1-dry-run"])
    captured = json.loads(capsys.readouterr().out)
    assert value == captured
    assert value["gpu_launched"] is False
    assert value["original_failed_attempt_revalidated"] is True
    assert value["new_retry_paths_exclusive_and_disjoint"] is True
    assert value["recipe"]["worker_extension"] == retry_r1.WORKER_EXTENSION_R1
    assert value["content_sha256_before_self_field"] == retry_r1.canonical_sha256(
        retry_r1._without_self(value)
    )
