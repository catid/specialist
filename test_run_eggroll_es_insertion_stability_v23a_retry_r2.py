import copy
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_eggroll_es_insertion_stability_v23a_retry_r1 as r1
import run_eggroll_es_insertion_stability_v23a_retry_r2 as r2


MAIN_RUNS = Path("/home/catid/specialist/experiments/eggroll_es_hpo/runs")
MAIN_ORIGINAL_ATTEMPT = MAIN_RUNS / r1.runtime_v23a.ATTEMPT_NAME_V23A
MAIN_ORIGINAL_REPORT = (
    MAIN_RUNS / r1.runtime_v23a.EXPERIMENT_NAME_V23A
    / r1.runtime_v23a.REPORT_NAME_V23A
)
MAIN_R1_ATTEMPT = MAIN_RUNS / r1.ATTEMPT_NAME_R1
MAIN_R1_REPORT = MAIN_RUNS / r1.EXPERIMENT_NAME_R1 / r1.REPORT_NAME_R1


def _prereg():
    return r1.runtime_v23a._load_preregistration_v23a()


def _observation():
    expected = r2.expected_environment_contract_r2()
    return {
        key: copy.deepcopy(expected[key])
        for key in (
            "sys_prefix", "sys_executable", "compat_path", "upstream_path",
            "compat_precedes_upstream_in_worker_pythonpath", "versions",
            "vllm_network_helpers_callable_after_compatibility_shim",
            "es_nccl_llm_importable_after_compatibility_shim",
            "trainer_class_importable_after_compatibility_shim",
            "fresh_worker_subprocess_import_succeeds",
            "cuda_visible_devices", "cuda_device_count",
        )
    } | {
        "module_files": copy.deepcopy(expected["module_files"])
    }


def test_v23a_r2_environment_validator_is_exact_and_fail_closed():
    certificate = r2.validate_environment_observation_r2(_observation())
    assert certificate["completed_before_attempt_claim"] is True
    assert certificate["cuda_device_count"] == 4
    assert certificate["content_sha256_before_self_field"] == r2.canonical_sha256(
        r2._without_self(certificate)
    )
    for key, value in (
        ("sys_prefix", "/wrong/venv"),
        ("cuda_visible_devices", "0,1,2"),
        ("cuda_device_count", 3),
    ):
        changed = _observation()
        changed[key] = value
        with pytest.raises(RuntimeError, match="runtime environment changed"):
            r2.validate_environment_observation_r2(changed)
    changed = _observation()
    changed["module_files"]["vllm"]["file_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="module identity changed"):
        r2.validate_environment_observation_r2(changed)


def test_v23a_r2_recipe_preserves_all_scoring_mechanics():
    prereg = _prereg()
    seed_failure = r1.validate_original_failure_r1(
        MAIN_ORIGINAL_ATTEMPT, MAIN_ORIGINAL_REPORT
    )
    environment_failure = r2.validate_r1_environment_failure_r2(
        MAIN_R1_ATTEMPT, MAIN_R1_REPORT
    )
    implementation = {"bundle_sha256": "a" * 64}
    retry = r2.recipe_r2(prereg, implementation, seed_failure, environment_failure)
    inherited = r1.recipe_r1(prereg, implementation, seed_failure)
    for key in (
        "arms", "panel_contract", "fresh_basis", "runtime", "analysis",
        "authority", "worker_extension", "seed_domain_repair",
    ):
        assert retry[key] == inherited[key]
    assert retry["experiment_name"] == r2.EXPERIMENT_NAME_R2
    assert retry["same_preregistered_basis_gate_panels_arm_mapping_and_seed_repair"] is True
    assert retry["runtime_environment_contract_r2"]["completed_before_attempt_claim"] is True
    assert retry["content_sha256_before_self_field"] == r2.canonical_sha256(
        r2._without_self(retry)
    )


def test_v23a_r2_paths_are_disjoint_and_failed_attempt_is_preserved():
    r2_paths = {
        r2.OUTPUT_DIRECTORY_R2 / r2.ATTEMPT_NAME_R2,
        r2.OUTPUT_DIRECTORY_R2 / r2.EXPERIMENT_NAME_R2,
        r2.OUTPUT_DIRECTORY_R2 / r2.EXPERIMENT_NAME_R2 / r2.REPORT_NAME_R2,
    }
    earlier = {
        r1.ORIGINAL_ATTEMPT_PATH_R1,
        r1.ORIGINAL_REPORT_PATH_R1.parent,
        r1.ORIGINAL_REPORT_PATH_R1,
        MAIN_R1_ATTEMPT,
        MAIN_R1_REPORT.parent,
        MAIN_R1_REPORT,
    }
    assert r2_paths.isdisjoint(earlier)
    attempt, run_directory, report = (
        r2.OUTPUT_DIRECTORY_R2 / r2.ATTEMPT_NAME_R2,
        r2.OUTPUT_DIRECTORY_R2 / r2.EXPERIMENT_NAME_R2,
        r2.OUTPUT_DIRECTORY_R2 / r2.EXPERIMENT_NAME_R2 / r2.REPORT_NAME_R2,
    )
    assert attempt.exists()
    assert run_directory.exists()
    assert not report.exists()


def test_v23a_r2_real_launch_hashes_are_fail_closed():
    prereg = _prereg()
    implementation = {"bundle_sha256": "a" * 64}
    recipe = {"content_sha256_before_self_field": "b" * 64}
    with pytest.raises(ValueError, match="requires expected implementation"):
        r2.validate_runtime_r2(
            SimpleNamespace(
                v23a_r2_dry_run=False,
                expected_implementation_bundle_sha256=None,
                expected_recipe_sha256=None,
            ), prereg, implementation, recipe,
        )
    with pytest.raises(ValueError, match="implementation hash changed"):
        r2.validate_runtime_r2(
            SimpleNamespace(
                v23a_r2_dry_run=False,
                expected_implementation_bundle_sha256="0" * 64,
                expected_recipe_sha256="b" * 64,
            ), prereg, implementation, recipe,
        )


def test_v23a_r2_environment_certificate_precedes_attempt_claim_and_scoring():
    source = inspect.getsource(r2.run_exact_r2)
    assert source.index("certify_runtime_environment_r2()") < source.index(
        "_exclusive_write_json_v23a(attempt_path, attempt)"
    )
    assert source.index("certify_runtime_environment_r2()") < source.index(
        "estimate_insertion_stability_v23a()"
    )
    assert "_make_trainer_r2" in source
    factory = inspect.getsource(r2._make_trainer_r2)
    assert "experiment_name=EXPERIMENT_NAME_R2" in factory
    assert "save_best_models=False" in factory
    assert "eval_dataloader_dict={}" in factory


def test_v23a_r2_keeps_update_and_nontrain_surfaces_closed():
    prereg = _prereg()
    seed_failure = r1.validate_original_failure_r1(
        MAIN_ORIGINAL_ATTEMPT, MAIN_ORIGINAL_REPORT
    )
    environment_failure = r2.validate_r1_environment_failure_r2(
        MAIN_R1_ATTEMPT, MAIN_R1_REPORT
    )
    recipe = r2.recipe_r2(
        prereg, {"bundle_sha256": "a" * 64}, seed_failure, environment_failure
    )
    assert recipe["worker_update_surfaces_closed"] is True
    assert recipe["authority"]["model_update_allowed"] is False
    source = inspect.getsource(r2.run_exact_r2)
    assert '"model_update_applied": False' in source
    assert '"nontrain_surface_opened": False' in source


@pytest.mark.skipif(
    os.environ.get("V23A_R2_RUN_RUNTIME_CERT") != "1",
    reason="opt-in intended runtime dependency certificate",
)
def test_v23a_r2_real_runtime_environment_certificate():
    certificate = r2.certify_runtime_environment_r2()
    assert certificate["cuda_visible_devices"] == "0,1,2,3"
    assert certificate["cuda_device_count"] == 4
    assert certificate["vllm_network_helpers_callable_after_compatibility_shim"] is True
