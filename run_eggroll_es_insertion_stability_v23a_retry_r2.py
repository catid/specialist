#!/usr/bin/env python3
"""Exclusive V23A retry R2 with a pre-claim runtime certificate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import build_eggroll_es_v23a_runtime_failure_evidence_r2 as environment_failure_r2
import run_eggroll_es_insertion_stability_v23a_retry_r1 as runtime_r1


ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME_R2 = (
    "insertion_location_stability_v23a_authoritative_raw_seed_retry_r2"
)
OUTPUT_DIRECTORY_R2 = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
ATTEMPT_NAME_R2 = f".{EXPERIMENT_NAME_R2}.launch_attempt.json"
REPORT_NAME_R2 = "insertion_location_stability_v23a_seed_retry_r2.json"
R1_ATTEMPT_PATH_R2 = ROOT / environment_failure_r2.ATTEMPT_RELATIVE_PATH_R2
R1_REPORT_PATH_R2 = ROOT / environment_failure_r2.REPORT_RELATIVE_PATH_R2
ENVIRONMENT_FAILURE_PATH_R2 = environment_failure_r2.OUTPUT_PATH_R2
ENVIRONMENT_FAILURE_FILE_SHA256_R2 = (
    "4024c90e01ce05b388e547ee5b2c642ebfd28f35b21a85277adff707cfe066af"
)
ENVIRONMENT_FAILURE_CONTENT_SHA256_R2 = (
    "51ab2573fe04de32db0daaf9fc788b52dbbe40dc2b41e4eb096fd21bc6249ab7"
)
EXPECTED_VENV_PREFIX_R2 = str(ROOT / "es-at-scale/.venv")
EXPECTED_EXECUTABLE_R2 = str(ROOT / "es-at-scale/.venv/bin/python")
EXPECTED_COMPAT_R2 = str(ROOT / "eggroll_es_compat")
EXPECTED_UPSTREAM_R2 = str(ROOT / "es-at-scale")
EXPECTED_VERSIONS_R2 = {
    "ray": "2.56.0",
    "torch": "2.11.0+cu130",
    "vllm": "0.25.0",
}
EXPECTED_MODULE_FILES_R2 = {
    "es_trainer": {
        "path": str(ROOT / "es-at-scale/es_at_scale/trainer/es_trainer.py"),
        "file_sha256": "4f7044600af13b0a73f807aabb7c188610438d31501cc8a053d0f6efae14b42e",
    },
    "ray": {
        "path": str(ROOT / "es-at-scale/.venv/lib/python3.12/site-packages/ray/__init__.py"),
        "file_sha256": "a3845ca44927ca669778ca7489a522f4f5138e8303c7efe6818a193e1b15d376",
    },
    "torch": {
        "path": str(ROOT / "es-at-scale/.venv/lib/python3.12/site-packages/torch/__init__.py"),
        "file_sha256": "0387d8b811b289287479c8bfdf4e1dac3a71b246f938d82da1331cf2dc8bf001",
    },
    "vllm": {
        "path": str(ROOT / "es-at-scale/.venv/lib/python3.12/site-packages/vllm/__init__.py"),
        "file_sha256": "fd3708e5a13abe98c566afd79a6b1987cef70c0548bb19c1218ff6a7f9d43346",
    },
}
TEST_PATH_R2 = (ROOT / "test_run_eggroll_es_insertion_stability_v23a_retry_r2.py").resolve()
IMPLEMENTATION_PATHS_R2 = {
    "runtime_failure_evidence_builder_r2": Path(environment_failure_r2.__file__).resolve(),
    "runtime_failure_evidence_tests_r2": ROOT / "test_build_eggroll_es_v23a_runtime_failure_evidence_r2.py",
    "runtime_failure_evidence_r2": ENVIRONMENT_FAILURE_PATH_R2,
    "retry_runtime_r2": Path(__file__).resolve(),
    "retry_runtime_tests_r2": TEST_PATH_R2,
}


canonical_sha256 = runtime_r1.canonical_sha256
file_sha256 = runtime_r1.file_sha256
_seal = runtime_r1._seal


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def validate_r1_environment_failure_r2(attempt_path=None, report_path=None):
    attempt_path = R1_ATTEMPT_PATH_R2 if attempt_path is None else Path(attempt_path)
    report_path = R1_REPORT_PATH_R2 if report_path is None else Path(report_path)
    persisted = json.loads(ENVIRONMENT_FAILURE_PATH_R2.read_text(encoding="utf-8"))
    rebuilt = environment_failure_r2.build_runtime_failure_evidence_r2(
        attempt_path, report_path
    )
    if (
        file_sha256(ENVIRONMENT_FAILURE_PATH_R2)
        != ENVIRONMENT_FAILURE_FILE_SHA256_R2
        or persisted.get("content_sha256_before_self_field")
        != ENVIRONMENT_FAILURE_CONTENT_SHA256_R2
        or persisted.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(persisted))
        or rebuilt != persisted
    ):
        raise RuntimeError("v23a-r2 R1 environment failure evidence changed")
    return persisted


def expected_environment_contract_r2():
    return {
        "schema": "eggroll-es-v23a-runtime-environment-contract-r2",
        "sys_prefix": EXPECTED_VENV_PREFIX_R2,
        "sys_executable": EXPECTED_EXECUTABLE_R2,
        "compat_path": EXPECTED_COMPAT_R2,
        "upstream_path": EXPECTED_UPSTREAM_R2,
        "compat_precedes_upstream_in_worker_pythonpath": True,
        "versions": copy.deepcopy(EXPECTED_VERSIONS_R2),
        "module_files": copy.deepcopy(EXPECTED_MODULE_FILES_R2),
        "vllm_network_helpers_callable_after_compatibility_shim": True,
        "es_nccl_llm_importable_after_compatibility_shim": True,
        "trainer_class_importable_after_compatibility_shim": True,
        "fresh_worker_subprocess_import_succeeds": True,
        "cuda_visible_devices": "0,1,2,3",
        "cuda_device_count": 4,
        "completed_before_attempt_claim": True,
        "dataset_or_evaluation_surface_opened": False,
    }


def validate_environment_observation_r2(observation):
    expected = expected_environment_contract_r2()
    if not isinstance(observation, dict):
        raise RuntimeError("v23a-r2 runtime environment observation is missing")
    for key in (
        "sys_prefix", "sys_executable", "compat_path", "upstream_path",
        "compat_precedes_upstream_in_worker_pythonpath", "versions",
        "vllm_network_helpers_callable_after_compatibility_shim",
        "es_nccl_llm_importable_after_compatibility_shim",
        "trainer_class_importable_after_compatibility_shim",
        "fresh_worker_subprocess_import_succeeds",
        "cuda_visible_devices", "cuda_device_count",
    ):
        if observation.get(key) != expected[key]:
            raise RuntimeError(f"v23a-r2 runtime environment changed: {key}")
    modules = observation.get("module_files")
    if modules != expected["module_files"]:
        raise RuntimeError("v23a-r2 runtime module identity changed")
    certificate = copy.deepcopy(expected)
    certificate["observation_sha256"] = canonical_sha256(observation)
    return _seal(certificate)


def collect_environment_observation_r2():
    # This follows the real launcher ordering.  load_trainer installs the
    # relocated vLLM helper aliases and prepends sitecustomize for Ray workers.
    parent = runtime_r1.runtime_v23a.base.load_trainer()
    import ray
    import torch
    import vllm
    import vllm.utils
    import es_at_scale.trainer.es_trainer as es_trainer

    pythonpath = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    module_values = {
        "es_trainer": es_trainer,
        "ray": ray,
        "torch": torch,
        "vllm": vllm,
    }
    module_files = {}
    for key, module in module_values.items():
        path = Path(module.__file__).resolve()
        module_files[key] = {"path": str(path), "file_sha256": file_sha256(path)}
    child_environment = os.environ.copy()
    child_environment["CUDA_VISIBLE_DEVICES"] = ""
    child_code = (
        "import json,sys,vllm.utils; "
        "from es_at_scale.trainer.es_trainer import ESNcclLLM; "
        "print(json.dumps({'sys_prefix':sys.prefix,"
        "'sitecustomize':'sitecustomize' in sys.modules,"
        "'helpers':callable(vllm.utils.get_ip) and callable(vllm.utils.get_open_port),"
        "'trainer':ESNcclLLM.__name__},sort_keys=True))"
    )
    child = json.loads(subprocess.check_output(
        [sys.executable, "-c", child_code],
        env=child_environment,
        text=True,
        timeout=60,
    ))
    return {
        "sys_prefix": sys.prefix,
        "sys_executable": sys.executable,
        "compat_path": pythonpath[0] if len(pythonpath) > 0 else None,
        "upstream_path": pythonpath[1] if len(pythonpath) > 1 else None,
        "compat_precedes_upstream_in_worker_pythonpath": pythonpath[:2]
        == [EXPECTED_COMPAT_R2, EXPECTED_UPSTREAM_R2],
        "versions": {
            "ray": ray.__version__,
            "torch": torch.__version__,
            "vllm": vllm.__version__,
        },
        "vllm_network_helpers_callable_after_compatibility_shim": (
            callable(getattr(vllm.utils, "get_ip", None))
            and callable(getattr(vllm.utils, "get_open_port", None))
        ),
        "es_nccl_llm_importable_after_compatibility_shim": (
            es_trainer.ESNcclLLM.__name__ == "ESNcclLLM"
        ),
        "trainer_class_importable_after_compatibility_shim": isinstance(parent, type),
        "fresh_worker_subprocess_import_succeeds": child == {
            "helpers": True,
            "sitecustomize": True,
            "sys_prefix": EXPECTED_VENV_PREFIX_R2,
            "trainer": "ESNcclLLM",
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": torch.cuda.device_count(),
        "module_files": module_files,
    }


def certify_runtime_environment_r2():
    return validate_environment_observation_r2(collect_environment_observation_r2())


def implementation_identity_r2():
    inherited = runtime_r1.implementation_identity_r1()
    files = dict(inherited["files"])
    for key, path in IMPLEMENTATION_PATHS_R2.items():
        files[key] = {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
    if file_sha256(ENVIRONMENT_FAILURE_PATH_R2) != ENVIRONMENT_FAILURE_FILE_SHA256_R2:
        raise RuntimeError("v23a-r2 environment failure evidence file changed")
    return {
        "files": files,
        "inherited_v23a_r1_bundle_sha256": inherited["bundle_sha256"],
        "retry_overlay_bundle_sha256": canonical_sha256({
            key: files[key] for key in IMPLEMENTATION_PATHS_R2
        }),
        "bundle_sha256": canonical_sha256(files),
    }


def recipe_r2(preregistration, implementation, seed_failure, environment_failure):
    inherited = runtime_r1.recipe_r1(
        preregistration, implementation, seed_failure
    )
    value = copy.deepcopy(inherited)
    value.pop("content_sha256_before_self_field", None)
    value.update({
        "schema": "eggroll-es-insertion-location-runtime-recipe-v23a-seed-retry-r2",
        "experiment_name": EXPERIMENT_NAME_R2,
        "output_directory": str(OUTPUT_DIRECTORY_R2),
        "retry_of": {
            "experiment_name": runtime_r1.EXPERIMENT_NAME_R1,
            "failed_attempt_relative_path": environment_failure["failed_attempt"]["relative_path"],
            "failed_attempt_file_sha256": environment_failure["failed_attempt"]["file_sha256"],
            "failed_attempt_content_sha256": environment_failure["failed_attempt"]["content_sha256"],
            "failure_evidence_file_sha256": ENVIRONMENT_FAILURE_FILE_SHA256_R2,
            "failure_evidence_content_sha256": ENVIRONMENT_FAILURE_CONTENT_SHA256_R2,
            "original_seed_failure": copy.deepcopy(inherited["retry_of"]),
            "r1_attempt_immutable": True,
            "r1_compact_report_absent": True,
        },
        "runtime_environment_contract_r2": expected_environment_contract_r2(),
        "same_preregistered_basis_gate_panels_arm_mapping_and_seed_repair": True,
        "fresh_exclusive_retry_paths": {
            "attempt_name": ATTEMPT_NAME_R2,
            "run_directory_name": EXPERIMENT_NAME_R2,
            "report_name": REPORT_NAME_R2,
            "all_disjoint_from_original_and_r1": True,
        },
    })
    return _seal(value)


def _make_trainer_r2(preregistration, seed_failure):
    cls = runtime_r1.load_runtime_trainer_r1(preregistration, seed_failure)
    model = preregistration["arms"]["base_middle_late"]["model_path"]
    base = runtime_r1.runtime_v23a.base
    return cls(
        model_name=model, checkpoint=None, sigma=0.0003, alpha=0.0,
        population_size=32, reward_shaping="z-scores", num_iterations=1,
        max_tokens=1, batch_size=56, mini_batch_size=56,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template,
        train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
        n_vllm_engines=4, n_gpu_per_vllm_engine=1, logging="none",
        global_seed=43, use_gpus="0,1,2,3", experiment_name=EXPERIMENT_NAME_R2,
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10, output_directory=str(OUTPUT_DIRECTORY_R2),
    )


def _parser_r2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v23a-r2-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def validate_runtime_r2(args, preregistration, implementation, recipe):
    runtime_r1.prereg_v23a.validate_preregistration_v23a(preregistration)
    if any(
        os.environ.get(key)
        for key in runtime_r1.runtime_v23a.MOE_OVERRIDE_ENVIRONMENT_V23A
    ):
        raise ValueError("v23a-r2 rejects external MoE backend overrides")
    for expected, actual, label in (
        (args.expected_implementation_bundle_sha256, implementation["bundle_sha256"], "implementation"),
        (args.expected_recipe_sha256, recipe["content_sha256_before_self_field"], "recipe"),
    ):
        if not args.v23a_r2_dry_run and expected is None:
            raise ValueError(f"v23a-r2 real launch requires expected {label} hash")
        if expected is not None and expected != actual:
            raise ValueError(f"v23a-r2 {label} hash changed")


def run_exact_r2(
    preregistration, implementation, recipe, seed_failure, environment_failure,
):
    runtime_r1.validate_original_failure_r1()
    validate_r1_environment_failure_r2()
    environment_certificate = certify_runtime_environment_r2()
    attempt_path = OUTPUT_DIRECTORY_R2 / ATTEMPT_NAME_R2
    run_dir = OUTPUT_DIRECTORY_R2 / EXPERIMENT_NAME_R2
    if attempt_path.exists() or run_dir.exists():
        raise RuntimeError("v23a-r2 requires fresh exclusive retry attempt and run paths")
    provenance = runtime_r1.runtime_v23a._source_provenance_v23a(implementation)
    model_directory_audit = runtime_r1.runtime_v23a.validate_live_model_directories_v23a(
        preregistration
    )
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v23a-seed-retry-r2",
        "status": "launching",
        "phase": "before_trainer_creation",
        "recipe": recipe,
        "source_provenance": provenance,
        "retry_of": copy.deepcopy(recipe["retry_of"]),
        "runtime_environment_certificate": environment_certificate,
        "model_update_applied": False,
        "model_directory_audit": model_directory_audit,
        "nontrain_surface_opened": False,
    }
    runtime_r1.runtime_v23a._exclusive_write_json_v23a(attempt_path, attempt)
    if run_dir.exists():
        attempt.update({"status": "failed", "phase": "fresh_retry_run_reservation_race"})
        runtime_r1.runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
        raise RuntimeError("v23a-r2 run directory appeared after exclusive attempt claim")
    trainer = None
    failure = None
    failure_traceback = None
    result = None
    configured = None
    try:
        runtime_r1.runtime_v23a.base.set_seed(43)
        trainer = _make_trainer_r2(preregistration, seed_failure)
        panels = runtime_r1.runtime_v23a.mechanics_v23a.load_panel_bundle_v23a()
        configured = trainer.configure_insertion_stability_v23a(preregistration, panels)
        result = trainer.estimate_insertion_stability_v23a()
    except BaseException as error:
        failure = error
        failure_traceback = traceback.format_exc()
    finally:
        if trainer is not None:
            try:
                runtime_r1.runtime_v23a.base.close_trainer(trainer)
            except BaseException as cleanup_error:
                if failure is None:
                    failure = cleanup_error
                    failure_traceback = traceback.format_exc()
                else:
                    failure_traceback += "\nCleanup failure:\n" + traceback.format_exc()
    if failure is not None:
        attempt.update({
            "status": "failed",
            "phase": "inside_v23a_r2_train_only_runtime",
            "failure": {
                "type": type(failure).__name__,
                "message": str(failure),
                "traceback": failure_traceback,
            },
            "model_update_applied": False,
        })
        runtime_r1.runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
        raise failure
    estimator, gate, audit = result
    report = {
        "schema": "eggroll-es-insertion-location-stability-report-v23a-seed-retry-r2",
        "recipe": recipe,
        "configuration": configured,
        "estimator": estimator,
        "gate": gate,
        "runtime_audit": audit,
        "runtime_environment_certificate": environment_certificate,
        "implementation": implementation,
        "retry_of": copy.deepcopy(recipe["retry_of"]),
        "model_update_applied": False,
        "nontrain_surface_opened": False,
        "direct_action_taken": False,
    }
    runtime_r1.runtime_v23a._assert_compact_persistence_v23a(report)
    report_path = run_dir / REPORT_NAME_R2
    runtime_r1.runtime_v23a._exclusive_write_json_v23a(report_path, report)
    attempt.update({
        "status": "complete",
        "phase": "after_cleanup_and_compact_retry_report",
        "report_binding": {
            "path": str(report_path),
            "file_sha256": file_sha256(report_path),
            "content_sha256": report["content_sha256_before_self_field"],
        },
        "model_update_applied": False,
    })
    runtime_r1.runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    runtime_r1.runtime_v23a._assert_train_only_argv_v23a(argv)
    args = _parser_r2().parse_args(argv)
    preregistration = runtime_r1.runtime_v23a._load_preregistration_v23a()
    seed_failure = runtime_r1.validate_original_failure_r1()
    environment_failure = validate_r1_environment_failure_r2()
    implementation = implementation_identity_r2()
    recipe = recipe_r2(
        preregistration, implementation, seed_failure, environment_failure
    )
    validate_runtime_r2(args, preregistration, implementation, recipe)
    if args.v23a_r2_dry_run:
        payload = _seal({
            "schema": "eggroll-es-insertion-location-seed-retry-launch-dry-run-r2",
            "recipe": recipe,
            "implementation": implementation,
            "original_seed_failure_revalidated": True,
            "r1_environment_failure_revalidated": True,
            "new_retry_paths_exclusive_and_disjoint": True,
            "real_launch_requires_committed_bundle_recipe_and_runtime_environment": True,
            "gpu_launched": False,
        })
        runtime_r1.runtime_v23a._assert_compact_persistence_v23a(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_r2(
        preregistration, implementation, recipe, seed_failure, environment_failure
    )


if __name__ == "__main__":
    main()
