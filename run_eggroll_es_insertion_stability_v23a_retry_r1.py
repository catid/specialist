#!/usr/bin/env python3
"""Exclusive train-only retry for V23A with a narrow NumPy seed repair."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import traceback
from pathlib import Path

import build_eggroll_es_v23a_seed_failure_evidence_r1 as failure_evidence_r1
import eggroll_es_insertion_stability_preregistration_v23a as prereg_v23a
import eggroll_es_worker_v23a_retry_r1 as worker_r1
import run_eggroll_es_insertion_stability_v23a as runtime_v23a


ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME_R1 = (
    "insertion_location_stability_v23a_authoritative_raw_seed_retry_r1"
)
OUTPUT_DIRECTORY_R1 = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
ATTEMPT_NAME_R1 = f".{EXPERIMENT_NAME_R1}.launch_attempt.json"
REPORT_NAME_R1 = "insertion_location_stability_v23a_seed_retry_r1.json"
WORKER_EXTENSION_R1 = (
    "eggroll_es_worker_v23a_retry_r1."
    "InsertionLocationAuditWorkerExtensionV23ARetryR1"
)
ORIGINAL_ATTEMPT_PATH_R1 = ROOT / failure_evidence_r1.ATTEMPT_RELATIVE_PATH_R1
ORIGINAL_REPORT_PATH_R1 = ROOT / failure_evidence_r1.REPORT_RELATIVE_PATH_R1
FAILURE_EVIDENCE_PATH_R1 = failure_evidence_r1.OUTPUT_PATH_R1
FAILURE_EVIDENCE_FILE_SHA256_R1 = (
    "1fae9b7244eff3b532eb3fa93e769b0085728309460804c683a452028d399438"
)
FAILURE_EVIDENCE_CONTENT_SHA256_R1 = (
    "95d8542cd46c3743f4cb4db99159b485e10798766779b43024572925beeed922"
)
TEST_PATH_R1 = (ROOT / "test_run_eggroll_es_insertion_stability_v23a_retry_r1.py").resolve()
IMPLEMENTATION_PATHS_R1 = {
    "failure_evidence_builder_r1": Path(failure_evidence_r1.__file__).resolve(),
    "failure_evidence_tests_r1": ROOT / "test_build_eggroll_es_v23a_seed_failure_evidence_r1.py",
    "failure_evidence_r1": FAILURE_EVIDENCE_PATH_R1,
    "seed_repair_worker_r1": Path(worker_r1.__file__).resolve(),
    "seed_repair_worker_tests_r1": ROOT / "test_eggroll_es_worker_v23a_retry_r1.py",
    "retry_runtime_r1": Path(__file__).resolve(),
    "retry_runtime_tests_r1": TEST_PATH_R1,
}


canonical_sha256 = prereg_v23a.canonical_sha256
file_sha256 = prereg_v23a.file_sha256
_seal = runtime_v23a._seal_v23a


def _without_self(value):
    return {key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"}


def validate_original_failure_r1(
    attempt_path=None, report_path=None,
):
    attempt_path = ORIGINAL_ATTEMPT_PATH_R1 if attempt_path is None else attempt_path
    report_path = ORIGINAL_REPORT_PATH_R1 if report_path is None else report_path
    persisted = json.loads(FAILURE_EVIDENCE_PATH_R1.read_text(encoding="utf-8"))
    rebuilt = failure_evidence_r1.build_failure_evidence_r1(attempt_path, report_path)
    if (
        file_sha256(FAILURE_EVIDENCE_PATH_R1) != FAILURE_EVIDENCE_FILE_SHA256_R1
        or persisted.get("content_sha256_before_self_field")
        != FAILURE_EVIDENCE_CONTENT_SHA256_R1
        or persisted.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(persisted))
        or rebuilt != persisted
    ):
        raise RuntimeError("v23a-r1 original failure evidence changed")
    return persisted


def seed_projection_contract_r1(preregistration, failure_evidence):
    prereg_v23a.validate_preregistration_v23a(preregistration)
    seeds = prereg_v23a.perturbation_basis_v23a()["direction_seeds"]
    projections = [
        {"full_seed": seed, "numpy_legacy_seed": seed % (2**32)}
        for seed in seeds
    ]
    projected = [item["numpy_legacy_seed"] for item in projections]
    evidence = failure_evidence["seed_domain"]
    if (
        len(seeds) != 32 or len(set(seeds)) != 32
        or not all(2**32 - 1 < seed <= 2**63 - 1 for seed in seeds)
        or len(set(projected)) != 32 or any(seed == 0 for seed in projected)
        or canonical_sha256(seeds) != evidence["direction_seed_list_sha256"]
        or canonical_sha256(projections) != evidence["full_to_numpy_projection_sha256"]
        or evidence["numpy_projection_unique_count"] != 32
        or evidence["numpy_projection_contains_zero"] is not False
    ):
        raise RuntimeError("v23a-r1 seed projection contract changed")
    return {
        "schema": "eggroll-es-v23a-seed-projection-contract-r1",
        "direction_count": 32,
        "direction_seed_list_sha256": canonical_sha256(seeds),
        "full_to_numpy_projection_sha256": canonical_sha256(projections),
        "numpy_projection_rule": "full_seed modulo 2**32",
        "numpy_projection_unique_count": 32,
        "numpy_projection_contains_zero": False,
        "python_random_receives_full_seed": True,
        "torch_global_receives_full_seed": True,
        "torch_cuda_all_receives_full_seed": True,
        "explicit_torch_generator_receives_full_seed": True,
        "only_numpy_legacy_seed_is_projected": True,
        "perturbation_basis_unchanged": True,
    }


def implementation_identity_r1():
    inherited = runtime_v23a.implementation_identity_v23a()
    files = dict(inherited["files"])
    for key, path in IMPLEMENTATION_PATHS_R1.items():
        files[key] = {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
    if file_sha256(FAILURE_EVIDENCE_PATH_R1) != FAILURE_EVIDENCE_FILE_SHA256_R1:
        raise RuntimeError("v23a-r1 failure evidence file changed")
    return {
        "files": files,
        "inherited_v23a_bundle_sha256": inherited["bundle_sha256"],
        "retry_overlay_bundle_sha256": canonical_sha256({
            key: files[key] for key in IMPLEMENTATION_PATHS_R1
        }),
        "bundle_sha256": canonical_sha256(files),
    }


def recipe_r1(preregistration, implementation, failure_evidence):
    value = copy.deepcopy(runtime_v23a.recipe_v23a(preregistration, implementation))
    value.pop("content_sha256_before_self_field", None)
    projection = seed_projection_contract_r1(preregistration, failure_evidence)
    value.update({
        "schema": "eggroll-es-insertion-location-runtime-recipe-v23a-seed-retry-r1",
        "experiment_name": EXPERIMENT_NAME_R1,
        "worker_extension": WORKER_EXTENSION_R1,
        "output_directory": str(OUTPUT_DIRECTORY_R1),
        "retry_of": {
            "experiment_name": runtime_v23a.EXPERIMENT_NAME_V23A,
            "failed_attempt_relative_path": failure_evidence["failed_attempt"]["relative_path"],
            "failed_attempt_file_sha256": failure_evidence["failed_attempt"]["file_sha256"],
            "failed_attempt_content_sha256": failure_evidence["failed_attempt"]["content_sha256"],
            "failure_evidence_file_sha256": FAILURE_EVIDENCE_FILE_SHA256_R1,
            "failure_evidence_content_sha256": FAILURE_EVIDENCE_CONTENT_SHA256_R1,
            "original_attempt_immutable": True,
            "original_compact_report_absent": True,
        },
        "seed_domain_repair": projection,
        "same_preregistered_basis_gate_panels_and_arm_mapping": True,
        "fresh_exclusive_retry_paths": {
            "attempt_name": ATTEMPT_NAME_R1,
            "run_directory_name": EXPERIMENT_NAME_R1,
            "report_name": REPORT_NAME_R1,
            "all_disjoint_from_original": True,
        },
    })
    return _seal(value)


def _validate_worker_projection_reports_r1(reports, projection):
    if not isinstance(reports, list) or len(reports) != 4:
        raise RuntimeError("v23a-r1 worker seed certificate coverage changed")
    expected = {
        key: projection[key] for key in (
            "direction_count", "direction_seed_list_sha256",
            "full_to_numpy_projection_sha256", "numpy_projection_unique_count",
            "numpy_projection_contains_zero", "python_random_receives_full_seed",
            "torch_global_receives_full_seed", "torch_cuda_all_receives_full_seed",
            "explicit_torch_generator_receives_full_seed",
            "only_numpy_legacy_seed_is_projected",
        )
    }
    for report in reports:
        if (
            not isinstance(report, dict)
            or report.get("schema")
            != "eggroll-es-v23a-seed-projection-worker-certificate-r1"
            or any(report.get(key) != value for key, value in expected.items())
        ):
            raise RuntimeError("v23a-r1 worker seed certificate changed")
    if len({canonical_sha256(item) for item in reports}) != 1:
        raise RuntimeError("v23a-r1 workers disagree on the seed projection certificate")
    return {
        **expected,
        "worker_count": 4,
        "all_four_workers_identical": True,
        "worker_reports_sha256": canonical_sha256(reports),
        "new_retry_worker_extension_active": True,
        "certificate_completed_before_reference_scoring": True,
    }


class InsertionLocationRetryRuntimeMixinR1(
    runtime_v23a.InsertionLocationRuntimeMixinV23A,
):
    def configure_insertion_stability_v23a(self, preregistration, panel_bundle):
        projection = seed_projection_contract_r1(
            preregistration, self._v23a_r1_failure_evidence
        )
        seeds = prereg_v23a.perturbation_basis_v23a()["direction_seeds"]
        raw = self._resolve([
            engine.collective_rpc.remote(
                "seed_projection_certificate_v23a_r1",
                args=(
                    seeds,
                    projection["direction_seed_list_sha256"],
                    projection["full_to_numpy_projection_sha256"],
                ),
            )
            for engine in self.engines
        ])
        reports = [
            runtime_v23a._unwrap_one_v23a(item, "seed_projection_certificate_v23a_r1")
            for item in raw
        ]
        certificate = _validate_worker_projection_reports_r1(reports, projection)
        self._v23a_r1_seed_projection_certificate = certificate
        configured = super().configure_insertion_stability_v23a(
            preregistration, panel_bundle
        )
        configured["seed_domain_retry_r1"] = certificate
        return configured

    def estimate_insertion_stability_v23a(self):
        certificate = getattr(self, "_v23a_r1_seed_projection_certificate", None)
        if not isinstance(certificate, dict) or certificate.get("worker_count") != 4:
            raise RuntimeError("v23a-r1 pre-score seed certificate is missing")
        estimator, gate, audit = super().estimate_insertion_stability_v23a()
        audit.pop("content_sha256_before_self_field", None)
        audit["seed_domain_integrity_r1"] = {
            "direction_seed_list_sha256": certificate["direction_seed_list_sha256"],
            "full_to_numpy_projection_sha256": certificate["full_to_numpy_projection_sha256"],
            "all_four_workers_identical": certificate["all_four_workers_identical"],
            "full_seed_reaches_explicit_torch_generator_unchanged": True,
            "numpy_alone_receives_projected_seed": True,
            "new_retry_worker_extension_active": True,
            "certificate_completed_before_reference_scoring": True,
        }
        _seal(audit)
        runtime_v23a._assert_compact_persistence_v23a(audit)
        return estimator, gate, audit


def load_runtime_trainer_r1(preregistration, failure_evidence):
    prereg_v23a.validate_preregistration_v23a(preregistration)
    base_plan = preregistration["arms"]["base_middle_late"]["layer_plan"]
    bundle = runtime_v23a.anchor_v4.load_frozen_layer_plan_v4(
        base_plan["path"], expected_file_sha256=base_plan["file_sha256"],
        expected_plan_sha256=base_plan["plan_sha256"],
        expected_model_config_sha256=base_plan["model_config_sha256"],
    )
    parent = runtime_v23a.anchor_v4.load_trainer(bundle)

    class InsertionLocationRetryRuntimeTrainerR1(
        InsertionLocationRetryRuntimeMixinR1, parent,
    ):
        _v23a_r1_failure_evidence = copy.deepcopy(failure_evidence)

        def launch_engines(self, num_engines=4, n_gpu_per_vllm_engine=1,
                           model_name="unused", precision="bfloat16"):
            if num_engines != 4 or n_gpu_per_vllm_engine != 1:
                raise ValueError("v23a-r1 requires exactly four TP1 engines")
            import ray
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            from es_at_scale.trainer.es_trainer import ESNcclLLM
            pgs = [
                placement_group([{"GPU": 1, "CPU": 0}], strategy="PACK", lifetime="detached")
                for _ in range(4)
            ]
            ray.get([pg.ready() for pg in pgs])
            strategies = [
                PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0,
                )
                for pg in pgs
            ]
            engines = []
            for rank, arm in enumerate(prereg_v23a.ARM_ORDER_V23A):
                args = {
                    "model": preregistration["arms"][arm]["model_path"],
                    "tensor_parallel_size": 1,
                    "worker_extension_cls": WORKER_EXTENSION_R1,
                    "dtype": precision, "enable_prefix_caching": False,
                    "enforce_eager": True, "gpu_memory_utilization": 0.82,
                    "max_model_len": 2048,
                    "limit_mm_per_prompt": {"image": 0, "video": 0},
                    "mm_processor_cache_gb": 0, "skip_mm_profiling": True,
                    "moe_backend": "triton",
                }
                engines.append(ray.remote(
                    num_cpus=0, num_gpus=1, scheduling_strategy=strategies[rank]
                )(ESNcclLLM).remote(**args))
            return engines, pgs

    return InsertionLocationRetryRuntimeTrainerR1


def _make_trainer_r1(preregistration, failure_evidence):
    cls = load_runtime_trainer_r1(preregistration, failure_evidence)
    model = preregistration["arms"]["base_middle_late"]["model_path"]
    return cls(
        model_name=model, checkpoint=None, sigma=0.0003, alpha=0.0,
        population_size=32, reward_shaping="z-scores", num_iterations=1,
        max_tokens=1, batch_size=56, mini_batch_size=56,
        reward_function=runtime_v23a.base.specialist_reward,
        template_function=runtime_v23a.base.specialist_template,
        train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
        n_vllm_engines=4, n_gpu_per_vllm_engine=1, logging="none",
        global_seed=43, use_gpus="0,1,2,3", experiment_name=EXPERIMENT_NAME_R1,
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10, output_directory=str(OUTPUT_DIRECTORY_R1),
    )


def _parser_r1():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v23a-r1-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def validate_runtime_r1(args, preregistration, implementation, recipe):
    prereg_v23a.validate_preregistration_v23a(preregistration)
    if any(os.environ.get(key) for key in runtime_v23a.MOE_OVERRIDE_ENVIRONMENT_V23A):
        raise ValueError("v23a-r1 rejects external MoE backend overrides")
    for expected, actual, label in (
        (args.expected_implementation_bundle_sha256, implementation["bundle_sha256"],
         "implementation"),
        (args.expected_recipe_sha256, recipe["content_sha256_before_self_field"], "recipe"),
    ):
        if not args.v23a_r1_dry_run and expected is None:
            raise ValueError(f"v23a-r1 real launch requires expected {label} hash")
        if expected is not None and expected != actual:
            raise ValueError(f"v23a-r1 {label} hash changed")


def run_exact_r1(preregistration, implementation, recipe, failure_evidence):
    # Revalidate the original immutable failure immediately before claiming a
    # new path.  The original empty run directory is intentionally untouched.
    validate_original_failure_r1()
    attempt_path = OUTPUT_DIRECTORY_R1 / ATTEMPT_NAME_R1
    run_dir = OUTPUT_DIRECTORY_R1 / EXPERIMENT_NAME_R1
    if attempt_path.exists() or run_dir.exists():
        raise RuntimeError("v23a-r1 requires fresh exclusive retry attempt and run paths")
    provenance = runtime_v23a._source_provenance_v23a(implementation)
    model_directory_audit = runtime_v23a.validate_live_model_directories_v23a(
        preregistration
    )
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v23a-seed-retry-r1",
        "status": "launching", "phase": "before_trainer_creation",
        "recipe": recipe, "source_provenance": provenance,
        "retry_of": copy.deepcopy(recipe["retry_of"]),
        "model_update_applied": False,
        "model_directory_audit": model_directory_audit,
        "nontrain_surface_opened": False,
    }
    runtime_v23a._exclusive_write_json_v23a(attempt_path, attempt)
    if run_dir.exists():
        attempt.update({"status": "failed", "phase": "fresh_retry_run_reservation_race"})
        runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
        raise RuntimeError("v23a-r1 run directory appeared after exclusive attempt claim")
    trainer = None
    failure = None
    failure_traceback = None
    result = None
    configured = None
    try:
        runtime_v23a.base.set_seed(43)
        trainer = _make_trainer_r1(preregistration, failure_evidence)
        panels = runtime_v23a.mechanics_v23a.load_panel_bundle_v23a()
        configured = trainer.configure_insertion_stability_v23a(preregistration, panels)
        result = trainer.estimate_insertion_stability_v23a()
    except BaseException as error:
        failure = error
        failure_traceback = traceback.format_exc()
    finally:
        if trainer is not None:
            try:
                runtime_v23a.base.close_trainer(trainer)
            except BaseException as cleanup_error:
                if failure is None:
                    failure = cleanup_error
                    failure_traceback = traceback.format_exc()
                else:
                    failure_traceback += "\nCleanup failure:\n" + traceback.format_exc()
    if failure is not None:
        attempt.update({
            "status": "failed", "phase": "inside_v23a_r1_train_only_runtime",
            "failure": {
                "type": type(failure).__name__, "message": str(failure),
                "traceback": failure_traceback,
            },
            "model_update_applied": False,
        })
        runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
        raise failure
    estimator, gate, audit = result
    report = {
        "schema": "eggroll-es-insertion-location-stability-report-v23a-seed-retry-r1",
        "recipe": recipe, "configuration": configured, "estimator": estimator,
        "gate": gate, "runtime_audit": audit, "implementation": implementation,
        "retry_of": copy.deepcopy(recipe["retry_of"]),
        "model_update_applied": False, "nontrain_surface_opened": False,
        "direct_action_taken": False,
    }
    runtime_v23a._assert_compact_persistence_v23a(report)
    report_path = run_dir / REPORT_NAME_R1
    runtime_v23a._exclusive_write_json_v23a(report_path, report)
    attempt.update({
        "status": "complete", "phase": "after_cleanup_and_compact_retry_report",
        "report_binding": {
            "path": str(report_path), "file_sha256": file_sha256(report_path),
            "content_sha256": report["content_sha256_before_self_field"],
        },
        "model_update_applied": False,
    })
    runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    runtime_v23a._assert_train_only_argv_v23a(argv)
    args = _parser_r1().parse_args(argv)
    preregistration = runtime_v23a._load_preregistration_v23a()
    failure_evidence = validate_original_failure_r1()
    implementation = implementation_identity_r1()
    recipe = recipe_r1(preregistration, implementation, failure_evidence)
    validate_runtime_r1(args, preregistration, implementation, recipe)
    if args.v23a_r1_dry_run:
        payload = _seal({
            "schema": "eggroll-es-insertion-location-seed-retry-launch-dry-run-r1",
            "recipe": recipe, "implementation": implementation,
            "original_failed_attempt_revalidated": True,
            "new_retry_paths_exclusive_and_disjoint": True,
            "real_launch_requires_committed_bundle_and_recipe_hashes": True,
            "gpu_launched": False,
        })
        runtime_v23a._assert_compact_persistence_v23a(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_r1(preregistration, implementation, recipe, failure_evidence)


if __name__ == "__main__":
    main()
