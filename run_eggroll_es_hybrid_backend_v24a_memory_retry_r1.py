#!/usr/bin/env python3
"""Fresh V24A retry using exact vLLM model-load bytes for the memory gate."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

import build_eggroll_es_v24a_memory_failure_evidence_r1 as failure_evidence
import eggroll_es_hybrid_backend_preregistration_v24a as prereg
import eggroll_es_worker_v24a_memory_retry_r1 as memory_worker_r1
import run_eggroll_es_hybrid_backend_v24a as base
import train_eggroll_es_hybrid_backend_v24a_memory_retry_r1 as mechanics_r1


ROOT = Path(__file__).resolve().parent
CANONICAL_ROOT = Path("/home/catid/specialist")
EXPERIMENT_NAME_R1 = (
    "s6_v24a_hybrid_backend_train_only_compatibility_memory_retry_r1"
)
OUTPUT_DIRECTORY_R1 = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
ATTEMPT_NAME_R1 = f".{EXPERIMENT_NAME_R1}.launch_attempt.json"
REPORT_NAME_R1 = "hybrid_backend_compatibility_v24a_memory_retry_r1.json"
WORKER_EXTENSION_R1 = (
    "eggroll_es_worker_v24a_memory_retry_r1."
    "HybridBackendMemoryWorkerExtensionV24ARetryR1"
)
ORIGINAL_ATTEMPT_PATH_R1 = CANONICAL_ROOT / failure_evidence.ATTEMPT_RELATIVE_PATH_R1
ORIGINAL_REPORT_PATH_R1 = CANONICAL_ROOT / failure_evidence.REPORT_RELATIVE_PATH_R1
VLLM_MODEL_RUNNER_PATH_R1 = (
    CANONICAL_ROOT / failure_evidence.VLLM_MODEL_RUNNER_RELATIVE_PATH_R1
)
VLLM_GPU_WORKER_PATH_R1 = (
    CANONICAL_ROOT / failure_evidence.VLLM_GPU_WORKER_RELATIVE_PATH_R1
)
ORIGINAL_RUNTIME_PATH_R1 = (
    CANONICAL_ROOT / failure_evidence.ORIGINAL_RUNTIME_RELATIVE_PATH_R1
)
FAILURE_EVIDENCE_PATH_R1 = failure_evidence.OUTPUT_PATH_R1
FAILURE_EVIDENCE_FILE_SHA256_R1 = (
    "989c54fd2baba5fe58ed411cc0a5b8b14e55b9be5b55363283fa12f26830a48c"
)
FAILURE_EVIDENCE_CONTENT_SHA256_R1 = (
    "0773a2b4b24d53723da51b517970f50f2cb15bf5ced46c872e8d584b761fb594"
)
TEST_PATH_R1 = (
    ROOT / "test_run_eggroll_es_hybrid_backend_v24a_memory_retry_r1.py"
).resolve()
IMPLEMENTATION_PATHS_R1 = {
    "memory_failure_evidence_builder_v24a_r1": Path(failure_evidence.__file__).resolve(),
    "memory_failure_evidence_tests_v24a_r1": (
        ROOT / "test_build_eggroll_es_v24a_memory_failure_evidence_r1.py"
    ),
    "memory_failure_evidence_v24a_r1": FAILURE_EVIDENCE_PATH_R1,
    "model_load_memory_worker_v24a_r1": Path(memory_worker_r1.__file__).resolve(),
    "model_load_memory_worker_tests_v24a_r1": (
        ROOT / "test_eggroll_es_worker_v24a_memory_retry_r1.py"
    ),
    "memory_estimator_v24a_r1": Path(mechanics_r1.__file__).resolve(),
    "memory_runtime_v24a_r1": Path(__file__).resolve(),
    "memory_runtime_tests_v24a_r1": TEST_PATH_R1,
}


canonical_sha256 = base.canonical_sha256
file_sha256 = base.file_sha256
_seal = base._seal


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def validate_original_memory_failure_r1():
    persisted = json.loads(FAILURE_EVIDENCE_PATH_R1.read_text(encoding="utf-8"))
    rebuilt = failure_evidence.build_memory_failure_evidence_r1(
        ORIGINAL_ATTEMPT_PATH_R1,
        ORIGINAL_REPORT_PATH_R1,
        VLLM_MODEL_RUNNER_PATH_R1,
        VLLM_GPU_WORKER_PATH_R1,
        ORIGINAL_RUNTIME_PATH_R1,
    )
    if (
        file_sha256(FAILURE_EVIDENCE_PATH_R1)
        != FAILURE_EVIDENCE_FILE_SHA256_R1
        or persisted.get("content_sha256_before_self_field")
        != FAILURE_EVIDENCE_CONTENT_SHA256_R1
        or persisted.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(persisted))
        or rebuilt != persisted
    ):
        raise RuntimeError("v24a-r1 memory failure evidence changed")
    return persisted


def implementation_identity_r1():
    inherited = base.implementation_identity_v24a()
    files = dict(inherited["files"])
    for key, path in IMPLEMENTATION_PATHS_R1.items():
        files[key] = {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
    if file_sha256(FAILURE_EVIDENCE_PATH_R1) != FAILURE_EVIDENCE_FILE_SHA256_R1:
        raise RuntimeError("v24a-r1 memory failure evidence file changed")
    return {
        "files": files,
        "inherited_v24a_bundle_sha256": inherited["bundle_sha256"],
        "memory_retry_overlay_bundle_sha256": canonical_sha256({
            key: files[key] for key in IMPLEMENTATION_PATHS_R1
        }),
        "bundle_sha256": canonical_sha256(files),
    }


def recipe_r1(preregistration, implementation, failure):
    inherited = base.recipe_v24a(preregistration, implementation)
    value = copy.deepcopy(inherited)
    value.pop("content_sha256_before_self_field", None)
    value.update({
        "schema": "eggroll-es-hybrid-backend-runtime-recipe-v24a-memory-retry-r1",
        "experiment_name": EXPERIMENT_NAME_R1,
        "output_directory": str(OUTPUT_DIRECTORY_R1),
        "worker_extension": WORKER_EXTENSION_R1,
        "retry_of": {
            "experiment_name": base.EXPERIMENT_NAME,
            "failed_attempt_relative_path": failure["failed_attempt"]["relative_path"],
            "failed_attempt_file_sha256": failure["failed_attempt"]["file_sha256"],
            "failed_attempt_content_sha256": failure["failed_attempt"]["content_sha256"],
            "failure_evidence_file_sha256": FAILURE_EVIDENCE_FILE_SHA256_R1,
            "failure_evidence_content_sha256": FAILURE_EVIDENCE_CONTENT_SHA256_R1,
            "original_attempt_immutable": True,
            "original_compact_report_absent": True,
        },
        "memory_endpoint_repair_r1": {
            "sole_semantic_repair": (
                "memory gate input is exact int(self.model_runner.model_memory_usage)"
            ),
            "source_assignment": "self.model_memory_usage = m.consumed_memory",
            "duplicate_bf16_values_exact_required": True,
            "duplicate_hybrid_values_exact_required": True,
            "reduction_threshold_per_pair": 0.40,
            "preflight_before_reference_a_b_c_or_first_perturbation": True,
            "abort_early_on_missing_duplicate_or_subthreshold_value": True,
            "nvml_resident_bytes_diagnostic_only": True,
            "nvml_excluded_from_all_gates": True,
        },
        "same_preregistered_arms_basis_panels_schedule_quality_speed_bootstrap_guards_and_restore": True,
        "fresh_exclusive_retry_paths": {
            "attempt_name": ATTEMPT_NAME_R1,
            "run_directory_name": EXPERIMENT_NAME_R1,
            "report_name": REPORT_NAME_R1,
            "all_disjoint_from_original": True,
        },
    })
    value["seed_domain_repair"]["worker_extension"] = WORKER_EXTENSION_R1
    value["analysis"]["memory_reduction_definition"] = (
        "1 - hybrid_model_load_consumed_bytes / bf16_model_load_consumed_bytes"
    )
    value["gate"]["pair_pass"] = (
        "all 16 quality and own speed familywise LCBs meet thresholds, exact "
        "vLLM model-load memory reduction >= 0.40, and every runtime integrity audit is true"
    )
    value["nvml_memory_contract"].update({
        "diagnostic_only": True,
        "excluded_from_quality_speed_memory_and_global_pass": True,
    })
    return _seal(value)


class HybridBackendMemoryRetryMixinV24AR1:
    def configure_hybrid_backend_v24a(self, preregistration, panel_bundle):
        configured = super().configure_hybrid_backend_v24a(
            preregistration, panel_bundle
        )
        raw = self._resolve([
            engine.collective_rpc.remote(
                "model_load_memory_v24a_r1", args=(arm,)
            )
            for engine, arm in zip(self.engines, prereg.ARM_ORDER_V24A)
        ])
        reports = [
            base.runtime_v23a._unwrap_one_v23a(
                item, "model_load_memory_v24a_r1"
            )
            for item in raw
        ]
        model_load_consumed_bytes = {}
        for rank, arm in enumerate(prereg.ARM_ORDER_V24A):
            report = reports[rank]
            value = report.get("model_load_consumed_bytes")
            if (
                report.get("schema")
                != "eggroll-es-v24a-model-load-memory-worker-r1"
                or report.get("rank") != rank
                or report.get("world_size") != 4
                or report.get("arm") != arm
                or report.get("cuda_visible_devices") != str(rank)
                or type(value) is not int
                or value <= 0
                or report.get("source_object")
                != "self.model_runner.model_memory_usage"
                or report.get("source_assignment")
                != "self.model_memory_usage = m.consumed_memory"
                or report.get("measured_after_model_load_before_scoring") is not True
            ):
                raise RuntimeError("v24a-r1 model-load worker report changed")
            model_load_consumed_bytes[arm] = value
        preflight = mechanics_r1.validate_model_load_memory_preflight_r1(
            model_load_consumed_bytes
        )
        self._v24a_r1_model_load_consumed_bytes = model_load_consumed_bytes
        self._v24a_r1_memory_preflight = preflight
        configured = copy.deepcopy(configured)
        configured.pop("content_sha256_before_self_field", None)
        inherited_nvml = configured.pop("nvml_resident_bytes", None)
        if inherited_nvml != self._v24a_resident_bytes:
            raise RuntimeError("v24a-r1 inherited NVML diagnostic changed")
        configured.update({
            "schema": "eggroll-es-hybrid-backend-runtime-configuration-v24a-memory-retry-r1",
            "model_load_memory_worker_reports_sha256": canonical_sha256(reports),
            "model_load_memory_preflight": preflight,
            "nvml_resident_bytes_diagnostic": copy.deepcopy(inherited_nvml),
            "nvml_excluded_from_all_gates": True,
            "completed_before_reference_a_b_c_or_first_perturbation": True,
        })
        return _seal(configured)

    def estimate_hybrid_backend_v24a(self):
        if not hasattr(self, "_v24a_r1_memory_preflight"):
            raise RuntimeError("v24a-r1 memory preflight was not completed")
        schedule = prereg.signed_wave_schedule_v24a()
        panels, requests, request_identity = self._prepared_requests_v24a()
        reference_batches, _ = self._generate_v24a(requests, timed=False)
        phase_a = self._score_v24a(panels, reference_batches)
        repeat_batches, _ = self._generate_v24a(requests, timed=False)
        phase_b = self._score_v24a(panels, repeat_batches)
        pre_equal = base._phases_equal(phase_a, phase_b)
        if not all(pre_equal.values()):
            raise RuntimeError("v24a-r1 repeated full-context baseline is not exact")
        reference_scores = phase_a[0]
        unit_scores = {
            arm: np.full((5, 2, 32, 56), np.nan, dtype=np.float64)
            for arm in prereg.ARM_ORDER_V24A
        }
        timings = np.empty((64, 4), dtype=np.float64)
        restore_hashes = []
        dense_hashes = []
        for wave in schedule:
            perturbed = self._resolve([
                engine.collective_rpc.remote(
                    "perturb_self_weights",
                    args=(wave["direction_seed"], 0.0003, wave["negate"]),
                )
                for engine in self.engines
            ])
            base.runtime_v23a._validate_perturbation_results_v23a(perturbed)
            batches, elapsed_ns = self._generate_v24a(requests, timed=True)
            scores, commitments, _probes = self._score_v24a(panels, batches)
            timings[wave["signed_wave_index"]] = np.asarray(elapsed_ns) / 1e9
            sign_index = 0 if wave["sign"] == "plus" else 1
            for arm in prereg.ARM_ORDER_V24A:
                unit_scores[arm][:, sign_index, wave["direction_index"]] = scores[arm]
                dense_hashes.extend(commitments[arm])
            restore_hashes.append(self._restore_v24a())
        if (
            len(restore_hashes) != 64
            or len(dense_hashes) != 1_280
            or any(not np.isfinite(value).all() for value in unit_scores.values())
        ):
            raise RuntimeError("v24a-r1 signed-wave capture incomplete")
        boundary_sha = self._boundary_v24a()
        post_batches, _ = self._generate_v24a(requests, timed=False)
        phase_c = self._score_v24a(panels, post_batches)
        post_equal = base._phases_equal(phase_a, phase_c)
        if not all(post_equal.values()):
            raise RuntimeError("v24a-r1 post-population full-context guard is not exact")
        integrity_template = {
            "all_64_signed_waves_complete": True,
            "all_five_panels_every_arm_and_signed_wave": True,
            "same_direction_sign_rows_and_sampling_all_four_arms": True,
            "exact_selected_restore_every_arm_every_signed_wave": True,
            "unselected_partition_identity_unchanged": True,
            "full_context_a_b_equal_before_first_perturbation": True,
            "full_context_a_c_equal_after_population_audit": True,
            "distinct_gpu_assignment_verified": True,
            "timing_boundary_verified": True,
            "nvml_memory_boundary_verified": True,
            "model_load_memory_boundary_verified": True,
            "update_and_nontrain_surfaces_closed": True,
        }
        estimator = mechanics_r1.build_compact_summary_v24a_memory_retry_r1(
            unit_scores,
            reference_scores,
            timings,
            self._v24a_r1_model_load_consumed_bytes,
            self._v24a_resident_bytes,
            self._v24a_panel_bundle,
            {arm: dict(integrity_template) for arm in prereg.ARM_ORDER_V24A},
        )
        unit_scores = reference_scores = timings = None
        guard = {
            "schema": "eggroll-es-v24a-matched-full-context-guard-memory-retry-r1",
            "request_identity_sha256": request_identity,
            "same_request_list_object_value_shape_and_order_a_b_c": True,
            "phase_a": base._phase_identity(*phase_a),
            "phase_b": base._phase_identity(*phase_b),
            "phase_c": base._phase_identity(*phase_c),
            "a_b_exact": pre_equal,
            "a_c_exact": post_equal,
            "memory_preflight_completed_before_phase_a": True,
            "completed_before_first_perturbation_and_after_full_audit": True,
            "guard_requests_excluded_from_all_endpoints": True,
            "raw_outputs_or_scores_persisted": False,
        }
        audit = _seal({
            "schema": "eggroll-es-hybrid-backend-runtime-audit-v24a-memory-retry-r1",
            "signed_wave_schedule_sha256": canonical_sha256(schedule),
            "dense_result_commitments_sha256": canonical_sha256(dense_hashes),
            "restore_checks_sha256": canonical_sha256(restore_hashes),
            "population_boundary_audit_sha256": boundary_sha,
            "model_load_memory_preflight_content_sha256": (
                self._v24a_r1_memory_preflight["content_sha256_before_self_field"]
            ),
            "nvml_resident_bytes_diagnostic_only": True,
            "matched_full_context_guard": guard,
            "timed_signed_wave_count": 64,
            "perturbed_requests_all_engines": 71_680,
            "reference_requests_all_engines": 1_120,
            "guard_requests_all_engines": 2_240,
            "total_generation_requests": 75_040,
            "per_unit_scores_timing_vectors_raw_outputs_bootstrap_replicates_persisted": False,
            "model_update_applied": False,
            "nontrain_surface_opened": False,
        })
        base.runtime_v23a._assert_compact_persistence_v23a({
            "estimator": estimator, "audit": audit,
        })
        return estimator, audit


def load_runtime_trainer_r1(preregistration):
    prereg.validate_preregistration_v24a(preregistration)
    plan = preregistration["arms"]["bf16_a"]["layer_plan"]
    bundle = base.runtime_v23a.anchor_v4.load_frozen_layer_plan_v4(
        plan["path"],
        expected_file_sha256=plan["file_sha256"],
        expected_plan_sha256=plan["plan_sha256"],
        expected_model_config_sha256=prereg.BF16_CONFIG_SHA256_V24A,
    )
    parent = base.runtime_v23a.anchor_v4.load_trainer(bundle)

    class HybridBackendMemoryRetryRuntimeTrainerV24AR1(
        HybridBackendMemoryRetryMixinV24AR1,
        base.HybridBackendRuntimeMixinV24A,
        parent,
    ):
        def launch_engines(
            self,
            num_engines=4,
            n_gpu_per_vllm_engine=1,
            model_name="unused",
            precision="bfloat16",
        ):
            if num_engines != 4 or n_gpu_per_vllm_engine != 1:
                raise ValueError("v24a-r1 requires exactly four TP1 engines")
            import ray
            from es_at_scale.trainer.es_trainer import ESNcclLLM
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import (
                PlacementGroupSchedulingStrategy,
            )

            class TimedESNcclLLMV24AR1(ESNcclLLM):
                def generate_timed_v24a(self, *args, **kwargs):
                    import time
                    import torch

                    torch.cuda.synchronize()
                    started = time.perf_counter_ns()
                    outputs = super().generate(*args, **kwargs)
                    torch.cuda.synchronize()
                    return {
                        "outputs": outputs,
                        "elapsed_ns": time.perf_counter_ns() - started,
                    }

                def runtime_nvml_memory_v24a(self):
                    import os
                    import pynvml

                    token = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                    if not token.isdigit():
                        raise RuntimeError("v24a-r1 requires one physical GPU token")
                    physical = int(token)
                    pynvml.nvmlInit()
                    try:
                        info = pynvml.nvmlDeviceGetMemoryInfo(
                            pynvml.nvmlDeviceGetHandleByIndex(physical)
                        )
                        return {
                            "schema": "eggroll-es-v24a-nvml-memory",
                            "physical_gpu_id": physical,
                            "used_bytes": int(info.used),
                            "total_bytes": int(info.total),
                        }
                    finally:
                        pynvml.nvmlShutdown()

            pgs = [
                placement_group(
                    [{"GPU": 1, "CPU": 0}],
                    strategy="PACK",
                    lifetime="detached",
                )
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
            for rank, arm in enumerate(prereg.ARM_ORDER_V24A):
                engines.append(
                    ray.remote(
                        num_cpus=0,
                        num_gpus=1,
                        scheduling_strategy=strategies[rank],
                    )(TimedESNcclLLMV24AR1).remote(
                        model=preregistration["arms"][arm]["model_path"],
                        tensor_parallel_size=1,
                        worker_extension_cls=WORKER_EXTENSION_R1,
                        dtype=precision,
                        enable_prefix_caching=False,
                        enforce_eager=True,
                        gpu_memory_utilization=0.82,
                        max_model_len=2048,
                        limit_mm_per_prompt={"image": 0, "video": 0},
                        mm_processor_cache_gb=0,
                        skip_mm_profiling=True,
                        moe_backend="triton",
                    )
                )
            return engines, pgs

    return HybridBackendMemoryRetryRuntimeTrainerV24AR1


def _make_trainer_r1(preregistration):
    cls = load_runtime_trainer_r1(preregistration)
    runtime_base = base.runtime_v23a.base
    return cls(
        model_name=preregistration["model_contract"]["bf16"]["path"],
        checkpoint=None,
        sigma=0.0003,
        alpha=0.0,
        population_size=32,
        reward_shaping="z-scores",
        num_iterations=1,
        max_tokens=1,
        batch_size=56,
        mini_batch_size=56,
        reward_function=runtime_base.specialist_reward,
        template_function=runtime_base.specialist_template,
        train_dataloader=[],
        eval_dataloader_dict={},
        eval_freq=1,
        n_vllm_engines=4,
        n_gpu_per_vllm_engine=1,
        logging="none",
        global_seed=43,
        use_gpus="0,1,2,3",
        experiment_name=EXPERIMENT_NAME_R1,
        wandb_project="specialist-eggroll-es",
        save_best_models=False,
        reward_function_timeout=10,
        output_directory=str(OUTPUT_DIRECTORY_R1),
    )


def _parser_r1():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v24a-r1-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def validate_runtime_r1(args, preregistration, implementation, recipe):
    prereg.validate_preregistration_v24a(preregistration)
    if any(
        os.environ.get(key)
        for key in base.runtime_v23a.MOE_OVERRIDE_ENVIRONMENT_V23A
    ):
        raise ValueError("v24a-r1 rejects external MoE backend overrides")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise ValueError("v24a-r1 rejects the vLLM batch-invariant backend swap")
    for expected, actual, label in (
        (
            args.expected_implementation_bundle_sha256,
            implementation["bundle_sha256"],
            "implementation",
        ),
        (
            args.expected_recipe_sha256,
            recipe["content_sha256_before_self_field"],
            "recipe",
        ),
    ):
        if not args.v24a_r1_dry_run and expected is None:
            raise ValueError(f"v24a-r1 real launch requires expected {label} hash")
        if expected is not None and expected != actual:
            raise ValueError(f"v24a-r1 {label} hash changed")


def run_exact_r1(preregistration, implementation, recipe, failure):
    environment_certificate = base.runtime_r2.certify_runtime_environment_r2()
    attempt_path = OUTPUT_DIRECTORY_R1 / ATTEMPT_NAME_R1
    run_dir = OUTPUT_DIRECTORY_R1 / EXPERIMENT_NAME_R1
    if attempt_path.exists() or run_dir.exists():
        raise RuntimeError("v24a-r1 requires fresh exclusive attempt and run paths")
    provenance = base.runtime_v23a._source_provenance_v23a(implementation)
    live_model_audit = base.validate_live_contract_v24a(preregistration)
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v24a-memory-retry-r1",
        "status": "launching",
        "phase": "before_trainer_creation",
        "recipe": recipe,
        "retry_failure_evidence": failure,
        "source_provenance": provenance,
        "runtime_environment_certificate": environment_certificate,
        "live_model_audit": live_model_audit,
        "model_update_applied": False,
        "nontrain_surface_opened": False,
    }
    base.runtime_v23a._exclusive_write_json_v23a(attempt_path, attempt)
    if run_dir.exists():
        attempt.update({
            "status": "failed",
            "phase": "fresh_run_reservation_race",
            "failure": {
                "type": "RuntimeError",
                "message": "v24a-r1 run directory appeared after exclusive attempt claim",
            },
            "model_update_applied": False,
            "nontrain_surface_opened": False,
        })
        base.runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
        raise RuntimeError(
            "v24a-r1 run directory appeared after exclusive attempt claim"
        )
    trainer = None
    failure_value = None
    failure_traceback = None
    configured = None
    result = None
    try:
        base.runtime_v23a.base.set_seed(43)
        trainer = _make_trainer_r1(preregistration)
        panels = base.runtime_v23a.mechanics_v23a.load_panel_bundle_v23a()
        configured = trainer.configure_hybrid_backend_v24a(
            preregistration, panels
        )
        result = trainer.estimate_hybrid_backend_v24a()
    except BaseException as error:
        failure_value = error
        failure_traceback = traceback.format_exc()
    finally:
        if trainer is not None:
            try:
                base.runtime_v23a.base.close_trainer(trainer)
            except BaseException as cleanup_error:
                if failure_value is None:
                    failure_value = cleanup_error
                    failure_traceback = traceback.format_exc()
                else:
                    failure_traceback += "\nCleanup failure:\n" + traceback.format_exc()
    if failure_value is not None:
        attempt.update({
            "status": "failed",
            "phase": "inside_v24a_r1_train_only_runtime",
            "failure": {
                "type": type(failure_value).__name__,
                "message": str(failure_value),
                "traceback": failure_traceback,
            },
            "model_update_applied": False,
            "nontrain_surface_opened": False,
        })
        base.runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
        raise failure_value
    estimator, audit = result
    report = {
        "schema": "eggroll-es-hybrid-backend-compatibility-report-v24a-memory-retry-r1",
        "recipe": recipe,
        "configuration": configured,
        "estimator_and_gate": estimator,
        "runtime_audit": audit,
        "implementation": implementation,
        "retry_failure_evidence_content_sha256": (
            failure["content_sha256_before_self_field"]
        ),
        "model_update_applied": False,
        "nontrain_surface_opened": False,
        "direct_action_taken": False,
    }
    base.runtime_v23a._assert_compact_persistence_v23a(report)
    report_path = run_dir / REPORT_NAME_R1
    base.runtime_v23a._exclusive_write_json_v23a(report_path, report)
    attempt.update({
        "status": "complete",
        "phase": "after_cleanup_and_compact_report",
        "report_binding": {
            "path": str(report_path),
            "file_sha256": file_sha256(report_path),
            "content_sha256": report["content_sha256_before_self_field"],
        },
        "model_update_applied": False,
        "nontrain_surface_opened": False,
    })
    base.runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    base.runtime_v23a._assert_train_only_argv_v23a(argv)
    args = _parser_r1().parse_args(argv)
    failure = validate_original_memory_failure_r1()
    preregistration = base.load_preregistration_v24a()
    implementation = implementation_identity_r1()
    recipe = recipe_r1(preregistration, implementation, failure)
    validate_runtime_r1(args, preregistration, implementation, recipe)
    if args.v24a_r1_dry_run:
        payload = _seal({
            "schema": "eggroll-es-hybrid-backend-runtime-dry-run-v24a-memory-retry-r1",
            "recipe": recipe,
            "implementation": implementation,
            "retry_failure_evidence_content_sha256": (
                failure["content_sha256_before_self_field"]
            ),
            "fresh_exclusive_attempt_required": True,
            "real_launch_requires_exact_post_cherry_pick_implementation_and_recipe_hashes": True,
            "memory_preflight_precedes_a_b_c_and_first_perturbation": True,
            "nvml_is_diagnostic_only": True,
            "matched_full_context_guard_active": True,
            "gpu_launched": False,
        })
        base.runtime_v23a._assert_compact_persistence_v23a(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_r1(preregistration, implementation, recipe, failure)


if __name__ == "__main__":
    main()
