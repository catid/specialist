#!/usr/bin/env python3
"""Fail-closed four-GPU train-only runtime for the immutable V24A scan."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

import eggroll_es_hybrid_backend_preregistration_v24a as prereg
import eggroll_es_worker_v23a_retry_r1 as worker_r1
import run_eggroll_es_insertion_stability_v23a as runtime_v23a
import run_eggroll_es_insertion_stability_v23a_retry_r2 as runtime_r2
import train_eggroll_es_hybrid_backend_v24a as mechanics


ROOT = Path(__file__).resolve().parent
PREREG_PATH = prereg.PREREGISTRATION_PATH_V24A
PREREG_FILE_SHA256 = "20b65b6d8c849580782be9a348a6c8ed705135058e91c21d7b26546e51fb6756"
PREREG_CONTENT_SHA256 = "9abd0736cbfae2d4f930ae62c7d147a32f4a418443af7524367394d4036e19bb"
EXPERIMENT_NAME = "s6_v24a_hybrid_backend_train_only_compatibility_runtime"
OUTPUT_DIRECTORY = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
ATTEMPT_NAME = f".{EXPERIMENT_NAME}.launch_attempt.json"
REPORT_NAME = "hybrid_backend_compatibility_v24a.json"
WORKER_EXTENSION = (
    "eggroll_es_worker_v23a_retry_r1."
    "InsertionLocationAuditWorkerExtensionV23ARetryR1"
)
TEST_PATH = (ROOT / "test_run_eggroll_es_hybrid_backend_v24a.py").resolve()
IMPLEMENTATION_PATHS = {
    "hybrid_builder_v24": ROOT / "build_qwen36_hybrid_fp8_selected_bf16_v24.py",
    "hybrid_builder_tests_v24": ROOT / "test_build_qwen36_hybrid_fp8_selected_bf16_v24.py",
    "hybrid_audit_builder_v24": ROOT / "audit_qwen36_hybrid_fp8_selected_bf16_v24.py",
    "hybrid_audit_tests_v24": ROOT / "test_audit_qwen36_hybrid_fp8_selected_bf16_v24.py",
    "hybrid_audit_v24": prereg.AUDIT_PATH_V24A,
    "preregistration_module_v24a": Path(prereg.__file__).resolve(),
    "preregistration_tests_v24a": ROOT / "test_eggroll_es_hybrid_backend_preregistration_v24a.py",
    "preregistration_v24a": PREREG_PATH,
    "seed_repair_worker_r1": Path(worker_r1.__file__).resolve(),
    "seed_repair_worker_tests_r1": ROOT / "test_eggroll_es_worker_v23a_retry_r1.py",
    "estimator_v24a": Path(mechanics.__file__).resolve(),
    "runtime_v24a": Path(__file__).resolve(),
    "runtime_tests_v24a": TEST_PATH,
}


canonical_sha256 = prereg.canonical_sha256
file_sha256 = prereg.file_sha256
_seal = runtime_v23a._seal_v23a


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def load_preregistration_v24a():
    value = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    if (
        file_sha256(PREREG_PATH) != PREREG_FILE_SHA256
        or value.get("content_sha256_before_self_field") != PREREG_CONTENT_SHA256
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
    ):
        raise RuntimeError("v24a preregistration identity changed")
    return prereg.validate_preregistration_v24a(value)


def implementation_identity_v24a():
    inherited = runtime_r2.implementation_identity_r2()
    files = dict(inherited["files"])
    for key, path in IMPLEMENTATION_PATHS.items():
        files[key] = {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
    return {
        "files": files,
        "inherited_v23a_r2_bundle_sha256": inherited["bundle_sha256"],
        "v24a_overlay_bundle_sha256": canonical_sha256({
            key: files[key] for key in IMPLEMENTATION_PATHS
        }),
        "bundle_sha256": canonical_sha256(files),
    }


def seed_projection_contract_v24a(preregistration):
    seeds = prereg.perturbation_seeds_v24a()
    projections = [
        {"full_seed": seed, "numpy_legacy_seed": seed % (2**32)}
        for seed in seeds
    ]
    projected = [item["numpy_legacy_seed"] for item in projections]
    if (
        len(seeds) != 32
        or len(set(seeds)) != 32
        or any(seed < 0 or seed > 2**63 - 1 for seed in seeds)
        or len(set(projected)) != 32
        or canonical_sha256(seeds)
        != preregistration["fresh_basis"]["direction_seed_list_sha256"]
    ):
        raise RuntimeError("v24a seed projection contract changed")
    return {
        "schema": "eggroll-es-v24a-seed-projection-contract",
        "direction_count": 32,
        "direction_seed_list_sha256": canonical_sha256(seeds),
        "full_to_numpy_projection_sha256": canonical_sha256(projections),
        "numpy_projection_rule": "full_seed modulo 2**32",
        "numpy_projection_unique_count": 32,
        "python_and_torch_receive_full_seed": True,
        "only_numpy_legacy_seed_is_projected": True,
        "worker_extension": WORKER_EXTENSION,
    }


def recipe_v24a(preregistration, implementation):
    value = {
        "schema": "eggroll-es-hybrid-backend-runtime-recipe-v24a",
        "experiment_name": EXPERIMENT_NAME,
        "preregistration": {
            "path": str(PREREG_PATH),
            "file_sha256": PREREG_FILE_SHA256,
            "content_sha256": PREREG_CONTENT_SHA256,
        },
        "hybrid_checkpoint_audit": copy.deepcopy(
            preregistration["hybrid_checkpoint_audit"]
        ),
        "model_contract": copy.deepcopy(preregistration["model_contract"]),
        "arms": copy.deepcopy(preregistration["arms"]),
        "pairing": copy.deepcopy(preregistration["pairing"]),
        "fresh_basis": copy.deepcopy(preregistration["fresh_basis"]),
        "panel_contract": copy.deepcopy(preregistration["panel_contract"]),
        "runtime": copy.deepcopy(preregistration["runtime"]),
        "analysis": copy.deepcopy(preregistration["analysis"]),
        "gate": copy.deepcopy(preregistration["gate"]),
        "authority": copy.deepcopy(preregistration["authority"]),
        "seed_domain_repair": seed_projection_contract_v24a(preregistration),
        "matched_full_context_guard": {
            "phase_order": [
                "reference_full_context_a",
                "pre_population_repeat_full_context_b",
                "all_64_timed_signed_waves",
                "full_selected_and_unselected_population_audit",
                "post_population_full_context_c",
            ],
            "same_280_request_list_object_value_shape_and_order_a_b_c": True,
            "a_equals_b_before_first_perturbation": True,
            "a_equals_c_after_full_population_audit": True,
            "score_arrays_dense_commitments_and_probe_commitments_exact": True,
            "guard_calls_excluded_from_quality_speed_and_bootstrap_endpoints": True,
            "guard_requests_all_engines": 2_240,
            "reference_requests_all_engines": 1_120,
            "total_unperturbed_requests_including_guards": 3_360,
            "total_generation_requests_including_guards": 75_040,
            "raw_outputs_or_scores_persisted": False,
        },
        "timing_contract": {
            "clock": "time.perf_counter_ns inside each engine actor",
            "cuda_synchronize_before_and_after_generate": True,
            "only_64_perturbed_full_context_calls_timed": True,
            "a_b_c_guard_calls_excluded": True,
            "perturb_restore_bootstrap_and_controller_excluded": True,
        },
        "nvml_memory_contract": {
            "source": "pynvml.nvmlDeviceGetMemoryInfo(handle).used",
            "one_physical_gpu_measurement_per_arm": True,
            "after_model_load_worker_warmup_plan_install_and_reference_capture": True,
            "before_first_train_reward_score": True,
        },
        "restoration_contract": {
            "selected_reference_copy_restore_and_exact_hash_every_signed_wave": True,
            "perturbation_iterates_selected_plan_only": True,
            "full_selected_and_unselected_byte_audit_after_all_64_waves": True,
            "no_update_checkpoint_eval_or_union_surface": True,
        },
        "worker_extension": WORKER_EXTENSION,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "output_directory": str(OUTPUT_DIRECTORY),
        "fresh_exclusive_paths": {
            "attempt_name": ATTEMPT_NAME,
            "run_directory_name": EXPERIMENT_NAME,
            "report_name": REPORT_NAME,
        },
        "model_update_applied": False,
        "nontrain_surface_opened": False,
    }
    return _seal(value)


def validate_live_contract_v24a(preregistration):
    audit, model_seal, layer_plan = prereg._validate_bound_sources_v24a()
    if (
        audit["content_sha256_before_self_field"]
        != preregistration["hybrid_checkpoint_audit"]["content_sha256"]
        or model_seal["content_sha256_before_self_field"]
        != prereg.MODEL_SEAL_CONTENT_SHA256_V24A
        or layer_plan["plan_sha256"] != prereg.LAYER_PLAN_SHA256_V24A
    ):
        raise RuntimeError("v24a live model or plan contract changed")
    return _seal({
        "schema": "eggroll-es-v24a-live-model-contract-audit",
        "hybrid_audit_content_sha256": audit["content_sha256_before_self_field"],
        "model_seal_content_sha256": model_seal["content_sha256_before_self_field"],
        "layer_plan_sha256": layer_plan["plan_sha256"],
        "bf16_model_path": preregistration["model_contract"]["bf16"]["path"],
        "hybrid_model_path": preregistration["model_contract"]["hybrid"]["path"],
        "selected_partition_exact_bf16": True,
        "retained_hybrid_partition_exact_fp8": True,
    })


def _score_arrays_sha256(scores):
    identities = {}
    for arm in prereg.ARM_ORDER_V24A:
        values = np.asarray(scores[arm])
        if values.shape != (5, 56) or not np.isfinite(values).all():
            raise RuntimeError("v24a full-context score shape changed")
        identities[arm] = {
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "byte_sha256": hashlib.sha256(
                np.ascontiguousarray(values).tobytes(order="C")
            ).hexdigest(),
        }
    return canonical_sha256(identities)


def _phase_identity(scores, commitments, probes):
    return {
        "score_array_commitment_sha256": _score_arrays_sha256(scores),
        "dense_commitments_sha256": canonical_sha256(commitments),
        "probe_commitments_sha256": canonical_sha256(probes),
    }


def _phases_equal(reference, candidate):
    ref_scores, ref_commitments, ref_probes = reference
    scores, commitments, probes = candidate
    return {
        "all_four_score_arrays_exact": all(
            np.array_equal(ref_scores[arm], scores[arm])
            for arm in prereg.ARM_ORDER_V24A
        ),
        "all_dense_commitments_exact": ref_commitments == commitments,
        "all_probe_commitments_exact": ref_probes == probes,
    }


class HybridBackendRuntimeMixinV24A:
    def configure_hybrid_backend_v24a(self, preregistration, panel_bundle):
        prereg.validate_preregistration_v24a(preregistration)
        runtime_v23a.anchor_v13.validate_panel_bundle_v13(panel_bundle)
        if (
            len(self.engines) != 4
            or self.n_vllm_engines != 4
            or self.n_gpu_per_vllm_engine != 1
            or self.population_size != 32
            or not math.isclose(float(self.sigma), 0.0003, rel_tol=0.0, abs_tol=0.0)
            or not math.isclose(float(self.alpha), 0.0, rel_tol=0.0, abs_tol=0.0)
        ):
            raise RuntimeError("v24a exact four-arm alpha-zero recipe changed")
        projection = seed_projection_contract_v24a(preregistration)
        seeds = prereg.perturbation_seeds_v24a()
        raw_certificates = self._resolve([
            engine.collective_rpc.remote(
                "seed_projection_certificate_v23a_r1",
                args=(
                    seeds,
                    projection["direction_seed_list_sha256"],
                    projection["full_to_numpy_projection_sha256"],
                ),
            ) for engine in self.engines
        ])
        certificates = [
            runtime_v23a._unwrap_one_v23a(item, "seed_projection_certificate_v23a_r1")
            for item in raw_certificates
        ]
        if len(certificates) != 4 or len({canonical_sha256(item) for item in certificates}) != 1:
            raise RuntimeError("v24a seed certificates disagree")
        device_raw = self._resolve([
            engine.collective_rpc.remote("runtime_device_identity_v23a", args=(arm,))
            for engine, arm in zip(self.engines, prereg.ARM_ORDER_V24A)
        ])
        devices = [
            runtime_v23a._unwrap_one_v23a(item, "runtime_device_identity_v23a")
            for item in device_raw
        ]
        for rank, arm in enumerate(prereg.ARM_ORDER_V24A):
            report = devices[rank]
            if (
                report.get("rank") != rank
                or report.get("world_size") != 4
                or report.get("arm") != arm
                or report.get("cuda_visible_devices") != str(rank)
                or report.get("runtime_cuda_device") != 0
                or report.get("update_surfaces_closed") is not True
            ):
                raise RuntimeError("v24a arm-to-physical-GPU mapping changed")
        plan = preregistration["arms"]["bf16_a"]["layer_plan"]
        installs_raw = self._resolve([
            engine.collective_rpc.remote(
                "install_layer_plan_v4",
                args=(
                    Path(plan["path"]).read_bytes(), plan["file_sha256"],
                    plan["plan_sha256"],
                    runtime_v23a.anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4,
                    runtime_v23a.anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4,
                ),
            ) for engine in self.engines
        ])
        installs = [
            runtime_v23a._unwrap_one_v23a(item, "install_layer_plan_v4")
            for item in installs_raw
        ]
        if any(
            item.get("installed") is not True
            or item.get("plan_sha256", item.get("layer_plan_sha256")) != plan["plan_sha256"]
            or item.get("runtime_selected_parameter_count") != 23
            or item.get("selected_element_count") != 142_999_552
            for item in installs
        ):
            raise RuntimeError("v24a selected layer-plan installation changed")
        references_raw = self._resolve([
            engine.collective_rpc.remote("save_self_exact_reference", args=())
            for engine in self.engines
        ])
        references = [
            runtime_v23a._unwrap_one_v23a(item, "save_self_exact_reference")
            for item in references_raw
        ]
        selected_raw = self._resolve([
            engine.collective_rpc.remote("verify_self_exact_reference", args=())
            for engine in self.engines
        ])
        selected = [
            runtime_v23a._unwrap_one_v23a(item, "verify_self_exact_reference")
            for item in selected_raw
        ]
        if (
            any(item.get("passed") is not True for item in selected)
            or len({canonical_sha256(item["reference"]) for item in selected}) != 1
        ):
            raise RuntimeError("v24a selected BF16 reference identity differs across arms")
        memory = self._resolve([
            engine.runtime_nvml_memory_v24a.remote() for engine in self.engines
        ])
        resident_bytes = {}
        for rank, arm in enumerate(prereg.ARM_ORDER_V24A):
            report = memory[rank]
            if (
                report.get("schema") != "eggroll-es-v24a-nvml-memory"
                or report.get("physical_gpu_id") != rank
                or not isinstance(report.get("used_bytes"), int)
                or report["used_bytes"] <= 0
            ):
                raise RuntimeError("v24a NVML memory contract changed")
            resident_bytes[arm] = report["used_bytes"]
        self._v24a_preregistration = copy.deepcopy(preregistration)
        self._v24a_panel_bundle = copy.deepcopy(panel_bundle)
        self._v24a_references = {
            arm: references[index]["identity"]
            for index, arm in enumerate(prereg.ARM_ORDER_V24A)
        }
        self._v24a_resident_bytes = resident_bytes
        return _seal({
            "schema": "eggroll-es-hybrid-backend-runtime-configuration-v24a",
            "device_identity_sha256": canonical_sha256(devices),
            "installations_sha256": canonical_sha256(installs),
            "selected_reference_identity_sha256": canonical_sha256(selected[0]["reference"]),
            "combined_reference_identities_sha256": canonical_sha256(self._v24a_references),
            "seed_certificates_sha256": canonical_sha256(certificates),
            "nvml_resident_bytes": resident_bytes,
            "panel_bundle_content_sha256": panel_bundle["content_sha256_before_self_field"],
            "update_and_nontrain_surfaces_closed": True,
        })

    def _prepared_requests_v24a(self):
        panels = {}
        requests = []
        cursor = 0
        for panel_name in prereg.PANEL_NAMES_V24A:
            panel = self._v24a_panel_bundle["panels"][panel_name]
            prompts = [runtime_v23a.base.specialist_template(item) for item in panel["questions"]]
            dense = runtime_v23a.anchor_v4.prepare_gold_answer_items_v4(
                self.tokenizer, prompts, panel["answers"]
            )
            if len(dense) != 56:
                raise RuntimeError("v24a panel request count changed")
            panels[panel_name] = {"slice": (cursor, cursor + 56), "dense": dense}
            requests.extend({"prompt_token_ids": item["prompt_token_ids"]} for item in dense)
            cursor += 56
        if len(requests) != 280:
            raise RuntimeError("v24a full request context changed")
        return panels, requests, canonical_sha256(requests)

    def _generate_v24a(self, requests, timed):
        if timed:
            raw = self._resolve([
                engine.generate_timed_v24a.remote(
                    requests, self._dense_sampling_params_v4(0), use_tqdm=False
                ) for engine in self.engines
            ])
            batches = [item.get("outputs") for item in raw]
            elapsed = [item.get("elapsed_ns") for item in raw]
            if any(not isinstance(value, int) or value <= 0 for value in elapsed):
                raise RuntimeError("v24a actor generation timing changed")
        else:
            batches = self._resolve([
                engine.generate.remote(
                    requests, self._dense_sampling_params_v4(0), use_tqdm=False
                ) for engine in self.engines
            ])
            elapsed = None
        if len(batches) != 4 or any(len(batch) != len(requests) for batch in batches):
            raise RuntimeError("v24a all-four-engine generation incomplete")
        return batches, elapsed

    def _score_v24a(self, panels, batches):
        scores, commitments, probes = {}, {}, {}
        for rank, arm in enumerate(prereg.ARM_ORDER_V24A):
            arm_scores = np.empty((5, 56), dtype=np.float64)
            hashes = []
            for panel_index, panel_name in enumerate(prereg.PANEL_NAMES_V24A):
                panel = panels[panel_name]
                start, stop = panel["slice"]
                rewards, digest = runtime_v23a._score_panel_outputs_v23a(
                    panel["dense"], batches[rank][start:stop]
                )
                arm_scores[panel_index] = rewards
                hashes.append(digest)
            first = panels[prereg.PANEL_NAMES_V24A[0]]["dense"][0]
            probe = runtime_v23a.anchor_v4.score_gold_answer_outputs_v4(
                [first], [batches[rank][0]]
            )
            scores[arm] = arm_scores
            commitments[arm] = hashes
            probes[arm] = canonical_sha256(probe)
        return scores, commitments, probes

    def _restore_v24a(self):
        restored = self._resolve([
            engine.collective_rpc.remote("restore_self_weights_exact", args=())
            for engine in self.engines
        ])
        restored = [
            runtime_v23a._unwrap_one_v23a(item, "restore_self_weights_exact")
            for item in restored
        ]
        checks = self._resolve([
            engine.collective_rpc.remote("verify_self_exact_reference", args=())
            for engine in self.engines
        ])
        checks = [
            runtime_v23a._unwrap_one_v23a(item, "verify_self_exact_reference")
            for item in checks
        ]
        if restored != [True] * 4 or any(item.get("passed") is not True for item in checks):
            raise RuntimeError("v24a exact selected restore changed")
        return canonical_sha256(checks)

    def _boundary_v24a(self):
        raw = self._resolve([
            self.engines[rank].collective_rpc.remote(
                "audit_population_completion_v4",
                args=(4, 1, self._v24a_references[arm]["sha256"]),
            ) for rank, arm in enumerate(prereg.ARM_ORDER_V24A)
        ])
        reports = [
            runtime_v23a._unwrap_one_v23a(item, "audit_population_completion_v4")
            for item in raw
        ]
        if any(
            item.get("passed") is not True
            or item.get("current_identity") != self._v24a_references[arm]
            for item, arm in zip(reports, prereg.ARM_ORDER_V24A)
        ):
            raise RuntimeError("v24a full selected/unselected population audit changed")
        return canonical_sha256(reports)

    def estimate_hybrid_backend_v24a(self):
        schedule = prereg.signed_wave_schedule_v24a()
        panels, requests, request_identity = self._prepared_requests_v24a()
        reference_batches, _ = self._generate_v24a(requests, timed=False)
        phase_a = self._score_v24a(panels, reference_batches)
        repeat_batches, _ = self._generate_v24a(requests, timed=False)
        phase_b = self._score_v24a(panels, repeat_batches)
        pre_equal = _phases_equal(phase_a, phase_b)
        if not all(pre_equal.values()):
            raise RuntimeError("v24a repeated full-context baseline is not exact")
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
                ) for engine in self.engines
            ])
            runtime_v23a._validate_perturbation_results_v23a(perturbed)
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
            raise RuntimeError("v24a signed-wave capture incomplete")
        boundary_sha = self._boundary_v24a()
        post_batches, _ = self._generate_v24a(requests, timed=False)
        phase_c = self._score_v24a(panels, post_batches)
        post_equal = _phases_equal(phase_a, phase_c)
        if not all(post_equal.values()):
            raise RuntimeError("v24a post-population full-context guard is not exact")
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
            "update_and_nontrain_surfaces_closed": True,
        }
        estimator = mechanics.build_compact_summary_v24a(
            unit_scores, reference_scores, timings, self._v24a_resident_bytes,
            self._v24a_panel_bundle,
            {arm: dict(integrity_template) for arm in prereg.ARM_ORDER_V24A},
        )
        unit_scores = reference_scores = timings = None
        guard = {
            "schema": "eggroll-es-v24a-matched-full-context-guard",
            "request_identity_sha256": request_identity,
            "same_request_list_object_value_shape_and_order_a_b_c": True,
            "phase_a": _phase_identity(*phase_a),
            "phase_b": _phase_identity(*phase_b),
            "phase_c": _phase_identity(*phase_c),
            "a_b_exact": pre_equal,
            "a_c_exact": post_equal,
            "completed_before_first_perturbation_and_after_full_audit": True,
            "guard_requests_excluded_from_all_endpoints": True,
            "raw_outputs_or_scores_persisted": False,
        }
        audit = _seal({
            "schema": "eggroll-es-hybrid-backend-runtime-audit-v24a",
            "signed_wave_schedule_sha256": canonical_sha256(schedule),
            "dense_result_commitments_sha256": canonical_sha256(dense_hashes),
            "restore_checks_sha256": canonical_sha256(restore_hashes),
            "population_boundary_audit_sha256": boundary_sha,
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
        runtime_v23a._assert_compact_persistence_v23a({
            "estimator": estimator, "audit": audit
        })
        return estimator, audit

    @staticmethod
    def _closed_surface_v24a(*_args, **_kwargs):
        raise RuntimeError("v24a closes update checkpoint evaluation and union surfaces")

    configure_anchor = _closed_surface_v24a
    configure_train_panels_v13 = _closed_surface_v24a
    estimate_train_panels_v13 = _closed_surface_v24a
    estimate_step_coefficients = _closed_surface_v24a
    apply_seed_coefficients = _closed_surface_v24a
    train_step = _closed_surface_v24a
    evaluate_handle = _closed_surface_v24a
    evaluate_population_on_batch = _closed_surface_v24a
    eval_step = _closed_surface_v24a
    fit = _closed_surface_v24a


def load_runtime_trainer_v24a(preregistration):
    prereg.validate_preregistration_v24a(preregistration)
    plan = preregistration["arms"]["bf16_a"]["layer_plan"]
    bundle = runtime_v23a.anchor_v4.load_frozen_layer_plan_v4(
        plan["path"],
        expected_file_sha256=plan["file_sha256"],
        expected_plan_sha256=plan["plan_sha256"],
        expected_model_config_sha256=prereg.BF16_CONFIG_SHA256_V24A,
    )
    parent = runtime_v23a.anchor_v4.load_trainer(bundle)

    class HybridBackendRuntimeTrainerV24A(HybridBackendRuntimeMixinV24A, parent):
        def launch_engines(
            self, num_engines=4, n_gpu_per_vllm_engine=1,
            model_name="unused", precision="bfloat16",
        ):
            if num_engines != 4 or n_gpu_per_vllm_engine != 1:
                raise ValueError("v24a requires exactly four TP1 engines")
            import ray
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            from es_at_scale.trainer.es_trainer import ESNcclLLM

            class TimedESNcclLLMV24A(ESNcclLLM):
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
                        raise RuntimeError("v24a requires one physical GPU token")
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
                placement_group([{"GPU": 1, "CPU": 0}], strategy="PACK", lifetime="detached")
                for _ in range(4)
            ]
            ray.get([pg.ready() for pg in pgs])
            strategies = [
                PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0,
                ) for pg in pgs
            ]
            engines = []
            for rank, arm in enumerate(prereg.ARM_ORDER_V24A):
                engines.append(ray.remote(
                    num_cpus=0, num_gpus=1, scheduling_strategy=strategies[rank]
                )(TimedESNcclLLMV24A).remote(
                    model=preregistration["arms"][arm]["model_path"],
                    tensor_parallel_size=1,
                    worker_extension_cls=WORKER_EXTENSION,
                    dtype=precision,
                    enable_prefix_caching=False,
                    enforce_eager=True,
                    gpu_memory_utilization=0.82,
                    max_model_len=2048,
                    limit_mm_per_prompt={"image": 0, "video": 0},
                    mm_processor_cache_gb=0,
                    skip_mm_profiling=True,
                    moe_backend="triton",
                ))
            return engines, pgs

    return HybridBackendRuntimeTrainerV24A


def _make_trainer_v24a(preregistration):
    cls = load_runtime_trainer_v24a(preregistration)
    base = runtime_v23a.base
    return cls(
        model_name=preregistration["model_contract"]["bf16"]["path"],
        checkpoint=None, sigma=0.0003, alpha=0.0, population_size=32,
        reward_shaping="z-scores", num_iterations=1, max_tokens=1,
        batch_size=56, mini_batch_size=56,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template,
        train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
        n_vllm_engines=4, n_gpu_per_vllm_engine=1, logging="none",
        global_seed=43, use_gpus="0,1,2,3", experiment_name=EXPERIMENT_NAME,
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10, output_directory=str(OUTPUT_DIRECTORY),
    )


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v24a-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def validate_runtime_v24a(args, preregistration, implementation, recipe):
    prereg.validate_preregistration_v24a(preregistration)
    if any(
        os.environ.get(key) for key in runtime_v23a.MOE_OVERRIDE_ENVIRONMENT_V23A
    ):
        raise ValueError("v24a rejects external MoE backend overrides")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise ValueError("v24a rejects the vLLM batch-invariant backend swap")
    for expected, actual, label in (
        (args.expected_implementation_bundle_sha256, implementation["bundle_sha256"], "implementation"),
        (args.expected_recipe_sha256, recipe["content_sha256_before_self_field"], "recipe"),
    ):
        if not args.v24a_dry_run and expected is None:
            raise ValueError(f"v24a real launch requires expected {label} hash")
        if expected is not None and expected != actual:
            raise ValueError(f"v24a {label} hash changed")


def run_exact_v24a(preregistration, implementation, recipe):
    environment_certificate = runtime_r2.certify_runtime_environment_r2()
    attempt_path = OUTPUT_DIRECTORY / ATTEMPT_NAME
    run_dir = OUTPUT_DIRECTORY / EXPERIMENT_NAME
    if attempt_path.exists() or run_dir.exists():
        raise RuntimeError("v24a requires fresh exclusive attempt and run paths")
    provenance = runtime_v23a._source_provenance_v23a(implementation)
    live_model_audit = validate_live_contract_v24a(preregistration)
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v24a",
        "status": "launching",
        "phase": "before_trainer_creation",
        "recipe": recipe,
        "source_provenance": provenance,
        "runtime_environment_certificate": environment_certificate,
        "live_model_audit": live_model_audit,
        "model_update_applied": False,
        "nontrain_surface_opened": False,
    }
    runtime_v23a._exclusive_write_json_v23a(attempt_path, attempt)
    if run_dir.exists():
        attempt.update({
            "status": "failed",
            "phase": "fresh_run_reservation_race",
            "failure": {
                "type": "RuntimeError",
                "message": "v24a run directory appeared after exclusive attempt claim",
            },
            "model_update_applied": False,
            "nontrain_surface_opened": False,
        })
        runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
        raise RuntimeError(
            "v24a run directory appeared after exclusive attempt claim"
        )
    trainer = None
    failure = None
    failure_traceback = None
    configured = None
    result = None
    try:
        runtime_v23a.base.set_seed(43)
        trainer = _make_trainer_v24a(preregistration)
        panels = runtime_v23a.mechanics_v23a.load_panel_bundle_v23a()
        configured = trainer.configure_hybrid_backend_v24a(preregistration, panels)
        result = trainer.estimate_hybrid_backend_v24a()
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
            "status": "failed",
            "phase": "inside_v24a_train_only_runtime",
            "failure": {
                "type": type(failure).__name__,
                "message": str(failure),
                "traceback": failure_traceback,
            },
            "model_update_applied": False,
            "nontrain_surface_opened": False,
        })
        runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
        raise failure
    estimator, audit = result
    report = {
        "schema": "eggroll-es-hybrid-backend-compatibility-report-v24a",
        "recipe": recipe,
        "configuration": configured,
        "estimator_and_gate": estimator,
        "runtime_audit": audit,
        "implementation": implementation,
        "model_update_applied": False,
        "nontrain_surface_opened": False,
        "direct_action_taken": False,
    }
    runtime_v23a._assert_compact_persistence_v23a(report)
    report_path = run_dir / REPORT_NAME
    runtime_v23a._exclusive_write_json_v23a(report_path, report)
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
    runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    runtime_v23a._assert_train_only_argv_v23a(argv)
    args = _parser().parse_args(argv)
    preregistration = load_preregistration_v24a()
    implementation = implementation_identity_v24a()
    recipe = recipe_v24a(preregistration, implementation)
    validate_runtime_v24a(args, preregistration, implementation, recipe)
    if args.v24a_dry_run:
        payload = _seal({
            "schema": "eggroll-es-hybrid-backend-runtime-dry-run-v24a",
            "recipe": recipe,
            "implementation": implementation,
            "fresh_exclusive_attempt_required": True,
            "real_launch_requires_exact_committed_implementation_and_recipe_hashes": True,
            "matched_full_context_guard_active": True,
            "gpu_launched": False,
        })
        runtime_v23a._assert_compact_persistence_v23a(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v24a(preregistration, implementation, recipe)


if __name__ == "__main__":
    main()
