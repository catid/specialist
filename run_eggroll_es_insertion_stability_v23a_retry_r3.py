#!/usr/bin/env python3
"""V23A R3 retry with exact matched-full-context inference guards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

import build_eggroll_es_v23a_probe_context_failure_evidence_r3 as probe_failure_r3
import run_eggroll_es_insertion_stability_v23a_retry_r1 as runtime_r1
import run_eggroll_es_insertion_stability_v23a_retry_r2 as runtime_r2


ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME_R3 = (
    "insertion_location_stability_v23a_authoritative_raw_seed_retry_r3"
)
OUTPUT_DIRECTORY_R3 = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
ATTEMPT_NAME_R3 = f".{EXPERIMENT_NAME_R3}.launch_attempt.json"
REPORT_NAME_R3 = "insertion_location_stability_v23a_seed_retry_r3.json"
R2_ATTEMPT_PATH_R3 = ROOT / probe_failure_r3.ATTEMPT_RELATIVE_PATH_R3
R2_REPORT_PATH_R3 = ROOT / probe_failure_r3.REPORT_RELATIVE_PATH_R3
PROBE_FAILURE_PATH_R3 = probe_failure_r3.OUTPUT_PATH_R3
PROBE_FAILURE_FILE_SHA256_R3 = (
    "1b93d38b19d883899638945b0b215057179f181abcb43d42f89c9ba71f80b673"
)
PROBE_FAILURE_CONTENT_SHA256_R3 = (
    "6622f67662b7ba6c4a15ca3bbe0b3b2cb4af28e661731433d81d37033ab1031f"
)
TEST_PATH_R3 = (
    ROOT / "test_run_eggroll_es_insertion_stability_v23a_retry_r3.py"
).resolve()
IMPLEMENTATION_PATHS_R3 = {
    "probe_context_failure_builder_r3": Path(probe_failure_r3.__file__).resolve(),
    "probe_context_failure_tests_r3": (
        ROOT / "test_build_eggroll_es_v23a_probe_context_failure_evidence_r3.py"
    ),
    "probe_context_failure_r3": PROBE_FAILURE_PATH_R3,
    "retry_runtime_r3": Path(__file__).resolve(),
    "retry_runtime_tests_r3": TEST_PATH_R3,
}


canonical_sha256 = runtime_r1.canonical_sha256
file_sha256 = runtime_r1.file_sha256
_seal = runtime_r1._seal


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def validate_r2_probe_failure_r3(attempt_path=None, report_path=None):
    attempt_path = R2_ATTEMPT_PATH_R3 if attempt_path is None else Path(attempt_path)
    report_path = R2_REPORT_PATH_R3 if report_path is None else Path(report_path)
    persisted = json.loads(PROBE_FAILURE_PATH_R3.read_text(encoding="utf-8"))
    rebuilt = probe_failure_r3.build_probe_context_failure_evidence_r3(
        attempt_path, report_path
    )
    if (
        file_sha256(PROBE_FAILURE_PATH_R3) != PROBE_FAILURE_FILE_SHA256_R3
        or persisted.get("content_sha256_before_self_field")
        != PROBE_FAILURE_CONTENT_SHA256_R3
        or persisted.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(persisted))
        or rebuilt != persisted
    ):
        raise RuntimeError("v23a-r3 R2 probe context failure evidence changed")
    return persisted


def _score_arrays_sha256_r3(scores):
    arms = runtime_r1.prereg_v23a.ARM_ORDER_V23A
    if not isinstance(scores, dict) or tuple(scores) != arms:
        raise RuntimeError("v23a-r3 score arm coverage changed")
    identities = {}
    for arm in arms:
        values = np.asarray(scores[arm])
        if values.shape != (5, 56) or not np.isfinite(values).all():
            raise RuntimeError("v23a-r3 full-context score shape changed")
        raw = np.ascontiguousarray(values).tobytes(order="C")
        identities[arm] = {
            "dtype": values.dtype.str,
            "shape": list(values.shape),
            "byte_sha256": hashlib.sha256(raw).hexdigest(),
        }
    return canonical_sha256(identities)


def _phase_identity_r3(scores, commitments, probes):
    return {
        "dense_commitments_sha256": canonical_sha256(commitments),
        "probe_commitments_sha256": canonical_sha256(probes),
        "score_array_commitment_sha256": _score_arrays_sha256_r3(scores),
    }


def _exact_phase_comparison_r3(
    reference_scores, reference_commitments, reference_probes,
    candidate_scores, candidate_commitments, candidate_probes,
):
    arms = runtime_r1.prereg_v23a.ARM_ORDER_V23A
    score_equal = (
        isinstance(reference_scores, dict)
        and isinstance(candidate_scores, dict)
        and tuple(reference_scores) == arms
        and tuple(candidate_scores) == arms
        and all(np.array_equal(reference_scores[arm], candidate_scores[arm]) for arm in arms)
    )
    return {
        "all_four_score_arrays_exact": score_equal,
        "all_dense_commitments_exact": candidate_commitments == reference_commitments,
        "all_probe_commitments_exact": candidate_probes == reference_probes,
    }


class MatchedFullContextGuardMixinR3:
    """Add A/B/C full-batch identity guards without changing ES scoring."""

    def _prepared_train_requests_v23a(self):
        panels, requests, request_identity = super()._prepared_train_requests_v23a()
        if len(requests) != 280:
            raise RuntimeError("v23a-r3 requires the exact 280-request full context")
        self._v23a_r3_full_panels = panels
        # Preserve the exact list object for A, B, and C.
        self._v23a_r3_full_requests = requests
        self._v23a_r3_request_identity = request_identity
        self._v23a_r3_boundary_passed = False
        return panels, requests, request_identity

    def _score_batches_v23a(self, panels, batches):
        scores, commitments, probes = super()._score_batches_v23a(panels, batches)
        if not hasattr(self, "_v23a_r3_reference_scores"):
            # Phase A is the unchanged estimator reference.  Phase B repeats
            # the identical full request list and must pass before this method
            # returns to the parent's first perturbation loop.
            self._v23a_r3_reference_scores = {
                arm: np.array(scores[arm], copy=True)
                for arm in runtime_r1.prereg_v23a.ARM_ORDER_V23A
            }
            self._v23a_r3_reference_commitments = copy.deepcopy(commitments)
            self._v23a_r3_reference_probes = copy.deepcopy(probes)
            phase_a = _phase_identity_r3(scores, commitments, probes)
            repeat_batches = super()._generate_all_v23a(
                self._v23a_r3_full_requests
            )
            repeat_scores, repeat_commitments, repeat_probes = (
                super()._score_batches_v23a(panels, repeat_batches)
            )
            phase_b = _phase_identity_r3(
                repeat_scores, repeat_commitments, repeat_probes
            )
            comparison = _exact_phase_comparison_r3(
                self._v23a_r3_reference_scores,
                self._v23a_r3_reference_commitments,
                self._v23a_r3_reference_probes,
                repeat_scores, repeat_commitments, repeat_probes,
            )
            self._v23a_r3_guard = {
                "schema": "eggroll-es-v23a-matched-full-context-guard-r3",
                "request_identity_sha256": self._v23a_r3_request_identity,
                "requests_per_engine_each_phase": 280,
                "requests_all_engines_each_phase": 1_120,
                "same_request_list_object_all_three_phases": True,
                "same_request_list_value_identity_all_three_phases": True,
                "reference_phase_a": phase_a,
                "pre_population_repeat_phase_b": phase_b,
                "pre_population_exact_comparison": comparison,
                "pre_population_guard_completed_before_first_perturbation": True,
                "post_population_guard_completed_after_full_weight_audit": False,
                "guard_requests_excluded_from_endpoint_scoring": True,
                "raw_scores_or_responses_persisted": False,
            }
            repeat_scores = None
            repeat_batches = None
            if not all(comparison.values()):
                raise RuntimeError(
                    "v23a-r3 repeated full-context baseline is not exact"
                )
        return scores, commitments, probes

    def _boundary_audit_v23a(self):
        result = super()._boundary_audit_v23a()
        if result.get("passed") is not True:
            raise RuntimeError("v23a-r3 population boundary audit did not pass")
        self._v23a_r3_boundary_passed = True
        return result

    def _generate_all_v23a(self, requests):
        if len(requests) != 1:
            return super()._generate_all_v23a(requests)
        if (
            getattr(self, "_v23a_r3_boundary_passed", False) is not True
            or not hasattr(self, "_v23a_r3_guard")
        ):
            raise RuntimeError("v23a-r3 post guard reached before full weight audit")
        # The parent uses a one-request call as its post-population hook.  R3
        # deliberately replaces only that mismatched context with phase C,
        # the exact same 280-request list and order used by phases A and B.
        post_batches = super()._generate_all_v23a(self._v23a_r3_full_requests)
        post_scores, post_commitments, post_probes = super()._score_batches_v23a(
            self._v23a_r3_full_panels, post_batches
        )
        phase_c = _phase_identity_r3(post_scores, post_commitments, post_probes)
        comparison = _exact_phase_comparison_r3(
            self._v23a_r3_reference_scores,
            self._v23a_r3_reference_commitments,
            self._v23a_r3_reference_probes,
            post_scores, post_commitments, post_probes,
        )
        self._v23a_r3_guard.update({
            "post_population_phase_c": phase_c,
            "post_population_exact_comparison": comparison,
            "post_population_guard_completed_after_full_weight_audit": True,
            "full_context_phase_count": 3,
            "guard_only_phase_count": 2,
            "guard_requests_all_engines": 2_240,
        })
        post_scores = None
        if not all(comparison.values()):
            raise RuntimeError("v23a-r3 post-restore full-context guard is not exact")
        # The inherited compact probe now sees request zero from phase C in
        # exactly the same full-batch context as phase A.
        return post_batches

    def estimate_insertion_stability_v23a(self):
        estimator, gate, audit = super().estimate_insertion_stability_v23a()
        guard = getattr(self, "_v23a_r3_guard", None)
        if (
            not isinstance(guard, dict)
            or guard.get("post_population_guard_completed_after_full_weight_audit")
            is not True
            or not all(guard["pre_population_exact_comparison"].values())
            or not all(guard["post_population_exact_comparison"].values())
        ):
            raise RuntimeError("v23a-r3 full-context guard certificate is incomplete")
        audit.pop("content_sha256_before_self_field", None)
        audit.update({
            "post_restore_probe_requests_all_engines": 1_120,
            "full_context_reference_requests_all_engines": 1_120,
            "full_context_guard_requests_all_engines": 2_240,
            "total_unperturbed_requests_all_engines_including_guards": 3_360,
            "matched_full_context_guard_r3": copy.deepcopy(guard),
        })
        _seal(audit)
        runtime_r1.runtime_v23a._assert_compact_persistence_v23a(audit)
        return estimator, gate, audit


def implementation_identity_r3():
    inherited = runtime_r2.implementation_identity_r2()
    files = dict(inherited["files"])
    for key, path in IMPLEMENTATION_PATHS_R3.items():
        files[key] = {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
    if file_sha256(PROBE_FAILURE_PATH_R3) != PROBE_FAILURE_FILE_SHA256_R3:
        raise RuntimeError("v23a-r3 probe context failure evidence changed")
    return {
        "files": files,
        "inherited_v23a_r2_bundle_sha256": inherited["bundle_sha256"],
        "retry_overlay_bundle_sha256": canonical_sha256({
            key: files[key] for key in IMPLEMENTATION_PATHS_R3
        }),
        "bundle_sha256": canonical_sha256(files),
    }


def recipe_r3(
    preregistration, implementation, seed_failure,
    environment_failure, probe_failure,
):
    inherited = runtime_r2.recipe_r2(
        preregistration, implementation, seed_failure, environment_failure
    )
    value = copy.deepcopy(inherited)
    value.pop("content_sha256_before_self_field", None)
    value.update({
        "schema": "eggroll-es-insertion-location-runtime-recipe-v23a-seed-retry-r3",
        "experiment_name": EXPERIMENT_NAME_R3,
        "output_directory": str(OUTPUT_DIRECTORY_R3),
        "post_restore_probe_requests_all_engines": 1_120,
        "retry_of": {
            "experiment_name": runtime_r2.EXPERIMENT_NAME_R2,
            "failed_attempt_relative_path": probe_failure["failed_attempt"]["relative_path"],
            "failed_attempt_file_sha256": probe_failure["failed_attempt"]["file_sha256"],
            "failed_attempt_content_sha256": probe_failure["failed_attempt"]["content_sha256"],
            "failure_evidence_file_sha256": PROBE_FAILURE_FILE_SHA256_R3,
            "failure_evidence_content_sha256": PROBE_FAILURE_CONTENT_SHA256_R3,
            "prior_failure_chain": copy.deepcopy(inherited["retry_of"]),
            "r2_attempt_immutable": True,
            "r2_compact_report_absent": True,
        },
        "matched_full_context_guard_r3": {
            "phase_order": [
                "reference_full_context_a",
                "pre_population_repeat_full_context_b",
                "all_64_preregistered_signed_waves",
                "full_selected_and_unselected_weight_audit",
                "post_population_full_context_c",
            ],
            "requests_per_engine_each_full_context": 280,
            "requests_all_engines_each_full_context": 1_120,
            "reference_phase_requests_all_engines": 1_120,
            "guard_only_phase_count": 2,
            "guard_requests_all_engines": 2_240,
            "total_unperturbed_requests_all_engines_including_guards": 3_360,
            "pre_population_a_equals_b_exactly_before_first_perturbation": True,
            "post_population_a_equals_c_exactly_after_full_weight_audit": True,
            "score_arrays_dense_commitments_and_probe_commitments_all_exact": True,
            "same_request_identity_batch_shape_and_order_all_three_phases": True,
            "same_request_list_object_all_three_phases": True,
            "guard_requests_excluded_from_endpoint_scoring": True,
            "vllm_batch_invariant_backend_swap_disabled": True,
            "raw_scores_or_responses_persisted": False,
        },
        "same_preregistered_arms_basis_panels_endpoint_scoring_gate_and_seed_repair": True,
        "only_integrity_probe_batch_context_corrected": True,
        "fresh_exclusive_retry_paths": {
            "attempt_name": ATTEMPT_NAME_R3,
            "run_directory_name": EXPERIMENT_NAME_R3,
            "report_name": REPORT_NAME_R3,
            "all_disjoint_from_original_r1_and_r2": True,
        },
    })
    return _seal(value)


def load_runtime_trainer_r3(preregistration, seed_failure):
    parent = runtime_r1.load_runtime_trainer_r1(preregistration, seed_failure)

    class MatchedFullContextRuntimeTrainerR3(
        MatchedFullContextGuardMixinR3, parent,
    ):
        pass

    return MatchedFullContextRuntimeTrainerR3


def _make_trainer_r3(preregistration, seed_failure):
    cls = load_runtime_trainer_r3(preregistration, seed_failure)
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
        global_seed=43, use_gpus="0,1,2,3", experiment_name=EXPERIMENT_NAME_R3,
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10, output_directory=str(OUTPUT_DIRECTORY_R3),
    )


def _parser_r3():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v23a-r3-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def validate_runtime_r3(args, preregistration, implementation, recipe):
    runtime_r1.prereg_v23a.validate_preregistration_v23a(preregistration)
    if any(
        os.environ.get(key)
        for key in runtime_r1.runtime_v23a.MOE_OVERRIDE_ENVIRONMENT_V23A
    ):
        raise ValueError("v23a-r3 rejects external MoE backend overrides")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise ValueError("v23a-r3 rejects the vLLM batch-invariant backend swap")
    for expected, actual, label in (
        (args.expected_implementation_bundle_sha256, implementation["bundle_sha256"], "implementation"),
        (args.expected_recipe_sha256, recipe["content_sha256_before_self_field"], "recipe"),
    ):
        if not args.v23a_r3_dry_run and expected is None:
            raise ValueError(f"v23a-r3 real launch requires expected {label} hash")
        if expected is not None and expected != actual:
            raise ValueError(f"v23a-r3 {label} hash changed")


def run_exact_r3(
    preregistration, implementation, recipe, seed_failure,
    environment_failure, probe_failure,
):
    runtime_r1.validate_original_failure_r1()
    runtime_r2.validate_r1_environment_failure_r2()
    validate_r2_probe_failure_r3()
    environment_certificate = runtime_r2.certify_runtime_environment_r2()
    attempt_path = OUTPUT_DIRECTORY_R3 / ATTEMPT_NAME_R3
    run_dir = OUTPUT_DIRECTORY_R3 / EXPERIMENT_NAME_R3
    if attempt_path.exists() or run_dir.exists():
        raise RuntimeError("v23a-r3 requires fresh exclusive retry attempt and run paths")
    provenance = runtime_r1.runtime_v23a._source_provenance_v23a(implementation)
    model_directory_audit = runtime_r1.runtime_v23a.validate_live_model_directories_v23a(
        preregistration
    )
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v23a-seed-retry-r3",
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
        raise RuntimeError("v23a-r3 run directory appeared after exclusive attempt claim")
    trainer = None
    failure = None
    failure_traceback = None
    result = None
    configured = None
    try:
        runtime_r1.runtime_v23a.base.set_seed(43)
        trainer = _make_trainer_r3(preregistration, seed_failure)
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
            "phase": "inside_v23a_r3_train_only_runtime",
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
        "schema": "eggroll-es-insertion-location-stability-report-v23a-seed-retry-r3",
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
    report_path = run_dir / REPORT_NAME_R3
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
    args = _parser_r3().parse_args(argv)
    preregistration = runtime_r1.runtime_v23a._load_preregistration_v23a()
    seed_failure = runtime_r1.validate_original_failure_r1()
    environment_failure = runtime_r2.validate_r1_environment_failure_r2()
    probe_failure = validate_r2_probe_failure_r3()
    implementation = implementation_identity_r3()
    recipe = recipe_r3(
        preregistration, implementation, seed_failure,
        environment_failure, probe_failure,
    )
    validate_runtime_r3(args, preregistration, implementation, recipe)
    if args.v23a_r3_dry_run:
        payload = _seal({
            "schema": "eggroll-es-insertion-location-seed-retry-launch-dry-run-r3",
            "recipe": recipe,
            "implementation": implementation,
            "r2_probe_context_failure_revalidated": True,
            "matched_full_context_guard_preregistered": True,
            "new_retry_paths_exclusive_and_disjoint": True,
            "real_launch_requires_committed_bundle_recipe_and_runtime_environment": True,
            "gpu_launched": False,
        })
        runtime_r1.runtime_v23a._assert_compact_persistence_v23a(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_r3(
        preregistration, implementation, recipe, seed_failure,
        environment_failure, probe_failure,
    )


if __name__ == "__main__":
    main()
