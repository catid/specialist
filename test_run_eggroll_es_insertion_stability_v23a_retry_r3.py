import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run_eggroll_es_insertion_stability_v23a_retry_r1 as r1
import run_eggroll_es_insertion_stability_v23a_retry_r2 as r2
import run_eggroll_es_insertion_stability_v23a_retry_r3 as r3


MAIN_RUNS = Path("/home/catid/specialist/experiments/eggroll_es_hpo/runs")
MAIN_ORIGINAL_ATTEMPT = MAIN_RUNS / r1.runtime_v23a.ATTEMPT_NAME_V23A
MAIN_ORIGINAL_REPORT = (
    MAIN_RUNS / r1.runtime_v23a.EXPERIMENT_NAME_V23A
    / r1.runtime_v23a.REPORT_NAME_V23A
)
MAIN_R1_ATTEMPT = MAIN_RUNS / r1.ATTEMPT_NAME_R1
MAIN_R1_REPORT = MAIN_RUNS / r1.EXPERIMENT_NAME_R1 / r1.REPORT_NAME_R1
MAIN_R2_ATTEMPT = MAIN_RUNS / r2.ATTEMPT_NAME_R2
MAIN_R2_REPORT = MAIN_RUNS / r2.EXPERIMENT_NAME_R2 / r2.REPORT_NAME_R2


def _prereg():
    return r1.runtime_v23a._load_preregistration_v23a()


def _failures():
    seed = r1.validate_original_failure_r1(
        MAIN_ORIGINAL_ATTEMPT, MAIN_ORIGINAL_REPORT
    )
    environment = r2.validate_r1_environment_failure_r2(
        MAIN_R1_ATTEMPT, MAIN_R1_REPORT
    )
    probe = r3.validate_r2_probe_failure_r3(MAIN_R2_ATTEMPT, MAIN_R2_REPORT)
    return seed, environment, probe


class _FakeParent:
    def __init__(self, mismatch_b=False, mismatch_c=False):
        self.events = []
        self.generation_index = 0
        self.mismatch_b = mismatch_b
        self.mismatch_c = mismatch_c
        self.full_request_object_ids = []

    def _prepared_train_requests_v23a(self):
        return {"frozen": True}, [{"opaque": index} for index in range(280)], "r" * 64

    def _generate_all_v23a(self, requests):
        self.generation_index += 1
        phase = {1: "A", 2: "B", 3: "C"}[self.generation_index]
        self.events.append(f"generate_{phase}_{len(requests)}")
        if len(requests) == 280:
            self.full_request_object_ids.append(id(requests))
        return {"phase": phase, "request_count": len(requests)}

    def _score_batches_v23a(self, panels, batches):
        phase = batches["phase"]
        self.events.append(f"score_{phase}")
        value = 1.0
        if phase == "B" and self.mismatch_b:
            value = 2.0
        if phase == "C" and self.mismatch_c:
            value = 3.0
        scores = {
            arm: np.full((5, 56), value, dtype=np.float64)
            for arm in r1.prereg_v23a.ARM_ORDER_V23A
        }
        marker = "same" if value == 1.0 else f"changed-{phase}"
        commitments = {
            arm: [marker] * 5 for arm in r1.prereg_v23a.ARM_ORDER_V23A
        }
        probes = {arm: marker for arm in r1.prereg_v23a.ARM_ORDER_V23A}
        return scores, commitments, probes

    def _boundary_audit_v23a(self):
        self.events.append("full_weight_boundary_audit")
        return {"passed": True, "reports_sha256": "b" * 64}

    def estimate_insertion_stability_v23a(self):
        panels, requests, _identity = self._prepared_train_requests_v23a()
        batch_a = self._generate_all_v23a(requests)
        self._score_batches_v23a(panels, batch_a)
        self.events.append("first_perturbation")
        self._boundary_audit_v23a()
        self._generate_all_v23a([requests[0]])
        return {}, {}, {
            "schema": "fake-audit",
            "post_restore_probe_requests_all_engines": 4,
        }


class _Harness(r3.MatchedFullContextGuardMixinR3, _FakeParent):
    pass


def test_v23a_r3_full_context_schedule_is_exact_and_guard_only():
    harness = _Harness()
    _estimator, _gate, audit = harness.estimate_insertion_stability_v23a()
    assert harness.events == [
        "generate_A_280", "score_A",
        "generate_B_280", "score_B",
        "first_perturbation",
        "full_weight_boundary_audit",
        "generate_C_280", "score_C",
    ]
    assert len(set(harness.full_request_object_ids)) == 1
    guard = audit["matched_full_context_guard_r3"]
    assert guard["requests_per_engine_each_phase"] == 280
    assert guard["requests_all_engines_each_phase"] == 1_120
    assert guard["guard_requests_all_engines"] == 2_240
    assert guard["same_request_list_object_all_three_phases"] is True
    assert guard["same_request_list_value_identity_all_three_phases"] is True
    assert all(guard["pre_population_exact_comparison"].values())
    assert all(guard["post_population_exact_comparison"].values())
    assert guard["guard_requests_excluded_from_endpoint_scoring"] is True
    assert guard["raw_scores_or_responses_persisted"] is False
    assert audit["full_context_reference_requests_all_engines"] == 1_120
    assert audit["full_context_guard_requests_all_engines"] == 2_240
    assert audit["total_unperturbed_requests_all_engines_including_guards"] == 3_360
    assert "score_arrays" not in guard


def test_v23a_r3_repeat_guard_fails_before_first_perturbation():
    harness = _Harness(mismatch_b=True)
    with pytest.raises(RuntimeError, match="baseline is not exact"):
        harness.estimate_insertion_stability_v23a()
    assert harness.events == [
        "generate_A_280", "score_A", "generate_B_280", "score_B",
    ]
    assert "first_perturbation" not in harness.events


def test_v23a_r3_post_guard_occurs_only_after_full_weight_audit():
    harness = _Harness(mismatch_c=True)
    with pytest.raises(RuntimeError, match="post-restore full-context"):
        harness.estimate_insertion_stability_v23a()
    assert harness.events.index("full_weight_boundary_audit") < harness.events.index(
        "generate_C_280"
    )


def test_v23a_r3_recipe_changes_only_probe_context_and_paths():
    prereg = _prereg()
    seed, environment, probe = _failures()
    implementation = {"bundle_sha256": "a" * 64}
    retry = r3.recipe_r3(
        prereg, implementation, seed, environment, probe
    )
    inherited = r2.recipe_r2(prereg, implementation, seed, environment)
    for key in (
        "arms", "panel_contract", "fresh_basis", "runtime", "analysis",
        "authority", "worker_extension", "seed_domain_repair",
        "runtime_environment_contract_r2",
    ):
        assert retry[key] == inherited[key]
    assert retry["only_integrity_probe_batch_context_corrected"] is True
    assert retry[
        "same_preregistered_arms_basis_panels_endpoint_scoring_gate_and_seed_repair"
    ] is True
    guard = retry["matched_full_context_guard_r3"]
    assert guard["guard_requests_all_engines"] == 2_240
    assert guard["guard_requests_excluded_from_endpoint_scoring"] is True
    assert retry["post_restore_probe_requests_all_engines"] == 1_120
    assert retry["content_sha256_before_self_field"] == r3.canonical_sha256(
        r3._without_self(retry)
    )


def test_v23a_r3_paths_are_fresh_and_disjoint_from_all_attempts():
    r3_paths = {
        r3.OUTPUT_DIRECTORY_R3 / r3.ATTEMPT_NAME_R3,
        r3.OUTPUT_DIRECTORY_R3 / r3.EXPERIMENT_NAME_R3,
        r3.OUTPUT_DIRECTORY_R3 / r3.EXPERIMENT_NAME_R3 / r3.REPORT_NAME_R3,
    }
    earlier = {
        r1.ORIGINAL_ATTEMPT_PATH_R1, r1.ORIGINAL_REPORT_PATH_R1.parent,
        r1.ORIGINAL_REPORT_PATH_R1, MAIN_R1_ATTEMPT, MAIN_R1_REPORT.parent,
        MAIN_R1_REPORT, MAIN_R2_ATTEMPT, MAIN_R2_REPORT.parent, MAIN_R2_REPORT,
    }
    assert r3_paths.isdisjoint(earlier)
    assert all(not path.exists() for path in r3_paths)


def test_v23a_r3_real_launch_hashes_fail_closed():
    prereg = _prereg()
    implementation = {"bundle_sha256": "a" * 64}
    recipe = {"content_sha256_before_self_field": "b" * 64}
    with pytest.raises(ValueError, match="requires expected implementation"):
        r3.validate_runtime_r3(
            SimpleNamespace(
                v23a_r3_dry_run=False,
                expected_implementation_bundle_sha256=None,
                expected_recipe_sha256=None,
            ),
            prereg, implementation, recipe,
        )
    with pytest.raises(ValueError, match="recipe hash changed"):
        r3.validate_runtime_r3(
            SimpleNamespace(
                v23a_r3_dry_run=False,
                expected_implementation_bundle_sha256="a" * 64,
                expected_recipe_sha256="0" * 64,
            ),
            prereg, implementation, recipe,
        )


def test_v23a_r3_rejects_vllm_batch_invariant_backend_swap(monkeypatch):
    prereg = _prereg()
    args = SimpleNamespace(
        v23a_r3_dry_run=True,
        expected_implementation_bundle_sha256=None,
        expected_recipe_sha256=None,
    )
    implementation = {"bundle_sha256": "a" * 64}
    recipe = {"content_sha256_before_self_field": "b" * 64}
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(ValueError, match="batch-invariant backend swap"):
        r3.validate_runtime_r3(args, prereg, implementation, recipe)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    r3.validate_runtime_r3(args, prereg, implementation, recipe)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT")
    r3.validate_runtime_r3(args, prereg, implementation, recipe)


def test_v23a_r3_keeps_update_evaluation_and_checkpoint_surfaces_closed():
    prereg = _prereg()
    seed, environment, probe = _failures()
    recipe = r3.recipe_r3(
        prereg, {"bundle_sha256": "a" * 64}, seed, environment, probe
    )
    assert recipe["worker_update_surfaces_closed"] is True
    assert all(recipe["authority"][key] is False for key in (
        "model_update_allowed", "checkpoint_write_allowed",
        "evaluation_allowed", "dataset_promotion_allowed",
    ))
    factory = inspect.getsource(r3._make_trainer_r3)
    assert "checkpoint=None" in factory
    assert "alpha=0.0" in factory
    assert "eval_dataloader_dict={}" in factory
    assert "save_best_models=False" in factory
    source = inspect.getsource(r3.run_exact_r3)
    assert '"model_update_applied": False' in source
    assert '"nontrain_surface_opened": False' in source


def test_v23a_r3_certificate_and_failure_checks_precede_attempt_claim():
    source = inspect.getsource(r3.run_exact_r3)
    claim = source.index("_exclusive_write_json_v23a(attempt_path, attempt)")
    assert source.index("validate_r2_probe_failure_r3()") < claim
    assert source.index("certify_runtime_environment_r2()") < claim
    assert source.index("_make_trainer_r3") > claim
