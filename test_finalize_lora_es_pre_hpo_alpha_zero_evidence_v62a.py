#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

import finalize_lora_es_pre_hpo_alpha_zero_evidence_v62a as subject


def test_v62a_production_finalizer_verifies_complete_fail_closed_chain():
    value = subject.build_finalized_v62a()
    verification = value["verification"]
    assert verification[
        "stored_analysis_exactly_equals_independent_numeric_rebuild"
    ] is True
    assert all(
        item["forbidden_text_key_count"] == 0
        for item in verification["no_text_leakage"].values()
    )
    assert verification["actor_runtime_identities"]["physical_gpu_ids"] == [
        0, 1, 2, 3
    ]
    gpu = verification["gpu_activity_recomputed_from_numeric_log"]
    assert gpu["gpu_log_rows"] == 404
    assert gpu["samples_per_gpu"] == 101
    assert gpu["foreign_compute_process_observations"] == 0
    assert gpu["all_four_attributed_positive"] is True
    assert verification["v434_state_receipts"][
        "identical_before_after_periods"
    ] == 4
    cleanup = verification["report_hash_chain_cleanup_and_idle"]
    assert cleanup["engines_killed"] == cleanup["placement_groups_removed"] == 4
    assert cleanup["all_four_final_gpu_compute_process_lists_empty"] is True
    gate = value["observed_numeric_outcome_without_authorization"][
        "required_pre_hpo_gate"
    ]
    assert gate["passed"] is False
    assert gate["failed_gate_count"] == 3
    assert gate["passed_gate_count"] == 0
    assert value["frozen_non_authorization"][
        "hpo_population_launch_or_update_authorized"
    ] is False


def test_v62a_finalizer_preserves_numeric_results_and_pair_diagnosis():
    value = subject.build_finalized_v62a()
    numeric = value["observed_numeric_outcome_without_authorization"]
    primary = numeric["primary_generated_f1"]
    actor = numeric["actor_influence"]
    assert primary["point"] == 0.0015665791191098438
    assert primary["lcb"] == 9.740701817317855e-06
    assert primary["ucb"] == 0.0035177158566941503
    assert primary["halfwidth"] == 0.0017539875774384163
    assert actor["maximum_absolute_leave_one_actor_out_shift"] == (
        0.0012677874589318485
    )
    diagnosis = value["first_pair_vs_later_pair_descriptive_diagnosis"]
    assert diagnosis["pairs"][0]["point_mean_f1_delta"] == (
        0.0030980949759174593
    )
    assert diagnosis["pairs"][1]["point_mean_f1_delta"] == (
        3.506326230222757e-05
    )
    assert diagnosis["cold_start_or_order_effect_claimed"] is False
    assert diagnosis["causal_interpretation_authorized"] is False


def test_v62a_finalizer_is_outcome_agnostic_but_never_authorizes():
    failed = {
        "checks": {
            "null_primary_ci_contains_zero": False,
            "primary_ci_halfwidth_at_most_frozen_limit_inclusive": False,
            "actor_leave_one_out_shift_at_most_frozen_limit_inclusive": False,
        },
        "passed": False,
        "maximum_primary_ci_halfwidth_inclusive": (
            subject.analysis.MAX_PRIMARY_CI_HALFWIDTH_V62A
        ),
        "maximum_actor_leave_one_out_shift_inclusive": (
            subject.analysis.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62A
        ),
        "failure_action": "fail_closed_before_hpo_population_or_update",
        "authorizes_hpo_population_or_update": False,
    }
    passed = copy.deepcopy(failed)
    passed["checks"] = {key: True for key in failed["checks"]}
    passed["passed"] = True
    for gate in (failed, passed):
        observed = subject._gate_observation_v62a(gate)
        assert observed["authorizes_hpo_population_or_update"] is False
    assert subject._gate_observation_v62a(failed)["failed_gate_count"] == 3
    assert subject._gate_observation_v62a(passed)["failed_gate_count"] == 0


def test_v62a_analysis_and_text_tamper_fail_closed():
    sources = subject.production_sources_v62a()
    evidence = subject._read_self_hashed_v62a(sources.evidence)
    stored = subject._read_self_hashed_v62a(sources.analysis)
    changed = copy.deepcopy(stored)
    changed["primary_generated_f1"]["point"] += 1e-6
    with pytest.raises(RuntimeError, match="exact numeric rebuild"):
        subject._verify_analysis_rebuild_v62a(evidence, changed)
    leaked = copy.deepcopy(evidence)
    leaked["rows"][0]["question"] = "forbidden"
    with pytest.raises(RuntimeError, match="forbidden text-bearing key"):
        subject._verify_no_text_keys_v62a("evidence", leaked)


def test_v62a_actor_and_gpu_tamper_fail_closed(tmp_path):
    sources = subject.production_sources_v62a()
    prereg = subject._read_self_hashed_v62a(sources.preregistration)
    report = subject._read_self_hashed_v62a(sources.report)
    changed_report = copy.deepcopy(report)
    changed_report["actor_identities"][0]["max_num_seqs"] = 1
    with pytest.raises(RuntimeError, match="actor runtime identity"):
        subject._verify_actor_identities_v62a(changed_report, prereg)

    rows = sources.gpu_log_path.read_text(encoding="utf-8").splitlines()
    changed_row = json.loads(rows[0])
    changed_row["foreign_compute_pids"] = [999999]
    rows[0] = json.dumps(changed_row, sort_keys=True)
    gpu_path = tmp_path / "tampered_gpu.jsonl"
    gpu_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="foreign or mismatched"):
        subject._gpu_summary_v62a(
            gpu_path,
            subject.file_sha256_v62a(gpu_path),
            report,
        )


def test_v62a_unsealed_production_hash_fails_before_inputs(monkeypatch):
    monkeypatch.setattr(subject, "EVIDENCE_FILE_SHA256", None)
    with pytest.raises(RuntimeError, match="not sealed: evidence file"):
        subject.production_sources_v62a()
