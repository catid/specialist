#!/usr/bin/env python3

from __future__ import annotations

import copy
import json

import pytest

import finalize_lora_es_pre_hpo_alpha_zero_evidence_v62b as subject


def _production_inputs():
    sources = subject.production_sources_v62b()
    prereg = subject._read_self_hashed_v62b(sources.preregistration)
    evidence = subject._read_self_hashed_v62b(sources.evidence)
    stored = subject._read_self_hashed_v62b(sources.analysis)
    report = subject._read_self_hashed_v62b(sources.report)
    return sources, prereg, evidence, stored, report


def test_v62b_production_finalizer_verifies_complete_eligible_chain():
    value = subject.build_finalized_v62b()
    verification = value["verification"]
    assert verification[
        "stored_analysis_exactly_equals_independent_numeric_rebuild"
    ] is True
    assert all(
        item["forbidden_text_key_count"] == 0
        for item in verification["no_text_leakage"].values()
    )
    actors = verification["actor_runtime_identities"]
    assert actors["physical_gpu_ids"] == [0, 1, 2, 3]
    assert actors["enforce_eager_all_actors"] is True
    assert actors["batch_invariance_all_actors"] is False
    assert actors["max_num_seqs_all_actors"] == 68
    schedule = verification["fixed_complete_warmup_and_scored_schedule"]
    assert schedule["unscored_warmup_periods"] == 4
    assert schedule["scored_periods"] == 24
    assert schedule["warmup_generation_completions_discarded"] == 1088
    assert schedule["scored_generation_completions"] == 6528
    assert schedule["total_generation_completions"] == 7616
    assert schedule["all_scored_periods_included_without_early_stop"] is True
    states = verification["v434_state_receipts"]
    assert states["identical_before_after_periods"] == 28
    assert states["single_unchanged_master_receipt_across_all_periods"] is True
    gpu = verification["gpu_activity_recomputed_from_numeric_log"]
    assert gpu["gpu_log_rows"] == 2336
    assert gpu["samples_per_gpu"] == 584
    assert gpu["foreign_compute_process_observations"] == 0
    assert gpu["all_four_attributed_positive"] is True
    assert gpu["all_four_peak_utilization_percent"] == 100
    assert all(
        item["peak_utilization_percent"] == 100
        and item["positive_samples"] > 0
        for item in gpu["by_gpu"].values()
    )
    cleanup = verification["report_hash_chain_cleanup_and_idle"]
    assert cleanup["engines_killed"] == cleanup["placement_groups_removed"] == 4
    assert cleanup["all_four_gcs_states_removed"] is True
    assert cleanup["all_four_final_gpu_compute_process_lists_empty"] is True
    gate = value["observed_numeric_outcome_without_authorization"][
        "required_pre_hpo_gate"
    ]
    assert gate["passed"] is True
    assert gate["passed_gate_count"] == 3
    assert gate["failed_gate_count"] == 0
    eligibility = value["calibration_eligibility_observation"]
    assert eligibility[
        "eligible_for_later_separately_preregistered_hpo_work"
    ] is True
    assert eligibility["eligibility_is_not_launch_or_update_authority"] is True
    assert eligibility["hpo_population_launch_or_update_authorized"] is False


def test_v62b_finalizer_preserves_observed_pass_numbers_and_thresholds():
    value = subject.build_finalized_v62b()
    numeric = value["observed_numeric_outcome_without_authorization"]
    primary = numeric["primary_generated_f1"]
    actor = numeric["actor_influence"]
    gate = numeric["required_pre_hpo_gate"]
    assert primary["point"] == 0.0001196795720630882
    assert primary["lcb"] == -0.0006029408388919361
    assert primary["ucb"] == 0.0009279340613286808
    assert primary["halfwidth"] == 0.0007654374501103085
    assert actor["maximum_absolute_leave_one_actor_out_shift"] == (
        0.00023041336137876811
    )
    assert gate["maximum_primary_ci_halfwidth_inclusive"] == (
        0.000773822590292528
    )
    assert gate["maximum_actor_leave_one_out_shift_inclusive"] == (
        0.0012119648781783704
    )
    assert gate["checks"] == {
        "null_primary_ci_contains_zero": True,
        "primary_ci_halfwidth_at_most_frozen_limit_inclusive": True,
        "actor_leave_one_out_shift_at_most_frozen_limit_inclusive": True,
    }


def test_v62b_finalizer_is_outcome_agnostic_but_never_authorizes():
    failed = {
        "checks": {
            "null_primary_ci_contains_zero": False,
            "primary_ci_halfwidth_at_most_frozen_limit_inclusive": False,
            "actor_leave_one_out_shift_at_most_frozen_limit_inclusive": False,
        },
        "passed": False,
        "maximum_primary_ci_halfwidth_inclusive": (
            subject.analysis.MAX_PRIMARY_CI_HALFWIDTH_V62B
        ),
        "maximum_actor_leave_one_out_shift_inclusive": (
            subject.analysis.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62B
        ),
        "failure_action": "fail_closed_before_hpo_population_or_update",
        "authorizes_hpo_population_or_update": False,
    }
    passed = copy.deepcopy(failed)
    passed["checks"] = {key: True for key in failed["checks"]}
    passed["passed"] = True
    for gate in (failed, passed):
        observed = subject._gate_observation_v62b(gate)
        assert observed["authorizes_hpo_population_or_update"] is False
    assert subject._gate_observation_v62b(failed)["failed_gate_count"] == 3
    assert subject._gate_observation_v62b(passed)["failed_gate_count"] == 0


def test_v62b_analysis_and_text_tamper_fail_closed():
    _, _, evidence, stored, _ = _production_inputs()
    changed = copy.deepcopy(stored)
    changed["primary_generated_f1"]["point"] += 1e-6
    with pytest.raises(RuntimeError, match="exact numeric rebuild"):
        subject._verify_analysis_rebuild_v62b(evidence, changed)
    leaked = copy.deepcopy(evidence)
    leaked["rows"][0]["question"] = "forbidden"
    with pytest.raises(RuntimeError, match="forbidden text-bearing key"):
        subject._verify_no_text_keys_v62b("evidence", leaked)


def test_v62b_actor_state_and_schedule_tamper_fail_closed():
    _, prereg, evidence, stored, report = _production_inputs()
    changed_report = copy.deepcopy(report)
    changed_report["actor_identities"][0]["max_num_seqs"] = 1
    with pytest.raises(RuntimeError, match="actor runtime identity"):
        subject._verify_actor_identities_v62b(changed_report, prereg)

    changed_state = copy.deepcopy(evidence)
    changed_state["scored_state_receipts"][23]["after"] = {
        **changed_state["scored_state_receipts"][23]["after"],
        "bf16_runtime_values_sha256": "0" * 64,
    }
    with pytest.raises(RuntimeError, match="state receipts"):
        subject._verify_state_v62b(changed_state, report, prereg)

    changed_schedule = copy.deepcopy(evidence)
    changed_schedule["rows"][0]["scored_periods"].pop()
    with pytest.raises(RuntimeError, match="fixed warmup or scored schedule"):
        subject._verify_fixed_schedule_v62b(
            changed_schedule, report, prereg, stored
        )


def test_v62b_gpu_foreign_process_and_peak_tamper_fail_closed(tmp_path):
    sources, _, _, _, report = _production_inputs()
    rows = sources.gpu_log_path.read_text(encoding="utf-8").splitlines()
    changed_row = json.loads(rows[0])
    changed_row["foreign_compute_pids"] = [999999]
    rows[0] = json.dumps(changed_row, sort_keys=True)
    foreign_path = tmp_path / "foreign_gpu.jsonl"
    foreign_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="foreign or mismatched"):
        subject._gpu_summary_v62b(
            foreign_path,
            subject.file_sha256_v62b(foreign_path),
            report,
        )

    peak_rows = []
    for line in sources.gpu_log_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["gpu"] == 0 and item["utilization_percent"] == 100:
            item["utilization_percent"] = 99
        peak_rows.append(json.dumps(item, sort_keys=True))
    peak_path = tmp_path / "peak_gpu.jsonl"
    peak_path.write_text("\n".join(peak_rows) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="100-percent peak"):
        subject._gpu_summary_v62b(
            peak_path,
            subject.file_sha256_v62b(peak_path),
            report,
        )


def test_v62b_cleanup_and_non_authorization_tamper_fail_closed():
    sources, _, evidence, stored, report = _production_inputs()
    rebuilt = subject._verify_analysis_rebuild_v62b(evidence, stored)
    changed = copy.deepcopy(report)
    changed["cleanup"]["engine_kill_count"] = 3
    with pytest.raises(RuntimeError, match="cleanup changed"):
        subject._verify_report_v62b(changed, rebuilt, sources)
    changed = copy.deepcopy(report)
    changed["hpo_population_launch_authorized"] = True
    with pytest.raises(RuntimeError, match="cleanup changed"):
        subject._verify_report_v62b(changed, rebuilt, sources)


def test_v62b_unsealed_production_hash_fails_before_inputs(monkeypatch):
    monkeypatch.setattr(subject, "EVIDENCE_FILE_SHA256", None)
    with pytest.raises(RuntimeError, match="not sealed: evidence file"):
        subject.production_sources_v62b()
