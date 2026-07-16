#!/usr/bin/env python3

import copy
import json

import pytest

import finalize_lora_es_singleton_fcfs_evidence_v61d as subject
from test_lora_es_singleton_fcfs_calibration_v61d import _evidence_v61d


def _self_hashed(tmp_path, name, value):
    value = copy.deepcopy(value)
    value.pop("content_sha256_before_self_field", None)
    content_sha = subject.analysis.canonical_sha256_v61d(value)
    value["content_sha256_before_self_field"] = content_sha
    path = tmp_path / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return subject.SelfHashedSourceV61D(
        path, subject.file_sha256_v61d(path), content_sha
    ), value


def _repeat_rows(mean_f1=0.02, max_f1=0.05, mean_log=0.003, max_log=0.01):
    return [{
        "actor_rank": actor,
        "label": label,
        "periods": [0, 1],
        "generated_f1_mean_absolute_delta": mean_f1,
        "generated_f1_maximum_absolute_delta": max_f1,
        "generated_f1_band_counts": {},
        "teacher_logprob_mean_absolute_delta": mean_log,
        "teacher_logprob_maximum_absolute_delta": max_log,
        "teacher_logprob_band_counts": {},
    } for actor in range(4) for label in ("reference", "candidate")]


def _synthetic_sources(tmp_path):
    support_path = tmp_path / "support.json"
    support_path.write_bytes(subject.SUPPORT_AUDIT.read_bytes())
    support = subject.SelfHashedSourceV61D(
        support_path,
        subject.file_sha256_v61d(support_path),
        subject.SUPPORT_AUDIT_CONTENT_SHA256,
    )

    prior_report, _ = _self_hashed(tmp_path, "v61c-report.json", {
        "schema": "v61c-identical-state-paired-evaluator-report",
        "status": "complete_content_free_alpha_zero_characterization_sealed",
        "wall_runtime_seconds": 100.0,
        "evidence": {
            "generation_completions": 1088,
            "teacher_forced_requests": 1088,
        },
    })
    prior_intervals = {
        "generated_f1_delta": {"halfwidth": 0.02, "contains_zero": True},
        "generated_exact_delta": {"halfwidth": 0.0, "contains_zero": True},
        "generated_nonzero_delta": {"halfwidth": 0.0, "contains_zero": True},
        "teacher_forced_logprob_delta": {
            "halfwidth": 0.004,
            "contains_zero": True,
        },
    }
    prior_finalized, _ = _self_hashed(tmp_path, "v61c-finalized.json", {
        "schema": "v61c-paired-null-independent-finalizer",
        "status": "complete_numeric_only_evidence_verified_hpo_unauthorized",
        "source_hashes": {"report": {
            "file_sha256": prior_report.file_sha256,
            "content_sha256": prior_report.content_sha256,
        }},
        "ranking_primary_null": {
            "point": {
                "generated_f1_delta": 0.01,
                "generated_exact_delta": 0.0,
                "generated_nonzero_delta": 0.0,
                "teacher_forced_logprob_delta": 0.001,
            },
            "raw_primary_cluster_intervals": prior_intervals,
        },
        "ranking_metric_breakdown": {
            "within_actor_same_label_repeat": _repeat_rows(),
        },
        "exact_sentinel": {
            "passed": False,
            "individual_exact_flip_count": 2,
            "units_with_any_exact_flip": 1,
            "total_sentinel_units": 4,
            "flip_count_by_unit": {"0" * 64: 2},
            "flip_count_histogram_over_affected_units": {"2": 1},
            "maximum_absolute_individual_exact_delta": 1.0,
        },
    })

    prereg_value = json.loads(subject.PREREGISTRATION.read_text())
    prereg_value["matched_v61c_contract"].update({
        "v61c_finalizer_file_sha256": prior_finalized.file_sha256,
        "v61c_finalizer_content_sha256": prior_finalized.content_sha256,
    })
    prereg, prereg_value = _self_hashed(tmp_path, "prereg.json", prereg_value)

    evidence_value = _evidence_v61d()
    recipe = prereg_value["fixed_calibration_recipe"]
    state = {
        "canonical_fp32_master_sha256": recipe[
            "canonical_fp32_master_sha256"
        ],
        "bf16_runtime_values_sha256": recipe["bf16_runtime_values_sha256"],
        "canonical_master_identity_sha256": "1" * 64,
        "four_actor_certificate_sha256": "2" * 64,
    }
    receipts = [{
        "period_index": period,
        "before": copy.deepcopy(state),
        "after": copy.deepcopy(state),
        "identical_v434_state": True,
    } for period in range(4)]
    evidence_value["state_receipts"] = receipts
    evidence_value["numeric_state_receipts_sha256"] = (
        subject.analysis.canonical_sha256_v61d(receipts)
    )
    evidence, evidence_value = _self_hashed(
        tmp_path, "evidence.json", evidence_value
    )
    analysis_value = subject.analysis.build_analysis_v61d(evidence_value)
    analysis_source, analysis_value = _self_hashed(
        tmp_path, "analysis.json", analysis_value
    )

    actors = [{
        "schema": "singleton-fcfs-actor-identity-v61d",
        "pid": 1000 + gpu,
        "physical_gpu_id": gpu,
        "cuda_visible_devices": str(gpu),
        "cuda_current_device": 0,
        **copy.deepcopy(subject.analysis.RUNTIME_CONTROLS_V61D),
        "scheduler_class": "Scheduler",
        "singleton_active_sequence_limit": 1,
        "global_batch_invariance_claimed": False,
        "tuned_folder": prereg_value["runtime"]["tuned_folder"],
        "tuned_table_content_sha256": prereg_value["runtime"][
            "tuned_table_content_sha256"
        ],
    } for gpu in range(4)]
    gpu_rows = []
    gpu_activity = {"all_four_attributed_positive": True, "by_gpu": {}}
    for gpu in range(4):
        pid = 1000 + gpu
        for utilization in (10, 0):
            gpu_rows.append({
                "gpu": gpu,
                "expected_pid": pid,
                "compute_pids": [pid],
                "foreign_compute_pids": [],
                "utilization_percent": utilization,
                "memory_used_mib": 100,
            })
        gpu_activity["by_gpu"][str(gpu)] = {
            "expected_pid": pid,
            "samples": 2,
            "resident_samples": 2,
            "positive_samples": 1,
            "mean_resident_utilization_percent": 5.0,
            "peak_utilization_percent": 10,
            "peak_memory_used_mib": 100,
        }
    gpu_path = tmp_path / "gpu.jsonl"
    gpu_path.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in gpu_rows
    ))
    gpu_sha = subject.file_sha256_v61d(gpu_path)
    report_value = {
        "schema": "v61d-singleton-fcfs-paired-evaluator-report",
        "status": "complete_matched_content_free_characterization_sealed",
        "wall_runtime_seconds": 200.0,
        "preregistration_file_sha256": prereg.file_sha256,
        "preregistration_content_sha256": prereg.content_sha256,
        "support_audit_file_sha256": support.file_sha256,
        "support_audit_content_sha256": support.content_sha256,
        "runtime_determinism_controls": copy.deepcopy(
            subject.analysis.RUNTIME_CONTROLS_V61D
        ),
        "matched_v61c_panel_labels_metrics_bootstrap_thresholds": True,
        "v61c_thresholds_relaxed_or_changed": False,
        "panel_file_sha256": recipe["staged_panel_file_sha256"],
        "panel_content_sha256": recipe["staged_panel_content_sha256"],
        "master_state_receipt": state,
        "state_receipts_sha256": evidence_value[
            "numeric_state_receipts_sha256"
        ],
        "actor_identities": actors,
        "evidence": {
            "file_sha256": evidence.file_sha256,
            "content_sha256": evidence.content_sha256,
            "generation_completions": 1088,
            "teacher_forced_requests": 1088,
        },
        "analysis": {
            "file_sha256": analysis_source.file_sha256,
            "content_sha256": analysis_source.content_sha256,
        },
        "matched_v61c_finalizer": {
            "file_sha256": prior_finalized.file_sha256,
            "content_sha256": prior_finalized.content_sha256,
        },
        "gpu_activity": gpu_activity,
        "gpu_log_file_sha256": gpu_sha,
        "cleanup": {
            "engine_kill_count": 4,
            "placement_group_remove_count": 4,
            "all_four_gcs_states_removed": True,
            "after": [{"state": "REMOVED"} for _ in range(4)],
        },
        "final_gpu_idle": {
            "all_four_compute_process_lists_empty": True,
        },
        "alpha": 0.0,
        "adapter_update_or_candidate_materialization_performed": False,
        "holdback_ood_shadow_or_protected_opened": False,
        "raw_question_answer_or_generation_text_persisted": False,
        "hpo_selection_update_or_promotion_authorized": False,
    }
    report, _ = _self_hashed(tmp_path, "report.json", report_value)
    return subject.FinalizerSourcesV61D(
        preregistration=prereg,
        support_audit=support,
        evidence=evidence,
        analysis=analysis_source,
        report=report,
        gpu_log_path=gpu_path,
        gpu_log_file_sha256=gpu_sha,
        v61c_finalized=prior_finalized,
        v61c_report=prior_report,
    )


def test_v61d_finalizer_full_synthetic_fixture_is_numeric_only_and_closed(tmp_path):
    value = subject.build_finalized_v61d(_synthetic_sources(tmp_path))
    assert value["verification"][
        "stored_analysis_exactly_equals_independent_numeric_rebuild"
    ] is True
    assert value["verification"]["actor_runtime_identities"][
        "all_runtime_controls_exact"
    ] is True
    assert value["verification"]["gpu_activity_recomputed_from_numeric_log"][
        "foreign_compute_process_observations"
    ] == 0
    comparison = value["exact_sentinel_flip_and_concentration_comparison"]
    assert comparison["v61d"]["passed"] is True
    assert comparison["v61c"]["individual_exact_flip_count"] == 2
    performance = value["wall_runtime_and_request_throughput_comparison"]
    assert performance["wall_runtime_seconds"]["v61d_over_v61c"] == 2.0
    assert performance["request_units_per_second"]["v61d_over_v61c"] == 0.5
    encoded = json.dumps(value, sort_keys=True)
    assert '"question"' not in encoded and '"answer"' not in encoded
    assert value["frozen_decision_state"][
        "hpo_update_selection_or_promotion_authorized"
    ] is False


def test_v61d_production_finalizer_verifies_sealed_outputs():
    value = subject.build_finalized_v61d()
    verification = value["verification"]
    assert verification["preregistration_support_and_code_bindings_verified"]
    assert verification["v434_state_receipts"][
        "identical_before_after_periods"
    ] == 4
    assert verification["gpu_activity_recomputed_from_numeric_log"][
        "all_four_attributed_positive"
    ] is True
    assert verification["cleanup"]["engines_killed"] == 4
    assert value["exact_sentinel_flip_and_concentration_comparison"]["v61d"][
        "passed"
    ] is True
    assert value["frozen_decision_state"][
        "v61d_teacher_forced_logprob_primary_eligible"
    ] is False


def test_v61d_actor_identity_and_gpu_checks_fail_closed(tmp_path):
    sources = _synthetic_sources(tmp_path)
    prereg = subject._read_self_hashed_v61d(sources.preregistration)
    report = subject._read_self_hashed_v61d(sources.report)
    report["actor_identities"][0]["max_num_seqs"] = 68
    with pytest.raises(RuntimeError, match="actor runtime identity"):
        subject._verify_actor_identities_v61d(report, prereg)
    rows = sources.gpu_log_path.read_text().splitlines()
    changed = json.loads(rows[0])
    changed["foreign_compute_pids"] = [9999]
    rows[0] = json.dumps(changed, sort_keys=True)
    path = tmp_path / "foreign.jsonl"
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(RuntimeError, match="foreign or mismatched"):
        subject._gpu_summary_v61d(
            path, subject.file_sha256_v61d(path),
            subject._read_self_hashed_v61d(sources.report),
        )


def test_v61d_unsealed_production_hash_fails_before_file_access(monkeypatch):
    monkeypatch.setattr(subject, "EVIDENCE_FILE_SHA256", None)
    with pytest.raises(RuntimeError, match="not sealed: EVIDENCE_FILE_SHA256"):
        subject.production_sources_v61d()


def test_v61d_zero_width_comparison_never_invents_a_ratio():
    value = subject._scalar_comparison_v61d(0.0, 0.0)
    assert value["both_zero"] is True
    assert value["v61d_over_v61c"] is None
