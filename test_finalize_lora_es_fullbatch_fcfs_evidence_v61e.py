#!/usr/bin/env python3

import copy
import json

import pytest

import finalize_lora_es_fullbatch_fcfs_evidence_v61e as subject
from test_lora_es_fullbatch_fcfs_calibration_v61e import _evidence_v61e


def _write_self_hashed(tmp_path, name, value):
    value = copy.deepcopy(value)
    value.pop("content_sha256_before_self_field", None)
    content = subject.analysis.canonical_sha256_v61e(value)
    value["content_sha256_before_self_field"] = content
    path = tmp_path / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return subject.SelfHashedSourceV61E(
        path, subject.file_sha256_v61e(path), content
    ), value


def _synthetic_sources(tmp_path, sentinel_flip=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    prereg = subject.SelfHashedSourceV61E(
        subject.runtime.PREREGISTRATION,
        subject.PREREGISTRATION_FILE_SHA256,
        subject.PREREGISTRATION_CONTENT_SHA256,
    )
    support = subject.SelfHashedSourceV61E(
        subject.runtime.AUDIT,
        subject.SUPPORT_AUDIT_FILE_SHA256,
        subject.SUPPORT_AUDIT_CONTENT_SHA256,
    )
    prereg_value = subject._read_self_hashed_v61e(prereg)

    evidence_value = _evidence_v61e()
    if sentinel_flip:
        evidence_value["rows"][64]["periods"][1]["actors"][0][
            "generation"
        ] = {"f1": 1.0, "exact": 1, "nonzero": 1}
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
        subject.analysis.canonical_sha256_v61e(receipts)
    )
    evidence, evidence_value = _write_self_hashed(
        tmp_path, "evidence.json", evidence_value
    )
    analysis_value = subject.analysis.build_analysis_v61e(evidence_value)
    stored_analysis, _ = _write_self_hashed(
        tmp_path, "analysis.json", analysis_value
    )

    actors = [{
        "schema": "fullbatch-fcfs-actor-identity-v61e",
        "pid": 1000 + gpu,
        "physical_gpu_id": gpu,
        "cuda_visible_devices": str(gpu),
        "cuda_current_device": 0,
        **copy.deepcopy(subject.analysis.RUNTIME_CONTROLS_V61E),
        "scheduler_class": "Scheduler",
        "fullbatch_active_sequence_limit": 68,
        "effective_request_batch_size": 68,
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
    gpu_sha = subject.file_sha256_v61e(gpu_path)
    report_value = {
        "schema": "v61e-fullbatch-fcfs-paired-evaluator-report",
        "status": "complete_matched_content_free_characterization_sealed",
        "wall_runtime_seconds": 120.0,
        "preregistration_file_sha256": prereg.file_sha256,
        "preregistration_content_sha256": prereg.content_sha256,
        "support_audit_file_sha256": support.file_sha256,
        "support_audit_content_sha256": support.content_sha256,
        "runtime_determinism_controls": copy.deepcopy(
            subject.analysis.RUNTIME_CONTROLS_V61E
        ),
        "matched_numeric_sources": copy.deepcopy(
            subject.runtime.MATCHED_SOURCE_IDENTITIES_V61E
        ),
        "matched_v61c_panel_labels_metrics_bootstrap_thresholds": True,
        "v61c_effective_request_batch_size": 68,
        "v61c_thresholds_relaxed_or_changed": False,
        "panel_file_sha256": recipe["staged_panel_file_sha256"],
        "panel_content_sha256": recipe["staged_panel_content_sha256"],
        "prior_numeric_evidence_opened": False,
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
            "file_sha256": stored_analysis.file_sha256,
            "content_sha256": stored_analysis.content_sha256,
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
    report, _ = _write_self_hashed(tmp_path, "report.json", report_value)
    matched = subject.runtime.MATCHED_SOURCE_IDENTITIES_V61E
    return subject.FinalizerSourcesV61E(
        preregistration=prereg,
        support_audit=support,
        evidence=evidence,
        analysis=stored_analysis,
        report=report,
        gpu_log_path=gpu_path,
        gpu_log_file_sha256=gpu_sha,
        v61c_finalized=subject.SelfHashedSourceV61E(
            subject.V61C_FINALIZED,
            matched["v61c_finalizer"]["file_sha256"],
            matched["v61c_finalizer"]["content_sha256"],
        ),
        v61d_finalized=subject.SelfHashedSourceV61E(
            subject.V61D_FINALIZED,
            matched["v61d_finalizer"]["file_sha256"],
            matched["v61d_finalizer"]["content_sha256"],
        ),
    )


def test_v61e_finalizer_accepts_both_sentinel_outcomes_without_authorization(
    tmp_path,
):
    passed = subject.build_finalized_v61e(
        _synthetic_sources(tmp_path / "pass", sentinel_flip=False)
    )
    failed = subject.build_finalized_v61e(
        _synthetic_sources(tmp_path / "fail", sentinel_flip=True)
    )
    assert passed["observed_decision_inputs_without_authorization"][
        "v61e_exact_sentinel_passed"
    ] is True
    assert failed["observed_decision_inputs_without_authorization"][
        "v61e_exact_sentinel_passed"
    ] is False
    for value in (passed, failed):
        assert value["frozen_non_authorization"][
            "outcome_assumed_before_read"
        ] is False
        assert value["frozen_non_authorization"][
            "hpo_update_selection_or_promotion_authorized"
        ] is False


def test_v61e_production_finalizer_verifies_sealed_numeric_outputs():
    value = subject.build_finalized_v61e()
    verification = value["verification"]
    assert verification[
        "preregistration_support_code_and_prior_hash_chain_verified"
    ] is True
    assert verification[
        "stored_analysis_exactly_equals_independent_numeric_rebuild"
    ] is True
    assert verification["actor_runtime_identities"][
        "all_runtime_controls_exact"
    ] is True
    assert verification["v434_state_receipts"][
        "identical_before_after_periods"
    ] == 4
    assert verification["gpu_activity_recomputed_from_numeric_log"][
        "foreign_compute_process_observations"
    ] == 0
    assert verification["cleanup"]["engines_killed"] == 4
    sentinel = value["exact_sentinel_flip_and_concentration_comparison"]
    assert sentinel["v61e"]["individual_exact_flip_count"] == 1
    assert sentinel["affected_unit_hash_overlap_with_v61c"] == [
        "27c984ca0a7dbd4ffead6e79d7b691dc7a37856356a9a050a40603c11c6dbda7"
    ]


def test_v61e_actor_and_gpu_checks_fail_closed(tmp_path):
    sources = _synthetic_sources(tmp_path)
    prereg = subject._read_self_hashed_v61e(sources.preregistration)
    report = subject._read_self_hashed_v61e(sources.report)
    report["actor_identities"][0]["max_num_seqs"] = 1
    with pytest.raises(RuntimeError, match="actor runtime identity"):
        subject._verify_actor_identities_v61e(report, prereg)
    rows = sources.gpu_log_path.read_text().splitlines()
    changed = json.loads(rows[0])
    changed["foreign_compute_pids"] = [9999]
    rows[0] = json.dumps(changed, sort_keys=True)
    path = tmp_path / "foreign.jsonl"
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(RuntimeError, match="foreign or mismatched"):
        subject._gpu_summary_v61e(
            path,
            subject.file_sha256_v61e(path),
            subject._read_self_hashed_v61e(sources.report),
        )


def test_v61e_unsealed_hash_fails_before_live_path_access(monkeypatch):
    monkeypatch.setattr(subject, "EVIDENCE_FILE_SHA256", None)
    with pytest.raises(RuntimeError, match="not sealed: EVIDENCE_FILE_SHA256"):
        subject.production_sources_v61e()


def test_v61e_three_way_zero_baselines_do_not_invent_ratios():
    value = subject._three_way_v61e(0.0, 0.0, 0.0)
    assert value["v61e_over_v61d"] is None
    assert value["v61e_over_v61c"] is None
