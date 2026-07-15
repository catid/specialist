#!/usr/bin/env python3

import copy
import json

import pytest

import build_eggroll_es_v33a_negative_evidence_v33b as evidence


def _artifacts():
    attempt = json.loads(evidence.ATTEMPT_PATH_V33B.read_text(encoding="utf-8"))
    report = json.loads(evidence.REPORT_PATH_V33B.read_text(encoding="utf-8"))
    return attempt, report


def test_v33b_binds_exact_completed_v33a_artifacts_and_source_history():
    value = evidence.build_negative_evidence_v33b()
    assert value["artifacts"]["durable_attempt"] == {
        "relative_path": evidence.ATTEMPT_RELATIVE_PATH_V33B,
        "file_sha256": evidence.ATTEMPT_FILE_SHA256_V33B,
        "content_sha256": evidence.ATTEMPT_CONTENT_SHA256_V33B,
    }
    assert value["artifacts"]["compact_report"] == {
        "relative_path": evidence.REPORT_RELATIVE_PATH_V33B,
        "file_sha256": evidence.REPORT_FILE_SHA256_V33B,
        "content_sha256": evidence.REPORT_CONTENT_SHA256_V33B,
    }
    history = value["source_history"]
    assert history["preregistration_and_clean_launch_source_commit"] == (
        evidence.PREREGISTRATION_AND_SOURCE_COMMIT_V33B
    )
    assert history["candidate_input_freeze_commit"] == (
        evidence.CANDIDATE_INPUT_FREEZE_COMMIT_V33B
    )
    assert history["implementation_bundle_sha256"] == (
        evidence.IMPLEMENTATION_BUNDLE_SHA256_V33B
    )
    assert history["implementation_file_count"] == 68
    assert history["immutable_bound_file_count"] == 12


def test_v33b_recomputes_exact_compact_negative_gate_summary():
    result = evidence.build_negative_evidence_v33b()["aggregate_result"]
    assert result == {
        "endpoint_count": 12,
        "negative_point_estimate_count": 1,
        "zero_point_estimate_count": 0,
        "positive_point_estimate_count": 11,
        "negative_familywise_lcb_count": 12,
        "best_familywise_lcb": -0.078125,
        "worst_familywise_lcb": -0.2749971566674678,
        "all_observed_point_deltas_nonnegative": False,
        "all_familywise_lcbs_nonnegative": False,
        "gate_pass": False,
    }


def test_v33b_preserves_high_power_execution_cardinality_and_integrity():
    execution = evidence.build_negative_evidence_v33b()["aggregate_execution"]
    assert execution["direction_count"] == 64
    assert execution["engine_signed_direction_evaluation_count"] == 128
    assert execution["synchronized_four_engine_signed_wave_count"] == 32
    assert execution["perturbed_request_count_all_engines"] == 49_920
    assert execution["full_context_request_count_all_engines"] == 4_680
    assert execution["total_generation_request_count"] == 54_600
    assert execution[
        "all_runtime_integrity_guards_and_idle_boundaries_passed"
    ] is True
    assert execution["prelaunch_idle_certificate_sha256"] == (
        evidence.PRELAUNCH_IDLE_SHA256_V33B
    )
    assert execution["final_idle_certificate_sha256"] == (
        evidence.FINAL_IDLE_SHA256_V33B
    )


def test_v33b_retains_production_v13_and_grants_no_direct_authority():
    value = evidence.build_negative_evidence_v33b()
    assert value["decision"] == {
        "retain_production_dataset": True,
        "retain_v13_recipe": True,
        "v364_replacement_authorized": False,
        "direct_followup_authority": False,
        "direct_confirmation_authority": False,
        "dataset_promotion_authority": False,
        "model_update_authority": False,
        "checkpoint_write_authority": False,
        "evaluation_authority": False,
        "nontrain_data_surface_authority": False,
    }
    assert all(item is False for item in value["side_effects"].values())
    assert value["detailed_payloads_persisted"] is False
    assert value["nontrain_data_content_persisted"] is False


@pytest.mark.parametrize(
    "path_name,message",
    (
        ("ATTEMPT_PATH_V33B", "attempt file hash changed"),
        ("REPORT_PATH_V33B", "report file hash changed"),
    ),
)
def test_v33b_changed_artifact_bytes_fail_closed(
    monkeypatch, tmp_path, path_name, message,
):
    source = getattr(evidence, path_name)
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    monkeypatch.setattr(evidence, path_name, changed)
    with pytest.raises(RuntimeError, match=message):
        evidence.validate_bound_artifacts_v33b()


def test_v33b_tampered_endpoint_or_gate_fails_closed():
    _attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["summary"]["paired_bootstrap"]["endpoints"][
        "train_screen_cosine_median"
    ]["familywise_lcb"] = 0.0
    with pytest.raises(RuntimeError, match="exact 12 endpoint aggregates"):
        evidence._validate_gate_v33b(changed, changed["summary"])
    changed = copy.deepcopy(report)
    changed["gate"]["pass"] = True
    changed["gate"] = evidence._seal(changed["gate"])
    with pytest.raises(RuntimeError, match="gate self hash changed"):
        evidence._validate_gate_v33b(changed, changed["summary"])


def test_v33b_tampered_direction_cardinality_restore_or_boundary_fails_closed():
    attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["recipe"]["request_accounting"]["total_generation_requests"] += 1
    changed["recipe"] = evidence._seal(changed["recipe"])
    with pytest.raises(RuntimeError, match="cardinality changed"):
        evidence._validate_runtime_and_cardinality_v33b(attempt, changed)
    changed = copy.deepcopy(report)
    changed["runtime_audit"]["restore_checks_sha256"] = "0" * 64
    changed["runtime_audit"] = evidence._seal(changed["runtime_audit"])
    with pytest.raises(RuntimeError, match="runtime audit self hash changed"):
        evidence._validate_runtime_and_cardinality_v33b(attempt, changed)
    changed = copy.deepcopy(report)
    changed["summary"]["runtime_integrity"][
        "same_resident_perturbation_scored_both_versions"
    ] = False
    changed["summary"] = evidence._seal(changed["summary"])
    with pytest.raises(RuntimeError, match="summary self hash changed"):
        evidence._validate_runtime_and_cardinality_v33b(attempt, changed)


def test_v33b_tampered_side_effect_or_idle_binding_fails_closed():
    attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["checkpoint_written"] = True
    with pytest.raises(RuntimeError, match="forbidden mutation"):
        evidence._validate_closed_side_effects_v33b(attempt, changed)
    changed = copy.deepcopy(attempt)
    changed["final_idle_certificate_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="final-idle integrity"):
        evidence._validate_closed_side_effects_v33b(changed, report)


def test_v33b_tampered_source_history_or_implementation_fails_closed(monkeypatch):
    monkeypatch.setattr(evidence, "RUNTIME_FILE_SHA256_V33B", "0" * 64)
    with pytest.raises(RuntimeError, match="runtime history changed"):
        evidence.validate_history_v33b()
    attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["implementation"]["bundle_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="implementation binding changed"):
        evidence._validate_source_and_implementation_v33b(attempt, changed)


@pytest.mark.parametrize(
    "forbidden",
    (
        "questions", "answers", "prompt", "token", "responses", "rows",
        "records", "paired_bootstrap", "draws", "replicates", "heldout",
        "validation", "ood", "benchmark",
    ),
)
def test_v33b_compact_evidence_rejects_record_or_nontrain_detail_terms(forbidden):
    with pytest.raises(RuntimeError, match="forbidden detail terms"):
        evidence._assert_compact_v33b({forbidden: "not allowed"})


def test_v33b_build_is_deterministic_and_dry_run_does_not_write(capsys):
    first = evidence.build_negative_evidence_v33b()
    second = evidence.build_negative_evidence_v33b()
    assert first == second
    value = evidence.main(["--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert value == first
    assert output["content_sha256"] == first["content_sha256_before_self_field"]
    assert output["gate_pass"] is False
    assert output["direct_followup_authority"] is False


def test_v33b_materialized_evidence_is_exact_immutable_build_output():
    materialized = json.loads(evidence.OUTPUT_PATH_V33B.read_text(encoding="utf-8"))
    assert materialized == evidence.build_negative_evidence_v33b()
    assert materialized["content_sha256_before_self_field"] == (
        evidence.canonical_sha256(evidence._without_self(materialized))
    )


def test_v33b_exclusive_evidence_write_is_immutable(monkeypatch, tmp_path):
    output = tmp_path / "negative.json"
    monkeypatch.setattr(evidence, "OUTPUT_PATH_V33B", output)
    value = evidence.build_negative_evidence_v33b()
    evidence._exclusive_write_json_v33b(output, value)
    with pytest.raises(RuntimeError, match="already exists"):
        evidence._exclusive_write_json_v33b(output, value)
