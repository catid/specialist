import copy
import json

import pytest

import build_eggroll_es_v401_replacement_fraction_negative_evidence_v34d as evidence


def _artifacts():
    return (
        json.loads(evidence.ATTEMPT_PATH_V34D.read_text(encoding="utf-8")),
        json.loads(evidence.REPORT_PATH_V34D.read_text(encoding="utf-8")),
    )


def test_v34d_binds_exact_completed_compact_artifacts():
    value = evidence.build_negative_evidence_v34d()
    assert value["artifacts"] == {
        "durable_attempt": {
            "relative_path": evidence.ATTEMPT_RELATIVE_PATH_V34D,
            "file_sha256": evidence.ATTEMPT_FILE_SHA256_V34D,
            "content_sha256": evidence.ATTEMPT_CONTENT_SHA256_V34D,
        },
        "compact_report": {
            "relative_path": evidence.REPORT_RELATIVE_PATH_V34D,
            "file_sha256": evidence.REPORT_FILE_SHA256_V34D,
            "content_sha256": evidence.REPORT_CONTENT_SHA256_V34D,
        },
    }


def test_v34d_binds_source_implementation_recipe_model_and_lifecycle():
    bindings = evidence.build_negative_evidence_v34d()["frozen_bindings"]
    assert bindings == {
        "committed_source_certificate_sha256": (
            evidence.COMMITTED_SOURCE_CERTIFICATE_SHA256_V34D
        ),
        "implementation_bundle_sha256": evidence.IMPLEMENTATION_BUNDLE_SHA256_V34D,
        "recipe_content_sha256": evidence.RECIPE_CONTENT_SHA256_V34D,
        "runtime_environment_certificate_sha256": (
            evidence.RUNTIME_ENVIRONMENT_CERTIFICATE_SHA256_V34D
        ),
        "live_model_audit_sha256": evidence.LIVE_MODEL_AUDIT_SHA256_V34D,
        "prelaunch_idle_certificate_sha256": (
            evidence.PRELAUNCH_IDLE_CERTIFICATE_SHA256_V34D
        ),
        "final_idle_certificate_sha256": (
            evidence.FINAL_IDLE_CERTIFICATE_SHA256_V34D
        ),
        "postcleanup_binding_recheck_sha256": (
            evidence.POSTCLEANUP_BINDING_RECHECK_SHA256_V34D
        ),
        "configuration_content_sha256": evidence.CONFIGURATION_CONTENT_SHA256_V34D,
        "preanalysis_audit_content_sha256": (
            evidence.PREANALYSIS_AUDIT_CONTENT_SHA256_V34D
        ),
        "summary_content_sha256": evidence.SUMMARY_CONTENT_SHA256_V34D,
        "fixed_sequence_content_sha256": evidence.FIXED_SEQUENCE_CONTENT_SHA256_V34D,
        "authorization_gate_content_sha256": evidence.GATE_CONTENT_SHA256_V34D,
    }


def test_v34d_binds_v34b_v34c_prereg_panel_and_candidate_manifest_contracts():
    contracts = evidence.build_negative_evidence_v34d()[
        "transitively_frozen_source_contracts"
    ]
    assert contracts["v34b_commit"] == evidence.V34B_COMMIT_V34D
    assert contracts["v34b_source_files"] == evidence.V34B_SOURCE_FILES_V34D
    assert contracts["v34c_commit"] == evidence.V34C_COMMIT_V34D
    assert contracts["v34c_source_files"] == evidence.V34C_SOURCE_FILES_V34D
    assert contracts["preregistration"] == evidence.PREREGISTRATION_IDENTITY_V34D
    assert contracts["panels"] == evidence.PANEL_IDENTITIES_V34D
    assert contracts["candidate_manifest"] == (
        evidence.CANDIDATE_MANIFEST_IDENTITY_V34D
    )
    assert contracts["production"] == evidence.PRODUCTION_IDENTITY_V34D
    assert contracts["model_and_layer_plan"] == (
        evidence.MODEL_AND_LAYER_IDENTITIES_V34D
    )


def test_v34d_all_32_waves_four_gpus_and_54600_requests_are_bound():
    execution = evidence.build_negative_evidence_v34d()["aggregate_execution"]
    assert execution["synchronized_signed_wave_count"] == 32
    assert execution["physical_gpu_count"] == 4
    assert execution["all_four_tp1_engines_both_sources_every_wave"] is True
    assert execution["perturbed_request_count"] == 49_920
    assert execution["full_context_request_count"] == 4_680
    assert execution["fraction_specific_request_count"] == 0
    assert execution["total_generation_request_count"] == 54_600


def test_v34d_activity_configuration_cleanup_and_origin_integrity_pass():
    execution = evidence.build_negative_evidence_v34d()["aggregate_execution"]
    assert execution["configuration"] == {
        "content_sha256": evidence.CONFIGURATION_CONTENT_SHA256_V34D,
        "device_identity_sha256": (
            "a62153ba381c77400ecdaf2601326e174365def247f28e9ec00fba085e0f6a30"
        ),
        "installation_sha256": (
            "182b2d2f69d2860fea1c620da61721d2845a39b14ffd4ec8fe4e31ca09daaa61"
        ),
        "selected_reference_identity_sha256": (
            "8123122857a895cd94e8f45119ed30e770536517e0a1d0606b9a45432da6fd8f"
        ),
        "alpha_zero": True,
    }
    assert execution["all_runtime_integrity_audits_passed"] is True
    assert execution["origin_and_population_boundary_audits_passed"] is True
    assert execution["failure_cleanup_and_final_all_gpu_idle_passed"] is True


def test_v34d_full_context_a_b_c_guard_is_exact_and_excluded():
    execution = evidence.build_negative_evidence_v34d()["aggregate_execution"]
    exact = {
        "all_dense_result_commitments_exact": True,
        "all_source_engine_panel_score_arrays_exact": True,
    }
    assert execution["a_b_exact"] == exact
    assert execution["a_c_exact"] == exact
    assert execution["phase_a_b_c_commitments_identical"] is True
    assert execution["full_context_guard_excluded_from_fraction_analysis"] is True


def test_v34d_exact_5_percent_metrics_fail_all_12_familywise_lcbs():
    result = evidence.build_negative_evidence_v34d()["fixed_sequence_result"]
    assert result["tested_fraction"] == 0.05
    assert result["tested_fraction_passed"] is False
    assert result["endpoint_count"] == 12
    assert result["familywise_lcb_failure_count"] == 12
    assert result["all_12_familywise_lcbs_failed"] is True
    assert result["all_12_point_deltas_nonnegative"] is False
    assert result["exact_endpoints"] == evidence._expected_fraction_endpoints_v34d()
    assert all(
        endpoint["familywise_lcb"] < 0.0
        for endpoint in result["exact_endpoints"].values()
    )


def test_v34d_fixed_sequence_stops_and_does_not_infer_higher_fractions():
    result = evidence.build_negative_evidence_v34d()["fixed_sequence_result"]
    assert result["production_fraction"] == 0.0
    assert result["tested_fraction_count"] == 1
    assert result["stopped_at_first_failure"] is True
    assert result["largest_consecutively_passing_fraction"] == 0.0
    assert result["untested_fractions_after_first_failure"] == [0.1, 0.2, 0.4, 1.0]
    assert result["no_higher_fraction_result_inferred"] is True
    assert result["fraction_specific_model_request_count"] == 0


def test_v34d_retains_fraction_zero_and_closes_all_authorities():
    value = evidence.build_negative_evidence_v34d()
    assert value["decision"] == {
        "retain_production_at_fraction_0_0": True,
        "replacement_fraction_adoption_authority": False,
        "model_update_authority": False,
        "checkpoint_write_authority": False,
        "validation_heldout_ood_or_benchmark_evaluation_authority": False,
        "dataset_promotion_authority": False,
        "nontrain_reuse_authority": False,
    }
    assert all(item is False for item in value["side_effects"].values())


def test_v34d_scope_is_compact_only_with_no_gpu_or_eval_content_access():
    assert evidence.build_negative_evidence_v34d()["input_scope"] == {
        "compact_attempt_and_report_aggregates_only": True,
        "source_contract_constants_only": True,
        "panel_candidate_manifest_or_dataset_content_opened": False,
        "raw_stdout_or_runtime_logs_opened": False,
        "evaluation_validation_heldout_ood_or_benchmark_content_opened": False,
        "detailed_scores_responses_coefficients_or_bootstrap_draws_persisted": False,
        "gpu_launched": False,
    }


@pytest.mark.parametrize(
    "name,message",
    (
        ("ATTEMPT_PATH_V34D", "attempt file hash changed"),
        ("REPORT_PATH_V34D", "report file hash changed"),
    ),
)
def test_v34d_changed_compact_artifact_bytes_fail_closed(
    monkeypatch, tmp_path, name, message,
):
    source = getattr(evidence, name)
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    monkeypatch.setattr(evidence, name, changed)
    with pytest.raises(RuntimeError, match=message):
        evidence.validate_bound_artifacts_v34d()


def test_v34d_tampered_attempt_self_hash_or_report_binding_fails_closed(
    monkeypatch,
):
    attempt, report = _artifacts()
    changed = copy.deepcopy(attempt)
    changed["recipe_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="attempt self hash changed"):
        evidence._validate_attempt_v34d(changed, report)
    changed = copy.deepcopy(attempt)
    changed["report_binding"]["file_sha256"] = "0" * 64
    changed = evidence._seal(changed)
    monkeypatch.setattr(
        evidence,
        "ATTEMPT_CONTENT_SHA256_V34D",
        changed["content_sha256_before_self_field"],
    )
    with pytest.raises(RuntimeError, match="report binding changed"):
        evidence._validate_attempt_v34d(changed, report)


def test_v34d_tampered_configuration_or_origin_fails_closed(monkeypatch):
    _attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["configuration"]["all_four_tp1_engines"] = False
    changed["configuration"] = evidence._seal(changed["configuration"])
    monkeypatch.setattr(
        evidence,
        "CONFIGURATION_CONTENT_SHA256_V34D",
        changed["configuration"]["content_sha256_before_self_field"],
    )
    with pytest.raises(RuntimeError, match="configuration or origin"):
        evidence._validate_configuration_v34d(changed)


@pytest.mark.parametrize(
    "key,value",
    (
        ("synchronized_signed_wave_count", 31),
        ("total_generation_request_count", 54_599),
        ("all_four_tp1_engines_both_sources_every_wave", False),
    ),
)
def test_v34d_tampered_wave_request_or_activity_fails_closed(
    monkeypatch, key, value,
):
    _attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["preanalysis_runtime_audit"][key] = value
    changed["preanalysis_runtime_audit"] = evidence._seal(
        changed["preanalysis_runtime_audit"]
    )
    monkeypatch.setattr(
        evidence,
        "PREANALYSIS_AUDIT_CONTENT_SHA256_V34D",
        changed["preanalysis_runtime_audit"]["content_sha256_before_self_field"],
    )
    with pytest.raises(RuntimeError, match="32-wave, request, activity"):
        evidence._validate_preanalysis_v34d(changed)


def test_v34d_tampered_a_b_or_a_c_guard_fails_closed(monkeypatch):
    _attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["preanalysis_runtime_audit"]["full_context_guard"]["a_c_exact"][
        "all_dense_result_commitments_exact"
    ] = False
    changed["preanalysis_runtime_audit"] = evidence._seal(
        changed["preanalysis_runtime_audit"]
    )
    monkeypatch.setattr(
        evidence,
        "PREANALYSIS_AUDIT_CONTENT_SHA256_V34D",
        changed["preanalysis_runtime_audit"]["content_sha256_before_self_field"],
    )
    with pytest.raises(RuntimeError, match="A-B-C exact guard"):
        evidence._validate_preanalysis_v34d(changed)


def test_v34d_tampered_5_percent_metric_fails_closed(monkeypatch):
    _attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    analysis = changed["summary"]["fixed_sequence_analysis"]
    analysis["tested_fractions"][0]["endpoints"][
        "train_screen_cosine_median"
    ]["familywise_lcb"] = 0.0
    analysis = evidence._seal(analysis)
    monkeypatch.setattr(
        evidence,
        "FIXED_SEQUENCE_CONTENT_SHA256_V34D",
        analysis["content_sha256_before_self_field"],
    )
    changed["summary"]["fixed_sequence_analysis"] = analysis
    with pytest.raises(RuntimeError, match="exact 5-percent metrics"):
        evidence._validate_fixed_sequence_v34d(changed["summary"])


@pytest.mark.parametrize(
    "key,value",
    (
        ("stopped_at_first_failure", False),
        ("untested_fractions_after_first_failure", [0.1, 0.2, 0.4]),
        ("largest_consecutively_passing_fraction", 0.05),
    ),
)
def test_v34d_tampered_stop_or_higher_fraction_status_fails_closed(
    monkeypatch, key, value,
):
    _attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    analysis = changed["summary"]["fixed_sequence_analysis"]
    analysis[key] = value
    analysis = evidence._seal(analysis)
    monkeypatch.setattr(
        evidence,
        "FIXED_SEQUENCE_CONTENT_SHA256_V34D",
        analysis["content_sha256_before_self_field"],
    )
    changed["summary"]["fixed_sequence_analysis"] = analysis
    with pytest.raises(RuntimeError, match="fixed-sequence stop"):
        evidence._validate_fixed_sequence_v34d(changed["summary"])


def test_v34d_tampered_authority_or_side_effect_fails_closed():
    attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["gate"]["model_update_authorized"] = True
    changed["gate"] = evidence._seal(changed["gate"])
    with pytest.raises(RuntimeError, match="authorization gate self hash changed"):
        evidence._validate_closed_authority_v34d(attempt, changed)
    changed = copy.deepcopy(report)
    changed["nontrain_surface_opened"] = True
    with pytest.raises(RuntimeError, match="forbidden side effect"):
        evidence._validate_closed_authority_v34d(attempt, changed)


@pytest.mark.parametrize(
    "forbidden",
    (
        "questions", "answers", "prompts", "prompt_token_ids", "unit_scores",
        "responses", "coefficients", "bootstrap_replicates", "bootstrap_draws",
        "row_content", "row_sha256", "document_sha256", "unit_ids", "pids",
        "timings", "memory_samples",
    ),
)
def test_v34d_compact_evidence_rejects_detailed_payload_keys(forbidden):
    with pytest.raises(RuntimeError, match="forbidden detailed payload keys"):
        evidence._assert_compact_v34d({forbidden: []})


def test_v34d_build_is_deterministic_and_dry_run_does_not_write(capsys):
    first = evidence.build_negative_evidence_v34d()
    assert first == evidence.build_negative_evidence_v34d()
    assert first["content_sha256_before_self_field"] == evidence.canonical_sha256(
        evidence._without_self(first)
    )
    result = evidence.main(["--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert result == first
    assert output["content_sha256"] == first["content_sha256_before_self_field"]
    assert output["tested_fraction_passed"] is False
    assert output["higher_fractions_inferred"] is False
    assert output["gpu_launched"] is False


def test_v34d_materialized_evidence_is_exact_build_output():
    materialized = json.loads(evidence.OUTPUT_PATH_V34D.read_text(encoding="utf-8"))
    assert materialized == evidence.build_negative_evidence_v34d()


def test_v34d_exclusive_write_is_immutable(monkeypatch, tmp_path):
    output = tmp_path / "negative.json"
    monkeypatch.setattr(evidence, "OUTPUT_PATH_V34D", output)
    value = evidence.build_negative_evidence_v34d()
    evidence._exclusive_write_json_v34d(output, value)
    with pytest.raises(RuntimeError, match="already exists"):
        evidence._exclusive_write_json_v34d(output, value)
