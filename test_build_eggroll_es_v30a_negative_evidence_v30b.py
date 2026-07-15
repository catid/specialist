#!/usr/bin/env python3

import copy
import json

import pytest

import build_eggroll_es_v30a_negative_evidence_v30b as evidence


def _artifacts():
    attempt = json.loads(evidence.ATTEMPT_PATH_V30B.read_text(encoding="utf-8"))
    report = json.loads(evidence.REPORT_PATH_V30B.read_text(encoding="utf-8"))
    return attempt, report


def test_v30b_binds_exact_completed_v30a_artifacts_and_history():
    value = evidence.build_negative_evidence_v30b()
    assert value["artifacts"]["durable_attempt"] == {
        "relative_path": evidence.ATTEMPT_RELATIVE_PATH_V30B,
        "file_sha256": "c0a8b9e4e7b4607fa37eebfde0ae57fc561716ed94e48b25f01fef481f2fe995",
        "content_sha256": "a93d1bd28eff0607c9ce4ff165e8b3ae24428995cc4fbbadc2a99ed32d41cc01",
    }
    assert value["artifacts"]["compact_report"]["file_sha256"] == (
        "784dbeece2ffa3eace520db8083d9f6c6b1d66bfc688257f37de5b38d2d0a5af"
    )
    assert value["artifacts"]["compact_report"]["content_sha256"] == (
        "1073fe301dab129215e2c7c8e8877ecdd9be4fc9c3f71d920a788d2ec2e381b3"
    )
    history = value["source_history"]
    assert history["preregistration_commit"].startswith("c54cbf4")
    assert history["hardened_implementation_commit"].startswith("5448c17")
    assert history["clean_launch_source_commit"].startswith("a203f48")
    assert history["candidate_input_freeze_commit"].startswith("2e8a6b7")


def test_v30b_recomputes_exact_12_endpoint_negative_gate():
    value = evidence.build_negative_evidence_v30b()
    assert value["endpoint_gate_aggregates"] == evidence.EXPECTED_ENDPOINTS_V30B
    result = value["aggregate_result"]
    assert result == {
        "endpoint_count": 12,
        "negative_point_estimate_count": 7,
        "zero_point_estimate_count": 2,
        "positive_point_estimate_count": 3,
        "negative_familywise_lcb_count": 12,
        "best_familywise_lcb": -0.125,
        "worst_familywise_lcb": -0.44378228533250474,
        "all_familywise_lcbs_nonnegative": False,
        "gate_pass": False,
    }


def test_v30b_retains_production_v13_and_grants_no_followup_authority():
    value = evidence.build_negative_evidence_v30b()
    assert value["decision"] == {
        "retain_production_dataset": True,
        "retain_v13_recipe": True,
        "v389_replacement_authorized": False,
        "direct_followup_authority": False,
    }
    assert all(item is False for item in value["side_effects"].values())
    assert value["aggregate_execution"][
        "all_runtime_restore_boundary_preclaim_and_final_idle_integrity_passed"
    ] is True


@pytest.mark.parametrize(
    "path_name,message",
    (
        ("ATTEMPT_PATH_V30B", "attempt file hash changed"),
        ("REPORT_PATH_V30B", "report file hash changed"),
    ),
)
def test_v30b_changed_artifact_bytes_fail_closed(
    monkeypatch, tmp_path, path_name, message,
):
    source = getattr(evidence, path_name)
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    monkeypatch.setattr(evidence, path_name, changed)
    with pytest.raises(RuntimeError, match=message):
        evidence.validate_bound_artifacts_v30b()


def test_v30b_tampered_endpoint_or_gate_fails_closed():
    _attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["summary"]["paired_bootstrap"]["endpoints"][
        "train_screen_cosine_median"
    ]["familywise_lcb"] = 0.0
    with pytest.raises(RuntimeError, match="exact 12 endpoint aggregates"):
        evidence._validate_gate_v30b(changed, changed["summary"])
    changed = copy.deepcopy(report)
    changed["gate"]["pass"] = True
    changed["gate"] = evidence._seal(changed["gate"])
    with pytest.raises(RuntimeError, match="gate self hash changed"):
        evidence._validate_gate_v30b(changed, changed["summary"])


def test_v30b_tampered_cardinality_restore_or_boundary_fails_closed():
    attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["recipe"]["request_accounting"]["total_generation_requests"] += 1
    changed["recipe"] = evidence._seal(changed["recipe"])
    with pytest.raises(RuntimeError, match="cardinality changed"):
        evidence._validate_runtime_and_cardinality_v30b(attempt, changed)
    changed = copy.deepcopy(report)
    changed["runtime_audit"]["restore_checks_sha256"] = "0" * 64
    changed["runtime_audit"] = evidence._seal(changed["runtime_audit"])
    with pytest.raises(RuntimeError, match="runtime audit self hash changed"):
        evidence._validate_runtime_and_cardinality_v30b(attempt, changed)


def test_v30b_tampered_side_effect_or_idle_binding_fails_closed():
    attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["checkpoint_written"] = True
    with pytest.raises(RuntimeError, match="forbidden mutation"):
        evidence._validate_closed_side_effects_v30b(attempt, changed)
    changed = copy.deepcopy(attempt)
    changed["final_idle_certificate_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="final-idle integrity"):
        evidence._validate_closed_side_effects_v30b(changed, report)


def test_v30b_tampered_source_history_or_implementation_fails_closed(monkeypatch):
    monkeypatch.setattr(
        evidence, "RUNTIME_FILE_SHA256_V30B", "0" * 64,
    )
    with pytest.raises(RuntimeError, match="runtime history changed"):
        evidence.validate_history_v30b()
    attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["implementation"]["bundle_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="implementation binding changed"):
        evidence._validate_source_and_implementation_v30b(attempt, changed)


@pytest.mark.parametrize(
    "forbidden",
    (
        "questions", "answers", "prompt", "token", "responses", "rows",
        "units", "paired_bootstrap", "heldout", "validation", "ood", "benchmark",
    ),
)
def test_v30b_compact_evidence_rejects_record_or_nontrain_detail_terms(forbidden):
    with pytest.raises(RuntimeError, match="forbidden detail terms"):
        evidence._assert_compact_v30b({forbidden: "not allowed"})


def test_v30b_build_is_deterministic_and_dry_run_does_not_write(capsys):
    first = evidence.build_negative_evidence_v30b()
    second = evidence.build_negative_evidence_v30b()
    assert first == second
    value = evidence.main(["--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert value == first
    assert output["content_sha256"] == first["content_sha256_before_self_field"]
    assert output["gate_pass"] is False


def test_v30b_materialized_evidence_is_exact_immutable_build_output():
    materialized = json.loads(evidence.OUTPUT_PATH_V30B.read_text(encoding="utf-8"))
    assert materialized == evidence.build_negative_evidence_v30b()
    assert evidence.file_sha256(evidence.OUTPUT_PATH_V30B) == (
        "a39dd26b2b95ce84258f1ed1b199af29456314847369db25084f15829e7d6a91"
    )
    assert materialized["content_sha256_before_self_field"] == (
        "e1466718df3baf44268ea099bd507e12e8023807917610050d9e161774b45cfd"
    )


def test_v30b_exclusive_evidence_write_is_immutable(monkeypatch, tmp_path):
    output = tmp_path / "negative.json"
    monkeypatch.setattr(evidence, "OUTPUT_PATH_V30B", output)
    value = evidence.build_negative_evidence_v30b()
    evidence._exclusive_write_json_v30b(output, value)
    with pytest.raises(RuntimeError, match="already exists"):
        evidence._exclusive_write_json_v30b(output, value)
