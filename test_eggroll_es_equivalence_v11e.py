import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import report_eggroll_es_equivalence_v11e as report_v11e
import run_eggroll_es_anchor_equivalence_v11c as driver_v11c
import run_eggroll_es_anchor_equivalence_v11e as driver_v11e


def cli(dry=True):
    values = list(driver_v11e.FROZEN_REAL_ARGV_V11E)
    if dry:
        values.append("--v11c-dry-run")
    return values


@pytest.fixture(scope="module")
def effective_audit():
    return driver_v11e.audit_effective_downstream_cli_v11e(
        driver_v11e.FROZEN_REAL_ARGV_V11E
    )


def test_v11e_binds_exact_v11d_failure_document_attempt_and_source():
    binding = driver_v11e.bind_v11d_failure_v11e()
    assert driver_v11e.V11D_FAILURE_EVIDENCE_PATH_V11E.parent == driver_v11e.ROOT
    assert driver_v11e.V11D_FAILURE_EVIDENCE_PATH_V11E.name == (
        "S6_RESIDENT_SIGN_EQUIVALENCE_V11D_FAILURE_EVIDENCE_V11E.json"
    )
    assert binding["failure_message"] == "v5 requires every frozen anchor document"
    assert binding["file_sha256"] == driver_v11e.V11D_FAILURE_EVIDENCE_SHA256_V11E
    assert binding["content_sha256"] == driver_v11e.V11D_FAILURE_CONTENT_SHA256_V11E
    assert binding["failure_document_sha256"] == (
        driver_v11e.V11D_FAILURE_DOCUMENT_SHA256_V11E
    )
    assert binding["failure_document_commit"] == (
        driver_v11e.V11D_FAILURE_DOCUMENT_COMMIT_V11E
    )
    assert binding["source_driver_sha256"] == driver_v11e.V11D_SOURCE_SHA256_V11E
    assert binding["model_update_applied"] is False
    assert binding["sealed_data_opened_or_scored"] is False


def test_v11e_full_effective_projection_is_exact(effective_audit):
    assert effective_audit["field_count"] == 27
    assert effective_audit["mismatch_fields"] == []
    assert effective_audit["outer"] == driver_v11e.CANONICAL_DOWNSTREAM_V11E
    assert effective_audit["effective"] == driver_v11e.CANONICAL_DOWNSTREAM_V11E
    assert effective_audit["effective"]["anchor_items_per_step"] == 128
    assert effective_audit["effective"]["min_anchor_cosine"] == 0.8
    assert effective_audit["base_argv_sha256"] == driver_v11e.BASE_ARGV_SHA256_V11E


def test_v11e_rehashes_exact_v11d_pinned_runtime_bundle(monkeypatch):
    audit = driver_v11e.audit_v11c_implementation_v11e()
    assert set(audit) == driver_v11e.IMPLEMENTATION_AUDIT_KEYS_V11E
    assert audit["file_sha256"] == driver_v11e.V11C_IMPLEMENTATION_SHA256_V11E
    original = driver_v11e._file_sha256
    worker = driver_v11e.V11C_IMPLEMENTATION_PATHS_V11E["worker"]
    monkeypatch.setattr(
        driver_v11e, "_file_sha256",
        lambda path: "0" * 64 if path == worker else original(path),
    )
    with pytest.raises(RuntimeError, match="implementation bundle changed"):
        driver_v11e.audit_v11c_implementation_v11e()


def test_v11e_regression_omitted_forwarding_reproduces_only_two_mismatches():
    omitted = list(driver_v11e.FROZEN_REAL_ARGV_V11E)
    for flag in ("--anchor-items-per-step", "--min-anchor-cosine"):
        index = omitted.index(flag)
        del omitted[index:index + 2]
    projection = driver_v11e.inspect_downstream_projection_v11e(omitted)
    assert projection["field_count"] == 27
    assert projection["mismatch_fields"] == [
        "anchor_items_per_step", "min_anchor_cosine",
    ]
    assert projection["outer"]["anchor_items_per_step"] == 128
    assert projection["effective"]["anchor_items_per_step"] == 2
    assert projection["outer"]["min_anchor_cosine"] == 0.8
    assert projection["effective"]["min_anchor_cosine"] == 0.1
    with pytest.raises(RuntimeError, match="effective downstream runtime projection"):
        driver_v11e.audit_effective_downstream_cli_v11e(omitted)
    with pytest.raises(ValueError, match="exact frozen"):
        driver_v11e.main([*omitted, "--v11c-dry-run"])


def test_v11e_dry_run_proves_effective_delegated_values(capsys):
    result = driver_v11e.main(cli())
    assert result["schema"] == "eggroll-es-forwarded-anchor-dry-run-v11e"
    assert result["recipe_sha256"] == driver_v11e.EXPECTED_V11C_RECIPE_SHA256_V11E
    audit = result["effective_downstream_cli"]
    assert audit["field_count"] == 27
    assert audit["mismatch_fields"] == []
    assert audit["effective"]["anchor_items_per_step"] == 128
    assert audit["effective"]["anchor_max_input_tokens"] == 512
    assert audit["effective"]["min_anchor_cosine"] == 0.8
    assert audit["effective"]["ood_prose_max_input_tokens"] == 1024
    assert result["v11c_implementation"]["file_sha256"] == (
        driver_v11e.V11C_IMPLEMENTATION_SHA256_V11E
    )
    assert result["frozen_recipe_or_data_changed_from_v11d"] is False
    assert result["effective_cli_forwarding_corrected"] is True
    assert result["effective_cli_correction"] == (
        driver_v11e.EFFECTIVE_CLI_CORRECTION_V11E
    )
    assert "forwarded-anchor-dry-run-v11e" in capsys.readouterr().out


def test_v11e_cli_and_source_provenance_fail_closed_before_launch():
    changed = cli()
    changed[changed.index("--anchor-items-per-step") + 1] = "2"
    with pytest.raises(ValueError, match="exact frozen"):
        driver_v11e.main(changed)
    provenance = {
        "schema": "eggroll-es-v11e-committed-source-provenance",
        "repository_root": str(driver_v11e.ROOT),
        "relative_path": "run_eggroll_es_anchor_equivalence_v11e.py",
        "git_head": "a" * 40,
        "committed_blob_sha256": "b" * 64,
        "driver_file_sha256": "b" * 64,
        "extra": True,
    }
    with pytest.raises(RuntimeError, match="shape changed"):
        driver_v11e.validate_source_provenance_v11e(provenance)


def test_v11e_failure_telemetry_is_atomic_full_and_reraised(
    tmp_path, monkeypatch, effective_audit,
):
    attempt = tmp_path / "attempt.json"

    def fail(_argv):
        raise RuntimeError("synthetic V11e post-claim failure")

    monkeypatch.setattr(driver_v11c, "main", fail)
    monkeypatch.setattr(driver_v11e, "_source_provenance_v11e", lambda: {
        "schema": "synthetic-source-provenance",
    })
    with pytest.raises(RuntimeError, match="synthetic V11e post-claim failure"):
        driver_v11e.run_exact_retry_v11e(
            cli(dry=False), attempt, driver_v11e.bind_v11d_failure_v11e(),
            effective_audit,
        )
    payload = json.loads(attempt.read_text())
    assert payload["status"] == "failed"
    assert payload["failure"]["type"] == "RuntimeError"
    assert "synthetic V11e post-claim failure" in payload["failure"]["traceback"]
    assert payload["model_update_applied"] is False
    assert payload["sealed_data_opened_or_scored"] is False
    assert payload["frozen_recipe_or_data_changed_from_v11d"] is False
    assert payload["effective_cli_forwarding_corrected"] is True
    assert payload["effective_cli_correction"] == (
        driver_v11e.EFFECTIVE_CLI_CORRECTION_V11E
    )
    assert payload["content_sha256_before_self_field"] == driver_v11e._canonical({
        key: value for key, value in payload.items()
        if key != "content_sha256_before_self_field"
    })


def test_v11e_exclusive_sibling_reservation_has_one_winner(tmp_path):
    path = tmp_path / "attempt.json"
    barrier = threading.Barrier(2)

    def claim(value):
        barrier.wait()
        try:
            driver_v11e._exclusive_write_attempt_v11e(path, {"claim": value})
        except ValueError:
            return "rejected"
        return "success"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, (1, 2)))
    assert sorted(results) == ["rejected", "success"]


def test_v11e_inherited_journal_false_sentinel_uses_schema_validator(
    tmp_path, monkeypatch,
):
    driver_v11 = driver_v11c.driver_v11b.driver_v11
    monkeypatch.setattr(driver_v11, "FROZEN_OUTPUT_DIRECTORY_V11", tmp_path)
    run_dir = tmp_path / driver_v11e.EXPERIMENT_NAME_V11E
    run_dir.mkdir()
    journal_path = run_dir / driver_v11c.driver_v1.JOURNAL_NAME
    journal = {
        "schema": "eggroll-es-anchor-alpha-line-search-v11c",
        "snapshot": {"document_lcb_anchor_v5": {
            "ood_validation_heldout_as_objective": False,
        }},
    }
    journal_path.write_text(json.dumps(journal) + "\n")
    monkeypatch.setattr(
        driver_v11e, "validate_completed_journal_v11e",
        lambda loaded: {
            "content_sha256": "a" * 64,
            "sentinel": loaded["snapshot"]["document_lcb_anchor_v5"][
                "ood_validation_heldout_as_objective"
            ],
        },
    )
    binding = driver_v11e._build_journal_binding_v11e(journal_path)
    assert binding["content_sha256"] == "a" * 64


def _completed_attempt(effective_audit):
    run_dir = (
        driver_v11c.driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11
        / driver_v11e.EXPERIMENT_NAME_V11E
    ).resolve()
    journal_path = (run_dir / driver_v11c.driver_v1.JOURNAL_NAME).resolve()
    journal_binding = {
        "schema": "eggroll-es-v11e-journal-binding",
        "path": str(journal_path),
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "journal_schema": "eggroll-es-anchor-alpha-line-search-v11c",
    }
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v11e",
        "status": "complete",
        "phase": "after_v11c_driver_main",
        "experiment_name": driver_v11e.EXPERIMENT_NAME_V11E,
        "run_directory": str(run_dir),
        "run_directory_absent_before_attempt": True,
        "argv_sha256": driver_v11e._canonical(list(driver_v11e.FROZEN_REAL_ARGV_V11E)),
        "source_provenance": {
            "schema": "synthetic-provenance",
        },
        "v11d_failure_evidence": driver_v11e.bind_v11d_failure_v11e(),
        "v11c_recipe_sha256": driver_v11e.EXPECTED_V11C_RECIPE_SHA256_V11E,
        "v11c_implementation": driver_v11e.audit_v11c_implementation_v11e(),
        "effective_downstream_cli": effective_audit,
        "diagnostic_environment": dict(driver_v11e.DIAGNOSTIC_ENV_V11E),
        "frozen_recipe_or_data_changed_from_v11d": False,
        "effective_cli_forwarding_corrected": True,
        "effective_cli_correction": dict(
            driver_v11e.EFFECTIVE_CLI_CORRECTION_V11E
        ),
        "target_alpha_zero_only": True,
        "model_update_applied": False,
        "sealed_data_opened_or_scored": False,
        "run_directory_exists_after_attempt": True,
        "v11c_journal_exists_after_attempt": True,
        "journal_binding": journal_binding,
    }
    _reseal(attempt)
    return attempt, journal_path, journal_binding


def _reseal(attempt):
    attempt.pop("content_sha256_before_self_field", None)
    attempt["content_sha256_before_self_field"] = driver_v11e._canonical(attempt)


def test_v11e_completed_evidence_rejects_leakage_and_binding_changes(
    monkeypatch, effective_audit,
):
    attempt, journal_path, journal_binding = _completed_attempt(effective_audit)
    monkeypatch.setattr(driver_v11e, "validate_source_provenance_v11e", lambda _value: {})
    monkeypatch.setattr(driver_v11e, "_build_journal_binding_v11e", lambda _path: dict(journal_binding))
    fresh_calls = []
    real_effective_audit = driver_v11e.audit_effective_downstream_cli_v11e

    def fresh_effective(argv):
        fresh_calls.append(tuple(argv))
        return real_effective_audit(argv)

    monkeypatch.setattr(
        driver_v11e, "audit_effective_downstream_cli_v11e", fresh_effective,
    )
    assert driver_v11e.validate_launch_attempt_v11e(attempt, journal_path)
    assert fresh_calls == [tuple(driver_v11e.FROZEN_REAL_ARGV_V11E)]

    leaked = json.loads(json.dumps(attempt))
    leaked["secret_heldout_payload"] = "forbidden"
    _reseal(leaked)
    with pytest.raises(Exception, match="heldout"):
        driver_v11e.validate_launch_attempt_v11e(leaked, journal_path)

    changed = json.loads(json.dumps(attempt))
    changed["effective_downstream_cli"]["effective"]["min_anchor_cosine"] = 0.1
    changed["effective_downstream_cli"].pop("content_sha256_before_self_field")
    changed["effective_downstream_cli"]["content_sha256_before_self_field"] = (
        driver_v11e._canonical(changed["effective_downstream_cli"])
    )
    _reseal(changed)
    with pytest.raises(RuntimeError, match="evidence changed"):
        driver_v11e.validate_launch_attempt_v11e(changed, journal_path)

    changed_binding = json.loads(json.dumps(attempt))
    changed_binding["journal_binding"]["file_sha256"] = "0" * 64
    _reseal(changed_binding)
    with pytest.raises(RuntimeError, match="cryptographic binding"):
        driver_v11e.validate_launch_attempt_v11e(changed_binding, journal_path)


def test_v11e_reporter_scopes_v11c_and_requires_exact_sibling(
    tmp_path, monkeypatch,
):
    journal = tmp_path / "journal.json"
    attempt = tmp_path / "attempt.json"
    journal.write_text("{}\n")
    attempt.write_text("{}\n")
    offline = driver_v11e._offline_audit_v11e()
    monkeypatch.setattr(offline, "_assert_no_heldout", lambda *_args: None)
    monkeypatch.setattr(
        driver_v11e, "validate_completed_journal_v11e",
        lambda _journal: {"content_sha256": "journal-content"},
    )
    monkeypatch.setattr(
        driver_v11e, "validate_launch_attempt_v11e",
        lambda _attempt, _journal: {
            "content_sha256": "attempt-content",
            "v11d_failure_binding_sha256": "failure-binding",
            "effective_downstream_cli_sha256": "effective-binding",
            "v11c_implementation_bundle_sha256": "implementation-binding",
        },
    )

    def base_report(_path):
        assert driver_v11c.EXPERIMENT_NAME_V11C == driver_v11e.EXPERIMENT_NAME_V11E
        assert driver_v11c.EXPECTED_RECIPE_SHA256_V11C == (
            driver_v11e.EXPECTED_V11C_RECIPE_SHA256_V11E
        )
        return {"schema": "v11c", "metric": "v11c", "passed": True,
                "content_sha256_before_self_field": "old"}

    monkeypatch.setattr(report_v11e.report_v11c, "build_report", base_report)
    monkeypatch.setattr(driver_v11e, "_attempt_path_v11e", lambda _runtime: attempt)
    report = report_v11e.build_report(journal, attempt)
    assert report["schema"] == "eggroll-es-resident-sign-equivalence-report-v11e"
    assert report["v11e"]["effective_downstream_cli_sha256"] == "effective-binding"
    assert report["v11e"]["v11c_implementation_bundle_sha256"] == (
        "implementation-binding"
    )
    assert report["v11e"]["effective_cli_forwarding_corrected"] is True
    moved = tmp_path / "moved.json"
    moved.write_text("{}\n")
    with pytest.raises(RuntimeError, match="launch-attempt path"):
        report_v11e.build_report(journal, moved)


def test_v11e_offline_journal_validator_scopes_and_restores(monkeypatch):
    seen = {}

    def validate(journal):
        seen.update({
            "journal": journal,
            "name": driver_v11c.EXPERIMENT_NAME_V11C,
            "recipe": driver_v11c.EXPECTED_RECIPE_SHA256_V11C,
        })
        return {"equivalence": {"all_exact": True}}

    before = (driver_v11c.EXPERIMENT_NAME_V11C, driver_v11c.EXPECTED_RECIPE_SHA256_V11C)
    monkeypatch.setattr(driver_v11c, "validate_completed_journal_v11c", validate)
    assert driver_v11e.validate_completed_journal_v11e({"journal": True})[
        "equivalence"
    ]["all_exact"] is True
    assert seen["name"] == driver_v11e.EXPERIMENT_NAME_V11E
    assert seen["recipe"] == driver_v11e.EXPECTED_V11C_RECIPE_SHA256_V11E
    assert before == (driver_v11c.EXPERIMENT_NAME_V11C, driver_v11c.EXPECTED_RECIPE_SHA256_V11C)
