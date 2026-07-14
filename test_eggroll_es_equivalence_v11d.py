import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import run_eggroll_es_anchor_equivalence_v11c as driver_v11c
import run_eggroll_es_anchor_equivalence_v11d as driver_v11d
import report_eggroll_es_equivalence_v11d as report_v11d


def cli(dry=True):
    values = list(driver_v11d.FROZEN_REAL_ARGV_V11D)
    if dry:
        values.append("--v11c-dry-run")
    return values


def test_v11d_binds_exact_v11c_failure_and_implementation():
    binding = driver_v11d.bind_v11c_failure_v11d()
    assert binding["failure_phase"] == "post_engine_load_pre_journal_unknown"
    assert binding["model_update_applied"] is False
    assert binding["sealed_data_opened_or_scored"] is False
    assert binding["v11c_implementation"] == (
        driver_v11d.V11C_IMPLEMENTATION_SHA256_V11D
    )


def test_v11d_dry_run_has_new_recipe_and_restores_v11c_globals(capsys):
    before = (
        driver_v11c.EXPERIMENT_NAME_V11C,
        driver_v11c.EXPECTED_RECIPE_SHA256_V11C,
    )
    result = driver_v11d.main(cli())
    assert result["schema"] == "eggroll-es-durable-launch-dry-run-v11d"
    assert result["recipe_sha256"] == (
        driver_v11d.EXPECTED_V11C_RECIPE_SHA256_V11D
    )
    assert (
        result["v11c_failure_binding_sha256"]
        == driver_v11d.bind_v11c_failure_v11d()["binding_sha256"]
    )
    assert before == (
        driver_v11c.EXPERIMENT_NAME_V11C,
        driver_v11c.EXPECTED_RECIPE_SHA256_V11C,
    )
    assert "durable-launch-dry-run-v11d" in capsys.readouterr().out


def test_v11d_rejects_changed_name_or_failure(tmp_path):
    changed = cli()
    changed[changed.index("--experiment-name") + 1] = "changed"
    with pytest.raises(ValueError, match="exact frozen"):
        driver_v11d.main(changed)
    bad = tmp_path / "failure.md"
    bad.write_text("changed\n")
    with pytest.raises(RuntimeError, match="exact committed"):
        driver_v11d.bind_v11c_failure_v11d(bad)


def test_v11d_failure_telemetry_is_atomic_and_reraises(
    tmp_path, monkeypatch,
):
    attempt = tmp_path / "attempt.json"
    frozen_output = driver_v11c.driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11
    run_dir = frozen_output / driver_v11d.EXPERIMENT_NAME_V11D
    assert not run_dir.exists()

    def fail(_argv):
        raise RuntimeError("synthetic post-engine launch failure")

    monkeypatch.setattr(driver_v11c, "main", fail)
    monkeypatch.setattr(driver_v11d, "_source_provenance_v11d", lambda: {
        "schema": "synthetic-test-source-provenance"
    })
    with pytest.raises(RuntimeError, match="synthetic post-engine launch failure"):
        driver_v11d.run_exact_retry_v11d(
            cli(dry=False), attempt,
            driver_v11d.bind_v11c_failure_v11d(),
        )
    payload = json.loads(attempt.read_text())
    assert payload["status"] == "failed"
    assert payload["failure"]["type"] == "RuntimeError"
    assert "synthetic post-engine launch failure" in payload["failure"]["traceback"]
    assert payload["model_update_applied"] is False
    assert payload["sealed_data_opened_or_scored"] is False
    assert payload["content_sha256_before_self_field"] == (
        driver_v11c.driver_v1.canonical_sha256({
            key: value for key, value in payload.items()
            if key != "content_sha256_before_self_field"
        })
    )


def test_v11d_attempt_is_single_use(tmp_path):
    attempt = tmp_path / "attempt.json"
    attempt.write_text("{}\n")
    with pytest.raises(ValueError, match="already exists"):
        driver_v11d.run_exact_retry_v11d(
            cli(dry=False), attempt,
            driver_v11d.bind_v11c_failure_v11d(),
        )


def test_v11d_offline_validator_scopes_retry_globals(monkeypatch):
    seen = {}

    def validate(journal):
        seen["journal"] = journal
        seen["name"] = driver_v11c.EXPERIMENT_NAME_V11C
        seen["recipe"] = driver_v11c.EXPECTED_RECIPE_SHA256_V11C
        return {"equivalence": {"all_exact": True}}

    before = (
        driver_v11c.EXPERIMENT_NAME_V11C,
        driver_v11c.EXPECTED_RECIPE_SHA256_V11C,
    )
    monkeypatch.setattr(driver_v11c, "validate_completed_journal_v11c", validate)
    result = driver_v11d.validate_completed_journal_v11d({"journal": True})
    assert result["equivalence"]["all_exact"] is True
    assert seen == {
        "journal": {"journal": True},
        "name": driver_v11d.EXPERIMENT_NAME_V11D,
        "recipe": driver_v11d.EXPECTED_V11C_RECIPE_SHA256_V11D,
    }
    assert before == (
        driver_v11c.EXPERIMENT_NAME_V11C,
        driver_v11c.EXPECTED_RECIPE_SHA256_V11C,
    )


def test_v11d_journal_binding_accepts_schema_heldout_false_sentinel(
    tmp_path, monkeypatch,
):
    driver_v11 = driver_v11c.driver_v11b.driver_v11
    monkeypatch.setattr(driver_v11, "FROZEN_OUTPUT_DIRECTORY_V11", tmp_path)
    run_dir = tmp_path / driver_v11d.EXPERIMENT_NAME_V11D
    run_dir.mkdir()
    journal_path = run_dir / driver_v11c.driver_v1.JOURNAL_NAME
    journal = {
        "schema": "eggroll-es-anchor-alpha-line-search-v11c",
        "snapshot": {
            "document_lcb_anchor_v5": {
                "ood_validation_heldout_as_objective": False,
            },
        },
    }
    journal_path.write_text(json.dumps(journal) + "\n")
    monkeypatch.setattr(
        driver_v11d,
        "validate_completed_journal_v11d",
        lambda loaded: {
            "content_sha256": "a" * 64,
            "sentinel": loaded["snapshot"]["document_lcb_anchor_v5"][
                "ood_validation_heldout_as_objective"
            ],
        },
    )
    binding = driver_v11d._build_journal_binding_v11d(journal_path)
    assert binding["content_sha256"] == "a" * 64
    assert binding["path"] == str(journal_path.resolve())


def test_v11d_reporter_scopes_globals_and_binds_launch_attempt(
    tmp_path, monkeypatch,
):
    journal = tmp_path / "journal.json"
    attempt = tmp_path / "attempt.json"
    journal.write_text("{}\n")
    attempt.write_text("{}\n")
    offline = driver_v11c.driver_v11b.driver_v11.driver_v8.offline_audit
    monkeypatch.setattr(offline, "_assert_no_heldout", lambda *_args: None)
    monkeypatch.setattr(
        driver_v11d, "validate_completed_journal_v11d",
        lambda _journal: {"content_sha256": "journal-content"},
    )
    monkeypatch.setattr(
        driver_v11d, "validate_launch_attempt_v11d",
        lambda _attempt, _journal: {
            "content_sha256": "attempt-content",
            "v11c_failure_binding_sha256": "failure-binding",
        },
    )

    def base_report(_path):
        assert driver_v11c.EXPERIMENT_NAME_V11C == driver_v11d.EXPERIMENT_NAME_V11D
        assert driver_v11c.EXPECTED_RECIPE_SHA256_V11C == (
            driver_v11d.EXPECTED_V11C_RECIPE_SHA256_V11D
        )
        return {
            "schema": "v11c", "metric": "v11c", "passed": True,
            "content_sha256_before_self_field": "old",
        }

    monkeypatch.setattr(report_v11d.report_v11c, "build_report", base_report)
    monkeypatch.setattr(
        driver_v11d, "_attempt_path_v11d", lambda _runtime: attempt,
    )
    report = report_v11d.build_report(journal, attempt)
    assert report["schema"] == "eggroll-es-resident-sign-equivalence-report-v11d"
    assert report["passed"] is True
    assert report["v11d"]["launch_attempt_content_sha256"] == "attempt-content"
    assert report["v11d"]["journal_content_sha256"] == "journal-content"
    moved = tmp_path / "moved-attempt.json"
    moved.write_text(attempt.read_text())
    with pytest.raises(RuntimeError, match="launch-attempt path"):
        report_v11d.build_report(journal, moved)


def _completed_attempt_for_validation():
    run_dir = (
        driver_v11c.driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11
        / driver_v11d.EXPERIMENT_NAME_V11D
    ).resolve()
    journal_path = (run_dir / driver_v11c.driver_v1.JOURNAL_NAME).resolve()
    journal_binding = {
        "schema": "eggroll-es-v11d-journal-binding",
        "path": str(journal_path),
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "journal_schema": "eggroll-es-anchor-alpha-line-search-v11c",
    }
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v11d",
        "status": "complete",
        "phase": "after_v11c_driver_main",
        "experiment_name": driver_v11d.EXPERIMENT_NAME_V11D,
        "run_directory": str(run_dir),
        "run_directory_absent_before_attempt": True,
        "argv_sha256": driver_v11c.driver_v1.canonical_sha256(
            list(driver_v11d.FROZEN_REAL_ARGV_V11D)
        ),
        "source_provenance": {
            "schema": "eggroll-es-v11d-committed-source-provenance",
            "repository_root": str(driver_v11d.ROOT),
            "relative_path": "run_eggroll_es_anchor_equivalence_v11d.py",
            "git_head": "c" * 40,
            "committed_blob_sha256": "d" * 64,
            "driver_file_sha256": "d" * 64,
        },
        "v11c_failure_evidence": driver_v11d.bind_v11c_failure_v11d(),
        "v11c_recipe_sha256": driver_v11d.EXPECTED_V11C_RECIPE_SHA256_V11D,
        "diagnostic_environment": dict(driver_v11d.DIAGNOSTIC_ENV_V11D),
        "algorithm_or_data_changed_from_v11c": False,
        "target_alpha_zero_only": True,
        "model_update_applied": False,
        "sealed_data_opened_or_scored": False,
        "run_directory_exists_after_attempt": True,
        "v11c_journal_exists_after_attempt": True,
        "journal_binding": journal_binding,
    }
    _reseal_attempt(attempt)
    return attempt, journal_path, journal_binding


def _reseal_attempt(attempt):
    attempt.pop("content_sha256_before_self_field", None)
    attempt["content_sha256_before_self_field"] = (
        driver_v11c.driver_v1.canonical_sha256(attempt)
    )


def test_v11d_completed_attempt_rejects_extra_heldout_and_wrong_identity(
    monkeypatch,
):
    attempt, journal_path, journal_binding = _completed_attempt_for_validation()
    monkeypatch.setattr(
        driver_v11d, "validate_source_provenance_v11d", lambda _value: {},
    )
    monkeypatch.setattr(
        driver_v11d, "_build_journal_binding_v11d",
        lambda _path: dict(journal_binding),
    )
    assert driver_v11d.validate_launch_attempt_v11d(
        attempt, journal_path,
    )["content_sha256"] == attempt["content_sha256_before_self_field"]

    extra = json.loads(json.dumps(attempt))
    extra["extra"] = "content"
    _reseal_attempt(extra)
    with pytest.raises(RuntimeError, match="evidence changed"):
        driver_v11d.validate_launch_attempt_v11d(extra, journal_path)

    leaked = json.loads(json.dumps(attempt))
    leaked["secret_heldout_payload"] = "forbidden"
    _reseal_attempt(leaked)
    with pytest.raises(Exception, match="heldout"):
        driver_v11d.validate_launch_attempt_v11d(leaked, journal_path)

    leaked_value = json.loads(json.dumps(attempt))
    leaked_value["source_provenance"]["relative_path"] = (
        "nested/heldout/payload.json"
    )
    _reseal_attempt(leaked_value)
    with pytest.raises(Exception, match="heldout"):
        driver_v11d.validate_launch_attempt_v11d(
            leaked_value, journal_path,
        )

    for field, value in (
        ("run_directory", "/tmp/wrong"),
        ("argv_sha256", "0" * 64),
    ):
        wrong = json.loads(json.dumps(attempt))
        wrong[field] = value
        _reseal_attempt(wrong)
        with pytest.raises(RuntimeError, match="evidence changed"):
            driver_v11d.validate_launch_attempt_v11d(wrong, journal_path)


def test_v11d_attempt_journal_binding_and_report_path_are_strict(monkeypatch):
    attempt, journal_path, journal_binding = _completed_attempt_for_validation()
    monkeypatch.setattr(
        driver_v11d, "validate_source_provenance_v11d", lambda _value: {},
    )
    monkeypatch.setattr(
        driver_v11d, "_build_journal_binding_v11d",
        lambda _path: dict(journal_binding),
    )
    changed = json.loads(json.dumps(attempt))
    changed["journal_binding"]["file_sha256"] = "0" * 64
    _reseal_attempt(changed)
    with pytest.raises(RuntimeError, match="cryptographic binding"):
        driver_v11d.validate_launch_attempt_v11d(changed, journal_path)
    changed_content = json.loads(json.dumps(attempt))
    changed_content["journal_binding"]["content_sha256"] = "1" * 64
    _reseal_attempt(changed_content)
    with pytest.raises(RuntimeError, match="cryptographic binding"):
        driver_v11d.validate_launch_attempt_v11d(
            changed_content, journal_path,
        )
    with pytest.raises(RuntimeError, match="reporter journal path"):
        driver_v11d.validate_launch_attempt_v11d(
            attempt, journal_path.with_name("different.json"),
        )


def test_v11d_exclusive_reservation_rejects_second_claim(tmp_path):
    path = tmp_path / "attempt.json"
    barrier = threading.Barrier(2)

    def claim(value):
        barrier.wait()
        try:
            driver_v11d._exclusive_write_attempt_v11d(
                path, {"claim": value},
            )
        except ValueError as error:
            return type(error).__name__
        return "success"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, (1, 2)))
    assert sorted(results) == ["ValueError", "success"]


def test_v11d_source_provenance_rejects_extra_key_before_git_access():
    provenance = {
        "schema": "eggroll-es-v11d-committed-source-provenance",
        "repository_root": str(driver_v11d.ROOT),
        "relative_path": "run_eggroll_es_anchor_equivalence_v11d.py",
        "git_head": "a" * 40,
        "committed_blob_sha256": "b" * 64,
        "driver_file_sha256": "b" * 64,
        "extra": True,
    }
    with pytest.raises(RuntimeError, match="shape changed"):
        driver_v11d.validate_source_provenance_v11d(provenance)
