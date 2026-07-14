import json
from pathlib import Path

import pytest

import run_eggroll_es_anchor_equivalence_v11c as driver_v11c
import run_eggroll_es_anchor_equivalence_v11d as driver_v11d
import report_eggroll_es_equivalence_v11d as report_v11d


def cli(dry=True):
    driver_v11 = driver_v11c.driver_v11b.driver_v11
    spec = driver_v11c.anchor_v11c.anchor_v11b.anchor_v11.FROZEN_STABILITY_PLANS_V11[
        driver_v11.MIDDLE_LATE_PLAN_SHA256_V11
    ]
    values = [
        "--layer-plan-json", str(spec["path"]),
        "--expected-layer-plan-file-sha256", spec["file_sha256"],
        "--expected-layer-plan-sha256", driver_v11.MIDDLE_LATE_PLAN_SHA256_V11,
        "--expected-model-config-sha256",
        driver_v11c.anchor_v11c.anchor_v11b.anchor_v11.MODEL_CONFIG_SHA256_V11,
        "--v11c-stage", "equivalence_api_retry",
        "--v11c-v10-report", str(driver_v11.V10_REPORT_PATH_V11),
        "--v11c-failed-v11-journal",
        str(driver_v11c.driver_v11b.FAILED_V11_JOURNAL_PATH_V11B),
        "--v11c-v11b-failure-evidence",
        str(driver_v11c.V11B_FAILURE_PATH_V11C),
        "--v11c-perturbation-basis-seed", "20260714",
        "--population-size", "32", "--batch-size", "128",
        "--mini-batch-size", "64", "--seed", "43",
        "--target-alphas", "0",
        "--experiment-name", driver_v11d.EXPERIMENT_NAME_V11D,
    ]
    if dry:
        values.append("--v11c-dry-run")
    return values


def test_v11d_binds_exact_v11c_failure_and_implementation():
    binding = driver_v11d.bind_v11c_failure_v11d()
    assert binding["failure_phase"] == "post_engine_load_pre_journal_unknown"
    assert binding["model_update_applied"] is False
    assert binding["heldout_opened_or_scored"] is False
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
    with pytest.raises(ValueError, match="exact fresh experiment name"):
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
    assert payload["heldout_opened_or_scored"] is False
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
        lambda _attempt: {
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
    report = report_v11d.build_report(journal, attempt)
    assert report["schema"] == "eggroll-es-resident-sign-equivalence-report-v11d"
    assert report["passed"] is True
    assert report["v11d"]["launch_attempt_content_sha256"] == "attempt-content"
    assert report["v11d"]["journal_content_sha256"] == "journal-content"
