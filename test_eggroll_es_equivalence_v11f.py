import json

import pytest

import run_eggroll_es_anchor_equivalence_v11c as driver_v11c
import run_eggroll_es_anchor_equivalence_v11f as driver_v11f


def cli(dry=True):
    values = list(driver_v11f.FROZEN_REAL_ARGV_V11F)
    if dry:
        values.append("--v11c-dry-run")
    return values


def test_v11f_binds_exact_v11e_failure_and_scoring_semantics():
    binding = driver_v11f.bind_v11e_failure_v11f()
    assert binding["file_sha256"] == driver_v11f.V11E_HASHES
    assert binding["attempt_content_sha256"] == (
        driver_v11f.V11E_ATTEMPT_CONTENT_SHA256
    )
    assert binding["coefficient_plan_estimated"] is False
    assert binding["model_update_applied"] is False
    assert binding["baseline_validation_and_ood_scored"] is True
    assert binding["sealed_evaluation_data_opened_or_scored"] is False


def test_v11f_seed_audit_is_exact_and_negative_fails_closed():
    audit = driver_v11f.seed_forwarding_audit_v11f(
        driver_v11f.INHERITED_SEEDS_V11F
    )
    assert audit["seed_count"] == 32
    assert audit["incoming_seed_sha256"] == (
        driver_v11f.INHERITED_SEEDS_SHA256_V11F
    )
    assert audit["forwarded_seed_sha256"] == (
        driver_v11f.FROZEN_SEEDS_SHA256_V11F
    )
    assert audit["all_positions_corrected"] is True
    changed = list(driver_v11f.INHERITED_SEEDS_V11F)
    changed[0] += 1
    with pytest.raises(RuntimeError, match="inherited seed43"):
        driver_v11f.seed_forwarding_audit_v11f(changed)


def test_v11f_synthetic_delegate_receives_only_frozen_seeds(monkeypatch):
    seen = {}

    def delegate(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "delegated"

    monkeypatch.setattr(driver_v11f, "_ORIGINAL_V11C_EXECUTE", delegate)
    marker = object()
    result = driver_v11f.execute_line_search_v11f(
        marker, seeds=list(driver_v11f.INHERITED_SEEDS_V11F),
        targets=[0.0], sentinel="unchanged",
    )
    assert result == "delegated"
    assert seen["args"] == (marker,)
    assert seen["kwargs"]["seeds"] == driver_v11f.FROZEN_SEEDS_V11F
    assert seen["kwargs"]["targets"] == [0.0]
    assert seen["kwargs"]["sentinel"] == "unchanged"


def test_v11f_scoped_patch_restores_on_failure(monkeypatch):
    original = driver_v11f._ORIGINAL_V11C_EXECUTE
    assert driver_v11c.execute_line_search is original
    with pytest.raises(RuntimeError, match="synthetic"):
        with driver_v11f.scoped_seed_forwarding_v11f():
            assert driver_v11c.execute_line_search is (
                driver_v11f.execute_line_search_v11f
            )
            raise RuntimeError("synthetic scoped failure")
    assert driver_v11c.execute_line_search is original


def test_v11f_effective_cli_and_dry_artifact_prove_both_boundaries(
    capsys, monkeypatch,
):
    source = {"schema": "synthetic-committed-v11f-source"}
    monkeypatch.setattr(
        driver_v11f, "_source_provenance_v11f", lambda: dict(source),
    )
    audit = driver_v11f.audit_effective_cli_v11f(
        driver_v11f.FROZEN_REAL_ARGV_V11F
    )
    assert audit["field_count"] == 27
    assert audit["mismatch_fields"] == []
    assert audit["effective"] == driver_v11f.CANONICAL_DOWNSTREAM_V11F
    result = driver_v11f.main(cli())
    assert result["schema"] == "eggroll-es-seed-forwarding-dry-run-v11f"
    assert result["seed_forwarding_audit"]["passed"] is True
    assert result["source_provenance"] == source
    assert result["v11c_implementation"]["file_sha256"] == (
        driver_v11f.driver_v11e.V11C_IMPLEMENTATION_SHA256_V11E
    )
    assert "seed-forwarding-dry-run-v11f" in capsys.readouterr().out


def test_v11f_source_preflight_rejects_dirty_bytes(monkeypatch):
    def check_output(command, **_kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return "a" * 40 + "\n"
        if command[:2] == ["git", "show"]:
            return b"committed bytes"
        raise AssertionError(command)

    monkeypatch.setattr(driver_v11f.subprocess, "check_output", check_output)
    monkeypatch.setattr(driver_v11f, "_file_sha256", lambda _path: "0" * 64)
    with pytest.raises(RuntimeError, match="differs from committed"):
        driver_v11f._source_provenance_v11f()


def test_v11f_failure_attempt_is_exclusive_and_records_full_traceback(
    tmp_path, monkeypatch,
):
    attempt = tmp_path / "attempt.json"
    run_dir = tmp_path / driver_v11f.EXPERIMENT_NAME_V11F
    monkeypatch.setattr(driver_v11f, "RUNS", tmp_path)
    monkeypatch.setattr(driver_v11f, "_attempt_path", lambda: attempt)
    monkeypatch.setattr(
        driver_v11f, "_source_provenance_v11f",
        lambda: {"schema": "synthetic-source"},
    )
    monkeypatch.setattr(
        driver_v11c, "main",
        lambda _argv: (_ for _ in ()).throw(RuntimeError("synthetic V11f")),
    )
    binding = driver_v11f.bind_v11e_failure_v11f()
    cli_audit = driver_v11f.audit_effective_cli_v11f(
        driver_v11f.FROZEN_REAL_ARGV_V11F
    )
    seed_audit = driver_v11f.seed_forwarding_audit_v11f(
        driver_v11f.INHERITED_SEEDS_V11F
    )
    with pytest.raises(RuntimeError, match="synthetic V11f"):
        driver_v11f.run_exact_v11f(
            list(driver_v11f.FROZEN_REAL_ARGV_V11F), binding,
            cli_audit, seed_audit,
        )
    payload = json.loads(attempt.read_text())
    assert payload["status"] == "failed"
    assert "synthetic V11f" in payload["failure"]["traceback"]
    assert payload["model_update_applied"] is False
    assert payload["v11c_implementation"]["file_sha256"] == (
        driver_v11f.driver_v11e.V11C_IMPLEMENTATION_SHA256_V11E
    )
    assert payload["v11f_baseline_validation_and_ood_scored"] is False
    assert not run_dir.exists()
    with pytest.raises(ValueError, match="fresh"):
        driver_v11f.run_exact_v11f(
            list(driver_v11f.FROZEN_REAL_ARGV_V11F), binding,
            cli_audit, seed_audit,
        )


def test_v11f_success_is_not_complete_until_journal_is_bound(
    tmp_path, monkeypatch,
):
    attempt = tmp_path / "attempt.json"
    monkeypatch.setattr(driver_v11f, "RUNS", tmp_path)
    monkeypatch.setattr(driver_v11f, "_attempt_path", lambda: attempt)
    monkeypatch.setattr(
        driver_v11f, "_source_provenance_v11f",
        lambda: {"schema": "synthetic-source"},
    )
    def succeed(_argv):
        (tmp_path / driver_v11f.EXPERIMENT_NAME_V11F).mkdir()
        return {"ok": True}

    monkeypatch.setattr(driver_v11c, "main", succeed)
    binding = {"schema": "synthetic-v11e-binding"}
    journal_binding = {
        "schema": "eggroll-es-v11f-journal-binding",
        "content_sha256": "journal-content",
    }
    monkeypatch.setattr(
        driver_v11f, "_completed_journal_binding_v11f",
        lambda _run: dict(journal_binding),
    )
    monkeypatch.setattr(driver_v11f, "_baseline_scored", lambda _run: True)
    result = driver_v11f.run_exact_v11f(
        list(driver_v11f.FROZEN_REAL_ARGV_V11F), binding,
        {"schema": "synthetic-cli"}, {"schema": "synthetic-seeds"},
    )
    assert result == {"ok": True}
    payload = json.loads(attempt.read_text())
    assert payload["status"] == "complete"
    assert payload["journal_binding"] == journal_binding
    assert payload["v11f_baseline_validation_and_ood_scored"] is True


def test_v11f_journal_binding_failure_records_failed_not_complete(
    tmp_path, monkeypatch,
):
    attempt = tmp_path / "attempt.json"
    monkeypatch.setattr(driver_v11f, "RUNS", tmp_path)
    monkeypatch.setattr(driver_v11f, "_attempt_path", lambda: attempt)
    monkeypatch.setattr(
        driver_v11f, "_source_provenance_v11f",
        lambda: {"schema": "synthetic-source"},
    )

    def succeed(_argv):
        (tmp_path / driver_v11f.EXPERIMENT_NAME_V11F).mkdir()
        return {"engine_returned": True}

    monkeypatch.setattr(driver_v11c, "main", succeed)
    monkeypatch.setattr(
        driver_v11f, "_completed_journal_binding_v11f",
        lambda _run: (_ for _ in ()).throw(
            RuntimeError("synthetic incomplete journal")
        ),
    )
    with pytest.raises(RuntimeError, match="synthetic incomplete journal"):
        driver_v11f.run_exact_v11f(
            list(driver_v11f.FROZEN_REAL_ARGV_V11F),
            {"schema": "binding"}, {"schema": "cli"},
            {"schema": "seeds"},
        )
    payload = json.loads(attempt.read_text())
    assert payload["status"] == "failed"
    assert payload["phase"] == "inside_v11c_driver_main"
    assert "synthetic incomplete journal" in payload["failure"]["traceback"]
    assert "journal_binding" not in payload
