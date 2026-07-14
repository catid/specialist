import copy
import json

import pytest

import run_eggroll_es_anchor_equivalence_v11c as driver_v11c
import run_eggroll_es_anchor_equivalence_v11e as driver_v11e
import run_eggroll_es_anchor_equivalence_v11f as driver_v11f
import run_eggroll_es_anchor_equivalence_v11g as driver_v11g


def cli(dry=True):
    values = list(driver_v11g.FROZEN_REAL_ARGV_V11G)
    if dry:
        values.append("--v11c-dry-run")
    return values


def test_v11g_binds_exact_v11f_failure_source_and_policy_mismatch():
    binding = driver_v11g.bind_v11f_failure_v11g()
    assert binding["file_sha256"] == driver_v11g.V11F_HASHES
    assert binding["source_commit"] == driver_v11g.V11F_SOURCE_COMMIT
    assert binding["source_sha256"] == driver_v11g.V11F_SOURCE_SHA256
    assert binding["seed_sha256"] == driver_v11f.FROZEN_SEEDS_SHA256_V11F
    assert binding["coefficient_plan_estimated"] is True
    assert binding["policy_missing_exactly"] == (
        driver_v11g.POLICY_CORRECTION_V11G
    )
    assert binding["model_update_applied"] is False
    assert binding["baseline_validation_and_ood_scored"] is True


def test_v11g_policy_audit_injects_exactly_three_fields():
    audit = driver_v11g.policy_forwarding_audit_v11g(
        driver_v11g.BASE_POLICY_V11G
    )
    assert audit["missing_keys"] == sorted(
        driver_v11g.POLICY_CORRECTION_V11G
    )
    assert audit["injected_policy"] == driver_v11g.POLICY_CORRECTION_V11G
    assert audit["after_policy"] == driver_v11g.INHERITED_POLICY_V11G
    assert set(audit["after_policy"]) - set(audit["before_policy"]) == set(
        driver_v11g.POLICY_CORRECTION_V11G
    )


@pytest.mark.parametrize("mutation", ["missing_base", "conflict", "extra"])
def test_v11g_policy_audit_rejects_every_nonexact_base_policy(mutation):
    policy = copy.deepcopy(driver_v11g.BASE_POLICY_V11G)
    if mutation == "missing_base":
        policy.pop("alpha_order")
    elif mutation == "conflict":
        policy["document_lcb_anchor_required"] = False
    else:
        policy["unexpected"] = True
    with pytest.raises(RuntimeError, match="inner V4 base policy changed"):
        driver_v11g.policy_forwarding_audit_v11g(policy)


def test_v11g_inner_wrapper_delegates_unchanged_and_reseals(monkeypatch):
    seen = {}

    def inner(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return {
            "schema": "synthetic-v4", "policy": copy.deepcopy(
                driver_v11g.BASE_POLICY_V11G
            ),
            "content_sha256_before_self_field": "old",
        }

    monkeypatch.setattr(driver_v11g, "_ORIGINAL_V4_EXECUTE", inner)
    marker = object()
    journal = driver_v11g.execute_line_search_v11g_inner(
        marker, seeds=[1, 2], sentinel="unchanged",
    )
    assert seen == {
        "args": (marker,),
        "kwargs": {"seeds": [1, 2], "sentinel": "unchanged"},
    }
    assert journal["policy"] == driver_v11g.INHERITED_POLICY_V11G
    assert journal["content_sha256_before_self_field"] == driver_v11g._canonical({
        key: value for key, value in journal.items()
        if key != "content_sha256_before_self_field"
    })


def test_v11g_scoped_policy_patch_restores_on_failure():
    original = driver_v11g._ORIGINAL_V4_EXECUTE
    assert driver_v11c.driver_v4.execute_line_search is original
    with pytest.raises(RuntimeError, match="synthetic"):
        with driver_v11g.scoped_policy_forwarding_v11g():
            assert driver_v11c.driver_v4.execute_line_search is (
                driver_v11g.execute_line_search_v11g_inner
            )
            raise RuntimeError("synthetic scoped failure")
    assert driver_v11c.driver_v4.execute_line_search is original


def test_v11g_seed_forwarding_remains_exact_through_v11f(monkeypatch):
    seen = {}

    def delegate(*args, **kwargs):
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr(driver_v11f, "_ORIGINAL_V11C_EXECUTE", delegate)
    assert driver_v11f.execute_line_search_v11f(
        object(), seeds=driver_v11f.INHERITED_SEEDS_V11F,
        sentinel="unchanged",
    ) == "ok"
    assert seen["seeds"] == driver_v11f.FROZEN_SEEDS_V11F
    assert seen["sentinel"] == "unchanged"


def test_v11g_cli_and_dry_artifact_bind_seed_policy_source_and_bundle(
    monkeypatch, capsys,
):
    source = {"schema": "synthetic-committed-v11g-source"}
    monkeypatch.setattr(
        driver_v11g, "_source_provenance_v11g", lambda: dict(source),
    )
    cli_audit = driver_v11g.audit_effective_cli_v11g(
        driver_v11g.FROZEN_REAL_ARGV_V11G
    )
    assert cli_audit["field_count"] == 27
    assert cli_audit["mismatch_fields"] == []
    result = driver_v11g.main(cli())
    assert result["schema"] == "eggroll-es-policy-forwarding-dry-run-v11g"
    assert result["policy_forwarding_audit"]["passed"] is True
    assert result["seed_forwarding_audit"]["passed"] is True
    assert result["source_provenance"] == source
    assert result["v11c_implementation"]["file_sha256"] == (
        driver_v11e.V11C_IMPLEMENTATION_SHA256_V11E
    )
    assert "policy-forwarding-dry-run-v11g" in capsys.readouterr().out


def test_v11g_failure_telemetry_restores_both_patches(
    tmp_path, monkeypatch,
):
    attempt = tmp_path / "attempt.json"
    monkeypatch.setattr(driver_v11g, "RUNS", tmp_path)
    monkeypatch.setattr(driver_v11g, "_attempt_path", lambda: attempt)
    monkeypatch.setattr(
        driver_v11g, "_source_provenance_v11g",
        lambda: {"schema": "synthetic-source"},
    )
    monkeypatch.setattr(
        driver_v11c, "main",
        lambda _argv: (_ for _ in ()).throw(RuntimeError("synthetic V11g")),
    )
    with pytest.raises(RuntimeError, match="synthetic V11g"):
        driver_v11g.run_exact_v11g(
            list(driver_v11g.FROZEN_REAL_ARGV_V11G),
            {"schema": "failure"}, {"schema": "cli"},
            driver_v11g.policy_forwarding_audit_v11g(
                driver_v11g.BASE_POLICY_V11G
            ),
        )
    payload = json.loads(attempt.read_text())
    assert payload["status"] == "failed"
    assert "synthetic V11g" in payload["failure"]["traceback"]
    assert payload["model_update_applied"] is False
    assert driver_v11c.execute_line_search is driver_v11f._ORIGINAL_V11C_EXECUTE
    assert driver_v11c.driver_v4.execute_line_search is (
        driver_v11g._ORIGINAL_V4_EXECUTE
    )


def test_v11g_post_claim_run_directory_race_never_delegates(
    tmp_path, monkeypatch,
):
    attempt = tmp_path / "attempt.json"
    run_dir = tmp_path / driver_v11g.EXPERIMENT_NAME_V11G
    monkeypatch.setattr(driver_v11g, "RUNS", tmp_path)
    monkeypatch.setattr(driver_v11g, "_attempt_path", lambda: attempt)
    monkeypatch.setattr(
        driver_v11g, "_source_provenance_v11g",
        lambda: {"schema": "synthetic-source"},
    )
    original_write = driver_v11g._exclusive_write

    def claim_then_race(path, payload):
        original_write(path, payload)
        run_dir.mkdir()

    called = []
    monkeypatch.setattr(driver_v11g, "_exclusive_write", claim_then_race)
    monkeypatch.setattr(driver_v11c, "main", lambda _argv: called.append(True))
    with pytest.raises(ValueError, match="appeared after exclusive claim"):
        driver_v11g.run_exact_v11g(
            list(driver_v11g.FROZEN_REAL_ARGV_V11G),
            {"schema": "failure"}, {"schema": "cli"},
            {"schema": "policy"},
        )
    assert called == []
    payload = json.loads(attempt.read_text())
    assert payload["status"] == "failed"
    assert payload["phase"] == "exclusive_claim_detected_existing_run_directory"
    assert payload["failure"]["type"] == "FreshRunReservationError"
