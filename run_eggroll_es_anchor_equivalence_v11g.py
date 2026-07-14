#!/usr/bin/env python3
"""Fresh V11f retry restoring three inherited completion-policy fields."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import run_eggroll_es_anchor_equivalence_v11c as driver_v11c
import run_eggroll_es_anchor_equivalence_v11e as driver_v11e
import run_eggroll_es_anchor_equivalence_v11f as driver_v11f


ROOT = Path(__file__).resolve().parent
RUNS = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
EXPERIMENT_NAME_V11G = (
    "snapshot794_layer_v11g_middle_late_resident_sign_exact_v10_"
    "policy_forwarded_d43d44_a43a44_basis20260714"
)
EXPECTED_RECIPE_SHA256_V11G = (
    "9515e915fdb16eb15c815034f2f929fe2f786e3921a2a7d91062b5f39484385f"
)
V11F_NAME = driver_v11f.EXPERIMENT_NAME_V11F
V11F_RUN = (RUNS / V11F_NAME).resolve()
V11F_ATTEMPT = (RUNS / f".{V11F_NAME}.launch_attempt.json").resolve()
V11F_JOURNAL = (V11F_RUN / "alpha_line_search.json").resolve()
V11F_PLAN = (V11F_RUN / "anchor-plan-iteration-1.json").resolve()
V11F_IDENTITY = (
    V11F_RUN / "alpha-zero-identity-audit-iteration-1.json"
).resolve()
V11F_FAILURE_DOCUMENT = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_RESIDENT_SIGN_EQUIVALENCE_V11F_FAILURE.md"
).resolve()
V11F_FAILURE_DOCUMENT_COMMIT = "190ddb7922cf834b81101c5927a78fb447295831"
V11F_SOURCE_COMMIT = "d448c8a980bc6326a92b9938aaaeafe6114b0bab"
V11F_SOURCE_SHA256 = (
    "e4d50b4bac41c4c294e999dba61e8ced9aa95e5136f048c3db75350b97f999ff"
)
V11F_ATTEMPT_CONTENT_SHA256 = (
    "1111325aee79004764083e70ff32bb28f1f1f3353b8194a2ef5d99e639c77c36"
)
V11F_HASHES = {
    "attempt": "3b97db50578af3d30364988e16ffdb50e1d54dd81c9084fd072e9c8160d9cd39",
    "journal": "62cea23dbdba65f11d14f98d38d249dcbda3b3984739035990cf10db9e192eb4",
    "plan": "2909b74c6d2bcec1b7db5186b2c4ab151511e34195af3a89b8468f8709c03af2",
    "identity": "168ace19f541bac3a19a29757b3cfe198b19207c0852585c761b69f1cfc706a4",
    "document": "0e5f659ade028d77ab6fa7f82ad2ac89093de6d1aa045a56dd8bb106870ebb47",
}
POLICY_CORRECTION_V11G = {
    "document_lcb_anchor_required": True,
    "ood_validation_heldout_as_objective": False,
    "optimization_data": "train_and_anchor_only",
}
INHERITED_POLICY_V11G = copy.deepcopy(
    driver_v11c.driver_v11b.driver_v11.INHERITED_POLICY_V11
)
BASE_POLICY_V11G = {
    key: value for key, value in INHERITED_POLICY_V11G.items()
    if key not in POLICY_CORRECTION_V11G
}
EXPECTED_COMPLETE_POLICY_V11G = {
    **INHERITED_POLICY_V11G,
    **driver_v11c.driver_v11b.driver_v11.V11_POLICY,
    **driver_v11c.driver_v11b.V11B_POLICY,
    **driver_v11c.V11C_POLICY,
}
FROZEN_REAL_ARGV_V11G = list(driver_v11f.FROZEN_REAL_ARGV_V11F)
FROZEN_REAL_ARGV_V11G[
    FROZEN_REAL_ARGV_V11G.index("--experiment-name") + 1
] = EXPERIMENT_NAME_V11G
FROZEN_REAL_ARGV_V11G = tuple(FROZEN_REAL_ARGV_V11G)
CANONICAL_DOWNSTREAM_V11G = copy.deepcopy(
    driver_v11f.CANONICAL_DOWNSTREAM_V11F
)
CANONICAL_DOWNSTREAM_V11G["experiment_name"] = EXPERIMENT_NAME_V11G
DIAGNOSTIC_ENV_V11G = copy.deepcopy(driver_v11f.DIAGNOSTIC_ENV_V11F)
_ORIGINAL_V4_EXECUTE = driver_v11c.driver_v4.execute_line_search


def _file_sha256(path):
    return driver_v11c.driver_v1.file_sha256(path)


def _canonical(value):
    return driver_v11c.driver_v1.canonical_sha256(value)


def bind_v11f_failure_v11g():
    paths = {
        "attempt": V11F_ATTEMPT, "journal": V11F_JOURNAL,
        "plan": V11F_PLAN, "identity": V11F_IDENTITY,
        "document": V11F_FAILURE_DOCUMENT,
    }
    if any(_file_sha256(paths[key]) != value for key, value in V11F_HASHES.items()):
        raise RuntimeError("v11g exact V11f failure files changed")
    document = subprocess.check_output([
        "git", "show", f"{V11F_FAILURE_DOCUMENT_COMMIT}:"
        "experiments/eggroll_es_hpo/S6_RESIDENT_SIGN_EQUIVALENCE_V11F_FAILURE.md",
    ], cwd=ROOT)
    source = subprocess.check_output([
        "git", "show", f"{V11F_SOURCE_COMMIT}:"
        "run_eggroll_es_anchor_equivalence_v11f.py",
    ], cwd=ROOT)
    if (
        hashlib.sha256(document).hexdigest() != V11F_HASHES["document"]
        or hashlib.sha256(source).hexdigest() != V11F_SOURCE_SHA256
    ):
        raise RuntimeError("v11g committed V11f evidence changed")
    attempt = json.loads(V11F_ATTEMPT.read_text())
    journal = json.loads(V11F_JOURNAL.read_text())
    plan = json.loads(V11F_PLAN.read_text())
    identity = json.loads(V11F_IDENTITY.read_text())
    expected_missing = set(POLICY_CORRECTION_V11G)
    actual_policy = journal.get("policy", {})
    if (
        attempt.get("status") != "failed"
        or attempt.get("failure", {}).get("message")
        != "v11c journal identity, policy, or completion changed"
        or attempt.get("content_sha256_before_self_field")
        != V11F_ATTEMPT_CONTENT_SHA256
        or attempt.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in attempt.items()
            if key != "content_sha256_before_self_field"
        })
        or attempt.get("source_provenance", {}).get("git_head")
        != V11F_SOURCE_COMMIT
        or attempt.get("source_provenance", {}).get("file_sha256")
        != V11F_SOURCE_SHA256
        or attempt.get("model_update_applied") is not False
        or attempt.get("v11f_baseline_validation_and_ood_scored") is not True
        or journal.get("status") != "failed"
        or journal.get("failure", {}).get("phase")
        != "validating_complete_v11c_exact_equivalence"
        or not isinstance(journal.get("coefficient_plan"), dict)
        or journal.get("seeds") != driver_v11f.FROZEN_SEEDS_V11F
        or set(EXPECTED_COMPLETE_POLICY_V11G) - set(actual_policy)
        != expected_missing
        or set(actual_policy) - set(EXPECTED_COMPLETE_POLICY_V11G)
        or any(
            actual_policy[key] != value
            for key, value in EXPECTED_COMPLETE_POLICY_V11G.items()
            if key in actual_policy
        )
        or plan.get("seeds") != driver_v11f.FROZEN_SEEDS_V11F
        or plan.get("applied_alpha") != 0.0
        or plan.get("applications") != []
        or identity.get("passed") is not True
    ):
        raise RuntimeError("v11g V11f failure semantics changed")
    binding = {
        "schema": "eggroll-es-v11f-policy-forwarding-failure-v11g",
        "paths": {key: str(value) for key, value in paths.items()},
        "file_sha256": copy.deepcopy(V11F_HASHES),
        "attempt_content_sha256": V11F_ATTEMPT_CONTENT_SHA256,
        "failure_document_commit": V11F_FAILURE_DOCUMENT_COMMIT,
        "source_commit": V11F_SOURCE_COMMIT,
        "source_sha256": V11F_SOURCE_SHA256,
        "seed_sha256": driver_v11f.FROZEN_SEEDS_SHA256_V11F,
        "coefficient_plan_estimated": True,
        "policy_missing_exactly": copy.deepcopy(POLICY_CORRECTION_V11G),
        "model_update_applied": False,
        "baseline_validation_and_ood_scored": True,
        "sealed_evaluation_data_opened_or_scored": False,
    }
    binding["binding_sha256"] = _canonical(binding)
    return binding


def policy_forwarding_audit_v11g(policy):
    policy = copy.deepcopy(policy)
    missing = {key for key in INHERITED_POLICY_V11G if key not in policy}
    conflicts = {
        key: policy[key] for key in POLICY_CORRECTION_V11G
        if key in policy
    }
    if (
        policy != BASE_POLICY_V11G
        or missing != set(POLICY_CORRECTION_V11G)
        or conflicts
    ):
        raise RuntimeError("v11g inner V4 base policy changed")
    corrected = {**policy, **POLICY_CORRECTION_V11G}
    if corrected != INHERITED_POLICY_V11G:
        raise RuntimeError("v11g corrected inherited policy changed")
    audit = {
        "schema": "eggroll-es-policy-forwarding-audit-v11g",
        "before_policy": policy, "before_sha256": _canonical(policy),
        "missing_keys": sorted(missing),
        "injected_policy": copy.deepcopy(POLICY_CORRECTION_V11G),
        "after_policy": corrected, "after_sha256": _canonical(corrected),
        "delegate": "run_eggroll_es_anchor_line_search_v4.execute_line_search",
        "passed": True,
    }
    audit["content_sha256_before_self_field"] = _canonical(audit)
    return audit


def execute_line_search_v11g_inner(*args, **kwargs):
    journal = _ORIGINAL_V4_EXECUTE(*args, **kwargs)
    audit = policy_forwarding_audit_v11g(journal.get("policy"))
    journal["policy"] = copy.deepcopy(audit["after_policy"])
    journal.pop("content_sha256_before_self_field", None)
    journal["content_sha256_before_self_field"] = _canonical(journal)
    return journal


@contextlib.contextmanager
def scoped_policy_forwarding_v11g():
    if driver_v11c.driver_v4.execute_line_search is not _ORIGINAL_V4_EXECUTE:
        raise RuntimeError("v11g V4 execute delegate changed before patch")
    driver_v11c.driver_v4.execute_line_search = execute_line_search_v11g_inner
    try:
        yield
    finally:
        driver_v11c.driver_v4.execute_line_search = _ORIGINAL_V4_EXECUTE


def _patch_v11c_globals():
    prior = (driver_v11c.EXPERIMENT_NAME_V11C, driver_v11c.EXPECTED_RECIPE_SHA256_V11C)
    driver_v11c.EXPERIMENT_NAME_V11C = EXPERIMENT_NAME_V11G
    driver_v11c.EXPECTED_RECIPE_SHA256_V11C = EXPECTED_RECIPE_SHA256_V11G
    return prior


def _restore_v11c_globals(prior):
    driver_v11c.EXPERIMENT_NAME_V11C, driver_v11c.EXPECTED_RECIPE_SHA256_V11C = prior


def audit_effective_cli_v11g(argv):
    prior = _patch_v11c_globals()
    old_argv = sys.argv
    try:
        bundle, remaining = driver_v11c.anchor_v11c.parse_frozen_layer_plan_cli_v11c(list(argv))
        outer, base_argv = driver_v11c.validate_frozen_execution_cli_v11c(remaining, bundle)
        sys.argv = [old_argv[0], *base_argv]
        parsed = driver_v11c.driver_v1.parse_args()
    finally:
        sys.argv = old_argv
        _restore_v11c_globals(prior)
    effective = driver_v11e._normalize_effective_args_v11e(parsed)
    normalized_outer = driver_v11e._normalize_outer_execution_v11e(outer)
    if effective != CANONICAL_DOWNSTREAM_V11G or normalized_outer != effective:
        raise RuntimeError("v11g effective 27-field CLI changed")
    audit = {
        "schema": "eggroll-es-effective-cli-v11g", "field_count": 27,
        "outer": normalized_outer, "effective": effective,
        "mismatch_fields": [], "base_argv_sha256": _canonical(base_argv),
        "outer_argv_sha256": _canonical(list(argv)), "passed": True,
    }
    audit["content_sha256_before_self_field"] = _canonical(audit)
    return audit


def _source_provenance_v11g():
    relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    try:
        committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("v11g requires committed source at HEAD") from error
    current = _file_sha256(__file__)
    if hashlib.sha256(committed).hexdigest() != current:
        raise RuntimeError("v11g source differs from committed HEAD")
    return {"schema": "eggroll-es-v11g-source", "git_head": head,
            "relative_path": relative, "file_sha256": current}


def _runtime(argv):
    argv = list(argv)
    allowed = (list(FROZEN_REAL_ARGV_V11G), [*FROZEN_REAL_ARGV_V11G, "--v11c-dry-run"])
    if argv not in allowed:
        raise ValueError("v11g requires exact frozen real or dry CLI")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v11c-dry-run", action="store_true")
    runtime, _ = parser.parse_known_args(argv)
    return runtime


def _attempt_path():
    return RUNS / f".{EXPERIMENT_NAME_V11G}.launch_attempt.json"


def _completed_journal_binding(run_dir):
    path = (run_dir / driver_v11c.driver_v1.JOURNAL_NAME).resolve()
    expected = (RUNS / EXPERIMENT_NAME_V11G / driver_v11c.driver_v1.JOURNAL_NAME).resolve()
    if path != expected:
        raise RuntimeError("v11g completed journal path changed")
    journal = json.loads(path.read_text())
    if (
        journal.get("seeds") != driver_v11f.FROZEN_SEEDS_V11F
        or journal.get("policy") != EXPECTED_COMPLETE_POLICY_V11G
    ):
        raise RuntimeError("v11g completed seed or policy changed")
    audit = driver_v11c.validate_completed_journal_v11c(journal)
    return {
        "schema": "eggroll-es-v11g-journal-binding", "path": str(path),
        "file_sha256": _file_sha256(path),
        "content_sha256": audit["content_sha256"],
        "journal_schema": journal.get("schema"),
        "seed_sha256": _canonical(journal["seeds"]),
        "policy_sha256": _canonical(journal["policy"]),
    }


def _seal(payload):
    payload.pop("content_sha256_before_self_field", None)
    payload["content_sha256_before_self_field"] = _canonical(payload)


def _exclusive_write(path, payload):
    _seal(payload)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v11g launch-attempt evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _rewrite(path, payload):
    _seal(payload)
    driver_v11c.driver_v1.atomic_write_json(path, payload)


def run_exact_v11g(argv, failure_binding, cli_audit, policy_audit):
    attempt_path = _attempt_path()
    run_dir = RUNS / EXPERIMENT_NAME_V11G
    if attempt_path.exists() or run_dir.exists():
        raise ValueError("v11g requires fresh attempt and run paths")
    source = _source_provenance_v11g()
    implementation = driver_v11e.audit_v11c_implementation_v11e()
    seed_audit = driver_v11f.seed_forwarding_audit_v11f(
        driver_v11f.INHERITED_SEEDS_V11F
    )
    payload = {
        "schema": "eggroll-es-durable-launch-attempt-v11g",
        "status": "launching", "phase": "before_v11c_driver_main",
        "experiment_name": EXPERIMENT_NAME_V11G, "run_directory": str(run_dir),
        "source_provenance": source, "v11c_implementation": implementation,
        "v11f_failure_evidence": failure_binding, "effective_cli": cli_audit,
        "seed_forwarding_audit": seed_audit,
        "policy_forwarding_audit": policy_audit,
        "intended_recipe_or_data_changed_from_v11f": False,
        "effective_completion_policy_corrected": True,
        "target_alpha_zero_only": True, "model_update_applied": False,
        "prior_v11f_baseline_validation_and_ood_scored": True,
        "v11g_baseline_validation_and_ood_scored": False,
        "sealed_evaluation_data_opened_or_scored": False,
    }
    _exclusive_write(attempt_path, payload)
    if run_dir.exists():
        payload.update({
            "status": "failed",
            "phase": "exclusive_claim_detected_existing_run_directory",
            "failure": {"type": "FreshRunReservationError",
                        "message": "v11g run directory appeared after claim",
                        "traceback": ""},
            "run_directory_exists_after_attempt": True,
        })
        _rewrite(attempt_path, payload)
        raise ValueError("v11g run directory appeared after exclusive claim")
    old_env = {key: os.environ.get(key) for key in DIAGNOSTIC_ENV_V11G}
    os.environ.update(DIAGNOSTIC_ENV_V11G)
    prior = _patch_v11c_globals()
    journal_binding = None
    try:
        with scoped_policy_forwarding_v11g():
            with driver_v11f.scoped_seed_forwarding_v11f():
                result = driver_v11c.main(list(argv))
        journal_binding = _completed_journal_binding(run_dir)
    except BaseException as error:
        payload.update({
            "status": "failed", "phase": "inside_v11c_driver_main",
            "failure": {"type": type(error).__name__, "message": str(error),
                        "traceback": traceback.format_exc()},
            "run_directory_exists_after_attempt": run_dir.exists(),
            "v11c_journal_exists_after_attempt": (
                run_dir / driver_v11c.driver_v1.JOURNAL_NAME
            ).exists(),
            "v11g_baseline_validation_and_ood_scored": (
                driver_v11f._baseline_scored(run_dir)
            ),
        })
        _rewrite(attempt_path, payload)
        raise
    else:
        payload.update({
            "status": "complete", "phase": "after_v11c_driver_main",
            "run_directory_exists_after_attempt": True,
            "v11c_journal_exists_after_attempt": True,
            "v11g_baseline_validation_and_ood_scored": True,
            "journal_binding": journal_binding,
        })
        _rewrite(attempt_path, payload)
        return result
    finally:
        _restore_v11c_globals(prior)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    runtime = _runtime(argv)
    failure = bind_v11f_failure_v11g()
    cli_audit = audit_effective_cli_v11g(argv)
    policy_audit = policy_forwarding_audit_v11g(BASE_POLICY_V11G)
    seed_audit = driver_v11f.seed_forwarding_audit_v11f(
        driver_v11f.INHERITED_SEEDS_V11F
    )
    implementation = driver_v11e.audit_v11c_implementation_v11e()
    source = _source_provenance_v11g()
    if runtime.v11c_dry_run:
        prior = _patch_v11c_globals()
        try:
            result = driver_v11c.main(argv)
        finally:
            _restore_v11c_globals(prior)
        result = copy.deepcopy(result)
        result.update({
            "schema": "eggroll-es-policy-forwarding-dry-run-v11g",
            "v11f_failure_binding_sha256": failure["binding_sha256"],
            "effective_cli": cli_audit, "seed_forwarding_audit": seed_audit,
            "policy_forwarding_audit": policy_audit,
            "v11c_implementation": implementation, "source_provenance": source,
        })
        print(json.dumps(result, sort_keys=True))
        return result
    return run_exact_v11g(argv, failure, cli_audit, policy_audit)


if __name__ == "__main__":
    main()
