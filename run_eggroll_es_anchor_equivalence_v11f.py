#!/usr/bin/env python3
"""Fresh V11e retry restoring the frozen V11 perturbation basis."""

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

import numpy as np

import run_eggroll_es_anchor_equivalence_v11c as driver_v11c
import run_eggroll_es_anchor_equivalence_v11e as driver_v11e


ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME_V11F = (
    "snapshot794_layer_v11f_middle_late_resident_sign_exact_v10_"
    "seed_forwarded_d43d44_a43a44_basis20260714"
)
EXPECTED_RECIPE_SHA256_V11F = (
    "619c8ff944198d4db4f1280203b2f6f7b202a976ebe664e6f3b25de210c6abd3"
)
V11E_NAME = driver_v11e.EXPERIMENT_NAME_V11E
RUNS = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
V11E_ATTEMPT = (RUNS / f".{V11E_NAME}.launch_attempt.json").resolve()
V11E_RUN = (RUNS / V11E_NAME).resolve()
V11E_JOURNAL = (V11E_RUN / "alpha_line_search.json").resolve()
V11E_PLAN = (V11E_RUN / "anchor-plan-iteration-1.json").resolve()
V11E_IDENTITY = (
    V11E_RUN / "alpha-zero-identity-audit-iteration-1.json"
).resolve()
V11E_FAILURE_DOCUMENT = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_RESIDENT_SIGN_EQUIVALENCE_V11E_FAILURE.md"
).resolve()
V11E_FAILURE_DOCUMENT_COMMIT = "c21255925aaa4a8367a092c153a386cf20e26209"
V11E_HASHES = {
    "attempt": "4297a4dd0ca47aa61c0359b6aa06a281ff389821f79d42d92949d945d5e39ac3",
    "journal": "106168485a7cdc82726845f2f97384717a51daede75271ab9855006d01f716a6",
    "plan": "50afc13272e5f5cdd4aca51a37aa499c37bdef59ce6c3b4d2062798b5ad3ba62",
    "identity": "37ad7696840f00472c6b03a734b7ab084b9f5881a92175084fcae49d0f9c9ad9",
    "document": "c7c9afe977c4ada9c19c2e3eea1bf8e4d9039f1b5f720f0bc38507a1b6403e55",
}
V11E_ATTEMPT_CONTENT_SHA256 = (
    "314f3c1bd4cbe4d9614022e5b6570f963f88d80a4871d1a53b6d4b5523aa81a4"
)
INHERITED_SEEDS_V11F = np.random.default_rng(seed=43).integers(
    0, 2**30, size=32, dtype=np.int64,
).tolist()
FROZEN_SEEDS_V11F = list(
    driver_v11c.driver_v11b.driver_v11.PERTURBATION_SEEDS_V11
)
INHERITED_SEEDS_SHA256_V11F = (
    "78c046a76d8f31123ec42d189cf134b4424949f720173c7873417924c3401a89"
)
FROZEN_SEEDS_SHA256_V11F = (
    "07fa4900cd10fd17b678355389adcfa4f5ac7ec356be46088466cd32b032e6e1"
)
FROZEN_REAL_ARGV_V11F = list(driver_v11e.FROZEN_REAL_ARGV_V11E)
FROZEN_REAL_ARGV_V11F[
    FROZEN_REAL_ARGV_V11F.index("--experiment-name") + 1
] = EXPERIMENT_NAME_V11F
FROZEN_REAL_ARGV_V11F = tuple(FROZEN_REAL_ARGV_V11F)
CANONICAL_DOWNSTREAM_V11F = copy.deepcopy(
    driver_v11e.CANONICAL_DOWNSTREAM_V11E
)
CANONICAL_DOWNSTREAM_V11F["experiment_name"] = EXPERIMENT_NAME_V11F
DIAGNOSTIC_ENV_V11F = copy.deepcopy(driver_v11e.DIAGNOSTIC_ENV_V11E)
_ORIGINAL_V11C_EXECUTE = driver_v11c.execute_line_search


def _file_sha256(path):
    return driver_v11c.driver_v1.file_sha256(path)


def _canonical(value):
    return driver_v11c.driver_v1.canonical_sha256(value)


def bind_v11e_failure_v11f():
    paths = {
        "attempt": V11E_ATTEMPT, "journal": V11E_JOURNAL,
        "plan": V11E_PLAN, "identity": V11E_IDENTITY,
        "document": V11E_FAILURE_DOCUMENT,
    }
    if any(_file_sha256(paths[key]) != value for key, value in V11E_HASHES.items()):
        raise RuntimeError("v11f exact V11e failure files changed")
    committed = subprocess.check_output([
        "git", "show", f"{V11E_FAILURE_DOCUMENT_COMMIT}:"
        "experiments/eggroll_es_hpo/S6_RESIDENT_SIGN_EQUIVALENCE_V11E_FAILURE.md",
    ], cwd=ROOT)
    if hashlib.sha256(committed).hexdigest() != V11E_HASHES["document"]:
        raise RuntimeError("v11f committed V11e failure document changed")
    attempt = json.loads(V11E_ATTEMPT.read_text())
    journal = json.loads(V11E_JOURNAL.read_text())
    plan = json.loads(V11E_PLAN.read_text())
    identity = json.loads(V11E_IDENTITY.read_text())
    if (
        attempt.get("status") != "failed"
        or attempt.get("failure", {}).get("message")
        != "v11 resident-sign artifact identity changed"
        or attempt.get("content_sha256_before_self_field")
        != V11E_ATTEMPT_CONTENT_SHA256
        or attempt.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in attempt.items()
            if key != "content_sha256_before_self_field"
        })
        or attempt.get("model_update_applied") is not False
        or journal.get("status") != "failed"
        or journal.get("coefficient_plan") is not None
        or journal.get("failure", {}).get("message")
        != "v11 resident-sign artifact identity changed"
        or journal.get("seeds") != INHERITED_SEEDS_V11F
        or plan.get("seeds") != INHERITED_SEEDS_V11F
        or journal.get("snapshot", {}).get(
            "resident_sign_equivalence_v11", {}
        ).get("perturbation_seeds") != FROZEN_SEEDS_V11F
        or identity.get("passed") is not True
        or not journal.get("states")
        or journal["states"][0].get("ood_prose") is None
        or journal["states"][0].get("qa") is None
    ):
        raise RuntimeError("v11f V11e failure semantics changed")
    binding = {
        "schema": "eggroll-es-v11e-seed-forwarding-failure-v11f",
        "paths": {key: str(value) for key, value in paths.items()},
        "file_sha256": copy.deepcopy(V11E_HASHES),
        "attempt_content_sha256": V11E_ATTEMPT_CONTENT_SHA256,
        "failure_document_commit": V11E_FAILURE_DOCUMENT_COMMIT,
        "incoming_seed_sha256": INHERITED_SEEDS_SHA256_V11F,
        "required_seed_sha256": FROZEN_SEEDS_SHA256_V11F,
        "coefficient_plan_estimated": False,
        "model_update_applied": False,
        "baseline_validation_and_ood_scored": True,
        "sealed_evaluation_data_opened_or_scored": False,
    }
    binding["binding_sha256"] = _canonical(binding)
    return binding


def seed_forwarding_audit_v11f(incoming):
    incoming = list(incoming)
    if incoming != INHERITED_SEEDS_V11F:
        raise RuntimeError("v11f inherited seed43 population schedule changed")
    if (
        _canonical(incoming) != INHERITED_SEEDS_SHA256_V11F
        or _canonical(FROZEN_SEEDS_V11F) != FROZEN_SEEDS_SHA256_V11F
        or len(set(incoming) & set(FROZEN_SEEDS_V11F)) != 0
    ):
        raise RuntimeError("v11f seed-forwarding identity changed")
    audit = {
        "schema": "eggroll-es-seed-forwarding-audit-v11f",
        "seed_count": 32,
        "incoming_seeds": incoming,
        "incoming_seed_sha256": INHERITED_SEEDS_SHA256_V11F,
        "forwarded_seeds": list(FROZEN_SEEDS_V11F),
        "forwarded_seed_sha256": FROZEN_SEEDS_SHA256_V11F,
        "all_positions_corrected": all(
            left != right for left, right in zip(incoming, FROZEN_SEEDS_V11F)
        ),
        "delegate": "run_eggroll_es_anchor_equivalence_v11c.execute_line_search",
        "passed": True,
    }
    audit["content_sha256_before_self_field"] = _canonical(audit)
    return audit


def execute_line_search_v11f(*args, **kwargs):
    kwargs = dict(kwargs)
    seed_forwarding_audit_v11f(kwargs.get("seeds", ()))
    kwargs["seeds"] = list(FROZEN_SEEDS_V11F)
    return _ORIGINAL_V11C_EXECUTE(*args, **kwargs)


@contextlib.contextmanager
def scoped_seed_forwarding_v11f():
    if driver_v11c.execute_line_search is not _ORIGINAL_V11C_EXECUTE:
        raise RuntimeError("v11f V11c execute delegate changed before patch")
    driver_v11c.execute_line_search = execute_line_search_v11f
    try:
        yield
    finally:
        driver_v11c.execute_line_search = _ORIGINAL_V11C_EXECUTE


def _patch_v11c_globals():
    prior = (driver_v11c.EXPERIMENT_NAME_V11C, driver_v11c.EXPECTED_RECIPE_SHA256_V11C)
    driver_v11c.EXPERIMENT_NAME_V11C = EXPERIMENT_NAME_V11F
    driver_v11c.EXPECTED_RECIPE_SHA256_V11C = EXPECTED_RECIPE_SHA256_V11F
    return prior


def _restore_v11c_globals(prior):
    driver_v11c.EXPERIMENT_NAME_V11C, driver_v11c.EXPECTED_RECIPE_SHA256_V11C = prior


def audit_effective_cli_v11f(argv):
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
    if effective != CANONICAL_DOWNSTREAM_V11F or normalized_outer != effective:
        raise RuntimeError("v11f effective 27-field CLI changed")
    audit = {
        "schema": "eggroll-es-effective-cli-v11f", "field_count": 27,
        "outer": normalized_outer, "effective": effective,
        "mismatch_fields": [], "base_argv_sha256": _canonical(base_argv),
        "outer_argv_sha256": _canonical(list(argv)), "passed": True,
    }
    audit["content_sha256_before_self_field"] = _canonical(audit)
    return audit


def _source_provenance_v11f():
    relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    try:
        committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("v11f launch requires committed source at HEAD") from error
    current = _file_sha256(__file__)
    if hashlib.sha256(committed).hexdigest() != current:
        raise RuntimeError("v11f source differs from committed HEAD")
    return {"schema": "eggroll-es-v11f-source", "git_head": head,
            "relative_path": relative, "file_sha256": current}


def _runtime(argv):
    argv = list(argv)
    allowed = (list(FROZEN_REAL_ARGV_V11F), [*FROZEN_REAL_ARGV_V11F, "--v11c-dry-run"])
    if argv not in allowed:
        raise ValueError("v11f requires exact frozen real or dry CLI")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v11c-dry-run", action="store_true")
    runtime, _ = parser.parse_known_args(argv)
    return runtime


def _attempt_path():
    return RUNS / f".{EXPERIMENT_NAME_V11F}.launch_attempt.json"


def _seal(payload):
    payload.pop("content_sha256_before_self_field", None)
    payload["content_sha256_before_self_field"] = _canonical(payload)


def _exclusive_write(path, payload):
    _seal(payload)
    try:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError as error:
        raise ValueError("v11f launch-attempt evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _rewrite(path, payload):
    _seal(payload)
    driver_v11c.driver_v1.atomic_write_json(path, payload)


def _baseline_scored(run_dir):
    journal = run_dir / driver_v11c.driver_v1.JOURNAL_NAME
    if not journal.exists():
        return False
    loaded = json.loads(journal.read_text())
    states = loaded.get("states") or []
    return bool(states and states[0].get("qa") is not None and states[0].get("ood_prose") is not None)


def _completed_journal_binding_v11f(run_dir):
    path = (run_dir / driver_v11c.driver_v1.JOURNAL_NAME).resolve()
    expected = (RUNS / EXPERIMENT_NAME_V11F / driver_v11c.driver_v1.JOURNAL_NAME).resolve()
    if path != expected:
        raise RuntimeError("v11f completed journal path changed")
    journal = json.loads(path.read_text())
    if _canonical(journal.get("seeds")) != FROZEN_SEEDS_SHA256_V11F:
        raise RuntimeError("v11f completed journal used the wrong seed basis")
    audit = driver_v11c.validate_completed_journal_v11c(journal)
    return {
        "schema": "eggroll-es-v11f-journal-binding",
        "path": str(path), "file_sha256": _file_sha256(path),
        "content_sha256": audit["content_sha256"],
        "journal_schema": journal.get("schema"),
        "seed_sha256": _canonical(journal.get("seeds")),
    }


def run_exact_v11f(
    argv, binding, cli_audit, seed_audit, source_provenance=None,
):
    attempt_path = _attempt_path()
    run_dir = RUNS / EXPERIMENT_NAME_V11F
    if attempt_path.exists() or run_dir.exists():
        raise ValueError("v11f requires fresh attempt and run paths")
    implementation = driver_v11e.audit_v11c_implementation_v11e()
    payload = {
        "schema": "eggroll-es-durable-launch-attempt-v11f",
        "status": "launching", "phase": "before_v11c_driver_main",
        "experiment_name": EXPERIMENT_NAME_V11F, "run_directory": str(run_dir),
        "source_provenance": (
            source_provenance or _source_provenance_v11f()
        ),
        "v11e_failure_evidence": binding, "effective_cli": cli_audit,
        "seed_forwarding_audit": seed_audit,
        "v11c_implementation": implementation,
        "intended_recipe_or_data_changed_from_v11e": False,
        "effective_perturbation_directions_corrected": True,
        "target_alpha_zero_only": True, "model_update_applied": False,
        "prior_v11e_baseline_validation_and_ood_scored": True,
        "v11f_baseline_validation_and_ood_scored": False,
        "sealed_evaluation_data_opened_or_scored": False,
    }
    _exclusive_write(attempt_path, payload)
    if run_dir.exists():
        payload.update({
            "status": "failed",
            "phase": "exclusive_claim_detected_existing_run_directory",
            "failure": {
                "type": "FreshRunReservationError",
                "message": "v11f run directory appeared after exclusive claim",
                "traceback": "",
            },
            "run_directory_exists_after_attempt": True,
            "v11c_journal_exists_after_attempt": (
                run_dir / driver_v11c.driver_v1.JOURNAL_NAME
            ).exists(),
        })
        _rewrite(attempt_path, payload)
        raise ValueError("v11f run directory appeared after exclusive claim")
    old_env = {key: os.environ.get(key) for key in DIAGNOSTIC_ENV_V11F}
    os.environ.update(DIAGNOSTIC_ENV_V11F)
    prior = _patch_v11c_globals()
    journal_binding = None
    try:
        with scoped_seed_forwarding_v11f():
            result = driver_v11c.main(list(argv))
        journal_binding = _completed_journal_binding_v11f(run_dir)
    except BaseException as error:
        payload.update({
            "status": "failed", "phase": "inside_v11c_driver_main",
            "failure": {"type": type(error).__name__, "message": str(error),
                        "traceback": traceback.format_exc()},
            "run_directory_exists_after_attempt": run_dir.exists(),
            "v11c_journal_exists_after_attempt": (
                run_dir / driver_v11c.driver_v1.JOURNAL_NAME
            ).exists(),
            "v11f_baseline_validation_and_ood_scored": _baseline_scored(run_dir),
        })
        _rewrite(attempt_path, payload)
        raise
    else:
        payload.update({
            "status": "complete", "phase": "after_v11c_driver_main",
            "run_directory_exists_after_attempt": True,
            "v11c_journal_exists_after_attempt": True,
            "v11f_baseline_validation_and_ood_scored": _baseline_scored(run_dir),
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
    binding = bind_v11e_failure_v11f()
    cli_audit = audit_effective_cli_v11f(argv)
    seed_audit = seed_forwarding_audit_v11f(INHERITED_SEEDS_V11F)
    implementation = driver_v11e.audit_v11c_implementation_v11e()
    source_provenance = _source_provenance_v11f()
    if runtime.v11c_dry_run:
        prior = _patch_v11c_globals()
        try:
            result = driver_v11c.main(argv)
        finally:
            _restore_v11c_globals(prior)
        result = copy.deepcopy(result)
        result.update({
            "schema": "eggroll-es-seed-forwarding-dry-run-v11f",
            "v11e_failure_binding_sha256": binding["binding_sha256"],
            "effective_cli": cli_audit, "seed_forwarding_audit": seed_audit,
            "v11c_implementation": implementation,
            "source_provenance": source_provenance,
        })
        print(json.dumps(result, sort_keys=True))
        return result
    return run_exact_v11f(
        argv, binding, cli_audit, seed_audit, source_provenance,
    )


if __name__ == "__main__":
    main()
