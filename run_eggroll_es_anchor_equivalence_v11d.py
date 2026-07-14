#!/usr/bin/env python3
"""Evidence-bound V11c retry with durable pre-journal launch telemetry."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import run_eggroll_es_anchor_equivalence_v11c as driver_v11c


ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME_V11D = (
    "snapshot794_layer_v11d_middle_late_resident_sign_exact_v10_"
    "durable_launch_d43d44_a43a44_basis20260714"
)
EXPECTED_V11C_RECIPE_SHA256_V11D = (
    "3e91bc82ef50f528cbae4925931e5f2aee5b9c63e470c489dd11814b5df9f6a8"
)
V11C_FAILURE_PATH_V11D = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_RESIDENT_SIGN_EQUIVALENCE_V11C_FAILURE.md"
).resolve()
V11C_FAILURE_SHA256_V11D = (
    "49c79c4a982c5c37ab56b71b141771846edeabb267a13c255856acccd023b88c"
)
V11C_FAILURE_COMMIT_V11D = (
    "320298650d871102122c663d3115123bf4940271"
)
V11C_IMPLEMENTATION_SHA256_V11D = {
    "driver": "5bd650f727e3e32a0be530316c82043b012a623428fab47498f4d80f5ac48e76",
    "trainer": "c663b62f9d7990a2c59d8b46ad6258209b590bb29aa48946755a7d263a3d0799",
    "worker": "d75951483058de340185fc81f6ed050deeac1551107c0357349a1f311cdb2c22",
}
DIAGNOSTIC_ENV_V11D = {
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,NET",
    "RAY_DEDUP_LOGS": "0",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
}
FROZEN_REAL_ARGV_V11D = (
    "--layer-plan-json",
    str((ROOT / "experiments/layer_plans/middle_late_dense_v6.json").resolve()),
    "--expected-layer-plan-file-sha256",
    "d65d702969dcec7a56ca4fcf461d402c44642966191a57c2ef092ec339e3e3df",
    "--expected-layer-plan-sha256",
    "03745c603a6b48898b41afbd4d9121aef276d7e45ca1a3ae14607ec5d1042cb9",
    "--expected-model-config-sha256",
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
    "--v11c-stage", "equivalence_api_retry",
    "--v11c-v10-report",
    str((ROOT / "experiments/eggroll_es_hpo/S6_ANTITHETIC_CROSSED_V10_REPORT.json").resolve()),
    "--v11c-failed-v11-journal",
    str((ROOT / "experiments/eggroll_es_hpo/runs/snapshot794_layer_v11_middle_late_resident_sign_exact_v10_d43d44_a43a44_basis20260714/alpha_line_search.json").resolve()),
    "--v11c-v11b-failure-evidence",
    str((ROOT / "experiments/eggroll_es_hpo/S6_RESIDENT_SIGN_EQUIVALENCE_V11B_FAILURE.md").resolve()),
    "--v11c-perturbation-basis-seed", "20260714",
    "--train-dataset",
    "/tmp/specialist-s6-candidate-guarded-ead1b21/dataset/train",
    "--eval-dataset",
    "/tmp/specialist-s6-candidate-guarded-ead1b21/dataset/eval",
    "--population-size", "32",
    "--batch-size", "128",
    "--mini-batch-size", "64",
    "--seed", "43",
    "--target-alphas", "0",
    "--experiment-name", EXPERIMENT_NAME_V11D,
)
COMPLETED_ATTEMPT_KEYS_V11D = frozenset({
    "schema", "status", "phase", "experiment_name", "run_directory",
    "run_directory_absent_before_attempt", "argv_sha256",
    "source_provenance", "v11c_failure_evidence", "v11c_recipe_sha256",
    "diagnostic_environment", "algorithm_or_data_changed_from_v11c",
    "target_alpha_zero_only", "model_update_applied",
    "sealed_data_opened_or_scored", "run_directory_exists_after_attempt",
    "v11c_journal_exists_after_attempt", "journal_binding",
    "content_sha256_before_self_field",
})
JOURNAL_BINDING_KEYS_V11D = frozenset({
    "schema", "path", "file_sha256", "content_sha256", "journal_schema",
})
SOURCE_PROVENANCE_KEYS_V11D = frozenset({
    "schema", "repository_root", "relative_path", "git_head",
    "committed_blob_sha256", "driver_file_sha256",
})


def _file_sha256(path):
    return driver_v11c.driver_v1.file_sha256(path)


def _git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()


def validate_source_provenance_v11d(provenance):
    relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    if (
        not isinstance(provenance, dict)
        or set(provenance) != SOURCE_PROVENANCE_KEYS_V11D
        or provenance.get("schema")
        != "eggroll-es-v11d-committed-source-provenance"
        or provenance.get("repository_root") != str(ROOT)
        or provenance.get("relative_path") != relative
        or not isinstance(provenance.get("git_head"), str)
        or len(provenance["git_head"]) != 40
    ):
        raise RuntimeError("v11d source provenance shape changed")
    try:
        committed = subprocess.check_output(
            ["git", "show", f"{provenance['git_head']}:{relative}"],
            cwd=ROOT,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError("v11d source is not committed at recorded HEAD") from error
    committed_sha = hashlib.sha256(committed).hexdigest()
    current_sha = _file_sha256(__file__)
    if (
        provenance.get("committed_blob_sha256") != committed_sha
        or provenance.get("driver_file_sha256") != current_sha
        or committed_sha != current_sha
    ):
        raise RuntimeError("v11d source differs from its recorded commit")
    return {
        "git_head": provenance["git_head"],
        "driver_file_sha256": current_sha,
    }


def _source_provenance_v11d():
    relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    head = _git_head()
    try:
        committed = subprocess.check_output(
            ["git", "show", f"{head}:{relative}"], cwd=ROOT,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError("v11d launch requires its driver committed at HEAD") from error
    provenance = {
        "schema": "eggroll-es-v11d-committed-source-provenance",
        "repository_root": str(ROOT),
        "relative_path": relative,
        "git_head": head,
        "committed_blob_sha256": hashlib.sha256(committed).hexdigest(),
        "driver_file_sha256": _file_sha256(__file__),
    }
    validate_source_provenance_v11d(provenance)
    return provenance


def bind_v11c_failure_v11d(path=V11C_FAILURE_PATH_V11D):
    path = Path(path).resolve()
    if (
        path != V11C_FAILURE_PATH_V11D
        or _file_sha256(path) != V11C_FAILURE_SHA256_V11D
    ):
        raise RuntimeError("v11d requires exact committed V11c failure evidence")
    text = path.read_text()
    required = (
        "post-engine/pre-journal launch failure",
        "does **not** localize the failure to rendezvous",
        "cannot distinguish inter-engine group",
        "No parameter update was prepared",
        "sealed-heldout content were not scored",
        V11C_IMPLEMENTATION_SHA256_V11D["driver"],
        V11C_IMPLEMENTATION_SHA256_V11D["trainer"],
        V11C_IMPLEMENTATION_SHA256_V11D["worker"],
    )
    if not all(fragment in text for fragment in required):
        raise RuntimeError("v11d V11c failure evidence semantics changed")
    binding = {
        "schema": "eggroll-es-v11c-post-engine-failure-evidence-v11d",
        "path": str(path),
        "file_sha256": V11C_FAILURE_SHA256_V11D,
        "commit": V11C_FAILURE_COMMIT_V11D,
        "failure_phase": "post_engine_load_pre_journal_unknown",
        "model_loaded_on_all_four_gpus": True,
        "journal_created": False,
        "model_update_applied": False,
        "sealed_data_opened_or_scored": False,
        "v11c_implementation": copy.deepcopy(
            V11C_IMPLEMENTATION_SHA256_V11D
        ),
    }
    binding["binding_sha256"] = driver_v11c.driver_v1.canonical_sha256(
        binding
    )
    return binding


def _runtime_cli_v11d(argv):
    argv = list(argv)
    allowed = (
        list(FROZEN_REAL_ARGV_V11D),
        [*FROZEN_REAL_ARGV_V11D, "--v11c-dry-run"],
    )
    if argv not in allowed:
        raise ValueError("v11d requires the exact frozen real or dry-run CLI")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment-name")
    parser.add_argument(
        "--output-directory",
        default=str(driver_v11c.driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11),
    )
    parser.add_argument("--v11c-dry-run", action="store_true")
    runtime, _ = parser.parse_known_args(argv)
    if (
        Path(runtime.output_directory).resolve()
        != driver_v11c.driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11
    ):
        raise ValueError("v11d output directory changed")
    return runtime


def _attempt_path_v11d(runtime):
    output = Path(runtime.output_directory).resolve()
    return output / f".{runtime.experiment_name}.launch_attempt.json"


def _patch_v11c_retry_globals_v11d():
    prior = (
        driver_v11c.EXPERIMENT_NAME_V11C,
        driver_v11c.EXPECTED_RECIPE_SHA256_V11C,
    )
    driver_v11c.EXPERIMENT_NAME_V11C = EXPERIMENT_NAME_V11D
    driver_v11c.EXPECTED_RECIPE_SHA256_V11C = (
        EXPECTED_V11C_RECIPE_SHA256_V11D
    )
    return prior


def _restore_v11c_retry_globals_v11d(prior):
    (
        driver_v11c.EXPERIMENT_NAME_V11C,
        driver_v11c.EXPECTED_RECIPE_SHA256_V11C,
    ) = prior


def run_exact_retry_v11d(argv, attempt_path, failure_binding):
    attempt_path = Path(attempt_path).resolve()
    if attempt_path.exists():
        raise ValueError("v11d launch-attempt evidence already exists")
    run_dir = (
        driver_v11c.driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11
        / EXPERIMENT_NAME_V11D
    )
    if run_dir.exists():
        raise ValueError("v11d run directory already exists")
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "eggroll-es-durable-launch-attempt-v11d",
        "status": "launching",
        "phase": "before_v11c_driver_main",
        "experiment_name": EXPERIMENT_NAME_V11D,
        "run_directory": str(run_dir),
        "run_directory_absent_before_attempt": True,
        "argv_sha256": driver_v11c.driver_v1.canonical_sha256(
            list(FROZEN_REAL_ARGV_V11D)
        ),
        "source_provenance": _source_provenance_v11d(),
        "v11c_failure_evidence": copy.deepcopy(failure_binding),
        "v11c_recipe_sha256": EXPECTED_V11C_RECIPE_SHA256_V11D,
        "diagnostic_environment": copy.deepcopy(DIAGNOSTIC_ENV_V11D),
        "algorithm_or_data_changed_from_v11c": False,
        "target_alpha_zero_only": True,
        "model_update_applied": False,
        "sealed_data_opened_or_scored": False,
    }
    _exclusive_write_attempt_v11d(attempt_path, payload)
    if run_dir.exists():
        payload.update({
            "status": "failed",
            "phase": "exclusive_claim_detected_existing_run_directory",
            "failure": {
                "type": "FreshRunReservationError",
                "message": "v11d run directory appeared after exclusive claim",
                "traceback": "",
            },
            "run_directory_exists_after_attempt": True,
            "v11c_journal_exists_after_attempt": (
                run_dir / driver_v11c.driver_v1.JOURNAL_NAME
            ).exists(),
        })
        _write_attempt_v11d(attempt_path, payload)
        raise ValueError("v11d run directory appeared after exclusive claim")
    old_env = {key: os.environ.get(key) for key in DIAGNOSTIC_ENV_V11D}
    os.environ.update(DIAGNOSTIC_ENV_V11D)
    journal_binding = None
    try:
        result = driver_v11c.main(list(argv))
        journal_binding = _build_journal_binding_v11d(
            run_dir / driver_v11c.driver_v1.JOURNAL_NAME
        )
    except BaseException as error:
        payload.update({
            "status": "failed",
            "phase": "inside_v11c_driver_main",
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "run_directory_exists_after_attempt": run_dir.exists(),
            "v11c_journal_exists_after_attempt": (
                run_dir / driver_v11c.driver_v1.JOURNAL_NAME
            ).exists(),
        })
        if journal_binding is not None:
            payload["journal_binding"] = journal_binding
        _write_attempt_v11d(attempt_path, payload)
        raise
    else:
        payload.update({
            "status": "complete",
            "phase": "after_v11c_driver_main",
            "run_directory_exists_after_attempt": run_dir.exists(),
            "v11c_journal_exists_after_attempt": (
                run_dir / driver_v11c.driver_v1.JOURNAL_NAME
            ).exists(),
            "journal_binding": journal_binding,
        })
        _write_attempt_v11d(attempt_path, payload)
        return result
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_attempt_v11d(path, payload):
    payload.pop("content_sha256_before_self_field", None)
    payload["content_sha256_before_self_field"] = (
        driver_v11c.driver_v1.canonical_sha256(payload)
    )
    driver_v11c.driver_v1.atomic_write_json(path, payload)


def _exclusive_write_attempt_v11d(path, payload):
    path = Path(path).resolve()
    payload.pop("content_sha256_before_self_field", None)
    payload["content_sha256_before_self_field"] = (
        driver_v11c.driver_v1.canonical_sha256(payload)
    )
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
        )
    except FileExistsError as error:
        raise ValueError("v11d launch-attempt evidence already exists") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(raw)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        # Keep even a partial exclusive claim: an I/O failure must never make
        # the same experiment name silently retryable.
        raise


def _offline_audit_v11d():
    return driver_v11c.driver_v11b.driver_v11.driver_v8.offline_audit


def _build_journal_binding_v11d(journal_path):
    journal_path = Path(journal_path).resolve()
    expected = (
        driver_v11c.driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11
        / EXPERIMENT_NAME_V11D / driver_v11c.driver_v1.JOURNAL_NAME
    ).resolve()
    if journal_path != expected:
        raise RuntimeError("v11d journal binding path changed")
    journal = json.loads(journal_path.read_text())
    # The inherited, frozen journal schema deliberately contains policy
    # sentinels such as ``ood_validation_heldout_as_objective: false``.  Its
    # schema-aware validator below enforces those exact false values; a generic
    # substring scan would reject valid evidence before it could be validated.
    audit = validate_completed_journal_v11d(journal)
    return {
        "schema": "eggroll-es-v11d-journal-binding",
        "path": str(journal_path),
        "file_sha256": _file_sha256(journal_path),
        "content_sha256": audit["content_sha256"],
        "journal_schema": journal.get("schema"),
    }


def validate_launch_attempt_v11d(attempt, expected_journal_path=None):
    failure = bind_v11c_failure_v11d()
    if not isinstance(attempt, dict):
        raise RuntimeError("v11d completed launch-attempt evidence changed")
    _offline_audit_v11d()._assert_no_heldout(
        attempt, "v11d launch-attempt object",
    )
    validate_source_provenance_v11d(attempt.get("source_provenance"))
    expected_run = (
        driver_v11c.driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11
        / EXPERIMENT_NAME_V11D
    ).resolve()
    expected_argv_sha = driver_v11c.driver_v1.canonical_sha256(
        list(FROZEN_REAL_ARGV_V11D)
    )
    journal_binding = attempt.get("journal_binding")
    if (
        set(attempt) != COMPLETED_ATTEMPT_KEYS_V11D
        or attempt.get("schema") != "eggroll-es-durable-launch-attempt-v11d"
        or attempt.get("status") != "complete"
        or attempt.get("phase") != "after_v11c_driver_main"
        or attempt.get("experiment_name") != EXPERIMENT_NAME_V11D
        or Path(attempt.get("run_directory", "")).resolve() != expected_run
        or attempt.get("argv_sha256") != expected_argv_sha
        or attempt.get("v11c_recipe_sha256")
        != EXPECTED_V11C_RECIPE_SHA256_V11D
        or attempt.get("v11c_failure_evidence") != failure
        or attempt.get("diagnostic_environment") != DIAGNOSTIC_ENV_V11D
        or attempt.get("algorithm_or_data_changed_from_v11c") is not False
        or attempt.get("target_alpha_zero_only") is not True
        or attempt.get("model_update_applied") is not False
        or attempt.get("sealed_data_opened_or_scored") is not False
        or attempt.get("run_directory_absent_before_attempt") is not True
        or attempt.get("run_directory_exists_after_attempt") is not True
        or attempt.get("v11c_journal_exists_after_attempt") is not True
        or not isinstance(journal_binding, dict)
        or set(journal_binding) != JOURNAL_BINDING_KEYS_V11D
        or attempt.get("content_sha256_before_self_field")
        != driver_v11c.driver_v1.canonical_sha256({
            key: value for key, value in attempt.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v11d completed launch-attempt evidence changed")
    recorded_journal = Path(journal_binding["path"]).resolve()
    expected_journal = (
        expected_run / driver_v11c.driver_v1.JOURNAL_NAME
    ).resolve()
    if expected_journal_path is not None:
        supplied_journal = Path(expected_journal_path).resolve()
        if supplied_journal != expected_journal:
            raise RuntimeError("v11d reporter journal path changed")
    if recorded_journal != expected_journal:
        raise RuntimeError("v11d attempt is bound to the wrong journal")
    current_binding = _build_journal_binding_v11d(recorded_journal)
    if journal_binding != current_binding:
        raise RuntimeError("v11d attempt/journal cryptographic binding changed")
    return {
        "content_sha256": attempt["content_sha256_before_self_field"],
        "v11c_failure_binding_sha256": failure["binding_sha256"],
    }


def validate_completed_journal_v11d(journal):
    prior = _patch_v11c_retry_globals_v11d()
    try:
        return driver_v11c.validate_completed_journal_v11c(journal)
    finally:
        _restore_v11c_retry_globals_v11d(prior)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    runtime = _runtime_cli_v11d(argv)
    failure = bind_v11c_failure_v11d()
    prior = _patch_v11c_retry_globals_v11d()
    try:
        if runtime.v11c_dry_run:
            result = driver_v11c.main(argv)
            if result.get("recipe_sha256") != EXPECTED_V11C_RECIPE_SHA256_V11D:
                raise RuntimeError("v11d dry-run recipe identity changed")
            result = copy.deepcopy(result)
            result["schema"] = "eggroll-es-durable-launch-dry-run-v11d"
            result["v11c_failure_binding_sha256"] = failure["binding_sha256"]
            print(json.dumps(result, sort_keys=True))
            return result
        return run_exact_retry_v11d(
            argv, _attempt_path_v11d(runtime), failure,
        )
    finally:
        _restore_v11c_retry_globals_v11d(prior)


if __name__ == "__main__":
    main()
