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
        "heldout_opened_or_scored": False,
        "v11c_implementation": copy.deepcopy(
            V11C_IMPLEMENTATION_SHA256_V11D
        ),
    }
    binding["binding_sha256"] = driver_v11c.driver_v1.canonical_sha256(
        binding
    )
    return binding


def _runtime_cli_v11d(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment-name")
    parser.add_argument(
        "--output-directory",
        default=str(driver_v11c.driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11),
    )
    parser.add_argument("--v11c-dry-run", action="store_true")
    runtime, _ = parser.parse_known_args(list(argv))
    if runtime.experiment_name != EXPERIMENT_NAME_V11D:
        raise ValueError("v11d requires the exact fresh experiment name")
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
        "argv_sha256": driver_v11c.driver_v1.canonical_sha256(list(argv)),
        "source_provenance": _source_provenance_v11d(),
        "v11c_failure_evidence": copy.deepcopy(failure_binding),
        "v11c_recipe_sha256": EXPECTED_V11C_RECIPE_SHA256_V11D,
        "diagnostic_environment": copy.deepcopy(DIAGNOSTIC_ENV_V11D),
        "algorithm_or_data_changed_from_v11c": False,
        "target_alpha_zero_only": True,
        "model_update_applied": False,
        "heldout_opened_or_scored": False,
    }
    _write_attempt_v11d(attempt_path, payload)
    old_env = {key: os.environ.get(key) for key in DIAGNOSTIC_ENV_V11D}
    os.environ.update(DIAGNOSTIC_ENV_V11D)
    try:
        result = driver_v11c.main(list(argv))
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


def validate_launch_attempt_v11d(attempt):
    failure = bind_v11c_failure_v11d()
    if not isinstance(attempt, dict):
        raise RuntimeError("v11d completed launch-attempt evidence changed")
    validate_source_provenance_v11d(attempt.get("source_provenance"))
    if (
        attempt.get("schema") != "eggroll-es-durable-launch-attempt-v11d"
        or attempt.get("status") != "complete"
        or attempt.get("phase") != "after_v11c_driver_main"
        or attempt.get("experiment_name") != EXPERIMENT_NAME_V11D
        or attempt.get("v11c_recipe_sha256")
        != EXPECTED_V11C_RECIPE_SHA256_V11D
        or attempt.get("v11c_failure_evidence") != failure
        or attempt.get("diagnostic_environment") != DIAGNOSTIC_ENV_V11D
        or attempt.get("algorithm_or_data_changed_from_v11c") is not False
        or attempt.get("target_alpha_zero_only") is not True
        or attempt.get("model_update_applied") is not False
        or attempt.get("heldout_opened_or_scored") is not False
        or attempt.get("run_directory_absent_before_attempt") is not True
        or attempt.get("run_directory_exists_after_attempt") is not True
        or attempt.get("v11c_journal_exists_after_attempt") is not True
        or attempt.get("content_sha256_before_self_field")
        != driver_v11c.driver_v1.canonical_sha256({
            key: value for key, value in attempt.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v11d completed launch-attempt evidence changed")
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
