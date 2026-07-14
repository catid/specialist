#!/usr/bin/env python3
"""V11d evidence-bound retry with canonical downstream CLI forwarding."""

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
import run_eggroll_es_anchor_equivalence_v11d as driver_v11d


ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME_V11E = (
    "snapshot794_layer_v11e_middle_late_resident_sign_exact_v10_"
    "forwarded_anchor_d43d44_a43a44_basis20260714"
)
EXPECTED_V11C_RECIPE_SHA256_V11E = (
    "7edf7df5fad1208ebb22eaf38be543bbfba11c37e57016f77405729ea1e8323b"
)
V11D_FAILURE_EVIDENCE_PATH_V11E = (
    ROOT / "S6_RESIDENT_SIGN_EQUIVALENCE_V11D_FAILURE_EVIDENCE_V11E.json"
).resolve()
V11D_FAILURE_EVIDENCE_SHA256_V11E = (
    "5e75d421982e6eaa4f979692073c3018c37a5054636db611bb7f9a03cc2325b3"
)
V11D_FAILURE_CONTENT_SHA256_V11E = (
    "4f21152b42bd91b507327936fdfd16365022c22a057f48e3b546f1875ce6d1f3"
)
V11D_SOURCE_COMMIT_V11E = "25b9b2c9818f8d0ce6624f70aa39c11a7ec2666e"
V11D_SOURCE_SHA256_V11E = (
    "76862beb9957965eaf43f6629a6b5fa640e2ad8930a26eb3a51ce5c45741b675"
)
V11D_FAILURE_DOCUMENT_PATH_V11E = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_RESIDENT_SIGN_EQUIVALENCE_V11D_FAILURE.md"
).resolve()
V11D_FAILURE_DOCUMENT_SHA256_V11E = (
    "f3ff548151b9767f1a4cbfa0968d6ae69cffe2bb71d1c569b622dad534cd7353"
)
V11D_FAILURE_DOCUMENT_COMMIT_V11E = (
    "a6e4e6623d92aa35c999db29e819bfe0de6b09df"
)
DIAGNOSTIC_ENV_V11E = copy.deepcopy(driver_v11d.DIAGNOSTIC_ENV_V11D)
V11C_IMPLEMENTATION_PATHS_V11E = {
    "driver": (ROOT / "run_eggroll_es_anchor_equivalence_v11c.py").resolve(),
    "trainer": (ROOT / "train_eggroll_es_specialist_anchor_v11c.py").resolve(),
    "worker": (ROOT / "eggroll_es_worker_v11c.py").resolve(),
}
V11C_IMPLEMENTATION_SHA256_V11E = {
    "driver": "5bd650f727e3e32a0be530316c82043b012a623428fab47498f4d80f5ac48e76",
    "trainer": "c663b62f9d7990a2c59d8b46ad6258209b590bb29aa48946755a7d263a3d0799",
    "worker": "d75951483058de340185fc81f6ed050deeac1551107c0357349a1f311cdb2c22",
}
EFFECTIVE_CLI_CORRECTION_V11E = {
    "anchor_items_per_step": {"v11d_effective": 2, "v11e_effective": 128},
    "min_anchor_cosine": {"v11d_effective": 0.1, "v11e_effective": 0.8},
}

ANCHOR_JSONL_V11E = (ROOT / "data/general_prose_anchor_v1.jsonl").resolve()
ANCHOR_REPORT_V11E = (
    ROOT / "data/general_prose_anchor_v1.report.json"
).resolve()
OOD_PROSE_V11E = (ROOT / "data/ood_prose_v3.jsonl").resolve()
CANONICAL_DOWNSTREAM_V11E = {
    "model_name": str((ROOT / "models/Qwen3.6-35B-A3B").resolve()),
    "train_dataset": "/tmp/specialist-s6-candidate-guarded-ead1b21/dataset/train",
    "eval_dataset": "/tmp/specialist-s6-candidate-guarded-ead1b21/dataset/eval",
    "checkpoint": None,
    "sigma": 0.0003,
    "population_size": 32,
    "batch_size": 128,
    "mini_batch_size": 64,
    "max_tokens": 32,
    "seed": 43,
    "n_vllm_engines": 4,
    "n_gpu_per_vllm_engine": 1,
    "use_gpus": [0, 1, 2, 3],
    "eval_splits": ["validation", "ood_qa"],
    "target_alphas": [0.0],
    "anchor_prose_jsonl": str(ANCHOR_JSONL_V11E),
    "anchor_prose_report": str(ANCHOR_REPORT_V11E),
    "anchor_items_per_step": 128,
    "anchor_max_input_tokens": 512,
    "min_anchor_cosine": 0.8,
    "ood_prose_jsonl": str(OOD_PROSE_V11E),
    "ood_prose_max_input_tokens": 1024,
    "reward_function_timeout": 10,
    "output_directory": str((ROOT / "experiments/eggroll_es_hpo/runs").resolve()),
    "experiment_name": EXPERIMENT_NAME_V11E,
    "logging": "none",
    "wandb_project": "specialist-eggroll-es",
}

FROZEN_REAL_ARGV_V11E = (
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
    "--train-dataset", "/tmp/specialist-s6-candidate-guarded-ead1b21/dataset/train",
    "--eval-dataset", "/tmp/specialist-s6-candidate-guarded-ead1b21/dataset/eval",
    "--population-size", "32",
    "--batch-size", "128",
    "--mini-batch-size", "64",
    "--seed", "43",
    "--target-alphas", "0",
    "--experiment-name", EXPERIMENT_NAME_V11E,
    "--anchor-prose-jsonl", str(ANCHOR_JSONL_V11E),
    "--anchor-prose-report", str(ANCHOR_REPORT_V11E),
    "--anchor-items-per-step", "128",
    "--anchor-max-input-tokens", "512",
    "--min-anchor-cosine", "0.8",
    "--ood-prose-jsonl", str(OOD_PROSE_V11E),
    "--ood-prose-max-input-tokens", "1024",
)

FAILED_V11D_ATTEMPT_KEYS_V11E = frozenset({
    "schema", "status", "phase", "experiment_name", "run_directory",
    "run_directory_absent_before_attempt", "argv_sha256",
    "source_provenance", "v11c_failure_evidence", "v11c_recipe_sha256",
    "diagnostic_environment", "algorithm_or_data_changed_from_v11c",
    "target_alpha_zero_only", "model_update_applied",
    "sealed_data_opened_or_scored", "run_directory_exists_after_attempt",
    "v11c_journal_exists_after_attempt", "failure",
    "content_sha256_before_self_field",
})
FAILURE_KEYS_V11E = frozenset({"type", "message", "traceback"})
SOURCE_PROVENANCE_KEYS_V11E = frozenset({
    "schema", "repository_root", "relative_path", "git_head",
    "committed_blob_sha256", "driver_file_sha256",
})
COMPLETED_ATTEMPT_KEYS_V11E = frozenset({
    "schema", "status", "phase", "experiment_name", "run_directory",
    "run_directory_absent_before_attempt", "argv_sha256",
    "source_provenance", "v11d_failure_evidence", "v11c_recipe_sha256",
    "v11c_implementation", "effective_downstream_cli",
    "diagnostic_environment", "frozen_recipe_or_data_changed_from_v11d",
    "effective_cli_forwarding_corrected", "effective_cli_correction",
    "target_alpha_zero_only",
    "model_update_applied", "sealed_data_opened_or_scored",
    "run_directory_exists_after_attempt", "v11c_journal_exists_after_attempt",
    "journal_binding", "content_sha256_before_self_field",
})
JOURNAL_BINDING_KEYS_V11E = frozenset({
    "schema", "path", "file_sha256", "content_sha256", "journal_schema",
})
EFFECTIVE_AUDIT_KEYS_V11E = frozenset({
    "schema", "field_count", "outer", "effective", "mismatch_fields",
    "base_argv_sha256", "outer_argv_sha256", "parser", "passed",
    "content_sha256_before_self_field",
})
IMPLEMENTATION_AUDIT_KEYS_V11E = frozenset({
    "schema", "paths", "file_sha256", "bundle_sha256",
})
BASE_ARGV_SHA256_V11E = (
    "35155055256059b6e0a4d7e8888e12abd8fe91366cddc164c850ce125dce6ed9"
)


def _file_sha256(path):
    return driver_v11c.driver_v1.file_sha256(path)


def _canonical(value):
    return driver_v11c.driver_v1.canonical_sha256(value)


def audit_v11c_implementation_v11e():
    """Rehash the delegated runtime bundle immediately before use."""
    paths = {
        key: str(Path(path).resolve())
        for key, path in V11C_IMPLEMENTATION_PATHS_V11E.items()
    }
    file_sha256 = {
        key: _file_sha256(path)
        for key, path in V11C_IMPLEMENTATION_PATHS_V11E.items()
    }
    if (
        set(paths) != {"driver", "trainer", "worker"}
        or file_sha256 != V11C_IMPLEMENTATION_SHA256_V11E
        or file_sha256 != driver_v11d.V11C_IMPLEMENTATION_SHA256_V11D
    ):
        raise RuntimeError("v11e V11c delegated implementation bundle changed")
    return {
        "schema": "eggroll-es-v11c-implementation-bundle-v11e",
        "paths": paths,
        "file_sha256": file_sha256,
        "bundle_sha256": _canonical(file_sha256),
    }


def _offline_audit_v11e():
    return driver_v11c.driver_v11b.driver_v11.driver_v8.offline_audit


def _git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()


def validate_source_provenance_v11e(provenance):
    relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    if (
        not isinstance(provenance, dict)
        or set(provenance) != SOURCE_PROVENANCE_KEYS_V11E
        or provenance.get("schema")
        != "eggroll-es-v11e-committed-source-provenance"
        or provenance.get("repository_root") != str(ROOT)
        or provenance.get("relative_path") != relative
        or not isinstance(provenance.get("git_head"), str)
        or len(provenance["git_head"]) != 40
    ):
        raise RuntimeError("v11e source provenance shape changed")
    try:
        committed = subprocess.check_output(
            ["git", "show", f"{provenance['git_head']}:{relative}"], cwd=ROOT,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError("v11e source is not committed at recorded HEAD") from error
    committed_sha = hashlib.sha256(committed).hexdigest()
    current_sha = _file_sha256(__file__)
    if (
        provenance.get("committed_blob_sha256") != committed_sha
        or provenance.get("driver_file_sha256") != current_sha
        or committed_sha != current_sha
    ):
        raise RuntimeError("v11e source differs from its recorded commit")
    return {"git_head": provenance["git_head"], "driver_file_sha256": current_sha}


def _source_provenance_v11e():
    relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    head = _git_head()
    try:
        committed = subprocess.check_output(
            ["git", "show", f"{head}:{relative}"], cwd=ROOT,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError("v11e launch requires its driver committed at HEAD") from error
    provenance = {
        "schema": "eggroll-es-v11e-committed-source-provenance",
        "repository_root": str(ROOT),
        "relative_path": relative,
        "git_head": head,
        "committed_blob_sha256": hashlib.sha256(committed).hexdigest(),
        "driver_file_sha256": _file_sha256(__file__),
    }
    validate_source_provenance_v11e(provenance)
    return provenance


def bind_v11d_failure_v11e(path=V11D_FAILURE_EVIDENCE_PATH_V11E):
    path = Path(path).resolve()
    if path != V11D_FAILURE_EVIDENCE_PATH_V11E or _file_sha256(path) != V11D_FAILURE_EVIDENCE_SHA256_V11E:
        raise RuntimeError("v11e requires exact V11d durable failure evidence")
    document_path = V11D_FAILURE_DOCUMENT_PATH_V11E
    if _file_sha256(document_path) != V11D_FAILURE_DOCUMENT_SHA256_V11E:
        raise RuntimeError("v11e requires exact committed V11d failure document")
    try:
        committed_document = subprocess.check_output([
            "git", "show",
            f"{V11D_FAILURE_DOCUMENT_COMMIT_V11E}:"
            "experiments/eggroll_es_hpo/S6_RESIDENT_SIGN_EQUIVALENCE_V11D_FAILURE.md",
        ], cwd=ROOT)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("v11e V11d failure document commit is unavailable") from error
    if hashlib.sha256(committed_document).hexdigest() != V11D_FAILURE_DOCUMENT_SHA256_V11E:
        raise RuntimeError("v11e committed V11d failure document changed")
    document = document_path.read_text()
    required_document = (
        "v5 requires every frozen anchor document",
        "default-divergence bug",
        "defaults of 2 items per step and cosine 0.1",
        V11D_FAILURE_EVIDENCE_SHA256_V11E,
        V11D_FAILURE_CONTENT_SHA256_V11E,
        V11D_SOURCE_SHA256_V11E,
        "no model update was applied",
        "sealed data was not opened or scored",
    )
    if not all(fragment in document for fragment in required_document):
        raise RuntimeError("v11e committed V11d failure semantics changed")
    attempt = json.loads(path.read_text())
    _offline_audit_v11e()._assert_no_heldout(
        attempt, "v11e-bound V11d launch evidence",
    )
    failure = attempt.get("failure")
    source = attempt.get("source_provenance")
    expected_run = (
        driver_v11c.driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11
        / driver_v11d.EXPERIMENT_NAME_V11D
    ).resolve()
    if (
        set(attempt) != FAILED_V11D_ATTEMPT_KEYS_V11E
        or attempt.get("schema") != "eggroll-es-durable-launch-attempt-v11d"
        or attempt.get("status") != "failed"
        or attempt.get("phase") != "inside_v11c_driver_main"
        or attempt.get("experiment_name") != driver_v11d.EXPERIMENT_NAME_V11D
        or Path(attempt.get("run_directory", "")).resolve() != expected_run
        or attempt.get("argv_sha256") != _canonical(list(driver_v11d.FROZEN_REAL_ARGV_V11D))
        or attempt.get("v11c_recipe_sha256") != driver_v11d.EXPECTED_V11C_RECIPE_SHA256_V11D
        or attempt.get("v11c_failure_evidence")
        != driver_v11d.bind_v11c_failure_v11d()
        or attempt.get("diagnostic_environment") != driver_v11d.DIAGNOSTIC_ENV_V11D
        or attempt.get("algorithm_or_data_changed_from_v11c") is not False
        or attempt.get("target_alpha_zero_only") is not True
        or attempt.get("model_update_applied") is not False
        or attempt.get("sealed_data_opened_or_scored") is not False
        or attempt.get("run_directory_absent_before_attempt") is not True
        or attempt.get("run_directory_exists_after_attempt") is not True
        or attempt.get("v11c_journal_exists_after_attempt") is not False
        or not isinstance(failure, dict)
        or set(failure) != FAILURE_KEYS_V11E
        or failure.get("type") != "ValueError"
        or failure.get("message") != "v5 requires every frozen anchor document"
        or "train_eggroll_es_specialist_anchor_v5.py" not in failure.get("traceback", "")
        or "configure_anchor" not in failure.get("traceback", "")
        or attempt.get("content_sha256_before_self_field") != V11D_FAILURE_CONTENT_SHA256_V11E
        or attempt.get("content_sha256_before_self_field")
        != _canonical({key: value for key, value in attempt.items() if key != "content_sha256_before_self_field"})
    ):
        raise RuntimeError("v11e V11d failure evidence semantics changed")
    source_audit = driver_v11d.validate_source_provenance_v11d(source)
    if (
        source_audit.get("git_head") != V11D_SOURCE_COMMIT_V11E
        or source_audit.get("driver_file_sha256") != V11D_SOURCE_SHA256_V11E
    ):
        raise RuntimeError("v11e V11d committed source binding changed")
    binding = {
        "schema": "eggroll-es-v11d-cli-forwarding-failure-evidence-v11e",
        "path": str(path),
        "file_sha256": V11D_FAILURE_EVIDENCE_SHA256_V11E,
        "content_sha256": V11D_FAILURE_CONTENT_SHA256_V11E,
        "failure_document": str(document_path),
        "failure_document_sha256": V11D_FAILURE_DOCUMENT_SHA256_V11E,
        "failure_document_commit": V11D_FAILURE_DOCUMENT_COMMIT_V11E,
        "source_git_head": V11D_SOURCE_COMMIT_V11E,
        "source_driver_sha256": V11D_SOURCE_SHA256_V11E,
        "failure_type": failure["type"],
        "failure_message": failure["message"],
        "failure_phase": "post_engine_pre_journal_configure_anchor",
        "model_update_applied": False,
        "sealed_data_opened_or_scored": False,
    }
    binding["binding_sha256"] = _canonical(binding)
    return binding


def _patch_v11c_globals_v11e():
    prior = (driver_v11c.EXPERIMENT_NAME_V11C, driver_v11c.EXPECTED_RECIPE_SHA256_V11C)
    driver_v11c.EXPERIMENT_NAME_V11C = EXPERIMENT_NAME_V11E
    driver_v11c.EXPECTED_RECIPE_SHA256_V11C = EXPECTED_V11C_RECIPE_SHA256_V11E
    return prior


def _restore_v11c_globals_v11e(prior):
    driver_v11c.EXPERIMENT_NAME_V11C, driver_v11c.EXPECTED_RECIPE_SHA256_V11C = prior


def _normalize_effective_args_v11e(parsed):
    return {
        "model_name": str(Path(parsed.model_name).resolve()),
        "train_dataset": str(Path(parsed.train_dataset).resolve()),
        "eval_dataset": str(Path(parsed.eval_dataset).resolve()),
        "checkpoint": (
            str(Path(parsed.checkpoint).resolve()) if parsed.checkpoint else None
        ),
        "sigma": parsed.sigma,
        "population_size": parsed.population_size,
        "batch_size": parsed.batch_size,
        "mini_batch_size": parsed.mini_batch_size,
        "max_tokens": parsed.max_tokens,
        "seed": parsed.seed,
        "n_vllm_engines": parsed.n_vllm_engines,
        "n_gpu_per_vllm_engine": parsed.n_gpu_per_vllm_engine,
        "use_gpus": [int(value) for value in parsed.use_gpus.split(",")],
        "eval_splits": list(driver_v11c.driver_v1.validate_eval_splits(parsed.eval_splits)),
        "target_alphas": driver_v11c.driver_v1.parse_target_alphas(parsed.target_alphas),
        "anchor_prose_jsonl": str(Path(parsed.anchor_prose_jsonl).resolve()),
        "anchor_prose_report": str(Path(parsed.anchor_prose_report).resolve()),
        "anchor_items_per_step": parsed.anchor_items_per_step,
        "anchor_max_input_tokens": parsed.anchor_max_input_tokens,
        "min_anchor_cosine": parsed.min_anchor_cosine,
        "ood_prose_jsonl": str(Path(parsed.ood_prose_jsonl).resolve()),
        "ood_prose_max_input_tokens": parsed.ood_prose_max_input_tokens,
        "reward_function_timeout": parsed.reward_function_timeout,
        "output_directory": str(Path(parsed.output_directory).resolve()),
        "experiment_name": parsed.experiment_name,
        "logging": parsed.logging,
        "wandb_project": parsed.wandb_project,
    }


def _normalize_outer_execution_v11e(execution):
    return {
        "model_name": str(Path(execution["model_name"]).resolve()),
        "train_dataset": str(Path(execution["train_dataset"]).resolve()),
        "eval_dataset": str(Path(execution["eval_dataset"]).resolve()),
        "checkpoint": execution["checkpoint"],
        "sigma": execution["sigma"],
        "population_size": execution["population_size"],
        "batch_size": execution["batch_size"],
        "mini_batch_size": execution["mini_batch_size"],
        "max_tokens": execution["max_tokens"],
        "seed": execution["seed"],
        "n_vllm_engines": execution["engine_count"],
        "n_gpu_per_vllm_engine": execution["tp_per_engine"],
        "use_gpus": list(execution["gpu_ids"]),
        "eval_splits": list(execution["eval_splits"]),
        "target_alphas": list(execution["target_alphas"]),
        "anchor_prose_jsonl": str(Path(execution["anchor_prose_jsonl"]).resolve()),
        "anchor_prose_report": str(Path(execution["anchor_prose_report"]).resolve()),
        "anchor_items_per_step": execution["anchor_items_per_step"],
        "anchor_max_input_tokens": execution["anchor_max_input_tokens"],
        "min_anchor_cosine": execution["min_anchor_cosine"],
        "ood_prose_jsonl": str(Path(execution["ood_prose_jsonl"]).resolve()),
        "ood_prose_max_input_tokens": execution["ood_prose_max_input_tokens"],
        "reward_function_timeout": execution["reward_function_timeout"],
        "output_directory": str(Path(execution["output_directory"]).resolve()),
        "experiment_name": execution["experiment_name"],
        "logging": execution["logging"],
        "wandb_project": execution["wandb_project"],
    }


def inspect_downstream_projection_v11e(argv):
    """Return all 27 outer/delegated runtime fields without hiding mismatches."""
    prior = _patch_v11c_globals_v11e()
    old_argv = sys.argv
    try:
        bundle, remaining = driver_v11c.anchor_v11c.parse_frozen_layer_plan_cli_v11c(list(argv))
        execution, base_argv = driver_v11c.validate_frozen_execution_cli_v11c(remaining, bundle)
        sys.argv = [old_argv[0], *base_argv]
        parsed = driver_v11c.driver_v1.parse_args()
    finally:
        sys.argv = old_argv
        _restore_v11c_globals_v11e(prior)
    effective = _normalize_effective_args_v11e(parsed)
    outer = _normalize_outer_execution_v11e(execution)
    if set(effective) != set(outer) or set(effective) != set(CANONICAL_DOWNSTREAM_V11E):
        raise RuntimeError("v11e downstream projection field coverage changed")
    mismatches = [key for key in sorted(effective) if effective[key] != outer[key]]
    return {
        "outer": outer,
        "effective": effective,
        "mismatch_fields": mismatches,
        "field_count": len(effective),
        "base_argv": list(base_argv),
    }


def audit_effective_downstream_cli_v11e(argv):
    """Require all 27 delegated values to equal the outer frozen projection."""
    projection = inspect_downstream_projection_v11e(argv)
    effective = projection["effective"]
    outer = projection["outer"]
    if (
        projection["field_count"] != 27
        or projection["mismatch_fields"]
        or effective != CANONICAL_DOWNSTREAM_V11E
        or outer != CANONICAL_DOWNSTREAM_V11E
    ):
        raise RuntimeError("v11e effective downstream runtime projection changed")
    audit = {
        "schema": "eggroll-es-effective-downstream-cli-v11e",
        "field_count": projection["field_count"],
        "outer": outer,
        "effective": effective,
        "mismatch_fields": projection["mismatch_fields"],
        "base_argv_sha256": _canonical(projection["base_argv"]),
        "outer_argv_sha256": _canonical(list(argv)),
        "parser": str(Path(driver_v11c.driver_v1.__file__).resolve()),
        "passed": True,
    }
    audit["content_sha256_before_self_field"] = _canonical(audit)
    return audit


def _runtime_cli_v11e(argv):
    argv = list(argv)
    allowed = (list(FROZEN_REAL_ARGV_V11E), [*FROZEN_REAL_ARGV_V11E, "--v11c-dry-run"])
    if argv not in allowed:
        raise ValueError("v11e requires the exact frozen real or dry-run CLI")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment-name")
    parser.add_argument("--output-directory", default=str(driver_v11c.driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11))
    parser.add_argument("--v11c-dry-run", action="store_true")
    runtime, _ = parser.parse_known_args(argv)
    if runtime.experiment_name != EXPERIMENT_NAME_V11E:
        raise ValueError("v11e experiment name changed")
    if Path(runtime.output_directory).resolve() != driver_v11c.driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11:
        raise ValueError("v11e output directory changed")
    return runtime


def _attempt_path_v11e(runtime):
    return Path(runtime.output_directory).resolve() / f".{runtime.experiment_name}.launch_attempt.json"


def _exclusive_write_attempt_v11e(path, payload):
    path = Path(path).resolve()
    payload.pop("content_sha256_before_self_field", None)
    payload["content_sha256_before_self_field"] = _canonical(payload)
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v11e launch-attempt evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
        destination.write(raw)
        destination.flush()
        os.fsync(destination.fileno())


def _write_attempt_v11e(path, payload):
    payload.pop("content_sha256_before_self_field", None)
    payload["content_sha256_before_self_field"] = _canonical(payload)
    driver_v11c.driver_v1.atomic_write_json(path, payload)


def validate_completed_journal_v11e(journal):
    prior = _patch_v11c_globals_v11e()
    try:
        return driver_v11c.validate_completed_journal_v11c(journal)
    finally:
        _restore_v11c_globals_v11e(prior)


def _build_journal_binding_v11e(journal_path):
    journal_path = Path(journal_path).resolve()
    expected = (
        driver_v11c.driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11
        / EXPERIMENT_NAME_V11E / driver_v11c.driver_v1.JOURNAL_NAME
    ).resolve()
    if journal_path != expected:
        raise RuntimeError("v11e journal binding path changed")
    journal = json.loads(journal_path.read_text())
    # The inherited schema contains explicit false heldout sentinels and is
    # checked by its exact schema validator rather than a substring scanner.
    audit = validate_completed_journal_v11e(journal)
    return {
        "schema": "eggroll-es-v11e-journal-binding",
        "path": str(journal_path),
        "file_sha256": _file_sha256(journal_path),
        "content_sha256": audit["content_sha256"],
        "journal_schema": journal.get("schema"),
    }


def run_exact_retry_v11e(argv, attempt_path, failure_binding, effective_audit):
    attempt_path = Path(attempt_path).resolve()
    if attempt_path.exists():
        raise ValueError("v11e launch-attempt evidence already exists")
    run_dir = (
        driver_v11c.driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11
        / EXPERIMENT_NAME_V11E
    ).resolve()
    if run_dir.exists():
        raise ValueError("v11e run directory already exists")
    implementation_audit = audit_v11c_implementation_v11e()
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "eggroll-es-durable-launch-attempt-v11e",
        "status": "launching",
        "phase": "before_v11c_driver_main",
        "experiment_name": EXPERIMENT_NAME_V11E,
        "run_directory": str(run_dir),
        "run_directory_absent_before_attempt": True,
        "argv_sha256": _canonical(list(FROZEN_REAL_ARGV_V11E)),
        "source_provenance": _source_provenance_v11e(),
        "v11d_failure_evidence": copy.deepcopy(failure_binding),
        "v11c_recipe_sha256": EXPECTED_V11C_RECIPE_SHA256_V11E,
        "v11c_implementation": implementation_audit,
        "effective_downstream_cli": copy.deepcopy(effective_audit),
        "diagnostic_environment": copy.deepcopy(DIAGNOSTIC_ENV_V11E),
        "frozen_recipe_or_data_changed_from_v11d": False,
        "effective_cli_forwarding_corrected": True,
        "effective_cli_correction": copy.deepcopy(
            EFFECTIVE_CLI_CORRECTION_V11E
        ),
        "target_alpha_zero_only": True,
        "model_update_applied": False,
        "sealed_data_opened_or_scored": False,
    }
    _exclusive_write_attempt_v11e(attempt_path, payload)
    if run_dir.exists():
        payload.update({
            "status": "failed",
            "phase": "exclusive_claim_detected_existing_run_directory",
            "failure": {"type": "FreshRunReservationError", "message": "v11e run directory appeared after exclusive claim", "traceback": ""},
            "run_directory_exists_after_attempt": True,
            "v11c_journal_exists_after_attempt": (run_dir / driver_v11c.driver_v1.JOURNAL_NAME).exists(),
        })
        _write_attempt_v11e(attempt_path, payload)
        raise ValueError("v11e run directory appeared after exclusive claim")
    old_env = {key: os.environ.get(key) for key in DIAGNOSTIC_ENV_V11E}
    os.environ.update(DIAGNOSTIC_ENV_V11E)
    journal_binding = None
    try:
        result = driver_v11c.main(list(argv))
        journal_binding = _build_journal_binding_v11e(run_dir / driver_v11c.driver_v1.JOURNAL_NAME)
    except BaseException as error:
        payload.update({
            "status": "failed", "phase": "inside_v11c_driver_main",
            "failure": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
            "run_directory_exists_after_attempt": run_dir.exists(),
            "v11c_journal_exists_after_attempt": (run_dir / driver_v11c.driver_v1.JOURNAL_NAME).exists(),
        })
        if journal_binding is not None:
            payload["journal_binding"] = journal_binding
        _write_attempt_v11e(attempt_path, payload)
        raise
    else:
        payload.update({
            "status": "complete", "phase": "after_v11c_driver_main",
            "run_directory_exists_after_attempt": run_dir.exists(),
            "v11c_journal_exists_after_attempt": True,
            "journal_binding": journal_binding,
        })
        _write_attempt_v11e(attempt_path, payload)
        return result
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate_launch_attempt_v11e(attempt, expected_journal_path=None):
    if not isinstance(attempt, dict):
        raise RuntimeError("v11e completed launch-attempt evidence changed")
    _offline_audit_v11e()._assert_no_heldout(attempt, "v11e launch-attempt object")
    validate_source_provenance_v11e(attempt.get("source_provenance"))
    failure = bind_v11d_failure_v11e()
    fresh_implementation = audit_v11c_implementation_v11e()
    fresh_effective = audit_effective_downstream_cli_v11e(
        FROZEN_REAL_ARGV_V11E
    )
    expected_run = (
        driver_v11c.driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11
        / EXPERIMENT_NAME_V11E
    ).resolve()
    journal_binding = attempt.get("journal_binding")
    effective = attempt.get("effective_downstream_cli")
    if (
        set(attempt) != COMPLETED_ATTEMPT_KEYS_V11E
        or attempt.get("schema") != "eggroll-es-durable-launch-attempt-v11e"
        or attempt.get("status") != "complete"
        or attempt.get("phase") != "after_v11c_driver_main"
        or attempt.get("experiment_name") != EXPERIMENT_NAME_V11E
        or Path(attempt.get("run_directory", "")).resolve() != expected_run
        or attempt.get("argv_sha256") != _canonical(list(FROZEN_REAL_ARGV_V11E))
        or attempt.get("v11d_failure_evidence") != failure
        or attempt.get("v11c_recipe_sha256") != EXPECTED_V11C_RECIPE_SHA256_V11E
        or not isinstance(attempt.get("v11c_implementation"), dict)
        or set(attempt["v11c_implementation"])
        != IMPLEMENTATION_AUDIT_KEYS_V11E
        or attempt.get("v11c_implementation") != fresh_implementation
        or attempt.get("diagnostic_environment") != DIAGNOSTIC_ENV_V11E
        or attempt.get("frozen_recipe_or_data_changed_from_v11d") is not False
        or attempt.get("effective_cli_forwarding_corrected") is not True
        or attempt.get("effective_cli_correction")
        != EFFECTIVE_CLI_CORRECTION_V11E
        or attempt.get("target_alpha_zero_only") is not True
        or attempt.get("model_update_applied") is not False
        or attempt.get("sealed_data_opened_or_scored") is not False
        or attempt.get("run_directory_absent_before_attempt") is not True
        or attempt.get("run_directory_exists_after_attempt") is not True
        or attempt.get("v11c_journal_exists_after_attempt") is not True
        or not isinstance(effective, dict)
        or set(effective) != EFFECTIVE_AUDIT_KEYS_V11E
        or effective != fresh_effective
        or effective.get("schema") != "eggroll-es-effective-downstream-cli-v11e"
        or effective.get("field_count") != 27
        or effective.get("outer") != CANONICAL_DOWNSTREAM_V11E
        or effective.get("effective") != CANONICAL_DOWNSTREAM_V11E
        or effective.get("mismatch_fields") != []
        or effective.get("base_argv_sha256") != BASE_ARGV_SHA256_V11E
        or effective.get("outer_argv_sha256") != _canonical(list(FROZEN_REAL_ARGV_V11E))
        or Path(effective.get("parser", "")).resolve()
        != Path(driver_v11c.driver_v1.__file__).resolve()
        or effective.get("passed") is not True
        or effective.get("content_sha256_before_self_field")
        != _canonical({key: value for key, value in effective.items() if key != "content_sha256_before_self_field"})
        or not isinstance(journal_binding, dict)
        or set(journal_binding) != JOURNAL_BINDING_KEYS_V11E
        or attempt.get("content_sha256_before_self_field")
        != _canonical({key: value for key, value in attempt.items() if key != "content_sha256_before_self_field"})
    ):
        raise RuntimeError("v11e completed launch-attempt evidence changed")
    expected_journal = (expected_run / driver_v11c.driver_v1.JOURNAL_NAME).resolve()
    if expected_journal_path is not None and Path(expected_journal_path).resolve() != expected_journal:
        raise RuntimeError("v11e reporter journal path changed")
    if Path(journal_binding["path"]).resolve() != expected_journal:
        raise RuntimeError("v11e attempt is bound to the wrong journal")
    if journal_binding != _build_journal_binding_v11e(expected_journal):
        raise RuntimeError("v11e attempt/journal cryptographic binding changed")
    return {
        "content_sha256": attempt["content_sha256_before_self_field"],
        "v11d_failure_binding_sha256": failure["binding_sha256"],
        "effective_downstream_cli_sha256": effective["content_sha256_before_self_field"],
        "v11c_implementation_bundle_sha256": fresh_implementation[
            "bundle_sha256"
        ],
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    runtime = _runtime_cli_v11e(argv)
    failure = bind_v11d_failure_v11e()
    effective = audit_effective_downstream_cli_v11e(argv)
    implementation = audit_v11c_implementation_v11e()
    prior = _patch_v11c_globals_v11e()
    try:
        if runtime.v11c_dry_run:
            result = driver_v11c.main(argv)
            if result.get("recipe_sha256") != EXPECTED_V11C_RECIPE_SHA256_V11E:
                raise RuntimeError("v11e dry-run recipe identity changed")
            result = copy.deepcopy(result)
            result["schema"] = "eggroll-es-forwarded-anchor-dry-run-v11e"
            result["v11d_failure_binding_sha256"] = failure["binding_sha256"]
            result["effective_downstream_cli"] = effective
            result["v11c_implementation"] = implementation
            result["frozen_recipe_or_data_changed_from_v11d"] = False
            result["effective_cli_forwarding_corrected"] = True
            result["effective_cli_correction"] = copy.deepcopy(
                EFFECTIVE_CLI_CORRECTION_V11E
            )
            print(json.dumps(result, sort_keys=True))
            return result
        return run_exact_retry_v11e(argv, _attempt_path_v11e(runtime), failure, effective)
    finally:
        _restore_v11c_globals_v11e(prior)


if __name__ == "__main__":
    main()
