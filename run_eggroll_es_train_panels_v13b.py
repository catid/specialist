#!/usr/bin/env python3
"""Fresh V13 retry restoring the inherited middle-late runtime contract."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import sys
from pathlib import Path

import run_eggroll_es_train_panels_v13 as driver_v13


ROOT = Path(__file__).resolve().parent
RUNS = driver_v13.FROZEN_OUTPUT_DIRECTORY_V13
EXPERIMENT_NAME_V13B = (
    "snapshot794_layer_v13b_document_balanced_five_panel_alpha_zero_"
    "runtime_forwarded_resident_sign_basis20260714"
)
FAILED_ATTEMPT_V13B = (
    RUNS
    / ".snapshot794_layer_v13_document_balanced_five_panel_alpha_zero_"
    "resident_sign_basis20260714.launch_attempt.json"
).resolve()
FAILED_ATTEMPT_FILE_SHA256_V13B = (
    "c28a4190c559073a65d4cf0c50f55993d87b65cb7cf159b61df30bf804a130de"
)
FAILED_ATTEMPT_CONTENT_SHA256_V13B = (
    "b68b7bf5e2d656703a55a256591dafdde23f7e7e167935b5c03a170a378c5c96"
)
FAILED_SOURCE_COMMIT_V13B = "98e30b9e1ffccc588314cf284ae31b48bc001d2c"
FAILED_IMPLEMENTATION_BUNDLE_V13B = (
    "7e2eeab3aac0de7b15b98bfa1645a50c926a051269862620b681f99360ad66a7"
)
FAILED_RECIPE_CONTENT_SHA256_V13B = (
    "ec46d7c58e6825b2bea964baa4fd9241072d01a1290f98a850ae6f5ab9234d54"
)
PROTOCOL_PATH_V13B = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_DOCUMENT_BALANCED_TRAIN_PANELS_V13B_RETRY_PROTOCOL.md"
).resolve()
TEST_PATH_V13B = (ROOT / "test_eggroll_es_train_panels_v13b.py").resolve()

_BASE_IMPLEMENTATION_IDENTITY_V13 = driver_v13.implementation_identity_v13
_BASE_RECIPE_V13 = driver_v13.recipe_v13
_BASE_EXPERIMENT_NAME_V13 = driver_v13.EXPERIMENT_NAME_V13
_BASE_IMPLEMENTATION_PATHS_V13 = copy.deepcopy(
    driver_v13.IMPLEMENTATION_PATHS_V13
)


def _canonical(value):
    return driver_v13._canonical(value)


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_failed_v13_attempt_v13b(path=FAILED_ATTEMPT_V13B):
    path = Path(path).resolve()
    if path != FAILED_ATTEMPT_V13B:
        raise RuntimeError("v13b requires the canonical failed V13 attempt")
    if _file_sha256(path) != FAILED_ATTEMPT_FILE_SHA256_V13B:
        raise RuntimeError("v13b failed-attempt file identity changed")
    attempt = json.loads(path.read_text())
    expected_keys = {
        "schema", "status", "phase", "experiment_name", "run_directory",
        "source_provenance", "recipe", "target_alpha_zero_only",
        "model_update_applied", "sealed_or_nontrain_surface_opened",
        "failure", "run_directory_exists_after_attempt",
        "report_exists_after_attempt", "content_sha256_before_self_field",
    }
    source = attempt.get("source_provenance", {})
    recipe = attempt.get("recipe", {})
    failure = attempt.get("failure", {})
    if (
        set(attempt) != expected_keys
        or attempt.get("schema") != "eggroll-es-durable-launch-attempt-v13"
        or attempt.get("status") != "failed"
        or attempt.get("phase") != "inside_train_panel_diagnostic"
        or attempt.get("experiment_name") != _BASE_EXPERIMENT_NAME_V13
        or attempt.get("target_alpha_zero_only") is not True
        or attempt.get("model_update_applied") is not False
        or attempt.get("sealed_or_nontrain_surface_opened") is not False
        or attempt.get("run_directory_exists_after_attempt") is not True
        or attempt.get("report_exists_after_attempt") is not False
        or failure.get("type") != "RuntimeError"
        or failure.get("message")
        != "v4 controller has no frozen runtime expectation for this plan"
        or source.get("git_head") != FAILED_SOURCE_COMMIT_V13B
        or source.get("implementation_bundle_sha256")
        != FAILED_IMPLEMENTATION_BUNDLE_V13B
        or recipe.get("content_sha256_before_self_field")
        != FAILED_RECIPE_CONTENT_SHA256_V13B
        or attempt.get("content_sha256_before_self_field")
        != FAILED_ATTEMPT_CONTENT_SHA256_V13B
        or attempt.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in attempt.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v13b failed-attempt semantics changed")
    binding = {
        "schema": "eggroll-es-v13-runtime-forwarding-failure-binding-v13b",
        "path": str(path),
        "file_sha256": FAILED_ATTEMPT_FILE_SHA256_V13B,
        "content_sha256": FAILED_ATTEMPT_CONTENT_SHA256_V13B,
        "source_commit": FAILED_SOURCE_COMMIT_V13B,
        "failed_implementation_bundle_sha256": (
            FAILED_IMPLEMENTATION_BUNDLE_V13B
        ),
        "failed_recipe_content_sha256": FAILED_RECIPE_CONTENT_SHA256_V13B,
        "failure_type": failure["type"],
        "failure_message": failure["message"],
        "failure_phase": attempt["phase"],
        "model_update_applied": False,
        "report_written": False,
    }
    binding["binding_sha256"] = _canonical(binding)
    return binding


def implementation_paths_v13b():
    return {
        **_BASE_IMPLEMENTATION_PATHS_V13,
        "driver_v13b": Path(__file__).resolve(),
        "retry_protocol_v13b": PROTOCOL_PATH_V13B,
        "tests_v13b": TEST_PATH_V13B,
    }


def implementation_identity_v13b():
    base = _BASE_IMPLEMENTATION_IDENTITY_V13()
    paths = implementation_paths_v13b()
    files = {
        key: {
            "path": str(Path(path).resolve()),
            "file_sha256": _file_sha256(path),
        }
        for key, path in paths.items()
    }
    if {
        key: value for key, value in files.items()
        if key in base["files"]
    } != base["files"]:
        raise RuntimeError("v13b inherited implementation identity changed")
    return {"files": files, "bundle_sha256": _canonical(files)}


def _normalized_recipe(recipe):
    normalized = copy.deepcopy(recipe)
    for key in (
        "schema", "experiment_name", "implementation_bundle_sha256",
        "v13_failure_binding", "runtime_expectation_retry_v13b",
        "content_sha256_before_self_field",
    ):
        normalized.pop(key, None)
    return normalized


def recipe_v13b(args, bundle, arrow, panels, implementation):
    failure = bind_failed_v13_attempt_v13b()
    recipe = _BASE_RECIPE_V13(args, bundle, arrow, panels, implementation)
    failed_recipe = json.loads(FAILED_ATTEMPT_V13B.read_text())["recipe"]
    if _normalized_recipe(recipe) != _normalized_recipe(failed_recipe):
        raise RuntimeError("v13b changed the frozen diagnostic recipe")
    recipe.pop("content_sha256_before_self_field", None)
    recipe.update({
        "schema": "eggroll-es-five-panel-recipe-v13b",
        "v13_failure_binding": failure,
        "runtime_expectation_retry_v13b": {
            "only_runtime_change": (
                "validate middle-late installation with inherited frozen V6 "
                "runtime expectations"
            ),
            "recipe_or_data_changed": False,
            "model_update_allowed": False,
        },
    })
    recipe["content_sha256_before_self_field"] = _canonical(recipe)
    return recipe


@contextlib.contextmanager
def scoped_v13b():
    prior = {
        "experiment": driver_v13.EXPERIMENT_NAME_V13,
        "paths": driver_v13.IMPLEMENTATION_PATHS_V13,
        "implementation": driver_v13.implementation_identity_v13,
        "recipe": driver_v13.recipe_v13,
    }
    driver_v13.EXPERIMENT_NAME_V13 = EXPERIMENT_NAME_V13B
    driver_v13.IMPLEMENTATION_PATHS_V13 = implementation_paths_v13b()
    driver_v13.implementation_identity_v13 = implementation_identity_v13b
    driver_v13.recipe_v13 = recipe_v13b
    try:
        yield
    finally:
        driver_v13.EXPERIMENT_NAME_V13 = prior["experiment"]
        driver_v13.IMPLEMENTATION_PATHS_V13 = prior["paths"]
        driver_v13.implementation_identity_v13 = prior["implementation"]
        driver_v13.recipe_v13 = prior["recipe"]


def main(argv=None):
    bind_failed_v13_attempt_v13b()
    with scoped_v13b():
        return driver_v13.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
