#!/usr/bin/env python3
"""Seal the additive V73D systems-only Stage-A boundary receipt."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_v73d_systems_only_closure.json"
).resolve()
SCHEMA = "qwen36-v73d-systems-only-closure-v1"
STATUS = "v73d_runtime_closure_systems_only_no_quality_authority"

V73D_ROOT_MODULES = (
    "run_qwen36_v73d_exact_phase_profiler.py",
    "run_lora_es_v71_v72_profile_calibration_v73d.py",
)
MUTABLE_PREREG_BUILDER = (
    "build_qwen36_v73d_exact_phase_profiler_preregistration.py"
)
AUTHORITY_GUARD_MODULES = (
    *V73D_ROOT_MODULES,
    "qwen36_v73d_exact_phase_profiler_contract.py",
)
REQUIRED_FALSE_AUTHORITY_KEYS = (
    "semantic_quality_selection_or_hpo_performed",
    "checkpoint_snapshot_or_promotion_performed",
    "protected_dev_ood_or_holdout_opened",
    "checkpoint_snapshot_or_promotion_authorized",
)

# These strings are inspected only in Python source.  Any receipt representation
# is an opaque module identity and count; no boundary path is persisted.
QUARANTINED_MODULE_REFERENCES = frozenset({
    "recipe_evaluation_contract_v1",
    "build_eval_v3",
    "build_general_prose_anchor",
    "build_lora_es_mirrored_calibration_preregistration_v66",
})
SEMANTIC_BOUNDARY_LITERALS = (
    "recipe_evaluation_compute_contract_v1.json",
    "eval_qa_v3.jsonl",
)

AUTHORITATIVE_GUARD_RELATIVE = (
    "v73d_sitecustomize/v73d_path_open_guard.py"
)
MISTYPED_TASK_GUARD_RELATIVE = (
    "v73d_sitecustomize/v73d_actor_boundary_guard.py"
)
AUTHORITATIVE_GUARD_SHA256 = (
    "750a445693425cb859bd4a632f14cad98799df7f969d833458282efa9e9c481c"
)
BYTE_STABLE_STAGE_A_MANIFEST = (
    (
        "systems_target",
        "run_lora_es_v71_v72_profile_calibration_v73d.py",
        "189fa0649a2c5e36cd8a3b539927d114aed38b92c831d1a88f2e066dd188a45c",
    ),
    (
        "launcher_analyzer",
        "run_qwen36_v73d_exact_phase_profiler.py",
        "909413fde3dd1c109ec6a8ee491ae0353a871b5f35342d7f0333b4c6bd84eca4",
    ),
    (
        "runtime_contract",
        "qwen36_v73d_exact_phase_profiler_contract.py",
        "1c972cc1f1e24903847f8d80eef6090c97d6b08a69dd26ba9fcb564485f10cf0",
    ),
    (
        "actor_worker",
        "eggroll_es_worker_lora_v73d.py",
        "90acfa98c43c5ce7ed1b92d239cf24e2fba9870214fafb2b6befa8e2dfaece89",
    ),
    (
        "controller_sitecustomize",
        "v73d_sitecustomize/sitecustomize.py",
        "9efa5ff11293f64dc911268f72fd6d47a53ed31cf47515f1b87b810c61220bfe",
    ),
    (
        "path_open_guard",
        AUTHORITATIVE_GUARD_RELATIVE,
        AUTHORITATIVE_GUARD_SHA256,
    ),
    (
        "profiler_test",
        "test_qwen36_v73d_exact_phase_profiler.py",
        "f70b667e0200599e00753dc28ee8226127c599ef3ce3ad8597aa6301f329e78a",
    ),
    (
        "preregistration_builder_outside_runtime_closure",
        MUTABLE_PREREG_BUILDER,
        "10971a34e75ff45e57e6884357cbe9055e45c17e405913139dba7d9cb7d37562",
    ),
)

BOUNDARY_REGISTRY = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "quarantine_boundary_registry_v3.json"
).resolve()
BOUNDARY_REGISTRY_BUILDER = (
    ROOT / "build_quarantine_boundary_registry_v3.py"
).resolve()
BOUNDARY_REGISTRY_FILE_SHA256 = (
    "3d8ef097a1419e03f4b735e6f8d30e5a876b0a8e86c4b0f1ac100114cb7daf5d"
)
BOUNDARY_REGISTRY_CONTENT_SHA256 = (
    "5a8c6dbd3b0c225992bb2cc4ae5c02a5ec7774a175fe1a138c4e78260eed7fc7"
)
BOUNDARY_REGISTRY_BUILDER_SHA256 = (
    "e217bf08bd8884ef223ebaa35a7ca7490447ca458af4cba9270725945fa5ccc9"
)
BOUNDARY_POLICY_SHA256 = (
    "3758e272396ebc0cbed5a933469665530275c9eabe99f1c0d6210464e7d1a48c"
)

V73C_SEALED_SOURCE_COMMIT = "35d2837d1632a2b642ad9df15a7eb610d47cdc34"
V73C_PREDECESSOR_ARTIFACTS = (
    (
        "preregistration",
        "experiments/eggroll_es_hpo/preregistrations/"
        "qwen36_v73c_exact_phase_profiler.json",
        "b134a263b1548905d8f9c2373d9f8a8dd79ec2ba74808a5ca23c84f31dc71983",
        "6ea79bd1e948f728af0f93087d0e28ad576cef62bfc04be79180f30a0a93785a",
    ),
    (
        "profile_attempt",
        "experiments/eggroll_es_hpo/profiles/"
        ".v73c_timeline_lora_es_same_live_qwen36_exact_phase.attempt.json",
        "724b60b6ce33c85e129e5000e1753d8ebd25615f7ede97a17888533397785baf",
        "a3c0a4fdf829848580aed8fc9e7e2af05b2904eda250e30c234220dd3a63f24b",
    ),
    (
        "profile_failure",
        "experiments/eggroll_es_hpo/profiles/"
        "v73c_timeline_lora_es_same_live_qwen36_exact_phase/"
        "profile_failure_v73c.json",
        "ae76cdcf38a71ac7e27035a38d48867a91ed68248c91ef4535ce25b0d79a0c17",
        "fceaf41c3d8e9a09b20e86d554b3cb8b59d97488f7cd8c083e8ec7048013422a",
    ),
    (
        "nsys_report",
        "experiments/eggroll_es_hpo/profiles/"
        "v73c_timeline_lora_es_same_live_qwen36_exact_phase/"
        "v73c_timeline_lora_es_same_live_qwen36_exact_phase.nsys-rep",
        "dbdce8049120ed3ffc61834497dd5f52546d07a6d53bc54c9b0c123acfeaf53e",
        None,
    ),
    (
        "sqlite_export",
        "experiments/eggroll_es_hpo/profiles/"
        "v73c_timeline_lora_es_same_live_qwen36_exact_phase/"
        "v73c_timeline_lora_es_same_live_qwen36_exact_phase.sqlite",
        "73afe6d1f68f0c906224947781a236681ce4799d60a32da9cc7075dd66d4dc7d",
        None,
    ),
    (
        "run_attempt",
        "experiments/eggroll_es_hpo/runs/"
        ".v73c_timeline_lora_es_same_live_qwen36_exact_phase.attempt.json",
        "f73fd75b1a9780b2b8a46e7efebf3f514e7b9fb7723df8be2c4ca78c2db9e6b5",
        "28dce8692c500ae902993773247d5e7be3153e1aa216fc8a9d00191d9fe4c145",
    ),
    (
        "run_failure",
        "experiments/eggroll_es_hpo/runs/"
        "v73c_timeline_lora_es_same_live_qwen36_exact_phase/failure_v73c.json",
        "5708f456e7736944b7304c5a17486b82c15c5a451a51679734ff00e86cdefef7",
        "a5eb95f7d987112dec0a71683d668d0369a56d7a7838e66c986bf6132cafed4d",
    ),
    (
        "partial_phase_receipt",
        "experiments/eggroll_es_hpo/runs/"
        "v73c_timeline_lora_es_same_live_qwen36_exact_phase/"
        "exact_phase_ranges_v73c.json",
        "ad6f0c23b478b5688d92b0dde4f7acc0a17eace68f277cdf1dec90ecf3469b91",
        "25155879eca87b08180651c456b7c7ab74e1c0dd2a3f61ad80f5bb422449a38d",
    ),
)

# Exact Stage-A nodes supplied after fixture/loader inspection.  The separate
# local-CPU Ray actor node is intentionally excluded from the pre-seal set.
SYNTHETIC_GUARD_TEST_NODES = (
    "test_qwen36_v73d_exact_phase_profiler.py::"
    "test_runtime_roots_do_not_import_preregistration_builder",
    "test_qwen36_v73d_exact_phase_profiler.py::"
    "test_sitecustomize_guard_denies_every_claimed_open_and_resolution_api",
    "test_qwen36_v73d_exact_phase_profiler.py::"
    "test_guarded_runtime_import_records_denied_legacy_resolution_not_open",
    "test_qwen36_v73d_exact_phase_profiler.py::"
    "test_sitecustomize_guard_denies_exact_metadata_and_enumeration_apis",
    "test_qwen36_v73d_exact_phase_profiler.py::"
    "test_guard_denies_synthetic_prefix_descendant_without_persisting_path",
    "test_qwen36_v73d_exact_phase_profiler.py::"
    "test_guard_denies_bound_prefix_equal_and_descendant_paths",
    "test_qwen36_v73d_exact_phase_profiler.py::"
    "test_guard_denies_dir_fd_bypass_and_repo_symlink_escape",
    "test_qwen36_v73d_exact_phase_profiler.py::"
    "test_actor_subprocess_bootstrap_installs_guard_before_parent_imports",
    "test_qwen36_v73d_exact_phase_profiler.py::"
    "test_actor_bootstrap_misconfiguration_fails_before_parent_import[missing_env]",
    "test_qwen36_v73d_exact_phase_profiler.py::"
    "test_actor_bootstrap_misconfiguration_fails_before_parent_import[wrong_hash]",
    "test_qwen36_v73d_exact_phase_profiler.py::"
    "test_actor_bootstrap_misconfiguration_fails_before_parent_import[preimported]",
    "test_qwen36_v73d_exact_phase_profiler.py::"
    "test_ray_runtime_env_injection_preserves_existing_job_env_and_fails_collision",
    "test_qwen36_v73d_exact_phase_profiler.py::"
    "test_gpu_accounting_keeps_reserved_resident_useful_and_promotion_split",
    "test_qwen36_v73d_exact_phase_profiler.py::"
    "test_v73c_attempt_1_predecessor_is_immutable_and_nonpromotable",
    "test_qwen36_v73d_exact_phase_profiler.py::"
    "test_target_and_profiler_failure_receipts_preserve_gpu_accounting_split",
)
ZERO_SUCCESS_COUNTERS = (
    "successful_protected_opens",
    "successful_protected_resolves",
    "successful_protected_metadata",
    "successful_protected_enumerations",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_identity(relative: str, schema: str) -> str:
    return canonical_sha256({"schema": schema, "value": relative})


def verify_byte_stable_stage_a_manifest() -> dict[str, Any]:
    bindings = []
    for role, relative, expected in BYTE_STABLE_STAGE_A_MANIFEST:
        path = (ROOT / relative).resolve()
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.is_relative_to(ROOT)
            or file_sha256(path) != expected
        ):
            raise RuntimeError("V73D byte-stable Stage-A manifest changed")
        bindings.append({
            "role": role,
            "path_identity_sha256": _path_identity(
                relative, "repository-relative-stage-a-path-v1"
            ),
            "file_sha256": expected,
        })
    if (ROOT / MISTYPED_TASK_GUARD_RELATIVE).exists():
        raise RuntimeError("V73D corrected guard manifest became ambiguous")
    correction = {
        "schema": "v73d-stage-a-task-manifest-correction-v1",
        "task_message_guard_path": MISTYPED_TASK_GUARD_RELATIVE,
        "task_message_guard_path_exists": False,
        "authoritative_guard_path": AUTHORITATIVE_GUARD_RELATIVE,
        "authoritative_guard_file_sha256": AUTHORITATIVE_GUARD_SHA256,
        "confirmed_by_parent_task_correction": True,
    }
    return {
        "schema": "v73d-byte-stable-stage-a-manifest-v1",
        "binding_count": len(bindings),
        "bindings": bindings,
        "binding_set_sha256": canonical_sha256(bindings),
        "manifest_correction": correction,
        "preregistration_builder_bound_outside_runtime_closure": True,
    }


def analyze_import_graph(
    root: Path,
    root_modules: tuple[str, ...],
    *,
    authority_guard_modules: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    guard_modules = authority_guard_modules or root_modules
    pending = [(root / name).resolve() for name in root_modules]
    seen: set[Path] = set()
    edges: dict[Path, set[Path]] = {}
    quarantined_reference_modules: set[str] = set()
    semantic_reference_modules: set[str] = set()
    false_authority_keys: set[str] = set()
    true_authority_keys: set[str] = set()

    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file() or path.parent != root:
            raise RuntimeError("V73D import-graph module is absent or nonlocal")
        seen.add(path)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imported: set[Path] = set()
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
            ):
                names = [node.module.split(".")[0]]
            for name in names:
                if name in QUARANTINED_MODULE_REFERENCES:
                    quarantined_reference_modules.add(path.name)
                candidate = (root / f"{name}.py").resolve()
                if candidate.is_file() and candidate.parent == root:
                    imported.add(candidate)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(
                    reference in node.value
                    for reference in QUARANTINED_MODULE_REFERENCES
                ):
                    quarantined_reference_modules.add(path.name)
                if any(
                    literal in node.value for literal in SEMANTIC_BOUNDARY_LITERALS
                ):
                    semantic_reference_modules.add(path.name)
            if isinstance(node, ast.Dict) and path.name in guard_modules:
                for key_node, value_node in zip(node.keys, node.values):
                    if (
                        isinstance(key_node, ast.Constant)
                        and key_node.value in REQUIRED_FALSE_AUTHORITY_KEYS
                        and isinstance(value_node, ast.Constant)
                        and isinstance(value_node.value, bool)
                    ):
                        target = (
                            true_authority_keys
                            if value_node.value
                            else false_authority_keys
                        )
                        target.add(str(key_node.value))
        edges[path] = imported
        pending.extend(imported)

    entries = [
        {
            "module_path_identity_sha256": _path_identity(
                path.name, "repository-root-module-path-v1"
            ),
            "file_sha256": file_sha256(path),
            "local_import_path_identity_sha256s": sorted(
                _path_identity(child.name, "repository-root-module-path-v1")
                for child in edges[path]
            ),
        }
        for path in sorted(seen)
    ]
    path_identities = sorted(
        item["module_path_identity_sha256"] for item in entries
    )
    binding_identities = sorted(
        canonical_sha256({
            "schema": "local-module-byte-binding-v1",
            "module_path_identity_sha256": item[
                "module_path_identity_sha256"
            ],
            "file_sha256": item["file_sha256"],
        })
        for item in entries
    )
    edge_identities = sorted(
        canonical_sha256({
            "schema": "local-python-import-edge-v1",
            "source_path_identity_sha256": item[
                "module_path_identity_sha256"
            ],
            "target_path_identity_sha256": target,
        })
        for item in entries
        for target in item["local_import_path_identity_sha256s"]
    )

    def opaque_references(names: set[str], reference_class: str) -> list[str]:
        return sorted(
            canonical_sha256({
                "schema": "opaque-static-boundary-reference-v2",
                "module_path_identity_sha256": _path_identity(
                    name, "repository-root-module-path-v1"
                ),
                "reference_class": reference_class,
            })
            for name in names
        )

    quarantined_identities = opaque_references(
        quarantined_reference_modules, "quarantined_api"
    )
    semantic_identities = opaque_references(
        semantic_reference_modules, "semantic_evaluation_path"
    )
    authority_bindings = [
        {
            "module_path_identity_sha256": _path_identity(
                name, "repository-root-module-path-v1"
            ),
            "file_sha256": file_sha256(root / name),
        }
        for name in guard_modules
        if (root / name).resolve() in seen
    ]
    return {
        "root_modules": list(root_modules),
        "root_module_file_sha256": {
            name: file_sha256(root / name) for name in root_modules
        },
        "local_closure_modules": len(entries),
        "mutable_prereg_builder_reachable": any(
            path.name == MUTABLE_PREREG_BUILDER for path in seen
        ),
        "authority_guard_module_count": len(authority_bindings),
        "authority_guard_source_bindings": authority_bindings,
        "local_closure_path_identity_sha256s": path_identities,
        "local_closure_byte_binding_identity_sha256s": binding_identities,
        "local_import_edge_identity_sha256s": edge_identities,
        "local_closure_content_sha256": canonical_sha256(entries),
        "quarantined_boundary_reference_count": len(quarantined_identities),
        "quarantined_boundary_reference_identity_sha256s": (
            quarantined_identities
        ),
        "semantic_evaluation_path_reference_count": len(semantic_identities),
        "semantic_evaluation_path_reference_identity_sha256s": (
            semantic_identities
        ),
        "required_false_authority_rule_count": len(false_authority_keys),
        "true_quality_or_promotion_authority_rule_count": len(
            true_authority_keys
        ),
        "authority_rule_set_sha256": canonical_sha256(
            sorted(false_authority_keys)
        ),
    }


def verify_quarantine_boundary_registry_v3() -> dict[str, Any]:
    if (
        not BOUNDARY_REGISTRY.is_file()
        or BOUNDARY_REGISTRY.is_symlink()
        or file_sha256(BOUNDARY_REGISTRY) != BOUNDARY_REGISTRY_FILE_SHA256
        or file_sha256(BOUNDARY_REGISTRY_BUILDER)
        != BOUNDARY_REGISTRY_BUILDER_SHA256
    ):
        raise RuntimeError("V73D V3 quarantine boundary bytes changed")
    value = json.loads(BOUNDARY_REGISTRY.read_text(encoding="ascii"))
    body = dict(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    policy = value.get("ancestor_denial_policy", {})
    if not (
        claimed == BOUNDARY_REGISTRY_CONTENT_SHA256
        and canonical_sha256(body) == claimed
        and value.get("schema")
        == "specialist-content-free-quarantine-boundary-registry-v3"
        and value.get("status")
        == "active_fail_closed_content_free_quarantine_boundary"
        and value.get("exact_path_identity_count") == 3
        and value.get("prefix_identity_count") == 2
        and value.get("ancestor_denial_policy_sha256")
        == BOUNDARY_POLICY_SHA256
        and policy.get("lexical_deny_before_resolution_stat_hash_or_open")
        is True
        and policy.get(
            "lexically_allowed_resolution_rechecked_before_metadata_or_open"
        )
        is True
        and policy.get("resolved_target_outside_repository_root_denied") is True
        and value.get("content_minimization", {}).get(
            "plaintext_boundary_paths_persisted"
        )
        is False
        and value.get("content_minimization", {}).get(
            "source_bytes_or_semantics_persisted"
        )
        is False
    ):
        raise RuntimeError("V73D V3 quarantine boundary contract changed")
    return {
        "schema": "v73d-quarantine-boundary-registry-v3-binding-v1",
        "status": value["status"],
        "registry_path_identity_sha256": _path_identity(
            str(BOUNDARY_REGISTRY.relative_to(ROOT)),
            "repository-relative-quarantine-registry-path-v1",
        ),
        "file_sha256": BOUNDARY_REGISTRY_FILE_SHA256,
        "content_sha256": BOUNDARY_REGISTRY_CONTENT_SHA256,
        "builder_file_sha256": BOUNDARY_REGISTRY_BUILDER_SHA256,
        "ancestor_denial_policy_sha256": BOUNDARY_POLICY_SHA256,
        "exact_path_identity_count": value["exact_path_identity_count"],
        "exact_path_identity_set_sha256": canonical_sha256(
            value["exact_path_identity_sha256"]
        ),
        "prefix_identity_count": value["prefix_identity_count"],
        "prefix_identity_set_sha256": canonical_sha256(
            value["prefix_identity_sha256"]
        ),
        "plaintext_boundary_paths_persisted": False,
        "source_bytes_or_semantics_persisted": False,
    }


def _load_self_hashed_json(
    relative: str, expected_content_sha256: str
) -> dict[str, Any]:
    path = (ROOT / relative).resolve()
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise RuntimeError("V73C predecessor JSON is not an object")
    body = dict(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    if (
        claimed != expected_content_sha256
        or canonical_sha256(body) != expected_content_sha256
    ):
        raise RuntimeError("V73C predecessor content identity changed")
    return value


def verify_immutable_failed_v73c_predecessor() -> dict[str, Any]:
    bindings = []
    values: dict[str, dict[str, Any]] = {}
    for role, relative, expected_file, expected_content in (
        V73C_PREDECESSOR_ARTIFACTS
    ):
        path = (ROOT / relative).resolve()
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.is_relative_to(ROOT)
            or file_sha256(path) != expected_file
        ):
            raise RuntimeError("immutable failed V73C predecessor bytes changed")
        binding = {
            "role": role,
            "path_identity_sha256": _path_identity(
                relative, "repository-relative-v73c-predecessor-path-v1"
            ),
            "file_sha256": expected_file,
        }
        if expected_content is not None:
            values[role] = _load_self_hashed_json(relative, expected_content)
            binding["content_sha256"] = expected_content
        bindings.append(binding)

    prereg = values["preregistration"]
    profile_attempt = values["profile_attempt"]
    profile_failure = values["profile_failure"]
    run_attempt = values["run_attempt"]
    run_failure = values["run_failure"]
    phase = values["partial_phase_receipt"]
    if not (
        prereg.get("schema")
        == "qwen36-v73c-exact-phase-profiler-preregistration-v1"
        and profile_attempt.get("status")
        == "prelaunch_accepted_launching_fresh_no_commit_profile"
        and profile_failure.get("status") == "target_or_profiler_failed_closed"
        and profile_failure.get("returncode") == 1
        and run_attempt.get("status")
        == "launching_train_only_exact_equivalence_no_commit"
        and run_failure.get("type") == "ActorDiedError"
        and run_failure.get("partial_actor_cuda_work_log", {}).get("rows") == 0
        and run_failure.get("partial_host_process_log", {}).get("rows") == 0
        and run_failure.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        )
        is True
        and run_failure.get("protected_dev_ood_or_holdout_opened") is False
        and run_failure.get("checkpoint_snapshot_or_promotion_performed")
        is False
        and phase.get("complete") is False
        and phase.get("observed_phase_order") == ["setup"]
        and phase.get("contains_prompts_questions_answers_or_outputs") is False
    ):
        raise RuntimeError("immutable V73C failed predecessor status changed")
    return {
        "schema": "v73d-immutable-failed-v73c-predecessor-binding-v1",
        "status": "immutable_failed_closed_actor_sitecustomize_assumption",
        "v73c_sealed_source_commit": V73C_SEALED_SOURCE_COMMIT,
        "artifact_binding_count": len(bindings),
        "artifact_bindings": bindings,
        "artifact_binding_set_sha256": canonical_sha256(bindings),
        "partial_actor_cuda_receipt_count": 0,
        "final_all_four_gpus_idle": True,
        "protected_or_semantic_data_opened": False,
        "promotion_charged_gpu_seconds": 0,
        "artifact_plaintext_paths_persisted": False,
        "artifacts_must_not_be_deleted_modified_or_reused": True,
    }


def run_synthetic_guard_tests() -> dict[str, Any]:
    command = (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *SYNTHETIC_GUARD_TEST_NODES,
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("V73D synthetic/CPU-only Stage-A tests failed")
    test_files = sorted(
        {node.split("::", 1)[0] for node in SYNTHETIC_GUARD_TEST_NODES}
    )
    bindings = []
    for relative in test_files:
        path = (ROOT / relative).resolve()
        if not path.is_file() or path.parent != ROOT:
            raise RuntimeError("V73D selected test source is absent")
        bindings.append({
            "path_identity_sha256": _path_identity(
                relative, "repository-root-test-path-v1"
            ),
            "file_sha256": file_sha256(path),
        })
    return {
        "schema": "v73d-synthetic-runtime-guard-test-receipt-v1",
        "passed": True,
        "exit_code": 0,
        "selected_test_node_count": len(SYNTHETIC_GUARD_TEST_NODES),
        "selected_test_command_sha256": canonical_sha256(list(command[2:])),
        "test_source_bindings": bindings,
        "local_cpu_ray_actor_test_executed_by_stage_a": False,
        "model_cuda_or_gpu_launched_by_stage_a": False,
        "stdout_or_stderr_persisted": False,
        "zero_successful_protected_operation_attestation": {
            "scope": "byte_stable_selected_synthetic_subprocess_receipts",
            **{name: 0 for name in ZERO_SUCCESS_COUNTERS},
            "actual_stage_b_gpu_run_attested": False,
        },
    }


def validate_zero_success_attestation(value: dict[str, Any]) -> None:
    attestation = value.get(
        "zero_successful_protected_operation_attestation", {}
    )
    if (
        attestation.get("scope")
        != "byte_stable_selected_synthetic_subprocess_receipts"
        or any(attestation.get(name) != 0 for name in ZERO_SUCCESS_COUNTERS)
        or attestation.get("actual_stage_b_gpu_run_attested") is not False
    ):
        raise RuntimeError("V73D zero-success guard attestation changed")


def build_audit(
    synthetic_tests: dict[str, Any],
    byte_stable_manifest: dict[str, Any],
    boundary_registry: dict[str, Any],
    predecessor: dict[str, Any],
) -> dict[str, Any]:
    graph = analyze_import_graph(
        ROOT,
        V73D_ROOT_MODULES,
        authority_guard_modules=AUTHORITY_GUARD_MODULES,
    )
    opaque_count = (
        graph["quarantined_boundary_reference_count"]
        + graph["semantic_evaluation_path_reference_count"]
    )
    validate_zero_success_attestation(synthetic_tests)
    if not (
        graph["mutable_prereg_builder_reachable"] is False
        and graph["authority_guard_module_count"]
        == len(AUTHORITY_GUARD_MODULES)
        and graph["required_false_authority_rule_count"]
        == len(REQUIRED_FALSE_AUTHORITY_KEYS)
        and graph["true_quality_or_promotion_authority_rule_count"] == 0
        and opaque_count > 0
        and synthetic_tests.get("schema")
        == "v73d-synthetic-runtime-guard-test-receipt-v1"
        and synthetic_tests.get("passed") is True
        and synthetic_tests.get("exit_code") == 0
        and synthetic_tests.get("selected_test_node_count")
        == len(SYNTHETIC_GUARD_TEST_NODES)
        and synthetic_tests.get("stdout_or_stderr_persisted") is False
        and synthetic_tests.get("local_cpu_ray_actor_test_executed_by_stage_a")
        is False
        and synthetic_tests.get("model_cuda_or_gpu_launched_by_stage_a")
        is False
        and byte_stable_manifest == verify_byte_stable_stage_a_manifest()
        and boundary_registry == verify_quarantine_boundary_registry_v3()
        and predecessor == verify_immutable_failed_v73c_predecessor()
    ):
        raise RuntimeError("V73D Stage-A systems-only boundary is incomplete")
    audit = {
        "schema": SCHEMA,
        "status": STATUS,
        "created_at_utc": "2026-07-17T00:00:00+00:00",
        "import_graph": graph,
        "static_historical_references": {
            "opaque_only": True,
            "opaque_reference_binding_count": opaque_count,
            "does_not_authorize_path_resolution_or_open": True,
            "actual_open_denial_requires_distinct_postrun_receipt": True,
            "plaintext_boundary_paths_persisted": False,
        },
        "synthetic_runtime_guard_tests": synthetic_tests,
        "byte_stable_stage_a_manifest": byte_stable_manifest,
        "quarantine_boundary_registry_v3": boundary_registry,
        "immutable_failed_v73c_predecessor": predecessor,
        "authority": {
            "systems_trace_only": True,
            "semantic_evaluation_or_quarantined_boundary_resolved": False,
            "quality_selection_hpo_or_promotion_authorized": False,
            "historical_quarantined_lineage_rehabilitated": False,
            "stage_a_boundary_sealed": True,
            "stage_b_gpu_launch_authorized": False,
            "stage_b_requires_committed_v73d_preregistration": True,
        },
        "implementation": {
            "builder_file_sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    audit["content_sha256_before_self_field"] = canonical_sha256(audit)
    return audit


def validate_audit(audit: dict[str, Any]) -> None:
    body = {
        key: value
        for key, value in audit.items()
        if key != "content_sha256_before_self_field"
    }
    authority = audit.get("authority", {})
    graph = audit.get("import_graph", {})
    static = audit.get("static_historical_references", {})
    synthetic = audit.get("synthetic_runtime_guard_tests", {})
    opaque_count = (
        graph.get("quarantined_boundary_reference_count", 0)
        + graph.get("semantic_evaluation_path_reference_count", 0)
    )
    validate_zero_success_attestation(synthetic)
    if not (
        audit.get("schema") == SCHEMA
        and audit.get("status") == STATUS
        and audit.get("content_sha256_before_self_field")
        == canonical_sha256(body)
        and graph.get("root_modules") == list(V73D_ROOT_MODULES)
        and graph.get("mutable_prereg_builder_reachable") is False
        and graph.get("authority_guard_module_count")
        == len(AUTHORITY_GUARD_MODULES)
        and graph.get("required_false_authority_rule_count")
        == len(REQUIRED_FALSE_AUTHORITY_KEYS)
        and graph.get("true_quality_or_promotion_authority_rule_count") == 0
        and graph.get("local_closure_modules", 0) >= 3
        and opaque_count > 0
        and static.get("opaque_only") is True
        and static.get("opaque_reference_binding_count") == opaque_count
        and static.get("does_not_authorize_path_resolution_or_open") is True
        and static.get("actual_open_denial_requires_distinct_postrun_receipt")
        is True
        and static.get("plaintext_boundary_paths_persisted") is False
        and synthetic.get("selected_test_node_count")
        == len(SYNTHETIC_GUARD_TEST_NODES)
        and audit.get("byte_stable_stage_a_manifest")
        == verify_byte_stable_stage_a_manifest()
        and audit.get("quarantine_boundary_registry_v3")
        == verify_quarantine_boundary_registry_v3()
        and audit.get("immutable_failed_v73c_predecessor")
        == verify_immutable_failed_v73c_predecessor()
        and authority.get("systems_trace_only") is True
        and authority.get(
            "semantic_evaluation_or_quarantined_boundary_resolved"
        )
        is False
        and authority.get("quality_selection_hpo_or_promotion_authorized")
        is False
        and authority.get("historical_quarantined_lineage_rehabilitated")
        is False
        and authority.get("stage_a_boundary_sealed") is True
        and authority.get("stage_b_gpu_launch_authorized") is False
        and authority.get("stage_b_requires_committed_v73d_preregistration")
        is True
        and audit.get("implementation", {}).get("builder_file_sha256")
        == file_sha256(Path(__file__).resolve())
    ):
        raise RuntimeError("invalid V73D systems-only Stage-A audit")


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(
        "ascii"
    )
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    byte_stable_manifest = verify_byte_stable_stage_a_manifest()
    boundary_registry = verify_quarantine_boundary_registry_v3()
    predecessor = verify_immutable_failed_v73c_predecessor()
    synthetic_tests = run_synthetic_guard_tests()
    value = build_audit(
        synthetic_tests,
        byte_stable_manifest,
        boundary_registry,
        predecessor,
    )
    validate_audit(value)
    if args.check:
        persisted = json.loads(args.output.read_text(encoding="ascii"))
        if persisted != value:
            raise RuntimeError("persisted V73D Stage-A receipt differs")
    else:
        _write_exclusive(args.output.resolve(), value)
    print(json.dumps({
        "content_sha256": value["content_sha256_before_self_field"],
        "local_closure_modules": value["import_graph"][
            "local_closure_modules"
        ],
        "opaque_static_reference_bindings": value[
            "static_historical_references"
        ]["opaque_reference_binding_count"],
        "successful_protected_operations": 0,
        "stage_b_gpu_launch_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
