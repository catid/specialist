#!/usr/bin/env python3
"""Seal the additive V73G runtime closure and controller-bootstrap repair."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_v73g_systems_only_closure.json"
).resolve()
SCHEMA = "qwen36-v73e-systems-only-closure-v1"
STATUS = "v73e_runtime_closure_systems_only_no_quality_authority"
REQUIRED_PYTHON = (ROOT / "es-at-scale/.venv/bin/python").absolute()

ROOT_MODULES = (
    "run_qwen36_v73g_exact_phase_profiler.py",
    "run_lora_es_v71_v72_profile_calibration_v73g.py",
)
AUTHORITY_GUARD_MODULES = (
    *ROOT_MODULES,
    "qwen36_v73e_exact_phase_profiler_contract.py",
)
MUTABLE_PREREG_BUILDER = (
    "build_qwen36_v73g_exact_phase_profiler_preregistration.py"
)
TEST_MODULE = "test_qwen36_v73e_exact_phase_profiler.py"
SUCCESSOR_TEST_MODULE = "test_qwen36_v73g_controller_bootstrap.py"
REQUIRED_FALSE_AUTHORITY_KEYS = (
    "semantic_quality_selection_or_hpo_performed",
    "checkpoint_snapshot_or_promotion_performed",
    "protected_dev_ood_or_holdout_opened",
    "checkpoint_snapshot_or_promotion_authorized",
)
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
SYNTHETIC_GUARD_TEST_NODES = (
    f"{TEST_MODULE}::test_runtime_roots_do_not_import_preregistration_builder",
    f"{TEST_MODULE}::test_sitecustomize_guard_denies_every_claimed_open_and_resolution_api",
    f"{TEST_MODULE}::test_guarded_runtime_import_records_denied_legacy_resolution_not_open",
    f"{TEST_MODULE}::test_sitecustomize_guard_denies_exact_metadata_and_enumeration_apis",
    f"{TEST_MODULE}::test_guard_denies_synthetic_prefix_descendant_without_persisting_path",
    f"{TEST_MODULE}::test_guard_denies_bound_prefix_equal_and_descendant_paths",
    f"{TEST_MODULE}::test_guard_denies_dir_fd_bypass_and_repo_symlink_escape",
    f"{TEST_MODULE}::test_actor_subprocess_bootstrap_installs_guard_before_parent_imports",
    f"{TEST_MODULE}::test_actor_bootstrap_misconfiguration_fails_before_parent_import[missing_env]",
    f"{TEST_MODULE}::test_actor_bootstrap_misconfiguration_fails_before_parent_import[wrong_hash]",
    f"{TEST_MODULE}::test_actor_bootstrap_misconfiguration_fails_before_parent_import[preimported]",
    f"{TEST_MODULE}::test_ray_runtime_env_injection_preserves_existing_job_env_and_fails_collision",
    f"{TEST_MODULE}::test_gpu_accounting_keeps_reserved_resident_useful_and_promotion_split",
    f"{TEST_MODULE}::test_v73d_attempt_1_predecessor_is_immutable_and_nonpromotable",
    f"{TEST_MODULE}::test_target_and_profiler_failure_receipts_preserve_gpu_accounting_split",
    f"{SUCCESSOR_TEST_MODULE}::test_controller_validation_accepts_parents_imported_after_sitecustomize",
    f"{SUCCESSOR_TEST_MODULE}::test_actor_still_rejects_preimported_parent",
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


def _local_imports(
    path: Path,
    root: Path,
    authority_guard_modules: tuple[str, ...],
) -> tuple[set[Path], set[str], set[str], set[str], set[str]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imported: set[Path] = set()
    quarantined: set[str] = set()
    semantic: set[str] = set()
    false_keys: set[str] = set()
    true_keys: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module.split(".")[0]]
        for name in names:
            if name in QUARANTINED_MODULE_REFERENCES:
                quarantined.add(path.name)
            candidate = (root / f"{name}.py").resolve()
            if candidate.is_file() and candidate.parent == root:
                imported.add(candidate)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(name in node.value for name in QUARANTINED_MODULE_REFERENCES):
                quarantined.add(path.name)
            if any(value in node.value for value in SEMANTIC_BOUNDARY_LITERALS):
                semantic.add(path.name)
        if isinstance(node, ast.Dict) and path.name in authority_guard_modules:
            for key_node, value_node in zip(node.keys, node.values):
                if (
                    isinstance(key_node, ast.Constant)
                    and key_node.value in REQUIRED_FALSE_AUTHORITY_KEYS
                    and isinstance(value_node, ast.Constant)
                    and isinstance(value_node.value, bool)
                ):
                    (true_keys if value_node.value else false_keys).add(
                        str(key_node.value)
                    )
    return imported, quarantined, semantic, false_keys, true_keys


def analyze_import_graph(
    root: Path,
    root_modules: tuple[str, ...],
    *,
    authority_guard_modules: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    guards = authority_guard_modules or root_modules
    pending = [(root / name).resolve() for name in root_modules]
    seen: set[Path] = set()
    edges: dict[Path, set[Path]] = {}
    quarantined_modules: set[str] = set()
    semantic_modules: set[str] = set()
    false_keys: set[str] = set()
    true_keys: set[str] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file() or path.parent != root:
            raise RuntimeError("V73E import-graph module is absent or nonlocal")
        seen.add(path)
        imported, quarantined, semantic, false_seen, true_seen = _local_imports(
            path, root, guards
        )
        edges[path] = imported
        quarantined_modules.update(quarantined)
        semantic_modules.update(semantic)
        false_keys.update(false_seen)
        true_keys.update(true_seen)
        pending.extend(imported)
    entries = []
    for path in sorted(seen):
        identity = _path_identity(path.name, "repository-root-module-path-v1")
        entries.append({
            "module_path_identity_sha256": identity,
            "file_sha256": file_sha256(path),
            "local_import_path_identity_sha256s": sorted(
                _path_identity(child.name, "repository-root-module-path-v1")
                for child in edges[path]
            ),
        })
    authority = [
        {
            "module_path_identity_sha256": _path_identity(
                name, "repository-root-module-path-v1"
            ),
            "file_sha256": file_sha256(root / name),
        }
        for name in guards
        if (root / name).resolve() in seen
    ]

    def opaque(names: set[str], reference_class: str) -> list[str]:
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

    quarantined = opaque(quarantined_modules, "quarantined_api")
    semantic = opaque(semantic_modules, "semantic_evaluation_path")
    return {
        "root_modules": list(root_modules),
        "root_module_file_sha256": {
            name: file_sha256(root / name) for name in root_modules
        },
        "local_closure_modules": len(entries),
        "local_closure_content_sha256": canonical_sha256(entries),
        "mutable_prereg_builder_reachable": any(
            path.name == MUTABLE_PREREG_BUILDER for path in seen
        ),
        "authority_guard_module_count": len(authority),
        "authority_guard_source_bindings": authority,
        "quarantined_boundary_reference_count": len(quarantined),
        "quarantined_boundary_reference_identity_sha256s": quarantined,
        "semantic_evaluation_path_reference_count": len(semantic),
        "semantic_evaluation_path_reference_identity_sha256s": semantic,
        "required_false_authority_rule_count": len(false_keys),
        "true_quality_or_promotion_authority_rule_count": len(true_keys),
        "authority_rule_set_sha256": canonical_sha256(sorted(false_keys)),
    }


def run_synthetic_guard_tests() -> dict[str, Any]:
    command = [
        str(REQUIRED_PYTHON),
        "-m",
        "pytest",
        "-q",
        *SYNTHETIC_GUARD_TEST_NODES,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "V73E selected synthetic guard tests failed: "
            + completed.stdout[-2000:]
        )
    test_paths = (
        (ROOT / TEST_MODULE).resolve(),
        (ROOT / SUCCESSOR_TEST_MODULE).resolve(),
    )
    return {
        "schema": "v73e-synthetic-runtime-guard-test-receipt-v1",
        "passed": True,
        "exit_code": 0,
        "selected_test_node_count": len(SYNTHETIC_GUARD_TEST_NODES),
        "selected_test_node_set_sha256": canonical_sha256(
            sorted(SYNTHETIC_GUARD_TEST_NODES)
        ),
        "command_sha256": canonical_sha256(command),
        "test_source_bindings": [
            {
                "path_identity_sha256": _path_identity(
                    path.name, "repository-root-test-path-v1"
                ),
                "file_sha256": file_sha256(path),
            }
            for path in test_paths
        ],
        "stdout_or_stderr_persisted": False,
        "protected_source_fixture_or_loader_selected": False,
    }


def validate_zero_success_attestation(value: Mapping[str, Any]) -> None:
    attestation = value.get("zero_successful_protected_operation_attestation", {})
    if not (
        attestation.get("scope")
        == "byte_stable_selected_synthetic_subprocess_receipts"
        and all(attestation.get(name) == 0 for name in ZERO_SUCCESS_COUNTERS)
        and attestation.get("actual_stage_b_gpu_run_attested") is False
    ):
        raise RuntimeError("V73E Stage-A zero-success attestation changed")


def build_receipt() -> dict[str, Any]:
    graph = analyze_import_graph(
        ROOT,
        ROOT_MODULES,
        authority_guard_modules=AUTHORITY_GUARD_MODULES,
    )
    if not (
        graph["mutable_prereg_builder_reachable"] is False
        and graph["authority_guard_module_count"] == 3
        and graph["required_false_authority_rule_count"] == 4
        and graph["true_quality_or_promotion_authority_rule_count"] == 0
        and graph["local_closure_modules"] >= 3
    ):
        raise RuntimeError("V73E systems-only import closure authority changed")
    opaque_count = (
        graph["quarantined_boundary_reference_count"]
        + graph["semantic_evaluation_path_reference_count"]
    )
    if opaque_count <= 0:
        raise RuntimeError("V73E closure lost its opaque historical bindings")
    synthetic = run_synthetic_guard_tests()
    zero = {
        "scope": "byte_stable_selected_synthetic_subprocess_receipts",
        **{name: 0 for name in ZERO_SUCCESS_COUNTERS},
        "actual_stage_b_gpu_run_attested": False,
    }
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "import_graph": graph,
        "static_historical_references": {
            "opaque_reference_binding_count": opaque_count,
            "opaque_reference_binding_set_sha256": canonical_sha256(sorted(
                graph["quarantined_boundary_reference_identity_sha256s"]
                + graph["semantic_evaluation_path_reference_identity_sha256s"]
            )),
            "opaque_only": True,
            "does_not_authorize_path_resolution_or_open": True,
            "actual_open_denial_requires_distinct_postrun_receipt": True,
        },
        "synthetic_runtime_guard_tests": synthetic,
        "zero_successful_protected_operation_attestation": zero,
        "authority": {
            "systems_trace_only": True,
            "semantic_evaluation_or_quarantined_boundary_resolved": False,
            "quality_selection_hpo_or_promotion_authorized": False,
            "historical_quarantined_lineage_rehabilitated": False,
        },
        "implementation": {
            "builder_file_sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    validate_zero_success_attestation(result)
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def render(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")


def _atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output != OUTPUT:
        raise RuntimeError("V73E closure output path must remain canonical")
    value = build_receipt()
    payload = render(value)
    if args.check:
        if not output.is_file() or output.read_bytes() != payload:
            raise RuntimeError("V73E systems-only closure receipt is stale")
    else:
        if output.exists():
            raise FileExistsError(output)
        _atomic_create(output, payload)
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "local_closure_modules": value["import_graph"]["local_closure_modules"],
        "selected_test_node_count": value["synthetic_runtime_guard_tests"][
            "selected_test_node_count"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
