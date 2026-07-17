#!/usr/bin/env python3
"""Prove the V73C systems-only graph cannot resolve V1 or authorize quality."""

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


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_v73c_systems_only_closure.json"
).resolve()
V73C_ROOT_MODULES = (
    "run_qwen36_v73c_exact_phase_profiler.py",
    "run_lora_es_v71_v72_profile_calibration_v73c.py",
)
MUTABLE_PREREG_BUILDER = "build_qwen36_v73c_exact_phase_profiler_preregistration.py"
AUTHORITY_GUARD_MODULES = (
    *V73C_ROOT_MODULES,
    "qwen36_v73c_exact_phase_profiler_contract.py",
)
BYTE_STABLE_RUNTIME_GUARD_SET = (
    (
        "runtime_contract",
        "qwen36_v73c_exact_phase_profiler_contract.py",
        "4bfdf07798983c068f9eb01425c9ab64972636c6c613e09127658ffa429ba73f",
    ),
    (
        "launcher_analyzer",
        "run_qwen36_v73c_exact_phase_profiler.py",
        "78a786e08db7fb8ed2ed9896b0f00f67b9ae12003404f643e8a243cbc5ba014b",
    ),
    (
        "systems_target",
        "run_lora_es_v71_v72_profile_calibration_v73c.py",
        "3ff1beb2d8750158e789352394bd32e46eeb6d31adce5224f6a12a036b8d5179",
    ),
    (
        "worker",
        "eggroll_es_worker_lora_v73c.py",
        "fbed2d324178564214a8390487e95eae9090c01a597dc2582f28e6c16fb72dfb",
    ),
    (
        "path_guard_bootstrap",
        "v73c_sitecustomize/sitecustomize.py",
        "8dded79235ebd5370236c2cd8398749886769825695fd8af751e9e7af0e1d26e",
    ),
    (
        "path_guard",
        "v73c_sitecustomize/v73c_path_open_guard.py",
        "3bca3368fc638baa8ec40750f2ac88ad410834541864136f39beba0f54a66305",
    ),
    (
        "profiler_test",
        "test_qwen36_v73c_exact_phase_profiler.py",
        "d4887ac15a466ba0b382d1bb3c920b390c6605eda85aa27a2a2861c6cde9c6d1",
    ),
)
SCHEMA = "qwen36-v73c-systems-only-closure-v1"
FORBIDDEN_V1_REFERENCE = "recipe_evaluation_contract_v1"
FORBIDDEN_ROOT_LITERALS = (
    "recipe_evaluation_compute_contract_v1.json",
    "eval_qa_v3.jsonl",
)
REQUIRED_FALSE_AUTHORITY_KEYS = (
    "semantic_quality_selection_or_hpo_performed",
    "checkpoint_snapshot_or_promotion_performed",
    "protected_dev_ood_or_holdout_opened",
    "checkpoint_snapshot_or_promotion_authorized",
)
SYNTHETIC_GUARD_TEST_NODES = (
    "test_audit_v73c_systems_only_import_graph_v2.py",
    (
        "test_qwen36_v73c_exact_phase_profiler.py::"
        "test_runtime_roots_do_not_import_preregistration_builder"
    ),
    (
        "test_qwen36_v73c_exact_phase_profiler.py::"
        "test_phase_ledger_emits_exact_nonoverlapping_order"
    ),
    (
        "test_qwen36_v73c_exact_phase_profiler.py::"
        "test_fresh_path_collision_fails_closed"
    ),
    (
        "test_qwen36_v73c_exact_phase_profiler.py::"
        "test_synthetic_timeline_exact_phase_parser"
    ),
    (
        "test_qwen36_v73c_exact_phase_profiler.py::"
        "test_synthetic_timeline_fail_closed"
    ),
    (
        "test_qwen36_v73c_exact_phase_profiler.py::"
        "test_nccl_debug_transport_requires_explicit_content_free_lines"
    ),
    (
        "test_qwen36_v73c_exact_phase_profiler.py::"
        "test_nccl_debug_rejects_semantic_markers"
    ),
    (
        "test_qwen36_v73c_exact_phase_profiler.py::"
        "test_sitecustomize_guard_denies_every_claimed_open_and_resolution_api"
    ),
    (
        "test_qwen36_v73c_exact_phase_profiler.py::"
        "test_guarded_runtime_import_records_denied_legacy_resolution_not_open"
    ),
)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_byte_stable_runtime_guard_bindings() -> list[dict]:
    return [
        {
            "role": role,
            "path_identity_sha256": canonical_sha256({
                "schema": "repository-relative-runtime-path-v1",
                "value": relative,
            }),
            "file_sha256": expected,
        }
        for role, relative, expected in BYTE_STABLE_RUNTIME_GUARD_SET
    ]


def verify_byte_stable_runtime_guard_set() -> dict:
    bindings = _expected_byte_stable_runtime_guard_bindings()
    for binding, (_role, relative, expected) in zip(
        bindings, BYTE_STABLE_RUNTIME_GUARD_SET
    ):
        path = (ROOT / relative).resolve()
        if (
            not path.is_file()
            or not path.is_relative_to(ROOT)
            or file_sha256(path) != expected
            or binding["file_sha256"] != expected
        ):
            raise RuntimeError("V73C byte-stable runtime/guard binding changed")
    return {
        "schema": "v73c-byte-stable-runtime-guard-set-v1",
        "binding_count": len(bindings),
        "bindings": bindings,
        "binding_set_sha256": canonical_sha256(bindings),
        "mutable_preregistration_builder_included": False,
    }


def analyze_import_graph(
    root: Path,
    root_modules: tuple[str, ...],
    *,
    authority_guard_modules: tuple[str, ...] | None = None,
) -> dict:
    root = root.resolve()
    guard_modules = authority_guard_modules or root_modules
    pending = [(root / name).resolve() for name in root_modules]
    seen: set[Path] = set()
    edges: dict[Path, set[Path]] = {}
    forbidden_v1_references = []
    forbidden_root_literals = []
    false_authority_keys = set()
    true_authority_keys = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file() or path.parent != root:
            raise RuntimeError("V73C import-graph module is absent or nonlocal")
        seen.add(path)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imported = set()
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
            ):
                names = [node.module.split(".")[0]]
            for name in names:
                if name == FORBIDDEN_V1_REFERENCE:
                    forbidden_v1_references.append(path.name)
                candidate = (root / f"{name}.py").resolve()
                if candidate.is_file() and candidate.parent == root:
                    imported.add(candidate)
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and FORBIDDEN_V1_REFERENCE in node.value
            ):
                forbidden_v1_references.append(path.name)
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
        if FORBIDDEN_V1_REFERENCE in source:
            forbidden_v1_references.append(path.name)
        for literal in FORBIDDEN_ROOT_LITERALS:
            if literal in source:
                forbidden_root_literals.append(path.name)
        edges[path] = imported
        pending.extend(imported)

    entries = [
        {
            "module_path_identity_sha256": canonical_sha256({
                "schema": "repository-root-module-path-v1",
                "value": path.name,
            }),
            "file_sha256": file_sha256(path),
            "local_import_path_identity_sha256s": sorted(
                canonical_sha256({
                    "schema": "repository-root-module-path-v1",
                    "value": child.name,
                })
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
    quarantined_reference_identities = sorted(
        canonical_sha256({
            "schema": "opaque-static-boundary-reference-v1",
            "module_path_identity_sha256": canonical_sha256({
                "schema": "repository-root-module-path-v1",
                "value": name,
            }),
            "reference_class": "quarantined_api",
        })
        for name in set(forbidden_v1_references)
    )
    semantic_path_reference_identities = sorted(
        canonical_sha256({
            "schema": "opaque-static-boundary-reference-v1",
            "module_path_identity_sha256": canonical_sha256({
                "schema": "repository-root-module-path-v1",
                "value": name,
            }),
            "reference_class": "semantic_evaluation_path",
        })
        for name in set(forbidden_root_literals)
    )
    authority_guard_bindings = [
        {
            "module_path_identity_sha256": canonical_sha256({
                "schema": "repository-root-module-path-v1",
                "value": name,
            }),
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
        "authority_guard_module_count": len(authority_guard_bindings),
        "authority_guard_source_bindings": authority_guard_bindings,
        "local_closure_path_identity_sha256s": path_identities,
        "local_closure_byte_binding_identity_sha256s": binding_identities,
        "local_import_edge_identity_sha256s": edge_identities,
        "local_closure_content_sha256": canonical_sha256(entries),
        "quarantined_boundary_reference_count": len(
            quarantined_reference_identities
        ),
        "quarantined_boundary_reference_identity_sha256s": (
            quarantined_reference_identities
        ),
        "semantic_evaluation_path_reference_count": len(
            semantic_path_reference_identities
        ),
        "semantic_evaluation_path_reference_identity_sha256s": (
            semantic_path_reference_identities
        ),
        "required_false_authority_rule_count": len(false_authority_keys),
        "true_quality_or_promotion_authority_rule_count": len(
            true_authority_keys
        ),
        "authority_rule_set_sha256": canonical_sha256(
            sorted(false_authority_keys)
        ),
    }


def run_synthetic_guard_tests() -> dict:
    command = (sys.executable, "-m", "pytest", "-q", *SYNTHETIC_GUARD_TEST_NODES)
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("V73C synthetic runtime guard tests failed")
    test_files = sorted({node.split("::", 1)[0] for node in SYNTHETIC_GUARD_TEST_NODES})
    bindings = []
    for relative in test_files:
        path = (ROOT / relative).resolve()
        if not path.is_file() or path.parent != ROOT:
            raise RuntimeError("V73C synthetic guard test source is absent")
        bindings.append({
            "path_identity_sha256": canonical_sha256({
                "schema": "repository-root-test-path-v1",
                "value": relative,
            }),
            "file_sha256": file_sha256(path),
        })
    return {
        "schema": "v73c-synthetic-runtime-guard-test-receipt-v1",
        "passed": True,
        "exit_code": 0,
        "selected_test_node_count": len(SYNTHETIC_GUARD_TEST_NODES),
        "selected_test_command_sha256": canonical_sha256(list(command[2:])),
        "test_source_bindings": bindings,
        "stdout_or_stderr_persisted": False,
    }


def build_audit(synthetic_tests: dict, byte_stable_set: dict) -> dict:
    graph = analyze_import_graph(
        ROOT,
        V73C_ROOT_MODULES,
        authority_guard_modules=AUTHORITY_GUARD_MODULES,
    )
    if (
        graph["mutable_prereg_builder_reachable"] is not False
        or graph["authority_guard_module_count"] != len(AUTHORITY_GUARD_MODULES)
        or graph["required_false_authority_rule_count"]
        != len(REQUIRED_FALSE_AUTHORITY_KEYS)
        or graph["true_quality_or_promotion_authority_rule_count"] != 0
        or graph["semantic_evaluation_path_reference_count"] <= 0
    ):
        raise RuntimeError("V73C runtime authority guards are incomplete")
    if (
        synthetic_tests.get("schema")
        != "v73c-synthetic-runtime-guard-test-receipt-v1"
        or synthetic_tests.get("passed") is not True
        or synthetic_tests.get("exit_code") != 0
        or synthetic_tests.get("stdout_or_stderr_persisted") is not False
        or synthetic_tests.get("selected_test_node_count")
        != len(SYNTHETIC_GUARD_TEST_NODES)
    ):
        raise RuntimeError("V73C synthetic guard-test receipt is invalid")
    if byte_stable_set != verify_byte_stable_runtime_guard_set():
        raise RuntimeError("V73C byte-stable runtime/guard set is invalid")
    audit = {
        "schema": SCHEMA,
        "status": "v73c_runtime_closure_systems_only_no_quality_authority",
        "created_at_utc": "2026-07-17T00:00:00+00:00",
        "import_graph": graph,
        "static_historical_references": {
            "opaque_only": True,
            "opaque_reference_binding_count": (
                graph["quarantined_boundary_reference_count"]
                + graph["semantic_evaluation_path_reference_count"]
            ),
            "does_not_authorize_path_resolution_or_open": True,
            "actual_open_denial_requires_distinct_postrun_receipt": True,
        },
        "synthetic_runtime_guard_tests": synthetic_tests,
        "byte_stable_runtime_guard_set": byte_stable_set,
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
    audit["content_sha256_before_self_field"] = canonical_sha256(audit)
    return audit


def validate_audit(audit: dict) -> None:
    compact = {
        key: value for key, value in audit.items()
        if key != "content_sha256_before_self_field"
    }
    authority = audit.get("authority", {})
    graph = audit.get("import_graph", {})
    static_references = audit.get("static_historical_references", {})
    synthetic_tests = audit.get("synthetic_runtime_guard_tests", {})
    byte_stable_set = audit.get("byte_stable_runtime_guard_set", {})
    if (
        audit.get("schema") != SCHEMA
        or audit.get("status")
        != "v73c_runtime_closure_systems_only_no_quality_authority"
        or audit.get("content_sha256_before_self_field")
        != canonical_sha256(compact)
        or authority.get("systems_trace_only") is not True
        or authority.get(
            "semantic_evaluation_or_quarantined_boundary_resolved"
        ) is not False
        or authority.get("quality_selection_hpo_or_promotion_authorized")
        is not False
        or authority.get("historical_quarantined_lineage_rehabilitated")
        is not False
        or graph.get("required_false_authority_rule_count")
        != len(REQUIRED_FALSE_AUTHORITY_KEYS)
        or graph.get("mutable_prereg_builder_reachable") is not False
        or graph.get("authority_guard_module_count")
        != len(AUTHORITY_GUARD_MODULES)
        or len(graph.get("authority_guard_source_bindings", ()))
        != len(AUTHORITY_GUARD_MODULES)
        or graph.get("true_quality_or_promotion_authority_rule_count") != 0
        or graph.get("root_modules") != list(V73C_ROOT_MODULES)
        or graph.get("semantic_evaluation_path_reference_count", 0) <= 0
        or graph.get("quarantined_boundary_reference_count")
        != len(graph.get("quarantined_boundary_reference_identity_sha256s", ()))
        or graph.get("semantic_evaluation_path_reference_count")
        != len(
            graph.get("semantic_evaluation_path_reference_identity_sha256s", ())
        )
        or static_references.get("opaque_only") is not True
        or static_references.get("does_not_authorize_path_resolution_or_open")
        is not True
        or static_references.get(
            "actual_open_denial_requires_distinct_postrun_receipt"
        ) is not True
        or static_references.get("opaque_reference_binding_count")
        != (
            graph.get("quarantined_boundary_reference_count", 0)
            + graph.get("semantic_evaluation_path_reference_count", 0)
        )
        or synthetic_tests.get("schema")
        != "v73c-synthetic-runtime-guard-test-receipt-v1"
        or synthetic_tests.get("passed") is not True
        or synthetic_tests.get("exit_code") != 0
        or synthetic_tests.get("stdout_or_stderr_persisted") is not False
        or synthetic_tests.get("selected_test_node_count")
        != len(SYNTHETIC_GUARD_TEST_NODES)
        or byte_stable_set != verify_byte_stable_runtime_guard_set()
    ):
        raise RuntimeError("invalid V73C systems-only import-graph audit")


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
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
    byte_stable_set = verify_byte_stable_runtime_guard_set()
    synthetic_tests = run_synthetic_guard_tests()
    value = build_audit(synthetic_tests, byte_stable_set)
    validate_audit(value)
    if args.check:
        persisted = json.loads(args.output.read_text(encoding="ascii"))
        if persisted != value:
            raise RuntimeError("persisted V73C import-graph audit differs")
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
        "quality_hpo_promotion_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
