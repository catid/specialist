#!/usr/bin/env python3
"""Build the content-free exact-path and prefix quarantine guard registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "quarantine_boundary_registry_v3.json"
)
SCHEMA = "specialist-content-free-quarantine-boundary-registry-v3"
EXACT_RELATIVE_PATHS = (
    "data/eval_qa_v3.jsonl",
    "data/eval_qa.jsonl",
    "data/eval_qa_v2.jsonl",
)
RELATIVE_PREFIXES = (
    "data/manual_reviews",
    "experiments/sft_controls",
)
INCIDENT_BINDINGS = (
    (
        "protected_v1_eval_v3",
        "experiments/eggroll_es_hpo/incidents/protected_holdout_access_20260717_v1.json",
        "e20d2129a72fc2d314002a5448a8e8332296b0975e345f40140e6895247978ae",
        "df8856617f5facd6fedac21ab7b653681d2a38484f8e7d93f493bccd47932301",
    ),
    (
        "legacy_eval_collision_family",
        "experiments/eggroll_es_hpo/incidents/legacy_eval_collision_access_20260717_v2.json",
        "351baed6bda7805d5e3d5518ed03622bc13512b90d0df639f657cd57727529e1",
        "68b214cfad7326f15ad7f090130a14b60dab2e2ec478efbf32213e239d1097d4",
    ),
    (
        "recursive_prefix_lookup",
        "experiments/eggroll_es_hpo/incidents/recursive_lookup_access_20260717_v3.json",
        "68652b270d44bef8c33c95df6a39e88534baab8923613a5e8e936ed309d0b550",
        "1e98c7042d1f9279b1b18aba682a65e674f9890465c80bd0622e45c89c7a2907",
    ),
)


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(schema: str, value: str) -> str:
    return canonical_sha256({"schema": schema, "value": value})


def exact_path_identities() -> list[str]:
    return sorted(
        _identity("repository-relative-path-v1", value)
        for value in EXACT_RELATIVE_PATHS
    )


def prefix_identities() -> list[str]:
    return sorted(
        _identity("repository-relative-path-prefix-v1", value)
        for value in RELATIVE_PREFIXES
    )


def _validated_incident_bindings() -> list[dict]:
    bindings = []
    for role, relative, expected_file, expected_content in INCIDENT_BINDINGS:
        path = ROOT / relative
        if file_sha256(path) != expected_file:
            raise RuntimeError("quarantine incident receipt bytes changed")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        compact = {
            key: value for key, value in receipt.items()
            if key != "content_sha256_before_self_field"
        }
        if (
            receipt.get("content_sha256_before_self_field") != expected_content
            or canonical_sha256(compact) != expected_content
        ):
            raise RuntimeError("quarantine incident receipt content changed")
        bindings.append({
            "incident_role": role,
            "receipt_path_identity_sha256": _identity(
                "repository-relative-incident-receipt-path-v1", relative
            ),
            "receipt_file_sha256": expected_file,
            "receipt_content_sha256": expected_content,
        })
    return bindings


def build_registry() -> dict:
    exact = exact_path_identities()
    prefixes = prefix_identities()
    policy = {
        "exact_path_identity_schema": "repository-relative-path-v1",
        "prefix_identity_schema": "repository-relative-path-prefix-v1",
        "candidate_path_form": "repository_relative_lexical_normal_form",
        "repository_relative_to_bound_root": True,
        "path_separator": "forward_slash",
        "path_case_rule": "case_sensitive_no_case_folding",
        "unicode_rule": "exact_codepoints_no_unicode_normalization",
        "dot_segments_and_repeated_separators_collapsed": True,
        "absolute_path_or_parent_traversal_rejected": True,
        "exact_identity_equality_denied": True,
        "prefix_identity_equal_or_descendant_denied": True,
        "lexical_deny_before_resolution_stat_hash_or_open": True,
        "lexically_denied_candidate_is_never_resolved": True,
        "lexically_allowed_resolution_rechecked_before_metadata_or_open": True,
        "resolved_target_outside_repository_root_denied": True,
        "denied_api_classes": [
            "filesystem_open",
            "filesystem_resolution",
            "path_metadata",
        ],
    }
    registry = {
        "schema": SCHEMA,
        "status": "active_fail_closed_content_free_quarantine_boundary",
        "created_at_utc": "2026-07-17T00:00:00+00:00",
        "exact_path_identity_count": len(exact),
        "exact_path_identity_sha256": exact,
        "prefix_identity_count": len(prefixes),
        "prefix_identity_sha256": prefixes,
        "ancestor_denial_policy": policy,
        "ancestor_denial_policy_sha256": canonical_sha256(policy),
        "incident_bindings": _validated_incident_bindings(),
        "content_minimization": {
            "plaintext_boundary_paths_persisted": False,
            "source_bytes_or_semantics_persisted": False,
            "individual_prefix_file_inventory_persisted": False,
        },
        "implementation": {
            "builder_path_identity_sha256": _identity(
                "repository-root-python-path-v1", Path(__file__).name
            ),
            "builder_file_sha256": file_sha256(Path(__file__)),
        },
    }
    registry["content_sha256_before_self_field"] = canonical_sha256(registry)
    return registry


def validate_registry(registry: dict) -> None:
    compact = {
        key: value for key, value in registry.items()
        if key != "content_sha256_before_self_field"
    }
    policy = registry.get("ancestor_denial_policy", {})
    minimization = registry.get("content_minimization", {})
    if (
        registry.get("schema") != SCHEMA
        or registry.get("status")
        != "active_fail_closed_content_free_quarantine_boundary"
        or registry.get("content_sha256_before_self_field")
        != canonical_sha256(compact)
        or registry.get("exact_path_identity_count") != 3
        or registry.get("exact_path_identity_sha256") != exact_path_identities()
        or registry.get("prefix_identity_count") != 2
        or registry.get("prefix_identity_sha256") != prefix_identities()
        or registry.get("ancestor_denial_policy_sha256")
        != canonical_sha256(policy)
        or policy.get("prefix_identity_equal_or_descendant_denied") is not True
        or policy.get("exact_path_identity_schema")
        != "repository-relative-path-v1"
        or policy.get("prefix_identity_schema")
        != "repository-relative-path-prefix-v1"
        or policy.get("path_case_rule") != "case_sensitive_no_case_folding"
        or policy.get("path_separator") != "forward_slash"
        or policy.get("lexical_deny_before_resolution_stat_hash_or_open")
        is not True
        or policy.get("lexically_denied_candidate_is_never_resolved")
        is not True
        or policy.get(
            "lexically_allowed_resolution_rechecked_before_metadata_or_open"
        )
        is not True
        or registry.get("incident_bindings") != _validated_incident_bindings()
        or minimization.get("plaintext_boundary_paths_persisted") is not False
        or minimization.get("source_bytes_or_semantics_persisted") is not False
        or minimization.get("individual_prefix_file_inventory_persisted")
        is not False
        or registry.get("implementation", {}).get("builder_file_sha256")
        != file_sha256(Path(__file__))
    ):
        raise RuntimeError("invalid content-free quarantine boundary registry")


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
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
    value = build_registry()
    validate_registry(value)
    if args.check:
        persisted = json.loads(args.output.read_text(encoding="utf-8"))
        if persisted != value:
            raise RuntimeError("persisted quarantine boundary registry differs")
    else:
        _write_exclusive(args.output, value)
    print(json.dumps({
        "content_sha256": value["content_sha256_before_self_field"],
        "exact_path_identities": 3,
        "prefix_identities": 2,
        "plaintext_boundary_paths_persisted": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
