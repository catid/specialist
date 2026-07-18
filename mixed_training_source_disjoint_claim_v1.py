#!/usr/bin/env python3
"""Content-free source-disjoint claim contracts for the mixed snapshot.

This module deliberately does not know how to open protected, development,
final, holdout, OOD, terminal, incident, or manual-review data.  A claim is
made from exact live *training* inputs plus an already sealed source-split
authority whose public train membership is explicit and whose final membership
is aggregate-only.

The production flow has three separately content-addressed stages:

1. preregister an exact request from live safe inputs and normalized units;
2. have an independently invoked claim runner bind that request and emit only
   hashes, aggregate counts, and boundary booleans;
3. consume the authorization only with an externally supplied SHA-256 and seal
   the launch extension after recomputing the live input/candidate commitments.

This module's own CLI remains validation-only.  The mixed-snapshot builder
supplies live normalized units through three separate immutable stage modes.
Neither validator can accept an authorization without independently supplied
request and authorization hashes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = ROOT / "data/training_inventory/mixed_training_snapshot_v1"
CLAIM_REQUEST_PATH = OUTPUT_DIRECTORY / "source_disjoint_claim_request_v1.json"
CLAIM_AUTHORIZATION_PATH = (
    OUTPUT_DIRECTORY / "source_disjoint_claim_authorization_v1.json"
)
EXTENSION_PATH = OUTPUT_DIRECTORY / "source_disjoint_extension_v1.json"

REQUEST_SCHEMA = "mixed-training-source-disjoint-claim-request-v1"
AUTHORIZATION_SCHEMA = "mixed-training-source-disjoint-claim-authorization-v1"
EXTENSION_SCHEMA = "mixed-training-source-disjoint-extension-v1"
STATIC_INPUT_SCHEMA = "mixed-training-static-input-bindings-v1"
CANDIDATE_SET_SCHEMA = "mixed-training-candidate-set-commitment-v1"
OPAQUE_CONTRACT_SCHEMA = "mixed-training-opaque-collision-contract-v1"
IDENTITY_SCHEME = "sha256-canonical-mixed-training-unit-source-lineage-v1"

REQUEST_STATUS = "sealed_pending_independent_claim"
AUTHORIZATION_STATUS = "sealed_passed"
EXTENSION_STATUS = "accepted"

VARIANTS = ("protocol_core_100k", "full_authorized_markdown")
COMPONENTS = ("seed_qa", "generated_domain", "raw_markdown", "replay")
STREAM_BY_COMPONENT = {
    "seed_qa": "domain_qa",
    "generated_domain": "domain_qa",
    "raw_markdown": "raw_markdown",
    "replay": "replay",
}
FORMAT_BY_COMPONENT = {
    "seed_qa": "chat_assistant_only",
    "generated_domain": "chat_assistant_only",
    "raw_markdown": "raw_markdown_causal",
    "replay": "chat_assistant_only",
}
STREAMS = ("domain_qa", "raw_markdown", "replay")

STATIC_INPUT_ROLES = (
    "source_split_authority",
    "seed_qa_source",
    "seed_qa_semantic_authority",
    "seed_qa_decision_bundle",
    "core_markdown",
    "full_markdown",
    "full_markdown_manifest",
    "project_training_authorization",
    "replay_data",
    "replay_manifest",
    "generated_selection_contract",
    "generated_manifest",
    "generated_report",
    "generated_data",
)

PRODUCTION_ROLE_PATHS = {
    "source_split_authority": (
        ROOT / "data/training_inventory/source_group_split_authority_v1.json"
    ),
    "seed_qa_source": (
        ROOT
        / "data/training_inventory/high_information_domain_corpus_v1"
        / "seed_qa_train.jsonl"
    ),
    "seed_qa_semantic_authority": (
        ROOT
        / "data/training_inventory/high_information_domain_corpus_v1"
        / "seed_qa_semantic_authority_v1.json"
    ),
    "seed_qa_decision_bundle": (
        ROOT
        / "data/training_inventory/high_information_domain_corpus_v1"
        / "seed_qa_semantic_review_v1/decisions.jsonl"
    ),
    "core_markdown": (
        ROOT
        / "data/training_inventory/high_information_domain_corpus_v1"
        / "raw_continuation_train.jsonl"
    ),
    "full_markdown": (
        ROOT / "data/training_inventory/full_train_markdown_cpt_v1/train.jsonl"
    ),
    "full_markdown_manifest": (
        ROOT / "data/training_inventory/full_train_markdown_cpt_v1/manifest.json"
    ),
    "project_training_authorization": (
        ROOT / "data/site_corpora/registry/project_training_authorization_v1.json"
    ),
    "replay_data": (
        ROOT
        / "data/general_replay_v1/replay_authority_v1_150k"
        / "general_replay_authority_v1_150k.jsonl"
    ),
    "replay_manifest": (
        ROOT
        / "data/general_replay_v1/replay_authority_v1_150k/manifest.json"
    ),
    "generated_selection_contract": (
        ROOT
        / "data/training_inventory/high_information_domain_corpus_v1"
        / "category_balanced_candidate_selection_contract.json"
    ),
    "generated_manifest": (
        ROOT
        / "data/training_inventory/high_information_domain_corpus_v1"
        / "generated_domain_authority_v1/manifest.json"
    ),
    "generated_report": (
        ROOT
        / "data/training_inventory/high_information_domain_corpus_v1"
        / "generated_domain_authority_v1/report.json"
    ),
    "generated_data": (
        ROOT
        / "data/training_inventory/high_information_domain_corpus_v1"
        / "generated_domain_authority_v1/train.jsonl"
    ),
}

SOURCE_SPLIT_DECLARED_PATH = (
    "data/training_inventory/source_group_split_authority_v1.json"
)
RUNNER_DECLARED_PATH = "build_mixed_training_snapshot_v1.py"
CONTRACT_MODULE_DECLARED_PATH = "mixed_training_source_disjoint_claim_v1.py"
COMMAND_CONTRACT = {
    "request_argv": [
        "python", RUNNER_DECLARED_PATH, "--emit-source-disjoint-request"
    ],
    "authorization_argv": [
        "python",
        RUNNER_DECLARED_PATH,
        "--emit-source-disjoint-authorization",
        "--source-disjoint-request-sha256",
        "<independently-preregistered-request-sha256>",
    ],
    "final_build_argv": [
        "python",
        RUNNER_DECLARED_PATH,
        "--source-disjoint-request-sha256",
        "<independently-preregistered-request-sha256>",
        "--source-disjoint-authorization-sha256",
        "<independently-pinned-authorization-sha256>",
    ],
    "stages_are_separate_invocations": True,
    "environment_influences_claim_semantics": False,
    "output": "aggregate_hash_count_boolean_only_v1",
}

OUTPUT_BOUNDARY = {
    "protected_source_content_opened": False,
    "protected_identifiers_disclosed": False,
    "protected_text_urls_answers_or_per_item_metrics_emitted": False,
}
ZERO_CLAIMS = {
    "domain_source_group_not_in_train_count": 0,
    "domain_source_document_not_in_train_count": 0,
    "cross_split_source_group_collision_count": 0,
    "candidate_identity_mismatch_count": 0,
    "replay_authority_mismatch_count": 0,
}
MEMBERSHIP_KEYS = {
    "train_source_group_membership_commitment_sha256",
    "train_source_document_membership_commitment_sha256",
    "replay_source_group_membership_commitment_sha256",
}

FORBIDDEN_PATH_TOKENS = frozenset(
    {
        "benchmark",
        "benchmarks",
        "dev",
        "development",
        "developments",
        "eval",
        "evaluation",
        "evaluations",
        "final",
        "finals",
        "heldout",
        "holdout",
        "holdouts",
        "incident",
        "incidents",
        "manualreview",
        "manualreviews",
        "ood",
        "protected",
        "terminal",
        "terminals",
    }
)
FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "answer",
        "answers",
        "completion",
        "completions",
        "input_ids",
        "labels",
        "messages",
        "per_item_metrics",
        "question",
        "questions",
        "record_id",
        "records",
        "source_document_id",
        "source_group_id",
        "text",
        "texts",
        "unit_id",
        "url",
        "urls",
    }
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")

UNIT_KEYS = {
    "component",
    "unit_id",
    "source_group_id",
    "source_document_id",
    "training_format",
    "input_ids",
    "labels",
    "budget_token_count",
}
RUNNER_KEYS = {
    "path",
    "file_sha256",
    "contract_module_path",
    "contract_module_file_sha256",
    "command_contract_sha256",
}
SOURCE_SPLIT_KEYS = {
    "path",
    "file_sha256",
    "content_sha256",
    "train_source_group_membership_commitment_sha256",
    "final_source_group_membership_commitment_sha256",
    "final_records_redacted",
}
GENERATED_KEYS = {
    "manifest_file_sha256",
    "manifest_content_sha256",
    "report_file_sha256",
    "report_content_sha256",
    "dataset_file_sha256",
    "seed_replacement_receipt_sha256",
}
SEED_KEYS = {
    "authority_file_sha256",
    "authority_content_sha256",
    "decision_bundle_file_sha256",
    "admitted_record_identity_commitment_sha256",
}
CANDIDATE_SET_KEYS = {
    "schema",
    "unit_count",
    "source_group_count",
    "source_document_count",
    "component_unit_counts",
    "budget_tokens_by_stream",
    "unit_identity_commitment_sha256",
    "source_group_membership_commitment_sha256",
    "source_document_membership_commitment_sha256",
}
REQUEST_KEYS = {
    "schema",
    "status",
    "identity_scheme",
    "claim_runner",
    "source_split_authority",
    "static_inputs",
    "generated_domain_authority",
    "seed_qa_semantic_authority",
    "candidate_sets",
    "membership_commitments",
    "required_claims",
    "output_boundary",
    "content_sha256_before_self_field",
}
AUTHORIZATION_KEYS = {
    "schema",
    "status",
    "identity_scheme",
    "claim_request",
    "claim_runner",
    "source_split_authority",
    "static_input_set_commitment_sha256",
    "candidate_sets",
    "membership_commitments",
    "claims",
    "boundary",
    "opaque_receipt_sha256",
    "content_sha256_before_self_field",
}
EXTENSION_KEYS = {
    "schema",
    "status",
    "accepted_for_training",
    "claim_request",
    "claim_authorization",
    "claim_runner",
    "source_split_authority",
    "static_inputs",
    "generated_domain_authority",
    "seed_qa_semantic_authority",
    "candidate_sets",
    "membership_commitments",
    "opaque_collision_contract",
    "content_sha256_before_self_field",
}


class ContractError(RuntimeError):
    """Raised when a source-disjoint contract fails closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def artifact_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ContractError(f"{label}: exact field contract changed")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ContractError(f"{label}: lowercase SHA-256 required")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{label}: nonnegative integer required")
    return value


def _path_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for component in path.parts:
        collapsed = re.sub(r"[^a-z0-9]", "", component.casefold())
        if collapsed:
            tokens.add(collapsed)
        tokens.update(
            part
            for part in re.split(r"[^a-z0-9]+", component.casefold())
            if part
        )
    return tokens


def _validate_declared_safe_path(value: Any, expected: str, label: str) -> str:
    if not isinstance(value, str) or value != expected:
        raise ContractError(f"{label}: declared path changed")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ContractError(f"{label}: unsafe declared path")
    if _path_tokens(path) & FORBIDDEN_PATH_TOKENS:
        raise ContractError(f"{label}: forbidden path class")
    return value


def secure_regular_file(path: Path, *, expected: Path, label: str) -> Path:
    """Reject forbidden lexical paths and every symlink/hard-link alias."""

    lexical = Path(os.path.abspath(os.fspath(path)))
    expected_lexical = Path(os.path.abspath(os.fspath(expected)))
    if _path_tokens(lexical) & FORBIDDEN_PATH_TOKENS:
        raise ContractError(f"{label}: forbidden path class")
    if lexical != expected_lexical:
        raise ContractError(f"{label}: path is not the fixed expected path")
    current = Path(lexical.anchor)
    metadata = None
    for component in lexical.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ContractError(f"{label}: required file is missing") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ContractError(f"{label}: symlink aliases are forbidden")
    if metadata is None or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"{label}: regular file required")
    if metadata.st_nlink != 1:
        raise ContractError(f"{label}: hard-link aliases are forbidden")
    return lexical


def _validate_unique_files(paths: Iterable[Path]) -> None:
    identities: set[tuple[int, int]] = set()
    for path in paths:
        metadata = path.stat(follow_symlinks=False)
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in identities:
            raise ContractError("static inputs alias one another")
        identities.add(identity)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    if "content_sha256_before_self_field" in value:
        raise ContractError("artifact was already self-addressed")
    sealed = copy.deepcopy(dict(value))
    sealed["content_sha256_before_self_field"] = canonical_sha256(sealed)
    return sealed


def _validate_self_address(value: dict[str, Any], label: str) -> None:
    claimed = _require_sha256(
        value.get("content_sha256_before_self_field"),
        f"{label}.content_sha256_before_self_field",
    )
    unsigned = copy.deepcopy(value)
    unsigned.pop("content_sha256_before_self_field")
    if claimed != canonical_sha256(unsigned):
        raise ContractError(f"{label}: self-addressed content identity changed")


def _assert_content_free_output(value: Any, location: str = "artifact") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_OUTPUT_KEYS:
                raise ContractError(f"{location}: forbidden semantic field {key}")
            _assert_content_free_output(item, f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_content_free_output(item, f"{location}[{index}]")
        return
    if isinstance(value, str) and value.casefold().startswith(("http://", "https://")):
        raise ContractError(f"{location}: URL disclosure is forbidden")


def _load_canonical_artifact(
    path: Path,
    *,
    expected_path: Path,
    expected_file_sha256: str,
    label: str,
) -> dict[str, Any]:
    expected_digest = _require_sha256(expected_file_sha256, f"{label} expected digest")
    safe = secure_regular_file(path, expected=expected_path, label=label)
    raw = safe.read_bytes()
    if sha256_bytes(raw) != expected_digest:
        raise ContractError(f"{label}: independently supplied file digest differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label}: canonical JSON required") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label}: JSON object required")
    if artifact_bytes(value) != raw:
        raise ContractError(f"{label}: noncanonical artifact bytes")
    _validate_self_address(value, label)
    return value


def _validate_runner(value: Any) -> dict[str, Any]:
    runner = _require_exact_keys(value, RUNNER_KEYS, "claim runner")
    _validate_declared_safe_path(
        runner.get("path"), RUNNER_DECLARED_PATH, "claim runner"
    )
    _require_sha256(runner.get("file_sha256"), "claim runner file digest")
    _validate_declared_safe_path(
        runner.get("contract_module_path"),
        CONTRACT_MODULE_DECLARED_PATH,
        "claim contract module",
    )
    _require_sha256(
        runner.get("contract_module_file_sha256"),
        "claim contract module file digest",
    )
    expected_contract = canonical_sha256(COMMAND_CONTRACT)
    if runner.get("command_contract_sha256") != expected_contract:
        raise ContractError("claim runner command contract changed")
    return copy.deepcopy(runner)


def build_runner_binding(
    runner_path: Path,
    *,
    expected_runner_path: Path,
) -> dict[str, Any]:
    safe = secure_regular_file(
        runner_path,
        expected=expected_runner_path,
        label="claim runner implementation",
    )
    contract_module = Path(__file__).resolve()
    safe_contract_module = secure_regular_file(
        contract_module,
        expected=contract_module,
        label="claim contract module implementation",
    )
    return {
        "path": RUNNER_DECLARED_PATH,
        "file_sha256": file_sha256(safe),
        "contract_module_path": CONTRACT_MODULE_DECLARED_PATH,
        "contract_module_file_sha256": file_sha256(safe_contract_module),
        "command_contract_sha256": canonical_sha256(COMMAND_CONTRACT),
    }


def _validate_source_split(value: Any) -> dict[str, Any]:
    receipt = _require_exact_keys(
        value, SOURCE_SPLIT_KEYS, "source split authority"
    )
    _validate_declared_safe_path(
        receipt.get("path"), SOURCE_SPLIT_DECLARED_PATH, "source split authority"
    )
    for key in (
        "file_sha256",
        "content_sha256",
        "train_source_group_membership_commitment_sha256",
        "final_source_group_membership_commitment_sha256",
    ):
        _require_sha256(receipt.get(key), f"source split authority {key}")
    if receipt.get("final_records_redacted") is not True:
        raise ContractError("source split authority must redact final records")
    return copy.deepcopy(receipt)


def _validate_generated(value: Any) -> dict[str, Any]:
    receipt = _require_exact_keys(value, GENERATED_KEYS, "generated authority")
    for key in GENERATED_KEYS:
        _require_sha256(receipt.get(key), f"generated authority {key}")
    return copy.deepcopy(receipt)


def _validate_seed(value: Any) -> dict[str, Any]:
    receipt = _require_exact_keys(value, SEED_KEYS, "seed QA authority")
    for key in SEED_KEYS:
        _require_sha256(receipt.get(key), f"seed QA authority {key}")
    return copy.deepcopy(receipt)


def build_static_input_bindings(
    paths: Mapping[str, Path],
    *,
    expected_paths: Mapping[str, Path],
) -> dict[str, Any]:
    if set(paths) != set(STATIC_INPUT_ROLES) or set(expected_paths) != set(
        STATIC_INPUT_ROLES
    ):
        raise ContractError("static input role set changed")
    safe_paths: list[Path] = []
    bindings = []
    for index, role in enumerate(STATIC_INPUT_ROLES):
        safe = secure_regular_file(
            paths[role], expected=expected_paths[role], label=f"static input {role}"
        )
        safe_paths.append(safe)
        bindings.append(
            {"index": index, "role": role, "file_sha256": file_sha256(safe)}
        )
    _validate_unique_files(safe_paths)
    return {
        "schema": STATIC_INPUT_SCHEMA,
        "bindings": bindings,
        "bindings_commitment_sha256": canonical_sha256(bindings),
    }


def _validate_static_inputs(value: Any) -> dict[str, Any]:
    static = _require_exact_keys(
        value,
        {"schema", "bindings", "bindings_commitment_sha256"},
        "static input bindings",
    )
    if static.get("schema") != STATIC_INPUT_SCHEMA:
        raise ContractError("static input binding schema changed")
    bindings = static.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != len(STATIC_INPUT_ROLES):
        raise ContractError("static input binding count changed")
    normalized = []
    for index, (role, raw) in enumerate(zip(STATIC_INPUT_ROLES, bindings)):
        binding = _require_exact_keys(
            raw, {"index", "role", "file_sha256"}, f"static binding {index}"
        )
        if binding.get("index") != index or binding.get("role") != role:
            raise ContractError("static input binding order or role changed")
        normalized.append(
            {
                "index": index,
                "role": role,
                "file_sha256": _require_sha256(
                    binding.get("file_sha256"), f"static binding {role}"
                ),
            }
        )
    commitment = _require_sha256(
        static.get("bindings_commitment_sha256"), "static input commitment"
    )
    if commitment != canonical_sha256(normalized):
        raise ContractError("static input commitment changed")
    return {
        "schema": STATIC_INPUT_SCHEMA,
        "bindings": normalized,
        "bindings_commitment_sha256": commitment,
    }


def _binding_digest(static: Mapping[str, Any], role: str) -> str:
    index = STATIC_INPUT_ROLES.index(role)
    binding = static["bindings"][index]
    if binding["role"] != role:
        raise ContractError("static input role lookup changed")
    return binding["file_sha256"]


def _crosscheck_authority_files(
    static: Mapping[str, Any],
    source_split: Mapping[str, Any],
    generated: Mapping[str, Any],
    seed: Mapping[str, Any],
) -> None:
    expected = {
        "source_split_authority": source_split["file_sha256"],
        "seed_qa_semantic_authority": seed["authority_file_sha256"],
        "seed_qa_decision_bundle": seed["decision_bundle_file_sha256"],
        "generated_manifest": generated["manifest_file_sha256"],
        "generated_report": generated["report_file_sha256"],
        "generated_data": generated["dataset_file_sha256"],
    }
    for role, digest in expected.items():
        if _binding_digest(static, role) != digest:
            raise ContractError(f"{role}: authority receipt differs from live file")


def _load_live_json_role(
    paths: Mapping[str, Path],
    expected_paths: Mapping[str, Path],
    role: str,
) -> dict[str, Any]:
    safe = secure_regular_file(
        paths[role], expected=expected_paths[role], label=f"live metadata {role}"
    )
    try:
        value = json.loads(safe.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{role}: valid JSON metadata required") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{role}: metadata object required")
    return value


def _declared_content_hash(value: Mapping[str, Any], label: str) -> str:
    claimed = _require_sha256(
        value.get("content_sha256_before_self_field"), f"{label} content digest"
    )
    unsigned = copy.deepcopy(dict(value))
    unsigned.pop("content_sha256_before_self_field")
    alternate = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if claimed not in {canonical_sha256(unsigned), alternate}:
        raise ContractError(f"{label}: stale self-addressed metadata")
    return claimed


def _crosscheck_live_authority_metadata(
    paths: Mapping[str, Path],
    expected_paths: Mapping[str, Path],
    source_split: Mapping[str, Any],
    generated: Mapping[str, Any],
    seed: Mapping[str, Any],
) -> None:
    """Bind receipt content fields to fixed, safe metadata inputs.

    This intentionally opens only the three already registered training
    metadata roles plus the generated report.  It never follows a path stored
    inside one of those manifests and never opens a development/final source.
    """

    split_value = _load_live_json_role(
        paths, expected_paths, "source_split_authority"
    )
    if (
        split_value.get("schema")
        != "specialist-source-group-split-authority-v1"
        or split_value.get("status")
        != "sealed_source_disjoint_assignment_launch_still_gated"
        or _declared_content_hash(split_value, "source split authority")
        != source_split["content_sha256"]
    ):
        raise ContractError("source split authority metadata identity changed")
    assignments = split_value.get("assignments")
    if not isinstance(assignments, dict):
        raise ContractError("source split authority assignments are absent")
    train = assignments.get("train")
    final = assignments.get("final")
    train_records = train.get("records") if isinstance(train, dict) else None
    if not isinstance(train_records, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("source_group_id"), str)
        or not item["source_group_id"]
        for item in train_records
    ):
        raise ContractError("source split train membership records are absent")
    train_group_commitment = canonical_sha256(
        sorted(item["source_group_id"] for item in train_records)
    )
    if (
        not isinstance(train, dict)
        or not isinstance(final, dict)
        or train.get("source_group_membership_commitment_sha256")
        != source_split["train_source_group_membership_commitment_sha256"]
        or train_group_commitment
        != source_split["train_source_group_membership_commitment_sha256"]
        or final.get("source_group_membership_commitment_sha256")
        != source_split["final_source_group_membership_commitment_sha256"]
        or final.get("records_redacted") is not True
        or "records" in final
    ):
        raise ContractError("source split train/final commitment changed")

    generated_manifest = _load_live_json_role(
        paths, expected_paths, "generated_manifest"
    )
    if (
        _declared_content_hash(generated_manifest, "generated manifest")
        != generated["manifest_content_sha256"]
    ):
        raise ContractError("generated manifest content identity changed")
    dataset = generated_manifest.get("dataset")
    report = generated_manifest.get("report")
    replacement = generated_manifest.get("seed_qa_replacement")
    if (
        not isinstance(dataset, dict)
        or dataset.get("file_sha256") != generated["dataset_file_sha256"]
        or not isinstance(report, dict)
        or report.get("file_sha256") != generated["report_file_sha256"]
        or not isinstance(replacement, dict)
        or canonical_sha256(replacement)
        != generated["seed_replacement_receipt_sha256"]
    ):
        raise ContractError("generated manifest transitive receipts changed")
    generated_report = _load_live_json_role(
        paths, expected_paths, "generated_report"
    )
    if (
        _declared_content_hash(generated_report, "generated report")
        != generated["report_content_sha256"]
    ):
        raise ContractError("generated report content identity changed")

    seed_authority = _load_live_json_role(
        paths, expected_paths, "seed_qa_semantic_authority"
    )
    if (
        _declared_content_hash(seed_authority, "seed QA semantic authority")
        != seed["authority_content_sha256"]
        or seed_authority.get("admitted_record_identity_commitment_sha256")
        != seed["admitted_record_identity_commitment_sha256"]
    ):
        raise ContractError("seed QA semantic authority identity changed")
    decision_bundle = seed_authority.get("decision_bundle")
    if (
        not isinstance(decision_bundle, dict)
        or decision_bundle.get("file_sha256")
        != seed["decision_bundle_file_sha256"]
    ):
        raise ContractError("seed QA decision bundle receipt changed")


def _hash_string(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_unit(value: Any, *, location: str) -> tuple[dict[str, Any], str]:
    unit = _require_exact_keys(value, UNIT_KEYS, location)
    component = unit.get("component")
    if component not in COMPONENTS:
        raise ContractError(f"{location}: unknown component")
    for key in ("unit_id", "source_group_id", "source_document_id"):
        if not isinstance(unit.get(key), str) or not unit[key]:
            raise ContractError(f"{location}: nonempty {key} required")
    if unit.get("training_format") != FORMAT_BY_COMPONENT[component]:
        raise ContractError(f"{location}: training format differs from component")
    input_ids = unit.get("input_ids")
    labels = unit.get("labels")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or not isinstance(labels, list)
        or len(labels) != len(input_ids)
        or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in input_ids
        )
        or any(
            not isinstance(token, int)
            or isinstance(token, bool)
            or token < -100
            or (-100 < token < 0)
            for token in labels
        )
    ):
        raise ContractError(f"{location}: token or label contract changed")
    supervised = sum(label != -100 for label in labels)
    budget = _require_nonnegative_int(
        unit.get("budget_token_count"), f"{location}.budget_token_count"
    )
    if budget <= 0 or budget != supervised:
        raise ContractError(f"{location}: budget is not backed by supervised labels")
    if component == "raw_markdown":
        if labels != input_ids:
            raise ContractError(f"{location}: Markdown must supervise every token")
    elif -100 not in labels:
        raise ContractError(f"{location}: chat inputs must preserve masked context")
    commitment_payload = {
        "component": component,
        "unit_id_sha256": _hash_string(unit["unit_id"]),
        "source_group_id_sha256": _hash_string(unit["source_group_id"]),
        "source_document_id_sha256": _hash_string(unit["source_document_id"]),
        "training_format": unit["training_format"],
        "input_ids_sha256": canonical_sha256(input_ids),
        "labels_sha256": canonical_sha256(labels),
        "budget_token_count": budget,
    }
    normalized = {
        "component": component,
        "unit_id": unit["unit_id"],
        "source_group_id": unit["source_group_id"],
        "source_document_id": unit["source_document_id"],
        "training_format": unit["training_format"],
        "input_ids": list(input_ids),
        "labels": list(labels),
        "budget_token_count": budget,
    }
    return normalized, canonical_sha256(commitment_payload)


def _candidate_set(
    units: Sequence[Mapping[str, Any]], *, variant: str
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], str]]]:
    if not isinstance(units, Sequence) or isinstance(units, (str, bytes)) or not units:
        raise ContractError(f"{variant}: nonempty candidate units required")
    normalized: list[tuple[dict[str, Any], str]] = []
    unit_ids: set[str] = set()
    unit_digests: set[str] = set()
    component_counts = Counter()
    stream_tokens = Counter()
    source_groups: set[str] = set()
    source_documents: set[str] = set()
    for index, raw in enumerate(units):
        unit, digest = _normalize_unit(raw, location=f"{variant}[{index}]")
        if unit["unit_id"] in unit_ids or digest in unit_digests:
            raise ContractError(f"{variant}: duplicate candidate unit")
        unit_ids.add(unit["unit_id"])
        unit_digests.add(digest)
        normalized.append((unit, digest))
        component_counts[unit["component"]] += 1
        stream_tokens[STREAM_BY_COMPONENT[unit["component"]]] += unit[
            "budget_token_count"
        ]
        source_groups.add(unit["source_group_id"])
        source_documents.add(unit["source_document_id"])
    if any(component_counts[component] <= 0 for component in COMPONENTS):
        raise ContractError(f"{variant}: every training component must be represented")
    result = {
        "schema": CANDIDATE_SET_SCHEMA,
        "unit_count": len(normalized),
        "source_group_count": len(source_groups),
        "source_document_count": len(source_documents),
        "component_unit_counts": {
            component: component_counts[component] for component in COMPONENTS
        },
        "budget_tokens_by_stream": {
            stream: stream_tokens[stream] for stream in STREAMS
        },
        "unit_identity_commitment_sha256": canonical_sha256(
            sorted(unit_digests)
        ),
        "source_group_membership_commitment_sha256": canonical_sha256(
            sorted(_hash_string(item) for item in source_groups)
        ),
        "source_document_membership_commitment_sha256": canonical_sha256(
            sorted(_hash_string(item) for item in source_documents)
        ),
    }
    return result, normalized


def build_candidate_sets(
    units_by_variant: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[dict[str, Any], dict[str, list[tuple[dict[str, Any], str]]]]:
    if set(units_by_variant) != set(VARIANTS):
        raise ContractError("candidate variant set changed")
    commitments = {}
    normalized = {}
    cross_variant_unit_ids: dict[str, str] = {}
    for variant in VARIANTS:
        commitment, rows = _candidate_set(units_by_variant[variant], variant=variant)
        commitments[variant] = commitment
        normalized[variant] = rows
        for unit, digest in rows:
            prior = cross_variant_unit_ids.setdefault(unit["unit_id"], digest)
            if prior != digest:
                raise ContractError("same unit ID has different cross-variant content")
    return commitments, normalized


def _validate_candidate_set(value: Any, *, variant: str) -> dict[str, Any]:
    candidate = _require_exact_keys(value, CANDIDATE_SET_KEYS, variant)
    if candidate.get("schema") != CANDIDATE_SET_SCHEMA:
        raise ContractError(f"{variant}: candidate-set schema changed")
    for key in ("unit_count", "source_group_count", "source_document_count"):
        if _require_nonnegative_int(candidate.get(key), f"{variant}.{key}") <= 0:
            raise ContractError(f"{variant}.{key}: positive count required")
    component_counts = _require_exact_keys(
        candidate.get("component_unit_counts"), set(COMPONENTS), f"{variant} components"
    )
    if any(
        _require_nonnegative_int(component_counts[key], f"{variant}.{key}") <= 0
        for key in COMPONENTS
    ) or sum(component_counts.values()) != candidate["unit_count"]:
        raise ContractError(f"{variant}: component counts changed")
    budgets = _require_exact_keys(
        candidate.get("budget_tokens_by_stream"), set(STREAMS), f"{variant} streams"
    )
    if any(
        _require_nonnegative_int(budgets[key], f"{variant}.{key}") <= 0
        for key in STREAMS
    ):
        raise ContractError(f"{variant}: stream budgets changed")
    for key in (
        "unit_identity_commitment_sha256",
        "source_group_membership_commitment_sha256",
        "source_document_membership_commitment_sha256",
    ):
        _require_sha256(candidate.get(key), f"{variant}.{key}")
    return copy.deepcopy(candidate)


def _validate_candidate_sets(value: Any) -> dict[str, Any]:
    candidates = _require_exact_keys(value, set(VARIANTS), "candidate sets")
    return {
        variant: _validate_candidate_set(candidates[variant], variant=variant)
        for variant in VARIANTS
    }


def _validate_membership_set(values: Any, label: str) -> frozenset[str]:
    if not isinstance(values, frozenset) or not values or any(
        not isinstance(item, str) or not item for item in values
    ):
        raise ContractError(f"{label}: explicit nonempty frozen string set required")
    return values


def _build_membership_commitments(
    *,
    source_split_authority: Mapping[str, Any],
    train_source_group_ids: frozenset[str],
    train_source_document_ids: frozenset[str],
    replay_source_group_ids: frozenset[str],
) -> dict[str, str]:
    groups = _validate_membership_set(
        train_source_group_ids, "train source groups"
    )
    documents = _validate_membership_set(
        train_source_document_ids, "train source documents"
    )
    replay = _validate_membership_set(replay_source_group_ids, "replay source groups")
    group_commitment = canonical_sha256(sorted(groups))
    if (
        group_commitment
        != source_split_authority[
            "train_source_group_membership_commitment_sha256"
        ]
    ):
        raise ContractError(
            "train source-group membership set differs from sealed split authority"
        )
    return {
        "train_source_group_membership_commitment_sha256": group_commitment,
        "train_source_document_membership_commitment_sha256": canonical_sha256(
            sorted(documents)
        ),
        "replay_source_group_membership_commitment_sha256": canonical_sha256(
            sorted(replay)
        ),
    }


def _validate_membership_commitments(
    value: Any, *, source_split_authority: Mapping[str, Any]
) -> dict[str, str]:
    commitments = _require_exact_keys(
        value, MEMBERSHIP_KEYS, "membership commitments"
    )
    normalized = {
        key: _require_sha256(commitments.get(key), f"membership commitment {key}")
        for key in MEMBERSHIP_KEYS
    }
    if (
        normalized["train_source_group_membership_commitment_sha256"]
        != source_split_authority[
            "train_source_group_membership_commitment_sha256"
        ]
    ):
        raise ContractError(
            "train source-group commitment differs from source split authority"
        )
    return normalized


def _validate_boundary(value: Any, label: str) -> dict[str, bool]:
    boundary = _require_exact_keys(value, set(OUTPUT_BOUNDARY), label)
    if boundary != OUTPUT_BOUNDARY:
        raise ContractError(f"{label}: protected-output boundary changed")
    return copy.deepcopy(OUTPUT_BOUNDARY)


def _validate_claim_counts(value: Any, label: str) -> dict[str, int]:
    claims = _require_exact_keys(value, set(ZERO_CLAIMS), label)
    normalized = {
        key: _require_nonnegative_int(claims[key], f"{label}.{key}")
        for key in ZERO_CLAIMS
    }
    if normalized != ZERO_CLAIMS:
        raise ContractError(f"{label}: one or more source-disjoint claims failed")
    return normalized


def build_claim_request(
    *,
    static_input_paths: Mapping[str, Path],
    expected_static_input_paths: Mapping[str, Path],
    units_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    source_split_authority: Mapping[str, Any],
    generated_domain_authority: Mapping[str, Any],
    seed_qa_semantic_authority: Mapping[str, Any],
    runner_path: Path,
    expected_runner_path: Path,
    train_source_group_ids: frozenset[str],
    train_source_document_ids: frozenset[str],
    replay_source_group_ids: frozenset[str],
) -> tuple[dict[str, Any], bytes]:
    static = build_static_input_bindings(
        static_input_paths, expected_paths=expected_static_input_paths
    )
    source_split = _validate_source_split(source_split_authority)
    generated = _validate_generated(generated_domain_authority)
    seed = _validate_seed(seed_qa_semantic_authority)
    _crosscheck_authority_files(static, source_split, generated, seed)
    _crosscheck_live_authority_metadata(
        static_input_paths,
        expected_static_input_paths,
        source_split,
        generated,
        seed,
    )
    candidates, _ = build_candidate_sets(units_by_variant)
    membership_commitments = _build_membership_commitments(
        source_split_authority=source_split,
        train_source_group_ids=train_source_group_ids,
        train_source_document_ids=train_source_document_ids,
        replay_source_group_ids=replay_source_group_ids,
    )
    request = _seal(
        {
            "schema": REQUEST_SCHEMA,
            "status": REQUEST_STATUS,
            "identity_scheme": IDENTITY_SCHEME,
            "claim_runner": build_runner_binding(
                runner_path, expected_runner_path=expected_runner_path
            ),
            "source_split_authority": source_split,
            "static_inputs": static,
            "generated_domain_authority": generated,
            "seed_qa_semantic_authority": seed,
            "candidate_sets": candidates,
            "membership_commitments": membership_commitments,
            "required_claims": copy.deepcopy(ZERO_CLAIMS),
            "output_boundary": copy.deepcopy(OUTPUT_BOUNDARY),
        }
    )
    validate_claim_request(request)
    return request, artifact_bytes(request)


def validate_claim_request(value: Any) -> dict[str, Any]:
    request = _require_exact_keys(value, REQUEST_KEYS, "claim request")
    _validate_self_address(request, "claim request")
    if (
        request.get("schema") != REQUEST_SCHEMA
        or request.get("status") != REQUEST_STATUS
        or request.get("identity_scheme") != IDENTITY_SCHEME
    ):
        raise ContractError("claim request identity or status changed")
    source_split = _validate_source_split(request.get("source_split_authority"))
    normalized = {
        "schema": REQUEST_SCHEMA,
        "status": REQUEST_STATUS,
        "identity_scheme": IDENTITY_SCHEME,
        "claim_runner": _validate_runner(request.get("claim_runner")),
        "source_split_authority": source_split,
        "static_inputs": _validate_static_inputs(request.get("static_inputs")),
        "generated_domain_authority": _validate_generated(
            request.get("generated_domain_authority")
        ),
        "seed_qa_semantic_authority": _validate_seed(
            request.get("seed_qa_semantic_authority")
        ),
        "candidate_sets": _validate_candidate_sets(request.get("candidate_sets")),
        "membership_commitments": _validate_membership_commitments(
            request.get("membership_commitments"),
            source_split_authority=source_split,
        ),
        "required_claims": _validate_claim_counts(
            request.get("required_claims"), "required claims"
        ),
        "output_boundary": _validate_boundary(
            request.get("output_boundary"), "request output boundary"
        ),
        "content_sha256_before_self_field": request[
            "content_sha256_before_self_field"
        ],
    }
    _crosscheck_authority_files(
        normalized["static_inputs"],
        normalized["source_split_authority"],
        normalized["generated_domain_authority"],
        normalized["seed_qa_semantic_authority"],
    )
    if normalized != request:
        raise ContractError("claim request normalization changed bytes")
    _assert_content_free_output(request, "claim request")
    return copy.deepcopy(request)


def load_claim_request(
    path: Path, *, expected_path: Path, expected_file_sha256: str
) -> dict[str, Any]:
    value = _load_canonical_artifact(
        path,
        expected_path=expected_path,
        expected_file_sha256=expected_file_sha256,
        label="claim request",
    )
    return validate_claim_request(value)


def _expected_live_request(
    *,
    static_input_paths: Mapping[str, Path],
    expected_static_input_paths: Mapping[str, Path],
    units_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    source_split_authority: Mapping[str, Any],
    generated_domain_authority: Mapping[str, Any],
    seed_qa_semantic_authority: Mapping[str, Any],
    runner_path: Path,
    expected_runner_path: Path,
    train_source_group_ids: frozenset[str],
    train_source_document_ids: frozenset[str],
    replay_source_group_ids: frozenset[str],
) -> dict[str, Any]:
    return build_claim_request(
        static_input_paths=static_input_paths,
        expected_static_input_paths=expected_static_input_paths,
        units_by_variant=units_by_variant,
        source_split_authority=source_split_authority,
        generated_domain_authority=generated_domain_authority,
        seed_qa_semantic_authority=seed_qa_semantic_authority,
        runner_path=runner_path,
        expected_runner_path=expected_runner_path,
        train_source_group_ids=train_source_group_ids,
        train_source_document_ids=train_source_document_ids,
        replay_source_group_ids=replay_source_group_ids,
    )[0]


def _membership_claims(
    normalized_by_variant: Mapping[
        str, Sequence[tuple[Mapping[str, Any], str]]
    ],
    *,
    train_source_group_ids: frozenset[str],
    train_source_document_ids: frozenset[str],
    replay_source_group_ids: frozenset[str],
    membership_commitments: Mapping[str, str],
    source_split_authority: Mapping[str, Any],
) -> dict[str, int]:
    observed_commitments = _build_membership_commitments(
        source_split_authority=source_split_authority,
        train_source_group_ids=train_source_group_ids,
        train_source_document_ids=train_source_document_ids,
        replay_source_group_ids=replay_source_group_ids,
    )
    expected_commitments = _validate_membership_commitments(
        membership_commitments, source_split_authority=source_split_authority
    )
    if observed_commitments != expected_commitments:
        raise ContractError("claim membership sets differ from preregistration")
    unique: dict[str, Mapping[str, Any]] = {}
    unit_id_to_digest: dict[str, str] = {}
    for variant in VARIANTS:
        for unit, digest in normalized_by_variant[variant]:
            prior = unit_id_to_digest.setdefault(unit["unit_id"], digest)
            if prior != digest:
                raise ContractError("candidate identity mismatch across variants")
            unique.setdefault(digest, unit)
    group_misses = 0
    document_misses = 0
    replay_misses = 0
    for unit in unique.values():
        if unit["component"] == "replay":
            replay_misses += unit["source_group_id"] not in replay_source_group_ids
            continue
        group_misses += unit["source_group_id"] not in train_source_group_ids
        document_misses += (
            unit["source_document_id"] not in train_source_document_ids
        )
    claims = {
        "domain_source_group_not_in_train_count": int(group_misses),
        "domain_source_document_not_in_train_count": int(document_misses),
        # The sealed split authority asserts mutually exclusive partitions.  A
        # candidate proven to be in its explicit train partition cannot also be
        # in the redacted final partition.
        "cross_split_source_group_collision_count": int(group_misses),
        "candidate_identity_mismatch_count": 0,
        "replay_authority_mismatch_count": int(replay_misses),
    }
    if claims != ZERO_CLAIMS:
        raise ContractError(
            "source-disjoint claim failed with aggregate counts "
            + json.dumps(claims, sort_keys=True)
        )
    return claims


def _opaque_payload(authorization: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": OPAQUE_CONTRACT_SCHEMA,
        "identity_scheme": authorization["identity_scheme"],
        "claim_request": authorization["claim_request"],
        "claim_runner": authorization["claim_runner"],
        "source_split_authority": authorization["source_split_authority"],
        "static_input_set_commitment_sha256": authorization[
            "static_input_set_commitment_sha256"
        ],
        "candidate_sets": authorization["candidate_sets"],
        "membership_commitments": authorization["membership_commitments"],
        "claims": authorization["claims"],
        "boundary": authorization["boundary"],
    }


def build_claim_authorization(
    *,
    request_path: Path,
    expected_request_path: Path,
    expected_request_sha256: str,
    static_input_paths: Mapping[str, Path],
    expected_static_input_paths: Mapping[str, Path],
    units_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    source_split_authority: Mapping[str, Any],
    generated_domain_authority: Mapping[str, Any],
    seed_qa_semantic_authority: Mapping[str, Any],
    runner_path: Path,
    expected_runner_path: Path,
    train_source_group_ids: frozenset[str],
    train_source_document_ids: frozenset[str],
    replay_source_group_ids: frozenset[str],
) -> tuple[dict[str, Any], bytes]:
    request = load_claim_request(
        request_path,
        expected_path=expected_request_path,
        expected_file_sha256=expected_request_sha256,
    )
    live_request = _expected_live_request(
        static_input_paths=static_input_paths,
        expected_static_input_paths=expected_static_input_paths,
        units_by_variant=units_by_variant,
        source_split_authority=source_split_authority,
        generated_domain_authority=generated_domain_authority,
        seed_qa_semantic_authority=seed_qa_semantic_authority,
        runner_path=runner_path,
        expected_runner_path=expected_runner_path,
        train_source_group_ids=train_source_group_ids,
        train_source_document_ids=train_source_document_ids,
        replay_source_group_ids=replay_source_group_ids,
    )
    if request != live_request:
        raise ContractError("preregistered request differs from live candidate inputs")
    _, normalized = build_candidate_sets(units_by_variant)
    claims = _membership_claims(
        normalized,
        train_source_group_ids=train_source_group_ids,
        train_source_document_ids=train_source_document_ids,
        replay_source_group_ids=replay_source_group_ids,
        membership_commitments=request["membership_commitments"],
        source_split_authority=request["source_split_authority"],
    )
    request_receipt = {
        "file_sha256": _require_sha256(
            expected_request_sha256, "claim request external pin"
        ),
        "content_sha256": request["content_sha256_before_self_field"],
    }
    authorization_unsigned = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": AUTHORIZATION_STATUS,
        "identity_scheme": IDENTITY_SCHEME,
        "claim_request": request_receipt,
        "claim_runner": request["claim_runner"],
        "source_split_authority": request["source_split_authority"],
        "static_input_set_commitment_sha256": request["static_inputs"][
            "bindings_commitment_sha256"
        ],
        "candidate_sets": request["candidate_sets"],
        "membership_commitments": request["membership_commitments"],
        "claims": claims,
        "boundary": copy.deepcopy(OUTPUT_BOUNDARY),
    }
    authorization_unsigned["opaque_receipt_sha256"] = canonical_sha256(
        _opaque_payload(authorization_unsigned)
    )
    authorization = _seal(authorization_unsigned)
    validate_claim_authorization(authorization)
    return authorization, artifact_bytes(authorization)


def validate_claim_authorization(value: Any) -> dict[str, Any]:
    authorization = _require_exact_keys(
        value, AUTHORIZATION_KEYS, "claim authorization"
    )
    _validate_self_address(authorization, "claim authorization")
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("status") != AUTHORIZATION_STATUS
        or authorization.get("identity_scheme") != IDENTITY_SCHEME
    ):
        raise ContractError("claim authorization identity or status changed")
    request = _require_exact_keys(
        authorization.get("claim_request"),
        {"file_sha256", "content_sha256"},
        "claim authorization request receipt",
    )
    for key in request:
        _require_sha256(request[key], f"claim authorization request {key}")
    _validate_runner(authorization.get("claim_runner"))
    source_split = _validate_source_split(
        authorization.get("source_split_authority")
    )
    _require_sha256(
        authorization.get("static_input_set_commitment_sha256"),
        "claim authorization static input commitment",
    )
    _validate_candidate_sets(authorization.get("candidate_sets"))
    _validate_membership_commitments(
        authorization.get("membership_commitments"),
        source_split_authority=source_split,
    )
    _validate_claim_counts(authorization.get("claims"), "authorization claims")
    _validate_boundary(authorization.get("boundary"), "authorization boundary")
    opaque = _require_sha256(
        authorization.get("opaque_receipt_sha256"), "opaque receipt"
    )
    if opaque != canonical_sha256(_opaque_payload(authorization)):
        raise ContractError("opaque receipt is not the canonical claim hash")
    _assert_content_free_output(authorization, "claim authorization")
    return copy.deepcopy(authorization)


def load_claim_authorization(
    path: Path,
    *,
    expected_path: Path,
    expected_file_sha256: str,
) -> dict[str, Any]:
    value = _load_canonical_artifact(
        path,
        expected_path=expected_path,
        expected_file_sha256=expected_file_sha256,
        label="claim authorization",
    )
    return validate_claim_authorization(value)


def build_extension_from_values(
    *,
    request: Mapping[str, Any],
    expected_request_sha256: str,
    authorization: Mapping[str, Any],
    expected_authorization_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    """Seal an extension from already loaded, independently pinned parents."""

    request = validate_claim_request(request)
    authorization = validate_claim_authorization(authorization)
    request_digest = _require_sha256(
        expected_request_sha256, "claim request external pin"
    )
    authorization_digest = _require_sha256(
        expected_authorization_sha256, "authorization external pin"
    )
    if authorization["claim_request"] != {
        "file_sha256": request_digest,
        "content_sha256": request["content_sha256_before_self_field"],
    }:
        raise ContractError("authorization binds a different preregistered request")
    for key in (
        "claim_runner",
        "source_split_authority",
        "candidate_sets",
        "membership_commitments",
    ):
        if authorization[key] != request[key]:
            raise ContractError(f"authorization {key} differs from request")
    if (
        authorization["static_input_set_commitment_sha256"]
        != request["static_inputs"]["bindings_commitment_sha256"]
    ):
        raise ContractError("authorization static input commitment differs")
    authorization_receipt = {
        "file_sha256": authorization_digest,
        "content_sha256": authorization[
            "content_sha256_before_self_field"
        ],
    }
    extension = _seal(
        {
            "schema": EXTENSION_SCHEMA,
            "status": EXTENSION_STATUS,
            "accepted_for_training": True,
            "claim_request": authorization["claim_request"],
            "claim_authorization": authorization_receipt,
            "claim_runner": request["claim_runner"],
            "source_split_authority": request["source_split_authority"],
            "static_inputs": request["static_inputs"],
            "generated_domain_authority": request["generated_domain_authority"],
            "seed_qa_semantic_authority": request[
                "seed_qa_semantic_authority"
            ],
            "candidate_sets": request["candidate_sets"],
            "membership_commitments": request["membership_commitments"],
            "opaque_collision_contract": {
                "schema": OPAQUE_CONTRACT_SCHEMA,
                "status": "passed",
                "claims": authorization["claims"],
                "authorization_file_sha256": authorization_digest,
                "opaque_receipt_sha256": authorization[
                    "opaque_receipt_sha256"
                ],
                "boundary": authorization["boundary"],
            },
        }
    )
    validate_extension(
        extension,
        expected_request_sha256=request_digest,
        expected_authorization_sha256=authorization_digest,
        request=request,
        authorization=authorization,
    )
    return extension, artifact_bytes(extension)


def build_extension(
    *,
    request_path: Path,
    expected_request_path: Path,
    expected_request_sha256: str,
    authorization_path: Path,
    expected_authorization_path: Path,
    expected_authorization_sha256: str,
    static_input_paths: Mapping[str, Path],
    expected_static_input_paths: Mapping[str, Path],
    units_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    source_split_authority: Mapping[str, Any],
    generated_domain_authority: Mapping[str, Any],
    seed_qa_semantic_authority: Mapping[str, Any],
    runner_path: Path,
    expected_runner_path: Path,
    train_source_group_ids: frozenset[str],
    train_source_document_ids: frozenset[str],
    replay_source_group_ids: frozenset[str],
) -> tuple[dict[str, Any], bytes]:
    request = load_claim_request(
        request_path,
        expected_path=expected_request_path,
        expected_file_sha256=expected_request_sha256,
    )
    live_request = _expected_live_request(
        static_input_paths=static_input_paths,
        expected_static_input_paths=expected_static_input_paths,
        units_by_variant=units_by_variant,
        source_split_authority=source_split_authority,
        generated_domain_authority=generated_domain_authority,
        seed_qa_semantic_authority=seed_qa_semantic_authority,
        runner_path=runner_path,
        expected_runner_path=expected_runner_path,
        train_source_group_ids=train_source_group_ids,
        train_source_document_ids=train_source_document_ids,
        replay_source_group_ids=replay_source_group_ids,
    )
    if request != live_request:
        raise ContractError("claim request is stale against live candidate inputs")
    authorization = load_claim_authorization(
        authorization_path,
        expected_path=expected_authorization_path,
        expected_file_sha256=expected_authorization_sha256,
    )
    return build_extension_from_values(
        request=request,
        expected_request_sha256=expected_request_sha256,
        authorization=authorization,
        expected_authorization_sha256=expected_authorization_sha256,
    )


def validate_extension(
    value: Any,
    *,
    expected_request_sha256: str,
    expected_authorization_sha256: str,
    request: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    extension = _require_exact_keys(value, EXTENSION_KEYS, "source-disjoint extension")
    _validate_self_address(extension, "source-disjoint extension")
    request = validate_claim_request(request)
    authorization = validate_claim_authorization(authorization)
    request_digest = _require_sha256(
        expected_request_sha256, "extension expected request digest"
    )
    authorization_digest = _require_sha256(
        expected_authorization_sha256, "extension expected authorization digest"
    )
    if (
        extension.get("schema") != EXTENSION_SCHEMA
        or extension.get("status") != EXTENSION_STATUS
        or extension.get("accepted_for_training") is not True
    ):
        raise ContractError("source-disjoint extension status changed")
    expected_request_receipt = {
        "file_sha256": request_digest,
        "content_sha256": request["content_sha256_before_self_field"],
    }
    expected_authorization_receipt = {
        "file_sha256": authorization_digest,
        "content_sha256": authorization[
            "content_sha256_before_self_field"
        ],
    }
    if extension.get("claim_request") != expected_request_receipt:
        raise ContractError("extension request receipt differs")
    if extension.get("claim_authorization") != expected_authorization_receipt:
        raise ContractError("extension authorization receipt differs")
    for key in (
        "claim_runner",
        "source_split_authority",
        "static_inputs",
        "generated_domain_authority",
        "seed_qa_semantic_authority",
        "candidate_sets",
        "membership_commitments",
    ):
        if extension.get(key) != request.get(key):
            raise ContractError(f"extension {key} differs from request")
    opaque = _require_exact_keys(
        extension.get("opaque_collision_contract"),
        {
            "schema",
            "status",
            "claims",
            "authorization_file_sha256",
            "opaque_receipt_sha256",
            "boundary",
        },
        "opaque collision contract",
    )
    if (
        opaque.get("schema") != OPAQUE_CONTRACT_SCHEMA
        or opaque.get("status") != "passed"
        or opaque.get("authorization_file_sha256") != authorization_digest
        or opaque.get("opaque_receipt_sha256")
        != authorization["opaque_receipt_sha256"]
        or opaque.get("claims") != authorization["claims"]
        or opaque.get("boundary") != authorization["boundary"]
    ):
        raise ContractError("opaque collision contract differs from authorization")
    _validate_claim_counts(opaque["claims"], "extension claims")
    _validate_boundary(opaque["boundary"], "extension boundary")
    _assert_content_free_output(extension, "source-disjoint extension")
    return copy.deepcopy(extension)


def load_extension(
    path: Path,
    *,
    expected_path: Path,
    expected_file_sha256: str,
    expected_request_sha256: str,
    expected_authorization_sha256: str,
    request: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    value = _load_canonical_artifact(
        path,
        expected_path=expected_path,
        expected_file_sha256=expected_file_sha256,
        label="source-disjoint extension",
    )
    return validate_extension(
        value,
        expected_request_sha256=expected_request_sha256,
        expected_authorization_sha256=expected_authorization_sha256,
        request=request,
        authorization=authorization,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Validate fixed-path content-free source-disjoint artifacts. "
            "Artifact creation is performed only by the live mixed-snapshot "
            "integration using the pure functions in this module."
        )
    )
    subparsers = result.add_subparsers(dest="command", required=True)
    request = subparsers.add_parser("validate-request")
    request.add_argument("--expected-request-sha256", required=True)
    authorization = subparsers.add_parser("validate-authorization")
    authorization.add_argument("--expected-request-sha256", required=True)
    authorization.add_argument("--expected-authorization-sha256", required=True)
    extension = subparsers.add_parser("validate-extension")
    extension.add_argument("--expected-request-sha256", required=True)
    extension.add_argument("--expected-authorization-sha256", required=True)
    extension.add_argument("--expected-extension-sha256", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    request = load_claim_request(
        CLAIM_REQUEST_PATH,
        expected_path=CLAIM_REQUEST_PATH,
        expected_file_sha256=arguments.expected_request_sha256,
    )
    if arguments.command == "validate-request":
        print(
            json.dumps(
                {
                    "status": "validated",
                    "schema": request["schema"],
                    "request_file_sha256": arguments.expected_request_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    authorization = load_claim_authorization(
        CLAIM_AUTHORIZATION_PATH,
        expected_path=CLAIM_AUTHORIZATION_PATH,
        expected_file_sha256=arguments.expected_authorization_sha256,
    )
    if authorization["claim_request"] != {
        "file_sha256": arguments.expected_request_sha256,
        "content_sha256": request["content_sha256_before_self_field"],
    }:
        raise ContractError("authorization does not bind the expected request")
    if arguments.command == "validate-authorization":
        print(
            json.dumps(
                {
                    "status": "validated",
                    "schema": authorization["schema"],
                    "authorization_file_sha256": (
                        arguments.expected_authorization_sha256
                    ),
                    "opaque_receipt_sha256": authorization[
                        "opaque_receipt_sha256"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    extension = load_extension(
        EXTENSION_PATH,
        expected_path=EXTENSION_PATH,
        expected_file_sha256=arguments.expected_extension_sha256,
        expected_request_sha256=arguments.expected_request_sha256,
        expected_authorization_sha256=arguments.expected_authorization_sha256,
        request=request,
        authorization=authorization,
    )
    print(
        json.dumps(
            {
                "status": "validated",
                "schema": extension["schema"],
                "extension_file_sha256": arguments.expected_extension_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
