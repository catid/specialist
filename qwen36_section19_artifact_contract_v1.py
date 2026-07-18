#!/usr/bin/env python3
"""Fail-closed plan.md section 19 dataset and run-artifact authority.

The mixed-snapshot builder intentionally keeps its native row schema small.
This module derives the complete section 19 metadata surface from the already
validated packed sequences without copying source text.  It also seals the
required run files/directories with content-addressed, phase-aware receipts.

Protected evaluation data is outside this contract.  Only the selected train
snapshot authority and artifacts inside one run directory may be opened.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Mapping


DATASET_MANIFEST_SCHEMA = "qwen36-section19-dataset-manifest-v1"
DATASET_HASHES_SCHEMA = "qwen36-section19-dataset-hashes-v1"
RUN_ARTIFACT_CONTRACT_SCHEMA = "qwen36-section19-run-artifact-contract-v1"
ARTIFACT_RECEIPT_SCHEMA = "qwen36-section19-artifact-receipt-v1"
DERIVATION_SCHEMA = "qwen36-section19-segment-derivation-v1"

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
ZERO_SHA256 = hashlib.sha256(b"").hexdigest()

SECTION19_FIELDS = (
    "source_identifier",
    "split",
    "example_type",
    "domain_category",
    "token_count",
    "closed_book",
    "contains_source_context",
    "replay",
    "hard_negative",
    "deterministic_verifier",
    "generator",
    "verification_status",
    "data_lineage",
)

DETERMINISTIC_VERIFIER_TYPES = frozenset({
    "exact_text_v1",
    "normalized_exact_text_v1",
    "json_exact_v1",
    "tool_call_exact_v1",
    "python_function_cases_v1",
    "pinned_seed_hash_v1",
    "sealed_text_and_token_hash_v1",
})
NONDETERMINISTIC_VERIFIER_TYPES = frozenset({
    "approval_required_v1",
    "sealed_seed_qa_semantic_authority_v1",
})
GENERATED_VERIFIER_KEYS = frozenset({
    "structural",
    "nli",
    "semantic_judge_pass_1",
    "semantic_judge_pass_2",
    "selection",
})

FORBIDDEN_PATH_TOKENS = frozenset({
    "benchmark",
    "benchmarks",
    "eval",
    "evaluation",
    "evaluations",
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
})


class Section19ContractError(RuntimeError):
    """A section 19 derivation or run-artifact gate failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Section19ContractError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def self_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    _require(
        "content_sha256_before_self_field" not in result,
        "self-address field was supplied before sealing",
    )
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def validate_self_address(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label}: object required")
    claimed = value.get("content_sha256_before_self_field")
    unsigned = copy.deepcopy(value)
    unsigned.pop("content_sha256_before_self_field", None)
    _require(
        isinstance(claimed, str)
        and HEX64.fullmatch(claimed) is not None
        and canonical_sha256(unsigned) == claimed,
        f"{label}: stale or forged self address",
    )
    return value


def _path_tokens(path: Path) -> set[str]:
    result: set[str] = set()
    for component in path.parts:
        collapsed = re.sub(r"[^a-z0-9]", "", component.casefold())
        if collapsed:
            result.add(collapsed)
        result.update(
            token
            for token in re.split(r"[^a-z0-9]+", component.casefold())
            if token
        )
    return result


def _safe_relative_path(
    repository_root: Path,
    value: Any,
    *,
    label: str,
    allow_forbidden_tokens: bool = False,
) -> Path:
    _require(isinstance(value, str) and value, f"{label}: path is missing")
    declared = Path(value)
    _require(not declared.is_absolute(), f"{label}: absolute path is forbidden")
    _require(
        declared.as_posix() == value
        and value not in {".", ".."}
        and all(part not in {"", ".", ".."} for part in declared.parts),
        f"{label}: ambiguous path spelling",
    )
    if not allow_forbidden_tokens:
        _require(
            not (_path_tokens(declared) & FORBIDDEN_PATH_TOKENS),
            f"{label}: forbidden source path class",
        )
    root = Path(os.path.abspath(os.fspath(repository_root)))
    candidate = Path(os.path.abspath(os.fspath(root / declared)))
    _require(candidate.is_relative_to(root), f"{label}: path escapes repository")
    return candidate


def _secure_regular_file(path: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    current = Path(lexical.anchor)
    metadata = None
    for component in lexical.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise Section19ContractError(f"{label}: path is missing") from exc
        _require(
            not stat.S_ISLNK(metadata.st_mode),
            f"{label}: symlink alias is forbidden",
        )
    _require(
        metadata is not None and stat.S_ISREG(metadata.st_mode),
        f"{label}: regular file required",
    )
    _require(metadata.st_nlink == 1, f"{label}: hard-link alias is forbidden")
    return lexical


def _secure_directory(path: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = lexical.lstat()
    except OSError as exc:
        raise Section19ContractError(f"{label}: directory is missing") from exc
    _require(
        stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{label}: real directory required",
    )
    return lexical


def _secure_run_directory(
    run_directory: Path,
    *,
    repository_root: Path,
) -> Path:
    root = _secure_directory(repository_root, label="repository root")
    run = _secure_directory(run_directory, label="run directory")
    _require(run.is_relative_to(root), "run directory escapes repository")
    relative = run.relative_to(root)
    _require(
        not (_path_tokens(relative) & FORBIDDEN_PATH_TOKENS),
        "run directory uses a protected/evaluation path class",
    )
    return run


def builder_receipt(
    repository_root: Path,
    path: Path,
    *,
    role: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    candidate = _secure_regular_file(Path(path), label=f"builder {role}")
    _require(candidate.is_relative_to(root), f"builder {role}: outside repository")
    relative = candidate.relative_to(root).as_posix()
    _require(
        not (_path_tokens(Path(relative)) & FORBIDDEN_PATH_TOKENS),
        f"builder {role}: forbidden path class",
    )
    _require(isinstance(role, str) and role, "builder role is invalid")
    return {
        "role": role,
        "path": relative,
        "sha256": file_sha256(candidate),
    }


def builder_receipts(
    authority: Any,
    *,
    repository_root: Path,
    additional: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    """Bind the snapshot assembler plus every run-side metadata builder."""

    top = authority.top_manifest
    assembler = top.get("assembler")
    _require(isinstance(assembler, dict), "snapshot assembler receipt is absent")
    _require(
        set(assembler) == {"path", "sha256"},
        "snapshot assembler receipt fields changed",
    )
    assembler_path = _safe_relative_path(
        repository_root,
        assembler.get("path"),
        label="snapshot assembler",
    )
    observed = builder_receipt(
        repository_root,
        assembler_path,
        role="snapshot_assembler",
    )
    _require(
        observed["sha256"] == assembler.get("sha256"),
        "snapshot assembler receipt is stale",
    )
    result = {"snapshot_assembler": observed}
    for role, path in sorted(additional.items()):
        _require(role not in result, f"duplicate builder role: {role}")
        result[role] = builder_receipt(
            repository_root,
            Path(path),
            role=role,
        )
    paths = [item["path"] for item in result.values()]
    _require(len(paths) == len(set(paths)), "one builder has ambiguous duplicate roles")
    return result


def _verification_status(verifier: Any, *, label: str) -> str:
    _require(isinstance(verifier, dict) and verifier, f"{label}: verifier is absent")
    if "status" in verifier:
        status = verifier.get("status")
        _require(
            status in {"passed", "sealed_passed"},
            f"{label}: verification did not pass",
        )
        return status
    _require(
        set(verifier) == GENERATED_VERIFIER_KEYS,
        f"{label}: ambiguous verifier map",
    )
    _require(
        all(
            isinstance(receipt, dict) and receipt.get("status") == "passed"
            for receipt in verifier.values()
        ),
        f"{label}: generated verifier coverage did not pass",
    )
    return "passed"


def _deterministic_verifier(verifier: dict[str, Any], *, label: str) -> bool:
    if "type" in verifier:
        verifier_type = verifier.get("type")
        _require(
            isinstance(verifier_type, str) and verifier_type,
            f"{label}: verifier type is invalid",
        )
        if verifier_type in DETERMINISTIC_VERIFIER_TYPES:
            return True
        if verifier_type in NONDETERMINISTIC_VERIFIER_TYPES:
            return False
        raise Section19ContractError(
            f"{label}: verifier determinism is ambiguous: {verifier_type!r}"
        )
    _require(
        set(verifier) == GENERATED_VERIFIER_KEYS,
        f"{label}: verifier determinism is ambiguous",
    )
    # The generated-domain authority uses NLI plus two independent semantic
    # judges.  Structural hashing is deterministic, but answer acceptance is
    # not an objective deterministic verifier.
    return False


def _example_semantics(
    *,
    stream: str,
    training_format: str,
    metadata: dict[str, Any],
    label: str,
) -> tuple[str, bool, bool]:
    if stream == "raw_markdown":
        _require(
            training_format == "causal_next_token"
            and metadata.get("category") == "raw_domain_continuation"
            and metadata.get("replay") is False,
            f"{label}: ambiguous raw-Markdown semantics",
        )
        return "raw_markdown_cpt", False, True
    if stream == "replay":
        _require(
            training_format == "chat_assistant_only"
            and metadata.get("replay") is True,
            f"{label}: ambiguous replay semantics",
        )
        return "general_behavior_replay", True, False
    _require(
        stream == "domain_qa"
        and training_format == "chat_assistant_only"
        and metadata.get("replay") is False,
        f"{label}: ambiguous domain-QA semantics",
    )
    if "task_family" in metadata or "task_subtype" in metadata:
        family = metadata.get("task_family")
        subtype = metadata.get("task_subtype")
        _require(
            family in {"closed_book_application", "grounded_synthesis"}
            and isinstance(subtype, str)
            and subtype,
            f"{label}: generated example type is ambiguous",
        )
        closed_book = family == "closed_book_application"
        return f"generated_domain_qa:{subtype}", closed_book, not closed_book
    _require(
        metadata.get("category") == "precurated_seed_qa_opaque"
        and isinstance(metadata.get("fact_id"), str)
        and metadata["fact_id"],
        f"{label}: seed-QA example type is ambiguous",
    )
    return "precurated_seed_qa", True, False


def _segment_entry(
    sequence: dict[str, Any],
    segment: dict[str, Any],
    *,
    segment_index: int,
) -> dict[str, Any]:
    sequence_id = sequence.get("sequence_id")
    label = f"{sequence_id!r} segment {segment_index}"
    _require(isinstance(sequence_id, str) and sequence_id, f"{label}: sequence ID missing")
    metadata = segment.get("metadata")
    _require(isinstance(metadata, dict), f"{label}: metadata absent")
    source_identifier = segment.get("unit_id")
    _require(
        isinstance(source_identifier, str) and source_identifier,
        f"{label}: source identifier absent",
    )
    source_group = sequence.get("source_group_id")
    source_document = sequence.get("source_document_id")
    _require(
        isinstance(source_group, str)
        and source_group
        and isinstance(source_document, str)
        and source_document,
        f"{label}: source group/document identity absent",
    )
    start = segment.get("token_start")
    stop = segment.get("token_stop")
    source_start = segment.get("source_token_start")
    source_stop = segment.get("source_token_stop")
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) for value in (
            start, stop, source_start, source_stop,
        ))
        and 0 <= start < stop
        and 0 <= source_start < source_stop
        and stop - start == source_stop - source_start,
        f"{label}: token span is invalid",
    )
    budget = segment.get("budget_token_count")
    _require(
        isinstance(budget, int) and not isinstance(budget, bool) and budget > 0,
        f"{label}: supervised token count is invalid",
    )
    category = metadata.get("category")
    replay = metadata.get("replay")
    hard_negative = metadata.get("hard_negative")
    generator = metadata.get("generator")
    lineage = metadata.get("lineage")
    source_record = metadata.get("source_record_sha256")
    _require(isinstance(category, str) and category, f"{label}: category absent")
    _require(isinstance(replay, bool), f"{label}: replay flag invalid")
    _require(isinstance(hard_negative, bool), f"{label}: hard-negative flag invalid")
    _require(isinstance(generator, dict) and generator, f"{label}: generator absent")
    _require(isinstance(lineage, dict) and lineage, f"{label}: lineage absent")
    _require(
        isinstance(source_record, str) and HEX64.fullmatch(source_record) is not None,
        f"{label}: source-record identity absent",
    )
    verifier = metadata.get("verifier")
    verification_status = _verification_status(verifier, label=label)
    deterministic = _deterministic_verifier(verifier, label=label)
    example_type, closed_book, contains_context = _example_semantics(
        stream=sequence.get("stream"),
        training_format=sequence.get("training_format"),
        metadata=metadata,
        label=label,
    )
    _require(
        replay is (sequence.get("stream") == "replay"),
        f"{label}: replay derivation is inconsistent",
    )
    if "generation_mode" in metadata:
        _require(
            hard_negative
            is (metadata.get("generation_mode") == "calibrated_hard_negative"),
            f"{label}: generated hard-negative derivation is inconsistent",
        )
    identity_payload = {
        "sequence_id": sequence_id,
        "source_identifier": source_identifier,
        "source_token_start": source_start,
        "source_token_stop": source_stop,
        "metadata_sha256": canonical_sha256(metadata),
    }
    data_lineage = {
        "source_group_identifier": source_group,
        "source_document_identifier": source_document,
        "source_record_sha256": source_record,
        "source_context_identifier": metadata.get("source_context_id"),
        "resource_identifier": metadata.get("resource_id"),
        "artifact_identifier": metadata.get("artifact_id"),
        "request_identifier": metadata.get("request_id"),
        "candidate_example_identifier": metadata.get("candidate_example_id"),
        "lineage": copy.deepcopy(lineage),
        "lineage_sha256": canonical_sha256(lineage),
        "metadata_sha256": canonical_sha256(metadata),
    }
    return {
        "schema": DERIVATION_SCHEMA,
        "segment_id": "section19-segment-v1:" + canonical_sha256(identity_payload),
        "sequence_id": sequence_id,
        "sequence_sha256": canonical_sha256(sequence),
        "segment_index": segment_index,
        "source_token_start": source_start,
        "source_token_stop": source_stop,
        "supervised_token_count": budget,
        "source_identifier": source_identifier,
        "split": "train",
        "example_type": example_type,
        "domain_category": category,
        "token_count": stop - start,
        "closed_book": closed_book,
        "contains_source_context": contains_context,
        "replay": replay,
        "hard_negative": hard_negative,
        "deterministic_verifier": deterministic,
        "generator": copy.deepcopy(generator),
        "generator_identity_sha256": canonical_sha256(generator),
        "verification_status": verification_status,
        "verification_receipt_sha256": canonical_sha256(verifier),
        "data_lineage": data_lineage,
    }


def _validate_builder_set(
    builders: Any,
    *,
    repository_root: Path,
) -> dict[str, dict[str, Any]]:
    _require(isinstance(builders, dict) and builders, "builder set is absent")
    paths: list[str] = []
    for role, receipt in sorted(builders.items()):
        _require(isinstance(role, str) and role, "builder role is invalid")
        _require(
            isinstance(receipt, dict)
            and set(receipt) == {"role", "path", "sha256"}
            and receipt.get("role") == role,
            f"builder {role}: receipt fields changed",
        )
        path = _safe_relative_path(
            repository_root,
            receipt.get("path"),
            label=f"builder {role}",
        )
        _secure_regular_file(path, label=f"builder {role}")
        _require(
            isinstance(receipt.get("sha256"), str)
            and HEX64.fullmatch(receipt["sha256"]) is not None
            and file_sha256(path) == receipt["sha256"],
            f"builder {role}: stale receipt",
        )
        paths.append(receipt["path"])
    _require(len(paths) == len(set(paths)), "duplicate/ambiguous builder identity")
    return builders


def _dataset_identity(
    authority: Any,
    builders: dict[str, dict[str, Any]],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    top = authority.top_manifest
    variant = authority.variant_manifest
    reference = top.get("variants", {}).get(authority.variant)
    _require(isinstance(reference, dict), "variant file receipt is absent")
    top_path = _secure_regular_file(
        authority.top_manifest_path, label="top dataset manifest"
    )
    root = Path(repository_root).resolve()
    _require(top_path.is_relative_to(root), "top dataset manifest escapes repository")
    top_relative = top_path.relative_to(root).as_posix()
    _require(
        not (_path_tokens(Path(top_relative)) & FORBIDDEN_PATH_TOKENS),
        "top dataset manifest uses a forbidden path class",
    )
    variant_path = _safe_relative_path(
        repository_root,
        reference.get("manifest_path"),
        label="variant dataset manifest",
    )
    _secure_regular_file(variant_path, label="variant dataset manifest")
    top_content = top.get("content_sha256_before_self_field")
    variant_content = variant.get("content_sha256_before_self_field")
    _require(
        all(
            isinstance(value, str) and HEX64.fullmatch(value) is not None
            for value in (
                top_content,
                variant_content,
                reference.get("manifest_file_sha256"),
                variant.get("sequences", {}).get("sha256"),
                variant.get("schedule", {}).get("sha256"),
                authority.sequence_set_identity_sha256,
                authority.final_cursor_commitment_sha256,
            )
        ),
        "dataset authority identity is incomplete",
    )
    _require(
        file_sha256(variant_path) == reference["manifest_file_sha256"],
        "variant dataset manifest file receipt is stale",
    )
    return {
        "top_manifest_path": top_relative,
        "top_manifest_file_sha256": file_sha256(top_path),
        "top_manifest_content_sha256": top_content,
        "variant": authority.variant,
        "variant_manifest_path": reference["manifest_path"],
        "variant_manifest_file_sha256": reference["manifest_file_sha256"],
        "variant_manifest_content_sha256": variant_content,
        "sequence_file_sha256": variant["sequences"]["sha256"],
        "schedule_file_sha256": variant["schedule"]["sha256"],
        "sequence_set_identity_sha256": authority.sequence_set_identity_sha256,
        "final_cursor_commitment_sha256": authority.final_cursor_commitment_sha256,
        "builder_set_identity_sha256": canonical_sha256(builders),
    }


def build_dataset_manifest(
    authority: Any,
    *,
    repository_root: Path,
    builders: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive and seal every section 19 field for every packed segment."""

    validated_builders = _validate_builder_set(
        copy.deepcopy(dict(builders)),
        repository_root=repository_root,
    )
    sequences = authority.sequences
    _require(isinstance(sequences, dict) and sequences, "sequence authority is empty")
    entries: list[dict[str, Any]] = []
    segment_ids: set[str] = set()
    fragments_by_source: dict[str, list[dict[str, Any]]] = {}
    accounting: dict[str, dict[str, int]] = {}
    for sequence_id, sequence in sorted(sequences.items()):
        _require(
            isinstance(sequence, dict)
            and sequence.get("sequence_id") == sequence_id,
            "sequence authority key is ambiguous",
        )
        segments = sequence.get("segments")
        _require(isinstance(segments, list) and segments, f"{sequence_id}: segments absent")
        for index, segment in enumerate(segments):
            entry = _segment_entry(sequence, segment, segment_index=index)
            segment_id = entry["segment_id"]
            _require(segment_id not in segment_ids, "duplicate emitted segment identity")
            segment_ids.add(segment_id)
            entries.append(entry)
            fragments_by_source.setdefault(entry["source_identifier"], []).append(entry)
            stream = sequence["stream"]
            bucket = accounting.setdefault(
                stream,
                {"segments": 0, "input_tokens": 0, "supervised_tokens": 0},
            )
            bucket["segments"] += 1
            bucket["input_tokens"] += entry["token_count"]
            bucket["supervised_tokens"] += entry["supervised_token_count"]

    # One source identity may be fragmented, but it cannot name two documents,
    # carry two lineage objects, or repeat/overlap a source-token interval.
    for source_identifier, fragments in sorted(fragments_by_source.items()):
        immutable = {
            (
                item["data_lineage"]["source_group_identifier"],
                item["data_lineage"]["source_document_identifier"],
                item["data_lineage"]["source_record_sha256"],
                item["data_lineage"]["metadata_sha256"],
                item["split"],
                item["example_type"],
                item["domain_category"],
                item["closed_book"],
                item["contains_source_context"],
                item["replay"],
                item["hard_negative"],
                item["deterministic_verifier"],
                item["verification_status"],
            )
            for item in fragments
        }
        _require(
            len(immutable) == 1,
            f"source {source_identifier!r}: duplicate identity is ambiguous",
        )
        spans = sorted(
            (item["source_token_start"], item["source_token_stop"])
            for item in fragments
        )
        for previous, current in zip(spans, spans[1:]):
            _require(
                previous[1] <= current[0],
                f"source {source_identifier!r}: duplicate/overlapping fragment",
            )

    dataset_identity = _dataset_identity(
        authority,
        validated_builders,
        repository_root=repository_root,
    )
    body = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "status": "complete_launchable",
        "split": "train",
        "protected_evaluation_content_opened": False,
        "sensitive_raw_source_contents_logged": False,
        "field_contract": {
            "schema": DERIVATION_SCHEMA,
            "required_fields": list(SECTION19_FIELDS),
            "derivation_is_recomputed_not_trusted": True,
            "source_context_means_domain_source_material_present_in_training_example": True,
            "token_count_semantics": "input_tokens_in_emitted_segment",
            "supervised_token_count_semantics": "builder_budget_tokens_in_emitted_segment",
        },
        "dataset_identity": dataset_identity,
        "builders": validated_builders,
        "segments": entries,
        "segment_count": len(entries),
        "source_count": len(fragments_by_source),
        "accounting_by_stream": dict(sorted(accounting.items())),
        "segment_set_identity_sha256": canonical_sha256(entries),
    }
    return self_address(body)


def validate_dataset_manifest(
    value: Any,
    authority: Any,
    *,
    repository_root: Path,
    builders: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    validate_self_address(value, label="dataset manifest")
    _require(
        value.get("schema") == DATASET_MANIFEST_SCHEMA,
        "dataset manifest schema changed",
    )
    expected = build_dataset_manifest(
        authority,
        repository_root=repository_root,
        builders=builders,
    )
    _require(value == expected, "dataset manifest derivation is stale or forged")
    return value


def build_dataset_hashes(
    authority: Any,
    *,
    dataset_manifest_path: Path,
    dataset_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    validate_self_address(dataset_manifest, label="dataset manifest")
    path = _secure_regular_file(dataset_manifest_path, label="dataset manifest")
    identity = dataset_manifest.get("dataset_identity")
    _require(isinstance(identity, dict), "dataset identity is absent")
    body = {
        "schema": DATASET_HASHES_SCHEMA,
        "status": "sealed_passed",
        "dataset_manifest": {
            "path": path.name,
            "file_sha256": file_sha256(path),
            "content_sha256": dataset_manifest[
                "content_sha256_before_self_field"
            ],
            "segment_set_identity_sha256": dataset_manifest[
                "segment_set_identity_sha256"
            ],
            "segments": dataset_manifest["segment_count"],
        },
        "dataset_identity": copy.deepcopy(identity),
        "identity_sha256": canonical_sha256(identity),
    }
    _require(
        identity.get("sequence_set_identity_sha256")
        == authority.sequence_set_identity_sha256,
        "dataset hash authority differs from loaded sequence set",
    )
    return self_address(body)


def validate_dataset_files(
    run_directory: Path,
    *,
    repository_root: Path,
    authority: Any,
    builders: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = _secure_run_directory(
        run_directory, repository_root=repository_root
    )
    manifest_path = _secure_regular_file(
        run / "dataset_manifest.json", label="dataset manifest"
    )
    hashes_path = _secure_regular_file(
        run / "dataset_hashes.json", label="dataset hashes"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    validate_self_address(manifest, label="dataset manifest")
    validate_self_address(hashes, label="dataset hashes")
    _require(
        manifest.get("schema") == DATASET_MANIFEST_SCHEMA
        and manifest.get("status") == "complete_launchable",
        "dataset manifest status gate did not pass",
    )
    _require(
        hashes.get("schema") == DATASET_HASHES_SCHEMA
        and hashes.get("status") == "sealed_passed",
        "dataset hash status gate did not pass",
    )
    manifest_receipt = hashes.get("dataset_manifest")
    _require(
        isinstance(manifest_receipt, dict)
        and manifest_receipt.get("path") == "dataset_manifest.json"
        and manifest_receipt.get("file_sha256") == file_sha256(manifest_path)
        and manifest_receipt.get("content_sha256")
        == manifest["content_sha256_before_self_field"]
        and manifest_receipt.get("segment_set_identity_sha256")
        == manifest.get("segment_set_identity_sha256")
        and manifest_receipt.get("segments") == manifest.get("segment_count"),
        "dataset manifest receipt is stale or ambiguous",
    )
    _require(
        hashes.get("dataset_identity") == manifest.get("dataset_identity")
        and hashes.get("identity_sha256")
        == canonical_sha256(manifest["dataset_identity"]),
        "dataset identity binding changed",
    )
    _validate_builder_set(
        manifest.get("builders"), repository_root=repository_root
    )
    _require(
        manifest["dataset_identity"].get("builder_set_identity_sha256")
        == canonical_sha256(manifest["builders"]),
        "dataset builder-set identity changed",
    )
    validate_dataset_manifest(
        manifest,
        authority,
        repository_root=repository_root,
        builders=builders,
    )
    expected_hashes = build_dataset_hashes(
        authority,
        dataset_manifest_path=manifest_path,
        dataset_manifest=manifest,
    )
    _require(hashes == expected_hashes, "dataset hashes are stale or forged")
    return manifest, hashes


@dataclass(frozen=True)
class ArtifactSpec:
    kind: str
    mutability: str
    content_contract: str


REQUIRED_RUN_ARTIFACTS: dict[str, ArtifactSpec] = {
    "run_config.json": ArtifactSpec("file", "immutable", "self-addressed-run-config-v1"),
    "environment.txt": ArtifactSpec("file", "immutable", "utf8-environment-json-v1"),
    "pip_freeze.txt": ArtifactSpec("file", "immutable", "pip-freeze-lines-v1"),
    "git_commits.json": ArtifactSpec("file", "immutable", "git-commit-and-dirty-state-v1"),
    "model_config.json": ArtifactSpec("file", "immutable", "model-config-json-v1"),
    "adapter_config.json": ArtifactSpec("file", "immutable", "adapter-config-json-v1"),
    "trainable_parameters.txt": ArtifactSpec("file", "immutable", "jsonl-trainable-parameter-manifest-v1"),
    "dataset_manifest.json": ArtifactSpec("file", "immutable", DATASET_MANIFEST_SCHEMA),
    "dataset_hashes.json": ArtifactSpec("file", "immutable", DATASET_HASHES_SCHEMA),
    "training_metrics.jsonl": ArtifactSpec("file", "append_only", "qwen36-low-regression-sft-training-metric-v1"),
    "evaluation_results.json": ArtifactSpec("file", "external_stage", "evaluation-results-status-v1"),
    "routing_metrics.json": ArtifactSpec("file", "runtime_final", "qwen36-low-regression-sft-routing-summary-v1"),
    "memory_profile.json": ArtifactSpec("file", "runtime_final", "qwen36-low-regression-sft-memory-profile-v1"),
    "checkpoints": ArtifactSpec("directory", "append_only", "checkpoint-tree-v1"),
    "plots": ArtifactSpec("directory", "external_stage", "plot-tree-v1"),
    "samples": ArtifactSpec("directory", "external_stage", "sample-tree-v1"),
}

RUN_PHASES = frozenset({"launch_ready", "resume_ready", "training_complete", "complete"})


def pending_routing_metrics() -> dict[str, Any]:
    return self_address({
        "schema": "qwen36-low-regression-sft-routing-summary-v1",
        "status": "pending_training",
    })


def pending_memory_profile() -> dict[str, Any]:
    return self_address({
        "schema": "qwen36-low-regression-sft-memory-profile-v1",
        "status": "pending_training",
    })


def _load_json_file(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Section19ContractError(f"{label}: invalid UTF-8 JSON") from exc
    _require(isinstance(value, dict), f"{label}: JSON object required")
    return value


def _validate_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Section19ContractError(f"{label}: invalid UTF-8") from exc
    rows: list[dict[str, Any]] = []
    if not raw:
        return rows
    _require(raw.endswith(b"\n"), f"{label}: final newline is missing")
    for line_number, line in enumerate(text.splitlines(), 1):
        _require(line.strip(), f"{label}:{line_number}: blank row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Section19ContractError(
                f"{label}:{line_number}: invalid JSON"
            ) from exc
        _require(isinstance(row, dict), f"{label}:{line_number}: object required")
        rows.append(row)
    return rows


def _file_content_gate(path: Path, name: str, *, phase: str) -> tuple[str, str, int | None]:
    """Return (content schema, status, optional row count)."""

    if name == "run_config.json":
        value = _load_json_file(path, label=name)
        validate_self_address(value, label=name)
        _require(
            value.get("schema") == "qwen36-low-regression-expert-lora-sft-run-v1",
            "run_config.json: schema changed",
        )
        return value["schema"], "sealed", None
    if name == "environment.txt":
        value = _load_json_file(path, label=name)
        _require(
            all(key in value for key in ("python", "platform", "executable", "gpu")),
            "environment.txt: environment inventory is incomplete",
        )
        return "utf8-environment-json-v1", "sealed", None
    if name == "pip_freeze.txt":
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise Section19ContractError("pip_freeze.txt: invalid UTF-8") from exc
        _require(text.endswith("\n") and bool(text.strip()), "pip_freeze.txt: incomplete freeze")
        _require(all(line.strip() for line in text.splitlines()), "pip_freeze.txt: blank line")
        return "pip-freeze-lines-v1", "sealed", len(text.splitlines())
    if name == "git_commits.json":
        value = _load_json_file(path, label=name)
        _require(
            set(value) == {"specialist", "dirty_paths"}
            and isinstance(value.get("specialist"), str)
            and HEX40.fullmatch(value["specialist"]) is not None
            and isinstance(value.get("dirty_paths"), list)
            and all(isinstance(item, str) for item in value["dirty_paths"]),
            "git_commits.json: repository identity is incomplete",
        )
        return "git-commit-and-dirty-state-v1", "sealed", None
    if name in {"model_config.json", "adapter_config.json"}:
        value = _load_json_file(path, label=name)
        _require(bool(value), f"{name}: empty configuration")
        return REQUIRED_RUN_ARTIFACTS[name].content_contract, "sealed", None
    if name == "trainable_parameters.txt":
        rows = _validate_jsonl(path, label=name)
        _require(bool(rows), f"{name}: empty trainable-parameter manifest")
        names = [row.get("name") for row in rows]
        _require(
            all(isinstance(item, str) and item for item in names)
            and len(names) == len(set(names)),
            f"{name}: missing or duplicate parameter identity",
        )
        return REQUIRED_RUN_ARTIFACTS[name].content_contract, "sealed", len(rows)
    if name in {"dataset_manifest.json", "dataset_hashes.json"}:
        value = _load_json_file(path, label=name)
        validate_self_address(value, label=name)
        expected_schema = REQUIRED_RUN_ARTIFACTS[name].content_contract
        _require(value.get("schema") == expected_schema, f"{name}: schema changed")
        expected_status = "complete_launchable" if name == "dataset_manifest.json" else "sealed_passed"
        _require(value.get("status") == expected_status, f"{name}: status gate failed")
        rows = value.get("segment_count") if name == "dataset_manifest.json" else None
        return expected_schema, value["status"], rows
    if name == "training_metrics.jsonl":
        rows = _validate_jsonl(path, label=name)
        expected = REQUIRED_RUN_ARTIFACTS[name].content_contract
        _require(
            all(row.get("schema") == expected for row in rows),
            f"{name}: metric schema changed",
        )
        steps = [row.get("optimizer_step") for row in rows]
        _require(
            all(isinstance(step, int) and not isinstance(step, bool) for step in steps)
            and steps == list(range(1, len(rows) + 1)),
            f"{name}: duplicate or ambiguous optimizer-step coverage",
        )
        if phase in {"training_complete", "complete"}:
            _require(bool(rows), f"{name}: completed run has no metrics")
        return expected, "complete" if rows else "pending_training", len(rows)
    if name == "evaluation_results.json":
        value = _load_json_file(path, label=name)
        status = value.get("status")
        _require(isinstance(status, str) and status, f"{name}: status is absent")
        complete = status in {"complete", "passed", "sealed_complete"}
        if phase == "complete":
            _require(complete, f"{name}: final evaluation is incomplete")
        return value.get("schema", "evaluation-results-status-v1"), (
            "complete" if complete else "pending_external_evaluation"
        ), None
    if name == "routing_metrics.json":
        value = _load_json_file(path, label=name)
        validate_self_address(value, label=name)
        expected = REQUIRED_RUN_ARTIFACTS[name].content_contract
        _require(value.get("schema") == expected, f"{name}: schema changed")
        status = value.get("status")
        _require(status in {"pending_training", "complete", "disabled"}, f"{name}: status invalid")
        if phase in {"training_complete", "complete"}:
            _require(status in {"complete", "disabled"}, f"{name}: training output incomplete")
        return expected, status, None
    if name == "memory_profile.json":
        value = _load_json_file(path, label=name)
        validate_self_address(value, label=name)
        expected = REQUIRED_RUN_ARTIFACTS[name].content_contract
        _require(value.get("schema") == expected, f"{name}: schema changed")
        status = value.get("status")
        _require(status in {"pending_training", "complete"}, f"{name}: status invalid")
        if phase in {"training_complete", "complete"}:
            _require(status == "complete", f"{name}: training output incomplete")
        return expected, status, None
    raise AssertionError(f"unhandled section 19 artifact: {name}")


def _directory_entries(path: Path, *, label: str) -> list[dict[str, Any]]:
    root = _secure_directory(path, label=label)
    entries: list[dict[str, Any]] = []
    spellings: set[str] = set()
    inodes: set[tuple[int, int]] = set()
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted(directory_names):
            item = current_path / name
            metadata = item.lstat()
            _require(
                stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
                f"{label}: directory symlink or special node is forbidden",
            )
            relative = item.relative_to(root).as_posix()
            folded = relative.casefold()
            _require(folded not in spellings, f"{label}: ambiguous duplicate path")
            spellings.add(folded)
            entries.append({"path": relative, "kind": "directory"})
        for name in sorted(file_names):
            item = current_path / name
            metadata = item.lstat()
            _require(
                stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
                f"{label}: non-regular file is forbidden",
            )
            _require(metadata.st_nlink == 1, f"{label}: hard-link alias is forbidden")
            inode = (metadata.st_dev, metadata.st_ino)
            _require(inode not in inodes, f"{label}: duplicate inode identity")
            inodes.add(inode)
            relative = item.relative_to(root).as_posix()
            folded = relative.casefold()
            _require(folded not in spellings, f"{label}: ambiguous duplicate path")
            spellings.add(folded)
            entries.append({
                "path": relative,
                "kind": "file",
                "bytes": metadata.st_size,
                "sha256": file_sha256(item),
            })
    entries.sort(key=lambda item: (item["path"], item["kind"]))
    return entries


def _artifact_receipt(
    run_directory: Path,
    name: str,
    spec: ArtifactSpec,
    *,
    phase: str,
) -> dict[str, Any]:
    path = run_directory / name
    if spec.kind == "file":
        safe = _secure_regular_file(path, label=name)
        content_schema, status, rows = _file_content_gate(safe, name, phase=phase)
        receipt: dict[str, Any] = {
            "schema": ARTIFACT_RECEIPT_SCHEMA,
            "artifact": name,
            "kind": "file",
            "mutability": spec.mutability,
            "content_contract": spec.content_contract,
            "content_schema": content_schema,
            "status": status,
            "bytes": safe.stat().st_size,
            "sha256": file_sha256(safe),
        }
        if rows is not None:
            receipt["rows"] = rows
        return receipt
    _require(spec.kind == "directory", f"{name}: unknown artifact kind")
    entries = _directory_entries(path, label=name)
    if name == "checkpoints" and phase in {"training_complete", "complete"}:
        _require(any(item["kind"] == "file" for item in entries), "completed run has no checkpoint files")
    status = "populated" if entries else "present_empty"
    return {
        "schema": ARTIFACT_RECEIPT_SCHEMA,
        "artifact": name,
        "kind": "directory",
        "mutability": spec.mutability,
        "content_contract": spec.content_contract,
        "content_schema": spec.content_contract,
        "status": status,
        "entries": entries,
        "entry_count": len(entries),
        "tree_sha256": canonical_sha256(entries),
    }


def _run_config_identity(run_directory: Path) -> tuple[str, str]:
    path = _secure_regular_file(run_directory / "run_config.json", label="run_config.json")
    value = _load_json_file(path, label="run_config.json")
    validate_self_address(value, label="run_config.json")
    run_id = value.get("run_id")
    _require(isinstance(run_id, str) and run_id, "run_config.json: run ID missing")
    return run_id, value["content_sha256_before_self_field"]


def seal_run_artifacts(
    run_directory: Path,
    *,
    repository_root: Path,
    authority: Any,
    builders: Mapping[str, Mapping[str, Any]],
    phase: str,
) -> dict[str, Any]:
    """Seal the exact section 19 artifact inventory for one run phase."""

    _require(phase in RUN_PHASES, "unknown section 19 run phase")
    run = _secure_run_directory(
        run_directory, repository_root=repository_root
    )
    manifest, hashes = validate_dataset_files(
        run,
        repository_root=repository_root,
        authority=authority,
        builders=builders,
    )
    run_id, run_config_content = _run_config_identity(run)
    receipts = {
        name: _artifact_receipt(run, name, spec, phase=phase)
        for name, spec in REQUIRED_RUN_ARTIFACTS.items()
    }
    _require(
        set(receipts) == set(REQUIRED_RUN_ARTIFACTS),
        "required run artifact inventory is incomplete",
    )
    evaluation_complete = receipts["evaluation_results.json"]["status"] == "complete"
    training_complete = phase in {"training_complete", "complete"}
    body = {
        "schema": RUN_ARTIFACT_CONTRACT_SCHEMA,
        "status": phase,
        "run_id": run_id,
        "run_config_content_sha256": run_config_content,
        "dataset_manifest_content_sha256": manifest[
            "content_sha256_before_self_field"
        ],
        "dataset_hashes_content_sha256": hashes[
            "content_sha256_before_self_field"
        ],
        "dataset_identity_sha256": hashes["identity_sha256"],
        "required_artifact_paths": list(REQUIRED_RUN_ARTIFACTS),
        "receipts": receipts,
        "receipt_set_identity_sha256": canonical_sha256(receipts),
        "gates": {
            "all_required_artifacts_present": True,
            "all_receipts_content_addressed": True,
            "all_schema_status_gates_passed": True,
            "dataset_manifest_derivation_passed": True,
            "builder_and_dataset_identities_bound": True,
            "protected_evaluation_content_opened": False,
            "training_outputs_complete": training_complete,
            "evaluation_complete": evaluation_complete,
            "launch_authorized": phase in {"launch_ready", "resume_ready"},
            "selection_or_deployment_authorized": phase == "complete",
        },
    }
    return self_address(body)


def validate_run_artifact_contract(
    run_directory: Path,
    value: Any,
    *,
    repository_root: Path,
    authority: Any,
    builders: Mapping[str, Mapping[str, Any]],
    expected_phase: str,
) -> dict[str, Any]:
    validate_self_address(value, label="run artifact contract")
    _require(
        value.get("schema") == RUN_ARTIFACT_CONTRACT_SCHEMA,
        "run artifact contract schema changed",
    )
    expected = seal_run_artifacts(
        run_directory,
        repository_root=repository_root,
        authority=authority,
        builders=builders,
        phase=expected_phase,
    )
    _require(value == expected, "run artifact receipt is stale, forged, or ambiguous")
    return value


def validate_immutable_run_artifacts(
    run_directory: Path,
    value: Any,
    *,
    repository_root: Path,
    authority: Any,
    builders: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate prior immutable receipts before a legitimate resume refresh."""

    validate_self_address(value, label="prior run artifact contract")
    _require(
        value.get("schema") == RUN_ARTIFACT_CONTRACT_SCHEMA
        and value.get("status") in RUN_PHASES,
        "prior run artifact contract status changed",
    )
    receipts = value.get("receipts")
    _require(
        isinstance(receipts, dict) and set(receipts) == set(REQUIRED_RUN_ARTIFACTS),
        "prior artifact receipt inventory changed",
    )
    run = _secure_run_directory(
        run_directory, repository_root=repository_root
    )
    for name, spec in REQUIRED_RUN_ARTIFACTS.items():
        if spec.mutability != "immutable":
            continue
        observed = _artifact_receipt(
            run,
            name,
            spec,
            phase=value["status"],
        )
        _require(observed == receipts[name], f"immutable artifact changed: {name}")
    validate_dataset_files(
        run,
        repository_root=repository_root,
        authority=authority,
        builders=builders,
    )
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Small deterministic writer for callers and synthetic tests."""

    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
