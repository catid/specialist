#!/usr/bin/env python3
"""Fail-closed runtime reader for the immutable mixed SFT snapshot.

This module deliberately validates the launch metadata before opening either
``sequences.jsonl`` or ``schedule.jsonl``.  The latter files contain training
content; a provisional or source-disjointness-pending snapshot therefore
cannot be inspected accidentally by the trainer.
"""

from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
TOP_SCHEMA = "mixed-training-snapshot-authority-manifest-v1"
VARIANT_SCHEMA = "mixed-training-snapshot-variant-manifest-v1"
SEQUENCE_SCHEMA = "mixed-training-packed-sequence-v1"
SCHEDULE_SCHEMA = "mixed-training-schedule-cursor-v1"
CURSOR_ALGORITHM = "sha256-canonical-schedule-row-without-current-commitment-v1"
ZERO_COMMITMENT = "0" * 64
ALLOWED_VARIANTS = {"protocol_core_100k", "full_authorized_markdown"}
ALLOWED_STREAMS = {"domain_qa", "raw_markdown", "replay"}
STREAM_ORDER = ("domain_qa", "raw_markdown", "replay")
ALLOWED_FORMATS = {"chat_assistant_only", "causal_next_token"}
QWEN36_VOCAB_SIZE = 248_320
SEED_QA_ROWS = 357
SEED_QA_ASSISTANT_TOKENS = 9_153
EXPECTED_BUDGETS = {
    "protocol_core_100k": {
        "domain_qa": 750_000,
        "raw_markdown": 100_000,
        "replay": 150_000,
    },
    "full_authorized_markdown": {
        "domain_qa": 750_000,
        "raw_markdown": 970_455,
        "replay": 150_000,
    },
}
REQUIRED_METADATA = {
    "category",
    "replay",
    "hard_negative",
    "verifier",
    "generator",
    "rights",
    "safety_transfer_flags",
    "lineage",
}
ALLOWED_RIGHTS_DECISIONS = {
    "authorized_synthetic_or_train_only_replay",
    "sealed_semantically_verified_seed_qa_projection",
    "eligible_by_recorded_open_or_public_domain_basis",
    "project_training_authorization_override",
    "sealed_generated_row_rights_receipt",
}
SEQUENCE_KEYS = {
    "schema",
    "sequence_id",
    "stream",
    "training_format",
    "label_semantics",
    "source_group_id",
    "source_document_id",
    "input_ids",
    "attention_mask",
    "labels",
    "input_token_count",
    "budget_token_count",
    "shifted_supervised_token_count",
    "segments",
    "packing",
}
SEGMENT_KEYS = {
    "unit_id",
    "token_start",
    "token_stop",
    "source_token_start",
    "source_token_stop",
    "budget_token_count",
    "metadata",
}
SCHEDULE_KEYS = {
    "schema",
    "cursor",
    "variant",
    "sequence_id",
    "sequence_sha256",
    "stream",
    "budget_token_count",
    "cumulative_budget_tokens",
    "cumulative_stream_budget_tokens",
    "previous_cursor_commitment_sha256",
    "cursor_commitment_sha256",
}
RESUME_IDENTITY = "(sequence_set_identity_sha256,cursor,cursor_commitment_sha256)"
CURSOR_FORMULA = (
    "current=canonical_sha256(schedule_row after removing only "
    "cursor_commitment_sha256); first previous='0'*64"
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class SnapshotContractError(RuntimeError):
    """The mixed snapshot or its launch authority failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SnapshotContractError(message)


def canonical_bytes(value: Any) -> bytes:
    """Return the builder's exact canonical form, including its newline."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def compact_ascii_sha256(value: Any) -> str:
    """Return the alternate self-address form accepted by the assembler."""

    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _load_self_addressed(path: Path, schema: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing regular manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"manifest is not an object: {path}")
    _require(value.get("schema") == schema, f"manifest schema changed: {path}")
    claimed = value.get("content_sha256_before_self_field")
    unsigned = copy.deepcopy(value)
    unsigned.pop("content_sha256_before_self_field", None)
    _require(_is_hex64(claimed), f"manifest self address missing: {path}")
    _require(canonical_sha256(unsigned) == claimed, f"manifest self address changed: {path}")
    return value


def _secure_regular_path(path: Path, *, label: str) -> Path:
    """Reject path aliases before any launch-authorized content is opened."""

    root = Path(os.path.abspath(os.fspath(ROOT)))
    lexical = Path(os.path.abspath(os.fspath(path)))
    _require(lexical.is_relative_to(root), f"{label} path escapes repository")
    current = Path(lexical.anchor)
    metadata = None
    for component in lexical.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise SnapshotContractError(f"{label} is missing") from exc
        _require(not stat.S_ISLNK(metadata.st_mode), f"{label} uses a symlink alias")
    _require(
        metadata is not None and stat.S_ISREG(metadata.st_mode),
        f"{label} is not a regular file",
    )
    _require(metadata.st_nlink == 1, f"{label} uses a hard-link alias")
    return lexical


def _repo_path(value: Any, *, label: str, expected: Path | None = None) -> Path:
    _require(isinstance(value, str) and value, f"{label} path is missing")
    declared = Path(value)
    _require(not declared.is_absolute(), f"{label} path must be repository relative")
    lexical = Path(os.path.abspath(os.fspath(ROOT / declared)))
    if expected is not None:
        expected_lexical = Path(os.path.abspath(os.fspath(expected)))
        _require(lexical == expected_lexical, f"{label} path is not canonical")
    return _secure_regular_path(lexical, label=label)


def _verify_file_receipt(
    receipt: Any,
    *,
    label: str,
    expected: Path,
) -> Path:
    _require(isinstance(receipt, dict), f"{label} receipt is missing")
    _require(set(receipt) == {"path", "sha256", "rows"}, f"{label} receipt fields changed")
    path = _repo_path(receipt.get("path"), label=label, expected=expected)
    _require(_is_hex64(receipt.get("sha256")), f"{label} SHA-256 is invalid")
    _require(file_sha256(path) == receipt["sha256"], f"{label} file identity changed")
    _require(
        isinstance(receipt["rows"], int)
        and not isinstance(receipt["rows"], bool)
        and receipt["rows"] > 0,
        f"{label} row count is invalid",
    )
    return path


def _validate_generated_receipt(receipt: Any) -> dict[str, Any]:
    _require(isinstance(receipt, dict), "generated-domain authority receipt is absent")
    required_hashes = {
        "manifest_file_sha256",
        "manifest_content_sha256",
        "report_file_sha256",
        "report_content_sha256",
        "dataset_file_sha256",
    }
    _require(
        set(receipt) == required_hashes | {"rows", "accounting"},
        "generated-domain receipt fields changed",
    )
    _require(
        all(_is_hex64(receipt.get(key)) for key in required_hashes),
        "generated-domain receipt identity is invalid",
    )
    _require(
        isinstance(receipt.get("rows"), int)
        and not isinstance(receipt.get("rows"), bool)
        and receipt["rows"] > 0,
        "generated-domain receipt row count is invalid",
    )
    accounting = receipt.get("accounting")
    _require(
        isinstance(accounting, dict)
        and accounting.get("assistant_tokens") == 740_847,
        "generated-domain receipt token authority changed",
    )
    return receipt


def _validate_source_disjoint_receipt(
    receipt: Any,
    *,
    expected: Path,
    generated_manifest_file_sha256: str,
) -> dict[str, Any]:
    _require(isinstance(receipt, dict), "opaque source-disjoint extension receipt is absent")
    _require(
        set(receipt)
        == {"path", "file_sha256", "content_sha256", "accepted", "opaque_receipt_sha256"},
        "opaque source-disjoint receipt fields changed",
    )
    _require(receipt.get("accepted") is True, "opaque source-disjoint extension is not accepted")
    for key in ("file_sha256", "content_sha256", "opaque_receipt_sha256"):
        _require(_is_hex64(receipt.get(key)), f"source-disjoint {key} is invalid")
    path = _repo_path(receipt.get("path"), label="source-disjoint extension", expected=expected)
    _require(file_sha256(path) == receipt["file_sha256"], "source-disjoint file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "source-disjoint extension is not an object")
    unsigned = copy.deepcopy(value)
    claimed = unsigned.pop("content_sha256_before_self_field", None)
    _require(
        claimed == receipt["content_sha256"]
        and claimed in {canonical_sha256(unsigned), compact_ascii_sha256(unsigned)},
        "source-disjoint content identity changed",
    )
    opaque = value.get("opaque_collision_contract")
    _require(
        value.get("schema") == "mixed-training-source-disjoint-extension-v1"
        and value.get("status") == "accepted"
        and value.get("accepted_for_training") is True
        and value.get("generated_domain_manifest_file_sha256")
        == generated_manifest_file_sha256
        and isinstance(opaque, dict)
        and opaque.get("status") == "passed"
        and opaque.get("train_collision_count") == 0
        and opaque.get("protected_source_content_opened") is False
        and opaque.get("protected_identifiers_disclosed") is False
        and opaque.get("opaque_receipt_sha256") == receipt["opaque_receipt_sha256"],
        "source-disjoint semantic gate changed",
    )
    return receipt


def _validate_tokenizer_receipt(receipt: Any) -> dict[str, Any]:
    _require(isinstance(receipt, dict), "tokenizer identity is absent")
    _require(set(receipt) == {"path", "revision", "chat_template_sha256"}, "tokenizer receipt fields changed")
    _require(isinstance(receipt.get("path"), str) and receipt["path"], "tokenizer path is invalid")
    _require(isinstance(receipt.get("revision"), str) and receipt["revision"], "tokenizer revision is invalid")
    _require(_is_hex64(receipt.get("chat_template_sha256")), "chat-template identity is invalid")
    return receipt


def _validate_seed_qa_semantic_receipt(receipt: Any) -> dict[str, Any]:
    expected_keys = {
        "schema", "status", "semantic_correctness_verified",
        "eligible_for_training", "rows", "assistant_qwen36_tokens",
        "training_rows_admitted", "file_sha256", "content_sha256",
        "source_dataset_file_sha256",
    }
    _require(isinstance(receipt, dict), "seed QA semantic authority is absent")
    _require(set(receipt) == expected_keys, "seed QA semantic receipt fields changed")
    _require(
        receipt.get("schema") == "seed-qa-semantic-authority-v1"
        and receipt.get("status") == "sealed_passed"
        and receipt.get("semantic_correctness_verified") is True
        and receipt.get("eligible_for_training") is True,
        "seed QA semantic authority did not pass",
    )
    _require(
        receipt.get("rows") == SEED_QA_ROWS
        and receipt.get("training_rows_admitted") == SEED_QA_ROWS
        and receipt.get("assistant_qwen36_tokens")
        == SEED_QA_ASSISTANT_TOKENS,
        "seed QA semantic authority accounting changed",
    )
    for key in ("file_sha256", "content_sha256", "source_dataset_file_sha256"):
        _require(_is_hex64(receipt.get(key)), f"seed QA semantic {key} is invalid")
    return receipt


def _validate_rights_receipt(receipt: Any) -> None:
    _require(isinstance(receipt, dict), "rights-exclusion receipt is absent")
    _require(receipt.get("schema") == "mixed-training-rights-exclusion-receipt-v1", "rights receipt schema changed")
    _require(receipt.get("status") == "passed", "rights receipt did not pass")
    _require(receipt.get("unresolved_or_unauthorized_rows") == 0, "unresolved rights entered the snapshot")
    _require(receipt.get("forbidden_source_classes_opened") is False, "forbidden source class was opened")
    _require(receipt.get("rejected_unreviewed_or_ineligible_rows_included") is False, "ineligible row entered the snapshot")
    _require(receipt.get("source_rights_status_rewritten") is False, "source rights metadata was rewritten")
    _require(
        isinstance(receipt.get("rows"), int)
        and not isinstance(receipt.get("rows"), bool)
        and receipt["rows"] > 0,
        "rights receipt row count is invalid",
    )
    decisions = receipt.get("rights_decisions")
    _require(
        isinstance(decisions, dict)
        and decisions
        and all(
            isinstance(name, str)
            and name
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
            for name, count in decisions.items()
        ),
        "rights decisions are invalid",
    )
    _require(set(decisions).issubset(ALLOWED_RIGHTS_DECISIONS), "unknown rights decision entered snapshot")
    _require(sum(decisions.values()) == receipt["rows"], "rights decision accounting changed")


def _variant_reference(top: dict[str, Any], variant: str) -> dict[str, Any]:
    variants = top.get("variants")
    _require(isinstance(variants, dict), "top manifest variants are missing")
    reference = variants.get(variant)
    _require(isinstance(reference, dict), f"top manifest omitted variant {variant}")
    return reference


def validate_launch_manifests(
    top_manifest_path: Path,
    *,
    variant: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    """Validate all metadata gates before returning content-file paths."""

    _require(variant in ALLOWED_VARIANTS, "unknown mixed-snapshot variant")
    candidate = Path(top_manifest_path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    top_path = _secure_regular_path(candidate, label="top manifest")
    top = _load_self_addressed(top_path, TOP_SCHEMA)
    _require(top.get("status") == "complete_launchable", "snapshot is not complete and launchable")
    _require(top.get("training_launch_authorized") is True, "top manifest does not authorize training")
    _require(top.get("source_disjoint_extension_accepted") is True, "top source-disjoint gate is not accepted")
    _require(top.get("seed_qa_semantic_authority_passed") is True, "top seed QA semantic gate did not pass")
    _require(top.get("rights_exclusion_gate_passed") is True, "top rights-exclusion gate did not pass")
    _require(top.get("max_sequence_length") == 2048, "top sequence-length contract changed")
    packing_invariants = top.get("packing_invariants")
    _require(isinstance(packing_invariants, dict), "top packing invariants are absent")
    _require(
        packing_invariants.get("same_stream_format_source_group_and_document_only") is True
        and packing_invariants.get("cross_source_group") is False
        and packing_invariants.get("cross_document") is False,
        "top packing invariants changed",
    )
    gates = top.get("gates")
    _require(isinstance(gates, dict), "top launch gates are absent")
    for key in (
        "generated_domain_semantic_authority_passed",
        "seed_qa_semantic_authority_passed",
        "source_disjoint_extension_accepted",
        "rights_exclusion_gate_passed",
        "tokenizer_identity_passed",
        "all_variant_manifests_launch_authorized",
    ):
        _require(gates.get(key) is True, f"top launch gate did not pass: {key}")
    generated_receipt = _validate_generated_receipt(top.get("generated_domain_authority"))
    seed_qa_semantic_receipt = _validate_seed_qa_semantic_receipt(
        top.get("seed_qa_semantic_authority")
    )
    tokenizer_receipt = _validate_tokenizer_receipt(top.get("tokenizer"))
    disjoint = _validate_source_disjoint_receipt(
        gates.get("source_disjoint_extension_receipt"),
        expected=top_path.parent / "source_disjoint_extension_v1.json",
        generated_manifest_file_sha256=generated_receipt[
            "manifest_file_sha256"
        ],
    )

    reference = _variant_reference(top, variant)
    _require(set(top["variants"]) == ALLOWED_VARIANTS, "top manifest variant set changed")
    reference_keys = {
        "manifest_path",
        "manifest_file_sha256",
        "manifest_content_sha256",
        "budget_tokens",
        "sequence_file_sha256",
        "sequence_rows",
        "schedule_file_sha256",
        "schedule_rows",
        "sequence_set_identity_sha256",
        "schedule_final_cursor_commitment_sha256",
        "exact_budget_tokens_by_stream",
    }
    _require(set(reference) == reference_keys, f"{variant} top reference fields changed")
    manifest_value = reference.get("manifest_path") or reference.get("path")
    variant_path = _repo_path(
        manifest_value,
        label=f"{variant} manifest",
        expected=top_path.parent / variant / "manifest.json",
    )
    _require(
        _is_hex64(reference.get("manifest_file_sha256"))
        and file_sha256(variant_path) == reference["manifest_file_sha256"],
        f"{variant} manifest file identity changed",
    )
    manifest = _load_self_addressed(variant_path, VARIANT_SCHEMA)
    _require(
        reference.get("manifest_content_sha256")
        == manifest["content_sha256_before_self_field"],
        f"{variant} manifest content identity changed",
    )
    _require(manifest.get("variant") == variant, "variant manifest name changed")
    _require(manifest.get("status") == "complete_launchable", "variant is not complete and launchable")
    _require(manifest.get("training_launch_authorized") is True, "variant does not authorize training")
    _require(manifest.get("max_sequence_length") == 2048, "sequence-length contract changed")
    _require(manifest.get("source_disjoint_extension_accepted") is True, "variant source-disjoint gate is not accepted")
    _require(manifest.get("seed_qa_semantic_authority_passed") is True, "variant seed QA semantic gate did not pass")
    _require(manifest.get("rights_exclusion_gate_passed") is True, "variant rights gate did not pass")
    expected_budgets = EXPECTED_BUDGETS[variant]
    _require(manifest.get("budget_tokens_by_stream") == expected_budgets, "variant stream budgets changed")
    _require(manifest.get("budget_tokens") == sum(expected_budgets.values()), "variant total budget changed")
    _require(_validate_tokenizer_receipt(manifest.get("tokenizer")) == tokenizer_receipt, "variant tokenizer differs from top authority")
    _require(
        manifest.get("resume", {}).get("cursor_commitment_algorithm") == CURSOR_ALGORITHM,
        "cursor commitment algorithm changed",
    )
    resume = manifest.get("resume")
    _require(isinstance(resume, dict), "variant resume contract is missing")
    _require(resume.get("resume_identity") == RESUME_IDENTITY, "resume identity contract changed")
    _require(resume.get("cursor_commitment_formula") == CURSOR_FORMULA, "cursor formula changed")
    _require(resume.get("initial_cursor_commitment_sha256") == ZERO_COMMITMENT, "initial cursor commitment changed")
    packing = manifest.get("packing")
    _require(isinstance(packing, dict), "packing contract is missing")
    _require(packing.get("policy") == "same_stream_format_source_group_and_document_greedy_v1", "packing policy changed")
    _require(packing.get("same_source_group_and_document_only") is True, "same-document packing is not required")
    _require(packing.get("cross_document") is False, "cross-document packing is authorized")
    _require(packing.get("cross_source_group") is False, "cross-source packing is authorized")
    labels = manifest.get("label_semantics")
    _require(isinstance(labels, dict), "label-semantics contract is missing")
    _require(labels.get("chat") == "official_qwen_chat_assistant_only_v1", "chat label policy changed")
    _require(labels.get("markdown") == "causal_next_token_all_tokens_v1", "Markdown label policy changed")
    _require(labels.get("prompt_tokens_supervised") is False, "prompt supervision was enabled")
    variant_gates = manifest.get("gates")
    _require(isinstance(variant_gates, dict), "variant launch gates are absent")
    for key in (
        "generated_domain_semantic_authority_passed",
        "seed_qa_semantic_authority_passed",
        "source_disjoint_extension_accepted",
        "rights_exclusion_gate_passed",
        "tokenizer_identity_passed",
        "exact_token_accounting_passed",
        "packing_invariants_passed",
    ):
        _require(variant_gates.get(key) is True, f"variant launch gate did not pass: {key}")
    _require(variant_gates.get("generated_domain_receipt") == generated_receipt, "variant generated authority differs")
    _require(
        _validate_seed_qa_semantic_receipt(
            variant_gates.get("seed_qa_semantic_authority_receipt")
        ) == seed_qa_semantic_receipt,
        "variant seed QA semantic authority differs",
    )
    _require(variant_gates.get("source_disjoint_extension_receipt") == disjoint, "variant source-disjoint receipt differs")
    _validate_rights_receipt(variant_gates.get("rights_exclusion_receipt"))

    sequences_receipt = manifest.get("sequences")
    schedule_receipt = manifest.get("schedule")
    _require(isinstance(sequences_receipt, dict), "sequence-set receipt is absent")
    _require(isinstance(schedule_receipt, dict), "schedule receipt is absent")
    _require(reference.get("sequence_file_sha256") == sequences_receipt.get("sha256"), "top sequence identity differs")
    _require(reference.get("sequence_rows") == sequences_receipt.get("rows"), "top sequence row count differs")
    _require(reference.get("schedule_file_sha256") == schedule_receipt.get("sha256"), "top schedule identity differs")
    _require(reference.get("schedule_rows") == schedule_receipt.get("rows"), "top schedule row count differs")
    _require(reference.get("budget_tokens") == manifest["budget_tokens"], "top total budget differs")
    _require(reference.get("exact_budget_tokens_by_stream") == expected_budgets, "top stream budgets differ")
    _require(reference.get("sequence_set_identity_sha256") == manifest.get("sequence_set_identity_sha256"), "top sequence-set identity differs")
    _require(
        reference.get("schedule_final_cursor_commitment_sha256")
        == manifest.get("schedule_final_cursor_commitment_sha256"),
        "top final cursor commitment differs",
    )
    _require(resume.get("sequence_set_identity_sha256") == manifest.get("sequence_set_identity_sha256"), "manifest sequence-set identity differs")
    _require(resume.get("final_cursor_commitment_sha256") == manifest.get("schedule_final_cursor_commitment_sha256"), "manifest final commitment differs")

    # Only now, after every launch gate above, may content receipts open files.
    sequence_path = _verify_file_receipt(
        sequences_receipt,
        label="sequence set",
        expected=variant_path.parent / "sequences.jsonl",
    )
    schedule_path = _verify_file_receipt(
        schedule_receipt,
        label="schedule",
        expected=variant_path.parent / "schedule.jsonl",
    )
    return top, manifest, sequence_path, schedule_path


def _read_jsonl(
    path: Path,
    *,
    expected_rows: int,
    expected_sha256: str,
) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    _require(
        hashlib.sha256(raw).hexdigest() == expected_sha256,
        f"{path}: content changed after launch-metadata validation",
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SnapshotContractError(f"{path}: content is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(keepends=True), 1):
        _require(line.endswith("\n"), f"{path}:{line_number}: noncanonical line ending")
        _require(line.strip(), f"{path}:{line_number}: blank row")
        value = json.loads(line)
        _require(isinstance(value, dict), f"{path}:{line_number}: row is not an object")
        _require(
            canonical_bytes(value) == line.encode("utf-8"),
            f"{path}:{line_number}: row is not canonical JSON",
        )
        rows.append(value)
    _require(len(rows) == expected_rows, f"{path}: row count changed")
    return rows


def _int_list(value: Any, *, label: str) -> list[int]:
    _require(isinstance(value, list) and value, f"{label} must be a nonempty list")
    _require(
        all(isinstance(item, int) and not isinstance(item, bool) for item in value),
        f"{label} contains a non-integer",
    )
    return value


def _verifier_receipt_passed(receipt: Any) -> bool:
    if not isinstance(receipt, dict) or not receipt:
        return False
    if "status" in receipt:
        return receipt.get("status") in {"passed", "sealed_passed"}
    # Generated-domain rows retain the exact per-verifier receipt map rather
    # than a redundant aggregate status.
    return all(
        isinstance(item, dict) and item.get("status") == "passed"
        for item in receipt.values()
    )


def validate_sequences(rows: Iterable[dict[str, Any]], *, max_tokens: int = 2048) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    fragments_by_unit: dict[str, list[dict[str, Any]]] = {}
    for position, row in enumerate(rows):
        prefix = f"sequence row {position}"
        _require(set(row) == SEQUENCE_KEYS, f"{prefix}: fields changed")
        _require(row.get("schema") == SEQUENCE_SCHEMA, f"{prefix}: schema changed")
        sequence_id = row.get("sequence_id")
        _require(isinstance(sequence_id, str) and sequence_id, f"{prefix}: invalid ID")
        _require(sequence_id not in result, f"{prefix}: duplicate ID")
        stream = row.get("stream")
        training_format = row.get("training_format")
        _require(stream in ALLOWED_STREAMS, f"{prefix}: invalid stream")
        _require(training_format in ALLOWED_FORMATS, f"{prefix}: invalid format")
        expected_format = "causal_next_token" if stream == "raw_markdown" else "chat_assistant_only"
        _require(training_format == expected_format, f"{prefix}: stream/format mismatch")
        expected_semantics = (
            "causal_next_token_all_tokens_v1"
            if training_format == "causal_next_token"
            else "official_qwen_chat_assistant_only_v1"
        )
        _require(row.get("label_semantics") == expected_semantics, f"{prefix}: label semantics changed")
        source_group = row.get("source_group_id")
        source_document = row.get("source_document_id")
        _require(isinstance(source_group, str) and source_group, f"{prefix}: source group missing")
        _require(isinstance(source_document, str) and source_document, f"{prefix}: source document missing")

        input_ids = _int_list(row.get("input_ids"), label=f"{prefix} input IDs")
        attention = _int_list(row.get("attention_mask"), label=f"{prefix} attention mask")
        labels = _int_list(row.get("labels"), label=f"{prefix} labels")
        _require(1 <= len(input_ids) <= max_tokens, f"{prefix}: token length is outside contract")
        _require(len(attention) == len(input_ids) == len(labels), f"{prefix}: tensor lengths differ")
        _require(
            all(0 <= item < QWEN36_VOCAB_SIZE for item in input_ids),
            f"{prefix}: input token is outside the Qwen3.6 vocabulary",
        )
        _require(attention == [1] * len(input_ids), f"{prefix}: packed row contains padding or masking")
        _require(
            all(label == -100 or label == token for token, label in zip(input_ids, labels)),
            f"{prefix}: labels are neither masked nor exact causal targets",
        )
        if training_format == "causal_next_token":
            _require(labels == input_ids, f"{prefix}: Markdown is not all-token causal loss")
        else:
            _require(any(label == -100 for label in labels), f"{prefix}: chat prompt is unmasked")
            _require(any(label != -100 for label in labels[1:]), f"{prefix}: chat has no shifted target")
        _require(row.get("input_token_count") == len(input_ids), f"{prefix}: input token accounting changed")
        shifted = sum(label != -100 for label in labels[1:])
        _require(row.get("shifted_supervised_token_count") == shifted, f"{prefix}: shifted target count changed")
        budget = row.get("budget_token_count")
        _require(isinstance(budget, int) and not isinstance(budget, bool) and budget > 0, f"{prefix}: invalid budget")

        packing = row.get("packing")
        _require(isinstance(packing, dict), f"{prefix}: packing metadata missing")
        _require(
            set(packing) == {"policy", "max_tokens", "cross_source_group", "cross_document"},
            f"{prefix}: packing fields changed",
        )
        _require(
            packing.get("policy") == "same_stream_format_source_group_and_document_greedy_v1",
            f"{prefix}: packing policy changed",
        )
        _require(packing.get("max_tokens") == max_tokens, f"{prefix}: pack length changed")
        _require(packing.get("cross_source_group") is False, f"{prefix}: cross-source packing")
        _require(packing.get("cross_document") is False, f"{prefix}: cross-document packing")
        segments = row.get("segments")
        _require(isinstance(segments, list) and segments, f"{prefix}: segments missing")
        cursor = 0
        segment_budget = 0
        unit_ids: set[str] = set()
        for segment_index, segment in enumerate(segments):
            label = f"{prefix} segment {segment_index}"
            _require(isinstance(segment, dict), f"{label}: not an object")
            _require(set(segment) == SEGMENT_KEYS, f"{label}: fields changed")
            unit_id = segment.get("unit_id")
            _require(isinstance(unit_id, str) and unit_id and unit_id not in unit_ids, f"{label}: invalid/duplicate unit")
            unit_ids.add(unit_id)
            start, stop = segment.get("token_start"), segment.get("token_stop")
            _require(start == cursor and isinstance(stop, int) and stop > start, f"{label}: noncontiguous span")
            _require(stop <= len(input_ids), f"{label}: span exceeds sequence")
            source_start, source_stop = segment.get("source_token_start"), segment.get("source_token_stop")
            _require(
                isinstance(source_start, int)
                and isinstance(source_stop, int)
                and 0 <= source_start < source_stop
                and source_stop - source_start == stop - start,
                f"{label}: source span changed",
            )
            item_budget = segment.get("budget_token_count")
            _require(isinstance(item_budget, int) and not isinstance(item_budget, bool) and item_budget > 0, f"{label}: invalid budget")
            segment_labels = labels[start:stop]
            if training_format == "causal_next_token":
                _require(item_budget == stop - start, f"{label}: Markdown budget changed")
            else:
                _require(source_start == 0, f"{label}: chat unit was split")
                _require(
                    item_budget == sum(item != -100 for item in segment_labels),
                    f"{label}: assistant budget changed",
                )
                _require(any(item == -100 for item in segment_labels), f"{label}: chat prompt is unmasked")
                _require(any(item != -100 for item in segment_labels), f"{label}: chat assistant is unsupervised")
            segment_budget += item_budget
            metadata = segment.get("metadata")
            _require(isinstance(metadata, dict), f"{label}: metadata missing")
            _require(REQUIRED_METADATA.issubset(metadata), f"{label}: required lineage metadata missing")
            _require(isinstance(metadata.get("category"), str) and metadata["category"], f"{label}: category missing")
            _require(isinstance(metadata.get("hard_negative"), bool), f"{label}: hard-negative flag invalid")
            _require(
                _verifier_receipt_passed(metadata.get("verifier")),
                f"{label}: verifier receipt did not pass",
            )
            _require(isinstance(metadata.get("generator"), dict), f"{label}: generator receipt missing")
            _require(isinstance(metadata.get("rights"), dict), f"{label}: rights decision missing")
            _require(
                metadata["rights"].get("decision") in ALLOWED_RIGHTS_DECISIONS,
                f"{label}: rights decision is not authorized",
            )
            _require(isinstance(metadata.get("safety_transfer_flags"), list), f"{label}: safety flags invalid")
            _require(isinstance(metadata.get("lineage"), dict), f"{label}: lineage missing")
            _require(metadata.get("source_group_id", source_group) == source_group, f"{label}: source group differs")
            _require(metadata.get("source_document_id", source_document) == source_document, f"{label}: source document differs")
            _require(metadata.get("replay") is (stream == "replay"), f"{label}: replay flag differs")
            fragments_by_unit.setdefault(unit_id, []).append(
                {
                    "training_format": training_format,
                    "stream": stream,
                    "source_group_id": source_group,
                    "source_document_id": source_document,
                    "source_token_start": source_start,
                    "source_token_stop": source_stop,
                    "metadata_sha256": canonical_sha256(metadata),
                }
            )
            cursor = stop
        _require(cursor == len(input_ids), f"{prefix}: segments do not cover sequence")
        _require(segment_budget == budget, f"{prefix}: segment budget differs")
        identity = {
            "stream": stream,
            "training_format": training_format,
            "source_group_id": source_group,
            "source_document_id": source_document,
            "input_ids_sha256": canonical_sha256(input_ids),
            "labels_sha256": canonical_sha256(labels),
            "segment_spans": [
                {
                    key: segment[key]
                    for key in (
                        "unit_id",
                        "token_start",
                        "token_stop",
                        "source_token_start",
                        "source_token_stop",
                    )
                }
                for segment in segments
            ],
        }
        _require(
            sequence_id == "mixed-sequence-v1:" + canonical_sha256(identity),
            f"{prefix}: deterministic sequence identity changed",
        )
        result[sequence_id] = row
    _require(result, "sequence set is empty")
    for unit_id, fragments in fragments_by_unit.items():
        prefix = f"training unit {unit_id}"
        ownership = {
            (
                item["training_format"],
                item["stream"],
                item["source_group_id"],
                item["source_document_id"],
                item["metadata_sha256"],
            )
            for item in fragments
        }
        _require(len(ownership) == 1, f"{prefix}: fragment ownership changed")
        if fragments[0]["training_format"] == "chat_assistant_only":
            _require(len(fragments) == 1, f"{prefix}: chat unit duplicated or split")
            continue
        spans = sorted(
            (item["source_token_start"], item["source_token_stop"])
            for item in fragments
        )
        expected_start = 0
        for start, stop in spans:
            _require(start == expected_start, f"{prefix}: Markdown source spans overlap or have a gap")
            expected_start = stop
    return result


def validate_schedule(
    rows: Iterable[dict[str, Any]],
    *,
    variant: str,
    sequences: dict[str, dict[str, Any]],
    budgets: dict[str, int],
) -> dict[str, Any]:
    schedule = list(rows)
    _require(schedule, "schedule is empty")
    _require(variant in EXPECTED_BUDGETS, "unknown schedule variant")
    _require(budgets == EXPECTED_BUDGETS[variant], "variant stream budgets changed")
    _require(all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in budgets.values()), "variant budgets are invalid")
    queues = {
        stream: sorted(
            [sequence_id for sequence_id, row in sequences.items() if row["stream"] == stream]
        )
        for stream in STREAM_ORDER
    }
    queue_positions = {stream: 0 for stream in STREAM_ORDER}
    previous = ZERO_COMMITMENT
    delivered = {stream: 0 for stream in budgets}
    observed_ids: set[str] = set()
    for cursor, row in enumerate(schedule):
        prefix = f"schedule row {cursor}"
        _require(set(row) == SCHEDULE_KEYS, f"{prefix}: fields changed")
        _require(row.get("schema") == SCHEDULE_SCHEMA, f"{prefix}: schema changed")
        _require(row.get("cursor") == cursor, f"{prefix}: cursor is not contiguous")
        _require(row.get("variant") == variant, f"{prefix}: variant changed")
        sequence_id = row.get("sequence_id")
        _require(sequence_id in sequences, f"{prefix}: unknown sequence")
        _require(sequence_id not in observed_ids, f"{prefix}: duplicate sequence")
        available = [
            stream
            for stream in STREAM_ORDER
            if queue_positions[stream] < len(queues[stream])
        ]
        _require(available, f"{prefix}: schedule has excess rows")
        expected_stream = min(
            available,
            key=lambda item: (
                Fraction(delivered[item], budgets[item]),
                STREAM_ORDER.index(item),
            ),
        )
        expected_sequence_id = queues[expected_stream][queue_positions[expected_stream]]
        _require(
            row.get("stream") == expected_stream
            and sequence_id == expected_sequence_id,
            f"{prefix}: deterministic schedule order changed",
        )
        queue_positions[expected_stream] += 1
        observed_ids.add(sequence_id)
        sequence = sequences[sequence_id]
        _require(row.get("sequence_sha256") == canonical_sha256(sequence), f"{prefix}: sequence receipt changed")
        stream = row.get("stream")
        _require(stream == sequence["stream"], f"{prefix}: stream differs from sequence")
        count = row.get("budget_token_count")
        _require(count == sequence["budget_token_count"], f"{prefix}: budget differs from sequence")
        delivered[stream] += count
        _require(row.get("cumulative_budget_tokens") == sum(delivered.values()), f"{prefix}: cumulative budget changed")
        _require(row.get("cumulative_stream_budget_tokens") == delivered, f"{prefix}: stream cumulative changed")
        _require(row.get("previous_cursor_commitment_sha256") == previous, f"{prefix}: commitment chain broke")
        unsigned = copy.deepcopy(row)
        claimed = unsigned.pop("cursor_commitment_sha256", None)
        _require(_is_hex64(claimed), f"{prefix}: commitment missing")
        _require(canonical_sha256(unsigned) == claimed, f"{prefix}: commitment is invalid")
        previous = claimed
    _require(observed_ids == set(sequences), "schedule does not cover the exact sequence set")
    _require(delivered == budgets, "schedule did not deliver exact stream budgets")
    sequence_receipts = {
        sequence_id: canonical_sha256(sequence)
        for sequence_id, sequence in sequences.items()
    }
    sequence_set_identity = canonical_sha256({
        "variant": variant,
        "budgets": budgets,
        "sequence_receipts": dict(sorted(sequence_receipts.items())),
    })
    return {
        "cursor_count": len(schedule),
        "sequence_set_identity_sha256": sequence_set_identity,
        "initial_cursor_commitment_sha256": ZERO_COMMITMENT,
        "final_cursor_commitment_sha256": previous,
        "budget_tokens_by_stream": dict(delivered),
        "budget_tokens": sum(delivered.values()),
    }


@dataclass(frozen=True)
class MixedTrainingAuthority:
    variant: str
    top_manifest_path: Path
    top_manifest: dict[str, Any]
    variant_manifest: dict[str, Any]
    sequences: dict[str, dict[str, Any]]
    schedule: tuple[dict[str, Any], ...]
    sequence_set_identity_sha256: str
    final_cursor_commitment_sha256: str

    def sequence_for_cursor(self, cursor: int) -> dict[str, Any]:
        _require(0 <= cursor < len(self.schedule), "training cursor is outside schedule")
        return self.sequences[self.schedule[cursor]["sequence_id"]]


def load_training_authority(top_manifest_path: Path, *, variant: str) -> MixedTrainingAuthority:
    top, manifest, sequence_path, schedule_path = validate_launch_manifests(
        top_manifest_path, variant=variant
    )
    # Content files are opened only after every metadata launch gate above.
    sequence_rows = _read_jsonl(
        sequence_path,
        expected_rows=manifest["sequences"]["rows"],
        expected_sha256=manifest["sequences"]["sha256"],
    )
    schedule_rows = _read_jsonl(
        schedule_path,
        expected_rows=manifest["schedule"]["rows"],
        expected_sha256=manifest["schedule"]["sha256"],
    )
    sequences = validate_sequences(sequence_rows, max_tokens=manifest["max_sequence_length"])
    budgets = manifest.get("budget_tokens_by_stream")
    _require(isinstance(budgets, dict), "variant stream budgets are missing")
    schedule_audit = validate_schedule(
        schedule_rows,
        variant=variant,
        sequences=sequences,
        budgets=budgets,
    )
    resume = manifest.get("resume")
    _require(isinstance(resume, dict), "variant resume contract is missing")
    for key in (
        "sequence_set_identity_sha256",
        "initial_cursor_commitment_sha256",
        "final_cursor_commitment_sha256",
        "cursor_count",
    ):
        _require(resume.get(key) == schedule_audit[key], f"resume contract changed: {key}")
    _require(
        manifest.get("sequence_set_identity_sha256")
        == schedule_audit["sequence_set_identity_sha256"],
        "top-level sequence-set identity changed",
    )
    _require(
        manifest.get("schedule_final_cursor_commitment_sha256")
        == schedule_audit["final_cursor_commitment_sha256"],
        "top-level final cursor commitment changed",
    )
    _require(manifest.get("budget_tokens") == schedule_audit["budget_tokens"], "total token budget changed")
    return MixedTrainingAuthority(
        variant=variant,
        top_manifest_path=Path(top_manifest_path).resolve(),
        top_manifest=top,
        variant_manifest=manifest,
        sequences=sequences,
        schedule=tuple(schedule_rows),
        sequence_set_identity_sha256=schedule_audit["sequence_set_identity_sha256"],
        final_cursor_commitment_sha256=schedule_audit["final_cursor_commitment_sha256"],
    )


def validate_resume_identity(
    authority: MixedTrainingAuthority,
    state: dict[str, Any],
) -> int:
    _require(isinstance(state, dict), "resume state is not an object")
    _require(state.get("variant") == authority.variant, "resume variant differs")
    _require(
        state.get("sequence_set_identity_sha256")
        == authority.sequence_set_identity_sha256,
        "resume sequence-set identity differs",
    )
    cursor = state.get("cursor")
    _require(
        isinstance(cursor, int)
        and not isinstance(cursor, bool)
        and 0 <= cursor <= len(authority.schedule),
        "resume cursor is invalid",
    )
    expected_commitment = (
        ZERO_COMMITMENT
        if cursor == 0
        else authority.schedule[cursor - 1]["cursor_commitment_sha256"]
    )
    _require(
        state.get("cursor_commitment_sha256") == expected_commitment,
        "resume cursor commitment differs",
    )
    return cursor
