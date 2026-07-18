#!/usr/bin/env python3
"""Build protocol-compliant typed mixed SFT/CPT snapshots.

The production build is intentionally fail closed until both the separately
sealed generated-domain authority and the independent seed-QA semantic
authority exist.  While either is absent, this script emits only
content-addressed provisional/pending metadata; no train sequences or launch
authority are produced.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from itertools import groupby
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterable

from build_general_replay_corpus_v1 import load_qwen_tokenizer
from general_replay_v1 import canonical_bytes, canonical_sha256
from qwen_chat_masking_v1 import encode_chat_assistant_only
from run_general_replay_candidate_shard_v1 import (
    MODEL_DIRECTORY,
    MODEL_FILE_SHA256,
    MODEL_REVISION,
    validate_model_files,
)
import mixed_training_source_disjoint_claim_v1 as source_disjoint_claim


ROOT = Path(__file__).resolve().parent
INVENTORY = ROOT / "data/training_inventory"
HIGH_INFORMATION = INVENTORY / "high_information_domain_corpus_v1"
OUTPUT_DIRECTORY = INVENTORY / "mixed_training_snapshot_v1"
PROVISIONAL_MANIFEST = OUTPUT_DIRECTORY / "provisional_manifest.json"
SOURCE_DISJOINT_REQUEST = OUTPUT_DIRECTORY / "source_disjoint_claim_request_v1.json"
SOURCE_DISJOINT_AUTHORIZATION = (
    OUTPUT_DIRECTORY / "source_disjoint_claim_authorization_v1.json"
)
SOURCE_DISJOINT_EXTENSION = OUTPUT_DIRECTORY / "source_disjoint_extension_v1.json"
SOURCE_SPLIT_AUTHORITY = (
    INVENTORY / "source_group_split_authority_v1.json"
)
PENDING_SEED_QA_LEDGER = (
    OUTPUT_DIRECTORY / "pending_seed_qa_semantic_review_v1.jsonl"
)

SEED_QA = HIGH_INFORMATION / "seed_qa_train.jsonl"
SEED_QA_SEMANTIC_AUTHORITY = (
    HIGH_INFORMATION / "seed_qa_semantic_authority_v1.json"
)
SEED_QA_REVIEW_DIRECTORY = HIGH_INFORMATION / "seed_qa_semantic_review_v1"
SEED_QA_DECISION_BUNDLE = SEED_QA_REVIEW_DIRECTORY / "decisions.jsonl"
CORE_MARKDOWN = HIGH_INFORMATION / "raw_continuation_train.jsonl"
FULL_DIRECTORY = INVENTORY / "full_train_markdown_cpt_v1"
FULL_MARKDOWN = FULL_DIRECTORY / "train.jsonl"
FULL_MANIFEST = FULL_DIRECTORY / "manifest.json"
PROJECT_AUTHORIZATION = (
    ROOT / "data/site_corpora/registry/project_training_authorization_v1.json"
)
REPLAY_DIRECTORY = ROOT / "data/general_replay_v1/replay_authority_v1_150k"
REPLAY_DATA = REPLAY_DIRECTORY / "general_replay_authority_v1_150k.jsonl"
REPLAY_MANIFEST = REPLAY_DIRECTORY / "manifest.json"
SELECTION_CONTRACT = (
    HIGH_INFORMATION / "category_balanced_candidate_selection_contract.json"
)

GENERATED_DIRECTORY = HIGH_INFORMATION / "generated_domain_authority_v1"
GENERATED_DATA = GENERATED_DIRECTORY / "train.jsonl"
GENERATED_REPORT = GENERATED_DIRECTORY / "report.json"
GENERATED_MANIFEST = GENERATED_DIRECTORY / "manifest.json"

ASSEMBLER = Path(__file__).resolve()
BUILD_TIME = "2026-07-18T04:00:00Z"
MAX_SEQUENCE_TOKENS = 2_048
RESERVED_MODEL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")

STATIC_SHA256 = {
    SEED_QA: "8775b94f57d73d1c0a6d86cbeae4c59a299b09ae3a80b50267fe4f7da1ec9b9a",
    CORE_MARKDOWN: "c02ed51fa4329a0fa905eb17091550a714137eca3d54f03c50bdf393a45b9f79",
    FULL_MARKDOWN: "d92c99c2694ba4d626eacfb41e58c62f60e93a8f0be9ba8103dcae380fae0ff6",
    FULL_MANIFEST: "b8061602bc336098b40f4cabc65939e4dbdf4e30e76429735d58b7d040af3b15",
    PROJECT_AUTHORIZATION: "46c0026fb18c90f18f570f2adf6c2b60b846a1382e2d366e59189e187b70c741",
    REPLAY_DATA: "687b8801df8c6725787419c0db6073b243ed5785f250c5990d3f72cff67eb940",
    REPLAY_MANIFEST: "6e5e7fc095b647991dffcd3a0508468c64a9fc0ff8fd8b0b5d3367a35ecb803f",
    SELECTION_CONTRACT: "781580b2e968952cb55498ba5f609c473994fed5fbf5165688e3843dfc4e8784",
}
STATIC_SELF_SHA256 = {
    FULL_MANIFEST: "86f190717a024761a71219cf0e93535b408b52519dce43ff0ba45212aeb325ea",
    PROJECT_AUTHORIZATION: "6e24ce5fa5c93d15c5f152277b2f395a0b6d0c747b894928316ea667cc914225",
    REPLAY_MANIFEST: "573e4a2f480769d2ffd38593410b8d9b0fe37623af40fbe9046bd44b3b3a515d",
    SELECTION_CONTRACT: "fdfeedcd2fd46f02506a65e27bee50ab6e6995cb7f0233256af2a2913e23b07f",
}

BASE_GENERATED_ASSISTANT_TOKENS = 740_847
# Kept as the base-selection name for the existing static selection contract.
# The sealed generated authority may contain additional, independently
# receipted seed-QA replacement tokens.
GENERATED_ASSISTANT_TOKENS = BASE_GENERATED_ASSISTANT_TOKENS
SEED_QA_ROWS = 357
SEED_ASSISTANT_TOKENS = 9_153
DOMAIN_QA_ASSISTANT_TOKENS = 750_000
REPLAY_ASSISTANT_TOKENS = 150_000
CORE_MARKDOWN_TOKENS = 100_000
FULL_MARKDOWN_TOKENS = 970_455
GENERATED_TASK_FAMILY_TOKENS = {
    "closed_book_application": 540_847,
    "grounded_synthesis": 200_000,
}
GENERATED_CATEGORY_TOKENS = {
    "lineage_people_history": 148_169,
    "materials_inspection_care_equipment": 122_678,
    "rigging_hardpoints_uplines_mechanics": 200_000,
    "safety_anatomy_consent_risk": 150_000,
    "tying_knots_frictions_technique": 120_000,
}
GENERATED_MODE_TOKENS = {
    "calibrated_hard_negative": 56_250,
    "positive": 684_597,
}
GENERATED_SUBTYPE_TOKENS = {
    "application_scenario": 202_540,
    "calibrated_unanswerable": 18_032,
    "comparison_or_mechanism": 103_854,
    "conflict_or_scope_resolution": 29_085,
    "direct_explanation": 155_487,
    "evidence_grounded_answer": 90_129,
    "misconception_correction": 78_966,
    "multi_fact_synthesis": 62_754,
}

VARIANT_BUDGETS = {
    "protocol_core_100k": {
        "domain_qa": DOMAIN_QA_ASSISTANT_TOKENS,
        "raw_markdown": CORE_MARKDOWN_TOKENS,
        "replay": REPLAY_ASSISTANT_TOKENS,
    },
    "full_authorized_markdown": {
        "domain_qa": DOMAIN_QA_ASSISTANT_TOKENS,
        "raw_markdown": FULL_MARKDOWN_TOKENS,
        "replay": REPLAY_ASSISTANT_TOKENS,
    },
}

FORBIDDEN_PATH_TOKENS = frozenset({
    "benchmark", "benchmarks", "dev", "development", "developments",
    "eval", "evaluation", "evaluations", "final", "finals", "heldout",
    "holdout", "holdouts", "incident", "incidents", "manualreview",
    "manualreviews", "ood", "protected", "terminal", "terminals",
})
HEX64 = re.compile(r"^[0-9a-f]{64}$")
OPEN_RIGHTS_STATUSES = frozenset({
    "explicit_open_license",
    "federal_text_public_domain_presumption",
    "public_domain_in_usa_source_with_trademark_and_jurisdiction_limits",
})
GENERATED_RECEIPT_KEYS = {
    "primary_generation",
    "fill_generation",
    "structural_verification",
    "nli_verification",
    "two_pass_semantic_judges",
    "selector",
    "manual_review",
    "tokenizer_chat_template",
}
GENERATED_REQUIRED_ROW_KEYS = {
    "schema", "record_id", "candidate_example_id", "request_id",
    "source_context_id", "source_group_id", "resource_id", "split",
    "training_format", "messages", "tools", "assistant_mask",
    "assistant_qwen36_token_count", "task_family", "task_subtype",
    "generation_mode", "category", "question", "answer",
    "evidence_quote_sha256s", "verification_receipts",
    "manual_review_receipt", "dedupe", "generator", "rights_basis",
    "rights_authorization", "safety_transfer_flags", "lineage",
    "eligible_for_training",
}
SEED_QA_SEMANTIC_AUTHORITY_SCHEMA = "seed-qa-semantic-authority-v1"
PENDING_SEED_QA_SCHEMA = "pending-seed-qa-semantic-review-row-v1"
SEED_QA_DECISION_SCHEMA = "seed-qa-manual-semantic-decision-v1"
SEED_QA_REVIEW_METHOD = "manual_line_by_line_question_answer_evidence_v1"
SEED_QA_DECISION_KEYS = {
    "schema", "source_line_number", "record_id", "source_record_sha256",
    "reviewer_id", "review_method", "reviewed_question_answer_and_evidence",
    "reviewer_independent_of_source_generator", "decision",
    "semantic_correctness_verified", "evidence_entails_entire_answer",
    "question_is_user_useful", "question_is_self_contained",
    "answer_is_direct_and_well_formed", "safety_qualification_is_adequate",
    "reason_code", "notes", "decision_content_sha256",
}
SEED_QA_PASS_FLAGS = (
    "semantic_correctness_verified",
    "evidence_entails_entire_answer",
    "question_is_user_useful",
    "question_is_self_contained",
    "answer_is_direct_and_well_formed",
    "safety_qualification_is_adequate",
)
SEED_QA_EXCLUSION_REASONS = {
    "answer_not_fully_supported",
    "answer_factually_incorrect",
    "question_not_user_useful",
    "question_contains_unsupported_or_false_premise",
    "question_not_self_contained",
    "answer_not_direct_or_well_formed",
    "safety_qualification_inadequate",
    "multiple_quality_failures",
}
SEED_QA_AUTHORITY_KEYS = {
    "schema", "status", "semantic_correctness_verified",
    "eligible_for_training", "source_rows", "reviewed_rows",
    "training_rows_admitted", "excluded_rows",
    "source_assistant_qwen36_tokens", "assistant_qwen36_tokens",
    "excluded_assistant_qwen36_tokens",
    "replacement_generated_assistant_tokens_required", "source_dataset",
    "assignments", "decision_bundle", "decision_files",
    "admitted_record_identity_commitment_sha256", "exclusion_ledger",
    "review_contract", "content_sha256_before_self_field",
}
SEED_QA_RECEIPT_KEYS = {
    "schema", "status", "semantic_correctness_verified",
    "eligible_for_training", "path", "file_sha256", "content_sha256",
    "source_rows", "reviewed_rows", "training_rows_admitted",
    "excluded_rows", "source_assistant_qwen36_tokens",
    "assistant_qwen36_tokens", "excluded_assistant_qwen36_tokens",
    "replacement_generated_assistant_tokens_required", "source_dataset",
    "decision_bundle", "admitted_record_identity_commitment_sha256",
    "exclusion_ledger",
}
SEED_QA_REPLACEMENT_SCHEMA = "seed-qa-generated-replacement-receipt-v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compact_ascii_sha256(value: Any) -> str:
    """Canonical form used by the high-information inventory builders."""
    return sha256_bytes(json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _path_tokens(path: Path) -> set[str]:
    tokens = set()
    for component in path.parts:
        collapsed = re.sub(r"[^a-z0-9]", "", component.casefold())
        if collapsed:
            tokens.add(collapsed)
        tokens.update(
            item for item in re.split(r"[^a-z0-9]+", component.casefold())
            if item
        )
    return tokens


def secure_regular_input(
    path: Path, role: str, *, exact: Path | None = None
) -> Path:
    """Reject forbidden lexical paths, aliases, hard links, and non-files."""
    lexical = Path(os.path.abspath(os.fspath(path)))
    if _path_tokens(lexical) & FORBIDDEN_PATH_TOKENS:
        raise RuntimeError(f"{role}: forbidden source path class")
    if exact is not None and lexical != Path(os.path.abspath(os.fspath(exact))):
        raise RuntimeError(f"{role}: input must use its canonical path")
    current = Path(lexical.anchor)
    metadata = None
    for component in lexical.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ValueError(f"{role}: input is missing") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{role}: symlink aliases are forbidden")
    if metadata is None or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{role}: regular file required")
    if metadata.st_nlink != 1:
        raise RuntimeError(f"{role}: hard-link aliases are forbidden")
    return lexical


def pinned_bytes(path: Path, expected_sha256: str, role: str) -> bytes:
    if not HEX64.fullmatch(expected_sha256):
        raise ValueError(f"{role}: malformed content digest")
    safe = secure_regular_input(path, role, exact=path)
    raw = safe.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise RuntimeError(f"{role}: stale content hash")
    return raw


def load_jsonl_bytes(raw: bytes, role: str) -> list[dict]:
    result = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{role} line {line_number}: blank rows forbidden")
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{role} line {line_number}: invalid JSON") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{role} line {line_number}: object required")
        result.append(item)
    if not result:
        raise ValueError(f"{role}: empty input forbidden")
    return result


def load_pinned_jsonl(path: Path, role: str) -> list[dict]:
    return load_jsonl_bytes(pinned_bytes(path, STATIC_SHA256[path], role), role)


def load_pinned_json(path: Path, role: str) -> dict:
    value = json.loads(pinned_bytes(path, STATIC_SHA256[path], role))
    if not isinstance(value, dict):
        raise ValueError(f"{role}: object required")
    validate_self_hash(value, STATIC_SELF_SHA256[path], role)
    return value


def validate_self_hash(value: dict, expected: str | None, role: str) -> str:
    declared = value.get("content_sha256_before_self_field")
    if not isinstance(declared, str) or not HEX64.fullmatch(declared):
        raise RuntimeError(f"{role}: missing self hash")
    payload = dict(value)
    payload.pop("content_sha256_before_self_field")
    # The replay authority uses newline-terminated UTF-8 canonical JSON; the
    # high-information inventory uses compact ASCII canonical JSON.  Both are
    # explicit, deterministic contracts, and the pinned declared digest tells
    # us which contract owns the input.
    if declared not in {canonical_sha256(payload), compact_ascii_sha256(payload)}:
        raise RuntimeError(f"{role}: stale self hash")
    if expected is not None and declared != expected:
        raise RuntimeError(f"{role}: unexpected sealed identity")
    return declared


def _contains_reserved(value: Any) -> bool:
    if isinstance(value, str):
        return any(token in value for token in RESERVED_MODEL_TOKENS)
    if isinstance(value, dict):
        return any(_contains_reserved(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_reserved(item) for item in value)
    return False


def _require_unique(rows: list[dict], key: str, role: str) -> None:
    values = [row.get(key) for row in rows]
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{role}: {key} required")
    if len(set(values)) != len(values):
        raise RuntimeError(f"{role}: duplicate {key}")


def _authorized_resources(authorization: dict) -> dict[str, dict]:
    if (
        authorization.get("schema") != "site-corpus-project-training-authorization-v1"
        or authorization.get("authorized_by") != "user"
        or authorization.get("supersedes_public_rights_gate_for_project_use") is not True
        or authorization.get("changes_registry_rights_status") is not False
        or authorization.get("not_a_public_license_determination") is not True
    ):
        raise RuntimeError("project training authorization changed")
    result = {}
    for item in authorization.get("resources", []):
        if item.get("project_training_authorized") is not True:
            raise RuntimeError("project authorization contains an unauthorized row")
        result[item["resource_id"]] = item
    if set(result) != {"crash_restraint", "rope365", "rope_topia", "shibari_atlas"}:
        raise RuntimeError("project authorization resource scope changed")
    return result


def authorize_rights(
    rights_basis: Any,
    *,
    resource_id: str | None,
    artifact_id: str | None,
    component: str,
    authorized_resources: dict[str, dict],
    generated_rights_receipt: dict | None = None,
) -> dict:
    """Return a preserved rights decision or reject unauthorized content."""
    if component == "replay":
        return {
            "decision": "authorized_synthetic_or_train_only_replay",
            "original_rights_basis": rights_basis,
        }
    if component == "seed_qa":
        if (
            not isinstance(generated_rights_receipt, dict)
            or generated_rights_receipt.get("decision") != "pass"
            or generated_rights_receipt.get("semantic_correctness_verified")
            is not True
            or any(
                generated_rights_receipt.get(flag) is not True
                for flag in SEED_QA_PASS_FLAGS
            )
        ):
            raise RuntimeError(
                "seed_qa: sealed semantic correctness authority is required"
            )
        return {
            "decision": "sealed_semantically_verified_seed_qa_projection",
            "original_rights_basis": rights_basis,
            "original_status_preserved": True,
            "semantic_authority": generated_rights_receipt,
        }
    if not isinstance(rights_basis, dict):
        raise RuntimeError(f"{component}: rights basis is absent")
    status = rights_basis.get("status")
    if status in OPEN_RIGHTS_STATUSES:
        return {
            "decision": "eligible_by_recorded_open_or_public_domain_basis",
            "original_rights_basis": rights_basis,
        }
    if status == "legacy_manifest_gap":
        authorization = authorized_resources.get(resource_id)
        if (
            authorization is None
            or authorization.get("artifact_id") != artifact_id
            or authorization.get("project_training_authorized") is not True
        ):
            raise RuntimeError(f"{component}: unresolved rights lack scoped authorization")
        return {
            "decision": "project_training_authorization_override",
            "original_rights_basis": rights_basis,
            "authorization": authorization,
            "public_license_status_rewritten": False,
        }
    if component == "generated_domain" and generated_rights_receipt is not None:
        if (
            generated_rights_receipt.get("status") == "passed"
            and generated_rights_receipt.get("source_rights_status_preserved") is True
        ):
            return {
                "decision": "sealed_generated_row_rights_receipt",
                "original_rights_basis": rights_basis,
                "authorization": generated_rights_receipt,
            }
    raise RuntimeError(f"{component}: unauthorized rights status {status!r}")


def _unit(
    *, unit_id: str, stream: str, training_format: str,
    source_group_id: str, source_document_id: str, input_ids: list[int],
    labels: list[int], budget_token_count: int, order_key: tuple,
    metadata: dict,
) -> dict:
    if (
        not unit_id or not source_group_id or not source_document_id
        or len(input_ids) != len(labels) or not input_ids
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0
               for item in input_ids)
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or (item < 0 and item != -100)
            for item in labels
        )
        or not isinstance(budget_token_count, int) or budget_token_count <= 0
    ):
        raise ValueError("invalid normalized training unit")
    if stream not in {"domain_qa", "raw_markdown", "replay"}:
        raise ValueError("invalid training stream")
    return {
        "unit_id": unit_id,
        "stream": stream,
        "training_format": training_format,
        "source_group_id": source_group_id,
        "source_document_id": source_document_id,
        "input_ids": input_ids,
        "labels": labels,
        "budget_token_count": budget_token_count,
        "order_key": order_key,
        "metadata": metadata,
    }


def normalize_replay_rows(rows: list[dict], tokenizer: Any) -> list[dict]:
    _require_unique(rows, "row_id", "replay authority")
    result = []
    for row in rows:
        if (
            row.get("schema") != "general-behavior-replay-row-v1"
            or row.get("template_policy") != "official_qwen_apply_chat_template_v1"
            or row.get("assistant_mask", {}).get("policy") != "assistant_only_v1"
            or row.get("verifier", {}).get("status") != "passed"
            or not isinstance(row.get("generator"), dict)
            or _contains_reserved(row.get("messages"))
        ):
            raise RuntimeError("replay row contract changed")
        encoded = encode_chat_assistant_only(
            tokenizer, row["messages"], enable_thinking=False, tools=row["tools"]
        )
        if encoded["assistant_token_count"] != row.get("assistant_token_count"):
            raise RuntimeError("replay assistant-token count changed")
        source_group = row["source_group_id"]
        source_document = row.get("lineage", {}).get(
            "source_document_identity_sha256", source_group
        )
        rights = authorize_rights(
            None,
            resource_id=None,
            artifact_id=None,
            component="replay",
            authorized_resources={},
        )
        result.append(_unit(
            unit_id=row["row_id"],
            stream="replay",
            training_format="chat_assistant_only",
            source_group_id=source_group,
            source_document_id=source_document,
            input_ids=encoded["input_ids"],
            labels=encoded["labels"],
            budget_token_count=encoded["assistant_token_count"],
            order_key=(row["row_id"],),
            metadata={
                "category": row["category"],
                "replay": True,
                "hard_negative": bool(row.get("lineage", {}).get("hard_negative", False)),
                "verifier": row["verifier"],
                "generator": row["generator"],
                "rights": rights,
                "safety_transfer_flags": [],
                "lineage": row["lineage"],
                "source_record_sha256": canonical_sha256(row),
            },
        ))
    if sum(item["budget_token_count"] for item in result) != REPLAY_ASSISTANT_TOKENS:
        raise RuntimeError("replay budget changed")
    return result


def _validated_seed_encoding(row: dict, tokenizer: Any) -> dict:
    if (
        row.get("schema") != "high-information-seed-qa-v1"
        or row.get("assistant_supervision") is not True
        or row.get("enable_thinking") is not False
        or row.get("hidden_reasoning_supervision") is not False
        or row.get("training_family") != "closed_book_seed_qa"
        or not isinstance(row.get("source_group_id"), str)
        or not row["source_group_id"]
        or not HEX64.fullmatch(str(row.get("document_sha256", "")))
        or (
            row.get("evidence_sha256") is not None
            and not HEX64.fullmatch(str(row.get("evidence_sha256")))
        )
        or not isinstance(row.get("safety_transfer_flags"), list)
        or _contains_reserved([row.get("question"), row.get("answer")])
    ):
        raise RuntimeError("seed QA row contract changed")
    messages = [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]
    encoded = encode_chat_assistant_only(
        tokenizer, messages, enable_thinking=False, tools=[]
    )
    if encoded["assistant_token_count"] != row.get(
        "assistant_qwen36_token_count"
    ):
        raise RuntimeError("seed QA assistant-token count changed")
    return encoded


def pending_seed_qa_ledger(
    rows: list[dict], tokenizer: Any
) -> tuple[list[dict], dict]:
    """Preserve unverified seed rows as metadata-only pending exclusions."""

    _require_unique(rows, "record_id", "seed QA")
    pending = []
    for row in rows:
        encoded = _validated_seed_encoding(row, tokenizer)
        safety_status = row.get("safety_transfer_status")
        if (
            not isinstance(safety_status, str)
            or "semantic verification required" not in safety_status
        ):
            raise RuntimeError(
                "seed QA row no longer carries its semantic-review requirement"
            )
        pending.append({
            "schema": PENDING_SEED_QA_SCHEMA,
            "record_id": row["record_id"],
            "source_record_sha256": canonical_sha256(row),
            "source_group_id": row["source_group_id"],
            "source_document_identity_sha256": row["document_sha256"],
            "assistant_qwen36_token_count": encoded["assistant_token_count"],
            "disposition": "pending_excluded_from_training",
            "reason": "semantic_correctness_authority_absent",
            "required_authority_path": relative(SEED_QA_SEMANTIC_AUTHORITY),
            "required_authority_schema": SEED_QA_SEMANTIC_AUTHORITY_SCHEMA,
            "semantic_content_copied_to_ledger": False,
        })
    if (
        len(pending) != SEED_QA_ROWS
        or sum(item["assistant_qwen36_token_count"] for item in pending)
        != SEED_ASSISTANT_TOKENS
    ):
        raise RuntimeError("pending seed QA accounting changed")
    payload = _jsonl_payload(pending)
    receipt_body = {
        "schema": "pending-seed-qa-semantic-review-receipt-v1",
        "status": "pending_excluded",
        "path": relative(PENDING_SEED_QA_LEDGER),
        "file_sha256": sha256_bytes(payload),
        "rows": len(pending),
        "assistant_qwen36_tokens": sum(
            item["assistant_qwen36_token_count"] for item in pending
        ),
        "source_dataset": {
            "path": relative(SEED_QA),
            "file_sha256": STATIC_SHA256[SEED_QA],
        },
        "record_identity_commitment_sha256": canonical_sha256([
            {
                "record_id": item["record_id"],
                "source_record_sha256": item["source_record_sha256"],
            }
            for item in pending
        ]),
        "semantic_content_emitted": False,
        "training_rows_admitted": 0,
    }
    return pending, {
        **receipt_body,
        "content_sha256": canonical_sha256(receipt_body),
    }


def _seed_decision_body(decision: dict) -> dict:
    return {
        key: value
        for key, value in decision.items()
        if key != "decision_content_sha256"
    }


def _validate_seed_decision(
    row: dict, source_line_number: int, decision: Any
) -> dict:
    if not isinstance(decision, dict) or set(decision) != SEED_QA_DECISION_KEYS:
        raise RuntimeError("seed QA manual decision fields changed")
    if (
        decision.get("schema") != SEED_QA_DECISION_SCHEMA
        or decision.get("source_line_number") != source_line_number
        or decision.get("record_id") != row["record_id"]
        or decision.get("source_record_sha256")
        != compact_ascii_sha256(row)
        or not isinstance(decision.get("reviewer_id"), str)
        or not decision["reviewer_id"].startswith("codex-manual-review/")
        or decision.get("review_method") != SEED_QA_REVIEW_METHOD
        or decision.get("reviewed_question_answer_and_evidence") is not True
        or decision.get("reviewer_independent_of_source_generator") is not True
        or decision.get("decision") not in {"pass", "exclude"}
        or any(type(decision.get(flag)) is not bool for flag in SEED_QA_PASS_FLAGS)
        or not isinstance(decision.get("notes"), str)
        or not decision["notes"].strip()
        or decision.get("decision_content_sha256")
        != compact_ascii_sha256(_seed_decision_body(decision))
    ):
        raise RuntimeError("seed QA manual decision contract changed")
    if decision["decision"] == "pass":
        if (
            any(decision[flag] is not True for flag in SEED_QA_PASS_FLAGS)
            or decision.get("reason_code") != "fully_supported_useful_seed_qa"
        ):
            raise RuntimeError("seed QA passing decision is not fully proven")
    elif (
        all(decision[flag] is True for flag in SEED_QA_PASS_FLAGS)
        or decision.get("reason_code") not in SEED_QA_EXCLUSION_REASONS
    ):
        raise RuntimeError("seed QA exclusion has no explicit quality failure")
    return decision


def _seed_authority_receipt(
    authority: dict, *, authority_raw: bytes
) -> dict:
    receipt = {
        "schema": authority["schema"],
        "status": authority["status"],
        "semantic_correctness_verified": authority[
            "semantic_correctness_verified"
        ],
        "eligible_for_training": authority["eligible_for_training"],
        "path": relative(SEED_QA_SEMANTIC_AUTHORITY),
        "file_sha256": sha256_bytes(authority_raw),
        "content_sha256": authority["content_sha256_before_self_field"],
        "source_rows": authority["source_rows"],
        "reviewed_rows": authority["reviewed_rows"],
        "training_rows_admitted": authority["training_rows_admitted"],
        "excluded_rows": authority["excluded_rows"],
        "source_assistant_qwen36_tokens": authority[
            "source_assistant_qwen36_tokens"
        ],
        "assistant_qwen36_tokens": authority["assistant_qwen36_tokens"],
        "excluded_assistant_qwen36_tokens": authority[
            "excluded_assistant_qwen36_tokens"
        ],
        "replacement_generated_assistant_tokens_required": authority[
            "replacement_generated_assistant_tokens_required"
        ],
        "source_dataset": authority["source_dataset"],
        "decision_bundle": authority["decision_bundle"],
        "admitted_record_identity_commitment_sha256": authority[
            "admitted_record_identity_commitment_sha256"
        ],
        "exclusion_ledger": authority["exclusion_ledger"],
    }
    if set(receipt) != SEED_QA_RECEIPT_KEYS:
        raise RuntimeError("seed QA semantic receipt construction changed")
    return receipt


def load_seed_qa_semantic_authority(
    rows: list[dict], tokenizer: Any, authorized_resources: dict[str, dict]
) -> tuple[list[dict], dict, dict[str, dict]]:
    """Load and independently validate the live authority and decision bytes."""

    authority_path = secure_regular_input(
        SEED_QA_SEMANTIC_AUTHORITY,
        "seed QA semantic authority",
        exact=SEED_QA_SEMANTIC_AUTHORITY,
    )
    authority_raw = authority_path.read_bytes()
    authority = json.loads(authority_raw)
    if not isinstance(authority, dict) or set(authority) != SEED_QA_AUTHORITY_KEYS:
        raise RuntimeError("seed QA semantic authority fields changed")
    authority_content_sha256 = validate_self_hash(
        authority, None, "seed QA semantic authority"
    )
    integer_fields = (
        "source_rows", "reviewed_rows", "training_rows_admitted",
        "excluded_rows", "source_assistant_qwen36_tokens",
        "assistant_qwen36_tokens", "excluded_assistant_qwen36_tokens",
        "replacement_generated_assistant_tokens_required",
    )
    if (
        authority.get("schema") != SEED_QA_SEMANTIC_AUTHORITY_SCHEMA
        or authority.get("status") != "sealed_passed"
        or authority.get("semantic_correctness_verified") is not True
        or authority.get("eligible_for_training") is not True
        or any(
            not isinstance(authority.get(key), int)
            or isinstance(authority.get(key), bool)
            or authority[key] < 0
            for key in integer_fields
        )
        or authority.get("source_rows") != SEED_QA_ROWS
        or authority.get("reviewed_rows") != SEED_QA_ROWS
        or authority.get("training_rows_admitted")
        + authority.get("excluded_rows", -1)
        != SEED_QA_ROWS
        or authority.get("training_rows_admitted", 0) <= 0
        or authority.get("source_assistant_qwen36_tokens")
        != SEED_ASSISTANT_TOKENS
        or authority.get("assistant_qwen36_tokens", -1)
        + authority.get("excluded_assistant_qwen36_tokens", -1)
        != SEED_ASSISTANT_TOKENS
        or authority.get("replacement_generated_assistant_tokens_required")
        != authority.get("excluded_assistant_qwen36_tokens")
        or authority.get("source_dataset")
        != {
            "path": relative(SEED_QA),
            "file_sha256": STATIC_SHA256[SEED_QA],
        }
    ):
        raise RuntimeError("seed QA semantic authority accounting changed")
    if authority_content_sha256 != authority["content_sha256_before_self_field"]:
        raise RuntimeError("seed QA semantic authority identity changed")

    bundle_receipt = authority.get("decision_bundle")
    if (
        not isinstance(bundle_receipt, dict)
        or set(bundle_receipt) != {"path", "file_sha256", "bytes", "rows"}
        or bundle_receipt.get("path") != relative(SEED_QA_DECISION_BUNDLE)
        or not HEX64.fullmatch(str(bundle_receipt.get("file_sha256", "")))
        or bundle_receipt.get("rows") != SEED_QA_ROWS
        or not isinstance(bundle_receipt.get("bytes"), int)
        or isinstance(bundle_receipt.get("bytes"), bool)
        or bundle_receipt["bytes"] <= 0
    ):
        raise RuntimeError("seed QA decision-bundle receipt changed")
    bundle_raw = pinned_bytes(
        SEED_QA_DECISION_BUNDLE,
        bundle_receipt["file_sha256"],
        "seed QA semantic decision bundle",
    )
    if len(bundle_raw) != bundle_receipt["bytes"]:
        raise RuntimeError("seed QA decision-bundle byte count changed")
    decisions_list = load_jsonl_bytes(bundle_raw, "seed QA semantic decisions")
    if len(decisions_list) != len(rows):
        raise RuntimeError("seed QA semantic decision coverage changed")
    decisions = {}
    for source_line_number, (row, decision) in enumerate(
        zip(rows, decisions_list, strict=True), 1
    ):
        validated = _validate_seed_decision(row, source_line_number, decision)
        if validated["record_id"] in decisions:
            raise RuntimeError("seed QA semantic decision identity duplicated")
        decisions[validated["record_id"]] = validated

    token_by_record = {
        row["record_id"]: _validated_seed_encoding(row, tokenizer)[
            "assistant_token_count"
        ]
        for row in rows
    }
    admitted = [item for item in decisions_list if item["decision"] == "pass"]
    excluded = [item for item in decisions_list if item["decision"] == "exclude"]
    admitted_tokens = sum(token_by_record[item["record_id"]] for item in admitted)
    excluded_tokens = sum(token_by_record[item["record_id"]] for item in excluded)
    expected_exclusion_ledger = [
        {
            "record_id": item["record_id"],
            "source_record_sha256": item["source_record_sha256"],
            "decision_content_sha256": item["decision_content_sha256"],
            "reason_code": item["reason_code"],
            "assistant_qwen36_tokens": token_by_record[item["record_id"]],
        }
        for item in excluded
    ]
    expected_admitted_commitment = compact_ascii_sha256([
        {
            "record_id": item["record_id"],
            "source_record_sha256": item["source_record_sha256"],
            "decision_content_sha256": item["decision_content_sha256"],
        }
        for item in admitted
    ])
    review = authority.get("review_contract")
    if (
        authority.get("training_rows_admitted") != len(admitted)
        or authority.get("excluded_rows") != len(excluded)
        or authority.get("assistant_qwen36_tokens") != admitted_tokens
        or authority.get("excluded_assistant_qwen36_tokens") != excluded_tokens
        or authority.get("exclusion_ledger") != expected_exclusion_ledger
        or authority.get("admitted_record_identity_commitment_sha256")
        != expected_admitted_commitment
        or not isinstance(review, dict)
        or review.get("method") != SEED_QA_REVIEW_METHOD
        or review.get("all_question_answer_evidence_triplets_manually_inspected")
        is not True
        or review.get("lineage_treated_as_semantic_authority") is not False
        or review.get("automated_score_treated_as_semantic_authority") is not False
        or review.get("unresolved_rows") != 0
        or review.get("protected_evaluation_content_opened") is not False
    ):
        raise RuntimeError("seed QA semantic decision accounting changed")
    units = normalize_seed_rows(
        rows,
        tokenizer,
        authorized_resources,
        semantic_decisions=decisions,
    )
    if (
        len(units) != len(admitted)
        or sum(unit["budget_token_count"] for unit in units) != admitted_tokens
    ):
        raise RuntimeError("seed QA admitted-unit accounting changed")
    return units, _seed_authority_receipt(
        authority, authority_raw=authority_raw
    ), decisions


def normalize_seed_rows(
    rows: list[dict], tokenizer: Any, authorized_resources: dict[str, dict],
    *, semantic_decisions: dict[str, dict] | None = None,
) -> list[dict]:
    """Normalize seed QA only under exact, per-row semantic decisions.

    The source projection and its content hashes prove lineage, not answer
    correctness.  A caller must supply one manually sealed disposition for
    every source row.  Explicit exclusions contribute exactly zero training
    tokens; only ``pass`` decisions become units.
    """

    _require_unique(rows, "record_id", "seed QA")
    if not isinstance(semantic_decisions, dict) or set(semantic_decisions) != {
        row["record_id"] for row in rows
    }:
        raise RuntimeError("seed QA sealed semantic decision coverage is absent")
    result = []
    for row in rows:
        encoded = _validated_seed_encoding(row, tokenizer)
        decision = semantic_decisions[row["record_id"]]
        if (
            not isinstance(decision, dict)
            or decision.get("decision") not in {"pass", "exclude"}
            or decision.get("source_record_sha256")
            != compact_ascii_sha256(row)
            or not HEX64.fullmatch(
                str(decision.get("decision_content_sha256", ""))
            )
        ):
            raise RuntimeError("seed QA semantic decision is absent or unsealed")
        if decision["decision"] == "exclude":
            continue
        if any(decision.get(flag) is not True for flag in SEED_QA_PASS_FLAGS):
            raise RuntimeError("seed QA passing decision has an unproven flag")
        rights = authorize_rights(
            row.get("rights_basis"),
            resource_id=None,
            artifact_id=None,
            component="seed_qa",
            authorized_resources=authorized_resources,
            generated_rights_receipt=decision,
        )
        result.append(_unit(
            unit_id=row["record_id"],
            stream="domain_qa",
            training_format="chat_assistant_only",
            source_group_id=row["source_group_id"],
            source_document_id=row["document_sha256"],
            input_ids=encoded["input_ids"],
            labels=encoded["labels"],
            budget_token_count=encoded["assistant_token_count"],
            order_key=(row["record_id"],),
            metadata={
                "category": "precurated_seed_qa_opaque",
                "replay": False,
                "hard_negative": False,
                "verifier": {
                    "type": "sealed_seed_qa_semantic_authority_v1",
                    "status": "sealed_passed",
                    "semantic_correctness_verified": True,
                    "evidence_sha256": row["evidence_sha256"],
                    "authority_content_sha256": decision[
                        "decision_content_sha256"
                    ],
                },
                "generator": {
                    "type": "precurated_source_row",
                    "status": "generator_not_declared_by_source",
                },
                "rights": rights,
                "rights_status": row["rights_status"],
                "safety_transfer_flags": row["safety_transfer_flags"],
                "safety_transfer_status": row["safety_transfer_status"],
                "lineage": row["lineage"],
                "fact_id": row["fact_id"],
                "source_record_sha256": compact_ascii_sha256(row),
            },
        ))
    passed = [
        item for item in semantic_decisions.values()
        if item.get("decision") == "pass"
    ]
    if len(result) != len(passed):
        raise RuntimeError("seed QA accounting changed")
    return result


def _markdown_document_id(row: dict, component: str) -> str:
    if component == "core_markdown":
        return row.get("lineage", {}).get("source_document_identity_sha256")
    return row.get("source_document_identity_sha256")


def normalize_markdown_rows(
    rows: list[dict], tokenizer: Any, *, component: str,
    authorized_resources: dict[str, dict], expected_tokens: int,
) -> list[dict]:
    _require_unique(rows, "record_id", component)
    result = []
    schema = (
        "high-information-raw-continuation-v1"
        if component == "core_markdown"
        else "full-train-markdown-cpt-row-v1"
    )
    for row in rows:
        text = row.get("text")
        declared_count = row.get("qwen36_token_count")
        if (
            row.get("schema") != schema
            or row.get("assistant_supervision") is not False
            or row.get("training_format")
            not in {"causal_next_token_text", "causal_next_token_markdown"}
            or not isinstance(text, str) or not text
            or _contains_reserved(text)
            or sha256_bytes(text.encode("utf-8")) != row.get("text_sha256")
            or not isinstance(declared_count, int) or declared_count <= 0
        ):
            raise RuntimeError(f"{component}: Markdown row contract changed")
        input_ids = list(tokenizer.encode(text, add_special_tokens=False))
        if len(input_ids) != declared_count:
            raise RuntimeError(f"{component}: Markdown token count changed")
        document_id = _markdown_document_id(row, component)
        if not isinstance(document_id, str) or not document_id:
            raise RuntimeError(f"{component}: source document identity missing")
        rights = authorize_rights(
            row.get("rights_basis"),
            resource_id=row.get("resource_id"),
            artifact_id=row.get("artifact_id"),
            component=component,
            authorized_resources=authorized_resources,
        )
        order = (
            row.get("paragraph_index", -1), row["record_id"]
        ) if component == "core_markdown" else (
            row.get("byte_start", -1), row["record_id"]
        )
        result.append(_unit(
            unit_id=row["record_id"],
            stream="raw_markdown",
            training_format="causal_next_token",
            source_group_id=row["source_group_id"],
            source_document_id=document_id,
            input_ids=input_ids,
            labels=list(input_ids),
            budget_token_count=len(input_ids),
            order_key=order,
            metadata={
                "category": "raw_domain_continuation",
                "replay": False,
                "hard_negative": False,
                "verifier": {
                    "type": "sealed_text_and_token_hash_v1",
                    "status": "passed",
                    "text_sha256": row["text_sha256"],
                },
                "generator": {"type": "source_document", "status": "not_generated"},
                "rights": rights,
                "safety_transfer_flags": row["safety_transfer_flags"],
                "lineage": row["lineage"],
                "resource_id": row["resource_id"],
                "artifact_id": row["artifact_id"],
                "source_record_sha256": canonical_sha256(row),
            },
        ))
    if sum(item["budget_token_count"] for item in result) != expected_tokens:
        raise RuntimeError(f"{component}: exact token accounting changed")
    return result


def _validate_receipt_map(receipts: Any, role: str) -> None:
    if not isinstance(receipts, dict) or set(receipts) != GENERATED_RECEIPT_KEYS:
        raise RuntimeError(f"{role}: verifier receipt coverage changed")
    for name, receipt in receipts.items():
        if not isinstance(receipt, dict) or receipt.get("status") != "sealed_passed":
            raise RuntimeError(f"{role}: {name} is not sealed and passed")
        digest = receipt.get("content_sha256") or receipt.get("file_sha256")
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise RuntimeError(f"{role}: {name} has no content identity")


def _validate_seed_replacement_receipt(
    value: Any, seed_receipt: dict
) -> dict:
    expected = {
        "schema": SEED_QA_REPLACEMENT_SCHEMA,
        "seed_qa_semantic_authority_file_sha256": seed_receipt[
            "file_sha256"
        ],
        "seed_qa_semantic_authority_content_sha256": seed_receipt[
            "content_sha256"
        ],
        "replacement_assistant_tokens_required": seed_receipt[
            "replacement_generated_assistant_tokens_required"
        ],
        "replacement_assistant_tokens_selected": seed_receipt[
            "replacement_generated_assistant_tokens_required"
        ],
        "base_generated_assistant_tokens": BASE_GENERATED_ASSISTANT_TOKENS,
        "total_generated_assistant_tokens": (
            BASE_GENERATED_ASSISTANT_TOKENS
            + seed_receipt["replacement_generated_assistant_tokens_required"]
        ),
    }
    if value != expected:
        raise RuntimeError("generated seed-QA replacement receipt changed")
    return expected


def _validate_dynamic_generated_accounting(
    accounting: Any, *, replacement_tokens: int, selection_contract: dict
) -> dict:
    total = BASE_GENERATED_ASSISTANT_TOKENS + replacement_tokens
    bases = {
        "by_source": selection_contract.get("accepted_token_targets", {}).get(
            "source_tokens"
        ),
        "by_task_family": GENERATED_TASK_FAMILY_TOKENS,
        "by_category": GENERATED_CATEGORY_TOKENS,
        "by_generation_mode": GENERATED_MODE_TOKENS,
        "by_task_subtype": GENERATED_SUBTYPE_TOKENS,
    }
    if (
        not isinstance(accounting, dict)
        or set(accounting) != {"assistant_tokens", *bases}
        or accounting.get("assistant_tokens") != total
        or not isinstance(bases["by_source"], dict)
        or not bases["by_source"]
    ):
        raise RuntimeError("generated-domain token accounting changed")
    for name, base in bases.items():
        observed = accounting.get(name)
        if (
            not isinstance(observed, dict)
            or set(observed) != set(base)
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < base[key]
                for key, value in observed.items()
            )
            or sum(observed.values()) != total
            or sum(observed[key] - base[key] for key in base)
            != replacement_tokens
        ):
            raise RuntimeError(
                f"generated-domain {name} replacement accounting changed"
            )
    return accounting


def domain_qa_accounting_receipt(
    seed_receipt: dict, generated_receipt: dict
) -> dict:
    replacement = generated_receipt["seed_qa_replacement"]
    value = {
        "schema": "mixed-domain-qa-token-accounting-v1",
        "base_generated_assistant_tokens": replacement[
            "base_generated_assistant_tokens"
        ],
        "replacement_generated_assistant_tokens": replacement[
            "replacement_assistant_tokens_selected"
        ],
        "generated_assistant_tokens": replacement[
            "total_generated_assistant_tokens"
        ],
        "admitted_seed_assistant_tokens": seed_receipt[
            "assistant_qwen36_tokens"
        ],
        "excluded_seed_assistant_tokens": seed_receipt[
            "excluded_assistant_qwen36_tokens"
        ],
        "domain_qa_assistant_tokens": (
            replacement["total_generated_assistant_tokens"]
            + seed_receipt["assistant_qwen36_tokens"]
        ),
    }
    if (
        value["domain_qa_assistant_tokens"] != DOMAIN_QA_ASSISTANT_TOKENS
        or value["admitted_seed_assistant_tokens"]
        + value["excluded_seed_assistant_tokens"]
        != SEED_ASSISTANT_TOKENS
    ):
        raise RuntimeError("mixed domain-QA token accounting changed")
    return value


def load_generated_authority(
    tokenizer: Any,
    authorized_resources: dict[str, dict],
    seed_receipt: dict,
    selection_contract: dict,
) -> tuple[list[dict], dict]:
    """Load only the canonical final authority, never semantic shard outputs."""
    manifest_path = secure_regular_input(
        GENERATED_MANIFEST, "generated-domain manifest", exact=GENERATED_MANIFEST
    )
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    manifest_self = validate_self_hash(manifest, None, "generated-domain manifest")
    if (
        manifest.get("schema") != "high-information-generated-domain-authority-manifest-v1"
        or manifest.get("status") != "sealed_verified_training_authority"
        or manifest.get("eligible_for_training") is not True
        or manifest.get("training_launch_authorized_by_this_authority") is not True
        or manifest.get("selection_contract", {}).get("path") != relative(SELECTION_CONTRACT)
        or manifest.get("selection_contract", {}).get("file_sha256")
        != STATIC_SHA256[SELECTION_CONTRACT]
        or manifest.get("selection_contract", {}).get("content_sha256")
        != STATIC_SELF_SHA256[SELECTION_CONTRACT]
    ):
        raise RuntimeError("generated-domain authority is absent or unsealed")
    _validate_receipt_map(manifest.get("receipts"), "generated-domain manifest")
    replacement_receipt = _validate_seed_replacement_receipt(
        manifest.get("seed_qa_replacement"), seed_receipt
    )
    total_generated_tokens = replacement_receipt[
        "total_generated_assistant_tokens"
    ]
    dataset = manifest.get("dataset", {})
    report_receipt = manifest.get("report", {})
    if (
        dataset.get("path") != relative(GENERATED_DATA)
        or report_receipt.get("path") != relative(GENERATED_REPORT)
        or not HEX64.fullmatch(str(dataset.get("file_sha256", "")))
        or not HEX64.fullmatch(str(report_receipt.get("file_sha256", "")))
        or dataset.get("assistant_qwen36_tokens") != total_generated_tokens
        or dataset.get("schema") != "high-information-domain-training-row-v1"
    ):
        raise RuntimeError("generated-domain output receipt changed")
    report_raw = pinned_bytes(
        GENERATED_REPORT, report_receipt["file_sha256"], "generated-domain report"
    )
    report = json.loads(report_raw)
    report_self = validate_self_hash(report, report_receipt.get("content_sha256"),
                                     "generated-domain report")
    if (
        report.get("schema") != "high-information-generated-domain-authority-report-v1"
        or report.get("status") != "sealed_verified_training_authority"
        or report.get("all_rows_eligible_for_training") is not True
        or report.get("exact_atomic_selection_solved") is not True
        or report.get("deficit_tokens") != 0
    ):
        raise RuntimeError("generated-domain report is not a sealed exact authority")
    if _validate_seed_replacement_receipt(
        report.get("seed_qa_replacement"), seed_receipt
    ) != replacement_receipt:
        raise RuntimeError("generated report replacement receipt differs")
    raw = pinned_bytes(GENERATED_DATA, dataset["file_sha256"], "generated-domain rows")
    rows = load_jsonl_bytes(raw, "generated-domain rows")
    if len(rows) != dataset.get("rows"):
        raise RuntimeError("generated-domain row receipt changed")
    units = normalize_generated_rows(rows, tokenizer, authorized_resources)
    accounting = generated_accounting(units)
    expected = _validate_dynamic_generated_accounting(
        accounting,
        replacement_tokens=replacement_receipt[
            "replacement_assistant_tokens_selected"
        ],
        selection_contract=selection_contract,
    )
    if manifest.get("accounting") != expected or report.get("accounting") != expected:
        raise RuntimeError("generated-domain multidimensional accounting changed")
    return units, {
        "manifest_file_sha256": sha256_bytes(manifest_raw),
        "manifest_content_sha256": manifest_self,
        "report_file_sha256": sha256_bytes(report_raw),
        "report_content_sha256": report_self,
        "dataset_file_sha256": sha256_bytes(raw),
        "rows": len(rows),
        "accounting": accounting,
        "seed_qa_replacement": replacement_receipt,
    }


def normalize_generated_rows(
    rows: list[dict], tokenizer: Any, authorized_resources: dict[str, dict]
) -> list[dict]:
    _require_unique(rows, "record_id", "generated-domain authority")
    for key in (
        "candidate_example_id", "request_id", "source_context_id",
    ):
        _require_unique(rows, key, "generated-domain authority")
    exact_dedupe = set()
    result = []
    for row in rows:
        if not GENERATED_REQUIRED_ROW_KEYS.issubset(row):
            raise RuntimeError("generated-domain row omitted mandatory metadata")
        if (
            row["schema"] != "high-information-domain-training-row-v1"
            or row["split"] != "train"
            or row["training_format"] != "chat_assistant_only"
            or row["eligible_for_training"] is not True
            or row["task_family"] not in GENERATED_TASK_FAMILY_TOKENS
            or row["task_subtype"] not in GENERATED_SUBTYPE_TOKENS
            or row["generation_mode"] not in GENERATED_MODE_TOKENS
            or row["category"] not in GENERATED_CATEGORY_TOKENS
            or not isinstance(row["hard_negative"], bool)
            or row["hard_negative"]
            != (row["generation_mode"] == "calibrated_hard_negative")
            or row["messages"][-1].get("role") != "assistant"
            or row["messages"][-1].get("content") != row["answer"]
            or row["question"] not in [
                message.get("content") for message in row["messages"]
                if message.get("role") == "user"
            ]
            or _contains_reserved(row["messages"])
        ):
            raise RuntimeError("generated-domain row semantic contract changed")
        if (
            row["assistant_mask"] != {
                "policy": "assistant_only_v1",
                "assistant_message_indices": [len(row["messages"]) - 1],
                "system_tokens": False,
                "user_tokens": False,
                "tool_result_tokens": False,
            }
            or not isinstance(row["evidence_quote_sha256s"], list)
            or any(not HEX64.fullmatch(str(item)) for item in row["evidence_quote_sha256s"])
        ):
            raise RuntimeError("generated-domain mask or evidence receipt changed")
        verification = row["verification_receipts"]
        required_verification = {
            "structural", "nli", "semantic_judge_pass_1",
            "semantic_judge_pass_2", "selection",
        }
        if (
            not isinstance(verification, dict)
            or set(verification) != required_verification
            or any(item.get("status") != "passed" for item in verification.values())
        ):
            raise RuntimeError("generated-domain row verifier coverage changed")
        manual = row["manual_review_receipt"]
        if not isinstance(manual, dict) or manual.get("status") not in {
            "not_required", "resolved_pass",
        }:
            raise RuntimeError("generated-domain manual review is unresolved")
        dedupe = row["dedupe"]
        if (
            not isinstance(dedupe, dict)
            or not HEX64.fullmatch(str(dedupe.get("exact_key_sha256", "")))
            or not isinstance(dedupe.get("near_duplicate_cluster_id"), str)
            or not dedupe["near_duplicate_cluster_id"]
            or dedupe["exact_key_sha256"] in exact_dedupe
        ):
            raise RuntimeError("generated-domain dedupe contract changed")
        exact_dedupe.add(dedupe["exact_key_sha256"])
        encoded = encode_chat_assistant_only(
            tokenizer, row["messages"], enable_thinking=False, tools=row["tools"]
        )
        if encoded["assistant_token_count"] != row["assistant_qwen36_token_count"]:
            raise RuntimeError("generated-domain assistant token count changed")
        rights = authorize_rights(
            row["rights_basis"],
            resource_id=row["resource_id"],
            artifact_id=row.get("lineage", {}).get("artifact_id"),
            component="generated_domain",
            authorized_resources=authorized_resources,
            generated_rights_receipt=row["rights_authorization"],
        )
        result.append(_unit(
            unit_id=row["record_id"],
            stream="domain_qa",
            training_format="chat_assistant_only",
            source_group_id=row["source_group_id"],
            source_document_id=row["lineage"]["source_document_identity_sha256"],
            input_ids=encoded["input_ids"],
            labels=encoded["labels"],
            budget_token_count=encoded["assistant_token_count"],
            order_key=(row["request_id"], row["candidate_example_id"]),
            metadata={
                "category": row["category"],
                "task_family": row["task_family"],
                "task_subtype": row["task_subtype"],
                "generation_mode": row["generation_mode"],
                "replay": False,
                "hard_negative": row["hard_negative"],
                "verifier": verification,
                "manual_review_receipt": manual,
                "dedupe": dedupe,
                "generator": row["generator"],
                "rights": rights,
                "safety_transfer_flags": row["safety_transfer_flags"],
                "lineage": row["lineage"],
                "candidate_example_id": row["candidate_example_id"],
                "request_id": row["request_id"],
                "source_context_id": row["source_context_id"],
                "resource_id": row["resource_id"],
                "source_record_sha256": canonical_sha256(row),
            },
        ))
    return result


def generated_accounting(units: list[dict]) -> dict:
    source = Counter()
    task = Counter()
    category = Counter()
    mode = Counter()
    subtype = Counter()
    total = 0
    for unit in units:
        count = unit["budget_token_count"]
        metadata = unit["metadata"]
        source[metadata["resource_id"]] += count
        task[metadata["task_family"]] += count
        category[metadata["category"]] += count
        mode[metadata["generation_mode"]] += count
        subtype[metadata["task_subtype"]] += count
        total += count
    return {
        "assistant_tokens": total,
        "by_source": dict(sorted(source.items())),
        "by_task_family": dict(sorted(task.items())),
        "by_category": dict(sorted(category.items())),
        "by_generation_mode": dict(sorted(mode.items())),
        "by_task_subtype": dict(sorted(subtype.items())),
    }


def _fragment(unit: dict, start: int, stop: int) -> dict:
    labels = unit["labels"][start:stop]
    if unit["training_format"] == "causal_next_token":
        budget = stop - start
    else:
        if start or stop != len(unit["input_ids"]):
            raise RuntimeError("chat units may not be split")
        budget = sum(label != -100 for label in labels)
    return {
        "unit": unit,
        "source_token_start": start,
        "source_token_stop": stop,
        "input_ids": unit["input_ids"][start:stop],
        "labels": labels,
        "budget_token_count": budget,
    }


def pack_units(units: list[dict], *, max_tokens: int = MAX_SEQUENCE_TOKENS) -> list[dict]:
    """Greedily pack only identical stream/format/group/document keys."""
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 2:
        raise ValueError("packing cap must be at least two tokens")
    ids = [unit.get("unit_id") for unit in units]
    if len(set(ids)) != len(ids):
        raise RuntimeError("normalized training unit identity duplicated")
    ordered = sorted(units, key=lambda unit: (
        unit["stream"], unit["training_format"], unit["source_group_id"],
        unit["source_document_id"], unit["order_key"], unit["unit_id"],
    ))
    result = []

    def emit(fragments: list[dict]) -> None:
        if not fragments:
            return
        first = fragments[0]["unit"]
        keys = {
            (
                item["unit"]["stream"], item["unit"]["training_format"],
                item["unit"]["source_group_id"], item["unit"]["source_document_id"],
            )
            for item in fragments
        }
        if len(keys) != 1:
            raise RuntimeError("cross-source or cross-format packing attempted")
        input_ids = []
        labels = []
        segments = []
        cursor = 0
        for item in fragments:
            length = len(item["input_ids"])
            input_ids.extend(item["input_ids"])
            labels.extend(item["labels"])
            segments.append({
                "unit_id": item["unit"]["unit_id"],
                "token_start": cursor,
                "token_stop": cursor + length,
                "source_token_start": item["source_token_start"],
                "source_token_stop": item["source_token_stop"],
                "budget_token_count": item["budget_token_count"],
                "metadata": item["unit"]["metadata"],
            })
            cursor += length
        if not input_ids or len(input_ids) > max_tokens or len(input_ids) != len(labels):
            raise RuntimeError("packed sequence length contract failed")
        identity = {
            "stream": first["stream"],
            "training_format": first["training_format"],
            "source_group_id": first["source_group_id"],
            "source_document_id": first["source_document_id"],
            "input_ids_sha256": canonical_sha256(input_ids),
            "labels_sha256": canonical_sha256(labels),
            "segment_spans": [
                {
                    key: segment[key] for key in (
                        "unit_id", "token_start", "token_stop",
                        "source_token_start", "source_token_stop",
                    )
                }
                for segment in segments
            ],
        }
        sequence_id = "mixed-sequence-v1:" + canonical_sha256(identity)
        result.append({
            "schema": "mixed-training-packed-sequence-v1",
            "sequence_id": sequence_id,
            "stream": first["stream"],
            "training_format": first["training_format"],
            "label_semantics": (
                "official_qwen_chat_assistant_only_v1"
                if first["training_format"] == "chat_assistant_only"
                else "causal_next_token_all_tokens_v1"
            ),
            "source_group_id": first["source_group_id"],
            "source_document_id": first["source_document_id"],
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "input_token_count": len(input_ids),
            "budget_token_count": sum(
                segment["budget_token_count"] for segment in segments
            ),
            "shifted_supervised_token_count": sum(
                label != -100 for label in labels[1:]
            ),
            "segments": segments,
            "packing": {
                "policy": "same_stream_format_source_group_and_document_greedy_v1",
                "max_tokens": max_tokens,
                "cross_source_group": False,
                "cross_document": False,
            },
        })

    key_function = lambda unit: (
        unit["stream"], unit["training_format"], unit["source_group_id"],
        unit["source_document_id"],
    )
    for _, group_values in groupby(ordered, key=key_function):
        current: list[dict] = []
        current_tokens = 0
        for unit in group_values:
            length = len(unit["input_ids"])
            if unit["training_format"] == "chat_assistant_only":
                if length > max_tokens:
                    raise RuntimeError("chat row exceeds max length; truncation forbidden")
                if current and current_tokens + length > max_tokens:
                    emit(current)
                    current = []
                    current_tokens = 0
                current.append(_fragment(unit, 0, length))
                current_tokens += length
                continue
            source_cursor = 0
            while source_cursor < length:
                capacity = max_tokens - current_tokens
                take = min(capacity, length - source_cursor)
                current.append(_fragment(unit, source_cursor, source_cursor + take))
                current_tokens += take
                source_cursor += take
                if current_tokens == max_tokens:
                    emit(current)
                    current = []
                    current_tokens = 0
        emit(current)
    if len({row["sequence_id"] for row in result}) != len(result):
        raise RuntimeError("packed sequence identity duplicated")
    if sum(row["budget_token_count"] for row in result) != sum(
        unit["budget_token_count"] for unit in units
    ):
        raise RuntimeError("packing changed exact budget tokens")
    return result


def build_schedule(
    sequences: list[dict], *, variant: str, budgets: dict[str, int]
) -> tuple[list[dict], dict]:
    """Deterministically interleave streams and seal every resume cursor."""
    queues = {
        stream: sorted(
            [row for row in sequences if row["stream"] == stream],
            key=lambda row: row["sequence_id"],
        )
        for stream in budgets
    }
    observed = {
        stream: sum(row["budget_token_count"] for row in rows)
        for stream, rows in queues.items()
    }
    if observed != budgets:
        raise RuntimeError(f"{variant}: exact schedule budgets changed")
    sequence_receipts = {
        row["sequence_id"]: canonical_sha256(row) for row in sequences
    }
    sequence_set_identity = canonical_sha256({
        "variant": variant,
        "budgets": budgets,
        "sequence_receipts": dict(sorted(sequence_receipts.items())),
    })
    delivered = {stream: 0 for stream in budgets}
    positions = {stream: 0 for stream in budgets}
    previous = "0" * 64
    schedule = []
    stream_order = list(budgets)
    while any(positions[stream] < len(queues[stream]) for stream in stream_order):
        available = [
            stream for stream in stream_order
            if positions[stream] < len(queues[stream])
        ]
        # Lowest exact delivered fraction runs next; ties use declared stream order.
        stream = min(
            available,
            key=lambda item: (
                Fraction(delivered[item], budgets[item]), stream_order.index(item)
            ),
        )
        sequence = queues[stream][positions[stream]]
        positions[stream] += 1
        count = sequence["budget_token_count"]
        delivered[stream] += count
        base = {
            "schema": "mixed-training-schedule-cursor-v1",
            "cursor": len(schedule),
            "variant": variant,
            "sequence_id": sequence["sequence_id"],
            "sequence_sha256": sequence_receipts[sequence["sequence_id"]],
            "stream": stream,
            "budget_token_count": count,
            "cumulative_budget_tokens": sum(delivered.values()),
            "cumulative_stream_budget_tokens": dict(delivered),
            "previous_cursor_commitment_sha256": previous,
        }
        commitment = canonical_sha256(base)
        row = {**base, "cursor_commitment_sha256": commitment}
        schedule.append(row)
        previous = commitment
    if delivered != budgets or len({row["sequence_id"] for row in schedule}) != len(sequences):
        raise RuntimeError(f"{variant}: schedule coverage or accounting changed")
    return schedule, {
        "sequence_set_identity_sha256": sequence_set_identity,
        "initial_cursor_commitment_sha256": schedule[0][
            "previous_cursor_commitment_sha256"
        ],
        "final_cursor_commitment_sha256": previous,
        "cursor_count": len(schedule),
        "resume_identity": (
            "(sequence_set_identity_sha256,cursor,cursor_commitment_sha256)"
        ),
        "cursor_commitment_algorithm": (
            "sha256-canonical-schedule-row-without-current-commitment-v1"
        ),
        "cursor_commitment_formula": (
            "current=canonical_sha256(schedule_row after removing only "
            "cursor_commitment_sha256); first previous='0'*64"
        ),
    }


def _jsonl_payload(rows: Iterable[dict]) -> bytes:
    return b"".join(canonical_bytes(row) for row in rows)


def _json_payload(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def validate_static_manifests() -> tuple[dict[str, dict], dict]:
    full = load_pinned_json(FULL_MANIFEST, "full Markdown manifest")
    replay = load_pinned_json(REPLAY_MANIFEST, "replay authority manifest")
    authorization = load_pinned_json(PROJECT_AUTHORIZATION, "project authorization")
    selection = load_pinned_json(SELECTION_CONTRACT, "generated selection contract")
    if (
        full.get("schema") != "full-train-markdown-cpt-manifest-v1"
        or full.get("dataset", {}).get("file_sha256") != STATIC_SHA256[FULL_MARKDOWN]
        or full.get("dataset", {}).get("qwen36_tokens") != FULL_MARKDOWN_TOKENS
        or full.get("boundary", {}).get(
            "development_final_protected_holdout_ood_terminal_incident_or_manual_review_opened"
        ) is not False
        or replay.get("schema") != "general-replay-final-authority-manifest-v1"
        or replay.get("outputs", {}).get("rows", {}).get("sha256")
        != STATIC_SHA256[REPLAY_DATA]
        or replay.get("outputs", {}).get("rows", {}).get("assistant_tokens")
        != REPLAY_ASSISTANT_TOKENS
        or selection.get("accepted_token_targets", {}).get("task_family_tokens")
        != GENERATED_TASK_FAMILY_TOKENS
        or selection.get("token_accounting", {}).get(
            "category_balanced_generated_assistant_tokens"
        ) != GENERATED_ASSISTANT_TOKENS
        or selection.get("training_launch_authorized") is not False
    ):
        raise RuntimeError("static mixed-snapshot authority contract changed")
    return _authorized_resources(authorization), {
        "full_manifest": full,
        "replay_manifest": replay,
        "selection_contract": selection,
    }


def load_static_units(tokenizer: Any) -> tuple[dict[str, list[dict]], dict]:
    authorized_resources, manifests = validate_static_manifests()
    seed_rows = load_pinned_jsonl(SEED_QA, "seed QA")
    pending_seed_rows, pending_seed_receipt = pending_seed_qa_ledger(
        seed_rows, tokenizer
    )
    seed_units: list[dict] = []
    seed_receipt = None
    seed_decisions = None
    if SEED_QA_SEMANTIC_AUTHORITY.exists():
        seed_units, seed_receipt, seed_decisions = (
            load_seed_qa_semantic_authority(
                seed_rows, tokenizer, authorized_resources
            )
        )
    core_rows = load_pinned_jsonl(CORE_MARKDOWN, "core Markdown")
    full_rows = load_pinned_jsonl(FULL_MARKDOWN, "full Markdown")
    replay_rows = load_pinned_jsonl(REPLAY_DATA, "replay authority")
    units = {
        # The source projection proves lineage only.  Until the separate
        # authority and every per-row decision pass validation these rows
        # remain explicit pending exclusions and contribute zero tokens.
        "seed_qa": seed_units,
        "core_markdown": normalize_markdown_rows(
            core_rows, tokenizer, component="core_markdown",
            authorized_resources=authorized_resources,
            expected_tokens=CORE_MARKDOWN_TOKENS,
        ),
        "full_markdown": normalize_markdown_rows(
            full_rows, tokenizer, component="full_markdown",
            authorized_resources=authorized_resources,
            expected_tokens=FULL_MARKDOWN_TOKENS,
        ),
        "replay": normalize_replay_rows(replay_rows, tokenizer),
    }
    return units, {
        "authorized_resources": authorized_resources,
        "manifests": manifests,
        "pending_seed_qa_rows": pending_seed_rows,
        "pending_seed_qa_receipt": pending_seed_receipt,
        "seed_qa_semantic_gate": (
            {
                "status": "sealed_passed",
                "semantic_authority": seed_receipt,
                "training_rows_admitted": seed_receipt[
                    "training_rows_admitted"
                ],
                "excluded_rows": seed_receipt["excluded_rows"],
                "assistant_qwen36_tokens_admitted": seed_receipt[
                    "assistant_qwen36_tokens"
                ],
                "assistant_qwen36_tokens_excluded": seed_receipt[
                    "excluded_assistant_qwen36_tokens"
                ],
                "replacement_generated_assistant_tokens_required": seed_receipt[
                    "replacement_generated_assistant_tokens_required"
                ],
                "assistant_qwen36_token_shortfall": 0,
            }
            if seed_receipt is not None
            else {
                "status": "missing",
                "required_path": relative(SEED_QA_SEMANTIC_AUTHORITY),
                "required_schema": SEED_QA_SEMANTIC_AUTHORITY_SCHEMA,
                "pending_ledger": pending_seed_receipt,
                "training_rows_admitted": 0,
                "assistant_qwen36_token_shortfall": SEED_ASSISTANT_TOKENS,
            }
        ),
        "seed_qa_semantic_receipt": seed_receipt,
        "seed_qa_semantic_decisions": seed_decisions,
    }


def provisional_artifact(
    static_units: dict[str, list[dict]], seed_qa_gate: dict
) -> tuple[bytes, dict]:
    static_component_tokens = {
        name: sum(unit["budget_token_count"] for unit in units)
        for name, units in static_units.items()
    }
    static_packing = {}
    for variant, markdown_key in (
        ("protocol_core_100k", "core_markdown"),
        ("full_authorized_markdown", "full_markdown"),
    ):
        units = [
            *static_units["seed_qa"], *static_units[markdown_key],
            *static_units["replay"],
        ]
        packed = pack_units(units)
        by_stream = Counter()
        for row in packed:
            by_stream[row["stream"]] += row["budget_token_count"]
        admitted_seed_tokens = sum(
            unit["budget_token_count"] for unit in static_units["seed_qa"]
        )
        seed_shortfall = seed_qa_gate.get(
            "assistant_qwen36_token_shortfall", 0
        )
        generated_shortfall = (
            BASE_GENERATED_ASSISTANT_TOKENS
            if seed_qa_gate.get("status") != "sealed_passed"
            else DOMAIN_QA_ASSISTANT_TOKENS - admitted_seed_tokens
        )
        expected_static = {
            **(
                {"domain_qa": admitted_seed_tokens}
                if admitted_seed_tokens
                else {}
            ),
            "raw_markdown": (
                CORE_MARKDOWN_TOKENS
                if variant == "protocol_core_100k" else FULL_MARKDOWN_TOKENS
            ),
            "replay": REPLAY_ASSISTANT_TOKENS,
        }
        if dict(by_stream) != expected_static:
            raise RuntimeError("provisional static packing accounting changed")
        static_packing[variant] = {
            "packed_sequences": len(packed),
            "static_budget_tokens_by_stream": expected_static,
            "static_budget_tokens": sum(expected_static.values()),
            "target_budget_tokens": sum(VARIANT_BUDGETS[variant].values()),
            "generated_domain_shortfall_assistant_tokens": generated_shortfall,
            "base_generated_domain_shortfall_assistant_tokens": (
                BASE_GENERATED_ASSISTANT_TOKENS
            ),
            "seed_qa_replacement_shortfall_assistant_tokens": (
                seed_qa_gate.get(
                    "replacement_generated_assistant_tokens_required", 0
                )
            ),
            "seed_qa_semantic_shortfall_assistant_tokens": (
                seed_shortfall
            ),
            "total_domain_qa_shortfall_assistant_tokens": (
                generated_shortfall + seed_shortfall
            ),
            "training_sequences_emitted": False,
        }
    assembler_sha256 = file_sha256(ASSEMBLER)
    manifest = {
        "schema": "mixed-training-snapshot-provisional-gate-v1",
        "status": "blocked_missing_sealed_generated_domain_authority",
        "built_at": BUILD_TIME,
        "beads_task": "specialist-j59.7",
        "training_launch_authorized": False,
        "training_snapshot_materialized": False,
        "generated_domain_gate": {
            "status": "missing",
            "required_manifest_path": relative(GENERATED_MANIFEST),
            "required_report_path": relative(GENERATED_REPORT),
            "required_dataset_path": relative(GENERATED_DATA),
            "manifest_schema": "high-information-generated-domain-authority-manifest-v1",
            "report_schema": "high-information-generated-domain-authority-report-v1",
            "dataset_schema": "high-information-domain-training-row-v1",
            "required_assistant_tokens": (
                BASE_GENERATED_ASSISTANT_TOKENS
                if seed_qa_gate.get("status") != "sealed_passed"
                else DOMAIN_QA_ASSISTANT_TOKENS
                - sum(
                    unit["budget_token_count"]
                    for unit in static_units["seed_qa"]
                )
            ),
            "base_required_assistant_tokens": BASE_GENERATED_ASSISTANT_TOKENS,
            "seed_qa_replacement_assistant_tokens_required": seed_qa_gate.get(
                "replacement_generated_assistant_tokens_required", 0
            ),
            "required_task_family_tokens": GENERATED_TASK_FAMILY_TOKENS,
            "required_category_tokens": GENERATED_CATEGORY_TOKENS,
            "required_generation_mode_tokens": GENERATED_MODE_TOKENS,
            "required_task_subtype_tokens": GENERATED_SUBTYPE_TOKENS,
            "required_receipts": sorted(GENERATED_RECEIPT_KEYS),
            "atomic_rows_may_be_truncated_padded_borrowed_or_tolerated": False,
            "deficit_authority_required_if_exact_solution_absent": True,
        },
        "seed_qa_semantic_gate": seed_qa_gate,
        "static_inputs": [
            {"path": relative(path), "sha256": digest}
            for path, digest in sorted(STATIC_SHA256.items(), key=lambda item: str(item[0]))
        ],
        "static_component_accounting": static_component_tokens,
        "variants": static_packing,
        "packing_contract": {
            "max_tokens": MAX_SEQUENCE_TOKENS,
            "same_stream_format_source_group_and_document_only": True,
            "cross_source_group": False,
            "cross_document": False,
            "chat_label_semantics": "official_qwen_chat_assistant_only_v1",
            "markdown_label_semantics": "causal_next_token_all_tokens_v1",
        },
        "resume_contract": {
            "status": "implemented_but_not_materialized_until_gate_passes",
            "identity": "(sequence_set_identity_sha256,cursor,cursor_commitment_sha256)",
            "cursor_commitment_algorithm": (
                "sha256-canonical-schedule-row-without-current-commitment-v1"
            ),
            "first_previous_cursor_commitment_sha256": "0" * 64,
        },
        "assembler": {"path": relative(ASSEMBLER), "sha256": assembler_sha256},
        "boundary": {
            "development_final_protected_holdout_ood_terminal_incident_or_manual_review_opened": False,
            "semantic_candidate_shards_accepted_directly": False,
            "gpu_or_training_job_touched": False,
        },
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    return _json_payload(manifest), manifest


def seed_qa_semantic_provisional_artifact(
    static_units: dict[str, list[dict]],
    generated_units: list[dict],
    generated_receipt: dict,
    seed_qa_gate: dict,
) -> tuple[bytes, dict]:
    """Emit a deterministic blocker after generated rows pass but seeds do not."""

    if (
        seed_qa_gate.get("status") != "missing"
        or seed_qa_gate.get("training_rows_admitted") != 0
        or seed_qa_gate.get("assistant_qwen36_token_shortfall")
        != SEED_ASSISTANT_TOKENS
    ):
        raise RuntimeError("seed QA pending gate changed")
    variants = {}
    for variant, markdown_key in (
        ("protocol_core_100k", "core_markdown"),
        ("full_authorized_markdown", "full_markdown"),
    ):
        units = [
            *generated_units,
            *static_units[markdown_key],
            *static_units["replay"],
        ]
        packed = pack_units(units)
        observed = Counter()
        for row in packed:
            observed[row["stream"]] += row["budget_token_count"]
        expected_available = {
            "domain_qa": GENERATED_ASSISTANT_TOKENS,
            "raw_markdown": (
                CORE_MARKDOWN_TOKENS
                if variant == "protocol_core_100k" else FULL_MARKDOWN_TOKENS
            ),
            "replay": REPLAY_ASSISTANT_TOKENS,
        }
        if dict(observed) != expected_available:
            raise RuntimeError("seed QA provisional accounting changed")
        target = VARIANT_BUDGETS[variant]
        shortfall = {
            stream: target[stream] - expected_available[stream]
            for stream in target
        }
        if shortfall != {
            "domain_qa": SEED_ASSISTANT_TOKENS,
            "raw_markdown": 0,
            "replay": 0,
        }:
            raise RuntimeError("seed QA provisional shortfall changed")
        variants[variant] = {
            "packed_sequences_inspected_not_emitted": len(packed),
            "available_budget_tokens_by_stream": expected_available,
            "target_budget_tokens_by_stream": target,
            "shortfall_tokens_by_stream": shortfall,
            "training_sequences_emitted": False,
        }
    manifest = {
        "schema": "mixed-training-snapshot-provisional-gate-v1",
        "status": "blocked_missing_sealed_seed_qa_semantic_authority",
        "built_at": BUILD_TIME,
        "beads_task": "specialist-j59.7",
        "training_launch_authorized": False,
        "training_snapshot_materialized": False,
        "generated_domain_gate": {
            "status": "sealed_passed",
            **generated_receipt,
        },
        "seed_qa_semantic_gate": seed_qa_gate,
        "source_disjoint_gate": {
            "status": "not_evaluated_until_seed_qa_semantic_gate_passes",
            "required_path": relative(SOURCE_DISJOINT_EXTENSION),
        },
        "variants": variants,
        "packing_contract": {
            "max_sequence_length": MAX_SEQUENCE_TOKENS,
            "same_stream_format_source_group_and_document_only": True,
            "cross_source_group": False,
            "cross_document": False,
        },
        "assembler": {"path": relative(ASSEMBLER), "sha256": file_sha256(ASSEMBLER)},
        "boundary": {
            "development_final_protected_holdout_ood_terminal_incident_or_manual_review_opened": False,
            "semantic_candidate_shards_accepted_directly": False,
            "unverified_seed_qa_rows_admitted": False,
            "gpu_or_training_job_touched": False,
        },
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    return _json_payload(manifest), manifest


def rights_exclusion_receipt(units: list[dict]) -> dict:
    decisions = Counter()
    for unit in units:
        rights = unit.get("metadata", {}).get("rights")
        if not isinstance(rights, dict) or not isinstance(rights.get("decision"), str):
            raise RuntimeError("training unit lacks a resolved rights decision")
        decisions[rights["decision"]] += 1
    payload = {
        "schema": "mixed-training-rights-exclusion-receipt-v1",
        "status": "passed",
        "rows": len(units),
        "rights_decisions": dict(sorted(decisions.items())),
        "unresolved_or_unauthorized_rows": 0,
        "forbidden_source_classes_opened": False,
        "rejected_unreviewed_or_ineligible_rows_included": False,
        "source_rights_status_rewritten": False,
    }
    return {**payload, "content_sha256": canonical_sha256(payload)}


def validate_seed_qa_semantic_receipt(receipt: Any) -> dict:
    if (
        not isinstance(receipt, dict)
        or set(receipt) != SEED_QA_RECEIPT_KEYS
        or receipt.get("schema") != SEED_QA_SEMANTIC_AUTHORITY_SCHEMA
        or receipt.get("status") != "sealed_passed"
        or receipt.get("semantic_correctness_verified") is not True
        or receipt.get("eligible_for_training") is not True
        or receipt.get("path") != relative(SEED_QA_SEMANTIC_AUTHORITY)
        or receipt.get("source_rows") != SEED_QA_ROWS
        or receipt.get("reviewed_rows") != SEED_QA_ROWS
        or not isinstance(receipt.get("training_rows_admitted"), int)
        or isinstance(receipt.get("training_rows_admitted"), bool)
        or not isinstance(receipt.get("excluded_rows"), int)
        or isinstance(receipt.get("excluded_rows"), bool)
        or receipt["training_rows_admitted"] + receipt["excluded_rows"]
        != SEED_QA_ROWS
        or receipt["training_rows_admitted"] <= 0
        or receipt.get("source_assistant_qwen36_tokens")
        != SEED_ASSISTANT_TOKENS
        or not isinstance(receipt.get("assistant_qwen36_tokens"), int)
        or isinstance(receipt.get("assistant_qwen36_tokens"), bool)
        or not isinstance(
            receipt.get("excluded_assistant_qwen36_tokens"), int
        )
        or isinstance(receipt.get("excluded_assistant_qwen36_tokens"), bool)
        or receipt["assistant_qwen36_tokens"]
        + receipt["excluded_assistant_qwen36_tokens"]
        != SEED_ASSISTANT_TOKENS
        or receipt.get("replacement_generated_assistant_tokens_required")
        != receipt["excluded_assistant_qwen36_tokens"]
        or not HEX64.fullmatch(str(receipt.get("file_sha256", "")))
        or not HEX64.fullmatch(str(receipt.get("content_sha256", "")))
        or receipt.get("source_dataset")
        != {
            "path": relative(SEED_QA),
            "file_sha256": STATIC_SHA256[SEED_QA],
        }
        or not isinstance(receipt.get("decision_bundle"), dict)
        or receipt["decision_bundle"].get("path")
        != relative(SEED_QA_DECISION_BUNDLE)
        or not HEX64.fullmatch(
            str(receipt["decision_bundle"].get("file_sha256", ""))
        )
        or not HEX64.fullmatch(
            str(receipt.get("admitted_record_identity_commitment_sha256", ""))
        )
        or not isinstance(receipt.get("exclusion_ledger"), list)
        or len(receipt["exclusion_ledger"]) != receipt["excluded_rows"]
    ):
        raise RuntimeError("sealed seed QA semantic authority receipt is absent")
    return receipt


def _source_disjoint_static_input_paths() -> dict[str, Path]:
    """Return the fixed safe inputs bound by the opaque claim request."""

    paths = {
        "source_split_authority": SOURCE_SPLIT_AUTHORITY,
        "seed_qa_source": SEED_QA,
        "seed_qa_semantic_authority": SEED_QA_SEMANTIC_AUTHORITY,
        "seed_qa_decision_bundle": SEED_QA_DECISION_BUNDLE,
        "core_markdown": CORE_MARKDOWN,
        "full_markdown": FULL_MARKDOWN,
        "full_markdown_manifest": FULL_MANIFEST,
        "project_training_authorization": PROJECT_AUTHORIZATION,
        "replay_data": REPLAY_DATA,
        "replay_manifest": REPLAY_MANIFEST,
        "generated_selection_contract": SELECTION_CONTRACT,
        "generated_manifest": GENERATED_MANIFEST,
        "generated_report": GENERATED_REPORT,
        "generated_data": GENERATED_DATA,
    }
    if set(paths) != set(source_disjoint_claim.STATIC_INPUT_ROLES):
        raise RuntimeError("source-disjoint static input role set changed")
    for role, expected in source_disjoint_claim.PRODUCTION_ROLE_PATHS.items():
        if Path(paths[role]).resolve() != Path(expected).resolve():
            raise RuntimeError(
                f"source-disjoint production path changed for {role}"
            )
    return paths


def _source_split_authority_and_train_groups() -> tuple[dict, frozenset[str]]:
    """Read only the public train records and aggregate-only final receipt."""

    path = secure_regular_input(
        SOURCE_SPLIT_AUTHORITY,
        "source split authority",
        exact=SOURCE_SPLIT_AUTHORITY,
    )
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("source split authority is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("source split authority must be an object")
    content_sha256 = validate_self_hash(value, None, "source split authority")
    assignments = value.get("assignments")
    train = assignments.get("train") if isinstance(assignments, dict) else None
    final = assignments.get("final") if isinstance(assignments, dict) else None
    records = train.get("records") if isinstance(train, dict) else None
    if (
        value.get("schema") != "specialist-source-group-split-authority-v1"
        or value.get("status")
        != "sealed_source_disjoint_assignment_launch_still_gated"
        or not isinstance(records, list)
        or not records
        or not isinstance(final, dict)
        or final.get("records_redacted") is not True
        or "records" in final
    ):
        raise RuntimeError("source split authority contract changed")
    group_ids = [
        record.get("source_group_id")
        for record in records
        if isinstance(record, dict)
    ]
    if (
        len(group_ids) != len(records)
        or any(not isinstance(item, str) or not item for item in group_ids)
        or len(set(group_ids)) != len(group_ids)
    ):
        raise RuntimeError("source split train membership is invalid")
    train_commitment = train.get(
        "source_group_membership_commitment_sha256"
    )
    final_commitment = final.get(
        "source_group_membership_commitment_sha256"
    )
    if (
        train_commitment
        != source_disjoint_claim.canonical_sha256(sorted(group_ids))
        or not HEX64.fullmatch(str(final_commitment or ""))
    ):
        raise RuntimeError("source split membership commitment changed")
    return {
        "path": source_disjoint_claim.SOURCE_SPLIT_DECLARED_PATH,
        "file_sha256": sha256_bytes(raw),
        "content_sha256": content_sha256,
        "train_source_group_membership_commitment_sha256": train_commitment,
        "final_source_group_membership_commitment_sha256": final_commitment,
        "final_records_redacted": True,
    }, frozenset(group_ids)


def _independent_source_memberships() -> tuple[frozenset[str], frozenset[str]]:
    """Derive document/replay membership only from fixed train projections.

    Candidate units are intentionally not accepted as an argument.  This keeps
    a candidate-only source identity from authorizing itself by being inserted
    into the membership set passed to the claim runner.
    """

    seed_rows = load_pinned_jsonl(SEED_QA, "source-disjoint seed projection")
    core_rows = load_pinned_jsonl(
        CORE_MARKDOWN, "source-disjoint core Markdown projection"
    )
    full_rows = load_pinned_jsonl(
        FULL_MARKDOWN, "source-disjoint full Markdown projection"
    )
    replay_rows = load_pinned_jsonl(
        REPLAY_DATA, "source-disjoint replay authority"
    )
    documents: set[str] = set()
    for row in seed_rows:
        document = row.get("document_sha256")
        if (
            row.get("schema") != "high-information-seed-qa-v1"
            or not HEX64.fullmatch(str(document or ""))
        ):
            raise RuntimeError("seed train document membership changed")
        documents.add(document)
    for row in core_rows:
        document = row.get("lineage", {}).get(
            "source_document_identity_sha256"
        )
        if (
            row.get("schema") != "high-information-raw-continuation-v1"
            or not HEX64.fullmatch(str(document or ""))
        ):
            raise RuntimeError("core Markdown document membership changed")
        documents.add(document)
    for row in full_rows:
        document = row.get("source_document_identity_sha256")
        if (
            row.get("schema") != "full-train-markdown-cpt-row-v1"
            or row.get("split") != "train"
            or not HEX64.fullmatch(str(document or ""))
        ):
            raise RuntimeError("full Markdown document membership changed")
        documents.add(document)
    replay_groups = {
        row.get("source_group_id")
        for row in replay_rows
        if isinstance(row, dict)
    }
    if (
        not documents
        or not replay_groups
        or any(not isinstance(item, str) or not item for item in replay_groups)
    ):
        raise RuntimeError("independent source membership is empty or invalid")
    return frozenset(documents), frozenset(replay_groups)


def _source_disjoint_unit(unit: dict, *, component: str) -> dict:
    training_format = (
        "raw_markdown_causal"
        if component == "raw_markdown"
        else "chat_assistant_only"
    )
    return {
        "component": component,
        "unit_id": unit["unit_id"],
        "source_group_id": unit["source_group_id"],
        "source_document_id": unit["source_document_id"],
        "training_format": training_format,
        "input_ids": list(unit["input_ids"]),
        "labels": list(unit["labels"]),
        "budget_token_count": unit["budget_token_count"],
    }


def _source_disjoint_units_by_variant(
    static_units: dict[str, list[dict]], generated_units: list[dict]
) -> dict[str, list[dict]]:
    common = [
        *(
            _source_disjoint_unit(unit, component="seed_qa")
            for unit in static_units["seed_qa"]
        ),
        *(
            _source_disjoint_unit(unit, component="generated_domain")
            for unit in generated_units
        ),
    ]
    replay = [
        _source_disjoint_unit(unit, component="replay")
        for unit in static_units["replay"]
    ]
    return {
        "protocol_core_100k": [
            *common,
            *(
                _source_disjoint_unit(unit, component="raw_markdown")
                for unit in static_units["core_markdown"]
            ),
            *replay,
        ],
        "full_authorized_markdown": [
            *common,
            *(
                _source_disjoint_unit(unit, component="raw_markdown")
                for unit in static_units["full_markdown"]
            ),
            *replay,
        ],
    }


def _assert_independent_candidate_membership(
    units_by_variant: dict[str, list[dict]],
    *,
    train_source_group_ids: frozenset[str],
    train_source_document_ids: frozenset[str],
    replay_source_group_ids: frozenset[str],
) -> None:
    seen: set[tuple[str, str]] = set()
    for units in units_by_variant.values():
        for unit in units:
            identity = (unit["component"], unit["unit_id"])
            if identity in seen:
                continue
            seen.add(identity)
            if unit["component"] == "replay":
                if unit["source_group_id"] not in replay_source_group_ids:
                    raise RuntimeError(
                        "candidate replay group is absent from sealed replay rows"
                    )
                continue
            if unit["source_group_id"] not in train_source_group_ids:
                raise RuntimeError(
                    "candidate domain group is absent from the train split"
                )
            if unit["source_document_id"] not in train_source_document_ids:
                raise RuntimeError(
                    "candidate domain document is absent from fixed train projections"
                )


def _source_disjoint_context(
    static_units: dict[str, list[dict]],
    generated_units: list[dict],
    generated_receipt: dict,
    seed_qa_semantic_receipt: dict,
) -> dict:
    source_split, train_groups = _source_split_authority_and_train_groups()
    train_documents, replay_groups = _independent_source_memberships()
    units_by_variant = _source_disjoint_units_by_variant(
        static_units, generated_units
    )
    _assert_independent_candidate_membership(
        units_by_variant,
        train_source_group_ids=train_groups,
        train_source_document_ids=train_documents,
        replay_source_group_ids=replay_groups,
    )
    generated_authority = {
        "manifest_file_sha256": generated_receipt["manifest_file_sha256"],
        "manifest_content_sha256": generated_receipt[
            "manifest_content_sha256"
        ],
        "report_file_sha256": generated_receipt["report_file_sha256"],
        "report_content_sha256": generated_receipt["report_content_sha256"],
        "dataset_file_sha256": generated_receipt["dataset_file_sha256"],
        "seed_replacement_receipt_sha256": (
            source_disjoint_claim.canonical_sha256(
                generated_receipt["seed_qa_replacement"]
            )
        ),
    }
    seed_authority = {
        "authority_file_sha256": seed_qa_semantic_receipt["file_sha256"],
        "authority_content_sha256": seed_qa_semantic_receipt[
            "content_sha256"
        ],
        "decision_bundle_file_sha256": seed_qa_semantic_receipt[
            "decision_bundle"
        ]["file_sha256"],
        "admitted_record_identity_commitment_sha256": (
            seed_qa_semantic_receipt[
                "admitted_record_identity_commitment_sha256"
            ]
        ),
    }
    runner_path = ASSEMBLER.resolve()
    return {
        "static_input_paths": _source_disjoint_static_input_paths(),
        "expected_static_input_paths": dict(
            source_disjoint_claim.PRODUCTION_ROLE_PATHS
        ),
        "units_by_variant": units_by_variant,
        "source_split_authority": source_split,
        "generated_domain_authority": generated_authority,
        "seed_qa_semantic_authority": seed_authority,
        "runner_path": runner_path,
        "expected_runner_path": runner_path,
        "train_source_group_ids": train_groups,
        "train_source_document_ids": train_documents,
        "replay_source_group_ids": replay_groups,
    }


def build_source_disjoint_request_payload(
    static_units: dict[str, list[dict]],
    generated_units: list[dict],
    generated_receipt: dict,
    seed_qa_semantic_receipt: dict,
) -> tuple[dict, bytes]:
    context = _source_disjoint_context(
        static_units,
        generated_units,
        generated_receipt,
        seed_qa_semantic_receipt,
    )
    return source_disjoint_claim.build_claim_request(**context)


def build_source_disjoint_authorization_payload(
    static_units: dict[str, list[dict]],
    generated_units: list[dict],
    generated_receipt: dict,
    seed_qa_semantic_receipt: dict,
    *,
    expected_request_sha256: str,
) -> tuple[dict, bytes]:
    context = _source_disjoint_context(
        static_units,
        generated_units,
        generated_receipt,
        seed_qa_semantic_receipt,
    )
    return source_disjoint_claim.build_claim_authorization(
        request_path=SOURCE_DISJOINT_REQUEST,
        expected_request_path=SOURCE_DISJOINT_REQUEST,
        expected_request_sha256=expected_request_sha256,
        **context,
    )


def load_source_disjoint_extension(
    static_units: dict[str, list[dict]],
    generated_units: list[dict],
    generated_receipt: dict,
    seed_qa_semantic_receipt: dict,
    *,
    expected_request_sha256: str,
    expected_authorization_sha256: str,
) -> tuple[dict, bytes]:
    """Build the exact extension from externally pinned immutable parents."""

    context = _source_disjoint_context(
        static_units,
        generated_units,
        generated_receipt,
        seed_qa_semantic_receipt,
    )
    extension, payload = source_disjoint_claim.build_extension(
        request_path=SOURCE_DISJOINT_REQUEST,
        expected_request_path=SOURCE_DISJOINT_REQUEST,
        expected_request_sha256=expected_request_sha256,
        authorization_path=SOURCE_DISJOINT_AUTHORIZATION,
        expected_authorization_path=SOURCE_DISJOINT_AUTHORIZATION,
        expected_authorization_sha256=expected_authorization_sha256,
        **context,
    )
    receipt = {
        "schema": "mixed-training-source-disjoint-extension-receipt-v1",
        "status": "accepted",
        "accepted": True,
        "claim_request": {
            "path": relative(SOURCE_DISJOINT_REQUEST),
            "file_sha256": expected_request_sha256,
            "content_sha256": extension["claim_request"]["content_sha256"],
        },
        "claim_authorization": {
            "path": relative(SOURCE_DISJOINT_AUTHORIZATION),
            "file_sha256": expected_authorization_sha256,
            "content_sha256": extension["claim_authorization"][
                "content_sha256"
            ],
        },
        "extension": {
            "path": relative(SOURCE_DISJOINT_EXTENSION),
            "file_sha256": sha256_bytes(payload),
            "content_sha256": extension[
                "content_sha256_before_self_field"
            ],
        },
        "opaque_receipt_sha256": extension["opaque_collision_contract"][
            "opaque_receipt_sha256"
        ],
        "static_input_set_commitment_sha256": extension["static_inputs"][
            "bindings_commitment_sha256"
        ],
        "candidate_sets_commitment_sha256": (
            source_disjoint_claim.canonical_sha256(
                extension["candidate_sets"]
            )
        ),
    }
    return receipt, payload


def source_disjoint_provisional_artifact(
    static_units: dict[str, list[dict]], generated_units: list[dict],
    generated_receipt: dict, seed_qa_semantic_receipt: dict,
    *, request_pin_supplied: bool = False,
    authorization_pin_supplied: bool = False,
) -> tuple[bytes, dict]:
    variants = {}
    for variant, markdown_key in (
        ("protocol_core_100k", "core_markdown"),
        ("full_authorized_markdown", "full_markdown"),
    ):
        units = [
            *static_units["seed_qa"], *generated_units,
            *static_units[markdown_key], *static_units["replay"],
        ]
        packed = pack_units(units)
        observed = Counter()
        for row in packed:
            observed[row["stream"]] += row["budget_token_count"]
        if dict(observed) != VARIANT_BUDGETS[variant]:
            raise RuntimeError("source-disjoint provisional accounting changed")
        variants[variant] = {
            "packed_sequences": len(packed),
            "exact_budget_tokens_by_stream": VARIANT_BUDGETS[variant],
            "exact_budget_tokens": sum(VARIANT_BUDGETS[variant].values()),
            "training_sequences_emitted": False,
        }
    manifest = {
        "schema": "mixed-training-snapshot-provisional-gate-v1",
        "status": "blocked_missing_external_source_disjoint_contract_pins",
        "built_at": BUILD_TIME,
        "beads_task": "specialist-j59.7",
        "training_launch_authorized": False,
        "training_snapshot_materialized": False,
        "generated_domain_gate": {
            "status": "sealed_passed",
            **generated_receipt,
        },
        "source_disjoint_gate": {
            "status": "external_request_and_authorization_pins_required",
            "request_path": relative(SOURCE_DISJOINT_REQUEST),
            "authorization_path": relative(SOURCE_DISJOINT_AUTHORIZATION),
            "extension_path": relative(SOURCE_DISJOINT_EXTENSION),
            "required_schema": "mixed-training-source-disjoint-extension-v1",
            "request_pin_supplied": request_pin_supplied,
            "authorization_pin_supplied": authorization_pin_supplied,
            "request_or_authorization_opened": False,
            "expected_pins_derived_internally": False,
            "must_bind_generated_manifest_file_sha256": True,
            "must_bind_seed_qa_semantic_authority_file_sha256": True,
            "must_bind_seed_qa_semantic_authority_content_sha256": True,
            "generated_domain_manifest_file_sha256": generated_receipt[
                "manifest_file_sha256"
            ],
            "seed_qa_semantic_authority_file_sha256": (
                seed_qa_semantic_receipt["file_sha256"]
            ),
            "seed_qa_semantic_authority_content_sha256": (
                seed_qa_semantic_receipt["content_sha256"]
            ),
            "must_disclose_protected_identifiers_or_content": False,
        },
        "variants": variants,
        "packing_contract": {
            "max_sequence_length": MAX_SEQUENCE_TOKENS,
            "same_stream_format_source_group_and_document_only": True,
            "cross_source_group": False,
            "cross_document": False,
        },
        "assembler": {"path": relative(ASSEMBLER), "sha256": file_sha256(ASSEMBLER)},
        "boundary": {
            "development_final_protected_holdout_ood_terminal_incident_or_manual_review_opened": False,
            "semantic_candidate_shards_accepted_directly": False,
            "gpu_or_training_job_touched": False,
        },
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    return _json_payload(manifest), manifest


def build_variant(
    *, variant: str, seed_units: list[dict], generated_units: list[dict],
    markdown_units: list[dict], replay_units: list[dict], output_directory: Path,
    source_disjoint_receipt: dict, generated_receipt: dict,
    seed_qa_semantic_receipt: dict,
) -> tuple[dict[Path, bytes], dict]:
    if variant not in VARIANT_BUDGETS:
        raise ValueError("unknown mixed snapshot variant")
    units = [*seed_units, *generated_units, *markdown_units, *replay_units]
    unit_ids = [unit["unit_id"] for unit in units]
    if len(set(unit_ids)) != len(unit_ids):
        raise RuntimeError(f"{variant}: duplicate unit identity across components")
    sequences = pack_units(units)
    schedule, resume = build_schedule(
        sequences, variant=variant, budgets=VARIANT_BUDGETS[variant]
    )
    sequence_payload = _jsonl_payload(sequences)
    schedule_payload = _jsonl_payload(schedule)
    rights_receipt = rights_exclusion_receipt(units)
    seed_qa_semantic_receipt = validate_seed_qa_semantic_receipt(
        seed_qa_semantic_receipt
    )
    domain_qa_accounting = domain_qa_accounting_receipt(
        seed_qa_semantic_receipt, generated_receipt
    )
    if (
        sum(unit["budget_token_count"] for unit in seed_units)
        != domain_qa_accounting["admitted_seed_assistant_tokens"]
        or sum(unit["budget_token_count"] for unit in generated_units)
        != domain_qa_accounting["generated_assistant_tokens"]
    ):
        raise RuntimeError("variant domain-QA component accounting changed")
    manifest = {
        "schema": "mixed-training-snapshot-variant-manifest-v1",
        "status": "complete_launchable",
        "variant": variant,
        "max_sequence_length": MAX_SEQUENCE_TOKENS,
        "budget_tokens_by_stream": VARIANT_BUDGETS[variant],
        "budget_tokens": sum(VARIANT_BUDGETS[variant].values()),
        "domain_qa_accounting": domain_qa_accounting,
        "sequences": {
            "path": relative(output_directory / "sequences.jsonl"),
            "sha256": sha256_bytes(sequence_payload),
            "rows": len(sequences),
        },
        "schedule": {
            "path": relative(output_directory / "schedule.jsonl"),
            "sha256": sha256_bytes(schedule_payload),
            "rows": len(schedule),
        },
        "resume": resume,
        "sequence_set_identity_sha256": resume["sequence_set_identity_sha256"],
        "schedule_final_cursor_commitment_sha256": resume[
            "final_cursor_commitment_sha256"
        ],
        "label_semantics": {
            "chat": "official_qwen_chat_assistant_only_v1",
            "markdown": "causal_next_token_all_tokens_v1",
            "prompt_tokens_supervised": False,
        },
        "packing": {
            "policy": "same_stream_format_source_group_and_document_greedy_v1",
            "same_source_group_and_document_only": True,
            "cross_document": False,
            "cross_source_group": False,
        },
        "tokenizer": {
            "path": str(MODEL_DIRECTORY),
            "revision": MODEL_REVISION,
            "chat_template_sha256": MODEL_FILE_SHA256["chat_template.jinja"],
        },
        "gates": {
            "generated_domain_semantic_authority_passed": True,
            "generated_domain_receipt": generated_receipt,
            "seed_qa_semantic_authority_passed": True,
            "seed_qa_semantic_authority_receipt": seed_qa_semantic_receipt,
            "source_disjoint_extension_accepted": True,
            "source_disjoint_extension_receipt": source_disjoint_receipt,
            "rights_exclusion_gate_passed": True,
            "rights_exclusion_receipt": rights_receipt,
            "tokenizer_identity_passed": True,
            "exact_token_accounting_passed": True,
            "packing_invariants_passed": True,
        },
        "source_disjoint_extension_accepted": True,
        "seed_qa_semantic_authority_passed": True,
        "rights_exclusion_gate_passed": True,
        "training_launch_authorized": True,
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    return {
        output_directory / "sequences.jsonl": sequence_payload,
        output_directory / "schedule.jsonl": schedule_payload,
        output_directory / "manifest.json": _json_payload(manifest),
    }, manifest


def construct(
    *,
    expected_source_disjoint_request_sha256: str | None = None,
    expected_source_disjoint_authorization_sha256: str | None = None,
) -> tuple[dict[Path, bytes], dict]:
    validate_model_files(MODEL_DIRECTORY)
    tokenizer = load_qwen_tokenizer(
        str(MODEL_DIRECTORY), MODEL_REVISION,
        MODEL_FILE_SHA256["chat_template.jinja"],
    )
    static_units, static_authority = load_static_units(tokenizer)
    pending_seed_payload = _jsonl_payload(
        static_authority["pending_seed_qa_rows"]
    )
    if sha256_bytes(pending_seed_payload) != static_authority[
        "pending_seed_qa_receipt"
    ]["file_sha256"]:
        raise RuntimeError("pending seed QA ledger receipt changed")
    seed_gate = static_authority["seed_qa_semantic_gate"]
    base_payloads = (
        {PENDING_SEED_QA_LEDGER: pending_seed_payload}
        if seed_gate.get("status") != "sealed_passed"
        else {}
    )
    if not GENERATED_MANIFEST.exists():
        payload, manifest = provisional_artifact(
            static_units, seed_gate
        )
        return {**base_payloads, PROVISIONAL_MANIFEST: payload}, manifest
    if seed_gate.get("status") != "sealed_passed":
        # The generated authority is defined relative to the excluded seed
        # token count, so it cannot be authenticated before the seed review is
        # sealed.  Do not open it speculatively.
        payload, manifest = provisional_artifact(static_units, seed_gate)
        manifest["status"] = (
            "blocked_missing_sealed_seed_qa_semantic_authority"
        )
        manifest["generated_domain_gate"]["status"] = (
            "present_but_not_opened_until_seed_authority_is_sealed"
        )
        manifest["content_sha256_before_self_field"] = canonical_sha256({
            key: value
            for key, value in manifest.items()
            if key != "content_sha256_before_self_field"
        })
        return {
            **base_payloads,
            PROVISIONAL_MANIFEST: _json_payload(manifest),
        }, manifest
    authorized_resources = static_authority["authorized_resources"]
    generated_units, generated_receipt = load_generated_authority(
        tokenizer,
        authorized_resources,
        static_authority["seed_qa_semantic_receipt"],
        static_authority["manifests"]["selection_contract"],
    )
    if (
        sum(unit["budget_token_count"] for unit in generated_units)
        != generated_receipt["seed_qa_replacement"][
            "total_generated_assistant_tokens"
        ]
    ):
        raise RuntimeError("generated-domain exact token gate failed")
    if (
        expected_source_disjoint_request_sha256 is None
        or expected_source_disjoint_authorization_sha256 is None
    ):
        payload, manifest = source_disjoint_provisional_artifact(
            static_units,
            generated_units,
            generated_receipt,
            static_authority["seed_qa_semantic_receipt"],
            request_pin_supplied=(
                expected_source_disjoint_request_sha256 is not None
            ),
            authorization_pin_supplied=(
                expected_source_disjoint_authorization_sha256 is not None
            ),
        )
        return {**base_payloads, PROVISIONAL_MANIFEST: payload}, manifest
    source_disjoint_receipt, source_disjoint_extension_payload = (
        load_source_disjoint_extension(
            static_units,
            generated_units,
            generated_receipt,
            static_authority["seed_qa_semantic_receipt"],
            expected_request_sha256=(
                expected_source_disjoint_request_sha256
            ),
            expected_authorization_sha256=(
                expected_source_disjoint_authorization_sha256
            ),
        )
    )
    payloads = {
        SOURCE_DISJOINT_EXTENSION: source_disjoint_extension_payload,
    }
    variants = {}
    for variant, markdown_key in (
        ("protocol_core_100k", "core_markdown"),
        ("full_authorized_markdown", "full_markdown"),
    ):
        output = OUTPUT_DIRECTORY / variant
        built, manifest = build_variant(
            variant=variant,
            seed_units=static_units["seed_qa"],
            generated_units=generated_units,
            markdown_units=static_units[markdown_key],
            replay_units=static_units["replay"],
            output_directory=output,
            source_disjoint_receipt=source_disjoint_receipt,
            generated_receipt=generated_receipt,
            seed_qa_semantic_receipt=static_authority[
                "seed_qa_semantic_receipt"
            ],
        )
        payloads.update(built)
        variant_manifest_payload = built[output / "manifest.json"]
        variants[variant] = {
            "manifest_path": relative(output / "manifest.json"),
            "manifest_file_sha256": sha256_bytes(variant_manifest_payload),
            "manifest_content_sha256": manifest["content_sha256_before_self_field"],
            "budget_tokens": manifest["budget_tokens"],
            "sequence_file_sha256": manifest["sequences"]["sha256"],
            "sequence_rows": manifest["sequences"]["rows"],
            "schedule_file_sha256": manifest["schedule"]["sha256"],
            "schedule_rows": manifest["schedule"]["rows"],
            "sequence_set_identity_sha256": manifest[
                "sequence_set_identity_sha256"
            ],
            "schedule_final_cursor_commitment_sha256": manifest[
                "schedule_final_cursor_commitment_sha256"
            ],
            "exact_budget_tokens_by_stream": manifest[
                "budget_tokens_by_stream"
            ],
        }
    domain_qa_accounting = domain_qa_accounting_receipt(
        static_authority["seed_qa_semantic_receipt"], generated_receipt
    )
    top = {
        "schema": "mixed-training-snapshot-authority-manifest-v1",
        "status": "complete_launchable",
        "built_at": BUILD_TIME,
        "generated_domain_authority": generated_receipt,
        "seed_qa_semantic_authority": static_authority[
            "seed_qa_semantic_receipt"
        ],
        "domain_qa_accounting": domain_qa_accounting,
        "variants": variants,
        "tokenizer": {
            "path": str(MODEL_DIRECTORY),
            "revision": MODEL_REVISION,
            "chat_template_sha256": MODEL_FILE_SHA256["chat_template.jinja"],
        },
        "assembler": {"path": relative(ASSEMBLER), "sha256": file_sha256(ASSEMBLER)},
        "gates": {
            "generated_domain_semantic_authority_passed": True,
            "seed_qa_semantic_authority_passed": True,
            "source_disjoint_extension_accepted": True,
            "source_disjoint_extension_receipt": source_disjoint_receipt,
            "rights_exclusion_gate_passed": True,
            "tokenizer_identity_passed": True,
            "all_variant_manifests_launch_authorized": True,
        },
        "source_disjoint_extension_accepted": True,
        "seed_qa_semantic_authority_passed": True,
        "rights_exclusion_gate_passed": True,
        "max_sequence_length": MAX_SEQUENCE_TOKENS,
        "packing_invariants": {
            "same_stream_format_source_group_and_document_only": True,
            "cross_source_group": False,
            "cross_document": False,
        },
        "training_launch_authorized": True,
    }
    top["content_sha256_before_self_field"] = canonical_sha256(top)
    top_path = OUTPUT_DIRECTORY / "manifest.json"
    payloads[top_path] = _json_payload(top)
    return payloads, top


def _load_ready_source_disjoint_inputs() -> tuple[
    dict[str, list[dict]], list[dict], dict, dict
]:
    """Load the safe training side only for one separated claim stage."""

    validate_model_files(MODEL_DIRECTORY)
    tokenizer = load_qwen_tokenizer(
        str(MODEL_DIRECTORY),
        MODEL_REVISION,
        MODEL_FILE_SHA256["chat_template.jinja"],
    )
    static_units, static_authority = load_static_units(tokenizer)
    seed_gate = static_authority["seed_qa_semantic_gate"]
    if seed_gate.get("status") != "sealed_passed":
        raise RuntimeError(
            "source-disjoint contract emission requires the sealed seed QA authority"
        )
    if not GENERATED_MANIFEST.exists():
        raise RuntimeError(
            "source-disjoint contract emission requires the sealed generated authority"
        )
    generated_units, generated_receipt = load_generated_authority(
        tokenizer,
        static_authority["authorized_resources"],
        static_authority["seed_qa_semantic_receipt"],
        static_authority["manifests"]["selection_contract"],
    )
    if (
        sum(unit["budget_token_count"] for unit in generated_units)
        != generated_receipt["seed_qa_replacement"][
            "total_generated_assistant_tokens"
        ]
    ):
        raise RuntimeError("generated-domain exact token gate failed")
    return (
        static_units,
        generated_units,
        generated_receipt,
        static_authority["seed_qa_semantic_receipt"],
    )


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def emit_immutable_artifact(
    path: Path, payload: bytes, *, check: bool
) -> str:
    """Create once, or validate byte identity without replacing the artifact."""

    if path.exists():
        safe = secure_regular_input(
            path, f"immutable source-disjoint artifact {path.name}", exact=path
        )
        if safe.read_bytes() != payload:
            raise FileExistsError(
                f"immutable source-disjoint artifact differs: {relative(path)}"
            )
        return "validated_existing"
    if check:
        raise RuntimeError(
            f"immutable source-disjoint artifact is missing: {relative(path)}"
        )
    atomic_write(path, payload)
    return "created"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--check", action="store_true")
    result.add_argument("--require-sealed-generated-authority", action="store_true")
    stages = result.add_mutually_exclusive_group()
    stages.add_argument("--emit-source-disjoint-request", action="store_true")
    stages.add_argument(
        "--emit-source-disjoint-authorization", action="store_true"
    )
    result.add_argument("--source-disjoint-request-sha256")
    result.add_argument("--source-disjoint-authorization-sha256")
    return result


def blocker_shortfalls(manifest: dict) -> dict[str, int]:
    generated = (
        manifest.get("generated_domain_gate", {}).get(
            "required_assistant_tokens", GENERATED_ASSISTANT_TOKENS
        )
        if manifest.get("generated_domain_gate", {}).get("status") == "missing"
        else 0
    )
    seed = manifest.get("seed_qa_semantic_gate", {}).get(
        "assistant_qwen36_token_shortfall", 0
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (generated, seed)
    ):
        raise RuntimeError("provisional shortfall accounting is invalid")
    return {
        "generated_domain_shortfall_assistant_tokens": generated,
        "seed_qa_semantic_shortfall_assistant_tokens": seed,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.require_sealed_generated_authority and not GENERATED_MANIFEST.exists():
        raise RuntimeError(
            "sealed generated-domain authority missing; exact shortfall is "
            f"{GENERATED_ASSISTANT_TOKENS} assistant tokens"
        )
    if arguments.emit_source_disjoint_request:
        if (
            arguments.source_disjoint_request_sha256 is not None
            or arguments.source_disjoint_authorization_sha256 is not None
        ):
            raise ValueError(
                "request emission does not accept a derived request or authorization pin"
            )
        static, generated, generated_receipt, seed_receipt = (
            _load_ready_source_disjoint_inputs()
        )
        request, payload = build_source_disjoint_request_payload(
            static, generated, generated_receipt, seed_receipt
        )
        disposition = emit_immutable_artifact(
            SOURCE_DISJOINT_REQUEST, payload, check=arguments.check
        )
        print(json.dumps({
            "status": disposition,
            "stage": "source_disjoint_request",
            "path": relative(SOURCE_DISJOINT_REQUEST),
            "file_sha256": sha256_bytes(payload),
            "content_sha256": request["content_sha256_before_self_field"],
            "authorization_or_snapshot_emitted": False,
        }, sort_keys=True))
        return 0
    if arguments.emit_source_disjoint_authorization:
        if arguments.source_disjoint_request_sha256 is None:
            raise ValueError(
                "authorization emission requires an externally preregistered request SHA-256"
            )
        if arguments.source_disjoint_authorization_sha256 is not None:
            raise ValueError(
                "authorization emission cannot accept or derive its own authorization pin"
            )
        static, generated, generated_receipt, seed_receipt = (
            _load_ready_source_disjoint_inputs()
        )
        authorization, payload = build_source_disjoint_authorization_payload(
            static,
            generated,
            generated_receipt,
            seed_receipt,
            expected_request_sha256=(
                arguments.source_disjoint_request_sha256
            ),
        )
        disposition = emit_immutable_artifact(
            SOURCE_DISJOINT_AUTHORIZATION, payload, check=arguments.check
        )
        print(json.dumps({
            "status": disposition,
            "stage": "source_disjoint_authorization",
            "path": relative(SOURCE_DISJOINT_AUTHORIZATION),
            "file_sha256": sha256_bytes(payload),
            "content_sha256": authorization[
                "content_sha256_before_self_field"
            ],
            "request_file_sha256": arguments.source_disjoint_request_sha256,
            "extension_or_snapshot_emitted": False,
        }, sort_keys=True))
        return 0
    supplied_pins = (
        arguments.source_disjoint_request_sha256 is not None,
        arguments.source_disjoint_authorization_sha256 is not None,
    )
    if supplied_pins[0] != supplied_pins[1]:
        raise ValueError(
            "final build requires both external source-disjoint request and authorization SHA-256 pins"
        )
    payloads, manifest = construct(
        expected_source_disjoint_request_sha256=(
            arguments.source_disjoint_request_sha256
        ),
        expected_source_disjoint_authorization_sha256=(
            arguments.source_disjoint_authorization_sha256
        ),
    )
    if arguments.check:
        stale = []
        for path, payload in payloads.items():
            try:
                safe = secure_regular_input(path, f"mixed snapshot output {path.name}", exact=path)
            except (ValueError, RuntimeError):
                stale.append(relative(path))
                continue
            if safe.read_bytes() != payload:
                stale.append(relative(path))
        if stale:
            raise RuntimeError("mixed snapshot artifacts are stale: " + ", ".join(stale))
        return 0
    expected_paths = set(payloads)
    existing_known = {
        PROVISIONAL_MANIFEST,
        OUTPUT_DIRECTORY / "manifest.json",
        *(
            OUTPUT_DIRECTORY / variant / name
            for variant in VARIANT_BUDGETS
            for name in ("sequences.jsonl", "schedule.jsonl", "manifest.json")
        ),
    }
    stale_mode_outputs = [path for path in existing_known - expected_paths if path.exists()]
    if stale_mode_outputs:
        raise RuntimeError(
            "opposite-mode mixed snapshot outputs exist; remove only after explicit audit: "
            + ", ".join(relative(path) for path in stale_mode_outputs)
        )
    immutable_outputs = {SOURCE_DISJOINT_EXTENSION} & expected_paths
    for path in immutable_outputs:
        if path.exists():
            safe = secure_regular_input(
                path,
                f"immutable source-disjoint artifact {path.name}",
                exact=path,
            )
            if safe.read_bytes() != payloads[path]:
                raise FileExistsError(
                    "immutable source-disjoint extension differs from the "
                    "externally pinned request/authorization chain"
                )
    mutable_outputs = expected_paths - immutable_outputs
    if any(path.exists() for path in mutable_outputs):
        raise FileExistsError("mixed snapshot build requires fresh outputs")
    for path in sorted(expected_paths):
        if path in immutable_outputs and path.exists():
            continue
        atomic_write(path, payloads[path])
    print(json.dumps({
        "status": manifest["status"],
        "outputs": [relative(path) for path in sorted(payloads)],
        "training_launch_authorized": manifest["training_launch_authorized"],
        **blocker_shortfalls(manifest),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
