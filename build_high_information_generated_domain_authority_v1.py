#!/usr/bin/env python3
"""Build the fail-closed global generated-domain selection authority.

Primary and quality-fill semantic lanes remain separate until this builder has
validated every structural, NLI, semantic-output, report, and receipt lineage.
It then applies one global exact/near-duplicate policy and a deterministic
atomic-token MILP under simultaneous source, category, family, subtype, and
generation-mode caps.  A second atomic selection replaces every token removed
by independent manual seed-QA review.  If either exact selection does not
exist, the builder emits exact deficits and no training rows.  It never pads,
truncates, borrows, or treats an unresolved manual-review row as eligible.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

import build_category_balanced_selection_contract_v1 as balance
import build_high_information_domain_corpus_v1 as corpus
import build_seed_qa_semantic_authority_v1 as seed_authority
import run_high_information_fill_nli_prefilter_v1 as fill_nli
import run_high_information_fill_semantic_judge_v1 as fill_judge
import run_high_information_nli_prefilter_v1 as primary_nli
import run_high_information_semantic_judge_shard_v1 as primary_judge
import verify_high_information_candidates_v1 as structural
import verify_high_information_generation_pass_v1 as generation_pass
from qwen_chat_masking_v1 import encode_chat_assistant_only


AUTHORITY_DIRECTORY = corpus.OUTPUT_DIR / "generated_domain_authority_v1"
TRAIN_OUTPUT = AUTHORITY_DIRECTORY / "train.jsonl"
REPORT_OUTPUT = AUTHORITY_DIRECTORY / "report.json"
MANIFEST_OUTPUT = AUTHORITY_DIRECTORY / "manifest.json"
DEFICIT_OUTPUT = AUTHORITY_DIRECTORY / "deficit_report.json"
MANUAL_QUEUE_OUTPUT = AUTHORITY_DIRECTORY / "manual_review_queue_v1.jsonl"
MANUAL_RESOLUTIONS = AUTHORITY_DIRECTORY / "manual_review_resolutions_v1.jsonl"
SELECTION_SCAFFOLD = (
    corpus.OUTPUT_DIR / "generated_domain_selection_authority_v1.scaffold.json"
)
SELECTION_CONTRACT = balance.OUTPUT

ROW_SCHEMA = "high-information-domain-training-row-v1"
REPORT_SCHEMA = "high-information-generated-domain-authority-report-v1"
MANIFEST_SCHEMA = "high-information-generated-domain-authority-manifest-v1"
DEFICIT_SCHEMA = "high-information-generated-domain-deficit-report-v1"
SCAFFOLD_SCHEMA = "high-information-generated-domain-selection-scaffold-v1"
MANUAL_QUEUE_SCHEMA = "high-information-selection-manual-review-queue-row-v1"
MANUAL_RESOLUTION_SCHEMA = (
    "high-information-selection-manual-review-resolution-v1"
)
MANUAL_REVIEW_PROTOCOL = {
    "schema": "high-information-selection-manual-review-protocol-v1",
    "allowed_resolutions": ["resolved_pass", "resolved_reject"],
    "candidate_identity_fields": [
        "selection_candidate_id",
        "candidate_example_id",
        "semantic_record_sha256",
        "queue_row_sha256",
    ],
    "reviewer_identity_required": True,
    "free_form_reasoning_persisted": False,
    "nonconsensus_selected_unresolved": False,
    "judge_required_review_selected_unresolved": False,
}
MANUAL_REVIEW_PROTOCOL_SHA256 = (
    "6f844b610ee212b6411ebb3f398fd29fcdc4a66d9c1aaf66996699e38cf9448c"
)

SELECTION_CONTRACT_FILE_SHA256 = (
    "781580b2e968952cb55498ba5f609c473994fed5fbf5165688e3843dfc4e8784"
)
SELECTION_CONTRACT_SELF_SHA256 = (
    "fdfeedcd2fd46f02506a65e27bee50ab6e6995cb7f0233256af2a2913e23b07f"
)
PRIMARY_JUDGE_FILE_SHA256 = fill_judge.PRIMARY_JUDGE_FILE_SHA256
FILL_JUDGE_FILE_SHA256 = (
    "cef5c00d55cfe153dc54a5a5209b83982a182b224a5d5b6001d29ebfd2469b2d"
)
PRIMARY_NLI_FILE_SHA256 = fill_judge.PRIMARY_NLI_FILE_SHA256
FILL_NLI_FILE_SHA256 = fill_judge.FILL_NLI_FILE_SHA256
STRUCTURAL_FILE_SHA256 = (
    "30a85ebafcfca4cc9064d938e134c31948873ee2df90bbd38f6f88663ecac943"
)
CANDIDATE_GROUPING = fill_judge.CANDIDATE_GROUPING

EXACT_KEY_ALGORITHM = "sha256-nfkc-casefold-whitespace-question-nul-answer-v1"
NEAR_CLUSTER_ALGORITHM = (
    "source-group-aware-three-token-shingle-jaccard-0.82-v1"
)
NEAR_JACCARD_THRESHOLD = 0.82
MAX_COMMON_SHINGLE_POSTING = 128
OTHER_ACCEPT_REVIEW_MODULUS = 100
OTHER_ACCEPT_REVIEW_BUCKET = 0
MAX_RENDERED_CHAT_TOKENS = 2_048
TOKEN_LENGTH_GATE_SCHEMA = "high-information-rendered-chat-length-gate-v1"
TOKEN_LENGTH_RECEIPT_SCHEMA = (
    "high-information-rendered-chat-length-candidate-receipt-v1"
)
SEED_REPLACEMENT_CONTRACT_SCHEMA = "seed-qa-generated-replacement-contract-v1"
SEED_REPLACEMENT_SCHEMA = "seed-qa-generated-replacement-receipt-v1"
SEED_REPLACEMENT_DEFICIT_SCHEMA = (
    "seed-qa-generated-replacement-deficit-receipt-v1"
)
SEED_REPLACEMENT_RECEIPT_KEYS = {
    "schema",
    "seed_qa_semantic_authority_file_sha256",
    "seed_qa_semantic_authority_content_sha256",
    "replacement_assistant_tokens_required",
    "replacement_assistant_tokens_selected",
    "base_generated_assistant_tokens",
    "total_generated_assistant_tokens",
}
BASE_GENERATED_ASSISTANT_TOKENS = 740_847
DOMAIN_QA_ASSISTANT_TOKENS = 750_000
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)

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
    "schema",
    "record_id",
    "candidate_example_id",
    "request_id",
    "source_context_id",
    "source_group_id",
    "resource_id",
    "split",
    "training_format",
    "messages",
    "tools",
    "assistant_mask",
    "assistant_qwen36_token_count",
    "task_family",
    "task_subtype",
    "generation_mode",
    "category",
    "hard_negative",
    "question",
    "answer",
    "evidence_quote_sha256s",
    "verification_receipts",
    "manual_review_receipt",
    "dedupe",
    "generator",
    "rights_basis",
    "rights_authorization",
    "safety_transfer_flags",
    "lineage",
    "eligible_for_training",
}


def _self_address(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("content_sha256_before_self_field", None)
    return corpus.canonical_sha256(unsigned)


def _require_self_address(value: dict, label: str) -> None:
    if value.get("content_sha256_before_self_field") != _self_address(value):
        raise RuntimeError(f"{label} content address changed")


def _regular_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"{label} must be a non-symlink regular file")
    return resolved


def _json_payload(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sealed_receipt(value: Any) -> dict[str, Any]:
    return {
        "status": "sealed_passed",
        "content_sha256": corpus.canonical_sha256(value),
    }


def load_seed_qa_replacement_contract() -> dict[str, Any]:
    """Recompute and bind the complete independent seed-QA authority.

    The generated selector may trust neither an authority's token counters nor
    its self hash in isolation.  Rebuilding it from the pinned seed source,
    exact assignments, and every independently authored decision makes the
    replacement target a derived fact.
    """

    authority_path = seed_authority._secure_regular_file(
        seed_authority.AUTHORITY, "seed-QA semantic authority"
    )
    authority_raw = authority_path.read_bytes()
    observed = json.loads(authority_raw)
    seed_authority._require(
        isinstance(observed, dict), "seed-QA semantic authority is not an object"
    )
    rows = seed_authority.load_source()
    assignments = seed_authority._load_assignments(rows)
    decisions, file_receipts = seed_authority.load_decisions(rows, assignments)
    bundle, expected = seed_authority.build_authority(
        rows, assignments, decisions, file_receipts
    )
    seed_authority._require(
        observed == expected, "seed-QA semantic authority is stale or forged"
    )
    bundle_path = seed_authority._secure_regular_file(
        seed_authority.DECISION_BUNDLE, "seed-QA semantic decision bundle"
    )
    seed_authority._require(
        bundle_path.read_bytes() == bundle,
        "seed-QA semantic decision bundle is stale",
    )
    required = expected["replacement_generated_assistant_tokens_required"]
    admitted = expected["assistant_qwen36_tokens"]
    seed_authority._require(
        admitted + BASE_GENERATED_ASSISTANT_TOKENS + required
        == DOMAIN_QA_ASSISTANT_TOKENS,
        "seed-QA replacement token accounting changed",
    )
    result = {
        "schema": SEED_REPLACEMENT_CONTRACT_SCHEMA,
        "seed_qa_semantic_authority_file_sha256": corpus.sha256_bytes(
            authority_raw
        ),
        "seed_qa_semantic_authority_content_sha256": expected[
            "content_sha256_before_self_field"
        ],
        "replacement_assistant_tokens_required": required,
        "base_generated_assistant_tokens": BASE_GENERATED_ASSISTANT_TOKENS,
        "total_generated_assistant_tokens_required": (
            BASE_GENERATED_ASSISTANT_TOKENS + required
        ),
    }
    return validate_seed_qa_replacement_contract(result)


def validate_seed_qa_replacement_contract(value: Any) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "seed_qa_semantic_authority_file_sha256",
        "seed_qa_semantic_authority_content_sha256",
        "replacement_assistant_tokens_required",
        "base_generated_assistant_tokens",
        "total_generated_assistant_tokens_required",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError("seed-QA replacement contract fields changed")
    required = value.get("replacement_assistant_tokens_required")
    if (
        value.get("schema") != SEED_REPLACEMENT_CONTRACT_SCHEMA
        or not HEX64.fullmatch(
            str(value.get("seed_qa_semantic_authority_file_sha256", ""))
        )
        or not HEX64.fullmatch(
            str(value.get("seed_qa_semantic_authority_content_sha256", ""))
        )
        or type(required) is not int
        or required < 0
        or value.get("base_generated_assistant_tokens")
        != BASE_GENERATED_ASSISTANT_TOKENS
        or value.get("total_generated_assistant_tokens_required")
        != BASE_GENERATED_ASSISTANT_TOKENS + required
    ):
        raise RuntimeError("seed-QA replacement contract changed")
    return value


def _implementation_receipts() -> list[dict[str, str]]:
    expected = {
        Path(primary_judge.__file__).resolve(): PRIMARY_JUDGE_FILE_SHA256,
        Path(fill_judge.__file__).resolve(): FILL_JUDGE_FILE_SHA256,
        Path(primary_nli.__file__).resolve(): PRIMARY_NLI_FILE_SHA256,
        Path(fill_nli.__file__).resolve(): FILL_NLI_FILE_SHA256,
        Path(structural.__file__).resolve(): STRUCTURAL_FILE_SHA256,
    }
    receipts = []
    for path, digest in expected.items():
        observed = corpus.file_sha256(path)
        if observed != digest:
            raise RuntimeError(f"generated-domain implementation changed: {path.name}")
        receipts.append({"path": corpus.relative(path), "file_sha256": observed})
    builder = Path(__file__).resolve()
    receipts.append({
        "path": corpus.relative(builder),
        "file_sha256": corpus.file_sha256(builder),
    })
    return receipts


def load_selection_contract() -> dict:
    path = _regular_file(SELECTION_CONTRACT, "category-balanced selection contract")
    if corpus.file_sha256(path) != SELECTION_CONTRACT_FILE_SHA256:
        raise RuntimeError("category-balanced selection contract file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require_self_address(value, "category-balanced selection contract")
    if (
        value.get("schema") != "category-balanced-candidate-selection-contract-v1"
        or value.get("content_sha256_before_self_field")
        != SELECTION_CONTRACT_SELF_SHA256
        or value.get("selection_materialized") is not False
        or value.get("training_launch_authorized") is not False
    ):
        raise RuntimeError("category-balanced selection semantics changed")
    return value


def selection_targets(contract: dict) -> dict[str, Any]:
    accepted = contract["accepted_token_targets"]
    targets = {
        "assistant_tokens": contract["token_accounting"][
            "category_balanced_generated_assistant_tokens"
        ],
        "by_source": dict(accepted["source_tokens"]),
        "by_category": dict(accepted["category_tokens"]),
        "by_task_family": dict(accepted["task_family_tokens"]),
        "by_task_subtype": dict(accepted["task_subtype_tokens"]),
        "by_generation_mode": dict(accepted["generation_mode_tokens"]),
    }
    validate_targets(targets)
    return targets


def validate_targets(targets: Mapping[str, Any]) -> None:
    total = targets.get("assistant_tokens")
    if not isinstance(total, int) or total <= 0:
        raise RuntimeError("generated-domain total target changed")
    for axis in (
        "by_source",
        "by_category",
        "by_task_family",
        "by_task_subtype",
        "by_generation_mode",
    ):
        values = targets.get(axis)
        if (
            not isinstance(values, dict)
            or not values
            or any(not isinstance(key, str) or not key for key in values)
            or any(not isinstance(value, int) or value < 0 for value in values.values())
            or sum(values.values()) != total
        ):
            raise RuntimeError(f"generated-domain target axis changed: {axis}")


def source_categories(contract: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    targets = contract["accepted_token_targets"]["source_tokens"]
    for category, sources in balance.CATEGORY_SOURCE_TARGETS.items():
        for source in sources:
            if source in result:
                raise RuntimeError("selection source belongs to multiple categories")
            result[source] = category
    if set(result) != set(targets):
        raise RuntimeError("selection source/category map changed")
    return result


def semantic_input_states() -> list[dict[str, Any]]:
    states = []
    for lane in ("primary", "fill"):
        for shard in range(4):
            if lane == "primary":
                paths = primary_judge.output_paths(shard, smoke=False)
                required = ("output", "report")
            else:
                paths = fill_judge.output_paths(shard, smoke=False)
                required = ("output", "report", "receipt")
            artifacts = {
                name: {
                    "path": corpus.relative(paths[name]),
                    "regular_non_symlink_file_present": (
                        not paths[name].is_symlink() and paths[name].is_file()
                    ),
                }
                for name in required
            }
            states.append({
                "lane": lane,
                "gpu_shard": shard,
                "artifacts": artifacts,
                "sealed_set_present": all(
                    item["regular_non_symlink_file_present"]
                    for item in artifacts.values()
                ),
            })
    return states


def build_scaffold() -> dict:
    if corpus.canonical_sha256(MANUAL_REVIEW_PROTOCOL) != MANUAL_REVIEW_PROTOCOL_SHA256:
        raise RuntimeError("manual-review protocol changed")
    contract = load_selection_contract()
    targets = selection_targets(contract)
    states = semantic_input_states()
    blockers = [
        f"missing_{state['lane']}_semantic_seal:gpu{state['gpu_shard']}"
        for state in states
        if not state["sealed_set_present"]
    ]
    seed_replacement = None
    if seed_authority.AUTHORITY.is_symlink():
        raise RuntimeError("seed-QA semantic authority may not be a symlink")
    if not seed_authority.AUTHORITY.is_file():
        blockers.append("missing_sealed_seed_qa_semantic_authority")
    else:
        seed_replacement = load_seed_qa_replacement_contract()
    value = {
        "schema": SCAFFOLD_SCHEMA,
        "status": (
            "ready_for_receipt_validation_and_global_selection"
            if not blockers
            else "blocked_pending_primary_and_fill_semantic_judges"
        ),
        "selection_may_run": not blockers,
        "blockers": blockers,
        "semantic_input_states": states,
        "selection_contract": {
            "path": corpus.relative(SELECTION_CONTRACT),
            "file_sha256": SELECTION_CONTRACT_FILE_SHA256,
            "content_sha256": SELECTION_CONTRACT_SELF_SHA256,
        },
        "targets": targets,
        "seed_qa_replacement": {
            "required_authority_path": corpus.relative(seed_authority.AUTHORITY),
            "required_authority_schema": seed_authority.AUTHORITY_SCHEMA,
            "contract": seed_replacement,
            "replacement_selection_is_atomic_and_disjoint_from_base": True,
        },
        "deduplication": {
            "exact_key_algorithm": EXACT_KEY_ALGORITHM,
            "near_cluster_algorithm": NEAR_CLUSTER_ALGORITHM,
            "near_jaccard_threshold": NEAR_JACCARD_THRESHOLD,
            "max_common_shingle_posting": MAX_COMMON_SHINGLE_POSTING,
            "same_fact_multiple_views_require_different_task_and_information": True,
        },
        "selection": {
            "solver": "scipy.optimize.milp_highs_binary_two_stage_v1",
            "stage_1": "maximize_atomic_assistant_tokens_under_every_axis_cap",
            "stage_2": "hold_stage_1_total_and_maximize_deterministic_quality_priority",
            "seed_replacement_strategy": (
                "sequential_exact_atomic_selection_then_"
                "coupled_2n_lexicographic_fallback_v1"
            ),
            "coupled_fallback_objectives": [
                "maximize_exact_base_tokens",
                "hold_base_and_maximize_replacement_tokens",
                "hold_both_totals_and_maximize_total_quality",
            ],
            "coupled_fallback_dedupe_scope": (
                "exact_near_and_request_groups_across_base_and_replacement"
            ),
            "maximum_total_rendered_chat_tokens": MAX_RENDERED_CHAT_TOKENS,
            "rendered_chat_length_gate_receipt_schema": TOKEN_LENGTH_GATE_SCHEMA,
            "overlength_candidates_may_enter_selector": False,
            "overlength_factual_text_may_be_rewritten_to_pass": False,
            "padding_allowed": False,
            "truncation_allowed": False,
            "cross_source_borrowing_allowed": False,
            "cross_category_borrowing_allowed": False,
            "overshoot_or_tolerance_reported_as_exact": False,
            "exact_per_axis_deficits_required_when_unsolved": True,
        },
        "manual_review": {
            "resolution_path": corpus.relative(MANUAL_RESOLUTIONS),
            "resolution_schema": MANUAL_RESOLUTION_SCHEMA,
            "protocol_sha256": MANUAL_REVIEW_PROTOCOL_SHA256,
            "protocol": MANUAL_REVIEW_PROTOCOL,
            "judge_nonconsensus_may_be_selected_unresolved": False,
            "judge_required_review_may_be_selected_unresolved": False,
            "deterministic_other_accept_sample_modulus": OTHER_ACCEPT_REVIEW_MODULUS,
            "deterministic_other_accept_sample_bucket": OTHER_ACCEPT_REVIEW_BUCKET,
        },
        "final_authority_paths": {
            "dataset": corpus.relative(TRAIN_OUTPUT),
            "report": corpus.relative(REPORT_OUTPUT),
            "manifest": corpus.relative(MANIFEST_OUTPUT),
            "deficit_report": corpus.relative(DEFICIT_OUTPUT),
            "manual_review_queue": corpus.relative(MANUAL_QUEUE_OUTPUT),
        },
        "final_interface": {
            "row_schema": ROW_SCHEMA,
            "report_schema": REPORT_SCHEMA,
            "manifest_schema": MANIFEST_SCHEMA,
            "required_row_fields": sorted(GENERATED_REQUIRED_ROW_KEYS),
            "required_manifest_receipts": sorted(GENERATED_RECEIPT_KEYS),
        },
        "implementation_receipts": _implementation_receipts(),
        "selection_materialized": False,
        "eligible_for_training": False,
        "training_launch_authorized": False,
    }
    value["content_sha256_before_self_field"] = _self_address(value)
    return value


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def exact_key(question: str, answer: str) -> str:
    payload = f"{normalize_text(question)}\0{normalize_text(answer)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _shingles(question: str, answer: str) -> frozenset[str]:
    tokens = TOKEN_RE.findall(normalize_text(f"{question} {answer}"))
    if len(tokens) < 3:
        return frozenset(tokens)
    return frozenset("\x1f".join(tokens[index : index + 3]) for index in range(len(tokens) - 2))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 1.0 if left == right else 0.0
    return len(left & right) / len(left | right)


def attach_deduplication(candidates: Sequence[dict]) -> list[dict]:
    """Attach exact hashes and deterministic source-aware near clusters."""

    rows = [dict(candidate) for candidate in candidates]
    identities = [row.get("selection_candidate_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in identities):
        raise RuntimeError("selection candidate identity is absent")
    if len(identities) != len(set(identities)):
        raise RuntimeError("selection candidate identity is duplicated")
    rows.sort(key=lambda value: value["selection_candidate_id"])
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a == b:
            return
        if rows[a]["selection_candidate_id"] <= rows[b]["selection_candidate_id"]:
            parent[b] = a
        else:
            parent[a] = b

    shingles = [
        _shingles(row["question"], row["answer"]) for row in rows
    ]
    posting: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        candidates_to_compare: set[int] = set()
        source_group = row["source_group_id"]
        for shingle in sorted(shingles[index]):
            key = (source_group, shingle)
            values = posting[key]
            if len(values) <= MAX_COMMON_SHINGLE_POSTING:
                candidates_to_compare.update(values)
        for other in sorted(candidates_to_compare):
            if _jaccard(shingles[index], shingles[other]) >= NEAR_JACCARD_THRESHOLD:
                union(index, other)
        for shingle in shingles[index]:
            posting[(source_group, shingle)].append(index)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        clusters[find(index)].append(index)
    for members in clusters.values():
        base_identity = sorted(rows[index]["selection_candidate_id"] for index in members)
        base_sha = corpus.canonical_sha256(base_identity)
        information_signatures = {
            index: corpus.canonical_sha256({
                "evidence_quote_sha256s": rows[index]["evidence_quote_sha256s"],
                "normalized_answer": normalize_text(rows[index]["answer"]),
            })
            for index in members
        }
        task_information_pairs = Counter(
            (rows[index]["task_subtype"], information_signatures[index])
            for index in members
        )
        for index in members:
            row = rows[index]
            exact = exact_key(row["question"], row["answer"])
            pair = (row["task_subtype"], information_signatures[index])
            distinct_task = len({rows[item]["task_subtype"] for item in members}) > 1
            distinct_information = len(set(information_signatures.values())) > 1
            if distinct_task and distinct_information and task_information_pairs[pair] == 1:
                conflict_identity: Any = {
                    "base": base_sha,
                    "task_subtype": pair[0],
                    "information_signature": pair[1],
                }
            else:
                conflict_identity = {"base": base_sha, "shared_fact_view": True}
            row["dedupe"] = {
                "exact_key_sha256": exact,
                "near_duplicate_cluster_id": (
                    "near-cluster-v1:" + corpus.canonical_sha256(conflict_identity)
                ),
                "near_duplicate_base_cluster_sha256": base_sha,
                "exact_key_algorithm": EXACT_KEY_ALGORITHM,
                "near_cluster_algorithm": NEAR_CLUSTER_ALGORITHM,
            }
    return rows


def _token_length_candidate_receipt(
    candidate: Mapping[str, Any], *, total_tokens: int, status: str
) -> dict[str, Any]:
    receipt = {
        "schema": TOKEN_LENGTH_RECEIPT_SCHEMA,
        "status": status,
        "selection_candidate_id": candidate.get("selection_candidate_id"),
        "candidate_example_id": candidate.get("candidate_example_id"),
        "request_id": candidate.get("request_id"),
        "source_context_id": candidate.get("source_context_id"),
        "source_group_id": candidate.get("source_group_id"),
        "lane": candidate.get("lane"),
        "structural_review_sha256": candidate.get("structural_review_sha256"),
        "semantic_record_sha256": candidate.get("semantic_record_sha256"),
        "rendered_chat_token_count": total_tokens,
        "maximum_tokens": MAX_RENDERED_CHAT_TOKENS,
        "factual_text_rewritten": False,
        "eligible_for_length_gated_selection": status == "passed",
    }
    receipt["content_sha256_before_self_field"] = _self_address(receipt)
    return receipt


def apply_rendered_chat_length_gate(
    candidates: Sequence[dict], tokenizer: Any | None = None
) -> tuple[list[dict], dict[str, Any]]:
    """Render every candidate exactly and exclude >2,048-token atomic chats."""

    tokenizer = corpus.load_tokenizer() if tokenizer is None else tokenizer
    result: list[dict] = []
    exclusions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in candidates:
        candidate = dict(source)
        candidate_id = candidate.get("selection_candidate_id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in seen
        ):
            raise RuntimeError("length-gated selection candidate identity changed")
        seen.add(candidate_id)
        request = candidate.get("request")
        context = candidate.get("context")
        if not isinstance(request, dict) or not isinstance(context, dict):
            raise RuntimeError("length-gated candidate lacks request/context lineage")
        messages = structural.candidate_training_messages(
            request=request,
            context=context,
            question=candidate["question"],
            answer=candidate["answer"],
        )
        encoded = encode_chat_assistant_only(
            tokenizer,
            messages,
            enable_thinking=False,
        )
        total_tokens = encoded.get("total_token_count")
        assistant_tokens = encoded.get("assistant_token_count")
        if (
            encoded.get("mask_method") != corpus.ASSISTANT_MASK_METHOD
            or not isinstance(total_tokens, int)
            or total_tokens <= 0
            or not isinstance(assistant_tokens, int)
            or assistant_tokens <= 0
            or assistant_tokens != candidate.get("assistant_qwen36_token_count")
        ):
            raise RuntimeError("selection candidate official chat rendering changed")
        status = (
            "passed"
            if total_tokens <= MAX_RENDERED_CHAT_TOKENS
            else "excluded_rendered_chat_exceeds_token_limit"
        )
        receipt = _token_length_candidate_receipt(
            candidate,
            total_tokens=total_tokens,
            status=status,
        )
        candidate["rendered_chat_token_count"] = total_tokens
        candidate["token_length_gate_receipt"] = receipt
        result.append(candidate)
        if status != "passed":
            exclusions.append(receipt)
    exclusions.sort(
        key=lambda value: (
            str(value["lane"]),
            str(value["request_id"]),
            str(value["candidate_example_id"]),
        )
    )
    aggregate = {
        "schema": TOKEN_LENGTH_GATE_SCHEMA,
        "status": "sealed_passed_with_lineage_exclusions",
        "maximum_rendered_chat_tokens": MAX_RENDERED_CHAT_TOKENS,
        "candidates": len(result),
        "passed": len(result) - len(exclusions),
        "excluded_overlength": len(exclusions),
        "factual_text_rewritten": False,
        "all_candidates_length_gated_before_selection": True,
        "exclusions": exclusions,
        "exclusion_commitment_sha256": corpus.canonical_sha256(exclusions),
        "tokenizer": {
            "tokenizer_json_sha256": corpus.TOKENIZER_JSON_SHA256,
            "tokenizer_config_sha256": corpus.TOKENIZER_CONFIG_SHA256,
            "assistant_mask_method": corpus.ASSISTANT_MASK_METHOD,
        },
    }
    aggregate["content_sha256_before_self_field"] = _self_address(aggregate)
    return result, aggregate


def manual_review_required(candidate: dict) -> tuple[bool, list[str]]:
    result = candidate["semantic_result"]
    reasons = []
    if result.get("judge_consensus_passed") is not True:
        reasons.append("two_pass_judge_nonconsensus")
    if result.get("manual_review_required") is True:
        reasons.extend(
            f"judge:{reason}" for reason in result.get("manual_review_reasons", [])
        )
    if (
        result.get("judge_consensus_passed") is True
        and result.get("manual_review_required") is not True
        and int(hashlib.sha256(candidate["selection_candidate_id"].encode()).hexdigest()[:8], 16)
        % OTHER_ACCEPT_REVIEW_MODULUS
        == OTHER_ACCEPT_REVIEW_BUCKET
    ):
        reasons.append("deterministic_other_accept_review_sample")
    return bool(reasons), sorted(set(reasons))


def make_manual_queue(candidates: Sequence[dict]) -> list[dict]:
    queue = []
    for candidate in sorted(candidates, key=lambda value: value["selection_candidate_id"]):
        length_receipt = candidate.get("token_length_gate_receipt")
        if not isinstance(length_receipt, dict):
            raise RuntimeError("manual queue candidate lacks token-length gate receipt")
        if length_receipt.get("status") == "excluded_rendered_chat_exceeds_token_limit":
            continue
        if length_receipt.get("status") != "passed":
            raise RuntimeError("manual queue candidate has invalid token-length status")
        required, reasons = manual_review_required(candidate)
        if not required:
            continue
        row = {
            "schema": MANUAL_QUEUE_SCHEMA,
            "selection_candidate_id": candidate["selection_candidate_id"],
            "candidate_example_id": candidate["candidate_example_id"],
            "semantic_record_sha256": candidate["semantic_record_sha256"],
            "source_group_id": candidate["source_group_id"],
            "resource_id": candidate["resource_id"],
            "category": candidate["category"],
            "review_reasons": reasons,
            "protocol_sha256": MANUAL_REVIEW_PROTOCOL_SHA256,
            "eligible_for_training": False,
        }
        row["content_sha256_before_self_field"] = _self_address(row)
        queue.append(row)
    return queue


def validate_manual_resolutions(
    rows: Sequence[dict], queue: Sequence[dict]
) -> tuple[dict[str, dict], dict[str, Any]]:
    queued = {row["selection_candidate_id"]: row for row in queue}
    expected_fields = {
        "schema",
        "selection_candidate_id",
        "candidate_example_id",
        "semantic_record_sha256",
        "queue_row_sha256",
        "status",
        "reviewer_identity_sha256",
        "protocol_sha256",
        "content_sha256_before_self_field",
    }
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise RuntimeError("manual-review resolution schema changed")
        _require_self_address(row, "manual-review resolution")
        candidate_id = row["selection_candidate_id"]
        queue_row = queued.get(candidate_id)
        if (
            queue_row is None
            or candidate_id in result
            or row["schema"] != MANUAL_RESOLUTION_SCHEMA
            or row["candidate_example_id"] != queue_row["candidate_example_id"]
            or row["semantic_record_sha256"]
            != queue_row["semantic_record_sha256"]
            or row["queue_row_sha256"]
            != queue_row["content_sha256_before_self_field"]
            or row["status"] not in {"resolved_pass", "resolved_reject"}
            or row["protocol_sha256"] != MANUAL_REVIEW_PROTOCOL_SHA256
            or not HEX64.fullmatch(str(row["reviewer_identity_sha256"]))
        ):
            raise RuntimeError("manual-review resolution lineage changed")
        result[candidate_id] = row
    receipt = {
        "schema": "high-information-selection-manual-review-receipt-v1",
        "queue_rows": len(queue),
        "resolved_rows": len(result),
        "resolved_pass": sum(row["status"] == "resolved_pass" for row in result.values()),
        "resolved_reject": sum(row["status"] == "resolved_reject" for row in result.values()),
        "unresolved": len(queue) - len(result),
        "queue_commitment_sha256": corpus.canonical_sha256(
            [row["content_sha256_before_self_field"] for row in queue]
        ),
        "resolution_commitment_sha256": corpus.canonical_sha256(
            [
                result[key]["content_sha256_before_self_field"]
                for key in sorted(result)
            ]
        ),
        "unresolved_rows_training_eligible": False,
    }
    receipt["content_sha256_before_self_field"] = _self_address(receipt)
    return result, receipt


def apply_semantic_eligibility(
    candidates: Sequence[dict], resolutions: Mapping[str, dict]
) -> list[dict]:
    result = []
    for source in candidates:
        candidate = dict(source)
        length_receipt = candidate.get("token_length_gate_receipt")
        if not isinstance(length_receipt, dict):
            raise RuntimeError("semantic candidate lacks token-length gate receipt")
        length_status = length_receipt.get("status")
        if length_status not in {
            "passed",
            "excluded_rendered_chat_exceeds_token_limit",
        }:
            raise RuntimeError("semantic candidate token-length status changed")
        semantic = candidate["semantic_result"]
        nli_verdict = candidate["nli_result"]["verdict"]
        expected_nli = (
            "not_applicable"
            if candidate["generation_mode"] == "calibrated_hard_negative"
            else "pass"
        )
        required, reasons = manual_review_required(candidate)
        resolution = resolutions.get(candidate["selection_candidate_id"])
        nli_passed = nli_verdict == expected_nli
        if length_status == "excluded_rendered_chat_exceeds_token_limit":
            status = "rejected_rendered_chat_exceeds_token_limit"
            eligible = False
            manual_receipt = {"status": "not_applicable_token_length_rejected"}
        elif not nli_passed:
            status = "rejected_independent_nli"
            eligible = False
            manual_receipt = {"status": "not_applicable_nli_rejected"}
        elif required:
            if resolution is None:
                status = "manual_review_unresolved"
                eligible = False
                manual_receipt = {
                    "status": "unresolved",
                    "review_reasons": reasons,
                }
            elif resolution["status"] == "resolved_pass":
                status = "passed_after_manual_resolution"
                eligible = True
                manual_receipt = {
                    "status": "resolved_pass",
                    "resolution_content_sha256": resolution[
                        "content_sha256_before_self_field"
                    ],
                    "reviewer_identity_sha256": resolution[
                        "reviewer_identity_sha256"
                    ],
                }
            else:
                status = "manual_review_resolved_reject"
                eligible = False
                manual_receipt = {
                    "status": "resolved_reject",
                    "resolution_content_sha256": resolution[
                        "content_sha256_before_self_field"
                    ],
                }
        elif semantic.get("judge_consensus_passed") is True:
            status = "passed_two_pass_consensus"
            eligible = True
            manual_receipt = {"status": "not_required"}
        else:
            status = "manual_review_unresolved"
            eligible = False
            manual_receipt = {"status": "unresolved", "review_reasons": reasons}
        candidate["selection_eligibility_status"] = status
        candidate["selection_eligible"] = eligible
        candidate["manual_review_receipt"] = manual_receipt
        result.append(candidate)
    return result


def candidate_quality_score(candidate: dict, pool_supply: Mapping[str, int]) -> int:
    """Integer lexicographic-quality proxy used only after token maximization."""

    semantic = candidate["semantic_result"]
    score = 0
    if candidate["selection_eligibility_status"] == "passed_two_pass_consensus":
        score += 1_000_000
    elif candidate["selection_eligibility_status"] == "passed_after_manual_resolution":
        score += 800_000
    if candidate["category"] != "lineage_people_history":
        score += 100_000
    if candidate["task_subtype"] in {
        "application_scenario",
        "comparison_or_mechanism",
        "conflict_or_scope_resolution",
        "misconception_correction",
        "multi_fact_synthesis",
    }:
        score += 50_000
    probabilities = candidate["nli_result"].get("probabilities")
    if isinstance(probabilities, dict):
        score += int(10_000 * float(probabilities.get("entailment", 0.0)))
    score += max(0, 2_000 - 100 * len(semantic.get("manual_review_reasons", [])))
    supply = max(1, pool_supply.get(candidate["resource_id"], 1))
    score += min(5_000, 10_000_000 // supply)
    lexical = int(hashlib.sha256(candidate["selection_candidate_id"].encode()).hexdigest()[:8], 16)
    score = score * 10_000 + (9_999 - lexical % 10_000)
    return score


def _axis_accounting(candidates: Iterable[dict]) -> dict[str, Any]:
    counters = {
        "by_source": Counter(),
        "by_category": Counter(),
        "by_task_family": Counter(),
        "by_task_subtype": Counter(),
        "by_generation_mode": Counter(),
    }
    total = 0
    for candidate in candidates:
        tokens = candidate["assistant_qwen36_token_count"]
        total += tokens
        counters["by_source"][candidate["resource_id"]] += tokens
        counters["by_category"][candidate["category"]] += tokens
        counters["by_task_family"][candidate["task_family"]] += tokens
        counters["by_task_subtype"][candidate["task_subtype"]] += tokens
        counters["by_generation_mode"][candidate["generation_mode"]] += tokens
    return {
        "assistant_tokens": total,
        **{
            axis: dict(sorted(counter.items()))
            for axis, counter in counters.items()
        },
    }


def _axis_deficits(targets: Mapping[str, Any], accounting: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "assistant_tokens": targets["assistant_tokens"] - accounting["assistant_tokens"]
    }
    for axis in (
        "by_source",
        "by_category",
        "by_task_family",
        "by_task_subtype",
        "by_generation_mode",
    ):
        result[axis] = {
            key: target - accounting.get(axis, {}).get(key, 0)
            for key, target in sorted(targets[axis].items())
        }
        if any(value < 0 for value in result[axis].values()):
            raise RuntimeError("selection exceeded an exact target cap")
    return result


def solve_atomic_selection(
    candidates: Sequence[dict], targets: Mapping[str, Any]
) -> dict[str, Any]:
    """Two-stage deterministic binary MILP; exactness is proved after solving."""

    validate_targets(targets)
    for candidate in candidates:
        receipt = candidate.get("token_length_gate_receipt")
        if not isinstance(receipt, dict):
            raise RuntimeError("selection candidate lacks token-length gate receipt")
        _require_self_address(receipt, "selection candidate token-length gate receipt")
        status = receipt.get("status")
        total = receipt.get("rendered_chat_token_count")
        if (
            not isinstance(total, int)
            or total <= 0
            or receipt.get("maximum_tokens") != MAX_RENDERED_CHAT_TOKENS
            or receipt.get("factual_text_rewritten") is not False
            or status
            not in {"passed", "excluded_rendered_chat_exceeds_token_limit"}
            or (status == "passed" and total > MAX_RENDERED_CHAT_TOKENS)
            or (
                status == "excluded_rendered_chat_exceeds_token_limit"
                and total <= MAX_RENDERED_CHAT_TOKENS
            )
            or (
                status == "excluded_rendered_chat_exceeds_token_limit"
                and candidate.get("selection_eligible") is True
            )
        ):
            raise RuntimeError("selection candidate token-length gate changed")
    eligible = [candidate for candidate in candidates if candidate.get("selection_eligible") is True]
    eligible.sort(key=lambda value: value["selection_candidate_id"])
    allowed_axes = {
        "resource_id": targets["by_source"],
        "category": targets["by_category"],
        "task_family": targets["by_task_family"],
        "task_subtype": targets["by_task_subtype"],
        "generation_mode": targets["by_generation_mode"],
    }
    for candidate in eligible:
        tokens = candidate.get("assistant_qwen36_token_count")
        if not isinstance(tokens, int) or tokens <= 0:
            raise RuntimeError("selection candidate token count changed")
        for field, allowed in allowed_axes.items():
            if candidate.get(field) not in allowed:
                raise RuntimeError(f"selection candidate has an unknown {field}")
        dedupe = candidate.get("dedupe")
        if (
            not isinstance(dedupe, dict)
            or not HEX64.fullmatch(str(dedupe.get("exact_key_sha256", "")))
            or not isinstance(dedupe.get("near_duplicate_cluster_id"), str)
        ):
            raise RuntimeError("selection candidate lacks dedupe identity")
    if not eligible:
        accounting = _axis_accounting([])
        return {
            "schema": "high-information-atomic-selection-result-v1",
            "solver_status": "optimal_empty_pool",
            "exact_solution": False,
            "selected_candidate_ids": [],
            "selected_candidate_commitment_sha256": corpus.canonical_sha256([]),
            "accounting": accounting,
            "deficits": _axis_deficits(targets, accounting),
            "padding_used": False,
            "borrowing_used": False,
        }

    import numpy as np
    from scipy import __version__ as scipy_version
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix, vstack

    constraint_rows: list[tuple[list[int], list[float], float]] = []
    for field, target_axis in allowed_axes.items():
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, candidate in enumerate(eligible):
            grouped[candidate[field]].append(index)
        for key, cap in sorted(target_axis.items()):
            indices = grouped.get(key, [])
            coefficients = [float(eligible[index]["assistant_qwen36_token_count"]) for index in indices]
            constraint_rows.append((indices, coefficients, float(cap)))
    for group_name in ("exact_key_sha256", "near_duplicate_cluster_id", "request_id"):
        grouped = defaultdict(list)
        for index, candidate in enumerate(eligible):
            value = (
                candidate["request_id"]
                if group_name == "request_id"
                else candidate["dedupe"][group_name]
            )
            grouped[value].append(index)
        for _, indices in sorted(grouped.items()):
            if len(indices) > 1:
                constraint_rows.append((indices, [1.0] * len(indices), 1.0))

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    upper = []
    for row_index, (indices, coefficients, cap) in enumerate(constraint_rows):
        row_indices.extend([row_index] * len(indices))
        column_indices.extend(indices)
        values.extend(coefficients)
        upper.append(cap)
    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(constraint_rows), len(eligible)),
    ).tocsr()
    constraint = LinearConstraint(
        matrix,
        np.full(len(constraint_rows), -np.inf),
        np.asarray(upper),
    )
    tokens = np.asarray(
        [candidate["assistant_qwen36_token_count"] for candidate in eligible],
        dtype=float,
    )
    bounds = Bounds(np.zeros(len(eligible)), np.ones(len(eligible)))
    integrality = np.ones(len(eligible), dtype=int)
    options = {"presolve": True, "mip_rel_gap": 0.0}
    stage_one = milp(
        c=-tokens,
        integrality=integrality,
        bounds=bounds,
        constraints=constraint,
        options=options,
    )
    if not stage_one.success or stage_one.x is None:
        raise RuntimeError("atomic token maximization did not prove an optimum")
    stage_one_indices = [index for index, value in enumerate(stage_one.x) if value > 0.5]
    maximum_tokens = sum(eligible[index]["assistant_qwen36_token_count"] for index in stage_one_indices)

    total_row = coo_matrix(tokens.reshape(1, -1)).tocsr()
    stage_two_matrix = vstack([matrix, total_row], format="csr")
    stage_two_constraint = LinearConstraint(
        stage_two_matrix,
        np.concatenate([np.full(len(constraint_rows), -np.inf), [maximum_tokens]]),
        np.concatenate([np.asarray(upper), [maximum_tokens]]),
    )
    supply = Counter()
    for candidate in eligible:
        supply[candidate["resource_id"]] += candidate["assistant_qwen36_token_count"]
    quality = np.asarray(
        [candidate_quality_score(candidate, supply) for candidate in eligible],
        dtype=float,
    )
    stage_two = milp(
        c=-quality,
        integrality=integrality,
        bounds=bounds,
        constraints=stage_two_constraint,
        options=options,
    )
    if not stage_two.success or stage_two.x is None:
        raise RuntimeError("quality-priority selection did not prove an optimum")
    selected = [candidate for candidate, value in zip(eligible, stage_two.x, strict=True) if value > 0.5]
    accounting = _axis_accounting(selected)
    if accounting["assistant_tokens"] != maximum_tokens:
        raise RuntimeError("atomic solver token reconstruction changed")
    deficits = _axis_deficits(targets, accounting)
    for group_name in ("exact_key_sha256", "near_duplicate_cluster_id", "request_id"):
        observed = [
            (
                candidate["request_id"]
                if group_name == "request_id"
                else candidate["dedupe"][group_name]
            )
            for candidate in selected
        ]
        if len(observed) != len(set(observed)):
            raise RuntimeError(f"atomic selection duplicated {group_name}")
    exact = maximum_tokens == targets["assistant_tokens"]
    if exact and any(
        value != 0
        for axis in deficits.values()
        for value in (axis.values() if isinstance(axis, dict) else [axis])
    ):
        raise RuntimeError("total exactness did not imply per-axis exactness")
    selected_ids = sorted(candidate["selection_candidate_id"] for candidate in selected)
    return {
        "schema": "high-information-atomic-selection-result-v1",
        "solver_status": "optimal",
        "solver": {"implementation": "scipy.optimize.milp", "scipy_version": scipy_version},
        "eligible_candidates": len(eligible),
        "exact_solution": exact,
        "selected_candidate_ids": selected_ids,
        "selected_candidate_commitment_sha256": corpus.canonical_sha256(selected_ids),
        "accounting": accounting,
        "deficits": deficits,
        "padding_used": False,
        "truncation_used": False,
        "borrowing_used": False,
        "overshoot_or_tolerance_reported_as_exact": False,
    }


SELECTION_GROUP_FIELDS = (
    "exact_key_sha256",
    "near_duplicate_cluster_id",
    "request_id",
)


def _selection_candidate_index(
    candidates: Sequence[dict],
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for candidate in candidates:
        candidate_id = candidate.get("selection_candidate_id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in result
        ):
            raise RuntimeError("selection candidate identity coverage changed")
        result[candidate_id] = candidate
    return result


def _selection_group_value(candidate: Mapping[str, Any], field: str) -> str:
    if field == "request_id":
        value = candidate.get("request_id")
        if not isinstance(value, str) or not value:
            raise RuntimeError("selection candidate request identity changed")
        return value
    dedupe = candidate.get("dedupe")
    if not isinstance(dedupe, dict):
        raise RuntimeError("selection candidate dedupe identity changed")
    value = dedupe.get(field)
    if (
        not isinstance(value, str)
        or not value
        or (field == "exact_key_sha256" and not HEX64.fullmatch(value))
    ):
        raise RuntimeError(f"selection candidate {field} changed")
    return value


def _selection_ids(
    selection: Mapping[str, Any], index: Mapping[str, dict], role: str
) -> list[str]:
    selected_ids = selection.get("selected_candidate_ids")
    if (
        not isinstance(selected_ids, list)
        or any(not isinstance(value, str) or not value for value in selected_ids)
        or selected_ids != sorted(selected_ids)
        or len(selected_ids) != len(set(selected_ids))
        or not set(selected_ids) <= set(index)
    ):
        raise RuntimeError(f"{role} selected-candidate identity changed")
    if selection.get("selected_candidate_commitment_sha256") != (
        corpus.canonical_sha256(selected_ids)
    ):
        raise RuntimeError(f"{role} selected-candidate commitment changed")
    return selected_ids


def _groups_are_unique(candidates: Sequence[dict], role: str) -> None:
    for field in SELECTION_GROUP_FIELDS:
        values = [_selection_group_value(candidate, field) for candidate in candidates]
        if len(values) != len(set(values)):
            raise RuntimeError(f"{role} duplicated {field}")


def _all_deficits_zero(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_deficits_zero(item) for item in value.values())
    return type(value) is int and value == 0


def _validate_accounting_against_targets(
    accounting: Mapping[str, Any], targets: Mapping[str, Any]
) -> None:
    validate_targets(targets)
    if (
        targets["assistant_tokens"] != BASE_GENERATED_ASSISTANT_TOKENS
        or accounting.get("assistant_tokens") != targets["assistant_tokens"]
    ):
        raise RuntimeError("base atomic selection total target changed")
    for axis in (
        "by_source",
        "by_category",
        "by_task_family",
        "by_task_subtype",
        "by_generation_mode",
    ):
        observed = accounting.get(axis)
        expected = targets[axis]
        if (
            not isinstance(observed, dict)
            or not set(observed) <= set(expected)
            or any(observed.get(key, 0) != value for key, value in expected.items())
        ):
            raise RuntimeError(f"base atomic selection {axis} target changed")


def _validate_base_selection_component(
    candidates: Sequence[dict],
    selection: Mapping[str, Any],
    *,
    base_targets: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict], list[str], dict[str, Any]]:
    index = _selection_candidate_index(candidates)
    if (
        not isinstance(selection, Mapping)
        or selection.get("schema")
        != "high-information-atomic-selection-result-v1"
        or selection.get("solver_status") != "optimal"
        or selection.get("exact_solution") is not True
        or selection.get("padding_used") is not False
        or selection.get("truncation_used") is not False
        or selection.get("borrowing_used") is not False
        or selection.get("overshoot_or_tolerance_reported_as_exact") is not False
        or not _all_deficits_zero(selection.get("deficits"))
    ):
        raise RuntimeError("base atomic selection contract changed")
    selected_ids = _selection_ids(selection, index, "base atomic selection")
    selected = [index[candidate_id] for candidate_id in selected_ids]
    if any(
        candidate.get("selection_eligible") is not True
        or type(candidate.get("assistant_qwen36_token_count")) is not int
        or candidate["assistant_qwen36_token_count"] <= 0
        for candidate in selected
    ):
        raise RuntimeError("base atomic selection contains an ineligible candidate")
    accounting = _axis_accounting(selected)
    if (
        selection.get("accounting") != accounting
        or accounting["assistant_tokens"] != BASE_GENERATED_ASSISTANT_TOKENS
    ):
        raise RuntimeError("base atomic selection accounting changed")
    if base_targets is not None:
        _validate_accounting_against_targets(accounting, base_targets)
    _groups_are_unique(selected, "base atomic selection")
    return index, selected_ids, accounting


def _validate_replacement_selection_component(
    candidates: Sequence[dict], selection: Mapping[str, Any]
) -> tuple[dict[str, dict], list[str], dict[str, Any]]:
    index = _selection_candidate_index(candidates)
    if not isinstance(selection, Mapping):
        raise RuntimeError("seed replacement selection contract changed")
    required = selection.get("required_assistant_tokens")
    if (
        selection.get("schema")
        != "high-information-seed-replacement-selection-v1"
        or type(required) is not int
        or required < 0
        or selection.get("padding_used") is not False
        or selection.get("truncation_used") is not False
        or selection.get("borrowing_used") is not False
        or selection.get("overshoot_or_tolerance_reported_as_exact") is not False
    ):
        raise RuntimeError("seed replacement selection contract changed")
    selected_ids = _selection_ids(selection, index, "seed replacement selection")
    selected = [index[candidate_id] for candidate_id in selected_ids]
    if any(
        candidate.get("selection_eligible") is not True
        or type(candidate.get("assistant_qwen36_token_count")) is not int
        or candidate["assistant_qwen36_token_count"] <= 0
        for candidate in selected
    ):
        raise RuntimeError("seed replacement selection contains an ineligible candidate")
    accounting = _axis_accounting(selected)
    selected_tokens = accounting["assistant_tokens"]
    expected_exact = selected_tokens == required
    if (
        selection.get("accounting") != accounting
        or selected_tokens > required
        or selection.get("deficit_assistant_tokens")
        != required - selected_tokens
        or selection.get("exact_solution") is not expected_exact
    ):
        raise RuntimeError("seed replacement selection accounting changed")
    _groups_are_unique(selected, "seed replacement selection")
    return index, selected_ids, accounting


def solve_seed_replacement_selection(
    candidates: Sequence[dict],
    base_selection: Mapping[str, Any],
    required_tokens: int,
) -> dict[str, Any]:
    """Select an exact, disjoint atomic replacement for excluded seed tokens."""

    if type(required_tokens) is not int or required_tokens < 0:
        raise RuntimeError("seed replacement prerequisites changed")
    index, base_id_list, _ = _validate_base_selection_component(
        candidates, base_selection
    )
    base_ids = set(base_id_list)

    blocked = {
        field: {
            _selection_group_value(index[candidate_id], field)
            for candidate_id in base_ids
        }
        for field in SELECTION_GROUP_FIELDS
    }
    eligible = []
    for candidate in candidates:
        candidate_id = candidate.get("selection_candidate_id")
        tokens = candidate.get("assistant_qwen36_token_count")
        if candidate_id in base_ids or candidate.get("selection_eligible") is not True:
            continue
        if type(tokens) is not int or tokens <= 0:
            raise RuntimeError("seed replacement candidate token count changed")
        if tokens > required_tokens:
            continue
        if any(
            _selection_group_value(candidate, field) in blocked[field]
            for field in SELECTION_GROUP_FIELDS
        ):
            continue
        eligible.append(candidate)
    eligible.sort(key=lambda value: value["selection_candidate_id"])

    empty_accounting = _axis_accounting([])
    if required_tokens == 0:
        result = {
            "schema": "high-information-seed-replacement-selection-v1",
            "solver_status": "exact_zero_target",
            "eligible_candidates": 0,
            "exact_solution": True,
            "required_assistant_tokens": 0,
            "selected_candidate_ids": [],
            "selected_candidate_commitment_sha256": corpus.canonical_sha256([]),
            "accounting": empty_accounting,
            "deficit_assistant_tokens": 0,
            "padding_used": False,
            "truncation_used": False,
            "borrowing_used": False,
            "overshoot_or_tolerance_reported_as_exact": False,
        }
        _validate_replacement_selection_component(candidates, result)
        return result
    if not eligible:
        result = {
            "schema": "high-information-seed-replacement-selection-v1",
            "solver_status": "optimal_empty_pool",
            "eligible_candidates": 0,
            "exact_solution": False,
            "required_assistant_tokens": required_tokens,
            "selected_candidate_ids": [],
            "selected_candidate_commitment_sha256": corpus.canonical_sha256([]),
            "accounting": empty_accounting,
            "deficit_assistant_tokens": required_tokens,
            "padding_used": False,
            "truncation_used": False,
            "borrowing_used": False,
            "overshoot_or_tolerance_reported_as_exact": False,
        }
        _validate_replacement_selection_component(candidates, result)
        return result

    import numpy as np
    from scipy import __version__ as scipy_version
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    constraint_rows: list[tuple[list[int], list[float], float]] = []
    for field in SELECTION_GROUP_FIELDS:
        grouped: dict[str, list[int]] = defaultdict(list)
        for candidate_index, candidate in enumerate(eligible):
            grouped[_selection_group_value(candidate, field)].append(candidate_index)
        for _, indices in sorted(grouped.items()):
            if len(indices) > 1:
                constraint_rows.append((indices, [1.0] * len(indices), 1.0))

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    upper: list[float] = []
    for row_index, (indices, coefficients, cap) in enumerate(constraint_rows):
        row_indices.extend([row_index] * len(indices))
        column_indices.extend(indices)
        values.extend(coefficients)
        upper.append(cap)
    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(constraint_rows), len(eligible)),
    ).tocsr()
    group_constraint = LinearConstraint(
        matrix,
        np.full(len(constraint_rows), -np.inf),
        np.asarray(upper),
    )
    token_vector = np.asarray(
        [candidate["assistant_qwen36_token_count"] for candidate in eligible],
        dtype=float,
    )
    bounds = Bounds(np.zeros(len(eligible)), np.ones(len(eligible)))
    integrality = np.ones(len(eligible), dtype=int)
    options = {"presolve": True, "mip_rel_gap": 0.0}
    total_cap = LinearConstraint(
        coo_matrix(token_vector.reshape(1, -1)).tocsr(),
        np.asarray([-np.inf]),
        np.asarray([float(required_tokens)]),
    )
    stage_one = milp(
        c=-token_vector,
        integrality=integrality,
        bounds=bounds,
        constraints=(group_constraint, total_cap),
        options=options,
    )
    if not stage_one.success or stage_one.x is None:
        raise RuntimeError("seed replacement token maximization did not prove an optimum")
    maximum_tokens = sum(
        candidate["assistant_qwen36_token_count"]
        for candidate, selected in zip(eligible, stage_one.x, strict=True)
        if selected > 0.5
    )
    total_exact = LinearConstraint(
        coo_matrix(token_vector.reshape(1, -1)).tocsr(),
        np.asarray([float(maximum_tokens)]),
        np.asarray([float(maximum_tokens)]),
    )
    supply = Counter()
    for candidate in eligible:
        supply[candidate["resource_id"]] += candidate["assistant_qwen36_token_count"]
    quality = np.asarray(
        [candidate_quality_score(candidate, supply) for candidate in eligible],
        dtype=float,
    )
    stage_two = milp(
        c=-quality,
        integrality=integrality,
        bounds=bounds,
        constraints=(group_constraint, total_exact),
        options=options,
    )
    if not stage_two.success or stage_two.x is None:
        raise RuntimeError("seed replacement quality selection did not prove an optimum")
    selected = [
        candidate
        for candidate, chosen in zip(eligible, stage_two.x, strict=True)
        if chosen > 0.5
    ]
    accounting = _axis_accounting(selected)
    if accounting["assistant_tokens"] != maximum_tokens:
        raise RuntimeError("seed replacement token reconstruction changed")
    for field in SELECTION_GROUP_FIELDS:
        values_seen = [
            _selection_group_value(candidate, field) for candidate in selected
        ]
        if len(values_seen) != len(set(values_seen)):
            raise RuntimeError(f"seed replacement duplicated {field}")
        if set(values_seen) & blocked[field]:
            raise RuntimeError(f"seed replacement overlaps base {field}")
    selected_ids = sorted(candidate["selection_candidate_id"] for candidate in selected)
    result = {
        "schema": "high-information-seed-replacement-selection-v1",
        "solver_status": "optimal",
        "solver": {"implementation": "scipy.optimize.milp", "scipy_version": scipy_version},
        "eligible_candidates": len(eligible),
        "exact_solution": maximum_tokens == required_tokens,
        "required_assistant_tokens": required_tokens,
        "selected_candidate_ids": selected_ids,
        "selected_candidate_commitment_sha256": corpus.canonical_sha256(selected_ids),
        "accounting": accounting,
        "deficit_assistant_tokens": required_tokens - maximum_tokens,
        "padding_used": False,
        "truncation_used": False,
        "borrowing_used": False,
        "overshoot_or_tolerance_reported_as_exact": False,
    }
    _validate_replacement_selection_component(candidates, result)
    return result


def solve_coupled_seed_replacement_fallback(
    candidates: Sequence[dict],
    targets: Mapping[str, Any],
    initial_base_selection: Mapping[str, Any],
    required_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Jointly re-optimize base and replacement after a sequential deficit.

    The lexicographic objectives are base target attainment, replacement target
    attainment, and then total quality.  Exact/near/request constraints span
    both variable banks, so changing the base cannot introduce a duplicate in
    the replacement merely to recover token feasibility.
    """

    validate_targets(targets)
    if type(required_tokens) is not int or required_tokens < 0:
        raise RuntimeError("coupled seed replacement target changed")
    _validate_base_selection_component(
        candidates,
        initial_base_selection,
        base_targets=targets,
    )
    _selection_candidate_index(candidates)
    allowed_axes = {
        "resource_id": targets["by_source"],
        "category": targets["by_category"],
        "task_family": targets["by_task_family"],
        "task_subtype": targets["by_task_subtype"],
        "generation_mode": targets["by_generation_mode"],
    }
    eligible = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.get("selection_eligible") is True
        ),
        key=lambda value: value["selection_candidate_id"],
    )
    if not eligible:
        raise RuntimeError("coupled fallback lost the proven exact base pool")
    for candidate in eligible:
        tokens = candidate.get("assistant_qwen36_token_count")
        receipt = candidate.get("token_length_gate_receipt")
        if (
            type(tokens) is not int
            or tokens <= 0
            or not isinstance(receipt, dict)
            or receipt.get("status") != "passed"
            or receipt.get("maximum_tokens") != MAX_RENDERED_CHAT_TOKENS
            or receipt.get("factual_text_rewritten") is not False
            or type(receipt.get("rendered_chat_token_count")) is not int
            or receipt["rendered_chat_token_count"] <= 0
            or receipt["rendered_chat_token_count"] > MAX_RENDERED_CHAT_TOKENS
        ):
            raise RuntimeError("coupled fallback candidate eligibility changed")
        _require_self_address(
            receipt, "coupled fallback candidate token-length gate receipt"
        )
        for field, allowed in allowed_axes.items():
            if candidate.get(field) not in allowed:
                raise RuntimeError(
                    f"coupled fallback candidate has an unknown {field}"
                )
        for field in SELECTION_GROUP_FIELDS:
            _selection_group_value(candidate, field)

    import numpy as np
    from scipy import __version__ as scipy_version
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix, vstack

    count = len(eligible)
    token_values = [
        candidate["assistant_qwen36_token_count"] for candidate in eligible
    ]
    constraint_rows: list[
        tuple[list[int], list[float], float, float]
    ] = []

    # Base variables occupy [0, count); replacement variables occupy
    # [count, 2 * count).  Only base variables are subject to legacy axes.
    for field, target_axis in allowed_axes.items():
        grouped: dict[str, list[int]] = defaultdict(list)
        for candidate_index, candidate in enumerate(eligible):
            grouped[candidate[field]].append(candidate_index)
        for key, cap in sorted(target_axis.items()):
            indices = grouped.get(key, [])
            constraint_rows.append((
                indices,
                [float(token_values[index]) for index in indices],
                -np.inf,
                float(cap),
            ))

    # One constraint per global dedupe/request group spans both banks.  This
    # also prevents assigning one physical candidate to both components.
    for field in SELECTION_GROUP_FIELDS:
        grouped = defaultdict(list)
        for candidate_index, candidate in enumerate(eligible):
            grouped[_selection_group_value(candidate, field)].append(
                candidate_index
            )
        for _, indices in sorted(grouped.items()):
            # Exact-key rows are retained even for singleton groups because
            # they enforce b_i + r_i <= 1.  For near/request singletons that
            # same physical-candidate exclusion is already implied by exact.
            if field != "exact_key_sha256" and len(indices) == 1:
                continue
            both_banks = [*indices, *(count + index for index in indices)]
            constraint_rows.append((
                both_banks,
                [1.0] * len(both_banks),
                -np.inf,
                1.0,
            ))

    replacement_columns = [count + index for index in range(count)]
    constraint_rows.append((
        replacement_columns,
        [float(value) for value in token_values],
        -np.inf,
        float(required_tokens),
    ))

    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for row_index, (indices, values, minimum, maximum) in enumerate(
        constraint_rows
    ):
        row_indices.extend([row_index] * len(indices))
        column_indices.extend(indices)
        coefficients.extend(values)
        lower.append(minimum)
        upper.append(maximum)
    matrix = coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(len(constraint_rows), 2 * count),
    ).tocsr()
    constraint = LinearConstraint(
        matrix, np.asarray(lower), np.asarray(upper)
    )
    bounds = Bounds(np.zeros(2 * count), np.ones(2 * count))
    integrality = np.ones(2 * count, dtype=int)
    options = {"presolve": True, "mip_rel_gap": 0.0}
    base_vector = np.asarray(
        [*token_values, *([0] * count)], dtype=float
    )
    replacement_vector = np.asarray(
        [*([0] * count), *token_values], dtype=float
    )

    stage_one = milp(
        c=-base_vector,
        integrality=integrality,
        bounds=bounds,
        constraints=constraint,
        options=options,
    )
    if not stage_one.success or stage_one.x is None:
        raise RuntimeError("coupled base token maximization did not prove an optimum")
    maximum_base = sum(
        token_values[index]
        for index in range(count)
        if stage_one.x[index] > 0.5
    )
    if maximum_base != targets["assistant_tokens"]:
        raise RuntimeError("coupled fallback contradicted the proven exact base")

    base_total_row = coo_matrix(base_vector.reshape(1, -1)).tocsr()
    stage_two_matrix = vstack([matrix, base_total_row], format="csr")
    stage_two_constraint = LinearConstraint(
        stage_two_matrix,
        np.concatenate([np.asarray(lower), [float(maximum_base)]]),
        np.concatenate([np.asarray(upper), [float(maximum_base)]]),
    )
    stage_two = milp(
        c=-replacement_vector,
        integrality=integrality,
        bounds=bounds,
        constraints=stage_two_constraint,
        options=options,
    )
    if not stage_two.success or stage_two.x is None:
        raise RuntimeError(
            "coupled replacement token maximization did not prove an optimum"
        )
    maximum_replacement = sum(
        token_values[index]
        for index in range(count)
        if stage_two.x[count + index] > 0.5
    )

    replacement_total_row = coo_matrix(
        replacement_vector.reshape(1, -1)
    ).tocsr()
    stage_three_matrix = vstack(
        [stage_two_matrix, replacement_total_row], format="csr"
    )
    stage_three_constraint = LinearConstraint(
        stage_three_matrix,
        np.concatenate([
            np.asarray(lower),
            [float(maximum_base), float(maximum_replacement)],
        ]),
        np.concatenate([
            np.asarray(upper),
            [float(maximum_base), float(maximum_replacement)],
        ]),
    )
    supply = Counter()
    for candidate in eligible:
        supply[candidate["resource_id"]] += candidate[
            "assistant_qwen36_token_count"
        ]
    quality_values = [
        candidate_quality_score(candidate, supply) for candidate in eligible
    ]
    quality_vector = np.asarray(
        [*quality_values, *quality_values], dtype=float
    )
    stage_three = milp(
        c=-quality_vector,
        integrality=integrality,
        bounds=bounds,
        constraints=stage_three_constraint,
        options=options,
    )
    if not stage_three.success or stage_three.x is None:
        raise RuntimeError("coupled quality selection did not prove an optimum")

    base_selected = [
        candidate
        for index, candidate in enumerate(eligible)
        if stage_three.x[index] > 0.5
    ]
    replacement_selected = [
        candidate
        for index, candidate in enumerate(eligible)
        if stage_three.x[count + index] > 0.5
    ]
    base_accounting = _axis_accounting(base_selected)
    replacement_accounting = _axis_accounting(replacement_selected)
    if (
        base_accounting["assistant_tokens"] != maximum_base
        or replacement_accounting["assistant_tokens"]
        != maximum_replacement
    ):
        raise RuntimeError("coupled solver token reconstruction changed")
    base_ids = sorted(
        candidate["selection_candidate_id"] for candidate in base_selected
    )
    replacement_ids = sorted(
        candidate["selection_candidate_id"]
        for candidate in replacement_selected
    )
    solver = {
        "implementation": "scipy.optimize.milp",
        "scipy_version": scipy_version,
        "mode": "coupled_base_replacement_lexicographic_fallback_v1",
    }
    base_result = {
        "schema": "high-information-atomic-selection-result-v1",
        "solver_status": "optimal",
        "solver": solver,
        "eligible_candidates": count,
        "exact_solution": True,
        "selected_candidate_ids": base_ids,
        "selected_candidate_commitment_sha256": corpus.canonical_sha256(
            base_ids
        ),
        "accounting": base_accounting,
        "deficits": _axis_deficits(targets, base_accounting),
        "padding_used": False,
        "truncation_used": False,
        "borrowing_used": False,
        "overshoot_or_tolerance_reported_as_exact": False,
    }
    replacement_result = {
        "schema": "high-information-seed-replacement-selection-v1",
        "solver_status": "optimal",
        "solver": solver,
        "eligible_candidates": count,
        "exact_solution": maximum_replacement == required_tokens,
        "required_assistant_tokens": required_tokens,
        "selected_candidate_ids": replacement_ids,
        "selected_candidate_commitment_sha256": corpus.canonical_sha256(
            replacement_ids
        ),
        "accounting": replacement_accounting,
        "deficit_assistant_tokens": required_tokens - maximum_replacement,
        "padding_used": False,
        "truncation_used": False,
        "borrowing_used": False,
        "overshoot_or_tolerance_reported_as_exact": False,
    }
    _validate_base_selection_component(
        candidates, base_result, base_targets=targets
    )
    _validate_replacement_selection_component(
        candidates, replacement_result
    )
    combine_base_and_seed_replacement_selection(
        candidates,
        base_result,
        replacement_result,
        base_targets=targets,
    )
    return base_result, replacement_result


def combine_base_and_seed_replacement_selection(
    candidates: Sequence[dict],
    base_selection: Mapping[str, Any],
    replacement_selection: Mapping[str, Any],
    *,
    base_targets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the single immutable selection identity used by final rows."""

    index, base_ids, _ = _validate_base_selection_component(
        candidates, base_selection, base_targets=base_targets
    )
    replacement_index, replacement_ids, _ = (
        _validate_replacement_selection_component(
            candidates, replacement_selection
        )
    )
    if set(index) != set(replacement_index):
        raise RuntimeError("combined generated candidate identity changed")
    selected_ids = sorted(
        [*base_ids, *replacement_ids]
    )
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("combined generated selection duplicated candidate identity")
    selected = [index[candidate_id] for candidate_id in selected_ids]
    if base_targets is not None:
        for candidate in selected:
            for field, axis in (
                ("resource_id", "by_source"),
                ("category", "by_category"),
                ("task_family", "by_task_family"),
                ("task_subtype", "by_task_subtype"),
                ("generation_mode", "by_generation_mode"),
            ):
                if candidate.get(field) not in base_targets[axis]:
                    raise RuntimeError(
                        f"combined generated selection has unknown {field}"
                    )
    _groups_are_unique(selected, "combined generated selection")
    accounting = _axis_accounting(selected)
    required_replacement = replacement_selection["required_assistant_tokens"]
    expected_tokens = (
        BASE_GENERATED_ASSISTANT_TOKENS + required_replacement
    )
    deficit = expected_tokens - accounting["assistant_tokens"]
    if deficit < 0:
        raise RuntimeError("combined generated selection exceeded its token target")
    exact = (
        replacement_selection.get("exact_solution") is True and deficit == 0
    )
    return {
        "schema": "high-information-combined-atomic-selection-result-v1",
        "solver_status": "optimal" if exact else "exact_component_unsolved",
        "exact_solution": exact,
        "required_assistant_tokens": expected_tokens,
        "deficit_assistant_tokens": deficit,
        "selected_candidate_ids": selected_ids,
        "selected_candidate_commitment_sha256": corpus.canonical_sha256(selected_ids),
        "accounting": accounting,
        "base_selection": dict(base_selection),
        "seed_qa_replacement_selection": dict(replacement_selection),
        "padding_used": False,
        "truncation_used": False,
        "borrowing_used": False,
        "overshoot_or_tolerance_reported_as_exact": False,
    }


def validate_seed_qa_replacement_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != SEED_REPLACEMENT_RECEIPT_KEYS:
        raise RuntimeError("final seed-QA replacement receipt fields changed")
    required = value.get("replacement_assistant_tokens_required")
    selected = value.get("replacement_assistant_tokens_selected")
    if (
        value.get("schema") != SEED_REPLACEMENT_SCHEMA
        or not HEX64.fullmatch(
            str(value.get("seed_qa_semantic_authority_file_sha256", ""))
        )
        or not HEX64.fullmatch(
            str(value.get("seed_qa_semantic_authority_content_sha256", ""))
        )
        or type(required) is not int
        or required < 0
        or type(selected) is not int
        or selected != required
        or value.get("base_generated_assistant_tokens")
        != BASE_GENERATED_ASSISTANT_TOKENS
        or value.get("total_generated_assistant_tokens")
        != BASE_GENERATED_ASSISTANT_TOKENS + selected
    ):
        raise RuntimeError("final seed-QA replacement receipt changed")
    return value


def build_seed_qa_replacement_receipt(
    contract: Mapping[str, Any],
    candidates: Sequence[dict],
    replacement_selection: Mapping[str, Any],
) -> dict[str, Any]:
    contract = validate_seed_qa_replacement_contract(contract)
    _, _, accounting = _validate_replacement_selection_component(
        candidates, replacement_selection
    )
    required = contract["replacement_assistant_tokens_required"]
    selected = accounting["assistant_tokens"]
    if (
        replacement_selection.get("exact_solution") is not True
        or replacement_selection.get("required_assistant_tokens") != required
        or selected != required
    ):
        raise RuntimeError("cannot issue an exact seed-QA replacement receipt")
    return validate_seed_qa_replacement_receipt({
        "schema": SEED_REPLACEMENT_SCHEMA,
        "seed_qa_semantic_authority_file_sha256": contract[
            "seed_qa_semantic_authority_file_sha256"
        ],
        "seed_qa_semantic_authority_content_sha256": contract[
            "seed_qa_semantic_authority_content_sha256"
        ],
        "replacement_assistant_tokens_required": required,
        "replacement_assistant_tokens_selected": selected,
        "base_generated_assistant_tokens": BASE_GENERATED_ASSISTANT_TOKENS,
        "total_generated_assistant_tokens": (
            BASE_GENERATED_ASSISTANT_TOKENS + selected
        ),
    })


def build_seed_qa_replacement_deficit_receipt(
    contract: Mapping[str, Any],
    candidates: Sequence[dict],
    replacement_selection: Mapping[str, Any],
) -> dict[str, Any]:
    contract = validate_seed_qa_replacement_contract(contract)
    _, _, accounting = _validate_replacement_selection_component(
        candidates, replacement_selection
    )
    required = contract["replacement_assistant_tokens_required"]
    selected = accounting["assistant_tokens"]
    if (
        replacement_selection.get("exact_solution") is not False
        or replacement_selection.get("required_assistant_tokens") != required
        or selected >= required
        or replacement_selection.get("deficit_assistant_tokens")
        != required - selected
    ):
        raise RuntimeError("seed-QA replacement deficit receipt changed")
    body = {
        "schema": SEED_REPLACEMENT_DEFICIT_SCHEMA,
        "status": "exact_atomic_replacement_unsolved",
        "seed_qa_semantic_authority_file_sha256": contract[
            "seed_qa_semantic_authority_file_sha256"
        ],
        "seed_qa_semantic_authority_content_sha256": contract[
            "seed_qa_semantic_authority_content_sha256"
        ],
        "replacement_assistant_tokens_required": required,
        "replacement_assistant_tokens_selected": selected,
        "replacement_assistant_tokens_deficit": required - selected,
        "base_generated_assistant_tokens": BASE_GENERATED_ASSISTANT_TOKENS,
        "total_generated_assistant_tokens_selected": (
            BASE_GENERATED_ASSISTANT_TOKENS + selected
        ),
        "training_rows_emitted": False,
    }
    body["content_sha256_before_self_field"] = _self_address(body)
    return body


def _primary_semantic_shard(
    shard: int, contexts: Mapping[str, dict], requests: Mapping[str, dict]
) -> tuple[list[dict], dict]:
    review_path, summary_path = primary_nli.structural_paths(shard)
    packets, structural_summary = primary_nli.load_structural_packets(
        shard, review_path, summary_path
    )
    nli_index, nli_report = primary_judge.load_nli_results(shard, packets)
    groups = fill_judge.groups_by_request(packets)
    paths = primary_judge.output_paths(shard, smoke=False)
    _regular_file(paths["output"], "primary semantic output")
    _regular_file(paths["report"], "primary semantic report")
    output_payload = paths["output"].read_bytes()
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    primary_judge._require_self_address(report, "primary semantic report")
    run_contract = report.get("run_contract")
    if not isinstance(run_contract, dict):
        raise RuntimeError("primary semantic report lacks a run contract")
    primary_judge._require_self_address(run_contract, "primary semantic run contract")
    if (
        report.get("schema") != "high-information-semantic-judge-report-v1"
        or report.get("status") != "complete_manual_and_global_selection_pending"
        or report.get("gpu_shard") != shard
        or report.get("request_groups") != len(groups)
        or report.get("candidates") != len(packets)
        or report.get("output") != corpus.relative(paths["output"])
        or report.get("output_sha256") != corpus.sha256_bytes(output_payload)
        or report.get("semantic_verification_completed") is not False
        or report.get("training_rows_emitted") is not False
        or run_contract.get("worker_file_sha256") != PRIMARY_JUDGE_FILE_SHA256
        or run_contract.get("candidate_grouping") != CANDIDATE_GROUPING
        or run_contract.get("request_batch_size") != 16
        or run_contract.get("guided_schema_sha256") != fill_judge.GUIDED_SCHEMA_SHA256
    ):
        raise RuntimeError("primary semantic shard receipt changed")
    rows = [json.loads(line) for line in output_payload.decode().splitlines() if line.strip()]
    if len(rows) != len(groups):
        raise RuntimeError("primary semantic output row count changed")
    candidates = []
    contract_sha = run_contract["content_sha256_before_self_field"]
    for row, group in zip(rows, groups, strict=True):
        primary_judge.validate_record(row, group, contract_sha)
        packet = group[0]
        context = contexts[packet["source_context_id"]]
        request = requests[packet["request_id"]]
        result = row["results"][0]
        candidates.append(_selection_candidate(
            lane="primary",
            packet=packet,
            context=context,
            request=request,
            semantic_result=result,
            semantic_record=row,
            nli_result=nli_index[packet["candidate_example_id"]],
            structural_summary=structural_summary,
            semantic_shard_receipt={
                "output_sha256": report["output_sha256"],
                "report_self_sha256": report["content_sha256_before_self_field"],
                "run_contract_sha256": contract_sha,
            },
        ))
    receipt = {
        "lane": "primary",
        "gpu_shard": shard,
        "candidates": len(candidates),
        "structural_review_sha256": structural_summary["report_file_sha256"],
        "nli_output_sha256": nli_report["output_sha256"],
        "semantic_output_sha256": report["output_sha256"],
        "semantic_report_self_sha256": report["content_sha256_before_self_field"],
        "semantic_run_contract_sha256": contract_sha,
    }
    return candidates, receipt


def _fill_semantic_args(shard: int) -> argparse.Namespace:
    return argparse.Namespace(
        shard_index=shard,
        gpu_index=None,
        model_directory=fill_judge.MODEL_DIRECTORY,
        request_batch_size=16,
        max_model_len=16_384,
        max_tokens=3_072,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        smoke=False,
        check_plan=False,
        check_output=True,
    )


def _fill_semantic_shard(shard: int) -> tuple[list[dict], dict]:
    completion = fill_judge.validate_completed_output(_fill_semantic_args(shard))
    packets, structural_summary = fill_nli.load_structural_packets(shard)
    nli_index, nli_identity = fill_judge.load_fill_nli_results(shard, packets)
    groups = fill_judge.groups_by_request(packets)
    paths = fill_judge.output_paths(shard, smoke=False)
    output_payload = paths["output"].read_bytes()
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    contract_sha = report["run_contract"]["content_sha256_before_self_field"]
    pass_contract_sha = structural_summary["generation_pass_contract"][
        "content_sha256_before_self_field"
    ]
    rows = [json.loads(line) for line in output_payload.decode().splitlines() if line.strip()]
    if len(rows) != len(groups):
        raise RuntimeError("fill semantic output row count changed")
    (_, _, _, _, contexts, requests) = generation_pass.load_generation_pass(
        fill_judge.PASS_ID, shard
    )
    candidates = []
    for row, group in zip(rows, groups, strict=True):
        fill_judge.validate_record(
            row,
            group,
            contract_sha,
            pass_contract_sha,
            nli_identity["output_sha256"],
            nli_identity["receipt_self_sha256"],
        )
        packet = group[0]
        context = contexts[packet["source_context_id"]]
        request = requests[packet["request_id"]]
        candidates.append(_selection_candidate(
            lane="fill",
            packet=packet,
            context=context,
            request=request,
            semantic_result=row["results"][0],
            semantic_record=row,
            nli_result=nli_index[packet["candidate_example_id"]],
            structural_summary=structural_summary,
            semantic_shard_receipt={
                "output_sha256": completion["output_sha256"],
                "report_self_sha256": completion["report_self_sha256"],
                "receipt_self_sha256": completion["receipt_self_sha256"],
                "run_contract_sha256": completion["run_contract_sha256"],
            },
        ))
    receipt = {
        "lane": "fill",
        "gpu_shard": shard,
        "candidates": len(candidates),
        "generation_pass_contract_sha256": pass_contract_sha,
        "structural_review_sha256": structural_summary["report_file_sha256"],
        "nli_output_sha256": nli_identity["output_sha256"],
        "nli_receipt_self_sha256": nli_identity["receipt_self_sha256"],
        "semantic_output_sha256": completion["output_sha256"],
        "semantic_report_self_sha256": completion["report_self_sha256"],
        "semantic_receipt_self_sha256": completion["receipt_self_sha256"],
        "semantic_run_contract_sha256": completion["run_contract_sha256"],
    }
    return candidates, receipt


def _selection_candidate(
    *,
    lane: str,
    packet: dict,
    context: dict,
    request: dict,
    semantic_result: dict,
    semantic_record: dict,
    nli_result: dict,
    structural_summary: dict,
    semantic_shard_receipt: dict,
) -> dict:
    candidate_id = corpus.content_id("global-selection-candidate-v1", {
        "lane": lane,
        "candidate_example_id": packet["candidate_example_id"],
        "semantic_record_sha256": semantic_record[
            "content_sha256_before_self_field"
        ],
    })
    return {
        "selection_candidate_id": candidate_id,
        "lane": lane,
        "candidate_example_id": packet["candidate_example_id"],
        "request_id": packet["request_id"],
        "source_context_id": packet["source_context_id"],
        "source_group_id": packet["source_group_id"],
        "resource_id": context["resource_id"],
        "artifact_id": context["artifact_id"],
        "task_family": packet["task_family"],
        "task_subtype": packet["task_subtype"],
        "generation_mode": packet["generation_mode"],
        "question": packet["question"],
        "answer": packet["answer"],
        "assistant_qwen36_token_count": packet["assistant_qwen36_token_count"],
        "evidence_quote_sha256s": packet["evidence_quote_sha256s"],
        "rights_basis": packet["rights_basis"],
        "safety_transfer_flags": packet["safety_transfer_flags"],
        "context": context,
        "request": request,
        "semantic_result": semantic_result,
        "semantic_record_sha256": semantic_record[
            "content_sha256_before_self_field"
        ],
        "semantic_pass_output_sha256s": semantic_record["pass_output_sha256s"],
        "nli_result": nli_result,
        "nli_result_sha256": nli_result["content_sha256_before_self_field"],
        "structural_review_sha256": structural_summary["report_file_sha256"],
        "semantic_shard_receipt": semantic_shard_receipt,
    }


def load_all_semantic_candidates() -> tuple[list[dict], dict[str, Any]]:
    states = semantic_input_states()
    if any(not state["sealed_set_present"] for state in states):
        raise RuntimeError("global selection requires every primary and fill semantic seal")
    _, primary_contexts, primary_requests = structural.load_plan()
    candidates = []
    shard_receipts = []
    for shard in range(4):
        rows, receipt = _primary_semantic_shard(
            shard, primary_contexts, primary_requests
        )
        candidates.extend(rows)
        shard_receipts.append(receipt)
    for shard in range(4):
        rows, receipt = _fill_semantic_shard(shard)
        candidates.extend(rows)
        shard_receipts.append(receipt)
    return candidates, {
        "shards": shard_receipts,
        "primary_candidates": sum(
            item["candidates"] for item in shard_receipts if item["lane"] == "primary"
        ),
        "fill_candidates": sum(
            item["candidates"] for item in shard_receipts if item["lane"] == "fill"
        ),
        "content_sha256": corpus.canonical_sha256(shard_receipts),
    }


def load_manual_resolution_rows() -> list[dict]:
    if not MANUAL_RESOLUTIONS.exists():
        return []
    path = _regular_file(MANUAL_RESOLUTIONS, "manual-review resolutions")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prepare_global_pool(
    candidates: Sequence[dict],
    *,
    category_map: Mapping[str, str],
    manual_resolution_rows: Sequence[dict],
    tokenizer: Any | None = None,
) -> tuple[list[dict], list[dict], dict, dict]:
    categorized = []
    for source in candidates:
        candidate = dict(source)
        resource = candidate["resource_id"]
        if resource not in category_map:
            raise RuntimeError("semantic candidate resource lacks a category target")
        candidate["category"] = category_map[resource]
        categorized.append(candidate)
    length_gated, length_receipt = apply_rendered_chat_length_gate(
        categorized, tokenizer
    )
    deduplicated = attach_deduplication(length_gated)
    queue = make_manual_queue(deduplicated)
    resolutions, manual_receipt = validate_manual_resolutions(
        manual_resolution_rows, queue
    )
    eligible = apply_semantic_eligibility(deduplicated, resolutions)
    return eligible, queue, manual_receipt, length_receipt


def make_training_row(candidate: dict, selection_receipt_sha256: str) -> dict:
    if candidate.get("selection_eligible") is not True:
        raise RuntimeError("ineligible semantic candidate cannot become a training row")
    length_receipt = candidate.get("token_length_gate_receipt")
    if (
        not isinstance(length_receipt, dict)
        or length_receipt.get("status") != "passed"
        or length_receipt.get("rendered_chat_token_count", MAX_RENDERED_CHAT_TOKENS + 1)
        > MAX_RENDERED_CHAT_TOKENS
    ):
        raise RuntimeError("overlength or unmeasured candidate cannot become a training row")
    _require_self_address(length_receipt, "candidate token-length gate receipt")
    context = candidate["context"]
    request = candidate["request"]
    messages = structural.candidate_training_messages(
        request=request,
        context=context,
        question=candidate["question"],
        answer=candidate["answer"],
    )
    manual = candidate["manual_review_receipt"]
    consensus = candidate["semantic_result"].get("judge_consensus_passed") is True
    semantic_basis = "two_pass_consensus" if consensus else "resolved_manual_review"
    verification = {
        "structural": {
            "status": "passed",
            "review_file_sha256": candidate["structural_review_sha256"],
        },
        "nli": {
            "status": "passed",
            "result_content_sha256": candidate["nli_result_sha256"],
            "raw_verdict": candidate["nli_result"]["verdict"],
        },
        "semantic_judge_pass_1": {
            "status": "passed",
            "basis": semantic_basis,
            "raw_output_sha256": candidate["semantic_pass_output_sha256s"][
                primary_judge.PASS_NAMES[0]
            ],
        },
        "semantic_judge_pass_2": {
            "status": "passed",
            "basis": semantic_basis,
            "raw_output_sha256": candidate["semantic_pass_output_sha256s"][
                primary_judge.PASS_NAMES[1]
            ],
        },
        "selection": {
            "status": "passed",
            "selector_receipt_sha256": selection_receipt_sha256,
            "rendered_chat_token_count": length_receipt[
                "rendered_chat_token_count"
            ],
            "maximum_tokens": MAX_RENDERED_CHAT_TOKENS,
            "token_length_gate_receipt_sha256": length_receipt[
                "content_sha256_before_self_field"
            ],
        },
    }
    identity = {
        "selection_candidate_id": candidate["selection_candidate_id"],
        "selector_receipt_sha256": selection_receipt_sha256,
    }
    row = {
        "schema": ROW_SCHEMA,
        "record_id": corpus.content_id("domain-training-row-v1", identity),
        "candidate_example_id": candidate["candidate_example_id"],
        "request_id": candidate["request_id"],
        "source_context_id": candidate["source_context_id"],
        "source_group_id": candidate["source_group_id"],
        "resource_id": candidate["resource_id"],
        "split": "train",
        "training_format": "chat_assistant_only",
        "messages": messages,
        "tools": None,
        "assistant_mask": {
            "policy": "assistant_only_v1",
            "assistant_message_indices": [len(messages) - 1],
            "system_tokens": False,
            "user_tokens": False,
            "tool_result_tokens": False,
        },
        "assistant_qwen36_token_count": candidate[
            "assistant_qwen36_token_count"
        ],
        "task_family": candidate["task_family"],
        "task_subtype": candidate["task_subtype"],
        "generation_mode": candidate["generation_mode"],
        "category": candidate["category"],
        "hard_negative": candidate["generation_mode"] == "calibrated_hard_negative",
        "question": candidate["question"],
        "answer": candidate["answer"],
        "evidence_quote_sha256s": candidate["evidence_quote_sha256s"],
        "verification_receipts": verification,
        "manual_review_receipt": manual,
        "dedupe": candidate["dedupe"],
        "generator": {
            "lane": candidate["lane"],
            "generation_pass_id": (
                fill_judge.PASS_ID if candidate["lane"] == "fill" else "primary"
            ),
            "semantic_shard_receipt": candidate["semantic_shard_receipt"],
        },
        "rights_basis": candidate["rights_basis"],
        "rights_authorization": {
            "status": "passed",
            "source_rights_status_preserved": True,
            "public_license_status_rewritten": False,
        },
        "safety_transfer_flags": candidate["safety_transfer_flags"],
        "lineage": {
            **context["lineage"],
            "artifact_id": context["artifact_id"],
            "selection_candidate_id": candidate["selection_candidate_id"],
            "semantic_record_sha256": candidate["semantic_record_sha256"],
            "nli_result_sha256": candidate["nli_result_sha256"],
        },
        "eligible_for_training": True,
    }
    if not GENERATED_REQUIRED_ROW_KEYS.issubset(row):
        raise RuntimeError("generated-domain training row interface changed")
    return row


def build_deficit_report(
    *,
    selection: dict,
    pool: Sequence[dict],
    queue: Sequence[dict],
    input_receipt: dict,
    token_length_receipt: dict,
) -> dict:
    report = {
        "schema": DEFICIT_SCHEMA,
        "status": "exact_atomic_selection_unsolved_no_training_rows_emitted",
        "selection": selection,
        "candidate_pool": {
            "semantic_candidates": len(pool),
            "selection_eligible": sum(
                candidate.get("selection_eligible") is True for candidate in pool
            ),
            "manual_review_unresolved": sum(
                candidate.get("selection_eligibility_status")
                == "manual_review_unresolved"
                for candidate in pool
            ),
            "rendered_chat_overlength_excluded": sum(
                candidate.get("selection_eligibility_status")
                == "rejected_rendered_chat_exceeds_token_limit"
                for candidate in pool
            ),
            "manual_review_queue_rows": len(queue),
        },
        "semantic_input_receipt": input_receipt,
        "rendered_chat_length_gate": token_length_receipt,
        "padding_used": False,
        "truncation_used": False,
        "cross_source_borrowing_used": False,
        "cross_category_borrowing_used": False,
        "train_dataset_emitted": False,
        "eligible_for_training": False,
        "training_launch_authorized": False,
    }
    report["content_sha256_before_self_field"] = _self_address(report)
    return report


def build_final_authority(
    *,
    pool: Sequence[dict],
    selection: dict,
    input_receipt: dict,
    manual_receipt: dict,
    token_length_receipt: dict,
    selection_contract: dict,
    seed_qa_replacement: dict,
) -> tuple[bytes, dict, dict]:
    if not isinstance(selection, dict):
        raise RuntimeError("final authority selection changed")
    base_component = selection.get("base_selection")
    replacement_component = selection.get("seed_qa_replacement_selection")
    if not isinstance(base_component, dict) or not isinstance(
        replacement_component, dict
    ):
        raise RuntimeError("final authority selection components changed")
    reconstructed = combine_base_and_seed_replacement_selection(
        pool,
        base_component,
        replacement_component,
        base_targets=selection_targets(selection_contract),
    )
    if selection != reconstructed or selection.get("exact_solution") is not True:
        raise RuntimeError("final authority requires an exact atomic selection")
    seed_qa_replacement = validate_seed_qa_replacement_receipt(
        seed_qa_replacement
    )
    replacement_accounting = replacement_component["accounting"]
    if (
        seed_qa_replacement["replacement_assistant_tokens_required"]
        != replacement_component["required_assistant_tokens"]
        or seed_qa_replacement["replacement_assistant_tokens_selected"]
        != replacement_accounting["assistant_tokens"]
        or seed_qa_replacement["total_generated_assistant_tokens"]
        != selection["accounting"]["assistant_tokens"]
    ):
        raise RuntimeError("final seed-QA replacement receipt changed")
    index = _selection_candidate_index(pool)
    selected = [index[candidate_id] for candidate_id in selection["selected_candidate_ids"]]
    selector_receipt = {
        "selection_contract_sha256": SELECTION_CONTRACT_SELF_SHA256,
        "selection_result_sha256": corpus.canonical_sha256(selection),
        "selected_candidate_commitment_sha256": selection[
            "selected_candidate_commitment_sha256"
        ],
        "rendered_chat_length_gate_sha256": token_length_receipt[
            "content_sha256_before_self_field"
        ],
        "builder_file_sha256": corpus.file_sha256(Path(__file__).resolve()),
    }
    selector_receipt_sha = corpus.canonical_sha256(selector_receipt)
    rows = [make_training_row(candidate, selector_receipt_sha) for candidate in selected]
    rows.sort(key=lambda value: (value["request_id"], value["candidate_example_id"]))
    payload = corpus.jsonl_payload(rows)
    accounting = _axis_accounting(selected)
    if accounting != selection["accounting"]:
        raise RuntimeError("final row accounting differs from atomic selection")
    receipts = {
        "primary_generation": _sealed_receipt([
            item for item in input_receipt["shards"] if item["lane"] == "primary"
        ]),
        "fill_generation": _sealed_receipt([
            item for item in input_receipt["shards"] if item["lane"] == "fill"
        ]),
        "structural_verification": _sealed_receipt([
            item["structural_review_sha256"] for item in input_receipt["shards"]
        ]),
        "nli_verification": _sealed_receipt([
            item["nli_output_sha256"] for item in input_receipt["shards"]
        ]),
        "two_pass_semantic_judges": _sealed_receipt([
            item["semantic_output_sha256"] for item in input_receipt["shards"]
        ]),
        "selector": {"status": "sealed_passed", "content_sha256": selector_receipt_sha},
        "manual_review": {
            "status": "sealed_passed",
            "content_sha256": manual_receipt["content_sha256_before_self_field"],
        },
        "tokenizer_chat_template": _sealed_receipt({
            "tokenizer_json_sha256": corpus.TOKENIZER_JSON_SHA256,
            "tokenizer_config_sha256": corpus.TOKENIZER_CONFIG_SHA256,
            "assistant_mask_method": corpus.ASSISTANT_MASK_METHOD,
            "maximum_rendered_chat_tokens": MAX_RENDERED_CHAT_TOKENS,
            "length_gate_receipt_sha256": token_length_receipt[
                "content_sha256_before_self_field"
            ],
        }),
    }
    if set(receipts) != GENERATED_RECEIPT_KEYS:
        raise RuntimeError("generated-domain authority receipt coverage changed")
    report = {
        "schema": REPORT_SCHEMA,
        "status": "sealed_verified_training_authority",
        "rows": len(rows),
        "accounting": accounting,
        "selection": selection,
        "seed_qa_replacement": seed_qa_replacement,
        "rendered_chat_length_gate": token_length_receipt,
        "all_rows_eligible_for_training": True,
        "exact_atomic_selection_solved": True,
        "deficit_tokens": 0,
        "padding_used": False,
        "truncation_used": False,
        "borrowing_used": False,
        "training_launch_authorized_by_this_authority": True,
    }
    report["content_sha256_before_self_field"] = _self_address(report)
    report_payload = _json_payload(report)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "sealed_verified_training_authority",
        "eligible_for_training": True,
        "training_launch_authorized_by_this_authority": True,
        "selection_contract": {
            "path": corpus.relative(SELECTION_CONTRACT),
            "file_sha256": SELECTION_CONTRACT_FILE_SHA256,
            "content_sha256": SELECTION_CONTRACT_SELF_SHA256,
        },
        "dataset": {
            "path": corpus.relative(TRAIN_OUTPUT),
            "file_sha256": corpus.sha256_bytes(payload),
            "rows": len(rows),
            "assistant_qwen36_tokens": accounting["assistant_tokens"],
            "schema": ROW_SCHEMA,
        },
        "report": {
            "path": corpus.relative(REPORT_OUTPUT),
            "file_sha256": corpus.sha256_bytes(report_payload),
            "content_sha256": report["content_sha256_before_self_field"],
        },
        "accounting": accounting,
        "seed_qa_replacement": seed_qa_replacement,
        "rendered_chat_length_gate": {
            "content_sha256": token_length_receipt[
                "content_sha256_before_self_field"
            ],
            "maximum_rendered_chat_tokens": MAX_RENDERED_CHAT_TOKENS,
            "candidates": token_length_receipt["candidates"],
            "passed": token_length_receipt["passed"],
            "excluded_overlength": token_length_receipt["excluded_overlength"],
            "exclusion_commitment_sha256": token_length_receipt[
                "exclusion_commitment_sha256"
            ],
        },
        "receipts": receipts,
        "implementation_receipts": _implementation_receipts(),
    }
    manifest["content_sha256_before_self_field"] = _self_address(manifest)
    return payload, report, manifest


def _require_canonical_authority_absent() -> None:
    existing = [
        path.resolve().as_posix()
        for path in (TRAIN_OUTPUT, REPORT_OUTPUT, MANIFEST_OUTPUT)
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise RuntimeError(
            "sealed generated-domain authority already exists: "
            + ", ".join(existing)
        )


def _write_immutable_authority_artifact(path: Path, payload: bytes) -> None:
    """Atomically create one canonical artifact without replacement semantics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Linking a complete, fsynced inode is atomic and fails with
            # EEXIST rather than replacing an authority created concurrently.
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(
                f"immutable generated-domain authority already exists: {path}"
            ) from error
        directory = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def build_global_authority(*, write: bool) -> dict:
    # Canonical authority artifacts are immutable.  Refuse every mutating
    # invocation before inspecting current semantic inputs when any prior or
    # partial authority path exists; in particular, never leave an old train
    # dataset looking current while writing a new blocked/deficit report.
    if write:
        _require_canonical_authority_absent()
    scaffold = build_scaffold()
    if not scaffold["selection_may_run"]:
        if write:
            corpus.atomic_write(SELECTION_SCAFFOLD, _json_payload(scaffold))
        return scaffold
    contract = load_selection_contract()
    targets = selection_targets(contract)
    category_map = source_categories(contract)
    candidates, input_receipt = load_all_semantic_candidates()
    pool, queue, manual_receipt, token_length_receipt = prepare_global_pool(
        candidates,
        category_map=category_map,
        manual_resolution_rows=load_manual_resolution_rows(),
    )
    seed_qa_replacement_contract = scaffold["seed_qa_replacement"]["contract"]
    if not isinstance(seed_qa_replacement_contract, dict):
        raise RuntimeError("seed-QA replacement contract disappeared")
    validate_seed_qa_replacement_contract(seed_qa_replacement_contract)
    base_selection = solve_atomic_selection(pool, targets)
    if not base_selection["exact_solution"]:
        report = build_deficit_report(
            selection=base_selection,
            pool=pool,
            queue=queue,
            input_receipt=input_receipt,
            token_length_receipt=token_length_receipt,
        )
        report["seed_qa_replacement_contract"] = seed_qa_replacement_contract
        report["seed_qa_replacement_status"] = (
            "not_attempted_base_atomic_selection_unsolved"
        )
        report["content_sha256_before_self_field"] = _self_address(report)
        if write:
            corpus.atomic_write(SELECTION_SCAFFOLD, _json_payload(scaffold))
            corpus.atomic_write(DEFICIT_OUTPUT, _json_payload(report))
            corpus.atomic_write(MANUAL_QUEUE_OUTPUT, corpus.jsonl_payload(queue))
        return report
    replacement_selection = solve_seed_replacement_selection(
        pool,
        base_selection,
        seed_qa_replacement_contract[
            "replacement_assistant_tokens_required"
        ],
    )
    if not replacement_selection["exact_solution"]:
        base_selection, replacement_selection = (
            solve_coupled_seed_replacement_fallback(
                pool,
                targets,
                base_selection,
                seed_qa_replacement_contract[
                    "replacement_assistant_tokens_required"
                ],
            )
        )
    selection = combine_base_and_seed_replacement_selection(
        pool,
        base_selection,
        replacement_selection,
        base_targets=targets,
    )
    if not selection["exact_solution"]:
        report = build_deficit_report(
            selection=selection,
            pool=pool,
            queue=queue,
            input_receipt=input_receipt,
            token_length_receipt=token_length_receipt,
        )
        report["seed_qa_replacement_contract"] = (
            seed_qa_replacement_contract
        )
        report["seed_qa_replacement_deficit"] = (
            build_seed_qa_replacement_deficit_receipt(
                seed_qa_replacement_contract,
                pool,
                replacement_selection,
            )
        )
        report["content_sha256_before_self_field"] = _self_address(report)
        if write:
            corpus.atomic_write(SELECTION_SCAFFOLD, _json_payload(scaffold))
            corpus.atomic_write(DEFICIT_OUTPUT, _json_payload(report))
            corpus.atomic_write(MANUAL_QUEUE_OUTPUT, corpus.jsonl_payload(queue))
        return report
    # Recompute the independent seed authority after candidate loading and both
    # solver stages.  A concurrent or accidental authority mutation must not be
    # covered by a receipt captured earlier in this long-running build.
    if load_seed_qa_replacement_contract() != seed_qa_replacement_contract:
        raise RuntimeError("seed-QA replacement authority changed during selection")
    seed_qa_replacement = build_seed_qa_replacement_receipt(
        seed_qa_replacement_contract,
        pool,
        replacement_selection,
    )
    payload, report, manifest = build_final_authority(
        pool=pool,
        selection=selection,
        input_receipt=input_receipt,
        manual_receipt=manual_receipt,
        token_length_receipt=token_length_receipt,
        selection_contract=contract,
        seed_qa_replacement=seed_qa_replacement,
    )
    if write:
        _write_immutable_authority_artifact(TRAIN_OUTPUT, payload)
        _write_immutable_authority_artifact(REPORT_OUTPUT, _json_payload(report))
        _write_immutable_authority_artifact(MANIFEST_OUTPUT, _json_payload(manifest))
        corpus.atomic_write(MANUAL_QUEUE_OUTPUT, corpus.jsonl_payload(queue))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--write", action="store_true")
    result.add_argument("--check-scaffold", action="store_true")
    result.add_argument("--print", action="store_true", dest="print_value")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.check_scaffold:
        value = build_scaffold()
        if SELECTION_SCAFFOLD.is_symlink() or not SELECTION_SCAFFOLD.is_file():
            raise RuntimeError("generated-domain selection scaffold is missing")
        if SELECTION_SCAFFOLD.read_bytes() != _json_payload(value):
            raise RuntimeError("generated-domain selection scaffold is stale")
    else:
        value = build_global_authority(write=args.write)
    if args.print_value:
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        print(json.dumps({
            "schema": value["schema"],
            "status": value["status"],
            "content_sha256": value.get("content_sha256_before_self_field"),
            "training_launch_authorized": value.get(
                "training_launch_authorized_by_this_authority",
                value.get("training_launch_authorized", False),
            ),
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
