#!/usr/bin/env python3
"""Assemble the exact, sealed 150k-token general replay authority.

Only explicitly pinned train-only synthetic/reference artifacts and manually
approved rows are eligible.  The builder deliberately has no discovery path:
every input is named here, content addressed, and validated before use.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from build_general_replay_corpus_v1 import load_qwen_tokenizer
from general_replay_v1 import (
    CATEGORY_PERCENT,
    REPLAY_ROW_SCHEMA,
    ROOT,
    assistant_token_count,
    build_reference_compiler_rows,
    canonical_bytes,
    canonical_sha256,
    exact_token_group_subset,
    file_sha256,
    safe_regular_input,
    token_budgets,
    validate_approval_ledger,
    validate_prompt_specs,
    validate_seed_authority,
)
from run_general_replay_candidate_shard_v1 import (
    MODEL_DIRECTORY,
    MODEL_FILE_SHA256,
    MODEL_IDENTITY_SHA256,
    MODEL_NAME,
    MODEL_REVISION,
    validate_model_files,
)


DATA = ROOT / "data/general_replay_v1"
MANUAL = DATA / "manual_approvals"
PRIORITY = DATA / "approval_priority_review_v1_scale32"
SOURCE_PACKET = DATA / "approval_review_packet_v1_scale32"
OUTPUT_DIRECTORY = DATA / "replay_authority_v1_150k"
OUTPUT_ROWS = OUTPUT_DIRECTORY / "general_replay_authority_v1_150k.jsonl"
OUTPUT_REPORT = OUTPUT_DIRECTORY / "report.json"
OUTPUT_MANIFEST = OUTPUT_DIRECTORY / "manifest.json"
TARGET_ASSISTANT_TOKENS = 150_000
BUILD_TIME = "2026-07-18T03:00:00Z"
ASSEMBLER = Path(__file__).resolve()

BASE_SPECS = DATA / "prompt_specs_v2_scale32.jsonl"
BASE_SPEC_REPORT = DATA / "prompt_specs_v2_scale32.report.json"
BASE_ROWS = DATA / "deterministic_reference_rows_v1_scale32.jsonl"
BASE_ROW_REPORT = DATA / "deterministic_reference_rows_v1_scale32.report.json"
SUPPLEMENT_SPECS = DATA / "objective_supplement_prompt_specs_v2r1_150k.jsonl"
SUPPLEMENT_ROWS = DATA / "objective_supplement_reference_rows_v2r1_150k.jsonl"
SUPPLEMENT_REPORT = DATA / "objective_supplement_v2r1_150k.report.json"
PRIORITY_MANIFEST = PRIORITY / "manifest.json"
SOURCE_PACKET_MANIFEST = SOURCE_PACKET / "manifest.json"
BACKUP_MANIFEST = DATA / "ordinary_conversation_backup_queue_v1/manifest.json"
SEED_ROWS = ROOT / "data/general_qa_proxy_anchor_v43h.jsonl"
SEED_REPORT = ROOT / "data/general_qa_proxy_anchor_v43h.report.json"

ORDINARY_LEDGER = MANUAL / "ordinary_conversation.reviewer_conversation.jsonl"
ORDINARY_REPORT = MANUAL / "ordinary_conversation.reviewer_conversation.report.json"
SAFETY_LEDGER = MANUAL / "safety.reviewer_safety.jsonl"
SAFETY_REPORT = MANUAL / "safety.reviewer_safety.report.json"
UNCERTAINTY_LEDGER = MANUAL / "uncertainty.reviewer_uncertainty.jsonl"
UNCERTAINTY_REPORT = MANUAL / "uncertainty.reviewer_uncertainty.report.json"
AUTHORED_SOURCE = MANUAL / "ordinary_conversation.codex_authored_source_v1.json"
AUTHORED_ROWS = MANUAL / "ordinary_conversation.codex_authored_replacements_v1.jsonl"
AUTHORED_REPORT = MANUAL / "ordinary_conversation.codex_authored_replacements_v1.report.json"

PINNED_SHA256 = {
    BASE_SPECS: "d7336146ebf405b3f225725f4e790398dc537ff5153621eefcf9fb27789caceb",
    BASE_SPEC_REPORT: "949013e9ea1048a73307514866d5455d4ee3a2bbfe93dd3d0b9bc726e0d7a736",
    BASE_ROWS: "3410a9a5b694d7dc227d71828beb97928fa720add81dbec23aadf574d428491e",
    BASE_ROW_REPORT: "6deb09f06641a9c67df0e9abe0a3e3ae05206e22bec8e0ad60cdd4084aee2dc1",
    SUPPLEMENT_SPECS: "a994730986b2d83d4c5600d40dd73fc569c057ee5dfdbcf2c797647682346227",
    SUPPLEMENT_ROWS: "dff034543ec921be271fb7afe12d0503c72d77a5ef67f24f899635592cbaa717",
    SUPPLEMENT_REPORT: "c99cd551725fa0a19b8a32343cd24d5c4efd5f564112591373f650c885ed0133",
    PRIORITY_MANIFEST: "489bae35267ae18e44fe8a0e5a596ecce4761b2f4a066fd81c42e954a3f85c3f",
    SOURCE_PACKET_MANIFEST: "1dcd64b81e7bdaeefea6b25ad6ac9124a069db96ef04cbc336fedc9347f1ea78",
    ORDINARY_LEDGER: "493b2ef8f0c0a63564aef903681d22df06bd33630d6993af878814506eedc2c8",
    ORDINARY_REPORT: "150e1b5ddee8dbd4bd97197eb29884ce905cf0c5c7e3b4d92f615d3268c83c26",
    SAFETY_LEDGER: "1c0a2e25f1306de9fabf1e3aae93118a7844fc6416c9d69c2ec6b52b4e515e81",
    SAFETY_REPORT: "0136516c588673c66aa556f7dcf297d3c4d13a4ed263c234c19ab20547bf3b87",
    UNCERTAINTY_LEDGER: "5a71cc739a9bb57fdd461cac6645e0dd8904498d75a062f947651760b966576e",
    UNCERTAINTY_REPORT: "dd6c719e1a0092acd8637292efbeccdae9c8cda1d596ef7ab63a6e2460b0ae97",
    AUTHORED_SOURCE: "79e64714b009624568b6afda61705c1afd88fada1d57dd63ca36f88bd0405e70",
    AUTHORED_ROWS: "ee9741e6006d938d39cf84999655047fb9627ff6460a0dce02b7b5c9c3ca30e6",
    AUTHORED_REPORT: "2a5c8df3f991c87f5660dc007ea5c8e65c520c13e57121ffe569124c1cc100b9",
    BACKUP_MANIFEST: "975210502f96728ad997049a11598a4496760b85ac57a0fb2021b14413cd4d40",
    SEED_ROWS: "d250980c5b88308452aba4ee8d3e43090b1444c250547c2515c838593f2f391f",
    SEED_REPORT: "b176f4ea0d1d4b466c83b68c243bfb4a9e315df8574d9ba2c606333fbfc4ee8f",
}

REPORT_SELF_HASHES = {
    BASE_SPEC_REPORT: "7a629a26b74fbef4b1cccf5d1089a084b5d9c09855e861a4d4900b1864021566",
    BASE_ROW_REPORT: "adbdf323a43c5e5c4541fc2e133e21506f1ba2834417b26982d79e87318a0f0f",
    SUPPLEMENT_REPORT: "6c67d16e4c2155750fa86fa0fcc39f828ace2ac9610c9e5bab354b386bda4516",
    PRIORITY_MANIFEST: "1d933ee89f48d225838f32016d98a7bbb64445e63f74d6ebe6d8e4f37e14ba52",
    SOURCE_PACKET_MANIFEST: "d11abba66b36919b0df8e12df0a54905834fb5e407b04f2129f00a00ae4f4709",
    SAFETY_REPORT: "8fa0678042294801a10b269f018ad8a690ff9163cae829da54c2b3f0d3892db1",
    UNCERTAINTY_REPORT: "1ddf4b941fda533dc0c434a1801b9f10eab8c9f6d0f3f7d40dd6ffef4a7b77af",
    AUTHORED_REPORT: "05631af283e0cda1a9faf049a00d541f437ba20a6b59470209922ee5d8175738",
    BACKUP_MANIFEST: "51eb5508ab208ce910f91acbe5756c59beb20ca1292f6d4c3d674ba79a8de1cb",
}

SUPERSEDED = {
    DATA / "objective_supplement_prompt_specs_v1_150k.jsonl": "38b979b7abcfa3028ee7f2038f4054081237f514462eba9ea6d95cb051600935",
    DATA / "objective_supplement_reference_rows_v1_150k.jsonl": "2dc5e79099978c0050afda59d10efbdf43e1cb1843d44e7f45441d99ae644206",
    DATA / "objective_supplement_v1_150k.report.json": "3720241f23851ea7281859027c28e246d29f7d5f6cda62cefb1e6950c915be4e",
    DATA / "objective_supplement_prompt_specs_v2_150k.jsonl": "498419b5fd0b006dc8f68d9fad971dc6c90ab240f8d0a475295160549e6b183f",
    DATA / "objective_supplement_reference_rows_v2_150k.jsonl": "f1c8ec155cf19a2dc41a60fb472956f1af40aa2a09c41916057141e3da40b6cf",
    DATA / "objective_supplement_v2_150k.report.json": "b0939dadaf39d5665f0477c6af1647006671acf7d337471982f1ae801ef106c2",
}

OBJECTIVE_FULL_CATEGORIES = (
    "coding_debugging",
    "mathematical_reasoning",
    "json_structured_data",
    "instruction_following",
    "multilingual",
)
MANUAL_CONFIG = {
    "ordinary_conversation": {
        "ledger": ORDINARY_LEDGER,
        "report": ORDINARY_REPORT,
        "report_schema": "general-replay-manual-review-report-v1",
        "report_self": None,
        "rubric": "ordinary-conversation-warm-practical-v1",
        "reviewer": "codex-manual-conversation-reviewer",
        "reviewed": 83,
        "approved": 28,
        "rejected": 55,
        "approved_tokens": 5_053,
    },
    "safety_refusal": {
        "ledger": SAFETY_LEDGER,
        "report": SAFETY_REPORT,
        "report_schema": "general-replay-safety-manual-review-report-v1",
        "report_self": REPORT_SELF_HASHES[SAFETY_REPORT],
        "rubric": "safe-boundary-constructive-alternative-v1",
        "reviewer": "reviewer_safety",
        "reviewed": 31,
        "approved": 31,
        "rejected": 0,
        "approved_tokens": 7_500,
    },
    "uncertainty_hallucination_resistance": {
        "ledger": UNCERTAINTY_LEDGER,
        "report": UNCERTAINTY_REPORT,
        "report_schema": "general-replay-uncertainty-manual-review-report-v1",
        "report_self": REPORT_SELF_HASHES[UNCERTAINTY_REPORT],
        "rubric": "uncertainty-no-fabrication-v1",
        "reviewer": "reviewer_uncertainty",
        "reviewed": 35,
        "approved": 35,
        "rejected": 0,
        "approved_tokens": 7_500,
    },
}

TOP_LEVEL_ROW_KEYS = {
    "schema", "row_id", "category", "source_group_id", "messages", "tools",
    "assistant_mask", "assistant_token_count", "lineage", "generator",
    "verifier", "template_policy",
}
ASSISTANT_MASK = {
    "policy": "assistant_only_v1",
    "system_tokens": False,
    "user_tokens": False,
    "tool_result_tokens": False,
}
ALLOWED_SOURCE_KINDS = {
    "deterministic_reference_compiler_v1",
    "approved_proxy_anchor_v43h",
    "manual_approved_base_model_generation_v1",
    "codex_manually_authored_and_reviewed",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CHUNK_NAME = re.compile(
    r"^(ordinary_conversation|safety_refusal|"
    r"uncertainty_hallucination_resistance)\.priority-[0-9]{3}\.jsonl$"
)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _pinned_bytes(path: Path, expected_sha256: str, role: str) -> bytes:
    if not HEX64.fullmatch(expected_sha256):
        raise ValueError(f"{role}: malformed pinned digest")
    safe = safe_regular_input(path, role, exact=path)
    raw = safe.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError(f"{role}: content identity changed")
    return raw


def _load_jsonl(raw: bytes, role: str) -> list[dict]:
    rows = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{role} line {line_number}: blank rows forbidden")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{role} line {line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{role} line {line_number}: object required")
        rows.append(value)
    if not rows:
        raise ValueError(f"{role}: empty JSONL forbidden")
    return rows


def _load_pinned_jsonl(path: Path, role: str) -> tuple[list[dict], bytes]:
    raw = _pinned_bytes(path, PINNED_SHA256[path], role)
    return _load_jsonl(raw, role), raw


def _load_pinned_json(path: Path, role: str) -> tuple[dict, bytes]:
    raw = _pinned_bytes(path, PINNED_SHA256[path], role)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{role}: object required")
    return value, raw


def validate_self_hash(value: dict, expected: str, role: str) -> None:
    actual_field = value.get("content_sha256_before_self_field")
    if actual_field != expected:
        raise RuntimeError(f"{role}: declared self identity changed")
    payload = dict(value)
    payload.pop("content_sha256_before_self_field")
    if canonical_sha256(payload) != expected:
        raise RuntimeError(f"{role}: self hash failed")


def _canonical_jsonl(rows: list[dict]) -> bytes:
    return b"".join(canonical_bytes(row) for row in rows)


def _rows_equal_by_identity(left: list[dict], right: list[dict]) -> bool:
    """Compare sealed row content while treating JSONL order as non-semantic."""
    left_by_id = {row.get("row_id"): row for row in left}
    right_by_id = {row.get("row_id"): row for row in right}
    return (
        None not in left_by_id
        and None not in right_by_id
        and len(left_by_id) == len(left)
        and len(right_by_id) == len(right)
        and left_by_id == right_by_id
    )


def _validate_tokenizer_contract(report: dict, role: str) -> None:
    tokenizer = report.get("tokenizer")
    if not isinstance(tokenizer, dict) or (
        tokenizer.get("path") != str(MODEL_DIRECTORY)
        or tokenizer.get("revision") != MODEL_REVISION
        or tokenizer.get("chat_template_sha256")
        != MODEL_FILE_SHA256["chat_template.jinja"]
    ):
        raise RuntimeError(f"{role}: tokenizer contract changed")


def load_objective_authorities(tokenizer: Any) -> tuple[list[dict], list[dict], dict]:
    base_specs, _ = _load_pinned_jsonl(BASE_SPECS, "base prompt specs")
    base_rows, base_row_raw = _load_pinned_jsonl(BASE_ROWS, "base reference rows")
    base_spec_report, _ = _load_pinned_json(BASE_SPEC_REPORT, "base prompt report")
    base_row_report, _ = _load_pinned_json(BASE_ROW_REPORT, "base reference report")
    validate_self_hash(
        base_spec_report, REPORT_SELF_HASHES[BASE_SPEC_REPORT], "base prompt report"
    )
    validate_self_hash(
        base_row_report, REPORT_SELF_HASHES[BASE_ROW_REPORT], "base reference report"
    )
    validate_prompt_specs(base_specs)
    if (
        base_spec_report.get("schema") != "general-replay-prompt-spec-report-v1"
        or base_spec_report.get("rows") != 2_560
        or base_spec_report.get("output_sha256") != PINNED_SHA256[BASE_SPECS]
        or base_spec_report.get("policy", {}).get("direct_benchmark_prompts") is not False
        or base_spec_report.get("policy", {}).get("protected_or_evaluation_sources_opened") is not False
    ):
        raise RuntimeError("base prompt authority changed")
    if (
        base_row_report.get("schema")
        != "general-replay-deterministic-reference-report-v1"
        or base_row_report.get("rows") != 1_800
        or base_row_report.get("assistant_tokens") != 87_160
        or base_row_report.get("output_sha256") != PINNED_SHA256[BASE_ROWS]
        or base_row_report.get("policy", {}).get("base_generations_promoted") is not False
        or base_row_report.get("policy", {}).get("subjective_rows_compiled") is not False
    ):
        raise RuntimeError("base reference authority changed")
    _validate_tokenizer_contract(base_row_report, "base reference report")
    rebuilt_base, rebuilt_base_audit = build_reference_compiler_rows(base_specs, tokenizer)
    if _canonical_jsonl(rebuilt_base) != base_row_raw or rebuilt_base != base_rows:
        raise RuntimeError("base deterministic references no longer recompile exactly")
    if rebuilt_base_audit["assistant_tokens"] != 87_160:
        raise RuntimeError("base deterministic reference accounting changed")

    supplement_specs, _ = _load_pinned_jsonl(
        SUPPLEMENT_SPECS, "v2r1 supplement prompt specs"
    )
    supplement_rows, _ = _load_pinned_jsonl(
        SUPPLEMENT_ROWS, "v2r1 supplement reference rows"
    )
    supplement_report, _ = _load_pinned_json(
        SUPPLEMENT_REPORT, "v2r1 supplement report"
    )
    validate_self_hash(
        supplement_report, REPORT_SELF_HASHES[SUPPLEMENT_REPORT],
        "v2r1 supplement report",
    )
    validate_prompt_specs(supplement_specs)
    outputs = supplement_report.get("outputs", {})
    policy = supplement_report.get("policy", {})
    if (
        supplement_report.get("schema")
        != "general-replay-objective-supplement-report-v2"
        or supplement_report.get("revision") != "v2r1"
        or supplement_report.get("all_diversity_gates_pass") is not True
        or supplement_report.get("all_scoped_category_targets_exact") is not True
        or outputs.get("prompt_specs") != {
            "rows": 1_915, "sha256": PINNED_SHA256[SUPPLEMENT_SPECS]
        }
        or outputs.get("reference_rows") != {
            "rows": 1_915,
            "assistant_tokens": 37_015,
            "sha256": PINNED_SHA256[SUPPLEMENT_ROWS],
        }
        or policy.get("deterministic_synthetic_sources_only") is not True
        or policy.get("direct_benchmark_prompts") is not False
        or policy.get("protected_or_evaluation_sources_opened") is not False
        or policy.get("base_generations_promoted") is not False
    ):
        raise RuntimeError("v2r1 supplement authority changed")
    _validate_tokenizer_contract(supplement_report, "v2r1 supplement report")
    rebuilt_supplement, rebuilt_supplement_audit = build_reference_compiler_rows(
        supplement_specs, tokenizer
    )
    # The supplement builder preserves exact-subset selection order, whereas
    # the reference compiler emits spec-id order.  The stored file remains
    # hash pinned; recompilation must reproduce every row exactly by row_id.
    if (
        not _rows_equal_by_identity(rebuilt_supplement, supplement_rows)
        or rebuilt_supplement_audit["assistant_tokens"] != 37_015
    ):
        raise RuntimeError("v2r1 deterministic references no longer recompile exactly")
    return base_rows, supplement_rows, {
        "base_specs": base_specs,
        "base_reference_compiler": rebuilt_base_audit,
        "supplement_reference_compiler": rebuilt_supplement_audit,
    }


def select_objective_rows(
    base_rows: list[dict], supplement_rows: list[dict], seed_rows: list[dict]
) -> tuple[list[dict], dict[str, list[dict]]]:
    components: dict[str, list[dict]] = {}
    components["base_reference_full_categories"] = [
        row for row in base_rows if row["category"] in OBJECTIVE_FULL_CATEGORIES
    ]
    components["base_reference_tool_exact_subset"] = exact_token_group_subset(
        [row for row in base_rows if row["category"] == "tool_use_function_calling"],
        15_000,
    )
    components["base_reference_long_context_exact_subset"] = exact_token_group_subset(
        [row for row in base_rows if row["category"] == "long_context"], 7_500
    )
    components["v2r1_diversified_objective_supplement"] = list(supplement_rows)
    components["mandatory_train_only_proxy_seed"] = list(seed_rows)
    rows = [row for component in components.values() for row in component]
    return rows, components


def _validate_priority_item(item: dict, spec_by_id: dict[str, dict], role: str) -> dict:
    expected = {
        "schema", "spec_id", "source_group_id", "prompt_identity_sha256",
        "category", "messages", "tools", "expected_response_format",
        "rubric_id", "required_approvals", "candidates", "preselection",
    }
    if not isinstance(item, dict) or set(item) != expected:
        raise ValueError(f"{role}: review item schema changed")
    if item["schema"] != "general-replay-approval-review-item-v1":
        raise ValueError(f"{role}: review item version changed")
    spec = spec_by_id.get(item["spec_id"])
    if spec is None:
        raise RuntimeError(f"{role}: review item is not a pinned prompt spec")
    rubric = spec["verifier"]
    if rubric["type"] != "approval_required_v1":
        raise RuntimeError(f"{role}: objective prompt entered manual review")
    if (
        item["source_group_id"] != spec["source_group_id"]
        or item["prompt_identity_sha256"] != spec["prompt_identity_sha256"]
        or item["category"] != spec["category"]
        or item["messages"] != spec["messages"]
        or item["tools"] != spec["tools"]
        or item["expected_response_format"] != spec["expected_response_format"]
        or item["rubric_id"] != rubric["config"]["rubric_id"]
        or item["required_approvals"] != rubric["config"]["required_approvals"]
        or item["preselection"] != {
            "policy": "deterministic_obvious_miss_filter_then_minimum_rows_v1",
            "quality_status": "pending_human_review",
            "auto_approved": False,
        }
    ):
        raise RuntimeError(f"{role}: prompt or rubric lineage changed")
    if not isinstance(item["candidates"], list) or len(item["candidates"]) != 1:
        raise RuntimeError(f"{role}: priority item must contain one candidate")
    candidate = item["candidates"][0]
    if set(candidate) != {
        "assistant_message", "assistant_token_count", "candidate_index",
        "response_sha256",
    }:
        raise ValueError(f"{role}: candidate schema changed")
    if (
        not HEX64.fullmatch(candidate["response_sha256"])
        or not isinstance(candidate["candidate_index"], int)
        or isinstance(candidate["candidate_index"], bool)
        or not 0 <= candidate["candidate_index"] < 4
        or not isinstance(candidate["assistant_token_count"], int)
        or candidate["assistant_token_count"] <= 0
        or candidate["assistant_message"].get("role") != "assistant"
        or not isinstance(candidate["assistant_message"].get("content"), str)
    ):
        raise ValueError(f"{role}: candidate identity changed")
    return candidate


def load_priority_records(spec_by_id: dict[str, dict]) -> tuple[dict[str, list[dict]], dict, list[dict]]:
    manifest, _ = _load_pinned_json(PRIORITY_MANIFEST, "priority review manifest")
    validate_self_hash(
        manifest, REPORT_SELF_HASHES[PRIORITY_MANIFEST], "priority review manifest"
    )
    source_manifest, _ = _load_pinned_json(
        SOURCE_PACKET_MANIFEST, "source review packet manifest"
    )
    validate_self_hash(
        source_manifest, REPORT_SELF_HASHES[SOURCE_PACKET_MANIFEST],
        "source review packet manifest",
    )
    if (
        manifest.get("schema")
        != "general-replay-priority-approval-review-manifest-v1"
        or manifest.get("policy", {}).get("auto_approvals_created") is not False
        or manifest.get("source_manifest", {}).get("sha256")
        != PINNED_SHA256[SOURCE_PACKET_MANIFEST]
        or manifest.get("source_manifest", {}).get("content_sha256")
        != REPORT_SELF_HASHES[SOURCE_PACKET_MANIFEST]
        or source_manifest.get("schema")
        != "general-replay-approval-review-packet-manifest-v1"
        or source_manifest.get("policy", {}).get(
            "direct_benchmark_or_protected_sources_used"
        ) is not False
        or source_manifest.get("policy", {}).get("auto_approvals_created") is not False
    ):
        raise RuntimeError("manual review packet authority changed")

    by_category: dict[str, list[dict]] = defaultdict(list)
    inputs = []
    seen_paths = set()
    for entry in manifest.get("chunks", []):
        name = entry.get("path")
        if not isinstance(name, str) or CHUNK_NAME.fullmatch(name) is None:
            raise RuntimeError("priority chunk path escaped its sealed namespace")
        if name in seen_paths:
            raise RuntimeError("priority chunk path duplicated")
        seen_paths.add(name)
        path = PRIORITY / name
        raw = _pinned_bytes(path, entry.get("sha256"), f"priority chunk {name}")
        records = _load_jsonl(raw, f"priority chunk {name}")
        if (
            len(records) != entry.get("rows")
            or sum(record["candidates"][0]["assistant_token_count"] for record in records)
            != entry.get("assistant_tokens")
            or any(record.get("category") != entry.get("category") for record in records)
        ):
            raise RuntimeError(f"priority chunk {name}: manifest accounting changed")
        for index, item in enumerate(records):
            _validate_priority_item(item, spec_by_id, f"{name} row {index + 1}")
            by_category[item["category"]].append(item)
        inputs.append({
            "role": "manual_priority_chunk",
            "path": _relative(path),
            "sha256": entry["sha256"],
            "rows": entry["rows"],
            "assistant_tokens": entry["assistant_tokens"],
            "included_as_rows": True,
        })
    expected_categories = set(MANUAL_CONFIG)
    if set(by_category) != expected_categories:
        raise RuntimeError("priority packet category coverage changed")
    for category, records in by_category.items():
        declared = manifest["categories"][category]
        if (
            len(records) != declared["priority_review_rows"]
            or sum(item["candidates"][0]["assistant_token_count"] for item in records)
            != declared["selected_assistant_tokens"]
        ):
            raise RuntimeError(f"{category}: priority manifest accounting changed")
    return dict(by_category), manifest, inputs


def manual_candidate_to_row(
    item: dict, approval: dict, tokenizer: Any, *, ledger_sha256: str,
    priority_manifest_sha256: str,
) -> dict:
    candidate = item["candidates"][0]
    if (
        approval["status"] != "approved"
        or approval["spec_id"] != item["spec_id"]
        or approval["response_sha256"] != candidate["response_sha256"]
        or approval["rubric_id"] != item["rubric_id"]
    ):
        raise RuntimeError("manual approval does not address the sealed candidate")
    messages = [*item["messages"], candidate["assistant_message"]]
    count = assistant_token_count(tokenizer, messages, item["tools"])
    if count != candidate["assistant_token_count"]:
        raise RuntimeError("manually approved candidate token count changed")
    return {
        "schema": REPLAY_ROW_SCHEMA,
        "row_id": "replay-manual-approved-v1-" + candidate["response_sha256"][:24],
        "category": item["category"],
        "source_group_id": item["source_group_id"],
        "messages": messages,
        "tools": item["tools"],
        "assistant_mask": {
            **ASSISTANT_MASK,
            "assistant_message_indices": [len(messages) - 1],
        },
        "assistant_token_count": count,
        "lineage": {
            "source_kind": "manual_approved_base_model_generation_v1",
            "spec_id": item["spec_id"],
            "prompt_identity_sha256": item["prompt_identity_sha256"],
            "response_sha256": candidate["response_sha256"],
            "priority_manifest_sha256": priority_manifest_sha256,
            "approval_ledger_sha256": ledger_sha256,
            "candidate_index": candidate["candidate_index"],
            "direct_benchmark_prompt": False,
            "protected_source_used": False,
        },
        "generator": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "identity_sha256": MODEL_IDENTITY_SHA256,
            "candidate_index": candidate["candidate_index"],
        },
        "verifier": {
            "type": "manual_rubric_review_v1",
            "status": "passed",
            "decision": "approved",
            "rubric_id": approval["rubric_id"],
            "reviewer": approval["reviewer"],
            "reviewed_at": approval["reviewed_at"],
            "reason": approval["reason"],
            "response_sha256": approval["response_sha256"],
        },
        "template_policy": "official_qwen_apply_chat_template_v1",
    }


def _validate_manual_report(category: str, report: dict, config: dict) -> None:
    if config["report_self"] is not None:
        validate_self_hash(report, config["report_self"], f"{category} review report")
    if (
        report.get("schema") != config["report_schema"]
        or report.get("category") != category
        or report.get("rubric_id") != config["rubric"]
        or report.get("reviewer") != config["reviewer"]
        or report.get("counts") != {
            "reviewed": config["reviewed"],
            "approved": config["approved"],
            "rejected": config["rejected"],
        }
    ):
        raise RuntimeError(f"{category}: manual review report changed")
    if category == "ordinary_conversation":
        if report.get("ledger_sha256") != PINNED_SHA256[ORDINARY_LEDGER]:
            raise RuntimeError("ordinary conversation ledger authority changed")
    else:
        if (
            report.get("ledger", {}).get("sha256") != PINNED_SHA256[config["ledger"]]
            or report.get("tokens", {}).get("approved") != config["approved_tokens"]
            or report.get("tokens", {}).get("exact_budget_ready") is not True
            or report.get("policy", {}).get("auto_approvals_created") is not False
        ):
            raise RuntimeError(f"{category}: review approval contract changed")


def load_manual_rows(
    priority_records: dict[str, list[dict]], tokenizer: Any,
    priority_manifest_sha256: str,
) -> tuple[list[dict], dict[str, list[dict]], list[dict]]:
    rows = []
    components: dict[str, list[dict]] = {}
    inputs = []
    for category, config in MANUAL_CONFIG.items():
        approvals, _ = _load_pinned_jsonl(config["ledger"], f"{category} approval ledger")
        report, _ = _load_pinned_json(config["report"], f"{category} review report")
        _validate_manual_report(category, report, config)
        approval_by_response = validate_approval_ledger(approvals)
        records = priority_records[category]
        candidates = [item["candidates"][0] for item in records]
        candidate_hashes = {item["response_sha256"] for item in candidates}
        if set(approval_by_response) != candidate_hashes:
            raise RuntimeError(f"{category}: approval ledger coverage is not exact")
        if len(candidate_hashes) != len(candidates):
            raise RuntimeError(f"{category}: priority response identity duplicated")
        statuses = Counter(item["status"] for item in approvals)
        if statuses != Counter({
            "approved": config["approved"], "rejected": config["rejected"]
        }):
            raise RuntimeError(f"{category}: approval decision counts changed")
        selected = []
        rejected_tokens = 0
        for item in records:
            candidate = item["candidates"][0]
            approval = approval_by_response[candidate["response_sha256"]]
            if (
                approval["spec_id"] != item["spec_id"]
                or approval["rubric_id"] != item["rubric_id"]
                or approval["reviewer"] != config["reviewer"]
            ):
                raise RuntimeError(f"{category}: approval identity changed")
            if approval["status"] == "approved":
                selected.append(manual_candidate_to_row(
                    item, approval, tokenizer,
                    ledger_sha256=PINNED_SHA256[config["ledger"]],
                    priority_manifest_sha256=priority_manifest_sha256,
                ))
            else:
                rejected_tokens += candidate["assistant_token_count"]
        if (
            len(selected) != config["approved"]
            or sum(item["assistant_token_count"] for item in selected)
            != config["approved_tokens"]
        ):
            raise RuntimeError(f"{category}: approved token accounting changed")
        if category == "ordinary_conversation" and rejected_tokens != 9_947:
            raise RuntimeError("ordinary conversation rejected token accounting changed")
        component = "manual_approved_original_" + category
        components[component] = selected
        rows.extend(selected)
        for role, path in (("manual_approval_ledger", config["ledger"]),
                           ("manual_review_report", config["report"])):
            inputs.append({
                "role": role,
                "path": _relative(path),
                "sha256": PINNED_SHA256[path],
                "included_as_rows": role == "manual_approval_ledger",
            })
    return rows, components, inputs


def load_authored_rows(tokenizer: Any) -> tuple[list[dict], list[dict]]:
    source, _ = _load_pinned_json(AUTHORED_SOURCE, "authored conversation source")
    rows, _ = _load_pinned_jsonl(AUTHORED_ROWS, "authored conversation rows")
    report, _ = _load_pinned_json(AUTHORED_REPORT, "authored conversation report")
    validate_self_hash(
        report, REPORT_SELF_HASHES[AUTHORED_REPORT], "authored conversation report"
    )
    if (
        source.get("schema") != "general-replay-codex-authored-source-v1"
        or source.get("category") != "ordinary_conversation"
        or source.get("author") != "codex-manual-conversation-reviewer"
        or source.get("reviewer") != "codex-manual-conversation-reviewer"
        or source.get("lineage") != "codex_manually_authored_and_reviewed"
        or not isinstance(source.get("items"), list)
        or len(source["items"]) != 58
        or report.get("schema")
        != "general-replay-codex-authored-ordinary-conversation-report-v1"
        or report.get("status") != "exact_replacement_budget_ready"
        or report.get("construction_lineage")
        != "codex_manually_authored_and_reviewed"
        or report.get("source_artifact") != {
            "path": _relative(AUTHORED_SOURCE),
            "rows": 58,
            "sha256": PINNED_SHA256[AUTHORED_SOURCE],
        }
        or report.get("output_artifact") != {
            "path": _relative(AUTHORED_ROWS),
            "rows": 58,
            "assistant_tokens": 9_947,
            "sha256": PINNED_SHA256[AUTHORED_ROWS],
        }
    ):
        raise RuntimeError("authored conversation authority changed")
    _validate_tokenizer_contract(report, "authored conversation report")
    source_by_id = {item.get("local_id"): item for item in source["items"]}
    if len(source_by_id) != 58 or None in source_by_id:
        raise RuntimeError("authored source identity duplicated")
    for index, row in enumerate(rows):
        local_id = row.get("lineage", {}).get("local_id")
        item = source_by_id.get(local_id)
        if (
            item is None
            or row.get("category") != "ordinary_conversation"
            or row.get("lineage", {}).get("source_kind")
            != "codex_manually_authored_and_reviewed"
            or row.get("lineage", {}).get("source_artifact_sha256")
            != PINNED_SHA256[AUTHORED_SOURCE]
            or row.get("messages") != [
                {"role": "user", "content": item["user_prompt"]},
                {"role": "assistant", "content": item["assistant_response"]},
            ]
            or row.get("verifier", {}).get("status") != "passed"
            or row.get("verifier", {}).get("reviewer")
            != "codex-manual-conversation-reviewer"
            or assistant_token_count(tokenizer, row["messages"], row["tools"])
            != row.get("assistant_token_count")
        ):
            raise RuntimeError(f"authored conversation row {index + 1} changed")
    if len(rows) != 58 or sum(row["assistant_token_count"] for row in rows) != 9_947:
        raise RuntimeError("authored conversation replacement accounting changed")
    inputs = [
        {"role": "authored_source", "path": _relative(AUTHORED_SOURCE),
         "sha256": PINNED_SHA256[AUTHORED_SOURCE], "included_as_rows": True},
        {"role": "authored_rows", "path": _relative(AUTHORED_ROWS),
         "sha256": PINNED_SHA256[AUTHORED_ROWS], "included_as_rows": True},
        {"role": "authored_report", "path": _relative(AUTHORED_REPORT),
         "sha256": PINNED_SHA256[AUTHORED_REPORT], "included_as_rows": False},
    ]
    return rows, inputs


def validate_final_rows(
    rows: list[dict], tokenizer: Any, expected_budgets: dict[str, int]
) -> dict:
    if not rows:
        raise ValueError("final replay authority cannot be empty")
    if set(expected_budgets) - set(CATEGORY_PERCENT):
        raise ValueError("unsupported expected category")
    row_ids = set()
    groups = set()
    prompt_identities = set()
    tokens = Counter()
    counts = Counter()
    source_kinds = Counter()
    verifier_types = Counter()
    expected_mask_keys = {*ASSISTANT_MASK, "assistant_message_indices"}
    for index, row in enumerate(rows):
        role = f"final row {index + 1}"
        if not isinstance(row, dict) or set(row) != TOP_LEVEL_ROW_KEYS:
            raise ValueError(f"{role}: top-level schema changed")
        if row["schema"] != REPLAY_ROW_SCHEMA:
            raise ValueError(f"{role}: replay row schema changed")
        if row["category"] not in expected_budgets:
            raise RuntimeError(f"{role}: row outside authorized categories")
        if row["row_id"] in row_ids or row["source_group_id"] in groups:
            raise RuntimeError(f"{role}: duplicate row or source group")
        row_ids.add(row["row_id"])
        groups.add(row["source_group_id"])
        if not isinstance(row["messages"], list) or len(row["messages"]) < 2:
            raise ValueError(f"{role}: complete chat required")
        if row["messages"][-1].get("role") != "assistant":
            raise ValueError(f"{role}: final message must be assistant")
        if any(message.get("role") == "assistant" for message in row["messages"][:-1]):
            raise ValueError(f"{role}: only one assistant target is allowed")
        if not isinstance(row["tools"], list):
            raise ValueError(f"{role}: tools must be a list")
        if (
            not isinstance(row["assistant_mask"], dict)
            or set(row["assistant_mask"]) != expected_mask_keys
            or {key: row["assistant_mask"][key] for key in ASSISTANT_MASK}
            != ASSISTANT_MASK
            or row["assistant_mask"]["assistant_message_indices"]
            != [len(row["messages"]) - 1]
        ):
            raise RuntimeError(f"{role}: assistant-only mask changed")
        if row["template_policy"] != "official_qwen_apply_chat_template_v1":
            raise RuntimeError(f"{role}: unofficial chat template")
        count = assistant_token_count(tokenizer, row["messages"], row["tools"])
        if count <= 0 or count != row["assistant_token_count"]:
            raise RuntimeError(f"{role}: assistant token count changed")
        lineage = row["lineage"]
        source_kind = lineage.get("source_kind")
        if (
            source_kind not in ALLOWED_SOURCE_KINDS
            or lineage.get("direct_benchmark_prompt") is not False
            or lineage.get("protected_source_access", False) is not False
            or lineage.get("protected_source_used", False) is not False
        ):
            raise RuntimeError(f"{role}: unauthorized or protected lineage")
        verifier = row["verifier"]
        if verifier.get("status") != "passed" or verifier.get("type") == "approval_required_v1":
            raise RuntimeError(f"{role}: row is not verified")
        if source_kind == "manual_approved_base_model_generation_v1" and (
            verifier.get("type") != "manual_rubric_review_v1"
            or verifier.get("decision") != "approved"
        ):
            raise RuntimeError(f"{role}: generated row lacks manual approval")
        if source_kind == "codex_manually_authored_and_reviewed" and (
            verifier.get("type") != "codex_manual_rubric_review_v1"
            or verifier.get("decision") != "approved"
        ):
            raise RuntimeError(f"{role}: authored row lacks manual approval")
        prompt_identity = canonical_sha256({
            "messages": row["messages"][:-1], "tools": row["tools"]
        })
        if prompt_identity in prompt_identities:
            raise RuntimeError(f"{role}: duplicate prompt content")
        prompt_identities.add(prompt_identity)
        tokens[row["category"]] += count
        counts[row["category"]] += 1
        source_kinds[source_kind] += 1
        verifier_types[verifier["type"]] += 1
    if dict(tokens) != expected_budgets:
        raise RuntimeError(
            f"category assistant-token budgets are not exact: {dict(tokens)}"
        )
    return {
        "rows": len(rows),
        "assistant_tokens": sum(tokens.values()),
        "rows_by_category": dict(sorted(counts.items())),
        "assistant_tokens_by_category": dict(sorted(tokens.items())),
        "rows_by_source_kind": dict(sorted(source_kinds.items())),
        "rows_by_verifier_type": dict(sorted(verifier_types.items())),
        "unique_row_ids": len(row_ids),
        "unique_source_group_ids": len(groups),
        "unique_prompt_contents": len(prompt_identities),
    }


def _component_report(components: dict[str, list[dict]]) -> dict:
    result = {}
    for name, rows in sorted(components.items()):
        by_category = defaultdict(lambda: {"rows": 0, "assistant_tokens": 0})
        for row in rows:
            by_category[row["category"]]["rows"] += 1
            by_category[row["category"]]["assistant_tokens"] += row["assistant_token_count"]
        result[name] = {
            "rows": len(rows),
            "assistant_tokens": sum(row["assistant_token_count"] for row in rows),
            "by_category": dict(sorted(by_category.items())),
        }
    return result


def _input_entry(role: str, path: Path, *, included: bool) -> dict:
    return {
        "role": role,
        "path": _relative(path),
        "sha256": PINNED_SHA256[path],
        "included_as_rows": included,
    }


def _verify_exclusions() -> list[dict]:
    backup, _ = _load_pinned_json(BACKUP_MANIFEST, "unreviewed backup manifest")
    validate_self_hash(
        backup, REPORT_SELF_HASHES[BACKUP_MANIFEST], "unreviewed backup manifest"
    )
    if backup.get("status") != "preserved_unreviewed_not_promoted_after_quality_pivot":
        raise RuntimeError("unreviewed backup exclusion status changed")
    result = [{
        "path": _relative(BACKUP_MANIFEST),
        "sha256": PINNED_SHA256[BACKUP_MANIFEST],
        "reason": "unreviewed_backup_queue_not_promoted",
        "included_as_rows": False,
    }]
    for path, digest in sorted(SUPERSEDED.items(), key=lambda item: str(item[0])):
        _pinned_bytes(path, digest, f"superseded supplement {path.name}")
        result.append({
            "path": _relative(path),
            "sha256": digest,
            "reason": "superseded_by_diversified_v2r1",
            "included_as_rows": False,
        })
    result.append({
        "artifact": "ordinary_conversation_priority_candidates",
        "rows": 55,
        "assistant_tokens": 9_947,
        "reason": "manually_rejected",
        "included_as_rows": False,
    })
    result.append({
        "artifact_classes": [
            "benchmark", "protected", "development", "final_holdout", "ood",
            "terminal", "incident", "unreviewed_generation", "weak_base_generation",
        ],
        "reason": "class_forbidden_by_replay_authority_policy",
        "included_as_rows": False,
    })
    return result


def build_artifacts(tokenizer: Any | None = None) -> tuple[bytes, bytes, bytes, dict, dict]:
    validate_model_files(MODEL_DIRECTORY)
    if tokenizer is None:
        tokenizer = load_qwen_tokenizer(
            str(MODEL_DIRECTORY), MODEL_REVISION,
            MODEL_FILE_SHA256["chat_template.jinja"],
        )
    base_rows, supplement_rows, objective_audit = load_objective_authorities(tokenizer)
    base_specs, _ = _load_pinned_jsonl(BASE_SPECS, "base prompt specs for manual lineage")
    spec_by_id = {item["spec_id"]: item for item in base_specs}

    seed_rows, _ = validate_seed_authority()
    for row in seed_rows:
        row["assistant_token_count"] = assistant_token_count(
            tokenizer, row["messages"], row["tools"]
        )
    if sum(row["assistant_token_count"] for row in seed_rows) != 5_771:
        raise RuntimeError("mandatory seed token accounting changed")
    objective_rows, components = select_objective_rows(
        base_rows, supplement_rows, seed_rows
    )

    priority_records, priority_manifest, priority_inputs = load_priority_records(
        spec_by_id
    )
    manual_rows, manual_components, manual_inputs = load_manual_rows(
        priority_records, tokenizer, PINNED_SHA256[PRIORITY_MANIFEST]
    )
    authored_rows, authored_inputs = load_authored_rows(tokenizer)
    components.update(manual_components)
    components["codex_manually_authored_conversation_replacements"] = authored_rows

    rows = sorted(
        [*objective_rows, *manual_rows, *authored_rows],
        key=lambda row: (row["category"], row["row_id"]),
    )
    budgets = token_budgets(TARGET_ASSISTANT_TOKENS)
    validation = validate_final_rows(rows, tokenizer, budgets)
    if validation["assistant_tokens"] != TARGET_ASSISTANT_TOKENS:
        raise RuntimeError("final replay total is not exact")
    exclusions = _verify_exclusions()
    output_bytes = _canonical_jsonl(rows)
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    assembler_sha256 = file_sha256(ASSEMBLER)
    component_report = _component_report(components)

    report = {
        "schema": "general-replay-final-authority-report-v1",
        "status": "sealed_exact_150k_replay_authority_ready",
        "built_at": BUILD_TIME,
        "beads_task": "specialist-j59.6",
        "target_assistant_tokens": TARGET_ASSISTANT_TOKENS,
        "target_assistant_tokens_by_category": budgets,
        "output": {
            "path": _relative(OUTPUT_ROWS),
            "sha256": output_sha256,
            "rows": validation["rows"],
            "assistant_tokens": validation["assistant_tokens"],
        },
        "validation": validation,
        "components": component_report,
        "objective_reverification": objective_audit,
        "manual_review": {
            "priority_manifest_file_sha256": PINNED_SHA256[PRIORITY_MANIFEST],
            "priority_manifest_content_sha256": priority_manifest[
                "content_sha256_before_self_field"
            ],
            "ordinary_original_approved_rows": 28,
            "ordinary_original_approved_assistant_tokens": 5_053,
            "ordinary_authored_replacement_rows": 58,
            "ordinary_authored_replacement_assistant_tokens": 9_947,
            "ordinary_combined_rows": 86,
            "ordinary_combined_assistant_tokens": 15_000,
            "safety_approved_rows": 31,
            "safety_approved_assistant_tokens": 7_500,
            "uncertainty_approved_rows": 35,
            "uncertainty_approved_assistant_tokens": 7_500,
        },
        "tokenizer": {
            "path": str(MODEL_DIRECTORY),
            "revision": MODEL_REVISION,
            "chat_template_sha256": MODEL_FILE_SHA256["chat_template.jinja"],
            "template_policy": "official_qwen_apply_chat_template_v1",
            "loss_mask_policy": "assistant_only_v1",
        },
        "assembler": {
            "path": _relative(ASSEMBLER),
            "sha256": assembler_sha256,
        },
        "policy": {
            "exact_per_category_and_total_assistant_tokens": True,
            "unique_row_ids": True,
            "unique_source_group_ids": True,
            "official_chat_template_only": True,
            "assistant_output_loss_only": True,
            "deterministic_objective_verifiers_reapplied": True,
            "generated_subjective_rows_require_exact_manual_approval": True,
            "direct_benchmark_prompts": False,
            "protected_development_holdout_ood_terminal_or_incident_sources_opened": False,
            "unreviewed_or_rejected_rows_promoted": False,
            "superseded_supplements_promoted": False,
            "gpu_or_evaluation_work_started": False,
            "fail_closed_on_input_or_contract_drift": True,
        },
        "closure": {
            "specialist-j59.6": "all_authority_gates_passed",
            "ready_for_source_disjoint_contract_extension": True,
            "authorizes_live_training": False,
        },
    }
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()

    inputs = [
        _input_entry("base_prompt_specs", BASE_SPECS, included=False),
        _input_entry("base_prompt_report", BASE_SPEC_REPORT, included=False),
        _input_entry("base_reference_rows", BASE_ROWS, included=True),
        _input_entry("base_reference_report", BASE_ROW_REPORT, included=False),
        _input_entry("v2r1_supplement_specs", SUPPLEMENT_SPECS, included=False),
        _input_entry("v2r1_supplement_rows", SUPPLEMENT_ROWS, included=True),
        _input_entry("v2r1_supplement_report", SUPPLEMENT_REPORT, included=False),
        _input_entry("priority_review_manifest", PRIORITY_MANIFEST, included=False),
        _input_entry("source_review_packet_manifest", SOURCE_PACKET_MANIFEST, included=False),
        _input_entry("mandatory_proxy_seed", SEED_ROWS, included=True),
        _input_entry("mandatory_proxy_seed_report", SEED_REPORT, included=False),
        *priority_inputs,
        *manual_inputs,
        *authored_inputs,
    ]
    inputs.sort(key=lambda item: (item["role"], item["path"]))
    authority_identity = canonical_sha256({
        "output_sha256": output_sha256,
        "report_sha256": report_sha256,
        "inputs": inputs,
        "exclusions": exclusions,
        "assembler_sha256": assembler_sha256,
    })
    manifest = {
        "schema": "general-replay-final-authority-manifest-v1",
        "status": "content_addressed_and_fail_closed",
        "built_at": BUILD_TIME,
        "authority_identity_sha256": authority_identity,
        "assembler": {
            "path": _relative(ASSEMBLER),
            "sha256": assembler_sha256,
        },
        "inputs": inputs,
        "exclusions": exclusions,
        "outputs": {
            "rows": {
                "path": _relative(OUTPUT_ROWS),
                "sha256": output_sha256,
                "rows": validation["rows"],
                "assistant_tokens": validation["assistant_tokens"],
            },
            "report": {
                "path": _relative(OUTPUT_REPORT),
                "sha256": report_sha256,
                "content_sha256": report["content_sha256_before_self_field"],
            },
        },
        "policy": report["policy"],
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return output_bytes, report_bytes, manifest_bytes, report, manifest


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--check", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_bytes, report_bytes, manifest_bytes, report, manifest = build_artifacts()
    expected = {
        OUTPUT_ROWS: output_bytes,
        OUTPUT_REPORT: report_bytes,
        OUTPUT_MANIFEST: manifest_bytes,
    }
    if args.check:
        for path, content in expected.items():
            safe = safe_regular_input(path, f"sealed output {path.name}", exact=path)
            if safe.read_bytes() != content:
                raise RuntimeError(f"sealed replay authority changed: {path.name}")
        return 0
    if any(path.exists() for path in expected):
        raise FileExistsError("replay authority build requires fresh outputs")
    for path, content in expected.items():
        _atomic_write(path, content)
    print(json.dumps({
        "output": _relative(OUTPUT_ROWS),
        "output_sha256": report["output"]["sha256"],
        "report": _relative(OUTPUT_REPORT),
        "report_sha256": manifest["outputs"]["report"]["sha256"],
        "manifest": _relative(OUTPUT_MANIFEST),
        "manifest_content_sha256": manifest["content_sha256_before_self_field"],
        "rows": report["output"]["rows"],
        "assistant_tokens": report["output"]["assistant_tokens"],
        "status": report["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
