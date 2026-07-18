#!/usr/bin/env python3
"""Validate the response-by-response uncertainty ledger and sealed report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from general_replay_v1 import (
    ROOT,
    canonical_sha256,
    safe_regular_input,
    validate_approval_ledger,
)


CATEGORY = "uncertainty_hallucination_resistance"
RUBRIC_ID = "uncertainty-no-fabrication-v1"
REVIEWER = "reviewer_uncertainty"
TARGET_ASSISTANT_TOKENS = 7_500
EXPECTED_ROWS = 35
EXPECTED_CHUNKS = 5
REPLACEMENT_PROVENANCE = "codex_manually_authored_and_reviewed"
DEFAULT_PACKET = (
    ROOT / "data/general_replay_v1/approval_priority_review_v1_scale32"
)
DEFAULT_LEDGER = (
    ROOT
    / "data/general_replay_v1/manual_approvals/uncertainty.reviewer_uncertainty.jsonl"
)
DEFAULT_REPORT = (
    ROOT
    / "data/general_replay_v1/manual_approvals/uncertainty.reviewer_uncertainty.report.json"
)
HEX64 = re.compile(r"[0-9a-f]{64}")
RFC3339_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _load_json(path: Path, label: str) -> tuple[bytes, dict]:
    checked = safe_regular_input(path, label)
    raw = checked.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return raw, value


def _validate_self_hash(value: dict, schema: str, label: str) -> None:
    if value.get("schema") != schema:
        raise RuntimeError(f"{label} schema changed")
    claimed = value.get("content_sha256_before_self_field")
    unsigned = dict(value)
    unsigned.pop("content_sha256_before_self_field", None)
    if claimed != canonical_sha256(unsigned):
        raise RuntimeError(f"{label} content identity changed")


def _load_source_candidates(priority_manifest: dict) -> dict[str, dict]:
    source_entry = priority_manifest.get("source_manifest")
    if not isinstance(source_entry, dict):
        raise RuntimeError("priority source-manifest lineage missing")
    source_path = Path(source_entry["path"])
    source_raw, source_manifest = _load_json(
        source_path, "source approval-packet manifest"
    )
    _validate_self_hash(
        source_manifest,
        "general-replay-approval-review-packet-manifest-v1",
        "source approval-packet manifest",
    )
    if (
        hashlib.sha256(source_raw).hexdigest() != source_entry.get("sha256")
        or source_manifest["content_sha256_before_self_field"]
        != source_entry.get("content_sha256")
        or source_manifest.get("policy", {}).get("auto_approvals_created")
        is not False
        or source_manifest.get("policy", {}).get(
            "direct_benchmark_or_protected_sources_used"
        )
        is not False
    ):
        raise RuntimeError("source approval-packet lineage changed")

    candidates: dict[str, dict] = {}
    source_directory = source_path.parent
    for chunk in source_manifest["chunks"]:
        if chunk["category"] != CATEGORY:
            continue
        chunk_path = safe_regular_input(
            source_directory / chunk["path"], "source uncertainty review chunk"
        )
        raw = chunk_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != chunk["sha256"]:
            raise RuntimeError("source uncertainty review chunk changed")
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            for candidate in item["candidates"]:
                digest = candidate["response_sha256"]
                if digest in candidates:
                    raise RuntimeError("source uncertainty response identity duplicated")
                candidates[digest] = {
                    "candidate": candidate,
                    "spec_id": item["spec_id"],
                    "source_group_id": item["source_group_id"],
                    "prompt_identity_sha256": item["prompt_identity_sha256"],
                    "rubric_id": item["rubric_id"],
                }
    return candidates


def _load_priority_rows(packet: Path) -> tuple[bytes, dict, list[dict], list[dict]]:
    manifest_raw, manifest = _load_json(packet / "manifest.json", "priority manifest")
    _validate_self_hash(
        manifest,
        "general-replay-priority-approval-review-manifest-v1",
        "priority manifest",
    )
    category_contract = manifest.get("categories", {}).get(CATEGORY)
    if category_contract != {
        "target_assistant_tokens": TARGET_ASSISTANT_TOKENS,
        "priority_review_rows": EXPECTED_ROWS,
        "priority_review_chunks": EXPECTED_CHUNKS,
        "selected_assistant_tokens": TARGET_ASSISTANT_TOKENS,
        "exact_token_target_reachable_if_all_selected_rows_are_approved": True,
        "quality_status": "pending_human_review",
    }:
        raise RuntimeError("uncertainty priority-review contract changed")
    if manifest.get("policy", {}).get("auto_approvals_created") is not False:
        raise RuntimeError("priority packet no longer forbids auto-approval")

    source_candidates = _load_source_candidates(manifest)
    rows: list[dict] = []
    pinned_chunks: list[dict] = []
    for entry in manifest["chunks"]:
        if entry["category"] != CATEGORY:
            continue
        path = safe_regular_input(
            packet / entry["path"], "priority uncertainty chunk"
        )
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise RuntimeError("priority uncertainty chunk changed")
        chunk_rows = [
            json.loads(line) for line in raw.decode().splitlines() if line.strip()
        ]
        if (
            len(chunk_rows) != entry["rows"]
            or sum(
                item["candidates"][0]["assistant_token_count"]
                for item in chunk_rows
            )
            != entry["assistant_tokens"]
        ):
            raise RuntimeError("priority uncertainty chunk accounting changed")
        rows.extend(chunk_rows)
        pinned_chunks.append({
            "path": entry["path"],
            "sha256": entry["sha256"],
            "rows": entry["rows"],
            "assistant_tokens": entry["assistant_tokens"],
        })

    if len(rows) != EXPECTED_ROWS or len(pinned_chunks) != EXPECTED_CHUNKS:
        raise RuntimeError("priority uncertainty coverage changed")
    seen_specs: set[str] = set()
    seen_groups: set[str] = set()
    seen_responses: set[str] = set()
    tokens = 0
    for index, item in enumerate(rows):
        if (
            item.get("schema") != "general-replay-approval-review-item-v1"
            or item.get("category") != CATEGORY
            or item.get("rubric_id") != RUBRIC_ID
            or item.get("preselection", {}).get("auto_approved") is not False
        ):
            raise RuntimeError(f"priority uncertainty row {index} contract changed")
        candidates = item.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise RuntimeError(
                "each priority uncertainty row must address one response"
            )
        candidate = candidates[0]
        digest = candidate.get("response_sha256")
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise RuntimeError("priority uncertainty response identity malformed")
        if (
            item["spec_id"] in seen_specs
            or item["source_group_id"] in seen_groups
            or digest in seen_responses
        ):
            raise RuntimeError("priority uncertainty identity duplicated")
        source = source_candidates.get(digest)
        if source is None or source != {
            "candidate": candidate,
            "spec_id": item["spec_id"],
            "source_group_id": item["source_group_id"],
            "prompt_identity_sha256": item["prompt_identity_sha256"],
            "rubric_id": item["rubric_id"],
        }:
            raise RuntimeError("priority response does not match sealed source packet")
        seen_specs.add(item["spec_id"])
        seen_groups.add(item["source_group_id"])
        seen_responses.add(digest)
        tokens += candidate["assistant_token_count"]
    if tokens != TARGET_ASSISTANT_TOKENS:
        raise RuntimeError("priority uncertainty token total changed")
    return manifest_raw, manifest, rows, pinned_chunks


def build_report(packet: Path, ledger: Path) -> dict:
    manifest_raw, manifest, rows, pinned_chunks = _load_priority_rows(packet)
    ledger_path = safe_regular_input(ledger, "manual uncertainty approval ledger")
    ledger_raw = ledger_path.read_bytes()
    ledger_rows = [
        json.loads(line) for line in ledger_raw.decode().splitlines() if line.strip()
    ]
    indexed = validate_approval_ledger(ledger_rows)
    expected = {
        item["candidates"][0]["response_sha256"]: item for item in rows
    }
    if set(indexed) != set(expected):
        raise RuntimeError(
            "manual uncertainty ledger must decide every priority response once"
        )

    reviewed_at_values: set[str] = set()
    decisions: list[dict] = []
    approved_rows = 0
    rejected_rows = 0
    approved_tokens = 0
    rejected_tokens = 0
    for item in rows:
        candidate = item["candidates"][0]
        digest = candidate["response_sha256"]
        decision = indexed[digest]
        if (
            decision["spec_id"] != item["spec_id"]
            or decision["rubric_id"] != RUBRIC_ID
            or decision["reviewer"] != REVIEWER
            or len(decision["reason"].strip()) < 20
            or not RFC3339_UTC.fullmatch(decision["reviewed_at"])
        ):
            raise RuntimeError(
                "manual uncertainty decision lineage or rationale changed"
            )
        reviewed_at_values.add(decision["reviewed_at"])
        token_count = candidate["assistant_token_count"]
        if decision["status"] == "approved":
            approved_rows += 1
            approved_tokens += token_count
        else:
            rejected_rows += 1
            rejected_tokens += token_count
        decisions.append({
            "spec_id": item["spec_id"],
            "response_sha256": digest,
            "assistant_tokens": token_count,
            "status": decision["status"],
            "reason": decision["reason"],
        })
    if len(reviewed_at_values) != 1:
        raise RuntimeError("manual uncertainty ledger must have one sealed review time")

    exact_budget_ready = (
        rejected_rows == 0
        and approved_rows == EXPECTED_ROWS
        and approved_tokens == TARGET_ASSISTANT_TOKENS
    )
    report = {
        "schema": "general-replay-uncertainty-manual-review-report-v1",
        "status": (
            "approved_exact_uncertainty_budget"
            if exact_budget_ready
            else "review_complete_replacements_required"
        ),
        "category": CATEGORY,
        "rubric_id": RUBRIC_ID,
        "reviewer": REVIEWER,
        "reviewed_at": next(iter(reviewed_at_values)),
        "review_protocol": {
            "decision_source": "manual_human_judgment",
            "human_reviewed_one_response_at_a_time": True,
            "automatic_approval_or_proxy_judgment_used": False,
            "review_order": "sealed_priority_packet_spec_order",
            "explicit_reject_conditions": [
                "confident_guess",
                "invented_fact",
                "false_premise_acceptance",
                "fake_citation",
                "overbroad_certainty",
            ],
        },
        "packet": {
            "path": _relative(packet),
            "manifest_file_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "manifest_content_sha256": manifest[
                "content_sha256_before_self_field"
            ],
            "source_manifest_file_sha256": manifest["source_manifest"]["sha256"],
            "source_manifest_content_sha256": manifest["source_manifest"][
                "content_sha256"
            ],
            "pinned_uncertainty_chunks": pinned_chunks,
        },
        "ledger": {
            "path": _relative(ledger_path),
            "sha256": hashlib.sha256(ledger_raw).hexdigest(),
            "rows": len(ledger_rows),
            "nonempty_reasons": sum(bool(item["reason"].strip()) for item in ledger_rows),
        },
        "counts": {
            "reviewed": len(decisions),
            "approved": approved_rows,
            "rejected": rejected_rows,
        },
        "tokens": {
            "target": TARGET_ASSISTANT_TOKENS,
            "reviewed": approved_tokens + rejected_tokens,
            "approved": approved_tokens,
            "rejected": rejected_tokens,
            "exact_budget_ready": exact_budget_ready,
            "shortfall": TARGET_ASSISTANT_TOKENS - approved_tokens,
        },
        "replacement_review": {
            "required": rejected_rows > 0,
            "rows_authored_and_reviewed": 0,
            "assistant_tokens_approved": 0,
            "required_provenance_label_if_used": REPLACEMENT_PROVENANCE,
            "human_written_label_forbidden": True,
            "reason": (
                "No priority response was rejected; the sealed set already reaches "
                "the exact 7,500-token target."
                if rejected_rows == 0
                else "Rejected rows require manually authored and reviewed replacements."
            ),
        },
        "identity_validation": {
            "priority_manifest_self_hash_match": True,
            "priority_chunk_hashes_match": True,
            "source_manifest_file_and_self_hashes_match": True,
            "priority_candidates_match_source_packet_by_response_sha256": True,
            "unique_spec_ids": len({item["spec_id"] for item in rows}),
            "unique_source_group_ids": len(
                {item["source_group_id"] for item in rows}
            ),
            "unique_response_sha256": len(expected),
            "ledger_response_set_exact_match": True,
            "spec_response_rubric_lineage_match": True,
        },
        "decisions": decisions,
        "policy": {
            "auto_approvals_created": False,
            "deterministic_proxy_used_for_decisions": False,
            "protected_development_or_final_sources_read": False,
            "gpu_or_training_work_started": False,
        },
    }
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    return report


def report_bytes(report: dict) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    result.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    result.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--print-report", action="store_true")
    result.add_argument("--check", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = build_report(args.packet, args.ledger)
    rendered = report_bytes(report)
    if args.check:
        path = safe_regular_input(args.report, "sealed uncertainty review report")
        if path.read_bytes() != rendered:
            raise RuntimeError("sealed uncertainty review report changed")
    if args.print_report:
        print(rendered.decode(), end="")
    if not args.check and not args.print_report:
        print(json.dumps({
            "status": report["status"],
            "content_sha256": report["content_sha256_before_self_field"],
            "ledger_sha256": report["ledger"]["sha256"],
            "approved_rows": report["counts"]["approved"],
            "approved_tokens": report["tokens"]["approved"],
            "replacement_required": report["replacement_review"]["required"],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
