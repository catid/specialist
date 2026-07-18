#!/usr/bin/env python3
"""Validate a human ledger against the exact priority-review packet."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from general_replay_v1 import (
    ROOT,
    canonical_sha256,
    safe_regular_input,
    validate_approval_ledger,
)


DEFAULT_PACKET = (
    ROOT / "data/general_replay_v1/approval_priority_review_v1_scale32"
)


def load_priority_rows(packet: Path) -> tuple[dict, dict[str, dict]]:
    manifest_path = safe_regular_input(packet / "manifest.json", "priority manifest")
    manifest = json.loads(manifest_path.read_text())
    self_hash = manifest.get("content_sha256_before_self_field")
    unsigned = dict(manifest)
    unsigned.pop("content_sha256_before_self_field", None)
    if (
        manifest.get("schema")
        != "general-replay-priority-approval-review-manifest-v1"
        or self_hash != canonical_sha256(unsigned)
        or manifest.get("policy", {}).get("auto_approvals_created") is not False
    ):
        raise RuntimeError("priority review manifest changed")
    rows = {}
    for entry in manifest["chunks"]:
        path = safe_regular_input(packet / entry["path"], "priority review chunk")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise RuntimeError("priority review chunk changed")
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            candidates = item.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 1:
                raise RuntimeError("priority review item must address one response")
            response = candidates[0]
            digest = response["response_sha256"]
            if digest in rows:
                raise RuntimeError("priority response identity duplicated")
            rows[digest] = {
                "spec_id": item["spec_id"],
                "rubric_id": item["rubric_id"],
                "category": item["category"],
                "assistant_token_count": response["assistant_token_count"],
            }
    return manifest, rows


def validate(packet: Path, approval_ledger: Path) -> dict:
    manifest, expected = load_priority_rows(packet)
    ledger_path = safe_regular_input(approval_ledger, "human approval ledger")
    approvals = [
        json.loads(line) for line in ledger_path.read_text().splitlines()
        if line.strip()
    ]
    indexed = validate_approval_ledger(approvals)
    if set(indexed) != set(expected):
        raise RuntimeError(
            "human ledger must contain exactly one decision per priority response"
        )
    decisions = Counter()
    tokens = Counter()
    for digest, approval in indexed.items():
        reference = expected[digest]
        if (
            approval["spec_id"] != reference["spec_id"]
            or approval["rubric_id"] != reference["rubric_id"]
        ):
            raise RuntimeError("human approval response lineage mismatch")
        key = (reference["category"], approval["status"])
        decisions[key] += 1
        tokens[key] += reference["assistant_token_count"]
    all_approved = all(item["status"] == "approved" for item in indexed.values())
    return {
        "schema": "general-replay-priority-approval-validation-v1",
        "status": (
            "approved_exact_subjective_budget"
            if all_approved
            else "review_complete_with_rejections_reselection_required"
        ),
        "priority_manifest_content_sha256": manifest[
            "content_sha256_before_self_field"
        ],
        "approval_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "decisions": {
            category: {
                status: {
                    "rows": decisions[(category, status)],
                    "assistant_tokens": tokens[(category, status)],
                }
                for status in ("approved", "rejected")
            }
            for category in manifest["categories"]
        },
        "exact_budget_ready": all_approved,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    result.add_argument("--approval-ledger", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(validate(args.packet, args.approval_ledger), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
