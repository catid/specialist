#!/usr/bin/env python3
"""Select the smallest exact-token subjective set for human review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from general_replay_v1 import ROOT, canonical_bytes, canonical_sha256


DIRECTORY = ROOT / "data/general_replay_v1"
DEFAULT_SOURCE = DIRECTORY / "approval_review_packet_v1_scale32"
DEFAULT_OUTPUT = DIRECTORY / "approval_priority_review_v1_scale32"
TARGETS = {
    "ordinary_conversation": 15_000,
    "safety_refusal": 7_500,
    "uncertainty_hallucination_resistance": 7_500,
}
CHUNK_ROWS = 8


def deterministic_precheck(item: dict, candidate: dict) -> bool:
    """Remove only obvious format/rubric misses; this is not an approval."""
    text = candidate["assistant_message"]["content"].strip()
    lowered = text.casefold()
    category = item["category"]
    if category == "ordinary_conversation":
        paragraphs = [
            part for part in re.split(r"\n\s*\n", text) if part.strip()
        ]
        return len(paragraphs) == 2 and 40 <= len(text.split()) <= 220
    if category == "safety_refusal":
        boundary = any(phrase in lowered for phrase in (
            "can't help", "cannot help", "won't help", "not able to",
            "unable to", "can't provide", "cannot provide", "can't assist",
            "cannot assist", "cannot comply", "must decline", "must refuse",
        ))
        alternative = any(phrase in lowered for phrase in (
            "instead", "can help", "could help", "safe", "consider",
            "alternative", "offer",
        ))
        return boundary or alternative
    if category == "uncertainty_hallucination_resistance":
        uncertain = any(phrase in lowered for phrase in (
            "cannot determine", "can't determine", "not provided",
            "insufficient", "missing", "not enough information", "unknown",
        ))
        evidence = any(phrase in lowered for phrase in (
            "evidence", "need", "provide", "record", "date", "table",
            "log", "owner",
        ))
        case = re.search(r"U-[0-9]{4}", item["messages"][-1]["content"])
        return (
            uncertain and evidence and case is not None
            and case.group(0).casefold() in lowered
        )
    raise ValueError(f"unsupported priority-review category {category}")


def minimum_exact_group_choices(records: list[dict], target: int) -> list[tuple[dict, dict]]:
    """Minimize reviewed rows while selecting at most one response per spec."""
    # total -> (row count, linked predecessor).  A linked predecessor stores
    # (prior node, item, candidate), so later state replacement is safe.
    reachable: dict[int, tuple[int, object]] = {0: (0, None)}
    ordered = sorted(records, key=lambda item: item["spec_id"])
    for item in ordered:
        choices = [
            candidate for candidate in item["candidates"]
            if deterministic_precheck(item, candidate)
        ]
        before = list(sorted(reachable.items()))
        for subtotal, (row_count, node) in before:
            for candidate in sorted(
                    choices,
                    key=lambda value: (
                        -value["assistant_token_count"],
                        value["response_sha256"],
                    )):
                updated = subtotal + candidate["assistant_token_count"]
                if updated > target:
                    continue
                proposed = (row_count + 1, (node, item, candidate))
                existing = reachable.get(updated)
                if existing is None or proposed[0] < existing[0]:
                    reachable[updated] = proposed
    if target not in reachable:
        raise RuntimeError("prechecked review candidates cannot hit exact target")
    selected = []
    node = reachable[target][1]
    while node is not None:
        node, item, candidate = node
        selected.append((item, candidate))
    selected.reverse()
    if len({item["source_group_id"] for item, _ in selected}) != len(selected):
        raise AssertionError("priority review selected a source group twice")
    return selected


def build_artifacts(source_directory: Path) -> tuple[dict[str, bytes], bytes, dict]:
    source_manifest_path = source_directory / "manifest.json"
    source_manifest_raw = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_manifest_raw)
    if (
        source_manifest.get("schema")
        != "general-replay-approval-review-packet-manifest-v1"
        or source_manifest.get("policy", {}).get("auto_approvals_created") is not False
    ):
        raise RuntimeError("source approval packet contract changed")
    records = []
    for entry in source_manifest["chunks"]:
        path = source_directory / entry["path"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise RuntimeError("source approval chunk identity changed")
        records.extend(
            json.loads(line) for line in raw.decode().splitlines()
            if line.strip()
        )

    artifacts = {}
    chunks = []
    categories = {}
    for category, target in TARGETS.items():
        category_records = [
            item for item in records if item["category"] == category
        ]
        selected = minimum_exact_group_choices(category_records, target)
        selected.sort(key=lambda pair: pair[0]["spec_id"])
        output_records = []
        for item, candidate in selected:
            output = dict(item)
            output["candidates"] = [candidate]
            output["preselection"] = {
                "policy": "deterministic_obvious_miss_filter_then_minimum_rows_v1",
                "quality_status": "pending_human_review",
                "auto_approved": False,
            }
            output_records.append(output)
        category_chunks = 0
        for offset in range(0, len(output_records), CHUNK_ROWS):
            chunk = output_records[offset:offset + CHUNK_ROWS]
            name = f"{category}.priority-{offset // CHUNK_ROWS:03d}.jsonl"
            content = b"".join(canonical_bytes(item) for item in chunk)
            artifacts[name] = content
            chunks.append({
                "path": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "category": category,
                "rows": len(chunk),
                "assistant_tokens": sum(
                    item["candidates"][0]["assistant_token_count"]
                    for item in chunk
                ),
            })
            category_chunks += 1
        categories[category] = {
            "target_assistant_tokens": target,
            "priority_review_rows": len(output_records),
            "priority_review_chunks": category_chunks,
            "selected_assistant_tokens": sum(
                candidate["assistant_token_count"]
                for _, candidate in selected
            ),
            "exact_token_target_reachable_if_all_selected_rows_are_approved": True,
            "quality_status": "pending_human_review",
        }
    manifest = {
        "schema": "general-replay-priority-approval-review-manifest-v1",
        "status": "pending_human_review",
        "source_manifest": {
            "path": str(source_manifest_path),
            "sha256": hashlib.sha256(source_manifest_raw).hexdigest(),
            "content_sha256": source_manifest["content_sha256_before_self_field"],
        },
        "chunk_row_capacity": CHUNK_ROWS,
        "categories": categories,
        "chunks": chunks,
        "rubrics": source_manifest["rubrics"],
        "approval_ledger_contract": source_manifest["approval_ledger_contract"],
        "review_instruction": (
            "Review every selected response against its rubric and emit one "
            "response-addressed approved or rejected ledger row. Any rejection "
            "invalidates the exact-token readiness claim and requires a backup "
            "selection from the full source packet."
        ),
        "policy": {
            "auto_approvals_created": False,
            "approval_ledger_created": False,
            "precheck_is_human_approval": False,
            "all_quality_statuses_pending": True,
            "duplicate_or_padding_fill_used": False,
        },
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    return artifacts, manifest_bytes, manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-directory", type=Path, default=DEFAULT_SOURCE)
    result.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--check", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    artifacts, manifest_bytes, manifest = build_artifacts(args.source_directory)
    manifest_path = args.output_directory / "manifest.json"
    if args.check:
        for name, content in artifacts.items():
            if (args.output_directory / name).read_bytes() != content:
                raise RuntimeError(f"priority review chunk changed: {name}")
        if manifest_path.read_bytes() != manifest_bytes:
            raise RuntimeError("priority review manifest changed")
        return 0
    if args.output_directory.exists():
        raise FileExistsError("priority review build requires a fresh directory")
    args.output_directory.mkdir(parents=True)
    for name, content in artifacts.items():
        (args.output_directory / name).write_bytes(content)
    manifest_path.write_bytes(manifest_bytes)
    print(json.dumps({
        "output_directory": str(args.output_directory),
        "manifest": str(manifest_path),
        "manifest_content_sha256": manifest["content_sha256_before_self_field"],
        "rows": sum(
            item["priority_review_rows"]
            for item in manifest["categories"].values()
        ),
        "chunks": len(artifacts),
        "status": manifest["status"],
        "auto_approvals_created": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
