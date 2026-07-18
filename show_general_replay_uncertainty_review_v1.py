#!/usr/bin/env python3
"""Display exactly one sealed uncertainty-priority response for human review."""

from __future__ import annotations

import argparse
import hashlib
import json

from general_replay_v1 import ROOT, canonical_sha256


PACKET = ROOT / "data/general_replay_v1/approval_priority_review_v1_scale32"


def load_rows() -> list[dict]:
    raw_manifest = (PACKET / "manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    unsigned = dict(manifest)
    claimed = unsigned.pop("content_sha256_before_self_field", None)
    if (
        claimed != canonical_sha256(unsigned)
        or manifest.get("schema")
        != "general-replay-priority-approval-review-manifest-v1"
    ):
        raise RuntimeError("priority review manifest changed")
    rows = []
    for entry in manifest["chunks"]:
        if entry["category"] != "uncertainty_hallucination_resistance":
            continue
        raw = (PACKET / entry["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise RuntimeError("uncertainty review chunk changed")
        rows.extend(
            json.loads(line) for line in raw.decode().splitlines() if line.strip()
        )
    if len(rows) != 35:
        raise RuntimeError("uncertainty priority row count changed")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int, help="one-based uncertainty review index")
    args = parser.parse_args()
    rows = load_rows()
    if not 1 <= args.index <= len(rows):
        raise ValueError("uncertainty review index must be 1-35")
    item = rows[args.index - 1]
    candidate = item["candidates"][0]
    print(f"REVIEW {args.index}/{len(rows)}")
    print("spec_id:", item["spec_id"])
    print("source_group_id:", item["source_group_id"])
    print("response_sha256:", candidate["response_sha256"])
    print("assistant_tokens:", candidate["assistant_token_count"])
    print("rubric_id:", item["rubric_id"])
    print("\nPROMPT\n" + item["messages"][-1]["content"])
    print("\nRESPONSE\n" + candidate["assistant_message"]["content"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
