#!/usr/bin/env python3
"""Validate per-resource crawl coverage and emit a compact tracked report."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "sources" / "rope_resources_v1.json"
DEFAULT_COVERAGE_DIR = ROOT / "data" / "raw" / "rope_resources_v1_coverage"
DEFAULT_OUTPUT = ROOT / "data" / "rope_resources_coverage_v1.json"
OUTCOMES = ("fetched", "skipped", "blocked", "failed")
STATUSES = {"complete", "partial", "blocked", "failed", "empty"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def build(manifest_path: Path, coverage_dir: Path, output_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "rope-resource-manifest-v1":
        raise ValueError(f"{manifest_path}: unsupported manifest schema")
    resources = manifest.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ValueError(f"{manifest_path}: resources must be a non-empty list")
    resource_ids = [resource.get("id") for resource in resources]
    if any(not isinstance(resource_id, str) for resource_id in resource_ids):
        raise ValueError(f"{manifest_path}: invalid resource id")
    if len(resource_ids) != len(set(resource_ids)):
        raise ValueError(f"{manifest_path}: duplicate resource id")

    manifest_sha256 = file_sha256(manifest_path)
    expected_files = {f"{resource_id}.coverage.json" for resource_id in resource_ids}
    actual_files = {path.name for path in coverage_dir.glob("*.coverage.json")}
    missing = sorted(expected_files - actual_files)
    extra = sorted(actual_files - expected_files)
    if missing or extra:
        raise ValueError(f"coverage files disagree with manifest: missing={missing}, extra={extra}")

    totals = collections.Counter()
    status_counts = collections.Counter()
    details = {}
    completed_at = []
    for resource_id in resource_ids:
        path = coverage_dir / f"{resource_id}.coverage.json"
        coverage = json.loads(path.read_text())
        if coverage.get("schema") != "resource-corpus-coverage-v1":
            raise ValueError(f"{path}: unsupported coverage schema")
        if coverage.get("resource_id") != resource_id:
            raise ValueError(f"{path}: resource id mismatch")
        if coverage.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"{path}: stale manifest digest")
        status = coverage.get("status")
        if status not in STATUSES:
            raise ValueError(f"{path}: invalid status {status!r}")
        counts = coverage.get("counts")
        results = coverage.get("results")
        if not isinstance(counts, dict) or not isinstance(results, dict):
            raise ValueError(f"{path}: missing counts/results")
        reason_counts = {}
        for outcome in OUTCOMES:
            items = results.get(outcome)
            count = counts.get(outcome)
            if not isinstance(items, list) or count != len(items):
                raise ValueError(f"{path}: {outcome} count/result mismatch")
            totals[outcome] += count
            reason_counts[outcome] = dict(sorted(collections.Counter(
                item.get("reason", "missing_reason") for item in items
                if isinstance(item, dict)
            ).items()))
        robots = coverage.get("robots") or {}
        details[resource_id] = {
            "status": status,
            "counts": {outcome: counts[outcome] for outcome in OUTCOMES},
            "reason_counts": reason_counts,
            "robots_status": robots.get("status"),
            "robots_ai_train_no": bool(robots.get("ai_train_no", False)),
            "coverage": portable_path(path),
            "coverage_sha256": file_sha256(path),
        }
        status_counts[status] += 1
        if isinstance(coverage.get("completed_at"), str):
            completed_at.append(coverage["completed_at"])

    report = {
        "schema": "rope-resource-coverage-report-v1",
        "manifest": portable_path(manifest_path),
        "manifest_sha256": manifest_sha256,
        "coverage_dir": portable_path(coverage_dir),
        "source_run_completed_at": max(completed_at) if completed_at else None,
        "counts": {
            "resources": len(resource_ids),
            "statuses": dict(sorted(status_counts.items())),
            "outcomes": {outcome: totals[outcome] for outcome in OUTCOMES},
        },
        "resources": details,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--coverage-dir", type=Path, default=DEFAULT_COVERAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.manifest, args.coverage_dir, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
