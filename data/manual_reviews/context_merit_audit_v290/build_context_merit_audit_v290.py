#!/usr/bin/env python3
"""Integrate three distinct-document technique/equipment additions after v289."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

sys.setrecursionlimit(5000)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V289_DIR = DATA / "manual_reviews/context_merit_audit_v289"
ADDITION_DIR = DATA / "manual_reviews/technique_equipment_additions_v1"
sys.path[:0] = [str(ROOT), str(V289_DIR), str(ADDITION_DIR)]
import build_context_merit_audit_v289 as previous
import build_technique_equipment_additions as additions_builder
import eggroll_es_train_panel_sampler_v13 as panel_rules

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v290.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v290.jsonl"
REPORT = OUT_DIR / "report_context_merit_v290.json"
REVIEWER = "codex-context-merit-audit-v290"
REVIEWED_AT = "2026-07-15"
ADDITIONS = additions_builder.OUTPUT
EXPECTED_ADDITIONS_SHA256 = "058cb668b8a243b1656d52dde21b93f19229d76d0408c6e6bce3b7b494169cf9"
EXPECTED_OUTPUT_SHA256 = "1afd8517320e5465ad3f52d915bc9391b19ca56a32a2db3ffcd713f88442acf1"

RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST
ACTIVE_DATASET = previous.ACTIVE_DATASET
ACTIVE_REPORT = previous.ACTIVE_REPORT
ACTIVE_CURATIONS = previous.ACTIVE_CURATIONS
QUALITY_MERIT_CURATION = previous.QUALITY_MERIT_CURATION
TASUKI_CURATION = previous.TASUKI_CURATION
CORE = previous.CORE
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
CONTEXT_CURATIONS = previous.OUTPUT_CONTEXT_CURATIONS
PRIOR_PROJECTION_CURATIONS = ()
OUTPUT_CONTEXT_CURATIONS = CONTEXT_CURATIONS
OUTPUT_PROJECTION_CURATIONS = ()
PRIOR_PENDING_ADDITIONS = (*previous.PRIOR_PENDING_ADDITIONS, ADDITIONS)

V283_FROZEN = ROOT / "experiments/eggroll_es_hpo/dataset_candidates/v283/train_qa_context_merit_v283.jsonl"
REPLAY_STEPS = (
    (284, "107a43403edbc51a6b2b72f5c964648e865d384fe2156ce12c84b7cbcb63c614"),
    (285, "172340fec38e97a8dcf2a70b15157af3a9ce627d359dd780d6b7043324e819ae"),
    (286, "338514d23d367ae3a8240ef686b47227e8443e23942120540d9a3085a40c69c2"),
    (287, "527ef93fac6f53b1dc6a433523f62244cad9978bf36103bbb56a2f61d9c0a1b5"),
    (288, "cdaf70a32714f3a4e248618688fcfc840d10eade0cafca0e1e9c9b15e65ab2c3"),
    (289, "17948a72bd383f9445f269567c1fe4964468cecc5892259cb88cfcaf217a5cdb"),
)
BASE_PRODUCTION_INPUTS = (V283_FROZEN,)
PRODUCTION_INPUTS = (V283_FROZEN, ADDITIONS)
SEALED_EVAL_PATHS = (
    DATA / "eval_qa.jsonl",
    DATA / "eval_qa_v2.jsonl",
    DATA / "eval_qa_v3.jsonl",
    DATA / "ood_qa.jsonl",
    DATA / "ood_qa_v3.jsonl",
)
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated corrected training projection through context-merit v289",
    "direct_rows_without_prior_curation": 1,
    "rows": 492,
    "sha256": "17948a72bd383f9445f269567c1fe4964468cecc5892259cb88cfcaf217a5cdb",
}
EXPECTED_CAPACITY = {
    "before": {"conflict_units": 199, "equipment_material": 15, "resources_general": 71, "safety_consent": 72, "technique": 41},
    "after": {"conflict_units": 202, "equipment_material": 17, "resources_general": 71, "safety_consent": 72, "technique": 42},
}


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def build_projection_with_inputs(output: Path, report: Path, curations, inputs) -> None:
    command = [
        sys.executable,
        "build_curated_qa.py",
        "--inputs",
        *(portable(path) for path in inputs),
        "--eval",
        *(portable(path) for path in SEALED_EVAL_PATHS),
        "--curation",
        *(portable(path) for path in curations),
        "--output",
        str(output),
        "--report",
        str(report),
    ]
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def build_v289_baseline(output: Path, report: Path) -> None:
    """Replay v284-v289 sequentially from the immutable v283 candidate."""
    replay_dir = output.parent / f".{output.name}.v289-replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    current = V283_FROZEN
    if file_sha256(current) != "83d14d9d42740c836b49a8ec9e4237766e9d751c827c21d4d2c79500ee4bc3b9":
        raise ValueError("immutable v283 candidate drift")
    for version, expected_sha256 in REPLAY_STEPS:
        next_output = output if version == 289 else replay_dir / f"v{version}.jsonl"
        next_report = report if version == 289 else replay_dir / f"v{version}.report.json"
        curation = DATA / "manual_reviews" / f"context_merit_audit_v{version}" / f"pending_curation_context_merit_v{version}.jsonl"
        build_projection_with_inputs(next_output, next_report, (curation,), (current,))
        if len(read_jsonl(next_output)) != 492 or file_sha256(next_output) != expected_sha256:
            raise ValueError(f"sequential v{version} replay drift")
        current = next_output


def build_projection(output: Path, report: Path, curations=()) -> None:
    if tuple(curations):
        raise ValueError("v290 has no curation decisions")
    replay_dir = output.parent / f".{output.name}.v290-input"
    replay_dir.mkdir(parents=True, exist_ok=True)
    baseline = replay_dir / "v289.jsonl"
    baseline_report = replay_dir / "v289.report.json"
    build_v289_baseline(baseline, baseline_report)
    build_projection_with_inputs(output, report, (), (baseline, ADDITIONS))


def prior_decision_artifacts() -> tuple[Path, ...]:
    paths = []
    for version in range(1, 290):
        directory = DATA / "manual_reviews" / f"context_merit_audit_v{version}"
        paths.extend(
            (
                directory / f"context_merit_audit_v{version}.jsonl",
                directory / f"pending_curation_context_merit_v{version}.jsonl",
                directory / f"report_context_merit_v{version}.json",
            )
        )
    return tuple(paths)


def normalize_url(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    if not parsed.scheme or not parsed.netloc:
        return str(value).strip()
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"si", "fbclid", "gclid"}
    ]
    path = re.sub(r"/+", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, urlencode(sorted(query)), ""))


def row_urls(row: dict) -> set[str]:
    scalar = (
        "url",
        "evidence_url",
        "canonical_url",
        "supplied_url",
        "original_ropetopia_url",
        "title_evidence_url",
        "url_evidence_url",
        "canonical_resource_url",
    )
    arrays = ("urls", "canonical_urls", "supplied_urls")
    values = {row[key] for key in scalar if row.get(key)}
    values.update(url for key in arrays for url in row.get(key, ()) if url)
    return {normalize_url(value) for value in values}


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def conservative_capacity(rows: list[dict]) -> dict[str, int]:
    semantic_ids = panel_rules.build_semantic_clusters(rows)
    disjoint = DisjointSet(len(rows))
    first = {}

    def connect(identifier: str, index: int) -> None:
        if identifier in first:
            disjoint.union(index, first[identifier])
        else:
            first[identifier] = index

    for index, (row, semantic_id) in enumerate(zip(rows, semantic_ids)):
        connect(f"document:{row['document_sha256']}", index)
        for url in row_urls(row):
            connect(f"url:{url}", index)
        lineage = row.get("source_lineage") or {}
        for key in ("raw", "raw_document", "raw_successor_document"):
            if lineage.get(key):
                connect(f"lineage:{key}:{json.dumps(lineage[key], sort_keys=True)}", index)
        connect(f"semantic:{semantic_id}", index)

    components = defaultdict(list)
    for index in range(len(rows)):
        components[disjoint.find(index)].append(index)
    strata = Counter()
    for indices in components.values():
        row_strata = Counter(panel_rules.classify_stratum(rows[index]) for index in indices)
        dominant = max(
            panel_rules.STRATA,
            key=lambda name: (row_strata[name], panel_rules._TIE_PRIORITY[name]),
        )
        strata[dominant] += 1
    return {"conflict_units": len(components), **{name: strata[name] for name in panel_rules.STRATA}}


def observe() -> dict:
    with tempfile.TemporaryDirectory(prefix=".v290-observation-", dir=OUT_DIR) as temp:
        directory = Path(temp)
        baseline = directory / "baseline.jsonl"
        baseline_report = directory / "baseline.report.json"
        build_v289_baseline(baseline, baseline_report)
        baseline_rows = read_jsonl(baseline)
        if (len(baseline_rows), file_sha256(baseline)) != (492, PROJECTED_SELECTION_BASELINE["sha256"]):
            raise ValueError("v289 baseline drift")

        output = directory / "projection.jsonl"
        report = directory / "projection.report.json"
        dataset_bytes = []
        report_bytes = []
        for _ in (1, 2):
            build_projection(output, report)
            dataset_bytes.append(output.read_bytes())
            report_bytes.append(report.read_bytes())
        parsed_report = json.loads(report_bytes[0])
        rows = read_jsonl(output)
        return {
            "baseline_capacity": conservative_capacity(baseline_rows),
            "dataset_equal": dataset_bytes[0] == dataset_bytes[1],
            "dataset_sha256": hashlib.sha256(dataset_bytes[0]).hexdigest(),
            "eval_fact_count": parsed_report["eval_fact_count"],
            "output_capacity": conservative_capacity(rows),
            "report_equal": report_bytes[0] == report_bytes[1],
            "rows": len(rows),
        }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    additions_builder.main()
    if file_sha256(ADDITIONS) != EXPECTED_ADDITIONS_SHA256:
        raise ValueError("addition artifact drift")
    addition_rows = read_jsonl(ADDITIONS)
    audits = []
    for index, row in enumerate(addition_rows, 1):
        source_path = ROOT / row["source_lineage"]["raw_document"]
        document = json.loads(source_path.read_text())
        if row["url"] != document["url"] or row["document_sha256"] != document["document_sha256"]:
            raise ValueError("addition source lineage drift")
        if not all(line in document["text"] for line in row["evidence"].splitlines()):
            raise ValueError("addition evidence drift")
        audits.append(
            {
                "audit_index": index,
                "decision": "add",
                "document_sha256": row["document_sha256"],
                "fact_id": row["fact_id"],
                "proposed_answer": row["answer"],
                "proposed_question": row["question"],
                "projection_lineage": {
                    "baseline_rows": 492,
                    "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
                    "prior_context_merit_review": False,
                },
                "reason": row["paraphrase_rationale"],
                "reason_code": f"add_distinct_{row['topic']}_fact",
                "review_pass": "distinct_document_technique_equipment_additions",
                "reviewed_at": REVIEWED_AT,
                "reviewer": REVIEWER,
                "risk_features": CORE.risk_features(row),
                "schema": "context-merit-audit-v290",
                "source": row["source"],
                "source_document": portable(source_path),
                "source_document_file_sha256": file_sha256(source_path),
                "source_support": "manual_paraphrase",
                "support_evidence": row["evidence"],
                "support_evidence_sha256": text_sha256(row["evidence"]),
                "url": row["url"],
            }
        )
    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, [])
    observed = observe()
    if not observed["dataset_equal"] or not observed["report_equal"]:
        raise ValueError("v290 projection is nondeterministic")
    if observed["rows"] != 495 or observed["eval_fact_count"] != 612:
        raise ValueError("v290 row/eval aggregate drift")
    if observed["baseline_capacity"] != EXPECTED_CAPACITY["before"]:
        raise ValueError("v290 baseline capacity drift")
    if observed["output_capacity"] != EXPECTED_CAPACITY["after"]:
        raise ValueError(f"v290 output capacity drift: {observed['output_capacity']}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observed["dataset_sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError("v290 output hash drift")
    report = {
        "active_baseline": {
            "dataset": {"path": portable(ACTIVE_DATASET), "rows": 784, "sha256": file_sha256(ACTIVE_DATASET)},
            "report": {"path": portable(ACTIVE_REPORT), "sha256": file_sha256(ACTIVE_REPORT)},
        },
        "addition_artifact": {"path": portable(ADDITIONS), "rows": 3, "sha256": file_sha256(ADDITIONS)},
        "audit": {"by_decision": {"add": 3, "drop": 0, "edit": 0, "keep": 0}, "path": portable(AUDIT), "rows": 3, "sha256": file_sha256(AUDIT)},
        "conservative_capacity": {
            "after": observed["output_capacity"],
            "before": observed["baseline_capacity"],
            "delta": {key: observed["output_capacity"][key] - observed["baseline_capacity"][key] for key in observed["baseline_capacity"]},
            "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
        },
        "frozen_prior_decision_artifacts": [
            {"path": portable(path), "sha256": file_sha256(path)} for path in prior_decision_artifacts()
        ],
        "isolated_build_projection": {
            "automated_projection_runs": 2,
            "new_additions_applied": 3,
            "output_rows": observed["rows"],
            "output_sha256": observed["dataset_sha256"],
            "repeat_dataset_byte_identical": observed["dataset_equal"],
            "repeat_projection_report_byte_identical": observed["report_equal"],
            "sealed_eval_fact_count_reported_by_tooling": observed["eval_fact_count"],
            "sequential_replay": {
                "base": {"path": portable(V283_FROZEN), "sha256": file_sha256(V283_FROZEN)},
                "verified_steps": [version for version, _ in REPLAY_STEPS],
            },
        },
        "new_pending_curation": {"decisions": 0, "path": portable(CURATION), "sha256": file_sha256(CURATION)},
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "schema": "context-merit-audit-report-v290",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
