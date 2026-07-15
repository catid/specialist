#!/usr/bin/env python3
"""Build three manually authored additions from distinct technique/equipment pages."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V289_DIR = ROOT / "data/manual_reviews/context_merit_audit_v289"
sys.path[:0] = [str(ROOT), str(V289_DIR)]
import build_context_merit_audit_v289 as previous
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact, has_protocol_tokens, leakage_reason, normalize_text, parse_qa, stable_fact_id

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
OUTPUT = OUT_DIR / "pending_additions_technique_equipment_tranche_01_v1.jsonl"
REPORT = OUT_DIR / "report_technique_equipment_tranche_01_v1.json"
REVIEWER = "codex-technique-equipment-additions-v1"
REVIEWED_AT = "2026-07-15"
BASELINE_ROWS = 492
BASELINE_SHA256 = "17948a72bd383f9445f269567c1fe4964468cecc5892259cb88cfcaf217a5cdb"
EXPECTED_OUTPUT_SHA256 = "058cb668b8a243b1656d52dde21b93f19229d76d0408c6e6bce3b7b494169cf9"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
COVERAGE_REPORT = DATA / "rope_resources_coverage_v1.json"

SOURCES = {
    "carabiner_profile": {
        "path": DATA / "raw/rope_resources_v1/rope365__917bc6534539143cae83.json",
        "url": "https://rope365.com/knis-equipment-primer/",
        "document_sha256": "80aeef8bde68b0d4c6109f004eedef239ce0ba5d1aaa0a2d998d8c29f581fae7",
        "markers": (
            "Basically you want to avoid anything that has an I-Beam or skeletonized design.",
        ),
    },
    "rope_end_maintenance": {
        "path": DATA / "raw/rope_resources_v1/rope365__11ea3b338375c2127fa4.json",
        "url": "https://rope365.com/rope-ends/",
        "document_sha256": "15a23b9d96394e1dc93323ec84c6055a12177e0197c9f5276af0922c342c9d4f",
        "markers": (
            "Knots can be untied, whipping needs to be redone, and melted rope must be cut.",
        ),
    },
    "extension_tradeoff": {
        "path": DATA / "raw/rope_resources_v1/rope365__76f666ac4cb6ff1c2f23.json",
        "url": "https://rope365.com/extending-rope/",
        "document_sha256": "c7726534db53d49cbeff183b3658be1bf18b117bd6f92293c56b978059f9bb88",
        "markers": (
            "This simple method is very fast and leaves no bulk at the connection point.",
            "It allows for more flexibility in the location of the extension, works on any kind of rope, is more solid than the lark’s head extension but will leave a part of the tail hanging.",
        ),
    },
}

FACTS = (
    {
        "source_key": "carabiner_profile",
        "topic": "carabiner_profile",
        "question": "What carabiner profile does Rope365's equipment primer recommend for bondage rope?",
        "answer": "Use a round profile, avoiding sharp D-shapes and I-beam or skeletonized designs.",
        "paraphrase_rationale": "The answer condenses the primer's profile recommendation and named designs to avoid without adding a load rating or product endorsement.",
    },
    {
        "source_key": "rope_end_maintenance",
        "topic": "rope_end_maintenance",
        "question": "Which rope-end treatments can be undone for maintenance, and which require redoing or cutting?",
        "answer": "Knots can be untied, whipping must be redone, and melted ends must be cut off.",
        "paraphrase_rationale": "The answer preserves all three maintenance consequences from the comparison paragraph in a standalone sentence.",
    },
    {
        "source_key": "extension_tradeoff",
        "topic": "rope_extension_tradeoff",
        "question": "How do Rope365's lark's-head and square-knot rope extensions differ?",
        "answer": "The lark's head is fast and leaves no bulk but may untie if tension is lost; the square knot is more solid and flexible in placement but takes more steps and leaves a tail.",
        "paraphrase_rationale": "The answer combines the source's speed, bulk, solidity, placement, and tail tradeoffs without introducing a new technique.",
    },
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def portable(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def evidence(document: dict, markers: tuple[str, ...]) -> str:
    selected = []
    for marker in markers:
        matches = [line for line in document["text"].splitlines() if marker in line]
        if len(matches) != 1:
            raise ValueError(f"evidence drift: {marker}")
        selected.append(matches[0])
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate source line selected as evidence")
    return "\n".join(selected)


def build_baseline(path: Path, report: Path) -> list[dict]:
    previous.build_projection(path, report, previous.OUTPUT_PROJECTION_CURATIONS)
    rows = read_jsonl(path)
    if (len(rows), file_sha256(path)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v289 baseline drift")
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    documents = {}
    for key, specification in SOURCES.items():
        document = json.loads(specification["path"].read_text())
        if document["url"] != specification["url"]:
            raise ValueError(f"{key}: URL drift")
        if document.get("document_sha256") != specification["document_sha256"]:
            raise ValueError(f"{key}: document identity drift")
        if text_sha256(document["text"]) != specification["document_sha256"]:
            raise ValueError(f"{key}: text hash drift")
        documents[key] = document

    with tempfile.TemporaryDirectory(prefix="technique-equipment-v1-", dir=OUT_DIR) as temp:
        baseline_path = Path(temp) / "v289.jsonl"
        baseline_report = Path(temp) / "v289.report.json"
        baseline = build_baseline(baseline_path, baseline_report)

    baseline_facts = [EvalFact(row["question"], row["answer"], row["fact_id"], "train") for row in baseline]
    baseline_questions = {normalize_text(row["question"]) for row in baseline}
    baseline_pairs = {(normalize_text(row["question"]), normalize_text(row["answer"])) for row in baseline}
    baseline_documents = {row["document_sha256"] for row in baseline}
    baseline_urls = {row["url"].rstrip("/").casefold() for row in baseline}
    rows = []
    for fact in FACTS:
        specification = SOURCES[fact["source_key"]]
        document = documents[fact["source_key"]]
        question, answer = fact["question"], fact["answer"]
        support = evidence(document, specification["markers"])
        if not question.endswith("?") or "\n" in question or "\n" in answer:
            raise ValueError("addition is not standalone one-line Q&A")
        if has_protocol_tokens(question) or has_protocol_tokens(answer):
            raise ValueError("protocol token in addition")
        rendered = f"Question: {question}\nAnswer: {answer}"
        if parse_qa(rendered) != (question, answer):
            raise ValueError("canonical Q&A round-trip failed")
        pair = (normalize_text(question), normalize_text(answer))
        if pair in baseline_pairs or pair[0] in baseline_questions:
            raise ValueError("exact collision with v289 training projection")
        collision = leakage_reason(question, answer, baseline_facts)
        if collision:
            raise ValueError(f"semantic collision with v289 training projection: {collision}")
        if specification["document_sha256"] in baseline_documents:
            raise ValueError("source document is not novel to the projection")
        if specification["url"].rstrip("/").casefold() in baseline_urls:
            raise ValueError("source URL is not novel to the projection")
        row = {
            "answer": answer,
            "claim_type": "instructional",
            "document_sha256": specification["document_sha256"],
            "evidence": support,
            "evidence_sha256": text_sha256(support),
            "evidence_url": specification["url"],
            "fact_id": stable_fact_id(question, answer),
            "kind": "qa_resource_manual_fact",
            "paraphrase_rationale": fact["paraphrase_rationale"],
            "quality_schema": "manual-resource-fact-v1",
            "question": question,
            "resource_id": "rope365",
            "reviewer": REVIEWER,
            "source": "rope365",
            "source_lineage": {
                "artifact": portable(OUTPUT),
                "collection_coverage": portable(COVERAGE_REPORT),
                "raw_document": portable(specification["path"]),
                "resource_manifest": portable(RESOURCE_MANIFEST),
            },
            "text": rendered,
            "topic": fact["topic"],
            "url": specification["url"],
            "verified_at": REVIEWED_AT,
        }
        rows.append(row)

    if len({row["fact_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate fact IDs within addition tranche")
    write_jsonl(OUTPUT, rows)
    output_sha = file_sha256(OUTPUT)
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and output_sha != EXPECTED_OUTPUT_SHA256:
        raise ValueError("addition artifact hash drift")
    strata = Counter(classify_stratum(row) for row in rows)
    report = {
        "artifact": {"path": portable(OUTPUT), "rows": len(rows), "sha256": output_sha},
        "baseline": {"path": "/tmp/deterministic-v289-projection", "rows": BASELINE_ROWS, "sha256": BASELINE_SHA256},
        "method": {
            "authoring": "manual full-source review and hand-authored Q&A",
            "collision_scope": "v289 train-only projection; sealed collisions delegated to integration tooling",
            "selection": "one useful fact from each of three distinct, previously unrepresented source documents and URLs",
        },
        "new_independent_inputs": {
            "document_sha256s": len({row["document_sha256"] for row in rows}),
            "urls": len({row["url"] for row in rows}),
            "expected_strata": dict(sorted(strata.items())),
        },
        "reviewed_at": REVIEWED_AT,
        "reviewer": REVIEWER,
        "schema": "manual-technique-equipment-additions-report-v1",
        "sources": {
            key: {
                "document_sha256": specification["document_sha256"],
                "file_sha256": file_sha256(specification["path"]),
                "path": portable(specification["path"]),
                "url": specification["url"],
            }
            for key, specification in sorted(SOURCES.items())
        },
        "status": "segregated_pending_integration",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
