#!/usr/bin/env python3
"""Build three manual technique QAs from distinct, previously unused pages."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V290_DIR = ROOT / "data/manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V290_DIR)]
import build_context_merit_audit_v290 as baseline_builder
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact, has_protocol_tokens, leakage_reason, normalize_text, parse_qa, stable_fact_id

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
OUTPUT = OUT_DIR / "pending_additions_technique_tranche_02_v1.jsonl"
REPORT = OUT_DIR / "report_technique_tranche_02_v1.json"
REVIEWER = "codex-technique-additions-v2"
REVIEWED_AT = "2026-07-15"
BASELINE_ROWS = 495
BASELINE_SHA256 = "1afd8517320e5465ad3f52d915bc9391b19ca56a32a2db3ffcd713f88442acf1"
EXPECTED_OUTPUT_SHA256 = "3125b51e446ee47b708d155b7fbb4fa0640476b422a29d897429e2e0525917d7"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
SOURCES = {
    "tension_control": {
        "path": DATA / "raw/rope_resources_v1/rope365__1f02fa6488558b05bf1f.json",
        "url": "https://rope365.com/tension/",
        "document_sha256": "559d1c29c6bb09e09b5111acb8fe466e51cad0ad48295ebfa67c1575b56f9864",
        "markers": ("Day 58: Uniformity – Wrap around the body and focus on keeping even tension. Practice switching hands and feeling the tension in your body. Try a communication game: every time the person being tied feel a variation of tension, they make a sound to indicate the issue.",),
    },
    "clove_hitch_lock": {
        "path": DATA / "raw/rope_resources_v1/rope365__46835d0ae181726e3e3a.json",
        "url": "https://rope365.com/clove-hitch-lock/",
        "document_sha256": "8e9b8f3bd7e68050d465f0cad337e0f0f332c1f14011ce72e2df25752da6398f",
        "markers": ("Since a clove hitch can slide, tying it right after a change of direction will limit its movement.",),
    },
    "cinch_structure": {
        "path": DATA / "raw/rope_resources_v1/rope365__e156eea7b8dfda8b39a9.json",
        "url": "https://rope365.com/cinches/",
        "document_sha256": "7ca2cd8184ce40a87bf174138af5769ed0f8d459a30ef5e356ecb0b56beda571",
        "markers": ("A cinch is a technique to add solidity to a structure by inserting a rope in a gap between two columns and catching a rope on the other side. It prevents the structure from rotating or sliding, making it harder to escape, and it is a great way to make things tighter. Cinches are called kannuki 閂 in Japanese.",),
    },
}
FACTS = (
    {
        "source_key": "tension_control",
        "topic": "tension_control",
        "question": "How can partners practice noticing uneven rope tension during wraps?",
        "answer": "Keep the wraps even and have the person being tied signal whenever they feel the tension change.",
        "paraphrase_rationale": "This turns the page's uniformity exercise and communication game into one concise partner drill without prescribing a tension level.",
    },
    {
        "source_key": "clove_hitch_lock",
        "topic": "clove_hitch_lock",
        "question": "Why place two half-hitches just after a change of direction?",
        "answer": "A clove hitch can slide, and tying this variant after the direction change limits its movement.",
        "paraphrase_rationale": "This preserves the source's reason for positioning its two-half-hitches variant without claiming the lock cannot move or fail.",
    },
    {
        "source_key": "cinch_structure",
        "topic": "cinch_structure",
        "question": "What does a cinch add to a two-column rope structure?",
        "answer": "It catches rope through the gap between the columns to prevent rotation or sliding and make the structure more solid.",
        "paraphrase_rationale": "This condenses the source's definition and structural purpose while omitting its subjective tightness and escape claims.",
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


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def build_baseline(path: Path, report: Path) -> list[dict]:
    baseline_builder.build_projection(path, report)
    rows = read_jsonl(path)
    if (len(rows), file_sha256(path)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v290 baseline drift")
    return rows


def select_evidence(document: dict, markers: tuple[str, ...]) -> str:
    lines = []
    for marker in markers:
        matches = [line for line in document["text"].splitlines() if marker in line]
        if len(matches) != 1:
            raise ValueError(f"evidence drift: {marker}")
        lines.append(matches[0])
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    documents = {}
    for key, source in SOURCES.items():
        document = json.loads(source["path"].read_text())
        if (document["url"], document["document_sha256"], text_sha256(document["text"])) != (source["url"], source["document_sha256"], source["document_sha256"]):
            raise ValueError(f"{key}: source identity drift")
        documents[key] = document
    with tempfile.TemporaryDirectory(prefix="technique-v2-", dir=OUT_DIR) as temp:
        baseline_path = Path(temp) / "v290.jsonl"
        baseline = build_baseline(baseline_path, Path(temp) / "v290.report.json")
    facts = [EvalFact(row["question"], row["answer"], row["fact_id"], "train") for row in baseline]
    questions = {normalize_text(row["question"]) for row in baseline}
    pairs = {(normalize_text(row["question"]), normalize_text(row["answer"])) for row in baseline}
    documents_in_baseline = {row["document_sha256"] for row in baseline}
    urls_in_baseline = {row["url"].rstrip("/").casefold() for row in baseline}
    rows = []
    for fact in FACTS:
        source = SOURCES[fact["source_key"]]
        question, answer = fact["question"], fact["answer"]
        pair = normalize_text(question), normalize_text(answer)
        if not question.endswith("?") or "\n" in question or "\n" in answer:
            raise ValueError("addition is not standalone one-line Q&A")
        if has_protocol_tokens(question) or has_protocol_tokens(answer) or parse_qa(f"Question: {question}\nAnswer: {answer}") != (question, answer):
            raise ValueError("non-canonical addition")
        if pair in pairs or pair[0] in questions or leakage_reason(question, answer, facts):
            raise ValueError("collision with v290 train-only baseline")
        if source["document_sha256"] in documents_in_baseline or source["url"].rstrip("/").casefold() in urls_in_baseline:
            raise ValueError("source document or URL is not novel")
        evidence = select_evidence(documents[fact["source_key"]], source["markers"])
        rendered = f"Question: {question}\nAnswer: {answer}"
        rows.append({
            "answer": answer, "claim_type": "instructional", "document_sha256": source["document_sha256"],
            "evidence": evidence, "evidence_sha256": text_sha256(evidence), "evidence_url": source["url"],
            "fact_id": stable_fact_id(question, answer), "kind": "qa_resource_manual_fact",
            "paraphrase_rationale": fact["paraphrase_rationale"], "quality_schema": "manual-resource-fact-v1",
            "question": question, "resource_id": "rope365", "reviewer": REVIEWER, "source": "rope365",
            "source_lineage": {"artifact": portable(OUTPUT), "raw_document": portable(source["path"]), "resource_manifest": portable(RESOURCE_MANIFEST)},
            "text": rendered, "topic": fact["topic"], "url": source["url"], "verified_at": REVIEWED_AT,
        })
    if len(rows) != 3 or len({row["fact_id"] for row in rows}) != 3:
        raise ValueError("addition tranche identity drift")
    write_jsonl(OUTPUT, rows)
    output_sha = file_sha256(OUTPUT)
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and output_sha != EXPECTED_OUTPUT_SHA256:
        raise ValueError("addition artifact hash drift")
    strata = Counter(classify_stratum(row) for row in rows)
    report = {
        "artifact": {"path": portable(OUTPUT), "rows": 3, "sha256": output_sha},
        "baseline": {"rows": BASELINE_ROWS, "sha256": BASELINE_SHA256},
        "method": {"authoring": "manual full-source review and hand-authored Q&A", "collision_scope": "v290 train-only projection; sealed collisions delegated to integration tooling", "selection": "one useful technique fact from each of three distinct, previously unrepresented source documents and URLs"},
        "new_independent_inputs": {"document_sha256s": 3, "expected_strata": dict(sorted(strata.items())), "urls": 3},
        "reviewed_at": REVIEWED_AT, "reviewer": REVIEWER, "schema": "manual-technique-additions-report-v2",
        "sources": {key: {"document_sha256": value["document_sha256"], "file_sha256": file_sha256(value["path"]), "path": portable(value["path"]), "url": value["url"]} for key, value in sorted(SOURCES.items())},
        "status": "segregated_pending_integration",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
