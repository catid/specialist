#!/usr/bin/env python3
"""Build a fourth distinct-document manual technique/equipment tranche."""
from __future__ import annotations

import hashlib, json, sys, tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V292_DIR = ROOT / "data/manual_reviews/context_merit_audit_v292"
sys.path[:0] = [str(ROOT), str(V292_DIR)]
import build_context_merit_audit_v292 as baseline_builder
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact, has_protocol_tokens, leakage_reason, normalize_text, parse_qa, stable_fact_id

DATA = ROOT / "data"; OUT_DIR = Path(__file__).resolve().parent
OUTPUT = OUT_DIR / "pending_additions_technique_equipment_tranche_04_v1.jsonl"
REPORT = OUT_DIR / "report_technique_equipment_tranche_04_v1.json"
BASELINE_ROWS = 501
BASELINE_SHA256 = "3a1b3ca8e9ada0be0a8758fd9b9cc4ce01d17164746d587dcb028a0e916a7a17"
EXPECTED_OUTPUT_SHA256 = "b929fde8bf96fe7dafb9dbc02f0827c0b20310c30a00683b398bc070481d7e9a"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
SOURCES = {
    "rope_dye_material_match": {
        "path": DATA / "raw/rope_resources_v1/rope365__7e518bc9ebc9f4fd433e.json", "url": "https://rope365.com/rope-color/",
        "document_sha256": "3592435741f8c9ec3a4e914e04d96495d4421a51e5112a5fd6cfc55b9cc30f42",
        "markers": ("The first step is to find a dye that will match the material of your rope. Using products for dyeing clothing is a good place to start, check the type of fibre it can be used on. Most dyes made for natural fibres are unlikely to not work on synthetic fibres and vice versa. If your rope has a mixed material construction, you might have mixed results with the dye only working on some part of the rope (ex: cotton rope with a nylon core, a jute rope reinforced with synthetic fibre).",),
    },
    "tail_management_drill": {
        "path": DATA / "raw/rope_resources_v1/rope365__28764d9f7debbae39e5c.json", "url": "https://rope365.com/handling/",
        "document_sha256": "24c0287d8fc3a525cc5c6853e6fce18d4ebfee7a08d026adbbebd344c8603f6c",
        "markers": ("| Day 226: Tail Management – It is a common struggle for the tail of the rope to snag or get wrapped around something when it should not. Practice wrapping the body and throwing the tail of the rope where you want to go next to improve your flow of tying. As the rope gets shorter, practice keeping the rope straight so the tail slides in smoothly in small gaps. |",),
    },
    "partner_movement": {
        "path": DATA / "raw/rope_resources_v1/rope365__f1daf7c0610cc26f063d.json", "url": "https://rope365.com/movement/",
        "document_sha256": "bfc239e1c41785d0dd6d54633f404f22ea98cdd6ff393e23a2c61e89e0ad749d",
        "markers": ("Day 65: Moving Around – Don’t stay in place while tying. Explore how moving around your partner can improve your efficiency and change how you interact together.",),
    },
}
FACTS = (
    {"source_key": "rope_dye_material_match", "topic": "rope_dye_material_match", "question": "Why should rope dye be matched to the rope material?", "answer": "Natural- and synthetic-fibre dyes are not interchangeable, and mixed-material rope may take colour unevenly.", "paraphrase_rationale": "This preserves the compatibility warning and mixed-material consequence without recommending a specific dye or process."},
    {"source_key": "tail_management_drill", "topic": "tail_management_drill", "question": "What wrap drill can reduce rope-tail snags?", "answer": "Send the tail toward its next destination as you wrap, then keep the shortening rope straight so it slides smoothly through small gaps.", "paraphrase_rationale": "This condenses both stages of the source's tail-management drill into an actionable but bounded practice cue."},
    {"source_key": "partner_movement", "topic": "partner_movement", "question": "Why practice moving around a partner while tying?", "answer": "Moving around can improve tying efficiency and change how the partners interact.", "paraphrase_rationale": "This directly restates the stated purpose of the movement exercise without introducing a particular movement pattern."},
)


def file_sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def text_sha256(text): return hashlib.sha256(text.encode()).hexdigest()
def portable(path): return str(Path(path).resolve().relative_to(ROOT))
def read_jsonl(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def write_jsonl(path, rows): Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def build_baseline(path, report):
    baseline_builder.build_projection(path, report); rows = read_jsonl(path)
    if (len(rows), file_sha256(path)) != (BASELINE_ROWS, BASELINE_SHA256): raise ValueError("v292 baseline drift")
    return rows


def evidence(document, markers):
    selected = []
    for marker in markers:
        matches = [line for line in document["text"].splitlines() if marker in line]
        if len(matches) != 1: raise ValueError(f"evidence drift: {marker}")
        selected.append(matches[0])
    return "\n".join(selected)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); documents = {}
    for key, source in SOURCES.items():
        document = json.loads(source["path"].read_text())
        if (document["url"], document["document_sha256"], text_sha256(document["text"])) != (source["url"], source["document_sha256"], source["document_sha256"]): raise ValueError(f"{key}: source identity drift")
        documents[key] = document
    with tempfile.TemporaryDirectory(prefix="technique-equipment-v4-", dir=OUT_DIR) as temp:
        baseline = build_baseline(Path(temp) / "v292.jsonl", Path(temp) / "v292.report.json")
    baseline_facts = [EvalFact(r["question"], r["answer"], r["fact_id"], "train") for r in baseline]
    questions = {normalize_text(r["question"]) for r in baseline}; pairs = {(normalize_text(r["question"]), normalize_text(r["answer"])) for r in baseline}
    docs = {r["document_sha256"] for r in baseline}; urls = {r["url"].rstrip("/").casefold() for r in baseline}; rows = []
    for fact in FACTS:
        source = SOURCES[fact["source_key"]]; question, answer = fact["question"], fact["answer"]; pair = normalize_text(question), normalize_text(answer)
        if not question.endswith("?") or "\n" in question or "\n" in answer: raise ValueError("non-standalone Q&A")
        if has_protocol_tokens(question) or has_protocol_tokens(answer) or parse_qa(f"Question: {question}\nAnswer: {answer}") != (question, answer): raise ValueError("non-canonical Q&A")
        if pair in pairs or pair[0] in questions or leakage_reason(question, answer, baseline_facts): raise ValueError("collision with v292 train-only baseline")
        if source["document_sha256"] in docs or source["url"].rstrip("/").casefold() in urls: raise ValueError("non-novel source")
        support = evidence(documents[fact["source_key"]], source["markers"]); rendered = f"Question: {question}\nAnswer: {answer}"
        rows.append({"answer": answer, "claim_type": "instructional", "document_sha256": source["document_sha256"], "evidence": support, "evidence_sha256": text_sha256(support), "evidence_url": source["url"], "fact_id": stable_fact_id(question, answer), "kind": "qa_resource_manual_fact", "paraphrase_rationale": fact["paraphrase_rationale"], "quality_schema": "manual-resource-fact-v1", "question": question, "resource_id": "rope365", "reviewer": "codex-technique-equipment-additions-v4", "source": "rope365", "source_lineage": {"artifact": portable(OUTPUT), "raw_document": portable(source["path"]), "resource_manifest": portable(RESOURCE_MANIFEST)}, "text": rendered, "topic": fact["topic"], "url": source["url"], "verified_at": "2026-07-15"})
    if len(rows) != 3 or len({r["fact_id"] for r in rows}) != 3: raise ValueError("tranche identity drift")
    write_jsonl(OUTPUT, rows); output_sha = file_sha256(OUTPUT)
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and output_sha != EXPECTED_OUTPUT_SHA256: raise ValueError("artifact hash drift")
    strata = Counter(classify_stratum(r) for r in rows)
    report = {"artifact": {"path": portable(OUTPUT), "rows": 3, "sha256": output_sha}, "baseline": {"rows": BASELINE_ROWS, "sha256": BASELINE_SHA256}, "method": {"authoring": "manual full-source review and hand-authored Q&A", "collision_scope": "v292 train-only projection; sealed collisions delegated to integration tooling", "selection": "one useful fact from each of three distinct, previously unrepresented source documents and URLs"}, "new_independent_inputs": {"document_sha256s": 3, "expected_strata": dict(sorted(strata.items())), "urls": 3}, "reviewed_at": "2026-07-15", "reviewer": "codex-technique-equipment-additions-v4", "schema": "manual-technique-equipment-additions-report-v4", "sources": {key: {"document_sha256": value["document_sha256"], "file_sha256": file_sha256(value["path"]), "path": portable(value["path"]), "url": value["url"]} for key, value in sorted(SOURCES.items())}, "status": "segregated_pending_integration"}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
