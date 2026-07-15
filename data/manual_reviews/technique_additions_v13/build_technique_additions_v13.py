#!/usr/bin/env python3
"""Build three hand-reviewed technique QAs from distinct Rope365 documents."""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V301_DIR = DATA / "manual_reviews/context_merit_audit_v301"
V12_DIR = DATA / "manual_reviews/equipment_safety_additions_v12"
sys.path[:0] = [str(ROOT), str(V301_DIR), str(V12_DIR)]
import build_context_merit_audit_v301 as baseline_builder
import build_equipment_safety_additions_v12 as prior
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact, has_protocol_tokens, leakage_reason, normalize_text, parse_qa, stable_fact_id

OUT_DIR = Path(__file__).resolve().parent
OUTPUT = OUT_DIR / "pending_additions_technique_tranche_13_v1.jsonl"
REPORT = OUT_DIR / "report_technique_tranche_13_v1.json"
BASELINE_ROWS = 528
BASELINE_SHA256 = "cbba5461f1f60bd1770ef6a776be9747d20fd58745ff20263bc4983f31dddbf7"
EXPECTED_OUTPUT_SHA256 = "5ec0b4744704e2f0f409993ac4847c1c3d9c3a63537e93db80535f34d7923e6f"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"

file_sha256 = prior.file_sha256
text_sha256 = prior.text_sha256
portable = prior.portable
read_jsonl = prior.read_jsonl
write_jsonl = prior.write_jsonl
select_evidence = prior.select_evidence

SOURCES = {
    "frog_tie_definition": {
        "path": DATA / "raw/rope_resources_v1/rope365__5dbad9c884b98d608a5d.json",
        "url": "https://rope365.com/frog-tie/",
        "document_sha256": "212d54bc7381254693951fccce211cf57d70080e74832d37bf1a59c9176cfde8",
        "markers": (
            "The frog tie refers to a tie that keeps a limb folded like a bent arm or a bent leg. It is a fast and simple way to create movement restriction. The bent leg tie is also called futomomo shibari 太腿縛り (thigh tie) or futo 太 (fat) for short. The frog tie is a popular tie among self-tier and a good way to practice if you don’t have a partner around.",
        ),
    },
    "box_tie_reference": {
        "path": DATA / "raw/rope_resources_v1/rope365__41e404ddab92f95286e6.json",
        "url": "https://rope365.com/box-tie/",
        "document_sha256": "2e50eba3cae54a6ff0a5c13bcf03a35ef0ad7fa0b1336357e32679ab2547cd8c",
        "markers": (
            "Its long history puts it among the most classic ties of the Japanese bondage style. It is often used as a reference because its structure contains most of the Japanese bondage-style building blocks. Therefore, when tying it, you practice everything you need.",
        ),
    },
    "upward_elbow_alignment": {
        "path": DATA / "raw/rope_resources_v1/rope365__b4e7fccc6f185c6f124d.json",
        "url": "https://rope365.com/mobility/",
        "document_sha256": "8da4e5a7a38ed67e44eaf89157495555bb0012c1d5ce63c23d1e5841377552eb",
        "markers": (
            "Ties like the bunny ears and the riffle tie will place an arm with the bent elbow pointing up. It can be quite challenging to hold this position vertically as the gravity is pulling the arm down. When it starts to flail, the arm will often push against the rope, increasing the speed at which we will experience nerve and circulation issues. One strategy to reduce this is to get the arm as close as possible to the head so that it aligns vertically with the shoulder, reducing the impact of gravity.",
            "Which method gets you closer to a vertical position? When the arm gets tired, lying on the ground is a great way to free yourself or your partner from the pull of gravity.",
        ),
    },
}

FACTS = (
    {
        "source_key": "frog_tie_definition",
        "topic": "frog_tie_definition",
        "question": "What feature defines a frog tie, and what practical effect does it have?",
        "answer": "It keeps an arm or leg folded, creating movement restriction quickly and simply.",
        "paraphrase_rationale": "This combines the source's structural definition and intended effect without turning an alias into trivia.",
    },
    {
        "source_key": "box_tie_reference",
        "topic": "box_tie_reference",
        "question": "Why does Rope365 use the box tie as a reference structure for Japanese-style rope?",
        "answer": "Its structure contains most Japanese-style building blocks, so practicing it exercises many core elements.",
        "paraphrase_rationale": "This retains the source's pedagogical reason for treating the box tie as a reference rather than listing its many aliases.",
    },
    {
        "source_key": "upward_elbow_alignment",
        "topic": "upward_elbow_alignment",
        "question": "How can an upward-pointing elbow be positioned to reduce the effect of gravity in a tie?",
        "answer": "Bring the arm close to the head so it aligns vertically with the shoulder; if it tires, lying down also removes gravity's pull.",
        "paraphrase_rationale": "This preserves both source-supported positioning options and does not claim they eliminate nerve or circulation risk.",
    },
)


def build_baseline(path: Path, report: Path) -> list[dict]:
    baseline_builder.build_projection(path, report)
    rows = read_jsonl(path)
    if (len(rows), file_sha256(path)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v301 baseline drift")
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    documents = {}
    for key, source in SOURCES.items():
        document = json.loads(source["path"].read_text())
        identity = document["url"], document["document_sha256"], text_sha256(document["text"])
        if identity != (source["url"], source["document_sha256"], source["document_sha256"]):
            raise ValueError(f"{key}: source identity drift")
        documents[key] = document

    with tempfile.TemporaryDirectory(prefix="technique-v13-", dir=OUT_DIR) as temp:
        baseline = build_baseline(Path(temp) / "v301.jsonl", Path(temp) / "v301.report.json")
    baseline_facts = [EvalFact(row["question"], row["answer"], row["fact_id"], "train") for row in baseline]
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
        rendered = f"Question: {question}\nAnswer: {answer}"
        if has_protocol_tokens(question) or has_protocol_tokens(answer) or parse_qa(rendered) != (question, answer):
            raise ValueError("non-canonical addition")
        if pair in pairs or pair[0] in questions or leakage_reason(question, answer, baseline_facts):
            raise ValueError("collision with v301 train-only baseline")
        if source["document_sha256"] in documents_in_baseline or source["url"].rstrip("/").casefold() in urls_in_baseline:
            raise ValueError("source document or URL is not novel")
        evidence = select_evidence(documents[fact["source_key"]], source["markers"])
        rows.append({
            "answer": answer,
            "claim_type": "instructional",
            "document_sha256": source["document_sha256"],
            "evidence": evidence,
            "evidence_sha256": text_sha256(evidence),
            "evidence_url": source["url"],
            "fact_id": stable_fact_id(question, answer),
            "kind": "qa_resource_manual_fact",
            "paraphrase_rationale": fact["paraphrase_rationale"],
            "quality_schema": "manual-resource-fact-v1",
            "question": question,
            "resource_id": "rope365",
            "reviewer": "codex-technique-additions-v13",
            "source": "rope365",
            "source_lineage": {"artifact": portable(OUTPUT), "raw_document": portable(source["path"]), "resource_manifest": portable(RESOURCE_MANIFEST)},
            "text": rendered,
            "topic": fact["topic"],
            "url": source["url"],
            "verified_at": "2026-07-15",
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
        "method": {"authoring": "manual full-source review and hand-authored Q&A", "collision_scope": "v301 train-only projection; sealed collisions delegated to integration tooling", "selection": "one transferable technique concept from each of three distinct, previously unrepresented documents and URLs"},
        "new_independent_inputs": {"document_sha256s": 3, "expected_strata": dict(sorted(strata.items())), "urls": 3},
        "reviewed_at": "2026-07-15",
        "reviewer": "codex-technique-additions-v13",
        "schema": "manual-technique-additions-report-v13",
        "status": "segregated_pending_integration",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
