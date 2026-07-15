#!/usr/bin/env python3
"""Build three manually reviewed structure, technique, and psychology QAs."""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V307 = DATA / "manual_reviews/context_merit_audit_v307"
V18 = DATA / "manual_reviews/hogtie_mobility_safety_additions_v18"
sys.path[:0] = [str(ROOT), str(V307), str(V18)]

import build_context_merit_audit_v307 as baseline_builder
import build_hogtie_mobility_safety_additions_v18 as prior
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact, has_protocol_tokens, leakage_reason, normalize_text, parse_qa, stable_fact_id

OUT_DIR = Path(__file__).resolve().parent
OUTPUT = OUT_DIR / "pending_additions_structure_psychology_tranche_19_v1.jsonl"
REPORT = OUT_DIR / "report_structure_psychology_tranche_19_v1.json"
BASELINE_ROWS = 545
BASELINE_SHA256 = "deff6ff8a1902d6b83d6d81351d7596811e863008ecbebc42914fdcd9ed64df3"
EXPECTED_OUTPUT_SHA256 = "bd5163d47ffd2017e0fd3031b8145ac879bccfe25615a6d2ce1d35f2e9a6c7f9"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"

file_sha256, text_sha256, portable = prior.file_sha256, prior.text_sha256, prior.portable
read_jsonl, write_jsonl, select_evidence = prior.read_jsonl, prior.write_jsonl, prior.select_evidence

SOURCES = {
    "shoulder_anchor": {
        "path": DATA / "raw/rope_resources_v1/rope365__ca0c7f2cc2075d371a1e.json",
        "url": "https://rope365.com/hogtie-shoulders-to-ankles/",
        "document_sha256": "8b7b5b5d8504b04c81489b54388c6b81f01c67d302b2614ed4226366e703db56",
        "markers": (
            "The hogtie position can be difficult to maintain and attaching first to the shoulders helps removing some pressure from the wrists and make the tie more sustainable. To achieve this, it is great to start with a solid shoulder structure.",
        ),
    },
    "chain_stitch": {
        "path": DATA / "raw/rope_resources_v1/rope365__255e7fa505a2f3f096a9.json",
        "url": "https://rope365.com/more-techniques/",
        "document_sha256": "961189902e2693b8d1fe46686efeadc87d2f217688c6cb40a74de688755754fc",
        "markers": (
            "These can be very fast to tie and untie because there is no need to pull the whole rope through. Use this technique to wrap the body quickly.",
        ),
    },
    "psychological_play": {
        "path": DATA / "raw/rope_resources_v1/rope365__2f1fb4976e59f3455632.json",
        "url": "https://rope365.com/mind/",
        "document_sha256": "258644741aabef170eedd3bcc74142cdf2416465b036a89a2cc4618bce3f41f1",
        "markers": (
            "Bondage can go beyond the body and become a very cerebral experience. The vast world of BDSM is full of ideas of concepts you can combine with rope to intensify the experience. It can be used to bring someone to a happy place, or to explore darker sides of themselves. Playing with the mind can also become a high risk activity, mental health self-awareness and communication are crucial to mitigate those risks.",
        ),
    },
}

FACTS = (
    {
        "source_key": "shoulder_anchor",
        "topic": "shoulder_anchor_load_distribution",
        "question": "What wrist-pressure safety benefit does Rope365 describe for a shoulder structure in a hogtie?",
        "answer": "Anchoring to the shoulders first can take some pressure off the wrists before the ankles are connected.",
        "paraphrase_rationale": "This preserves the source's load-distribution rationale without reproducing its higher-risk armbinder instructions or anatomical claims.",
    },
    {
        "source_key": "chain_stitch",
        "topic": "chain_stitch_speed",
        "question": "What advantage does Rope365 describe for a chain-stitch rope pattern?",
        "answer": "It can be fast to tie and untie because the whole rope does not need to be pulled through.",
        "paraphrase_rationale": "This keeps the stated handling advantage and omits the same page's high-force pulley prompt.",
    },
    {
        "source_key": "psychological_play",
        "topic": "psychological_play_risk",
        "question": "What risk-mitigation priorities does Rope365 give for psychological rope play?",
        "answer": "Because mind-focused play can be high risk, it calls mental-health self-awareness and communication crucial.",
        "paraphrase_rationale": "This retains the page's explicit risk framing and two mitigation priorities without operationalizing its humiliation, fear, or isolation prompts.",
    },
)


def build_baseline(path: Path, report: Path) -> list[dict]:
    baseline_builder.build_projection(path, report)
    rows = read_jsonl(path)
    if (len(rows), file_sha256(path)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v307 drift")
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = {}
    for key, source in SOURCES.items():
        document = json.loads(source["path"].read_text())
        if (document["url"], document["document_sha256"], text_sha256(document["text"])) != (
            source["url"], source["document_sha256"], source["document_sha256"]
        ):
            raise ValueError(f"{key}: source drift")
        docs[key] = document
    with tempfile.TemporaryDirectory(prefix="structure-psychology-v19-", dir=OUT_DIR) as temp:
        d = Path(temp)
        baseline = build_baseline(d / "v307.jsonl", d / "v307.report.json")

    train_facts = [EvalFact(r["question"], r["answer"], r["fact_id"], "train") for r in baseline]
    questions = {normalize_text(r["question"]) for r in baseline}
    pairs = {(normalize_text(r["question"]), normalize_text(r["answer"])) for r in baseline}
    document_ids = {r["document_sha256"] for r in baseline}
    urls = {r["url"].rstrip("/").casefold() for r in baseline}
    rows = []
    for fact in FACTS:
        source = SOURCES[fact["source_key"]]
        question, answer = fact["question"], fact["answer"]
        pair = normalize_text(question), normalize_text(answer)
        rendered = f"Question: {question}\nAnswer: {answer}"
        if (
            not question.endswith("?") or "\n" in question + answer
            or has_protocol_tokens(question) or has_protocol_tokens(answer)
            or parse_qa(rendered) != (question, answer)
        ):
            raise ValueError("noncanonical")
        if pair in pairs or pair[0] in questions or leakage_reason(question, answer, train_facts):
            raise ValueError("train collision")
        if source["document_sha256"] in document_ids or source["url"].rstrip("/").casefold() in urls:
            raise ValueError("source not novel")
        evidence = select_evidence(docs[fact["source_key"]], source["markers"])
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
            "reviewer": "codex-structure-psychology-additions-v19",
            "source": "rope365",
            "source_lineage": {"artifact": portable(OUTPUT), "raw_document": portable(source["path"]), "resource_manifest": portable(RESOURCE_MANIFEST)},
            "text": rendered,
            "topic": fact["topic"],
            "url": source["url"],
            "verified_at": "2026-07-15",
        })

    if len(rows) != 3 or len({r["fact_id"] for r in rows}) != 3:
        raise ValueError("identity drift")
    write_jsonl(OUTPUT, rows)
    output_sha = file_sha256(OUTPUT)
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and output_sha != EXPECTED_OUTPUT_SHA256:
        raise ValueError("artifact drift")
    strata = Counter(classify_stratum(r) for r in rows)
    REPORT.write_text(json.dumps({
        "artifact": {"path": portable(OUTPUT), "rows": 3, "sha256": output_sha},
        "baseline": {"rows": BASELINE_ROWS, "sha256": BASELINE_SHA256},
        "excluded_source": [
            {"url": "https://rope365.com/games/", "decision": "reject", "reason": "Its opening group-risk lesson duplicates the existing multi-person consent and exit-planning row; hazardous race, tug, and gag prompts were not operationalized."},
            {"url": "https://rope365.com/more-techniques/", "decision": "partial_use", "reason": "Only the chain-stitch handling fact was retained; the high-force pulley tightening prompt was excluded."},
            {"url": "https://rope365.com/hogtie-shoulders-to-ankles/", "decision": "partial_use", "reason": "Only the shoulder load-distribution rationale was retained; high-risk armbinder instructions and source-level nerve mapping were excluded."},
        ],
        "method": {"authoring": "manual full-source review and hand-authored Q&A", "collision_scope": "v307 train-only projection; sealed collisions delegated to integration tooling", "selection": "one bounded fact from each of three distinct new documents"},
        "new_independent_inputs": {"document_sha256s": 3, "expected_strata": dict(sorted(strata.items())), "urls": 3},
        "reviewed_at": "2026-07-15",
        "reviewer": "codex-structure-psychology-additions-v19",
        "schema": "manual-structure-psychology-additions-report-v19",
        "status": "segregated_pending_integration",
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
