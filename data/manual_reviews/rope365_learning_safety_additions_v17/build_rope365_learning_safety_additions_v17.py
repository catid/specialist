#!/usr/bin/env python3
"""Build three manually reviewed Rope365 learning and safety QAs."""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V305 = DATA / "manual_reviews/context_merit_audit_v305"
V16 = DATA / "manual_reviews/anatomy_creativity_additions_v16"
sys.path[:0] = [str(ROOT), str(V305), str(V16)]

import build_context_merit_audit_v305 as baseline_builder
import build_anatomy_creativity_additions_v16 as prior
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact, has_protocol_tokens, leakage_reason, normalize_text, parse_qa, stable_fact_id

OUT_DIR = Path(__file__).resolve().parent
OUTPUT = OUT_DIR / "pending_additions_rope365_learning_safety_tranche_17_v1.jsonl"
REPORT = OUT_DIR / "report_rope365_learning_safety_tranche_17_v1.json"
BASELINE_ROWS = 539
BASELINE_SHA256 = "80eba8b89487052c10e046328211282e159ae334661b6776efba571d4e2824bc"
EXPECTED_OUTPUT_SHA256 = "64f8573ae90d544bced28945af616f5299c5113ac597c1dcf42d0f886957226d"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"

file_sha256, text_sha256, portable = prior.file_sha256, prior.text_sha256, prior.portable
read_jsonl, write_jsonl, select_evidence = prior.read_jsonl, prior.write_jsonl, prior.select_evidence

SOURCES = {
    "predicament_monitoring": {
        "path": DATA / "raw/rope_resources_v1/rope365__4111f411231f58a2f896.json",
        "url": "https://rope365.com/predicaments/",
        "document_sha256": "172134eb0818598f9a333de84207ff8c37c47e0c746feb8cc3feb5d790e793d2",
        "markers": (
            "Creating choices for the person being tied represents an interesting challenge, full of possibilities. A predicament is a dilemma where all the options have a negative impact. In bondage this usually takes the form of a system where the person tied has a few movement options that will create different sensations. Moving in rope can create unpredictable situations. Keep an eye to make sure the ties stay safe and be ready to untie or cut the rope quickly.",
        ),
    },
    "controlled_challenge": {
        "path": DATA / "raw/rope_resources_v1/rope365__83fa5400b96b79d87dab.json",
        "url": "https://rope365.com/challenges/",
        "document_sha256": "c2959b96a9db85027857567d13dfdad2277207956161224eb17de12fb0266cc7",
        "markers": (
            "We grow a lot from mistakes and tough situations. We can fast track these learning by challenging ourselves to explore things. Results may vary, we might find new solutions to common problems, something new to enjoy, and learn a bit more about ourselves and how we react when we face difficulties.",
            "The goal of this week is to step outside of the comfort zone and learn to face challenges in a controlled context.",
        ),
    },
    "spring_path": {
        "path": DATA / "raw/rope_resources_v1/rope365__8968990b22adfe53bf97.json",
        "url": "https://rope365.com/spring/",
        "document_sha256": "e423e2d0c49135de542ccecd5bc58d70a191d40e22f4a8b5b649f73e424b853c",
        "markers": (
            "The progression of Spring is not linear. Weeks 1-2 will take you through the core techniques, it is recommended for beginners to start there once they’ve covered the topic of the Getting Started section. The experienced practitioners will find tips and tricks to solidify their foundation. Weeks 3-7 covers multiple variations of classic ties. Then Weeks 8-13 are all about the core concepts of rope bondage and ways to play with rope. The chapters don’t have to be explored in a linear way, maybe only one of the tie variations will suit your needs, maybe you’ll want to skip right away to the more intimate activities. You get to create your own adventure.",
        ),
    },
}

FACTS = (
    {
        "source_key": "predicament_monitoring",
        "topic": "predicament_monitoring",
        "question": "What monitoring and exit preparation does Rope365 recommend for predicament bondage?",
        "answer": "Watch for unpredictable movement, keep checking that the ties remain safe, and be ready to untie or cut the rope quickly.",
        "paraphrase_rationale": "This preserves the page's bounded monitoring and emergency-exit advice while excluding its later self-choking and other hazardous prompts.",
    },
    {
        "source_key": "controlled_challenge",
        "topic": "controlled_challenge_learning",
        "question": "How can a controlled rope challenge support learning according to Rope365?",
        "answer": "It can reveal new solutions, new preferences, and how you react to difficulties while you step outside your comfort zone in a controlled setting.",
        "paraphrase_rationale": "This combines the source's stated learning outcomes with its controlled-context boundary and does not reproduce its riskier challenge prompts.",
    },
    {
        "source_key": "spring_path",
        "topic": "spring_learning_path",
        "question": "How does Rope365 suggest a beginner navigate its Spring curriculum?",
        "answer": "After covering Getting Started, begin with Weeks 1–2 for core techniques; later chapters can be explored nonlinearly to match personal needs.",
        "paraphrase_rationale": "This retains the recommended beginner entry point and the source's non-linear navigation guidance without cataloging every week.",
    },
)


def build_baseline(path: Path, report: Path) -> list[dict]:
    baseline_builder.build_projection(path, report)
    rows = read_jsonl(path)
    if (len(rows), file_sha256(path)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v305 drift")
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = {}
    for key, source in SOURCES.items():
        document = json.loads(source["path"].read_text())
        observed = (document["url"], document["document_sha256"], text_sha256(document["text"]))
        expected = (source["url"], source["document_sha256"], source["document_sha256"])
        if observed != expected:
            raise ValueError(f"{key}: source drift")
        docs[key] = document

    with tempfile.TemporaryDirectory(prefix="rope365-learning-safety-v17-", dir=OUT_DIR) as temp:
        temp_dir = Path(temp)
        baseline = build_baseline(temp_dir / "v305.jsonl", temp_dir / "v305.report.json")

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
            not question.endswith("?")
            or "\n" in question + answer
            or has_protocol_tokens(question)
            or has_protocol_tokens(answer)
            or parse_qa(rendered) != (question, answer)
        ):
            raise ValueError("noncanonical")
        if pair in pairs or pair[0] in questions or leakage_reason(question, answer, train_facts):
            raise ValueError("train collision")
        if source["document_sha256"] in document_ids or source["url"].rstrip("/").casefold() in urls:
            raise ValueError("source not novel")
        evidence = select_evidence(docs[fact["source_key"]], source["markers"])
        rows.append(
            {
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
                "reviewer": "codex-rope365-learning-safety-additions-v17",
                "source": "rope365",
                "source_lineage": {
                    "artifact": portable(OUTPUT),
                    "raw_document": portable(source["path"]),
                    "resource_manifest": portable(RESOURCE_MANIFEST),
                },
                "text": rendered,
                "topic": fact["topic"],
                "url": source["url"],
                "verified_at": "2026-07-15",
            }
        )

    if len(rows) != 3 or len({r["fact_id"] for r in rows}) != 3:
        raise ValueError("identity drift")
    write_jsonl(OUTPUT, rows)
    output_sha = file_sha256(OUTPUT)
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and output_sha != EXPECTED_OUTPUT_SHA256:
        raise ValueError("artifact drift")
    strata = Counter(classify_stratum(r) for r in rows)
    REPORT.write_text(
        json.dumps(
            {
                "artifact": {"path": portable(OUTPUT), "rows": 3, "sha256": output_sha},
                "baseline": {"rows": BASELINE_ROWS, "sha256": BASELINE_SHA256},
                "excluded_source": [
                    {
                        "url": "https://rope365.com/pain/",
                        "decision": "reject",
                        "reason": "Its safe small-steps and prior-risk-research lesson duplicates the stronger new-technique risk-reduction sequence already in v305; hazardous later prompts were not operationalized.",
                    },
                    {
                        "url": "https://rope365.com/predicaments/",
                        "decision": "partial_use",
                        "reason": "Only the opening monitoring and fast-exit guidance was retained; later self-choking and other high-risk prompts were explicitly excluded.",
                    },
                ],
                "method": {
                    "authoring": "manual full-source review and hand-authored Q&A",
                    "collision_scope": "v305 train-only projection; sealed collisions delegated to integration tooling",
                    "selection": "one bounded fact from each of three distinct new documents",
                },
                "new_independent_inputs": {
                    "document_sha256s": 3,
                    "expected_strata": dict(sorted(strata.items())),
                    "urls": 3,
                },
                "reviewed_at": "2026-07-15",
                "reviewer": "codex-rope365-learning-safety-additions-v17",
                "schema": "manual-rope365-learning-safety-additions-report-v17",
                "status": "segregated_pending_integration",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
