#!/usr/bin/env python3
"""Build three manually reviewed hogtie mobility-safety QAs."""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V306 = DATA / "manual_reviews/context_merit_audit_v306"
V17 = DATA / "manual_reviews/rope365_learning_safety_additions_v17"
sys.path[:0] = [str(ROOT), str(V306), str(V17)]

import build_context_merit_audit_v306 as baseline_builder
import build_rope365_learning_safety_additions_v17 as prior
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact, has_protocol_tokens, leakage_reason, normalize_text, parse_qa, stable_fact_id

OUT_DIR = Path(__file__).resolve().parent
OUTPUT = OUT_DIR / "pending_additions_hogtie_mobility_safety_tranche_18_v1.jsonl"
REPORT = OUT_DIR / "report_hogtie_mobility_safety_tranche_18_v1.json"
BASELINE_ROWS = 542
BASELINE_SHA256 = "ca43bcbd324a267afaef20aa69f1d2b8859b633335f6242133ac4e0233deccb4"
EXPECTED_OUTPUT_SHA256 = "5470244d47d37746d8587d10371bcbee5b834d87df78a741fb0fd103a2b830bd"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"

file_sha256, text_sha256, portable = prior.file_sha256, prior.text_sha256, prior.portable
read_jsonl, write_jsonl, select_evidence = prior.read_jsonl, prior.write_jsonl, prior.select_evidence

SOURCES = {
    "ankle_angle": {
        "path": DATA / "raw/rope_resources_v1/rope365__c90bcce96210a1a8c390.json",
        "url": "https://rope365.com/hogtie-ankles-crossing/",
        "document_sha256": "e083ef2b6d44c5bca2998ff82759df160d0913ca310d7762bcf0c2072a062bd7",
        "markers": (
            "We have to be mindful when playing in this position, changing the angle of the ankles can make the wraps around ankle more tight, causing a painful compression.",
            "- There is some space in the tie around the ankle to prevent compression with movement",
        ),
    },
    "rolling_pivot": {
        "path": DATA / "raw/rope_resources_v1/rope365__377d038c4dbf9423e3b6.json",
        "url": "https://rope365.com/hogtie-flipping/",
        "document_sha256": "c7f6d5d31614646a045dc71654f41b5cd5a7964b9ddf8d29d481377d3b1cda84",
        "markers": (
            "Moving around will be different if the tie is asymetric. Each direction will have a different result. It is safer to flip using a knee or an elbow as the pivot point, than putting a lot of weight on an ankle or wrist. It is interesting to create positions where one limb pins down another one, creating an intensified feeling of restriction.",
        ),
    },
    "crossed_leg_joints": {
        "path": DATA / "raw/rope_resources_v1/rope365__ec5a9b5c6a2209e73b93.json",
        "url": "https://rope365.com/hogtie-legs-crossing/",
        "document_sha256": "a126d132aa60596e519e44ae4edc568adff56869fa4b28939bc7428337b65f9d",
        "markers": (
            "The first step of the crossed leg tie is to experiment and observe the range of motion of the hips, knees and ankle can move to create the position. We have to be careful not to push a joint in a direction that is painful. For example, knees can only support a certain amount of twist in them.",
            "- The joint positions are within their comfort zone with no pain from the position itself",
        ),
    },
}

FACTS = (
    {
        "source_key": "ankle_angle",
        "topic": "ankle_angle_compression",
        "question": "Why should a crossing-ankles tie leave some space around the ankles?",
        "answer": "Changing the ankle angle can tighten the wraps and cause painful compression, so space helps accommodate movement.",
        "paraphrase_rationale": "This combines the page's angle-change mechanism with its checklist instruction without endorsing tighter experimental variants.",
    },
    {
        "source_key": "rolling_pivot",
        "topic": "safer_hogtie_pivot",
        "question": "What safer pivot points does Rope365 recommend when rolling someone in an asymmetric hogtie?",
        "answer": "Use a knee or elbow as the pivot rather than putting substantial body weight on an ankle or wrist.",
        "paraphrase_rationale": "This retains the source's relative safety comparison and avoids presenting hogtie rolling as risk-free.",
    },
    {
        "source_key": "crossed_leg_joints",
        "topic": "crossed_leg_joint_assessment",
        "question": "What should partners assess before tying a crossed-leg hogtie position?",
        "answer": "Explore the hips’, knees’, and ankles’ ranges of motion, and do not push any joint in a direction that causes pain.",
        "paraphrase_rationale": "This turns the source's joint-by-joint exploration and pain boundary into a concise pre-tying assessment.",
    },
)


def build_baseline(path: Path, report: Path) -> list[dict]:
    baseline_builder.build_projection(path, report)
    rows = read_jsonl(path)
    if (len(rows), file_sha256(path)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v306 drift")
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

    with tempfile.TemporaryDirectory(prefix="hogtie-mobility-safety-v18-", dir=OUT_DIR) as temp:
        d = Path(temp)
        baseline = build_baseline(d / "v306.jsonl", d / "v306.report.json")

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
                "reviewer": "codex-hogtie-mobility-safety-additions-v18",
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
                        "url": "https://rope365.com/hogtie-wrists-to-ankles/",
                        "decision": "reject",
                        "reason": "Its hand-movement and thumb-sensation monitoring guidance duplicates existing box-tie monitoring rows.",
                    },
                    {
                        "url": "https://rope365.com/hogtie-asymmetry/",
                        "decision": "reject",
                        "reason": "The useful body-adaptation theme is already covered, while the page's one-side-tighter prompt was not operationalized.",
                    },
                ],
                "method": {
                    "authoring": "manual full-source review and hand-authored Q&A",
                    "collision_scope": "v306 train-only projection; sealed collisions delegated to integration tooling",
                    "selection": "one bounded movement-safety fact from each of three distinct new documents",
                },
                "new_independent_inputs": {
                    "document_sha256s": 3,
                    "expected_strata": dict(sorted(strata.items())),
                    "urls": 3,
                },
                "reviewed_at": "2026-07-15",
                "reviewer": "codex-hogtie-mobility-safety-additions-v18",
                "schema": "manual-hogtie-mobility-safety-additions-report-v18",
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
