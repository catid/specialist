#!/usr/bin/env python3
"""Build the first manually authored, source-pinned safety additions tranche.

This script contains explicit human-authored questions, answers, and evidence.
It does not generate Q&A from source text.  Its job is to fail closed if the
stored documents drift, evidence is not extractive, or a candidate collides
with active or held-out Q&A.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from qa_quality import (EvalFact, has_protocol_tokens, leakage_reason,
                        normalize_text, parse_qa, qa_pair_from_record,
                        stable_fact_id)


DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
OUTPUT = OUT_DIR / "pending_additions_safety_tranche_01_v1.jsonl"
REPORT = OUT_DIR / "report_safety_tranche_01_v1.json"
REVIEWER = "codex-resource-safety-additions"
REVIEWED_AT = "2026-07-14"

ACTIVE_DATASET = DATA / "train_qa_curated_v1.jsonl"
ACTIVE_REPORT = DATA / "train_qa_curated_v1.report.json"
ACTIVE_CURATIONS = (
    DATA / "train_qa_curated_v1.curation.jsonl",
    DATA / "train_qa_kinbakutoday.curation.jsonl",
)
RESOURCE_MANIFEST = ROOT / "sources" / "rope_resources_v1.json"
COVERAGE_REPORT = DATA / "rope_resources_coverage_v1.json"
CRASH_COVERAGE = (
    DATA / "raw" / "rope_resources_v1_coverage" /
    "crash_restraint.coverage.json"
)

SOURCE_DOCUMENTS = {
    "safety": {
        "path": DATA / "raw" / "rope_resources_v1" /
                "rope365__7b5d548036392d65fec7.json",
        "url": "https://rope365.com/safety/",
        "document_sha256":
            "f83771d810b2b197ec3a0fb58660ac529b996951413e58fa87ff48e37f52a6d7",
    },
    "communication": {
        "path": DATA / "raw" / "rope_resources_v1" /
                "rope365__b602b6493b5eb6f55206.json",
        "url": "https://rope365.com/communication/",
        "document_sha256":
            "ba41f96db0578f593930a21a579f6a30f3658b100da8390fea2edbdf5b4abb3d",
    },
}

# Each item below was manually authored after reading the complete stored page
# and manually checking the active dataset for equivalent facts.  Evidence is
# deliberately sentence- or bullet-sized so every answer remains auditable.
MANUAL_FACTS = (
    {
        "document": "safety",
        "topic": "incident_preparation",
        "question": (
            "What preparation does Rope365 call the best way to get ready "
            "for a possible incident during rope bondage?"
        ),
        "answer": (
            "visualize what you are about to do and figure out what could "
            "go wrong"
        ),
        "evidence": (
            "The best way to be prepared for an incident is to visualize "
            "what you are about to do and figure out what could go wrong."
        ),
    },
    {
        "document": "safety",
        "topic": "emergency_preparation",
        "question": (
            "What training does Rope365 recommend for learning emergency "
            "incident response?"
        ),
        "answer": "Participating in a first aid class",
        "evidence": (
            "Participating in a first aid class is a good way to get "
            "training for emergency incident response and will cover many "
            "common injuries."
        ),
    },
    {
        "document": "safety",
        "topic": "supervision",
        "question": (
            "What does Rope365 say a rigger should do with a bound partner "
            "at all times?"
        ),
        "answer": "Stay with your bound partner at all times",
        "evidence": (
            "- Stay with your bound partner at all times (but it’s ok to "
            "make them believe you are gone 🙂"
        ),
    },
    {
        "document": "safety",
        "topic": "fall_prevention",
        "question": (
            "How does Rope365 recommend reducing tripping hazards in the "
            "rope-play space?"
        ),
        "answer": "Clean up your play space",
        "evidence": (
            "- Clean up your play space to prevent tripping hazards (mats, "
            "rope bag, other people, clutter, etc.),"
        ),
    },
    {
        "document": "safety",
        "topic": "technique_risk_assessment",
        "question": (
            "What does Rope365 recommend doing before trying a new rope "
            "technique?"
        ),
        "answer": "take the time to research the specific risks",
        "evidence": (
            "- Before trying a new technique, take the time to research the "
            "specific risks,"
        ),
    },
    {
        "document": "communication",
        "topic": "partner_vetting",
        "question": (
            "What does Rope365 mean by a safety call when first meeting a "
            "new rope partner?"
        ),
        "answer": (
            "someone who will check back on you after a certain period of "
            "time"
        ),
        "evidence": (
            "It is good practice to make the first contact with a new "
            "partner in a public space (or with trusted people around), have "
            "a safety call (someone who will check back on you after a "
            "certain period of time), and check for references."
        ),
    },
    {
        "document": "communication",
        "topic": "negotiation",
        "question": (
            "Which list-based negotiation approach does Rope365 say gives "
            "better results than a blocklist?"
        ),
        "answer": "An allowlist",
        "evidence": "An allowlist gives a better result than a blocklist.",
    },
    {
        "document": "communication",
        "topic": "partner_monitoring",
        "question": (
            "What three signals does Rope365 recommend monitoring in a "
            "partner during tying?"
        ),
        "answer": "their words, their breathing, their whole body",
        "evidence": (
            "Communication during tying is complex, it’s an exchange from "
            "both sides. Monitor your partner: their words, their breathing, "
            "their whole body."
        ),
    },
    {
        "document": "communication",
        "topic": "aftercare",
        "question": "When does Rope365 say aftercare needs are best discussed?",
        "answer": "beforehand, when emotions aren’t high",
        "evidence": (
            "It is best to discuss this beforehand, when emotions aren’t "
            "high, but you might also need to adapt depending on the "
            "circumstances."
        ),
    },
    {
        "document": "communication",
        "topic": "self_assessment",
        "question": (
            "Which two self-scan questions does Rope365 suggest asking "
            "before tying or being tied?"
        ),
        "answer": (
            "“How does my body feel?” and “In what emotional state am I?”"
        ),
        "evidence": (
            "Ask yourself: “How does my body feel?” and “In what emotional "
            "state am I?” Then translate it to the situation of starting to "
            "use rope, and ask yourself: “What can I give or receive, right "
            "now?”."
        ),
    },
)

EVAL_QA = (
    DATA / "eval_qa.jsonl",
    DATA / "eval_qa_v2.jsonl",
    DATA / "eval_qa_v3.jsonl",
    DATA / "ood_qa.jsonl",
    DATA / "ood_qa_v3.jsonl",
)
HELDOUT_DOCUMENTS = (
    DATA / "heldout_docs.jsonl",
    DATA / "ood_prose.jsonl",
    DATA / "ood_prose_v3.jsonl",
)

# Recorded after two byte-identical isolated builds using the pending ledger,
# all five held-out QA artifacts above, and both active curation ledgers.  The
# projected dataset was compared by fact_id with the active S5 snapshot.
ISOLATED_PROJECTION = {
    "build_script": "build_curated_qa.py",
    "validated_runs": 2,
    "repeat_byte_identical": True,
    "eval_fact_count": 612,
    "output_rows": 794,
    "output_sha256":
        "00d7b07ae95cefb6b406db02705914ee9d9be17bf9295c62f570d49fbed50630",
    "rope365_rows": 288,
    "active_fact_ids_preserved": 784,
    "active_fact_ids_removed": 0,
    "pending_fact_ids_added": 10,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def portable(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def normalized_url(value: str) -> str:
    return value.strip().casefold().rstrip("/")


def qa_facts(paths: tuple[Path, ...], split: str) -> list[EvalFact]:
    facts = []
    for path in paths:
        for index, item in enumerate(read_jsonl(path), 1):
            if "text" in item:
                pair = qa_pair_from_record(item)
            else:
                question, answer = item.get("question"), item.get("answer")
                pair = ((question, answer) if isinstance(question, str) and
                        isinstance(answer, str) and question and answer else None)
            if pair:
                facts.append(EvalFact(
                    *pair,
                    item_id=item.get("item_id", f"{portable(path)}:{index}"),
                    split=split,
                ))
    return facts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = {}
    for name, specification in SOURCE_DOCUMENTS.items():
        path = specification["path"]
        document = json.loads(path.read_text())
        if document["url"] != specification["url"]:
            raise ValueError(f"{name}: stored URL drift")
        if document["document_sha256"] != specification["document_sha256"]:
            raise ValueError(f"{name}: declared document hash drift")
        if text_sha256(document["text"]) != specification["document_sha256"]:
            raise ValueError(f"{name}: stored text hash mismatch")
        documents[name] = document

    crash = json.loads(CRASH_COVERAGE.read_text())
    if crash.get("collection_mode") != "reference_only":
        raise ValueError("Crash Restraint collection restriction drifted")

    active_facts = qa_facts((ACTIVE_DATASET,), "active_training")
    eval_facts = qa_facts(EVAL_QA, "heldout_qa")
    active_questions = {fact.normalized_question for fact in active_facts}
    active_pairs = {
        (fact.normalized_question, fact.normalized_answer)
        for fact in active_facts
    }

    heldout_urls = collections.Counter()
    source_urls = {
        normalized_url(specification["url"])
        for specification in SOURCE_DOCUMENTS.values()
    }
    for path in (*EVAL_QA, *HELDOUT_DOCUMENTS):
        for item in read_jsonl(path):
            url = item.get("url") or item.get("evidence_url")
            if isinstance(url, str) and normalized_url(url) in source_urls:
                heldout_urls[portable(path)] += 1
    if heldout_urls:
        raise ValueError(f"source document URL appears held out: {heldout_urls}")

    rows = []
    seen_questions = set()
    seen_pairs = set()
    seen_fact_ids = set()
    counts_by_document = collections.Counter()
    counts_by_topic = collections.Counter()
    for position, fact in enumerate(MANUAL_FACTS, 1):
        document_name = fact["document"]
        document = documents[document_name]
        question = fact["question"]
        answer = fact["answer"]
        evidence = fact["evidence"]
        location = f"manual fact {position}"

        if not question.endswith("?") or "\n" in question or "\n" in answer:
            raise ValueError(f"{location}: QA is not standalone and one-line")
        if has_protocol_tokens(question) or has_protocol_tokens(answer):
            raise ValueError(f"{location}: protocol token in QA")
        if evidence not in document["text"]:
            raise ValueError(f"{location}: evidence is not exact stored text")
        if answer not in evidence:
            raise ValueError(f"{location}: answer is not exact evidence text")
        rendered = f"Question: {question}\nAnswer: {answer}"
        if parse_qa(rendered) != (question, answer):
            raise ValueError(f"{location}: canonical QA did not round-trip")

        question_key = normalize_text(question)
        pair_key = (question_key, normalize_text(answer))
        fact_id = stable_fact_id(question, answer)
        if question_key in active_questions or pair_key in active_pairs:
            raise ValueError(f"{location}: exact collision with active data")
        active_leak = leakage_reason(question, answer, active_facts)
        if active_leak:
            raise ValueError(
                f"{location}: semantic collision with active data: {active_leak}")
        eval_leak = leakage_reason(question, answer, eval_facts)
        if eval_leak:
            raise ValueError(f"{location}: held-out leakage: {eval_leak}")
        if (question_key in seen_questions or pair_key in seen_pairs or
                fact_id in seen_fact_ids):
            raise ValueError(f"{location}: duplicate within pending tranche")

        raw_path = SOURCE_DOCUMENTS[document_name]["path"]
        row = {
            "answer": answer,
            "claim_type": "instructional",
            "document_sha256": document["document_sha256"],
            "evidence": evidence,
            "evidence_sha256": text_sha256(evidence),
            "evidence_url": document["url"],
            "fact_id": fact_id,
            "kind": "qa_resource_manual_fact",
            "quality_schema": "manual-resource-fact-v1",
            "question": question,
            "resource_id": "rope365",
            "reviewer": REVIEWER,
            "source": "rope365",
            "source_lineage": {
                "artifact": portable(OUTPUT),
                "collection_coverage": portable(COVERAGE_REPORT),
                "raw_document": portable(raw_path),
                "resource_manifest": portable(RESOURCE_MANIFEST),
            },
            "text": rendered,
            "topic": fact["topic"],
            "url": document["url"],
            "verified_at": REVIEWED_AT,
        }
        rows.append(row)
        seen_questions.add(question_key)
        seen_pairs.add(pair_key)
        seen_fact_ids.add(fact_id)
        counts_by_document[document_name] += 1
        counts_by_topic[fact["topic"]] += 1

    with OUTPUT.open("w") as destination:
        for row in rows:
            destination.write(json.dumps(
                row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "schema": "manual-resource-safety-additions-report-v1",
        "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT,
        "status": "segregated_pending_not_promoted",
        "method": {
            "authoring": "manual_question_answer_and_evidence_selection",
            "answer_support": "exact_substring_of_exact_stored_evidence",
            "evidence_support": "exact_substring_of_hash_pinned_stored_document",
            "selection": (
                "durable standalone safety and communication facts absent "
                "from the active dataset"
            ),
            "excluded_classes": [
                "volatile promotions",
                "unsupported medical claims",
                "active semantic duplicates",
                "held-out fact or document overlap",
            ],
        },
        "crash_restraint": {
            "additions": 0,
            "coverage_artifact": portable(CRASH_COVERAGE),
            "coverage_sha256": file_sha256(CRASH_COVERAGE),
            "reason": (
                "reference_only; no stored source text is available for "
                "new evidence-pinned authoring"
            ),
        },
        "sources": {
            name: {
                "path": portable(specification["path"]),
                "file_sha256": file_sha256(specification["path"]),
                "document_sha256": specification["document_sha256"],
                "url": specification["url"],
            }
            for name, specification in sorted(SOURCE_DOCUMENTS.items())
        },
        "provenance": {
            portable(RESOURCE_MANIFEST): file_sha256(RESOURCE_MANIFEST),
            portable(COVERAGE_REPORT): file_sha256(COVERAGE_REPORT),
        },
        "active_baseline": {
            "dataset": {
                "path": portable(ACTIVE_DATASET),
                "rows": len(active_facts),
                "sha256": file_sha256(ACTIVE_DATASET),
            },
            "report": {
                "path": portable(ACTIVE_REPORT),
                "sha256": file_sha256(ACTIVE_REPORT),
            },
            "curation": [
                {"path": portable(path), "sha256": file_sha256(path)}
                for path in ACTIVE_CURATIONS
            ],
        },
        "leakage_checks": {
            "active_facts_compared": len(active_facts),
            "heldout_facts_compared": len(eval_facts),
            "heldout_qa_artifacts": [portable(path) for path in EVAL_QA],
            "heldout_document_artifacts": [
                portable(path) for path in HELDOUT_DOCUMENTS
            ],
            "source_url_matches": 0,
            "candidate_collisions": 0,
        },
        "isolated_build_projection": ISOLATED_PROJECTION,
        "output": {
            "path": portable(OUTPUT),
            "rows": len(rows),
            "sha256": file_sha256(OUTPUT),
            "unique_fact_ids": len(seen_fact_ids),
            "unique_questions": len(seen_questions),
            "by_document": dict(sorted(counts_by_document.items())),
            "by_topic": dict(sorted(counts_by_topic.items())),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
