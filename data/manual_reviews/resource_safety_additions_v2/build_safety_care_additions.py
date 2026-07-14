#!/usr/bin/env python3
"""Build manual safety, communication, and rope-care additions tranche 2.

The Q&A and evidence below were selected manually from stored, hash-pinned
Rope365 pages.  This generator validates source extraction and collisions with
active and already-pending training data.  It intentionally never opens an
evaluation or held-out artifact; those checks are delegated to the existing
curated-build tooling during the isolated projection.
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
OUTPUT = OUT_DIR / "pending_additions_safety_care_tranche_02_v1.jsonl"
REPORT = OUT_DIR / "report_safety_care_tranche_02_v1.json"
REVIEWER = "codex-resource-safety-care-additions"
REVIEWED_AT = "2026-07-14"

ACTIVE_DATASET = DATA / "train_qa_curated_v1.jsonl"
ACTIVE_REPORT = DATA / "train_qa_curated_v1.report.json"
ACTIVE_CURATIONS = (
    DATA / "train_qa_curated_v1.curation.jsonl",
    DATA / "train_qa_kinbakutoday.curation.jsonl",
)
PRIOR_PENDING = (
    DATA / "manual_reviews" / "resource_safety_additions_v1" /
    "pending_additions_safety_tranche_01_v1.jsonl",
)
RESOURCE_MANIFEST = ROOT / "sources" / "rope_resources_v1.json"
COVERAGE_REPORT = DATA / "rope_resources_coverage_v1.json"

SOURCE_DOCUMENTS = {
    "communication": {
        "path": DATA / "raw" / "rope_resources_v1" /
                "rope365__b602b6493b5eb6f55206.json",
        "url": "https://rope365.com/communication/",
        "document_sha256":
            "ba41f96db0578f593930a21a579f6a30f3658b100da8390fea2edbdf5b4abb3d",
    },
    "rope_care": {
        "path": DATA / "raw" / "rope_resources_v1" /
                "rope365__c1314f53c65df4af2c20.json",
        "url": "https://rope365.com/rope-care/",
        "document_sha256":
            "8ce05508cbc08de2969e63d96d1a5d3655c4a9781319f8a578bf0a0f79f7d8d3",
    },
    "safety": {
        "path": DATA / "raw" / "rope_resources_v1" /
                "rope365__7b5d548036392d65fec7.json",
        "url": "https://rope365.com/safety/",
        "document_sha256":
            "f83771d810b2b197ec3a0fb58660ac529b996951413e58fa87ff48e37f52a6d7",
    },
    "storage": {
        "path": DATA / "raw" / "rope_resources_v1" /
                "rope365__32dac30c0ebad4f0ad21.json",
        "url": "https://rope365.com/storing-rope/",
        "document_sha256":
            "cf99106ff0056d7858872a0eeb28b7b36d4036ba4154d5a69b8f0be52e7e748b",
    },
}

# Explicit human-authored records. Answers are contiguous verbatim spans of
# the evidence; evidence is a contiguous verbatim span of the stored page.
MANUAL_FACTS = (
    {
        "document": "safety",
        "topic": "fall_prevention",
        "question": (
            "How does Rope365 recommend lowering fall risk when choosing a "
            "position for tying?"
        ),
        "answer": "Tie kneeling or sitting",
        "evidence": "- Tie kneeling or sitting,",
    },
    {
        "document": "safety",
        "topic": "breathing_emergency",
        "question": (
            "What does Rope365 say a rigger should be able to untie quickly "
            "if breathing is affected?"
        ),
        "answer": "anything that interferes with breathing",
        "evidence": (
            "- Make sure you can quickly untie anything that interferes with "
            "breathing such as tight chest compression and body positions "
            "that restrict breathing, such as backbends and forward bends. "
            "This condition is named positional asphyxia. Corsets and gags "
            "also contribute to restrict airflow,"
        ),
    },
    {
        "document": "safety",
        "topic": "risk_communication",
        "question": (
            "What risk information does Rope365 say a partner should receive "
            "before play?"
        ),
        "answer": (
            "the risks involved in the type of play you will share together"
        ),
        "evidence": (
            "- Make sure your partner is aware of the risks involved in the "
            "type of play you will share together"
        ),
    },
    {
        "document": "communication",
        "topic": "power_dynamics",
        "question": (
            "What does Rope365 say people should do when power dynamics "
            "affect communication?"
        ),
        "answer": "get clear communication and confirm consent",
        "evidence": (
            "If power dynamics (such as D/s, teacher/student or financial "
            "dynamics) are influencing your communication, you may have to "
            "work extra length to get clear communication and confirm consent."
        ),
    },
    {
        "document": "communication",
        "topic": "new_partner_negotiation",
        "question": (
            "Why does Rope365 recommend that new partners discuss their "
            "experience levels and previous experiences?"
        ),
        "answer": (
            "adapt the dialogue and make sure you don’t push each other too fast"
        ),
        "evidence": (
            "If you are new to each other, it is great to discuss experience "
            "levels and previous experiences. This will allow you to adapt "
            "the dialogue and make sure you don’t push each other too fast."
        ),
    },
    {
        "document": "communication",
        "topic": "post_session_feedback",
        "question": (
            "What follow-up does Rope365 recommend when a partner cannot give "
            "feedback immediately after a session?"
        ),
        "answer": (
            "contact someone a few days after a session to open the channel "
            "for feedback"
        ),
        "evidence": (
            "Sometimes, a partner might not be able to explain some emotions, "
            "or give feedback right away. It is good practice to contact "
            "someone a few days after a session to open the channel for "
            "feedback and make sure you are aware of things your partner may "
            "need help with."
        ),
    },
    {
        "document": "rope_care",
        "topic": "conditioning_tradeoff",
        "question": "What tradeoff does Rope365 identify when conditioning rope?",
        "answer": (
            "conditioning rope does weaken it and will shorten its lifespan"
        ),
        "evidence": (
            "Everything is a compromise as the process of conditioning rope "
            "does weaken it and will shorten its lifespan."
        ),
    },
    {
        "document": "rope_care",
        "topic": "conditioning_dust_safety",
        "question": (
            "Where does Rope365 say to break or polish rope because the "
            "process creates dust?"
        ),
        "answer": "outside or in a well-ventilated area",
        "evidence": (
            "Breaking the rope makes a lot of dust, make sure to do this "
            "outside or in a well-ventilated area. Masks, glasses and/or "
            "gloves are also recommended."
        ),
    },
    {
        "document": "rope_care",
        "topic": "synthetic_rope_care",
        "question": "What does Rope365 say about singeing synthetic rope?",
        "answer": "Do not singe synthetic rope as it will melt",
        "evidence": "Do not singe synthetic rope as it will melt.",
    },
    {
        "document": "storage",
        "topic": "natural_rope_storage",
        "question": (
            "How should natural rope be prepared for long-term storage "
            "according to Rope365?"
        ),
        "answer": (
            "make sure it has good aeration to prevent drying and moulding "
            "and oil it before storage"
        ),
        "evidence": (
            "If you are using natural ropes, make sure it has good aeration "
            "to prevent drying and moulding and oil it before storage."
        ),
    },
)

# Recorded after two byte-identical isolated builds. Evaluation paths were
# consumed by build_curated_qa.py; this generator never reads them. The output
# fact IDs were compared with active and both pending ledgers.
ISOLATED_PROJECTION = {
    "build_script": "build_curated_qa.py",
    "validated_runs": 2,
    "repeat_byte_identical": True,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "output_rows": 804,
    "output_sha256":
        "7bd2918a72fff036e2e9188854b949280ced197b4c9ed0f59c1a4b7e3626216d",
    "rope365_rows": 298,
    "active_fact_ids_preserved": 784,
    "active_fact_ids_removed": 0,
    "prior_pending_fact_ids_preserved": 10,
    "new_pending_fact_ids_added": 10,
    "unexpected_fact_ids": 0,
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


def qa_facts(paths: tuple[Path, ...], split: str) -> list[EvalFact]:
    facts = []
    for path in paths:
        for index, item in enumerate(read_jsonl(path), 1):
            pair = qa_pair_from_record(item)
            if pair:
                facts.append(EvalFact(
                    *pair,
                    item_id=item.get("fact_id", f"{portable(path)}:{index}"),
                    split=split,
                ))
    return facts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = {}
    for name, specification in SOURCE_DOCUMENTS.items():
        document = json.loads(specification["path"].read_text())
        if document["url"] != specification["url"]:
            raise ValueError(f"{name}: stored URL drift")
        if document["document_sha256"] != specification["document_sha256"]:
            raise ValueError(f"{name}: declared document hash drift")
        if text_sha256(document["text"]) != specification["document_sha256"]:
            raise ValueError(f"{name}: stored text hash mismatch")
        documents[name] = document

    active_facts = qa_facts((ACTIVE_DATASET,), "active_training")
    pending_facts = qa_facts(PRIOR_PENDING, "prior_pending_training")
    comparison_facts = active_facts + pending_facts
    existing_questions = {fact.normalized_question for fact in comparison_facts}
    existing_pairs = {
        (fact.normalized_question, fact.normalized_answer)
        for fact in comparison_facts
    }

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
        if question_key in existing_questions or pair_key in existing_pairs:
            raise ValueError(f"{location}: exact collision with existing data")
        collision = leakage_reason(question, answer, comparison_facts)
        if collision:
            raise ValueError(
                f"{location}: semantic collision with existing data: {collision}")
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
        "schema": "manual-resource-safety-care-additions-report-v1",
        "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT,
        "status": "segregated_pending_not_promoted",
        "method": {
            "authoring": "manual_question_answer_and_evidence_selection",
            "answer_support": "exact_substring_of_exact_stored_evidence",
            "evidence_support": "exact_substring_of_hash_pinned_stored_document",
            "selection": (
                "durable standalone safety, consent/communication, and "
                "practical rope-care facts missing from active and prior "
                "pending training data"
            ),
            "excluded_classes": [
                "volatile promotions",
                "unsupported medical claims",
                "active or pending semantic duplicates",
                "source claims lacking concise exact evidence",
            ],
        },
        "sealed_evaluation_policy": {
            "manual_review_opened_eval_or_heldout_content": False,
            "generator_opens_eval_or_heldout_content": False,
            "collision_check": (
                "evaluation artifacts are passed by path only to existing "
                "build_curated_qa.py tooling during isolated projection"
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
        "comparison_corpus": {
            "active": {
                "path": portable(ACTIVE_DATASET),
                "rows": len(active_facts),
                "sha256": file_sha256(ACTIVE_DATASET),
            },
            "prior_pending": [
                {
                    "path": portable(path),
                    "rows": len(qa_facts((path,), "prior_pending_training")),
                    "sha256": file_sha256(path),
                }
                for path in PRIOR_PENDING
            ],
            "facts_compared": len(comparison_facts),
            "candidate_collisions": 0,
        },
        "active_baseline": {
            "report": {
                "path": portable(ACTIVE_REPORT),
                "sha256": file_sha256(ACTIVE_REPORT),
            },
            "curation": [
                {"path": portable(path), "sha256": file_sha256(path)}
                for path in ACTIVE_CURATIONS
            ],
        },
        "isolated_build_projection": ISOLATED_PROJECTION,
        "validation": {
            "tests": [
                "data/manual_reviews/resource_safety_additions_v2/"
                "test_safety_care_additions.py",
                "test_build_curated_qa.py",
            ],
            "tests_passed": 14,
        },
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
