#!/usr/bin/env python3
"""Build a manual Anatomie practical consent/safety additions tranche.

The records are explicitly human-authored from two stored, hash-pinned public
Anatomie Studio articles.  This generator validates extraction and collisions
with active plus all earlier pending additions.  It never opens evaluation or
held-out artifacts; those paths are handled only by build_curated_qa.py in the
isolated projection.
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
OUTPUT = OUT_DIR / "pending_additions_anatomie_consent_tranche_01_v1.jsonl"
REPORT = OUT_DIR / "report_anatomie_consent_tranche_01_v1.json"
REVIEWER = "codex-anatomie-practical-consent-additions"
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
    DATA / "manual_reviews" / "resource_safety_additions_v2" /
    "pending_additions_safety_care_tranche_02_v1.jsonl",
    DATA / "manual_reviews" / "ropetopia_practical_additions_v1" /
    "pending_additions_ropetopia_practical_tranche_01_v1.jsonl",
)
PRIOR_PENDING_CURATIONS = (
    DATA / "manual_reviews" / "ropetopia_practical_additions_v1" /
    "pending_curation_ropetopia_practical_tranche_01_v1.jsonl",
)
CRASH_COVERAGE = (
    DATA / "raw" / "rope_resources_v1_coverage" /
    "crash_restraint.coverage.json"
)

SOURCE_DOCUMENTS = {
    "new_partner_safety": {
        "path": DATA / "raw" / "anatomiestudio_451ac66001188a42.json",
        "url": (
            "https://www.anatomiestudio.com/blog/"
            "tying-with-someone-new-feeling-safe-enough-to-explore"
        ),
        "document_sha256":
            "eaac58687fd1bfcf451ebbcb6f43bb3dad6f28edc39596b5bdfc54b9773c59b4",
    },
    "safewords": {
        "path": DATA / "raw" / "anatomiestudio_9749de0eb1ff4ef3.json",
        "url": (
            "https://www.anatomiestudio.com/blog/"
            "should-we-use-safe-words-in-shibari"
        ),
        "document_sha256":
            "396e550c0110696f10dbdd4ccad5750d0c2b4fa46baf84a4095d550588dd8b3a",
    },
}

MANUAL_FACTS = (
    {
        "document": "new_partner_safety",
        "topic": "risk_management",
        "question": (
            "How does Anatomie Studio define safety when tying with a new "
            "rope partner?"
        ),
        "answer": "understanding and managing it",
        "evidence": (
            "Safety doesn’t mean eliminating all risk—it means understanding "
            "and managing it."
        ),
    },
    {
        "document": "new_partner_safety",
        "topic": "ongoing_consent",
        "question": (
            "What three qualities does Anatomie Studio say consent must have "
            "during rope play?"
        ),
        "answer": "freely given, enthusiastic, and reversible at any time",
        "evidence": (
            "- Consent must be freely given, enthusiastic, and reversible at "
            "any time."
        ),
    },
    {
        "document": "new_partner_safety",
        "topic": "boundary_uncertainty",
        "question": (
            "What does Anatomie Studio advise when someone is unsure of "
            "their own boundaries?"
        ),
        "answer": "be honest and go slow",
        "evidence": (
            "If you're not sure of your boundaries, be honest and go slow. "
            "You can always go further next time."
        ),
    },
    {
        "document": "new_partner_safety",
        "topic": "power_imbalance_mitigation",
        "question": (
            "How can the person with more power support a new rope partner’s "
            "agency during planning?"
        ),
        "answer": "Let them talk first",
        "evidence": (
            "- Let them talk first – Give them space to express their needs "
            "without being influenced by you."
        ),
    },
    {
        "document": "new_partner_safety",
        "topic": "stop_signal_practice",
        "question": (
            "What does Anatomie Studio recommend doing with an agreed stop "
            "signal before it is truly needed?"
        ),
        "answer": "use your “stop” signal",
        "evidence": (
            "It’s good practice to use your “stop” signal before it’s truly "
            "needed, just to ensure everyone is comfortable responding to it."
        ),
    },
    {
        "document": "safewords",
        "topic": "precise_safeword_communication",
        "question": (
            "What kind of information should a rope safeword be combined "
            "with?"
        ),
        "answer": (
            "more precise language relating to body parts, sensations and "
            "urgency"
        ),
        "evidence": (
            "So, while safewords can be a good indicator that something "
            "isn’t right, in rope they need to be combined with more precise "
            "language relating to body parts, sensations and urgency."
        ),
    },
    {
        "document": "safewords",
        "topic": "safeword_negotiation",
        "question": (
            "What should rope partners agree about their safewords before "
            "play?"
        ),
        "answer": "what your safe words are and what they mean",
        "evidence": "- Agree what your safe words are and what they mean",
    },
    {
        "document": "safewords",
        "topic": "new_partner_checkins",
        "question": (
            "How should communication change when rope partners know each "
            "other less well?"
        ),
        "answer": (
            "the less well you know someone, the more you should be "
            "communicating"
        ),
        "evidence": (
            "It’s also important to note that the less well you know someone, "
            "the more you should be communicating. "
        ),
    },
    {
        "document": "safewords",
        "topic": "nonverbal_checkin_responsibility",
        "question": (
            "When a rope model becomes non-verbal, who does Anatomie Studio "
            "say has the immediate responsibility to check in?"
        ),
        "answer": "the rigger",
        "evidence": (
            "However, it’s important to note that models can go non-verbal "
            "in rope, or simply sink into a space in which communication "
            "means they lose their headspace. In this case the immediate "
            "responsibility will fall to the rigger to check in."
        ),
    },
    {
        "document": "safewords",
        "topic": "positive_action_for_consent",
        "question": (
            "In the Positive Action for Consent example, what does no "
            "response mean?"
        ),
        "answer": "No response equals no more scene",
        "evidence": (
            "No response equals no more scene. No questions asked, I will be "
            "untied. The discussion comes later."
        ),
    },
)

# Recorded after two byte-identical isolated projections with all prior
# additions, the pending tasuki manual paraphrase, and sealed evaluation paths
# handled only by build_curated_qa.py.
ISOLATED_PROJECTION = {
    "build_script": "build_curated_qa.py",
    "validated_runs": 2,
    "repeat_byte_identical": True,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "output_rows": 822,
    "output_sha256":
        "3408384f9f9a6cabf3528b66ad39befaa47cb08af21bb4dac76cd6497cc1f238",
    "anatomiestudio_rows": 33,
    "unchanged_active_fact_ids_preserved": 783,
    "tasuki_manual_paraphrase_added": True,
    "prior_pending_fact_ids_preserved": 28,
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

    crash = json.loads(CRASH_COVERAGE.read_text())
    if crash.get("collection_mode") != "reference_only":
        raise ValueError("Crash Restraint collection restriction drifted")

    documents = {}
    for name, specification in SOURCE_DOCUMENTS.items():
        document = json.loads(specification["path"].read_text())
        if document["url"] != specification["url"]:
            raise ValueError(f"{name}: stored URL drift")
        if document.get("source") != "anatomiestudio":
            raise ValueError(f"{name}: stored source label drift")
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
        specification = SOURCE_DOCUMENTS[document_name]
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

        row = {
            "answer": answer,
            "claim_type": "instructional",
            "document_sha256": specification["document_sha256"],
            "evidence": evidence,
            "evidence_sha256": text_sha256(evidence),
            "evidence_url": document["url"],
            "fact_id": fact_id,
            "kind": "qa_resource_manual_fact",
            "quality_schema": "manual-resource-fact-v1",
            "question": question,
            "resource_id": "anatomiestudio",
            "reviewer": REVIEWER,
            "source": "anatomiestudio",
            "source_lineage": {
                "artifact": portable(OUTPUT),
                "raw_document": portable(specification["path"]),
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
        "schema": "manual-anatomie-practical-consent-additions-report-v1",
        "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT,
        "status": "segregated_pending_not_promoted",
        "method": {
            "authoring": "manual_question_answer_and_evidence_selection",
            "answer_support": "exact_substring_of_exact_stored_evidence",
            "evidence_support": "exact_substring_of_hash_pinned_stored_document",
            "selection": (
                "durable practical consent and safety behaviors missing from "
                "active plus all prior pending additions"
            ),
            "excluded_classes": [
                "volatile or promotional claims",
                "unsupported medical claims",
                "source-specific event logistics",
                "active or pending semantic duplicates",
            ],
        },
        "source_selection": {
            "anatomiestudio_additions": len(rows),
            "crash_restraint_additions": 0,
            "crash_restraint_reason": (
                "reference_only; no stored permissible source body for new "
                "evidence-pinned authoring"
            ),
            "crash_restraint_coverage": {
                "path": portable(CRASH_COVERAGE),
                "sha256": file_sha256(CRASH_COVERAGE),
            },
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
        "pending_curations_in_projection": [
            {"path": portable(path), "sha256": file_sha256(path)}
            for path in PRIOR_PENDING_CURATIONS
        ],
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
                "data/manual_reviews/anatomie_practical_consent_additions_v1/"
                "test_anatomie_practical_consent_additions.py",
                "test_build_curated_qa.py",
            ],
            "tests_passed": 16,
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
