#!/usr/bin/env python3
"""Build a pending RopeTopia-lineage practical/safety additions tranche.

The recovered RopeTopia host exposes only a demo and not its former article
bodies.  The records below therefore use stored public WykD successor copies
for three articles that correspond to entries in the manually reviewed
RopeTopia manifest.  They are labeled transparently as successor sources; no
claim is made that the recovered-site bodies were available or copied.

This generator also stages one evidence-based edit for an active Rope365 fact.
It never opens evaluation or held-out artifacts.  Those collision checks are
delegated to the existing curated-build tooling in an isolated projection.
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
ADDITIONS = OUT_DIR / "pending_additions_ropetopia_practical_tranche_01_v1.jsonl"
CURATION = OUT_DIR / "pending_curation_ropetopia_practical_tranche_01_v1.jsonl"
REPORT = OUT_DIR / "report_ropetopia_practical_tranche_01_v1.json"
REVIEWER = "codex-ropetopia-practical-additions"
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
)
ROPETOPIA_MANIFEST = ROOT / "sources" / "rope_topia_manual_v1.json"

SOURCE_DOCUMENTS = {
    "injury_accountability": {
        "path": DATA / "raw" / "wykd_19d6a26116e26c70.json",
        "url": (
            "https://wykd.com/random/2013/11/30/"
            "self-awareness-luck-and-responsibility-in-rope-bondage-injuries/"
        ),
        "document_sha256":
            "5aa46db70e83e88c8d2fc9b0fb6a0e28e3339ce8d061e2a1ed04073dac5bdc98",
        "manifest_resource_id":
            "luck_self_awareness_responsibility_injuries",
        "original_ropetopia_url": (
            "https://rope-topia.com/2013/11/"
            "luck-self-awareness-responsibility-rope-bondage-injuries/"
        ),
    },
    "newcomer_consent": {
        "path": DATA / "raw" / "wykd_944e4e6d621a97c9.json",
        "url": (
            "https://wykd.com/learning/2012/11/03/"
            "newness-and-getting-out-into-the-kink-community/"
        ),
        "document_sha256":
            "03bb1af918ad3ed44208dc4805e48e40718f47d3ffc8994d47a54035e70140d2",
        "manifest_resource_id": "out_into_kink_community",
        "original_ropetopia_url": (
            "https://rope-topia.com/newcomers-information/"
            "out-into-the-kink-community/"
        ),
    },
    "ichinawa": {
        "path": DATA / "raw" / "wykd_a74fec63b0114fff.json",
        "url": (
            "https://wykd.com/shibari-kinbaku-bondage-teaching/2012/10/14/"
            "ichinawa-ippon-me-no-nawa-and-one-rope/"
        ),
        "document_sha256":
            "ca4af3374c7b1554a16f969a19a55d3e8acd8ff776f675eaec19b1779ec32511",
        "manifest_resource_id": "ichinawa_ippon_me_no_nawa_one_rope",
        "original_ropetopia_url": (
            "https://rope-topia.com/2012/10/"
            "ichinawa-ippon-me-no-nawa-and-one-rope/"
        ),
    },
}

EDIT_SOURCE = {
    "path": DATA / "raw" / "rope_resources_v1" /
            "rope365__7c520e466d199159381f.json",
    "url": "https://rope365.com/box-tie-chest/",
    "document_sha256":
        "86e7633af6f52f59fa1aff78bc590977607c79c662889ec3ffa253f702fd9755",
}

MANUAL_FACTS = (
    {
        "document": "injury_accountability",
        "topic": "recurring_injury_response",
        "question": (
            "What should a rigger do after causing multiple injuries that "
            "are not happening to other people?"
        ),
        "answer": (
            "stop rigging so much, to go and look at the common themes in "
            "these injuries, to work very hard to stop them occurring again"
        ),
        "evidence": (
            "Now I would expect someone who has inflicted multiple injuries "
            "which aren’t happening to other people to stop rigging so much, "
            "to go and look at the common themes in these injuries, to work "
            "very hard to stop them occurring again."
        ),
    },
    {
        "document": "injury_accountability",
        "topic": "recurring_injury_accountability",
        "question": (
            "When injuries recur across multiple models and sessions, what "
            "common factor does the article say is likely?"
        ),
        "answer": "the person tying",
        "evidence": (
            "When it happens to multiple models in multiple sessions you "
            "have to look for the factor that’s common to all the injuries. "
            "And that common factor is likely to be the person tying."
        ),
    },
    {
        "document": "newcomer_consent",
        "topic": "event_consent",
        "question": (
            "What should newcomers to kink events expect regarding physical "
            "contact?"
        ),
        "answer": (
            "Not to be touched, slapped, spanked, groped, played with or "
            "otherwise molested without consent"
        ),
        "evidence": (
            "- Not to be touched, slapped, spanked, groped, played with or "
            "otherwise molested without consent"
        ),
    },
    {
        "document": "newcomer_consent",
        "topic": "dominant_consent",
        "question": (
            "When does the newcomer article say a dominant has the right to "
            "tell a submissive what to do?"
        ),
        "answer": "when you have actual consent from the individual",
        "evidence": (
            "The only time you will is when you have actual consent from the "
            "individual. You have no automatic right to demand anything of "
            "anyone."
        ),
    },
    {
        "document": "newcomer_consent",
        "topic": "submissive_consent",
        "question": (
            "What two conditions does the newcomer article give for a new "
            "submissive to follow instructions in a D/s context?"
        ),
        "answer": "you actually want to and consent to it",
        "evidence": (
            "New submissives, you absolutely do not have to submit  obey or "
            "otherwise do what anyone tells you in a d/s context unless you "
            "actually want to and consent to it. Nobody has any automatic "
            "right to demand anything of you."
        ),
    },
    {
        "document": "ichinawa",
        "topic": "japanese_rope_terminology",
        "question": "When would WykD use Ipponnawa rather than Ichinawa?",
        "answer": "if you were counting ropes",
        "evidence": (
            "It would be Ipponnawa if you were counting ropes. Or slightly "
            "more accurately Ippon me no nawa."
        ),
    },
    {
        "document": "ichinawa",
        "topic": "ichinawa_technique",
        "question": (
            "How does WykD distinguish Ichinawa from merely practicing with "
            "one rope?"
        ),
        "answer": (
            "a technique in its own right that specifically only ever uses "
            "one rope"
        ),
        "evidence": (
            "the Ichinawa technique which is a technique that does not exist "
            "just for practice but as a technique in its own right that "
            "specifically only ever uses one rope."
        ),
    },
    {
        "document": "ichinawa",
        "topic": "terminology_variation",
        "question": (
            "Does WykD claim Ichinawa is the only correct term for the "
            "technique?"
        ),
        "answer": "not necessarily the only correct usage",
        "evidence": (
            "I must stress that this is ‘a correct usage‘ but not necessarily "
            "the only correct usage."
        ),
    },
)

CURATION_DECISION = {
    "action": "edit",
    "fact_id": "fact-aa8ca266f2b3c713cbb1",
    "expected_question": "What shape does the tasuki technique create in the back?",
    "expected_answer": "X",
    "question": "What shape does the tasuki technique create in the back?",
    "answer": "an X on the back",
    "evidence": (
        "The kimono tie is inspired by the Japanese tasuki, which is a "
        "technique to tie the kimono sleeves that creates a X in the back."
    ),
    "evidence_url": "https://rope365.com/box-tie-chest/",
    "support_type": "manual_paraphrase",
    "paraphrase_rationale": (
        "The source's 'a X in the back' unambiguously denotes an X located "
        "on the back. The edit corrects the article and preposition without "
        "changing the stated shape or body location."
    ),
    "reason": (
        "The bare answer 'X' is technically correct but underspecified. The "
        "stored source says the tasuki creates 'a X in the back'; the staged "
        "manual paraphrase preserves that meaning in grammatical English."
    ),
    "reason_code": "answer_too_bare",
    "reviewer": REVIEWER,
    "reviewed_at": REVIEWED_AT,
}

# Recorded after two byte-identical isolated builds with the active curation
# ledgers, this pending edit, both prior pending additions, and sealed eval
# paths passed only to build_curated_qa.py.
ISOLATED_PROJECTION = {
    "build_script": "build_curated_qa.py",
    "validated_runs": 2,
    "repeat_byte_identical": True,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "output_rows": 812,
    "output_sha256":
        "3d80b0ca574e3e8f550d483829b8897db8f53458e70ef282f9686393dc7eec5e",
    "wykd_successor_rows": 8,
    "unchanged_active_fact_ids_preserved": 783,
    "original_edited_fact_id_removed": True,
    "edited_fact_id_added": True,
    "prior_pending_fact_ids_preserved": 20,
    "new_pending_fact_ids_added": 8,
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

    manifest = json.loads(ROPETOPIA_MANIFEST.read_text())
    manifest_resources = {
        item["id"]: item for item in manifest["resources"]
    }
    documents = {}
    for name, specification in SOURCE_DOCUMENTS.items():
        document = json.loads(specification["path"].read_text())
        if document["url"] != specification["url"]:
            raise ValueError(f"{name}: stored successor URL drift")
        if document.get("source") != "wykd":
            raise ValueError(f"{name}: successor source is not labeled WykD")
        if text_sha256(document["text"]) != specification["document_sha256"]:
            raise ValueError(f"{name}: stored text hash mismatch")
        resource = manifest_resources[specification["manifest_resource_id"]]
        if resource["canonical_url"] != specification["original_ropetopia_url"]:
            raise ValueError(f"{name}: RopeTopia manifest mapping drift")
        documents[name] = document

    edit_document = json.loads(EDIT_SOURCE["path"].read_text())
    if edit_document["url"] != EDIT_SOURCE["url"]:
        raise ValueError("tasuki edit source URL drift")
    if edit_document["document_sha256"] != EDIT_SOURCE["document_sha256"]:
        raise ValueError("tasuki edit declared document hash drift")
    if text_sha256(edit_document["text"]) != EDIT_SOURCE["document_sha256"]:
        raise ValueError("tasuki edit source text hash mismatch")
    if CURATION_DECISION["evidence"] not in edit_document["text"]:
        raise ValueError("tasuki edit evidence is not exact stored text")
    if CURATION_DECISION.get("support_type") != "manual_paraphrase":
        raise ValueError("tasuki grammar repair must be marked manual_paraphrase")
    if not CURATION_DECISION.get("paraphrase_rationale", "").strip():
        raise ValueError("tasuki manual paraphrase lacks a rationale")

    active_rows = read_jsonl(ACTIVE_DATASET)
    matching_edit_rows = [
        row for row in active_rows
        if row.get("fact_id") == CURATION_DECISION["fact_id"]
    ]
    if len(matching_edit_rows) != 1:
        raise ValueError("expected exactly one active tasuki fact")
    edit_row = matching_edit_rows[0]
    edit_pair = qa_pair_from_record(edit_row)
    if edit_pair != (
            CURATION_DECISION["expected_question"],
            CURATION_DECISION["expected_answer"]):
        raise ValueError("tasuki curation decision is stale")
    if edit_row.get("document_sha256") != EDIT_SOURCE["document_sha256"]:
        raise ValueError("tasuki active fact source hash drift")

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
            "original_ropetopia_url":
                specification["original_ropetopia_url"],
            "quality_schema": "manual-resource-fact-v1",
            "question": question,
            "resource_id": "rope_topia_wykd_successor",
            "reviewer": REVIEWER,
            "source": "wykd",
            "source_lineage": {
                "artifact": portable(ADDITIONS),
                "raw_successor_document": portable(specification["path"]),
                "relationship": (
                    "WykD successor article corresponding to a RopeTopia "
                    "manifest entry; not a recovered RopeTopia article body"
                ),
                "ropetopia_manifest": portable(ROPETOPIA_MANIFEST),
                "ropetopia_manifest_resource_id":
                    specification["manifest_resource_id"],
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

    with ADDITIONS.open("w") as destination:
        for row in rows:
            destination.write(json.dumps(
                row, ensure_ascii=False, sort_keys=True) + "\n")
    CURATION.write_text(json.dumps(
        CURATION_DECISION, ensure_ascii=False, sort_keys=True) + "\n")

    edited_fact_id = stable_fact_id(
        CURATION_DECISION["question"], CURATION_DECISION["answer"])
    report = {
        "schema": "manual-ropetopia-practical-additions-report-v1",
        "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT,
        "status": "segregated_pending_not_promoted",
        "source_scope": {
            "recovered_ropetopia_article_bodies_used": False,
            "reason": (
                "The recovered RopeTopia host is a limited demo without the "
                "former article bodies."
            ),
            "successor_documents_used": len(SOURCE_DOCUMENTS),
            "relationship": (
                "Stored public WykD successor articles corresponding to "
                "manually indexed RopeTopia resources"
            ),
        },
        "method": {
            "authoring": "manual_question_answer_and_evidence_selection",
            "answer_support": "exact_substring_of_exact_stored_evidence",
            "evidence_support": "exact_substring_of_hash_pinned_stored_document",
            "selection": (
                "durable practical, safety, consent, and terminology facts "
                "missing from active plus both prior pending tranches"
            ),
            "excluded_classes": [
                "volatile or promotional claims",
                "contextless personal trivia",
                "unsupported medical claims",
                "active or pending semantic duplicates",
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
                "successor_url": specification["url"],
                "original_ropetopia_url":
                    specification["original_ropetopia_url"],
                "manifest_resource_id":
                    specification["manifest_resource_id"],
            }
            for name, specification in sorted(SOURCE_DOCUMENTS.items())
        },
        "ropetopia_manifest": {
            "path": portable(ROPETOPIA_MANIFEST),
            "sha256": file_sha256(ROPETOPIA_MANIFEST),
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
        "pending_curation": {
            "path": portable(CURATION),
            "sha256": file_sha256(CURATION),
            "decisions": 1,
            "original_fact_id": CURATION_DECISION["fact_id"],
            "edited_fact_id": edited_fact_id,
            "original_answer": CURATION_DECISION["expected_answer"],
            "edited_answer": CURATION_DECISION["answer"],
            "support_type": CURATION_DECISION["support_type"],
            "merit_verdict": (
                "edit warranted; staged a concise grammatical manual "
                "paraphrase with an explicit meaning-preservation rationale"
            ),
            "source": {
                "path": portable(EDIT_SOURCE["path"]),
                "file_sha256": file_sha256(EDIT_SOURCE["path"]),
                "document_sha256": EDIT_SOURCE["document_sha256"],
                "url": EDIT_SOURCE["url"],
            },
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
                "data/manual_reviews/ropetopia_practical_additions_v1/"
                "test_ropetopia_practical_additions.py",
                "test_build_curated_qa.py",
            ],
            "tests_passed": 16,
        },
        "output": {
            "path": portable(ADDITIONS),
            "rows": len(rows),
            "sha256": file_sha256(ADDITIONS),
            "unique_fact_ids": len(seen_fact_ids),
            "unique_questions": len(seen_questions),
            "by_document": dict(sorted(counts_by_document.items())),
            "by_topic": dict(sorted(counts_by_topic.items())),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
