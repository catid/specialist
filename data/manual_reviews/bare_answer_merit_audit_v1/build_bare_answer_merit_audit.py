#!/usr/bin/env python3
"""Audit selected active short answers for clarity and source support.

This is a merit audit, not a length-normalization pass. Correct technical terms
remain concise when their questions provide sufficient context. One low-value
fill-in-the-blank fact is already covered by a prior pending quality-merit drop,
so this tranche records that disposition without creating a duplicate curation
decision. Evaluation and held-out artifacts are never opened here.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from qa_quality import normalize_text, qa_pair_from_record


DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "bare_answer_merit_audit_v1.jsonl"
CURATION = OUT_DIR / "pending_curation_bare_answer_merit_v1.jsonl"
REPORT = OUT_DIR / "report_bare_answer_merit_v1.json"
REVIEWER = "codex-bare-answer-merit-audit"
REVIEWED_AT = "2026-07-14"

ACTIVE_DATASET = DATA / "train_qa_curated_v1.jsonl"
ACTIVE_REPORT = DATA / "train_qa_curated_v1.report.json"
ACTIVE_CURATIONS = (
    DATA / "train_qa_curated_v1.curation.jsonl",
    DATA / "train_qa_kinbakutoday.curation.jsonl",
)
PRIOR_PENDING_ADDITIONS = (
    DATA / "manual_reviews" / "resource_safety_additions_v1" /
    "pending_additions_safety_tranche_01_v1.jsonl",
    DATA / "manual_reviews" / "resource_safety_additions_v2" /
    "pending_additions_safety_care_tranche_02_v1.jsonl",
    DATA / "manual_reviews" / "ropetopia_practical_additions_v1" /
    "pending_additions_ropetopia_practical_tranche_01_v1.jsonl",
    DATA / "manual_reviews" / "anatomie_practical_consent_additions_v1" /
    "pending_additions_anatomie_consent_tranche_01_v1.jsonl",
)
TASUKI_CURATION = (
    DATA / "manual_reviews" / "ropetopia_practical_additions_v1" /
    "pending_curation_ropetopia_practical_tranche_01_v1.jsonl"
)
QUALITY_MERIT_CURATION = (
    DATA / "manual_reviews" / "kinbakutoday_third_pass" /
    "overlap_quality_merit_reaudit_v1" /
    "pending_curation_quality_merit_v1.jsonl"
)

SPECS = (
    {
        "fact_id": "fact-069a861dbb2bea9e47ca",
        "question": (
            "In Anatomie Studio's consent guidance, how should mixed "
            "messages be interpreted?"
        ),
        "answer": "no",
        "raw_path": DATA / "raw" / "anatomiestudio_27ecdd4d7c9a5560.json",
        "url": "https://www.anatomiestudio.com/etiquette-and-consent",
        "document_sha256":
            "1d66ad91809311443abe27e549840e5436c510a1e66fe1e2cefd6d68c078d3c4",
        "evidence": "Mixed messages mean “no.”",
        "decision": "keep",
        "reason_code": "concise_answer_context_complete",
        "reason": (
            "The question names the consent rule and asks how mixed messages "
            "should be interpreted; the one-word answer 'no' is the exact, "
            "unambiguous source response and needs no padding."
        ),
    },
    {
        "fact_id": "fact-f4886b3a29c4d198cd92",
        "question": (
            "What is the name of the set foundation chest harness in "
            "Classical Kinbaku?"
        ),
        "answer": "TK",
        "raw_path": DATA / "raw" / "kinbakutoday_e55b7fa7c543e266.json",
        "url": "https://www.kinbakutoday.com/kasumi-hourai-interview/",
        "document_sha256":
            "faafc3d3bd0ffc86c41dcd890a2e8e84173ab18bcdaea1c801c57d9557bf6bed",
        "evidence": (
            "Though it is true that Classical Kinbaku does have a set "
            "foundation chest harness (TK) that all subsequent ties are "
            "built on, Classical Kinbaku uses leftover rope to decorate or "
            "tie things according to inspiration at that moment in time."
        ),
        "decision": "keep",
        "reason_code": "technical_term_context_complete",
        "reason": (
            "TK is the exact technical name supplied by the source, and the "
            "question already identifies its Classical Kinbaku harness role."
        ),
    },
    {
        "fact_id": "fact-dc790f8267b3115b450a",
        "question": (
            "What Japanese concept related to time or space is mentioned as "
            "being used to enhance a sense of mystery in rope bondage "
            "workshops?"
        ),
        "answer": "Ma",
        "raw_path": DATA / "raw" / "kinbakutoday_be37dae4ec8e0d88.json",
        "url": (
            "https://www.kinbakutoday.com/"
            "kinbaku-an-evolving-era-part-1/"
        ),
        "document_sha256":
            "363c1e619b58cdad816f1ae9166ed858dc54438809e818920ce7b21df68a2c12",
        "evidence": (
            "Japanese words and concepts are used, not completely "
            "superfluously, but often to enhance a sense of mystery, feeding "
            "the necessity to follow workshops to uncover secrets in "
            "concepts like Zen, wabi–sabi or Ma (time; pause/space)."
        ),
        "decision": "keep",
        "reason_code": "technical_term_context_complete",
        "reason": (
            "Ma is the exact named Japanese concept, while the question "
            "already supplies its time/space meaning and workshop context."
        ),
    },
    {
        "fact_id": "fact-6c642360467617e05b13",
        "question": "What phrase completes the statement '100 individuals, 100 ...'?",
        "answer": "Ryū",
        "raw_path": DATA / "raw" / "kinbakutoday_dfc9527c49ca8ad6.json",
        "url": (
            "https://www.kinbakutoday.com/"
            "kinbaku-an-evolving-era-part-3/"
        ),
        "document_sha256":
            "056e41bc760cb1cbf85f8316d633ec68b79afb947d1f2337ec5ec6316745eb0a",
        "evidence": (
            "To paraphrase the Osaka Alcatraz Circle’s purpose from two "
            "decades ago, “There are various approaches. Some of them may be "
            "different from yours. If there are 100 people, there should be "
            "100 approaches.” 100 individuals, 100 Ryū."
        ),
        "decision": "drop_already_pending",
        "reason_code": "contextless_or_low_value",
        "reason": (
            "The fill-in fragment tests completion of a slogan rather than "
            "the durable idea that individual practitioners may have "
            "different approaches. A matching merit-based drop is already "
            "staged in the prior quality-merit curation ledger."
        ),
    },
    {
        "fact_id": "fact-08b5cb358fb1dafc0859",
        "question": "What technical word is used to describe the twist of the rope?",
        "answer": "Lay",
        "raw_path": DATA / "raw" / "esinem_b249eddc1f5e1864.json",
        "url": (
            "https://www.esinem.com/"
            "untangling-shibari-rope-and-fixing-other-problems/"
        ),
        "document_sha256":
            "f78e9432711b63d7641695edbfc171b7b035f7d4037001f0ff5a5b3c7c3f5541",
        "evidence": (
            "‘Lay’ is the technical word used to describe the twist of the rope."
        ),
        "decision": "keep",
        "reason_code": "technical_term_context_complete",
        "reason": (
            "Lay is the exact technical term and the question fully states "
            "the rope property it names."
        ),
    },
)

# Recorded after two byte-identical isolated builds using the six production
# source inputs, every active and pending curation ledger, all four prior
# pending addition tranches, and sealed evaluation paths supplied only to the
# existing builder. The manual reviewer did not open evaluation content.
ISOLATED_PROJECTION = {
    "active_after_quality_merit_drops": 762,
    "build_script": "build_curated_qa.py",
    "output_rows": 800,
    "output_sha256":
        "bade85c91d1f6ad7891ebcd58078c3dd93af46592cd1360b4a206ec83075237e",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_byte_identical": True,
    "reviewed_already_pending_drop_applied": True,
    "reviewed_keep_fact_ids_preserved": 4,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as destination:
        for row in rows:
            destination.write(json.dumps(
                row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    active = {row["fact_id"]: row for row in read_jsonl(ACTIVE_DATASET)}
    quality_pending = {
        row["fact_id"]: row for row in read_jsonl(QUALITY_MERIT_CURATION)
    }
    audits = []
    for index, specification in enumerate(SPECS, 1):
        fact_id = specification["fact_id"]
        if fact_id not in active:
            raise ValueError(f"{fact_id}: reviewed fact is not active")
        row = active[fact_id]
        if qa_pair_from_record(row) != (
                specification["question"], specification["answer"]):
            raise ValueError(f"{fact_id}: active QA drift")
        if row.get("document_sha256") != specification["document_sha256"]:
            raise ValueError(f"{fact_id}: active document hash drift")
        if row.get("url") != specification["url"]:
            raise ValueError(f"{fact_id}: active URL drift")

        document = json.loads(specification["raw_path"].read_text())
        if document["url"] != specification["url"]:
            raise ValueError(f"{fact_id}: raw URL drift")
        if text_sha256(document["text"]) != specification["document_sha256"]:
            raise ValueError(f"{fact_id}: raw text hash mismatch")
        if specification["evidence"] not in document["text"]:
            raise ValueError(f"{fact_id}: evidence is not exact stored text")
        if normalize_text(specification["answer"]) not in normalize_text(
                specification["evidence"]):
            raise ValueError(f"{fact_id}: answer lacks source support")

        if specification["decision"] == "drop_already_pending":
            pending = quality_pending.get(fact_id)
            if pending is None:
                raise ValueError(f"{fact_id}: expected prior pending drop")
            if (pending["action"] != "drop" or
                    pending["expected_question"] != specification["question"] or
                    pending["expected_answer"] != specification["answer"] or
                    pending["reason_code"] != specification["reason_code"]):
                raise ValueError(f"{fact_id}: prior pending drop drift")
            pending_path = portable(QUALITY_MERIT_CURATION)
        else:
            if fact_id in quality_pending:
                raise ValueError(f"{fact_id}: keep conflicts with pending drop")
            pending_path = None

        audits.append({
            "answer": specification["answer"],
            "audit_index": index,
            "decision": specification["decision"],
            "document_sha256": specification["document_sha256"],
            "evidence": specification["evidence"],
            "evidence_sha256": text_sha256(specification["evidence"]),
            "fact_id": fact_id,
            "pending_curation": pending_path,
            "question": specification["question"],
            "raw_document": portable(specification["raw_path"]),
            "reason": specification["reason"],
            "reason_code": specification["reason_code"],
            "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER,
            "schema": "bare-answer-merit-audit-v1",
            "source_support": "normalized_extractive",
            "url": specification["url"],
        })

    write_jsonl(AUDIT, audits)
    # No new decision is warranted: four facts are context-complete and the
    # sole drop is already represented in the append-safe quality-merit ledger.
    CURATION.write_text("")

    decision_counts = collections.Counter(row["decision"] for row in audits)
    reason_counts = collections.Counter(row["reason_code"] for row in audits)
    report = {
        "schema": "bare-answer-merit-audit-report-v1",
        "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT,
        "status": "segregated_pending_not_promoted",
        "method": {
            "principle": (
                "Judge standalone clarity in the context of the complete "
                "question; do not lengthen correct technical terms merely "
                "because their answers are short."
            ),
            "source_support": (
                "Every answer and evidence span verified against a stored, "
                "document-hash-pinned source."
            ),
            "heldout_content_opened": False,
        },
        "audit": {
            "path": portable(AUDIT),
            "sha256": file_sha256(AUDIT),
            "rows": len(audits),
            "by_decision": dict(sorted(decision_counts.items())),
            "by_reason": dict(sorted(reason_counts.items())),
        },
        "new_pending_curation": {
            "path": portable(CURATION),
            "sha256": file_sha256(CURATION),
            "decisions": 0,
            "reason": (
                "Four reviewed facts merit keeping; the sole merit drop is "
                "already append-safely staged in a prior pending ledger."
            ),
        },
        "prior_pending": {
            "additions": [
                {
                    "path": portable(path),
                    "rows": len(read_jsonl(path)),
                    "sha256": file_sha256(path),
                }
                for path in PRIOR_PENDING_ADDITIONS
            ],
            "curations": [
                {
                    "path": portable(path),
                    "rows": len(read_jsonl(path)),
                    "sha256": file_sha256(path),
                }
                for path in (QUALITY_MERIT_CURATION, TASUKI_CURATION)
            ],
        },
        "active_baseline": {
            "dataset": {
                "path": portable(ACTIVE_DATASET),
                "rows": len(active),
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
        "sealed_evaluation_policy": {
            "manual_review_opened_eval_or_heldout_content": False,
            "generator_opens_eval_or_heldout_content": False,
            "collision_check": (
                "sealed evaluation paths are handled only by existing "
                "build_curated_qa.py during isolated projection"
            ),
        },
        "isolated_build_projection": ISOLATED_PROJECTION,
        "validation": {
            "combined_tests_passed": 16,
            "test_paths": [
                "data/manual_reviews/bare_answer_merit_audit_v1/"
                "test_bare_answer_merit_audit.py",
                "test_build_curated_qa.py",
            ],
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
