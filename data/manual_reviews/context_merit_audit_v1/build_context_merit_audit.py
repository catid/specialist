#!/usr/bin/env python3
"""Audit the next 25 unreviewed active QAs at deterministic merit risk.

The ranking considers only short questions, pronoun-heavy questions, bare
answers, and likely named-person trivia. Existing manual-review identifiers
and active rows carrying review/curation provenance are excluded before
ranking. Selection is then frozen and every decision is checked against a
hash-pinned local source. Evaluation and held-out artifacts are never opened.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from qa_quality import normalize_text, qa_pair_from_record


DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v1.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v1.jsonl"
REPORT = OUT_DIR / "report_context_merit_v1.json"
REVIEWER = "codex-context-merit-audit"
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
RESOURCE_MANIFEST = ROOT / "sources" / "rope_resources_v1.json"

PRONOUNS = frozenset({
    "it", "its", "they", "them", "their", "theirs", "this", "that",
    "these", "those", "he", "him", "his", "she", "her", "hers", "there",
})
PERSON_INTERROGATIVES = frozenset({"who", "whom", "whose", "when", "where"})
ID_FIELDS = frozenset({
    "fact_id", "original_fact_id", "audited_fact_id", "active_fact_id",
})
# This tranche predates every context-merit ledger.  Keep its replay isolated
# from later context_merit_audit_v* siblings while retaining the contemporaneous
# non-context review ledgers that formed the original exclusion set.
PRIOR_CONTEXT_MERIT_DIRS = frozenset()


def raw(name: str) -> Path:
    return DATA / "raw" / name


# Decisions are manual. Evidence markers select the exact stored paragraph;
# they are not an automated merit judgment.
SPECS = (
    {
        "fact_id": "fact-1149b6fa977f50a6331c",
        "source_path": raw("kinbakutoday_b776fccd348e2538.json"),
        "evidence_marker":
            "This style has been further developed by Nureki’s deshi, Naka Akira",
        "decision": "keep",
        "reason_code": "rope_history_lineage_context_complete",
        "reason": (
            "Naka Akira's relationship to Nureki and Sugiura explains the "
            "development of a semenawa lineage; it is contextual rope history, "
            "not an isolated name lookup."
        ),
    },
    {
        "fact_id": "fact-e4fdcf1c671cd7be49ce",
        "source_path": raw("esinem_c6f5b8953b05d597.json"),
        "evidence_marker":
            "thanks to Stefano Laforgia, rope artist and BDSM educator",
        "decision": "drop",
        "reason_code": "volatile_or_promotional_person_trivia",
        "reason": (
            "The person's name appears only in a dated promotional workshop "
            "biography and contributes no durable technique, safety, or resource "
            "guidance."
        ),
    },
    {
        "fact_id": "fact-f29a8c4521af38a94e7b",
        "source_path": raw("kinbakutoday_fbda0ee2dffbc811.json"),
        "evidence_marker":
            "Yukimura’s development of his own personal rope style and the influence of Minomura Kou",
        "decision": "keep",
        "reason_code": "rope_history_lineage_context_complete",
        "reason": (
            "The question preserves a source-attributed influence on a distinct "
            "rope style and is meaningful lineage history."
        ),
    },
    {
        "fact_id": "fact-d6315222649f19a9aeb9",
        "source_path": raw("kinbakutoday_3fe23acd4ef143eb.json"),
        "evidence_marker": "Seems to me like Minomura Kou was his biggest inspiration",
        "decision": "edit",
        "question": (
            "Who did Naka Akira suggest was Nureki Chimuo’s biggest inspiration "
            "after Kitan Club led them to meet and work together?"
        ),
        "answer": "Minomura Kou",
        "reason_code": "restore_subjective_attribution_and_context",
        "reason": (
            "The original presents an interviewee's qualified impression as an "
            "unattributed fact; the edit restores Naka Akira's attribution and "
            "the source's collaboration context."
        ),
    },
    {
        "fact_id": "fact-d51f978bc64b51b4c65e",
        "source_path": raw("rope365_682937f92222bf87.json"),
        "evidence_marker": "The History & Myths of Japanese Bondage by Midori",
        "decision": "keep",
        "reason_code": "useful_named_resource",
        "reason": (
            "The title-and-author pair identifies a specific further-learning "
            "resource listed by Rope365."
        ),
    },
    {
        "fact_id": "fact-0a49ad482749ece6643a",
        "source_path": raw("esinem_05356ae92c0e84da.json"),
        "evidence_marker": "militaristic background of its antecedent, hojojutsu",
        "decision": "keep",
        "reason_code": "technical_history_term_context_complete",
        "reason": (
            "Hojojutsu is the exact historical antecedent named by the source, "
            "and the question states why the relationship matters."
        ),
    },
    {
        "fact_id": "fact-e41fdaf9f974e31e9179",
        "source_path": raw("kinbakutoday_43095aa305e35a41.json"),
        "evidence_marker": "when the shibarite feels that something may be wrong",
        "decision": "keep",
        "reason_code": "safety_term_context_complete",
        "reason": (
            "The concise term is embedded in actionable partner-checking safety "
            "context and should not be padded."
        ),
    },
    {
        "fact_id": "fact-789b90e4c19cf97de3a7",
        "source_path": raw("kinbakutoday_454370c8a4f42708.json"),
        "evidence_marker": "Kazuya Mukai’s live performances with his model Junko Aoki",
        "decision": "keep",
        "reason_code": "rope_performance_history_context_complete",
        "reason": (
            "The model is identified within a dated performance lineage that the "
            "source says influenced later Japanese kinbaku shows."
        ),
    },
    {
        "fact_id": "fact-45ad14e9dd7ee2526c24",
        "source_path": raw("rope365_1616ffce57d993f3.json"),
        "evidence_marker": "How To Tie A Boxtie (TK) by Bondage Tuition",
        "decision": "keep",
        "reason_code": "useful_named_resource",
        "reason": (
            "The title-and-author pair identifies a specific box-tie learning "
            "resource in Rope365's bibliography."
        ),
    },
    {
        "fact_id": "fact-bf1a51903fec57e72a27",
        "source_path": raw("kinbakutoday_6241df5273c2d304.json"),
        "evidence_marker":
            "Takase Shinobu is a model who worked with Minomura Ko in the Beautiful Bindings series",
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": (
            "The row converts a bracketed biographical aside into a named-model "
            "lookup without preserving any reusable rope-history insight."
        ),
    },
    {
        "fact_id": "fact-a771b06601588485ef2b",
        "source_path": raw("kinbakutoday_20b12901d740b104.json"),
        "evidence_marker":
            "The museum features the collection of the late Nawa Yumio",
        "decision": "edit",
        "question": (
            "Whose collection forms the basis of the Meiji University Crime and "
            "Punishment Museum, and what was he known for?"
        ),
        "answer": (
            "Nawa Yumio, who was Japan’s leading authority on Edo era punishment "
            "and law enforcement"
        ),
        "reason_code": "add_durable_historical_context",
        "reason": (
            "The edit retains the museum attribution while adding the source's "
            "reason that Nawa Yumio is relevant to restraint and punishment "
            "history."
        ),
    },
    {
        "fact_id": "fact-e96b48731c29761bc94d",
        "source_path": raw("wikipedia_093ebd176b6adaaf.json"),
        "evidence_marker": "the man who coined \"Safe, Sane, and Consensual S/M\"",
        "decision": "keep",
        "reason_code": "consent_framework_history_context_complete",
        "reason": (
            "Authorship of the influential SSC formulation is durable BDSM "
            "consent-framework history, and the organization is named."
        ),
    },
    {
        "fact_id": "fact-9fbbb47c1dabbf7b10f2",
        "source_path": raw("kinbakutoday_d7aad07b39a5b1e1.json"),
        "evidence_marker": "carrying a document, titled “The Bonds of Kinbaku”",
        "decision": "drop",
        "reason_code": "contextless_or_low_value",
        "reason": (
            "Remembering the title of a privately presented succession document "
            "is low-value lineage trivia and does not teach a transferable "
            "historical or practical concept."
        ),
    },
    {
        "fact_id": "fact-7f44ac0d2f1de5926b66",
        "source_path": raw("rope365_1616ffce57d993f3.json"),
        "evidence_marker": "their teacher Akechi Denki",
        "decision": "keep",
        "reason_code": "technique_design_lineage_context_complete",
        "reason": (
            "The question ties the teacher attribution directly to the design "
            "lineage of a specific two-rope box-tie structure."
        ),
    },
    {
        "fact_id": "fact-98221d2915fb3a6f8637",
        "source_path": raw("esinem_502c90cb43cc9dec.json"),
        "evidence_marker": "the one person continuing his name Osada Steve",
        "decision": "drop",
        "reason_code": "volatile_or_promotional_person_trivia",
        "reason": (
            "The claim is a dated promotional aside about a person's name and "
            "online presence, not stable rope instruction or contextual history."
        ),
    },
    {
        "fact_id": "fact-ae00d3e5d00d8fe80feb",
        "source_path": raw("kinbakutoday_d4dcb268cb41c5e4.json"),
        "evidence_marker": "a ventriloquized interiority that makes seme",
        "decision": "keep",
        "reason_code": "critical_history_concept_context_complete",
        "reason": (
            "The source explicitly defines this phrase, and the question fully "
            "states the critical concept it names."
        ),
    },
    {
        "fact_id": "fact-61797044f6aa991a119a",
        "source_path": raw("kinbakutoday_3376e34b04fe0fb1.json"),
        "evidence_marker": "Founded in 1983",
        "decision": "keep",
        "reason_code": "rope_media_history_context_complete",
        "reason": (
            "Cinemagic's founding year anchors the history of a major Japanese "
            "rope and SM film studio; the short answer is unambiguous."
        ),
    },
    {
        "fact_id": "fact-7be09ff3273023906b25",
        "source_path": raw("kinbakutoday_eb9778bb44eeecfc.json"),
        "evidence_marker": "editing the SM magazine Sun & Moon",
        "decision": "keep",
        "reason_code": "rope_media_history_context_complete",
        "reason": (
            "The editor-publication relationship is concise, self-contained "
            "history of early Japanese SM and kinbaku media."
        ),
    },
    {
        "fact_id": "fact-c06bbd52eab400cf8bf0",
        "source_path": raw("kinbakutoday_f57559bbb4c8b826.json"),
        "evidence_marker":
            "The name itself, Kinbiken, is a contraction of Kinbakubi kenkyūkai",
        "decision": "keep",
        "reason_code": "technical_history_term_context_complete",
        "reason": (
            "Kinbiken is the exact contraction for a historically relevant "
            "bondage study group; no explanatory padding is needed."
        ),
    },
    {
        "fact_id": "fact-77971e5d6be6742d6b54",
        "source_path": raw("kinbakutoday_938724eb415ca5c0.json"),
        "evidence_marker":
            "published his Nihon keibatsu fūzoku zukan 日本刑罰風俗図史 in 1948",
        "decision": "keep",
        "reason_code": "restraint_history_context_complete",
        "reason": (
            "The publication date anchors a named historical work used in the "
            "source's analysis of punishment and restraint depictions."
        ),
    },
    {
        "fact_id": "fact-c5329b6b0c847b0ee1bc",
        "source_path": raw("rope365_085ee19dfa760651.json"),
        "evidence_marker": "Chest Binder: Inspired by a design by honey_bare",
        "decision": "drop",
        "reason_code": "contextless_or_low_value",
        "reason": (
            "A creator-handle lookup from a photo credit supplies little durable "
            "instruction and the question leaves 'the tutorial' unidentified."
        ),
    },
    {
        "fact_id": "fact-1f47985bc9f137f5cb02",
        "source_path": raw("kinbakutoday_5a6e15f57e52a345.json"),
        "evidence_marker":
            "I think of Tamai as the person who established the modern SM show",
        "decision": "edit",
        "question": (
            "Whom does Ugo credit with establishing the modern SM show by "
            "opening SM theatre shows to the public?"
        ),
        "answer": "Tamai Keiyuu",
        "reason_code": "restore_full_name_and_historical_context",
        "reason": (
            "The edit replaces an ambiguous surname-only answer with the full "
            "source name and connects the attribution to opening the shows to "
            "the public."
        ),
    },
    {
        "fact_id": "fact-109571ca967c263f9444",
        "source_path": RESOURCE_MANIFEST,
        "resource_id": "austin_rope_slingers",
        "decision": "keep",
        "reason_code": "owner_requested_resource_directory",
        "reason": (
            "The stable canonical URL directly satisfies resource-discovery "
            "questions while avoiding volatile meeting-day claims."
        ),
    },
    {
        "fact_id": "fact-019df482b79d13db9a0f",
        "source_path": raw("esinem_d00f706084758774.json"),
        "evidence_marker": "Adding lubrication in the form of wax reduced friction",
        "decision": "keep",
        "reason_code": "rope_care_test_result_context_complete",
        "reason": (
            "The answer is an exact material from the source's rope-friction "
            "tests, and the question states the observed effect."
        ),
    },
    {
        "fact_id": "fact-426bc3d2c6101548619f",
        "source_path": raw("kinbakutoday_d4dcb268cb41c5e4.json"),
        "evidence_marker": "What I call 悦虐, “ecstatic cruelty,” means this",
        "decision": "keep",
        "reason_code": "historical_source_term_context_complete",
        "reason": (
            "The one-word Japanese term is explicitly defined and translated in "
            "the question, making it self-contained historical-source knowledge."
        ),
    },
)

EXPECTED_SELECTION = tuple(specification["fact_id"] for specification in SPECS)

# Recorded after two identity-checked isolated builds using the six production
# inputs, all 38 prior pending additions, the canonical 22 quality-merit drops,
# the tasuki edit, this tranche, and sealed evaluation paths supplied only to
# the existing builder. The projected dataset was byte-identical both times.
ISOLATED_PROJECTION = {
    "active_after_canonical_quality_merit_drops": 762,
    "active_after_this_tranche": 757,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 5,
    "new_edits_applied": 3,
    "output_rows": 795,
    "output_sha256":
        "298376bb14ee4187b93d0d5f9bd16ef3053c1d20a6b03914abf3db7aa64d4779",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 17,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "tasuki_edit_applied": True,
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


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w’'-]+", text, flags=re.UNICODE)


def reviewed_fact_ids() -> set[str]:
    reviewed = set()
    manual_root = DATA / "manual_reviews"
    for path in sorted(manual_root.rglob("*.jsonl")):
        if OUT_DIR in path.parents:
            continue
        review_dir = path.relative_to(manual_root).parts[0]
        if (re.fullmatch(r"context_merit_audit_v\d+", review_dir) and
                review_dir not in PRIOR_CONTEXT_MERIT_DIRS):
            continue
        for row in read_jsonl(path):
            for field in ID_FIELDS:
                value = row.get(field)
                if isinstance(value, str) and value.startswith("fact-"):
                    reviewed.add(value)
            for value in row.get("candidate_fact_ids", []):
                if isinstance(value, str) and value.startswith("fact-"):
                    reviewed.add(value)
    return reviewed


def risk_features(row: dict) -> dict:
    question_tokens = tokens(row["question"])
    answer_tokens = tokens(row["answer"])
    lowered = [token.lower() for token in question_tokens]
    pronoun_count = sum(token in PRONOUNS for token in lowered)
    short_question_points = (
        max(0, 10 - len(question_tokens)) if len(question_tokens) <= 9 else 0
    )
    answer_length = len(answer_tokens)
    bare_answer_points = (
        6 if answer_length <= 1 else
        5 if answer_length == 2 else
        2 if answer_length <= 4 else 0
    )
    likely_named_person = bool(re.search(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|\b[A-Z][a-z]+[’']s\b",
        row["question"],
    ))
    named_person_trivia_points = (
        6 if lowered and lowered[0] in PERSON_INTERROGATIVES and
        likely_named_person else 0
    )
    score = (short_question_points + 3 * pronoun_count +
             bare_answer_points + named_person_trivia_points)
    return {
        "answer_tokens": answer_length,
        "bare_answer_points": bare_answer_points,
        "named_person_trivia_points": named_person_trivia_points,
        "pronoun_count": pronoun_count,
        "question_tokens": len(question_tokens),
        "risk_score": score,
        "short_question_points": short_question_points,
    }


def ranked_unreviewed(active_rows: list[dict]) -> tuple[list[dict], int]:
    excluded_ids = reviewed_fact_ids()
    ranked = []
    for active_index, row in enumerate(active_rows, 1):
        if (row["fact_id"] in excluded_ids or "review" in row or
                "curation" in row):
            continue
        features = risk_features(row)
        ranked.append((
            -features["risk_score"], features["question_tokens"],
            features["answer_tokens"], row["fact_id"], active_index,
            row, features,
        ))
    ranked.sort(key=lambda item: item[:4])
    return [
        {"active_index": item[4], "row": item[5], "features": item[6]}
        for item in ranked
    ], len(excluded_ids)


def source_evidence(specification: dict, active_row: dict) -> tuple[str, str]:
    path = specification["source_path"]
    if path == RESOURCE_MANIFEST:
        manifest = json.loads(path.read_text())
        resources = {
            resource["id"]: resource for resource in manifest["resources"]
        }
        resource = resources[specification["resource_id"]]
        if (resource["name"] != "Austin Rope Slingers" or
                resource["canonical_url"] != active_row["url"] or
                file_sha256(path) != active_row["document_sha256"]):
            raise ValueError("Austin resource manifest drift")
        expected_answer = (
            f'{resource["name"]}: {resource["canonical_url"]}'
        )
        if active_row["answer"] != expected_answer:
            raise ValueError("Austin resource answer drift")
        return expected_answer, "manifest_composite"

    document = json.loads(path.read_text())
    if document["url"] != active_row["url"]:
        raise ValueError(f'{active_row["fact_id"]}: source URL drift')
    if text_sha256(document["text"]) != active_row["document_sha256"]:
        raise ValueError(f'{active_row["fact_id"]}: source text hash drift')
    marker = specification["evidence_marker"]
    matching = [line for line in document["text"].splitlines()
                if marker in line]
    if len(matching) != 1:
        raise ValueError(
            f'{active_row["fact_id"]}: expected one evidence paragraph')
    evidence = matching[0]
    supported_answer = specification.get("answer", active_row["answer"])
    if normalize_text(supported_answer) not in normalize_text(evidence):
        raise ValueError(
            f'{active_row["fact_id"]}: answer absent from evidence')
    return evidence, "normalized_extractive"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    active_rows = read_jsonl(ACTIVE_DATASET)
    active = {row["fact_id"]: row for row in active_rows}
    ranked, excluded_id_count = ranked_unreviewed(active_rows)
    selected = ranked[:25]
    selected_ids = tuple(item["row"]["fact_id"] for item in selected)
    if selected_ids != EXPECTED_SELECTION:
        raise ValueError(
            "deterministic selection drift:\nexpected " +
            repr(EXPECTED_SELECTION) + "\nfound " + repr(selected_ids))

    audits = []
    curations = []
    for audit_index, (specification, ranked_item) in enumerate(
            zip(SPECS, selected), 1):
        row = ranked_item["row"]
        fact_id = specification["fact_id"]
        if row["fact_id"] != fact_id or active[fact_id] != row:
            raise ValueError(f"{fact_id}: selected active row drift")
        question, answer = qa_pair_from_record(row)
        evidence, support_type = source_evidence(specification, row)
        decision = specification["decision"]
        audit = {
            "active_answer": answer,
            "active_index": ranked_item["active_index"],
            "active_question": question,
            "audit_index": audit_index,
            "decision": decision,
            "document_sha256": row["document_sha256"],
            "fact_id": fact_id,
            "reason": specification["reason"],
            "reason_code": specification["reason_code"],
            "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER,
            "risk_features": ranked_item["features"],
            "schema": "context-merit-audit-v1",
            "source": row["source"],
            "source_document": portable(specification["source_path"]),
            "source_document_file_sha256": file_sha256(
                specification["source_path"]),
            "source_support": support_type,
            "support_evidence": evidence,
            "support_evidence_sha256": text_sha256(evidence),
            "url": row["url"],
        }
        if decision == "edit":
            audit["edited_answer"] = specification["answer"]
            audit["edited_question"] = specification["question"]
        audits.append(audit)

        if decision in {"drop", "edit"}:
            curation = {
                "action": decision,
                "document_sha256": row["document_sha256"],
                "evidence_url": row["url"],
                "expected_answer": answer,
                "expected_question": question,
                "fact_id": fact_id,
                "reason": specification["reason"],
                "reason_code": specification["reason_code"],
                "reviewed_at": REVIEWED_AT,
                "reviewer": REVIEWER,
                "source_lineage": row["source_lineage"],
            }
            if decision == "edit":
                curation.update({
                    "answer": specification["answer"],
                    "evidence": evidence,
                    "question": specification["question"],
                    "support_type": "extractive",
                })
            curations.append(curation)

    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)

    decision_counts = collections.Counter(row["decision"] for row in audits)
    reason_counts = collections.Counter(row["reason_code"] for row in audits)
    report = {
        "schema": "context-merit-audit-report-v1",
        "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT,
        "status": "segregated_pending_not_promoted",
        "selection": {
            "active_rows": len(active_rows),
            "eligible_unreviewed_rows": len(ranked),
            "excluded_ledger_fact_ids": excluded_id_count,
            "excluded_active_review_or_curation_provenance": sum(
                "review" in row or "curation" in row for row in active_rows),
            "rows_selected": len(selected),
            "fact_ids_in_rank_order": list(selected_ids),
            "ranking": {
                "score": (
                    "short_question_points + 3*pronoun_count + "
                    "bare_answer_points + named_person_trivia_points"
                ),
                "tie_break": (
                    "risk_score descending, question tokens ascending, answer "
                    "tokens ascending, fact_id ascending"
                ),
                "short_question_points": (
                    "max(0, 10-question_tokens) for questions of at most 9 tokens"
                ),
                "bare_answer_points": "6 for <=1 token, 5 for 2, 2 for 3-4",
                "named_person_trivia_points": (
                    "6 for who/whom/whose/when/where plus a likely proper name"
                ),
            },
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
            "decisions": len(curations),
            "by_action": dict(sorted(collections.Counter(
                row["action"] for row in curations).items())),
        },
        "prior_pending": {
            "additions": [
                {"path": portable(path), "rows": len(read_jsonl(path)),
                 "sha256": file_sha256(path)}
                for path in PRIOR_PENDING_ADDITIONS
            ],
            "curations": [
                {"path": portable(path), "rows": len(read_jsonl(path)),
                 "sha256": file_sha256(path)}
                for path in (QUALITY_MERIT_CURATION, TASUKI_CURATION)
            ],
        },
        "active_baseline": {
            "dataset": {"path": portable(ACTIVE_DATASET),
                        "rows": len(active_rows),
                        "sha256": file_sha256(ACTIVE_DATASET)},
            "report": {"path": portable(ACTIVE_REPORT),
                       "sha256": file_sha256(ACTIVE_REPORT)},
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
            "combined_tests_expected": 17,
            "test_paths": [
                "data/manual_reviews/context_merit_audit_v1/"
                "test_context_merit_audit.py",
                "test_build_curated_qa.py",
            ],
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
