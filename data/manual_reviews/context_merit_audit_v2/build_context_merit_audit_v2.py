#!/usr/bin/env python3
"""Audit the next non-overlapping 25 active QAs at deterministic merit risk."""

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
AUDIT = OUT_DIR / "context_merit_audit_v2.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v2.jsonl"
REPORT = OUT_DIR / "report_context_merit_v2.json"
REVIEWER = "codex-context-merit-audit-v2"
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
CONTEXT_MERIT_V1_CURATION = (
    DATA / "manual_reviews" / "context_merit_audit_v1" /
    "pending_curation_context_merit_v1.jsonl"
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
ACTIVE_REVIEW_FIELDS = frozenset({
    "review", "curation", "reviewer", "verified_at",
})
# V2 may consume V1, but no later context-merit tranche.  Freezing this ancestry
# prevents future context_merit_audit_v* siblings from changing historical
# selection while preserving all other contemporaneous review ledgers.
PRIOR_CONTEXT_MERIT_DIRS = frozenset({"context_merit_audit_v1"})


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-65c43946e82ed5b6e3a2",
        "source_path": raw("esinem_9a5aab43708932b3.json"),
        "evidence_marker":
            "demonstrate how to use ‘ingredients’ in shibari",
        "decision": "keep",
        "reason_code": "teaching_framework_term_context_complete",
        "reason": (
            "The source explicitly defines ingredients as recurring tie "
            "components used in a transferable problem-solving method."
        ),
    },
    {
        "fact_id": "fact-9d2d0301b00637f79be6",
        "source_path": raw("kinbakutoday_5a6e15f57e52a345.json"),
        "evidence_marker":
            "started his periodical SM theatre show in Tokyo in 1976",
        "decision": "keep",
        "reason_code": "rope_performance_history_context_complete",
        "reason": (
            "The year anchors a named public SM-theatre development in Japanese "
            "rope-performance history."
        ),
    },
    {
        "fact_id": "fact-f24a16ac83f7b77c4494",
        "source_path": raw("kinbakutoday_f6ccdaa49bed3fa5.json"),
        "evidence_marker":
            "It was well known that Nureki had a strong dislike for rope performance",
        "decision": "edit",
        "question":
            "Why did Nureki Chimuo and Naka Akira object to rope performance?",
        "answer": (
            "performance often presents kinbaku in a context that misses the "
            "essential communication between the people engaging in the act"
        ),
        "reason_code": "replace_person_lookup_with_substantive_rationale",
        "reason": (
            "The original tests only a person's name; the edit preserves the "
            "source's reusable explanation about communication and performance."
        ),
    },
    {
        "fact_id": "fact-0957d4b41e197d0c95c3",
        "source_path": raw("esinem_da195f44f6ae32c4.json"),
        "evidence_marker": "ubiquitous Akechi-style 2-TK",
        "decision": "drop",
        "reason_code": "volatile_or_promotional",
        "reason": (
            "The unsupported claim that nearly everyone first learned this style "
            "appears in a sales page contrasting it with a promoted tutorial."
        ),
    },
    {
        "fact_id": "fact-eb4cd9947c4f82a880bf",
        "source_path": raw("esinem_2b762f05bc1bf364.json"),
        "evidence_marker": "training in Aikido from a very young age",
        "decision": "keep",
        "reason_code": "technique_development_lineage_context_complete",
        "reason": (
            "The martial art is connected directly to techniques incorporated "
            "into Yagami Ren's named kinbaku and hojojutsu styles."
        ),
    },
    {
        "fact_id": "fact-305e711b9258cb4cd9a2",
        "source_path": raw("kinbakutoday_9a72524da47a3539.json"),
        "evidence_marker":
            "our Semenawa favours the experimentation of suffering, not pain",
        "decision": "keep",
        "reason_code": "style_term_context_complete",
        "reason": (
            "Semenawa is the exact named style and the question includes the "
            "source's defining suffering-versus-pain distinction."
        ),
    },
    {
        "fact_id": "fact-7d40fbdc3504caa3efcf",
        "source_path": raw("esinem_dce6c59fa90ecae0.json"),
        "evidence_marker": "‘rigger’ as a word for the active party",
        "decision": "keep",
        "reason_code": "role_term_context_complete",
        "reason": (
            "The answer is the exact role term and the question preserves the "
            "source's warning that it can be confused with theatrical rigging."
        ),
    },
    {
        "fact_id": "fact-bc11688b1543ec8b2d9f",
        "source_path": raw("kinbakutoday_f57559bbb4c8b826.json"),
        "evidence_marker":
            "deep meaning of kinbaku, to find kinbaku-bi",
        "evidence_lines": 2,
        "decision": "keep",
        "reason_code": "aesthetic_concept_context_complete",
        "reason": (
            "Kinbaku-bi is a source-defined aesthetic concept tied to context, "
            "demonstration, connection, and the meaning of kinbaku."
        ),
    },
    {
        "fact_id": "fact-ebafa7a4f0ee861970d5",
        "source_path": raw("esinem_bebe8839b92231b9.json"),
        "evidence_marker":
            "Rope bondage, especially the Japanese form, Shibari, can be considered an art form",
        "decision": "drop",
        "reason_code": "semantic_duplicate",
        "reason": (
            "This generic Shibari-as-art lookup duplicates stronger retained "
            "definitions and supplies no technique, safety, or distinct concept."
        ),
    },
    {
        "fact_id": "fact-1a17204bc89767b3d93e",
        "source_path": raw("kinbakutoday_d4dcb268cb41c5e4.json"),
        "evidence_marker": "I leave seme untranslated because no single English term",
        "decision": "keep",
        "reason_code": "nuanced_historical_term_context_complete",
        "reason": (
            "The question carries the full semantic range that explains why the "
            "source deliberately leaves seme untranslated."
        ),
    },
    {
        "fact_id": "fact-ace591d75c7226db653a",
        "source_path": raw("kinbakutoday_4f9dec06e4af751a.json"),
        "evidence_marker": "onnen (怨念) does not simply name anger or resentment",
        "decision": "keep",
        "reason_code": "nuanced_historical_term_context_complete",
        "reason": (
            "The source explicitly distinguishes onnen from ordinary resentment, "
            "and the question states its durable historical meaning."
        ),
    },
    {
        "fact_id": "fact-978096701201424beddd",
        "source_path": raw("kinbakutoday_a358fd398f91040a.json"),
        "evidence_marker": "Kinbaku means “tight binding”",
        "decision": "keep",
        "reason_code": "core_term_context_complete",
        "reason": (
            "This is a direct, self-contained translation of a core rope term; "
            "the two-word answer needs no padding."
        ),
    },
    {
        "fact_id": "fact-c997a81f923d51ec6ddc",
        "source_path": raw("rope365_6f46d5169ca32ec7.json"),
        "evidence_marker": "Itō Seiu (1882 – 1961)",
        "evidence_occurrence": 2,
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": (
            "An isolated lifespan lookup adds less reusable rope knowledge than "
            "retained facts about Itō Seiu's work and artistic influence."
        ),
    },
    {
        "fact_id": "fact-cffa44c03400fc16f70f",
        "source_path": raw("esinem_f2dfde25be14a7a8.json"),
        "evidence_marker":
            "profoundly influenced by the work of ukiyoe artist Tsukioka Yoshitoshi",
        "decision": "keep",
        "reason_code": "artistic_lineage_context_complete",
        "reason": (
            "The influence connects a named ukiyo-e artist directly to Itō "
            "Seiu's development and the visual history of kinbaku."
        ),
    },
    {
        "fact_id": "fact-12a26a2774ec26f80047",
        "source_path": raw("kinbakutoday_2265a4f9ae40d83a.json"),
        "evidence_marker": "would later start Uramado in 1955",
        "decision": "keep",
        "reason_code": "rope_media_history_context_complete",
        "reason": (
            "The year anchors the start of a historically significant Japanese "
            "SM and bondage publication."
        ),
    },
    {
        "fact_id": "fact-3a4869e89e813235976a",
        "source_path": raw("wikipedia_2151448295a2af9b.json"),
        "evidence_marker": "femdom (short for female dominance)",
        "decision": "keep",
        "reason_code": "bdsm_term_context_complete",
        "reason": (
            "Femdom is a standard concise BDSM term whose expansion is fully "
            "specified by the question."
        ),
    },
    {
        "fact_id": "fact-501aa559677afe168fad",
        "source_path": raw("esinem_5c862cbce9ff02bd.json"),
        "evidence_marker":
            "spotted tenugui, which are the cotton Japanese washcloths",
        "decision": "keep",
        "reason_code": "equipment_material_context_complete",
        "reason": (
            "Cotton is the exact material in the source's explanation of "
            "tenugui; the commercial stock context does not make the material "
            "answer time-sensitive."
        ),
    },
    {
        "fact_id": "fact-5ab3eb30b97381b1fbe4",
        "source_path": RESOURCE_MANIFEST,
        "resource_id": "house_of_bound_tutorials",
        "decision": "keep",
        "reason_code": "owner_requested_resource_directory",
        "reason": (
            "The canonical URL is an explicitly requested rope-tutorial resource "
            "and answers where the videos can be purchased without copying paid "
            "content."
        ),
    },
    {
        "fact_id": "fact-3c85f97bbd348e7d84e0",
        "source_path": raw("kinbakutoday_4f9dec06e4af751a.json"),
        "evidence_marker":
            "Fujimi Iku was, in fact, one of the many additional names that Nureki Chimuo used",
        "decision": "keep",
        "reason_code": "rope_media_authorship_context_complete",
        "reason": (
            "The pseudonym helps identify Nureki's published work and supports the "
            "source's substantive history of masks in kinbaku media."
        ),
    },
    {
        "fact_id": "fact-621763f7f46d14be96dc",
        "source_path": raw("rope365_7fe5be31cb8dc67e.json"),
        "evidence_marker": "draws it’s strength from a bowline structure",
        "decision": "keep",
        "reason_code": "knot_structure_context_complete",
        "reason": (
            "The question is a precise source-attributed structural claim within "
            "a discussion comparing rope joins."
        ),
    },
    {
        "fact_id": "fact-810a5a529b69a8a826ca",
        "source_path": raw("esinem_dba752124ca21181.json"),
        "evidence_marker":
            "studied under the grand master of newaza (floor-work), Yukimura Haruki",
        "decision": "drop",
        "reason_code": "volatile_or_promotional_person_trivia",
        "reason": (
            "The teacher-name lookup comes from a dated tour promotion and does "
            "not preserve a technique, safety principle, or useful resource."
        ),
    },
    {
        "fact_id": "fact-c6cb7af841aa7500d80a",
        "source_path": raw("rope365_d7cb8892cca8b93a.json"),
        "evidence_marker": "I still consider he influenced my style a lot",
        "decision": "drop",
        "reason_code": "contextless_or_personal_anecdote",
        "reason": (
            "The vague 'author's style' question reduces a personal learning "
            "anecdote to a name lookup and offers no transferable guidance."
        ),
    },
    {
        "fact_id": "fact-7047486e0f5774a65324",
        "source_path": raw("esinem_888997ffcb0c181d.json"),
        "evidence_marker": "routinely used UV ropes, so we had ‘cyber shows’",
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": (
            "The rope color/effect in one venue's past shows is isolated "
            "performance trivia with no durable practice or safety value."
        ),
    },
    {
        "fact_id": "fact-8d6fa405856a471d80eb",
        "source_path": raw("rope365_682937f92222bf87.json"),
        "evidence_marker": "Starting with Osada Eikichi in the 1960s onward",
        "decision": "keep",
        "reason_code": "rope_performance_history_context_complete",
        "reason": (
            "The source places Osada Eikichi at the start of rope performance as "
            "an evolving art form, a meaningful historical attribution."
        ),
    },
    {
        "fact_id": "fact-7c645dabc1427a6e1e14",
        "source_path": raw("kinbakutoday_edba1220873364c8.json"),
        "evidence_marker": "Initially a battlefield technique",
        "decision": "keep",
        "reason_code": "restraint_history_context_complete",
        "reason": (
            "The initial battlefield role explains hojojutsu's later shift into "
            "law-enforcement restraint and is durable rope history."
        ),
    },
)

EXPECTED_SELECTION = tuple(specification["fact_id"] for specification in SPECS)
ISOLATED_PROJECTION = {
    "active_after_canonical_quality_merit_drops": 762,
    "active_after_context_merit_v1": 757,
    "active_after_this_tranche": 751,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 6,
    "new_edits_applied": 1,
    "output_rows": 789,
    "output_sha256":
        "eb5aed32f4a072ea5ba0699cab4fd38fc35836a28c7c9cb322cdb7e75b7f0609",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 18,
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
    short_points = (
        max(0, 10 - len(question_tokens)) if len(question_tokens) <= 9 else 0
    )
    answer_length = len(answer_tokens)
    bare_points = (6 if answer_length <= 1 else 5 if answer_length == 2
                   else 2 if answer_length <= 4 else 0)
    likely_name = bool(re.search(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|\b[A-Z][a-z]+[’']s\b",
        row["question"],
    ))
    person_points = (6 if lowered and
                     lowered[0] in PERSON_INTERROGATIVES and likely_name else 0)
    return {
        "answer_tokens": answer_length,
        "bare_answer_points": bare_points,
        "named_person_trivia_points": person_points,
        "pronoun_count": pronoun_count,
        "question_tokens": len(question_tokens),
        "risk_score": short_points + 3 * pronoun_count + bare_points + person_points,
        "short_question_points": short_points,
    }


def ranked_unreviewed(active_rows: list[dict]) -> tuple[list[dict], int, int]:
    excluded_ids = reviewed_fact_ids()
    ranked = []
    provenance_excluded = 0
    for active_index, row in enumerate(active_rows, 1):
        has_provenance = any(field in row for field in ACTIVE_REVIEW_FIELDS)
        if row["fact_id"] in excluded_ids or has_provenance:
            provenance_excluded += has_provenance
            continue
        features = risk_features(row)
        ranked.append((
            -features["risk_score"], features["question_tokens"],
            features["answer_tokens"], row["fact_id"], active_index,
            row, features,
        ))
    ranked.sort(key=lambda item: item[:4])
    return ([
        {"active_index": item[4], "row": item[5], "features": item[6]}
        for item in ranked
    ], len(excluded_ids), provenance_excluded)


def source_evidence(specification: dict, active_row: dict) -> tuple[str, str]:
    path = specification["source_path"]
    if path == RESOURCE_MANIFEST:
        manifest = json.loads(path.read_text())
        resources = {item["id"]: item for item in manifest["resources"]}
        resource = resources[specification["resource_id"]]
        if (resource["canonical_url"] != active_row["url"] or
                file_sha256(path) != active_row["document_sha256"]):
            raise ValueError("resource manifest drift")
        expected = f'{resource["name"]}: {resource["canonical_url"]}'
        if active_row["answer"] != expected:
            raise ValueError("resource-directory answer drift")
        return expected, "manifest_composite"

    document = json.loads(path.read_text())
    if document["url"] != active_row["url"]:
        raise ValueError(f'{active_row["fact_id"]}: source URL drift')
    if text_sha256(document["text"]) != active_row["document_sha256"]:
        raise ValueError(f'{active_row["fact_id"]}: source text hash drift')
    lines = document["text"].splitlines()
    indices = [index for index, line in enumerate(lines)
               if specification["evidence_marker"] in line]
    expected_occurrence = specification.get("evidence_occurrence", 1)
    if len(indices) < expected_occurrence:
        raise ValueError(
            f'{active_row["fact_id"]}: evidence occurrence is absent')
    start = indices[expected_occurrence - 1]
    evidence = "\n".join(lines[
        start:start + specification.get("evidence_lines", 1)])
    answer = specification.get("answer", active_row["answer"])
    if normalize_text(answer) not in normalize_text(evidence):
        raise ValueError(f'{active_row["fact_id"]}: answer absent from evidence')
    return evidence, "normalized_extractive"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    active_rows = read_jsonl(ACTIVE_DATASET)
    ranked, excluded_id_count, provenance_excluded = ranked_unreviewed(active_rows)
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
        if row["fact_id"] != fact_id:
            raise ValueError(f"{fact_id}: selected row drift")
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
            "schema": "context-merit-audit-v2",
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
            audit.update({
                "edited_answer": specification["answer"],
                "edited_question": specification["question"],
            })
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
    decisions = collections.Counter(row["decision"] for row in audits)
    reasons = collections.Counter(row["reason_code"] for row in audits)
    report = {
        "schema": "context-merit-audit-report-v2",
        "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT,
        "status": "segregated_pending_not_promoted",
        "selection": {
            "active_rows": len(active_rows),
            "eligible_unreviewed_rows": len(ranked),
            "excluded_ledger_fact_ids": excluded_id_count,
            "excluded_active_review_provenance": provenance_excluded,
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
            },
        },
        "audit": {
            "path": portable(AUDIT), "sha256": file_sha256(AUDIT),
            "rows": len(audits),
            "by_decision": dict(sorted(decisions.items())),
            "by_reason": dict(sorted(reasons.items())),
        },
        "new_pending_curation": {
            "path": portable(CURATION), "sha256": file_sha256(CURATION),
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
                for path in (QUALITY_MERIT_CURATION, TASUKI_CURATION,
                              CONTEXT_MERIT_V1_CURATION)
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
                "data/manual_reviews/context_merit_audit_v2/"
                "test_context_merit_audit_v2.py",
                "test_build_curated_qa.py",
            ],
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
