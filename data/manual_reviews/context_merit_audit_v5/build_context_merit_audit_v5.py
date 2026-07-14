#!/usr/bin/env python3
"""Audit context-merit tranche v5 without rewriting prior tranches."""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
V4_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v4"
sys.path[:0] = [str(ROOT), str(V4_DIR)]

import build_context_merit_audit_v4 as previous
from qa_quality import normalize_text, qa_pair_from_record


common = previous.common
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v5.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v5.jsonl"
REPORT = OUT_DIR / "report_context_merit_v5.json"
REVIEWER = "codex-context-merit-audit-v5"
REVIEWED_AT = "2026-07-14"
RESOURCE_MANIFEST = ROOT / "sources" / "rope_resources_v1.json"

ACTIVE_DATASET = previous.ACTIVE_DATASET
ACTIVE_REPORT = previous.ACTIVE_REPORT
ACTIVE_CURATIONS = previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS = previous.PRIOR_PENDING_ADDITIONS
QUALITY_MERIT_CURATION = previous.QUALITY_MERIT_CURATION
TASUKI_CURATION = previous.TASUKI_CURATION
CONTEXT_CURATIONS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"pending_curation_context_merit_v{version}.jsonl"
    for version in (1, 2, 3, 4)
)
PRIOR_CONTEXT_MERIT_DIRS = frozenset(
    path.parent.name for path in CONTEXT_CURATIONS
)

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
portable = previous.portable
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
risk_features = previous.risk_features


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-9b9ba4f465c2a44a290f",
        "source_path": RESOURCE_MANIFEST,
        "resource_id": "tethered_together",
        "decision": "keep",
        "reason_code": "owner_requested_resource_directory",
        "reason": (
            "The stable convention homepage is an explicitly requested US rope "
            "resource and does not memorize volatile event dates or prices."
        ),
    },
    {
        "fact_id": "fact-dff1d2131b0db9449d06",
        "source_path": RESOURCE_MANIFEST,
        "resource_id": "de_giotto_rope",
        "decision": "keep",
        "reason_code": "owner_requested_resource_directory",
        "reason": (
            "The owner explicitly requested that this natural-fiber rope vendor "
            "remain available as a resource recommendation."
        ),
    },
    {
        "fact_id": "fact-d219f277448544afa318",
        "source_path": RESOURCE_MANIFEST,
        "resource_id": "my_nawashi",
        "decision": "keep",
        "reason_code": "owner_requested_resource_directory",
        "reason": (
            "The owner explicitly requested this natural-fiber bondage-rope shop; "
            "the QA stores only its durable directory entry."
        ),
    },
    {
        "fact_id": "fact-149d24b1d4497bda46dc",
        "source_path": raw("kinbakutoday_98c19e5a4c2038e0.json"),
        "marker": "instruction, performance, and cultural artifact",
        "decision": "edit",
        "question": (
            "What three roles does Kinbaku Today attribute to Yamaguchi "
            "Tokiko’s Self Bondage Classroom?"
        ),
        "answer": "instruction, performance, and cultural artifact",
        "reason_code": "replace_publication_date_with_historical_significance",
        "reason": (
            "The edit replaces an isolated publication year with the source's "
            "concise explanation of why the work matters to rope history."
        ),
    },
    {
        "fact_id": "fact-38c6234e2a72dd14287f",
        "source_path": raw("esinem_b249eddc1f5e1864.json"),
        "marker": "High-stranding is the term used",
        "decision": "keep",
        "reason_code": "rope_care_term_context_complete",
        "reason": (
            "High-stranding names a practical three-strand-rope imbalance and the "
            "question supplies enough context to make the term reusable."
        ),
    },
    {
        "fact_id": "fact-5e7ec77ea97c66a99295",
        "source_path": raw("kinbakutoday_3ee6ab45e0f9b1ae.json"),
        "marker": "Technique was always second to heart",
        "decision": "keep",
        "reason_code": "rope_education_principle_context_complete",
        "reason": (
            "The answer preserves the article's central warning that technical "
            "skill must not eclipse motivation, relationship, and care."
        ),
    },
    {
        "fact_id": "fact-727f61b65c50361dea3a",
        "source_path": raw("kinbakutoday_f370696af0359092.json"),
        "marker": "identify himself as “Seiu (晴雨)” on 1918",
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": (
            "The year a historical artist adopted a name is isolated biography "
            "trivia and teaches no rope history, method, or safety context."
        ),
    },
    {
        "fact_id": "fact-7383fd25dcc0bc511528",
        "source_path": raw("kinbakutoday_92c9fc29a66300a0.json"),
        "marker": "1977 picture “Fairy in a Cage,”",
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": (
            "An isolated film release year is media trivia; the QA omits the "
            "film's more relevant political, sexual, and kinbaku context."
        ),
    },
    {
        "fact_id": "fact-93d4e57a60d967450ac2",
        "source_path": raw("kinbakutoday_fd0d7ba4b6589765.json"),
        "marker": "video industry in 1988",
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": (
            "The year Yukimura entered video work is a personal-career lookup and "
            "does not capture his style, teaching, or contribution to rope."
        ),
    },
    {
        "fact_id": "fact-dd9527e1735235d3d2d9",
        "source_path": raw("kinbakutoday_4f9dec06e4af751a.json"),
        "marker": (
            "photographs came from elsewhere before the story was written "
            "around them"
        ),
        "decision": "edit",
        "question": (
            "What does Kinbaku Today cautiously infer from the photographs’ "
            "1950s-era visual vocabulary?"
        ),
        "answer": (
            "the photographs came from elsewhere before the story was written "
            "around them"
        ),
        "reason_code": "replace_bare_era_with_source_criticism",
        "reason": (
            "The edit turns an awkward era-label lookup into the article's "
            "explicitly qualified provenance inference."
        ),
    },
    {
        "fact_id": "fact-e5950fbfe9d881d40bb3",
        "source_path": RESOURCE_MANIFEST,
        "resource_id": "shibari_study",
        "decision": "keep",
        "reason_code": "owner_requested_resource_directory",
        "reason": (
            "The owner supplied Shibari Study as an intermediate suspension "
            "resource; the QA records its durable catalog URL, not paid content."
        ),
    },
    {
        "fact_id": "fact-396293a77723206abb84",
        "source_path": RESOURCE_MANIFEST,
        "resource_id": "knothead_nylon",
        "decision": "keep",
        "reason_code": "owner_requested_resource_directory",
        "reason": (
            "The owner explicitly requested this synthetic bondage-rope vendor, "
            "and the answer avoids volatile product and price claims."
        ),
    },
    {
        "fact_id": "fact-5c02852cca5c4979d32e",
        "source_path": RESOURCE_MANIFEST,
        "resource_id": "strugglers_knot_somerville_bowline",
        "decision": "keep",
        "reason_code": "owner_requested_resource_directory",
        "reason": (
            "This is an explicitly requested lesson URL for two named knots; the "
            "QA stores only verified video metadata and location."
        ),
    },
    {
        "fact_id": "fact-12234c1f51530e1e0526",
        "source_path": raw("esinem_5d1d44089d8bedf1.json"),
        "marker": "signature L-friction",
        "decision": "keep",
        "reason_code": "technique_term_context_complete",
        "reason": (
            "The named friction is a concrete structural feature of Kazami's ties "
            "and the question identifies the relevant lineage and technique."
        ),
    },
    {
        "fact_id": "fact-1e25e0594b6e1382ab22",
        "source_path": raw("rope365_62f7e527bb35b47d.json"),
        "marker": "ropes (10-15ft)",
        "decision": "keep",
        "reason_code": "source_attributed_equipment_guidance",
        "reason": (
            "The answer is the source's clearly scoped starting-kit range for "
            "short ropes rather than a universal safety limit."
        ),
    },
    {
        "fact_id": "fact-24adb04721d158df31dc",
        "source_path": raw("kinbakutoday_5a6e15f57e52a345.json"),
        "marker": "Seme (torture) was the equivalent term for sadomasochism",
        "decision": "keep",
        "reason_code": "historical_terminology_context_complete",
        "reason": (
            "The QA captures Ito Seiu's historically situated usage around 1900, "
            "not a claim that the terms are equivalent today."
        ),
    },
    {
        "fact_id": "fact-366cdcf2556613c8f486",
        "source_path": raw("wikipedia_2151448295a2af9b.json"),
        "marker": "also called subspace, for the submissive",
        "decision": "keep",
        "reason_code": "bdsm_state_term_context_complete",
        "reason": (
            "Subspace is a common BDSM state term and the question specifies the "
            "submissive context instead of presenting a free-floating label."
        ),
    },
    {
        "fact_id": "fact-498f0c0b7d716b63aca4",
        "source_path": raw("kinbakutoday_dc3acb0dba2a7693.json"),
        "marker": "Semenawa (rope punishment/torture)",
        "decision": "keep",
        "reason_code": "style_term_context_complete",
        "reason": (
            "Semenawa is a reusable rope-style term and its punishment/torture "
            "context is explicitly supplied in both the question and source."
        ),
    },
    {
        "fact_id": "fact-4d843edd13c9109507f6",
        "source_path": raw("kinbakutoday_df9c70212a927199.json"),
        "marker": "hazukashii (恥ずかしい) can be translated",
        "decision": "keep",
        "reason_code": "style_emotional_term_context_complete",
        "reason": (
            "Hazukashii is a central emotional concept in Yukimura-ryū, and the "
            "question accurately anchors the source's three concise glosses."
        ),
    },
    {
        "fact_id": "fact-54c61697cebf904795ea",
        "source_path": raw("rope365_c73bc6fb66977a2d.json"),
        "marker": "Consent can be withdrawn at any point during play",
        "decision": "edit",
        "question": "According to Rope365, when can consent be withdrawn?",
        "answer": "at any point during play",
        "reason_code": "replace_slogan_with_actionable_consent_guidance",
        "reason": (
            "The edit replaces an oversimplified consent-versus-abuse slogan with "
            "clear, actionable guidance that consent remains revocable."
        ),
    },
    {
        "fact_id": "fact-63f198061bef3120d9fc",
        "source_path": raw("rope365_a19e1d759fa86b73.json"),
        "marker": "rope martial art hojojutsu",
        "decision": "drop",
        "reason_code": "semantic_duplicate",
        "reason": (
            "A retained QA already explains the direct relationship between "
            "Hojōjutsu and inline hojo cuffs with more useful context."
        ),
    },
    {
        "fact_id": "fact-858c6134c00101da642e",
        "source_path": raw("kinbakutoday_91bd5cf512bd9ecf.json"),
        "marker": "Neck Play (kubi-nawa)",
        "decision": "drop",
        "reason_code": "unsafe_or_medically_unsupported",
        "reason": (
            "A bare neck-play label from a video contents list offers no risk, "
            "anatomy, consent, or emergency context for a high-consequence topic."
        ),
    },
    {
        "fact_id": "fact-8d167c9038da67560089",
        "source_path": raw("wikipedia_2151448295a2af9b.json"),
        "marker": "compound term sado-masochism",
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": (
            "The year a psychoanalyst coined a broad BDSM term is tangential "
            "etymology trivia and contributes little to rope-specific assistance."
        ),
    },
    {
        "fact_id": "fact-9384fac510ec4a4a68c4",
        "source_path": raw("kinbakutoday_e05ae8c0bba371f0.json"),
        "marker": "Kamasutra is called Shijuhatte",
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": (
            "The name of a traditional non-rope sex-position list is tangential "
            "to the article's rope book and does not answer a rope-practice need."
        ),
    },
    {
        "fact_id": "fact-db4e6b687bd1fba0dde5",
        "source_path": raw("rope365_25f1b23eb40be00e.json"),
        "marker": "family of knots used to extend rope is called bends",
        "decision": "keep",
        "reason_code": "knot_family_term_context_complete",
        "reason": (
            "Bends is the standard family name for rope-extension knots and the "
            "question gives the exact functional context."
        ),
    },
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v4": 740,
    "active_after_this_tranche": 733,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 7,
    "new_edits_applied": 3,
    "output_rows": 771,
    "output_sha256":
        "8d19730069ea51a3547d699b5dc42bdb8b05c3e3e3706faed981e1cae99fdeca",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 15,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


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
            for field in common.ID_FIELDS:
                value = row.get(field)
                if isinstance(value, str) and value.startswith("fact-"):
                    reviewed.add(value)
            reviewed.update(
                value for value in row.get("candidate_fact_ids", [])
                if isinstance(value, str) and value.startswith("fact-"))
    return reviewed


def ranked_unreviewed(rows: list[dict]) -> tuple[list[dict], int, int]:
    reviewed = reviewed_fact_ids()
    ranked = []
    provenance_count = 0
    for index, row in enumerate(rows, 1):
        provenance = any(field in row for field in common.ACTIVE_REVIEW_FIELDS)
        if row["fact_id"] in reviewed or provenance:
            provenance_count += provenance
            continue
        features = risk_features(row)
        ranked.append((
            -features["risk_score"], features["question_tokens"],
            features["answer_tokens"], row["fact_id"], index, row, features))
    ranked.sort(key=lambda item: item[:4])
    return ([{"active_index": item[4], "row": item[5],
              "features": item[6]} for item in ranked],
            len(reviewed), provenance_count)


def source_evidence(spec: dict, active: dict) -> tuple[str, str]:
    path = spec["source_path"]
    if path == RESOURCE_MANIFEST:
        resources = {item["id"]: item for item in
                     json.loads(path.read_text())["resources"]}
        resource = resources[spec["resource_id"]]
        url = resource.get("recommendation_url", resource["canonical_url"])
        answer = f'{resource["name"]}: {url}'
        if (file_sha256(path) != active["document_sha256"] or
                active["url"] != url or active["answer"] != answer):
            raise ValueError("resource manifest drift")
        return answer, "manifest_composite"
    document = json.loads(path.read_text())
    if (document["url"] != active["url"] or
            text_sha256(document["text"]) != active["document_sha256"]):
        raise ValueError(f'{active["fact_id"]}: source drift')
    matches = [line for line in document["text"].splitlines()
               if spec["marker"] in line]
    if len(matches) != 1:
        raise ValueError(f'{active["fact_id"]}: evidence marker drift')
    evidence = matches[0]
    answer = spec.get("answer", active["answer"])
    if normalize_text(answer) not in normalize_text(evidence):
        raise ValueError(f'{active["fact_id"]}: unsupported answer')
    return evidence, "normalized_extractive"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    active_rows = read_jsonl(ACTIVE_DATASET)
    ranked, excluded, provenance = ranked_unreviewed(active_rows)
    selected = ranked[:25]
    selected_ids = tuple(item["row"]["fact_id"] for item in selected)
    if selected_ids != EXPECTED_SELECTION:
        raise ValueError(f"selection drift: {selected_ids!r}")
    audits, curations = [], []
    for audit_index, (spec, selected_item) in enumerate(zip(SPECS, selected), 1):
        row = selected_item["row"]
        question, answer = qa_pair_from_record(row)
        evidence, support_type = source_evidence(spec, row)
        audit = {
            "active_answer": answer, "active_index": selected_item["active_index"],
            "active_question": question, "audit_index": audit_index,
            "decision": spec["decision"],
            "document_sha256": row["document_sha256"],
            "fact_id": row["fact_id"], "reason": spec["reason"],
            "reason_code": spec["reason_code"], "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER, "risk_features": selected_item["features"],
            "schema": "context-merit-audit-v5", "source": row["source"],
            "source_document": portable(spec["source_path"]),
            "source_document_file_sha256": file_sha256(spec["source_path"]),
            "source_support": support_type, "support_evidence": evidence,
            "support_evidence_sha256": text_sha256(evidence), "url": row["url"],
        }
        if spec["decision"] == "edit":
            audit.update(edited_question=spec["question"],
                         edited_answer=spec["answer"])
        audits.append(audit)
        if spec["decision"] in {"drop", "edit"}:
            curation = {
                "action": spec["decision"],
                "document_sha256": row["document_sha256"],
                "evidence_url": row["url"], "expected_answer": answer,
                "expected_question": question, "fact_id": row["fact_id"],
                "reason": spec["reason"], "reason_code": spec["reason_code"],
                "reviewed_at": REVIEWED_AT, "reviewer": REVIEWER,
                "source_lineage": row["source_lineage"],
            }
            if spec["decision"] == "edit":
                curation.update(answer=spec["answer"], evidence=evidence,
                                question=spec["question"],
                                support_type="extractive")
            curations.append(curation)
    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    decisions = collections.Counter(row["decision"] for row in audits)
    report = {
        "schema": "context-merit-audit-report-v5", "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT, "status": "segregated_pending_not_promoted",
        "selection": {
            "active_rows": len(active_rows), "eligible_unreviewed_rows": len(ranked),
            "excluded_ledger_fact_ids": excluded,
            "excluded_active_review_provenance": provenance,
            "rows_selected": 25, "fact_ids_in_rank_order": list(selected_ids),
            "ranking": {
                "score": "short_question_points + 3*pronoun_count + bare_answer_points + named_person_trivia_points",
                "tie_break": "risk_score descending, question tokens ascending, answer tokens ascending, fact_id ascending",
            },
        },
        "audit": {"path": portable(AUDIT), "sha256": file_sha256(AUDIT),
                  "rows": 25, "by_decision": dict(sorted(decisions.items())),
                  "by_reason": dict(sorted(collections.Counter(
                      row["reason_code"] for row in audits).items()))},
        "new_pending_curation": {
            "path": portable(CURATION), "sha256": file_sha256(CURATION),
            "decisions": len(curations), "by_action": dict(sorted(
                collections.Counter(row["action"] for row in curations).items()))},
        "prior_pending": {
            "additions": [{"path": portable(path), "rows": len(read_jsonl(path)),
                           "sha256": file_sha256(path)}
                          for path in PRIOR_PENDING_ADDITIONS],
            "curations": [{"path": portable(path), "rows": len(read_jsonl(path)),
                           "sha256": file_sha256(path)} for path in
                          (QUALITY_MERIT_CURATION, TASUKI_CURATION,
                           *CONTEXT_CURATIONS)]},
        "active_baseline": {
            "dataset": {"path": portable(ACTIVE_DATASET), "rows": len(active_rows),
                        "sha256": file_sha256(ACTIVE_DATASET)},
            "report": {"path": portable(ACTIVE_REPORT),
                       "sha256": file_sha256(ACTIVE_REPORT)},
            "curation": [{"path": portable(path), "sha256": file_sha256(path)}
                         for path in ACTIVE_CURATIONS]},
        "sealed_evaluation_policy": {
            "manual_review_opened_eval_or_heldout_content": False,
            "generator_opens_eval_or_heldout_content": False,
            "collision_check": "sealed evaluation paths are handled only by build_curated_qa.py during isolated projection"},
        "isolated_build_projection": ISOLATED_PROJECTION,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
