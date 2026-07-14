#!/usr/bin/env python3
"""Audit context-merit tranche v4 without rewriting prior tranches."""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
V2_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v2"
sys.path[:0] = [str(ROOT), str(V2_DIR)]

import build_context_merit_audit_v2 as common
from qa_quality import normalize_text, qa_pair_from_record


DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v4.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v4.jsonl"
REPORT = OUT_DIR / "report_context_merit_v4.json"
REVIEWER = "codex-context-merit-audit-v4"
REVIEWED_AT = "2026-07-14"
RESOURCE_MANIFEST = ROOT / "sources" / "rope_resources_v1.json"

ACTIVE_DATASET = common.ACTIVE_DATASET
ACTIVE_REPORT = common.ACTIVE_REPORT
ACTIVE_CURATIONS = common.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS = common.PRIOR_PENDING_ADDITIONS
QUALITY_MERIT_CURATION = common.QUALITY_MERIT_CURATION
TASUKI_CURATION = common.TASUKI_CURATION
CONTEXT_CURATIONS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"pending_curation_context_merit_v{version}.jsonl"
    for version in (1, 2, 3)
)
PRIOR_CONTEXT_MERIT_DIRS = frozenset(
    path.parent.name for path in CONTEXT_CURATIONS
)

file_sha256 = common.file_sha256
text_sha256 = common.text_sha256
portable = common.portable
read_jsonl = common.read_jsonl
write_jsonl = common.write_jsonl
risk_features = common.risk_features


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-150321729907852ef08e",
        "source_path": raw("esinem_62d7dca7b38dbd4e.json"),
        "marker": "meaning literally ‘tight binding’",
        "decision": "drop",
        "reason_code": "semantic_duplicate",
        "reason": (
            "A retained QA already gives the same literal translation of kinbaku "
            "from a clearer dedicated definition."
        ),
    },
    {
        "fact_id": "fact-afb269c7fdc5968e1793",
        "source_path": raw("esinem_62d7dca7b38dbd4e.json"),
        "marker": "closed in January 2009",
        "decision": "keep",
        "reason_code": "rope_media_history_context_complete",
        "reason": (
            "The closure date is stable context for the history and decline of a "
            "major Japanese SM and kinbaku publication."
        ),
    },
    {
        "fact_id": "fact-d596019f1c21d094ee06",
        "source_path": raw("kinbakutoday_e7f3e175c6e3bfd7.json"),
        "marker": "aibunawa or “caressing rope.”",
        "decision": "keep",
        "reason_code": "style_translation_context_complete",
        "reason": (
            "The two-word answer is the source's direct English translation of a "
            "named Yukimura rope style."
        ),
    },
    {
        "fact_id": "fact-d70379ce60d247a310e7",
        "source_path": raw("rope365_8ae9e3d93b31601b.json"),
        "marker": "Cow hitch aka Lark’s head",
        "decision": "keep",
        "reason_code": "knot_alias_context_complete",
        "reason": (
            "Lark's head is the exact common alias for a cow hitch and is useful "
            "rope vocabulary."
        ),
    },
    {
        "fact_id": "fact-f4353d8b3fa71ac0c87f",
        "source_path": raw("kinbakutoday_c5e568667b495473.json"),
        "marker": "It literally means “sitting long.”",
        "decision": "drop",
        "reason_code": "contextless_or_low_value",
        "reason": (
            "The source itself notes the position is not especially useful for "
            "tying, and the isolated literal gloss adds little rope knowledge."
        ),
    },
    {
        "fact_id": "fact-f58100ebf25a99f10d86",
        "source_path": raw("rope365_095aa0f0eea4c62c.json"),
        "marker": "the cuff is only locked in one direction",
        "decision": "edit",
        "question":
            "What limitation does Rope365 note about a cow-hitch cuff’s lock?",
        "answer": (
            "the cuff is only locked in one direction; if you pull on the origin "
            "rope, the cuff may tighten"
        ),
        "reason_code": "restore_safety_limitation",
        "reason": (
            "The original bare phrase omits the practical consequence; the edit "
            "retains the source's full warning about tightening."
        ),
    },
    {
        "fact_id": "fact-0aa98abb926464bc987f",
        "source_path": raw("rope365_6f46d5169ca32ec7.json"),
        "marker": "caressing style, also known as aibunawa",
        "decision": "keep",
        "reason_code": "style_alias_context_complete",
        "reason": (
            "The term links Yukimura Haruki's caressing approach to its named "
            "style, adding context beyond the separate literal translation."
        ),
    },
    {
        "fact_id": "fact-1295a165c3a8afd2ff47",
        "source_path": raw("esinem_9e9a3ecc90913bf1.json"),
        "marker": "iconic diamond (hishi) pattern",
        "decision": "edit",
        "question": "What geometric shape does Esinem identify with hishi?",
        "answer": "diamond",
        "reason_code": "replace_tutorial_lookup_with_term_definition",
        "reason": (
            "The edit removes deictic tutorial wording and preserves the reusable "
            "meaning of the Japanese pattern term."
        ),
    },
    {
        "fact_id": "fact-1b54b36d46cd6dea2db5",
        "source_path": raw("rope365_3b12e667f648a22b.json"),
        "marker": "all four seasons do not touch on the subject of suspension",
        "decision": "drop",
        "reason_code": "volatile_or_promotional",
        "reason": (
            "Remembering an omission from a particular course edition is "
            "curriculum trivia that can change and teaches no suspension safety."
        ),
    },
    {
        "fact_id": "fact-1f3109791d464292aab8",
        "source_path": raw("rope365_d7cb8892cca8b93a.json"),
        "marker": "I first learned it from OsakaDan",
        "decision": "drop",
        "reason_code": "contextless_or_personal_anecdote",
        "reason": (
            "The teacher-name lookup is incidental to a personal learning story "
            "and duplicates retained substantive tasuki coverage."
        ),
    },
    {
        "fact_id": "fact-3f30298063c10c1a2702",
        "source_path": raw("rope365_f2d9e825760af158.json"),
        "marker": "commonly used in Japanese martial art Hojōjutsu",
        "decision": "keep",
        "reason_code": "restraint_history_context_complete",
        "reason": (
            "The answer connects inline hojo cuffs to the named martial restraint "
            "tradition where the source says they were commonly used."
        ),
    },
    {
        "fact_id": "fact-54b5e92e0ada0d0f492f",
        "source_path": raw("rope365_095aa0f0eea4c62c.json"),
        "marker": "cow hitch is technically two half hitches",
        "decision": "keep",
        "reason_code": "knot_structure_context_complete",
        "reason": (
            "The count is an exact structural property of the cow hitch and the "
            "question supplies the knot context."
        ),
    },
    {
        "fact_id": "fact-7fade041650a54ac24f3",
        "source_path": raw("kinbakutoday_d5d373e4a55ff204.json"),
        "marker": "native Japanese word is sarugutsuwa",
        "decision": "keep",
        "reason_code": "historical_equipment_term_context_complete",
        "reason": (
            "Sarugutsuwa is the exact traditional Japanese gag term and the "
            "question states its bondage context."
        ),
    },
    {
        "fact_id": "fact-b5b9bf877797db33c34c",
        "source_path": raw("wikipedia_914d249c3d7d542c.json"),
        "marker": "target of censors in 1930",
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": (
            "The isolated censorship year is biographical trivia and omits the "
            "more useful context of how censorship affected Itō's work."
        ),
    },
    {
        "fact_id": "fact-fee7de51d520ffac2394",
        "source_path": raw("rope365_15518f0912cce205.json"),
        "marker": "based on a design by Tifereth",
        "decision": "keep",
        "reason_code": "technique_design_credit_context_complete",
        "reason": (
            "The question names the Angel Tie pattern, making the creator credit "
            "specific and useful rather than a free-floating handle lookup."
        ),
    },
    {
        "fact_id": "fact-fd929790183178649bcb",
        "source_path": RESOURCE_MANIFEST,
        "resource_id": "atx_empty_space",
        "decision": "keep",
        "reason_code": "owner_requested_resource_directory",
        "reason": (
            "The stable public calendar URL is an explicitly requested Austin "
            "rope-community resource and avoids memorizing event dates."
        ),
    },
    {
        "fact_id": "fact-36ce0e031716dd3cebff",
        "source_path": raw("kinbakutoday_5783480db327a3ba.json"),
        "marker": "Attitude toward “torment” (責め)",
        "decision": "keep",
        "reason_code": "historical_source_term_context_complete",
        "reason": (
            "The Japanese term is explicitly paired with torment in a historical "
            "participant's discussion; the concise answer is context-complete."
        ),
    },
    {
        "fact_id": "fact-4cad0a397023824f5b94",
        "source_path": raw("esinem_0b1850ee9a40c337.json"),
        "marker": "classic Akechi derived version",
        "decision": "keep",
        "reason_code": "technique_lineage_context_complete",
        "reason": (
            "The source attributes the familiar classic gote form to Akechi in a "
            "technical discussion of upper-cinch variants."
        ),
    },
    {
        "fact_id": "fact-58d0899e951e3758bdbb",
        "source_path": raw("kinbakutoday_490a87ec78c3d64b.json"),
        "marker": "Tokugawa bakufu keiji zufu, which was published in 1878",
        "decision": "keep",
        "reason_code": "restraint_history_context_complete",
        "reason": (
            "The date anchors a named historical source depicting rope, public "
            "shame, and punishment in the Edo-period context."
        ),
    },
    {
        "fact_id": "fact-5caeb388d651dc813b96",
        "source_path": raw("kinbakutoday_7fb6f1e0e0d186b2.json"),
        "marker": "trust or shinrai (信頼)",
        "decision": "drop",
        "reason_code": "contextless_or_low_value",
        "reason": (
            "The single-word language lookup comes from a business-travel "
            "anecdote and does not preserve useful rope trust or consent guidance."
        ),
    },
    {
        "fact_id": "fact-7e567cc256ae27660f3f",
        "source_path": raw("wikipedia_a97e19abfe49afa9.json"),
        "marker": "was typically hemp in material",
        "decision": "keep",
        "reason_code": "historical_rope_material_context_complete",
        "reason": (
            "Hemp is the exact typical material stated for the historical honnawa "
            "main rope and is durable equipment history."
        ),
    },
    {
        "fact_id": "fact-d0e463d5eba63cdd67f4",
        "source_path": raw("kinbakutoday_1c6818645cd43d8f.json"),
        "marker": "SM magazines of the 1970s",
        "decision": "drop",
        "reason_code": "semantic_duplicate",
        "reason": (
            "The decade lookup adds nothing beyond the retained 1976 Phantom "
            "Mirror resource fact from the same sentence."
        ),
    },
    {
        "fact_id": "fact-e6699b754430cc9327e6",
        "source_path": raw("esinem_f2dfde25be14a7a8.json"),
        "marker": "novella by Edogawa Ranpo with some whipping is from 1928",
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": (
            "The publication year of a tangential novella is general literature "
            "trivia rather than reusable kinbaku history."
        ),
    },
    {
        "fact_id": "fact-fd5c96b72210af42fd05",
        "source_path": raw("rope365_b781bc1188743976.json"),
        "marker": "use hitches to connect the ropes together",
        "decision": "keep",
        "reason_code": "rope_crafting_context_complete",
        "reason": (
            "Hitches are the exact technique suggested for connecting lines in a "
            "space-filling rope-crafting exercise."
        ),
    },
    {
        "fact_id": "fact-3733f5fd6b66feb1b5ed",
        "source_path": raw("kinbakutoday_209cfdfa24ad7561.json"),
        "marker": "work of Nureki Chimuo and Sugiura Norio",
        "decision": "keep",
        "reason_code": "rope_art_lineage_context_complete",
        "reason": (
            "The pair directly connects Nureki's ties and Sugiura's photography "
            "to the development of Muku's bondage drawings."
        ),
    },
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v3": 748,
    "active_after_this_tranche": 740,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 8,
    "new_edits_applied": 2,
    "output_rows": 778,
    "output_sha256":
        "78cbf2f96347aa49042c568472a356ffd9ddcdd93f1a9048f57c1e4e6640e635",
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
            "schema": "context-merit-audit-v4", "source": row["source"],
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
        "schema": "context-merit-audit-report-v4", "reviewer": REVIEWER,
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
