#!/usr/bin/env python3
"""Audit context-merit tranche v12 while keeping v1-v11 byte-frozen."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V11_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v11"
sys.path[:0] = [str(ROOT), str(V11_DIR)]
import build_context_merit_audit_v11 as previous

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v12.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v12.jsonl"
REPORT = OUT_DIR / "report_context_merit_v12.json"
REVIEWER = "codex-context-merit-audit-v12"
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
    for version in range(1, 12)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-e999c13513d61c7a27f6", "source_path": raw("esinem_bebe8839b92231b9.json"),
     "marker": "termed as ‘rope stoned’ or sub-spaced", "decision": "edit",
     "question": "What colloquial terms does Esinem use for a dreamy, relaxed state after a bondage session?",
     "answer": "rope stoned or sub-spaced", "reason_code": "replace_single_term_with_complete_source_gloss",
     "reason": "The edit preserves both source-attributed colloquial terms without presenting the subjective state as universal."},
    {"fact_id": "fact-1a8bcfd32de096abb8cd", "source_path": raw("rope365_3d015bb41296c9fa.json"),
     "marker": "Most of these ties have high risks, often starting with a loop around the neck and cinching on nerves", "decision": "edit",
     "question": "What risk warning does Rope365 give about the historical box ties on this page?",
     "answer": "most of these ties have high risks, often starting with a loop around the neck and cinching on nerves",
     "reason_code": "replace_style_lookup_with_explicit_risk_warning",
     "reason": "The edit replaces a martial-style quiz with the page's explicit warning about neck loops and nerve cinches."},
    {"fact_id": "fact-473bd38e12136efbc71a", "source_path": raw("rope365_bdb2a15da368be35.json"),
     "marker": "bring the heel as close as possible", "decision": "drop",
     "reason_code": "unsafe_or_medically_unsupported",
     "reason": "A restriction-tightening frog-tie micro-step is detached from the page's knee, circulation, nerve, monitoring, and release warnings."},
    {"fact_id": "fact-9c51769429e04c647e6d", "source_path": raw("esinem_76f7a14d2b86ea11.json"),
     "marker": "Inversions carry a high risk of serious injury if you lose control or drop your partner", "decision": "edit",
     "question": "Why does Esinem describe inversions as carrying a high risk of serious injury?",
     "answer": "if you lose control or drop your partner since their head is likely to be the first thing to hit the ground",
     "reason_code": "replace_body_part_lookup_with_inversion_risk",
     "reason": "The edit retains the complete causal warning rather than reducing inversion risk to a body-part answer."},
    {"fact_id": "fact-9d6631a6817a7cc59cf1", "source_path": raw("esinem_bebe8839b92231b9.json"),
     "marker": "Jim Stewart, author and founder of Fetters", "decision": "drop",
     "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "The author-and-founder lookup is personal trivia and omits the consent and risk context needed for struggle or escape play."},
    {"fact_id": "fact-c7fc884e5d3c535f459c", "source_path": raw("kinbakutoday_5783480db327a3ba.json"),
     "marker": "rare moments in early postwar bondage culture where a model’s own voice appears in print", "decision": "drop",
     "reason_code": "semantic_duplicate",
     "reason": "A retained edited QA already uses this exact evidence to explain Kawabata Tanako's page's historical significance."},
    {"fact_id": "fact-dc64136a01ef117c4fca", "source_path": raw("rope365_6cc5a0fc438e230f.json"),
     "marker": "use a crossing hitch for the middle of the X", "decision": "drop",
     "reason_code": "unsafe_or_medically_unsupported",
     "reason": "A body-harness construction micro-step lacks the pressure, anatomy, consent, monitoring, and release context needed for safe assistance."},
    {"fact_id": "fact-a7e675bdc6f1952c8406", "source_path": raw("rope365_02b5a435a4822809.json"),
     "marker": "Communication is key and consent is particularly important with this kind of play", "decision": "edit",
     "question": "What does Rope365 emphasize for pressure-point play?",
     "answer": "communication is key and consent is particularly important",
     "reason_code": "replace_pain_compliance_term_with_consent_warning",
     "reason": "The edit retains the source's consent warning instead of making pain compliance a context-free technique label."},
    {"fact_id": "fact-e2fe113f6dcb3c25fa4f", "source_path": raw("kinbakutoday_86c13124375e690a.json"),
     "marker": "Perhaps the most famous of these authors was Oniroku Dan", "decision": "keep",
     "reason_code": "rope_literary_history_context_complete",
     "reason": "The author is explicitly situated in the early fiction that shaped Japanese rope, SM, and bondage culture."},
    {"fact_id": "fact-327cafa3cd46654ccea8", "source_path": raw("rope365_8e6f4abea3bf4f6d.json"),
     "marker": "Pushing rope in a gap is generally inefficient and can damage it", "decision": "edit",
     "question": "Why does Rope365 recommend pulling rather than pushing rope into a gap?",
     "answer": "pushing rope in a gap is generally inefficient and can damage it",
     "reason_code": "replace_technique_lookup_with_handling_rationale",
     "reason": "The edit preserves the rope-handling rationale instead of asking only for a coined technique name."},
    {"fact_id": "fact-79ea40e1492359a99725", "source_path": raw("rope365_8a4c32897b80339b.json"),
     "marker": "person’s weight adds to the pressure of the rope on the body", "decision": "keep",
     "reason_code": "suspension_nerve_risk_context_complete",
     "reason": "The QA directly links suspension-amplified rope pressure to nerve damage in the page's detailed injury discussion."},
    {"fact_id": "fact-5a85f8f68faa51623d91", "source_path": raw("esinem_f2dfde25be14a7a8.json"),
     "marker": "The Meiroku Zasshi", "decision": "edit",
     "question": "What 1874 source does Esinem cite as evidence that the word kinbaku already existed in the nineteenth century?",
     "answer": "The Meiroku Zasshi", "reason_code": "replace_outdated_origin_claim_with_earlier_source",
     "reason": "The edit removes the article's explicitly outdated Kitan Club theory and retains its cited nineteenth-century evidence."},
    {"fact_id": "fact-e03d73de7713c52bfcde", "source_path": raw("kinbakutoday_1f811a9abc9520b4.json"),
     "marker": "in part influenced by the late Akechi Denki san’s ropework", "decision": "keep",
     "reason_code": "source_attributed_lineage_context_complete",
     "reason": "Aotsuki carefully qualifies the source-attributed partial influence, making this meaningful lineage context rather than a universal style claim."},
    {"fact_id": "fact-fc130f5c83f7850ebf85", "source_path": raw("esinem_f070e94e0fc41757.json"),
     "marker": "foundered the Toronto Kinbaku Salon", "decision": "drop",
     "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "The organization-name lookup appears in promotional visit copy and adds little durable Kazami-style knowledge."},
    {"fact_id": "fact-61369f572c05bbc6974c", "source_path": raw("kinbakutoday_5110ea5e4aec6400.json"),
     "marker": "Bakuyūkai was created by Takumi Miura with Akechi as its President", "decision": "keep",
     "reason_code": "historical_training_organization_context_complete",
     "reason": "The creator and president roles identify a study group in the source's discussion of early formal kinbaku teaching."},
    {"fact_id": "fact-7ebce2d94e6dd71d7414", "source_path": raw("kinbakutoday_7fb6f1e0e0d186b2.json"),
     "marker": "makaseru, which means to entrust something to someone else", "decision": "keep",
     "reason_code": "cultural_term_definition_context_complete",
     "reason": "The Japanese term is directly and cautiously defined in the source's broader discussion of trust."},
    {"fact_id": "fact-dd1fc47634facb140247", "source_path": raw("rope365_6812791b49f1f1a2.json"),
     "marker": "Crash Restraints by Topologist and RachelKi", "decision": "drop",
     "reason_code": "semantic_duplicate",
     "reason": "The owner-curated directory already retains Crash Restraint's canonical URL and intended suspension-learning level."},
    {"fact_id": "fact-c38b04fe2c4b0d3a3dc5", "source_path": raw("rope365_0efae4ea91870475.json"),
     "marker": "avoid adding lateral pressure on those tendons", "decision": "edit",
     "question": "What pressure does Rope365 say should be avoided around the knee tendons?",
     "answer": "lateral pressure on those tendons", "reason_code": "replace_anatomy_quiz_with_knee_safety_warning",
     "reason": "The edit replaces general bone recall with the source's rope-relevant warning about vulnerable shearing forces."},
    {"fact_id": "fact-19ef1ddd27dbd7f4202f", "source_path": raw("rope365_62f7e527bb35b47d.json"),
     "marker": "cotton/coconut/bamboo should not be oiled or waxed", "decision": "keep",
     "reason_code": "source_attributed_material_care_context_complete",
     "reason": "The QA preserves a direct, bounded care warning for three named natural-fiber materials."},
    {"fact_id": "fact-5a55da2a99d029cec233", "source_path": raw("kinbakutoday_9770f3785268b837.json"),
     "marker": "kan’nōnawa (carnal/sensual rope)", "decision": "edit",
     "question": "What does kan’nōnawa mean in the Year of the Bakushi description?",
     "answer": "carnal/sensual rope", "reason_code": "replace_person_lookup_with_style_gloss",
     "reason": "The edit makes the source's useful style gloss the answer rather than another Nureki attribution."},
    {"fact_id": "fact-60913387cb5ab2e0fb9e", "source_path": raw("wikipedia_093ebd176b6adaaf.json"),
     "marker": "Risk-aware consensual kink", "decision": "drop",
     "reason_code": "semantic_duplicate",
     "reason": "A retained QA already expands RACK exactly and supplies clearer source-specific consent context."},
    {"fact_id": "fact-b51c63f23722bb4c8a20", "source_path": raw("kinbakutoday_cf9d44df63fd716c.json"),
     "marker": "modern Kinbaku era in the 1950’s and 60’s", "decision": "edit",
     "question": "Which decades does Master K identify as the beginning of the modern kinbaku era?",
     "answer": "the 1950’s and 60’s", "reason_code": "repair_singular_decade_question",
     "reason": "The edit fixes the singular-decade mismatch while retaining the interviewee-attributed historical period."},
    {"fact_id": "fact-bbb9cc4bf2e011d2f076", "source_path": raw("wikipedia_2151448295a2af9b.json"),
     "marker": "safe, sane and consensual", "decision": "drop",
     "reason_code": "semantic_duplicate",
     "reason": "A retained edited QA already expands SSC exactly, so this duplicate would only overweight the acronym."},
    {"fact_id": "fact-f039341f5dca43990ad6", "source_path": RESOURCE_MANIFEST,
     "resource_id": "tetruss", "decision": "keep",
     "reason_code": "owner_requested_resource_directory",
     "reason": "The owner explicitly requested that Tetruss remain memorized as the recommended portable suspension-frame resource."},
    {"fact_id": "fact-517f5ab1f5e32057e9b0", "source_path": raw("rope365_3d015bb41296c9fa.json"),
     "marker": "cinches, they are flat with no bulk between the arms and the torso", "decision": "keep",
     "reason_code": "box_tie_self_evaluation_context_complete",
     "reason": "The conditional checklist item is a concrete fit precaution and does not prescribe cinches as universally required."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v11": 680,
    "active_after_this_tranche": 672,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 8,
    "new_edits_applied": 9,
    "output_rows": 710,
    "output_sha256": "d24e7218cefbbae9d5ed421bc5d38882a4d4ec9707c7d22764417ef5350e6e42",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 8,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


@contextlib.contextmanager
def patched_previous():
    replacements = {
        "OUT_DIR": OUT_DIR, "AUDIT": AUDIT, "CURATION": CURATION,
        "REPORT": REPORT, "REVIEWER": REVIEWER, "REVIEWED_AT": REVIEWED_AT,
        "CONTEXT_CURATIONS": CONTEXT_CURATIONS, "SPECS": SPECS,
        "EXPECTED_SELECTION": EXPECTED_SELECTION,
        "ISOLATED_PROJECTION": ISOLATED_PROJECTION,
    }
    originals = {name: getattr(previous, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(previous, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(previous, name, value)


def ranked_unreviewed(rows: list[dict]) -> tuple[list[dict], int, int]:
    with patched_previous():
        return previous.ranked_unreviewed(rows)


def main() -> None:
    with patched_previous():
        previous.main()
    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v12"
    write_jsonl(AUDIT, audits)
    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v12"
    report["audit"]["sha256"] = file_sha256(AUDIT)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
