#!/usr/bin/env python3
"""Audit context-merit tranche v15 while keeping v1-v14 byte-frozen."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V14_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v14"
sys.path[:0] = [str(ROOT), str(V14_DIR)]
import build_context_merit_audit_v14 as previous

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v15.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v15.jsonl"
REPORT = OUT_DIR / "report_context_merit_v15.json"
REVIEWER = "codex-context-merit-audit-v15"
REVIEWED_AT = "2026-07-14"
ACTIVE_DATASET = previous.ACTIVE_DATASET
ACTIVE_REPORT = previous.ACTIVE_REPORT
ACTIVE_CURATIONS = previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS = previous.PRIOR_PENDING_ADDITIONS
QUALITY_MERIT_CURATION = previous.QUALITY_MERIT_CURATION
TASUKI_CURATION = previous.TASUKI_CURATION
CONTEXT_CURATIONS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"pending_curation_context_merit_v{version}.jsonl"
    for version in range(1, 15)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-cf9e216f175243807b5f", "source_path": raw("kinbakutoday_5a6e15f57e52a345.json"),
     "marker": "two separate activities: Strip Theatre", "decision": "edit",
     "question": "What two live-performance activities does the interview distinguish in 1960s Japan?",
     "answer": "Strip Theatre (including movie theatre), and Underground Theatre",
     "reason_code": "replace_magazine_lookup_with_performance_history",
     "reason": "The edit replaces a magazine-name lookup with the interview's substantive distinction between two live-performance settings."},
    {"fact_id": "fact-cfde2822a7d1a048f5da", "source_path": raw("wikipedia_a97e19abfe49afa9.json"),
     "marker": "3–4 millimeters in diameter", "decision": "drop", "reason_code": "unsafe_or_context_incomplete",
     "reason": "The diameter of historical capture cord is not modern bondage guidance and could encourage hazardous use of very thin cord."},
    {"fact_id": "fact-2efe44e69d2ba27de2ef", "source_path": raw("rope365_5c6e72d053d47d66.json"),
     "marker": "square knot single-column tie", "decision": "drop", "reason_code": "unsafe_or_context_incomplete",
     "reason": "An isolated box-tie starting-knot instruction omits anatomy, fit, monitoring, loading, and emergency-release context."},
    {"fact_id": "fact-69ab8eacd4b4f1f541bd", "source_path": raw("rope365_5f637b24ae808142.json"),
     "marker": "hexagonal shape aka kikkō", "decision": "drop", "reason_code": "unsafe_or_context_incomplete",
     "reason": "A load-bearing torso attachment shape is presented without enough placement, loading, monitoring, or release guidance."},
    {"fact_id": "fact-bd83fd95d4339e81c64f", "source_path": raw("kinbakutoday_454370c8a4f42708.json"),
     "marker": "‘All suspension book’ (オール吊りの本)", "decision": "drop", "reason_code": "contextless_or_low_value",
     "reason": "A photo-book title and model lookup adds little durable value relative to the interview's broader oral history."},
    {"fact_id": "fact-c0cc30ba0b937b0f11f1", "source_path": raw("kinbakutoday_cbbff84b319d0813.json"),
     "marker": "prevention of escape their primary object", "decision": "keep", "reason_code": "hojojutsu_function_context_complete",
     "reason": "The named author's statement directly identifies the historical primary function of capture rope."},
    {"fact_id": "fact-0711cae41e4f9fb240b1", "source_path": raw("rope365_c73bc6fb66977a2d.json"),
     "marker": "use of condoms", "decision": "drop", "reason_code": "contextless_or_low_value",
     "reason": "A bare example detached from the source's negotiation questions does not teach a complete sexual-safety practice."},
    {"fact_id": "fact-211255b835fed84a6246", "source_path": raw("rope365_c73bc6fb66977a2d.json"),
     "marker": "An allowlist gives a better result than a blocklist", "decision": "edit",
     "question": "What negotiation approach does Rope365 recommend instead of accepting “everything is fine”?",
     "answer": "An allowlist gives a better result than a blocklist",
     "reason_code": "replace_phrase_recall_with_negotiation_principle",
     "reason": "The edit turns a phrase-recall question into the source's reusable negotiation principle."},
    {"fact_id": "fact-28b9282fc0151c2e818c", "source_path": raw("rope365_f250f228cc370052.json"),
     "marker": "Two fingers gap inside the cuff", "decision": "drop", "reason_code": "unsafe_or_overgeneralized",
     "reason": "A fixed finger-width fit rule is anatomy-dependent and lacks circulation, sensation, movement, loading, and monitoring context."},
    {"fact_id": "fact-640b5d11e9620de2ef12", "source_path": raw("esinem_77368ccdc66acad0.json"),
     "marker": "Bottoming is not shameful. Pain is not shameful. Submission is not shameful", "decision": "edit",
     "question": "What three affirmations does the rope performer make about bottoming, pain, and submission?",
     "answer": "Bottoming is not shameful. Pain is not shameful. Submission is not shameful",
     "reason_code": "replace_malformed_list_with_complete_affirmations",
     "reason": "The edit replaces a malformed three-word list with the source's complete affirmations in a discussion of agency and consent."},
    {"fact_id": "fact-cbf057c39ac5a4aa3b22", "source_path": raw("kinbakutoday_1775dd4176b24104.json"),
     "marker": "injuries and mistakes are telling you", "decision": "edit",
     "question": "What first lesson does the author say injuries and mistakes can teach a rigger?",
     "answer": "injuries and mistakes are telling you that you may not know as much as you think you know",
     "reason_code": "replace_school_label_with_injury_learning",
     "reason": "The edit replaces a vague school-of-thought label with the article's concrete lesson about responding to injuries and mistakes."},
    {"fact_id": "fact-d472741cbc60d803162e", "source_path": raw("esinem_4a64b6d0e45a7c34.json"),
     "marker": "Akechi tied to the left as he was left-handed", "decision": "drop", "reason_code": "anecdotal_or_overgeneralized",
     "reason": "The source presents the handedness explanation as hearsay and immediately questions whether a deeper reason exists."},
    {"fact_id": "fact-d5da093fe5fa61907e5f", "source_path": raw("rope365_a19e1d759fa86b73.json"),
     "marker": "Farmers and Sailors", "decision": "drop", "reason_code": "contextless_or_low_value",
     "reason": "Two headings from an unpublished course outline do not provide a historical claim or explanation."},
    {"fact_id": "fact-1a5410c4a2f8fe7fddf6", "source_path": raw("rope365_c73bc6fb66977a2d.json"),
     "marker": "Interview with Heather Elizabeth", "decision": "drop", "reason_code": "volatile_or_promotional",
     "reason": "A title from a promotional podcast list is less useful than the source's directly stated consent guidance."},
    {"fact_id": "fact-994ebfad714f14f3be5d", "source_path": raw("kinbakutoday_86c13124375e690a.json"),
     "marker": "Flower and Snake", "decision": "keep", "reason_code": "kinbaku_literary_history_context_complete",
     "reason": "The source directly identifies a culturally influential work and situates its adaptations in early Japanese kinbaku publishing."},
    {"fact_id": "fact-c5b7b45a142ef29c7a32", "source_path": raw("rope365_7b0aa0a7481d6e35.json"),
     "marker": "zenpou takate shibari 前方高手縛り", "decision": "keep", "reason_code": "translated_pose_term_context_complete",
     "reason": "The Japanese term, characters, and English pose gloss occur together in the source."},
    {"fact_id": "fact-fa78f985ebbbb2370a5f", "source_path": raw("rope365_c73bc6fb66977a2d.json"),
     "marker": "first contact with a new partner in a public space", "decision": "keep", "reason_code": "new_partner_safety_context_complete",
     "reason": "The public first-contact recommendation is directly supported alongside safety calls and reference checks."},
    {"fact_id": "fact-474bee747d1af1e3b77c", "source_path": raw("kinbakutoday_d33c9aadf7e593e7.json"),
     "marker": "Somatics for Rope Bottoms", "decision": "keep", "reason_code": "rope_bottoming_resource_context_complete",
     "reason": "The source identifies a stable, named book resource written specifically for people who like being in rope."},
    {"fact_id": "fact-491c19ed0b516b7c2070", "source_path": raw("rope365_27a1135f100eebcf.json"),
     "marker": "Pick a pattern (box tie, frog tie)", "decision": "drop", "reason_code": "unsafe_or_context_incomplete",
     "reason": "A speed-and-handedness exercise names restrictive patterns without anatomy, risk, monitoring, or release context."},
    {"fact_id": "fact-8ffdaa8035d2e57c1900", "source_path": raw("kinbakutoday_d4dcb268cb41c5e4.json"),
     "marker": "allowed seme-e and writing about female masochism", "decision": "edit",
     "question": "What did the feminine pen name Kita Reiko make possible in early SM publishing?",
     "answer": "it allowed seme-e and writing about female masochism to appear as if they emerged from a woman’s own aesthetic and erotic interiority",
     "reason_code": "replace_magazine_list_with_authorship_analysis",
     "reason": "The edit replaces a publication list with the source's substantive analysis of gendered pseudonymous authorship."},
    {"fact_id": "fact-f05b17a22ab0bd03af7b", "source_path": raw("rope365_8ae9e3d93b31601b.json"),
     "marker": "ushiro takate kote shibari 後ろ高手 小手縛り", "decision": "keep",
     "reason_code": "translated_tie_term_context_complete",
     "reason": "The source directly pairs the Japanese term and characters with the hands-high-in-back gloss."},
    {"fact_id": "fact-16926b4cd64215195ce8", "source_path": raw("kinbakutoday_86086b3e64ae77fa.json"),
     "marker": "Talking with a trained counselor can be a good idea", "decision": "keep",
     "reason_code": "post_incident_support_context_complete",
     "reason": "The source appropriately identifies trained counseling as one option for processing injury or consent-violation experiences."},
    {"fact_id": "fact-3b647333a1ace1f99666", "source_path": raw("rope365_1616ffce57d993f3.json"),
     "marker": "ulnar nerve is often exposed at the elbow", "decision": "keep",
     "reason_code": "nerve_anatomy_context_complete",
     "reason": "The source directly connects the exposed ulnar nerve at the elbow with the familiar funny-bone location."},
    {"fact_id": "fact-531fe868840964d4d4af", "source_path": raw("rope365_5c6e72d053d47d66.json"),
     "marker": "rope pressure should be on the forearms", "decision": "keep",
     "reason_code": "wrist_pressure_safety_context_complete",
     "reason": "The answer preserves the source's explicit precaution to avoid direct pressure on wrists and hands."},
    {"fact_id": "fact-ec22c5ec841d990b5c62", "source_path": raw("kinbakutoday_5354aadb617a8182.json"),
     "marker": "Shunga erotic paintings", "decision": "keep",
     "reason_code": "rope_art_history_context_complete",
     "reason": "The source directly identifies the historical artwork offered as evidence of older Japanese rope-play imagery."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v14": 662,
    "active_after_this_tranche": 652,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 10,
    "new_edits_applied": 5,
    "output_rows": 690,
    "output_sha256": "ff67453be01515d9d3fa8cdb7c7dfb5416c9db032a669e1ce048183a9674bd74",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 10,
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
        row["schema"] = "context-merit-audit-v15"
    write_jsonl(AUDIT, audits)
    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v15"
    report["audit"]["sha256"] = file_sha256(AUDIT)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
