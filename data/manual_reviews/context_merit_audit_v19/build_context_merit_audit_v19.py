#!/usr/bin/env python3
"""Audit context-merit tranche v19 while keeping v1-v18 byte-frozen."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V18_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v18"
sys.path[:0] = [str(ROOT), str(V18_DIR)]
import build_context_merit_audit_v18 as previous

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v19.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v19.jsonl"
REPORT = OUT_DIR / "report_context_merit_v19.json"
REVIEWER = "codex-context-merit-audit-v19"
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
    for version in range(1, 19)
)
PRIOR_CONTEXT_MERIT_DIRS = frozenset(
    path.parent.name for path in CONTEXT_CURATIONS
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-c00392e30695bb869e65", "source_path": raw("rope365_da67938bae2edac4.json"),
     "marker": "get the rope back more easily after an execution", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete",
     "reason": "A disappearing-restraint exercise based on execution and prisoner-transfer claims lacks modern body-safety and release safeguards."},
    {"fact_id": "fact-bce15b367fb8de140d70", "source_path": raw("kinbakutoday_df9c70212a927199.json"),
     "marker": "blushing, hiding one’s face", "decision": "keep",
     "reason_code": "hazukashii_reactions_context_complete",
     "reason": "The source directly lists these as possible expressions of playful, self-conscious vulnerability rather than degradation."},
    {"fact_id": "fact-9abcb867551f5f6b8c0b", "source_path": raw("rope365_6a23940f6bc37fc1.json"),
     "marker": "less stable, impossible to lock", "decision": "keep",
     "reason_code": "quick_release_tradeoffs_context_complete",
     "reason": "The source directly states three operational compromises that accompany the emergency-release benefit."},
    {"fact_id": "fact-0df8d417f95717af4ff9", "source_path": raw("rope365_a44c30b18f15504d.json"),
     "marker": "don’t press on exposed nerves", "decision": "keep",
     "reason_code": "connecting_line_nerve_precaution_context_complete",
     "reason": "The self-evaluation item gives a general high-value precaution against loading exposed nerves with connecting lines."},
    {"fact_id": "fact-eb71e8a1d6f37bc559a4", "source_path": raw("kinbakutoday_454370c8a4f42708.json"),
     "marker": "thick (about ø8mm) jute rope", "decision": "drop",
     "reason_code": "contextless_or_personal_anecdote",
     "reason": "A historical individual's personal rope diameter is not general material or safety guidance."},
    {"fact_id": "fact-0de09627431b33c05134", "source_path": raw("rope365_c73bc6fb66977a2d.json"),
     "marker": "What do you need right now?", "decision": "keep",
     "reason_code": "aftercare_support_prompt_context_complete",
     "reason": "The source gives this open question as a concrete way to support a partner, especially after a paused or stopped scene."},
    {"fact_id": "fact-19f5c3cbdee2080e3817", "source_path": raw("kinbakutoday_e7f3e175c6e3bfd7.json"),
     "marker": "feeling of endurance and helplessness", "decision": "drop",
     "reason_code": "unsafe_or_consent_context_incomplete",
     "reason": "The surrounding description repeatedly frames a partner as having no choice and being unable to resist, without explicit consent or stop context."},
    {"fact_id": "fact-1e7e96d027a985a6e8ed", "source_path": raw("esinem_0c0cea3da6ebacc2.json"),
     "marker": "observation of an expert at work", "decision": "drop",
     "reason_code": "volatile_or_promotional",
     "reason": "A generalized cultural apprenticeship claim appears in a commercial video-series announcement and is not independently sourced."},
    {"fact_id": "fact-2676e5d4761356b170d3", "source_path": raw("kinbakutoday_c799ecf7b51f866b.json"),
     "marker": "a source from which things flow", "decision": "keep",
     "reason_code": "source_attributed_ryuu_definition",
     "reason": "The answer preserves Wayne Muromoto's explicitly attributed martial-arts explanation and its contrast with technique."},
    {"fact_id": "fact-2e4c926036b797b6b2e8", "source_path": raw("rope365_3d305a5499a4db8c.json"),
     "marker": "tighten it in a single movement", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete",
     "reason": "An isolated hitch-closing microstep lacks material, loading, placement, monitoring, and failure context."},
    {"fact_id": "fact-2fbcab98b5f1824d57e9", "source_path": raw("rope365_603ebb6ae5af91f0.json"),
     "marker": "8m jute ropes of 6mm diameter", "decision": "drop",
     "reason_code": "contextless_or_low_value",
     "reason": "A site's tutorial-production rope specification is metadata, not a recommendation or reusable safety fact."},
    {"fact_id": "fact-e48096c82fe174ec30b6", "source_path": raw("rope365_6a23940f6bc37fc1.json"),
     "marker": "should be avoided if structural stability is required", "decision": "edit",
     "question": "When does Rope365 say sliding cuffs should be avoided?",
     "answer": "if structural stability is required",
     "reason_code": "replace_cuff_names_with_stability_limitation",
     "reason": "The edit replaces two cuff names with the source's explicit warning about when the sliding-cuff concept is unsuitable."},
    {"fact_id": "fact-0b35f772b45160d95a39", "source_path": raw("rope365_db7655d85616d029.json"),
     "marker": "mechanical advantage when pulling on the rope", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete",
     "reason": "The source immediately suggests using the pulley advantage to tighten a hogtie but gives no force, anatomy, monitoring, or release limits."},
    {"fact_id": "fact-45b6fd8394bc104b35d2", "source_path": raw("wikipedia_2151448295a2af9b.json"),
     "marker": "dropping a ball or ringing a bell", "decision": "keep",
     "reason_code": "nonverbal_safety_signal_context_complete",
     "reason": "The source gives these as concrete nonverbal alternatives when a participant's speech is restricted."},
    {"fact_id": "fact-56153390adce08719f5b", "source_path": raw("rope365_0b4f44c8fabfc202.json"),
     "marker": "Industrial ropes such as nylon and cotton", "decision": "keep",
     "reason_code": "rope_strength_label_context_complete",
     "reason": "The answer is narrowly limited to which manufactured ropes are likely to carry an official tensile-strength label."},
    {"fact_id": "fact-93a04bd6a8226eb8fcdc", "source_path": raw("rope365_6ab0025843fd73d5.json"),
     "marker": "Naka’s butt harness or Roughmercies’ Butt Ball", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete",
     "reason": "Two names for a load-bearing hip structure omit construction, loading, inspection, monitoring, and release context."},
    {"fact_id": "fact-de7d7356c62d5f3a074e", "source_path": raw("esinem_87458162bcec570b.json"),
     "marker": "single/double column ties and those using reverse tension", "decision": "edit",
     "question": "What two broad construction categories does the tutorial identify?",
     "answer": "those based on single/double column ties and those using reverse tension",
     "reason_code": "replace_malformed_category_list_with_complete_categories",
     "reason": "The edit restores the source's complete parallel wording for the two construction categories."},
    {"fact_id": "fact-79c048429e93f627201e", "source_path": raw("kinbakutoday_d7aad07b39a5b1e1.json"),
     "marker": "Bakushi: Scenes from Nikkatsu’s Roman Porno SM Dramas", "decision": "keep",
     "reason_code": "kinbaku_history_book_resource_context_complete",
     "reason": "The source directly identifies a named biography of an influential first-generation Japanese movie rigger."},
    {"fact_id": "fact-db5df8a1f479dc24a573", "source_path": raw("kinbakutoday_df9c70212a927199.json"),
     "marker": "an attack on a person’s sense of self", "decision": "keep",
     "reason_code": "humiliation_definition_context_complete",
     "reason": "The answer preserves the source's general description of harsher humiliation and supports its contrast with hazukashii."},
    {"fact_id": "fact-2fb6e92d3f5ecebcf609", "source_path": raw("rope365_8a4c32897b80339b.json"),
     "marker": "pulling the rope gently", "decision": "edit",
     "question": "What lower-risk rope-handling method does Rope365 recommend to avoid friction burns?",
     "answer": "pulling the rope gently",
     "reason_code": "replace_hand_barrier_with_gentle_handling",
     "reason": "The edit selects the source's simpler lower-risk prevention method instead of teaching a hand barrier during a fast pull."},
    {"fact_id": "fact-413559b2ecbc7bc3c775", "source_path": raw("kinbakutoday_697df66f16b641f3.json"),
     "marker": "middle of the 1970s", "decision": "drop",
     "reason_code": "source_reliability_or_overgeneralized",
     "reason": "An unsourced essay makes a broad first-contact claim for all Europeans that is too strong to retain as historical fact."},
    {"fact_id": "fact-e2956756d879d3d20c8f", "source_path": raw("esinem_5d1d44089d8bedf1.json"),
     "marker": "three major ingredients", "manual_evidence": "Balance\nSuspension line creativity\nAdaptation",
     "decision": "drop", "reason_code": "volatile_or_promotional",
     "reason": "A three-item suspension slogan from a commercial tutorial announcement lacks technical or safety definitions."},
    {"fact_id": "fact-825d11652acaf41b5bae", "source_path": raw("rope365_1616ffce57d993f3.json"),
     "marker": "L friction and a half hitch", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete",
     "reason": "An isolated box-tie finishing microstep lacks anatomy, loading, monitoring, and emergency-release context."},
    {"fact_id": "fact-3f801d7bd29593aff368", "source_path": raw("rope365_c73bc6fb66977a2d.json"),
     "marker": "How does my body feel?", "decision": "edit",
     "question": "What two self-scan questions does Rope365 suggest before tying or being tied?",
     "answer": "Ask yourself: “How does my body feel?” and “In what emotional state am I?”",
     "reason_code": "replace_section_title_with_pre_scene_self_scan",
     "reason": "The edit replaces a section-title lookup with concrete physical and emotional readiness questions."},
    {"fact_id": "fact-4844ac7f2672a5f62e82", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["deep_dive_single_columns", "hip_harness_playlist", "strugglers_knot_somerville_bowline"],
     "evidence": "Deep Dive into Single Columns: https://www.youtube.com/watch?v=vnDvjAaQU8g; Hip Harness Videos: https://www.youtube.com/playlist?list=PLkrdRffh_Gg2S9QccbRyiLE5x4SIacgoM; Struggler's Knot and Somerville Bowline: https://www.youtube.com/watch?v=OsIcEtCoKHo",
     "decision": "keep", "reason_code": "owner_requested_resource_directory",
     "reason": "The owner explicitly requested that all three exact tutorial links remain memorized for relevant rope-resource questions."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v18": 630,
    "active_after_this_tranche": 619,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 11,
    "new_edits_applied": 4,
    "output_rows": 657,
    "output_sha256": "1f7a87c35f0c336c9cfec52388d6e31338cb6074119d7f8af3ce8e9a38287a40",
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
        "PRIOR_CONTEXT_MERIT_DIRS": PRIOR_CONTEXT_MERIT_DIRS,
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
        row["schema"] = "context-merit-audit-v19"
    write_jsonl(AUDIT, audits)
    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v19"
    report["audit"]["sha256"] = file_sha256(AUDIT)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
