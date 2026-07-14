#!/usr/bin/env python3
"""Audit context-merit tranche v18 while keeping v1-v17 byte-frozen."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V17_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v17"
sys.path[:0] = [str(ROOT), str(V17_DIR)]
import build_context_merit_audit_v17 as previous

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v18.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v18.jsonl"
REPORT = OUT_DIR / "report_context_merit_v18.json"
REVIEWER = "codex-context-merit-audit-v18"
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
    for version in range(1, 18)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-a6e93ed860c45d0b353d", "source_path": raw("esinem_64c4212a11867d61.json"),
     "marker": "basic building block of shibari", "decision": "keep",
     "reason_code": "single_column_foundation_context_complete",
     "reason": "The source directly characterizes the single-column tie as a foundational shibari element and situates it as a common starting tie."},
    {"fact_id": "fact-d7f5bb80b07cc9a258da", "source_path": raw("esinem_b249eddc1f5e1864.json"),
     "marker": "loops getting caught in each other", "decision": "keep",
     "reason_code": "rope_untangling_context_complete",
     "reason": "The source directly distinguishes loop entanglement from actual knots while explaining how rope tangles form."},
    {"fact_id": "fact-99d3f29be1c8ebe8a59b", "source_path": raw("rope365_30eaedb9cb1e6d5b.json"),
     "marker": "arm span of the person tying", "decision": "keep",
     "reason_code": "rope_length_guideline_context_complete",
     "reason": "The source presents arm span as one common personal measurement guideline, not as a universal required rope length."},
    {"fact_id": "fact-d18ad9be4afafc80d1cb", "source_path": raw("kinbakutoday_5a6e15f57e52a345.json"),
     "marker": "meaning of a word is always drifting and changing", "decision": "edit",
     "question": "What conclusion does Ugo draw about historical kinbaku and shibari terminology?",
     "answer": "the meaning of a word is always drifting and changing",
     "reason_code": "replace_awkward_historical_fragment_with_language_caution",
     "reason": "The edit replaces an awkward era-specific fragment with the interviewee's explicit caution against freezing changing terminology."},
    {"fact_id": "fact-6e34b036a3d193c49324", "source_path": raw("rope365_1db41038891a4e1f.json"),
     "marker": "double coin knot and/or triple crown knot", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete",
     "reason": "A decorative chest-harness microstep lacks placement, tension, breathing, monitoring, loading, and release context."},
    {"fact_id": "fact-f2ce7292f1b3a5d5d140", "source_path": raw("kinbakutoday_5110ea5e4aec6400.json"),
     "marker": "We must respect them", "decision": "edit",
     "question": "What principle does Kazami Ranki give for relating to different rope styles?",
     "answer": "We must respect them. Each of them has a different way of tying, but that is individuality as people say, and we should not deny individuality",
     "reason_code": "replace_person_list_with_style_respect_principle",
     "reason": "The edit replaces a five-person list with Kazami's substantive principle of respecting stylistic individuality."},
    {"fact_id": "fact-2a7d7893e51e8bb68a2c", "source_path": raw("wikipedia_2151448295a2af9b.json"),
     "marker": "encourage discussion and negotiation in advance", "decision": "keep",
     "reason_code": "bdsm_contract_purpose_context_complete",
     "reason": "The source directly states that such documents support prior negotiation and shared understanding, while noting they are not legally binding."},
    {"fact_id": "fact-35aa8d5c46826a467bd3", "source_path": raw("esinem_4a64b6d0e45a7c34.json"),
     "marker": "Dexterity in avoiding friction movement", "decision": "keep",
     "reason_code": "rope_handling_term_context_complete",
     "reason": "The term is directly glossed and followed by practical context about preventing unplanned rubbing, pinching, and knot impact."},
    {"fact_id": "fact-b2de0d8786d7488a796b", "source_path": raw("rope365_f43c9fde09431a5f.json"),
     "marker": "alternating going over and under", "decision": "keep",
     "reason_code": "weaving_definition_context_complete",
     "reason": "The source directly defines the foundational over-under action that creates a weaving structure."},
    {"fact_id": "fact-d71077841714573434f6", "source_path": raw("rope365_da67938bae2edac4.json"),
     "marker": "hayanawa and the honnawa techniques", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete",
     "reason": "A historical capture-over-restraint exercise includes neck, nerve, choking, and escape hazards without modern instructional safeguards."},
    {"fact_id": "fact-f2b05e4a3c1b94190295", "source_path": raw("esinem_dc5ce54e30fa890c.json"),
     "marker": "Kinoko Hajime and Kazami Ranki", "decision": "drop",
     "reason_code": "volatile_or_promotional",
     "reason": "Instructor-lineage trivia in a commercial technique tutorial announcement adds little durable knowledge."},
    {"fact_id": "fact-5b492ea062f8b5f976cb", "source_path": raw("rope365_c73bc6fb66977a2d.json"),
     "marker": "Save the relationship, not the scene", "decision": "keep",
     "reason_code": "incident_priority_context_complete",
     "reason": "The phrase concisely preserves the source's instruction to prioritize people and relationships when something goes wrong."},
    {"fact_id": "fact-6ba5f96154209f20abae", "source_path": raw("kinbakutoday_fd0d7ba4b6589765.json"),
     "marker": "Aibunawa (愛撫縄）or Aibu no Nawa", "decision": "keep",
     "reason_code": "kinbaku_style_term_context_complete",
     "reason": "The source directly gives the Japanese characters and both romanized forms for Yukimura's named signature style."},
    {"fact_id": "fact-a09f4a17066c23e7d4e8", "source_path": raw("rope365_9a5c5810310fa0f0.json"),
     "marker": "avoid rope on the front of the neck", "decision": "keep",
     "reason_code": "neck_avoidance_context_complete",
     "reason": "The source explicitly gives this high-value placement precaution in its improvisation guidance."},
    {"fact_id": "fact-a0afb7701efe4fe1a94d", "source_path": raw("kinbakutoday_00507964d2e38f67.json"),
     "marker": "Jouen – The world of Minomura Kou", "decision": "drop",
     "reason_code": "contextless_or_low_value",
     "reason": "A film-title lookup adds less durable value than the same source's retained historical information about Minomura's publications."},
    {"fact_id": "fact-b02267f4c9ca9d05c632", "source_path": raw("rope365_441f9cc87ead6159.json"),
     "marker": "lark’s head or a square knot", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete",
     "reason": "An isolated rope-extension recommendation omits material, loading, tail, inspection, and failure-mode considerations."},
    {"fact_id": "fact-e6cbb01165dcdaf1af1f", "source_path": raw("kinbakutoday_c4f56ccca10ec95c.json"),
     "marker": "world of rope is big enough", "decision": "edit",
     "question": "What view does the author take on the supposed opposition between rope as art and rope as kink?",
     "answer": "the world of rope is big enough to accommodate all interests, from the most artistic to the most perverted",
     "reason_code": "replace_time_bound_theme_with_inclusive_viewpoint",
     "reason": "The edit replaces a time-bound discourse label with the author's substantive position that artistic and kinky interests can coexist."},
    {"fact_id": "fact-980c927b8529e9618af4", "source_path": raw("esinem_457ffe4f18d12793.json"),
     "marker": "over-oiling or waxing", "decision": "edit",
     "question": "What two common jute-rope preparation mistakes does Esinem identify?",
     "answer": "over-oiling or waxing, making the rope heavy and greasy. Secondly, poor burning off",
     "reason_code": "replace_fragmented_list_with_complete_preparation_mistakes",
     "reason": "The edit replaces a compressed list with the source's complete description of the first mistake and explicit naming of the second."},
    {"fact_id": "fact-a7d8ef2d7b59d52c4c2d", "source_path": raw("rope365_f68f65483c9db322.json"),
     "marker": "stimulate regular practice of rope bondage", "decision": "drop",
     "reason_code": "volatile_or_promotional",
     "reason": "A site's self-described project aim is promotional metadata rather than reusable rope knowledge."},
    {"fact_id": "fact-d4877ffdbfde2bfbd238", "source_path": raw("kinbakutoday_fe60466ee5e6689e.json"),
     "marker": "simplicity, rusticity, imperfection and transience of life", "decision": "keep",
     "reason_code": "wabi_sabi_context_complete",
     "reason": "The source explicitly labels these as a general, oversimplified characterization and explains that deeper cultural context is needed."},
    {"fact_id": "fact-6a836e308b27cd02b225", "source_path": raw("rope365_bdb2a15da368be35.json"),
     "marker": "Monitor sensation on the leg", "decision": "edit",
     "question": "What monitoring precaution does Rope365 give for nerve risk in a frog tie?",
     "answer": "Monitor sensation on the leg when tying and be mindful of the amount of time spent in this tie",
     "reason_code": "replace_nerve_list_with_monitoring_precaution",
     "reason": "The edit makes the page's actionable monitoring and duration precaution retrievable instead of only listing two nerve names."},
    {"fact_id": "fact-6c669507afcf171c5d53", "source_path": raw("rope365_1db41038891a4e1f.json"),
     "marker": "shinju 真珠 (pearl) and munenawa 胸縄", "decision": "keep",
     "reason_code": "translated_chest_harness_terms_context_complete",
     "reason": "The source directly pairs both Japanese terms and characters with their English glosses."},
    {"fact_id": "fact-e2ea016383ae5f8fb99c", "source_path": raw("kinbakutoday_fb4cd3fac3469c42.json"),
     "marker": "reason – why you tie – and the target", "decision": "drop",
     "reason_code": "overgeneralized_or_context_incomplete",
     "reason": "A personal essay's fragmentary claim about what makes a session great does not define its ambiguous 'target' and overstates a subjective view."},
    {"fact_id": "fact-92a9f9e3596df652fe26", "source_path": raw("rope365_6a23940f6bc37fc1.json"),
     "marker": "wise to pick one and master it", "decision": "edit",
     "question": "What learning approach does Rope365 recommend before comparing many single-column ties?",
     "answer": "At first, it is wise to pick one and master it, and down the road, it becomes interesting to compare different methods",
     "reason_code": "replace_bowline_list_with_learning_sequence",
     "reason": "The edit replaces an unsupported list of bowlines to try with the page's safer learning sequence of mastery before comparison."},
    {"fact_id": "fact-7821b554008b1a2889dd", "source_path": raw("kinbakutoday_20d1161d0b76162f.json"),
     "marker": "an agreement and a confidential relationship with a rope bottom", "decision": "keep",
     "reason_code": "source_attributed_relationship_principle",
     "reason": "The answer preserves Nawanojyoh's explicitly attributed advice to ground practice in agreement and a trusted relationship rather than self-gratification."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v17": 637,
    "active_after_this_tranche": 630,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 7,
    "new_edits_applied": 6,
    "output_rows": 668,
    "output_sha256": "02d7dbbb860f875d9ee4a3f8d94e2dcc43eb09e97548fd7029da2e0ebaf3026a",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 12,
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
        row["schema"] = "context-merit-audit-v18"
    write_jsonl(AUDIT, audits)
    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v18"
    report["audit"]["sha256"] = file_sha256(AUDIT)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
