#!/usr/bin/env python3
"""Audit context-merit tranche v20 while keeping v1-v19 byte-frozen."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V19_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v19"
sys.path[:0] = [str(ROOT), str(V19_DIR)]
import build_context_merit_audit_v19 as previous

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v20.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v20.jsonl"
REPORT = OUT_DIR / "report_context_merit_v20.json"
REVIEWER = "codex-context-merit-audit-v20"
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
    for version in range(1, 20)
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
    {"fact_id": "fact-21378f6ec8d9d50d3cab", "source_path": raw("kinbakutoday_d7ec5a22f631b417.json"),
     "marker": "division between the public and the private", "decision": "drop",
     "reason_code": "source_reliability_or_overgeneralized",
     "reason": "The essay makes a sweeping contrast between Japanese and Western sexuality and shame without evidence adequate for a general cultural claim."},
    {"fact_id": "fact-ee8e2cd6104d741c4967", "source_path": raw("esinem_87458162bcec570b.json"),
     "marker": "culinary ingredients can be substituted", "decision": "keep",
     "reason_code": "interchangeable_component_analogy_context_complete",
     "reason": "The analogy supports a transferable construction lesson: functionally similar frictions or hitches can be substituted when adapting a tie."},
    {"fact_id": "fact-3197ad9ce06e85ef650c", "source_path": raw("rope365_f2d9e825760af158.json"),
     "marker": "harder to avoid the radial nerve", "decision": "edit",
     "question": "Which nerves does Rope365 warn are harder to avoid when placing upper-arm cuffs?",
     "answer": "the radial nerve on the top of the arm, and the ulnar nerve as we tie closer to the elbow",
     "reason_code": "clarify_cuff_nerve_locations",
     "reason": "The edit preserves the named nerves together with their placement context instead of presenting them as an anatomy list."},
    {"fact_id": "fact-776563e2782c24c9b599", "source_path": raw("rope365_9e8f79e4c8e703e4.json"),
     "marker": "actively learn from my teachers Nawashi Akechi Kanna and Pedro Cordas", "decision": "drop",
     "reason_code": "contextless_or_personal_anecdote",
     "reason": "A website author's current teacher-name list is personal biography rather than durable technique, safety, or history guidance."},
    {"fact_id": "fact-193cce55c0a1ce6eadcf", "source_path": raw("rope365_e5eb58544cbf8458.json"),
     "marker": "overextension at the knee or on the front of the tibia bone", "decision": "edit",
     "question": "Which pain concerns does Rope365 say to check when trying frog-tie starting points?",
     "answer": "overextension at the knee or on the front of the tibia bone",
     "reason_code": "replace_starting_microstep_with_pain_check",
     "reason": "The edit replaces a repeated starting-step lookup with the source's more useful knee and tibia pain check."},
    {"fact_id": "fact-339cfc45e7f31c668db5", "source_path": raw("kinbakutoday_0d87b1ac1a49f2d4.json"),
     "marker": "magazines, reader letters, captions, advertisements, censorship battles", "decision": "keep",
     "reason_code": "historical_source_criticism_context_complete",
     "reason": "The answer gives concrete primary-source categories for evaluating kinbaku history beyond master and lineage narratives."},
    {"fact_id": "fact-59ed68f7f1f2816cebbe", "source_path": raw("rope365_8e6f4abea3bf4f6d.json"),
     "marker": "360 friction, loop around or wrap around", "decision": "keep",
     "reason_code": "friction_term_context_complete",
     "reason": "The source directly defines the alternate names for a complete turn of rope around rope within a broader explanation of friction."},
    {"fact_id": "fact-957aca0799be5e77cb7e", "source_path": raw("kinbakutoday_1775dd4176b24104.json"),
     "marker": "learning to be competent, not knowledgeable", "decision": "keep",
     "reason_code": "teaching_competence_distinction_context_complete",
     "reason": "The source-attributed distinction is substantive and supports the surrounding warning that performing a tie competently is not sufficient preparation to teach it."},
    {"fact_id": "fact-d0eb70eeb3dfc75c36a6", "source_path": raw("kinbakutoday_15d8c1aba60b33e2.json"),
     "marker": "tool for creating the emotion of the experience", "decision": "edit",
     "question": "What does the author say rope should serve rather than become an end in itself?",
     "answer": "a tool for creating the emotion of the experience",
     "reason_code": "clarify_source_attributed_rope_philosophy",
     "reason": "The edit removes a vague context phrase and clearly attributes the author's view that rope serves the participants' emotional experience."},
    {"fact_id": "fact-ae6225a28f8e8e94895e", "source_path": raw("kinbakutoday_011f67c75b8f999f.json"),
     "marker": "looking at video performances and dissecting them step by step", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete",
     "reason": "A personal account of reverse-engineering performances is not a safe general learning recipe and omits supervised instruction and risk controls."},
    {"fact_id": "fact-82c84e1f83e6a246e2cf", "source_path": raw("rope365_5a4dc6c711aaa005.json"),
     "marker": "open the end of the rope to shorten one or two strands", "decision": "keep",
     "reason_code": "twisted_rope_maintenance_context_complete",
     "reason": "The question identifies naturally uneven strand stretch, and the source gives a specific repair for rebalancing twisted rope."},
    {"fact_id": "fact-35fbbb7942300fe1a282", "source_path": raw("esinem_c5067be0733b39c0.json"),
     "marker": "mastering Osada’s TK’s and yoko zuri", "decision": "drop",
     "reason_code": "source_reliability_or_overgeneralized",
     "reason": "A dated opinion post claims that everyone mastered two styles while offering no evidence and using mastery as a rhetorical boast."},
    {"fact_id": "fact-086715bb73267aad6cf6", "source_path": raw("rope365_f4ec955e36db6b06.json"),
     "marker": "your legs, your chest, your arms, your forehead", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete",
     "reason": "An isolated list of body parts for moving a partner omits consent, balance, force, fall prevention, and anatomy context."},
    {"fact_id": "fact-0a19d2c041161db341b3", "source_path": raw("rope365_1616ffce57d993f3.json"),
     "marker": "outside of the stem, along the back", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete",
     "reason": "A box-tie cinch-routing microstep is unsafe as standalone guidance without the page's anatomy, tension, monitoring, and emergency-release context."},
    {"fact_id": "fact-5c0dfe7cff15eed20905", "source_path": raw("rope365_a44c30b18f15504d.json"),
     "marker": "hands ability to close and open", "decision": "edit",
     "question": "What hand movement does Rope365 recommend monitoring in a box tie for signs of nerve impingement?",
     "answer": "the hands’ ability to close and open",
     "reason_code": "clarify_box_tie_nerve_monitoring",
     "reason": "The edit makes the safety purpose explicit and corrects the possessive while preserving the source's hand-movement check."},
    {"fact_id": "fact-aadae478a8a110b37947", "source_path": raw("rope365_0c46b0f988c7a825.json"),
     "marker": "the armpit, the elbow and the wrist", "decision": "keep",
     "reason_code": "upper_limb_tension_warning_context_complete",
     "reason": "The question keeps the source's warning that placement and tension are critical at these three vulnerable upper-limb areas."},
    {"fact_id": "fact-cc7974c5ed6d6f8efa1b", "source_path": raw("kinbakutoday_43095aa305e35a41.json"),
     "marker": "their fear of stopping in the middle of a session", "decision": "edit",
     "question": "What fears does Tessin Doyama say can make it hard to stop in the middle of a rope session?",
     "answer": "“maybe he/ she won’t tie me anymore” or “they won’t think i’m a real submissive”",
     "reason_code": "restore_both_barriers_to_stopping",
     "reason": "The edit restores both social fears in the source and frames them as barriers to exercising the right to stop."},
    {"fact_id": "fact-9bc384f683d5fb2073e3", "source_path": raw("rope365_42818bd00e645bda.json"),
     "marker": "SM耽美文学別巻 SMプレイ (SM Play – You can play SM)", "decision": "drop",
     "reason_code": "contextless_or_low_value",
     "reason": "A book-title lookup attached to a high-flexibility reverse-prayer variation adds less value than the page's adaptation and emergency-release cautions."},
    {"fact_id": "fact-a7e362c67c6ee220d7e1", "source_path": raw("kinbakutoday_d7aad07b39a5b1e1.json"),
     "marker": "cotton rope sold in Japan as “honey rope”", "decision": "drop",
     "reason_code": "contextless_or_personal_anecdote",
     "reason": "A historical rigger's material choice for film actresses is an anecdote, not a general material or safety recommendation."},
    {"fact_id": "fact-f6819a1cedb44d6048a2", "source_path": raw("rope365_a3b2e9e479b0c70f.json"),
     "marker": "Tie them in a daisy chain or put them in a pillowcase", "decision": "keep",
     "reason_code": "wash_tangle_prevention_context_complete",
     "reason": "The source directly gives two practical ways to keep washable cotton and synthetic rope from tangling in a machine."},
    {"fact_id": "fact-36b5e01a7b534076f61a", "source_path": raw("rope365_a44c30b18f15504d.json"),
     "marker": "Chest harnesses and box ties makes excellent structure", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete",
     "reason": "An isolated recommendation to attach hogtie designs to chest or box structures omits breathing, nerve, pressure, and release constraints."},
    {"fact_id": "fact-44480871b875f72e8c29", "source_path": raw("rope365_9a5c5810310fa0f0.json"),
     "marker": "letting the rope lead me", "decision": "keep",
     "reason_code": "improvisation_framework_context_complete",
     "reason": "The source defines this as an improvisational exercise after explicitly discussing touch negotiation and basic placement cautions."},
    {"fact_id": "fact-0717fb4cb44551293095", "source_path": raw("kinbakutoday_0d87b1ac1a49f2d4.json"),
     "marker": "SMの思想史 (A History of SM Thought)", "decision": "keep",
     "reason_code": "postwar_sm_history_resource_context_complete",
     "reason": "The title identifies a substantive scholarly resource about postwar Japanese SM culture and historical source criticism."},
    {"fact_id": "fact-d0ab81fad8b39983bc97", "source_path": raw("wikipedia_5368de8bae5a78d8.json"),
     "marker": "width, about equal to the diameter of the rope", "decision": "keep",
     "reason_code": "whipping_proportion_context_complete",
     "reason": "The source-attributed proportion is a concise, reusable guideline for finishing rope ends to resist fraying."},
    {"fact_id": "fact-d11aba1283f69293bf55", "source_path": raw("rope365_3d015bb41296c9fa.json"),
     "marker": "forearms in an horizontal position without crossing the wrists",
     "manual_evidence": "Bring the hands together in the back with the forearms in an horizontal position without crossing the wrists so that it pushes the elbow out.",
     "decision": "edit",
     "question": "How are the forearms positioned in Rope365’s more accessible ‘Elbows Outward’ variation?",
     "answer": "in a horizontal position without crossing the wrists",
     "reason_code": "correct_article_in_accessible_position_description",
     "reason": "The edit fixes the source's grammatical article while preserving the exact arm position and its accessibility framing."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(EXPECTED_SELECTION, (
        91, 124, 591, 596, 341, 399, 232, 160, 23, 469, 220, 598, 514,
        510, 389, 581, 206, 102, 459, 350, 594, 338, 285, 20, 93,
    ))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v19",
    "eligible_unreviewed_rows": 30,
    "excluded_active_review_provenance": 302,
    "excluded_ledger_fact_ids": 1520,
    "rows": 657,
    "sha256": "1f7a87c35f0c336c9cfec52388d6e31338cb6074119d7f8af3ce8e9a38287a40",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v19": 619,
    "active_after_this_tranche": 610,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 9,
    "new_edits_applied": 6,
    "output_rows": 648,
    "output_sha256": "46cf30a0e49a9daaf874462e20448d52f43f73b551b611e5734a52b09065349a",
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
        row["schema"] = "context-merit-audit-v20"
        row["active_index"] = PROJECTED_ACTIVE_INDICES[row["fact_id"]]
    write_jsonl(AUDIT, audits)

    curations = read_jsonl(CURATION)
    for row in curations:
        if row["fact_id"] == "fact-d11aba1283f69293bf55":
            row["support_type"] = "manual_paraphrase"
            row["paraphrase_rationale"] = (
                "Corrects the source's article from ‘an horizontal’ to "
                "‘a horizontal’ without changing the described position."
            )
    write_jsonl(CURATION, curations)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v20"
    report["selection"]["active_rows"] = PROJECTED_SELECTION_BASELINE["rows"]
    report["selection"]["eligible_unreviewed_rows"] = 30
    report["selection"]["excluded_active_review_provenance"] = 302
    report["selection"]["excluded_ledger_fact_ids"] = 1520
    report["selection"]["projected_baseline"] = PROJECTED_SELECTION_BASELINE
    report["audit"]["sha256"] = file_sha256(AUDIT)
    report["new_pending_curation"]["sha256"] = file_sha256(CURATION)
    report["new_pending_curation"]["edit_support_types"] = {
        "extractive": 5, "manual_paraphrase": 1,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
