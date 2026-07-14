#!/usr/bin/env python3
"""Audit context-merit tranche v16 while keeping v1-v15 byte-frozen."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V15_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v15"
sys.path[:0] = [str(ROOT), str(V15_DIR)]
import build_context_merit_audit_v15 as previous

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v16.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v16.jsonl"
REPORT = OUT_DIR / "report_context_merit_v16.json"
REVIEWER = "codex-context-merit-audit-v16"
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
    for version in range(1, 16)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-25c03ed0ec7b6318da72", "source_path": raw("esinem_63054bcce13821a0.json"),
     "marker": "designed to gauge the overall readiness", "decision": "edit",
     "question": "According to the tutorial announcement, what is Aisatsu shibari designed to gauge?",
     "answer": "the overall readiness of your rope partner for an emotional exchange",
     "reason_code": "replace_term_lookup_with_source_attributed_purpose",
     "reason": "The edit replaces a bilingual term lookup with the source's stated relational purpose while keeping the claim explicitly attributed."},
    {"fact_id": "fact-42e3526e742dc48ac095", "source_path": raw("rope365_8a4c32897b80339b.json"),
     "marker": "wrist, elbows, knees, ankles", "decision": "keep",
     "reason_code": "nerve_risk_mitigation_context_complete",
     "reason": "The source explicitly identifies these joints in its advice to tie loosely or away from joints to mitigate nerve compression."},
    {"fact_id": "fact-9a4bfdb0a5a5a2f8c993", "source_path": raw("rope365_0efae4ea91870475.json"),
     "marker": "get the spine straight as a starting position", "decision": "drop",
     "reason_code": "unsafe_or_medically_unsupported",
     "reason": "A prescriptive spinal-torsion starting position lacks medical qualification, screening, range-of-motion limits, monitoring, and stop criteria."},
    {"fact_id": "fact-a376faafe5340d14f4fd", "source_path": raw("esinem_c93603a2c7d48255.json"),
     "marker": "more slack allows the model more flexibility", "decision": "edit",
     "question": "Why does Esinem recommend additional wrist slack in the gote discussed?",
     "answer": "more slack allows the model more flexibility to change wrist positions should discomfort, circulation issues or pressure on a nerve become a problem",
     "reason_code": "replace_name_lookup_with_wrist_safety_rationale",
     "reason": "The edit replaces a naming preference with the article's explicit rationale for giving the wrists more adjustment space."},
    {"fact_id": "fact-bc86aee004f716e30751", "source_path": raw("rope365_9a5c5810310fa0f0.json"),
     "marker": "Make sure to discuss what kind of touching", "decision": "edit",
     "question": "What should partners discuss before improvising with rope?",
     "answer": "Make sure to discuss what kind of touching and what parts of the body are good to tie on before starting to improvise",
     "reason_code": "replace_technique_lookup_with_pre_scene_discussion",
     "reason": "The edit replaces a technique-name lookup with the source's concrete pre-improvisation discussion prompt."},
    {"fact_id": "fact-2f3e2526a82142759634", "source_path": raw("kinbakutoday_381214111022a952.json"),
     "marker": "rope bottom that will lose the feeling of connection first", "decision": "keep",
     "reason_code": "source_attributed_partner_attunement",
     "reason": "The answer preserves Yukimura's explicitly attributed observation about how quickly a partner senses disengagement."},
    {"fact_id": "fact-73bfd275bc05b00e360b", "source_path": raw("rope365_f250f228cc370052.json"),
     "marker": "knot can compact and become difficult to untie", "decision": "edit",
     "question": "What emergency-relevant limitation does Rope365 give for the Somerville bowline?",
     "answer": "with a lot of pressure, the knot can compact and become difficult to untie, especially when the tail is under tension",
     "reason_code": "replace_device_lookup_with_release_limitation",
     "reason": "The edit replaces a generic cutting-device answer with the knot condition that can make emergency release difficult."},
    {"fact_id": "fact-a8f6bd49d376a4ef43c6", "source_path": raw("rope365_7fe5be31cb8dc67e.json"),
     "marker": "joining ability of a triple-fisherman", "decision": "drop", "reason_code": "unsafe_or_context_incomplete",
     "reason": "A joining-knot recommendation from a source framed as one contributor's adaptations lacks validated load, tail, and failure limits."},
    {"fact_id": "fact-c2b4559b9b0d740a6d3f", "source_path": raw("kinbakutoday_b776fccd348e2538.json"),
     "marker": "classic Ebi (shrimp) tie", "decision": "drop", "reason_code": "volatile_or_promotional",
     "reason": "A tie-identification question from a commercial streaming-video announcement is promotional and lacks usable safety context."},
    {"fact_id": "fact-f4586789416e5e080353", "source_path": raw("rope365_1616ffce57d993f3.json"),
     "marker": "ulnar nerve in the elbow", "decision": "drop", "reason_code": "semantic_duplicate",
     "reason": "The immediately preceding audited tranche already retains the same source's ulnar-nerve-at-the-elbow fact with fuller context."},
    {"fact_id": "fact-6ad3ef0f91eb2f4932f0", "source_path": raw("rope365_5a4dc6c711aaa005.json"),
     "marker": "Synthetic and cotton ropes are easy to wash", "decision": "keep",
     "reason_code": "rope_hygiene_context_complete",
     "reason": "The source directly distinguishes easy-to-wash synthetic and cotton ropes from natural fibers weakened by wet cleaning."},
    {"fact_id": "fact-e81b85d567ed74cf39f3", "source_path": raw("rope365_a3b2e9e479b0c70f.json"),
     "marker": "cotton or nylon rope 6mm (1/4 inch)", "decision": "drop",
     "reason_code": "unsafe_or_overgeneralized",
     "reason": "A fixed hardware-store buying prescription omits construction, chemical treatment, load, use case, inspection, and supplier-quality considerations."},
    {"fact_id": "fact-68a885d8cedec1a9b4e9", "source_path": raw("rope365_7fe5be31cb8dc67e.json"),
     "marker": "adding a L-friction", "decision": "drop", "reason_code": "unsafe_or_context_incomplete",
     "reason": "An isolated modification to a potentially loaded harness lacks placement, loading, monitoring, and release context."},
    {"fact_id": "fact-77e3d75f250828ed7077", "source_path": raw("rope365_27a1135f100eebcf.json"),
     "marker": "one-handed lark’s head", "decision": "drop", "reason_code": "unsafe_or_context_incomplete",
     "reason": "A speed-oriented, one-handed rope-extension microstep omits joining conditions, loading limits, inspection, and failure modes."},
    {"fact_id": "fact-7e6623a80710cd2ba0e3", "source_path": raw("rope365_8012cecdbbfc3fc6.json"),
     "marker": "box tie pattern", "decision": "drop", "reason_code": "unsafe_or_context_incomplete",
     "reason": "An isolated restrictive harness-construction step lacks anatomy, fit, nerve-risk, loading, monitoring, and release context."},
    {"fact_id": "fact-ba692623a8506581339b", "source_path": raw("rope365_25f1b23eb40be00e.json"),
     "marker": "quick slip knot is a good way to mark the middle", "decision": "keep",
     "reason_code": "rope_handling_utility_context_complete",
     "reason": "The answer concerns a non-body rope-handling use and the source explicitly notes that slip knots themselves are unstable."},
    {"fact_id": "fact-529b939b863c223295c2", "source_path": raw("kinbakutoday_c799ecf7b51f866b.json"),
     "marker": "kokoro is the “why” and the katachi is the “how”", "decision": "edit",
     "question": "How does the author define kokoro and katachi in tying?",
     "answer": "The kokoro is the “why” and the katachi is the “how” of tying",
     "reason_code": "replace_abstract_relation_with_direct_definitions",
     "reason": "The edit replaces a circular relationship question with the author's direct, concise definitions of both concepts."},
    {"fact_id": "fact-9ff31d5fea6485f24db2", "source_path": raw("wikipedia_2151448295a2af9b.json"),
     "marker": "dungeon monitors (DMs) provide an additional safety net", "decision": "edit",
     "question": "Who provides an additional safety net at many clubs and group-organized BDSM events?",
     "answer": "dungeon monitors (DMs)",
     "reason_code": "remove_unsupported_professional_label",
     "reason": "The edit removes the unsupported claim that dungeon monitors are professionals while preserving their stated event-safety role."},
    {"fact_id": "fact-af6b6f7d74945786a476", "source_path": raw("rope365_8e92591933438133.json"),
     "marker": "make a crossing hitch every time two rope cross", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete",
     "reason": "A group bondage-game microstep lacks responsibility, consent, body placement, loading, monitoring, and emergency-release context."},
    {"fact_id": "fact-bf91b4f8c612c46d312c", "source_path": raw("rope365_02b5a435a4822809.json"),
     "marker": "two-point threshold", "decision": "keep", "reason_code": "touch_perception_definition_context_complete",
     "reason": "The source directly defines a standard tactile-acuity concept and situates it as an observation about differing touch sensitivity."},
    {"fact_id": "fact-43edf18670f78cc2320a", "source_path": raw("rope365_02b5a435a4822809.json"),
     "marker": "ulnar nerve is running in this location", "decision": "drop",
     "reason_code": "unsafe_or_medically_unsupported",
     "reason": "The claim appears inside prescriptive pain-compliance instructions and does not adequately establish anatomical location or safe application."},
    {"fact_id": "fact-a10102648dda8f012292", "source_path": raw("rope365_c45c6927d294c427.json"),
     "marker": "Munter hitch was originally designed", "decision": "edit",
     "question": "How does Rope365 distinguish the purpose of a Munter hitch from a crossing hitch?",
     "answer": "The Munter hitch was originally designed to be used in belay systems to add friction on moving rope, while the crossing hitch is designed to hold things in place and prevent movement",
     "reason_code": "replace_hitch_lookup_with_functional_distinction",
     "reason": "The edit replaces a name lookup with the source's functional distinction between two often-confused hitches."},
    {"fact_id": "fact-7c22eef0a56b8897d476", "source_path": raw("rope365_1a9f4f2fef392a9c.json"),
     "marker": "torii shibari 鳥居縛り", "decision": "keep",
     "reason_code": "translated_pose_term_context_complete",
     "reason": "The source directly pairs the Japanese term and characters with the named open-knees pose and torii-gate gloss."},
    {"fact_id": "fact-3a1c7e1cf8f26d0fc5ee", "source_path": raw("esinem_f2dfde25be14a7a8.json"),
     "marker": "it is an open question", "decision": "edit",
     "question": "What conclusion does the quoted researcher reach about when kinbaku gained its modern erotic meaning?",
     "answer": "it is an open question",
     "reason_code": "replace_novel_lookup_with_historical_uncertainty",
     "reason": "The edit replaces a title attached to an overbroad reading claim with the source's explicit uncertainty about the historical transition."},
    {"fact_id": "fact-959b57aac76574cb1c30", "source_path": raw("rope365_fc6af4b2e76d4cb4.json"),
     "marker": "keep the ropes on your muscles", "decision": "edit",
     "question": "What placement precaution does the source give after locating the common peroneal nerve near the knee?",
     "answer": "keep the ropes on your muscles, away from the boney structures around your knee caps",
     "reason_code": "replace_nerve_lookup_with_placement_precaution",
     "reason": "The edit makes the source's practical nerve-avoidance precaution retrievable instead of only asking for the nerve's name."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v15": 652,
    "active_after_this_tranche": 642,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 10,
    "new_edits_applied": 9,
    "output_rows": 680,
    "output_sha256": "41310a560facf9a827caa9f1a04e30d5038706d76d5c479b7aff8550123a9603",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 6,
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
        row["schema"] = "context-merit-audit-v16"
    write_jsonl(AUDIT, audits)
    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v16"
    report["audit"]["sha256"] = file_sha256(AUDIT)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
