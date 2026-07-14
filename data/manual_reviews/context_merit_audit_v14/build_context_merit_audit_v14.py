#!/usr/bin/env python3
"""Audit context-merit tranche v14 while keeping v1-v13 byte-frozen."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V13_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v13"
sys.path[:0] = [str(ROOT), str(V13_DIR)]
import build_context_merit_audit_v13 as previous

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v14.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v14.jsonl"
REPORT = OUT_DIR / "report_context_merit_v14.json"
REVIEWER = "codex-context-merit-audit-v14"
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
    for version in range(1, 14)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-e3e697801b6d4cab437f", "source_path": raw("kinbakutoday_7d5a17ab0e59924f.json"),
     "marker": "tying starts with the heart and uses technique to express that", "decision": "edit",
     "question": "What contrast did Chiba Eizo draw between tying in Japan and in the West?",
     "answer": "he felt that in Japan, tying starts with the heart and uses technique to express that. In the West he said we start with the technique and eventually hope to find the heart",
     "reason_code": "replace_fragment_with_source_attributed_contrast",
     "reason": "The edit replaces a two-word fragment with the complete, explicitly attributed contrast in the source."},
    {"fact_id": "fact-fb98f437712003cfcb5e", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["ropecraft"], "evidence": "ROPECRAFT: https://ropecraft.net/", "decision": "keep",
     "reason_code": "owner_requested_resource_directory",
     "reason": "The owner explicitly requested that ROPECRAFT remain memorized as a national rope-convention resource."},
    {"fact_id": "fact-5b41f161d5e544be5df1", "source_path": raw("wikipedia_a97e19abfe49afa9.json"),
     "marker": "more than 25 traditional ties", "decision": "keep",
     "reason_code": "hojojutsu_reference_scope_context_complete",
     "reason": "The source directly states the scope of a named instructional reference on traditional hojojutsu ties."},
    {"fact_id": "fact-820fe500767c3a60b808", "source_path": raw("rope365_5c6e72d053d47d66.json"),
     "marker": "box tie position", "decision": "drop", "reason_code": "unsafe_or_context_incomplete",
     "reason": "A one-rope box-tie claim omits fit, anatomy, nerve-risk, loading, monitoring, and release context."},
    {"fact_id": "fact-89d30b473e2d3d1b2ff2", "source_path": raw("kinbakutoday_00507964d2e38f67.json"),
     "marker": "the black book", "decision": "keep", "reason_code": "kinbaku_history_reference_context_complete",
     "reason": "The source directly gives the established nickname of a named historical photo book."},
    {"fact_id": "fact-b477422fa24722f8c36c", "source_path": raw("rope365_f2d9e825760af158.json"),
     "marker": "very careful as we tie closer to the elbows", "decision": "edit",
     "question": "What caution does Rope365 give when adding arm cuffs to a box-tie structure?",
     "answer": "We have to be very careful as we tie closer to the elbows",
     "reason_code": "replace_person_attribution_with_elbow_caution",
     "reason": "The edit replaces person attribution with the page's explicit placement caution near the elbows."},
    {"fact_id": "fact-38823ae7deffa824a4d9", "source_path": raw("kinbakutoday_eeada73647c2bd19.json"),
     "marker": "heart and spirit mattered more than the techniques you used", "decision": "edit",
     "question": "Why does the source describe Yukimura Haruki as an heir to Minomura’s style of rope?",
     "answer": "he came from a world of rope where heart and spirit mattered more than the techniques you used",
     "reason_code": "replace_style_lookup_with_explained_lineage",
     "reason": "The edit preserves the source's substantive explanation of the claimed lineage instead of only naming it."},
    {"fact_id": "fact-5dd070f289170de1736a", "source_path": raw("kinbakutoday_aeb5f266584c30d1.json"),
     "marker": "The Indecent Rope Kamasutra", "decision": "keep", "reason_code": "translated_historical_title_context_complete",
     "reason": "The source directly explains the literal translation and wordplay of a named historical title."},
    {"fact_id": "fact-752f9ab76a82f2cd45f7", "source_path": raw("kinbakutoday_f8d47d490455d57a.json"),
     "marker": "range of communication, expression, and connection is unlimited", "decision": "edit",
     "question": "Why did Yukimura say floorwork allows more connection than suspension?",
     "answer": "When you play on the floor, the range of communication, expression, and connection is unlimited",
     "reason_code": "replace_style_lookup_with_connection_rationale",
     "reason": "The edit replaces a style label with Yukimura's stated rationale about communication and connection."},
    {"fact_id": "fact-7ede9ac62dca815ace93", "source_path": raw("rope365_8a4c32897b80339b.json"),
     "marker": "important to discuss your risk profile ahead of tying", "decision": "edit",
     "question": "What does Rope365 say should be discussed before tying when rope marks could be problematic?",
     "answer": "it is important to discuss your risk profile ahead of tying",
     "reason_code": "replace_location_recall_with_pre_scene_discussion",
     "reason": "The edit turns a body-location list into the source's actionable pre-scene risk discussion."},
    {"fact_id": "fact-d05649b6636e5ae0770a", "source_path": raw("kinbakutoday_df9c70212a927199.json"),
     "marker": "hazukashii doesn’t carry with it some of the darker connotations", "decision": "edit",
     "question": "How does the source distinguish hazukashii in Yukimura-style rope from harsher Western humiliation?",
     "answer": "hazukashii doesn’t carry with it some of the darker connotations that we think of as humiliation in the West, such as degradation, ridicule, or even cruelty",
     "reason_code": "replace_style_lookup_with_emotional_distinction",
     "reason": "The edit teaches the source's careful emotional distinction instead of asking which style uses the term."},
    {"fact_id": "fact-2434e4e894af5c3b05f7", "source_path": raw("rope365_5a4dc6c711aaa005.json"),
     "marker": "looser lay rope", "decision": "keep", "reason_code": "rope_maintenance_context_complete",
     "reason": "The source directly states the rope construction on which the specified maintenance problem is easier to fix."},
    {"fact_id": "fact-2a1bf6a6cce932c51390", "source_path": raw("rope365_5fdb5e78c2471772.json"),
     "marker": "Twisted or braided", "decision": "keep", "reason_code": "rope_construction_fundamentals_context_complete",
     "reason": "The answer identifies the two foundational rope-construction types in the source's inspection exercise."},
    {"fact_id": "fact-575ef3a6bf4535b20142", "source_path": raw("esinem_64c4212a11867d61.json"),
     "marker": "TK or box-tie", "decision": "keep", "reason_code": "rope_terminology_alias_context_complete",
     "reason": "The source directly pairs gote with the common TK and box-tie aliases."},
    {"fact_id": "fact-8b2e77f484617e89490f", "source_path": raw("esinem_f81cb7dad6da7191.json"),
     "marker": "the ‘devil tie’", "decision": "drop", "reason_code": "volatile_or_promotional",
     "reason": "A promotional nickname for a hazardous arm position adds little durable value and lacks practical safety context."},
    {"fact_id": "fact-e866abe07a39ba58e2f5", "source_path": raw("kinbakutoday_20d1161d0b76162f.json"),
     "marker": "Bayū horse oil", "decision": "drop", "reason_code": "anecdotal_or_overgeneralized",
     "reason": "One interviewee's personal jute-processing choice should not be learned as general rope-care guidance."},
    {"fact_id": "fact-ea1217bbec2888215c53", "source_path": raw("esinem_7b67baa6649b24c5.json"),
     "marker": "The ‘no-pinch cinch’", "decision": "drop", "reason_code": "volatile_or_promotional",
     "reason": "A tutorial topic-list lookup is promotional, contextless, and does not teach how to evaluate pressure or fit."},
    {"fact_id": "fact-62163ba3520628ad592e", "source_path": raw("rope365_d37244d61d012d8b.json"),
     "marker": "ashiura awase shibari 足裏合わせ縛り", "decision": "keep", "reason_code": "translated_pose_alias_context_complete",
     "reason": "The source directly gives the Japanese name and characters for a named soles-together pose."},
    {"fact_id": "fact-7ce4e8e3d6027761a09d", "source_path": raw("rope365_f8805cf88e36205d.json"),
     "marker": "offset the first knot", "decision": "drop", "reason_code": "unsafe_or_context_incomplete",
     "reason": "An isolated harness-starting microstep lacks placement, loading, monitoring, and release context."},
    {"fact_id": "fact-8cafd5b7290db95a430a", "source_path": raw("kinbakutoday_40bb93d8b1902e71.json"),
     "marker": "erotic, humiliating, and sexual", "decision": "drop", "reason_code": "contextless_or_low_value",
     "reason": "A one-line characterization of a particular promotional video is subjective and not durable instructional knowledge."},
    {"fact_id": "fact-2c9a10e01992649a4655", "source_path": raw("rope365_0b4f44c8fabfc202.json"),
     "marker": "contain chemicals that are bad for the skin", "decision": "edit",
     "question": "Why does Rope365 say rope material choice can affect bondage safety?",
     "answer": "some can be unsafe to use for bondage as they contain chemicals that are bad for the skin",
     "reason_code": "replace_vague_material_list_with_skin_safety_warning",
     "reason": "The edit replaces a vague novelty claim with the source's explicit warning about skin-unsafe chemicals."},
    {"fact_id": "fact-46cf0a9daee2c52085aa", "source_path": raw("rope365_0a23b686a4f493e1.json"),
     "marker": "requires maintained tension in order to keep its grip", "decision": "edit",
     "question": "What limitation does Rope365 give for a half hitch?",
     "answer": "it requires maintained tension in order to keep its grip",
     "reason_code": "replace_knot_lookup_with_tension_limitation",
     "reason": "The edit replaces a general knot-name lookup with the source's important operational limitation."},
    {"fact_id": "fact-5348a63f92617a3fc4f1", "source_path": raw("rope365_d9f9d21430ff4bba.json"),
     "marker": "Edo period torture", "decision": "drop", "reason_code": "source_reliability_or_context_incomplete",
     "reason": "The page labels its underlying transcript as automated and potentially mistaken, making the isolated attribution too weak to retain."},
    {"fact_id": "fact-62d30642329bd659a877", "source_path": raw("kinbakutoday_43095aa305e35a41.json"),
     "marker": "can lead to a big accident", "decision": "edit",
     "question": "What safety warning does the article connect to the temptation to think a small problem is acceptable?",
     "answer": "The feeling of “this degree should be OK” can lead to a big accident",
     "reason_code": "replace_translation_lookup_with_safety_lesson",
     "reason": "The edit preserves the article's actionable warning against ignoring uneasiness instead of only translating an idiom."},
    {"fact_id": "fact-7a2364ae08d837da2d48", "source_path": raw("rope365_c706558d746d11de.json"),
     "marker": "Ichinawa and Ipponnawa", "decision": "keep", "reason_code": "rope_improvisation_terms_context_complete",
     "reason": "The source directly gives both Japanese terms used for its one-rope improvisation activity."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v13": 669,
    "active_after_this_tranche": 662,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 7,
    "new_edits_applied": 9,
    "output_rows": 700,
    "output_sha256": "9bfd45ef01cbaacddabb84ad92bfa45d854afa031ba93bfee796ff61b9906d97",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 9,
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
        row["schema"] = "context-merit-audit-v14"
    write_jsonl(AUDIT, audits)
    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v14"
    report["audit"]["sha256"] = file_sha256(AUDIT)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
