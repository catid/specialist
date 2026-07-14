#!/usr/bin/env python3
"""Deterministically re-audit the next weak-context surviving keeps in v29."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V28_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v28"
sys.path[:0] = [str(ROOT), str(V28_DIR)]
import build_context_merit_audit_v28 as previous

BASE = previous.BASE
CORE = previous.CORE

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v29.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v29.jsonl"
REPORT = OUT_DIR / "report_context_merit_v29.json"
REVIEWER = "codex-context-merit-audit-v29"
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
    for version in range(1, 29)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 29)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


def resource(fact_id: str, resource_ids: list[str], evidence: str,
             reason: str) -> dict:
    return {
        "fact_id": fact_id,
        "source_path": RESOURCE_MANIFEST,
        "resource_ids": resource_ids,
        "evidence": evidence,
        "decision": "keep",
        "reason_code": "owner_requested_resource_directory",
        "reason": reason,
    }


SPECS = (
    {"fact_id": "fact-ba692623a8506581339b",
     "source_path": raw("rope365_25f1b23eb40be00e.json"),
     "marker": "quick slip knot is a good way to mark the middle of the rope",
     "decision": "keep",
     "reason_code": "rope_handling_utility_context_complete",
     "reason": "The answer is a non-body rope-handling use, and the surrounding source explicitly warns that slip knots are unstable."},
    {"fact_id": "fact-bf91b4f8c612c46d312c",
     "source_path": raw("rope365_02b5a435a4822809.json"),
     "marker": "Every person is different, a spot that tickles for someone",
     "decision": "edit",
     "question": "What does Rope365 emphasize about communication and consent in pressure-point play?",
     "answer": "Communication is key and consent is particularly important with this kind of play.",
     "reason_code": "replace_touch_term_with_individual_consent_context",
     "reason": "The edit replaces an awkward tactile-acuity term lookup with the source's more useful communication and consent guidance."},
    {"fact_id": "fact-7c22eef0a56b8897d476",
     "source_path": raw("rope365_1a9f4f2fef392a9c.json"),
     "marker": "torii shibari 鳥居縛り",
     "decision": "drop",
     "reason_code": "duplicate_high_risk_pose_alias",
     "reason": "The translated open-knees pose alias adds no safety context, and a more useful discussion-first vulnerability lesson from the same source already survives the projection."},
    resource(
        "fact-68b0c2a8c5e4c12eb2af", ["shibari_supply_bamboo"],
        "Shibari Supply: https://shibarisupply.com/",
        "The project owner explicitly requested that Shibari Supply remain memorized as the suspension-grade bamboo resource."),
    resource(
        "fact-d45ad2b3ebf26fc99369", ["theduchy"],
        "The Duchy: https://www.theduchy.com/",
        "The project owner explicitly requested that The Duchy remain memorized for intermediate floor-based rope lessons."),
    resource(
        "fact-d9349b944ec02a40316b", ["xpole_a_frame"],
        "X-POLE A-FRAME: https://xpoleus.com/shop-all/aerial/a-frame/xpole-a-frame/",
        "The project owner explicitly requested that the X-POLE A-FRAME link remain memorized as a semi-permanent frame resource."),
    resource(
        "fact-e6435c62d16bab5ba44f", ["chromaknotz"],
        "ChromaKnotz: https://chromaknotz.square.site/",
        "The project owner explicitly requested that ChromaKnotz remain memorized as a synthetic bondage-rope supplier."),
    {"fact_id": "fact-162dece22cec751a9eb0",
     "source_path": raw("rope365_603ebb6ae5af91f0.json"),
     "marker": "Bight – U-shaped arc of rope, usually the middle",
     "decision": "keep",
     "reason_code": "rope_vocabulary_context_complete",
     "reason": "The source directly defines a foundational rope term used throughout the training material."},
    resource(
        "fact-6dc2d6b921d3d72b3d33", ["emt_shears"],
        "EMT Shears: https://www.amazon.com/dp/B0195QI218",
        "The project owner explicitly requested that the supplied EMT-shears link remain memorized as emergency equipment."),
    resource(
        "fact-d8460d7680f1fee4ee02", ["tethered_together", "ropecraft"],
        "Tethered Together: https://tetheredtogether.net/; ROPECRAFT: https://ropecraft.net/",
        "The project owner explicitly requested that both U.S. national rope-convention websites remain memorized."),
    {"fact_id": "fact-ca976547430364e279f7",
     "source_path": raw("rope365_62f7e527bb35b47d.json"),
     "marker": "For a beginner especially, I recommend a 2-ply jute",
     "decision": "edit",
     "question": "What jute construction does Knis recommend for a beginner, and why not loose-lay one-ply jute?",
     "answer": "For a beginner especially, I recommend a 2-ply jute, or a 1-ply that is tightly laid. Loose-lay 1-ply jute is a favorite for the experienced, but it takes finesse.",
     "reason_code": "add_attribution_and_material_rationale",
     "reason": "The edit makes the recommendation explicitly source-attributed and retains the author's reason that loose-lay one-ply jute requires more finesse."},
    {"fact_id": "fact-d0ef49d5f4d72d637ebe",
     "source_path": raw("kinbakutoday_381214111022a952.json"),
     "marker": "Ryū. A noun used as a suffix",
     "decision": "keep",
     "reason_code": "japanese_suffix_definition_context_complete",
     "reason": "The source directly defines the Japanese suffix in the context of an individual style or school of thought."},
    resource(
        "fact-185791df4abf9480c370",
        ["atx_empty_space", "austin_rope_slingers", "bight_bound"],
        "The Empty Space (an Austin rope-community event calendar): https://www.atxempty.space/space-schedule; Austin Rope Slingers (an Austin rope group with regular meetings): https://www.austinropeslingers.com/; BightBound (Austin-area educators for intermediate and advanced rope topics): https://www.bightbound.com/",
        "The project owner explicitly requested that all three Austin community resources and their purposes remain memorized."),
    {"fact_id": "fact-01afc7260575c4f173f0",
     "source_path": raw("kinbakutoday_df9c70212a927199.json"),
     "marker": "Hazukashii is fundamentally an internal effect",
     "decision": "edit",
     "question": "How does the article distinguish hazukashii from humiliation in Yukimura-style rope?",
     "answer": "Hazukashii is fundamentally an internal effect, which emerges from the emotional response of a person being tied, where humiliation is the result of an external act that is imposed on the person being humiliated.",
     "reason_code": "replace_metaphor_with_cultural_distinction",
     "reason": "The edit replaces a bare metaphor with the article's explicit distinction between an internally emerging emotional response and an externally imposed act."},
    resource(
        "fact-f37fe09cfbc59840bb8d", ["rw_rope"],
        "RW Rope: https://www.rwrope.com/",
        "The project owner explicitly requested that RW Rope remain memorized as the synthetic-upline supplier, without crawling unrelated catalog pages."),
    {"fact_id": "fact-9ad07ddb01e5389309a3",
     "source_path": raw("kinbakutoday_f6ccdaa49bed3fa5.json"),
     "marker": "continuation of a series of gatherings started in 1986",
     "decision": "edit",
     "question": "When did the series of gatherings continued by Kinbiken begin, and who led it?",
     "answer": "The meeting is a continuation of a series of gatherings started in 1986, led by Nureki Chimuo, Naka sensei’s teacher and mentor.",
     "reason_code": "replace_translation_lookup_with_historical_lineage",
     "reason": "The edit replaces a translation lookup with the source's more substantive date and leadership lineage for the gatherings."},
    resource(
        "fact-2360512946a94178f61d", ["hip_harness_playlist"],
        "Hip Harness Videos: https://www.youtube.com/playlist?list=PLkrdRffh_Gg2S9QccbRyiLE5x4SIacgoM",
        "The project owner explicitly requested that this exact hip-harness video playlist remain memorized."),
    resource(
        "fact-1cd85d3b5453269dc5d6", ["deep_dive_single_columns"],
        "Deep Dive into Single Columns: https://www.youtube.com/watch?v=vnDvjAaQU8g",
        "The project owner explicitly requested that this exact deep-dive video on single-column ties remain memorized."),
    {"fact_id": "fact-452f36d13d2af7da6837",
     "source_path": raw("wikipedia_2151448295a2af9b.json"),
     "marker": "Red – meaning: stop immediately and check the status",
     "manual_evidence": "The traffic light system (TLS) is the most commonly used set of safewords.\n\nRed – meaning: stop immediately and check the status of your partner\nYellow – meaning: slow down, be careful\nGreen – meaning: I'm all good, we can start. If used it's normally uttered by everyone involved before the scene can start.",
     "decision": "edit",
     "question": "What do red, yellow, and green mean in the traffic-light safeword system?",
     "answer": "Red means stop immediately and check the partner’s status; yellow means slow down and be careful; green means everyone is okay to start.",
     "reason_code": "replace_safeword_name_with_operational_meanings",
     "reason": "The edit replaces a framework-name lookup with the operational meaning of all three signals.",
     "support_type": "manual_paraphrase",
     "paraphrase_rationale": "Converts the source's list formatting and first-person green signal into one concise, faithful sentence without changing the stop, caution, or ready meanings."},
    resource(
        "fact-49a87d11b1076c90e07c", ["bight_bound"],
        "BightBound: https://www.bightbound.com/",
        "The project owner explicitly requested that BightBound remain memorized for intermediate and advanced Austin-area rope education."),
    {"fact_id": "fact-d1e1db177afb183343ad",
     "source_path": raw("rope365_6812791b49f1f1a2.json"),
     "marker": "Remedial Ropes by Shay Tiziano",
     "decision": "drop",
     "reason_code": "volatile_secondary_resource_listing",
     "reason": "This is a dated secondary directory entry rather than an owner-requested resource, and it provides no URL, safety guidance, or durable substance of its own."},
    {"fact_id": "fact-3c9a23eab0f09a85e3b8",
     "source_path": raw("kinbakutoday_fd81f3bdea43d0ae.json"),
     "marker": "heart (kokoro), connection (kankei), and communication",
     "decision": "drop",
     "reason_code": "unsafe_coercive_surrounding_context",
     "reason": "The benign relational slogan is embedded in an article that repeatedly frames overcoming refusal as a route to acceptance, making it unsafe to retain without a consent critique the source does not supply."},
    {"fact_id": "fact-5bf2058daea8746fc3e2",
     "source_path": raw("rope365_1616ffce57d993f3.json"),
     "marker": "Extend the rope between the arm and the torso",
     "decision": "keep",
     "reason_code": "rope_extension_nerve_precaution_context_complete",
     "reason": "The source explicitly warns against placing a bulky rope extension where many nerves are exposed near the armpit."},
    {"fact_id": "fact-a6e93ed860c45d0b353d",
     "source_path": raw("esinem_64c4212a11867d61.json"),
     "marker": "basic building block of shibari",
     "decision": "drop",
     "reason_code": "duplicate_promotional_foundation_claim",
     "reason": "This generic foundation claim comes from a tutorial promotion, while v28 already retained the same source's more useful explanation of why the opening tie matters."},
    {"fact_id": "fact-d7f5bb80b07cc9a258da",
     "source_path": raw("esinem_b249eddc1f5e1864.json"),
     "marker": "method of flicking the rope usually does the trick",
     "decision": "edit",
     "question": "What does Esinem say usually works for a rope tangle encountered while tying?",
     "answer": "the method of flicking the rope usually does the trick",
     "reason_code": "replace_tangle_cause_with_practical_remedy",
     "reason": "The edit replaces recall of a tangle's cause with the source's concise, directly actionable remedy for a tangle encountered while tying."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(EXPECTED_SELECTION, (
        351, 257, 266, 502, 439, 256, 440, 232, 254, 535,
        406, 201, 501, 349, 255, 233, 444, 448, 242, 435,
        534, 381, 456, 251, 248,
    ))
}
SECONDARY_PRIOR_VERSIONS = {
    fact_id: (16 if index < 3 else 17 if index < 23 else 18)
    for index, fact_id in enumerate(EXPECTED_SELECTION)
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v28",
    "eligible_prior_keeps_before_reaudit_exclusion": 255,
    "secondary_eligible_prior_keeps": 55,
    "rows": 591,
    "sha256": "790549d4a1a9f65c7538ea50e6eb6f329b5bd6ae429cd4cac12cf38bee8e2b6e",
    "v21_v28_reviewed_fact_ids_excluded": 200,
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v28": 553,
    "active_after_this_tranche": 549,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 4,
    "new_edits_applied": 6,
    "output_rows": 587,
    "output_sha256": "e05be81d4c4e2cc9038c9225cbb4372b2bbc627cb4ba219c4dc93c14ad87ba13",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 15,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def prior_decision_artifacts() -> tuple[Path, ...]:
    paths = []
    for version in range(1, 29):
        directory = DATA / "manual_reviews" / f"context_merit_audit_v{version}"
        paths.extend((
            directory / f"context_merit_audit_v{version}.jsonl",
            directory / f"pending_curation_context_merit_v{version}.jsonl",
            directory / f"report_context_merit_v{version}.json",
        ))
    return tuple(paths)


def secondary_ranked(rows: list[dict]) -> list[dict]:
    by_id = {row["fact_id"]: (index, row)
             for index, row in enumerate(rows, 1)}
    occurrences: dict[str, int] = {}
    prior_keeps: dict[str, int] = {}
    for version, path in enumerate(CONTEXT_AUDITS[:20], 1):
        for audit in read_jsonl(path):
            fact_id = audit["fact_id"]
            occurrences[fact_id] = occurrences.get(fact_id, 0) + 1
            if audit["decision"] == "keep":
                prior_keeps[fact_id] = version
    rereviewed = {
        row["fact_id"] for path in CONTEXT_AUDITS[20:]
        for row in read_jsonl(path)
    }
    candidates = []
    for fact_id, version in prior_keeps.items():
        if (occurrences[fact_id] != 1 or fact_id not in by_id or
                fact_id in rereviewed):
            continue
        index, row = by_id[fact_id]
        features = CORE.risk_features(row)
        candidates.append((
            -features["risk_score"], features["question_tokens"],
            features["answer_tokens"], fact_id, index, row, features, version,
        ))
    candidates.sort(key=lambda item: item[:4])
    ranked = [{
        "active_index": item[4], "row": item[5], "features": item[6],
        "prior_version": item[7],
    } for item in candidates]
    if len(ranked) != 55:
        raise ValueError(f"v29 secondary candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:25]) != EXPECTED_SELECTION:
        raise ValueError("v29 secondary selection drift")
    return ranked


def selected_ranked(rows: list[dict]) -> tuple[list[dict], int, int]:
    return secondary_ranked(rows), 0, 0


@contextlib.contextmanager
def patched_base():
    replacements = {
        "OUT_DIR": OUT_DIR, "AUDIT": AUDIT, "CURATION": CURATION,
        "REPORT": REPORT, "REVIEWER": REVIEWER,
        "REVIEWED_AT": REVIEWED_AT,
        "CONTEXT_CURATIONS": CONTEXT_CURATIONS, "SPECS": SPECS,
        "EXPECTED_SELECTION": EXPECTED_SELECTION,
        "ISOLATED_PROJECTION": ISOLATED_PROJECTION,
    }
    originals = {name: getattr(BASE, name) for name in replacements}
    original_ranking = CORE.ranked_unreviewed
    try:
        for name, value in replacements.items():
            setattr(BASE, name, value)
        CORE.ranked_unreviewed = selected_ranked
        yield
    finally:
        CORE.ranked_unreviewed = original_ranking
        for name, value in originals.items():
            setattr(BASE, name, value)


def ranked_unreviewed(rows: list[dict]) -> tuple[list[dict], int, int]:
    with patched_base():
        return BASE.ranked_unreviewed(rows)


def main() -> None:
    with patched_base():
        BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        fact_id = row["fact_id"]
        version = SECONDARY_PRIOR_VERSIONS[fact_id]
        prior_path = (DATA / "manual_reviews" /
                      f"context_merit_audit_v{version}" /
                      f"context_merit_audit_v{version}.jsonl")
        row["schema"] = "context-merit-audit-v29"
        row["active_index"] = PROJECTED_ACTIVE_INDICES[fact_id]
        row["review_pass"] = "secondary_prior_keep_reaudit"
        row["prior_review"] = {
            "decision": "keep",
            "path": str(prior_path.relative_to(ROOT)),
            "sha256": file_sha256(prior_path),
            "version": version,
        }
    write_jsonl(AUDIT, audits)

    curations = read_jsonl(CURATION)
    paraphrase = "fact-452f36d13d2af7da6837"
    for row in curations:
        if row["fact_id"] == paraphrase:
            row["support_type"] = "manual_paraphrase"
            row["paraphrase_rationale"] = next(
                spec["paraphrase_rationale"] for spec in SPECS
                if spec["fact_id"] == paraphrase)
    write_jsonl(CURATION, curations)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v29"
    report["selection"].update({
        "active_rows": 591,
        "eligible_unreviewed_rows": 0,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "primary_fact_ids": [],
        "secondary_fact_ids": list(EXPECTED_SELECTION),
        "secondary_eligible_prior_keeps": 55,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "secondary_ranking": {
            "candidate_rule": (
                "a sole v1-v20 context-merit decision is keep, the fact "
                "survives the v28 projection, and the fact was not reviewed "
                "again in v21 through v28"
            ),
            "score": (
                "short_question_points + 3*pronoun_count + "
                "bare_answer_points + named_person_trivia_points"
            ),
            "tie_break": (
                "risk_score descending, question tokens ascending, answer "
                "tokens ascending, fact_id ascending"
            ),
        },
    })
    report["audit"]["sha256"] = file_sha256(AUDIT)
    report["new_pending_curation"]["sha256"] = file_sha256(CURATION)
    report["new_pending_curation"]["edit_support_types"] = {
        "extractive": 5, "manual_paraphrase": 1,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
