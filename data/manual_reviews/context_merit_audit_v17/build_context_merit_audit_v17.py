#!/usr/bin/env python3
"""Audit context-merit tranche v17 while keeping v1-v16 byte-frozen."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V16_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v16"
sys.path[:0] = [str(ROOT), str(V16_DIR)]
import build_context_merit_audit_v16 as previous

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v17.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v17.jsonl"
REPORT = OUT_DIR / "report_context_merit_v17.json"
REVIEWER = "codex-context-merit-audit-v17"
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
    for version in range(1, 17)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
SPECIAL = previous.previous.previous.previous
SPECIAL_SOURCE_EVIDENCE = SPECIAL.source_evidence


def raw(name: str) -> Path:
    return DATA / "raw" / name


def resource(fact_id: str, resource_ids: list[str], evidence: str,
             reason: str) -> dict:
    return {
        "fact_id": fact_id, "source_path": RESOURCE_MANIFEST,
        "resource_ids": resource_ids, "evidence": evidence,
        "decision": "keep", "reason_code": "owner_requested_resource_directory",
        "reason": reason,
    }


SPECS = (
    resource("fact-68b0c2a8c5e4c12eb2af", ["shibari_supply_bamboo"],
             "Shibari Supply: https://shibarisupply.com/",
             "The owner explicitly requested that Shibari Supply remain memorized as the suspension-grade bamboo resource."),
    resource("fact-d45ad2b3ebf26fc99369", ["theduchy"],
             "The Duchy: https://www.theduchy.com/",
             "The owner explicitly requested that The Duchy remain memorized for intermediate floor-based rope lessons."),
    resource("fact-d9349b944ec02a40316b", ["xpole_a_frame"],
             "X-POLE A-FRAME: https://xpoleus.com/shop-all/aerial/a-frame/xpole-a-frame/",
             "The owner explicitly requested that the X-POLE A-FRAME link remain memorized as a semi-permanent frame resource."),
    resource("fact-e6435c62d16bab5ba44f", ["chromaknotz"],
             "ChromaKnotz: https://chromaknotz.square.site/",
             "The owner explicitly requested that ChromaKnotz remain memorized as a synthetic bondage-rope supplier."),
    {"fact_id": "fact-162dece22cec751a9eb0", "source_path": raw("rope365_603ebb6ae5af91f0.json"),
     "marker": "Bight – U-shaped arc of rope", "decision": "keep",
     "reason_code": "rope_vocabulary_context_complete",
     "reason": "The source directly defines a foundational rope term used throughout the training material."},
    resource("fact-6dc2d6b921d3d72b3d33", ["emt_shears"],
             "EMT Shears: https://www.amazon.com/dp/B0195QI218",
             "The owner explicitly requested that the supplied EMT-shears link remain memorized as emergency equipment."),
    {"fact_id": "fact-f281961568d3138d6480", "source_path": raw("kinbakutoday_454370c8a4f42708.json"),
     "marker": "took all of his works and burnt them", "decision": "drop",
     "reason_code": "anecdotal_or_overgeneralized",
     "reason": "The interviewee explicitly frames this account as something heard from others, while a later passage only says the destruction is believed."},
    resource("fact-d8460d7680f1fee4ee02", ["tethered_together", "ropecraft"],
             "Tethered Together: https://tetheredtogether.net/; ROPECRAFT: https://ropecraft.net/",
             "The owner explicitly requested that both U.S. national rope-convention websites remain memorized."),
    {"fact_id": "fact-ca976547430364e279f7", "source_path": raw("rope365_62f7e527bb35b47d.json"),
     "marker": "2-ply jute, or a 1-ply that is tightly laid", "decision": "keep",
     "reason_code": "source_attributed_beginner_material_guidance",
     "reason": "The recommendation is explicitly limited to beginners choosing jute and contrasts manageable construction with loose-lay rope requiring more finesse."},
    {"fact_id": "fact-d0ef49d5f4d72d637ebe", "source_path": raw("kinbakutoday_381214111022a952.json"),
     "marker": "a fashion, a way, a style, manner", "decision": "keep",
     "reason_code": "japanese_suffix_definition_context_complete",
     "reason": "The source directly defines the Japanese suffix in the context of an individual style or school of thought."},
    {"fact_id": "fact-d4132da7aa39ec19ca12", "source_path": raw("rope365_d41bd680406d8e96.json"),
     "marker": "pull the leg in different directions in floorwork", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete",
     "reason": "A foot-harness manipulation purpose lacks load, joint, nerve, circulation, monitoring, and emergency-release context."},
    resource("fact-185791df4abf9480c370", ["atx_empty_space", "austin_rope_slingers", "bight_bound"],
             "The Empty Space (an Austin rope-community event calendar): https://www.atxempty.space/space-schedule; Austin Rope Slingers (an Austin rope group with regular meetings): https://www.austinropeslingers.com/; BightBound (Austin-area educators for intermediate and advanced rope topics): https://www.bightbound.com/",
             "The owner explicitly requested that all three Austin community resources and their purposes remain memorized."),
    {"fact_id": "fact-01afc7260575c4f173f0", "source_path": raw("kinbakutoday_df9c70212a927199.json"),
     "marker": "a feeling of emotional nakedness", "decision": "keep",
     "reason_code": "hazukashii_definition_context_complete",
     "reason": "The source's concise metaphor complements the retained distinction between hazukashii and degradation by emphasizing emotional visibility and vulnerability."},
    resource("fact-f37fe09cfbc59840bb8d", ["rw_rope"],
             "RW Rope: https://www.rwrope.com/",
             "The owner explicitly requested that RW Rope remain memorized as the synthetic-upline supplier, without crawling unrelated catalog pages."),
    {"fact_id": "fact-1263411b5e120944bb5b", "source_path": raw("esinem_4cdf0843c58d3796.json"),
     "marker": "Yukimura Haruki, Nureki Chimuo or Marai Masato", "decision": "drop",
     "reason_code": "volatile_or_promotional",
     "reason": "A stylistic name list in a commercial vertical-suspension tutorial announcement is subjective and lacks durable explanatory value."},
    {"fact_id": "fact-9ad07ddb01e5389309a3", "source_path": raw("kinbakutoday_f6ccdaa49bed3fa5.json"),
     "marker": "society for the study of beautiful bondage", "decision": "keep",
     "reason_code": "kinbaku_history_term_context_complete",
     "reason": "The source directly translates a historically named study group and situates its continuation and lineage."},
    resource("fact-2360512946a94178f61d", ["hip_harness_playlist"],
             "Hip Harness Videos: https://www.youtube.com/playlist?list=PLkrdRffh_Gg2S9QccbRyiLE5x4SIacgoM",
             "The owner explicitly requested that this exact hip-harness video playlist remain memorized."),
    resource("fact-1cd85d3b5453269dc5d6", ["deep_dive_single_columns"],
             "Deep Dive into Single Columns: https://www.youtube.com/watch?v=vnDvjAaQU8g",
             "The owner explicitly requested that this exact deep-dive video on single-column ties remain memorized."),
    {"fact_id": "fact-9989c5de84c0045597c9", "source_path": raw("esinem_2b762f05bc1bf364.json"),
     "marker": "Tying beautifully and intensely", "decision": "drop",
     "reason_code": "volatile_or_promotional",
     "reason": "An obscure stylistic slogan is taken from a dated masterclass promotion and is not independently explained or contextualized."},
    {"fact_id": "fact-050c23dc5936b4bcb93e", "source_path": raw("rope365_1616ffce57d993f3.json"),
     "marker": "inspired by Akechi Denki and Nawashi Kanna", "decision": "drop",
     "reason_code": "contextless_or_low_value",
     "reason": "Person-attribution trivia for a restrictive box-tie design adds less value than the same page's retained nerve and extension-placement precautions."},
    {"fact_id": "fact-452f36d13d2af7da6837", "source_path": raw("wikipedia_2151448295a2af9b.json"),
     "marker": "traffic light system (TLS)", "decision": "keep",
     "reason_code": "safeword_framework_context_complete",
     "reason": "The source directly names the widely used safeword framework and immediately defines its three signals."},
    resource("fact-49a87d11b1076c90e07c", ["bight_bound"],
             "BightBound: https://www.bightbound.com/",
             "The owner explicitly requested that BightBound remain memorized for intermediate and advanced Austin-area rope education."),
    {"fact_id": "fact-d1e1db177afb183343ad", "source_path": raw("rope365_6812791b49f1f1a2.json"),
     "marker": "Remedial Ropes by Shay Tiziano", "decision": "keep",
     "reason_code": "rope_safety_resource_context_complete",
     "reason": "The source explicitly identifies Remedial Ropes as a website dedicated to rope-bondage safety."},
    {"fact_id": "fact-3c9a23eab0f09a85e3b8", "source_path": raw("kinbakutoday_fd81f3bdea43d0ae.json"),
     "marker": "heart (kokoro), connection (kankei), and communication", "decision": "keep",
     "reason_code": "source_attributed_sm_principle",
     "reason": "The answer concisely preserves Chiba Eizo's explicitly attributed description of SM's relational essence."},
    {"fact_id": "fact-5bf2058daea8746fc3e2", "source_path": raw("rope365_1616ffce57d993f3.json"),
     "marker": "between the arm and the torso", "decision": "keep",
     "reason_code": "rope_extension_nerve_precaution_context_complete",
     "reason": "The source explicitly warns against placing a bulky rope extension where many nerves are exposed near the armpit."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v16": 642,
    "active_after_this_tranche": 637,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 5,
    "new_edits_applied": 0,
    "output_rows": 675,
    "output_sha256": "d2ed2de0c864044931042d0a3e3aca12b9516fc67d7ea35f4964169b2eb67e3a",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 20,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def source_evidence(spec: dict, active: dict) -> tuple[str, str]:
    if spec["source_path"] != RESOURCE_MANIFEST:
        return SPECIAL_SOURCE_EVIDENCE(spec, active)
    manifest = json.loads(RESOURCE_MANIFEST.read_text())
    resources = {item["id"]: item for item in manifest["resources"]}
    selected = [resources[resource_id] for resource_id in spec["resource_ids"]]
    evidence = spec["evidence"]
    primary_url = (selected[0].get("recommendation_url") or
                   selected[0]["canonical_url"])
    if (file_sha256(RESOURCE_MANIFEST) != active["document_sha256"] or
            active["answer"] != evidence or active["url"] != primary_url):
        raise ValueError(f'{active["fact_id"]}: resource manifest drift')
    for item in selected:
        url = item.get("recommendation_url") or item["canonical_url"]
        if item["name"] not in evidence or url not in evidence:
            raise ValueError(f'{active["fact_id"]}: unsupported resource composite')
    return evidence, "manifest_composite"


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
    original_evidence = SPECIAL.source_evidence
    try:
        for name, value in replacements.items():
            setattr(previous, name, value)
        SPECIAL.source_evidence = source_evidence
        yield
    finally:
        SPECIAL.source_evidence = original_evidence
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
        row["schema"] = "context-merit-audit-v17"
    write_jsonl(AUDIT, audits)
    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v17"
    report["audit"]["sha256"] = file_sha256(AUDIT)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
