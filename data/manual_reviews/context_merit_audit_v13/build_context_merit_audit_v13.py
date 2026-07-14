#!/usr/bin/env python3
"""Audit context-merit tranche v13 while keeping v1-v12 byte-frozen."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V12_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v12"
sys.path[:0] = [str(ROOT), str(V12_DIR)]
import build_context_merit_audit_v12 as previous
from qa_quality import normalize_text

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v13.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v13.jsonl"
REPORT = OUT_DIR / "report_context_merit_v13.json"
REVIEWER = "codex-context-merit-audit-v13"
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
    for version in range(1, 13)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
BASE = previous.previous
BASE_SOURCE_EVIDENCE = BASE.source_evidence


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-472b2a9e3213a841dc03", "source_path": raw("kinbakutoday_46508eda7d6c6203.json"),
     "marker": "Nine Gates of Osada-ryu", "decision": "drop", "reason_code": "volatile_or_promotional",
     "reason": "A proprietary name for unspecified verbally transmitted intangibles is not defined well enough to teach durable rope knowledge."},
    {"fact_id": "fact-c44bed0a3db7c0c73dbc", "source_path": raw("kinbakutoday_1775dd4176b24104.json"),
     "marker": "my own ignorance and my own incompetence", "decision": "keep", "reason_code": "source_attributed_safety_reflection",
     "reason": "The author explicitly attributes caused injuries to knowledge and competence gaps that education, practice, and disclosure can reduce."},
    {"fact_id": "fact-0a9bde892f63c70189ff", "source_path": raw("esinem_77368ccdc66acad0.json"),
     "marker": "Bondage & Discipline", "decision": "keep", "reason_code": "bdsm_term_definition_context_complete",
     "manual_evidence": "BD: Bondage & Discipline\nDs: Domination & submission\nSM: Sadism & Masochism",
     "reason": "The three expansions correctly unpack the paired meanings represented by the BDSM acronym."},
    {"fact_id": "fact-568e19d1c85f56b99ce1", "source_path": raw("rope365_8a4c32897b80339b.json"),
     "marker": "solid stick like a marlinspike", "decision": "keep", "reason_code": "emergency_kit_context_complete",
     "reason": "The tool is narrowly recommended for loosening hard-to-untie knots alongside on-body safety shears and incident planning."},
    {"fact_id": "fact-d3fea024ebdfcbb3f05d", "source_path": raw("kinbakutoday_3376e34b04fe0fb1.json"),
     "marker": "Jo-En: The World of Minomura Kou", "decision": "drop", "reason_code": "contextless_or_low_value",
     "reason": "The documentary-title lookup adds less durable value than the source's broader film-studio and artist history."},
    {"fact_id": "fact-e9eb8f51dc0b8c206656", "source_path": raw("esinem_a337d2fe39281aa2.json"),
     "marker": "how well you read your partner’s desires", "decision": "edit",
     "question": "Beyond learning patterns, what does Esinem say matters in shibari?",
     "answer": "how well you read your partner’s desires", "reason_code": "replace_book_lookup_with_partner_attunement",
     "reason": "The edit replaces another title quiz with the source's substantive point about reading a partner rather than tying mechanically."},
    {"fact_id": "fact-dc125508d925fcfc272b", "source_path": raw("rope365_682937f92222bf87.json"),
     "marker": "Violence, unsafe practices, and lack of consent", "decision": "keep", "reason_code": "bondage_history_caution_context_complete",
     "reason": "The answer concisely preserves the source's warning against romanticizing bondage history."},
    {"fact_id": "fact-e9f3b6fe52713d660e59", "source_path": raw("rope365_7fe5be31cb8dc67e.json"),
     "marker": "replace the half-moon with a full-moon friction", "decision": "drop", "reason_code": "unsafe_or_medically_unsupported",
     "reason": "A harness-locking modification for slippery rope lacks body placement, loading, monitoring, and release context."},
    {"fact_id": "fact-4471cc69e2bb67b1059c", "source_path": raw("kinbakutoday_d4dcb268cb41c5e4.json"),
     "marker": "how to distinguish seme from mere violence, vulgar appetite, or crude spectacle", "decision": "keep", "reason_code": "postwar_media_ethics_context_complete",
     "reason": "The source frames a substantive ethical and aesthetic problem in postwar Japanese SM publishing."},
    {"fact_id": "fact-5887674cdfee74c139d8", "source_path": raw("wikipedia_b67076372882395d.json"),
     "marker": "form of asphyxia which occurs when someone's position prevents the person from breathing adequately", "decision": "edit",
     "question": "What is positional, or postural, asphyxia?",
     "answer": "a form of asphyxia which occurs when someone's position prevents the person from breathing adequately",
     "reason_code": "replace_alias_repetition_with_breathing_definition",
     "reason": "The edit makes the safety-relevant breathing mechanism retrievable instead of asking for two names already present in the question."},
    {"fact_id": "fact-35a0f1a70111fab6fbff", "source_path": raw("kinbakutoday_c5e568667b495473.json"),
     "marker": "ashi o kuzushite ii yo", "decision": "keep", "reason_code": "partner_comfort_phrase_context_complete",
     "reason": "The phrase is translated and situated as permission to leave an uncomfortable formal kneeling position."},
    {"fact_id": "fact-c532ff0c7f1fa262440e", "source_path": raw("kinbakutoday_cbbff84b319d0813.json"),
     "marker": "how these older policing techniques, primarily designed for the male body, could be adapted to tying the female form", "decision": "edit",
     "question": "What question did Itoh Seiu’s 1953 article raise about older policing techniques?",
     "answer": "how these older policing techniques, primarily designed for the male body, could be adapted to tying the female form",
     "reason_code": "replace_article_title_with_historical_question",
     "reason": "The edit preserves the article's substantive historical transformation instead of only its title."},
    {"fact_id": "fact-7bda72c32de4c31716bc", "source_path": raw("esinem_d6aaa3b84555d86c.json"),
     "marker": "toxic and abusive relationship which does not reflect the consent and consideration due in true BDSM", "decision": "edit",
     "question": "Why does Esinem say Fifty Shades of Grey can give a wrong impression of BDSM?",
     "answer": "it depicts a toxic and abusive relationship which does not reflect the consent and consideration due in true BDSM",
     "reason_code": "replace_book_lookup_with_consent_distinction",
     "reason": "The edit makes the source's consent-and-abuse distinction the answer rather than testing the commercial series' title."},
    {"fact_id": "fact-222cfb0d001c49fdaf22", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["crash_restraint"], "evidence": "Crash Restraint: https://crash-restraint.com/", "decision": "keep",
     "reason_code": "owner_requested_resource_directory", "reason": "The owner explicitly requested that Crash Restraint remain memorized as the introductory suspension resource."},
    {"fact_id": "fact-9aad2d24020dfc169e35", "source_path": raw("wikipedia_914d249c3d7d542c.json"),
     "marker": "father of modern kinbaku", "decision": "keep", "reason_code": "kinbaku_history_title_context_complete",
     "reason": "The recognized title concisely identifies Ito Seiu's historical standing in modern kinbaku."},
    {"fact_id": "fact-e14d15f8a97e0a51371e", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["subspace_designs"], "evidence": "Subspace Designs: https://www.subspacedesigns.shop/", "decision": "keep",
     "reason_code": "owner_requested_resource_directory", "reason": "The owner explicitly requested that Subspace Designs remain memorized as the rigging-plate resource."},
    {"fact_id": "fact-47a176a9c9f1245c596f", "source_path": raw("rope365_773cf4d4be0e2895.json"),
     "marker": "do not press into the armpit or inside of the arms", "decision": "keep", "reason_code": "cinch_placement_safety_context_complete",
     "reason": "The conditional self-evaluation item gives a concrete pressure-placement precaution."},
    {"fact_id": "fact-ca3a4fcac4583ae3a5ca", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["my_nawashi", "de_giotto_rope"],
     "evidence": "My Nawashi: https://www.etsy.com/shop/MyNawashi; De Giotto Rope: https://degiottorope.com/", "decision": "keep",
     "reason_code": "owner_requested_resource_directory", "reason": "Both natural-fiber bondage-rope suppliers were explicitly supplied for durable resource lookup."},
    {"fact_id": "fact-63283786303c4df91008", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["rw_rope", "chromaknotz", "knothead_nylon"],
     "evidence": "RW Rope (synthetic upline rope): https://www.rwrope.com/; ChromaKnotz (synthetic bondage rope): https://chromaknotz.square.site/; Knot Head Nylon (synthetic bondage rope): https://knotheadnylon.net/", "decision": "keep",
     "reason_code": "owner_requested_resource_directory", "reason": "The owner explicitly requested that all three synthetic-rope supplier URLs remain memorized."},
    {"fact_id": "fact-2ed5cfecf9fd87c86405", "source_path": raw("wikipedia_a19eb7f4fbb670b4.json"),
     "marker": "self-rigging or self-suspension", "decision": "keep", "reason_code": "self_tying_term_context_complete",
     "reason": "The source directly defines the terms for a bondage rigger tying themself."},
    {"fact_id": "fact-74b67e379ff5d1c80099", "source_path": raw("rope365_c00bc803927bda26.json"),
     "marker": "adjustable shrimp tie, aka ebi shibari", "decision": "keep", "reason_code": "tie_alias_context_complete",
     "reason": "The named Japanese tie is directly paired with its descriptive English alias."},
    {"fact_id": "fact-b3bd38749af148c9d3ab", "source_path": raw("rope365_0c46b0f988c7a825.json"),
     "marker": "constantly check the hand motricity to monitor the nerves", "decision": "edit",
     "question": "What does Rope365 say should be constantly checked in a folded-arm chicken-wing tie?",
     "answer": "the hand motricity to monitor the nerves", "reason_code": "replace_tie_alias_with_nerve_monitoring",
     "reason": "The edit replaces another pose alias with the page's explicit upper-limb nerve-monitoring precaution."},
    {"fact_id": "fact-b959d354a761f5315fae", "source_path": raw("rope365_3d305a5499a4db8c.json"),
     "marker": "will hold tension once tied; it is not as solid as other locking hitches but also remains easy to untie", "decision": "edit",
     "question": "What tradeoff does Rope365 give for the reverse crossing hitch?",
     "answer": "The reverse crossing hitch will hold tension once tied; it is not as solid as other locking hitches but also remains easy to untie",
     "reason_code": "replace_hitch_alias_with_operational_tradeoff",
     "reason": "The edit preserves the source's hold, solidity, and release tradeoff rather than only a second name."},
    {"fact_id": "fact-cc0fc55bd4d815f700d2", "source_path": raw("rope365_6a23940f6bc37fc1.json"),
     "marker": "capsize under heavy loads or when using slippery rope", "decision": "edit",
     "question": "When can Rope365’s reversed French bowline still capsize?",
     "answer": "under heavy loads or when using slippery rope", "reason_code": "replace_knot_alias_with_capsize_warning",
     "reason": "The edit replaces an alias lookup with the source's important load-and-material limitation."},
    {"fact_id": "fact-d8c4b352c62b38c298e0", "source_path": raw("wikipedia_ea35c24ae8ca2151.json"),
     "marker": "Modern understanding distinguishes consensual BDSM practices from non-consensual sexual violence", "decision": "edit",
     "question": "What distinction does the source say modern understanding makes about sadomasochism?",
     "answer": "consensual BDSM practices from non-consensual sexual violence", "reason_code": "replace_author_etymology_with_consent_distinction",
     "reason": "The edit replaces general author trivia with a direct distinction between consensual BDSM and sexual violence."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v12": 672, "active_after_this_tranche": 669,
    "build_script": "build_curated_qa.py", "new_drops_applied": 3,
    "new_edits_applied": 8, "output_rows": 707,
    "output_sha256": "5d09acd6c5c0b5aa31de6d6ab51e405318ef11fc7db9b76639ab31624c1b5e39",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True, "reviewed_keep_fact_ids_preserved": 14,
    "sealed_eval_fact_count_reported_by_tooling": 612, "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def source_evidence(spec: dict, active: dict) -> tuple[str, str]:
    if "manual_evidence" in spec:
        document = json.loads(spec["source_path"].read_text())
        evidence = spec["manual_evidence"]
        if (document["url"] != active["url"] or
                text_sha256(document["text"]) != active["document_sha256"] or
                normalize_text(evidence) not in normalize_text(document["text"]) or
                not all(normalize_text(part) in normalize_text(evidence)
                        for part in active["answer"].split(", "))):
            raise ValueError(f'{active["fact_id"]}: multiline evidence drift')
        return evidence, "source_composite"
    if spec["source_path"] != RESOURCE_MANIFEST:
        return BASE_SOURCE_EVIDENCE(spec, active)
    manifest = json.loads(RESOURCE_MANIFEST.read_text())
    resources = {item["id"]: item for item in manifest["resources"]}
    selected = [resources[resource_id] for resource_id in spec["resource_ids"]]
    evidence = spec["evidence"]
    if (file_sha256(RESOURCE_MANIFEST) != active["document_sha256"] or
            active["answer"] != evidence or
            active["url"] != selected[0]["canonical_url"]):
        raise ValueError(f'{active["fact_id"]}: resource manifest drift')
    for resource in selected:
        if (resource["name"] not in evidence or
                resource["canonical_url"] not in evidence):
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
    original_evidence = BASE.source_evidence
    try:
        for name, value in replacements.items():
            setattr(previous, name, value)
        BASE.source_evidence = source_evidence
        yield
    finally:
        BASE.source_evidence = original_evidence
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
        row["schema"] = "context-merit-audit-v13"
    write_jsonl(AUDIT, audits)
    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v13"
    report["audit"]["sha256"] = file_sha256(AUDIT)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
