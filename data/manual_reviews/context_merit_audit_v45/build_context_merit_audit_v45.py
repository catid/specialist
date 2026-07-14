#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v44 projection rows in v45."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V44_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v44"
sys.path[:0] = [str(ROOT), str(V44_DIR)]
import build_context_merit_audit_v44 as previous
from qa_quality import normalize_text

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = previous.EVIDENCE_PATCH_MODULE
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v45.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v45.jsonl"
REPORT = OUT_DIR / "report_context_merit_v45.json"
REVIEWER = "codex-context-merit-audit-v45"
REVIEWED_AT = "2026-07-14"
RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST
ACTIVE_DATASET = previous.ACTIVE_DATASET
ACTIVE_REPORT = previous.ACTIVE_REPORT
ACTIVE_CURATIONS = previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS = previous.PRIOR_PENDING_ADDITIONS
QUALITY_MERIT_CURATION = previous.QUALITY_MERIT_CURATION
TASUKI_CURATION = previous.TASUKI_CURATION
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl

CONTEXT_CURATIONS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"pending_curation_context_merit_v{version}.jsonl"
    for version in range(1, 45)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 45)
)


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-3a5c5a58043db9e5e509",
        "source_path": ROOT / "sources" / "manual_facts" / "resource_group_b.jsonl",
        "marker": "The Tetruss frame is a quality crafted, light-weight aluminum alloy.",
        "decision": "keep",
        "reason_code": "owner_requested_manufacturer_frame_material",
        "reason": (
            "The manufacturer-attributed material answer is useful when evaluating "
            "the owner-requested portable suspension-frame resource."
        ),
    },
    {
        "fact_id": "fact-b75b3b1be438caa0831b",
        "source_path": raw("kinbakutoday_b6ca615acc15e5f5.json"),
        "marker": "Kasumi san had brought a bamboo spar",
        "decision": "drop",
        "reason_code": "incidental_exhibition_setup_material",
        "reason": (
            "The bamboo spar is an incidental detail from one exhibition photo "
            "shoot, not equipment guidance or a supported safety recommendation."
        ),
    },
    {
        "fact_id": "fact-369710878e64e9733141",
        "source_path": raw("rope365_25f1b23eb40be00e.json"),
        "marker": "Loops can have a lot of fun uses",
        "decision": "drop",
        "reason_code": "underspecified_loop_attachment_point",
        "reason": (
            "The answer names only an attachment point and omits the loading, "
            "stability, and use context needed to make the statement practically useful."
        ),
    },
    {
        "fact_id": "fact-450cc6ca8581da4c2681",
        "source_path": raw("kinbakutoday_463d41ffc7b6725a.json"),
        "marker": "There are many aspects of safety.",
        "decision": "edit",
        "question": (
            "In Milla Reika's account, how do physical and mental safety differ "
            "in Kinbaku?"
        ),
        "answer": (
            "Physical safety includes risks such as nerve injury or dropping "
            "someone; mental safety includes protecting the trust of a partner "
            "who has surrendered control."
        ),
        "support_type": "manual_paraphrase",
        "paraphrase_rationale": (
            "The answer expands the source's two labels into their adjacent, "
            "explicit examples and states the physical-versus-trust distinction "
            "in a parallel standalone sentence."
        ),
        "paraphrase_support_fragments": (
            "physical aspect such as potential nerve injuries or dropping someone",
            "Damaging that trust or breaking that trust can cause very deep injuries",
        ),
        "reason_code": "replace_two_word_safety_labels_with_consequences",
        "reason": (
            "The edit preserves the important distinction while explaining the "
            "visible physical risks and potentially invisible harm from broken trust."
        ),
    },
    {
        "fact_id": "fact-e587f02d5dbf2375dc9e",
        "source_path": raw("kinbakutoday_dfc9527c49ca8ad6.json"),
        "marker": "We should view desires to be ‘Deshi’",
        "decision": "edit",
        "question": (
            "In Sin's discussion of unverified 'Deshi' claims, what does he say "
            "the term means in context?"
        ),
        "answer": (
            "It means a student or follower, not proof that a benefactor declared "
            "a lineage."
        ),
        "support_type": "manual_paraphrase",
        "paraphrase_rationale": (
            "The answer combines the source's literal contextual meaning with its "
            "explicit warning about assertions lacking a recorded proclamation."
        ),
        "paraphrase_support_fragments": (
            "without recorded proclamation by benefactors",
            "it just means student, or follower",
        ),
        "reason_code": "add_lineage_caveat_to_deshi_meaning",
        "reason": (
            "The edit replaces a contextless translation quiz with the source's "
            "transferable caution about treating a title as lineage evidence."
        ),
    },
    {
        "fact_id": "fact-1a28fe890ccaa23e1d53",
        "source_path": raw("anatomiestudio_451ac66001188a42.json"),
        "marker": "Safety doesn’t mean eliminating all risk",
        "decision": "keep",
        "reason_code": "protected_new_partner_risk_management",
        "reason": (
            "The concise definition frames safer new-partner play as understanding "
            "and managing risk rather than promising that risk can be eliminated."
        ),
    },
    {
        "fact_id": "fact-cca2ce649c4a186e5da3",
        "source_path": DATA / "rope_topia_manual_v1.jsonl",
        "marker": "<loc>https://rope-topia.com/nerve-and-circulation-problems/</loc>",
        "decision": "keep",
        "allow_document_sha_mismatch": True,
        "reason_code": "owner_requested_nerve_circulation_resource_url",
        "reason": (
            "The sitemap-backed canonical URL preserves the owner-requested safety "
            "resource without inferring facts from its gated article body."
        ),
    },
    {
        "fact_id": "fact-e7a338d404977ce267a6",
        "source_path": raw("rope_resources_v1/rope365__7b5d548036392d65fec7.json"),
        "marker": "- Tie kneeling or sitting,",
        "decision": "keep",
        "reason_code": "protected_low_position_fall_prevention",
        "reason": (
            "Choosing a kneeling or sitting position is a direct, practical way "
            "to reduce the consequences of a fall while hands are restrained."
        ),
    },
    {
        "fact_id": "fact-fcd73b701180fa9c8f57",
        "source_path": raw("kinbakutoday_beb97b222e367198.json"),
        "marker": "Yamaguchi Tokiko’s “Self Bondage Classroom,”",
        "decision": "drop",
        "reason_code": "redundant_work_title_lookup",
        "reason": (
            "The title lookup is redundant with the substantially more useful v39 "
            "row that preserves this page's historical, non-instructional framing."
        ),
        "allow_document_sha_mismatch": True,
    },
    {
        "fact_id": "fact-28154fce3bea393c93cd",
        "source_path": raw("kinbakutoday_be37dae4ec8e0d88.json"),
        "marker": "Ugo describes the rise of the Japanese SM scene",
        "decision": "edit",
        "question": (
            "According to Marc's summary of Ugo's view, what helped drive the "
            "rise of Japan's SM scene?"
        ),
        "answer": (
            "Several perceptive and resourceful businesspeople who recognized "
            "a market."
        ),
        "support_type": "manual_paraphrase",
        "paraphrase_rationale": (
            "The answer keeps the source's attribution, plural actors, and market "
            "recognition while replacing the awkward phrase 'the merits of'."
        ),
        "paraphrase_support_fragments": (
            "several perceptive and resourceful businessmen",
            "saw a market",
        ),
        "reason_code": "clarify_attributed_japanese_sm_business_thesis",
        "reason": (
            "The edit turns a vague person lookup into a clearly attributed historical "
            "thesis about commercial actors recognizing demand."
        ),
    },
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (20, 276, 133, 124, 29, 54, 443, 70, 259, 504))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v44",
    "direct_rows_without_prior_curation": 266,
    "eligible_unreviewed_direct_rows": 116,
    "prior_context_reviewed_direct_rows_excluded": 150,
    "rows": 555,
    "sha256": "489933ac1e1ae563e4894fffbba0627b428d3384843f4d76a3a86fa9285c4d13",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v44": 517,
    "active_after_this_tranche": 514,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 3,
    "new_edits_applied": 3,
    "output_rows": 552,
    "output_sha256": "4e7b4cd68cefc09eedd78c81f54ae2c5d3c64930ab4adbb6b09dab52525bf2c2",
    "prior_pending_addition_fact_ids_preserved": 37,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 4,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}

PRODUCTION_INPUTS = previous.PRODUCTION_INPUTS
SEALED_EVAL_PATHS = previous.SEALED_EVAL_PATHS
PRIOR_PROJECTION_CURATIONS = (
    *ACTIVE_CURATIONS, QUALITY_MERIT_CURATION, TASUKI_CURATION,
    *CONTEXT_CURATIONS,
)


def prior_decision_artifacts() -> tuple[Path, ...]:
    paths = []
    for version in range(1, 45):
        directory = DATA / "manual_reviews" / f"context_merit_audit_v{version}"
        paths.extend((directory / f"context_merit_audit_v{version}.jsonl",
                      directory / f"pending_curation_context_merit_v{version}.jsonl",
                      directory / f"report_context_merit_v{version}.json"))
    return tuple(paths)


def build_projection(output: Path, report: Path,
                     curations: tuple[Path, ...]) -> None:
    previous.build_projection(output, report, curations)


def prior_reviewed_fact_ids() -> set[str]:
    return {row["fact_id"] for path in CONTEXT_AUDITS for row in read_jsonl(path)}


def ranked_unreviewed_direct(rows: list[dict]) -> list[dict]:
    reviewed = prior_reviewed_fact_ids()
    candidates = []
    for index, row in enumerate(rows, 1):
        if row.get("curation") or row["fact_id"] in reviewed:
            continue
        features = CORE.risk_features(row)
        candidates.append((-features["risk_score"], features["question_tokens"],
                           features["answer_tokens"], row["fact_id"], index,
                           row, features))
    candidates.sort(key=lambda item: item[:4])
    ranked = [{"active_index": item[4], "row": item[5], "features": item[6]}
              for item in candidates]
    if len(ranked) != 116:
        raise ValueError(f"v45 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v45 selection drift")
    return ranked


def selected_ranked(rows: list[dict]) -> tuple[list[dict], int, int]:
    return ranked_unreviewed_direct(rows)[:10], 0, 0


def source_text_for(spec: dict, active: dict) -> str:
    path = spec["source_path"]
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        if document["url"] != active["url"]:
            raise ValueError(f'{active["fact_id"]}: source URL drift')
        return document["text"]
    rows = read_jsonl(path)
    matching = [row for row in rows if row.get("fact_id") == active["fact_id"]]
    if not matching:
        matching = [row for row in rows
                    if row.get("question") == active["question"]]
    if len(matching) != 1:
        raise ValueError(f'{active["fact_id"]}: JSONL source row drift')
    source = matching[0]
    if source.get("url", source.get("evidence_url")) != active["url"]:
        raise ValueError(f'{active["fact_id"]}: JSONL source URL drift')
    return source["evidence"]


def source_evidence(spec: dict, active: dict) -> tuple[str, str]:
    source_text = source_text_for(spec, active)
    active_evidence = active.get("evidence", "")
    acceptable_hashes = {text_sha256(source_text)}
    if active_evidence:
        acceptable_hashes.add(text_sha256(active_evidence))
    if (active["document_sha256"] not in acceptable_hashes and
            not spec.get("allow_document_sha_mismatch")):
        raise ValueError(f'{active["fact_id"]}: source hash drift')
    if (active_evidence and
            normalize_text(active_evidence) not in normalize_text(source_text)):
        raise ValueError(f'{active["fact_id"]}: stored evidence drift')
    matches = [line for line in source_text.splitlines()
               if spec["marker"] in line]
    if len(matches) != 1:
        raise ValueError(f'{active["fact_id"]}: evidence marker drift')
    evidence = matches[0]
    answer = spec.get("answer", active["answer"])
    support_type = spec.get("support_type", "normalized_extractive")
    if support_type == "manual_paraphrase":
        for fragment in spec["paraphrase_support_fragments"]:
            if normalize_text(fragment) not in normalize_text(evidence):
                raise ValueError(f'{active["fact_id"]}: unsupported paraphrase fragment')
    elif normalize_text(answer) not in normalize_text(evidence):
        raise ValueError(f'{active["fact_id"]}: unsupported answer')
    return evidence, support_type


@contextlib.contextmanager
def patched_base(projected_dataset: Path):
    replacements = {
        "OUT_DIR": OUT_DIR, "AUDIT": AUDIT, "CURATION": CURATION,
        "REPORT": REPORT, "REVIEWER": REVIEWER, "REVIEWED_AT": REVIEWED_AT,
        "CONTEXT_CURATIONS": CONTEXT_CURATIONS, "SPECS": SPECS,
        "EXPECTED_SELECTION": EXPECTED_SELECTION,
        "ISOLATED_PROJECTION": ISOLATED_PROJECTION,
    }
    originals = {name: getattr(BASE, name) for name in replacements}
    original_ranking = CORE.ranked_unreviewed
    original_active = CORE.ACTIVE_DATASET
    original_evidence = EVIDENCE_PATCH_MODULE.source_evidence
    try:
        for name, value in replacements.items():
            setattr(BASE, name, value)
        CORE.ranked_unreviewed = selected_ranked
        CORE.ACTIVE_DATASET = projected_dataset
        EVIDENCE_PATCH_MODULE.source_evidence = source_evidence
        yield
    finally:
        EVIDENCE_PATCH_MODULE.source_evidence = original_evidence
        CORE.ACTIVE_DATASET = original_active
        CORE.ranked_unreviewed = original_ranking
        for name, value in originals.items():
            setattr(BASE, name, value)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v44-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v44.jsonl"
        projected_report = Path(temp) / "projection-v44.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 555:
            raise ValueError("v44 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v44 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v45"
        row["review_pass"] = "first_context_merit_review_of_v44_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 555,
            "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
            "prior_context_merit_review": False,
        }
        spec = next(spec for spec in SPECS if spec["fact_id"] == row["fact_id"])
        if spec.get("support_type") == "manual_paraphrase":
            row["paraphrase_rationale"] = spec["paraphrase_rationale"]
    write_jsonl(AUDIT, audits)

    curations = read_jsonl(CURATION)
    for row in curations:
        spec = next(spec for spec in SPECS if spec["fact_id"] == row["fact_id"])
        if spec.get("support_type") == "manual_paraphrase":
            row["support_type"] = "manual_paraphrase"
            row["paraphrase_rationale"] = spec["paraphrase_rationale"]
    write_jsonl(CURATION, curations)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v45"
    report["active_baseline"] = {
        "dataset": {"path": str(ACTIVE_DATASET.relative_to(ROOT)), "rows": 784,
                    "sha256": file_sha256(ACTIVE_DATASET)},
        "report": {"path": str(ACTIVE_REPORT.relative_to(ROOT)),
                   "sha256": file_sha256(ACTIVE_REPORT)},
        "curation": [{"path": str(path.relative_to(ROOT)),
                      "sha256": file_sha256(path)} for path in ACTIVE_CURATIONS],
    }
    report["selection"].update({
        "active_rows": 555, "eligible_unreviewed_rows": 116,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0, "rows_selected": 10,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v44 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v44"
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
    report["audit"]["rows"] = len(audits)
    report["audit"]["sha256"] = file_sha256(AUDIT)
    report["new_pending_curation"]["sha256"] = file_sha256(CURATION)
    report["new_pending_curation"]["edit_support_types"] = {
        "extractive": 0, "manual_paraphrase": 3,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
