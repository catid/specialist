#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v36 projection rows in v37."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V36_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v36"
sys.path[:0] = [str(ROOT), str(V36_DIR)]
import build_context_merit_audit_v36 as previous
from qa_quality import normalize_text

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = BASE.previous.previous
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v37.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v37.jsonl"
REPORT = OUT_DIR / "report_context_merit_v37.json"
REVIEWER = "codex-context-merit-audit-v37"
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
    for version in range(1, 37)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 37)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


def resource(name: str) -> Path:
    return DATA / "raw" / "rope_resources_v1" / name


SPECS = (
    {"fact_id": "fact-169751fdb0d2ab43dd0a",
     "source_path": raw("kinbakutoday_e55b7fa7c543e266.json"),
     "marker": "everything starts and ends with the gote",
     "decision": "edit",
     "question": "What old Classical Kinbaku concept does Kasumi say many new Western ideas break away from?",
     "answer": "everything starts and ends with the gote (arms restrained behind the back)",
     "reason_code": "replace_gote_term_lookup_with_creative_contrast",
     "reason": "The edit preserves the definition of gote while making it serve Kasumi's substantive contrast between a traditional constraint and Western experimentation."},
    {"fact_id": "fact-372357c6d61a3f8b6ce6",
     "source_path": raw("anatomiestudio_27ecdd4d7c9a5560.json"),
     "marker": "We call these conversations ‘negotiation’",
     "decision": "keep",
     "reason_code": "useful_consent_negotiation_definition",
     "reason": "The studio directly defines negotiation as the conversation supporting enthusiastic, active consent for activities and touching."},
    {"fact_id": "fact-b7aeaf00b0f273ef166e",
     "source_path": DATA / "rope_resource_factual_qa_v1.jsonl",
     "marker": "event focused on rope bondage with additional programming",
     "decision": "keep",
     "reason_code": "useful_owner_requested_event_scope",
     "reason": "The official resource's main focus and additional programming are useful, direct facts for recommending the owner-requested convention."},
    {"fact_id": "fact-dc790f8267b3115b450a",
     "source_path": raw("kinbakutoday_be37dae4ec8e0d88.json"),
     "marker": "Japanese words and concepts are used, not completely superfluously",
     "decision": "edit",
     "question": "What caution does Marc give about using Japanese concepts such as Zen, wabi-sabi, or Ma in rope workshops?",
     "answer": "Japanese words and concepts are used, not completely superfluously, but often to enhance a sense of mystery, feeding the necessity to follow workshops to uncover secrets in concepts like Zen, wabi–sabi or Ma (time; pause/space). Deepening one’s insight into certain aspects of Japanese culture is profoundly interesting, but not absolutely required to do rope bondage.",
     "reason_code": "replace_ma_term_lookup_with_attributed_teaching_critique",
     "reason": "The edit replaces a one-word Japanese-concept lookup with Marc's explicit caution against using such concepts to manufacture mystery or required instruction."},
    {"fact_id": "fact-bd0d9f53704c5d277b9f",
     "source_path": raw("kinbakutoday_0fde39bdb08f42b9.json"),
     "marker": "Uramado (translation “Rear Window”), which ran from 1956 to 1965",
     "decision": "keep",
     "reason_code": "durable_significant_publication_date_range",
     "reason": "The run dates are a durable historical anchor for a publication the source describes as significant in forming Japanese kinbaku culture."},
    {"fact_id": "fact-920250f8d4d0e8f09be1",
     "source_path": raw("rope365_5fdb5e78c2471772.json"),
     "marker": "Find your perfect rope size",
     "decision": "edit",
     "question": "How does Rope365 suggest experimenting to find a preferred rope length and diameter?",
     "answer": "Start with a very long rope, tie the same favorite tie, shorten and retie it repeatedly until it feels too short, and compare different diameters.",
     "reason_code": "replace_bare_diameter_answer_with_comparison_experiment",
     "reason": "The edit replaces a bare measurement lookup with a faithful, grammatical paraphrase of the source's repeatable comparison experiment for exploring length and diameter preferences.",
     "support_type": "manual_paraphrase",
     "paraphrase_rationale": "Repairs the source's malformed interrogative while preserving its sequence: begin long, repeat the same tie at shorter lengths until the rope feels too short, and compare diameters.",
     "paraphrase_support_fragments": (
         "Acquire a very long rope",
         "tie your favourite tie with it",
         "Then cut it a little shorter",
         "At what point does the rope feels too short",
         "Try different rope diameters as well",
     )},
    {"fact_id": "fact-cf2419d2cc1e0f68bb19",
     "source_path": raw("rope365_c89abf7c3a5c30e1.json"),
     "marker": "Yuki knot in the honour of Yukimura Haruki",
     "decision": "drop",
     "reason_code": "redundant_quick_release_eponym_lookup",
     "reason": "This duplicates the Yuki-knot attribution reviewed in v34, where the row was replaced by the more useful quick-release tradeoff and failure caution."},
    {"fact_id": "fact-f561614230ed4573ea9b",
     "source_path": raw("kinbakutoday_fbc68faadb8dbfca.json"),
     "marker": "images photographed by Alao Yokogi",
     "decision": "drop",
     "reason_code": "incidental_photo_book_photographer_credit",
     "reason": "The photographer credit is narrow publication trivia; v32 already removed the paired tying credit from the same photo-book passage for the same merit reason."},
    {"fact_id": "fact-004dc043eac35f6f8258",
     "source_path": raw("kinbakutoday_de20a4adcc8ec0d5.json"),
     "marker": "determined the nature and direction of the magazine Kitan Club",
     "decision": "keep",
     "reason_code": "substantive_model_work_publication_impact",
     "reason": "The translated historical source explicitly describes how Kawabata's bondage work shaped Kitan Club's subsequent nature and direction."},
    {"fact_id": "fact-27bc42ce45fb7840ea43",
     "source_path": raw("rope365_15518f0912cce205.json"),
     "marker": "Some of these positions may be difficult to sustain",
     "decision": "edit",
     "question": "What preparation and comfort caution does Rope365 give before exploring elbow positions?",
     "answer": "Some of these positions may be difficult to sustain while others can be very comfortable. Warm up the arms, especially the shoulders, and find out which one is your favourite.",
     "reason_code": "replace_trex_alias_with_elbow_preparation_caution",
     "reason": "The edit replaces an alternate-name lookup with the source's broader warning about sustainable positions and warming up the arms and shoulders."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (358, 161, 5, 262, 402, 282, 525, 526, 478, 120))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v36",
    "direct_rows_without_prior_curation": 307,
    "eligible_unreviewed_direct_rows": 196,
    "prior_context_reviewed_direct_rows_excluded": 111,
    "rows": 573,
    "sha256": "b80dd2e09a5e28724a8c4865c188d6a30c00e36f28c3c0d4c694989fb590bdb1",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v36": 535,
    "active_after_this_tranche": 533,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 2,
    "new_edits_applied": 4,
    "output_rows": 571,
    "output_sha256": "a058be0567dd23a389a07ad1d0f4d9d9ee1650df4808a4e58998f89ce0c83b7f",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 4,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}

PRODUCTION_INPUTS = (
    DATA / "train_qa_verified_leakfree_v2.jsonl",
    DATA / "train_qa_manual_v1.jsonl",
    DATA / "rope_resource_qa_v1.jsonl",
    DATA / "rope_resource_factual_qa_v1.jsonl",
    DATA / "rope_resource_manual_v1.jsonl",
    DATA / "rope_topia_manual_v1.jsonl",
    *PRIOR_PENDING_ADDITIONS,
)
SEALED_EVAL_PATHS = (
    DATA / "eval_qa.jsonl", DATA / "eval_qa_v2.jsonl",
    DATA / "eval_qa_v3.jsonl", DATA / "ood_qa.jsonl",
    DATA / "ood_qa_v3.jsonl",
)
PRIOR_PROJECTION_CURATIONS = (
    *ACTIVE_CURATIONS, QUALITY_MERIT_CURATION, TASUKI_CURATION,
    *CONTEXT_CURATIONS,
)


def prior_decision_artifacts() -> tuple[Path, ...]:
    paths = []
    for version in range(1, 37):
        directory = DATA / "manual_reviews" / f"context_merit_audit_v{version}"
        paths.extend((
            directory / f"context_merit_audit_v{version}.jsonl",
            directory / f"pending_curation_context_merit_v{version}.jsonl",
            directory / f"report_context_merit_v{version}.json",
        ))
    return tuple(paths)


def build_projection(output: Path, report: Path,
                     curations: tuple[Path, ...]) -> None:
    command = [
        sys.executable, "build_curated_qa.py",
        "--inputs", *(str(path.relative_to(ROOT)) for path in PRODUCTION_INPUTS),
        "--eval", *(str(path.relative_to(ROOT)) for path in SEALED_EVAL_PATHS),
        "--curation", *(str(path.relative_to(ROOT)) for path in curations),
        "--output", str(output), "--report", str(report),
    ]
    subprocess.run(command, cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)


def prior_reviewed_fact_ids() -> set[str]:
    return {
        row["fact_id"] for path in CONTEXT_AUDITS
        for row in read_jsonl(path)
    }


def ranked_unreviewed_direct(rows: list[dict]) -> list[dict]:
    reviewed = prior_reviewed_fact_ids()
    candidates = []
    for index, row in enumerate(rows, 1):
        if row.get("curation") or row["fact_id"] in reviewed:
            continue
        features = CORE.risk_features(row)
        candidates.append((
            -features["risk_score"], features["question_tokens"],
            features["answer_tokens"], row["fact_id"], index, row, features,
        ))
    candidates.sort(key=lambda item: item[:4])
    ranked = [{
        "active_index": item[4], "row": item[5], "features": item[6],
    } for item in candidates]
    if len(ranked) != 196:
        raise ValueError(f"v37 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v37 selection drift")
    return ranked


def selected_ranked(rows: list[dict]) -> tuple[list[dict], int, int]:
    return ranked_unreviewed_direct(rows)[:10], 0, 0


def source_evidence(spec: dict, active: dict) -> tuple[str, str]:
    """Validate full-source and manual-resource evidence without weakening pins."""
    if spec["source_path"].suffix == ".jsonl":
        source_rows = [
            row for row in read_jsonl(spec["source_path"])
            if row["fact_id"] == active["fact_id"]
        ]
        if len(source_rows) != 1:
            raise ValueError(f'{active["fact_id"]}: manual source row drift')
        source_row = source_rows[0]
        document = {
            "url": source_row["url"],
            "text": source_row["evidence"],
        }
    else:
        document = json.loads(spec["source_path"].read_text())
    if document["url"] != active["url"]:
        raise ValueError(f'{active["fact_id"]}: source URL drift')
    document_hash = text_sha256(document["text"])
    active_evidence = active.get("evidence", "")
    evidence_hash = text_sha256(active_evidence) if active_evidence else ""
    if active["document_sha256"] not in {document_hash, evidence_hash}:
        raise ValueError(f'{active["fact_id"]}: source hash drift')
    if active_evidence and normalize_text(active_evidence) not in normalize_text(
            document["text"]):
        raise ValueError(f'{active["fact_id"]}: stored evidence drift')
    matches = [line for line in document["text"].splitlines()
               if spec["marker"] in line]
    if len(matches) != 1:
        raise ValueError(f'{active["fact_id"]}: evidence marker drift')
    evidence = matches[0]
    answer = spec.get("answer", active["answer"])
    support_type = spec.get("support_type", "normalized_extractive")
    if support_type == "manual_paraphrase":
        if not spec.get("paraphrase_rationale"):
            raise ValueError(f'{active["fact_id"]}: missing paraphrase rationale')
        for fragment in spec.get("paraphrase_support_fragments", ()):
            if normalize_text(fragment) not in normalize_text(evidence):
                raise ValueError(
                    f'{active["fact_id"]}: unsupported paraphrase fragment')
    elif normalize_text(answer) not in normalize_text(evidence):
        raise ValueError(f'{active["fact_id"]}: unsupported answer')
    return evidence, support_type


@contextlib.contextmanager
def patched_base(projected_dataset: Path):
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
    with tempfile.TemporaryDirectory(prefix=".v36-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v36.jsonl"
        projected_report = Path(temp) / "projection-v36.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 573:
            raise ValueError("v36 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v36 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v37"
        row["review_pass"] = "first_context_merit_review_of_v36_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 573,
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
    report["schema"] = "context-merit-audit-report-v37"
    report["active_baseline"] = {
        "dataset": {"path": str(ACTIVE_DATASET.relative_to(ROOT)), "rows": 784,
                    "sha256": file_sha256(ACTIVE_DATASET)},
        "report": {"path": str(ACTIVE_REPORT.relative_to(ROOT)),
                   "sha256": file_sha256(ACTIVE_REPORT)},
        "curation": [
            {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
            for path in ACTIVE_CURATIONS
        ],
    }
    report["selection"].update({
        "active_rows": 573,
        "eligible_unreviewed_rows": 196,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "rows_selected": len(EXPECTED_SELECTION),
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v36 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v36"
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
        "extractive": 3, "manual_paraphrase": 1,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
