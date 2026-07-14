#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v39 projection rows in v40."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V39_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v39"
sys.path[:0] = [str(ROOT), str(V39_DIR)]
import build_context_merit_audit_v39 as previous
from qa_quality import normalize_text

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = BASE.previous.previous
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v40.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v40.jsonl"
REPORT = OUT_DIR / "report_context_merit_v40.json"
REVIEWER = "codex-context-merit-audit-v40"
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
    for version in range(1, 40)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 40)
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
    {"fact_id": "fact-7021b57b2f77bed4d7db",
     "source_path": raw("kinbakutoday_432c8adfc1abe686.json"),
     "marker": "For instance EMT Shears or rescue hooks",
     "decision": "keep",
     "reason_code": "essential_emergency_cutting_tool",
     "reason": "The rope-bottom safety guide explicitly recommends EMT shears or rescue hooks as a reliable emergency means to cut rope."},
    {"fact_id": "fact-29f4f925dae3470cb141",
     "source_path": raw("rope365_d9c48a4547717047.json"),
     "marker": "The knot is pulled and centred so it won’t come undone in transport",
     "decision": "keep",
     "reason_code": "practical_transport_security_check",
     "reason": "This is a direct, practical coiling check for keeping stored rope bundles secured during transport."},
    {"fact_id": "fact-034e26ba2559271b6ca6",
     "source_path": raw("anatomiestudio_451ac66001188a42.json"),
     "marker": "It’s good practice to use your “stop” signal before it’s truly needed",
     "decision": "keep",
     "reason_code": "protected_actionable_stop_signal_rehearsal",
     "reason": "The protected manual addition already pairs the recommended action with its timing in the question, providing concise, actionable stop-signal rehearsal guidance while preserving its original fact ID."},
    {"fact_id": "fact-dc4a50db71ea781ba039",
     "source_path": raw("kinbakutoday_63fcb1570feac169.json"),
     "marker": "Before you study the rope, study your partner",
     "decision": "edit",
     "question": "What partner-first principle does the beginner article recommend before studying rope technique?",
     "answer": "Before you study the rope, study your partner. Learn how to read them and how to create experiences for them. Learn to communicate with your body, your hands, your body position, your timing, and your intention.",
     "reason_code": "replace_book_title_with_partner_first_learning_principle",
     "reason": "The edit replaces a promotional book-title lookup with the article's central beginner principle: learn to read and communicate with the partner before prioritizing technique."},
    {"fact_id": "fact-50327a0e2c633b008204",
     "source_path": raw("kinbakutoday_82071039cb003b58.json"),
     "marker": "catalysts such as 50 Shades opened a floodgate",
     "decision": "drop",
     "reason_code": "incidental_pop_culture_catalyst",
     "reason": "The title lookup is incidental pop-culture trivia and adds less durable value than the same source's already preserved historical phase and lineage analysis."},
    {"fact_id": "fact-0a5f05a0867d04b87aaf",
     "source_path": raw("kinbakutoday_b2454d5b6578b8c6.json"),
     "marker": "the underlying strength of Osada-ryu is not that it produces people who tie the same",
     "decision": "edit",
     "question": "What does Scott identify as the underlying strength of Osada-ryu training?",
     "answer": "For me, the underlying strength of Osada-ryu is not that it produces people who tie the same, but rather it gives people the skill, both technically and in terms of being able to fully connect with the person they are tying, to create their own “voice”, so they tie quite differently.",
     "reason_code": "replace_mental_state_lookup_with_individual_voice_principle",
     "reason": "The edit replaces an abstract mental-state phrase with Scott's substantive teaching principle: technical and relational skills should enable an individual voice rather than uniform imitation."},
    {"fact_id": "fact-08e978b02856459f01c3",
     "source_path": raw("rope365_25f1b23eb40be00e.json"),
     "marker": "a noose can create an adjustable loop",
     "decision": "drop",
     "reason_code": "contextless_noose_function",
     "reason": "The bare noose-function lookup omits the same passage's instability and tightening cautions; v35 already preserves the safer, explicit slip-knot tightening warning."},
    {"fact_id": "fact-593da20dffb61934105a",
     "source_path": raw("rope365_c89abf7c3a5c30e1.json"),
     "marker": "can be released with one hand",
     "decision": "drop",
     "reason_code": "redundant_context_incomplete_quick_release_feature",
     "reason": "The one-hand feature omits the quick release's accidental-release limitation, while v34 already preserves the same source's fuller benefit and failure-mode warning."},
    {"fact_id": "fact-a82d0daf33ebbadbd67e",
     "source_path": resource("austin_rope_slingers__61f35c89b63907d255b2.json"),
     "marker": "Austin Rope Slingers is a volunteer-run and donation-supported peer rope group",
     "decision": "keep",
     "reason_code": "owner_requested_austin_peer_group_model",
     "reason": "The official page's organizational model is useful context for recommending the owner-requested Austin Rope Slingers community resource."},
    {"fact_id": "fact-34c77c437152b75d6d55",
     "source_path": raw("rope365_c89abf7c3a5c30e1.json"),
     "marker": "You’ll need to start from a change of direction or some friction",
     "decision": "edit",
     "question": "What does Rope365 say about the unlocked entry and locked exit of its slipped-half-hitch quick release?",
     "answer": "You’ll need to start from a change of direction or some friction on the hitch’s origin as the entry point will not be locked. Then we make a half hitch on the loop created by the first half hitch to lock the exit into place.",
     "reason_code": "complete_slipped_hitch_locking_conditions",
     "reason": "The edit replaces a partial construction lookup with the source's essential distinction: origin friction or a direction change is needed because the entry remains unlocked, while a second half hitch locks the exit."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (351, 182, 164, 260, 459, 25, 269, 49, 56, 96))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v39",
    "direct_rows_without_prior_curation": 290,
    "eligible_unreviewed_direct_rows": 166,
    "prior_context_reviewed_direct_rows_excluded": 124,
    "rows": 569,
    "sha256": "5404ae9a2dd206d0d7499f3e87bac8c59d7ca8c7fd5cbfa4eb022dc7f2c5e67f",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v39": 531,
    "active_after_this_tranche": 528,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 3,
    "new_edits_applied": 3,
    "output_rows": 566,
    "output_sha256": "531847b6438eedf9e2b2b19c3ad01738b2857607403577a4ddb36b8226511ab5",
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
    for version in range(1, 40):
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
    if len(ranked) != 166:
        raise ValueError(f"v40 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v40 selection drift")
    return ranked


def selected_ranked(rows: list[dict]) -> tuple[list[dict], int, int]:
    return ranked_unreviewed_direct(rows)[:10], 0, 0


def source_evidence(spec: dict, active: dict) -> tuple[str, str]:
    """Validate full-source and manual-resource evidence without weakening pins."""
    declared_document_hash = ""
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
        declared_document_hash = source_row.get("document_sha256", "")
    else:
        document = json.loads(spec["source_path"].read_text())
    if document["url"] != active["url"]:
        raise ValueError(f'{active["fact_id"]}: source URL drift')
    document_hash = text_sha256(document["text"])
    active_evidence = active.get("evidence", "")
    evidence_hash = text_sha256(active_evidence) if active_evidence else ""
    if active["document_sha256"] not in {
            document_hash, evidence_hash, declared_document_hash}:
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
    if normalize_text(answer) not in normalize_text(evidence):
        raise ValueError(f'{active["fact_id"]}: unsupported answer')
    return evidence, "normalized_extractive"


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
    with tempfile.TemporaryDirectory(prefix=".v39-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v39.jsonl"
        projected_report = Path(temp) / "projection-v39.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 569:
            raise ValueError("v39 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v39 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v40"
        row["review_pass"] = "first_context_merit_review_of_v39_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 569,
            "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
            "prior_context_merit_review": False,
        }
    write_jsonl(AUDIT, audits)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v40"
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
        "active_rows": 569,
        "eligible_unreviewed_rows": 166,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "rows_selected": len(EXPECTED_SELECTION),
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v39 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v39"
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
        "extractive": 3, "manual_paraphrase": 0,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
