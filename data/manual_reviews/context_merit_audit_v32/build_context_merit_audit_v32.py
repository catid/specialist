#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v31 projection rows in v32."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V31_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v31"
sys.path[:0] = [str(ROOT), str(V31_DIR)]
import build_context_merit_audit_v31 as previous
from qa_quality import normalize_text

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = BASE.previous.previous
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v32.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v32.jsonl"
REPORT = OUT_DIR / "report_context_merit_v32.json"
REVIEWER = "codex-context-merit-audit-v32"
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
    for version in range(1, 32)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 32)
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
    {"fact_id": "fact-5908273d31d98f365d7f",
     "source_path": raw("kinbakutoday_b2454d5b6578b8c6.json"),
     "marker": "techniques that occur in plain sight but are hidden",
     "decision": "edit",
     "question": "How does Scott extend the idea of urawaza beyond techniques performed behind the model?",
     "answer": "those techniques that occur in plain sight but are hidden because people aren’t aware of them",
     "reason_code": "replace_term_lookup_with_full_urawaza_nuance",
     "reason": "The edit replaces a one-word term lookup with the interviewee's useful extension of the concept to visible but unnoticed technique."},
    {"fact_id": "fact-a1f1d1f33a8afa0968b3",
     "source_path": raw("kinbakutoday_82071039cb003b58.json"),
     "marker": "In 1989, Akechi appeared on a general TV quiz program",
     "decision": "drop",
     "reason_code": "incidental_television_year_trivia",
     "reason": "The television-appearance year is incidental biography; a surviving row from the same passage already preserves its substantive art-and-craft point."},
    {"fact_id": "fact-eb1b37c41aea69ebce8d",
     "source_path": resource("austin_rope_slingers__61f35c89b63907d255b2.json"),
     "marker": "We welcome all skill levels, abilities, and identities",
     "decision": "keep",
     "reason_code": "durable_inclusive_local_resource_audience",
     "reason": "The official group page directly states a durable and useful inclusivity detail for the owner-requested Austin resource."},
    {"fact_id": "fact-8f6a7cce77beccb93906",
     "source_path": raw("kinbakutoday_dfc9527c49ca8ad6.json"),
     "marker": "Matsui Kenji radically changed Akechi’s tying",
     "decision": "drop",
     "reason_code": "personal_named_influence_trivia",
     "reason": "The named-influence claim is one participant's aside, while a surviving row from the article already covers its substantive mutual-communication argument."},
    {"fact_id": "fact-7bd39b7dad474eed8c67",
     "source_path": raw("kinbakutoday_fbc68faadb8dbfca.json"),
     "marker": "feature Hayashiba tied by the legendary Akechi Denki",
     "decision": "drop",
     "reason_code": "incidental_photo_book_credit_trivia",
     "reason": "Identifying the person who tied one photo-book section is narrow credit trivia and adds no broadly useful history, safety, or practice guidance."},
    {"fact_id": "fact-6b86aaed7f1648cc32d2",
     "source_path": raw("anatomiestudio_144932682af9c846.json"),
     "marker": "dry the ropes under tension to help them return",
     "decision": "edit",
     "question": "What does Anatomie Studio recommend if jute rope accidentally gets wet, despite advising against washing it?",
     "answer": "dry the ropes under tension to help them return to their original form",
     "reason_code": "restore_accidental_wetting_and_no_wash_context",
     "reason": "The edit preserves the recovery step while making clear that it applies to accidental wetting and does not endorse washing jute rope."},
    {"fact_id": "fact-97d52aa2d87dadfd8226",
     "source_path": raw("anatomiestudio_9749de0eb1ff4ef3.json"),
     "marker": "immediate responsibility will fall to the rigger to check in",
     "decision": "keep",
     "reason_code": "nonverbal_checkin_responsibility_context_complete",
     "reason": "The source directly assigns immediate check-in responsibility when a model becomes non-verbal, a useful safeguard alongside joint responsibility."},
    {"fact_id": "fact-3da2299eb958dd6cdc1b",
     "source_path": raw("kinbakutoday_0fde39bdb08f42b9.json"),
     "marker": "cross cultural communication flows in both directions",
     "decision": "edit",
     "question": "How does the Uramado article characterize cross-cultural influence between Japanese and Western bondage art?",
     "answer": "cross cultural communication flows in both directions",
     "reason_code": "replace_editor_lookup_with_bidirectional_influence",
     "reason": "The edit replaces an editor-name lookup with the article's substantive caution that artistic influence was bidirectional."},
    {"fact_id": "fact-65ef46afd70ce50c2b6b",
     "source_path": raw("rope365_8e6f4abea3bf4f6d.json"),
     "marker": "fast to tie and don’t compact into something hard to untie",
     "decision": "edit",
     "question": "What tradeoff does Rope365 give for using frictions instead of knots?",
     "answer": "they are fast to tie and don’t compact into something hard to untie but they require tension to stay in place",
     "reason_code": "replace_bare_tension_answer_with_friction_tradeoff",
     "reason": "The edit retains the tension limitation together with the speed and ease-of-untying benefits that make it meaningful."},
    {"fact_id": "fact-5ef444d1d86e741e5a62",
     "source_path": raw("kinbakutoday_be37dae4ec8e0d88.json"),
     "marker": "research multiple sources and mindsets to incorporate into creating their own styles",
     "decision": "edit",
     "question": "What does Marc say independent practitioners can do to create their own rope styles?",
     "answer": "research multiple sources and mindsets to incorporate into creating their own styles",
     "reason_code": "replace_personal_influence_label_with_learning_principle",
     "reason": "The edit replaces a vague personal-influence label with the participant's explicit learning principle for developing an individual style."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (366, 102, 536, 528, 535, 54, 408, 526, 185, 363))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v31",
    "direct_rows_without_prior_curation": 336,
    "eligible_unreviewed_direct_rows": 246,
    "prior_context_reviewed_direct_rows_excluded": 90,
    "rows": 579,
    "sha256": "0707cb2678ff2e7ba6f3eac1355eac2236d4217dcf163a7bf8802be3ff09106d",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v31": 541,
    "active_after_this_tranche": 538,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 3,
    "new_edits_applied": 5,
    "output_rows": 576,
    "output_sha256": "cb024742297927177f94e801363c78db7cc5bd7f7d2d9a6cdb376d299bf23c72",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 2,
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
    for version in range(1, 32):
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
    if len(ranked) != 246:
        raise ValueError(f"v32 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v32 selection drift")
    return ranked


def selected_ranked(rows: list[dict]) -> tuple[list[dict], int, int]:
    return ranked_unreviewed_direct(rows)[:10], 0, 0


def source_evidence(spec: dict, active: dict) -> tuple[str, str]:
    """Validate full-source and manual-resource evidence without weakening pins."""
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
    with tempfile.TemporaryDirectory(prefix=".v31-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v31.jsonl"
        projected_report = Path(temp) / "projection-v31.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 579:
            raise ValueError("v31 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v31 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v32"
        row["review_pass"] = "first_context_merit_review_of_v31_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 579,
            "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
            "prior_context_merit_review": False,
        }
    write_jsonl(AUDIT, audits)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v32"
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
        "active_rows": 579,
        "eligible_unreviewed_rows": 246,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "rows_selected": len(EXPECTED_SELECTION),
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v31 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v31"
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
        "extractive": 5, "manual_paraphrase": 0,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
