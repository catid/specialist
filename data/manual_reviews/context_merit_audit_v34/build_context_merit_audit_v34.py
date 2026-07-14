#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v33 projection rows in v34."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V33_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v33"
sys.path[:0] = [str(ROOT), str(V33_DIR)]
import build_context_merit_audit_v33 as previous
from qa_quality import normalize_text

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = BASE.previous.previous
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v34.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v34.jsonl"
REPORT = OUT_DIR / "report_context_merit_v34.json"
REVIEWER = "codex-context-merit-audit-v34"
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
    for version in range(1, 34)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 34)
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
    {"fact_id": "fact-3a9b55a5c1a8d50f9ca2",
     "source_path": DATA / "rope_resource_factual_qa_v1.jsonl",
     "marker": "certified aerospace-grade 6061-T6 aluminum",
     "decision": "keep",
     "reason_code": "useful_attributed_ring_material_specification",
     "reason": "The manufacturer-attributed alloy specification is concrete equipment information for the owner-requested rigging-plate resource."},
    {"fact_id": "fact-a3833a3bcbe7d6dc8400",
     "source_path": raw("anatomiestudio_27ecdd4d7c9a5560.json"),
     "marker": "If you have doubts, don’t proceed",
     "decision": "keep",
     "reason_code": "clear_consent_doubt_safeguard",
     "reason": "The instruction to check in and not proceed when enthusiasm is doubtful is direct, useful consent guidance."},
    {"fact_id": "fact-568b90dc769339c13dc8",
     "source_path": raw("kinbakutoday_7aa19131bb45f119.json"),
     "marker": "The nawashi should stay in the background",
     "decision": "edit",
     "question": "What performance principle does Randa Mai illustrate by comparing the nawashi to a bunraku kuroko?",
     "answer": "The nawashi should stay in the background. The center of a performance is the woman who is being manipulated by rope",
     "reason_code": "replace_theater_label_with_model_centering_principle",
     "reason": "The edit replaces a theater-name lookup with the explicitly stated principle that the person being tied, not the rope artist, should remain the performance's center."},
    {"fact_id": "fact-ca89337cc988e783b057",
     "source_path": raw("kinbakutoday_241155b848764148.json"),
     "marker": "it was almost impossible to bring the topic up with your partner",
     "decision": "edit",
     "question": "Why did Chiba Eizo say it was difficult to discuss rope with a partner when he began tying?",
     "answer": "it was almost impossible to bring the topic up with your partner as it was considered to be “extremely perverted”",
     "reason_code": "replace_bare_label_with_historical_social_context",
     "reason": "The edit retains the quotation but makes it answer a meaningful historical question about the stigma that made partners difficult to find."},
    {"fact_id": "fact-95bbc0c7bc7ccc546b29",
     "source_path": raw("rope365_c89abf7c3a5c30e1.json"),
     "marker": "They may not be as solid as other locking methods",
     "decision": "edit",
     "question": "What benefit and failure-mode caution does Rope365 give for quick releases?",
     "answer": "They may not be as solid as other locking methods, but they are great in case of emergency, or for ties that may need adjustments along the way. We just need to be extra careful in case it unties accidentally with movement.",
     "reason_code": "replace_honorific_name_lookup_with_quick_release_tradeoff",
     "reason": "The edit replaces an eponym lookup with the source's operational benefit and warning that quick releases aid emergencies or adjustments but can release accidentally."},
    {"fact_id": "fact-9d92c53eb7191b793ff5",
     "source_path": raw("kinbakutoday_463d41ffc7b6725a.json"),
     "marker": "Top masturbation",
     "decision": "drop",
     "reason_code": "redundant_crude_label_lookup",
     "reason": "The crude label is redundant after v33 preserved the same passage's more useful operational no-wastage test for structure and partner experience."},
    {"fact_id": "fact-266712b588b63dd2e2d6",
     "source_path": DATA / "rope_resource_factual_qa_v1.jsonl",
     "marker": "30 feet of premium 6mm jute bondage rope",
     "decision": "keep",
     "reason_code": "useful_attributed_rope_product_specification",
     "reason": "The manufacturer-attributed diameter and material are concrete product-selection details for the owner-requested natural-fiber rope resource."},
    {"fact_id": "fact-a76f92cc6917c5bb16c8",
     "source_path": raw("kinbakutoday_be37dae4ec8e0d88.json"),
     "marker": "We had little or no access to moving images",
     "decision": "edit",
     "question": "According to Marc, how did limited access to moving images shape early Western rope learning?",
     "answer": "We had little or no access to moving images hence the focus on techniques and patterns instead of on interaction.",
     "reason_code": "replace_insider_term_lookup_with_learning_history",
     "reason": "The edit replaces the incidental 'pixel knot' nickname with the participant's substantive explanation for an early emphasis on patterns over interaction."},
    {"fact_id": "fact-300196e05741d520a754",
     "source_path": raw("kinbakutoday_57d1ad7ef6bbe56a.json"),
     "marker": "“Shibaru” in the Kun’yomi pronunciation",
     "decision": "keep",
     "reason_code": "useful_terminology_etymology",
     "reason": "The reading is a directly supported linguistic fact that helps explain the shared root of shibari and kinbaku."},
    {"fact_id": "fact-afe55d75f72d32dd32a5",
     "source_path": DATA / "rope_resource_factual_qa_v1.jsonl",
     "marker": "mix of fundamental technique",
     "decision": "keep",
     "reason_code": "useful_local_educator_teaching_focus",
     "reason": "The official FAQ's stated balance of fundamentals, reasons, and play applications is useful when recommending the owner-requested Austin educator."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (13, 335, 379, 358, 7, 365, 3, 359, 238, 160))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v33",
    "direct_rows_without_prior_curation": 322,
    "eligible_unreviewed_direct_rows": 226,
    "prior_context_reviewed_direct_rows_excluded": 96,
    "rows": 575,
    "sha256": "24245f86d3cceba0e1b6b119bb2cd5e519e8b2379381cae1cc1720d9a120c671",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v33": 537,
    "active_after_this_tranche": 536,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 1,
    "new_edits_applied": 4,
    "output_rows": 574,
    "output_sha256": "dbda721170f9005357bacade032dcc80c27841469b6495a1dc68a0c850c70d2d",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 5,
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
    for version in range(1, 34):
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
    if len(ranked) != 226:
        raise ValueError(f"v34 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v34 selection drift")
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
    with tempfile.TemporaryDirectory(prefix=".v33-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v33.jsonl"
        projected_report = Path(temp) / "projection-v33.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 575:
            raise ValueError("v33 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v33 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v34"
        row["review_pass"] = "first_context_merit_review_of_v33_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 575,
            "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
            "prior_context_merit_review": False,
        }
    write_jsonl(AUDIT, audits)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v34"
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
        "active_rows": 575,
        "eligible_unreviewed_rows": 226,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "rows_selected": len(EXPECTED_SELECTION),
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v33 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v33"
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
        "extractive": 4, "manual_paraphrase": 0,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
