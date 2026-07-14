#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v35 projection rows in v36."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V35_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v35"
sys.path[:0] = [str(ROOT), str(V35_DIR)]
import build_context_merit_audit_v35 as previous
from qa_quality import normalize_text

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = BASE.previous.previous
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v36.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v36.jsonl"
REPORT = OUT_DIR / "report_context_merit_v36.json"
REVIEWER = "codex-context-merit-audit-v36"
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
    for version in range(1, 36)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 36)
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
    {"fact_id": "fact-159661089b5be0767def",
     "source_path": raw("kinbakutoday_1dccbc876bee51b6.json"),
     "marker": "the main focus is between the two parties involved in the kinbaku",
     "decision": "edit",
     "question": "What does Kamui describe as the main focus of kinbaku performance, regardless of its audience?",
     "answer": "the main focus is between the two parties involved in the kinbaku",
     "reason_code": "replace_style_name_lookup_with_partner_focus",
     "reason": "The edit replaces a niche style-name lookup with Kamui's substantive principle that the interaction between the two participants remains central."},
    {"fact_id": "fact-9340247445e556ef2ff7",
     "source_path": raw("kinbakutoday_0fde39bdb08f42b9.json"),
     "marker": "In 1963, Uramado featured images from Ruiz",
     "decision": "drop",
     "reason_code": "incidental_illustrator_credit_trivia",
     "reason": "The name of one illustrator in one 1963 feature is narrow credit trivia; v32 already preserves the source's substantive point about bidirectional artistic influence."},
    {"fact_id": "fact-f4886b3a29c4d198cd92",
     "source_path": raw("kinbakutoday_e55b7fa7c543e266.json"),
     "marker": "set foundation chest harness (TK)",
     "decision": "keep",
     "reason_code": "useful_classical_structure_context",
     "reason": "The source directly identifies the foundation harness on which subsequent Classical Kinbaku ties are built, useful structural context without teaching the tie."},
    {"fact_id": "fact-03cbcf694e260c90a1bf",
     "source_path": raw("anatomiestudio_27ecdd4d7c9a5560.json"),
     "marker": "Silence is not consent, freezing is not consent",
     "decision": "keep",
     "reason_code": "clear_freezing_is_not_consent_safeguard",
     "reason": "The studio's direct statement that freezing is not consent is concise, context-complete safety guidance."},
    {"fact_id": "fact-5dd51fd0f935594ae914",
     "source_path": raw("kinbakutoday_241155b848764148.json"),
     "marker": "April issue was seized by the government",
     "decision": "keep",
     "reason_code": "durable_censorship_history_anchor",
     "reason": "The seizure year anchors the article's substantive account of postwar obscenity trials and Kitan Club's shutdown."},
    {"fact_id": "fact-8c5889da31b9f4a4356a",
     "source_path": raw("kinbakutoday_e55b7fa7c543e266.json"),
     "marker": "suspension rope is deliberately angled rather than perfectly vertical",
     "decision": "edit",
     "question": "How does Kasumi say a suspension rope can incorporate asymmetrical beauty reminiscent of ikebana?",
     "answer": "the suspension rope is deliberately angled rather than perfectly vertical",
     "reason_code": "replace_flower_style_lookup_with_design_principle",
     "reason": "The edit replaces a one-word art-form lookup with the concrete asymmetrical design choice the comparison was meant to explain."},
    {"fact_id": "fact-129d81751a6ad48e6494",
     "source_path": DATA / "rope_resource_factual_qa_v1.jsonl",
     "marker": "1/4 inch, double braid nylon",
     "decision": "keep",
     "reason_code": "useful_attributed_synthetic_rope_specification",
     "reason": "The vendor-attributed diameter, material, and construction are concrete product-selection details for the owner-requested synthetic-rope resource."},
    {"fact_id": "fact-a4501a08619f5fc7886c",
     "source_path": raw("rope_resources_v1/rope365__b602b6493b5eb6f55206.json"),
     "marker": "adapt the dialogue and make sure you don’t push each other too fast",
     "decision": "keep",
     "reason_code": "useful_new_partner_pacing_guidance",
     "reason": "The source directly connects experience-level discussion to adapting communication and avoiding overly rapid progression with a new partner."},
    {"fact_id": "fact-c4e80596fa7f64a094ca",
     "source_path": DATA / "rope_resource_factual_qa_v1.jsonl",
     "marker": "Only wash on the gentle cycle in cold water",
     "decision": "keep",
     "reason_code": "useful_vendor_specific_synthetic_rope_care",
     "reason": "The vendor-attributed washing and drying directions are concrete care guidance for its specified nylon rope."},
    {"fact_id": "fact-c3905b07a4b9926553ec",
     "source_path": raw("kinbakutoday_e55b7fa7c543e266.json"),
     "marker": "Many schools of Classical Kinbaku have no concept or set a precise requirement",
     "decision": "edit",
     "question": "How does Kasumi say many schools of Classical Kinbaku treat formal levels and completion?",
     "answer": "Many schools of Classical Kinbaku have no concept or set a precise requirement for ‘levels’, refusing to set a ‘completion’ or clearly delineate a student’s level.",
     "reason_code": "replace_bare_levels_answer_with_formal_endpoint_context",
     "reason": "The edit replaces a one-word answer with the full claim that many schools avoid both precise levels and a defined completion point."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (243, 475, 247, 494, 106, 218, 2, 558, 1, 141))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v35",
    "direct_rows_without_prior_curation": 311,
    "eligible_unreviewed_direct_rows": 206,
    "prior_context_reviewed_direct_rows_excluded": 105,
    "rows": 574,
    "sha256": "1d7839996e8249d0528ca8b99204f90766388ff9d742333ecfb1a5e670893429",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v35": 536,
    "active_after_this_tranche": 535,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 1,
    "new_edits_applied": 3,
    "output_rows": 573,
    "output_sha256": "b80dd2e09a5e28724a8c4865c188d6a30c00e36f28c3c0d4c694989fb590bdb1",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 6,
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
    for version in range(1, 36):
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
    if len(ranked) != 206:
        raise ValueError(f"v36 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v36 selection drift")
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
    with tempfile.TemporaryDirectory(prefix=".v35-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v35.jsonl"
        projected_report = Path(temp) / "projection-v35.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 574:
            raise ValueError("v35 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v35 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v36"
        row["review_pass"] = "first_context_merit_review_of_v35_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 574,
            "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
            "prior_context_merit_review": False,
        }
    write_jsonl(AUDIT, audits)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v36"
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
        "active_rows": 574,
        "eligible_unreviewed_rows": 206,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "rows_selected": len(EXPECTED_SELECTION),
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v35 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v35"
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
