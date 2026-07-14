#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v40 projection rows in v41."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V40_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v40"
sys.path[:0] = [str(ROOT), str(V40_DIR)]
import build_context_merit_audit_v40 as previous
from qa_quality import normalize_text

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = BASE.previous.previous
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v41.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v41.jsonl"
REPORT = OUT_DIR / "report_context_merit_v41.json"
REVIEWER = "codex-context-merit-audit-v41"
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
    for version in range(1, 41)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 41)
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
    {"fact_id": "fact-c5ea3f3d6d7c81f3ef91",
     "source_path": DATA / "rope_resource_factual_qa_v1.jsonl",
     "marker": "Lief's Rope Conditioning Tutorial",
     "decision": "keep",
     "reason_code": "owner_requested_house_of_bound_tutorial",
     "reason": "The manually verified tutorial title directly supports the owner-requested House of Bound video resource."},
    {"fact_id": "fact-648fe093b612e2a9c2aa",
     "source_path": resource("atx_empty_space__a33a4d4c949a724b153d.json"),
     "marker": "The Empty Space is a Queer, POC owned and operated community space",
     "decision": "keep",
     "reason_code": "owner_requested_austin_space_identity",
     "reason": "The official page's ownership and operation description is durable, useful context for the owner-requested Austin community resource."},
    {"fact_id": "fact-63c27a92425059c6c63b",
     "source_path": raw("anatomiestudio_9749de0eb1ff4ef3.json"),
     "marker": "Agree what your safe words are and what they mean",
     "decision": "keep",
     "reason_code": "essential_safeword_meaning_agreement",
     "reason": "Agreeing both the safewords and their operational meanings is direct, essential pre-play communication guidance."},
    {"fact_id": "fact-96dc72c01f6c4218c69c",
     "source_path": DATA / "rope_resource_factual_qa_v1.jsonl",
     "marker": "modern fusion of rope bondage styles",
     "decision": "keep",
     "reason_code": "owner_requested_bightbound_scope",
     "reason": "The manually verified style description provides useful scope for recommending the owner-requested BightBound educators."},
    {"fact_id": "fact-11af85954681aed5764b",
     "source_path": DATA / "rope_topia_manual_v1.jsonl",
     "marker": "https://rope-topia.com/newcomers-information/identifying-predatory-behaviour/",
     "decision": "keep",
     "reason_code": "owner_requested_predatory_behavior_resource_url",
     "reason": "The sitemap-backed canonical predatory-behaviour URL is intentionally retained as an owner-requested safety resource."},
    {"fact_id": "fact-e5480624bacddfe277a5",
     "source_path": DATA / "rope_resource_manual_v1.jsonl",
     "marker": "different types of rope, the anatomy, and communication",
     "decision": "keep",
     "reason_code": "owner_requested_free_safety_course_scope",
     "reason": "The manually verified course scope identifies an owner-requested free safety resource covering rope types, anatomy, and communication.",
     "support_type": "manual_paraphrase",
     "paraphrase_rationale": "The retained manual fact omits the source's article before 'anatomy' while preserving the same three course topics.",
     "paraphrase_support_fragments": (
         "different types of rope", "the anatomy", "communication",
     )},
    {"fact_id": "fact-d9cfaf6512e5daf9474c",
     "source_path": raw("rope365_15518f0912cce205.json"),
     "marker": "the tension on the arm should be very loose",
     "decision": "keep",
     "reason_code": "essential_angel_tie_tension_precaution",
     "reason": "The page directly states the very-loose arm-tension precaution as necessary for making the pattern safer."},
    {"fact_id": "fact-82258f3d54c1714346bd",
     "source_path": DATA / "rope_resource_factual_qa_v1.jsonl",
     "marker": "single and double column ties to basic frictions, and rope handling",
     "decision": "edit",
     "question": "Which foundational topics does Shibari Study list for its Beginner 1 level?",
     "answer": "single- and double-column ties, basic frictions, and rope handling",
     "reason_code": "repair_beginner_course_scope_list",
     "reason": "The edit preserves the owner-requested Beginner 1 scope while repairing the source-derived answer's ambiguous 'ties to' list grammar.",
     "support_type": "manual_paraphrase",
     "paraphrase_rationale": "Hyphenates the compound modifiers and turns the source's 'from ... to ...' range into a clear three-item list without changing its topics.",
     "paraphrase_support_fragments": (
         "single and double column ties", "basic frictions", "rope handling",
     )},
    {"fact_id": "fact-e27da1eeb2a2294df669",
     "source_path": DATA / "rope_resource_manual_v1.jsonl",
     "marker": "attach uplines to limbs or harnesses and master various lock-off methods",
     "decision": "keep",
     "reason_code": "owner_requested_free_uplines_course_scope",
     "reason": "The manually verified vendor-attributed scope helps route users to the owner-requested free uplines and lock-offs resource without adding procedural claims.",
     "support_type": "manual_paraphrase",
     "paraphrase_rationale": "The retained manual fact nominalizes the vendor's 'learn how to attach' and 'master' wording without changing the two stated course topics.",
     "paraphrase_support_fragments": (
         "attach uplines to limbs or harnesses",
         "lock-off methods for suspension lines",
     )},
    {"fact_id": "fact-3f4e286ced700dd8cae2",
     "source_path": raw("anatomiestudio_451ac66001188a42.json"),
     "marker": "If you're not sure of your boundaries, be honest and go slow",
     "decision": "keep",
     "reason_code": "useful_boundary_uncertainty_guidance",
     "reason": "Being honest, going slowly, and leaving further exploration for another time is concise, useful consent guidance when boundaries are uncertain."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (468, 85, 333, 55, 444, 501, 125, 464, 200, 159))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v40",
    "direct_rows_without_prior_curation": 284,
    "eligible_unreviewed_direct_rows": 156,
    "prior_context_reviewed_direct_rows_excluded": 128,
    "rows": 566,
    "sha256": "531847b6438eedf9e2b2b19c3ad01738b2857607403577a4ddb36b8226511ab5",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v40": 528,
    "active_after_this_tranche": 528,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 0,
    "new_edits_applied": 1,
    "output_rows": 566,
    "output_sha256": "3508923a183b639b9a3546f3edb8d972e11189e522b7686e62cd4ec134e46bc2",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 9,
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
    for version in range(1, 41):
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
    if len(ranked) != 156:
        raise ValueError(f"v41 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v41 selection drift")
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
    with tempfile.TemporaryDirectory(prefix=".v40-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v40.jsonl"
        projected_report = Path(temp) / "projection-v40.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 566:
            raise ValueError("v40 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v40 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v41"
        row["review_pass"] = "first_context_merit_review_of_v40_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 566,
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
    report["schema"] = "context-merit-audit-report-v41"
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
        "active_rows": 566,
        "eligible_unreviewed_rows": 156,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "rows_selected": len(EXPECTED_SELECTION),
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v40 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v40"
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
        "extractive": 0, "manual_paraphrase": 1,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
