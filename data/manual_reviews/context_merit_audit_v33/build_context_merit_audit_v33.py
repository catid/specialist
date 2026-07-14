#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v32 projection rows in v33."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V32_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v32"
sys.path[:0] = [str(ROOT), str(V32_DIR)]
import build_context_merit_audit_v32 as previous
from qa_quality import normalize_text

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = BASE.previous.previous
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v33.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v33.jsonl"
REPORT = OUT_DIR / "report_context_merit_v33.json"
REVIEWER = "codex-context-merit-audit-v33"
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
    for version in range(1, 33)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 33)
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
    {"fact_id": "fact-4800b37595f5ea331ae9",
     "source_path": raw("kinbakutoday_4f417369f7269c51.json"),
     "marker": "I have to leave no trace on the lady’s body",
     "decision": "edit",
     "question": "What constraint does Hige place on his bondage with amateur partners, in contrast with performance shows?",
     "answer": "I have to leave no trace on the lady’s body",
     "reason_code": "replace_culturally_loaded_emotion_trivia_with_practice_constraint",
     "reason": "The edit replaces a culturally loaded emotion lookup with Hige's concrete contrast between stage effects and his no-trace constraint with amateur partners."},
    {"fact_id": "fact-4fa3aa175aa6e0ecba4a",
     "source_path": raw("kinbakutoday_16c99b3e83d22af6.json"),
     "marker": "It was only one more tool, never the end point",
     "decision": "edit",
     "question": "In the author's account of Nureki's rope progressions, what role did suspension play?",
     "answer": "one more tool, never the end point or the goal of the tie itself",
     "reason_code": "replace_vague_era_lookup_with_suspension_role",
     "reason": "The edit replaces a vague era label with the article's practical point that suspension served the progression rather than being its goal."},
    {"fact_id": "fact-8f2ac7403fb535b93713",
     "source_path": raw("kinbakutoday_e55b7fa7c543e266.json"),
     "marker": "final knot that holds the obi of a kimono",
     "decision": "keep",
     "reason_code": "useful_kimono_knot_lineage",
     "reason": "The source directly identifies honmusubi and explains its concrete use in kimono fastening, preserving useful cultural context rather than a bare unexplained term."},
    {"fact_id": "fact-5b53d2d052203f6a73f0",
     "source_path": raw("kinbakutoday_be37dae4ec8e0d88.json"),
     "marker": "History, lineage and Hojōjutsu are dragged into the game",
     "decision": "edit",
     "question": "What does Marc say history, lineage, and Hojōjutsu are used to do in some Western rope education?",
     "answer": "lift rope bondage from taboo, make it nobler and give it a sense of being an ancient discipline",
     "reason_code": "replace_awkward_discipline_lookup_with_attributed_critique",
     "reason": "The edit converts an awkward term-recovery prompt into the participant's explicitly attributed critique of how historical framing can be used."},
    {"fact_id": "fact-4f7772ab91a636e50d78",
     "source_path": raw("kinbakutoday_82071039cb003b58.json"),
     "marker": "Kitan Club was launched in 1947",
     "decision": "keep",
     "reason_code": "durable_publication_history_anchor",
     "reason": "The launch year is a durable historical anchor for a publication the source describes as nurturing Japanese kink culture through 1975."},
    {"fact_id": "fact-9a7f21e1535c114f63e1",
     "source_path": raw("kinbakutoday_b2454d5b6578b8c6.json"),
     "marker": "Yukimura sensei, who passed away in 2016",
     "decision": "drop",
     "reason_code": "incidental_personal_death_year_trivia",
     "reason": "The death year is incidental biography and does not preserve the interview's more useful lessons about connection, teaching, or rope practice."},
    {"fact_id": "fact-bc6e8b0da047b1104063",
     "source_path": raw("kinbakutoday_82071039cb003b58.json"),
     "marker": "usage wasn’t widespread until the end of the 90s",
     "decision": "edit",
     "question": "According to the article, what did the delayed widespread use of the term Kinbakushi reflect?",
     "answer": "people didn’t care who tied the models",
     "reason_code": "replace_term_date_lookup_with_historical_significance",
     "reason": "The edit retains the terminology history while asking for the source's substantive interpretation of why the term spread late."},
    {"fact_id": "fact-c5663b10f92293ca2073",
     "source_path": raw("kinbakutoday_463d41ffc7b6725a.json"),
     "marker": "Each rope has to have a function",
     "decision": "edit",
     "question": "Under Milla Reika's no-wastage principle, what two questions test whether a rope has a function?",
     "answer": "Is it adding to the structure of the tie? Or, is it adding something to the pleasure or to the experience of my partner?",
     "reason_code": "replace_named_concept_lookup_with_operational_principle",
     "reason": "The edit replaces a one-word concept lookup with the interviewee's operational test for whether each rope contributes to structure or partner experience."},
    {"fact_id": "fact-eb477d063e72bae0adb8",
     "source_path": raw("anatomiestudio_27ecdd4d7c9a5560.json"),
     "marker": "Suspension is edge play",
     "decision": "keep",
     "reason_code": "direct_suspension_risk_classification",
     "reason": "The studio's direct classification of suspension as risky edge play is concise, context-complete safety guidance."},
    {"fact_id": "fact-fac07d5539b95226941c",
     "source_path": DATA / "rope_resource_manual_v1.jsonl",
     "marker": "asking other local bottoms",
     "decision": "keep",
     "reason_code": "useful_partner_vetting_channel",
     "reason": "The official resource directly recommends consulting other local bottoms as one practical channel for screening potential rope partners."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (5, 213, 246, 22, 105, 103, 357, 139, 52, 534))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v32",
    "direct_rows_without_prior_curation": 328,
    "eligible_unreviewed_direct_rows": 236,
    "prior_context_reviewed_direct_rows_excluded": 92,
    "rows": 576,
    "sha256": "cb024742297927177f94e801363c78db7cc5bd7f7d2d9a6cdb376d299bf23c72",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v32": 538,
    "active_after_this_tranche": 537,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 1,
    "new_edits_applied": 5,
    "output_rows": 575,
    "output_sha256": "24245f86d3cceba0e1b6b119bb2cd5e519e8b2379381cae1cc1720d9a120c671",
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
    for version in range(1, 33):
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
    if len(ranked) != 236:
        raise ValueError(f"v33 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v33 selection drift")
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
    with tempfile.TemporaryDirectory(prefix=".v32-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v32.jsonl"
        projected_report = Path(temp) / "projection-v32.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 576:
            raise ValueError("v32 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v32 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v33"
        row["review_pass"] = "first_context_merit_review_of_v32_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 576,
            "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
            "prior_context_merit_review": False,
        }
    write_jsonl(AUDIT, audits)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v33"
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
        "active_rows": 576,
        "eligible_unreviewed_rows": 236,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "rows_selected": len(EXPECTED_SELECTION),
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v32 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v32"
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
