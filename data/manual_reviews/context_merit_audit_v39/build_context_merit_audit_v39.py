#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v38 projection rows in v39."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V38_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v38"
sys.path[:0] = [str(ROOT), str(V38_DIR)]
import build_context_merit_audit_v38 as previous
from qa_quality import normalize_text

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = BASE.previous.previous
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v39.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v39.jsonl"
REPORT = OUT_DIR / "report_context_merit_v39.json"
REVIEWER = "codex-context-merit-audit-v39"
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
    for version in range(1, 39)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 39)
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
    {"fact_id": "fact-c13d971e01ce85fc8b11",
     "source_path": raw("rope365_25f1b23eb40be00e.json"),
     "marker": "The goal of this week is to learn and practice a few knots",
     "decision": "edit",
     "question": "What does Rope365 recommend doing to learn which knots may be useful in different tying contexts?",
     "answer": "Learn and practice a few knots, then consider which tying contexts make each knot useful.",
     "reason_code": "replace_flower_family_lookup_with_learning_goal",
     "reason": "The edit replaces a cultural label lookup with a grammatical, faithful paraphrase of the page's learning objective: practice a few knots and consider the contexts in which each may help.",
     "support_type": "manual_paraphrase",
     "paraphrase_rationale": "Repairs the source's awkward 'analyze in which context' wording and avoids overstating 'a few knots' as additional knot families.",
     "paraphrase_support_fragments": (
         "learn and practice a few knots",
         "analyze in which context these can be useful when tying someone up",
     )},
    {"fact_id": "fact-ef2cab8c383642b09db1",
     "source_path": raw("kinbakutoday_beb97b222e367198.json"),
     "marker": "This video is not presented as a guide to reproduce the material",
     "decision": "edit",
     "question": "Why does the page present Yamaguchi Tokiko's self-bondage text as a historical artifact rather than a how-to guide?",
     "answer": "The page treats the text as a historical artifact showing how rope, fantasy, and self-transformation were imagined in mid-century Japanese fetish writing, rather than as a how-to guide.",
     "reason_code": "replace_vague_period_with_noninstructional_historical_frame",
     "reason": "The edit replaces a vague period label with a grammatical paraphrase of the page's explicit safety boundary and rationale for treating the text as cultural history, not instruction.",
     "support_type": "manual_paraphrase",
     "paraphrase_rationale": "States the source's historical-artifact framing in clean language and contrasts it with a how-to guide without preserving the awkward phrase 'guide to reproduce' or turning the framing into a prohibition.",
     "paraphrase_support_fragments": (
         "not presented as a guide to reproduce the material",
         "reads the text as a historical artifact",
         "how rope, fantasy, and self-transformation were imagined",
     )},
    {"fact_id": "fact-47d8b1211f8d76a70629",
     "source_path": raw("rope365_c89abf7c3a5c30e1.json"),
     "marker": "the origin still needs tension to stay in play",
     "decision": "drop",
     "reason_code": "redundant_context_incomplete_quick_release_component",
     "reason": "The isolated component lookup is context-incomplete, while v34 already preserves the same source's more useful quick-release benefit and accidental-release warning."},
    {"fact_id": "fact-dd18848777e752fcf551",
     "source_path": raw("rope365_d9c48a4547717047.json"),
     "marker": "The bight is easy to distinguish from the rest of the rope",
     "decision": "keep",
     "reason_code": "practical_ready_to_tie_storage_check",
     "reason": "Keeping the bight distinguishable is a direct, practical coiling check that helps a stored rope be ready for the next tie."},
    {"fact_id": "fact-1ed8b5fc38ca0a0bd5e5",
     "source_path": raw("kinbakutoday_aee46e8c98aa2e99.json"),
     "marker": "After that, I asked the riggers what it was that they wanted",
     "decision": "edit",
     "question": "What two perspectives did the author ask an advanced workshop to compare after tying?",
     "answer": "I asked the riggers what it was that they wanted their bottoms to feel and asked the bottoms what they actually felt",
     "reason_code": "replace_master_lookup_with_intention_feedback_comparison",
     "reason": "The edit replaces a person lookup with the article's concrete exercise for comparing the tying partner's intended experience with the tied partner's actual experience."},
    {"fact_id": "fact-90aabff4c2e19aa164e9",
     "source_path": raw("kinbakutoday_be37dae4ec8e0d88.json"),
     "marker": "We had little or no access to moving images hence the focus",
     "decision": "edit",
     "question": "According to Marc, why did early Western rope study focus more on techniques and patterns than interaction?",
     "answer": "We had little or no access to moving images hence the focus on techniques and patterns instead of on interaction.",
     "reason_code": "replace_magazine_lookup_with_source_limitation_explanation",
     "reason": "The edit replaces magazine-name trivia with Marc's substantive explanation that scarce moving-image access pushed early learners toward static technique and pattern reconstruction."},
    {"fact_id": "fact-f2b4b7ddd1fc5e86ca2b",
     "source_path": raw("kinbakutoday_82071039cb003b58.json"),
     "marker": "Ito started his Kinbaku in 1916",
     "decision": "edit",
     "question": "How does Ugo date Ito Seiu's early Kinbaku work and Studies on Seme?",
     "answer": "Ito started his Kinbaku in 1916, if not earlier, and published his Studies on Seme in 1928.",
     "reason_code": "replace_person_lookup_with_caveated_history",
     "reason": "The edit retains the durable history while restoring the source's 'if not earlier' qualification and the relationship between the practice and publication dates."},
    {"fact_id": "fact-f7e802bf0b2759290dc6",
     "source_path": raw("anatomiestudio_144932682af9c846.json"),
     "marker": "all parties are able to give informed consent",
     "decision": "keep",
     "reason_code": "essential_hygiene_risk_consent_rule",
     "reason": "The article directly requires informed consent for rope-hygiene and sharing choices and follows it with concrete boundaries to agree in advance."},
    {"fact_id": "fact-3f37e62da3adf7c03140",
     "source_path": raw("kinbakutoday_0fde39bdb08f42b9.json"),
     "marker": "cross cultural communication flows in both directions",
     "decision": "edit",
     "question": "Why does the Uramado article argue that bondage-art influence likely flowed from the West back into Japan too?",
     "answer": "It is hard to imagine that images flowing into magazines as successful as Uramado and to a lesser degree Kitan Club didn’t shape and influence Japanese approaches to SM and bondage.",
     "reason_code": "replace_architecture_lookup_with_bidirectional_influence_evidence",
     "reason": "The edit replaces a decorative-element lookup with the article's core historical inference that widely circulated Western imagery likely influenced Japanese practice in return."},
    {"fact_id": "fact-c7b5502ef5ea14893404",
     "source_path": DATA / "rope_resource_manual_v1.jsonl",
     "marker": "ask lots of \"why\" questions",
     "decision": "keep",
     "reason_code": "useful_instructor_understanding_probe",
     "reason": "Asking instructors why they make particular choices is concise, actionable guidance for testing understanding rather than relying on authority alone."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (467, 41, 355, 293, 522, 279, 520, 284, 142, 269))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v38",
    "direct_rows_without_prior_curation": 297,
    "eligible_unreviewed_direct_rows": 176,
    "prior_context_reviewed_direct_rows_excluded": 121,
    "rows": 570,
    "sha256": "2cbc3790de21a4f38a8e8ad578485cbbb99c141706f33d6dab8f55e786bbd915",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v38": 532,
    "active_after_this_tranche": 531,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 1,
    "new_edits_applied": 6,
    "output_rows": 569,
    "output_sha256": "5404ae9a2dd206d0d7499f3e87bac8c59d7ca8c7fd5cbfa4eb022dc7f2c5e67f",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 3,
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
    for version in range(1, 39):
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
    if len(ranked) != 176:
        raise ValueError(f"v39 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v39 selection drift")
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
    with tempfile.TemporaryDirectory(prefix=".v38-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v38.jsonl"
        projected_report = Path(temp) / "projection-v38.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 570:
            raise ValueError("v38 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v38 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v39"
        row["review_pass"] = "first_context_merit_review_of_v38_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 570,
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
    report["schema"] = "context-merit-audit-report-v39"
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
        "active_rows": 570,
        "eligible_unreviewed_rows": 176,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "rows_selected": len(EXPECTED_SELECTION),
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v38 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v38"
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
        "extractive": 4, "manual_paraphrase": 2,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
