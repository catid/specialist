#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v37 projection rows in v38."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V37_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v37"
sys.path[:0] = [str(ROOT), str(V37_DIR)]
import build_context_merit_audit_v37 as previous
from qa_quality import normalize_text

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = BASE.previous.previous
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v38.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v38.jsonl"
REPORT = OUT_DIR / "report_context_merit_v38.json"
REVIEWER = "codex-context-merit-audit-v38"
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
    for version in range(1, 38)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 38)
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
    {"fact_id": "fact-b1976a3e15c9d8fae3fc",
     "source_path": raw("kinbakutoday_aee46e8c98aa2e99.json"),
     "marker": "reflecting to a great extent, the personality of the people practicing it",
     "decision": "edit",
     "question": "Why does the author expect shibari practice to differ across cultures?",
     "answer": "Because shibari reflects the people practicing it—their local BDSM customs, beliefs, and views about what a rope interaction should be and what its goals are.",
     "reason_code": "replace_person_credit_lookup_with_cultural_difference_thesis",
     "reason": "The edit replaces a single-person credit with a grammatical, faithful paraphrase of the article's thesis that local customs, beliefs, and interaction goals shape rope practice.",
     "support_type": "manual_paraphrase",
     "paraphrase_rationale": "Repairs the source's dangling 'result of shibari ... reflecting' construction while retaining its stated explanation: practice reflects local customs, beliefs, and interaction goals.",
     "paraphrase_support_fragments": (
         "reflecting to a great extent, the personality of the people practicing it",
         "the customs of your local BDSM community",
         "your set of beliefs in general",
         "what a rope interaction should be like, what its goals are",
     )},
    {"fact_id": "fact-c1697f4f94aa32b0b9f6",
     "source_path": raw("rope365_5fdb5e78c2471772.json"),
     "marker": "The best way to discover what we prefer is to try different ropes",
     "decision": "edit",
     "question": "How does Rope365 recommend discovering personal rope preferences?",
     "answer": "Try different ropes and compare their properties to discover your own preferences.",
     "reason_code": "replace_bare_object_with_experimentation_method",
     "reason": "The edit changes a bare object answer into a grammatical paraphrase of the source's recommendation to compare rope properties and experiment rather than assume one universal preference.",
     "support_type": "manual_paraphrase",
     "paraphrase_rationale": "Resolves the source's singular 'experiment with it' and shifting we/you pronouns while preserving its advice to try ropes, compare properties, and identify a personal fit.",
     "paraphrase_support_fragments": (
         "observe the spectrum of rope properties",
         "The best way to discover what we prefer is to try different ropes",
         "talk with others and see what they like and why",
     )},
    {"fact_id": "fact-97e8d7d04ad0051b1a9c",
     "source_path": DATA / "rope_topia_manual_v1.jsonl",
     "marker": "https://rope-topia.com/safety-cutters/",
     "decision": "keep",
     "reason_code": "owner_requested_safety_resource_url",
     "reason": "The sitemap-backed canonical safety-cutter URL is intentionally retained so the model can return the owner-requested Rope-topia resource."},
    {"fact_id": "fact-f282a4f4635ebd41890a",
     "source_path": DATA / "rope_topia_manual_v1.jsonl",
     "marker": "https://rope-topia.com/newcomers-information/",
     "decision": "keep",
     "reason_code": "owner_requested_newcomer_resource_url",
     "reason": "The sitemap-backed canonical newcomer-hub URL is intentionally retained so the model can return the owner-requested Rope-topia resource."},
    {"fact_id": "fact-55da0d1915d37b1d077f",
     "source_path": raw("kinbakutoday_36c008200d681448.json"),
     "marker": "Sin: Can you explain a little about Barajūjikan (Rosencreutz)?",
     "evidence_lines": 2,
     "support_type": "source_composite",
     "decision": "keep",
     "reason_code": "durable_regional_sm_venue_history",
     "reason": "The interview directly identifies the venue and describes it as the first SM bar in western Japan, a meaningful regional history anchor."},
    {"fact_id": "fact-ac7fbf74e38372bfbbbd",
     "source_path": raw("kinbakutoday_e55b7fa7c543e266.json"),
     "marker": "The visual characteristics of Modern Kinbaku include symmetry",
     "decision": "edit",
     "question": "Which visual features does Kasumi associate with Modern Kinbaku's reproducibility?",
     "answer": "The visual characteristics of Modern Kinbaku include symmetry and involve joining ropes, which makes it highly reproducible.",
     "reason_code": "replace_category_lookup_with_clean_reproducibility_features",
     "reason": "The edit replaces a category-name lookup with the source's clean follow-up sentence about symmetry and joined ropes, avoiding the preceding source typo without guessing at a correction."},
    {"fact_id": "fact-e9f1e475dd738b456f78",
     "source_path": DATA / "rope_topia_manual_v1.jsonl",
     "marker": "https://rope-topia.com/rope-bottom-guide/",
     "decision": "keep",
     "reason_code": "owner_requested_bottom_guide_resource_url",
     "reason": "The sitemap-backed canonical bottom-guide URL is intentionally retained so the model can return the owner-requested Rope-topia resource."},
    {"fact_id": "fact-27b7b67cf46b7aac33c4",
     "source_path": raw("kinbakutoday_1dccbc876bee51b6.json"),
     "marker": "hibarimusubi-type connection",
     "decision": "drop",
     "reason_code": "narrow_advanced_connection_preference",
     "reason": "The named connection is a narrow description of one artist's advanced suspension practice and lacks the context needed to teach or safely apply it."},
    {"fact_id": "fact-62c5c25b674662828ae4",
     "source_path": raw("kinbakutoday_dfc9527c49ca8ad6.json"),
     "marker": "Yukimura came into the scene in the 1990s",
     "decision": "keep",
     "reason_code": "durable_generational_history_anchor",
     "reason": "The decade anchors the source's substantive generational account from Nureki's photography through Akechi's performance to Yukimura's reciprocal interaction philosophy."},
    {"fact_id": "fact-63c654d8cdad2602da36",
     "source_path": raw("rope_resources_v1/rope365__b602b6493b5eb6f55206.json"),
     "marker": "An allowlist gives a better result than a blocklist",
     "decision": "keep",
     "reason_code": "useful_positive_negotiation_structure",
     "reason": "The source directly recommends stating what is wanted and expected, not merely exclusions, which is useful negotiation guidance."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (521, 181, 452, 449, 398, 390, 451, 388, 107, 477))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v37",
    "direct_rows_without_prior_curation": 301,
    "eligible_unreviewed_direct_rows": 186,
    "prior_context_reviewed_direct_rows_excluded": 115,
    "rows": 571,
    "sha256": "a058be0567dd23a389a07ad1d0f4d9d9ee1650df4808a4e58998f89ce0c83b7f",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v37": 533,
    "active_after_this_tranche": 532,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 1,
    "new_edits_applied": 3,
    "output_rows": 570,
    "output_sha256": "2cbc3790de21a4f38a8e8ad578485cbbb99c141706f33d6dab8f55e786bbd915",
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
    for version in range(1, 38):
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
    if len(ranked) != 186:
        raise ValueError(f"v38 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v38 selection drift")
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
    lines = document["text"].splitlines()
    matches = [index for index, line in enumerate(lines)
               if spec["marker"] in line]
    if len(matches) != 1:
        raise ValueError(f'{active["fact_id"]}: evidence marker drift')
    start = matches[0]
    evidence = "\n".join(lines[start:start + spec.get("evidence_lines", 1)])
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
    with tempfile.TemporaryDirectory(prefix=".v37-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v37.jsonl"
        projected_report = Path(temp) / "projection-v37.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 571:
            raise ValueError("v37 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v37 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v38"
        row["review_pass"] = "first_context_merit_review_of_v37_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 571,
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
    report["schema"] = "context-merit-audit-report-v38"
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
        "active_rows": 571,
        "eligible_unreviewed_rows": 186,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "rows_selected": len(EXPECTED_SELECTION),
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v37 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v37"
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
        "extractive": 1, "manual_paraphrase": 2,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
