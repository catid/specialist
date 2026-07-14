#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v51 projection rows in v52."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V51_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v51"
sys.path[:0] = [str(ROOT), str(V51_DIR)]
import build_context_merit_audit_v51 as previous

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = previous.EVIDENCE_PATCH_MODULE
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v52.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v52.jsonl"
REPORT = OUT_DIR / "report_context_merit_v52.json"
REVIEWER = "codex-context-merit-audit-v52"
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
    for version in range(1, 52)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 52)
)


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-4dc3a8b24b77fb1a8d5a",
        "source_path": DATA / "rope_topia_manual_v1.jsonl",
        "marker": "<loc>https://rope-topia.com/2012/09/yin-yoga-for-bondage/</loc>",
        "decision": "keep",
        "allow_document_sha_mismatch": True,
        "reason_code": "owner_requested_yin_yoga_resource_url",
        "reason": (
            "The sitemap-backed canonical URL preserves an owner-requested mobility "
            "resource without inferring content from the unavailable article body."
        ),
    },
    {
        "fact_id": "fact-bfa47830949fa286cd78",
        "source_path": raw("kinbakutoday_56b268785cbff3bb.json"),
        "marker": "Showa, Heisei: 45 Sugiura Norio (1972-2017)",
        "decision": "drop",
        "reason_code": "incidental_photo_collection_title_lookup",
        "reason": (
            "The exact exhibition-volume title is isolated proper-name trivia; the "
            "retrospective's photographic history is more useful training material."
        ),
    },
    {
        "fact_id": "fact-83b9ff0d76b61cd77be6",
        "source_path": raw("anatomiestudio_451ac66001188a42.json"),
        "marker": "Consent must be freely given, enthusiastic, and reversible at any time",
        "decision": "keep",
        "reason_code": "practical_consent_properties",
        "reason": (
            "The three properties are concise, actionable consent guidance for rope play."
        ),
    },
    {
        "fact_id": "fact-4f46839d5672259e679c",
        "source_path": ROOT / "sources" / "manual_facts" / "resource_group_b.jsonl",
        "marker": "makes no guarantee of safety, suitability, or longevity",
        "decision": "keep",
        "reason_code": "vendor_bamboo_safety_disclaimer",
        "reason": (
            "The vendor's explicit disclaimer is important context for anyone evaluating "
            "natural bamboo as suspension equipment."
        ),
    },
    {
        "fact_id": "fact-08a27108ab9b8483340c",
        "source_path": raw("kinbakutoday_b6ca615acc15e5f5.json"),
        "marker": "wanted to express the distinctive tempestuousness of Kinbaku without modification",
        "decision": "keep",
        "reason_code": "attributed_no_retouch_artistic_rationale",
        "reason": (
            "The answer preserves the photographer's substantive artistic reason rather "
            "than merely recalling an exhibition detail."
        ),
    },
    {
        "fact_id": "fact-64a4807147c057265799",
        "source_path": raw("rope_resources_v1/rope365__b602b6493b5eb6f55206.json"),
        "marker": "How does my body feel?\u201d and \u201cIn what emotional state am I?",
        "decision": "keep",
        "reason_code": "pre_scene_physical_emotional_self_scan",
        "reason": (
            "The two questions prompt a useful physical and emotional readiness check "
            "before tying or being tied."
        ),
    },
    {
        "fact_id": "fact-bb9e3bb1450b30eb26fb",
        "source_path": DATA / "rope_resource_manual_v1.jsonl",
        "marker": "put it in a mesh bag, machine wash and lay it out or hang it to dry",
        "support_type": "manual_paraphrase",
        "paraphrase_support_fragments": (
            "put it in a mesh bag",
            "machine wash",
            "lay it out or hang it to dry",
        ),
        "paraphrase_rationale": (
            "The answer adds only the implied pronoun needed for grammatical parallelism "
            "while preserving the source's three care steps."
        ),
        "decision": "keep",
        "reason_code": "vendor_nylon_washing_and_drying_guidance",
        "reason": (
            "The attributed answer gives a complete, material-specific washing and drying "
            "procedure for the vendor's nylon rope."
        ),
    },
    {
        "fact_id": "fact-316aa473edd7e937bb6c",
        "source_path": ROOT / "sources" / "manual_facts" / "resource_group_b.jsonl",
        "marker": "cut node-to-node, and flame cured with precision",
        "decision": "drop",
        "reason_code": "incidental_product_processing_detail",
        "reason": (
            "How a particular vendor cuts and cures this listing is product trivia and "
            "does not establish load safety or fitness for suspension."
        ),
    },
    {
        "fact_id": "fact-570d276be37adb886b79",
        "source_path": raw("kinbakutoday_1b6cf9eed83c2ec4.json"),
        "marker": "Kinukawa-san, Otsuka-san, Aikawa-san, and Hanasaka-san",
        "decision": "drop",
        "reason_code": "isolated_historical_model_name_list",
        "reason": (
            "Memorizing four model names is low-value roster trivia; the essay's analysis "
            "of recurring models and reader attachment is more substantive."
        ),
    },
    {
        "fact_id": "fact-4d68b129fbee9d884b45",
        "source_path": raw("rope365_25f1b23eb40be00e.json"),
        "marker": "disappear completely, without leaving a tangle",
        "decision": "keep",
        "reason_code": "exploding_knot_functional_distinction",
        "reason": (
            "The answer clearly distinguishes the functional result of an exploding knot "
            "from an ordinary quick release."
        ),
    },
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (438, 256, 347, 15, 304, 485, 65, 13, 451, 157))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v51",
    "direct_rows_without_prior_curation": 248,
    "eligible_unreviewed_direct_rows": 46,
    "prior_context_reviewed_direct_rows_excluded": 202,
    "rows": 546,
    "sha256": "48fbff4e40596e9ce544196db7dde7301acfb592ab63a1112997f4a5132ff56a",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v51": 508,
    "active_after_this_tranche": 505,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 3,
    "new_edits_applied": 0,
    "output_rows": 543,
    "output_sha256": "21ca3f4e5be03d7dad647ec5a175f67227df064343c6f6a1ba336325ac96637e",
    "prior_pending_addition_fact_ids_preserved": 37,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 7,
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
    for version in range(1, 52):
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
    if len(ranked) != 46:
        raise ValueError(f"v52 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v52 selection drift")
    return ranked


def selected_ranked(rows: list[dict]) -> tuple[list[dict], int, int]:
    return ranked_unreviewed_direct(rows)[:10], 0, 0


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
        EVIDENCE_PATCH_MODULE.source_evidence = previous.previous.previous.previous.source_evidence
        yield
    finally:
        EVIDENCE_PATCH_MODULE.source_evidence = original_evidence
        CORE.ACTIVE_DATASET = original_active
        CORE.ranked_unreviewed = original_ranking
        for name, value in originals.items():
            setattr(BASE, name, value)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v51-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v51.jsonl"
        projected_report = Path(temp) / "projection-v51.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 546:
            raise ValueError("v51 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v51 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v52"
        row["review_pass"] = "first_context_merit_review_of_v51_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 546,
            "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
            "prior_context_merit_review": False,
        }
        spec = next(spec for spec in SPECS if spec["fact_id"] == row["fact_id"])
        if spec.get("support_type") == "manual_paraphrase":
            row["paraphrase_rationale"] = spec["paraphrase_rationale"]
    write_jsonl(AUDIT, audits)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v52"
    report["active_baseline"] = {
        "dataset": {"path": str(ACTIVE_DATASET.relative_to(ROOT)), "rows": 784,
                    "sha256": file_sha256(ACTIVE_DATASET)},
        "report": {"path": str(ACTIVE_REPORT.relative_to(ROOT)),
                   "sha256": file_sha256(ACTIVE_REPORT)},
        "curation": [{"path": str(path.relative_to(ROOT)),
                      "sha256": file_sha256(path)} for path in ACTIVE_CURATIONS],
    }
    report["selection"].update({
        "active_rows": 546, "eligible_unreviewed_rows": 46,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0, "rows_selected": 10,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v51 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v51"
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
        "extractive": 0, "manual_paraphrase": 0,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
