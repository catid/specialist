#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v41 projection rows in v42."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V41_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v41"
sys.path[:0] = [str(ROOT), str(V41_DIR)]
import build_context_merit_audit_v41 as previous
from qa_quality import normalize_text

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v42.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v42.jsonl"
REPORT = OUT_DIR / "report_context_merit_v42.json"
REVIEWER = "codex-context-merit-audit-v42"
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
CORE = previous.CORE

CONTEXT_CURATIONS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"pending_curation_context_merit_v{version}.jsonl"
    for version in range(1, 42)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 42)
)
PRIOR_PROJECTION_CURATIONS = (
    *ACTIVE_CURATIONS, QUALITY_MERIT_CURATION, TASUKI_CURATION,
    *CONTEXT_CURATIONS,
)


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-a90111f186e1b44206d9",
        "source_path": raw("kinbakutoday_52a834250dbada36.json"),
        "marker": "there seem to be two prominent types",
        "decision": "edit",
        "question": (
            "In a Kinbaku Today interview, what two prominent ways of "
            "viewing rope play does Bingo describe?"
        ),
        "answer": (
            "People who view rope play as art or performance, and people who "
            "view it as inner psychology and communication."
        ),
        "support_type": "manual_paraphrase",
        "paraphrase_rationale": (
            "The answer converts the source's awkward repeated construction "
            "into a concise parallel comparison while preserving both "
            "viewpoints."
        ),
        "paraphrase_support_fragments": (
            "rope-play is an art and/or performance",
            "inner psychology and communication",
        ),
        "reason_code": "clarify_attributed_rope_play_viewpoints",
        "reason": (
            "The edit replaces a vague personal-reference question with an "
            "explicitly attributed comparison of two substantive ways of "
            "viewing rope play."
        ),
    },
    {
        "fact_id": "fact-97058c8a514e94adf2eb",
        "source_path": raw("anatomiestudio_27ecdd4d7c9a5560.json"),
        "marker": (
            "If you would like to watch, keep to a reasonable distance and "
            "be unintrusive."
        ),
        "decision": "keep",
        "reason_code": "useful_observer_etiquette",
        "reason": (
            "The official studio guidance gives concise, practical etiquette "
            "for observing a scene without intruding."
        ),
    },
    {
        "fact_id": "fact-5e938d022e3357d2d904",
        "source_path": raw("kinbakutoday_9af201153c54e4d5.json"),
        "marker": "including literature such as Bizarre",
        "decision": "drop",
        "reason_code": "incidental_literature_title_trivia",
        "reason": (
            "The four-title lookup is incidental historical trivia; the "
            "source's broader access and social-change claims are more useful."
        ),
    },
    {
        "fact_id": "fact-0f91ea0b7cf8408b77d2",
        "source_path": raw("rope365_c89abf7c3a5c30e1.json"),
        "marker": (
            "You’ll need to start from a change of direction or some friction "
            "on the hitch’s origin"
        ),
        "decision": "drop",
        "reason_code": "redundant_partial_slipped_hitch_condition",
        "reason": (
            "The row duplicates the entry condition already retained in v40, "
            "where it is paired with the essential locked-exit distinction."
        ),
    },
    {
        "fact_id": "fact-1de730232db74a37d37d",
        "source_path": DATA / "rope_topia_manual_v1.jsonl",
        "marker": (
            "https://rope-topia.com/newcomers-information/"
            "so-youre-new-to-the-kink-scene/"
        ),
        "decision": "keep",
        "reason_code": "owner_requested_newcomer_resource_url",
        "reason": (
            "The sitemap-backed canonical URL is intentionally retained as "
            "an owner-requested Rope-topia newcomer resource."
        ),
    },
    {
        "fact_id": "fact-a8b8fc2fcc29541ac904",
        "source_path": DATA / "rope_topia_manual_v1.jsonl",
        "marker": (
            "https://rope-topia.com/2013/11/luck-self-awareness-"
            "responsibility-rope-bondage-injuries/"
        ),
        "decision": "keep",
        "reason_code": "owner_requested_injury_resource_url",
        "reason": (
            "The sitemap-backed canonical URL preserves an owner-requested "
            "Rope-topia article specifically indexed for injury responsibility."
        ),
    },
    {
        "fact_id": "fact-192be625b19c3639bdc1",
        "source_path": raw("rope365_d9c48a4547717047.json"),
        "marker": (
            "Once we are done tying, it is best to untangle our ropes and "
            "store them for the next time."
        ),
        "decision": "drop",
        "reason_code": "redundant_generic_post_tie_storage",
        "reason": (
            "The generic untangle-and-store line is redundant with richer "
            "coiling rows already retained for bight access and transport."
        ),
    },
    {
        "fact_id": "fact-b512f05ececd78e2fd3a",
        "source_path": raw("wykd_19d6a26116e26c70.json"),
        "marker": (
            "Now I would expect someone who has inflicted multiple injuries"
        ),
        "decision": "edit",
        "question": (
            "What corrective steps does the article recommend when a rigger "
            "causes repeated injuries that others are not causing?"
        ),
        "answer": (
            "They should reduce how much they rig, examine the common themes "
            "in the injuries, and work hard to prevent them from recurring."
        ),
        "support_type": "manual_paraphrase",
        "paraphrase_rationale": (
            "The answer converts the source's nonparallel first-person "
            "infinitive list into a grammatical third-person summary without "
            "changing its three corrective steps."
        ),
        "paraphrase_support_fragments": (
            "stop rigging so much",
            "look at the common themes in these injuries",
            "work very hard to stop them occurring again",
        ),
        "reason_code": "clarify_repeated_injury_corrective_steps",
        "reason": (
            "The edit keeps the high-value accountability guidance while "
            "repairing the source-derived answer's nonparallel list grammar; "
            "curation lineage preserves the protected addition."
        ),
    },
    {
        "fact_id": "fact-c91029574adc7c6ead1b",
        "source_path": DATA / "rope_topia_manual_v1.jsonl",
        "marker": (
            "https://rope-topia.com/portfolio-items/kinbaku-today-rope-is-"
            "not-about-rope/portfolioCats-102-70-123-72-57/"
        ),
        "decision": "keep",
        "reason_code": "owner_requested_rope_not_about_rope_url",
        "reason": (
            "The current sitemap-backed URL is retained as owner-requested "
            "Rope-topia resource metadata without inferring gated article text."
        ),
    },
    {
        "fact_id": "fact-37e7995e1842fb2c3d1c",
        "source_path": raw("anatomiestudio_144932682af9c846.json"),
        "marker": (
            "use a different rope (or even a cloth) for the parts of ties "
            "that are more likely to get bodily fluids on them"
        ),
        "decision": "keep",
        "reason_code": "useful_bodily_fluid_hygiene_alternative",
        "reason": (
            "The source directly offers a different rope or cloth as a "
            "practical way to avoid contaminating hard-to-clean natural rope."
        ),
    },
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (385, 330, 344, 280, 449, 446, 327, 326, 460, 133))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v41",
    "direct_rows_without_prior_curation": 283,
    "eligible_unreviewed_direct_rows": 146,
    "prior_context_reviewed_direct_rows_excluded": 137,
    "rows": 566,
    "sha256": "3508923a183b639b9a3546f3edb8d972e11189e522b7686e62cd4ec134e46bc2",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v41": 528,
    "active_after_this_tranche": 525,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 3,
    "new_edits_applied": 2,
    "output_rows": 563,
    "output_sha256": "2a2997ee684be4df0c16df269f99638dc4c5d43b3dde100529e6bad2d293521e",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 5,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def prior_decision_artifacts() -> tuple[Path, ...]:
    paths = []
    for version in range(1, 42):
        directory = DATA / "manual_reviews" / f"context_merit_audit_v{version}"
        paths.extend((
            directory / f"context_merit_audit_v{version}.jsonl",
            directory / f"pending_curation_context_merit_v{version}.jsonl",
            directory / f"report_context_merit_v{version}.json",
        ))
    return tuple(paths)


def build_projection(output: Path, report: Path,
                     curations: tuple[Path, ...]) -> None:
    previous.build_projection(output, report, curations)


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
    if len(ranked) != 146:
        raise ValueError(f"v42 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v42 selection drift")
    return ranked


def selected_ranked(rows: list[dict]) -> tuple[list[dict], int, int]:
    return ranked_unreviewed_direct(rows)[:10], 0, 0


def source_evidence(spec: dict, active: dict) -> tuple[str, str]:
    """Validate the exact source line or documented manual paraphrase."""
    declared_document_hash = ""
    if spec["source_path"].suffix == ".jsonl":
        source_rows = [
            row for row in read_jsonl(spec["source_path"])
            if row["fact_id"] == active["fact_id"]
        ]
        if len(source_rows) != 1:
            raise ValueError(f'{active["fact_id"]}: manual source row drift')
        source_row = source_rows[0]
        document = {"url": source_row["url"], "text": source_row["evidence"]}
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
def patched_previous():
    replacements = {
        "OUT_DIR": OUT_DIR,
        "AUDIT": AUDIT,
        "CURATION": CURATION,
        "REPORT": REPORT,
        "REVIEWER": REVIEWER,
        "REVIEWED_AT": REVIEWED_AT,
        "CONTEXT_CURATIONS": CONTEXT_CURATIONS,
        "CONTEXT_AUDITS": CONTEXT_AUDITS,
        "PRIOR_PROJECTION_CURATIONS": PRIOR_PROJECTION_CURATIONS,
        "SPECS": SPECS,
        "EXPECTED_SELECTION": EXPECTED_SELECTION,
        "PROJECTED_ACTIVE_INDICES": PROJECTED_ACTIVE_INDICES,
        "PROJECTED_SELECTION_BASELINE": PROJECTED_SELECTION_BASELINE,
        "ISOLATED_PROJECTION": ISOLATED_PROJECTION,
        "prior_decision_artifacts": prior_decision_artifacts,
        "prior_reviewed_fact_ids": prior_reviewed_fact_ids,
        "ranked_unreviewed_direct": ranked_unreviewed_direct,
        "selected_ranked": selected_ranked,
        "source_evidence": source_evidence,
    }
    originals = {name: getattr(previous, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(previous, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(previous, name, value)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with patched_previous():
        previous.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v42"
        row["review_pass"] = "first_context_merit_review_of_v41_projection_row"
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
    report["schema"] = "context-merit-audit-report-v42"
    report["selection"].update({
        "active_rows": 566,
        "eligible_unreviewed_rows": 146,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "rows_selected": 10,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v41 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v41"
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
        "extractive": 0, "manual_paraphrase": 2,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
