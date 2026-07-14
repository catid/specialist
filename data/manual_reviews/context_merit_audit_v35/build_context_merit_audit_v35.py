#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v34 projection rows in v35."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V34_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v34"
sys.path[:0] = [str(ROOT), str(V34_DIR)]
import build_context_merit_audit_v34 as previous
from qa_quality import normalize_text

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = BASE.previous.previous
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v35.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v35.jsonl"
REPORT = OUT_DIR / "report_context_merit_v35.json"
REVIEWER = "codex-context-merit-audit-v35"
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
    for version in range(1, 35)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 35)
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
    {"fact_id": "fact-491a27251098d7671181",
     "source_path": raw("rope365_25f1b23eb40be00e.json"),
     "marker": "The handcuff knot is a popular type of knots",
     "decision": "edit",
     "question": "What use and body-safety risk does Rope365 identify for the handcuff knot?",
     "answer": "The handcuff knot is a popular type of knots to capture two limbs at once. Since it’s a type of slip knot, it comes with the risk that it may tighten when put directly on the body.",
     "reason_code": "add_slip_knot_tightening_risk_to_handcuff_use",
     "reason": "The edit preserves the two-limb use while adding the source's important warning that this slip-knot form may tighten on the body."},
    {"fact_id": "fact-666f3197bddf34c429f4",
     "source_path": DATA / "rope_resource_factual_qa_v1.jsonl",
     "marker": "facilitated self-organizing peer education",
     "decision": "keep",
     "reason_code": "useful_owner_requested_convention_format",
     "reason": "The official page and reviewed evidence directly define the peer-led Ropenspace format for the owner-requested convention resource."},
    {"fact_id": "fact-fbd8611c45f6aa212804",
     "source_path": raw("kinbakutoday_82071039cb003b58.json"),
     "marker": "Kinbaku culture in Japan has its roots in theater",
     "decision": "edit",
     "question": "Where does the article place the cultural roots of kinbaku, while distinguishing later use of Hojōjutsu techniques?",
     "answer": "Kinbaku culture in Japan has its roots in theater.",
     "reason_code": "replace_influence_lookup_with_attributed_roots_claim",
     "reason": "The edit replaces a bare Kabuki answer with the article's broader, explicitly attributed distinction between theatrical roots and later Hojōjutsu research."},
    {"fact_id": "fact-f00215ad545910d26d3e",
     "source_path": raw("rope365_c89abf7c3a5c30e1.json"),
     "marker": "Bending forward will restrict the ability to breathe",
     "decision": "edit",
     "question": "Why does Rope365 recommend a quick release for the shrimp tie?",
     "answer": "Bending forward will restrict the ability to breathe, which is why using a quick release is a good idea for this tie.",
     "reason_code": "replace_japanese_term_lookup_with_breathing_safeguard",
     "reason": "The edit replaces a Japanese term lookup with the source's safety rationale for a quick release in a breathing-restrictive position."},
    {"fact_id": "fact-f7cdcc2b1832447442c5",
     "source_path": raw("kinbakutoday_eb5b3fd68f12e8e9.json"),
     "marker": "Empathy is key to really understand kinbaku",
     "decision": "edit",
     "question": "According to Sin, what quality is key to really understanding kinbaku?",
     "answer": "Empathy",
     "reason_code": "make_vague_empathy_claim_explicitly_attributed",
     "reason": "The edit preserves the useful answer while replacing a vague passive prompt with direct attribution to the interviewee."},
    {"fact_id": "fact-069a861dbb2bea9e47ca",
     "source_path": raw("anatomiestudio_27ecdd4d7c9a5560.json"),
     "marker": "Respect indecision (it is not a yes). Mixed messages mean",
     "decision": "keep",
     "reason_code": "clear_mixed_message_consent_safeguard",
     "reason": "Treating mixed messages as no is concise, direct consent guidance from the studio's published policy."},
    {"fact_id": "fact-3ae839a47cb7d1fb48e4",
     "source_path": raw("kinbakutoday_eb5b3fd68f12e8e9.json"),
     "marker": "Kouma in Japanese rope industry terms means",
     "decision": "edit",
     "question": "According to Sin, which two words make up the name KOUMANAWA, and what does he say they mean?",
     "answer": "Kouma in Japanese rope industry terms means ‘Jute’, and Nawa, ‘Rope’.",
     "reason_code": "replace_broad_term_lookup_with_attributed_brand_etymology",
     "reason": "The edit avoids presenting a niche terminology claim as universal and preserves it as the interviewee's explanation of the KOUMANAWA name."},
    {"fact_id": "fact-8b42806c54bd80c545ea",
     "source_path": raw("rope365_5fdb5e78c2471772.json"),
     "marker": "Melting (synthetic only)",
     "decision": "keep",
     "reason_code": "useful_material_specific_rope_ending_fact",
     "reason": "The source directly limits melting to synthetic rope, a concise and useful material-specific maintenance fact."},
    {"fact_id": "fact-08b5cb358fb1dafc0859",
     "source_path": raw("esinem_b249eddc1f5e1864.json"),
     "marker": "‘Lay’ is the technical word used to describe the twist",
     "decision": "edit",
     "question": "How does Esinem define rope lay, including its strict measurement sense?",
     "answer": "‘Lay’ is the technical word used to describe the twist of the rope. To be strictly correct, ‘lay’ is used as a unit of measure which relates to the distance over which a full twist spreads.",
     "reason_code": "expand_bare_lay_term_to_precise_definition",
     "reason": "The edit turns a one-word lookup into the source's precise definition of lay as both twist and the distance of a full twist."},
    {"fact_id": "fact-4792da521af19b4e36b1",
     "source_path": raw("kinbakutoday_be37dae4ec8e0d88.json"),
     "marker": "The name Nureki Chimuo as a Kinbakushi emerged in 1973",
     "decision": "keep",
     "reason_code": "durable_public_role_history_anchor",
     "reason": "The year marks the documented emergence of Nureki's Kinbakushi name after his magazine photo-shoot work, a meaningful public-role milestone rather than private biography."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (128, 71, 393, 264, 313, 94, 205, 496, 357, 100))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v34",
    "direct_rows_without_prior_curation": 317,
    "eligible_unreviewed_direct_rows": 216,
    "prior_context_reviewed_direct_rows_excluded": 101,
    "rows": 574,
    "sha256": "dbda721170f9005357bacade032dcc80c27841469b6495a1dc68a0c850c70d2d",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v34": 536,
    "active_after_this_tranche": 536,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 0,
    "new_edits_applied": 6,
    "output_rows": 574,
    "output_sha256": "1d7839996e8249d0528ca8b99204f90766388ff9d742333ecfb1a5e670893429",
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
    for version in range(1, 35):
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
    if len(ranked) != 216:
        raise ValueError(f"v35 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v35 selection drift")
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
    with tempfile.TemporaryDirectory(prefix=".v34-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v34.jsonl"
        projected_report = Path(temp) / "projection-v34.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 574:
            raise ValueError("v34 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v34 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v35"
        row["review_pass"] = "first_context_merit_review_of_v34_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 574,
            "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
            "prior_context_merit_review": False,
        }
    write_jsonl(AUDIT, audits)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v35"
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
        "eligible_unreviewed_rows": 216,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "rows_selected": len(EXPECTED_SELECTION),
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v34 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v34"
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
        "extractive": 6, "manual_paraphrase": 0,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
