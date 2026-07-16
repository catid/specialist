#!/usr/bin/env python3
"""Repair five existing history/lineage Q&As against sealed source snapshots.

This bounded pass uses sealed v444 as its only candidate base.  It does not
ingest any pending dense-site corpus.  The five edits replace visual-detail or
title recall and inherited awkward wording with standalone, evidence-backed
historical claims that a rope learner could plausibly ask about.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
V444 = DATA / "manual_reviews/context_merit_audit_v444"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V444), str(V290)]

import build_context_merit_audit_v444 as previous
import build_context_merit_audit_v290 as core
from qa_quality import stable_fact_id


AUDIT = HERE / "context_merit_audit_v445.jsonl"
CURATION = HERE / "pending_curation_context_merit_v445.jsonl"
REPORT = HERE / "report_context_merit_v445.json"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
BASELINE_ROWS = 518
BASELINE_SHA256 = "41c8df268b82bfc33a4f9f74adfbdd5519aa892464569937afeb39cbb0a54565"
EXPECTED_OUTPUT_SHA256 = "0f3f5f17cae500edfad26a359d436d3e7609815f76b9d19bf37f8a24fdbc8b5c"
EXPECTED_CAPACITY_BEFORE = {
    "conflict_units": 261,
    "equipment_material": 22,
    "resources_general": 81,
    "safety_consent": 86,
    "technique": 72,
}
EXPECTED_CAPACITY_AFTER = {
    "conflict_units": 261,
    "equipment_material": 22,
    "resources_general": 80,
    "safety_consent": 86,
    "technique": 73,
}
OWNER_SOURCE = previous.OWNER_SOURCE
OWNER_FACT_IDS = previous.OWNER_FACT_IDS
REVIEWED_AT = "2026-07-16"
REVIEWER = "codex-context-merit-audit-v445"

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
portable = previous.portable
conservative_capacity = previous.conservative_capacity


EDIT_SPECS = {
    "fact-ae75486e9ca88421d04c": {
        "active_index": 132,
        "question": "In Kinbaku Today’s “Signs of Shame: Understanding Japanese Bondage Images,” what did the sign beside the monks in the 1878 public-shaming illustration display?",
        "answer": "The sign displayed details of the monks’ offenses for the public to see.",
        "edited_question": "How did the 1878 illustration discussed by Kinbaku Today combine rope and signage to stage the monks’ public shaming?",
        "edited_answer": "The monks were tied around their upper bodies for show and emotional effect, while a sign displayed their offenses to passersby.",
        "source_document": "data/raw/kinbakutoday_490a87ec78c3d64b.json",
        "source_document_file_sha256": "ba842888bfbb7f73c918e310a7a7f488ee18f6436379e532d904a1bd3bf576f3",
        "source_document_chars": 4545,
        "reason_code": "replace_sign_detail_recall_with_public_shaming_function",
        "reason": "The replacement turns a narrow visual-detail lookup into the source-supported historical function of both the display binding and the offense sign.",
    },
    "fact-004dc5980fc87def0cb9": {
        "active_index": 141,
        "question": "In “Nureki, Kinbiken, and the Aesthetics of Kinbaku,” what shorter name is identified as a contraction of Kinbakubi kenkyūkai?",
        "answer": "The contraction of Kinbakubi kenkyūkai is Kinbiken.",
        "edited_question": "What does the name Kinbiken abbreviate, and what focus did that name signal?",
        "edited_answer": "Kinbiken abbreviates Kinbakubi kenkyūkai, meaning “Beautiful Bondage Study Group,” and signals an aesthetic focus that the source says distinguished it from most rope study groups or dojos.",
        "source_document": "data/raw/kinbakutoday_f57559bbb4c8b826.json",
        "source_document_file_sha256": "7bdb311f25847ea18eb63342ac4030236566f4921985ec01f0401c40215d16f9",
        "source_document_chars": 5250,
        "reason_code": "complete_kinbiken_name_and_aesthetic_focus",
        "reason": "The replacement preserves the name expansion while adding the durable historical significance that the source explicitly attaches to Kinbiken's focus on kinbaku aesthetics.",
    },
    "fact-e1ab827121ac0e6176ac": {
        "active_index": 155,
        "question": "In the Yukimura-ryū lesson, what did Yukimura place above displaying rope technique?",
        "answer": "He placed bringing out the rope bottom’s beauty through the partners’ relationship and communication above self-satisfied technique, while still requiring basic safety knowledge and skill.",
        "edited_question": "What did Yukimura prioritize over displaying rope technique?",
        "edited_answer": "Yukimura prioritized bringing out the rope bottom’s beauty through the partners’ relationship and communication, while still requiring basic knowledge and technique for safety.",
        "source_document": "data/raw/kinbakutoday_381214111022a952.json",
        "source_document_file_sha256": "7d2758400af9c40ad38de7de1cbf1e234bb2ab3721e30e9382b78a106657b997",
        "source_document_chars": 12826,
        "reason_code": "naturalize_yukimura_partner_centered_priority",
        "reason": "The replacement removes the awkward pronoun-led construction and states Yukimura's partner-centered priority as a natural standalone answer without dropping the safety qualification.",
    },
    "fact-c6233ab851ba850483ea": {
        "active_index": 242,
        "question": "What does the servant holding rope in the 1930 illustration suggest to the post's author?",
        "answer": "Because the servant likely lacked access to formal Hojōjutsu knowledge, the image supports the author’s view that Edo torture rope was unrelated to Hojōjutsu.",
        "edited_question": "What lineage inference does the “Edo is not dead” post draw from its illustration of a servant holding rope?",
        "edited_answer": "The post treats the servant’s stated lack of access to formal Hojōjutsu as support for separating Edo torture rope from Hojōjutsu.",
        "source_document": "data/raw/kinbakutoday_938724eb415ca5c0.json",
        "source_document_file_sha256": "64a95a1fdefb574053e0c62646acf148becd776e3aeb5015e013dafce44b7967",
        "source_document_chars": 3558,
        "reason_code": "attribute_edo_torture_rope_lineage_inference",
        "reason": "The replacement makes the historical-lineage question standalone and keeps the contested conclusion explicitly attributed to the post rather than presenting it as settled fact.",
    },
    "fact-322a35d322334d0f533c": {
        "active_index": 343,
        "question": "What topic did Tsujimura Takashi’s 1953 Kitan Club work address?",
        "answer": "It addressed “the psychological background of the impulse to do Seme.”",
        "edited_question": "What contrast does a participant quoted from Tsujimura Takashi’s 1953 work draw between soft rope and the rope sensation they wanted?",
        "edited_answer": "The participant says rope that is too soft destroys the effect; despite feeling some concern for the model, they wanted rough rope digging into the model’s skin.",
        "source_document": "data/raw/kinbakutoday_c799ecf7b51f866b.json",
        "source_document_file_sha256": "f340f9aabc81c7291f07d4bc1b8f8b085201493414f7c726c86d237326e35c6a",
        "source_document_chars": 9044,
        "reason_code": "replace_work_title_recall_with_participant_perspective",
        "reason": "The replacement trades work-title recall for the source's substantive early participant perspective on softness, roughness, concern for the model, and desired sensation.",
    },
}


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v444 baseline drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v445-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v444.jsonl"
    build_baseline(base, inputs / "v444.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v445-observe-", dir=HERE) as tmp:
        directory = Path(tmp)
        out = directory / "out.jsonl"
        projection_report = directory / "out.report.json"
        datasets, reports = [], []
        for _ in (1, 2):
            build_projection(out, projection_report)
            datasets.append(out.read_bytes())
            reports.append(projection_report.read_bytes())
        rows = read_jsonl(out)
        return {
            "after": conservative_capacity(rows),
            "before": conservative_capacity(before),
            "dataset_equal": datasets[0] == datasets[1],
            "eval": json.loads(reports[0])["eval_fact_count"],
            "report_equal": reports[0] == reports[1],
            "rows": len(rows),
            "sha256": hashlib.sha256(datasets[0]).hexdigest(),
        }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v445-base-", dir=HERE) as tmp:
        directory = Path(tmp)
        base = directory / "v444.jsonl"
        build_baseline(base, directory / "v444.report.json")
        before = read_jsonl(base)
    by_fact = {row["fact_id"]: (index, row) for index, row in enumerate(before, 1)}

    audits, curations = [], []
    for audit_index, (fact_id, spec) in enumerate(EDIT_SPECS.items(), 1):
        active_index, row = by_fact[fact_id]
        if active_index != spec["active_index"]:
            raise ValueError(f"active index drift: {fact_id}")
        if (row["question"], row["answer"]) != (spec["question"], spec["answer"]):
            raise ValueError(f"active Q&A drift: {fact_id}")
        source_path = ROOT / spec["source_document"]
        if file_sha256(source_path) != spec["source_document_file_sha256"]:
            raise ValueError(f"source file drift: {fact_id}")
        document = json.loads(source_path.read_text())
        if document["url"] != row["url"] or len(document["text"]) != spec["source_document_chars"]:
            raise ValueError(f"source document metadata drift: {fact_id}")
        if row["evidence"] not in document["text"]:
            raise ValueError(f"evidence absent from full source snapshot: {fact_id}")

        rationale = spec["reason"]
        new_fact_id = stable_fact_id(spec["edited_question"], spec["edited_answer"])
        curations.append({
            "action": "edit", "answer": spec["edited_answer"],
            "document_sha256": row["document_sha256"], "evidence": row["evidence"],
            "evidence_url": row["url"], "expected_answer": row["answer"],
            "expected_question": row["question"], "fact_id": fact_id,
            "paraphrase_rationale": rationale, "question": spec["edited_question"],
            "reason": rationale, "reason_code": spec["reason_code"],
            "reviewed_at": REVIEWED_AT, "reviewer": REVIEWER,
            "source_lineage": {"raw_document": spec["source_document"]},
            "support_type": "manual_paraphrase",
        })
        audits.append({
            "active_answer": row["answer"], "active_index": active_index,
            "active_question": row["question"], "audit_index": audit_index,
            "decision": "edit", "document_sha256": row["document_sha256"],
            "edited_answer": spec["edited_answer"],
            "edited_fact_id": new_fact_id,
            "edited_question": spec["edited_question"],
            "fact_id": fact_id,
            "projection_lineage": {
                "baseline_rows": BASELINE_ROWS, "baseline_sha256": BASELINE_SHA256,
                "prior_context_merit_review": True,
            },
            "reason": rationale, "reason_code": spec["reason_code"],
            "review_class": "bounded_existing_qa_history_lineage_repair",
            "reviewed_at": REVIEWED_AT, "reviewer": REVIEWER,
            "schema": "context-merit-audit-v445", "source": row["source"],
            "source_document": spec["source_document"],
            "source_document_chars": spec["source_document_chars"],
            "source_document_file_sha256": spec["source_document_file_sha256"],
            "source_support": "manual_paraphrase",
            "support_evidence": row["evidence"],
            "support_evidence_sha256": text_sha256(row["evidence"]),
            "url": row["url"],
        })

    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    observation = observe(before)
    if not observation["dataset_equal"] or not observation["report_equal"]:
        raise ValueError(f"nondeterministic projection: {observation}")
    if (observation["rows"], observation["eval"]) != (518, 612):
        raise ValueError(f"projection count drift: {observation}")
    if (observation["before"] != EXPECTED_CAPACITY_BEFORE or
            observation["after"] != EXPECTED_CAPACITY_AFTER):
        raise ValueError(f"capacity drift: {observation}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observation["sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError(f"projection hash drift: {observation}")

    REPORT.write_text(json.dumps({
        "audit": {
            "by_decision": dict(Counter(row["decision"] for row in audits)),
            "by_review_class": dict(Counter(row["review_class"] for row in audits)),
            "path": portable(AUDIT), "rows": len(audits), "sha256": file_sha256(AUDIT),
        },
        "conservative_capacity": {
            "after": observation["after"], "before": observation["before"],
            "delta": {key: observation["after"][key] - observation["before"][key] for key in observation["before"]},
            "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
        },
        "domain_gap_notes": {
            "filled_by_this_pass": False,
            "remaining": [
                "Step-by-step knot and friction construction is thinner than conceptual and safety coverage.",
                "Upline routing, lockoffs, load sharing, and force-path reasoning remain sparse.",
                "Hardpoint assessment for trees, indoor structures, and ceiling-tension systems remains sparse.",
                "Rig-type selection and evaluation remain sparse.",
            ],
        },
        "isolated_build_projection": {
            "automated_projection_runs": 2, "new_additions_applied": 0,
            "output_rows": observation["rows"], "output_sha256": observation["sha256"],
            "repeat_dataset_byte_identical": observation["dataset_equal"],
            "repeat_projection_report_byte_identical": observation["report_equal"],
            "sealed_eval_fact_count_reported_by_tooling": observation["eval"],
        },
        "new_pending_curation": {
            "by_action": {"edit": 5}, "decisions": 5,
            "path": portable(CURATION), "sha256": file_sha256(CURATION),
        },
        "schema": "context-merit-audit-report-v445",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "source_snapshot_inventory": {
            "documents": len(EDIT_SPECS),
            "paths": [spec["source_document"] for spec in EDIT_SPECS.values()],
            "total_characters": sum(spec["source_document_chars"] for spec in EDIT_SPECS.values()),
        },
        "training_layer_contract": {
            "cleaned_markdown_site_corpora": "Distinct first-class future training artifacts; unfinished corpus output was neither read nor ingested by v445.",
            "derived_qa": "Distinct first-class training layer; v445 changes only existing sealed Q&A using attached evidence and pinned raw snapshots.",
            "deduplication_policy": "Do not collapse a cleaned Markdown document and its derived Q&A merely because they express the same source knowledge.",
            "provenance_requirement": "Link future derived Q&A to the cleaned source document through stable source URL and document or snapshot hashes.",
        },
        "v52_isolation": {"active_v52_inputs_modified": False, "candidate_status": "isolated_pending_not_promoted"},
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
