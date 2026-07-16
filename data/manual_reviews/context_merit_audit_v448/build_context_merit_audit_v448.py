#!/usr/bin/env python3
"""Curate ten existing anatomy, emergency, bottoming, and consent Q&As.

The only candidate base is sealed v447.  Four rows receive evidence-backed
precision repairs, one ambiguous consent fragment is quarantined, and five
operationally complete rows are retained.  This pass reviews no Rope365 or
Crash row, reads no pending corpus, and adds no new site-derived fact.
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
V447 = DATA / "manual_reviews/context_merit_audit_v447"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V447), str(V290)]

import build_context_merit_audit_v447 as previous
import build_context_merit_audit_v290 as core
from qa_quality import stable_fact_id


AUDIT = HERE / "context_merit_audit_v448.jsonl"
CURATION = HERE / "pending_curation_context_merit_v448.jsonl"
REPORT = HERE / "report_context_merit_v448.json"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
BASELINE_ROWS = 513
BASELINE_SHA256 = "fca887a130f04ed5c6f229eb81aaeccd1bf39aada0620cc9507a7d6b759e8518"
EXPECTED_OUTPUT_SHA256 = "9f817a4669172576284514aaf65c9d5f5389ab3cd1a80a204f4cd61abd175cd0"
EXPECTED_CAPACITY_BEFORE = {
    "conflict_units": 260,
    "equipment_material": 22,
    "resources_general": 78,
    "safety_consent": 85,
    "technique": 75,
}
EXPECTED_CAPACITY_AFTER = {
    "conflict_units": 260,
    "equipment_material": 22,
    "resources_general": 78,
    "safety_consent": 85,
    "technique": 75,
}
OWNER_SOURCE = previous.OWNER_SOURCE
OWNER_FACT_IDS = previous.OWNER_FACT_IDS
REVIEWED_AT = "2026-07-16"
REVIEWER = "codex-context-merit-audit-v448"

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
portable = previous.portable
conservative_capacity = previous.conservative_capacity


SPECS = {
    "fact-fd3a5c93a843ae2301df": {
        "active_index": 1,
        "question": "According to Clover’s guide for rope bottoms, what responsibility does the rigger have regarding the bottom’s fitness level and body type?",
        "answer": "The rigger should ensure that what they are doing is appropriate for the bottom’s fitness level and body type.",
        "decision": "edit",
        "edited_question": "How does Clover’s rope-bottom guide divide responsibility for adapting rope to a person’s fitness and body?",
        "edited_answer": "The guide says the rigger must adapt what they do to the bottom’s fitness and body type, while the bottom must communicate and give feedback before, during, and after rope.",
        "source_document": "data/raw/kinbakutoday_432c8adfc1abe686.json",
        "source_document_file_sha256": "7b5e201f9da39aa8553d16e0dcbc2d05ae354d96260e2eba798b6280d2933a9c",
        "source_document_chars": 15822,
        "reason_code": "complete_shared_responsibility_for_body_specific_adaptation",
        "reason": "The replacement removes an absolute-sounding rigger-only formulation and restores the source's paired responsibilities: body-specific adaptation by the rigger and timely feedback by the bottom.",
        "review_class": "bottoming_shared_responsibility_edit",
    },
    "fact-dd18828d1ac8ea14c12e": {
        "active_index": 47,
        "question": "How do Tethered Together’s collected 2025 rules route medical emergencies?",
        "answer": "Call 911 for a life-threatening emergency and notify event staff; request basic non-emergency first aid through the hospitality desk or First Aid Station.",
        "decision": "keep",
        "source_document": "data/raw/rope_resources_v1/tethered_together__cfcc0aab47acc802dd94.json",
        "source_document_file_sha256": "f2b5a5f45da0f89a8166d1bff0cece4fd9d81ebee802a26cade03b6057f0a4f8",
        "source_document_chars": 9347,
        "evidence": "First Aid Staff Members will be available to assist with any basic medical needs. In the event of a life-threatening emergency, call 911. Please notify event staff in the event of an emergency. DMs will be present at all parties and will serve as first responders in the event that a situation arises there.\nFirst Aid\nIf you need basic first aid during our hours of operation, please ask the hospitality desk to radio for non-emergency first aid if no one is at the First Aid Station.",
        "reason_code": "keep_precise_life_threatening_vs_basic_first_aid_routing",
        "reason": "The attributed event-policy row clearly separates life-threatening emergency escalation to 911 from basic non-emergency first-aid routing and includes staff notification.",
        "review_class": "precise_emergency_boundary_keep",
    },
    "fact-bf758c1e936b85b768a6": {
        "active_index": 150,
        "question": "In the Positive Action for Consent example, what should happen if the rope bottom does not return the agreed squeeze?",
        "answer": "The rigger should untie them and end the scene.",
        "decision": "keep",
        "source_document": "data/raw/anatomiestudio_9749de0eb1ff4ef3.json",
        "source_document_file_sha256": "2ce0036f18ab2fbb48a91470e56fa9105557b89744b329509898b7315cf9812d",
        "source_document_chars": 11525,
        "reason_code": "keep_no_response_means_release_boundary",
        "reason": "The row preserves the example's conservative pre-agreed boundary: no positive response means untying and ending the scene, with discussion deferred until afterward.",
        "review_class": "precise_emergency_boundary_keep",
    },
    "fact-f6befcfa1002eaeafbfc": {
        "active_index": 185,
        "question": "What conservative action does the rope-bottom guide recommend if any body part starts tingling and the cause is uncertain?",
        "answer": "Tell the rigger and come out of the tie; you can be tied again later.",
        "decision": "keep",
        "source_document": "data/raw/kinbakutoday_432c8adfc1abe686.json",
        "source_document_file_sha256": "7b5e201f9da39aa8553d16e0dcbc2d05ae354d96260e2eba798b6280d2933a9c",
        "source_document_chars": 15822,
        "reason_code": "keep_uncertain_tingling_conservative_release",
        "reason": "The row avoids diagnosing tingling as either nerve or circulation trouble and gives the source's conservative escalation boundary: communicate and leave the tie rather than testing a treatment shortcut.",
        "review_class": "nerve_circulation_uncertainty_keep",
    },
    "fact-c141cf8bb05560ebd308": {
        "active_index": 308,
        "question": "What responsibility does Anatomie Studio assign when a rope model becomes non-verbal?",
        "answer": "Safety remains a joint responsibility, but the rigger has the immediate responsibility to check in.",
        "decision": "keep",
        "source_document": "data/raw/anatomiestudio_9749de0eb1ff4ef3.json",
        "source_document_file_sha256": "2ce0036f18ab2fbb48a91470e56fa9105557b89744b329509898b7315cf9812d",
        "source_document_chars": 11525,
        "reason_code": "keep_joint_safety_and_immediate_nonverbal_checkin",
        "reason": "The row distinguishes ongoing joint responsibility from the rigger's immediate duty to initiate a check-in when the model cannot communicate verbally.",
        "review_class": "scene_management_keep",
    },
    "fact-de53adb46c47ecb4b18c": {
        "active_index": 428,
        "question": "Which three factors does RopeTopia identify in nerve problems, and how quickly should people respond?",
        "answer": "It identifies position, pressure, and duration, and says the person in rope should speak up immediately and the rigger should act immediately.",
        "decision": "edit",
        "edited_question": "What three variables does RopeTopia’s non-medical safety article associate with nerve risk, and what response does it recommend when a problem is suspected?",
        "edited_answer": "The article names position, pressure, and duration, and recommends that the tied person report symptoms immediately and the rigger respond immediately rather than waiting to determine the cause.",
        "source_document": "data/site_corpora/rope_topia/evidence_snapshots.json",
        "source_document_file_sha256": "8a1ccd882c37993fc5d236585e31feec35a79b11e951a01d3dea11290144e3c5",
        "source_document_chars": 4110,
        "source_kind": "rope_topia_evidence_snapshot",
        "source_resource_id": "nerve_and_circulation_problems",
        "reason_code": "qualify_nonmedical_nerve_variables_and_immediate_response",
        "reason": "The replacement keeps the position-pressure-duration model and immediate response boundary while explicitly attributing it to a non-medical safety article and omitting its unsupported absolute that nerve problems always arise instantly.",
        "review_class": "medical_uncertainty_edit",
    },
    "fact-0b9a4ae597884d8dedb5": {
        "active_index": 440,
        "question": "Why can upper cinches pose a nerve risk, according to the upper-limb surgeon quoted by Esinem?",
        "answer": "The upper-cinch area is a danger zone where the radial, median, and ulnar nerves can be compressed against the upper-arm bone.",
        "decision": "edit",
        "edited_question": "What anatomy and uncertainty does the upper-limb surgeon quoted by Esinem identify around upper cinches?",
        "edited_answer": "The surgeon describes the upper-cinch region as a danger zone where the radial, median, and ulnar nerves may be compressed against the humerus; he says the informal sensitivity test is not proven precise and does not justify deciding that upper cinches are safe.",
        "source_document": "data/raw/esinem_0b1850ee9a40c337.json",
        "source_document_file_sha256": "b75e93a99fe3b6309cacc73794c53d260492aeb759cbfbb9a17f49b78d1635a4",
        "source_document_chars": 5964,
        "evidence_layout": "noncontiguous_exact_paragraphs",
        "reason_code": "complete_upper_cinch_anatomy_with_test_uncertainty",
        "reason": "The replacement retains the surgeon-attributed nerve anatomy while restoring his explicit uncertainty about the informal sensitivity test and its inability to establish that upper cinches are safe.",
        "review_class": "medical_uncertainty_edit",
    },
    "fact-18a6771a5b827a23cbde": {
        "active_index": 449,
        "question": "Why does Anatomie Studio say a verbal “yes” may not be sufficient evidence of comfort?",
        "answer": "Someone can say yes while still feeling uncomfortable, so partners should also pay attention to nonverbal cues.",
        "decision": "drop",
        "source_document": "data/raw/anatomiestudio_451ac66001188a42.json",
        "source_document_file_sha256": "5572a9e872549927462b7314d0ae28ee4f12a18e43b4f225e2cf5c0cfb0662c3",
        "source_document_chars": 7446,
        "reason_code": "quarantine_ambiguous_verbal_yes_fragment_with_operational_survivors",
        "reason": "The one-line row can be misread as permission to override explicit speech based on subjective cues and omits what to do next. Active facts ea0053be5b3e79d7b7d1 and 096e13f59527794a75e4 retain the operational rules to check in, stop if doubt remains, and never treat freezing or silence as consent.",
        "review_class": "ambiguous_consent_fragment_drop",
        "redundant_survivors": ["fact-ea0053be5b3e79d7b7d1", "fact-096e13f59527794a75e4"],
    },
    "fact-3f20ab03978caecf9f40": {
        "active_index": 458,
        "question": "Why does Esinem recommend extra safety space in a gote wrist binding?",
        "answer": "Extra slack reduces pressure as the harness loads and lets the tied person reposition their wrists if discomfort, circulation trouble, or nerve pressure develops.",
        "decision": "edit",
        "edited_question": "What two preventive functions does Esinem give for extra slack in a gote wrist binding?",
        "edited_answer": "He says extra slack reduces pressure that can develop as a suspended harness loads and leaves room for the tied person to change wrist position if discomfort, circulation issues, or nerve pressure appears.",
        "source_document": "data/raw/esinem_c93603a2c7d48255.json",
        "source_document_file_sha256": "e9e1e785b0a7d417f37a0460314056e05bd455100a7b928d2532aa0e12e6a335",
        "source_document_chars": 7616,
        "evidence_layout": "noncontiguous_exact_paragraphs",
        "reason_code": "frame_gote_wrist_slack_as_prevention_not_treatment",
        "reason": "The replacement makes the two source-supported preventive functions explicit and avoids presenting wrist repositioning as a diagnosis or treatment for nerve or circulation symptoms.",
        "review_class": "injury_prevention_edit",
    },
    "fact-51c482bc9acd697bd59a": {
        "active_index": 492,
        "question": "Why does Tethered Together recommend avoiding mid-scene renegotiation when someone is in subspace or not clear-minded?",
        "answer": "Their apparent agreement may not amount to informed consent in that state.",
        "decision": "keep",
        "source_document": "data/raw/rope_resources_v1/tethered_together__d5594a12694c5468f698.json",
        "source_document_file_sha256": "f711813799ffd98933650483b5efa050e057b75ae79d956f93e59679c7e25d71",
        "source_document_chars": 3709,
        "reason_code": "keep_state_dependent_midscene_consent_boundary",
        "reason": "The attributed policy row gives a precise capacity boundary: apparent agreement during subspace or impaired clarity may not constitute informed consent, so renegotiation should be avoided.",
        "review_class": "scene_management_keep",
    },
}


CONSENT_SURVIVORS = {
    "fact-ea0053be5b3e79d7b7d1": {
        "question": "What does Anatomie Studio recommend if someone doubts a partner's verbal or physical enthusiasm?",
        "answer": "Anatomie Studio recommends checking in and not proceeding if doubt remains.",
    },
    "fact-096e13f59527794a75e4": {
        "question": "How does Anatomie Studio say silence, freezing, indecision, or mixed messages should be interpreted during a rope interaction?",
        "answer": "Anatomie Studio says none of these signals should be treated as a yes; consent must be clearly and mutually communicated.",
    },
}


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v447 baseline drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v448-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v447.jsonl"
    build_baseline(base, inputs / "v447.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def validate_source(spec: dict, row: dict, evidence: str) -> None:
    source_path = ROOT / spec["source_document"]
    if file_sha256(source_path) != spec["source_document_file_sha256"]:
        raise ValueError(f"source file drift: {row['fact_id']}")
    document = json.loads(source_path.read_text())
    if spec.get("source_kind") == "rope_topia_evidence_snapshot":
        entry = next(
            item for item in document["archive_pages"]
            if item["resource_id"] == spec["source_resource_id"]
        )
        if (entry["canonical_url"] != row["url"] or
                entry["exact_qa_evidence"] != evidence or
                entry["archive"]["extracted_body_chars_manually_reviewed"] != spec["source_document_chars"]):
            raise ValueError(f"RopeTopia evidence snapshot drift: {row['fact_id']}")
        return
    if document["url"] != row["url"] or len(document["text"]) != spec["source_document_chars"]:
        raise ValueError(f"source document metadata drift: {row['fact_id']}")
    if spec.get("evidence_layout") == "noncontiguous_exact_paragraphs":
        paragraphs = evidence.split("\n")
        if len(paragraphs) < 2 or any(paragraph not in document["text"] for paragraph in paragraphs):
            raise ValueError(f"exact evidence paragraph absent from full source snapshot: {row['fact_id']}")
    elif evidence not in document["text"]:
        raise ValueError(f"evidence absent from full source snapshot: {row['fact_id']}")


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v448-observe-", dir=HERE) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v448-base-", dir=HERE) as tmp:
        directory = Path(tmp)
        base = directory / "v447.jsonl"
        build_baseline(base, directory / "v447.report.json")
        before = read_jsonl(base)
    by_fact = {row["fact_id"]: (index, row) for index, row in enumerate(before, 1)}

    for fact_id, expected in CONSENT_SURVIVORS.items():
        row = by_fact[fact_id][1]
        if (row["question"], row["answer"]) != (expected["question"], expected["answer"]):
            raise ValueError(f"consent survivor drift: {fact_id}")

    audits, curations = [], []
    for audit_index, (fact_id, spec) in enumerate(SPECS.items(), 1):
        active_index, row = by_fact[fact_id]
        if active_index != spec["active_index"]:
            raise ValueError(f"active index drift: {fact_id}")
        if (row["question"], row["answer"]) != (spec["question"], spec["answer"]):
            raise ValueError(f"active Q&A drift: {fact_id}")
        evidence = spec.get("evidence", row["evidence"])
        validate_source(spec, row, evidence)

        decision = spec["decision"]
        if decision == "edit":
            curations.append({
                "action": "edit", "answer": spec["edited_answer"],
                "document_sha256": row["document_sha256"], "evidence": evidence,
                "evidence_url": row["url"], "expected_answer": row["answer"],
                "expected_question": row["question"], "fact_id": fact_id,
                "paraphrase_rationale": spec["reason"], "question": spec["edited_question"],
                "reason": spec["reason"], "reason_code": spec["reason_code"],
                "reviewed_at": REVIEWED_AT, "reviewer": REVIEWER,
                "source_lineage": {"source_document": spec["source_document"]},
                "support_type": "manual_paraphrase",
            })
        elif decision == "drop":
            curations.append({
                "action": "drop", "expected_answer": row["answer"],
                "expected_question": row["question"], "fact_id": fact_id,
                "reason": spec["reason"], "reason_code": spec["reason_code"],
                "reviewed_at": REVIEWED_AT, "reviewer": REVIEWER,
            })
        elif decision != "keep":
            raise ValueError(f"unsupported decision: {decision}")

        audit = {
            "active_answer": row["answer"], "active_index": active_index,
            "active_question": row["question"], "audit_index": audit_index,
            "decision": decision, "document_sha256": row["document_sha256"],
            "fact_id": fact_id,
            "projection_lineage": {
                "baseline_rows": BASELINE_ROWS, "baseline_sha256": BASELINE_SHA256,
                "prior_context_merit_review": True,
            },
            "reason": spec["reason"], "reason_code": spec["reason_code"],
            "review_class": spec["review_class"], "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER, "schema": "context-merit-audit-v448",
            "source": row["source"], "source_document": spec["source_document"],
            "source_document_chars": spec["source_document_chars"],
            "source_document_file_sha256": spec["source_document_file_sha256"],
            "source_support": "manual_paraphrase" if decision == "edit" else "full_snapshot_review",
            "support_evidence": evidence,
            "support_evidence_layout": spec.get("evidence_layout", "contiguous_exact_span"),
            "support_evidence_sha256": text_sha256(evidence), "url": row["url"],
        }
        if decision == "edit":
            audit.update({
                "edited_answer": spec["edited_answer"],
                "edited_fact_id": stable_fact_id(spec["edited_question"], spec["edited_answer"]),
                "edited_question": spec["edited_question"],
            })
        if spec.get("redundant_survivors"):
            audit["redundant_survivors"] = spec["redundant_survivors"]
        audits.append(audit)

    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    observation = observe(before)
    if not observation["dataset_equal"] or not observation["report_equal"]:
        raise ValueError(f"nondeterministic projection: {observation}")
    if (observation["rows"], observation["eval"]) != (512, 612):
        raise ValueError(f"projection count drift: {observation}")
    if observation["before"] != EXPECTED_CAPACITY_BEFORE:
        raise ValueError(f"baseline capacity drift: {observation}")
    if EXPECTED_CAPACITY_AFTER is not None and observation["after"] != EXPECTED_CAPACITY_AFTER:
        raise ValueError(f"candidate capacity drift: {observation}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observation["sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError(f"projection hash drift: {observation}")

    unique_sources = {spec["source_document"]: spec for spec in SPECS.values()}
    curation_counts = Counter(row["action"] for row in curations)
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
        "curation_outcome": {
            "ambiguous_consent_rows_quarantined": 1,
            "bottoming_shared_responsibility_repairs": 1,
            "injury_prevention_repairs": 1,
            "medical_uncertainty_repairs": 2,
            "operational_rows_kept": 5,
        },
        "isolated_build_projection": {
            "automated_projection_runs": 2, "new_additions_applied": 0,
            "output_rows": observation["rows"], "output_sha256": observation["sha256"],
            "repeat_dataset_byte_identical": observation["dataset_equal"],
            "repeat_projection_report_byte_identical": observation["report_equal"],
            "sealed_eval_fact_count_reported_by_tooling": observation["eval"],
        },
        "new_pending_curation": {
            "by_action": dict(curation_counts), "decisions": len(curations),
            "path": portable(CURATION), "sha256": file_sha256(CURATION),
        },
        "source_boundary": {
            "crash_rows_reviewed": 0, "pending_or_new_corpus_inputs": 0,
            "rope365_rows_reviewed": 0,
        },
        "source_snapshot_inventory": {
            "documents": len(unique_sources), "reviewed_rows": len(SPECS),
            "paths": sorted(unique_sources),
            "total_unique_characters": sum(spec["source_document_chars"] for spec in unique_sources.values()),
        },
        "surviving_consent_guidance": {
            "dropped_fact": "fact-18a6771a5b827a23cbde",
            "surviving_facts": sorted(CONSENT_SURVIVORS),
        },
        "schema": "context-merit-audit-report-v448",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "training_layer_contract": {
            "cleaned_markdown_site_corpora": "Distinct first-class future training artifacts; no corpus output was ingested by v448.",
            "derived_qa": "Distinct first-class training layer; v448 changes only existing sealed Q&A using attached evidence and pinned source snapshots.",
            "deduplication_policy": "Do not collapse a cleaned Markdown document and its derived Q&A merely because they express the same source knowledge.",
            "provenance_requirement": "Link future derived Q&A to the cleaned source document through stable source URL and document or snapshot hashes.",
        },
        "v52_isolation": {"active_v52_inputs_modified": False, "candidate_status": "isolated_pending_not_promoted"},
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
