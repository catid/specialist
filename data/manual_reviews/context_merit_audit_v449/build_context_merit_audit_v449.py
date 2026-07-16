#!/usr/bin/env python3
"""Curate ten existing equipment, upline, hardpoint, and cutter Q&As.

The only candidate base is sealed v448. Six rows receive evidence-backed
scope or attribution repairs, three low-merit commerce/load-claim rows are
quarantined, and one manufacturer-manual row is retained. No pending corpus,
Crash corpus, or new site-derived fact is ingested.
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
V448 = DATA / "manual_reviews/context_merit_audit_v448"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V448), str(V290)]

import build_context_merit_audit_v448 as previous
import build_context_merit_audit_v290 as core
from qa_quality import stable_fact_id


AUDIT = HERE / "context_merit_audit_v449.jsonl"
CURATION = HERE / "pending_curation_context_merit_v449.jsonl"
REPORT = HERE / "report_context_merit_v449.json"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
BASELINE_ROWS = 512
BASELINE_SHA256 = "9f817a4669172576284514aaf65c9d5f5389ab3cd1a80a204f4cd61abd175cd0"
EXPECTED_OUTPUT_SHA256 = "a94794c4564082bf43c64d4cd92203f4319e680e388e40a087511ab57965b950"
EXPECTED_CAPACITY_BEFORE = {
    "conflict_units": 260,
    "equipment_material": 22,
    "resources_general": 78,
    "safety_consent": 85,
    "technique": 75,
}
EXPECTED_CAPACITY_AFTER = {
    "conflict_units": 260,
    "equipment_material": 21,
    "resources_general": 78,
    "safety_consent": 85,
    "technique": 76,
}
OWNER_SOURCE = previous.OWNER_SOURCE
OWNER_FACT_IDS = previous.OWNER_FACT_IDS
REVIEWED_AT = "2026-07-16"
REVIEWER = "codex-context-merit-audit-v449"

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
portable = previous.portable
conservative_capacity = previous.conservative_capacity


SPECS = {
    "fact-96558de2a3b8c8236060": {
        "active_index": 172,
        "question": "What carabiner profile does Rope365's equipment primer recommend for bondage rope?",
        "answer": "Rope365’s equipment primer recommends a round profile and advises avoiding sharp D-shapes and I-beam or skeletonized designs.",
        "decision": "edit",
        "edited_question": "According to Rope365’s equipment primer, what carabiner contact geometry does it prefer for bondage rope, and what does it advise avoiding?",
        "edited_answer": "The primer prefers a smooth, round rope-contact profile and advises avoiding sharp angles and I-beam or skeletonized sections that can concentrate load on narrow bondage rope.",
        "source_document": "data/raw/rope_resources_v1/rope365__917bc6534539143cae83.json",
        "source_document_file_sha256": "dec9f9d59ad0af3cada607c35feb18ac6e0f7770b6f5ff0cdb69e5eacfe87561",
        "source_document_chars": 17241,
        "claim_scope": "instructional contact-geometry judgment, not a hardware rating",
        "reason_code": "scope_carabiner_advice_to_rope_contact_geometry",
        "reason": "The replacement preserves the primer's narrow-rope contact-geometry advice while avoiding a blanket implication that every D-shaped carabiner is unsuitable regardless of its rope-bearing surface.",
        "review_class": "rigging_contact_geometry_edit",
    },
    "fact-d1f2499dd9d5ade6aac3": {
        "active_index": 188,
        "question": "What construction and static breaking point does Chromaknotz state for its rope?",
        "answer": "It describes the rope as 1/4-inch double-braid nylon with a vendor-stated single-strand static breaking point of 1,280; the page does not specify a unit for that figure.",
        "decision": "drop",
        "source_document": "data/raw/chromaknotz_questions_answers_20260714.json",
        "source_document_file_sha256": "cf9142aad40d4e0d233fa5285bb072006191a1235a9570a6bb8e91bc1a1593c7",
        "source_document_chars": 910,
        "claim_scope": "vendor claim with an unspecified unit and no usable working-load context",
        "reason_code": "quarantine_ununitized_vendor_breaking_point",
        "reason": "A bare vendor number with no unit cannot support a breaking-load or working-load inference. The owner resource fact 7b9730efad5e1f58d941 preserves the supplier URL, rope construction, care guidance, and the missing-unit caveat without presenting this as an actionable rating.",
        "review_class": "ununitized_vendor_load_claim_drop",
        "redundant_survivors": ["fact-7b9730efad5e1f58d941"],
    },
    "fact-b7799df10cc9d701bf7b": {
        "active_index": 195,
        "question": "What cutting and extraction features does the Amazon listing describe on the Leatherman Raptor Rescue, and how does its holster support quick access?",
        "answer": "It lists emergency or trauma shears, strap and ring cutters, and a carbide glass breaker; the shears lock in the holster so they can be carried upside-down on a belt for quick access.",
        "decision": "drop",
        "source_document": "data/raw/amazon_leatherman_raptor_rescue_20260714.json",
        "source_document_file_sha256": "81dcb9ddeb5e6514b9a8df71660458bc37dda9337c483533782e062c19d4e6d2",
        "source_document_chars": 436,
        "claim_scope": "volatile commerce listing and brand-feature trivia",
        "reason_code": "quarantine_volatile_cutter_product_features",
        "reason": "The feature inventory is commerce-specific and does not establish whether the tool cuts the rope actually in use. Owner resource fact 7997bd2f70052a765e40 keeps the requested listing and the actionable readiness rule to test cutters on the actual rope beforehand.",
        "review_class": "volatile_product_feature_drop",
        "redundant_survivors": ["fact-7997bd2f70052a765e40"],
    },
    "fact-f97e40f1614c223532a0": {
        "active_index": 235,
        "question": "What does Shibari Study's free Uplines and Lock-offs 101 course cover?",
        "answer": "It teaches attaching uplines to limbs or harnesses and multiple suspension-line lock-offs, with related topics including safety, weight handling, basic anchoring, bamboo and carabiner lock-offs, and Y-hanger or epsilon configurations.",
        "decision": "edit",
        "edited_question": "What does Shibari Study’s listing say its free Uplines and Lock-offs 101 course teaches?",
        "edited_answer": "The listing says the course teaches attaching uplines to limbs or harnesses and several lock-off methods for suspension lines.",
        "evidence": "Learn how to attach uplines to limbs or harnesses and master various lock-off methods for suspension lines.",
        "source_document": "data/rope_resource_manual_v1.jsonl",
        "source_document_file_sha256": "caee3a972ee70f6e8f23cd3de561bc673c54705e64df58cfc0c867e75995c3bc",
        "source_document_chars": 11044,
        "source_kind": "manual_resource_jsonl",
        "source_fact_id": "fact-e27da1eeb2a2294df669",
        "claim_scope": "course-listing description, not validated rigging instruction",
        "reason_code": "reduce_upline_course_claim_to_exact_listing_scope",
        "reason": "The replacement removes an expanded catalog of related topics and duplicated evidence that are not supported by the cited manual-fact row, retaining only the exact course-listing claim.",
        "review_class": "course_scope_provenance_edit",
    },
    "fact-3516b4ec53a9b638759f": {
        "active_index": 247,
        "question": "What emergency rope-cutting equipment does Clover’s guide say a rigger should have?",
        "answer": "A rigger should have EMT shears or rescue hooks as a reliable way to cut rope in an emergency.",
        "decision": "edit",
        "edited_question": "What emergency-cutting preparation does Clover’s rope-bottom guide recommend before tying?",
        "edited_answer": "The guide recommends confirming that the rigger has a way to cut rope in an emergency, such as EMT shears or a rescue hook; it says the bottom can bring their own shears for the rigger to use.",
        "source_document": "data/raw/kinbakutoday_432c8adfc1abe686.json",
        "source_document_file_sha256": "7b5e201f9da39aa8553d16e0dcbc2d05ae354d96260e2eba798b6280d2933a9c",
        "source_document_chars": 15822,
        "claim_scope": "source-attributed emergency-equipment readiness recommendation",
        "reason_code": "frame_cutters_as_pre_scene_readiness_not_guarantee",
        "reason": "The replacement keeps the concrete cutter examples but frames them as a pre-scene availability check, avoiding an unsupported implication that naming a tool alone guarantees reliable cutting performance.",
        "review_class": "emergency_cutting_readiness_edit",
    },
    "fact-630d49608d80d3502fd4": {
        "active_index": 281,
        "question": "What material does Subspace Designs use for custom suspension rings, and what tradeoff does it claim?",
        "answer": "It uses certified aerospace-grade 6061-T6 aluminum, which it describes as combining exceptional strength with minimal weight.",
        "decision": "drop",
        "source_document": "data/raw/subspace_designs_custom_order_deposit_20260714.json",
        "source_document_file_sha256": "f9d46af9f2634aecd7f7ccaf73b5f8628ce9b6bbcdd2f1c53e8dcae96e44a656",
        "source_document_chars": 542,
        "claim_scope": "vendor material and qualitative strength claim without load evidence",
        "reason_code": "quarantine_material_only_suspension_strength_implication",
        "reason": "An alloy designation and qualitative 'exceptional strength' language do not establish a working load or suitability for a suspension setup. Owner resource fact 5dcda16beacf5b48a099 preserves the recommendation, material, customization, and explicit independent-verification caveat.",
        "review_class": "material_only_load_implication_drop",
        "redundant_survivors": ["fact-5dcda16beacf5b48a099"],
    },
    "fact-7451ab73e98769310440": {
        "active_index": 296,
        "question": "What portability, load, and space specifications does Tetruss state for its standard rig?",
        "answer": "Tetruss says its aluminum-alloy frame weighs just over 20 pounds, supports up to 300 pounds of dynamic/live load, assembles in minutes, fits an eight-foot ceiling, and has an eight-foot triangular footprint.",
        "decision": "edit",
        "edited_question": "Which planning specifications does Tetruss advertise for its standard portable rig?",
        "edited_answer": "The vendor lists an aluminum-alloy frame weighing just over 20 pounds, an eight-foot triangular footprint for an eight-foot ceiling, assembly in minutes, and support for 300 pounds of dynamic or live load; these are manufacturer claims for the standard rig.",
        "source_document": "data/raw/tetruss_standard_rig_20260714.json",
        "source_document_file_sha256": "40acc131e485c3746d3950ed1529b90b37b2f6380478ab909580ad5566c591c1",
        "source_document_chars": 926,
        "claim_scope": "manufacturer product-page planning and load claims",
        "reason_code": "label_tetruss_load_and_space_values_as_vendor_claims",
        "reason": "The replacement keeps useful planning figures while explicitly identifying the product page as the manufacturer claim and limiting the figures to the standard rig rather than unknown or modified configurations.",
        "review_class": "vendor_frame_specification_edit",
    },
    "fact-ff959dc94af0a7cc2013": {
        "active_index": 306,
        "question": "What quality checks does RopeTopia recommend for emergency shears?",
        "answer": "It recommends shears that do not bend easily, whose handles do not slide past each other, whose blades do not flex apart, and whose rivet is tight enough to prevent blade play.",
        "decision": "edit",
        "edited_question": "What preliminary construction checks does RopeTopia recommend when assessing emergency shears?",
        "edited_answer": "It recommends checking that the shears resist bending, the handles do not slide past one another, the blades do not flex apart, and the rivet has no play that separates the blades.",
        "source_document": "data/site_corpora/rope_topia/evidence_snapshots.json",
        "source_document_file_sha256": "8a1ccd882c37993fc5d236585e31feec35a79b11e951a01d3dea11290144e3c5",
        "source_document_chars": 5348,
        "source_kind": "rope_topia_evidence_snapshot",
        "source_resource_id": "safety_cutters",
        "claim_scope": "non-medical preliminary construction inspection",
        "reason_code": "scope_shear_quality_checks_as_preliminary_inspection",
        "reason": "The replacement retains every source-supported physical check while calling them preliminary construction checks, avoiding an implication that visual and hand inspection alone proves cutting performance.",
        "review_class": "emergency_shear_construction_edit",
    },
    "fact-2640eaa6aabb5f80411e": {
        "active_index": 321,
        "question": "What safety inspection does Rope365 recommend for a center-weighted tripod, and why?",
        "answer": "Rope365 recommends inspecting a center-weighted tripod’s lashings or welds because rotational force can cause shearing.",
        "decision": "edit",
        "edited_question": "What tripod-specific failure mode and inspection does Rope365’s hardpoint article identify?",
        "edited_answer": "It says a center-weighted load can create rotational force that shears the top connection, so the tripod’s lashings or welds should be inspected regularly.",
        "source_document": "data/raw/rope_resources_v1/rope365__153b0f6d1f2932a77626.json",
        "source_document_file_sha256": "e07d2b4391907054633d81e9b9f76df55237c778b5e4440d19aff55440f58e73",
        "source_document_chars": 7268,
        "claim_scope": "context-dependent tripod failure-mode judgment, not anchor approval",
        "reason_code": "name_tripod_failure_mode_without_implying_structure_approval",
        "reason": "The replacement names the center-connection failure mode and limited inspection action without implying that an unknown tripod, lashing, weld, anchor, tree, or frame is safe after inspection.",
        "review_class": "tripod_failure_mode_edit",
    },
    "fact-11d008e884b0aadc723d": {
        "active_index": 368,
        "question": "What use limits and load values does X-POLE's 2024 A-FRAME manual state?",
        "answer": "It describes the frame for home and light-studio aerial fitness, not professional performance or dynamic drops; it reports a 700-kilogram minimum breaking load and says its low-level training use applies a 5:1 working-load value of 140 kilograms (309 pounds).",
        "decision": "keep",
        "source_document": "data/raw/xpole_a_frame_manual_v2_2024_20260714.json",
        "source_document_file_sha256": "1e401fd75e11d6537e1987169e8b78e39301195f48567b08930aa15966b0294f",
        "source_document_chars": 1458,
        "evidence_layout": "noncontiguous_exact_paragraphs",
        "claim_scope": "manufacturer manual with explicit use limits, breaking load, safety factor, and working load",
        "reason_code": "keep_manual_rated_values_with_use_limits_and_safety_factor",
        "reason": "The row attributes values to the manufacturer manual, distinguishes minimum breaking load from a 5:1 working-load value, and retains the manual's exclusion of professional performance and dynamic drops.",
        "review_class": "rated_manual_scope_keep",
    },
}


DROP_SURVIVORS = {
    "fact-7b9730efad5e1f58d941": "https://chromaknotz.square.site/",
    "fact-7997bd2f70052a765e40": "https://www.amazon.com/dp/B0195QI218",
    "fact-5dcda16beacf5b48a099": "https://www.subspacedesigns.shop/",
}


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v448 baseline drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v449-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v448.jsonl"
    build_baseline(base, inputs / "v448.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def validate_source(spec: dict, row: dict, evidence: str) -> None:
    source_path = ROOT / spec["source_document"]
    if file_sha256(source_path) != spec["source_document_file_sha256"]:
        raise ValueError(f"source file drift: {row['fact_id']}")
    if spec.get("source_kind") == "manual_resource_jsonl":
        if len(source_path.read_text()) != spec["source_document_chars"]:
            raise ValueError(f"manual resource artifact length drift: {row['fact_id']}")
        entry = next(item for item in read_jsonl(source_path) if item["fact_id"] == spec["source_fact_id"])
        if entry["evidence_url"] != row["url"] or entry["evidence"] != evidence:
            raise ValueError(f"manual resource evidence drift: {row['fact_id']}")
        return
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
            raise ValueError(f"exact evidence paragraph absent from source: {row['fact_id']}")
    elif evidence not in document["text"]:
        raise ValueError(f"evidence absent from full source snapshot: {row['fact_id']}")


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v449-observe-", dir=HERE) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v449-base-", dir=HERE) as tmp:
        directory = Path(tmp)
        base = directory / "v448.jsonl"
        build_baseline(base, directory / "v448.report.json")
        before = read_jsonl(base)
    by_fact = {row["fact_id"]: (index, row) for index, row in enumerate(before, 1)}

    for fact_id, url in DROP_SURVIVORS.items():
        if by_fact[fact_id][1]["url"] != url:
            raise ValueError(f"drop survivor drift: {fact_id}")

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
            "claim_scope": spec["claim_scope"], "decision": decision,
            "document_sha256": row["document_sha256"], "fact_id": fact_id,
            "projection_lineage": {
                "baseline_rows": BASELINE_ROWS, "baseline_sha256": BASELINE_SHA256,
                "prior_context_merit_review": True,
            },
            "reason": spec["reason"], "reason_code": spec["reason_code"],
            "review_class": spec["review_class"], "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER, "schema": "context-merit-audit-v449",
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
    if (observation["rows"], observation["eval"]) != (509, 612):
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
            "by_claim_scope": dict(Counter(row["claim_scope"] for row in audits)),
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
            "claim_scope_and_attribution_repairs": 6,
            "low_merit_or_unsafe_claim_rows_quarantined": 3,
            "rated_manual_rows_kept": 1,
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
            "protected_eval_heldout_ood_shadow_probe_rows_opened": 0,
        },
        "source_snapshot_inventory": {
            "documents": len(unique_sources), "reviewed_rows": len(SPECS),
            "paths": sorted(unique_sources),
            "total_unique_characters": sum(spec["source_document_chars"] for spec in unique_sources.values()),
        },
        "surviving_owner_resource_facts": {
            "dropped_direct_facts": sorted(
                fact_id for fact_id, spec in SPECS.items() if spec["decision"] == "drop"
            ),
            "surviving_facts": sorted(DROP_SURVIVORS),
        },
        "schema": "context-merit-audit-report-v449",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "training_layer_contract": {
            "cleaned_markdown_site_corpora": "Distinct first-class future training artifacts; no corpus output was ingested by v449.",
            "derived_qa": "Distinct first-class training layer; v449 changes only existing sealed Q&A using attached evidence and pinned source snapshots.",
            "deduplication_policy": "Do not collapse a cleaned Markdown document and its derived Q&A merely because they express the same source knowledge.",
            "provenance_requirement": "Link future derived Q&A to the cleaned source document through stable source URL and document or snapshot hashes.",
        },
        "v52_isolation": {"active_v52_inputs_modified": False, "candidate_status": "isolated_pending_not_promoted"},
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
