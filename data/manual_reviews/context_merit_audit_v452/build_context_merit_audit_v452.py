#!/usr/bin/env python3
"""Curate ten existing rope-material and maintenance Q&As.

The only candidate base is sealed v451. Five rows receive evidence-backed
scope or natural-question repairs, three redundant commercial/vendor rows are
quarantined, and two operational maintenance rows are retained. No new corpus
or protected evaluation artifact is read or changed.
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
V451 = DATA / "manual_reviews/context_merit_audit_v451"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V451), str(V290)]

import build_context_merit_audit_v451 as previous
import build_context_merit_audit_v290 as core
from qa_quality import stable_fact_id


AUDIT = HERE / "context_merit_audit_v452.jsonl"
CURATION = HERE / "pending_curation_context_merit_v452.jsonl"
REPORT = HERE / "report_context_merit_v452.json"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
BASELINE_ROWS = 504
BASELINE_SHA256 = "d91cbfbc0c4865e57e84f9a187402f39573b2039d075fa4159e7262023ba3ebe"
EXPECTED_OUTPUT_SHA256 = "78af3c99df92fcfba6451803fa04ba9b9d4a4c781c4f5a99b0acef0496807195"
EXPECTED_CAPACITY_BEFORE = {
    "conflict_units": 258,
    "equipment_material": 21,
    "resources_general": 77,
    "safety_consent": 84,
    "technique": 76,
}
EXPECTED_CAPACITY_AFTER = {
    "conflict_units": 258,
    "equipment_material": 21,
    "resources_general": 77,
    "safety_consent": 84,
    "technique": 76,
}
OWNER_SOURCE = previous.OWNER_SOURCE
OWNER_FACT_IDS = previous.OWNER_FACT_IDS
REVIEWED_AT = "2026-07-16"
REVIEWER = "codex-context-merit-audit-v452"

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
portable = previous.portable
conservative_capacity = previous.conservative_capacity


SPECS = {
    "fact-f2eaf9d6f97b647d504c": {
        "active_index": 53,
        "question": "How does Chromaknotz say its nylon rope should be washed and dried, and what caution does it give about fabric softener?",
        "answer": "Machine-wash it on the gentle cycle in cold water inside a mesh bag, then line-dry it out of direct sunlight and never use a clothes dryer; Chromaknotz says a drop of fabric softener can add scent and softness, but too much makes the rope more slippery.",
        "decision": "drop",
        "source_document": "data/raw/chromaknotz_questions_answers_20260714.json",
        "source_document_file_sha256": "cf9142aad40d4e0d233fa5285bb072006191a1235a9570a6bb8e91bc1a1593c7",
        "source_document_chars": 910,
        "material_scope": "vendor-specific care plus an optional friction-altering additive",
        "reason_code": "quarantine_redundant_fabric_softener_vendor_advice",
        "reason": "The direct vendor row adds an optional fabric-softener suggestion that changes rope slipperiness. Owner fact 7b9730efad5e1f58d941 retains the requested supplier URL and the core cold-wash, mesh-bag, and line-dry instructions without that additive advice.",
        "review_class": "vendor_additive_advice_drop",
        "redundant_survivors": ["fact-7b9730efad5e1f58d941"],
    },
    "fact-ead3a4a26fcd40415f3e": {
        "active_index": 57,
        "question": "How does deGiotto describe the length, material, end finish, and delivered condition of its 30-foot Jute Shibari Rope?",
        "answer": "deGiotto describes the product as 30 feet of premium 6 mm jute bondage rope finished with overhand knots and delivered conditioned, buttered, and ready to use.",
        "decision": "drop",
        "source_document": "data/raw/degiotto_jute_30_product_20260714.json",
        "source_document_file_sha256": "11febc534bee304c8dbd813c171efc284a71ec8ce2aa8e8557d1be13ce5ebc07",
        "source_document_chars": 444,
        "material_scope": "single product listing and promotional condition labels",
        "reason_code": "quarantine_degiotto_product_listing_trivia",
        "reason": "The row is a single product's size and marketing description rather than durable material or care knowledge. Owner fact 697f713f2af3f44c8639 preserves the requested De Giotto resource and its scoped jute-care guidance.",
        "review_class": "product_listing_trivia_drop",
        "redundant_survivors": ["fact-697f713f2af3f44c8639"],
    },
    "fact-48ddaa99bfb92a968b7d": {
        "active_index": 58,
        "question": "How does deGiotto say to dry its wet jute rope, and what problem is the setup meant to prevent?",
        "answer": "After dripping, stretch the rope fully, secure its ends with twine so it is firm and at least four feet high, hang about one pound at the center, and let it dry without retightening; this counters wet jute fibers curling into shorter, fatter, mushy rope.",
        "decision": "edit",
        "edited_question": "What drying method does deGiotto prescribe specifically for its own washed jute, and what change is it intended to counter?",
        "edited_answer": "For its own jute, the vendor says to let dripping stop, fully extend the rope, secure the ends with twine under light tension, hang about one pound at the center, and leave it until completely dry; the method is intended to counter wet fibers curling into shorter, thicker rope.",
        "source_document": "data/raw/degiotto_care_maintenance_20260714.json",
        "source_document_file_sha256": "1c73eb6781cf22b8f2c8a835be9bef262de7168b2b7e99c2297ac37e9bf021fe",
        "source_document_chars": 3609,
        "material_scope": "manufacturer-specific jute drying process, not universal jute guidance",
        "reason_code": "scope_degiotto_jute_drying_to_its_own_rope",
        "reason": "The replacement restores the care page's explicit scope: construction, material quality, and dyes vary, so this method is the manufacturer's instruction for its own jute rather than a universal recipe.",
        "review_class": "manufacturer_specific_drying_edit",
    },
    "fact-1505e15e61043ad0d5cd": {
        "active_index": 72,
        "question": "How does Knot Head Nylon say to clean and condition its dirty nylon rope?",
        "answer": "It says no jute-style conditioning is needed; put the rope in a mesh bag, machine-wash it, then lay it out or hang it to dry.",
        "decision": "drop",
        "evidence": "put it in a mesh bag, machine wash and lay it out or hang it to dry",
        "source_document": "data/rope_resource_manual_v1.jsonl",
        "source_document_file_sha256": "caee3a972ee70f6e8f23cd3de561bc673c54705e64df58cfc0c867e75995c3bc",
        "source_document_chars": 11044,
        "source_kind": "manual_resource_jsonl",
        "source_fact_id": "fact-bb9e3bb1450b30eb26fb",
        "material_scope": "manufacturer care instruction duplicated by owner resource fact",
        "reason_code": "quarantine_redundant_knothead_wash_instruction",
        "reason": "Owner fact 63e179cc596d19256952 already preserves the requested supplier URL and the same manufacturer-specific mesh-bag, machine-wash, and air-dry guidance alongside the material's handling tradeoffs.",
        "review_class": "redundant_vendor_care_drop",
        "redundant_survivors": ["fact-63e179cc596d19256952"],
    },
    "fact-068dc63eefb0785e40ac": {
        "active_index": 79,
        "question": "How does Rope365 contrast washing synthetic or cotton rope with washing hemp or jute?",
        "answer": "Synthetic and cotton rope are easy to wash after each use, while hemp and jute weaken when wet and need time to dry under tension.",
        "decision": "edit",
        "edited_question": "What wash-and-replacement tradeoff does Rope365 describe between more washable rope and hemp or jute?",
        "edited_answer": "The page describes synthetic and cotton rope as easier to wash, while saying hemp and jute lose strength while wet and take time to dry under tension; it warns that frequent washing may mean replacing a kit sooner.",
        "source_document": "data/raw/rope_resources_v1/rope365__114980387712f4b723a1.json",
        "source_document_file_sha256": "8a810f62ae9201e9abd48f535e778a7fefe9d8dcc9e2e89b6d99efbedf2809db",
        "source_document_chars": 5874,
        "material_scope": "source-attributed fiber-family tradeoff, not universal wash compatibility",
        "reason_code": "remove_easy_after_each_use_universal_and_restore_replacement_tradeoff",
        "reason": "The replacement removes the blanket implication that every cotton or synthetic construction is suitable for washing after every use and restores the source's replacement-frequency tradeoff.",
        "review_class": "fiber_washing_scope_edit",
    },
    "fact-5d1c212aa6ce9470432d": {
        "active_index": 95,
        "question": "How does Rope365 say to repair high stranding caused by pulling one rope strand?",
        "answer": "Rope365 says to retwist the rope and massage the strand back into place.",
        "decision": "edit",
        "edited_question": "What simulated maintenance exercise does Rope365 give for a deliberately pulled rope strand?",
        "edited_answer": "It says to pull one strand to create a high-stranding problem, then practise retwisting and massaging that strand back into place.",
        "source_document": "data/raw/rope_resources_v1/rope365__9360400375aedb1be302.json",
        "source_document_file_sha256": "15be0de807dcc93ceff6458ed36b27941372a334af333835aab9dc00b256bb6b",
        "source_document_chars": 2404,
        "material_scope": "deliberately simulated strand-level exercise, not every internal imbalance",
        "reason_code": "scope_retwist_instruction_as_simulated_pulled_strand_exercise",
        "reason": "The replacement restores that this is a deliberately created practice problem, preventing the short answer from being generalized to internal yarn imbalance or strength-critical damage.",
        "review_class": "maintenance_exercise_scope_edit",
    },
    "fact-fa0f81e7dcb203c7b762": {
        "active_index": 101,
        "question": "How does Rope365 suggest preventing washable cotton or synthetic rope from tangling in a washing machine?",
        "answer": "Daisy-chain the rope or place it in a pillowcase before machine washing.",
        "decision": "keep",
        "source_document": "data/raw/rope_resources_v1/rope365__077aecb334166abba18c.json",
        "source_document_file_sha256": "c54386ced8ae21d0a909ceac03acab37c9dd2b46da7893f03b410bd625d50c33",
        "source_document_chars": 8350,
        "material_scope": "anti-tangle preparation only for rope already determined washable",
        "reason_code": "keep_washable_rope_anti_tangle_preparation",
        "reason": "The question is already limited to washable cotton or synthetic rope and the answer gives only the source's mechanical anti-tangle preparation, without claiming that all rope is machine washable.",
        "review_class": "wash_preparation_keep",
    },
    "fact-402a66bcc7613f16d407": {
        "active_index": 122,
        "question": "If jute rope accidentally gets wet, how does Anatomie Studio suggest drying it to help restore its original form?",
        "answer": "Anatomie Studio suggests drying the rope under tension.",
        "decision": "edit",
        "edited_question": "What limited response does Anatomie Studio suggest after jute rope is accidentally wetted?",
        "edited_answer": "It suggests drying the rope under tension to help it return toward its prior form, while the same article advises against routine wet washing of jute.",
        "source_document": "data/raw/anatomiestudio_144932682af9c846.json",
        "source_document_file_sha256": "3e77a59322805eb277e5cc7e86fb5592e71b889b3c95197c2c401d2533d2c1c8",
        "source_document_chars": 3243,
        "material_scope": "accidental-wetting response, not a recommended cleaning process",
        "reason_code": "distinguish_accidental_wetting_response_from_routine_jute_washing",
        "reason": "The replacement keeps the drying response but restores the article's boundary that wetting is an accident to manage, not its recommended jute-cleaning method.",
        "review_class": "accidental_wetting_scope_edit",
    },
    "fact-3cdbfb40d9c4ecc13a39": {
        "active_index": 440,
        "question": "Why does Anatomie Studio advise against washing jute rope in a washing machine?",
        "answer": "Getting jute wet makes it swell and tightens its twist, which can leave the dried rope spongy and springy.",
        "decision": "keep",
        "source_document": "data/raw/anatomiestudio_144932682af9c846.json",
        "source_document_file_sha256": "3e77a59322805eb277e5cc7e86fb5592e71b889b3c95197c2c401d2533d2c1c8",
        "source_document_chars": 3243,
        "material_scope": "article-attributed physical handling consequence",
        "reason_code": "keep_jute_machine_wash_handling_consequence",
        "reason": "The row is explicitly attributed and gives the article's physical reason for its recommendation without claiming sterilization or universal pathogen safety.",
        "review_class": "jute_machine_wash_keep",
    },
    "fact-623a86d3b001295d28f4": {
        "active_index": 473,
        "question": "Why does Rope365 recommend washing rope only when needed?",
        "answer": "Cleaning progressively weakens rope, and some fiber types are more vulnerable than others.",
        "decision": "edit",
        "edited_question": "Which competing considerations does Rope365 say should shape how often rope is cleaned?",
        "edited_answer": "It says repeated cleaning can weaken rope, vulnerability varies by fiber, and partners’ hygiene needs and the rope’s use also matter, so cleaning frequency is a tradeoff rather than a fixed rule.",
        "source_document": "data/raw/rope_resources_v1/rope365__114980387712f4b723a1.json",
        "source_document_file_sha256": "8a810f62ae9201e9abd48f535e778a7fefe9d8dcc9e2e89b6d99efbedf2809db",
        "source_document_chars": 5874,
        "material_scope": "maintenance-and-hygiene tradeoff, not a fixed wash schedule",
        "reason_code": "restore_hygiene_and_fiber_context_to_wash_frequency",
        "reason": "The replacement restores the source's hygiene and consent context and avoids presenting 'only when needed' as a universal schedule detached from material and use.",
        "review_class": "wash_frequency_tradeoff_edit",
    },
}


OWNER_SURVIVORS = {
    "fact-7b9730efad5e1f58d941": "https://chromaknotz.square.site/",
    "fact-697f713f2af3f44c8639": "https://degiottorope.com/",
    "fact-63e179cc596d19256952": "https://knotheadnylon.net/",
}


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v451 baseline drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v452-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v451.jsonl"
    build_baseline(base, inputs / "v451.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def validate_source(spec: dict, row: dict, evidence: str) -> None:
    source_path = ROOT / spec["source_document"]
    if file_sha256(source_path) != spec["source_document_file_sha256"]:
        raise ValueError(f"source file drift: {row['fact_id']}")
    if spec.get("source_kind") == "manual_resource_jsonl":
        if len(source_path.read_text()) != spec["source_document_chars"]:
            raise ValueError(f"manual resource length drift: {row['fact_id']}")
        entry = next(item for item in read_jsonl(source_path) if item["fact_id"] == spec["source_fact_id"])
        if entry["evidence_url"] != row["url"] or entry["evidence"] != evidence:
            raise ValueError(f"manual resource evidence drift: {row['fact_id']}")
        return
    document = json.loads(source_path.read_text())
    if document["url"] != row["url"] or len(document["text"]) != spec["source_document_chars"]:
        raise ValueError(f"source document metadata drift: {row['fact_id']}")
    if evidence not in document["text"]:
        raise ValueError(f"evidence absent from full source snapshot: {row['fact_id']}")


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v452-observe-", dir=HERE) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v452-base-", dir=HERE) as tmp:
        directory = Path(tmp)
        base = directory / "v451.jsonl"
        build_baseline(base, directory / "v451.report.json")
        before = read_jsonl(base)
    by_fact = {row["fact_id"]: (index, row) for index, row in enumerate(before, 1)}

    for fact_id, expected_url in OWNER_SURVIVORS.items():
        if by_fact[fact_id][1]["url"] != expected_url:
            raise ValueError(f"owner survivor drift: {fact_id}")

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
            "fact_id": fact_id, "material_scope": spec["material_scope"],
            "projection_lineage": {
                "baseline_rows": BASELINE_ROWS, "baseline_sha256": BASELINE_SHA256,
                "prior_context_merit_review": True,
            },
            "reason": spec["reason"], "reason_code": spec["reason_code"],
            "review_class": spec["review_class"], "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER, "schema": "context-merit-audit-v452",
            "source": row["source"], "source_document": spec["source_document"],
            "source_document_chars": spec["source_document_chars"],
            "source_document_file_sha256": spec["source_document_file_sha256"],
            "source_support": "manual_paraphrase" if decision == "edit" else "full_snapshot_review",
            "support_evidence": evidence, "support_evidence_sha256": text_sha256(evidence),
            "url": row["url"],
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
    if (observation["rows"], observation["eval"]) != (501, 612):
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
            "by_material_scope": dict(Counter(row["material_scope"] for row in audits)),
            "by_review_class": dict(Counter(row["review_class"] for row in audits)),
            "path": portable(AUDIT), "rows": len(audits), "sha256": file_sha256(AUDIT),
        },
        "conservative_capacity": {
            "after": observation["after"], "before": observation["before"],
            "delta": {key: observation["after"][key] - observation["before"][key] for key in observation["before"]},
            "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
        },
        "curation_outcome": {
            "material_or_maintenance_scope_repairs": 5,
            "operational_maintenance_rows_kept": 2,
            "redundant_vendor_or_product_rows_quarantined": 3,
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
            "new_or_pending_corpus_inputs": 0,
            "protected_eval_heldout_holdout_ood_shadow_probe_rows_opened": 0,
            "training_or_eval_artifacts_modified": 0,
        },
        "source_snapshot_inventory": {
            "documents": len(unique_sources), "reviewed_rows": len(SPECS),
            "paths": sorted(unique_sources),
            "total_unique_characters": sum(spec["source_document_chars"] for spec in unique_sources.values()),
        },
        "surviving_owner_resource_facts": {
            "surviving_facts": sorted(OWNER_SURVIVORS),
            "urls": sorted(OWNER_SURVIVORS.values()),
        },
        "schema": "context-merit-audit-report-v452",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "training_layer_contract": {
            "cleaned_markdown_site_corpora": "Distinct first-class future training artifacts; no corpus output was ingested by v452.",
            "derived_qa": "Distinct first-class training layer; v452 changes only existing sealed Q&A using attached evidence and pinned source snapshots.",
            "deduplication_policy": "Do not collapse a cleaned Markdown document and its derived Q&A merely because they express the same source knowledge.",
            "provenance_requirement": "Link future derived Q&A to cleaned source documents through stable source URL and document or snapshot hashes.",
        },
        "v52_isolation": {"active_v52_inputs_modified": False, "candidate_status": "isolated_pending_not_promoted"},
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
