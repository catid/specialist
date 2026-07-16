#!/usr/bin/env python3
"""Curate ten existing knot, friction, handling, and pattern Q&As.

The only candidate base is sealed v446.  Five rows are repaired from eight
already-pinned raw snapshots, two low-merit rows are quarantined, and three
operationally complete rows are retained.  No pending corpus is read and no
new Rope365, Crash, or other site-derived fact is added.
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
V446 = DATA / "manual_reviews/context_merit_audit_v446"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V446), str(V290)]

import build_context_merit_audit_v446 as previous
import build_context_merit_audit_v290 as core
from qa_quality import stable_fact_id


AUDIT = HERE / "context_merit_audit_v447.jsonl"
CURATION = HERE / "pending_curation_context_merit_v447.jsonl"
REPORT = HERE / "report_context_merit_v447.json"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
BASELINE_ROWS = 515
BASELINE_SHA256 = "7ab0562e978c6a15f7122fd3a62e8cade2daff80a0c0a0228dc069cf1b59d900"
EXPECTED_OUTPUT_SHA256 = "fca887a130f04ed5c6f229eb81aaeccd1bf39aada0620cc9507a7d6b759e8518"
EXPECTED_CAPACITY_BEFORE = {
    "conflict_units": 260,
    "equipment_material": 22,
    "resources_general": 78,
    "safety_consent": 86,
    "technique": 74,
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
REVIEWER = "codex-context-merit-audit-v447"

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
portable = previous.portable
conservative_capacity = previous.conservative_capacity


SPECS = {
    "fact-f6faf5703b157e22121d": {
        "active_index": 122,
        "question": "How should the securing knot be positioned in Rope365’s coiling checklist so it will not come undone during transport?",
        "answer": "Pull and center the securing knot so it will not come undone during transport.",
        "decision": "drop",
        "source_document": "data/raw/rope365_d9c48a4547717047.json",
        "source_document_file_sha256": "f389862cd7e586e607ddc99d82e174662dcd3f4f489081c7dc3598ea9548d92b",
        "source_document_chars": 2093,
        "reason_code": "quarantine_coiling_visual_checklist_tautology",
        "reason": "The answer merely restates the checklist instruction, while the text gives no topology for how to pull or center the securing knot. Active fact-4d9cba14a6eb00ad7e1d preserves the page's useful supported storage guidance.",
        "review_class": "unsupported_visual_step_drop",
        "redundant_survivor": "fact-4d9cba14a6eb00ad7e1d",
    },
    "fact-e0bf197583f166eda344": {
        "active_index": 159,
        "question": "What advantage does Rope365 describe for a chain-stitch rope pattern?",
        "answer": "It can be fast to tie and untie because the whole rope does not need to be pulled through.",
        "decision": "edit",
        "edited_question": "What function and handling advantage does Rope365 give for chain-stitch rope patterns?",
        "edited_answer": "Chain-stitch or slip-stitch techniques create body-wrapping rope patterns that can be tied and untied quickly because the whole rope does not have to be pulled through.",
        "source_document": "data/raw/rope_resources_v1/rope365__255e7fa505a2f3f096a9.json",
        "source_document_file_sha256": "bf4a7bf3c440845d9bd370675276ac63d160749350d4fa86113d76f1939f2b63",
        "source_document_chars": 3677,
        "reason_code": "complete_chain_stitch_pattern_function_and_handling",
        "reason": "The replacement adds the source-supported body-wrapping function and alternate slip-stitch name to the existing pull-through handling advantage.",
        "review_class": "pattern_handling_edit",
    },
    "fact-3ceef5612e46edd56a3f": {
        "active_index": 164,
        "question": "What benefits and limitation does Rope365 give for using frictions instead of knots?",
        "answer": "Frictions are fast and do not compact into something hard to untie, but they require tension to stay in place.",
        "decision": "keep",
        "source_document": "data/raw/rope365_8e6f4abea3bf4f6d.json",
        "source_document_file_sha256": "dc1b43711764d1c6b2093db14048fc13d5dc2e5e3c091b62818bdc6a307e4f3c",
        "source_document_chars": 4580,
        "reason_code": "keep_complete_friction_function_tradeoff",
        "reason": "The row gives the operational advantages of speed and easy untying together with the essential limitation that the friction depends on maintained tension.",
        "review_class": "complete_operational_keep",
    },
    "fact-5b6a4e8c9d886765a7e3": {
        "active_index": 172,
        "question": "What category of tie does Rope365 call “Monoblock”?",
        "answer": "Rope365 calls Monoblock a package tie.",
        "decision": "edit",
        "edited_question": "How does Rope365 describe the structure and effect of its Monoblock package tie?",
        "edited_answer": "It folds both legs to the waist for a more restrictive booty-basket variation, using either a two-column tie between the waist and legs or an expanded booty basket that also captures the calves.",
        "evidence": "Day 133: Monoblock – Alt package tie. Tie both leg folded to the waist for an even more restrictive version of the booty basket. This can be with a two column tie between the waist and the legs, or by expanding a booty basket to include the calfs.",
        "source_document": "data/raw/rope365_6ab0025843fd73d5.json",
        "source_document_file_sha256": "e4bc41d5fe854a35d86dbda53f1b27fdb0cb244ac70abf155c2b6579e7a8c97f",
        "source_document_chars": 3521,
        "reason_code": "replace_monoblock_label_recall_with_pattern_architecture",
        "reason": "The replacement turns a category-label lookup into the source's actual folded-leg architecture, alternative constructions, and increased restriction.",
        "review_class": "pattern_architecture_edit",
    },
    "fact-ab74813c97221c1c95f6": {
        "active_index": 205,
        "question": "What does a cinch add to a two-column rope structure?",
        "answer": "A cinch passes rope through the gap between two columns and catches rope on the other side, preventing rotation or sliding and making the structure more solid.",
        "decision": "keep",
        "source_document": "data/raw/rope_resources_v1/rope365__e156eea7b8dfda8b39a9.json",
        "source_document_file_sha256": "fe120f0bba58d42fc8b3d778e3b4be464f400afa636dc729e1fbd2dca15248f8",
        "source_document_chars": 2678,
        "reason_code": "keep_complete_cinch_topology_and_function",
        "reason": "The row identifies both sides of the cinch topology and explains its concrete anti-rotation, anti-sliding, and solidity functions.",
        "review_class": "complete_operational_keep",
    },
    "fact-bc382032cb4fc9559958": {
        "active_index": 224,
        "question": "What does Rope365 call the family of knots used to extend rope?",
        "answer": "The family of knots used to extend rope is called bends.",
        "decision": "edit",
        "edited_question": "What does Rope365 mean by bends, and what tradeoff does it note among them?",
        "edited_answer": "Rope365 calls knots used to extend rope “bends”; some are easier to tie and others sturdier, with the sheet bend noted as popular in rope bondage.",
        "source_document": "data/raw/rope365_25f1b23eb40be00e.json",
        "source_document_file_sha256": "b07e3a74c93de379408d9212fbb5c62c23e214a0f73b8c367c9b691dc4b3ad97",
        "source_document_chars": 3439,
        "reason_code": "complete_bends_function_tradeoff_and_example",
        "reason": "The replacement retains the terminology but adds the source's ease-versus-sturdiness tradeoff and a relevant example instead of asking only for a label.",
        "review_class": "knot_function_tradeoff_edit",
    },
    "fact-14a53d0e9489d92e067d": {
        "active_index": 225,
        "question": "What does Rope365 call the vertical part used to anchor horizontal and diagonal wraps in its chest-binder tie?",
        "answer": "Rope365 calls that vertical anchor the stem.",
        "decision": "edit",
        "edited_question": "How does Rope365 build and apply the stem in its chest-binder structure?",
        "edited_answer": "It uses a vertical underlying line to anchor horizontal and diagonal wraps that press and shape chest tissue; the page advises moving the skin upward so pressure stays perpendicular to the tissue and rib cage, reducing uncomfortable folds and twists.",
        "evidence": "Chest Binder\nOne specific application of chest rope is creating a chest binder, a structure that will shape chest tissue to be more flat, generally to increase a masculinity aesthetic. In this particular tie, we create an underlying structure with a vertical steam, then we can use that stem as the anchor of the horizontal and diagonal wraps that we will use to press on the skin. It is important to move the skin up as we shape it so that the pressure is perpendicular to the tissue and the rib cage to minimize uncomfortable folds and twists in the skin. Look at resources for chest binding safety for more information on long term usage.",
        "source_document": "data/raw/rope365_085ee19dfa760651.json",
        "source_document_file_sha256": "c0cf325c168dacfa1adac27d9193983b76bf74b8fad31fea50eff5cdd6d5b2db",
        "source_document_chars": 2588,
        "reason_code": "replace_stem_label_with_chest_binder_architecture_and_pressure_guidance",
        "reason": "The replacement explains the stem's placement and anchor role and restores the source's pressure-direction guidance for reducing uncomfortable skin folds and twists.",
        "review_class": "pattern_architecture_edit",
    },
    "fact-8a95cdcfa51aeebfc626": {
        "active_index": 358,
        "question": "What two limitations does Rope365 identify for slip knots?",
        "answer": "Rope365 says slip knots tighten and are not stable.",
        "decision": "edit",
        "edited_question": "What functional tradeoffs does Rope365 give for basic slip knots?",
        "edited_answer": "Slip knots are fast and can serve for quick capture or an adjustable loop, but they tighten and are not stable, so those limitations must be considered when choosing where to use them.",
        "source_document": "data/raw/rope365_25f1b23eb40be00e.json",
        "source_document_file_sha256": "b07e3a74c93de379408d9212fbb5c62c23e214a0f73b8c367c9b691dc4b3ad97",
        "source_document_chars": 3439,
        "reason_code": "complete_slip_knot_uses_and_limitations",
        "reason": "The replacement restores the source's fast quick-capture and adjustable-loop uses while retaining the tightening and instability limitations needed to evaluate use.",
        "review_class": "knot_function_tradeoff_edit",
    },
    "fact-42526efd7c67bf9799a3": {
        "active_index": 411,
        "question": "Which hitch does Rope365 describe as a climbing belay shape popularized by Mr. Munter?",
        "answer": "Rope365 identifies it as the Munter Hitch.",
        "decision": "drop",
        "evidence": "Munter Hitch\nMore turns make the frictions progressively more stable. This specific shape was made popular by Mr. Munter as a belay technique in climbing. It may be stable if the rope is grippy, but may slide with more slippery rope.",
        "source_document": "data/raw/rope365_8e6f4abea3bf4f6d.json",
        "source_document_file_sha256": "dc1b43711764d1c6b2093db14048fc13d5dc2e5e3c091b62818bdc6a307e4f3c",
        "source_document_chars": 4580,
        "reason_code": "quarantine_munter_name_lookup_redundant_with_active_function",
        "reason": "The row asks only for a hitch label. Active fact-b381baeee64c518a057b teaches the Munter hitch's moving-rope friction function and contrasts it with a crossing hitch, so the name lookup adds no operational value.",
        "review_class": "label_only_redundant_drop",
        "redundant_survivor": "fact-b381baeee64c518a057b",
    },
    "fact-f5e457d26ab18081df76": {
        "active_index": 510,
        "question": "Why should neither a square knot nor a granny knot be used when solidity is critical?",
        "answer": "Rope365 says neither knot is fully secure and both can capsize if their ends are jerked.",
        "decision": "keep",
        "source_document": "data/raw/rope365_bc4120687a97c8c3.json",
        "source_document_file_sha256": "647352e2b07b2bcdcd00dc350f141e355cb7929421cf6a68f4b386b4207c6abc",
        "source_document_chars": 2380,
        "reason_code": "keep_attributed_square_granny_critical_solidity_warning",
        "reason": "The warning is explicitly attributed to Rope365, states the critical-use boundary, and includes the evidence-supported capsize failure mode rather than making an unsupported universal safety claim.",
        "review_class": "complete_operational_keep",
    },
}


REDUNDANT_SURVIVORS = {
    "fact-4d9cba14a6eb00ad7e1d": {
        "question": "Why might Rope365 recommend leaving an overhand-hank securing knot loose for long-term storage?",
        "answer": "A tight knot can kink fragile natural rope, while a loose knot reduces that storage stress.",
    },
    "fact-b381baeee64c518a057b": {
        "question": "How does Rope365 distinguish the purpose of a Munter hitch from a crossing hitch?",
        "answer": "A Munter hitch adds friction to moving rope in belay systems, whereas a crossing hitch holds rope in place and resists movement.",
    },
}


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v446 baseline drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v447-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v446.jsonl"
    build_baseline(base, inputs / "v446.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v447-observe-", dir=HERE) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v447-base-", dir=HERE) as tmp:
        directory = Path(tmp)
        base = directory / "v446.jsonl"
        build_baseline(base, directory / "v446.report.json")
        before = read_jsonl(base)
    by_fact = {row["fact_id"]: (index, row) for index, row in enumerate(before, 1)}

    for fact_id, expected in REDUNDANT_SURVIVORS.items():
        row = by_fact[fact_id][1]
        if (row["question"], row["answer"]) != (expected["question"], expected["answer"]):
            raise ValueError(f"redundant survivor drift: {fact_id}")

    audits, curations = [], []
    for audit_index, (fact_id, spec) in enumerate(SPECS.items(), 1):
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
        evidence = spec.get("evidence", row["evidence"])
        if evidence not in document["text"]:
            raise ValueError(f"evidence absent from full source snapshot: {fact_id}")

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
                "source_lineage": {"raw_document": spec["source_document"]},
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
            "reviewer": REVIEWER, "schema": "context-merit-audit-v447",
            "source": row["source"], "source_document": spec["source_document"],
            "source_document_chars": spec["source_document_chars"],
            "source_document_file_sha256": spec["source_document_file_sha256"],
            "source_support": "manual_paraphrase" if decision == "edit" else "full_snapshot_review",
            "support_evidence": evidence,
            "support_evidence_sha256": text_sha256(evidence), "url": row["url"],
        }
        if decision == "edit":
            audit.update({
                "edited_answer": spec["edited_answer"],
                "edited_fact_id": stable_fact_id(spec["edited_question"], spec["edited_answer"]),
                "edited_question": spec["edited_question"],
            })
        if spec.get("redundant_survivor"):
            audit["redundant_survivor"] = spec["redundant_survivor"]
        audits.append(audit)

    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    observation = observe(before)
    if not observation["dataset_equal"] or not observation["report_equal"]:
        raise ValueError(f"nondeterministic projection: {observation}")
    if (observation["rows"], observation["eval"]) != (513, 612):
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
            "complete_operational_rows_kept": 3,
            "knot_function_tradeoff_repairs": 2,
            "low_merit_rows_quarantined": 2,
            "pattern_architecture_repairs": 2,
            "pattern_handling_repairs": 1,
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
        "redundancy_quarantine": {
            "dropped_to_surviving_fact": {
                fact_id: spec["redundant_survivor"]
                for fact_id, spec in SPECS.items() if spec.get("redundant_survivor")
            },
        },
        "schema": "context-merit-audit-report-v447",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "source_snapshot_inventory": {
            "documents": len(unique_sources), "reviewed_rows": len(SPECS),
            "paths": sorted(unique_sources),
            "total_unique_characters": sum(spec["source_document_chars"] for spec in unique_sources.values()),
        },
        "training_layer_contract": {
            "cleaned_markdown_site_corpora": "Distinct first-class future training artifacts; unfinished corpus output was neither read nor ingested by v447.",
            "derived_qa": "Distinct first-class training layer; v447 changes only existing sealed Q&A using attached evidence and pinned raw snapshots.",
            "deduplication_policy": "Do not collapse a cleaned Markdown document and its derived Q&A merely because they express the same source knowledge.",
            "provenance_requirement": "Link future derived Q&A to the cleaned source document through stable source URL and document or snapshot hashes.",
        },
        "v52_isolation": {"active_v52_inputs_modified": False, "candidate_status": "isolated_pending_not_promoted"},
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
