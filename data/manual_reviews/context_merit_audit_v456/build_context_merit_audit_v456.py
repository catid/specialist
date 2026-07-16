#!/usr/bin/env python3
"""Curate eight existing WykD terminology, safety, and consent Q&As.

The only candidate base is sealed v455. Two useful rows are retained, two are
repaired into more natural and complete questions, and four generic or
redundant rows are quarantined. The three pinned WykD snapshots were read in
full. No raw corpus, requested-resource fact, active training artifact, or
protected evaluation artifact is changed.
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
V455 = DATA / "manual_reviews/context_merit_audit_v455"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V455), str(V290)]

import build_context_merit_audit_v455 as previous
import build_context_merit_audit_v290 as core
from qa_quality import stable_fact_id


AUDIT = HERE / "context_merit_audit_v456.jsonl"
CURATION = HERE / "pending_curation_context_merit_v456.jsonl"
REPORT = HERE / "report_context_merit_v456.json"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
BASELINE_ROWS = 497
BASELINE_SHA256 = "3c12dc3811b8c6b79a7bde5795c27a21ebb2e22073eff036da7b7405279202f6"
EXPECTED_OUTPUT_ROWS = 493
EXPECTED_OUTPUT_SHA256 = "c399f0175f6f10837c0b2e582ca3453a13c768838f18ae76225529547ebf237b"
EXPECTED_CAPACITY_BEFORE = {
    "conflict_units": 258,
    "equipment_material": 21,
    "resources_general": 77,
    "safety_consent": 86,
    "technique": 74,
}
EXPECTED_CAPACITY_AFTER = {
    "conflict_units": 258,
    "equipment_material": 21,
    "resources_general": 77,
    "safety_consent": 86,
    "technique": 74,
}
OWNER_SOURCE = previous.OWNER_SOURCE
OWNER_FACT_IDS = previous.OWNER_FACT_IDS
REVIEWED_AT = "2026-07-16"
REVIEWER = "codex-context-merit-audit-v456"

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
portable = previous.portable
conservative_capacity = previous.conservative_capacity


SOURCES = {
    "03bb1af918ad3ed44208dc4805e48e40718f47d3ffc8994d47a54035e70140d2": {
        "path": "data/raw/wykd_944e4e6d621a97c9.json",
        "file_sha256": "5d495bf93380ad381f23068c4e51ffab4d519822f32c9db8f33aa07d1d8d2add",
        "chars": 3264,
        "url": "https://wykd.com/learning/2012/11/03/newness-and-getting-out-into-the-kink-community/",
    },
    "ca4af3374c7b1554a16f969a19a55d3e8acd8ff776f675eaec19b1779ec32511": {
        "path": "data/raw/wykd_a74fec63b0114fff.json",
        "file_sha256": "91fc248bcbee1123883003e54d1e0eced40dcedace26cf7f6ac7e3538c0e503f",
        "chars": 3453,
        "url": "https://wykd.com/shibari-kinbaku-bondage-teaching/2012/10/14/ichinawa-ippon-me-no-nawa-and-one-rope/",
    },
    "5aa46db70e83e88c8d2fc9b0fb6a0e28e3339ce8d061e2a1ed04073dac5bdc98": {
        "path": "data/raw/wykd_19d6a26116e26c70.json",
        "file_sha256": "0014ef6d8f97070601e19654be7fff70df9cd5941c66407d776c7bf35f0867b4",
        "chars": 2938,
        "url": "https://wykd.com/random/2013/11/30/self-awareness-luck-and-responsibility-in-rope-bondage-injuries/",
    },
}


EVENT_BOUNDARY_EVIDENCE = """- To be treated with respect
- Not to be touched, slapped, spanked, groped, played with or otherwise molested without consent
- Not to have your personal space disregarded
- Never to be told that you ‘have’ to do things
- Not to be told how you ‘should’ behave
- To not have anyone make you feel uncomfortable or scared"""


SPECS = {
    "fact-6ac3aaf9ec5026e96536": {
        "active_index": 30,
        "question": "Does WykD claim that Ichinawa is the only correct name for the one-rope technique?",
        "answer": "No. WykD calls it a correct usage, but not necessarily the only correct usage.",
        "decision": "edit",
        "edited_question": "What naming caveat does WykD give for the term Ichinawa?",
        "edited_answer": "WykD presents Ichinawa as a correct name for the one-rope technique, but not necessarily the only correct name.",
        "subject_scope": "source-attributed terminology with explicit linguistic uncertainty",
        "reason_code": "repair_ichinawa_yes_no_recall_as_naming_caveat",
        "reason": "The replacement removes an awkward yes-or-no prompt while preserving the useful warning that one supported Japanese term should not be presented as uniquely correct.",
        "review_class": "terminology_uncertainty_edit",
    },
    "fact-cc58bd014cc6acfaaf2b": {
        "active_index": 113,
        "question": "How does WykD distinguish Ichinawa from simply practicing with one rope?",
        "answer": "Ichinawa is a distinct technique designed to use exactly one rope, whereas one-rope practice only specifies how many ropes are being used.",
        "decision": "keep",
        "subject_scope": "named one-rope technique versus practice constraint",
        "reason_code": "keep_ichinawa_technique_practice_distinction",
        "reason": "The row asks a natural conceptual question and preserves the article's useful distinction between a technique designed around one rope and a practice exercise limited to one rope.",
        "review_class": "technique_definition_keep",
    },
    "fact-cf160d03f19be3d053ab": {
        "active_index": 180,
        "question": "What corrective steps does WykD’s article on self-awareness and responsibility in rope-bondage injuries recommend when a rigger causes repeated injuries that others are not causing?",
        "answer": "WykD says a rigger causing repeated injuries that others are not causing should reduce how much they rig, examine common themes in the injuries, and work hard to prevent recurrence.",
        "decision": "keep",
        "subject_scope": "actionable response to a repeated ordinary-bondage injury pattern",
        "reason_code": "keep_repeated_injury_corrective_steps",
        "reason": "The row gives three concrete corrective actions and remains carefully limited to repeated injuries that differ from peers' outcomes, rather than implying that all rope risk can be eliminated.",
        "review_class": "injury_accountability_keep",
    },
    "fact-65dd28005112459d348b": {
        "active_index": 310,
        "question": "What self-audit does WykD recommend when similar injuries recur across multiple partners and sessions?",
        "answer": "WykD recommends looking for the common factor across the incidents, including the tying practices of the person present in each case.",
        "decision": "drop",
        "subject_scope": "one step already contained in the surviving corrective-action row",
        "reason_code": "quarantine_redundant_repeated_injury_common_factor_row",
        "reason": "This row isolates the common-factor step already contained in the stronger corrective-actions row from the same document; keeping both overweights one short argument without adding a distinct response.",
        "review_class": "injury_accountability_redundancy_drop",
        "substantive_survivors": ["fact-cf160d03f19be3d053ab"],
    },
    "fact-4bc2eb7e10f590c0de00": {
        "active_index": 314,
        "question": "What should newcomers to kink events be able to expect regarding physical contact?",
        "answer": "They should not be touched or otherwise played with without consent.",
        "decision": "edit",
        "edited_question": "What baseline boundaries does WykD say newcomers should expect at kink events?",
        "edited_answer": "They should expect respect and personal space, no touching or play without consent, no pressure to do things or behave a prescribed way, and no conduct that makes them uncomfortable or scared.",
        "support_evidence": EVENT_BOUNDARY_EVIDENCE,
        "subject_scope": "compact event-boundary checklist rather than one isolated contact rule",
        "reason_code": "restore_newcomer_event_boundary_context",
        "reason": "The replacement turns one generic consent statement into the source's more useful compact boundary checklist while staying within one contiguous passage.",
        "review_class": "event_boundaries_edit",
    },
    "fact-7a5ce8a9d47b0f300d9b": {
        "active_index": 341,
        "question": "What two conditions does WykD’s newcomer article give for a new submissive to follow an instruction in a D/s context?",
        "answer": "They must actually want to follow the instruction and consent to it.",
        "decision": "drop",
        "subject_scope": "generic role-specific restatement of voluntary consent",
        "reason_code": "quarantine_generic_submissive_consent_restatement",
        "reason": "The row reduces voluntary consent to two recall words and duplicates the richer surviving event-boundaries row plus many broader consent rows in the dataset.",
        "review_class": "generic_role_consent_drop",
        "substantive_survivors": ["fact-4bc2eb7e10f590c0de00"],
    },
    "fact-5b7af186de803e302123": {
        "active_index": 362,
        "question": "When does WykD’s newcomer article say a dominant has the right to tell a submissive what to do?",
        "answer": "WykD says a dominant has that right only when the individual has actually consented.",
        "decision": "drop",
        "subject_scope": "generic role-specific restatement of no automatic authority",
        "reason_code": "quarantine_generic_dominant_authority_consent_restatement",
        "reason": "The row is a second role-labeled formulation of the same voluntary-consent principle and adds no rope-specific or operational detail beyond the surviving boundary checklist.",
        "review_class": "generic_role_consent_drop",
        "substantive_survivors": ["fact-4bc2eb7e10f590c0de00"],
    },
    "fact-b6e8fcc013815e7afeb2": {
        "active_index": 364,
        "question": "When would WykD use Ipponnawa rather than Ichinawa?",
        "answer": "WykD says Ipponnawa applies when counting ropes; more precisely, it may be phrased as Ippon me no nawa.",
        "decision": "drop",
        "subject_scope": "isolated Japanese counting-form recall",
        "reason_code": "quarantine_ipponnawa_counting_form_recall",
        "reason": "The row asks for a narrow wording distinction from a single author's translator consultation. The two surviving Ichinawa rows retain the durable technique distinction and the necessary warning against a uniquely correct label.",
        "review_class": "isolated_terminology_recall_drop",
        "substantive_survivors": ["fact-6ac3aaf9ec5026e96536", "fact-cc58bd014cc6acfaaf2b"],
    },
}


def requested_urls() -> set[str]:
    resources = json.loads(RESOURCE_MANIFEST.read_text())["resources"]
    return {
        url
        for resource in resources
        for url in (resource["canonical_url"], resource.get("recommendation_url"))
        if url
    }


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v455 baseline drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v456-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v455.jsonl"
    build_baseline(base, inputs / "v455.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def validate_source(row: dict, evidence: str) -> dict:
    source = SOURCES[row["document_sha256"]]
    source_path = ROOT / source["path"]
    if file_sha256(source_path) != source["file_sha256"]:
        raise ValueError(f"source file drift: {row['fact_id']}")
    document = json.loads(source_path.read_text())
    if document["url"] != source["url"] or row["url"] != source["url"]:
        raise ValueError(f"source URL drift: {row['fact_id']}")
    if len(document["text"]) != source["chars"]:
        raise ValueError(f"source character drift: {row['fact_id']}")
    if text_sha256(document["text"]) != row["document_sha256"]:
        raise ValueError(f"source text drift: {row['fact_id']}")
    if evidence not in document["text"]:
        raise ValueError(f"evidence absent from full source snapshot: {row['fact_id']}")
    return source


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v456-observe-", dir=HERE) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v456-base-", dir=HERE) as tmp:
        directory = Path(tmp)
        base = directory / "v455.jsonl"
        build_baseline(base, directory / "v455.report.json")
        before = read_jsonl(base)
    by_fact = {row["fact_id"]: (index, row) for index, row in enumerate(before, 1)}

    audits, curations = [], []
    for audit_index, (fact_id, spec) in enumerate(SPECS.items(), 1):
        active_index, row = by_fact[fact_id]
        if active_index != spec["active_index"]:
            raise ValueError(f"active index drift: {fact_id}")
        if (row["question"], row["answer"]) != (spec["question"], spec["answer"]):
            raise ValueError(f"active Q&A drift: {fact_id}")
        evidence = spec.get("support_evidence", row["evidence"])
        source = validate_source(row, evidence)

        decision = spec["decision"]
        if decision == "edit":
            curations.append({
                "action": "edit",
                "answer": spec["edited_answer"],
                "document_sha256": row["document_sha256"],
                "evidence": evidence,
                "evidence_url": row["url"],
                "expected_answer": row["answer"],
                "expected_question": row["question"],
                "fact_id": fact_id,
                "paraphrase_rationale": spec["reason"],
                "question": spec["edited_question"],
                "reason": spec["reason"],
                "reason_code": spec["reason_code"],
                "reviewed_at": REVIEWED_AT,
                "reviewer": REVIEWER,
                "source_lineage": {"source_document": source["path"]},
                "support_type": "manual_paraphrase",
            })
        elif decision == "drop":
            curations.append({
                "action": "drop",
                "expected_answer": row["answer"],
                "expected_question": row["question"],
                "fact_id": fact_id,
                "reason": spec["reason"],
                "reason_code": spec["reason_code"],
                "reviewed_at": REVIEWED_AT,
                "reviewer": REVIEWER,
            })
        elif decision != "keep":
            raise ValueError(f"unsupported decision: {decision}")

        audit = {
            "active_answer": row["answer"],
            "active_index": active_index,
            "active_question": row["question"],
            "audit_index": audit_index,
            "decision": decision,
            "document_sha256": row["document_sha256"],
            "fact_id": fact_id,
            "projection_lineage": {
                "baseline_rows": BASELINE_ROWS,
                "baseline_sha256": BASELINE_SHA256,
                "prior_context_merit_review": True,
                "source_absent_from_recent_passes": "v448-v455",
            },
            "reason": spec["reason"],
            "reason_code": spec["reason_code"],
            "review_class": spec["review_class"],
            "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER,
            "schema": "context-merit-audit-v456",
            "source": row["source"],
            "source_document": source["path"],
            "source_document_chars": source["chars"],
            "source_document_file_sha256": source["file_sha256"],
            "source_document_text_sha256": row["document_sha256"],
            "source_support": "manual_paraphrase" if decision == "edit" else "full_snapshot_review",
            "subject_scope": spec["subject_scope"],
            "support_evidence": evidence,
            "support_evidence_sha256": text_sha256(evidence),
            "url": row["url"],
        }
        if decision == "edit":
            audit.update({
                "edited_answer": spec["edited_answer"],
                "edited_fact_id": stable_fact_id(spec["edited_question"], spec["edited_answer"]),
                "edited_question": spec["edited_question"],
            })
        if spec.get("substantive_survivors"):
            audit["substantive_survivors"] = spec["substantive_survivors"]
        audits.append(audit)

    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    observation = observe(before)
    if not observation["dataset_equal"] or not observation["report_equal"]:
        raise ValueError(f"nondeterministic projection: {observation}")
    if (observation["rows"], observation["eval"]) != (EXPECTED_OUTPUT_ROWS, 612):
        raise ValueError(f"projection count drift: {observation}")
    if observation["before"] != EXPECTED_CAPACITY_BEFORE:
        raise ValueError(f"baseline capacity drift: {observation}")
    if EXPECTED_CAPACITY_AFTER is not None and observation["after"] != EXPECTED_CAPACITY_AFTER:
        raise ValueError(f"candidate capacity drift: {observation}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observation["sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError(f"projection hash drift: {observation}")

    with tempfile.TemporaryDirectory(prefix=".v456-resource-check-", dir=HERE) as tmp:
        projected = Path(tmp) / "projected.jsonl"
        build_projection(projected, Path(tmp) / "projected.report.json")
        projected_rows = read_jsonl(projected)
        blob = projected.read_text()
    urls = requested_urls()
    if len(urls) != 24 or any(url not in blob for url in urls):
        raise ValueError("requested resource coverage drift")
    if {row["fact_id"] for row in projected_rows if row.get("source") == OWNER_SOURCE} != set(OWNER_FACT_IDS):
        raise ValueError("owner resource fact drift")

    curation_counts = Counter(row["action"] for row in curations)
    unique_source_chars = sum(source["chars"] for source in SOURCES.values())
    REPORT.write_text(json.dumps({
        "audit": {
            "by_decision": dict(Counter(row["decision"] for row in audits)),
            "by_review_class": dict(Counter(row["review_class"] for row in audits)),
            "by_subject_scope": dict(Counter(row["subject_scope"] for row in audits)),
            "path": portable(AUDIT),
            "rows": len(audits),
            "sha256": file_sha256(AUDIT),
        },
        "baseline_checkpoint": {
            "commit": "edbc5805160ab607012b6d3a1b24f1443b3c999f",
            "rows": BASELINE_ROWS,
            "sha256": BASELINE_SHA256,
            "version": "v455",
        },
        "conservative_capacity": {
            "after": observation["after"],
            "before": observation["before"],
            "delta": {
                key: observation["after"][key] - observation["before"][key]
                for key in observation["before"]
            },
            "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
        },
        "curation_outcome": {
            "high_value_rows_kept": 2,
            "natural_or_context_rich_repairs": 2,
            "redundant_generic_or_isolated_rows_quarantined": 4,
        },
        "isolated_build_projection": {
            "automated_projection_runs": 2,
            "new_additions_applied": 0,
            "output_rows": observation["rows"],
            "output_sha256": observation["sha256"],
            "repeat_dataset_byte_identical": observation["dataset_equal"],
            "repeat_projection_report_byte_identical": observation["report_equal"],
            "sealed_eval_fact_count_reported_by_tooling": observation["eval"],
        },
        "manual_slice": {
            "candidate_rows": len(SPECS),
            "recent_passes_without_source": [f"v{version}" for version in range(448, 456)],
            "source": "wykd",
            "subjects": ["Ichinawa terminology", "repeated-injury accountability", "newcomer event boundaries"],
        },
        "new_pending_curation": {
            "by_action": dict(curation_counts),
            "decisions": len(curations),
            "path": portable(CURATION),
            "sha256": file_sha256(CURATION),
        },
        "requested_resource_coverage": {
            "manifest_urls": len(urls),
            "manifest_urls_present": sum(url in blob for url in urls),
            "owner_resource_facts": len(OWNER_FACT_IDS),
        },
        "schema": "context-merit-audit-report-v456",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "source_boundary": {
            "new_or_pending_corpus_inputs": 0,
            "protected_eval_heldout_holdout_ood_shadow_probe_rows_opened": 0,
            "raw_markdown_or_snapshot_documents_modified": 0,
            "scout_files_opened_or_modified": 0,
            "training_or_eval_artifacts_modified": 0,
        },
        "source_snapshot_inventory": {
            "documents": len(SOURCES),
            "paths": sorted(source["path"] for source in SOURCES.values()),
            "reviewed_rows": len(SPECS),
            "total_unique_characters": unique_source_chars,
        },
        "training_layer_contract": {
            "cleaned_markdown_site_corpora": "Distinct first-class source artifacts; no raw corpus output was edited or ingested by v456.",
            "derived_qa": "Distinct first-class training layer; v456 changes only existing v455 Q&A using three fully reviewed pinned source snapshots.",
            "deduplication_policy": "Do not collapse a cleaned Markdown document and its derived Q&A merely because they express the same source knowledge.",
            "provenance_requirement": "Every reviewed row is pinned to source URL, document text hash, snapshot file hash, and exact supporting evidence.",
        },
        "v52_isolation": {
            "active_v52_inputs_modified": False,
            "candidate_status": "isolated_pending_not_promoted",
        },
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
