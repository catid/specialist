#!/usr/bin/env python3
"""Curate seven existing Crash Restraint practical and safety Q&As.

The only candidate base is the committed v456 projection. Two useful rows are
retained, four rows are repaired for naturalness and/or full-snapshot
provenance, and one row that duplicates a preserved requested-resource fact is
quarantined. Three pinned Crash Restraint snapshots were read in full. No raw
corpus, requested-resource fact, active training artifact, or protected
evaluation artifact is changed.
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
V456 = DATA / "manual_reviews/context_merit_audit_v456"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V456), str(V290)]

import build_context_merit_audit_v456 as previous
import build_context_merit_audit_v290 as core
from qa_quality import stable_fact_id


AUDIT = HERE / "context_merit_audit_v457.jsonl"
CURATION = HERE / "pending_curation_context_merit_v457.jsonl"
ADDITIONS = HERE / "replacement_additions_context_merit_v457.jsonl"
REPORT = HERE / "report_context_merit_v457.json"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
BASELINE_ROWS = 493
BASELINE_SHA256 = "c399f0175f6f10837c0b2e582ca3453a13c768838f18ae76225529547ebf237b"
EXPECTED_OUTPUT_ROWS = 492
EXPECTED_OUTPUT_SHA256 = "029a54332329eb317b2d14b6acb7cf34e7f3dc3fef269cd0b9c21a0de4b7b903"
EXPECTED_ADDITIONS_SHA256 = "7adcc24da1a17f23fd1c324c079f77bab50debe38b8a3444edf10c77f35f6623"
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
CRASH_OWNER_FACT_ID = "fact-95c6c6763c0722f57258"
REVIEWED_AT = "2026-07-16"
REVIEWER = "codex-context-merit-audit-v457"

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
portable = previous.portable
conservative_capacity = previous.conservative_capacity


SOURCES = {
    "kit": {
        "path": "data/raw/crash_restraint_building_rope_kit_20260714.json",
        "file_sha256": "b27d6f73f317bb34ed982b2d9348fbe09ee25c8ee7968c7bbb2cd34950a3c386",
        "text_sha256": "fe4c22af6a09fc40d7cd142fc56046161f04cd3d12980cc4f4e5455ed3712cf2",
        "chars": 1193,
        "url": "https://crash-restraint.com/ties/3",
    },
    "getting_started": {
        "path": "data/raw/crash_restraint_getting_started_20260714.json",
        "file_sha256": "7475e17fcd75dac8645f54fe1f894b5887c47a676f5261e85d668a0347bef4cf",
        "text_sha256": "37750811075d061caf9cd9a3067e3ff8bb0a17bc06fd96d7629db161315858a7",
        "chars": 1984,
        "url": "https://crash-restraint.com/ties/107",
    },
    "negotiation": {
        "path": "data/raw/crash_restraint_negotiation_consent_20260714.json",
        "file_sha256": "2541d79b35fc188d08f7940af444f892101305de2831c8e74b31db83ef53dd3f",
        "text_sha256": "353114c2622953d7b524e881e8291be284fd73d38d2caa78ce29bf56c940ea70",
        "chars": 2958,
        "url": "https://crash-restraint.com/ties/272",
    },
}


SCREENING_EVIDENCE = """Crash Restraint highly recommends using the Rope Bottoms' Share Group to check people before letting them tie you. Predators may move between cities after bans or establish themselves as community leaders. Checking potential partners through the group and by asking other local bottoms is not a silver bullet, but can greatly reduce risk."""

INSTRUCTOR_EVIDENCE = """There are no standards for rope education, and a tremendous amount of misinformation and confusion gets passed around. Attending classes at reputable community venues does not necessarily mean the information is correct. Instructors can be hired to teach at top-tier venues or websites even if their content is misleading or omits crucial safety information. Absorb what you can with a grain of salt. Ask whether what you are told makes sense and whether it agrees with what you have learned from other sources. In person, ask lots of "why" questions; if an instructor cannot answer, they may be repeating material without understanding it."""

CONSTRUCTION_EVIDENCE = """The page advises against rope with an outer braided sheath around a straight-fiber core, including kernmantle construction, because cored rope is difficult to handle for bondage and produces bulky knots. It recommends either twisted rope, most commonly three-strand but sometimes four-strand, or a solid braid with no core."""

SAFEWORD_EVIDENCE = """Especially in unexpected sexual situations, people very commonly find themselves unable to safeword or otherwise communicate. Fear, overwhelm, or shock can activate survival instincts that override normal decision-making, and restraints can make freezing the most available response. This is expected human brain chemistry, not evidence that someone is irresponsible or weak-willed. You cannot rely on a partner being able to speak before they are hurt, which makes good up-front negotiation important."""


SPECS = {
    "fact-e5ce60c4fd9db37aee2c": {
        "active_index": 54,
        "question": "How does Crash Restraint recommend screening a potential rope partner?",
        "answer": "Crash Restraint recommends checking a potential partner through the Rope Bottoms' Share Group and asking other local bottoms; neither method is foolproof, but both can reduce risk.",
        "decision": "edit",
        "edited_question": "What two checks does Crash Restraint recommend before being tied by a new rope partner, and what limitation does it note?",
        "edited_answer": "It recommends checking the person through the Rope Bottoms' Share Group and asking local bottoms. Neither check is foolproof, but both can reduce risk.",
        "source_key": "getting_started",
        "support_evidence": SCREENING_EVIDENCE,
        "subject_scope": "partner screening with an explicit limitation",
        "reason_code": "repair_screening_prompt_and_pin_full_snapshot",
        "reason": "The replacement asks directly for both screening channels and the no-silver-bullet caveat, while replacing an excerpt hash with the hash of the fully reviewed source snapshot.",
        "review_class": "partner_screening_provenance_edit",
    },
    "fact-ffae0ea0e90cd5b475fe": {
        "active_index": 55,
        "question": "How does Crash Restraint suggest testing a rope instructor's understanding?",
        "answer": "Crash Restraint recommends asking many \"why\" questions, checking whether the answers make sense and agree with other sources, and being wary when an instructor cannot explain the material.",
        "decision": "drop",
        "source_key": "getting_started",
        "support_evidence": INSTRUCTOR_EVIDENCE,
        "subject_scope": "instructor due diligence already retained in a requested-resource fact",
        "reason_code": "quarantine_instructor_guidance_duplicate_of_owner_fact",
        "reason": "The preserved Crash Restraint directory fact already gives the requested URL and the page's ask-why rather than accept-claims guidance. Keeping this second row would double-weight the same resource-evaluation point.",
        "review_class": "requested_resource_overlap_drop",
        "substantive_survivors": [CRASH_OWNER_FACT_ID],
    },
    "fact-adc1ff28de883b23cbe2": {
        "active_index": 304,
        "question": "What rope-length mix does Crash Restraint recommend when building a kit?",
        "answer": "Build the kit mainly from the longest rope you can handle comfortably, plus several shorter ropes for finishing ties.",
        "decision": "keep",
        "source_key": "kit",
        "subject_scope": "comfort-limited kit length mix",
        "reason_code": "keep_practical_rope_length_mix",
        "reason": "The row gives adaptable kit-building advice without inventing a fixed rope length, and its question and answer are concise and directly supported.",
        "review_class": "equipment_practical_keep",
    },
    "fact-0e71801b2635ca535827": {
        "active_index": 402,
        "question": "Which reactions does Crash Restraint say new partners should discuss how to interpret?",
        "answer": "Laughter, silence, crying, and becoming non-verbal, because those reactions can mean different things for different people.",
        "decision": "edit",
        "edited_question": "Why should new rope partners discuss how to interpret laughter, silence, crying, or becoming non-verbal?",
        "edited_answer": "Each reaction can mean different things for different people, and prior experience may help the tying partner recognize what is happening.",
        "source_key": "negotiation",
        "subject_scope": "individual interpretation of ambiguous scene reactions",
        "reason_code": "repair_reaction_list_as_practical_why_question",
        "reason": "The replacement turns a list-recall prompt into the practical reason for discussing ambiguous reactions while preserving the source's uncertainty.",
        "review_class": "reaction_interpretation_edit",
    },
    "fact-5981b7ffa1fa4c8f4e56": {
        "active_index": 404,
        "question": "Which rope constructions does Crash Restraint recommend for bondage, and why does it advise against cored rope?",
        "answer": "It recommends twisted rope—usually three-strand, sometimes four-strand—or a solid braid with no core; it says cored constructions such as kernmantle are difficult to handle for bondage and make bulky knots.",
        "decision": "edit",
        "edited_question": "What coreless rope constructions does Crash Restraint recommend for bondage, and why does it discourage cored rope?",
        "edited_answer": "It recommends twisted rope—usually three-strand, sometimes four-strand—or a solid braid with no core. It discourages cored constructions such as kernmantle because it considers them difficult to handle for bondage and says they make bulky knots.",
        "source_key": "kit",
        "support_evidence": CONSTRUCTION_EVIDENCE,
        "subject_scope": "source-attributed coreless construction guidance",
        "reason_code": "normalize_construction_evidence_and_pin_full_snapshot",
        "reason": "The replacement keeps the useful selection rationale, makes the source's judgment explicit, removes duplicated evidence, and replaces its evidence hash with the full snapshot hash.",
        "review_class": "equipment_provenance_edit",
    },
    "fact-86febe48f917a6c97935": {
        "active_index": 419,
        "question": "Why are up-front negotiation and active check-ins needed even when rope partners use a safeword?",
        "answer": "Fear, overwhelm, or shock can make someone freeze or unable to communicate, so partners cannot rely on a safeword alone and should agree on signals and check-ins beforehand.",
        "decision": "edit",
        "edited_question": "Why does Crash Restraint say rope partners should not rely on a safeword alone?",
        "edited_answer": "Fear, overwhelm, or shock can make someone freeze or unable to communicate, so Crash Restraint recommends good up-front negotiation rather than assuming a partner will always be able to safeword.",
        "source_key": "negotiation",
        "support_evidence": SAFEWORD_EVIDENCE,
        "subject_scope": "communication failure as a reason for up-front negotiation",
        "reason_code": "remove_overpacked_checkin_claim_and_pin_full_snapshot",
        "reason": "The replacement asks one causal question, stays within one exact source passage, removes duplicated evidence, and replaces the excerpt hash with the full snapshot hash.",
        "review_class": "consent_provenance_edit",
    },
    "fact-9d22482f111a0c1e3ddf": {
        "active_index": 434,
        "question": "Why does Crash Restraint advise beginners to avoid black or very dark rope?",
        "answer": "Dark rope makes it harder for both the learner and the teacher to see what is happening.",
        "decision": "keep",
        "source_key": "kit",
        "subject_scope": "beginner visibility when choosing rope color",
        "reason_code": "keep_beginner_color_visibility_reason",
        "reason": "The row is a concise, practical visibility consideration for both learner and teacher and avoids presenting color as a general safety certification.",
        "review_class": "equipment_practical_keep",
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
        raise ValueError("v456 baseline drift")


def build_projection(out: Path, report: Path) -> None:
    if len(read_jsonl(ADDITIONS)) != 4:
        raise ValueError("v457 replacement addition drift")
    if EXPECTED_ADDITIONS_SHA256 != "PENDING" and file_sha256(ADDITIONS) != EXPECTED_ADDITIONS_SHA256:
        raise ValueError("v457 replacement addition hash drift")
    inputs = out.parent / f".{out.name}.v457-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v456.jsonl"
    build_baseline(base, inputs / "v456.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base, ADDITIONS))


def validate_source(row: dict, spec: dict, evidence: str) -> dict:
    source = SOURCES[spec["source_key"]]
    source_path = ROOT / source["path"]
    if file_sha256(source_path) != source["file_sha256"]:
        raise ValueError(f"source file drift: {row['fact_id']}")
    document = json.loads(source_path.read_text())
    if document["url"] != source["url"] or row["url"] != source["url"]:
        raise ValueError(f"source URL drift: {row['fact_id']}")
    if len(document["text"]) != source["chars"]:
        raise ValueError(f"source character drift: {row['fact_id']}")
    if text_sha256(document["text"]) != source["text_sha256"]:
        raise ValueError(f"source text drift: {row['fact_id']}")
    if evidence not in document["text"]:
        raise ValueError(f"evidence absent from full source snapshot: {row['fact_id']}")
    if spec["decision"] == "keep" and row["document_sha256"] != source["text_sha256"]:
        raise ValueError(f"kept row lacks full-snapshot hash: {row['fact_id']}")
    return source


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v457-observe-", dir=HERE) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v457-base-", dir=HERE) as tmp:
        directory = Path(tmp)
        base = directory / "v456.jsonl"
        build_baseline(base, directory / "v456.report.json")
        before = read_jsonl(base)
    by_fact = {row["fact_id"]: (index, row) for index, row in enumerate(before, 1)}
    additions, audits, curations = [], [], []

    for audit_index, (fact_id, spec) in enumerate(SPECS.items(), 1):
        active_index, row = by_fact[fact_id]
        if active_index != spec["active_index"]:
            raise ValueError(f"active index drift: {fact_id}")
        if row["question"] != spec["question"] or row["answer"] != spec["answer"]:
            raise ValueError(f"active Q&A drift: {fact_id}")
        evidence = spec.get("support_evidence", row["evidence"])
        source = validate_source(row, spec, evidence)
        decision = spec["decision"]

        if decision == "edit":
            edited_fact_id = stable_fact_id(spec["edited_question"], spec["edited_answer"])
            curations.append({
                "action": "drop",
                "expected_answer": row["answer"],
                "expected_question": row["question"],
                "fact_id": fact_id,
                "reason": spec["reason"],
                "reason_code": spec["reason_code"],
                "replacement_fact_id": edited_fact_id,
                "reviewed_at": REVIEWED_AT,
                "reviewer": REVIEWER,
            })
            replacement = dict(row)
            replacement.update({
                "answer": spec["edited_answer"],
                "curation": {
                    "action": "edit_via_fully_pinned_replacement",
                    "decision_file": portable(CURATION),
                    "original_fact_id": fact_id,
                    "paraphrase_rationale": spec["reason"],
                    "reason": spec["reason"],
                    "reason_code": spec["reason_code"],
                    "reviewed_at": REVIEWED_AT,
                    "reviewer": REVIEWER,
                    "support_type": "manual_paraphrase",
                },
                "document_sha256": source["text_sha256"],
                "evidence": evidence,
                "evidence_sha256": text_sha256(evidence),
                "evidence_url": source["url"],
                "fact_id": edited_fact_id,
                "paraphrase_rationale": spec["reason"],
                "quality_schema": "curated-qa-v1",
                "question": spec["edited_question"],
                "reviewer": REVIEWER,
                "source_lineage": {
                    "artifact": portable(ADDITIONS),
                    "raw_document": source["path"],
                    "resource_manifest": portable(RESOURCE_MANIFEST),
                },
                "text": f"Question: {spec['edited_question']}\nAnswer: {spec['edited_answer']}",
                "url": source["url"],
            })
            additions.append(replacement)
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
            "active_document_sha256": row["document_sha256"],
            "active_index": active_index,
            "active_question": row["question"],
            "audit_index": audit_index,
            "decision": decision,
            "fact_id": fact_id,
            "full_snapshot_hash_repaired": decision == "edit" and row["document_sha256"] != source["text_sha256"],
            "projection_lineage": {
                "baseline_rows": BASELINE_ROWS,
                "baseline_sha256": BASELINE_SHA256,
                "prior_context_merit_review": True,
                "source_absent_from_recent_passes": "v448-v456",
            },
            "reason": spec["reason"],
            "reason_code": spec["reason_code"],
            "review_class": spec["review_class"],
            "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER,
            "schema": "context-merit-audit-v457",
            "source": row["source"],
            "source_document": source["path"],
            "source_document_chars": source["chars"],
            "source_document_file_sha256": source["file_sha256"],
            "source_document_text_sha256": source["text_sha256"],
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
    write_jsonl(ADDITIONS, additions)
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

    with tempfile.TemporaryDirectory(prefix=".v457-resource-check-", dir=HERE) as tmp:
        projected = Path(tmp) / "projected.jsonl"
        build_projection(projected, Path(tmp) / "projected.report.json")
        projected_rows = read_jsonl(projected)
        blob = projected.read_text()
    urls = requested_urls()
    owner_rows = {row["fact_id"]: row for row in projected_rows if row.get("source") == OWNER_SOURCE}
    if len(urls) != 24 or any(url not in blob for url in urls):
        raise ValueError("requested resource coverage drift")
    if set(owner_rows) != set(OWNER_FACT_IDS) or CRASH_OWNER_FACT_ID not in owner_rows:
        raise ValueError("owner resource fact drift")

    curation_counts = Counter(row["action"] for row in curations)
    REPORT.write_text(json.dumps({
        "audit": {
            "by_decision": dict(Counter(row["decision"] for row in audits)),
            "by_review_class": dict(Counter(row["review_class"] for row in audits)),
            "full_snapshot_hash_repairs": sum(row["full_snapshot_hash_repaired"] for row in audits),
            "path": portable(AUDIT),
            "rows": len(audits),
            "sha256": file_sha256(AUDIT),
        },
        "baseline_checkpoint": {
            "commit": "a777f45329af5c42606e791494b1f7cfc2aa1467",
            "rows": BASELINE_ROWS,
            "sha256": BASELINE_SHA256,
            "version": "v456",
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
            "natural_or_provenance_repairs": 4,
            "requested_resource_overlap_quarantined": 1,
        },
        "isolated_build_projection": {
            "automated_projection_runs": 2,
            "new_additions_applied": len(additions),
            "output_rows": observation["rows"],
            "output_sha256": observation["sha256"],
            "repeat_dataset_byte_identical": observation["dataset_equal"],
            "repeat_projection_report_byte_identical": observation["report_equal"],
            "sealed_eval_fact_count_reported_by_tooling": observation["eval"],
        },
        "manual_slice": {
            "candidate_rows": len(SPECS),
            "recent_passes_without_source": [f"v{version}" for version in range(448, 457)],
            "source": "crash_restraint",
            "subjects": ["partner and instructor due diligence", "rope-kit choices", "negotiation and reaction interpretation"],
        },
        "new_pending_curation": {
            "by_action": dict(curation_counts),
            "decisions": len(curations),
            "path": portable(CURATION),
            "sha256": file_sha256(CURATION),
        },
        "replacement_additions": {
            "path": portable(ADDITIONS),
            "rows": len(additions),
            "sha256": file_sha256(ADDITIONS),
        },
        "requested_resource_coverage": {
            "crash_restraint_owner_fact_present": CRASH_OWNER_FACT_ID in owner_rows,
            "manifest_urls": len(urls),
            "manifest_urls_present": sum(url in blob for url in urls),
            "owner_resource_facts": len(OWNER_FACT_IDS),
        },
        "schema": "context-merit-audit-report-v457",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
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
            "total_unique_characters": sum(source["chars"] for source in SOURCES.values()),
        },
        "training_layer_contract": {
            "cleaned_markdown_site_corpora": "Distinct first-class source artifacts; no raw corpus output was edited or ingested by v457.",
            "deduplication_policy": "Do not collapse a cleaned Markdown document and its derived Q&A merely because they express the same source knowledge.",
            "derived_qa": "Distinct first-class training layer; v457 changes only existing v456 Q&A after full review of three pinned source snapshots.",
            "provenance_requirement": "Every reviewed row is pinned to source URL, full-document text hash, snapshot file hash, and exact supporting evidence.",
        },
        "v52_isolation": {
            "active_v52_inputs_modified": False,
            "candidate_status": "isolated_pending_not_promoted",
        },
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
