#!/usr/bin/env python3
"""Curate ten existing teaching, evaluation, and accessibility Q&As.

The only candidate base is sealed v450. Five rows receive evidence-backed
scope or natural-question repairs, three navigation/catalog/promotional rows
are quarantined, and two useful access-format rows are retained. No new corpus
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
V450 = DATA / "manual_reviews/context_merit_audit_v450"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V450), str(V290)]

import build_context_merit_audit_v450 as previous
import build_context_merit_audit_v290 as core
from qa_quality import stable_fact_id


AUDIT = HERE / "context_merit_audit_v451.jsonl"
CURATION = HERE / "pending_curation_context_merit_v451.jsonl"
REPORT = HERE / "report_context_merit_v451.json"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
BASELINE_ROWS = 507
BASELINE_SHA256 = "21c303744ff3407bb296ffff9a46600f9a7cea9d6ef5d9c47fce6d7ffcc28051"
EXPECTED_OUTPUT_SHA256 = "d91cbfbc0c4865e57e84f9a187402f39573b2039d075fa4159e7262023ba3ebe"
EXPECTED_CAPACITY_BEFORE = {
    "conflict_units": 259,
    "equipment_material": 21,
    "resources_general": 78,
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
REVIEWER = "codex-context-merit-audit-v451"

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
portable = previous.portable
conservative_capacity = previous.conservative_capacity


SPECS = {
    "fact-25dbc4015907f5fd7399": {
        "active_index": 96,
        "question": "How does Rope365 suggest a beginner navigate its Spring curriculum?",
        "answer": "After covering Getting Started, begin with Weeks 1–2 for core techniques; later chapters can be explored nonlinearly to match personal needs.",
        "decision": "drop",
        "source_document": "data/raw/rope_resources_v1/rope365__8968990b22adfe53bf97.json",
        "source_document_file_sha256": "2efb7c7a63123f44e75245ede3d4c89bcd99a718e4ea7ecea82b092115eb09d8",
        "source_document_chars": 1683,
        "knowledge_scope": "site-specific curriculum navigation",
        "reason_code": "quarantine_rope365_curriculum_navigation",
        "reason": "The answer is navigation for one site's named seasonal curriculum rather than durable rope knowledge, teaching evaluation, or an actionable safety principle. Many substantive Rope365 facts remain active.",
        "review_class": "curriculum_navigation_drop",
    },
    "fact-1f9a55dbfeb8ef49d682": {
        "active_index": 107,
        "question": "How does Tethered Together route event accessibility requests versus questions about the facility itself?",
        "answer": "Submit event requests through its accessibility channel, but direct facility-access questions to the host hotel.",
        "decision": "keep",
        "source_document": "data/raw/rope_resources_v1/tethered_together__0b96b6c5488d3a68347e.json",
        "source_document_file_sha256": "fd70ba59fc91a356a2def96b5911492d54652fa3c8fd328a99f11874e4d0ad8a",
        "source_document_chars": 2007,
        "evidence_layout": "noncontiguous_exact_paragraphs",
        "knowledge_scope": "actionable division of event and facility accessibility responsibility",
        "reason_code": "keep_accessibility_request_responsibility_boundary",
        "reason": "The row gives an actionable and source-specific accessibility boundary: the event handles attendee requests, while the host hotel answers facility-access questions.",
        "review_class": "accessibility_routing_keep",
    },
    "fact-8c0b17522514f90e0aff": {
        "active_index": 140,
        "question": "In “On Not Teaching Rope,” how are competence and knowledge distinguished?",
        "answer": "Competence is important for keeping your partner safe. Knowledge is important, even essential, for teaching others how to become competent.",
        "decision": "edit",
        "edited_question": "What distinction does the author of “On Not Teaching Rope” draw between competent execution and teaching?",
        "edited_answer": "He argues that competence supports safer execution, while teaching also requires understanding how and why the tie works well enough to help other people become competent.",
        "evidence": "The major problem with this is that when you learn rope, particularly the way most Westerners learn rope in intensives or workshops, you are learning to be competent, not knowledgeable. That is distinction is incredibly important when it comes to teaching. Competence is important for keeping your partner safe. Knowledge is important, even essential, for teaching others how to become competent.\nCompetence is important. It is what is going to keep your partner safe, but it doesn’t mean you understand what you are doing and why you are doing it, which are the minimal qualifications for teaching someone else how to do it.",
        "evidence_layout": "noncontiguous_exact_paragraphs",
        "source_document": "data/raw/kinbakutoday_1775dd4176b24104.json",
        "source_document_file_sha256": "baafd0d56da7e144e69ac5466e90dd59e518abd09c42e047d585e122742676ff",
        "source_document_chars": 10177,
        "knowledge_scope": "author's teaching criterion, not a universal credential rule",
        "reason_code": "naturalize_competence_vs_teaching_knowledge_distinction",
        "reason": "The replacement identifies the claim as the author's argument and softens the original safety absolute while preserving the core distinction between execution and explanatory understanding.",
        "review_class": "teaching_knowledge_edit",
    },
    "fact-f95de33b068862ee0807": {
        "active_index": 229,
        "question": "What does Shibari Study's free General Rope Safety course cover?",
        "answer": "It covers rope types, anatomy, and communication in the context of shibari risks, with related topics including consent and negotiation, cutting tools, nerve concerns, fainting, pain and personal risk profiles, and psychological considerations.",
        "decision": "edit",
        "edited_question": "Which subjects does Shibari Study’s listing explicitly name for its free General Rope Safety course?",
        "edited_answer": "The listing says the course focuses on shibari safety and risk and includes rope types, anatomy, and communication.",
        "evidence": "This free course focuses on bondage safety and risks associated with shibari. This includes information about different types of rope, the anatomy, and communication.",
        "source_document": "data/rope_resource_manual_v1.jsonl",
        "source_document_file_sha256": "caee3a972ee70f6e8f23cd3de561bc673c54705e64df58cfc0c867e75995c3bc",
        "source_document_chars": 11044,
        "source_kind": "manual_resource_jsonl",
        "source_fact_id": "fact-e5480624bacddfe277a5",
        "knowledge_scope": "course-listing claim, not validated safety instruction",
        "reason_code": "reduce_general_safety_course_to_exact_listing_scope",
        "reason": "The replacement removes an expanded topic catalog and duplicated evidence that the cited manual-fact row does not support, retaining only the exact course-listing claim.",
        "review_class": "course_scope_provenance_edit",
    },
    "fact-dd266500c73a6b1894fb": {
        "active_index": 258,
        "question": "What in-person and virtual learning formats does Reclamation Rope describe?",
        "answer": "In-person labs and tastings offer beginner instruction, peer learning, and top-or-bottom experience; virtual vibe sessions offer online practice, questions, stories, and peer sharing.",
        "decision": "keep",
        "source_document": "data/raw/rope_resources_v1/atx_empty_space__a9598a537db1b5e31278.json",
        "source_document_file_sha256": "c9a18b8318cca11f0d806e74883260b8c477b4c03a55e8b1200669fd9f53b6f5",
        "source_document_chars": 1425,
        "evidence_layout": "noncontiguous_exact_paragraphs",
        "knowledge_scope": "access formats for a BIPOC community education program",
        "reason_code": "keep_in_person_and_remote_access_formats",
        "reason": "The row concisely distinguishes in-person and remote participation formats without retaining volatile calendar dates, prices, or signup navigation.",
        "review_class": "learning_access_format_keep",
    },
    "fact-cf7359b9fc5365a0f2bc": {
        "active_index": 321,
        "question": "What should a beginner verify when choosing an online shibari school?",
        "answer": "First decide whether its aesthetic and philosophy match their goals, then check the teacher's training and teaching ability through credentials and feedback from students.",
        "decision": "edit",
        "edited_question": "What fit and evidence checks does Esinem suggest when evaluating an online shibari school?",
        "edited_answer": "He suggests first checking whether the school’s aesthetics and philosophy fit the learner’s goals, then asking who trained the teacher, reviewing the teacher’s background, and asking students what and how deeply the teacher teaches.",
        "source_document": "data/raw/esinem_7c6ce8a699d42f64.json",
        "source_document_file_sha256": "4a38350ca0edb0dd7c482ba7e400860624586e76b65dcf51b8824122c0efd871",
        "source_document_chars": 3731,
        "knowledge_scope": "source-attributed school-fit and evidence checks, not proof of competence",
        "reason_code": "make_online_school_evaluation_checks_concrete_and_attributed",
        "reason": "The replacement turns the abstract phrase 'check credentials and feedback' into the source's concrete questions while keeping the advice explicitly attributed rather than treating those checks as proof.",
        "review_class": "teacher_evaluation_edit",
    },
    "fact-56aa8740933974aa3631": {
        "active_index": 327,
        "question": "What should someone understand before progressing to advanced suspension training?",
        "answer": "They need solid basic competence and should understand nerve-injury risks and common mistakes before progressing.",
        "decision": "edit",
        "edited_question": "Which prerequisites does Esinem’s suspension-course page tell learners not to skip?",
        "edited_answer": "It says learners should already have experience with fairly complex ties and should study nerve injury and common mistakes; otherwise, they should return to basic practice rather than use the course as a shortcut.",
        "source_document": "data/raw/esinem_aed5c1bb8a4ad341.json",
        "source_document_file_sha256": "32ee0dd41f491b8451bec428de7a59a3315869a4fafd2ebe951b8b054c853545",
        "source_document_chars": 1783,
        "evidence_layout": "noncontiguous_exact_paragraphs",
        "knowledge_scope": "prerequisite warning extracted from a commercial course page",
        "reason_code": "replace_advanced_training_generalization_with_exact_prerequisites",
        "reason": "The original synthetic question calls this 'advanced suspension training,' whereas the page sells an introductory suspension bundle. The replacement strips the sales claim and preserves its concrete do-not-skip-the-basics warning.",
        "review_class": "suspension_prerequisite_edit",
    },
    "fact-b208bf4d56ecbcdd377a": {
        "active_index": 332,
        "question": "What teaching value does Kinbaku Today attribute to Nureki Chimuo’s World of Rope video?",
        "answer": "The article says the video pairs Nureki’s signature tying with CineMagic’s production to make his technique and approach easy to follow, then uses a series of ties and poses to emphasize aesthetic form.",
        "decision": "drop",
        "source_document": "data/raw/kinbakutoday_b776fccd348e2538.json",
        "source_document_file_sha256": "a2f63bc64be180ef08f7d7056951d1e2381a3ebc00ea9a072f2dac3c071d8891",
        "source_document_chars": 2183,
        "knowledge_scope": "promotional video description and title-specific trivia",
        "reason_code": "quarantine_promotional_video_teaching_value",
        "reason": "The article is announcing a commercial download and the row repeats its qualitative promotional assessment rather than a durable teaching principle or independently supported evaluation.",
        "review_class": "promotional_video_trivia_drop",
    },
    "fact-5dd7bd5d46d0b5a85c47": {
        "active_index": 402,
        "question": "Which foundational topics does Shibari Study list for its Beginner 1 level?",
        "answer": "Beginner 1 covers single- and double-column ties, basic frictions, and rope handling.",
        "decision": "drop",
        "source_document": "sources/manual_facts/resource_group_a.jsonl",
        "source_document_file_sha256": "281bd54dbc594fcdc6f35fde28ef3472bf6e5b759336cd5fe6338c25af877963",
        "source_document_chars": 3743,
        "source_kind": "manual_fact_packet",
        "source_question": "Which foundational topics does Shibari Study list for its Beginner 1 level?",
        "knowledge_scope": "vendor catalog-level navigation",
        "reason_code": "quarantine_shibari_study_level_catalog_trivia",
        "reason": "The row is a vendor's level-label catalog summary, not an explanation of the techniques or why they are foundational. Owner fact 966eb937bef6831c819a preserves the requested Shibari Study resource and its public free primers.",
        "review_class": "course_catalog_trivia_drop",
        "redundant_survivors": ["fact-966eb937bef6831c819a"],
    },
    "fact-1b9797430b377f3720df": {
        "active_index": 436,
        "question": "Why can teaching or explaining rope deepen the teacher’s understanding?",
        "answer": "Putting rope knowledge into words requires and tests a deep understanding, which can reveal what the teacher still needs to learn.",
        "decision": "edit",
        "edited_question": "What two reflective practices does Rope365’s sharing page say can reveal gaps in a practitioner’s understanding?",
        "edited_answer": "Explaining rope work in words tests whether it is understood deeply, while reviewing photos or video can reveal details missed during the act of tying.",
        "source_document": "data/raw/rope_resources_v1/rope365__6e0315421ed26706d440.json",
        "source_document_file_sha256": "bb5f4022a33756bc115b53c21cb71660791780f71e51087e20fcdc2f190bd47e",
        "source_document_chars": 3933,
        "knowledge_scope": "reflective teaching and visual self-review",
        "reason_code": "complete_teaching_reflection_with_visual_review",
        "reason": "The replacement asks a natural evaluation question and restores the source's second feedback mechanism: reviewing images or video for details missed while tying.",
        "review_class": "reflective_practice_edit",
    },
}


OWNER_SURVIVORS = {
    "fact-966eb937bef6831c819a": "https://shibaristudy.com/",
    "fact-749ad10e8c33dcc729eb": "https://tetheredtogether.net/",
    "fact-433bdbf1247c4ed36895": "https://www.atxempty.space/space-schedule",
}


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v450 baseline drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v451-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v450.jsonl"
    build_baseline(base, inputs / "v450.report.json")
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
    if spec.get("source_kind") == "manual_fact_packet":
        if len(source_path.read_text()) != spec["source_document_chars"]:
            raise ValueError(f"manual packet length drift: {row['fact_id']}")
        entry = next(item for item in read_jsonl(source_path) if item["question"] == spec["source_question"])
        if entry["evidence_url"] != row["url"] or entry["evidence"] != evidence:
            raise ValueError(f"manual packet evidence drift: {row['fact_id']}")
        return
    document = json.loads(source_path.read_text())
    if document["url"] != row["url"] or len(document["text"]) != spec["source_document_chars"]:
        raise ValueError(f"source document metadata drift: {row['fact_id']}")
    if spec.get("evidence_layout") == "noncontiguous_exact_paragraphs":
        paragraphs = evidence.split("\n")
        if len(paragraphs) < 2 or any(paragraph not in document["text"] for paragraph in paragraphs):
            raise ValueError(f"exact evidence paragraph absent from source: {row['fact_id']}")
    elif evidence not in document["text"]:
        raise ValueError(f"evidence absent from full source snapshot: {row['fact_id']}")


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v451-observe-", dir=HERE) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v451-base-", dir=HERE) as tmp:
        directory = Path(tmp)
        base = directory / "v450.jsonl"
        build_baseline(base, directory / "v450.report.json")
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
            "fact_id": fact_id, "knowledge_scope": spec["knowledge_scope"],
            "projection_lineage": {
                "baseline_rows": BASELINE_ROWS, "baseline_sha256": BASELINE_SHA256,
                "prior_context_merit_review": True,
            },
            "reason": spec["reason"], "reason_code": spec["reason_code"],
            "review_class": spec["review_class"], "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER, "schema": "context-merit-audit-v451",
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
    if (observation["rows"], observation["eval"]) != (504, 612):
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
            "by_knowledge_scope": dict(Counter(row["knowledge_scope"] for row in audits)),
            "by_review_class": dict(Counter(row["review_class"] for row in audits)),
            "path": portable(AUDIT), "rows": len(audits), "sha256": file_sha256(AUDIT),
        },
        "conservative_capacity": {
            "after": observation["after"], "before": observation["before"],
            "delta": {key: observation["after"][key] - observation["before"][key] for key in observation["before"]},
            "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
        },
        "curation_outcome": {
            "teaching_or_evaluation_repairs": 5,
            "useful_access_rows_kept": 2,
            "navigation_catalog_or_promotion_rows_quarantined": 3,
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
        "schema": "context-merit-audit-report-v451",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "training_layer_contract": {
            "cleaned_markdown_site_corpora": "Distinct first-class future training artifacts; no corpus output was ingested by v451.",
            "derived_qa": "Distinct first-class training layer; v451 changes only existing sealed Q&A using attached evidence and pinned source snapshots.",
            "deduplication_policy": "Do not collapse a cleaned Markdown document and its derived Q&A merely because they express the same source knowledge.",
            "provenance_requirement": "Link future derived Q&A to cleaned source documents through stable source URL and document or snapshot hashes.",
        },
        "v52_isolation": {"active_v52_inputs_modified": False, "candidate_status": "isolated_pending_not_promoted"},
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
