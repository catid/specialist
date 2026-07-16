#!/usr/bin/env python3
"""Curate ten existing anatomy, breathing, joint, and nerve-safety Q&As.

The only candidate base is sealed v449. Six rows receive evidence-backed
scope or safety repairs, two vague or symptom-treatment rows are quarantined,
and two complete breathing rows are retained. No new corpus or protected
evaluation artifact is read or changed.
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
V449 = DATA / "manual_reviews/context_merit_audit_v449"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V449), str(V290)]

import build_context_merit_audit_v449 as previous
import build_context_merit_audit_v290 as core
from qa_quality import stable_fact_id


AUDIT = HERE / "context_merit_audit_v450.jsonl"
CURATION = HERE / "pending_curation_context_merit_v450.jsonl"
REPORT = HERE / "report_context_merit_v450.json"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
BASELINE_ROWS = 509
BASELINE_SHA256 = "a94794c4564082bf43c64d4cd92203f4319e680e388e40a087511ab57965b950"
EXPECTED_OUTPUT_SHA256 = "21c303744ff3407bb296ffff9a46600f9a7cea9d6ef5d9c47fce6d7ffcc28051"
EXPECTED_CAPACITY_BEFORE = {
    "conflict_units": 260,
    "equipment_material": 21,
    "resources_general": 78,
    "safety_consent": 85,
    "technique": 76,
}
EXPECTED_CAPACITY_AFTER = {
    "conflict_units": 259,
    "equipment_material": 21,
    "resources_general": 78,
    "safety_consent": 84,
    "technique": 76,
}
OWNER_SOURCE = previous.OWNER_SOURCE
OWNER_FACT_IDS = previous.OWNER_FACT_IDS
REVIEWED_AT = "2026-07-16"
REVIEWER = "codex-context-merit-audit-v450"

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
portable = previous.portable
conservative_capacity = previous.conservative_capacity


SPECS = {
    "fact-68885affbfbcd4150b1a": {
        "active_index": 27,
        "question": "Before exploring restrictive elbow positions, what preparation does Rope365 recommend?",
        "answer": "Rope365 recommends warming up the arms, especially the shoulders, and identifying positions that can be sustained comfortably.",
        "decision": "edit",
        "edited_question": "How does Rope365 suggest exploring which restrictive elbow positions suit a particular person?",
        "edited_answer": "It suggests warming up the arms, especially the shoulders, then trying different elbow angles to learn which positions that person finds difficult or comfortable.",
        "source_document": "data/raw/rope_resources_v1/rope365__e9ea1664e01d07630055.json",
        "source_document_file_sha256": "69f8ec3bfad08e7b6750575f953e08ba57f5a3ebcbc51c391ec5a420e0523da5",
        "source_document_chars": 2570,
        "medical_scope": "individual mobility exploration; warming up is not an injury guarantee",
        "reason_code": "scope_elbow_warmup_as_person_specific_exploration",
        "reason": "The replacement reflects the source's exploratory framing and person-to-person variation without implying that warming up makes a restrictive elbow position sustainable or safe.",
        "review_class": "mobility_exploration_edit",
    },
    "fact-900a637b768f23c85fe9": {
        "active_index": 36,
        "question": "How can an upward-pointing elbow be positioned to reduce the effect of gravity in a tie?",
        "answer": "Bring the arm close to the head so it aligns vertically with the shoulder; if it tires, lying down also removes gravity's pull.",
        "decision": "edit",
        "edited_question": "What positioning and fatigue response does Rope365 suggest for a tie with an upward-pointing elbow?",
        "edited_answer": "It suggests bringing the arm toward the head so it is closer to vertical over the shoulder; if the arm tires, move to the ground to remove the continuing pull of gravity rather than sustaining the loaded position.",
        "source_document": "data/raw/rope_resources_v1/rope365__b4e7fccc6f185c6f124d.json",
        "source_document_file_sha256": "8ae753672a554590d33f1350752ce3573913a0c9be1cc0fb137829f9b22741d8",
        "source_document_chars": 12243,
        "evidence_layout": "noncontiguous_exact_paragraphs",
        "medical_scope": "gravity reduction and fatigue exit, not treatment of nerve or circulation symptoms",
        "reason_code": "complete_upward_elbow_position_with_fatigue_release",
        "reason": "The replacement makes the source's fatigue boundary operational and keeps alignment as a load-reduction strategy rather than a treatment for nerve or circulation symptoms.",
        "review_class": "fatigue_release_edit",
    },
    "fact-0176aea640642c18ec05": {
        "active_index": 101,
        "question": "How does Rope365 suggest coming out of a repeatedly practiced folded-leg tie?",
        "answer": "Gently unfold the knee, check in with the joint, get up slowly, and use a small amount of comfortable movement while the body readjusts.",
        "decision": "edit",
        "edited_question": "What boundary does Rope365’s repeated folded-leg practice note give if the practice is causing physical or mental harm?",
        "edited_answer": "It says to stop the repeated practice and find another way to proceed if sustaining it is causing physical or mental harm.",
        "evidence": "If, however, you feel that you are hurting yourself mentally or physically by sustaining the practice, stop and find another way to go about it.",
        "source_document": "data/raw/rope_resources_v1/rope365__fc84f60465325619f5ad.json",
        "source_document_file_sha256": "9532124244ce0595e8ed0a3f024e3ff60c2e643a1633e9d05c2f659e7320618a",
        "source_document_chars": 3471,
        "medical_scope": "stop boundary, replacing vague post-tie joint-recovery advice",
        "reason_code": "replace_joint_readjustment_shortcut_with_stop_boundary",
        "reason": "The original answer resembles unsupported recovery guidance and uses the vague phrase 'check in with the joint.' The replacement preserves the same source's clear and defensible stop boundary for repeated physical or mental harm.",
        "review_class": "joint_recovery_shortcut_edit",
    },
    "fact-fecdb35d2e926add5dff": {
        "active_index": 168,
        "question": "What breathing and release warning does Rope365 give for the shrimp tie?",
        "answer": "Bending forward makes breathing difficult and you have to be prepared to untie quickly.",
        "decision": "keep",
        "source_document": "data/raw/rope_resources_v1/rope365__3cf203e0becb0a3c87c6.json",
        "source_document_file_sha256": "05e69af7cb53a26e3d3d33803f458ae8d70d8a1bb1775e1e3a20d6a339408241",
        "source_document_chars": 2884,
        "medical_scope": "source-attributed breathing restriction and rapid-release readiness",
        "reason_code": "keep_shrimp_breathing_and_release_warning",
        "reason": "The row directly attributes the position's breathing restriction and the need for rapid-release readiness without adding a diagnosis, load guarantee, or symptom-treatment shortcut.",
        "review_class": "breathing_release_keep",
    },
    "fact-7ed7d35693518e461cca": {
        "active_index": 169,
        "question": "What breathing consideration does Rope365 give when adding a waistline?",
        "answer": "The waist is an important part of breathing, so adding a waistline can contribute to the tie's intensity.",
        "decision": "drop",
        "source_document": "data/raw/rope_resources_v1/rope365__03448edb627f81c3e962.json",
        "source_document_file_sha256": "2babe6157cd41893ed5afc70033e38320b82de43965d6c7aaae63a07a3de7efe",
        "source_document_chars": 3589,
        "medical_scope": "vague breathing-risk statement without a monitoring or release boundary",
        "reason_code": "quarantine_waist_breathing_intensity_fragment",
        "reason": "The row frames interference with breathing as generic 'intensity' but gives no prevention, monitoring, or release action. Active breathing facts fecdb35d2e926add5dff and 396cb8d124a8ac7de43c retain operational warnings for restrictive positions and tightening chest structures.",
        "review_class": "vague_breathing_intensity_drop",
        "redundant_survivors": ["fact-fecdb35d2e926add5dff", "fact-396cb8d124a8ac7de43c"],
    },
    "fact-c286658bc31e049f3caa": {
        "active_index": 179,
        "question": "What comfort check does Rope365 recommend when trying frog-tie starting positions?",
        "answer": "Rope365 recommends asking about any pain, especially knee overextension or pressure on the front of the shin.",
        "decision": "edit",
        "edited_question": "Which two distinct pain sources does Rope365’s frog-tie starting-point checklist ask partners to check?",
        "edited_answer": "It asks them to check for pain from overextending the knee and for localized pressure on the front of the tibia or shin.",
        "source_document": "data/raw/rope_resources_v1/rope365__e1b4d511580682c46d2d.json",
        "source_document_file_sha256": "fdd87653e483bf9e18059c23c0c8ab8b751fed5d799f0f15b347604c75aee240",
        "source_document_chars": 3178,
        "medical_scope": "separate joint-extension pain from local shin pressure",
        "reason_code": "distinguish_knee_extension_from_tibia_pressure",
        "reason": "The replacement makes the checklist's two mechanisms explicit rather than collapsing them into a generic comfort question.",
        "review_class": "joint_pressure_distinction_edit",
    },
    "fact-83d12c75b561a6ab6e14": {
        "active_index": 329,
        "question": "What should you monitor in the tied hand during a folded-arm chicken-wing tie?",
        "answer": "Continuously monitor hand movement and sensation, including pain or numbness, because the position can affect exposed arm nerves.",
        "decision": "edit",
        "edited_question": "What separate hand observations does Rope365’s chicken-wing checklist call for without assigning numbness to a single cause?",
        "edited_answer": "The checklist separately asks about painful sensations, hand sensation and movement, and whether numbness develops; the page discusses both exposed nerves and circulation rather than treating one sign as a diagnosis.",
        "source_document": "data/raw/rope_resources_v1/rope365__118a41bb0bd96d9bc195.json",
        "source_document_file_sha256": "910c3412388618bc65c1eaf7332b92a1b03541714cf233b52a06aa177395b94c",
        "source_document_chars": 2211,
        "evidence_layout": "noncontiguous_exact_paragraphs",
        "medical_scope": "observation separates pain, motor function, sensation, and numbness without diagnosis",
        "reason_code": "separate_hand_observations_and_preserve_cause_uncertainty",
        "reason": "The replacement preserves the checklist's separate observations and avoids implying that pain or numbness alone identifies nerve rather than circulation involvement.",
        "review_class": "nerve_circulation_uncertainty_edit",
    },
    "fact-396cb8d124a8ac7de43c": {
        "active_index": 330,
        "question": "What should you monitor while opening the diamonds in a chest harness?",
        "answer": "Rope365 says to monitor the tied person's breathing while opening the diamonds because the harness becomes progressively tighter.",
        "decision": "keep",
        "source_document": "data/raw/rope_resources_v1/rope365__b190861046dbc85a6875.json",
        "source_document_file_sha256": "7f58c37deef4c8b0052e603a8ba7efd266ce8735f224beb37385aa322d34eb3f",
        "source_document_chars": 3657,
        "medical_scope": "direct breathing monitoring during a progressively tightening harness",
        "reason_code": "keep_progressive_tightening_breathing_monitor",
        "reason": "The row names the changing mechanism and the observation required while the tie is being tightened, without adding unsupported medical claims.",
        "review_class": "breathing_monitor_keep",
    },
    "fact-15bfb5a6bc44cbb40909": {
        "active_index": 331,
        "question": "What shoulder-tension nerve warning and mitigation does Rope365 give for a third rope on a box tie?",
        "answer": "Very high shoulder tension can cause clavicle-area numbness or weakness when raising the arm; Rope365 recommends keeping shoulder rope loose, or doubling it to spread the load if more tension is used.",
        "decision": "edit",
        "edited_question": "Which risk factors and preventive tension choice does Rope365 identify for a third rope crossing the shoulders?",
        "edited_answer": "It associates very high top-of-shoulder tension, suspension or dive loading, session duration, and cumulative exposure with nerve risk, and recommends keeping the shoulder rope loose.",
        "source_document": "data/raw/rope_resources_v1/rope365__f85b7dd6c9c6a4a08bfd.json",
        "source_document_file_sha256": "f3c5740ed5b099ba371432d82493ac7c648cbdb7e6f264730344f92f3c0e8031",
        "source_document_chars": 6644,
        "medical_scope": "risk-factor prevention; omits high-tension mitigation as a symptom shortcut",
        "reason_code": "retain_loose_shoulder_prevention_and_remove_high_tension_shortcut",
        "reason": "The replacement keeps the source's prevention and cumulative-risk context while removing the suggestion that doubling rope makes deliberately higher shoulder tension sufficiently mitigated.",
        "review_class": "shoulder_nerve_prevention_edit",
    },
    "fact-04c31b72bb2803a1ff1a": {
        "active_index": 367,
        "question": "What warning signs and response does Rope365 give for ulnar-nerve strain in a chicken-wing tie?",
        "answer": "Tingling in the little and ring fingers is a warning sign; open the 'wings' more and reduce elbow compression.",
        "decision": "drop",
        "source_document": "data/raw/rope_resources_v1/rope365__118a41bb0bd96d9bc195.json",
        "source_document_file_sha256": "910c3412388618bc65c1eaf7332b92a1b03541714cf233b52a06aa177395b94c",
        "source_document_chars": 2211,
        "medical_scope": "symptom-treatment shortcut after finger tingling",
        "reason_code": "quarantine_tingling_adjust_and_continue_shortcut",
        "reason": "The row can be read as advice to adjust and continue after neurologic symptoms. Active fact f6befcfa1002eaeafbfc retains the conservative source-backed rule to communicate and come out of the tie when tingling occurs and the cause is uncertain.",
        "review_class": "symptom_treatment_shortcut_drop",
        "redundant_survivors": ["fact-f6befcfa1002eaeafbfc"],
    },
}


SURVIVORS = {
    "fact-fecdb35d2e926add5dff": (
        "What breathing and release warning does Rope365 give for the shrimp tie?",
        "Bending forward makes breathing difficult and you have to be prepared to untie quickly.",
    ),
    "fact-396cb8d124a8ac7de43c": (
        "What should you monitor while opening the diamonds in a chest harness?",
        "Rope365 says to monitor the tied person's breathing while opening the diamonds because the harness becomes progressively tighter.",
    ),
    "fact-f6befcfa1002eaeafbfc": (
        "What conservative action does the rope-bottom guide recommend if any body part starts tingling and the cause is uncertain?",
        "Tell the rigger and come out of the tie; you can be tied again later.",
    ),
}


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v449 baseline drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v450-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v449.jsonl"
    build_baseline(base, inputs / "v449.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def validate_source(spec: dict, row: dict, evidence: str) -> None:
    source_path = ROOT / spec["source_document"]
    if file_sha256(source_path) != spec["source_document_file_sha256"]:
        raise ValueError(f"source file drift: {row['fact_id']}")
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
    with tempfile.TemporaryDirectory(prefix=".v450-observe-", dir=HERE) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v450-base-", dir=HERE) as tmp:
        directory = Path(tmp)
        base = directory / "v449.jsonl"
        build_baseline(base, directory / "v449.report.json")
        before = read_jsonl(base)
    by_fact = {row["fact_id"]: (index, row) for index, row in enumerate(before, 1)}

    for fact_id, expected in SURVIVORS.items():
        row = by_fact[fact_id][1]
        if (row["question"], row["answer"]) != expected:
            raise ValueError(f"operational survivor drift: {fact_id}")

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
            "fact_id": fact_id, "medical_scope": spec["medical_scope"],
            "projection_lineage": {
                "baseline_rows": BASELINE_ROWS, "baseline_sha256": BASELINE_SHA256,
                "prior_context_merit_review": True,
            },
            "reason": spec["reason"], "reason_code": spec["reason_code"],
            "review_class": spec["review_class"], "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER, "schema": "context-merit-audit-v450",
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
    if (observation["rows"], observation["eval"]) != (507, 612):
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
            "by_medical_scope": dict(Counter(row["medical_scope"] for row in audits)),
            "by_review_class": dict(Counter(row["review_class"] for row in audits)),
            "path": portable(AUDIT), "rows": len(audits), "sha256": file_sha256(AUDIT),
        },
        "conservative_capacity": {
            "after": observation["after"], "before": observation["before"],
            "delta": {key: observation["after"][key] - observation["before"][key] for key in observation["before"]},
            "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
        },
        "curation_outcome": {
            "anatomy_breathing_or_injury_repairs": 6,
            "operational_breathing_rows_kept": 2,
            "vague_or_symptom_treatment_rows_quarantined": 2,
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
        "surviving_operational_guidance": {
            "dropped_facts": sorted(
                fact_id for fact_id, spec in SPECS.items() if spec["decision"] == "drop"
            ),
            "surviving_facts": sorted(SURVIVORS),
        },
        "schema": "context-merit-audit-report-v450",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "training_layer_contract": {
            "cleaned_markdown_site_corpora": "Distinct first-class future training artifacts; no corpus output was ingested by v450.",
            "derived_qa": "Distinct first-class training layer; v450 changes only existing sealed Q&A using attached evidence and pinned source snapshots.",
            "deduplication_policy": "Do not collapse a cleaned Markdown document and its derived Q&A merely because they express the same source knowledge.",
            "provenance_requirement": "Link future derived Q&A to cleaned source documents through stable source URL and document or snapshot hashes.",
        },
        "v52_isolation": {"active_v52_inputs_modified": False, "candidate_status": "isolated_pending_not_promoted"},
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
