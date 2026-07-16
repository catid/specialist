#!/usr/bin/env python3
"""Replace every active RopeTopia URL-trivia row with evidence or a drop.

The baseline is the exact v442 train-only projection.  Eight title/URL rows
are replaced by useful archive/successor-supported facts; seven pages without
usable text bodies are removed.  Existing useful successor facts are audited
and retained.  No V52 input is modified and no protected semantic content is
opened manually.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
V290 = DATA / "manual_reviews/context_merit_audit_v290"
CORPUS_DIR = DATA / "site_corpora/rope_topia"
sys.path[:0] = [str(ROOT), str(V290), str(CORPUS_DIR)]

import build_context_merit_audit_v290 as core
import build_rope_topia_corpus as corpus
from qa_quality import stable_fact_id


AUDIT = HERE / "context_merit_audit_v443.jsonl"
CURATION = HERE / "pending_curation_context_merit_v443.jsonl"
ADDITIONS = HERE / "pending_additions_context_merit_v443.jsonl"
REPORT = HERE / "report_context_merit_v443.json"
V440_FROZEN = ROOT / "experiments/sft_controls/v53a_train_refresh_v440_fold3/train_projection_v440.jsonl"
V441_CURATION = DATA / "manual_reviews/context_merit_audit_v441/pending_curation_context_merit_v441.jsonl"
V442_CURATION = DATA / "manual_reviews/context_merit_audit_v442/pending_curation_context_merit_v442.jsonl"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
ROPETOPIA_MANIFEST = ROOT / "sources/rope_topia_manual_v1.json"

BASELINE_ROWS = 531
BASELINE_SHA256 = "048688bd57829c70101fd0003116f4cef0ed37df3cd4b6d443f4e3b0ebd74555"
V440_SHA256 = "8988b14443dd3ab51be98be24b1aec6bc82ec4233241f3422cb0887c842af078"
V441_SHA256 = "c5a53b330bf4c338aeab2d78e074a24b2d353d81465ec0265e443300fadbb410"
EXPECTED_OUTPUT_SHA256 = "9f8e28292f65b4c8f0928f3fbde353a431686753bb3af42435b2a1d43c2b4d0d"
EXPECTED_CAPACITY_BEFORE = {
    "conflict_units": 259,
    "equipment_material": 23,
    "resources_general": 84,
    "safety_consent": 81,
    "technique": 71,
}
EXPECTED_CAPACITY_AFTER = {
    "conflict_units": 266,
    "equipment_material": 23,
    "resources_general": 85,
    "safety_consent": 86,
    "technique": 72,
}
REVIEWED_AT = "2026-07-16"
REVIEWER = "codex-context-merit-audit-v443"

file_sha256 = core.file_sha256
text_sha256 = core.text_sha256
read_jsonl = core.read_jsonl
write_jsonl = core.write_jsonl
portable = core.portable
conservative_capacity = core.conservative_capacity


DIRECT = {
    "out_into_kink_community": (381, "fact-00c3b2aae602e4f946a9", "Where can I find Rope-topia’s “Getting out in the kink community” page?"),
    "safety_cutters": (382, "fact-2a378b89a3783e6177cf", "Where can I find Rope-topia’s guide to safety cutters?"),
    "identifying_predatory_behaviour": (383, "fact-341638ff48c8aec7ecd2", "Where can I find Rope-topia’s “Identifying predatory behaviour” page?"),
    "joining_rope": (384, "fact-5ef1eecad47b64637110", "Where can I find Rope-topia’s Joining Rope tutorial?"),
    "nerve_and_circulation_problems": (385, "fact-260d03d32d55f9371d8c", "Where can I find Rope-topia’s “Nerve and Circulation Problems in Shibari” page?"),
    "newcomers_information": (386, "fact-ab0940e437fedca7017b", "Where can I find Rope-topia’s newcomers information hub?"),
    "kinbaku_today_rope_not_about_rope": (387, "fact-fe524e9867f9b0728d24", "Where can I find Rope-topia’s portfolio article “Rope is not about Rope”?"),
    "luck_self_awareness_responsibility_injuries": (388, "fact-43a434b124480c4789fb", "Where can I find Rope-topia’s post about luck, self-awareness, responsibility, and rope-bondage injuries?"),
    "ichinawa_ippon_me_no_nawa_one_rope": (389, "fact-064f48e286485f35f886", "Where can I find Rope-topia’s post “Ichinawa, Ippon me no nawa and One rope”?"),
    "rope_bottom_guide": (390, "fact-23156860a668837a3966", "Where can I find Rope-topia’s Rope Bottom Guide?"),
    "new_to_kink_scene": (391, "fact-49ad2dfa9593d06babc3", "Where can I find Rope-topia’s “So you’re new to the kink scene” page?"),
    "strugglers_knot": (392, "fact-944dd870c5a406e3df18", "Where can I find Rope-topia’s Strugglers Knot tutorial?"),
    "wet_treating_rope": (393, "fact-eac2eb6d862ee212602f", "Where can I find Rope-topia’s Wet treating rope tutorial?"),
    "wicked_fast_bowline": (394, "fact-a277b650168c4f6aeb6b", "Where can I find Rope-topia’s Wicked Fast Bowline tutorial?"),
    "yin_yoga_for_bondage": (395, "fact-7f1bd0c0cee41d8cfd54", "Where can I find Rope-topia’s “Yin Yoga for Bondage” post?"),
}


REPLACEMENTS = {
    "out_into_kink_community": {
        "question": "Which sweeping claims does RopeTopia advise newcomers to scrutinize?",
        "answer": "It advises scrutinizing claims that all or “true” dominants or submissives must behave a certain way, along with claims that someone is or is not a “natural.”",
        "topic": "newcomer_critical_thinking",
    },
    "safety_cutters": {
        "question": "What quality checks does RopeTopia recommend for emergency shears?",
        "answer": "It recommends shears that do not bend easily, whose handles do not slide past each other, whose blades do not flex apart, and whose rivet is tight enough to prevent blade play.",
        "topic": "emergency_cutter_quality",
    },
    "identifying_predatory_behaviour": {
        "question": "Which isolation tactics does RopeTopia list as warning signs of predatory behavior?",
        "answer": "Trying to isolate someone from information or friends, or preventing them from speaking with experienced community members.",
        "topic": "predatory_isolation_warning_signs",
    },
    "nerve_and_circulation_problems": {
        "question": "Which three factors does RopeTopia identify in nerve problems, and how quickly should people respond?",
        "answer": "It identifies position, pressure, and duration, and says the person in rope should speak up immediately and the rigger should act immediately.",
        "topic": "nerve_warning_response",
    },
    "luck_self_awareness_responsibility_injuries": {
        "question": "How does RopeTopia distinguish accepted rope risk from repeated injuries?",
        "answer": "It acknowledges that rope—especially suspension—carries risk, but says injuries should not be treated as routine and repeated injuries warrant serious self-questioning.",
        "topic": "recurring_injury_accountability",
    },
    "ichinawa_ippon_me_no_nawa_one_rope": {
        "question": "When would WykD use Ipponnawa rather than Ichinawa?",
        "answer": "WykD says Ipponnawa applies when counting ropes; more precisely, it may be phrased as Ippon me no nawa.",
        "topic": "japanese_rope_counting_terminology",
        "successor": True,
    },
    "new_to_kink_scene": {
        "question": "What due diligence does RopeTopia recommend before playing with someone new?",
        "answer": "It recommends obtaining several references, taking their comments seriously, talking with people who know or have played with the person, and observing how that person plays and behaves.",
        "topic": "new_partner_due_diligence",
    },
    "yin_yoga_for_bondage": {
        "question": "Why does RopeTopia’s guest article say Yin Yoga may help a rope bottom beyond flexibility?",
        "answer": "The guest writer says its contemplative practice can cultivate mindfulness and body awareness while exposing practitioners to physical discomfort, thoughts, and emotional responses that may also arise during bondage.",
        "topic": "contemplative_body_awareness",
    },
}


DROP_REASON_CODES = {
    "joining_rope": "video_only_no_textual_joining_method",
    "newcomers_information": "navigation_hub_without_distinct_substantive_fact",
    "kinbaku_today_rope_not_about_rope": "missing_archive_body_and_duplicate_article_source",
    "rope_bottom_guide": "download_metadata_without_guide_body",
    "strugglers_knot": "video_only_no_textual_knot_method",
    "wet_treating_rope": "video_only_no_textual_rope_treatment_method",
    "wicked_fast_bowline": "video_only_no_textual_bowline_method",
}


TRIVIA_PATTERNS = (
    re.compile(r"\bwhich\s+(?:canonical|current)\b.*\burl\b", re.I),
    re.compile(r"\bwhat\s+url\b", re.I),
    re.compile(r"\bwhere\s+is\b.*\blisted\b", re.I),
    re.compile(r"\bwhere\s+can\s+i\s+find\b", re.I),
)


def build_baseline(out: Path, report: Path) -> None:
    """Replay only v441-v442 from the already sealed exact v440 projection."""
    if (len(read_jsonl(V440_FROZEN)), file_sha256(V440_FROZEN)) != (531, V440_SHA256):
        raise ValueError("sealed v440 drift")
    replay = out.parent / f".{out.name}.v442-replay"
    replay.mkdir(parents=True, exist_ok=True)
    v441 = replay / "v441.jsonl"
    core.build_projection_with_inputs(v441, replay / "v441.report.json", (V441_CURATION,), (V440_FROZEN,))
    if (len(read_jsonl(v441)), file_sha256(v441)) != (531, V441_SHA256):
        raise ValueError("v441 replay drift")
    core.build_projection_with_inputs(out, report, (V442_CURATION,), (v441,))
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v442 replay drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v443-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v442.jsonl"
    build_baseline(base, inputs / "v442.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base, ADDITIONS))


def archive_evidence() -> dict[str, dict]:
    snapshots = json.loads(corpus.EVIDENCE.read_text())
    return {row["resource_id"]: row for row in snapshots["archive_pages"]}


def make_additions(resources: dict[str, dict]) -> list[dict]:
    archive = archive_evidence()
    rows = []
    for resource_id, spec in REPLACEMENTS.items():
        original = resources[resource_id]
        snapshot = archive[resource_id]
        evidence = snapshot["exact_qa_evidence"]
        question, answer = spec["question"], spec["answer"]
        if spec.get("successor"):
            successor_spec = corpus.SUCCESSORS[resource_id]
            source_path = ROOT / successor_spec["path"]
            source = json.loads(source_path.read_text())
            if evidence not in source["text"]:
                raise ValueError("Ichinawa successor evidence drift")
            source_name = "wykd"
            url = successor_spec["url"]
            document_sha256 = successor_spec["text_sha256"]
            lineage = {
                "raw_successor_document": successor_spec["path"],
                "relationship": "WykD successor article corresponding to the recovered RopeTopia body",
                "ropetopia_archive_capture_timestamp": snapshot["archive"]["capture_timestamp"],
                "ropetopia_manifest_resource_id": resource_id,
            }
            evidence_url = url
        else:
            source_name = "rope_topia_archive"
            url = original["canonical_url"]
            document_sha256 = snapshot["archive"]["extracted_body_sha256"]
            lineage = {
                "archive_evidence_snapshot": "data/site_corpora/rope_topia/evidence_snapshots.json",
                "archive_capture_timestamp": snapshot["archive"]["capture_timestamp"],
                "archive_cdx_digest": snapshot["archive"]["cdx_digest"],
                "archive_html_sha256": snapshot["archive"]["html_sha256"],
                "relationship": "exact evidence excerpt selected after full archived RopeTopia body review",
                "ropetopia_manifest_resource_id": resource_id,
            }
            evidence_url = snapshot["archive"]["replay_url"]
        rows.append({
            "answer": answer,
            "claim_type": "instructional",
            "document_sha256": document_sha256,
            "evidence": evidence,
            "evidence_sha256": text_sha256(evidence),
            "evidence_url": evidence_url,
            "fact_id": stable_fact_id(question, answer),
            "kind": "qa_resource_manual_fact",
            "original_ropetopia_url": original["canonical_url"],
            "quality_schema": "manual-resource-fact-v1",
            "question": question,
            "resource_id": "rope_topia_recovered_content",
            "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER,
            "source": source_name,
            "source_lineage": lineage,
            "text": f"Question: {question}\nAnswer: {answer}",
            "topic": spec["topic"],
            "url": url,
        })
    return rows


def direct_rows(before: list[dict]) -> dict[str, dict]:
    manifest = json.loads(ROPETOPIA_MANIFEST.read_text())
    id_by_url = {row["canonical_url"]: row["id"] for row in manifest["resources"]}
    selected = {}
    for index, row in enumerate(before, 1):
        if row.get("source") != "rope_topia":
            continue
        resource_id = id_by_url.get(row.get("url"))
        if resource_id is None or resource_id not in DIRECT:
            raise ValueError(f"unexpected direct RopeTopia row: {row.get('fact_id')}")
        expected_index, fact_id, question = DIRECT[resource_id]
        if (index, row["fact_id"], row["question"], row["answer"]) != (
            expected_index, fact_id, question, row["url"]
        ):
            raise ValueError(f"direct RopeTopia inventory drift: {resource_id}")
        selected[resource_id] = row
    if set(selected) != set(DIRECT):
        raise ValueError("direct RopeTopia inventory incomplete")
    return selected


def successor_rows(before: list[dict]) -> list[tuple[int, dict]]:
    rows = []
    for index, row in enumerate(before, 1):
        lineage = row.get("source_lineage") or {}
        if row.get("source") == "wykd" and lineage.get("ropetopia_manifest_resource_id"):
            source_path = ROOT / lineage["raw_successor_document"]
            source = json.loads(source_path.read_text())
            if row.get("evidence") not in source["text"]:
                raise ValueError(f"successor evidence drift: {row['fact_id']}")
            rows.append((index, row))
    if len(rows) != 7:
        raise ValueError(f"successor inventory drift: {len(rows)}")
    return rows


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v443-observe-", dir=HERE) as tmp:
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
    corpus.build()
    resources = {row["id"]: row for row in json.loads(ROPETOPIA_MANIFEST.read_text())["resources"]}
    additions = make_additions(resources)
    write_jsonl(ADDITIONS, additions)

    with tempfile.TemporaryDirectory(prefix=".v443-base-", dir=HERE) as tmp:
        directory = Path(tmp)
        base = directory / "v442.jsonl"
        build_baseline(base, directory / "v442.report.json")
        before = read_jsonl(base)

    direct = direct_rows(before)
    successors = successor_rows(before)
    addition_by_resource = {row["source_lineage"]["ropetopia_manifest_resource_id"]: row for row in additions}
    corpus_inventory = {row["resource_id"]: row for row in json.loads(corpus.MANIFEST.read_text())["inventory"]}
    curations, audits = [], []
    audit_index = 0
    for resource_id, (active_index, fact_id, _) in sorted(DIRECT.items(), key=lambda item: item[1][0]):
        active = direct[resource_id]
        disposition = "replace" if resource_id in REPLACEMENTS else "drop"
        reason_code = (
            "replace_url_title_trivia_with_archive_or_successor_fact"
            if disposition == "replace" else DROP_REASON_CODES[resource_id]
        )
        reason = (
            "The URL-recall row teaches page-location metadata; a distinct useful fact is supported by exact recovered evidence."
            if disposition == "replace" else corpus_inventory[resource_id]["reason"]
        )
        curations.append({
            "action": "drop",
            "expected_answer": active["answer"],
            "expected_question": active["question"],
            "fact_id": fact_id,
            "reason": reason,
            "reason_code": reason_code,
            "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER,
        })
        audit_index += 1
        audit = {
            "active_answer": active["answer"],
            "active_index": active_index,
            "active_question": active["question"],
            "audit_index": audit_index,
            "decision": disposition,
            "document_sha256": active["document_sha256"],
            "fact_id": fact_id,
            "projection_lineage": {"baseline_rows": BASELINE_ROWS, "baseline_sha256": BASELINE_SHA256},
            "reason": reason,
            "reason_code": reason_code,
            "resource_id": resource_id,
            "review_pass": "complete_ropetopia_url_title_navigation_trivia_repair",
            "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER,
            "schema": "context-merit-audit-v443",
            "source": active["source"],
            "url": active["url"],
        }
        if disposition == "replace":
            replacement = addition_by_resource[resource_id]
            audit.update({
                "replacement_answer": replacement["answer"],
                "replacement_evidence": replacement["evidence"],
                "replacement_evidence_sha256": replacement["evidence_sha256"],
                "replacement_evidence_url": replacement["evidence_url"],
                "replacement_fact_id": replacement["fact_id"],
                "replacement_question": replacement["question"],
                "replacement_source_lineage": replacement["source_lineage"],
            })
        audits.append(audit)

    for active_index, active in successors:
        audit_index += 1
        lineage = active["source_lineage"]
        source_path = ROOT / lineage["raw_successor_document"]
        audits.append({
            "active_answer": active["answer"],
            "active_index": active_index,
            "active_question": active["question"],
            "audit_index": audit_index,
            "decision": "keep",
            "document_sha256": active["document_sha256"],
            "fact_id": active["fact_id"],
            "projection_lineage": {"baseline_rows": BASELINE_ROWS, "baseline_sha256": BASELINE_SHA256},
            "reason": "The row asks a useful content question, its answer is supported by exact contiguous successor evidence, and it contains no URL/title/navigation trivia.",
            "reason_code": "keep_useful_exact_successor_content_fact",
            "resource_id": lineage["ropetopia_manifest_resource_id"],
            "review_pass": "complete_ropetopia_url_title_navigation_trivia_repair",
            "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER,
            "schema": "context-merit-audit-v443",
            "source": active["source"],
            "source_document": lineage["raw_successor_document"],
            "source_document_file_sha256": file_sha256(source_path),
            "source_snapshot_chars_manually_reviewed": len(json.loads(source_path.read_text())["text"]),
            "support_evidence": active["evidence"],
            "support_evidence_sha256": text_sha256(active["evidence"]),
            "url": active["url"],
        })

    write_jsonl(CURATION, curations)
    write_jsonl(AUDIT, audits)
    observation = observe(before)
    if not observation["dataset_equal"] or not observation["report_equal"]:
        raise ValueError(f"nondeterministic projection: {observation}")
    if (observation["rows"], observation["eval"]) != (524, 612):
        raise ValueError(f"projection count drift: {observation}")
    if observation["before"] != EXPECTED_CAPACITY_BEFORE:
        raise ValueError(f"baseline capacity drift: {observation}")
    if EXPECTED_CAPACITY_AFTER is not None and observation["after"] != EXPECTED_CAPACITY_AFTER:
        raise ValueError(f"candidate capacity drift: {observation}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observation["sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError(f"projection hash drift: {observation}")

    corpus_artifacts = {
        name: {"path": portable(path), "sha256": file_sha256(path)}
        for name, path in {
            "evidence": corpus.EVIDENCE,
            "manifest": corpus.MANIFEST,
            "markdown": corpus.MARKDOWN,
            "report": corpus.REPORT,
        }.items()
    }
    REPORT.write_text(json.dumps({
        "audit": {
            "by_decision": dict(Counter(row["decision"] for row in audits)),
            "direct_url_trivia_rows_inventoried": 15,
            "path": portable(AUDIT),
            "rows": len(audits),
            "sha256": file_sha256(AUDIT),
            "successor_content_rows_inventoried": 7,
        },
        "conservative_capacity": {
            "after": observation["after"],
            "before": observation["before"],
            "delta": {key: observation["after"][key] - observation["before"][key] for key in observation["before"]},
            "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
        },
        "dense_source_corpus": corpus_artifacts,
        "isolated_build_projection": {
            "automated_projection_runs": 2,
            "new_additions_applied": 8,
            "output_rows": observation["rows"],
            "output_sha256": observation["sha256"],
            "repeat_dataset_byte_identical": observation["dataset_equal"],
            "repeat_projection_report_byte_identical": observation["report_equal"],
            "sealed_eval_fact_count_reported_by_tooling": observation["eval"],
        },
        "new_pending_additions": {"path": portable(ADDITIONS), "rows": 8, "sha256": file_sha256(ADDITIONS)},
        "new_pending_curation": {"by_action": {"drop": 15}, "decisions": 15, "path": portable(CURATION), "sha256": file_sha256(CURATION)},
        "replacement_accounting": {
            "archive_supported_replacements": 7,
            "drop_only_without_usable_body": 7,
            "successor_supported_replacements": 1,
            "useful_successor_rows_preserved": 7,
        },
        "schema": "context-merit-audit-report-v443",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "v52_isolation": {
            "active_v52_inputs_modified": False,
            "candidate_status": "isolated_pending_not_promoted",
        },
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
