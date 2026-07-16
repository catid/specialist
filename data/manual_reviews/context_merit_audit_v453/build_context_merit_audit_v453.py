#!/usr/bin/env python3
"""Curate six existing Kita Reiko history and terminology Q&As.

The only candidate base is sealed v452. Two analytical history rows receive
stronger attribution and durable framing, two one-off label-recall rows are
quarantined, and two rows that preserve uncertainty or translation limits are
retained. No corpus or protected evaluation artifact is read or changed.
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
V452 = DATA / "manual_reviews/context_merit_audit_v452"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V452), str(V290)]

import build_context_merit_audit_v452 as previous
import build_context_merit_audit_v290 as core
from qa_quality import stable_fact_id


AUDIT = HERE / "context_merit_audit_v453.jsonl"
CURATION = HERE / "pending_curation_context_merit_v453.jsonl"
REPORT = HERE / "report_context_merit_v453.json"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
SOURCE_DOCUMENT = "data/raw/kinbakutoday_d4dcb268cb41c5e4.json"
SOURCE_FILE_SHA256 = "f6e44831df16225b3d5922ca17dd0785b75389528060c3409c4fa5def8198c16"
SOURCE_TEXT_SHA256 = "262868b407f8efd43d239cd401eaa69bdbc0c8e2beb75af67ca9ffa0f098257a"
SOURCE_CHARS = 23469
SOURCE_URL = "https://www.kinbakutoday.com/writing-as-the-other-minomura-ko-kita-reiko/"
BASELINE_ROWS = 501
BASELINE_SHA256 = "78af3c99df92fcfba6451803fa04ba9b9d4a4c781c4f5a99b0acef0496807195"
EXPECTED_OUTPUT_SHA256 = "1245a8c31f5c984f8af9aa0f5af87927b3014de6c27ac60a997d165833446491"
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
REVIEWER = "codex-context-merit-audit-v453"

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
portable = previous.portable
conservative_capacity = previous.conservative_capacity


SPECS = {
    "fact-84fe512655ff109a657e": {
        "active_index": 141,
        "question": "In “The Woman Who Wasn’t There,” what phrase describes a male-authored feminine voice that presents male fantasy as women’s own desire?",
        "answer": "The phrase is “ventriloquized interiority.”",
        "decision": "drop",
        "history_scope": "article-coined analytical label without standalone historical content",
        "reason_code": "quarantine_one_off_ventriloquized_interiority_label",
        "reason": "The row tests recall of the article author's one-off analytical phrase rather than the underlying publishing history. The edited Kita Reiko pseudonym row retains the substantive claim in plain language.",
        "review_class": "coined_label_recall_drop",
        "substantive_survivors": ["fact-620c1fbb2313214967c2"],
    },
    "fact-56db389c190dd20466bd": {
        "active_index": 142,
        "question": "In “The Woman Who Wasn’t There,” what publishing problem does Reiko’s conflict with her male patrons dramatize?",
        "answer": "It dramatizes the problem of distinguishing seme from mere violence, vulgar appetite, or crude spectacle.",
        "decision": "edit",
        "edited_question": "How does Kinbaku Today say early postwar SM magazines reframed fantasies that could look violent, pathological, or private?",
        "edited_answer": "The article argues that the magazines presented those fantasies as aesthetic, relational, and even feminine, while distinguishing seme from mere violence or crude spectacle.",
        "history_scope": "article-attributed interpretation of early postwar SM publishing",
        "reason_code": "replace_fiction_plot_prompt_with_historical_media_analysis",
        "reason": "The replacement removes the story-plot dependency and asks for the source's broader historical interpretation, while marking that interpretation as the article's argument rather than settled fact.",
        "review_class": "historical_interpretation_edit",
    },
    "fact-ed15766580bb03fdee1b": {
        "active_index": 143,
        "question": "In “The Woman Who Wasn’t There,” which Japanese term does the narrator use for “ecstatic cruelty”?",
        "answer": "The narrator uses the term 悦虐.",
        "decision": "drop",
        "history_scope": "isolated character recall from one fictional narrator's wording",
        "reason_code": "quarantine_story_specific_ecstatic_cruelty_characters",
        "reason": "The answer is an isolated pair of characters from one quoted story, with no reading, usage guidance, or durable rope-domain definition. The surviving seme row retains the source's genuinely useful warning about translation range.",
        "review_class": "story_term_recall_drop",
        "substantive_survivors": ["fact-7f6d91c3cb62736182e0"],
    },
    "fact-ca5241d923ca75bba2c5": {
        "active_index": 187,
        "question": "What did the feminine pen name Kita Reiko make possible in early SM publishing?",
        "answer": "The name allowed seme-e and writing about female masochism to appear as if they emerged from a woman’s own aesthetic and erotic interiority.",
        "decision": "edit",
        "edited_question": "What publishing function does Kinbaku Today attribute to Suma Toshiyuki’s feminine pseudonym Kita Reiko?",
        "edited_answer": "The article argues that the pseudonym let his seme-e and writing about female masochism appear to come from a woman’s own aesthetic and erotic interiority, rather than merely hiding the author’s identity.",
        "history_scope": "source-attributed interpretation of a gendered publication persona",
        "reason_code": "attribute_kita_reiko_pseudonym_function_to_article",
        "reason": "The replacement identifies Suma Toshiyuki and explicitly attributes the claimed publishing function to the article, avoiding an unsupported assertion about private intent while preserving the useful media-history analysis.",
        "review_class": "pseudonym_function_attribution_edit",
    },
    "fact-99d19f48be3fa51f5bb7": {
        "active_index": 205,
        "question": "What does Kinbaku Today’s “The Woman Who Wasn’t There” say the uncertainty around S-ko’s reader letter reveals?",
        "answer": "Whether the letter was genuine, edited, or fabricated, it shows that the magazine wanted to stage a woman reader’s recognition as evidence.",
        "decision": "keep",
        "history_scope": "explicitly uncertain archival interpretation",
        "reason_code": "keep_sko_letter_authenticity_uncertainty",
        "reason": "The question is clearly attributed and the answer foregrounds all three possible provenance states instead of treating the reader letter as authenticated testimony.",
        "review_class": "archival_uncertainty_keep",
    },
    "fact-7f6d91c3cb62736182e0": {
        "active_index": 419,
        "question": "Which term does “The Woman Who Wasn’t There” leave untranslated because no single English word covers its full range?",
        "answer": "The term is seme, whose range includes punishment, torment, erotic pressure, accusation, and aestheticized cruelty.",
        "decision": "keep",
        "history_scope": "article-scoped terminology with an explicit translation limitation",
        "reason_code": "keep_seme_translation_range_and_limitation",
        "reason": "The source scope is explicit, the answer gives a useful semantic range, and the premise rejects a false one-to-one universal translation rather than inventing one.",
        "review_class": "translation_limitation_keep",
    },
}


def requested_urls() -> set[str]:
    resources = json.loads(RESOURCE_MANIFEST.read_text())["resources"]
    return {
        url for resource in resources
        for url in (resource["canonical_url"], resource.get("recommendation_url")) if url
    }


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v452 baseline drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v453-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v452.jsonl"
    build_baseline(base, inputs / "v452.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def validate_source(row: dict, evidence: str) -> None:
    source_path = ROOT / SOURCE_DOCUMENT
    if file_sha256(source_path) != SOURCE_FILE_SHA256:
        raise ValueError(f"source file drift: {row['fact_id']}")
    document = json.loads(source_path.read_text())
    if document["url"] != SOURCE_URL or row["url"] != SOURCE_URL:
        raise ValueError(f"source URL drift: {row['fact_id']}")
    if len(document["text"]) != SOURCE_CHARS or text_sha256(document["text"]) != SOURCE_TEXT_SHA256:
        raise ValueError(f"source text drift: {row['fact_id']}")
    if row["document_sha256"] != SOURCE_TEXT_SHA256 or evidence not in document["text"]:
        raise ValueError(f"evidence absent from full source snapshot: {row['fact_id']}")


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v453-observe-", dir=HERE) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v453-base-", dir=HERE) as tmp:
        directory = Path(tmp)
        base = directory / "v452.jsonl"
        build_baseline(base, directory / "v452.report.json")
        before = read_jsonl(base)
    by_fact = {row["fact_id"]: (index, row) for index, row in enumerate(before, 1)}

    audits, curations = [], []
    for audit_index, (fact_id, spec) in enumerate(SPECS.items(), 1):
        active_index, row = by_fact[fact_id]
        if active_index != spec["active_index"]:
            raise ValueError(f"active index drift: {fact_id}")
        if (row["question"], row["answer"]) != (spec["question"], spec["answer"]):
            raise ValueError(f"active Q&A drift: {fact_id}")
        evidence = row["evidence"]
        validate_source(row, evidence)

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
                "source_lineage": {"source_document": SOURCE_DOCUMENT},
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
            "fact_id": fact_id, "history_scope": spec["history_scope"],
            "projection_lineage": {
                "baseline_rows": BASELINE_ROWS, "baseline_sha256": BASELINE_SHA256,
                "prior_context_merit_review": True,
            },
            "reason": spec["reason"], "reason_code": spec["reason_code"],
            "review_class": spec["review_class"], "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER, "schema": "context-merit-audit-v453",
            "source": row["source"], "source_document": SOURCE_DOCUMENT,
            "source_document_chars": SOURCE_CHARS,
            "source_document_file_sha256": SOURCE_FILE_SHA256,
            "source_document_text_sha256": SOURCE_TEXT_SHA256,
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
        if spec.get("substantive_survivors"):
            audit["substantive_survivors"] = spec["substantive_survivors"]
        audits.append(audit)

    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    observation = observe(before)
    if not observation["dataset_equal"] or not observation["report_equal"]:
        raise ValueError(f"nondeterministic projection: {observation}")
    if (observation["rows"], observation["eval"]) != (499, 612):
        raise ValueError(f"projection count drift: {observation}")
    if observation["before"] != EXPECTED_CAPACITY_BEFORE:
        raise ValueError(f"baseline capacity drift: {observation}")
    if EXPECTED_CAPACITY_AFTER is not None and observation["after"] != EXPECTED_CAPACITY_AFTER:
        raise ValueError(f"candidate capacity drift: {observation}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observation["sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError(f"projection hash drift: {observation}")

    with tempfile.TemporaryDirectory(prefix=".v453-resource-check-", dir=HERE) as tmp:
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
    REPORT.write_text(json.dumps({
        "audit": {
            "by_decision": dict(Counter(row["decision"] for row in audits)),
            "by_history_scope": dict(Counter(row["history_scope"] for row in audits)),
            "by_review_class": dict(Counter(row["review_class"] for row in audits)),
            "path": portable(AUDIT), "rows": len(audits), "sha256": file_sha256(AUDIT),
        },
        "conservative_capacity": {
            "after": observation["after"], "before": observation["before"],
            "delta": {key: observation["after"][key] - observation["before"][key] for key in observation["before"]},
            "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
        },
        "curation_outcome": {
            "historical_attribution_or_framing_repairs": 2,
            "one_off_label_or_story_term_rows_quarantined": 2,
            "uncertainty_or_translation_limit_rows_kept": 2,
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
        "requested_resource_coverage": {
            "manifest_urls": len(urls), "manifest_urls_present": sum(url in blob for url in urls),
            "owner_resource_facts": len(OWNER_FACT_IDS),
        },
        "source_boundary": {
            "new_or_pending_corpus_inputs": 0,
            "protected_eval_heldout_holdout_ood_shadow_probe_rows_opened": 0,
            "training_or_eval_artifacts_modified": 0,
        },
        "source_snapshot_inventory": {
            "documents": 1, "paths": [SOURCE_DOCUMENT], "reviewed_rows": len(SPECS),
            "total_unique_characters": SOURCE_CHARS,
        },
        "schema": "context-merit-audit-report-v453",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "training_layer_contract": {
            "cleaned_markdown_site_corpora": "Distinct first-class future training artifacts; no corpus output was ingested by v453.",
            "derived_qa": "Distinct first-class training layer; v453 changes only existing sealed Q&A using attached evidence and one pinned source snapshot.",
            "deduplication_policy": "Do not collapse a cleaned Markdown document and its derived Q&A merely because they express the same source knowledge.",
            "provenance_requirement": "Link future derived Q&A to cleaned source documents through stable source URL and document or snapshot hashes.",
        },
        "v52_isolation": {"active_v52_inputs_modified": False, "candidate_status": "isolated_pending_not_promoted"},
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
