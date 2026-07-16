#!/usr/bin/env python3
"""Curate ten existing history, people, lineage, and terminology Q&As.

The only candidate base is sealed v445.  Six rows are repaired from their
pinned source snapshots, three low-merit rows are quarantined, and one useful
terminology row is retained unchanged.  No unfinished dense corpus is read or
ingested and no new site-derived fact is added.
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
V445 = DATA / "manual_reviews/context_merit_audit_v445"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V445), str(V290)]

import build_context_merit_audit_v445 as previous
import build_context_merit_audit_v290 as core
from qa_quality import stable_fact_id


AUDIT = HERE / "context_merit_audit_v446.jsonl"
CURATION = HERE / "pending_curation_context_merit_v446.jsonl"
REPORT = HERE / "report_context_merit_v446.json"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
BASELINE_ROWS = 518
BASELINE_SHA256 = "0f3f5f17cae500edfad26a359d436d3e7609815f76b9d19bf37f8a24fdbc8b5c"
EXPECTED_OUTPUT_SHA256 = "7ab0562e978c6a15f7122fd3a62e8cade2daff80a0c0a0228dc069cf1b59d900"
EXPECTED_CAPACITY_BEFORE = {
    "conflict_units": 261,
    "equipment_material": 22,
    "resources_general": 80,
    "safety_consent": 86,
    "technique": 73,
}
EXPECTED_CAPACITY_AFTER = {
    "conflict_units": 260,
    "equipment_material": 22,
    "resources_general": 78,
    "safety_consent": 86,
    "technique": 74,
}
OWNER_SOURCE = previous.OWNER_SOURCE
OWNER_FACT_IDS = previous.OWNER_FACT_IDS
REVIEWED_AT = "2026-07-16"
REVIEWER = "codex-context-merit-audit-v446"

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
portable = previous.portable
conservative_capacity = previous.conservative_capacity


SPECS = {
    "fact-e6da7ef1ee7440696ac1": {
        "active_index": 19,
        "question": "According to Sin, which two words make up the name KOUMANAWA, and what does he say they mean?",
        "answer": "Kouma in Japanese rope industry terms means ‘Jute’, and Nawa, ‘Rope’.",
        "decision": "edit",
        "edited_question": "What Japanese rope-industry meaning does Sin give for the name KOUMANAWA?",
        "edited_answer": "Sin says Kouma means “jute” in Japanese rope-industry terminology and nawa means “rope,” so the name combines the two as “jute rope.”",
        "source_document": "data/raw/kinbakutoday_eb5b3fd68f12e8e9.json",
        "source_document_file_sha256": "6aa05d60d17e94bd986584787a51bec1b9ace144e8b553dd42a82f088cd9d261",
        "source_document_chars": 25896,
        "reason_code": "focus_koumanawa_etymology_on_attributed_rope_terminology",
        "reason": "The replacement makes the durable rope-industry terminology the focus while retaining Sin's attribution and avoiding a universal linguistic claim.",
        "review_class": "terminology_edit",
    },
    "fact-1b10087655b51d39fb72": {
        "active_index": 109,
        "question": "How does the article “Yukimura Ryū” define the suffix ryū?",
        "answer": "It means a fashion, way, style, manner, or an individual’s school of thought.",
        "decision": "keep",
        "source_document": "data/raw/kinbakutoday_381214111022a952.json",
        "source_document_file_sha256": "7d2758400af9c40ad38de7de1cbf1e234bb2ab3721e30e9382b78a106657b997",
        "source_document_chars": 12826,
        "reason_code": "keep_source_attributed_ryu_suffix_definition",
        "reason": "The question clearly attributes a complete terminology definition to the article, the pronoun has an unambiguous antecedent, and the listed meanings are fully supported.",
        "review_class": "terminology_keep",
    },
    "fact-558694f6a7e4f740243d": {
        "active_index": 114,
        "question": "How does Ugo date Ito Seiu's early Kinbaku work and Studies on Seme?",
        "answer": "Ito started his Kinbaku in 1916, if not earlier, and published his Studies on Seme in 1928.",
        "decision": "drop",
        "source_document": "data/raw/kinbakutoday_82071039cb003b58.json",
        "source_document_file_sha256": "bd24cb40980b88ec0cf053e8df7c5ac21c071ba7acf863ab1afbc1070b8ac961",
        "source_document_chars": 22191,
        "reason_code": "quarantine_mere_dates_redundant_with_active_historical_context",
        "reason": "The row asks only for two dates. Active fact-8bc60f112e90fb781203 already preserves the same passage's substantive theater-root and later-Hojōjutsu relationship, so chronology-only recall adds little durable value.",
        "review_class": "mere_chronology_drop",
        "redundant_survivor": "fact-8bc60f112e90fb781203",
    },
    "fact-e9bfdce0244705acf780": {
        "active_index": 128,
        "question": "In “Edo Is Not Dead,” which 1948 volume by Itoh Seiyu is discussed in relation to Edo torture-rope imagery?",
        "answer": "The volume is Nihon keibatsu fūzoku zukan (日本刑罰風俗図史).",
        "decision": "edit",
        "edited_question": "What prior-source context does “Edo Is Not Dead” give for Itoh Seiyu’s 1948 punishment volume?",
        "edited_answer": "The post says Seiyu was already well aware of many earlier books on the subject when he published Nihon keibatsu fūzoku zukan in 1948.",
        "source_document": "data/raw/kinbakutoday_938724eb415ca5c0.json",
        "source_document_file_sha256": "64a95a1fdefb574053e0c62646acf148becd776e3aeb5015e013dafce44b7967",
        "source_document_chars": 3558,
        "reason_code": "replace_volume_title_recall_with_prior_source_context",
        "reason": "The replacement preserves the named volume but asks about the source's historically useful point that Seiyu knew a preceding literature rather than treating the title itself as the lesson.",
        "review_class": "historical_context_edit",
    },
    "fact-b56873ed7a678d043d4d": {
        "active_index": 312,
        "question": "What role did CineMagic, founded in 1983, play in Japanese SM and rope-bondage media?",
        "answer": "It played a vital role in the growing SM and rope-bondage market in Japan.",
        "decision": "edit",
        "edited_question": "What contributions does Kinbaku Today’s memorial attribute to CineMagic founder Yoshimura Shoichi?",
        "edited_answer": "The memorial credits Yoshimura with growing CineMagic into a major 1980s rope and SM studio, working with Minomura Kou and Nureki Chimuo, producing Jo-En: The World of Minomura Kou and Season of Infidelity, and helping bakushi including Naka Akira get their start.",
        "evidence": "It is with sadness that we report the passing of Cinemagic Founder and President Yoshimura Shoichi. Yoshimura Shoichi was a fundamental figure in the history of Japanese SM films, working closely with legendary bakushi Minomura Kou and Nureki Chimuo and producing the documentary Jo-En: The World of Minomura Kou (directed by Yukimura Haruki) and the Oniroku Dan classic story Season of Infidelity.\nCinemagic grew to one of the largest and most important rope and SM studios in the 1980s. Founded in 1983, the production company played an vital role in the growing SM and rope bondage market in Japan. With the release of their first film in May of 1983, Etsubaku! Aiyakko A, Cinemagic began to establish itself as one of the top adult video companies in Japan.\nYoshimura san was a great benefactor for many of Japan’s bakushi and gave many of the great rope masters their start (including Akira Naka, as he described in an interview with Osada Steve).",
        "source_document": "data/raw/kinbakutoday_3376e34b04fe0fb1.json",
        "source_document_file_sha256": "9e001557f45a4f700738a879f86da9469ab9660322c7549b56a6b193626caf99",
        "source_document_chars": 1021,
        "reason_code": "replace_vague_cinemagic_importance_with_documented_contributions",
        "reason": "The replacement turns a vague importance claim into an attributed record of Yoshimura's studio-building, collaborations, productions, and support for other bakushi.",
        "review_class": "documented_contribution_edit",
    },
    "fact-b34d490cfff5f65c5e81": {
        "active_index": 313,
        "question": "What roles did Minomura Kou serve for SM Collector?",
        "answer": "Minomura Kou served as an adviser, artist, author, bakushi, and critic for SM Collector.",
        "decision": "drop",
        "source_document": "data/raw/kinbakutoday_e2c96f38ced1cb5d.json",
        "source_document_file_sha256": "de7ff1efdfa682fe9ba748e63b416d07dcba02d2e56079dc89a713b7c62c62ab",
        "source_document_chars": 2600,
        "reason_code": "quarantine_role_list_redundant_with_active_magazine_significance",
        "reason": "The row is a role-list lookup fully subsumed by active fact-d9cd6ee0dfa1a1dd55da, which uses the same roles to explain why Minomura made SM Collector historically important.",
        "review_class": "redundant_role_list_drop",
        "redundant_survivor": "fact-d9cd6ee0dfa1a1dd55da",
    },
    "fact-cde2703b4a0aa3b6907f": {
        "active_index": 369,
        "question": "What was significant about the World of Rope video CineMagic released in 1989?",
        "answer": "It was the first of eight videos in Nureki Chimuo's landmark World of Rope series.",
        "decision": "edit",
        "edited_question": "What teaching value does Kinbaku Today attribute to Nureki Chimuo’s World of Rope video?",
        "edited_answer": "The article says the video pairs Nureki’s signature tying with CineMagic’s production to make his technique and approach easy to follow, then uses a series of ties and poses to emphasize aesthetic form.",
        "evidence": "The video features Nureki tying model Takahashi Megumi. This video combines Nureki’s signature tying style with CineMagic’s exceptional production values to create a video which is easy to follow and which provides great insight into Nureki’s technique and approach to kinbaku.\nThe video features a series of ties, each of which is illustrated with a collection of poses designed to highlight the aesthetic power of the form.",
        "source_document": "data/raw/kinbakutoday_b776fccd348e2538.json",
        "source_document_file_sha256": "a2f63bc64be180ef08f7d7056951d1e2381a3ebc00ea9a072f2dac3c071d8891",
        "source_document_chars": 2183,
        "reason_code": "replace_series_milestone_with_world_of_rope_teaching_value",
        "reason": "The replacement trades a release-count milestone for the article's substantive explanation of how the video communicates Nureki's technique, approach, and aesthetic use of poses.",
        "review_class": "documented_contribution_edit",
    },
    "fact-9110de496d2d11b8776e": {
        "active_index": 376,
        "question": "When did the series of gatherings continued by Kinbiken begin, and who led it?",
        "answer": "The gatherings began in 1986 under Nureki Chimuo, Naka sensei’s teacher and mentor.",
        "decision": "edit",
        "edited_question": "How does “Visiting Kinbiken” connect Naka Akira’s Kinbiken gathering to Nureki Chimuo?",
        "edited_answer": "The article describes the group as continuing gatherings that Nureki Chimuo—Naka Akira’s teacher and mentor—began leading in 1986.",
        "source_document": "data/raw/kinbakutoday_f6ccdaa49bed3fa5.json",
        "source_document_file_sha256": "31df3f1ad06d2a04da434818b6894f2bd1b481b142400997ddd07ec0a04dd086",
        "source_document_chars": 7180,
        "reason_code": "replace_kinbiken_date_lookup_with_lineage_continuity",
        "reason": "The replacement keeps the chronology but makes Nureki's teacher-mentor relationship to Naka and the group's continuity across generations the durable lesson.",
        "review_class": "lineage_relationship_edit",
    },
    "fact-36bcf08552b0e1711cda": {
        "active_index": 446,
        "question": "Why did Junko Mabuki’s run as Nikkatsu’s SM Queen end after 1980–81?",
        "answer": "After starring in ten films over two years, she retired in 1981 because she could not sustain the pace and rigorous demands of production.",
        "decision": "edit",
        "edited_question": "How does Kinbaku Today connect Junko Mabuki’s film roles to later Japanese bondage and SM imagery?",
        "edited_answer": "The article says her roles as teachers, nurses, secretaries, and airline stewardesses became iconic wardrobe in 1980s bondage and SM magazines.",
        "source_document": "data/raw/kinbakutoday_4f26f20c5f1dc7ba.json",
        "source_document_file_sha256": "b005e69110fb980e50161a01cd14ff639d93b069bcdb18d8af31dcb6527e23ad",
        "source_document_chars": 3920,
        "reason_code": "replace_retirement_minutiae_with_junko_mabuki_cultural_influence",
        "reason": "The replacement removes personal retirement detail and preserves the source's documented cultural contribution: how Mabuki's film roles shaped later bondage and SM magazine iconography.",
        "review_class": "documented_contribution_edit",
    },
    "fact-e652e2e850e409151a31": {
        "active_index": 472,
        "question": "Why does Kinbaku Today’s “Nikkatsu’s Mystery Bakushi” call Hitoshi Sharaku, the credited bondage adviser for Fairy in a Cage, a mystery bakushi?",
        "answer": "No one—not even people involved in many Nikkatsu productions—seems to know who the credited adviser was.",
        "decision": "drop",
        "source_document": "data/raw/kinbakutoday_92c9fc29a66300a0.json",
        "source_document_file_sha256": "93c467ba81f33881f6d4119d3b7c9fe0b87f435d32c380090677902850635316",
        "source_document_chars": 1119,
        "reason_code": "quarantine_unresolved_credited_identity_trivia",
        "reason": "The row asks why an otherwise undocumented credited adviser is considered mysterious; the snapshot offers no technique, contribution, or supported lineage relationship beyond the unresolved identity.",
        "review_class": "archive_identity_trivia_drop",
    },
}


REDUNDANT_SURVIVORS = {
    "fact-8bc60f112e90fb781203": {
        "question": "Where does Kinbaku Today’s “Kinbaku—An Evolving Era—Part 2” place the cultural roots of kinbaku, while distinguishing later use of Hojōjutsu techniques?",
        "answer": "It places kinbaku’s cultural roots in theater, while saying that modern kinbakushi later researched Hojōjutsu and incorporated some of its techniques.",
    },
    "fact-d9cd6ee0dfa1a1dd55da": {
        "question": "Why does “SM Collector: A New Life for an Old Title?” call the 1972–1985 magazine one of the most important of its second wave?",
        "answer": "It attributes the magazine’s importance largely to Minomura Kou, who served as adviser, artist, author, bakushi, and critic.",
    },
}


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v445 baseline drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v446-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v445.jsonl"
    build_baseline(base, inputs / "v445.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v446-observe-", dir=HERE) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v446-base-", dir=HERE) as tmp:
        directory = Path(tmp)
        base = directory / "v445.jsonl"
        build_baseline(base, directory / "v445.report.json")
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
            "reviewer": REVIEWER, "schema": "context-merit-audit-v446",
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
    if (observation["rows"], observation["eval"]) != (515, 612):
        raise ValueError(f"projection count drift: {observation}")
    if observation["before"] != EXPECTED_CAPACITY_BEFORE:
        raise ValueError(f"baseline capacity drift: {observation}")
    if EXPECTED_CAPACITY_AFTER is not None and observation["after"] != EXPECTED_CAPACITY_AFTER:
        raise ValueError(f"candidate capacity drift: {observation}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observation["sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError(f"projection hash drift: {observation}")

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
            "documented_contribution_repairs": 3,
            "historical_context_repairs": 1,
            "lineage_relationship_repairs": 1,
            "low_merit_rows_quarantined": 3,
            "terminology_repairs": 1,
            "terminology_rows_kept": 1,
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
        "schema": "context-merit-audit-report-v446",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "source_snapshot_inventory": {
            "documents": len(SPECS),
            "paths": [spec["source_document"] for spec in SPECS.values()],
            "total_characters": sum(spec["source_document_chars"] for spec in SPECS.values()),
        },
        "training_layer_contract": {
            "cleaned_markdown_site_corpora": "Distinct first-class future training artifacts; unfinished corpus output was neither read nor ingested by v446.",
            "derived_qa": "Distinct first-class training layer; v446 changes only existing sealed Q&A using attached evidence and pinned raw snapshots.",
            "deduplication_policy": "Do not collapse a cleaned Markdown document and its derived Q&A merely because they express the same source knowledge.",
            "provenance_requirement": "Link future derived Q&A to the cleaned source document through stable source URL and document or snapshot hashes.",
        },
        "v52_isolation": {"active_v52_inputs_modified": False, "candidate_status": "isolated_pending_not_promoted"},
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
