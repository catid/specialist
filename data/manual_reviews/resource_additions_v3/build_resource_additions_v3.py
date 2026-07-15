#!/usr/bin/env python3
"""Build three manual resource QAs from distinct, previously unused pages."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V291_DIR = ROOT / "data/manual_reviews/context_merit_audit_v291"
sys.path[:0] = [str(ROOT), str(V291_DIR)]
import build_context_merit_audit_v291 as baseline_builder
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact, has_protocol_tokens, leakage_reason, normalize_text, parse_qa, stable_fact_id

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
OUTPUT = OUT_DIR / "pending_additions_resource_tranche_03_v1.jsonl"
REPORT = OUT_DIR / "report_resource_tranche_03_v1.json"
BASELINE_ROWS = 498
BASELINE_SHA256 = "ed516dffd88a6300945ead3b83062ca667d1b18977a6c96a8bcb6724880830fa"
EXPECTED_OUTPUT_SHA256 = "fc43f07ac642d33a477d548bbe35729ddd01ba55d09a6bed539110b9863e7636"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
SOURCES = {
    "learning_resource_roles": {
        "path": DATA / "raw/rope_resources_v1/rope365__aecd17aade64e5360148.json",
        "url": "https://rope365.com/getting-started/",
        "document_sha256": "b03a6511afeb7396afb5884883a264b4957eb7709389d67c8cf3682eacfb589a",
        "markers": ("This website is here to be your companion, so you can easily access ideas to learn, practice, and explore. It’s a learning program, it’s a practice guide to improve, it’s a compendium of ideas to keep your rope journey fresh and new.",),
    },
    "online_resource_directory": {
        "path": DATA / "raw/rope_resources_v1/rope365__5a84060d1c422a8c3b32.json",
        "url": "https://rope365.com/websites/",
        "document_sha256": "35c1d304f641b4207288ca161b06d9eb642eacf99e2acf22977482a934620a30",
        "markers": (
            "- Kinbaku Today by Zetsu – News and blog about Japanese rope bondage for the western audience.",
            "- Rope study by Ma’iitsoh Yazhi – The website of the Richmond, Virginia group. Includes free online classes: introduction to rope bondage and introduction to bottoming.",
            "- Tokyo Bound by Osada Steve – Blog with lots of interview and other information about Japanese bondage",
        ),
    },
    "free_video_locations": {
        "path": DATA / "raw/rope_resources_v1/rope365__8198e26dda1deeda15ff.json",
        "url": "https://rope365.com/videos/",
        "document_sha256": "0d27d6e36f5851d22020992e3ea9afcf63520fa9f0f74d157b50cf46bac50c56",
        "markers": ("You can find videos of the Rope365 activities on Youtube, Vimeo and also within the activity pages of Rope365.",),
    },
}
FACTS = (
    {
        "source_key": "learning_resource_roles", "topic": "learning_resource_roles",
        "question": "What three roles does Rope365 describe for its website?",
        "answer": "A learning program, a practice guide for improvement, and a compendium of ideas for keeping study fresh.",
        "paraphrase_rationale": "The answer retains the source's three descriptions while shortening promotional phrasing into a useful resource summary.",
    },
    {
        "source_key": "online_resource_directory", "topic": "online_resource_directory",
        "question": "Which sites in Rope365's directory provide Japanese-rope news, free introductory classes, and Japanese-bondage interviews, respectively?",
        "answer": "Kinbaku Today, Rope Study, and Tokyo Bound, respectively.",
        "paraphrase_rationale": "The mapping is assembled directly from three directory entries and preserves each entry's stated resource focus.",
    },
    {
        "source_key": "free_video_locations", "topic": "free_video_resources",
        "question": "Where does Rope365 say its free activity videos can be found?",
        "answer": "On YouTube, Vimeo, and within Rope365 activity pages.",
        "paraphrase_rationale": "The answer is a concise restatement of the three locations explicitly named by the source.",
    },
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def portable(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def build_baseline(path: Path, report: Path) -> list[dict]:
    baseline_builder.build_projection(path, report)
    rows = read_jsonl(path)
    if (len(rows), file_sha256(path)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v291 baseline drift")
    return rows


def evidence(document: dict, markers: tuple[str, ...]) -> str:
    selected = []
    for marker in markers:
        matches = [line for line in document["text"].splitlines() if marker in line]
        if len(matches) != 1:
            raise ValueError(f"evidence drift: {marker}")
        selected.append(matches[0])
    return "\n".join(selected)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    documents = {}
    for key, source in SOURCES.items():
        document = json.loads(source["path"].read_text())
        if (document["url"], document["document_sha256"], text_sha256(document["text"])) != (source["url"], source["document_sha256"], source["document_sha256"]):
            raise ValueError(f"{key}: source identity drift")
        documents[key] = document
    with tempfile.TemporaryDirectory(prefix="resources-v3-", dir=OUT_DIR) as temp:
        baseline = build_baseline(Path(temp) / "v291.jsonl", Path(temp) / "v291.report.json")
    baseline_facts = [EvalFact(row["question"], row["answer"], row["fact_id"], "train") for row in baseline]
    questions = {normalize_text(row["question"]) for row in baseline}
    pairs = {(normalize_text(row["question"]), normalize_text(row["answer"])) for row in baseline}
    document_ids = {row["document_sha256"] for row in baseline}
    urls = {row["url"].rstrip("/").casefold() for row in baseline}
    rows = []
    for fact in FACTS:
        source = SOURCES[fact["source_key"]]
        question, answer = fact["question"], fact["answer"]
        pair = normalize_text(question), normalize_text(answer)
        if not question.endswith("?") or "\n" in question or "\n" in answer:
            raise ValueError("addition is not standalone one-line Q&A")
        if has_protocol_tokens(question) or has_protocol_tokens(answer) or parse_qa(f"Question: {question}\nAnswer: {answer}") != (question, answer):
            raise ValueError("non-canonical addition")
        if pair in pairs or pair[0] in questions or leakage_reason(question, answer, baseline_facts):
            raise ValueError("collision with v291 train-only baseline")
        if source["document_sha256"] in document_ids or source["url"].rstrip("/").casefold() in urls:
            raise ValueError("source document or URL is not novel")
        support = evidence(documents[fact["source_key"]], source["markers"])
        rendered = f"Question: {question}\nAnswer: {answer}"
        rows.append({
            "answer": answer, "claim_type": "resource_navigation", "document_sha256": source["document_sha256"],
            "evidence": support, "evidence_sha256": text_sha256(support), "evidence_url": source["url"],
            "fact_id": stable_fact_id(question, answer), "kind": "qa_resource_manual_fact",
            "paraphrase_rationale": fact["paraphrase_rationale"], "quality_schema": "manual-resource-fact-v1",
            "question": question, "resource_id": "rope365", "reviewer": "codex-resource-additions-v3", "source": "rope365",
            "source_lineage": {"artifact": portable(OUTPUT), "raw_document": portable(source["path"]), "resource_manifest": portable(RESOURCE_MANIFEST)},
            "text": rendered, "topic": fact["topic"], "url": source["url"], "verified_at": "2026-07-15",
        })
    if len(rows) != 3 or len({row["fact_id"] for row in rows}) != 3:
        raise ValueError("addition tranche identity drift")
    write_jsonl(OUTPUT, rows)
    output_sha = file_sha256(OUTPUT)
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and output_sha != EXPECTED_OUTPUT_SHA256:
        raise ValueError("addition artifact hash drift")
    strata = Counter(classify_stratum(row) for row in rows)
    report = {
        "artifact": {"path": portable(OUTPUT), "rows": 3, "sha256": output_sha},
        "baseline": {"rows": BASELINE_ROWS, "sha256": BASELINE_SHA256},
        "method": {"authoring": "manual full-source review and hand-authored Q&A", "collision_scope": "v291 train-only projection; sealed collisions delegated to integration tooling", "selection": "one useful navigation fact from each of three distinct, previously unrepresented source documents and URLs"},
        "new_independent_inputs": {"document_sha256s": 3, "expected_strata": dict(sorted(strata.items())), "urls": 3},
        "reviewed_at": "2026-07-15", "reviewer": "codex-resource-additions-v3", "schema": "manual-resource-additions-report-v3",
        "sources": {key: {"document_sha256": value["document_sha256"], "file_sha256": file_sha256(value["path"]), "path": portable(value["path"]), "url": value["url"]} for key, value in sorted(SOURCES.items())},
        "status": "segregated_pending_integration",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
