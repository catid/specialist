#!/usr/bin/env python3
"""Build three manually reviewed QAs from distinct Rope365 practice pages."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V298_DIR = DATA / "manual_reviews/context_merit_audit_v298"
sys.path[:0] = [str(ROOT), str(V298_DIR)]
import build_context_merit_audit_v298 as baseline_builder
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact, has_protocol_tokens, leakage_reason, normalize_text, parse_qa, stable_fact_id

OUT_DIR = Path(__file__).resolve().parent
OUTPUT = OUT_DIR / "pending_additions_technique_safety_tranche_10_v1.jsonl"
REPORT = OUT_DIR / "report_technique_safety_tranche_10_v1.json"
BASELINE_ROWS = 519
BASELINE_SHA256 = "20530df56ea5eca3d0e775f455a1448b91a997f5f0f4c2492868f7ee492b01ff"
EXPECTED_OUTPUT_SHA256 = "010dcb95ed3fc890f3c6d6d526be6807680cff696bd81375c099792e1bc2f75c"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"

SOURCES = {
    "hogtie_composition": {
        "path": DATA / "raw/rope_resources_v1/rope365__63efb59eb55aec811e0a.json",
        "url": "https://rope365.com/hog-tie/",
        "document_sha256": "2f873741c0b9977f447c476eb6bd6f1521981785a0e1b1569e4e305fd60ceae8",
        "markers": (
            "| Day 36: Ankles to Wrists – Make a minimalist hogtie by tying the ankles and wrists together. | ",
            "| Day 37: Ankles to Chest – Add a bit more structure by using a chest harness or a box tie as an anchor to attach the legs. | ",
        ),
    },
    "asymmetric_adaptation": {
        "path": DATA / "raw/rope_resources_v1/rope365__2e1c8399aa3ea0a48552.json",
        "url": "https://rope365.com/asymmetric-arms/",
        "document_sha256": "656059761648ca28d836e5a2b4c65977a9ac1b7500e2ffa865a2550bbc01a4a5",
        "markers": (
            "Our bodies aren’t completely symmetric. Differences can be big or small, and it becomes really important to have asymmetric ties in our toolbox to adapt when injuries happen.",
        ),
    },
    "hero_pose_support": {
        "path": DATA / "raw/rope_resources_v1/rope365__dd1eb0774eab03f2b704.json",
        "url": "https://rope365.com/sitting/",
        "document_sha256": "30e3c32d156ed65af0d4ea56ed9af0d6aa85a0698c15314a543a95d53a7490ab",
        "markers": (
            "Day 143: Hero Pose – Alt. petanko-zuwari ぺたんこ座り (flat sitting). Tie the feet on each side in a kneeling position. This position may be difficult depending on flexibility. Sit on a block or a cushion to make it easier.",
        ),
    },
}

FACTS = (
    {
        "source_key": "hogtie_composition",
        "topic": "hogtie_composition",
        "question": "How does a minimalist hogtie differ from a more structured version?",
        "answer": "A minimalist version connects the ankles and wrists, while a more structured version can attach the legs to a chest harness or box tie.",
        "paraphrase_rationale": "This contrasts the two explicitly described constructions without adding hidden video steps or recommending either version.",
    },
    {
        "source_key": "asymmetric_adaptation",
        "topic": "asymmetric_adaptation",
        "question": "Why can asymmetric arm ties be useful for adapting to a person?",
        "answer": "Bodies are not perfectly symmetric, and asymmetric options can accommodate side-to-side differences or injuries.",
        "paraphrase_rationale": "This preserves the source's body-variation and injury-adaptation rationale without suggesting a tie is automatically safe for an injury.",
    },
    {
        "source_key": "hero_pose_support",
        "topic": "hero_pose_support",
        "question": "How can hero pose be modified when tying someone who finds the position difficult?",
        "answer": "Place a block or cushion under the seated person.",
        "paraphrase_rationale": "This is the source's explicit flexibility modification in a short standalone answer.",
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
        raise ValueError("v298 baseline drift")
    return rows


def select_evidence(document: dict, markers: tuple[str, ...]) -> str:
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
        identity = document["url"], document["document_sha256"], text_sha256(document["text"])
        if identity != (source["url"], source["document_sha256"], source["document_sha256"]):
            raise ValueError(f"{key}: source identity drift")
        documents[key] = document

    with tempfile.TemporaryDirectory(prefix="technique-safety-v10-", dir=OUT_DIR) as temp:
        baseline = build_baseline(Path(temp) / "v298.jsonl", Path(temp) / "v298.report.json")
    baseline_facts = [EvalFact(row["question"], row["answer"], row["fact_id"], "train") for row in baseline]
    questions = {normalize_text(row["question"]) for row in baseline}
    pairs = {(normalize_text(row["question"]), normalize_text(row["answer"])) for row in baseline}
    documents_in_baseline = {row["document_sha256"] for row in baseline}
    urls_in_baseline = {row["url"].rstrip("/").casefold() for row in baseline}
    rows = []
    for fact in FACTS:
        source = SOURCES[fact["source_key"]]
        question, answer = fact["question"], fact["answer"]
        pair = normalize_text(question), normalize_text(answer)
        if not question.endswith("?") or "\n" in question or "\n" in answer:
            raise ValueError("addition is not standalone one-line Q&A")
        rendered = f"Question: {question}\nAnswer: {answer}"
        if has_protocol_tokens(question) or has_protocol_tokens(answer) or parse_qa(rendered) != (question, answer):
            raise ValueError("non-canonical addition")
        if pair in pairs or pair[0] in questions or leakage_reason(question, answer, baseline_facts):
            raise ValueError("collision with v298 train-only baseline")
        if source["document_sha256"] in documents_in_baseline or source["url"].rstrip("/").casefold() in urls_in_baseline:
            raise ValueError("source document or URL is not novel")
        evidence = select_evidence(documents[fact["source_key"]], source["markers"])
        rows.append({
            "answer": answer,
            "claim_type": "instructional",
            "document_sha256": source["document_sha256"],
            "evidence": evidence,
            "evidence_sha256": text_sha256(evidence),
            "evidence_url": source["url"],
            "fact_id": stable_fact_id(question, answer),
            "kind": "qa_resource_manual_fact",
            "paraphrase_rationale": fact["paraphrase_rationale"],
            "quality_schema": "manual-resource-fact-v1",
            "question": question,
            "resource_id": "rope365",
            "reviewer": "codex-technique-safety-additions-v10",
            "source": "rope365",
            "source_lineage": {"artifact": portable(OUTPUT), "raw_document": portable(source["path"]), "resource_manifest": portable(RESOURCE_MANIFEST)},
            "text": rendered,
            "topic": fact["topic"],
            "url": source["url"],
            "verified_at": "2026-07-15",
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
        "method": {"authoring": "manual full-source review and hand-authored Q&A", "collision_scope": "v298 train-only projection; sealed collisions delegated to integration tooling", "selection": "one bounded fact from each of three distinct, previously unrepresented documents and URLs"},
        "new_independent_inputs": {"document_sha256s": 3, "expected_strata": dict(sorted(strata.items())), "urls": 3},
        "reviewed_at": "2026-07-15",
        "reviewer": "codex-technique-safety-additions-v10",
        "schema": "manual-technique-safety-additions-report-v10",
        "status": "segregated_pending_integration",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
