#!/usr/bin/env python3
"""Build three hand-reviewed equipment/safety QAs from distinct documents."""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V300_DIR = DATA / "manual_reviews/context_merit_audit_v300"
V11_DIR = DATA / "manual_reviews/safety_additions_v11"
sys.path[:0] = [str(ROOT), str(V300_DIR), str(V11_DIR)]
import build_context_merit_audit_v300 as baseline_builder
import build_safety_additions_v11 as prior
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact, has_protocol_tokens, leakage_reason, normalize_text, parse_qa, stable_fact_id

OUT_DIR = Path(__file__).resolve().parent
OUTPUT = OUT_DIR / "pending_additions_equipment_safety_tranche_12_v1.jsonl"
REPORT = OUT_DIR / "report_equipment_safety_tranche_12_v1.json"
BASELINE_ROWS = 525
BASELINE_SHA256 = "b0315512acb0af95ff5fd0f0af835b21fcda293ff87c3cf9329a0ca1e44493a0"
EXPECTED_OUTPUT_SHA256 = "7dddff797d1e3c3bc8480b34b75b319d021190b22f7c4c8037ffb421d78f351e"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"

file_sha256 = prior.file_sha256
text_sha256 = prior.text_sha256
portable = prior.portable
read_jsonl = prior.read_jsonl
write_jsonl = prior.write_jsonl
select_evidence = prior.select_evidence

SOURCES = {
    "starter_kit_contents": {
        "path": DATA / "raw/rope_resources_v1/knothead_nylon__43ec85a40bb78a37db64.json",
        "url": "https://knotheadnylon.net/rope-kits",
        "document_sha256": "97aa829b04625e632c84dbbaa01cce10a1a24c75b0743c5b3c5d0848bc536fc8",
        "resource_id": "knothead_nylon",
        "source": "knothead_nylon",
        "markers": ("AmorPorCuerda Starter Kit (1x30', 4x15', Safety Shears, & Rope Bag)",),
    },
    "kit_quantity_planning": {
        "path": DATA / "raw/ropeconnections_d011c8053b8d931c.json",
        "url": "https://www.ropeconnections.com/what-should-you-have-in-your-rope-kit/",
        "document_sha256": "adcea1f65deab9c864ec45873145605b6fd3670ea9765814e6eb11a66aa71361",
        "resource_id": "ropeconnections",
        "source": "ropeconnections",
        "markers": (
            "However, that recipe is how I work out how much rope to have in my kit. I think about who I’m going to be tying; the number of lengths going into the ties, and the ties I’ll use; and then I’ll generally add a spare length plus some short bits. Plans may change; they frequently do. And short bits are always useful to have.",
        ),
    },
    "bamboo_knee_safety": {
        "path": DATA / "raw/rope_resources_v1/rope365__49a449996512426400ca.json",
        "url": "https://rope365.com/bamboo-postures/",
        "document_sha256": "4cd7ca39bad33039840fe7c78e399841765a7405a8ef2fa157476abb7abceca9",
        "resource_id": "rope365",
        "source": "rope365",
        "markers": (
            "Day 306: Behind the Knee – Create a trapeze by trapping the pole behind the knees and play with moving the pole around to move the body with it. Note that the some nerves are vulnerable at the top of the back of the calf, avoid high compression in this location and monitor foot movement.",
        ),
    },
}

FACTS = (
    {
        "source_key": "starter_kit_contents",
        "topic": "starter_kit_contents",
        "question": "What rope lengths and cutting equipment does Knot Head Nylon list in its AmorPorCuerda Starter Kit?",
        "answer": "One 30-foot rope, four 15-foot ropes, and safety shears; the kit also lists a rope bag.",
        "paraphrase_rationale": "This preserves the vendor-listed kit contents while excluding volatile price and availability details.",
    },
    {
        "source_key": "kit_quantity_planning",
        "topic": "kit_quantity_planning",
        "question": "How does the Rope Connections guide suggest planning how much rope to pack?",
        "answer": "Consider who and what you plan to tie, then add a spare full length and some shorter pieces.",
        "paraphrase_rationale": "This condenses the source's planning rationale without turning its personal ten-length example into a universal requirement.",
    },
    {
        "source_key": "bamboo_knee_safety",
        "topic": "bamboo_knee_safety",
        "question": "What lower-leg safety guidance does Rope365 give for a pole trapped behind the knees?",
        "answer": "Avoid high compression at the top of the back of the calf and monitor foot movement because nerves there are vulnerable.",
        "paraphrase_rationale": "This retains the exact vulnerable area and monitoring action without adding medical guarantees.",
    },
)


def document_sha256(document: dict) -> str:
    return document.get("document_sha256") or text_sha256(document["text"])


def build_baseline(path: Path, report: Path) -> list[dict]:
    baseline_builder.build_projection(path, report)
    rows = read_jsonl(path)
    if (len(rows), file_sha256(path)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v300 baseline drift")
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    documents = {}
    for key, source in SOURCES.items():
        document = json.loads(source["path"].read_text())
        identity = document["url"], document_sha256(document), text_sha256(document["text"])
        if identity != (source["url"], source["document_sha256"], source["document_sha256"]):
            raise ValueError(f"{key}: source identity drift")
        documents[key] = document

    with tempfile.TemporaryDirectory(prefix="equipment-safety-v12-", dir=OUT_DIR) as temp:
        baseline = build_baseline(Path(temp) / "v300.jsonl", Path(temp) / "v300.report.json")
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
            raise ValueError("collision with v300 train-only baseline")
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
            "resource_id": source["resource_id"],
            "reviewer": "codex-equipment-safety-additions-v12",
            "source": source["source"],
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
        "method": {"authoring": "manual full-source review and hand-authored Q&A", "collision_scope": "v300 train-only projection; sealed collisions delegated to integration tooling", "selection": "one bounded equipment or safety fact from each of three distinct, previously unrepresented documents and URLs"},
        "new_independent_inputs": {"document_sha256s": 3, "expected_strata": dict(sorted(strata.items())), "urls": 3},
        "reviewed_at": "2026-07-15",
        "reviewer": "codex-equipment-safety-additions-v12",
        "schema": "manual-equipment-safety-additions-report-v12",
        "status": "segregated_pending_integration",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
