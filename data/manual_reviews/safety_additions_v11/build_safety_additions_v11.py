#!/usr/bin/env python3
"""Build three manually reviewed safety QAs from distinct source documents."""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V299_DIR = DATA / "manual_reviews/context_merit_audit_v299"
V10_DIR = DATA / "manual_reviews/technique_safety_additions_v10"
sys.path[:0] = [str(ROOT), str(V299_DIR), str(V10_DIR)]
import build_context_merit_audit_v299 as baseline_builder
import build_technique_safety_additions_v10 as prior
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact, has_protocol_tokens, leakage_reason, normalize_text, parse_qa, stable_fact_id

OUT_DIR = Path(__file__).resolve().parent
OUTPUT = OUT_DIR / "pending_additions_safety_tranche_11_v1.jsonl"
REPORT = OUT_DIR / "report_safety_tranche_11_v1.json"
BASELINE_ROWS = 522
BASELINE_SHA256 = "6f65b7ac17fd30e92cb843d8fcc158e0532b725d61af834501bd592a709918d0"
EXPECTED_OUTPUT_SHA256 = "630cec95eabab4cf266c26e6e2072c72cda2de5c0e0217fcc05c83170d18bedb"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"

file_sha256 = prior.file_sha256
text_sha256 = prior.text_sha256
portable = prior.portable
read_jsonl = prior.read_jsonl
write_jsonl = prior.write_jsonl
select_evidence = prior.select_evidence

SOURCES = {
    "mummification_release": {
        "path": DATA / "raw/rope_resources_v1/rope365__342822199a7fe54eec9b.json",
        "url": "https://rope365.com/body-challenges/",
        "document_sha256": "983c1e42c3f6d04e9faffd3e89b0d955f9b8479a8cc55d19a8b340382692b418",
        "resource_id": "rope365",
        "source": "rope365",
        "markers": (
            "| Day 167: Mummification – Wrap the entire body to create a full-body restraint. Add more rope to create a cocoon by wrapping the body progressively more and more. Make sure to use a few quick-releases and keep the scissors close in case the tie becomes overwhelming. Inspirations and Resources | ",
        ),
    },
    "tripod_shear_inspection": {
        "path": DATA / "raw/rope_resources_v1/rope365__153b0f6d1f2932a77626.json",
        "url": "https://rope365.com/diy-freestanding-hardpoints/",
        "document_sha256": "5eeb3548377c02365b9333aee6184d1b7b84f562d695072ff017cc1bfb5898b9",
        "resource_id": "rope365",
        "source": "rope365",
        "markers": (
            "Tripods also have a unique weakness you should pay attention to. The rotational force of a center-weighted load can cause shearing, similar to twisting the top off of a bottle of soda. You should regularly inspect the lashing/welding of any center-weighted tripod for this reason, and this isa big reason why many of the steel-topper providers have moved to a more distributed mounting pattern.",
        ),
    },
    "mid_scene_renegotiation": {
        "path": DATA / "raw/rope_resources_v1/tethered_together__d5594a12694c5468f698.json",
        "url": "https://tetheredtogether.net/consent-policy/",
        "document_sha256": "730bcefd52c490a099a9a9c699823968c6e23ebb4e0c24b359ad5c471c65138b",
        "resource_id": "tethered_together",
        "source": "tethered_together",
        "markers": (
            "- Depending on all participants’ state of mind, we recommend that you don’t renegotiate in the middle of your scene. When a person is in subspace or otherwise not in a clear state of mind, you may not have informed consent even though that person may appear to agree in the heat of the moment.",
        ),
    },
}

FACTS = (
    {
        "source_key": "mummification_release",
        "topic": "mummification_release",
        "question": "What emergency-release precautions does Rope365 recommend for a mummification tie?",
        "answer": "Include several quick releases and keep scissors close in case the tie becomes overwhelming.",
        "paraphrase_rationale": "This keeps the source's two explicit release precautions together and avoids adding unshown construction advice.",
    },
    {
        "source_key": "tripod_shear_inspection",
        "topic": "tripod_shear_inspection",
        "question": "What safety inspection does Rope365 recommend for a center-weighted tripod, and why?",
        "answer": "Inspect its lashings or welds because rotational force can cause shearing.",
        "paraphrase_rationale": "This preserves the source's specific failure mechanism and corresponding inspection target in a standalone answer.",
    },
    {
        "source_key": "mid_scene_renegotiation",
        "topic": "mid_scene_renegotiation",
        "question": "Why does Tethered Together recommend avoiding mid-scene renegotiation when someone is in subspace or not clear-minded?",
        "answer": "Their apparent agreement may not amount to informed consent in that state.",
        "paraphrase_rationale": "This attributes the event's recommendation and retains its informed-consent rationale without generalizing policy into law.",
    },
)


def build_baseline(path: Path, report: Path) -> list[dict]:
    baseline_builder.build_projection(path, report)
    rows = read_jsonl(path)
    if (len(rows), file_sha256(path)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v299 baseline drift")
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    documents = {}
    for key, source in SOURCES.items():
        document = json.loads(source["path"].read_text())
        identity = document["url"], document["document_sha256"], text_sha256(document["text"])
        if identity != (source["url"], source["document_sha256"], source["document_sha256"]):
            raise ValueError(f"{key}: source identity drift")
        documents[key] = document

    with tempfile.TemporaryDirectory(prefix="safety-v11-", dir=OUT_DIR) as temp:
        baseline = build_baseline(Path(temp) / "v299.jsonl", Path(temp) / "v299.report.json")
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
            raise ValueError("collision with v299 train-only baseline")
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
            "reviewer": "codex-safety-additions-v11",
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
        "method": {"authoring": "manual full-source review and hand-authored Q&A", "collision_scope": "v299 train-only projection; sealed collisions delegated to integration tooling", "selection": "one bounded safety fact from each of three distinct, previously unrepresented documents and URLs"},
        "new_independent_inputs": {"document_sha256s": 3, "expected_strata": dict(sorted(strata.items())), "urls": 3},
        "reviewed_at": "2026-07-15",
        "reviewer": "codex-safety-additions-v11",
        "schema": "manual-safety-additions-report-v11",
        "status": "segregated_pending_integration",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
