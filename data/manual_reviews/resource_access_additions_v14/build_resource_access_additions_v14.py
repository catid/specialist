#!/usr/bin/env python3
"""Build three manually reviewed resource-access and creative-practice QAs."""
from __future__ import annotations
import json, sys, tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]; DATA = ROOT / "data"
V302 = DATA / "manual_reviews/context_merit_audit_v302"; V13 = DATA / "manual_reviews/technique_additions_v13"
sys.path[:0] = [str(ROOT), str(V302), str(V13)]
import build_context_merit_audit_v302 as baseline_builder
import build_technique_additions_v13 as prior
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact, has_protocol_tokens, leakage_reason, normalize_text, parse_qa, stable_fact_id

OUT_DIR = Path(__file__).resolve().parent
OUTPUT = OUT_DIR / "pending_additions_resource_access_tranche_14_v1.jsonl"
REPORT = OUT_DIR / "report_resource_access_tranche_14_v1.json"
BASELINE_ROWS = 531
BASELINE_SHA256 = "e71677ffdf831747d3e5f5287fce3812ed0a604b6eba114d28e5c0b5125e6a0e"
EXPECTED_OUTPUT_SHA256 = "3c7b6f7b54e81626129ca4580f9d9f6a3a7185b38846701cf6214eb3baef1b63"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
file_sha256, text_sha256, portable = prior.file_sha256, prior.text_sha256, prior.portable
read_jsonl, write_jsonl, select_evidence = prior.read_jsonl, prior.write_jsonl, prior.select_evidence

SOURCES = {
 "accessibility_routing": {"path": DATA/"raw/rope_resources_v1/tethered_together__0b96b6c5488d3a68347e.json", "url":"https://tetheredtogether.net/accessibility/", "document_sha256":"cd95429633b6847d752ee5c755e92186923a22bd85e5f45af08a4238bdd465f9", "source":"tethered_together", "resource_id":"tethered_together", "markers":("We want everyone to be able to enjoy Tethered Together. As such, the form below provides an avenue for attendees to provide information about their accessibility requests. Having as much information as possible will allow us the ability to fulfill reasonable requests.", "If you have questions about the accessibility of the facility, please contact our host hotel.")},
 "membership_access": {"path": DATA/"raw/rope_resources_v1/shibari_study__7b7df424f9d0383080e9.json", "url":"https://shibaristudy.com/pages/membership", "document_sha256":"eb13b35ba6a0eef9c81a3065711677b9afdd1b6649b177c533fd728f0739e82b", "source":"shibari_study", "resource_id":"shibari_study", "markers":("After signing up for a Shibari Study membership, you will have full access to the Shibari Study library of over 800 tutorials and regular live classes on our website and free mobile app as well as access to our exclusive member-only Discord community. These tutorials are available to stream as well as download on your device for on-the-go practice.",)},
 "furniture_ideation": {"path": DATA/"raw/rope_resources_v1/rope365__f67df1b75d78b633132e.json", "url":"https://rope365.com/furniture/", "document_sha256":"88d9146f8f973f0d7bb4a4bad516d693ef51eae27fac4a3adede78820fe31142", "source":"rope365", "resource_id":"rope365", "markers":("Tying people to objects brings infinite possibilities. Each piece of furniture enables us to create different shapes with the body. Look around you and find objects that inspire you: chairs, tables, sofas, desks, almost any piece of solid material can be a source of ideas.",)},
}
FACTS = (
 {"source_key":"accessibility_routing", "topic":"accessibility_routing", "question":"How does Tethered Together route event accessibility requests versus questions about the facility itself?", "answer":"Submit event requests through its accessibility channel, but direct facility-access questions to the host hotel.", "paraphrase_rationale":"This preserves the page's two destinations without copying a personal email address or a volatile deadline."},
 {"source_key":"membership_access", "topic":"membership_access", "question":"How can Shibari Study members access tutorials for practice away from a computer?", "answer":"They can use the mobile app and either stream tutorials or download them to a device for on-the-go practice.", "paraphrase_rationale":"This retains durable access modes while excluding prices, discounts, trial length, and promotional counts."},
 {"source_key":"furniture_ideation", "topic":"furniture_ideation", "question":"How does Rope365 suggest using furniture when designing a tie?", "answer":"Look for solid objects whose different parts can inspire and support different body shapes.", "paraphrase_rationale":"This distills the creative method without implying that every nearby object is structurally suitable."},
)

def build_baseline(path: Path, report: Path):
 baseline_builder.build_projection(path, report); rows=read_jsonl(path)
 if (len(rows), file_sha256(path)) != (BASELINE_ROWS, BASELINE_SHA256): raise ValueError("v302 baseline drift")
 return rows

def main():
 OUT_DIR.mkdir(parents=True, exist_ok=True); documents={}
 for key,s in SOURCES.items():
  d=json.loads(s["path"].read_text())
  if (d["url"],d["document_sha256"],text_sha256(d["text"])) != (s["url"],s["document_sha256"],s["document_sha256"]): raise ValueError(f"{key}: source drift")
  documents[key]=d
 with tempfile.TemporaryDirectory(prefix="resource-access-v14-",dir=OUT_DIR) as t: baseline=build_baseline(Path(t)/"v302.jsonl",Path(t)/"v302.report.json")
 facts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train") for r in baseline]
 qs={normalize_text(r["question"]) for r in baseline}; pairs={(normalize_text(r["question"]),normalize_text(r["answer"])) for r in baseline}
 docs={r["document_sha256"] for r in baseline}; urls={r["url"].rstrip("/").casefold() for r in baseline}; rows=[]
 for f in FACTS:
  s=SOURCES[f["source_key"]]; q,a=f["question"],f["answer"]; pair=normalize_text(q),normalize_text(a); rendered=f"Question: {q}\nAnswer: {a}"
  if not q.endswith("?") or "\n" in q+a or has_protocol_tokens(q) or has_protocol_tokens(a) or parse_qa(rendered)!=(q,a): raise ValueError("non-canonical addition")
  if pair in pairs or pair[0] in qs or leakage_reason(q,a,facts): raise ValueError("train collision")
  if s["document_sha256"] in docs or s["url"].rstrip("/").casefold() in urls: raise ValueError("source not novel")
  ev=select_evidence(documents[f["source_key"]],s["markers"])
  rows.append({"answer":a,"claim_type":"instructional","document_sha256":s["document_sha256"],"evidence":ev,"evidence_sha256":text_sha256(ev),"evidence_url":s["url"],"fact_id":stable_fact_id(q,a),"kind":"qa_resource_manual_fact","paraphrase_rationale":f["paraphrase_rationale"],"quality_schema":"manual-resource-fact-v1","question":q,"resource_id":s["resource_id"],"reviewer":"codex-resource-access-additions-v14","source":s["source"],"source_lineage":{"artifact":portable(OUTPUT),"raw_document":portable(s["path"]),"resource_manifest":portable(RESOURCE_MANIFEST)},"text":rendered,"topic":f["topic"],"url":s["url"],"verified_at":"2026-07-15"})
 if len(rows)!=3 or len({r["fact_id"] for r in rows})!=3: raise ValueError("identity drift")
 write_jsonl(OUTPUT,rows); sha=file_sha256(OUTPUT)
 if EXPECTED_OUTPUT_SHA256!="PENDING" and sha!=EXPECTED_OUTPUT_SHA256: raise ValueError("artifact drift")
 strata=Counter(classify_stratum(r) for r in rows)
 REPORT.write_text(json.dumps({"artifact":{"path":portable(OUTPUT),"rows":3,"sha256":sha},"baseline":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"method":{"authoring":"manual full-source review and hand-authored Q&A","collision_scope":"v302 train-only projection; sealed collisions delegated to integration tooling","selection":"one bounded access or creative-practice fact from three distinct new documents"},"new_independent_inputs":{"document_sha256s":3,"expected_strata":dict(sorted(strata.items())),"urls":3},"reviewed_at":"2026-07-15","reviewer":"codex-resource-access-additions-v14","schema":"manual-resource-access-additions-report-v14","status":"segregated_pending_integration"},ensure_ascii=False,indent=2,sort_keys=True)+"\n")

if __name__ == "__main__": main()
