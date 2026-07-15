#!/usr/bin/env python3
"""Build three manually reviewed learning-method QAs from one source document."""
from __future__ import annotations
import json,sys,tempfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; DATA=ROOT/"data"; V308=DATA/"manual_reviews/context_merit_audit_v308"; V19=DATA/"manual_reviews/structure_psychology_additions_v19"
sys.path[:0]=[str(ROOT),str(V308),str(V19)]
import build_context_merit_audit_v308 as baseline_builder
import build_structure_psychology_additions_v19 as prior
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact,has_protocol_tokens,leakage_reason,normalize_text,parse_qa,stable_fact_id
OUT_DIR=Path(__file__).resolve().parent; OUTPUT=OUT_DIR/"pending_additions_learning_method_tranche_20_v1.jsonl"; REPORT=OUT_DIR/"report_learning_method_tranche_20_v1.json"
BASELINE_ROWS=548; BASELINE_SHA256="24ecadeeff3e4d7fc726e4dcc2429fb154843fb37477f5a02e45a9dd05b36da5"; EXPECTED_OUTPUT_SHA256="18db883efaab6ed5faf552b07fb49ca33a9b68c20401fe73579bd4c5982c2ee4"; RESOURCE_MANIFEST=ROOT/"sources/rope_resources_v1.json"
file_sha256,text_sha256,portable=prior.file_sha256,prior.text_sha256,prior.portable; read_jsonl,write_jsonl,select_evidence=prior.read_jsonl,prior.write_jsonl,prior.select_evidence
SOURCE={"path":DATA/"raw/rope_resources_v1/rope365__cf6e780bd4f6b387435e.json","url":"https://rope365.com/learning/","document_sha256":"d86a023f3eaf2e1dd1a1804f513a51218dfd17dcdd81eb103ca6df0ba3ae8aff"}
FACTS=(
 {"topic":"source_before_repetition","question":"Why does Rope365 recommend learning a technique from a good source before practicing it repeatedly?","answer":"Repeatedly practicing an inefficient or dangerous technique can create bad habits that are hard to break.","markers":("A common pitfall is practicing inefficient or dangerous techniques, which will create bad habits that are hard to break. This is why learning from a good source is essential before going into practice mode.",),"paraphrase_rationale":"This preserves the stated practice failure mode and the reason for source selection without claiming any source is infallible."},
 {"topic":"online_image_limitations","question":"Why should rope learners be cautious about treating online photos as instruction?","answer":"Photos omit the tying process, which can create unrealistic expectations, and online representation can reflect social biases.","markers":("This abundance also comes with some pitfalls, seeing pictures without seeing their process may lead to unrealistic expectations of what rope can be or should be. Society’s biases filter through online representation, but rope is an adaptable medium that can be for people of any age, ethnicity, gender, sexual orientation, weight, etc.",),"paraphrase_rationale":"This combines the source's process-visibility and representation cautions without converting its inclusive claim into a safety guarantee."},
 {"topic":"building_block_learning","question":"How does Rope365 say its activities should be used instead of as step-by-step recipes?","answer":"Learn the building blocks and their tradeoffs so you can create your own version, since there is usually more than one way to achieve a result.","markers":("The goal is not to follow a step-by-step guide for everything, but rather to understand the building blocks in order to be able to create your own version of different concepts. There is always more than one way to do anything with different pros and cons.",),"paraphrase_rationale":"This preserves the page's intended learning model and explicit tradeoff framing."},
)
def build_baseline(path,report):
 baseline_builder.build_projection(path,report); rows=read_jsonl(path)
 if (len(rows),file_sha256(path))!=(BASELINE_ROWS,BASELINE_SHA256): raise ValueError("v308 drift")
 return rows
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True); document=json.loads(SOURCE["path"].read_text())
 if (document["url"],document["document_sha256"],text_sha256(document["text"]))!=(SOURCE["url"],SOURCE["document_sha256"],SOURCE["document_sha256"]): raise ValueError("source drift")
 with tempfile.TemporaryDirectory(prefix="learning-method-v20-",dir=OUT_DIR) as t: baseline=build_baseline(Path(t)/"v308.jsonl",Path(t)/"v308.report.json")
 facts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train") for r in baseline]; qs={normalize_text(r["question"]) for r in baseline}; pairs={(normalize_text(r["question"]),normalize_text(r["answer"])) for r in baseline}; docids={r["document_sha256"] for r in baseline}; urls={r["url"].rstrip("/").casefold() for r in baseline}; rows=[]
 if SOURCE["document_sha256"] in docids or SOURCE["url"].rstrip("/").casefold() in urls: raise ValueError("source not novel")
 for f in FACTS:
  q,a=f["question"],f["answer"]; pair=normalize_text(q),normalize_text(a); rendered=f"Question: {q}\nAnswer: {a}"
  if not q.endswith("?") or "\n" in q+a or has_protocol_tokens(q) or has_protocol_tokens(a) or parse_qa(rendered)!=(q,a): raise ValueError("noncanonical")
  if pair in pairs or pair[0] in qs or leakage_reason(q,a,facts): raise ValueError("train collision")
  ev=select_evidence(document,f["markers"]); rows.append({"answer":a,"claim_type":"instructional","document_sha256":SOURCE["document_sha256"],"evidence":ev,"evidence_sha256":text_sha256(ev),"evidence_url":SOURCE["url"],"fact_id":stable_fact_id(q,a),"kind":"qa_resource_manual_fact","paraphrase_rationale":f["paraphrase_rationale"],"quality_schema":"manual-resource-fact-v1","question":q,"resource_id":"rope365","reviewer":"codex-learning-method-additions-v20","source":"rope365","source_lineage":{"artifact":portable(OUTPUT),"raw_document":portable(SOURCE["path"]),"resource_manifest":portable(RESOURCE_MANIFEST)},"text":rendered,"topic":f["topic"],"url":SOURCE["url"],"verified_at":"2026-07-15"})
 if len(rows)!=3 or len({r["fact_id"] for r in rows})!=3: raise ValueError("identity drift")
 write_jsonl(OUTPUT,rows); sha=file_sha256(OUTPUT)
 if EXPECTED_OUTPUT_SHA256!="PENDING" and sha!=EXPECTED_OUTPUT_SHA256: raise ValueError("artifact drift")
 strata=Counter(classify_stratum(r) for r in rows); REPORT.write_text(json.dumps({"artifact":{"path":portable(OUTPUT),"rows":3,"sha256":sha},"baseline":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"excluded_source":[{"url":"https://rope365.com/bed/","decision":"reject","reason":"The page offers attachment ideas but no anchor-integrity or load check, so no unstated safety rule was inferred."},{"url":"https://rope365.com/fall/","decision":"reject","reason":"Its prerequisite navigation duplicates the existing Spring curriculum row."},{"url":"https://rope365.com/winter/","decision":"reject","reason":"Its navigation is redundant and its catalog includes unsafe self-choking and other high-risk prompts."}],"method":{"authoring":"manual full-source review and hand-authored Q&A","collision_scope":"v308 train-only projection; sealed collisions delegated to integration tooling","selection":"three distinct learning lessons from one fully reviewed new document; counted as one conservative conflict unit"},"new_independent_inputs":{"document_sha256s":1,"expected_strata":dict(sorted(strata.items())),"urls":1},"reviewed_at":"2026-07-15","reviewer":"codex-learning-method-additions-v20","schema":"manual-learning-method-additions-report-v20","status":"segregated_pending_integration"},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
