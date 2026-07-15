#!/usr/bin/env python3
"""Build three manually reviewed anatomy/creative-practice QAs."""
from __future__ import annotations
import json,sys,tempfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; DATA=ROOT/"data"; V304=DATA/"manual_reviews/context_merit_audit_v304"; V15=DATA/"manual_reviews/multiperson_chest_additions_v15"
sys.path[:0]=[str(ROOT),str(V304),str(V15)]
import build_context_merit_audit_v304 as baseline_builder
import build_multiperson_chest_additions_v15 as prior
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact,has_protocol_tokens,leakage_reason,normalize_text,parse_qa,stable_fact_id
OUT_DIR=Path(__file__).resolve().parent; OUTPUT=OUT_DIR/"pending_additions_anatomy_creativity_tranche_16_v1.jsonl"; REPORT=OUT_DIR/"report_anatomy_creativity_tranche_16_v1.json"; BASELINE_ROWS=536; BASELINE_SHA256="b9260425ba23c54413c771840f584f621120f3891a1b6ef1380fe732f38de68d"; EXPECTED_OUTPUT_SHA256="447a11189f62ed84e504a9858e101e50370fbe39accba0fb04406d09cba7bf1f"; RESOURCE_MANIFEST=ROOT/"sources/rope_resources_v1.json"
file_sha256,text_sha256,portable=prior.file_sha256,prior.text_sha256,prior.portable; read_jsonl,write_jsonl,select_evidence=prior.read_jsonl,prior.write_jsonl,prior.select_evidence
SOURCES={
 "finger_caution":{"path":DATA/"raw/rope_resources_v1/rope365__e8eb7de51d99e927226e.json","url":"https://rope365.com/hands-and-fingers/","document_sha256":"ad74e9276dc15eea62a9f72f53f6c9a32c6a8c3eb08c9826c086b515fb8c88b1","markers":("Hands are a fascinating part of the body with their many bones, nerves, muscles, and their ability to feel and manipulate objects with precision. There is also a very powerful psychological aspect of controlling the movement of hands and fingers. This complexity comes at the cost of fragility, we have to be mindful when tying the fingers as injuring them can have lasting effects on someone’s life.",)},
 "shoes_foot_tie":{"path":DATA/"raw/rope_resources_v1/rope365__79dd5614e734dbf898fa.json","url":"https://rope365.com/feet-and-toes/","document_sha256":"e11a81c20016e5624fdddbb5d4832d26c27525bc8fa8deb4bc05c115d53a6628","markers":("Day 169: Shoes and Heels – Tying with shoes on changes how the rope feels and ankles will behave. It adds some protection to the feet and heels make an interesting attachment point. Bonus: how is tying with socks different from a bare foot.",)},
 "pattern_transfer":{"path":DATA/"raw/rope_resources_v1/rope365__16b487bde6f7f1c12241.json","url":"https://rope365.com/creativity/","document_sha256":"d442e9f505abc9c28fe22fb35cdfa238ecf554936bbcafd569e77059ab4e2220","markers":("| Day 255: Pattern – Pick a shape, a pattern, and apply the technique to different body parts and structures. Make diamonds everywhere on the body. How many types of ladder can you make? Can you tie your favorite box tie pattern on your lower body? | ",)},
}
FACTS=(
 {"source_key":"finger_caution","topic":"finger_tying_caution","question":"Why does Rope365 urge extra safety caution when tying fingers?","answer":"Hands contain many small bones, nerves, and muscles needed for precise function, so a finger injury can have lasting effects.","paraphrase_rationale":"This preserves the anatomical reason and lasting-impact warning without operationalizing the page's risky finger restraints."},
 {"source_key":"shoes_foot_tie","topic":"shoes_change_foot_tie","question":"How can shoes change a foot tie according to Rope365?","answer":"They change how the rope feels and how the ankles behave, add some foot protection, and can provide heel attachment points.","paraphrase_rationale":"This combines the source's three stated effects without treating shoes as sufficient protection."},
 {"source_key":"pattern_transfer","topic":"pattern_transfer","question":"How does Rope365 suggest using an existing pattern to generate a new tie?","answer":"Apply the shape or pattern to different body parts or structures, such as moving a familiar box-tie pattern to the lower body.","paraphrase_rationale":"This turns the day prompt into a transferable creative method while retaining its concrete example."},
)
def build_baseline(path,report):
 baseline_builder.build_projection(path,report); rows=read_jsonl(path)
 if (len(rows),file_sha256(path))!=(BASELINE_ROWS,BASELINE_SHA256): raise ValueError("v304 drift")
 return rows
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True); docs={}
 for k,s in SOURCES.items():
  d=json.loads(s["path"].read_text())
  if (d["url"],d["document_sha256"],text_sha256(d["text"]))!=(s["url"],s["document_sha256"],s["document_sha256"]): raise ValueError(f"{k}: source drift")
  docs[k]=d
 with tempfile.TemporaryDirectory(prefix="anatomy-creativity-v16-",dir=OUT_DIR) as t: baseline=build_baseline(Path(t)/"v304.jsonl",Path(t)/"v304.report.json")
 facts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train") for r in baseline]; qs={normalize_text(r["question"]) for r in baseline}; pairs={(normalize_text(r["question"]),normalize_text(r["answer"])) for r in baseline}; docids={r["document_sha256"] for r in baseline}; urls={r["url"].rstrip("/").casefold() for r in baseline}; rows=[]
 for f in FACTS:
  s=SOURCES[f["source_key"]]; q,a=f["question"],f["answer"]; pair=normalize_text(q),normalize_text(a); rendered=f"Question: {q}\nAnswer: {a}"
  if not q.endswith("?") or "\n" in q+a or has_protocol_tokens(q) or has_protocol_tokens(a) or parse_qa(rendered)!=(q,a): raise ValueError("noncanonical")
  if pair in pairs or pair[0] in qs or leakage_reason(q,a,facts): raise ValueError("train collision")
  if s["document_sha256"] in docids or s["url"].rstrip("/").casefold() in urls: raise ValueError("source not novel")
  ev=select_evidence(docs[f["source_key"]],s["markers"]); rows.append({"answer":a,"claim_type":"instructional","document_sha256":s["document_sha256"],"evidence":ev,"evidence_sha256":text_sha256(ev),"evidence_url":s["url"],"fact_id":stable_fact_id(q,a),"kind":"qa_resource_manual_fact","paraphrase_rationale":f["paraphrase_rationale"],"quality_schema":"manual-resource-fact-v1","question":q,"resource_id":"rope365","reviewer":"codex-anatomy-creativity-additions-v16","source":"rope365","source_lineage":{"artifact":portable(OUTPUT),"raw_document":portable(s["path"]),"resource_manifest":portable(RESOURCE_MANIFEST)},"text":rendered,"topic":f["topic"],"url":s["url"],"verified_at":"2026-07-15"})
 if len(rows)!=3 or len({r["fact_id"] for r in rows})!=3: raise ValueError("identity drift")
 write_jsonl(OUTPUT,rows); sha=file_sha256(OUTPUT)
 if EXPECTED_OUTPUT_SHA256!="PENDING" and sha!=EXPECTED_OUTPUT_SHA256: raise ValueError("artifact drift")
 strata=Counter(classify_stratum(r) for r in rows); REPORT.write_text(json.dumps({"artifact":{"path":portable(OUTPUT),"rows":3,"sha256":sha},"baseline":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"excluded_source":{"url":"https://rope365.com/body/","decision":"reject","reason":"Risky pressure-point and neck practice prompts were not operationalized; the safe anatomy meta-point duplicates stronger existing rows."},"method":{"authoring":"manual full-source review and hand-authored Q&A","collision_scope":"v304 train-only projection; sealed collisions delegated to integration tooling","selection":"one bounded fact from each of three distinct new documents"},"new_independent_inputs":{"document_sha256s":3,"expected_strata":dict(sorted(strata.items())),"urls":3},"reviewed_at":"2026-07-15","reviewer":"codex-anatomy-creativity-additions-v16","schema":"manual-anatomy-creativity-additions-report-v16","status":"segregated_pending_integration"},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
