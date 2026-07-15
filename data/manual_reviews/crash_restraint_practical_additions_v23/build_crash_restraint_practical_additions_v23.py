#!/usr/bin/env python3
"""Build three manually reviewed Crash Restraint practical QAs from train-only sources."""
from __future__ import annotations
import json,sys,tempfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V342=DATA/"manual_reviews/context_merit_audit_v342";sys.path[:0]=[str(ROOT),str(V342)]
import build_context_merit_audit_v342 as baseline_builder
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact,has_protocol_tokens,leakage_reason,normalize_text,parse_qa,stable_fact_id
OUT_DIR=Path(__file__).resolve().parent;OUTPUT=OUT_DIR/"pending_additions_crash_restraint_practical_tranche_23_v1.jsonl";REPORT=OUT_DIR/"report_crash_restraint_practical_tranche_23_v1.json";BASELINE_ROWS=528;BASELINE_SHA256="d49115f9f277f4d68cde82abc9efdf85ddbd98282120f8736d6e475d104c0c1e";EXPECTED_OUTPUT_SHA256="f4557c2d7bcb6aa1463f85dfe257350a176c571750cc0603fd5cbd77159aa3cf";RESOURCE_MANIFEST=ROOT/"sources/rope_resources_v1.json"
file_sha256,text_sha256,portable=baseline_builder.file_sha256,baseline_builder.text_sha256,baseline_builder.portable;read_jsonl,write_jsonl=baseline_builder.read_jsonl,baseline_builder.write_jsonl
SOURCES={
 "rope_kit":{"path":DATA/"raw/crash_restraint_building_rope_kit_20260714.json","url":"https://crash-restraint.com/ties/3","document_sha256":"fe4c22af6a09fc40d7cd142fc56046161f04cd3d12980cc4f4e5455ed3712cf2"},
 "negotiation":{"path":DATA/"raw/crash_restraint_negotiation_consent_20260714.json","url":"https://crash-restraint.com/ties/272","document_sha256":"353114c2622953d7b524e881e8291be284fd73d38d2caa78ce29bf56c940ea70"},
}
FACTS=(
 {"source_key":"rope_kit","topic":"beginner_rope_color_visibility","question":"Why does Crash Restraint advise beginners to avoid black or very dark rope?","answer":"Dark rope makes it harder for both the learner and the teacher to see what is happening.","evidence":"Color\nThe page advises beginners to avoid black or very dark rope because it is harder for the learner or teacher to see what is happening.","paraphrase_rationale":"This preserves the source's specific visibility reason without treating rope color as a general safety certification."},
 {"source_key":"rope_kit","topic":"rope_kit_length_mix","question":"What rope-length mix does Crash Restraint recommend when building a kit?","answer":"Build the kit mainly from the longest rope you can handle comfortably, plus several shorter ropes for finishing ties.","evidence":"Length\nIt recommends building a kit mainly from the longest rope the user can comfortably handle, plus several shorter ropes for finishing ties.","paraphrase_rationale":"This converts the source's kit-planning recommendation into a direct answer while keeping comfort, rather than a fixed length, as the limit."},
 {"source_key":"negotiation","topic":"new_partner_reaction_interpretation","question":"Which reactions does Crash Restraint say new partners should discuss how to interpret?","answer":"Laughter, silence, crying, and becoming non-verbal, because those reactions can mean different things for different people.","evidence":"What Does a Good or Bad Time Look Like?\nWith a new partner, discuss how to interpret their reactions: laughter, silence, crying, or becoming non-verbal can have different meanings. Past experience can help the top recognize what is happening.","paraphrase_rationale":"This retains the source's four named reactions and the practical reason for discussing them without assigning any universal meaning."},
)
def build_baseline(path,report):
 baseline_builder.build_projection(path,report);rows=read_jsonl(path)
 if (len(rows),file_sha256(path))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v342 drift")
 return rows
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix="crash-restraint-practical-v23-",dir=OUT_DIR) as t:baseline=build_baseline(Path(t)/"v342.jsonl",Path(t)/"v342.report.json")
 facts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train") for r in baseline];qs={normalize_text(r["question"]) for r in baseline};pairs={(normalize_text(r["question"]),normalize_text(r["answer"])) for r in baseline};rows=[]
 for f in FACTS:
  s=SOURCES[f["source_key"]];d=json.loads(s["path"].read_text())
  if (d["url"],text_sha256(d["text"]))!=(s["url"],s["document_sha256"]):raise ValueError("source drift")
  q,a=f["question"],f["answer"];pair=normalize_text(q),normalize_text(a);rendered=f"Question: {q}\nAnswer: {a}"
  if not q.endswith("?") or "\n" in q+a or has_protocol_tokens(q) or has_protocol_tokens(a) or parse_qa(rendered)!=(q,a):raise ValueError("noncanonical")
  if pair in pairs or pair[0] in qs or leakage_reason(q,a,facts):raise ValueError("train collision")
  if f["evidence"] not in d["text"]:raise ValueError("evidence drift")
  ev=f["evidence"];rows.append({"answer":a,"claim_type":"instructional","document_sha256":s["document_sha256"],"evidence":ev,"evidence_sha256":text_sha256(ev),"evidence_url":s["url"],"fact_id":stable_fact_id(q,a),"kind":"qa_resource_manual_fact","paraphrase_rationale":f["paraphrase_rationale"],"quality_schema":"manual-resource-fact-v1","question":q,"resource_id":"crash_restraint","reviewer":"codex-crash-restraint-practical-additions-v23","source":"crash_restraint","source_lineage":{"artifact":portable(OUTPUT),"raw_document":portable(s["path"]),"resource_manifest":portable(RESOURCE_MANIFEST)},"text":rendered,"topic":f["topic"],"url":s["url"],"verified_at":"2026-07-15"})
 if len(rows)!=3 or len({r["fact_id"] for r in rows})!=3:raise ValueError("identity drift")
 write_jsonl(OUTPUT,rows);sha=file_sha256(OUTPUT)
 if EXPECTED_OUTPUT_SHA256!="PENDING" and sha!=EXPECTED_OUTPUT_SHA256:raise ValueError("artifact drift")
 strata=Counter(classify_stratum(r) for r in rows);REPORT.write_text(json.dumps({"artifact":{"path":portable(OUTPUT),"rows":3,"sha256":sha},"baseline":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"method":{"authoring":"manual train-only source review and hand-authored Q&A","collision_scope":"v342 train-only projection; sealed collisions delegated to integration tooling","selection":"three distinct practical facts from two underused owner-recommended Crash Restraint documents"},"new_independent_inputs":{"document_sha256s":0,"expected_strata":dict(sorted(strata.items())),"urls":0},"reviewed_at":"2026-07-15","reviewer":"codex-crash-restraint-practical-additions-v23","schema":"manual-crash-restraint-practical-additions-report-v23","status":"segregated_pending_integration"},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
