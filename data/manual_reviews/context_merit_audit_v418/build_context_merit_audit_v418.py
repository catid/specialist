#!/usr/bin/env python3
"""Complete three tutorial, nerve, and pole-precaution train-only answers."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V417=DATA/"manual_reviews/context_merit_audit_v417";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V417),str(V290)]
import build_context_merit_audit_v417 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v418.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v418.jsonl";REPORT=OUT_DIR/"report_context_merit_v418.json";BASELINE_ROWS=531;BASELINE_SHA256="ba2ca0ac7d352d2c9c739c48458dc9785223b7c18eada866ab01d05b521b8664";EXPECTED_OUTPUT_SHA256="ce5d5480f1605431a70de46e706c70f16a1bedbc3e87c62d11b31153afd86250"
EXPECTED_CAPACITY_BEFORE={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71};EXPECTED_CAPACITY_AFTER={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V417/"context_merit_audit_v417.jsonl",V417/"pending_curation_context_merit_v417.jsonl",V417/"report_context_merit_v417.json")
SPECS=(
 {"fact_id":"fact-b0b6086736d974c906d5","active_index":278,"expected_question":"What minimum safety checks does ESINEM suggest when evaluating a shibari tutorial?","expected_answer":"Check whether the tie has needed slack, can tighten accidentally, or creates strangulation or another serious risk.","question":"What minimum safety checks does ESINEM suggest when evaluating a shibari tutorial?","answer":"ESINEM suggests checking whether the tie has needed slack, can tighten accidentally, or creates strangulation or another serious risk.","reason_code":"complete_tutorial_safety_checks","reason":"The replacement turns the source-supported imperative into a complete attributed answer while retaining all three minimum checks."},
 {"fact_id":"fact-1a1cf65ecae6196bec6d","active_index":280,"expected_question":"What nerve-monitoring and placement checks does Rope365 give for an open-diamond box tie?","expected_answer":"Check hand sensation and finger/wrist movement, adjust arm placement away from sensitive spots, and prevent armpit pressure.","question":"What nerve-monitoring and placement checks does Rope365 give for an open-diamond box tie?","answer":"Rope365 says to check hand sensation and finger and wrist movement, adjust arm placement away from sensitive spots, and prevent armpit pressure.","reason_code":"complete_open_diamond_checks","reason":"The replacement turns the source-supported imperatives into a complete attributed answer while retaining every monitoring and placement check."},
 {"fact_id":"fact-5488149fb5203908b487","active_index":281,"expected_question":"What observable precautions does Rope365 give when a pole is trapped behind the knees?","expected_answer":"Avoid high compression at the upper back of the calf and keep monitoring the tied person's foot movement.","question":"What observable precautions does Rope365 give when a pole is trapped behind the knees?","answer":"Rope365 advises avoiding high compression at the upper back of the calf and continuing to monitor the tied person’s foot movement.","reason_code":"complete_behind_knee_pole_checks","reason":"The replacement turns the source-supported imperatives into a complete attributed answer while retaining both compression and movement precautions."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v417 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v418-input";d.mkdir(parents=True,exist_ok=True);base=d/"v417.jsonl";build_baseline(base,d/"v417.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v418-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v418-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v417.jsonl";build_baseline(base,d/"v417.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v418","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"tutorial_nerve_pole_answer_completeness_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v418","schema":"context-merit-audit-v418","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(531,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v418","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
