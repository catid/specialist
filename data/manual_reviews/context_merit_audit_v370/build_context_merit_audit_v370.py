#!/usr/bin/env python3
"""Complete three practical train-only cuff, risk, and coiling answers."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V369=DATA/"manual_reviews/context_merit_audit_v369";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V369),str(V290)]
import build_context_merit_audit_v369 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v370.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v370.jsonl";REPORT=OUT_DIR/"report_context_merit_v370.json";BASELINE_ROWS=531;BASELINE_SHA256="99f39bc4791620e7dd38a15096f3c69c09dc3371a8e799579a2bb7c1bd1a5d4e";EXPECTED_OUTPUT_SHA256="86a39722ffc7d9110e72d2912041fc0974ddd794794af1c76f502e18f6edee56"
EXPECTED_CAPACITY_BEFORE={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71};EXPECTED_CAPACITY_AFTER={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V369/"context_merit_audit_v369.jsonl",V369/"pending_curation_context_merit_v369.jsonl",V369/"report_context_merit_v369.json")
SPECS=(
 {"fact_id":"fact-5c8e11633892885cb02b","active_index":31,"expected_question":"During Rope365’s inline-cuff self-evaluation, how much space should remain inside the cuff?","expected_answer":"two fingers","question":"During Rope365’s inline-cuff self-evaluation, how much space should remain inside the cuff?","answer":"Leave enough space for two fingers inside the cuff so it is neither too tight nor too loose.","reason_code":"complete_inline_cuff_spacing_guidance","reason":"The replacement turns a measurement fragment into the full fit check supported by the source."},
 {"fact_id":"fact-d0c0b7fea97ed3dc2bd2","active_index":48,"expected_question":"How does Anatomie Studio define safety when tying with a new rope partner?","expected_answer":"understanding and managing risk","question":"How does Anatomie Studio define safety when tying with a new rope partner?","answer":"Safety means understanding and managing risk, not eliminating every risk.","reason_code":"complete_new_partner_safety_definition","reason":"The replacement restores the source’s explicit contrast between risk management and the impossible elimination of all risk."},
 {"fact_id":"fact-3e451aebc0169c4a8746","active_index":118,"expected_question":"How should the securing knot be positioned in Rope365’s coiling checklist so it will not come undone during transport?","expected_answer":"pulled and centred","question":"How should the securing knot be positioned in Rope365’s coiling checklist so it will not come undone during transport?","answer":"Pull and center the securing knot so it will not come undone during transport.","reason_code":"complete_coiling_knot_position_instruction","reason":"The replacement turns the position fragment into the complete transport-security instruction supported by the checklist."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v369 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v370-input";d.mkdir(parents=True,exist_ok=True);base=d/"v369.jsonl";build_baseline(base,d/"v369.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v370-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v370-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v369.jsonl";build_baseline(base,d/"v369.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v370","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"practical_cuff_risk_coiling_answer_completeness_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v370","schema":"context-merit-audit-v370","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(531,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v370","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
