#!/usr/bin/env python3
"""Complete three cuff, finger, and cinch-placement train-only answers."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V421=DATA/"manual_reviews/context_merit_audit_v421";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V421),str(V290)]
import build_context_merit_audit_v421 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v422.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v422.jsonl";REPORT=OUT_DIR/"report_context_merit_v422.json";BASELINE_ROWS=531;BASELINE_SHA256="36a7bd4e519dd290f1026bfe4fb754a41ba4ff6f69ae50d32a17b361f5a5e7a5";EXPECTED_OUTPUT_SHA256="41e4c23472c50d69208a7d569d8229bf2c65e74f5d2587ee6a65301147402812"
EXPECTED_CAPACITY_BEFORE={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71};EXPECTED_CAPACITY_AFTER={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V421/"context_merit_audit_v421.jsonl",V421/"pending_curation_context_merit_v421.jsonl",V421/"report_context_merit_v421.json")
SPECS=(
 {"fact_id":"fact-e44040e7be0db98af59f","active_index":33,"expected_question":"During Rope365’s inline-cuff self-evaluation, how much space should remain inside the cuff?","expected_answer":"Leave enough space for two fingers inside the cuff so it is neither too tight nor too loose.","question":"During Rope365’s inline-cuff self-evaluation, how much space should remain inside the cuff?","answer":"Rope365 says to leave enough space for two fingers inside the cuff so it is neither too tight nor too loose.","reason_code":"complete_inline_cuff_spacing","reason":"The replacement turns the source-supported imperative into a complete attributed answer while retaining the two-finger measure and tension rationale."},
 {"fact_id":"fact-32ebbdd169a13338e287","active_index":119,"expected_question":"How should the fingers be positioned in Rope365's base kimono tie, and what does that preserve?","expected_answer":"Keep the fingers free rather than trapping them against the forearms so the unexpanded base remains escapable.","question":"How should the fingers be positioned in Rope365's base kimono tie, and what does that preserve?","answer":"Rope365 says to keep the fingers free rather than trapping them against the forearms so the unexpanded base remains escapable.","reason_code":"complete_kimono_finger_placement","reason":"The replacement turns the source-supported imperative into a complete attributed answer while retaining both finger placement and escapability."},
 {"fact_id":"fact-4065ea6bf9cff5674743","active_index":419,"expected_question":"Where should box-tie cinches avoid pressing around the upper arms?","expected_answer":"Keep cinches out of the armpits and away from the insides of the arms.","question":"Where should box-tie cinches avoid pressing around the upper arms?","answer":"Rope365 says box-tie cinches should stay out of the armpits and away from the insides of the arms.","reason_code":"complete_upper_arm_cinch_placement","reason":"The replacement turns the source-supported imperative into a complete attributed answer while retaining both avoided areas."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v421 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v422-input";d.mkdir(parents=True,exist_ok=True);base=d/"v421.jsonl";build_baseline(base,d/"v421.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v422-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v422-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v421.jsonl";build_baseline(base,d/"v421.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v422","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"cuff_finger_cinch_answer_completeness_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v422","schema":"context-merit-audit-v422","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(531,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v422","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
