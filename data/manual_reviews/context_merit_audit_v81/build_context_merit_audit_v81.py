#!/usr/bin/env python3
"""Audit all remaining direct Q&A from Rope365's quick-release lesson."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V80_DIR=DATA/"manual_reviews/context_merit_audit_v80";sys.path[:0]=[str(ROOT),str(V80_DIR)]
import build_context_merit_audit_v80 as previous
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v81.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v81.jsonl";REPORT=OUT_DIR/"report_context_merit_v81.json";REVIEWER="codex-context-merit-audit-v81";REVIEWED_AT="2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET=previous.ACTIVE_DATASET;ACTIVE_REPORT=previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION=previous.QUALITY_MERIT_CURATION;TASUKI_CURATION=previous.TASUKI_CURATION;CORE=previous.CORE;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl
CONTEXT_CURATIONS=previous.OUTPUT_CONTEXT_CURATIONS;PRIOR_PROJECTION_CURATIONS=previous.OUTPUT_PROJECTION_CURATIONS;OUTPUT_CONTEXT_CURATIONS=(*CONTEXT_CURATIONS,CURATION);OUTPUT_PROJECTION_CURATIONS=(*PRIOR_PROJECTION_CURATIONS,CURATION);SOURCE=DATA/"raw/rope_resources_v1/rope365__125a6aa4fb7e0a18174d.json"
SPECS=(
 {"fact_id":"fact-e4607b09df701ffca88e","active_index":52,"marker":"great in case of emergency, or for ties that may need adjustments along the way","decision":"keep","reason_code":"retain_quick_release_use_cases","reason":"The existing Q&A clearly retains the source’s two useful cases for a less-solid quick-release lock."},
 {"fact_id":"fact-a3fb0bf625fecebfa5e0","active_index":287,"marker":"make a half hitch on the loop created by the first half hitch to lock the exit into place","decision":"keep","reason_code":"retain_slipped_half_hitch_exit_lock","reason":"The existing Q&A accurately identifies the extra half hitch that locks the slipped half hitch’s exit."},
 {"fact_id":"fact-5737467ffbd1d79335da","active_index":389,"marker":"monitoring the capacity to open and close the hands","decision":"edit","question":"When a quick-release cuff binds the arms behind the back, what hand function does Rope365 say to monitor?","answer":"the ability to open and close the hands","reason_code":"clarify_quick_release_cuff_motor_check","reason":"The revised Q&A names hand function directly and replaces the awkward ‘what capacity’ construction while preserving the motor check."},
 {"fact_id":"fact-c6715c55bd2f15780891","active_index":423,"marker":"All cuffs are located in a safe place on the body, away from joints and nerves","decision":"keep","reason_code":"retain_quick_release_cuff_location_check","reason":"The existing Q&A clearly states the checklist’s joint-and-nerve avoidance requirement for cuff placement."},
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES={s["fact_id"]:s["active_index"] for s in SPECS};PROJECTED_SELECTION_BASELINE={"description":"isolated corrected training projection through context-merit v80","direct_rows_without_prior_curation":125,"rope365_quick_release_rows_selected":4,"rows":536,"sha256":"b504ce60ef758e1f3661dd265dc2417efb49cf72f1fab3ee0794f591ef3a30f3"};EXPECTED_OUTPUT_SHA256="c7eec4ce0778e3c1a720a0bcce24560826d28c2031476a54a1c1dee053b63af3"
def build_projection(o,r,c):previous.build_projection(o,r,c)
def prior_decision_artifacts():
 out=[]
 for v in range(1,81):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def evidence(doc,marker):
 matches=[line for line in doc["text"].splitlines() if marker in line]
 if len(matches)!=1:raise ValueError(f"evidence drift: {marker}")
 return matches[0]
def observation():
 with tempfile.TemporaryDirectory(prefix=".v81-observation-",dir=OUT_DIR) as t:
  d=Path(t);ds=d/"projection.jsonl";rp=d/"projection.report.json";db=[];rb=[]
  for _ in (1,2):build_projection(ds,rp,OUTPUT_PROJECTION_CURATIONS);db.append(ds.read_bytes());rb.append(rp.read_bytes())
  p=json.loads(rb[0]);n=dict(p);n["output"]="<projection-output>";nb=(json.dumps(n,indent=2,sort_keys=True)+"\n").encode();return {"dataset_equal":db[0]==db[1],"dataset_sha256":hashlib.sha256(db[0]).hexdigest(),"report_equal":rb[0]==rb[1],"report_normalized_sha256":hashlib.sha256(nb).hexdigest(),"rows":db[0].count(b"\n"),"eval_fact_count":p["eval_fact_count"]}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v80-projection-",dir=OUT_DIR) as t:
  d=Path(t);base=d/"v80.jsonl";br=d/"v80.report.json";build_projection(base,br,PRIOR_PROJECTION_CURATIONS);rows=read_jsonl(base)
  if len(rows)!=536 or file_sha256(base)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v80 projection drift")
  by={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)}
  if {f:by[f][0] for f in EXPECTED_SELECTION}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v81 candidate drift")
 doc=json.loads(SOURCE.read_text());audits=[];curations=[]
 for ai,s in enumerate(SPECS,1):
  active=by[s["fact_id"]][1];ev=evidence(doc,s["marker"])
  if active["document_sha256"]!=doc["document_sha256"]:raise ValueError(f"{s['fact_id']}: quick-release lineage drift")
  a={"active_answer":active["answer"],"active_index":s["active_index"],"active_question":active["question"],"audit_index":ai,"decision":s["decision"],"document_sha256":active["document_sha256"],"fact_id":s["fact_id"],"projection_lineage":{"active_index":s["active_index"],"baseline_rows":536,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"rope365_quick_release_reaudit","reviewed_at":REVIEWED_AT,"reviewer":REVIEWER,"risk_features":CORE.risk_features(active),"schema":"context-merit-audit-v81","source":doc["source"],"source_document":str(SOURCE.relative_to(ROOT)),"source_document_file_sha256":file_sha256(SOURCE),"source_support":"normalized_extractive" if s["decision"]=="keep" else "manual_paraphrase","support_evidence":ev,"support_evidence_sha256":text_sha256(ev),"url":doc["url"]}
  if s["decision"]=="edit":
   a.update(edited_answer=s["answer"],edited_question=s["question"],paraphrase_rationale=s["reason"]);curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":ev,"evidence_url":doc["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":s["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":REVIEWED_AT,"reviewer":REVIEWER,"source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append(a)
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observation()
 if not o["dataset_equal"] or not o["report_equal"] or o["rows"]!=536 or o["eval_fact_count"]!=612:raise ValueError("v81 deterministic projection drift")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["dataset_sha256"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("v81 output hash drift")
 report={"active_baseline":{"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]},"audit":{"by_decision":{"edit":1,"keep":3},"by_reason":{s["reason_code"]:1 for s in SPECS},"path":str(AUDIT.relative_to(ROOT)),"rows":4,"sha256":file_sha256(AUDIT)},"frozen_prior_decision_artifacts":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()],"isolated_build_projection":{"active_after_context_merit_v80":498,"active_after_this_tranche":498,"automated_projection_runs":2,"build_script":"build_curated_qa.py","determinism_comparison_scope":"identical inputs, curation chain, and output/report paths","new_drops_applied":0,"new_edits_applied":1,"output_rows":o["rows"],"output_sha256":o["dataset_sha256"],"prior_pending_addition_fact_ids_preserved":36,"projection_report_normalized_sha256":o["report_normalized_sha256"],"repeat_dataset_byte_identical":o["dataset_equal"],"repeat_projection_report_byte_identical":o["report_equal"],"reviewed_keep_fact_ids_preserved":3,"sealed_eval_fact_count_reported_by_tooling":o["eval_fact_count"],"unexpected_fact_ids":0},"new_pending_curation":{"by_action":{"edit":1},"decisions":1,"edit_support_types":{"extractive":0,"manual_paraphrase":1},"path":str(CURATION.relative_to(ROOT)),"sha256":file_sha256(CURATION)},"projected_baseline":PROJECTED_SELECTION_BASELINE,"schema":"context-merit-audit-report-v81","sealed_evaluation_policy":{"automated_collision_tool":"build_curated_qa.py","automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-id collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False},"selection":{"active_rows":536,"projected_baseline":PROJECTED_SELECTION_BASELINE,"ranking":{"candidate_rule":"all remaining direct Q&A from one fully read Rope365 quick-release lesson","score":"manual technique accuracy, safety, and answer grammar review","tie_break":"active projection order"},"rows_selected":4}};REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
