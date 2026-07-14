#!/usr/bin/env python3
"""Audit four remaining Rope365 incident-prevention and response rows."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V81_DIR=DATA/"manual_reviews/context_merit_audit_v81";sys.path[:0]=[str(ROOT),str(V81_DIR)]
import build_context_merit_audit_v81 as previous
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v82.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v82.jsonl";REPORT=OUT_DIR/"report_context_merit_v82.json";REVIEWER="codex-context-merit-audit-v82";REVIEWED_AT="2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET=previous.ACTIVE_DATASET;ACTIVE_REPORT=previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION=previous.QUALITY_MERIT_CURATION;TASUKI_CURATION=previous.TASUKI_CURATION;CORE=previous.CORE;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl
CONTEXT_CURATIONS=previous.OUTPUT_CONTEXT_CURATIONS;PRIOR_PROJECTION_CURATIONS=previous.OUTPUT_PROJECTION_CURATIONS;OUTPUT_CONTEXT_CURATIONS=(*CONTEXT_CURATIONS,CURATION);OUTPUT_PROJECTION_CURATIONS=(*PRIOR_PROJECTION_CURATIONS,CURATION);SOURCE=DATA/"raw/rope_resources_v1/rope365__7b5d548036392d65fec7.json"
SPECS=(
 {"fact_id":"fact-52aaf228bd4fb0739e29","active_index":86,"marker":"Clean up your play space to prevent tripping hazards","decision":"keep","reason_code":"retain_clear_play_space_cleanup_guidance","reason":"The existing Q&A directly states Rope365’s practical method for reducing tripping hazards."},
 {"fact_id":"fact-495c6aa066f884214063","active_index":224,"marker":"Before trying a new technique, take the time to research the specific risks","decision":"keep","reason_code":"retain_new_technique_risk_research_guidance","reason":"The existing Q&A clearly preserves the source’s pre-technique risk-research instruction."},
 {"fact_id":"fact-aa51767b03dd040b7b37","active_index":274,"marker":"supervise if you want to self-tie (known as spotter)","decision":"keep","reason_code":"retain_spotter_term_and_function","reason":"The existing Q&A concisely teaches both the spotter term and its self-tying supervision function."},
 {"fact_id":"fact-568e19d1c85f56b99ce1","active_index":338,"marker":"a solid stick like a marlinspike to your kit to help untie knots in case they become hard to untie","decision":"edit","question":"What tool does Rope365 suggest for loosening knots that become hard to untie?","answer":"a solid stick such as a marlinspike","reason_code":"attribute_and_naturalize_marlinspike_guidance","reason":"The revised Q&A identifies Rope365, asks directly about the tool’s function, and normalizes the example phrase without changing it."},
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES={s["fact_id"]:s["active_index"] for s in SPECS};PROJECTED_SELECTION_BASELINE={"description":"isolated corrected training projection through context-merit v81","direct_rows_without_prior_curation":124,"rope365_incident_rows_selected":4,"rows":536,"sha256":"c7eec4ce0778e3c1a720a0bcce24560826d28c2031476a54a1c1dee053b63af3"};EXPECTED_OUTPUT_SHA256="877401498fcd4ea756c1db45605d072893928c2751bae69605a68fd80ce009e1"
def build_projection(o,r,c):previous.build_projection(o,r,c)
def prior_decision_artifacts():
 out=[]
 for v in range(1,82):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def evidence(doc,marker):
 matches=[line for line in doc["text"].splitlines() if marker in line]
 if len(matches)!=1:raise ValueError(f"evidence drift: {marker}")
 return matches[0]
def observation():
 with tempfile.TemporaryDirectory(prefix=".v82-observation-",dir=OUT_DIR) as t:
  d=Path(t);ds=d/"projection.jsonl";rp=d/"projection.report.json";db=[];rb=[]
  for _ in (1,2):build_projection(ds,rp,OUTPUT_PROJECTION_CURATIONS);db.append(ds.read_bytes());rb.append(rp.read_bytes())
  p=json.loads(rb[0]);n=dict(p);n["output"]="<projection-output>";nb=(json.dumps(n,indent=2,sort_keys=True)+"\n").encode();return {"dataset_equal":db[0]==db[1],"dataset_sha256":hashlib.sha256(db[0]).hexdigest(),"report_equal":rb[0]==rb[1],"report_normalized_sha256":hashlib.sha256(nb).hexdigest(),"rows":db[0].count(b"\n"),"eval_fact_count":p["eval_fact_count"]}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v81-projection-",dir=OUT_DIR) as t:
  d=Path(t);base=d/"v81.jsonl";br=d/"v81.report.json";build_projection(base,br,PRIOR_PROJECTION_CURATIONS);rows=read_jsonl(base)
  if len(rows)!=536 or file_sha256(base)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v81 projection drift")
  by={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)}
  if {f:by[f][0] for f in EXPECTED_SELECTION}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v82 candidate drift")
 doc=json.loads(SOURCE.read_text());audits=[];curations=[]
 for ai,s in enumerate(SPECS,1):
  active=by[s["fact_id"]][1];ev=evidence(doc,s["marker"])
  if active["document_sha256"]!=doc["document_sha256"]:raise ValueError(f"{s['fact_id']}: safety lineage drift")
  a={"active_answer":active["answer"],"active_index":s["active_index"],"active_question":active["question"],"audit_index":ai,"decision":s["decision"],"document_sha256":active["document_sha256"],"fact_id":s["fact_id"],"projection_lineage":{"active_index":s["active_index"],"baseline_rows":536,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"rope365_incident_prevention_and_response_reaudit","reviewed_at":REVIEWED_AT,"reviewer":REVIEWER,"risk_features":CORE.risk_features(active),"schema":"context-merit-audit-v82","source":doc["source"],"source_document":str(SOURCE.relative_to(ROOT)),"source_document_file_sha256":file_sha256(SOURCE),"source_support":"normalized_extractive" if s["decision"]=="keep" else "manual_paraphrase","support_evidence":ev,"support_evidence_sha256":text_sha256(ev),"url":doc["url"]}
  if s["decision"]=="edit":
   a.update(edited_answer=s["answer"],edited_question=s["question"],paraphrase_rationale=s["reason"]);curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":ev,"evidence_url":doc["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":s["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":REVIEWED_AT,"reviewer":REVIEWER,"source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append(a)
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observation()
 if not o["dataset_equal"] or not o["report_equal"] or o["rows"]!=536 or o["eval_fact_count"]!=612:raise ValueError("v82 deterministic projection drift")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["dataset_sha256"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("v82 output hash drift")
 report={"active_baseline":{"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]},"audit":{"by_decision":{"edit":1,"keep":3},"by_reason":{s["reason_code"]:1 for s in SPECS},"path":str(AUDIT.relative_to(ROOT)),"rows":4,"sha256":file_sha256(AUDIT)},"frozen_prior_decision_artifacts":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()],"isolated_build_projection":{"active_after_context_merit_v81":498,"active_after_this_tranche":498,"automated_projection_runs":2,"build_script":"build_curated_qa.py","determinism_comparison_scope":"identical inputs, curation chain, and output/report paths","new_drops_applied":0,"new_edits_applied":1,"output_rows":o["rows"],"output_sha256":o["dataset_sha256"],"prior_pending_addition_fact_ids_preserved":36,"projection_report_normalized_sha256":o["report_normalized_sha256"],"repeat_dataset_byte_identical":o["dataset_equal"],"repeat_projection_report_byte_identical":o["report_equal"],"reviewed_keep_fact_ids_preserved":3,"sealed_eval_fact_count_reported_by_tooling":o["eval_fact_count"],"unexpected_fact_ids":0},"new_pending_curation":{"by_action":{"edit":1},"decisions":1,"edit_support_types":{"extractive":0,"manual_paraphrase":1},"path":str(CURATION.relative_to(ROOT)),"sha256":file_sha256(CURATION)},"projected_baseline":PROJECTED_SELECTION_BASELINE,"schema":"context-merit-audit-report-v82","sealed_evaluation_policy":{"automated_collision_tool":"build_curated_qa.py","automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-id collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False},"selection":{"active_rows":536,"projected_baseline":PROJECTED_SELECTION_BASELINE,"ranking":{"candidate_rule":"four remaining incident-prevention and response Q&A from the previously fully read Rope365 safety page","score":"manual safety, source attribution, and answer grammar review","tie_break":"active projection order"},"rows_selected":4}};REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
