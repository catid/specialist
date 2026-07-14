#!/usr/bin/env python3
"""Audit four more consent and communication rows from one Rope365 source."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V75_DIR=DATA/"manual_reviews/context_merit_audit_v75";sys.path[:0]=[str(ROOT),str(V75_DIR)]
import build_context_merit_audit_v75 as previous
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v76.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v76.jsonl";REPORT=OUT_DIR/"report_context_merit_v76.json";REVIEWER="codex-context-merit-audit-v76";REVIEWED_AT="2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET=previous.ACTIVE_DATASET;ACTIVE_REPORT=previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION=previous.QUALITY_MERIT_CURATION;TASUKI_CURATION=previous.TASUKI_CURATION;CORE=previous.CORE;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl
CONTEXT_CURATIONS=previous.OUTPUT_CONTEXT_CURATIONS;PRIOR_PROJECTION_CURATIONS=previous.OUTPUT_PROJECTION_CURATIONS;OUTPUT_CONTEXT_CURATIONS=(*CONTEXT_CURATIONS,CURATION);OUTPUT_PROJECTION_CURATIONS=(*PRIOR_PROJECTION_CURATIONS,CURATION);SOURCE=previous.SOURCE
SPECS=(
 {"fact_id":"fact-961251799c9b231ff6c1","active_index":232,"marker":"If power dynamics (such as D/s, teacher/student or financial dynamics) are influencing your communication","decision":"edit","question":"What does Rope365 recommend when power dynamics may affect communication?","answer":"Work harder to establish clear communication and confirm consent.","reason_code":"repair_power_dynamic_communication_guidance","reason":"The revised Q&A turns the source’s ungrammatical ‘work extra length’ phrase into a direct instruction while preserving clear communication and consent confirmation."},
 {"fact_id":"fact-2cb7ea98d3938f5ce55e","active_index":349,"marker":"Monitor your partner: their words, their breathing, their whole body","decision":"edit","question":"What three kinds of signals does Rope365 recommend monitoring while tying a partner?","answer":"the partner’s words, breathing, and whole-body responses","reason_code":"clarify_whole_body_monitoring_answer","reason":"The revised answer replaces the vague phrase ‘their whole body’ with whole-body responses while preserving the source’s three monitoring channels."},
 {"fact_id":"fact-4cc9e4229c016704eeee","active_index":383,"marker":"It is best to discuss this beforehand, when emotions aren’t high","decision":"keep","reason_code":"retain_clear_aftercare_timing_guidance","reason":"The existing Q&A clearly and concisely states when Rope365 recommends discussing aftercare needs."},
 {"fact_id":"fact-63c654d8cdad2602da36","active_index":449,"marker":"An allowlist gives a better result than a blocklist","decision":"keep","reason_code":"retain_clear_allowlist_negotiation_guidance","reason":"The existing Q&A accurately preserves Rope365’s concise comparison between allowlist and blocklist negotiation."},
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES={s["fact_id"]:s["active_index"] for s in SPECS};PROJECTED_SELECTION_BASELINE={"description":"isolated corrected training projection through context-merit v75","direct_rows_without_prior_curation":135,"rope365_communication_rows_selected":4,"rows":536,"sha256":"555a70967b3de5462174995928569a0795df5b4335ab7584934998ea35e1a619"};EXPECTED_OUTPUT_SHA256="fc0ba9ef386d0c63ee98c9641591e05d4d09c9b16a6f6d2002b155be83a651cf"
def build_projection(o,r,c):previous.build_projection(o,r,c)
def prior_decision_artifacts():
 out=[]
 for v in range(1,76):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def source_evidence(doc,marker):
 matches=[line for line in doc["text"].splitlines() if marker in line]
 if len(matches)!=1:raise ValueError(f"evidence marker drift: {marker}")
 return matches[0]
def deterministic_projection():
 with tempfile.TemporaryDirectory(prefix=".v76-observation-",dir=OUT_DIR) as t:
  d=Path(t);ds=d/"projection.jsonl";rp=d/"projection.report.json";datasets=[];reports=[]
  for _ in (1,2):build_projection(ds,rp,OUTPUT_PROJECTION_CURATIONS);datasets.append(ds.read_bytes());reports.append(rp.read_bytes())
  parsed=json.loads(reports[0]);normalized=dict(parsed);normalized["output"]="<projection-output>";nb=(json.dumps(normalized,indent=2,sort_keys=True)+"\n").encode();return {"dataset_equal":datasets[0]==datasets[1],"dataset_sha256":hashlib.sha256(datasets[0]).hexdigest(),"report_equal":reports[0]==reports[1],"report_normalized_sha256":hashlib.sha256(nb).hexdigest(),"rows":datasets[0].count(b"\n"),"eval_fact_count":parsed["eval_fact_count"]}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v75-projection-",dir=OUT_DIR) as t:
  d=Path(t);base=d/"v75.jsonl";br=d/"v75.report.json";build_projection(base,br,PRIOR_PROJECTION_CURATIONS);rows=read_jsonl(base)
  if len(rows)!=536 or file_sha256(base)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v75 projection drift")
  by={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)}
  if {f:by[f][0] for f in EXPECTED_SELECTION}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v76 candidate drift")
 doc=json.loads(SOURCE.read_text())
 if doc.get("document_sha256")!="ba41f96db0578f593930a21a579f6a30f3658b100da8390fea2edbdf5b4abb3d":raise ValueError("Rope365 communication document drift")
 audits=[];curations=[]
 for ai,s in enumerate(SPECS,1):
  active=by[s["fact_id"]][1];evidence=source_evidence(doc,s["marker"])
  audit={"active_answer":active["answer"],"active_index":s["active_index"],"active_question":active["question"],"audit_index":ai,"decision":s["decision"],"document_sha256":active["document_sha256"],"fact_id":s["fact_id"],"projection_lineage":{"active_index":s["active_index"],"baseline_rows":536,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"rope365_consent_and_monitoring_reaudit","reviewed_at":REVIEWED_AT,"reviewer":REVIEWER,"risk_features":CORE.risk_features(active),"schema":"context-merit-audit-v76","source":doc["source"],"source_document":str(SOURCE.relative_to(ROOT)),"source_document_file_sha256":file_sha256(SOURCE),"source_support":"normalized_extractive" if s["decision"]=="keep" else "manual_paraphrase","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":doc["url"]}
  if s["decision"]=="edit":
   audit.update(edited_answer=s["answer"],edited_question=s["question"],paraphrase_rationale=s["reason"]);curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":doc["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":s["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":REVIEWED_AT,"reviewer":REVIEWER,"source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append(audit)
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);observed=deterministic_projection()
 if not observed["dataset_equal"] or not observed["report_equal"] or observed["rows"]!=536 or observed["eval_fact_count"]!=612:raise ValueError("v76 deterministic projection drift")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and observed["dataset_sha256"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("v76 output hash drift")
 report={"active_baseline":{"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]},"audit":{"by_decision":{"edit":2,"keep":2},"by_reason":{s["reason_code"]:1 for s in SPECS},"path":str(AUDIT.relative_to(ROOT)),"rows":4,"sha256":file_sha256(AUDIT)},"frozen_prior_decision_artifacts":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()],"isolated_build_projection":{"active_after_context_merit_v75":498,"active_after_this_tranche":498,"automated_projection_runs":2,"build_script":"build_curated_qa.py","determinism_comparison_scope":"identical inputs, curation chain, and output/report paths","new_drops_applied":0,"new_edits_applied":2,"output_rows":observed["rows"],"output_sha256":observed["dataset_sha256"],"prior_pending_addition_fact_ids_preserved":36,"projection_report_normalized_sha256":observed["report_normalized_sha256"],"repeat_dataset_byte_identical":observed["dataset_equal"],"repeat_projection_report_byte_identical":observed["report_equal"],"reviewed_keep_fact_ids_preserved":2,"sealed_eval_fact_count_reported_by_tooling":observed["eval_fact_count"],"unexpected_fact_ids":0},"new_pending_curation":{"by_action":{"edit":2},"decisions":2,"edit_support_types":{"extractive":0,"manual_paraphrase":2},"path":str(CURATION.relative_to(ROOT)),"sha256":file_sha256(CURATION)},"projected_baseline":PROJECTED_SELECTION_BASELINE,"schema":"context-merit-audit-report-v76","sealed_evaluation_policy":{"automated_collision_tool":"build_curated_qa.py","automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-id collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False},"selection":{"active_rows":536,"projected_baseline":PROJECTED_SELECTION_BASELINE,"ranking":{"candidate_rule":"four more consent, monitoring, and aftercare Q&A from the fully read Rope365 communication page","score":"manual actionability and standalone clarity review","tie_break":"active projection order"},"rows_selected":4}};REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
