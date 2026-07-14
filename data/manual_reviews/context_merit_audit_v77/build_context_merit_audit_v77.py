#!/usr/bin/env python3
"""Audit four jute hygiene rows from one fully reviewed Anatomie Studio article."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V76_DIR=DATA/"manual_reviews/context_merit_audit_v76";sys.path[:0]=[str(ROOT),str(V76_DIR)]
import build_context_merit_audit_v76 as previous
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v77.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v77.jsonl";REPORT=OUT_DIR/"report_context_merit_v77.json";REVIEWER="codex-context-merit-audit-v77";REVIEWED_AT="2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET=previous.ACTIVE_DATASET;ACTIVE_REPORT=previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION=previous.QUALITY_MERIT_CURATION;TASUKI_CURATION=previous.TASUKI_CURATION;CORE=previous.CORE;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl
CONTEXT_CURATIONS=previous.OUTPUT_CONTEXT_CURATIONS;PRIOR_PROJECTION_CURATIONS=previous.OUTPUT_PROJECTION_CURATIONS;OUTPUT_CONTEXT_CURATIONS=(*CONTEXT_CURATIONS,CURATION);OUTPUT_PROJECTION_CURATIONS=(*PRIOR_PROJECTION_CURATIONS,CURATION);SOURCE=DATA/"raw/anatomiestudio_144932682af9c846.json"
SPECS=(
 {"fact_id":"fact-947f495396a7edf09414","active_index":59,"marker":"You could mark the ends with a certain colour","decision":"edit","question":"How does Anatomie Studio suggest visually distinguishing a designated rope?","answer":"mark its ends with a distinct color","reason_code":"attribute_and_naturalize_designated_rope_marking","reason":"The revised Q&A names the source and replaces the vague ‘a certain colour’ phrase with a concise, equivalent marking instruction."},
 {"fact_id":"fact-37e7995e1842fb2c3d1c","active_index":172,"marker":"use a different rope (or even a cloth) for the parts of ties that are more likely to get bodily fluids on them","decision":"edit","question":"What does Anatomie Studio suggest using for parts of a tie likely to contact bodily fluids?","answer":"a different rope or even a cloth","reason_code":"attribute_bodily_fluid_contact_hygiene_option","reason":"The revised question identifies the source and the answer removes unnecessary parentheses while preserving both suggested barriers."},
 {"fact_id":"fact-6a31ec21b3340a7860de","active_index":267,"marker":"perhaps the most practical one, is to assign a specific rope to a specific person (and role)","decision":"edit","question":"What practical hygiene strategy does Anatomie Studio suggest when jute rope is difficult to clean?","answer":"Assign a specific rope to a specific person and role.","reason_code":"frame_person_and_role_assignment_as_hygiene_strategy","reason":"The revised Q&A identifies the source and turns a vague ‘option’ question into a clear, actionable hygiene strategy."},
 {"fact_id":"fact-59847c1427eaee28d7de","active_index":436,"marker":"intensive wash and spin cycle we wouldn’t recommend it","decision":"keep","reason_code":"retain_clear_jute_machine_wash_warning","reason":"The existing Q&A clearly attributes and states the intensive wash-and-spin method Anatomie Studio advises against for jute."},
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES={s["fact_id"]:s["active_index"] for s in SPECS};PROJECTED_SELECTION_BASELINE={"description":"isolated corrected training projection through context-merit v76","direct_rows_without_prior_curation":133,"anatomiestudio_hygiene_rows_selected":4,"rows":536,"sha256":"fc0ba9ef386d0c63ee98c9641591e05d4d09c9b16a6f6d2002b155be83a651cf"};EXPECTED_OUTPUT_SHA256="5b39cd99de489938ec97e3eb58ef4241a42f15ae86ea13ec4000b6e8c249080f"
def build_projection(o,r,c):previous.build_projection(o,r,c)
def prior_decision_artifacts():
 out=[]
 for v in range(1,77):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def source_evidence(doc,marker):
 matches=[line for line in doc["text"].splitlines() if marker in line]
 if len(matches)!=1:raise ValueError(f"evidence marker drift: {marker}")
 return matches[0]
def deterministic_projection():
 with tempfile.TemporaryDirectory(prefix=".v77-observation-",dir=OUT_DIR) as t:
  d=Path(t);ds=d/"projection.jsonl";rp=d/"projection.report.json";datasets=[];reports=[]
  for _ in (1,2):build_projection(ds,rp,OUTPUT_PROJECTION_CURATIONS);datasets.append(ds.read_bytes());reports.append(rp.read_bytes())
  parsed=json.loads(reports[0]);normalized=dict(parsed);normalized["output"]="<projection-output>";nb=(json.dumps(normalized,indent=2,sort_keys=True)+"\n").encode();return {"dataset_equal":datasets[0]==datasets[1],"dataset_sha256":hashlib.sha256(datasets[0]).hexdigest(),"report_equal":reports[0]==reports[1],"report_normalized_sha256":hashlib.sha256(nb).hexdigest(),"rows":datasets[0].count(b"\n"),"eval_fact_count":parsed["eval_fact_count"]}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v76-projection-",dir=OUT_DIR) as t:
  d=Path(t);base=d/"v76.jsonl";br=d/"v76.report.json";build_projection(base,br,PRIOR_PROJECTION_CURATIONS);rows=read_jsonl(base)
  if len(rows)!=536 or file_sha256(base)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v76 projection drift")
  by={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)}
  if {f:by[f][0] for f in EXPECTED_SELECTION}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v77 candidate drift")
 doc=json.loads(SOURCE.read_text())
 audits=[];curations=[]
 for ai,s in enumerate(SPECS,1):
  active=by[s["fact_id"]][1];evidence=source_evidence(doc,s["marker"])
  if active["document_sha256"]!="b316fd1a708dcb7688d44826496dd58b8c06d20b9c01d8daa73e99e7704abc53":raise ValueError(f"{s['fact_id']}: Anatomie document lineage drift")
  audit={"active_answer":active["answer"],"active_index":s["active_index"],"active_question":active["question"],"audit_index":ai,"decision":s["decision"],"document_sha256":active["document_sha256"],"fact_id":s["fact_id"],"projection_lineage":{"active_index":s["active_index"],"baseline_rows":536,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"anatomiestudio_jute_hygiene_reaudit","reviewed_at":REVIEWED_AT,"reviewer":REVIEWER,"risk_features":CORE.risk_features(active),"schema":"context-merit-audit-v77","source":doc["source"],"source_document":str(SOURCE.relative_to(ROOT)),"source_document_file_sha256":file_sha256(SOURCE),"source_support":"normalized_extractive" if s["decision"]=="keep" else "manual_paraphrase","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":doc["url"]}
  if s["decision"]=="edit":
   audit.update(edited_answer=s["answer"],edited_question=s["question"],paraphrase_rationale=s["reason"]);curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":doc["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":s["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":REVIEWED_AT,"reviewer":REVIEWER,"source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append(audit)
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);observed=deterministic_projection()
 if not observed["dataset_equal"] or not observed["report_equal"] or observed["rows"]!=536 or observed["eval_fact_count"]!=612:raise ValueError("v77 deterministic projection drift")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and observed["dataset_sha256"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("v77 output hash drift")
 report={"active_baseline":{"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]},"audit":{"by_decision":{"edit":3,"keep":1},"by_reason":{s["reason_code"]:1 for s in SPECS},"path":str(AUDIT.relative_to(ROOT)),"rows":4,"sha256":file_sha256(AUDIT)},"frozen_prior_decision_artifacts":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()],"isolated_build_projection":{"active_after_context_merit_v76":498,"active_after_this_tranche":498,"automated_projection_runs":2,"build_script":"build_curated_qa.py","determinism_comparison_scope":"identical inputs, curation chain, and output/report paths","new_drops_applied":0,"new_edits_applied":3,"output_rows":observed["rows"],"output_sha256":observed["dataset_sha256"],"prior_pending_addition_fact_ids_preserved":36,"projection_report_normalized_sha256":observed["report_normalized_sha256"],"repeat_dataset_byte_identical":observed["dataset_equal"],"repeat_projection_report_byte_identical":observed["report_equal"],"reviewed_keep_fact_ids_preserved":1,"sealed_eval_fact_count_reported_by_tooling":observed["eval_fact_count"],"unexpected_fact_ids":0},"new_pending_curation":{"by_action":{"edit":3},"decisions":3,"edit_support_types":{"extractive":0,"manual_paraphrase":3},"path":str(CURATION.relative_to(ROOT)),"sha256":file_sha256(CURATION)},"projected_baseline":PROJECTED_SELECTION_BASELINE,"schema":"context-merit-audit-report-v77","sealed_evaluation_policy":{"automated_collision_tool":"build_curated_qa.py","automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-id collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False},"selection":{"active_rows":536,"projected_baseline":PROJECTED_SELECTION_BASELINE,"ranking":{"candidate_rule":"four practical hygiene Q&A from one fully reviewed Anatomie Studio jute-care article","score":"manual attribution, completeness, and actionable hygiene value review","tie_break":"active projection order"},"rows_selected":4}};REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
