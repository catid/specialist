#!/usr/bin/env python3
"""Supersede malformed v65 wording and make sealed/determinism claims exact v69."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V68_DIR=DATA/"manual_reviews/context_merit_audit_v68";sys.path[:0]=[str(ROOT),str(V68_DIR)]
import build_context_merit_audit_v68 as previous
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v69.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v69.jsonl";REPORT=OUT_DIR/"report_context_merit_v69.json";REVIEWER,REVIEWED_AT="codex-context-merit-audit-v69","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION;file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl
V65_DIR=DATA/"manual_reviews/context_merit_audit_v65";V65_CURATION=V65_DIR/"pending_curation_context_merit_v65.jsonl";SOURCE=DATA/"raw/anatomiestudio_144932682af9c846.json"
ORIGINAL_FACT_ID="fact-da15b630db4ec0ed79cf";DERIVED_FACT_ID="fact-4a509d1c5e33658857de";MALFORMED_QUESTION="After wet jute rope dries, how does Anatomie Studio say its tightened twist may make the rope feel?";CORRECTED_QUESTION="After wet jute rope dries, how may its tightened twist make the rope feel, according to Anatomie Studio?";ANSWER="spongy and springy"
PROJECTED_SELECTION_BASELINE={"description":"isolated cumulative training projection through context-merit v68","direct_rows_without_prior_curation":150,"rows":537,"sha256":"7140b838c0dd8b39a1f458ed23567982c5c2432e04ccdcd1e0d38255418e6d2d"}
EXPECTED_OUTPUT_SHA256="53ab9887a265e8d490849875dd98bf92ab33a88ce6f6c5e54ffb571c1d86d3ad"
CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,69));PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
OUTPUT_CONTEXT_CURATIONS=tuple(path for v,path in enumerate(CONTEXT_CURATIONS,1) if v!=65);OUTPUT_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*OUTPUT_CONTEXT_CURATIONS,CURATION)
def build_projection(o,r,c):previous.build_projection(o,r,c)
def prior_decision_artifacts():
 out=[]
 for v in range(1,69):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def corrected_curation_rows():
 rows=read_jsonl(V65_CURATION);by={r["fact_id"]:r for r in rows}
 if set(by)!={ORIGINAL_FACT_ID,"fact-069a861dbb2bea9e47ca","fact-f7e802bf0b2759290dc6"}:raise ValueError("v65 curation membership drift")
 corrected=[]
 for row in rows:
  item=dict(row)
  if item["fact_id"]==ORIGINAL_FACT_ID:item.update(question=CORRECTED_QUESTION,reason="The corrected question removes the ungrammatical ‘does … may make’ construction while preserving the source, mechanism, and exact answer.",reason_code="repair_v65_wet_jute_modal_grammar",reviewer=REVIEWER,reviewed_at=REVIEWED_AT)
  corrected.append(item)
 return corrected
def source_evidence():
 doc=json.loads(SOURCE.read_text());marker="making the ropes feel spongy and springy when they dry out";matches=[line for line in doc["text"].splitlines() if marker in line]
 if len(matches)!=1:raise ValueError("wet-jute evidence marker drift")
 return doc,matches[0]
def projection_observation():
 with tempfile.TemporaryDirectory(prefix=".v69-observation-",dir=OUT_DIR) as t:
  d=Path(t);datasets=[];reports=[];ds=d/"projection.jsonl";rp=d/"projection.report.json"
  for n in (1,2):
   build_projection(ds,rp,OUTPUT_PROJECTION_CURATIONS);datasets.append(ds.read_bytes());reports.append(rp.read_bytes())
  parsed=json.loads(reports[0]);normalized=dict(parsed);normalized["output"]="<projection-output>";normalized_bytes=(json.dumps(normalized,indent=2,sort_keys=True)+"\n").encode();return {"dataset_bytes_identical":datasets[0]==datasets[1],"dataset_sha256":sha_bytes(datasets[0]),"output_rows":datasets[0].count(b"\n"),"projection_report_bytes_identical":reports[0]==reports[1],"projection_report_normalized_sha256":sha_bytes(normalized_bytes),"sealed_eval_fact_count_reported_by_tooling":parsed["eval_fact_count"]}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v68-baseline-",dir=OUT_DIR) as t:
  ds=Path(t)/"v68.jsonl";rp=Path(t)/"v68.report.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS);rows=read_jsonl(ds)
  if len(rows)!=537 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v68 projection drift")
  matches=[(i,r) for i,r in enumerate(rows,1) if r.get("curation",{}).get("original_fact_id")==ORIGINAL_FACT_ID]
  if len(matches)!=1 or matches[0][1]["fact_id"]!=DERIVED_FACT_ID or (matches[0][1]["question"],matches[0][1]["answer"])!=(MALFORMED_QUESTION,ANSWER):raise ValueError("v65 derived row drift")
  active_index=matches[0][0]
 write_jsonl(CURATION,corrected_curation_rows());doc,evidence=source_evidence()
 audit={"active_answer":ANSWER,"active_index":active_index,"active_question":MALFORMED_QUESTION,"audit_index":1,"decision":"edit","document_sha256":"b316fd1a708dcb7688d44826496dd58b8c06d20b9c01d8daa73e99e7704abc53","edited_answer":ANSWER,"edited_question":CORRECTED_QUESTION,"fact_id":ORIGINAL_FACT_ID,"projection_lineage":{"active_index":active_index,"baseline_rows":537,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"derived_fact_id":DERIVED_FACT_ID,"supersedes_context_merit_version":65},"reason":"The modal construction in the v65 question is ungrammatical; the corrected form keeps the same supported care fact.","reason_code":"repair_v65_wet_jute_modal_grammar","review_pass":"independent_audit_fidelity_repair","reviewed_at":REVIEWED_AT,"reviewer":REVIEWER,"schema":"context-merit-audit-v69","source":doc["source"],"source_document":str(SOURCE.relative_to(ROOT)),"source_document_file_sha256":file_sha256(SOURCE),"source_support":"normalized_extractive","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":doc["url"]};write_jsonl(AUDIT,[audit])
 observed=projection_observation()
 if not observed["dataset_bytes_identical"] or not observed["projection_report_bytes_identical"]:raise ValueError("projection determinism drift")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and observed["dataset_sha256"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("v69 output hash drift")
 report={"active_baseline":{"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]},"audit":{"by_decision":{"edit":1},"by_reason":{"repair_v65_wet_jute_modal_grammar":1},"path":str(AUDIT.relative_to(ROOT)),"rows":1,"sha256":file_sha256(AUDIT)},"frozen_prior_decision_artifacts":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()],"isolated_build_projection":{"active_after_context_merit_v68":499,"active_after_this_tranche":499,"automated_projection_runs":2,"build_script":"build_curated_qa.py","determinism_comparison_scope":"identical inputs, curation chain, and output/report paths","new_drops_applied":0,"new_edits_applied":1,"output_rows":observed["output_rows"],"output_sha256":observed["dataset_sha256"],"prior_pending_addition_fact_ids_preserved":36,"projection_report_normalized_sha256":observed["projection_report_normalized_sha256"],"repeat_dataset_byte_identical":observed["dataset_bytes_identical"],"repeat_projection_report_byte_identical":observed["projection_report_bytes_identical"],"reviewed_keep_fact_ids_preserved":0,"sealed_eval_fact_count_reported_by_tooling":observed["sealed_eval_fact_count_reported_by_tooling"],"unexpected_fact_ids":0},"new_pending_curation":{"by_action":{"edit":3},"carried_v65_decisions":2,"decisions":3,"edit_support_types":{"extractive":3,"manual_paraphrase":0},"path":str(CURATION.relative_to(ROOT)),"sha256":file_sha256(CURATION),"superseded_path":str(V65_CURATION.relative_to(ROOT)),"superseded_path_excluded_from_v69_output_projection":True},"projected_baseline":PROJECTED_SELECTION_BASELINE,"schema":"context-merit-audit-report-v69","sealed_evaluation_policy":{"automated_collision_tool":"build_curated_qa.py","automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-id collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False},"supersession":{"corrected_original_fact_id":ORIGINAL_FACT_ID,"malformed_derived_fact_id":DERIVED_FACT_ID,"reason":"v65 bundled three decisions, so v69 replaces that decision file in the isolated curation chain, corrects one decision, and carries the other two semantically unchanged."}};REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
