#!/usr/bin/env python3
"""Complete three qualities, traditions, and finishing train-only answers."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V379=DATA/"manual_reviews/context_merit_audit_v379";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V379),str(V290)]
import build_context_merit_audit_v379 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v380.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v380.jsonl";REPORT=OUT_DIR/"report_context_merit_v380.json";BASELINE_ROWS=531;BASELINE_SHA256="22f0064379b3cabe69856d2bb75a057fc006b1dd147638c9ec57f25f0b222073";EXPECTED_OUTPUT_SHA256="e46dbbb15b0be545bc60c37acbb1270a1fa7a6f2c0d6beb670d5f7dc2e17b899"
EXPECTED_CAPACITY_BEFORE={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71};EXPECTED_CAPACITY_AFTER={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V379/"context_merit_audit_v379.jsonl",V379/"pending_curation_context_merit_v379.jsonl",V379/"report_context_merit_v379.json")
SPECS=(
 {"fact_id":"fact-92d4e2a73e122f71c8de","active_index":15,"expected_question":"According to Randa Mai, what qualities should a rope artist bring to bondage?","expected_answer":"an affinity for rope, attentive communication, and accountability for the bound person’s physical and mental state","question":"According to Randa Mai, what qualities should a rope artist bring to bondage?","answer":"Randa Mai says a rope artist should have an affinity for rope, communicate attentively, and remain accountable for the bound person’s physical and mental state.","reason_code":"complete_randa_mai_artist_qualities","reason":"The replacement converts the supported three-part list into a complete attributed answer without adding a new quality."},
 {"fact_id":"fact-ee52b59e3f910a4b414f","active_index":23,"expected_question":"According to Ugo’s account in “Kinbaku – An Evolving Era – Part 3,” which three traditions were often forcibly associated with kinbaku in the West?","expected_answer":"the tea ceremony, calligraphy and Budō","question":"According to Ugo’s account in “Kinbaku – An Evolving Era – Part 3,” which three traditions were often forcibly associated with kinbaku in the West?","answer":"They were the tea ceremony, calligraphy, and Budō.","reason_code":"complete_ugo_associated_traditions","reason":"The replacement makes the source-supported three-item list a direct complete answer."},
 {"fact_id":"fact-f83681aa3a6424064b30","active_index":32,"expected_question":"For which type of rope does Rope365 list melting as an end-finishing method?","expected_answer":"synthetic only","question":"For which type of rope does Rope365 list melting as an end-finishing method?","answer":"Rope365 lists melting only for synthetic rope.","reason_code":"complete_synthetic_melting_finish","reason":"The replacement turns the source’s parenthetical limitation into a grammatical answer with explicit attribution."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v379 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v380-input";d.mkdir(parents=True,exist_ok=True);base=d/"v379.jsonl";build_baseline(base,d/"v379.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v380-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v380-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v379.jsonl";build_baseline(base,d/"v379.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v380","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"qualities_traditions_finishing_answer_completeness_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v380","schema":"context-merit-audit-v380","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(531,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v380","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
