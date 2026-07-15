#!/usr/bin/env python3
"""Integrate the fourteenth distinct-document resource-access tranche."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; DATA=ROOT/"data"
V302=DATA/"manual_reviews/context_merit_audit_v302"; V290=DATA/"manual_reviews/context_merit_audit_v290"; ADD=DATA/"manual_reviews/resource_access_additions_v14"
sys.path[:0]=[str(ROOT),str(V302),str(V290),str(ADD)]
import build_context_merit_audit_v302 as previous
import build_context_merit_audit_v290 as core
import build_resource_access_additions_v14 as additions_builder
OUT_DIR=Path(__file__).resolve().parent; AUDIT=OUT_DIR/"context_merit_audit_v303.jsonl"; CURATION=OUT_DIR/"pending_curation_context_merit_v303.jsonl"; REPORT=OUT_DIR/"report_context_merit_v303.json"; ADDITIONS=additions_builder.OUTPUT
EXPECTED_ADDITIONS_SHA256="3c7b6f7b54e81626129ca4580f9d9f6a3a7185b38846701cf6214eb3baef1b63"; EXPECTED_OUTPUT_SHA256="8ace985d0a66db9638cd303a3fb47a271a367365094b3b6b4ec401f1d7369401"
BASELINE_ROWS=531; BASELINE_SHA256="e71677ffdf831747d3e5f5287fce3812ed0a604b6eba114d28e5c0b5125e6a0e"
EXPECTED_CAPACITY={"before":{"conflict_units":238,"equipment_material":22,"resources_general":79,"safety_consent":79,"technique":58},"after":{"conflict_units":241,"equipment_material":22,"resources_general":81,"safety_consent":79,"technique":59}}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST; file_sha256=previous.file_sha256; text_sha256=previous.text_sha256; read_jsonl=previous.read_jsonl; write_jsonl=previous.write_jsonl; conservative_capacity=previous.conservative_capacity; portable=previous.portable
PRIOR=(V302/"context_merit_audit_v302.jsonl",V302/"pending_curation_context_merit_v302.jsonl",V302/"report_context_merit_v302.json")
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256): raise ValueError("v302 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v303-input"; d.mkdir(parents=True,exist_ok=True); base=d/"v302.jsonl"; build_baseline(base,d/"v302.report.json"); core.build_projection_with_inputs(out,report,(),(base,ADDITIONS))
def observe():
 with tempfile.TemporaryDirectory(prefix=".v303-observe-",dir=OUT_DIR) as t:
  d=Path(t); base=d/"base.jsonl"; build_baseline(base,d/"base.report.json"); before=read_jsonl(base); out=d/"out.jsonl"; report=d/"out.report.json"; datasets=[]; reports=[]
  for _ in (1,2): build_projection(out,report); datasets.append(out.read_bytes()); reports.append(report.read_bytes())
  rows=read_jsonl(out); return {"rows":len(rows),"sha":hashlib.sha256(datasets[0]).hexdigest(),"eval":json.loads(reports[0])["eval_fact_count"],"dataset_equal":datasets[0]==datasets[1],"report_equal":reports[0]==reports[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True); additions_builder.main()
 if file_sha256(ADDITIONS)!=EXPECTED_ADDITIONS_SHA256: raise ValueError("addition drift")
 audits=[]
 for i,r in enumerate(read_jsonl(ADDITIONS),1):
  p=ROOT/r["source_lineage"]["raw_document"]; d=json.loads(p.read_text())
  if (r["url"],r["document_sha256"])!=(d["url"],d["document_sha256"]) or not all(x in d["text"] for x in r["evidence"].splitlines()): raise ValueError("lineage drift")
  audits.append({"audit_index":i,"decision":"add","document_sha256":r["document_sha256"],"fact_id":r["fact_id"],"proposed_answer":r["answer"],"proposed_question":r["question"],"reason":r["paraphrase_rationale"],"reason_code":f"add_distinct_{r['topic']}_fact","review_pass":"distinct_document_resource_access_additions","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v303","schema":"context-merit-audit-v303","source":r["source"],"source_document":portable(p),"source_document_file_sha256":file_sha256(p),"source_support":"manual_paraphrase","support_evidence":r["evidence"],"support_evidence_sha256":text_sha256(r["evidence"]),"url":r["url"]})
 write_jsonl(AUDIT,audits); write_jsonl(CURATION,[]); o=observe()
 if not o["dataset_equal"] or not o["report_equal"] or (o["rows"],o["eval"])!=(534,612) or o["before"]!=EXPECTED_CAPACITY["before"] or o["after"]!=EXPECTED_CAPACITY["after"]: raise ValueError(f"projection drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256: raise ValueError("output drift")
 delta={k:o["after"][k]-o["before"][k] for k in o["before"]}
 report={"addition_artifact":{"path":portable(ADDITIONS),"rows":3,"sha256":file_sha256(ADDITIONS)},"audit":{"by_decision":{"add":3,"drop":0,"edit":0,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":delta,"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":3,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["dataset_equal"],"repeat_projection_report_byte_identical":o["report_equal"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":0,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v303","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}}
 REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
