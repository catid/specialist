#!/usr/bin/env python3
"""Integrate year-attributed Tethered Together policy facts."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V309=DATA/"manual_reviews/context_merit_audit_v309";V290=DATA/"manual_reviews/context_merit_audit_v290";ADD=DATA/"manual_reviews/tethered_policy_additions_v21";sys.path[:0]=[str(ROOT),str(V309),str(V290),str(ADD)]
import build_context_merit_audit_v309 as previous
import build_context_merit_audit_v290 as core
import build_tethered_policy_additions_v21 as additions_builder
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v310.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v310.jsonl";REPORT=OUT_DIR/"report_context_merit_v310.json";ADDITIONS=additions_builder.OUTPUT;EXPECTED_ADDITIONS_SHA256="a270e1943b4a3d5d9cdf0f03248054c0a6b9584dfb621794a363e1181e5944af";EXPECTED_OUTPUT_SHA256="3e1de39bb4f5bb6e3de7a8b03f04e40bfdd80066fb87d6ff8c3b5cf1870fc1e7";BASELINE_ROWS=551;BASELINE_SHA256="7b4d4985605d2b3454bef76015de9ccb3aa1ae06a9d8d763c44f96755cca8b4a"
EXPECTED_CAPACITY={"before":{"conflict_units":256,"equipment_material":22,"resources_general":82,"safety_consent":86,"technique":66},"after":{"conflict_units":257,"equipment_material":22,"resources_general":82,"safety_consent":87,"technique":66}}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V309/"context_merit_audit_v309.jsonl",V309/"pending_curation_context_merit_v309.jsonl",V309/"report_context_merit_v309.json")
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v309 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v310-input";d.mkdir(parents=True,exist_ok=True);base=d/"v309.jsonl";build_baseline(base,d/"v309.report.json");core.build_projection_with_inputs(out,report,(),(base,ADDITIONS))
def observe():
 with tempfile.TemporaryDirectory(prefix=".v310-observe-",dir=OUT_DIR) as t:
  d=Path(t);base=d/"base.jsonl";build_baseline(base,d/"base.report.json");before=read_jsonl(base);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True);additions_builder.main()
 if EXPECTED_ADDITIONS_SHA256!="PENDING" and file_sha256(ADDITIONS)!=EXPECTED_ADDITIONS_SHA256:raise ValueError("addition drift")
 audits=[]
 for i,r in enumerate(read_jsonl(ADDITIONS),1):
  p=ROOT/r["source_lineage"]["raw_document"];d=json.loads(p.read_text())
  if (r["url"],r["document_sha256"])!=(d["url"],d["document_sha256"]) or not all(x in d["text"] for x in r["evidence"].splitlines()):raise ValueError("lineage")
  audits.append({"audit_index":i,"decision":"add","document_sha256":r["document_sha256"],"fact_id":r["fact_id"],"proposed_answer":r["answer"],"proposed_question":r["question"],"reason":r["paraphrase_rationale"],"reason_code":f"add_distinct_{r['topic']}_fact","review_pass":"shared_document_year_attributed_event_policy","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v310","schema":"context-merit-audit-v310","source":r["source"],"source_document":portable(p),"source_document_file_sha256":file_sha256(p),"source_support":"manual_paraphrase","support_evidence":r["evidence"],"support_evidence_sha256":text_sha256(r["evidence"]),"url":r["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,[]);o=observe()
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(554,612) or o["before"]!=EXPECTED_CAPACITY["before"] or o["after"]!=EXPECTED_CAPACITY["after"]:raise ValueError(f"projection drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 delta={k:o["after"][k]-o["before"][k] for k in o["before"]};REPORT.write_text(json.dumps({"addition_artifact":{"path":portable(ADDITIONS),"rows":3,"sha256":file_sha256(ADDITIONS)},"audit":{"by_decision":{"add":3,"drop":0,"edit":0,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":delta,"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":3,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":0,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v310","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
