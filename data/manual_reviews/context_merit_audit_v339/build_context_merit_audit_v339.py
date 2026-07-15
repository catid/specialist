#!/usr/bin/env python3
"""Integrate three train-only safety/mechanics additions."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V338=DATA/"manual_reviews/context_merit_audit_v338";V290=DATA/"manual_reviews/context_merit_audit_v290";ADD=DATA/"manual_reviews/safety_mechanics_additions_v22";sys.path[:0]=[str(ROOT),str(V338),str(V290),str(ADD)]
import build_context_merit_audit_v338 as previous
import build_context_merit_audit_v290 as core
import build_safety_mechanics_additions_v22 as additions_builder
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v339.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v339.jsonl";REPORT=OUT_DIR/"report_context_merit_v339.json";ADDITIONS=additions_builder.OUTPUT;EXPECTED_ADDITIONS_SHA256="611b632876f5aafaef7b3c9a5f905c90b30de7172d2eae50e46d8fb46fff1914";EXPECTED_OUTPUT_SHA256="8f029f4789a489cec272a8e2108a7b493e51243d71a9aa101d02d6ebd33f1e56";BASELINE_ROWS=526;BASELINE_SHA256="18ce02136c2c7993dd9dfe8dbde228632ad25a64d73be56dd4bc7d808d016509"
EXPECTED_CAPACITY={"before":{"conflict_units":260,"equipment_material":23,"resources_general":84,"safety_consent":80,"technique":73},"after":{"conflict_units":260,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":72}}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V338/"context_merit_audit_v338.jsonl",V338/"pending_curation_context_merit_v338.jsonl",V338/"report_context_merit_v338.json")
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v338 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v339-input";d.mkdir(parents=True,exist_ok=True);base=d/"v338.jsonl";build_baseline(base,d/"v338.report.json");core.build_projection_with_inputs(out,report,(),(base,ADDITIONS))
def observe():
 with tempfile.TemporaryDirectory(prefix=".v339-observe-",dir=OUT_DIR) as t:
  d=Path(t);base=d/"base.jsonl";build_baseline(base,d/"base.report.json");before=read_jsonl(base);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True);additions_builder.main()
 if EXPECTED_ADDITIONS_SHA256!="PENDING" and file_sha256(ADDITIONS)!=EXPECTED_ADDITIONS_SHA256:raise ValueError("addition drift")
 audits=[]
 for i,r in enumerate(read_jsonl(ADDITIONS),1):
  p=ROOT/r["source_lineage"]["raw_document"];d=json.loads(p.read_text())
  if (r["url"],r["document_sha256"])!=(d["url"],d["document_sha256"]) or r["evidence"] not in d["text"]:raise ValueError("lineage")
  audits.append({"audit_index":i,"decision":"add","document_sha256":r["document_sha256"],"fact_id":r["fact_id"],"proposed_answer":r["answer"],"proposed_question":r["question"],"reason":r["paraphrase_rationale"],"reason_code":f"add_distinct_{r['topic']}_fact","review_pass":"train_only_safety_mechanics_additions","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v339","schema":"context-merit-audit-v339","source":r["source"],"source_document":portable(p),"source_document_file_sha256":file_sha256(p),"source_support":"manual_paraphrase","support_evidence":r["evidence"],"support_evidence_sha256":text_sha256(r["evidence"]),"url":r["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,[]);o=observe()
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(529,612) or o["before"]!=EXPECTED_CAPACITY["before"]:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY["after"]!="PENDING" and o["after"]!=EXPECTED_CAPACITY["after"]:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 delta={k:o["after"][k]-o["before"][k] for k in o["before"]};REPORT.write_text(json.dumps({"addition_artifact":{"path":portable(ADDITIONS),"rows":3,"sha256":file_sha256(ADDITIONS)},"audit":{"by_decision":{"add":3,"drop":0,"edit":0,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":delta,"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":3,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":0,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v339","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
