#!/usr/bin/env python3
"""Integrate the ninth distinct-document community tranche."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";VP=DATA/"manual_reviews/context_merit_audit_v297";VC=DATA/"manual_reviews/context_merit_audit_v290";AD=DATA/"manual_reviews/community_additions_v9";sys.path[:0]=[str(ROOT),str(VP),str(VC),str(AD)]
import build_context_merit_audit_v297 as previous
import build_context_merit_audit_v290 as core
import build_community_additions_v9 as additions_builder
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v298.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v298.jsonl";REPORT=OUT_DIR/"report_context_merit_v298.json";ADDITIONS=additions_builder.OUTPUT;EXPECTED_ADDITIONS_SHA256="9b96a61054a7cc36d1a7ef34b3d5576d476c8c9f9663142ceab14ed40ad2b16e";EXPECTED_OUTPUT_SHA256="20530df56ea5eca3d0e775f455a1448b91a997f5f0f4c2492868f7ee492b01ff";BASELINE_ROWS=516;BASELINE_SHA256="93097a170a71269f46b538fdfdf40a22e9d228795abbf0262deb52cc8aa4535c"
EXPECTED_CAPACITY={"before":{"conflict_units":223,"equipment_material":21,"resources_general":77,"safety_consent":72,"technique":53},"after":{"conflict_units":226,"equipment_material":21,"resources_general":79,"safety_consent":73,"technique":53}}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(VP/"context_merit_audit_v297.jsonl",VP/"pending_curation_context_merit_v297.jsonl",VP/"report_context_merit_v297.json")
def build_baseline(o,r):
 previous.build_projection(o,r)
 if(len(read_jsonl(o)),file_sha256(o))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v297 baseline drift")
def build_projection(o,r):
 d=Path(o).parent/f".{Path(o).name}.v298-input";d.mkdir(parents=True,exist_ok=True);base=d/"v297.jsonl";build_baseline(base,d/"v297.report.json");core.build_projection_with_inputs(o,r,(),(base,ADDITIONS))
def observe():
 with tempfile.TemporaryDirectory(prefix=".v298-observation-",dir=OUT_DIR)as temp:
  d=Path(temp);base=d/"base.jsonl";build_baseline(base,d/"base.report.json");br=read_jsonl(base);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in(1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"baseline_capacity":conservative_capacity(br),"dataset_equal":ds[0]==ds[1],"dataset_sha256":hashlib.sha256(ds[0]).hexdigest(),"eval_fact_count":json.loads(rs[0])["eval_fact_count"],"output_capacity":conservative_capacity(rows),"report_equal":rs[0]==rs[1],"rows":len(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True);additions_builder.main()
 if file_sha256(ADDITIONS)!=EXPECTED_ADDITIONS_SHA256:raise ValueError("addition drift")
 audits=[]
 for i,row in enumerate(read_jsonl(ADDITIONS),1):
  p=ROOT/row["source_lineage"]["raw_document"];d=json.loads(p.read_text())
  if(row["url"],row["document_sha256"])!=(d["url"],d["document_sha256"])or not all(x in d["text"]for x in row["evidence"].splitlines()):raise ValueError("source/evidence drift")
  audits.append({"audit_index":i,"decision":"add","document_sha256":row["document_sha256"],"fact_id":row["fact_id"],"proposed_answer":row["answer"],"proposed_question":row["question"],"reason":row["paraphrase_rationale"],"reason_code":f"add_distinct_{row['topic']}_fact","review_pass":"distinct_document_community_additions","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v298","schema":"context-merit-audit-v298","source":row["source"],"source_document":portable(p),"source_document_file_sha256":file_sha256(p),"source_support":"manual_paraphrase","support_evidence":row["evidence"],"support_evidence_sha256":text_sha256(row["evidence"]),"url":row["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,[]);o=observe()
 if not o["dataset_equal"]or not o["report_equal"]or(o["rows"],o["eval_fact_count"])!=(519,612):raise ValueError("projection drift")
 if o["baseline_capacity"]!=EXPECTED_CAPACITY["before"]or o["output_capacity"]!=EXPECTED_CAPACITY["after"]:raise ValueError(f"capacity drift:{o['baseline_capacity']}->{o['output_capacity']}")
 if EXPECTED_OUTPUT_SHA256!="PENDING"and o["dataset_sha256"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("hash drift")
 report={"addition_artifact":{"path":portable(ADDITIONS),"rows":3,"sha256":file_sha256(ADDITIONS)},"audit":{"by_decision":{"add":3,"drop":0,"edit":0,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["output_capacity"],"before":o["baseline_capacity"],"delta":{k:o["output_capacity"][k]-o["baseline_capacity"][k]for k in o["baseline_capacity"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)}for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":3,"output_rows":o["rows"],"output_sha256":o["dataset_sha256"],"repeat_dataset_byte_identical":o["dataset_equal"],"repeat_projection_report_byte_identical":o["report_equal"],"sealed_eval_fact_count_reported_by_tooling":o["eval_fact_count"]},"new_pending_curation":{"decisions":0,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v298","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}};REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
