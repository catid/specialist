#!/usr/bin/env python3
"""Integrate the eighth distinct-document technique tranche."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V296=DATA/"manual_reviews/context_merit_audit_v296";V290=DATA/"manual_reviews/context_merit_audit_v290";AD=DATA/"manual_reviews/technique_additions_v8";sys.path[:0]=[str(ROOT),str(V296),str(V290),str(AD)]
import build_context_merit_audit_v296 as previous
import build_context_merit_audit_v290 as core
import build_technique_additions_v8 as additions_builder
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v297.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v297.jsonl";REPORT=OUT_DIR/"report_context_merit_v297.json";ADDITIONS=additions_builder.OUTPUT;EXPECTED_ADDITIONS_SHA256="211c2e4dfcfe22e4daa77fe893c16f93f82bd1c40c2fabcc1f314d763ef506db";EXPECTED_OUTPUT_SHA256="93097a170a71269f46b538fdfdf40a22e9d228795abbf0262deb52cc8aa4535c";BASELINE_ROWS=513;BASELINE_SHA256="9038c7b518e2962083f6579dddd257b9e19f7e00b8119b9f5fd8bc0a1421ae4c"
EXPECTED_CAPACITY={"before":{"conflict_units":220,"equipment_material":21,"resources_general":77,"safety_consent":72,"technique":50},"after":{"conflict_units":223,"equipment_material":21,"resources_general":77,"safety_consent":72,"technique":53}}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable
PRIOR_ARTIFACTS=(V296/"context_merit_audit_v296.jsonl",V296/"pending_curation_context_merit_v296.jsonl",V296/"report_context_merit_v296.json")
def build_baseline(out,rep):
 previous.build_projection(out,rep)
 if(len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v296 baseline drift")
def build_projection(out,rep):
 d=Path(out).parent/f".{Path(out).name}.v297-input";d.mkdir(parents=True,exist_ok=True);base=d/"v296.jsonl";build_baseline(base,d/"v296.report.json");core.build_projection_with_inputs(out,rep,(),(base,ADDITIONS))
def observe():
 with tempfile.TemporaryDirectory(prefix=".v297-observation-",dir=OUT_DIR)as temp:
  d=Path(temp);base=d/"base.jsonl";build_baseline(base,d/"base.report.json");base_rows=read_jsonl(base);out=d/"out.jsonl";rep=d/"out.report.json";datasets=[];reports=[]
  for _ in(1,2):build_projection(out,rep);datasets.append(out.read_bytes());reports.append(rep.read_bytes())
  rows=read_jsonl(out);return{"baseline_capacity":conservative_capacity(base_rows),"dataset_equal":datasets[0]==datasets[1],"dataset_sha256":hashlib.sha256(datasets[0]).hexdigest(),"eval_fact_count":json.loads(reports[0])["eval_fact_count"],"output_capacity":conservative_capacity(rows),"report_equal":reports[0]==reports[1],"rows":len(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True);additions_builder.main()
 if file_sha256(ADDITIONS)!=EXPECTED_ADDITIONS_SHA256:raise ValueError("addition drift")
 audits=[]
 for i,row in enumerate(read_jsonl(ADDITIONS),1):
  path=ROOT/row["source_lineage"]["raw_document"];doc=json.loads(path.read_text())
  if(row["url"],row["document_sha256"])!=(doc["url"],doc["document_sha256"])or not all(line in doc["text"]for line in row["evidence"].splitlines()):raise ValueError("source/evidence drift")
  audits.append({"audit_index":i,"decision":"add","document_sha256":row["document_sha256"],"fact_id":row["fact_id"],"proposed_answer":row["answer"],"proposed_question":row["question"],"reason":row["paraphrase_rationale"],"reason_code":f"add_distinct_{row['topic']}_fact","review_pass":"distinct_document_technique_additions","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v297","schema":"context-merit-audit-v297","source":row["source"],"source_document":portable(path),"source_document_file_sha256":file_sha256(path),"source_support":"manual_paraphrase","support_evidence":row["evidence"],"support_evidence_sha256":text_sha256(row["evidence"]),"url":row["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,[]);o=observe()
 if not o["dataset_equal"]or not o["report_equal"]:raise ValueError("nondeterministic")
 if(o["rows"],o["eval_fact_count"])!=(516,612):raise ValueError("row/eval drift")
 if o["baseline_capacity"]!=EXPECTED_CAPACITY["before"]or o["output_capacity"]!=EXPECTED_CAPACITY["after"]:raise ValueError(f"capacity drift:{o['baseline_capacity']}->{o['output_capacity']}")
 if EXPECTED_OUTPUT_SHA256!="PENDING"and o["dataset_sha256"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("hash drift")
 report={"addition_artifact":{"path":portable(ADDITIONS),"rows":3,"sha256":file_sha256(ADDITIONS)},"audit":{"by_decision":{"add":3,"drop":0,"edit":0,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["output_capacity"],"before":o["baseline_capacity"],"delta":{k:o["output_capacity"][k]-o["baseline_capacity"][k]for k in o["baseline_capacity"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)}for p in PRIOR_ARTIFACTS]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":3,"output_rows":o["rows"],"output_sha256":o["dataset_sha256"],"repeat_dataset_byte_identical":o["dataset_equal"],"repeat_projection_report_byte_identical":o["report_equal"],"sealed_eval_fact_count_reported_by_tooling":o["eval_fact_count"]},"new_pending_curation":{"decisions":0,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v297","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}};REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
