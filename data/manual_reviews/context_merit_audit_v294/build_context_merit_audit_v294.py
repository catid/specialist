#!/usr/bin/env python3
"""Integrate the fifth distinct-document technique tranche."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V293=DATA/"manual_reviews/context_merit_audit_v293";AD=DATA/"manual_reviews/technique_additions_v5";sys.path[:0]=[str(ROOT),str(V293),str(AD)]
import build_context_merit_audit_v293 as previous
import build_technique_additions_v5 as additions_builder
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v294.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v294.jsonl";REPORT=OUT_DIR/"report_context_merit_v294.json";ADDITIONS=additions_builder.OUTPUT
EXPECTED_ADDITIONS_SHA256="5258c9a759bb5a5f4bce26671a5476c3f7e6454e6700c9aaed8d6195e20fc9fd";EXPECTED_OUTPUT_SHA256="b9768507b55aa6650f86b1cc4cf849d268f4d0c619f53384b3b74eca2af61c5b";BASELINE_ROWS=504;BASELINE_SHA256="1913fbab4e3b0638683b44f02682cd9316548750046f13c70bd8abfe7327d960"
EXPECTED_CAPACITY={"before":{"conflict_units":211,"equipment_material":18,"resources_general":74,"safety_consent":72,"technique":47},"after":{"conflict_units":214,"equipment_material":18,"resources_general":74,"safety_consent":72,"technique":50}}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable
def build_baseline(output,report):
 previous.build_projection(output,report)
 if (len(read_jsonl(output)),file_sha256(output))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v293 baseline drift")
def build_projection(output,report):
 d=Path(output).parent/f".{Path(output).name}.v294-input";d.mkdir(parents=True,exist_ok=True);base=d/"v293.jsonl";build_baseline(base,d/"v293.report.json");previous.previous.previous.previous.build_projection_with_inputs(output,report,(),(base,ADDITIONS))
def prior_decision_artifacts():return tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/name for v in range(1,294) for name in (f"context_merit_audit_v{v}.jsonl",f"pending_curation_context_merit_v{v}.jsonl",f"report_context_merit_v{v}.json"))
def observe():
 with tempfile.TemporaryDirectory(prefix=".v294-observation-",dir=OUT_DIR) as temp:
  d=Path(temp);base=d/"baseline.jsonl";build_baseline(base,d/"baseline.report.json");base_rows=read_jsonl(base);out=d/"projection.jsonl";rep=d/"projection.report.json";datasets=[];reports=[]
  for _ in (1,2):build_projection(out,rep);datasets.append(out.read_bytes());reports.append(rep.read_bytes())
  rows=read_jsonl(out);return{"baseline_capacity":conservative_capacity(base_rows),"dataset_equal":datasets[0]==datasets[1],"dataset_sha256":hashlib.sha256(datasets[0]).hexdigest(),"eval_fact_count":json.loads(reports[0])["eval_fact_count"],"output_capacity":conservative_capacity(rows),"report_equal":reports[0]==reports[1],"rows":len(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True);additions_builder.main()
 if file_sha256(ADDITIONS)!=EXPECTED_ADDITIONS_SHA256:raise ValueError("addition drift")
 audits=[]
 for i,row in enumerate(read_jsonl(ADDITIONS),1):
  path=ROOT/row["source_lineage"]["raw_document"];doc=json.loads(path.read_text())
  if (row["url"],row["document_sha256"])!=(doc["url"],doc["document_sha256"]) or not all(line in doc["text"] for line in row["evidence"].splitlines()):raise ValueError("source/evidence drift")
  audits.append({"audit_index":i,"decision":"add","document_sha256":row["document_sha256"],"fact_id":row["fact_id"],"proposed_answer":row["answer"],"proposed_question":row["question"],"reason":row["paraphrase_rationale"],"reason_code":f"add_distinct_{row['topic']}_fact","review_pass":"distinct_document_technique_additions","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v294","schema":"context-merit-audit-v294","source":row["source"],"source_document":portable(path),"source_document_file_sha256":file_sha256(path),"source_support":"manual_paraphrase","support_evidence":row["evidence"],"support_evidence_sha256":text_sha256(row["evidence"]),"url":row["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,[]);o=observe()
 if not o["dataset_equal"] or not o["report_equal"]:raise ValueError("nondeterministic")
 if (o["rows"],o["eval_fact_count"])!=(507,612):raise ValueError("row/eval drift")
 if o["baseline_capacity"]!=EXPECTED_CAPACITY["before"] or o["output_capacity"]!=EXPECTED_CAPACITY["after"]:raise ValueError(f"capacity drift: {o['baseline_capacity']} -> {o['output_capacity']}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["dataset_sha256"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("hash drift")
 report={"addition_artifact":{"path":portable(ADDITIONS),"rows":3,"sha256":file_sha256(ADDITIONS)},"audit":{"by_decision":{"add":3,"drop":0,"edit":0,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["output_capacity"],"before":o["baseline_capacity"],"delta":{k:o["output_capacity"][k]-o["baseline_capacity"][k] for k in o["baseline_capacity"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"frozen_prior_decision_artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in prior_decision_artifacts()],"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":3,"output_rows":o["rows"],"output_sha256":o["dataset_sha256"],"repeat_dataset_byte_identical":o["dataset_equal"],"repeat_projection_report_byte_identical":o["report_equal"],"sealed_eval_fact_count_reported_by_tooling":o["eval_fact_count"]},"new_pending_curation":{"decisions":0,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"projected_baseline":{"description":"complete train-only candidate through v293","rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"schema":"context-merit-audit-report-v294","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}};REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
