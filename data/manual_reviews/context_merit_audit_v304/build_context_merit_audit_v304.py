#!/usr/bin/env python3
"""Integrate the two-row multi-person/chest tranche without filler."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; DATA=ROOT/"data"; V303=DATA/"manual_reviews/context_merit_audit_v303"; V290=DATA/"manual_reviews/context_merit_audit_v290"; ADD=DATA/"manual_reviews/multiperson_chest_additions_v15"
sys.path[:0]=[str(ROOT),str(V303),str(V290),str(ADD)]
import build_context_merit_audit_v303 as previous
import build_context_merit_audit_v290 as core
import build_multiperson_chest_additions_v15 as additions_builder
OUT_DIR=Path(__file__).resolve().parent; AUDIT=OUT_DIR/"context_merit_audit_v304.jsonl"; CURATION=OUT_DIR/"pending_curation_context_merit_v304.jsonl"; REPORT=OUT_DIR/"report_context_merit_v304.json"; ADDITIONS=additions_builder.OUTPUT
EXPECTED_ADDITIONS_SHA256="60b1ce76d1c2e9f522611ce848613f5c185caa57595036b848f60097f660ae21"; EXPECTED_OUTPUT_SHA256="b9260425ba23c54413c771840f584f621120f3891a1b6ef1380fe732f38de68d"; BASELINE_ROWS=534; BASELINE_SHA256="8ace985d0a66db9638cd303a3fb47a271a367365094b3b6b4ec401f1d7369401"
EXPECTED_CAPACITY={"before":{"conflict_units":241,"equipment_material":22,"resources_general":81,"safety_consent":79,"technique":59},"after":{"conflict_units":243,"equipment_material":22,"resources_general":81,"safety_consent":80,"technique":60}}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST; file_sha256=previous.file_sha256; text_sha256=previous.text_sha256; read_jsonl=previous.read_jsonl; write_jsonl=previous.write_jsonl; conservative_capacity=previous.conservative_capacity; portable=previous.portable
PRIOR=(V303/"context_merit_audit_v303.jsonl",V303/"pending_curation_context_merit_v303.jsonl",V303/"report_context_merit_v303.json")
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256): raise ValueError("v303 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v304-input"; d.mkdir(parents=True,exist_ok=True); base=d/"v303.jsonl"; build_baseline(base,d/"v303.report.json"); core.build_projection_with_inputs(out,report,(),(base,ADDITIONS))
def observe():
 with tempfile.TemporaryDirectory(prefix=".v304-observe-",dir=OUT_DIR) as t:
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
  audits.append({"audit_index":i,"decision":"add","document_sha256":r["document_sha256"],"fact_id":r["fact_id"],"proposed_answer":r["answer"],"proposed_question":r["question"],"reason":r["paraphrase_rationale"],"reason_code":f"add_distinct_{r['topic']}_fact","review_pass":"distinct_document_multiperson_chest_additions","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v304","schema":"context-merit-audit-v304","source":r["source"],"source_document":portable(p),"source_document_file_sha256":file_sha256(p),"source_support":"manual_paraphrase","support_evidence":r["evidence"],"support_evidence_sha256":text_sha256(r["evidence"]),"url":r["url"]})
 write_jsonl(AUDIT,audits); write_jsonl(CURATION,[]); o=observe()
 if not o["dataset_equal"] or not o["report_equal"] or (o["rows"],o["eval"])!=(536,612) or o["before"]!=EXPECTED_CAPACITY["before"] or o["after"]!=EXPECTED_CAPACITY["after"]: raise ValueError(f"projection drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256: raise ValueError("output drift")
 delta={k:o["after"][k]-o["before"][k] for k in o["before"]}
 report={"addition_artifact":{"path":portable(ADDITIONS),"rows":2,"sha256":file_sha256(ADDITIONS)},"audit":{"by_decision":{"add":2,"drop":0,"edit":0,"keep":0},"path":portable(AUDIT),"rows":2,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":delta,"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"excluded_third_candidate":json.loads(additions_builder.REPORT.read_text())["excluded_third_candidate"],"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":2,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["dataset_equal"],"repeat_projection_report_byte_identical":o["report_equal"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":0,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v304","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}}
 REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
