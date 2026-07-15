#!/usr/bin/env python3
"""Complete three maintenance, principles, and hygiene train-only answers."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V361=DATA/"manual_reviews/context_merit_audit_v361";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V361),str(V290)]
import build_context_merit_audit_v361 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v362.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v362.jsonl";REPORT=OUT_DIR/"report_context_merit_v362.json";BASELINE_ROWS=531;BASELINE_SHA256="475f45a4fea0bf6bf9a89efbc2e90ba517266ef006bc0db97fbd618ab54cd48d";EXPECTED_OUTPUT_SHA256="dbadb010da461aa0d580917e7f7eb2fe53b35dfb31a92629a17564e424ffceb9"
EXPECTED_CAPACITY_BEFORE={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71};EXPECTED_CAPACITY_AFTER={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V361/"context_merit_audit_v361.jsonl",V361/"pending_curation_context_merit_v361.jsonl",V361/"report_context_merit_v361.json")
SPECS=(
 {"fact_id":"fact-a16be8c0ea339d4547c5","active_index":195,"expected_question":"What did Esinem’s limited natural-fiber friction tests report after wax was added?","expected_answer":"reduced friction and extended life","question":"What did Esinem’s limited natural-fiber friction tests report after wax was added?","answer":"Adding wax reduced friction and extended the rope’s life in those tests.","reason_code":"complete_limited_wax_test_result","reason":"The replacement converts the result fragment into a complete answer while retaining the question’s important limited-test scope."},
 {"fact_id":"fact-543e9d6a816b29c35e1c","active_index":199,"expected_question":"What did Yukimura consider the essence of Yukimura Ryuu?","expected_answer":"understanding the foundations and the principles","question":"What did Yukimura consider the essence of Yukimura Ryuu?","answer":"He considered understanding its foundations and principles the essence; the ties were tools for finding its heart.","reason_code":"complete_yukimura_foundations_principles_answer","reason":"The replacement completes the answer and restores the source’s distinction between foundational principles and ties as tools."},
 {"fact_id":"fact-231cd365c3c42a1bdcd9","active_index":208,"expected_question":"What does Anatomie Studio suggest using for parts of a tie likely to contact bodily fluids?","expected_answer":"a different rope or even a cloth","question":"What does Anatomie Studio suggest using for parts of a tie likely to contact bodily fluids?","answer":"Use a different rope or a cloth for the parts of the tie likely to contact bodily fluids.","reason_code":"complete_bodily_fluid_barrier_guidance","reason":"The replacement turns the object fragment into a direct, practical hygiene step without extending beyond the source’s options."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v361 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v362-input";d.mkdir(parents=True,exist_ok=True);base=d/"v361.jsonl";build_baseline(base,d/"v361.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v362-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v362-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v361.jsonl";build_baseline(base,d/"v361.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v362","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"maintenance_principles_hygiene_answer_completeness_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v362","schema":"context-merit-audit-v362","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(531,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v362","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
