#!/usr/bin/env python3
"""Complete three role, shape, and terminology train-only answers."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V394=DATA/"manual_reviews/context_merit_audit_v394";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V394),str(V290)]
import build_context_merit_audit_v394 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v395.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v395.jsonl";REPORT=OUT_DIR/"report_context_merit_v395.json";BASELINE_ROWS=531;BASELINE_SHA256="e541e4ca91e2cc750ce4af2021578dfa6e2c3459f8273798117bd26eeca9f320";EXPECTED_OUTPUT_SHA256="41fed7952fda82d22896e3e4c5733c2f2b8f0583d38bb258afbf302593979f0c"
EXPECTED_CAPACITY_BEFORE={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71};EXPECTED_CAPACITY_AFTER={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V394/"context_merit_audit_v394.jsonl",V394/"pending_curation_context_merit_v394.jsonl",V394/"report_context_merit_v394.json")
SPECS=(
 {"fact_id":"fact-dacb96c9a6e3a6eb4a88","active_index":312,"expected_question":"What roles did Minomura Kou serve for SM Collector?","expected_answer":"adviser, artist, author, bakushi and critic","question":"What roles did Minomura Kou serve for SM Collector?","answer":"Minomura Kou served as an adviser, artist, author, bakushi, and critic for SM Collector.","reason_code":"complete_minomura_kou_roles","reason":"The replacement turns the source-supported five-role list into a complete answer without changing its scope."},
 {"fact_id":"fact-b97b252cc6fc305c9196","active_index":323,"expected_question":"What shape does the tasuki technique create in the back?","expected_answer":"an X on the back","question":"What shape does the tasuki technique create in the back?","answer":"The tasuki technique creates an X shape across the back.","reason_code":"complete_tasuki_shape","reason":"The replacement turns the source-supported shape fragment into a complete answer."},
 {"fact_id":"fact-6979c87b4dbde886851a","active_index":338,"expected_question":"What term does Rope365 use for a tie that offers agency only among painful, difficult, or otherwise bad options?","expected_answer":"Predicament","question":"What term does Rope365 use for a tie that offers agency only among painful, difficult, or otherwise bad options?","answer":"Rope365 calls this a predicament tie.","reason_code":"complete_predicament_term","reason":"The replacement turns the source-supported term into a complete attributed answer without expanding its definition."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v394 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v395-input";d.mkdir(parents=True,exist_ok=True);base=d/"v394.jsonl";build_baseline(base,d/"v394.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v395-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v395-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v394.jsonl";build_baseline(base,d/"v394.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v395","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"role_shape_terminology_answer_completeness_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v395","schema":"context-merit-audit-v395","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(531,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v395","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
