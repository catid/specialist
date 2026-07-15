#!/usr/bin/env python3
"""Complete three supervision, communication, and hand-monitoring train-only answers."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V407=DATA/"manual_reviews/context_merit_audit_v407";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V407),str(V290)]
import build_context_merit_audit_v407 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v408.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v408.jsonl";REPORT=OUT_DIR/"report_context_merit_v408.json";BASELINE_ROWS=531;BASELINE_SHA256="b1797a9ecb13c570a965d1c13900afb13b2b5dd7680240508f491c4a17d90de9";EXPECTED_OUTPUT_SHA256="8029ef56068f3f542de9e1b9c4a10081ae8caf66938df812179824331f730037"
EXPECTED_CAPACITY_BEFORE={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71};EXPECTED_CAPACITY_AFTER={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V407/"context_merit_audit_v407.jsonl",V407/"pending_curation_context_merit_v407.jsonl",V407/"report_context_merit_v407.json")
SPECS=(
 {"fact_id":"fact-913cccf5f2d04e84b1b6","active_index":17,"expected_question":"According to Rope365, what supervision is required while someone is bound?","expected_answer":"Stay with the bound partner at all times.","question":"According to Rope365, what supervision is required while someone is bound?","answer":"Rope365 says to stay with the bound partner at all times.","reason_code":"complete_bound_partner_supervision","reason":"The replacement turns the source-supported imperative into a complete attributed answer without weakening the continuous-supervision requirement."},
 {"fact_id":"fact-8baaaff99e1ba4f7bd52","active_index":231,"expected_question":"What does Rope365 recommend when power dynamics may affect communication?","expected_answer":"Work harder to establish clear communication and confirm consent.","question":"What does Rope365 recommend when power dynamics may affect communication?","answer":"Rope365 recommends making extra effort to establish clear communication and confirm consent.","reason_code":"complete_power_dynamics_communication","reason":"The replacement turns the source-supported imperative into a complete attributed answer while preserving both communication and consent."},
 {"fact_id":"fact-160f6789dbea183062e3","active_index":374,"expected_question":"When a quick-release cuff binds the arms behind the back, what hand function does Rope365 say to monitor?","expected_answer":"Monitor whether the person can open and close their hands.","question":"When a quick-release cuff binds the arms behind the back, what hand function does Rope365 say to monitor?","answer":"Rope365 says to monitor whether the person can open and close their hands.","reason_code":"complete_quick_release_hand_monitoring","reason":"The replacement turns the source-supported imperative into a complete attributed answer without adding a monitoring criterion."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v407 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v408-input";d.mkdir(parents=True,exist_ok=True);base=d/"v407.jsonl";build_baseline(base,d/"v407.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v408-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v408-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v407.jsonl";build_baseline(base,d/"v407.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v408","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"supervision_communication_hand_monitoring_answer_completeness_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v408","schema":"context-merit-audit-v408","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(531,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v408","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
