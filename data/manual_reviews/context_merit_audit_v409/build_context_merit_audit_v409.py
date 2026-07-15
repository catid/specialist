#!/usr/bin/env python3
"""Complete three pose, communication, and stop-signal train-only answers."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V408=DATA/"manual_reviews/context_merit_audit_v408";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V408),str(V290)]
import build_context_merit_audit_v408 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v409.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v409.jsonl";REPORT=OUT_DIR/"report_context_merit_v409.json";BASELINE_ROWS=531;BASELINE_SHA256="8029ef56068f3f542de9e1b9c4a10081ae8caf66938df812179824331f730037";EXPECTED_OUTPUT_SHA256="7bc66895c8709997f5baf655442993e6bb764264935517ab40d586253bd2be69"
EXPECTED_CAPACITY_BEFORE={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71};EXPECTED_CAPACITY_AFTER={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V408/"context_merit_audit_v408.jsonl",V408/"pending_curation_context_merit_v408.jsonl",V408/"report_context_merit_v408.json")
SPECS=(
 {"fact_id":"fact-567c8d85a1c4ed3a6fd7","active_index":39,"expected_question":"How can hero pose be modified when tying someone who finds the position difficult?","expected_answer":"Place a block or cushion under the seated person.","question":"How can hero pose be modified when tying someone who finds the position difficult?","answer":"Rope365 suggests placing a block or cushion under the seated person.","reason_code":"complete_hero_pose_support","reason":"The replacement turns the source-supported imperative into a complete attributed answer without adding a modification."},
 {"fact_id":"fact-05c23d8f949353e4d3a7","active_index":87,"expected_question":"How does Rope365 recommend phrasing communication when someone may be nervous, distressed, or under intense stimuli?","expected_answer":"Favour simple words and simple sentence structures.","question":"How does Rope365 recommend phrasing communication when someone may be nervous, distressed, or under intense stimuli?","answer":"Rope365 recommends using simple words and simple sentence structures.","reason_code":"complete_simple_communication_guidance","reason":"The replacement turns the source-supported imperative into a complete attributed answer while retaining both simplicity criteria."},
 {"fact_id":"fact-ca3d6cbf29c907e33166","active_index":208,"expected_question":"What does Anatomie Studio recommend doing with an agreed stop signal before a new-partner rope scene?","expected_answer":"Practise using it before it is truly needed.","question":"What does Anatomie Studio recommend doing with an agreed stop signal before a new-partner rope scene?","answer":"Anatomie Studio recommends practising the stop signal before it is truly needed.","reason_code":"complete_stop_signal_practice","reason":"The replacement turns the source-supported imperative into a complete attributed answer without changing when to practise it."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v408 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v409-input";d.mkdir(parents=True,exist_ok=True);base=d/"v408.jsonl";build_baseline(base,d/"v408.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v409-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v409-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v408.jsonl";build_baseline(base,d/"v408.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v409","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"pose_communication_stop_signal_answer_completeness_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v409","schema":"context-merit-audit-v409","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(531,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v409","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
