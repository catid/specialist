#!/usr/bin/env python3
"""Expand terse safety/communication answers and remove one same-document duplicate."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V317=DATA/"manual_reviews/context_merit_audit_v317";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V317),str(V290)]
import build_context_merit_audit_v317 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v318.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v318.jsonl";REPORT=OUT_DIR/"report_context_merit_v318.json";BASELINE_ROWS=546;BASELINE_SHA256="570be41f1bc1e9f26883387a4c543b144a4d5d412cbf677cd37addde6a8fc1e0";EXPECTED_OUTPUT_SHA256="fadaa3d594515494745be850541af9273858c79b1ae7c3c21be123213f88a042"
EXPECTED_CAPACITY_BEFORE={"conflict_units":258,"equipment_material":23,"resources_general":83,"safety_consent":84,"technique":68};EXPECTED_CAPACITY_AFTER={"conflict_units":258,"equipment_material":23,"resources_general":83,"safety_consent":84,"technique":68}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V317/"context_merit_audit_v317.jsonl",V317/"pending_curation_context_merit_v317.jsonl",V317/"report_context_merit_v317.json")
SPECS=(
 {"action":"edit","fact_id":"fact-09e116b082db0024df51","active_index":11,"expected_question":"According to Rope365, what should a rigger be able to do quickly when a tie interferes with breathing?","expected_answer":"untie anything causing the interference","question":"What release preparation does Rope365 recommend for anything that interferes with breathing?","answer":"Be able to untie it quickly, including tight chest compression or restrictive body positions such as backbends and forward bends.","reason_code":"expand_breathing_release_fragment_to_examples","reason":"The replacement keeps the urgent release action and names the evidence's concrete rope and position examples."},
 {"action":"edit","fact_id":"fact-8b7a070b9968b94b61fc","active_index":122,"expected_question":"How should rope partners adjust communication when they know each other less well?","expected_answer":"They should communicate more.","question":"How should rope partners adjust communication when they know each other less well?","answer":"Increase the amount and explicitness of communication as familiarity decreases.","reason_code":"replace_communication_fragment_with_clear_guidance","reason":"The replacement expresses the source's proportional communication guidance as a complete, reusable answer."},
 {"action":"edit","fact_id":"fact-4e780132c19846100d51","active_index":239,"expected_question":"What does Rope365 say should be discussed before tying when rope marks could be problematic?","expected_answer":"it is important to discuss your risk profile ahead of tying","question":"What should partners discuss before tying when rope marks may be a problem?","answer":"Until they know how the person's skin responds, assume marks may appear wherever rope touches and discuss beforehand which locations—such as the face, neck, or forearms—would be problematic.","reason_code":"expand_rope_mark_fragment_to_pre_scene_boundary","reason":"The replacement captures both the evidence's uncertainty about individual marking and its concrete pre-scene discussion guidance."},
 {"action":"edit","fact_id":"fact-ba81a0bd33f5fcc14100","active_index":359,"expected_question":"What training does Rope365 recommend for emergency incident response?","expected_answer":"a first-aid class","question":"What formal preparation does Rope365 recommend for common injuries and emergency incident response?","answer":"Take a first-aid class, alongside planning what could go wrong in the intended activity.","reason_code":"expand_first_aid_fragment_with_incident_planning_context","reason":"The replacement retains the formal training recommendation and restores the evidence's surrounding incident-planning context."},
 {"action":"drop","fact_id":"fact-ca9fc211047adc633b1c","active_index":479,"expected_question":"Why did Yuki say basic knowledge and technique are necessary in kinbaku?","expected_answer":"to protect the rope bottoms’ safety","reason_code":"drop_yukimura_safety_fragment_duplicate","reason":"The richer same-document row fact-e1ab827121ac0e6176ac already combines required safety knowledge and skill with Yukimura's emphasis on relationship, communication, and the tied person's beauty."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v317 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v318-input";d.mkdir(parents=True,exist_ok=True);base=d/"v317.jsonl";build_baseline(base,d/"v317.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v318-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v318-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v317.jsonl";build_baseline(base,d/"v317.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  common={"action":s["action"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v318","source_lineage":active["source_lineage"]}
  if s["action"]=="edit":common.update(answer=s["answer"],paraphrase_rationale=s["reason"],question=s["question"],support_type="manual_paraphrase")
  curations.append(common);audit={"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":s["action"],"document_sha256":active["document_sha256"],"fact_id":active["fact_id"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"terse_safety_and_communication_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v318","schema":"context-merit-audit-v318","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]}
  if s["action"]=="edit":audit.update(edited_answer=s["answer"],edited_question=s["question"],paraphrase_rationale=s["reason"])
  audits.append(audit)
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(545,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":1,"edit":4,"keep":0},"path":portable(AUDIT),"rows":5,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":5,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v318","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
