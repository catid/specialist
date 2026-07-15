#!/usr/bin/env python3
"""Replace three generic safety labels with concrete roles and failure mechanisms."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V332=DATA/"manual_reviews/context_merit_audit_v332";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V332),str(V290)]
import build_context_merit_audit_v332 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v333.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v333.jsonl";REPORT=OUT_DIR/"report_context_merit_v333.json";BASELINE_ROWS=527;BASELINE_SHA256="5cc243de89903ca289aba16bc1ffcd28073cfcb51b8c96b8c0ca4f195046f55d";EXPECTED_OUTPUT_SHA256="8a4a2c80178d558037ec5f95dd274a870ac8f223f5ccd6a5281ea25557627dd8"
EXPECTED_CAPACITY_BEFORE={"conflict_units":260,"equipment_material":23,"resources_general":84,"safety_consent":80,"technique":73};EXPECTED_CAPACITY_AFTER={"conflict_units":260,"equipment_material":23,"resources_general":84,"safety_consent":80,"technique":73}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V332/"context_merit_audit_v332.jsonl",V332/"pending_curation_context_merit_v332.jsonl",V332/"report_context_merit_v332.json")
SPECS=(
 {"fact_id":"fact-bbfbdaa4409d7207656a","active_index":361,"expected_question":"What use and body-safety risk does Rope365 identify for the handcuff knot?","expected_answer":"The handcuff knot is a popular type of knots to capture two limbs at once. Since it’s a type of slip knot, it comes with the risk that it may tighten when put directly on the body.","question":"What use and body risk does Rope365 identify for the handcuff knot?","answer":"A handcuff knot can capture two limbs at once, but its slipped loops may tighten when the knot is placed directly on the body.","reason_code":"repair_handcuff_knot_grammar_and_tightening_mechanism","reason":"The replacement repairs the source-derived grammar and states the same two-limb use and slip-knot tightening risk directly."},
 {"fact_id":"fact-224ae798deadad238e97","active_index":445,"expected_question":"Who provides an additional safety net at many clubs and group-organized BDSM events?","expected_answer":"dungeon monitors (DMs)","question":"What event-safety role does the source assign to dungeon monitors at many clubs and group-organized BDSM events?","answer":"Dungeon monitors help ensure that house rules are followed and safewords are respected.","reason_code":"replace_dungeon_monitor_title_recall_with_observable_role","reason":"The replacement preserves the source's actor while teaching the concrete house-rule and safeword responsibilities instead of only the title."},
 {"fact_id":"fact-5bb1cd165ed0707113bb","active_index":474,"expected_question":"Why does Esinem say suspension should not be done by rote?","expected_answer":"To be safe and creative, you need to understand what you are doing and why.","question":"Why does Esinem say suspension steps should be understood rather than copied by rote?","answer":"Copying steps without understanding why they are done makes dangerous mistakes through ignorance easier; the page therefore emphasizes learning each action's reason.","reason_code":"replace_safe_creative_slogan_with_suspension_failure_mechanism","reason":"The replacement retains the source's understanding-over-rote principle while making its stated failure mechanism—dangerous mistakes through ignorance—explicit."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v332 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v333-input";d.mkdir(parents=True,exist_ok=True);base=d/"v332.jsonl";build_baseline(base,d/"v332.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v333-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v333-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v332.jsonl";build_baseline(base,d/"v332.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v333","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"generic_safety_to_observable_role_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v333","schema":"context-merit-audit-v333","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(527,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v333","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
