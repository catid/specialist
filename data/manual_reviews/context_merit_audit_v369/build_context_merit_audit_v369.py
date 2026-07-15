#!/usr/bin/env python3
"""Contextualize three standalone train-only event and rope-safety Q&A rows."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V368=DATA/"manual_reviews/context_merit_audit_v368";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V368),str(V290)]
import build_context_merit_audit_v368 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v369.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v369.jsonl";REPORT=OUT_DIR/"report_context_merit_v369.json";BASELINE_ROWS=531;BASELINE_SHA256="eadc2799aa01292e8bd41f561a8eb9f601b6c7f86fbe8f372bccf1ae7aaf59cc";EXPECTED_OUTPUT_SHA256="99f39bc4791620e7dd38a15096f3c69c09dc3371a8e799579a2bb7c1bd1a5d4e"
EXPECTED_CAPACITY_BEFORE={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71};EXPECTED_CAPACITY_AFTER={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V368/"context_merit_audit_v368.jsonl",V368/"pending_curation_context_merit_v368.jsonl",V368/"report_context_merit_v368.json")
SPECS=(
 {"fact_id":"fact-e721e604c74ef70d341e","active_index":248,"expected_question":"What event-safety role does the source assign to dungeon monitors at many clubs and group-organized BDSM events?","expected_answer":"Dungeon monitors help ensure that house rules are followed and safewords are respected.","question":"According to Wikipedia’s BDSM overview, what event-safety role do dungeon monitors serve at many clubs and group-organized events?","answer":"Dungeon monitors help ensure that house rules are followed and safewords are respected.","reason_code":"name_source_in_dungeon_monitor_role","reason":"The replacement names the reference source and removes a hidden-metadata dependency while preserving the stated event-safety role."},
 {"fact_id":"fact-46c0cba03dcc399f852c","active_index":263,"expected_question":"What injuries or discomfort does the source say good rope handling should prevent?","expected_answer":"Good rope handling should prevent unnecessary skin friction and fatigue, unanticipated pinching, and accidental impacts from knots at the rope ends.","question":"According to Esinem’s “Does Direction Matter for a Gote?”, what injuries or discomfort should good rope handling prevent?","answer":"Good rope handling should prevent unnecessary skin friction and fatigue, unanticipated pinching, and accidental impacts from knots at the rope ends.","reason_code":"name_work_in_rope_handling_harm_question","reason":"The replacement names the article so the attributed rope-handling guidance remains standalone."},
 {"fact_id":"fact-c628ecb155d227e9afd0","active_index":282,"expected_question":"What nonverbal safety signals does the source give as alternatives when speech is restricted?","expected_answer":"dropping a ball or ringing a bell","question":"Which nonverbal safety signals does Wikipedia’s BDSM overview give as alternatives when speech is restricted?","answer":"Dropping a ball or ringing a bell can serve as a nonverbal safety signal.","reason_code":"name_source_and_complete_nonverbal_signal_answer","reason":"The replacement names the reference source and converts the two-item fragment into a complete safety-signal answer."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v368 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v369-input";d.mkdir(parents=True,exist_ok=True);base=d/"v368.jsonl";build_baseline(base,d/"v368.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v369-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v369-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v368.jsonl";build_baseline(base,d/"v368.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v369","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"standalone_event_rope_safety_source_context_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v369","schema":"context-merit-audit-v369","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(531,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v369","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
