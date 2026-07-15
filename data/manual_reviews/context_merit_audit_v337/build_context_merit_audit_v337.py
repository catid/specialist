#!/usr/bin/env python3
"""Repair three source-grounded historical/performance answer fragments."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V336=DATA/"manual_reviews/context_merit_audit_v336";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V336),str(V290)]
import build_context_merit_audit_v336 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v337.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v337.jsonl";REPORT=OUT_DIR/"report_context_merit_v337.json";BASELINE_ROWS=526;BASELINE_SHA256="4f4768d838128fc718ea07a5990a970a9b8f9002a7e7d2c8a5120f1eaef50b61";EXPECTED_OUTPUT_SHA256="f865336529bfae02f65f8fd28a3daa4bcd98c72fe9cae69b711b227cbc386507"
EXPECTED_CAPACITY_BEFORE={"conflict_units":260,"equipment_material":23,"resources_general":84,"safety_consent":80,"technique":73};EXPECTED_CAPACITY_AFTER={"conflict_units":260,"equipment_material":23,"resources_general":84,"safety_consent":80,"technique":73}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V336/"context_merit_audit_v336.jsonl",V336/"pending_curation_context_merit_v336.jsonl",V336/"report_context_merit_v336.json")
SPECS=(
 {"fact_id":"fact-4cd0b80f55752daf3c3a","active_index":287,"expected_question":"What performance principle does Randa Mai illustrate by comparing the nawashi to a bunraku kuroko?","expected_answer":"The nawashi should stay in the background. The center of a performance is the woman who is being manipulated by rope","question":"What performance principle does Randa Mai illustrate by comparing the nawashi to a bunraku kuroko?","answer":"Randa Mai says the nawashi should remain in the background like a bunraku kuroko, keeping the person being tied at the center of the performance.","reason_code":"complete_randa_mai_performance_principle","reason":"The replacement combines the source's two fragmentary sentences into a standalone attributed answer while preserving its performer-focus analogy."},
 {"fact_id":"fact-3e6cb3c9486a186adf69","active_index":300,"expected_question":"What publishing milestone does the source attribute to Minomura Kou?","expected_answer":"the bakushi to first produce photo books in Japan featuring kinbaku","question":"What publishing milestone does the source attribute to Minomura Kou?","answer":"The source identifies Minomura Kou as the first bakushi to produce kinbaku photo books in Japan.","reason_code":"complete_minomura_publishing_milestone","reason":"The replacement turns an awkward noun fragment into a complete, explicitly source-attributed historical claim."},
 {"fact_id":"fact-3fc7a117d6055048fa1a","active_index":367,"expected_question":"What was significant about the World of Rope video CineMagic released in 1989?","expected_answer":"Nureki Chimuo’s first of eight videos in his landmark “World of Rope” series","question":"What was significant about the World of Rope video CineMagic released in 1989?","answer":"It was the first of eight videos in Nureki Chimuo's landmark World of Rope series.","reason_code":"complete_world_of_rope_milestone","reason":"The replacement converts the possessive source fragment into a clear standalone answer without adding chronology or significance beyond the evidence."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v336 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v337-input";d.mkdir(parents=True,exist_ok=True);base=d/"v336.jsonl";build_baseline(base,d/"v336.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v337-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v337-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v336.jsonl";build_baseline(base,d/"v336.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v337","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"historical_performance_publishing_answer_polish_train_only","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v337","schema":"context-merit-audit-v337","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(526,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v337","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
