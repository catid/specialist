#!/usr/bin/env python3
"""Complete three train-only answers whose copied wording omitted part of the question."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V346=DATA/"manual_reviews/context_merit_audit_v346";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V346),str(V290)]
import build_context_merit_audit_v346 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v347.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v347.jsonl";REPORT=OUT_DIR/"report_context_merit_v347.json";BASELINE_ROWS=531;BASELINE_SHA256="4316c22395589fd107b99b23df7e710e1330c565dc4dcf350bef314181f3a2b8";EXPECTED_OUTPUT_SHA256="94290ecdb6a821b020ef3e965a35acc62b2bec6533d945d61847128f1fe28c01"
EXPECTED_CAPACITY_BEFORE={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71};EXPECTED_CAPACITY_AFTER={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V346/"context_merit_audit_v346.jsonl",V346/"pending_curation_context_merit_v346.jsonl",V346/"report_context_merit_v346.json")
SPECS=(
 {"fact_id":"fact-5b761c5dcc5ae4e835cb","active_index":377,"expected_question":"When did the series of gatherings continued by Kinbiken begin, and who led it?","expected_answer":"The meeting is a continuation of a series of gatherings started in 1986, led by Nureki Chimuo, Naka sensei’s teacher and mentor.","question":"When did the series of gatherings continued by Kinbiken begin, and who led it?","answer":"The gatherings began in 1986 under Nureki Chimuo, Naka sensei’s teacher and mentor.","reason_code":"answer_kinbiken_date_and_leader_directly","reason":"The replacement directly answers both requested fields and removes copied present-tense framing about the later meeting."},
 {"fact_id":"fact-3b7d53022bf8e721d388","active_index":416,"expected_question":"Where does the article place the cultural roots of kinbaku, while distinguishing later use of Hojōjutsu techniques?","expected_answer":"Kinbaku culture in Japan has its roots in theater.","question":"Where does the article place the cultural roots of kinbaku, while distinguishing later use of Hojōjutsu techniques?","answer":"It places kinbaku’s cultural roots in theater, while saying that modern kinbakushi later researched Hojōjutsu and incorporated some of its techniques.","reason_code":"complete_theater_roots_hojojutsu_distinction","reason":"The replacement answers the question's second clause with the source's explicit distinction between cultural roots and later technical borrowing."},
 {"fact_id":"fact-38f2787a9da122bafb77","active_index":456,"expected_question":"Why can teaching or explaining rope deepen the teacher’s understanding?","expected_answer":"One needs a deep understanding of something to be able to explain it with words.","question":"Why can teaching or explaining rope deepen the teacher’s understanding?","answer":"Putting rope knowledge into words requires and tests a deep understanding, which can reveal what the teacher still needs to learn.","reason_code":"make_explanation_learning_mechanism_explicit","reason":"The replacement directly states why explanation deepens understanding and retains the source's point that the exercise forces improvement."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v346 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v347-input";d.mkdir(parents=True,exist_ok=True);base=d/"v346.jsonl";build_baseline(base,d/"v346.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v347-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v347-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v346.jsonl";build_baseline(base,d/"v346.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v347","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"question_answer_completeness_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v347","schema":"context-merit-audit-v347","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(531,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v347","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
