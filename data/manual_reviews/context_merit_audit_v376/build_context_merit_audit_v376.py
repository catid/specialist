#!/usr/bin/env python3
"""Complete three placement, publishing, and terminology train-only answers."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V375=DATA/"manual_reviews/context_merit_audit_v375";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V375),str(V290)]
import build_context_merit_audit_v375 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v376.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v376.jsonl";REPORT=OUT_DIR/"report_context_merit_v376.json";BASELINE_ROWS=531;BASELINE_SHA256="93d447af1fe6569aff2c972b9b4bc194450de54f0e5887bc86e3113bdb250751";EXPECTED_OUTPUT_SHA256="1e0c4e64e63dd1a74803c06541bb19c1d8359fef1548574848595e7cdea028e5"
EXPECTED_CAPACITY_BEFORE={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71};EXPECTED_CAPACITY_AFTER={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V375/"context_merit_audit_v375.jsonl",V375/"pending_curation_context_merit_v375.jsonl",V375/"report_context_merit_v375.json")
SPECS=(
 {"fact_id":"fact-7b4df0ec457a83e330b2","active_index":139,"expected_question":"In Rope365’s box-tie self-evaluation checklist, how should any cinches lie?","expected_answer":"flat with no bulk between the arms and the torso","question":"In Rope365’s box-tie self-evaluation checklist, how should any cinches lie?","answer":"Any cinches should lie flat, with no bulk between the arms and torso.","reason_code":"complete_box_tie_cinch_placement","reason":"The replacement converts the source-supported checklist fragment into a complete answer without changing its placement guidance."},
 {"fact_id":"fact-4acbea7398adadf358ba","active_index":148,"expected_question":"In “The Woman Who Wasn’t There,” what publishing problem does Reiko’s conflict with her male patrons dramatize?","expected_answer":"how to distinguish seme from mere violence, vulgar appetite, or crude spectacle","question":"In “The Woman Who Wasn’t There,” what publishing problem does Reiko’s conflict with her male patrons dramatize?","answer":"It dramatizes the problem of distinguishing seme from mere violence, vulgar appetite, or crude spectacle.","reason_code":"complete_seme_publishing_distinction","reason":"The replacement turns the supported clause into a complete response while preserving every contrast named by the source."},
 {"fact_id":"fact-c18687263b02155112ce","active_index":181,"expected_question":"What conclusion does Ugo draw about historical kinbaku and shibari terminology?","expected_answer":"the meaning of a word is always drifting and changing","question":"What conclusion does Ugo draw about historical kinbaku and shibari terminology?","answer":"Ugo concludes that a word’s meaning is always drifting and changing.","reason_code":"complete_ugo_terminology_drift","reason":"The replacement supplies the requested attribution and makes the source-supported conclusion a complete sentence."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v375 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v376-input";d.mkdir(parents=True,exist_ok=True);base=d/"v375.jsonl";build_baseline(base,d/"v375.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v376-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v376-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v375.jsonl";build_baseline(base,d/"v375.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v376","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"placement_publishing_terminology_answer_completeness_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v376","schema":"context-merit-audit-v376","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(531,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v376","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
