#!/usr/bin/env python3
"""Drop four weaker semantic copies while preserving partner-specific consent rows."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V329=DATA/"manual_reviews/context_merit_audit_v329";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V329),str(V290)]
import build_context_merit_audit_v329 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v330.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v330.jsonl";REPORT=OUT_DIR/"report_context_merit_v330.json";BASELINE_ROWS=531;BASELINE_SHA256="a4a99444d03a5cf8f76396db90e2c640224a1941fdb99740cf96c093ce2ee1a9";EXPECTED_OUTPUT_SHA256="8a6d1dc33e6ecf63ff3884f6dbded3d43cae99e4788e34dd929f008de1761a33"
EXPECTED_CAPACITY_BEFORE={"conflict_units":260,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":72};EXPECTED_CAPACITY_AFTER={"conflict_units":260,"equipment_material":23,"resources_general":84,"safety_consent":80,"technique":73}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V329/"context_merit_audit_v329.jsonl",V329/"pending_curation_context_merit_v329.jsonl",V329/"report_context_merit_v329.json")
SPECS=(
 {"fact_id":"fact-08377c26e9a87dcb188a","active_index":18,"expected_question":"According to Ugo's account, how did Yukimura believe kinbaku play should develop?","expected_answer":"through circular, real-time interaction between the person tying and the person being tied","retained_fact":"fact-ca480c5cb3377d9d70ed","reason_code":"drop_yukimura_circular_interaction_duplicate","reason":"The retained same-document row explains the same circular interaction as immediate nonverbal messages and repeated feedback that develops the scene in real time."},
 {"fact_id":"fact-1facadfc23e069195901","active_index":112,"expected_question":"How does the Uramado article characterize cross-cultural influence between Japanese and Western bondage art?","expected_answer":"cross cultural communication flows in both directions","retained_fact":"fact-9dd323109ac8dd2996bc","reason_code":"drop_uramado_bidirectionality_fragment_duplicate","reason":"The retained same-document row states the same bidirectional point and identifies the Japanese-to-Western and Western-to-Japanese influence directions explicitly."},
 {"fact_id":"fact-162db1b78ad88be3d402","active_index":195,"expected_question":"What did Tamai Keiyuu establish in Tokyo in 1976?","expected_answer":"his periodical SM theatre show","retained_fact":"fact-9ed3d4b53b4fac1d5b16","reason_code":"drop_tamai_show_title_date_recall_duplicate","reason":"The retained same-document row teaches the broader historical point that opening SM theatre shows to the public led Ugo to credit Tamai with establishing the modern SM show."},
 {"fact_id":"fact-d219f277448544afa318","active_index":409,"expected_question":"Where can someone find the recommended My Nawashi natural-fiber rope shop?","expected_answer":"My Nawashi: https://www.etsy.com/shop/MyNawashi","retained_fact":"fact-2d054b924311ba9b82a2","reason_code":"drop_unevidenced_my_nawashi_bare_url_duplicate","reason":"The retained owner-resource row preserves the exact My Nawashi URL and category while explicitly limiting its claim because the shop page was not reliably inspectable."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v329 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v330-input";d.mkdir(parents=True,exist_ok=True);base=d/"v329.jsonl";build_baseline(base,d/"v329.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v330-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v330-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v329.jsonl";build_baseline(base,d/"v329.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]];retained=by_fact[s["retained_fact"]][1]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence") or retained.get("evidence")
  if not evidence:raise ValueError("missing active and retained evidence")
  curations.append({"action":"drop","document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v330","source_lineage":active["source_lineage"]})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"drop","document_sha256":active["document_sha256"],"fact_id":active["fact_id"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"retained_fact_id":s["retained_fact"],"review_pass":"substantive_semantic_duplicate_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v330","schema":"context-merit-audit-v330","source":active["source"],"source_support":"manual_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(527,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":4,"edit":0,"keep":0},"path":portable(AUDIT),"rows":4,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":4,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v330","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
