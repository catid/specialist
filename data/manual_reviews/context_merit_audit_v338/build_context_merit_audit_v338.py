#!/usr/bin/env python3
"""Clarify three source-grounded rope handling and risk answers."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V337=DATA/"manual_reviews/context_merit_audit_v337";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V337),str(V290)]
import build_context_merit_audit_v337 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v338.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v338.jsonl";REPORT=OUT_DIR/"report_context_merit_v338.json";BASELINE_ROWS=526;BASELINE_SHA256="f865336529bfae02f65f8fd28a3daa4bcd98c72fe9cae69b711b227cbc386507";EXPECTED_OUTPUT_SHA256="18ce02136c2c7993dd9dfe8dbde228632ad25a64d73be56dd4bc7d808d016509"
EXPECTED_CAPACITY_BEFORE={"conflict_units":260,"equipment_material":23,"resources_general":84,"safety_consent":80,"technique":73};EXPECTED_CAPACITY_AFTER={"conflict_units":260,"equipment_material":23,"resources_general":84,"safety_consent":80,"technique":73}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V337/"context_merit_audit_v337.jsonl",V337/"pending_curation_context_merit_v337.jsonl",V337/"report_context_merit_v337.json")
SPECS=(
 {"fact_id":"fact-b529f3b3d973305d7556","active_index":224,"expected_question":"What does Rope365 recommend when high stranding appears inside a strand and strength matters?","expected_answer":"It is recommended to avoid this kind of rope when strength is important as this is generally the sign that the tension inside the rope might weaken it.","question":"What does Rope365 recommend when high stranding appears inside a strand and strength matters?","answer":"Avoid that rope for strength-critical use, because internal high stranding usually indicates uneven tension from rope making that may weaken it.","reason_code":"clarify_internal_high_stranding_action","reason":"The replacement converts the source's awkward passive sentence into a direct action and preserves its stated weakening mechanism."},
 {"fact_id":"fact-edb3a4e80df3438b2d10","active_index":307,"expected_question":"What risk warning does Rope365 give about the historical box ties on this page?","expected_answer":"most of these ties have high risks, often starting with a loop around the neck and cinching on nerves","question":"What risk warning does Rope365 give about the historical box ties on this page?","answer":"Rope365 calls most of these ties high-risk because they often begin with a neck loop and place cinches over nerves; it presents them as inspiration that should be adapted to mitigate those risks.","reason_code":"complete_historical_box_tie_risk_warning","reason":"The replacement completes the fragment and includes the source's practical mitigation instruction rather than leaving only a hazard list."},
 {"fact_id":"fact-68abd64f4d0054e6ebf0","active_index":488,"expected_question":"Why does Rope365 recommend pulling rather than pushing rope into a gap?","expected_answer":"pushing rope in a gap is generally inefficient and can damage it","question":"Why does Rope365 recommend pulling rather than pushing rope into a gap?","answer":"Pushing rope into a gap is inefficient and can damage it, so Rope365 recommends pulling it through, using a finger-hooking technique in tight gaps.","reason_code":"complete_pull_not_push_gap_guidance","reason":"The replacement turns the source fragment into actionable handling guidance and retains the supported tight-gap technique."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v337 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v338-input";d.mkdir(parents=True,exist_ok=True);base=d/"v337.jsonl";build_baseline(base,d/"v337.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v338-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v338-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v337.jsonl";build_baseline(base,d/"v337.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v338","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"rope_handling_and_risk_guidance_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v338","schema":"context-merit-audit-v338","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(526,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v338","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
