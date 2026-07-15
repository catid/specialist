#!/usr/bin/env python3
"""Remove categorical safety claims and strengthen observable/source-learning guidance."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V316=DATA/"manual_reviews/context_merit_audit_v316";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V316),str(V290)]
import build_context_merit_audit_v316 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v317.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v317.jsonl";REPORT=OUT_DIR/"report_context_merit_v317.json";BASELINE_ROWS=549;BASELINE_SHA256="9ff8bfd6ddfe4910d8181cb128076eafe7ab78905829cbcecdb9a6d997ee6c1d";EXPECTED_OUTPUT_SHA256="570be41f1bc1e9f26883387a4c543b144a4d5d412cbf677cd37addde6a8fc1e0"
EXPECTED_CAPACITY_BEFORE={"conflict_units":258,"equipment_material":22,"resources_general":82,"safety_consent":86,"technique":68};EXPECTED_CAPACITY_AFTER={"conflict_units":258,"equipment_material":23,"resources_general":83,"safety_consent":84,"technique":68}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V316/"context_merit_audit_v316.jsonl",V316/"pending_curation_context_merit_v316.jsonl",V316/"report_context_merit_v316.json")
SPECS=(
 {"action":"drop","fact_id":"fact-af7a364f6458ae53a75e","active_index":120,"expected_question":"How should arm tension be set in Rope365's Angel Tie, and why?","expected_answer":"Keep the tension on the arms very loose; Rope365 says this is necessary to make the tie safer.","reason_code":"drop_loose_tension_makes_tie_safe_claim","reason":"A one-line pattern caption cannot establish that very loose arm tension makes this demanding elbow position safe; stronger dataset rows teach range-of-motion assessment, pain boundaries, monitoring, and rapid exit."},
 {"action":"edit","fact_id":"fact-4bc645e666b38cccde6a","active_index":291,"expected_question":"What lower-leg safety guidance does Rope365 give for a pole trapped behind the knees?","expected_answer":"Avoid high compression at the top of the back of the calf and monitor foot movement because nerves there are vulnerable.","question":"What observable precautions does Rope365 give when a pole is trapped behind the knees?","answer":"Avoid high compression at the upper back of the calf and keep monitoring the tied person's foot movement.","reason_code":"replace_nerve_claim_with_observable_lower_leg_checks","reason":"The replacement preserves the evidence's directly actionable pressure and movement checks without turning its anatomy assertion into the answer."},
 {"action":"drop","fact_id":"fact-5d08494bafee9ca3ad51","active_index":306,"expected_question":"What placement rule does Rope365 give for cuffs used with quick releases?","expected_answer":"Place every cuff in a safe area away from joints and nerves.","reason_code":"drop_vague_safe_cuff_area_tautology","reason":"The row labels an unspecified body area safe without teaching how to assess it; retained rows give specific partner-aware cuff placement, range-of-motion, pressure, and monitoring guidance."},
 {"action":"edit","fact_id":"fact-853b83ba7ebb436c37c3","active_index":508,"expected_question":"Why does Rope365 recommend critical thinking when using rope resources?","expected_answer":"Rope is a rather recent practice in history and we are constantly learning to make it safer.","question":"How does Rope365 recommend combining in-person and online rope learning?","answer":"Use in-person learning for tactile subtleties when available, use books and videos to explore and get started, and evaluate every resource critically because rope knowledge continues to evolve.","reason_code":"expand_resource_fragment_to_learning_method","reason":"The replacement captures the evidence's practical division of labor among in-person, book, and video learning plus its critical-thinking instruction."},
 {"action":"drop","fact_id":"fact-4eb9f483117417868619","active_index":515,"expected_question":"Why does Rope365 say rope material choice can affect bondage safety?","expected_answer":"some can be unsafe to use for bondage as they contain chemicals that are bad for the skin","reason_code":"drop_unspecified_bad_chemicals_claim","reason":"The source names neither a material nor a chemical and provides no way to evaluate the claim; retained material rows give specific construction, handling, and care tradeoffs."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v316 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v317-input";d.mkdir(parents=True,exist_ok=True);base=d/"v316.jsonl";build_baseline(base,d/"v316.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v317-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v317-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v316.jsonl";build_baseline(base,d/"v316.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  common={"action":s["action"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v317","source_lineage":active["source_lineage"]}
  if s["action"]=="edit":common.update(answer=s["answer"],paraphrase_rationale=s["reason"],question=s["question"],support_type="manual_paraphrase")
  curations.append(common);audit={"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":s["action"],"document_sha256":active["document_sha256"],"fact_id":active["fact_id"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"categorical_safety_and_resource_learning_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v317","schema":"context-merit-audit-v317","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]}
  if s["action"]=="edit":audit.update(edited_answer=s["answer"],edited_question=s["question"],paraphrase_rationale=s["reason"])
  audits.append(audit)
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(546,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":3,"edit":2,"keep":0},"path":portable(AUDIT),"rows":5,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":5,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v317","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
