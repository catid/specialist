#!/usr/bin/env python3
"""Replace safety fragments with direct monitoring and check-in guidance."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V315=DATA/"manual_reviews/context_merit_audit_v315";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V315),str(V290)]
import build_context_merit_audit_v315 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v316.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v316.jsonl";REPORT=OUT_DIR/"report_context_merit_v316.json";BASELINE_ROWS=551;BASELINE_SHA256="ce5158eb24e7df07089a66f9b0f28b515c3371e83080178e485024cfbb0a181a";EXPECTED_OUTPUT_SHA256="9ff8bfd6ddfe4910d8181cb128076eafe7ab78905829cbcecdb9a6d997ee6c1d"
EXPECTED_CAPACITY_BEFORE={"conflict_units":258,"equipment_material":22,"resources_general":82,"safety_consent":87,"technique":67};EXPECTED_CAPACITY_AFTER={"conflict_units":258,"equipment_material":22,"resources_general":82,"safety_consent":86,"technique":68}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V315/"context_merit_audit_v315.jsonl",V315/"pending_curation_context_merit_v315.jsonl",V315/"report_context_merit_v315.json")
SPECS=(
 {"action":"edit","fact_id":"fact-6e4a58ea39fad9a8ef11","active_index":267,"expected_question":"What hand movement does Rope365 recommend monitoring in a box tie for signs of nerve impingement?","expected_answer":"the hands’ ability to close and open","question":"What ongoing checks does Rope365 recommend while someone is in this box-tie structure?","answer":"Watch for pressure on the tops of the shoulders and in the armpits, and keep checking whether the tied person can open and close their hands.","reason_code":"expand_box_tie_fragment_to_ongoing_checks","reason":"The replacement turns a one-phrase diagnostic prompt into the evidence's fuller behavior-based monitoring guidance."},
 {"action":"drop","fact_id":"fact-58150eaa15b1799d1949","active_index":292,"expected_question":"What lower-risk body positions does Rope365 recommend for tying?","expected_answer":"kneeling or sitting","reason_code":"drop_contextless_lower_risk_position_claim","reason":"A single source bullet does not establish that kneeling or sitting is generally lower risk; the dataset already teaches specific floor-level, mobility, balance, and fall precautions."},
 {"action":"edit","fact_id":"fact-f2f58e6ec3deace8cc7d","active_index":342,"expected_question":"What should the shibarite do when they feel something may be wrong?","expected_answer":"ask if the ukete is really ok or not","question":"When the person tying cannot tell whether a partner is comfortable or in too much pain, what does Tessin Doyama recommend?","answer":"Ask the partner directly whether they are okay rather than assuming from appearances.","reason_code":"replace_check_in_fragment_with_direct_partner_check","reason":"The replacement states the evidence's practical check-in behavior clearly and preserves its attribution."},
 {"action":"drop","fact_id":"fact-36db6917e8a0f04f49e1","active_index":427,"expected_question":"Where does Rope365 warn that exposed nerves may be encountered during its capture exercise?","expected_answer":"around the wrists and on the upper arms","reason_code":"drop_capture_nerve_location_lookup","reason":"This bare anatomy-location lookup adds little beyond the same-document retained warning that historical capture techniques deliberately place rope in dangerous areas."},
 {"action":"edit","fact_id":"fact-aea416046b115c8a99fd","active_index":452,"expected_question":"Which pain concerns does Rope365 say to check when trying frog-tie starting points?","expected_answer":"overextension at the knee or on the front of the tibia bone","question":"What comfort check does Rope365 recommend when trying frog-tie starting positions?","answer":"Ask about any pain, especially knee overextension or pressure on the front of the shin.","reason_code":"replace_anatomy_fragment_with_position_comfort_check","reason":"The replacement keeps the source's actionable pain check while avoiding a rote anatomical list."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v315 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v316-input";d.mkdir(parents=True,exist_ok=True);base=d/"v315.jsonl";build_baseline(base,d/"v315.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v316-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v316-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v315.jsonl";build_baseline(base,d/"v315.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  common={"action":s["action"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v316","source_lineage":active["source_lineage"]}
  if s["action"]=="edit":common.update(answer=s["answer"],paraphrase_rationale=s["reason"],question=s["question"],support_type="manual_paraphrase")
  curations.append(common);audit={"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":s["action"],"document_sha256":active["document_sha256"],"fact_id":active["fact_id"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"behavior_based_safety_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v316","schema":"context-merit-audit-v316","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]}
  if s["action"]=="edit":audit.update(edited_answer=s["answer"],edited_question=s["question"],paraphrase_rationale=s["reason"])
  audits.append(audit)
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(549,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":2,"edit":3,"keep":0},"path":portable(AUDIT),"rows":5,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":5,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v316","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
