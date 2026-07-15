#!/usr/bin/env python3
"""Replace nerve-name trivia with practical placement guidance and remove duplicates."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V314=DATA/"manual_reviews/context_merit_audit_v314";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V314),str(V290)]
import build_context_merit_audit_v314 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v315.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v315.jsonl";REPORT=OUT_DIR/"report_context_merit_v315.json";BASELINE_ROWS=553;BASELINE_SHA256="5c5c72d619a28a88a66fe0828a402efff21f17539f73a082f073d076efe23ae4";EXPECTED_OUTPUT_SHA256="ce5158eb24e7df07089a66f9b0f28b515c3371e83080178e485024cfbb0a181a"
EXPECTED_CAPACITY_BEFORE={"conflict_units":258,"equipment_material":22,"resources_general":81,"safety_consent":89,"technique":66};EXPECTED_CAPACITY_AFTER={"conflict_units":258,"equipment_material":22,"resources_general":82,"safety_consent":87,"technique":67}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V314/"context_merit_audit_v314.jsonl",V314/"pending_curation_context_merit_v314.jsonl",V314/"report_context_merit_v314.json")
SPECS=(
 {"action":"drop","fact_id":"fact-552093f3c34a13a156e5","active_index":174,"expected_question":"What caution does Rope365 give when adding arm cuffs to a box-tie structure?","expected_answer":"We have to be very careful as we tie closer to the elbows","reason_code":"drop_redundant_box_tie_caution_fragment","reason":"This incomplete caution is superseded by the same-document placement-and-tension row edited in this checkpoint."},
 {"action":"edit","fact_id":"fact-7dd1e04f22794215eeac","active_index":306,"expected_question":"What placement precaution does the source give after locating the common peroneal nerve near the knee?","expected_answer":"keep the ropes on your muscles, away from the boney structures around your knee caps","question":"What simpler rope-placement precaution does Rope365 give for the area around a bent knee?","answer":"Keep rope on muscular areas and away from bony structures around the kneecap.","reason_code":"replace_lay_nerve_localization_with_placement_rule","reason":"The replacement preserves the source's actionable placement rule without prompting learners to palpate or identify a named nerve."},
 {"action":"edit","fact_id":"fact-2042b3fa68e8136a183a","active_index":448,"expected_question":"Which nerves does Rope365 warn are harder to avoid when placing upper-arm cuffs?","expected_answer":"the radial nerve on the top of the arm, and the ulnar nerve as we tie closer to the elbow","question":"What placement-and-tension goal does Rope365 give for upper-arm cuffs in a box tie?","answer":"Place the cuff carefully, especially nearer the elbow, and use enough tension to hold it in place without digging into sensitive areas.","reason_code":"replace_nerve_name_lookup_with_cuff_safety_goal","reason":"The replacement retains the source's useful cuff-placement and tension guidance instead of training a standalone nerve-name lookup."},
 {"action":"drop","fact_id":"fact-47bc14300e96b90af0b4","active_index":463,"expected_question":"Which three major upper-limb nerves does the surgeon quoted by Esinem identify near the upper-cinch region?","expected_answer":"The radial, median, and ulnar nerves.","reason_code":"drop_duplicate_upper_limb_anatomy_lookup","reason":"This anatomy quiz duplicates the same document's retained practical warning that the upper-cinch region can compress major nerves against the upper-arm bone."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v314 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v315-input";d.mkdir(parents=True,exist_ok=True);base=d/"v314.jsonl";build_baseline(base,d/"v314.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v315-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v315-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v314.jsonl";build_baseline(base,d/"v314.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  common={"action":s["action"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v315","source_lineage":active["source_lineage"]}
  if s["action"]=="edit":common.update(answer=s["answer"],paraphrase_rationale=s["reason"],question=s["question"],support_type="manual_paraphrase")
  curations.append(common);audit={"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":s["action"],"document_sha256":active["document_sha256"],"fact_id":active["fact_id"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"medical_trivia_and_fragment_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v315","schema":"context-merit-audit-v315","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]}
  if s["action"]=="edit":audit.update(edited_answer=s["answer"],edited_question=s["question"],paraphrase_rationale=s["reason"])
  audits.append(audit)
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(551,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":2,"edit":2,"keep":0},"path":portable(AUDIT),"rows":4,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":4,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v315","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
