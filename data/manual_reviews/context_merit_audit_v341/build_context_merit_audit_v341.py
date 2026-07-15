#!/usr/bin/env python3
"""Drop one reversed sheet-bend rule and clarify two valid hitch mechanics answers."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V340=DATA/"manual_reviews/context_merit_audit_v340";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V340),str(V290)]
import build_context_merit_audit_v340 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v341.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v341.jsonl";REPORT=OUT_DIR/"report_context_merit_v341.json";BASELINE_ROWS=529;BASELINE_SHA256="a92f6c97345f7e0a0422584912777e1601bfe130b7c051ff2fb7fced3b2b70d1";EXPECTED_OUTPUT_SHA256="40f4b73c25cccfddc49da039b40483b469b9858deec9d00ca399cb490f5aa47a"
EXPECTED_CAPACITY_BEFORE={"conflict_units":260,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":72};EXPECTED_CAPACITY_AFTER={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V340/"context_merit_audit_v340.jsonl",V340/"pending_curation_context_merit_v340.jsonl",V340/"report_context_merit_v340.json")
SPECS=(
 {"action":"edit","fact_id":"fact-74a8aae072f9d68b8131","active_index":344,"expected_question":"What tradeoff does Rope365 give for the reverse crossing hitch?","expected_answer":"The reverse crossing hitch will hold tension once tied; it is not as solid as other locking hitches but also remains easy to untie","question":"What tradeoff does Rope365 give for the reverse crossing hitch?","answer":"It holds tension once tied and remains easy to untie, but Rope365 says it is less solid than other locking hitches.","reason_code":"complete_reverse_crossing_hitch_tradeoff","reason":"The replacement preserves all three source-stated properties in a complete, explicitly attributed comparison."},
 {"action":"drop","fact_id":"fact-eb156ba7c19b276d4719","active_index":378,"expected_question":"When does Rope365 say a sheet bend is appropriate for joining two ropes?","expected_answer":"two ropes of the same size, or a smaller bight to a larger rope’s running ends","reason_code":"drop_reversed_sheet_bend_size_relationship","reason":"The source places the bight in the smaller rope and the running end in the larger rope. Authoritative rescue references specify the opposite for unequal diameters: form the bight in the larger or thicker rope and tie with the smaller or thinner rope's working end.","external_verification":[{"authority":"Connecticut Fire Academy Recruit Firefighter Program","claim":"For different diameters, place the bight in the larger rope and use the smaller rope's working end.","url":"https://portal.ct.gov/-/media/CFPC/files/NEW-ITEMS-2019/Uploaded-Files/Instructor-Lesson-Plans/Uploaded-Files/Unit-8/SS-822-Instructor-Reference-Material.pdf"},{"authority":"United States rescue-service technical manual","claim":"Make the bight in the thicker rope and pass the thinner rope's end through and around it.","url":"https://www.govinfo.gov/content/pkg/GOVPUB-PR32_4400-9221d4eb219c06dc83dd43d80bbcbb99/pdf/GOVPUB-PR32_4400-9221d4eb219c06dc83dd43d80bbcbb99.pdf"}]},
 {"action":"edit","fact_id":"fact-2bff3d053f54e52b560e","active_index":453,"expected_question":"Why can crossing-hitch direction matter even when differences seem minimal at 90 degrees on a flat plane?","expected_answer":"the forces inside the rope will make the rope shift in different ways","question":"Why can crossing-hitch direction matter even when differences seem minimal at 90 degrees on a flat plane?","answer":"Rope bondage is three-dimensional, so aiming the rope left, right, up, or down changes its internal forces and can make it shift in different ways.","reason_code":"complete_crossing_hitch_direction_mechanics","reason":"The replacement connects the source fragment to its three-dimensional premise and lists only the directional variables named in the evidence."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v340 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v341-input";d.mkdir(parents=True,exist_ok=True);base=d/"v340.jsonl";build_baseline(base,d/"v340.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v341-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v341-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v340.jsonl";build_baseline(base,d/"v340.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  common={"action":s["action"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v341","source_lineage":active["source_lineage"]}
  audit={"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":s["action"],"document_sha256":active["document_sha256"],"fact_id":active["fact_id"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"technical_mechanics_correctness_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v341","schema":"context-merit-audit-v341","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]}
  if s["action"]=="edit":common.update({"answer":s["answer"],"paraphrase_rationale":s["reason"],"question":s["question"],"support_type":"manual_paraphrase"});audit.update({"edited_answer":s["answer"],"edited_question":s["question"],"paraphrase_rationale":s["reason"]})
  else:audit["external_verification"]=s["external_verification"]
  curations.append(common);audits.append(audit)
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(528,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":1,"edit":2,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v341","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
