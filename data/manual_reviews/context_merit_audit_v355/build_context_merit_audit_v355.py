#!/usr/bin/env python3
"""Name source works in three related standalone train-only history questions."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V354=DATA/"manual_reviews/context_merit_audit_v354";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V354),str(V290)]
import build_context_merit_audit_v354 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v355.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v355.jsonl";REPORT=OUT_DIR/"report_context_merit_v355.json";BASELINE_ROWS=531;BASELINE_SHA256="5f7c2cde761757865661b694d1c67a2ba3cecf4d22893646dea835c554006734";EXPECTED_OUTPUT_SHA256="9c342089cd61301c5a4b32ebbdd6f57aa469b6061b3c256d8ced8ca178497413"
EXPECTED_CAPACITY_BEFORE={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71};EXPECTED_CAPACITY_AFTER={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V354/"context_merit_audit_v354.jsonl",V354/"pending_curation_context_merit_v354.jsonl",V354/"report_context_merit_v354.json")
SPECS=(
 {"fact_id":"fact-8556c2474cb4f55f403e","active_index":99,"expected_question":"How does the article characterize Tsujimura's kinbaku-bi in relation to capture-rope techniques?","expected_answer":"It presents kinbaku-bi as a shift for which capture-rope techniques are insufficient because they do not foreground the beauty of the bound body.","question":"In “Exploring the Origins of Kinbaku,” how is Tsujimura’s kinbaku-bi related to capture-rope techniques?","answer":"It presents kinbaku-bi as a shift for which capture-rope techniques are insufficient because they do not foreground the beauty of the bound body.","reason_code":"name_work_in_tsujimura_kinbaku_bi_question","reason":"The replacement names the article and removes a context-dependent reference while preserving the technical-aesthetic comparison."},
 {"fact_id":"fact-3554c6493c5472730261","active_index":106,"expected_question":"How does the source connect Minomura Kou and Nureki Chimuo through Uramado?","expected_answer":"Minomura began Uramado, which Nureki later edited.","question":"According to “Nureki, Kinbiken, and the Aesthetics of Kinbaku,” how did Uramado connect Minomura Kou and Nureki Chimuo?","answer":"Minomura began Uramado, which Nureki later edited.","reason_code":"name_work_in_uramado_connection_question","reason":"The replacement names the work so the requested historical connection is self-contained outside source metadata."},
 {"fact_id":"fact-8220ca85fcd68cc1b74f","active_index":108,"expected_question":"How does the source ultimately characterize hojojutsu’s relationship to modern kinbaku?","expected_answer":"one influence among many: not a stable lineage, but a touchstone","question":"How does “Exploring the Origins of Kinbaku” ultimately characterize Hojōjutsu’s relationship to modern kinbaku?","answer":"It characterizes Hojōjutsu as one influence among many—a touchstone rather than a stable lineage.","reason_code":"name_work_and_complete_hojojutsu_influence_answer","reason":"The replacement names the article and turns the source fragment into a complete, carefully qualified answer."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v354 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v355-input";d.mkdir(parents=True,exist_ok=True);base=d/"v354.jsonl";build_baseline(base,d/"v354.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v355-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v355-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v354.jsonl";build_baseline(base,d/"v354.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v355","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"standalone_history_question_context_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v355","schema":"context-merit-audit-v355","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(531,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v355","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
