#!/usr/bin/env python3
"""Name the work in three otherwise contextless train-only questions."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V353=DATA/"manual_reviews/context_merit_audit_v353";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V353),str(V290)]
import build_context_merit_audit_v353 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v354.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v354.jsonl";REPORT=OUT_DIR/"report_context_merit_v354.json";BASELINE_ROWS=531;BASELINE_SHA256="e85f7aa241edfa4bb370731516a72aec1fba2a99694af38bcfa78c8b3bc164b5";EXPECTED_OUTPUT_SHA256="5f7c2cde761757865661b694d1c67a2ba3cecf4d22893646dea835c554006734"
EXPECTED_CAPACITY_BEFORE={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71};EXPECTED_CAPACITY_AFTER={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V353/"context_merit_audit_v353.jsonl",V353/"pending_curation_context_merit_v353.jsonl",V353/"report_context_merit_v353.json")
SPECS=(
 {"fact_id":"fact-aad5a532cfa52fa1bf1b","active_index":25,"expected_question":"Does the source say a harmed person is obligated to confront the rope partner?","expected_answer":"No. A harmed person is not obligated to confront the partner, especially if doing so could cause mental or emotional pain, fear, or other harm.","question":"According to Kinbaku Today’s “Stop the Gossiping and Griping,” is a harmed person obligated to confront the rope partner?","answer":"No. A harmed person is not obligated to confront the partner, especially if doing so could cause mental or emotional pain, fear, or other harm.","reason_code":"name_work_in_confrontation_obligation_question","reason":"The replacement names the article so the attribution remains meaningful when the Q&A is presented without hidden source metadata."},
 {"fact_id":"fact-88d431554c7babee6617","active_index":102,"expected_question":"How does the author define kokoro and katachi in tying?","expected_answer":"Kokoro is the “why” of tying, while katachi is the “how.”","question":"In “The Evolution of Rope,” how are kokoro and katachi defined in tying?","answer":"Kokoro is the “why” of tying, while katachi is the “how.”","reason_code":"name_work_in_kokoro_katachi_question","reason":"The replacement removes an unresolvable 'the author' reference by naming the source article."},
 {"fact_id":"fact-6544017636543c6e0c62","active_index":103,"expected_question":"How does the author distinguish competence from knowledge in rope teaching?","expected_answer":"Competence is important for keeping your partner safe. Knowledge is important, even essential, for teaching others how to become competent.","question":"In “On Not Teaching Rope,” how are competence and knowledge distinguished?","answer":"Competence is important for keeping your partner safe. Knowledge is important, even essential, for teaching others how to become competent.","reason_code":"name_work_in_competence_knowledge_question","reason":"The replacement names the essay and removes context-dependent author wording while preserving the substantive comparison."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v353 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v354-input";d.mkdir(parents=True,exist_ok=True);base=d/"v353.jsonl";build_baseline(base,d/"v353.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v354-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v354-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v353.jsonl";build_baseline(base,d/"v353.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v354","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"standalone_question_context_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v354","schema":"context-merit-audit-v354","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(531,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v354","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
