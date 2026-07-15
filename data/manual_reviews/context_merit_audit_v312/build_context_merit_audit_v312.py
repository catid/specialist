#!/usr/bin/env python3
"""Replace three fragmentary train-only QAs with evidence-complete answers."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V311=DATA/"manual_reviews/context_merit_audit_v311";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V311),str(V290)]
import build_context_merit_audit_v311 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v312.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v312.jsonl";REPORT=OUT_DIR/"report_context_merit_v312.json";BASELINE_ROWS=557;BASELINE_SHA256="a2aea845d6b13e0b7cd15ae9577eae88e3461e481cf5fa37eb7e31fb1681a00d";EXPECTED_OUTPUT_SHA256="8f80df0f54216511facd3e332a05350231d6f149655165ab08a2af72d19ae10d"
EXPECTED_CAPACITY={"before":{"conflict_units":260,"equipment_material":22,"resources_general":82,"safety_consent":89,"technique":67},"after":{"conflict_units":260,"equipment_material":22,"resources_general":82,"safety_consent":90,"technique":66}}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V311/"context_merit_audit_v311.jsonl",V311/"pending_curation_context_merit_v311.jsonl",V311/"report_context_merit_v311.json")
SPECS=(
 {"fact_id":"fact-8b2fae23c8f4d2dfb509","active_index":43,"expected_question":"How did Ken Marcus describe the concept behind the California Club collaboration between Takeshi Nagaike and Ernest Greene?","expected_answer":"East meets West bondage styles","question":"What did the California Club collaboration combine in its “East meets West” concept?","answer":"Ernest Greene contributed Western restraint sensibilities using leather, metal, and rope, while Takeshi Nagaike contributed traditional Japanese-style bondage.","reason_code":"replace_style_slogan_with_material_contrast","reason":"The original answer was an unexplained slogan; the replacement preserves the concept while stating each collaborator’s source-supported contribution."},
 {"fact_id":"fact-92c953cfdc7f24bfb1f7","active_index":229,"expected_question":"What does Rope365 call the tension between learning patterns and remaining present with a partner?","expected_answer":"paradox of patterns and presence","question":"Why can relying on a remembered rope pattern interfere with presence and communication?","answer":"Recalling past steps or anticipating the finished tie can pull attention away from the partner’s ongoing dialogue and toward the pattern.","reason_code":"replace_paradox_label_with_attention_tradeoff","reason":"The original answer named the paradox without explaining it; the replacement teaches the evidence’s partner-attention tradeoff."},
 {"fact_id":"fact-65614fa924123cdb0e03","active_index":296,"expected_question":"What lower-risk rope-handling method does Rope365 recommend to avoid friction burns?","expected_answer":"pulling the rope gently","question":"What two rope-handling precautions does Rope365 give for avoiding friction burns?","answer":"Pull the rope gently, or place a hand between the rope and skin when pulling quickly.","reason_code":"complete_friction_burn_precautions","reason":"The original answer omitted the source’s second handling precaution; the replacement preserves both supported options."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v311 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v312-input";d.mkdir(parents=True,exist_ok=True);base=d/"v311.jsonl";build_baseline(base,d/"v311.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v312-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v312-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v311.jsonl";build_baseline(base,d/"v311.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  common={"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v312","source_lineage":active["source_lineage"]}
  curations.append({"action":"edit","answer":s["answer"],"paraphrase_rationale":s["reason"],"question":s["question"],"support_type":"manual_paraphrase",**common})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"fragment_to_evidence_complete_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v312","schema":"context-merit-audit-v312","source":active["source"],"source_support":"manual_paraphrase","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(557,612) or o["before"]!=EXPECTED_CAPACITY["before"] or o["after"]!=EXPECTED_CAPACITY["after"]:raise ValueError(f"projection drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v312","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
