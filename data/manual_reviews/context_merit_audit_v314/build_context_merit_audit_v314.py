#!/usr/bin/env python3
"""Drop three unsupported standalone anatomy/diagnostic lookups."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V313=DATA/"manual_reviews/context_merit_audit_v313";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V313),str(V290)]
import build_context_merit_audit_v313 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v314.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v314.jsonl";REPORT=OUT_DIR/"report_context_merit_v314.json";BASELINE_ROWS=556;BASELINE_SHA256="6f163129d8775b672369a0b297a87f3a3307251ef56ca0b19acc41dabe3e97f5";EXPECTED_OUTPUT_SHA256="5c5c72d619a28a88a66fe0828a402efff21f17539f73a082f073d076efe23ae4"
EXPECTED_CAPACITY_BEFORE={"conflict_units":260,"equipment_material":22,"resources_general":82,"safety_consent":90,"technique":66};EXPECTED_CAPACITY_AFTER={"conflict_units":258,"equipment_material":22,"resources_general":81,"safety_consent":89,"technique":66}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V313/"context_merit_audit_v313.jsonl",V313/"pending_curation_context_merit_v313.jsonl",V313/"report_context_merit_v313.json")
SPECS=(
 {"fact_id":"fact-ae1c724d00d5c54d02cd","active_index":314,"expected_question":"What pressure does Rope365 say should be avoided around the knee tendons?","expected_answer":"lateral pressure on those tendons","reason_code":"drop_standalone_knee_anatomy_claim","reason":"This turns a non-medical source's broad anatomy paragraph into a standalone tendon-shearing claim; the dataset already teaches the safer behavior-based boundary of assessing range of motion and never pushing a joint toward pain."},
 {"fact_id":"fact-1575d99869e41e868091","active_index":350,"expected_question":"What term does Rope365 use for small spots of bleeding under the skin that may appear when circulation is restricted?","expected_answer":"petechiae","reason_code":"drop_standalone_diagnostic_term","reason":"This isolates a diagnostic term without useful response guidance; the dataset already gives stronger behavior-based circulation monitoring and fast-release instructions."},
 {"fact_id":"fact-053e647b8c55dc038f71","active_index":439,"expected_question":"Which breathing muscle can rope near the lower rib cage restrict by limiting its movement?","expected_answer":"the diaphragm","reason_code":"drop_standalone_breathing_anatomy_claim","reason":"This converts a non-medical source's exploratory passage into a categorical anatomy claim; existing rows more safely teach monitoring breathing and whole-body responses and releasing promptly when needed."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v313 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v314-input";d.mkdir(parents=True,exist_ok=True);base=d/"v313.jsonl";build_baseline(base,d/"v313.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v314-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v314-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v313.jsonl";build_baseline(base,d/"v313.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  common={"action":"drop","document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v314","source_lineage":active["source_lineage"]};curations.append(common)
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"drop","document_sha256":active["document_sha256"],"fact_id":active["fact_id"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"unsupported_medical_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v314","schema":"context-merit-audit-v314","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(553,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":3,"edit":0,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v314","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
