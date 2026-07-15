#!/usr/bin/env python3
"""Expand two evidence-rich fragments and drop a redundant origin lookup."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V312=DATA/"manual_reviews/context_merit_audit_v312";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V312),str(V290)]
import build_context_merit_audit_v312 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v313.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v313.jsonl";REPORT=OUT_DIR/"report_context_merit_v313.json";BASELINE_ROWS=557;BASELINE_SHA256="8f80df0f54216511facd3e332a05350231d6f149655165ab08a2af72d19ae10d";EXPECTED_OUTPUT_SHA256="6f163129d8775b672369a0b297a87f3a3307251ef56ca0b19acc41dabe3e97f5";DIAMETER_SOURCE=DATA/"raw/esinem_a94ba59553f0cc78.json"
EXPECTED_CAPACITY_BEFORE={"conflict_units":260,"equipment_material":22,"resources_general":82,"safety_consent":90,"technique":66};EXPECTED_CAPACITY_AFTER={"conflict_units":260,"equipment_material":22,"resources_general":82,"safety_consent":90,"technique":66}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V312/"context_merit_audit_v312.jsonl",V312/"pending_curation_context_merit_v312.jsonl",V312/"report_context_merit_v312.json")
SPECS=(
 {"action":"edit","fact_id":"fact-823afcbda61e35fbecc3","active_index":77,"expected_question":"How does Rope365 define weaving when lines cross perpendicularly?","expected_answer":"alternating going over and under","question":"How does Rope365 explain why weaving becomes structurally more solid as more rope is added?","answer":"Perpendicular lines alternate over and under; although individual ropes can move, their combined friction becomes more solid as more lines are added.","reason_code":"replace_weaving_fragment_with_structural_mechanism","reason":"The original answer retained only the over-under definition; the replacement includes the evidence’s explanation of how combined friction increases solidity."},
 {"action":"drop","fact_id":"fact-73ddc1423ab531baad8e","active_index":181,"expected_question":"What conclusion does the quoted researcher reach about when kinbaku gained its modern erotic meaning?","expected_answer":"it is an open question","reason_code":"drop_origin_uncertainty_fragment_duplicate","reason":"This five-word conclusion duplicates the richer same-document row fact-79e0566755b06b99034c, which explains the 1874 evidence, possible Taishō-era shift, and incomplete archives."},
 {"action":"edit","fact_id":"fact-9aa934300c8f8a874c3f","active_index":497,"expected_question":"Why does Esinem call rope-diameter measurement more art than science?","expected_answer":"so many factors influence diameter","question":"Which processing variables does ESINEM say can change a rope’s measured diameter?","answer":"Moisture, wet versus dry processing, drying tension, agitation, temperature and duration, additives such as conditioner, and rope construction can all affect diameter.","reason_code":"replace_diameter_fragment_with_named_variables","reason":"The original answer merely said many factors matter; the replacement names the variables listed in the fully reviewed source."},
)
DIAMETER_MARKERS=("As I have always said, rope diameter measurement is more an art than a science since so many factors influence diameter. I’m guessing moisture content must be a factor since dry treatment seems to shrink the diameter. As for wet treatment, the results can be even more variable. Some factors that might come into play are:","- Amount of tension when drying. The more you apply, the more it will stretch and reduce in diameter.","- Amount of agitation. I suspect that more aggressive treatments like washing machines tend to relax the rope and open the plies increasing diameter.","- Temperature and duration of washing are likely to have an effect.","- Additives such as conditioner are likely to relax the rope and open the plies increasing diameter.","How rigorous you make the wet stage will depend upon the construction of the rope. Tightly twisted double yarn ropes can take more aggressive treatment that would kill a loose laid single yarn rope.")
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v312 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v313-input";d.mkdir(parents=True,exist_ok=True);base=d/"v312.jsonl";build_baseline(base,d/"v312.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v313-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v313-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v312.jsonl";build_baseline(base,d/"v312.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};diameter=json.loads(DIAMETER_SOURCE.read_text())
 if diameter["url"]!="https://www.esinem.com/rope-diameters-and-treatment/" or text_sha256(diameter["text"])!="992ad10f5eb0227e3fda5935db5ede8b280941663fa056117d96e904f8fec396":raise ValueError("diameter source drift")
 diameter_lines=diameter["text"].splitlines();support=[]
 for marker in DIAMETER_MARKERS:
  matches=[line for line in diameter_lines if marker in line]
  if len(matches)!=1:raise ValueError(f"diameter evidence drift: {marker}")
  support.append(matches[0])
 diameter_evidence="\n".join(support);audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=diameter_evidence if s["fact_id"]=="fact-9aa934300c8f8a874c3f" else active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  common={"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v313","source_lineage":active["source_lineage"]}
  curation={"action":s["action"],**common}
  if s["action"]=="edit":curation.update(answer=s["answer"],paraphrase_rationale=s["reason"],question=s["question"],support_type="manual_paraphrase")
  curations.append(curation);audit={"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":s["action"],"document_sha256":active["document_sha256"],"fact_id":active["fact_id"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"misleading_fragment_and_duplicate_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v313","schema":"context-merit-audit-v313","source":active["source"],"source_document":portable(DIAMETER_SOURCE) if s["fact_id"]=="fact-9aa934300c8f8a874c3f" else None,"source_support":"manual_paraphrase" if s["action"]=="edit" else "same_document_duplicate_analysis","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]}
  if s["action"]=="edit":audit.update(edited_answer=s["answer"],edited_question=s["question"],paraphrase_rationale=s["reason"])
  audits.append(audit)
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(556,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":1,"edit":2,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v313","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
