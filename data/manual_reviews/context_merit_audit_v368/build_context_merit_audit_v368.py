#!/usr/bin/env python3
"""Name sources in three standalone train-only history and concept Q&A rows."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V367=DATA/"manual_reviews/context_merit_audit_v367";V290=DATA/"manual_reviews/context_merit_audit_v290";sys.path[:0]=[str(ROOT),str(V367),str(V290)]
import build_context_merit_audit_v367 as previous
import build_context_merit_audit_v290 as core
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v368.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v368.jsonl";REPORT=OUT_DIR/"report_context_merit_v368.json";BASELINE_ROWS=531;BASELINE_SHA256="359599ab2f7edd2d0028429cb6a35e0cb08d3a7105b2c618ddc78b8586415bf1";EXPECTED_OUTPUT_SHA256="eadc2799aa01292e8bd41f561a8eb9f601b6c7f86fbe8f372bccf1ae7aaf59cc"
EXPECTED_CAPACITY_BEFORE={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71};EXPECTED_CAPACITY_AFTER={"conflict_units":259,"equipment_material":23,"resources_general":84,"safety_consent":81,"technique":71}
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;file_sha256=previous.file_sha256;text_sha256=previous.text_sha256;read_jsonl=previous.read_jsonl;write_jsonl=previous.write_jsonl;conservative_capacity=previous.conservative_capacity;portable=previous.portable;PRIOR=(V367/"context_merit_audit_v367.jsonl",V367/"pending_curation_context_merit_v367.jsonl",V367/"report_context_merit_v367.json")
SPECS=(
 {"fact_id":"fact-91b622bc112a7e651abf","active_index":101,"expected_question":"How does the article distinguish hazukashii from humiliation in Yukimura-style rope?","expected_answer":"Hazukashii arises internally from the tied person's emotional response, whereas humiliation results from an externally imposed act.","question":"How does “Hazukashii and Humiliation in Yukimura-ryu” distinguish hazukashii from humiliation?","answer":"Hazukashii arises internally from the tied person’s emotional response, whereas humiliation results from an externally imposed act.","reason_code":"name_work_in_hazukashii_humiliation_contrast","reason":"The replacement names the article and normalizes the possessive while preserving its internal-versus-external distinction."},
 {"fact_id":"fact-2a0097650d3c5d707af9","active_index":106,"expected_question":"How does the source connect Naka Akira to Nureki and Sugiura Norio?","expected_answer":"It describes Naka as Nureki’s deshi who further developed the style and worked closely with Sugiura Norio.","question":"According to “Nureki Chimuo: World of Rope,” how is Naka Akira connected to Nureki and Sugiura Norio?","answer":"It describes Naka as Nureki’s deshi who further developed the style and worked closely with Sugiura Norio.","reason_code":"name_work_in_naka_lineage_question","reason":"The replacement names the article so its lineage and collaboration claim remains interpretable without hidden source metadata."},
 {"fact_id":"fact-2b4db2bb622ec33b0baf","active_index":156,"expected_question":"What 1948 volume by Itoh Seiyu does the source discuss in relation to Edo torture-rope imagery?","expected_answer":"Nihon keibatsu fūzoku zukan 日本刑罰風俗図史","question":"In “Edo Is Not Dead,” which 1948 volume by Itoh Seiyu is discussed in relation to Edo torture-rope imagery?","answer":"The volume is Nihon keibatsu fūzoku zukan (日本刑罰風俗図史).","reason_code":"name_work_and_complete_itoh_volume_answer","reason":"The replacement names the article and turns the title fragment into a complete bibliographic answer."},
)
def build_baseline(out,report):
 previous.build_projection(out,report)
 if (len(read_jsonl(out)),file_sha256(out))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v367 drift")
def build_projection(out,report):
 d=out.parent/f".{out.name}.v368-input";d.mkdir(parents=True,exist_ok=True);base=d/"v367.jsonl";build_baseline(base,d/"v367.report.json");core.build_projection_with_inputs(out,report,(CURATION,),(base,))
def observe(before):
 with tempfile.TemporaryDirectory(prefix=".v368-observe-",dir=OUT_DIR) as t:
  d=Path(t);out=d/"out.jsonl";rep=d/"out.report.json";ds=[];rs=[]
  for _ in (1,2):build_projection(out,rep);ds.append(out.read_bytes());rs.append(rep.read_bytes())
  rows=read_jsonl(out);return{"rows":len(rows),"sha":hashlib.sha256(ds[0]).hexdigest(),"eval":json.loads(rs[0])["eval_fact_count"],"de":ds[0]==ds[1],"re":rs[0]==rs[1],"before":conservative_capacity(before),"after":conservative_capacity(rows)}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v368-base-",dir=OUT_DIR) as t:d=Path(t);base=d/"v367.jsonl";build_baseline(base,d/"v367.report.json");before=read_jsonl(base)
 by_fact={r["fact_id"]:(i,r) for i,r in enumerate(before,1)};audits=[];curations=[]
 for audit_index,s in enumerate(SPECS,1):
  index,active=by_fact[s["fact_id"]]
  if index!=s["active_index"] or (active["question"],active["answer"])!=(s["expected_question"],s["expected_answer"]):raise ValueError(f"candidate drift {s['fact_id']}")
  evidence=active.get("evidence")
  if not evidence:raise ValueError("missing evidence")
  curations.append({"action":"edit","answer":s["answer"],"document_sha256":active["document_sha256"],"evidence":evidence,"evidence_url":active["url"],"expected_answer":active["answer"],"expected_question":active["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"question":s["question"],"reason":s["reason"],"reason_code":s["reason_code"],"reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v368","source_lineage":active["source_lineage"],"support_type":"manual_paraphrase"})
  audits.append({"active_answer":active["answer"],"active_index":index,"active_question":active["question"],"audit_index":audit_index,"decision":"edit","document_sha256":active["document_sha256"],"edited_answer":s["answer"],"edited_question":s["question"],"fact_id":active["fact_id"],"paraphrase_rationale":s["reason"],"projection_lineage":{"active_index":index,"baseline_rows":BASELINE_ROWS,"baseline_sha256":BASELINE_SHA256},"reason":s["reason"],"reason_code":s["reason_code"],"review_pass":"standalone_history_concept_source_context_train_only_cleanup","reviewed_at":"2026-07-15","reviewer":"codex-context-merit-audit-v368","schema":"context-merit-audit-v368","source":active["source"],"source_support":"manual_source_and_dataset_context_review","support_evidence":evidence,"support_evidence_sha256":text_sha256(evidence),"url":active["url"]})
 write_jsonl(AUDIT,audits);write_jsonl(CURATION,curations);o=observe(before)
 if not o["de"] or not o["re"] or (o["rows"],o["eval"])!=(531,612) or o["before"]!=EXPECTED_CAPACITY_BEFORE:raise ValueError(f"projection drift {o}")
 if EXPECTED_CAPACITY_AFTER!="PENDING" and o["after"]!=EXPECTED_CAPACITY_AFTER:raise ValueError(f"capacity drift {o}")
 if EXPECTED_OUTPUT_SHA256!="PENDING" and o["sha"]!=EXPECTED_OUTPUT_SHA256:raise ValueError("output drift")
 REPORT.write_text(json.dumps({"audit":{"by_decision":{"drop":0,"edit":3,"keep":0},"path":portable(AUDIT),"rows":3,"sha256":file_sha256(AUDIT)},"conservative_capacity":{"after":o["after"],"before":o["before"],"delta":{k:o["after"][k]-o["before"][k] for k in o["before"]},"grouping":"shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},"prior_checkpoint":{"candidate":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"artifacts":[{"path":portable(p),"sha256":file_sha256(p)} for p in PRIOR]},"isolated_build_projection":{"automated_projection_runs":2,"new_additions_applied":0,"output_rows":o["rows"],"output_sha256":o["sha"],"repeat_dataset_byte_identical":o["de"],"repeat_projection_report_byte_identical":o["re"],"sealed_eval_fact_count_reported_by_tooling":o["eval"]},"new_pending_curation":{"decisions":3,"path":portable(CURATION),"sha256":file_sha256(CURATION)},"schema":"context-merit-audit-report-v368","sealed_evaluation_policy":{"automated_collision_tool_reads_sealed_content":True,"automated_read_scope":"fact-ID collision exclusion and aggregate eval_fact_count reporting only","manual_worker_opened_eval_or_heldout_content":False,"manual_worker_received_eval_or_heldout_content":False}},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
