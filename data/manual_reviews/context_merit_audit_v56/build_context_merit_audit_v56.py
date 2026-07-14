#!/usr/bin/env python3
"""Audit the final directly curatable v55 projection rows in v56."""
from __future__ import annotations
import contextlib,json,sys,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data"
V55_DIR=DATA/"manual_reviews/context_merit_audit_v55";sys.path[:0]=[str(ROOT),str(V55_DIR)]
import build_context_merit_audit_v55 as previous
BASE,CORE=previous.BASE,previous.CORE;EVIDENCE_PATCH_MODULE=previous.EVIDENCE_PATCH_MODULE
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v56.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v56.jsonl";REPORT=OUT_DIR/"report_context_merit_v56.json"
REVIEWER,REVIEWED_AT="codex-context-merit-audit-v56","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION
file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl
CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,56))
CONTEXT_AUDITS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"context_merit_audit_v{v}.jsonl" for v in range(1,56))
def raw(n):return DATA/"raw"/n

SPECS=(
 {"fact_id":"fact-8f4e200335536704487f","source_path":ROOT/"sources/manual_facts/resource_group_b.jsonl","marker":"a strap and ring cutter for enhanced capabilities in rescue situations","support_type":"manual_paraphrase","paraphrase_support_fragments":["a strap and ring cutter"],"paraphrase_rationale":"The answer expands the coordinated product-listing phrase to make clear that strap cutter and ring cutter are two separate implements.","decision":"keep","reason_code":"rescue_tool_cutter_capabilities","reason":"The attributed answer distinguishes the rescue tool's strap and ring cutters and helps users understand the supplied equipment link."},
 {"fact_id":"fact-d54ba75e58ecb1508618","source_path":raw("wykd_944e4e6d621a97c9.json"),"marker":"unless you actually want to and consent to it","decision":"keep","reason_code":"submissive_instructions_require_want_and_consent","reason":"The answer directly rejects automatic submission obligations and states both desire and consent as requirements."},
 {"fact_id":"fact-a9d68256fa3f2a7aac5d","source_path":raw("kinbakutoday_dfc9527c49ca8ad6.json"),"marker":"creativity, improvisation and a personal touch","decision":"keep","reason_code":"attributed_conformity_creativity_critique","reason":"The explicitly attributed answer captures the essay's substantive critique of insecurity-driven conformity in rope practice."},
 {"fact_id":"fact-5a79fb4622156c02aaaa","source_path":ROOT/"sources/manual_facts/resource_group_b.jsonl","marker":"extended from its lowest height of 2255mm","support_type":"manual_paraphrase","paraphrase_support_fragments":["2255mm (89 Inches)","3480mm (137.00 Inches)","3480mm (11.42ft)","2255mm (7.5ft)"],"paraphrase_rationale":"The answer pairs the manual's prose millimetre/inch dimensions with the corresponding feet labels from the same extracted figure and normalizes spacing and precision.","decision":"keep","reason_code":"manufacturer_frame_clearance_dimensions","reason":"The frame's minimum and maximum heights are useful planning specifications for the owner-supplied equipment resource."},
 {"fact_id":"fact-ed87a03d0c407521412b","source_path":raw("kinbakutoday_c364f23ce34ae761.json"),"marker":"The feeling of the restraint, the feeling of the material on my skin, the physical pressure","decision":"edit","question":"In “When Does the Sex Start?”, which rope sensations does the author describe as deeply arousing and satisfying?","answer":"the feeling of the restraint, the feeling of the material on my skin, the physical pressure","reason_code":"identify_personal_account_in_sensation_question","reason":"The edit replaces a context-dependent author reference with the article title while preserving the exact first-person sensory account."},
 {"fact_id":"fact-ffed9737e22b7deeb020","source_path":raw("kinbakutoday_73b16e835ab63cc2.json"),"marker":"who is my partner and what do they need?","decision":"drop","reason_code":"duplicate_partner_needs_question_after_v54_edit","reason":"The same partner-needs contrast is already retained in the clearer, broader v54 edit, making this row semantically redundant."},
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES=dict(zip(EXPECTED_SELECTION,(21,359,344,113,24,299)))
PROJECTED_SELECTION_BASELINE={"description":"isolated cumulative training projection through context-merit v55","direct_rows_without_prior_curation":239,"eligible_unreviewed_direct_rows":6,"prior_context_reviewed_direct_rows_excluded":233,"rows":541,"sha256":"cbe89653c42b6b0a871f64f7627610ce2edd86bfb30e92f786e4997585ad5cf6"}
ISOLATED_PROJECTION={"active_after_context_merit_v55":503,"active_after_this_tranche":502,"build_script":"build_curated_qa.py","new_drops_applied":1,"new_edits_applied":1,"output_rows":540,"output_sha256":"3178dc973f017cb4820223ecbb2a772faa072fc2eddb80fb54b23091f7034b24","prior_pending_addition_fact_ids_preserved":37,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":4,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS,SEALED_EVAL_PATHS=previous.PRODUCTION_INPUTS,previous.SEALED_EVAL_PATHS
PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
def prior_decision_artifacts():
 out=[]
 for v in range(1,56):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def build_projection(o,r,c):previous.build_projection(o,r,c)
def prior_reviewed_fact_ids():return {r["fact_id"] for p in CONTEXT_AUDITS for r in read_jsonl(p)}
def ranked_unreviewed_direct(rows):
 reviewed=prior_reviewed_fact_ids();c=[]
 for i,row in enumerate(rows,1):
  if row.get("curation") or row["fact_id"] in reviewed:continue
  f=CORE.risk_features(row);c.append((-f["risk_score"],f["question_tokens"],f["answer_tokens"],row["fact_id"],i,row,f))
 c.sort(key=lambda x:x[:4]);ranked=[{"active_index":x[4],"row":x[5],"features":x[6]} for x in c]
 if len(ranked)!=6:raise ValueError(f"v56 candidate drift: {len(ranked)}")
 if tuple(x["row"]["fact_id"] for x in ranked)!=EXPECTED_SELECTION:raise ValueError("v56 selection drift")
 return ranked
def selected_ranked(rows):return ranked_unreviewed_direct(rows),0,0
@contextlib.contextmanager
def patched_base(ds):
 rep={"OUT_DIR":OUT_DIR,"AUDIT":AUDIT,"CURATION":CURATION,"REPORT":REPORT,"REVIEWER":REVIEWER,"REVIEWED_AT":REVIEWED_AT,"CONTEXT_CURATIONS":CONTEXT_CURATIONS,"SPECS":SPECS,"EXPECTED_SELECTION":EXPECTED_SELECTION,"ISOLATED_PROJECTION":ISOLATED_PROJECTION};orig={n:getattr(BASE,n) for n in rep};ranking,active,evidence=CORE.ranked_unreviewed,CORE.ACTIVE_DATASET,EVIDENCE_PATCH_MODULE.source_evidence
 try:
  for n,v in rep.items():setattr(BASE,n,v)
  CORE.ranked_unreviewed=selected_ranked;CORE.ACTIVE_DATASET=ds;EVIDENCE_PATCH_MODULE.source_evidence=previous.previous.previous.previous.previous.previous.previous.previous.source_evidence;yield
 finally:
  EVIDENCE_PATCH_MODULE.source_evidence=evidence;CORE.ACTIVE_DATASET=active;CORE.ranked_unreviewed=ranking
  for n,v in orig.items():setattr(BASE,n,v)
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v55-projection-",dir=OUT_DIR) as t:
  ds=Path(t)/"v55.jsonl";rp=Path(t)/"r.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=541 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v55 projection drift")
  with patched_base(ds):BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:
  row.update(schema="context-merit-audit-v56",review_pass="first_context_merit_review_of_v55_projection_row",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":541,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":False})
  spec=next(s for s in SPECS if s["fact_id"]==row["fact_id"])
  if spec.get("support_type")=="manual_paraphrase":row["paraphrase_rationale"]=spec["paraphrase_rationale"]
 write_jsonl(AUDIT,audits);report=json.loads(REPORT.read_text());report["schema"]="context-merit-audit-report-v56"
 report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=541,eligible_unreviewed_rows=6,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=6,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"the row survives the v55 projection, has no prior curation metadata, and its fact_id has no context-merit decision in v1 through v55","score":"short_question_points + 3*pronoun_count + bare_answer_points + named_person_trivia_points","tie_break":"risk_score descending, question tokens ascending, answer tokens ascending, fact_id ascending"})
 report["audit"]["rows"]=len(audits);report["audit"]["sha256"]=file_sha256(AUDIT);report["new_pending_curation"]["sha256"]=file_sha256(CURATION);report["new_pending_curation"]["edit_support_types"]={"extractive":1,"manual_paraphrase":0};report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()]
 REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
