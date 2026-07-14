#!/usr/bin/env python3
"""Semantic-redundancy and resource-question naturalness audit v57."""
from __future__ import annotations
import contextlib,json,sys,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data"
V56_DIR=DATA/"manual_reviews/context_merit_audit_v56";sys.path[:0]=[str(ROOT),str(V56_DIR)]
import build_context_merit_audit_v56 as previous
BASE,CORE=previous.BASE,previous.CORE;EVIDENCE_PATCH_MODULE=previous.EVIDENCE_PATCH_MODULE
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v57.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v57.jsonl";REPORT=OUT_DIR/"report_context_merit_v57.json"
REVIEWER,REVIEWED_AT="codex-context-merit-audit-v57","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION
file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl
CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,57))
CONTEXT_AUDITS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"context_merit_audit_v{v}.jsonl" for v in range(1,57))
def raw(n):return DATA/"raw"/n

SPECS=(
 {"fact_id":"fact-c0cc30ba0b937b0f11f1","source_path":raw("kinbakutoday_cbbff84b319d0813.json"),"marker":"the prevention of escape their primary object","decision":"drop","reason_code":"duplicate_hojojutsu_escape_purpose","reason":"The same attributed answer is already retained in a more informative question contrasting restraint with kinbaku's aesthetic aim."},
 {"fact_id":"fact-eb477d063e72bae0adb8","source_path":raw("anatomiestudio_27ecdd4d7c9a5560.json"),"marker":"Suspension is edge play","decision":"keep","reason_code":"retain_clear_studio_suspension_classification","reason":"The studio-attributed classification is concise safety context and is clearer than the redundant generic term lookup removed in this tranche."},
 {"fact_id":"fact-ba0bcde9e13c55a82239","source_path":raw("esinem_77368ccdc66acad0.json"),"marker":"Rope suspension falls under the term “edge play” because it is inherently dangerous","decision":"drop","reason_code":"duplicate_edge_play_term_lookup","reason":"The one-word term lookup duplicates the clearer Anatomie Studio classification retained in this tranche; the longer ESINEM account remains represented by other substantive facts."},
 {"fact_id":"fact-64a4807147c057265799","source_path":raw("rope_resources_v1/rope365__b602b6493b5eb6f55206.json"),"marker":"How does my body feel?” and “In what emotional state am I?","decision":"drop","reason_code":"duplicate_pre_scene_self_scan","reason":"An earlier edited row already asks the same two Rope365 readiness questions and includes the same answer with only an introductory phrase added."},
 {"fact_id":"fact-7d3c573dece9b1f178e2","source_path":DATA/"rope_topia_manual_v1.jsonl","marker":"<loc>https://rope-topia.com/portfolio-items/joining-rope/</loc>","allow_document_sha_mismatch":True,"decision":"edit","question":"Where can I find Rope-topia’s Joining Rope tutorial?","answer":"https://rope-topia.com/portfolio-items/joining-rope/","reason_code":"naturalize_joining_rope_resource_lookup","reason":"The edit preserves the sitemap-backed tutorial URL while replacing manifest-oriented wording with a natural resource question."},
 {"fact_id":"fact-f271fc3eb63a3caa0e08","source_path":DATA/"rope_topia_manual_v1.jsonl","marker":"<loc>https://rope-topia.com/portfolio-items/wet-treating-rope/</loc>","allow_document_sha_mismatch":True,"decision":"edit","question":"Where can I find Rope-topia’s Wet treating rope tutorial?","answer":"https://rope-topia.com/portfolio-items/wet-treating-rope/","reason_code":"naturalize_wet_treating_resource_lookup","reason":"The edit preserves the sitemap-backed tutorial URL while replacing manifest-oriented wording with a natural resource question."},
 {"fact_id":"fact-5c67a11fec80968c6494","source_path":DATA/"rope_topia_manual_v1.jsonl","marker":"<loc>https://rope-topia.com/portfolio-items/strugglers-knot/</loc>","allow_document_sha_mismatch":True,"decision":"edit","question":"Where can I find Rope-topia’s Strugglers Knot tutorial?","answer":"https://rope-topia.com/portfolio-items/strugglers-knot/","reason_code":"naturalize_strugglers_knot_resource_lookup","reason":"The edit preserves the sitemap-backed tutorial URL while replacing manifest-oriented wording with a natural resource question."},
 {"fact_id":"fact-9262d7a403e6f990b346","source_path":DATA/"rope_topia_manual_v1.jsonl","marker":"<loc>https://rope-topia.com/portfolio-items/wicked-fast-bowline/</loc>","allow_document_sha_mismatch":True,"decision":"edit","question":"Where can I find Rope-topia’s Wicked Fast Bowline tutorial?","answer":"https://rope-topia.com/portfolio-items/wicked-fast-bowline/","reason_code":"naturalize_wicked_bowline_resource_lookup","reason":"The edit preserves the sitemap-backed tutorial URL while replacing manifest-oriented wording with a natural resource question."},
 {"fact_id":"fact-e9f1e475dd738b456f78","source_path":DATA/"rope_topia_manual_v1.jsonl","marker":"<loc>https://rope-topia.com/rope-bottom-guide/</loc>","allow_document_sha_mismatch":True,"decision":"edit","question":"Where can I find Rope-topia’s Rope Bottom Guide?","answer":"https://rope-topia.com/rope-bottom-guide/","reason_code":"naturalize_rope_bottom_guide_lookup","reason":"The edit preserves the owner-requested guide URL while replacing manifest-oriented wording with a natural resource question."},
 {"fact_id":"fact-97e8d7d04ad0051b1a9c","source_path":DATA/"rope_topia_manual_v1.jsonl","marker":"<loc>https://rope-topia.com/safety-cutters/</loc>","allow_document_sha_mismatch":True,"decision":"edit","question":"Where can I find Rope-topia’s guide to safety cutters?","answer":"https://rope-topia.com/safety-cutters/","reason_code":"naturalize_safety_cutters_resource_lookup","reason":"The edit preserves the owner-requested safety-resource URL while replacing manifest-oriented wording with a natural resource question."},
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES=dict(zip(EXPECTED_SELECTION,(18,53,328,479,432,435,434,436,428,429)))
PROJECTED_SELECTION_BASELINE={"description":"isolated cumulative training projection through context-merit v56","direct_rows_without_prior_curation":237,"semantic_candidates_selected":10,"rows":540,"sha256":"3178dc973f017cb4820223ecbb2a772faa072fc2eddb80fb54b23091f7034b24"}
ISOLATED_PROJECTION={"active_after_context_merit_v56":502,"active_after_this_tranche":499,"build_script":"build_curated_qa.py","new_drops_applied":3,"new_edits_applied":6,"output_rows":537,"output_sha256":"abc8e7286f697141a789e2ae4253ee93db4e744b784ff9757e9690eba299a1c1","prior_pending_addition_fact_ids_preserved":36,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":1,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS,SEALED_EVAL_PATHS=previous.PRODUCTION_INPUTS,previous.SEALED_EVAL_PATHS
PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
def prior_decision_artifacts():
 out=[]
 for v in range(1,57):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def build_projection(o,r,c):previous.build_projection(o,r,c)
def semantic_candidates(rows):
 by_id={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)};ranked=[]
 for fact_id in EXPECTED_SELECTION:
  if fact_id not in by_id:raise ValueError(f"v57 missing semantic candidate {fact_id}")
  i,row=by_id[fact_id];ranked.append({"active_index":i,"row":row,"features":CORE.risk_features(row)})
 if {x["row"]["fact_id"]:x["active_index"] for x in ranked}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v57 candidate index drift")
 return ranked
def selected_ranked(rows):return semantic_candidates(rows),0,0
@contextlib.contextmanager
def patched_base(ds):
 rep={"OUT_DIR":OUT_DIR,"AUDIT":AUDIT,"CURATION":CURATION,"REPORT":REPORT,"REVIEWER":REVIEWER,"REVIEWED_AT":REVIEWED_AT,"CONTEXT_CURATIONS":CONTEXT_CURATIONS,"SPECS":SPECS,"EXPECTED_SELECTION":EXPECTED_SELECTION,"ISOLATED_PROJECTION":ISOLATED_PROJECTION};orig={n:getattr(BASE,n) for n in rep};ranking,active,evidence=CORE.ranked_unreviewed,CORE.ACTIVE_DATASET,EVIDENCE_PATCH_MODULE.source_evidence
 try:
  for n,v in rep.items():setattr(BASE,n,v)
  CORE.ranked_unreviewed=selected_ranked;CORE.ACTIVE_DATASET=ds;EVIDENCE_PATCH_MODULE.source_evidence=previous.previous.previous.previous.previous.previous.previous.previous.previous.source_evidence;yield
 finally:
  EVIDENCE_PATCH_MODULE.source_evidence=evidence;CORE.ACTIVE_DATASET=active;CORE.ranked_unreviewed=ranking
  for n,v in orig.items():setattr(BASE,n,v)
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v56-projection-",dir=OUT_DIR) as t:
  ds=Path(t)/"v56.jsonl";rp=Path(t)/"r.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=540 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v56 projection drift")
  with patched_base(ds):BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:row.update(schema="context-merit-audit-v57",review_pass="semantic_redundancy_and_resource_naturalness_reaudit",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":540,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True})
 write_jsonl(AUDIT,audits);report=json.loads(REPORT.read_text());report["schema"]="context-merit-audit-report-v57"
 report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=540,eligible_unreviewed_rows=0,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=10,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"manual second-pass selection from exact-answer and high-overlap question pairs, plus repeated manifest-oriented resource prompts","score":"manual semantic redundancy and standalone question naturalness","tie_break":"source utility, answer completeness, then owner URL preservation"})
 report["audit"]["rows"]=len(audits);report["audit"]["sha256"]=file_sha256(AUDIT);report["new_pending_curation"]["sha256"]=file_sha256(CURATION);report["new_pending_curation"]["edit_support_types"]={"extractive":6,"manual_paraphrase":0};report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()]
 REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
