#!/usr/bin/env python3
"""Naturalize the remaining direct Rope-topia resource prompts in v58."""
from __future__ import annotations
import contextlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V57_DIR=DATA/"manual_reviews/context_merit_audit_v57";sys.path[:0]=[str(ROOT),str(V57_DIR)]
import build_context_merit_audit_v57 as previous
BASE,CORE=previous.BASE,previous.CORE;EVIDENCE_PATCH_MODULE=previous.EVIDENCE_PATCH_MODULE
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v58.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v58.jsonl";REPORT=OUT_DIR/"report_context_merit_v58.json";REVIEWER,REVIEWED_AT="codex-context-merit-audit-v58","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION
file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl
CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,58));CONTEXT_AUDITS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"context_merit_audit_v{v}.jsonl" for v in range(1,58))
ROPE_TOPIA=DATA/"rope_topia_manual_v1.jsonl"
def resource(fact_id,url,question,code):return {"fact_id":fact_id,"source_path":ROPE_TOPIA,"marker":f"<loc>{url}</loc>","allow_document_sha_mismatch":True,"decision":"edit","question":question,"answer":url,"reason_code":code,"reason":"The edit preserves the sitemap-backed Rope-topia URL while replacing manifest-oriented wording with a natural resource question."}
SPECS=(
 resource("fact-6b6b6fcbc60678bd5d07","https://rope-topia.com/newcomers-information/out-into-the-kink-community/","Where can I find Rope-topia’s “Getting out in the kink community” page?","naturalize_getting_out_resource_lookup"),
 resource("fact-11af85954681aed5764b","https://rope-topia.com/newcomers-information/identifying-predatory-behaviour/","Where can I find Rope-topia’s “Identifying predatory behaviour” page?","naturalize_predatory_behaviour_resource_lookup"),
 resource("fact-f282a4f4635ebd41890a","https://rope-topia.com/newcomers-information/","Where can I find Rope-topia’s newcomers information hub?","naturalize_newcomers_hub_lookup"),
 resource("fact-a8b8fc2fcc29541ac904","https://rope-topia.com/2013/11/luck-self-awareness-responsibility-rope-bondage-injuries/","Where can I find Rope-topia’s post about luck, self-awareness, responsibility, and rope-bondage injuries?","naturalize_injury_responsibility_post_lookup"),
 resource("fact-1de730232db74a37d37d","https://rope-topia.com/newcomers-information/so-youre-new-to-the-kink-scene/","Where can I find Rope-topia’s “So you’re new to the kink scene” page?","naturalize_new_to_kink_resource_lookup"),
 resource("fact-cca2ce649c4a186e5da3","https://rope-topia.com/nerve-and-circulation-problems/","Where can I find Rope-topia’s “Nerve and Circulation Problems in Shibari” page?","naturalize_nerve_circulation_resource_lookup"),
 resource("fact-4dc3a8b24b77fb1a8d5a","https://rope-topia.com/2012/09/yin-yoga-for-bondage/","Where can I find Rope-topia’s “Yin Yoga for Bondage” post?","naturalize_yin_yoga_resource_lookup"),
 resource("fact-c91029574adc7c6ead1b","https://rope-topia.com/portfolio-items/kinbaku-today-rope-is-not-about-rope/portfolioCats-102-70-123-72-57/","Where can I find Rope-topia’s portfolio article “Rope is not about Rope”?","naturalize_rope_not_about_rope_lookup"),
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES=dict(zip(EXPECTED_SELECTION,(428,429,430,431,432,433,434,438)))
PROJECTED_SELECTION_BASELINE={"description":"isolated cumulative training projection through context-merit v57","direct_rows_without_prior_curation":228,"resource_prompts_selected":8,"rows":537,"sha256":"abc8e7286f697141a789e2ae4253ee93db4e744b784ff9757e9690eba299a1c1"}
ISOLATED_PROJECTION={"active_after_context_merit_v57":499,"active_after_this_tranche":499,"build_script":"build_curated_qa.py","new_drops_applied":0,"new_edits_applied":8,"output_rows":537,"output_sha256":"eb92d933fbd57e615a800c3a03a4177d8390d5b4b84f228ae3e7be74e18fe557","prior_pending_addition_fact_ids_preserved":36,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":0,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS,SEALED_EVAL_PATHS=previous.PRODUCTION_INPUTS,previous.SEALED_EVAL_PATHS;PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
def prior_decision_artifacts():
 out=[]
 for v in range(1,58):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def build_projection(o,r,c):previous.build_projection(o,r,c)
def selected(rows):
 by={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)};out=[{"active_index":by[f][0],"row":by[f][1],"features":CORE.risk_features(by[f][1])} for f in EXPECTED_SELECTION]
 if {x["row"]["fact_id"]:x["active_index"] for x in out}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v58 candidate drift")
 return out
def selected_ranked(rows):return selected(rows),0,0
@contextlib.contextmanager
def patched_base(ds):
 rep={"OUT_DIR":OUT_DIR,"AUDIT":AUDIT,"CURATION":CURATION,"REPORT":REPORT,"REVIEWER":REVIEWER,"REVIEWED_AT":REVIEWED_AT,"CONTEXT_CURATIONS":CONTEXT_CURATIONS,"SPECS":SPECS,"EXPECTED_SELECTION":EXPECTED_SELECTION,"ISOLATED_PROJECTION":ISOLATED_PROJECTION};orig={n:getattr(BASE,n) for n in rep};ranking,active,evidence=CORE.ranked_unreviewed,CORE.ACTIVE_DATASET,EVIDENCE_PATCH_MODULE.source_evidence
 try:
  for n,v in rep.items():setattr(BASE,n,v)
  CORE.ranked_unreviewed=selected_ranked;CORE.ACTIVE_DATASET=ds;EVIDENCE_PATCH_MODULE.source_evidence=previous.previous.previous.previous.previous.previous.previous.previous.previous.previous.source_evidence;yield
 finally:
  EVIDENCE_PATCH_MODULE.source_evidence=evidence;CORE.ACTIVE_DATASET=active;CORE.ranked_unreviewed=ranking
  for n,v in orig.items():setattr(BASE,n,v)
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v57-projection-",dir=OUT_DIR) as t:
  ds=Path(t)/"v57.jsonl";rp=Path(t)/"r.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=537 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v57 projection drift")
  with patched_base(ds):BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:row.update(schema="context-merit-audit-v58",review_pass="resource_question_naturalness_reaudit",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":537,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True})
 write_jsonl(AUDIT,audits);report=json.loads(REPORT.read_text());report["schema"]="context-merit-audit-report-v58";report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=537,eligible_unreviewed_rows=0,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=8,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"remaining direct Rope-topia metadata prompts in the v57 projection","score":"manual standalone question naturalness","tie_break":"manifest order"})
 report["audit"]["rows"]=len(audits);report["audit"]["sha256"]=file_sha256(AUDIT);report["new_pending_curation"]["sha256"]=file_sha256(CURATION);report["new_pending_curation"]["edit_support_types"]={"extractive":8,"manual_paraphrase":0};report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()];REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
