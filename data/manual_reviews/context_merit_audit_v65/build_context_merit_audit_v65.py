#!/usr/bin/env python3
"""Reaudit concise Anatomie Studio consent, hygiene, and check-in rows v65."""
from __future__ import annotations
import contextlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V64_DIR=DATA/"manual_reviews/context_merit_audit_v64";sys.path[:0]=[str(ROOT),str(V64_DIR)]
import build_context_merit_audit_v64 as previous
BASE,CORE=previous.BASE,previous.CORE;EVIDENCE_PATCH_MODULE=previous.EVIDENCE_PATCH_MODULE;OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v65.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v65.jsonl";REPORT=OUT_DIR/"report_context_merit_v65.json";REVIEWER,REVIEWED_AT="codex-context-merit-audit-v65","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION;file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl
CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,65));CONTEXT_AUDITS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"context_merit_audit_v{v}.jsonl" for v in range(1,65))
def raw(n):return DATA/"raw"/n
def keep(fact,path,marker,code,reason):return {"fact_id":fact,"source_path":path,"marker":marker,"decision":"keep","reason_code":code,"reason":reason}
def edit(fact,path,marker,question,answer,code,reason):return {"fact_id":fact,"source_path":path,"marker":marker,"decision":"edit","question":question,"answer":answer,"reason_code":code,"reason":reason}
ETIQUETTE=raw("anatomiestudio_27ecdd4d7c9a5560.json");HYGIENE=raw("anatomiestudio_144932682af9c846.json");SAFEWORDS=raw("anatomiestudio_9749de0eb1ff4ef3.json")
SPECS=(
 edit("fact-da15b630db4ec0ed79cf",HYGIENE,"making the ropes feel spongy and springy when they dry out","After wet jute rope dries, how does Anatomie Studio say its tightened twist may make the rope feel?","spongy and springy","clarify_wet_jute_texture_mechanism","The revised question preserves the useful care warning while making the tightened-twist mechanism and source explicit."),
 keep("fact-eb477d063e72bae0adb8",ETIQUETTE,"Suspension is edge play","retain_explicit_suspension_risk_classification","The concise attributed answer accurately preserves Anatomie Studio’s explicit classification of suspension as risky edge play."),
 edit("fact-069a861dbb2bea9e47ca",ETIQUETTE,"– Mixed messages mean “no.”","How does Anatomie Studio say mixed consent messages should be interpreted?","“no”","repair_mixed_messages_answer_rendering","The revised answer uses the source’s quoted term and the question states the consent context naturally."),
 keep("fact-372357c6d61a3f8b6ce6",ETIQUETTE,"We call these conversations ‘negotiation’","retain_clear_negotiation_definition","The question already supplies the studio and consent context, and the one-word answer is the exact term defined by the source."),
 edit("fact-f7e802bf0b2759290dc6",HYGIENE,"make sure all parties are able to give informed consent","In Anatomie Studio’s rope-hygiene guidance, what must all parties be able to give before deciding how rope will be used?","informed consent","add_hygiene_context_to_informed_consent","The revised question makes clear that this consent requirement concerns hygiene choices about shared or intimate rope use."),
 keep("fact-a3833a3bcbe7d6dc8400",ETIQUETTE,"If you have doubts, don’t proceed","retain_actionable_enthusiasm_check","The existing question and answer form a direct, actionable rule for responding to doubt about a partner’s enthusiasm."),
 keep("fact-97d52aa2d87dadfd8226",SAFEWORDS,"immediate responsibility will fall to the rigger to check in","retain_nonverbal_model_check_in_responsibility","The attributed row clearly assigns the immediate check-in responsibility when the model cannot communicate verbally."),
 keep("fact-03cbcf694e260c90a1bf",ETIQUETTE,"freezing is not consent","retain_freeze_response_consent_rule","The concise answer directly teaches that a freeze response must not be treated as consent."),
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES=dict(zip(EXPECTED_SELECTION,(55,62,110,190,288,329,380,462)))
PROJECTED_SELECTION_BASELINE={"description":"isolated cumulative training projection through context-merit v64","anatomie_rows_selected":8,"direct_rows_without_prior_curation":168,"rows":537,"sha256":"22e81a31c5ec4531a22d6ba3de67c0d986b07bab19088cc27d08164d7167c249"}
ISOLATED_PROJECTION={"active_after_context_merit_v64":499,"active_after_this_tranche":499,"build_script":"build_curated_qa.py","new_drops_applied":0,"new_edits_applied":3,"output_rows":537,"output_sha256":"25690dda4b0be5d677e3d2cb7a64013155c5d4156adb84bf34cf459b3659e131","prior_pending_addition_fact_ids_preserved":36,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":5,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS,SEALED_EVAL_PATHS=previous.PRODUCTION_INPUTS,previous.SEALED_EVAL_PATHS;PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
def prior_decision_artifacts():
 out=[]
 for v in range(1,65):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def build_projection(o,r,c):previous.build_projection(o,r,c)
def selected(rows):
 by={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)};out=[{"active_index":by[f][0],"row":by[f][1],"features":CORE.risk_features(by[f][1])} for f in EXPECTED_SELECTION]
 if {x["row"]["fact_id"]:x["active_index"] for x in out}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v65 candidate drift")
 return out
def selected_ranked(rows):return selected(rows),0,0
def evidence_validator():
 module=previous
 while not hasattr(module,"source_evidence"):module=module.previous
 return module.source_evidence
@contextlib.contextmanager
def patched_base(ds):
 rep={"OUT_DIR":OUT_DIR,"AUDIT":AUDIT,"CURATION":CURATION,"REPORT":REPORT,"REVIEWER":REVIEWER,"REVIEWED_AT":REVIEWED_AT,"CONTEXT_CURATIONS":CONTEXT_CURATIONS,"SPECS":SPECS,"EXPECTED_SELECTION":EXPECTED_SELECTION,"ISOLATED_PROJECTION":ISOLATED_PROJECTION};orig={n:getattr(BASE,n) for n in rep};ranking,active,evidence=CORE.ranked_unreviewed,CORE.ACTIVE_DATASET,EVIDENCE_PATCH_MODULE.source_evidence
 try:
  for n,v in rep.items():setattr(BASE,n,v)
  CORE.ranked_unreviewed=selected_ranked;CORE.ACTIVE_DATASET=ds;EVIDENCE_PATCH_MODULE.source_evidence=evidence_validator();yield
 finally:
  EVIDENCE_PATCH_MODULE.source_evidence=evidence;CORE.ACTIVE_DATASET=active;CORE.ranked_unreviewed=ranking
  for n,v in orig.items():setattr(BASE,n,v)
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v64-projection-",dir=OUT_DIR) as t:
  ds=Path(t)/"v64.jsonl";rp=Path(t)/"r.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=537 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v64 projection drift")
  with patched_base(ds):BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:row.update(schema="context-merit-audit-v65",review_pass="anatomie_consent_hygiene_and_checkin_reaudit",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":537,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True})
 write_jsonl(AUDIT,audits);report=json.loads(REPORT.read_text());report["schema"]="context-merit-audit-report-v65";report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=537,eligible_unreviewed_rows=0,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=8,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"concise direct Anatomie Studio rows about consent, hygiene, suspension risk, and communication responsibility","score":"manual full-source actionability, answer completeness, and contextual sufficiency review","tie_break":"active projection order"})
 report["audit"]["rows"]=len(audits);report["audit"]["sha256"]=file_sha256(AUDIT);report["new_pending_curation"]["sha256"]=file_sha256(CURATION);report["new_pending_curation"]["edit_support_types"]={"extractive":3,"manual_paraphrase":0};report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()];REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
