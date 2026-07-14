#!/usr/bin/env python3
"""Repair source attribution and historical-subject clarity in direct rows v62."""
from __future__ import annotations
import contextlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V61_DIR=DATA/"manual_reviews/context_merit_audit_v61";sys.path[:0]=[str(ROOT),str(V61_DIR)]
import build_context_merit_audit_v61 as previous
BASE,CORE=previous.BASE,previous.CORE;EVIDENCE_PATCH_MODULE=previous.EVIDENCE_PATCH_MODULE;OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v62.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v62.jsonl";REPORT=OUT_DIR/"report_context_merit_v62.json";REVIEWER,REVIEWED_AT="codex-context-merit-audit-v62","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION;file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl
CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,62));CONTEXT_AUDITS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"context_merit_audit_v{v}.jsonl" for v in range(1,62))
def raw(n):return DATA/"raw"/n
def edit(fact,path,marker,question,answer,code,reason,**extra):return {"fact_id":fact,"source_path":path,"marker":marker,"decision":"edit","question":question,"answer":answer,"reason_code":code,"reason":reason,**extra}
SPECS=(
 edit("fact-4792da521af19b4e36b1",raw("kinbakutoday_be37dae4ec8e0d88.json"),"The name Nureki Chimuo as a Kinbakushi emerged in 1973","In Ugo’s historical account within “Kinbaku – An Evolving Era – Part 1,” when did the name Nureki Chimuo emerge as a kinbakushi?","1973","attribute_nureki_1973_to_ugo_account","The source-specific wording preserves a useful chronology fact while identifying it as Ugo’s account in a multi-author debate."),
 edit("fact-5dd51fd0f935594ae914",raw("kinbakutoday_241155b848764148.json"),"Kitan Club, shut down for most of 1955 after their April issue was seized by the government","According to “Rope: Community and Culture,” in which year was Kitan Club shut down for most of the year after the government seized its April issue?","1955","clarify_kitan_club_censorship_chronology","The revised question makes the historically meaningful censorship event explicit instead of asking for a date without context."),
 edit("fact-978096701201424beddd",raw("kinbakutoday_a358fd398f91040a.json"),"Kinbaku means “tight binding”","In “My Vision of Kinbaku,” what is the term kinbaku literally said to mean?","tight binding","attribute_kinbaku_translation_to_article","The source-specific wording avoids presenting one article’s translation as an unattributed universal definition."),
 edit("fact-c06bbd52eab400cf8bf0",raw("kinbakutoday_f57559bbb4c8b826.json"),"The name itself, Kinbiken, is a contraction of Kinbakubi kenkyūkai","In “Nureki, Kinbiken, and the Aesthetics of Kinbaku,” what shorter name is identified as a contraction of Kinbakubi kenkyūkai?","Kinbiken","naturalize_kinbiken_contraction_question","The revised wording names the article and asks naturally for the shortened name rather than an abstract contraction."),
 edit("fact-d596019f1c21d094ee06",raw("kinbakutoday_e7f3e175c6e3bfd7.json"),"aibunawa or “caressing rope.”","How does “Aibunawa and Semenawa: Pleasure and Endurance” translate aibunawa into English?","caressing rope","attribute_aibunawa_translation_to_article","The revised wording attaches the translation to the source article while retaining the exact supported answer."),
 edit("fact-b784b4278644a2f88656",raw("rope365_6f46d5169ca32ec7.json"),"co-founder of Bakuyukai, the first Japanese rope dojo","According to Rope365’s rope-artist references, what was the name of the first Japanese rope dojo, co-founded by Akechi Denki?","Bakuyukai","attribute_bakuyukai_claim_to_rope365","The revised wording preserves useful historical context while attributing the first-dojo claim to Rope365."),
 edit("fact-55da0d1915d37b1d077f",raw("kinbakutoday_36c008200d681448.json"),"Sin: Can you explain a little about Barajūjikan (Rosencreutz)?","In the Ryuzaki Asuka interview, which bar does Asuka identify as the first SM bar in western Japan?","Barajūjikan (Rosencreutz)","attribute_first_western_japan_sm_bar_claim","The interview attribution distinguishes Asuka’s historical identification from an unattributed categorical claim.",evidence_end_marker="it was the first SM bar in western Japan"),
 edit("fact-004dc043eac35f6f8258",raw("kinbakutoday_de20a4adcc8ec0d5.json"),"Kawabata’s bondage work determined the nature and direction of the magazine Kitan Club thereafter","According to the translated 1955 essay “The Quality of Bondage Models,” which magazine’s later nature and direction were shaped by Tanako Kawabata’s bondage work?","Kitan Club","clarify_kawabata_work_as_historical_subject","The revised question names the translated essay and correctly identifies Kawabata’s work, rather than the author’s work generally, as the subject of the claim."),
 edit("fact-cffa44c03400fc16f70f",raw("esinem_f2dfde25be14a7a8.json"),"Itoh Seiu was profoundly influenced by the work of ukiyoe artist Tsukioka Yoshitoshi","In NuitDeTokyo’s comments reproduced in ESINEM’s “The Origin of the Word ‘Kinbaku,’” which ukiyo-e artist is said to have profoundly influenced Itoh Seiu?","Tsukioka Yoshitoshi","attribute_yoshitoshi_claim_to_quoted_comment","The revised wording accurately attributes the claim to the reproduced comments instead of flattening nested source provenance."),
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES=dict(zip(EXPECTED_SELECTION,(121,125,228,249,251,374,375,449,480)))
PROJECTED_SELECTION_BASELINE={"description":"isolated cumulative training projection through context-merit v61","direct_rows_without_prior_curation":193,"source_attribution_rows_selected":9,"rows":537,"sha256":"a8603dae26aec6c1882cb3bc26f5f0644390562fae82998bf46d10653c71d4e7"}
ISOLATED_PROJECTION={"active_after_context_merit_v61":499,"active_after_this_tranche":499,"build_script":"build_curated_qa.py","new_drops_applied":0,"new_edits_applied":9,"output_rows":537,"output_sha256":"0b3cb9c61fc14e62bb7b9bbd01fbc8eec33c35b099c106e496becc59d4840653","prior_pending_addition_fact_ids_preserved":36,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":0,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS,SEALED_EVAL_PATHS=previous.PRODUCTION_INPUTS,previous.SEALED_EVAL_PATHS;PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
def prior_decision_artifacts():
 out=[]
 for v in range(1,62):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def build_projection(o,r,c):previous.build_projection(o,r,c)
def selected(rows):
 by={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)};out=[{"active_index":by[f][0],"row":by[f][1],"features":CORE.risk_features(by[f][1])} for f in EXPECTED_SELECTION]
 if {x["row"]["fact_id"]:x["active_index"] for x in out}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v62 candidate drift")
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
 with tempfile.TemporaryDirectory(prefix=".v61-projection-",dir=OUT_DIR) as t:
  ds=Path(t)/"v61.jsonl";rp=Path(t)/"r.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=537 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v61 projection drift")
  with patched_base(ds):BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:row.update(schema="context-merit-audit-v62",review_pass="source_attribution_and_historical_subject_reaudit",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":537,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True})
 write_jsonl(AUDIT,audits);report=json.loads(REPORT.read_text());report["schema"]="context-merit-audit-report-v62";report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=537,eligible_unreviewed_rows=0,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=9,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"direct historical and translation questions whose source provenance or grammatical subject was underspecified","score":"manual full-source attribution and standalone naturalness review","tie_break":"active projection order"})
 report["audit"]["rows"]=len(audits);report["audit"]["sha256"]=file_sha256(AUDIT);report["new_pending_curation"]["sha256"]=file_sha256(CURATION);report["new_pending_curation"]["edit_support_types"]={"extractive":9,"manual_paraphrase":0};report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()];REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
