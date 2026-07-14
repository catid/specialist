#!/usr/bin/env python3
"""Repair ambiguous standalone attribution in direct training questions v59."""
from __future__ import annotations
import contextlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V58_DIR=DATA/"manual_reviews/context_merit_audit_v58";sys.path[:0]=[str(ROOT),str(V58_DIR)]
import build_context_merit_audit_v58 as previous
BASE,CORE=previous.BASE,previous.CORE;EVIDENCE_PATCH_MODULE=previous.EVIDENCE_PATCH_MODULE;OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v59.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v59.jsonl";REPORT=OUT_DIR/"report_context_merit_v59.json";REVIEWER,REVIEWED_AT="codex-context-merit-audit-v59","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION
file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl;CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,59));CONTEXT_AUDITS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"context_merit_audit_v{v}.jsonl" for v in range(1,59))
def raw(n):return DATA/"raw"/n
def edit(fact,path,marker,question,answer,code,reason):return {"fact_id":fact,"source_path":path,"marker":marker,"decision":"edit","question":question,"answer":answer,"reason_code":code,"reason":reason}
SPECS=(
 edit("fact-4af7fdc6e5bd4c92eb36",raw("kinbakutoday_c364f23ce34ae761.json"),"I might be (sexually) aroused, but that doesn’t necessarily mean I want to have sex with you.","In “When Does the Sex Start?”, what phrase does the author suggest normalizing to clarify that arousal does not imply wanting sex?","I might be (sexually) aroused, but that doesn’t necessarily mean I want to have sex with you.","identify_sexuality_article_in_normalization_question","The article title makes the first-person recommendation independently attributable while preserving its exact wording."),
 edit("fact-ae00d3e5d00d8fe80feb",raw("kinbakutoday_d4dcb268cb41c5e4.json"),"a ventriloquized interiority that makes seme appear to arise from women themselves","In “The Woman Who Wasn’t There,” what phrase describes a male-authored feminine voice that presents male fantasy as women’s own desire?","ventriloquized interiority","identify_article_in_critical_term_question","The title replaces a context-dependent author reference while preserving the article's defined critical term."),
 edit("fact-58af822afb984d0315d9",raw("kinbakutoday_011f67c75b8f999f.json"),"Humility is necessary to question oneself all the time","In “One Way Into Kinbaku,” what quality is described as necessary for continually questioning oneself?","Humility","identify_personal_essay_in_humility_question","The title replaces an ambiguous author reference without changing the essay's answer."),
 edit("fact-1a17204bc89767b3d93e",raw("kinbakutoday_d4dcb268cb41c5e4.json"),"I leave seme untranslated because no single English term carries its range","Which term does “The Woman Who Wasn’t There” leave untranslated because no single English word covers its full range?","seme","identify_article_in_untranslated_term_question","The title makes the terminology question stand alone while retaining the source's explicit translation rationale."),
 edit("fact-a9d68256fa3f2a7aac5d",raw("kinbakutoday_dfc9527c49ca8ad6.json"),"creativity, improvisation and a personal touch","In “Kinbaku – An Evolving Era – Part 3,” which three qualities are said to decline with insecurity and approval-seeking?","creativity, improvisation and a personal touch","identify_evolving_era_essay_in_creativity_question","The edit replaces a generic author reference with the named essay while preserving the attributed critique."),
 edit("fact-00fe778028241519d8af",raw("wykd_19d6a26116e26c70.json"),"common factor is likely to be the person tying","According to WykD’s injury-responsibility article, what common factor is likely when injuries recur across multiple models and sessions?","the person tying","identify_injury_article_in_recurrence_question","The edit names the source context so the safety lesson stands alone without overstating it as universal causation."),
 edit("fact-866a8b187a9b5eade0b8",raw("kinbakutoday_70cbcc84801cccd0.json"),"Kokoro is in the people who are engaging, communicating, and sharing themselves with each other","According to “The Heart of Kinbaku,” where is kokoro located rather than in rope or technique?","in the people who are engaging, communicating, and sharing themselves with each other","identify_heart_of_kinbaku_in_kokoro_question","The title replaces a context-dependent author reference while retaining the article's relational answer."),
 edit("fact-d7160c670538be792592",raw("kinbakutoday_dfc9527c49ca8ad6.json"),"tea ceremony, calligraphy and Budō","According to Ugo’s account in “Kinbaku – An Evolving Era – Part 3,” which three traditions were often forcibly associated with kinbaku in the West?","the tea ceremony, calligraphy and Budō","attribute_evolving_era_tradition_claim","Naming Ugo and the essay makes this historical critique explicitly attributable outside its source page."),
 edit("fact-d51f978bc64b51b4c65e",raw("rope365_682937f92222bf87.json"),"The History & Myths of Japanese Bondage by Midori","Who wrote the Rope365 history resource “The History & Myths of Japanese Bondage”?","Midori","correct_book_label_and_identify_resource_context","The edit removes the unsupported label 'book' and accurately frames the item as a named resource in Rope365's history list."),
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES=dict(zip(EXPECTED_SELECTION,(281,282,296,335,341,388,428,470,486)))
PROJECTED_SELECTION_BASELINE={"description":"isolated cumulative training projection through context-merit v58","direct_rows_without_prior_curation":220,"standalone_attribution_rows_selected":9,"rows":537,"sha256":"eb92d933fbd57e615a800c3a03a4177d8390d5b4b84f228ae3e7be74e18fe557"}
ISOLATED_PROJECTION={"active_after_context_merit_v58":499,"active_after_this_tranche":499,"build_script":"build_curated_qa.py","new_drops_applied":0,"new_edits_applied":9,"output_rows":537,"output_sha256":"f0af15d3159836aa0fa1d7798e1d426802489f75e20e880dbe4069a4c963d313","prior_pending_addition_fact_ids_preserved":36,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":0,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS,SEALED_EVAL_PATHS=previous.PRODUCTION_INPUTS,previous.SEALED_EVAL_PATHS;PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
def prior_decision_artifacts():
 out=[]
 for v in range(1,59):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def build_projection(o,r,c):previous.build_projection(o,r,c)
def selected(rows):
 by={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)};out=[{"active_index":by[f][0],"row":by[f][1],"features":CORE.risk_features(by[f][1])} for f in EXPECTED_SELECTION]
 if {x["row"]["fact_id"]:x["active_index"] for x in out}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v59 candidate drift")
 return out
def selected_ranked(rows):return selected(rows),0,0
@contextlib.contextmanager
def patched_base(ds):
 rep={"OUT_DIR":OUT_DIR,"AUDIT":AUDIT,"CURATION":CURATION,"REPORT":REPORT,"REVIEWER":REVIEWER,"REVIEWED_AT":REVIEWED_AT,"CONTEXT_CURATIONS":CONTEXT_CURATIONS,"SPECS":SPECS,"EXPECTED_SELECTION":EXPECTED_SELECTION,"ISOLATED_PROJECTION":ISOLATED_PROJECTION};orig={n:getattr(BASE,n) for n in rep};ranking,active,evidence=CORE.ranked_unreviewed,CORE.ACTIVE_DATASET,EVIDENCE_PATCH_MODULE.source_evidence
 try:
  for n,v in rep.items():setattr(BASE,n,v)
  CORE.ranked_unreviewed=selected_ranked;CORE.ACTIVE_DATASET=ds;EVIDENCE_PATCH_MODULE.source_evidence=previous.previous.previous.previous.previous.previous.previous.previous.previous.previous.previous.source_evidence;yield
 finally:
  EVIDENCE_PATCH_MODULE.source_evidence=evidence;CORE.ACTIVE_DATASET=active;CORE.ranked_unreviewed=ranking
  for n,v in orig.items():setattr(BASE,n,v)
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v58-projection-",dir=OUT_DIR) as t:
  ds=Path(t)/"v58.jsonl";rp=Path(t)/"r.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=537 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v58 projection drift")
  with patched_base(ds):BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:row.update(schema="context-merit-audit-v59",review_pass="standalone_attribution_and_label_reaudit",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":537,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True})
 write_jsonl(AUDIT,audits);report=json.loads(REPORT.read_text());report["schema"]="context-merit-audit-report-v59";report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=537,eligible_unreviewed_rows=0,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=9,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"direct questions with ambiguous author/article wording or a source-type label unsupported by the full source","score":"manual standalone attribution and source-label accuracy","tie_break":"active projection order"})
 report["audit"]["rows"]=len(audits);report["audit"]["sha256"]=file_sha256(AUDIT);report["new_pending_curation"]["sha256"]=file_sha256(CURATION);report["new_pending_curation"]["edit_support_types"]={"extractive":9,"manual_paraphrase":0};report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()];REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
