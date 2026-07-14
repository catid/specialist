#!/usr/bin/env python3
"""Naturalize and attribute remaining direct historical/cultural term rows v63."""
from __future__ import annotations
import contextlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V62_DIR=DATA/"manual_reviews/context_merit_audit_v62";sys.path[:0]=[str(ROOT),str(V62_DIR)]
import build_context_merit_audit_v62 as previous
BASE,CORE=previous.BASE,previous.CORE;EVIDENCE_PATCH_MODULE=previous.EVIDENCE_PATCH_MODULE;OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v63.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v63.jsonl";REPORT=OUT_DIR/"report_context_merit_v63.json";REVIEWER,REVIEWED_AT="codex-context-merit-audit-v63","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION;file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl
CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,63));CONTEXT_AUDITS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"context_merit_audit_v{v}.jsonl" for v in range(1,63))
def raw(n):return DATA/"raw"/n
def edit(fact,path,marker,question,answer,code,reason,**extra):return {"fact_id":fact,"source_path":path,"marker":marker,"decision":"edit","question":question,"answer":answer,"reason_code":code,"reason":reason,**extra}
SPECS=(
 edit("fact-62c5c25b674662828ae4",raw("kinbakutoday_dfc9527c49ca8ad6.json"),"following Nureki and Akechi, Yukimura came into the scene in the 1990s","In Ugo’s contribution to “Kinbaku – An Evolving Era – Part 3,” during which decade does he say Yukimura entered the scene after Nureki and Akechi?","the 1990s","attribute_yukimura_decade_to_ugo_account","The revised wording preserves the chronology while attributing it to Ugo’s contribution in a multi-author debate."),
 edit("fact-4f7772ab91a636e50d78",raw("kinbakutoday_82071039cb003b58.json"),"Kitan Club was launched in 1947","In Ugo’s four-phase history in “Kinbaku – An Evolving Era – Part 2,” when does he say Kitan Club was launched?","1947","attribute_kitan_club_launch_to_ugo_history","The revised wording identifies both the speaker and the historical framework supporting the date."),
 edit("fact-c7f97d425dddd7aecbb5",raw("kinbakutoday_4f26f20c5f1dc7ba.json"),"Roman Porno was a genre created by Nikkatsu studios to produce erotic films for mainstream release","In “Junko Mabuki: Japan’s Second Queen of SM,” what name is given to Nikkatsu’s genre of erotic films produced for mainstream release?","Roman Porno","identify_nikkatsu_genre_in_named_article","The named-article context makes the short answer stand alone without broadening the source’s historical claim."),
 edit("fact-17b5a49e3a54220c3c36",raw("kinbakutoday_c5e568667b495473.json"),"正座 seiza","According to “Kinbaku Vocabulary 101: Sitting Positions in Japanese,” what Japanese term names formal kneeling with the legs folded under the thighs and the buttocks resting on the heels?","seiza","attribute_seiza_definition_to_vocabulary_article","The revised question names the vocabulary source and describes the position in natural language.",evidence_end_marker="buttocks resting on the heels"),
 edit("fact-300196e05741d520a754",raw("kinbakutoday_57d1ad7ef6bbe56a.json"),"“Shibaru” in the Kun’yomi pronunciation","In “What’s in a Name: Kinbaku and Shibari,” what Kun’yomi reading is given for the kanji 縛?","Shibaru","attribute_shibaru_reading_to_etymology_article","The revised wording attributes the linguistic reading to the named etymology article."),
 edit("fact-e39c053ace018b31331b",raw("kinbakutoday_7113a15b5e5e3aa3.json"),"it’s called Hashira Shibari","What name does “The Evocative Power of Hashira Shibari” give to tying a model to a vertical post?","Hashira Shibari","naturalize_hashira_shibari_definition","The revised question replaces awkward location wording with a direct definition tied to the source article."),
 edit("fact-3d3373b7b595e1b6324b",raw("kinbakutoday_34dec041941868d9.json"),"in the posturing called “Omoi-ire”, a Kabuki actor gives expression to the Omoi, without resort to language","In “Itoh Seiu: Urami and the Drama of Rope,” which Kabuki term names the wordless expression of omoi through the actor’s face and body?","Omoi-ire","attribute_omoi_ire_term_to_article","The source-specific wording preserves the cultural term while making its function clear."),
 edit("fact-bd0d9f53704c5d277b9f",raw("kinbakutoday_0fde39bdb08f42b9.json"),"Uramado (translation “Rear Window”), which ran from 1956 to 1965","According to “Uramado: West Meets East in Fetish Art,” during which years did the magazine Uramado run?","1956 to 1965","attribute_uramado_run_to_named_article","The revised wording attributes the publication dates to the article rather than presenting them without provenance."),
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES=dict(zip(EXPECTED_SELECTION,(132,133,242,257,258,261,270,380)))
PROJECTED_SELECTION_BASELINE={"description":"isolated cumulative training projection through context-merit v62","direct_rows_without_prior_curation":184,"historical_and_cultural_rows_selected":8,"rows":537,"sha256":"0b3cb9c61fc14e62bb7b9bbd01fbc8eec33c35b099c106e496becc59d4840653"}
ISOLATED_PROJECTION={"active_after_context_merit_v62":499,"active_after_this_tranche":499,"build_script":"build_curated_qa.py","new_drops_applied":0,"new_edits_applied":8,"output_rows":537,"output_sha256":"c4c83814e52d4d30188e5a25058b5c078dc63e6910e75ac91164752b56e7c081","prior_pending_addition_fact_ids_preserved":36,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":0,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS,SEALED_EVAL_PATHS=previous.PRODUCTION_INPUTS,previous.SEALED_EVAL_PATHS;PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
def prior_decision_artifacts():
 out=[]
 for v in range(1,63):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def build_projection(o,r,c):previous.build_projection(o,r,c)
def selected(rows):
 by={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)};out=[{"active_index":by[f][0],"row":by[f][1],"features":CORE.risk_features(by[f][1])} for f in EXPECTED_SELECTION]
 if {x["row"]["fact_id"]:x["active_index"] for x in out}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v63 candidate drift")
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
 with tempfile.TemporaryDirectory(prefix=".v62-projection-",dir=OUT_DIR) as t:
  ds=Path(t)/"v62.jsonl";rp=Path(t)/"r.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=537 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v62 projection drift")
  with patched_base(ds):BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:row.update(schema="context-merit-audit-v63",review_pass="historical_and_cultural_source_naturalness_reaudit",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":537,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True})
 write_jsonl(AUDIT,audits);report=json.loads(REPORT.read_text());report["schema"]="context-merit-audit-report-v63";report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=537,eligible_unreviewed_rows=0,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=8,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"remaining direct historical dates and cultural terms lacking precise source context or natural standalone wording","score":"manual full-source provenance and naturalness review","tie_break":"active projection order"})
 report["audit"]["rows"]=len(audits);report["audit"]["sha256"]=file_sha256(AUDIT);report["new_pending_curation"]["sha256"]=file_sha256(CURATION);report["new_pending_curation"]["edit_support_types"]={"extractive":8,"manual_paraphrase":0};report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()];REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
