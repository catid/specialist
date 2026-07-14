#!/usr/bin/env python3
"""Improve standalone context for remaining direct technical/cultural prompts v60."""
from __future__ import annotations
import contextlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V59_DIR=DATA/"manual_reviews/context_merit_audit_v59";sys.path[:0]=[str(ROOT),str(V59_DIR)]
import build_context_merit_audit_v59 as previous
BASE,CORE=previous.BASE,previous.CORE;EVIDENCE_PATCH_MODULE=previous.EVIDENCE_PATCH_MODULE;OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v60.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v60.jsonl";REPORT=OUT_DIR/"report_context_merit_v60.json";REVIEWER,REVIEWED_AT="codex-context-merit-audit-v60","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION;file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl
CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,60));CONTEXT_AUDITS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"context_merit_audit_v{v}.jsonl" for v in range(1,60))
def raw(n):return DATA/"raw"/n
def edit(fact,path,marker,question,answer,code):return {"fact_id":fact,"source_path":path,"marker":marker,"decision":"edit","question":question,"answer":answer,"reason_code":code,"reason":"The edit names the lesson or article so the source-grounded answer remains clear when the question is presented by itself."}
SPECS=(
 edit("fact-517f5ab1f5e32057e9b0",raw("rope365_3d015bb41296c9fa.json"),"If you used cinches, they are flat with no bulk between the arms and the torso","In Rope365’s box-tie self-evaluation checklist, how should any cinches lie?","flat with no bulk between the arms and the torso","identify_box_tie_checklist_in_cinch_question"),
 edit("fact-dc125508d925fcfc272b",raw("rope365_682937f92222bf87.json"),"Violence, unsafe practices, and lack of consent","Which three darker aspects of bondage history does Rope365’s history overview name?","Violence, unsafe practices, and lack of consent","identify_rope365_history_overview"),
 edit("fact-d4e83b39d08de0c48f56",raw("rope365_d9c48a4547717047.json"),"asking if and how you can help","How does Rope365 recommend checking whether you may help with another person’s rope?","asking if and how you can help","clarify_coiling_etiquette_question"),
 edit("fact-426bc3d2c6101548619f",raw("kinbakutoday_d4dcb268cb41c5e4.json"),"悦虐, “ecstatic cruelty,”","In “The Woman Who Wasn’t There,” which Japanese term does the narrator use for “ecstatic cruelty”?","悦虐","identify_article_in_ecstatic_cruelty_term_question"),
 edit("fact-35a0f1a70111fab6fbff",raw("kinbakutoday_c5e568667b495473.json"),"ashi o kuzushite ii yo","In “Kinbaku Vocabulary 101: Sitting Positions in Japanese,” which phrase tells a model they may shift from formal kneeling?","ashi o kuzushite ii yo","identify_vocabulary_article_in_sitting_phrase_question"),
 edit("fact-4471cc69e2bb67b1059c",raw("kinbakutoday_d4dcb268cb41c5e4.json"),"how to distinguish seme from mere violence, vulgar appetite, or crude spectacle","In “The Woman Who Wasn’t There,” what publishing problem does Reiko’s conflict with her male patrons dramatize?","how to distinguish seme from mere violence, vulgar appetite, or crude spectacle","identify_article_in_publishing_problem_question"),
 edit("fact-65c43946e82ed5b6e3a2",raw("esinem_9a5aab43708932b3.json"),"the components that are repeated in ties","What term does ESINEM use for components that recur across shibari ties?","ingredients","identify_esinem_ingredients_article"),
 edit("fact-bc11688b1543ec8b2d9f",raw("kinbakutoday_f57559bbb4c8b826.json"),"deep meaning of kinbaku, to find kinbaku-bi","In “Nureki, Kinbiken, and the Aesthetics of Kinbaku,” what term names kinbaku’s deep aesthetic meaning?","kinbaku-bi","identify_nureki_aesthetics_article"),
 edit("fact-7d40fbdc3504caa3efcf",raw("esinem_dce6c59fa90ecae0.json"),"‘rigger’ as a word for the active party","In ESINEM’s article on neutral bondage terms, which word names the active party but can also mean a theatrical equipment rigger?","rigger","identify_neutral_terms_article"),
 edit("fact-388ba93105c948d44738",raw("kinbakutoday_f370696af0359092.json"),"were wood printed theater posters in which scenes of plays were sequentially depicted","In Kinbaku Today’s “Tsuji Banzuke,” what were the wood-printed theater posters that depicted a whole story sequentially called?","Tsuji Banzuke","identify_tsuji_banzuke_article"),
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES=dict(zip(EXPECTED_SELECTION,(26,27,68,261,288,297,331,332,339,377)))
PROJECTED_SELECTION_BASELINE={"description":"isolated cumulative training projection through context-merit v59","direct_rows_without_prior_curation":211,"standalone_context_rows_selected":10,"rows":537,"sha256":"f0af15d3159836aa0fa1d7798e1d426802489f75e20e880dbe4069a4c963d313"}
ISOLATED_PROJECTION={"active_after_context_merit_v59":499,"active_after_this_tranche":499,"build_script":"build_curated_qa.py","new_drops_applied":0,"new_edits_applied":10,"output_rows":537,"output_sha256":"c1277333dd8889cf0cee0e49be5a00d044ace042fad7bbd6ca7489bee955279c","prior_pending_addition_fact_ids_preserved":36,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":0,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS,SEALED_EVAL_PATHS=previous.PRODUCTION_INPUTS,previous.SEALED_EVAL_PATHS;PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
def prior_decision_artifacts():
 out=[]
 for v in range(1,60):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def build_projection(o,r,c):previous.build_projection(o,r,c)
def selected(rows):
 by={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)};out=[{"active_index":by[f][0],"row":by[f][1],"features":CORE.risk_features(by[f][1])} for f in EXPECTED_SELECTION]
 if {x["row"]["fact_id"]:x["active_index"] for x in out}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v60 candidate drift")
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
 with tempfile.TemporaryDirectory(prefix=".v59-projection-",dir=OUT_DIR) as t:
  ds=Path(t)/"v59.jsonl";rp=Path(t)/"r.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=537 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v59 projection drift")
  with patched_base(ds):BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:row.update(schema="context-merit-audit-v60",review_pass="standalone_lesson_and_article_context_reaudit",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":537,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True})
 write_jsonl(AUDIT,audits);report=json.loads(REPORT.read_text());report["schema"]="context-merit-audit-report-v60";report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=537,eligible_unreviewed_rows=0,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=10,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"direct technical or cultural questions whose generic text/checklist/term phrasing loses source context","score":"manual standalone lesson and article clarity","tie_break":"active projection order"})
 report["audit"]["rows"]=len(audits);report["audit"]["sha256"]=file_sha256(AUDIT);report["new_pending_curation"]["sha256"]=file_sha256(CURATION);report["new_pending_curation"]["edit_support_types"]={"extractive":10,"manual_paraphrase":0};report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()];REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
