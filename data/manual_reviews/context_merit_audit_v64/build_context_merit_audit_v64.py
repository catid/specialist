#!/usr/bin/env python3
"""Repair fragmentary and awkward Rope365 direct Q&A rows v64."""
from __future__ import annotations
import contextlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V63_DIR=DATA/"manual_reviews/context_merit_audit_v63";sys.path[:0]=[str(ROOT),str(V63_DIR)]
import build_context_merit_audit_v63 as previous
BASE,CORE=previous.BASE,previous.CORE;EVIDENCE_PATCH_MODULE=previous.EVIDENCE_PATCH_MODULE;OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v64.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v64.jsonl";REPORT=OUT_DIR/"report_context_merit_v64.json";REVIEWER,REVIEWED_AT="codex-context-merit-audit-v64","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION;file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl
CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,64));CONTEXT_AUDITS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"context_merit_audit_v{v}.jsonl" for v in range(1,64))
def raw(n):return DATA/"raw"/n
def edit(fact,path,marker,question,answer,code,reason,**extra):return {"fact_id":fact,"source_path":path,"marker":marker,"decision":"edit","question":question,"answer":answer,"reason_code":code,"reason":reason,**extra}
SPECS=(
 edit("fact-54b5e92e0ada0d0f492f",raw("rope365_095aa0f0eea4c62c.json"),"cow hitch is technically two half hitches","According to Rope365, how many symmetrically tied half hitches make up a cow hitch?","two","naturalize_cow_hitch_composition_question","The revised question states the symmetry relationship directly and attributes the definition to Rope365."),
 edit("fact-29f4f925dae3470cb141",raw("rope365_d9c48a4547717047.json"),"The knot is pulled and centred so it won’t come undone in transport","How should the securing knot be positioned in Rope365’s coiling checklist so it will not come undone during transport?","pulled and centred","repair_coiling_checklist_grammar","The revised question makes the adjective-phrase answer grammatically complete."),
 edit("fact-a23aee2e599cfbccafd5",raw("rope365_2ea01101bf29d77c.json"),"Most twisted ropes are made from 3 strands as it is a very stable structure","According to Rope365’s construction lesson, how many strands are most twisted ropes made from?","3 strands","naturalize_twisted_rope_strand_question","The revised wording removes a dangling causal clause while preserving the exact construction fact."),
 edit("fact-dd18848777e752fcf551",raw("rope365_d9c48a4547717047.json"),"The bight is easy to distinguish from the rest of the rope","In Rope365’s coiling checklist, which part should remain easy to distinguish from the rest of the rope?","The bight","attribute_bight_check_to_coiling_lesson","The source-specific wording makes the checklist context explicit."),
 edit("fact-e4f16fcd18549d4a1e80",raw("rope365_25f1b23eb40be00e.json"),"Since it’s a type of slip knot, it comes with the risk that it may tighten when put directly on the body","According to Rope365, what can happen when a handcuff knot is placed directly on the body?","The knot may tighten.","replace_ambiguous_handcuff_pronoun","The answer replaces an ambiguous pronoun with its explicit referent while preserving the source’s safety warning.",support_type="manual_paraphrase",paraphrase_support_fragments=("risk that it may tighten","put directly on the body")),
 edit("fact-7755f8eb9d6e2e97d23d",raw("rope365_5fdb5e78c2471772.json"),"Keep an untreated rope as your comparison tool","In Rope365’s conditioning exercise, what should be kept as the comparison control?","an untreated rope","naturalize_conditioning_control_question","The revised question uses clear experiment language and retains the exact comparison item."),
 edit("fact-ba692623a8506581339b",raw("rope365_25f1b23eb40be00e.json"),"a quick slip knot is a good way to mark the middle of the rope","Which knot does Rope365 suggest for quickly marking the middle of a rope that may be used again later?","quick slip knot","naturalize_middle_marking_knot_question","The revised question removes unnecessary wording and identifies Rope365 as the source."),
 edit("fact-fd5c96b72210af42fd05",raw("rope365_b781bc1188743976.json"),"use hitches to connect the ropes together","What does Rope365’s crafting exercise suggest using to connect ropes while filling a space?","hitches","naturalize_space_filling_hitches_question","The revised wording retains the exercise context and asks directly for the connecting technique."),
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES=dict(zip(EXPECTED_SELECTION,(102,204,261,292,309,327,335,340)))
PROJECTED_SELECTION_BASELINE={"description":"isolated cumulative training projection through context-merit v63","direct_rows_without_prior_curation":176,"rope365_language_rows_selected":8,"rows":537,"sha256":"c4c83814e52d4d30188e5a25058b5c078dc63e6910e75ac91164752b56e7c081"}
ISOLATED_PROJECTION={"active_after_context_merit_v63":499,"active_after_this_tranche":499,"build_script":"build_curated_qa.py","new_drops_applied":0,"new_edits_applied":8,"output_rows":537,"output_sha256":"22e81a31c5ec4531a22d6ba3de67c0d986b07bab19088cc27d08164d7167c249","prior_pending_addition_fact_ids_preserved":36,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":0,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS,SEALED_EVAL_PATHS=previous.PRODUCTION_INPUTS,previous.SEALED_EVAL_PATHS;PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
def prior_decision_artifacts():
 out=[]
 for v in range(1,64):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def build_projection(o,r,c):previous.build_projection(o,r,c)
def selected(rows):
 by={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)};out=[{"active_index":by[f][0],"row":by[f][1],"features":CORE.risk_features(by[f][1])} for f in EXPECTED_SELECTION]
 if {x["row"]["fact_id"]:x["active_index"] for x in out}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v64 candidate drift")
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
 with tempfile.TemporaryDirectory(prefix=".v63-projection-",dir=OUT_DIR) as t:
  ds=Path(t)/"v63.jsonl";rp=Path(t)/"r.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=537 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v63 projection drift")
  with patched_base(ds):BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:
  row.update(schema="context-merit-audit-v64",review_pass="rope365_fragment_and_grammar_reaudit",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":537,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True})
  spec=next(s for s in SPECS if s["fact_id"]==row["fact_id"])
  if spec.get("support_type")=="manual_paraphrase":row["paraphrase_rationale"]=spec["reason"]
 write_jsonl(AUDIT,audits)
 curations=read_jsonl(CURATION)
 for row in curations:
  spec=next(s for s in SPECS if s["fact_id"]==row["fact_id"])
  if spec.get("support_type")=="manual_paraphrase":row.update(support_type="manual_paraphrase",paraphrase_rationale=spec["reason"])
 write_jsonl(CURATION,curations);report=json.loads(REPORT.read_text());report["schema"]="context-merit-audit-report-v64";report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=537,eligible_unreviewed_rows=0,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=8,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"remaining direct Rope365 technique rows with fragmentary answers, dangling clauses, ambiguous pronouns, or underspecified exercise context","score":"manual full-source answer completeness and question naturalness review","tie_break":"active projection order"})
 report["audit"]["rows"]=len(audits);report["audit"]["sha256"]=file_sha256(AUDIT);report["new_pending_curation"]["sha256"]=file_sha256(CURATION);report["new_pending_curation"]["edit_support_types"]={"extractive":7,"manual_paraphrase":1};report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()];REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
