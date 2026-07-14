#!/usr/bin/env python3
"""Complete fragmented agency and responsibility answers in direct rows v67."""
from __future__ import annotations
import contextlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V66_DIR=DATA/"manual_reviews/context_merit_audit_v66";sys.path[:0]=[str(ROOT),str(V66_DIR)]
import build_context_merit_audit_v66 as previous
BASE,CORE=previous.BASE,previous.CORE;EVIDENCE_PATCH_MODULE=previous.EVIDENCE_PATCH_MODULE;OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v67.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v67.jsonl";REPORT=OUT_DIR/"report_context_merit_v67.json";REVIEWER,REVIEWED_AT="codex-context-merit-audit-v67","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION;file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl
CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,67));CONTEXT_AUDITS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"context_merit_audit_v{v}.jsonl" for v in range(1,67))
def raw(n):return DATA/"raw"/n
def keep(fact,path,marker,code,reason):return {"fact_id":fact,"source_path":path,"marker":marker,"decision":"keep","reason_code":code,"reason":reason}
def paraphrase(fact,path,marker,question,answer,fragments,code,reason):return {"fact_id":fact,"source_path":path,"marker":marker,"decision":"edit","question":question,"answer":answer,"support_type":"manual_paraphrase","paraphrase_support_fragments":fragments,"paraphrase_rationale":reason,"reason_code":code,"reason":reason}
ICHINAWA=raw("wykd_a74fec63b0114fff.json");NEWNESS=raw("wykd_944e4e6d621a97c9.json");BOTTOM_GUIDE=raw("kinbakutoday_432c8adfc1abe686.json")
SPECS=(
 paraphrase("fact-4f233aa2c03b91505209",ICHINAWA,"a correct usage‘ but not necessarily the only correct usage","Does WykD claim that Ichinawa is the only correct name for the one-rope technique?","No. WykD calls it a correct usage, but not necessarily the only correct usage.",("a correct usage","not necessarily the only correct usage"),"complete_ichinawa_yes_no_answer","The answer now responds directly to the yes-or-no question and preserves WykD’s explicit qualification."),
 keep("fact-fcea3c04b3e8979d1e24",ICHINAWA,"a technique in its own right that specifically only ever uses one rope","retain_ichinawa_technique_distinction","The existing row clearly distinguishes a named one-rope technique from merely limiting a practice exercise to one rope."),
 paraphrase("fact-eee35bea21f484fef5bf",BOTTOM_GUIDE,"It is the riggers responsibility to ensure what they are doing, is right for your fitness level and body type","According to Clover’s guide for rope bottoms, what responsibility does the rigger have regarding the bottom’s fitness level and body type?","The rigger should ensure that what they are doing is appropriate for the bottom’s fitness level and body type.",("riggers responsibility","ensure what they are doing","right for your fitness level and body type"),"complete_rigger_fitness_responsibility_answer","The answer becomes a complete standalone sentence and replaces second-person wording with the role named in the question."),
 paraphrase("fact-ee87d53a840fc422951b",NEWNESS,"Not to be touched, slapped, spanked, groped, played with or otherwise molested without consent","What should newcomers to kink events be able to expect regarding physical contact?","They should not be touched or otherwise played with without consent.",("Not to be touched","played with or otherwise molested without consent"),"complete_newcomer_contact_expectation","The answer turns a list fragment into a concise complete sentence while retaining the source’s consent boundary."),
 keep("fact-1de93de4783e8a1c41db",BOTTOM_GUIDE,"If your fingers begin to feel tingly tell your rigger immediately","retain_immediate_tingling_report_action","The existing row gives an unambiguous, time-sensitive action for a possible circulation or nerve warning."),
 paraphrase("fact-d54ba75e58ecb1508618",NEWNESS,"unless you actually want to and consent to it","What two conditions does WykD’s newcomer article give for a new submissive to follow an instruction in a D/s context?","They must actually want to follow the instruction and consent to it.",("actually want to","consent to it"),"complete_submissive_agency_conditions","The answer restates the two source conditions as a complete sentence with an explicit subject."),
 paraphrase("fact-7dedf38874e487d636c9",NEWNESS,"The only time you will is when you have actual consent from the individual","When does WykD’s newcomer article say a dominant has the right to tell a submissive what to do?","Only when that individual has actually consented.",("only time","actual consent from the individual"),"complete_dominant_authority_condition","The answer removes a context-dependent clause and states the individual-consent condition directly."),
 keep("fact-597fa7fd78d5fdcb4f10",BOTTOM_GUIDE,"communicate and give feedback before, during and after bondage","retain_three_stage_bottom_feedback_timing","The existing row clearly preserves the guide’s useful before-during-after communication rule."),
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES=dict(zip(EXPECTED_SELECTION,(50,104,267,324,333,362,388,390)))
PROJECTED_SELECTION_BASELINE={"agency_and_responsibility_rows_selected":8,"description":"isolated cumulative training projection through context-merit v66","direct_rows_without_prior_curation":159,"rows":537,"sha256":"36447ae0664fa198e3d1d12d38f7b7fb503e2ad981ea366a408a45988e1e0a0a"}
ISOLATED_PROJECTION={"active_after_context_merit_v66":499,"active_after_this_tranche":499,"build_script":"build_curated_qa.py","new_drops_applied":0,"new_edits_applied":5,"output_rows":537,"output_sha256":"03981c219dfb2fbfcb885a24e55dcf42712792a4c2aeff0e2c32f4c09900b863","prior_pending_addition_fact_ids_preserved":36,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":3,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS,SEALED_EVAL_PATHS=previous.PRODUCTION_INPUTS,previous.SEALED_EVAL_PATHS;PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
def prior_decision_artifacts():
 out=[]
 for v in range(1,67):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def build_projection(o,r,c):previous.build_projection(o,r,c)
def selected(rows):
 by={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)};out=[{"active_index":by[f][0],"row":by[f][1],"features":CORE.risk_features(by[f][1])} for f in EXPECTED_SELECTION]
 if {x["row"]["fact_id"]:x["active_index"] for x in out}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v67 candidate drift")
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
 with tempfile.TemporaryDirectory(prefix=".v66-projection-",dir=OUT_DIR) as t:
  ds=Path(t)/"v66.jsonl";rp=Path(t)/"r.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=537 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v66 projection drift")
  with patched_base(ds):BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:
  row.update(schema="context-merit-audit-v67",review_pass="agency_and_responsibility_answer_reaudit",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":537,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True})
  spec=next(s for s in SPECS if s["fact_id"]==row["fact_id"])
  if spec.get("support_type")=="manual_paraphrase":row["paraphrase_rationale"]=spec["paraphrase_rationale"]
 write_jsonl(AUDIT,audits)
 curations=read_jsonl(CURATION)
 for row in curations:
  spec=next(s for s in SPECS if s["fact_id"]==row["fact_id"])
  if spec.get("support_type")=="manual_paraphrase":row.update(support_type="manual_paraphrase",paraphrase_rationale=spec["paraphrase_rationale"])
 write_jsonl(CURATION,curations);report=json.loads(REPORT.read_text());report["schema"]="context-merit-audit-report-v67";report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=537,eligible_unreviewed_rows=0,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=8,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"remaining direct rows about naming, newcomer agency, physical-contact consent, and rigger/bottom responsibilities with list-fragment or dependent-clause answers","score":"manual full-source completeness, agency clarity, and actionability review","tie_break":"active projection order"})
 report["audit"]["rows"]=len(audits);report["audit"]["sha256"]=file_sha256(AUDIT);report["new_pending_curation"]["sha256"]=file_sha256(CURATION);report["new_pending_curation"]["edit_support_types"]={"extractive":0,"manual_paraphrase":5};report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()];REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
