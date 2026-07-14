#!/usr/bin/env python3
"""Replace ambiguous pronoun and fragment answers in concise technique rows v66."""
from __future__ import annotations
import contextlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V65_DIR=DATA/"manual_reviews/context_merit_audit_v65";sys.path[:0]=[str(ROOT),str(V65_DIR)]
import build_context_merit_audit_v65 as previous
BASE,CORE=previous.BASE,previous.CORE;EVIDENCE_PATCH_MODULE=previous.EVIDENCE_PATCH_MODULE;OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v66.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v66.jsonl";REPORT=OUT_DIR/"report_context_merit_v66.json";REVIEWER,REVIEWED_AT="codex-context-merit-audit-v66","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION;file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl
CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,66));CONTEXT_AUDITS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"context_merit_audit_v{v}.jsonl" for v in range(1,66))
def raw(n):return DATA/"raw"/n
def keep(fact,path,marker,code,reason):return {"fact_id":fact,"source_path":path,"marker":marker,"decision":"keep","reason_code":code,"reason":reason}
def paraphrase(fact,path,marker,question,answer,fragments,code,reason):return {"fact_id":fact,"source_path":path,"marker":marker,"decision":"edit","question":question,"answer":answer,"support_type":"manual_paraphrase","paraphrase_support_fragments":fragments,"paraphrase_rationale":reason,"reason_code":code,"reason":reason}
NEW_PARTNER=raw("anatomiestudio_451ac66001188a42.json");QUICK=raw("rope365_c89abf7c3a5c30e1.json");SINGLE=raw("rope365_f250f228cc370052.json");MORE_KNOTS=raw("rope365_25f1b23eb40be00e.json")
SPECS=(
 keep("fact-9e1d84a06b0b7efe1dc3",SINGLE,"A single column tie is a cuff that will not tighten when we pull on the tail","retain_clear_single_column_definition","The attributed row already gives a concise functional definition rather than merely naming a pattern."),
 paraphrase("fact-1a28fe890ccaa23e1d53",NEW_PARTNER,"Safety doesn’t mean eliminating all risk—it means understanding and managing it.","How does Anatomie Studio define safety when tying with a new rope partner?","understanding and managing risk",("eliminating all risk","understanding and managing it"),"replace_ambiguous_safety_pronoun","The answer replaces an unclear pronoun with the exact concept it refers to: risk."),
 keep("fact-392013e296aa55be8770",SINGLE,"“Column” refers to anything you can tie around","retain_clear_column_terminology","The source-specific question and concise answer clearly define Rope365’s broad use of the term column."),
 paraphrase("fact-907df6b7f8557379cae0",QUICK,"extra careful in case it unties accidentally with movement","What accidental failure does Rope365 warn can occur when a quick-release knot moves?","The quick-release knot may untie accidentally with movement.",("extra careful","unties accidentally with movement"),"replace_quick_release_failure_pronoun","The answer names the quick-release knot explicitly instead of relying on an ambiguous pronoun."),
 paraphrase("fact-4d68b129fbee9d884b45",MORE_KNOTS,"we need the knot to disappear completely, without leaving a tangle","What distinguishes Rope365’s exploding knots from an ordinary quick release?","They disappear completely without leaving a tangle.",("disappear completely","without leaving a tangle"),"complete_exploding_knot_answer_sentence","The answer turns a bare verb fragment into a complete sentence without changing the source-supported distinction."),
 paraphrase("fact-034e26ba2559271b6ca6",NEW_PARTNER,"good practice to use your “stop” signal before it’s truly needed","What does Anatomie Studio recommend doing with an agreed stop signal before a new-partner rope scene?","Practise using it before it is truly needed.",("good practice to use","stop” signal","before it’s truly needed"),"clarify_stop_signal_rehearsal","The answer makes explicit that partners should rehearse the agreed signal before an urgent situation arises."),
 paraphrase("fact-0e51d9584ecb0ef8f14d",QUICK,"continue to tie with the tail and untie a quick release without untying the rest","What happens if a quick-release tail continues into the rest of a tie and the quick release is then undone?","It adds some slack but does not completely untie the continuing tie.",("continue to tie with the tail","add some slack","not untie completely"),"clarify_continuing_tie_quick_release_effect","The revised question and answer name the continuing tie explicitly, resolving what the source’s pronouns refer to."),
 paraphrase("fact-a11454575f9c29e93d81",QUICK,"when you pull a rope through, it will become locked like with the bight of a single column tie","What happens when rope is pulled through the loop of a slipped half hitch?","The slipped half hitch becomes locked, like the bight of a single-column tie.",("pull a rope through","become locked","bight of a single column tie"),"replace_slipped_half_hitch_pronoun","The answer identifies the slipped half hitch as the structure that becomes locked rather than answering only with “it.”"),
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES=dict(zip(EXPECTED_SELECTION,(19,63,128,152,188,192,256,257)))
PROJECTED_SELECTION_BASELINE={"description":"isolated cumulative training projection through context-merit v65","direct_rows_without_prior_curation":165,"pronoun_and_fragment_rows_selected":8,"rows":537,"sha256":"25690dda4b0be5d677e3d2cb7a64013155c5d4156adb84bf34cf459b3659e131"}
ISOLATED_PROJECTION={"active_after_context_merit_v65":499,"active_after_this_tranche":499,"build_script":"build_curated_qa.py","new_drops_applied":0,"new_edits_applied":6,"output_rows":537,"output_sha256":"36447ae0664fa198e3d1d12d38f7b7fb503e2ad981ea366a408a45988e1e0a0a","prior_pending_addition_fact_ids_preserved":36,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":2,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS,SEALED_EVAL_PATHS=previous.PRODUCTION_INPUTS,previous.SEALED_EVAL_PATHS;PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
def prior_decision_artifacts():
 out=[]
 for v in range(1,66):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def build_projection(o,r,c):previous.build_projection(o,r,c)
def selected(rows):
 by={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)};out=[{"active_index":by[f][0],"row":by[f][1],"features":CORE.risk_features(by[f][1])} for f in EXPECTED_SELECTION]
 if {x["row"]["fact_id"]:x["active_index"] for x in out}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v66 candidate drift")
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
 with tempfile.TemporaryDirectory(prefix=".v65-projection-",dir=OUT_DIR) as t:
  ds=Path(t)/"v65.jsonl";rp=Path(t)/"r.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=537 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v65 projection drift")
  with patched_base(ds):BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:
  row.update(schema="context-merit-audit-v66",review_pass="pronoun_and_fragment_answer_reaudit",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":537,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True})
  spec=next(s for s in SPECS if s["fact_id"]==row["fact_id"])
  if spec.get("support_type")=="manual_paraphrase":row["paraphrase_rationale"]=spec["paraphrase_rationale"]
 write_jsonl(AUDIT,audits)
 curations=read_jsonl(CURATION)
 for row in curations:
  spec=next(s for s in SPECS if s["fact_id"]==row["fact_id"])
  if spec.get("support_type")=="manual_paraphrase":row.update(support_type="manual_paraphrase",paraphrase_rationale=spec["paraphrase_rationale"])
 write_jsonl(CURATION,curations);report=json.loads(REPORT.read_text());report["schema"]="context-merit-audit-report-v66";report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=537,eligible_unreviewed_rows=0,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=8,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"remaining concise direct answers whose pronouns or verb fragments depend on hidden source context, plus adjacent clear controls","score":"manual full-source referent clarity and standalone answer review","tie_break":"active projection order"})
 report["audit"]["rows"]=len(audits);report["audit"]["sha256"]=file_sha256(AUDIT);report["new_pending_curation"]["sha256"]=file_sha256(CURATION);report["new_pending_curation"]["edit_support_types"]={"extractive":0,"manual_paraphrase":6};report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()];REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
