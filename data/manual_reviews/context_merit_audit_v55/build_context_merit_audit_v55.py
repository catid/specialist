#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v54 projection rows in v55."""
from __future__ import annotations
import contextlib,json,sys,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data"
V54_DIR=DATA/"manual_reviews/context_merit_audit_v54";sys.path[:0]=[str(ROOT),str(V54_DIR)]
import build_context_merit_audit_v54 as previous
BASE,CORE=previous.BASE,previous.CORE;EVIDENCE_PATCH_MODULE=previous.EVIDENCE_PATCH_MODULE
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v55.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v55.jsonl";REPORT=OUT_DIR/"report_context_merit_v55.json"
REVIEWER,REVIEWED_AT="codex-context-merit-audit-v55","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION
file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl
CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,55))
CONTEXT_AUDITS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"context_merit_audit_v{v}.jsonl" for v in range(1,55))
def raw(n):return DATA/"raw"/n

SPECS=(
 {"fact_id":"fact-7c62e275e3a4b24e9d59","source_path":DATA/"rope_topia_manual_v1.jsonl","marker":"<loc>https://rope-topia.com/2012/10/ichinawa-ippon-me-no-nawa-and-one-rope/</loc>","decision":"edit","question":"Where can I find Rope-topia’s post “Ichinawa, Ippon me no nawa and One rope”?","answer":"https://rope-topia.com/2012/10/ichinawa-ippon-me-no-nawa-and-one-rope/","allow_document_sha_mismatch":True,"reason_code":"naturalize_owner_requested_resource_lookup","reason":"The edit retains the sitemap-backed owner-requested resource URL while phrasing the lookup as a natural user question."},
 {"fact_id":"fact-0e51d9584ecb0ef8f14d","source_path":raw("rope365_c89abf7c3a5c30e1.json"),"marker":"it will add some slack but not untie completely","decision":"keep","reason_code":"quick_release_tail_continuation_caveat","reason":"The answer explains an important limitation when a quick release is incorporated into a continuing tie."},
 {"fact_id":"fact-19cb4bbddc40bd77dacf","source_path":raw("kinbakutoday_7aa19131bb45f119.json"),"marker":"The most important point is having an affinity towards rope","decision":"edit","question":"According to Randa Mai, what qualities should a rope artist bring to bondage?","answer":"an affinity for rope, attentive communication, and accountability for the bound person’s physical and mental state","support_type":"manual_paraphrase","paraphrase_support_fragments":["affinity towards rope","communication must be there","fully accountable for the woman’s physical and mental state"],"paraphrase_rationale":"The answer condenses three adjacent, explicitly stated responsibilities and replaces gender-specific wording with an equivalent role description.","reason_code":"replace_single_trait_with_full_attributed_principle","reason":"The edit turns a narrow sadist-versus-affinity prompt into the interview's more useful combined principle of craft, communication, and care."},
 {"fact_id":"fact-d4e83b39d08de0c48f56","source_path":raw("rope365_d9c48a4547717047.json"),"marker":"asking if and how you can help","decision":"keep","reason_code":"ask_before_handling_another_persons_rope","reason":"The answer gives a concise etiquette rule: ask before helping with another person's rope ritual."},
 {"fact_id":"fact-7dedf38874e487d636c9","source_path":raw("wykd_944e4e6d621a97c9.json"),"marker":"when you have actual consent from the individual","decision":"keep","reason_code":"authority_requires_specific_consent","reason":"The answer directly counters assumed D/s authority and grounds instructions in the individual's actual consent."},
 {"fact_id":"fact-96ca7f10d8c5800dee8d","source_path":DATA/"rope_resource_manual_v1.jsonl","marker":"Nylon is more slippery than jute or hemp","decision":"keep","reason_code":"manufacturer_attributed_nylon_friction_limit","reason":"The explicitly attributed material-property warning is directly relevant to adapting technique for nylon rope."},
 {"fact_id":"fact-4634be1b1daf9ac325da","source_path":raw("rope_resources_v1/rope365__7b5d548036392d65fec7.json"),"marker":"visualize what you are about to do and figure out what could go wrong","decision":"keep","reason_code":"pre_scene_incident_visualization","reason":"Mentally rehearsing failure scenarios is practical preparation for emergency response."},
 {"fact_id":"fact-e4607b09df701ffca88e","source_path":raw("rope365_c89abf7c3a5c30e1.json"),"marker":"great in case of emergency, or for ties that may need adjustments along the way","decision":"keep","reason_code":"quick_release_use_cases","reason":"The answer identifies the two practical situations where a less-solid quick release may be appropriate."},
 {"fact_id":"fact-5737467ffbd1d79335da","source_path":raw("rope365_c89abf7c3a5c30e1.json"),"marker":"monitoring the capacity to open and close the hands","decision":"keep","reason_code":"upper_limb_motor_function_monitoring","reason":"Monitoring hand opening and closing is concrete, actionable nerve-risk guidance during restrictive arm ties."},
 {"fact_id":"fact-066e3cb9daabfe6bb40c","source_path":raw("anatomiestudio_27ecdd4d7c9a5560.json"),"marker":"The person wishing to initiate an act or change an act is responsible for initiating the conversation about consent","decision":"keep","reason_code":"initiator_starts_consent_conversation","reason":"The answer clearly assigns responsibility for renewed consent when an activity starts or changes."},
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES=dict(zip(EXPECTED_SELECTION,(433,223,229,66,389,515,289,39,393,113)))
PROJECTED_SELECTION_BASELINE={"description":"isolated cumulative training projection through context-merit v54","direct_rows_without_prior_curation":241,"eligible_unreviewed_direct_rows":16,"prior_context_reviewed_direct_rows_excluded":225,"rows":541,"sha256":"2a97f1131e6ee0436064add802eefd6f07bd7e8110481c9afaccae1efc806442"}
ISOLATED_PROJECTION={"active_after_context_merit_v54":503,"active_after_this_tranche":503,"build_script":"build_curated_qa.py","new_drops_applied":0,"new_edits_applied":2,"output_rows":541,"output_sha256":"cbe89653c42b6b0a871f64f7627610ce2edd86bfb30e92f786e4997585ad5cf6","prior_pending_addition_fact_ids_preserved":37,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":8,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS,SEALED_EVAL_PATHS=previous.PRODUCTION_INPUTS,previous.SEALED_EVAL_PATHS
PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
def prior_decision_artifacts():
 out=[]
 for v in range(1,55):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def build_projection(o,r,c):previous.build_projection(o,r,c)
def prior_reviewed_fact_ids():return {r["fact_id"] for p in CONTEXT_AUDITS for r in read_jsonl(p)}
def ranked_unreviewed_direct(rows):
 reviewed=prior_reviewed_fact_ids();c=[]
 for i,row in enumerate(rows,1):
  if row.get("curation") or row["fact_id"] in reviewed:continue
  f=CORE.risk_features(row);c.append((-f["risk_score"],f["question_tokens"],f["answer_tokens"],row["fact_id"],i,row,f))
 c.sort(key=lambda x:x[:4]);ranked=[{"active_index":x[4],"row":x[5],"features":x[6]} for x in c]
 if len(ranked)!=16:raise ValueError(f"v55 candidate drift: {len(ranked)}")
 if tuple(x["row"]["fact_id"] for x in ranked[:10])!=EXPECTED_SELECTION:raise ValueError("v55 selection drift")
 return ranked
def selected_ranked(rows):return ranked_unreviewed_direct(rows)[:10],0,0
@contextlib.contextmanager
def patched_base(ds):
 rep={"OUT_DIR":OUT_DIR,"AUDIT":AUDIT,"CURATION":CURATION,"REPORT":REPORT,"REVIEWER":REVIEWER,"REVIEWED_AT":REVIEWED_AT,"CONTEXT_CURATIONS":CONTEXT_CURATIONS,"SPECS":SPECS,"EXPECTED_SELECTION":EXPECTED_SELECTION,"ISOLATED_PROJECTION":ISOLATED_PROJECTION};orig={n:getattr(BASE,n) for n in rep};ranking,active,evidence=CORE.ranked_unreviewed,CORE.ACTIVE_DATASET,EVIDENCE_PATCH_MODULE.source_evidence
 try:
  for n,v in rep.items():setattr(BASE,n,v)
  CORE.ranked_unreviewed=selected_ranked;CORE.ACTIVE_DATASET=ds;EVIDENCE_PATCH_MODULE.source_evidence=previous.previous.previous.previous.previous.previous.previous.source_evidence;yield
 finally:
  EVIDENCE_PATCH_MODULE.source_evidence=evidence;CORE.ACTIVE_DATASET=active;CORE.ranked_unreviewed=ranking
  for n,v in orig.items():setattr(BASE,n,v)
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v54-projection-",dir=OUT_DIR) as t:
  ds=Path(t)/"v54.jsonl";rp=Path(t)/"r.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=541 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v54 projection drift")
  with patched_base(ds):BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:
  row.update(schema="context-merit-audit-v55",review_pass="first_context_merit_review_of_v54_projection_row",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":541,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":False})
  spec=next(s for s in SPECS if s["fact_id"]==row["fact_id"])
  if spec.get("support_type")=="manual_paraphrase":row["paraphrase_rationale"]=spec["paraphrase_rationale"]
 write_jsonl(AUDIT,audits)
 curations=read_jsonl(CURATION)
 for row in curations:
  spec=next(s for s in SPECS if s["fact_id"]==row["fact_id"])
  if spec.get("support_type")=="manual_paraphrase":row.update(support_type="manual_paraphrase",paraphrase_rationale=spec["paraphrase_rationale"])
 write_jsonl(CURATION,curations);report=json.loads(REPORT.read_text());report["schema"]="context-merit-audit-report-v55"
 report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=541,eligible_unreviewed_rows=16,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=10,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"the row survives the v54 projection, has no prior curation metadata, and its fact_id has no context-merit decision in v1 through v54","score":"short_question_points + 3*pronoun_count + bare_answer_points + named_person_trivia_points","tie_break":"risk_score descending, question tokens ascending, answer tokens ascending, fact_id ascending"})
 report["audit"]["rows"]=len(audits);report["audit"]["sha256"]=file_sha256(AUDIT);report["new_pending_curation"]["sha256"]=file_sha256(CURATION);report["new_pending_curation"]["edit_support_types"]={"extractive":1,"manual_paraphrase":1};report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()]
 REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
