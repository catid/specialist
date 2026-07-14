#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v53 projection rows in v54."""
from __future__ import annotations
import contextlib, json, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]; DATA=ROOT/"data"
V53_DIR=DATA/"manual_reviews/context_merit_audit_v53";sys.path[:0]=[str(ROOT),str(V53_DIR)]
import build_context_merit_audit_v53 as previous
BASE,CORE=previous.BASE,previous.CORE;EVIDENCE_PATCH_MODULE=previous.EVIDENCE_PATCH_MODULE
OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v54.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v54.jsonl";REPORT=OUT_DIR/"report_context_merit_v54.json"
REVIEWER,REVIEWED_AT="codex-context-merit-audit-v54","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION
file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl
CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,54))
CONTEXT_AUDITS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"context_merit_audit_v{v}.jsonl" for v in range(1,54))
def raw(n):return DATA/"raw"/n

SPECS=(
 {"fact_id":"fact-f2f2561a20d4f7b3ef11","source_path":raw("rope_resources_v1/rope365__b602b6493b5eb6f55206.json"),"marker":"safety call (someone who will check back on you after a certain period of time)","decision":"keep","reason_code":"new_partner_safety_call_definition","reason":"The definition makes a practical new-partner safety measure understandable and actionable."},
 {"fact_id":"fact-d64b1477fee41ef24905","source_path":raw("rope_resources_v1/rope365__b602b6493b5eb6f55206.json"),"marker":"contact someone a few days after a session to open the channel for feedback","decision":"keep","reason_code":"delayed_post_scene_feedback_followup","reason":"Following up later accommodates partners who cannot immediately process emotions or feedback after a scene."},
 {"fact_id":"fact-bda97c904a9412765ad2","source_path":raw("rope_resources_v1/rope365__7b5d548036392d65fec7.json"),"marker":"quickly untie anything that interferes with breathing","decision":"keep","reason_code":"rapid_release_of_breathing_interference","reason":"The answer states a critical emergency-readiness requirement for chest compression and restrictive positions."},
 {"fact_id":"fact-e809dfeb985b73cdfc0f","source_path":ROOT/"sources/manual_facts/resource_group_b.jsonl","marker":"diameter is approximate (around 4” to 4.4”)","decision":"drop","reason_code":"incidental_bamboo_product_dimension","reason":"A current product's approximate diameter range is vendor-listing trivia and does not establish suspension safety or capacity."},
 {"fact_id":"fact-872e45e0ff772b90fa8e","source_path":raw("kinbakutoday_b2454d5b6578b8c6.json"),"marker":"ease of tying, connection, aesthetics etc","decision":"keep","reason_code":"attributed_multifactor_tying_decision","reason":"The attributed answer captures a useful teaching principle: choices are evaluated across technique, connection, and aesthetics."},
 {"fact_id":"fact-d7160c670538be792592","source_path":raw("kinbakutoday_dfc9527c49ca8ad6.json"),"marker":"tea ceremony, calligraphy and Budō","decision":"keep","reason_code":"attributed_orientalist_association_critique","reason":"The explicitly attributed list supports the essay's substantive warning against inventing traditional lineages for kinbaku."},
 {"fact_id":"fact-e2c7f7f358e7040deee0","source_path":raw("kinbakutoday_73b16e835ab63cc2.json"),"marker":"less as a question of “what do I want?” and more of a question of “who is my partner and what do they need?”","decision":"edit","question":"What shift does Rope and Kindness propose for thinking about a rope scene?","answer":"less as a question of “what do I want?” and more of a question of “who is my partner and what do they need?”","reason_code":"replace_negative_caveat_with_partner_centered_shift","reason":"The edit replaces a narrow disclaimer with the article's substantive contrast between self-focused wants and partner-focused needs."},
 {"fact_id":"fact-6a31ec21b3340a7860de","source_path":raw("anatomiestudio_144932682af9c846.json"),"marker":"assign a specific rope to a specific person (and role)","decision":"keep","reason_code":"partner_specific_rope_hygiene_strategy","reason":"Assigning rope to a person and role is practical harm-reduction guidance when natural fibre cannot be reliably cleaned."},
 {"fact_id":"fact-eee35bea21f484fef5bf","source_path":raw("kinbakutoday_432c8adfc1abe686.json"),"marker":"riggers responsibility to ensure what they are doing, is right for your fitness level and body type","decision":"keep","reason_code":"rigger_body_specific_suitability_responsibility","reason":"The answer gives important body-specific safety responsibility while the full source also preserves the bottom's communication role."},
 {"fact_id":"fact-20da52aaee9ec65a1b35","source_path":raw("kinbakutoday_73b16e835ab63cc2.json"),"marker":"presumes the two sides want different things","decision":"drop","reason_code":"redundant_abstract_negotiation_framing","reason":"This abstract premise is redundant once the same paragraph's clearer partner-needs contrast is retained as the v54 edit."},
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES=dict(zip(EXPECTED_SELECTION,(183,221,189,13,327,475,204,234,249,23)))
PROJECTED_SELECTION_BASELINE={"description":"isolated cumulative training projection through context-merit v53","direct_rows_without_prior_curation":244,"eligible_unreviewed_direct_rows":26,"prior_context_reviewed_direct_rows_excluded":218,"rows":543,"sha256":"16abbe4e68c83fe1e43341b6a5bdce4b3be4377879542ec0074a5e92ca37f264"}
ISOLATED_PROJECTION={"active_after_context_merit_v53":505,"active_after_this_tranche":503,"build_script":"build_curated_qa.py","new_drops_applied":2,"new_edits_applied":1,"output_rows":541,"output_sha256":"2a97f1131e6ee0436064add802eefd6f07bd7e8110481c9afaccae1efc806442","prior_pending_addition_fact_ids_preserved":37,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":7,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS,SEALED_EVAL_PATHS=previous.PRODUCTION_INPUTS,previous.SEALED_EVAL_PATHS
PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
def prior_decision_artifacts():
 out=[]
 for v in range(1,54):
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
 if len(ranked)!=26:raise ValueError(f"v54 candidate drift: {len(ranked)}")
 if tuple(x["row"]["fact_id"] for x in ranked[:10])!=EXPECTED_SELECTION:raise ValueError("v54 selection drift")
 return ranked
def selected_ranked(rows):return ranked_unreviewed_direct(rows)[:10],0,0
@contextlib.contextmanager
def patched_base(ds):
 rep={"OUT_DIR":OUT_DIR,"AUDIT":AUDIT,"CURATION":CURATION,"REPORT":REPORT,"REVIEWER":REVIEWER,"REVIEWED_AT":REVIEWED_AT,"CONTEXT_CURATIONS":CONTEXT_CURATIONS,"SPECS":SPECS,"EXPECTED_SELECTION":EXPECTED_SELECTION,"ISOLATED_PROJECTION":ISOLATED_PROJECTION};orig={n:getattr(BASE,n) for n in rep};ranking,active,evidence=CORE.ranked_unreviewed,CORE.ACTIVE_DATASET,EVIDENCE_PATCH_MODULE.source_evidence
 try:
  for n,v in rep.items():setattr(BASE,n,v)
  CORE.ranked_unreviewed=selected_ranked;CORE.ACTIVE_DATASET=ds;EVIDENCE_PATCH_MODULE.source_evidence=previous.previous.previous.previous.previous.previous.source_evidence;yield
 finally:
  EVIDENCE_PATCH_MODULE.source_evidence=evidence;CORE.ACTIVE_DATASET=active;CORE.ranked_unreviewed=ranking
  for n,v in orig.items():setattr(BASE,n,v)
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v53-projection-",dir=OUT_DIR) as t:
  ds=Path(t)/"v53.jsonl";rp=Path(t)/"r.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=543 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v53 projection drift")
  with patched_base(ds):BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:row.update(schema="context-merit-audit-v54",review_pass="first_context_merit_review_of_v53_projection_row",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":543,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":False})
 write_jsonl(AUDIT,audits);report=json.loads(REPORT.read_text());report["schema"]="context-merit-audit-report-v54"
 report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=543,eligible_unreviewed_rows=26,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=10,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"the row survives the v53 projection, has no prior curation metadata, and its fact_id has no context-merit decision in v1 through v53","score":"short_question_points + 3*pronoun_count + bare_answer_points + named_person_trivia_points","tie_break":"risk_score descending, question tokens ascending, answer tokens ascending, fact_id ascending"})
 report["audit"]["rows"]=len(audits);report["audit"]["sha256"]=file_sha256(AUDIT);report["new_pending_curation"]["sha256"]=file_sha256(CURATION);report["new_pending_curation"]["edit_support_types"]={"extractive":1,"manual_paraphrase":0};report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()]
 REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
