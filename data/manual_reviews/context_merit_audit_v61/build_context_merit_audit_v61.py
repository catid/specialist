#!/usr/bin/env python3
"""Naturalize remaining generic name/term questions in direct rows v61."""
from __future__ import annotations
import contextlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V60_DIR=DATA/"manual_reviews/context_merit_audit_v60";sys.path[:0]=[str(ROOT),str(V60_DIR)]
import build_context_merit_audit_v60 as previous
BASE,CORE=previous.BASE,previous.CORE;EVIDENCE_PATCH_MODULE=previous.EVIDENCE_PATCH_MODULE;OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v61.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v61.jsonl";REPORT=OUT_DIR/"report_context_merit_v61.json";REVIEWER,REVIEWED_AT="codex-context-merit-audit-v61","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION;file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl
CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,61));CONTEXT_AUDITS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"context_merit_audit_v{v}.jsonl" for v in range(1,61))
def raw(n):return DATA/"raw"/n
def edit(fact,path,marker,question,answer,code):return {"fact_id":fact,"source_path":path,"marker":marker,"decision":"edit","question":question,"answer":answer,"reason_code":code,"reason":"The edit replaces generic name/term wording with the named source context while retaining the exact supported answer."}
SPECS=(
 edit("fact-a54c90c5b9fb003b1d0b",raw("rope365_9a5c5810310fa0f0.json"),"paradox of patterns and presence","What does Rope365 call the tension between learning patterns and remaining present with a partner?","paradox of patterns and presence","identify_rope365_improvisation_paradox"),
 edit("fact-11facb540676c51e7ebb",raw("rope365_8ae9e3d93b31601b.json"),"Predicament – A situation","What term does Rope365 use for a tie that offers agency only among painful, difficult, or otherwise bad options?","Predicament","identify_rope365_predicament_term"),
 edit("fact-db4e6b687bd1fba0dde5",raw("rope365_25f1b23eb40be00e.json"),"family of knots used to extend rope is called bends","What does Rope365 call the family of knots used to extend rope?","bends","identify_rope365_bends_term"),
 edit("fact-c49ef46be45ea5bf079e",raw("kinbakutoday_dc711609dfc7a35f.json"),"create the nawajiri, the point of connection, control and communication","In “Tying the One Rope Gote,” what term names the point of connection, control, and communication created by catching the ropes?","nawajiri","identify_one_rope_gote_term"),
 edit("fact-4a5cebf4e405af18a2dd",raw("rope365_f43c9fde09431a5f.json"),"Teardrop Harness","Which chest-harness design does Rope365’s Weavings lesson identify as using weaving for strength?","Teardrop Harness","identify_rope365_teardrop_harness"),
 edit("fact-f4886b3a29c4d198cd92",raw("kinbakutoday_e55b7fa7c543e266.json"),"set foundation chest harness (TK)","In the Kasumi Hourai interview, what name is given to Classical Kinbaku’s set foundation chest harness?","TK","identify_kasumi_interview_tk_term"),
 edit("fact-8f2ac7403fb535b93713",raw("kinbakutoday_e55b7fa7c543e266.json"),"final knot that holds the obi of a kimono is tied using a honmusubi","According to Kasumi Hourai, which square knot is used for the final knot holding a kimono’s obi?","honmusubi","attribute_honmusubi_to_kasumi_interview"),
 edit("fact-3c85f97bbd348e7d84e0",raw("kinbakutoday_4f9dec06e4af751a.json"),"Fujimi Iku was, in fact, one of the many additional names that Nureki Chimuo used","In “Ghosts of Kinbaku History,” which pseudonym is identified as one Nureki Chimuo used in published work?","Fujimi Iku","identify_ghosts_article_pseudonym_claim"),
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES=dict(zip(EXPECTED_SELECTION,(26,251,252,253,254,255,256,458)))
PROJECTED_SELECTION_BASELINE={"description":"isolated cumulative training projection through context-merit v60","direct_rows_without_prior_curation":201,"generic_term_rows_selected":8,"rows":537,"sha256":"c1277333dd8889cf0cee0e49be5a00d044ace042fad7bbd6ca7489bee955279c"}
ISOLATED_PROJECTION={"active_after_context_merit_v60":499,"active_after_this_tranche":499,"build_script":"build_curated_qa.py","new_drops_applied":0,"new_edits_applied":8,"output_rows":537,"output_sha256":"a8603dae26aec6c1882cb3bc26f5f0644390562fae82998bf46d10653c71d4e7","prior_pending_addition_fact_ids_preserved":36,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":0,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS,SEALED_EVAL_PATHS=previous.PRODUCTION_INPUTS,previous.SEALED_EVAL_PATHS;PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
def prior_decision_artifacts():
 out=[]
 for v in range(1,61):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def build_projection(o,r,c):previous.build_projection(o,r,c)
def selected(rows):
 by={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)};out=[{"active_index":by[f][0],"row":by[f][1],"features":CORE.risk_features(by[f][1])} for f in EXPECTED_SELECTION]
 if {x["row"]["fact_id"]:x["active_index"] for x in out}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v61 candidate drift")
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
 with tempfile.TemporaryDirectory(prefix=".v60-projection-",dir=OUT_DIR) as t:
  ds=Path(t)/"v60.jsonl";rp=Path(t)/"r.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=537 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v60 projection drift")
  with patched_base(ds):BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:row.update(schema="context-merit-audit-v61",review_pass="generic_name_and_term_naturalness_reaudit",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":537,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True})
 write_jsonl(AUDIT,audits);report=json.loads(REPORT.read_text());report["schema"]="context-merit-audit-report-v61";report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=537,eligible_unreviewed_rows=0,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=8,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"remaining direct questions using generic what-is-the-name or according-to-the-text wording","score":"manual source context and question naturalness","tie_break":"active projection order"})
 report["audit"]["rows"]=len(audits);report["audit"]["sha256"]=file_sha256(AUDIT);report["new_pending_curation"]["sha256"]=file_sha256(CURATION);report["new_pending_curation"]["edit_support_types"]={"extractive":8,"manual_paraphrase":0};report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()];REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
