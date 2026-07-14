#!/usr/bin/env python3
"""Replace remaining text/excerpt prompts with precise article attribution v68."""
from __future__ import annotations
import contextlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V67_DIR=DATA/"manual_reviews/context_merit_audit_v67";sys.path[:0]=[str(ROOT),str(V67_DIR)]
import build_context_merit_audit_v67 as previous
BASE,CORE=previous.BASE,previous.CORE;EVIDENCE_PATCH_MODULE=previous.EVIDENCE_PATCH_MODULE;OUT_DIR=Path(__file__).resolve().parent;AUDIT=OUT_DIR/"context_merit_audit_v68.jsonl";CURATION=OUT_DIR/"pending_curation_context_merit_v68.jsonl";REPORT=OUT_DIR/"report_context_merit_v68.json";REVIEWER,REVIEWED_AT="codex-context-merit-audit-v68","2026-07-14"
RESOURCE_MANIFEST=previous.RESOURCE_MANIFEST;ACTIVE_DATASET,ACTIVE_REPORT=previous.ACTIVE_DATASET,previous.ACTIVE_REPORT;ACTIVE_CURATIONS=previous.ACTIVE_CURATIONS;PRIOR_PENDING_ADDITIONS=previous.PRIOR_PENDING_ADDITIONS;QUALITY_MERIT_CURATION,TASUKI_CURATION=previous.QUALITY_MERIT_CURATION,previous.TASUKI_CURATION;file_sha256,text_sha256,read_jsonl,write_jsonl=previous.file_sha256,previous.text_sha256,previous.read_jsonl,previous.write_jsonl
CONTEXT_CURATIONS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"pending_curation_context_merit_v{v}.jsonl" for v in range(1,68));CONTEXT_AUDITS=tuple(DATA/"manual_reviews"/f"context_merit_audit_v{v}"/f"context_merit_audit_v{v}.jsonl" for v in range(1,68))
def raw(n):return DATA/"raw"/n
def keep(fact,path,marker,code,reason):return {"fact_id":fact,"source_path":path,"marker":marker,"decision":"keep","reason_code":code,"reason":reason}
def edit(fact,path,marker,question,answer,code,reason,**extra):return {"fact_id":fact,"source_path":path,"marker":marker,"decision":"edit","question":question,"answer":answer,"reason_code":code,"reason":reason,**extra}
REACTIONS=raw("esinem_77368ccdc66acad0.json");YUKIMURA=raw("kinbakutoday_381214111022a952.json");ERA2=raw("kinbakutoday_82071039cb003b58.json")
SPECS=(
 keep("fact-2f3e2526a82142759634",YUKIMURA,"always the rope bottom that will lose the feeling of connection first","retain_attributed_connection_warning","The existing question attributes the observation to Yukimura and the answer clearly identifies who notices lost connection first."),
 edit("fact-0a9bde892f63c70189ff",REACTIONS,"BD: Bondage & Discipline","In AmeKitsune’s “On Rope Performance Reactions,” what three paired subsets are placed under the BDSM acronym?","BD: Bondage & Discipline; Ds: Domination & submission; SM: Sadism & Masochism","replace_in_the_text_with_named_article","The revised question replaces the contextless phrase “the text” with the reprinted article and author, while the answer preserves the source’s labels.",evidence_end_marker="SM: Sadism & Masochism"),
 edit("fact-18128e1fc133eaf19e89",REACTIONS,"That’s Risk Aware Consensual Kink","In AmeKitsune’s “On Rope Performance Reactions,” what does the acronym RACK stand for?","Risk Aware Consensual Kink","attribute_rack_expansion_to_named_article","The revised wording retains the useful acronym expansion while making its source explicit."),
 edit("fact-d0ef49d5f4d72d637ebe",YUKIMURA,"Ryū. A noun used as a suffix to mean a fashion, a way, a style, manner, or an individual’s school of thought","How does the article “Yukimura Ryū” define the suffix ryū?","a fashion, a way, a style, manner, or an individual’s school of thought","replace_excerpt_with_named_article","The revised question replaces an unspecified excerpt with the article title and retains its exact definition."),
 edit("fact-365b99f4c4b01456b018",ERA2,"Kinbaku has aspects of art, evidenced by its use in traditional Kabuki and Bunraku","In Ugo’s historical account within “Kinbaku – An Evolving Era – Part 2,” which traditional performance arts does he cite as evidence of kinbaku’s artistic aspects?","traditional Kabuki and Bunraku","attribute_performance_art_claim_to_ugo","The revised question identifies the speaker and debate article behind the historical-artistic claim."),
)
EXPECTED_SELECTION=tuple(s["fact_id"] for s in SPECS);PROJECTED_ACTIVE_INDICES=dict(zip(EXPECTED_SELECTION,(42,157,237,246,477)))
PROJECTED_SELECTION_BASELINE={"contextless_source_prompts_selected":5,"description":"isolated cumulative training projection through context-merit v67","direct_rows_without_prior_curation":154,"rows":537,"sha256":"03981c219dfb2fbfcb885a24e55dcf42712792a4c2aeff0e2c32f4c09900b863"}
ISOLATED_PROJECTION={"active_after_context_merit_v67":499,"active_after_this_tranche":499,"build_script":"build_curated_qa.py","new_drops_applied":0,"new_edits_applied":4,"output_rows":537,"output_sha256":"7140b838c0dd8b39a1f458ed23567982c5c2432e04ccdcd1e0d38255418e6d2d","prior_pending_addition_fact_ids_preserved":36,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":1,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS,SEALED_EVAL_PATHS=previous.PRODUCTION_INPUTS,previous.SEALED_EVAL_PATHS;PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)
def prior_decision_artifacts():
 out=[]
 for v in range(1,68):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}";out.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(out)
def build_projection(o,r,c):previous.build_projection(o,r,c)
def selected(rows):
 by={r["fact_id"]:(i,r) for i,r in enumerate(rows,1)};out=[{"active_index":by[f][0],"row":by[f][1],"features":CORE.risk_features(by[f][1])} for f in EXPECTED_SELECTION]
 if {x["row"]["fact_id"]:x["active_index"] for x in out}!=PROJECTED_ACTIVE_INDICES:raise ValueError("v68 candidate drift")
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
 with tempfile.TemporaryDirectory(prefix=".v67-projection-",dir=OUT_DIR) as t:
  ds=Path(t)/"v67.jsonl";rp=Path(t)/"r.json";build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=537 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]:raise ValueError("v67 projection drift")
  with patched_base(ds):BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:row.update(schema="context-merit-audit-v68",review_pass="contextless_source_prompt_reaudit",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":537,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":True})
 write_jsonl(AUDIT,audits);report=json.loads(REPORT.read_text());report["schema"]="context-merit-audit-report-v68";report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=537,eligible_unreviewed_rows=0,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=5,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"remaining direct questions using contextless source labels such as text or excerpt, plus adjacent attribution control","score":"manual full-source provenance and standalone question review","tie_break":"active projection order"})
 report["audit"]["rows"]=len(audits);report["audit"]["sha256"]=file_sha256(AUDIT);report["new_pending_curation"]["sha256"]=file_sha256(CURATION);report["new_pending_curation"]["edit_support_types"]={"extractive":4,"manual_paraphrase":0};report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()];REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
