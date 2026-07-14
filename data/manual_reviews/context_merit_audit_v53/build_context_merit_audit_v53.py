#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v52 projection rows in v53."""
from __future__ import annotations
import contextlib, json, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V52_DIR = ROOT / "data/manual_reviews/context_merit_audit_v52"
sys.path[:0] = [str(ROOT), str(V52_DIR)]
import build_context_merit_audit_v52 as previous

BASE, CORE = previous.BASE, previous.CORE
EVIDENCE_PATCH_MODULE = previous.EVIDENCE_PATCH_MODULE
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v53.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v53.jsonl"
REPORT = OUT_DIR / "report_context_merit_v53.json"
REVIEWER, REVIEWED_AT = "codex-context-merit-audit-v53", "2026-07-14"
RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST
ACTIVE_DATASET, ACTIVE_REPORT = previous.ACTIVE_DATASET, previous.ACTIVE_REPORT
ACTIVE_CURATIONS = previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS = previous.PRIOR_PENDING_ADDITIONS
QUALITY_MERIT_CURATION, TASUKI_CURATION = previous.QUALITY_MERIT_CURATION, previous.TASUKI_CURATION
file_sha256, text_sha256 = previous.file_sha256, previous.text_sha256
read_jsonl, write_jsonl = previous.read_jsonl, previous.write_jsonl

CONTEXT_CURATIONS = tuple(DATA / "manual_reviews" / f"context_merit_audit_v{v}" /
                          f"pending_curation_context_merit_v{v}.jsonl" for v in range(1, 53))
CONTEXT_AUDITS = tuple(DATA / "manual_reviews" / f"context_merit_audit_v{v}" /
                       f"context_merit_audit_v{v}.jsonl" for v in range(1, 53))
def raw(name: str) -> Path: return DATA / "raw" / name

SPECS = (
 {"fact_id":"fact-ea3c8d2b8a4c06b768e0","source_path":raw("rope_resources_v1/rope365__c1314f53c65df4af2c20.json"),"marker":"outside or in a well-ventilated area","decision":"keep","reason_code":"rope_dust_ventilation_guidance","reason":"Dust-producing rope processing should be done outside or with ventilation; this is direct exposure-reduction guidance."},
 {"fact_id":"fact-4c78d54846efaa4feef0","source_path":DATA/"rope_resource_manual_v1.jsonl","marker":"unable to safeword or otherwise communicate","support_type":"manual_paraphrase","paraphrase_support_fragments":("unable to safeword or otherwise communicate",),"paraphrase_rationale":"The concise answer generalizes the source's statement that people commonly find themselves unable to communicate, without changing its meaning.","decision":"keep","reason_code":"do_not_rely_only_on_safewords","reason":"The answer explains why active monitoring and prior negotiation remain necessary even when a safeword exists."},
 {"fact_id":"fact-3527f9be2d46d3c13e4e","source_path":raw("anatomiestudio_27ecdd4d7c9a5560.json"),"marker":"Consent is not a contract; people can change their minds","decision":"keep","reason_code":"consent_remains_revocable","reason":"The answer clearly states the framework's reason that prior agreement cannot make consent irrevocable."},
 {"fact_id":"fact-907df6b7f8557379cae0","source_path":raw("rope365_c89abf7c3a5c30e1.json"),"marker":"unties accidentally with movement","decision":"keep","reason_code":"quick_release_movement_failure","reason":"The answer identifies the central accidental-release failure mode that users must monitor."},
 {"fact_id":"fact-51dce9783812bf1a03c2","source_path":raw("kinbakutoday_73b16e835ab63cc2.json"),"marker":"understand our partner, our connection, and our selves","decision":"keep","reason_code":"attributed_partner_centered_rope_ethic","reason":"The attributed answer coherently summarizes the article's partner-centered definition of kindness in rope."},
 {"fact_id":"fact-bda90c25c55ec1a107f5","source_path":raw("anatomiestudio_27ecdd4d7c9a5560.json"),"marker":"Especially when transitioning from one activity to another","decision":"keep","reason_code":"transition_specific_consent_checkins","reason":"Activity transitions are a concrete point where consent should be checked and may be renegotiated or withheld."},
 {"fact_id":"fact-6d4688c283637a5abb61","source_path":raw("kinbakutoday_dfc9527c49ca8ad6.json"),"marker":"circular interaction between him and the woman","decision":"edit","question":"According to Ugo's account, how did Yukimura believe kinbaku play should develop?","answer":"through circular, real-time interaction between the person tying and the person being tied","support_type":"manual_paraphrase","paraphrase_support_fragments":("circular interaction between him and the woman","mutual communication in real time between the one tying and the one being tied"),"paraphrase_rationale":"The answer combines the paragraph's named circular-interaction concept with its immediate, gender-neutral explanation of the two participants.","reason_code":"clarify_and_attribute_yukimura_interaction_model","reason":"The edit replaces awkward person-specific wording with a clear, attributed description of the source's mutual-feedback model."},
 {"fact_id":"fact-a45715221b107ff37347","source_path":raw("rope_resources_v1/rope365__7b5d548036392d65fec7.json"),"marker":"Stay with your bound partner at all times","decision":"keep","reason_code":"continuous_bound_partner_supervision","reason":"Continuous presence is critical, direct guidance for responding if breathing, fainting, or another emergency occurs."},
 {"fact_id":"fact-866a8b187a9b5eade0b8","source_path":raw("kinbakutoday_70cbcc84801cccd0.json"),"marker":"Kokoro is in the people who are engaging, communicating, and sharing themselves with each other","decision":"keep","reason_code":"explicitly_attributed_kokoro_thesis","reason":"The question attributes the essay's central, coherent distinction between human connection and rope technique."},
 {"fact_id":"fact-a11454575f9c29e93d81","source_path":raw("rope365_c89abf7c3a5c30e1.json"),"marker":"when you pull a rope through, it will become locked","decision":"keep","reason_code":"slipped_half_hitch_loop_locking_behavior","reason":"The answer describes the specific mechanical consequence of pulling rope through the loop."},
)
EXPECTED_SELECTION = tuple(s["fact_id"] for s in SPECS)
PROJECTED_ACTIVE_INDICES = dict(zip(EXPECTED_SELECTION,(415,508,537,120,204,390,335,189,418,226)))
PROJECTED_SELECTION_BASELINE={"description":"isolated cumulative training projection through context-merit v52","direct_rows_without_prior_curation":245,"eligible_unreviewed_direct_rows":36,"prior_context_reviewed_direct_rows_excluded":209,"rows":543,"sha256":"21ca3f4e5be03d7dad647ec5a175f67227df064343c6f6a1ba336325ac96637e"}
ISOLATED_PROJECTION={"active_after_context_merit_v52":505,"active_after_this_tranche":505,"build_script":"build_curated_qa.py","new_drops_applied":0,"new_edits_applied":1,"output_rows":543,"output_sha256":"16abbe4e68c83fe1e43341b6a5bdce4b3be4377879542ec0074a5e92ca37f264","prior_pending_addition_fact_ids_preserved":37,"repeat_dataset_byte_identical":True,"reviewed_keep_fact_ids_preserved":9,"sealed_eval_fact_count_reported_by_tooling":612,"unexpected_fact_ids":0,"validated_runs":2}
PRODUCTION_INPUTS, SEALED_EVAL_PATHS = previous.PRODUCTION_INPUTS, previous.SEALED_EVAL_PATHS
PRIOR_PROJECTION_CURATIONS=(*ACTIVE_CURATIONS,QUALITY_MERIT_CURATION,TASUKI_CURATION,*CONTEXT_CURATIONS)

def prior_decision_artifacts():
 paths=[]
 for v in range(1,53):
  d=DATA/"manual_reviews"/f"context_merit_audit_v{v}"
  paths.extend((d/f"context_merit_audit_v{v}.jsonl",d/f"pending_curation_context_merit_v{v}.jsonl",d/f"report_context_merit_v{v}.json"))
 return tuple(paths)
def build_projection(output,report,curations): previous.build_projection(output,report,curations)
def prior_reviewed_fact_ids(): return {r["fact_id"] for p in CONTEXT_AUDITS for r in read_jsonl(p)}
def ranked_unreviewed_direct(rows):
 reviewed=prior_reviewed_fact_ids(); candidates=[]
 for i,row in enumerate(rows,1):
  if row.get("curation") or row["fact_id"] in reviewed: continue
  f=CORE.risk_features(row); candidates.append((-f["risk_score"],f["question_tokens"],f["answer_tokens"],row["fact_id"],i,row,f))
 candidates.sort(key=lambda x:x[:4]); ranked=[{"active_index":x[4],"row":x[5],"features":x[6]} for x in candidates]
 if len(ranked)!=36: raise ValueError(f"v53 candidate drift: {len(ranked)}")
 if tuple(x["row"]["fact_id"] for x in ranked[:10])!=EXPECTED_SELECTION: raise ValueError("v53 selection drift")
 return ranked
def selected_ranked(rows): return ranked_unreviewed_direct(rows)[:10],0,0

@contextlib.contextmanager
def patched_base(projected_dataset):
 replacements={"OUT_DIR":OUT_DIR,"AUDIT":AUDIT,"CURATION":CURATION,"REPORT":REPORT,"REVIEWER":REVIEWER,"REVIEWED_AT":REVIEWED_AT,"CONTEXT_CURATIONS":CONTEXT_CURATIONS,"SPECS":SPECS,"EXPECTED_SELECTION":EXPECTED_SELECTION,"ISOLATED_PROJECTION":ISOLATED_PROJECTION}
 originals={n:getattr(BASE,n) for n in replacements}; ranking=CORE.ranked_unreviewed; active=CORE.ACTIVE_DATASET; evidence=EVIDENCE_PATCH_MODULE.source_evidence
 try:
  for n,v in replacements.items(): setattr(BASE,n,v)
  CORE.ranked_unreviewed=selected_ranked; CORE.ACTIVE_DATASET=projected_dataset
  EVIDENCE_PATCH_MODULE.source_evidence=previous.previous.previous.previous.previous.source_evidence
  yield
 finally:
  EVIDENCE_PATCH_MODULE.source_evidence=evidence; CORE.ACTIVE_DATASET=active; CORE.ranked_unreviewed=ranking
  for n,v in originals.items(): setattr(BASE,n,v)

def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=".v52-projection-",dir=OUT_DIR) as temp:
  ds=Path(temp)/"projection-v52.jsonl"; rp=Path(temp)/"projection-v52.report.json"
  build_projection(ds,rp,PRIOR_PROJECTION_CURATIONS)
  if len(read_jsonl(ds))!=543 or file_sha256(ds)!=PROJECTED_SELECTION_BASELINE["sha256"]: raise ValueError("v52 projection drift")
  with patched_base(ds): BASE.main()
 audits=read_jsonl(AUDIT)
 for row in audits:
  row.update(schema="context-merit-audit-v53",review_pass="first_context_merit_review_of_v52_projection_row",projection_lineage={"active_index":PROJECTED_ACTIVE_INDICES[row["fact_id"]],"baseline_rows":543,"baseline_sha256":PROJECTED_SELECTION_BASELINE["sha256"],"prior_context_merit_review":False})
  spec=next(s for s in SPECS if s["fact_id"]==row["fact_id"])
  if spec.get("support_type")=="manual_paraphrase": row["paraphrase_rationale"]=spec["paraphrase_rationale"]
 write_jsonl(AUDIT,audits)
 curations=read_jsonl(CURATION)
 for row in curations:
  spec=next(s for s in SPECS if s["fact_id"]==row["fact_id"])
  if spec.get("support_type")=="manual_paraphrase": row.update(support_type="manual_paraphrase",paraphrase_rationale=spec["paraphrase_rationale"])
 write_jsonl(CURATION,curations)
 report=json.loads(REPORT.read_text()); report["schema"]="context-merit-audit-report-v53"
 report["active_baseline"]={"dataset":{"path":str(ACTIVE_DATASET.relative_to(ROOT)),"rows":784,"sha256":file_sha256(ACTIVE_DATASET)},"report":{"path":str(ACTIVE_REPORT.relative_to(ROOT)),"sha256":file_sha256(ACTIVE_REPORT)},"curation":[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in ACTIVE_CURATIONS]}
 report["selection"].update(active_rows=543,eligible_unreviewed_rows=36,excluded_active_review_provenance=0,excluded_ledger_fact_ids=0,rows_selected=10,projected_baseline=PROJECTED_SELECTION_BASELINE,ranking={"candidate_rule":"the row survives the v52 projection, has no prior curation metadata, and its fact_id has no context-merit decision in v1 through v52","score":"short_question_points + 3*pronoun_count + bare_answer_points + named_person_trivia_points","tie_break":"risk_score descending, question tokens ascending, answer tokens ascending, fact_id ascending"})
 report["audit"]["rows"]=len(audits); report["audit"]["sha256"]=file_sha256(AUDIT); report["new_pending_curation"]["sha256"]=file_sha256(CURATION); report["new_pending_curation"]["edit_support_types"]={"extractive":0,"manual_paraphrase":1}
 report["frozen_prior_decision_artifacts"]=[{"path":str(p.relative_to(ROOT)),"sha256":file_sha256(p)} for p in prior_decision_artifacts()]
 REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
