#!/usr/bin/env python3
"""Build three manually reviewed practice and balance-safety QAs."""
from __future__ import annotations
import json,sys,tempfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V310=DATA/"manual_reviews/context_merit_audit_v310";V21=DATA/"manual_reviews/tethered_policy_additions_v21";sys.path[:0]=[str(ROOT),str(V310),str(V21)]
import build_context_merit_audit_v310 as baseline_builder
import build_tethered_policy_additions_v21 as prior
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact,has_protocol_tokens,leakage_reason,normalize_text,parse_qa,stable_fact_id
OUT_DIR=Path(__file__).resolve().parent;OUTPUT=OUT_DIR/"pending_additions_practice_balance_tranche_22_v1.jsonl";REPORT=OUT_DIR/"report_practice_balance_tranche_22_v1.json";BASELINE_ROWS=554;BASELINE_SHA256="3e1de39bb4f5bb6e3de7a8b03f04e40bfdd80066fb87d6ff8c3b5cf1870fc1e7";EXPECTED_OUTPUT_SHA256="360f2cae907f02ab328c19e1a96275ac24ee98d530b49aff41f45201ee9fba83";RESOURCE_MANIFEST=ROOT/"sources/rope_resources_v1.json"
file_sha256,text_sha256,portable=prior.file_sha256,prior.text_sha256,prior.portable;read_jsonl,write_jsonl,select_evidence=prior.read_jsonl,prior.write_jsonl,prior.select_evidence
SOURCES={
 "nondominant":{"path":DATA/"raw/rope_resources_v1/rope365__fdb2f550a18df68fe068.json","url":"https://rope365.com/multitasking/","document_sha256":"77fcac1d1d51de8954af4188926f10752a6d087275d2456e10debe9e8ee51cfb","markers":("| Day 233: Switching Sides – Practice your favourite single column tie but switching to your non-dominant hand leading. Pick a pattern (box tie, frog tie) and measure how much time it takes to tie it. Now try again with only your dominant hand. Now try again with your non-dominant hand. Now try tying with both hands again. Has your score improved? | ",)},
 "readjustment":{"path":DATA/"raw/rope_resources_v1/rope365__fc84f60465325619f5ad.json","url":"https://rope365.com/frog-chaos/","document_sha256":"b0e46951e4374fff0c257a2f76ce6d04693651fb07dea69337b4b2f6fbb932b8","markers":("- When you get out of the rope, take a moment to let the body readjust. Do some small physical activity that feels comfortable and good in that moment. Maybe in this case you could for example start by gently unfolding your knee and checking in with the joint before getting up slowly and pacing around a bit.",)},
 "balance":{"path":DATA/"raw/rope_resources_v1/rope365__e7b05d49c5f7d5c65292.json","url":"https://rope365.com/work-out/","document_sha256":"9d867d4814995dfcbbc3804c26c83cc99d77e427566f20d0ad7e263c1b08b99f","markers":("| Day 271: Equilibrium – Can you challenge the body to maintain balance while doing rope? Try balancing on one foot while tying or being tied. Look into yoga balance poses and acro yoga techniques. Make sure someone is close to catch the person tied up should they lose their balance. ",)},
}
FACTS=(
 {"source_key":"nondominant","topic":"nondominant_hand_practice","question":"How does Rope365 suggest practicing a familiar tie with the non-dominant hand?","answer":"Time the normal version, repeat it with each hand leading separately, then return to both hands and compare the result.","paraphrase_rationale":"This preserves the source's staged comparison exercise without reproducing its riskier mouth or neck-start prompts."},
 {"source_key":"readjustment","topic":"folded_leg_readjustment","question":"How does Rope365 suggest coming out of a repeatedly practiced folded-leg tie?","answer":"Gently unfold the knee, check in with the joint, get up slowly, and use a small amount of comfortable movement while the body readjusts.","paraphrase_rationale":"This retains the source's gradual readjustment sequence without presenting it as treatment for an injury."},
 {"source_key":"balance","topic":"balance_catcher","question":"What safety support does Rope365 recommend when a tied person practices a balance challenge?","answer":"Keep someone close enough to catch the tied person if they lose their balance.","paraphrase_rationale":"This preserves the explicit nearby-catcher precaution and does not endorse the page's combat or loaded-strength prompts."},
)
def build_baseline(path,report):
 baseline_builder.build_projection(path,report);rows=read_jsonl(path)
 if (len(rows),file_sha256(path))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v310 drift")
 return rows
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True);docs={}
 for k,s in SOURCES.items():
  d=json.loads(s["path"].read_text())
  if (d["url"],d["document_sha256"],text_sha256(d["text"]))!=(s["url"],s["document_sha256"],s["document_sha256"]):raise ValueError(f"{k}: source drift")
  docs[k]=d
 with tempfile.TemporaryDirectory(prefix="practice-balance-v22-",dir=OUT_DIR) as t:baseline=build_baseline(Path(t)/"v310.jsonl",Path(t)/"v310.report.json")
 facts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train") for r in baseline];qs={normalize_text(r["question"]) for r in baseline};pairs={(normalize_text(r["question"]),normalize_text(r["answer"])) for r in baseline};docids={r["document_sha256"] for r in baseline};urls={r["url"].rstrip("/").casefold() for r in baseline};rows=[]
 for f in FACTS:
  s=SOURCES[f["source_key"]];q,a=f["question"],f["answer"];pair=normalize_text(q),normalize_text(a);rendered=f"Question: {q}\nAnswer: {a}"
  if not q.endswith("?") or "\n" in q+a or has_protocol_tokens(q) or has_protocol_tokens(a) or parse_qa(rendered)!=(q,a):raise ValueError("noncanonical")
  if pair in pairs or pair[0] in qs or leakage_reason(q,a,facts):raise ValueError("train collision")
  if s["document_sha256"] in docids or s["url"].rstrip("/").casefold() in urls:raise ValueError("source not novel")
  ev=select_evidence(docs[f["source_key"]],s["markers"]);rows.append({"answer":a,"claim_type":"instructional","document_sha256":s["document_sha256"],"evidence":ev,"evidence_sha256":text_sha256(ev),"evidence_url":s["url"],"fact_id":stable_fact_id(q,a),"kind":"qa_resource_manual_fact","paraphrase_rationale":f["paraphrase_rationale"],"quality_schema":"manual-resource-fact-v1","question":q,"resource_id":"rope365","reviewer":"codex-practice-balance-additions-v22","source":"rope365","source_lineage":{"artifact":portable(OUTPUT),"raw_document":portable(s["path"]),"resource_manifest":portable(RESOURCE_MANIFEST)},"text":rendered,"topic":f["topic"],"url":s["url"],"verified_at":"2026-07-15"})
 if len(rows)!=3 or len({r["fact_id"] for r in rows})!=3:raise ValueError("identity drift")
 write_jsonl(OUTPUT,rows);sha=file_sha256(OUTPUT)
 if EXPECTED_OUTPUT_SHA256!="PENDING" and sha!=EXPECTED_OUTPUT_SHA256:raise ValueError("artifact drift")
 strata=Counter(classify_stratum(r) for r in rows);REPORT.write_text(json.dumps({"artifact":{"path":portable(OUTPUT),"rows":3,"sha256":sha},"baseline":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"excluded_source":[{"url":"https://rope365.com/multitasking/","decision":"partial_use","reason":"Only the non-dominant-hand exercise was retained; mouth use and neck-start prompts were excluded."},{"url":"https://rope365.com/frog-chaos/","decision":"partial_use","reason":"Only gradual post-tie readjustment was retained; intentional pinch and excessive-rope prompts were excluded."},{"url":"https://rope365.com/work-out/","decision":"partial_use","reason":"Only the nearby-catcher precaution was retained; combat, loaded-strength, and deep-stretch prompts were excluded."}],"method":{"authoring":"manual full-source review and hand-authored Q&A","collision_scope":"v310 train-only projection; sealed collisions delegated to integration tooling","selection":"one bounded practice or safety fact from each of three distinct new documents"},"new_independent_inputs":{"document_sha256s":3,"expected_strata":dict(sorted(strata.items())),"urls":3},"reviewed_at":"2026-07-15","reviewer":"codex-practice-balance-additions-v22","schema":"manual-practice-balance-additions-report-v22","status":"segregated_pending_integration"},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
