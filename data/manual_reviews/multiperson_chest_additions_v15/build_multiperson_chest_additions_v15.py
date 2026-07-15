#!/usr/bin/env python3
"""Build two manually reviewed planning/adaptation QAs from distinct sources."""
from __future__ import annotations
import json,sys,tempfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; DATA=ROOT/"data"
V303=DATA/"manual_reviews/context_merit_audit_v303"; V14=DATA/"manual_reviews/resource_access_additions_v14"
sys.path[:0]=[str(ROOT),str(V303),str(V14)]
import build_context_merit_audit_v303 as baseline_builder
import build_resource_access_additions_v14 as prior
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact,has_protocol_tokens,leakage_reason,normalize_text,parse_qa,stable_fact_id
OUT_DIR=Path(__file__).resolve().parent; OUTPUT=OUT_DIR/"pending_additions_multiperson_chest_tranche_15_v1.jsonl"; REPORT=OUT_DIR/"report_multiperson_chest_tranche_15_v1.json"
BASELINE_ROWS=534; BASELINE_SHA256="8ace985d0a66db9638cd303a3fb47a271a367365094b3b6b4ec401f1d7369401"; EXPECTED_OUTPUT_SHA256="60b1ce76d1c2e9f522611ce848613f5c185caa57595036b848f60097f660ae21"
RESOURCE_MANIFEST=ROOT/"sources/rope_resources_v1.json"
file_sha256,text_sha256,portable=prior.file_sha256,prior.text_sha256,prior.portable
read_jsonl,write_jsonl,select_evidence=prior.read_jsonl,prior.write_jsonl,prior.select_evidence
SOURCES={
 "multiperson_planning":{"path":DATA/"raw/rope_resources_v1/rope365__b226e866cb8fed895e36.json","url":"https://rope365.com/tied-together/","document_sha256":"cd6623d2591eebd747d7c0f268bf1d4639584e1957fb12a57f10ea14dd00ee47","markers":("Creating a tie that involves more than one person expands the possibilities of position in rope. The different dynamics between the participants may lead to different kinds of play. This also means new risks to mitigate, preparing exit plans if more than one person experiences a problem during the tie, and making sure to cover what is acceptable for each participant, not only with the person tying, but also between each of the individuals in the tie.",)},
 "chest_adaptation":{"path":DATA/"raw/rope_resources_v1/rope365__3140d87c61daffdbce52.json","url":"https://rope365.com/chest/","document_sha256":"10dadd499953b44c64a79389ce4415b8b3ed4b46977ae8753dc9664185df443e","markers":("The larger surface of the torso makes it an interesting location to explore different patterns and structures. Each person is different and the same idea will result in something unique. Chest harnesses are also known as shinju 真珠 (pearl) and munenawa 胸縄 (chest rope).","The goal of this week is to experiment with different structures, play with them and learn to create something unique by adapting it to the person being tied.")},
}
FACTS=(
 {"source_key":"multiperson_planning","topic":"multiperson_exit_consent_planning","question":"What exit and consent planning does Rope365 recommend before a tie involving multiple participants?","answer":"Agree what is acceptable between every participant and prepare exit plans in case more than one person experiences a problem.","paraphrase_rationale":"This preserves the source's two multi-person planning duties without reproducing its optional play prompts."},
 {"source_key":"chest_adaptation","topic":"person_adapted_chest_structure","question":"Why does Rope365 say a chest-harness structure should be adapted to the person being tied?","answer":"Each person is different, so the same structural idea will produce a unique result for that person.","paraphrase_rationale":"This retains the source's adaptation rationale and omits aliases and day-number exercises."},
)
def build_baseline(path,report):
 baseline_builder.build_projection(path,report); rows=read_jsonl(path)
 if (len(rows),file_sha256(path))!=(BASELINE_ROWS,BASELINE_SHA256): raise ValueError("v303 baseline drift")
 return rows
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True); docs={}
 for k,s in SOURCES.items():
  d=json.loads(s["path"].read_text())
  if (d["url"],d["document_sha256"],text_sha256(d["text"]))!=(s["url"],s["document_sha256"],s["document_sha256"]): raise ValueError(f"{k}: source drift")
  docs[k]=d
 with tempfile.TemporaryDirectory(prefix="multiperson-chest-v15-",dir=OUT_DIR) as t: baseline=build_baseline(Path(t)/"v303.jsonl",Path(t)/"v303.report.json")
 facts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train") for r in baseline]; qs={normalize_text(r["question"]) for r in baseline}; pairs={(normalize_text(r["question"]),normalize_text(r["answer"])) for r in baseline}; docids={r["document_sha256"] for r in baseline}; urls={r["url"].rstrip("/").casefold() for r in baseline}; rows=[]
 for f in FACTS:
  s=SOURCES[f["source_key"]]; q,a=f["question"],f["answer"]; pair=normalize_text(q),normalize_text(a); rendered=f"Question: {q}\nAnswer: {a}"
  if not q.endswith("?") or "\n" in q+a or has_protocol_tokens(q) or has_protocol_tokens(a) or parse_qa(rendered)!=(q,a): raise ValueError("noncanonical")
  if pair in pairs or pair[0] in qs or leakage_reason(q,a,facts): raise ValueError("train collision")
  if s["document_sha256"] in docids or s["url"].rstrip("/").casefold() in urls: raise ValueError("source not novel")
  ev=select_evidence(docs[f["source_key"]],s["markers"])
  rows.append({"answer":a,"claim_type":"instructional","document_sha256":s["document_sha256"],"evidence":ev,"evidence_sha256":text_sha256(ev),"evidence_url":s["url"],"fact_id":stable_fact_id(q,a),"kind":"qa_resource_manual_fact","paraphrase_rationale":f["paraphrase_rationale"],"quality_schema":"manual-resource-fact-v1","question":q,"resource_id":"rope365","reviewer":"codex-multiperson-chest-additions-v15","source":"rope365","source_lineage":{"artifact":portable(OUTPUT),"raw_document":portable(s["path"]),"resource_manifest":portable(RESOURCE_MANIFEST)},"text":rendered,"topic":f["topic"],"url":s["url"],"verified_at":"2026-07-15"})
 if len(rows)!=2 or len({r["fact_id"] for r in rows})!=2: raise ValueError("identity drift")
 write_jsonl(OUTPUT,rows); sha=file_sha256(OUTPUT)
 if EXPECTED_OUTPUT_SHA256!="PENDING" and sha!=EXPECTED_OUTPUT_SHA256: raise ValueError("artifact drift")
 strata=Counter(classify_stratum(r) for r in rows)
 REPORT.write_text(json.dumps({"artifact":{"path":portable(OUTPUT),"rows":2,"sha256":sha},"baseline":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"excluded_third_candidate":{"decision":"reject","reason":"The fully reviewed head page's technique prompts are unsafe to operationalize, while its mouth-hygiene line duplicates a stronger existing advance-consent checklist."},"method":{"authoring":"manual full-source review and hand-authored Q&A","collision_scope":"v303 train-only projection; sealed collisions delegated to integration tooling","selection":"two bounded nonredundant facts from distinct new documents; no filler third row"},"new_independent_inputs":{"document_sha256s":2,"expected_strata":dict(sorted(strata.items())),"urls":2},"reviewed_at":"2026-07-15","reviewer":"codex-multiperson-chest-additions-v15","schema":"manual-multiperson-chest-additions-report-v15","status":"segregated_pending_integration"},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
