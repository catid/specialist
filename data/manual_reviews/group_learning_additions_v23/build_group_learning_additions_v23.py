#!/usr/bin/env python3
"""Build three manually reviewed group-consent and learning-quality QAs."""
from __future__ import annotations
import json,sys,tempfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V318=DATA/"manual_reviews/context_merit_audit_v318";V22=DATA/"manual_reviews/practice_balance_additions_v22";sys.path[:0]=[str(ROOT),str(V318),str(V22)]
import build_context_merit_audit_v318 as baseline_builder
import build_practice_balance_additions_v22 as prior
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact,has_protocol_tokens,leakage_reason,normalize_text,parse_qa,stable_fact_id
OUT_DIR=Path(__file__).resolve().parent;OUTPUT=OUT_DIR/"pending_additions_group_learning_tranche_23_v1.jsonl";REPORT=OUT_DIR/"report_group_learning_tranche_23_v1.json";BASELINE_ROWS=545;BASELINE_SHA256="fadaa3d594515494745be850541af9273858c79b1ae7c3c21be123213f88a042";EXPECTED_OUTPUT_SHA256="98d04e92b9435a415f34b4d19ca2ac0ad4b4857eaa5f8bf6073dfc3c2f676b72";RESOURCE_MANIFEST=ROOT/"sources/rope_resources_v1.json"
file_sha256,text_sha256,portable=prior.file_sha256,prior.text_sha256,prior.portable;read_jsonl,write_jsonl,select_evidence=prior.read_jsonl,prior.write_jsonl,prior.select_evidence
SOURCES={
 "group":{"path":DATA/"raw/rope_resources_v1/rope365__e68382315faf70515544.json","url":"https://rope365.com/games/","document_sha256":"e173f571b4125be3661c391b9ff7713c05f2bd8adc4b124209f7946b7fa85f3c","markers":("When more people are involved, it is important to make sure all of them are aware of the risks involved, and comfortable with the people they will be tying with. Create a space for all to be able to state their boundaries and adjust when necessary.",)},
 "school":{"path":DATA/"raw/esinem_7c6ce8a699d42f64.json","url":"https://www.esinem.com/choosing-online-shibari-schools/","document_sha256":"204bd3da49603804c40b7f34f44ebfd2e7345a861f073c1adb2a4b3e120daa0f","markers":("The first thing to decide is whether the school reflects your vision of shibari, e.g. Two Knotty Boys style or more Japanese, both in terms of aesthetics and philosophy. Only then can you narrow your search. Then you need to look at the credentials of the teacher. Who taught them? Can they teach? To all the questions proof is easily available. Most teachers should be happy to produce their CV. It shouldn’t be hard to find students to ask if they can teach, what they teach and the depth of their teaching.",)},
 "tutorial":{"path":DATA/"raw/esinem_833896e319c39a27.json","url":"https://www.esinem.com/not-all-shibari-tutorials-are-equal/","document_sha256":"713506830872db86570cefbce41499a7b410aa4a44b8b7667b4324867e07e4e4","markers":("- Safety: Obviously, a tie should be safe. This is not always easy for the unskilled to assess. However, as a bare minimun, you should look at the basics of bondage safety, e.g. is there slack if it’s needed, will it tighten accidentally, does it create a strangulation or other serious risk?",)},
}
FACTS=(
 {"source_key":"group","source":"rope365","resource_id":"rope365","topic":"group_rope_consent_preparation","question":"What consent preparation does Rope365 recommend before a group rope game?","answer":"Make sure everyone understands the risks and is comfortable with their tying partners, and provide space to state boundaries and adjust the activity.","paraphrase_rationale":"This retains the page's multi-person risk, partner-comfort, and boundary process while excluding its racing, mouth-held-object, tugging, hook, and other hazardous game prompts."},
 {"source_key":"school","source":"esinem","resource_id":"esinem","topic":"online_school_vetting","question":"What should a beginner verify when choosing an online shibari school?","answer":"First decide whether its aesthetic and philosophy match their goals, then check the teacher's training and teaching ability through credentials and feedback from students.","paraphrase_rationale":"This preserves the source's school-fit and teacher-verification criteria while excluding promotional testimonials and sales claims."},
 {"source_key":"tutorial","source":"esinem","resource_id":"esinem","topic":"tutorial_minimum_safety_checks","question":"What minimum safety checks does ESINEM suggest when evaluating a shibari tutorial?","answer":"Check whether the tie has needed slack, can tighten accidentally, or creates strangulation or another serious risk.","paraphrase_rationale":"This isolates the source's concrete minimum checks without treating the author's promotional claims as instruction."},
)
def build_baseline(path,report):
 baseline_builder.build_projection(path,report);rows=read_jsonl(path)
 if (len(rows),file_sha256(path))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v318 drift")
 return rows
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True);docs={}
 for k,s in SOURCES.items():
  d=json.loads(s["path"].read_text())
  if (d["url"],text_sha256(d["text"]))!=(s["url"],s["document_sha256"]):raise ValueError(f"{k}: source drift")
  if d.get("document_sha256") not in (None,s["document_sha256"]):raise ValueError(f"{k}: document identity drift")
  docs[k]=d
 with tempfile.TemporaryDirectory(prefix="group-learning-v23-",dir=OUT_DIR) as t:baseline=build_baseline(Path(t)/"v318.jsonl",Path(t)/"v318.report.json")
 facts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train") for r in baseline];qs={normalize_text(r["question"]) for r in baseline};pairs={(normalize_text(r["question"]),normalize_text(r["answer"])) for r in baseline};docids={r["document_sha256"] for r in baseline};urls={r["url"].rstrip("/").casefold() for r in baseline};rows=[]
 for f in FACTS:
  s=SOURCES[f["source_key"]];q,a=f["question"],f["answer"];pair=normalize_text(q),normalize_text(a);rendered=f"Question: {q}\nAnswer: {a}"
  if not q.endswith("?") or "\n" in q+a or has_protocol_tokens(q) or has_protocol_tokens(a) or parse_qa(rendered)!=(q,a):raise ValueError("noncanonical")
  if pair in pairs or pair[0] in qs or leakage_reason(q,a,facts):raise ValueError("train collision")
  if s["document_sha256"] in docids or s["url"].rstrip("/").casefold() in urls:raise ValueError("source not novel")
  ev=select_evidence(docs[f["source_key"]],s["markers"]);rows.append({"answer":a,"claim_type":"instructional","document_sha256":s["document_sha256"],"evidence":ev,"evidence_sha256":text_sha256(ev),"evidence_url":s["url"],"fact_id":stable_fact_id(q,a),"kind":"qa_resource_manual_fact","paraphrase_rationale":f["paraphrase_rationale"],"quality_schema":"manual-resource-fact-v1","question":q,"resource_id":f["resource_id"],"reviewer":"codex-group-learning-additions-v23","source":f["source"],"source_lineage":{"artifact":portable(OUTPUT),"raw_document":portable(s["path"]),"resource_manifest":portable(RESOURCE_MANIFEST)},"text":rendered,"topic":f["topic"],"url":s["url"],"verified_at":"2026-07-15"})
 if len(rows)!=3 or len({r["fact_id"] for r in rows})!=3:raise ValueError("identity drift")
 write_jsonl(OUTPUT,rows);sha=file_sha256(OUTPUT)
 if EXPECTED_OUTPUT_SHA256!="PENDING" and sha!=EXPECTED_OUTPUT_SHA256:raise ValueError("artifact drift")
 strata=Counter(classify_stratum(r) for r in rows);REPORT.write_text(json.dumps({"artifact":{"path":portable(OUTPUT),"rows":3,"sha256":sha},"baseline":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"excluded_source":[{"url":"https://rope365.com/games/","decision":"partial_use","reason":"Only group risk, comfort, and boundary preparation was retained; racing, tugging, mouth-held-object, hook, and other hazardous games were excluded."},{"url":"https://www.esinem.com/choosing-online-shibari-schools/","decision":"partial_use","reason":"Only school-fit and teacher-verification criteria were retained; testimonials and promotional claims were excluded."},{"url":"https://www.esinem.com/not-all-shibari-tutorials-are-equal/","decision":"partial_use","reason":"Only the concrete minimum safety checks were retained; promotional comparisons and unsupported superiority claims were excluded."}],"method":{"authoring":"manual full-source review and hand-authored Q&A","collision_scope":"v318 train-only projection; sealed collisions delegated to integration tooling","selection":"one bounded consent or learning-quality fact from each of three distinct new documents"},"new_independent_inputs":{"document_sha256s":3,"expected_strata":dict(sorted(strata.items())),"urls":3},"reviewed_at":"2026-07-15","reviewer":"codex-group-learning-additions-v23","schema":"manual-group-learning-additions-report-v23","status":"segregated_pending_integration"},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
