#!/usr/bin/env python3
"""Build three manually reviewed safety/mechanics QAs from train-only source documents."""
from __future__ import annotations
import json,sys,tempfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V338=DATA/"manual_reviews/context_merit_audit_v338";sys.path[:0]=[str(ROOT),str(V338)]
import build_context_merit_audit_v338 as baseline_builder
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact,has_protocol_tokens,leakage_reason,normalize_text,parse_qa,stable_fact_id
OUT_DIR=Path(__file__).resolve().parent;OUTPUT=OUT_DIR/"pending_additions_safety_mechanics_tranche_22_v1.jsonl";REPORT=OUT_DIR/"report_safety_mechanics_tranche_22_v1.json";BASELINE_ROWS=526;BASELINE_SHA256="18ce02136c2c7993dd9dfe8dbde228632ad25a64d73be56dd4bc7d808d016509";EXPECTED_OUTPUT_SHA256="611b632876f5aafaef7b3c9a5f905c90b30de7172d2eae50e46d8fb46fff1914";RESOURCE_MANIFEST=ROOT/"sources/rope_resources_v1.json"
file_sha256,text_sha256,portable=baseline_builder.file_sha256,baseline_builder.text_sha256,baseline_builder.portable;read_jsonl,write_jsonl=baseline_builder.read_jsonl,baseline_builder.write_jsonl
SOURCES={
 "box_tie_front":{"path":DATA/"raw/rope_resources_v1/rope365__f85b7dd6c9c6a4a08bfd.json","url":"https://rope365.com/box-tie-front-v/","document_sha256":"f33691052bb0ee1da4ca61ca79a77754eb0ab1f1f4cb493e0710bcbee5bd940f"},
 "freestanding_hardpoints":{"path":DATA/"raw/rope_resources_v1/rope365__153b0f6d1f2932a77626.json","url":"https://rope365.com/diy-freestanding-hardpoints/","document_sha256":"5eeb3548377c02365b9333aee6184d1b7b84f562d695072ff017cc1bfb5898b9"},
 "hands_and_fingers":{"path":DATA/"raw/rope_resources_v1/rope365__e8eb7de51d99e927226e.json","url":"https://rope365.com/hands-and-fingers/","document_sha256":"ad74e9276dc15eea62a9f72f53f6c9a32c6a8c3eb08c9826c086b515fb8c88b1"},
}
FACTS=(
 {"source_key":"box_tie_front","topic":"third_rope_shoulder_nerve_precaution","question":"What shoulder-tension nerve warning and mitigation does Rope365 give for a third rope on a box tie?","answer":"Very high shoulder tension can cause clavicle-area numbness or weakness when raising the arm; Rope365 recommends keeping shoulder rope loose, or doubling it to spread the load if more tension is used.","markers":("Nerve issues with the third rope can happen with a very high level of tension on the top of the shoulders. Problems to the supraclavicular nerve can cause a numb patch on the clavicle region and the axillary be injured, causing weakness to raise the arm. These are easy to avoid by keeping the rope of the shoulder loose. If you wish to explore a higher level of tension, you can mitigate by doubling the ropes to spread the load on a wider surface. This kind of injury are more likely in suspension (especially dive positions) but can happen in floorwork if the rope is ultra tight. The length of the session and the cumulative aspect of nerve damage can also be contributing factors as these injuries are harder to monitor during the tie.",),"paraphrase_rationale":"This combines the source's observable symptom warning with its primary loose-rope precaution and qualified load-spreading mitigation."},
 {"source_key":"freestanding_hardpoints","topic":"frame_leg_splay_tipping_precaution","question":"Why does Rope365 call bridles or leg lashings important on freestanding suspension structures?","answer":"They help prevent dangerous leg splay and uneven loading; Rope365 also warns that dynamic movement can tip a structure with a high center of gravity, so tipping risk must be controlled.","markers":("Splay and uneven loading can be incredibly dangerous when considering structures with legs. Regardless of design, bridles, or some lashing should be applied to the legs to keep the design structural. This is why most manufacturers require their use. Please also be careful with the high center of gravity that can be encountered when doing dynamic movements. A lot of people see these “swingsets”, and can’t help themselves, but try to remember that actual swingsets are usually anchored to the floor. Most hardpoint “failure” and injury comes from tipping.",),"paraphrase_rationale":"This preserves the source's structural role for leg restraint and its separate high-center-of-gravity tipping warning without presenting a design as certified."},
 {"source_key":"hands_and_fingers","topic":"wrist_lock_range_precaution","question":"What injury precaution does Rope365 give for a wrist-lock position that pulls the hand backward?","answer":"Take time and care, and do not pull the wrist beyond its range limit, because forcing it too far can cause injury.","markers":("Day 180: Wrist Lock – Pulling the hand back into a joint lock is a very restrictive position. Explore how this principle can be used in a tie by tying the palm or finger and pulling the wrist backward. Take time and care as pulling the wrist against its limit can lead to an injury if we push too far.",),"paraphrase_rationale":"This extracts the source's range-of-motion safety boundary without operationalizing the restraint beyond the context needed to identify it."},
)
def build_baseline(path,report):
 baseline_builder.build_projection(path,report);rows=read_jsonl(path)
 if (len(rows),file_sha256(path))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v338 drift")
 return rows
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix="safety-mechanics-v22-",dir=OUT_DIR) as t:baseline=build_baseline(Path(t)/"v338.jsonl",Path(t)/"v338.report.json")
 facts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train") for r in baseline];qs={normalize_text(r["question"]) for r in baseline};pairs={(normalize_text(r["question"]),normalize_text(r["answer"])) for r in baseline};rows=[]
 for f in FACTS:
  s=SOURCES[f["source_key"]];d=json.loads(s["path"].read_text())
  if (d["url"],d["document_sha256"],text_sha256(d["text"]))!=(s["url"],s["document_sha256"],s["document_sha256"]):raise ValueError("source drift")
  q,a=f["question"],f["answer"];pair=normalize_text(q),normalize_text(a);rendered=f"Question: {q}\nAnswer: {a}"
  if not q.endswith("?") or "\n" in q+a or has_protocol_tokens(q) or has_protocol_tokens(a) or parse_qa(rendered)!=(q,a):raise ValueError("noncanonical")
  if pair in pairs or pair[0] in qs or leakage_reason(q,a,facts):raise ValueError("train collision")
  if not all(m in d["text"] for m in f["markers"]):raise ValueError("evidence marker drift")
  ev="\n".join(f["markers"]);rows.append({"answer":a,"claim_type":"instructional_safety","document_sha256":s["document_sha256"],"evidence":ev,"evidence_sha256":text_sha256(ev),"evidence_url":s["url"],"fact_id":stable_fact_id(q,a),"kind":"qa_resource_manual_fact","paraphrase_rationale":f["paraphrase_rationale"],"quality_schema":"manual-resource-fact-v1","question":q,"resource_id":"rope365","reviewer":"codex-safety-mechanics-additions-v22","source":"rope365","source_lineage":{"artifact":portable(OUTPUT),"raw_document":portable(s["path"]),"resource_manifest":portable(RESOURCE_MANIFEST)},"text":rendered,"topic":f["topic"],"url":s["url"],"verified_at":"2026-07-15"})
 if len(rows)!=3 or len({r["fact_id"] for r in rows})!=3:raise ValueError("identity drift")
 write_jsonl(OUTPUT,rows);sha=file_sha256(OUTPUT)
 if EXPECTED_OUTPUT_SHA256!="PENDING" and sha!=EXPECTED_OUTPUT_SHA256:raise ValueError("artifact drift")
 strata=Counter(classify_stratum(r) for r in rows);REPORT.write_text(json.dumps({"artifact":{"path":portable(OUTPUT),"rows":3,"sha256":sha},"baseline":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"method":{"authoring":"manual train-only full-source review and hand-authored Q&A","collision_scope":"v338 train-only projection; sealed collisions delegated to integration tooling","selection":"three distinct safety facts from three manually reviewed Rope365 source documents"},"new_independent_inputs":{"document_sha256s":0,"expected_strata":dict(sorted(strata.items())),"urls":0},"reviewed_at":"2026-07-15","reviewer":"codex-safety-mechanics-additions-v22","schema":"manual-safety-mechanics-additions-report-v22","status":"segregated_pending_integration"},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
