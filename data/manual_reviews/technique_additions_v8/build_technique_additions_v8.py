#!/usr/bin/env python3
"""Build three manual technique QAs from new structural-practice pages."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V296=DATA/"manual_reviews/context_merit_audit_v296";sys.path[:0]=[str(ROOT),str(V296)]
import build_context_merit_audit_v296 as baseline_builder
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact,has_protocol_tokens,leakage_reason,normalize_text,parse_qa,stable_fact_id
OUT_DIR=Path(__file__).resolve().parent;OUTPUT=OUT_DIR/"pending_additions_technique_tranche_08_v1.jsonl";REPORT=OUT_DIR/"report_technique_tranche_08_v1.json";BASELINE_ROWS=513;BASELINE_SHA256="9038c7b518e2962083f6579dddd257b9e19f7e00b8119b9f5fd8bc0a1421ae4c";EXPECTED_OUTPUT_SHA256="211c2e4dfcfe22e4daa77fe893c16f93f82bd1c40c2fabcc1f314d763ef506db";RESOURCE_MANIFEST=ROOT/"sources/rope_resources_v1.json"
SOURCES={
 "narrow_cinch_bulk":{"path":DATA/"raw/rope_resources_v1/rope365__4b71bf74a2741be37598.json","url":"https://rope365.com/frog-cinches/","document_sha256":"01d185cbe42ecafaaba10ec01769a54036c943309b2fcb82ed6bbe8b53fab15e","markers":("- Avoid bulk: Make sure the rope goes into a straight line, good tension will help prevent rope from bunching up in the cinch. Avoid rope extension in the cinch.",)},
 "asymmetric_harness_methods":{"path":DATA/"raw/rope_resources_v1/rope365__b6a3b96502814b070a59.json","url":"https://rope365.com/chestasymmetry/","document_sha256":"f97e46428df4c106cdcc0832792441b2e749e384d706ae0e2ccc3ab8a8aa4e77","markers":("There are several approaches to bringing asymmetry into chest ties. An asymmetric starting point is interesting. For example, you can use a structure you like and offset the first knot. You can also start with a symmetric tie and distort it afterward. If you balance the forces in the tie, it can also evolve into a very solid structure. Let your creativity guide you!",)},
 "front_stem_attachment":{"path":DATA/"raw/rope_resources_v1/rope365__f85b7dd6c9c6a4a08bfd.json","url":"https://rope365.com/box-tie-front-v/","document_sha256":"f33691052bb0ee1da4ca61ca79a77754eb0ab1f1f4cb493e0710bcbee5bd940f","markers":("One of the most common ways to shape the front of the box tie is by going over the shoulder and catching the wraps in front to create a stem in the front. This add the possibility to attach to the front without worrying about sliding to the side. It is also a great way to add tension and shape the breasts or pectoral region.",)},
}
FACTS=(
 {"source_key":"narrow_cinch_bulk","topic":"narrow_cinch_bulk","question":"How can bulk be reduced when threading a cinch through a narrow gap?","answer":"Keep the rope in a straight line with enough tension to prevent bunching, and do not place a rope extension in the cinch.","paraphrase_rationale":"This keeps all three anti-bulk measures from the source without extending them into a broader safety claim."},
 {"source_key":"asymmetric_harness_methods","topic":"asymmetric_harness_methods","question":"What are two ways to introduce asymmetry into a chest harness?","answer":"Begin with an off-centre starting point, or build a symmetric structure and distort it afterward.","paraphrase_rationale":"This condenses the two construction approaches explicitly offered by the source."},
 {"source_key":"front_stem_attachment","topic":"front_stem_attachment","question":"What structural advantage does a front stem add to a box tie?","answer":"By catching the front wraps over the shoulder, it creates a front attachment point that is less likely to slide sideways.","paraphrase_rationale":"This preserves the source's stated attachment advantage while omitting optional shaping and tension effects."},
)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tsha(t):return hashlib.sha256(t.encode()).hexdigest()
def portable(p):return str(Path(p).resolve().relative_to(ROOT))
def read(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x.strip()]
def write(p,rows):Path(p).write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n"for r in rows))
def baseline(p,rep):
 baseline_builder.build_projection(p,rep);rows=read(p)
 if(len(rows),sha(p))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v296 baseline drift")
 return rows
def support(doc,markers):
 out=[]
 for m in markers:
  hits=[line for line in doc["text"].splitlines()if m in line]
  if len(hits)!=1:raise ValueError(f"evidence drift:{m}")
  out.append(hits[0])
 return"\n".join(out)
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True);docs={}
 for k,s in SOURCES.items():
  d=json.loads(s["path"].read_text())
  if(d["url"],d["document_sha256"],tsha(d["text"]))!=(s["url"],s["document_sha256"],s["document_sha256"]):raise ValueError(f"{k}:source drift")
  docs[k]=d
 with tempfile.TemporaryDirectory(prefix="technique-v8-",dir=OUT_DIR)as temp:base=baseline(Path(temp)/"v296.jsonl",Path(temp)/"v296.report.json")
 facts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train")for r in base];qs={normalize_text(r["question"])for r in base};pairs={(normalize_text(r["question"]),normalize_text(r["answer"]))for r in base};docids={r["document_sha256"]for r in base};urls={r["url"].rstrip("/").casefold()for r in base};rows=[]
 for f in FACTS:
  s=SOURCES[f["source_key"]];q,a=f["question"],f["answer"];pair=normalize_text(q),normalize_text(a)
  if not q.endswith("?")or"\n"in q or"\n"in a or has_protocol_tokens(q)or has_protocol_tokens(a)or parse_qa(f"Question: {q}\nAnswer: {a}")!=(q,a):raise ValueError("noncanonical")
  if pair in pairs or pair[0]in qs or leakage_reason(q,a,facts):raise ValueError("train collision")
  if s["document_sha256"]in docids or s["url"].rstrip("/").casefold()in urls:raise ValueError("non-novel source")
  ev=support(docs[f["source_key"]],s["markers"]);render=f"Question: {q}\nAnswer: {a}";rows.append({"answer":a,"claim_type":"instructional","document_sha256":s["document_sha256"],"evidence":ev,"evidence_sha256":tsha(ev),"evidence_url":s["url"],"fact_id":stable_fact_id(q,a),"kind":"qa_resource_manual_fact","paraphrase_rationale":f["paraphrase_rationale"],"quality_schema":"manual-resource-fact-v1","question":q,"resource_id":"rope365","reviewer":"codex-technique-additions-v8","source":"rope365","source_lineage":{"artifact":portable(OUTPUT),"raw_document":portable(s["path"]),"resource_manifest":portable(RESOURCE_MANIFEST)},"text":render,"topic":f["topic"],"url":s["url"],"verified_at":"2026-07-15"})
 if len(rows)!=3 or len({r["fact_id"]for r in rows})!=3:raise ValueError("identity drift")
 write(OUTPUT,rows);outsha=sha(OUTPUT)
 if EXPECTED_OUTPUT_SHA256!="PENDING"and outsha!=EXPECTED_OUTPUT_SHA256:raise ValueError("hash drift")
 strata=Counter(classify_stratum(r)for r in rows);report={"artifact":{"path":portable(OUTPUT),"rows":3,"sha256":outsha},"baseline":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"method":{"authoring":"manual full-source review and hand-authored Q&A","collision_scope":"v296 train-only projection; sealed collisions delegated to integration tooling","selection":"one bounded technique fact from each of three new documents and URLs"},"new_independent_inputs":{"document_sha256s":3,"expected_strata":dict(sorted(strata.items())),"urls":3},"reviewed_at":"2026-07-15","reviewer":"codex-technique-additions-v8","schema":"manual-technique-additions-report-v8","sources":{k:{"document_sha256":v["document_sha256"],"file_sha256":sha(v["path"]),"path":portable(v["path"]),"url":v["url"]}for k,v in sorted(SOURCES.items())},"status":"segregated_pending_integration"};REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
