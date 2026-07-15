#!/usr/bin/env python3
"""Build three manual equipment/resource QAs from distinct new documents."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V295=DATA/"manual_reviews/context_merit_audit_v295";sys.path[:0]=[str(ROOT),str(V295)]
import build_context_merit_audit_v295 as baseline_builder
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact,has_protocol_tokens,leakage_reason,normalize_text,parse_qa,stable_fact_id
OUT_DIR=Path(__file__).resolve().parent;OUTPUT=OUT_DIR/"pending_additions_equipment_tranche_07_v1.jsonl";REPORT=OUT_DIR/"report_equipment_tranche_07_v1.json";BASELINE_ROWS=510;BASELINE_SHA256="53dfec56416923431838a74914ba3900553aae8cfc23c95d42b8169792c61b1f";EXPECTED_OUTPUT_SHA256="5becbbe974fc068c74789b7ddad05f572ffe8c9cad1059a7946604fcf3bae726";RESOURCE_MANIFEST=ROOT/"sources/rope_resources_v1.json"
SOURCES={
 "texas_rope_vendor_materials":{"path":DATA/"raw/rope_resources_v1/austin_rope_slingers__76c874894072bdc4b3f9.json","url":"https://www.austinropeslingers.com/rope-101-resources/","document_sha256":"46475111b24234a74718317ff767509dc5b1010525e9cf507dbb267e0efc6928","markers":("- KnottieKittie – Hemp and Nylon","- DeGiotto – Jute, Hemp, Silk, and Synthetic","- RavenClawRope – Jute")},
 "a_frame_support_documents":{"path":DATA/"raw/rope_resources_v1/xpole_a_frame__4503d099f45260a68e60.json","url":"https://xpoleus.com/support/a-frame/","document_sha256":"57e7aa6c2b42bc5943a6296e6ba267c0247a7004964616b034d1671930af3aef","markers":("A-FRAME Set Up","A-Frame Load Test Certificate","A-FRAME Manual - 2.0","A-FRAME Footprint and build dimensions","How to use a strop, spanset or sling","Aerial Hoops Instruction Sheet","A-Frame Figure 8 Load Test Certificate","Extended Top Bar Load Test Certificate")},
 "custom_hardware_design_input":{"path":DATA/"raw/rope_resources_v1/subspace_designs__10d8ce5448a30149b028.json","url":"https://www.subspacedesigns.shop/subspacecustomorders","document_sha256":"4190a89de76eb741ac09ebd83abbab5d3c347de117e5a7beaac593ac256c3eb4","markers":("2. Get ready to share your vision! Bring a sketch of your idea to our meeting, and we’ll dive deep into your concept and refine it for machining. A simple napkin sketch will do (bonus points for dimensions, and super bonus points if you have a CAD file in .step format)!",)},
}
FACTS=(
 {"source_key":"texas_rope_vendor_materials","topic":"texas_rope_vendor_materials","question":"Which materials does Austin Rope Slingers list for KnottieKittie, DeGiotto, and RavenClawRope?","answer":"KnottieKittie sells hemp and nylon; DeGiotto sells jute, hemp, silk, and synthetic rope; RavenClawRope sells jute.","paraphrase_rationale":"This preserves the material-to-vendor mapping from the local-source section without copying prices, inventory, or an endorsement."},
 {"source_key":"a_frame_support_documents","topic":"a_frame_support_documents","question":"What official A-frame equipment documents does X-POLE's support page provide?","answer":"Setup guidance, load-test certificates, the version 2.0 manual, footprint and build dimensions, sling-use guidance, and an aerial-hoop instruction sheet.","paraphrase_rationale":"This summarizes the official support-document categories while avoiding conflicting numerical load claims and volatile product details."},
 {"source_key":"custom_hardware_design_input","topic":"custom_hardware_design_input","question":"What design material should a customer bring to a custom hardware consultation?","answer":"A sketch of the idea, ideally with dimensions or a CAD file in STEP format.","paraphrase_rationale":"This condenses the vendor's requested design inputs without copying lead-time, deposit, or sales language."},
)
def file_sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def text_sha256(t):return hashlib.sha256(t.encode()).hexdigest()
def portable(p):return str(Path(p).resolve().relative_to(ROOT))
def read_jsonl(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x.strip()]
def write_jsonl(p,rows):Path(p).write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n"for r in rows))
def build_baseline(p,rep):
 baseline_builder.build_projection(p,rep);rows=read_jsonl(p)
 if(len(rows),file_sha256(p))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v295 baseline drift")
 return rows
def evidence(doc,markers):
 selected=[]
 for marker in markers:
  matches=[line for line in doc["text"].splitlines()if marker in line]
  if not matches:raise ValueError(f"evidence drift: {marker}")
  if matches[0]not in selected:selected.append(matches[0])
 return"\n".join(selected)
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True);documents={}
 for k,s in SOURCES.items():
  d=json.loads(s["path"].read_text())
  if(d["url"],d["document_sha256"],text_sha256(d["text"]))!=(s["url"],s["document_sha256"],s["document_sha256"]):raise ValueError(f"{k}: source drift")
  documents[k]=d
 with tempfile.TemporaryDirectory(prefix="equipment-v7-",dir=OUT_DIR)as temp:baseline=build_baseline(Path(temp)/"v295.jsonl",Path(temp)/"v295.report.json")
 basefacts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train")for r in baseline];qs={normalize_text(r["question"])for r in baseline};pairs={(normalize_text(r["question"]),normalize_text(r["answer"]))for r in baseline};docs={r["document_sha256"]for r in baseline};urls={r["url"].rstrip("/").casefold()for r in baseline};rows=[]
 for f in FACTS:
  s=SOURCES[f["source_key"]];q,a=f["question"],f["answer"];pair=normalize_text(q),normalize_text(a)
  if not q.endswith("?")or"\n"in q or"\n"in a or has_protocol_tokens(q)or has_protocol_tokens(a)or parse_qa(f"Question: {q}\nAnswer: {a}")!=(q,a):raise ValueError("noncanonical")
  if pair in pairs or pair[0]in qs or leakage_reason(q,a,basefacts):raise ValueError("train collision")
  if s["document_sha256"]in docs or s["url"].rstrip("/").casefold()in urls:raise ValueError("non-novel source")
  support=evidence(documents[f["source_key"]],s["markers"]);render=f"Question: {q}\nAnswer: {a}";rows.append({"answer":a,"claim_type":"equipment_navigation","document_sha256":s["document_sha256"],"evidence":support,"evidence_sha256":text_sha256(support),"evidence_url":s["url"],"fact_id":stable_fact_id(q,a),"kind":"qa_resource_manual_fact","paraphrase_rationale":f["paraphrase_rationale"],"quality_schema":"manual-resource-fact-v1","question":q,"resource_id":documents[f["source_key"]]["resource_id"],"reviewer":"codex-equipment-additions-v7","source":documents[f["source_key"]]["source"],"source_lineage":{"artifact":portable(OUTPUT),"raw_document":portable(s["path"]),"resource_manifest":portable(RESOURCE_MANIFEST)},"text":render,"topic":f["topic"],"url":s["url"],"verified_at":"2026-07-15"})
 if len(rows)!=3 or len({r["fact_id"]for r in rows})!=3:raise ValueError("identity drift")
 write_jsonl(OUTPUT,rows);sha=file_sha256(OUTPUT)
 if EXPECTED_OUTPUT_SHA256!="PENDING"and sha!=EXPECTED_OUTPUT_SHA256:raise ValueError("hash drift")
 strata=Counter(classify_stratum(r)for r in rows);report={"artifact":{"path":portable(OUTPUT),"rows":3,"sha256":sha},"baseline":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"method":{"authoring":"manual full-source review and hand-authored Q&A","collision_scope":"v295 train-only projection; sealed collisions delegated to integration tooling","selection":"one durable equipment or vendor-navigation fact from each of three distinct, previously unrepresented documents and URLs"},"new_independent_inputs":{"document_sha256s":3,"expected_strata":dict(sorted(strata.items())),"urls":3},"reviewed_at":"2026-07-15","reviewer":"codex-equipment-additions-v7","schema":"manual-equipment-additions-report-v7","sources":{k:{"document_sha256":v["document_sha256"],"file_sha256":file_sha256(v["path"]),"path":portable(v["path"]),"url":v["url"]}for k,v in sorted(SOURCES.items())},"status":"segregated_pending_integration"};REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
