#!/usr/bin/env python3
"""Build three manually reviewed directory/reference QAs from new pages."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V294=DATA/"manual_reviews/context_merit_audit_v294";sys.path[:0]=[str(ROOT),str(V294)]
import build_context_merit_audit_v294 as baseline_builder
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact,has_protocol_tokens,leakage_reason,normalize_text,parse_qa,stable_fact_id
OUT_DIR=Path(__file__).resolve().parent;OUTPUT=OUT_DIR/"pending_additions_resource_tranche_06_v1.jsonl";REPORT=OUT_DIR/"report_resource_tranche_06_v1.json";BASELINE_ROWS=507;BASELINE_SHA256="b9768507b55aa6650f86b1cc4cf849d268f4d0c619f53384b3b74eca2af61c5b";EXPECTED_OUTPUT_SHA256="3c552115a93015bdf90cfb69c98449a0e1060f2d8e5d42878ab3ff0dbd8f1589";RESOURCE_MANIFEST=ROOT/"sources/rope_resources_v1.json"
SOURCES={
 "book_selection":{"path":DATA/"raw/rope_resources_v1/rope365__b2d86cf0a71999889698.json","url":"https://rope365.com/books/","document_sha256":"cddbb4a99e637b11213b2b853401472bbeb5d022834fde125c4d4a3ba2637275","markers":("| Better Bondage for Every Body by Evie Vane If you are looking to acquire core knowledge and discover the diversity of the rope community, this is what you need. It showcases topics like nerve anatomy, neuroscience, warming up the body and pain processing. This is not a how-to tie book but includes a few ties to adapt to different flexibility range. | |",)},
 "community_search":{"path":DATA/"raw/rope_resources_v1/rope365__d386503129844bc811bd.json","url":"https://rope365.com/communities/","document_sha256":"8e188b368a79bc4c6b73171773c1bf6fb829aa2a5327b994755c40a7ac1efbd3","markers":("- Join Fetlife and search for groups in your area (free registration)","- Check Shibari Map for rope jam and classes.","- Check the list below for resources by region.")},
 "artist_reference_coverage":{"path":DATA/"raw/rope_resources_v1/rope365__d99c6a7fa089567b35e6.json","url":"https://rope365.com/rope-artists-references/","document_sha256":"4d1f83931a3d68a617d6166d2684f4e575e9999fe8bf49393c68ba32e9674537","markers":("- Itō Seiu (1882 – 1961)","- Akechi Denki (1940 – 2005)","- Chimuo Nureki","- Yukimura Haruki (1948-2016)","- Osada Eikechi","Coming soon!")},
}
FACTS=(
 {"source_key":"book_selection","topic":"book_selection","question":"Which starting book does Rope365 recommend for core knowledge and diverse body representation?","answer":"Better Bondage for Every Body by Evie Vane.","paraphrase_rationale":"This resource-selection answer retains the title, author, core-knowledge focus, and diversity criterion stated by the source."},
 {"source_key":"community_search","topic":"community_search","question":"Which online tools does Rope365 suggest for finding local groups and classes?","answer":"FetLife area groups, Shibari Map, and its region-by-region resource list.","paraphrase_rationale":"This combines three explicitly listed discovery paths into a concise navigation answer without endorsing an unvetted group."},
 {"source_key":"artist_reference_coverage","topic":"artist_reference_coverage","question":"Which artists currently have developed profiles on Rope365's reference page, rather than “coming soon” placeholders?","answer":"Itō Seiu, Akechi Denki, and Yukimura Haruki.","paraphrase_rationale":"The answer distinguishes the three populated artist sections from the Chimuo Nureki and Osada Eikechi placeholders."},
)
def file_sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def text_sha256(text):return hashlib.sha256(text.encode()).hexdigest()
def portable(path):return str(Path(path).resolve().relative_to(ROOT))
def read_jsonl(path):return[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def write_jsonl(path,rows):Path(path).write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in rows))
def build_baseline(path,report):
 baseline_builder.build_projection(path,report);rows=read_jsonl(path)
 if(len(rows),file_sha256(path))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v294 baseline drift")
 return rows
def evidence(doc,markers):
 out=[]
 for marker in markers:
  matches=[line for line in doc["text"].splitlines() if marker in line]
  if not matches:raise ValueError(f"evidence drift: {marker}")
  line=matches[0]
  if line not in out:out.append(line)
 return"\n".join(out)
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True);documents={}
 for k,s in SOURCES.items():
  d=json.loads(s["path"].read_text())
  if(d["url"],d["document_sha256"],text_sha256(d["text"]))!=(s["url"],s["document_sha256"],s["document_sha256"]):raise ValueError(f"{k}: source drift")
  documents[k]=d
 with tempfile.TemporaryDirectory(prefix="resource-v6-",dir=OUT_DIR) as temp:baseline=build_baseline(Path(temp)/"v294.jsonl",Path(temp)/"v294.report.json")
 basefacts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train") for r in baseline];qs={normalize_text(r["question"]) for r in baseline};pairs={(normalize_text(r["question"]),normalize_text(r["answer"])) for r in baseline};docs={r["document_sha256"] for r in baseline};urls={r["url"].rstrip("/").casefold() for r in baseline};rows=[]
 for fact in FACTS:
  s=SOURCES[fact["source_key"]];q,a=fact["question"],fact["answer"];pair=normalize_text(q),normalize_text(a)
  if not q.endswith("?") or "\n"in q or"\n"in a or has_protocol_tokens(q)or has_protocol_tokens(a)or parse_qa(f"Question: {q}\nAnswer: {a}")!=(q,a):raise ValueError("noncanonical Q&A")
  if pair in pairs or pair[0]in qs or leakage_reason(q,a,basefacts):raise ValueError("train collision")
  if s["document_sha256"]in docs or s["url"].rstrip("/").casefold()in urls:raise ValueError("non-novel source")
  support=evidence(documents[fact["source_key"]],s["markers"]);render=f"Question: {q}\nAnswer: {a}";rows.append({"answer":a,"claim_type":"resource_navigation","document_sha256":s["document_sha256"],"evidence":support,"evidence_sha256":text_sha256(support),"evidence_url":s["url"],"fact_id":stable_fact_id(q,a),"kind":"qa_resource_manual_fact","paraphrase_rationale":fact["paraphrase_rationale"],"quality_schema":"manual-resource-fact-v1","question":q,"resource_id":"rope365","reviewer":"codex-resource-additions-v6","source":"rope365","source_lineage":{"artifact":portable(OUTPUT),"raw_document":portable(s["path"]),"resource_manifest":portable(RESOURCE_MANIFEST)},"text":render,"topic":fact["topic"],"url":s["url"],"verified_at":"2026-07-15"})
 if len(rows)!=3 or len({r["fact_id"]for r in rows})!=3:raise ValueError("identity drift")
 write_jsonl(OUTPUT,rows);sha=file_sha256(OUTPUT)
 if EXPECTED_OUTPUT_SHA256!="PENDING"and sha!=EXPECTED_OUTPUT_SHA256:raise ValueError("hash drift")
 strata=Counter(classify_stratum(r)for r in rows);report={"artifact":{"path":portable(OUTPUT),"rows":3,"sha256":sha},"baseline":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"method":{"authoring":"manual full-source review and hand-authored Q&A","collision_scope":"v294 train-only projection; sealed collisions delegated to integration tooling","selection":"one navigation fact from each of three distinct, previously unrepresented source documents and URLs"},"new_independent_inputs":{"document_sha256s":3,"expected_strata":dict(sorted(strata.items())),"urls":3},"reviewed_at":"2026-07-15","reviewer":"codex-resource-additions-v6","schema":"manual-resource-additions-report-v6","sources":{k:{"document_sha256":v["document_sha256"],"file_sha256":file_sha256(v["path"]),"path":portable(v["path"]),"url":v["url"]}for k,v in sorted(SOURCES.items())},"status":"segregated_pending_integration"};REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
