#!/usr/bin/env python3
"""Build three durable community-format QAs from new source pages."""
from __future__ import annotations
import hashlib,json,sys,tempfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V297=DATA/"manual_reviews/context_merit_audit_v297";sys.path[:0]=[str(ROOT),str(V297)]
import build_context_merit_audit_v297 as baseline_builder
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact,has_protocol_tokens,leakage_reason,normalize_text,parse_qa,stable_fact_id
OUT_DIR=Path(__file__).resolve().parent;OUTPUT=OUT_DIR/"pending_additions_community_tranche_09_v1.jsonl";REPORT=OUT_DIR/"report_community_tranche_09_v1.json";BASELINE_ROWS=516;BASELINE_SHA256="93097a170a71269f46b538fdfdf40a22e9d228795abbf0262deb52cc8aa4535c";EXPECTED_OUTPUT_SHA256="9b96a61054a7cc36d1a7ef34b3d5576d476c8c9f9663142ceab14ed40ad2b16e";MANIFEST=ROOT/"sources/rope_resources_v1.json"
SOURCES={
 "lab_consent_rule":{"path":DATA/"raw/rope_resources_v1/austin_rope_slingers__4d6ac1f27a5e8eb8d89b.json","url":"https://www.austinropeslingers.com/resources/rope-lab-rules/","document_sha256":"2ae0ef12145bb955fe5cd3a5f12dae439bbe06452e5d139d30bf4b1317ee9b65","markers":("Ask before joining a tie, practice, or scene.","If you’re at Rope Lab please feel free to ask questions but do not participate without clear permission.")},
 "reclamation_learning_formats":{"path":DATA/"raw/rope_resources_v1/atx_empty_space__a9598a537db1b5e31278.json","url":"https://www.atxempty.space/reclamationrope","document_sha256":"1e03db4526aabb598a36f8dd38ee100f0f1c3dca63426b0f2e46b42ab79a6c0d","markers":("Our in-person labs and rope tastings include beginner rope instruction, peer to peer learning and an opportunity to experience rope (as top or bottom).", "Our virtual vibe sessions are online kinky kickback style. We gather on Zoom to practice rope, ask questions, story tell, try new things and vibe out together. Join this space for peer-to-peer education and sharing despite where you may be based in the world.")},
 "loaner_rope":{"path":DATA/"raw/rope_resources_v1/austin_rope_slingers__36ed1cdde55a1699e94a.json","url":"https://www.austinropeslingers.com/faq/","document_sha256":"db3f3fb90e1015919834a3b7a0d15e42f688993d1b4f29d60fa578a3431474cd","markers":("Q) Do I need to bring my own rope?", "You’re certainly welcome to, but we have a loaner rope box if you don’t have any yet.")},
}
FACTS=(
 {"source_key":"lab_consent_rule","topic":"lab_consent_rule","question":"What consent rule applies before someone joins a tie, practice, or scene at Austin Rope Slingers?","answer":"Get clear permission from the participants; questions are welcome, but participation is not allowed without it.","paraphrase_rationale":"This preserves the lab's distinction between asking questions and joining an activity without adding a broader policy."},
 {"source_key":"reclamation_learning_formats","topic":"reclamation_learning_formats","question":"What in-person and virtual learning formats does Reclamation Rope describe?","answer":"In-person labs and tastings offer beginner instruction, peer learning, and top-or-bottom experience; virtual vibe sessions offer online practice, questions, stories, and peer sharing.","paraphrase_rationale":"This summarizes the durable format and purpose of both offerings while excluding their volatile schedule."},
 {"source_key":"loaner_rope","topic":"loaner_rope","question":"Does Austin Rope Slingers say a newcomer must bring their own rope?","answer":"No; its FAQ says a loaner rope box is available for people who do not have rope yet.","paraphrase_rationale":"This directly summarizes the FAQ's durable newcomer-access information without relying on schedule or price details."},
)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tsha(t):return hashlib.sha256(t.encode()).hexdigest()
def portable(p):return str(Path(p).resolve().relative_to(ROOT))
def read(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x.strip()]
def write(p,rs):Path(p).write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n"for r in rs))
def baseline(p,rep):
 baseline_builder.build_projection(p,rep);rs=read(p)
 if(len(rs),sha(p))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v297 baseline drift")
 return rs
def evidence(d,ms):
 out=[]
 for m in ms:
  hits=[line for line in d["text"].splitlines()if m in line]
  if not hits:raise ValueError(f"evidence drift:{m}")
  if hits[0]not in out:out.append(hits[0])
 return"\n".join(out)
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True);docs={}
 for k,s in SOURCES.items():
  d=json.loads(s["path"].read_text())
  if(d["url"],d["document_sha256"],tsha(d["text"]))!=(s["url"],s["document_sha256"],s["document_sha256"]):raise ValueError("source drift")
  docs[k]=d
 with tempfile.TemporaryDirectory(prefix="community-v9-",dir=OUT_DIR)as temp:base=baseline(Path(temp)/"v297.jsonl",Path(temp)/"v297.report.json")
 facts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train")for r in base];qs={normalize_text(r["question"])for r in base};pairs={(normalize_text(r["question"]),normalize_text(r["answer"]))for r in base};docids={r["document_sha256"]for r in base};urls={r["url"].rstrip("/").casefold()for r in base};rows=[]
 for f in FACTS:
  s=SOURCES[f["source_key"]];q,a=f["question"],f["answer"];pair=normalize_text(q),normalize_text(a)
  if not q.endswith("?")or"\n"in q or"\n"in a or has_protocol_tokens(q)or has_protocol_tokens(a)or parse_qa(f"Question: {q}\nAnswer: {a}")!=(q,a):raise ValueError("noncanonical")
  if pair in pairs or pair[0]in qs or leakage_reason(q,a,facts):raise ValueError("train collision")
  if s["document_sha256"]in docids or s["url"].rstrip("/").casefold()in urls:raise ValueError("non-novel source")
  ev=evidence(docs[f["source_key"]],s["markers"]);render=f"Question: {q}\nAnswer: {a}";d=docs[f["source_key"]];rows.append({"answer":a,"claim_type":"community_navigation","document_sha256":s["document_sha256"],"evidence":ev,"evidence_sha256":tsha(ev),"evidence_url":s["url"],"fact_id":stable_fact_id(q,a),"kind":"qa_resource_manual_fact","paraphrase_rationale":f["paraphrase_rationale"],"quality_schema":"manual-resource-fact-v1","question":q,"resource_id":d["resource_id"],"reviewer":"codex-community-additions-v9","source":d["source"],"source_lineage":{"artifact":portable(OUTPUT),"raw_document":portable(s["path"]),"resource_manifest":portable(MANIFEST)},"text":render,"topic":f["topic"],"url":s["url"],"verified_at":"2026-07-15"})
 write(OUTPUT,rows);outsha=sha(OUTPUT)
 if EXPECTED_OUTPUT_SHA256!="PENDING"and outsha!=EXPECTED_OUTPUT_SHA256:raise ValueError("hash drift")
 strata=Counter(classify_stratum(r)for r in rows);report={"artifact":{"path":portable(OUTPUT),"rows":3,"sha256":outsha},"baseline":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"method":{"authoring":"manual full-source review and hand-authored Q&A","collision_scope":"v297 train-only projection; sealed collisions delegated to integration tooling","selection":"one durable community-format fact from each of three new documents and URLs"},"new_independent_inputs":{"document_sha256s":3,"expected_strata":dict(sorted(strata.items())),"urls":3},"reviewed_at":"2026-07-15","reviewer":"codex-community-additions-v9","schema":"manual-community-additions-report-v9","status":"segregated_pending_integration"};REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
