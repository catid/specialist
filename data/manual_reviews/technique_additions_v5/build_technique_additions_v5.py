#!/usr/bin/env python3
"""Build three manual technique QAs from distinct structural-practice pages."""
from __future__ import annotations
import hashlib, json, sys, tempfile
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; DATA = ROOT / "data"; V293_DIR = DATA / "manual_reviews/context_merit_audit_v293"; sys.path[:0] = [str(ROOT), str(V293_DIR)]
import build_context_merit_audit_v293 as baseline_builder
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact, has_protocol_tokens, leakage_reason, normalize_text, parse_qa, stable_fact_id
OUT_DIR = Path(__file__).resolve().parent; OUTPUT = OUT_DIR / "pending_additions_technique_tranche_05_v1.jsonl"; REPORT = OUT_DIR / "report_technique_tranche_05_v1.json"
BASELINE_ROWS = 504; BASELINE_SHA256 = "1913fbab4e3b0638683b44f02682cd9316548750046f13c70bd8abfe7327d960"; EXPECTED_OUTPUT_SHA256 = "5258c9a759bb5a5f4bce26671a5476c3f7e6454e6700c9aaed8d6195e20fc9fd"; RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
SOURCES = {
 "restricted_movement_feedback": {"path": DATA / "raw/rope_resources_v1/rope365__a647a43555bde33c420b.json", "url": "https://rope365.com/assembling-ties/", "document_sha256": "08aca4bd46b5e97c2133074a52408ae1469b2acf2b3395b1f8a3ba07033fa509", "markers": ("To move around in a futomomo can be an interesting way of exploring both your body and the tie. Once you are tied up, take a moment to walk on your knees, roll around, crawl and do whatever is possible for you to do. Find out how it feels for you to move in this restriction; wonderful, horrible, fun, sexy? This way you can learn more together with your partner about what dynamics you appreciate when being in ropes. Also, if the futomomo disintegrates through your movement, the person tying you can learn a lot about frictions and stability from that.",)},
 "add_second_column": {"path": DATA / "raw/rope_resources_v1/rope365__8cda4e4fbfe05ce2b00b.json", "url": "https://rope365.com/more-columns/", "document_sha256": "f8f574fcfe8f0dec7a01807b19e44a51d089356b47d7de8d1eb34de90c16d722", "markers": ("Starting from a single column tie on one column then adding a second column is a great strategy to change position during the tie without having to retie the starting point.",)},
 "clove_hitch_stack_tradeoff": {"path": DATA / "raw/rope_resources_v1/rope365__f4a15e598bb62feddb04.json", "url": "https://rope365.com/clove-hitch/", "document_sha256": "4d54e41a0ae892ac0a3b6a92d21cb1259076ba6cae159951fd002ea47b151d36", "markers": ("To make a clove hitch using the middle of the rope you need to pull the whole rope through twice but there is a trick to avoid this if the end of the column is accessible. Just create a loop and insert the column into it. You can also prepare the clove hitch in your hand and slide the whole column into it then tighten. This method also works for faster half hitches.", "This method makes it a bit more tricky to avoid undesired twists and the tension can be a challenge to maintain compared with threading with the tail.")},
}
FACTS = (
 {"source_key": "restricted_movement_feedback", "topic": "restricted_movement_feedback", "question": "What can moving in a futomomo reveal to both partners?", "answer": "It can reveal which restricted-movement dynamics the tied person enjoys, while any disintegration shows the tying partner where friction and stability need work.", "paraphrase_rationale": "This preserves the source's two-sided feedback lesson while omitting its optional movement examples."},
 {"source_key": "add_second_column", "topic": "add_second_column", "question": "What is one advantage of starting with a single-column tie before adding a second column?", "answer": "It lets the partners change position during the tie without retying the starting point.", "paraphrase_rationale": "This is a direct, standalone paraphrase of the sequencing advantage stated by the source."},
 {"source_key": "clove_hitch_stack_tradeoff", "topic": "clove_hitch_stack_tradeoff", "question": "What tradeoff does the clove-hitch stack method have compared with threading the tail?", "answer": "It avoids pulling the whole rope through twice when the column end is accessible, but unwanted twists and tension control can be harder.", "paraphrase_rationale": "This combines the source's efficiency benefit and handling drawback without adding an unsupported recommendation."},
)
def file_sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def text_sha256(text): return hashlib.sha256(text.encode()).hexdigest()
def portable(path): return str(Path(path).resolve().relative_to(ROOT))
def read_jsonl(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def write_jsonl(path, rows): Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True)+"\n" for r in rows))
def build_baseline(path, report):
 baseline_builder.build_projection(path, report); rows=read_jsonl(path)
 if (len(rows),file_sha256(path))!=(BASELINE_ROWS,BASELINE_SHA256): raise ValueError("v293 baseline drift")
 return rows
def evidence(document, markers):
 out=[]
 for marker in markers:
  matches=[line for line in document["text"].splitlines() if marker in line]
  if len(matches)!=1: raise ValueError(f"evidence drift: {marker}")
  out.append(matches[0])
 return "\n".join(out)
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True);documents={}
 for key,source in SOURCES.items():
  document=json.loads(source["path"].read_text())
  if (document["url"],document["document_sha256"],text_sha256(document["text"]))!=(source["url"],source["document_sha256"],source["document_sha256"]): raise ValueError(f"{key}: source drift")
  documents[key]=document
 with tempfile.TemporaryDirectory(prefix="technique-v5-",dir=OUT_DIR) as temp: baseline=build_baseline(Path(temp)/"v293.jsonl",Path(temp)/"v293.report.json")
 facts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train") for r in baseline]; questions={normalize_text(r["question"]) for r in baseline}; pairs={(normalize_text(r["question"]),normalize_text(r["answer"])) for r in baseline}; docs={r["document_sha256"] for r in baseline}; urls={r["url"].rstrip("/").casefold() for r in baseline}; rows=[]
 for fact in FACTS:
  source=SOURCES[fact["source_key"]];q,a=fact["question"],fact["answer"];pair=normalize_text(q),normalize_text(a)
  if not q.endswith("?") or "\n" in q or "\n" in a or has_protocol_tokens(q) or has_protocol_tokens(a) or parse_qa(f"Question: {q}\nAnswer: {a}")!=(q,a): raise ValueError("non-canonical Q&A")
  if pair in pairs or pair[0] in questions or leakage_reason(q,a,facts): raise ValueError("train collision")
  if source["document_sha256"] in docs or source["url"].rstrip("/").casefold() in urls: raise ValueError("non-novel source")
  support=evidence(documents[fact["source_key"]],source["markers"]);rendered=f"Question: {q}\nAnswer: {a}"
  rows.append({"answer":a,"claim_type":"instructional","document_sha256":source["document_sha256"],"evidence":support,"evidence_sha256":text_sha256(support),"evidence_url":source["url"],"fact_id":stable_fact_id(q,a),"kind":"qa_resource_manual_fact","paraphrase_rationale":fact["paraphrase_rationale"],"quality_schema":"manual-resource-fact-v1","question":q,"resource_id":"rope365","reviewer":"codex-technique-additions-v5","source":"rope365","source_lineage":{"artifact":portable(OUTPUT),"raw_document":portable(source["path"]),"resource_manifest":portable(RESOURCE_MANIFEST)},"text":rendered,"topic":fact["topic"],"url":source["url"],"verified_at":"2026-07-15"})
 if len(rows)!=3 or len({r["fact_id"] for r in rows})!=3: raise ValueError("identity drift")
 write_jsonl(OUTPUT,rows);sha=file_sha256(OUTPUT)
 if EXPECTED_OUTPUT_SHA256!="PENDING" and sha!=EXPECTED_OUTPUT_SHA256: raise ValueError("hash drift")
 strata=Counter(classify_stratum(r) for r in rows);report={"artifact":{"path":portable(OUTPUT),"rows":3,"sha256":sha},"baseline":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"method":{"authoring":"manual full-source review and hand-authored Q&A","collision_scope":"v293 train-only projection; sealed collisions delegated to integration tooling","selection":"one useful technique fact from each of three distinct, previously unrepresented source documents and URLs"},"new_independent_inputs":{"document_sha256s":3,"expected_strata":dict(sorted(strata.items())),"urls":3},"reviewed_at":"2026-07-15","reviewer":"codex-technique-additions-v5","schema":"manual-technique-additions-report-v5","sources":{k:{"document_sha256":v["document_sha256"],"file_sha256":file_sha256(v["path"]),"path":portable(v["path"]),"url":v["url"]} for k,v in sorted(SOURCES.items())},"status":"segregated_pending_integration"};REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
