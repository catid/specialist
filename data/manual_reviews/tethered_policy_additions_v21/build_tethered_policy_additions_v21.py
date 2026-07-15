#!/usr/bin/env python3
"""Build three manually reviewed, year-attributed Tethered Together policy QAs."""
from __future__ import annotations
import json,sys,tempfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/"data";V309=DATA/"manual_reviews/context_merit_audit_v309";V20=DATA/"manual_reviews/learning_method_additions_v20"
sys.path[:0]=[str(ROOT),str(V309),str(V20)]
import build_context_merit_audit_v309 as baseline_builder
import build_learning_method_additions_v20 as prior
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact,has_protocol_tokens,leakage_reason,normalize_text,parse_qa,stable_fact_id
OUT_DIR=Path(__file__).resolve().parent;OUTPUT=OUT_DIR/"pending_additions_tethered_policy_tranche_21_v1.jsonl";REPORT=OUT_DIR/"report_tethered_policy_tranche_21_v1.json";BASELINE_ROWS=551;BASELINE_SHA256="7b4d4985605d2b3454bef76015de9ccb3aa1ae06a9d8d763c44f96755cca8b4a";EXPECTED_OUTPUT_SHA256="a270e1943b4a3d5d9cdf0f03248054c0a6b9584dfb621794a363e1181e5944af";RESOURCE_MANIFEST=ROOT/"sources/rope_resources_v1.json"
file_sha256,text_sha256,portable=prior.file_sha256,prior.text_sha256,prior.portable;read_jsonl,write_jsonl,select_evidence=prior.read_jsonl,prior.write_jsonl,prior.select_evidence
SOURCE={"path":DATA/"raw/rope_resources_v1/tethered_together__cfcc0aab47acc802dd94.json","url":"https://tetheredtogether.net/general-policy-information/","document_sha256":"8cd9d29b428d6dd202088a9473472c65f048fe6d72364d02e4ba0bfd63216ece"}
FACTS=(
 {"topic":"attendee_confidentiality","question":"What confidentiality rule do Tethered Together’s collected 2025 rules give for recognizing another attendee?","answer":"Do not share their presence or information without explicit permission; outing an attendee is treated as a serious offense.","markers":("All activities that take place at Tethered Together should be treated with confidentiality. Adhere to the photography policy, and do not share information about other attendees. If you happen to see someone you know from the vanilla world – don’t make it weird, and do not share that information with anyone without explicit permission. Outing an attendee is considered a serious offense and could result in your expulsion from this event, and future events.",),"paraphrase_rationale":"This preserves the collected policy's confidentiality boundary and consequence while explicitly dating the rule in the question."},
 {"topic":"photography_active_consent","question":"What photography consent rule appears in Tethered Together’s collected 2025 event policy?","answer":"Create active consent and photograph only consenting people; nonconsenting bystanders may not appear even if the photographer plans to crop or blur them.","markers":("Tethered Together does not hire official event photographers. We do allow attendees who create active consent to take photographs using their personal cell phones. Only consenting attendees may be photographed and absolutely no people who do not consent (people walking by, etc.) may be in any photographs taken even if you plan to blur or crop them out.",),"paraphrase_rationale":"This retains the active-consent and incidental-bystander rules while omitting volatile enforcement and device details."},
 {"topic":"event_emergency_routing","question":"How do Tethered Together’s collected 2025 rules route medical emergencies?","answer":"Call 911 for a life-threatening emergency and notify event staff; request basic non-emergency first aid through the hospitality desk or First Aid Station.","markers":("First Aid Staff Members will be available to assist with any basic medical needs. In the event of a life-threatening emergency, call 911. Please notify event staff in the event of an emergency. DMs will be present at all parties and will serve as first responders in the event that a situation arises there.","If you need basic first aid during our hours of operation, please ask the hospitality desk to radio for non-emergency first aid if no one is at the First Aid Station."),"paraphrase_rationale":"This combines the source's life-threatening and basic-first-aid routes without implying event staff replace emergency services."},
)
def build_baseline(path,report):
 baseline_builder.build_projection(path,report);rows=read_jsonl(path)
 if (len(rows),file_sha256(path))!=(BASELINE_ROWS,BASELINE_SHA256):raise ValueError("v309 drift")
 return rows
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True);d=json.loads(SOURCE["path"].read_text())
 if (d["url"],d["document_sha256"],text_sha256(d["text"]))!=(SOURCE["url"],SOURCE["document_sha256"],SOURCE["document_sha256"]):raise ValueError("source drift")
 with tempfile.TemporaryDirectory(prefix="tethered-policy-v21-",dir=OUT_DIR) as t:baseline=build_baseline(Path(t)/"v309.jsonl",Path(t)/"v309.report.json")
 facts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train") for r in baseline];qs={normalize_text(r["question"]) for r in baseline};pairs={(normalize_text(r["question"]),normalize_text(r["answer"])) for r in baseline};docids={r["document_sha256"] for r in baseline};urls={r["url"].rstrip("/").casefold() for r in baseline};rows=[]
 if SOURCE["document_sha256"] in docids or SOURCE["url"].rstrip("/").casefold() in urls:raise ValueError("source not novel")
 for f in FACTS:
  q,a=f["question"],f["answer"];pair=normalize_text(q),normalize_text(a);rendered=f"Question: {q}\nAnswer: {a}"
  if not q.endswith("?") or "\n" in q+a or has_protocol_tokens(q) or has_protocol_tokens(a) or parse_qa(rendered)!=(q,a):raise ValueError("noncanonical")
  if pair in pairs or pair[0] in qs or leakage_reason(q,a,facts):raise ValueError("train collision")
  ev=select_evidence(d,f["markers"]);rows.append({"answer":a,"claim_type":"event_policy_attributed","document_sha256":SOURCE["document_sha256"],"evidence":ev,"evidence_sha256":text_sha256(ev),"evidence_url":SOURCE["url"],"fact_id":stable_fact_id(q,a),"kind":"qa_resource_manual_fact","paraphrase_rationale":f["paraphrase_rationale"],"policy_year":2025,"quality_schema":"manual-resource-fact-v1","question":q,"resource_id":"tethered_together","reviewer":"codex-tethered-policy-additions-v21","source":"tethered_together","source_lineage":{"artifact":portable(OUTPUT),"raw_document":portable(SOURCE["path"]),"resource_manifest":portable(RESOURCE_MANIFEST)},"text":rendered,"topic":f["topic"],"url":SOURCE["url"],"verified_at":"2026-07-15"})
 if len(rows)!=3 or len({r["fact_id"] for r in rows})!=3:raise ValueError("identity drift")
 write_jsonl(OUTPUT,rows);sha=file_sha256(OUTPUT)
 if EXPECTED_OUTPUT_SHA256!="PENDING" and sha!=EXPECTED_OUTPUT_SHA256:raise ValueError("artifact drift")
 strata=Counter(classify_stratum(r) for r in rows);REPORT.write_text(json.dumps({"artifact":{"path":portable(OUTPUT),"rows":3,"sha256":sha},"baseline":{"rows":BASELINE_ROWS,"sha256":BASELINE_SHA256},"method":{"authoring":"manual full-source review and hand-authored Q&A","collision_scope":"v309 train-only projection; sealed collisions delegated to integration tooling","selection":"three distinct, explicitly year-attributed rules from one fully reviewed event-policy document","temporal_handling":"Questions identify the collected policy as 2025; answers do not present it as current universal policy."},"new_independent_inputs":{"document_sha256s":1,"expected_strata":dict(sorted(strata.items())),"urls":1},"reviewed_at":"2026-07-15","reviewer":"codex-tethered-policy-additions-v21","schema":"manual-tethered-policy-additions-report-v21","status":"segregated_pending_integration"},ensure_ascii=False,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
