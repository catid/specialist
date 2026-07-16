#!/usr/bin/env python3
import hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path
from curated import RECORDS, TAXONOMY

HERE=Path(__file__).resolve().parent
DATE="2026-07-16"
MEDIA_TIES={36,71,78,104,106,115,122,208,209}
MEDIA_URLS={f"https://crash-restraint.com/ties/{x}" for x in MEDIA_TIES}|{
 "https://crash-restraint.com/blog/files/Chest%20Loading%20Takate%20Kote%20Handout.pdf",
 "https://crash-restraint.com/blog/files/ashley%20whippings.pdf",
 "https://crash-restraint.com/blog/files/somerville_bowline_cannon.pdf"}

def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def reason(url, disposition):
 if disposition=="included": return "Durable rope knowledge paraphrased in the direct-training corpus."
 if disposition=="media": return "No sufficient same-domain prose to reconstruct the visual construction; media limitation recorded without inference."
 if disposition=="gated": return "Redirects to authentication and contains no public instructional content."
 if disposition=="error": return "Discovery endpoint returned a non-success status."
 if "/ties/category/" in url: return "Category/navigation index; all substantive child pages were inventoried directly."
 if url.endswith("/ties/264"): return "Site release log and account/UI changes; no independent durable rope instruction."
 if url.endswith("/robots.txt"): return "Discovery-control file, not instructional content."
 if "/users/" in url: return "Account-management shell, not rope instruction."
 if url.endswith("/tou") or url.endswith("/contact"): return "Legal/contact shell, unrelated to durable rope knowledge."
 if url=="https://crash-restraint.com/": return "Landing/navigation page; substantive pages inventoried directly."
 return "Non-instructional shell or duplicate."

def render():
 snap=json.loads((HERE/"source_snapshot.json").read_text())
 included={r["url"] for r in RECORDS}
 lines=["# Crash Restraint — canonical direct-training corpus","",f"Retrieval date: {DATE}","", "This clean-room corpus is an information-dense original paraphrase of the accessible same-domain instructional text. It is not Q&A and not a site mirror. Page titles and URLs identify provenance; navigation, account controls, marketing, and repeated chrome are removed.","", "All medical, engineering, numerical, and risk-tolerance statements remain claims of the cited source unless a section explicitly says otherwise. The corpus distinguishes uncertainty and does not turn source experience into medical consensus, certification, or a universal safe threshold. Visual-only steps are marked and never inferred.",""]
 cats=Counter()
 for r in sorted(RECORDS,key=lambda x:x["url"]):
  lines += [f"## {r['title']}","",f"Source URL: {r['url']}",""]
  if r.get("level"): lines += [f"Source level: {r['level']}",""]
  for s in r["sections"]:
   lines += [f"### {s['heading']}","", "Categories: "+", ".join(s["categories"]),"",s["text"],""]
   cats.update(s["categories"])
 md="\n".join(lines).rstrip()+"\n"
 entries=[]
 for x in snap["pages"]:
  u=x["url"]
  if u in included:d="included"
  elif u in MEDIA_URLS:d="media"
  elif u.endswith("/users/preferences"):d="gated"
  elif x["http_status"]!=200:d="error"
  else:d="excluded"
  entries.append({**x,"disposition":d,"reason":reason(u,d)})
 manifest={"schema":"site-corpus-manifest-v1","resource_id":"crash_restraint","retrieved_at":DATE,"direct_training_ready":True,"non_qa":True,"entries":sorted(entries,key=lambda x:x["url"])}
 mtxt=json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False)+"\n"
 gaps={
  "lineage_history_people":"Scattered technique attribution only; no comprehensive history or lineage survey.",
  "bottoming_skills":"Strong self-advocacy and symptom material, but the author explicitly lacks regular bottoming experience.",
  "accessibility_adaptation":"Body-shape adaptations are present; disability- and mobility-specific protocols are sparse.",
  "aesthetics_performance":"Some decorative architecture and performance cautions; little systematic composition or photography instruction.",
  "emergency_procedures":"Cutting, fainting, falls, and lowering are covered, but this is not a complete current first-aid curriculum."}
 dispositions=Counter(e["disposition"] for e in entries)
 report={"schema":"site-corpus-report-v1","resource_id":"crash_restraint","retrieved_at":DATE,"role":"canonical_markdown_direct_training_source","direct_training_ready":True,"non_qa":True,"source_claim_policy":"Source-specific claims are attributed; medical and engineering uncertainty is preserved.","counts":{"records":len(RECORDS),"sections":sum(len(r["sections"]) for r in RECORDS),"words":len(re.findall(r"\b[\w’'-]+\b",md)),"urls_inventory":len(entries),"dispositions":dict(sorted(dispositions.items()))},"supported_categories":[c for c in TAXONOMY if cats[c]],"category_section_counts":{c:cats[c] for c in TAXONOMY if cats[c]},"genuine_gaps":gaps,"hashes":{"markdown_sha256":sha(md),"manifest_sha256":sha(mtxt),"source_snapshot_sha256":hashlib.sha256((HERE/"source_snapshot.json").read_bytes()).hexdigest()}}
 return md,mtxt,json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+"\n"

def build():
 md,mf,rp=render();(HERE/"crash_restraint.md").write_text(md);(HERE/"manifest.json").write_text(mf);(HERE/"report.json").write_text(rp)
if __name__=="__main__": build()
