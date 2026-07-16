import hashlib,json,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import builder
from curated import TAXONOMY

def load():
 return (HERE/"crash_restraint.md").read_text(),json.loads((HERE/"manifest.json").read_text()),json.loads((HERE/"report.json").read_text())
def test_hashes_and_determinism():
 md,mf,rp=builder.render(); assert hashlib.sha256(md.encode()).hexdigest()==json.loads(rp)["hashes"]["markdown_sha256"]
 assert hashlib.sha256(mf.encode()).hexdigest()==json.loads(rp)["hashes"]["manifest_sha256"]
 assert (md,mf,rp)==builder.render()
def test_inventory_and_citations_equal():
 md,m,r=load(); snap=json.loads((HERE/"source_snapshot.json").read_text())
 assert {e["url"] for e in m["entries"]}=={e["url"] for e in snap["pages"]}
 cited=set(re.findall(r"^Source URL: (https://\S+)$",md,re.M)); included={e["url"] for e in m["entries"] if e["disposition"]=="included"}; assert cited==included
def test_taxonomy_mapping():
 md,m,r=load(); assert set(r["supported_categories"])==set(TAXONOMY)
 labels=re.findall(r"^Categories: (.+)$",md,re.M); assert len(labels)==r["counts"]["sections"]
 assert all(set(x.split(", "))<=set(TAXONOMY) for x in labels)
def test_direct_training_and_no_ui_or_qa_leakage():
 md,m,r=load(); assert m["direct_training_ready"] and m["non_qa"] and r["role"]=="canonical_markdown_direct_training_source"
 low=md.lower();
 for bad in ["sign in to save progress","show related ties","cookie policy","site news & updates","question:","answer:","heldout","benchmark id","qa row"]: assert bad not in low
def test_dispositions_and_media_limitations():
 md,m,r=load(); assert r["counts"]["urls_inventory"]==181
 assert {"included","excluded","media","gated","error"}==set(r["counts"]["dispositions"])
 assert "Media limitation:" in md
