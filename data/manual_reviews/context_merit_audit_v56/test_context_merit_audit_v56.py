#!/usr/bin/env python3
"""Focused regression tests for context audit v56."""
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import build_context_merit_audit_v56 as b
from qa_quality import normalize_text

class V56(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory(prefix=".test-v56-",dir=HERE);o=Path(cls.temp.name)
  cls.base=o/"base.jsonl";cls.br=o/"base.report.json";b.build_projection(cls.base,cls.br,b.PRIOR_PROJECTION_CURATIONS);cls.baseline=b.read_jsonl(cls.base)
  cur=(*b.PRIOR_PROJECTION_CURATIONS,b.CURATION);cls.ds=[]
  for n in (1,2):
   p=o/f"p{n}.jsonl";b.build_projection(p,o/f"r{n}.json",cur);cls.ds.append(p)
  cls.rows=b.read_jsonl(cls.ds[0]);cls.pr=json.loads((o/"r1.json").read_text());cls.audit=b.read_jsonl(b.AUDIT);cls.curation=b.read_jsonl(b.CURATION);cls.report=json.loads(b.REPORT.read_text())
 @classmethod
 def tearDownClass(cls): cls.temp.cleanup()
 def test_01_baseline_rows(self): self.assertEqual(len(self.baseline),541)
 def test_02_baseline_hash(self): self.assertEqual(b.file_sha256(self.base),b.PROJECTED_SELECTION_BASELINE["sha256"])
 def test_03_eval_tooling_count(self): self.assertEqual(json.loads(self.br.read_text())["eval_fact_count"],612)
 def test_04_selection(self):
  ranked=b.ranked_unreviewed_direct(self.baseline);self.assertEqual(len(ranked),6);self.assertEqual(tuple(x["row"]["fact_id"] for x in ranked),b.EXPECTED_SELECTION)
 def test_05_indices(self): self.assertEqual({x["row"]["fact_id"]:x["active_index"] for x in b.ranked_unreviewed_direct(self.baseline)},b.PROJECTED_ACTIVE_INDICES)
 def test_06_direct_counts(self):
  reviewed=b.prior_reviewed_fact_ids();d=[r for r in self.baseline if not r.get("curation")];self.assertEqual((len(d),sum(r["fact_id"] in reviewed for r in d),sum(r["fact_id"] not in reviewed for r in d)),(239,233,6))
 def test_07_decisions(self): self.assertEqual({x:sum(r["decision"]==x for r in self.audit) for x in ("keep","drop","edit")},{"keep":4,"drop":1,"edit":1})
 def test_08_curation(self): self.assertEqual((len(self.curation),{r["action"] for r in self.curation}),(2,{"drop","edit"}))
 def test_09_sources(self):
  au={r["fact_id"]:r for r in self.audit}
  for s in b.SPECS:self.assertEqual(b.file_sha256(s["source_path"]),au[s["fact_id"]]["source_document_file_sha256"])
 def test_10_evidence(self):
  au={r["fact_id"]:r for r in self.audit}
  for s in b.SPECS:
   r=au[s["fact_id"]];self.assertEqual(b.text_sha256(r["support_evidence"]),r["support_evidence_sha256"])
   for f in s.get("paraphrase_support_fragments",()):self.assertIn(normalize_text(f),normalize_text(r["support_evidence"]))
 def test_11_lineage(self):
  for r in self.audit:self.assertEqual((r["projection_lineage"]["baseline_rows"],r["projection_lineage"]["baseline_sha256"]),(541,b.PROJECTED_SELECTION_BASELINE["sha256"]))
 def test_12_report(self):
  self.assertEqual((self.report["schema"],self.report["audit"]["by_decision"],self.report["new_pending_curation"]["edit_support_types"]),("context-merit-audit-report-v56",{"drop":1,"edit":1,"keep":4},{"extractive":1,"manual_paraphrase":0}))
 def test_13_report_projection(self): self.assertEqual((self.report["isolated_build_projection"]["output_rows"],self.report["isolated_build_projection"]["output_sha256"]),(540,"3178dc973f017cb4820223ecbb2a772faa072fc2eddb80fb54b23091f7034b24"))
 def test_14_prior_pins(self):
  pins=self.report["frozen_prior_decision_artifacts"];self.assertEqual(len(pins),165)
  for r in pins:self.assertEqual(b.file_sha256(b.ROOT/r["path"]),r["sha256"])
 def test_15_projection_hash(self): self.assertEqual([hashlib.sha256(p.read_bytes()).hexdigest() for p in self.ds],["3178dc973f017cb4820223ecbb2a772faa072fc2eddb80fb54b23091f7034b24"]*2)
 def test_16_projection_actions(self):
  m={r.get("curation",{}).get("original_fact_id"):r for r in self.rows if r.get("curation")};edit=next(x for x in b.SPECS if x["decision"]=="edit");drop=next(x for x in b.SPECS if x["decision"]=="drop");self.assertEqual((m[edit["fact_id"]]["question"],m[edit["fact_id"]]["answer"]),(edit["question"],edit["answer"]));self.assertNotIn(drop["fact_id"],{r["fact_id"] for r in self.rows}|set(m))
 def test_17_additions(self):
  expected={r["fact_id"] for p in b.PRIOR_PENDING_ADDITIONS for r in b.read_jsonl(p)};expected.remove("fact-93c032484cf3a72fcc5c");represented={r["fact_id"] for r in self.rows}|{r.get("curation",{}).get("original_fact_id") for r in self.rows};self.assertEqual(len(expected),37);self.assertTrue(expected<=represented)
 def test_18_urls(self):
  m=json.loads(b.RESOURCE_MANIFEST.read_text());u={x for r in m["resources"] for x in (r["canonical_url"],r.get("recommendation_url")) if x};blob="\n".join(json.dumps(r) for r in self.rows);self.assertEqual((len(u),{x for x in u if x not in blob}),(24,set()))
 def test_19_unique(self): self.assertEqual((len({r["fact_id"] for r in self.rows}),len({r["question"] for r in self.rows}),self.pr["eval_fact_count"]),(540,540,612))
if __name__=="__main__":unittest.main()
