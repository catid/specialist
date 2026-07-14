#!/usr/bin/env python3
"""Focused regression tests for pronoun/fragment answer audit v66."""
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import build_context_merit_audit_v66 as b
class V66(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory(prefix=".test-v66-",dir=HERE);o=Path(cls.temp.name);cls.base=o/"base.jsonl";cls.br=o/"base.report.json";b.build_projection(cls.base,cls.br,b.PRIOR_PROJECTION_CURATIONS);cls.baseline=b.read_jsonl(cls.base);cur=(*b.PRIOR_PROJECTION_CURATIONS,b.CURATION);cls.ds=[]
  for n in (1,2):p=o/f"p{n}.jsonl";b.build_projection(p,o/f"r{n}.json",cur);cls.ds.append(p)
  cls.rows=b.read_jsonl(cls.ds[0]);cls.pr=json.loads((o/"r1.json").read_text());cls.audit=b.read_jsonl(b.AUDIT);cls.curation=b.read_jsonl(b.CURATION);cls.report=json.loads(b.REPORT.read_text())
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def test_01_baseline_rows(self):self.assertEqual(len(self.baseline),537)
 def test_02_baseline_hash(self):self.assertEqual(b.file_sha256(self.base),b.PROJECTED_SELECTION_BASELINE["sha256"])
 def test_03_eval_tooling_count(self):self.assertEqual(json.loads(self.br.read_text())["eval_fact_count"],612)
 def test_04_selection(self):self.assertEqual(tuple(x["row"]["fact_id"] for x in b.selected(self.baseline)),b.EXPECTED_SELECTION)
 def test_05_indices(self):self.assertEqual({x["row"]["fact_id"]:x["active_index"] for x in b.selected(self.baseline)},b.PROJECTED_ACTIVE_INDICES)
 def test_06_direct_count(self):self.assertEqual(sum(not r.get("curation") for r in self.baseline),165)
 def test_07_decisions(self):self.assertEqual({x:sum(r["decision"]==x for r in self.audit) for x in ("keep","drop","edit")},{"keep":2,"drop":0,"edit":6})
 def test_08_curation(self):self.assertEqual((len(self.curation),{r["action"] for r in self.curation}),(6,{"edit"}))
 def test_09_sources(self):
  au={r["fact_id"]:r for r in self.audit}
  for s in b.SPECS:self.assertEqual(b.file_sha256(s["source_path"]),au[s["fact_id"]]["source_document_file_sha256"])
 def test_10_evidence(self):
  for r in self.audit:self.assertEqual(b.text_sha256(r["support_evidence"]),r["support_evidence_sha256"])
 def test_11_lineage(self):
  for r in self.audit:self.assertEqual((r["review_pass"],r["projection_lineage"]["baseline_rows"],r["projection_lineage"]["baseline_sha256"]),("pronoun_and_fragment_answer_reaudit",537,b.PROJECTED_SELECTION_BASELINE["sha256"]))
 def test_12_report(self):self.assertEqual((self.report["schema"],self.report["audit"]["by_decision"],self.report["new_pending_curation"]["edit_support_types"]),("context-merit-audit-report-v66",{"edit":6,"keep":2},{"extractive":0,"manual_paraphrase":6}))
 def test_13_report_projection(self):self.assertEqual((self.report["isolated_build_projection"]["output_rows"],self.report["isolated_build_projection"]["output_sha256"]),(537,b.ISOLATED_PROJECTION["output_sha256"]))
 def test_14_prior_pins(self):
  pins=self.report["frozen_prior_decision_artifacts"];self.assertEqual(len(pins),195)
  for r in pins:self.assertEqual(b.file_sha256(b.ROOT/r["path"]),r["sha256"])
 def test_15_projection_hash(self):self.assertEqual([hashlib.sha256(p.read_bytes()).hexdigest() for p in self.ds],[b.ISOLATED_PROJECTION["output_sha256"]]*2)
 def test_16_projection_edits(self):
  m={r.get("curation",{}).get("original_fact_id"):r for r in self.rows if r.get("curation")}
  for s in (x for x in b.SPECS if x["decision"]=="edit"):self.assertEqual((m[s["fact_id"]]["question"],m[s["fact_id"]]["answer"]),(s["question"],s["answer"]))
 def test_17_additions(self):
  expected={r["fact_id"] for p in b.PRIOR_PENDING_ADDITIONS for r in b.read_jsonl(p)};expected-={"fact-93c032484cf3a72fcc5c","fact-64a4807147c057265799"};represented={r["fact_id"] for r in self.rows}|{r.get("curation",{}).get("original_fact_id") for r in self.rows};self.assertEqual(len(expected),36);self.assertTrue(expected<=represented)
 def test_18_urls(self):
  m=json.loads(b.RESOURCE_MANIFEST.read_text());u={x for r in m["resources"] for x in (r["canonical_url"],r.get("recommendation_url")) if x};blob="\n".join(json.dumps(r) for r in self.rows);self.assertEqual((len(u),{x for x in u if x not in blob}),(24,set()))
 def test_19_unique(self):self.assertEqual((len({r["fact_id"] for r in self.rows}),len({r["question"] for r in self.rows}),self.pr["eval_fact_count"]),(537,537,612))
if __name__=="__main__":unittest.main()
