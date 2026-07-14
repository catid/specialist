#!/usr/bin/env python3
"""Focused regression tests for Rope365 rope-care audit v80."""
import json,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import build_context_merit_audit_v80 as b
class V80(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory(prefix=".test-v80-",dir=HERE);o=Path(cls.temp.name);cls.base=o/"base.jsonl";cls.br=o/"base.report.json";b.build_projection(cls.base,cls.br,b.PRIOR_PROJECTION_CURATIONS);cls.baseline=b.read_jsonl(cls.base);cls.ds=o/"projection.jsonl";cls.rp=o/"projection.report.json";cls.db=[];cls.rb=[]
  for _ in (1,2):b.build_projection(cls.ds,cls.rp,b.OUTPUT_PROJECTION_CURATIONS);cls.db.append(cls.ds.read_bytes());cls.rb.append(cls.rp.read_bytes())
  cls.rows=b.read_jsonl(cls.ds);cls.pr=json.loads(cls.rp.read_text());cls.audit=b.read_jsonl(b.AUDIT);cls.curation=b.read_jsonl(b.CURATION);cls.report=json.loads(b.REPORT.read_text())
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def test_01_baseline(self):self.assertEqual((len(self.baseline),b.file_sha256(self.base)),(536,b.PROJECTED_SELECTION_BASELINE["sha256"]))
 def test_02_eval(self):self.assertEqual(json.loads(self.br.read_text())["eval_fact_count"],612)
 def test_03_selection(self):
  by={r["fact_id"]:i for i,r in enumerate(self.baseline,1)};self.assertEqual({f:by[f] for f in b.EXPECTED_SELECTION},b.PROJECTED_ACTIVE_INDICES)
 def test_04_direct(self):self.assertEqual(sum(not r.get("curation") for r in self.baseline),126)
 def test_05_decisions(self):self.assertEqual({x:sum(r["decision"]==x for r in self.audit) for x in ("keep","edit","drop")},{"keep":2,"edit":1,"drop":0})
 def test_06_curation(self):self.assertEqual((len(self.curation),{r["action"] for r in self.curation}),(1,{"edit"}))
 def test_07_support(self):self.assertEqual({r["support_type"] for r in self.curation},{"manual_paraphrase"});self.assertTrue(self.curation[0]["paraphrase_rationale"])
 def test_08_source(self):
  for r in self.audit:self.assertEqual(r["source_document_file_sha256"],b.file_sha256(b.SOURCE));self.assertEqual(b.text_sha256(r["support_evidence"]),r["support_evidence_sha256"])
 def test_09_keeps(self):
  projected={r["fact_id"]:r for r in self.rows}
  for r in self.audit:
   if r["decision"]=="keep":self.assertEqual((projected[r["fact_id"]]["question"],projected[r["fact_id"]]["answer"]),(r["active_question"],r["active_answer"]))
 def test_10_lineage(self):
  for r in self.audit:self.assertEqual((r["review_pass"],r["projection_lineage"]["baseline_sha256"]),("rope365_rope_care_reaudit",b.PROJECTED_SELECTION_BASELINE["sha256"]))
 def test_11_projection(self):self.assertEqual((len(self.rows),b.file_sha256(self.ds)),(536,b.EXPECTED_OUTPUT_SHA256))
 def test_12_determinism(self):self.assertEqual((self.db[0],self.rb[0]),(self.db[1],self.rb[1]))
 def test_13_policy(self):
  p=self.report["sealed_evaluation_policy"];self.assertEqual((p["manual_worker_opened_eval_or_heldout_content"],p["manual_worker_received_eval_or_heldout_content"],p["automated_collision_tool_reads_sealed_content"],self.report["isolated_build_projection"]["repeat_projection_report_byte_identical"]),(False,False,True,True));self.assertNotIn("generator_opens_eval_or_heldout_content",p)
 def test_14_pins(self):
  pins=self.report["frozen_prior_decision_artifacts"];self.assertEqual(len(pins),237)
  for r in pins:self.assertEqual(b.file_sha256(b.ROOT/r["path"]),r["sha256"])
 def test_15_action(self):
  s=next(s for s in b.SPECS if s["decision"]=="edit");r=next(r for r in self.rows if r.get("curation",{}).get("original_fact_id")==s["fact_id"]);self.assertEqual((r["question"],r["answer"]),(s["question"],s["answer"]))
 def test_16_additions(self):
  expected={r["fact_id"] for p in b.PRIOR_PENDING_ADDITIONS for r in b.read_jsonl(p)};expected-={"fact-93c032484cf3a72fcc5c","fact-64a4807147c057265799"};represented={r["fact_id"] for r in self.rows}|{r.get("curation",{}).get("original_fact_id") for r in self.rows};self.assertEqual(len(expected),36);self.assertTrue(expected<=represented)
 def test_17_urls(self):
  m=json.loads(b.RESOURCE_MANIFEST.read_text());u={x for r in m["resources"] for x in (r["canonical_url"],r.get("recommendation_url")) if x};blob="\n".join(json.dumps(r) for r in self.rows);self.assertEqual((len(u),{x for x in u if x not in blob}),(24,set()))
 def test_18_unique(self):self.assertEqual((len({r["fact_id"] for r in self.rows}),len({r["question"] for r in self.rows}),self.pr["eval_fact_count"]),(536,536,612))
 def test_19_report(self):self.assertEqual((self.report["schema"],self.report["audit"]["by_decision"],self.report["isolated_build_projection"]["output_sha256"]),("context-merit-audit-report-v80",{"edit":1,"keep":2},b.EXPECTED_OUTPUT_SHA256))
if __name__=="__main__":unittest.main()
