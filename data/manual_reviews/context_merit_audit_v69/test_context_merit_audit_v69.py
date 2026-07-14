#!/usr/bin/env python3
"""Focused regression tests for audit-fidelity repair v69."""
import json,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import build_context_merit_audit_v69 as b
class V69(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory(prefix=".test-v69-",dir=HERE);o=Path(cls.temp.name);cls.base=o/"base.jsonl";cls.br=o/"base.report.json";b.build_projection(cls.base,cls.br,b.PRIOR_PROJECTION_CURATIONS);cls.baseline=b.read_jsonl(cls.base);cls.ds=o/"projection.jsonl";cls.rp=o/"projection.report.json";cls.dataset_bytes=[];cls.report_bytes=[]
  for n in (1,2):
   b.build_projection(cls.ds,cls.rp,b.OUTPUT_PROJECTION_CURATIONS);cls.dataset_bytes.append(cls.ds.read_bytes());cls.report_bytes.append(cls.rp.read_bytes())
  cls.rows=b.read_jsonl(cls.ds);cls.audit=b.read_jsonl(b.AUDIT);cls.curation=b.read_jsonl(b.CURATION);cls.report=json.loads(b.REPORT.read_text());cls.pr=json.loads(cls.rp.read_text())
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def test_01_baseline(self):self.assertEqual((len(self.baseline),b.file_sha256(self.base)),(537,b.PROJECTED_SELECTION_BASELINE["sha256"]))
 def test_02_eval_tooling_count(self):self.assertEqual(json.loads(self.br.read_text())["eval_fact_count"],612)
 def test_03_audit(self):self.assertEqual((len(self.audit),self.audit[0]["fact_id"],self.audit[0]["decision"]),(1,b.ORIGINAL_FACT_ID,"edit"))
 def test_04_corrected_question(self):
  rows=[r for r in self.rows if r.get("curation",{}).get("original_fact_id")==b.ORIGINAL_FACT_ID];self.assertEqual([(r["question"],r["answer"]) for r in rows],[(b.CORRECTED_QUESTION,b.ANSWER)])
 def test_05_malformed_absent(self):self.assertNotIn(b.MALFORMED_QUESTION,{r["question"] for r in self.rows})
 def test_06_curation_supersession(self):self.assertEqual(({r["fact_id"] for r in self.curation},len(self.curation)),({b.ORIGINAL_FACT_ID,"fact-069a861dbb2bea9e47ca","fact-f7e802bf0b2759290dc6"},3))
 def test_07_carried_v65_semantics(self):
  old={r["fact_id"]:r for r in b.read_jsonl(b.V65_CURATION)};new={r["fact_id"]:r for r in self.curation}
  for f in ("fact-069a861dbb2bea9e47ca","fact-f7e802bf0b2759290dc6"):self.assertEqual((new[f]["question"],new[f]["answer"]),(old[f]["question"],old[f]["answer"]))
 def test_08_source(self):self.assertEqual(b.file_sha256(b.SOURCE),self.audit[0]["source_document_file_sha256"])
 def test_09_evidence(self):self.assertEqual(b.text_sha256(self.audit[0]["support_evidence"]),self.audit[0]["support_evidence_sha256"])
 def test_10_output(self):self.assertEqual((len(self.rows),b.file_sha256(self.ds)),(537,b.EXPECTED_OUTPUT_SHA256))
 def test_11_dataset_determinism(self):self.assertEqual(self.dataset_bytes[0],self.dataset_bytes[1])
 def test_12_report_determinism(self):self.assertEqual(self.report_bytes[0],self.report_bytes[1])
 def test_13_report_observations(self):
  normalized=dict(self.pr);normalized["output"]="<projection-output>";digest=b.sha_bytes((json.dumps(normalized,indent=2,sort_keys=True)+"\n").encode());self.assertEqual((self.report["isolated_build_projection"]["repeat_dataset_byte_identical"],self.report["isolated_build_projection"]["repeat_projection_report_byte_identical"],self.report["isolated_build_projection"]["projection_report_normalized_sha256"]),(True,True,digest))
 def test_14_sealed_policy(self):
  p=self.report["sealed_evaluation_policy"];self.assertEqual((p["manual_worker_opened_eval_or_heldout_content"],p["manual_worker_received_eval_or_heldout_content"],p["automated_collision_tool_reads_sealed_content"]),(False,False,True));self.assertNotIn("generator_opens_eval_or_heldout_content",p)
 def test_15_prior_pins(self):
  pins=self.report["frozen_prior_decision_artifacts"];self.assertEqual(len(pins),204)
  for r in pins:self.assertEqual(b.file_sha256(b.ROOT/r["path"]),r["sha256"])
 def test_16_direct_count(self):self.assertEqual(sum(not r.get("curation") for r in self.rows),150)
 def test_17_additions(self):
  expected={r["fact_id"] for p in b.PRIOR_PENDING_ADDITIONS for r in b.read_jsonl(p)};expected-={"fact-93c032484cf3a72fcc5c","fact-64a4807147c057265799"};represented={r["fact_id"] for r in self.rows}|{r.get("curation",{}).get("original_fact_id") for r in self.rows};self.assertEqual(len(expected),36);self.assertTrue(expected<=represented)
 def test_18_urls(self):
  m=json.loads(b.RESOURCE_MANIFEST.read_text());u={x for r in m["resources"] for x in (r["canonical_url"],r.get("recommendation_url")) if x};blob="\n".join(json.dumps(r) for r in self.rows);self.assertEqual((len(u),{x for x in u if x not in blob}),(24,set()))
 def test_19_unique(self):self.assertEqual((len({r["fact_id"] for r in self.rows}),len({r["question"] for r in self.rows}),self.pr["eval_fact_count"]),(537,537,612))
if __name__=="__main__":unittest.main()
