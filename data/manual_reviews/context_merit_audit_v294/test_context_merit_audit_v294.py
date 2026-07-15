#!/usr/bin/env python3
import json,sys,tempfile,unittest
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import build_context_merit_audit_v294 as b
class Test(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory(prefix=".test-v294-",dir=HERE);d=Path(cls.temp.name);cls.base=d/"base.jsonl";b.build_baseline(cls.base,d/"base.report.json");cls.out=d/"out.jsonl";cls.rep=d/"out.report.json";cls.datasets=[];cls.reports=[]
  for _ in (1,2):b.build_projection(cls.out,cls.rep);cls.datasets.append(cls.out.read_bytes());cls.reports.append(cls.rep.read_bytes())
  cls.rows=b.read_jsonl(cls.out);cls.add=b.read_jsonl(b.ADDITIONS);cls.audit=b.read_jsonl(b.AUDIT);cls.report=json.loads(b.REPORT.read_text())
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def test_inputs(self):self.assertEqual((len(b.read_jsonl(self.base)),b.file_sha256(self.base)),(504,b.BASELINE_SHA256));self.assertEqual((len(self.add),b.file_sha256(b.ADDITIONS)),(3,b.EXPECTED_ADDITIONS_SHA256));self.assertEqual((len({r["url"] for r in self.add}),len({r["document_sha256"] for r in self.add})),(3,3))
 def test_audit(self):self.assertEqual((len(self.audit),{r["decision"] for r in self.audit},b.read_jsonl(b.CURATION)),(3,{"add"},[]))
 def test_projection(self):self.assertEqual((len(self.rows),b.file_sha256(self.out)),(507,b.EXPECTED_OUTPUT_SHA256));self.assertTrue({r["fact_id"] for r in self.add}.issubset({r["fact_id"] for r in self.rows}))
 def test_determinism_eval(self):self.assertEqual(self.datasets[0],self.datasets[1]);self.assertEqual(self.reports[0],self.reports[1]);self.assertEqual(json.loads(self.rep.read_text())["eval_fact_count"],612);self.assertFalse(self.report["sealed_evaluation_policy"]["manual_worker_opened_eval_or_heldout_content"])
 def test_pins(self):pins=self.report["frozen_prior_decision_artifacts"];self.assertEqual(len(pins),879);self.assertTrue(all(b.file_sha256(b.ROOT/r["path"])==r["sha256"] for r in pins))
 def test_guards(self):
  norm=lambda v:" ".join(v.casefold().split());self.assertEqual({k:sum(n>1 for n in Counter(norm(r[k]) for r in self.rows).values()) for k in ("fact_id","question","answer")},{"fact_id":0,"question":0,"answer":0});blob=self.out.read_text();self.assertEqual({t:blob.count(t) for t in ("<|im_start|>","<|im_end|>","</think>")},{"<|im_start|>":0,"<|im_end|>":0,"</think>":0})
 def test_urls_capacity(self):
  m=json.loads(b.RESOURCE_MANIFEST.read_text());blob=self.out.read_text();urls={u for r in m["resources"] for u in (r["canonical_url"],r.get("recommendation_url")) if u};self.assertEqual((len(urls),sum(u in blob for u in urls)),(24,24));self.assertEqual(b.conservative_capacity(b.read_jsonl(self.base)),b.EXPECTED_CAPACITY["before"]);self.assertEqual(b.conservative_capacity(self.rows),b.EXPECTED_CAPACITY["after"]);self.assertEqual(self.report["conservative_capacity"]["delta"],{"conflict_units":3,"equipment_material":0,"resources_general":0,"safety_consent":0,"technique":3})
 def test_report(self):self.assertEqual((self.report["schema"],self.report["isolated_build_projection"]["output_sha256"]),("context-merit-audit-report-v294",b.EXPECTED_OUTPUT_SHA256))
if __name__=="__main__":unittest.main()
