#!/usr/bin/env python3
import json,sys,tempfile,unittest
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import build_context_merit_audit_v360 as b
class ContextMeritV360(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  b.main();cls.t=tempfile.TemporaryDirectory(prefix=".test-v360-",dir=HERE);d=Path(cls.t.name);cls.base=d/"base.jsonl";b.build_baseline(cls.base,d/"base.report.json");cls.out=d/"out.jsonl";cls.rep=d/"out.report.json";cls.ds=[];cls.rs=[]
  for _ in (1,2):b.build_projection(cls.out,cls.rep);cls.ds.append(cls.out.read_bytes());cls.rs.append(cls.rep.read_bytes())
  cls.rows=b.read_jsonl(cls.out);cls.report=json.loads(b.REPORT.read_text());cls.curations=b.read_jsonl(b.CURATION)
 @classmethod
 def tearDownClass(cls):cls.t.cleanup()
 def test_projection_actions(self):
  self.assertEqual((len(b.read_jsonl(self.base)),b.file_sha256(self.base),len(self.rows),b.file_sha256(self.out)),(531,b.BASELINE_SHA256,531,b.EXPECTED_OUTPUT_SHA256));self.assertEqual((self.ds[0],self.rs[0]),(self.ds[1],self.rs[1]));facts={r["fact_id"] for r in self.rows};self.assertTrue(all(s["fact_id"] not in facts for s in b.SPECS));edited={r.get("curation",{}).get("original_fact_id"):r for r in self.rows};self.assertEqual({k:(v["question"],v["answer"]) for k,v in edited.items() if k in {s["fact_id"] for s in b.SPECS}},{s["fact_id"]:(s["question"],s["answer"]) for s in b.SPECS})
 def test_curation_support(self):self.assertEqual((len(self.curations),Counter(r["action"] for r in self.curations)),(3,Counter({"edit":3})));self.assertTrue(all(r["evidence"] for r in self.curations))
 def test_eval_duplicate_protocol(self):
  self.assertEqual(json.loads(self.rep.read_text())["eval_fact_count"],612);n=lambda x:" ".join(x.casefold().split());self.assertEqual({k:sum(v>1 for v in Counter(n(r[k]) for r in self.rows).values()) for k in ("fact_id","question","answer")},{"fact_id":0,"question":0,"answer":0});self.assertFalse(any(x in self.out.read_text() for x in ("<|im_start|>","<|im_end|>","</think>")))
 def test_urls_capacity_policy(self):
  m=json.loads(b.RESOURCE_MANIFEST.read_text());urls={u for r in m["resources"] for u in (r["canonical_url"],r.get("recommendation_url")) if u};blob=self.out.read_text();self.assertEqual((len(urls),sum(u in blob for u in urls)),(24,24));self.assertEqual(b.conservative_capacity(self.rows),b.EXPECTED_CAPACITY_AFTER);self.assertFalse(self.report["sealed_evaluation_policy"]["manual_worker_opened_eval_or_heldout_content"]);self.assertFalse(self.report["sealed_evaluation_policy"]["manual_worker_received_eval_or_heldout_content"])
if __name__=="__main__":unittest.main()
