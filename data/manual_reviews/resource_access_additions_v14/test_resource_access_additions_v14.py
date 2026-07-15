#!/usr/bin/env python3
import json,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); import build_resource_access_additions_v14 as b
class ResourceAccessV14(unittest.TestCase):
 @classmethod
 def setUpClass(cls): b.main(); cls.rows=b.read_jsonl(b.OUTPUT); cls.report=json.loads(b.REPORT.read_text())
 def test_baseline_artifact(self):
  with tempfile.TemporaryDirectory(prefix=".test-resource-v14-",dir=HERE) as t:
   p=Path(t); rows=b.build_baseline(p/"base.jsonl",p/"base.report.json"); self.assertEqual((len(rows),b.file_sha256(p/"base.jsonl")),(531,b.BASELINE_SHA256))
  self.assertEqual((len(self.rows),b.file_sha256(b.OUTPUT)),(3,b.EXPECTED_OUTPUT_SHA256))
 def test_lineage(self):
  self.assertEqual((len({r['url'] for r in self.rows}),len({r['document_sha256'] for r in self.rows})),(3,3))
  for r in self.rows:
   d=json.loads((b.ROOT/r['source_lineage']['raw_document']).read_text()); self.assertEqual((r['url'],r['document_sha256']),(d['url'],d['document_sha256'])); self.assertTrue(all(x in d['text'] for x in r['evidence'].splitlines()))
 def test_quality(self):
  blob=b.OUTPUT.read_text(); self.assertEqual({x:blob.count(x) for x in ('<|im_start|>','<|im_end|>','</think>')},{'<|im_start|>':0,'<|im_end|>':0,'</think>':0})
 def test_report(self):
  self.assertEqual(self.report['new_independent_inputs']['expected_strata'],{'resources_general':2,'technique':1})
if __name__=='__main__': unittest.main()
