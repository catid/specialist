#!/usr/bin/env python3
import json,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); import build_multiperson_chest_additions_v15 as b
class AdditionsV15(unittest.TestCase):
 @classmethod
 def setUpClass(cls): b.main(); cls.rows=b.read_jsonl(b.OUTPUT); cls.report=json.loads(b.REPORT.read_text())
 def test_identity(self):
  with tempfile.TemporaryDirectory(prefix=".test-v15-",dir=HERE) as t:
   p=Path(t); rows=b.build_baseline(p/"b.jsonl",p/"b.report.json"); self.assertEqual((len(rows),b.file_sha256(p/"b.jsonl")),(534,b.BASELINE_SHA256))
  self.assertEqual((len(self.rows),b.file_sha256(b.OUTPUT)),(2,b.EXPECTED_OUTPUT_SHA256))
 def test_lineage(self):
  self.assertEqual((len({r['url'] for r in self.rows}),len({r['document_sha256'] for r in self.rows})),(2,2))
  for r in self.rows:
   d=json.loads((b.ROOT/r['source_lineage']['raw_document']).read_text()); self.assertEqual((r['url'],r['document_sha256']),(d['url'],d['document_sha256'])); self.assertTrue(all(x in d['text'] for x in r['evidence'].splitlines()))
 def test_quality_and_rejection(self):
  blob=b.OUTPUT.read_text(); self.assertFalse(any(x in blob for x in ('<|im_start|>','<|im_end|>','</think>'))); self.assertEqual(self.report['new_independent_inputs']['expected_strata'],{'safety_consent':1,'technique':1}); self.assertEqual(self.report['excluded_third_candidate']['decision'],'reject')
if __name__=='__main__': unittest.main()
