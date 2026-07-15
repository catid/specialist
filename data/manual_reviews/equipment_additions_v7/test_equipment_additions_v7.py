#!/usr/bin/env python3
import json,sys,unittest
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import build_equipment_additions_v7 as b
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import parse_qa,stable_fact_id
class Test(unittest.TestCase):
 @classmethod
 def setUpClass(cls):b.main();cls.rows=b.read_jsonl(b.OUTPUT);cls.report=json.loads(b.REPORT.read_text())
 def test_identity(self):self.assertEqual((len(self.rows),b.file_sha256(b.OUTPUT)),(3,b.EXPECTED_OUTPUT_SHA256))
 def test_sources(self):
  self.assertEqual((len({r["url"]for r in self.rows}),len({r["document_sha256"]for r in self.rows})),(3,3))
  for r in self.rows:
   d=json.loads((b.ROOT/r["source_lineage"]["raw_document"]).read_text());self.assertEqual((d["url"],d["document_sha256"]),(r["url"],r["document_sha256"]));self.assertTrue(all(line in d["text"]for line in r["evidence"].splitlines()))
 def test_content(self):
  for r in self.rows:self.assertEqual(parse_qa(r["text"]),(r["question"],r["answer"]));self.assertEqual(r["fact_id"],stable_fact_id(r["question"],r["answer"]))
  self.assertEqual(Counter(classify_stratum(r)for r in self.rows),Counter({"equipment_material":3}))
 def test_report(self):self.assertEqual(self.report["new_independent_inputs"]["expected_strata"],{"equipment_material":3})
if __name__=="__main__":unittest.main()
