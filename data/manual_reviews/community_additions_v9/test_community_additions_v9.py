#!/usr/bin/env python3
import json,sys,unittest
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import build_community_additions_v9 as b
from eggroll_es_train_panel_sampler_v13 import classify_stratum
class Test(unittest.TestCase):
 @classmethod
 def setUpClass(cls):b.main();cls.rows=b.read(b.OUTPUT);cls.report=json.loads(b.REPORT.read_text())
 def test_identity(self):self.assertEqual((len(self.rows),b.sha(b.OUTPUT)),(3,b.EXPECTED_OUTPUT_SHA256))
 def test_sources(self):self.assertEqual((len({r["url"]for r in self.rows}),len({r["document_sha256"]for r in self.rows})),(3,3));self.assertTrue(all(all(line in json.loads((b.ROOT/r["source_lineage"]["raw_document"]).read_text())["text"]for line in r["evidence"].splitlines())for r in self.rows))
 def test_strata(self):self.assertEqual(Counter(classify_stratum(r)for r in self.rows),Counter({"resources_general":2,"safety_consent":1}))
 def test_report(self):self.assertEqual(self.report["new_independent_inputs"]["expected_strata"],{"resources_general":2,"safety_consent":1})
if __name__=="__main__":unittest.main()
