#!/usr/bin/env python3
import json,sys,tempfile,unittest
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import build_tethered_policy_additions_v21 as b
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact,has_protocol_tokens,leakage_reason,normalize_text,parse_qa
class TetheredPolicyAdditionsV21(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  b.main();cls.rows=b.read_jsonl(b.OUTPUT)
  with tempfile.TemporaryDirectory(prefix=".test-tethered-policy-v21-",dir=HERE) as t:d=Path(t);cls.baseline=b.build_baseline(d/"v309.jsonl",d/"v309.report.json")
 def test_artifact_strata_temporal(self):self.assertEqual((len(self.rows),b.file_sha256(b.OUTPUT)),(3,b.EXPECTED_OUTPUT_SHA256));self.assertEqual(Counter(classify_stratum(r) for r in self.rows),Counter({"safety_consent":2,"resources_general":1}));self.assertTrue(all(r["policy_year"]==2025 and "2025" in r["question"] for r in self.rows))
 def test_evidence_collisions_protocol(self):
  facts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train") for r in self.baseline];qs={normalize_text(r["question"]) for r in self.baseline};pairs={(normalize_text(r["question"]),normalize_text(r["answer"])) for r in self.baseline};d=json.loads(b.SOURCE["path"].read_text())
  for r in self.rows:self.assertTrue(all(x in d["text"] for x in r["evidence"].splitlines()));self.assertNotIn(normalize_text(r["question"]),qs);self.assertNotIn((normalize_text(r["question"]),normalize_text(r["answer"])),pairs);self.assertIsNone(leakage_reason(r["question"],r["answer"],facts));self.assertEqual(parse_qa(r["text"]),(r["question"],r["answer"]));self.assertFalse(has_protocol_tokens(r["text"]))
if __name__=="__main__":unittest.main()
