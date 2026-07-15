#!/usr/bin/env python3
import json, sys, tempfile, unittest
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import build_structure_psychology_additions_v19 as b
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact, has_protocol_tokens, leakage_reason, normalize_text, parse_qa

class StructurePsychologyAdditionsV19(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b.main(); cls.rows=b.read_jsonl(b.OUTPUT); cls.report=json.loads(b.REPORT.read_text())
        with tempfile.TemporaryDirectory(prefix=".test-structure-psychology-v19-",dir=HERE) as t:
            d=Path(t); cls.baseline=b.build_baseline(d/"v307.jsonl",d/"v307.report.json")
    def test_artifact_strata(self):
        self.assertEqual((len(self.rows),b.file_sha256(b.OUTPUT)),(3,b.EXPECTED_OUTPUT_SHA256))
        self.assertEqual(Counter(classify_stratum(r) for r in self.rows),Counter({"safety_consent":2,"technique":1}))
    def test_lineage_evidence_collisions_protocol(self):
        facts=[EvalFact(r["question"],r["answer"],r["fact_id"],"train") for r in self.baseline]
        qs={normalize_text(r["question"]) for r in self.baseline}; pairs={(normalize_text(r["question"]),normalize_text(r["answer"])) for r in self.baseline}
        for r in self.rows:
            raw=b.ROOT/r["source_lineage"]["raw_document"]; d=json.loads(raw.read_text())
            self.assertEqual((r["url"],r["document_sha256"]),(d["url"],d["document_sha256"]))
            self.assertTrue(all(x in d["text"] for x in r["evidence"].splitlines()))
            self.assertNotIn(normalize_text(r["question"]),qs); self.assertNotIn((normalize_text(r["question"]),normalize_text(r["answer"])),pairs)
            self.assertIsNone(leakage_reason(r["question"],r["answer"],facts)); self.assertEqual(parse_qa(r["text"]),(r["question"],r["answer"])); self.assertFalse(has_protocol_tokens(r["text"]))
    def test_exclusions(self): self.assertEqual(Counter(x["decision"] for x in self.report["excluded_source"]),Counter({"partial_use":2,"reject":1}))
if __name__=="__main__": unittest.main()
