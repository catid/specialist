#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_hogtie_mobility_safety_additions_v18 as b
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import EvalFact, has_protocol_tokens, leakage_reason, normalize_text, parse_qa


class HogtieMobilitySafetyAdditionsV18(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b.main()
        cls.rows = b.read_jsonl(b.OUTPUT)
        cls.report = json.loads(b.REPORT.read_text())
        with tempfile.TemporaryDirectory(prefix=".test-hogtie-mobility-safety-v18-", dir=HERE) as temp:
            d = Path(temp)
            cls.baseline = b.build_baseline(d / "v306.jsonl", d / "v306.report.json")

    def test_artifact_and_strata(self):
        self.assertEqual((len(self.rows), b.file_sha256(b.OUTPUT)), (3, b.EXPECTED_OUTPUT_SHA256))
        self.assertEqual(Counter(classify_stratum(r) for r in self.rows), Counter({"safety_consent": 2, "technique": 1}))

    def test_lineage_evidence_and_collisions(self):
        facts = [EvalFact(r["question"], r["answer"], r["fact_id"], "train") for r in self.baseline]
        questions = {normalize_text(r["question"]) for r in self.baseline}
        pairs = {(normalize_text(r["question"]), normalize_text(r["answer"])) for r in self.baseline}
        for row in self.rows:
            raw = b.ROOT / row["source_lineage"]["raw_document"]
            document = json.loads(raw.read_text())
            self.assertEqual((row["url"], row["document_sha256"]), (document["url"], document["document_sha256"]))
            self.assertTrue(all(part in document["text"] for part in row["evidence"].splitlines()))
            self.assertNotIn(normalize_text(row["question"]), questions)
            self.assertNotIn((normalize_text(row["question"]), normalize_text(row["answer"])), pairs)
            self.assertIsNone(leakage_reason(row["question"], row["answer"], facts))
            self.assertEqual(parse_qa(row["text"]), (row["question"], row["answer"]))
            self.assertFalse(has_protocol_tokens(row["text"]))

    def test_rejections(self):
        self.assertEqual({x["decision"] for x in self.report["excluded_source"]}, {"reject"})
        self.assertEqual(len(self.report["excluded_source"]), 2)


if __name__ == "__main__":
    unittest.main()
