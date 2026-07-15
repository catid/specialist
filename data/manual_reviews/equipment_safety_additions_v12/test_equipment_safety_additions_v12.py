#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_equipment_safety_additions_v12 as b


class EquipmentSafetyAdditionsV12(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b.main()
        cls.rows = b.read_jsonl(b.OUTPUT)
        cls.report = json.loads(b.REPORT.read_text())

    def test_baseline_and_artifact(self):
        with tempfile.TemporaryDirectory(prefix=".test-equipment-safety-v12-", dir=HERE) as temp:
            p = Path(temp)
            rows = b.build_baseline(p / "baseline.jsonl", p / "baseline.report.json")
            self.assertEqual((len(rows), b.file_sha256(p / "baseline.jsonl")), (525, b.BASELINE_SHA256))
        self.assertEqual((len(self.rows), b.file_sha256(b.OUTPUT)), (3, b.EXPECTED_OUTPUT_SHA256))

    def test_distinct_sources_and_lineage(self):
        self.assertEqual((len({r["url"] for r in self.rows}), len({r["document_sha256"] for r in self.rows})), (3, 3))
        for row in self.rows:
            document = json.loads((b.ROOT / row["source_lineage"]["raw_document"]).read_text())
            self.assertEqual((row["url"], row["document_sha256"]), (document["url"], b.document_sha256(document)))
            self.assertTrue(all(line in document["text"] for line in row["evidence"].splitlines()))

    def test_quality(self):
        blob = b.OUTPUT.read_text()
        self.assertEqual({token: blob.count(token) for token in ("<|im_start|>", "<|im_end|>", "</think>")}, {"<|im_start|>": 0, "<|im_end|>": 0, "</think>": 0})
        self.assertTrue(all(row["question"].endswith("?") and row["answer"].endswith(".") for row in self.rows))

    def test_report(self):
        self.assertEqual(self.report["artifact"], {"path": b.portable(b.OUTPUT), "rows": 3, "sha256": b.EXPECTED_OUTPUT_SHA256})
        self.assertEqual(self.report["new_independent_inputs"]["expected_strata"], {"equipment_material": 1, "safety_consent": 2})


if __name__ == "__main__":
    unittest.main()
