#!/usr/bin/env python3
"""Regression tests for the first distinct-document technique/equipment additions."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_technique_equipment_additions as b
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import parse_qa, stable_fact_id


class TechniqueEquipmentAdditionsV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        b.main()
        cls.rows = b.read_jsonl(b.OUTPUT)
        cls.report = json.loads(b.REPORT.read_text())

    def test_01_identity(self) -> None:
        self.assertEqual((len(self.rows), b.file_sha256(b.OUTPUT)), (3, b.EXPECTED_OUTPUT_SHA256))

    def test_02_sources_are_distinct_and_pinned(self) -> None:
        self.assertEqual(len({row["document_sha256"] for row in self.rows}), 3)
        self.assertEqual(len({row["url"] for row in self.rows}), 3)
        for row in self.rows:
            path = b.ROOT / row["source_lineage"]["raw_document"]
            document = json.loads(path.read_text())
            self.assertEqual((document["url"], document["document_sha256"]), (row["url"], row["document_sha256"]))
            self.assertTrue(all(line in document["text"] for line in row["evidence"].splitlines()))

    def test_03_canonical_qa(self) -> None:
        for row in self.rows:
            self.assertEqual(parse_qa(row["text"]), (row["question"], row["answer"]))
            self.assertEqual(row["fact_id"], stable_fact_id(row["question"], row["answer"]))

    def test_04_expected_strata(self) -> None:
        self.assertEqual(Counter(classify_stratum(row) for row in self.rows), Counter({"equipment_material": 2, "technique": 1}))

    def test_05_train_only_novelty(self) -> None:
        with tempfile.TemporaryDirectory(prefix="technique-equipment-test-", dir=HERE) as temp:
            baseline = Path(temp) / "v289.jsonl"
            report = Path(temp) / "v289.report.json"
            base_rows = b.build_baseline(baseline, report)
        self.assertTrue({row["document_sha256"] for row in self.rows}.isdisjoint({row["document_sha256"] for row in base_rows}))
        self.assertTrue({row["url"] for row in self.rows}.isdisjoint({row["url"] for row in base_rows}))

    def test_06_report(self) -> None:
        self.assertEqual(self.report["schema"], "manual-technique-equipment-additions-report-v1")
        self.assertEqual(self.report["new_independent_inputs"]["expected_strata"], {"equipment_material": 2, "technique": 1})


if __name__ == "__main__":
    unittest.main()
