#!/usr/bin/env python3
"""Regression tests for the second distinct-document technique tranche."""
from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_technique_additions_v2 as b
from eggroll_es_train_panel_sampler_v13 import classify_stratum
from qa_quality import parse_qa, stable_fact_id


class TechniqueAdditionsV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        b.main()
        cls.rows = b.read_jsonl(b.OUTPUT)
        cls.report = json.loads(b.REPORT.read_text())

    def test_01_identity(self) -> None:
        self.assertEqual((len(self.rows), b.file_sha256(b.OUTPUT)), (3, b.EXPECTED_OUTPUT_SHA256))

    def test_02_distinct_pinned_sources(self) -> None:
        self.assertEqual((len({r["document_sha256"] for r in self.rows}), len({r["url"] for r in self.rows})), (3, 3))
        for row in self.rows:
            document = json.loads((b.ROOT / row["source_lineage"]["raw_document"]).read_text())
            self.assertEqual((document["url"], document["document_sha256"]), (row["url"], row["document_sha256"]))
            self.assertTrue(all(line in document["text"] for line in row["evidence"].splitlines()))

    def test_03_canonical_qa(self) -> None:
        for row in self.rows:
            self.assertEqual(parse_qa(row["text"]), (row["question"], row["answer"]))
            self.assertEqual(row["fact_id"], stable_fact_id(row["question"], row["answer"]))

    def test_04_strata(self) -> None:
        self.assertEqual(Counter(classify_stratum(row) for row in self.rows), Counter({"technique": 3}))

    def test_05_report(self) -> None:
        self.assertEqual(self.report["new_independent_inputs"], {"document_sha256s": 3, "expected_strata": {"technique": 3}, "urls": 3})


if __name__ == "__main__":
    unittest.main()
