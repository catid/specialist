#!/usr/bin/env python3
"""Identity-only checks for the S6 validation-document guard."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from build_eval_v3 import normalize_source_url


LEDGER = HERE / "pending_curation_s6_eval_disjointness_v1.jsonl"
REPORT = HERE / "report_s6_eval_disjointness_guard_v1.json"
ADDITIONS = (
    ROOT / "data/manual_reviews/resource_safety_additions_v2/"
    "pending_additions_safety_care_tranche_02_v1.jsonl"
)
DOMAIN = ROOT / "data/eval_qa_v3.jsonl"
OOD_QA = ROOT / "data/ood_qa_v3.jsonl"
OOD_PROSE = ROOT / "data/ood_prose_v3.jsonl"
FACT_ID = "fact-94b42b931282e211349e"
NORMALIZED_URL = "web://rope365.com/storing-rope"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


class S6EvalDisjointnessGuardTest(unittest.TestCase):
    def test_guard_matches_the_reviewed_addition_exactly(self) -> None:
        decision, = read_jsonl(LEDGER)
        source = next(row for row in read_jsonl(ADDITIONS)
                      if row["fact_id"] == FACT_ID)
        self.assertEqual(decision["action"], "drop")
        self.assertEqual(decision["expected_question"], source["question"])
        self.assertEqual(decision["expected_answer"], source["answer"])
        self.assertEqual(decision["document_sha256"],
                         source["document_sha256"])
        self.assertEqual(decision["evidence_url"], source["evidence_url"])
        self.assertEqual(decision["reason_code"],
                         "validation_document_overlap")

    def test_collision_is_validation_only(self) -> None:
        # Inspect only split and normalized URL identity; never question or
        # answer content from the frozen domain evaluation artifact.
        matching_splits = {
            row["split"] for row in read_jsonl(DOMAIN)
            if normalize_source_url(row["url"]) == NORMALIZED_URL
        }
        self.assertEqual(matching_splits, {"validation"})
        self.assertNotIn("heldout", matching_splits)

    def test_frozen_evaluation_hashes_are_pinned(self) -> None:
        self.assertEqual(sha256(DOMAIN),
                         "ab9a391e249910e876826dfab9c8e2f8e17a7b8695e6f018a3e515e5aa69603b")
        self.assertEqual(sha256(OOD_QA),
                         "25a48b9494134731e51043047afadb340291a9ae3e9cfec9d9cfd8c73ddb255d")
        self.assertEqual(sha256(OOD_PROSE),
                         "3299457c7a23dfb0eb10408b2226b6231e291b519a52325feed607d901605e57")

    def test_report_pins_guarded_and_unguarded_projections(self) -> None:
        report = json.loads(REPORT.read_text())
        self.assertEqual(report["schema"],
                         "s6-eval-disjointness-guard-report-v1")
        self.assertFalse(report["collision"]["heldout_document_collision"])
        self.assertTrue(report["collision"]["validation_document_collision"])
        self.assertEqual(report["unguarded_projection"]["output_rows"], 795)
        guarded = report["guarded_projection"]
        self.assertEqual(guarded["output_rows"], 794)
        self.assertEqual(
            guarded["output_sha256"],
            "f7127c38c7b540eaf9cf4349d1a1b8076e171da7f8ea43c11068ad1c311bb776",
        )
        self.assertTrue(guarded["candidate_eval_artifacts_byte_identical_to_frozen_v3"])
        self.assertTrue(guarded["disjointness_passed"])
        self.assertTrue(guarded["repeat_build_byte_identical"])
        self.assertEqual(guarded["validated_builds"], 2)

    def test_report_hashes_match_tracked_inputs(self) -> None:
        report = json.loads(REPORT.read_text())
        self.assertEqual(report["curation"]["sha256"], sha256(LEDGER))
        self.assertEqual(report["source_addition"]["sha256"],
                         sha256(ADDITIONS))


if __name__ == "__main__":
    unittest.main()
