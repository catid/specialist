from __future__ import annotations

import hashlib
import json
import math
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = read_json(ROOT / "manifest.json")
        cls.provenance = read_json(ROOT / "source_snapshot/provenance.json")
        cls.dispositions = read_jsonl(ROOT / "dispositions.jsonl")
        cls.corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        cls.report = (ROOT / "REPORT.md").read_text(encoding="utf-8")

    def test_manifest_declares_direct_non_qa_training_artifact(self) -> None:
        self.assertEqual(self.manifest["schema"], "site-corpus-manifest-v1")
        self.assertEqual(
            self.manifest["resource_id"], "usfs_rigging_for_trail_work"
        )
        self.assertEqual(self.manifest["artifact_role"], "direct_training_markdown")
        self.assertTrue(self.manifest["direct_training_ready"])
        self.assertTrue(self.manifest["non_qa"])
        self.assertEqual(self.manifest["training_document_count"], 1)
        self.assertGreaterEqual(self.manifest["direct_training_word_count"], 6000)

    def test_source_identity_and_remote_pdf_digest_are_exact(self) -> None:
        publication = self.provenance["publication"]
        self.assertEqual(
            publication["title"],
            "Rigging for Trail Work: Principles, Techniques, and Lessons from the Backcountry",
        )
        self.assertEqual(
            publication["authors"],
            ["David E. Michael", "Jedediah J. Talbot", "John S. Glenn"],
        )
        self.assertEqual(publication["publication_date_as_displayed"], "August 2024")
        self.assertEqual(publication["publication_number"], "2223-2806-NTDP")
        pdf = self.provenance["remote_objects"]["pdf"]
        self.assertEqual(pdf["content_length"], 22_614_422)
        self.assertEqual(
            pdf["sha256"],
            "72128d121adeac0bc668f0acd0f9b09cdb134ed997601b2d792ac3283da33d74",
        )
        self.assertFalse(pdf["retained_in_repository"])
        self.assertFalse(list(ROOT.rglob("*.pdf")))

    def test_snapshot_hashes_and_newline_normalization_are_explicit(self) -> None:
        remote = self.provenance["remote_objects"]
        expected_remote = {
            "mods_xml": (
                9001,
                "3f33bd253963358992746af9bb8394bd400fa1003ecb795895b2368f6987261b",
            ),
            "robots_txt": (
                4695,
                "628d45022bceabd6fb36b85c8b63425f01baba5cef77eb982469e763a074f66a",
            ),
        }
        for key, (length, digest) in expected_remote.items():
            item = remote[key]
            self.assertEqual(item["remote_byte_length"], length)
            self.assertEqual(item["remote_sha256"], digest)
            path = ROOT / item["retained_path"]
            self.assertEqual(item["retained_byte_length"], path.stat().st_size)
            self.assertEqual(item["retained_sha256"], sha256(path))
            self.assertEqual(item["retained_byte_length"], length + 1)
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            self.assertIn("terminal LF", item["normalization"])
        self.assertIn(
            "no rule disallowing",
            remote["robots_txt"]["access_interpretation"],
        )
        self.assertIn(
            "not a copyright license",
            remote["robots_txt"]["access_interpretation"],
        )

    def test_output_hashes_and_lengths_are_exact(self) -> None:
        for item in self.manifest["outputs"].values():
            path = ROOT / item["path"]
            self.assertEqual(item["byte_length"], path.stat().st_size)
            self.assertEqual(item["sha256"], sha256(path))

    def test_rights_scope_excludes_noncleared_expressive_components(self) -> None:
        rights = self.provenance["rights_review"]
        self.assertEqual(
            rights["decision"],
            "include_manually_transformed_eligible_federal_text_with_component_exclusions",
        )
        self.assertFalse(rights["legal_opinion"])
        for expected in (
            "all photographs",
            "all illustrations and figure artwork",
            "expressive source table layouts",
            "decorative quotations",
            "vendor descriptions and product promotion",
        ):
            self.assertIn(expected, rights["excluded_components"])
        self.assertNotIn("![", self.corpus)
        self.assertNotRegex(self.corpus.lower(), r"<img\b")
        image_disposition = next(
            row
            for row in self.dispositions
            if row["record_id"] == "usfs-rigging-023"
        )
        self.assertFalse(image_disposition["direct_training_included"])
        self.assertEqual(image_disposition["disposition"], "excluded_rights_caution")

    def test_prominent_nontransfer_warning_is_complete(self) -> None:
        opening = self.corpus[:2600]
        for phrase in (
            "It is not a human-suspension manual.",
            "It does not certify bondage rope",
            "an indoor hardpoint",
            "No numerical value below is a design value for suspending a person.",
            "Do not infer indoor-hardpoint adequacy or tree adequacy",
            "Manufacturer instructions and a qualified evaluation control",
        ):
            self.assertIn(phrase, opening)
        self.assertEqual(
            self.manifest["human_suspension_transfer_status"],
            "mechanics_only_no_certification_or_design_values",
        )
        forbidden = self.provenance["domain_transfer"]["forbidden_inferences"]
        self.assertIn("human-suspension certification", forbidden)
        self.assertIn("tree or limb certification", forbidden)
        self.assertIn("indoor hardpoint or structural certification", forbidden)

    def test_no_unsafe_certification_or_universalized_source_heuristics(self) -> None:
        lower = self.corpus.lower()
        forbidden_claims = (
            "safe for human suspension",
            "approved for human suspension",
            "this tree is safe",
            "this ceiling is safe",
            "a 10:1 factor makes",
            "five percent of fibers",
            "never saddle a dead horse",
            "bolt torque (foot-pounds)",
            "wire rope clips are about 80-percent efficient",
        )
        for phrase in forbidden_claims:
            self.assertNotIn(phrase, lower)
        for required_caution in (
            "not a universal rule",
            "heuristics, not cross-domain specifications",
            "not a generic derating formula",
            "field procedures are deliberately not reproduced",
            "source supplies no acceptance method",
        ):
            self.assertIn(required_caution, lower)

    def test_non_qa_surface_and_no_url_or_title_trivia(self) -> None:
        for marker in (
            "<|im_start|>",
            "<|im_end|>",
            "</think>",
            "Question:",
            "Answer:",
            "Which canonical",
            "What is the URL",
            "http://",
            "https://",
            "govinfo.gov",
        ):
            self.assertNotIn(marker, self.corpus)
        self.assertNotRegex(self.corpus, r"\bQ\s*:\s")
        self.assertNotRegex(self.corpus, r"\bA\s*:\s")
        self.assertIn("rather than taught as a fact", self.corpus)
        self.assertIn("non-Q&A", self.corpus)

    def test_manual_page_citation_density_and_map(self) -> None:
        citations = self.corpus.count("(USDA manual,")
        self.assertGreaterEqual(citations, 45)
        for page_marker in (
            "| 11–20 |",
            "| 37–56 |",
            "| 71–77 |",
            "| 123–124 |",
            "| 127–168 |",
        ):
            self.assertIn(page_marker, self.corpus)
        self.assertIn(
            "references and vendor material excluded",
            self.corpus,
        )
        self.assertIn(
            "wire-rope termination appendix excluded",
            self.corpus,
        )

    def test_symmetric_two_leg_table_recomputes(self) -> None:
        expected = {
            90: 500,
            75: 518,
            60: 577,
            45: 707,
            30: 1000,
            15: 1932,
            5: 5737,
        }
        for angle, shown in expected.items():
            calculated = round(1000 / (2 * math.sin(math.radians(angle))))
            self.assertEqual(calculated, shown)
            formatted = f"{shown:,}" if shown >= 1000 else str(shown)
            self.assertIn(f"| {angle}° | {formatted} lb |", self.corpus)
        self.assertIn(
            "`T = (W / n) × csc(θ) = W / (n × sin(θ))`",
            self.corpus,
        )
        self.assertIn("Angle of each leg above horizontal", self.corpus)
        self.assertNotIn("Angle of each leg from vertical", self.corpus)
        self.assertIn(
            "`line tension on either side = W / (2 × sin(θ))`",
            self.corpus,
        )
        self.assertIn("not a capacity table", self.corpus)

    def test_source_errata_are_preserved_not_silently_fixed(self) -> None:
        for page, phrase in (
            ("128", "csc(A) = 1/sin(A) = c/a"),
            ("133", "unit is degrees"),
            ("149", "lateral component"),
            ("152", "ambiguous"),
            ("160", "ambiguous"),
        ):
            self.assertIn(page, self.corpus)
            self.assertIn(phrase, self.corpus)
        self.assertIn("false confidence", self.corpus)
        appendix = next(
            row
            for row in self.dispositions
            if row["record_id"] == "usfs-rigging-021"
        )
        self.assertEqual(appendix["manual_pages"], "127-168")
        self.assertEqual(
            appendix["disposition"], "included_transformed_with_errata"
        )

    def test_dispositions_cover_includes_exclusions_and_narrowing(self) -> None:
        self.assertEqual(len(self.dispositions), 25)
        self.assertEqual(
            len({row["record_id"] for row in self.dispositions}),
            len(self.dispositions),
        )
        included = [
            row for row in self.dispositions if row["direct_training_included"]
        ]
        excluded = [
            row for row in self.dispositions if not row["direct_training_included"]
        ]
        self.assertEqual(len(included), 15)
        self.assertEqual(len(excluded), 10)
        by_id = {row["record_id"]: row for row in self.dispositions}
        self.assertEqual(by_id["usfs-rigging-019"]["manual_pages"], "117-120")
        self.assertFalse(by_id["usfs-rigging-019"]["direct_training_included"])
        self.assertEqual(by_id["usfs-rigging-020b"]["manual_pages"], "123-124")
        self.assertTrue(by_id["usfs-rigging-020b"]["direct_training_included"])
        self.assertEqual(
            by_id["usfs-rigging-012"]["disposition"],
            "included_principles_excluded_procedures",
        )
        self.assertIn("construction recipes", by_id["usfs-rigging-012"]["treatment"])
        self.assertFalse(by_id["usfs-rigging-014"]["direct_training_included"])

    def test_split_hygiene_and_protected_data_boundary(self) -> None:
        split = self.provenance["split_hygiene"]
        self.assertTrue(split["non_qa"])
        self.assertFalse(split["protected_data_accessed_during_build"])
        self.assertEqual(
            set(split["protected_data_classes_not_accessed"]),
            {"validation", "OOD", "shadow", "sealed holdout", "protected QA"},
        )
        self.assertIn(
            "before Markdown chunking or QA derivation",
            split["document_disjoint_requirement"],
        )
        self.assertEqual(
            split["document_disjoint_requirement"],
            self.manifest["document_disjoint_requirement"],
        )
        self.assertIn(
            "validation, OOD, shadow, sealed-holdout, and protected-QA",
            self.manifest["protected_split_requirement"],
        )

    def test_manual_review_boundary_is_documented(self) -> None:
        method = self.provenance["review_method"]
        self.assertEqual(method["mode"], "manual_source_review_and_hand_cleanup")
        self.assertIn("manually selected", method["automation_boundary"])
        self.assertFalse(method["source_pdf_retained"])
        self.assertFalse(method["source_figures_retained"])
        self.assertFalse(method["source_tables_retained_verbatim"])
        self.assertIn("manually curated", self.report)
        self.assertIn("written and checked by hand", self.report)


if __name__ == "__main__":
    unittest.main()
