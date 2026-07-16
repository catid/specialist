import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
REPORT = (ROOT / "REPORT.md").read_text(encoding="utf-8")
MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
PROVENANCE = json.loads(
    (ROOT / "source_snapshot" / "provenance.json").read_text(encoding="utf-8")
)


def jsonl(name):
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


SOURCES = jsonl("sources.jsonl")
DISPOSITIONS = jsonl("dispositions.jsonl")


class CorpusTests(unittest.TestCase):
    def test_required_files_exist(self):
        required = {
            "CORPUS.md",
            "REPORT.md",
            "manifest.json",
            "sources.jsonl",
            "dispositions.jsonl",
            "source_snapshot/provenance.json",
            "tests/test_corpus.py",
        }
        self.assertEqual(
            required,
            {str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()},
        )

    def test_source_ledger_has_four_official_sources(self):
        self.assertEqual(len(SOURCES), 4)
        self.assertEqual(
            {s["id"] for s in SOURCES},
            {"NIST-LANDING", "NBSIR-76-1159", "NIST-RIGHTS", "NIST-SERIES-RIGHTS"},
        )
        self.assertTrue(all("nist.gov" in s["url"] for s in SOURCES))
        self.assertTrue(all(s["status"] == 200 for s in SOURCES))

    def test_primary_source_hash_and_size(self):
        pdf = next(s for s in SOURCES if s["id"] == "NBSIR-76-1159")
        self.assertEqual(pdf["bytes"], 2119358)
        self.assertEqual(
            pdf["sha256"],
            "ab1fee99f1bb18bf48756847b33d41a23210978e8cdf6ebc15206ec45ec2e4ee",
        )

    def test_landing_hash(self):
        landing = next(s for s in SOURCES if s["id"] == "NIST-LANDING")
        self.assertEqual(
            landing["sha256"],
            "9055934f3fd83c762b081e29ab56b120413da9c6eed0dfa18329454167522bda",
        )

    def test_rights_hashes(self):
        by_id = {s["id"]: s for s in SOURCES}
        self.assertEqual(
            by_id["NIST-RIGHTS"]["sha256"],
            "94fe90461b9cc408e51da393ff3c01598ff1ca7e08df11f290056c6d896f107b",
        )
        self.assertEqual(
            by_id["NIST-SERIES-RIGHTS"]["sha256"],
            "f2c122419692e40b64c7b83d00a539923a9f7d8c3c1734c4fcc6a9d61b646026",
        )

    def test_no_source_body_is_retained(self):
        self.assertTrue(all(not s["retained_in_repository"] for s in SOURCES))
        forbidden_suffixes = {".pdf", ".html", ".htm", ".png", ".jpg", ".jpeg", ".webp"}
        self.assertFalse(any(p.suffix.lower() in forbidden_suffixes for p in ROOT.rglob("*")))
        self.assertFalse(PROVENANCE["manual_review"]["source_bodies_retained"])

    def test_front_loaded_context_and_limitation(self):
        front = CORPUS[:1800].lower()
        for phrase in (
            "september 1976",
            "helicopter external-cargo",
            "prototype",
            "inadequate end fittings",
            "not a current equipment standard",
            "rather than fully substantiated",
        ):
            self.assertIn(phrase, front)

    def test_identity_is_complete(self):
        for value in (
            "NBSIR 76-1159",
            "Nixon Halsey",
            "Leonard Mordfin",
            "10.6028/NBS.IR.76-1159",
            "September 1976",
        ):
            self.assertIn(value, CORPUS)

    def test_date_discrepancy_is_preserved(self):
        self.assertIn("January 1, 1976", CORPUS)
        self.assertIn("September 1976", CORPUS)
        self.assertEqual(
            PROVENANCE["report_identity"]["landing_page_publication_date"], "1976-01-01"
        )
        self.assertEqual(PROVENANCE["report_identity"]["report_imprint_date"], "1976-09")

    def test_specimen_design_is_preserved(self):
        for phrase in (
            "26 prototype sling-leg specimens",
            "parallel-strand",
            "cabled-strand",
            "low, medium, and high",
            "six basic rope styles",
            "nine specimen styles",
        ):
            self.assertIn(phrase, CORPUS)

    def test_required_experimental_topics_are_preserved(self):
        for phrase in (
            "Increasing-load-spectrum fatigue",
            "Alternating-block-spectrum fatigue",
            "Residual-strength follow-up",
            "Simulated weathering",
            "Permanent elongation",
        ):
            self.assertIn(phrase, CORPUS)

    def test_fatigue_counts_and_failure_modes_are_preserved(self):
        for phrase in (
            "Five specimens entered this path",
            "Seven specimens entered this path",
            "twelve specimens were fatigue-loaded",
            "partial strand failure",
            "free-length fatigue failure",
            "pulled out of a fitting",
        ):
            self.assertIn(phrase, CORPUS)

    def test_weathering_limits_are_preserved(self):
        for phrase in (
            "different specimens",
            "did not test their combined effect",
            "unexpectedly high temperatures",
            "potting material had softened excessively",
            "visible fiber discoloration",
        ):
            self.assertIn(phrase, CORPUS)

    def test_elongation_uncertainty_is_preserved(self):
        section = CORPUS.split("### Permanent elongation", 1)[1].split("## Why", 1)[0]
        self.assertIn("approximate and unsubstantiated", section)
        self.assertIn("not a service limit", section)
        self.assertIn("not a", section)

    def test_end_fitting_is_governing_interpretation(self):
        lower = CORPUS.lower()
        self.assertGreaterEqual(len(re.findall(r"end[- ]fitting", lower)), 15)
        self.assertIn("censored-measurement problem", lower)
        self.assertIn("did not determine the true strength", lower)
        self.assertIn("prevented a thorough evaluation", lower)

    def test_no_numeric_load_or_rating_recipe(self):
        self.assertIsNone(re.search(r"\b(?:lbf|kn|mn|hz)\b", CORPUS, re.IGNORECASE))
        self.assertIsNone(re.search(r"\bworking load\b", CORPUS, re.IGNORECASE))
        self.assertIsNone(re.search(r"\b\d+(?:\.\d+)?\s*%", CORPUS))

    def test_no_end_fitting_build_instructions(self):
        lower = CORPUS.lower()
        for phrase in (
            "drill a",
            "cut the",
            "mix ratio",
            "potting recipe",
            "step-by-step",
        ):
            self.assertNotIn(phrase, lower)
        for phrase in ("attach the fitting", "fill the fitting", "assemble the fitting"):
            self.assertNotIn(phrase, lower)

    def test_no_markdown_images_or_source_tables(self):
        self.assertNotIn("![", CORPUS)
        self.assertNotRegex(CORPUS, r"(?m)^\|.+\|$")

    def test_no_vendor_or_proprietary_details_in_corpus(self):
        self.assertNotRegex(CORPUS, r"(?i)phosphor[- ]bronze|manufacturer\s+name|purchase|price")

    def test_no_transfer_boundary_is_explicit(self):
        front = CORPUS[:1800].lower()
        for phrase in (
            "bondage rope",
            "human suspension",
            "natural-fiber rope",
            "knots",
            "uplines",
            "body contact",
            "care or retirement rules",
            "load ratings",
            "helicopter operation",
        ):
            self.assertIn(phrase, front)

    def test_claim_citations_are_dense(self):
        citations = re.findall(r"\[(?:NBSIR-76-1159|NIST-LANDING):[^\]]+\]", CORPUS)
        self.assertGreaterEqual(len(citations), 28)
        corpus_without_key = CORPUS.split("## Citation key", 1)[0]
        substantive = [
            p
            for p in re.split(r"\n\s*\n", corpus_without_key)
            if len(p.split()) >= 35 and not p.startswith("#")
        ]
        uncited = [p for p in substantive if "[NBSIR-76-1159:" not in p and "[NIST-LANDING:" not in p]
        self.assertEqual(uncited, [], "Every substantive corpus paragraph must carry a claim citation")

    def test_disposition_inventory_is_manual_and_complete(self):
        self.assertGreaterEqual(len(DISPOSITIONS), 23)
        decisions = {d["component"]: d["decision"] for d in DISPOSITIONS}
        self.assertEqual(decisions["Source tables 1 through 11"], "exclude")
        self.assertEqual(
            decisions["Source figures 1 through 8 and scanned page images"], "exclude"
        )
        self.assertEqual(decisions["End fittings"], "include_failure_evidence_only")
        self.assertEqual(decisions["Cross-domain operational transfer"], "exclude")

    def test_manual_review_covers_all_pages(self):
        self.assertTrue(PROVENANCE["manual_review"]["completed"])
        self.assertEqual(PROVENANCE["extraction_audit"]["pdf_pages"], 44)
        for page_range in ("1–4", "5–8", "10–16", "17–23", "24–31", "33–42"):
            self.assertIn(f"PDF pages {page_range}", REPORT)
        for page_number in (9, 32, 43, 44):
            self.assertIn(f"PDF page {page_number}", REPORT)

    def test_scan_and_component_provenance_is_recorded(self):
        audit = PROVENANCE["extraction_audit"]
        self.assertEqual(audit["raster_image_xobjects"], 88)
        self.assertEqual(audit["source_listed_figures"], 8)
        self.assertEqual(audit["source_listed_tables"], 11)
        self.assertEqual(audit["pdf_metadata_creator"], "Digitized by the Internet Archive")
        self.assertIn("LuraDocument", audit["pdf_metadata_producer"])

    def test_rights_screen_is_cautious(self):
        rights = PROVENANCE["rights_screen"]
        self.assertFalse(rights["legal_advice"])
        self.assertIn("third-party", rights["nist_series_notice"])
        self.assertIn("proprietary experimental fitting image", rights["provenance_cautions"])
        self.assertIn("This is a provenance screen, not legal advice", REPORT)

    def test_split_hygiene(self):
        split = MANIFEST["split_hygiene"]
        self.assertEqual(split["allowed"], ["training_source"])
        self.assertEqual(
            set(split["forbidden"]), {"validation", "holdout", "evaluation", "ood", "probe"}
        )
        self.assertIn("must remain source-only", REPORT)

    def test_manifest_source_ids_match_ledger(self):
        self.assertEqual(set(MANIFEST["source_ids"]), {s["id"] for s in SOURCES})

    def test_manifest_file_hashes(self):
        expected_files = {
            "CORPUS.md",
            "REPORT.md",
            "sources.jsonl",
            "dispositions.jsonl",
            "source_snapshot/provenance.json",
            "tests/test_corpus.py",
        }
        self.assertEqual(set(MANIFEST["files"]), expected_files)
        for relative, metadata in MANIFEST["files"].items():
            data = (ROOT / relative).read_bytes()
            self.assertEqual(metadata["bytes"], len(data))
            self.assertEqual(metadata["sha256"], hashlib.sha256(data).hexdigest())

    def test_manifest_has_no_self_hash_cycle(self):
        self.assertNotIn("manifest.json", MANIFEST["files"])


if __name__ == "__main__":
    unittest.main()
