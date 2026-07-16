import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    for line_number, line in enumerate(
        (ROOT / path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AssertionError(f"{path}:{line_number}: {error}") from error
    return rows


def sha256(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


class Nistir6096CorpusTests(unittest.TestCase):
    def test_manifest_schema_readiness_and_exact_artifact_set(self):
        manifest = read_json("manifest.json")
        self.assertEqual(manifest["schema"], "site-corpus-manifest")
        self.assertEqual(manifest["schema_version"], "1.0")
        self.assertEqual(
            manifest["corpus_schema"],
            "non-bondage-historical-concrete-anchor-review-markdown",
        )
        self.assertEqual(manifest["corpus_schema_version"], "1.0")
        self.assertEqual(
            manifest["package_id"], "nistir_6096_post_installed_anchors_review"
        )
        self.assertIs(manifest["direct_training_ready"], True)

        expected = {
            "CORPUS.md",
            "README.md",
            "REPORT.md",
            "components.jsonl",
            "dispositions.jsonl",
            "manifest.json",
            "source_snapshot/provenance.json",
            "sources.jsonl",
            "surfaces.jsonl",
            "tests/test_corpus.py",
        }
        actual = {
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        self.assertEqual(actual, expected)
        self.assertFalse(any(path.suffix.lower() in {".pdf", ".html", ".jpg", ".png"} for path in ROOT.rglob("*")))

    def test_artifact_hashes_are_complete_and_exact(self):
        manifest = read_json("manifest.json")
        expected_paths = {
            "CORPUS.md",
            "README.md",
            "REPORT.md",
            "components.jsonl",
            "dispositions.jsonl",
            "source_snapshot/provenance.json",
            "sources.jsonl",
            "surfaces.jsonl",
            "tests/test_corpus.py",
        }
        artifacts = manifest["artifacts"]
        self.assertEqual({item["path"] for item in artifacts}, expected_paths)
        for item in artifacts:
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(item["sha256"], sha256(item["path"]))

    def test_exact_allowed_sources_identity_rights_and_checksums(self):
        sources = read_jsonl("sources.jsonl")
        self.assertEqual(len(sources), 3)
        by_id = {row["source_id"]: row for row in sources}
        self.assertEqual(
            set(by_id),
            {"NIST-PUBREC-6096", "NISTIR-6096-1998", "NIST-TECHSERIES-RIGHTS"},
        )
        expected = {
            "NIST-PUBREC-6096": (
                "https://www.nist.gov/publications/post-installed-anchors-literature-review",
                51556,
                "2d2d93325e52467669693cfc30fa5d3484e1cff56e492ef2adc175fa90ec70c0",
                "2f301dde0b78138bfaa11b6c113f976b6acc4945548bc6ef9fc1a9848b2cd6cfddbdce09a8219d3579c22326dc30bc266a9d101fd4b8b0037bf977733f27d7b0",
            ),
            "NISTIR-6096-1998": (
                "https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir6096.pdf",
                7053078,
                "8b6b586e65e61a634b29af302bfa686d4bb357a304f661363f9863028c6c88d7",
                "6920361c3cc96be340a758ac14fdb544df53af52de3798bc926e4d6bf0f22004d19e198f501603a14bdca299b377caa6c2025d2668b2b68f2ea6b8f9b22ea146",
            ),
            "NIST-TECHSERIES-RIGHTS": (
                "https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications",
                94086,
                "421c832ad4aac065a5ac5e2db167c643d2e3f97b58b851d8d21b2b730a932be7",
                "79236270c1c0e3a562e89fe6f2a4e18929f98e0665e87ff05f27f72243b911d9d0c148d4063ab666590c423305838d48590978a6f33971dd4cfae9942c3942e7",
            ),
        }
        for source_id, (url, size, digest256, digest512) in expected.items():
            row = by_id[source_id]
            self.assertEqual(row["source_url"], url)
            self.assertEqual(row["http_status"], 200)
            self.assertEqual(row["body_bytes"], size)
            self.assertEqual(row["body_sha256"], digest256)
            self.assertEqual(row["body_sha512"], digest512)
            self.assertIs(row["direct_training_ready"], True)

        report = by_id["NISTIR-6096-1998"]
        self.assertEqual(report["authors"], ["Geraldine S. Cheok", "Long T. Phan"])
        self.assertEqual(report["publication_period"], "January 1998")
        self.assertEqual(report["page_count"], 109)
        self.assertIn("public_domain_us_federal_work", report["rights_status"])
        rights = by_id["NIST-TECHSERIES-RIGHTS"]
        self.assertEqual(rights["rights_status"], "rights_provenance_only")
        self.assertIn("no technical-series policy prose used", rights["rights_scope"])

    def test_retrieval_route_pdf_metadata_and_audit_provenance(self):
        provenance = read_json("source_snapshot/provenance.json")
        policy = provenance["retrieval_policy"]
        self.assertEqual(policy["allowed_bodies_retrieved"], 3)
        self.assertEqual(len(policy["allowed_urls"]), 3)
        self.assertIs(policy["site_crawl_performed"], False)
        self.assertIs(policy["mirrors_or_archives_used"], False)
        self.assertEqual(policy["linked_source_bodies_retrieved"], 0)

        pdf = provenance["pdf_metadata"]
        self.assertEqual(pdf["pages"], 109)
        self.assertIs(pdf["encrypted"], False)
        self.assertIs(pdf["text_layer"], False)
        self.assertEqual(provenance["bibliographic_resolution"]["publication_year"], 1998)
        self.assertEqual(
            provenance["bibliographic_resolution"]["authors"],
            ["Geraldine S. Cheok", "Long T. Phan"],
        )
        self.assertEqual(provenance["rights"]["rights_page_use"], "rights provenance only")
        self.assertIn("third-party", provenance["rights"]["third_party_caveat"].lower())

        audit = provenance["audit"]
        self.assertEqual(audit["html_surfaces"], 14)
        self.assertEqual(audit["pdf_pages"], 109)
        self.assertEqual(audit["pdf_pages_partially_retained"], 26)
        self.assertEqual(audit["pdf_pages_excluded"], 83)
        self.assertEqual(audit["blank_pages"], 6)
        self.assertEqual(audit["figures_excluded"], 13)
        self.assertEqual(audit["tables_excluded"], 4)
        self.assertEqual(audit["component_records"], 27)
        self.assertIs(audit["direct_training_ready"], True)

    def test_every_pdf_page_has_exactly_one_manual_disposition(self):
        rows = read_jsonl("dispositions.jsonl")
        self.assertEqual(len(rows), 109)
        self.assertEqual([row["pdf_page"] for row in rows], list(range(1, 110)))
        self.assertTrue(all(row["source_id"] == "NISTIR-6096-1998" for row in rows))
        self.assertTrue(all(row["audit"] == "manual OCR, text and visual review" for row in rows))
        partial = {row["pdf_page"] for row in rows if row["decision"] == "partial"}
        self.assertEqual(
            partial,
            {1, 2, 3, 4, 5, 6, 11, 12, 15, 36, 37, 58, 59, 72, 73, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88},
        )
        self.assertEqual(sum(row["decision"] == "exclude" for row in rows), 83)
        blank = {row["pdf_page"] for row in rows if "blank scan page" in row["excluded"]}
        self.assertEqual(blank, {7, 9, 44, 64, 76, 82})
        self.assertTrue(all(row["retained"] != "none" for row in rows if row["decision"] == "partial"))
        self.assertTrue(all(row["retained"] == "none" for row in rows if row["decision"] == "exclude"))

    def test_every_figure_table_and_controlled_component_is_excluded(self):
        rows = read_jsonl("components.jsonl")
        self.assertEqual(len(rows), 27)
        self.assertTrue(all(row["source_id"] == "NISTIR-6096-1998" for row in rows))
        self.assertTrue(all(row["decision"] == "exclude" and row["retained"] == "none" for row in rows))
        figures = {row["component_id"]: row["pdf_pages"] for row in rows if row["component_type"] == "figure"}
        self.assertEqual(
            figures,
            {
                "figure-2-1": [16], "figure-3-1": [40], "figure-3-2": [40],
                "figure-3-3": [41], "figure-3-4": [42], "figure-3-5": [43],
                "figure-4-1": [61], "figure-4-2": [62], "figure-4-3": [63],
                "figure-5-1": [74], "figure-5-2": [74], "figure-5-3": [75],
                "figure-a-1": [98],
            },
        )
        tables = {row["component_id"]: row["pdf_pages"] for row in rows if row["component_type"] == "table"}
        self.assertEqual(
            tables,
            {"table-2-1": [14], "table-3-1": [23], "table-3-2": [38], "table-4-1": [60]},
        )
        controlled = [row for row in rows if row["component_type"] == "controlled_exclusion_block"]
        self.assertEqual(len(controlled), 10)

    def test_html_surfaces_are_complete_and_rights_use_is_provenance_only(self):
        rows = read_jsonl("surfaces.jsonl")
        self.assertEqual(len(rows), 14)
        self.assertEqual(sum(row["source_id"] == "NIST-PUBREC-6096" for row in rows), 8)
        self.assertEqual(sum(row["source_id"] == "NIST-TECHSERIES-RIGHTS" for row in rows), 6)
        retained = [row for row in rows if row["decision"] != "exclude"]
        self.assertEqual(len(retained), 5)
        rights_retained = [row for row in retained if row["source_id"] == "NIST-TECHSERIES-RIGHTS"]
        self.assertEqual(len(rights_retained), 1)
        self.assertEqual(rights_retained[0]["decision"], "partial_rights_provenance_only")
        self.assertIn("rights provenance", rights_retained[0]["audit"])

    def test_corpus_is_dense_claim_cited_and_explicitly_non_bondage(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        paragraphs = [
            paragraph
            for paragraph in re.split(r"\n\s*\n", corpus)
            if paragraph and not paragraph.startswith("#")
        ]
        self.assertEqual(len(corpus.split()), 1786)
        self.assertEqual(len(paragraphs), 36)
        allowed_labels = (
            "Non-bondage concrete-anchor evidence:",
            "Non-bondage concrete-anchor evidence boundary:",
        )
        for paragraph in paragraphs:
            self.assertTrue(paragraph.startswith(allowed_labels), paragraph[:100])
            citations = re.findall(r"\[[^\]]+\]", paragraph)
            self.assertEqual(len(citations), 1, paragraph[:100])
            self.assertRegex(citations[0], r"NIST(?:IR-6096-1998|-PUBREC-6096|-TECHSERIES-RIGHTS)")
        self.assertEqual(corpus.count("NIST-TECHSERIES-RIGHTS"), 1)
        rights_paragraph = next(p for p in paragraphs if "NIST-TECHSERIES-RIGHTS" in p)
        self.assertIn("evidence boundary", rights_paragraph)
        self.assertIn("rights surface", rights_paragraph)

    def test_corpus_contains_only_bounded_historical_synthesis(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        required = [
            "Geraldine S. Cheok",
            "Long T. Phan",
            "historical secondary review",
            "steel failure",
            "concrete-cone failure",
            "edge breakout or bursting",
            "splitting failure",
            "Pull-out or slip",
            "anchor-to-bonding-medium interface",
            "bonding-medium-to-concrete interface",
            "tension-only, shear-only, and combined tension-and-shear",
            "static, cyclic, and reversed-cyclic",
            "Cracked and uncracked concrete",
            "substantial scatter across data assembled from separate studies",
            "ongoing, not yet reported, or omitted because it was proprietary",
            "current site-specific qualified engineering verification",
        ]
        for phrase in required:
            self.assertIn(phrase.lower(), corpus.lower())

        without_citations = re.sub(r"\[[^\]]+\]", "", corpus)
        self.assertEqual(set(re.findall(r"\b\d+(?:\.\d+)?\b", without_citations)), {"1998", "6096"})
        self.assertNotIn("http://", corpus)
        self.assertNotIn("https://", corpus)
        self.assertNotIn("![", corpus)
        self.assertNotRegex(corpus, r"(?m)^>")
        self.assertNotRegex(corpus, r"(?m)^\s*\|.*\|\s*$")

        banned = [
            r"\bACI\b", r"\bASTM\b", r"\bCEB\b", r"\bCCD\b", r"\bETAG\b", r"\bICC\b", r"\bVAC\b",
            r"\bMPa\b", r"\bpsi\b", r"\bkN\b", r"\bmillimet(?:er|re)s?\b", r"\binches?\b",
            r"\bepoxy\b", r"\bpolyester\b", r"\bvinylester\b", r"\bmanufacturer\b",
            r"\bdrill(?:ed|ing)?\b", r"\btorque(?:d|ing)?\b", r"\bhole cleaning\b",
        ]
        for pattern in banned:
            self.assertNotRegex(corpus, re.compile(pattern, re.IGNORECASE))

    def test_no_current_selection_certification_or_human_support_inference(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        self.assertIn("does not approve, select, size, proof-test, certify, install, or evaluate", corpus)
        self.assertIn("provides no operational selection, sizing, testing, proof-load, or certification claim", corpus)
        self.assertIn("appropriately qualified professionals", corpus)
        paragraphs = [p for p in re.split(r"\n\s*\n", corpus) if p and not p.startswith("#")]
        for paragraph in paragraphs:
            if re.search(r"\b(?:ceiling|hardpoint|body-support|human-suspension)\b", paragraph, re.I):
                self.assertTrue(paragraph.startswith("Non-bondage concrete-anchor evidence boundary:"))
                self.assertRegex(paragraph, r"\b(?:does not|provides no)\b")

    def test_report_documents_manual_audit_and_absolute_exclusions(self):
        report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
        for phrase in [
            "All 109 page images were manually inspected in ten ordered contact sheets",
            "Twenty-six pages are partially retained and eighty-three are excluded",
            "six explicit blank pages",
            "thirteen figures and four tables",
            "rights page is used only for rights provenance",
            "current site-specific verification by appropriately qualified engineering professionals",
            "Nothing in this package approves, selects, sizes, proof-tests, certifies, or evaluates",
        ]:
            self.assertIn(phrase, report)


if __name__ == "__main__":
    unittest.main()
