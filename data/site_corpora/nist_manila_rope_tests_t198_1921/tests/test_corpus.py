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


class NistT198CorpusTests(unittest.TestCase):
    def test_manifest_schema_readiness_and_exact_artifact_set(self):
        manifest = read_json("manifest.json")
        self.assertEqual(manifest["schema"], "site-corpus-manifest")
        self.assertEqual(manifest["schema_version"], "1.0")
        self.assertEqual(
            manifest["corpus_schema"],
            "non-operational-historical-manila-rope-test-design-markdown",
        )
        self.assertEqual(manifest["corpus_schema_version"], "1.0")
        self.assertEqual(manifest["package_id"], "nist_manila_rope_tests_t198_1921")
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
        forbidden_suffixes = {".pdf", ".html", ".htm", ".jpg", ".jpeg", ".png", ".tif", ".txt"}
        self.assertFalse(
            any(path.suffix.lower() in forbidden_suffixes for path in ROOT.rglob("*"))
        )

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
            {"NIST-CATALOG-T198", "NBS-T198-1921", "NIST-TECHSERIES-RIGHTS"},
        )
        expected = {
            "NIST-CATALOG-T198": (
                "https://www.nist.gov/nist-research-library/technologic-papers-1921",
                83289,
                "41af780343b7be8bb12fab9c41742eab7492f11562dd809c804356af57d18df6",
                "6e9efb85d839b2df50bf533e316f4dd5f19c1b9ef881dbeaf16cbb410628ecc5ec776847b44f9b124939ccc887764224279f3c2ae9d7a0220ed3b614a9ef4081",
            ),
            "NBS-T198-1921": (
                "https://nvlpubs.nist.gov/nistpubs/nbstechnologic/nbstechnologicpaperT198.pdf",
                1207742,
                "bceb261d3ac009b71046b67a0ad400c632360bb7a61ba0c88c222ca31beb1f32",
                "c0f521ef8080a0a309a0a26abe914d315f2d9ebe37e8f297e5aebb83e893691b441b97cd938c08d66611e05d2ed0d35a4c7576552506f3f2144db3b976f6f840",
            ),
            "NIST-TECHSERIES-RIGHTS": (
                "https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications",
                66782,
                "35fc175a4a334c5a108b51fb5e098916a9cd2e842478bc5a426a6fb3e96f83f2",
                "d4a9c2abcab62bc55501d8ee3d07930fdd6cebc116c48bdf8ec22ba1c9bbbbc5f36e95e714d62ab1fe75aaa39f6c7329977624bfd11d2a2cd4f07c89d74e9db8",
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

        report = by_id["NBS-T198-1921"]
        self.assertEqual(report["authors"], ["Ambrose H. Stang", "Lory R. Strickenberg"])
        self.assertEqual(report["publication_date"], "1921-09-15")
        self.assertEqual(report["report_date"], "1921-02-26")
        self.assertEqual(report["doi"], "10.6028/nbst.6078")
        self.assertEqual(report["page_count"], 13)
        self.assertIn("public_domain_us_federal_work", report["rights_status"])
        rights = by_id["NIST-TECHSERIES-RIGHTS"]
        self.assertEqual(rights["rights_status"], "rights_provenance_only")
        self.assertIn("no policy prose used", rights["rights_scope"])
        self.assertIn("no longer updated", by_id["NIST-CATALOG-T198"]["page_status"])

    def test_retrieval_doi_robots_pdf_and_rights_provenance(self):
        provenance = read_json("source_snapshot/provenance.json")
        policy = provenance["retrieval_policy"]
        self.assertEqual(policy["allowed_bodies_retrieved"], 3)
        self.assertEqual(len(policy["allowed_urls"]), 3)
        self.assertIs(policy["site_crawl_performed"], False)
        self.assertIs(policy["mirrors_or_archives_used"], False)
        self.assertIs(policy["search_snippets_used_as_evidence"], False)
        self.assertEqual(policy["linked_source_bodies_retrieved"], 0)
        self.assertIs(policy["source_bodies_redistributed"], False)

        doi = provenance["doi_resolution"]
        self.assertEqual(doi["requested_url"], "https://doi.org/10.6028/nbst.6078")
        self.assertEqual(doi["initial_http_status"], 302)
        self.assertEqual(doi["final_http_status"], 200)
        self.assertEqual(doi["redirect_count"], 1)
        self.assertIs(doi["final_body_matches_direct_pdf_sha256"], True)

        robots = {row["host"]: row for row in provenance["robots_audit"]}
        self.assertEqual(set(robots), {"www.nist.gov", "nvlpubs.nist.gov"})
        self.assertEqual(robots["www.nist.gov"]["http_status"], 200)
        self.assertIs(robots["www.nist.gov"]["target_paths_disallowed"], False)
        self.assertEqual(robots["nvlpubs.nist.gov"]["http_status"], 404)
        self.assertIs(robots["nvlpubs.nist.gov"]["policy_available_at_host_root"], False)
        self.assertIn("no host crawl", robots["nvlpubs.nist.gov"]["retrieval_constraint"])

        pdf = provenance["pdf_metadata"]
        self.assertEqual(pdf["pages"], 13)
        self.assertIs(pdf["encrypted"], False)
        self.assertIs(pdf["text_layer"], True)
        self.assertIn("hidden OCR", pdf["text_layer_description"])
        bibliography = provenance["bibliographic_resolution"]
        self.assertEqual(bibliography["authors"], ["Ambrose H. Stang", "Lory R. Strickenberg"])
        self.assertEqual(bibliography["publication_year"], 1921)
        self.assertEqual(bibliography["doi"], "10.6028/nbst.6078")
        self.assertEqual(provenance["rights"]["rights_page_use"], "rights provenance only")
        self.assertIn("third-party", provenance["rights"]["third_party_caveat"].lower())
        self.assertIn("Republished courtesy", provenance["rights"]["attribution"])

    def test_every_pdf_page_has_exactly_one_manual_disposition(self):
        rows = read_jsonl("dispositions.jsonl")
        self.assertEqual(len(rows), 13)
        self.assertEqual([row["pdf_page"] for row in rows], list(range(1, 14)))
        self.assertTrue(all(row["source_id"] == "NBS-T198-1921" for row in rows))
        self.assertTrue(
            all(row["audit"] == "manual OCR, text and page-resolution visual review" for row in rows)
        )
        partial = {row["pdf_page"] for row in rows if row["decision"] == "partial"}
        excluded = {row["pdf_page"] for row in rows if row["decision"] == "exclude"}
        self.assertEqual(partial, {1, 3, 4, 7, 8, 9, 11, 12, 13})
        self.assertEqual(excluded, {2, 5, 6, 10})
        blank = {row["pdf_page"] for row in rows if "blank scan verso" in row["excluded"]}
        self.assertEqual(blank, {2})
        self.assertTrue(all(row["retained"] != "none" for row in rows if row["decision"] == "partial"))
        self.assertTrue(all(row["retained"] == "none" for row in rows if row["decision"] == "exclude"))

    def test_every_figure_table_and_controlled_component_is_excluded(self):
        rows = read_jsonl("components.jsonl")
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(row["source_id"] == "NBS-T198-1921" for row in rows))
        self.assertTrue(all(row["decision"] == "exclude" and row["retained"] == "none" for row in rows))
        figures = {row["component_id"]: row["pdf_pages"] for row in rows if row["component_type"] == "figure"}
        self.assertEqual(
            figures,
            {
                "figure-1": [5],
                "figure-2": [6],
                "figure-3": [10],
                "figure-4": [11],
                "figure-5": [12],
            },
        )
        tables = {row["component_id"]: row["pdf_pages"] for row in rows if row["component_type"] == "table"}
        self.assertEqual(tables, {"table-1": [8], "table-2": [9], "table-3": [9]})
        controlled = [row for row in rows if row["component_type"] == "controlled_exclusion_block"]
        self.assertEqual(len(controlled), 8)
        controlled_text = " ".join(row["excluded"] for row in controlled).lower()
        for phrase in [
            "numerical performance",
            "every equation",
            "specimen-preparation recipe",
            "procurement requirements",
            "period claims",
            "third-party",
        ]:
            self.assertIn(phrase, controlled_text)

    def test_html_surfaces_are_complete_and_rights_use_is_provenance_only(self):
        rows = read_jsonl("surfaces.jsonl")
        self.assertEqual(len(rows), 14)
        self.assertEqual(sum(row["source_id"] == "NIST-CATALOG-T198" for row in rows), 8)
        self.assertEqual(sum(row["source_id"] == "NIST-TECHSERIES-RIGHTS" for row in rows), 6)
        other_catalog = next(row for row in rows if row["surface_id"] == "catalog-06-other-publications")
        self.assertEqual(other_catalog["decision"], "exclude")
        self.assertEqual(other_catalog["retained"], "none")
        rights_retained = [
            row
            for row in rows
            if row["source_id"] == "NIST-TECHSERIES-RIGHTS" and row["decision"] != "exclude"
        ]
        self.assertEqual(len(rights_retained), 2)
        self.assertTrue(all(row["decision"] == "partial_rights_provenance_only" for row in rights_retained))
        self.assertTrue(all("rights provenance" in row["audit"] for row in rights_retained))

    def test_corpus_is_dense_claim_cited_and_explicitly_non_operational(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        paragraphs = [
            paragraph
            for paragraph in re.split(r"\n\s*\n", corpus)
            if paragraph and not paragraph.startswith("#")
        ]
        self.assertEqual(len(corpus.split()), 1829)
        self.assertEqual(len(paragraphs), 32)
        allowed_labels = (
            "Historical non-operational rope evidence:",
            "Historical non-operational rope evidence boundary:",
        )
        for paragraph in paragraphs:
            self.assertTrue(paragraph.startswith(allowed_labels), paragraph[:100])
            citations = re.findall(r"\[[^\]]+\]", paragraph)
            self.assertEqual(len(citations), 1, paragraph[:100])
            self.assertRegex(citations[0], r"(?:NBS-T198-1921|NIST-CATALOG-T198|NIST-TECHSERIES-RIGHTS)")
        self.assertEqual(corpus.count("NIST-TECHSERIES-RIGHTS"), 1)
        rights_paragraph = next(p for p in paragraphs if "NIST-TECHSERIES-RIGHTS" in p)
        self.assertTrue(rights_paragraph.startswith("Historical non-operational rope evidence boundary:"))
        self.assertIn("rights", rights_paragraph.lower())

    def test_corpus_contains_required_historical_scope_limits_and_negative_evidence(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        required = [
            "Ambrose H. Stang",
            "Lory R. Strickenberg",
            "commercial three-strand, regular-lay Manila rope",
            "chiefly through government purchase-order testing",
            "yarns are the smaller constituent elements grouped into strands",
            "Rope lay is the axial distance",
            "strand lay is the axial distance",
            "regular lay when yarns turn around the strand axis",
            "hard-laid and soft-laid",
            "same nominal diameter from different makers",
            "not part of an investigational program",
            "roughly half reportedly failed at or near a splice",
            "different coverage and should not be treated as a complete rectangular dataset",
            "no well-defined proportional limit",
            "cannot isolate a causal manufacturer effect",
            "6 mm jute or hemp",
            "rope care, cleaning, hygiene, drying, storage, inspection, or retirement",
            "knots or frictions",
            "body contact; uplines; anchors; bondage; or human suspension",
            "No operational calculator or rule can be generated",
        ]
        for phrase in required:
            self.assertIn(phrase.lower(), corpus.lower())

        without_citations = re.sub(r"\[[^\]]+\]", "", corpus)
        numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", without_citations))
        self.assertEqual(numbers, {"07", "10.6028", "16", "18", "1921", "2026", "368", "6", "6078"})
        self.assertNotIn("http://", corpus)
        self.assertNotIn("https://", corpus)
        self.assertNotIn("![", corpus)
        self.assertNotRegex(corpus, r"(?m)^>")
        self.assertNotRegex(corpus, r"(?m)^\s*\|.*\|\s*$")

    def test_no_raw_performance_formula_procedure_or_period_standard_leakage(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        banned = [
            r"\bWhitlock\b",
            r"\bPanama Canal\b",
            r"\b307-C\b",
            r"\bGovernment standard specifications\b",
            r"\b(?:feet|foot|inches?|ounces?|pounds?)\b",
            r"\b(?:lb|lbs|lbf|psi|kN)\b",
            r"\b(?:3700|5000|6300)\b",
            r"\b600\s*000\b",
            r"\b100\s*000\b",
            r"\bsoak(?:ed|ing)?\b",
            r"\bovernight\b",
            r"\bmoving head\b",
            r"\bP\s*=",
            r"\bL\s*=",
            r"\bN\s*=",
            r"d\s*\(\s*d\s*\+",
        ]
        for pattern in banned:
            self.assertNotRegex(corpus, re.compile(pattern, re.IGNORECASE))

        for paragraph in re.split(r"\n\s*\n", corpus):
            if re.search(
                r"\b(?:breaking loads?|strength|formulas?|equations?|coefficients?|working loads?|safety factors?|tolerance|ranking)\b",
                paragraph,
                re.IGNORECASE,
            ):
                self.assertTrue(
                    paragraph.startswith("Historical non-operational rope evidence boundary:")
                    or "does not establish strength" in paragraph
                    or "fitted empirical relations" in paragraph,
                    paragraph,
                )

    def test_no_current_selection_certification_or_human_support_inference(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        self.assertIn("Nothing here approves, selects, sizes, rates, tests, certifies, installs, inspects, retires, or evaluates", corpus)
        self.assertIn("It cannot be used to generate an operational calculator or rule", (ROOT / "REPORT.md").read_text(encoding="utf-8"))
        self.assertIn("appropriately qualified evaluation", corpus)
        paragraphs = [p for p in re.split(r"\n\s*\n", corpus) if p and not p.startswith("#")]
        for paragraph in paragraphs:
            if re.search(
                r"\b(?:6 mm|jute|hemp|care|hygiene|retirement|knots?|working loads?|body contact|uplines?|anchors?|hardpoint|bondage|human suspension)\b",
                paragraph,
                re.IGNORECASE,
            ):
                self.assertTrue(
                    paragraph.startswith("Historical non-operational rope evidence boundary:"),
                    paragraph,
                )
                self.assertRegex(
                    paragraph.lower(),
                    r"(?:cannot|does not|nothing|no operational|not studied|not establish)",
                )

    def test_report_documents_manual_audit_absolute_exclusions_and_surprise(self):
        report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
        for phrase in [
            "Every physical page was extracted for text screening, rendered, and manually inspected in two ordered contact sheets",
            "Nine pages are partially retained and four are excluded",
            "five figures and three tables",
            "rights page is used only for rights provenance",
            "returned HTTP 404 for `/robots.txt`",
            "frequent termination-region failure can confound interpretation of rope-body performance",
            "The source concerns commercial three-strand regular-lay Manila rope",
            "It cannot transfer to modern products, 6 mm jute or hemp",
            "Nothing in this package approves, selects, sizes, rates, tests, certifies, installs, inspects, retires, or evaluates",
            "cannot be used to generate an operational calculator or rule",
        ]:
            self.assertIn(phrase, report)

        audit = read_json("source_snapshot/provenance.json")["audit"]
        self.assertEqual(audit["pdf_pages"], 13)
        self.assertEqual(audit["pdf_pages_partially_retained"], 9)
        self.assertEqual(audit["pdf_pages_excluded"], 4)
        self.assertEqual(audit["blank_pages"], 1)
        self.assertEqual(audit["figures_excluded"], 5)
        self.assertEqual(audit["tables_excluded"], 3)
        self.assertEqual(audit["component_records"], 16)
        self.assertIs(audit["direct_training_ready"], True)


if __name__ == "__main__":
    unittest.main()
