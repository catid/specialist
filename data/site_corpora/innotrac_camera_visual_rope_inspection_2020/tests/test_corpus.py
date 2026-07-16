import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def read_json(relative_path: str):
    return json.loads(read_text(relative_path))


def read_jsonl(relative_path: str):
    return [json.loads(line) for line in read_text(relative_path).splitlines() if line]


def sha256(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


class InnoTracCorpusTests(unittest.TestCase):
    maxDiff = None

    def test_exact_package_file_set_and_no_source_bodies(self):
        files = {
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        self.assertEqual(
            files,
            {
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
            },
        )
        forbidden_suffixes = {
            ".csv", ".doc", ".docx", ".epub", ".gif", ".htm", ".html",
            ".jpeg", ".jpg", ".pdf", ".png", ".svg", ".tif", ".tiff", ".webp",
        }
        self.assertFalse(any(path.suffix.lower() in forbidden_suffixes for path in ROOT.rglob("*")))

    def test_locked_content_hashes(self):
        expected = {
            "CORPUS.md": "84fb8d18bb2fe1b1492dd7c96f5ffb04f6a134435a66f244064e3e9e13842f17",
            "README.md": "681357970a7107c1c81e0d878fca53b6494bbae1efc5fa20d6b8b4efd966e8c3",
            "REPORT.md": "fed6f367bcdec5bdb3af064cc76d11e1c3dae40a67631af71d6ea41af4246cd2",
            "components.jsonl": "ab1f50b703611fa77f7365b5f4e40c8def3f53782bd8167ebb48092410ff998d",
            "dispositions.jsonl": "e19eb89a78758d44311abf130931ce7246a6235c24b060c33d32e5cd109e0b6f",
            "source_snapshot/provenance.json": "ab29228b8d33d26f9d391184c804749934ce3109497333707ad90f8ddc6e164a",
            "sources.jsonl": "379087b23e89114cce16d99074a9b4b78afac7305dd446f58c6d7ab53d1f9577",
            "surfaces.jsonl": "9221bd4a47c68206dd1a5e6fd206487302eac66458b4fa43cc740b1322aab623",
        }
        self.assertEqual({path: sha256(path) for path in expected}, expected)

    def test_exact_sources_rights_and_checksums(self):
        rows = read_jsonl("sources.jsonl")
        self.assertEqual(len(rows), 3)
        by_id = {row["source_id"]: row for row in rows}
        self.assertEqual(
            set(by_id),
            {"INNOTRAC-RECORD-462", "INNOTRAC-462-2020", "CC-BY-4.0-LEGALCODE"},
        )
        record = by_id["INNOTRAC-RECORD-462"]
        pdf = by_id["INNOTRAC-462-2020"]
        license_row = by_id["CC-BY-4.0-LEGALCODE"]
        self.assertEqual(record["body_bytes"], 20192)
        self.assertEqual(record["body_sha256"], "9f62e69a04e9453ca88d617b6b4456c7e0e6ad9d63a72854746c5ec0b3b20092")
        self.assertEqual(pdf["body_bytes"], 670058)
        self.assertEqual(pdf["body_sha256"], "f7b64101a3d8df7b76c8a8ecb06e631a1a892985e104bd10248f19d923559af2")
        self.assertEqual(pdf["page_count"], 9)
        self.assertEqual(license_row["body_bytes"], 48970)
        self.assertEqual(license_row["body_sha256"], "6d55b998ed5c54f43426d059a8c549ed58a3321e5463e6a6af1c6b56ab78c333")
        self.assertEqual(record["doi"], pdf["doi"])
        self.assertEqual(pdf["doi"], "10.14464/innotrac.v1i0.462")
        self.assertEqual(record["publication_date"], "2020-12-03")
        self.assertEqual(pdf["available_online_date"], "2020-12-07")
        self.assertEqual(record["license_spdx"], "CC-BY-4.0")
        self.assertEqual(pdf["license_spdx"], "CC-BY-4.0")
        self.assertEqual(license_row["rights_status"], "rights_provenance_only")
        self.assertTrue(all(row["manual_surface_audit"] for row in (record, license_row)))
        self.assertTrue(pdf["manual_page_audit"] and pdf["manual_component_audit"])
        self.assertTrue(all(row["direct_training_ready"] for row in rows))

    def test_provenance_retrieval_robots_pdf_and_disclosures(self):
        provenance = read_json("source_snapshot/provenance.json")
        policy = provenance["retrieval_policy"]
        self.assertEqual(policy["allowed_bodies_retrieved"], 3)
        self.assertFalse(policy["site_crawl_performed"])
        self.assertFalse(policy["mirrors_or_archives_used"])
        self.assertFalse(policy["search_snippets_used_as_evidence"])
        self.assertEqual(policy["linked_reference_bodies_retrieved"], 0)
        self.assertEqual(policy["product_or_project_bodies_retrieved"], 0)
        self.assertFalse(policy["source_bodies_redistributed"])
        self.assertTrue(all(not row["target_paths_disallowed"] for row in provenance["robots_audit"]))
        self.assertTrue(provenance["doi_resolution"]["final_body_matches_direct_record_sha256"])
        pdf = provenance["pdf_metadata"]
        self.assertEqual(pdf["pages"], 9)
        self.assertTrue(pdf["tagged"] and pdf["text_layer"])
        self.assertFalse(pdf["encrypted"])
        rights = provenance["rights"]
        self.assertEqual(rights["license_spdx"], "CC-BY-4.0")
        self.assertIn("Patent and trademark rights are not licensed", rights["patent_trademark_boundary"])
        disclosures = provenance["disclosures"]
        for field in (
            "formal_funding_statement_present",
            "conflict_of_interest_statement_present",
            "data_availability_statement_present",
            "code_availability_statement_present",
            "protocol_or_preregistration_present",
        ):
            self.assertFalse(disclosures[field])

    def test_every_pdf_page_has_one_manual_disposition(self):
        rows = read_jsonl("dispositions.jsonl")
        self.assertEqual(len(rows), 9)
        self.assertEqual([row["pdf_page"] for row in rows], list(range(1, 10)))
        self.assertEqual([row["printed_page"] for row in rows], [str(page) for page in range(55, 64)])
        self.assertTrue(all(row["source_id"] == "INNOTRAC-462-2020" for row in rows))
        self.assertTrue(all(row["decision"] == "partial" for row in rows))
        self.assertTrue(all(row["audit"].startswith("manual") for row in rows))
        self.assertTrue(all(row["retained"] and row["excluded"] for row in rows))

    def test_components_reconcile_all_figures_and_exclusions(self):
        rows = read_jsonl("components.jsonl")
        self.assertEqual(len(rows), 18)
        figures = [row for row in rows if row["component_type"] == "figure"]
        controls = [row for row in rows if row["component_type"] != "figure"]
        self.assertEqual([row["component_id"] for row in figures], [f"figure-{n}" for n in range(1, 10)])
        self.assertEqual([row["pdf_pages"] for row in figures], [[4], [4], [5], [6], [7], [7], [8], [8], [8]])
        self.assertEqual(len(controls), 9)
        self.assertEqual(sum(row["component_type"] == "logical_rule_or_equation" for row in controls), 1)
        self.assertTrue(all(row["decision"] == "exclude" and row["retained"] == "none" for row in rows))
        control_text = " ".join(row["excluded"] for row in controls).lower()
        for required in ("threshold", "camera count", "algorithms", "registered marks", "standards", "remaining-life", "twelve reference"):
            self.assertIn(required, control_text)

    def test_all_material_html_surfaces_are_disposed(self):
        rows = read_jsonl("surfaces.jsonl")
        self.assertEqual(len(rows), 15)
        self.assertEqual(sum(row["source_id"] == "INNOTRAC-RECORD-462" for row in rows), 9)
        self.assertEqual(sum(row["source_id"] == "CC-BY-4.0-LEGALCODE" for row in rows), 6)
        self.assertTrue(all(row["decision"] in {"partial", "exclude", "partial_rights_provenance_only"} for row in rows))
        license_rows = [row for row in rows if row["source_id"] == "CC-BY-4.0-LEGALCODE"]
        self.assertTrue(all("rights provenance" in row["audit"] for row in license_rows))

    def test_manifest_statistics_and_artifact_hashes(self):
        manifest = read_json("manifest.json")
        self.assertTrue(manifest["direct_training_ready"])
        self.assertEqual(manifest["rights_status"], "cc-by-4.0")
        self.assertEqual(
            manifest["statistics"],
            {
                "allowed_source_bodies": 3,
                "article_record_surfaces": 9,
                "license_surfaces": 6,
                "html_surfaces": 15,
                "pdf_pages": 9,
                "pdf_pages_partially_retained": 9,
                "pdf_pages_excluded": 0,
                "blank_pages": 0,
                "figures_excluded": 9,
                "tables_present": 0,
                "tables_excluded": 0,
                "logical_rules_or_equations_excluded": 1,
                "reference_entries_excluded": 12,
                "controlled_exclusion_blocks": 9,
                "component_records": 18,
                "corpus_bytes": 13097,
                "corpus_words": 1648,
                "corpus_claim_paragraphs": 32,
                "corpus_evidence_paragraphs": 23,
                "corpus_boundary_paragraphs": 9,
                "corpus_citations": 32,
            },
        )
        artifacts = manifest["artifacts"]
        self.assertEqual(
            set(artifacts),
            {
                "CORPUS.md", "README.md", "REPORT.md", "components.jsonl",
                "dispositions.jsonl", "source_snapshot/provenance.json", "sources.jsonl",
                "surfaces.jsonl", "tests/test_corpus.py",
            },
        )
        for relative_path, metadata in artifacts.items():
            self.assertEqual(metadata["bytes"], (ROOT / relative_path).stat().st_size)
            self.assertEqual(metadata["sha256"], sha256(relative_path))

    def test_corpus_is_dense_claim_cited_and_source_bounded(self):
        corpus = read_text("CORPUS.md")
        self.assertEqual(len(corpus.encode("utf-8")), 13097)
        self.assertEqual(len(corpus.split()), 1648)
        paragraphs = [part for part in corpus.split("\n\n") if not part.startswith("#")]
        self.assertEqual(len(paragraphs), 32)
        self.assertEqual(sum(part.startswith("Industrial inspection-method evidence:") for part in paragraphs), 23)
        self.assertEqual(sum(part.startswith("Industrial inspection-method evidence boundary:") for part in paragraphs), 9)
        citation_pattern = re.compile(r"\[(INNOTRAC-462-2020|INNOTRAC-RECORD-462|CC-BY-4\.0-LEGALCODE), [^\]]+\]$")
        self.assertTrue(all(citation_pattern.search(part) for part in paragraphs))
        self.assertEqual(len(re.findall(r"\[[^\]]+\]", corpus)), 32)

    def test_required_method_lessons_and_limitations_are_present(self):
        corpus = read_text("CORPUS.md").lower()
        required = (
            "normally sees the surface facing that viewing position",
            "fatigue or perceptual habituation",
            "no continuous image record",
            "multi-angle does not mean complete coverage",
            "one or more cameras were used",
            "ellipse rather than a direct diameter",
            "considering several indicators together",
            "complete sample inventory",
            "held-out detection study",
            "false-positive or false-negative analysis",
            "surface observability is not internal condition",
            "discard criteria for high-modulus fibre ropes as insufficiently researched",
        )
        for phrase in required:
            self.assertIn(phrase, corpus)

    def test_no_procedure_product_figure_or_numeric_rule_leakage(self):
        corpus = read_text("CORPUS.md")
        lower = corpus.lower()
        for forbidden in (
            "fibrespect", "winspect", "dyneema", "technora", "vectran",
            "microsoft", "excel", "iso 4309", "zim", "figure 1", "figure 2",
            "figure 3", "figure 4", "figure 5", "figure 6", "figure 7",
            "figure 8", "figure 9", "variance of deflection", "if fringes",
        ):
            self.assertNotIn(forbidden, lower)
        self.assertNotRegex(corpus, r"https?://|www\.")
        self.assertNotRegex(corpus, r"!\[[^\]]*\]\([^)]*\)")
        self.assertNotRegex(corpus, r"(?im)^\s*\|.*\|\s*$")
        self.assertNotRegex(corpus, r"(?i)\b\d+(?:\.\d+)?\s*(?:%|mm|cm|m|kn|newtons?|seconds?|minutes?|hours?|pixels?)\b")
        self.assertNotRegex(corpus, r"(?i)\b(?:threshold|discard point)\s*(?:=|:|is|at)\s*\d")

    def test_absolute_nontransfer_boundary_is_explicit(self):
        corpus = read_text("CORPUS.md").lower()
        boundary = corpus[corpus.index("## absolute application boundary") :]
        for phrase in (
            "natural-fibre, jute, or hemp shibari rope",
            "visual bondage-rope inspection",
            "rope care, cleaning, hygiene, drying, storage",
            "knots or frictions",
            "working loads",
            "body contact",
            "uplines",
            "anchors",
            "human suspension",
            "nothing here approves, selects, rates, calibrates, installs, validates, certifies, or evaluates",
            "no threshold, discard rule, remaining-life estimator, internal-condition classifier, or operational calculator",
        ):
            self.assertIn(phrase, boundary)

    def test_report_records_manual_audit_and_validation_surprise(self):
        report = read_text("REPORT.md").lower()
        for phrase in (
            "every physical page was extracted for text screening, rendered, and manually inspected",
            "all nine numbered figures",
            "article has no tables",
            "no complete sample inventory",
            "several-camera concept and experiments described as using one or more cameras",
            "surface imaging does not establish internal condition",
            "the three allowed source bodies are not redistributed",
        ):
            self.assertIn(phrase, report)


if __name__ == "__main__":
    unittest.main()
