import hashlib
import json
import re
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def jsonl(name):
    return [
        json.loads(line)
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class MaibZargaCorpusTests(unittest.TestCase):
    def test_required_files_and_manifest_schema(self):
        required = {
            "CORPUS.md",
            "README.md",
            "REPORT.md",
            "components.jsonl",
            "dispositions.jsonl",
            "manifest.json",
            "source_snapshot/provenance.json",
            "sources.jsonl",
            "tests/test_corpus.py",
        }
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "site-corpus-manifest-v1.0")
        self.assertEqual(manifest["corpus_schema_version"], "rights-filtered-markdown-v1.0")
        self.assertTrue(manifest["direct_training_ready"])
        self.assertEqual(set(manifest["files"]), required - {"manifest.json"})

    def test_manifest_hashes(self):
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["hash_algorithm"], "sha256")
        for relative, expected in manifest["files"].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_exact_official_source_and_ogl_scope(self):
        rows = jsonl("sources.jsonl")
        self.assertEqual(len(rows), 1)
        source = rows[0]
        self.assertEqual(source["source_id"], "MAIB-ZARGA-2017")
        self.assertEqual(source["corporate_author"], "Marine Accident Investigation Branch")
        self.assertEqual(source["report_number"], "13/2017")
        self.assertEqual(source["publication_date"], "2017-06-15")
        self.assertEqual(source["page_count"], 116)
        self.assertEqual(source["license_spdx"], "OGL-UK-3.0")
        self.assertEqual(
            source["body_sha256"],
            "1a0b6aaab013406d0b7a604786d2eaea8a989ee850a77198fd74f0ae659538a4",
        )
        self.assertTrue(source["landing_url"].startswith("https://www.gov.uk/"))
        self.assertTrue(source["body_url"].startswith("https://assets.publishing.service.gov.uk/"))
        self.assertTrue(source["manual_page_audit"])
        self.assertTrue(source["manual_component_audit"])
        self.assertTrue(source["direct_training_ready"])

    def test_every_physical_page_has_one_manual_disposition(self):
        rows = jsonl("dispositions.jsonl")
        self.assertEqual(len(rows), 116)
        self.assertEqual([row["pdf_page"] for row in rows], list(range(1, 117)))
        self.assertEqual(len({row["pdf_page"] for row in rows}), 116)
        self.assertTrue(all(row["audit"] == "manual text and visual review" for row in rows))
        self.assertTrue(all(row["decision"] in {"partial", "exclude"} for row in rows))
        self.assertEqual(sum(row["decision"] == "partial" for row in rows), 26)
        for row in rows:
            if 15 <= row["pdf_page"] <= 115:
                self.assertEqual(row["report_page"], row["pdf_page"] - 14)
            else:
                self.assertIsNone(row["report_page"])

    def test_every_component_category_is_audited_and_excluded(self):
        rows = jsonl("components.jsonl")
        self.assertEqual(len(rows), 89)
        self.assertEqual(len({row["component_id"] for row in rows}), 89)
        self.assertTrue(all(row["decision"] == "exclude" for row in rows))
        self.assertTrue(all(row["rights_boundary"] for row in rows))
        self.assertTrue(all(row["reason"] for row in rows))
        counts = Counter(row["component_type"] for row in rows)
        self.assertEqual(
            counts,
            {"figure": 41, "table": 5, "annex": 16, "credited_source": 27},
        )
        self.assertEqual(
            [row["component_id"] for row in rows if row["component_type"] == "figure"],
            [f"figure-{n:02d}" for n in range(1, 42)],
        )
        self.assertEqual(
            [row["component_id"] for row in rows if row["component_type"] == "table"],
            [f"table-{n:02d}" for n in range(1, 6)],
        )
        self.assertEqual(
            [row["component_id"] for row in rows if row["component_type"] == "annex"],
            [f"annex-{letter}" for letter in "ABCDEFGHIJKLMNOP"],
        )
        provenance = json.loads((ROOT / "source_snapshot/provenance.json").read_text())
        self.assertEqual(provenance["audit"]["component_records"], 89)
        self.assertFalse(provenance["audit"]["annex_bundle_retrieved"])

    def test_corpus_is_dense_and_claim_cited(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        self.assertGreaterEqual(len(corpus.split()), 2200)
        citations = re.findall(r"\[MAIB-ZARGA-2017,[^\]]+\]", corpus)
        self.assertGreaterEqual(len(citations), 50)
        for paragraph in [p for p in corpus.split("\n\n") if p and not p.startswith("#")]:
            if len(paragraph.split()) >= 35:
                self.assertIn("[MAIB-ZARGA-2017,", paragraph, paragraph[:120])

    def test_required_failure_analysis_knowledge(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        required = [
            "Marine Accident Investigation Branch",
            "Report No. 13/2017",
            "long-lay, low-twist HMPE load-bearing core enclosed by a tightly fitted braided jacket",
            "Protection of the core and observability of the core are therefore different properties",
            "a rope can be under overall tensile load while individual internal components experience axial compression",
            "The jacket was not mechanically neutral",
            "line and fitting compatibility had not been evaluated as a complete system",
            "an apparently acceptable cover did not establish that the load-bearing core was intact",
            "no recognized nondestructive method capable of assessing the overall condition",
            "data-integration and information-flow problem",
            "Builder, supplier, operator, and assessor",
            "offshore test evidence did not transfer directly",
            "It reports no conflict-of-interest statement, funding statement, or data-availability statement",
        ]
        for phrase in required:
            self.assertIn(phrase, corpus)

    def test_domain_boundary_and_historical_limit(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        lower = corpus.lower()
        disclaimer = (
            "This single industrial maritime failure does not validate or certify bondage rope "
            "or human suspension."
        )
        self.assertIn(disclaimer, corpus)
        self.assertEqual(lower.count("bondage"), 1)
        self.assertEqual(lower.count("human suspension"), 1)
        self.assertIn("As framed in 2017", corpus)
        prohibited = [
            "jute",
            "hemp",
            "natural fiber",
            "natural-fiber",
            "body loading",
            "hardpoint",
            "upline",
            "anchor system",
            "snap-back",
            "retirement",
            "discard",
            "repair",
            "bridon",
            "dyneema",
            "marlow",
            "samson",
            "ocimf",
            "dnv",
            "stasco",
            "q-max",
            "q-flex",
            "minimum breaking load",
            "working load limit",
            " mbl",
            " wll",
        ]
        for phrase in prohibited:
            self.assertNotIn(phrase, lower, phrase)

    def test_no_quantitative_or_recipe_leakage(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        measurement = re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:%|mm|cm|metres?|meters?|tonnes?|hours?|cycles?|kn|mpa|gpa|°\s*c)\b",
            re.IGNORECASE,
        )
        self.assertIsNone(measurement.search(corpus))
        without_citations = re.sub(r"\[MAIB-ZARGA-2017,[^\]]+\]", "", corpus)
        numbers = re.findall(r"\b\d+(?:\.\d+)?(?:/\d+)?\b", without_citations)
        self.assertEqual(sorted(numbers), sorted(["13/2017", "2", "2015", "2017", "2017", "3.0"]))
        recipe_terms = [
            "remove from service",
            "should be inspected",
            "inspection interval",
            "service interval",
            "acceptance criterion",
            "rejection criterion",
            "safety factor",
            "bend ratio",
            "load threshold",
            "service life",
        ]
        lower = corpus.lower()
        for phrase in recipe_terms:
            self.assertNotIn(phrase, lower, phrase)

    def test_report_records_conservative_rights_filter(self):
        report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
        required = [
            "All 116 page images",
            "41 numbered figures",
            "5 numbered tables",
            "16 annexes",
            "27 credited source blocks",
            "89 auditable exclusion records",
            "Every photograph, figure, diagram, plot, table, caption, logo, annex",
            "Standards and legal text",
            "Equations, measurements, loads, nominal breaking values, ratios, dimensions",
            "Rope-selection, care, inspection, service-decision, or retirement recipes",
        ]
        for phrase in required:
            self.assertIn(phrase, report)


if __name__ == "__main__":
    unittest.main()
