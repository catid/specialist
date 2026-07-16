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


class ThrowBagCorpusTests(unittest.TestCase):
    def test_manifest_schema_and_required_files(self):
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
        self.assertEqual(manifest["schema"], "site-corpus-manifest")
        self.assertEqual(manifest["schema_version"], "1.0")
        self.assertEqual(manifest["corpus_schema"], "rights-filtered-markdown")
        self.assertTrue(manifest["direct_training_ready"])
        self.assertEqual(set(manifest["files"]), required - {"manifest.json"})

    def test_manifest_hashes(self):
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["hash_algorithm"], "sha256")
        for relative, expected in manifest["files"].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_exact_first_party_source_and_rights(self):
        sources = jsonl("sources.jsonl")
        self.assertEqual(len(sources), 1)
        source = sources[0]
        self.assertEqual(source["schema"], "site-corpus-source")
        self.assertEqual(source["schema_version"], "1.0")
        self.assertEqual(source["source_id"], "MAIB-THROWBAG-2019")
        self.assertEqual(source["report_number"], "2/2019")
        self.assertEqual(source["corporate_author"], "Marine Accident Investigation Branch")
        self.assertEqual(source["publication_date"], "2019-01-31")
        self.assertEqual(source["page_count"], 8)
        self.assertEqual(source["license_spdx"], "OGL-UK-3.0")
        self.assertEqual(
            source["body_sha256"],
            "1d29a09b23c21d40aca2e0da4087b63aa648b77816e6dc80e8abecb4d2d16b66",
        )
        self.assertTrue(source["landing_url"].startswith("https://www.gov.uk/"))
        self.assertTrue(source["body_url"].startswith("https://assets.publishing.service.gov.uk/"))
        self.assertTrue(source["manual_page_audit"])
        self.assertTrue(source["manual_component_audit"])
        self.assertTrue(source["direct_training_ready"])

    def test_every_page_has_one_manual_disposition(self):
        rows = jsonl("dispositions.jsonl")
        self.assertEqual(len(rows), 8)
        self.assertEqual([row["pdf_page"] for row in rows], list(range(1, 9)))
        self.assertTrue(all(row["audit"] == "manual text and visual review" for row in rows))
        self.assertTrue(all(row["decision"] in {"partial", "exclude"} for row in rows))
        self.assertEqual(
            {row["pdf_page"] for row in rows if row["decision"] == "partial"},
            {1, 2, 3, 5, 6, 7},
        )

    def test_every_component_is_audited_and_excluded(self):
        rows = jsonl("components.jsonl")
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({row["component_id"] for row in rows}), 16)
        self.assertTrue(all(row["decision"] == "exclude" for row in rows))
        self.assertTrue(all(row["rights_boundary"] for row in rows))
        self.assertTrue(all(row["reason"] for row in rows))
        self.assertEqual(
            Counter(row["component_type"] for row in rows),
            {"credited_source": 11, "figure": 3, "visual": 1, "table": 1},
        )
        self.assertEqual(
            [row["component_id"] for row in rows if row["component_type"] == "figure"],
            ["figure-01", "figure-02", "figure-03"],
        )
        provenance = json.loads((ROOT / "source_snapshot/provenance.json").read_text())
        self.assertEqual(provenance["schema"], "site-corpus-provenance")
        self.assertEqual(provenance["schema_version"], "1.0")
        self.assertEqual(provenance["audit"]["component_records"], 16)

    def test_corpus_is_dense_and_claim_cited(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        self.assertGreaterEqual(len(corpus.split()), 1400)
        citations = re.findall(r"\[MAIB-THROWBAG-2019,[^\]]+\]", corpus)
        self.assertGreaterEqual(len(citations), 25)
        for paragraph in [p for p in corpus.split("\n\n") if p and not p.startswith("#")]:
            if len(paragraph.split()) >= 35:
                self.assertIn("[MAIB-THROWBAG-2019,", paragraph, paragraph[:120])

    def test_required_qualitative_failure_analysis(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        required = [
            "Marine Accident Investigation Branch",
            "Report No. 2/2019",
            "several rope segments connected by thermal fusion",
            "A connection between segments was the location that separated",
            "joined samples separated under much less load than intact rope samples",
            "other rescue lines with comparable fused joints",
            "Random inspection also cannot repair an uncontrolled process",
            "Traceability, batch control, process validation, finished-product checking, and independent verification",
            "None is a complete substitute for the others",
            "This is a useful distinction between recurrence and incidence",
            "manufacturer-only quality control lacked third-party oversight",
            "It reports no conflict-of-interest statement, funding statement, or data-availability statement",
        ]
        for phrase in required:
            self.assertIn(phrase, corpus)

    def test_explicit_domain_boundary_and_no_transfer(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        lower = corpus.lower()
        disclaimer = (
            "This one rescue-product defect case does not validate a bondage-rope inspection "
            "or joining rule."
        )
        self.assertIn(disclaimer, corpus)
        self.assertEqual(lower.count("bondage"), 1)
        prohibited = [
            "human suspension",
            "body contact",
            "hardpoint",
            "upline",
            "anchor",
            "knot",
            "jute",
            "hemp",
            "natural fiber",
            "natural-fiber",
            "riber",
            "warrington",
            "british rowing",
            "rnli",
            "tti testing",
            "nfpa",
            "polypropylene",
            "minimum breaking load",
            "customer notification",
            "product recall",
            "safety bulletin",
        ]
        for phrase in prohibited:
            self.assertNotIn(phrase, lower, phrase)

    def test_no_measurement_ratio_material_or_recipe_leakage(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        lower = corpus.lower()
        measurement = re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:%|mm|cm|metres?|meters?|kn|kgf|times?)\b",
            re.IGNORECASE,
        )
        self.assertIsNone(measurement.search(corpus))
        without_citations = re.sub(r"\[MAIB-THROWBAG-2019,[^\]]+\]", "", corpus)
        numbers = re.findall(r"\b\d+(?:\.\d+)?(?:/\d+)?\b", without_citations)
        self.assertEqual(
            sorted(numbers),
            sorted(["2/2019", "24", "2018", "2019", "31", "2019", "3.0"]),
        )
        prohibited = [
            "twelve times",
            "twelfth",
            "12 times",
            "test ratio",
            "material was",
            "outer braid",
            "inner core",
            "heat setting",
            "fusion temperature",
            "melt the",
            "joining instructions",
            "rescue instructions",
            "throw the bag",
            "pull the casualty",
            "serial number",
            "inspection interval",
            "random 10",
        ]
        for phrase in prohibited:
            self.assertNotIn(phrase, lower, phrase)

    def test_report_records_audit_and_exclusions(self):
        report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
        required = [
            "All eight physical page images",
            "sixteen exclusion records",
            "three numbered figures",
            "one additional branded visual",
            "one unnumbered particulars table",
            "eleven credited or externally controlled source blocks",
            "Measurements, loads, counts, dimensions, ratios, test values",
            "Thermal-fusion or joining recipes",
            "Rescue-use instructions, emergency procedures, product notifications, recall advice",
        ]
        for phrase in required:
            self.assertIn(phrase, report)


if __name__ == "__main__":
    unittest.main()

