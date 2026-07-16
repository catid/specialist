import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def jsonl(name):
    return [
        json.loads(line)
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class CorpusPackageTests(unittest.TestCase):
    def test_required_files_exist(self):
        required = {
            "CORPUS.md",
            "README.md",
            "REPORT.md",
            "sources.jsonl",
            "dispositions.jsonl",
            "components.jsonl",
            "manifest.json",
            "source_snapshot/provenance.json",
            "tests/test_corpus.py",
        }
        self.assertEqual(
            set(json.loads((ROOT / "manifest.json").read_text())["files"]),
            required - {"manifest.json"},
        )
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_manifest_hashes(self):
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["algorithm"], "sha256")
        self.assertTrue(manifest["direct_training_ready"])
        for relative, expected in manifest["files"].items():
            digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, relative)

    def test_exact_source_and_rights(self):
        sources = jsonl("sources.jsonl")
        self.assertEqual(len(sources), 1)
        source = sources[0]
        self.assertEqual(source["source_id"], "PHM-CM-2017")
        self.assertEqual(
            source["title"],
            "Condition Monitoring Technologies for Synthetic Fiber Ropes - a Review",
        )
        self.assertEqual(
            source["authors"], ["Espen Oland", "Rune Schlanbusch", "Shaun Falconer"]
        )
        self.assertEqual(source["publication_year"], 2017)
        self.assertEqual(source["doi"], "10.36001/ijphm.2017.v8i2.2619")
        self.assertEqual(source["license_spdx"], "CC-BY-3.0-US")
        self.assertEqual(source["page_count"], 14)
        self.assertEqual(
            source["body_sha256"],
            "f267e478801fca291139ab8da89b2069a9dcdd037b5385a4af65c46517d2e317",
        )
        self.assertTrue(source["landing_url"].startswith("https://papers.phmsociety.org/"))
        self.assertTrue(source["body_url"].startswith("https://papers.phmsociety.org/"))
        self.assertTrue(source["manual_page_audit"])
        self.assertTrue(source["manual_component_audit"])
        self.assertTrue(source["direct_training_ready"])

    def test_one_manual_disposition_per_page(self):
        rows = jsonl("dispositions.jsonl")
        self.assertEqual(len(rows), 14)
        self.assertEqual([row["page"] for row in rows], list(range(1, 15)))
        self.assertTrue(all(row["audit"] == "manual text and visual review" for row in rows))
        self.assertEqual(
            {row["page"] for row in rows if row["decision"] == "exclude"},
            {7, 12, 13, 14},
        )
        self.assertTrue(all(row["decision"] in {"partial", "exclude"} for row in rows))

    def test_every_visual_component_is_excluded(self):
        rows = jsonl("components.jsonl")
        self.assertEqual(len(rows), 19)
        self.assertEqual([row["component"] for row in rows], [f"Figure {n}" for n in range(1, 20)])
        self.assertTrue(all(row["type"] == "figure" for row in rows))
        self.assertTrue(all(row["decision"] == "exclude" for row in rows))
        self.assertEqual(
            [row["page"] for row in rows],
            [1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 9, 9, 9, 10, 10, 10],
        )
        provenance = json.loads((ROOT / "source_snapshot/provenance.json").read_text())
        self.assertEqual(provenance["audit"]["numbered_figures"], 19)
        self.assertEqual(provenance["audit"]["tables"], 0)

    def test_corpus_is_dense_and_claim_cited(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        self.assertGreaterEqual(len(corpus.split()), 1700)
        citations = re.findall(r"\[PHM-CM-2017, pp?\.[^\]]+\]", corpus)
        self.assertGreaterEqual(len(citations), 40)
        for paragraph in [p for p in corpus.split("\n\n") if p and not p.startswith("#")]:
            if paragraph.startswith("This is offshore") or len(paragraph.split()) >= 30:
                self.assertIn("[PHM-CM-2017,", paragraph, paragraph[:100])

    def test_required_knowledge_and_transparency(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        required = [
            "Espen Oland, Rune Schlanbusch, and Shaun Falconer",
            "10.36001/ijphm.2017.v8i2.2619",
            "filaments are twisted into yarn",
            "strands form sub-ropes",
            "embedded and nonembedded",
            "continuous versus discrete",
            "Computer vision and geometry",
            "Acoustic, vibration, and wave propagation",
            "No sensing class in the review covers every failure mode",
            "represents the state of the literature surveyed in 2017",
            "Norwegian Research Council",
            "SFI Offshore Mechatronics",
            "does not report a conflict-of-interest statement or a data-availability statement",
        ]
        for phrase in required:
            self.assertIn(phrase, corpus)

    def test_domain_boundary_and_no_operational_transfer(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        lower = corpus.lower()
        disclaimer = (
            "This is offshore synthetic-fiber-rope condition-monitoring evidence, "
            "not visual bondage-rope inspection advice."
        )
        self.assertIn(disclaimer, corpus)
        self.assertEqual(lower.count("bondage"), 1)
        prohibited = [
            "jute",
            "hemp",
            "natural fiber",
            "natural-fiber",
            "human suspension",
            "body loading",
            "hardpoint",
            "upline",
            "anchor system",
            "remaining useful life",
            "retirement rule",
            "discard rule",
            "repair rule",
            "national oilwell",
            "macgregor",
            "samson rope",
            "cortland",
            "dyneema",
            "det norske veritas",
        ]
        for phrase in prohibited:
            self.assertNotIn(phrase, lower, phrase)
        measurement = re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:%|°\s*c|kn|mpa|gpa|mm|cm|m/s|hz|khz|mhz|cycles?|hours?)\b",
            re.IGNORECASE,
        )
        self.assertIsNone(measurement.search(corpus))

    def test_report_records_exclusion_policy(self):
        report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
        for phrase in [
            "All nineteen figures and their captions were excluded",
            "Patents, standards, vendors, product identities, trademarks",
            "Numerical values, equations, thresholds",
            "Discard, remaining-life, repair, retirement",
            "Any transfer to body loading, human suspension, anchors, uplines",
        ]:
            self.assertIn(phrase, report)


if __name__ == "__main__":
    unittest.main()

