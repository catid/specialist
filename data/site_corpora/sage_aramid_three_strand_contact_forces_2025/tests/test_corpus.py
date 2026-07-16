import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = {
    "CORPUS.md",
    "README.md",
    "REPORT.md",
    "dispositions.jsonl",
    "manifest.json",
    "sources.jsonl",
    "source_snapshot/provenance.json",
    "tests/test_corpus.py",
}


def jsonl(path: Path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CorpusPackageTests(unittest.TestCase):
    def test_required_files_exist(self):
        actual = {
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        self.assertTrue(REQUIRED_FILES.issubset(actual))
        self.assertFalse(any(path.suffix.lower() == ".pdf" for path in ROOT.rglob("*")))

    def test_source_record_and_verified_rights(self):
        records = jsonl(ROOT / "sources.jsonl")
        self.assertEqual(len(records), 1)
        source = records[0]
        self.assertEqual(source["source_id"], "UTW-SAGE-2025")
        self.assertEqual(source["page_count"], 17)
        self.assertEqual(source["license_spdx"], "CC-BY-4.0")
        self.assertEqual(source["license_url"], "https://creativecommons.org/licenses/by/4.0/")
        self.assertTrue(source["landing_url"].startswith("https://research.utwente.nl/"))
        self.assertTrue(source["pdf_url"].startswith("https://ris.utwente.nl/"))
        self.assertRegex(source["pdf_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(source["pdf_sha512"], r"^[0-9a-f]{128}$")
        self.assertFalse(source["local_source_included"])

    def test_provenance_matches_source_record(self):
        source = jsonl(ROOT / "sources.jsonl")[0]
        provenance = json.loads((ROOT / "source_snapshot/provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["source_id"], source["source_id"])
        self.assertEqual(provenance["retrieval"]["pdf"]["pages"], 17)
        self.assertEqual(provenance["retrieval"]["pdf"]["sha256"], source["pdf_sha256"])
        self.assertEqual(provenance["retrieval"]["pdf"]["sha512"], source["pdf_sha512"])
        self.assertEqual(provenance["retrieval"]["landing"]["sha256"], source["landing_sha256"])
        self.assertEqual(provenance["rights"]["status"], "verified")

    def test_exactly_one_disposition_per_page(self):
        rows = jsonl(ROOT / "dispositions.jsonl")
        self.assertEqual(len(rows), 17)
        self.assertEqual([row["page"] for row in rows], list(range(1, 18)))
        self.assertEqual(len({row["page"] for row in rows}), 17)
        self.assertTrue(all(row["source_id"] == "UTW-SAGE-2025" for row in rows))
        self.assertTrue(all(row["decision"] in {"partial", "exclude"} for row in rows))
        self.assertEqual(rows[-1]["decision"], "exclude")
        self.assertEqual(rows[-1]["retained"], [])

    def test_manual_inconsistency_resolution_is_recorded(self):
        page_sixteen = jsonl(ROOT / "dispositions.jsonl")[15]
        excluded = " ".join(page_sixteen["excluded"]).lower()
        self.assertIn("contradictory contact-width sentence", excluded)
        report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("Manual resolution of a source inconsistency", report)

    def test_corpus_is_substantial_and_cited(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        words = re.findall(r"\b[\w’'-]+\b", corpus)
        self.assertGreaterEqual(len(words), 1500)
        citations = re.findall(r"\[UTW-SAGE-2025, pp?\. [^\]]+\]", corpus)
        self.assertGreaterEqual(len(citations), 25)
        self.assertNotRegex(corpus, r"\[[A-Z0-9-]+, pp?\. [^\]]+\](?!\))")

    def test_required_scientific_content_and_disclosures(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        required = [
            "Twaron 2200",
            "laid three-strand rope contains three twisted strands",
            "strand is assembled from twisted yarns",
            "yarn is assembled from many filaments",
            "Pressure-sensitive film",
            "contact pressure",
            "contact width",
            "effective strand twist",
            "unadjusted analytical model",
            "available from the corresponding author upon reasonable request",
            "Netherlands Organization for Scientific Research",
            "Teijin Aramid BV",
            "Vincent van Bommel",
            "Dr. Bo Cornelissen",
            "Oday Allan reported financial support",
        ]
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, corpus)

    def test_quantitative_and_domain_exclusions(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        prohibited_patterns = {
            "pressure unit": r"\bMPa\b",
            "force unit": r"\bkN\b|\bN/tex\b|\bN/m\b",
            "percentage": r"\d\s*%",
            "twist rate": r"rotations?/m",
            "equation environment": r"\\begin\{|\\frac\{|\$\$",
            "markdown image": r"!\[[^\]]*\]\(",
            "unsafe transfer domains": r"\b(?:bondage|shibari|kinbaku|hardpoint|upline|suspension|jute|hemp)\b",
            "operational claim": r"\b(?:working load|load rating|service life|retirement criterion|safe to use)\b",
        }
        for label, pattern in prohibited_patterns.items():
            with self.subTest(label=label):
                self.assertNotRegex(corpus, re.compile(pattern, re.IGNORECASE))

    def test_model_and_transfer_limits_are_explicit(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        self.assertIn("dataset-specific correction, not a universal predictor", corpus)
        self.assertIn("should not be generalized to other rope materials or constructions", corpus)
        self.assertIn("projection was not a direct measurement", corpus)

    def test_manifest_integrity(self):
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "specialist-corpus-manifest-v1")
        expected = REQUIRED_FILES - {"manifest.json"}
        records = {record["path"]: record for record in manifest["files"]}
        self.assertEqual(set(records), expected)
        for relative, record in records.items():
            path = ROOT / relative
            with self.subTest(path=relative):
                self.assertEqual(record["bytes"], path.stat().st_size)
                self.assertEqual(record["sha256"], sha256(path))


if __name__ == "__main__":
    unittest.main()
