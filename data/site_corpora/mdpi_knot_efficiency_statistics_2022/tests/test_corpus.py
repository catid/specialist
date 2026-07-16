import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
REPORT = (ROOT / "REPORT.md").read_text(encoding="utf-8")
MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
PROVENANCE = json.loads(
    (ROOT / "source_snapshot" / "provenance.json").read_text(encoding="utf-8")
)


def jsonl(name):
    return [
        json.loads(line)
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


SOURCES = jsonl("sources.jsonl")
DISPOSITIONS = jsonl("dispositions.jsonl")


class CorpusTests(unittest.TestCase):
    def test_exact_file_inventory(self):
        expected = {
            "CORPUS.md",
            "README.md",
            "REPORT.md",
            "dispositions.jsonl",
            "manifest.json",
            "source_snapshot/provenance.json",
            "sources.jsonl",
            "tests/test_corpus.py",
        }
        actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()}
        self.assertEqual(actual, expected)

    def test_article_identity(self):
        for value in (
            "Ján Šimon",
            "Branislav Ftorek",
            "Basic Statistical Properties of the Knot Efficiency",
            "September 15, 2022",
            "August 14, 2022",
            "September 10, 2022",
            "10.3390/sym14091926",
        ):
            self.assertIn(value, CORPUS)

    def test_front_loaded_nontransfer_boundary(self):
        front = CORPUS[:1900].lower()
        for term in (
            "dynamic",
            "cyclic",
            "wet",
            "aged",
            "damaged",
            "natural-fiber",
            "bondage",
            "body contact",
            "uplines",
            "human suspension",
            "does not choose a knot",
            "safety factor",
            "safe load",
        ):
            self.assertIn(term, front)

    def test_measured_vs_calculated(self):
        for phrase in (
            "directly observed destructive-test outcomes",
            "η = X/Y",
            "is calculated from them; it is not directly measured",
            "X and Y therefore have their own distributions",
        ):
            self.assertIn(phrase, CORPUS)

    def test_independent_draw_assumption(self):
        for phrase in (
            "independent draws",
            "different pieces are used",
            "pieces are shuffled",
            "design and modeling assumption",
        ):
            self.assertIn(phrase, CORPUS)

    def test_prior_data_provenance(self):
        section = CORPUS.split("## Provenance of the normal model", 1)[1].split(
            "## Two common", 1
        )[0]
        for phrase in (
            "roughly 200 straight-rope breaking tests",
            "roughly 80 knotted-rope breaking tests",
            "Kolmogorov–Smirnov",
            "Shapiro–Wilk",
            "cited earlier work, not a new 2022 experimental campaign",
        ):
            self.assertIn(phrase, section)

    def test_failure_to_reject_is_not_proof(self):
        for phrase in (
            "Failure to reject a normal model is not proof",
            "test power",
            "Other distributions may also be compatible",
            "supported working approximation, not an established physical law",
        ):
            self.assertIn(phrase, CORPUS)

    def test_single_pair_problem(self):
        section = CORPUS.split("### One knotted break", 1)[1].split(
            "### A ratio", 1
        )[0]
        self.assertIn("one draw from a potentially broad distribution", section)
        self.assertIn("different pieces", section)
        self.assertIn("cannot estimate either source distribution", section)

    def test_ratio_of_means_problem(self):
        section = CORPUS.split("### A ratio of two small-sample means", 1)[1].split(
            "## From two", 1
        )[0]
        self.assertIn("expectation of a quotient is not the quotient of expectations", section)
        self.assertIn("Discarding the two input variances", section)
        self.assertIn("finite-sample estimation error", section)

    def test_pdf_plain_language(self):
        for phrase in (
            "probability density function (PDF)",
            "height at one point is not itself a probability",
            "Probability comes from area over an interval",
            "skew, multiple peaks, or long tails",
        ):
            self.assertIn(phrase, CORPUS)

    def test_cdf_plain_language(self):
        for phrase in (
            "cumulative distribution function (CDF)",
            "at or below a threshold",
            "Subtracting CDF values",
            "require numerical evaluation",
        ):
            self.assertIn(phrase, CORPUS)

    def test_nonphysical_tail_and_truncation(self):
        for phrase in (
            "tiny mathematical probability to negative breaking strength",
            "denominator is arbitrarily close to zero",
            "does not have a finite global mean or variance",
            "or conditional summaries",
            "conditioning changes the population",
        ):
            self.assertIn(phrase, CORPUS)

    def test_central_tendencies(self):
        for phrase in (
            "mode",
            "median",
            "truncated mean",
            "need not coincide",
            "not generally the exact ratio distribution’s mean",
        ):
            self.assertIn(phrase, CORPUS)

    def test_dispersion(self):
        for phrase in (
            "source variances influence the ratio curve",
            "truncated variance",
            "center without spread invites false precision",
        ):
            self.assertIn(phrase, CORPUS)

    def test_tolerance_interval_caveat(self):
        section = CORPUS.split("## Dispersion and the paper’s “tolerance intervals”", 1)[1].split(
            "## Exact model", 1
        )[0]
        for phrase in (
            "equal-tail central ranges",
            "quantiles of the modeled efficiency distribution",
            "not confidence intervals",
            "not engineering tolerances",
            "safe-load intervals",
        ):
            self.assertIn(phrase, section)

    def test_solid_approximation_scope(self):
        for phrase in (
            "restricts attention to a stated practical parameter sector",
            "closed-form CDF",
            "truncated mean and variance still require numerical integration",
            "upper bounds for PDF, CDF, truncated-mean, and truncated-variance errors",
            "without verifying its parameter assumptions",
        ):
            self.assertIn(phrase, CORPUS)

    def test_normal_approximation_is_case_dependent(self):
        for phrase in (
            "case-dependent",
            "worn-rope example is broad and asymmetric",
            "new-rope example is narrow and symmetric",
            "not a controlled new-versus-worn aging experiment",
        ):
            self.assertIn(phrase, CORPUS)

    def test_no_numeric_efficiencies_or_ranges(self):
        self.assertIsNone(re.search(r"\b\d+(?:\.\d+)?\s*%", CORPUS))
        self.assertNotRegex(
            CORPUS,
            r"(?i)(?:efficienc(?:y|ies)|retained strength)[^\n]{0,30}"
            r"(?:=|of|from|between)\s*\d+(?:\.\d+)?",
        )

    def test_no_load_values_ratings_or_choices(self):
        self.assertIsNone(re.search(r"\b\d+(?:\.\d+)?\s*(?:kN|lbf|N)\b", CORPUS))
        for phrase in (
            "working load limit",
            "safe working load",
            "recommended knot",
            "choose the",
            "safety factor of",
        ):
            self.assertNotIn(phrase, CORPUS.lower())

    def test_no_operational_equation_set(self):
        for token in ("\\int", "erf(", "Theorem 1", "Equation (", "$$", "\\["):
            self.assertNotIn(token, CORPUS)
        self.assertEqual(CORPUS.count("η = X/Y"), 1)

    def test_no_source_visual_or_tabular_content(self):
        self.assertNotIn("![", CORPUS)
        self.assertNotRegex(CORPUS, r"(?m)^\|.+\|$")
        forbidden = {".pdf", ".xml", ".html", ".png", ".jpg", ".jpeg", ".webp"}
        self.assertFalse(any(p.suffix.lower() in forbidden for p in ROOT.rglob("*")))
        self.assertTrue(all(not s["retained_in_repository"] for s in SOURCES))

    def test_claim_citation_density(self):
        body = CORPUS.split("## Attribution and adaptation notice", 1)[0]
        citations = re.findall(r"\[SYM-2022:[^\]]+\]", body)
        self.assertGreaterEqual(len(citations), 35)
        substantive = [
            p
            for p in re.split(r"\n\s*\n", body)
            if len(p.split()) >= 34 and not p.startswith("#")
        ]
        self.assertEqual([p for p in substantive if "[SYM-2022:" not in p], [])

    def test_source_ledger(self):
        self.assertEqual(len(SOURCES), 4)
        self.assertEqual(
            {s["id"] for s in SOURCES},
            {"DOI-RESOLVER", "SYM-2022-PDF", "SYM-2022-XML", "CC-BY-4.0"},
        )
        by_id = {s["id"]: s for s in SOURCES}
        self.assertEqual(by_id["SYM-2022-PDF"]["pdf_pages"], 25)
        self.assertEqual(
            by_id["SYM-2022-PDF"]["sha256"],
            "2094e47d703011b122d4ff159ebcbcabbe8d108151c1da3476ae1c775ad8240e",
        )
        self.assertEqual(
            by_id["SYM-2022-XML"]["sha256"],
            "ea297f445f8cb58decdbb95efeb988a69208062eb1988509a68af39e6616bf9f",
        )

    def test_landing_access_attempt_is_documented(self):
        attempt = PROVENANCE["access_attempts"][0]
        self.assertEqual(attempt["status"], 403)
        self.assertFalse(attempt["used_as_content_source"])
        self.assertEqual(attempt["url"], "https://www.mdpi.com/2073-8994/14/9/1926")

    def test_rights_and_attribution(self):
        rights = PROVENANCE["rights_screen"]
        self.assertTrue(rights["article_license_notice_in_pdf"])
        self.assertTrue(rights["article_license_notice_in_xml"])
        self.assertFalse(rights["legal_advice"])
        notice = CORPUS.split("## Attribution and adaptation notice", 1)[1]
        for value in (
            "Ján Šimon",
            "Branislav Ftorek",
            "10.3390/sym14091926",
            "Creative Commons Attribution 4.0 International",
            "changes wording, selection, ordering, and emphasis",
            "No endorsement",
        ):
            self.assertIn(value, notice)

    def test_manual_page_review_and_components(self):
        self.assertTrue(PROVENANCE["manual_review"]["completed"])
        audit = PROVENANCE["extraction_audit"]
        self.assertEqual(audit["pdf_pages"], 25)
        self.assertEqual(audit["source_numbered_figures"], 9)
        self.assertEqual(audit["source_numbered_tables"], 4)
        self.assertEqual(audit["raster_image_xobjects"], 2)
        for group in ("1–3", "4–5", "6–8", "9–13", "14–16", "17–19", "20–24"):
            self.assertIn(f"PDF pages {group}", REPORT)
        self.assertIn("PDF page 25", REPORT)

    def test_section_inventory(self):
        sections = PROVENANCE["section_inventory"]
        self.assertEqual(len(sections), 10)
        self.assertIn("2.5. Solid Approximation Error Management", sections)
        self.assertIn("4. Conclusions", sections)

    def test_disposition_inventory_is_section_and_page_level(self):
        self.assertGreaterEqual(len(DISPOSITIONS), 36)
        self.assertTrue(all("pages" in d and "section" in d for d in DISPOSITIONS))
        by_component = {d["component"]: d["decision"] for d in DISPOSITIONS}
        self.assertEqual(by_component["One knotted and one straight break strategy"], "include_critique")
        self.assertEqual(by_component["Figures 1-9, Tables 1-4, equations and proofs"], "exclude")
        self.assertEqual(by_component["Cross-domain and operational transfer"], "exclude")

    def test_readme_scope_and_command(self):
        self.assertIn("training-source-only", README)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover", README)
        self.assertIn("Do not derive validation, holdout, evaluation, OOD, or probe", README)

    def test_split_hygiene(self):
        split = MANIFEST["split_hygiene"]
        self.assertEqual(split["allowed"], ["training_source"])
        self.assertEqual(
            set(split["forbidden"]), {"validation", "holdout", "evaluation", "ood", "probe"}
        )

    def test_manifest_sources_match(self):
        self.assertEqual(set(MANIFEST["source_ids"]), {s["id"] for s in SOURCES})

    def test_manifest_hashes(self):
        expected = {
            "CORPUS.md",
            "README.md",
            "REPORT.md",
            "dispositions.jsonl",
            "source_snapshot/provenance.json",
            "sources.jsonl",
            "tests/test_corpus.py",
        }
        self.assertEqual(set(MANIFEST["files"]), expected)
        for relative, metadata in MANIFEST["files"].items():
            data = (ROOT / relative).read_bytes()
            self.assertEqual(metadata["bytes"], len(data))
            self.assertEqual(metadata["sha256"], hashlib.sha256(data).hexdigest())

    def test_manifest_avoids_self_hash_cycle(self):
        self.assertNotIn("manifest.json", MANIFEST["files"])


if __name__ == "__main__":
    unittest.main()
