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
        actual = {str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file()}
        self.assertEqual(actual, expected)

    def test_article_identity(self):
        for value in (
            "Sanford B. Newman",
            "J. H. Curtiss",
            "A Statistical Analysis of Some Mechanical Properties of Manila Rope",
            "RP1847",
            "volume 39",
            "December 1947",
            "551–559",
            "Washington, May 26, 1947",
        ):
            self.assertIn(value, CORPUS)

    def test_verified_doi_and_rejected_doi(self):
        self.assertIn("10.6028/jres.039.039", CORPUS)
        self.assertIn("DOI ending in `.037` resolves to an unrelated uranium paper", CORPUS)
        audit = PROVENANCE["doi_audit"]
        self.assertEqual(audit["verified_identifier"]["doi"], "10.6028/jres.039.039")
        self.assertEqual(audit["verified_identifier"]["decision"], "accept")
        self.assertEqual(audit["provided_identifier"]["doi"], "10.6028/jres.039.037")
        self.assertEqual(audit["provided_identifier"]["decision"], "reject_as_unrelated")
        self.assertIn("uranium", audit["provided_identifier"]["crossref_title"].lower())

    def test_front_loaded_nontransfer_boundary(self):
        front = CORPUS[:2300].lower()
        for term in (
            "modern natural-fiber bondage rope",
            "knots",
            "wet or aged behavior",
            "rope care",
            "body contact",
            "uplines",
            "anchors",
            "human suspension",
            "safe load",
            "rating",
            "retirement rule",
            "not cyclic, impact, shock",
        ):
            self.assertIn(term, front)

    def test_dataset_provenance_and_counts(self):
        section = CORPUS.split("## What the dataset actually represents", 1)[1].split(
            "## What was measured", 1
        )[0]
        for phrase in (
            "more than 800 samples",
            "863 rope observations",
            "1938 through 1941",
            "government agency",
            "rope works and contractors",
            "administrative stream of purchased material",
            "not a probability sample",
            "far from homogeneous",
        ):
            self.assertIn(phrase, section)

    def test_three_strand_scope_and_four_strand_exclusion(self):
        for phrase in (
            "three-strand",
            "Some four-strand data existed",
            "not enough for statistical treatment",
            "No inference from the fitted three-strand relationships to four-strand rope is supported",
        ):
            self.assertIn(phrase, CORPUS)

    def test_variability_sources(self):
        for phrase in (
            "fiber quality",
            "method of fabrication",
            "previous treatment",
            "not separately randomized or modeled as recorded predictors",
        ):
            self.assertIn(phrase, CORPUS)

    def test_circumference_and_weight_measurement(self):
        section = CORPUS.split("### Circumference and weight", 1)[1].split(
            "### Static breaking endpoint", 1
        )[0]
        for phrase in (
            "unspliced specimen",
            "controlled temperature and relative-humidity atmosphere",
            "nominal-size-dependent preload",
            "single fiber was passed snugly around the loaded rope",
            "gauge length was marked while the preload remained applied",
            "after unloading, that section was cut out and weighed",
        ):
            self.assertIn(phrase, section)

    def test_static_breaking_endpoint(self):
        section = CORPUS.split("### Static breaking endpoint", 1)[1].split(
            "## Why 863 observations", 1
        )[0]
        for phrase in (
            "separate specimen with an eye splice at each end",
            "splices—not the full rope—were immersed briefly",
            "one of two machine types",
            "increased tension until maximum load was reached",
            "one or more strands failed",
            "maximum static tensile load under that apparatus and endpoint definition",
            "does not present a cross-machine equivalence study",
        ):
            self.assertIn(phrase, section)

    def test_large_n_does_not_fix_sampling(self):
        section = CORPUS.split("## Why 863 observations were not automatically representative", 1)[1].split(
            "## Logarithmic regression", 1
        )[0]
        for phrase in (
            "random sampling from a clearly defined homogeneous population",
            "those conditions did not hold",
            "deliberately simple mathematical description",
            "did not remove selection bias",
            "Sampling validity and model validity are separate from sample size",
        ):
            self.assertIn(phrase, section)

    def test_log_model_assumptions(self):
        section = CORPUS.split("## Logarithmic regression and its assumptions", 1)[1].split(
            "## Means, individual dispersion", 1
        )[0]
        for phrase in (
            "raw dispersion increased with the property mean",
            "base-ten logarithms",
            "unweighted least squares",
            "approximately stabilized variance",
            "mean log response is linear in log predictor",
            "log-scale standard deviation is constant",
            "geometric mean, not the arithmetic mean",
            "normal distribution on the log scale",
            "punched-card machinery",
        ):
            self.assertIn(phrase, section)

    def test_conditional_relationship_warning(self):
        self.assertIn(
            "a mean strength at fixed weight cannot generally be obtained by algebraically combining mean relationships fitted at fixed nominal diameter",
            CORPUS,
        )

    def test_tolerance_and_formal_precision_caveats(self):
        section = CORPUS.split("## Means, individual dispersion, and the paper’s tolerance limits", 1)[1].split(
            "## The anomalies", 1
        )[0]
        for phrase in (
            "modeled individual measurements, not uncertainty intervals for the sample means",
            "normally distributed",
            "not ratings, safe-load intervals, procurement rules, or modern acceptance limits",
            "usual random-sampling interpretation was unwarranted",
            "real accuracy of the curves",
            "largely a matter of faith",
        ):
            self.assertIn(phrase, section)

    def test_three_anomalies(self):
        section = CORPUS.split("## The anomalies are evidence, not cleanup noise", 1)[1].split(
            "## Limits on historical", 1
        )[0]
        for phrase in (
            "First, several nominal-size groups",
            "apparent substitution or misclassification",
            "Second, observations at fixed nominal diameter tended to cluster",
            "nonrandom sampling",
            "Third, one very small nominal-size subgroup",
            "poorly represented by the analytic fit",
            "increase the pooled residual spread",
        ):
            self.assertIn(phrase, section)

    def test_no_numeric_property_rating_or_tolerance_output(self):
        for pattern in (
            r"\b\d+(?:\.\d+)?\s*%",
            r"\b\d+(?:\.\d+)?\s*(?:lb|lbf|kn|n|kg|in\.?|ft\.?|psi)\b",
            r"(?i)(?:strength|safe[- ]?load|rating|tolerance)[^\n]{0,30}(?:=|of|from|between)\s*\d+(?:\.\d+)?",
        ):
            self.assertIsNone(re.search(pattern, CORPUS, flags=re.IGNORECASE))

    def test_no_formula_or_procurement_recipe(self):
        for token in ("log Y =", "Y=k", "10^", "Equation (", "\\int", "$$", "\\["):
            self.assertNotIn(token, CORPUS)
        for phrase in (
            "working load limit",
            "safe working load",
            "shall be accepted",
            "purchase this size",
            "retire when",
            "safety factor of",
        ):
            self.assertNotIn(phrase, CORPUS.lower())

    def test_no_source_components_or_bodies(self):
        self.assertNotIn("![", CORPUS)
        self.assertNotRegex(CORPUS, r"(?m)^\|.+\|$")
        forbidden = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".txt", ".html"}
        self.assertFalse(any(path.suffix.lower() in forbidden for path in ROOT.rglob("*")))
        self.assertTrue(all(not source["retained_in_repository"] for source in SOURCES))

    def test_claim_citation_density(self):
        body = CORPUS.split("## Public-domain adaptation and attribution", 1)[0]
        citations = re.findall(r"\[(?:NBS-1947|DOI-AUDIT|NIST-RIGHTS):[^\]]+\]", body)
        self.assertGreaterEqual(len(citations), 34)
        substantive = [
            paragraph
            for paragraph in re.split(r"\n\s*\n", body)
            if len(paragraph.split()) >= 34 and not paragraph.startswith("#")
        ]
        self.assertEqual(
            [paragraph for paragraph in substantive if not re.search(r"\[(?:NBS-1947|DOI-AUDIT|NIST-RIGHTS):", paragraph)],
            [],
        )

    def test_source_ledger(self):
        self.assertEqual(len(SOURCES), 4)
        self.assertEqual(
            {source["id"] for source in SOURCES},
            {"NBS-1947-PDF", "DOI-VERIFIED-039", "NIST-RIGHTS-FAQ", "NIST-TECH-SERIES-RIGHTS"},
        )
        by_id = {source["id"]: source for source in SOURCES}
        self.assertEqual(by_id["NBS-1947-PDF"]["pdf_pages"], 9)
        self.assertEqual(
            by_id["NBS-1947-PDF"]["sha256"],
            "5429d55bb31652b297703ba861f9b64aa43f3e6d3de6ae811ebc0e847fe67741",
        )
        self.assertEqual(
            by_id["DOI-VERIFIED-039"]["redirect_location"],
            "https://nvlpubs.nist.gov/nistpubs/jres/39/jresv39n6p551_A1b.pdf",
        )

    def test_rights_and_attribution(self):
        rights = PROVENANCE["rights_screen"]
        self.assertTrue(rights["federal_public_domain_us"])
        self.assertTrue(rights["foreign_rights_permission_noted"])
        self.assertTrue(rights["technical_series_policy_verified"])
        self.assertFalse(rights["legal_advice"])
        notice = CORPUS.split("## Public-domain adaptation and attribution", 1)[1]
        for phrase in (
            "Republished courtesy of the National Institute of Standards and Technology",
            "Not copyrightable in the United States",
            "changes wording, selection, organization, and emphasis",
            "No endorsement",
        ):
            self.assertIn(phrase, notice)

    def test_document_and_extraction_audit(self):
        document = PROVENANCE["document_audit"]
        self.assertEqual(document["pdf_pages"], 9)
        self.assertEqual(document["page_images"], 9)
        self.assertEqual(document["page_pixels_per_inch"], 600)
        self.assertTrue(document["searchable_ocr_layer"])
        extraction = PROVENANCE["extraction_audit"]
        self.assertEqual(extraction["layout_text_bytes"], 72737)
        self.assertEqual(extraction["layout_whitespace_words"], 6304)
        self.assertEqual(
            extraction["layout_text_sha256"],
            "03842ed99c5d6f7ceafc1600af90b11960e8076108ac0efa03adf1e6f4e1b56a",
        )
        self.assertFalse(extraction["retained_in_repository"])

    def test_complete_page_dispositions(self):
        review = PROVENANCE["manual_review"]
        self.assertTrue(review["completed"])
        self.assertEqual(review["reviewed_pages"], 9)
        self.assertEqual(review["page_set"], list(range(1, 10)))
        self.assertEqual(len(DISPOSITIONS), 9)
        self.assertEqual({item["page"] for item in DISPOSITIONS}, set(range(1, 10)))
        self.assertEqual({item["journal_page"] for item in DISPOSITIONS}, set(range(551, 560)))
        by_page = {item["page"]: item for item in DISPOSITIONS}
        self.assertEqual(by_page[6]["decision"], "exclude")
        self.assertEqual(by_page[7]["decision"], "exclude")
        self.assertEqual(by_page[9]["decision"], "identity_only")

    def test_report_page_review(self):
        for page in range(1, 10):
            self.assertIn(f"PDF page {page} / journal page {page + 550}", REPORT)

    def test_section_inventory(self):
        sections = PROVENANCE["section_inventory"]
        self.assertEqual(len(sections), 6)
        self.assertIn("II. Methods of Test", sections)
        self.assertIn("III. Methods of Statistical Analysis", sections)
        self.assertIn("IV. Results and Discussion", sections)

    def test_readme_scope_doi_and_command(self):
        self.assertIn("training-source-only", README)
        self.assertIn("10.6028/jres.039.039", README)
        self.assertIn("`.037` DOI is documented as an unrelated-paper binding", README)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover", README)
        self.assertIn("Do not derive validation, holdout, evaluation, OOD, or probe", README)

    def test_split_hygiene(self):
        split = MANIFEST["split_hygiene"]
        self.assertEqual(split["allowed"], ["training_source"])
        self.assertEqual(
            set(split["forbidden"]), {"validation", "holdout", "evaluation", "ood", "probe"}
        )

    def test_manifest_sources_match(self):
        self.assertEqual(set(MANIFEST["source_ids"]), {source["id"] for source in SOURCES})

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
