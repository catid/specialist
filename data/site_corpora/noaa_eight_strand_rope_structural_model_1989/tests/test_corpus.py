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

    def test_identity_and_publication_binding(self):
        for value in (
            "Youjiang Wang",
            "Stanley Backer",
            "Structural Modeling Of The Tensile Behavior Of Eight-Strand Ropes",
            "MITSG 89-28",
            "MIT-T-89-002",
            "NA86AA-D-SG089",
            "RT-11",
            "1989",
        ):
            self.assertIn(value, CORPUS)

    def test_front_loaded_scope_and_nontransfer(self):
        front = CORPUS[:3000].lower()
        for phrase in (
            "eight-strand plaited synthetic rope",
            "qualitative",
            "natural-fiber rope",
            "knots",
            "splices",
            "rope care",
            "retirement criteria",
            "bondage",
            "body contact",
            "uplines",
            "anchors",
            "hardpoints",
            "human suspension",
            "no working-load rule",
        ):
            self.assertIn(phrase, front)

    def test_hierarchy_is_explicit(self):
        section = CORPUS.split("## Rope, strand, and plied-yarn hierarchy", 1)[1].split(
            "## From unloaded geometry", 1
        )[0]
        for phrase in (
            "three nested geometric levels",
            "eight plaited strands",
            "core, sublayer, and surface-layer positions",
            "rotating offset",
            "four symmetry-related pairs",
            "periodic sinusoidal curves",
        ):
            self.assertIn(phrase, section)

    def test_stretched_geometry_and_equilibrium(self):
        section = CORPUS.split("## From unloaded geometry to stretched geometry", 1)[1].split(
            "## Two friction endpoints", 1
        )[0]
        for phrase in (
            "mapped length must match the imposed overall extension",
            "summed axial strand force must remain in equilibrium",
            "local coordinate to material cross-sections",
            "differentiation amplified high-frequency noise",
            "position-dependent axial-strain mapping",
            "piecewise-linear numerical approximation",
            "compromise between accuracy and computational cost",
        ):
            self.assertIn(phrase, section)

    def test_two_friction_endpoints_and_scale_distinction(self):
        section = CORPUS.split("## Two friction endpoints inside each strand", 1)[1].split(
            "## How constituent behavior", 1
        )[0]
        for phrase in (
            "between adjacent plied yarns within a strand",
            "distinct from",
            "no-friction endpoint",
            "constant along a given plied yarn",
            "no-relative-motion endpoint",
            "local strain varies along the yarn",
            "core layer through the sublayer to the surface layer",
            "idealized bounds",
            "no calibrated transition",
        ):
            self.assertIn(phrase.lower(), section.lower())

    def test_load_aggregation_and_predamage_limit(self):
        section = CORPUS.split("## How constituent behavior is aggregated", 1)[1].split(
            "## Why strands press", 1
        )[0]
        for phrase in (
            "resolves the corresponding constituent force into the rope-axis direction",
            "sums yarn contributions",
            "combines symmetry-related strand contributions",
            "explicitly unrealistic damage treatment",
            "before the initiation of load-induced damage",
            "does not model progressive failure",
            "nothing here can be used to predict breaking load, fatigue life, residual strength, or retirement timing",
        ):
            self.assertIn(phrase, section)

    def test_contact_pressure_is_bounded(self):
        section = CORPUS.split("## Why strands press on one another", 1)[1].split(
            "## Why strands move", 1
        )[0]
        for phrase in (
            "not locally self-equilibrating",
            "does not solve a complete pressure field",
            "assumptions about contact area and transverse compressive behavior",
            "idealized cylinder-contact analogy",
            "crossing strands",
            "parallel strands on the rope exterior",
            "parallel strands inside the rope",
            "principal high-pressure regions",
            "not a quantified wear or life law",
        ):
            self.assertIn(phrase, section)

    def test_relative_motion_and_hotspot_logic(self):
        section = CORPUS.split("## Why strands move relative to one another", 1)[1].split(
            "## Lateral contraction", 1
        )[0]
        for phrase in (
            "sliding along a neighboring strand",
            "rotation about changing pivot points",
            "same three contact classes",
            "Pressure and motion are complementary, not interchangeable",
            "Pressure supplies normal contact",
            "relative motion supplies rubbing",
            "internal-abrasion hotspots",
            "no controlled wear-validation data",
        ):
            self.assertIn(phrase, section)

    def test_lateral_contraction_is_controlling_limitation(self):
        section = CORPUS.split("## Lateral contraction controls the result", 1)[1].split(
            "## Relationship", 1
        )[0]
        for phrase in (
            "affects rope radius, strand radius, path amplitudes, local curvature, constituent strain",
            "rope and strand may contract differently",
            "do not establish which describes the studied ropes",
            "strongly influenced by this assumed ratio",
            "can only be qualitative",
            "downstream of the same geometry",
        ):
            self.assertIn(phrase, section)

    def test_1990_relationship_is_nonconflating(self):
        section = CORPUS.split("## Relationship to the separate 1990 deterioration report", 1)[1].split(
            "## Durable scientific lessons", 1
        )[0]
        for phrase in (
            "first step",
            "stops before damage evolution",
            "double-braided synthetic rope",
            "not eight-strand plaited rope",
            "attributes its inherited structural model to Seo’s earlier dissertation",
            "programmatically and conceptually",
            "not interchangeable stages",
            "would require a new, explicitly validated model",
        ):
            self.assertIn(phrase, section)

    def test_no_numeric_model_parameters_or_calculators(self):
        for forbidden in (
            "0.3",
            "0.6",
            "120 mm",
            "29 mm",
            "11,490",
            "turns/period",
            "nineteen",
            "19 plied",
            "Equation (",
            "Eq. ",
            "\\int",
            "$$",
            "\\[",
        ):
            self.assertNotIn(forbidden, CORPUS)
        for pattern in (
            r"\b\d+(?:\.\d+)?\s*%",
            r"\b\d+(?:\.\d+)?\s*(?:mm|cm|mpa|kpa|kn|lbf)\b",
            r"\b\d+(?:\.\d+)?\s*(?:cycles?|turns?)\b",
        ):
            self.assertIsNone(re.search(pattern, CORPUS, flags=re.IGNORECASE))

    def test_no_operational_or_design_recipe(self):
        for phrase in (
            "working load limit",
            "safe working load",
            "safety factor of",
            "retire when",
            "inspection interval of",
            "tie the",
            "splice the",
            "blend the fibers",
            "use this pressure",
        ):
            self.assertNotIn(phrase, CORPUS.lower())

    def test_no_source_visual_tabular_or_scan_content(self):
        self.assertNotIn("![", CORPUS)
        self.assertNotRegex(CORPUS, r"(?m)^\|.+\|$")
        forbidden = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".txt", ".html"}
        self.assertFalse(any(path.suffix.lower() in forbidden for path in ROOT.rglob("*")))
        self.assertTrue(all(not source["retained_in_repository"] for source in SOURCES))

    def test_claim_citation_density(self):
        body = CORPUS.split("## Public-domain adaptation and source notice", 1)[0]
        citations = re.findall(r"\[(?:NOAA-1989|NOAA-IR-42461|RELATED-NOAA-1990):[^\]]+\]", body)
        self.assertGreaterEqual(len(citations), 40)
        substantive = [
            paragraph
            for paragraph in re.split(r"\n\s*\n", body)
            if len(paragraph.split()) >= 34 and not paragraph.startswith("#")
        ]
        uncited = [
            paragraph
            for paragraph in substantive
            if not re.search(r"\[(?:NOAA-1989|NOAA-IR-42461|RELATED-NOAA-1990):", paragraph)
        ]
        self.assertEqual(uncited, [])

    def test_source_ledger_and_exact_sha512_binding(self):
        self.assertEqual(len(SOURCES), 3)
        by_id = {source["id"]: source for source in SOURCES}
        self.assertEqual(
            set(by_id),
            {"NOAA-IR-42461", "NOAA-1989-PDF", "RELATED-NOAA-1990-PDF"},
        )
        current = by_id["NOAA-1989-PDF"]
        self.assertEqual(current["pdf_pages"], 51)
        self.assertEqual(
            current["sha512"],
            "d791640336f80bfb98abe5863476bb366502ca26744b4a59fb4ee09422c3f9ade3c50e5c3471671cc4e0c98d913a0734b9652f1c718f63cd5c586ed78a6ed85e",
        )
        self.assertTrue(current["record_checksum_match"])
        related = by_id["RELATED-NOAA-1990-PDF"]
        self.assertEqual(related["role"][0], "relationship_only")
        self.assertNotIn("primary_scientific_content", related["role"])

    def test_public_domain_rights_screen(self):
        rights = PROVENANCE["rights_screen"]
        self.assertEqual(rights["record_designation"], "Public Domain")
        self.assertEqual(rights["main_document_checksum_algorithm"], "sha512")
        self.assertTrue(rights["main_document_checksum_bound"])
        self.assertFalse(rights["source_bodies_retained"])
        self.assertFalse(rights["legal_advice"])
        notice = CORPUS.split("## Public-domain adaptation and source notice", 1)[1]
        for phrase in ("Public Domain", "changes wording", "No endorsement"):
            self.assertIn(phrase, notice)

    def test_hybrid_scan_and_dual_ocr_audit(self):
        document = PROVENANCE["document_audit"]
        self.assertEqual(document["pdf_pages"], 51)
        self.assertEqual(document["page_images"], 51)
        self.assertTrue(document["hybrid_scan_with_hidden_text"])
        self.assertEqual(document["page_pixels_per_inch"], 300)
        self.assertEqual(document["embedded_text_characters"], 56921)
        ocr = PROVENANCE["ocr_audit"]
        self.assertEqual(ocr["embedded_layer"]["outputs"], 51)
        self.assertEqual(ocr["independent_tesseract"]["outputs"], 51)
        self.assertEqual(
            ocr["independent_tesseract"]["ordered_sha256"],
            "a8b9639328fc5b1517282eb49fc6ad35d027889a5dbd377c05f493f6b6315247",
        )
        self.assertFalse(ocr["retained_in_repository"])

    def test_complete_manual_page_review_and_direct_claim_checks(self):
        review = PROVENANCE["manual_review"]
        self.assertTrue(review["completed"])
        self.assertEqual(review["reviewed_pages"], 51)
        self.assertEqual(review["contact_sheet_review_pages"], 51)
        self.assertEqual(len(DISPOSITIONS), 51)
        self.assertEqual({item["page"] for item in DISPOSITIONS}, set(range(1, 52)))
        included_pages = {
            item["page"] for item in DISPOSITIONS if item["decision"].startswith("include")
        }
        self.assertEqual(included_pages, set(review["claim_pages_directly_checked"]))
        self.assertTrue(all("section" in item and "rationale" in item for item in DISPOSITIONS))

    def test_report_names_every_page(self):
        for page in range(1, 52):
            self.assertIn(f"PDF page {page} —", REPORT)

    def test_figure_and_appendix_dispositions(self):
        by_page = {item["page"]: item for item in DISPOSITIONS}
        figure_pages = {6, 8, 9, 12, 14, 17, 20, 25, 26, 29, 30, 33, 34, 35, 36, 38, 40, 43, 46, 47, 48}
        for page in figure_pages:
            self.assertEqual(by_page[page]["decision"], "exclude")
        for page in range(42, 52):
            self.assertEqual(by_page[page]["decision"], "exclude")

    def test_relationship_audit_has_different_populations(self):
        relation = PROVENANCE["relationship_audit"]
        self.assertFalse(relation["direct_extension_claimed"])
        self.assertEqual(
            relation["source_1989_population"], "eight-strand plaited synthetic rope"
        )
        self.assertEqual(
            relation["source_1990_population"], "synthetic double-braided rope"
        )
        self.assertIn("does not list", relation["source_lineage_note"])

    def test_manifest_hashes_and_statistics(self):
        self.assertEqual(MANIFEST["schema"], "site-corpus-manifest-v1")
        self.assertEqual(MANIFEST["stats"]["source_pdf_pages_reviewed"], 51)
        self.assertEqual(MANIFEST["stats"]["disposition_rows"], 51)
        self.assertEqual(MANIFEST["stats"]["sources"], 3)
        for relative, metadata in MANIFEST["files"].items():
            payload = (ROOT / relative).read_bytes()
            self.assertEqual(len(payload), metadata["bytes"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), metadata["sha256"])

    def test_training_only_split_hygiene(self):
        split = MANIFEST["split_hygiene"]
        self.assertEqual(split["allowed"], ["training_source"])
        for forbidden in ("validation", "holdout", "evaluation", "ood", "probe"):
            self.assertIn(forbidden, split["forbidden"])
        self.assertIn("training-source-only", README)


if __name__ == "__main__":
    unittest.main()
