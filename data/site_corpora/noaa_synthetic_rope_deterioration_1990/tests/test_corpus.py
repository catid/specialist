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

    def test_identity_and_title_spelling(self):
        for value in (
            "Moon Hwo Seo",
            "Stanley Backer",
            "John F. Mandell",
            "Modeling Of Synthetic Fiber Ropes Deterioration",
            "Modelling of Synthetic Fiber Ropes Deterioration",
            "MITSG 90-18",
            "NA86AA-D-SG089",
            "RT-11",
            "1990",
        ):
            self.assertIn(value, CORPUS)

    def test_front_loaded_nontransfer_boundary(self):
        front = CORPUS[:2600].lower()
        for term in (
            "natural-fiber rope",
            "knots",
            "rope care",
            "retirement criteria",
            "bondage",
            "body contact",
            "uplines",
            "anchors",
            "human suspension",
            "safe-load rule",
            "operational recommendation",
        ):
            self.assertIn(term, front)

    def test_two_damage_clocks(self):
        section = CORPUS.split("## One rope, two competing deterioration clocks", 1)[1].split(
            "## How rope geometry", 1
        )[0]
        for phrase in (
            "Tensile creep-fatigue",
            "accumulated tensile exposure time",
            "Internal abrasion",
            "friction cycles",
            "respond differently to cycling frequency",
            "competing failure routes",
            "not safe or unsafe operating categories",
        ):
            self.assertIn(phrase, section)

    def test_internal_abrasion_mechanism(self):
        for phrase in (
            "sinusoidal undulation superposed on a circular helix",
            "lateral contact pressure",
            "relative motion between opposing components",
            "rope’s own moving components",
            "surface of highest curvature",
            "Lost cross-sectional area",
        ):
            self.assertIn(phrase, CORPUS)

    def test_movement_is_assumption_not_fact(self):
        for phrase in (
            "could add component motion",
            "assumes relative movement on every loading cycle",
            "modeling choice",
            "necessary and sufficient conditions for motion were explicitly described as uncertain",
        ):
            self.assertIn(phrase, CORPUS)

    def test_inherited_evidence_is_labeled(self):
        section = CORPUS.split("## Evidence for internal fiber-to-fiber abrasion", 1)[1].split(
            "## Tensile creep-fatigue input", 1
        )[0]
        for phrase in (
            "summarizes earlier microscopy",
            "come from cited prior work",
            "were not independently reproduced in this 1990 report",
            "observing abrasion does not by itself prove abrasion controlled final failure",
        ):
            self.assertIn(phrase, section)

    def test_creep_fatigue_scope(self):
        for phrase in (
            "nylon 6.6",
            "polyethylene terephthalate",
            "tested high-stress ranges",
            "time-based cumulative-damage rule",
            "Frequency independent” does not mean frequency disappears",
            "comparatively sudden",
        ):
            self.assertIn(phrase, CORPUS)

    def test_wear_input_and_low_pressure_gap(self):
        section = CORPUS.split("## Yarn-on-yarn wear input", 1)[1].split(
            "## Structural and failure assumptions", 1
        )[0]
        for phrase in (
            "number of friction cycles",
            "normal contact pressure",
            "three-region empirical wear curve",
            "low-pressure region was a major evidence gap",
            "little at very low pressure",
            "hypothetical wear parameters rather than reporting new yarn observations",
        ):
            self.assertIn(phrase, section)

    def test_structural_assumptions_are_explicit(self):
        section = CORPUS.split("## Structural and failure assumptions", 1)[1].split(
            "## Internal wear versus external wear", 1
        )[0]
        for phrase in (
            "zero-friction and infinite-friction cases",
            "identical strands",
            "square array of identical filaments",
            "worn layers are assumed not to alter local strand curvature",
            "uniform through the bundle",
            "weakest strand location",
            "define the model’s population",
        ):
            self.assertIn(phrase, section)

    def test_internal_external_wear_distinction(self):
        for phrase in (
            "distinguishes internal component-on-component wear from external abrasion",
            "assumes internal and external wear are independent",
            "short localized wear zone",
            "longer, effectively distributed zone",
            "Failure location and termination damage are part of the experimental outcome",
        ):
            self.assertIn(phrase, CORPUS)

    def test_hysteretic_heating_is_bounded(self):
        section = CORPUS.split("## Hysteretic heating as a test confound", 1)[1].split(
            "## What the validation", 1
        )[0]
        for phrase in (
            "energy dissipated through hysteresis",
            "when cyclic tests are not run wet",
            "does not present a thermal model, temperature measurements, or a heating threshold",
            "qualitative confound",
            "not a third calibrated deterioration law",
        ):
            self.assertIn(phrase, section)

    def test_validation_limits(self):
        section = CORPUS.split("## What the validation comparisons do and do not show", 1)[1].split(
            "## Durable lessons", 1
        )[0]
        for phrase in (
            "many references did not describe rope structure in enough detail",
            "assume the compared ropes are geometrically similar",
            "wear dominance at lower applied tension was a model prediction below that evidence",
            "only a small set of mid-span failures",
            "very-low-tension region uncertain",
            "residual-strength comparison is mixed",
            "external-wear extension reproduced a qualitative trend but missed the observations materially",
            "no modern uncertainty analysis",
        ):
            self.assertIn(phrase, section.lower() if phrase.startswith("residual") else section)

    def test_no_numeric_load_life_or_rating_outputs(self):
        for pattern in (
            r"\b\d+(?:\.\d+)?\s*%",
            r"\b\d+(?:\.\d+)?\s*(?:psi|kpa|mpa|kn|n|lbf)\b",
            r"\b\d+(?:\.\d+)?\s*(?:hz|s/cycle|cycles?|months?)\b",
        ):
            self.assertIsNone(re.search(pattern, CORPUS, flags=re.IGNORECASE))

    def test_no_commercial_test_materials(self):
        for name in ("DuPont", "Samson", "Crawford"):
            self.assertNotIn(name, CORPUS)

    def test_no_operational_formula_or_recipe(self):
        for token in ("log N", "Teye", "Equation (", "\\int", "$$", "\\["):
            self.assertNotIn(token, CORPUS)
        for phrase in (
            "working load limit",
            "safe working load",
            "retire when",
            "inspection interval of",
            "safety factor of",
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
        citations = re.findall(r"\[(?:NOAA-1990|NOAA-IR):[^\]]+\]", body)
        self.assertGreaterEqual(len(citations), 40)
        substantive = [
            paragraph
            for paragraph in re.split(r"\n\s*\n", body)
            if len(paragraph.split()) >= 34 and not paragraph.startswith("#")
        ]
        self.assertEqual(
            [paragraph for paragraph in substantive if not re.search(r"\[(?:NOAA-1990|NOAA-IR):", paragraph)],
            [],
        )

    def test_source_ledger_and_checksum(self):
        self.assertEqual(len(SOURCES), 2)
        self.assertEqual({source["id"] for source in SOURCES}, {"NOAA-IR-9887", "NOAA-1990-PDF"})
        by_id = {source["id"]: source for source in SOURCES}
        self.assertEqual(by_id["NOAA-1990-PDF"]["pdf_pages"], 41)
        self.assertEqual(
            by_id["NOAA-1990-PDF"]["sha256"],
            "7a29a928c08d6257f1558eac2d63a26c8e662240e01ee87400cedc95b70d242f",
        )
        self.assertTrue(by_id["NOAA-1990-PDF"]["record_checksum_match"])

    def test_public_domain_rights_screen(self):
        rights = PROVENANCE["rights_screen"]
        self.assertEqual(rights["record_designation"], "Public Domain")
        self.assertTrue(rights["main_document_checksum_bound"])
        self.assertFalse(rights["source_bodies_retained"])
        self.assertFalse(rights["legal_advice"])
        notice = CORPUS.split("## Public-domain adaptation and source notice", 1)[1]
        for phrase in ("Public Domain", "changes wording", "No endorsement"):
            self.assertIn(phrase, notice)

    def test_scan_and_ocr_audit(self):
        document = PROVENANCE["document_audit"]
        self.assertEqual(document["pdf_pages"], 41)
        self.assertTrue(document["image_only_scan"])
        self.assertEqual(document["embedded_text_characters"], 0)
        self.assertEqual(document["page_images"], 41)
        self.assertEqual(document["page_pixels_per_inch"], 160)
        ocr = PROVENANCE["ocr_audit"]
        self.assertEqual(ocr["outputs"], 41)
        self.assertEqual(ocr["ordered_concatenation_bytes"], 40254)
        self.assertEqual(ocr["ordered_whitespace_words"], 6762)
        self.assertEqual(
            ocr["ordered_concatenation_sha256"],
            "272238d64f26759d0437e5fd8529d55e478c101f6392923cef8dfd106fd808f8",
        )
        self.assertFalse(ocr["retained_in_repository"])

    def test_complete_manual_page_review(self):
        review = PROVENANCE["manual_review"]
        self.assertTrue(review["completed"])
        self.assertEqual(review["reviewed_pages"], 41)
        self.assertEqual(len(DISPOSITIONS), 41)
        self.assertEqual({item["page"] for item in DISPOSITIONS}, set(range(1, 42)))
        self.assertTrue(all("section" in item and "rationale" in item for item in DISPOSITIONS))

    def test_blank_references_and_figure_dispositions(self):
        by_page = {item["page"]: item for item in DISPOSITIONS}
        self.assertEqual(by_page[3]["decision"], "exclude")
        self.assertEqual(by_page[23]["decision"], "exclude")
        self.assertEqual(by_page[24]["decision"], "exclude")
        for page in range(25, 42):
            self.assertEqual(by_page[page]["decision"], "exclude")
            self.assertEqual(by_page[page]["section"], "Figures")

    def test_report_records_every_page_group(self):
        for phrase in (
            "PDF page 1:",
            "PDF page 2:",
            "PDF page 3:",
            "PDF pages 6–7",
            "PDF pages 23–24",
            "PDF page 25",
            "PDF page 41",
        ):
            self.assertIn(phrase, REPORT)

    def test_section_inventory(self):
        sections = PROVENANCE["section_inventory"]
        self.assertEqual(len(sections), 13)
        self.assertIn("2. Rope Deterioration Model", sections)
        self.assertIn("4D. The Effect of Low and High Load Wear Regimes", sections)
        self.assertIn("Figures 1 through 13d", sections)

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
