import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
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
            "REPORT.md",
            "dispositions.jsonl",
            "manifest.json",
            "source_snapshot/provenance.json",
            "sources.jsonl",
            "tests/test_corpus.py",
        }
        actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()}
        self.assertEqual(actual, expected)

    def test_bibliography(self):
        for value in (
            "J. Šimon",
            "V. Dekýš",
            "P. Palček",
            "Revision of Commonly Used Loop Knots Efficiencies",
            "10.12693/APhysPolA.138.404",
            "November 15, 2019",
            "April 24, 2020",
            "404–420",
        ):
            self.assertIn(value, CORPUS)

    def test_front_loaded_boundary(self):
        front = CORPUS[:2200].lower()
        for phrase in (
            "expert-tied",
            "slowly to rupture",
            "synthetic polyamide low-stretch kernmantle",
            "natural-fiber rope",
            "bondage rope",
            "body contact",
            "dynamic loading",
            "cyclic loading",
            "uplines",
            "human suspension",
            "numerical knot-efficiency values and rankings are intentionally absent",
        ):
            self.assertIn(phrase, front)

    def test_all_eight_knot_families(self):
        headings = re.findall(r"(?m)^### \d+\. (.+)$", CORPUS)
        self.assertEqual(
            headings,
            [
                "Figure-eight loop",
                "Double figure-eight loop",
                "Figure-nine loop",
                "Overhand loop",
                "Bowline",
                "Left-hand bowline",
                "Bowline on a bight",
                "Alpine butterfly loop",
            ],
        )

    def test_rope_inventory_is_exact(self):
        expected = {
            "G": ("10.5 mm", "2015"),
            "B": ("10.5 mm", "2006"),
            "EW": ("10.5 mm", "2014"),
            "L1": ("11.0 mm", "2006"),
            "ER": ("10.5 mm", "2008"),
            "L2": ("9.1 mm", "2004"),
            "G8": ("8 mm", "2015–2018"),
        }
        inventory = CORPUS.split("The exact neutral specimen inventory is:", 1)[1].split(
            "The codes above", 1
        )[0]
        for code, values in expected.items():
            line = next(line for line in inventory.splitlines() if line.startswith(f"- **{code}:**"))
            for value in values:
                self.assertIn(value, line)

    def test_rope_histories(self):
        for phrase in (
            "new and unused",
            "painting vertical structures",
            "color affecting both sheath and core",
            "speleological rescue",
            "sheath was worn",
            "mountain rescue",
            "no mission or wear-stage entry",
        ):
            self.assertIn(phrase, CORPUS)

    def test_no_trademarks_or_vendor_names_in_corpus(self):
        for mark in (
            "Gilmonte",
            "Beal",
            "Edelweiss",
            "Lanex",
            "Edelrid",
            "Tendon",
            "Profistatic",
            "FLIR",
            "TESCAN",
        ):
            self.assertNotIn(mark, CORPUS)

    def test_apparatus_is_static_and_calibrated(self):
        for phrase in (
            "180 mm per minute",
            "slow-static displacement test",
            "independently calibrated every two years",
            "below two tenths of one percent",
            "stable laboratory conditions",
        ):
            self.assertIn(phrase, CORPUS)

    def test_tying_control_is_cautious(self):
        for phrase in (
            "correct tying and dressing",
            "photographic documentation",
            "does not report a blinded tying protocol",
            "does not identify every tyer",
            "effect of improper dressing had not been seriously measured",
        ):
            self.assertIn(phrase, CORPUS)

    def test_geometry_and_load_states(self):
        for phrase in (
            "standard-load",
            "cross-load",
            "ring-load",
            "geometry I",
            "geometry O",
            "not step-by-step tying directions",
        ):
            self.assertIn(phrase, CORPUS)

    def test_replication_counts(self):
        for count in (81, 22, 44, 19, 24, 108, 48, 51, 101, 32, 18):
            self.assertRegex(CORPUS, rf"\b{count}\b")

    def test_ratio_uncertainty(self):
        for phrase in (
            "ratio distribution is asymmetric",
            "not generally the ratio of the two sample means",
            "ratio-of-means shortcut can bias",
            "confidence interval",
        ):
            self.assertIn(phrase, CORPUS)

    def test_efficiency_nonconstancy_is_bounded(self):
        section = CORPUS.split("For the figure-eight and overhand families", 1)[1].split(
            "## Loading", 1
        )[0]
        self.assertIn("did not establish this relationship for every loop knot", section)
        self.assertIn("one context-free efficiency number", section)

    def test_break_and_untying_are_distinct(self):
        self.assertGreaterEqual(CORPUS.lower().count("untied"), 3)
        self.assertIn("a rupture-based efficiency calculation is not the relevant statistic", CORPUS)
        self.assertIn("different endpoint from rupture", CORPUS)

    def test_thermal_causality_is_unresolved(self):
        for phrase in (
            "localized surface temperatures approached the melting region of polyamide",
            "not sufficient to determine whether",
            "causal order unresolved",
            "not evidence for a usable temperature limit",
        ):
            self.assertIn(phrase, CORPUS)

    def test_microscopy_observation_and_hypothesis_are_separate(self):
        for phrase in (
            "broken ends were thickened",
            "flake-like surface structure",
            "presented the explanation as a hypothesis",
        ):
            self.assertIn(phrase, CORPUS)

    def test_stated_limitations(self):
        limits = CORPUS.split("## What the experiment does not establish", 1)[1]
        for phrase in (
            "not bends, hitches, or friction hitches",
            "not dynamic climbing rope or natural fiber",
            "effect of realistic faster or impact loading remained unknown",
            "wetness",
            "sheath smoothness",
            "one rope and one cord",
            "do not turn these slow-static tests into cyclic validation",
        ):
            self.assertIn(phrase, limits)

    def test_no_portable_efficiency_values_or_rankings(self):
        self.assertIsNone(re.search(r"efficienc[^\n]{0,80}\d+(?:\.\d+)?\s*%", CORPUS, re.I))
        self.assertIsNone(re.search(r"\b(?:highest|strongest|best|safest)\s+(?:efficiency|knot)", CORPUS, re.I))
        self.assertNotIn("ranked from", CORPUS.lower())

    def test_no_load_or_rating_values(self):
        self.assertIsNone(re.search(r"\b\d+(?:\.\d+)?\s*kN\b", CORPUS, re.I))
        self.assertNotIn("working load", CORPUS.lower())
        self.assertNotIn("safe working", CORPUS.lower())

    def test_no_tying_recipe_or_field_recommendation(self):
        for phrase in (
            "tie this knot",
            "wrap the rope",
            "pass the end",
            "use this knot for",
            "we recommend",
            "should be preferred",
        ):
            self.assertNotIn(phrase, CORPUS.lower())

    def test_no_source_visuals_tables_or_bodies(self):
        self.assertNotIn("![", CORPUS)
        self.assertNotRegex(CORPUS, r"(?m)^\|.+\|$")
        forbidden = {".pdf", ".html", ".htm", ".png", ".jpg", ".jpeg", ".webp"}
        self.assertFalse(any(p.suffix.lower() in forbidden for p in ROOT.rglob("*")))
        self.assertTrue(all(not s["retained_in_repository"] for s in SOURCES))

    def test_claim_citations_are_dense(self):
        body = CORPUS.split("## Attribution and adaptation notice", 1)[0]
        citations = re.findall(r"\[APP-138-404:[^\]]+\]", body)
        self.assertGreaterEqual(len(citations), 34)
        substantive = [
            p
            for p in re.split(r"\n\s*\n", body)
            if len(p.split()) >= 34 and not p.startswith("#") and not p.startswith("-")
        ]
        self.assertEqual(
            [p for p in substantive if "[APP-138-404:" not in p and "[APPA-ABOUT:" not in p],
            [],
        )

    def test_official_source_ledger(self):
        self.assertEqual(len(SOURCES), 4)
        self.assertEqual(
            {s["id"] for s in SOURCES},
            {"DOI-RESOLVER", "APP-138-404", "APPA-ABOUT", "CC-BY-4.0"},
        )
        pdf = next(s for s in SOURCES if s["id"] == "APP-138-404")
        self.assertEqual(pdf["pdf_pages"], 17)
        self.assertEqual(pdf["bytes"], 2840659)
        self.assertEqual(
            pdf["sha256"], "f3a590f7ce727494dde386b95fd903ad3a5ba49b33d1d07c9bfbeff1e0d5b17e"
        )

    def test_doi_and_pdf_are_byte_identical(self):
        by_id = {s["id"]: s for s in SOURCES}
        self.assertEqual(by_id["DOI-RESOLVER"]["sha256"], by_id["APP-138-404"]["sha256"])
        self.assertEqual(by_id["DOI-RESOLVER"]["bytes"], by_id["APP-138-404"]["bytes"])

    def test_rights_hashes_and_attribution(self):
        by_id = {s["id"]: s for s in SOURCES}
        self.assertEqual(
            by_id["APPA-ABOUT"]["sha256"],
            "1ad2cd22245581fe207d5947bdc31bcb02a543ea28a4ef7d6c826a473dc3e7bf",
        )
        self.assertEqual(
            by_id["CC-BY-4.0"]["sha256"],
            "231a5dac65bbf135ba27145969a63cd289faadc172f1512c4810a6c60ba91036",
        )
        notice = CORPUS.split("## Attribution and adaptation notice", 1)[1]
        for value in ("J. Šimon", "V. Dekýš", "P. Palček", "CC BY 4.0", "changes wording"):
            self.assertIn(value, notice)

    def test_rights_screen_records_pdf_notice_gap(self):
        rights = PROVENANCE["rights_screen"]
        self.assertFalse(rights["article_pdf_embedded_cc_notice_found"])
        self.assertFalse(rights["legal_advice"])
        self.assertIn("source PDF does not itself display a CC notice", rights["provenance_cautions"])
        self.assertIn("current publisher policy", REPORT)

    def test_manual_review_and_component_counts(self):
        audit = PROVENANCE["extraction_audit"]
        self.assertTrue(PROVENANCE["manual_review"]["completed"])
        self.assertEqual(audit["pdf_pages"], 17)
        self.assertEqual(audit["source_numbered_figures"], 31)
        self.assertEqual(audit["source_numbered_tables"], 5)
        self.assertEqual(audit["raster_image_xobjects"], 0)
        for page_group in ("1–2", "3–4", "5–6", "7–12", "13–14", "15–17"):
            self.assertIn(f"PDF pages {page_group}", REPORT)

    def test_disposition_inventory(self):
        self.assertGreaterEqual(len(DISPOSITIONS), 32)
        by_component = {d["component"]: d["decision"] for d in DISPOSITIONS}
        self.assertEqual(by_component["Ranking Table V and comparison Figure 22"], "exclude")
        self.assertEqual(by_component["Bowline results"], "include_failure_mode_only")
        self.assertEqual(by_component["Cross-domain and operational transfer"], "exclude")

    def test_split_hygiene(self):
        split = MANIFEST["split_hygiene"]
        self.assertEqual(split["allowed"], ["training_source"])
        self.assertEqual(
            set(split["forbidden"]), {"validation", "holdout", "evaluation", "ood", "probe"}
        )
        self.assertIn("training-source-only", REPORT)

    def test_manifest_sources_match(self):
        self.assertEqual(set(MANIFEST["source_ids"]), {s["id"] for s in SOURCES})

    def test_manifest_file_hashes(self):
        expected = {
            "CORPUS.md",
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
