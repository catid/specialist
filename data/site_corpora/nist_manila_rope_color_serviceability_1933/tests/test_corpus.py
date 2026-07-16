import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEXICAL_TOKEN_RE = re.compile(r"\w+(?:[’'-]\w+)*|[^\w\s]", re.UNICODE)


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def read_json(relative_path: str):
    return json.loads(read_text(relative_path))


def read_jsonl(relative_path: str):
    return [json.loads(line) for line in read_text(relative_path).splitlines() if line]


def sha256(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


class NistManilaRopeColorCorpusTests(unittest.TestCase):
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
            ".jpeg", ".jpg", ".pdf", ".png", ".svg", ".tif", ".tiff",
            ".txt", ".webp",
        }
        self.assertFalse(any(path.suffix.lower() in forbidden_suffixes for path in ROOT.rglob("*")))

    def test_locked_content_hashes(self):
        expected = {
            "CORPUS.md": "ebd170b0b0fddb4ac8e9042fadea7503865dda5df53738cf953b423c4a897ce3",
            "README.md": "303af7eb8a176e68999043d81c9a139ab619cfe5d3184e8cad5691a225296ae1",
            "REPORT.md": "6cfb982ec60f014c9c3276a1e27d8071275d8cda8b54749cb0071b7e8ac1a80a",
            "components.jsonl": "36b8952a2d6213aded41a825615f36e81cab7143031236116d82a1f7695c221b",
            "dispositions.jsonl": "639cb966304ae39a94077140934cba2e213ccd42cb9284bec0ff636a157ffecc",
            "source_snapshot/provenance.json": "1f8e3fc3ec98929b7c103d11f61f518f2d045cab2304e124de899416c77ff0ca",
            "sources.jsonl": "f202a47f9d589054763d860f790ff366082d5d4a6cdab89ac6c953540e598cff",
            "surfaces.jsonl": "875aacc70e813e556ed5d5e9b514f1fc8caa4b32d0ff06cb21bcdd472e2e4c8e",
        }
        self.assertEqual({path: sha256(path) for path in expected}, expected)

    def test_exact_pdf_identity_checksum_and_metadata(self):
        rows = read_jsonl("sources.jsonl")
        self.assertEqual(len(rows), 6)
        by_id = {row["source_id"]: row for row in rows}
        self.assertEqual(
            set(by_id),
            {
                "NIST-RP627", "NIST-VOLUME-11", "NIST-JOURNAL-ABOUT",
                "NIST-TECHNICAL-SERIES-RIGHTS", "NIST-MAIN-ROBOTS",
                "NVLPUBS-ROBOTS-404",
            },
        )
        pdf = by_id["NIST-RP627"]
        self.assertEqual(pdf["authors"], ["Genevieve Becker", "William D. Appel"])
        self.assertEqual(pdf["research_paper"], "RP627")
        self.assertEqual(pdf["volume"], 11)
        self.assertEqual(pdf["publication_year"], 1933)
        self.assertEqual(pdf["printed_pages"], "811-822")
        self.assertEqual(pdf["doi"], "10.6028/jres.011.057")
        self.assertEqual(pdf["body_bytes"], 1082392)
        self.assertEqual(pdf["pdf_pages"], 12)
        self.assertEqual(
            pdf["body_sha256"],
            "fc29fe3c20edda1d8372c05083e9154d328b225c6959a9f1b8f4ab6edc5debd1",
        )
        self.assertEqual(
            pdf["body_sha512"],
            "bec7afc264e1659011a2c9a483be4842d9ba7f6c3805c74204d909c03085a547a124636f191ab0b57a6dbd445e143cf5a06c4ac78eb79cbb0460fa4dd43e2cbd",
        )
        self.assertTrue(pdf["manual_page_audit"] and pdf["manual_component_audit"])
        self.assertFalse(pdf["source_body_redistributed"])

    def test_official_html_and_robots_checksums(self):
        rows = {row["source_id"]: row for row in read_jsonl("sources.jsonl")}
        expected = {
            "NIST-VOLUME-11": (68858, "5a3493ccbb93123d0e9889ab16829dacb9ecaf1ad726580e777025c6e41622fc"),
            "NIST-JOURNAL-ABOUT": (53579, "e7b04e06265cb650e71149f46fa7b494a6de21da849f16804f28178bacfe06ab"),
            "NIST-TECHNICAL-SERIES-RIGHTS": (66782, "659f93f6ae02bde070d27de443c7a9dafa18c5f258dfcb394aadfd36300d02e6"),
            "NIST-MAIN-ROBOTS": (2425, "0fed8806709b6ff2716921723b5077d1c19d152f760db983af60988cae0031dd"),
            "NVLPUBS-ROBOTS-404": (1380, "6d219a49a0b5c00cb8245fb679d96f80ca4c89b0fae870ee3cd8d0eac09f4dff"),
        }
        for source_id, (size, digest) in expected.items():
            self.assertEqual(rows[source_id]["body_bytes"], size)
            self.assertEqual(rows[source_id]["body_sha256"], digest)
            self.assertFalse(rows[source_id]["source_body_redistributed"])
        self.assertEqual(rows["NIST-MAIN-ROBOTS"]["http_status"], 200)
        self.assertFalse(rows["NIST-MAIN-ROBOTS"]["target_paths_disallowed"])
        self.assertEqual(rows["NVLPUBS-ROBOTS-404"]["http_status"], 404)
        self.assertIn("not treated as affirmative permission", rows["NVLPUBS-ROBOTS-404"]["interpretation"])

    def test_provenance_identity_access_pdf_and_rights(self):
        provenance = read_json("source_snapshot/provenance.json")
        identity = provenance["source_identity"]
        self.assertEqual(identity["doi"], "10.6028/jres.011.057")
        self.assertEqual(identity["article_dateline"], "Washington, September 16, 1933")
        self.assertIn("Cordage Institute", identity["becker_affiliation_note"])
        self.assertIn("Textile Section", identity["appel_affiliation_note"])
        self.assertTrue(identity["identity_cross_checked_between_official_volume_page_pdf_and_doi"])
        access = provenance["access_audit"]
        self.assertFalse(access["site_crawl_performed"])
        self.assertFalse(access["search_endpoint_used"])
        self.assertEqual(access["known_official_pdf_body_gets"], 1)
        self.assertEqual(access["third_party_or_reference_bodies_retrieved"], 0)
        resolution = provenance["doi_resolution"]
        self.assertTrue(resolution["target_content_length_matches_acquired_pdf"])
        self.assertTrue(resolution["target_etag_matches_acquired_pdf"])
        self.assertFalse(resolution["second_pdf_body_acquired"])
        metadata = provenance["pdf_metadata"]
        self.assertEqual(metadata["pages"], 12)
        self.assertFalse(metadata["tagged"])
        self.assertFalse(metadata["encrypted"])
        self.assertFalse(metadata["javascript"])
        self.assertFalse(metadata["forms"])
        self.assertTrue(metadata["ocr_text_layer"])
        self.assertEqual(metadata["rgb_scan_image_objects"], 24)
        self.assertEqual(metadata["mask_objects"], 12)
        rights = provenance["rights"]
        self.assertFalse(rights["worldwide_public_domain_claimed"])
        self.assertTrue(rights["third_party_component_caveat"])
        self.assertTrue(rights["attribution_provided"] and rights["change_notice_provided"])
        self.assertFalse(rights["endorsement_implied"])

    def test_every_pdf_page_has_one_manual_disposition(self):
        rows = read_jsonl("dispositions.jsonl")
        self.assertEqual(len(rows), 12)
        self.assertEqual([row["pdf_page"] for row in rows], list(range(1, 13)))
        self.assertEqual([row["printed_page"] for row in rows], list(range(811, 823)))
        self.assertTrue(all(row["source_id"] == "NIST-RP627" for row in rows))
        self.assertTrue(all(row["audit"] == "manual OCR-text and rendered-page review" for row in rows))
        excluded_pages = [row["pdf_page"] for row in rows if row["decision"] == "exclude"]
        self.assertEqual(excluded_pages, [4, 6, 7, 10, 11])
        partial_pages = [row["pdf_page"] for row in rows if row["decision"].startswith("partial")]
        self.assertEqual(partial_pages, [1, 2, 3, 5, 8, 9, 12])

    def test_figures_tables_scans_and_references_are_reconciled(self):
        rows = read_jsonl("components.jsonl")
        self.assertEqual(len(rows), 16)
        figures = [row for row in rows if row["component_type"] == "figure"]
        tables = [row for row in rows if row["component_type"] == "table"]
        self.assertEqual([row["component_id"] for row in figures], [f"figure-{n}" for n in range(1, 5)])
        self.assertEqual([row["printed_pages"] for row in figures], [[814], [816], [817], [819]])
        self.assertEqual([row["component_id"] for row in tables], [f"table-{n}" for n in range(1, 6)])
        self.assertEqual([row["printed_pages"] for row in tables], [[815], [819], [820], [821], [821]])
        self.assertTrue(all(row["decision"] == "exclude" and row["retained"] == "none" for row in figures + tables))
        scans = next(row for row in rows if row["component_id"] == "scan-image-layer")
        self.assertEqual(scans["rgb_image_objects"] + scans["mask_objects"], 36)
        references = next(row for row in rows if row["component_id"] == "footnotes-and-reference-bodies")
        self.assertEqual(references["footnote_markers"], 21)
        self.assertEqual(references["affiliation_notes_partially_retained"], [1, 2])
        self.assertIn("none was acquired", references["excluded"])

    def test_all_official_web_surfaces_have_manual_decisions(self):
        rows = read_jsonl("surfaces.jsonl")
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(row["decision"] for row in rows))
        self.assertTrue(all(row["audit"].startswith("manual") for row in rows))
        rights = [row for row in rows if "rights" in row["surface_id"]]
        self.assertEqual(len(rights), 2)
        doi = next(row for row in rows if row["surface_id"] == "doi-resolution")
        self.assertIn("matching content length and ETag", doi["retained"])

    def test_complete_manual_review_and_forbidden_artifact_boundary(self):
        review = read_json("source_snapshot/provenance.json")["manual_review"]
        self.assertTrue(review["all_pdf_pages_text_screened"])
        self.assertTrue(review["all_pdf_pages_rendered_and_visually_inspected"])
        self.assertEqual(review["ordered_contact_sheets"], 2)
        self.assertEqual(review["numbered_figures"], 4)
        self.assertEqual(review["numbered_tables"], 5)
        self.assertEqual(review["external_reference_bodies_acquired"], 0)
        self.assertFalse(review["existing_corpus_bodies_inspected"])
        self.assertFalse(review["qa_trainer_eval_holdout_ood_probe_or_experiment_artifacts_inspected"])

    def test_novelty_audit_used_only_allowed_metadata_surfaces(self):
        novelty = read_json("source_snapshot/provenance.json")["novelty_audit"]
        self.assertEqual(novelty["searched_existing_file_classes"], ["manifest.json", "REPORT.md", "README.md"])
        self.assertFalse(novelty["existing_corpus_bodies_searched_or_opened"])
        self.assertEqual(len(novelty["candidate_phrases_with_zero_matches"]), 7)
        self.assertIn("finished-rope surface", novelty["candidate_phrases_with_zero_matches"])
        self.assertIn("Boston Navy Yard", novelty["candidate_phrases_with_zero_matches"])

    def test_corpus_is_concise_cited_and_counted(self):
        corpus = read_text("CORPUS.md")
        self.assertEqual(len(corpus.encode("utf-8")), 6989)
        self.assertEqual(len(corpus.split()), 980)
        self.assertEqual(len(LEXICAL_TOKEN_RE.findall(corpus)), 1242)
        paragraphs = [part for part in corpus.split("\n\n") if not part.startswith("#")]
        self.assertEqual(len(paragraphs), 17)
        self.assertEqual(sum(part.startswith("Historical source evidence:") for part in paragraphs), 11)
        self.assertEqual(sum(part.startswith("Historical source boundary:") for part in paragraphs), 6)
        self.assertTrue(all(re.search(r"\[[^\[\]]+\]$", part) for part in paragraphs))
        self.assertNotRegex(corpus, r"https?://|www\.")
        self.assertNotRegex(corpus, r"!\[[^\]]*\]\([^)]*\)")
        self.assertNotRegex(corpus, r"(?m)^\s*\|.*\|\s*$")

    def test_required_evidence_and_limitations_are_present(self):
        corpus = read_text("CORPUS.md").lower()
        for phrase in (
            "10.6028/jres.011.057",
            "research associate at the bureau of standards for the cordage institute",
            "boston navy yard",
            "abaca fiber",
            "constituent fiber color",
            "finished-rope surface appearance",
            "construction, yarn size, strand arrangement, twist, and dirt or dust",
            "lubricant’s own color",
            "exposure to light and air",
            "abaca fiber can be bleached",
            "not concerned with the relationship between fiber color and rope serviceability",
            "did not test whether a rope remained fit for use",
            "unspecified navy department indication",
            "does not establish that dark rope is safe",
            "supply by those organizations is not navy qualification",
        ):
            self.assertIn(phrase, corpus)

    def test_no_value_scale_instrument_solvent_recipe_or_reference_leakage(self):
        corpus = read_text("CORPUS.md")
        lower = corpus.lower()
        for forbidden in (
            "figure 1", "figure 2", "figure 3", "figure 4", "table 1",
            "table 2", "table 3", "table 4", "table 5", "becker value",
            "reflectance", "wavelength", "magnesium oxide", "photometer",
            "spectrophotometer", "soxhlet", "petroleum ether", "munsell",
            "wratten", "eastman kodak", "mineral oil", "woololein", "degras",
            "meat cutter", "spatula", "filter paper", "t-r-601", "f.s. no.",
            "saleeby", "nickerson", "troland", "international critical tables",
            "optical society of america", "mcnicholas", "working drawings",
        ):
            self.assertNotIn(forbidden, lower)
        without_citations = re.sub(r"\[[^\[\]]+\]", "", corpus)
        self.assertNotRegex(
            without_citations,
            r"(?i)\b\d+(?:\.\d+)?\s*(?:%|percent|mm|cm|inches?|feet|units?|hours?|degrees?|wavelengths?)\b",
        )

    def test_no_color_discard_rule_or_modern_transfer(self):
        corpus = read_text("CORPUS.md").lower()
        boundary = corpus[corpus.index("## absolute non-transfer boundary") :]
        for phrase in (
            "no color threshold",
            "cannot be used to accept, reject, retire, clean, grade, or load a rope",
            "color alone cannot establish material identity, internal condition, strength, remaining life, contamination, hygiene, or serviceability",
            "jute, hemp, rope cleaning, hygiene, drying, storage, retirement, working loads",
            "body contact",
            "uplines",
            "anchors",
            "hardpoints",
            "bondage",
            "restraint",
            "body lowering",
            "human suspension",
            "no current inspection, procurement, maintenance, or safety recommendation",
        ):
            self.assertIn(phrase, boundary)
        self.assertNotRegex(corpus, r"(?i)\b(?:discard|retire|reject|accept)\s+(?:at|above|below|when)\b")

    def test_rights_attribution_change_and_component_caveats(self):
        corpus = read_text("CORPUS.md").lower()
        readme = read_text("README.md").lower()
        report = read_text("REPORT.md").lower()
        for phrase in (
            "not subject to copyright in the united states",
            "possible foreign rights",
            "third-party works or components can differ",
            "makes no worldwide public-domain claim",
        ):
            self.assertIn(phrase, corpus)
        self.assertIn("manually rewritten, quotation-light, narrowed derivative", readme)
        self.assertIn("no endorsement", readme)
        self.assertIn("all 12 physical pages were extracted for text screening, rendered, and manually inspected", report)
        self.assertIn("no referenced body was acquired", report)
        self.assertIn("the source bodies are not redistributed", report)

    def test_manifest_statistics_token_method_and_artifact_hashes(self):
        manifest = read_json("manifest.json")
        self.assertTrue(manifest["direct_training_ready"])
        self.assertEqual(manifest["rights_status"], "journal-paper-not-subject-to-us-copyright; foreign-and-third-party-caveats-recorded")
        statistics = manifest["statistics"]
        self.assertEqual(statistics["pdf_pages_manually_reviewed"], 12)
        self.assertEqual(statistics["figures_excluded"], 4)
        self.assertEqual(statistics["tables_excluded"], 5)
        self.assertEqual(statistics["scan_image_and_mask_objects_excluded"], 36)
        self.assertEqual(statistics["corpus_bytes"], 6989)
        self.assertEqual(statistics["corpus_words"], 980)
        self.assertEqual(statistics["corpus_lexical_tokens"], 1242)
        self.assertEqual(manifest["token_count_method"], "Unicode regex lexical tokens: \\w+(?:[’'-]\\w+)*|[^\\w\\s]")
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


if __name__ == "__main__":
    unittest.main()
