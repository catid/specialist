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


class CoirRopesFrenchPolynesiaCorpusTests(unittest.TestCase):
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
            "CORPUS.md": "3cc2a6007de29f2e8d8f6eda186fa75f4ce7a1064fb0a86c8e84b55ba89f7948",
            "README.md": "1688e9439dd3e62e9e3b7a4be502cd36d211e105cb6c0dea52c45b21f14b5d70",
            "REPORT.md": "3c73980c9a5a599736f1030ae95ee9b02a23c82fcab23b72bcefc4b674173b30",
            "components.jsonl": "e683a67eca3b72d534f5939dd35f92eedeedf65785159b58e69ca74666941d19",
            "dispositions.jsonl": "a0600b883ebc3281fbeaf5f580703a82297f6e5dacac1d0564c94e962a1fc16a",
            "source_snapshot/provenance.json": "40d520f2bc8bfc690a934e6a25e9412953db359915a848da16c30d823c06100e",
            "sources.jsonl": "b146dbc52f78f94a6e7868abca12445aed6c8599f831b918798718ba1a63ab32",
            "surfaces.jsonl": "775aae189335e1b087e2b5825721ee6285de6ed1e6f5c5f3721289dbc2f07cfd",
        }
        self.assertEqual({path: sha256(path) for path in expected}, expected)

    def test_exact_pdf_identity_checksum_and_metadata(self):
        rows = {row["source_id"]: row for row in read_jsonl("sources.jsonl")}
        self.assertEqual(set(rows), {"IFREMER-COIR-ROPE-2024-PDF", "DOI-CSL-COIR-ROPE-2024"})
        pdf = rows["IFREMER-COIR-ROPE-2024-PDF"]
        self.assertEqual(
            pdf["authors"],
            ["Louis Le Gué", "Peter Davies", "Mael Arhant", "Benoit Vincent", "Benoit Parnaudeau"],
        )
        self.assertEqual(pdf["doi"], "10.1016/j.clcb.2024.100111")
        self.assertEqual(pdf["publication_year"], 2024)
        self.assertEqual(pdf["article_number"], "100111")
        self.assertEqual(pdf["body_bytes"], 14365672)
        self.assertEqual(pdf["pdf_pages"], 11)
        self.assertEqual(
            pdf["body_sha256"],
            "385b2cb3184a64c4c78979a894d3f715d449197ed526a0af5149767996ec85f3",
        )
        self.assertEqual(
            pdf["body_sha512"],
            "3412b3ca1e7fabb893a70bbe760bc7e916ffd990772a451bc76dc34bbd1da606d3564d70d897e27440c996f623f126edb9c552d231d40b21a4b1369876375da4",
        )
        self.assertTrue(pdf["manual_page_audit"] and pdf["manual_component_audit"])
        self.assertTrue(pdf["independent_component_audit"])
        self.assertFalse(pdf["source_body_redistributed"])

    def test_doi_metadata_checksum_identity_license_and_funder_boundary(self):
        doi = next(row for row in read_jsonl("sources.jsonl") if row["source_id"] == "DOI-CSL-COIR-ROPE-2024")
        self.assertEqual(doi["body_bytes"], 16952)
        self.assertEqual(doi["body_sha256"], "d690a247716845903ee16a4639bb2f392e3bb7aadedd6df16f14cb77203edfeb")
        self.assertEqual(
            doi["body_sha512"],
            "a77360294468beced7cc7d971b98cf4777aba0afc9f046c4171883c08f2a12cf8e2c0073f307a90d6248f7d5d84c841d1c358f34d6612250e843b82fa150b726",
        )
        self.assertTrue(doi["title_author_doi_issue_match_pdf"])
        self.assertEqual(doi["version_of_record_license"], "CC BY 4.0")
        self.assertFalse(doi["funder_field_present"])
        self.assertEqual(doi["reference_metadata_entries"], 43)
        self.assertFalse(doi["reference_metadata_retained"] or doi["source_body_redistributed"])

    def test_provenance_rights_access_pdf_and_forbidden_artifacts(self):
        provenance = read_json("source_snapshot/provenance.json")
        identity = provenance["source_identity"]
        self.assertEqual(identity["doi"], "10.1016/j.clcb.2024.100111")
        self.assertEqual(identity["available_online"], "2024-10-05")
        self.assertTrue(identity["identity_cross_checked_between_pdf_xmp_first_page_and_doi_metadata"])
        access = provenance["access_audit"]
        self.assertFalse(access["site_crawl_performed"] or access["search_endpoint_used"])
        self.assertEqual(access["cited_source_bodies_acquired"], 0)
        self.assertEqual(access["source_bodies_in_package"], 0)
        metadata = provenance["pdf_metadata"]
        self.assertEqual(metadata["pages"], 11)
        self.assertTrue(metadata["tagged"] and metadata["usable_text_layer"])
        self.assertFalse(metadata["encrypted"] or metadata["javascript"] or metadata["forms"])
        self.assertEqual(metadata["embedded_files"], 0)
        self.assertEqual(metadata["embedded_raster_objects"], 7)
        self.assertEqual(metadata["numbered_figures"], 15)
        self.assertEqual(metadata["numbered_tables"], 6)
        rights = provenance["rights"]
        self.assertEqual(rights["license_identifier"], "CC-BY-4.0")
        self.assertTrue(rights["license_visible_on_pdf_first_page"])
        self.assertTrue(rights["license_in_pdf_xmp"] and rights["license_in_doi_metadata_for_version_of_record"])
        self.assertTrue(rights["attribution_provided"] and rights["change_notice_provided"])
        self.assertFalse(rights["endorsement_implied"])
        boundary = provenance["forbidden_artifact_boundary"]
        self.assertTrue(all(value == "not opened or searched" for value in boundary.values()))

    def test_every_pdf_page_has_one_manual_disposition_and_page_10_is_split(self):
        rows = read_jsonl("dispositions.jsonl")
        self.assertEqual(len(rows), 11)
        self.assertEqual([row["pdf_page"] for row in rows], list(range(1, 12)))
        self.assertEqual([row["logical_page"] for row in rows], list(range(1, 12)))
        self.assertFalse(rows[0]["visible_page_numeral"])
        self.assertTrue(all(row["visible_page_numeral"] for row in rows[1:]))
        self.assertTrue(all(row["audit"] == "manual text-layer and rendered-page review plus independent component audit" for row in rows))
        page_10 = rows[9]
        self.assertEqual(page_10["decision"], "component-split-partial-retain")
        self.assertIn("right-column reference list", page_10["excluded"])
        self.assertIn("column interleaving explicitly resolved", page_10["excluded"])
        self.assertEqual(rows[10]["decision"], "exclude")
        self.assertEqual(rows[10]["retained"], "none")

    def test_all_figures_tables_and_raster_objects_are_excluded(self):
        rows = read_jsonl("components.jsonl")
        self.assertEqual(len(rows), 32)
        figures = [row for row in rows if row["component_type"] == "figure"]
        tables = [row for row in rows if row["component_type"] == "table"]
        self.assertEqual([row["component_id"] for row in figures], [f"figure-{n}" for n in range(1, 16)])
        self.assertEqual(
            [row["pdf_pages"] for row in figures],
            [[2], [3], [3], [4], [5], [5], [5], [6], [6], [7], [7], [8], [9], [9], [10]],
        )
        self.assertEqual([row["component_id"] for row in tables], [f"table-{n}" for n in range(1, 7)])
        self.assertEqual([row["pdf_pages"] for row in tables], [[2], [2], [4], [5], [6], [8]])
        self.assertTrue(all(row["decision"] == "exclude" and row["retained"] == "none" for row in figures + tables))
        raster = next(row for row in rows if row["component_id"] == "embedded-raster-object-inventory")
        self.assertEqual(raster["embedded_raster_objects"], 7)
        self.assertEqual(raster["publisher_branding_objects"], 2)
        self.assertEqual(raster["composite_figure_objects"], 5)
        sem = next(row for row in rows if row["component_id"] == "externally-contributed-sem-images")
        self.assertEqual(sem["credited_provider"], "Nicolas Gayet")
        self.assertEqual(sem["figure_components"], ["figure-1", "figure-4", "figure-5", "figure-8"])
        self.assertEqual(sem["decision"], "exclude")

    def test_back_matter_references_funding_interest_and_data_are_precise(self):
        rows = {row["component_id"]: row for row in read_jsonl("components.jsonl")}
        self.assertEqual(rows["reference-list"]["reference_entries"], 43)
        self.assertEqual(rows["reference-list"]["decision"], "exclude")
        self.assertEqual(rows["doi-reference-metadata"]["decision"], "exclude")
        self.assertEqual(rows["acknowledgements"]["decision"], "exclude")
        funding = rows["funding-statement-absence"]
        self.assertEqual(funding["decision"], "retain-absence-only")
        self.assertIn("absence does not establish no funding", funding["retained"])
        conflict = rows["competing-interest-statement"]
        self.assertIn("Polyacht supplied materials", conflict["retained"])
        self.assertIn("not independent verification", conflict["retained"])
        data = rows["data-availability-statement"]
        self.assertEqual(data["decision"], "retain")
        self.assertIn("made available on request", data["retained"])

    def test_all_source_surfaces_have_manual_decisions(self):
        rows = read_jsonl("surfaces.jsonl")
        self.assertEqual(len(rows), 18)
        self.assertTrue(all(row["decision"] and row["audit"].startswith("manual") for row in rows))
        references = next(row for row in rows if row["surface_id"] == "references")
        self.assertEqual(references["location"], "page 10 right column and page 11")
        self.assertIn("all 43 reference entries", references["excluded"])
        funding = next(row for row in rows if row["surface_id"] == "funding-disclosure-audit")
        self.assertEqual(funding["decision"], "retain-absence-only")
        doi = next(row for row in rows if row["surface_id"] == "doi-csl-metadata")
        self.assertIn("absence of funder field", doi["retained"])

    def test_manual_review_and_novelty_boundary(self):
        provenance = read_json("source_snapshot/provenance.json")
        review = provenance["manual_review"]
        self.assertTrue(review["all_pdf_pages_text_screened_sequentially"])
        self.assertTrue(review["all_pdf_pages_rendered_and_visually_inspected"])
        self.assertTrue(review["independent_component_audit_completed"])
        self.assertTrue(review["page_10_column_interleaving_manually_resolved"])
        self.assertEqual(review["ordered_contact_sheets"], 2)
        self.assertEqual(review["reference_entries_reconciled"], 43)
        self.assertFalse(review["existing_corpus_bodies_inspected"])
        self.assertFalse(review["qa_trainer_eval_holdout_ood_probe_or_experiment_artifacts_inspected"])
        novelty = provenance["novelty_audit"]
        self.assertEqual(novelty["searched_existing_file_classes"], ["manifest.json", "README.md", "REPORT.md"])
        self.assertEqual(novelty["allowed_files_scanned"], 88)
        self.assertFalse(novelty["existing_corpus_bodies_searched_or_opened"])
        self.assertEqual(len(novelty["candidate_phrases_with_zero_file_matches"]), 7)

    def test_corpus_is_concise_cited_and_counted(self):
        corpus = read_text("CORPUS.md")
        self.assertEqual(len(corpus.encode("utf-8")), 7296)
        self.assertEqual(len(corpus.split()), 954)
        self.assertEqual(len(LEXICAL_TOKEN_RE.findall(corpus)), 1254)
        paragraphs = [part.strip() for part in corpus.split("\n\n") if not part.startswith("#")]
        self.assertEqual(len(paragraphs), 19)
        self.assertEqual(sum(part.startswith("Source evidence:") for part in paragraphs), 15)
        self.assertEqual(sum(part.startswith("Source boundary:") for part in paragraphs), 4)
        self.assertTrue(all(re.search(r"\[[^\[\]]+\]$", part) for part in paragraphs))
        self.assertNotRegex(corpus, r"https?://|www\.")
        self.assertNotRegex(corpus, r"!\[[^\]]*\]\([^)]*\)")
        self.assertNotRegex(corpus, r"(?m)^\s*\|.*\|\s*$")

    def test_required_evidence_and_limits_are_present(self):
        corpus = read_text("CORPUS.md").lower()
        for phrase in (
            "10.1016/j.clcb.2024.100111",
            "pearl farming in french polynesia",
            "individual coir fibers and finished coir ropes",
            "machine-twisted and hand-braided constructions",
            "dry and water-saturated laboratory states",
            "fiber-to-rope prediction limit",
            "construction and braiding quality",
            "hand-braided ropes also differed from one another",
            "fiber length alone likewise did not explain",
            "no distinctive trend or statistically significant change",
            "did not report in-situ aging",
            "production and transport only",
            "excluded use and end-of-life because those data were unavailable",
            "no dedicated funding section or funding-source statement",
            "that absence is not evidence that the work received no funding",
            "polyacht supplied materials",
            "benoit parnaudeau was a polyacht employee",
            "did not influence the study’s design, interpretation, or reporting",
            "data will be made available on request",
            "do not establish service life",
        ):
            self.assertIn(phrase, corpus)

    def test_no_values_recipes_visuals_products_or_reference_leakage(self):
        corpus = read_text("CORPUS.md")
        lower = corpus.lower()
        evidence_only = "\n".join(
            paragraph for paragraph in corpus.split("\n\n") if paragraph.startswith("Source evidence:")
        ).lower()
        for forbidden in (
            "figure 1", "figure 2", "figure 3", "figure 4", "figure 5",
            "table 1", "table 2", "table 3", "table 4", "table 5", "table 6",
            "dynamic vapour", "scanning electron", "scikit-image", "ecoinvent",
            "openlca", "load cell", "pneumatic clamp", "distilled water", "tap water",
            "freshwater", "seawater", "defibrator", "sieve shaker", "skeleton method",
            "hdpe", "benchmark", "best rope", "product rating", "recommended product",
            "agrios", "andrady", "bismarck", "kulkarni", "mohanty", "rajan",
            "recipe", "protocol", "immersion time", "soak time",
        ):
            self.assertNotIn(forbidden, evidence_only)
        without_citations = re.sub(r"\[[^\[\]]+\]", "", corpus)
        self.assertNotRegex(
            without_citations,
            r"(?i)\b\d+(?:\.\d+)?\s*(?:%|percent|mm|cm|km|g|kg|n|tex|hours?|weeks?|degrees?|wh|kwh|gwh)\b",
        )
        self.assertNotRegex(lower, r"(?i)\b(?:select|buy|purchase|certify|approve|use)\s+(?:the|this|a)\s+(?:rope|product)\b")

    def test_funding_absence_is_not_misrepresented(self):
        corpus = read_text("CORPUS.md").lower()
        provenance = read_json("source_snapshot/provenance.json")["funding_disclosure"]
        self.assertIn("contains no dedicated funding section or funding-source statement", corpus)
        self.assertIn("absence is not evidence that the work received no funding", corpus)
        for forbidden in (
            "the study had no funding",
            "the work had no funding",
            "the research had no funding",
            "the study was not funded",
            "the work was not funded",
        ):
            self.assertNotIn(forbidden, corpus)
        self.assertFalse(provenance["dedicated_pdf_funding_section_present"])
        self.assertFalse(provenance["pdf_funding_source_statement_present"])
        self.assertFalse(provenance["doi_metadata_funder_field_present"])
        self.assertFalse(provenance["absence_interpreted_as_no_funding"])

    def test_no_modern_material_care_load_or_human_transfer(self):
        corpus = read_text("CORPUS.md").lower()
        boundary = corpus[corpus.index("## absolute non-transfer boundary") :]
        for phrase in (
            "nothing retained here rates, selects, certifies, approves, or recommends a rope or product",
            "no laboratory result, construction label, wet-retention observation, retting result, or life-cycle boundary establishes current quality, safety, remaining life, strength, working load, or suitability for a different task",
            "jute",
            "hemp",
            "rope hygiene",
            "cleaning",
            "drying",
            "storage",
            "care",
            "retirement",
            "body contact",
            "knots",
            "frictions",
            "restraint",
            "body lowering",
            "working loads",
            "uplines",
            "anchors",
            "hardpoints",
            "bondage",
            "human suspension",
            "no instruction or safety rule",
        ):
            self.assertIn(phrase, boundary)

    def test_rights_attribution_change_and_nonendorsement(self):
        corpus = read_text("CORPUS.md").lower()
        readme = read_text("README.md").lower()
        report = read_text("REPORT.md").lower()
        self.assertIn("creative commons attribution 4.0 license", corpus)
        self.assertIn("cc by permits reuse with attribution, license notice, and an indication of changes", corpus)
        self.assertIn("does not imply endorsement", corpus)
        self.assertIn("manually rewritten, quotation-light, narrowed derivative", readme)
        self.assertIn("externally contributed sem images", readme)
        self.assertIn("no endorsement is implied", report)
        self.assertIn("source body and doi metadata are checksum-bound but not redistributed", report)

    def test_manifest_statistics_token_method_and_artifact_hashes(self):
        manifest = read_json("manifest.json")
        self.assertTrue(manifest["direct_training_ready"])
        self.assertEqual(manifest["rights_status"], "cc-by-4.0; attribution-license-change-and-component-boundaries-recorded")
        statistics = manifest["statistics"]
        self.assertEqual(statistics["pdf_pages_manually_reviewed"], 11)
        self.assertEqual(statistics["figures_excluded"], 15)
        self.assertEqual(statistics["tables_excluded"], 6)
        self.assertEqual(statistics["embedded_raster_objects_excluded"], 7)
        self.assertEqual(statistics["reference_entries_excluded"], 43)
        self.assertEqual(statistics["corpus_bytes"], 7296)
        self.assertEqual(statistics["corpus_words"], 954)
        self.assertEqual(statistics["corpus_lexical_tokens"], 1254)
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
