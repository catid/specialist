from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = read_json(ROOT / "manifest.json")
        cls.provenance = read_json(ROOT / "source_snapshot/provenance.json")
        cls.sources = read_jsonl(ROOT / "sources.jsonl")
        cls.dispositions = read_jsonl(ROOT / "dispositions.jsonl")
        cls.corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        cls.report = (ROOT / "REPORT.md").read_text(encoding="utf-8")

    def test_manifest_declares_one_direct_non_qa_document(self) -> None:
        self.assertEqual(self.manifest["schema"], "site-corpus-manifest-v1")
        self.assertEqual(
            self.manifest["resource_id"],
            "mdpi_tree_climbing_friction_hitch_2021",
        )
        self.assertEqual(
            self.manifest["artifact_role"],
            "canonical_markdown_direct_training_source",
        )
        self.assertTrue(self.manifest["direct_training_ready"])
        self.assertTrue(self.manifest["non_qa"])
        self.assertEqual(self.manifest["training_document_count"], 1)
        self.assertEqual(self.manifest["direct_training_word_count"], 2945)
        self.assertEqual(self.manifest["whitespace_delimited_word_count"], 2913)

    def test_article_identity_is_exact(self) -> None:
        article = self.provenance["article"]
        self.assertEqual(
            article["title"],
            "Tensile Strength of Ropes and Friction Hitch Used in Tree Climbing Work",
        )
        self.assertEqual(len(article["authors"]), 6)
        self.assertEqual(article["journal"], "Forests")
        self.assertEqual(article["year"], 2021)
        self.assertEqual(article["volume"], 12)
        self.assertEqual(article["issue"], 11)
        self.assertEqual(article["article_number"], 1457)
        self.assertEqual(article["publication_date"], "2021-10-26")
        self.assertEqual(article["doi"], "10.3390/f12111457")

    def test_five_source_ledger_records_have_exact_identities(self) -> None:
        self.assertEqual(len(self.sources), 5)
        by_id = {source["source_id"]: source for source in self.sources}
        self.assertEqual(
            set(by_id), {"MDPI-PDF", "MDPI-HTML", "DOI", "CC-BY", "CC-LEGAL"}
        )
        pdf = by_id["MDPI-PDF"]
        self.assertEqual(pdf["raw_byte_length"], 17049016)
        self.assertEqual(
            pdf["raw_sha256"],
            "55832e9cb3ed23942315922fad01a5c79fd239e0fcf50b690fddb58bd519910d",
        )
        self.assertEqual(pdf["retrieved_at"], "2026-07-16T10:58:54Z")
        self.assertEqual(pdf["http_status"], 200)
        self.assertEqual(pdf["content_type"], "application/pdf")
        self.assertEqual(pdf["last_modified"], "Tue, 26 Oct 2021 07:06:26 GMT")
        doi = by_id["DOI"]
        self.assertEqual(doi["http_status"], 302)
        self.assertEqual(doi["response_body_byte_length"], 167)
        self.assertEqual(
            doi["response_body_sha256"],
            "d0fcd73743f846637da34424e1b43c247b1918f81fa4044bd14f6f23114ee8bb",
        )
        self.assertEqual(
            doi["location"], "https://www.mdpi.com/1999-4907/12/11/1457"
        )

    def test_cc_source_hashes_and_total_bytes_are_exact(self) -> None:
        by_id = {source["source_id"]: source for source in self.sources}
        deed = by_id["CC-BY"]
        legal = by_id["CC-LEGAL"]
        self.assertEqual(deed["raw_byte_length"], 32178)
        self.assertEqual(
            deed["raw_sha256"],
            "231a5dac65bbf135ba27145969a63cd289faadc172f1512c4810a6c60ba91036",
        )
        self.assertEqual(legal["raw_byte_length"], 48970)
        self.assertEqual(
            legal["raw_sha256"],
            "6d55b998ed5c54f43426d059a8c549ed58a3321e5463e6a6af1c6b56ab78c333",
        )
        summary = self.provenance["retrieval_summary"]
        self.assertEqual(summary["hashed_successful_body_byte_length_total"], 17130331)
        self.assertEqual(summary["article_pdf_byte_length"], 17049016)
        self.assertEqual(summary["doi_response_body_byte_length"], 167)
        self.assertEqual(summary["license_metadata_byte_length_total"], 81148)

    def test_pdf_complete_review_metrics_are_exact(self) -> None:
        pdf = next(source for source in self.sources if source["source_id"] == "MDPI-PDF")
        self.assertEqual(pdf["pdf_page_count"], 12)
        self.assertEqual(pdf["pdf_creation_date"], "2021-10-26T06:47:41Z")
        self.assertEqual(pdf["pdf_modification_date"], "2021-10-26T07:06:26Z")
        self.assertEqual(pdf["audit_text_extractor"], "pypdf 6.14.2")
        self.assertEqual(pdf["audit_text_byte_length"], 39105)
        self.assertEqual(pdf["audit_text_word_count"], 6149)
        self.assertEqual(
            pdf["audit_text_sha256"],
            "78d26d2c0e2b155f421dfc46534143aac1cbbb49929c82c8483d77a09861e01d",
        )
        self.assertEqual(pdf["raster_image_occurrences"], 10)
        audit = self.provenance["content_audit"]
        self.assertEqual(audit["pdf_digital_page_count"], 12)
        self.assertEqual(audit["pdf_raster_image_occurrences"], 10)
        self.assertEqual(audit["source_images_reused"], 0)
        self.assertEqual(audit["source_table_layouts_reused"], 0)
        self.assertEqual(audit["source_references_reproduced"], 0)

    def test_html_access_denial_is_not_misrepresented_as_article_content(self) -> None:
        html = next(source for source in self.sources if source["source_id"] == "MDPI-HTML")
        self.assertEqual(html["browser_http_status"], 200)
        self.assertEqual(html["browser_rendered_line_count"], 493)
        self.assertEqual(html["shell_http_status"], 403)
        self.assertEqual(html["shell_access_denial_byte_length"], 402)
        self.assertEqual(
            html["shell_access_denial_sha256"],
            "8c5572a5ee28579ebd140703df19e31ccb28e1830a83ee5c77398e95b4d88df4",
        )
        self.assertIsNone(html["raw_article_content_sha256"])
        self.assertIn("denial digest", self.report)
        self.assertIn("not misrepresented as article content", self.report)

    def test_network_stayed_on_official_article_doi_and_cc_routes(self) -> None:
        network = self.provenance["network_boundary"]
        self.assertEqual(len(network["successful_or_identity_routes"]), 5)
        self.assertEqual(len(network["unsuccessful_official_article_routes"]), 4)
        self.assertIn("official MDPI article PDF", network["official_pdf_asset_access"])
        self.assertIn("302", network["doi_resolution"])
        for key in (
            "article_reference_routes_accessed",
            "standards_routes_accessed",
            "manufacturer_routes_accessed",
            "image_routes_accessed",
            "promotional_routes_accessed",
            "other_third_party_routes_accessed",
        ):
            self.assertFalse(network[key])

    def test_raw_sources_extractions_images_and_tables_are_not_retained(self) -> None:
        files = sorted(
            str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file()
        )
        self.assertFalse(any(name.endswith((".pdf", ".html", ".xml")) for name in files))
        self.assertFalse(any("extract" in name.lower() for name in files))
        self.assertFalse(any(name.endswith((".png", ".jpg", ".jpeg", ".svg")) for name in files))
        self.assertEqual(
            sorted(path.name for path in (ROOT / "source_snapshot").iterdir()),
            ["provenance.json"],
        )
        method = self.provenance["normalization_and_review_method"]
        self.assertFalse(method["raw_pdf_retained"])
        self.assertFalse(method["raw_html_retained"])
        self.assertFalse(method["audit_text_retained"])
        self.assertFalse(method["raw_source_direct_training"])

    def test_cc_by_attribution_changes_and_no_endorsement_are_complete(self) -> None:
        rights = self.provenance["rights_review"]
        self.assertEqual(
            rights["license_name"], "Creative Commons Attribution 4.0 International"
        )
        self.assertEqual(rights["license_identifier"], "CC BY 4.0")
        self.assertTrue(rights["title_attributed"])
        self.assertTrue(rights["all_authors_attributed"])
        self.assertTrue(rights["doi_linked"])
        self.assertTrue(rights["canonical_article_linked"])
        self.assertTrue(rights["license_linked"])
        self.assertTrue(rights["adaptation_marked"])
        self.assertTrue(rights["changes_described"])
        self.assertFalse(rights["endorsement_implied"])
        self.assertFalse(rights["additional_restrictions_added"])
        self.assertFalse(rights["legal_opinion"])
        self.assertFalse(rights["license_sources_direct_training"])
        for phrase in (
            "by Leonardo Bianchini, Rodolfo Picchio, Andrea Colantoni, Marco Scotolati, Valerio Di Stefano, and Massimo Cecchini",
            "Creative Commons Attribution 4.0 International licence",
            "Changes made:",
            "No author, publisher, manufacturer, or institution endorses this adaptation",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(self.manifest["rights_status"], "cc_by_4_0_attributed_adapted_no_endorsement")

    def test_limitations_precede_research_question_and_results(self) -> None:
        limitations = self.corpus.index("## Limitations first")
        question = self.corpus.index("## Research question and stated purpose")
        results = self.corpus.index("## Observed system-level extrema")
        self.assertLess(limitations, question)
        self.assertLess(limitations, results)
        for phrase in (
            "preliminary laboratory study of two particular synthetic arborist rope systems",
            "not a field trial and did not place a person on the system",
            "slow pulls on semi-static rope",
            "short 250 mm machine stroke",
            "no dynamic drops, falls, catches, shock loads",
            "adapted a standardized fall-protection-rope test method",
            "not universal breaking loads",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["loading_scope_status"],
            "slow_semistatic_250mm_modified_bench_method_no_dynamic_or_cyclic_transfer",
        )

    def test_exact_two_rope_systems_and_hitch_cord_are_retained(self) -> None:
        for phrase in (
            "**System A:** the article's Axis model, an 11 mm synthetic rope listed as polyester/polyamide",
            "**System X:** the article's XTC 16 model, a synthetic polyamide rope listed as 13 mm",
            "Both systems used an 8 mm polyester/Technora hitch cord",
            "later calls the larger rope 12.7 mm",
            "preserves that source discrepancy",
        ):
            self.assertIn(phrase, self.corpus)
        design = self.provenance["study_design_audit"]
        self.assertEqual(len(design["rope_systems"]), 2)
        self.assertEqual(design["hitch_cord"], "8 mm polyester/Technora cord")
        self.assertEqual(
            self.manifest["specimen_status"],
            "axis_11mm_and_xtc16_13mm_with_8mm_synthetic_hitch_cord_exact_only",
        )

    def test_two_hitches_four_treatments_nine_runs_and_36_total_are_exact(self) -> None:
        for phrase in (
            "**Prusik** (`P`) and **Valdotain tresse** (`T`)",
            "`AP`: Axis with Prusik",
            "`AT`: Axis with Valdotain tresse",
            "`XP`: XTC 16 with Prusik",
            "`XT`: XTC 16 with Valdotain tresse",
            "There were 36 traction tests: four treatments with nine runs per treatment",
            "two work ropes × two hitches × nine runs equals 36 tests",
        ):
            self.assertIn(phrase, self.corpus)
        design = self.provenance["study_design_audit"]
        self.assertEqual(design["test_count"], 36)
        self.assertEqual(design["treatment_count"], 4)
        self.assertEqual(design["runs_per_treatment"], 9)
        self.assertEqual(design["treatments"], ["AP", "AT", "XP", "XT"])
        self.assertEqual(
            self.manifest["factorial_design_status"],
            "two_ropes_by_two_hitches_four_treatments_nine_runs_each_36_total",
        )

    def test_replication_structure_is_not_overstated(self) -> None:
        for phrase in (
            "three pieces of each rope and three cords",
            "three traction tests carried out on each set",
            "not described simply as nine independently sourced, fresh, complete systems",
            "large number of recorded points does not create the material diversity",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["replication_status"],
            "nine_runs_per_treatment_with_three_piece_three_cord_repeated_test_structure",
        )

    def test_bench_observation_window_and_measures_are_complete(self) -> None:
        for phrase in (
            "250 mm available stroke",
            "five later positions separated by 50 mm",
            "applied tensile load on the assembled system, `C`, in kilonewtons",
            "friction-hitch slip along the work rope, `S`, in centimeters",
            "change in work-rope diameter, `Δf`, in millimeters",
            "temperature near the hitch at the beginning and end",
            "Ambient temperature and humidity were recorded",
        ):
            self.assertIn(phrase, self.corpus)
        design = self.provenance["study_design_audit"]
        self.assertEqual(design["bench_stroke_mm"], 250)
        self.assertEqual(design["observation_positions"], 6)
        self.assertEqual(design["position_spacing_mm"], 50)
        self.assertEqual(len(design["measured_outcomes"]), 5)
        self.assertEqual(
            self.manifest["measurement_status"],
            "load_slip_diameter_change_endpoint_temperature_and_ambient_context",
        )

    def test_statistical_plan_and_model_limits_are_retained(self) -> None:
        for phrase in (
            "checked distributional assumptions and variance homogeneity",
            "one-factor analysis of variance",
            "factorial multivariate analysis of variance",
            "rope (`A` or `X`) and hitch (`P` or `T`) as factors",
            "family-wise 95% confidence level",
            "`p = 0.05`",
            "These models describe covariation in the observed bench data",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["statistics_status"],
            "factorial_multivariate_posthoc_and_regression_relationships_not_field_calculators",
        )

    def test_observed_extrema_are_kept_distinct_from_ratings_and_means(self) -> None:
        for phrase in (
            "observed maximum load of 18.7 kN",
            "maximum hitch slip of 9.6 cm",
            "maximum diameter change of 3 mm",
            "must not be relabeled as minimum breaking strengths, safe working loads",
            "11.9 kN for `AP`",
            "15.1 kN for `XP`",
            "9.3 kN for `AT`",
            "8.6 kN for `XT`",
            "individual 18.7 kN maximum and these treatment means describe different summaries",
        ):
            self.assertIn(phrase, self.corpus)
        extrema = self.provenance["results_audit"]["abstract_extrema"]
        self.assertEqual(extrema["observed_maximum_load_kn"], 18.7)
        self.assertEqual(extrema["observed_maximum_slip_cm"], 9.6)
        self.assertEqual(extrema["observed_maximum_diameter_change_mm"], 3.0)
        self.assertIn("never a safe load", extrema["rating_status"])
        self.assertEqual(
            self.manifest["rating_status"],
            "observed_values_only_no_breaking_strength_working_load_safety_factor_or_threshold",
        )

    def test_hitch_effects_and_coupled_tradeoff_are_retained(self) -> None:
        for phrase in (
            "hitch type had a statistically significant effect on load and on slip",
            "reported `p < 0.001` for each",
            "Prusik treatments developed greater grip, produced higher machine load, and slipped less",
            "does not mean “more grip is safer” or “less slip is better”",
            "Greater grip co-occurred with greater load and more diameter reduction",
            "did not define an optimal balance",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["hitch_result_status"],
            "hitch_affected_load_and_slip_prusik_more_grip_higher_load_less_slip_no_safer_claim",
        )

    def test_rope_effect_and_confounded_specimen_properties_are_retained(self) -> None:
        for phrase in (
            "Rope type had a statistically significant effect on diameter change",
            "XTC 16 specimen showed greater diameter deformation",
            "cannot isolate which of those correlated specimen differences caused the effect",
            "rope model bundles several properties",
            "not a clean test of diameter alone or polymer alone",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["rope_result_status"],
            "rope_affected_diameter_xtc_more_deformation_no_single_property_causal_claim",
        )

    def test_regression_relationships_and_non_extrapolation_are_complete(self) -> None:
        for phrase in (
            "`0.445` for load versus diameter reduction",
            "`0.418` for load versus hitch slip",
            "`0.378` for diameter reduction versus slip",
            "adjusted `R² = 0.535`",
            "included nonlinear terms",
            "substantial variation remained",
            "do not validate extrapolation beyond the stroke",
        ):
            self.assertIn(phrase, self.corpus)
        fits = self.provenance["results_audit"]["adjusted_r_squared"]
        self.assertEqual(fits["load_vs_diameter_reduction"], 0.445)
        self.assertEqual(fits["load_vs_slip"], 0.418)
        self.assertEqual(fits["diameter_reduction_vs_slip"], 0.378)
        self.assertEqual(fits["combined_load_vs_slip_and_diameter_reduction"], 0.535)
        self.assertEqual(
            self.manifest["regression_status"],
            "partial_nonlinear_associations_no_causality_failure_prediction_or_extrapolation",
        )

    def test_temperature_results_are_endpoints_not_dynamic_assurance(self) -> None:
        for phrase in (
            "6.2 °C (`AP`)",
            "6.8 °C (`AT`)",
            "6.7 °C (`XP`)",
            "3.1 °C (`XT`)",
            "did not find a significant difference among treatments",
            "not the same as “no heating occurred”",
            "no dynamic slide, fall, or repeated cycle",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["temperature_status"],
            "start_end_changes_no_treatment_difference_no_dynamic_peak_or_heat_safety_claim",
        )

    def test_preliminary_no_best_knot_purpose_is_repeated(self) -> None:
        for phrase in (
            "The authors expressly said the goal was **not** to define the best knot",
            "Any claim that the experiment selected a universally superior hitch contradicts its stated purpose",
            "preliminary comparison raised research questions; it did not identify a best knot",
            "authors sought relationships and future research directions, not a best-knot ranking",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["purpose_status"],
            "preliminary_relationship_study_explicitly_not_best_knot_selection",
        )

    def test_dynamic_cyclic_fall_and_human_domains_are_explicitly_untested(self) -> None:
        for phrase in (
            "There were no dynamic drops, falls, catches, shock loads",
            "or cyclic hysteresis trials",
            "future dynamic falling-body and cyclic-loading experiments",
            "did not place a person on the system",
            "no human body/casualty/physiology",
        ):
            if phrase == "no human body/casualty/physiology":
                self.assertIn("no human participant", self.report)
            else:
                self.assertIn(phrase, self.corpus)
        untested = self.provenance["limitation_boundary"]["untested_loading_domains"]
        self.assertEqual(len(untested), 9)
        self.assertIn("human suspension", untested)
        self.assertIn("cyclic loading", untested)
        self.assertEqual(
            self.manifest["dynamic_transfer_status"],
            "forbidden_no_drop_fall_shock_cycle_hysteresis_swing_live_climber_or_rescue_test",
        )

    def test_natural_fiber_bondage_and_human_suspension_transfer_is_forbidden(self) -> None:
        for phrase in (
            "did not test recreational rope bondage, human suspension, natural-fiber rope",
            "transfer from synthetic arborist equipment to natural-fiber rope",
            "transfer from occupational tree-climbing equipment to rope bondage, shibari, kinbaku",
            "anatomy, circulation, nerve safety, skin pressure, consent, emergency release",
        ):
            self.assertIn(phrase, self.corpus)
        forbidden = self.provenance["limitation_boundary"]["forbidden_transfers"]
        self.assertEqual(len(forbidden), 11)
        self.assertEqual(
            self.manifest["cross_domain_status"],
            "forbidden_no_natural_fiber_bondage_shibari_kinbaku_or_human_suspension_generalization",
        )

    def test_no_unsafe_recipe_trademark_promotion_or_standards_detail(self) -> None:
        lower = self.corpus.lower()
        for marker in (
            "to tie the hitch, first",
            "step-by-step",
            "preload the",
            "attach the carabiner",
            "wraps and braids",
            "manufacturer's instructions",
            "standard en ",
            "uni en",
            "safe working load is",
            "recommended working load",
            "22 kn",
            "24 kn",
        ):
            self.assertNotIn(marker, lower)
        self.assertNotRegex(self.corpus, r"\bEN\s*\d{3,5}\b")
        boundary = self.provenance["component_and_trademark_boundary"]
        self.assertEqual(boundary["minimally_retained_model_labels"], ["Axis", "XTC 16"])
        self.assertEqual(len(boundary["excluded_classes"]), 8)
        self.assertEqual(
            self.manifest["operational_detail_status"],
            "excluded_no_tying_assembly_preload_connection_climbing_suspension_or_rescue_recipe",
        )

    def test_claim_level_citations_are_valid_and_numerous(self) -> None:
        citations = re.findall(
            r"\[((?:MDPI-PDF|DOI|CC-BY)): ([^\]]+)\]", self.corpus
        )
        self.assertEqual(len(citations), 43)
        valid_ids = {source["source_id"] for source in self.sources}
        self.assertTrue(all(source_id in valid_ids for source_id, _ in citations))
        self.assertEqual(self.manifest["claim_level_citation_count"], 43)
        policy = self.provenance["citation_policy"]
        self.assertTrue(policy["claim_level_required"])
        self.assertTrue(policy["license_and_identity_citations_metadata_only"])

    def test_dispositions_are_complete_and_sequential(self) -> None:
        self.assertEqual(len(self.dispositions), 56)
        self.assertEqual(
            [row["record_id"] for row in self.dispositions],
            [f"mdpi-hitch-{n:03d}" for n in range(1, 57)],
        )
        self.assertEqual(
            sum(row["direct_training_included"] for row in self.dispositions), 36
        )
        self.assertEqual(
            sum(not row["direct_training_included"] for row in self.dispositions), 20
        )
        self.assertEqual(self.manifest["disposition_record_count"], 56)
        self.assertEqual(self.manifest["included_or_narrowed_disposition_count"], 36)
        self.assertEqual(self.manifest["excluded_or_metadata_disposition_count"], 20)

    def test_urls_are_only_required_article_doi_and_license_attribution(self) -> None:
        urls = re.findall(r"https?://[^)\s]+", self.corpus)
        self.assertEqual(
            urls,
            [
                "https://doi.org/10.3390/f12111457",
                "https://www.mdpi.com/1999-4907/12/11/1457",
                "https://creativecommons.org/licenses/by/4.0/",
            ],
        )
        self.assertIn("Do not generate recall questions", self.corpus)
        self.assertIn("DOI digits", self.corpus)

    def test_corpus_is_prose_not_chat_qa_or_identifier_drill(self) -> None:
        lower = self.corpus.lower()
        for marker in (
            "<|im_start|>",
            "<|im_end|>",
            "</think>",
            '"question":',
            '"answer":',
        ):
            self.assertNotIn(marker, lower)
        self.assertNotRegex(self.corpus, r"(?m)^Q:\s|^A:\s")
        policy = self.manifest["qa_derivation_policy"]
        for forbidden in (
            "author order",
            "DOI",
            "URL",
            "publication date",
            "manufacturer",
            "brand",
            "standard",
            "figure",
            "table",
            "reference",
            "file hash",
            "tying",
            "assembly",
            "rating",
            "recommendation",
        ):
            self.assertIn(forbidden, policy["forbidden"])
        self.assertTrue(policy["inherit_document_split"])

    def test_split_hygiene_and_protected_boundary(self) -> None:
        split = self.provenance["split_hygiene"]
        self.assertEqual(split["source_document_unit"], "doi-10.3390-f12111457")
        self.assertFalse(split["protected_data_accessed_during_build"])
        self.assertEqual(len(split["protected_data_classes_not_accessed"]), 13)
        self.assertIn("one source-family split", split["document_disjoint_requirement"])
        self.assertEqual(
            self.manifest["document_disjoint_requirement"],
            split["document_disjoint_requirement"],
        )
        self.assertIn("protected-QA", self.manifest["protected_split_requirement"])

    def test_output_hashes_and_lengths(self) -> None:
        for item in self.manifest["outputs"].values():
            path = ROOT / item["path"]
            self.assertEqual(item["byte_length"], path.stat().st_size)
            self.assertEqual(item["sha256"], sha256(path))

    def test_report_manifest_and_corpus_counts_match(self) -> None:
        regex_count = len(re.findall(r"[\w\u2019'-]+", self.corpus, flags=re.UNICODE))
        whitespace_count = len(self.corpus.split())
        h2_count = sum(line.startswith("## ") for line in self.corpus.splitlines())
        h3_count = sum(line.startswith("### ") for line in self.corpus.splitlines())
        citation_count = len(
            re.findall(r"\[(?:MDPI-PDF|DOI|CC-BY): [^\]]+\]", self.corpus)
        )
        self.assertEqual(regex_count, 2945)
        self.assertEqual(whitespace_count, 2913)
        self.assertEqual(h2_count, 16)
        self.assertEqual(h3_count, 0)
        self.assertEqual(citation_count, 43)
        self.assertEqual(self.manifest["direct_training_word_count"], regex_count)
        self.assertEqual(
            self.manifest["whitespace_delimited_word_count"], whitespace_count
        )
        self.assertEqual(self.manifest["h2_section_count"], h2_count)
        self.assertEqual(self.manifest["h3_subsection_count"], h3_count)
        self.assertIn("2,913-whitespace-word", self.report)
        self.assertIn("43 claim-level citations", self.report)
        self.assertIn("56 disposition records", self.report)


if __name__ == "__main__":
    unittest.main()
