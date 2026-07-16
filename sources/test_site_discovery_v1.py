#!/usr/bin/env python3
"""Contract tests for source discovery and canonical Markdown corpus queues."""

import json
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
CORPUS_QUEUE = ROOT / "site_corpus_queue_v1.json"
DISCOVERY_QUEUE = ROOT / "site_discovery_queue_v1.json"
CANDIDATE_LEDGER = ROOT / "site_discovery_candidates_v1.jsonl"
DISCOVERY_REPORT = ROOT / "site_discovery_report_v1.md"


SCORE_FIELDS = {
    "information_density": "information_density_score_0_to_5",
    "technical_specificity": "technical_specificity_score_0_to_5",
    "provenance_quality": "provenance_quality_score_0_to_5",
    "durability": "durability_score_0_to_5",
    "novel_coverage": "novel_coverage_score_0_to_5",
    "safety_context": "safety_context_score_0_to_5",
    "duplication_penalty": "duplication_penalty_0_to_5",
    "commerce_noise_penalty": "commerce_noise_penalty_0_to_5",
}


class SourceCorpusContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = json.loads(CORPUS_QUEUE.read_text(encoding="utf-8"))
        cls.discovery = json.loads(DISCOVERY_QUEUE.read_text(encoding="utf-8"))
        cls.candidates = [
            json.loads(line)
            for line in CANDIDATE_LEDGER.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.report = DISCOVERY_REPORT.read_text(encoding="utf-8")

    def test_markdown_and_qa_are_distinct_required_layers(self) -> None:
        layers = self.corpus["dataset_layers"]
        markdown = layers["canonical_markdown_source_corpus"]
        qa = layers["derived_manual_qa"]
        self.assertTrue(markdown["required"])
        self.assertTrue(markdown["direct_training_ready"])
        self.assertTrue(markdown["non_qa"])
        self.assertTrue(qa["must_link_to_markdown_source_record"])
        self.assertTrue(qa["automatic_generation_without_manual_review_forbidden"])

    def test_approved_taxonomy_is_complete_and_unique(self) -> None:
        expected = {
            "lineage_history_people",
            "ties_knots_frictions",
            "uplines_suspension_hardpoints",
            "anatomy_injury_prevention",
            "rope_materials_maintenance",
            "rigging_mechanics",
            "consent_scene_management",
            "bottoming_skills",
            "pattern_architecture",
            "emergency_procedures",
            "rope_handling",
            "teaching_evaluation",
            "accessibility_adaptation",
            "aesthetics_performance",
            "terminology_cultural_context",
        }
        ids = [row["category_id"] for row in self.corpus["knowledge_taxonomy"]["categories"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), expected)
        for row in self.corpus["knowledge_taxonomy"]["categories"]:
            self.assertGreaterEqual(len(row["coverage"].split()), 8)

    def test_every_requested_resource_has_one_unique_queue_entry(self) -> None:
        queue = self.corpus["queue"]
        ids = [row["resource_id"] for row in queue]
        urls = [row["url"] for row in queue]
        self.assertGreaterEqual(len(queue), 25)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(urls), len(set(urls)))
        self.assertIn("crash_restraint", ids)
        crash = next(row for row in queue if row["resource_id"] == "crash_restraint")
        self.assertIn("entire accessible site inventory", crash["scope"])

    def test_rope365_completion_record_is_preserved(self) -> None:
        rope365 = next(
            row for row in self.corpus["queue"] if row["resource_id"] == "rope365"
        )
        self.assertEqual(rope365["status"], "complete")
        self.assertEqual(rope365["worker"], "rope365_corpus_clean")
        self.assertEqual(
            rope365["commit"], "30222126ffaf67404417f8dd892c18c2f956e1e0"
        )
        self.assertEqual(
            set(rope365["artifacts"]), {"manifest", "markdown", "report", "tests"}
        )

    def test_discovery_is_evidence_first_and_separate_from_extraction(self) -> None:
        contract = self.discovery["discovery_contract"]
        self.assertTrue(contract["evaluate_the_actual_source_not_search_snippets"])
        self.assertTrue(contract["deduplicate_against_existing_corpus_queue"])
        self.assertTrue(contract["separate_source_discovery_from_corpus_extraction"])
        self.assertTrue(contract["sealed_eval_ood_shadow_holdout_and_existing_qa_access_forbidden"])
        self.assertIn("novel_coverage_score_0_to_5", self.discovery["candidate_fields"])
        self.assertIn("accept_targeted_scope", self.discovery["decisions"])

    def test_candidate_ledger_has_complete_manual_review_batches(self) -> None:
        required = set(self.discovery["candidate_fields"]) | {"review_batch"}
        batches = Counter()
        for row in self.candidates:
            self.assertTrue(required.issubset(row), row["candidate_id"])
            batches[row["review_batch"]] += 1
        self.assertTrue(batches)
        for batch_id, count in batches.items():
            self.assertRegex(batch_id, r"^discovery_batch_\d{3}$")
            self.assertGreaterEqual(count, 10)
            self.assertLessEqual(count, 20)

    def test_candidate_identity_access_and_evidence_fields_are_valid(self) -> None:
        ids = [row["candidate_id"] for row in self.candidates]
        urls = [row["canonical_url"] for row in self.candidates]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(urls), len(set(urls)))
        for row in self.candidates:
            parsed = urlparse(row["canonical_url"])
            self.assertIn(parsed.scheme, {"http", "https"})
            self.assertTrue(parsed.netloc)
            self.assertIs(type(row["accessible"]), bool)
            self.assertTrue(row["access_notes"].strip())
            self.assertTrue(row["discovered_from"])
            self.assertTrue(row["representative_pages"])
            datetime.fromisoformat(row["checked_at_utc"].replace("Z", "+00:00"))
            for page in row["representative_pages"]:
                self.assertEqual(set(page), {"url", "note"})
                self.assertIn(urlparse(page["url"]).scheme, {"http", "https"})
                self.assertTrue(page["note"].strip())

    def test_candidate_scores_follow_the_approved_formula_exactly(self) -> None:
        allowed_decisions = set(self.discovery["decisions"])
        for row in self.candidates:
            scores = {name: row[field] for name, field in SCORE_FIELDS.items()}
            for score in scores.values():
                self.assertIs(type(score), int)
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 5)
            expected = (
                2 * scores["information_density"]
                + 2 * scores["technical_specificity"]
                + scores["provenance_quality"]
                + scores["durability"]
                + 2 * scores["novel_coverage"]
                + scores["safety_context"]
                - 2 * scores["duplication_penalty"]
                - 2 * scores["commerce_noise_penalty"]
            )
            self.assertEqual(row["priority_score"], expected, row["candidate_id"])
            self.assertIn(row["decision"], allowed_decisions)

    def test_candidate_taxonomy_is_approved(self) -> None:
        approved = {
            row["category_id"]
            for row in self.corpus["knowledge_taxonomy"]["categories"]
        }
        for row in self.candidates:
            categories = row["supported_taxonomy_categories"]
            self.assertTrue(categories)
            self.assertEqual(len(categories), len(set(categories)))
            self.assertTrue(set(categories).issubset(approved), row["candidate_id"])

    def test_accepted_candidates_are_pending_in_extraction_queue(self) -> None:
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        legacy_urls = {
            row["url"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" not in row
        }
        accepted = [
            row for row in self.candidates if row["decision"].startswith("accept_")
        ]
        self.assertTrue(accepted)
        for candidate in accepted:
            self.assertNotIn(candidate["canonical_url"], legacy_urls)
            queued = queue_by_id[candidate["candidate_id"]]
            self.assertEqual(queued["status"], "pending")
            self.assertEqual(queued["url"], candidate["canonical_url"])
            self.assertEqual(queued["scope"], candidate["recommended_crawl_scope"])
            self.assertEqual(
                queued["discovery_candidate_id"], candidate["candidate_id"]
            )
            self.assertEqual(
                queued["discovery_priority_score"], candidate["priority_score"]
            )
            self.assertEqual(
                queued["supported_taxonomy_categories"],
                candidate["supported_taxonomy_categories"],
            )

    def test_deferred_and_rejected_candidates_are_not_queued(self) -> None:
        queued_candidate_ids = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        expected_accepted = {
            row["candidate_id"]
            for row in self.candidates
            if row["decision"].startswith("accept_")
        }
        self.assertEqual(queued_candidate_ids, expected_accepted)

    def test_batch_002_reuse_and_cross_domain_limits_are_enforced(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        for candidate_id in {
            "vvolfy_aerial_rig_manual",
            "performance_real_kinbaku_ritual",
        }:
            row = by_id[candidate_id]
            self.assertEqual(row["decision"], "reject")
            self.assertIn("ai-train=no", row["access_notes"])

        for candidate_id in {
            "simply_circus_rigging_essentials",
            "simply_circus_rescue",
        }:
            row = by_id[candidate_id]
            self.assertEqual(row["decision"], "defer")
            self.assertIn("clearance", row["recommended_crawl_scope"])

        for candidate_id in {
            "petzl_professional_technical_tips",
            "dmm_technical_knowledge",
            "arboricultural_association_safety_guides",
            "treeconsult_rigging_research",
            "fedec_safety_rigging_manual",
            "equity_fit_to_fly",
            "hilti_anchor_technical_guides",
            "animated_knots_search_rescue",
            "usbr_rope_access_guidelines",
        }:
            row = by_id[candidate_id]
            self.assertTrue(row["decision"].startswith("accept_"))
            scope = row["recommended_crawl_scope"].lower()
            self.assertTrue(
                "bondage" in scope
                or "human suspension" in scope
                or "human-suspension" in scope
            )

    def test_batch_003_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_003"
        }
        self.assertEqual(len(batch), 16)
        expected = {
            "accept_high_priority": {
                "europepmc_rope_neuropathy_study",
                "europepmc_icar_suspension_syndrome",
                "ncsf_consent_incident_toolkit",
                "usfs_rigging_for_trail_work",
            },
            "accept_targeted_scope": {
                "ontario_live_performance_safety",
                "enhance_uk_disability_bondage",
                "actsafe_performer_flying_rigging",
                "ashra_bondage_injury_prevalence_2025",
                "icar_rope_connection_recommendations",
                "alef_healing_shibari_study",
            },
            "defer": {
                "itra_rope_rescue_documents",
                "kanna_kagura_bakushi_biographies",
                "carleton_rope_bondage_thesis",
                "anzcor_suspension_first_aid_guideline",
            },
            "reject": {
                "durham_comparative_rope_thesis",
                "cleveland_clinic_suspension_syndrome",
            },
        }
        for decision, candidate_ids in expected.items():
            self.assertEqual(
                {
                    candidate_id
                    for candidate_id, row in batch.items()
                    if row["decision"] == decision
                },
                candidate_ids,
            )

    def test_batch_003_europe_pmc_routes_and_licenses_are_explicit(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        expected = {
            "europepmc_rope_neuropathy_study": ("PMC10294117", "CC BY 3.0"),
            "europepmc_icar_suspension_syndrome": ("PMC10710713", "CC BY 4.0"),
        }
        for candidate_id, (pmcid, license_name) in expected.items():
            row = by_id[candidate_id]
            self.assertEqual(row["decision"], "accept_high_priority")
            self.assertEqual(urlparse(row["canonical_url"]).netloc, "europepmc.org")
            api_pages = [
                page["url"]
                for page in row["representative_pages"]
                if "fullTextXML" in page["url"]
            ]
            self.assertEqual(
                api_pages,
                [
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/"
                    f"{pmcid}/fullTextXML"
                ],
            )
            self.assertIn(license_name, row["access_notes"])
            scope = row["recommended_crawl_scope"]
            self.assertIn("EMBL-EBI Europe PMC API only", scope)
            self.assertIn("US PMC", scope)

        self.assertIn(
            "Cureus",
            by_id["europepmc_rope_neuropathy_study"]["recommended_crawl_scope"],
        )
        self.assertIn(
            "Springer",
            by_id["europepmc_icar_suspension_syndrome"][
                "recommended_crawl_scope"
            ],
        )

    def test_europe_pmc_article_identifiers_are_bound(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        expected = {
            "europepmc_rope_neuropathy_study": {
                "pmcid": "PMC10294117",
                "doi": "10.7759/cureus.39588",
                "pmid": "37384078",
                "stale_pmids": {"37324199"},
            },
            "europepmc_icar_suspension_syndrome": {
                "pmcid": "PMC10710713",
                "doi": "10.1186/s13049-023-01164-z",
                "pmid": "38071341",
                "stale_pmids": {"38081341"},
            },
            "europepmc_bdsm_fatality_review": {
                "pmcid": "PMC8813685",
                "doi": "10.1007/s00414-021-02674-0",
                "pmid": "34383118",
                "stale_pmids": set(),
            },
        }
        for candidate_id, identifiers in expected.items():
            row = by_id[candidate_id]
            self.assertEqual(
                row["canonical_url"],
                "https://europepmc.org/article/MED/" + identifiers["pmid"],
            )
            serialized = json.dumps(row, sort_keys=True)
            for identifier in {
                identifiers["pmcid"],
                identifiers["doi"],
                identifiers["pmid"],
            }:
                self.assertIn(identifier, serialized)
            for stale_pmid in identifiers["stale_pmids"]:
                self.assertNotIn(stale_pmid, serialized)

            queued = queue_by_id[candidate_id]
            self.assertEqual(queued["url"], row["canonical_url"])
            for identifier in {
                identifiers["pmcid"],
                identifiers["doi"],
                identifiers["pmid"],
            }:
                self.assertIn(identifier, queued["scope"])
            for stale_pmid in identifiers["stale_pmids"]:
                self.assertNotIn(stale_pmid, json.dumps(queued, sort_keys=True))

        queued_med_urls = {
            row["url"]
            for row in self.corpus["queue"]
            if "europepmc.org/article/MED/" in row["url"]
        }
        self.assertEqual(
            queued_med_urls,
            {
                "https://europepmc.org/article/MED/37384078",
                "https://europepmc.org/article/MED/38071341",
                "https://europepmc.org/article/MED/34383118",
            },
        )

    def test_batch_003_restricted_rights_are_not_queued(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        for candidate_id in {
            "itra_rope_rescue_documents",
            "kanna_kagura_bakushi_biographies",
            "carleton_rope_bondage_thesis",
            "anzcor_suspension_first_aid_guideline",
        }:
            row = by_id[candidate_id]
            self.assertEqual(row["decision"], "defer")
            self.assertIn("permission", row["recommended_crawl_scope"].lower())

        self.assertEqual(
            by_id["durham_comparative_rope_thesis"]["decision"], "reject"
        )
        self.assertIn(
            "robots",
            by_id["durham_comparative_rope_thesis"]["access_notes"].lower(),
        )
        self.assertEqual(
            by_id["cleveland_clinic_suspension_syndrome"]["decision"],
            "reject",
        )
        self.assertIn(
            "repackage",
            by_id["cleveland_clinic_suspension_syndrome"][
                "recommended_crawl_scope"
            ],
        )

    def test_batch_003_cross_domain_and_evidence_limits_are_queued(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        for candidate_id in {
            "europepmc_icar_suspension_syndrome",
            "ontario_live_performance_safety",
            "usfs_rigging_for_trail_work",
            "actsafe_performer_flying_rigging",
            "icar_rope_connection_recommendations",
        }:
            scope = by_id[candidate_id]["recommended_crawl_scope"].lower()
            self.assertTrue(
                "bondage" in scope
                or "human suspension" in scope
                or "human-suspension" in scope
            )

        healing_scope = by_id["alef_healing_shibari_study"][
            "recommended_crawl_scope"
        ].lower()
        self.assertIn("do not claim efficacy", healing_scope)
        abstract_scope = by_id["ashra_bondage_injury_prevalence_2025"][
            "recommended_crawl_scope"
        ].lower()
        self.assertIn("conference-abstract status", abstract_scope)
        self.assertIn("do not reproduce", abstract_scope)

        ontario = by_id["ontario_live_performance_safety"]
        ontario_pages = {page["url"] for page in ontario["representative_pages"]}
        self.assertIn(
            "https://www.ontario.ca/document/"
            "safety-guidelines-live-performance-industry/"
            "rigging-systems-and-fall-arrest",
            ontario_pages,
        )
        self.assertNotIn(
            "https://www.ontario.ca/document/"
            "safety-guidelines-live-performance-industry/rigging",
            ontario_pages,
        )
        self.assertIn("404", ontario["access_notes"])

    def test_batch_004_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_004"
        }
        self.assertEqual(len(batch), 12)
        expected = {
            "accept_high_priority": {
                "heartland_kinbaku_public_guides",
            },
            "accept_targeted_scope": {
                "osada_ryu_primary_writings",
                "hajime_kinoko_official_profile",
                "go_arisue_official_profile",
                "devil_mask_studio_curriculum",
                "rope_study_progression",
            },
            "defer": {
                "usfs_national_tree_climbing_guide",
                "ritsumeikan_nureki_biography_interviews",
                "antitled_early_kitan_club_bibliography",
                "fullcircle_bondage_beginner_handout",
                "reborn_ropes_technical_guides",
            },
            "reject": {
                "german_wikibooks_shibari_manual",
            },
        }
        for decision, candidate_ids in expected.items():
            self.assertEqual(
                {
                    candidate_id
                    for candidate_id, row in batch.items()
                    if row["decision"] == decision
                },
                candidate_ids,
            )

        accepted = {
            candidate_id
            for candidate_id, row in batch.items()
            if row["decision"].startswith("accept_")
        }
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        self.assertEqual(len(accepted), 6)
        self.assertTrue(accepted.issubset(queued))

    def test_batch_004_access_and_rights_blocks_are_explicit(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}

        usfs = by_id["usfs_national_tree_climbing_guide"]
        self.assertEqual(usfs["decision"], "defer")
        self.assertIn("universal", usfs["access_notes"].lower())
        self.assertIn("robots", usfs["access_notes"].lower())

        antitled = by_id["antitled_early_kitan_club_bibliography"]
        self.assertEqual(antitled["decision"], "defer")
        self.assertIn("*_pdf", antitled["recommended_crawl_scope"])
        self.assertIn("permission", antitled["recommended_crawl_scope"].lower())

        ritsumeikan = by_id["ritsumeikan_nureki_biography_interviews"]
        self.assertEqual(ritsumeikan["decision"], "defer")
        self.assertIn(
            "permission", ritsumeikan["recommended_crawl_scope"].lower()
        )

        fullcircle = by_id["fullcircle_bondage_beginner_handout"]
        self.assertEqual(fullcircle["decision"], "defer")
        self.assertIn("404", fullcircle["access_notes"])
        self.assertIn("archive", fullcircle["recommended_crawl_scope"].lower())

    def test_batch_004_quality_gates_unsafe_or_disputed_sources(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}

        reborn = by_id["reborn_ropes_technical_guides"]
        self.assertEqual(reborn["decision"], "defer")
        reborn_scope = reborn["recommended_crawl_scope"].lower()
        for disputed_term in {"jbo", "somerville", "reef/granny"}:
            self.assertIn(disputed_term, reborn_scope)

        wikibook = by_id["german_wikibooks_shibari_manual"]
        self.assertEqual(wikibook["decision"], "reject")
        wikibook_scope = wikibook["recommended_crawl_scope"].lower()
        self.assertIn("90°c", wikibook_scope)
        self.assertIn("gun oil", wikibook_scope)

        heartland = by_id["heartland_kinbaku_public_guides"]
        self.assertEqual(heartland["decision"], "accept_high_priority")
        heartland_scope = heartland["recommended_crawl_scope"].lower()
        self.assertIn("adjudicate or quarantine", heartland_scope)
        self.assertIn("roughly-ten-bottom", heartland_scope)

    def test_batch_004_first_party_claim_types_are_preserved(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        expected_scope_markers = {
            "osada_ryu_primary_writings": "first-person",
            "hajime_kinoko_official_profile": "self-described",
            "go_arisue_official_profile": "source-labeled",
        }
        for candidate_id, marker in expected_scope_markers.items():
            row = by_id[candidate_id]
            self.assertEqual(row["decision"], "accept_targeted_scope")
            self.assertIn(marker, row["recommended_crawl_scope"].lower())

    def test_batch_005_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_005"
        }
        self.assertEqual(len(batch), 12)
        expected = {
            "accept_high_priority": {
                "usda_wood_handbook_structural_wood",
                "europepmc_bdsm_fatality_review",
            },
            "accept_targeted_scope": {
                "yukimura_ryu_official_archive",
                "shibaru_life_history_series",
                "willcat_tension_curriculum",
                "esinem_rope_maker_articles",
            },
            "defer": {
                "akechi_kanna_official_note",
                "ropeconnections_smart_way_manual",
                "rope_bondage_affective_embodiments",
                "tanaka_kinbaku_globalization_abstract",
            },
            "reject": {
                "ropebite_pittsburgh_guides",
                "remedial_ropes_wordpress_archive",
            },
        }
        for decision, candidate_ids in expected.items():
            self.assertEqual(
                {
                    candidate_id
                    for candidate_id, row in batch.items()
                    if row["decision"] == decision
                },
                candidate_ids,
            )

        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        accepted = expected["accept_high_priority"] | expected["accept_targeted_scope"]
        self.assertEqual(len(accepted), 6)
        self.assertTrue(accepted.issubset(queued))

    def test_batch_005_restricted_rights_are_not_queued(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}

        kanna = by_id["akechi_kanna_official_note"]
        self.assertEqual(kanna["decision"], "defer")
        kanna_access = kanna["access_notes"].lower()
        for marker in {"reproduction", "modification", "adaptation"}:
            self.assertIn(marker, kanna_access)
        self.assertIn("written permission", kanna["recommended_crawl_scope"].lower())

        manual = by_id["ropeconnections_smart_way_manual"]
        self.assertEqual(manual["decision"], "defer")
        self.assertIn("all rights reserved", manual["access_notes"].lower())
        self.assertIn("retrieval-system", manual["recommended_crawl_scope"].lower())
        self.assertIn("written permission", manual["recommended_crawl_scope"].lower())

        ethnography = by_id["rope_bondage_affective_embodiments"]
        self.assertEqual(ethnography["decision"], "defer")
        self.assertIn("CC BY-NC-ND 4.0", ethnography["access_notes"])
        self.assertIn("permission", ethnography["recommended_crawl_scope"].lower())

        tanaka = by_id["tanaka_kinbaku_globalization_abstract"]
        self.assertEqual(tanaka["decision"], "defer")
        self.assertIn("Abstract License Flag", tanaka["access_notes"])
        self.assertIn("*_pdf", tanaka["recommended_crawl_scope"])

    def test_batch_005_rejects_unsafe_or_empty_sources(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}

        ropebite = by_id["ropebite_pittsburgh_guides"]
        self.assertEqual(ropebite["decision"], "reject")
        ropebite_scope = ropebite["recommended_crawl_scope"].lower()
        for marker in {
            "ibuprofen",
            "vitamin b12",
            "boil rope",
            "gas flame",
            "oven",
        }:
            self.assertIn(marker, ropebite_scope)

        remedial = by_id["remedial_ropes_wordpress_archive"]
        self.assertEqual(remedial["decision"], "reject")
        self.assertFalse(remedial["accessible"])
        self.assertIn("six", remedial["access_notes"].lower())
        self.assertIn("404", remedial["access_notes"])
        remedial_scope = remedial["recommended_crawl_scope"].lower()
        for marker in {"url titles", "search snippets", "archive", "mirror"}:
            self.assertIn(marker, remedial_scope)

    def test_batch_005_accepted_scopes_are_safely_bounded(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}

        wood = by_id["usda_wood_handbook_structural_wood"]
        wood_scope = wood["recommended_crawl_scope"].lower()
        self.assertIn("does not certify", wood_scope)
        self.assertIn("qualified structural engineer", wood_scope)
        self.assertIn("third-party", wood_scope)

        fatality = by_id["europepmc_bdsm_fatality_review"]
        fatality_scope = fatality["recommended_crawl_scope"].lower()
        self.assertIn("17-case publication-derived denominator", fatality_scope)
        self.assertIn("lack of a population denominator", fatality_scope)
        self.assertIn("sensational", fatality_scope)

        shibaru = by_id["shibaru_life_history_series"]
        shibaru_pages = {page["url"] for page in shibaru["representative_pages"]}
        self.assertEqual(
            shibaru_pages,
            {
                "https://shibaru.life/2015/09/history-of-shibari-1-hojojutsu/",
                "https://shibaru.life/2015/09/history-of-shibari-2-kabuki/",
                "https://shibaru.life/2015/09/history-of-shibari-3-ukiyoe/",
                "https://shibaru.life/2018/04/history-of-shibari-4-kitan-club/",
            },
        )
        shibaru_scope = shibaru["recommended_crawl_scope"].lower()
        self.assertIn("full translation", shibaru_scope)
        self.assertIn("kannuki", shibaru_scope)

        esinem_scope = by_id["esinem_rope_maker_articles"][
            "recommended_crawl_scope"
        ].lower()
        self.assertIn("rope lay", esinem_scope)
        self.assertIn("american death triangle", esinem_scope)
        self.assertIn("sales claims", esinem_scope)

    def test_batch_005_claim_types_and_pedagogy_limits_are_preserved(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}

        yukimura_scope = by_id["yukimura_ryu_official_archive"][
            "recommended_crawl_scope"
        ].lower()
        self.assertIn("first-person", yukimura_scope)
        self.assertIn("claim type", yukimura_scope)
        self.assertIn("2006/2007 contradiction", yukimura_scope)

        willcat_scope = by_id["willcat_tension_curriculum"][
            "recommended_crawl_scope"
        ].lower()
        self.assertIn("under-construction", willcat_scope)
        self.assertIn("absolute safety", willcat_scope)
        self.assertIn("validated instructions", willcat_scope)

    def test_twisted_windows_second_audit_is_evidence_gated(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        row = by_id["twisted_windows"]
        self.assertEqual(row["decision"], "accept_targeted_scope")
        self.assertEqual(row["priority_score"], 29)
        self.assertIn("numbering may be out of order", row["access_notes"])
        scope = row["recommended_crawl_scope"].lower()
        for marker in {
            "claim-to-citation matrix",
            "two-finger rule",
            "capillary-refill",
            "harness-hang",
            "ibuprofen",
            "vitamin b12",
            "qualified clinicians",
        }:
            self.assertIn(marker, scope)

        queued = next(
            item
            for item in self.corpus["queue"]
            if item["resource_id"] == "twisted_windows"
        )
        self.assertEqual(queued["discovery_priority_score"], 29)
        self.assertEqual(queued["scope"], row["recommended_crawl_scope"])

    def test_batch_006_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_006"
        }
        self.assertEqual(len(batch), 11)
        expected = {
            "accept_targeted_scope": {
                "voxbody_core_curriculum",
                "temple_nyc_core_curriculum",
                "edo_tokyo_seiu_ito_museum",
            },
            "defer": {
                "house_cordee_bottoming_advice",
                "berkeley_binding_practice_thesis",
                "edinburgh_holding_rope_thesis",
                "esta_performer_flying_rigging_point_standards",
                "uan_shibari_ergonomics_thesis",
            },
            "reject": {
                "lb_shibari_dojo_class_resources",
                "clinical_neurophysiology_repeated_radial_neuropathy",
                "all_tied_up_rope_type_report",
            },
        }
        for decision, candidate_ids in expected.items():
            self.assertEqual(
                {
                    candidate_id
                    for candidate_id, row in batch.items()
                    if row["decision"] == decision
                },
                candidate_ids,
            )

        queued_from_batch = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if row.get("discovery_candidate_id") in batch
        }
        self.assertEqual(queued_from_batch, expected["accept_targeted_scope"])

    def test_batch_006_rights_and_access_blocks_are_explicit(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}

        house = by_id["house_cordee_bottoming_advice"]
        self.assertEqual(house["decision"], "defer")
        self.assertIn("not to copy", house["access_notes"].lower())
        for marker in {"permission", "peter martin", "mya/fox"}:
            self.assertIn(marker, house["recommended_crawl_scope"].lower())

        esta = by_id["esta_performer_flying_rigging_point_standards"]
        self.assertEqual(esta["decision"], "defer")
        esta_text = (
            esta["access_notes"] + " " + esta["recommended_crawl_scope"]
        ).lower()
        for marker in {
            "e1.43-2025",
            "e1.56-2026",
            "single-computer",
            "merge",
            "adapt",
            "translate",
            "network",
        }:
            self.assertIn(marker, esta_text)
        self.assertFalse(esta["accessible"])

        edinburgh = by_id["edinburgh_holding_rope_thesis"]
        self.assertEqual(edinburgh["decision"], "defer")
        self.assertFalse(edinburgh["accessible"])
        self.assertIn("GPTBot", edinburgh["access_notes"])
        self.assertIn("ChatGPT-User", edinburgh["access_notes"])
        edinburgh_scope = edinburgh["recommended_crawl_scope"].lower()
        self.assertIn("do not access", edinburgh_scope)
        self.assertIn("download", edinburgh_scope)

        uan = by_id["uan_shibari_ergonomics_thesis"]
        self.assertEqual(uan["decision"], "defer")
        self.assertIn("CC BY-NC-ND 4.0", uan["access_notes"])
        uan_scope = uan["recommended_crawl_scope"].lower()
        self.assertIn("do not copy, translate, transform", uan_scope)
        self.assertIn("written permission", uan_scope)

        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        for blocked in [house, esta, edinburgh, uan]:
            self.assertNotIn(blocked["candidate_id"], queued)

    def test_batch_006_rejects_hazardous_duplicate_or_empty_sources(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}

        dojo = by_id["lb_shibari_dojo_class_resources"]
        self.assertEqual(dojo["decision"], "reject")
        dojo_scope = dojo["recommended_crawl_scope"].lower()
        for marker in {
            "breath-control-with-rope",
            "neck-rope",
            "pre-scene stretching",
            "timed endurance",
            "bodyweight",
        }:
            self.assertIn(marker, dojo_scope)

        clinical = by_id[
            "clinical_neurophysiology_repeated_radial_neuropathy"
        ]
        self.assertEqual(clinical["decision"], "reject")
        clinical_text = (
            clinical["access_notes"] + " " + clinical["recommended_crawl_scope"]
        )
        for marker in {
            "10.1016/j.clinph.2019.04.557",
            "PMC10294117",
            "10.7759/cureus.39588",
            "37384078",
            "95.3",
            "77.3",
        }:
            self.assertIn(marker, clinical_text)
        clinical_scope = clinical["recommended_crawl_scope"].lower()
        self.assertIn("single-patient", clinical_scope)
        self.assertIn("prevalence", clinical_scope)

        report = by_id["all_tied_up_rope_type_report"]
        self.assertEqual(report["decision"], "reject")
        report_text = (
            report["access_notes"] + " " + report["recommended_crawl_scope"]
            + " "
            + report["decision_reason"]
            + " "
            + " ".join(page["note"] for page in report["representative_pages"])
        ).lower()
        self.assertIn("quiz-result", report_text)
        self.assertIn("lead magnet", report_text)
        self.assertIn("no substantive technical rope knowledge", report_text)

    def test_batch_006_accepted_scopes_are_safely_bounded(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}

        vox = by_id["voxbody_core_curriculum"]
        self.assertEqual(vox["decision"], "accept_targeted_scope")
        vox_scope = vox["recommended_crawl_scope"].lower()
        for marker in {
            "fundamentals, progressions, ascent, and onward",
            "14-month",
            "both-partners",
            "non-ryu",
            "certify suspension competence",
        }:
            self.assertIn(marker, vox_scope)

        temple = by_id["temple_nyc_core_curriculum"]
        self.assertEqual(temple["decision"], "accept_targeted_scope")
        temple_scope = temple["recommended_crawl_scope"].lower()
        for marker in {"level 0-4", "uplines", "lab/study-group", "certify"}:
            self.assertIn(marker, temple_scope)

        museum = by_id["edo_tokyo_seiu_ito_museum"]
        self.assertEqual(museum["decision"], "accept_targeted_scope")
        museum_text = (
            museum["access_notes"] + " " + museum["recommended_crawl_scope"]
        ).lower()
        for marker in {"1882", "1961", "age 78", "seme-e"}:
            self.assertIn(marker, museum_text)
        self.assertIn("do not label ito the father", museum_text)

    def test_report_covers_each_review_batch_and_decision(self) -> None:
        for batch_id in {row["review_batch"] for row in self.candidates}:
            self.assertIn(batch_id, self.report)
        for decision in set(self.discovery["decisions"]):
            self.assertIn(decision, self.report)
        self.assertIn("Latest batch: `discovery_batch_006`", self.report)


if __name__ == "__main__":
    unittest.main()
