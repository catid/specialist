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
CORPUS_REGISTRY = (
    ROOT.parent
    / "data/site_corpora/registry/site_corpus_registry_v1.json"
)


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
        cls.registry = json.loads(CORPUS_REGISTRY.read_text(encoding="utf-8"))

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

    def test_continuous_source_scout_is_active_and_identity_aware(self) -> None:
        self.assertEqual(self.discovery["status"], "active_continuous_worker")
        contract = self.discovery["discovery_contract"]
        for field in {
            "deduplicate_by_normalized_title_doi_pmid_pmcid_document_and_source_identity",
            "public_access_is_not_reuse_permission",
            "record_article_or_page_level_rights_basis",
            "component_rights_audit_required_before_extraction",
            "metadata_only_when_body_rights_or_access_are_unclear",
            "preserve_source_domain_and_block_unsupported_transfer",
        }:
            self.assertTrue(contract[field], field)

        cadence = self.discovery["cadence"]
        self.assertEqual(cadence["worker_role"], "rights_aware_source_scout")
        self.assertTrue(cadence["keep_discovery_worker_running_when_a_subworker_slot_is_available"])
        self.assertTrue(cadence["reassign_worker_immediately_after_each_bounded_batch"])
        self.assertTrue(cadence["rotate_search_toward_taxonomy_and_evidence_gaps"])
        self.assertTrue(cadence["commit_bounded_reviewed_batches"])
        self.assertTrue(cadence["never_start_extraction_without_a_distinct_dedicated_site_worker"])

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

    def test_accepted_candidates_have_pending_or_registered_extraction(self) -> None:
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        registry_by_id = {
            row["resource_id"]: row for row in self.registry["artifacts"]
        }
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
            self.assertIn(queued["status"], {"pending", "complete"})
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
            if queued["status"] == "complete":
                artifact = registry_by_id[candidate["candidate_id"]]
                self.assertEqual(
                    queued["registry_artifact_id"], artifact["artifact_id"]
                )

    def test_registered_discovery_candidates_are_not_left_pending(self) -> None:
        registry_by_id = {
            row["resource_id"]: row for row in self.registry["artifacts"]
        }
        for queued in self.corpus["queue"]:
            if "discovery_candidate_id" not in queued:
                continue
            artifact = registry_by_id.get(queued["resource_id"])
            if artifact is None:
                self.assertEqual(queued["status"], "pending")
                self.assertNotIn("registry_artifact_id", queued)
            else:
                self.assertEqual(queued["status"], "complete")
                self.assertEqual(
                    queued["registry_artifact_id"], artifact["artifact_id"]
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
                "alef_healing_shibari_study",
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
            "europepmc_entrapment_neuropathy_review": {
                "pmcid": "PMC7382548",
                "doi": "10.1097/PR9.0000000000000829",
                "pmid": "32766466",
                "stale_pmids": set(),
            },
            "openmind_tangled_physics_knots_2024": {
                "pmcid": "PMC11495958",
                "doi": "10.1162/opmi_a_00159",
                "pmid": "39439589",
                "stale_pmids": set(),
            },
            "springer_bdsm_consent_norms_2024": {
                "pmcid": "PMC10844416",
                "doi": "10.1007/s10508-023-02741-0",
                "pmid": "38017253",
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
                "https://europepmc.org/article/MED/32766466",
                "https://europepmc.org/article/MED/39439589",
                "https://europepmc.org/article/MED/38017253",
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

        healing = by_id["alef_healing_shibari_study"]
        self.assertEqual(healing["decision"], "reject")
        healing_text = json.dumps(healing, sort_keys=True).lower()
        for marker in {
            "some friends",
            "felt-healing interview prompt",
            "all-caucasian",
            "temporarily losing consciousness",
            "do not copy, paraphrase, transform",
            "rejected_unsafe_therapeutic_overclaim_and_positive_selection",
        }:
            self.assertIn(marker, healing_text)
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
                "usfs_national_tree_climbing_guide",
            },
            "accept_targeted_scope": {
                "osada_ryu_primary_writings",
                "hajime_kinoko_official_profile",
                "go_arisue_official_profile",
                "devil_mask_studio_curriculum",
                "rope_study_progression",
            },
            "defer": {
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
        self.assertEqual(len(accepted), 7)
        self.assertTrue(accepted.issubset(queued))

    def test_batch_004_access_and_rights_blocks_are_explicit(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}

        usfs = by_id["usfs_national_tree_climbing_guide"]
        self.assertEqual(usfs["decision"], "accept_high_priority")
        self.assertTrue(usfs["accessible"])
        self.assertEqual(
            usfs["canonical_url"],
            "https://www.govinfo.gov/app/details/GOVPUB-A13-PURL-gpo215987",
        )
        usfs_text = (
            usfs["access_notes"] + " " + usfs["recommended_crawl_scope"]
        ).lower()
        for marker in {
            "govinfo",
            "public domain",
            "body-weight anchor tests",
            "human suspension",
            "bondage",
        }:
            self.assertIn(marker, usfs_text)
        self.assertEqual(usfs["training_use"], "direct_training_bounded")
        self.assertIn("17 u.s.c.", usfs["rights_basis"].lower())

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

    def test_batch_007_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_007"
        }
        self.assertEqual(len(batch), 11)
        expected = {
            "accept_high_priority": {
                "usfs_region5_hazard_tree_2022",
                "europepmc_entrapment_neuropathy_review",
            },
            "accept_targeted_scope": {
                "ndl_rope_history_authority_data",
                "nps_new_river_climbing_guide_curriculum",
            },
            "defer": {
                "army_tc_21_24_2025_rappelling",
                "live_performance_australia_performer_hazards_2024",
                "sedici_shibari_stage_paper",
                "ritsumeikan_kitan_murakami_paper",
            },
            "reject": {
                "openstax_anatomy_physiology",
                "ropewiki_riggings",
                "army_tm_3_34_86_rigging",
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
        self.assertEqual(
            queued_from_batch,
            expected["accept_high_priority"] | expected["accept_targeted_scope"],
        )

    def test_batch_007_rights_and_access_blocks_are_explicit(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}

        openstax = by_id["openstax_anatomy_physiology"]
        self.assertEqual(openstax["decision"], "reject")
        self.assertFalse(openstax["accessible"])
        openstax_text = (
            openstax["access_notes"] + " " + openstax["recommended_crawl_scope"]
        ).lower()
        for marker in {
            "large language models",
            "gptbot",
            "/books/",
            "mirror",
            "permission",
        }:
            self.assertIn(marker, openstax_text)

        lpa = by_id["live_performance_australia_performer_hazards_2024"]
        self.assertEqual(lpa["decision"], "defer")
        lpa_text = (lpa["access_notes"] + " " + lpa["recommended_crawl_scope"]).lower()
        for marker in {
            "personal or internal corporate",
            "commercial gain",
            "written permission",
            "five-minute",
        }:
            self.assertIn(marker, lpa_text)

        sedici = by_id["sedici_shibari_stage_paper"]
        self.assertEqual(sedici["decision"], "defer")
        sedici_text = (
            sedici["access_notes"] + " " + sedici["recommended_crawl_scope"]
        ).lower()
        for marker in {"cc by-nc-sa 4.0", "crawl-delay 8", "written commercial"}:
            self.assertIn(marker, sedici_text)

        rits = by_id["ritsumeikan_kitan_murakami_paper"]
        self.assertEqual(rits["decision"], "defer")
        rits_text = (rits["access_notes"] + " " + rits["recommended_crawl_scope"]).lower()
        for marker in {"no reuse license", "30-second", "api", "oai", "permission"}:
            self.assertIn(marker, rits_text)

        for candidate_id in {
            "army_tc_21_24_2025_rappelling",
            "army_tm_3_34_86_rigging",
        }:
            army = by_id[candidate_id]
            self.assertFalse(army["accessible"])
            self.assertIn("mirror", army["recommended_crawl_scope"].lower())

    def test_batch_007_accepted_scopes_preserve_nontransfer_limits(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}

        tree = by_id["usfs_region5_hazard_tree_2022"]
        self.assertEqual(tree["decision"], "accept_high_priority")
        tree_scope = tree["recommended_crawl_scope"].lower()
        for marker in {
            "no inspection predicts all failures",
            "certified arborist",
            "root lifting",
            "uncompensated lean",
            "tree health establishes anchor capacity",
            "bondage setup",
        }:
            self.assertIn(marker, tree_scope)

        nerve = by_id["europepmc_entrapment_neuropathy_review"]
        self.assertEqual(nerve["decision"], "accept_high_priority")
        nerve_text = (
            nerve["access_notes"] + " " + nerve["recommended_crawl_scope"]
        ).lower()
        for marker in {
            "pmc7382548",
            "32766466",
            "10.1097/pr9.0000000000000829",
            "cc by 4.0",
            "preclinical models",
            "safe rope pressure",
            "qualified medical evaluation",
        }:
            self.assertIn(marker, nerve_text)

        ndl = by_id["ndl_rope_history_authority_data"]
        self.assertEqual(ndl["decision"], "accept_targeted_scope")
        ndl_text = (ndl["access_notes"] + " " + ndl["recommended_crawl_scope"]).lower()
        for marker in {
            "00023065",
            "00731050",
            "034485143",
            "web ndl authorities",
            "personal-information",
            "teaching relationships",
        }:
            self.assertIn(marker, ndl_text)

        nps = by_id["nps_new_river_climbing_guide_curriculum"]
        self.assertEqual(nps["decision"], "accept_targeted_scope")
        nps_scope = nps["recommended_crawl_scope"].lower()
        for marker in {
            "24-hour minimum plus proficiency exam",
            "redundancy and load distribution",
            "natural-versus-artificial anchors",
            "do not claim the page is current amga curriculum",
            "human suspension",
        }:
            self.assertIn(marker, nps_scope)

    def test_batch_007_rejects_obsolete_or_unsafe_domain_transfer(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}

        ropewiki = by_id["ropewiki_riggings"]
        self.assertEqual(ropewiki["decision"], "reject")
        ropewiki_text = (
            ropewiki["access_notes"] + " " + ropewiki["recommended_crawl_scope"]
        ).lower()
        for marker in {
            "cc by-nc-sa",
            "fiddlestick",
            "retrievable",
            "canyoneering",
            "human suspension",
            "crawl-delay",
        }:
            self.assertIn(marker, ropewiki_text)

        manual = by_id["army_tm_3_34_86_rigging"]
        self.assertEqual(manual["decision"], "reject")
        manual_text = (
            manual["access_notes"] + " " + manual["recommended_crawl_scope"]
        ).lower()
        for marker in {
            "identical to superseded fm 5-125",
            "no doctrine changes",
            "1995",
            "improvised lifting structures",
            "human suspension",
        }:
            self.assertIn(marker, manual_text)

    def test_batch_008_decisions_and_training_use_are_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_008"
        }
        self.assertEqual(len(batch), 12)
        expected = {
            "accept_high_priority": {
                "hse_treework_lifting_and_climbing",
            },
            "accept_targeted_scope": {
                "gutenberg_verrill_knots_splices",
                "gutenberg_dana_pearl_ropes_tackle",
                "nist_cordage_fiber_heating_8308",
            },
            "defer": {
                "niosh_suspension_scaffold_failures",
                "saail_autism_kink_toolkit",
                "cordage_institute_publications",
                "samson_rope_technical_documents",
                "wykd_practitioner_archive",
                "kink_clinical_guidelines_2023",
                "ropewalk_story_of_rope",
            },
            "reject": {
                "louis_kordexe_rope_articles",
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

        for row in batch.values():
            self.assertTrue(row["rights_basis"].strip())
            if row["decision"].startswith("accept_"):
                self.assertEqual(row["training_use"], "direct_training_bounded")
            elif row["decision"] == "defer":
                self.assertTrue(row["training_use"].startswith("reference_only"))
            else:
                self.assertEqual(row["training_use"], "rejected_no_use")

        accepted = expected["accept_high_priority"] | expected["accept_targeted_scope"]
        queue = {
            row["discovery_candidate_id"]: row
            for row in self.corpus["queue"]
            if row.get("discovery_candidate_id") in accepted
        }
        self.assertEqual(set(queue), accepted)
        for candidate_id, row in queue.items():
            self.assertEqual(row["training_use"], "direct_training_bounded")
            self.assertTrue(row["rights_basis"].strip(), candidate_id)

    def test_batch_008_rights_and_domain_transfer_limits_are_explicit(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}

        hse = by_id["hse_treework_lifting_and_climbing"]
        hse_text = (
            hse["access_notes"]
            + " "
            + hse["recommended_crawl_scope"]
            + " "
            + hse["rights_basis"]
        ).lower()
        for marker in {
            "open government licence v3.0",
            "eligible crown text",
            "separate lines and anchors",
            "rescue",
            "human suspension",
            "third-party",
        }:
            self.assertIn(marker, hse_text)

        for candidate_id in {
            "gutenberg_verrill_knots_splices",
            "gutenberg_dana_pearl_ropes_tackle",
        }:
            source = by_id[candidate_id]
            text = (
                source["access_notes"]
                + " "
                + source["recommended_crawl_scope"]
                + " "
                + source["rights_basis"]
            ).lower()
            for marker in {
                "public domain in the usa",
                "territorial warning",
                "human suspension",
                "historical",
            }:
                self.assertIn(marker, text)

        nist = by_id["nist_cordage_fiber_heating_8308"]
        nist_text = (
            nist["access_notes"] + " " + nist["recommended_crawl_scope"]
        ).lower()
        for marker in {
            "approved for public release",
            "abaca",
            "haitian sisal",
            "do not generalize results to jute",
            "bulk-bale",
        }:
            self.assertIn(marker, nist_text)

        for candidate_id in {
            "saail_autism_kink_toolkit",
            "cordage_institute_publications",
            "samson_rope_technical_documents",
            "wykd_practitioner_archive",
            "kink_clinical_guidelines_2023",
            "ropewalk_story_of_rope",
        }:
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertIn(
                "permission",
                source["recommended_crawl_scope"].lower(),
                candidate_id,
            )

        niosh = by_id["niosh_suspension_scaffold_failures"]
        self.assertEqual(niosh["training_use"], "reference_only_policy_conflict")
        niosh_text = (
            niosh["access_notes"] + " " + niosh["recommended_crawl_scope"]
        ).lower()
        for marker in {
            "not to make substantive changes",
            "1992",
            "obsolete",
            "human suspension",
        }:
            self.assertIn(marker, niosh_text)

    def test_batch_008_rejects_unsafe_rope_conditioning_content(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        source = by_id["louis_kordexe_rope_articles"]
        self.assertEqual(source["decision"], "reject")
        self.assertEqual(source["training_use"], "rejected_no_use")
        text = (
            source["access_notes"] + " " + source["recommended_crawl_scope"]
        ).lower()
        for marker in {
            "boiling",
            "100 friction pulls",
            "torch",
            "oil/wax",
            "bacterial",
            "mold",
        }:
            self.assertIn(marker, text)

    def test_batch_009_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_009"
        }
        self.assertEqual(len(batch), 12)
        expected = {
            "accept_high_priority": {
                "kink_education_code_of_conduct",
            },
            "accept_targeted_scope": {
                "ninds_peripheral_neuropathy_2026",
                "nist_rope_yarn_bending_fatigue_t300",
            },
            "defer": {
                "nhs_neuropathy_compartment_warning_signs",
                "safe_work_australia_industrial_rope_access_2022",
                "karada_house_consent_accessibility_policies",
                "international_guild_knot_tyers_public_library",
                "consent_academy_public_resources",
                "transport_nsw_fibre_rope_guide",
                "arxiv_frictional_sliding_strength_2604",
            },
            "reject": {
                "studio_allegory_current_primary_metadata",
                "hitchin_bitches_skillshare_archive",
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

    def test_batch_009_accepted_sources_have_explicit_reuse_bounds(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        accepted = {
            "kink_education_code_of_conduct",
            "ninds_peripheral_neuropathy_2026",
            "nist_rope_yarn_bending_fatigue_t300",
        }
        for candidate_id in accepted:
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertTrue(source["decision"].startswith("accept_"))
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])

        kecc = by_id["kink_education_code_of_conduct"]
        kecc_text = json.dumps(kecc, sort_keys=True).lower()
        for marker in {
            "cc by-sa 4.0",
            "sharealike",
            "2019/version-1",
            "demo-volunteer",
            "do not present kecc as law, certification",
        }:
            self.assertIn(marker, kecc_text)

        ninds = by_id["ninds_peripheral_neuropathy_2026"]
        ninds_text = json.dumps(ninds, sort_keys=True).lower()
        for marker in {
            "march 13, 2026",
            "motor-versus-autonomic",
            "prolonged pressure",
            "do not derive a safe rope placement",
            "all treatment and medication content",
        }:
            self.assertIn(marker, ninds_text)

        nist = by_id["nist_rope_yarn_bending_fatigue_t300"]
        nist_text = json.dumps(nist, sort_keys=True).lower()
        for marker in {
            "1925",
            "10.6028/nbst.8322",
            "repeated bending",
            "no. 18 manila yarn",
            "do not generalize manila rope-yarn results",
        }:
            self.assertIn(marker, nist_text)

    def test_batch_009_restricted_and_stale_sources_are_not_queued(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        expected_markers = {
            "safe_work_australia_industrial_rope_access_2022": {
                "cc by 4.0",
                "non-commercial",
                "rights conflict",
            },
            "transport_nsw_fibre_rope_guide": {
                "personal or non-commercial",
                "overrides",
                "third-party",
            },
            "arxiv_frictional_sliding_strength_2604": {
                "cc by-nc-nd 4.0",
                "noderivatives",
                "do not copy",
            },
            "international_guild_knot_tyers_public_library": {
                "all rights reserved",
                "verification",
                "permission",
            },
            "karada_house_consent_accessibility_policies": {
                "no open reuse license",
                "permission",
            },
            "consent_academy_public_resources": {
                "patreon",
                "no open reuse license",
                "permission",
            },
        }
        for candidate_id, markers in expected_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in markers:
                self.assertIn(marker, text, candidate_id)

        nhs = by_id["nhs_neuropathy_compartment_warning_signs"]
        self.assertEqual(nhs["decision"], "defer")
        self.assertNotIn(nhs["candidate_id"], queued)
        nhs_text = json.dumps(nhs, sort_keys=True).lower()
        for marker in {
            "october 10, 2025",
            "february 9, 2026",
            "overdue",
            "never claim rope bondage causes compartment syndrome",
        }:
            self.assertIn(marker, nhs_text)

        for candidate_id in {
            "studio_allegory_current_primary_metadata",
            "hitchin_bitches_skillshare_archive",
        }:
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "reject")
            self.assertEqual(source["training_use"], "rejected_low_yield")
            self.assertNotIn(candidate_id, queued)

    def test_batch_010_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_010"
        }
        self.assertEqual(len(batch), 12)
        expected = {
            "accept_high_priority": {
                "mdpi_tree_climbing_friction_hitch_2021",
            },
            "accept_targeted_scope": {
                "nist_aramid_rope_sling_fatigue_1976",
                "openrn_nursing_skills_targeted_assessment",
                "uk_mca_rope_access_lifting_2024_2026",
                "uk_mca_tensioned_rope_hazards_2024",
            },
            "defer": {
                "bowline_self_locking_mechanics_2025",
                "bdsm_marks_injury_exploration_2023",
                "kink_injury_healthcare_utilization_2021",
                "hse_hsg221_offshore_lifting_2007",
                "worksafe_nz_industrial_rope_access_2012",
            },
            "reject": {
                "cambridge_geninka_slavery_tokugawa_2023",
                "mdpi_bdsm_complexity_consent_2025",
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

    def test_batch_010_accepted_sources_preserve_evidence_bounds(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "mdpi_tree_climbing_friction_hitch_2021": {
                "36 total tests",
                "nine replicates per treatment",
                "dynamic and cyclic",
                "short quasi-static bench stroke",
                "cc by 4.0",
            },
            "nist_aramid_rope_sling_fatigue_1976": {
                "26 prototype aramid sling-leg specimens",
                "inadequate manufacturer end fittings",
                "incompletely substantiated",
                "helicopter external-cargo sling",
            },
            "openrn_nursing_skills_targeted_assessment": {
                "chapters 6, 9, 13, and 14",
                "sensory-versus-motor",
                "not rope-specific guidance or medical advice",
                "cc by 4.0",
            },
            "uk_mca_rope_access_lifting_2024_2026": {
                "january 7, 2026",
                "mounts, fixings, attachments",
                "load path",
                "open government licence v3.0",
            },
            "uk_mca_tensioned_rope_hazards_2024": {
                "snap-back",
                "snagged-line sudden release",
                "massive-vessel-load",
                "open government licence v3.0",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertTrue(source["decision"].startswith("accept_"))
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_010_restricted_sources_and_rejections_are_not_queued(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        deferred_markers = {
            "bowline_self_locking_mechanics_2025": {
                "cc by-nc-sa 4.0",
                "nearly inextensible elastic rod",
                "licensing workaround",
            },
            "bdsm_marks_injury_exploration_2023": {
                "cc by-nc 4.0",
                "n=513",
                "population prevalence",
            },
            "kink_injury_healthcare_utilization_2021": {
                "cc by-nc-nd 4.0",
                "n=1,398",
                "noderivatives",
            },
            "hse_hsg221_offshore_lifting_2007": {
                "all rights reserved",
                "specific notice",
                "written permission",
            },
            "worksafe_nz_industrial_rope_access_2012": {
                "iraanz",
                "pre-2015-law",
                "adaptation permission",
            },
        }
        for candidate_id, required in deferred_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

        geninka = by_id["cambridge_geninka_slavery_tokugawa_2023"]
        self.assertEqual(geninka["decision"], "reject")
        self.assertEqual(geninka["training_use"], "rejected_out_of_scope")
        self.assertNotIn(geninka["candidate_id"], queued)
        self.assertIn("enslavement and bonded labor", geninka["decision_reason"])
        self.assertIn("never use the article to assert a shibari", geninka["recommended_crawl_scope"].lower())

        consent = by_id["mdpi_bdsm_complexity_consent_2025"]
        self.assertEqual(consent["decision"], "reject")
        self.assertEqual(consent["training_use"], "rejected_safety_and_low_transfer")
        self.assertNotIn(consent["candidate_id"], queued)
        consent_text = json.dumps(consent, sort_keys=True).lower()
        for marker in {
            "no empirical practice evidence",
            "capacity",
            "immediate withdrawal",
            "kecc",
        }:
            self.assertIn(marker, consent_text)

    def test_batch_011_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_011"
        }
        self.assertEqual(len(batch), 12)
        expected = {
            "accept_high_priority": {
                "acta_loop_knot_efficiency_experiments_2020",
            },
            "accept_targeted_scope": {
                "mdpi_knot_efficiency_statistics_2022",
                "nist_pleated_synthetic_rope_1986",
                "hse_indg367_fall_arrest_rope_inspection",
                "openmind_tangled_physics_knots_2024",
            },
            "defer": {
                "mdpi_static_anchor_knot_tests_2024",
                "hse_rr708_suspension_trauma_first_aid_2009",
                "wjem_suspension_trauma_systematic_review_2021",
            },
            "reject": {
                "mdpi_dynamic_lanyard_prototypes_2020",
                "hse_general_loler_guidance",
                "openreview_human_knot_tying_robotics_2025",
                "tokyo_weekender_shibari_history_2024",
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
        self.assertEqual(
            queued_from_batch,
            expected["accept_high_priority"] | expected["accept_targeted_scope"],
        )

    def test_batch_011_accepted_sources_preserve_evidence_bounds(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "acta_loop_knot_efficiency_experiments_2020": {
                "eight tested loop-knot families",
                "standard-load versus cross-load",
                "break-versus-untying",
                "cc by 4.0",
                "natural-fiber or bondage rope",
            },
            "mdpi_knot_efficiency_statistics_2022": {
                "single paired break",
                "ratio of small-sample means",
                "approximately 200 straight-rope and 80 knotted-rope",
                "failure to reject normality and proof of normality",
            },
            "nist_pleated_synthetic_rope_1986": {
                "20 specimens",
                "four untreated replicates per size",
                "triplicate temperature groups",
                "monotonic quasi-static",
                "mired-vehicle kinetic-recovery",
            },
            "hse_indg367_fall_arrest_rope_inspection": {
                "whole-length visual and tactile",
                "sufficiently independent and impartial",
                "single well-defined usable-life boundary",
                "open government licence v3.0",
            },
            "openmind_tangled_physics_knots_2024": {
                "pmcid pmc11495958",
                "doi 10.1162/opmi_a_00159",
                "five-experiment structure",
                "topology recognition and strength judgment",
                "resistance-to-undoing judgments",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertTrue(source["decision"].startswith("accept_"))
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_011_deferred_and_rejected_sources_are_safely_excluded(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        deferred_markers = {
            "mdpi_static_anchor_knot_tests_2024": {
                "safest-knot conclusion",
                "misinterpretation could cause fall accidents",
                "specialist safety adjudication",
            },
            "hse_rr708_suspension_trauma_first_aid_2009": {
                "http 404",
                "no cache, archive",
                "clinical currency",
            },
            "wjem_suspension_trauma_systematic_review_2021": {
                "copyright world journal of emergency medicine",
                "public full-text access alone",
                "qualified clinical review",
            },
        }
        for candidate_id, required in deferred_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

        rejection_markers = {
            "mdpi_dynamic_lanyard_prototypes_2020": {
                "16 total dynamic specimens",
                "energy absorber",
                "rejected_underpowered_high_consequence",
            },
            "hse_general_loler_guidance": {
                "duplicates already queued",
                "uk legal scope",
                "rejected_redundant_legal_scope",
            },
            "openreview_human_knot_tying_robotics_2025": {
                "30 human demonstrations",
                "rgb-d",
                "rejected_low_transfer_visual_robotics",
            },
            "tokyo_weekender_shibari_history_2024": {
                "tawara rice-bag tying",
                "father of kinbaku",
                "all rights reserved",
            },
        }
        for candidate_id, required in rejection_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "reject")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_012_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_012"
        }
        self.assertEqual(len(batch), 12)
        expected = {
            "accept_high_priority": {
                "noaa_synthetic_rope_deterioration_1990",
            },
            "accept_targeted_scope": {
                "nist_manila_rope_statistics_1947",
                "elsevier_hemp_rope_weakest_link_2025",
                "osha_anchor_planning_existing_structures",
                "scientific_reports_synthetic_rope_tensile_2023",
                "springer_suspension_syndrome_crossover_2019",
            },
            "defer": {
                "bmc_tourniquet_nerve_compression_review_2020",
                "elsevier_natural_filament_water_swelling_2025",
                "ndl_seiu_ito_digitized_works",
            },
            "reject": {
                "plos_mooring_rope_mechanics_2025",
                "osha_advanced_rigging_workbook",
                "osha_natural_synthetic_sling_guide",
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
        self.assertEqual(
            queued_from_batch,
            expected["accept_high_priority"] | expected["accept_targeted_scope"],
        )

    def test_batch_012_accepted_sources_preserve_evidence_bounds(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "noaa_synthetic_rope_deterioration_1990": {
                "all 41 pages",
                "internal fiber-to-fiber abrasion",
                "tensile creep and fatigue",
                "numerical service-life predictions",
                "public domain",
            },
            "nist_manila_rope_statistics_1947": {
                "863 observations",
                "more than 800 samples",
                "nonrandom sampling",
                "insufficient four-strand evidence",
                "10.6028/jres.039.039",
                "10.6028/jres.039.037 resolves to an unrelated uranium paper",
                "numerical strength or rating values",
            },
            "elsevier_hemp_rope_weakest_link_2025": {
                "one-rope and one-supplier",
                "first-strand-break endpoint",
                "independent-link premise",
                "omitted friction and rope hierarchy",
                "capacity calculators",
            },
            "osha_anchor_planning_existing_structures": {
                "paragraphs h(1) and h(2)",
                "qualified-person evaluation",
                "connector pull-through",
                "must not reduce system strength",
                "never certifies",
            },
            "scientific_reports_synthetic_rope_tensile_2023": {
                "220 conducted or planned versus 196 included results",
                "nominal-versus-solid-or-effective-diameter",
                "ann black-box limitation",
                "request-only data and code",
                "small-to-large-rope extrapolation",
            },
            "springer_suspension_syndrome_crossover_2019": {
                "20 healthy male nonprofessional climbers",
                "40 crossover tests",
                "published correction",
                "timing or proportions as a safety threshold",
                "do not transfer occupational or climbing-harness physiology",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertTrue(source["decision"].startswith("accept_"))
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            self.assertEqual(
                queued["discovery_priority_score"], source["priority_score"]
            )
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_012_exclusions_and_prior_rights_update_are_explicit(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        deferred_markers = {
            "bmc_tourniquet_nerve_compression_review_2020": {
                "commercial interests",
                "qualified independent clinical reviewer",
                "republished figure",
                "never treat a tourniquet cuff as equivalent to bondage rope",
            },
            "elsevier_natural_filament_water_swelling_2025": {
                "filaments rather than laid",
                "cc by-nc-nd 4.0",
                "noderivatives",
                "do not copy, paraphrase, transform",
            },
            "ndl_seiu_ito_digitized_works": {
                "public reading access does not by itself",
                "work by work",
                "japanese-language",
                "do not bulk mirror, ocr, translate",
            },
        }
        for candidate_id, required in deferred_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

        rejection_markers = {
            "plos_mooring_rope_mechanics_2025": {
                "do not report replicate counts",
                "all three authors are employees",
                "yarn-to-rope overgeneralization",
                "rejected_underreported_commercial_high_consequence",
            },
            "osha_advanced_rigging_workbook": {
                "308-page instructor workbook",
                "federal hosting does not make",
                "do not assume public domain",
                "rejected_mixed_rights_redundant_operational_noise",
            },
            "osha_natural_synthetic_sling_guide": {
                "not to use knots in place of splices",
                "personnel platforms",
                "rejected_redundant_high_consequence_industrial_scope",
            },
        }
        for candidate_id, required in rejection_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "reject")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

        friction = by_id["arxiv_frictional_sliding_strength_2604"]
        self.assertEqual(friction["review_batch"], "discovery_batch_009")
        self.assertEqual(friction["decision"], "defer")
        self.assertNotIn(friction["candidate_id"], queued)
        friction_text = json.dumps(friction, sort_keys=True).lower()
        for marker in {
            "10.1016/j.jmps.2026.106628",
            "cc by-nc 4.0",
            "cc by-nc-nd 4.0",
            "noderivatives",
            "neither route is copied",
        }:
            self.assertIn(marker, friction_text)

    def test_batch_013_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_013"
        }
        self.assertEqual(len(batch), 12)
        expected = {
            "accept_high_priority": {
                "noaa_eight_strand_rope_structural_model_1989",
            },
            "accept_targeted_scope": {
                "mdpi_jute_yarn_uncertainty_2017",
                "elsevier_rayon_rope_biodegradation_2025",
                "usfs_wood_handbook_fastening_ch8_2021",
                "osha_rds_anchor_evaluation_2017",
            },
            "defer": {
                "noaa_marine_rope_design_brief_1986",
                "elsevier_climbing_rope_degradation_2025",
                "pmc_nerve_microcirculation_review_2013",
                "canada_cci_textile_conservation",
                "elsevier_synthetic_rope_fatigue_methodology_2024",
            },
            "reject": {
                "cdc_wire_rope_terminations_1978",
                "deanexa_high_risk_rope_advice",
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
        self.assertEqual(
            queued_from_batch,
            expected["accept_high_priority"] | expected["accept_targeted_scope"],
        )

    def test_batch_013_accepted_sources_preserve_scale_and_system_bounds(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "noaa_eight_strand_rope_structural_model_1989": {
                "all 51 pages",
                "two limiting friction assumptions",
                "lateral-contraction-ratio sensitivity",
                "qualitative-only conclusion",
                "d791640336f80bfb98abe5863476bb366502ca26744b4a59fb4ee09422c3f9ade3c50e5c3471671cc4e0c98d913a0734b9652f1c718f63cd5c586ed78a6ed85e",
            },
            "mdpi_jute_yarn_uncertainty_2017": {
                "15 specimens",
                "fiber-versus-yarn comparison",
                "one yarn batch",
                "pmcid pmc5459092",
                "finished jute rope",
            },
            "elsevier_rayon_rope_biodegradation_2025": {
                "elementary-fiber versus yarn versus eight-yarn braided-rope",
                "unreported-coating caveat",
                "request-only data",
                "seawater protocol as a care recipe",
                "cc by 4.0",
            },
            "usfs_wood_handbook_fastening_ch8_2021": {
                "structural member, fastener, connection and complete load path",
                "along-grain versus across-grain",
                "qualified local professional",
                "diy ceiling or hardpoint recipe",
                "public domain",
            },
            "osha_rds_anchor_evaluation_2017": {
                "qualified-person determination",
                "known use and shock-load history",
                "deciding not to proceed is an option",
                "not current legal advice and not an inspection checklist",
                "never certifies",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertTrue(source["decision"].startswith("accept_"))
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_013_rights_and_contamination_quarantines_are_explicit(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        deferred_markers = {
            "noaa_marine_rope_design_brief_1986": {
                "returns http 404",
                "no cache, archive, mirror",
                "22c85311e08d83d5b44900dc9f0a3dbe4731cba23842e21a70ee3dc3341b3b00",
                "reference_only_missing_official_body",
            },
            "elsevier_climbing_rope_degradation_2025": {
                "cc by-nc-nd 4.0",
                "one dynamic 9.8 mm polyamide-6 kernmantle rope type",
                "visual-retirement rules",
                "qualified textile-rope engineering review",
            },
            "pmc_nerve_microcirculation_review_2013": {
                "cc by-nc-sa 3.0",
                "sharealike",
                "animal compression",
                "never treat",
            },
            "canada_cci_textile_conservation": {
                "noncommercial reproduction",
                "museum fabric handling",
                "commercial derivative and model-training permission",
                "cannot be converted into rope care",
            },
            "elsevier_synthetic_rope_fatigue_methodology_2024": {
                "cc by-nc-nd 4.0",
                "full-scale test-bench",
                "periodic bending-over-pulley-or-drum",
                "qualified textile-rope engineering review",
            },
        }
        for candidate_id, required in deferred_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

        rejection_markers = {
            "cdc_wire_rope_terminations_1978": {
                "non-peer-reviewed contractor",
                "metallic mining",
                "material-system mismatch",
                "rejected_out_of_material_high_consequence_numeric_scope",
            },
            "deanexa_high_risk_rope_advice": {
                "open-flame defuzzing",
                "replicable diy human-suspension recipe",
                "personal-experience",
                "rejected_unsupported_personal_high_risk_instruction",
            },
        }
        for candidate_id, required in rejection_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "reject")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_014_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_014"
        }
        self.assertEqual(len(batch), 11)
        expected = {
            "accept_high_priority": {
                "sage_aramid_three_strand_contact_forces_2025",
            },
            "accept_targeted_scope": {
                "phm_synthetic_fiber_rope_condition_monitoring_review_2017",
                "springer_bdsm_consent_norms_2024",
                "tandf_within_person_consent_variability_2021",
            },
            "defer": {
                "sage_yarn_abrasion_failure_mechanisms_2024",
                "elsevier_hemp_rope_hierarchy_statistics_2024",
                "arxiv_ancient_art_laying_rope_2010",
                "elsevier_bowline_self_locking_2025",
                "tandf_autistic_adults_bdsm_2024",
                "tandf_plant_natural_fiber_rope_treatments_2024",
            },
            "reject": {
                "deadheavy_rope_resources_directory",
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
        self.assertEqual(
            queued_from_batch,
            expected["accept_high_priority"] | expected["accept_targeted_scope"],
        )

    def test_batch_014_accepted_sources_preserve_methods_and_limits(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "sage_aramid_three_strand_contact_forces_2025": {
                "teijin aramid bv",
                "pressure-film placement and calibration limits",
                "specific-material-and-construction limitation",
                "declared financial-support conflict",
            },
            "phm_synthetic_fiber_rope_condition_monitoring_review_2017": {
                "cc by 3.0 united states",
                "continuous-versus-discrete monitoring distinction",
                "permission-reproduced component",
                "2017 review date",
            },
            "springer_bdsm_consent_norms_2024": {
                "84 of 116",
                "indirect new-partner open-ended measure",
                "null injunctive",
                "sensitivity-analysis discrepancies",
            },
            "tandf_within_person_consent_variability_2021": {
                "28-day experience-sampling design",
                "1,189 analytic partnered events",
                "modest reliability",
                "not a validated rope policy",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertTrue(source["decision"].startswith("accept_"))
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_014_deferrals_and_site_contamination_are_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        deferred_markers = {
            "sage_yarn_abrasion_failure_mechanisms_2024": {
                "cloudflare challenge",
                "no mirror",
                "checksum",
                "reference_only_access_blocked_cc_by",
            },
            "elsevier_hemp_rope_hierarchy_statistics_2024": {
                "http 403",
                "unauthorized minimized metadata",
                "no mirror",
                "reference_only_access_blocked_cc_by",
            },
            "arxiv_ancient_art_laying_rope_2010": {
                "arxiv nonexclusive-distribution license",
                "commercial derivative",
                "reference_only_arxiv_distribution_license",
            },
            "elsevier_bowline_self_locking_2025": {
                "vps elastomeric rod with nitinol core",
                "cc by-nc 4.0",
                "rescue-harness statement",
            },
            "tandf_autistic_adults_bdsm_2024": {
                "six-adult",
                "noncommercial",
                "cloudflare challenge",
            },
            "tandf_plant_natural_fiber_rope_treatments_2024": {
                "boiling",
                "concrete-reinforcement purpose",
                "independent natural-textile engineering",
            },
        }
        for candidate_id, required in deferred_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

        deadheavy = by_id["deadheavy_rope_resources_directory"]
        self.assertEqual(deadheavy["decision"], "reject")
        self.assertNotIn(deadheavy["candidate_id"], queued)
        deadheavy_text = json.dumps(deadheavy, sort_keys=True).lower()
        for marker in {
            "url-trivia failure",
            "adapted code of conduct",
            "rejected_url_directory_not_factual_corpus",
        }:
            self.assertIn(marker, deadheavy_text)

        prior_site_markers = {
            "shibari_safety": {
                "neither a medical professional nor a certified instructor",
                "10:1 suspension guidance",
                "boiling and oil recipes",
            },
            "shibari_academy": {
                "waiting two hours",
                "nerve injuries occur instantly",
                "terms prohibit reproduction",
            },
            "shibari_news": {
                "1,094 post urls",
                "personal noncommercial extracts",
                "no editor, author, historian or safety reviewer",
            },
        }
        for candidate_id, required in prior_site_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "reject")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_015_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_015"
        }
        self.assertEqual(len(batch), 14)
        expected = {
            "accept_high_priority": {
                "maib_zarga_hmpe_inspection_failure_2017",
            },
            "accept_targeted_scope": {
                "springer_disability_sexual_rights_access_choice_pleasure_2024",
                "sage_intellectual_disability_consent_interviews_2024",
                "bsee_auxiliary_line_abrasion_failure_alert_2022",
                "mdpi_trans_msm_kink_negotiation_case_studies_2022",
                "nps_textile_fiber_aging_appendix_k_2002",
                "maib_throwbag_hidden_fused_joints_2019",
                "springer_sexual_strangulation_safety_perceptions_2025",
                "nps_maritime_rope_construction_history",
            },
            "defer": {
                "bsee_polyester_subrope_inspection_uncertainty_2007",
                "shibari_lounge_uk_risk_accessibility",
                "fema_structural_collapse_search_technician_2021",
                "usfs_crc_jute_kenaf_fiber_chemistry_2007",
                "uscg_cutter_seamanship_line_handling_2020",
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
        self.assertEqual(
            queued_from_batch,
            expected["accept_high_priority"] | expected["accept_targeted_scope"],
        )

    def test_batch_015_accepts_preserve_source_limits_and_transfer_blocks(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "springer_disability_sexual_rights_access_choice_pleasure_2024": {
                "theoretical",
                "jurisdiction-specific",
                "right to another person's sexual participation",
                "third-party",
            },
            "sage_intellectual_disability_consent_interviews_2024": {
                "22",
                "verbal",
                "participant quotation",
                "rope-specific claims",
            },
            "bsee_auxiliary_line_abrasion_failure_alert_2022": {
                "wire rope",
                "adjacent components",
                "line-of-fire",
                "body rope",
            },
            "mdpi_trans_msm_kink_negotiation_case_studies_2022": {
                "three",
                "participant perceptions",
                "causal",
                "hiv",
            },
            "maib_zarga_hmpe_inspection_failure_2017": {
                "load-bearing core",
                "axial-compression",
                "offshore-sector test methodology",
                "human suspension",
            },
            "nps_textile_fiber_aging_appendix_k_2002": {
                "museum",
                "museum-textile evidence",
                "treatment",
                "retirement",
            },
            "maib_throwbag_hidden_fused_joints_2019": {
                "thermally fused",
                "random",
                "traceability",
                "bondage rope",
            },
            "springer_sexual_strangulation_safety_perceptions_2025": {
                "single final open question",
                "risk-prompt",
                "coding-reliability",
                "breath-play",
            },
            "nps_maritime_rope_construction_history": {
                "yarn",
                "strand",
                "counter-twist",
                "historical",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertTrue(source["decision"].startswith("accept_"))
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_015_deferrals_are_rights_or_access_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        markers = {
            "bsee_polyester_subrope_inspection_uncertainty_2007": {
                "contractor",
                "permission",
                "public domain",
                "remaining-life",
            },
            "shibari_lounge_uk_risk_accessibility": {
                "license conflict",
                "noncommercial",
                "no-modification",
                "written clarification",
            },
            "fema_structural_collapse_search_technician_2021": {
                "robots",
                "disallow",
                "metadata",
                "role profile, not instruction",
            },
            "usfs_crc_jute_kenaf_fiber_chemistry_2007": {
                "taylor & francis",
                "no claim to original united states government works",
                "robots",
                "permission",
            },
            "uscg_cutter_seamanship_line_handling_2020": {
                "403",
                "redundan",
                "human suspension",
                "allowed official route",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_016_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_016"
        }
        self.assertEqual(len(batch), 12)
        expected = {
            "accept_targeted_scope": {
                "nist_nbsir_76_1146_personal_fall_safety_1977",
                "gutenberg_hasluck_knotting_splicing_1907",
                "mansoura_cotton_yarn_dynamic_friction_part2_1986",
                "wong_mcgrouther_surgical_knot_security_review_2023",
                "frontiers_bazzini_action_observation_execution_knots_2022",
                "plos_cross_physical_observational_knot_learning_2017",
                "frontiers_dempsey_coping_model_process_feedback_2017",
                "frontiers_ohrn_repeated_knot_skill_transfer_2025",
            },
            "defer": {
                "nps_conserve_o_gram_leather_dressings_2025",
                "smithsonian_nayland_blake_oral_history_2016",
                "nps_npf_lgbtq_america_leather_history_2016",
                "springer_gunning_disability_bdsm_communication_2023",
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

    def test_batch_016_accepts_preserve_methods_and_transfer_blocks(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "nist_nbsir_76_1146_personal_fall_safety_1977": {
                "date conflict",
                "obsolete non-bondage",
                "environmental conditioning",
                "human-suspension",
            },
            "gutenberg_hasluck_knotting_splicing_1907": {
                "verrill and dana",
                "territorial warning",
                "nooses",
                "figure-dependent",
            },
            "mansoura_cotton_yarn_dynamic_friction_part2_1986": {
                "1986, 2020 and 2021",
                "noarchive",
                "yarn is not finished rope",
                "knot security",
            },
            "wong_mcgrouther_surgical_knot_security_review_2023": {
                "single-extractor",
                "registered protocol",
                "static and cyclic",
                "secure bondage knot",
            },
            "frontiers_bazzini_action_observation_execution_knots_2022": {
                "54 knot-naive",
                "intermediate-step",
                "no-retention",
                "no-safety-outcome",
            },
            "plos_cross_physical_observational_knot_learning_2017": {
                "22 completers",
                "surprise reconstruction",
                "confound",
                "exploratory fmri",
            },
            "frontiers_dempsey_coping_model_process_feedback_2017": {
                "baseline self-efficacy imbalance",
                "process-versus-outcome",
                "self-efficacy and competence",
                "human-suspension",
            },
            "frontiers_ohrn_repeated_knot_skill_transfer_2025": {
                "two analyzed chains",
                "no functional/load testing",
                "tested educational prescription",
                "noose stimulus",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertEqual(source["decision"], "accept_targeted_scope")
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_016_deferrals_are_permission_or_policy_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        markers = {
            "nps_conserve_o_gram_leather_dressings_2025": {
                "joint authorship",
                "nonfederal",
                "page-specific",
                "treatment steps",
            },
            "smithsonian_nayland_blake_oral_history_2016": {
                "ai-train=no",
                "ordinary copyright",
                "first-person recollections",
                "universal history",
            },
            "nps_npf_lgbtq_america_leather_history_2016": {
                "all rights reserved",
                "national park foundation",
                "federal hosting",
                "japanese rope lineage",
            },
            "springer_gunning_disability_bdsm_communication_2023": {
                "exclusive license",
                "ended",
                "seven-person kink subset",
                "participant quote",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_017_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_017"
        }
        self.assertEqual(len(batch), 12)
        expected = {
            "accept_targeted_scope": {
                "fhwa_t5140_34_adhesive_anchors",
                "usace_ep_385_1_101_fall_protection",
                "usbr_testing_verifying_rope_access_anchors",
                "marchi_trees_supports_anchors_review",
                "detter_nondestructive_tree_anchorage",
                "gioffre_hemp_rope_environmental_aging",
                "zimniewska_hemp_fibre_review",
                "kaaronen_cross_cultural_knots",
                "singh_elastic_rod_capstan",
                "forer_westlake_chronic_pain_bdsm",
                "bochenska_knot_tying_instruction_video_rct",
            },
            "defer": {
                "fhwa_hrt_14_071_concrete_barrier_nde",
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

    def test_batch_017_accepts_preserve_domain_and_method_limits(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "fhwa_t5140_34_adhesive_anchors": {
                "no casualty detail",
                "continuous inspection",
                "diy ceiling",
                "human-suspension",
            },
            "usace_ep_385_1_101_fall_protection": {
                "qualified-person",
                "site-specific fall and rescue",
                "targeted official public-library",
                "intentional human suspension",
            },
            "usbr_testing_verifying_rope_access_anchors": {
                "protected final report document 1018",
                "non-additive",
                "diy proof-test",
                "human-suspension",
            },
            "marchi_trees_supports_anchors_review": {
                "root-plate",
                "empirical calibration",
                "non-bondage forestry",
                "human-suspension",
            },
            "detter_nondestructive_tree_anchorage": {
                "greater-than-280-tree",
                "arbosafe gmbh",
                "diy pulling test",
                "human-suspension",
            },
            "gioffre_hemp_rope_environmental_aging": {
                "one-commercial-rope",
                "no-calendar-life",
                "jute",
                "retirement threshold",
            },
            "zimniewska_hemp_fibre_review": {
                "narrative-not-systematic",
                "raw-fibre",
                "hemp-to-jute",
                "human-suspension",
            },
            "kaaronen_cross_cultural_knots": {
                "338",
                "86",
                "direct transmission",
                "kinbaku-lineage proof",
            },
            "singh_elastic_rod_capstan": {
                "perfectly flexible filament",
                "even without friction",
                "not a textile-rope experiment",
                "operational load",
            },
            "forer_westlake_chronic_pain_bdsm": {
                "525-person",
                "201 self-reported",
                "84.4-percent-white",
                "pain treatment",
            },
            "bochenska_knot_tying_instruction_video_rct": {
                "video alone was not established as superior",
                "all-rights-reserved preprint",
                "no retention",
                "rope teaching",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertEqual(source["decision"], "accept_targeted_scope")
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_017_contractor_rights_correction_is_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        source = by_id["fhwa_hrt_14_071_concrete_barrier_nde"]
        self.assertEqual(source["decision"], "defer")
        self.assertNotIn(source["candidate_id"], queued)
        text = json.dumps(source, sort_keys=True).lower()
        for marker in {
            "performed under contract",
            "no explicit federal authorship",
            "metadata only",
            "written fhwa or contractor",
            "federal funding and hosting do not establish public-domain status",
        }:
            self.assertIn(marker, text)

    def test_batch_018_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_018"
        }
        self.assertEqual(len(batch), 15)
        expected = {
            "accept_targeted_scope": {
                "hse_oc_282_31_rope_evacuation",
                "dasci_surgical_knot_training_rct_2023",
                "andy_buru_history_japanese_rope_bondage",
                "doj_ada_effective_communication",
                "nistir_6096_post_installed_anchors_review",
            },
            "defer": {
                "gutenberg_jutsum_knots_bends_splices_30983",
                "gutenberg_aldridge_marlinespike_knots_78376",
                "ontario_firefighter_rope_rescue_guidance",
                "nzqa_rope_rescue_skill_standards_40866_40867",
                "cbe_hemp_sisal_rope_temperature_2015",
                "cjds_goldberg_disability_bdsm_law_2018",
                "fema_e74_nonstructural_ceiling_systems",
                "nycu_orientalist_gaze_japanese_rope_bondage",
                "western_canada_mine_rescue_manual",
            },
            "reject": {"hojojutsu_research_society_primary"},
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

    def test_batch_018_accepts_mirror_queue_and_preserve_limits(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "hse_oc_282_31_rope_evacuation": {
                "2003 version and 2013 review date",
                "error under stress",
                "non-bondage self-evacuation",
                "human-suspension",
            },
            "dasci_surgical_knot_training_rct_2023": {
                "124-dental-student",
                "peyton four-step",
                "no retention",
                "no transfer to rope technique",
            },
            "andy_buru_history_japanese_rope_bondage": {
                "written-text-only",
                "first-person practitioner",
                "commercial book/school context",
                "consensus history",
            },
            "doj_ada_effective_communication": {
                "consult the person",
                "clarify rather than guess",
                "covered-entity legal context",
                "makes a technique safe",
            },
            "nistir_6096_post_installed_anchors_review": {
                "historical secondary-review",
                "qualified engineering verification",
                "non-bondage concrete-anchor",
                "human-suspension system",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertEqual(source["decision"], "accept_targeted_scope")
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_018_deferrals_and_rejection_are_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        markers = {
            "gutenberg_jutsum_knots_bends_splices_30983": {
                "prohibit automated tools",
                "manual single-item",
                "metadata only",
            },
            "gutenberg_aldridge_marlinespike_knots_78376": {
                "prohibit automated tools",
                "high duplication",
                "metadata only",
            },
            "ontario_firefighter_rope_rescue_guidance": {
                "no visible ontario open government licence label",
                "unmodified noncommercial",
                "explicit permission",
            },
            "nzqa_rope_rescue_skill_standards_40866_40867": {
                "citation or reference",
                "commercial uses",
                "no body markdown",
            },
            "cbe_hemp_sisal_rope_temperature_2015": {
                "no creative commons license",
                "supplier-confounded",
                "model-training permission",
            },
            "cjds_goldberg_disability_bdsm_law_2018": {
                "cc by-nc-nd 4.0",
                "transformed commercial",
                "explicit permission",
            },
            "fema_e74_nonstructural_ceiling_systems": {
                "http 403",
                "applied technology council",
                "federal sponsorship and hosting do not establish",
            },
            "nycu_orientalist_gaze_japanese_rope_bondage": {
                "ordinary copyright",
                "speaker, reporter and university permission",
                "consensus scholarship",
            },
            "western_canada_mine_rescue_manual": {
                "all rights reserved",
                "mixed public-private compilation",
                "component-level clearance",
            },
            "hojojutsu_research_society_primary": {
                "named ai and data bots",
                "combative restraint",
                "do not crawl",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            self.assertIn(source["decision"], {"defer", "reject"})
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_019_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_019"
        }
        self.assertEqual(len(batch), 10)
        expected = {
            "accept_targeted_scope": {
                "hse_temporary_works_faqs",
                "bennett_pierre_knot_spatial_skill_2025",
                "lang_coping_error_video_knot_training_2023",
            },
            "defer": {
                "usfs_fpl_wood_condition_assessment_2025",
                "wang_natural_rope_marine_degradation_2021",
                "wire_accessible_consent_guidelines_2025",
                "levy_operant_surgical_knot_training_2016",
                "liu_self_directed_video_feedback_2026",
                "wahlborg_friction_knot_terminology_2025",
                "nhs_fainting_first_aid_currency_gate",
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

    def test_batch_019_accepts_mirror_queue_and_preserve_limits(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "hse_temporary_works_faqs": {
                "2026-03-11",
                "site assumptions",
                "design and coordination are distinct roles",
                "human suspension",
            },
            "bennett_pierre_knot_spatial_skill_2025": {
                "279 and 147",
                "explicitly label every teaching use an inference",
                "did not train learners",
                "gender-performance claims",
            },
            "lang_coping_error_video_knot_training_2023": {
                "55 laparoscopically naive",
                "no significant difference",
                "right-handed-only",
                "no retention",
                "human-suspension",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertEqual(source["decision"], "accept_targeted_scope")
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_019_deferrals_are_rights_access_or_currency_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        markers = {
            "usfs_fpl_wood_condition_assessment_2025": {
                "mixed federal/nonfederal",
                "111 physical",
                "federal hosting does not",
                "human-suspension",
            },
            "wang_natural_rope_marine_degradation_2021": {
                "official pdf route returned http 403",
                "do not reconstruct",
                "marine immersion",
                "retirement claim",
            },
            "wire_accessible_consent_guidelines_2025": {
                "restricted access",
                "mary ann liebert",
                "participatory",
                "legal or clinical equivalence",
            },
            "levy_operant_surgical_knot_training_2016": {
                "association of bone and joint surgeons",
                "tagteach",
                "single method-skilled",
                "human-suspension",
            },
            "liu_self_directed_video_feedback_2026": {
                "cc by-nc-nd 4.0",
                "62-participant",
                "technology-company",
                "deepseek or gpt",
            },
            "wahlborg_friction_knot_terminology_2025": {
                "personal use only",
                "160 complete",
                "side-by-side",
                "shibari",
            },
            "nhs_fainting_first_aid_currency_gate": {
                "2026-02-23",
                "2025-03-15",
                "overdue",
                "ordinary fainting guidance is a bondage-suspension protocol",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_020_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_020"
        }
        self.assertEqual(len(batch), 10)
        expected = {
            "accept_targeted_scope": {
                "ebrahim_household_knot_video_2024",
                "raythatha_video_adjunct_knot_retention_2024",
                "drury_wool_rope_aquaculture_review_2022",
                "martin_ecological_knot_motor_learning_2025",
            },
            "defer": {
                "sun_feedback_valence_suturing_rct_2026",
                "lu_video_peer_feedback_residents_2025",
                "edo_four_cs_prevention_leaflet",
                "johnson_consent_negotiation_thesis_2025",
                "war_department_shop_work_rope_lesson_1942",
            },
            "reject": {"prorope_sisal_inhouse_test_2026"},
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

    def test_batch_020_accepts_mirror_queue_and_preserve_limits(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "ebrahim_household_knot_video_2024": {
                "304 eligible students",
                "71 voluntary participants",
                "no control group",
                "no retention test",
                "human-suspension safety",
            },
            "raythatha_video_adjunct_knot_retention_2024": {
                "29 knot-naive",
                "only 6 intervention and 8 control",
                "weak and nonsignificant",
                "human-suspension safety",
            },
            "drury_wool_rope_aquaculture_review_2022": {
                "strength and durability in the intended aquaculture use were untested",
                "not finished rope",
                "do not transfer wool findings to hemp or jute",
                "human-suspension performance",
            },
            "martin_ecological_knot_motor_learning_2025": {
                "42 adults",
                "low-to-moderate lab-to-ecological correlations",
                "absence of long-term retention",
                "human-suspension safety",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertEqual(source["decision"], "accept_targeted_scope")
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_020_restricted_components_and_commerce_are_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        markers = {
            "sun_feedback_valence_suturing_rct_2026": {
                "cc by-nc-nd 4.0",
                "42-person three-arm",
                "explicit permission",
                "rope-teaching validation",
            },
            "lu_video_peer_feedback_residents_2025": {
                "cc by-nc-nd 4.0",
                "33-resident",
                "no significant suturing difference",
                "no retention",
            },
            "edo_four_cs_prevention_leaflet": {
                "cc by-nc-nd",
                "confiance, conscience, communication and consentement",
                "do not download, ocr, translate",
                "complete consent or rope-safety protocol",
            },
            "johnson_consent_negotiation_thesis_2025": {
                "1,118-participant",
                "all-rights-reserved",
                "do not teach the abstract's role comparison",
                "written author permission",
            },
            "war_department_shop_work_rope_lesson_1942": {
                "bell system practices",
                "printed pages 99 through 133",
                "page-and-paragraph component provenance audit",
                "1942 historical evidence",
            },
            "prorope_sisal_inhouse_test_2026": {
                "five-test in-house provenance",
                "ai-assisted",
                "ordinary-copyright",
                "human-suspension safety",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            self.assertIn(source["decision"], {"defer", "reject"})
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_021_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_021"
        }
        self.assertEqual(len(batch), 12)
        expected = {
            "accept_targeted_scope": {
                "nist_manila_rope_tests_t198_1921",
                "gutenberg_brady_kedge_anchor_77729",
                "gutenberg_dana_seamans_friend_40958",
                "innotrac_camera_visual_rope_inspection_2020",
            },
            "defer": {
                "imca_natural_fibre_ladder_failure_2021",
                "ccohs_fibre_rope_slings_2023",
                "quartz_rope_accessible_curriculum",
                "rope_office_hours_accessibility_negotiation",
                "safelink_alberta_kinky_sex_toolkit_2024",
                "sedici_latam_shibari_genealogy_187026",
                "waseda_ito_seiu_semeba_manuscript",
            },
            "reject": {"rackwiki_bondage_safety_contamination"},
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

    def test_batch_021_accepts_mirror_queue_and_preserve_limits(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "nist_manila_rope_tests_t198_1921": {
                "t198",
                "commercial three-strand regular-lay",
                "purchase-order sampling",
                "6 mm jute or hemp",
                "human suspension",
            },
            "gutenberg_brady_kedge_anchor_77729": {
                "sanctioned mirror",
                "human-user-only main-site terms",
                "automatically generated summary",
                "human-suspension validation",
            },
            "gutenberg_dana_seamans_friend_40958": {
                "sanctioned mirror",
                "semantic deduplication",
                "maritime law",
                "human-suspension safety",
            },
            "innotrac_camera_visual_rope_inspection_2020": {
                "one-sided manual surface viewing",
                "discard criteria",
                "surface imaging does not establish internal condition",
                "human-suspension fitness",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertEqual(source["decision"], "accept_targeted_scope")
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_021_permissions_ai_access_and_contamination_are_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        markers = {
            "imca_natural_fibre_ladder_failure_2021": {
                "express derivative commercial model-training permission",
                "member-supplied components",
                "single-incident provenance",
            },
            "ccohs_fibre_rope_slings_2023": {
                "internal-copy limitation",
                "written permission",
                "industrial sling context",
            },
            "quartz_rope_accessible_curriculum": {
                "ordinary-copyright",
                "tied-person co-teaching",
                "independent current clinical corroboration",
            },
            "rope_office_hours_accessibility_negotiation": {
                "gptbot",
                "chatgpt-user",
                "do not crawl, copy, paraphrase",
            },
            "safelink_alberta_kinky_sex_toolkit_2024": {
                "ai-train=no",
                "aids committee",
                "do not retrieve, copy, paraphrase",
            },
            "sedici_latam_shibari_genealogy_187026": {
                "cc by-nc-sa 4.0",
                "converter's access authorization",
                "situated genealogy and universal history",
            },
            "rackwiki_bondage_safety_contamination": {
                "oldid 2073",
                "incomplete license rendering",
                "do not copy, paraphrase, translate",
            },
            "waseda_ito_seiu_semeba_manuscript": {
                "robots-disallowed image route",
                "do not access, copy, ocr",
                "staged imagery, not a tying manual",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            self.assertIn(source["decision"], {"defer", "reject"})
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

        for candidate_id in {
            "rope_office_hours_accessibility_negotiation",
            "safelink_alberta_kinky_sex_toolkit_2024",
            "waseda_ito_seiu_semeba_manuscript",
        }:
            self.assertFalse(by_id[candidate_id]["accessible"])

    def test_batch_022_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_022"
        }
        self.assertEqual(len(batch), 10)
        expected = {
            "accept_targeted_scope": {
                "nist_manila_rope_color_serviceability_1933",
                "datta_jute_water_retting_2024",
            },
            "defer": {
                "uiaa_sharp_edges_rope_cuts_2025",
                "irata_historical_rope_failure_bulletins",
                "bmc_climbing_rope_guide",
                "bdsm_dojo_absolute_bondage_safety_resources",
                "oman_dynamic_climbing_rope_wear_2025",
                "harukumo_juku_lineage_encyclopedia",
                "jstage_nukada_japanese_knotting_1956_1958",
            },
            "reject": {"kinbakuwiki_german_reference"},
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

    def test_batch_022_accepts_mirror_queue_and_preserve_limits(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "nist_manila_rope_color_serviceability_1933": {
                "did not test serviceability",
                "lubricant",
                "color-based keep or discard rule",
                "human suspension",
            },
            "datta_jute_water_retting_2024": {
                "microbial production stage",
                "crijaf sona",
                "finished rope",
                "dirty or safe",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertEqual(source["decision"], "accept_targeted_scope")
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_022_rights_access_and_contamination_are_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        markers = {
            "uiaa_sharp_edges_rope_cuts_2025": {
                "derivative commercial model-training permission",
                "abandoned-test limitation",
                "dynamic rope incidents",
            },
            "irata_historical_rope_failure_bulletins": {
                "ai-train=no",
                "do not retrieve, copy, paraphrase",
                "historical recommendations",
            },
            "bmc_climbing_rope_guide": {
                "personal noncommercial",
                "prior written permission",
                "dynamic synthetic climbing-rope guidance",
            },
            "bdsm_dojo_absolute_bondage_safety_resources": {
                "google drive",
                "qualified current clinical review",
                "unspecified medical-professional",
            },
            "oman_dynamic_climbing_rope_wear_2025": {
                "cc by-nc-nd",
                "dynamic polyamide kernmantle",
                "every exact value",
            },
            "kinbakuwiki_german_reference": {
                "do not crawl, copy, paraphrase",
                "role labels",
                "unsafe cultural generalization",
            },
            "harukumo_juku_lineage_encyclopedia": {
                "rope-collar exercises",
                "simplest or safest",
                "first-party accounts",
            },
            "jstage_nukada_japanese_knotting_1956_1958": {
                "pdf robots prohibition",
                "obsolete and offensive",
                "strangulation",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            self.assertIn(source["decision"], {"defer", "reject"})
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

        self.assertFalse(
            by_id["irata_historical_rope_failure_bulletins"]["accessible"]
        )

    def test_batch_023_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_023"
        }
        self.assertEqual(len(batch), 10)
        expected = {
            "accept_targeted_scope": {
                "coir_ropes_french_polynesia_2024",
                "scientific_reports_biodegradable_gillnet_knot_2024",
                "scientific_reports_silk_slip_knot_mechanics_2016",
            },
            "defer": {
                "qeswachaka_festuca_rope_mechanics_2024",
                "brin_banana_peduncle_rope_2025",
                "navsea_nstm_chapter_613_rope_1999",
                "bioresources_antibacterial_natural_fibres_2014",
                "rice_straw_rope_mechanics_2022",
                "matsushita_cotton_gillnet_degradation_2008",
            },
            "reject": {"e3s_hibiscus_bark_rope_2020"},
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

    def test_batch_023_accepts_mirror_queue_and_preserve_limits(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "coir_ropes_french_polynesia_2024": {
                "machine-twisted and hand-braided",
                "individual-fiber response",
                "polyacht",
                "human-support rope",
            },
            "scientific_reports_biodegradable_gillnet_knot_2024": {
                "double weaver's knot",
                "prior material degradation",
                "only three specimens",
                "dry synthetic fishing monofilament",
            },
            "scientific_reports_silk_slip_knot_mechanics_2016": {
                "raw versus chemically degummed",
                "one topology loosened while the other tightened",
                "single silk fiber is not finished rope",
                "human-suspension rule",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertEqual(source["decision"], "accept_targeted_scope")
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_023_gated_and_rejected_sources_are_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        markers = {
            "qeswachaka_festuca_rope_mechanics_2024": {
                "community-centered review",
                "four named bridge-renewal communities",
                "heritage bridge study",
            },
            "brin_banana_peduncle_rope_2025": {
                "chemical-safety review",
                "specimen-code inconsistencies",
                "hazardous treatment",
            },
            "navsea_nstm_chapter_613_rope_1999": {
                "third-party 1999 mirror",
                "current authenticated text",
                "not by itself a copyright or ai-training license",
            },
            "bioresources_antibacterial_natural_fibres_2014": {
                "noncommercial use only",
                "do not automatically survive retting",
                "finished-rope hygiene",
            },
            "rice_straw_rope_mechanics_2022": {
                "ordinary elsevier copyright",
                "one rice variety",
                "rice straw is not jute",
            },
            "e3s_hibiscus_bark_rope_2020": {
                "seven chemically treated single fibers",
                "conflicting pdf doi",
                "absence of finished-rope manufacture or testing",
            },
            "matsushita_cotton_gillnet_degradation_2008": {
                "japanese society of fisheries science",
                "treated cotton mesh",
                "seawater exposure is not hygiene",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            self.assertIn(source["decision"], {"defer", "reject"})
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_024_decisions_are_complete_and_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_024"
        }
        self.assertEqual(len(batch), 10)
        expected = {
            "accept_high_priority": {
                "escalona_rope_sheave_static_contact_2023",
            },
            "accept_targeted_scope": {
                "doe_std_1090_2020_hoisting_rigging",
                "atsb_pilot_ladder_failure_occurrence_2025",
                "osha_shipyard_anchorages_attachments_etool",
                "nasa_std_8719_9c_lifting_2024",
            },
            "defer": {
                "sano_clove_hitch_mechanics_2022",
                "asan_real_talk_disability_sexual_health_toolkit",
                "patil_topological_mechanics_knots_2020",
                "mod_dio_natural_fibre_climbing_rope_failure_2021",
                "kinbakuodyssey_odys_naka_lineage_profile",
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
        self.assertEqual(
            queued_from_batch,
            expected["accept_high_priority"] | expected["accept_targeted_scope"],
        )

    def test_batch_024_accepts_mirror_queue_and_preserve_limits(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "escalona_rope_sheave_static_contact_2023": {
                "fully stuck and saturated-friction",
                "ideal rod without bending stiffness",
                "dynamic or impending-slip capstan",
                "safe-load rule",
            },
            "doe_std_1090_2020_hoisting_rigging": {
                "foreword's inventory",
                "asme, ansi",
                "hostile-environment governance",
                "does not qualify a domestic hardpoint",
            },
            "atsb_pilot_ladder_failure_occurrence_2025": {
                "did not independently verify",
                "operator-reported",
                "chemical contamination",
                "not established causes",
            },
            "osha_shipyard_anchorages_attachments_etool": {
                "support condition and connection geometry",
                "visible attachment point is not a qualified hardpoint",
                "vsra",
                "does not qualify any tree",
            },
            "nasa_std_8719_9c_lifting_2024": {
                "mishaps and close calls",
                "expressly non-comprehensive",
                "voluntary-consensus-standard",
                "does not qualify a rope",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertTrue(source["decision"].startswith("accept_"))
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_024_gated_sources_are_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        markers = {
            "sano_clove_hitch_mechanics_2022": {
                "ai-train=no",
                "do not retrieve, copy, paraphrase",
                "cc by-nc-nd",
            },
            "asan_real_talk_disability_sexual_health_toolkit": {
                "direct communication rather than routing through support people",
                "rope-scene application as an inference",
                "ordinary copyright",
            },
            "patil_topological_mechanics_knots_2020": {
                "exclusive aaas license",
                "some rights reserved",
                "specialized elastic",
            },
            "mod_dio_natural_fibre_climbing_rope_failure_2021": {
                "missing landing or status record",
                "british standards",
                "not a laboratory cause finding",
            },
            "kinbakuodyssey_odys_naka_lineage_profile": {
                "self-report label",
                "akira naka",
                "naka-style authenticity",
                "copyright 2026",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

        self.assertFalse(by_id["sano_clove_hitch_mechanics_2022"]["accessible"])

    def test_batch_025_decisions_are_deterministic_and_accepted_are_queued(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_025"
        }
        expected = {
            "accept_high_priority": {
                "consent_gov_au_national_consent_resources",
                "ndis_supported_decision_making_intimacy",
                "esafety_consent_and_intimate_image_privacy",
                "govuk_disability_relationships_thematic_report_2025",
                "ndl_seiu_seme_no_hanashi_1929",
            },
            "accept_targeted_scope": {
                "govuk_trauma_informed_practice_definition_2022",
                "govuk_accessible_communication_formats_2026",
            },
            "defer": {
                "nsw_make_no_doubt_consent",
                "better_health_disability_sexuality_pages",
                "ndl_torinawajutsu_osso_kanshi_1921",
            },
        }
        self.assertEqual(len(batch), 10)
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
        self.assertEqual(
            queued_from_batch,
            expected["accept_high_priority"] | expected["accept_targeted_scope"],
        )

    def test_batch_025_accepts_mirror_queue_and_preserve_limits(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "consent_gov_au_national_consent_resources": {
                "free, voluntary and informed",
                "body language",
                "images and photographs",
                "certifies a rope scene",
            },
            "ndis_supported_decision_making_intimacy": {
                "decisions made by them and not for them",
                "right to intimacy and sexual expression",
                "language, mode of communication",
                "do not infer",
            },
            "esafety_consent_and_intimate_image_privacy": {
                "does not authorize onward sharing",
                "recording or screenshot",
                "images, videos and songs",
                "jurisdiction-specific",
            },
            "govuk_trauma_informed_practice_definition_2022": {
                "safety, trust, choice, collaboration, empowerment and cultural consideration",
                "this is not trauma treatment",
                "rope-scene application is an inference",
            },
            "govuk_disability_relationships_thematic_report_2025": {
                "nothing about us without us",
                "assumed asexuality",
                "alternative communication",
                "third-party quotation",
            },
            "govuk_accessible_communication_formats_2026": {
                "easy read is not the same as plain language",
                "html",
                "format proves understanding or consent",
                "makaton",
            },
            "ndl_seiu_seme_no_hanashi_1929": {
                "pdm",
                "japanese-language historical reviewer",
                "no direct hojojutsu-to-kinbaku lineage inference",
                "operational restraint",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertTrue(source["decision"].startswith("accept_"))
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_025_gated_sources_are_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        markers = {
            "nsw_make_no_doubt_consent": {
                "third party copyright limitations",
                "social-media",
                "default site",
            },
            "better_health_disability_sexuality_pages": {
                "adapt, reproduce, store",
                "written permission",
                "medical",
            },
            "ndl_torinawajutsu_osso_kanshi_1921": {
                "commissioner for cultural affairs",
                "internet publication by arbitration",
                "actionable restraint",
                "metadata only",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_026_decisions_are_deterministic_and_accepted_are_queued(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_026"
        }
        expected = {
            "accept_targeted_scope": {
                "rijksmuseum_yoshitoshi_adachi_moor_records",
                "artic_oshu_adachi_ga_hara_kabuki_1777",
            },
            "defer": {
                "smithsonian_ito_seiu_art_knots_ropes_seme_e",
                "japan_arts_council_adachigahara_performance_context",
                "nga_yoshitoshi_lonely_house_adachi_moor_1885",
                "secca_introduction_to_consent_easy_english_2024",
                "safer_me_safer_you_national_social_sexual_safety_guidelines_2025",
                "wwda_neve_intellectual_disability_sexual_consent_guide",
                "sacid_relationship_wise_easy_read_bundle",
                "japan_house_london_kumihimo_wrapping_context",
            },
            "reject": {
                "cleveland_open_access_knotting_object_metadata",
            },
        }
        self.assertEqual(len(batch), 11)
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

    def test_batch_026_accepts_mirror_queue_and_block_lineage_transfer(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "rijksmuseum_yoshitoshi_adachi_moor_records": {
                "public domain",
                "modern kinbaku",
                "every image",
                "keep the 1885 and 1890 objects distinct",
            },
            "artic_oshu_adachi_ga_hara_kabuki_1777": {
                "cc0",
                "same narrative",
                "ichimura theater",
                "consent or safety",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertEqual(source["decision"], "accept_targeted_scope")
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_026_gated_and_false_relevance_sources_are_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        markers = {
            "smithsonian_ito_seiu_art_knots_ropes_seme_e": {
                "usage conditions apply",
                "conflicting",
                "exact-object cc0",
            },
            "japan_arts_council_adachigahara_performance_context": {
                "non-personal storage",
                "shared place or title",
                "modern kinbaku lineage",
            },
            "nga_yoshitoshi_lonely_house_adachi_moor_1885": {
                "all rights reserved",
                "underlying public-domain artwork",
                "prefer the rights-clear rijksmuseum",
            },
            "secca_introduction_to_consent_easy_english_2024": {
                "all rights reserved",
                "multiple communication modes",
                "format proves comprehension or consent",
            },
            "safer_me_safer_you_national_social_sexual_safety_guidelines_2025": {
                "personal, non-commercial",
                "lived-experience quotations",
                "unsupported transfer to individuals",
            },
            "wwda_neve_intellectual_disability_sexual_consent_guide": {
                "express approval",
                "body language can supplement but not replace",
                "disability means incapacity",
            },
            "sacid_relationship_wise_easy_read_bundle": {
                "free cart",
                "photosymbols",
                "do not enter checkout",
            },
            "japan_house_london_kumihimo_wrapping_context": {
                "commercial use without permission",
                "direct kinbaku lineage",
                "decorative cord",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

        rejected = by_id["cleveland_open_access_knotting_object_metadata"]
        self.assertEqual(rejected["decision"], "reject")
        self.assertNotIn(rejected["candidate_id"], queued)
        rejected_text = json.dumps(rejected, sort_keys=True).lower()
        self.assertIn("false-relevance", rejected_text)
        self.assertIn("keyword coincidence", rejected_text)
        self.assertIn("does not teach bondage-relevant", rejected_text)

    def test_batch_027_accepts_only_the_exact_cc_by_sa_guidelines_work(self) -> None:
        batch = [
            row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_027"
        ]
        self.assertEqual(len(batch), 10)
        self.assertEqual(
            Counter(row["decision"] for row in batch),
            Counter({"accept_targeted_scope": 1, "defer": 9}),
        )

        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {
            row["discovery_candidate_id"]: row
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        source = by_id["intimacy_on_set_guidelines_2019"]
        queued = queue_by_id[source["candidate_id"]]
        self.assertEqual(source["decision"], "accept_targeted_scope")
        self.assertEqual(source["priority_score"], 38)
        self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
        self.assertEqual(queued["training_use"], source["training_use"])
        self.assertEqual(queued["rights_basis"], source["rights_basis"])
        text = json.dumps(source, sort_keys=True).lower()
        for marker in {
            "cc by-sa 4.0",
            "sharealike",
            "surrounding site content",
            "rope-scene application as an inference",
            "does not teach tying",
        }:
            self.assertIn(marker, text)

    def test_batch_027_permission_and_component_gates_are_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        markers = {
            "la_rope_yukimura_style_lineage_and_concepts": {
                "all rights reserved",
                "self-identification",
                "nonverbal response as consent",
            },
            "performance_philosophy_being_surface_2022": {
                "cc by-nc-sa 4.0",
                "noncommercial",
                "violent or self-injury method",
            },
            "tender_container_how_to_self_suspend_performance": {
                "no open license",
                "photograph",
                "title is instructional",
            },
            "kinbaku_society_berlin_magazine_archive": {
                "purchase-gated",
                "do not enter a cart",
                "independently corroborated",
            },
            "platform_caring_for_limits_sex_workers_opera_2022": {
                "all rights reserved",
                "participant quotations",
                "alternative role",
            },
            "uk_nos_intimacy_coordination_productions_2025": {
                "no clear open license",
                "standard owner",
                "competence outcomes and a curriculum",
            },
            "edit_media_teaching_intimacy_coordination_guide": {
                "all rights reserved",
                "sentence-level provenance",
                "third-party quotations",
            },
            "intimacy_practitioners_sa_protocols_2021": {
                "all rights reserved",
                "south african film",
                "cannot be silently converted into rope rules",
            },
            "ut_austin_staging_intimacy_policies": {
                "theatrical intimacy education",
                "component-level provenance",
                "no open reuse license",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_028_decisions_and_queue_are_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_028"
        }
        expected = {
            "accept_targeted_scope": {
                "commons_midori_zoe_rope_interview_2018",
                "endless_knot_isobel_williams_shibari_transcript_2021",
            },
            "defer": {
                "escola_shibari_137_incident_synthesis",
                "kinbaku_japan_voices_yuna_transcripts",
                "rope_radar_negotiation_framework",
                "ayco_ace_basic_risk_assessment_2020",
                "uba_4toscuro_shibari_photography_2024",
                "udistrital_shibari_communication_thesis_2026",
                "mdpi_park_sabu_japanese_gay_magazines_2025",
                "commons_shay_tiziano_beginner_column_tie_pair",
                "kohl_mugo_african_queer_rope_perspectives_2017",
            },
            "reject": {
                "soulrope_art_of_shibari_as_writing_2023",
            },
        }
        self.assertEqual(len(batch), 12)
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

    def test_batch_028_accepts_mirror_queue_and_preserve_testimony_limits(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "commons_midori_zoe_rope_interview_2018": {
                "cc by 3.0",
                "false claim",
                "particular style",
                "never replaces direct ongoing check-ins",
                "no audiovisual media",
            },
            "endless_knot_isobel_williams_shibari_transcript_2021": {
                "cc by-sa 4.0",
                "mostly automatic",
                "gorgone",
                "john clegg quotation",
                "universal safety protocol",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertEqual(source["decision"], "accept_targeted_scope")
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_028_permission_access_and_safety_gates_are_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        markers = {
            "escola_shibari_137_incident_synthesis": {
                "private fetlife origin",
                "prevalence estimate",
                "methods, provenance, privacy and ethics review",
            },
            "kinbaku_japan_voices_yuna_transcripts": {
                "ordinary copyright",
                "secondary medical summaries",
                "direct written permission",
            },
            "rope_radar_negotiation_framework": {
                "never expanding practices or boundaries",
                "completed radar proves consent or safety",
                "private and workshop-use statements",
            },
            "ayco_ace_basic_risk_assessment_2020": {
                "template permission is not inherited",
                "quoted third-party definitions",
                "certifies a bondage hardpoint",
            },
            "uba_4toscuro_shibari_photography_2024": {
                "cc by 3.0 versus cc by-nc 4.0",
                "three coauthors",
                "no external sample",
            },
            "udistrital_shibari_communication_thesis_2026": {
                "deposit agreement does not substitute",
                "bodily response alone",
                "therapeutic, healing, transformative",
            },
            "mdpi_park_sabu_japanese_gay_magazines_2025": {
                "akamai",
                "researchgate",
                "third-party magazine",
            },
            "commons_shay_tiziano_beginner_column_tie_pair": {
                "use computer vision",
                "suspension-suitable",
                "current safety review",
            },
            "kohl_mugo_african_queer_rope_perspectives_2017": {
                "cc by-nc-sa 4.0",
                "universal fact",
                "safeword or consent as a complete physical-safety protocol",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

        rejected = by_id["soulrope_art_of_shibari_as_writing_2023"]
        self.assertEqual(rejected["decision"], "reject")
        self.assertNotIn(rejected["candidate_id"], queued)
        rejected_text = json.dumps(rejected, sort_keys=True).lower()
        self.assertIn("cc by-nc 4.0", rejected_text)
        self.assertIn("weak source-critical history", rejected_text)
        self.assertIn("unsafe unsupported claims", rejected_text)

    def test_batch_029_decisions_and_queue_are_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_029"
        }
        expected = {
            "accept_targeted_scope": {
                "ndl_kitan_club_serial_bibliography",
                "sprechkontakt_lut074_shibari_podcast",
            },
            "defer": {
                "rope_podcast_fox_mya_oral_history_archive",
                "diva_bundna_positioner_kinbaku_thesis",
                "knot_without_risk_v1",
                "seilhafen_consent_communication_basics",
                "commons_shay_tiziano_self_suspension_video_set",
                "vleinad_diverse_dynamics_rope_workshop",
                "cuni_therapeutic_potential_shibari_thesis",
            },
            "reject": {
                "kinky_queeries_rope_bondage_episode_pair",
                "smithsonian_bo_shibari_kyogen_false_relevance",
                "chaotic_devices_rope_studio_safety_handouts",
                "homosaurus_rope_bondage_vocabulary",
                "rope_rituals_nerve_risks_v3",
            },
        }
        self.assertEqual(len(batch), 14)
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

    def test_batch_029_accepts_mirror_queue_and_block_identifier_trivia(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "ndl_kitan_club_serial_bibliography": {
                "no restriction on purpose",
                "ndl holdings",
                "kinbaku lineage",
                "memorization question",
            },
            "sprechkontakt_lut074_shibari_podcast": {
                "cc by-sa 4.0",
                "orf broadcast",
                "independent corroboration",
                "episode url",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertEqual(source["decision"], "accept_targeted_scope")
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_029_permission_review_and_false_fact_sources_are_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        deferred_markers = {
            "rope_podcast_fox_mya_oral_history_archive": {
                "all rights reserved",
                "do not automatically ingest",
                "listener-identifying trauma",
            },
            "diva_bundna_positioner_kinbaku_thesis": {
                "anubis",
                "open access is not permission",
                "participant-privacy",
            },
            "knot_without_risk_v1": {
                "cc by-nc-nd 4.0",
                "attempting to identify them",
                "independent clinical review",
            },
            "seilhafen_consent_communication_basics": {
                "0-10 maximum-tolerable-pain",
                "victim-blame",
                "not law, medical advice",
            },
            "commons_shay_tiziano_self_suspension_video_set": {
                "run computer vision",
                "hardpoint capacity",
                "current safety review",
            },
            "vleinad_diverse_dynamics_rope_workshop": {
                "cc by-nc-nd 4.0",
                "participant names",
                "universal rope roles",
            },
            "cuni_therapeutic_potential_shibari_thesis": {
                "six semi-structured",
                "perceived potential",
                "review ethics, recruitment",
            },
        }
        for candidate_id, required in deferred_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

        rejected_markers = {
            "kinky_queeries_rope_bondage_episode_pair": {
                "coming soon",
                "canonical-link memorization",
            },
            "smithsonian_bo_shibari_kyogen_false_relevance": {
                "kyogen",
                "modern kinbaku lineage",
            },
            "chaotic_devices_rope_studio_safety_handouts": {
                "pseudo-scientific",
                "palpation",
            },
            "homosaurus_rope_bondage_vocabulary": {
                "taxonomy-flattening",
                "identifier",
            },
            "rope_rituals_nerve_risks_v3": {
                "thumb-pressure",
                "ice treatment",
            },
        }
        for candidate_id, required in rejected_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "reject")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_030_decisions_and_queue_are_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_030"
        }
        expected = {
            "accept_targeted_scope": {
                "apj_ito_seiu_artist_record",
                "ndl_postwar_sm_serial_metadata_set",
                "usp_takato_yamamoto_erotic_grotesque_art_2024",
            },
            "defer": {
                "edo_tokyo_ito_seiu_ghost_painting_exhibition",
                "polypublie_nine_circus_apparatus_dynamic_forces",
                "circusrigging_single_point_direct_shared_pulley_systems",
                "christian_red_cutie_crusher_wheelchair_rope_access",
                "agorha_pourcine_shibari_art_thesis_metadata",
                "suspendulum_frame_specification_sheet",
                "kyushu_tinsley_seiu_kusozu_2017",
            },
            "reject": {
                "commons_adriannespring_obsession_bondage_video",
                "wikibooks_rope_making_for_bondage_use",
                "internet_archive_calafou_shibari_workshop",
                "edinburgh_psychotherapist_rope_self_case_thesis",
            },
        }
        self.assertEqual(len(batch), 14)
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

    def test_batch_030_accepts_mirror_queue_and_block_identifier_trivia(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "apj_ito_seiu_artist_record": {
                "government standard terms 2.0",
                "embedded wikipedia",
                "kinbaku lineage",
                "url-memory tasks",
            },
            "ndl_postwar_sm_serial_metadata_set": {
                "no restriction on purpose",
                "holdings gaps",
                "complete publication run",
                "memorization questions",
            },
            "usp_takato_yamamoto_erotic_grotesque_art_2024": {
                "cc by 4.0",
                "independently cross-check",
                "third-party components",
                "source urls",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertEqual(source["decision"], "accept_targeted_scope")
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_030_permission_review_and_unsafe_sources_are_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        deferred_markers = {
            "edo_tokyo_ito_seiu_ghost_painting_exhibition": {
                "ordinary copyright",
                "suzuki toshio",
                "kinbaku lineage",
            },
            "polypublie_nine_circus_apparatus_dynamic_forces": {
                "cc by-nc-nd 4.0",
                "118 movements",
                "do not certify or directly set values",
            },
            "circusrigging_single_point_direct_shared_pulley_systems": {
                "members-only article",
                "do not access",
                "does not certify a bondage upline",
            },
            "christian_red_cutie_crusher_wheelchair_rope_access": {
                "both coauthors",
                "asking instead of assuming",
                "all wheelchair users",
            },
            "agorha_pourcine_shibari_art_thesis_metadata": {
                "no thesis body",
                "do not infer",
                "bibliographic trivia",
            },
            "suspendulum_frame_specification_sheet": {
                "request-only certificate",
                "do not copy or train on load",
                "independent structural or aerial-rigging review",
            },
            "kyushu_tinsley_seiu_kusozu_2017": {
                "open-access but unlicensed",
                "direct kinbaku lineage",
                "public access is not derivative",
            },
        }
        for candidate_id, required in deferred_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

        rejected_markers = {
            "commons_adriannespring_obsession_bondage_video": {
                "run computer vision",
                "ancient art",
                "human derivative or copyright review",
            },
            "wikibooks_rope_making_for_bondage_use": {
                "powered-drill",
                "open-flame singeing",
                "permissive rights do not override",
            },
            "internet_archive_calafou_shibari_workshop": {
                "public domain mark",
                "do not identify speakers",
                "not a license or dedication",
            },
            "edinburgh_psychotherapist_rope_self_case_thesis": {
                "single self-case",
                "improves therapist presence",
                "sensitive clinical components",
            },
        }
        for candidate_id, required in rejected_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "reject")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_031_decisions_and_queue_are_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_031"
        }
        expected = {
            "accept_targeted_scope": {
                "cambridge_obryan_consent_sexual_media_2024",
                "apj_yokohama_araki_authority_work_metadata",
            },
            "defer": {
                "hood_consent_complicating_agency_araki_2019",
                "umich_yi_araki_crossing_boundaries_interview_2011",
                "pussy_palace_ferguson_rope_oral_history",
                "staedel_araki_untitled_kinbaku_work_record",
                "kunsthalle_basel_mcclodden_kinbaku_paintings_2023",
                "petite_pretzel_tips_smaller_riggers",
                "kinkrx_understanding_tk_components",
                "shibari_dojo_munich_level_framework",
            },
            "reject": {
                "qagoma_araki_visual_consent_essay",
                "galleri_riis_araki_kinbaku_2008",
            },
        }
        self.assertEqual(len(batch), 12)
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

    def test_batch_031_accepts_mirror_queue_and_block_provenance_trivia(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        markers = {
            "cambridge_obryan_consent_sexual_media_2024": {
                "cc by 4.0",
                "jurisdiction-specific law",
                "third-party case",
                "memorization questions",
            },
            "apj_yokohama_araki_authority_work_metadata": {
                "government standard terms 2.0",
                "cc by-nc-nd",
                "silver dye breach print",
                "inventory numbers",
                "never qa targets",
            },
        }
        for candidate_id, required in markers.items():
            source = by_id[candidate_id]
            queued = queue_by_id[candidate_id]
            self.assertEqual(source["decision"], "accept_targeted_scope")
            self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
            self.assertEqual(queued["training_use"], source["training_use"])
            self.assertEqual(queued["rights_basis"], source["rights_basis"])
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_031_permission_privacy_and_visual_consent_are_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        deferred_markers = {
            "hood_consent_complicating_agency_araki_2019": {
                "allegations as allegations",
                "appearance, expression, rope looseness",
                "permission",
            },
            "umich_yi_araki_crossing_boundaries_interview_2011": {
                "self-presentation",
                "sexual access",
                "consent, technique or safety authority",
            },
            "pussy_palace_ferguson_rope_oral_history": {
                "citation permission",
                "transcript transformation",
                "biography",
            },
            "staedel_araki_untitled_kinbaku_work_record": {
                "all rights reserved",
                "traditional japanese",
                "permission",
            },
            "kunsthalle_basel_mcclodden_kinbaku_paintings_2023": {
                "progressive breath",
                "reenact",
                "permission",
            },
            "petite_pretzel_tips_smaller_riggers": {
                "martial-arts transfer",
                "text-based instructional material",
                "weight limits",
            },
            "kinkrx_understanding_tk_components": {
                "blueprint image",
                "text-complete",
                "suspendability",
            },
            "shibari_dojo_munich_level_framework": {
                "under three minutes",
                "nonverbal connection",
                "explicit prior agreement",
            },
        }
        for candidate_id, required in deferred_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

        rejected_markers = {
            "qagoma_araki_visual_consent_essay": {
                "loose-looking bindings",
                "continued distribution",
                "artist's publicity statement",
            },
            "galleri_riis_araki_kinbaku_2008": {
                "lower-body knot placement",
                "reconstruct ties",
                "traditional japanese technique",
            },
        }
        for candidate_id, required in rejected_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "reject")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_batch_032_decisions_and_queue_are_deterministic(self) -> None:
        batch = {
            row["candidate_id"]: row
            for row in self.candidates
            if row["review_batch"] == "discovery_batch_032"
        }
        expected = {
            "accept_targeted_scope": {
                "ndl_practitioner_publication_metadata_set",
            },
            "defer": {
                "rope_collective_code_of_ethics",
                "plura_unbounded_neurodivergent_rope_lab",
                "oblige_gote_accessibility_workshop_outline",
                "jcbp_consent_based_performance_archive",
                "escholarship_pennington_consent_work_2024",
                "ucr_sanford_running_ends_rope_performance",
                "daruma_berlin_level_prerequisite_framework",
                "graydancer_ropecast_oral_history_archive",
                "luke_george_rope_performance_archive",
                "dalhousie_rae_emancipate_me_harder_2025",
                "koshenkova_restraint_release_glass_shibari",
            },
            "reject": {
                "natalie_rose_adaptive_rope_class_syllabi",
                "loc_rope_and_twine_information_1917",
                "bondash_beginner_rope_safety_site",
                "fetish_com_japanese_bondage_history",
                "that_rope_place_singapore_site",
                "aasect_lexa_grace_rope_ce_listing_2026",
                "newcastle_shibari_curriculum_and_consent_listing",
                "laneway_fundamental_rope_bondage_class",
            },
        }
        self.assertEqual(len(batch), 20)
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

    def test_batch_032_ndl_accept_mirrors_queue_and_blocks_record_trivia(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        candidate_id = "ndl_practitioner_publication_metadata_set"
        source = by_id[candidate_id]
        queued = queue_by_id[candidate_id]
        self.assertEqual(source["decision"], "accept_targeted_scope")
        self.assertEqual(queued["scope"], source["recommended_crawl_scope"])
        self.assertEqual(queued["training_use"], source["training_use"])
        self.assertEqual(queued["rights_basis"], source["rights_basis"])
        text = json.dumps(source, sort_keys=True).lower()
        for marker in {
            "junko takahashi",
            "photography responsibility",
            "isbn",
            "must never become memorization questions",
            "no restriction on usage or purpose",
        }:
            self.assertIn(marker, text)

    def test_batch_032_permission_access_privacy_and_safety_are_quarantined(self) -> None:
        by_id = {row["candidate_id"]: row for row in self.candidates}
        queued = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        deferred_markers = {
            "rope_collective_code_of_ethics": {
                "not available for redistribution or reuse",
                "written commercial derivative model-training permission",
            },
            "plura_unbounded_neurodivergent_rope_lab": {
                "temporary ramp",
                "blanket high-strength-upline requirement",
                "platform rightsholder",
            },
            "oblige_gote_accessibility_workshop_outline": {
                "fit any body safely",
                "do not infer construction",
                "complete text-based instructional materials",
            },
            "jcbp_consent_based_performance_archive": {
                "http 403",
                "cc by-nc 4.0",
                "do not bypass access",
            },
            "escholarship_pennington_consent_work_2024": {
                "cc by-nc-nd 4.0",
                "participant pseudonyms",
                "framework is universal",
            },
            "ucr_sanford_running_ends_rope_performance": {
                "black, queer, trans",
                "generalization to black, queer, trans or nonbinary people",
                "permission",
            },
            "daruma_berlin_level_prerequisite_framework": {
                "floorwork is a complete practice",
                "internal level names",
                "permission",
            },
            "graydancer_ropecast_oral_history_archive": {
                "http 403",
                "do not bypass libsyn",
                "creative commons music",
            },
            "luke_george_rope_performance_archive": {
                "audience-participation instructions",
                "multiple artists",
                "permission",
            },
            "dalhousie_rae_emancipate_me_harder_2025": {
                "six-canadian-professional-practitioner",
                "therapeutic",
                "consent cannot be withdrawn immediately",
            },
            "koshenkova_restraint_release_glass_shibari": {
                "japanese versus european",
                "permission",
                "image",
            },
        }
        for candidate_id, required in deferred_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "defer")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

        rejected_markers = {
            "natalie_rose_adaptive_rope_class_syllabi": {
                "breath restriction",
                "structurally sound for every body",
                "permission alone",
            },
            "loc_rope_and_twine_information_1917": {
                "public domain",
                "obsolete",
                "brand",
            },
            "bondash_beginner_rope_safety_site": {
                "thirty minutes",
                "two minutes",
                "two fingers",
            },
            "fetish_com_japanese_bondage_history": {
                "spiritual journey",
                "unsuspecting",
                "four rules",
            },
            "that_rope_place_singapore_site": {
                "commerce-heavy",
                "learning-pathway",
                "product",
            },
            "aasect_lexa_grace_rope_ce_listing_2026": {
                "event title",
                "makes a therapist competent",
                "actual course",
            },
            "newcastle_shibari_curriculum_and_consent_listing": {
                "tk is a universal progression gate",
                "self-described lineage",
                "commerce",
            },
            "laneway_fundamental_rope_bondage_class": {
                "totally safe",
                "not necessarily professionally trained",
                "no-storage",
            },
        }
        for candidate_id, required in rejected_markers.items():
            source = by_id[candidate_id]
            self.assertEqual(source["decision"], "reject")
            self.assertNotIn(candidate_id, queued)
            text = json.dumps(source, sort_keys=True).lower()
            for marker in required:
                self.assertIn(marker, text, candidate_id)

    def test_report_covers_each_review_batch_and_decision(self) -> None:
        for batch_id in {row["review_batch"] for row in self.candidates}:
            self.assertIn(batch_id, self.report)
        for decision in set(self.discovery["decisions"]):
            self.assertIn(decision, self.report)
        self.assertIn("Latest batch: `discovery_batch_032`", self.report)


if __name__ == "__main__":
    unittest.main()
