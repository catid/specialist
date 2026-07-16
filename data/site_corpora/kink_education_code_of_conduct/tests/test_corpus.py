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
            self.manifest["resource_id"], "kink_education_code_of_conduct"
        )
        self.assertEqual(
            self.manifest["artifact_role"],
            "canonical_markdown_direct_training_source",
        )
        self.assertTrue(self.manifest["direct_training_ready"])
        self.assertTrue(self.manifest["non_qa"])
        self.assertEqual(self.manifest["training_document_count"], 1)
        self.assertEqual(self.manifest["direct_training_word_count"], 4070)
        self.assertEqual(self.manifest["whitespace_delimited_word_count"], 4129)

    def test_source_ledger_has_exact_five_allowlisted_records(self) -> None:
        expected = {
            "KECC-FULL": (
                "https://www.thekecc.org/fullVersion.html",
                "https://www.thekecc.org/fullVersion.html",
                38172,
                "903d1c07dfada79ceee507ad13cbd5d41aa7c8c722228b038122221be93ca2cc",
                "2026-07-16T10:42:15Z",
                "Fri, 21 Mar 2025 18:08:47 GMT",
            ),
            "KECC-PDF": (
                "https://www.thekecc.org/KECC.pdf",
                "https://www.thekecc.org/KECC.pdf",
                174888,
                "2429e5806ecf2427828676b9aaa8e95c5ac247a8406c6d3b4dd6e1d51b973657",
                "2026-07-16T10:42:15Z",
                "Thu, 25 Apr 2019 22:14:50 GMT",
            ),
            "KECC-HOME": (
                "https://www.thekecc.org/",
                "https://www.thekecc.org/",
                3945,
                "9e6e9b1f281b7042ce5a8b3e41f398ff82d4d03be319f8f3d10ef14ebb4287a8",
                "2026-07-16T10:42:15Z",
                "Fri, 21 Mar 2025 18:08:47 GMT",
            ),
            "CC-DEED": (
                "http://creativecommons.org/licenses/by-sa/4.0/",
                "https://creativecommons.org/licenses/by-sa/4.0/",
                35744,
                "17de6b7071e8f4816b103fb45aff75f71c94821f303534fc3e21cb68bd0f7148",
                "2026-07-16T10:42:29Z",
                "Tue, 09 Jun 2026 14:26:11 GMT",
            ),
            "CC-LEGAL": (
                "https://creativecommons.org/licenses/by-sa/4.0/legalcode.en",
                "https://creativecommons.org/licenses/by-sa/4.0/legalcode.en",
                51859,
                "a7dbad04e9a44a69a06d2ea5f20cceccb163091550591ed41ac610f112789246",
                "2026-07-16T10:42:35Z",
                "Tue, 09 Jun 2026 14:26:11 GMT",
            ),
        }
        self.assertEqual(len(self.sources), 5)
        self.assertEqual({source["source_id"] for source in self.sources}, set(expected))
        for source in self.sources:
            requested, resolved, size, digest, retrieved, modified = expected[
                source["source_id"]
            ]
            self.assertEqual(source["requested_url"], requested)
            self.assertEqual(source["resolved_url"], resolved)
            self.assertEqual(source["raw_byte_length"], size)
            self.assertEqual(source["raw_sha256"], digest)
            self.assertEqual(source["retrieved_at"], retrieved)
            self.assertEqual(source["last_modified"], modified)
            self.assertEqual(source["http_status"], 200)
            self.assertFalse(source["direct_training"])
            self.assertFalse(source["raw_retained"])

    def test_source_sizes_and_audit_metrics_sum_exactly(self) -> None:
        self.assertEqual(sum(source["raw_byte_length"] for source in self.sources), 304608)
        summary = self.provenance["retrieval_summary"]
        self.assertEqual(summary["source_record_count"], 5)
        self.assertEqual(summary["kecc_record_count"], 3)
        self.assertEqual(summary["license_metadata_record_count"], 2)
        self.assertEqual(summary["raw_byte_length_total"], 304608)
        self.assertEqual(summary["kecc_raw_byte_length_total"], 217005)
        self.assertEqual(summary["license_metadata_raw_byte_length_total"], 87603)
        self.assertFalse(summary["raw_sources_retained_in_repository"])
        metrics = {
            source["source_id"]: source for source in self.sources
        }
        self.assertEqual(metrics["KECC-FULL"]["normalized_audit_word_count"], 4978)
        self.assertEqual(metrics["KECC-HOME"]["normalized_audit_word_count"], 278)
        self.assertEqual(metrics["CC-DEED"]["normalized_audit_word_count"], 401)
        self.assertEqual(metrics["CC-LEGAL"]["normalized_audit_word_count"], 2766)

    def test_pdf_identity_metadata_and_complete_review_are_recorded(self) -> None:
        pdf = next(source for source in self.sources if source["source_id"] == "KECC-PDF")
        self.assertEqual(pdf["content_type"], "application/pdf")
        self.assertEqual(pdf["pdf_page_count"], 24)
        self.assertEqual(pdf["pdf_creation_date"], "2019-04-25T22:14:31Z")
        self.assertEqual(pdf["pdf_modification_date"], "2019-04-25T22:14:31Z")
        self.assertEqual(pdf["audit_text_extractor"], "pypdf 6.14.2")
        self.assertEqual(pdf["audit_text_byte_length"], 41547)
        self.assertEqual(pdf["audit_text_word_count"], 6197)
        self.assertEqual(
            pdf["audit_text_sha256"],
            "01422dcd358d1980015a34dad995066127f6222cd2f3a9b2d9fdc176268d1644",
        )
        self.assertEqual(self.provenance["content_audit"]["pdf"]["digital_pages"], 24)
        self.assertEqual(
            self.provenance["content_audit"]["pdf"]["raster_xobject_occurrences"], 0
        )

    def test_network_access_stayed_inside_allowlist(self) -> None:
        network = self.provenance["network_boundary"]
        self.assertEqual(len(network["allowlisted_requested_routes"]), 5)
        self.assertEqual(len(network["resolved_routes"]), 5)
        for key in (
            "kecc_short_versions_accessed",
            "kecc_faq_html_accessed",
            "kecc_adoption_html_accessed",
            "kecc_contact_html_accessed",
            "third_party_kink_pages_accessed",
            "font_routes_accessed",
            "script_or_stylesheet_routes_accessed",
            "image_routes_accessed",
            "other_routes_accessed",
        ):
            self.assertFalse(network[key])
        self.assertIn("PDF contains its own FAQ", network["note"])

    def test_raw_sources_audit_extractions_and_media_are_not_retained(self) -> None:
        files = sorted(
            str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file()
        )
        self.assertFalse(any(name.endswith((".html", ".htm", ".pdf")) for name in files))
        self.assertFalse(any("extract" in name.lower() for name in files))
        self.assertEqual(
            sorted(path.name for path in (ROOT / "source_snapshot").iterdir()),
            ["provenance.json"],
        )
        method = self.provenance["normalization_and_review_method"]
        self.assertFalse(method["raw_html_retained"])
        self.assertFalse(method["raw_pdf_retained"])
        self.assertFalse(method["audit_extractions_retained"])
        self.assertFalse(method["raw_source_direct_training"])
        audit = self.provenance["content_audit"]
        self.assertEqual(audit["full_html"]["images"], 1)
        self.assertEqual(audit["homepage_html"]["images"], 1)
        self.assertEqual(len(audit["external_components_not_fetched"]), 5)

    def test_cc_by_sa_attribution_link_changes_and_sharealike_are_complete(self) -> None:
        rights = self.provenance["rights_review"]
        self.assertEqual(
            rights["license_name"],
            "Creative Commons Attribution-ShareAlike 4.0 International",
        )
        self.assertEqual(rights["license_identifier"], "CC BY-SA 4.0")
        self.assertEqual(rights["creator_credit"], "v. 1 KECC Collective")
        self.assertTrue(rights["adaptation_marked"])
        self.assertTrue(rights["changes_described"])
        self.assertTrue(rights["license_linked"])
        self.assertTrue(rights["creator_attributed"])
        self.assertFalse(rights["endorsement_implied"])
        self.assertFalse(rights["additional_restrictions_added"])
        self.assertFalse(rights["legal_opinion"])
        self.assertFalse(rights["license_sources_direct_training"])
        for phrase in (
            "created by the v. 1 KECC Collective",
            "25 April 2019",
            "Creative Commons Attribution-ShareAlike 4.0 International licence",
            "Changes made:",
            "No endorsement by the KECC Collective is implied",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(self.manifest["sharealike_status"], "cc_by_sa_4_0_adapted_output")

    def test_historical_v1_status_and_no_current_consensus_are_prominent(self) -> None:
        for phrase in (
            "first version, not a timeless or universal code",
            "expected practices and terminology to change",
            "not proof of present-day consensus",
            "The reviewed sources therefore provide a historically situated ethics proposal",
            "does not establish current community consensus",
        ):
            self.assertIn(phrase, self.corpus)
        status = self.provenance["version_status"]
        self.assertEqual(status["source_version"], "1")
        self.assertEqual(status["pdf_visible_date"], "2019-04-25")
        self.assertIn("must not present", status["corpus_rule"])
        self.assertEqual(
            self.manifest["version_status"],
            "version_1_pdf_dated_2019_historical_not_current_consensus",
        )

    def test_educator_producer_discussion_roles_remain_distinct(self) -> None:
        for phrase in (
            "Every main topic in the full version has three perspectives",
            "The **educator** part",
            "The **producer** part",
            "The **discussion** part",
            "These roles should not be collapsed",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["role_structure_status"],
            "educator_producer_discussion_roles_preserved",
        )

    def test_modeling_consent_retains_capacity_continuity_and_visible_context(self) -> None:
        for phrase in (
            "understand their own capacity and the capacity of others involved",
            "free from coercion, force, and manipulation",
            "Doubt or confusion is a stop signal",
            "Classroom touch carries no implied permission",
            "specific verbal agreement from each person before touching them",
            "avoid initiating a new negotiation in front of the classroom audience",
            "distinguishes *having* consent from making consent legible to students",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["consent_modeling_status"],
            "capacity_informed_explicit_continuous_noncoerced_and_visible",
        )

    def test_student_demo_power_and_safeguards_are_complete(self) -> None:
        for phrase in (
            "substantial power differential",
            "Social pressure can make refusal or withdrawal difficult",
            "Pressure to be entertaining",
            "use another educator or an existing play partner first",
            "in-class student volunteer only as a last resort",
            "do not call on a particular person",
            "If no one volunteers promptly, the demonstration is skipped",
            "unambiguous, enthusiastic agreement",
            "may stop for any reason at any time",
            "formal plan communicated to the venue and class",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["student_demo_status"],
            "power_differential_preference_order_anti_pressure_opt_out_and_care",
        )

    def test_competence_disclosure_and_risk_profile_respect_are_retained(self) -> None:
        for phrase in (
            "teach only material they understand thoroughly",
            "possess relevant professional qualifications",
            "had the material reviewed by a qualified professional",
            "are offering personal opinion",
            "distinguish confidence from competence",
            "does not pressure a student to exceed that student's risk profile",
            "Students may observe rather than perform",
            "expects diligence and honest limits from expert amateurs",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["competence_status"],
            "topic_specific_competence_qualification_review_or_opinion_disclosure",
        )

    def test_inclusion_accessibility_and_event_response_are_retained(self) -> None:
        for phrase in (
            "use a person's stated pronouns",
            "across genders, body types, and physical abilities",
            "support different learning needs",
            "does not depend on hostile intent",
            "seek diversity among staff and educators",
            "a formal route for reporting consent incidents",
            "at least one staff member trained to respond",
            "A published policy or trained staff role does not prove accessibility",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["inclusion_status"],
            "nondiscrimination_accessible_learning_diverse_governance_and_response",
        )

    def test_accountability_contacts_circles_and_limitations_are_retained(self) -> None:
        for phrase in (
            "feedback, acknowledge a mistake, apologize, repair what can be repaired, learn, and change patterns",
            "an **accountability contact** is a trusted intake person",
            "An **accountability circle** is a group",
            "At least one contact should be independent",
            "Multiple contacts let a reporter avoid someone they do not trust",
            "One designated repository can help identify patterns",
            "do not themselves establish investigator competence",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["accountability_status"],
            "contacts_circles_independence_access_pattern_memory_and_non_enforcement_limits",
        )

    def test_educator_student_and_mentoring_boundaries_are_retained(self) -> None:
        for phrase in (
            "professional demeanor in public classes and private lessons",
            "do not cruise or engage in recreational play",
            "formal policy for sex, play, or romance with students",
            "explicit discussion of the educational power differential",
            "Mentoring begins with negotiated boundaries",
            "blanket prohibition would be preferable in most cases",
            "exchanging professional opportunity for sex or play",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["educator_student_boundary_status"],
            "formal_policy_power_discussion_mentoring_boundaries_and_no_status_trade",
        )

    def test_professional_conduct_and_bidirectional_vetting_are_retained(self) -> None:
        for phrase in (
            "complete, accurate answers during booking",
            "use a real vetting process",
            "responsibilities proportionate to developing expertise",
            "listen respectfully to community reports",
            "A denial, reference, credential, or completed questionnaire does not prove absence of risk",
            "References supplied by an applicant are not sufficient on their own",
            "Vetting runs in both directions",
            "booking an educator gives that person status, access, and an apparent institutional endorsement",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["vetting_status"],
            "bidirectional_multi_source_contextual_and_not_safety_clearance",
        )

    def test_injury_consent_incident_and_privacy_tensions_are_retained(self) -> None:
        for phrase in (
            "consent incidents they caused or were accused of",
            "injuries they caused or were accused of",
            "protect the privacy of people who may have been harmed",
            "do not pressure people to remain silent",
            "respect private forums",
            "privacy and disclosure as a genuine conflict",
            "greater weight to the privacy of people who may have been harmed",
            "proportionate, non-punitive, respectful",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["privacy_status"],
            "incident_disclosure_non_retaliation_need_to_know_and_unresolved_balancing",
        )

    def test_no_legal_certification_enforcement_or_safety_overclaim(self) -> None:
        for phrase in (
            "rather than a precise legal code",
            "did not claim to monitor adopters or centrally enforce compliance",
            "did not mean perfection, membership in the Collective, professional certification",
            "does not supply a central tribunal, investigation standard, sanction table",
            "not legal advice",
            "does not establish current community consensus, legal requirements, professional licensure",
            "does not prove that an educator is competent because they adopted it",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["authority_status"],
            "no_current_consensus_law_certification_enforcement_or_safety_proof",
        )

    def test_operational_techniques_and_portable_time_thresholds_are_absent(self) -> None:
        lower = self.corpus.lower()
        for marker in (
            "breath play",
            "blood play",
            "needle play",
            "hypnosis",
            "how to tie",
            "step-by-step",
            "24 hours",
            "one month waiting",
            "kink producer network group",
            "contact@thekecc.org",
            "sui generis database",
        ):
            self.assertNotIn(marker, lower)
        self.assertEqual(
            self.manifest["operational_detail_status"],
            "excluded_no_kink_demo_medical_legal_investigation_or_aftercare_procedure",
        )

    def test_urls_exist_only_for_required_source_and_license_attribution(self) -> None:
        urls = re.findall(r"https?://[^)\s]+", self.corpus)
        self.assertEqual(
            urls,
            [
                "https://www.thekecc.org/fullVersion.html",
                "https://www.thekecc.org/KECC.pdf",
                "https://creativecommons.org/licenses/by-sa/4.0/",
            ],
        )
        self.assertIn("URLs as factual trivia", self.corpus)
        self.assertIn("Do not generate recall questions", self.corpus)

    def test_claim_level_citations_are_valid_and_numerous(self) -> None:
        citations = re.findall(
            r"\[((?:KECC-(?:HOME|FULL|PDF)|CC-DEED)): ([^\]]+)\]", self.corpus
        )
        self.assertEqual(len(citations), 58)
        valid_ids = {source["source_id"] for source in self.sources}
        self.assertTrue(all(source_id in valid_ids for source_id, _ in citations))
        self.assertEqual(self.manifest["claim_level_citation_count"], 58)
        policy = self.provenance["citation_policy"]
        self.assertTrue(policy["claim_level_required"])
        self.assertEqual(policy["source_id_mapping_file"], "sources.jsonl")
        self.assertEqual(policy["license_citation_role"], "metadata only")

    def test_dispositions_are_complete_and_sequential(self) -> None:
        self.assertEqual(len(self.dispositions), 50)
        self.assertEqual(
            [row["record_id"] for row in self.dispositions],
            [f"kecc-v1-{n:03d}" for n in range(1, 51)],
        )
        self.assertEqual(
            sum(row["direct_training_included"] for row in self.dispositions), 38
        )
        self.assertEqual(
            sum(not row["direct_training_included"] for row in self.dispositions), 12
        )
        self.assertEqual(self.manifest["disposition_record_count"], 50)
        self.assertEqual(self.manifest["included_or_narrowed_disposition_count"], 38)
        self.assertEqual(self.manifest["excluded_disposition_count"], 12)

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
            "individual member",
            "URL",
            "page title",
            "section order",
            "date",
            "hash",
            "licence boilerplate",
            "named outside organization",
            "practice example",
            "current consensus",
            "certification",
            "enforcement",
            "safety proof",
        ):
            self.assertIn(forbidden, policy["forbidden"])
        self.assertTrue(policy["inherit_document_split"])

    def test_split_hygiene_and_protected_boundary(self) -> None:
        split = self.provenance["split_hygiene"]
        self.assertEqual(split["source_document_unit"], "KECC-v1-2019")
        self.assertFalse(split["protected_data_accessed_during_build"])
        self.assertEqual(len(split["protected_data_classes_not_accessed"]), 11)
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
            re.findall(
                r"\[(?:KECC-(?:HOME|FULL|PDF)|CC-DEED): [^\]]+\]", self.corpus
            )
        )
        self.assertEqual(regex_count, 4070)
        self.assertEqual(whitespace_count, 4129)
        self.assertEqual(h2_count, 14)
        self.assertEqual(h3_count, 22)
        self.assertEqual(citation_count, 58)
        self.assertEqual(self.manifest["direct_training_word_count"], regex_count)
        self.assertEqual(
            self.manifest["whitespace_delimited_word_count"], whitespace_count
        )
        self.assertEqual(self.manifest["h2_section_count"], h2_count)
        self.assertEqual(self.manifest["h3_subsection_count"], h3_count)
        self.assertIn("4,129-whitespace-word", self.report)
        self.assertIn("50 manual disposition records", self.report)
        self.assertIn("58 claim-level source citations", self.report)


if __name__ == "__main__":
    unittest.main()
