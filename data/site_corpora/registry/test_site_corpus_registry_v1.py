#!/usr/bin/env python3
"""Offline contract tests for the first-class Markdown corpus registry."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "site_corpus_registry_v1.json"
CONFIG = HERE / "source_registry_config_v1.json"
BUILDER = HERE / "build_registry_v1.py"
TOKENIZER = ROOT / "models/Qwen3.6-35B-A3B/tokenizer.json"
TOKENIZER_CONFIG = ROOT / "models/Qwen3.6-35B-A3B/tokenizer_config.json"
WORD_RE = re.compile(r"[\w\u2019'-]+", flags=re.UNICODE)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class SiteCorpusRegistryV1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.artifacts = cls.registry["artifacts"]
        cls.by_id = {item["resource_id"]: item for item in cls.artifacts}

    def test_registry_is_exactly_reproducible_from_tracked_corpora(self) -> None:
        venv_python = ROOT / ".venv/bin/python"
        python = venv_python if venv_python.exists() else Path(sys.executable)
        result = subprocess.run(
            [str(python), str(BUILDER), "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_markdown_is_a_first_class_unmerged_layer(self) -> None:
        self.assertEqual(self.registry["schema"], "site-corpus-registry-v1")
        self.assertEqual(self.registry["schema_version"], 1)
        layer = self.registry["dataset_layer"]
        self.assertEqual(layer["layer_id"], "canonical_markdown_source_corpus")
        self.assertTrue(layer["first_class_training_layer"])
        self.assertTrue(layer["non_qa"])
        self.assertFalse(layer["merge_performed"])
        self.assertFalse(layer["chunking_performed"])
        self.assertFalse(layer["split_assignment_performed"])
        self.assertTrue(layer["omission_of_registered_artifact_is_fatal"])
        self.assertTrue(layer["source_caps_required_before_snapshot"])
        self.assertTrue(layer["token_volume_alone_must_not_determine_source_weight"])

    def test_every_record_has_one_real_hash_bound_markdown_and_manifest(self) -> None:
        self.assertGreaterEqual(len(self.artifacts), 9)
        self.assertEqual(len(self.by_id), len(self.artifacts))
        markdown_paths = [item["markdown_path"] for item in self.artifacts]
        manifest_paths = [item["manifest_path"] for item in self.artifacts]
        self.assertEqual(len(markdown_paths), len(set(markdown_paths)))
        self.assertEqual(len(manifest_paths), len(set(manifest_paths)))

        for item in self.artifacts:
            markdown_path = ROOT / item["markdown_path"]
            manifest_path = ROOT / item["manifest_path"]
            markdown_bytes = markdown_path.read_bytes()
            manifest_bytes = manifest_path.read_bytes()
            self.assertEqual(sha256_bytes(markdown_bytes), item["markdown_sha256"])
            self.assertEqual(sha256_bytes(manifest_bytes), item["manifest_sha256"])
            self.assertEqual(len(markdown_bytes), item["byte_length"])
            self.assertTrue(item["declared_direct_training_ready"])
            self.assertTrue(item["non_qa"])
            self.assertEqual(item["dataset_layer"], "canonical_markdown_source_corpus")
            self.assertTrue(item["manifest_schema"]["name"])

    def test_word_counts_are_complete_artifact_counts(self) -> None:
        for item in self.artifacts:
            text = (ROOT / item["markdown_path"]).read_text(encoding="utf-8")
            self.assertEqual(len(WORD_RE.findall(text)), item["unicode_word_count"])
            self.assertEqual(
                len(text.split()), item["whitespace_delimited_word_count"]
            )

    def test_qwen36_tokenizer_identity_and_counts_are_sealed(self) -> None:
        tokenizer = self.registry["tokenizer"]
        self.assertEqual(tokenizer["model_family"], "Qwen3.6-35B-A3B")
        self.assertFalse(tokenizer["add_special_tokens"])
        self.assertEqual(
            sha256_bytes(TOKENIZER.read_bytes()), tokenizer["tokenizer_json_sha256"]
        )
        self.assertEqual(
            sha256_bytes(TOKENIZER_CONFIG.read_bytes()),
            tokenizer["tokenizer_config_sha256"],
        )
        for item in self.artifacts:
            self.assertGreater(item["qwen36_token_count"], 0)

    def test_totals_equal_record_sums(self) -> None:
        totals = self.registry["totals"]
        self.assertEqual(totals["artifact_count"], len(self.artifacts))
        self.assertEqual(
            totals["source_document_split_group_count"], len(self.artifacts)
        )
        for field in {
            "byte_length",
            "unicode_word_count",
            "whitespace_delimited_word_count",
            "qwen36_token_count",
        }:
            self.assertEqual(totals[field], sum(item[field] for item in self.artifacts))

    def test_source_identity_deterministically_binds_the_split_group(self) -> None:
        group_ids = []
        for item in self.artifacts:
            identity_sha = sha256_bytes(
                canonical_json_bytes(item["source_document_identity"])
            )
            self.assertEqual(identity_sha, item["source_document_identity_sha256"])
            split = item["required_single_document_split_group"]
            self.assertEqual(split["group_id"], f"source-document-v1:{identity_sha}")
            self.assertTrue(split["required"])
            self.assertTrue(split["cross_split_reuse_forbidden"])
            self.assertEqual(split["assignment_unit"], "source_document")
            self.assertEqual(
                split["assign_before"], "markdown_chunking_or_qa_derivation"
            )
            self.assertIn("every chunk", split["members"])
            self.assertIn("every derived QA", split["members"])
            group_ids.append(split["group_id"])
        self.assertEqual(len(group_ids), len(set(group_ids)))

    def test_rights_basis_is_explicit_without_turning_robots_into_a_license(self) -> None:
        allowed_statuses = {
            "explicit_open_license",
            "federal_text_public_domain_presumption",
            "legacy_manifest_gap",
        }
        for item in self.artifacts:
            rights = item["rights_basis"]
            self.assertIn(rights["status"], allowed_statuses)
            self.assertTrue(rights["basis"])
            self.assertTrue(rights["license"])
            self.assertIs(type(rights["attribution_required"]), bool)
            self.assertTrue(rights["promotion_gate"])
            self.assertTrue(rights["limitations"])
            rights_text = json.dumps(rights).lower()
            self.assertNotIn("robots is a copyright license", rights_text)

        legacy = {
            item["resource_id"]
            for item in self.artifacts
            if item["rights_basis"]["status"] == "legacy_manifest_gap"
        }
        self.assertEqual(
            legacy, {"crash_restraint", "rope365", "rope_topia", "shibari_atlas"}
        )
        for resource_id in legacy:
            rights = self.by_id[resource_id]["rights_basis"]
            self.assertEqual(rights["license"], "not_recorded")
            self.assertEqual(
                rights["promotion_gate"], "rights_review_required_before_new_snapshot"
            )

    def test_open_article_and_federal_source_identities_are_exact(self) -> None:
        expected_pmcids = {
            "europepmc_rope_neuropathy_study": "PMC10294117",
            "europepmc_icar_suspension_syndrome": "PMC10710713",
            "europepmc_bdsm_fatality_review": "PMC8813685",
            "europepmc_entrapment_neuropathy_review": "PMC7382548",
        }
        for resource_id, pmcid in expected_pmcids.items():
            item = self.by_id[resource_id]
            self.assertEqual(item["source_document_identity"]["pmcid"], pmcid)
            self.assertEqual(item["rights_basis"]["status"], "explicit_open_license")

        federal = self.by_id["usfs_rigging_for_trail_work"]
        self.assertEqual(
            federal["source_document_identity"]["govinfo_package_id"],
            "GOVPUB-A13-PURL-gpo235248",
        )
        self.assertEqual(
            federal["rights_basis"]["status"],
            "federal_text_public_domain_presumption",
        )

        hse = self.by_id["hse_treework_lifting_and_climbing"]
        self.assertEqual(
            hse["source_document_identity"]["canonical_url"],
            "https://www.hse.gov.uk/treework/safety-topics/arboriculture.htm",
        )
        self.assertEqual(hse["rights_basis"]["status"], "explicit_open_license")
        self.assertEqual(
            hse["rights_basis"]["license"], "Open Government Licence v3.0"
        )

        kecc = self.by_id["kink_education_code_of_conduct"]
        self.assertEqual(
            kecc["source_document_identity"]["source_id"], "KECC-v1-2019"
        )
        self.assertEqual(kecc["source_document_identity"]["version"], "1")
        self.assertEqual(kecc["rights_basis"]["status"], "explicit_open_license")
        self.assertEqual(kecc["rights_basis"]["license"], "CC BY-SA 4.0")
        self.assertIn(
            "no_current_consensus_law_certification_enforcement_or_safety_proof",
            kecc["safety_transfer_flags"],
        )

        hitch = self.by_id["mdpi_tree_climbing_friction_hitch_2021"]
        self.assertEqual(
            hitch["source_document_identity"]["doi"], "10.3390/f12111457"
        )
        self.assertEqual(hitch["rights_basis"]["status"], "explicit_open_license")
        self.assertEqual(hitch["rights_basis"]["license"], "CC BY 4.0")
        self.assertIn(
            "no_natural_fiber_bondage_shibari_kinbaku_or_human_suspension_transfer",
            hitch["safety_transfer_flags"],
        )

        loop_knots = self.by_id["acta_loop_knot_efficiency_experiments_2020"]
        self.assertEqual(
            loop_knots["source_document_identity"]["doi"],
            "10.12693/APhysPolA.138.404",
        )
        self.assertEqual(
            loop_knots["rights_basis"]["status"], "explicit_open_license"
        )
        self.assertEqual(loop_knots["rights_basis"]["license"], "CC BY 4.0")
        self.assertIn(
            "no_portable_efficiency_ranking_rating_safe_load_or_knot_selection_rule",
            loop_knots["safety_transfer_flags"],
        )
        self.assertIn(
            "no_natural_fiber_bondage_body_contact_upline_or_human_suspension_transfer",
            loop_knots["safety_transfer_flags"],
        )

        aramid = self.by_id["nist_aramid_rope_sling_fatigue_1976"]
        self.assertEqual(
            aramid["source_document_identity"]["doi"],
            "10.6028/NBS.IR.76-1159",
        )
        self.assertEqual(
            aramid["source_document_identity"]["report_number"],
            "NBSIR 76-1159",
        )
        self.assertEqual(
            aramid["rights_basis"]["status"],
            "federal_text_public_domain_presumption",
        )
        self.assertIn(
            "end_fitting_inadequacy_and_censored_measurements_must_be_front_loaded",
            aramid["safety_transfer_flags"],
        )
        self.assertIn(
            "prototype_observations_are_not_fully_substantiated_rope_body_conclusions",
            aramid["safety_transfer_flags"],
        )

    def test_safety_and_domain_transfer_flags_cannot_be_empty(self) -> None:
        for item in self.artifacts:
            flags = item["safety_transfer_flags"]
            self.assertGreaterEqual(len(flags), 3)
            self.assertEqual(flags, sorted(set(flags)))
        self.assertIn(
            "no_safe_duration_pressure_or_placement_inference",
            self.by_id["europepmc_rope_neuropathy_study"][
                "safety_transfer_flags"
            ],
        )
        self.assertIn(
            "not_validated_as_bondage_suspension_protocol",
            self.by_id["europepmc_icar_suspension_syndrome"][
                "safety_transfer_flags"
            ],
        )
        self.assertIn(
            "no_bondage_rope_tree_limb_indoor_hardpoint_or_hardware_certification",
            self.by_id["usfs_rigging_for_trail_work"]["safety_transfer_flags"],
        )
        self.assertIn(
            "no_tree_branch_anchor_hardpoint_bondage_or_human_suspension_certification",
            self.by_id["hse_treework_lifting_and_climbing"][
                "safety_transfer_flags"
            ],
        )

    def test_policy_exclusion_notices_are_not_training_artifacts(self) -> None:
        excluded = {
            item["resource_id"]: item["reason"]
            for item in self.registry["excluded_nontraining_manifests"]
        }
        self.assertEqual(
            set(excluded), {"shibari_study", "theduchy", "tree_anchor_mechanics"}
        )
        self.assertIn("prohibits", excluded["shibari_study"])
        self.assertIn("disallowed", excluded["theduchy"])
        self.assertIn("permission", excluded["tree_anchor_mechanics"])
        for resource_id in excluded:
            self.assertNotIn(resource_id, self.by_id)

    def test_config_and_source_tree_fingerprints_are_bound(self) -> None:
        self.assertEqual(
            sha256_bytes(CONFIG.read_bytes()), self.registry["config"]["sha256"]
        )
        rows = []
        for item in self.artifacts:
            rows.extend(
                [
                    {"path": item["manifest_path"], "sha256": item["manifest_sha256"]},
                    {"path": item["markdown_path"], "sha256": item["markdown_sha256"]},
                ]
            )
        fingerprint = sha256_bytes(
            canonical_json_bytes(sorted(rows, key=lambda row: row["path"]))
        )
        self.assertEqual(fingerprint, self.registry["source_tree_fingerprint_sha256"])

    def test_root_readme_routes_snapshot_builders_through_registry(self) -> None:
        readme = (ROOT / "data/site_corpora/README.md").read_text(encoding="utf-8")
        for marker in {
            "registry/site_corpus_registry_v1.json",
            "omission",
            "split group",
            "rights",
            "Qwen3.6-35B-A3B",
        }:
            self.assertIn(marker, readme)


if __name__ == "__main__":
    unittest.main()
