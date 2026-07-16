import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    for line_number, line in enumerate(
        (ROOT / path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise AssertionError(f"{path}:{line_number}: {error}") from error
    return rows


def sha256(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


class HseTemporaryWorksCorpusTests(unittest.TestCase):
    def test_manifest_schema_readiness_and_exact_artifact_set(self):
        manifest = read_json("manifest.json")
        self.assertEqual(manifest["schema"], "site-corpus-manifest")
        self.assertEqual(manifest["schema_version"], "1.0")
        self.assertEqual(
            manifest["corpus_schema"],
            "non-bondage-uk-construction-governance-markdown",
        )
        self.assertEqual(manifest["corpus_schema_version"], "1.0")
        self.assertEqual(manifest["package_id"], "hse_temporary_works_faqs")
        self.assertIs(manifest["direct_training_ready"], True)
        self.assertEqual(manifest["training_artifacts"], ["CORPUS.md"])
        self.assertNotIn("CORPUS.md", manifest["non_training_audit_artifacts"])

        expected = {
            "CORPUS.md",
            "README.md",
            "REPORT.md",
            "components.jsonl",
            "manifest.json",
            "source_snapshot/provenance.json",
            "sources.jsonl",
            "surfaces.jsonl",
            "tests/test_corpus.py",
        }
        actual = {
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        self.assertEqual(actual, expected)
        forbidden_suffixes = {".html", ".htm", ".txt", ".svg", ".jpg", ".png", ".pdf"}
        self.assertFalse(any(path.suffix.lower() in forbidden_suffixes for path in ROOT.rglob("*")))

    def test_artifact_hashes_are_complete_and_exact(self):
        manifest = read_json("manifest.json")
        expected_paths = {
            "CORPUS.md",
            "README.md",
            "REPORT.md",
            "components.jsonl",
            "source_snapshot/provenance.json",
            "sources.jsonl",
            "surfaces.jsonl",
            "tests/test_corpus.py",
        }
        artifacts = manifest["artifacts"]
        self.assertEqual({item["path"] for item in artifacts}, expected_paths)
        for item in artifacts:
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(item["sha256"], sha256(item["path"]))

    def test_exact_five_body_route_identity_and_checksums(self):
        sources = read_jsonl("sources.jsonl")
        self.assertEqual(len(sources), 5)
        by_id = {row["source_id"]: row for row in sources}
        self.assertEqual(
            set(by_id),
            {
                "HSE-TW-FAQ-2026",
                "HSE-COPYRIGHT-OGL",
                "HSE-ROBOTS-2026",
                "TNA-OGL-V3",
                "TNA-ROBOTS-2026",
            },
        )
        expected = {
            "HSE-TW-FAQ-2026": (
                "https://www.hse.gov.uk/construction/faq-temporary-works.htm",
                42358,
                "449ac06366ec4bff513c6a6a2ec2ebe023e99cf5c70b2f6e6ff598c7e7861b1e",
                "8c9f46f9cd43473275585893a01fb923b38fedd564c2d196e472ca74ef46f4443be66c72ad0a9565a5c5dec102de1c7964137b81ee72fe45434ae227f9144e04",
            ),
            "HSE-COPYRIGHT-OGL": (
                "https://www.hse.gov.uk/help/copyright.htm",
                39349,
                "9b2c40591b2d83a890f27b8962397b8dbd78b62cc54dac6e29f805b944b9391d",
                "7e61ff604fa45429068cbea7d520a744246f987f012e5e84c34134e73a9ecb93d0c1377156fa18ce328b29d50b015076e44ba546eb129ec9693b36275b7d19b7",
            ),
            "HSE-ROBOTS-2026": (
                "https://www.hse.gov.uk/robots.txt",
                975,
                "1ee4f6b052e21d81c354359caddf6daf334aa7dddb8b01e39a1e71e834f43fdd",
                "74f254aa74c4c1fff61d4f9b95444812119d2a0fe26fd101bc2fe44c4f744efae4a671618681242418ac47e5cb4d38711124867605b7b7aeb02e50884385ded4",
            ),
            "TNA-OGL-V3": (
                "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
                10450,
                "f5b2b9f2af63647cde889fa6c3508f5705925295b74912c68caa28dd37e64aa5",
                "640c599709cb045b7c7ae7bda6798dac5bc9d1c826a2e85fc1ac4ab5683b68f705937d1665f2a4ebb8b12b9186e57ffc276b846fa9484f8b0e69b498d416f3e2",
            ),
            "TNA-ROBOTS-2026": (
                "https://www.nationalarchives.gov.uk/robots.txt",
                1322,
                "62d23c63718241a6b062af3fc717293e9597cc017fb0f30708b49d9c23f735f1",
                "6cf3bbc92238ccb72ffef3fb2a569e8b364e2f2848c5bc2e877d5880990fe8e275b872f6f9f78d2cb33eb6f03133bbe94b2b8cd99fae293a4e18f0c9681fe9aa",
            ),
        }
        for source_id, (url, size, digest256, digest512) in expected.items():
            row = by_id[source_id]
            self.assertEqual(row["source_url"], url)
            self.assertEqual(row["http_status"], 200)
            self.assertEqual(row["body_bytes"], size)
            self.assertEqual(row["body_sha256"], digest256)
            self.assertEqual(row["body_sha512"], digest512)

        faq = by_id["HSE-TW-FAQ-2026"]
        self.assertEqual(faq["displayed_update_date"], "2026-03-11")
        self.assertEqual(faq["license_spdx"], "OGL-UK-3.0")
        self.assertIs(faq["direct_training_ready"], True)
        self.assertTrue(all(not row["direct_training_ready"] for key, row in by_id.items() if key != "HSE-TW-FAQ-2026"))

    def test_retrieval_boundary_displayed_date_and_rights(self):
        provenance = read_json("source_snapshot/provenance.json")
        policy = provenance["retrieval_policy"]
        self.assertEqual(policy["allowed_bodies_retrieved"], 5)
        self.assertEqual(len(policy["allowed_urls"]), 5)
        self.assertEqual(policy["technical_source_bodies"], 1)
        self.assertEqual(policy["rights_and_access_provenance_bodies"], 4)
        self.assertIs(policy["site_crawl_performed"], False)
        self.assertIs(policy["mirrors_or_archives_used"], False)
        self.assertIs(policy["search_snippets_used"], False)
        self.assertEqual(policy["linked_technical_bodies_retrieved"], 0)
        self.assertEqual(policy["linked_rights_bodies_retrieved"], 1)

        dates = provenance["displayed_date_resolution"]
        self.assertEqual(dates["displayed_update_date"], "2026-03-11")
        self.assertEqual(dates["displayed_human_date"], "11 March 2026")
        self.assertEqual(dates["http_last_modified"], "2026-06-24T18:40:57Z")
        self.assertIn("Preserve", dates["decision"])

        rights = provenance["rights"]
        self.assertEqual(rights["license_spdx"], "OGL-UK-3.0")
        self.assertIs(rights["hse_robots"]["faq_route_disallowed"], False)
        self.assertIs(rights["hse_robots"]["copyright_route_disallowed"], False)
        self.assertIs(rights["hse_robots"]["ai_content_signal_present"], False)
        self.assertEqual(
            rights["national_archives_robots"]["content_signal"],
            {"ai-train": "no", "search": "yes", "ai-input": "no"},
        )
        self.assertIn("No National Archives licence-page wording", rights["national_archives_robots"]["enforcement"])

    def test_all_material_surfaces_have_manual_dispositions(self):
        rows = read_jsonl("surfaces.jsonl")
        self.assertEqual(len(rows), 49)
        counts = {}
        for row in rows:
            counts[row["source_id"]] = counts.get(row["source_id"], 0) + 1
            self.assertIn("manual", row["audit"])
            self.assertTrue(row["excluded"])
        self.assertEqual(
            counts,
            {
                "HSE-TW-FAQ-2026": 21,
                "HSE-COPYRIGHT-OGL": 12,
                "TNA-OGL-V3": 11,
                "HSE-ROBOTS-2026": 3,
                "TNA-ROBOTS-2026": 2,
            },
        )
        decisions = {name: sum(row["decision"] == name for row in rows) for name in {row["decision"] for row in rows}}
        self.assertEqual(
            decisions,
            {
                "exclude": 22,
                "partial": 11,
                "partial_rights_provenance_only": 13,
                "partial_access_provenance_only": 3,
            },
        )
        ids = {row["surface_id"] for row in rows}
        self.assertTrue(
            {
                "faq-08-duration-criticality",
                "faq-09-self-organisation",
                "faq-10-coordinator-law-standard",
                "faq-11-site-assumptions",
                "faq-12-proprietary-load-context",
                "faq-13-designer-competence",
                "faq-14-design-versus-coordination",
                "faq-20-displayed-update-date",
                "rights-04-online-crown-ogl",
                "tna-robots-01-content-signal",
            }.issubset(ids)
        )
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        cited_surface_ids = set(
            re.findall(r"\b(?:faq|rights)-\d{2}-[a-z0-9-]+\b", corpus)
        )
        self.assertTrue(cited_surface_ids)
        self.assertTrue(cited_surface_ids.issubset(ids))

    def test_components_reconcile_retained_scope_and_all_exclusions(self):
        rows = read_jsonl("components.jsonl")
        surface_ids = {row["surface_id"] for row in read_jsonl("surfaces.jsonl")}
        self.assertEqual(len(rows), 30)
        by_type = {}
        for row in rows:
            by_type[row["component_type"]] = by_type.get(row["component_type"], 0) + 1
            self.assertIn("manual", row["audit"])
            self.assertTrue(set(row["locator"]).issubset(surface_ids))
        self.assertEqual(
            by_type,
            {
                "retained_identity": 1,
                "retained_governance_principle": 7,
                "controlled_exclusion_block": 17,
                "rights_provenance_component": 3,
                "access_provenance_component": 2,
            },
        )
        included = [row for row in rows if row["training_use"] == "include_paraphrase"]
        self.assertEqual(len(included), 8)
        self.assertTrue(all(row["source_id"] == "HSE-TW-FAQ-2026" for row in included))
        tna = next(row for row in rows if row["component_id"] == "ogl-provenance-body-training-exclusion")
        self.assertEqual(tna["training_use"], "exclude_entire_body")
        self.assertIn("ai-train=no", tna["excluded"])

    def test_corpus_is_dense_claim_cited_and_explicitly_non_bondage(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        paragraphs = [
            paragraph
            for paragraph in re.split(r"\n\s*\n", corpus)
            if paragraph and not paragraph.startswith("#")
        ]
        self.assertEqual(len(corpus.split()), 1212)
        self.assertEqual(len(paragraphs), 25)
        self.assertEqual(len(re.findall(r"\[[^\]]+\]", corpus)), 25)
        allowed_labels = (
            "Non-bondage UK construction-governance evidence:",
            "Non-bondage UK construction-governance evidence boundary:",
        )
        for paragraph in paragraphs:
            self.assertTrue(paragraph.startswith(allowed_labels), paragraph[:120])
            citations = re.findall(r"\[[^\]]+\]", paragraph)
            self.assertEqual(len(citations), 1, paragraph[:120])
            self.assertRegex(citations[0], r"HSE-(?:TW-FAQ-2026|COPYRIGHT-OGL)")
        self.assertNotIn("TNA-OGL-V3", corpus)
        self.assertNotIn("TNA-ROBOTS-2026", corpus)
        self.assertNotIn("ai-train", corpus.lower())

    def test_required_governance_principles_are_present(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8").lower()
        required = [
            "11 march 2026",
            "short duration does not make temporary works less important",
            "responsibility spans the lifecycle and its interfaces",
            "generic or standard solution remains conditional on its underlying assumptions",
            "actual site and surrounding conditions fit those assumptions",
            "proprietary or supplier-provided systems do not remove the need for competent assessment of relevant loads and conditions",
            "actions from the side are not considered",
            "atypical actions and nearby work are context questions",
            "relevant training and experience",
            "design and coordination are distinct roles",
            "coordination is broader than producing a design",
            "not a standard, calculation method, complete legal statement, or universal instruction",
        ]
        for phrase in required:
            self.assertIn(phrase, corpus)

    def test_no_standard_numeric_product_recipe_or_linked_body_leakage(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        without_citations = re.sub(r"\[[^\]]+\]", "", corpus)
        self.assertEqual(set(re.findall(r"\b\d+(?:\.\d+)?\b", without_citations)), {"11", "2026"})
        self.assertNotIn("http://", corpus)
        self.assertNotIn("https://", corpus)
        self.assertNotIn("![", corpus)
        self.assertNotRegex(corpus, r"(?m)^>")
        self.assertNotRegex(corpus, r"(?m)^\s*\|.*\|\s*$")

        banned = [
            r"\bBS\s*5975\b",
            r"\bBritish Standard\b",
            r"\blegal requirement\b",
            r"\bparty in control\b",
            r"\bunacceptable risk\b",
            r"\bAcrows?\b",
            r"\bscaffolds?\b",
            r"\bprops?\b",
            r"\bshoring\b",
            r"\bfalsework\b",
            r"\bformwork\b",
            r"\btrench(?:es)?\b",
            r"\bcranes?\b",
            r"\bpiling\b",
            r"\bhardstanding\b",
            r"\btower\b",
            r"\bwall\b",
            r"\bfloor\b",
            r"\broof\b",
            r"\bbeams?\b",
            r"\bopenings?\b",
            r"\b\d+\s*(?:m|mm|kg|kn)\b",
        ]
        for pattern in banned:
            self.assertNotRegex(corpus, re.compile(pattern, re.IGNORECASE))

    def test_rights_attribution_and_tna_training_exclusion_are_explicit(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
        attribution = (
            "Contains public sector information published by the Health and Safety "
            "Executive and licensed under the Open Government Licence."
        )
        self.assertIn(attribution, readme)
        self.assertIn(attribution, report)
        self.assertIn("National Archives robots surface records `ai-train=no`", readme)
        self.assertIn("No wording or substantive content from that licence page is included in `CORPUS.md`", report)

    def test_no_human_support_design_selection_or_certification_inference(self):
        corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        self.assertIn(
            "Nothing here designs, approves, selects, sizes, installs, sets up, tests, proof-loads, certifies, or evaluates",
            corpus,
        )
        self.assertIn("supplies no transferable human-support engineering decision", corpus)
        paragraphs = [p for p in re.split(r"\n\s*\n", corpus) if p and not p.startswith("#")]
        for paragraph in paragraphs:
            if re.search(r"\b(?:ceiling|hardpoints?|uplines?|body-support|human-suspension)\b", paragraph, re.I):
                self.assertTrue(paragraph.startswith("Non-bondage UK construction-governance evidence boundary:"))
                self.assertRegex(paragraph, r"\b(?:Nothing here|no transferable|None of those)\b")

    def test_report_records_complete_manual_audit_and_exclusions(self):
        report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
        for phrase in [
            "all eight accordion questions and their paragraphs",
            "Both robots bodies were reviewed line by line",
            "No site crawl, mirror, archive, search snippet, or linked technical document was used",
            "Every numerical example, value, dimension, calculation",
            "Every proprietary name, product example, equipment type",
            "All logos, SVG paths, icons, images, illustrations, multimedia",
            "not a design, approval, selection, sizing, installation, setup, testing, proof-load, or certification source",
        ]:
            self.assertIn(phrase, report)


if __name__ == "__main__":
    unittest.main()
