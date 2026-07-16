import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def read_json(relative_path: str):
    return json.loads(read_text(relative_path))


def read_jsonl(relative_path: str):
    return [json.loads(line) for line in read_text(relative_path).splitlines() if line]


def sha256(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


class BradyKedgeAnchorCorpusTests(unittest.TestCase):
    maxDiff = None

    def test_exact_package_file_set_and_no_source_body(self):
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
            "CORPUS.md": "580e3af2534aad5b0040f58c833bf7c443c1359d4d739e8527a0fa2d03e9834a",
            "README.md": "42cf19fdb0cf2e80b0584f30455321d7c2adcb6fb1f3ab37064647759b81ec76",
            "REPORT.md": "47d12c24e25845b47d14b51218db988dd924aa00f3e3ced6a0354249b692e13d",
            "components.jsonl": "731b57c25c4a113de45b7d9ff2cf028db7aaf577fa28fd6035a0181c44638ad4",
            "dispositions.jsonl": "d3e060066d1b48504a7ab01dc93c8d12038cde45e248847279fcbfa235afb1c7",
            "source_snapshot/provenance.json": "6b303aa8f871ff8d4f6f69233f3c8bfb0b867b3304130feff5eb30d32c1c43e2",
            "sources.jsonl": "07a1e7588c5fc692743555a838dfa58573ae847ec2934a4ab1be5d9e0cca58d4",
            "surfaces.jsonl": "6e2716b8bd443b46d47e92813a61e347362be2c808a95b31275fd0694734217f",
        }
        self.assertEqual({path: sha256(path) for path in expected}, expected)

    def test_exact_source_checksums_and_single_item_retrieval(self):
        rows = read_jsonl("sources.jsonl")
        self.assertEqual(len(rows), 2)
        by_id = {row["source_id"]: row for row in rows}
        self.assertEqual(set(by_id), {"BRADY-77729-TEXT", "PGLAF-ROBOTS-404"})
        text_source = by_id["BRADY-77729-TEXT"]
        self.assertEqual(text_source["ebook_id"], 77729)
        self.assertEqual(text_source["edition"], "sixth")
        self.assertEqual(text_source["print_year"], 1852)
        self.assertEqual(text_source["body_bytes"], 1479083)
        self.assertEqual(text_source["line_count"], 26071)
        self.assertEqual(text_source["whitespace_word_count"], 239369)
        self.assertEqual(
            text_source["body_sha256"],
            "7a1c173c1f3c73201ab21f741130094a57765a923ff26d503c4c0f4c93a58cd2",
        )
        self.assertEqual(
            text_source["body_sha512"],
            "bd81079fc2faaf79c319df659f97ab4fe52598c5cbd18c01b1491d572d7f1dc96aa61e920d6dffafd9d1689f01b57633e1a6ad0e97fe51e1f87acdd4934f0c1a",
        )
        self.assertTrue(text_source["manual_full_text_audit"])
        self.assertFalse(text_source["source_body_redistributed"])
        robots = by_id["PGLAF-ROBOTS-404"]
        self.assertEqual(robots["http_status"], 404)
        self.assertEqual(robots["body_bytes"], 162)
        self.assertEqual(
            robots["body_sha256"],
            "766c1d6bcb81d3e983fb7adbc19c616d7fc01dafb7893738edc242e2adc59c07",
        )
        self.assertIn("not treated as affirmative permission", robots["interpretation"])

    def test_pre_acquisition_policy_mirror_and_rights_audit(self):
        provenance = read_json("source_snapshot/provenance.json")
        audit = provenance["pre_acquisition_audit"]
        self.assertTrue(audit["completed_before_body_get"])
        self.assertTrue(audit["main_site_human_only_terms_observed"])
        self.assertFalse(audit["main_site_ebook_body_retrieved_or_automated"])
        self.assertEqual(audit["main_robots_general_agent_disallow"], "/ebooks/search")
        self.assertIn("gutenberg.pglaf.org", audit["mirror_registry_finding"])
        self.assertEqual(audit["mirror_robots"]["http_status"], 404)
        self.assertIn("not treated as affirmative permission", audit["mirror_robots"]["interpretation"])
        selected = [row for row in audit["candidate_header_checks"] if row["selected"]]
        self.assertEqual(len(selected), 1)
        self.assertIn("/7/7/7/2/77729/77729-0.txt", selected[0]["url"])
        retrieval = provenance["retrieval"]
        self.assertEqual(retrieval["known_item_body_gets"], 1)
        self.assertEqual(retrieval["main_site_body_gets"], 0)
        self.assertFalse(retrieval["site_crawl_performed"])
        self.assertFalse(retrieval["source_body_redistributed"])

    def test_complete_manual_line_coverage_and_second_half_result(self):
        provenance = read_json("source_snapshot/provenance.json")
        review = provenance["manual_full_text_review"]
        self.assertTrue(review["complete"])
        self.assertEqual(
            review["review_ranges"],
            [
                {"start_line": 1, "end_line": 13000, "complete": True},
                {
                    "start_line": 13001,
                    "end_line": 26071,
                    "complete": True,
                    "eligible_exact_new_findings": 0,
                },
            ],
        )
        self.assertTrue(review["ranges_contiguous"])
        self.assertTrue(review["ranges_non_overlapping"])
        self.assertTrue(review["all_source_lines_covered"])
        self.assertFalse(review["existing_corpus_bodies_inspected"])
        self.assertEqual(review["illustration_markers_count"], 101)
        self.assertTrue(review["all_illustrations_excluded"])
        self.assertTrue(review["part_xi_read_through_end"])

    def test_dispositions_are_contiguous_complete_and_bounded(self):
        rows = read_jsonl("dispositions.jsonl")
        self.assertEqual(len(rows), 14)
        self.assertEqual(rows[0]["start_line"], 1)
        self.assertEqual(rows[-1]["end_line"], 26071)
        for previous, current in zip(rows, rows[1:]):
            self.assertEqual(previous["end_line"] + 1, current["start_line"])
        self.assertTrue(all(row["source_id"] == "BRADY-77729-TEXT" for row in rows))
        self.assertTrue(all(row["audit"].startswith("manual sequential review") for row in rows))
        partial = [row for row in rows if row["decision"].startswith("partial")]
        self.assertEqual([(row["start_line"], row["end_line"]) for row in partial], [(1, 831), (832, 1940), (11660, 12550), (25915, 26070)])
        part_xi = next(row for row in rows if row["start_line"] == 12551)
        self.assertEqual(part_xi["decision"], "exclude")
        self.assertIn("every table", part_xi["audit"])

    def test_novelty_audit_never_opened_existing_corpus_bodies(self):
        novelty = read_json("source_snapshot/provenance.json")["novelty_audit"]
        self.assertEqual(novelty["searched_existing_file_classes"], ["manifest.json", "REPORT.md", "README.md"])
        self.assertFalse(novelty["existing_corpus_bodies_searched_or_opened"])
        self.assertEqual(len(novelty["candidate_phrases_with_zero_matches"]), 17)
        self.assertFalse(novelty["ordinary_bowline_and_figure_eight_directions_retained"])

    def test_web_surfaces_have_manual_decisions_and_no_auto_summary(self):
        rows = read_jsonl("surfaces.jsonl")
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(row["reviewed_at"] == "2026-07-16" for row in rows))
        main = [row for row in rows if row["url"].startswith("https://www.gutenberg.org/")]
        self.assertEqual(len(main), 7)
        self.assertTrue(all(not row["body_automated"] for row in main))
        item = next(row for row in rows if row["source_id"] == "GUTENBERG-ITEM-77729")
        self.assertIn("automatically generated summary", item["excluded"])
        mirror_robots = next(row for row in rows if row["source_id"] == "PGLAF-ROBOTS-404")
        self.assertIn("not treated as permission", mirror_robots["interpretation"])

    def test_components_lock_all_controlled_exclusions(self):
        rows = read_jsonl("components.jsonl")
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["decision"] == "exclude" for row in rows))
        illustrations = next(row for row in rows if row["component_id"] == "illustration-markers")
        self.assertEqual(illustrations["count_in_transcription"], 101)
        self.assertEqual(illustrations["retained"], "none as visual evidence")
        joined = " ".join(row["excluded"] for row in rows).lower()
        for phrase in (
            "executable sequences",
            "chemical",
            "breaking strain",
            "naval",
            "body lowering",
            "human suspension",
            "automatic item summary",
            "current suitability",
            "raw source text",
        ):
            self.assertIn(phrase, joined)

    def test_corpus_is_dense_cited_and_source_bounded(self):
        corpus = read_text("CORPUS.md")
        self.assertEqual(len(corpus.encode("utf-8")), 16174)
        self.assertEqual(len(corpus.split()), 2248)
        paragraphs = [part for part in corpus.split("\n\n") if not part.startswith("#")]
        self.assertEqual(len(paragraphs), 38)
        self.assertEqual(sum(part.startswith("Historical ropework evidence:") for part in paragraphs), 32)
        self.assertEqual(sum(part.startswith("Historical ropework boundary:") for part in paragraphs), 6)
        self.assertTrue(all(re.search(r"\[[^\[\]]+\]$", part) for part in paragraphs))
        self.assertNotRegex(corpus, r"https?://|www\.")
        self.assertNotRegex(corpus, r"!\[[^\]]*\]\([^)]*\)")
        self.assertNotRegex(corpus, r"(?m)^\s*\|.*\|\s*$")

    def test_required_exact_new_vocabulary_and_decomposition_are_present(self):
        corpus = read_text("CORPUS.md").lower()
        for phrase in (
            "component-to-family progression",
            "spanish fox",
            "knittle",
            "sennit",
            "sheet bend",
            "becket bend",
            "short splice",
            "long splice",
            "flemish eye",
            "artificial eye",
            "grommet",
            "worming",
            "parceling",
            "serving",
            "throat seizing",
            "snaking",
            "walling",
            "crowning",
            "sea gasket",
            "harbor gasket",
            "panch or wrought mat",
            "sword mat",
            "shell",
            "sheave",
            "pin",
            "fiddle block",
            "shoe block",
            "sister block",
            "dead-eye",
            "bull’s-eye",
            "shoulder block",
            "tail block",
            "snatch block",
        ):
            self.assertIn(phrase, corpus)

    def test_no_executable_numeric_image_or_operational_leakage(self):
        corpus = read_text("CORPUS.md")
        lower = corpus.lower()
        for forbidden in (
            "take the end of",
            "pass the end through",
            "haul it taut",
            "stick the end under",
            "plate no.",
            "figure 1",
            "table 492",
            "gun-tackle purchase",
            "top burton",
            "blackening the rigging",
            "lamp-black",
            "litharge",
            "breaking strain in tons",
            "man-rope knot",
        ):
            self.assertNotIn(forbidden, lower)
        prose_without_citations = re.sub(r"\[[^\[\]]+\]", "", corpus)
        self.assertNotRegex(
            prose_without_citations,
            r"(?i)\b\d+(?:\.\d+)?\s*(?:%|mm|cm|inches?|feet|fathoms?|pounds?|tons?|kn|newtons?)\b",
        )

    def test_rights_fidelity_and_absolute_nontransfer_are_explicit(self):
        corpus = read_text("CORPUS.md").lower()
        report = read_text("REPORT.md").lower()
        readme = read_text("README.md").lower()
        for phrase in (
            "public domain in the united states",
            "registered trademark",
            "outside the united states",
            "does not imply project gutenberg endorsement",
            "redistributes no source body or image",
        ):
            self.assertIn(phrase, corpus)
        for phrase in (
            "does not claim diplomatic print fidelity",
            "body lowering, restraint",
            "human support",
            "human suspension",
            "cannot be used to choose, teach, validate, rate, load, install, certify, or approve",
            "no source body, source image, or automated summary is redistributed",
        ):
            self.assertIn(phrase, report)
        self.assertIn("the main `gutenberg.org` ebook body was not downloaded or automated", readme)

    def test_manifest_statistics_and_artifact_hashes(self):
        manifest = read_json("manifest.json")
        self.assertTrue(manifest["direct_training_ready"])
        self.assertEqual(manifest["rights_status"], "public-domain-in-usa-source; independent rewritten derivative")
        self.assertEqual(manifest["statistics"]["source_lines_manually_reviewed"], 26071)
        self.assertEqual(manifest["statistics"]["illustration_markers_excluded"], 101)
        self.assertEqual(manifest["statistics"]["corpus_bytes"], 16174)
        self.assertEqual(manifest["statistics"]["corpus_words"], 2248)
        self.assertEqual(manifest["statistics"]["corpus_claim_paragraphs"], 38)
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
