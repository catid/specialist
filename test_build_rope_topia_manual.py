import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import build_rope_topia_manual
from build_leakfree_qa import eval_facts
from qa_quality import leakage_reason, parse_qa


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "sources" / "rope_topia_manual_v1.json"
EVAL = [ROOT / "data" / "eval_qa.jsonl",
        ROOT / "data" / "eval_qa_v2.jsonl"]
BASELINES = [ROOT / "data" / "train_qa_curated_v1.jsonl"]
TRACKED_OUTPUT = ROOT / "data" / "rope_topia_manual_v1.jsonl"
TRACKED_REPORT = ROOT / "data" / "rope_topia_manual_v1.report.json"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RopeTopiaManualTests(unittest.TestCase):
    def build_temp(self, directory):
        directory = Path(directory)
        output = directory / "rope_topia.jsonl"
        report_path = directory / "report.json"
        report = build_rope_topia_manual.build(
            MANIFEST, output, report_path, EVAL, BASELINES)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        return report, rows, output, report_path

    def write_manifest(self, directory, manifest):
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_real_manifest_builds_every_resource_as_metadata_only(self):
        with tempfile.TemporaryDirectory() as directory:
            report, rows, _, _ = self.build_temp(directory)
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(report["counts"]["resources"], 15)
        self.assertEqual(report["counts"]["output"], 15)
        self.assertEqual(report["counts"]["gated_resources"], 15)
        self.assertEqual(report["counts"]["excluded_urls"], 8)
        self.assertEqual(report["counts"]["accessible_demo_pages"], 4)
        self.assertEqual(report["counts"]["accessible_demo_endpoints"], 8)
        self.assertEqual(report["counts"]["unavailable_public_endpoints"], 5)
        self.assertEqual(
            report["counts"]["gated_relevant_endpoints_including_blog"], 24)
        self.assertEqual(report["counts"]["article_content_qa"], 0)
        self.assertEqual(
            {row["canonical_url"] for row in rows},
            {item["canonical_url"] for item in manifest["resources"]},
        )
        self.assertEqual(
            {row["supplied_url"] for row in rows},
            {item["supplied_url"] for item in manifest["resources"]},
        )
        resources = {item["id"]: item for item in manifest["resources"]}
        for row in rows:
            source = resources[row["resource_id"]]
            self.assertFalse(row["article_content_available"])
            self.assertEqual(row["access_status"], "demo_gate")
            self.assertEqual(
                row["content_use"], "resource_metadata_only_due_demo_gate")
            self.assertEqual(row["answer"], row["canonical_url"])
            self.assertEqual(row["url"], row["canonical_url"])
            self.assertEqual(row["evidence"], row["url_evidence"])
            self.assertEqual(row["evidence_url"],
                             row["url_evidence_url"])
            self.assertEqual(row["title_evidence"],
                             source["title_evidence"])
            self.assertEqual(row["title_evidence_url"],
                             source["title_evidence_url"])
            self.assertEqual(row["document_sha256"], file_sha256(MANIFEST))
            self.assertIn(row["canonical_url"], row["evidence"])
            self.assertEqual(parse_qa(row["text"]),
                             (row["question"], row["answer"]))
            self.assertLessEqual(len(row["evidence"].split()), 16)

    def test_tracked_output_matches_deterministic_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, output, _ = self.build_temp(directory)
            self.assertEqual(output.read_bytes(), TRACKED_OUTPUT.read_bytes())

    def test_tracked_report_provenance_hashes_match_artifacts(self):
        report = json.loads(TRACKED_REPORT.read_text())
        self.assertEqual(report["manifest_sha256"], file_sha256(MANIFEST))
        self.assertEqual(report["output_sha256"],
                         file_sha256(TRACKED_OUTPUT))
        for path, expected in report["eval_inputs"].items():
            self.assertEqual(expected, file_sha256(ROOT / path))
        for path, expected in report["baseline_inputs"].items():
            self.assertEqual(expected, file_sha256(ROOT / path))

    def test_report_is_deterministic_for_same_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, _, _, report_path = self.build_temp(directory)
            first = report_path.read_bytes()
            self.build_temp(directory)
            self.assertEqual(first, report_path.read_bytes())

    def test_no_row_collides_with_evaluation_facts(self):
        facts = eval_facts(EVAL)
        rows = [json.loads(line)
                for line in TRACKED_OUTPUT.read_text().splitlines()]
        for row in rows:
            self.assertIsNone(
                leakage_reason(row["question"], row["answer"], facts),
                row["resource_id"],
            )

    def test_sitemap_portfolio_url_is_canonical_and_stripped_path_is_404(self):
        rows = [json.loads(line)
                for line in TRACKED_OUTPUT.read_text().splitlines()]
        row = next(item for item in rows
                   if item["resource_id"] ==
                   "kinbaku_today_rope_not_about_rope")
        self.assertEqual(
            row["canonical_url"],
            "https://rope-topia.com/portfolio-items/"
            "kinbaku-today-rope-is-not-about-rope/"
            "portfolioCats-102-70-123-72-57/",
        )
        self.assertEqual(row["supplied_url"], row["canonical_url"])
        self.assertNotIn("canonicalization_reason", row)
        manifest = json.loads(MANIFEST.read_text())
        unavailable = {
            item["url"]: item["http_status"]
            for item in manifest["live_access"]["unavailable_public_endpoints"]
        }
        self.assertEqual(
            unavailable[
                "https://rope-topia.com/portfolio-items/"
                "kinbaku-today-rope-is-not-about-rope/"
            ],
            404,
        )

    def test_non_gated_content_resource_is_rejected(self):
        manifest = json.loads(MANIFEST.read_text())
        manifest["resources"][0]["access_status"] = "content_available"
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(directory, manifest)
            with self.assertRaisesRegex(
                    ValueError, "must be marked demo_gate"):
                build_rope_topia_manual.load_manifest(path)

    def test_url_without_source_evidence_is_rejected(self):
        manifest = json.loads(MANIFEST.read_text())
        manifest["resources"][0]["url_evidence"] = "<loc>missing</loc>"
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(directory, manifest)
            with self.assertRaisesRegex(ValueError,
                                        "must contain exactly"):
                build_rope_topia_manual.load_manifest(path)

    def test_duplicate_question_is_rejected(self):
        manifest = json.loads(MANIFEST.read_text())
        manifest["resources"][1]["question"] = copy.deepcopy(
            manifest["resources"][0]["question"])
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(directory, manifest)
            with self.assertRaisesRegex(ValueError, "duplicate question"):
                build_rope_topia_manual.load_manifest(path)


if __name__ == "__main__":
    unittest.main()
