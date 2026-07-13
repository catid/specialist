import json
import tempfile
import unittest
from pathlib import Path

from build_resource_coverage_report import build, file_sha256


class ResourceCoverageReportTests(unittest.TestCase):
    def fixture(self, root):
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema": "rope-resource-manifest-v1",
            "resources": [{"id": "one"}, {"id": "two"}],
        }))
        coverage_dir = root / "coverage"
        coverage_dir.mkdir()
        digest = file_sha256(manifest)
        for resource_id, status, outcome in (
                ("one", "complete", "fetched"),
                ("two", "blocked", "blocked")):
            results = {key: [] for key in
                       ("fetched", "skipped", "blocked", "failed")}
            results[outcome].append({"url": "https://example.test/",
                                     "reason": "fixture"})
            counts = {key: len(value) for key, value in results.items()}
            (coverage_dir / f"{resource_id}.coverage.json").write_text(
                json.dumps({
                    "schema": "resource-corpus-coverage-v1",
                    "resource_id": resource_id,
                    "manifest_sha256": digest,
                    "status": status,
                    "counts": counts,
                    "results": results,
                    "robots": {"status": "ok", "ai_train_no": False},
                    "completed_at": "2026-07-13T00:00:00+00:00",
                }))
        return manifest, coverage_dir

    def test_valid_coverage_builds_compact_reason_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, coverage_dir = self.fixture(root)
            report = build(manifest, coverage_dir, root / "report.json")
            self.assertEqual(report["counts"]["resources"], 2)
            self.assertEqual(report["counts"]["statuses"],
                             {"blocked": 1, "complete": 1})
            self.assertEqual(
                report["resources"]["two"]["reason_counts"]["blocked"],
                {"fixture": 1},
            )

    def test_stale_manifest_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, coverage_dir = self.fixture(root)
            manifest.write_text(manifest.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "stale manifest digest"):
                build(manifest, coverage_dir, root / "report.json")

    def test_missing_resource_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, coverage_dir = self.fixture(root)
            (coverage_dir / "two.coverage.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing=.*two"):
                build(manifest, coverage_dir, root / "report.json")


if __name__ == "__main__":
    unittest.main()
