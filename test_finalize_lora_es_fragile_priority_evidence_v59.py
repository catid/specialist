#!/usr/bin/env python3
"""Regression tests for the aggregate-only V59 evidence finalizer."""

from __future__ import annotations

import unittest

import finalize_lora_es_fragile_priority_evidence_v59 as subject


class V59EvidenceFinalizerTests(unittest.TestCase):
    def test_completed_artifacts_prove_the_train_pass_without_protected_access(self):
        value = subject.build_evidence_v59()
        self.assertEqual(
            value["status"],
            "complete_scientific_train_pass_candidate_saved_wrapper_telemetry_false_negative",
        )
        self.assertEqual(value["scientific_result"]["selected_target_norm_ratio"], 0.25)
        self.assertEqual(
            set(value["scientific_result"]["ratio_0p25_gate_checks"].values()),
            {True},
        )
        self.assertTrue(value["candidate_snapshot"]["readback_verified"])
        self.assertFalse(value["optimizer_master_committed"])
        self.assertFalse(value["protected_semantics_opened"])
        self.assertFalse(value["ood_shadow_opened"])
        self.assertFalse(value["terminal_holdout_opened"])
        self.assertEqual(
            value["content_sha256_before_self_field"],
            subject.design52.canonical_sha256_v52({
                key: item for key, item in value.items()
                if key != "content_sha256_before_self_field"
            }),
        )

    def test_no_work_projection_barrier_requires_residency_not_positive_util(self):
        rows = []
        for gpu in range(4):
            rows.append({
                "phase": "barrier", "gpu": gpu, "expected_pid": 100 + gpu,
                "compute_pids": [100 + gpu], "utilization_percent": 0,
                "memory_used_mib": 123,
            })
        summary = subject._phase_summary(rows, "barrier", require_positive=False)
        self.assertEqual(set(summary["by_gpu"]), {"0", "1", "2", "3"})
        with self.assertRaisesRegex(RuntimeError, "GPU activity contract failed"):
            subject._phase_summary(rows, "barrier", require_positive=True)

    def test_compute_phase_requires_every_gpu_and_expected_pid_residency(self):
        rows = []
        for gpu in range(4):
            rows.append({
                "phase": "compute", "gpu": gpu, "expected_pid": 200 + gpu,
                "compute_pids": [200 + gpu], "utilization_percent": 100,
                "memory_used_mib": 456,
            })
        rows[-1]["compute_pids"] = []
        with self.assertRaisesRegex(RuntimeError, "GPU activity contract failed"):
            subject._phase_summary(rows, "compute", require_positive=True)

    def test_evidence_contains_only_aggregate_metric_names_and_values(self):
        value = subject.build_evidence_v59()
        result = value["scientific_result"]
        self.assertEqual(result["evaluated_ratio_prefix"], [0.5, 0.375, 0.25])
        self.assertEqual(result["per_ratio_failed_checks"]["0.25"], [])
        self.assertGreater(
            result["ratio_0p25_paired_actor_median_deltas"]["domain"], 0,
        )
        self.assertGreater(
            result["ratio_0p25_paired_actor_median_deltas"]["qa_generation_f1"], 0,
        )


if __name__ == "__main__":
    unittest.main()
