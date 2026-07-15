#!/usr/bin/env python3
import hashlib
import unittest

import build_train_shadow_folds_v37a as folds
import run_sft_train_only_control_v36a as runtime


class TrainShadowFoldsV37ATest(unittest.TestCase):
    def test_complete_conflict_disjoint_crossfit(self):
        existing = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folds.OUTPUT_DIR.iterdir() if path.is_file()
        }
        value = folds.build()
        self.assertEqual(existing, {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folds.OUTPUT_DIR.iterdir() if path.is_file()
        })
        self.assertEqual(len(value["folds"]), 5)
        self.assertEqual(value["policy"]["conflict_units"], 259)
        self.assertFalse(
            value["policy"]["external_evaluation_ood_holdout_or_benchmark_opened"]
        )
        self.assertEqual(value["coverage"]["all_fold_shadow_dev_rows"], 531)
        self.assertEqual(value["coverage"]["all_fold_train_rows"], 531 * 4)
        self.assertEqual(
            [
                (item["train"]["rows"], item["shadow_dev"]["rows"])
                for item in value["folds"]
            ],
            [(372, 159), (408, 123), (452, 79), (448, 83), (444, 87)],
        )
        self.assertTrue(all(
            item["train_dev_conflict_unit_intersection"] == 0
            and not any(item["train_dev_edge_identity_intersections"].values())
            for item in value["folds"]
        ))
        self.assertEqual(
            [item["shadow_dev"]["conflict_units"] for item in value["folds"]],
            [54, 52, 52, 51, 50],
        )
        eligible = [
            item["fold"] for item in value["folds"]
            if item["train"]["rows"] % 28 == 0
            and item["shadow_dev"]["conflict_units"] >= 50
            and not any(item["train_dev_edge_identity_intersections"].values())
        ]
        self.assertEqual(eligible, [3])
        self.assertEqual(value["selection_firewall"]["confirmatory_fold"], 3)
        commitments = value["content_free_unit_commitments"]
        self.assertEqual(len(commitments), 259)
        self.assertEqual(
            len({row for unit in commitments for row in unit["row_sha256"]}), 531,
        )
        self.assertEqual(
            value["content_sha256_before_self_field"],
            folds.EXPECTED_MANIFEST_CONTENT_SHA256,
        )
        self.assertEqual(
            value["content_sha256_before_self_field"],
            runtime.canonical_sha256({
                key: item for key, item in value.items()
                if key != "content_sha256_before_self_field"
            }),
        )


if __name__ == "__main__":
    unittest.main()
